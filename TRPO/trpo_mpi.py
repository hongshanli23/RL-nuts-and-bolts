from baselines.common import explained_variance, zipsame, dataset
from baselines import logger
import baselines.common.tf_util as U
import tensorflow as tf, numpy as np
import time
import os.path as osp
from baselines.common import colorize
from collections import deque
from baselines.common import set_global_seeds
from baselines.common.models import get_network_builder
from baselines.common.mpi_adam import MpiAdam
from baselines.common.cg import cg
from baselines.common.policies import PolicyWithValue
from baselines.common.vec_env.vec_env import VecEnv
from contextlib import contextmanager

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

def traj_segment_generator(pi, env, horizon, max_loss=-100, reward_cb=None):
    """
    TODO: set reward range 
    for example in the pendulum problem, reward is from 
    [-16.27, 0], I want to say if the reward is below -10, then 
    it is a loss, if a reward is above -10, then it is a gain.
    Simply renorm it to [-8, 8]
    Is way I can make it into a finite episode task. 
    I can do it by applying a reward callback

    max_loss: max number of points one can loss in an episode
        used for infinite environment
    """
    # Initialize state variables
    t = 0
    ac = env.action_space.sample()              # an action
    new = True
    rew = 0.0
    ob = env.reset()
    if not isinstance(env, VecEnv):
        ob = np.expand_dims(ob, axis=0)

    cur_ep_ret = 0
    cur_ep_len = 0
    ep_rets = []
    ep_lens = []

    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon)]) # curr state
    rews = np.zeros(horizon, 'float32')          # reward
    vpreds = np.zeros(horizon, 'float32')        # predicted value
    news = np.zeros(horizon, 'int32')            # new episode ? Done
    acs = np.array([ac for _ in range(horizon)]) # actions 
    prevacs = acs.copy()                         # previous action

    while True:
        prevac = ac
        ob = tf.constant(ob)
        ac, vpred, _, _ = pi.step(ob)            # take an action on curr state
                                                 # and predict the state value
        ac = ac.numpy()
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:
            yield {"ob" : obs, "rew" : rews, "vpred" : vpreds, "new" : news,
                    "ac" : acs, "prevac" : prevacs, "nextvpred": vpred * (1 - new),
                    "ep_rets" : ep_rets, "ep_lens" : ep_lens}
            
            _, vpred, _, _ = pi.step(ob)
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_lens = []
        i = t % horizon
        obs[i] = ob
        vpreds[i] = vpred.numpy()
        news[i] = new
        acs[i] = ac
        prevacs[i] = prevac

        ob, rew, new, _ = env.step(ac)
        if reward_cb:
            rew = reward_cb(rew)
        if not isinstance(env, VecEnv):
          ob = np.expand_dims(ob, axis=0)
        rews[i] = rew

        cur_ep_ret += rew # episodic return 
        if cur_ep_ret < max_loss:
            new = True

        cur_ep_len += 1   # episodic length # number of steps survived so far
        if new: 
            new = True
            ep_rets.append(cur_ep_ret) # add previous episodic return to ep_rets
            ep_lens.append(cur_ep_len)
            cur_ep_ret = 0             # reset total episodic reward
            cur_ep_len = 0
            ob = env.reset()
            if not isinstance(env, VecEnv):
              ob = np.expand_dims(ob, axis=0)
        t += 1     

def add_vtarg_and_adv(seg, gamma, lam):
    new = np.append(seg["new"], 0) # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    T = len(seg["rew"])
    seg["adv"] = gaelam = np.empty(T, 'float32')
    rew = seg["rew"]
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1-new[t+1]
        delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]

def learn(*,
        network,
        env,
        total_timesteps,
        timesteps_per_batch=1024, # what to train on
        max_kl=0.001,
        cg_iters=10,
        gamma=0.99,
        lam=1.0, # advantage estimation
        seed=None,
        ent_coef=0.0,
        cg_damping=1e-2,
        cg_density=5,
        vf_stepsize=3e-4,
        vf_iters =3,
        max_episodes=0, max_iters=0,  # time constraint
        callback=None,
        load_path=None,
        **network_kwargs
        ):
    '''
    learn a policy function with TRPO algorithm

    Parameters:
    ----------

    network                 neural network to learn. Can be either string ('mlp', 'cnn', 'lstm', 'lnlstm' for basic types)
                            or function that takes input placeholder and returns tuple (output, None) for feedforward nets
                            or (output, (state_placeholder, state_output, mask_placeholder)) for recurrent nets

    env                     environment (one of the gym environments or wrapped via baselines.common.vec_env.VecEnv-type class

    timesteps_per_batch     timesteps per gradient estimation batch

    max_kl                  max KL divergence between old policy and new policy ( KL(pi_old || pi) )

    ent_coef                coefficient of policy entropy term in the optimization objective

    cg_iters                number of iterations of conjugate gradient algorithm

    cg_damping              conjugate gradient damping

    vf_stepsize             learning rate for adam optimizer used to optimie value function loss

    vf_iters                number of iterations of value function optimization iterations per each policy optimization step

    total_timesteps           max number of timesteps

    max_episodes            max number of episodes

    max_iters               maximum number of policy optimization iterations

    callback                function to be called with (locals(), globals()) each policy optimization step

    load_path               str, path to load the model from (default: None, i.e. no model is loaded)

    **network_kwargs        keyword arguments to the policy / network builder. See baselines.common/policies.py/build_policy and arguments to a particular type of network

    Returns:
    -------

    learnt model
    '''
    
    print("local arguments")
    for k, v in locals().items():
        print(k, v)

    if MPI is not None:
        nworkers = MPI.COMM_WORLD.Get_size()
        rank = MPI.COMM_WORLD.Get_rank()
    else:
        nworkers = 1
        rank = 0

    set_global_seeds(seed)

    np.set_printoptions(precision=3)
    # Setup losses and stuff
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space

    if isinstance(network, str):
        network = get_network_builder(network)(**network_kwargs)

    with tf.name_scope("pi"):
        pi_policy_network = network(ob_space.shape)
        pi_value_network = network(ob_space.shape)
        pi = PolicyWithValue(ac_space, pi_policy_network, pi_value_network)
    with tf.name_scope("oldpi"):
        old_pi_policy_network = network(ob_space.shape)
        old_pi_value_network = network(ob_space.shape)
        oldpi = PolicyWithValue(ac_space, old_pi_policy_network, old_pi_value_network)

    pi_var_list = pi_policy_network.trainable_variables + list(pi.pdtype.trainable_variables)
    old_pi_var_list = old_pi_policy_network.trainable_variables + list(oldpi.pdtype.trainable_variables)
    vf_var_list = pi_value_network.trainable_variables + pi.value_fc.trainable_variables
    old_vf_var_list = old_pi_value_network.trainable_variables + oldpi.value_fc.trainable_variables

    if load_path is not None:
        load_path = osp.expanduser(load_path)
        ckpt = tf.train.Checkpoint(model=pi)
        manager = tf.train.CheckpointManager(ckpt, load_path, max_to_keep=None)
        ckpt.restore(manager.latest_checkpoint)

    vfadam = MpiAdam(vf_var_list)

    get_flat = U.GetFlat(pi_var_list)
    set_from_flat = U.SetFromFlat(pi_var_list)
    loss_names = ["optimgain", "meankl", "entloss", "surrgain", "entropy"]
    shapes = [var.get_shape().as_list() for var in pi_var_list]

    
    # update behavior to target
    def assign_old_eq_new():
        for pi_var, old_pi_var in zip(pi_var_list, old_pi_var_list):
            old_pi_var.assign(pi_var)
        for vf_var, old_vf_var in zip(vf_var_list, old_vf_var_list):
            old_vf_var.assign(vf_var)

    #@tf.function
    def compute_lossandgrad(ob, ac, atarg):
        """
        ob: observations:
        ac: actions
        atarg: advantage
        """
        # tf.GradientTape builds a computation graph
        with tf.GradientTape() as tape:
            old_policy_latent = oldpi.policy_network(ob)
            old_pd, _ = oldpi.pdtype.pdfromlatent(old_policy_latent)
            policy_latent = pi.policy_network(ob)
            pd, _ = pi.pdtype.pdfromlatent(policy_latent)
            # pd = probability distribution
            
            # compute KL div with old policy dist
            kloldnew = old_pd.kl(pd)
            ent = pd.entropy()
            meankl = tf.reduce_mean(kloldnew)
            meanent = tf.reduce_mean(ent)
            entbonus = ent_coef * meanent
            
            # importance sampling ration
            ratio = tf.exp(pd.logp(ac) - old_pd.logp(ac))
            
            # surrogate gain
            # want to maximize it
            print('ratio and atarg')
            print(ratio[:5], atarg[:5])
            surrgain = tf.reduce_mean(ratio * atarg)
            #surrgain = tf.reduce_mean(ratio)
            optimgain = surrgain + entbonus
            losses = [optimgain, meankl, entbonus, surrgain, meanent]
        
        # surrogate poligy gradient
        gradients = tape.gradient(optimgain, pi_var_list)
        
        # U.flatgrad is my flatten
        # They clipped gradient
        # I think that's why theirs is more stable
        
        return losses + [U.flatgrad(gradients, pi_var_list)]

    @tf.function
    def compute_losses(ob, ac, atarg):
        # note that no gradient tape is used
        old_policy_latent = oldpi.policy_network(ob)
        old_pd, _ = oldpi.pdtype.pdfromlatent(old_policy_latent)
        policy_latent = pi.policy_network(ob)
        pd, _ = pi.pdtype.pdfromlatent(policy_latent)
        kloldnew = old_pd.kl(pd)
        ent = pd.entropy()
        meankl = tf.reduce_mean(kloldnew)
        meanent = tf.reduce_mean(ent)
        entbonus = ent_coef * meanent
        ratio = tf.exp(pd.logp(ac) - old_pd.logp(ac))
        # DEBUG
        surrgain = tf.reduce_mean(ratio * atarg)
        #surrgain = tf.reduce_mean(ratio)
        optimgain = surrgain + entbonus
        losses = [optimgain, meankl, entbonus, surrgain, meanent]
        return losses

    #ob shape should be [batch_size, ob_dim], merged nenv
    #ret shape should be [batch_size]
    @tf.function
    def compute_vflossandgrad(ob, ret):
        with tf.GradientTape() as tape:
            pi_vf = pi.value(ob)
            vferr = tf.reduce_mean(tf.square(pi_vf - ret))
        
        # interesting here, they computed gradient of critic
        # and flatten it to a vector
        # and clipped it
        return U.flatgrad(tape.gradient(vferr, vf_var_list), vf_var_list)

    @tf.function
    def compute_fvp(flat_tangent, ob, ac, atarg):
        """Fisher vector product"""
        with tf.GradientTape() as outter_tape:
            # nested gradient tape
            # this is how you do double derivative in 
            # tensorflow
            with tf.GradientTape() as inner_tape:
                # compute latent parameters of old policy
                # mean / std of gaussian
                old_policy_latent = oldpi.policy_network(ob)
                
                # get a distribution from the latent variable
                old_pd, _ = oldpi.pdtype.pdfromlatent(old_policy_latent)
                
                # latent parameters for the target policy
                policy_latent = pi.policy_network(ob)
                
                # dist of target policy
                pd, _ = pi.pdtype.pdfromlatent(policy_latent)
                
                # kl with old policy
                kloldnew = old_pd.kl(pd)
                
                print('KL shape:', kloldnew.shape)
                meankl = tf.reduce_mean(kloldnew)
                print("mean kl shape:", meankl.shape)
            
            # dkl/ params
            klgrads = inner_tape.gradient(meankl, pi_var_list)
            start = 0
            tangents = []
            for shape in shapes:
                sz = U.intprod(shape)
                tangents.append(tf.reshape(flat_tangent[start:start+sz], shape))
                start += sz
                
            # dot product between dKL/params and tangent 
            gvp = tf.add_n([tf.reduce_sum(g*tangent) for (g, tangent) in zipsame(klgrads, tangents)])
        
        # direct method compute A*tangent
        # A is the Hessian of KL
        hessians_products = outter_tape.gradient(gvp, pi_var_list)
        
        # makes the result flat
        fvp = U.flatgrad(hessians_products, pi_var_list)
        return fvp

    @contextmanager
    def timed(msg):
        if rank == 0:
            print(colorize(msg, color='magenta'))
            tstart = time.time()
            yield
            print(colorize("done in %.3f seconds"%(time.time() - tstart), color='magenta'))
        else:
            yield

    def allmean(x):
        assert isinstance(x, np.ndarray)
        if MPI is not None:
            out = np.empty_like(x)
            MPI.COMM_WORLD.Allreduce(x, out, op=MPI.SUM)
            out /= nworkers
        else:
            out = np.copy(x)

        return out
    
    # theta init (initial policy)
    th_init = get_flat()
    if MPI is not None:
        MPI.COMM_WORLD.Bcast(th_init, root=0)

    set_from_flat(th_init)
    vfadam.sync()
    print("Init param sum", th_init.sum(), flush=True)

    # Prepare for rollouts
    # ----------------------------------------
    
    # TODO the setup is just for problems like Pendulum
    # infinite episodic tasks
    # want to make it finite by considering some rewards as gain
    # some as loss

    def reward_cb(rew):
        rew += 8.0
        return rew

    seg_gen = traj_segment_generator(pi, env, 
        timesteps_per_batch, max_loss=-100, reward_cb=reward_cb)

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=10) # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=10) # rolling buffer for episode rewards

    if sum([max_iters>0, total_timesteps>0, max_episodes>0])==0:
        # noththing to be done
        return pi

    assert sum([max_iters>0, total_timesteps>0, max_episodes>0]) < 2, \
        'out of max_iters, total_timesteps, and max_episodes only one should be specified'

    while True:
        if callback: callback(locals(), globals())
        if total_timesteps and timesteps_so_far >= total_timesteps:
            break
        elif max_episodes and episodes_so_far >= max_episodes:
            break
        elif max_iters and iters_so_far >= max_iters:
            break
        logger.log("********** Iteration %i ************"%iters_so_far)

        with timed("sampling"):
            seg = seg_gen.__next__()
        
        add_vtarg_and_adv(seg, gamma, lam)

        # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
        ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
        ob = sf01(ob)
        vpredbefore = seg["vpred"] # predicted value function before udpate
        atarg = (atarg - atarg.mean()) / atarg.std() # standardized advantage function estimate
        
        print('XXXXXX atarg shape', atarg.shape)
        print('atarg. mean', atarg.mean())

        if hasattr(pi, "ret_rms"): pi.ret_rms.update(tdlamret)
        if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob) # update running mean/std for policy

        args = ob, ac, atarg
        fvpargs = [arr[::cg_density] for arr in args]
        def fisher_vector_product(p):
            return allmean(compute_fvp(p, *fvpargs).numpy()) + cg_damping * p

        assign_old_eq_new() # set old parameter values to new parameter values
        with timed("computegrad"):
            *lossbefore, g = compute_lossandgrad(*args)
        lossbefore = allmean(np.array(lossbefore))
        g = g.numpy()
        g = allmean(g)
        if np.allclose(g, 0):
            logger.log("Got zero gradient. not updating")
        else:
            with timed("cg"):
                print('### sum of g', g.sum())
                stepdir = cg(fisher_vector_product, g, cg_iters=cg_iters, verbose=True)
            assert np.isfinite(stepdir).all()
            shs = .5*stepdir.dot(fisher_vector_product(stepdir))
            lm = np.sqrt(shs / max_kl)
            # logger.log("lagrange multiplier:", lm, "gnorm:", np.linalg.norm(g))
            fullstep = stepdir / lm
            expectedimprove = g.dot(fullstep)
            surrbefore = lossbefore[0]
            stepsize = 1.0
            thbefore = get_flat()
            for _ in range(10):
                thnew = thbefore + fullstep * stepsize
                set_from_flat(thnew)
                meanlosses = surr, kl, *_ = allmean(np.array(compute_losses(*args)))
                improve = surr - surrbefore
                logger.log("Expected: %.3f Actual: %.3f"%(expectedimprove, improve))
                if not np.isfinite(meanlosses).all():
                    logger.log("Got non-finite value of losses -- bad!")
                elif kl > max_kl * 1.5:
                    logger.log("violated KL constraint. shrinking step.")
                elif improve < 0:
                    logger.log("surrogate didn't improve. shrinking step.")
                else:
                    logger.log("Stepsize OK!")
                    break
                stepsize *= .5
            else:
                logger.log("couldn't compute a good step")
                set_from_flat(thbefore)
            if nworkers > 1 and iters_so_far % 20 == 0:
                paramsums = MPI.COMM_WORLD.allgather((thnew.sum(), vfadam.getflat().sum())) # list of tuples
                assert all(np.allclose(ps, paramsums[0]) for ps in paramsums[1:])

        for (lossname, lossval) in zip(loss_names, meanlosses):
            logger.record_tabular(lossname, lossval)

        with timed("vf"):

            for _ in range(vf_iters):
                for (mbob, mbret) in dataset.iterbatches((seg["ob"], seg["tdlamret"]),
                include_final_partial_batch=False, batch_size=64):
                    mbob = sf01(mbob)
                    g = allmean(compute_vflossandgrad(mbob, mbret).numpy())
                    vfadam.update(g, vf_stepsize)

        logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))

        lrlocal = (seg["ep_lens"], seg["ep_rets"]) # local values
        print("ep_lens and ep_rets", lrlocal)
        
        if MPI is not None:
            listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal) # list of tuples
        else:
            listoflrpairs = [lrlocal]

        lens, rews = map(flatten_lists, zip(*listoflrpairs))
        lenbuffer.extend(lens)
        rewbuffer.extend(rews)
        

        logger.record_tabular("EpLenMean", np.mean(lenbuffer))
        logger.record_tabular("EpRewMean", np.mean(rewbuffer))
        logger.record_tabular("EpThisIter", len(lens))
        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens)
        iters_so_far += 1

        logger.record_tabular("EpisodesSoFar", episodes_so_far)
        logger.record_tabular("TimestepsSoFar", timesteps_so_far)
        logger.record_tabular("TimeElapsed", time.time() - tstart)

        if rank==0:
            logger.dump_tabular()

    return pi

def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]

def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])
