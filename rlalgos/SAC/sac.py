

# When computing loss of a policy, should we use the action from the
# replay buffer (older policy), or should we use the action sampled
# from the current policy ?

# According to SAC paper, it seems that value functions are updated
# using actions sampled from the old policy, whereas the policy is
# updated through the actions sampled from the current policy via
# reparametrization trick. Why it makes sense to do it this way?

# For value net update, using the action sampled from the current
# policy amounts to a better label. The current policy makes an
# action so that the corresponding state action value Q(s, a) is a more
# accurate approximation of the state value V(s).

# For policy update, we need the policy gradient of the current
# policy, so naturally we should sample an action (for a state
# retrieved from replay buffer).

# In DDPG, the same better label for value function is given by
# the state action approximation of the target net over action
# sampled from target policy (on the next state).

# the reparametrization makes the distribution of
# acs_curr independent from the parameters of the
# policy net.
# reparametrization is a way to write the expectation
# independent from the parameter

# The advantage of reparametrization
# https://gregorygundersen.com/blog/2018/04/29/reparameterization/
# 1. It allows us to re-write gradient of expectation
# as expectation of gradient. Hence, we can use Monte
# Carlo method to estimate the gradient.
# 2. Stability: reparametrization limits the variance
# of the estimate. It basically caps the variance of
# the estimate to the variance of N(0,1)
# To see how reparam helps stability, checkout
# https://nbviewer.jupyter.org/github/gokererdogan/Notebooks/blob/master/Reparameterization%20Trick.ipynb

def compute_loss(policy, Q1, Q1_targ, Q2, Q2_targ,
                 batch, gamma, alpha):
    obs, acs, rews, nxs, dones = batch['obs0'], batch['actions'],\
        batch['rewards'], batch['obs1'], batch['terminals1']

    obs, acs, rews, nxs, dones = to_tensor(
        obs, acs, rews, nxs, dones)
    # target for value net
    with torch.no_grad():
        # next action needs to be sampled from the CURRENT policy
        # in contrast to DDPG
        nxa = policy(nxs)
        nxv = torch.minimum(
            Q1_targ(nxs, nxa), Q2_targ(nxs, nxa)
        )

    assert rews.shape == nxv.shape, f"{rews.shape}, {nxv.shape}"

    # target for value net
    y = rews + gamma*(1-dones)*(nxv - alpha*policy.log_proba(nxa))

    # value loss
    q1, q2 = Q1(obs, acs), Q2(obs, acs)
    Q1_loss = F.mse_loss(q1, y)
    Q2_loss = F.mse_loss(q2, y)

    # policy loss
    # sample an action from \pi(\cdot | s) via reparam
    m, std = policy(obs)
    acs_curr = torch.tanh(
        m + std*Gaussian(0, 1).sample())

    # log probability of acs_curr
    # pushing m + std * \zeta changes the distribtution
    # log probability should be the log probability
    # of m + std*Gaussian(0,1)
    # Ultimately, we want to encourage entropy, so it is
    # equivalent to make std a bit larger
    policy_loss = torch.mean(
        torch.minimum(q1, q2) - alpha*policy.log_proba(

    # predicted q-value for the current state and action
    q_pred = value_net(obs, acs)
    value_loss = F.mse_loss(q_pred, q_targ)

    # policy loss
    policy_loss = -value_net(obs, policy(obs)).mean()

    res = {
            "policy_loss": policy_loss,
            "value_loss": value_loss
          }


def SAC(*,
        ):

    Q1 = QNet()
    Q2 = QNet()
    policy = StochasticPolicy()

    replay_buffer = Memory(
        limit=buf_size,
        action_shape=ac_space.shape
    )


    replay_buffer = Memory(
        limit=buf_size,
        action_shape=ac_space.shape,
        observation_shape=ob_space.shape
    )

    best_ret = np.float('-inf')
    rolling_buf_episode_rets = deque(maxlen=10)
    curr_state = env.reset()
    policy.reset()

    step=0
    while step <= nsteps:
        if step < warm_up_steps:
            action = policy.random_action()
        else:
            action = policy.step(curr_state)
        nx, rew, done, _ = env.step(action)
        # record to the replay buffer
        assert nx.shape == ob_space.shape, f"{nx.shape},{ob_space.shape}"
        assert action.shape == ac_space.shape, f"{action.shape},{ac_space.shape}"
        replay_buffer.append(
            obs0=curr_state, action=action, reward=rew, obs1=nx, terminal1=done
        )
        episode_rews += rew
        if done:
            curr_state = env.reset()
            policy.reset()  # reset random process
            rolling_buf_episode_rets.append(episode_rews)
            episode_rews = 0
        else:
            curr_state = nx

        # train after warm up steps
        if step < warm_up_steps: continue
        batch = replay_buffer.sample(batch_size)
        losses = compute_loss(policy, Q1, Q2, batch,
                              gamma, alpha)
