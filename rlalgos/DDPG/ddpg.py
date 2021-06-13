# if the goal is to be sample efficient
# then we should not use parallel env

from collections import deque
import time

from 


def DDPG(*,
    env_name,
    nsteps,
    niters,
    gamma,
    pi_lr,
    v_lr,
    ployak,
    batch_size,
    log_interval,
    max_grad_norm,
    log_dir,
    ckpt_dir,
    **network_kwargs,
):
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    logger.configure(dir=log_dir) 
    
    env = gym.make(env_name)
    ob_space = env.observation_space
    ac_space = env.action_space
    
    policy = DeterministicPolicy(
        ob_space=ob_space, ac_space=ac_space, ckpt_dir=ckpt_dir,
        **network_kwargs
    )
    target_policy = DeterministicPolicy(
        ob_space=ob_space, ac_space=ac_space, ckpt_dir=ckpt_dir,
        **network_kwargs
    )

    value_net = QNetForContinuousAction(
        ob_space=ob_space, ac_space=ac_space, ckpt_dir=ckpt_dir,
        **network_kwargs
    )
    
    target_value_net = QNetForContinuousAction(
        ob_space=ob_space, ac_space=ac_space, ckpt_dir=ckpt_dir,
        **network_kwargs
    )

    replay_buffer = ReplayBuffer(
        size=buf_size,
        batch_size=batch_size
    )
    
    rolling_buf_episode_rets = deque(maxlen=100) 
    curr_state = env.reset()
    for i in range(1 + niters):
        # sample nsteps experiences and save to replay buffer
        for _ in range(nsteps):
            ac = policy.step(curr_state)
            nx, rew, done, _ = env.step(ac)
            replay_buffer.append(
                s=curr_state, a=ac, r=rew, nx=nx, done=done
            )
            curr_state = nx
            if done:
                curr_state = env.reset()
                
        # train the policy and value
        for _ in range(nupdates):
            obs, acs, rews, nxs, dones = replay_buffer.sample()
            # target for qnet
            with torch.no_grad():
                nx_state_vals = target_value_net(
                    nxs, target_policy(nxs)
                )
            # Q_targ(s', \mu_targ(s'))
            q_targ = rews + gamma * (1 - dones) * nx_state_vals
            
            # Q(s, a)
            q_pred = value_net(obs, acs)
            q_loss = F.mse(q_pred, q_targ)
            voptimizer.zero_grad()
            q_loss.backward()
            voptimizer.step()

            # update policy through value
            policy_loss = -value_net(obs, policy(acs))
            poptimizer.zero_grad()
            policy_loss.backward()
            poptimizer.step()
        
        if i % model_update_freqency:
            # update target value net and policy
            # through linear interpolation
            for p, p_targ in zip(policy.parameters(), 
                target_policy.parameters()):
                p_targ.copy_(polyak*p_targ + (1-polyak)*p)

            for p, p_targ in zip(value_net.parameters(),
                target_value_net.parameters()):
                p_targ.copy_(polyak*p_targ + (1-polyak)*p)
        
        if i % log_frequency == 0 or i == 1:
            # loss from policy and value 
            for k, v in lossvals:
                logger.record_tabular(k, np.mean(v))
            
            # evaluate the policy $n_trials times
            # TODO use parallel env sampler here
            trajectory = evaluate_policy(env_name, policy, n_trials)

            rolling_buf_episode_rets.extend(rets)
            ret = np.mean(rolling_buf_episode_rets)
            logger.record_tabular("ma_ep_ret", ret)
            logger.record_tabular("mean_step_rew", np.mean(
                trajectory["rews"])

            pw, tpw = policy.weight(), target_policy.weight()
            vw, tvw = value_net.weight(), target_value_net.weight()
            logger.record_tabular("policy_net_weight", pw)
            logger.record_tabular("target_policy_net_weight", tpw)
            logger.record_tabular("value_net_weight", vw)
            logger.record_tabular("target_value_net_weight", tvw)
            
            if ret > best_ret:
                best_ret = ret
                policy.save_ckpt('best')
                torch.save(poptimizer, os.path.join(ckpt_dir, 'poptim-best.pth'))
                torch.save(voptimizer, os.path.join(ckpt_dir, 'voptim-best.pth'))
    
    pi.save_ckpt('final')
    torch.save(poptimizer, os.path.join(ckpt_dir, 'poptim-final.pth'))
    torch.save(voptimizer, os.path.join(ckpt_dir, 'voptim-final.pth'))
    return 




