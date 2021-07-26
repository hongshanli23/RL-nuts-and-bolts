


def compute_loss(policy, Q1, Q1_targ, Q2, Q2_targ,
                 batch, gamma, alpha):
    obs, acs, rews, nxs, dones = batch['obs0'], batch['actions'],\
        batch['rewards'], batch['obs1'], batch['terminals1']

    obs, acs, rews, nxs, dones = to_tensor(
        obs, acs, rews, nxs, dones)
    # target for value net
    with torch.no_grad():

        nx_state_vals = target_value_net(nxs,
                                         policy(nxs))
    assert rews.shape == nx_state_vals.shape, f"{rews.shape}, {nx_state_vals.shape}"
    q_targ = rews + (1 - dones)*gamma*nx_state_vals

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
