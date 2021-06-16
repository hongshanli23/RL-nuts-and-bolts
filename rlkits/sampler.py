import numpy as np
from rlkits.env_batch import EnvBatch
from rlkits.env_batch import SingleEnvBatch, ParallelEnvBatch




class ParallelEnvTrajectorySampler:
    """Sample a trajectory"""
    def __init__(self, env, policy, nsteps, reward_transform=None, gamma=0.99):
        """
        nsteps: number of steps to sample each time
        """
        print("Type of the environment", type(env))
        if not isinstance(env, ParallelEnvBatch):
            env = SingleEnvBatch(env)
        
        self.env = env
        self.n = self.env.nenvs
     
        self.policy = policy
        self.curr_state = self.env.reset()
        self.total_timesteps = 0
        self.nsteps = nsteps
        self.gamma = gamma
        
        if reward_transform:
            self.reward_transform = reward_transform
        else:
            self.reward_transform = lambda x: x
        
        self.curr_ep_rew = np.zeros(self.n) # current episode reward (curmulative)
        self.curr_ep_len = np.zeros(self.n)   # current episode length
        
        
        # initialize arrays to be returned for each sampling step
        self.obs = np.array([self.curr_state for _ in range(self.nsteps)], 
                            dtype=np.float32)
        self.rews = np.zeros((self.nsteps, self.n), np.float32)
        self.vpreds = np.zeros((self.nsteps, self.n), np.float32)
        self.dones = np.zeros((self.nsteps, self.n), np.bool)
        
        # prediction of next state
        self.nextvpreds = self.vpreds.copy() 
        
        ac = self.env.action_space.sample()
        self.actions = np.array([ac for _ in range(self.nsteps)])
        
        # DEADLY BUG
        # self.log_prob = np.array([ac for _ in range(self.nsteps)])
        
        # make sure to set dtype of log prob to float32
        # if left unset, it will take the data type 
        # of ac (int for discrete action)
        self.log_prob = np.array(
            [ac for _ in range(self.nsteps)], dtype=np.float32)
        
        # estimated state action value
        #self.Q = self.vpreds.copy()
        #self.adv = self.vpreds.copy()
    
    
    def __call__(self, callback=None):
        """  
        callback: callback on trajectory
        """
        # episode reward and episode length
        # of this sampling
        ep_rets = [[] for _ in range(self.n)]
        ep_lens = [[] for _ in range(self.n)]
          
        for i in range(self.nsteps):
            action, log_prob, vpred = self.policy.step(self.curr_state)
            nx_state, rew, done, _ = self.env.step(action)           
            rew = self.reward_transform(rew)
            
            self.curr_ep_rew += rew
            self.curr_ep_len[~done] +=1
            
            for j in range(self.n):
                if done[j]==True:
                    ep_rets[j].append(self.curr_ep_rew[j].item())
                    ep_lens[j].append(self.curr_ep_len[j].item())
            
            self.curr_ep_rew[done] = 0.0
            self.curr_ep_len[done] = 0
            
            nextvpred = self.policy.predict_state_value(nx_state)
         
            self.obs[i] = self.curr_state
            self.actions[i] = action
            self.log_prob[i] = log_prob
            self.rews[i] = rew
            self.vpreds[i] = vpred
            self.dones[i] = done
            self.nextvpreds[i] = nextvpred
            self.total_timesteps+=self.n
            
            self.curr_state = nx_state
            
        # add the curr_ep_rew to each ep_rets
        # add curr_ep_len to each ep_lens
        for j in range(self.n):
            ep_rets[j].append(self.curr_ep_rew[j])
            ep_lens[j].append(self.curr_ep_len[j])
        
        
        trajectory = {
            "obs" : self.obs,
            "actions":self.actions,
            "log_prob":self.log_prob,
            "rews": self.rews,
            "vpreds": self.vpreds,
            "dones":self.dones,
            "ep_rets" : ep_rets,
            "ep_lens" : ep_lens
        }
        
        if callback:
            callback(self, trajectory)
        
        # compute estimate state action value
        return trajectory
    

### sampler callbacks ####

def estimate_Q(sampler, trajectory):
    """Callback for estimating state action value in a parallel env"""
    Q = np.zeros_like(sampler.vpreds, dtype=np.float32)
    
    Gt = sampler.nextvpreds[-1]
    for t in reversed(range(sampler.nsteps)):
        not_done = ~trajectory['dones'][t]
        Q[t] = sampler.rews[t] + sampler.gamma * Gt * not_done
        Gt = Q[t]
    trajectory['Q'] = Q
    return

### Helpers ####



def aggregate_experience(arr):
    """
    aggregate experiences from all envs 
    each expr from one env can be used for one update
    I want to expr from the same env to stick together
    This means I need to tranpose the array so that
    (nenvs, nsteps, ...)
    so that when I reshape (C style) the array to merge the first two axes
    the exprs from the same env are contiguous     
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])

def shuffle_experience(trajectory):
    """Shuffle the experience in-place
    
    trajectory: dictionary of aggregated experience
    """
    idx = np.array(range(len(trajectory['obs'])))
    np.random.shuffle(idx)
    
    for k, v in trajectory.items():
        if isinstance(v, np.ndarray):
            trajectory[k] = v[idx]
    return trajectory