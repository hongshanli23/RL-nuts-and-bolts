My implementation is still not stable

```
Traceback (most recent call last):
  File "v1.py", line 173, in <module>
    ckpt_dir='/tmp/0/',
  File "v1.py", line 129, in PPO_clip
    loss = compute_loss(pi, trajectory, midx, eps) 
  File "v1.py", line 44, in compute_loss
    dist = pi.dist(pi.policy_net(obs))
  File "/home/ubuntu/reinforcement-learning/RL-notes/rlkits/policies.py", line 109, in dist
    return Normal(mean, torch.exp(logstd))
  File "/home/ubuntu/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages/torch/distributions/normal.py", line 50, in __init__
    super(Normal, self).__init__(batch_shape, validate_args=validate_args)
  File "/home/ubuntu/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages/torch/distributions/distribution.py", line 53, in __init__
    raise ValueError("The parameter {} has invalid values".format(param))
ValueError: The parameter loc has invalid values
```
This means standard deviation goes to 0 


Still did not see signs of life. Nan problem is gone after I replace ReLU by tanh to make sure each activation is clipped in certain range. 
I think it is a hyperparameter issues. For that reason I think it is worth-while to put my thing onto SageMaker. Or try openai's parameters. Try to run openai's ppo implementation on Pendulum tomorrow. 