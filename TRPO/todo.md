## May 9 2020

### Todo
It is a good idea to try various hyperaparemters on openai's implementation of trpo on pendulum. Do it on SageMaker, kick off a bunch of training jobs with different parameters. 
Maybe I should even try with Hyperparameter tunning jobs. 

I want to see if there's sign of life

get the trained model and visualize how it behaves on the envioronment. 


### Have done
I did use sagemaker for the first time for a real use case. I build two images, one for generic RL uses, one is for openai. This is considered a two stage docker image build. It's more efficient, because the first stage takes long time, so we don't change it that much; second stage build fast, so we can change it more frequently as we roll out more experiments. 

I transformed rewards so that good actions have positive rewards and bad actions have negative rewards. 

After doing parallel experiments on SageMaker, I found that longer trajectory leads to more stable training. 

### Plan to do

* Deeper network but narrower. 
* Publish a notebook in reinforcement_learning


## May 11 2020
in my torch implementation, update value fn correctly
