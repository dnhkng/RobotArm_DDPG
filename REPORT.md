# DDPG for Continuous Control

This project was completed using the a Deep Deterministic Policy Gradient (DDPG) algorithm.  This type of machine learning algorithm is usefuls for controlling systems with a continous action-space. Some key features of DDPG's are

  - they are deterministic
  - output is a single real value
  - continuous action space
  - off-policy for sample efficiency
  - model-free

# Implementation
The code is based on the DDPG bipedel demo from [Udacity](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-bipedal).

The key changes are as follows. In the model:

  * batch-normalization was performed in the actor and critic networks

In the Agent:
 - the following parameters were modified:
     * WEIGHT_DECAY reduced to 0.0
     * BATCH_SIZE increased to 1024
 - the learning code was moved to it's own function
     * this allowed learning to be called when needed

The following bugs were removed:
 
 > 'torch.nn.utils.clip_grad_norm'

is deprecated, and changed to:
> 'torch.nn.utils.clip_grad_norm_'

The Ornstein-Uhlenbeck noise generator contained a bug that slowed (and sometimes reversed) learning:
>dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])

fixed with 
>dx = self.theta * (self.mu - x) + self.sigma * np.array([np.random.randn() for i in range(len(x))])

The error is dure to the use of the wrong random number generator.  The python random.random() returns a random float in the range [0.0, 1.0). The Numpy random.randn() return a random float sampled from a univariate “normal” (Gaussian) distribution of mean 0 and variance 1. The change allows negative numbers to be generated, and thus allows for the proper exploration noise to be generated.


# Results

In the naive system, I used a deque of the 100 last score to determine when the program was finished. However the system was scoring over 35 points per episode after the 15th episode, so an average over 30 was found after just 54 epsisodes. This is shown in the chart below.

![Chart 1](Images/chart.png)

To fulfil the criteria: "**the average (over 100 episodes) of those average scores is at least +30**", the code was rewritten to run a full 100 episodes. Of course this resulted in an average score over 30; after 100 episodes the average score was 32.08, and the chart of learning is shown below.

![Chart 2](Images/chart2.png)


The final model was run, and the results captured and are shown below:

![Animation](Images/Animation.gif)

# Future Work

This code used Action noise to help in exploration. It will be interesting to try using [Parameter Noise](https://blog.openai.com/better-exploration-with-parameter-noise/) to try to increase the learning rate, as this has been already demontrated. Also, expanding to Distributed Distributional Deterministic Policy Gradients ([D4PG](https://arxiv.org/abs/1804.08617)), would be interesting to test.