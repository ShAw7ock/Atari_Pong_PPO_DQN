# Atari_Pong_DDPG
There are three algorithms in my code for OpenAI Atari Game Pong
You can check the main agent algorithms in the directory 'algos' with DQN, DDPG and PPO.
But only the DQN seem work finally, DDPG and PPO can not train the agent efficiently.

# Update at 2021/01/04
The reason why PPO and DDPG Algorithm can not work at first is that the Convolution Layer is too simple and
the features can't make the agent train usefully.

So I change the Convolution Layers of the 'ppo_networks.py' in 'utils' and add the 'ddpg_net_complicate.py'
Since the complex convolution layers are used, the PPO Algorithm can work well and train the agent usefully.

But with the difference between DDPG and PPO (DDPG is the deterministic action algorithm but PPO is not, so DDPG will use
the complex neural networks frequently to select the deterministic actions), DDPG Algorithm can not train the agent usefully

# Thanks for using ShAw7ock's code!
