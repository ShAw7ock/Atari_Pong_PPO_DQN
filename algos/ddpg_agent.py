from gym import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from utils.networks import ActorNetwork, CriticNetwork
from utils.misc import soft_update, hard_update


class DDPGAgent():
    def __init__(self, observation_space,
                 action_space,
                 replay_buffer,
                 hidden_sizes=256,
                 critic_lr=0.001,
                 actor_lr=0.002,
                 batch_size=32,
                 gamma=0.90,
                 tau=0.01,
                 use_cuda=False):

        self.observation_space = observation_space
        self.action_space = action_space
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.device = 'cuda:0' if use_cuda and torch.cuda.is_available() else 'cpu'

        self.critic = CriticNetwork(observation_space=observation_space, hidden_sizes=hidden_sizes)
        self.target_critic = CriticNetwork(observation_space=observation_space, hidden_sizes=hidden_sizes)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=critic_lr)
        hard_update(self.target_critic, self.critic)

        self.actor = ActorNetwork(observation_space=observation_space, action_space=action_space,
                                  hidden_sizes=hidden_sizes)
        self.target_actor = ActorNetwork(observation_space=observation_space, action_space=action_space,
                                         hidden_sizes=hidden_sizes)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=actor_lr)
        hard_update(self.target_actor, self.actor)

        self.exploration = 0.3      # epsilon for eps-greedy

    def update(self):
        pass

    def update_all_target(self):
        pass

    def reset_noise(self):
        pass

    def scale_noise(self):
        pass

    def step(self):
        pass

    def get_params(self):
        return {'actor': self.actor.state_dict(),
                'critic': self.critic.state_dict(),
                'target_actor': self.target_actor.state_dict(),
                'target_critic': self.target_critic.state_dict(),
                'actor_optimizer': self.actor_optim.state_dict(),
                'critic_optimizer': self.critic_optim.state_dict()}

    def load_params(self, params):
        self.actor.load_state_dict(params['actor'])
        self.critic.load_state_dict(params['critic'])
        self.target_actor.load_state_dict(params['target_actor'])
        self.target_critic.load_state_dict(params['target_critic'])
        self.actor_optim.load_state_dict(params['actor_optimizer'])
        self.critic_optim.load_state_dict(params['critic_optimizer'])
