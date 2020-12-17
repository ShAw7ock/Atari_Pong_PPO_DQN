from gym import spaces
import numpy as np

from utils.dqn_neurips_network import DQN
from utils.memory import ReplayBuffer
import torch
import torch.nn.functional as F


class DQNAgent:
    def __init__(self,
                 observation_space: spaces.Box,
                 action_space: spaces.Discrete,
                 replay_buffer: ReplayBuffer,
                 lr=1e-4,
                 batch_size=32,
                 gamma=0.99,
                 device=torch.device("cpu")):
        """
        Initialise the DQN algorithm using the Adam optimiser
        :param action_space: the action space of the environment
        :param observation_space: the state space of the environment
        :param replay_buffer: storage for experience replay
        :param lr: the learning rate for Adam
        :param batch_size: the batch size
        :param gamma: the discount factor
        """

        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.gamma = gamma
        self.device = "cuda:0" if torch.cuda.is_available() else device

        self.policy_network = DQN(observation_space, action_space).to(self.device)
        self.target_network = DQN(observation_space, action_space).to(self.device)
        self.update_target()
        self.target_network.eval()

        self.optimiser = torch.optim.RMSprop(self.policy_network.parameters(), lr=lr)

    def update(self):
        """
        Optimise the TD-error over a single minibatch of transitions
        :return: the loss
        """
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = np.array(states) / 255.0
        next_states = np.array(next_states) / 255.0
        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).long().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        dones = torch.from_numpy(dones).float().to(self.device)

        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            max_next_q_values, _ = next_q_values.max(1)
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values

        input_q_values = self.policy_network(states)
        input_q_values = input_q_values.gather(1, actions.unsqueeze(1)).squeeze()

        loss = F.smooth_l1_loss(input_q_values, target_q_values)

        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()
        del states
        del next_states
        return loss.item()

    def update_target(self):
        """
        Update the target Q-network by copying the weights from the current Q-network
        """
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def step(self, state: np.ndarray):
        """
        Select an action greedily from the Q-network given the state
        :param state: the current state
        :return: the action to take
        """
        state = np.array(state) / 255.0
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_network(state)   # q_values: tensor([[]]) so q_values.max(dim=1)
            _, action = q_values.max(dim=1)
            return action.item()

    def save(self, filename):
        torch.save(self.policy_network.state_dict(), filename)
