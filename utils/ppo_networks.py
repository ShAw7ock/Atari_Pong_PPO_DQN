import torch
import torch.nn as nn
from gym import spaces


class ActorNetwork(nn.Module):
    def __init__(self, observation_space, action_space, hidden_sizes=256):
        super(ActorNetwork, self).__init__()
        assert type(observation_space) == spaces.Box
        assert len(observation_space.shape) == 3
        assert type(action_space) == spaces.Discrete

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=observation_space.shape[0], out_channels=16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2),
            nn.ReLU()
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(32 * 9 * 9, hidden_sizes),
            nn.ReLU(),
            nn.Linear(hidden_sizes, action_space.n),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        conv_out = self.conv_layers(state).view(state.size()[0], -1)
        probs = self.fc_layers(conv_out)
        return probs


class CriticNetwork(nn.Module):
    def __init__(self, observation_space, hidden_sizes=256):
        super(CriticNetwork, self).__init__()
        assert type(observation_space) == spaces.Box
        assert len(observation_space.shape) == 3

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=observation_space.shape[0], out_channels=16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2),
            nn.ReLU()
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(32 * 9 * 9, hidden_sizes),
            nn.ReLU(),
            nn.Linear(hidden_sizes, 1)
        )

    def forward(self, state):
        conv_out = self.conv_layers(state).view(state.size()[0], -1)
        value = self.fc_layers(conv_out)
        return value
