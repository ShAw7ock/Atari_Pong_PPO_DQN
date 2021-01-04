from gym import spaces
import torch
import torch.nn as nn


class ActorNetwork(nn.Module):
    def __init__(self, observation_space, action_space, hidden_sizes=256):
        super(ActorNetwork, self).__init__()
        assert type(observation_space) == spaces.Box
        assert len(observation_space.shape) == 3
        assert type(action_space) == spaces.Discrete

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=observation_space.shape[0], out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.resnetBlock = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32)
        )
        self.denseLayer = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU()
        )
        self.relu = nn.ReLU()

        self.fc_layers = nn.Sequential(
            nn.Linear(2 * 10 * 10, hidden_sizes),
            nn.ReLU(),
            nn.Linear(hidden_sizes, action_space.n),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        x0 = self.conv_layers(state)
        x1 = self.resnetBlock(x0) + x0
        x2 = self.relu(x1)
        x3 = self.resnetBlock(x2) + x2
        x4 = self.relu(x3)
        x5 = self.resnetBlock(x4) + x4
        x6 = self.relu(x5)
        x7 = self.resnetBlock(x6) + x6
        x8 = self.relu(x7)
        x9 = self.denseLayer(x8)
        probs = self.fc_layers(x9.view(state.size()[0], -1))
        return probs


class CriticNetwork(nn.Module):
    def __init__(self, observation_space, action_dim=1, hidden_sizes=256):
        super(CriticNetwork, self).__init__()
        assert type(observation_space) == spaces.Box
        assert len(observation_space.shape) == 3

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=observation_space.shape[0], out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.resnetBlock = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32)
        )
        self.denseLayer = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU()
        )
        self.relu = nn.ReLU()
        self.fc_obs = nn.Linear(2 * 10 * 10, hidden_sizes)
        self.fc_act = nn.Linear(action_dim, hidden_sizes)
        self.fc_out = nn.Linear(hidden_sizes, 1)

    def forward(self,state, action):
        x0 = self.conv_layers(state)
        x1 = self.resnetBlock(x0) + x0
        x2 = self.relu(x1)
        x3 = self.resnetBlock(x2) + x2
        x4 = self.relu(x3)
        x5 = self.resnetBlock(x4) + x4
        x6 = self.relu(x5)
        x7 = self.resnetBlock(x6) + x6
        x8 = self.relu(x7)
        conv_out_obs = self.denseLayer(x8).view(state.size()[0], -1)
        out_obs = self.fc_obs(conv_out_obs)
        out_act = self.fc_act(action)
        out_obs_act = out_obs + out_act
        out = self.relu(out_obs_act)
        value = self.fc_out(out)

        return value