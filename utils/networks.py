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
            nn.Conv2d(in_channels=observation_space.shape[0], out_channels=16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2),
            nn.ReLU()
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(32 * 9 * 9, hidden_sizes),
            nn.ReLU(),
            nn.Linear(hidden_sizes, action_space.n),
            nn.Softmax()
        )

    def forward(self, state):
        conv_out = self.conv_layers(state).view(state.size()[0], -1)
        probs = self.fc_layers(conv_out)
        return probs


class CriticNetwork(nn.Module):
    def __init__(self, observation_space, action_dim=1, hidden_sizes=256):
        super(CriticNetwork, self).__init__()
        assert type(observation_space) == spaces.Box
        assert len(observation_space.shape) == 3

        self.conv_layers_observation = nn.Sequential(
            nn.Conv2d(in_channels=observation_space.shape[0], out_channels=16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2),
            nn.ReLU()
        )

        self.fc_obs = nn.Linear(32 * 9 * 9, hidden_sizes)
        self.fc_act = nn.Linear(action_dim, hidden_sizes)
        self.fc_out = nn.Linear(hidden_sizes, 1)
        self.nonLinearity = nn.ReLU

    def forward(self, state, action):
        conv_out_obs = self.conv_layers_observation(state).view(state.size()[0], -1)
        out_obs = self.fc_obs(conv_out_obs)
        out_act = self.fc_act(action)
        out = out_obs + out_act
        out = self.nonLinearity(out)
        return self.fc_out(out)
