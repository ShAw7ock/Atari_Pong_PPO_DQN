import numpy as np
import torch
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torch.nn.functional as F
from utils.ppo_networks import ActorNetwork, CriticNetwork


class PPOAgent:
    def __init__(self, observation_space,
                 action_space,
                 hidden_sizes=256,
                 critic_lr=0.001,
                 actor_lr=0.002,
                 gamma=0.99,
                 batch_size=32,
                 eps_clip=0.2,
                 k_epochs=4,
                 ):
        self.observation_space = observation_space
        self.action_space = action_space
        self.hidden_sizes = hidden_sizes
        self.critic_lr = critic_lr
        self.actor_lr = actor_lr
        self.gamma = gamma
        self.batch_size = 32
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.critic = CriticNetwork(observation_space, hidden_sizes=hidden_sizes).to(self.device)
        # self.target_critic = CriticNetwork(observation_space, hidden_sizes=hidden_sizes).to(self.device)
        self.critic_optim = torch.optim.RMSprop(self.critic.parameters(), lr=critic_lr)

        self.actor = ActorNetwork(observation_space, action_space, hidden_sizes=hidden_sizes).to(self.device)
        # self.target_actor = ActorNetwork(observation_space, action_space, hidden_sizes=hidden_sizes).to(self.device)
        self.actor_optim = torch.optim.RMSprop(self.actor.parameters(), lr=actor_lr)

        self.loss_func = torch.nn.MSELoss()
        # PPO algorithm is on-policy and don't need a complicate buffer, so that the memory need to clear after update
        self.memory = []

    def step(self, state):
        state = np.array(state) / 255.0
        state_as_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_probs = self.actor(state_as_tensor)
        dist = Categorical(action_probs.squeeze(0))
        action = dist.sample()

        return action.item(), action_probs[:, action.item()].item()

    def get_value(self, state):
        state = np.array(state) / 255.0
        state_as_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            value = self.critic(state_as_tensor)
        return value.item

    def store_transition(self, transition):
        self.memory.append(transition)

    def update(self):
        actions = torch.tensor([m.action for m in self.memory], dtype=torch.long).view(-1, 1).to(self.device)
        rewards = [m.reward for m in self.memory]
        is_terminals = [m.done for m in self.memory]
        old_action_log_probs = torch.tensor([m.a_log_prob for m in self.memory],
                                            dtype=torch.float).view(-1, 1).to(self.device)

        discounted_reward = 0
        mc_rewards = []
        # Compute the Monte-Carlo rewards within one trajectory (in an update_freq interval)
        for reward, is_terminal in zip(reversed(rewards), reversed(is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + self.gamma * discounted_reward
            mc_rewards.insert(0, discounted_reward)
        mc_rewards = torch.tensor(mc_rewards, dtype=torch.float).to(self.device)

        for kt in range(self.k_epochs):
            for index in BatchSampler(SubsetRandomSampler(range(len(self.memory))), self.batch_size, False):
                mc_rewards_index = mc_rewards[index].view(-1, 1)
                states = self.encode_sample(index)
                states_as_tensor = torch.from_numpy(states).float().to(self.device)
                value_index = self.critic(states_as_tensor)
                delta = mc_rewards_index - value_index
                advantage = delta.detach()

                action_prob = self.actor(states_as_tensor).gather(1, actions[index])

                ratio = (action_prob / old_action_log_probs[index])
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage

                actor_loss = -torch.min(surr1, surr2).mean()
                self.actor_optim.zero_grad()
                actor_loss.backward()
                self.actor_optim.step()

                critic_loss = self.loss_func(mc_rewards_index, value_index)
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

        del self.memory[:]

    def save(self, filename):
        net_param = {'actor': self.actor.state_dict(),
                     'critic': self.critic.state_dict(),
                     'actor_optimizer': self.actor_optim.state_dict(),
                     'critic_optimizer': self.critic_optim.state_dict()}
        torch.save(net_param, filename)

    def load_params(self, params):
        self.actor.load_state_dict(params['actor'])
        self.critic.load_state_dict(params['critic'])
        self.actor_optim.load_state_dict(params['actor_optimizer'])
        self.critic_optim.load_state_dict(params['critic_optimizer'])

    def encode_sample(self, indices):
        states = []
        for i in indices:
            state = self.memory[i].state
            states.append(np.array(state) / 255.0)
        return np.array(states)
