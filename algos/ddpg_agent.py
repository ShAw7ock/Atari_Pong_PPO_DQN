from gym import spaces
import numpy as np
import torch
import torch.optim as optim
from utils.networks import ActorNetwork, CriticNetwork
from utils.misc import soft_update, hard_update


class DDPGAgent:
    def __init__(self, observation_space,
                 action_space,
                 replay_buffer,
                 hidden_sizes=256,
                 critic_lr=0.001,
                 actor_lr=0.002,
                 batch_size=32,
                 gamma=0.90,
                 tau=0.01):

        self.observation_space = observation_space
        self.action_space = action_space
        self.replay_buffer = replay_buffer
        self.hidden_sizes = hidden_sizes
        self.critic_lr = critic_lr
        self.actor_lr = actor_lr
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        self.critic = CriticNetwork(observation_space=observation_space, hidden_sizes=hidden_sizes).to(self.device)
        self.target_critic = CriticNetwork(observation_space=observation_space, hidden_sizes=hidden_sizes).to(self.device)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=critic_lr)
        hard_update(self.target_critic, self.critic)

        self.actor = ActorNetwork(observation_space=observation_space, action_space=action_space,
                                  hidden_sizes=hidden_sizes).to(self.device)
        self.target_actor = ActorNetwork(observation_space=observation_space, action_space=action_space,
                                         hidden_sizes=hidden_sizes).to(self.device)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=actor_lr)
        hard_update(self.target_actor, self.actor)

        self.loss_fuc = torch.nn.MSELoss()

    def update(self):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)   # arrays
        next_actions = self.batch_step(next_states, mode='target')

        states_tensor = np.array(states) / 255.0
        next_states_tensor = np.array(next_states) / 255.0
        states_tensor = torch.from_numpy(states_tensor).float().to(self.device)
        actions_tensor = torch.from_numpy(actions).float().to(self.device)
        rewards_tensor = torch.from_numpy(rewards).float().to(self.device)
        next_states_tensor = torch.from_numpy(next_states_tensor).float().to(self.device)
        next_actions_tensor = torch.from_numpy(next_actions).float().to(self.device)
        dones_tensor = torch.from_numpy(dones).float().to(self.device)

        target_q = rewards_tensor + (1 - dones_tensor) * self.gamma *\
                   self.target_critic(next_states_tensor, next_actions_tensor.unsqueeze(1))
        actual_q = self.critic(states_tensor, actions_tensor.unsqueeze(1))
        critic_loss = self.loss_fuc(target_q, actual_q)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        train_actions = self.batch_step(states, mode='net')
        train_actions_tensor = torch.from_numpy(train_actions).float().to(self.device)
        q = self.critic(states_tensor, train_actions_tensor.unsqueeze(1))
        actor_loss = -torch.mean(q)
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

    def update_target(self):
        soft_update(self.target_critic, self.critic, self.tau)
        soft_update(self.target_actor, self.actor, self.tau)

    def step(self, obs):
        """
        Take a step forward in environment with (or not) exploration action
        Inputs:
            obs (ndarray): Observations for this agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            action (int): Actions for this agent
        Used for 'train' mode
        """
        state = np.array(obs) / 255.0
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_probs = self.actor(state).cpu().numpy()
            return np.random.choice(np.arange(action_probs.shape[1]), p=action_probs.ravel())

    def batch_step(self, batch_obs, mode):
        """
        Because function 'step' can only receive one piece of state,
        but DDPG algorithms need batch training.
        This function is used to receive batch observations and return batch actions.
        mode = ['net', 'target']
        """
        if not type(batch_obs) == np.ndarray:
            batch_obs = batch_obs.cpu().numpy()
        batch_actions = []
        for index in range(batch_obs.shape[0]):
            state = batch_obs[index] / 255.0
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                if mode == 'target':
                    action_probs = self.target_actor(state).cpu().numpy()
                else:
                    action_probs = self.actor(state).cpu().numpy()
                action = np.random.choice(np.arange(action_probs.shape[1]), p=action_probs.ravel())
                batch_actions.append(action)
        return np.array(batch_actions)

    def step_best(self, obs):
        """
        This function chooses the action based on the best action
        but not the probability.
        Used for 'evaluation' mode.
        """
        state = np.array(obs) / 255.0
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_probs = self.actor(state)
            _, action = action_probs.max(dim=1)
            return action

    def save(self, filename):
        net_param = {'actor': self.actor.state_dict(),
                     'critic': self.critic.state_dict(),
                     'target_actor': self.target_actor.state_dict(),
                     'target_critic': self.target_critic.state_dict(),
                     'actor_optimizer': self.actor_optim.state_dict(),
                     'critic_optimizer': self.critic_optim.state_dict()}
        torch.save(net_param, filename)

    def load_params(self, params):
        self.actor.load_state_dict(params['actor'])
        self.critic.load_state_dict(params['critic'])
        self.target_actor.load_state_dict(params['target_actor'])
        self.target_critic.load_state_dict(params['target_critic'])
        self.actor_optim.load_state_dict(params['actor_optimizer'])
        self.critic_optim.load_state_dict(params['critic_optimizer'])
