import argparse
import os
from pathlib import Path
import torch
import numpy as np
import gym
from utils.memory import ReplayBuffer
from utils.env_wrappers import *
from algos.ddpg_agent import DDPGAgent
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')


def train_your_nicest_model():
    run(config)


def run(config):
    np.random.seed(config.seed)

    assert "NoFrameskip" in config.env, "Require environment with no frameskip"
    env = gym.make(config.env)
    env.seed(config.seed)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeEnv(env)
    env = FireResetEnv(env)
    env = WarpFrame(env)
    env = PyTorchFrame(env)
    env = ClipRewardEnv(env)
    env = FrameStack(env, 4)
    env = gym.wrappers.Monitor(env, './video/', video_callable=False, force=True)

    replay_buffer = ReplayBuffer(config.buffer_size)

    agent = DDPGAgent(
        env.observation_space,
        env.action_space,
        replay_buffer,
        hidden_sizes=config.hidden_sizes,
        critic_lr=config.critic_lr,
        actor_lr=config.actor_lr,
        batch_size=config.batch_size,
        gamma=config.discounted_factor,
        tau=config.tau,
        use_cuda=config.use_cuda
    )

    total_rewards = []
    episodes_count = 0

    for episode_i in range(config.num_episodes):
        episode_reward = 0.0
        done = False
        state = env.reset()
        while not done:
            action = agent.step(state)
            next_state, reward, done, info = env.step(action)
            agent.replay_buffer.add(state, action, reward, next_state, float(done))
            episode_reward += reward
            episodes_count += 1

            if len(replay_buffer) > config.batch_size and episodes_count > config.learning_start:
                agent.update()
            if episodes_count > config.learning_start and episodes_count % config.update_target_freq == 0:
                agent.update_target()

            if done:
                total_rewards.append(episode_reward)
                break
            state = next_state

        if done and episode_i % config.print_freq == 0:
            mean_100ep_reward = np.mean(total_rewards[-101:-1])
            print("********************************************************")
            print("steps: {}".format(episodes_count))
            print("episodes: {}".format(episode_i))
            print("mean 100 episode reward: {}".format(mean_100ep_reward))
            print("********************************************************")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='PongNoFrameskip-v4', type=str)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--buffer_size', default=5000, type=int)
    parser.add_argument('--actor_lr', default=0.002, type=float)
    parser.add_argument('--critic_lr', default=0.001, type=float)
    parser.add_argument('--discounted_factor', default=0.90, type=float)
    parser.add_argument('--tau', default=0.01, type=float)
    parser.add_argument('--hidden_sizes', default=256, type=int)
    parser.add_argument('--num_episodes', default=1000, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--use_cuda', default=True, type=bool)
    parser.add_argument('--learning_start', default=10000, type=int)
    parser.add_argument('--update_target_freq', default=1000, type=int)
    parser.add_argument('--print_freq', default=10, type=int)

    config = parser.parse_args()

    train_your_nicest_model()
