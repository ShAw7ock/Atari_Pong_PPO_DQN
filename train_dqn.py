import argparse
import os
import random
from pathlib import Path
import torch
import numpy as np
import gym
from utils.memory import ReplayBuffer
from utils.env_wrappers import *
from algos.dqn_agent import DQNAgent
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')


def train_model():
    run(config)


def run(config):
    model_dir = Path('./dqn_models')
    if not model_dir.exists():
        curr_run = 'run1'
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in model_dir.iterdir()
                         if str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exst_run_nums) + 1)
    run_dir = model_dir / curr_run
    figures_dir = run_dir / 'figures'

    os.makedirs(str(run_dir))
    os.makedirs(str(figures_dir))
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    if config.saved_model:
        e_greedy_start = 0.01
    else:
        e_greedy_start = 1

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
    # env = gym.wrappers.Monitor(env, './video/', video_callable=lambda episode_id: episode_id % 1 == 0, force=True)

    replay_buffer = ReplayBuffer(config.buffer_size)

    agent = DQNAgent(
        env.observation_space,
        env.action_space,
        replay_buffer,
        lr=config.learning_rate,
        batch_size=config.batch_size,
        gamma=config.discounted_factor
    )

    if config.saved_model:
        print(f"Loading the networks parameters - { config.saved_model } ")
        agent.policy_network.load_state_dict(torch.load(config.saved_model))

    episode_time_steps = config.e_greedy_fraction * float(config.num_steps)
    total_rewards = [0.0]
    mean_100ep_rewards = []

    state = env.reset()
    for step_i in range(config.num_steps):
        fraction = min(1.0, float(step_i) / episode_time_steps)
        episode_threshold = e_greedy_start + fraction * (config.e_greedy_end - e_greedy_start)
        if random.random() > episode_threshold:
            action = agent.step(state)
        else:
            action = env.action_space.sample()

        if config.display:
            env.render()

        next_state, reward, done, info = env.step(action)
        agent.replay_buffer.add(state, action, reward, next_state, float(done))

        state = next_state
        total_rewards[-1] += reward
        if done:
            state = env.reset()
            total_rewards.append(0.0)

        if len(replay_buffer) > config.batch_size and step_i > config.learning_start:
            agent.update()
        if step_i > config.learning_start and step_i % config.target_update_freq == 0:
            agent.update_target()

        if config.display:
            env.render()

        num_episode = len(total_rewards)

        if done and num_episode % config.print_freq == 0:
            mean_100ep_reward = round(np.mean(total_rewards[-101:-1]), 1)
            print("********************************************************")
            print("steps: {}".format(step_i))
            print("episodes: {}".format(num_episode))
            print("mean 100 episode reward: {}".format(mean_100ep_reward))
            print("********************************************************")
            np.savetxt(str(run_dir) + '/total_rewards.csv', total_rewards, delimiter=',', fmt='%1.3f')
            mean_100ep_rewards.append(mean_100ep_reward)

        if done and num_episode % config.save_model_freq == 0:
            # os.makedirs(str(run_dir / 'incremental'), exist_ok=True)
            # agent.save(str(run_dir / 'incremental' / ('model_ep%i.pt' % num_episode)))
            agent.save(str(run_dir / 'model.pt'))

    agent.save(str(run_dir / 'model.pt'))
    env.close()

    index = list(range(len(total_rewards)))
    plt.plot(index, total_rewards)
    plt.ylabel('Total Rewards')
    plt.savefig(str(figures_dir) + '/reward_curve.jpg')
    # plt.show()
    plt.close()

    index = list(range(len(mean_100ep_rewards)))
    plt.plot(index, mean_100ep_rewards)
    plt.ylabel('mean_100ep_reward')
    plt.savefig(str(figures_dir) + '/mean_100ep_reward_curve.jpg')
    # plt.show()
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Mode')
    parser.add_argument('--env', default='PongNoFrameskip-v4', type=str)
    parser.add_argument('--saved_model', default=None, type=str,
                        help='Load the model you have save before (for example: ./dqn_models/run1/model.pt)')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--buffer_size', default=5000, type=int)
    parser.add_argument('--learning_rate', default=1e-4, type=float)
    parser.add_argument('--discounted_factor', default=0.99, type=float)
    parser.add_argument('--num_steps', default=int(1e6), type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--learning_start', default=10000, type=int, help='count with inner step')
    parser.add_argument('--target_update_freq', default=1000, type=int, help='count with inner step')
    parser.add_argument('--print_freq', default=10, type=int, help='count with outer episode')
    parser.add_argument('--save_model_freq', default=10, type=int, help='count with outer episode')
    parser.add_argument('--e_greedy_end', default=0.01, type=float, help='e-greedy end threshold')
    parser.add_argument('--e_greedy_fraction', default=0.1, type=float, help='fraction of num-steps')
    parser.add_argument('--display', default=False, type=bool, help='Render the env while running')

    config = parser.parse_args()

    train_model()
