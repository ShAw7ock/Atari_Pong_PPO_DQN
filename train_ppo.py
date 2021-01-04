import argparse
import os
import math
from pathlib import Path
from collections import namedtuple
import torch
import numpy as np
import gym
from utils.env_wrappers import *
from algos.ppo_agent import PPOAgent
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')


def train_model():
    run(config)


def run(config):
    model_dir = Path('./ppo_models')
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

    os.makedirs(str(run_dir), exist_ok=True)
    os.makedirs(str(figures_dir))
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    Transition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward', 'done'])

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

    agent = PPOAgent(
        env.observation_space,
        env.action_space,
        hidden_sizes=config.hidden_sizes,
        critic_lr=config.critic_lr,
        actor_lr=config.actor_lr,
        gamma=config.discounted_factor,
        batch_size=config.batch_size,
        eps_clip=config.eps_clip,
        k_epochs=config.k_epochs
    )

    if config.saved_model:
        print(f"Loading the networks parameters - { config.saved_model } ")
        agent.load_params(torch.load(config.saved_model))

    total_rewards = [0.0]
    mean_100ep_rewards = []

    state = env.reset()
    for step_i in range(config.num_steps):
        if config.display:
            env.render()

        # Run step func (with target actor network)
        action, action_prob = agent.step(state)
        next_state, reward, done, _ = env.step(action)
        trans = Transition(state, action, action_prob, reward, done)
        if config.display:
            env.render()

        agent.store_transition(trans)
        state = next_state

        total_rewards[-1] += reward

        if len(agent.memory) > config.batch_size and step_i % config.update_freq == 0:
            agent.update()

        if done:
            state = env.reset()
            total_rewards.append(0.0)

        num_episode = len(total_rewards)

        if done and num_episode % config.print_freq == 0:
            mean_100ep_reward = round(np.mean(total_rewards[-101:-1]), 1)
            print("********************************************************")
            print("steps: {}".format(step_i))
            print("episodes: {}".format(num_episode))
            print("mean 100 episode reward: {}".format(mean_100ep_reward))
            print("********************************************************")
            with open(str(run_dir) + '/episodes_reward.csv', 'ab') as file:
                np.savetxt(file, total_rewards[-config.print_freq-1:-1], delimiter=',', fmt='%1.2f')
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
                        help='Load the model you have save before (for example: ./ddpg_models/run1/model.pt)')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--actor_lr', default=0.001, type=float)
    parser.add_argument('--critic_lr', default=0.003, type=float)
    parser.add_argument('--discounted_factor', default=0.99, type=float)
    parser.add_argument('--eps_clip', default=0.2, type=float)
    parser.add_argument('--k_epochs', default=4, type=int)
    parser.add_argument('--hidden_sizes', default=256, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_steps', default=int(2e6), type=int)
    parser.add_argument('--update_freq', default=2000, type=int, help='count with inner step')
    parser.add_argument('--print_freq', default=10, type=int, help='count with outer episode')
    parser.add_argument('--save_model_freq', default=10, type=int, help='count with outer episode')
    parser.add_argument('--display', default=False, type=bool, help='Render the env while running')

    config = parser.parse_args()

    train_model()
