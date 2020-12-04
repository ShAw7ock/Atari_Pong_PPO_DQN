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


def train_model():
    run(config)


def run(config):
    model_dir = Path('/.models')
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

    agent = DDPGAgent(
        env.observation_space,
        env.action_space,
        replay_buffer,
        hidden_sizes=config.hidden_sizes,
        critic_lr=config.critic_lr,
        actor_lr=config.actor_lr,
        batch_size=config.batch_size,
        gamma=config.discounted_factor,
        tau=config.tau
    )

    if config.saved_model:
        print(f"Loading the networks parameters - { config.saved_model } ")
        agent.load_params(torch.load(config.saved_model))

    total_rewards = []
    episodes_count = 0

    for episode_i in range(config.num_episodes):
        episode_reward = 0.0
        state = env.reset()

        if config.display:
            env.render()

        while True:
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

            if config.display:
                env.render()

        if episode_i % config.print_freq == 0:
            mean_100ep_reward = np.mean(total_rewards[-101:-1])
            print("********************************************************")
            print("steps: {}".format(episodes_count))
            print("episodes: {}".format(episode_i))
            print("mean 100 episode reward: {}".format(mean_100ep_reward))
            print("********************************************************")

        if episode_i % config.save_model_freq == 0:
            agent.save(str(run_dir / ('model_ep%i.pt' % episode_i)))
            agent.save(str(run_dir / 'model.pt'))

    agent.save(str(run_dir / 'model.pt'))
    env.close()

    index = list(range(len(total_rewards)))
    plt.plot(index, total_rewards)
    plt.ylabel('Total Rewards')
    plt.savefig(str(figures_dir) + '/reward_curve.jpg')
    # plt.show()
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Mode')
    parser.add_argument('--env', default='PongNoFrameskip-v4', type=str)
    parser.add_argument('--saved_model', default=None, type=str,
                        help='If you wanna load the model you have save before (for example: models/run1/model.pt)')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--buffer_size', default=5000, type=int)
    parser.add_argument('--actor_lr', default=0.002, type=float)
    parser.add_argument('--critic_lr', default=0.001, type=float)
    parser.add_argument('--discounted_factor', default=0.90, type=float)
    parser.add_argument('--tau', default=0.01, type=float)
    parser.add_argument('--hidden_sizes', default=256, type=int)
    parser.add_argument('--num_episodes', default=1000, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--learning_start', default=10000, type=int)
    parser.add_argument('--update_target_freq', default=1000, type=int)
    parser.add_argument('--print_freq', default=10, type=int)
    parser.add_argument('--save_model_freq', default=100, type=int)
    parser.add_argument('--display', default=False, type=bool, help='Render the env while running')

    config = parser.parse_args()

    train_model()
