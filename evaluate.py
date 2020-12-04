"""
Evaluation for save the video with the best model you have trained
"""
import argparse
import os
from pathlib import Path
import time
import torch
from utils.memory import ReplayBuffer
from utils.env_wrappers import *
from algos.ddpg_agent import DDPGAgent


def evaluate_model():
    run(config)


def run(config):
    if config.saved_model is None:
        raise Exception("In Evaluation Mode, the saved model couldn't be None")

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
    ifi = 1 / config.fps

    replay_buffer = ReplayBuffer(config.buffer_size)

    agent = DDPGAgent(env.observation_space, env.action_space, replay_buffer)
    print(f"Loading the networks parameters - {config.saved_model} ")
    agent.load_params(torch.load(config.saved_model))

    episodes_count = 0
    for episode_i in range(config.num_episodes):
        state = env.reset()
        episode_reward = 0.0
        if config.display:
            env.render()
        while True:
            calc_start = time.time()
            action = agent.step_best(state)
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            episodes_count += 1
            if done:
                break
            state = next_state
            if config.display:
                calc_end = time.time()
                elapsed = calc_end - calc_start
                if elapsed < ifi:
                    time.sleep(ifi - elapsed)
                env.render()

        print("********************************************************")
        print("steps: {}".format(episodes_count))
        print("episodes: {}".format(episode_i))
        print("mean 100 episode reward: {}".format(episode_reward))
        print("********************************************************")

    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Mode')
    parser.add_argument('--env', default='PongNoFrameskip-v4', type=str)
    parser.add_argument('--saved_model', default=None, type=str, required=True,
                        help='Load the model you have saved before (for example: models/run1/model.pt)')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--buffer_size', default=5000, type=int)
    parser.add_argument('--num_episodes', default=10, type=int)
    parser.add_argument('--display', default=True, type=bool, help='Render the env while running')
    parser.add_argument('--fps', default=30, type=int)

    config = parser.parse_args()

    evaluate_model()
