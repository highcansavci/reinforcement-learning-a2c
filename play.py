import argparse
import os
import numpy as np
import torch

from atari_wrappers.monitor import make_atari, wrap_deepmind, Monitor
from model.a2c_train import Agent
import imageio
import time


def get_args():
    # Get some basic command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--env', help='environment ID', default='BreakoutNoFrameskip-v4')
    return parser.parse_args()


def get_agent(env, device, n_steps=5, n_stack=1, total_timesteps=int(80e6),
              vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5, lr=7e-4,
              epsilon=1e-5, alpha=0.99):
    agent = Agent(obs_space=env.observation_space,
                  action_space=env.action_space, n_envs=1, n_steps=n_steps, n_stack=n_stack,
                  ent_coef=ent_coef, vf_coef=vf_coef, max_grad_norm=max_grad_norm,
                  lr=lr, alpha=alpha, epsilon=epsilon, total_timesteps=total_timesteps, device=device)
    return agent


def main():
    env_id = get_args().env
    env = make_atari(env_id)
    env = wrap_deepmind(env, frame_stack=True, clip_rewards=False, episode_life=True)
    env = Monitor(env)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = get_agent(env, device)

    # check for save path
    save_path = os.path.join('models', env_id + '.save')
    agent.load(save_path, device=device)

    obs = env.reset()
    renders = []
    while True:
        obs = np.expand_dims(obs.__array__(), axis=0)
        a, v = agent.step(obs)
        obs, reward, done, truncated, info = env.step(a)
        env.render()
        if done:
            print(info)
            env.reset()


if __name__ == '__main__':
    main()
