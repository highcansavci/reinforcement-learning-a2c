import torch.cuda

from synchronization.subprocess_vector_env import SubprocessVectorEnv
from atari_wrappers.monitor import make_atari, wrap_deepmind, Monitor
from model.a2c_model import A2C
from model.a2c_train import learn

import os

import gym
import argparse
import logging

MODEL_PATH = 'models'
SEED = 0


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--env', help='environment ID', default='ALE/Breakout-v5')
    parser.add_argument('-s', '--steps', help='training steps', type=int, default=int(80e6))
    parser.add_argument('--nenv', help='No. of environments', type=int, default=16)
    return parser.parse_args()


def train(env_id, num_timesteps, num_cpu, device):
    def make_env(rank):
        def _thunk():
            env_ = make_atari(env_id)
            env_.seed(SEED + rank)
            gym.logger.setLevel(logging.WARN)
            env_ = wrap_deepmind(env_)
            env_ = Monitor(env_, rank)
            return env_
        return _thunk

    env = SubprocessVectorEnv([make_env(i) for i in range(num_cpu)])
    learn(env=env, seed=SEED, total_timesteps=int(num_timesteps * 1.1), device=device)
    env.close()
    pass


def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(MODEL_PATH, exist_ok=True)
    train(args.env, args.steps, num_cpu=args.nenv, device=device)


if __name__ == "__main__":
    main()
