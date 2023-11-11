import gym
import numpy as np


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        gym.Wrapper.__init__(self, env)
        self._state_buffer = np.zeros((2,) + env.observation_space.shape, dtype='uint8')
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            state, reward, done, truncated, info = self.env.step(action)
            if i == self._skip - 2:
                self._state_buffer[0] = state
            if i == self._skip - 1:
                self._state_buffer[1] = state
            total_reward += reward
            if done or truncated:
                break

        max_frame = self._state_buffer.max(axis=0)
        return max_frame, total_reward, done, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

