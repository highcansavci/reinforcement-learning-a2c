from collections import deque
import gym
import numpy as np
from gym import spaces


class FrameStack(gym.Wrapper):
    def __init__(self, env, num_stack_frame):
        gym.Wrapper.__init__(self, env)
        self.num_stack_frame = num_stack_frame
        self.frames = deque([], maxlen=num_stack_frame)
        shape_ = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(shape_[0], shape_[1], shape_[2] * self.num_stack_frame))

    def _get_state(self):
        assert len(self.frames) == self.num_stack_frame
        return LazyFrames(list(self.frames))

    def reset(self, **kwargs):
        state, _ = self.env.reset()
        for _ in range(self.num_stack_frame):
            self.frames.append(state)
        return self._get_state()

    def step(self, action):
        state, reward, done, truncated, info = self.env.step(action)
        self.frames.append(state)
        return self._get_state(), reward, done, truncated, info


class LazyFrames:
    def __init__(self, frames):
        self._frames = frames

    def __array__(self, dtype=None):
        out = np.concatenate(self._frames, axis=2)
        if dtype is not None:
            out = out.astype(dtype)
        return out