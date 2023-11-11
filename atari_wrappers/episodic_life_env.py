import gym


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        state, reward, done, truncated, info = self.env.step(action)
        self.was_real_done = done or truncated
        lives = self.env.unwrapped.ale.lives()
        if 0 < lives < self.lives:
            done = True
        self.lives = lives
        return state, reward, done, truncated, info

    def reset(self, **kwargs):
        if self.was_real_done:
            state = self.env.reset(**kwargs)
        else:
            state, _, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return state
