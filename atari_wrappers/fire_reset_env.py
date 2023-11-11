import gym


class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == "FIRE"
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        state, _, done, truncated, _ = self.env.step(1)
        if done or truncated:
            self.env.reset(**kwargs)
        state, _, done, truncated, _ = self.env.step(2)
        if done or truncated:
            self.env.reset(**kwargs)
        return state

    def step(self, action):
        return self.env.step(action)