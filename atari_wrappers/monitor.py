import gym

from atari_wrappers.clip_reward_env import ClipRewardEnv
from atari_wrappers.episodic_life_env import EpisodicLifeEnv
from atari_wrappers.fire_reset_env import FireResetEnv
from atari_wrappers.frame_stack_wrapper import FrameStack
from atari_wrappers.max_and_skip_env import MaxAndSkipEnv
from atari_wrappers.noop_reset_env import NoopResetEnv
from atari_wrappers.warp_frame_wrapper import WarpFrame


def make_atari(env_id):
    env = gym.make(env_id)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    return env


def wrap_deepmind(env, episode_life=True, clip_rewards=True, frame_stack=False):
    if episode_life:
        env = EpisodicLifeEnv(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    if frame_stack:
        env = FrameStack(env, 4)
    return env

class Monitor(gym.Wrapper):
    def __init__(self, env, rank=0):
        gym.Wrapper.__init__(self, env)
        self.rank = rank
        self.rewards = []
        self.total_reward = []
        self.summaries_dict = {'reward': 0,
                               'episode_length': 0,
                               'total_reward': 0,
                               'total_episode_length': 0}
        env = self.env
        while True:
            if hasattr(env, 'was_real_done'):
                self.episodic_env = env
            if not hasattr(env, 'env'):
                break
            env = env.env

    def reset(self):
        self.summaries_dict['reward'] = -1
        self.summaries_dict['episode_length'] = -1
        self.summaries_dict['total_reward'] = -1
        self.summaries_dict['total_episode_length'] = -1
        self.rewards = []
        env = self.env
        if self.episodic_env.was_real_done:
            self.summaries_dict['total_reward'] = -1
            self.summaries_dict['total_episode_length'] = -1
            self.total_reward = []
        return self.env.reset()

    def step(self, action):
        state, reward, done, truncated, info = self.env.step(action)
        self.rewards.append(reward)
        self.total_reward.append(reward)
        if done or truncated:
            self.summaries_dict['reward'] = sum(self.rewards)
            self.summaries_dict['episode_length'] = len(self.rewards)

            if self.episodic_env.was_real_done:
                self.summaries_dict['total_reward'] = sum(self.total_reward)
                self.summaries_dict['total_episode_length'] = len(self.total_reward)

        info = self.summaries_dict.copy()
        return state, reward, done, truncated, info
