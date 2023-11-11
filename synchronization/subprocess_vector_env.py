import numpy as np
from multiprocessing import Process, Pipe
import cloudpickle


def worker(remote, parent_remote, env_function_wrapper):
    parent_remote.close()
    env = env_function_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            state, reward, done, truncated, info = env.step(data)
            if done or truncated:
                state = env.reset()
            remote.send((state, reward, done, truncated, info))
        elif cmd == "reset":
            state = env.reset()
            remote.send(state)
        elif cmd == "reset_task":
            state = env.reset_task()
            remote.send(state)
        elif cmd == "close":
            remote.close()
            break
        elif cmd == "get_spaces":
            remote.send((env.action_space, env.observation_space))
        elif cmd == "get_id":
            remote.send(env.spec.id)
        else:
            raise NotImplementedError


class CloudpickleWrapper:
    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        return cloudpickle.dumps(self.x)

    def __setstate__(self, state):
        import pickle
        self.x = pickle.loads(state)


class SubprocessVectorEnv:
    def __init__(self, env_functions):
        self.closed = False
        n_envs = len(env_functions)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(n_envs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_function)))
                   for (work_remote, remote, env_function) in zip(self.work_remotes, self.remotes, env_functions)]
        for p in self.ps:
            p.daemon = True
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        self.action_space, self.observation_space = self.remotes[0].recv()

        self.remotes[0].send(('get_id', None))
        self.env_id = self.remotes[0].recv()

    def step(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        results = [remote.recv() for remote in self.remotes]
        states, rewards, dones, truncateds, infos = zip(*results)
        return np.stack(states), np.stack(rewards), np.stack(dones), np.stack(truncateds), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return

        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()

        self.closed = True

    @property
    def num_envs(self):
        return len(self.remotes)

