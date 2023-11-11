import torch
import torch.nn.functional as F
import numpy as np
from model.a2c_model import A2C, TRANSFORM
import time
import os


def set_global_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def discount_with_dones(rewards, dones, truncates, gamma):
    discounted = []
    reward_ = 0
    for reward, done, truncate in zip(rewards[::-1], dones[::-1], truncates[::-1]):
        reward_ = reward + gamma * reward_ * (1. - done) * (1. - truncate)  # fixed off by one bug
        discounted.append(reward_)
    return discounted[::-1]


class Agent:
    def __init__(self, device, obs_space, action_space, n_envs, n_steps, n_stack,
                 ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, lr=7e-4,
                 alpha=0.99, epsilon=1e-5, gamma=0.99, gae_lambda=0.9, total_timesteps=int(80e6)):
        self.max_grad_norm = max_grad_norm
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.gamma = gamma
        self.n_envs = n_envs
        self.n_steps = n_steps
        self.gae_lambda = gae_lambda
        n_height, n_width, n_channel = obs_space.shape
        self.network = A2C(n_channel * n_stack, action_space.n).to(device)
        self.optimizer = torch.optim.Adam(self.network.parameters(),
                                          lr=lr,
                                          weight_decay=alpha,
                                          eps=epsilon)

    def train(self, states, rewards, actions, values):
        self.network.train()
        states = torch.from_numpy(states.astype(np.float32)).permute(0, 3, 1, 2)
        states = states / 255.0
        hx = torch.zeros(1, self.n_envs * self.n_steps, 256, dtype=torch.float32)
        cx = torch.zeros(1, self.n_envs * self.n_steps, 256, dtype=torch.float32)
        value, logit, (hx, cx) = self.network((states, (hx, cx)))
        prob = F.softmax(logit, dim=-1)
        log_prob = F.log_softmax(logit, dim=-1)
        entropy = -(log_prob * prob).sum(1, keepdim=True)

        reward_ = value.detach()

        policy_loss = 0
        value_loss = 0
        gae = torch.zeros(1, 1)

        for i in reversed(range(len(rewards) - 1)):
            reward_ = self.gamma * reward_ + rewards[i]
            advantage = reward_ - values[i]
            value_loss = value_loss + self.vf_coef * advantage.pow(2)

            delta_t = rewards[i] + self.gamma * values[i + 1] - values[i]
            gae = gae * self.gamma * self.gae_lambda + delta_t

            policy_loss = policy_loss - log_prob * gae.detach() - self.ent_coef * entropy

        self.optimizer.zero_grad()
        (policy_loss.mean() + self.vf_coef * value_loss.mean()).backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)

        self.optimizer.step()
        return policy_loss.mean(), value_loss.mean(), entropy

    def load(self, save_name, device):
        self.network.load_state_dict(torch.load(save_name, map_location=device))

    def save(self, save_name):
        torch.save(self.network.state_dict(), save_name)

    def step(self, state):
        self.network.eval()
        with torch.no_grad():
            hx = torch.zeros(1, self.n_envs, 256, dtype=torch.float32)
            cx = torch.zeros(1, self.n_envs, 256, dtype=torch.float32)
            state_tensor = torch.from_numpy(state.astype(np.float32)).permute(0, 3, 1, 2)
            value, logit, (hx, cx) = self.network((state_tensor, (hx, cx)))
            _, pred = torch.max(logit, dim=1)
            return pred, value


class Runner:
    def __init__(self, env, agent, n_steps=5, n_stack=4, gamma=0.99):
        self.env = env
        self.agent = agent
        nh, nw, nc = env.observation_space.shape
        n_env = env.num_envs
        self.batch_ob_shape = (n_env * n_steps, nh, nw, nc * n_stack)
        self.state = np.zeros((n_env, nh, nw, nc * n_stack), dtype=np.uint8)
        self.nc = nc
        state = env.reset()
        self.update_state(state)
        self.gamma = gamma
        self.n_steps = n_steps
        self.dones = [False for _ in range(n_env)]
        self.truncates = [False for _ in range(n_env)]
        self.total_rewards = []  # store all workers' total rewards
        self.real_total_rewards = []

    def update_state(self, state):
        self.state = np.roll(self.state, shift=-self.nc, axis=3)
        self.state[:, :, :, -self.nc:] = state

    def run(self):
        mb_states, mb_rewards, mb_actions, mb_values, mb_dones, mb_truncates = [], [], [], [], [], []
        for n in range(self.n_steps):
            actions, values = self.agent.step(self.state)
            mb_states.append(np.copy(self.state))
            mb_actions.append(actions)
            mb_values.append(values)
            mb_dones.append(self.dones)
            mb_truncates.append(self.truncates)
            obs, rewards, dones, truncates, infos = self.env.step(actions)
            for done, truncate, info in zip(dones, truncates, infos):
                if done or truncate:
                    self.total_rewards.append(info['reward'])
                    if info['total_reward'] != -1:
                        self.real_total_rewards.append(info['total_reward'])
            self.dones = dones
            self.truncates = truncates
            for n, (done, truncate) in enumerate(zip(dones, truncates)):
                if done or truncate:
                    self.state[n] = self.state[n] * 0
            self.update_state(obs)
            mb_rewards.append(rewards)
        mb_dones.append(self.dones)
        mb_truncates.append(self.truncates)
        # batch of steps to batch of rollouts
        mb_states = np.asarray(mb_states, dtype=np.uint8).swapaxes(1, 0).reshape(self.batch_ob_shape)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_actions = np.asarray(mb_actions, dtype=np.int32).swapaxes(1, 0)
        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.bool_).swapaxes(1, 0)
        mb_dones = mb_dones[:, 1:]
        mb_truncates = np.asarray(mb_truncates, dtype=np.bool_).swapaxes(1, 0)
        mb_truncates = mb_truncates[:, 1:]
        last_values = self.agent.step(self.state)[1].tolist()
        # discount/bootstrap off value fn
        for n, (rewards, dones, truncates, value) in enumerate(zip(mb_rewards, mb_dones, mb_truncates, last_values)):
            rewards = rewards.tolist()
            dones = dones.tolist()
            if dones[-1] == 0:
                rewards = discount_with_dones(rewards + [value], dones + [0], truncates + [0], self.gamma)
            else:
                rewards = discount_with_dones(rewards, dones, truncates, self.gamma)
            mb_rewards[n] = np.array(rewards).flatten().tolist()
        mb_rewards = mb_rewards.flatten()
        mb_actions = mb_actions.flatten()
        mb_values = mb_values.flatten()
        return mb_states, mb_rewards, mb_actions, mb_values


def learn(device, env, seed, n_steps=5, n_stack=4, total_timesteps=int(80e6),
          vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5, lr=7e-4,
          epsilon=1e-5, alpha=0.99, gamma=0.99, log_interval=1000):
    set_global_seeds(seed)

    n_envs = env.num_envs
    env_id = env.env_id
    save_name = os.path.join('models', env_id + '.save')
    ob_space = env.observation_space
    ac_space = env.action_space
    agent = Agent(obs_space=ob_space, action_space=ac_space, n_envs=n_envs,
                  n_steps=n_steps, n_stack=n_stack,
                  ent_coef=ent_coef, vf_coef=vf_coef,
                  max_grad_norm=max_grad_norm,
                  lr=lr, alpha=alpha, epsilon=epsilon, total_timesteps=total_timesteps, device=device)
    if os.path.exists(save_name):
        agent.load(save_name, device)

    runner = Runner(env, agent, n_steps=n_steps, n_stack=n_stack, gamma=gamma)

    n_batch = n_envs * n_steps
    t_start = time.time()
    for update in range(1, total_timesteps // n_batch + 1):
        states, rewards, actions, values = runner.run()
        policy_loss, value_loss, policy_entropy = agent.train(states, rewards, actions, values)
        n_seconds = time.time() - t_start
        fps = int((update * n_batch) / n_seconds)
        if update % log_interval == 0 or update == 1:
            print(' - - - - - - - ')
            print("n_updates", update)
            print("total_timesteps", update * n_batch)
            print("fps", fps)
            print("policy_entropy", float(policy_entropy.mean().item()))
            print("value_loss", float(value_loss.item()))

            # total reward
            r = runner.total_rewards[-100:]  # get last 100
            tr = runner.real_total_rewards[-100:]
            if len(r) == 100:
                print("avg reward (last 100):", np.mean(r))
            if len(tr) == 100:
                print("avg total reward (last 100):", np.mean(tr))
                print("max (last 100):", np.max(tr))

            agent.save(save_name)

    env.close()
    agent.save(save_name)
