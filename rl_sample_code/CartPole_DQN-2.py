import numpy as np
import gym
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


class Net(nn.Module):
    def __init__(self, n_states, n_actions, n_hidden):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_states, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.out = nn.Linear(n_hidden, n_actions)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)  # ReLU activation
        x = self.fc2(x)
        x = F.relu(x)  # ReLU activation
        actions_value = self.out(x)
        return actions_value


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for DDPG agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(
            obs=self.obs_buf[idxs],
            obs2=self.obs2_buf[idxs],
            act=self.act_buf[idxs],
            rew=self.rew_buf[idxs],
            done=self.done_buf[idxs],
        )
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}


class DQN(object):
    def __init__(
        self,
        n_states,
        n_actions,
        n_hidden,
        batch_size,
        lr,
        epsilon,
        gamma,
        target_replace_iter,
        memory_capacity,
    ):
        self.q_net = Net(n_states, n_actions, n_hidden)

        self.target_net = deepcopy(self.q_net)

        self.replay_buffer = ReplayBuffer(
            obs_dim=n_states, act_dim=env.action_space.shape, size=memory_capacity
        )

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_func = nn.MSELoss()
        self.learn_step_counter = 0  # 讓 target network 知道什麼時候要更新

        self.n_states = n_states
        self.n_actions = n_actions
        self.n_hidden = n_hidden
        self.batch_size = batch_size
        self.lr = lr
        self.epsilon = epsilon
        self.gamma = gamma
        self.target_replace_iter = target_replace_iter
        self.memory_capacity = memory_capacity

    def choose_action(self, state):
        x = torch.unsqueeze(torch.FloatTensor(state), 0)

        # epsilon-greedy
        if np.random.uniform() < self.epsilon:  # 隨機
            action = np.random.randint(0, self.n_actions)
        else:
            actions_value = self.q_net(x)  # 以現有 q_net 得出各個 action 的分數
            action = torch.max(actions_value, 1)[1].data.numpy()[0]  # 挑選最高分的 action

        return action

    def learn(self):
        # 隨機取樣 batch_size 個 experience
        batch = self.replay_buffer.sample_batch(self.batch_size)
        o, a, r, o2, d = (
            batch["obs"],
            batch["act"],
            batch["rew"],
            batch["obs2"],
            batch["done"],
        )
        # print(o.shape)
        # print(a.shape)
        # print(r.shape)
        # print(o2.shape)
        # print(d.shape)

        # 計算現有 eval net 和 target net 得出 Q value 的落差
        q_val = (
            self.q_net(o).gather(1, a.reshape(len(a), 1).type(torch.int64)).squeeze()
        )  # 重新計算這些 experience 當下 eval net 所得出的 Q value

        with torch.no_grad():
            q_next = self.target_net(o2).detach()
        q_target = (
            r + self.gamma * q_next.max(1)[0]
        )  # .view(self.batch_size, 1)  # 計算這些 experience 當下 target net 所得出的 Q value
        # print("q_val:", q_val.shape)
        # print("q_val:", q_val)
        # print("a:", a.shape)
        # print("q_target:", q_target.shape)
        # print("q_target:", q_target)
        # print("q_next.max(1)[0]:", q_next.max(1)[0])
        # print(
        #     "q_next.max(1)[0].view(self.batch_size, 1):",
        #     q_next.max(1)[0].view(self.batch_size, 1),
        # )
        # quit()
        loss = self.loss_func(q_val, q_target)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 每隔一段時間 (target_replace_iter), 更新 target net，即複製 eval net 到 target net
        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_replace_iter == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())


env = gym.make("CartPole-v0")

# Environment parameters
n_actions = env.action_space.n
n_states = env.observation_space.shape[0]
print("n_actions", n_actions)
print("n_states", n_states)

# Hyper parameters
n_hidden = 50
batch_size = 32
lr = 0.01  # learning rate
epsilon = 0.1  # epsilon-greedy
gamma = 0.99  # reward discount factor
target_replace_iter = 100  # target network 更新間隔
memory_capacity = 10000
n_episodes = 10000

# 建立 DQN
dqn = DQN(
    n_states,
    n_actions,
    n_hidden,
    batch_size,
    lr,
    epsilon,
    gamma,
    target_replace_iter,
    memory_capacity,
)

# 學習
for i_episode in range(n_episodes):
    t = 0
    rewards = 0
    state = env.reset()
    while True:
        time.sleep(0.01)
        # env.render()

        # 選擇 action
        action = dqn.choose_action(state)
        next_state, reward, done, info = env.step(action)

        # 儲存 experience
        dqn.replay_buffer.store(state, action, reward, next_state, done)

        # 累積 reward
        rewards += reward

        # 有足夠 experience 後進行訓練
        if dqn.replay_buffer.size > (dqn.replay_buffer.max_size) * 0.1:
            dqn.learn()

        # 進入下一 state
        state = next_state

        if done:
            if i_episode % 500 == 0:
                print(
                    "Episode {} finished after {} timesteps, total rewards {}".format(
                        i_episode, t + 1, rewards
                    )
                )
            break

        t += 1

env.close()
