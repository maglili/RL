import gym
import numpy as np
from numpy.random import default_rng
import math
import random
import matplotlib.pyplot as plt

seed_val = 0
random.seed(seed_val)
np.random.seed(seed_val)
rng = default_rng(seed_val)  # random generator


def choose_action(state, q_table, action_space, epsilon):
    """
    behavior policy: epsilon-greedy policy.
    """
    if rng.random() < epsilon:
        while True:
            amax = np.argmax(q_table[state])
            a = action_space.sample()
            if a != amax:
                break
        return action_space.sample()
    else:
        return np.argmax(q_table[state])


def get_state(observation, n_buckets, state_bounds):
    """
    mapping continue state to discrete state.
    """
    state = [0] * len(observation)
    for i, s in enumerate(observation):  # 每個 feature 有不同的分配
        l, u = state_bounds[i][0], state_bounds[i][1]  # 每個 feature 值的範圍上下限
        if s <= l:  # 低於下限，分配為 0
            state[i] = 0
        elif s >= u:  # 高於上限，分配為最大值
            state[i] = n_buckets[i] - 1
        else:  # 範圍內，依比例分配
            state[i] = int(((s - l) / (u - l)) * n_buckets[i])

    return tuple(state)


env = gym.make("CartPole-v0")
env.action_space.np_random.seed(seed_val)

# 準備 Q table
## Environment 中各個 feature 的 bucket 分配數量
## 1 代表任何值皆表同一 state，也就是這個 feature 其實不重要
n_buckets = (1, 1, 6, 3)

## Action 數量
n_actions = env.action_space.n

## State 範圍
state_bounds = list(zip(env.observation_space.low, env.observation_space.high))
state_bounds[1] = [-0.5, 0.5]
state_bounds[3] = [-math.radians(50), math.radians(50)]

## Q table，每個 state-action pair 存一值
q_table = np.zeros(n_buckets + (n_actions,))

# hyperpamaters
# epsilon-greedy; 隨時間遞減
get_epsilon = lambda i: max(0.01, min(1, 1.0 - math.log10((i + 1) / 25)))
# learning rate; 隨時間遞減
get_lr = lambda i: max(0.01, min(0.5, 1.0 - math.log10((i + 1) / 25)))
# discount factor
gamma = 1
# save accumulated reward
return_list = []
epochs = 200

# Q-learning
for i_episode in range(epochs):
    epsilon = get_epsilon(i_episode)
    lr = get_lr(i_episode)
    env.seed(i_episode)
    observation = env.reset()
    rewards = 0
    state = get_state(observation, n_buckets, state_bounds)  # 將連續值轉成離散
    t = 0
    while True:
        env.render()

        action = choose_action(state, q_table, env.action_space, epsilon)
        observation, reward, done, info = env.step(action)

        rewards += reward
        next_state = get_state(observation, n_buckets, state_bounds)

        # 更新 Q table
        q_next_max = np.amax(q_table[next_state])
        q_table[state + (action,)] += lr * (
            reward + gamma * q_next_max - q_table[state + (action,)]
        )

        state = next_state

        if done:
            env.render()
            print(
                "Episode {} finished after {} timesteps, total rewards {}. epsilon {}, lr {}".format(
                    i_episode, t + 1, rewards, epsilon, lr
                )
            )
            return_list.append(rewards)
            break
        t += 1
env.close()

# plot return vs eposodes
plt.plot(return_list)
plt.xlabel("episodes")
plt.ylabel("accumulated reward")
plt.title("Return vs episodes")
plt.savefig("./return-CartPole.png", bbox_inches="tight")
plt.show()
