import gym
import numpy as np
import random

random.seed(0)
np.random.seed(0)


# 1. Load Environment and Q-table structure

# env = gym.make("FrozenLake8x8-v0")
env = gym.make("FrozenLake-v0")
Q = np.zeros([env.observation_space.n, env.action_space.n])


# 2. Parameters of Q-leanring
alpha = 0.628
gamma = 0.9
epochs = 500
return_list = []  # rewards per episode calculate

# 3. Q-learning Algorithm
for i in range(epochs):

    s = env.reset()
    rAll = 0

    for t in range(250):
        env.render()

        # Choose action from Q table
        # if np.random.random_sample() < 0.05:  # 有 ε 的機率會選擇隨機 action
        #     a = env.action_space.sample()
        # else:
        #     a = np.argmax(Q[s, :])

        a = np.argmax(
            Q[s, :] + np.random.randn(1, env.action_space.n) * (1.0 / (i + 1))
        )

        # Get new state & reward from environment
        s1, r, done, info = env.step(a)
        # Update Q-Table with new knowledge
        Q[s, a] = Q[s, a] + alpha * (r + gamma * np.max(Q[s1, :]) - Q[s, a])
        rAll += r
        s = s1
        if done:
            print(
                "Episode finished after {} timesteps, total rewards {}".format(
                    t + 1, rAll
                )
            )
            break
    return_list.append(rAll)
    env.render()
env.close()
print("Reward Sum on all episodes " + str(sum(return_list) / epochs))
print("Final Values Q-Table\n", Q)
