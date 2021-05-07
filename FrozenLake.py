import gym
import numpy as np
import random
from numpy.random import default_rng
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (
    plt.rcParams["figure.figsize"][0] * 2,
    plt.rcParams["figure.figsize"][1],
)

random.seed(0)
np.random.seed(0)
rng = default_rng(0)


# 1. Load Environment and Q-table structure

env = gym.make("FrozenLake-v0")  # "FrozenLake8x8-v0"
Q = np.zeros([env.observation_space.n, env.action_space.n])
print()
print("env.observation_space:", env.observation_space)
print("env.action_space:", env.action_space)
print("env.observation_space.n:", env.observation_space.n)
print("env.action_space.n:", env.action_space.n)
input("enter any key to continue\n")


# 2. Parameters of Q-leanring
alpha = 0.628
gamma = 1
epochs = 500
return_list = []  # rewards per episode calculate

# 3. Q-learning Algorithm
for i in range(epochs):

    s = env.reset()
    env.seed(i)
    rAll = 0
    t = 0
    while True:
        env.render()

        # Choose action from Q table
        # if np.random.random_sample() < 0.05:  # 有 ε 的機率會選擇隨機 action
        #     a = env.action_space.sample()
        # else:
        #     a = np.argmax(Q[s, :])

        # a = np.argmax(
        #     Q[s, :] + np.random.randn(1, env.action_space.n) * (1.0 / (i + 1))
        # )
        a = np.argmax(
            Q[s, :] + rng.standard_normal((1, env.action_space.n)) * (1.0 / (i + 1))
        )

        # Get new state & reward from environment
        s_prime, r, done, info = env.step(a)

        # Update Q-Table with new knowledge
        Q[s, a] = Q[s, a] + alpha * (r + gamma * np.max(Q[s_prime, :]) - Q[s, a])
        rAll += r
        s = s_prime

        # if episode done
        if done:
            print(
                "Episode {} finished after {} timesteps, total rewards {}".format(
                    i, t + 1, rAll
                )
            )
            break
        t += 1

    return_list.append(rAll)
env.close()

print("Reward Sum on all episodes " + str(sum(return_list) / epochs))
print("Final Values Q-Table\n", Q)

# plot return vs eposodes
plt.plot(return_list)
plt.xlabel("episodes")
plt.ylabel("accumulated reward")
plt.title("Return vs episodes")
plt.savefig("./return-frozenlake.png")
plt.show()
