import gym
import math
import numpy as np
import random
from numpy.random import default_rng
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (
    plt.rcParams["figure.figsize"][0] * 2,
    plt.rcParams["figure.figsize"][1],
)

seed_val = 0
random.seed(seed_val)
np.random.seed(seed_val)
rng = default_rng(seed_val)  # random generator


# 1. Load Environment and Q-table structure

env = gym.make("FrozenLake-v0")
Q = np.zeros([env.observation_space.n, env.action_space.n])
print()
print("env.observation_space:", env.observation_space)
print("env.action_space:", env.action_space)
print("env.observation_space.n:", env.observation_space.n)
print("env.action_space.n:", env.action_space.n)
print("initial Q table:\n", Q)
print("-" * 10)
input("enter any key to continue\n")


# 2. Parameters of Q-leanring
get_lr = lambda i: max(
    0.01, min(0.7, 1.0 - math.log10((i + 1) / 100))
)  # learning rate; 隨時間遞減
gamma = 1
epochs = 500
return_list = []  # rewards per episode calculate


# 3. Q-learning Algorithm
for i in range(epochs):
    alpha = get_lr(i)  # 0.628
    s = env.reset()
    env.seed(i)
    rAll = 0
    t = 0
    while True:
        env.render()
        print("-" * 10)

        # Choose action from Q table
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
            env.render()
            print("-" * 10)
            print("*" * 50)
            print(
                "Episode {} finished after {} timesteps, total rewards {}".format(
                    i, t + 1, rAll
                )
            )
            print("*" * 40)
            break

        # incresed timestep
        t += 1

    return_list.append(rAll)

env.close()

print()
print("Mean return on all episodes " + str(sum(return_list) / epochs))
print("Final Values Q-Table\n", Q)

# plot return vs eposodes
plt.plot(return_list)
plt.xlabel("episodes")
plt.ylabel("accumulated reward")
plt.title("Return vs episodes")
plt.savefig("./return-FrozenLake.png")
plt.show()
