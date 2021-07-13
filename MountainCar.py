import gym
import time
import matplotlib.pyplot as plt


def my_intelligent_agent_fn(obs):
    if obs[1] < 0:
        action = 0
    if obs[1] >= 0:
        action = 2

    return action


env = gym.make("MountainCar-v0")

# Observation and action space
obs_space = env.observation_space
action_space = env.action_space
print("The observation space: {}".format(obs_space))
print("Upper Bound for Env Observation", env.observation_space.high)
print("Lower Bound for Env Observation", env.observation_space.low)
print()
print("The action space: {}".format(action_space))


# # Number of steps you run the agent for
# num_steps = 500

# obs = env.reset()
# cumulative_return = 0

# for step in range(num_steps):
#     # take random action, but you can also do something more intelligent
#     # action = my_intelligent_agent_fn(obs)
#     action = env.action_space.sample()

#     # apply the action
#     obs, reward, done, info = env.step(action)
#     cumulative_return += reward

#     # Render the env
#     env.render()

#     # Wait a bit before the next frame unless you want to see a crazy fast video
#     time.sleep(0.001)

#     # If the epsiode is up, then start another one
#     if done:
#         print("done")
#         print("cumulative_return", cumulative_return)
#         cumulative_return = 0
#         env.reset()

# # Close the env
# env.close()
