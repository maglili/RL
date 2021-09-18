import gym
from copy import deepcopy
import numpy as np
import random
import torch
import torch.nn as nn
from torch.optim import Adam
import time


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


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
        print(combined_shape(size, act_dim))
        quit()

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


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


class MLPQFunction(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def forward(self, obs):
        q = self.q(torch.cat([obs], dim=-1))
        return torch.squeeze(q, -1)  # Critical to ensure q has right shape.


class MLPActorCritic(nn.Module):
    def __init__(
        self,
        observation_space,
        action_space,
        hidden_sizes=(256, 256),
        activation=nn.ReLU,
    ):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.n

        # build policy and value functions
        self.q = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        print("Q network:\n", self.q)

    def act(self, obs):
        with torch.no_grad():
            actions_value = self.q(obs)
            return torch.max(actions_value, 0)[1].data.numpy()


def dqn(
    env_fn,
    actor_critic=MLPActorCritic,
    replay_size=int(1e6),
    ac_kwargs=dict(),
    gamma=0.99,
    q_lr=1e-3,
    polyak=0.995,
    epsilon=0.1,
    num_test_episodes=10,
    max_ep_len=1000,
    steps_per_epoch=4000,
    epochs=100,
    start_steps=10000,
    update_after=1000,
    update_every=50,
    batch_size=100,
    save_freq=1,
):
    seed_val = 0
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(0)

    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape
    print("obs_dim:", obs_dim)
    print("act_dim:", act_dim)
    print(env.action_space.n)

    # Create actor-critic module and target networks
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    ac_targ = deepcopy(ac)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False

    # Set up experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Count variables
    var_counts = count_vars(ac.q)
    print("model parameters:", var_counts)

    # Set up function for computing DDPG Q-loss
    def compute_loss_q(data):
        o, a, r, o2, d = (
            data["obs"],
            data["act"],
            data["rew"],
            data["obs2"],
            data["done"],
        )

        q = ac.q(o).gather(1, a.reshape(len(a), 1).type(torch.int64))
        # Bellman backup for Q function
        with torch.no_grad():
            q_pi_targ = ac_targ.q(o2).max(1)[0].reshape(len(o2), 1)
            backup = r + gamma * (1 - d) * q_pi_targ

        # MSE loss against Bellman backup
        loss_q = ((q - backup) ** 2).mean()
        # Useful info for logging
        loss_info = dict(QVals=q.detach().numpy())
        return loss_q, loss_info

    # Set up optimizers for policy and q-function
    q_optimizer = Adam(ac.q.parameters(), lr=q_lr)

    def update(data):
        # First run one gradient descent step for Q.
        q_optimizer.zero_grad()
        loss_q, loss_info = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()

        # Unfreeze Q-network so you can optimize it at next DDPG step.
        for p in ac.q.parameters():
            p.requires_grad = True

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

    def get_action(o):
        # epsilon-greedy
        if np.random.uniform() < epsilon:
            action = np.random.randint(0, env.action_space.n)
        else:
            action = ac.act(torch.as_tensor(o, dtype=torch.float32))
        return action

    def test_agent():
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            while not (d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                o, r, d, _ = test_env.step(get_action(o))
                ep_ret += r
                ep_len += 1
            print("test-" + str(j), "ep_ret: {}, ep_len: {}".format(ep_ret, ep_len))
        print("=" * 40)

    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):

        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards,
        # use the learned policy (with some noise, via act_noise).
        if t > start_steps:
            a = get_action(o)
        else:
            a = env.action_space.sample()

        # Step the env
        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len == max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d)

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        o = o2

        # End of trajectory handling
        if d or (ep_len == max_ep_len):
            o, ep_ret, ep_len = env.reset(), 0, 0

        # Update handling
        if t >= update_after and t % update_every == 0:
            for _ in range(update_every):
                batch = replay_buffer.sample_batch(batch_size)
                update(data=batch)

        # End of epoch handling
        if (t + 1) % steps_per_epoch == 0:
            epoch = (t + 1) // steps_per_epoch

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs):
                pass

            # Test the performance of the deterministic version of the agent.
            test_agent()


if __name__ == "__main__":
    env_name = "CartPole-v1"
    hid = 256
    l = 2
    gamma = 0.99
    replay_size = int(1e6)
    q_lr = 1e-2
    polyak = 0.1
    dqn(
        lambda: gym.make(env_name),
        actor_critic=MLPActorCritic,
        replay_size=replay_size,
        ac_kwargs=dict(hidden_sizes=[hid] * l),
        gamma=gamma,
        q_lr=q_lr,
        polyak=polyak,
    )
