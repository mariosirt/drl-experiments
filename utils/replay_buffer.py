"""A Replay Buffer implementation similar to the one
    in https://keras.io/examples/rl/deep_q_network_breakout/"""

import numpy as np
import tensorflow as tf


class ReplayBuffer:
    """An implmentation of a ReplayBuffer for exploiting
    the positive effects of experience replay in DQNs"""

    def __init__(self, size):
        self.action_history = []
        self.state_history = []
        self.state_next_history = []
        self.rewards_history = []
        self.done_history = []
        self.buffer_count = 0
        self.size = size

    def __len__(self):
        return len(self.action_history)

    def add(self, action, state, state_next, done, reward):
        """adds a tuple to the buffer

        Parameters
        ----------
        action: int
            an action selected by the agent
        state: np.array
            an array containing pixel values of the observed screen
        state_next: np.array
            an array containing pixel values of the next observed screen
        done: bool
            a boolean which is set to True if task was solved in episode
        reward: int
            a scalar reward as feedback for an action
        """

        if self.buffer_count >= self.size:
            self.action_history[self.buffer_count%self.size] = action
            self.state_history[self.buffer_count%self.size] = state
            self.state_next_history[self.buffer_count%self.size] = state_next
            self.done_history[self.buffer_count%self.size] = done
            self.rewards_history[self.buffer_count%self.size] = reward

        else:
            self.action_history.append(action)
            self.state_history.append(state)
            self.state_next_history.append(state_next)
            self.done_history.append(done)
            self.rewards_history.append(reward)
        self.buffer_count += 1

    def sample(self, batch_size):
        """Returns a mini_batch sampled from the ReplayBuffer

        Parameters
        ----------
        batch_size: int
            The size of the minibatch

        Returns
        ----------
        state_sample: np.array
            A random state sample of size batch_size from the ReplayBuffer
        state_next: np.array
            A random state_next sample of size batch_size from the ReplayBuffer
        rewards_sample: np.array
            A random rewards sample of size batch_size from the ReplayBuffer
        action_sample: np.array
            A random action sample of size batch_size from the ReplayBuffer
        done_sample: np.array
            A random done sample of size batch_size from the ReplayBuffer
        """
        indices = np.random.choice(
            range(len(self.done_history)), size=batch_size)

        state_sample = np.array(
            [self.state_history[i] for i in indices])
        state_next_sample = np.array(
            [self.state_next_history[i] for i in indices])
        rewards_sample = [self.rewards_history[i] for i in indices]
        action_sample = [self.action_history[i] for i in indices]
        done_sample = tf.convert_to_tensor(
            [float(self.done_history[i]) for i in indices]
        )
        return state_sample, state_next_sample, rewards_sample, action_sample, done_sample
