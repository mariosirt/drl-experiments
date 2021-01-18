'''https://keras.io/examples/rl/deep_q_network_breakout/'''
import time
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np


from replay_buffer import ReplayBuffer
from models import Models


class Agent():
    """A class representation of an agent using a DQN to solve different atari learning tasks."""

    def __init__(self, env, model, model_target, epsilon):
        self.env = env
        self.num_actions = self.env.action_space.n
        self.model = model
        self.model_target = model_target
        self.epsilon = epsilon
        self.logger = tf.summary.create_file_writer("logs")
        self.logger.set_as_default()
        self.frame_count = 0

    def train(self,
              max_memory_length,
              batch_size,
              gamma,
              learning_rate,
              max_episode_steps,
              max_episodes,
              max_frames,
              epsilon_random_frames,
              epsilon_greedy_frames,
              update_after_actions,
              update_target_network,
              save_model_steps,
              save_model_path,
              ddqn
              ):
        """Method for learning a dqn or double dqn

        Parameters
        ----------
        max_memory_length: int
            Max memory size for replay buffer

        batch_size: int
            batch size for sampling experience replay data before performing a gradient update

        gamma: float
            discount factor gamma for controlling the importance of future rewards

        learning_rate: float
            learning_rate of optimizer

        max_episode_steps: int
            maximum number of steps per episode

        max_episodes: int
            maximum number of episode per training


        epsilon_random_frames: float
            Number of first n random frames before acting greedily

        epsilon_greedy_frames: float
            Number of greedy frames after random frames

        update_after_actions: int
            Number for controlling after how many actions an update
            should occure ( like in original dqn paper )

        update_target_network: int
            Number of passed frames after which one has to update the target network

        save_model_steps: int
            Number of steps between saving a model

        save_model_path: String
            Path for saving a model

        ddqn: bool
            boolean for whether to train with or without double dqn for reducing overestimation
        """

        optimizer, running_reward, episode_reward_history,\
            memory, loss_function = initialize_training(
                learning_rate, max_memory_length)

        for episode_count in range(0, max_episodes):
            state = np.array(self.env.reset())
            episode_reward = 0

            for _ in range(1, max_episode_steps):
                self.frame_count += 1

                random = True
                if self.frame_count < epsilon_random_frames or self.epsilon > np.random.rand(1)[0]:
                    action = np.random.choice(self.num_actions)
                else:
                    action = self.select_action_greedily(state)
                    random = False

                self.decay_epsilon(epsilon_greedy_frames)
                state_next, reward, done, _ = self.env.step(action)

                episode_reward += reward
                tf.summary.scalar("random", data=random, step=self.frame_count)
                tf.summary.scalar("reward", data=reward, step=self.frame_count)
                tf.summary.scalar("action", data=action, step=self.frame_count)

                memory.add(action, state, state_next, done, reward)

                state = np.array(state_next)

                if self.frame_count % update_after_actions == 0 and memory.__len__() > batch_size:
                    self.train_step(
                        memory, batch_size, ddqn, gamma, loss_function, optimizer)

                if self.frame_count % update_target_network == 0:
                    self.model_target.set_weights(self.model.get_weights())

                if self.frame_count % save_model_steps == 0:
                    Models.save_model(self.model, save_model_path)

                memory.bufferfy()

                if done:
                    break

            tf.summary.scalar("episode_reward",
                              data=episode_reward, step=episode_count)
            episode_reward_history.append(episode_reward)
            if len(episode_reward_history) > 100:
                del episode_reward_history[:1]
            running_reward = np.mean(episode_reward_history)

            if running_reward > 40:  # Condition to consider the task solved
                print("Solved at episode {}!".format(episode_count))
                break
            if self.frame_count >= max_frames:
                print("Max frames completed")
                break

    def train_step(self, memory, batch_size, ddqn, gamma, loss_function, optimizer):
        """Method for train step and updating weights of model

        Parameters
        ----------
        memory: ReplayBuffer
            the ReplayBuffer used for sampling

        batch_size: int
            Sampling Batch Size
        ddqn: bool
            variable for deciding whether tu use ddqn or vanilla dqn
        gamma: float
            discount factor
        loss_function: keras.losses.Huber
            loss function for updating gradients
        optimizer: keras.optimizers.Adam
            optimizer
        """
        state_sample, state_next_sample, rewards_sample, action_sample, \
            done_sample = memory.sample(batch_size)

        if ddqn:
            qs_next_model = self.model(state_next_sample)

            argmax_qs_next = tf.argmax(qs_next_model, axis=-1)
            next_action_mask = tf.one_hot(argmax_qs_next,
                                          self.num_actions, on_value=1., off_value=0.)

            qs_next_target = self.model_target(state_next_sample)

            tf.summary.scalar("Q Value estimates", data=np.mean(
                qs_next_target), step=self.frame_count)
            masked_qs_next = tf.reduce_sum(tf.multiply(next_action_mask,
                                                       qs_next_target), axis=-1)

            target = rewards_sample + \
                (1. - done_sample) * gamma * masked_qs_next

        else:

            qs_next = self.model_target.predict(
                state_next_sample)

            tf.summary.scalar("Q Value estimates", data=np.mean(
                qs_next), step=self.frame_count)

            max_qs_next = tf.reduce_max(qs_next, axis=-1)

            target = rewards_sample + (1.-done_sample) * gamma * \
                max_qs_next

        with tf.GradientTape() as tape:
            qs_curr = self.model(state_sample)
            masks = tf.one_hot(
                action_sample, self.num_actions, on_value=1., off_value=0.)
            masked_qs = tf.multiply(qs_curr, masks)

            masked_qs = tf.reduce_sum(masked_qs, axis=-1)
            loss = loss_function(target, masked_qs)

        grads = tape.gradient(
            loss, self.model.trainable_variables)
        optimizer.apply_gradients(
            zip(grads, self.model.trainable_variables))

    def test(self, max_episodes, max_episode_steps):
        """Method for testing a dqn model

        Parameters
        ----------
        max_episode_steps: int
            maximum number of steps per episode

        max_episodes: int
            maximum number of episode per training
        """

        for episode_count in range(0, max_episodes):
            state = np.array(self.env.reset())
            episode_reward = 0

            for _ in range(1, max_episode_steps):
                self.env.render()
                time.sleep(0.05)
                self.frame_count += 1

                random = True
                if np.random.rand(1)[0] < 0.05:
                    action = np.random.choice(self.num_actions)
                else:
                    action = self.select_action_greedily(state)
                    random = False

                tf.summary.scalar("random", data=random, step=self.frame_count)

                state_next, reward, done, _ = self.env.step(action)
                state_next = np.array(state_next)

                episode_reward += reward
                tf.summary.scalar("reward", data=reward, step=self.frame_count)
                tf.summary.scalar("action", data=action, step=self.frame_count)

                state = state_next

                if done:
                    time.sleep(3)
                    break

            tf.summary.scalar("episode_reward",
                              data=episode_reward, step=episode_count)

    def select_action_greedily(self, state):
        """Return an action greedily based on current dqn predictions

        Parameters
        ----------
        state: np.array
            The current state

        Returns
        -------
        action: int
            one of the possible actions but selected greedily based on current dqn preds
        """
        state_tensor = tf.convert_to_tensor(state)
        state_tensor = tf.expand_dims(state_tensor, 0)
        action_probs = self.model(state_tensor, training=False)

        action = tf.argmax(action_probs[0].numpy())

        return action

    def decay_epsilon(self, epsilon_greedy_frames):
        """Decays epsilon based on number of greedy frames"""
        epsilon_interval = 1.0 - 0.1  # epsilon_max - epsilon_min
        self.epsilon -= epsilon_interval/epsilon_greedy_frames
        self.epsilon = max(self.epsilon, 0.1)  # epsilon_min is 0.1


def initialize_training(learning_rate, max_memory_length):
    """Initialize training variables

    Parameters
    ----------
    learning_rate: float
        The learning rate for the optimizer
    max_memory_length: int
        The maximal size of the replay buffer

    Returns
    -------
    optimizer: keras.otptimizers.Adam
        an initialised optimizer
    running_reward: int
        the initial running reward
    memory: ReplayBuffer
        the initial replay buffer memory
    episode_reward_history: list
        the initial history of episode rewards for solving the task
    loss_function: keras.losses.Huber
        the initialised loss function
    """

    optimizer = keras.optimizers.Adam(
        learning_rate=learning_rate, clipnorm=1.0)

    running_reward = 0

    memory = ReplayBuffer(max_memory_length)
    episode_reward_history = []

    loss_function = keras.losses.Huber()

    return optimizer, running_reward, episode_reward_history, memory, loss_function
