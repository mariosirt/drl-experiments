"""mostly based on neat implementation of https://github.com/openai/spinningup/blob/master/spinup/examples/pytorch/pg_math/1_simple_pg.py"""

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import tensorflow as tf
import gym
from scipy.stats import norm
import numpy as np
import math
from tensorboard.compat.proto import summary_pb2


def model(features, activations=[nn.ReLU, nn.Softmax]):
    """A simple representation of a model for vanilla gradient (simple MLP)

    Params:
    --------
    features: a list of numbers of features per layer
    activations: the used activation functions, which per default are all same being ReLU

    returns a sequential MLP
    """
    layers = []
    for i in range(len(features)-1):
        if len(activations) <= 2:
            activ = activations[0] if i < len(features)-2 else activations[-1]
        else:
            assert len(activations) == len(features)
            activ = activations[i]
        layers.append(nn.Linear(features[i], features[i+1], activ()))
    return nn.Sequential(*layers)


def kl_divergence(act_old, act_curr):
    mu1, std1 = norm.fit(act_old)
    mu2, std2 = norm.fit(act_curr)

    range_data = np.arange(-10, 10, 0.001)
    p = norm.pdf(range_data, mu1, std1)
    q = norm.pdf(range_data, mu2, std2)
    q[q == 0] = 1
    res = np.sum(np.where(p != 0, p * np.log(p/q), 0))
    return res if not np.isnan(res) else 0


class Agent(object):

    def __init__(self, env_name="CartPole-v0", mlp_features=[32]):
        self.env = gym.make(env_name)
        self.obs_dim = self.env.observation_space.shape[0]
        self.act_num = self.env.action_space.n
        self.mlp = model([self.obs_dim]+mlp_features+[self.act_num])
        self.logger = tf.summary.create_file_writer("logs")
        self.logger.set_as_default()

    def loss_func(self, obs, acts, weights):
        logp = self.get_policy(obs, -1, -1).log_prob(acts)
        return -(logp * weights).mean()

    def get_policy(self, obs, frame_count, ep_count):
        return Categorical(logits=self.mlp(obs))

    def get_action(self, obs, frame_count, ep_count):
        print(self.mlp(obs).detach().numpy(), frame_count, ep_count)
        tf.summary.histogram(name="episode{}".format(ep_count), data=(
            self.mlp(obs).detach().numpy()), step=frame_count, buckets=1)
        return self.get_policy(obs, frame_count, ep_count).sample().item()

    def train_rollouts(self, batch_size, ep_count):
        obs_batch = []
        acts_batch = []
        weights_batch = []
        returns_batch = []
        ep_lens_batch = []

        obs = self.env.reset()
        done = False
        ep_rewards = []

        frame_count = 0
        frame_count_cum = 0
        old_frame_count = 0
        while True:
            #frame_img = self.env.render(mode="rgb_array")
            #tf.summary.image(name="episode{}".format(ep_count), data=tf.expand_dims(frame_img,0), step=frame_count)
            obs_batch.append(obs.copy())
            action = self.get_action(torch.as_tensor(
                obs, dtype=torch.float32), frame_count, ep_count)
            obs, reward, done, _ = self.env.step(action)

            acts_batch.append(action)
            ep_rewards.append(reward)

            if done:
                ep_return, ep_len = sum(ep_rewards), len(ep_rewards)
                returns_batch.append(ep_return)
                ep_lens_batch.append(ep_len)
                tf.summary.scalar(name="episode_reward",
                                  data=ep_return, step=ep_count)
                tf.summary.scalar(name="episode_length",
                                  data=ep_len, step=ep_count)

                weights_batch += ([ep_return]*ep_len)

                obs, done, ep_rewards = self.env.reset(), False, []

                if ep_count >= 1:
                    # print(len(acts_batch))
                    #print(old_frame_count, frame_count)
                    act_old = acts_batch[old_frame_count:-frame_count]
                    act_curr = acts_batch[-frame_count:]
                    #print(len(act_old), len(act_curr))
                    #distance = stats.wasserstein_distance(act_old, act_curr)

                    distance = kl_divergence(act_old, act_curr)
                    # print(distance)
                    tf.summary.scalar(name="distance_actions",
                                      data=distance, step=ep_count)
                    old_frame_count = frame_count_cum - frame_count

                ep_count += 1
                frame_count = 0
                if len(obs_batch) > batch_size:
                    break
            frame_count += 1
            frame_count_cum += 1

        return obs_batch, acts_batch, weights_batch, returns_batch, ep_lens_batch, ep_count

    def train(self, lr=1e-2, epochs=50, batch_size=5000):
        optimizer = Adam(self.mlp.parameters(), lr)

        ep_count = 0
        for i in range(1, epochs):
            # print(ep_count)
            obs_batch, acts_batch, weights_batch, returns_batch, ep_lens_batch, ep_count = self.train_rollouts(
                batch_size, ep_count)
            optimizer.zero_grad()
            loss_batch = self.loss_func(obs=torch.as_tensor(obs_batch, dtype=torch.float32),
                                        acts=torch.as_tensor(
                                            acts_batch, dtype=torch.float32),
                                        weights=torch.as_tensor(weights_batch, dtype=torch.float32))
            loss_batch.backward()
            optimizer.step()
            print("Training epoch {} with loss {} and return {} and episode length {} ".format(
                i, loss_batch, np.mean(returns_batch), np.mean(ep_lens_batch)))
