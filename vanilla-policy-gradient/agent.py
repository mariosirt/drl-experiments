"""mostly based on neat implementation of https://github.com/openai/spinningup/blob/master/spinup/examples/pytorch/pg_math/1_simple_pg.py"""

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import gym
import numpy as np

def model(features, activations=[nn.ReLU, nn.Identity]):
    """A simple representation of a model for vanilla gradient (simple MLP)
    
    Params:
    --------
    features: a list of numbers of features per layer
    activations: the used activation functions, which per default are all same being ReLU
    
    returns a sequential MLP
    """
    layers= []
    for i in range(len(features)-1):
        if len(activations)<=2:
            activ = activations[0] if i< len(features)-2 else activations[-1]
        else:
            assert len(activations) == len(features)
            activ = activations[i]
        layers.append(nn.Linear(features[i], features[i+1],activ()))
    return nn.Sequential(*layers)



class Agent(object):


    def __init__(self, env_name="CartPole-v0", mlp_features =[32]):
        self.env = gym.make(env_name)
        self.obs_dim = self.env.observation_space.shape[0]
        self.act_num = self.env.action_space.n
        self.mlp = model([self.obs_dim]+mlp_features+[self.act_num])


    def loss_func(self, obs, acts, weights):
        logp = self.get_policy(obs).log_prob(acts)
        return -(logp * weights).mean()

    def get_policy(self,obs):
        return Categorical(logits=self.mlp(obs))
    
    def get_action(self, obs):
        return self.get_policy(obs).sample().item()
    
    def train_rollouts(self, batch_size):
        obs_batch = []
        acts_batch = []
        weights_batch = []
        returns_batch = []
        ep_lens_batch = []

        obs= self.env.reset()
        done = False
        ep_rewards = []

        while True:

            obs_batch.append(obs.copy())
            action = self.get_action(torch.as_tensor(obs, dtype= torch.float32))
            obs, reward, done, _ = self.env.step(action)

            acts_batch.append(action)
            ep_rewards.append(reward)

            if done:
                ep_return, ep_len = sum(ep_rewards), len(ep_rewards)
                returns_batch.append(ep_return)
                ep_lens_batch.append(ep_len)

                weights_batch += ([ep_return]*ep_len)

                obs, done, ep_rewards = self.env.reset(), False, []

                if len(obs_batch) > batch_size:
                    break
        return obs_batch, acts_batch, weights_batch, returns_batch, ep_lens_batch
    
    def train(self, lr=1e-2, epochs=50, batch_size=5000):
        optimizer = Adam(self.mlp.parameters(), lr)


        for i in range(1,epochs):
            obs_batch, acts_batch, weights_batch, returns_batch, ep_lens_batch = self.train_rollouts(batch_size)
            optimizer.zero_grad()
            loss_batch = self.loss_func(obs= torch.as_tensor(obs_batch, dtype=torch.float32),
                                   acts= torch.as_tensor(acts_batch, dtype=torch.float32),
                                   weights = torch.as_tensor(weights_batch, dtype=torch.float32))
            loss_batch.backward()
            optimizer.step()
            print("Training epoch {} with loss {} and return {} and episode length {} ".format(i, loss_batch, np.mean(returns_batch), np.mean(ep_lens_batch)))