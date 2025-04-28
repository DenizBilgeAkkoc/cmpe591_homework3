# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

LOG_STD_MAX = 2
LOG_STD_MIN = -20 
MAX_ACTION = 2.0

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dims=[256, 256]):
        super().__init__()
        layers = []
        layers.append(nn.Linear(obs_dim, hidden_dims[0]))
        layers.append(nn.ReLU())
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

        self.mean_layer = nn.Linear(hidden_dims[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_dims[-1], act_dim)

        self.act_dim = act_dim
        self.action_scale = torch.tensor(MAX_ACTION, dtype=torch.float32)


    def forward(self, obs):
        net_out = self.net(obs)
        mean = self.mean_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def get_distribution(self, obs):
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        return Normal(mean, std)

    def sample(self, obs):
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        normal = Normal(mean, std)

        action = normal.rsample() # This part is critic allows gradient flow through sample

        log_prob = normal.log_prob(action)
        log_prob = log_prob.sum(axis=-1, keepdim=True) # basic log prob sum

        return action, log_prob, mean


class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dims = [256, 256]):
        super().__init__()
        layers = []
        layers.append(nn.Linear(obs_dim + act_dim, hidden_dims[0]))
        layers.append(nn.ReLU())
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dims[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, obs, act):
        q_input = torch.cat([obs, act], dim=-1)
        q_value = self.net(q_input)
        return q_value