import torch
from torch import optim
import numpy as np
from torch.distributions.normal import Normal
import torch.nn.functional as F
from model import VPG 

GAMMA = 0.99
LEARNING_RATE = 5e-6
EPS = 1e-6  # Small number for numerical stability

class Agent():
    def __init__(self, obs_dim, act_dim, lr = LEARNING_RATE):
        print(f"Initializing Agent: obs_dim={obs_dim}, act_dim={act_dim}, lr={lr}")
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = GAMMA

        # --- Policy Network and Optimizer ---
        self.model = VPG(obs_dim=obs_dim, act_dim=act_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.log_probs = []
        self.rewards = []
        self.baseline = 0  # Initialize the baseline
        self.baseline_alpha = 0.9  # Exponential moving average factor


    def _clear_memory(self):
        """Clears the memory buffers after an update."""
        self.log_probs = []
        self.rewards = []

    def sample_action(self, state):
        if not isinstance(state, torch.Tensor):
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        else:
             state_tensor = state.unsqueeze(0) if state.ndim == 1 else state

        distrib = self.model.get_distribution(state_tensor)
        action_tensor = distrib.sample() # Shape (1, act_dim)

        log_prob = distrib.log_prob(action_tensor).sum(axis=-1) # Shape (1,7)
        self.entropy = distrib.entropy().sum(dim=-1).mean()
        self.log_probs.append(log_prob)

        return action_tensor.squeeze(0).detach().numpy() 

    def store_reward(self, reward):
        self.rewards.append(reward)

    def _calculate_discounted_rewards(self):
        returns = []
        discounted_sum = 0.0
        for reward in reversed(self.rewards):
            discounted_sum = reward + self.gamma * discounted_sum
            returns.insert(0, discounted_sum) 

        returns = torch.tensor(returns, dtype=torch.float32)
        self.baseline = self.baseline_alpha * self.baseline + (1 - self.baseline_alpha) * returns.mean().item()
        advantages = returns - self.baseline

        return advantages

    def update(self):

        returns = self._calculate_discounted_rewards() 

        log_probs_tensor = torch.cat(self.log_probs) 
        
        policy_loss = - (log_probs_tensor * returns).sum()# - self.entropy_constant * self.entropy
        self.optimizer.zero_grad() 
        policy_loss.backward()    

        self.optimizer.step()      

        self._clear_memory()

        return policy_loss.item()
