# agent.py
import torch
from torch import optim
import torch.nn.functional as F
import numpy as np
import copy 
from model import Actor, Critic
from replay_buffer import ReplayBuffer

GAMMA = 0.99           
TAU = 0.005 # Target network update rate
LEARNING_RATE = 3e-4 
POLICY_UPDATE_FREQ = 2 # Ones in every 2 critic updates
REPLAY_BUFFER_CAPACITY = 1000000
BATCH_SIZE = 256       
INITIAL_RANDOM_STEPS = 10000 

class SACAgent:
    def __init__(self, obs_dim, act_dim):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = GAMMA
        self.tau = TAU
        self.policy_update_freq = POLICY_UPDATE_FREQ

        # Actor initialization
        self.actor = Actor(obs_dim, act_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LEARNING_RATE)

        # Critic initialization
        self.critic1 = Critic(obs_dim, act_dim)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=LEARNING_RATE)
        self.critic2 = Critic(obs_dim, act_dim)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=LEARNING_RATE)

        self.target_critic1 = Critic(obs_dim, act_dim)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic1.eval() 
        self.target_critic2 = Critic(obs_dim, act_dim)
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        self.target_critic2.eval() 

        # Alpha initialization
        self.target_entropy = -torch.tensor(act_dim, dtype=torch.float32)
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=3e-4)
        self.alpha = self.log_alpha.exp().detach() # Detach initial alpha

        self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_CAPACITY)

        self.total_updates = 0

    def select_action_test(self, state): # gets the mean just for testing
        if not isinstance(state, torch.Tensor):
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        else:
            state_tensor = state.unsqueeze(0)
        _, _, mean = self.actor.sample(state_tensor)
        action_np = mean.detach().squeeze(0).numpy()
        return action_np

    def select_action(self, state): # normal sampling during training
        if not isinstance(state, torch.Tensor):
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        else:
            state_tensor = state.unsqueeze(0)
        action, _, _ = self.actor.sample(state_tensor)
        action_np = action.detach().squeeze(0).numpy()
        return action_np

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def _update_critic(self, states, actions, rewards, next_states, dones):
        with torch.no_grad():
            next_actions, next_log_probs, _ = self.actor.sample(next_states)

            q1_target_next = self.target_critic1(next_states, next_actions)
            q2_target_next = self.target_critic2(next_states, next_actions)

            q_target_next = torch.min(q1_target_next, q2_target_next)

            # Add entropy term: Q_target = r + gamma * (1 - done) * (min_Q_target - alpha * log_pi)
            q_target = rewards + self.gamma * (1.0 - dones) * (q_target_next - self.alpha * next_log_probs)

        q1_current = self.critic1(states, actions)
        q2_current = self.critic2(states, actions)

        critic1_loss = F.mse_loss(q1_current, q_target)
        critic2_loss = F.mse_loss(q2_current, q_target)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()


    def _update_actor_and_alpha(self, states):
        actions_pi, log_probs_pi, _ = self.actor.sample(states)

        q1_pi = self.critic1(states, actions_pi)
        q2_pi = self.critic2(states, actions_pi)
        q_pi = torch.min(q1_pi, q2_pi) 

        # Loss = E[alpha * log_pi - Q] 
        actor_loss = (self.alpha * log_probs_pi - q_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha * (log_probs_pi + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.alpha = self.log_alpha.exp().detach() 


    def _update_target_networks(self):
        for target_param, local_param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
        for target_param, local_param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)


    def update(self, batch_size): # all updates in one function
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        self._update_critic(states, actions, rewards, next_states, dones)

        if self.total_updates % self.policy_update_freq == 0: # less frequent than critic updates
            self._update_actor_and_alpha(states)
            self._update_target_networks() 
        self.total_updates += 1

    def save(self, filepath):
        torch.save({
            'model': self.actor.state_dict(),
        }, filepath)
        print(f"SAC Agent saved to {filepath}")

