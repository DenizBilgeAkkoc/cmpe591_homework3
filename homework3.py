import time
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal  # Use Normal distribution for continuous actions

import numpy as np
import matplotlib.pyplot as plt

import environment


class Hw3Env(environment.BaseEnv):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # Continuous action space: delta in x and y coordinates
        self._delta_limit = 0.03
        self.action_space = [(-self._delta_limit, self._delta_limit), (-self._delta_limit, self._delta_limit)]
        self._action_dim = 2  # Dimension of the continuous action space

        self._goal_thresh = 0.01
        self._max_timesteps = 500

    def _create_scene(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        scene = environment.create_tabletop_scene()
        obj_pos = [np.random.uniform(0.25, 0.75),
                   np.random.uniform(-0.3, 0.3),
                   1.5]
        goal_pos = [np.random.uniform(0.25, 0.75),
                    np.random.uniform(-0.3, 0.3),
                    1.025]
        environment.create_object(scene, "box", pos=obj_pos, quat=[0, 0, 0, 1],
                                  size=[0.03, 0.03, 0.03], rgba=[0.8, 0.2, 0.2, 1],
                                  name="obj1")
        environment.create_visual(scene, "cylinder", pos=goal_pos, quat=[0, 0, 0, 1],
                                  size=[0.05, 0.005], rgba=[0.2, 1.0, 0.2, 1],
                                  name="goal")
        return scene

    def high_level_state(self):
        ee_pos = self.data.site(self._ee_site).xpos[:2]
        obj_pos = self.data.body("obj1").xpos[:2]
        goal_pos = self.data.site("goal").xpos[:2]
        return np.concatenate([ee_pos, obj_pos, goal_pos])

    def reward(self):
        state = self.high_level_state()
        ee_pos = state[:2]
        obj_pos = state[2:4]
        goal_pos = state[4:6]
        ee_to_obj = max(100 * np.linalg.norm(ee_pos - obj_pos), 1)
        obj_to_goal = max(100 * np.linalg.norm(obj_pos - goal_pos), 1)
        return 1 / (ee_to_obj) + 1 / (obj_to_goal)

    def is_terminal(self):
        obj_pos = self.data.body("obj1").xpos[:2]
        goal_pos = self.data.site("goal").xpos[:2]
        return np.linalg.norm(obj_pos - goal_pos) < self._goal_thresh

    def is_truncated(self):
        return self._t >= self._max_timesteps

    def step(self, action):
        # Ensure the action is within the defined bounds
        action = np.clip(action, [-self._delta_limit, -self._delta_limit], [self._delta_limit, self._delta_limit])

        ee_pos = self.data.site(self._ee_site).xpos[:2]
        target_pos = np.concatenate([ee_pos, [1.08]])
        target_pos[:2] = np.clip(target_pos[:2] + action, [0.25, -0.3], [0.75, 0.3])
        self._set_ee_in_cartesian(target_pos, rotation=[-90, 0, 180], n_splits=30, threshold=0.04)
        self._t += 1

        next_state = self.high_level_state()
        reward = self.reward()
        terminal = self.is_terminal()
        truncated = self.is_truncated()
        return next_state, reward, terminal, truncated
    
class ReinforcePolicy(nn.Module):
    def __init__(self, state_size, action_size):
        super(ReinforcePolicy, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 64)
        self.mean_head = nn.Linear(64, action_size)
        self.log_std_head = nn.Linear(64, action_size)  # Output log standard deviation for stability

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        mean = self.mean_head(x)
        log_std = self.log_std_head(x) # saw this on medium Policy Parameterization for a Continuous Action Space article
        std = torch.exp(log_std)
        return mean, std

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        mean, std = self.forward(state)
        distribution = Normal(mean, std)
        action = distribution.sample()
        log_prob = distribution.log_prob(action).sum(dim=-1)  # Sum log probs for all action dimensions
        return action.cpu().numpy()[0], log_prob
    
def train(policy, optimizer, num_episodes, max_steps_per_episode, discount_factor, print_every_episode, env):
    episode_rewards_history = deque(maxlen=100)
    all_episode_rewards = []

    for episode in range(1, num_episodes + 1):
        saved_log_probabilities = []
        episode_rewards = []
        current_state = env.reset()
        env._set_joint_position({env._gripper_idx: 0.80})

        current_state = env.high_level_state()
        total_reward_this_episode = 0

        for step in range(max_steps_per_episode):
            action, log_prob = policy.act(current_state)
            saved_log_probabilities.append(log_prob)
            next_state, reward, done, truncated = env.step(action)
            episode_rewards.append(reward)
            total_reward_this_episode += reward
            current_state = next_state
            if done or truncated:
                break

        episode_rewards_history.append(total_reward_this_episode)
        all_episode_rewards.append(total_reward_this_episode)

        returns = deque(maxlen=max_steps_per_episode)
        n_steps = len(episode_rewards)
        for i in range(n_steps)[::-1]:
            discounted_return = returns[0] if returns else 0
            returns.appendleft(discount_factor * discounted_return + episode_rewards[i])

        # Normalize returns
        returns_tensor = torch.tensor(list(returns))
        if returns_tensor.std() > 0:
            # Normalize returns
            returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-9)
        else:
            # Handle the case where std is zero (e.g., all returns are the same)
            returns_tensor = torch.zeros_like(returns_tensor) # Or some other appropriate handling


        policy_loss = []
        for log_prob, discounted_return in zip(saved_log_probabilities, returns_tensor):
            policy_loss.append(-log_prob * discounted_return)
        policy_loss = torch.stack(policy_loss).sum()  # Stack log probabilities

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        print(f"Episode {episode}\tReward: {total_reward_this_episode:.2f}\tAverage Reward (Last 100): {np.mean(episode_rewards_history):.2f}")
        if episode % print_every_episode == 0:
            print(f"\nEpisode {episode}\tAverage Reward (Last 100): {np.mean(episode_rewards_history):.2f}")

    # Save the trained model
    torch.save(policy.state_dict(), 'reinforce_hw3_continuous_model.pth')
    print("\nTraining finished! Model saved as reinforce_hw3_continuous_model.pth")
    return all_episode_rewards

def test(env, num_evaluation_episodes, max_steps_per_episode, policy):
    evaluation_rewards = []
    for _ in range(num_evaluation_episodes):
        current_state = env.reset()
        current_state = env.high_level_state()
        total_reward = 0
        for _ in range(max_steps_per_episode):
            action, _ = policy.act(current_state)
            next_state, reward, done, truncated = env.step(action)
            total_reward += reward
            current_state = next_state
            if done or truncated:
                break
        evaluation_rewards.append(total_reward)
    mean_reward = np.mean(evaluation_rewards)
    std_reward = np.std(evaluation_rewards)
    print(f"Evaluation Mean Reward: {mean_reward:.2f}, Std Reward: {std_reward:.2f}")
    return mean_reward, std_reward

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    env = Hw3Env(render_mode="offscreen")

    state_size = env.high_level_state().shape[0]
    action_size = env._action_dim  # Updated action size

    # Hyperparameters
    training_episodes = 200
    evaluation_episodes = 10
    max_episode_steps = 500
    gamma = 0.997
    learning_rate = 3e-4
    print_every = 100

    # Initializing policy and optimizer
    policy = ReinforcePolicy(state_size, action_size).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

    # Training the agent
    all_rewards = train(policy, optimizer, training_episodes, max_episode_steps, gamma, print_every, env)

    # Evaluating the agent
    mean_reward, std_reward = test(env, evaluation_episodes, max_episode_steps, policy)

    # To plot and save all episode rewards
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, training_episodes + 1), all_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Total Reward per Episode during Training (Continuous Actions)")
    plt.grid(True)
    plt.savefig("reinforce_training_rewards_continuous.png")
    plt.show()