# test.py
import torch
import gymnasium as gym
import numpy as np
import os
from agent import SACAgent 

MODEL_PATH = "sac_plots/sac_agent_episode_10000.pt" 
NUM_TEST_EPISODES = 10     
MAX_EPISODE_STEPS = 200    


if __name__ == "__main__":
    env = gym.make('Pusher-v5', render_mode='human')
    act_dim = env.action_space.shape[0]
    obs_dim = env.observation_space.shape[0]

    agent = SACAgent(obs_dim=obs_dim, act_dim=act_dim)

    checkpoint = torch.load(MODEL_PATH)
    agent.actor.load_state_dict(checkpoint['model'])
    print(f"Successfully loaded model from {MODEL_PATH}")

    agent.actor.eval()

    print(f"Starting testing for {NUM_TEST_EPISODES} episodes...")
    all_rewards = []

    for i_episode in range(NUM_TEST_EPISODES):
        observation, info = env.reset()
        episode_reward = 0.0
        done = False
        truncated = False
        steps = 0

        while not done and not truncated:
            action = agent.select_action_test(observation)
            observation, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward
            steps += 1

            done = terminated
            if steps >= MAX_EPISODE_STEPS:
                truncated = True 

        all_rewards.append(episode_reward)
        print(f"Episode {i_episode+1}: Reward = {episode_reward:.2f}, Steps = {steps}")

    env.close()
    print("\nTesting finished.")
    print(f"Average reward over {NUM_TEST_EPISODES} episodes: {np.mean(all_rewards):.2f}")