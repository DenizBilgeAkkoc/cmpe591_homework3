import torch
import gymnasium as gym
import numpy as np
import os
import time
from agent import Agent

ENV_NAME = 'Pusher-v5'
MODEL_PATH = "reinforce_plots/model_final.pt"
NUM_TEST_EPISODES = 10 
MAX_EPISODE_STEPS = 200 
RENDER_MODE = 'human' 
SEED = 42 


def test_agent(model_path: str):
    """Loads a trained agent and tests its performance."""

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    print(f"--- Testing Agent ---")
    print(f"Environment: {ENV_NAME}")
    print(f"Model Path: {model_path}")
    print(f"Number of Test Episodes: {NUM_TEST_EPISODES}")
    print(f"Render Mode: {RENDER_MODE}")

    try:
        env = gym.make(ENV_NAME, render_mode=RENDER_MODE, max_episode_steps=MAX_EPISODE_STEPS)
    except Exception as e:
        print(f"Error creating environment '{ENV_NAME}': {e}")
        return

    act_dim = env.action_space.shape[0]
    observation_dim = env.observation_space.shape[0]
    agent = Agent(obs_dim=observation_dim, act_dim=act_dim) 

    try:
        agent.model.load_state_dict(torch.load(model_path))
        agent.model.eval()
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"Error loading model weights from {model_path}: {e}")
        env.close()
        return


    episode_rewards = []
    episode_steps = []

    for i in range(NUM_TEST_EPISODES):
        observation, info = env.reset(seed=SEED + i if SEED is not None else None)
        done = False
        truncated = False
        cumulative_reward = 0.0
        current_steps = 0

        while not done and not truncated:
            with torch.no_grad(): 
                state_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
                dist = agent.model.get_distribution(state_tensor)
                action_tensor = dist.mean
                action_np = action_tensor.squeeze(0).cpu().numpy()

            observation, reward, terminated, truncated, info = env.step(action_np)


            cumulative_reward += reward
            current_steps += 1
            done = terminated

        print(f"Test Episode {i+1}: Steps={current_steps}, Reward={cumulative_reward:.2f}")
        episode_rewards.append(cumulative_reward)
        episode_steps.append(current_steps)

    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    avg_steps = np.mean(episode_steps)

    print("\n--- Test Results ---")
    print(f"Average Reward over {NUM_TEST_EPISODES} episodes: {avg_reward:.2f} +/- {std_reward:.2f}")
    print(f"Average Steps per episode: {avg_steps:.1f}")


    env.close()
    print("Testing finished.")

if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
         print(f"Error: Model path '{MODEL_PATH}' does not exist.")
         print("Please update MODEL_PATH in the script to point to your saved .pt file.")
    else:
        test_agent(MODEL_PATH)