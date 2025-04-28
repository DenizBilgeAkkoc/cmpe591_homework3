import torch
import gymnasium as gym
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import datetime
from agent import Agent

SAVE_DIR = "reinforce_plots" 
os.makedirs(SAVE_DIR, exist_ok=True)
RENDER_MODE = None 
MAX_EPISODE_STEPS = 200
ENV_NAME = 'Pusher-v5'
NUM_EPISODES = 1000000
SAVE_INTERVAL = 1000
PLOT_WINDOW = 100

if __name__ == "__main__":
    env = gym.make('Pusher-v5', render_mode=None) 
    act_dim = env.action_space.shape[0]
    obs_dim = env.observation_space.shape[0]
    agent = Agent(obs_dim=obs_dim, act_dim=act_dim) 

    all_episode_rewards = []
    moving_avg_rewards = []
    start_time = time.time()

    for i in range(NUM_EPISODES):
        episode_start_time = time.time()
        observation, info = env.reset()
        observation = np.array(observation)

        done = False
        truncated = False
        cumulative_reward = 0.0
        episode_steps = 0

        while not done and not truncated:
            action = agent.sample_action(observation)

            # Step environment
            observation, reward, terminated, truncated, info = env.step(action)
            observation = np.array(observation)

            agent.store_reward(reward)

            cumulative_reward += reward

            done = terminated
            episode_steps += 1

        loss = agent.update() 

        # --- Logging ---
        all_episode_rewards.append(cumulative_reward)
        if len(all_episode_rewards) >= PLOT_WINDOW:
            current_moving_avg = np.mean(all_episode_rewards[-PLOT_WINDOW:])
        else:
            current_moving_avg = np.mean(all_episode_rewards)
        moving_avg_rewards.append(current_moving_avg) # Store for plotting

        episode_end_time = time.time()
        episode_duration = episode_end_time - episode_start_time
        elapsed_time = episode_end_time - start_time
        episodes_completed = i + 1
        avg_time_per_episode = elapsed_time / episodes_completed
        remaining_episodes = NUM_EPISODES - episodes_completed
        estimated_time_remaining = avg_time_per_episode * remaining_episodes
        time_remaining_td = datetime.timedelta(seconds=int(estimated_time_remaining))
        time_elapsed_td = datetime.timedelta(seconds=int(elapsed_time))

        print(f"Ep={episodes_completed}/{NUM_EPISODES}, Steps={episode_steps}, Loss={loss:.4f}, Reward={cumulative_reward:.2f}, AvgRew({PLOT_WINDOW})={current_moving_avg:.2f}")
        print(f"Elapsed: {time_elapsed_td}, Est. Remain: {time_remaining_td}")

        # --- Saving Model and Plot ---
        if (i + 1) % SAVE_INTERVAL == 0 or i == NUM_EPISODES - 1:
            # Save Model
            model_save_path = os.path.join(SAVE_DIR, f"model_episode_{i+1}.pt")
            torch.save(agent.model.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path}")

            # Save Plot
            plot_save_path = os.path.join(SAVE_DIR, f"rewards_plot_episode_{i+1}.png")
            plt.figure(figsize=(12, 6))
            plt.plot(all_episode_rewards, label='Raw Episode Rewards', alpha=0.3)
            plt.plot(moving_avg_rewards, label=f'Moving Average ({PLOT_WINDOW} episodes)', color='red', linewidth=2)
            plt.xlabel("Episode")
            plt.ylabel("Cumulative Reward")
            plt.title(f"Training Rewards (Episode {i+1}) - {ENV_NAME}")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(plot_save_path)
            plt.close()
            print(f"Reward plot saved to {plot_save_path}")

    # --- Final Saves ---
    final_model_path = os.path.join(SAVE_DIR, "model_final.pt")
    if not os.path.exists(final_model_path):
        torch.save(agent.model.state_dict(), final_model_path)
        print(f"Final model saved to {final_model_path}")

    rews_path = os.path.join(SAVE_DIR, "rewards.npy")
    np.save(rews_path, np.array(all_episode_rewards))
    print(f"Reward history saved to {rews_path}")

    final_plot_path = os.path.join(SAVE_DIR, "rewards_plot_final.png")
    if not os.path.exists(final_plot_path): 
        plt.figure(figsize=(12, 6))
        plt.plot(all_episode_rewards, label='Raw Episode Rewards', alpha=0.3)
        plt.plot(moving_avg_rewards, label=f'Moving Average ({PLOT_WINDOW} episodes)', color='red', linewidth=2)
        plt.xlabel("Episode")
        plt.ylabel("Cumulative Reward")
        plt.title(f"Training Rewards (Final) - {ENV_NAME}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(final_plot_path)
        plt.close()
        print(f"Final reward plot saved to {final_plot_path}")

    env.close()
    print("Training finished.")