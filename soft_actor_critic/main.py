import torch
import gymnasium as gym
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import datetime
from collections import deque 

from agent import SACAgent, BATCH_SIZE, INITIAL_RANDOM_STEPS, REPLAY_BUFFER_CAPACITY

SAVE_DIR = "sac_plots" 
os.makedirs(SAVE_DIR, exist_ok=True)
MAX_EPISODE_STEPS = 200 
NUM_EPISODES = 10000 
SAVE_INTERVAL = 1000 
PLOT_WINDOW = 100 
LOG_INTERVAL = 10 

if __name__ == "__main__":
    env = gym.make('Pusher-v5', render_mode=None) 

    act_dim = env.action_space.shape[0]
    obs_dim = env.observation_space.shape[0]
    agent = SACAgent(obs_dim=obs_dim, act_dim=act_dim)

    all_episode_rewards = []
    moving_avg_rewards_deque = deque(maxlen=PLOT_WINDOW)
    moving_avg_rewards = []
    start_time = time.time()
    total_steps = 0

    for i_episode in range(NUM_EPISODES):
        episode_start_time = time.time()
        observation, info = env.reset()

        done = False
        truncated = False
        cumulative_reward = 0.0
        episode_steps = 0

        while not done and not truncated:
            if total_steps < INITIAL_RANDOM_STEPS: # random sampling to fill the buffer
                action = env.action_space.sample()
            else:
                action = agent.select_action(observation) # normal sampling

            observation, reward, terminated, truncated, info = env.step(action)

            agent.store_transition(observation, action, reward, observation, terminated)

            cumulative_reward += reward
            episode_steps += 1
            total_steps += 1

            if total_steps >= INITIAL_RANDOM_STEPS: # start training after filling the buffer
                agent.update(BATCH_SIZE)

            done = terminated
            if episode_steps >= MAX_EPISODE_STEPS:
                 truncated = True

        all_episode_rewards.append(cumulative_reward)

        moving_avg_rewards_deque.append(cumulative_reward)
        current_moving_avg = np.mean(moving_avg_rewards_deque)
        moving_avg_rewards.append(current_moving_avg)


        # logging and saving
        if (i_episode + 1) % LOG_INTERVAL == 0:
            episode_end_time = time.time()
            episode_duration = episode_end_time - episode_start_time
            elapsed_time = episode_end_time - start_time
            avg_steps = total_steps / (i_episode + 1)

            print(f"Ep={i_episode+1}/{NUM_EPISODES}, Steps={episode_steps}, TotalSteps={total_steps}, Reward={cumulative_reward:.2f}, AvgRew({PLOT_WINDOW})={current_moving_avg:.2f}")
            print(f"  Episode Time: {episode_duration:.2f}s, Elapsed: {datetime.timedelta(seconds=int(elapsed_time))}")


        # --- Saving Model and Plot ---
        if (i_episode + 1) % SAVE_INTERVAL == 0 or i_episode == NUM_EPISODES - 1:
            model_save_path = os.path.join(SAVE_DIR, f"sac_agent_episode_{i_episode+1}.pt")
            agent.save(model_save_path)

            plot_save_path = os.path.join(SAVE_DIR, f"rewards_plot_episode_{i_episode+1}.png")
            plt.figure(figsize=(12, 6))
            plt.plot(all_episode_rewards, label='Raw Episode Rewards', alpha=0.3)
            plt.plot(moving_avg_rewards, label=f'Moving Average ({PLOT_WINDOW} episodes)', color='red', linewidth=2)
            plt.xlabel("Episode")
            plt.ylabel("Cumulative Reward")
            plt.title(f"SAC Training Rewards (Episode {i_episode+1})")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(plot_save_path)
            plt.close()
            print(f"Reward plot saved to {plot_save_path}")

    # --- Final Saves ---
    final_model_path = os.path.join(SAVE_DIR, "sac_agent_final.pt")
    if not os.path.exists(final_model_path):
        agent.save(final_model_path)

    final_plot_path = os.path.join(SAVE_DIR, "rewards_plot_final.png")
    if not os.path.exists(final_plot_path):
        plt.figure(figsize=(12, 6))
        plt.plot(all_episode_rewards, label='Raw Episode Rewards', alpha=0.3)
        plt.plot(moving_avg_rewards, label=f'Moving Average ({PLOT_WINDOW} episodes)', color='red', linewidth=2)
        plt.xlabel("Episode")
        plt.ylabel("Cumulative Reward")
        plt.title(f"SAC Training Rewards (Final)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(final_plot_path)
        plt.close()
        print(f"Final reward plot saved to {final_plot_path}")

    env.close()
    print("Training finished.")