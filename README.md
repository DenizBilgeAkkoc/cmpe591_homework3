# CMPE591 Homework 3

This repository contains the implementation of reinforcement learning algorithms for solving the `Pusher-v5` environment. The project includes training and testing scripts for two main approaches: Soft Actor Critic (SAC) and Vanilla Policy Gradient (REINFORCE).

---

## Repository Structure

```
cmpe591_homework_3/
│
├── soft_actor_critic/
│   ├── main.py  # Training script for SAC
│   ├── test.py   # Testing script for SAC
│   ├── ...
│
├── vanilla_policy_gradient/
    ├── main.py  # Training script for Vanilla Policy Gradient
    ├── test.py   # Testing script for Vanilla Policy Gradient
    ├── ...

```

---

## Training and Testing

### Training
- **Soft Actor Critic (SAC)**
  - The training script for SAC is located in `soft_actor_critic/main.py`.

- **Vanilla Policy Gradient (REINFORCE)**
  - The training script for Vanilla Policy Gradient is located in `vanilla_policy_gradient/main.py`.

### Testing
- **Soft Actor Critic (SAC)**
  - The testing script for SAC is located in `soft_actor_critic/test.py`.
  - Here is the real output for the test script (Please keep in mind that sac is trained only on 10000 epochs while reinforce trained on 1M):
    ```python
    Successfully loaded model from sac_plots/sac_agent_final.pt
    Starting testing for 10 episodes...
    Episode 1: Reward = -22.44, Steps = 100
    Episode 2: Reward = -40.46, Steps = 100
    Episode 3: Reward = -32.70, Steps = 100
    Episode 4: Reward = -38.54, Steps = 100
    Episode 5: Reward = -29.17, Steps = 100
    Episode 6: Reward = -39.68, Steps = 100
    Episode 7: Reward = -26.50, Steps = 100
    Episode 8: Reward = -23.91, Steps = 100
    Episode 9: Reward = -22.62, Steps = 100
    Episode 10: Reward = -31.10, Steps = 100
    
    Testing finished.
    Average reward over 10 episodes: -30.71
    ```

- **Vanilla Policy Gradient (REINFORCE)**
  - The testing script for Vanilla Policy Gradient is located in `vanilla_policy_gradient/test.py`.
  - here is the real output for the test script:
    ```python
    --- Testing Agent ---
    Environment: Pusher-v5
    Model Path: reinforce_plots/model_final.pt
    Number of Test Episodes: 10
    Render Mode: human
    Initializing Agent: obs_dim=23, act_dim=7, lr=5e-06
    Initializing VPG Model: obs_dim=23, act_dim=7, hidden_layers=[128, 128]
    Model weights loaded successfully.
    Test Episode 1: Steps=200, Reward=-36.64
    Test Episode 2: Steps=200, Reward=-38.25
    Test Episode 3: Steps=200, Reward=-39.82
    Test Episode 4: Steps=200, Reward=-33.11
    Test Episode 5: Steps=200, Reward=-37.40
    Test Episode 6: Steps=200, Reward=-37.60
    Test Episode 7: Steps=200, Reward=-35.03
    Test Episode 8: Steps=200, Reward=-34.25
    Test Episode 9: Steps=200, Reward=-54.59
    Test Episode 10: Steps=200, Reward=-39.17
    
    --- Test Results ---
    Average Reward over 10 episodes: -38.59 +/- 5.71
    Average Steps per episode: 200.0
    Testing finished.
    ```

---

## Final Reward Plots

The training process for the algorithms is visualized through reward plots. Below are the final reward plots for each algorithm:

### Part 1: Soft Actor Critic (SAC)
![Final Reward Plot - SAC](https://github.com/DenizBilgeAkkoc/cmpe591_homework3/blob/main/soft_actor_critic/sac_plots/rewards_plot_final.png)

 I know the plot above is a terrible one because I forgot to clip while plotting so here is a less trained version with clipped plot just to see the it converges: 

![ Clipped Reward Plot - SAC ](https://github.com/DenizBilgeAkkoc/cmpe591_homework3/blob/main/soft_actor_critic/sac_plots_clipped/rewards_plot_clipped__final.png)

### Part 2: Vanilla Policy Gradient (REINFORCE)
![Final Reward Plot - REINFORCE](https://github.com/DenizBilgeAkkoc/cmpe591_homework3/blob/main/vanilla_policy_gradient/reinforce_plots/rewards_plot_final.png)

---

## How to Run

 Run the training scripts:
   ```bash
   python soft_actor_critic/main.py
   python vanilla_policy_gradient/main.py
   ```

Run the testing scripts:
   ```bash
   python soft_actor_critic/test.py
   python vanilla_policy_gradient/test.py
   ```

