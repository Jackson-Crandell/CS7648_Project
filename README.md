# CS7648_Project
The aim of this project is to improve learning efficiency of the [Deep Tamer algorithm](https://arxiv.org/pdf/1709.10163v2.pdf) through the incorporation of an active learning acquisition function. We will utilize the [Minigrid](https://minigrid.farama.org/content/basic_usage/) environment to do this. First, we will obtain baselines of a human, RL agent, [TAMER](https://ieeexplore.ieee.org/abstract/document/4640845) agent, and Deep TAMER agent. Once we obtain 
those baselines, we will implement our own version of Deep TAMER + Active Learning to demonstrate it's success over all baselines. 

## Setup Instructions

1. `pip install minigrid`
2. (Optionally) git clone [rl-starter-files](https://github.com/lcswillems/rl-starter-files)
3. (Optionally) git clone torch-ac

## Minigrid Environment

### Minigrid Empty 5x5
This environment is a simple 2D go-to-goal task where the agent (red arrow) learns to reach the goal (green square).

![](https://github.com/Jackson-Crandell/CS7648_Project/blob/main/media/MiniGrid-Empty-5x5-v0.png?raw=true)

## Current Progress

### RL Agents
For our RL agents, we implemented a PPO and A2C algorithm based on the rl-starter-files to train our agents. 

*PPO Training Curve*

![](https://github.com/Jackson-Crandell/CS7648_Project/blob/main/media/PPO_Empty_training.png?raw=true)

*PPO Evaluation (100 Episodes)*

![](https://github.com/Jackson-Crandell/CS7648_Project/blob/main/media/PPO_Empty_Scatter.png?raw=true)

*A2C Training Curve*

![](https://github.com/Jackson-Crandell/CS7648_Project/blob/main/media/A2C_Empty_training.png?raw=true)

*A2C Evaluation (100 Episodes)*

![](https://github.com/Jackson-Crandell/CS7648_Project/blob/main/media/A2C_Empty_Scatter.png?raw=true)

### Human Agent

Human Evaluation (10 Episodes)

![](https://github.com/Jackson-Crandell/CS7648_Project/blob/main/media/Minigrid_Empty_Human_rewards.png?raw=true)



## Repository References

[Gymnasium](https://gymnasium.farama.org/content/basic_usage/)

[Minigrid](https://minigrid.farama.org/content/basic_usage/)

[Minigrid Github](https://github.com/Farama-Foundation/Minigrid)

[RL Starter Files](https://github.com/lcswillems/rl-starter-files)

[Torch AC (Actor Critic algorithms for Deep RL- for Minigrid)](https://github.com/lcswillems/torch-ac)
