
# Summary of Grid World Reinforcement Learning Tasks

## Overview

In this project, we explored various reinforcement learning algorithms applied to Grid World environments. Our focus was on implementing and analyzing Value Iteration, Policy Iteration, TD-Learning, and Q-Learning. We also considered the impact of different parameters like immediate rewards, noise in actions, and the number of episodes on the learning process.

## Tasks and Implementation

### 1. Environment Setup

- **Grid World Environment**: We created a Grid World environment represented as a matrix with specific dynamics, including non-deterministic state transitions and rewards.
- **Transition and Rewards Matrices**: Developed functions to generate transition and reward matrices based on grid dimensions, noise levels, terminal states, and wall positions.

### 2. Algorithms Implementation

- **Value and Policy Iteration**: Implemented these dynamic programming methods to solve the Grid World as a Markov Decision Process.
- **TD-Learning and Q-Learning**: Implemented these model-free reinforcement learning methods. TD-Learning updates value estimates based on subsequent states, while Q-Learning learns the value of action-state pairs.

### 3. Experimental Analysis

- **Comparative Analysis**: Set up experiments to compare the algorithms in terms of convergence time, policy changes with immediate rewards, and the impact of learning parameters.
- **Noise Model Adjustment**: Modified the transition matrix creation to incorporate a specific noise model in actions.

### 4. Visualization and Logging

- Utilized Matplotlib for visualizing results like policy changes and learning curves.
- Created functions for logging data during experiments for further analysis.

## Running the Code in Jupyter Notebook

1. **Setup**: Ensure you have a Jupyter Notebook environment with Python, Numpy, and Matplotlib installed.
2. **Code Integration**: Copy the provided Python code snippets into separate cells in your Jupyter Notebook.
3. **Grid Configuration**: Place the grid configuration file (`grid1.txt`) in the same directory as your notebook or provide the correct path to the file.
4. **Run Experiments**: Execute the cells in order, starting from environment setup to algorithm implementations and experiments.
5. **Data Analysis and Visualization**: Run the logging and visualization cells to collect data and visualize the results of the experiments.

## Conclusion

This project demonstrates the application and analysis of key reinforcement learning algorithms in a controlled Grid World environment. Through these experiments, we gain insights into the behavior of different learning methods and the impact of various environmental factors and algorithm parameters.
