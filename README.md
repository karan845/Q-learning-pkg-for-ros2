# ROS2 Q-Learning Grid Navigation with Real-Time Visualization

This repository contains a ROS2 node written in Python that demonstrates Q-learning on a 40x40 grid environment. The agent learns to navigate from a start cell to one of several goal cells while avoiding obstacles. A shaping reward based on the Manhattan distance to the nearest goal is used to guide the learning process. After training, the final Q‑table is printed and the best (greedy) path is extracted and animated using Matplotlib.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running the Program](#running-the-program)
- [Code Structure](#code-structure)
- [Troubleshooting](#troubleshooting)


## Overview

The ROS2 node in this project trains an agent using Q-learning in a grid world where:
- **S**: Marks the start position (top-left corner).
- **G**: Represents the goal cells (four cells in the bottom right).
- **X**: Denotes obstacles.
- **.**: Represents open cells.

The agent receives:
- A high reward when reaching a goal.
- A penalty for invalid moves.
- A shaping reward based on the Manhattan distance to the closest goal.

After training for a specified number of episodes, the node:
1. Prints the entire Q‑table (a 3D NumPy array).
2. Extracts the best path from the learned Q‑table using a greedy policy.
3. Visualizes the path in real time using Matplotlib.

## Features

- **ROS2 Integration**: Uses `rclpy` to implement the ROS2 node.
- **Q-learning**: Implements the Q-learning algorithm with parameters like learning rate, discount factor, and an epsilon-greedy policy.
- **Real-time Visualization**: Animates the path of the agent on the grid.
- **Custom Grid Environment**: A 40x40 grid with obstacles, a start cell, and multiple goal cells.
- **Path Extraction**: Extracts and displays the best path based on the trained Q‑table.

## Prerequisites

Before you begin, ensure you have met the following requirements:

- **ROS2 Installation**: You need to have ROS2 installed (e.g., Foxy, Humble, or Rolling). For installation instructions, visit the [ROS2 installation guide](https://docs.ros.org/en/foxy/Installation.html).
- **Python 3**: The script uses Python 3.
- **Python Packages**: Install the following Python packages:
  - `numpy`
  - `matplotlib`
  - `rclpy` (comes with the ROS2 Python client library)

## Installation

Follow these steps to set up the project:

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/karan845/Q-learning-pkg-for-ros2.git
   cd ros2-q-learning-grid
   ```

2. **Set Up Your ROS2 Environment:**

   Source your ROS2 installation. For example, if you are using ROS2 Foxy:

   ```bash
   source /opt/ros/foxy/setup.bash
   ```

3. **Create a ROS2 Workspace (if you don’t have one already):**

   ```bash
   mkdir -p ~/ros2_ws/src
   cd ~/ros2_ws/src
   ```

4. **Copy or Link the Repository into Your Workspace:**

   ```bash
   ln -s /path/to/ros2-q-learning-grid .
   ```

5. **Install Python Dependencies:**

   It is recommended to use a Python virtual environment:

   ```bash
   python3 -m venv qlearning_env
   source qlearning_env/bin/activate
   pip install numpy matplotlib
   ```

   _Note_: `rclpy` is part of your ROS2 installation and does not need to be installed via pip.

6. **Build the Workspace:**

   Return to your workspace root and build using `colcon`:

   ```bash
   cd ~/ros2_ws
   colcon build --packages-select ros2_q_learning_grid
   ```

   Replace `ros2_q_learning_grid` with the actual package name if different.

7. **Source the Workspace:**

   After building, source the local setup file:

   ```bash
   source install/setup.bash
   ```

## Running the Program

There are two common ways to run the node:

### Option 1: Running as a Standalone Python Script

If you are not integrating into a larger ROS2 project, you can run the script directly:

```bash
python3 path/to/q_learning_node.py
```

### Option 2: Running as a ROS2 Node

1. Make sure your workspace is sourced:

   ```bash
   source ~/ros2_ws/install/setup.bash
   ```

2. Use the `ros2 run` command:

   ```bash
   ros2 run ros2_q_learning_grid q_learning_node
   ```

   Ensure that your package’s `setup.py` and entry points are correctly configured so that ROS2 can locate the node.

Once the node starts, you should see logging messages in the console showing the progress of training. After training:
- The final Q‑table will be printed.
- The best path will be extracted and animated in a Matplotlib window.

## Code Structure

- **q_learning_node.py**:  
  The main file that defines:
  - The `GridEnvironment` class, which sets up the grid, start, goal, and obstacles.
  - The `QLearningNode` class, a ROS2 node that performs Q-learning, prints the Q‑table, extracts the best path, and visualizes the path.
  - The `main()` function that initializes ROS2 and spins the node briefly.

- **Helper Functions**:
  - `manhattan_distance`: Calculates the Manhattan distance between two grid positions.

## Troubleshooting

- **ROS2 Environment Issues**:  
  Make sure you source the correct ROS2 environment (`/opt/ros/<distro>/setup.bash`) before running or building the workspace.

- **Dependency Errors**:  
  If you encounter missing module errors, double-check that you have installed the required Python packages:
  ```bash
  pip install numpy matplotlib
  ```

- **Visualization Problems**:  
  If the Matplotlib window does not appear or the animation is not smooth, ensure that your Python environment supports interactive plotting. You may try running the script in a different backend or updating Matplotlib.

- **Infinite Loop in Path Extraction**:  
  The code has a safeguard for loop detection during the path extraction. If you see warnings about loops, check the training parameters or adjust the grid/obstacle configuration.

By following the above instructions, you should be able to install, build, and run the ROS2 Q-Learning Grid Navigation node. Happy coding and experimenting with reinforcement learning!
