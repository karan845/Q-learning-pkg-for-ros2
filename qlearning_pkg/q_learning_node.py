#!/usr/bin/env python3
"""
ROS2 Node for Q-learning on a 40x40 grid with real-time path visualization.

This node trains an agent to navigate from the start cell ('S') to one of the goal cells ('G')
while avoiding obstacles ('X'). The agent is rewarded using a shaping reward that encourages moves
that reduce the Manhattan distance to the nearest goal. After training with Q-learning, the final Q‑table 
is printed to the console. Then, the node extracts the best path using the learned Q‑table and animates that 
path on a matplotlib window. The cumulative path is drawn as a blue line, and the current agent position is 
marked as a red dot. Grid lines and annotations are added for clarity.
"""

import rclpy
from rclpy.node import Node
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# -----------------------------
# Helper Function
# -----------------------------
def manhattan_distance(pos1, pos2):
    """
    Compute the Manhattan distance between two positions.

    Args:
        pos1 (tuple): (row, col) coordinates of the first position.
        pos2 (tuple): (row, col) coordinates of the second position.

    Returns:
        int: Manhattan distance.
    """
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


# -----------------------------
# Grid Environment Definition
# -----------------------------
class GridEnvironment:
    """
    Defines a 40x40 grid environment.

    The grid is represented as a 2D numpy array of strings:
      - '.' represents an open cell.
      - 'S' marks the start cell.
      - 'G' marks the goal cells (four cells in the bottom right).
      - 'X' marks obstacles.
    """
    def __init__(self):
        self.grid_size = 40  # Grid dimensions: 40 rows x 40 columns
        # Create the grid along with start, goal area, and obstacles.
        self.grid, self.start, self.goal_area, self.obstacles = self.create_grid()

    def create_grid(self):
        """
        Create the grid, assign the start and goal cells, and add obstacles.

        Returns:
            grid (np.array): A 40x40 numpy array containing grid symbols.
            start (tuple): Coordinates of the start cell.
            goal_area (list): List of coordinates for goal cells.
            obstacles (list): List of groups of obstacle coordinates.
        """
        # Initialize a grid filled with '.' to represent open space.
        grid = np.full((self.grid_size, self.grid_size), '.')
        
        # Define the start cell at the top-left corner.
        start = (0, 0)
        grid[start] = 'S'
        
        # Define the goal area (four cells in the bottom right) and mark them.
        goal_area = [(38, 38), (38, 39), (39, 38), (39, 39)]
        for goal in goal_area:
            grid[goal] = 'G'
        
        # Define obstacles as groups of coordinates.
        obstacles = [
            [(8, 8), (8, 9), (9, 8), (9, 9), (10, 8)],                     # Square with an extension
            [(18, 18), (18, 19), (19, 18), (19, 19), (20, 18), (19, 17)],      # L-shape
            [(25, 25), (25, 26), (26, 25), (26, 26), (27, 25)],               # Square with an extension
            [(30, 10), (30, 11), (31, 10), (31, 11), (32, 10)],               # Rectangular block
            [(35, 35), (35, 36), (36, 35), (36, 36), (37, 35)]                # Square with an extension
        ]
        
        # Mark obstacles on the grid with 'X'.
        for obs_group in obstacles:
            for obs in obs_group:
                grid[obs] = 'X'
                
        return grid, start, goal_area, obstacles

    def is_valid_position(self, position):
        """
        Check whether a given position is within grid boundaries and is not an obstacle.

        Args:
            position (tuple): (row, col) coordinates.

        Returns:
            bool: True if the position is valid; otherwise, False.
        """
        i, j = position
        # Check if the position is within the grid boundaries.
        if i < 0 or i >= self.grid_size or j < 0 or j >= self.grid_size:
            return False
        # Check if the cell is an obstacle.
        if self.grid[position] == 'X':
            return False
        return True

    def step(self, state, action):
        """
        Execute an action from a given state.

        Action encoding:
          - 0: move up (i - 1, j)
          - 1: move down (i + 1, j)
          - 2: move left (i, j - 1)
          - 3: move right (i, j + 1)

        The reward structure is as follows:
          - If the move is invalid (off-grid or into an obstacle), the agent receives -10 and stays in place.
          - If the move is valid and reaches a goal cell, the agent receives 1000 and the episode terminates.
          - Otherwise, a shaped reward is computed as a base penalty (-1) minus a factor (0.1)
            times the Manhattan distance to the nearest goal.

        Args:
            state (tuple): Current (row, col) position.
            action (int): Action to execute.

        Returns:
            next_state (tuple): New position after the move.
            reward (float): Reward received for the move.
            done (bool): True if a goal is reached; otherwise, False.
        """
        i, j = state
        
        # Calculate the next state based on the chosen action.
        if action == 0:      # Move up
            next_state = (i - 1, j)
        elif action == 1:    # Move down
            next_state = (i + 1, j)
        elif action == 2:    # Move left
            next_state = (i, j - 1)
        elif action == 3:    # Move right
            next_state = (i, j + 1)
        else:
            next_state = state
        
        # Check if the move is invalid.
        if not self.is_valid_position(next_state):
            next_state = state
            reward = -10
            done = False
            return next_state, reward, done
        else:
            # If the new state is one of the goal cells, assign a high reward and finish the episode.
            if next_state in self.goal_area:
                reward = 1000
                done = True
            else:
                # For a valid non-goal move, compute a shaped reward:
                # Base penalty (-1) minus a penalty proportional to the Manhattan distance to the nearest goal.
                base_penalty = -1
                factor = 0.1  # Shaping factor; can be tuned.
                distances = [manhattan_distance(next_state, goal) for goal in self.goal_area]
                min_distance = min(distances)
                reward = base_penalty - factor * min_distance
                done = False
                
        return next_state, reward, done


# -----------------------------
# Q-Learning ROS2 Node with Visualization and Q-value Printing
# -----------------------------
class QLearningNode(Node):
    """
    ROS2 Node that implements Q-learning on a 40x40 grid environment.

    The node trains an agent to navigate from the start ('S') to a goal ('G') while avoiding obstacles.
    After training, the final Q-table (with all learned Q-values) is printed to the console.
    Then, the best path is extracted by following the greedy (highest-Q) policy and animated using matplotlib.
    """
    def __init__(self):
        super().__init__('q_learning_node')
        self.get_logger().info('Q-Learning Node started.')

        # Initialize the grid environment.
        self.env = GridEnvironment()
        self.grid_size = self.env.grid_size

        # Define the number of possible actions: 0 (up), 1 (down), 2 (left), 3 (right).
        self.num_actions = 4

        # Initialize the Q-table: a 3D array with dimensions [grid_size, grid_size, num_actions].
        # Initially, all Q-values are set to 0.
        self.Q_table = np.zeros((self.grid_size, self.grid_size, self.num_actions))

        # Set Q-learning hyperparameters.
        self.alpha = 0.1           # Learning rate.
        self.gamma = 0.95          # Discount factor; emphasizes long-term rewards.
        self.epsilon = 1.0         # Initial exploration rate.
        self.epsilon_min = 0.01    # Minimum exploration rate.
        self.epsilon_decay = 0.995 # Decay factor for epsilon after each episode.

        # Set training parameters for the large grid.
        self.num_episodes = 5000           # Total training episodes.
        self.max_steps_per_episode = 2000  # Maximum steps allowed per episode.

        # Train the Q-learning agent.
        self.train_agent()

        # --- NEW: Print the final Q-values after training ---
        self.get_logger().info("Final Q-values (Q_table):")
        # Print the entire Q_table (a 3D numpy array). Note: This output can be very large.
        print(self.Q_table)

        # Extract the best path from the learned Q-table.
        best_path = self.get_best_path()
        self.get_logger().info("Best path found:")
        self.get_logger().info(str(best_path))

        # Visualize the best path using matplotlib.
        self.visualize_path(best_path)

    def choose_action(self, state):
        """
        Select an action based on the epsilon-greedy policy.

        With probability epsilon, the agent takes a random action (exploration).
        Otherwise, it takes the action with the highest Q-value in the current state (exploitation).

        Args:
            state (tuple): Current (row, col) position.

        Returns:
            int: The chosen action (0, 1, 2, or 3).
        """
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)  # Explore: return a random action.
        else:
            i, j = state
            # Exploit: return the action with the highest Q-value.
            return int(np.argmax(self.Q_table[i, j, :]))

    def train_agent(self):
        """
        Train the Q-learning agent over the specified number of episodes.

        For each episode, the agent starts at the starting cell and interacts with the environment.
        The Q-table is updated at each step using the Q-learning update rule.
        """
        for episode in range(self.num_episodes):
            state = self.env.start  # Reset the agent's state to the start cell at the beginning of each episode.
            done = False
            step = 0

            # Run the episode until the agent reaches a goal or exceeds the maximum number of steps.
            while not done and step < self.max_steps_per_episode:
                i, j = state
                action = self.choose_action(state)  # Select an action using epsilon-greedy policy.
                next_state, reward, done = self.env.step(state, action)  # Execute the action.
                ni, nj = next_state

                # Q-learning update rule:
                # Q(s,a) = Q(s,a) + alpha * (reward + gamma * max(Q(s', a')) - Q(s,a))
                old_value = self.Q_table[i, j, action]
                next_max = np.max(self.Q_table[ni, nj, :])
                self.Q_table[i, j, action] = old_value + self.alpha * (reward + self.gamma * next_max - old_value)

                state = next_state  # Transition to the next state.
                step += 1

            # Decay the exploration rate after each episode.
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            # Log training progress every 50 episodes.
            if episode % 50 == 0:
                self.get_logger().info(f"Episode {episode} completed. Epsilon: {self.epsilon:.3f}")

    def get_best_path(self):
        """
        Extract the best path from the start cell to a goal cell by following the greedy policy.

        The method starts at the initial state and always selects the action with the highest Q-value.
        Extraction stops when a goal cell is reached or if a loop is detected.

        Returns:
            list: A list of (row, col) coordinates representing the best path.
        """
        state = self.env.start
        best_path = [state]
        visited = set()
        visited.add(state)
        max_extraction_steps = 1000  # Safety limit to avoid infinite loops.

        for _ in range(max_extraction_steps):
            i, j = state
            action = int(np.argmax(self.Q_table[i, j, :]))
            next_state, reward, done = self.env.step(state, action)
            best_path.append(next_state)

            # Stop if a goal cell is reached.
            if next_state in self.env.goal_area:
                break

            # Detect loops: if the state was already visited, warn and stop extraction.
            if next_state in visited:
                self.get_logger().warn("Loop detected during path extraction. Stopping.")
                break

            visited.add(next_state)
            state = next_state

        return best_path

    def visualize_path(self, best_path):
        """
        Visualize the grid and animate the best path using matplotlib.

        The grid cells are colored as follows:
          - Open cells ('.') are white.
          - Obstacles ('X') are black.
          - Start cell ('S') is green.
          - Goal cells ('G') are red.
        As the animation proceeds:
          - The cumulative path is drawn as a blue line.
          - The current agent position is marked with a red circle.
          - The start and goal cells are annotated.
          - Grid lines and a legend are added for clarity.
          - The title is updated with the current step number.

        Args:
            best_path (list): List of (row, col) coordinates representing the path.
        """
        # Create a numeric grid based on the symbol mapping.
        symbol_to_code = {'.': 0, 'X': 1, 'S': 2, 'G': 3}
        num_grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                num_grid[i, j] = symbol_to_code[self.env.grid[i, j]]

        # Define a custom colormap: white for open, black for obstacles, green for start, red for goal.
        custom_cmap = ListedColormap(['white', 'black', 'green', 'red'])

        # Enable interactive mode to update the plot in real time.
        plt.ion()
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_title("Q-learning Path Visualization")
        ax.imshow(num_grid, cmap=custom_cmap)
        ax.set_xticks(np.arange(-.5, self.grid_size, 1), minor=True)
        ax.set_yticks(np.arange(-.5, self.grid_size, 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)

        # Annotate the start and goal cells.
        ax.text(0, 0, "Start", color="blue", fontsize=8, ha="center", va="center")
        for goal in self.env.goal_area:
            ax.text(goal[1], goal[0], "Goal", color="yellow", fontsize=8, ha="center", va="center")

        # Initialize lists for cumulative path coordinates (convert grid indices: x=column, y=row).
        x_coords = []
        y_coords = []

        # Animate the best path: update the cumulative path after each step.
        for step, state in enumerate(best_path):
            y_coords.append(state[0])
            x_coords.append(state[1])
            ax.clear()
            ax.imshow(num_grid, cmap=custom_cmap)
            ax.set_title(f"Q-learning Path Visualization (Step {step})")
            ax.set_xticks(np.arange(-.5, self.grid_size, 1), minor=True)
            ax.set_yticks(np.arange(-.5, self.grid_size, 1), minor=True)
            ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
            
            # Draw the cumulative path as a blue line with markers.
            ax.plot(x_coords, y_coords, 'b.-', markersize=8, linewidth=2, label="Path")
            # Mark the current agent position as a red circle.
            ax.plot(x_coords[-1], y_coords[-1], 'ro', markersize=10, label="Agent")
            # Re-annotate start and goal cells.
            ax.text(0, 0, "Start", color="blue", fontsize=8, ha="center", va="center")
            for goal in self.env.goal_area:
                ax.text(goal[1], goal[0], "Goal", color="yellow", fontsize=8, ha="center", va="center")
            # Add a legend on the first step.
            if step == 0:
                ax.legend(loc="upper right")
            plt.pause(0.2)

        # Turn off interactive mode and display the final plot.
        plt.ioff()
        plt.show()


# -----------------------------
# Main Function to Launch the ROS2 Node
# -----------------------------
def main(args=None):
    """
    Main entry point for the ROS2 node.

    Initializes the ROS2 system, creates an instance of the Q-learning node (which
    handles training, printing the final Q-values, and visualization), and then shuts down.
    """
    rclpy.init(args=args)
    node = QLearningNode()
    # Spin briefly; the heavy work is done in the node's __init__.
    rclpy.spin_once(node, timeout_sec=1)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
