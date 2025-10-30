# Grid World Environment Setup
# Run this notebook first to create the custom Gymnasium environments

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple

class StaticGridWorld(gym.Env):
    """Grid world with fixed agent start, obstacles, and goal positions."""
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(self, width=10, height=10, render_mode=None):
        super().__init__()
        self.width = width
        self.height = height
        self.render_mode = render_mode
        
        # Action space: 0=up, 1=right, 2=down, 3=left
        self.action_space = spaces.Discrete(4)
        
        # Observation space: flattened grid (agent=1, obstacle=2, goal=3, empty=0)
        self.observation_space = spaces.Box(
            low=0, high=3, shape=(width * height,), dtype=np.float32
        )
        
        # Fixed positions
        self.start_pos = (0, 0)
        self.goal_pos = (width - 1, height - 1)
        self.obstacles = self._generate_fixed_obstacles()
        
        self.agent_pos = None
        self.steps = 0
        self.max_steps = width * height * 2
        
    def _generate_fixed_obstacles(self):
        """Generate fixed obstacle positions."""
        obstacles = set()
        np.random.seed(42)  # Fixed seed for reproducibility
        num_obstacles = max(1, (self.width * self.height) // 10)
        
        while len(obstacles) < num_obstacles:
            pos = (np.random.randint(0, self.width), np.random.randint(0, self.height))
            if pos != self.start_pos and pos != self.goal_pos:
                obstacles.add(pos)
        
        return obstacles
    
    def _get_obs(self):
        """Get current observation as flattened grid."""
        grid = np.zeros((self.width, self.height), dtype=np.float32)
        
        # Mark obstacles
        for obs in self.obstacles:
            grid[obs] = 2
        
        # Mark goal
        grid[self.goal_pos] = 3
        
        # Mark agent
        grid[self.agent_pos] = 1
        
        return grid.flatten()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = self.start_pos
        self.steps = 0
        
        observation = self._get_obs()
        info = {}
        
        return observation, info
    
    def step(self, action):
        self.steps += 1
        
        # Calculate new position
        x, y = self.agent_pos
        if action == 0:  # up
            new_pos = (x, max(0, y - 1))
        elif action == 1:  # right
            new_pos = (min(self.width - 1, x + 1), y)
        elif action == 2:  # down
            new_pos = (x, min(self.height - 1, y + 1))
        else:  # left
            new_pos = (max(0, x - 1), y)
        
        # Check if new position is valid
        if new_pos not in self.obstacles:
            self.agent_pos = new_pos
        
        # Calculate reward
        if self.agent_pos == self.goal_pos:
            reward = 1.0
            terminated = True
        else:
            reward = -0.01  # Small penalty for each step
            terminated = False
        
        truncated = self.steps >= self.max_steps
        
        observation = self._get_obs()
        info = {}
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        if self.render_mode == "human" or self.render_mode == "rgb_array":
            grid = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            grid[:, :] = [255, 255, 255]  # White background
            
            # Draw obstacles
            for obs in self.obstacles:
                grid[obs[1], obs[0]] = [0, 0, 0]  # Black
            
            # Draw goal
            grid[self.goal_pos[1], self.goal_pos[0]] = [0, 255, 0]  # Green
            
            # Draw agent
            grid[self.agent_pos[1], self.agent_pos[0]] = [255, 0, 0]  # Red
            
            if self.render_mode == "human":
                plt.imshow(grid)
                plt.axis('off')
                plt.show()
            
            return grid


class DynamicGridWorld(gym.Env):
    """Grid world where the goal moves randomly each episode."""
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(self, width=10, height=10, render_mode=None):
        super().__init__()
        self.width = width
        self.height = height
        self.render_mode = render_mode
        
        # Action space: 0=up, 1=right, 2=down, 3=left
        self.action_space = spaces.Discrete(4)
        
        # Observation space: flattened grid
        self.observation_space = spaces.Box(
            low=0, high=3, shape=(width * height,), dtype=np.float32
        )
        
        # Fixed positions
        self.start_pos = (0, 0)
        self.obstacles = self._generate_fixed_obstacles()
        
        # Dynamic goal
        self.goal_pos = None
        self.agent_pos = None
        self.steps = 0
        self.max_steps = width * height * 2
        
    def _generate_fixed_obstacles(self):
        """Generate fixed obstacle positions."""
        obstacles = set()
        np.random.seed(42)
        num_obstacles = max(1, (self.width * self.height) // 10)
        
        while len(obstacles) < num_obstacles:
            pos = (np.random.randint(0, self.width), np.random.randint(0, self.height))
            if pos != self.start_pos:
                obstacles.add(pos)
        
        return obstacles
    
    def _generate_random_goal(self):
        """Generate a random goal position (not on obstacle or start)."""
        while True:
            pos = (np.random.randint(0, self.width), np.random.randint(0, self.height))
            if pos != self.start_pos and pos not in self.obstacles:
                return pos
    
    def _get_obs(self):
        """Get current observation as flattened grid."""
        grid = np.zeros((self.width, self.height), dtype=np.float32)
        
        # Mark obstacles
        for obs in self.obstacles:
            grid[obs] = 2
        
        # Mark goal
        grid[self.goal_pos] = 3
        
        # Mark agent
        grid[self.agent_pos] = 1
        
        return grid.flatten()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = self.start_pos
        self.goal_pos = self._generate_random_goal()  # Random goal each episode
        self.steps = 0
        
        observation = self._get_obs()
        info = {}
        
        return observation, info
    
    def step(self, action):
        self.steps += 1
        
        # Calculate new position
        x, y = self.agent_pos
        if action == 0:  # up
            new_pos = (x, max(0, y - 1))
        elif action == 1:  # right
            new_pos = (min(self.width - 1, x + 1), y)
        elif action == 2:  # down
            new_pos = (x, min(self.height - 1, y + 1))
        else:  # left
            new_pos = (max(0, x - 1), y)
        
        # Check if new position is valid
        if new_pos not in self.obstacles:
            self.agent_pos = new_pos
        
        # Calculate reward
        if self.agent_pos == self.goal_pos:
            reward = 1.0
            terminated = True
        else:
            reward = -0.01
            terminated = False
        
        truncated = self.steps >= self.max_steps
        
        observation = self._get_obs()
        info = {}
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        if self.render_mode == "human" or self.render_mode == "rgb_array":
            grid = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            grid[:, :] = [255, 255, 255]
            
            for obs in self.obstacles:
                grid[obs[1], obs[0]] = [0, 0, 0]
            
            grid[self.goal_pos[1], self.goal_pos[0]] = [0, 255, 0]
            grid[self.agent_pos[1], self.agent_pos[0]] = [255, 0, 0]
            
            if self.render_mode == "human":
                plt.imshow(grid)
                plt.axis('off')
                plt.show()
            
            return grid


# Test the environments
if __name__ == "__main__":
    print("Testing StaticGridWorld...")
    env_static = StaticGridWorld(width=10, height=10)
    obs, info = env_static.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env_static.action_space}")
    
    print("\nTesting DynamicGridWorld...")
    env_dynamic = DynamicGridWorld(width=10, height=10)
    obs, info = env_dynamic.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env_dynamic.action_space}")
    
    # Test a few steps
    print("\nTesting environment dynamics...")
    for i in range(5):
        action = env_static.action_space.sample()
        obs, reward, terminated, truncated, info = env_static.step(action)
        print(f"Step {i+1}: Action={action}, Reward={reward:.3f}, Done={terminated or truncated}")
    
    print("\nEnvironments ready! Save this file as '01_environment.ipynb'")