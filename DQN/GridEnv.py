import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def get_user_input():
    """Ask user for grid world configuration."""
    print("=" * 50)
    print("GRID WORLD CONFIGURATION")
    print("=" * 50)
    
    # Grid size
    height = int(input("\nGrid height: "))
    width = int(input("Grid width: "))
    
    # Number of obstacles
    num_obstacles = int(input("\nNumber of obstacles: "))
    
    # Obstacle placement
    print("\nObstacle placement:")
    print("1. Random")
    print("2. In a row")
    print("3. In a column")
    print("4. Specific position")
    obs_placement = input("Choose (1-4): ")
    
    if obs_placement == '2':
        obs_row = int(input(f"Which row (0-{height-1}): "))
        obs_col = None
        obs_specific = None
    elif obs_placement == '3':
        obs_col = int(input(f"Which column (0-{width-1}): "))
        obs_row = None
        obs_specific = None
    elif obs_placement == '4':
        obs_row = int(input(f"Row (0-{height-1}): "))
        obs_col = int(input(f"Column (0-{width-1}): "))
        obs_specific = (obs_row, obs_col)
        obs_row = None
        obs_col = None
    else:
        obs_row = None
        obs_col = None
        obs_specific = None
    
    # Agent placement
    print("\nAgent placement:")
    print("1. Random")
    print("2. Specific position")
    agent_placement = input("Choose (1-2): ")
    
    if agent_placement == '2':
        agent_row = int(input(f"Agent row (0-{height-1}): "))
        agent_col = int(input(f"Agent column (0-{width-1}): "))
        agent_pos = (agent_row, agent_col)
    else:
        agent_pos = None
    
    # Goal placement
    print("\nGoal placement:")
    print("1. Random")
    print("2. Specific position")
    goal_placement = input("Choose (1-2): ")
    
    if goal_placement == '2':
        goal_row = int(input(f"Goal row (0-{height-1}): "))
        goal_col = int(input(f"Goal column (0-{width-1}): "))
        goal_pos = (goal_row, goal_col)
    else:
        goal_pos = None
    
    return {
        'height': height,
        'width': width,
        'num_obstacles': num_obstacles,
        'obs_row': obs_row,
        'obs_col': obs_col,
        'obs_specific': obs_specific,
        'agent_pos': agent_pos,
        'goal_pos': goal_pos
    }


def create_grid_world(config):
    """Create grid world based on user configuration."""
    height = config['height']
    width = config['width']
    
    # Initialize empty grid
    obstacles = np.zeros((height, width), dtype=bool)
    
    # Place obstacles
    if config['obs_row'] is not None:
        # Place in a row
        row = config['obs_row']
        for i in range(min(config['num_obstacles'], width)):
            obstacles[row, i] = True
    elif config['obs_col'] is not None:
        # Place in a column
        col = config['obs_col']
        for i in range(min(config['num_obstacles'], height)):
            obstacles[i, col] = True
    elif config['obs_specific'] is not None:
        # Place at specific position
        r, c = config['obs_specific']
        obstacles[r, c] = True
    else:
        # Place randomly
        count = 0
        while count < config['num_obstacles']:
            r = np.random.randint(0, height)
            c = np.random.randint(0, width)
            if not obstacles[r, c]:
                obstacles[r, c] = True
                count += 1
    
    # Place agent
    if config['agent_pos'] is not None:
        agent_pos = config['agent_pos']
    else:
        # Random placement
        while True:
            r = np.random.randint(0, height)
            c = np.random.randint(0, width)
            if not obstacles[r, c]:
                agent_pos = (r, c)
                break
    
    # Place goal
    if config['goal_pos'] is not None:
        goal_pos = config['goal_pos']
    else:
        # Random placement
        while True:
            r = np.random.randint(0, height)
            c = np.random.randint(0, width)
            if not obstacles[r, c] and (r, c) != agent_pos:
                goal_pos = (r, c)
                break
    
    return obstacles, agent_pos, goal_pos


def plot_grid_world(height, width, obstacles, agent_pos, goal_pos):
    """Plot the grid world using matplotlib."""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Set beige background
    ax.set_facecolor('#F5F5DC')
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect('equal')
    
    # Draw grid lines
    for i in range(height + 1):
        ax.axhline(i, color='gray', linewidth=0.5)
    for j in range(width + 1):
        ax.axvline(j, color='gray', linewidth=0.5)
    
    # Draw obstacles (dark red)
    for i in range(height):
        for j in range(width):
            if obstacles[i, j]:
                rect = Rectangle((j, height - i - 1), 1, 1, 
                               facecolor='#8B0000', edgecolor='black', linewidth=1)
                ax.add_patch(rect)
    
    # Draw goal (green)
    goal_rect = Rectangle((goal_pos[1], height - goal_pos[0] - 1), 1, 1,
                          facecolor='#00FF00', edgecolor='black', linewidth=2)
    ax.add_patch(goal_rect)
    
    # Draw agent (blue)
    agent_rect = Rectangle((agent_pos[1], height - agent_pos[0] - 1), 1, 1,
                           facecolor='#0000FF', edgecolor='black', linewidth=2)
    ax.add_patch(agent_rect)
    
    # Labels
    ax.set_title('Grid World Environment', fontsize=16, fontweight='bold')
    ax.set_xlabel('Column', fontsize=12)
    ax.set_ylabel('Row', fontsize=12)
    ax.set_xticks(range(width))
    ax.set_yticks(range(height))
    ax.invert_yaxis()
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#0000FF', edgecolor='black', label='Agent'),
        Patch(facecolor='#00FF00', edgecolor='black', label='Goal'),
        Patch(facecolor='#8B0000', edgecolor='black', label='Obstacle')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Get configuration from user
    config = get_user_input()
    
    # Create grid world
    obstacles, agent_pos, goal_pos = create_grid_world(config)
    
    # Plot it
    print("\nGenerating grid world...")
    plot_grid_world(config['height'], config['width'], obstacles, agent_pos, goal_pos)