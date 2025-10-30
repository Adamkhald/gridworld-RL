# Stable-Baselines3 DQN Implementation
# Uses the production-ready SB3 library for training

import numpy as np
import matplotlib.pyplot as plt
import pickle
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import torch

import sys
sys.path.append('.')
from environment import StaticGridWorld, DynamicGridWorld

# Set random seed
torch.manual_seed(42)
np.random.seed(42)

print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")


class TrainingCallback:
    """Custom callback to track episode rewards during training."""
    
    def __init__(self):
        self.episode_rewards = []
        self.episode_lengths = []
    
    def on_episode_end(self, episode_reward, episode_length):
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)


def train_sb3_dqn(env, total_timesteps=100000, verbose=True):
    """Train Stable-Baselines3 DQN agent."""
    
    # Wrap environment with Monitor to track statistics
    env = Monitor(env)
    
    # Create DQN model
    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-3,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=32,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1000,
        exploration_fraction=0.1,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        policy_kwargs=dict(net_arch=[128, 128]),
        verbose=1 if verbose else 0,
        seed=42
    )
    
    # Train the model
    model.learn(total_timesteps=total_timesteps)
    
    return model, env


def extract_training_data(monitor_env):
    """Extract episode rewards and lengths from Monitor wrapper."""
    results = monitor_env.get_episode_rewards()
    lengths = monitor_env.get_episode_lengths()
    return results, lengths


def evaluate_sb3_agent(model, env, num_episodes=100):
    """Evaluate trained SB3 agent."""
    mean_reward, std_reward = evaluate_policy(
        model, 
        env, 
        n_eval_episodes=num_episodes,
        deterministic=True
    )
    return mean_reward, std_reward


# Training on Static Grid World
print("=" * 60)
print("Training Stable-Baselines3 DQN on Static Grid World")
print("=" * 60)

env_static = StaticGridWorld(width=10, height=10)
model_static, monitor_static = train_sb3_dqn(
    env_static, 
    total_timesteps=100000, 
    verbose=True
)

# Extract training data
rewards_static, lengths_static = extract_training_data(monitor_static)

# Evaluate
mean_reward, std_reward = evaluate_sb3_agent(
    model_static, 
    StaticGridWorld(width=10, height=10), 
    num_episodes=100
)
print(f"\nStatic Environment Evaluation: {mean_reward:.2f} ± {std_reward:.2f}")


# Training on Dynamic Grid World
print("\n" + "=" * 60)
print("Training Stable-Baselines3 DQN on Dynamic Grid World")
print("=" * 60)

env_dynamic = DynamicGridWorld(width=10, height=10)
model_dynamic, monitor_dynamic = train_sb3_dqn(
    env_dynamic, 
    total_timesteps=100000, 
    verbose=True
)

# Extract training data
rewards_dynamic, lengths_dynamic = extract_training_data(monitor_dynamic)

# Evaluate
mean_reward, std_reward = evaluate_sb3_agent(
    model_dynamic, 
    DynamicGridWorld(width=10, height=10), 
    num_episodes=100
)
print(f"\nDynamic Environment Evaluation: {mean_reward:.2f} ± {std_reward:.2f}")


# Plot training curves
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Static environment rewards
axes[0, 0].plot(rewards_static)
axes[0, 0].set_title('Stable-Baselines3 DQN - Static Environment: Episode Rewards')
axes[0, 0].set_xlabel('Episode')
axes[0, 0].set_ylabel('Reward')
axes[0, 0].grid(True)

# Dynamic environment rewards
axes[0, 1].plot(rewards_dynamic)
axes[0, 1].set_title('Stable-Baselines3 DQN - Dynamic Environment: Episode Rewards')
axes[0, 1].set_xlabel('Episode')
axes[0, 1].set_ylabel('Reward')
axes[0, 1].grid(True)

# Static environment lengths
axes[1, 0].plot(lengths_static)
axes[1, 0].set_title('Stable-Baselines3 DQN - Static Environment: Episode Lengths')
axes[1, 0].set_xlabel('Episode')
axes[1, 0].set_ylabel('Length')
axes[1, 0].grid(True)

# Dynamic environment lengths
axes[1, 1].plot(lengths_dynamic)
axes[1, 1].set_title('Stable-Baselines3 DQN - Dynamic Environment: Episode Lengths')
axes[1, 1].set_xlabel('Episode')
axes[1, 1].set_ylabel('Length')
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig('sb3_dqn_training.png')
plt.show()


# Save models and results
model_static.save('sb3_dqn_static')
model_dynamic.save('sb3_dqn_dynamic')

with open('sb3_dqn_results.pkl', 'wb') as f:
    pickle.dump({
        'rewards_static': rewards_static,
        'rewards_dynamic': rewards_dynamic,
        'lengths_static': lengths_static,
        'lengths_dynamic': lengths_dynamic
    }, f)

print("\nModels and results saved!")
print("- sb3_dqn_static.zip")
print("- sb3_dqn_dynamic.zip")
print("- sb3_dqn_results.pkl")


# Additional analysis: Plot moving averages
window = 50

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Static environment moving average
if len(rewards_static) >= window:
    moving_avg_static = np.convolve(rewards_static, np.ones(window)/window, mode='valid')
    axes[0].plot(moving_avg_static)
    axes[0].set_title(f'SB3 DQN - Static Environment: {window}-Episode Moving Average')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Average Reward')
    axes[0].grid(True)

# Dynamic environment moving average
if len(rewards_dynamic) >= window:
    moving_avg_dynamic = np.convolve(rewards_dynamic, np.ones(window)/window, mode='valid')
    axes[1].plot(moving_avg_dynamic)
    axes[1].set_title(f'SB3 DQN - Dynamic Environment: {window}-Episode Moving Average')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Average Reward')
    axes[1].grid(True)

plt.tight_layout()
plt.savefig('sb3_dqn_moving_averages.png')
plt.show()

print("\nTraining complete! Check the plots for performance analysis.")