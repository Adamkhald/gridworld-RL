# Comparison of All Three DQN Implementations
# Loads and evaluates all trained models side-by-side

import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
from stable_baselines3 import DQN

import sys
sys.path.append('.')
from environment import StaticGridWorld, DynamicGridWorld

# Import network architectures
import torch.nn as nn

class QNetwork(nn.Module):
    """Classical DQN network."""
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(QNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, x):
        return self.network(x)


class DeepMindQNetwork(nn.Module):
    """DeepMind DQN network."""
    def __init__(self, num_frames, width, height, action_dim):
        super(DeepMindQNetwork, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(num_frames, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        
        conv_out_size = 64 * width * height
        
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Load training results
print("Loading training results...")

with open('classical_dqn_results.pkl', 'rb') as f:
    classical_results = pickle.load(f)

with open('deepmind_dqn_results.pkl', 'rb') as f:
    deepmind_results = pickle.load(f)

with open('sb3_dqn_results.pkl', 'rb') as f:
    sb3_results = pickle.load(f)

print("✓ All results loaded successfully")


# Load models
print("\nLoading trained models...")

# Classical DQN models
state_dim = 100  # 10x10 grid
action_dim = 4

classical_static = QNetwork(state_dim, action_dim).to(device)
classical_static.load_state_dict(torch.load('classical_dqn_static.pth', map_location=device))
classical_static.eval()

classical_dynamic = QNetwork(state_dim, action_dim).to(device)
classical_dynamic.load_state_dict(torch.load('classical_dqn_dynamic.pth', map_location=device))
classical_dynamic.eval()

# DeepMind DQN models
deepmind_static = DeepMindQNetwork(4, 10, 10, action_dim).to(device)
deepmind_static.load_state_dict(torch.load('deepmind_dqn_static.pth', map_location=device))
deepmind_static.eval()

deepmind_dynamic = DeepMindQNetwork(4, 10, 10, action_dim).to(device)
deepmind_dynamic.load_state_dict(torch.load('deepmind_dqn_dynamic.pth', map_location=device))
deepmind_dynamic.eval()

# Stable-Baselines3 models
sb3_static = DQN.load('sb3_dqn_static')
sb3_dynamic = DQN.load('sb3_dqn_dynamic')

print("✓ All models loaded successfully")


# Evaluate all models
def evaluate_classical(model, env, num_episodes=100):
    """Evaluate classical DQN model."""
    total_rewards = []
    
    for _ in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        
        while True:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = model(state_tensor)
                action = q_values.argmax(1).item()
            
            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            
            if terminated or truncated:
                break
        
        total_rewards.append(episode_reward)
    
    return np.mean(total_rewards), np.std(total_rewards)


def evaluate_deepmind(model, env, num_episodes=100):
    """Evaluate DeepMind DQN model."""
    from collections import deque
    total_rewards = []
    
    for _ in range(num_episodes):
        obs, _ = env.reset()
        frames = deque(maxlen=4)
        frame_2d = obs.reshape(10, 10)
        for _ in range(4):
            frames.append(frame_2d)
        
        episode_reward = 0
        
        while True:
            state = np.array(frames, dtype=np.float32)
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = model(state_tensor)
                action = q_values.argmax(1).item()
            
            obs, reward, terminated, truncated, _ = env.step(action)
            frame_2d = obs.reshape(10, 10)
            frames.append(frame_2d)
            episode_reward += reward
            
            if terminated or truncated:
                break
        
        total_rewards.append(episode_reward)
    
    return np.mean(total_rewards), np.std(total_rewards)


def evaluate_sb3(model, env, num_episodes=100):
    """Evaluate SB3 DQN model."""
    from stable_baselines3.common.evaluation import evaluate_policy
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=num_episodes)
    return mean_reward, std_reward


print("\n" + "=" * 60)
print("EVALUATION ON STATIC ENVIRONMENT")
print("=" * 60)

env_static = StaticGridWorld(width=10, height=10)

classical_static_mean, classical_static_std = evaluate_classical(classical_static, env_static)
print(f"Classical DQN:        {classical_static_mean:.2f} ± {classical_static_std:.2f}")

deepmind_static_mean, deepmind_static_std = evaluate_deepmind(deepmind_static, env_static)
print(f"DeepMind DQN:         {deepmind_static_mean:.2f} ± {deepmind_static_std:.2f}")

sb3_static_mean, sb3_static_std = evaluate_sb3(sb3_static, env_static)
print(f"Stable-Baselines3:    {sb3_static_mean:.2f} ± {sb3_static_std:.2f}")


print("\n" + "=" * 60)
print("EVALUATION ON DYNAMIC ENVIRONMENT")
print("=" * 60)

env_dynamic = DynamicGridWorld(width=10, height=10)

classical_dynamic_mean, classical_dynamic_std = evaluate_classical(classical_dynamic, env_dynamic)
print(f"Classical DQN:        {classical_dynamic_mean:.2f} ± {classical_dynamic_std:.2f}")

deepmind_dynamic_mean, deepmind_dynamic_std = evaluate_deepmind(deepmind_dynamic, env_dynamic)
print(f"DeepMind DQN:         {deepmind_dynamic_mean:.2f} ± {deepmind_dynamic_std:.2f}")

sb3_dynamic_mean, sb3_dynamic_std = evaluate_sb3(sb3_dynamic, env_dynamic)
print(f"Stable-Baselines3:    {sb3_dynamic_mean:.2f} ± {sb3_dynamic_std:.2f}")


# Plot comprehensive comparison
fig = plt.figure(figsize=(18, 12))

# Training curves - Static Environment
ax1 = plt.subplot(3, 2, 1)
ax1.plot(classical_results['rewards_static'], label='Classical DQN', alpha=0.7)
ax1.plot(deepmind_results['rewards_static'], label='DeepMind DQN', alpha=0.7)
ax1.plot(sb3_results['rewards_static'], label='SB3 DQN', alpha=0.7)
ax1.set_title('Static Environment: Training Rewards', fontsize=12, fontweight='bold')
ax1.set_xlabel('Episode')
ax1.set_ylabel('Reward')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Training curves - Dynamic Environment
ax2 = plt.subplot(3, 2, 2)
ax2.plot(classical_results['rewards_dynamic'], label='Classical DQN', alpha=0.7)
ax2.plot(deepmind_results['rewards_dynamic'], label='DeepMind DQN', alpha=0.7)
ax2.plot(sb3_results['rewards_dynamic'], label='SB3 DQN', alpha=0.7)
ax2.set_title('Dynamic Environment: Training Rewards', fontsize=12, fontweight='bold')
ax2.set_xlabel('Episode')
ax2.set_ylabel('Reward')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Moving averages - Static
window = 50
ax3 = plt.subplot(3, 2, 3)
if len(classical_results['rewards_static']) >= window:
    ma_classical = np.convolve(classical_results['rewards_static'], np.ones(window)/window, mode='valid')
    ax3.plot(ma_classical, label='Classical DQN', linewidth=2)
if len(deepmind_results['rewards_static']) >= window:
    ma_deepmind = np.convolve(deepmind_results['rewards_static'], np.ones(window)/window, mode='valid')
    ax3.plot(ma_deepmind, label='DeepMind DQN', linewidth=2)
if len(sb3_results['rewards_static']) >= window:
    ma_sb3 = np.convolve(sb3_results['rewards_static'], np.ones(window)/window, mode='valid')
    ax3.plot(ma_sb3, label='SB3 DQN', linewidth=2)
ax3.set_title(f'Static Environment: {window}-Episode Moving Average', fontsize=12, fontweight='bold')
ax3.set_xlabel('Episode')
ax3.set_ylabel('Average Reward')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Moving averages - Dynamic
ax4 = plt.subplot(3, 2, 4)
if len(classical_results['rewards_dynamic']) >= window:
    ma_classical = np.convolve(classical_results['rewards_dynamic'], np.ones(window)/window, mode='valid')
    ax4.plot(ma_classical, label='Classical DQN', linewidth=2)
if len(deepmind_results['rewards_dynamic']) >= window:
    ma_deepmind = np.convolve(deepmind_results['rewards_dynamic'], np.ones(window)/window, mode='valid')
    ax4.plot(ma_deepmind, label='DeepMind DQN', linewidth=2)
if len(sb3_results['rewards_dynamic']) >= window:
    ma_sb3 = np.convolve(sb3_results['rewards_dynamic'], np.ones(window)/window, mode='valid')
    ax4.plot(ma_sb3, label='SB3 DQN', linewidth=2)
ax4.set_title(f'Dynamic Environment: {window}-Episode Moving Average', fontsize=12, fontweight='bold')
ax4.set_xlabel('Episode')
ax4.set_ylabel('Average Reward')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Final performance comparison - Static
ax5 = plt.subplot(3, 2, 5)
models = ['Classical\nDQN', 'DeepMind\nDQN', 'SB3\nDQN']
means = [classical_static_mean, deepmind_static_mean, sb3_static_mean]
stds = [classical_static_std, deepmind_static_std, sb3_static_std]
bars = ax5.bar(models, means, yerr=stds, capsize=10, color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.7)
ax5.set_title('Static Environment: Final Performance', fontsize=12, fontweight='bold')
ax5.set_ylabel('Mean Reward (100 episodes)')
ax5.grid(True, alpha=0.3, axis='y')

# Final performance comparison - Dynamic
ax6 = plt.subplot(3, 2, 6)
means = [classical_dynamic_mean, deepmind_dynamic_mean, sb3_dynamic_mean]
stds = [classical_dynamic_std, deepmind_dynamic_std, sb3_dynamic_std]
bars = ax6.bar(models, means, yerr=stds, capsize=10, color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.7)
ax6.set_title('Dynamic Environment: Final Performance', fontsize=12, fontweight='bold')
ax6.set_ylabel('Mean Reward (100 episodes)')
ax6.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('complete_dqn_comparison.png', dpi=150)
plt.show()

print("\n" + "=" * 60)
print("COMPARISON SUMMARY")
print("=" * 60)
print("\nComparison plot saved as 'complete_dqn_comparison.png'")
print("\nAll three DQN variants have been trained and evaluated.")
print("Check the visualization for detailed performance comparison!")