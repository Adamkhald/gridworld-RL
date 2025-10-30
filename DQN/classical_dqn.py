# Classical Deep Q-Network Implementation
# Implements the foundational DQN with experience replay and target network

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
import pickle
from typing import List, Tuple

# Import environment from previous notebook
import sys
sys.path.append('.')
from environment import StaticGridWorld, DynamicGridWorld

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class ReplayBuffer:
    """Experience replay buffer for storing and sampling transitions."""
    
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32)
        )
    
    def __len__(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    """Neural network for approximating Q-values."""
    
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


class ClassicalDQN:
    """Classical DQN agent with experience replay and target network."""
    
    def __init__(
        self,
        state_dim,
        action_dim,
        learning_rate=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_size=10000,
        batch_size=64,
        target_update_freq=100
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Q-networks
        self.q_network = QNetwork(state_dim, action_dim).to(device)
        self.target_network = QNetwork(state_dim, action_dim).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        self.steps = 0
    
    def select_action(self, state, training=True):
        """Epsilon-greedy action selection."""
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax(1).item()
    
    def train_step(self):
        """Perform one training step."""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)
        
        # Current Q-values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q-values
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Compute loss
        loss = nn.MSELoss()(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return loss.item()
    
    def update_epsilon(self):
        """Decay epsilon."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)


def train_dqn(env, agent, num_episodes=1000, verbose=True):
    """Train DQN agent."""
    episode_rewards = []
    episode_lengths = []
    losses = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        
        while True:
            action = agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.replay_buffer.push(state, action, reward, next_state, done)
            loss = agent.train_step()
            
            if loss is not None:
                losses.append(loss)
            
            state = next_state
            episode_reward += reward
            episode_length += 1
            
            if done:
                break
        
        agent.update_epsilon()
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        if verbose and (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_length = np.mean(episode_lengths[-100:])
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Avg Length: {avg_length:.1f} | "
                  f"Epsilon: {agent.epsilon:.3f}")
    
    return episode_rewards, episode_lengths, losses


def evaluate_agent(env, agent, num_episodes=100):
    """Evaluate trained agent."""
    total_rewards = []
    
    for _ in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        
        while True:
            action = agent.select_action(state, training=False)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            state = next_state
            
            if terminated or truncated:
                break
        
        total_rewards.append(episode_reward)
    
    return np.mean(total_rewards), np.std(total_rewards)


# Training on Static Grid World
print("=" * 60)
print("Training Classical DQN on Static Grid World")
print("=" * 60)

env_static = StaticGridWorld(width=10, height=10)
state_dim = env_static.observation_space.shape[0]
action_dim = env_static.action_space.n

agent_static = ClassicalDQN(
    state_dim=state_dim,
    action_dim=action_dim,
    learning_rate=1e-3,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.01,
    epsilon_decay=0.995,
    buffer_size=10000,
    batch_size=64,
    target_update_freq=100
)

rewards_static, lengths_static, losses_static = train_dqn(
    env_static, agent_static, num_episodes=1000, verbose=True
)

# Evaluate on static environment
mean_reward, std_reward = evaluate_agent(env_static, agent_static, num_episodes=100)
print(f"\nStatic Environment Evaluation: {mean_reward:.2f} ± {std_reward:.2f}")


# Training on Dynamic Grid World
print("\n" + "=" * 60)
print("Training Classical DQN on Dynamic Grid World")
print("=" * 60)

env_dynamic = DynamicGridWorld(width=10, height=10)

agent_dynamic = ClassicalDQN(
    state_dim=state_dim,
    action_dim=action_dim,
    learning_rate=1e-3,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.01,
    epsilon_decay=0.995,
    buffer_size=10000,
    batch_size=64,
    target_update_freq=100
)

rewards_dynamic, lengths_dynamic, losses_dynamic = train_dqn(
    env_dynamic, agent_dynamic, num_episodes=1000, verbose=True
)

# Evaluate on dynamic environment
mean_reward, std_reward = evaluate_agent(env_dynamic, agent_dynamic, num_episodes=100)
print(f"\nDynamic Environment Evaluation: {mean_reward:.2f} ± {std_reward:.2f}")


# Plot training curves
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Static environment rewards
axes[0, 0].plot(rewards_static)
axes[0, 0].set_title('Classical DQN - Static Environment: Episode Rewards')
axes[0, 0].set_xlabel('Episode')
axes[0, 0].set_ylabel('Reward')
axes[0, 0].grid(True)

# Dynamic environment rewards
axes[0, 1].plot(rewards_dynamic)
axes[0, 1].set_title('Classical DQN - Dynamic Environment: Episode Rewards')
axes[0, 1].set_xlabel('Episode')
axes[0, 1].set_ylabel('Reward')
axes[0, 1].grid(True)

# Static environment losses
axes[1, 0].plot(losses_static)
axes[1, 0].set_title('Classical DQN - Static Environment: Training Loss')
axes[1, 0].set_xlabel('Training Step')
axes[1, 0].set_ylabel('Loss')
axes[1, 0].grid(True)

# Dynamic environment losses
axes[1, 1].plot(losses_dynamic)
axes[1, 1].set_title('Classical DQN - Dynamic Environment: Training Loss')
axes[1, 1].set_xlabel('Training Step')
axes[1, 1].set_ylabel('Loss')
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig('classical_dqn_training.png')
plt.show()


# Save models and results
torch.save(agent_static.q_network.state_dict(), 'classical_dqn_static.pth')
torch.save(agent_dynamic.q_network.state_dict(), 'classical_dqn_dynamic.pth')

with open('classical_dqn_results.pkl', 'wb') as f:
    pickle.dump({
        'rewards_static': rewards_static,
        'rewards_dynamic': rewards_dynamic,
        'lengths_static': lengths_static,
        'lengths_dynamic': lengths_dynamic,
        'losses_static': losses_static,
        'losses_dynamic': losses_dynamic
    }, f)

print("\nModels and results saved!")
print("- classical_dqn_static.pth")
print("- classical_dqn_dynamic.pth")
print("- classical_dqn_results.pkl")
