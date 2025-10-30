# DeepMind DQN Implementation (Nature 2015)
# Features convolutional architecture, frame stacking, reward clipping

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
import pickle

import sys
sys.path.append('.')
from environment import StaticGridWorld, DynamicGridWorld

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class FrameStack:
    """Stacks consecutive frames for temporal information."""
    
    def __init__(self, width, height, num_frames=4):
        self.width = width
        self.height = height
        self.num_frames = num_frames
        self.frames = deque(maxlen=num_frames)
    
    def reset(self, initial_frame):
        """Reset with initial frame repeated."""
        frame_2d = initial_frame.reshape(self.width, self.height)
        for _ in range(self.num_frames):
            self.frames.append(frame_2d)
        return self.get_state()
    
    def update(self, new_frame):
        """Add new frame and return stacked state."""
        frame_2d = new_frame.reshape(self.width, self.height)
        self.frames.append(frame_2d)
        return self.get_state()
    
    def get_state(self):
        """Return stacked frames as numpy array."""
        return np.array(self.frames, dtype=np.float32)


class ReplayBuffer:
    """Experience replay buffer."""
    
    def __init__(self, capacity=50000):
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


class DeepMindQNetwork(nn.Module):
    """Convolutional Q-Network architecture inspired by DeepMind Nature paper."""
    
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
        
        # Calculate flattened size
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


class DeepMindDQN:
    """DeepMind DQN agent with frame stacking and additional stability features."""
    
    def __init__(
        self,
        width,
        height,
        action_dim,
        num_frames=4,
        learning_rate=1e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay_steps=10000,
        buffer_size=50000,
        batch_size=32,
        target_update_freq=1000
    ):
        self.width = width
        self.height = height
        self.action_dim = action_dim
        self.num_frames = num_frames
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_start = epsilon_start
        self.epsilon_decay_steps = epsilon_decay_steps
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Q-networks
        self.q_network = DeepMindQNetwork(num_frames, width, height, action_dim).to(device)
        self.target_network = DeepMindQNetwork(num_frames, width, height, action_dim).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.RMSprop(self.q_network.parameters(), lr=learning_rate, alpha=0.95, eps=0.01)
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
        """Perform one training step with gradient clipping."""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Reward clipping (DeepMind approach)
        rewards = np.clip(rewards, -1, 1)
        
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
        
        # Huber loss (more stable than MSE)
        loss = nn.SmoothL1Loss()(current_q, target_q)
        
        # Optimize with gradient clipping
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.optimizer.step()
        
        # Update target network
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return loss.item()
    
    def update_epsilon(self):
        """Linear epsilon decay."""
        epsilon_range = self.epsilon_start - self.epsilon_end
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon_start - (epsilon_range * self.steps / self.epsilon_decay_steps)
        )


def train_deepmind_dqn(env, agent, num_episodes=2000, verbose=True):
    """Train DeepMind DQN agent with frame stacking."""
    episode_rewards = []
    episode_lengths = []
    losses = []
    
    frame_stack = FrameStack(agent.width, agent.height, agent.num_frames)
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        state = frame_stack.reset(obs)
        episode_reward = 0
        episode_length = 0
        
        while True:
            action = agent.select_action(state, training=True)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state = frame_stack.update(next_obs)
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


def evaluate_deepmind_agent(env, agent, num_episodes=100):
    """Evaluate trained DeepMind agent."""
    total_rewards = []
    frame_stack = FrameStack(agent.width, agent.height, agent.num_frames)
    
    for _ in range(num_episodes):
        obs, _ = env.reset()
        state = frame_stack.reset(obs)
        episode_reward = 0
        
        while True:
            action = agent.select_action(state, training=False)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            state = frame_stack.update(next_obs)
            episode_reward += reward
            
            if terminated or truncated:
                break
        
        total_rewards.append(episode_reward)
    
    return np.mean(total_rewards), np.std(total_rewards)


# Training on Static Grid World
print("=" * 60)
print("Training DeepMind DQN on Static Grid World")
print("=" * 60)

env_static = StaticGridWorld(width=10, height=10)
action_dim = env_static.action_space.n

agent_static = DeepMindDQN(
    width=10,
    height=10,
    action_dim=action_dim,
    num_frames=4,
    learning_rate=1e-4,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.1,
    epsilon_decay_steps=10000,
    buffer_size=50000,
    batch_size=32,
    target_update_freq=1000
)

rewards_static, lengths_static, losses_static = train_deepmind_dqn(
    env_static, agent_static, num_episodes=2000, verbose=True
)

mean_reward, std_reward = evaluate_deepmind_agent(env_static, agent_static, num_episodes=100)
print(f"\nStatic Environment Evaluation: {mean_reward:.2f} ± {std_reward:.2f}")


# Training on Dynamic Grid World
print("\n" + "=" * 60)
print("Training DeepMind DQN on Dynamic Grid World")
print("=" * 60)

env_dynamic = DynamicGridWorld(width=10, height=10)

agent_dynamic = DeepMindDQN(
    width=10,
    height=10,
    action_dim=action_dim,
    num_frames=4,
    learning_rate=1e-4,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.1,
    epsilon_decay_steps=10000,
    buffer_size=50000,
    batch_size=32,
    target_update_freq=1000
)

rewards_dynamic, lengths_dynamic, losses_dynamic = train_deepmind_dqn(
    env_dynamic, agent_dynamic, num_episodes=2000, verbose=True
)

mean_reward, std_reward = evaluate_deepmind_agent(env_dynamic, agent_dynamic, num_episodes=100)
print(f"\nDynamic Environment Evaluation: {mean_reward:.2f} ± {std_reward:.2f}")


# Plot training curves
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

axes[0, 0].plot(rewards_static)
axes[0, 0].set_title('DeepMind DQN - Static Environment: Episode Rewards')
axes[0, 0].set_xlabel('Episode')
axes[0, 0].set_ylabel('Reward')
axes[0, 0].grid(True)

axes[0, 1].plot(rewards_dynamic)
axes[0, 1].set_title('DeepMind DQN - Dynamic Environment: Episode Rewards')
axes[0, 1].set_xlabel('Episode')
axes[0, 1].set_ylabel('Reward')
axes[0, 1].grid(True)

axes[1, 0].plot(losses_static)
axes[1, 0].set_title('DeepMind DQN - Static Environment: Training Loss')
axes[1, 0].set_xlabel('Training Step')
axes[1, 0].set_ylabel('Loss')
axes[1, 0].grid(True)

axes[1, 1].plot(losses_dynamic)
axes[1, 1].set_title('DeepMind DQN - Dynamic Environment: Training Loss')
axes[1, 1].set_xlabel('Training Step')
axes[1, 1].set_ylabel('Loss')
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig('deepmind_dqn_training.png')
plt.show()


# Save models and results
torch.save(agent_static.q_network.state_dict(), 'deepmind_dqn_static.pth')
torch.save(agent_dynamic.q_network.state_dict(), 'deepmind_dqn_dynamic.pth')

with open('deepmind_dqn_results.pkl', 'wb') as f:
    pickle.dump({
        'rewards_static': rewards_static,
        'rewards_dynamic': rewards_dynamic,
        'lengths_static': lengths_static,
        'lengths_dynamic': lengths_dynamic,
        'losses_static': losses_static,
        'losses_dynamic': losses_dynamic
    }, f)

print("\nModels and results saved!")
print("- deepmind_dqn_static.pth")
print("- deepmind_dqn_dynamic.pth")
print("- deepmind_dqn_results.pkl")