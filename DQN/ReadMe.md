# DQN Grid World Training Project

## Project Structure

```
dqn_gridworld/
├── README.md
├── 01_environment.py          # Custom Gymnasium grid world environment
├── 02_classical_dqn.py        # Classical DQN implementation
├── 03_deepmind_dqn.py         # DeepMind-style DQN (Nature 2015)
├── 04_stable_baselines_dqn.py # Stable-Baselines3 DQN
└── 05_comparison.py           # Compare all three models
```

## File Descriptions

### 01_environment.py
Defines two custom Gymnasium environments:
- **StaticGridWorld**: Grid world with fixed agent, obstacles, and goal positions
- **DynamicGridWorld**: Grid world where the goal moves randomly each episode

Both environments are configurable for width and height.

### 02_classical_dqn.py
Implementation of the classical Deep Q-Network with:
- **Experience Replay Buffer**: Stores transitions for random sampling
- **Target Network**: Separate network for stable Q-value targets
- **Epsilon-Greedy Exploration**: Decaying exploration strategy
- **Neural Network**: Fully connected layers for Q-value approximation

### 03_deepmind_dqn.py
DeepMind's DQN approach (Nature 2015) featuring:
- **Convolutional Architecture**: Processes spatial grid information
- **Frame Stacking**: Concatenates multiple consecutive frames
- **Reward Clipping**: Normalizes rewards to [-1, 1]
- **Gradient Clipping**: Stabilizes training
- **Longer Training**: Extended episode counts for convergence

### 04_stable_baselines_dqn.py
Uses Stable-Baselines3 library:
- **Pre-built DQN**: Reliable, tested implementation
- **Vectorized Environments**: Parallel training capability
- **Built-in Logging**: TensorBoard integration
- **Easy Evaluation**: Standardized evaluation metrics

### 05_comparison.py
Compares the three trained models:
- Loads saved models from each implementation
- Evaluates performance on both static and dynamic environments
- Visualizes training curves and final policies
- Generates comparison metrics

## Model Evolution

### Classical DQN
The foundational algorithm that introduced:
1. Experience replay to break temporal correlations
2. Fixed Q-targets using a separate target network
3. Deep neural networks as function approximators

### DeepMind DQN
Enhanced the classical approach with:
1. Convolutional layers for spatial feature extraction
2. Frame stacking for temporal information
3. Additional stability techniques (reward/gradient clipping)
4. Architecture optimized for visual/spatial tasks

### Stable-Baselines3 DQN
Production-ready implementation featuring:
1. Optimized hyperparameters from extensive testing
2. Robust handling of edge cases
3. Integration with standard RL tooling
4. Proven reliability across diverse tasks

## Usage

Run notebooks in order:
1. `01_environment.py` - Set up environments
2. `02_classical_dqn.py` - Train classical DQN
3. `03_deepmind_dqn.py` - Train DeepMind DQN
4. `04_stable_baselines_dqn.py` - Train SB3 DQN
5. `05_comparison.py` - Compare results

Each training notebook saves models to disk for later evaluation.