# UC Berkeley Pacman AI - Reinforcement Learning Project

## Quick Setup
1. Download project files from Berkeley AI website
2. Place files in a directory (e.g., `reinforcement/`)
3. Ensure Python 3.x is installed

## Modified Files

### 1. `valueIterationAgents.py`
Implements **Value Iteration** algorithm:
- `runValueIteration()` - Batch value iteration updates
- `computeQValueFromValues()` - Calculates Q(s,a) values
- `computeActionFromValues()` - Returns optimal action (policy)

### 2. `qlearningAgents.py`
Implements **Q-Learning** agent:
- `getQValue()` - Returns Q(state, action)
- `computeValueFromQValues()` - Returns max Q-value
- `computeActionFromQValues()` - Returns greedy action
- `getAction()` - Epsilon-greedy exploration
- `update()` - Q-learning update rule
- Bonus: `ApproximateQAgent` for feature-based Q-learning

### 3. `analysis.py`
Parameter tuning for different agent behaviors:
- **question2a**: Low discount, no noise → risk cliff for close exit
- **question2b**: Low discount, some noise → avoid cliff, close exit
- **question2c**: High discount, no noise → risk cliff for distant exit
- **question2d**: High discount, some noise → avoid cliff, distant exit
- **question2e**: High living reward → never terminate

## Running Tests
```bash
# Test Value Iteration
python autograder.py -q q1

# Test Analysis (parameter tuning)
python autograder.py -q q2

# Test Q-Learning
python autograder.py -q q3

# Run all tests
python autograder.py