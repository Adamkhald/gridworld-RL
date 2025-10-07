# 🎮 GridWorld Reinforcement Learning

Implementation and comparison of Monte Carlo and Q-Learning algorithms for gridworld navigation.

## 📂 Project Files

- **`notebook.ipynb`**: Monte Carlo implementation with random environment
- **`generic_approach.ipynb`**: Generic gridworld approach testing both Monte Carlo and Q-Learning algorithms

## 📊 Results (Generic Approach - 10,000 Episodes)

### Training Performance

| Algorithm | Avg Reward | Success Rate | Avg Steps |
|-----------|------------|--------------|-----------|
| Monte Carlo | -193.59 | 0.4% | 192.4 |
| Q-Learning | -180.02 | 9.0% | 181.0 |

**Winner**: Q-Learning outperformed Monte Carlo across all metrics

### Test Performance (20 Episodes)
- Both algorithms: 0.0% success rate on new random environments

## 📁 Generated Outputs

```
output/
├── gifs/          # Training episode animations
├── plots/         # Performance visualizations
└── models/        # Saved Q-tables (.npy files)
```

```

Open and run the desired notebook file.
