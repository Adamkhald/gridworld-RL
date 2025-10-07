# GridWorld 5x5

A simple 5x5 GridWorld environment for reinforcement learning experiments.

## Prerequisites

* Python 3.10+
* numpy
* matplotlib
* gym

Install packages with:

```bash
conda create --name stats_env python=3.10 numpy matplotlib gym
conda activate stats_env
```

## How it Works

* Agent starts at `(0,0)` and goal is `(4,4)`.
* Actions: `0=UP`, `1=RIGHT`, `2=DOWN`, `3=LEFT`.
* Reward: `-1` per step, `+10` for reaching the goal.
* Use `render_plot()` (function) to visualize.
* Run episodes using the Q-learnind and the modified one.
