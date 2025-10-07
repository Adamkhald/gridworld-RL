# GridWorld RL 🎮

A comprehensive reinforcement learning playground for exploring and mastering grid world environments. This project provides an interactive framework for experimenting with various RL algorithms, visualizing agent behavior, and understanding the fundamentals of intelligent decision-making in discrete spaces.

![Grid World Agent Navigation](https://miro.medium.com/v2/resize:fit:1400/1*3WbjfQu3IS9pcQN0MfJK2g.gif)
*Example of an RL agent learning to navigate a grid world environment*

## 🎯 About This Project

GridWorld RL is designed as an educational and experimental platform for understanding reinforcement learning concepts through grid-based environments. Built on top of the Gymnasium framework, it offers a hands-on approach to learning how agents interact with their environment, make decisions, and improve over time.

## 🤖 What Are Grid Worlds?

Grid worlds are discrete environments represented as 2D grids where agents navigate from cell to cell. Each cell can contain:
- **Empty spaces** - Free movement
- **Obstacles** - Blocking positions
- **Goals** - Target destinations
- **Rewards/Penalties** - Feedback signals

![MiniGrid Environment](https://gymnasium.farama.org/_images/minigrid-door-key-8x8.gif)
*Agent navigating a MiniGrid environment with doors and keys*

These simple yet powerful environments are perfect for understanding core RL concepts like:
- State representation
- Action selection
- Reward shaping
- Policy learning
- Value estimation

## 🧠 Reinforcement Learning Fundamentals

This project implements and compares multiple RL algorithms:

![Q-Learning Process](https://www.kdnuggets.com/wp-content/uploads/arya_reinforcement_learning_q_learning_1.gif)
*Visualization of Q-Learning agent exploring and learning optimal paths*

### **Q-Learning**
- Model-free, off-policy algorithm
- Learns optimal action-value function
- Balances exploration vs exploitation

### **SARSA (State-Action-Reward-State-Action)**
- On-policy temporal difference learning
- Updates based on actual agent behavior
- More conservative than Q-Learning

### **DQN (Deep Q-Network)**
- Neural network-based Q-Learning
- Handles larger state spaces
- Uses experience replay and target networks

## 🏗️ Built With Gymnasium

This project leverages **Gymnasium** (formerly OpenAI Gym), the standard framework for developing and comparing RL algorithms. Gymnasium provides:
- Standardized environment interface
- Consistent action and observation spaces
- Built-in rendering capabilities
- Extensive environment library compatibility

![Gymnasium Environments](Approximator-HomeWork\Upgrade-in-the-philosophy\output\gifs\test_episode_4.gif)
*Various grid world configurations in Gymnasium framework*

## ✨ Features

- **Multiple Grid World Environments**: Experiment with various layouts and challenges
- **Agent Comparison**: Run multiple algorithms side-by-side
- **Performance Tracking**: Visualize learning curves, rewards, and convergence
- **Customizable Parameters**: Tune hyperparameters like learning rate, discount factor, and exploration strategy
- **Interactive Visualizations**: Watch your agents learn in real-time
- **Comprehensive Documentation**: Auto-generated docs for all components

#

## 📊 Example Use Cases

- Learn RL fundamentals through hands-on experimentation
- Prototype new algorithm variations
- Benchmark different approaches on custom environments
- Visualize policy evolution over training
- Compare exploration strategies (ε-greedy, softmax, UCB)

## 👨‍💻 About the Author

**Adam Khald**  
2nd Year AI & Data Science Engineering Student  
ENSAM Meknes

This project was developed as part of my journey in understanding and implementing reinforcement learning algorithms. Feel free to reach out for collaborations or discussions about RL!

## 🤝 Contributing

Contributions are welcome! Whether it's:
- Adding new algorithms
- Creating custom environments
- Improving visualizations
- Fixing bugs
- Enhancing documentation

## 📚 Learning Resources

- [Sutton & Barto - Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book.html)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [Deep RL Course by Hugging Face](https://huggingface.co/learn/deep-rl-course)

## 📝 License

This project is open source and available under the MIT License.

---

*Happy Learning! 🎓🤖*