# Autonomous Mobile Robot Navigation System

🤖 A comprehensive AGV/AMR navigation system featuring multiple path planning algorithms, reinforcement learning, and dynamic obstacle avoidance.

## 🎯 Project Overview

This project implements a complete autonomous navigation system for mobile robots (AGV/AMR) in warehouse and manufacturing environments. It compares classical planning algorithms (BFS, DFS, A*, RRT) with reinforcement learning approaches, and includes realistic considerations like sensor noise, battery consumption, and dynamic obstacles.

## ✨ Features

- **Multiple Planning Algorithms**: BFS, DFS, A*, RRT
- **Reinforcement Learning**: Q-Learning with adaptive behavior
- **Dynamic Obstacle Avoidance**: DWA (Dynamic Window Approach)
- **Sensor Simulation**: LIDAR, IMU with realistic noise models
- **Control Systems**: Path following, PID control
- **Performance Benchmarking**: Compare algorithms across scenarios
- **Interactive Visualizations**: See the robot in action

## 🚀 Quick Start
```bash
# Clone the repository
git clone https://github.com/davidequattrocchi10/autonomous-mobile-robot.git
cd autonomous-mobile-robot

# Install dependencies
pip install -r requirements.txt

# Run a demo
python examples/basic_navigation.py
```

## 📊 Project Status

- [x] Project structure
- [ ] Environment simulation
- [ ] Path planning algorithms
- [ ] Reinforcement learning
- [ ] Dynamic obstacle avoidance
- [ ] Sensor simulation
- [ ] Control systems
- [ ] Full system integration

## 🗂️ Project Structure
```
autonomous-mobile-robot/
├── src/              # Core source code
├── notebooks/        # Jupyter notebooks for experiments
├── tests/            # Unit tests
├── docs/             # Documentation
├── scenarios/        # Test scenarios
└── examples/         # Quick demos
```

## 📚 Documentation

See the [docs](./docs) folder for detailed documentation:
- [Architecture](./docs/architecture.md)
- [Algorithms](./docs/algorithms.md)
- [Getting Started](./docs/getting_started.md)

## 🤝 Contributing

This is a learning project, but feedback and suggestions are welcome!

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details

## 🙏 Acknowledgments

Built as a comprehensive learning project to understand autonomous robot navigation from first principles.

---

**Status**: 🚧 Under Active Development
```
