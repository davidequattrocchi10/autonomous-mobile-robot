"""
Test script to understand GridWorld behavior.

Run this to see how the environment works!
"""

import sys
# Ass the current directory to Python's module search path
sys.path.append('.')  

from src.environment.grid_world import GridWorld

# Create environment
print("Creating 10x10 grid world...")
env = GridWorld(width=10, height=10)
print(env)
print()

# Add some obstacles
print("Adding obstacles...")
env.add_obstacle(2, 2)
env.add_obstacle(2, 3)
env.add_obstacle(2, 4)
env.add_obstacle_rect((5, 5), (7, 7))  # Add a box
print(f"Total obstacles: {env.grid.sum()}")
print()

# Test validity
print("Testing validity checks:")
print(f"Is (0, 0) valid? {env.is_valid(0, 0)}")  # Should be True
print(f"Is (2, 2) valid? {env.is_valid(2, 2)}")  # Should be False (obstacle)
print(f"Is (20, 20) valid? {env.is_valid(20, 20)}")  # Should be False (out of bounds)
print()

# Test neighbors
print("Getting neighbors of (0, 0):")
neighbors = env.get_neighbors(0, 0)
print(f"4-connected neighbors: {neighbors}")
print()

print("Getting neighbors of (5, 5) with obstacles:")
neighbors = env.get_neighbors(5, 5)
print(f"Neighbors: {neighbors}")
print("(Some directions blocked by obstacles!)")
print()

# Visualize
print("Rendering visualization...")
env.render(
    start=(0, 0),
    goal=(9, 9),
    title="Test Grid World"
)