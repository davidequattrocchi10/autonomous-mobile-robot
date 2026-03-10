"""
Compare BFS, DFS, A*, and RRT algorithms.

This script demonstrates the differences between algorithms
with clear, colorful visualizations.

Key comparison:
  BFS   → shortest path, explores many nodes (exhaustive, level-by-level)
  DFS   → not shortest, explores deeply (exhaustive, depth-first)
  A*    → shortest path, explores fewest nodes (heuristic-guided)
  RRT   → not guaranteed shortest, but finds a path quickly via random
           sampling — shines in cluttered and high-dimensional spaces
"""

import os
import sys
sys.path.append('.')

from src.environment.grid_world import GridWorld
from src.planning.graph_search import BFS, DFS, AStar
from src.planning.rrt import RRT
import matplotlib.pyplot as plt
import numpy as np


def create_test_scenario():
    """Create a test environment with obstacles."""
    env = GridWorld(width=20, height=20)
    
    # Add some obstacles to make it interesting
    env.add_obstacle(18, 5)
    env.add_obstacle(17, 6)
    env.add_obstacle(15, 7)
    env.add_obstacle(18, 14)
    env.add_obstacle(15, 12)
    env.add_obstacle(13, 12)
    env.add_obstacle(12, 12)

    # Vertical walls
    for i in range(5, 15):
        env.add_obstacle(i, 8)

    for i in range(12, 18):
        env.add_obstacle(i, 3)
    
    for i in range(6, 11):
        env.add_obstacle(i, 16)

    # Horizontal wall (with gap)
    for j in range(3, 10):
        if j != 6:  # Leave a gap
            env.add_obstacle(10, j)
    
    # Box obstacle
    env.add_obstacle_rect((15, 15), (17, 17))
    
    return env


def run_comparison():
    """Run all algorithms and compare results."""
    
    print("="*60)
    print("PATH PLANNING ALGORITHM COMPARISON")
    print("="*60)
    
    # Create environment
    env = create_test_scenario()
    start = (2, 2)
    goal = (18, 18)
    
    # Initialize algorithms
    # RRT uses seed=None so every run shows a different random tree —
    # this is intentional: it illustrates the probabilistic nature of RRT.
    algorithms = [
        BFS(env),
        DFS(env),
        AStar(env, heuristic='manhattan'),
        RRT(env, max_iterations=3000, goal_bias=0.15),
    ]
    
    # Run each algorithm
    results = []
    for algo in algorithms:
        print(f"\nRunning {algo.__class__.__name__}...")
        path = algo.search(start, goal)
        stats = algo.get_stats()
        results.append((algo, path, stats))
        
        # Print statistics
        print(f"  ✓ Success: {stats['success']}")
        print(f"  ✓ Path length: {stats['path_length']}")
        print(f"  ✓ Nodes expanded: {stats['nodes_expanded']}")
        print(f"  ✓ Nodes visited: {stats['nodes_visited']}")
        print(f"  ✓ Time: {stats['time_seconds']:.4f}s")
    
    # Visualize all results
    print("\n" + "="*60)
    print("VISUALIZING RESULTS...")
    print("="*60)
    
    
    fig, axes = plt.subplots(1, 4, figsize=(28, 8))
    
    for idx, (algo, path, stats) in enumerate(results):
        ax = axes[idx]
        env.render( start=start, goal=goal, path=path, visited=algo.visited, stats=stats, ax=ax, show_legend=False  )
        
    legend_elements = env.get_legend_elements()
    axes[2].legend(handles=legend_elements,
              loc='lower center',
              ncol=len(legend_elements),
              fontsize=10,
              frameon=True,
              bbox_to_anchor=(0.5, -0.18))
    plt.subplots_adjust(bottom=0.25)
    
    plt.tight_layout()
    
    os.makedirs('images', exist_ok=True)
    plt.savefig('images/algorithm_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    # Print analysis
    print("\n" + "="*60)
    print("ANALYSIS - What You're Seeing:")
    print("="*60)
    
    print("\n🔵 LIGHT BLUE = Explored nodes")
    print("   More blue = more exploration = less efficient")
    
    print("\n🟡 YELLOW/GOLD = Final path")
    print("   Shorter = better (when we want optimal path)")
    
    print("\n" + "="*60)
    print("ALGORITHM COMPARISON:")
    print("="*60)
    
    bfs_stats   = results[0][2]
    dfs_stats   = results[1][2]
    astar_stats = results[2][2]
    rrt_stats   = results[3][2]

    print(f"\n📊 BFS:")
    print(f"   Path length:    {bfs_stats['path_length']} ← OPTIMAL (shortest)")
    print(f"   Nodes explored: {bfs_stats['nodes_visited']} ← Explores MANY nodes")
    print(f"   ✓ Guarantees shortest path on unweighted grids")
    print(f"   ✗ Memory intensive (stores entire frontier)")

    print(f"\n📊 DFS:")
    print(f"   Path length:    {dfs_stats['path_length']} ← May be LONGER")
    print(f"   Nodes explored: {dfs_stats['nodes_visited']}")
    print(f"   ✓ Memory efficient (only stores current path)")
    print(f"   ✗ Doesn't guarantee shortest path")

    print(f"\n📊 A* (Manhattan):")
    print(f"   Path length:    {astar_stats['path_length']} ← OPTIMAL (same as BFS)")
    print(f"   Nodes explored: {astar_stats['nodes_visited']} ← FEWER than BFS!")
    print(f"   ✓ Guarantees shortest path")
    print(f"   ✓ Most efficient — best for known, static environments")

    print(f"\n📊 RRT:")
    if rrt_stats['success']:
        print(f"   Path length:    {rrt_stats['path_length']} ← NOT guaranteed optimal")
    else:
        print(f"   Path length:    N/A (path not found — try more iterations)")
    print(f"   Nodes in tree:  {rrt_stats['nodes_visited']} ← grows randomly, not exhaustively")
    print(f"   ✓ Probabilistically complete")
    print(f"   ✓ Shines in high-dimensional / continuous spaces")
    print(f"   ✗ Path is longer than necessary (use RRT* for optimality)")
    print(f"   ✗ Results vary between runs (probabilistic — try it again!)")

if __name__ == "__main__":
    run_comparison()