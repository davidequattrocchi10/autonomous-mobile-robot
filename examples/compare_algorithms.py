"""
Compare BFS, DFS, and A* algorithms.

This script demonstrates the differences between algorithms
with clear, colorful visualizations.
"""

import sys
sys.path.append('.')

from src.environment.grid_world import GridWorld
from src.planning.graph_search import BFS, DFS, AStar
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.lines import Line2D


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
    algorithms = [
        BFS(env),
        DFS(env),
        AStar(env, heuristic='manhattan'),
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
    
    # Define colors (RGB format)
    COLOR_FREE = np.array([1.0, 1.0, 1.0])      # White
    COLOR_OBSTACLE = np.array([0.2, 0.2, 0.2])  # Dark gray  
    COLOR_VISITED = np.array([0.7, 0.9, 1.0])   # Light blue
    COLOR_PATH = np.array([1.0, 0.8, 0.0])      # Yellow/Gold
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    
    for idx, (algo, path, stats) in enumerate(results):
        ax = axes[idx]
        
        # Create RGB visualization
        display_grid = np.ones((env.height, env.width, 3))
        
        # Paint obstacles
        obstacle_mask = env.grid == 1
        display_grid[obstacle_mask] = COLOR_OBSTACLE
        
        # Paint ALL visited cells (this is key!)
        for pos in algo.visited:
            if env.grid[pos] == 0:  # Not an obstacle
                display_grid[pos] = COLOR_VISITED
        
        # Paint path ON TOP (darker/different color)
        if path:
            for pos in path:
                if pos != start and pos != goal and env.grid[pos] == 0:
                    display_grid[pos] = COLOR_PATH
        
        # Display
        ax.imshow(display_grid)
        ax.grid(True, which='minor', color='gray', linewidth=0.5, alpha=0.3)
        ax.set_xticks(np.arange(-0.5, env.width, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, env.height, 1), minor=True)
        
        # Mark start and goal
        ax.plot(start[1], start[0], 'o', color='green', markersize=15, 
               markeredgecolor='darkgreen', markeredgewidth=3, zorder=5)
        ax.plot(goal[1], goal[0], '*', color='red', markersize=20, 
               markeredgecolor='darkred', markeredgewidth=3, zorder=5)
        
        # Title with stats
        title = f"{stats['algorithm']}\n"
        title += f"Path Length: {stats['path_length']} | Nodes Explored: {stats['nodes_visited']}"
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Add a single shared legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='green', markersize=10, markeredgecolor='darkgreen', markeredgewidth=2, linestyle='None', label='Start'), 
        Line2D([0], [0], marker='*', color='red', markersize=12, markeredgecolor='darkred', markeredgewidth=2, linestyle='None', label='Goal'),
        # mpatches.Patch(color='green', label='Start'),
        # mpatches.Patch(color='red', label='Goal'),
        mpatches.Patch(color=COLOR_OBSTACLE, label='Obstacle'),
        mpatches.Patch(color=COLOR_VISITED, label='Explored Nodes'),
        mpatches.Patch(color=COLOR_PATH, label='Final Path'),
    ]
    
    plt.tight_layout()
    ax.legend(
        handles=legend_elements,
        loc='lower center',
        ncol=5,
        fontsize=12,
        frameon=True,
        bbox_to_anchor=(-0.5, -0.2)
    )
    plt.subplots_adjust(bottom=0.15)

    
    

    plt.savefig('algorithm_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ Saved visualization to: algorithm_comparison.png")
    plt.show()
    
    # Print analysis
    print("\n" + "="*60)
    print("ANALYSIS - What You're Seeing:")
    print("="*60)
    
    print("\n🔵 LIGHT BLUE = Explored nodes")
    print("   These are all cells the algorithm checked")
    print("   More blue = more exploration = less efficient")
    
    print("\n🟡 YELLOW/GOLD = Final path")
    print("   The actual route from start to goal")
    print("   Shorter = better (when we want optimal path)")
    
    print("\n" + "="*60)
    print("ALGORITHM COMPARISON:")
    print("="*60)
    
    bfs_stats = results[0][2]
    dfs_stats = results[1][2]
    astar_stats = results[2][2]
    
    print(f"\n📊 BFS:")
    print(f"   Path length: {bfs_stats['path_length']} ← OPTIMAL (shortest)")
    print(f"   Nodes explored: {bfs_stats['nodes_visited']} ← Explores MANY nodes")
    print(f"   Pattern: Spreads evenly like ripples in water")
    print(f"   ✓ Guarantees shortest path")
    print(f"   ✗ Memory intensive (explores many nodes)")
    
    print(f"\n📊 DFS:")
    print(f"   Path length: {dfs_stats['path_length']} ← May be LONGER")
    print(f"   Nodes explored: {dfs_stats['nodes_visited']}")
    print(f"   Pattern: Goes deep in one direction, then backtracks")
    print(f"   ✓ Memory efficient")
    print(f"   ✗ Doesn't guarantee shortest path")
    
    print(f"\n📊 A* (Manhattan):")
    print(f"   Path length: {astar_stats['path_length']} ← OPTIMAL (same as BFS)")
    print(f"   Nodes explored: {astar_stats['nodes_visited']} ← FEWER than BFS!")
    print(f"   Pattern: Heads toward goal intelligently")
    print(f"   ✓ Guarantees shortest path")
    print(f"   ✓ More efficient than BFS (explores fewer nodes)")
    print(f"   ✓ BEST CHOICE for known environments!")
    
    print("\n" + "="*60)
    print("KEY INSIGHT:")
    print("="*60)
    print("Notice how A* has:")
    print(f"  • SAME path length as BFS ({astar_stats['path_length']} steps)")
    print(f"  • But explored {bfs_stats['nodes_visited'] - astar_stats['nodes_visited']} FEWER nodes!")
    print(f"  • That's {100 * (bfs_stats['nodes_visited'] - astar_stats['nodes_visited']) / bfs_stats['nodes_visited']:.1f}% more efficient!")
    print("\nThis is why A* is the gold standard for pathfinding! 🏆")
    print("="*60)


if __name__ == "__main__":
    run_comparison()