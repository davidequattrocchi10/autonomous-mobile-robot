"""
Test and visualize dynamic obstacles with collision avoidance.

This demo shows:
- Different obstacle movement patterns
- Priority-based collision resolution
- Obstacles avoiding walls and each other
- Realistic warehouse scenario
"""

import sys
sys.path.append('.')

from src.environment.grid_world import GridWorld
from src.environment.obstacle_manager import ObstacleManager, ObstacleType
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.animation import FuncAnimation


def create_warehouse_environment():
    """
    Create a realistic warehouse layout.
    
    Layout includes:
    - Perimeter walls
    - Internal shelf rows
    - Aisles for navigation
    """
    env = GridWorld(width=30, height=30)
    
    # Perimeter walls
    for i in range(30):
        env.add_obstacle(0, i)  # Top wall
        env.add_obstacle(29, i)  # Bottom wall
        env.add_obstacle(i, 0)  # Left wall
        env.add_obstacle(i, 29)  # Right wall
    
    # Internal shelving (vertical rows)
    for col in [8, 14, 20]:
        for row in range(5, 25):
            if not (row >= 10 and row <= 15):  # Leave gaps for crossing
                env.add_obstacle(row, col)
    
    return env


def setup_realistic_obstacles(obs_manager):
    """
    Add realistic obstacles for warehouse scenario.
    
    Scenario:
    - Forklifts following delivery routes (waypoints)
    - Workers wandering (random walk)
    - Conveyor belt (linear motion)
    """
    
    # FORKLIFT 1: Delivery route (MEDIUM PRIORITY)
    forklift = obs_manager.add_obstacle(
        start_pos=(3, 3),
        obstacle_type=ObstacleType.WAYPOINT,
        size=1
    )
    obs_manager.obstacles[forklift].set_waypoints([
        (3, 3),   # Start
        (3, 25),  # Right side
        (27, 25), # Bottom right
        (27, 3),  # Bottom left
    ])

    
    # WORKERS: Random walk (LOW PRIORITY)
    worker_positions = [(7, 7), (15, 15), (22, 22)]
    for i, pos in enumerate(worker_positions, 1):
        worker = obs_manager.add_obstacle(
            start_pos=pos,
            obstacle_type=ObstacleType.RANDOM_WALK,
            size=1
        )
    

    # CONVEYOR BELT: Linear motion (HIGH PRIORITY)
    conveyor = obs_manager.add_obstacle(
        start_pos=(10, 3),
        obstacle_type=ObstacleType.LINEAR,
        size=1
    )

    print(f"\n  Total: {len(obs_manager.obstacles)} dynamic obstacles")


def visualize_warehouse_simulation():
    """
    Animate warehouse scenario with collision avoidance.
    
    Watch for:
    - Forklifts following routes
    - Workers wandering randomly
    - Workers YIELDING to forklifts (priority system!)
    - All obstacles avoiding walls
    """
    print("="*60)
    print("🏭 WAREHOUSE OBSTACLE SIMULATION")
    print("="*60)
    
    # Create environment and obstacles
    env = create_warehouse_environment()
    obs_manager = ObstacleManager(dt=0.1, cell_size=0.5)  # 100ms steps, 50cm cells
    setup_realistic_obstacles(obs_manager)
    
    print("\n🎬 Starting animation...")
    print("   - Forklifts (🟦 blue) have medium priority")
    print("   - Workers (🟥 red) have low priority")
    print("   - Conveyor (🟩 green) has high priority")
    print()
    
    # Setup animation
    fig, ax = plt.subplots(figsize=(14, 14))
    
    # Statistics tracking
    collision_counts = []
    
    def update_frame(frame):
        """Update function called for each frame."""
        ax.clear()
        
        # Update all obstacles (with collision avoidance!)
        obs_manager.update_all(env)
        
        # Create visualization
        display_grid = np.ones((env.height, env.width, 3))
        
        # Static obstacles (dark gray - walls and shelves)
        obstacle_mask = env.grid == 1
        display_grid[obstacle_mask] = np.array([0.2, 0.2, 0.2])
        
        # Dynamic obstacles (colored by type and priority)
        for obstacle in obs_manager.obstacles.values():
            for cell in obstacle.get_occupied_cells():
                if 0 <= cell[0] < env.height and 0 <= cell[1] < env.width:
                    if obstacle.obstacle_type == ObstacleType.WAYPOINT:
                        # Forklifts: Blue (high priority)
                        display_grid[cell] = np.array([0.3, 0.5, 1.0])
                    elif obstacle.obstacle_type == ObstacleType.RANDOM_WALK:
                        # Workers: Red (low priority)
                        display_grid[cell] = np.array([1.0, 0.4, 0.4])
                    elif obstacle.obstacle_type == ObstacleType.LINEAR:
                        # Conveyor: Green
                        display_grid[cell] = np.array([0.4, 1.0, 0.4])
        
        # Display
        ax.imshow(display_grid)
        ax.grid(True, which='minor', color='gray', linewidth=0.5, alpha=0.2)
        ax.set_xticks(np.arange(-0.5, env.width, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, env.height, 1), minor=True)
        
        # Get statistics
        stats = obs_manager.get_statistics()
        total_blocked = stats['total_moves_blocked']
        collision_counts.append(total_blocked)
        
        # Title with statistics
        title = f'Warehouse Simulation (Time: {frame * obs_manager.dt:.1f}s)\n'
        title += f'Total Blocked Moves: {total_blocked} | Active Obstacles: {stats["total_obstacles"]}'
        ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
        
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Legend
        legend_elements = [
            mpatches.Patch(color=[0.3, 0.5, 1.0], label='Forklift (Priority 2)'),
            mpatches.Patch(color=[1.0, 0.4, 0.4], label='Worker (Priority 1)'),
            mpatches.Patch(color=[0.2, 0.2, 0.2], label='Walls/Shelves'),
            mpatches.Patch(color=[0.4, 1.0, 0.4], label='Conveyor (Priority 3)'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10,
                 framealpha=0.9, edgecolor='black')
    
    # Create animation
    num_frames = 100
    anim = FuncAnimation(fig, update_frame, frames=num_frames, 
                        interval=200, repeat=True)
    
    plt.tight_layout()
    plt.show()
    
    # Print final statistics
    print("\n" + "="*60)
    print("📊 SIMULATION STATISTICS")
    print("="*60)
    
    stats = obs_manager.get_statistics()
    print(f"📦 Total obstacles: {stats['total_obstacles']}")
    print(f"🚫 Total blocked moves: {stats['total_moves_blocked']}")
    print()
    
    # Print individual obstacle statistics
    print("Per-obstacle statistics:")
    for obs in obs_manager.obstacles.values():
        print(f"  {obs.obstacle_type.value:15s} (ID {obs.id}): "
              f"{obs.moves_blocked:3d} blocked moves, "
              f"path length: {len(obs.trajectory)}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    visualize_warehouse_simulation()