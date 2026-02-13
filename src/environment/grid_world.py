"""
Grid World Environment Module

This module provides a 2D grid environment for robot navigation.
It handles obstacles, validation, and visualization.
"""


import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Set

class GridWorld:
    """
    A 2D grid environment for path planning.
    
    The grid uses matrix coordinates:
    - grid[0][0] is top-left
    - grid[row][col] where row increases downward
    
    Cell values:
    - 0: Free space (robot can move here)
    - 1: Obstacle (robot cannot move here)
    """
    
    def __init__(self, width: int = 10, height: int = 10):
        """
        Initialize grid environment.
        
        Args:
            width: Number of columns (x-axis)
            height: Number of rows (y-axis)
        """
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width), dtype=int)
        
    def add_obstacle(self, row: int, col: int) -> None:
        """Add a single obstacle at specified position."""
        if not self.is_in_bounds(row, col):
            raise ValueError(f"Position ({row}, {col}) out of bounds!")
        self.grid[row, col] = 1
    
    def add_obstacle_rect(self, top_left: Tuple[int, int], 
                         bottom_right: Tuple[int, int]) -> None:
        """
        Add rectangular obstacle.
        
        Useful for: walls, shelves, machine areas
        
        Args:
            top_left: (row, col) of top-left corner
            bottom_right: (row, col) of bottom-right corner
        """
        r1, c1 = top_left
        r2, c2 = bottom_right
        
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                if self.is_in_bounds(r, c):
                    self.grid[r, c] = 1
    
    def is_in_bounds(self, row: int, col: int) -> bool:
        """Check if position is within grid bounds."""
        return 0 <= row < self.height and 0 <= col < self.width
    
    def is_valid(self, row: int, col: int) -> bool:
        """Check if position is valid (in bounds and not obstacle)."""
        return self.is_in_bounds(row, col) and self.grid[row, col] == 0
    
    def get_neighbors(self, row: int, col: int, 
                     connectivity: int = 4) -> List[Tuple[int, int]]:
        """
        Get valid neighboring cells.
        
        Args:
            row, col: Current position
            connectivity: 4 or 8 (4-connected or 8-connected) -> possible movements
            
        Returns:
            List of (row, col) tuples for valid neighbors
        """
        neighbors = []
        
        if connectivity == 4:
            # 4-connected (no diagonals)
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        elif connectivity == 8:
            # 8-connected (includes diagonals)
            directions = [
                (-1, 0), (1, 0), (0, -1), (0, 1),  # Cardinal
                (-1, -1), (-1, 1), (1, -1), (1, 1)  # Diagonal
            ]
        else:
            raise ValueError("Connectivity must be 4 or 8")
        
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if self.is_valid(new_row, new_col):
                neighbors.append((new_row, new_col))
        
        return neighbors
    
    def get_grid_copy(self) -> np.ndarray:
        """Get a copy of the grid (prevents external modification)."""
        return self.grid.copy()
    
    def reset(self) -> None:
        """Clear all obstacles."""
        self.grid = np.zeros((self.height, self.width), dtype=int)
    
    def render(self, 
           start: Optional[Tuple[int, int]] = None,
           goal: Optional[Tuple[int, int]] = None,
           path: Optional[List[Tuple[int, int]]] = None,
           visited: Optional[Set[Tuple[int, int]]] = None,
           stats: Optional[dict] = None,
           ax= None,
           show_legend: bool = True) -> None:
        """
        Visualize the grid environment with clear color coding.
        
        Args:
            start: Starting position
            goal: Goal position
            path: Solution path
            visited: Explored cells
            stats: Dictionary of statistics to display in title
            ax: Matplotlib axis to draw on 
            show_legend: Whether to show color legend
        """
        
        # Create RGB image
        # Shape: (height, width, 3) for RGB
        display_grid = np.ones((self.height, self.width, 3))
        
        # Define colors (RGB format, values 0-1)
        COLOR_FREE = np.array([1.0, 1.0, 1.0])      # White
        COLOR_OBSTACLE = np.array([0.2, 0.2, 0.2])  # Dark gray
        COLOR_VISITED = np.array([0.7, 0.9, 1.0])   # Light blue
        COLOR_PATH = np.array([1.0, 0.8, 0.0])      # Yellow/Gold
        
        # Paint obstacles
        obstacle_mask = self.grid == 1
        display_grid[obstacle_mask] = COLOR_OBSTACLE
        
        # Paint visited cells (explored but not in path)
        if visited:
            for pos in visited:
                if self.grid[pos] == 0:  # Not an obstacle
                    display_grid[pos] = COLOR_VISITED
        
        # Paint path (overwrites visited for path cells)
        if path:
            for pos in path:
                if pos != start and pos != goal and self.grid[pos] == 0:
                    display_grid[pos] = COLOR_PATH

        ax.imshow(display_grid)
        
        # Add grid lines
        ax.set_xticks(np.arange(-0.5, self.width, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.height, 1), minor=True)
        ax.grid(which='minor', color='gray', linewidth=0.5, alpha=0.3)
        
        # Labels
        ax.set_xticks(range(self.width))
        ax.set_yticks(range(self.height))
        
        # Mark start (large green circle)
        if start:
            ax.plot(start[1], start[0], 'o', color='green', markersize=15, 
                markeredgecolor='darkgreen', markeredgewidth=3, label='Start', zorder=5)
        
        # Mark goal (large red star)
        if goal:
            ax.plot(goal[1], goal[0], '*', color='red', markersize=20, 
                markeredgecolor='darkred', markeredgewidth=3, label='Goal', zorder=5)

        title = f"{stats['algorithm']}\n"
        title += f"Path Length: {stats['path_length']} | Nodes Explored: {stats['nodes_visited']}"
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        if show_legend:
            legend_elements = self.get_legend_elements()

            plt.tight_layout()

            ax.legend(
                handles=legend_elements,
                loc='lower center',
                ncol=len(legend_elements),
                fontsize=12,
                frameon=True,
                bbox_to_anchor=(0.5, -0.15)
            )

            plt.subplots_adjust(bottom=0.18)

    def get_legend_elements(self):
        from matplotlib.lines import Line2D
        import matplotlib.patches as mpatches

        COLOR_OBSTACLE = np.array([0.2, 0.2, 0.2])
        COLOR_VISITED = np.array([0.7, 0.9, 1.0])
        COLOR_PATH = np.array([1.0, 0.8, 0.0])

        return [
            Line2D([0], [0], marker='o', color='green', markersize=10,
                markeredgecolor='darkgreen', markeredgewidth=2,
                linestyle='None', label='Start'),
            Line2D([0], [0], marker='*', color='red', markersize=12,
                markeredgecolor='darkred', markeredgewidth=2,
                linestyle='None', label='Goal'),
            mpatches.Patch(color=COLOR_OBSTACLE, label='Obstacle'),
            mpatches.Patch(color=COLOR_VISITED, label='Explored'),
            mpatches.Patch(color=COLOR_PATH, label='Path'),
        ]

    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"GridWorld({self.width}x{self.height}, obstacles={np.sum(self.grid)})"