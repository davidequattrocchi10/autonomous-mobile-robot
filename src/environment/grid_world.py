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
               title: str = "Grid World") -> None:
        """
        Visualize the grid environment.
        
        Args:
            start: Starting position
            goal: Goal position
            path: Solution path
            visited: Explored cells
            title: Plot title
        """
        # Create visualization grid
        display_grid = self.grid.copy().astype(float)
        
        # Mark visited cells (light gray)
        if visited:
            for pos in visited:
                if display_grid[pos] == 0:
                    display_grid[pos] = 0.3
        
        # Mark path (darker gray)
        if path:
            for pos in path:
                if pos != start and pos != goal:
                    display_grid[pos] = 0.6
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(display_grid, cmap='gray_r', vmin=0, vmax=1)
        
        # Add grid lines
        ax.set_xticks(np.arange(-0.5, self.width, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.height, 1), minor=True)
        ax.grid(which='minor', color='lightgray', linewidth=0.5)
        
        # Labels
        ax.set_xticks(range(self.width))
        ax.set_yticks(range(self.height))
        
        # Mark start (green circle)
        if start:
            ax.plot(start[1], start[0], 'go', markersize=15, 
                   markeredgecolor='darkgreen', markeredgewidth=2, label='Start')
        
        # Mark goal (red star)
        if goal:
            ax.plot(goal[1], goal[0], 'r*', markersize=20, 
                   markeredgecolor='darkred', markeredgewidth=2, label='Goal')
        
        if start or goal:
            ax.legend(loc='upper right')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"GridWorld({self.width}x{self.height}, obstacles={np.sum(self.grid)})"