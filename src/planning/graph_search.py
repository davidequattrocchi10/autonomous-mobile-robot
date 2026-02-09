"""
Graph Search Algorithms for Path Planning

This module implements BFS, DFS, and A* for grid-based navigation.
All algorithms follow a common pattern but differ in frontier management.
"""

from collections import deque
import heapq
from typing import List, Tuple, Optional, Set, Dict
from abc import ABC, abstractmethod
import time


class PathPlanner(ABC):
    """
    Abstract base class for path planning algorithms.

    - Enforces consistent interface
    - Shares common code (DRY principle)
    - Makes it easy to add new algorithms
    """
    
    def __init__(self, environment):
        """
        Initialize planner with environment.
        
        Args:
            environment: GridWorld instance
        """
        self.env = environment
        self.path = None
        self.visited = set()
        self.stats = {}  # Store performance metrics
    
    @abstractmethod
    def search(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find path from start to goal.
        
        Returns:
            List of (row, col) tuples representing path, or None if no path exists
        """
        pass
    
    def reconstruct_path(self, parent: Dict, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Reconstruct path from parent pointers.
        
        This is the SAME for all algorithms!
        
        Example:
            parent = {(1,0): (0,0), (1,1): (1,0), (2,1): (1,1)}
            goal = (2,1)
            
            Trace back:
            (2,1) → parent[(2,1)] = (1,1)
            (1,1) → parent[(1,1)] = (1,0)  
            (1,0) → parent[(1,0)] = (0,0)
            (0,0) → None (start reached!)
            
            Reverse: [(0,0), (1,0), (1,1), (2,1)]
        """
        path = []
        current = goal
        
        while current is not None:
            path.append(current)
            current = parent.get(current)
        
        path.reverse()
        return path
    
    def get_stats(self) -> Dict:
        """Return statistics about last search."""
        return self.stats


class BFS(PathPlanner):
    """
    Breadth-First Search (BFS)
    
    Strategy: Explore layer by layer
    Data structure: Queue (FIFO)
    Guarantees: Shortest path in unweighted graphs
    
    Time complexity: O(V + E) where V=nodes, E=edges/links
    Space complexity: O(V) - can be memory-intensive!
    """
    
    def search(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """BFS implementation."""
        
        # Validation
        if not self.env.is_valid(*start):
            raise ValueError(f"Start position {start} is not valid!")
        if not self.env.is_valid(*goal):
            raise ValueError(f"Goal position {goal} is not valid!")
        
        # Initialize
        start_time = time.time()
        queue = deque([start])  # FIFO queue
        self.visited = {start}
        parent = {start: None}
        
        nodes_expanded = 0
        
        # Main search loop
        while queue:
            current = queue.popleft()  # Take FIRST element (FIFO)
            nodes_expanded += 1
            
            # Goal check
            if current == goal:
                self.path = self.reconstruct_path(parent, start, goal)
                self.stats = {
                    'algorithm': 'BFS',
                    'path_length': len(self.path),
                    'nodes_expanded': nodes_expanded,
                    'nodes_visited': len(self.visited),
                    'time_seconds': time.time() - start_time,
                    'success': True
                }
                return self.path
            
            # Expand neighbors
            for neighbor in self.env.get_neighbors(*current):
                if neighbor not in self.visited:
                    self.visited.add(neighbor)
                    parent[neighbor] = current
                    queue.append(neighbor)  # Add to END of queue
        
        # No path found
        self.stats = {
            'algorithm': 'BFS',
            'path_length': 0,
            'nodes_expanded': nodes_expanded,
            'nodes_visited': len(self.visited),
            'time_seconds': time.time() - start_time,
            'success': False
        }
        return None


class DFS(PathPlanner):
    """
    Depth-First Search
    
    Strategy: Explore one path deeply before backtracking
    Data structure: Stack (LIFO)
    Guarantees: Finds A path (not necessarily shortest)
    
    Time complexity: O(V + E) where V=nodes, E=edges/links
    Space complexity: O(h) where h=max depth - memory efficient!
    """
    
    def search(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """DFS implementation."""
        
        # Validation
        if not self.env.is_valid(*start):
            raise ValueError(f"Start position {start} is not valid!")
        if not self.env.is_valid(*goal):
            raise ValueError(f"Goal position {goal} is not valid!")
        
        # Initialize
        start_time = time.time()
        stack = [start]  # Use list as stack
        self.visited = {start}
        parent = {start: None}
        
        nodes_expanded = 0
        
        # Main search loop
        while stack:
            current = stack.pop()  # Take LAST element (LIFO)
            nodes_expanded += 1
            
            # Goal check
            if current == goal:
                self.path = self.reconstruct_path(parent, start, goal)
                self.stats = {
                    'algorithm': 'DFS',
                    'path_length': len(self.path),
                    'nodes_expanded': nodes_expanded,
                    'nodes_visited': len(self.visited),
                    'time_seconds': time.time() - start_time,
                    'success': True
                }
                return self.path
            
            # Expand neighbors
            for neighbor in self.env.get_neighbors(*current):
                if neighbor not in self.visited:
                    self.visited.add(neighbor)
                    parent[neighbor] = current
                    stack.append(neighbor)  # Add to END (becomes top of stack)
        
        # No path found
        self.stats = {
            'algorithm': 'DFS',
            'path_length': 0,
            'nodes_expanded': nodes_expanded,
            'nodes_visited': len(self.visited),
            'time_seconds': time.time() - start_time,
            'success': False
        }
        return None


class AStar(PathPlanner):
    """
    A* Search Algorithm
    
    Strategy: Best-first search guided by heuristic
    Data structure: Priority queue (min-heap)
    Guarantees: Shortest path (if heuristic is admissible)
    
    Key concept: f(n) = g(n) + h(n)
    - g(n): Cost from start to n (actual cost so far)
    - h(n): Estimated cost from n to goal (heuristic)
    - f(n): Total estimated cost of path through n
    
    Time complexity: O(E) with good heuristic
    Space complexity: O(V)
    """
    
    def __init__(self, environment, heuristic='manhattan'):
        """
        Initialize A* planner.
        
        Args:
            environment: GridWorld instance
            heuristic: 'manhattan', 'euclidean', or 'chebyshev'
        """
        super().__init__(environment)
        self.heuristic_type = heuristic
    
    def heuristic(self, pos: Tuple[int, int], goal: Tuple[int, int]) -> float:
        """
        Calculate heuristic (estimated cost to goal).
        
        Manhattan distance: |x1-x2| + |y1-y2|
        - Admissible for 4-connected grids
        
        Euclidean distance: sqrt((x1-x2)² + (y1-y2)²)
        - Straight-line distance
        - Admissible but less informed for grids
        
        Chebyshev distance: max(|x1-x2|, |y1-y2|)
        - Admissible for 8-connected grids
        - "King's move" in chess
        """
        if self.heuristic_type == 'manhattan':
            return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
        elif self.heuristic_type == 'euclidean':
            return ((pos[0] - goal[0])**2 + (pos[1] - goal[1])**2)**0.5
        elif self.heuristic_type == 'chebyshev':
            return max(abs(pos[0] - goal[0]), abs(pos[1] - goal[1]))
        else:
            raise ValueError(f"Unknown heuristic: {self.heuristic_type}")
    
    def search(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """A* implementation."""
        
        # Validation
        if not self.env.is_valid(*start):
            raise ValueError(f"Start position {start} is not valid!")
        if not self.env.is_valid(*goal):
            raise ValueError(f"Goal position {goal} is not valid!")
        
        # Initialize
        start_time = time.time()
        
        # g_score[n] = cost of cheapest path from start to n
        g_score = {start: 0}
        
        # f_score[n] = g_score[n] + h(n)
        f_score = {start: self.heuristic(start, goal)}
        
        # Priority queue: (f_score, counter, position)
        # counter prevents comparison of positions when f_scores are equal
        counter = 0
        open_set = [(f_score[start], counter, start)]
        
        self.visited = set()
        parent = {start: None}
        
        nodes_expanded = 0
        
        # Main search loop
        while open_set:
            # Get node with lowest f_score
            current_f, _, current = heapq.heappop(open_set)
            
            # Skip if already visited
            if current in self.visited:
                continue
            
            self.visited.add(current)
            nodes_expanded += 1
            
            # Goal check
            if current == goal:
                self.path = self.reconstruct_path(parent, start, goal)
                self.stats = {
                    'algorithm': f'A* ({self.heuristic_type})',
                    'path_length': len(self.path),
                    'nodes_expanded': nodes_expanded,
                    'nodes_visited': len(self.visited),
                    'time_seconds': time.time() - start_time,
                    'success': True
                }
                return self.path
            
            # Expand neighbors
            for neighbor in self.env.get_neighbors(*current):
                if neighbor in self.visited:
                    continue
                
                # Calculate tentative g_score
                tentative_g = g_score[current] + 1  # Cost to move to neighbor
                
                # If this is a better path to neighbor
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    parent[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal)
                    
                    counter += 1
                    heapq.heappush(open_set, (f_score[neighbor], counter, neighbor))
        
        # No path found
        self.stats = {
            'algorithm': f'A* ({self.heuristic_type})',
            'path_length': 0,
            'nodes_expanded': nodes_expanded,
            'nodes_visited': len(self.visited),
            'time_seconds': time.time() - start_time,
            'success': False
        }
        return None