"""
Dynamic Obstacle Management

Handles moving obstacles with collision avoidance and priority-based updates.

Real-world modeling:
- People wandering (random walk)
- Forklifts on routes (waypoint following)
- AGVs with fixed paths (linear motion)

Key features:
- Collision detection (static and dynamic obstacles)
- Priority-based movement (forklifts > people)
- Predictive planning support
- Realistic time/speed calibration
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, TYPE_CHECKING
from enum import Enum

# Avoid circular import for type hints
if TYPE_CHECKING:
    from src.environment.grid_world import GridWorld

class ObstacleType(Enum):
    """
    Types of obstacle movement patterns (Enum for clarity)
    Each type models real-world entities:
    - STATIC: Parked carts, temporary blockages
    - RANDOM_WALK: Human workers, wandering pedestrians
    - LINEAR: Conveyor belts, simple AGVs
    - WAYPOINT: Forklifts, delivery robots with routes
    """
    STATIC = "static"           # Doesn't move
    RANDOM_WALK = "random_walk" 
    LINEAR = "linear"           
    WAYPOINT = "waypoint"      


class DynamicObstacle:
    """
    Represents a moving obstacle in the environment.
    
     Responsibilities:
    - Track own position and movement state
    - Update position based on movement type
    - Check validity of moves (collision avoidance)
    - Predict future position for proactive planning
    
    Each obstacle operates independently but receives world context
    to make informed movement decisions.
    """

    # Priority levels for collision resolution
    PRIORITY_MAP = {
        ObstacleType.STATIC: 0,
        ObstacleType.RANDOM_WALK: 1,
        ObstacleType.WAYPOINT: 2,
        ObstacleType.LINEAR: 3,
    }
    
    def __init__(self, 
                 obstacle_id: int,
                 start_pos: Tuple[int, int],
                 obstacle_type: ObstacleType = ObstacleType.RANDOM_WALK,
                 speed: float = 1.0,
                 size: int = 1):
        """
        Initialize dynamic obstacle.
        
        Args:
            obstacle_id: Unique identifier
            start_pos: Initial (row, col) position
            obstacle_type: Movement pattern type
            speed: Movement speed multiplier (1.0 = normal)
            size: Obstacle radius in cells (1 = single cell)
        """
        self.id = obstacle_id
        self.position = start_pos
        self.obstacle_type = obstacle_type
        self.speed = speed
        self.size = size
        self.priority = self.PRIORITY_MAP[obstacle_type]
        
        # Movement state
        self.velocity = (0, 0)  # Current direction (dr, dc)
        self.trajectory = [start_pos]  # Position history
        
        # For waypoint following
        self.waypoints: List[Tuple[int, int]] = []
        self.current_waypoint_idx = 0
        
        # Statistics
        self.moves_blocked = 0  # How many times movement was blocked
    
    def update(self, 
               grid: 'GridWorld',
               obstacle_manager: 'ObstacleManager') -> None:
        """
        Update obstacle position for one time step.
        
        This is the main update loop. It:
        1. Calculates intended next position
        2. Validates move (checks collisions)
        3. Updates position if valid, stays put if blocked
        
        Args:
            grid: GridWorld for static obstacle checking
            obstacle_manager: Manager for dynamic obstacle checking
        """
        if self.obstacle_type == ObstacleType.STATIC:
            return  # Don't move
        
        # Calculate intended new position based on type
        intended_pos = self._calculate_next_position(grid.grid.shape)
        
        # Validate move (collision checking)
        if self._is_move_valid(intended_pos, grid, obstacle_manager):
            # Move is safe, update position
            self.position = intended_pos
            self.trajectory.append(intended_pos)
        else:
            # Move blocked, stay in current position
            self.moves_blocked += 1
            # For LINEAR obstacles, reverse direction if hit wall
            if self.obstacle_type == ObstacleType.LINEAR:
                self.velocity = (-self.velocity[0], -self.velocity[1])


    def _calculate_next_position(self, 
                                 grid_shape: Tuple[int, int]) -> Tuple[int, int]:
        """
        Calculate intended next position based on obstacle type.
        
        This doesn't check validity, just calculates where the obstacle
        WANTS to go. Validation happens separately.
        
        Args:
            grid_shape: (height, width) of the grid
            
        Returns:
            Intended (row, col) position
        """
        if self.obstacle_type == ObstacleType.RANDOM_WALK:
            return self._random_walk_next(grid_shape)
        
        elif self.obstacle_type == ObstacleType.LINEAR:
            return self._linear_next(grid_shape)
        
        elif self.obstacle_type == ObstacleType.WAYPOINT:
            return self._waypoint_next(grid_shape)
        
        else:
            return self.position  # STATIC or unknown type, don't move
        
    
    def _random_walk_next(self, grid_shape: Tuple[int, int]) -> Tuple[int, int]:
        """
        Random walk: Choose random direction.
        
        Models: Pedestrians, wandering workers
        Includes option to stay still (realistic - people pause sometimes!)
        """
        # Possible moves: up, down, left, right, stay
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
        direction = directions[np.random.randint(0, len(directions))]
        
        new_pos = (self.position[0] + direction[0], 
                   self.position[1] + direction[1])
        
        self.velocity = direction
        return new_pos
    
    
    def _linear_next(self, grid_shape: Tuple[int, int]) -> Tuple[int, int]:
        """
        Linear motion: Continue in current direction.
        
        Models: Simple AGVs, conveyor belts
        Maintains momentum - doesn't change direction randomly
        """
        # Initialize velocity if not set
        if self.velocity == (0, 0):
            # Choose random initial direction
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            self.velocity = directions[np.random.randint(0, len(directions))]
        
        new_pos = (self.position[0] + self.velocity[0],
                   self.position[1] + self.velocity[1])
        
        return new_pos
    
    
    def _waypoint_next(self, grid_shape: Tuple[int, int]) -> Tuple[int, int]:
        """
        Waypoint following: Move toward next waypoint in sequence.
        
        Models: Forklifts, delivery robots with predefined routes
        Follows path: waypoint[0] → waypoint[1] → ... → waypoint[0] (loops)
        """
        if not self.waypoints:
            # No waypoints defined, stay still
            return self.position
        
        # Get current target waypoint
        target = self.waypoints[self.current_waypoint_idx]
        
        # Calculate direction toward target
        dr = target[0] - self.position[0]
        dc = target[1] - self.position[1]
        
        # Move one step toward target (prefer larger delta)
        if abs(dr) > abs(dc):
            move = (int(np.sign(dr)), 0)  # Move vertically
        elif abs(dc) > 0:
            move = (0, int(np.sign(dc)))  # Move horizontally
        else:
            move = (0, 0)  # At waypoint
        
        new_pos = (self.position[0] + move[0], 
                   self.position[1] + move[1])
        
        # Check if reached current waypoint
        if new_pos == target:
            # Advance to next waypoint (circular list)
            self.current_waypoint_idx = (self.current_waypoint_idx + 1) % len(self.waypoints)
        
        self.velocity = move
        return new_pos
    

    def _is_move_valid(self, 
                       new_pos: Tuple[int, int],
                       grid: 'GridWorld',
                       obstacle_manager: 'ObstacleManager') -> bool:
        """
        Check if move to new_pos is valid (collision-free).
        
        Checks three things:
        1. Within grid bounds?
        2. No static obstacle (wall)?
        3. No higher-priority dynamic obstacle?
        
        Args:
            new_pos: Intended (row, col) position
            grid: GridWorld for static obstacles
            obstacle_manager: Manager for dynamic obstacles
            
        Returns:
            True if move is safe, False if blocked
        """
        row, col = new_pos
        
        # Check 1: Grid bounds
        if not (0 <= row < grid.height and 0 <= col < grid.width):
            return False
        
        # Check 2: Static obstacles (walls, fixed objects)
        if grid.grid[row, col] == 1:
            return False
        
        # Check 3: Dynamic obstacles (other moving objects)
        # Only blocked by HIGHER priority obstacles
        blocking_obstacle = obstacle_manager.get_obstacle_at(new_pos)
        if blocking_obstacle is not None:
            # There's an obstacle at target position
            if blocking_obstacle.id == self.id:
                # It's me! (shouldn't happen, but safe)
                return True
            if blocking_obstacle.priority >= self.priority:
                # Higher or equal priority blocks me
                return False
        
        # All checks passed!
        return True
    
    
    def get_occupied_cells(self) -> List[Tuple[int, int]]:
        """
        Get cells occupied by THIS obstacle (just one obstacle)
        If size > 1, obstacle occupies multiple cells (larger object).
        Returns:
            List of (row, col) tuples -> all cells covered by this obstacle
        """
        cells = []
        for dr in range(-self.size + 1, self.size):
            for dc in range(-self.size + 1, self.size):
                cells.append((self.position[0] + dr, self.position[1] + dc))
        return cells
    
    def predict_position(self, steps_ahead: int) -> Tuple[int, int]:
        """
        Predict where this obstacle will be in N time steps.
        
        Used for: Proactive collision avoidance
        
        Prediction accuracy:
        - LINEAR: High (continues in same direction)
        - WAYPOINT: Medium (follows known path)
        - RANDOM_WALK: Low (unpredictable by nature)
        - STATIC: Perfect (doesn't move)
        
        Args:
            steps_ahead: Number of time steps to predict
            
        Returns:
            Predicted (row, col) position
        """
        if self.obstacle_type == ObstacleType.STATIC:
            return self.position
        
        elif self.obstacle_type == ObstacleType.LINEAR:
            # Simple linear extrapolation
            pred_row = self.position[0] + self.velocity[0] * steps_ahead
            pred_col = self.position[1] + self.velocity[1] * steps_ahead
            return (int(pred_row), int(pred_col))
        
        elif self.obstacle_type == ObstacleType.WAYPOINT:
            # Simulate waypoint following
            # (Simplified - assumes straight-line motion)
            if not self.waypoints:
                return self.position
            
            target = self.waypoints[self.current_waypoint_idx]
            # Rough estimate: move toward target
            dr = np.sign(target[0] - self.position[0]) * min(steps_ahead, abs(target[0] - self.position[0]))
            dc = np.sign(target[1] - self.position[1]) * min(steps_ahead, abs(target[1] - self.position[1]))
            return (self.position[0] + int(dr), self.position[1] + int(dc))
        
        else:
            # Random walk: can't predict accurately
            return self.position
    
    def set_waypoints(self, waypoints: List[Tuple[int, int]]) -> None:
        """
        Set waypoint path for WAYPOINT type obstacles.
        
        Args:
            waypoints: List of (row, col) positions to visit in order
        """
        self.waypoints = waypoints
        self.current_waypoint_idx = 0
    
    def __repr__(self) -> str:
        return (f"Obstacle(id={self.id}, pos={self.position}, "
                f"type={self.obstacle_type.value}, priority={self.priority})")


class ObstacleManager:
    """
    Manages collection of all dynamic obstacles.
    
    Responsibilities:
    - Create and remove obstacles
    - Update all obstacles in priority order
    - Answer collision queries for planners
    - Provide predictive information
    - Track simulation time
    
    This is the "world controller" for dynamic objects.
    """
    
    def __init__(self, dt: float = 0.1, cell_size: float = 0.5):
        """
        Initialize obstacle manager.
        
        Args:
            dt: Time step duration in seconds (default 100ms)
            cell_size: Size of one grid cell in meters (default 0.5m)
        """
        self.obstacles: Dict[int, DynamicObstacle] = {}
        self.next_id = 0
        self.time_step = 0
        
        # Time calibration
        self.dt = dt  # seconds per time step
        self.cell_size = cell_size  # meters per cell
    
    def add_obstacle(self, 
                     start_pos: Tuple[int, int],
                     obstacle_type: ObstacleType = ObstacleType.RANDOM_WALK,
                     speed: float = 1.0,
                     size: int = 1) -> int:
        """
        Add new dynamic obstacle to the environment.
        
        Args:
            start_pos: Initial (row, col) position
            obstacle_type: Movement pattern
            speed: Speed multiplier (1.0 = standard speed for type)
            size: Obstacle size in cells
            
        Returns:
            obstacle_id: Unique ID for this obstacle
        """
        obs_id = self.next_id
        self.next_id += 1
        
        obstacle = DynamicObstacle(obs_id, start_pos, obstacle_type, speed, size)
        self.obstacles[obs_id] = obstacle
        
        return obs_id
    
    def remove_obstacle(self, obstacle_id: int) -> None:
        """Remove obstacle from environment."""
        if obstacle_id in self.obstacles:
            del self.obstacles[obstacle_id]
    
    def update_all(self, grid: 'GridWorld') -> None:
        """
        Update all obstacles for one time step.
        
        Key feature: Updates in PRIORITY ORDER
        - Higher priority obstacles move first
        - Lower priority obstacles see updated positions
        - Prevents collisions through ordered updates
        
        Args:
            grid: GridWorld for collision checking
        """
        # Sort obstacles by priority (highest first)
        sorted_obstacles = sorted(
            self.obstacles.values(), 
            key=lambda obs: obs.priority, 
            reverse=True
        )
        
        # Update each obstacle in priority order
        for obstacle in sorted_obstacles:
            obstacle.update(grid, self)
        
        self.time_step += 1
    
    def get_all_occupied_cells(self) -> List[Tuple[int, int]]:
        """
        Get ALL cells occupied by ANY obstacle.
        
        Used by: Global path planners to know blocked areas
        
        Returns:
            List of (row, col) tuples for all occupied cells
        """
        occupied = []
        for obstacle in self.obstacles.values():
            occupied.extend(obstacle.get_occupied_cells())
        return occupied
    
    def is_cell_occupied(self, cell: Tuple[int, int]) -> bool:
        """
        Check if specific cell is currently occupied.
        
        Fast check for single-cell queries.
        
        Args:
            cell: (row, col) to check
            
        Returns:
            True if occupied, False if free
        """
        return cell in self.get_all_occupied_cells()
    
    def get_obstacle_at(self, cell: Tuple[int, int]) -> Optional[DynamicObstacle]:
        """
        Get the obstacle at specific cell (if any).
        
        Useful for: Identifying which obstacle is blocking a position
        
        Args:
            cell: (row, col) to query
            
        Returns:
            DynamicObstacle if present, None if cell is free
        """
        for obstacle in self.obstacles.values():
            if cell in obstacle.get_occupied_cells():
                return obstacle
        return None
    
    def predict_occupancy(self, steps_ahead: int) -> List[Tuple[int, int]]:
        """
        Predict which cells will be occupied N steps in future.
        
        Used for: Proactive path planning (avoid future collisions)
        
        Note: Prediction quality varies by obstacle type
        - LINEAR: Accurate
        - WAYPOINT: Good
        - RANDOM_WALK: Poor (inherently random)
        
        Args:
            steps_ahead: Number of time steps to predict
            
        Returns:
            List of predicted occupied (row, col) positions
        """
        predicted = []
        for obstacle in self.obstacles.values():
            pred_pos = obstacle.predict_position(steps_ahead)
            # Add all cells this obstacle will occupy (accounting for size)
            for dr in range(-obstacle.size + 1, obstacle.size):
                for dc in range(-obstacle.size + 1, obstacle.size):
                    predicted.append((pred_pos[0] + dr, pred_pos[1] + dc))
        return predicted
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about obstacle management.
        
        Useful for: Debugging, performance analysis
        """
        total_blocked = sum(obs.moves_blocked for obs in self.obstacles.values())
        
        return {
            'total_obstacles': len(self.obstacles),
            'time_step': self.time_step,
            'total_moves_blocked': total_blocked,
            'dt': self.dt,
            'cell_size': self.cell_size,
        }
    
    def reset(self) -> None:
        """Clear all obstacles and reset simulation."""
        self.obstacles = {}
        self.next_id = 0
        self.time_step = 0
    
    def __repr__(self) -> str:
        return f"ObstacleManager({len(self.obstacles)} obstacles, t={self.time_step})"