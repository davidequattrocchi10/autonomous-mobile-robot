"""
Unit Tests for GridWorld and ObstacleManager

These tests verify:
1. GridWorld: initialization, obstacle placement, validity checks, neighbor queries
2. ObstacleManager: obstacle lifecycle, occupancy queries, statistics

WHY we test this:
Every path-planning algorithm (BFS, DFS, A*) calls GridWorld.is_valid() and
get_neighbors() at every step. If those are wrong, the planners are wrong — silently.
Tests lock in the correct behavior so regressions are caught immediately.

Run with: pytest tests/test_environment.py -v
"""

import sys
sys.path.append('.')

import pytest
import numpy as np

from src.environment.grid_world import GridWorld
from src.environment.obstacle_manager import ObstacleManager, ObstacleType


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures: reusable environments shared across test classes
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def empty_env():
    """A clean 5×5 grid with no obstacles — blank slate for basic tests."""
    return GridWorld(width=5, height=5)


@pytest.fixture
def env_with_obstacles():
    """
    A 5×5 grid with:
      - Single obstacle at (1, 1)
      - Rectangular block from (3,3) to (4,4)
    """
    env = GridWorld(width=5, height=5)
    env.add_obstacle(1, 1)
    env.add_obstacle_rect((3, 3), (4, 4))
    return env


# ──────────────────────────────────────────────────────────────────────────────
# GridWorld: Initialization
# ──────────────────────────────────────────────────────────────────────────────

class TestGridWorldInitialization:

    def test_grid_shape_matches_dimensions(self, empty_env):
        """
        NumPy stores grids as (height, width) — i.e. grid[row][col].
        Verify the shape reflects this so array indexing works correctly.
        """
        assert empty_env.grid.shape == (5, 5)  # (height=5, width=5)

    def test_grid_starts_all_zeros(self, empty_env):
        """All cells must be free (0) at creation — no hidden obstacles."""
        assert np.all(empty_env.grid == 0)

    def test_width_and_height_stored_correctly(self):
        """Constructor args should map to the right dimension attributes."""
        env = GridWorld(width=10, height=7)
        assert env.width == 10
        assert env.height == 7
        assert env.grid.shape == (7, 10)  # (height, width)

    def test_repr_includes_dimensions(self, empty_env):
        """__repr__ should be human-readable — useful when debugging in the REPL."""
        r = repr(empty_env)
        assert "5x5" in r


# ──────────────────────────────────────────────────────────────────────────────
# GridWorld: Obstacle Placement
# ──────────────────────────────────────────────────────────────────────────────

class TestGridWorldObstacles:

    def test_add_single_obstacle_sets_cell_to_one(self, empty_env):
        """After add_obstacle(), the cell value must become 1 (obstacle)."""
        empty_env.add_obstacle(2, 3)
        assert empty_env.grid[2, 3] == 1

    def test_add_obstacle_out_of_bounds_raises_value_error(self, empty_env):
        """
        Adding an obstacle outside the grid must raise ValueError.
        WHY: Silent failure would place an obstacle "nowhere" and produce
        mysterious behaviour in planners downstream. Loud failures are safer.
        """
        with pytest.raises(ValueError):
            empty_env.add_obstacle(10, 0)   # Row out of bounds

        with pytest.raises(ValueError):
            empty_env.add_obstacle(0, 10)   # Col out of bounds

        with pytest.raises(ValueError):
            empty_env.add_obstacle(-1, 0)   # Negative index

    def test_add_obstacle_rect_fills_all_cells_in_region(self, empty_env):
        """
        Rectangular obstacles must fill every cell in [top_left, bottom_right].
        This models warehouse shelves, machines, and restricted zones.
        """
        empty_env.add_obstacle_rect((1, 1), (3, 3))

        # Every cell in the rectangle should be an obstacle
        for r in range(1, 4):
            for c in range(1, 4):
                assert empty_env.grid[r, c] == 1, f"Expected obstacle at ({r},{c})"

        # Cells just outside the rectangle must remain free
        assert empty_env.grid[0, 0] == 0   # Top-left corner
        assert empty_env.grid[4, 4] == 0   # Bottom-right corner

    def test_reset_clears_all_obstacles(self, env_with_obstacles):
        """
        reset() must remove every obstacle and return to a blank grid.
        Used when re-running planning experiments on a clean map.
        """
        env_with_obstacles.reset()
        assert np.all(env_with_obstacles.grid == 0)

    def test_get_grid_copy_is_independent_of_original(self, empty_env):
        """
        get_grid_copy() must return a deep copy, not a reference.
        WHY: Planning algorithms that modify the grid (e.g., for temporary marks)
        must not corrupt the real environment.
        """
        copy = empty_env.get_grid_copy()
        copy[0, 0] = 99          # Mutate the copy
        assert empty_env.grid[0, 0] == 0   # Original must be unchanged


# ──────────────────────────────────────────────────────────────────────────────
# GridWorld: Validity Checks
# ──────────────────────────────────────────────────────────────────────────────

class TestGridWorldValidation:

    def test_is_in_bounds_true_for_corners_and_center(self, empty_env):
        assert empty_env.is_in_bounds(0, 0) is True   # Top-left corner
        assert empty_env.is_in_bounds(4, 4) is True   # Bottom-right corner
        assert empty_env.is_in_bounds(2, 2) is True   # Center

    def test_is_in_bounds_false_outside_grid(self, empty_env):
        assert empty_env.is_in_bounds(-1, 0) is False  # Negative row
        assert empty_env.is_in_bounds(5,  0) is False  # Row == height (0-indexed)
        assert empty_env.is_in_bounds(0,  5) is False  # Col == width

    def test_is_valid_true_for_free_in_bounds_cell(self, empty_env):
        """
        A free, in-bounds cell is valid for robot movement.

        NOTE: is_valid() returns np.bool_ (a NumPy scalar), not Python's bool,
        because grid[row, col] == 0 is a NumPy comparison. We use == instead of
        'is' to compare values rather than object identity.
        """
        assert empty_env.is_valid(0, 0) == True  # noqa: E712

    def test_is_valid_false_for_obstacle_cell(self, env_with_obstacles):
        """An obstacle cell is NOT valid — the robot cannot occupy it."""
        assert env_with_obstacles.is_valid(1, 1) == False  # noqa: E712

    def test_is_valid_false_for_out_of_bounds(self, empty_env):
        """Out-of-bounds positions are never valid, regardless of content."""
        assert empty_env.is_valid(10, 10) is False
        assert empty_env.is_valid(-1, 0) is False


# ──────────────────────────────────────────────────────────────────────────────
# GridWorld: Neighbor Queries
# ──────────────────────────────────────────────────────────────────────────────

class TestGridWorldNeighbors:

    def test_corner_cell_has_two_4connected_neighbors(self, empty_env):
        """
        Corner (0,0) can only go right and down in 4-connectivity.
        Up and left are out of bounds.
        """
        neighbors = empty_env.get_neighbors(0, 0, connectivity=4)
        assert len(neighbors) == 2
        assert (1, 0) in neighbors   # Down
        assert (0, 1) in neighbors   # Right

    def test_center_cell_has_four_4connected_neighbors(self, empty_env):
        """
        Center of an obstacle-free grid has all 4 cardinal neighbors.
        None are blocked by walls or obstacles.
        """
        neighbors = empty_env.get_neighbors(2, 2, connectivity=4)
        assert len(neighbors) == 4

    def test_corner_cell_has_three_8connected_neighbors(self, empty_env):
        """
        In 8-connectivity, (0,0) adds the diagonal (1,1) to its neighbor list.
        Still limited to 3 because two sides are boundary walls.
        """
        neighbors = empty_env.get_neighbors(0, 0, connectivity=8)
        assert len(neighbors) == 3
        assert (1, 1) in neighbors   # Diagonal

    def test_center_cell_has_eight_8connected_neighbors(self, empty_env):
        """Center of obstacle-free grid has all 8 neighbors in 8-connectivity."""
        neighbors = empty_env.get_neighbors(2, 2, connectivity=8)
        assert len(neighbors) == 8

    def test_obstacle_cell_not_in_neighbors(self, empty_env):
        """
        Obstacle cells must NEVER appear in a neighbor list.
        WHY: If they did, planners would schedule moves into obstacles and crash.
        """
        empty_env.add_obstacle(1, 2)            # Place obstacle right of (1,1)
        neighbors = empty_env.get_neighbors(1, 1, connectivity=4)
        assert (1, 2) not in neighbors

    def test_invalid_connectivity_value_raises(self, empty_env):
        """
        Only connectivity=4 and connectivity=8 are supported.
        Any other value should fail loudly so bugs are obvious.
        """
        with pytest.raises(ValueError):
            empty_env.get_neighbors(0, 0, connectivity=6)


# ──────────────────────────────────────────────────────────────────────────────
# ObstacleManager
# ──────────────────────────────────────────────────────────────────────────────

class TestObstacleManager:

    @pytest.fixture
    def manager(self):
        """Fresh ObstacleManager for each test (no shared state)."""
        return ObstacleManager(dt=0.1, cell_size=0.5)

    @pytest.fixture
    def base_env(self):
        """10×10 obstacle-free grid — large enough that obstacles don't hit walls."""
        return GridWorld(width=10, height=10)

    def test_static_obstacle_does_not_move(self, manager, base_env):
        """
        STATIC obstacles model parked carts and fixed machinery — they never move.
        After several simulation steps, position must be unchanged.
        """
        obs_id = manager.add_obstacle(
            start_pos=(5, 5),
            obstacle_type=ObstacleType.STATIC
        )
        initial_pos = manager.obstacles[obs_id].position

        manager.update_all(base_env)   # Step 1
        manager.update_all(base_env)   # Step 2

        assert manager.obstacles[obs_id].position == initial_pos

    def test_is_cell_occupied_true_for_obstacle_cell(self, manager):
        """A cell that contains an obstacle must be reported as occupied."""
        manager.add_obstacle(start_pos=(3, 3), obstacle_type=ObstacleType.STATIC)
        assert manager.is_cell_occupied((3, 3)) is True

    def test_is_cell_occupied_false_for_empty_cell(self, manager):
        """A cell that contains no obstacle must be reported as free."""
        manager.add_obstacle(start_pos=(3, 3), obstacle_type=ObstacleType.STATIC)
        assert manager.is_cell_occupied((0, 0)) is False

    def test_get_statistics_has_all_required_keys(self, manager):
        """
        get_statistics() must always return these keys.
        The visualization scripts and future analysis code depend on this contract.
        """
        stats = manager.get_statistics()
        required_keys = {
            'total_obstacles', 'time_step',
            'total_moves_blocked', 'dt', 'cell_size'
        }
        assert required_keys.issubset(stats.keys())

    def test_statistics_track_obstacle_count_correctly(self, manager):
        """Obstacle count in statistics must match the number added."""
        assert manager.get_statistics()['total_obstacles'] == 0

        manager.add_obstacle((1, 1), ObstacleType.STATIC)
        assert manager.get_statistics()['total_obstacles'] == 1

        manager.add_obstacle((2, 2), ObstacleType.RANDOM_WALK)
        assert manager.get_statistics()['total_obstacles'] == 2

    def test_remove_obstacle_decreases_count(self, manager):
        """After remove_obstacle(), the obstacle should no longer be managed."""
        obs_id = manager.add_obstacle((5, 5), ObstacleType.STATIC)
        manager.remove_obstacle(obs_id)

        assert obs_id not in manager.obstacles
        assert manager.get_statistics()['total_obstacles'] == 0
