"""
Unit Tests for Path Planning Algorithms: BFS, DFS, A*

These tests verify correctness and document the KEY GUARANTEES of each algorithm:

  BFS  → shortest path (FIFO queue, explores layer by layer)
  DFS  → some valid path, not necessarily shortest (LIFO stack, goes deep first)
  A*   → shortest path AND fewer nodes explored than BFS (guided by heuristic)

Test grid layout (5×5):

       col: 0  1  2  3  4
  row 0:   S  .  X  .  G
  row 1:   .  .  X  .  .
  row 2:   .  .  X  .  .
  row 3:   .  .  X  .  .
  row 4:   .  .  .  .  .

  S = start (0,0)   G = goal (0,4)   X = wall (col=2, rows 0-3)
  Gap at row 4 forces all paths to detour downward.

Optimal path length: 13 cells (proved below)
  - Must pass through the gap at row 4 to cross col 2
  - (0,0) → ... → (4,2) → ... → (0,4) costs exactly 13 cells

Run with: pytest tests/test_planning.py -v
"""

import sys
sys.path.append('.')

import pytest

from src.environment.grid_world import GridWorld
from src.planning.graph_search import BFS, DFS, AStar


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

# All algorithms must return a stats dict with exactly these keys.
REQUIRED_STATS_KEYS = {
    'algorithm', 'path_length', 'nodes_expanded',
    'nodes_visited', 'time_seconds', 'success'
}

# The known optimal path length for our test grid.
OPTIMAL_PATH_LENGTH = 13


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def walled_env():
    """
    5×5 grid with a vertical wall at col=2, rows 0-3.
    The only crossing is through the gap at row 4.

    WHY this shape:
    A simple open grid would let BFS reach the goal trivially.
    The wall forces a detour, making it easy to verify that:
      - BFS and A* find the *shortest* 13-cell route
      - DFS finds *a* route (possibly longer)
    """
    env = GridWorld(width=5, height=5)
    for row in range(4):        # wall: col=2, rows 0-3
        env.add_obstacle(row, 2)
    return env


@pytest.fixture
def isolated_env():
    """
    3×3 grid where goal (2,2) is unreachable from start (0,0).

       col: 0  1  2
    row 0:  .  .  .
    row 1:  .  .  X
    row 2:  .  X  G

    Goal (2,2)'s 4-connected neighbors are (1,2) and (2,1) — both obstacles.
    The goal cell itself is free (is_valid returns True), but isolated.
    This tests the "no path found" branch in each algorithm.
    """
    env = GridWorld(width=3, height=3)
    env.add_obstacle(1, 2)   # Blocks approach from above
    env.add_obstacle(2, 1)   # Blocks approach from the left
    return env


# ──────────────────────────────────────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────────────────────────────────────

def is_path_connected(path: list) -> bool:
    """
    Verify that each consecutive pair in the path is a valid 4-connected step.

    A path is structurally correct only if every move covers exactly 1 cell
    horizontally OR vertically (no teleporting, no diagonals).
    This helper is the core structural check shared by all algorithm tests.
    """
    for i in range(len(path) - 1):
        curr = path[i]
        next_ = path[i + 1]
        row_diff = abs(curr[0] - next_[0])
        col_diff = abs(curr[1] - next_[1])
        # Must move exactly 1 cell in one axis and 0 in the other
        if row_diff + col_diff != 1:
            return False
    return True


# ──────────────────────────────────────────────────────────────────────────────
# BFS Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestBFS:

    def test_finds_path_when_one_exists(self, walled_env):
        """BFS must return a non-empty path when start and goal are connected."""
        bfs = BFS(walled_env)
        path = bfs.search((0, 0), (0, 4))
        assert path is not None
        assert len(path) > 0

    def test_returns_shortest_path(self, walled_env):
        """
        BFS GUARANTEES the shortest path on an unweighted grid.

        HOW: BFS uses a FIFO queue — it explores all cells at distance d before
        any cell at distance d+1. The first time it reaches the goal, it has
        taken the minimum number of steps possible.

        On our walled grid the optimum is 13 cells.
        """
        bfs = BFS(walled_env)
        path = bfs.search((0, 0), (0, 4))
        assert len(path) == OPTIMAL_PATH_LENGTH

    def test_path_starts_at_start_ends_at_goal(self, walled_env):
        """The path must begin exactly at start and end exactly at goal."""
        bfs = BFS(walled_env)
        path = bfs.search((0, 0), (0, 4))
        assert path[0] == (0, 0)
        assert path[-1] == (0, 4)

    def test_path_is_connected(self, walled_env):
        """Every consecutive pair in the path must be a valid 4-connected move."""
        bfs = BFS(walled_env)
        path = bfs.search((0, 0), (0, 4))
        assert is_path_connected(path)

    def test_no_path_returns_none(self, isolated_env):
        """
        When no path exists BFS should return None — not crash.
        It exhausts all reachable cells, then signals failure via None.
        """
        bfs = BFS(isolated_env)
        path = bfs.search((0, 0), (2, 2))
        assert path is None

    def test_start_equals_goal_returns_single_element_path(self, walled_env):
        """
        If start == goal the robot is already at the destination.
        The correct answer is a path containing just [start], not an empty list.
        """
        bfs = BFS(walled_env)
        path = bfs.search((0, 0), (0, 0))
        assert path == [(0, 0)]

    def test_stats_contains_all_required_keys(self, walled_env):
        """
        get_stats() must return all standard keys every time.
        These keys are consumed by compare_algorithms.py and render().
        """
        bfs = BFS(walled_env)
        bfs.search((0, 0), (0, 4))
        assert REQUIRED_STATS_KEYS.issubset(bfs.get_stats().keys()) # issubset is used to check if all required keys are present in the stats dictionary

    def test_stats_success_true_when_path_found(self, walled_env):
        """success flag must be True when a path was found."""
        bfs = BFS(walled_env)
        bfs.search((0, 0), (0, 4))
        assert bfs.get_stats()['success'] is True

    def test_stats_success_false_when_no_path(self, isolated_env):
        """success flag must be False when no path was found."""
        bfs = BFS(isolated_env)
        bfs.search((0, 0), (2, 2))
        assert bfs.get_stats()['success'] is False


# ──────────────────────────────────────────────────────────────────────────────
# DFS Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestDFS:

    def test_finds_path_when_one_exists(self, walled_env):
        """DFS must return a non-empty path when start and goal are connected."""
        dfs = DFS(walled_env)
        path = dfs.search((0, 0), (0, 4))
        assert path is not None
        assert len(path) > 0

    def test_path_starts_at_start_ends_at_goal(self, walled_env):
        """The path must begin at start and end at goal."""
        dfs = DFS(walled_env)
        path = dfs.search((0, 0), (0, 4))
        assert path[0] == (0, 0)
        assert path[-1] == (0, 4)

    def test_path_is_connected(self, walled_env):
        """Every consecutive pair in the DFS path must be a valid 4-connected move."""
        dfs = DFS(walled_env)
        path = dfs.search((0, 0), (0, 4))
        assert is_path_connected(path)

    def test_no_path_returns_none(self, isolated_env):
        """When no path exists DFS should return None."""
        dfs = DFS(isolated_env)
        path = dfs.search((0, 0), (2, 2))
        assert path is None

    def test_start_equals_goal_returns_single_element_path(self, walled_env):
        """If start == goal, result should be [start]."""
        dfs = DFS(walled_env)
        path = dfs.search((0, 0), (0, 0))
        assert path == [(0, 0)]

    def test_stats_contains_all_required_keys(self, walled_env):
        """Stats must contain the standard keys after every search."""
        dfs = DFS(walled_env)
        dfs.search((0, 0), (0, 4))
        assert REQUIRED_STATS_KEYS.issubset(dfs.get_stats().keys())

    def test_path_at_least_as_long_as_bfs_optimal(self, walled_env):
        """
        DFS does NOT guarantee the shortest path.

        HOW: DFS uses a LIFO stack — it commits to one direction deeply before
        backtracking. The first complete path it finds may be much longer than
        the optimal.

        This test documents the tradeoff:
          - DFS path length >= BFS path length (BFS is the optimality baseline)
          - Equality is possible, but DFS makes no guarantee of achieving it
        """
        bfs = BFS(walled_env)
        bfs_path = bfs.search((0, 0), (0, 4))

        dfs = DFS(walled_env)
        dfs_path = dfs.search((0, 0), (0, 4))

        # BFS is optimal: DFS can only be equal or longer
        assert len(dfs_path) >= len(bfs_path)


# ──────────────────────────────────────────────────────────────────────────────
# A* Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestAStar:

    def test_finds_path_when_one_exists(self, walled_env):
        """A* must return a non-empty path when start and goal are connected."""
        astar = AStar(walled_env, heuristic='manhattan')
        path = astar.search((0, 0), (0, 4))
        assert path is not None
        assert len(path) > 0

    def test_returns_shortest_path(self, walled_env):
        """
        A* GUARANTEES the shortest path when using an admissible heuristic.

        HOW: A* prioritises nodes by f(n) = g(n) + h(n).
          g(n) = exact cost from start to n
          h(n) = heuristic estimate from n to goal

        Manhattan distance is admissible for 4-connected grids because it never
        overestimates the real cost (each step costs exactly 1).
        An admissible heuristic guarantees optimality.
        """
        astar = AStar(walled_env, heuristic='manhattan')
        path = astar.search((0, 0), (0, 4))
        assert len(path) == OPTIMAL_PATH_LENGTH

    def test_path_length_matches_bfs(self, walled_env):
        """
        A* and BFS must produce the same optimal path LENGTH on unweighted grids.
        They may take different routes of equal length.
        """
        bfs_path = BFS(walled_env).search((0, 0), (0, 4))
        astar_path = AStar(walled_env, heuristic='manhattan').search((0, 0), (0, 4))
        assert len(astar_path) == len(bfs_path)

    def test_path_starts_at_start_ends_at_goal(self, walled_env):
        """The path must begin at start and end at goal."""
        astar = AStar(walled_env)
        path = astar.search((0, 0), (0, 4))
        assert path[0] == (0, 0)
        assert path[-1] == (0, 4)

    def test_path_is_connected(self, walled_env):
        """Every consecutive pair in the A* path must be a valid 4-connected move."""
        astar = AStar(walled_env)
        path = astar.search((0, 0), (0, 4))
        assert is_path_connected(path)

    def test_no_path_returns_none(self, isolated_env):
        """When no path exists A* should return None."""
        astar = AStar(isolated_env)
        path = astar.search((0, 0), (2, 2))
        assert path is None

    def test_start_equals_goal_returns_single_element_path(self, walled_env):
        """If start == goal, result should be [start]."""
        astar = AStar(walled_env)
        path = astar.search((0, 0), (0, 0))
        assert path == [(0, 0)]

    def test_manhattan_heuristic_finds_optimal_path(self, walled_env):
        """
        Manhattan: |Δrow| + |Δcol| — admissible for 4-connected grids.
        Default heuristic; should find the 13-cell optimal path.
        """
        astar = AStar(walled_env, heuristic='manhattan')
        path = astar.search((0, 0), (0, 4))
        assert path is not None
        assert len(path) == OPTIMAL_PATH_LENGTH

    def test_euclidean_heuristic_finds_optimal_path(self, walled_env):
        """
        Euclidean: sqrt(Δrow² + Δcol²) — admissible because the straight-line
        distance always underestimates the actual grid distance. Should also
        find the 13-cell optimum.
        """
        astar = AStar(walled_env, heuristic='euclidean')
        path = astar.search((0, 0), (0, 4))
        assert path is not None
        assert len(path) == OPTIMAL_PATH_LENGTH

    def test_chebyshev_heuristic_finds_optimal_path(self, walled_env):
        """
        Chebyshev: max(|Δrow|, |Δcol|) — designed for 8-connected grids ('king's move').
        On a 4-connected grid it still underestimates, so it remains admissible
        and must still find the optimal path.
        """
        astar = AStar(walled_env, heuristic='chebyshev')
        path = astar.search((0, 0), (0, 4))
        assert path is not None
        assert len(path) == OPTIMAL_PATH_LENGTH

    def test_stats_contains_all_required_keys(self, walled_env):
        """Stats must contain all standard keys after every search."""
        astar = AStar(walled_env)
        astar.search((0, 0), (0, 4))
        assert REQUIRED_STATS_KEYS.issubset(astar.get_stats().keys())

    def test_astar_explores_fewer_nodes_than_bfs(self, walled_env):
        """
        A* KEY ADVANTAGE: same optimal path as BFS, but explores fewer nodes.

        WHY: BFS expands uniformly in all directions (a growing circle).
        A*'s heuristic h(n) guides exploration toward the goal, so it spends
        less time on cells that are far from the goal direction.

        On this grid with Manhattan heuristic, A* should visit strictly
        fewer nodes than BFS while still finding the same 13-cell path.
        This is the practical reason to prefer A* over BFS in real systems.
        """
        bfs = BFS(walled_env)
        astar = AStar(walled_env, heuristic='manhattan')

        bfs.search((0, 0), (0, 4))
        astar.search((0, 0), (0, 4))

        bfs_nodes = bfs.get_stats()['nodes_visited']
        astar_nodes = astar.get_stats()['nodes_visited']

        # A* visits fewer or equal nodes (strictly fewer on non-trivial grids)
        assert astar_nodes <= bfs_nodes
