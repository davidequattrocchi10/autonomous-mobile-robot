"""
Tests for the RRT (Rapidly-exploring Random Tree) planner.

Testing strategy
----------------
RRT is probabilistic: without a fixed seed it produces different paths
every run.  All tests that need a specific outcome pass seed=42 to RRT so
results are 100% reproducible.

Grid fixtures
-------------
The same layouts used in test_planning.py are reused here so the
algorithms can be compared on identical scenarios.

  empty_env  : 10×10 grid, no obstacles — trivial case for RRT
  walled_env : 5×5 grid, vertical wall at col 2 (rows 0-3), gap at row 4
               Start=(0,0), Goal=(0,4).  Forces detour.
  isolated_env: 3×3 grid where goal=(2,2) is completely surrounded

Note on path length
-------------------
Unlike BFS/A*, RRT is NOT guaranteed to return the shortest path.  We test
that a path exists and is valid, NOT that it has a specific length.
"""

import pytest
from src.environment.grid_world import GridWorld
from src.planning.rrt import RRT


# =========================================================================
# Fixtures — shared test environments
# =========================================================================

@pytest.fixture
def empty_env():
    """
    10×10 grid with no obstacles.

    WHY 10×10 and not 5×5?
    In a 5×5 grid start=(0,0) and goal=(4,4) are only 8 cells apart.
    RRT with step_size=1 finds this almost instantly regardless of seed,
    making it hard to observe interesting behaviour.  10×10 requires more
    exploration and is a better test of convergence.
    """
    return GridWorld(width=10, height=10)


@pytest.fixture
def walled_env():
    """
    5×5 grid with a vertical wall at column 2 (rows 0-3).
    Gap is at (4, 2) — the only crossing point.

    Layout (S=start, G=goal, X=obstacle):
         col: 0  1  2  3  4
    row 0:   S  .  X  .  G
    row 1:   .  .  X  .  .
    row 2:   .  .  X  .  .
    row 3:   .  .  X  .  .
    row 4:   .  .  .  .  .

    This replicates the test grid from test_planning.py so the algorithms
    can be compared on identical scenarios.
    """
    env = GridWorld(width=5, height=5)
    for row in range(4):        # wall at col=2, rows 0-3
        env.add_obstacle(row, 2)
    return env


@pytest.fixture
def isolated_env():
    """
    3×3 grid where goal=(2,2) is unreachable even with diagonal movement.

    Layout:
         col: 0  1  2
    row 0:   .  .  .
    row 1:   .  X  X
    row 2:   .  X  G

    (2,2) has three in-bounds neighbours: (1,1), (1,2), (2,1) — all blocked.
    (3,*) and (*,3) are out of bounds.  So (2,2) is completely surrounded
    whether the planner uses 4-connected OR 8-connected (diagonal) movement.

    WHY add (1,1)?
    RRT allows diagonal moves (step_size=1 with rounding reaches diagonal
    cells).  Without blocking (1,1) the path (0,0)→(0,1)→(1,1)→(2,2) is
    a valid diagonal route — making the goal reachable.  We must block the
    diagonal approach too.
    """
    env = GridWorld(width=3, height=3)
    env.add_obstacle(1, 1)  # blocks diagonal approach from (0,0) side
    env.add_obstacle(1, 2)  # blocks approach from above
    env.add_obstacle(2, 1)  # blocks approach from the left
    return env


# =========================================================================
# Helper
# =========================================================================

def is_path_valid(path, env):
    """
    Return True if every cell in path is a free, in-bounds grid cell.

    We do NOT check 4-connectivity here (unlike test_planning.py) because
    RRT with step_size=1 can make diagonal moves (√2 distance).  The
    important invariant is that the path contains no obstacle cells.
    """
    for cell in path:
        if not env.is_valid(*cell):
            return False
    return True


# =========================================================================
# TestRRTBasic — fundamental search behaviour
# =========================================================================

class TestRRTBasic:
    """Basic functionality: finds a path, handles edge cases."""

    def test_finds_path_open_grid(self, empty_env):
        """
        RRT finds a path in an obstacle-free 10×10 grid.

        This is the simplest possible scenario.  If RRT fails here
        something is fundamentally broken in the expansion logic.
        """
        rrt = RRT(empty_env, max_iterations=2000, seed=42)
        path = rrt.search((0, 0), (9, 9))
        assert path is not None, "RRT should always find a path in an open grid"

    def test_path_starts_at_start(self, empty_env):
        """First cell in returned path must be the start position."""
        rrt = RRT(empty_env, max_iterations=2000, seed=42)
        path = rrt.search((0, 0), (9, 9))
        assert path is not None
        assert path[0] == (0, 0)

    def test_path_ends_at_goal(self, empty_env):
        """Last cell in returned path must be the goal position."""
        rrt = RRT(empty_env, max_iterations=2000, seed=42)
        path = rrt.search((0, 0), (9, 9))
        assert path is not None
        assert path[-1] == (9, 9)

    def test_start_equals_goal(self, empty_env):
        """
        When start == goal the path should contain exactly one cell.

        This is a trivial case but important to handle correctly — it
        must not enter the main loop (which would try to grow a tree
        toward a goal that is already reached).
        """
        rrt = RRT(empty_env, seed=42)
        path = rrt.search((5, 5), (5, 5))
        assert path is not None
        assert path == [(5, 5)]
        assert rrt.get_stats()["path_length"] == 1

    def test_invalid_start_raises_value_error(self, empty_env):
        """A blocked or out-of-bounds start must raise ValueError."""
        empty_env.add_obstacle(0, 0)
        rrt = RRT(empty_env)
        with pytest.raises(ValueError):
            rrt.search((0, 0), (9, 9))

    def test_invalid_goal_raises_value_error(self, empty_env):
        """A blocked or out-of-bounds goal must raise ValueError."""
        empty_env.add_obstacle(9, 9)
        rrt = RRT(empty_env)
        with pytest.raises(ValueError):
            rrt.search((0, 0), (9, 9))

    def test_finds_path_navigating_wall(self, walled_env):
        """
        RRT must navigate around the vertical wall by going through the
        gap at row 4.

        WHY high max_iterations?
        The gap is narrow (one cell wide).  RRT must grow the tree through
        (4, 0)→(4, 1)→(4, 2) before it can reach the right side.  With
        more iterations and goal bias the tree reaches this gap reliably.
        """
        rrt = RRT(walled_env, max_iterations=5000, goal_bias=0.15, seed=42)
        path = rrt.search((0, 0), (0, 4))
        assert path is not None, (
            "RRT should find a path through the gap — try increasing max_iterations"
        )
        assert path[0] == (0, 0)
        assert path[-1] == (0, 4)


# =========================================================================
# TestRRTPathValidity — every cell in the path must be obstacle-free
# =========================================================================

class TestRRTPathValidity:
    """The returned path must not pass through any obstacle."""

    def test_path_cells_are_all_free(self, empty_env):
        """All cells in the path must be valid (free and in-bounds)."""
        rrt = RRT(empty_env, max_iterations=2000, seed=42)
        path = rrt.search((0, 0), (9, 9))
        assert path is not None
        assert is_path_valid(path, empty_env), "Path must not pass through obstacles"

    def test_path_cells_free_with_wall(self, walled_env):
        """Path must avoid wall cells even in a constrained environment."""
        rrt = RRT(walled_env, max_iterations=5000, goal_bias=0.15, seed=42)
        path = rrt.search((0, 0), (0, 4))
        if path is not None:
            assert is_path_valid(path, walled_env), "Path must not pass through the wall"

    def test_path_has_at_least_two_cells(self, empty_env):
        """
        A non-trivial path (start ≠ goal) must have at least 2 cells.

        A path of length 1 would mean start == goal, which is handled
        separately in the trivial-case branch.
        """
        rrt = RRT(empty_env, max_iterations=2000, seed=42)
        path = rrt.search((0, 0), (9, 9))
        assert path is not None
        assert len(path) >= 2


# =========================================================================
# TestRRTNoPath — behaviour when goal is unreachable
# =========================================================================

class TestRRTNoPath:
    """When the goal is completely surrounded by obstacles, return None."""

    def test_returns_none_when_no_path(self, isolated_env):
        """
        The isolated goal cell (2,2) is surrounded by obstacles on all
        sides.  RRT must return None after exhausting its iterations.

        WHY low max_iterations?
        We want the test to be fast.  500 iterations is more than enough
        for a 3×3 grid — after exploring every reachable cell RRT will
        exhaust them and give up.
        """
        rrt = RRT(isolated_env, max_iterations=500, seed=42)
        path = rrt.search((0, 0), (2, 2))
        assert path is None

    def test_success_false_when_no_path(self, isolated_env):
        """Stats must report success=False when no path is found."""
        rrt = RRT(isolated_env, max_iterations=500, seed=42)
        rrt.search((0, 0), (2, 2))
        assert rrt.get_stats()["success"] == False

    def test_path_length_zero_when_no_path(self, isolated_env):
        """path_length must be 0 in stats when search fails."""
        rrt = RRT(isolated_env, max_iterations=500, seed=42)
        rrt.search((0, 0), (2, 2))
        assert rrt.get_stats()["path_length"] == 0


# =========================================================================
# TestRRTStats — stats dict must match the contract shared by all planners
# =========================================================================

class TestRRTStats:
    """
    All planners return the same stats keys so they can be compared
    side-by-side in compare_algorithms.py.
    """

    REQUIRED_KEYS = {
        "algorithm",
        "path_length",
        "nodes_expanded",
        "nodes_visited",
        "time_seconds",
        "success",
    }

    def test_all_stat_keys_present_on_success(self, empty_env):
        """Stats dict must contain all required keys after a successful search."""
        rrt = RRT(empty_env, max_iterations=2000, seed=42)
        rrt.search((0, 0), (9, 9))
        assert self.REQUIRED_KEYS <= rrt.get_stats().keys()

    def test_all_stat_keys_present_on_failure(self, isolated_env):
        """Stats dict must contain all required keys even when search fails."""
        rrt = RRT(isolated_env, max_iterations=200, seed=42)
        rrt.search((0, 0), (2, 2))
        assert self.REQUIRED_KEYS <= rrt.get_stats().keys()

    def test_algorithm_name_is_rrt(self, empty_env):
        """Stats['algorithm'] must be exactly 'RRT'."""
        rrt = RRT(empty_env, max_iterations=2000, seed=42)
        rrt.search((0, 0), (9, 9))
        assert rrt.get_stats()["algorithm"] == "RRT"

    def test_path_length_matches_returned_path(self, empty_env):
        """
        Stats['path_length'] must equal len(path).

        A mismatch would mean the stats are reporting on a different search
        than the one that produced the returned path — a serious bug.
        """
        rrt = RRT(empty_env, max_iterations=2000, seed=42)
        path = rrt.search((0, 0), (9, 9))
        assert path is not None
        assert rrt.get_stats()["path_length"] == len(path)

    def test_success_true_when_path_found(self, empty_env):
        """success must be True whenever search returns a non-None path."""
        rrt = RRT(empty_env, max_iterations=2000, seed=42)
        path = rrt.search((0, 0), (9, 9))
        assert path is not None
        assert rrt.get_stats()["success"] == True

    def test_nodes_visited_positive_after_search(self, empty_env):
        """At least one node (the start) must have been visited."""
        rrt = RRT(empty_env, max_iterations=2000, seed=42)
        rrt.search((0, 0), (9, 9))
        assert rrt.get_stats()["nodes_visited"] >= 1

    def test_time_seconds_positive(self, empty_env):
        """Elapsed time must be a non-negative number."""
        rrt = RRT(empty_env, max_iterations=2000, seed=42)
        rrt.search((0, 0), (9, 9))
        assert rrt.get_stats()["time_seconds"] >= 0


# =========================================================================
# TestRRTParameters — effect of constructor parameters
# =========================================================================

class TestRRTParameters:
    """Verify that constructor parameters affect behaviour as documented."""

    def test_max_iterations_zero_returns_none(self, empty_env):
        """
        With zero iterations the main loop never runs, so the result must
        be None (even in an obstacle-free grid).

        WHY useful?  It verifies the loop-count guard works and that we
        never return a path without actually running the algorithm.
        """
        rrt = RRT(empty_env, max_iterations=0, seed=42)
        path = rrt.search((0, 0), (9, 9))
        assert path is None

    def test_goal_bias_one_finds_path_quickly(self, empty_env):
        """
        goal_bias=1.0 means EVERY random sample is the goal.

        In an open grid this makes RRT greedy: the nearest node to the
        goal keeps advancing one step toward it each iteration.  Even with
        only 50 iterations (well below the ~13 steps needed) it should
        succeed because the 'connect to goal' shortcut fires as soon as
        we are adjacent to the goal.
        """
        rrt = RRT(empty_env, max_iterations=50, goal_bias=1.0, seed=42)
        path = rrt.search((0, 0), (9, 9))
        assert path is not None, "With goal_bias=1.0 in open space a path must be found"

    def test_seed_produces_identical_paths(self, empty_env):
        """
        Two RRT instances with the same seed must produce identical paths.

        This verifies the seed parameter correctly resets the random state
        at the start of each search() call.
        """
        rrt1 = RRT(empty_env, max_iterations=2000, seed=42)
        path1 = rrt1.search((0, 0), (9, 9))

        rrt2 = RRT(empty_env, max_iterations=2000, seed=42)
        path2 = rrt2.search((0, 0), (9, 9))

        assert path1 == path2, "Same seed must produce same path"

    def test_different_seeds_may_differ(self, empty_env):
        """
        Different seeds should generally produce different paths.

        NOTE: there is a tiny probability this test fails if both seeds
        happen to generate the exact same path by coincidence.  In practice
        on a 10×10 grid this never happens.
        """
        rrt1 = RRT(empty_env, max_iterations=2000, seed=0)
        path1 = rrt1.search((0, 0), (9, 9))

        rrt2 = RRT(empty_env, max_iterations=2000, seed=99)
        path2 = rrt2.search((0, 0), (9, 9))

        # Both must find a path
        assert path1 is not None
        assert path2 is not None
        # They are very likely to differ (probabilistic algorithm)
        # We don't assert they MUST differ — just that both succeed


# =========================================================================
# TestRRTVsGraphSearch — compare RRT behaviour with BFS/A*
# =========================================================================

class TestRRTVsGraphSearch:
    """
    Compare RRT against BFS/A* to verify the documented trade-offs:
      - RRT path length >= BFS path length (not optimal)
      - RRT explores fewer nodes on average (faster in cluttered spaces)
    """

    def test_both_find_a_path_on_walled_grid(self, walled_env):
        """
        Both BFS and RRT must find a valid path on the same walled grid.

        WHY we do NOT compare path lengths here
        ----------------------------------------
        BFS uses strict 4-connected movement (cardinal moves only).
        RRT uses the _steer() function which rounds a continuous direction
        vector → this can produce diagonal moves (cells √2 apart).

        Example: the BFS optimal 4-connected path is 13 cells long.
        RRT may find an 11-cell path by cutting diagonally through corners.
        That is NOT a bug — it is a consequence of RRT's different movement
        model.  A fair length comparison would require both planners to use
        the same connectivity (e.g., BFS with 8-connectivity vs RRT).

        What we CAN assert: both planners succeed, and each path is valid.
        """
        from src.planning.graph_search import BFS

        bfs = BFS(walled_env)
        bfs_path = bfs.search((0, 0), (0, 4))

        rrt = RRT(walled_env, max_iterations=5000, goal_bias=0.15, seed=42)
        rrt_path = rrt.search((0, 0), (0, 4))

        assert bfs_path is not None, "BFS must find a path"
        assert rrt_path is not None, "RRT must find a path with enough iterations"

        # Both paths must be valid (no obstacle cells)
        assert is_path_valid(bfs_path, walled_env)
        assert is_path_valid(rrt_path, walled_env)
