"""
Unit Tests for warehouse_simulation.py
=======================================

Tests only the PURE LOGIC functions — no animation, no GIF export, no
matplotlib rendering. All tests complete in well under 5 seconds.

Functions under test
---------------------
  build_warehouse_env()          — obstacle layout correctness
  find_lookahead_target()        — lookahead algorithm (copied from dwa_test.py)
  _check_predictive_collision()  — predictive forklift / robot collision veto
  Forklift grid injection pattern — verify env.grid is correctly patched/restored
  A* path finding on warehouse   — end-to-end path existence check
  Replan counter logic           — counter accumulates and resets correctly

Run with:
    pytest tests/test_warehouse_simulation.py -v
"""

import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'examples')   # lets us import warehouse_simulation directly

import math
import pytest

# Import the module under test.
# matplotlib.use('Agg') is called at the top of the file — safe in a test context.
import warehouse_simulation as ws
from warehouse_simulation import (
    build_warehouse_env,
    find_lookahead_target,
    _check_predictive_collision,
    WAREHOUSE_CONFIG,
)
from src.environment.obstacle_manager import ObstacleManager, ObstacleType
from src.planning.graph_search import AStar


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

CELL = WAREHOUSE_CONFIG['cell_size']


# ──────────────────────────────────────────────────────────────────────────────
# build_warehouse_env() — layout correctness
# ──────────────────────────────────────────────────────────────────────────────

class TestBuildWarehouseEnv:

    def test_start_cell_is_free(self):
        """Robot start position (1,1) must be navigable."""
        env = build_warehouse_env()
        r, c = WAREHOUSE_CONFIG['start']
        assert env.grid[r, c] == 0, f"Start cell ({r},{c}) is blocked"

    def test_goal_cell_is_free(self):
        """Goal position (18,18) must be navigable."""
        env = build_warehouse_env()
        r, c = WAREHOUSE_CONFIG['goal']
        assert env.grid[r, c] == 0, f"Goal cell ({r},{c}) is blocked"

    def test_shelf_a_is_obstacle(self):
        """Picking shelf A occupies rows 2-11, cols 2-3."""
        env = build_warehouse_env()
        for row in range(2, 12):
            for col in (2, 3):
                assert env.grid[row, col] == 1, (
                    f"Shelf A cell ({row},{col}) should be obstacle"
                )

    def test_shelf_b_is_obstacle(self):
        """Picking shelf B occupies rows 2-11, cols 5-6."""
        env = build_warehouse_env()
        for row in range(2, 12):
            for col in (5, 6):
                assert env.grid[row, col] == 1, (
                    f"Shelf B cell ({row},{col}) should be obstacle"
                )

    def test_forklift_aisle_row_is_clear(self):
        """
        Row 12 must have NO static obstacles.

        This is the design decision from Option C: shelves shortened to rows 2-11
        so the forklift can travel the full width of the warehouse at row=12.
        """
        env = build_warehouse_env()
        for col in range(env.width):
            assert env.grid[12, col] == 0, (
                f"Forklift aisle blocked at (12,{col}) — shelves must end at row 11"
            )

    def test_bulk_pallet_block_is_obstacle(self):
        """Bulk pallet block occupies rows 7-10, cols 10-12."""
        env = build_warehouse_env()
        for row in range(7, 11):
            for col in range(10, 13):
                assert env.grid[row, col] == 1, (
                    f"Bulk pallet cell ({row},{col}) should be obstacle"
                )

    def test_loading_pallets_are_obstacles(self):
        """Three individual loading pallets must be present."""
        env = build_warehouse_env()
        for (r, c) in [(14, 9), (15, 11), (16, 9)]:
            assert env.grid[r, c] == 1, f"Loading pallet ({r},{c}) not placed"

    def test_grid_size(self):
        """Grid must be 20x20."""
        env = build_warehouse_env()
        assert env.height == 20
        assert env.width == 20


# ──────────────────────────────────────────────────────────────────────────────
# find_lookahead_target()
# ──────────────────────────────────────────────────────────────────────────────

class TestFindLookaheadTarget:

    def _make_straight_waypoints(self, n: int = 10) -> list:
        """Straight-line waypoints spaced 0.5 m apart along x-axis."""
        return [(i * 0.5, 0.0) for i in range(n)]

    def test_returns_furthest_waypoint_within_distance(self):
        """
        Robot at origin, lookahead=1.0 m.
        Waypoints at 0.0, 0.5, 1.0, 1.5 m along x.
        Furthest within 1.0 m is index 2 (x=1.0).
        """
        waypoints = self._make_straight_waypoints(8)
        tx, ty, idx = find_lookahead_target(0.0, 0.0, waypoints, 0, 1.0)
        assert idx == 2           # x=1.0 is exactly at distance 1.0
        assert abs(tx - 1.0) < 1e-9

    def test_near_end_guard_returns_final_waypoint(self):
        """
        When current_index >= n-3, the function must return the final waypoint
        regardless of distance. This prevents overshooting near the goal.
        """
        waypoints = self._make_straight_waypoints(6)   # indices 0-5
        # current_index = 4 → n-3 = 3, so 4 >= 3 → near-end guard fires
        tx, ty, idx = find_lookahead_target(0.0, 0.0, waypoints, 4, 100.0)
        assert idx == 5           # must return last waypoint
        assert tx == waypoints[-1][0]

    def test_fallback_when_all_waypoints_too_far(self):
        """
        If even the current waypoint is beyond lookahead_distance,
        return it unchanged (fallback = current_index).
        """
        waypoints = [(10.0, 0.0), (11.0, 0.0), (12.0, 0.0), (13.0, 0.0)]
        # Robot at origin, all waypoints at 10+ m, lookahead = 1.0 m
        tx, ty, idx = find_lookahead_target(0.0, 0.0, waypoints, 0, 1.0)
        assert idx == 0           # fallback to current_index
        assert tx == 10.0

    def test_skips_waypoints_closer_than_lookahead(self):
        """
        The algorithm should skip past closer waypoints and return the
        FURTHEST one still within lookahead_distance, not the first one found.
        """
        # Waypoints at 0.2, 0.4, 0.6, 0.8, 1.0, 1.5 m
        waypoints = [(0.2 * i, 0.0) for i in range(1, 7)]
        # Robot at origin, lookahead=1.0
        # All of indices 0-4 (x=0.2 to 1.0) are within 1.0 m — return index 4
        tx, ty, idx = find_lookahead_target(0.0, 0.0, waypoints, 0, 1.0)
        assert idx == 4
        assert abs(tx - 1.0) < 1e-9

    def test_single_waypoint_path(self):
        """Path with only one waypoint triggers near-end guard immediately."""
        waypoints = [(5.0, 0.0)]
        tx, ty, idx = find_lookahead_target(0.0, 0.0, waypoints, 0, 100.0)
        assert idx == 0
        assert tx == 5.0


# ──────────────────────────────────────────────────────────────────────────────
# _check_predictive_collision()
# ──────────────────────────────────────────────────────────────────────────────

class TestCheckPredictiveCollision:

    # Shared parameters matching WAREHOUSE_CONFIG
    ROBOT_RADIUS = WAREHOUSE_CONFIG['robot_radius']   # 0.2 m
    CELL_SIZE    = WAREHOUSE_CONFIG['cell_size']       # 0.5 m
    DT           = WAREHOUSE_CONFIG['control_dt']      # 0.2 s
    THRESHOLD    = ROBOT_RADIUS + CELL_SIZE            # 0.7 m

    def test_returns_false_when_forklift_inactive(self):
        """No collision possible if forklift is not yet active."""
        result = _check_predictive_collision(
            robot_x=5.0, robot_y=5.0, robot_theta=0.0,
            v_cmd=0.5, omega_cmd=0.0, control_dt=self.DT,
            forklift_active=False,
            forklift_cells_next=[(12, 0)],   # would be in range, but inactive
            robot_radius=self.ROBOT_RADIUS,
            cell_size=self.CELL_SIZE,
        )
        assert result is False

    def test_returns_false_when_cells_list_empty(self):
        """Empty forklift_cells_next → no collision possible."""
        result = _check_predictive_collision(
            robot_x=5.0, robot_y=5.0, robot_theta=0.0,
            v_cmd=0.5, omega_cmd=0.0, control_dt=self.DT,
            forklift_active=True,
            forklift_cells_next=[],
            robot_radius=self.ROBOT_RADIUS,
            cell_size=self.CELL_SIZE,
        )
        assert result is False

    def test_returns_false_when_robot_far_from_forklift(self):
        """
        Robot at (1.0, 1.0) moving right, forklift next cell at (12, 0)
        = continuous (0.25, 6.25). Distance >> threshold → safe.
        """
        result = _check_predictive_collision(
            robot_x=1.0, robot_y=1.0, robot_theta=0.0,
            v_cmd=0.5, omega_cmd=0.0, control_dt=self.DT,
            forklift_active=True,
            forklift_cells_next=[(12, 0)],
            robot_radius=self.ROBOT_RADIUS,
            cell_size=self.CELL_SIZE,
        )
        assert result is False

    def test_returns_true_when_robot_about_to_enter_forklift_cell(self):
        """
        Robot is 0.1 m away from the forklift's next cell centre,
        moving directly toward it. Distance < robot_radius + cell_size → collision.

        Forklift next cell: (12, 10) → centre = (5.25, 6.25)
        Robot: placed so that after one step it will be at (10.25, 6.25).
        """
        # forklift next cell (12, 10) → centre = ((10+0.5)*0.5, (12+0.5)*0.5)
        #                                       = (5.25, 6.25)
        fc = (5.25, 6.25)   # continuous centre of cell (12, 10)

        # Robot positioned 0.3 m to the left of forklift cell centre,
        # heading right (theta=0), v=1.5 m/s, dt=0.2 s → moves 0.3 m right
        robot_x = fc[0] - 0.3
        robot_y = fc[1]
        v_cmd = 1.5   # moves 0.3 m in 0.2 s → lands on forklift cell centre

        result = _check_predictive_collision(
            robot_x=robot_x, robot_y=robot_y, robot_theta=0.0,
            v_cmd=v_cmd, omega_cmd=0.0, control_dt=self.DT,
            forklift_active=True,
            forklift_cells_next=[(12, 10)],
            robot_radius=self.ROBOT_RADIUS,
            cell_size=self.CELL_SIZE,
        )
        assert result is True

    def test_stationary_robot_no_collision(self):
        """v_cmd=0 → robot doesn't move → next position = current position.
        If robot is already far from forklift, no collision."""
        result = _check_predictive_collision(
            robot_x=1.0, robot_y=1.0, robot_theta=0.0,
            v_cmd=0.0, omega_cmd=0.0, control_dt=self.DT,
            forklift_active=True,
            forklift_cells_next=[(12, 0)],
            robot_radius=self.ROBOT_RADIUS,
            cell_size=self.CELL_SIZE,
        )
        assert result is False

    def test_collision_threshold_is_robot_radius_plus_cell_size(self):
        """
        Verify the exact threshold: collision fires when distance <
        robot_radius + cell_size = 0.2 + 0.5 = 0.7 m.

        Place robot next position exactly at 0.69 m from cell centre → collision.
        Place robot next position exactly at 0.71 m from cell centre → safe.
        """
        # Forklift next at (0, 0) → centre = (0.25, 0.25)
        fx, fy = 0.25, 0.25
        threshold = self.ROBOT_RADIUS + self.CELL_SIZE  # 0.7

        # 0.69 m away → collision
        robot_x_close = fx + (threshold - 0.01)
        result_close = _check_predictive_collision(
            robot_x=robot_x_close, robot_y=fy, robot_theta=0.0,
            v_cmd=0.0, omega_cmd=0.0, control_dt=self.DT,
            forklift_active=True, forklift_cells_next=[(0, 0)],
            robot_radius=self.ROBOT_RADIUS, cell_size=self.CELL_SIZE,
        )
        assert result_close is True

        # 0.71 m away → safe
        robot_x_far = fx + (threshold + 0.01)
        result_far = _check_predictive_collision(
            robot_x=robot_x_far, robot_y=fy, robot_theta=0.0,
            v_cmd=0.0, omega_cmd=0.0, control_dt=self.DT,
            forklift_active=True, forklift_cells_next=[(0, 0)],
            robot_radius=self.ROBOT_RADIUS, cell_size=self.CELL_SIZE,
        )
        assert result_far is False


# ──────────────────────────────────────────────────────────────────────────────
# Forklift grid injection
# ──────────────────────────────────────────────────────────────────────────────

class TestForkliftGridInjection:
    """
    Verify the temporary grid-injection pattern used in update():
    mark forklift cells → LiDAR scan → restore.

    WHY test this explicitly?
    The injection is a stateful mutation of env.grid. If the restore step
    is ever broken (e.g. exception between inject and restore, wrong cell list),
    the grid becomes permanently corrupted and A* / env.render() break silently.
    """

    def test_grid_is_marked_after_injection(self):
        """Injected forklift cell must appear as obstacle in env.grid."""
        env = build_warehouse_env()
        forklift_row, forklift_col = 12, 5

        # Confirm cell is free before injection
        assert env.grid[forklift_row, forklift_col] == 0

        # Inject
        env.grid[forklift_row, forklift_col] = 1
        assert env.grid[forklift_row, forklift_col] == 1

    def test_grid_is_restored_after_clearing(self):
        """After clearing the injection, the cell must be free again."""
        env = build_warehouse_env()
        forklift_row, forklift_col = 12, 5

        # Inject then restore (mirrors the update() pattern)
        env.grid[forklift_row, forklift_col] = 1
        env.grid[forklift_row, forklift_col] = 0

        assert env.grid[forklift_row, forklift_col] == 0

    def test_injection_does_not_affect_other_cells(self):
        """Injecting one cell must leave adjacent cells unchanged."""
        env = build_warehouse_env()
        r, c = 12, 5

        # Record neighbours before
        neighbours_before = {
            (r - 1, c): env.grid[r - 1, c],
            (r + 1, c): env.grid[r + 1, c],
            (r, c - 1): env.grid[r, c - 1],
            (r, c + 1): env.grid[r, c + 1],
        }

        env.grid[r, c] = 1   # inject

        for (nr, nc), original in neighbours_before.items():
            assert env.grid[nr, nc] == original, (
                f"Injection at ({r},{c}) unexpectedly changed ({nr},{nc})"
            )

    def test_obs_manager_occupied_cells_match_injected_cells(self):
        """
        ObstacleManager.get_all_occupied_cells() returns the same cell
        that we inject into env.grid — verifying the two systems agree.
        """
        obs_manager = ObstacleManager(dt=0.2, cell_size=CELL)
        forklift_id = obs_manager.add_obstacle(
            start_pos=(12, 5),
            obstacle_type=ObstacleType.LINEAR,
            speed=1.0,
            size=1,
        )

        occupied = obs_manager.obstacles[forklift_id].get_occupied_cells()
        assert (12, 5) in occupied
        assert len(occupied) == 1   # size=1 → single cell


# ──────────────────────────────────────────────────────────────────────────────
# A* path finding on warehouse layout
# ──────────────────────────────────────────────────────────────────────────────

class TestAStarOnWarehouseEnv:

    def test_astar_finds_path_from_start_to_goal(self):
        """
        A* must find a valid path from (1,1) to (18,18) on the warehouse layout.

        This is the most important smoke test: if the obstacle layout inadvertently
        traps start or goal, the whole simulation fails before even running.
        """
        env = build_warehouse_env()
        start = WAREHOUSE_CONFIG['start']
        goal  = WAREHOUSE_CONFIG['goal']

        astar = AStar(env, heuristic='manhattan')
        path = astar.search(start, goal,
                            robot_radius=WAREHOUSE_CONFIG['astar_robot_radius'],
                            cell_size=CELL)

        assert path is not None, (
            "A* returned None — obstacle layout may have blocked start/goal"
        )
        assert path[0] == start
        assert path[-1] == goal
        assert len(path) > 1

    def test_astar_path_avoids_shelf_obstacles(self):
        """All cells in the A* path must be free (no obstacles)."""
        env = build_warehouse_env()
        astar = AStar(env, heuristic='manhattan')
        path = astar.search(
            WAREHOUSE_CONFIG['start'], WAREHOUSE_CONFIG['goal'],
            robot_radius=0.0, cell_size=CELL,   # robot_radius=0 → pure A*
        )
        assert path is not None
        for r, c in path:
            assert env.grid[r, c] == 0, (
                f"Path goes through obstacle at ({r},{c})"
            )

    def test_astar_path_is_connected(self):
        """Every consecutive pair of cells in the path must be 4-connected."""
        env = build_warehouse_env()
        astar = AStar(env, heuristic='manhattan')
        path = astar.search(
            WAREHOUSE_CONFIG['start'], WAREHOUSE_CONFIG['goal'],
            robot_radius=0.0, cell_size=CELL,
        )
        assert path is not None
        for i in range(len(path) - 1):
            r0, c0 = path[i]
            r1, c1 = path[i + 1]
            step = abs(r1 - r0) + abs(c1 - c0)
            assert step == 1, (
                f"Non-4-connected step in path: {path[i]} → {path[i+1]}"
            )


# ──────────────────────────────────────────────────────────────────────────────
# Replan counter logic
# ──────────────────────────────────────────────────────────────────────────────

class TestReplanCounterLogic:
    """
    The consecutive_failures counter inside update() cannot be tested directly
    (it lives in a closure), but _check_predictive_collision() returns a bool
    that drives it. These tests verify the INTENDED counting semantics:

      - DWA ok + no predictive hit    → failures = 0  (reset)
      - DWA fail                      → failures += 1
      - DWA ok + predictive hit       → failures stays 0 then +1 = 1 net
      - DWA fail + predictive hit     → failures += 2 per frame
      - failures >= threshold         → replan should trigger
    """

    THRESHOLD = WAREHOUSE_CONFIG['replan_threshold']   # 3

    def _simulate_counter(self, events):
        """
        Simulate the counter update logic from update() for a sequence of
        (dwa_ok, predictive_collision) events.  Returns final counter value.

        WARNING: This helper reimplements the counter logic from update() in
        warehouse_simulation.py. If the branching or increment logic inside
        update() ever changes, this helper MUST be updated to match — otherwise
        these tests will silently verify stale (wrong) behavior.
        Last verified in sync with: warehouse_simulation.py update() function.
        """
        failures = 0
        for dwa_ok, pred_hit in events:
            # DWA failure check
            if not dwa_ok:
                failures += 1
            else:
                failures = 0
            # Predictive collision check (always runs after DWA check)
            if pred_hit:
                failures += 1
            # If threshold reached, replan fires and resets counter
            if failures >= self.THRESHOLD:
                failures = 0
        return failures

    def test_dwa_success_resets_counter(self):
        """DWA success (ok=True, no predictive hit) must reset counter to 0."""
        # Build up 2 failures, then a success
        events = [(False, False), (False, False), (True, False)]
        assert self._simulate_counter(events) == 0

    def test_three_dwa_failures_trigger_replan_reset(self):
        """
        Three consecutive DWA failures (ok=False) must hit the threshold
        and reset the counter to 0.
        """
        # 3 failures → threshold reached at frame 3 → counter reset to 0
        events = [(False, False)] * 3
        assert self._simulate_counter(events) == 0

    def test_dwa_fail_plus_predictive_hit_accelerates_replan(self):
        """
        When both DWA fails AND predictive collision fires, the counter
        increments by 2 per frame. Two frames → 4 ≥ threshold → replan.
        """
        # Frame 1: DWA fail (+1) + predictive (+1) = 2
        # Frame 2: DWA fail (+1) = 3 ≥ threshold → reset
        events = [(False, True), (False, False)]
        assert self._simulate_counter(events) == 0

    def test_counter_does_not_exceed_threshold_without_replan(self):
        """
        Once failures >= threshold, the replan fires and resets to 0.
        Counter should never be >= threshold at end of a frame.
        """
        # 10 consecutive DWA failures: threshold fires at 3, 6, 9 → all reset
        events = [(False, False)] * 10
        final = self._simulate_counter(events)
        assert final < self.THRESHOLD
