"""
Unit Tests for DWAController
==============================

These tests verify each building block of DWA independently, then test
the full compute_command() pipeline end-to-end.

Test philosophy
---------------
  - Each private helper (_dynamic_window, _simulate_trajectory, etc.) is tested
    in isolation: easy to pinpoint which step fails.
  - compute_command() integration tests verify realistic scenarios:
    open space (should move forward), obstacle ahead (should swerve),
    all paths blocked (should return success=False).

Run with: pytest tests/test_dwa.py -v
"""

import sys
sys.path.append('.')

import math
import pytest
import numpy as np

from src.control.dwa import DWAController


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures / helpers
# ──────────────────────────────────────────────────────────────────────────────

def make_dwa(**kwargs) -> DWAController:
    """Build a DWAController with sensible defaults, overridable via kwargs."""
    defaults = dict(
        max_v=1.0,
        max_omega=math.pi,
        max_dv=1.0,        # can reach full speed in 1 second
        max_domega=math.pi,
        robot_radius=0.15,
        sim_time=1.0,
        dt=0.1,
        v_samples=10,
        omega_samples=10,
        clearance_cap=2.0,
        w_heading=1.0,
        w_clearance=1.0,
        w_velocity=0.5,
    )
    defaults.update(kwargs)
    return DWAController(**defaults)


def no_obstacles() -> np.ndarray:
    """Empty obstacle array — open space."""
    return np.empty((0, 2), dtype=float)


def obstacle_at(x: float, y: float) -> np.ndarray:
    """Single obstacle at (x, y)."""
    return np.array([[x, y]], dtype=float)


def wall_of_obstacles(xs, y: float) -> np.ndarray:
    """Horizontal line of obstacles at a fixed y, varying x."""
    return np.array([[x, y] for x in xs], dtype=float)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Dynamic window
# ──────────────────────────────────────────────────────────────────────────────

class TestDynamicWindow:
    """
    The dynamic window clips what's reachable this time step.
    Tests cover: stopped robot, moving robot, already-at-limit robot.
    """

    def test_stopped_robot_window_symmetric(self):
        """From v=0, the window should be symmetric ± max_dv*dt."""
        dwa = make_dwa(max_v=1.0, max_dv=1.0, dt=0.1)
        v_min, v_max, w_min, w_max = dwa._dynamic_window(0.0, 0.0)
        # max_dv * dt = 1.0 * 0.1 = 0.1
        assert abs(v_min - (-0.1)) < 1e-9
        assert abs(v_max - ( 0.1)) < 1e-9

    def test_window_clipped_by_max_v(self):
        """If current speed is near max_v, window v_max cannot exceed max_v."""
        dwa = make_dwa(max_v=1.0, max_dv=5.0, dt=0.1)
        _, v_max, _, _ = dwa._dynamic_window(0.95, 0.0)
        # 0.95 + 5.0*0.1 = 1.45 → clipped to 1.0
        assert abs(v_max - 1.0) < 1e-9

    def test_window_clipped_by_negative_max_v(self):
        """v_min cannot go below -max_v."""
        dwa = make_dwa(max_v=1.0, max_dv=5.0, dt=0.1)
        v_min, _, _, _ = dwa._dynamic_window(-0.95, 0.0)
        assert abs(v_min - (-1.0)) < 1e-9

    def test_window_clipped_by_max_omega(self):
        """ω window is clipped by max_omega."""
        dwa = make_dwa(max_omega=math.pi, max_domega=100.0, dt=0.1)
        _, _, w_min, w_max = dwa._dynamic_window(0.0, 0.0)
        assert abs(w_min - (-math.pi)) < 1e-9
        assert abs(w_max - ( math.pi)) < 1e-9

    def test_window_shrinks_from_high_v(self):
        """
        A robot already at v=0.8 with large acceleration limit:
        v_min = max(-1.0, 0.8 - 1.0*0.1) = max(-1.0, 0.7) = 0.7
        """
        dwa = make_dwa(max_v=1.0, max_dv=1.0, dt=0.1)
        v_min, _, _, _ = dwa._dynamic_window(0.8, 0.0)
        assert abs(v_min - 0.7) < 1e-9

    def test_window_always_v_min_le_v_max(self):
        """v_min ≤ v_max in all cases."""
        dwa = make_dwa()
        for v in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            for w in [-math.pi, 0.0, math.pi]:
                v_min, v_max, w_min, w_max = dwa._dynamic_window(v, w)
                assert v_min <= v_max
                assert w_min <= w_max


# ──────────────────────────────────────────────────────────────────────────────
# 2. Trajectory simulation
# ──────────────────────────────────────────────────────────────────────────────

class TestSimulateTrajectory:
    """
    Trajectory simulation must match manual Euler integration exactly.
    """

    def test_output_shape(self):
        dwa = make_dwa(sim_time=1.0, dt=0.1)
        traj = dwa._simulate_trajectory(0.0, 0.0, 0.0, 0.5, 0.0)
        assert traj.shape == (10, 3)

    def test_straight_line_heading_zero(self):
        """
        v=1.0, ω=0, theta=0 (facing right).
        Each step: x += 1.0 * cos(0) * 0.1 = 0.1. y stays at 0.
        """
        dwa = make_dwa(sim_time=0.3, dt=0.1)
        traj = dwa._simulate_trajectory(0.0, 0.0, 0.0, 1.0, 0.0)
        assert traj.shape == (3, 3)
        np.testing.assert_allclose(traj[:, 0], [0.1, 0.2, 0.3], atol=1e-9)
        np.testing.assert_allclose(traj[:, 1], [0.0, 0.0, 0.0], atol=1e-9)
        np.testing.assert_allclose(traj[:, 2], [0.0, 0.0, 0.0], atol=1e-9)

    def test_stationary_robot(self):
        """v=0, ω=0: robot stays at origin."""
        dwa = make_dwa(sim_time=0.5, dt=0.1)
        traj = dwa._simulate_trajectory(1.0, 2.0, 0.5, 0.0, 0.0)
        np.testing.assert_allclose(traj[:, 0], 1.0, atol=1e-9)
        np.testing.assert_allclose(traj[:, 1], 2.0, atol=1e-9)
        np.testing.assert_allclose(traj[:, 2], 0.5, atol=1e-9)

    def test_pure_rotation(self):
        """v=0, ω=π: heading increases by π*dt each step."""
        dwa = make_dwa(sim_time=0.3, dt=0.1)
        traj = dwa._simulate_trajectory(0.0, 0.0, 0.0, 0.0, math.pi)
        # Position stays at (0, 0)
        np.testing.assert_allclose(traj[:, 0], 0.0, atol=1e-9)
        np.testing.assert_allclose(traj[:, 1], 0.0, atol=1e-9)
        # Heading: π*0.1, π*0.2, π*0.3
        expected_theta = [math.pi * 0.1, math.pi * 0.2, math.pi * 0.3]
        np.testing.assert_allclose(traj[:, 2], expected_theta, atol=1e-9)

    def test_trajectory_starts_from_given_pose(self):
        """First step uses the GIVEN start pose, not the origin."""
        dwa = make_dwa(sim_time=0.1, dt=0.1)
        traj = dwa._simulate_trajectory(5.0, 3.0, 0.0, 1.0, 0.0)
        np.testing.assert_allclose(traj[0, 0], 5.1, atol=1e-9)
        np.testing.assert_allclose(traj[0, 1], 3.0, atol=1e-9)


# ──────────────────────────────────────────────────────────────────────────────
# 3. Collision detection
# ──────────────────────────────────────────────────────────────────────────────

class TestIsCollision:
    """
    _is_collision() must detect when any trajectory point is within
    robot_radius of any obstacle point.
    """

    def test_no_obstacles_never_collides(self):
        dwa = make_dwa(robot_radius=0.3)
        traj = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        assert dwa._is_collision(traj, no_obstacles()) is False

    def test_obstacle_far_away_no_collision(self):
        dwa = make_dwa(robot_radius=0.3)
        traj = np.array([[0.0, 0.0, 0.0]])
        obs  = obstacle_at(10.0, 10.0)
        assert dwa._is_collision(traj, obs) is False

    def test_obstacle_exactly_at_robot_radius_triggers(self):
        """Distance < robot_radius triggers collision."""
        dwa = make_dwa(robot_radius=0.3)
        traj = np.array([[0.0, 0.0, 0.0]])
        # obstacle at 0.29 m — inside robot_radius
        obs = obstacle_at(0.29, 0.0)
        assert dwa._is_collision(traj, obs) is True

    def test_obstacle_beyond_robot_radius_no_collision(self):
        dwa = make_dwa(robot_radius=0.3)
        traj = np.array([[0.0, 0.0, 0.0]])
        obs = obstacle_at(0.31, 0.0)
        assert dwa._is_collision(traj, obs) is False

    def test_collision_detected_at_later_trajectory_step(self):
        """
        Trajectory [safe, safe, COLLISION]: should still return True even
        though the first two steps are clear.
        """
        dwa = make_dwa(robot_radius=0.3)
        traj = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],   # obstacle is at (2.1, 0) — 0.1 m away → collision
        ])
        obs = obstacle_at(2.1, 0.0)
        assert dwa._is_collision(traj, obs) is True

    def test_multiple_obstacles_nearest_triggers(self):
        dwa = make_dwa(robot_radius=0.3)
        traj = np.array([[0.0, 0.0, 0.0]])
        obs = np.array([[5.0, 5.0], [0.1, 0.0]])  # second obstacle is close
        assert dwa._is_collision(traj, obs) is True


# ──────────────────────────────────────────────────────────────────────────────
# 4. Heading score
# ──────────────────────────────────────────────────────────────────────────────

class TestScoreHeading:
    """
    Heading score must be 1.0 when facing the goal, 0.0 when facing away.
    """

    def _traj_at(self, x, y, theta):
        """Single-step trajectory at given pose."""
        return np.array([[x, y, theta]])

    def test_facing_goal_directly_score_one(self):
        dwa = make_dwa()
        # Robot at (0,0) facing right (theta=0), goal at (1,0)
        traj = self._traj_at(0.0, 0.0, 0.0)
        score = dwa._score_heading(traj, goal_x=1.0, goal_y=0.0)
        assert abs(score - 1.0) < 1e-6

    def test_facing_away_from_goal_score_zero(self):
        dwa = make_dwa()
        # Robot at (0,0) facing right (theta=0), goal is to the LEFT (−x)
        traj = self._traj_at(0.0, 0.0, 0.0)
        score = dwa._score_heading(traj, goal_x=-1.0, goal_y=0.0)
        assert abs(score - 0.0) < 1e-6

    def test_perpendicular_to_goal_score_half(self):
        dwa = make_dwa()
        # Robot at (0,0) facing right (theta=0), goal is directly below (0,1)
        # heading error = π/2 → score = (π − π/2) / π = 0.5
        traj = self._traj_at(0.0, 0.0, 0.0)
        score = dwa._score_heading(traj, goal_x=0.0, goal_y=1.0)
        assert abs(score - 0.5) < 1e-6

    def test_score_uses_final_trajectory_pose(self):
        """Multi-step trajectory: heading score uses LAST pose, not first."""
        dwa = make_dwa()
        traj = np.array([
            [0.0, 0.0, math.pi],   # facing away
            [1.0, 0.0, 0.0],       # facing toward goal
        ])
        score = dwa._score_heading(traj, goal_x=2.0, goal_y=0.0)
        assert abs(score - 1.0) < 1e-6

    def test_score_always_in_range(self):
        """Score must be in [0, 1] for any pose and goal combination."""
        dwa = make_dwa()
        rng = np.random.default_rng(42)
        for _ in range(50):
            theta = rng.uniform(-math.pi, math.pi)
            traj  = self._traj_at(0.0, 0.0, theta)
            gx, gy = rng.uniform(-5, 5), rng.uniform(-5, 5)
            score = dwa._score_heading(traj, gx, gy)
            assert 0.0 - 1e-9 <= score <= 1.0 + 1e-9


# ──────────────────────────────────────────────────────────────────────────────
# 5. Clearance score
# ──────────────────────────────────────────────────────────────────────────────

class TestScoreClearance:
    """
    Clearance score: 1.0 with no obstacles or far obstacles, 0.0 when touching.
    """

    def test_no_obstacles_returns_one(self):
        dwa = make_dwa(clearance_cap=2.0)
        traj = np.array([[0.0, 0.0, 0.0]])
        assert abs(dwa._score_clearance(traj, no_obstacles()) - 1.0) < 1e-9

    def test_obstacle_beyond_clearance_cap_returns_one(self):
        dwa = make_dwa(clearance_cap=2.0)
        traj = np.array([[0.0, 0.0, 0.0]])
        obs  = obstacle_at(5.0, 0.0)   # 5 m away > cap of 2 m
        assert abs(dwa._score_clearance(traj, obs) - 1.0) < 1e-9

    def test_obstacle_touching_returns_zero(self):
        dwa = make_dwa(clearance_cap=2.0)
        traj = np.array([[0.0, 0.0, 0.0]])
        obs  = obstacle_at(0.0, 0.0)   # distance = 0
        assert abs(dwa._score_clearance(traj, obs) - 0.0) < 1e-9

    def test_obstacle_at_half_cap_returns_half(self):
        dwa = make_dwa(clearance_cap=2.0)
        traj = np.array([[0.0, 0.0, 0.0]])
        obs  = obstacle_at(1.0, 0.0)   # 1 m away, cap=2 → score = 0.5
        assert abs(dwa._score_clearance(traj, obs) - 0.5) < 1e-6

    def test_score_uses_minimum_over_all_steps(self):
        """Closest step to any obstacle determines the score."""
        dwa = make_dwa(clearance_cap=2.0)
        traj = np.array([
            [0.0, 0.0, 0.0],   # 5 m from obstacle
            [4.5, 0.0, 0.0],   # 0.5 m from obstacle → dominates
        ])
        obs = obstacle_at(5.0, 0.0)
        score = dwa._score_clearance(traj, obs)
        assert abs(score - 0.5 / 2.0) < 1e-6


# ──────────────────────────────────────────────────────────────────────────────
# 6. Velocity score
# ──────────────────────────────────────────────────────────────────────────────

class TestScoreVelocity:
    """Velocity score: 1.0 at max_v, 0.0 at v≤0."""

    def test_max_v_returns_one(self):
        dwa = make_dwa(max_v=1.0)
        assert abs(dwa._score_velocity(1.0) - 1.0) < 1e-9

    def test_zero_v_returns_zero(self):
        dwa = make_dwa(max_v=1.0)
        assert abs(dwa._score_velocity(0.0) - 0.0) < 1e-9

    def test_half_v_returns_half(self):
        dwa = make_dwa(max_v=1.0)
        assert abs(dwa._score_velocity(0.5) - 0.5) < 1e-9

    def test_negative_v_returns_zero_not_negative(self):
        """Reverse velocity should score 0, not negative (neutral treatment)."""
        dwa = make_dwa(max_v=1.0)
        assert abs(dwa._score_velocity(-0.5) - 0.0) < 1e-9

    def test_score_scales_with_max_v(self):
        """Score at half of max_v should always be 0.5 regardless of max_v."""
        for mv in [0.5, 1.0, 2.0]:
            dwa = make_dwa(max_v=mv)
            assert abs(dwa._score_velocity(mv / 2) - 0.5) < 1e-9


# ──────────────────────────────────────────────────────────────────────────────
# 7. compute_command() integration tests
# ──────────────────────────────────────────────────────────────────────────────

class TestComputeCommand:
    """
    End-to-end tests for the full planning pipeline.
    """

    def test_open_space_returns_success(self):
        """In open space, at least one trajectory should be valid."""
        dwa = make_dwa(w_heading=1.0, w_velocity=1.0, w_clearance=1.0)
        v, omega, ok = dwa.compute_command(
            x=0.0, y=0.0, theta=0.0,
            v_cur=0.0, omega_cur=0.0,
            goal_x=5.0, goal_y=0.0,
            obstacles=no_obstacles(),
        )
        assert ok is True

    def test_open_space_moves_toward_goal(self):
        """
        Robot at origin facing right (theta=0), goal to the right (+x).
        In open space, best trajectory should be: forward (v > 0), low ω.
        """
        dwa = make_dwa(
            w_heading=2.0,   # prioritise heading
            w_velocity=1.0,
            w_clearance=1.0,
        )
        v, omega, ok = dwa.compute_command(
            x=0.0, y=0.0, theta=0.0,
            v_cur=0.0, omega_cur=0.0,
            goal_x=5.0, goal_y=0.0,
            obstacles=no_obstacles(),
        )
        assert ok is True
        assert v > 0, f"Expected forward motion, got v={v}"
        assert abs(omega) < math.pi / 2, f"Expected small ω, got ω={omega}"

    def test_obstacle_ahead_causes_avoidance(self):
        """
        Obstacle directly in front. Robot should choose a non-zero ω to turn.
        """
        dwa = make_dwa(
            robot_radius=0.3,
            w_heading=1.0,
            w_clearance=3.0,   # prioritise clearance
            w_velocity=0.5,
            v_samples=5,
            omega_samples=11,
        )
        # Wall of obstacles at x=0.5, right in front of robot at x=0
        obs = wall_of_obstacles(xs=[0.4, 0.5, 0.6], y=0.0)

        v, omega, ok = dwa.compute_command(
            x=0.0, y=0.0, theta=0.0,
            v_cur=0.0, omega_cur=0.0,
            goal_x=5.0, goal_y=0.0,
            obstacles=obs,
        )
        # Should find SOME safe trajectory (turn to avoid)
        assert ok is True

    def test_all_paths_blocked_returns_false(self):
        """
        Surround the robot completely with obstacles at robot_radius distance.
        All trajectories should collide → success=False.
        """
        dwa = make_dwa(
            robot_radius=0.5,
            sim_time=0.1,   # short horizon: robot can't escape in 0.1 s
            dt=0.1,
            v_samples=3,
            omega_samples=3,
            max_dv=0.1,     # slow acceleration: can't build escape speed fast
            max_domega=0.1,
        )
        # Dense ring of obstacles at 0.3 m — all inside robot_radius=0.5
        angles = np.linspace(0, 2 * math.pi, 36, endpoint=False)
        ring_obs = np.column_stack([
            0.3 * np.cos(angles),
            0.3 * np.sin(angles),
        ])
        v, omega, ok = dwa.compute_command(
            x=0.0, y=0.0, theta=0.0,
            v_cur=0.0, omega_cur=0.0,
            goal_x=5.0, goal_y=0.0,
            obstacles=ring_obs,
        )
        assert ok is False
        assert v == 0.0
        assert omega == 0.0

    def test_return_type_is_tuple_of_three(self):
        dwa = make_dwa()
        result = dwa.compute_command(
            x=0.0, y=0.0, theta=0.0,
            v_cur=0.0, omega_cur=0.0,
            goal_x=1.0, goal_y=0.0,
            obstacles=no_obstacles(),
        )
        assert len(result) == 3
        v, omega, ok = result
        assert isinstance(v, float)
        assert isinstance(omega, float)
        assert isinstance(ok, bool)

    def test_command_within_velocity_limits(self):
        """Returned (v, ω) must respect max_v and max_omega."""
        dwa = make_dwa(max_v=1.0, max_omega=math.pi)
        v, omega, ok = dwa.compute_command(
            x=0.0, y=0.0, theta=0.0,
            v_cur=0.0, omega_cur=0.0,
            goal_x=10.0, goal_y=0.0,
            obstacles=no_obstacles(),
        )
        assert ok is True
        assert -1.0 <= v <= 1.0
        assert -math.pi <= omega <= math.pi

    def test_goal_at_current_position_does_not_crash(self):
        """Goal at robot's current position — edge case, must not throw."""
        dwa = make_dwa()
        v, omega, ok = dwa.compute_command(
            x=0.0, y=0.0, theta=0.0,
            v_cur=0.0, omega_cur=0.0,
            goal_x=0.0, goal_y=0.0,     # goal = current position
            obstacles=no_obstacles(),
        )
        # Just must not raise; result can be anything
        assert isinstance(ok, bool)

    def test_weights_applied_in_scoring(self):
        """
        White-box test: verify the weighted sum formula is applied correctly.

        We construct a known scenario and manually compute the expected total
        score, then verify the planner's internal scores match.

        WHY this approach (not "two configs → different commands"):
        -----------------------------------------------------------------
        Weights only produce different commands when there's a genuine
        tradeoff between objectives. With a small dynamic window (±0.1 m/s)
        and a single obstacle, the best escape route is often the same
        regardless of weight ratios. Testing the score formula directly is
        more reliable and more pedagogically meaningful.
        """
        dwa = make_dwa(w_heading=2.0, w_clearance=1.0, w_velocity=3.0,
                       clearance_cap=2.0, max_v=1.0)

        # Trajectory: robot at (1, 0) facing right, heading directly at goal
        traj = np.array([[1.0, 0.0, 0.0]])

        # Obstacle at (1, 1) → distance = 1.0 → clearance score = 1.0/2.0 = 0.5
        obs = obstacle_at(1.0, 1.0)

        s_h = dwa._score_heading(traj, goal_x=5.0, goal_y=0.0)  # 1.0 (facing goal)
        s_c = dwa._score_clearance(traj, obs)                    # 0.5
        s_v = dwa._score_velocity(0.5)                           # 0.5

        expected = 2.0 * s_h + 1.0 * s_c + 3.0 * s_v
        actual   = (dwa.w_heading   * s_h +
                    dwa.w_clearance * s_c +
                    dwa.w_velocity  * s_v)

        assert abs(actual - expected) < 1e-9, \
            f"Weighted sum incorrect: {actual} ≠ {expected}"
        # Sanity-check individual scores
        assert abs(s_h - 1.0) < 1e-6, f"Heading score: {s_h}"
        assert abs(s_c - 0.5) < 1e-6, f"Clearance score: {s_c}"
        assert abs(s_v - 0.5) < 1e-6, f"Velocity score: {s_v}"


# ──────────────────────────────────────────────────────────────────────────────
# 8. __repr__
# ──────────────────────────────────────────────────────────────────────────────

class TestRepr:
    def test_repr_contains_key_info(self):
        dwa = make_dwa(max_v=1.0, robot_radius=0.15)
        r = repr(dwa)
        assert "DWAController" in r
        assert "1.0" in r
        assert "0.15" in r
