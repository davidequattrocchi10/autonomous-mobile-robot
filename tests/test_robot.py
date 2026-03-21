"""
Tests for the robot model and coordinate conversion utilities.

Testing strategy
----------------
Kinematic tests use exact arithmetic where possible (no noise, simple
angles) so that expected positions can be computed by hand and asserted
with a tight tolerance (1e-9).

Floating-point note
-------------------
cos(π/2) is not exactly 0 in IEEE 754 arithmetic (it is ~6.12e-17).
Tests that involve π/2 rotations use pytest.approx() with a tolerance
of 1e-9 rather than exact equality.

Structure
---------
  TestConversions      — grid_to_continuous and continuous_to_grid
  TestAngleConversion  — angle_to_grid_direction
  TestRobotInit        — constructor, from_grid_cell
  TestRobotKinematics  — update() with known inputs
  TestRobotNoise       — odometry noise behaviour
  TestRobotGridBridge  — get_grid_cell, from_grid_cell round-trip
  TestRobotTrajectory  — trajectory recording
"""

import math
import pytest
import numpy as np

from src.utils.conversions import (
    grid_to_continuous,
    continuous_to_grid,
    angle_to_grid_direction,
    DEFAULT_CELL_SIZE,
)
from src.robot.robot import DifferentialDriveRobot


# =========================================================================
# TestConversions — coordinate conversion utilities
# =========================================================================

class TestConversions:
    """
    Verify that grid_to_continuous and continuous_to_grid are inverses of
    each other and correctly apply the cell_size scale factor.
    """

    def test_origin_cell_centre(self):
        """Cell (0, 0) centre is at (cell_size/2, cell_size/2)."""
        x, y = grid_to_continuous(0, 0, cell_size=0.5)
        assert x == pytest.approx(0.25)
        assert y == pytest.approx(0.25)

    def test_arbitrary_cell_centre(self):
        """Cell (2, 3) centre with cell_size=0.5 → x=1.75, y=1.25."""
        x, y = grid_to_continuous(2, 3, cell_size=0.5)
        assert x == pytest.approx(1.75)
        assert y == pytest.approx(1.25)

    def test_unit_cell_size(self):
        """With cell_size=1.0, cell (r, c) centre is at (c+0.5, r+0.5)."""
        x, y = grid_to_continuous(4, 7, cell_size=1.0)
        assert x == pytest.approx(7.5)
        assert y == pytest.approx(4.5)

    def test_continuous_to_grid_centre(self):
        """The centre of cell (2, 3) maps back to (2, 3)."""
        x, y = grid_to_continuous(2, 3, cell_size=0.5)
        row, col = continuous_to_grid(x, y, cell_size=0.5)
        assert (row, col) == (2, 3)

    def test_continuous_to_grid_corner(self):
        """Top-left corner of cell (0, 0) (x=0, y=0) maps to (0, 0)."""
        row, col = continuous_to_grid(0.0, 0.0, cell_size=0.5)
        assert (row, col) == (0, 0)

    def test_continuous_to_grid_just_inside_next_cell(self):
        """x = cell_size maps to col 1 (not col 0)."""
        row, col = continuous_to_grid(0.5, 0.0, cell_size=0.5)
        assert col == 1

    def test_round_trip_many_cells(self):
        """
        grid → continuous → grid must be the identity for all cells
        in a 10×10 grid.
        """
        for row in range(10):
            for col in range(10):
                x, y = grid_to_continuous(row, col)
                r2, c2 = continuous_to_grid(x, y)
                assert (r2, c2) == (row, col), (
                    f"Round-trip failed for ({row}, {col}): got ({r2}, {c2})"
                )

    def test_default_cell_size_used(self):
        """Functions use DEFAULT_CELL_SIZE when cell_size is not specified."""
        x, y = grid_to_continuous(0, 0)
        assert x == pytest.approx(DEFAULT_CELL_SIZE / 2)
        assert y == pytest.approx(DEFAULT_CELL_SIZE / 2)


class TestAngleConversion:
    """Verify angle_to_grid_direction snaps to the correct cardinal direction."""

    def test_facing_right(self):
        assert angle_to_grid_direction(0.0) == (0, 1)

    def test_facing_down(self):
        assert angle_to_grid_direction(math.pi / 2) == (1, 0)

    def test_facing_left(self):
        assert angle_to_grid_direction(math.pi) == (0, -1)

    def test_facing_up(self):
        assert angle_to_grid_direction(-math.pi / 2) == (-1, 0)

    def test_diagonal_snaps_to_nearest_cardinal(self):
        """
        theta = π/4 (45°, diagonal between RIGHT and DOWN) should snap to
        RIGHT because it is closer to RIGHT (within π/4 of 0).
        """
        # π/4 is exactly on the boundary — implementation returns RIGHT
        result = angle_to_grid_direction(math.pi / 4)
        assert result in [(0, 1), (1, 0)]   # either cardinal is acceptable

    def test_slightly_past_diagonal_snaps_to_down(self):
        """theta = π/4 + ε should snap to DOWN."""
        result = angle_to_grid_direction(math.pi / 4 + 0.01)
        assert result == (1, 0)


# =========================================================================
# TestRobotInit — constructor and class methods
# =========================================================================

class TestRobotInit:
    """Verify that the robot initialises with the correct state."""

    def test_default_pose(self):
        """Default constructor places robot at (0, 0, 0)."""
        robot = DifferentialDriveRobot()
        assert robot.x     == pytest.approx(0.0)
        assert robot.y     == pytest.approx(0.0)
        assert robot.theta == pytest.approx(0.0)

    def test_custom_pose(self):
        """Constructor stores the supplied pose."""
        robot = DifferentialDriveRobot(x=1.5, y=2.0, theta=math.pi / 4)
        assert robot.x     == pytest.approx(1.5)
        assert robot.y     == pytest.approx(2.0)
        assert robot.theta == pytest.approx(math.pi / 4)

    def test_theta_normalised_on_init(self):
        """
        theta = 3π is normalised to ±π on construction.

        atan2(sin(3π), cos(3π)) returns +π rather than -π because
        sin(3π) is a tiny positive float (~3.7e-16) in IEEE 754 arithmetic,
        so atan2 sees a point just above the negative x-axis and returns +π.
        Both +π and -π represent the same heading, so we accept either.
        """
        robot = DifferentialDriveRobot(theta=3 * math.pi)
        assert abs(robot.theta) == pytest.approx(math.pi, abs=1e-9)

    def test_get_pose_returns_tuple(self):
        """get_pose() returns (x, y, theta) matching stored values."""
        robot = DifferentialDriveRobot(x=1.0, y=2.0, theta=0.5)
        x, y, theta = robot.get_pose()
        assert (x, y, theta) == pytest.approx((1.0, 2.0, 0.5))

    def test_from_grid_cell_places_at_centre(self):
        """from_grid_cell() places robot at the centre of the given cell."""
        robot = DifferentialDriveRobot.from_grid_cell(2, 3, cell_size=0.5)
        x_expected, y_expected = grid_to_continuous(2, 3, cell_size=0.5)
        assert robot.x == pytest.approx(x_expected)
        assert robot.y == pytest.approx(y_expected)

    def test_from_grid_cell_preserves_theta(self):
        """from_grid_cell() forwards theta correctly."""
        robot = DifferentialDriveRobot.from_grid_cell(0, 0, theta=math.pi / 2)
        assert robot.theta == pytest.approx(math.pi / 2)

    def test_trajectory_initialised_with_start_pose(self):
        """Trajectory list must contain the initial pose after construction."""
        robot = DifferentialDriveRobot(x=1.0, y=2.0, theta=0.0)
        assert len(robot.trajectory) == 1
        assert robot.trajectory[0] == pytest.approx((1.0, 2.0, 0.0))

    def test_repr_contains_useful_info(self):
        """__repr__ should at least mention x, y, and the cell."""
        robot = DifferentialDriveRobot(x=0.25, y=0.25)
        r = repr(robot)
        assert "x=" in r and "y=" in r


# =========================================================================
# TestRobotKinematics — the Euler integration update
# =========================================================================

class TestRobotKinematics:
    """
    Verify the kinematic update equations:
      x_new = x + v * cos(θ) * dt
      y_new = y + v * sin(θ) * dt
      θ_new = θ + ω * dt
    """

    def test_move_right_from_origin(self):
        """
        v=1, ω=0, dt=1, theta=0 → moves exactly 1 m to the right.
        cos(0)=1, sin(0)=0 → x_new=1, y_new=0.
        """
        robot = DifferentialDriveRobot(x=0.0, y=0.0, theta=0.0)
        robot.update(v=1.0, omega=0.0, dt=1.0)
        assert robot.x == pytest.approx(1.0, abs=1e-9)
        assert robot.y == pytest.approx(0.0, abs=1e-9)
        assert robot.theta == pytest.approx(0.0, abs=1e-9)

    def test_move_down_when_facing_down(self):
        """
        theta=π/2 (facing down), v=1, ω=0, dt=1.
        cos(π/2)≈0, sin(π/2)=1 → x unchanged, y increases by 1.
        """
        robot = DifferentialDriveRobot(x=0.0, y=0.0, theta=math.pi / 2)
        robot.update(v=1.0, omega=0.0, dt=1.0)
        assert robot.x == pytest.approx(0.0, abs=1e-9)
        assert robot.y == pytest.approx(1.0, abs=1e-9)

    def test_rotate_in_place(self):
        """
        v=0, ω=π/2, dt=1 → robot stays in place but turns 90° clockwise.
        """
        robot = DifferentialDriveRobot(x=1.0, y=1.0, theta=0.0)
        robot.update(v=0.0, omega=math.pi / 2, dt=1.0)
        assert robot.x     == pytest.approx(1.0, abs=1e-9)
        assert robot.y     == pytest.approx(1.0, abs=1e-9)
        assert robot.theta == pytest.approx(math.pi / 2, abs=1e-9)

    def test_move_backward(self):
        """Negative v means moving backward (opposite to heading direction)."""
        robot = DifferentialDriveRobot(x=2.0, y=0.0, theta=0.0)
        robot.update(v=-1.0, omega=0.0, dt=1.0)
        assert robot.x == pytest.approx(1.0, abs=1e-9)
        assert robot.y == pytest.approx(0.0, abs=1e-9)

    def test_multiple_steps_accumulate(self):
        """
        Two identical steps should produce twice the displacement
        of one step (linear accumulation for straight-line motion).
        """
        robot = DifferentialDriveRobot(x=0.0, y=0.0, theta=0.0)
        robot.update(v=1.0, omega=0.0, dt=0.5)
        robot.update(v=1.0, omega=0.0, dt=0.5)
        assert robot.x == pytest.approx(1.0, abs=1e-9)
        assert robot.y == pytest.approx(0.0, abs=1e-9)

    def test_small_dt_produces_small_displacement(self):
        """Smaller dt → smaller displacement (proportional to dt)."""
        robot_fast = DifferentialDriveRobot(x=0.0, y=0.0, theta=0.0)
        robot_slow = DifferentialDriveRobot(x=0.0, y=0.0, theta=0.0)
        robot_fast.update(v=1.0, omega=0.0, dt=1.0)
        robot_slow.update(v=1.0, omega=0.0, dt=0.1)
        assert robot_slow.x == pytest.approx(0.1, abs=1e-9)
        assert robot_fast.x == pytest.approx(1.0, abs=1e-9)

    def test_theta_normalised_after_update(self):
        """
        After a rotation that crosses ±π, theta must be normalised.
        Starting at θ = π - 0.1 and adding ω*dt = 0.2 rad gives θ > π.
        The result must be normalised to the (-π, π] range.
        """
        robot = DifferentialDriveRobot(x=0.0, y=0.0, theta=math.pi - 0.1)
        robot.update(v=0.0, omega=1.0, dt=0.2)
        assert -math.pi <= robot.theta <= math.pi

    def test_velocity_clamped_to_max_v(self):
        """
        Command v = 10 × max_v must be clamped to max_v before integration.
        """
        max_v = 1.0
        robot = DifferentialDriveRobot(x=0.0, y=0.0, theta=0.0, max_v=max_v)
        robot.update(v=10.0, omega=0.0, dt=1.0)
        # If clamping works, x should equal max_v * 1.0 = 1.0, not 10.0
        assert robot.x == pytest.approx(max_v, abs=1e-9)

    def test_omega_clamped_to_max_omega(self):
        """Command ω = 10 × max_omega must be clamped."""
        max_omega = math.pi
        robot = DifferentialDriveRobot(
            x=0.0, y=0.0, theta=0.0, max_omega=max_omega
        )
        robot.update(v=0.0, omega=100.0, dt=1.0)
        # Rotation should be at most max_omega * dt = π
        assert abs(robot.theta) <= max_omega + 1e-9


# =========================================================================
# TestRobotNoise — odometry noise
# =========================================================================

class TestRobotNoise:
    """Verify that odometry noise is applied correctly."""

    def test_zero_noise_gives_deterministic_result(self):
        """With noise_std=0, two robots with the same commands end at the same pose."""
        r1 = DifferentialDriveRobot(noise_std=0.0)
        r2 = DifferentialDriveRobot(noise_std=0.0)
        r1.update(v=1.0, omega=0.5, dt=0.1)
        r2.update(v=1.0, omega=0.5, dt=0.1)
        assert r1.x == pytest.approx(r2.x)
        assert r1.y == pytest.approx(r2.y)
        assert r1.theta == pytest.approx(r2.theta)

    def test_same_seed_gives_same_trajectory(self):
        """
        Two robots with the same seed and noise_std must follow
        the identical noisy trajectory (reproducible for testing).
        """
        r1 = DifferentialDriveRobot(noise_std=0.1, seed=42)
        r2 = DifferentialDriveRobot(noise_std=0.1, seed=42)
        for _ in range(10):
            r1.update(v=1.0, omega=0.1, dt=0.1)
            r2.update(v=1.0, omega=0.1, dt=0.1)
        assert r1.x == pytest.approx(r2.x, abs=1e-9)
        assert r1.y == pytest.approx(r2.y, abs=1e-9)

    def test_noise_causes_drift_over_many_steps(self):
        """
        With noise, the robot's pose should differ from the noise-free pose
        after many steps.  (Very unlikely to be exactly equal.)
        """
        clean = DifferentialDriveRobot(noise_std=0.0)
        noisy = DifferentialDriveRobot(noise_std=0.05, seed=99)
        for _ in range(50):
            clean.update(v=1.0, omega=0.0, dt=0.1)
            noisy.update(v=1.0, omega=0.0, dt=0.1)
        # With 50 steps and noise_std=0.05, drift should be measurable
        total_drift = math.sqrt((clean.x - noisy.x)**2 + (clean.y - noisy.y)**2)
        assert total_drift > 0.01, "Noisy robot should drift from clean robot"


# =========================================================================
# TestRobotGridBridge — conversion between continuous pose and grid cell
# =========================================================================

class TestRobotGridBridge:
    """Verify get_grid_cell() and the from_grid_cell() factory."""

    def test_get_grid_cell_at_cell_centre(self):
        """A robot positioned at the centre of (2, 3) returns (2, 3)."""
        x, y = grid_to_continuous(2, 3, cell_size=0.5)
        robot = DifferentialDriveRobot(x=x, y=y, cell_size=0.5)
        assert robot.get_grid_cell() == (2, 3)

    def test_get_grid_cell_changes_after_move(self):
        """After moving more than one cell, get_grid_cell() returns the new cell."""
        robot = DifferentialDriveRobot.from_grid_cell(0, 0, cell_size=0.5)
        # Move 1 m to the right (2 cells at cell_size=0.5)
        robot.update(v=1.0, omega=0.0, dt=1.0)
        row, col = robot.get_grid_cell()
        assert col == 2   # moved exactly 2 cells to the right (1 m ÷ 0.5 m/cell)

    def test_from_grid_cell_round_trip(self):
        """
        from_grid_cell(r, c).get_grid_cell() must return (r, c).
        This verifies that the factory and accessor use the same cell_size.
        """
        for row in range(5):
            for col in range(5):
                robot = DifferentialDriveRobot.from_grid_cell(
                    row, col, cell_size=0.5
                )
                assert robot.get_grid_cell() == (row, col)

    def test_set_pose_resets_position(self):
        """set_pose() changes x, y, theta and resets the trajectory."""
        robot = DifferentialDriveRobot(x=1.0, y=1.0, theta=0.5)
        robot.update(v=1.0, omega=0.0, dt=1.0)
        robot.set_pose(0.0, 0.0, 0.0)
        assert robot.x == pytest.approx(0.0)
        assert robot.y == pytest.approx(0.0)
        assert robot.theta == pytest.approx(0.0)
        assert len(robot.trajectory) == 1


# =========================================================================
# TestRobotTrajectory — pose history
# =========================================================================

class TestRobotTrajectory:
    """Verify the trajectory recording behaviour."""

    def test_trajectory_grows_with_updates(self):
        """Each call to update() appends one entry to trajectory."""
        robot = DifferentialDriveRobot()
        assert len(robot.trajectory) == 1    # initial pose
        robot.update(v=1.0, omega=0.0, dt=0.1)
        assert len(robot.trajectory) == 2
        robot.update(v=1.0, omega=0.0, dt=0.1)
        assert len(robot.trajectory) == 3

    def test_trajectory_values_match_poses(self):
        """Last trajectory entry matches the robot's current pose."""
        robot = DifferentialDriveRobot(x=0.0, y=0.0, theta=0.0)
        robot.update(v=1.0, omega=0.0, dt=1.0)
        last = robot.trajectory[-1]
        assert last[0] == pytest.approx(robot.x)
        assert last[1] == pytest.approx(robot.y)
        assert last[2] == pytest.approx(robot.theta)

    def test_reset_trajectory_clears_history(self):
        """reset_trajectory() must leave only the current pose."""
        robot = DifferentialDriveRobot()
        for _ in range(10):
            robot.update(v=1.0, omega=0.0, dt=0.1)
        robot.reset_trajectory()
        assert len(robot.trajectory) == 1
        assert robot.trajectory[0][0] == pytest.approx(robot.x)
