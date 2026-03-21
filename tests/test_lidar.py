"""
Unit Tests for LidarSensor
===========================

Test grid layout used in most tests (6×6):

       col: 0  1  2  3  4  5
  row 0:   .  .  X  X  X  .
  row 1:   .  .  X  .  .  .
  row 2:   .  .  X  .  .  .
  row 3:   .  .  .  .  .  .
  row 4:   .  .  .  .  .  .
  row 5:   .  .  .  .  .  .

  X = obstacle   . = free

Robot is placed at the centre of cell (3, 1) → continuous (0.75, 1.75) for
cell_size=0.5. Facing right (theta=0).

Verifies
--------
1.  Ray pointing at a wall returns distance < max_range.
2.  Ray pointing into free space returns max_range.
3.  360° scan on a robot fully surrounded by walls → all readings < max_range.
4.  get_endpoints() round-trips correctly with scan().
5.  Noise-free sensor is deterministic across repeated calls.
6.  Seeded noisy sensor is reproducible (same seed → same readings).
7.  get_beam_angles() returns correct count and span.
8.  Out-of-bounds rays are treated as obstacles (not crashes).
9.  scan() output shape is always (n_beams,).
10. Partial FOV sensor: beams fan around heading only.

Run with: pytest tests/test_lidar.py -v
"""

import sys
sys.path.append('.')

import math
import pytest
import numpy as np

from src.environment.grid_world import GridWorld
from src.sensors.lidar import LidarSensor
from src.utils.conversions import grid_to_continuous, DEFAULT_CELL_SIZE


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def open_env() -> GridWorld:
    """10×10 grid with no obstacles — rays always reach max_range."""
    return GridWorld(width=10, height=10)


@pytest.fixture
def walled_env() -> GridWorld:
    """
    6×6 grid with a partial wall at col=2 (rows 0-2).
    Robot placed at (row=3, col=1): faces right, wall is to the right and
    also to the upper-right.

       col: 0  1  2  3  4  5
  row 0:   .  .  X  X  X  .
  row 1:   .  .  X  .  .  .
  row 2:   .  .  X  .  .  .
  row 3:   .  .  .  .  .  .
    """
    env = GridWorld(width=6, height=6)
    for r in range(3):
        env.add_obstacle(r, 2)
    for c in range(2, 5):
        env.add_obstacle(0, c)
    return env


@pytest.fixture
def box_env() -> GridWorld:
    """
    5×5 grid with obstacles on all four edges — robot at center (2,2)
    is completely surrounded.

       col: 0  1  2  3  4
  row 0:   X  X  X  X  X
  row 1:   X  .  .  .  X
  row 2:   X  .  R  .  X
  row 3:   X  .  .  .  X
  row 4:   X  X  X  X  X
    """
    env = GridWorld(width=5, height=5)
    for c in range(5):
        env.add_obstacle(0, c)
        env.add_obstacle(4, c)
    for r in range(5):
        env.add_obstacle(r, 0)
        env.add_obstacle(r, 4)
    return env


# ──────────────────────────────────────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────────────────────────────────────

def make_lidar(env: GridWorld, **kwargs) -> LidarSensor:
    """Create a LidarSensor with sensible defaults, overridable via kwargs."""
    defaults = dict(n_beams=36, fov=2 * math.pi, max_range=5.0,
                    step_size=0.05, noise_std=0.0, cell_size=DEFAULT_CELL_SIZE)
    defaults.update(kwargs)
    return LidarSensor(env, **defaults)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Open environment — all rays hit max_range
# ──────────────────────────────────────────────────────────────────────────────

class TestOpenEnvironment:
    """In a fully open grid, every beam should saturate at max_range."""

    def test_all_beams_reach_max_range(self, open_env):
        # max_range=2.0: robot at (5,5) centre → (2.75m, 2.75m) in a 10×10 grid
        # (5m × 5m). Nearest boundary is 2.25m away (right/bottom walls).
        # max_range=2.0 < 2.25m guarantees no ray exits the grid, so all
        # readings should saturate at exactly max_range.
        lidar = make_lidar(open_env, max_range=2.0)
        x, y = grid_to_continuous(5, 5)
        ranges = lidar.scan(x, y, theta=0.0)
        assert np.allclose(ranges, 2.0), \
            f"Expected all 2.0, got min={ranges.min():.3f}, max={ranges.max():.3f}"

    def test_output_shape(self, open_env):
        lidar = make_lidar(open_env, n_beams=72)
        x, y = grid_to_continuous(5, 5)
        ranges = lidar.scan(x, y, theta=0.0)
        assert ranges.shape == (72,)

    def test_output_dtype_is_float(self, open_env):
        lidar = make_lidar(open_env)
        x, y = grid_to_continuous(5, 5)
        ranges = lidar.scan(x, y, theta=0.0)
        assert ranges.dtype == float or np.issubdtype(ranges.dtype, np.floating)


# ──────────────────────────────────────────────────────────────────────────────
# 2. Ray pointing directly at a wall
# ──────────────────────────────────────────────────────────────────────────────

class TestRayHitsWall:
    """
    Robot at cell (3, 1), facing right (theta=0).
    There is a wall at col=2 (rows 0-2). Row 3 col=2 is free.

    A ray fired straight right (angle=0) from (3,1) won't hit the partial
    wall (gap at row 3). But firing slightly upward (into rows 0-2) will hit.

    Instead we fire from (row=1, col=1) facing right — wall is at col=2.
    """

    def test_ray_hits_wall_returns_short_range(self, walled_env):
        lidar = make_lidar(walled_env, n_beams=1, fov=0.0001, max_range=5.0)
        # Robot centre of cell (1, 1) → (0.75, 0.75)
        x, y = grid_to_continuous(1, 1)
        # Single beam pointing exactly right (theta=0 → beam at angle ≈ 0)
        ranges = lidar.scan(x, y, theta=0.0)
        # Wall at col=2 is 0.5 m away from centre of col=1
        assert ranges[0] < lidar.max_range, \
            f"Expected hit < {lidar.max_range}, got {ranges[0]:.3f}"
        # Should be roughly 0.5 m (half a cell) ± step size tolerance
        assert ranges[0] < 1.0, \
            f"Wall is ~0.5m away, got {ranges[0]:.3f}"

    def test_ray_into_free_space_returns_valid_range(self, walled_env):
        lidar = make_lidar(walled_env, n_beams=1, fov=0.0001, max_range=3.0)
        # Robot at cell (3, 1), firing LEFT (theta=π) — free space in that direction
        x, y = grid_to_continuous(3, 1)
        ranges = lidar.scan(x, y, theta=math.pi)
        # The beam goes left: col 0 is free, then out of bounds (treated as obstacle)
        # So we expect a hit at the map boundary, not max_range — unless boundary
        # is further than max_range. Check it doesn't exceed max_range.
        assert 0 < ranges[0] <= lidar.max_range


# ──────────────────────────────────────────────────────────────────────────────
# 3. Robot surrounded by walls — all beams hit
# ──────────────────────────────────────────────────────────────────────────────

class TestBoxEnvironment:
    """In the box environment robot at center (2,2) is surrounded by walls."""

    def test_all_beams_hit_wall(self, box_env):
        lidar = make_lidar(box_env, n_beams=36, max_range=5.0)
        x, y = grid_to_continuous(2, 2)
        ranges = lidar.scan(x, y, theta=0.0)
        assert np.all(ranges < lidar.max_range), \
            f"Expected all beams to hit walls, max reading = {ranges.max():.3f}"

    def test_all_beams_roughly_symmetric(self, box_env):
        """A 5×5 box is symmetric, so all ranges should be similar."""
        lidar = make_lidar(box_env, n_beams=4, fov=2 * math.pi, max_range=5.0)
        x, y = grid_to_continuous(2, 2)
        # Fire 4 beams at 90° intervals (right, down, left, up)
        ranges = lidar.scan(x, y, theta=math.pi / 4)
        # All 4 readings should be within a step_size of each other
        assert ranges.max() - ranges.min() < 0.15, \
            f"Expected symmetric ranges, got: {ranges}"


# ──────────────────────────────────────────────────────────────────────────────
# 4. get_endpoints() round-trip
# ──────────────────────────────────────────────────────────────────────────────

class TestGetEndpoints:
    """get_endpoints() must reconstruct hit positions consistent with scan()."""

    def test_endpoints_shape(self, open_env):
        lidar = make_lidar(open_env, n_beams=36)
        x, y = grid_to_continuous(5, 5)
        ranges = lidar.scan(x, y, theta=0.0)
        endpoints = lidar.get_endpoints(x, y, theta=0.0, ranges=ranges)
        assert endpoints.shape == (36, 2)

    def test_endpoints_distance_matches_range(self, box_env):
        """
        Distance from robot to each endpoint must equal the corresponding range.
        The endpoint is: robot_pos + range * (cos(angle), sin(angle))
        So dist(robot, endpoint) == range (exactly, modulo float precision).
        """
        lidar = make_lidar(box_env, n_beams=36)
        x, y = grid_to_continuous(2, 2)
        theta = 0.0
        ranges = lidar.scan(x, y, theta)
        endpoints = lidar.get_endpoints(x, y, theta, ranges)

        for i in range(lidar.n_beams):
            dx = endpoints[i, 0] - x
            dy = endpoints[i, 1] - y
            dist = math.sqrt(dx**2 + dy**2)
            assert abs(dist - ranges[i]) < 1e-9, \
                f"Beam {i}: endpoint distance {dist:.6f} ≠ range {ranges[i]:.6f}"

    def test_endpoints_dtype(self, open_env):
        lidar = make_lidar(open_env, n_beams=10)
        x, y = grid_to_continuous(5, 5)
        ranges = lidar.scan(x, y, theta=0.0)
        endpoints = lidar.get_endpoints(x, y, theta=0.0, ranges=ranges)
        assert np.issubdtype(endpoints.dtype, np.floating)


# ──────────────────────────────────────────────────────────────────────────────
# 5. Determinism (noise-free)
# ──────────────────────────────────────────────────────────────────────────────

class TestDeterminism:
    """A noise-free sensor must return identical results across repeated calls."""

    def test_noise_free_is_deterministic(self, box_env):
        lidar = make_lidar(box_env, noise_std=0.0)
        x, y = grid_to_continuous(2, 2)
        r1 = lidar.scan(x, y, theta=0.0)
        r2 = lidar.scan(x, y, theta=0.0)
        assert np.array_equal(r1, r2), \
            "Noise-free scan must be identical on repeated calls"

    def test_different_headings_differ(self, walled_env):
        """Scanning at different headings should give different results
        in an asymmetric environment."""
        lidar = make_lidar(walled_env, n_beams=36, noise_std=0.0)
        x, y = grid_to_continuous(1, 1)
        r0   = lidar.scan(x, y, theta=0.0)         # facing right (toward wall)
        r180 = lidar.scan(x, y, theta=math.pi)     # facing left  (open space)
        # The two scans should not be identical
        assert not np.array_equal(r0, r180)


# ──────────────────────────────────────────────────────────────────────────────
# 6. Noisy sensor reproducibility
# ──────────────────────────────────────────────────────────────────────────────

class TestNoise:
    """Seeded noisy sensors must be reproducible; unseeded ones must differ."""

    def test_seeded_noise_is_reproducible(self, box_env):
        x, y = grid_to_continuous(2, 2)
        lidar_a = make_lidar(box_env, noise_std=0.05, seed=42)
        lidar_b = make_lidar(box_env, noise_std=0.05, seed=42)
        r_a = lidar_a.scan(x, y, theta=0.0)
        r_b = lidar_b.scan(x, y, theta=0.0)
        assert np.array_equal(r_a, r_b), \
            "Same seed must produce identical noisy scans"

    def test_different_seeds_produce_different_readings(self, box_env):
        x, y = grid_to_continuous(2, 2)
        lidar_a = make_lidar(box_env, noise_std=0.2, seed=1)
        lidar_b = make_lidar(box_env, noise_std=0.2, seed=2)
        r_a = lidar_a.scan(x, y, theta=0.0)
        r_b = lidar_b.scan(x, y, theta=0.0)
        assert not np.array_equal(r_a, r_b), \
            "Different seeds should produce different noisy readings"

    def test_noise_clamps_to_valid_range(self, box_env):
        """Noisy readings must stay in [0, max_range] regardless of noise magnitude."""
        lidar = make_lidar(box_env, noise_std=10.0, max_range=5.0, seed=0)
        x, y = grid_to_continuous(2, 2)
        ranges = lidar.scan(x, y, theta=0.0)
        assert np.all(ranges >= 0.0), f"Negative range detected: {ranges.min()}"
        assert np.all(ranges <= lidar.max_range), \
            f"Range exceeds max: {ranges.max()} > {lidar.max_range}"

    def test_noise_changes_readings(self, box_env):
        """Adding noise should produce readings different from noise-free scan."""
        x, y = grid_to_continuous(2, 2)
        clean = make_lidar(box_env, noise_std=0.0)
        noisy = make_lidar(box_env, noise_std=0.5, seed=99)
        r_clean = clean.scan(x, y, theta=0.0)
        r_noisy = noisy.scan(x, y, theta=0.0)
        assert not np.array_equal(r_clean, r_noisy)


# ──────────────────────────────────────────────────────────────────────────────
# 7. get_beam_angles()
# ──────────────────────────────────────────────────────────────────────────────

class TestGetBeamAngles:
    """Beam angles must span the correct arc and have the correct count."""

    def test_angle_count(self, open_env):
        lidar = make_lidar(open_env, n_beams=36)
        angles = lidar.get_beam_angles(theta=0.0)
        assert angles.shape == (36,)

    def test_full_fov_span(self, open_env):
        """
        For 360° FOV with theta=0:
          beam 0   should be at -π   (theta - fov/2)
          beam N-1 should be at -π + (N-1)/N * 2π
        """
        lidar = make_lidar(open_env, n_beams=36, fov=2 * math.pi)
        angles = lidar.get_beam_angles(theta=0.0)
        expected_first = -math.pi
        expected_last  = -math.pi + 35 / 36 * 2 * math.pi
        assert abs(angles[0]  - expected_first) < 1e-9
        assert abs(angles[-1] - expected_last)  < 1e-9

    def test_half_fov_around_heading(self, open_env):
        """
        For 180° FOV with theta=π/2 (facing down):
          beam 0 at  π/2 - π/2 = 0     (right)
          beam N-1 near π/2 + π/2 = π  (left), but not quite (spacing < fov/n)
        """
        lidar = make_lidar(open_env, n_beams=18, fov=math.pi)
        angles = lidar.get_beam_angles(theta=math.pi / 2)
        assert abs(angles[0] - 0.0) < 1e-9, f"First angle: {angles[0]}"
        assert angles[-1] < math.pi   # last beam is just below π

    def test_angles_rotate_with_theta(self, open_env):
        """Changing theta should shift all angles by the same amount."""
        lidar = make_lidar(open_env, n_beams=36)
        a0 = lidar.get_beam_angles(theta=0.0)
        a1 = lidar.get_beam_angles(theta=math.pi / 4)
        diff = a1 - a0
        # All differences should be π/4
        assert np.allclose(diff, math.pi / 4), \
            f"Expected uniform shift of π/4, got: {diff[:3]}"


# ──────────────────────────────────────────────────────────────────────────────
# 8. Out-of-bounds rays treated as obstacle hits
# ──────────────────────────────────────────────────────────────────────────────

class TestOutOfBounds:
    """Rays leaving the map boundary must stop (is_valid() returns False OOB)."""

    def test_ray_hits_boundary_not_max_range(self):
        """
        Small 3×3 grid, no obstacles, robot at corner (0,0).
        Max range = 10 m >> grid size (~1.5 m). Rays should hit the boundary
        well before max_range.
        """
        env = GridWorld(width=3, height=3)
        lidar = make_lidar(env, n_beams=4, fov=2 * math.pi, max_range=10.0)
        x, y = grid_to_continuous(0, 0)
        ranges = lidar.scan(x, y, theta=0.0)
        # At least some beams exit the 3×3 grid and hit the boundary
        assert np.any(ranges < lidar.max_range), \
            "Expected boundary hits for a robot at the corner of a small grid"

    def test_readings_never_exceed_max_range(self):
        """Boundary hits must still be capped at max_range."""
        env = GridWorld(width=3, height=3)
        lidar = make_lidar(env, n_beams=36, max_range=2.0)
        x, y = grid_to_continuous(1, 1)
        ranges = lidar.scan(x, y, theta=0.0)
        assert np.all(ranges <= lidar.max_range)


# ──────────────────────────────────────────────────────────────────────────────
# 9. Partial FOV sensor
# ──────────────────────────────────────────────────────────────────────────────

class TestPartialFOV:
    """A 180° FOV sensor should only scan the forward hemisphere."""

    def test_half_fov_beams_face_forward(self, open_env):
        """
        Robot facing right (theta=0), 180° FOV.
        All beam angles should be in [-π/2, π/2] (the right hemisphere).
        """
        lidar = make_lidar(open_env, n_beams=18, fov=math.pi)
        angles = lidar.get_beam_angles(theta=0.0)
        assert np.all(angles >= -math.pi / 2 - 1e-9)
        assert np.all(angles <= math.pi / 2 + 1e-9)

    def test_partial_fov_fewer_hits_than_full(self, box_env):
        """
        In a box environment, a 360° sensor sees walls everywhere.
        A 90° sensor (one quadrant only) sees fewer or equal hits — never more.
        """
        x, y = grid_to_continuous(2, 2)
        full  = make_lidar(box_env, n_beams=36, fov=2 * math.pi)
        partial = make_lidar(box_env, n_beams=9, fov=math.pi / 2)

        r_full    = full.scan(x, y, theta=0.0)
        r_partial = partial.scan(x, y, theta=0.0)

        hits_full    = np.sum(r_full    < full.max_range)
        hits_partial = np.sum(r_partial < partial.max_range)

        # The box surrounds the robot, so all beams in both scans should hit
        # +1 tolerance: boundary beams at exactly ±FOV/2 may be counted
        # differently due to floating-point rounding at the arc edges
        assert hits_partial <= hits_full + 1


# ──────────────────────────────────────────────────────────────────────────────
# 10. __repr__
# ──────────────────────────────────────────────────────────────────────────────

class TestRepr:
    def test_repr_contains_key_info(self, open_env):
        lidar = make_lidar(open_env, n_beams=36, max_range=5.0)
        r = repr(lidar)
        assert "LidarSensor" in r
        assert "36" in r
        assert "5.0" in r
