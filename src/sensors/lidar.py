"""
Simulated LIDAR Sensor
======================

LIDAR (Light Detection And Ranging) measures distances by firing laser beams
and timing the return of reflected light. A spinning LIDAR fires N beams in 
a 360° arc and returns N range readings — one per beam — telling the robot how far away 
the nearest obstacle is in each direction.

HOW THIS SIMULATION WORKS
--------------------------
Real LIDARs work in 3D with precise timing. We simulate a 2D top-down slice:

  1. The robot has a continuous pose (x, y, θ) in meters.
  2. The environment stores obstacles as a discrete grid.
  3. For each beam, we traverse a "ray" in continuous space, stepping by `step_size` meters each iteration.
  4. At each step we convert the ray tip to a grid cell and call env.is_valid() to check for an obstacle.
  5. First invalid cell = hit → return the distance accumulated so far.
  6. If no hit after `max_range` metres → return max_range (sensor saturation).

WHY STEP SIZE IS A FIXED FLOAT (Design Decision A)
----------------------------------------------------
We use a caller-specified `step_size` in raw meters (default 0.05 m) rather
than deriving it as a fraction of cell_size. Reason: it matches how every
other physical parameter in this project is specified — concrete meters, not
derived fractions. 

WHY RAYS RETURN max_range (NOT np.inf) ON MISS (Design Decision B)
--------------------------------------------------------------------
Real LIDAR hardware saturates at its rated range. Returning max_range mimics
that behaviour and avoids inf-handling in downstream consumers (DWA, obstacle
maps). Code that checks `range < max_range` cleanly identifies obstacles.

WHY BEAMS ARE ROBOT-RELATIVE (Design Decision C)
-------------------------------------------------
A real spinning LIDAR is bolted to the robot. Beam 0 is always directly
behind the robot (for 360° FOV), and the sweep rotates with the robot. This
is the physically correct model. World-absolute beams would mean the sensor
magically re-orients itself as the robot turns — which makes no sense.
"""

import math
from typing import Optional

import numpy as np

from src.environment.grid_world import GridWorld
from src.utils.conversions import continuous_to_grid, DEFAULT_CELL_SIZE


class LidarSensor:
    """
    Simulated 2D LIDAR sensor that ray-casts in continuous space and checks
    obstacles in the GridWorld's discrete grid.

    The sensor fires `n_beams` rays fanning out symmetrically around the
    robot's heading `theta`. For a 360° FOV this covers the full circle;
    for a 180° FOV it covers the front half.

    Parameters
    ----------
    environment : GridWorld
        The obstacle map. Ray-casting calls `env.is_valid(row, col)` to
        detect hits. Out-of-bounds cells are treated as obstacles.
    n_beams : int
        Number of rays. 36 → one every 10°, 360 → one every 1°.
    fov : float
        Angular span of the full scan in radians. Default 2π = 360°.
    max_range : float
        Maximum detectable distance in metres. Rays that reach this
        distance without hitting anything return max_range (sensor
        saturation — same behaviour as real hardware).
    step_size : float
        Spatial resolution of each ray in metres. Smaller = more accurate
        hit position but slower. 0.05 m is a good default for 0.5 m cells.
    noise_std : float
        Standard deviation of Gaussian range noise added after each ray
        cast. 0.0 = noise-free (deterministic). Units: metres.
    cell_size : float
        Physical size of one grid cell. Must match the GridWorld in use.
    seed : int | None
        Random seed for reproducible noise (useful in tests).
    """

    def __init__(
        self,
        environment: GridWorld,
        n_beams:    int   = 36,
        fov:        float = 2 * math.pi,
        max_range:  float = 5.0,
        step_size:  float = 0.05,
        noise_std:  float = 0.0,
        cell_size:  float = DEFAULT_CELL_SIZE,
        seed: Optional[int] = None,
    ) -> None:
        """
        Sets up the sensor's configuration. Setting Lidar parameters: 
        how many beams, how far it can see, how accurate it is — just configuring the hardware.
        """
        self._env      = environment       # private 
        self.n_beams   = n_beams
        self.fov       = fov
        self.max_range = max_range
        self.step_size = step_size
        self.noise_std = noise_std
        self.cell_size = cell_size

        # PCG64 Generator — same pattern as DifferentialDriveRobot.
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _cast_ray(self, x: float, y: float, angle: float) -> float:
        """
        Fires one single laser beam from position (x,y) in one direction. 
        Walks forward step by step until it hits a wall or reaches max distance. 
        Returns one number: how far the beam traveled. This is the core physics of the sensor.
        
        Algorithm
        ---------
        We step a "probe point" along the ray in increments of step_size.
        At each position we convert to grid indices and call is_valid():
          - is_valid() returns False for obstacles AND out-of-bounds cells,
            so the grid boundary acts like a wall automatically.
          - On a hit, we return the accumulated distance `d`.
          - If `d` reaches max_range with no hit, we return max_range.

        Why step along the ray rather than using a closed-form intersection?
        ----
        Closed-form ray-cell intersection (DDA / Bresenham) is faster but
        more complex to implement. Step-wise marching is simpler to read,
        debug, and explain — which fits this learning project's goals. The
        accuracy difference is small when step_size ≤ cell_size/10.

        Parameters
        ----------
        x, y : float
            Ray origin in continuous coordinates (metres).
        angle : float
            Beam direction in radians (robot-relative convention, y-downward).

        Returns
        -------
        float
            Distance to hit in metres, in [0, max_range].
        """
        # Compute once before the loop (expensive trigonometric operations)
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)

        d = 0.0
        while d < self.max_range:
            # Advance the probe point by one step (so the ray never checks its own origin cell -> position of the robot)
            d += self.step_size

            # Probe tip in continuous space
            tip_x = x + d * cos_a
            tip_y = y + d * sin_a

            # Convert to grid cell
            row, col = continuous_to_grid(tip_x, tip_y, self.cell_size)

            # is_valid() returns False for both obstacles AND out-of-bounds.
            if not self._env.is_valid(row, col):
                return d

        # Reached max_range with no obstacle → sensor saturation
        return self.max_range

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scan(self, x: float, y: float, theta: float) -> np.ndarray:
        """
        Fires ALL beams at once from the robot's current pose. 
        Calls _cast_ray N times, once per beam, each at a slightly different angle fanning out around the robot's heading. 
        Returns an array of N distances — one per beam. This is what DWA will call every time step.

        Noise is added AFTER each raw ray cast and clamped to [0, max_range]
        so readings are physically plausible (non-negative, within range).

        Parameters
        ----------
        x, y : float
            Robot position in continuous coordinates (metres).
        theta : float
            Robot heading in radians (y-downward convention).

        Returns
        -------
        np.ndarray, shape (n_beams,)
            Range reading for each beam in metres.
        """
        # Dividing fov by n_beams gives equal arc segments --> fov=2π, n_beams=36  →  angle_step = π/18 ≈ 10°
        angle_step = self.fov / self.n_beams

        # Beam 0 starts at theta - fov/2 so the sweep is symmetric
        beam_0_angle = theta - self.fov / 2.0

        ranges = np.empty(self.n_beams, dtype=float)

        for i in range(self.n_beams):
            # Absolute angle for this beam (world frame, y-downward)
            angle = beam_0_angle + i * angle_step

            # Cast the ray and get the raw (noise-free) range
            raw_range = self._cast_ray(x, y, angle)

            # Add Gaussian noise if configured, then clamp to valid range.
            if self.noise_std > 0.0:
                noisy = raw_range + self._rng.normal(0.0, self.noise_std)
                ranges[i] = float(np.clip(noisy, 0.0, self.max_range))
            else:
                ranges[i] = raw_range

        return ranges

    def get_beam_angles(self, theta: float) -> np.ndarray:
        """
        Return the absolute angle of each beam for the given robot heading.

        Answers the question: "which direction is each beam pointing?" Returns N angles. 
        Used by visualization and DWA to know not just how far an obstacle is, but in which direction

        The angles mirror exactly what scan() uses internally, so
        get_endpoints() can pair each angle with its corresponding range.

        Parameters
        ----------
        theta : float
            Robot heading in radians.

        Returns
        -------
        np.ndarray, shape (n_beams,)
            Absolute beam angles in radians.
        """
        angle_step  = self.fov / self.n_beams
        beam_0_angle = theta - self.fov / 2.0

        # np.arange produces beam indices [0, 1, ..., n_beams-1]
        # vectorised step (operates on all 36 values simultaneously in optimized C code under the hood).
        return beam_0_angle + np.arange(self.n_beams) * angle_step

    def get_endpoints(
        self, x: float, y: float, theta: float, ranges: np.ndarray
    ) -> np.ndarray:
        """
        Convert a scan's range readings to (x, y) hit-point coordinates.
        In practice, converts distances into actual (x,y) coordinates. 
        If beam 5 detected something 1.2m away at angle 45°, this computes the exact point in space where that obstacle is. 
        Useful for:
          - Matplotlib scatter plots of detected obstacles.
          - Building a local occupancy map around the robot.
          - DWA: checking which directions have obstacles nearby.

        Parameters
        ----------
        x, y : float
            Robot position (the ray origin) in metres.
        theta : float
            Robot heading in radians.
        ranges : np.ndarray, shape (n_beams,)
            Output of scan(x, y, theta) — range per beam in metres.

        Returns
        -------
        np.ndarray, shape (n_beams, 2)
            Row i is (endpoint_x, endpoint_y) for beam i.
        """
        angles = self.get_beam_angles(theta)

        # endpoint = origin + range * (cos(angle), sin(angle))
        endpoints = np.empty((self.n_beams, 2), dtype=float)
        endpoints[:, 0] = x + ranges * np.cos(angles)   # x-coordinates
        endpoints[:, 1] = y + ranges * np.sin(angles)   # y-coordinates

        return endpoints

    def __repr__(self) -> str:
        return (
            f"LidarSensor("
            f"n_beams={self.n_beams}, "
            f"fov={math.degrees(self.fov):.0f}°, "
            f"max_range={self.max_range}m, "
            f"step_size={self.step_size}m, "
            f"noise_std={self.noise_std}m)"
        )
