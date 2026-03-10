"""
Differential Drive Robot Model
================================

WHAT IS A DIFFERENTIAL DRIVE ROBOT?
-------------------------------------
A differential drive robot has two independently driven wheels on a common
axis. By varying the speed of each wheel it can:
  - Move straight (both wheels same speed)
  - Turn in place (wheels at equal but opposite speeds)
  - Follow a curved path (wheels at different speeds)

THE STATE: POSE (x, y, θ)
---------------------------
The robot's complete state at any moment is its POSE:
  x     — position along the horizontal axis (meters, increases rightward)
  y     — position along the vertical axis   (meters, increases downward)
  theta — heading angle

COORDINATE / ANGLE CONVENTION
-------------------------------
We use the same y-downward convention as GridWorld (see src/utils/conversions.py):
  theta =  0     → facing RIGHT  (+x)
  theta =  π/2   → facing DOWN   (+y = +row in grid)
  theta =  π     → facing LEFT   (-x)
  theta = -π/2   → facing UP     (-y = -row in grid)

WHY y-downward? GridWorld rows increase downward. Matching that convention
avoids coordinate flips when converting between the two systems.

CONTROL INPUTS: (v, ω)
------------------------
Rather than specifying left/right wheel speeds directly, we use the
abstracted command pair:
  v  — linear  velocity (m/s): how fast the robot moves forward
  ω  — angular velocity (rad/s): how fast the robot rotates

Positive ω rotates clockwise (increases theta) because theta increases
downward in my convention.

EULER INTEGRATION (the kinematic update)
------------------------------------------
Given pose (x, y, θ) and commands (v, ω) over a time step dt:

  x_new     = x     + v * cos(θ) * dt
  y_new     = y     + v * sin(θ) * dt
  theta_new = theta + ω * dt

ODOMETRY NOISE
---------------
Real wheel encoders are imperfect. Wheels slip, surfaces are uneven,
encoder resolution is finite. We model this by adding Gaussian noise
to (v, ω) before integration:
  v_noisy = v + N(0, noise_std)
  ω_noisy = ω + N(0, noise_std)

Over time, the noisy pose drifts away from the true pose — this is
called ODOMETRY DRIFT and is why real robots need localization (SLAM,
GPS, LIDAR matching) to correct the accumulated error.
"""

import math
from typing import Optional, Tuple

import numpy as np

from src.utils.conversions import (
    DEFAULT_CELL_SIZE,
    continuous_to_grid,
    grid_to_continuous,
)


class DifferentialDriveRobot:
    """
    Kinematic model of a differential drive robot in continuous 2D space.

    Responsibilities (Single Responsability Principle)
    ----------------
    - Maintain the robot's pose (x, y, theta)
    - Update the pose using Euler integration when given velocity commands
    - Optionally simulate odometry noise
    - Convert between continuous pose and grid cell

    What this class does NOT do
    ----------------------------
    - Collision detection (that's the environment's job)
    - Path planning       (that's the planner's job)
    - Sensor simulation   (that's the LIDAR's job)
    - Control decisions   (that's the DWA controller's job)

    Parameters
    ----------
    x, y : float
        Initial position in meters.
    theta : float
        Initial heading in radians (normalised to [-π, π] automatically).
    cell_size : float
        Physical size of one grid cell in meters.  Must match the
        GridWorld's cell_size used in the coordinate conversions.
    wheel_base : float
        Distance between the two wheels in meters.  Used only for
        informational purposes in this abstracted model (v, ω input).
    max_v : float
        Maximum linear velocity (m/s).  Commands exceeding this are clamped.
    max_omega : float
        Maximum angular velocity (rad/s).  Commands exceeding this are clamped.
    noise_std : float
        Standard deviation of Gaussian odometry noise applied to (v, ω).
        0.0 means a perfect, noise-free robot.
    seed : int | None
        Random seed for reproducible noise (useful in tests).
    """

    def __init__(
        self,
        x:          float = 0.0,
        y:          float = 0.0,
        theta:      float = 0.0,
        cell_size:  float = DEFAULT_CELL_SIZE,
        wheel_base: float = 0.3,
        max_v:      float = 1.0,
        max_omega:  float = math.pi,
        noise_std:  float = 0.0,         # noise 
        seed: Optional[int] = None,
    ) -> None:
        self.cell_size  = cell_size
        self.wheel_base = wheel_base
        self.max_v      = max_v         # Limit
        self.max_omega  = max_omega     # Limit
        self.noise_std  = noise_std

        # PCG64 is a specific mathematical algorithm (Permuted Congruential Generator, 64-bit) that produces high-quality pseudo-random numbers
        self._rng = np.random.default_rng(seed)   # →  Generator(PCG64)

        # Set initial pose (theta is normalised immediately)
        self.x     = float(x)
        self.y     = float(y)
        self.theta = self._normalize_angle(float(theta))

        # History of poses for trajectory visualisation Each entry is (x, y, theta) after a call to update()
        self.trajectory: list = [(self.x, self.y, self.theta)]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """
        It needs nothing from the instance (not self). It wrap an angle to the range [-π, π].

        Without this, theta grows without bound over a long
        run (e.g., after many clockwise turns theta could reach 100π).

        The atan2(sin, cos) trick is the standard way to normalise:
          atan2(sin(x), cos(x)) always returns a value in (-π, π].
        """
        return math.atan2(math.sin(angle), math.cos(angle))

    # ------------------------------------------------------------------
    # Core kinematics
    # ------------------------------------------------------------------

    def update(self, v: float, omega: float, dt: float) -> None:
        """
        Advance the robot's pose by one time step using Euler integration.

        The update equations (derived by the user):
          x_new     = x     + v * cos(θ) * dt
          y_new     = y     + v * sin(θ) * dt
          theta_new = theta + ω * dt

        Velocity commands are clamped to [−max_v, max_v] and
        [−max_omega, max_omega] before integration, then optionally
        corrupted with Gaussian noise to simulate odometry imperfection.

        Parameters
        ----------
        v : float
            Commanded linear velocity (m/s).  Positive = forward.
        omega : float
            Commanded angular velocity (rad/s).  Positive = clockwise
            (increases theta, which faces downward in my convention).
        dt : float
            Time step duration (seconds).  Keep ≤ 0.1 s for accuracy.
        """
        # 1. Clamp commands to the robot's physical limits
        v     = float(np.clip(v,     -self.max_v,     self.max_v))
        omega = float(np.clip(omega, -self.max_omega, self.max_omega))

        # 2. Add odometry noise (zero-mean Gaussian on each command)
        if self.noise_std > 0.0:
            v     += float(self._rng.normal(0.0, self.noise_std))
            omega += float(self._rng.normal(0.0, self.noise_std))

        # 3. Euler integration
        self.x     += v * math.cos(self.theta) * dt
        self.y     += v * math.sin(self.theta) * dt
        self.theta += omega * dt

        # 4. Normalise theta to keep it in [-π, π]
        self.theta = self._normalize_angle(self.theta)

        # 5. Record pose for trajectory visualisation
        self.trajectory.append((self.x, self.y, self.theta))

    # ------------------------------------------------------------------
    # Pose accessors
    # ------------------------------------------------------------------

    def get_pose(self) -> Tuple[float, float, float]:
        """Return the current pose as (x, y, theta)."""
        return (self.x, self.y, self.theta)

    def set_pose(self, x: float, y: float, theta: float) -> None:
        """
        Teleport the robot to a new pose (for initialization or testing).
        Normalizes theta and resets the trajectory.
        """
        self.x     = float(x)
        self.y     = float(y)
        self.theta = self._normalize_angle(float(theta))
        self.trajectory = [(self.x, self.y, self.theta)]

    # ------------------------------------------------------------------
    # Grid interface
    # ------------------------------------------------------------------

    def get_grid_cell(self) -> Tuple[int, int]:
        """
        Return the grid cell (row, col) that contains the robot's current
        continuous position.
        """
        return continuous_to_grid(self.x, self.y, self.cell_size)

    @classmethod
    def from_grid_cell(
        cls,                      #  the class itself (the blueprint for making robots)   
        row:        int,  
        col:        int,
        theta:      float = 0.0,
        cell_size:  float = DEFAULT_CELL_SIZE,
        **kwargs,                 # pass any extra parameters
    ) -> "DifferentialDriveRobot":
        """
        Convenience constructor: create a robot placed at the centre of
        a grid cell (row, col) facing direction theta.

        Example
        -------
        robot = DifferentialDriveRobot.from_grid_cell(2, 3, theta=0.0)
        # Robot starts at the centre of grid cell (row=2, col=3)
        """
        x, y = grid_to_continuous(row, col, cell_size)
        return cls(x=x, y=y, theta=theta, cell_size=cell_size, **kwargs)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def reset_trajectory(self) -> None:
        """Clear the stored trajectory (useful before a new run)."""
        self.trajectory = [(self.x, self.y, self.theta)]

    def __repr__(self) -> str:
        return (
            f"DifferentialDriveRobot("
            f"x={self.x:.3f}, y={self.y:.3f}, "
            f"θ={math.degrees(self.theta):.1f}°, "
            f"cell={self.get_grid_cell()})"
        )
