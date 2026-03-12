"""
Dynamic Window Approach (DWA) Local Planner
============================================

WHAT IS DWA?
-------------
DWA is a real-time local planner for mobile robots. It answers the question:
"Given where I am, what I sense, and where I want to go — what velocity
command should I send RIGHT NOW?"

It sits between the global planner (BFS/A* gives high-level waypoints) and
the robot actuators (robot.update() executes the command):

    Global planner (A*) ──► waypoint
                                │
                                ▼
    Sensor (LIDAR) ──► obstacle points
                                │
                                ▼
                       DWAController.compute_command()
                                │
                                ▼
                        (v, ω) command ──► robot.update()

WHY "DYNAMIC WINDOW"?
----------------------
The robot's motors have limits. You can't instantly jump from v=0 to v=1.0 —
acceleration takes time. The "dynamic window" is the set of (v, ω) velocities
the robot can PHYSICALLY REACH in the next time step given its current speed
and acceleration constraints. DWA only samples candidates inside this window,
so every command it returns is guaranteed to be executable.

THE THREE SCORING CRITERIA
---------------------------
For each candidate trajectory, DWA evaluates:

  1. HEADING   — does the trajectory end up facing the goal?
  2. CLEARANCE — how far does the trajectory stay from obstacles?
  3. VELOCITY  — how fast does the trajectory move forward?

A weighted sum combines the three. The weights let you tune the planner:
  - High w_heading → aggressive navigation toward goal
  - High w_clearance → cautious, obstacle-averse behaviour
  - High w_velocity → prefer fast trajectories

SINGLE RESPONSIBILITY
----------------------
DWAController ONLY selects velocity commands. It does NOT:
  - Move the robot (that's robot.update())
  - Read the sensor (that's lidar.scan())
  - Manage the environment (that's GridWorld)

Caller responsibilities (see examples/test_dwa.py):
  1. Call lidar.scan() to get ranges
  2. Call lidar.get_endpoints() to get (x,y) obstacle points
  3. Filter out max_range phantom points: obstacles = endpoints[ranges < lidar.max_range]
  4. Call dwa.compute_command() with the filtered obstacle array
  5. If success, call robot.update(v, omega, dt) with the result
  6. If not success, handle failure (stop, re-plan, spin in place, etc.)
"""

import math
from typing import Tuple

import numpy as np


class DWAController:
    """
    Dynamic Window Approach local planner.

    Samples a grid of (v, ω) velocity candidates within the robot's dynamic
    window, simulates each trajectory, discards collisions, scores survivors
    on heading/clearance/velocity, and returns the best command.

    Responsibilities
    ----------------
    - Decide (v, ω) given: current pose, current velocity, obstacle points, waypoint
    - Return a success flag so the caller can handle the all-collision case

    What this class does NOT do
    ----------------------------
    - Move the robot (robot.update() is the caller's job)
    - Read the LIDAR (lidar.scan() is the caller's job)
    - Filter phantom obstacle points (caller does: endpoints[ranges < max_range])

    Parameters
    ----------
    max_v : float
        Maximum linear velocity (m/s). Must match the robot's max_v.
    max_omega : float
        Maximum angular velocity (rad/s). Must match the robot's max_omega.
    max_dv : float
        Maximum linear acceleration (m/s²). Controls how quickly the robot
        can change its forward speed. Typical value: 0.5–2.0 m/s².
    max_domega : float
        Maximum angular acceleration (rad/s²). Controls turning agility.
        Typical value: π/2 to 2π rad/s².
    robot_radius : float
        Safety margin in metres. Trajectories that bring the robot within
        this distance of an obstacle are discarded. Should be ≥ actual
        robot half-width. Typical value: 0.15–0.3 m.
    sim_time : float
        How many seconds to simulate each trajectory. Longer = better
        foresight but more computation. Default 1.0 s.
    dt : float
        Euler integration step for trajectory simulation (seconds).
        Default 0.1 s → 10 steps per trajectory at sim_time=1.0.
    v_samples : int
        Number of linearly-spaced linear velocity candidates within the
        dynamic window. More → finer grid but slower. Default 10.
    omega_samples : int
        Number of angular velocity candidates. Default 20.
    clearance_cap : float
        Clearance distances above this value (metres) are all treated as
        equally safe — the score is capped at 1.0. Prevents far-away
        obstacles from dominating the clearance score. Default 2.0 m.
    w_heading : float
        Scoring weight for the heading criterion.
    w_clearance : float
        Scoring weight for the clearance criterion.
    w_velocity : float
        Scoring weight for the velocity criterion.
    """

    def __init__(
        self,
        max_v:         float,
        max_omega:     float,
        max_dv:        float,
        max_domega:    float,
        robot_radius:  float,
        sim_time:      float = 1.0,
        dt:            float = 0.1,
        v_samples:     int   = 10,
        omega_samples: int   = 20,
        clearance_cap: float = 2.0,
        w_heading:     float = 1.0,
        w_clearance:   float = 1.0,
        w_velocity:    float = 1.0,
    ) -> None:
        self.max_v         = max_v
        self.max_omega     = max_omega
        self.max_dv        = max_dv
        self.max_domega    = max_domega
        self.robot_radius  = robot_radius
        self.sim_time      = sim_time
        self.dt            = dt
        self.v_samples     = v_samples
        self.omega_samples = omega_samples
        self.clearance_cap = clearance_cap
        self.w_heading     = w_heading
        self.w_clearance   = w_clearance
        self.w_velocity    = w_velocity

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _dynamic_window(
        self, v_cur: float, omega_cur: float
    ) -> Tuple[float, float, float, float]:
        """
        Compute the dynamic window: the (v, ω) pairs physically reachable
        from the current velocity in one time step.

        WHY we need this
        -----------------
        Without acceleration limits, DWA would consider any (v, ω) up to
        max_v / max_omega, including those that require instantaneous speed
        changes. Those commands would be clamped by robot.update() anyway,
        but they'd produce unrealistic trajectory predictions. The dynamic
        window ensures every simulated trajectory is physically achievable.

        The window is the intersection of two constraints:
          1. Acceleration: how much speed can change in one dt?
          2. Absolute limits: never exceed max_v or max_omega regardless.

        Parameters
        ----------
        v_cur : float
            Current linear velocity (m/s).
        omega_cur : float
            Current angular velocity (rad/s).

        Returns
        -------
        (v_min, v_max, omega_min, omega_max) : Tuple[float, float, float, float]
        """
        # Acceleration bounds: current speed ± (max_acceleration × dt)
        v_min = v_cur   - self.max_dv     * self.dt
        v_max = v_cur   + self.max_dv     * self.dt
        w_min = omega_cur - self.max_domega * self.dt
        w_max = omega_cur + self.max_domega * self.dt

        # Intersect with absolute velocity limits
        v_min = max(v_min, -self.max_v)
        v_max = min(v_max,  self.max_v)
        w_min = max(w_min, -self.max_omega)
        w_max = min(w_max,  self.max_omega)

        return (v_min, v_max, w_min, w_max)

    def _simulate_trajectory(
        self, x: float, y: float, theta: float, v: float, omega: float
    ) -> np.ndarray:
        """
        Forward-simulate (v, ω) held constant for `sim_time` seconds.

        Uses the same Euler integration as DifferentialDriveRobot.update():
          x_new     = x     + v * cos(θ) * dt
          y_new     = y     + v * sin(θ) * dt
          theta_new = theta + ω * dt

        WHY hold (v, ω) constant during simulation?
        ---------------------------------------------
        DWA evaluates each (v, ω) pair as if the robot will maintain that
        velocity for the entire horizon. This is a deliberate simplification:
        real trajectories will deviate as new commands are issued, but for
        short horizons (1 s) the prediction is accurate enough to be useful
        for collision checking and scoring.

        Parameters
        ----------
        x, y, theta : float
            Starting pose.
        v : float
            Linear velocity to simulate (m/s).
        omega : float
            Angular velocity to simulate (rad/s).

        Returns
        -------
        np.ndarray, shape (n_steps, 3)
            Each row is (x, y, theta) at one integration step.
            n_steps = floor(sim_time / dt).
        """
        # Use round() to avoid float precision issues like int(0.3/0.1) = 2.
        # round(0.3/0.1) = round(2.9999...) = 3 as expected.
        n_steps = max(1, round(self.sim_time / self.dt))
        traj = np.empty((n_steps, 3), dtype=float)

        for i in range(n_steps):
            x     = x     + v     * math.cos(theta) * self.dt
            y     = y     + v     * math.sin(theta) * self.dt
            theta = theta + omega * self.dt
            traj[i] = (x, y, theta)

        return traj

    def _is_collision(
        self, trajectory: np.ndarray, obstacles: np.ndarray
    ) -> bool:
        """
        Return True if any trajectory pose comes within `robot_radius` of
        any obstacle point.

        Implementation uses numpy broadcasting for speed:
          trajectory poses: shape (n_steps, 2)  [x, y only — column 0 and 1]
          obstacles:        shape (n_obs,  2)

        Broadcasting computes all n_steps × n_obs distances simultaneously
        without a Python loop. For default settings (10 steps, 36 obstacles):
        → one call to np.sqrt on a 10×36 matrix — fast even inside a loop
          over 200 candidates.

        Parameters
        ----------
        trajectory : np.ndarray, shape (n_steps, 3)
            Simulated poses; columns 0,1 are (x, y).
        obstacles : np.ndarray, shape (n_obs, 2)
            Real obstacle hit-points in continuous coordinates (metres).
            Caller must have already filtered out max_range phantom points.

        Returns
        -------
        bool
            True = unsafe (discard this trajectory).
        """
        if len(obstacles) == 0:
            # No obstacles detected — trajectory is safe by definition
            return False

        # Expand dims so subtraction broadcasts:
        #   poses:     (n_steps, 1, 2)  −  obstacles: (1, n_obs, 2)
        #   result:    (n_steps, n_obs, 2)  → norms: (n_steps, n_obs)
        poses = trajectory[:, :2]                          # (n_steps, 2)
        diffs = poses[:, np.newaxis, :] - obstacles[np.newaxis, :, :]
        dists = np.sqrt((diffs ** 2).sum(axis=2))          # (n_steps, n_obs)

        return bool(dists.min() < self.robot_radius)

    def _score_heading(
        self, trajectory: np.ndarray, goal_x: float, goal_y: float
    ) -> float:
        """
        Score how well the final pose of the trajectory faces the goal.

        Algorithm
        ---------
        1. Take the final (x, y, θ) from the simulated trajectory.
        2. Compute the angle FROM that position TO the goal:
               α = atan2(goal_y − y_final, goal_x − x_final)
        3. Compute the angular error between α and the robot's heading θ_final,
           normalised to [0, π]:
               err = |atan2(sin(α − θ), cos(α − θ))|
        4. Score = (π − err) / π → [0, 1]
               1.0 means the robot points exactly at the goal.
               0.0 means the robot points directly away from the goal.

        WHY use atan2(sin, cos) for error?
        ------------------------------------
        Simple subtraction (α − θ) can give values outside [−π, π].
        The atan2(sin, cos) trick correctly handles angle wrapping, so
        the heading error is always in [0, π].

        Parameters
        ----------
        trajectory : np.ndarray, shape (n_steps, 3)
        goal_x, goal_y : float
            Target waypoint in continuous coordinates.

        Returns
        -------
        float
            Score in [0, 1].
        """
        x_final, y_final, theta_final = trajectory[-1]

        # Direction from final position to goal
        goal_angle = math.atan2(goal_y - y_final, goal_x - x_final)

        # Angular error, normalised to [0, π]
        err = abs(math.atan2(
            math.sin(goal_angle - theta_final),
            math.cos(goal_angle - theta_final)
        ))

        return (math.pi - err) / math.pi

    def _score_clearance(
        self, trajectory: np.ndarray, obstacles: np.ndarray
    ) -> float:
        """
        Score how far the trajectory stays from the nearest obstacle.

        Algorithm
        ---------
        1. Find the minimum Euclidean distance from any trajectory pose to
           any obstacle point.
        2. If there are no obstacles, return 1.0 (maximum safety).
        3. Cap the distance at `clearance_cap` (distances beyond this are
           all equally safe — no need to prefer a trajectory 10 m from walls
           over one 3 m from walls if both are far enough).
        4. Score = min(min_dist, clearance_cap) / clearance_cap → [0, 1]
               1.0 = min_dist ≥ clearance_cap (well clear)
               0.0 = trajectory is touching an obstacle

        NOTE: Trajectories that actually collide (dist < robot_radius) are
        already discarded by _is_collision() before scoring — this method
        only sees safe trajectories. But it still compares gradations of
        safety among those survivors.

        Parameters
        ----------
        trajectory : np.ndarray, shape (n_steps, 3)
        obstacles : np.ndarray, shape (n_obs, 2)

        Returns
        -------
        float
            Score in [0, 1].
        """
        if len(obstacles) == 0:
            return 1.0

        # Reuse the same broadcasting pattern as _is_collision
        poses = trajectory[:, :2]
        diffs = poses[:, np.newaxis, :] - obstacles[np.newaxis, :, :]
        min_dist = float(np.sqrt((diffs ** 2).sum(axis=2)).min())

        return min(min_dist, self.clearance_cap) / self.clearance_cap

    def _score_velocity(self, v: float) -> float:
        """
        Score how fast (and how forward) the trajectory moves.

        Score = v / max_v → [−1, 1] in principle, but clipped to [0, 1]
        because negative v (backing up) is penalised (score 0), not rewarded.

        WHY not allow negative rewards?
        ---------------------------------
        Allowing negative velocity scores would penalise backing-up trajectories
        relative to stopping (v=0, score=0). In practice we just want to reward
        forward progress, not actively punish backing up — in a narrow corridor
        the robot might need to reverse briefly to escape.
        Setting negative v → score 0 (same as stopped) is a neutral treatment
        that doesn't block reverse but doesn't reward it either.

        Parameters
        ----------
        v : float
            Linear velocity of this candidate (m/s).

        Returns
        -------
        float
            Score in [0, 1].
        """
        return max(0.0, v / self.max_v)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_command(
        self,
        x:         float,
        y:         float,
        theta:     float,
        v_cur:     float,
        omega_cur: float,
        goal_x:    float,
        goal_y:    float,
        obstacles: np.ndarray,
    ) -> Tuple[float, float, bool]:
        """
        Select the best (v, ω) command for the current situation.

        This is the only method the caller needs to call. Everything else
        in this class is a building block for this method.

        Caller checklist (BEFORE calling this method):
          1. lidar.scan(x, y, theta)           → ranges
          2. lidar.get_endpoints(x, y, theta, ranges) → endpoints
          3. obstacles = endpoints[ranges < lidar.max_range]  ← filter phantoms!
          4. v, ω, ok = dwa.compute_command(..., obstacles=obstacles)
          5. if ok: robot.update(v, ω, dt)
             else:  handle failure (stop / re-plan / spin)

        Parameters
        ----------
        x, y, theta : float
            Current robot pose in continuous coordinates.
        v_cur : float
            Current linear velocity (m/s).
        omega_cur : float
            Current angular velocity (rad/s).
        goal_x, goal_y : float
            Target waypoint in continuous coordinates.
        obstacles : np.ndarray, shape (N, 2)
            Real obstacle hit-points — phantom max_range points must be
            filtered out by the caller BEFORE passing here.

        Returns
        -------
        (v, ω, success) : Tuple[float, float, bool]
            v, ω — the velocity command to execute.
            success — False if all sampled trajectories collided; in that
                      case v=0.0 and ω=0.0. The caller should not blindly
                      execute the command without checking success.
        """
        # Step 1: Compute the dynamic window
        v_min, v_max, w_min, w_max = self._dynamic_window(v_cur, omega_cur)

        # Step 2: Build a uniform grid of (v, ω) candidates inside the window
        v_candidates  = np.linspace(v_min, v_max, self.v_samples)
        w_candidates  = np.linspace(w_min, w_max, self.omega_samples)

        best_score = -math.inf
        best_v     = 0.0
        best_omega = 0.0
        found_safe = False

        # Step 3–5: Simulate, filter, score
        for v in v_candidates:
            for omega in w_candidates:
                # 3. Simulate the trajectory
                traj = self._simulate_trajectory(x, y, theta, float(v), float(omega))

                # 4. Discard if any step collides with an obstacle
                if self._is_collision(traj, obstacles):
                    continue

                # 5. Score the surviving trajectory
                s_heading   = self._score_heading(traj, goal_x, goal_y)
                s_clearance = self._score_clearance(traj, obstacles)
                s_velocity  = self._score_velocity(float(v))

                score = (
                    self.w_heading   * s_heading   +
                    self.w_clearance * s_clearance +
                    self.w_velocity  * s_velocity
                )

                if score > best_score:
                    best_score = score
                    best_v     = float(v)
                    best_omega = float(omega)
                    found_safe = True

        # If no safe trajectory was found, stop and report failure
        if not found_safe:
            return (0.0, 0.0, False)

        return (best_v, best_omega, True)

    def __repr__(self) -> str:
        return (
            f"DWAController("
            f"max_v={self.max_v}, max_ω={self.max_omega:.2f}, "
            f"robot_radius={self.robot_radius}, "
            f"sim_time={self.sim_time}s, dt={self.dt}s, "
            f"candidates={self.v_samples}×{self.omega_samples})"
        )
