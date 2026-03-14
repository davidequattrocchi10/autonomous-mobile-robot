"""
DWA Local Planner Visual Demo  (A* + DWA hybrid)
==================================================

Global planning + local planning working together:

  A* (global)  — plans a collision-free path of grid cells from start to goal
                 ONCE, before the simulation starts.
  DWA (local)  — runs every control step, navigating toward the NEXT A* waypoint
                 rather than the distant final goal.

WHY chain the two planners?
-----------------------------
DWA is a local planner: it only sees obstacles within LIDAR range and optimises
over a short time horizon (~1 s). If the goal is far away and blocked by walls,
DWA's scoring function will keep trying to push the robot toward the goal
through the wall → all trajectories collide → robot stops.

A* solves this by providing intermediate waypoints that are always reachable
in the local neighbourhood. DWA only needs to navigate to the next waypoint
(~0.5 m away), which is almost always unobstructed from the robot's current
position.

Control loop each step:
  1. LIDAR scans the environment
  2. find_lookahead_target() finds the furthest A* waypoint within lookahead_distance
  3. DWA selects best (v, ω) to reach that lookahead point (not the immediate next cell)
  4. Robot executes the command
  5. If within waypoint_threshold of current waypoint → advance waypoint_index

Run from project root:
    python examples/dwa_test.py
"""

import sys
sys.path.append('.')

import math
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from src.environment.grid_world import GridWorld
from src.planning.graph_search import AStar
from src.robot.robot import DifferentialDriveRobot
from src.sensors.lidar import LidarSensor
from src.control.dwa import DWAController
from src.utils.conversions import grid_to_continuous, DEFAULT_CELL_SIZE


# ──────────────────────────────────────────────────────────────────────────────
# Environment
# ──────────────────────────────────────────────────────────────────────────────

def build_env() -> GridWorld:
    env = GridWorld(width=14, height=14)
    env.add_obstacle_rect((1, 1), (6, 3))    
    env.add_obstacle_rect((2, 10), (6, 11))  
    env.add_obstacle_rect((8, 5), (10, 8))   
    env.add_obstacle(1,6)
    
    return env


# ──────────────────────────────────────────────────────────────────────────────
# Drawing helpers
# ──────────────────────────────────────────────────────────────────────────────

CELL = DEFAULT_CELL_SIZE   # 0.5 m per cell

def c2p(x: float, y: float):
    """Continuous metres → matplotlib pixel coords (imshow convention)."""
    return x / CELL - 0.5, y / CELL - 0.5


def draw_robot(ax, x, y, theta, color='limegreen'):
    px, py = c2p(x, y)
    ax.plot(px, py, 'o', color=color, markersize=10,
            markeredgecolor='darkgreen', markeredgewidth=2, zorder=10)
    dx = math.cos(theta) * 0.6
    dy = math.sin(theta) * 0.6
    ax.annotate("", xy=(px + dx, py + dy), xytext=(px, py),
                arrowprops=dict(arrowstyle='->', color='darkgreen', lw=2), zorder=11)


def draw_lidar(ax, x, y, theta, lidar, ranges, alpha=0.35):
    endpoints = lidar.get_endpoints(x, y, theta, ranges)
    rx, ry = c2p(x, y)
    for i, r in enumerate(ranges):
        ex, ey = c2p(endpoints[i, 0], endpoints[i, 1])
        hit = r < lidar.max_range
        ax.plot([rx, ex], [ry, ey],
                color='red' if hit else 'dodgerblue',
                linewidth=0.5, alpha=alpha)


def draw_trajectories(ax, dwa, x, y, theta, v_cur, omega_cur, obstacles):
    """Draw all candidate trajectories (cyan=safe, pink=collision)."""
    v_min, v_max, w_min, w_max = dwa._dynamic_window(v_cur, omega_cur)
    vs = np.linspace(v_min, v_max, dwa.v_samples)
    ws = np.linspace(w_min, w_max, dwa.omega_samples)

    for v in vs:
        for omega in ws:
            traj = dwa._simulate_trajectory(x, y, theta, float(v), float(omega))
            collision = dwa._is_collision(traj, obstacles)
            xs = [c2p(p[0], p[1])[0] for p in traj]
            ys = [c2p(p[0], p[1])[1] for p in traj]
            color = 'lightcoral' if collision else 'cyan'
            ax.plot(xs, ys, color=color, linewidth=0.4, alpha=0.5)


def draw_path(ax, trajectory_history):
    if len(trajectory_history) < 2:
        return
    xs = [c2p(p[0], p[1])[0] for p in trajectory_history]
    ys = [c2p(p[0], p[1])[1] for p in trajectory_history]
    ax.plot(xs, ys, color='darkorange', linewidth=2, alpha=0.8, zorder=8)


def draw_astar_path(ax, waypoints):
    """
    Draw the A* global path as a dashed purple line with dots at each waypoint.

    Drawn BEFORE the orange robot trajectory (lower zorder) so the actual
    path taken is always visible on top of the planned path.
    """
    if len(waypoints) < 2:
        return
    xs = [c2p(wp[0], wp[1])[0] for wp in waypoints]
    ys = [c2p(wp[0], wp[1])[1] for wp in waypoints]
    # Dashed line connecting all waypoints
    ax.plot(xs, ys, '--', color='mediumpurple', linewidth=1.5, alpha=0.7, zorder=6)
    # Small dot at each waypoint so individual cells are visible
    ax.plot(xs, ys, '.', color='mediumpurple', markersize=4, alpha=0.8, zorder=7)


# ──────────────────────────────────────────────────────────────────────────────
# Lookahead targeting
# ──────────────────────────────────────────────────────────────────────────────

def find_lookahead_target(
    robot_x: float,
    robot_y: float,
    waypoints: List[Tuple[float, float]],
    current_index: int,
    lookahead_distance: float,
) -> Tuple[float, float, int]:
    """
    Find the furthest A* waypoint that is still within lookahead_distance
    of the robot's current position.

    WHY lookahead instead of the immediate next waypoint?
    -------------------------------------------------------
    When DWA aims at the cell directly ahead (~0.5 m away), the heading
    score gradient is very narrow: the robot must be almost perfectly aligned
    with a single cell before DWA "sees" it. At corners, the robot is still
    at speed when it reaches the corner cell and overshoots into the wall gap
    before DWA can react.

    Aiming 1–2 m ahead gives DWA a target it can "see" earlier. It begins
    curving toward the corner long before reaching it → smoother arcs,
    less corner-clipping.

    Algorithm
    ---------
    1. Near-end guard: within the last 3 waypoints, return the final
       waypoint directly so the robot homes in precisely on the goal.
    2. Walk forward from current_index through all remaining waypoints.
       Record the index of every waypoint whose distance from the robot
       is ≤ lookahead_distance. No early break: the A* path may curve so
       a later waypoint can re-enter the lookahead circle.
    3. Fallback: if even waypoints[current_index] is beyond
       lookahead_distance (pathological case), return it unchanged —
       same behaviour as the previous one-step targeting.

    Parameters
    ----------
    robot_x, robot_y : float
        Current robot position in continuous coordinates (metres).
    waypoints : list of (x, y) tuples
        Full A* path converted to continuous coordinates.
    current_index : int
        The waypoint the robot is currently advancing toward.
    lookahead_distance : float
        How far ahead (metres) to search for a target waypoint.

    Returns
    -------
    (target_x, target_y, furthest_index) : Tuple[float, float, int]
        The lookahead target coordinates and its index in `waypoints`.
        The caller can optionally use furthest_index to skip ahead in
        waypoint_index, but the advancement logic is independent.
    """
    n = len(waypoints)

    # Near-end guard: home in on the final waypoint precisely.
    # Without this, lookahead could jump PAST the goal on the last few cells.
    if current_index >= n - 3:
        return (*waypoints[-1], n - 1)

    # Walk forward, recording the furthest waypoint within lookahead_distance.
    # Start with current_index as the fallback so we always return something.
    furthest_index = current_index

    for i in range(current_index, n):
        wx, wy = waypoints[i]
        dist = math.sqrt((wx - robot_x) ** 2 + (wy - robot_y) ** 2)
        if dist <= lookahead_distance:
            furthest_index = i   # keep pushing the lookahead forward

    return (*waypoints[furthest_index], furthest_index)


# ──────────────────────────────────────────────────────────────────────────────
# Main simulation loop
# ──────────────────────────────────────────────────────────────────────────────

def main():
    env = build_env()

    # ── Robot ──────────────────────────────────────────────────────────
    start_row, start_col = 11, 1
    goal_row,  goal_col  = 1,  12
    goal_x, goal_y = grid_to_continuous(goal_row, goal_col, CELL)

    robot = DifferentialDriveRobot.from_grid_cell(
        start_row, start_col, theta=0.0,
        cell_size=CELL, max_v=0.8, max_omega=math.pi,
    )

    # ── A* global path ─────────────────────────────────────────────────
    # Plan the full path ONCE before the loop. DWA will chase the waypoints
    # one by one instead of aiming at the distant final goal directly.
    astar = AStar(env, heuristic='manhattan')
    grid_path = astar.search((start_row, start_col), (goal_row, goal_col), 0.6)

    if grid_path is None:
        print("ERROR: A* found no path from start to goal. Check the environment.")
        return

    # Convert each (row, col) cell to the (x, y) centre in continuous space.
    # This is the list DWA will work through as intermediate targets.
    waypoints = [grid_to_continuous(r, c, CELL) for (r, c) in grid_path]

    # Start at index 1: the robot is already at waypoints[0] (the start cell),
    # so we skip it and aim for the first real intermediate target immediately.
    waypoint_index = 1
    waypoint_threshold  = 0.6   # metres — how close = "waypoint reached"
    lookahead_distance  = 1.0   # metres — how far ahead to aim along the A* path

    print(f"Start:     {grid_to_continuous(start_row, start_col, CELL)}")
    print(f"Goal:      ({goal_x:.2f}, {goal_y:.2f})")
    print(f"A* path:   {len(grid_path)} cells")

    # ── LIDAR ──────────────────────────────────────────────────────────
    lidar = LidarSensor(
        environment=env,
        n_beams=36, fov=2 * math.pi, max_range=3.5,
        step_size=0.05, noise_std=0.0, cell_size=CELL,
    )

    # ── DWA ────────────────────────────────────────────────────────────
    dwa = DWAController(
        max_v=robot.max_v,
        max_omega=robot.max_omega,
        max_dv=0.6,
        max_domega=math.pi * 1.5,
        robot_radius=0.2,
        sim_time=0.8,
        dt=0.1,
        v_samples=8,
        omega_samples=16,
        clearance_cap=1.5,
        # More balanced
        w_heading=0.7,   # robot steers toward waypoint but not rigidly
        w_clearance=0.2,
        w_velocity=0.1,  # speed is a small bonus, not a strong drive
    )
    print(dwa)

    dt = 0.2         # control loop time step
    max_steps = 600
    goal_threshold = 0.6   # metres — close enough to declare success

    v_cur = 0.0
    omega_cur = 0.0
    history = [(robot.x, robot.y)]

    # ── Visualisation setup ────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle("A* + DWA Hybrid Planner Demo", fontsize=14, fontweight='bold')

    ax_full  = axes[0]   # left:  full-run overview (drawn after the loop)
    ax_close = axes[1]   # right: last-step trajectory fan

    # ── Simulation ─────────────────────────────────────────────────────
    last_scan_data = None   # store for right panel

    for step in range(max_steps):
        x, y, theta = robot.get_pose()

        # ── Final goal check ───────────────────────────────────────────
        dist_to_goal = math.sqrt((x - goal_x)**2 + (y - goal_y)**2)
        if dist_to_goal < goal_threshold:
            print(f"✓ Goal reached in {step} steps!")
            break

        # ── Waypoint for advancement check (unchanged logic) ───────────
        wp_x, wp_y = waypoints[waypoint_index]

        # ── Lookahead target for DWA ────────────────────────────────────
        # Instead of aiming at the immediate next cell (0.5 m away),
        # look up to lookahead_distance ahead along the A* path so the
        # robot starts turning corners earlier and clips less.
        target_x, target_y, lookahead_idx = find_lookahead_target(
            x, y, waypoints, waypoint_index, lookahead_distance
        )

        # 1. Sense
        ranges    = lidar.scan(x, y, theta)
        endpoints = lidar.get_endpoints(x, y, theta, ranges)
        obstacles = endpoints[ranges < lidar.max_range]   # filter phantoms

        # 2. Plan (DWA targets the lookahead point, not the immediate waypoint)
        v_cmd, omega_cmd, ok = dwa.compute_command(
            x, y, theta, v_cur, omega_cur,
            target_x, target_y, obstacles,
        )

        if not ok:
            print(f"Step {step}: DWA found no safe trajectory! Stopping.")
            v_cmd, omega_cmd = 0.0, 0.0

        # 3. Act
        robot.update(v_cmd, omega_cmd, dt)
        v_cur, omega_cur = v_cmd, omega_cmd
        history.append((robot.x, robot.y))

        # ── Advance waypoint if we're close enough ─────────────────────
        # Advancement still uses the distance to waypoints[waypoint_index]
        # (the immediate next cell), not the lookahead target. This keeps
        # waypoint_index progressing steadily along the path.
        dist_to_wp = math.sqrt((robot.x - wp_x)**2 + (robot.y - wp_y)**2)
        if dist_to_wp < waypoint_threshold and waypoint_index < len(waypoints) - 1:
            waypoint_index += 1

        # Store for the right panel close-up (include lookahead target)
        last_scan_data = (x, y, theta, ranges, endpoints, obstacles,
                          v_cmd, omega_cmd, target_x, target_y)

        if step % 10 == 0:
            print(f"Step {step:3d}: pos=({x:.2f},{y:.2f})  "
                  f"wp={waypoint_index}/{len(waypoints)-1}  "
                  f"la={lookahead_idx}  "
                  f"v={v_cmd:.2f}  ω={omega_cmd:.2f}  dist_goal={dist_to_goal:.2f}")
    else:
        print("Max steps reached without reaching goal.")

    # ── Left panel: full-run overview ──────────────────────────────────
    env.render(ax=ax_full, show_legend=False)
    ax_full.set_title(
        f"Full navigation run — A* ({len(grid_path)} cells) + DWA\n"
        f"purple dashes = A* plan   orange = actual path",
        fontsize=10,
    )

    # Draw A* path first (lower zorder) so orange trajectory is on top
    draw_astar_path(ax_full, waypoints)
    draw_path(ax_full, history)

    # Mark start and goal
    sx, sy = c2p(*grid_to_continuous(start_row, start_col, CELL))
    gx_p, gy_p = c2p(goal_x, goal_y)
    ax_full.plot(sx, sy, 'o', color='limegreen', markersize=12,
                 markeredgecolor='darkgreen', markeredgewidth=2, zorder=9)
    ax_full.plot(gx_p, gy_p, '*', color='red', markersize=18,
                 markeredgecolor='darkred', markeredgewidth=2, zorder=9)

    # Draw final robot pose
    draw_robot(ax_full, robot.x, robot.y, robot.theta)

    # ── Right panel: last-step trajectory fan ──────────────────────────
    if last_scan_data:
        lx, ly, lt, lranges, lendpoints, lobs, lv, lw, ltgt_x, ltgt_y = last_scan_data
        env.render(ax=ax_close, show_legend=False)
        ax_close.set_title(
            f"Last step — DWA trajectory fan\n"
            f"cyan circle = lookahead target   v={lv:.2f} m/s  ω={lw:.2f} rad/s",
            fontsize=10,
        )
        draw_trajectories(ax_close, dwa, lx, ly, lt, v_cur, omega_cur, lobs)
        draw_lidar(ax_close, lx, ly, lt, lidar, lranges, alpha=0.5)
        draw_robot(ax_close, lx, ly, lt)
        ax_close.plot(gx_p, gy_p, '*', color='red', markersize=18,
                      markeredgecolor='darkred', markeredgewidth=2, zorder=9)
        # Show the current waypoint index marker (purple diamond)
        cwx, cwy = c2p(*waypoints[min(waypoint_index, len(waypoints) - 1)])
        ax_close.plot(cwx, cwy, 'D', color='mediumpurple', markersize=8,
                      markeredgecolor='indigo', markeredgewidth=2, zorder=9)
        # Cyan circle: the lookahead target DWA was actually aiming at.
        # This lets us verify the lookahead is jumping ahead of the current
        # waypoint (it should be further along the path than the purple diamond).
        ltx, lty = c2p(ltgt_x, ltgt_y)
        ax_close.plot(ltx, lty, 'o', color='cyan', markersize=10,
                      markeredgecolor='deepskyblue', markeredgewidth=2, zorder=12)

    # ── Legend ─────────────────────────────────────────────────────────
    legend_elements = [
        Line2D([0], [0], marker='o', color='limegreen', markersize=8,
               markeredgecolor='darkgreen', linestyle='None', label='Robot'),
        Line2D([0], [0], marker='*', color='red', markersize=10,
               markeredgecolor='darkred', linestyle='None', label='Goal'),
        Line2D([0], [0], linestyle='--', color='mediumpurple', linewidth=1.5,
               label='A* plan'),
        Line2D([0], [0], color='darkorange', linewidth=2, label='Path taken'),
        Line2D([0], [0], color='cyan', linewidth=1.5, label='Safe traj.'),
        Line2D([0], [0], color='lightcoral', linewidth=1.5, label='Collision traj.'),
        Line2D([0], [0], color='red', linewidth=1, alpha=0.5, label='LIDAR hit'),
        Line2D([0], [0], marker='o', color='cyan', markersize=8,
               markeredgecolor='deepskyblue', linestyle='None', label='Lookahead target'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=8,
               fontsize=10, frameon=True, bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig('images/dwa_demo.png', dpi=120, bbox_inches='tight')
    print("Saved: images/dwa_demo.png")
    plt.show()


if __name__ == '__main__':
    main()
