"""
Warehouse AGV Two-Forklift Replanning Demo — Environment 2 ("Zigzag Warehouse").

Layout: two tall vertical shelves create Aisle A (cols 4-5) and Aisle B (cols 8-10),
plus two lower depots with a gap between them. The robot's A* path threads through both
"gates" (row 11 and row 16).

Forklift 1 (activated at frame 5)  enters at row 11, col 0 and stops at col 5.
DWA fails 3 times → replanning routes the robot.

Forklift 2 (activated at frame 50) enters at row 11, col 18 and stops at col 14, blocking
the robot towards the goal. DWA fails 3 times → replanning routes the robot to reach the goal.

Two independent replanning events prove the mechanism generalises to any environment.
Components: A* clearance-aware planning + DWA local control + LiDAR sensing +
general emergency replanning (obs_manager-driven, any number of obstacles).
"""

import sys
sys.path.append('.')

import math
from typing import List, Tuple
import numpy as np
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — works headless and for GIF export
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter

from src.environment.grid_world import GridWorld
from src.environment.obstacle_manager import ObstacleManager, ObstacleType
from src.planning.graph_search import AStar
from src.robot.robot import DifferentialDriveRobot
from src.sensors.lidar import LidarSensor
from src.control.dwa import DWAController
from src.utils.conversions import (
    grid_to_continuous, continuous_to_grid, DEFAULT_CELL_SIZE
)


# ──────────────────────────────────────────────────────────────────────────────
# Configuration  —  all tunable parameters in one place
# ──────────────────────────────────────────────────────────────────────────────

WAREHOUSE_CONFIG = {
    # Grid
    'grid_size': 20,
    'cell_size': 0.5,               # metres per cell

    # Robot start / goal (row, col)
    'start': (1, 1),
    'goal':  (18, 18),

    # Robot model
    'max_v': 0.8,                   # m/s
    'max_omega': math.pi,           # rad/s
    'noise_std': 0.0,               # no odometry noise for demo clarity

    # LiDAR
    'n_beams': 36,
    'fov': 2 * math.pi,             # 360°
    'lidar_max_range': 3.5,         # metres
    'lidar_noise_std': 0.0,

    # DWA
    'max_dv': 0.6,
    'max_domega': math.pi * 1.5,
    'robot_radius': 0.2,            # metres — physical collision radius
    'sim_time': 0.8,
    'dwa_dt': 0.1,
    'v_samples': 8,
    'omega_samples': 16,
    'w_heading': 0.7,
    'w_clearance': 0.2,
    'w_velocity': 0.1,

    # A* — larger robot_radius → wider clearance from walls in planned path
    'astar_robot_radius': 0.6,

    # Navigation
    'lookahead_distance': 1.0,      # metres — how far ahead DWA aims along A* path
    'waypoint_threshold': 0.6,      # metres — "close enough" to advance waypoint
    'goal_threshold': 0.6,          # metres — "close enough" to declare success
    'control_dt': 0.2,              # seconds — control loop time step
    'replan_threshold': 3,          # consecutive DWA failures before replanning

    # Dynamic obstacles — two forklifts moving in opposite directions.
    # F1 enters from the LEFT (velocity (0,+1), stops when col >= blocking_col).
    # F2 enters from the RIGHT (velocity (0,-1), stops when col <= blocking_col).
    # Each entry: start (row,col), direction, col where it stops, activation frame, speed.
    'forklifts': [
        {'start': (11,  0), 'velocity': (0,  1), 'blocking_col': 5, 'activation_step':  1, 'speed_steps': 7},
        {'start': (11, 18), 'velocity': (0, -1), 'blocking_col': 14, 'activation_step': 50, 'speed_steps': 7},
    ],
    # Shared advance cadence — both forklifts use the same speed_steps so
    # obs_manager.update_all() is called exactly once per speed tick.
    'forklift_speed_steps': 7,

    # Animation
    'max_frames': 500,
    'interval_ms': 80,              # ≈ 12 fps

    # Output paths
    'gif_path': 'images/warehouse_simulation_advanced_env_two.gif',
    'png_path': 'images/warehouse_simulation_advanced_env_two_final.png',
    'gif_fps': 10,
    'gif_dpi': 100,
}

# Convenience alias used by all drawing helpers (same pattern as dwa_test.py)
CELL = WAREHOUSE_CONFIG['cell_size']


# ──────────────────────────────────────────────────────────────────────────────
# Environment builder
# ──────────────────────────────────────────────────────────────────────────────

def build_warehouse_env() -> GridWorld:
    """
    Construct the 20×20 warehouse grid for Environment 2 — "Zigzag Warehouse".

    WHY this layout?
    Two tall vertical shelves (left + middle) create two parallel aisles:
      Aisle A  cols 4-5  (between left shelf and middle shelf)
      Aisle B  cols 8-10 (right of middle shelf)

    The robot's A* path naturally threads through Aisle A first (shorter from
    start), then between the two lower depots. Forklift 1 blocks Aisle A at
    row 11 (gate 1) and forklift 2 blocks the inter-depot gap at row 16
    (gate 2). Two replanning events each force a clearly different detour.

    Layout diagram (rows × cols, X = obstacle, . = free):
      rows  0-2 : mostly open — robot start (1,1) and scattered pallets
      rows  3-10: left shelf (cols 2-3) + middle shelf (cols 6-7)
                  + upper-right processing block (cols 11-15, rows 3-6)
      row  11   : ← gate 1 — forklift 1 stops here at col 5
      rows 12   : open corridor
      rows 13-17: depot A (cols 2-5) + depot B (cols 9-13)
      row  16   : ← gate 2 — forklift 2 stops here at col 7
      rows 18-19: mostly open — robot goal (18,18)
    """
    size = WAREHOUSE_CONFIG['grid_size']
    env = GridWorld(width=size, height=size)

    # ── Left shelf  (rows 3-10, cols 2-3) ──────────────────────────────
    # Tall vertical rack on the left — robot enters Aisle A (cols 4-5)
    env.add_obstacle_rect((3, 1), (9, 2))

    # ── Middle shelf  (rows 3-10, cols 6-7) ────────────────────────────
    # Parallel rack — separates Aisle A from Aisle B
    env.add_obstacle_rect((3, 6), (5, 7))

    # ── Upper-right processing block  (rows 3-6, cols 11-15) ───────────
    # Fills the upper-right zone; robot uses Aisle B (cols 8-10) to pass
    env.add_obstacle_rect((3, 11), (6, 15))

    # ── Depot A  (rows 13-17, cols 2-5) ────────────────────────────────
    # Lower-left storage depot — leaves gap (cols 6-8) toward Depot B
    env.add_obstacle_rect((13, 2), (17, 5))

    # ── Depot B  (rows 13-17, cols 9-13) ───────────────────────────────
    # Lower-right storage depot — forklift 2 closes the gap at col 7
    env.add_obstacle_rect((13, 9), (17, 13))

    # ── Scattered loading pallets  (1×1 each) ───────────────────────────
    env.add_obstacle(1,  10)
    env.add_obstacle(7,  16)
    env.add_obstacle(18,  6)

    return env


# ──────────────────────────────────────────────────────────────────────────────
# Drawing helpers
# ──────────────────────────────────────────────────────────────────────────────

def c2p(x: float, y: float) -> Tuple[float, float]:
    """
    Continuous metres → matplotlib pixel coordinates (imshow convention).

    WHY subtract 0.5?
    imshow places pixel (0, 0) at the centre of the top-left cell.
    The cell at (row, col) has its centre at continuous (col+0.5)*CELL,
    (row+0.5)*CELL. Dividing by CELL and subtracting 0.5 maps that back
    to the integer pixel index (col, row), which imshow expects.
    """
    return x / CELL - 0.5, y / CELL - 0.5


# ──────────────────────────────────────────────────────────────────────────────
# Lookahead targeting  (copied verbatim from examples/dwa_test.py)
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

    Aiming 1-2 m ahead gives DWA a target it can "see" earlier. It begins
    curving toward the corner long before reaching it → smoother arcs,
    less corner-clipping.

    Algorithm
    ---------
    1. Near-end guard: within the last 3 waypoints, return the final
       waypoint directly so the robot homes in precisely on the goal.
    2. Walk forward from current_index through all remaining waypoints.
       Record the index of every waypoint whose distance from the robot
       is <= lookahead_distance.
    3. Fallback: if even waypoints[current_index] is beyond
       lookahead_distance (pathological case), return it unchanged.
    """
    n = len(waypoints)

    # Near-end guard: home in on the final waypoint precisely.
    if current_index >= n - 3:
        return (*waypoints[-1], n - 1)

    furthest_index = current_index

    for i in range(current_index, n):
        wx, wy = waypoints[i]
        dist = math.sqrt((wx - robot_x) ** 2 + (wy - robot_y) ** 2)
        if dist <= lookahead_distance:
            furthest_index = i

    return (*waypoints[furthest_index], furthest_index)


# ──────────────────────────────────────────────────────────────────────────────
# Predictive collision check  (module-level for testability)
# ──────────────────────────────────────────────────────────────────────────────

def _check_predictive_collision(
    robot_x: float,
    robot_y: float,
    robot_theta: float,
    v_cmd: float,
    omega_cmd: float,
    control_dt: float,
    forklift_active: bool,
    forklift_cells_next: List[Tuple[int, int]],
    robot_radius: float,
    cell_size: float,
) -> bool:
    """
    Return True if the robot's NEXT position would overlap with the forklift's
    NEXT position, indicating an imminent collision.

    WHY this function is needed
    ----------------------------
    LiDAR only sees the forklift at its CURRENT grid cell. If the robot and
    forklift are moving toward the same cell in the same frame, neither DWA
    nor LiDAR detects the conflict — DWA picks a command it believes is safe,
    and the robot walks straight into the forklift.

    This check closes that one-step lookahead gap: it runs AFTER DWA selects
    a command but BEFORE the robot executes it, giving a final veto.

    Algorithm
    ----------
    1. Predict robot's next continuous position via first-order Euler integration
    2. If forklift is not active or has no predicted cells: safe, return False.
    3. For each forklift cell in forklift_cells_next:
           Convert cell (r, c) → cell-centre continuous coords
           dist = Euclidean distance from next robot pos to cell centre
           If dist < robot_radius + cell_size: COLLISION → return True
    4. All cells checked without a hit → return False.

    Parameters
    ----------
    robot_x, robot_y, robot_theta : float
        Current robot pose (metres, radians).
    v_cmd, omega_cmd : float
        Velocity commands DWA selected (omega not used — Euler integration
        for position only, heading change is small over one control_dt).
    control_dt : float
        Length of one control step in seconds.
    forklift_active : bool
        Whether the forklift is live in this simulation step.
    forklift_cells_next : list of (row, col)
        Grid cells the forklift is predicted to occupy AFTER its next move.
        Computed from current position + velocity WITHOUT calling update_all().
    robot_radius : float
        Physical robot radius in metres (matches DWA robot_radius).
    cell_size : float
        Grid cell size in metres (used to compute cell-centre coordinates).

    Returns
    -------
    bool
        True → emergency stop required; False → safe to execute command.
    """
    if not forklift_active or not forklift_cells_next:
        return False

    # Step 1: robot's predicted next position (Euler, first-order)
    next_x = robot_x + v_cmd * math.cos(robot_theta) * control_dt
    next_y = robot_y + v_cmd * math.sin(robot_theta) * control_dt

    # Step 3: distance check against every predicted forklift cell
    # Threshold: robot_radius (robot footprint) + cell_size (forklift footprint).
    # This is conservative — better to stop unnecessarily than to collide.
    collision_threshold = robot_radius + cell_size

    for (r, c) in forklift_cells_next:
        # Cell centre in continuous space (same formula as grid_to_continuous)
        fx = (c + 0.5) * cell_size
        fy = (r + 0.5) * cell_size
        dist = math.sqrt((next_x - fx) ** 2 + (next_y - fy) ** 2)
        if dist < collision_threshold:
            return True   # imminent collision — veto the command

    return False   # all cells safe


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    cfg = WAREHOUSE_CONFIG

    print("=" * 60)
    print("Warehouse AGV Navigation Demo")
    print("Mission: Deliver order from charging dock to dispatch")
    print("=" * 60)

    # ── Environment ────────────────────────────────────────────────────
    env = build_warehouse_env()

    start_row, start_col = cfg['start']
    goal_row,  goal_col  = cfg['goal']
    goal_x, goal_y = grid_to_continuous(goal_row, goal_col, CELL)

    # ── Robot ──────────────────────────────────────────────────────────
    # Initial heading: math.pi/4 (pointing toward bottom-right goal)
    robot = DifferentialDriveRobot.from_grid_cell(
        start_row, start_col, theta=math.pi / 4,
        cell_size=CELL,
        max_v=cfg['max_v'],
        max_omega=cfg['max_omega'],
    )

    # ── A* global path ─────────────────────────────────────────────────
    # clearance-aware: robot_radius=0.6 → A* avoids cells too close to walls
    astar = AStar(env, heuristic='manhattan')
    grid_path = astar.search(
        (start_row, start_col), (goal_row, goal_col),
        robot_radius=cfg['astar_robot_radius'],
        cell_size=CELL,
    )

    if grid_path is None:
        print("ERROR: A* found no path from start to goal.")
        print("  Check that start and goal are reachable in build_warehouse_env().")
        return

    waypoints = [grid_to_continuous(r, c, CELL) for (r, c) in grid_path]
    waypoint_index = 1   # skip waypoints[0] — robot already starts there

    print(f"A* path: {len(grid_path)} cells  "
          f"(clearance-aware, robot_radius={cfg['astar_robot_radius']} m)")

    # ── LiDAR ──────────────────────────────────────────────────────────
    lidar = LidarSensor(
        environment=env,
        n_beams=cfg['n_beams'],
        fov=cfg['fov'],
        max_range=cfg['lidar_max_range'],
        step_size=0.05,
        noise_std=cfg['lidar_noise_std'],
        cell_size=CELL,
    )

    # ── DWA ────────────────────────────────────────────────────────────
    dwa = DWAController(
        max_v=robot.max_v,
        max_omega=robot.max_omega,
        max_dv=cfg['max_dv'],
        max_domega=cfg['max_domega'],
        robot_radius=cfg['robot_radius'],
        sim_time=cfg['sim_time'],
        dt=cfg['dwa_dt'],
        v_samples=cfg['v_samples'],
        omega_samples=cfg['omega_samples'],
        clearance_cap=1.5,
        w_heading=cfg['w_heading'],
        w_clearance=cfg['w_clearance'],
        w_velocity=cfg['w_velocity'],
    )

    # ── Dynamic obstacles — two forklifts ───────────────────────────────
    # Forklift 1 moves LEFT→RIGHT (velocity (0,+1)); Forklift 2 moves
    # RIGHT→LEFT (velocity (0,-1)). Each stops at its blocking_col.
    # They are registered in obs_manager so the general replanning block
    # handles them automatically via obs_manager.obstacles.values().
    obs_manager = ObstacleManager(dt=cfg['control_dt'], cell_size=CELL)
    forklift_ids = []
    for fk_cfg in cfg['forklifts']:
        fid = obs_manager.add_obstacle(
            start_pos=fk_cfg['start'],
            obstacle_type=ObstacleType.LINEAR,
            speed=1.0,
            size=1,
        )
        # Each forklift has its own direction: (0,+1) = rightward, (0,-1) = leftward.
        obs_manager.obstacles[fid].velocity = fk_cfg['velocity']
        forklift_ids.append(fid)

    # ── Simulation state ───────────────────────────────────────────────
    v_cur = 0.0
    omega_cur = 0.0
    path = grid_path                            # current A* grid path
    trajectory_history = [grid_to_continuous(start_row, start_col, CELL)]
    consecutive_failures = 0
    # active_forklift_ids: forklifts that have been activated so far.
    # Using a set so we can call .add() without rebinding — no nonlocal needed.
    active_forklift_ids: set = set()
    replan_flash_counter = 0                    # counts down from 5 → 0
    replan_count = 0

    # ── Figure setup ────────────────────────────────────────────────────
    fig, (ax_sim, ax_status) = plt.subplots(
        1, 2,
        figsize=(14, 8),
        gridspec_kw={'width_ratios': [3, 1]},
    )
    fig.patch.set_facecolor('#1a1a2e')   # dark navy background

    # Left panel styling
    ax_sim.set_facecolor('#16213e')
    fig.suptitle(
        "Warehouse AGV — Double-Forklift Replanning Demo (Env 2)",
        fontsize=13, fontweight='bold', color='white', y=0.98,
    )

    # Right panel — status dashboard (static frame, only text changes)
    ax_status.set_facecolor('#0f3460')
    ax_status.set_xlim(0, 1)
    ax_status.set_ylim(0, 1)
    ax_status.axis('off')
    ax_status.set_title("System Status", color='white', fontsize=11,
                         fontweight='bold', pad=8)

    # Pre-create status text object — updated in-place every frame
    status_text = ax_status.text(
        0.1, 0.85, "",
        transform=ax_status.transAxes,
        fontsize=10, color='#e0e0e0',
        verticalalignment='top',
        fontfamily='monospace',
    )

    # ── Static start / goal markers (drawn once before animation) ───────
    # These are drawn on a persistent layer; the left panel is cleared and
    # redrawn each frame, so we re-draw them inside update() instead.
    sx_p, sy_p = c2p(*grid_to_continuous(start_row, start_col, CELL))
    gx_p, gy_p = c2p(goal_x, goal_y)

    # ── update() — FuncAnimation callback ───────────────────────────────
    # All mutable state lives here via nonlocal so the closure is self-contained.

    goal_reached = False   # flag to freeze animation after success

    def update(frame: int):
        nonlocal v_cur, omega_cur, waypoint_index, path, waypoints
        nonlocal consecutive_failures, replan_flash_counter
        nonlocal goal_reached, replan_count
        # active_forklift_ids is a set — mutated via .add(), no rebinding needed,
        # so nonlocal is not required (Python resolves the outer binding at lookup)

        # ── 1. Get current robot pose ───────────────────────────────────
        x, y, theta = robot.get_pose()

        # ── 2. Goal-reached check ───────────────────────────────────────
        dist_to_goal = math.sqrt((x - goal_x) ** 2 + (y - goal_y) ** 2)
        if dist_to_goal < cfg['goal_threshold'] and not goal_reached:
            goal_reached = True
            print(f"[OK] Goal reached in {frame} steps!")

        # After goal is reached, keep re-rendering the final frame
        # (animation continues for smooth GIF end) but skip control logic.
        if goal_reached:
            _redraw(ax_sim, ax_status, env, trajectory_history, waypoints,
                    x, y, theta, None, None, active_forklift_ids, obs_manager,
                    frame, waypoint_index, len(waypoints),
                    consecutive_failures, dist_to_goal,
                    replan_flash_counter, status_text, sx_p, sy_p, gx_p, gy_p,
                    success=True, replan_count=replan_count)
            return

        # ── 3. Activate / advance forklifts ─────────────────────────────
        # Each forklift has its own activation_step and direction. Once
        # activated it moves every forklift_speed_steps frames until it
        # reaches its blocking_col, then freezes (STATIC).
        # Stop condition is direction-aware: '>=' for rightward (dc>0),
        # '<=' for leftward (dc<0), so both directions work correctly.
        for i, fid in enumerate(forklift_ids):
            fk_cfg = cfg['forklifts'][i]
            if fid not in active_forklift_ids and frame == fk_cfg['activation_step']:
                active_forklift_ids.add(fid)
                print(f"[Step {frame}] Forklift {i + 1} activated at {fk_cfg['start']}")

        # Advance only ACTIVE forklifts — call fobs.update() individually instead
        # of obs_manager.update_all(), which would move every obstacle registered
        # in obs_manager (including forklifts not yet activated). This was the
        # root cause of F2 drifting from col 20 to col 12 before its activation
        # frame, stopping at the wrong column.
        if active_forklift_ids and frame % cfg['forklift_speed_steps'] == 0:
            for i, fid in enumerate(forklift_ids):
                if fid in active_forklift_ids:
                    fobs = obs_manager.obstacles[fid]
                    if fobs.obstacle_type != ObstacleType.STATIC:
                        fobs.update(env, obs_manager)   # one step, this forklift only
                        fk_cfg = cfg['forklifts'][i]
                        dc     = fk_cfg['velocity'][1]
                        target = fk_cfg['blocking_col']
                        reached = (dc > 0 and fobs.position[1] >= target) or \
                                  (dc < 0 and fobs.position[1] <= target)
                        if reached:
                            fobs.obstacle_type = ObstacleType.STATIC
                            fobs.velocity = (0, 0)   # clear stale velocity
                            print(f"[Step {frame}] Forklift {i + 1} stopped "
                                  f"at {fobs.position}")

        # ── 4. Temporarily inject forklift into env.grid so LiDAR sees it
        # WHY: LidarSensor.scan() ray-marches against env.grid (static array).
        # Injecting the forklift cell(s) before scanning and restoring after
        # lets the forklift appear as a natural obstacle — no special-casing
        # in LiDAR or DWA needed.
        forklift_cells = []
        if active_forklift_ids:
            for fid in active_forklift_ids:
                forklift_cells.extend(
                    (r, c)
                    for (r, c) in obs_manager.obstacles[fid].get_occupied_cells()
                    if 0 <= r < env.height and 0 <= c < env.width
                )
            for r, c in forklift_cells:
                env.grid[r, c] = 1

        # ── 5. LiDAR scan ───────────────────────────────────────────────
        ranges    = lidar.scan(x, y, theta)
        endpoints = lidar.get_endpoints(x, y, theta, ranges)

        # ── 6. Restore env.grid immediately ─────────────────────────────
        # Restoring here ensures A* replan and env.render() are unaffected.
        for r, c in forklift_cells:
            env.grid[r, c] = 0

        # ── 7. Build obstacle array for DWA ─────────────────────────────
        # Filter phantom points (beams that hit nothing return max_range endpoint
        # which is not a real obstacle — including them would make DWA think
        # there are walls everywhere at max range).
        obstacles = endpoints[ranges < lidar.max_range]

        # ── 8. Lookahead target ──────────────────────────────────────────
        target_x, target_y, _ = find_lookahead_target(
            x, y, waypoints, waypoint_index, cfg['lookahead_distance']
        )

        # ── 9. DWA command ───────────────────────────────────────────────
        v_cmd, omega_cmd, ok = dwa.compute_command(
            x, y, theta, v_cur, omega_cur,
            target_x, target_y, obstacles,
        )

        # ── 10. Failure counting ─────────────────────────────────────────
        # Also detect path-blocked-by-stopped-forklift so consecutive_failures
        # accumulates even when DWA finds a local micro-detour (in a wide open
        # aisle DWA returns ok=True but the global A* route is still blocked —
        # replanning is the correct response, not local wiggling).
        path_blocked_by_forklift = False
        if active_forklift_ids:
            for fid in active_forklift_ids:
                fobs = obs_manager.obstacles[fid]
                if fobs.obstacle_type == ObstacleType.STATIC:
                    fork_cells = set(
                        (r, c) for (r, c) in fobs.get_occupied_cells()
                        if 0 <= r < env.height and 0 <= c < env.width
                    )
                    if any((r, c) in fork_cells for (r, c) in path[waypoint_index:]):
                        path_blocked_by_forklift = True
                        break

        if not ok:
            consecutive_failures += 1
            v_cmd, omega_cmd = 0.0, 0.0   # stop while blocked
        elif path_blocked_by_forklift:
            consecutive_failures += 1     # global path blocked — count as failure
        else:
            consecutive_failures = 0

        # ──   Predictive collision check ──────────────────────────────
        # Compute where the forklift will be AFTER its next move (read-only).
        # This closes the one-frame blind spot: LiDAR sees the forklift at its
        # CURRENT cell, but both robot and forklift may move into the same cell
        # in this frame. The check runs after DWA selects a command but before
        # robot.update() so we can veto the command if needed.
        if active_forklift_ids:
            forklift_cells_next = []
            for fid in active_forklift_ids:
                fobs = obs_manager.obstacles[fid]
                dr, dc = fobs.velocity
                pred = (fobs.position[0] + dr, fobs.position[1] + dc)
                if 0 <= pred[0] < env.height and 0 <= pred[1] < env.width:
                    forklift_cells_next.append(pred)
            if forklift_cells_next and _check_predictive_collision(
                x, y, theta, v_cmd, omega_cmd,
                cfg['control_dt'], True, forklift_cells_next,
                cfg['robot_radius'], cfg['cell_size'],
            ):
                v_cmd, omega_cmd = 0.0, 0.0   # emergency stop
                consecutive_failures += 1      # counts toward replan threshold

        if consecutive_failures >= cfg['replan_threshold']:
            r_cur, c_cur = continuous_to_grid(x, y, CELL)
            # Clamp to valid grid bounds (robot might be slightly outside)
            r_cur = max(0, min(env.height - 1, r_cur))
            c_cur = max(0, min(env.width  - 1, c_cur))

            # Stop the robot while replanning — even if DWA was returning ok=True
            # (failures may have accumulated via path_blocked_by_forklift, meaning
            # the robot was still advancing toward the obstacle).
            v_cmd, omega_cmd = 0.0, 0.0

            # Inject ALL active dynamic obstacles + a 1-cell safety margin into
            # env.grid so A* is forced to plan a route with adequate DWA clearance.
            #
            # WHY iterate obs_manager instead of forklift only: in a real warehouse
            # the robot can encounter any number of dynamic obstacles (forklifts,
            # pallet movers, workers). Iterating the obstacle manager generalises
            # replanning to any scenario without changing this code.
            #
            # WHY always inject (not only when forklift_stopped): DWA failures can
            # accumulate while forklift_stopped is still False (frame-modulo timing
            # window). The old condition caused A* to replan on the original clear
            # grid, returning the exact same path.
            #
            # WHY margin=1: cell_size=0.5 m, robot_radius=0.2 m. The immediately
            # adjacent cells (e.g. (12,7) and (12,9)) are physically passable but
            # leave < 0.3 m clearance when the aisle is flanked by shelves or
            # pallets — DWA fails there in practice. Blocking them forces A* to
            # find a crossing ≥ 2 cells (1.0 m) away from the obstacle.
            replan_injected = []
            margin = max(1, math.ceil(cfg['robot_radius'] / CELL))   # = 1 cell
            for obs in obs_manager.obstacles.values():
                for (fr, fc) in obs.get_occupied_cells():
                    for dr in range(-margin, margin + 1):
                        for dc in range(-margin, margin + 1):
                            nr, nc = fr + dr, fc + dc
                            if (0 <= nr < env.height and 0 <= nc < env.width
                                    and env.grid[nr, nc] == 0):
                                env.grid[nr, nc] = 1
                                replan_injected.append((nr, nc))

            new_grid_path = astar.search(
                (r_cur, c_cur), (goal_row, goal_col),
                robot_radius=cfg['astar_robot_radius'],
                cell_size=CELL,
            )

            # Restore only the cells we actually changed — never touch pre-existing walls.
            # (LiDAR and rendering handle their own injection independently.)
            for (nr, nc) in replan_injected:
                env.grid[nr, nc] = 0

            if new_grid_path is not None:
                path      = new_grid_path
                waypoints = [grid_to_continuous(r, c, CELL) for (r, c) in path]
                waypoint_index = 1
                print(f"[Step {frame}] REPLANNING — new path: {len(path)} cells")
                print(f"[Step {frame}] New path first 5 cells: {new_grid_path[:5]}")
                replan_count += 1
            else:
                print(f"[Step {frame}] WARNING: A* found no path — waiting...")

            consecutive_failures = 0
            replan_flash_counter = 5   # show "REPLANNING..." text for 5 frames

        if replan_flash_counter > 0:
            replan_flash_counter -= 1

        # ── 11. Execute command ──────────────────────────────────────────
        robot.update(v_cmd, omega_cmd, cfg['control_dt'])
        v_cur     = v_cmd
        omega_cur = omega_cmd
        trajectory_history.append((robot.x, robot.y))

        # ── 12. Advance waypoint ─────────────────────────────────────────
        wp_x, wp_y = waypoints[waypoint_index]
        dist_to_wp = math.sqrt((robot.x - wp_x) ** 2 + (robot.y - wp_y) ** 2)
        if dist_to_wp < cfg['waypoint_threshold'] and waypoint_index < len(waypoints) - 1:
            waypoint_index += 1

        # ── 13. Redraw both panels ───────────────────────────────────────
        _redraw(ax_sim, ax_status, env, trajectory_history, waypoints,
                x, y, theta, ranges, endpoints, active_forklift_ids, obs_manager,
                frame, waypoint_index, len(waypoints),
                consecutive_failures, dist_to_goal,
                replan_flash_counter, status_text, sx_p, sy_p, gx_p, gy_p,
                v_cmd=v_cmd, success=False, replan_count=replan_count)

    # ── Run animation ───────────────────────────────────────────────────
    print("\nBuilding animation...")
    anim = FuncAnimation(
        fig, update,
        frames=cfg['max_frames'],
        interval=cfg['interval_ms'],
        repeat=False,
        blit=False,   # simpler, avoids artist-tracking issues
    )

    # ── Save GIF ────────────────────────────────────────────────────────
    print("Saving GIF... (this may take 30 seconds)")
    writer = PillowWriter(fps=cfg['gif_fps'])
    anim.save(cfg['gif_path'], writer=writer, dpi=cfg['gif_dpi'])
    print(f"Saved: {cfg['gif_path']}")

    # ── Save final frame as PNG ─────────────────────────────────────────
    fig.savefig(cfg['png_path'], dpi=cfg['gif_dpi'], bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f"Saved: {cfg['png_path']}")


# ──────────────────────────────────────────────────────────────────────────────
# Panel renderer  (called from update() every frame)
# ──────────────────────────────────────────────────────────────────────────────

def _redraw(
    ax_sim, ax_status, env, trajectory_history, waypoints,
    x, y, theta, ranges, endpoints, active_forklift_ids, obs_manager,
    frame, waypoint_index, n_waypoints,
    consecutive_failures, dist_to_goal,
    replan_flash_counter, status_text,
    sx_p, sy_p, gx_p, gy_p,
    v_cmd=0.0, success=False, replan_count=0,
):
    """
    Redraw both panels from scratch each frame.

    WHY redraw rather than updating artists in-place?
    With blit=False and a moderate frame count (350), a full redraw is clean
    and correct. Maintaining 36 LiDAR line references + a growing trajectory
    line + moving obstacle patches across frames adds complexity without
    meaningful speed gain for a demo that renders to GIF offline.
    """

    # ════════════════════════════════════════════════════════════════════
    # Left panel — main simulation view
    # ════════════════════════════════════════════════════════════════════
    ax_sim.cla()
    ax_sim.set_facecolor('#16213e')

    # 1. Static grid background
    env.render(ax=ax_sim, show_legend=False)

    # 2. A* path — dashed purple line with dots at each waypoint
    if len(waypoints) >= 2:
        wxs = [c2p(wp[0], wp[1])[0] for wp in waypoints]
        wys = [c2p(wp[0], wp[1])[1] for wp in waypoints]
        ax_sim.plot(wxs, wys, '--', color='mediumpurple',
                    linewidth=1.2, alpha=0.65, zorder=5)
        ax_sim.plot(wxs, wys, '.', color='mediumpurple',
                    markersize=3, alpha=0.7, zorder=6)

    # 3. Robot trajectory history — thin orange line
    if len(trajectory_history) >= 2:
        hxs = [c2p(p[0], p[1])[0] for p in trajectory_history]
        hys = [c2p(p[0], p[1])[1] for p in trajectory_history]
        ax_sim.plot(hxs, hys, color='darkorange',
                    linewidth=1.5, alpha=0.8, zorder=7)

    # 4. LiDAR beams (skip on success-freeze frames where ranges=None)
    if ranges is not None:
        rx_p, ry_p = c2p(x, y)
        for i, r in enumerate(ranges):
            ex_p, ey_p = c2p(endpoints[i, 0], endpoints[i, 1])
            hit = r < lidar_max_range_ref
            ax_sim.plot([rx_p, ex_p], [ry_p, ey_p],
                        color='red' if hit else 'dodgerblue',
                        linewidth=0.4, alpha=0.3, zorder=4)

    # 5. Dynamic obstacles — one rectangle per active forklift
    for fid in active_forklift_ids:
        fpos = obs_manager.obstacles[fid].position   # (row, col)
        # Place rectangle: (col-0.5) to (col+1.5) in pixel x,
        # (row-0.25) to (row+0.75) in pixel y  →  2×1 cell visual footprint
        fx_p = fpos[1] - 0.5      # pixel x = col - 0.5
        fy_p = fpos[0] - 0.25     # pixel y = row - 0.25
        forklift_patch = mpatches.FancyBboxPatch(
            (fx_p, fy_p), 2.0, 1.0,
            boxstyle="round,pad=0.05",
            facecolor='#e65c00', edgecolor='#ff9a00',
            linewidth=1.5, zorder=9,
        )
        ax_sim.add_patch(forklift_patch)
        ax_sim.text(fx_p + 1.0, fy_p + 0.5, "FORK",
                    color='white', fontsize=5, ha='center', va='center',
                    fontweight='bold', zorder=10)

    # 6. Start marker — green circle (static position, redrawn each frame)
    ax_sim.plot(sx_p, sy_p, 'o', color='limegreen', markersize=10,
                markeredgecolor='darkgreen', markeredgewidth=2, zorder=11)
    ax_sim.text(sx_p, sy_p - 0.8, "START", color='limegreen',
                fontsize=6, ha='center', va='top', zorder=11)

    # 7. Goal marker — red star
    ax_sim.plot(gx_p, gy_p, '*', color='red', markersize=16,
                markeredgecolor='darkred', markeredgewidth=1.5, zorder=11)
    ax_sim.text(gx_p, gy_p + 0.8, "DISPATCH", color='tomato',
                fontsize=6, ha='center', va='bottom', zorder=11)

    # 8. Robot body — green circle with heading arrow
    px, py = c2p(x, y)
    ax_sim.plot(px, py, 'o', color='limegreen', markersize=9,
                markeredgecolor='darkgreen', markeredgewidth=2, zorder=12)
    dx = math.cos(theta) * 0.7
    dy = math.sin(theta) * 0.7
    ax_sim.annotate("", xy=(px + dx, py + dy), xytext=(px, py),
                    arrowprops=dict(arrowstyle='->', color='darkgreen', lw=2),
                    zorder=13)

    # 9. Info text — bottom-left overlay
    info = (f"Step: {frame}  |  v: {v_cmd:.2f} m/s  |  dist: {dist_to_goal:.1f} m")
    ax_sim.text(0.01, 0.01, info, transform=ax_sim.transAxes,
                fontsize=8, color='#cccccc', va='bottom',
                bbox=dict(facecolor='#0d0d1a', alpha=0.6, pad=2))

    # 10. Replan flash text
    if replan_flash_counter > 0:
        ax_sim.text(0.5, 0.5, "REPLANNING...", transform=ax_sim.transAxes,
                    fontsize=14, color='yellow', fontweight='bold',
                    ha='center', va='center', alpha=0.9,
                    bbox=dict(facecolor='black', alpha=0.5, pad=4), zorder=20)

    # 11. Success overlay
    if success:
        ax_sim.text(0.5, 0.5, "[OK] DELIVERY COMPLETE", transform=ax_sim.transAxes,
                    fontsize=16, color='limegreen', fontweight='bold',
                    ha='center', va='center', alpha=0.95,
                    bbox=dict(facecolor='black', alpha=0.6, pad=6), zorder=20)

    # ════════════════════════════════════════════════════════════════════
    # Right panel — status dashboard (text update only, no cla())
    # ════════════════════════════════════════════════════════════════════
    n_active = len(active_forklift_ids)
    forklift_str = f"{n_active} ACTIVE" if n_active > 0 else "waiting..."

    status_lines = (
        f"{'─' * 22}\n"
        f"Waypoint:  {waypoint_index:3d} / {n_waypoints - 1}\n"
        f"\n"
        f"DWA fails: {consecutive_failures}\n"
        f"\n"
        f"Dist goal: {dist_to_goal:5.1f} m\n"
        f"\n"
        f"Forklifts: {forklift_str}\n"
        f"\n"
        f"Replans:   {replan_count}\n"
        f"{'─' * 22}\n"
    )
    if success:
        status_lines += "\n  DELIVERY\n  COMPLETE ✓"

    status_text.set_text(status_lines)


# ──────────────────────────────────────────────────────────────────────────────
# Module-level reference needed inside _redraw
# (avoids passing lidar object through the long arg list)
# ──────────────────────────────────────────────────────────────────────────────
lidar_max_range_ref = WAREHOUSE_CONFIG['lidar_max_range']


if __name__ == '__main__':
    main()
