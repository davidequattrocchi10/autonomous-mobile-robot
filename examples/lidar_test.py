"""
LIDAR Sensor Visual Demo
========================

Shows a top-down GridWorld with the robot (green dot) and LIDAR rays drawn
as lines from the robot to each hit point (red dots = obstacle hits,
blue dots = max-range readings).

Run from project root:
    python examples/test_lidar.py
"""

import sys
sys.path.append('.')

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from src.environment.grid_world import GridWorld
from src.sensors.lidar import LidarSensor
from src.utils.conversions import grid_to_continuous, DEFAULT_CELL_SIZE


# ──────────────────────────────────────────────────────────────────────────────
# Build the environment
# ──────────────────────────────────────────────────────────────────────────────

def build_environment() -> GridWorld:
    """
    A 12×12 warehouse-style grid with several obstacle blocks.
    Represents shelving units and a corner wall.
    """
    env = GridWorld(width=12, height=12)

    # Left shelf (rows 2-5, col 2-3)
    env.add_obstacle_rect((2, 2), (5, 3))

    # Right shelf (rows 2-5, col 8-9)
    env.add_obstacle_rect((2, 8), (5, 9))

    # Center obstacle (rows 7-9, col 4-7)
    env.add_obstacle_rect((7, 4), (9, 7))

    # Top-left corner cluster
    env.add_obstacle_rect((0, 0), (1, 1))

    return env


# ──────────────────────────────────────────────────────────────────────────────
# Drawing helpers
# ──────────────────────────────────────────────────────────────────────────────

def draw_lidar_rays(ax, x: float, y: float, endpoints: np.ndarray,
                    ranges: np.ndarray, max_range: float,
                    cell_size: float) -> None:
    """
    Draw LIDAR rays on the matplotlib axis.

    The GridWorld renders with imshow(), which places cell (row, col) at
    pixel position (col, row). Continuous coordinates need to be divided by
    cell_size to map to pixel positions.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    x, y : float
        Robot continuous position.
    endpoints : np.ndarray, shape (N, 2)
        Hit-point continuous coordinates from lidar.get_endpoints().
    ranges : np.ndarray, shape (N,)
        Range readings from lidar.scan().
    max_range : float
        Sensor max range — used to distinguish hits from free-space returns.
    cell_size : float
        Converts continuous metres to pixel units.
    """
    # Convert robot position to pixel units
    rx = x / cell_size - 0.5   # imshow uses cell centres at integer pixels
    ry = y / cell_size - 0.5

    for i in range(len(ranges)):
        ex = endpoints[i, 0] / cell_size - 0.5
        ey = endpoints[i, 1] / cell_size - 0.5

        is_hit = ranges[i] < max_range

        # Ray line: thin and semi-transparent so the grid stays readable
        color = 'red' if is_hit else 'dodgerblue'
        ax.plot([rx, ex], [ry, ey], color=color, linewidth=0.6, alpha=0.5)

        # Endpoint dot: larger for hits, smaller for max-range returns
        if is_hit:
            ax.plot(ex, ey, 'o', color='red', markersize=3, alpha=0.8)
        else:
            ax.plot(ex, ey, '.', color='dodgerblue', markersize=2, alpha=0.4)

    # Robot body
    ax.plot(rx, ry, 'o', color='limegreen', markersize=10,
            markeredgecolor='darkgreen', markeredgewidth=2, zorder=10)


def draw_heading_arrow(ax, x: float, y: float, theta: float,
                       length: float, cell_size: float) -> None:
    """Draw a short arrow showing the robot's heading."""
    rx = x / cell_size - 0.5
    ry = y / cell_size - 0.5
    dx = math.cos(theta) * length
    dy = math.sin(theta) * length
    ax.annotate("", xy=(rx + dx, ry + dy), xytext=(rx, ry),
                arrowprops=dict(arrowstyle='->', color='darkgreen', lw=2))


# ──────────────────────────────────────────────────────────────────────────────
# Main demo
# ──────────────────────────────────────────────────────────────────────────────

def main():
    env = build_environment()
    cell_size = DEFAULT_CELL_SIZE

    # ── Scenario A: robot in the open corridor (row=6, col=6) ──────────────
    robot_row_a, robot_col_a = 6, 6
    theta_a = 0.0   # facing right
    x_a, y_a = grid_to_continuous(robot_row_a, robot_col_a, cell_size)

    lidar_a = LidarSensor(
        environment=env,
        n_beams=72,            # one beam every 5°
        fov=2 * math.pi,       # 360°
        max_range=2.5,
        step_size=0.05,
        noise_std=0.0,         # noise-free for clarity
        cell_size=cell_size,
    )
    # Fires all beams
    ranges_a = lidar_a.scan(x_a, y_a, theta_a)
    # Compute the point where the obstacle is
    endpoints_a = lidar_a.get_endpoints(x_a, y_a, theta_a, ranges_a)

    # ── Scenario B: robot near left shelf (row=4, col=5) ──────────────────
    robot_row_b, robot_col_b = 4, 5
    theta_b = math.pi / 2   # facing down
    x_b, y_b = grid_to_continuous(robot_row_b, robot_col_b, cell_size)

    lidar_b = LidarSensor(
        environment=env,
        n_beams=36,
        fov=2 * math.pi,
        max_range=4.0,
        step_size=0.05,
        noise_std=0.05,        # small noise — observe scatter in readings
        cell_size=cell_size,
        seed=42,
    )
    ranges_b = lidar_b.scan(x_b, y_b, theta_b)
    endpoints_b = lidar_b.get_endpoints(x_b, y_b, theta_b, ranges_b)

    # ── Scenario C: robot near left shelf (row=4, col=5) but high noise ──────────────────
    robot_row_c, robot_col_c = 4, 5
    theta_c = math.pi / 2   # facing down
    x_c, y_c = grid_to_continuous(robot_row_c, robot_col_c, cell_size)

    lidar_c = LidarSensor(
        environment=env,
        n_beams=36,
        fov=2 * math.pi,
        max_range=4.0,
        step_size=0.05,
        noise_std=0.3,        # high noise 
        cell_size=cell_size,
        seed=42,
    )
    ranges_c = lidar_c.scan(x_c, y_c, theta_c)
    endpoints_c = lidar_c.get_endpoints(x_c, y_c, theta_c, ranges_c)

    # ── Layout ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 7))
    fig.suptitle("LIDAR Sensor Simulation", fontsize=16, fontweight='bold')

    for ax, (rx, ry, theta, ranges, endpoints, lidar, title) in zip(axes, [
        (x_a, y_a, theta_a, ranges_a, endpoints_a, lidar_a,
         f"Scenario A — open corridor\n72 beams, 360°, noise-free, 2.5m range"),
        (x_b, y_b, theta_b, ranges_b, endpoints_b, lidar_b,
         f"Scenario B — near shelf\n36 beams, 360°, noise σ=0.05m, 4m range"),
        (x_c, y_c, theta_c, ranges_c, endpoints_c, lidar_c,
         f"Scenario C — near shelf\n36 beams, 360°, noise σ=0.3m, 4m range"),
    ]):
        # Render the grid (no path/start/goal markers — we draw our own)
        env.render(ax=ax, show_legend=False)
        ax.set_title(title, fontsize=11, pad=8)

        draw_lidar_rays(ax, rx, ry, endpoints, ranges, lidar.max_range, cell_size)
        draw_heading_arrow(ax, rx, ry, theta, length=1.2, cell_size=cell_size)

    # Custom legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='limegreen', markersize=8,
               markeredgecolor='darkgreen', linestyle='None', label='Robot'),
        Line2D([0], [0], color='red', linewidth=1.5, label='Ray (hit)'),
        Line2D([0], [0], marker='o', color='red', markersize=5,
               linestyle='None', label='Hit point'),
        Line2D([0], [0], color='dodgerblue', linewidth=1.5, label='Ray (free)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4,
               fontsize=11, frameon=True, bbox_to_anchor=(0.5, 0.01))

    # Print summary stats
    hits_a = np.sum(ranges_a < lidar_a.max_range)
    hits_b = np.sum(ranges_b < lidar_b.max_range)
    hits_c = np.sum(ranges_c < lidar_c.max_range)
    print(f"Scenario A: {hits_a}/{lidar_a.n_beams} beams hit obstacles  "
          f"| avg range = {ranges_a.mean():.2f}m")
    print(f"Scenario B: {hits_b}/{lidar_b.n_beams} beams hit obstacles  "
          f"| avg range = {ranges_b.mean():.2f}m")
    print(f"Scenario C: {hits_c}/{lidar_c.n_beams} beams hit obstacles  "
          f"| avg range = {ranges_c.mean():.2f}m")
    print(lidar_a)
    print(lidar_b)
    print(lidar_c)

    plt.tight_layout(rect=[0, 0.07, 1, 1])
    plt.savefig('images/lidar_demo.png', dpi=120, bbox_inches='tight')
    print("Saved: images/lidar_demo.png")
    plt.show()


if __name__ == '__main__':
    main()
