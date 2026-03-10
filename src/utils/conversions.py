"""
Coordinate Conversion Utilities
================================

The project uses two coordinate systems that must talk to each other:

  GRID (discrete)      : (row, col) integer indices into the numpy array
  CONTINUOUS (physical): (x, y, θ) float values in meters

The robot model works in continuous space (physics needs smooth values).
The planners work in grid space (BFS, A*, RRT operate on cells).
The LIDAR casts rays in continuous space but checks obstacles in grid space.

Without a central conversion, every module would implement its own version
of the formula — subtle differences (off-by-one, wrong cell_size) would
produce silent bugs that are extremely hard to trace.

COORDINATE CONVENTION
----------------------
We deliberately match the GridWorld convention, where:
  - row 0 is at the TOP of the grid
  - row increases DOWNWARD
  - col 0 is at the LEFT of the grid
  - col increases RIGHTWARD

The continuous space mirrors this:
  - y = 0 is at the TOP, y increases DOWNWARD  (matches row direction)
  - x = 0 is at the LEFT, x increases RIGHTWARD (matches col direction)

This means:
  theta = 0      → robot faces RIGHT  (+x direction)
  theta = π/2    → robot faces DOWN   (+y direction, same as +row)
  theta = π      → robot faces LEFT   (-x direction)
  theta = -π/2   → robot faces UP     (-y direction, same as -row)

WHY NOT use the standard math convention (y upward)?
  → It would require flipping y every time we interact with the grid
    or with matplotlib's imshow (which also has y downward).
    Matching the grid convention eliminates one entire class of bugs.

CELL SIZE
----------
`cell_size` is the physical side length of one grid cell, in meters.
Default: 0.5 m → a 20×20 grid represents a 10 m × 10 m warehouse.

The cell centre is at (col + 0.5) * cell_size in x,
                 and (row + 0.5) * cell_size in y.
"""

import math
from typing import Tuple


# Default cell size used throughout the project when none is specified.
DEFAULT_CELL_SIZE: float = 0.5   # meters per cell


def grid_to_continuous(
    row: int,
    col: int,
    cell_size: float = DEFAULT_CELL_SIZE,
) -> Tuple[float, float]:
    """
    Convert a grid cell (row, col) to its centre in continuous space (x, y).

    The centre of cell (row, col) is at:
      x = (col + 0.5) * cell_size
      y = (row + 0.5) * cell_size

    WHY use the cell centre (+ 0.5)?
    When a planner produces a waypoint at grid cell (3, 5), the robot should
    navigate to the physical middle of that cell, not its top-left corner.

    Parameters
    ----------
    row, col : int
        Grid indices (non-negative integers).
    cell_size : float
        Physical side length of one cell in meters.

    Returns
    -------
    (x, y) : Tuple[float, float]
        Continuous coordinates of the cell centre.

    Examples
    --------
    >>> grid_to_continuous(0, 0)          # top-left cell centre
    (0.25, 0.25)
    >>> grid_to_continuous(0, 0, cell_size=1.0)
    (0.5, 0.5)
    >>> grid_to_continuous(2, 3)
    (1.75, 1.25)
    """
    x = (col + 0.5) * cell_size
    y = (row + 0.5) * cell_size
    return (x, y)


def continuous_to_grid(
    x: float,
    y: float,
    cell_size: float = DEFAULT_CELL_SIZE,
) -> Tuple[int, int]:
    """
    Convert continuous position (x, y) to the grid cell that contains it.

    The containing cell is found by integer division:
      col = floor(x / cell_size)
      row = floor(y / cell_size)

    WHY floor (not round)?
    A position at x = 0.99 is still inside col 0 if cell_size = 1.0.
    Floor gives the cell that *contains* the point, which is what
    collision detection and planning lookups need.

    Note: this function does NOT check whether the resulting cell is
    within the grid bounds — callers must validate with env.is_in_bounds()
    or env.is_valid() before using the result.

    Parameters
    ----------
    x, y : float
        Continuous coordinates in meters.
    cell_size : float
        Physical side length of one cell in meters.

    Returns
    -------
    (row, col) : Tuple[int, int]
        Grid indices of the cell containing (x, y).

    Examples
    --------
    >>> continuous_to_grid(0.25, 0.25)    # centre of top-left cell
    (0, 0)
    >>> continuous_to_grid(0.99, 0.99)    # still inside cell (0, 0) if cell_size=0.5
    (1, 1)
    >>> continuous_to_grid(1.75, 1.25)    # centre of cell (2, 3)
    (2, 3)
    """
    col = int(math.floor(x / cell_size))
    row = int(math.floor(y / cell_size))
    return (row, col)


def angle_to_grid_direction(theta: float) -> Tuple[int, int]:
    """
    Snap a continuous heading angle to the nearest 4-connected grid direction.

    Used when a continuous-space heading needs to be expressed as a grid move.
    Returns a (Δrow, Δcol) unit vector matching the GridWorld action convention.

    Mapping (with y-downward convention):
      theta ≈  0      → facing RIGHT → (Δrow= 0, Δcol=+1)
      theta ≈  π/2    → facing DOWN  → (Δrow=+1, Δcol= 0)
      theta ≈  π/-π   → facing LEFT  → (Δrow= 0, Δcol=-1)
      theta ≈ -π/2    → facing UP    → (Δrow=-1, Δcol= 0)

    Parameters
    ----------
    theta : float
        Heading angle in radians, normalised to [-π, π].

    Returns
    -------
    (dr, dc) : Tuple[int, int]
        Unit vector in grid coordinates.
    """
    # Divide the circle into 4 quadrants centred on each cardinal direction
    # and return the closest one.
    theta_norm = math.atan2(math.sin(theta), math.cos(theta))  # normalise

    if -math.pi / 4 <= theta_norm < math.pi / 4:
        return (0, 1)    # RIGHT
    elif math.pi / 4 <= theta_norm < 3 * math.pi / 4:
        return (1, 0)    # DOWN
    elif theta_norm >= 3 * math.pi / 4 or theta_norm < -3 * math.pi / 4:
        return (0, -1)   # LEFT
    else:
        return (-1, 0)   # UP
