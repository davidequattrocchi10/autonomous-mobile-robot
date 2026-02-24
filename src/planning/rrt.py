"""
RRT (Rapidly-exploring Random Tree) Path Planning
==================================================

WHY RRT EXISTS — the problem it solves
---------------------------------------
BFS and A* work by exhaustively visiting grid cells.
In a small 20×20 grid that is fine, but imagine:
  - A 3D environment with millions of cells
  - A robot with 6-DOF arms (state space = joint angles, not grid cells)
  - A continuous space with no natural discretisation

In these cases, graph search becomes intractable.  RRT avoids the
problem by *sampling* the space randomly instead of enumerating it.

HOW IT DIFFERS FROM A*
-----------------------
  A*     → knows the full map, searches it exhaustively with a heuristic
  RRT    → discovers the space through random samples (no heuristic needed)
  Result → RRT finds *a* valid path quickly, but NOT necessarily shortest.
           A* finds the *shortest* path, but at higher exploration cost.

KEY IDEA (the algorithm in one sentence)
-----------------------------------------
Grow a tree from the start by repeatedly:
  1. Sampling a random point in space
  2. Finding the tree node nearest to it
  3. Extending the tree one step toward that sample
  4. Stopping when the tree reaches the goal

PROBABILISTIC COMPLETENESS
---------------------------
RRT is "probabilistically complete": as max_iterations → ∞ the probability
of finding a path (if one exists) approaches 1.  It is NOT guaranteed to
find the shortest path — that requires the RRT* variant (future work).
"""

import math
import random
import time
from typing import Dict, List, Optional, Tuple

from src.environment.grid_world import GridWorld
from src.planning.graph_search import PathPlanner


class RRT(PathPlanner):
    """
    Rapidly-exploring Random Tree (RRT)

    Strategy   : Grow a tree by randomly sampling the space
    Data struct: List of nodes + parent dict (tree, not a graph)
    Guarantees : Probabilistically complete — finds a path if one exists
                 and max_iterations is large enough.
                 Does NOT guarantee shortest path.

    Parameters
    ----------
    environment : GridWorld
    max_iterations : int
        Maximum number of tree-extension attempts.  More = more likely to
        find a path in cluttered environments.  Default 1 000.
    step_size : int
        How many cells to move per extension step.  step_size=1 means the
        tree grows one cell at a time; larger values grow faster but require
        checking intermediate cells for collisions.  Default 1.
    goal_bias : float
        Probability [0, 1] that the random sample IS the goal.
        WHY: Pure random sampling wastes time far from the goal.  By aiming
        at the goal 10 % of the time we steer the tree toward it efficiently.
        Typical range: 0.05 – 0.20.  Default 0.10.
    seed : int | None
        Fix the random seed for reproducible results (useful in tests).
        None means a fresh random state each run.
    """

    def __init__(
        self,
        environment: GridWorld,
        max_iterations: int = 1000,
        step_size: int = 1,
        goal_bias: float = 0.10,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(environment)

        self.max_iterations = max_iterations
        self.step_size = step_size
        self.goal_bias = goal_bias
        self.seed = seed

        # Full ordered list of nodes in the tree — useful for visualisation
        # (render() uses self.visited which is the corresponding set)
        self.tree_nodes: List[Tuple[int, int]] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _euclidean(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """
        Euclidean (straight-line) distance between two grid cells.

        WHY Euclidean and not Manhattan?
        RRT is conceptually a continuous-space algorithm: it samples in 2-D
        and moves in the direction of the sample.  Euclidean distance is the
        natural notion of "nearest" in 2-D space, and it is consistent with
        the direction vector used in _steer().
        """
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    def _nearest_node(self, q_rand: Tuple[int, int]) -> Tuple[int, int]:
        """
        Return the tree node with smallest Euclidean distance to q_rand.

        Complexity: O(|tree|) per call → overall algorithm is O(n²).
        Acceptable for the grid sizes in this project; for large-scale use
        a KD-tree (scipy.spatial.KDTree) would reduce this to O(n log n).
        """
        return min(self.tree_nodes, key=lambda node: self._euclidean(node, q_rand))

    def _steer(
        self,
        q_near: Tuple[int, int],
        q_rand: Tuple[int, int],
    ) -> Tuple[int, int]:
        """
        Move from q_near toward q_rand by at most step_size cells.

        Algorithm
        ---------
        1. Compute the direction vector (dr, dc) from q_near to q_rand.
        2. Normalise it to unit length.
        3. Scale by step_size and round to the nearest grid cell → q_new.
        4. If q_rand is closer than step_size, jump directly to q_rand.
        5. Clamp the result to grid bounds (safety net).

        The rounding in step 3 allows diagonal moves (distance ≈ √2 per
        step) which makes the tree explore more naturally in open space.
        """
        dr = q_rand[0] - q_near[0]
        dc = q_rand[1] - q_near[1]
        dist = self._euclidean(q_near, q_rand)

        if dist == 0:
            # q_rand and q_near are the same cell — nothing to do
            return q_near

        if dist <= self.step_size:
            # q_rand is already within one step — jump directly to it
            return q_rand

        # Move step_size along the direction vector, then round to a grid cell
        new_r = round(q_near[0] + (dr / dist) * self.step_size)
        new_c = round(q_near[1] + (dc / dist) * self.step_size)

        # Clamp to grid boundaries (should rarely trigger, but defensive)
        new_r = max(0, min(new_r, self.env.height - 1))
        new_c = max(0, min(new_c, self.env.width - 1))

        return (new_r, new_c)

    def _is_collision_free(
        self,
        q_near: Tuple[int, int],
        q_new: Tuple[int, int],
    ) -> bool:
        """
        Check that the straight-line edge q_near → q_new is obstacle-free.

        For step_size = 1: only one cell (q_new) needs checking.
        For step_size > 1: we also check all cells that the straight line
        passes through (Bresenham-style interpolation).

        WHY check intermediate cells?
        If step_size = 5, q_new may be free but an obstacle halfway between
        q_near and q_new would make the edge invalid.  We must reject it.
        """
        # Always check the destination cell first
        if not self.env.is_valid(*q_new):
            return False

        if self.step_size <= 1:
            # Only one cell to verify — already done above
            return True

        # Walk along the segment and check every intermediate grid cell
        r0, c0 = q_near
        r1, c1 = q_new
        # Number of interpolation steps = Chebyshev distance (max of Δrow, Δcol)
        # This ensures every cell on the diagonal is visited
        steps = max(abs(r1 - r0), abs(c1 - c0))

        for i in range(1, steps):          # skip start (valid) and end (already checked)
            r = round(r0 + (r1 - r0) * i / steps)
            c = round(c0 + (c1 - c0) * i / steps)
            if not self.env.is_valid(r, c):
                return False

        return True

    def _sample_random(self, goal: Tuple[int, int]) -> Tuple[int, int]:
        """
        Sample a random free cell in the grid.

        Goal bias
        ---------
        With probability self.goal_bias we return the goal directly.
        This is the key trick that keeps RRT from wandering forever:
        without it the tree explores randomly and may take thousands of
        iterations before stumbling onto the goal region.

        Rejection sampling
        ------------------
        We draw random (row, col) pairs until we find a free cell.
        In typical warehouse grids (mostly free space) this is very fast.
        A fallback returns the goal if 200 draws all hit obstacles.
        """
        # Goal-biased sample
        if random.random() < self.goal_bias:
            return goal

        # Uniform random sample — retry until we land on a free cell
        for _ in range(200):
            r = random.randint(0, self.env.height - 1)
            c = random.randint(0, self.env.width - 1)
            if self.env.is_valid(r, c):
                return (r, c)

        # Fallback: aim for goal (grid is extremely cluttered)
        return goal

    # ------------------------------------------------------------------
    # Public interface (implements PathPlanner ABC)
    # ------------------------------------------------------------------

    def search(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> Optional[List[Tuple[int, int]]]:
        """
        Grow an RRT from start until the goal is reached.

        Returns
        -------
        List of (row, col) tuples from start to goal, or None if the goal
        could not be reached within max_iterations.

        Algorithm walkthrough
        ---------------------
        Each iteration:
          1. Sample  → q_rand  (random free cell, or goal w/ goal_bias)
          2. Nearest → q_near  (closest existing tree node to q_rand)
          3. Steer   → q_new   (move step_size toward q_rand)
          4. Extend  → if edge q_near→q_new is free, add q_new to tree
          5. Connect → if q_new is adjacent to goal, try a direct connection

        Note on "connect to goal" (step 5)
        -----------------------------------
        Without step 5, RRT would only finish when it *exactly* samples the
        goal cell — which has probability 1/(width×height) per iteration.
        By checking whether any newly-added node can see the goal directly,
        we dramatically reduce the average iterations needed.
        """
        # --- Input validation ---
        if not self.env.is_valid(*start):
            raise ValueError(f"Start position {start} is not valid!")
        if not self.env.is_valid(*goal):
            raise ValueError(f"Goal position {goal} is not valid!")

        start_time = time.time()

        # Set random seed for reproducibility (useful in tests)
        if self.seed is not None:
            random.seed(self.seed)

        # --- Trivial case: start IS goal ---
        if start == goal:
            self.path = [start]
            self.tree_nodes = [start]
            self.visited = {start}
            self.stats = {
                "algorithm": "RRT",
                "path_length": 1,
                "nodes_expanded": 0,
                "nodes_visited": 1,
                "time_seconds": time.time() - start_time,
                "success": True,
            }
            return self.path

        # --- Initialise tree ---
        self.tree_nodes = [start]
        # parent[node] = the node that generated it (for path reconstruction)
        parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
        # self.visited = all nodes in the tree (mirrors tree_nodes as a set)
        # Kept as a set for O(1) membership tests
        self.visited: set = {start}
        nodes_expanded = 0  # counts successful tree extensions

        # --- Main loop ---
        for _ in range(self.max_iterations):

            # Step 1 — Sample
            q_rand = self._sample_random(goal)

            # Step 2 — Nearest node in tree to the sample
            q_near = self._nearest_node(q_rand)

            # Step 3 — Steer: move step_size toward q_rand
            q_new = self._steer(q_near, q_rand)

            # Skip if we made no progress or this cell is already in the tree
            if q_new == q_near or q_new in self.visited:
                continue

            # Step 4 — Extend: add q_new only if the edge is obstacle-free
            if not self._is_collision_free(q_near, q_new):
                continue

            nodes_expanded += 1
            self.tree_nodes.append(q_new)
            parent[q_new] = q_near
            self.visited.add(q_new)

            # Step 5a — Did we land exactly on the goal?
            if q_new == goal:
                self.path = self.reconstruct_path(parent, start, goal)
                self.stats = {
                    "algorithm": "RRT",
                    "path_length": len(self.path),
                    "nodes_expanded": nodes_expanded,
                    "nodes_visited": len(self.visited),
                    "time_seconds": time.time() - start_time,
                    "success": True,
                }
                return self.path

            # Step 5b — Can we connect directly to goal from q_new?
            # WHY: We may never *exactly* sample the goal cell.  If q_new is
            # within step_size of the goal AND the edge is clear, we connect
            # immediately and avoid wasting further iterations.
            if (
                self._euclidean(q_new, goal) <= self.step_size
                and goal not in self.visited
                and self._is_collision_free(q_new, goal)
            ):
                parent[goal] = q_new
                self.visited.add(goal)
                self.path = self.reconstruct_path(parent, start, goal)
                self.stats = {
                    "algorithm": "RRT",
                    "path_length": len(self.path),
                    "nodes_expanded": nodes_expanded + 1,
                    "nodes_visited": len(self.visited),
                    "time_seconds": time.time() - start_time,
                    "success": True,
                }
                return self.path

        # --- Max iterations reached without finding the goal ---
        self.stats = {
            "algorithm": "RRT",
            "path_length": 0,
            "nodes_expanded": nodes_expanded,
            "nodes_visited": len(self.visited),
            "time_seconds": time.time() - start_time,
            "success": False,
        }
        return None
