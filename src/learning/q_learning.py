"""
Q-Learning: Reinforcement Learning for Grid Navigation
=======================================================

  Q-Learning → the agent knows NOTHING about the map upfront.
               It learns by trial and error: try an action, receive a
               reward, update internal knowledge, repeat.

THE CORE IDEA: THE Q-TABLE
---------------------------
The agent maintains a table  Q(state, action)  that answers:

    "If I am in state s and I take action a,
     how much TOTAL reward can I expect from here to the goal?"

Key word: TOTAL (not just immediate). The agent values actions that
lead to good long-term outcomes, not just the next step.

THE BELLMAN EQUATION — how the Q-table is updated
---------------------------------------------------
After every step (s, a) → (s', reward):

    Q(s, a) ← Q(s, a) + α * [  r  +  γ * max_a'(Q(s', a'))  -  Q(s, a)  ]
                    

THE TWO PHASES
--------------
  1. TRAINING  — agent explores the environment over many episodes,
                  collecting rewards and updating Q(s, a) after every step.

  2. EXECUTION — once trained, the agent follows the GREEDY POLICY:
                  at each state, take the action with the highest Q-value.

EXPLORATION vs. EXPLOITATION (ε-greedy)
----------------------------------------
During training, the agent must balance two competing needs:
  - Explore: try new actions to discover better strategies
  - Exploit: use what it already knows to collect high rewards

ε-greedy solves this:
  - With probability ε  → choose a RANDOM action (explore)
  - With probability 1-ε → choose the BEST known action (exploit)

ε starts at 1.0 (pure exploration) and decays each episode toward
ε_min (mostly exploitation). This way the agent explores freely at
first, then shifts to refining its best-known strategy.
"""

import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.environment.grid_world import GridWorld


# ---------------------------------------------------------------------------
# Action space  (must match GridWorld's 4-connected neighbour convention)
# ---------------------------------------------------------------------------

# Each action is a (Δrow, Δcol) vector
ACTIONS: List[Tuple[int, int]] = [
    (-1,  0),   # 0 = UP    (row decreases)
    ( 1,  0),   # 1 = DOWN  (row increases)
    ( 0, -1),   # 2 = LEFT  (col decreases)
    ( 0,  1),   # 3 = RIGHT (col increases)
]
ACTION_NAMES: List[str] = ["UP", "DOWN", "LEFT", "RIGHT"]
N_ACTIONS: int = len(ACTIONS)


# ---------------------------------------------------------------------------
# Reward constants  (tuned to make shorter paths more valuable)
# ---------------------------------------------------------------------------

REWARD_GOAL:         float =  100.0   # large positive  → reaching goal is desirable
REWARD_STEP:         float =   -1.0   # small negative  → encourages shorter paths
REWARD_INVALID_MOVE: float =   -5.0   # negative        → discourages hitting walls


class QLearningAgent:
    """
    Q-Learning agent for grid-based navigation.

    TWO-PHASE WORKFLOW
    ------------------
    1. agent.train(start, goal, episodes=500)   ← learns the Q-table
    2. path = agent.get_path(start, goal)       ← follows greedy policy

    Parameters
    ----------
    environment : GridWorld
        The grid environment (used for validity checks and bounds).
    alpha : float
        Learning rate (0 < α ≤ 1).  How strongly each new experience
        updates the Q-value. 
    gamma : float
        Discount factor (0 < γ ≤ 1).  How much the agent values future
        rewards vs. immediate ones.  γ = 0.95 means the agent plans ahead.
    epsilon : float
        Initial exploration rate.  Starts at 1.0 (fully random) and
        decays toward epsilon_min over training episodes.
    epsilon_decay : float
        Multiplicative decay applied to ε after each episode.
        ε_new = max(epsilon_min, ε * epsilon_decay)
    epsilon_min : float
        Minimum exploration rate.  Even a well-trained agent explores
        a little to avoid getting stuck in local optima.
    seed : int | None
        NumPy random seed for reproducibility (useful in tests).
    """

    def __init__(
        self,
        environment: GridWorld,
        alpha:         float = 0.1,
        gamma:         float = 0.95,
        epsilon:       float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min:   float = 0.01,
        seed: Optional[int]  = None,
    ) -> None:
        self.env           = environment
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.seed          = seed

        # Save the initial epsilon so train() can reset it on each call (for second or another times training runs)
        self._epsilon_initial = epsilon
        self.epsilon          = epsilon

        # Q-table: shape (height, width, N_ACTIONS), all zeros initially (no experience).
        self.q_table = np.zeros(
            (environment.height, environment.width, N_ACTIONS),
            dtype=float,
        )

        # Populated by train() and get_path()
        self.training_rewards: List[float] = []   # total reward per episode during training
        self.path: Optional[List[Tuple[int, int]]] = None 
        self.stats: Dict = {} # training and path-finding statistics 

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _choose_action(self, state: Tuple[int, int]) -> int:
        """
        ε-greedy action selection.

        With probability ε  → random action (explore new territory)
        With probability 1-ε → best known action (exploit Q-table)
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(N_ACTIONS)           # explore
        return int(np.argmax(self.q_table[state[0], state[1]]))  # exploit

    def _step(
        self,
        state:  Tuple[int, int],
        action: int,
        goal:   Tuple[int, int],
    ) -> Tuple[Tuple[int, int], float, bool]:
        """
        Execute one action from the current state.

        Returns
        -------
        next_state : (row, col)
            Where the agent ended up.  Equals state if the move was invalid.
        reward : float
            Immediate reward signal for this transition.
        done : bool
            True if the goal was reached.

        Reward design rationale
        -----------------------
        REWARD_GOAL (+100)         : large enough to dominate step costs.
        REWARD_STEP (-1)           : small step penalty drives the agent to
                                     prefer shorter paths (even after discount).
        REWARD_INVALID_MOVE (-5)   : teaches the agent to avoid walls without
                                     overwhelming the learning signal.
        """
        dr, dc = ACTIONS[action]
        new_r, new_c = state[0] + dr, state[1] + dc

        # Invalid move: out of bounds or hits an obstacle
        if not self.env.is_valid(new_r, new_c):
            # Agent stays in place and receives a penalty
            return state, REWARD_INVALID_MOVE, False

        next_state = (new_r, new_c)

        if next_state == goal:
            return next_state, REWARD_GOAL, True   # success!

        return next_state, REWARD_STEP, False       # normal step

    def _bellman_update(
        self,
        state:      Tuple[int, int],
        action:     int,
        reward:     float,
        next_state: Tuple[int, int],
    ) -> None:
        """
        Apply the Bellman update to Q(state, action).

        Formula:
            Q(s, a) ← Q(s, a) + α * [r + γ * max_a'(Q(s', a')) - Q(s, a)]
        
        Over time, this update rule allows the agent to learn accurate Q-values (Q-table converges to reliable values)

        """
        # best_next_q = max_a'(Q(s', a')) -> From the new cell,  what's the best Q-value available?
        best_next_q = float(np.max(self.q_table[next_state[0], next_state[1]]))

        # td_target = r + γ * best_next_q -> What's the REAL value of this action, now that we've seen what happened?
        td_target   = reward + self.gamma * best_next_q
        
        # td_error = td_target - Q(s, a) -> How wrong was our old estimate?
        td_error    = td_target - self.q_table[state[0], state[1], action]

        # If td_error > 0 → reality was better than expected → increase Q
        # If td_error < 0 → reality was worse than expected  → decrease Q
        self.q_table[state[0], state[1], action] += self.alpha * td_error

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def train(
        self,
        start:                  Tuple[int, int],
        goal:                   Tuple[int, int],
        episodes:               int = 500,
        max_steps_per_episode:  int = 200,
    ) -> List[float]:
        """
        Train the agent by running episodes.

        Each episode:
          1. Reset agent to `start`.
          2. Loop: choose action (ε-greedy) → step → Bellman update.
          3. Stop when goal is reached or max_steps exceeded.
          4. Decay ε.

        The Q-table is reset at the start of each train() call 

        Returns
        -------
        List[float]
            Total reward collected per episode.  Use this to plot
            the training curve and observe convergence.
        """
        if not self.env.is_valid(*start):
            raise ValueError(f"Start position {start} is not valid!")
        if not self.env.is_valid(*goal):
            raise ValueError(f"Goal position {goal} is not valid!")

        if self.seed is not None:
            np.random.seed(self.seed)

        start_time = time.time()

        # --- Reset state for a fresh training run ---
        self.q_table = np.zeros(
            (self.env.height, self.env.width, N_ACTIONS), dtype=float
        )
        self.epsilon          = self._epsilon_initial
        self.training_rewards = []

        # --- Training loop ---
        for episode in range(episodes):
            state        = start
            total_reward = 0.0

            for _ in range(max_steps_per_episode):
                # 1. Choose action
                action = self._choose_action(state)

                # 2. Execute action in the environment
                next_state, reward, done = self._step(state, action, goal)

                # 3. Update Q-table with Bellman equation
                self._bellman_update(state, action, reward, next_state)

                state        = next_state
                total_reward += rewar

                # Check if episode is done (goal reached)
                if done:
                    break

            # 4. Decay exploration rate after each episode so early episodes explore widely while later episodes refine
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            self.training_rewards.append(total_reward)

        self.stats = {
            "algorithm":     "Q-Learning",
            "episodes":      episodes,
            "final_epsilon": round(self.epsilon, 4), # how much exploration is left at the end of training
            "time_seconds":  time.time() - start_time,
        }

        return self.training_rewards

    def get_path(
        self,
        start:     Tuple[int, int],
        goal:      Tuple[int, int],
        max_steps: int = 500,
    ) -> Optional[List[Tuple[int, int]]]:
        """
        Walk from start to goal using only what was learned, no randomness, no exploration.

        Detects cycles (visited same cell twice)

        Call this AFTER train().  If called before training, the
        Q-table is all zeros and the agent will move randomly

        Returns
        -------
        List[(row, col)] from start to goal, or None if policy fails.
        """
        if not self.env.is_valid(*start):
            raise ValueError(f"Start position {start} is not valid!")
        if not self.env.is_valid(*goal):
            raise ValueError(f"Goal position {goal} is not valid!")

        state   = start
        path    = [state]
        visited = {state} # set is O(1) while list is O(n) → use set to track visited cells for cycle detection

        for _ in range(max_steps):
            if state == goal:
                self.path = path
                # Merge path stats into the existing stats dict
                self.stats.update({
                    "path_length": len(path),
                    "success":     True,
                })
                return path

            # Pure Greedy -> always take the action with the highest Q-value
            action = int(np.argmax(self.q_table[state[0], state[1]]))
            dr, dc = ACTIONS[action]

            # Calculate the next cell based on the chosen action
            new_r, new_c = state[0] + dr, state[1] + dc

            if not self.env.is_valid(new_r, new_c):
                # Greedy policy hits a wall → policy is inadequate
                break

            next_state = (new_r, new_c)

            # If we visit a cell we've already been to, the agent is stuck in a loop and we abort.
            if next_state in visited:
                # Cycle detected → policy is looping
                break

            visited.add(next_state)
            path.append(next_state)
            state = next_state

        # Policy failed to reach goal (Failure handling)
        self.path = None
        self.stats.update({
            "path_length": 0,
            "success":     False,
        })
        return None

    def get_best_action(self, state: Tuple[int, int]) -> int:
        """Return the index of the best action at `state` (for visualisation)."""
        return int(np.argmax(self.q_table[state[0], state[1]]))

    def get_q_values(self, state: Tuple[int, int]) -> np.ndarray:
        """Return all Q-values for `state` as a 1-D array of length N_ACTIONS."""
        return self.q_table[state[0], state[1]].copy()

    def get_stats(self) -> Dict:
        """Return training and path-finding statistics."""
        return self.stats
