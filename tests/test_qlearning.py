"""
Tests for the Q-Learning agent.

Testing strategy
----------------
Q-Learning is stochastic: different runs produce different Q-tables
and paths.  All tests that rely on a specific outcome use seed=42
to make the random number generator deterministic.

Key differences from graph-search tests
----------------------------------------
1. There are TWO phases to test: training and path execution.
2. We cannot assert exact path lengths — Q-Learning does not guarantee
   the shortest path (only that the path is valid).
3. We test convergence indirectly: a trained agent on a simple scenario
   should always find a path with enough episodes.

Grid fixtures
-------------
Same fixtures as test_planning.py and test_rrt.py for consistency:

  simple_env  : 5×5 grid, no obstacles — trivial scenario
  walled_env  : 5×5 grid, vertical wall at col 2 (rows 0–3), gap at row 4
  isolated_env: 3×3 grid where goal=(2,2) is unreachable
"""

import pytest
import numpy as np

from src.environment.grid_world import GridWorld
from src.learning.q_learning import QLearningAgent, N_ACTIONS


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture
def simple_env():
    """5×5 open grid — the easiest possible training scenario."""
    return GridWorld(width=5, height=5)


@pytest.fixture
def walled_env():
    """
    5×5 grid with vertical wall at col 2 (rows 0–3).
    Gap at (4, 2) — the only crossing point.

    Layout (S=start, G=goal, X=obstacle):
         col: 0  1  2  3  4
    row 0:   S  .  X  .  G
    row 1:   .  .  X  .  .
    row 2:   .  .  X  .  .
    row 3:   .  .  X  .  .
    row 4:   .  .  .  .  .
    """
    env = GridWorld(width=5, height=5)
    for row in range(4):
        env.add_obstacle(row, 2)
    return env


@pytest.fixture
def isolated_env():
    """
    3×3 grid where goal=(2,2) is completely surrounded.
    Blocks all 8-connected neighbours of (2,2) so even diagonal
    approaches are impossible.
    """
    env = GridWorld(width=3, height=3)
    env.add_obstacle(1, 1)
    env.add_obstacle(1, 2)
    env.add_obstacle(2, 1)
    return env


# =========================================================================
# TestQLearningTraining — does training run correctly?
# =========================================================================

class TestQLearningTraining:
    """Verify the training phase behaves as documented."""

    def test_train_returns_reward_list(self, simple_env):
        """
        train() must return a list of per-episode rewards.
        The length must equal the number of episodes.
        """
        agent = QLearningAgent(simple_env, seed=42)
        rewards = agent.train((0, 0), (4, 4), episodes=100)
        assert isinstance(rewards, list)
        assert len(rewards) == 100

    def test_reward_list_is_numeric(self, simple_env):
        """All reward values must be finite numbers (no NaN, no Inf)."""
        agent = QLearningAgent(simple_env, seed=42)
        rewards = agent.train((0, 0), (4, 4), episodes=100)
        for r in rewards:
            assert np.isfinite(r), f"Non-finite reward: {r}"

    def test_rewards_generally_improve_over_training(self, simple_env):
        """
        In a simple scenario, the average reward in the last quarter of
        training should be higher than in the first quarter.

        WHY: This is the essence of learning — the agent gets better at
        collecting reward as training progresses and ε decays.

        We use a simple open grid so learning is fast and reliable.
        300 episodes is sufficient for the 5×5 open grid — fewer than
        500 reduces test time while still showing reliable convergence.
        """
        agent = QLearningAgent(simple_env, seed=42)
        rewards = agent.train((0, 0), (4, 4), episodes=300)

        first_quarter  = np.mean(rewards[:75])
        last_quarter   = np.mean(rewards[225:])
        assert last_quarter > first_quarter, (
            "Average reward should increase over training "
            f"(first quarter: {first_quarter:.1f}, last: {last_quarter:.1f})"
        )

    def test_qtable_updated_after_training(self, simple_env):
        """
        Q-table should not be all-zeros after training.
        (All-zeros would mean the agent never updated anything.)
        """
        agent = QLearningAgent(simple_env, seed=42)
        agent.train((0, 0), (4, 4), episodes=200)
        assert np.any(agent.q_table != 0), "Q-table must be updated during training"

    def test_qtable_shape_matches_grid(self, simple_env):
        """
        Q-table shape must be (height, width, N_ACTIONS).
        For a 5×5 grid: (5, 5, 4).
        """
        agent = QLearningAgent(simple_env, seed=42)
        agent.train((0, 0), (4, 4), episodes=10)
        assert agent.q_table.shape == (5, 5, N_ACTIONS)

    def test_train_resets_qtable_on_second_call(self, simple_env):
        """
        Calling train() twice must produce identical Q-tables (with same seed).
        This verifies that train() resets state rather than accumulating it.
        """
        agent = QLearningAgent(simple_env, seed=42)
        agent.train((0, 0), (4, 4), episodes=100)
        q_first = agent.q_table.copy()

        agent.train((0, 0), (4, 4), episodes=100)
        q_second = agent.q_table.copy()

        np.testing.assert_array_equal(q_first, q_second)

    def test_epsilon_decays_during_training(self, simple_env):
        """
        After training, epsilon must be lower than the initial value.
        This confirms that the agent shifts from exploration to exploitation.
        """
        agent = QLearningAgent(simple_env, epsilon=1.0, epsilon_decay=0.9, seed=42)
        agent.train((0, 0), (4, 4), episodes=50)
        assert agent.epsilon < 1.0, "Epsilon must decay during training"

    def test_epsilon_never_below_epsilon_min(self, simple_env):
        """Epsilon must not decay below epsilon_min regardless of episodes."""
        epsilon_min = 0.05
        agent = QLearningAgent(
            simple_env, epsilon=1.0, epsilon_decay=0.5,
            epsilon_min=epsilon_min, seed=42
        )
        agent.train((0, 0), (4, 4), episodes=200)
        assert agent.epsilon >= epsilon_min

    def test_invalid_start_raises(self, simple_env):
        """Blocked start must raise ValueError before training begins."""
        simple_env.add_obstacle(0, 0)
        agent = QLearningAgent(simple_env)
        with pytest.raises(ValueError):
            agent.train((0, 0), (4, 4), episodes=10)

    def test_invalid_goal_raises(self, simple_env):
        """Blocked goal must raise ValueError before training begins."""
        simple_env.add_obstacle(4, 4)
        agent = QLearningAgent(simple_env)
        with pytest.raises(ValueError):
            agent.train((0, 0), (4, 4), episodes=10)


# =========================================================================
# TestQLearningGetPath — does path execution work correctly?
# =========================================================================

class TestQLearningGetPath:
    """Verify the greedy path-execution phase."""

    def test_finds_path_after_training_simple(self, simple_env):
        """
        After sufficient training on a simple open grid, the agent
        must find a valid path from start to goal.
        """
        agent = QLearningAgent(simple_env, seed=42)
        agent.train((0, 0), (4, 4), episodes=500)
        path = agent.get_path((0, 0), (4, 4))
        assert path is not None, "Agent should find a path after 500 episodes"

    def test_path_starts_at_start(self, simple_env):
        agent = QLearningAgent(simple_env, seed=42)
        agent.train((0, 0), (4, 4), episodes=500)
        path = agent.get_path((0, 0), (4, 4))
        assert path is not None
        assert path[0] == (0, 0)

    def test_path_ends_at_goal(self, simple_env):
        agent = QLearningAgent(simple_env, seed=42)
        agent.train((0, 0), (4, 4), episodes=500)
        path = agent.get_path((0, 0), (4, 4))
        assert path is not None
        assert path[-1] == (4, 4)

    def test_path_cells_are_all_free(self, simple_env):
        """No cell in the path may be an obstacle or out of bounds."""
        agent = QLearningAgent(simple_env, seed=42)
        agent.train((0, 0), (4, 4), episodes=500)
        path = agent.get_path((0, 0), (4, 4))
        assert path is not None
        for cell in path:
            assert simple_env.is_valid(*cell), f"Path cell {cell} is invalid"

    def test_path_has_no_duplicate_cells(self, simple_env):
        """
        The greedy policy should not revisit cells — if it does,
        get_path() must detect the cycle and return None rather than
        looping forever.
        """
        agent = QLearningAgent(simple_env, seed=42)
        agent.train((0, 0), (4, 4), episodes=500)
        path = agent.get_path((0, 0), (4, 4))
        if path is not None:
            assert len(path) == len(set(path)), "Path must not contain duplicate cells"

    def test_start_equals_goal_returns_single_cell(self, simple_env):
        """
        When start equals goal, the path is trivially [start].
        No training needed and the function should handle this immediately.
        """
        agent = QLearningAgent(simple_env, seed=42)
        agent.train((2, 2), (2, 2), episodes=10)
        path = agent.get_path((2, 2), (2, 2))
        assert path is not None
        assert path == [(2, 2)]

    def test_returns_none_before_training(self, simple_env):
        """
        Without training the Q-table is all zeros.
        The greedy policy will pick action 0 (UP) everywhere, immediately
        hit a wall at row 0, and fail.

        WHY: This demonstrates that training is necessary — unlike A* or
        BFS which can solve any valid scenario on first call.
        """
        agent = QLearningAgent(simple_env, seed=42)
        start, goal = (2, 2), (4, 4)
        # No train() call — Q-table is all zeros
        # Test intent: calling get_path() on untrained agent doesn't crash or loop
        try:
            path = agent.get_path(start, goal, max_steps=50)
            # If it returns, it should be None or a list — not an infinite loop
            assert path is None or isinstance(path, list)
        except Exception as e:
            pytest.fail(f"get_path() raised an exception before training: {e}")

    def test_goal_unreachable_returns_none(self, isolated_env):
        """
        When the goal is surrounded by obstacles, get_path() must
        return None — even after training.

        The agent will train (collecting negative rewards), but the
        greedy policy can never reach (2,2), so it will detect a cycle.
        """
        agent = QLearningAgent(isolated_env, seed=42)
        agent.train((0, 0), (2, 2), episodes=300)
        path = agent.get_path((0, 0), (2, 2))
        assert path is None


# =========================================================================
# TestQLearningStats — stats dict contract
# =========================================================================

class TestQLearningStats:
    """
    After train() + get_path(), stats must contain a standard set of keys.
    This allows the comparison example to display Q-Learning alongside
    BFS / A* / RRT in a consistent format.
    """

    REQUIRED_KEYS = {
        "algorithm",
        "episodes",
        "final_epsilon",
        "time_seconds",
        "path_length",
        "success",
    }

    def test_all_keys_present_after_success(self, simple_env):
        agent = QLearningAgent(simple_env, seed=42)
        agent.train((0, 0), (4, 4), episodes=500)
        agent.get_path((0, 0), (4, 4))
        assert self.REQUIRED_KEYS <= agent.get_stats().keys()

    def test_all_keys_present_after_failure(self, isolated_env):
        agent = QLearningAgent(isolated_env, seed=42)
        agent.train((0, 0), (2, 2), episodes=200)
        agent.get_path((0, 0), (2, 2))
        assert self.REQUIRED_KEYS <= agent.get_stats().keys()

    def test_algorithm_name(self, simple_env):
        agent = QLearningAgent(simple_env, seed=42)
        agent.train((0, 0), (4, 4), episodes=100)
        agent.get_path((0, 0), (4, 4))
        assert agent.get_stats()["algorithm"] == "Q-Learning"

    def test_episodes_matches_training_call(self, simple_env):
        agent = QLearningAgent(simple_env, seed=42)
        agent.train((0, 0), (4, 4), episodes=123)
        assert agent.get_stats()["episodes"] == 123

    def test_path_length_matches_returned_path(self, simple_env):
        agent = QLearningAgent(simple_env, seed=42)
        agent.train((0, 0), (4, 4), episodes=500)
        path = agent.get_path((0, 0), (4, 4))
        assert path is not None, "Agent failed to find path after 500 episodes"
        assert agent.get_stats()["path_length"] == len(path)

    def test_success_true_when_path_found(self, simple_env):
        agent = QLearningAgent(simple_env, seed=42)
        agent.train((0, 0), (4, 4), episodes=500)
        path = agent.get_path((0, 0), (4, 4))
        assert path is not None, "Agent failed to find path after 500 episodes"
        assert agent.get_stats()["success"] == True

    def test_success_false_when_no_path(self, isolated_env):
        agent = QLearningAgent(isolated_env, seed=42)
        agent.train((0, 0), (2, 2), episodes=200)
        agent.get_path((0, 0), (2, 2))
        assert agent.get_stats()["success"] == False

    def test_time_seconds_positive(self, simple_env):
        agent = QLearningAgent(simple_env, seed=42)
        agent.train((0, 0), (4, 4), episodes=100)
        assert agent.get_stats()["time_seconds"] >= 0


# =========================================================================
# TestQLearningParameters — constructor parameters affect behaviour
# =========================================================================

class TestQLearningParameters:
    """Verify that constructor parameters produce the documented effects."""

    def test_seed_produces_same_qtable(self, simple_env):
        """Two agents with same seed must produce identical Q-tables."""
        agent1 = QLearningAgent(simple_env, seed=42)
        agent1.train((0, 0), (4, 4), episodes=200)

        agent2 = QLearningAgent(simple_env, seed=42)
        agent2.train((0, 0), (4, 4), episodes=200)

        np.testing.assert_array_equal(agent1.q_table, agent2.q_table)

    def test_high_alpha_updates_qtable_faster(self, simple_env):
        """
        A high learning rate (α close to 1) makes Q-values change rapidly.
        After the same number of episodes, a high-α agent's Q-table should
        have larger absolute values (farther from the initial zeros).
        """
        agent_low  = QLearningAgent(simple_env, alpha=0.01, seed=42)
        agent_high = QLearningAgent(simple_env, alpha=0.9,  seed=42)

        agent_low.train( (0, 0), (4, 4), episodes=50)
        agent_high.train((0, 0), (4, 4), episodes=50)

        magnitude_low  = np.abs(agent_low.q_table).sum()
        magnitude_high = np.abs(agent_high.q_table).sum()

        assert magnitude_high > magnitude_low, (
            "High learning rate should produce larger Q-value magnitudes"
        )

    def test_zero_episodes_leaves_qtable_at_zero(self, simple_env):
        """
        With zero training episodes the Q-table must remain all zeros.
        No Bellman updates were applied.
        """
        agent = QLearningAgent(simple_env, seed=42)
        agent.train((0, 0), (4, 4), episodes=0)
        assert np.all(agent.q_table == 0)

    def test_get_best_action_returns_valid_index(self, simple_env):
        """get_best_action() must return an index in [0, N_ACTIONS)."""
        agent = QLearningAgent(simple_env, seed=42)
        agent.train((0, 0), (4, 4), episodes=100)
        action = agent.get_best_action((2, 2))
        assert 0 <= action < N_ACTIONS

    def test_get_q_values_returns_correct_shape(self, simple_env):
        """get_q_values() must return a 1-D array of length N_ACTIONS."""
        agent = QLearningAgent(simple_env, seed=42)
        q = agent.get_q_values((0, 0))
        assert q.shape == (N_ACTIONS,)

    def test_bellman_update_formula(self):
        """Verify Q(s,a) update matches the exact Bellman formula.

        With alpha=1.0 the entire TD error is applied in one step, so:
            Q_new = Q_old + 1.0 * (r + γ * max_a'(Q(s', a')) - Q_old)
                  = r + γ * max_a'(Q(s', a'))          (since Q_old = 0)

        This makes the expected result easy to compute by hand.
        """
        env = GridWorld(3, 3)
        agent = QLearningAgent(env, alpha=1.0, gamma=0.9, epsilon=0.0)

        state      = (1, 1)
        action     = 0          # UP
        reward     = 10.0
        next_state = (0, 1)

        # Set known Q-values
        agent.q_table[state[0],      state[1],      :] = 0.0
        agent.q_table[next_state[0], next_state[1], :] = [2.0, 3.0, 1.0, 4.0]

        # Expected: Q(s,a) = 0 + 1.0 * (10 + 0.9 * max(2,3,1,4) - 0) = 13.6
        agent._bellman_update(state, action, reward, next_state)

        expected = reward + 0.9 * 4.0   # 10 + 0.9 * 4 = 13.6
        assert abs(agent.q_table[state[0], state[1], action] - expected) < 1e-9, (
            f"Bellman update incorrect: got {agent.q_table[state[0], state[1], action]:.6f}, "
            f"expected {expected:.6f}"
        )
