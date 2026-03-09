"""
Q-Learning Training and Visualisation
======================================

This example shows the two phases of Q-Learning:

  PHASE 1 — TRAINING
    The agent explores the environment over many episodes, updating its
    Q-table after every step.  We record the total reward per episode
    and plot it as a "training curve" — the signature visualisation of RL.

  PHASE 2 — EXECUTION
    Once trained, the agent follows the greedy policy (always pick the
    action with the highest Q-value).  We visualise:
      • The final path from start to goal
      • The learned POLICY: arrows on every free cell showing the
        direction the agent would move if it were placed there

BFS / A* / RRT produce a single image: here is the path I found.
Q-Learning produces two images:
  1. A learning PROCESS  (training curve — was learning happening?)
  2. A learned POLICY    (not just one path, but a full decision map)

The policy visualisation is the key insight: the agent hasn't just
learned one path — it has learned what to do from EVERY reachable cell.
"""

import sys
sys.path.append('.')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

from src.environment.grid_world import GridWorld
from src.learning.q_learning import QLearningAgent, ACTIONS, N_ACTIONS


# ---------------------------------------------------------------------------
# Scenario — same obstacle layout as compare_algorithms.py
# ---------------------------------------------------------------------------

def create_scenario():
    """Create a 20×20 warehouse scenario with obstacles."""
    env = GridWorld(width=20, height=20)

    env.add_obstacle(18,  5)
    env.add_obstacle(17,  6)
    env.add_obstacle(15,  7)
    env.add_obstacle(18, 14)
    env.add_obstacle(15, 12)
    env.add_obstacle(13, 12)
    env.add_obstacle(12, 12)

    for i in range(5, 15):      # vertical wall at col 8
        env.add_obstacle(i, 8)
    for i in range(12, 18):     # vertical wall at col 3
        env.add_obstacle(i, 3)
    for i in range(6, 11):      # vertical wall at col 16
        env.add_obstacle(i, 16)

    for j in range(3, 10):      # horizontal wall at row 10 (gap at col 6)
        if j != 6:
            env.add_obstacle(10, j)

    env.add_obstacle_rect((15, 15), (17, 17))

    return env


# ---------------------------------------------------------------------------
# Policy visualisation helper
# ---------------------------------------------------------------------------

def draw_policy_arrows(ax, agent, env, start, goal):
    """
    Draw one arrow per free cell showing the agent's best action.

    How it works
    ------------
    For each free cell (row, col):
      - Look up the best action: argmax Q(row, col, *)
      - Translate to a direction vector in plot coordinates
      - Draw a small arrow centred on the cell

    matplotlib imshow coordinate system:
      - x = col  (increases rightward)
      - y = row  (increases downward — y-axis is inverted by imshow)
    So for action UP (dr=-1, dc=0): the arrow direction is (u=0, v=-1), which points visually upward on the plot.
    """
    xs, ys, us, vs, colors = [], [], [], [], []

    for row in range(env.height):
        for col in range(env.width):
            # Skip obstacles, start, and goal (already have markers)
            if not env.is_valid(row, col):
                continue
            if (row, col) == start or (row, col) == goal:
                continue

            # Best action at this cell
            action = agent.get_best_action((row, col))
            dr, dc = ACTIONS[action]

            xs.append(col)     # arrow x-position = column
            ys.append(row)     # arrow y-position = row
            us.append(dc)      # arrow x-direction = Δcol
            vs.append(dr)      # arrow y-direction = Δrow

            # Colour by Q-value magnitude: brighter = agent is more confident
            max_q = float(np.max(agent.get_q_values((row, col))))
            colors.append(max_q)
    
    # If there are no free cells, skip drawing arrows
    if not xs:
        return

    # Normalise colours to [0, 1]
    q_array  = np.array(colors)
    q_min, q_max = q_array.min(), q_array.max()

    # Normalize Q-values to [0, 1] for colour mapping
    if q_max > q_min:
        q_norm = (q_array - q_min) / (q_max - q_min)
    else:
        q_norm = np.zeros_like(q_array)

    # Colour map: red = low Q (uncertain), green = high Q (confident)
    cmap       = plt.cm.RdYlGn    # Red → Yellow → Green
    arrow_cols = [cmap(v) for v in q_norm]
    
    # quiver is matplotlib's dedicated function for drawing fields of arrows. 
    # Instead of drawing 300+ arrows in a loop, I pass all coordinates and directions at once — much faster and smoother.
    ax.quiver(
        xs, ys, us, vs,
        color=arrow_cols,
        scale=18,           # controls arrow length (higher = shorter arrows)
        width=0.004,        # shaft width
        headwidth=4,        # Arrowhead width
        headlength=4,
        alpha=0.75,
    )


# ---------------------------------------------------------------------------
# Training curve helper
# ---------------------------------------------------------------------------

def plot_training_curve(ax, rewards, window=30):
    """
    Plot per-episode rewards with a rolling-average overlay.

    Raw rewards are noisy (single episode = high variance).
    So in one episode the agent might get lucky and score +80, in the next hit several walls and score −20.
    The rolling average shows the actual learning trend.
    """
    episodes = np.arange(1, len(rewards) + 1)

    # Raw rewards (faint)
    ax.plot(episodes, rewards, color='steelblue', alpha=0.25, linewidth=0.8,
            label='Episode reward')

    # Rolling average (bold)
    if len(rewards) >= window:
        rolling = np.convolve(rewards, np.ones(window) / window, mode='valid')
        ax.plot(
            np.arange(window, len(rewards) + 1),
            rolling,
            color='steelblue', linewidth=2.0,
            label=f'Rolling avg (window={window})',
        )

    ax.axhline(0, color='grey', linewidth=0.6, linestyle='--')
    ax.set_xlabel('Episode', fontsize=11)
    ax.set_ylabel('Total reward', fontsize=11)
    ax.set_title('Phase 1 — Training curve', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)

    # Annotate the final rolling-average value
    if len(rewards) >= window:
        final_avg = rolling[-1]
        ax.annotate(
            f'Final avg: {final_avg:.0f}',
            xy=(len(rewards), final_avg),
            xytext=(-60, 10),
            textcoords='offset points',
            fontsize=9,
            color='steelblue',
            arrowprops=dict(arrowstyle='->', color='steelblue', lw=1.2),
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    print("=" * 60)
    print("Q-LEARNING — TRAINING AND VISUALISATION")
    print("=" * 60)

    env   = create_scenario()
    start = (2,  2)
    goal  = (18, 18)

    # -----------------------------------------------------------------------
    # PHASE 1 — Training
    # -----------------------------------------------------------------------
    EPISODES  = 1500
    MAX_STEPS = 300   # max steps per episode (proportional to grid area)

    print(f"\nPhase 1: Training for {EPISODES} episodes...")

    agent = QLearningAgent(
        env,
        alpha         = 0.1,
        gamma         = 0.95,
        epsilon       = 1.0,
        epsilon_decay = 0.997,  # slower decay for a harder 20×20 grid
        epsilon_min   = 0.01,
    )
    rewards = agent.train(start, goal, episodes=EPISODES,
                          max_steps_per_episode=MAX_STEPS)

    stats = agent.get_stats()
    print(f"  Training time : {stats['time_seconds']:.2f}s")
    print(f"  Final epsilon : {stats['final_epsilon']}")
    print(f"  Avg reward (last 100 eps): {np.mean(rewards[-100:]):.1f}")

    # -----------------------------------------------------------------------
    # PHASE 2 — Execution
    # -----------------------------------------------------------------------
    print("\nPhase 2: Following greedy policy...")

    path = agent.get_path(start, goal)

    if path:
        print(f"  Path found!  Length = {len(path)} cells")
    else:
        print("  No path found — try more episodes or a lower epsilon_decay")

    # -----------------------------------------------------------------------
    # Figure: 2-panel layout
    # -----------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle(
        "Q-Learning: Training Curve & Learned Policy",
        fontsize=15, fontweight='bold', y=1.01,
    )

    # --- Left panel: training curve ---
    plot_training_curve(ax1, rewards, window=50)

    # --- Right panel: policy + path ---
    env.render(
        start=start,
        goal=goal,
        path=path,
        visited=set(),      # Q-Learning doesn't track "visited" like BFS
        stats=None,
        title="Phase 2 — Learned policy & path",
        ax=ax2,
        show_legend=False,
    )
    draw_policy_arrows(ax2, agent, env, start, goal)

    # Custom legend for the right panel
    legend_elements = env.get_legend_elements()
    arrow_patch = mpatches.Patch(
        facecolor='none', edgecolor='steelblue',
        label='Policy arrow (green=high Q, red=low Q)',
    )
    ax2.legend(
        handles=legend_elements + [arrow_patch],
        loc='lower center',
        ncol=3,
        fontsize=8,
        frameon=True,
        bbox_to_anchor=(0.5, -0.15),
    )

    plt.tight_layout()
    plt.savefig('qlearning_result.png', dpi=150, bbox_inches='tight')
    print("\nFigure saved → qlearning_result.png")
    plt.show()

    # -----------------------------------------------------------------------
    # Analysis
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)

    print("""
LEFT PANEL — Training curve
  Each point = total reward collected in one episode.
  Raw values are noisy; the rolling average shows the trend.
  A good training run shows rewards INCREASING over time:
    • Early: agent explores randomly → hits walls → low reward
    • Later: agent exploits learned Q-table → reaches goal → high reward

RIGHT PANEL — Learned policy
  Arrows show what the agent WOULD DO from every free cell.
  This is the KEY difference from graph search:
    BFS/A*  → produces one path
    Q-agent → produces a full policy (decision at every cell)

  Arrow colour:
    Green = high Q-value  (agent is confident this is a good direction)
    Red   = low  Q-value  (agent is uncertain or this is a risky move)

  Yellow path = actual greedy path from start (green) to goal (red star)
""")


if __name__ == "__main__":
    run()
