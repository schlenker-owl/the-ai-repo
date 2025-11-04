from airoad.rl.gridworld import GridWorld
from airoad.rl.value_iteration import simulate_policy, value_iteration


def test_value_iteration_reaches_goal():
    env = GridWorld(
        rows=4,
        cols=4,
        start_pos=(3, 0),
        walls=((1, 1), (2, 1)),
        terminals={(0, 3): 1.0},
        step_reward=-0.02,
    )
    V, pi, info = value_iteration(env, gamma=0.99, tol=1e-9, max_iter=1000)

    # Policy should reach terminal quickly with positive return
    G, steps, ok = simulate_policy(env, pi, max_steps=50)
    assert ok, "Policy did not reach terminal"
    assert steps <= 20, f"Took too many steps: {steps}"
    assert G > 0.6, f"Return too low: {G}"

    # Value at start should reflect positive outlook
    assert V[env.start_state] > 0.6
    assert info["residual"] < 1e-6
