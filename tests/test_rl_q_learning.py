from airoad.rl.gridworld import GridWorld
from airoad.rl.q_learning import q_learning


def test_q_learning_learns_good_policy():
    env = GridWorld(
        rows=4,
        cols=4,
        start_pos=(3, 0),
        walls=((1, 1), (2, 1)),
        terminals={(0, 3): 1.0},
        step_reward=-0.02,
    )
    Q, pi = q_learning(
        env,
        episodes=400,  # fast & sufficient
        alpha=0.5,
        gamma=0.99,
        epsilon=0.1,
        max_steps_per_ep=60,
        random_state=0,
    )

    # Roll out greedy policy once
    G = 0.0
    s = env.reset()
    for t in range(30):
        a = int(pi[s])
        s, r, done = env.step(a)
        G += r
        if done:
            assert t + 1 <= 20
            assert G > 0.6
            return

    assert False, "Greedy policy failed to reach terminal within 30 steps"
