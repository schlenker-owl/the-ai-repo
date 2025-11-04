from __future__ import annotations

from typing import Tuple

import numpy as np

from .gridworld import Action, GridWorld


def greedy_policy_from_q(Q: np.ndarray) -> np.ndarray:
    return np.argmax(Q, axis=1).astype(int)


def q_learning(
    env: GridWorld,
    episodes: int = 400,
    alpha: float = 0.5,
    gamma: float = 0.99,
    epsilon: float = 0.1,
    max_steps_per_ep: int = 100,
    random_state: int | None = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tabular Q-learning with epsilon-greedy exploration.
    Returns (Q, greedy_policy).
    """
    nS, nA = env.n_states, env.n_actions
    Q = np.zeros((nS, nA), dtype=np.float64)
    rng = np.random.default_rng(random_state)

    for _ in range(episodes):
        s = env.reset()
        for _ in range(max_steps_per_ep):
            if rng.random() < epsilon:
                a: Action = int(rng.integers(0, nA))
            else:
                a = int(np.argmax(Q[s]))
            ns, r, done = env.step(a)
            td_target = r + (0.0 if done else gamma * np.max(Q[ns]))
            Q[s, a] = (1.0 - alpha) * Q[s, a] + alpha * td_target
            s = ns
            if done:
                break

    return Q, greedy_policy_from_q(Q)
