from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .gridworld import Action, GridWorld


def value_iteration(
    env: GridWorld,
    gamma: float = 0.99,
    tol: float = 1e-6,
    max_iter: int = 1000,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Deterministic VI for small tabular MDPs.
    Returns (V, pi, info) where:
      - V: value vector [n_states]
      - pi: greedy deterministic policy [n_states] with actions in {0..3}; arbitrary on terminals
      - info: {"iters": int, "residual": float}
    """
    nS, nA = env.n_states, env.n_actions
    V = np.zeros(nS, dtype=np.float64)

    def q_for_state(s: int, v: np.ndarray) -> np.ndarray:
        if env.is_terminal(s):
            return np.zeros(nA, dtype=np.float64)
        q = np.empty(nA, dtype=np.float64)
        for a in range(nA):
            ns, r, done = env.next_state(s, a)
            q[a] = r + (0.0 if done else gamma * v[ns])
        return q

    residual = np.inf
    iters = 0
    for it in range(max_iter):
        iters = it + 1
        V_new = V.copy()
        for s in range(nS):
            if env.is_terminal(s):
                V_new[s] = 0.0
            else:
                V_new[s] = np.max(q_for_state(s, V))
        residual = float(np.max(np.abs(V_new - V)))
        V = V_new
        if residual < tol:
            break

    # Greedy policy
    pi = np.zeros(nS, dtype=int)
    for s in range(nS):
        qs = q_for_state(s, V)
        pi[s] = int(np.argmax(qs)) if not env.is_terminal(s) else 0

    return V, pi, {"iters": iters, "residual": residual}


def simulate_policy(
    env: GridWorld,
    pi: np.ndarray,
    max_steps: int = 100,
) -> Tuple[float, int, bool]:
    """
    Roll out a deterministic policy once from env.start_state.
    Returns (return, steps, reached_terminal).
    """
    s = env.reset()
    G = 0.0
    for t in range(max_steps):
        a: Action = int(pi[s])
        s, r, done = env.step(a)
        G += r
        if done:
            return G, t + 1, True
    return G, max_steps, False
