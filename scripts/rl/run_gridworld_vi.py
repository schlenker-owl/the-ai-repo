#!/usr/bin/env python
from airoad.rl.gridworld import GridWorld
from airoad.rl.value_iteration import simulate_policy, value_iteration


def main():
    env = GridWorld()
    V, pi, info = value_iteration(env, gamma=0.99, tol=1e-9, max_iter=1000)
    G, steps, ok = simulate_policy(env, pi, max_steps=50)

    print("Value Iteration:")
    print(f" - iters: {info['iters']}, residual: {info['residual']:.3e}")
    print(f" - start value: {V[env.start_state]:.3f}")
    print(f" - rollout: return={G:.3f}, steps={steps}, reached={ok}")
    print(" - policy (0:U,1:R,2:D,3:L):")
    arrows = {0: "↑", 1: "→", 2: "↓", 3: "←"}
    for r in range(env.rows):
        row = []
        for c in range(env.cols):
            if (r, c) in env._walls:
                row.append("■")
            elif (r, c) in env.terminals:
                row.append("G")
            else:
                s = env.pos_to_idx((r, c))
                row.append(arrows[int(pi[s])])
        print(" ".join(row))


if __name__ == "__main__":
    main()
