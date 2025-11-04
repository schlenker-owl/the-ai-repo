#!/usr/bin/env python
from airoad.rl.gridworld import GridWorld
from airoad.rl.q_learning import q_learning
from airoad.rl.value_iteration import simulate_policy


def main():
    env = GridWorld()
    Q, pi = q_learning(env, episodes=400, alpha=0.5, gamma=0.99, epsilon=0.1, random_state=0)
    G, steps, ok = simulate_policy(env, pi, max_steps=50)
    print("Q-learning:")
    print(f" - rollout: return={G:.3f}, steps={steps}, reached={ok}")


if __name__ == "__main__":
    main()
