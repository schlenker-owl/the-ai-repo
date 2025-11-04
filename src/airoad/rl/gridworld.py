from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

Action = int  # 0=up, 1=right, 2=down, 3=left


@dataclass
class GridWorld:
    """
    Small deterministic GridWorld for tabular RL.

    - Coordinates: (row, col), row=0..rows-1 (top to bottom), col=0..cols-1 (left to right)
    - Actions: 0=up, 1=right, 2=down, 3=left
    - Walls: impassable cells; attempt to move into a wall/border -> stay
    - Terminals: dict[(row, col)] = reward added upon ENTERING that cell; episode ends
    - Step reward: added on every transition (including transitions that end in a terminal)

    Mapping:
      We build a dense state index over *valid* cells (non-walls), so s in [0, n_states).
    """

    rows: int = 4
    cols: int = 4
    start_pos: Tuple[int, int] = (3, 0)
    walls: Iterable[Tuple[int, int]] = ((1, 1), (2, 1))
    terminals: Dict[Tuple[int, int], float] = None
    step_reward: float = -0.02

    def __post_init__(self) -> None:
        if self.terminals is None:
            # goal at top-right
            self.terminals = {(0, 3): 1.0}

        self._walls = set(self.walls)
        self._valid_positions: List[Tuple[int, int]] = [
            (r, c) for r in range(self.rows) for c in range(self.cols) if (r, c) not in self._walls
        ]
        # Dense mapping for tabular arrays
        self._pos_to_state = {p: i for i, p in enumerate(self._valid_positions)}
        self._state_to_pos = list(self._valid_positions)

        if self.start_pos in self._walls:
            raise ValueError("start_pos cannot be a wall")
        if self.start_pos not in self._pos_to_state:
            raise ValueError("start_pos must be inside the grid")

        self.start_state: int = self._pos_to_state[self.start_pos]
        self.state: int = self.start_state  # for step()

        # Precompute terminal states in index space
        self._terminal_states = {
            self._pos_to_state[p]: rew
            for p, rew in self.terminals.items()
            if p in self._pos_to_state
        }

    # ---------- basic properties ----------

    @property
    def n_states(self) -> int:
        return len(self._state_to_pos)

    @property
    def n_actions(self) -> int:
        return 4

    def is_terminal(self, s: int) -> bool:
        return s in self._terminal_states

    # ---------- indexing ----------

    def idx_to_pos(self, s: int) -> Tuple[int, int]:
        return self._state_to_pos[s]

    def pos_to_idx(self, pos: Tuple[int, int]) -> int:
        return self._pos_to_state[pos]

    # ---------- dynamics ----------

    _DELTA = {
        0: (-1, 0),  # up
        1: (0, 1),  # right
        2: (1, 0),  # down
        3: (0, -1),  # left
    }

    def next_state(self, s: int, a: Action) -> Tuple[int, float, bool]:
        """
        Deterministic transition model from state index s with action a.
        Returns (s_next, reward, done).
        """
        if self.is_terminal(s):
            return s, 0.0, True

        r, c = self.idx_to_pos(s)
        dr, dc = self._DELTA[a]
        nr, nc = r + dr, c + dc

        # Check borders + walls; if invalid, stay
        if not (0 <= nr < self.rows and 0 <= nc < self.cols) or (nr, nc) in self._walls:
            ns = s
        else:
            ns = self._pos_to_state[(nr, nc)]

        rew = self.step_reward
        done = False
        if ns in self._terminal_states:
            rew += float(self._terminal_states[ns])
            done = True

        return ns, rew, done

    # ---------- episodic API for Q-learning convenience ----------

    def reset(self) -> int:
        self.state = self.start_state
        return self.state

    def step(self, a: Action) -> Tuple[int, float, bool]:
        ns, r, done = self.next_state(self.state, a)
        self.state = ns
        return ns, r, done
