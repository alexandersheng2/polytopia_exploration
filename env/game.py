import random
import numpy as np
from env.grid import Grid
from env.unit import Warrior

class PolytopiaEnv: 
    """
    Polytopia exploration environment.
 
    Gym-style interface:
        obs, info = env.reset()
        obs, reward, done, info = env.step(action)
 
    State (observation):
        A flat numpy array of length 110:
        - 100 values: fog map (10x10 flattened), 0.0 = unseen, 1.0 = seen
        - 2 values: warrior's (x, y) position normalized to [0, 1]
        - 8 values: one-hot encoding of the last action taken (all zero before the first step)

    Actions:
        0=up, 1=down, 2=left, 3=right,
        4=up-left, 5=up-right, 6=down-left, 7=down-right

    Rewards:
        +1  for each new tile revealed this step
        -0.05 for a valid move that reveals no new tiles (revisiting)
        -0.1 penalty for hitting the boundary (invalid move)
    """

    NUM_ACTIONS = 8

    def __init__(self, width: int = 10, height: int = 10):
        self.width = width
        self.height = height
        self.grid = Grid(width, height)
        self.warrior = None
        self.steps = 0
        self.max_steps = 120
        self.last_action = None

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def reset(self) -> tuple[np.ndarray, dict]:
        """Reset the environment and return the initial observation."""
        self.grid.reset()
        self.steps = 0

        # spawn warrior at random position
        start_x = random.randint(1, self.width - 2)
        start_y = random.randint(1, self.height - 2)
        self.warrior = Warrior(start_x, start_y)

        self.grid.reveal(self.warrior.x, self.warrior.y, self.warrior.vision_radius)

        return self._get_obs(), self._get_info()
    
    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        """
        Apply action and return (observation, reward, done, info).
        """
        self.steps += 1
        self.last_action = action

        nx, ny = self.warrior.get_next_position(action)

        # if out of bounds, penalize and don't move
        if not self.grid.in_bounds(nx, ny):
            reward = -0.1
            done = self.grid.all_explored() or self.steps >= self.max_steps
            return self._get_obs(), reward, done, self._get_info()

        # valid move
        before = self.grid.explored_count()
        self.warrior.move(action)
        self.grid.reveal(self.warrior.x, self.warrior.y, self.warrior.vision_radius)
        after = self.grid.explored_count()

        reward = float(after - before)
        if reward == 0.0:
            reward = -0.05  # revisiting an already-explored tile
        done = self.grid.all_explored() or self.steps >= self.max_steps

        return self._get_obs(), reward, done, self._get_info()
    
    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self):
        """print the grid to the terminal"""
        print(self._render_str())

    def _render_str(self) -> str:
        lines = []
        header = f"  Steps: {self.steps} | Explored: {self.grid.explored_count()}/{self.grid.total_tiles()}"
        lines.append(header)
        lines.append("  " + "──" * self.width)
        for y in range(self.height):
            row = " |"
            for x in range(self.width):
                if (x, y) == (self.warrior.x, self.warrior.y):
                    row += "W " # warrior
                elif self.grid.explored[y][x]:
                    row += ". " #explored
                else:
                    row += "█ "   # fog
            row += "|"
            lines.append(row)

        lines.append("  " + "──" * self.width)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        fog_flat = self.grid.explored.flatten().astype(np.float32)
        pos = np.array([
            self.warrior.x / (self.width - 1),
            self.warrior.y / (self.height - 1),
        ], dtype=np.float32)
        last_action_onehot = np.zeros(self.NUM_ACTIONS, dtype=np.float32)
        if self.last_action is not None:
            last_action_onehot[self.last_action] = 1.0
        return np.concatenate([fog_flat, pos, last_action_onehot])
    
    def _get_info(self) -> dict:
        return {
            "explored": self.grid.explored_count(),
            "total": self.grid.total_tiles(),
            "steps": self.steps,
            "position": self.warrior.position(),
        }