import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random

import numpy as np

from agents import registry
from env.game import PolytopiaEnv


def _newly_revealed(before: np.ndarray, after: np.ndarray) -> list[list[int]]:
    """Diff two explored-grids and return [x, y] pairs that flipped to True."""
    ys, xs = np.where(after & ~before)
    return [[int(x), int(y)] for y, x in zip(ys, xs)]


def run_trajectory(agent_id: str, seed: int | None = None) -> dict:
    """
    Run one full episode with the given agent and return the entire
    playthrough as a single JSON-serializable trajectory.
    """
    if seed is not None:
        random.seed(seed)

    spec = registry.get(agent_id)
    agent = spec.factory()
    env = PolytopiaEnv()

    obs, info = env.reset()

    frames = []
    empty = np.zeros_like(env.grid.explored)
    frames.append({
        "step": 0,
        "warriors": [{"id": 0, "x": env.warrior.x, "y": env.warrior.y}],
        "newly_revealed": _newly_revealed(empty, env.grid.explored),
        "action": None,
        "reward": 0.0,
        "explored_count": info["explored"],
    })

    while True:
        before = env.grid.explored.copy()
        action = agent.select_action(obs, env)
        obs, reward, done, info = env.step(action)

        frames.append({
            "step": info["steps"],
            "warriors": [{"id": 0, "x": env.warrior.x, "y": env.warrior.y}],
            "newly_revealed": _newly_revealed(before, env.grid.explored),
            "action": action,
            "reward": reward,
            "explored_count": info["explored"],
        })

        if done:
            break

    return {
        "agent_id": spec.id,
        "agent_name": spec.name,
        "grid_width": env.width,
        "grid_height": env.height,
        "max_steps": env.max_steps,
        "success": env.grid.all_explored(),
        "total_steps": info["steps"],
        "total_explored": info["explored"],
        "total_tiles": info["total"],
        "frames": frames,
    }
