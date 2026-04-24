import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import deque
import random
from env.game import PolytopiaEnv
from env.unit import Warrior

class GreedyAgent:
    """
    Always moves toward the nearest unexplored tile.
    Uses BFS to find the closest unseen tile and takes
    the first step of the path toward it.
    """

    def select_action(self, obs, env: PolytopiaEnv) -> int:
        """
        Find the nearest unexplored tile via BFS and return
        the action that moves one step toward it.
        """
        start = (env.warrior.x, env.warrior.y)
        target, path = self._bfs(start, env)

        if target is None:
            # no unexplored tiles found — pick randomly
            return random.randint(0, PolytopiaEnv.NUM_ACTIONS - 1)
        
        # path[0] is start, path[1] is the next tile to move to
        next_x, next_y = path[1]
        dx = next_x - env.warrior.x
        dy = next_y - env.warrior.y

        # find action that results in change of (dx, dy)
        for action, (adx, ady) in Warrior.ACTIONS.items():
            if adx == dx and ady == dy:
                return action
            
        # should never happen
        return random.randint(0, PolytopiaEnv.NUM_ACTIONS - 1)
    
    def _bfs(self, start: tuple, env: PolytopiaEnv):
        """
        Breadth first search from start position.
        Returns (target_tile, path) where path is a list of (x, y)
        positions from start to target, or (None, None) if no
        unexplored tiles exist.
        """
        queue = deque()
        queue.append([start])
        visited = {start}

        while queue:
            path = queue.popleft()
            x, y = path[-1]

            if not env.grid.explored[y][x]:
                return (x, y), path
            
            for dx, dy in Warrior.ACTIONS.values():
                nx, ny = x + dx, y + dy
                if env.grid.in_bounds(nx, ny) and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append(path + [(nx, ny)])

        return None, None # all tiles explored
    
from agents.base import run_episode, evaluate

if __name__ == "__main__":
    import sys
    render = "--render" in sys.argv
    episodes = 100
    for arg in sys.argv[1:]:
        if arg.startswith("--episodes="):
            episodes = int(arg.split("=")[1])
    print(f"\nRunning Greedy Agent for {episodes} episodes...\n")
    evaluate(GreedyAgent(), "Greedy", num_episodes=episodes, render=render)
