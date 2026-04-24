import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import random
from env.game import PolytopiaEnv

class RandomAgent:
    """
    picks a random action every step, a
    baseline to compare against other agents
    """
    def __init__(self, num_actions: int):
        self.num_actions = num_actions

    def select_action(self, obs, env) -> int:
        return random.randint(0, self.num_actions - 1)
        
from agents.base import run_episode, evaluate

from agents.base import run_episode, evaluate

if __name__ == "__main__":
    import sys
    render = "--render" in sys.argv
    episodes = 100
    for arg in sys.argv[1:]:
        if arg.startswith("--episodes="):
            episodes = int(arg.split("=")[1])
    print(f"\nRunning Random Agent for {episodes} episodes...\n")
    evaluate(RandomAgent(PolytopiaEnv.NUM_ACTIONS), "Random", num_episodes=episodes, render=render)