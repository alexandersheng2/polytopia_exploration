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

    def select_action(self, obs) -> int:
        return random.randint(0, self.num_actions - 1)
        
def run_episode(env: PolytopiaEnv, agent: RandomAgent, render: bool = False) -> dict:
    """
    Run one full episode and return a dict with:
        steps       - how many steps the episode took
        explored    - how many tiles were explored
        total       - total tiles in the grid
        success     - whether all tiles were explored
    """

    env = PolytopiaEnv()
    obs, info = env.reset()

    if render:
        env.render()

    while True:
        action = agent.select_action(obs)
        obs, reward, done, info = env.step(action)

        if render:
            env.render()
            print(f"  Reward: {reward:+.1f} | Explored: {info['explored']}/{info['total']}")

        if done:
            break
    
    return {
        "steps": info["steps"],
        "explored": info["explored"],
        "total": info["total"],
        "success": env.grid.all_explored(),
    }

def evaluate(num_episodes: int = 100, render: bool = False) -> None:
    """
    Run an agent for num_episodes and print summary stats
    """
    env = PolytopiaEnv
    agent = RandomAgent(num_actions=PolytopiaEnv.NUM_ACTIONS)

    results = []
    for episode in range(num_episodes):
        result = run_episode(env, agent, render = render)
        results.append(result)
        if not render:
            print(f"  Episode {episode + 1:>3} | Steps: {result['steps']:>3} | "
                  f"Explored: {result['explored']:>3}/{result['total']} | "
                  f"{'✓' if result['success'] else '✗'}")
    
    successes = [r for r in results if r["success"]]
    avg_steps = sum(r["steps"] for r in results) / len(results)
    avg_explored = sum(r["explored"] for r in results) / len(results)
    success_rate = len(successes) / len(results) * 100

    print(f"\n{'─' * 45}")
    print(f"  Episodes:      {num_episodes}")
    print(f"  Success rate:  {success_rate:.1f}%")
    print(f"  Avg steps:     {avg_steps:.1f}")
    print(f"  Avg explored:  {avg_explored:.1f}/{results[0]['total']}")
    if successes:
        avg_success_steps = sum(r["steps"] for r in successes) / len(successes)
        print(f"  Avg steps (successes only): {avg_success_steps:.1f}")
    print(f"{'─' * 45}")

if __name__ == "__main__":
    import sys
    render = "--render" in sys.argv
    episodes = 100

    for arg in sys.argv[1:]:
        if arg.startswith("--episodes--"):
            episodes = int(arg.split("=")[1])

    print(f"\nRunning Random Agent for {episodes} episodes...\n")
    evaluate(num_episodes=episodes, render=render)



    