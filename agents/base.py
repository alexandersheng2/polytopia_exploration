import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.game import PolytopiaEnv


def run_episode(env: PolytopiaEnv, agent, render: bool = False) -> dict:
    """
    Run one full episode and return the results.
    Works with any agent that implements select_action(obs, env).

    Returns a dict with:
        steps     — how many steps the episode took
        explored  — how many tiles were explored
        total     — total tiles in the grid
        success   — whether all tiles were explored
    """
    obs, info = env.reset()

    if render:
        env.render()

    while True:
        action = agent.select_action(obs, env)
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


def evaluate(agent, agent_name: str, num_episodes: int = 100, render: bool = False) -> dict:
    """
    Run an agent for many episodes and print summary stats.
    Works with any agent that implements select_action(obs, env).

    Returns the results list so callers can do further analysis.
    """
    env = PolytopiaEnv()
    results = []

    for episode in range(num_episodes):
        result = run_episode(env, agent, render=render)
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
    print(f"  Agent:         {agent_name}")
    print(f"  Episodes:      {num_episodes}")
    print(f"  Success rate:  {success_rate:.1f}%")
    print(f"  Avg steps:     {avg_steps:.1f}")
    print(f"  Avg explored:  {avg_explored:.1f}/{results[0]['total']}")
    if successes:
        avg_success_steps = sum(r["steps"] for r in successes) / len(successes)
        print(f"  Avg steps (successes only): {avg_success_steps:.1f}")
    print(f"{'─' * 45}")

    return results