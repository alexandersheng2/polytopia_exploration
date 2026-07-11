import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
 
import numpy as np
from env.game import PolytopiaEnv
from agents.dqn_agent import DQNAgent


def train(
    num_episodes: int = 3000,
    print_every: int = 100,
    save_path: str = "training/dqn_weights.pth",
):
    """
    Train the DQN agent and save the weights.
 
    Prints a summary every `print_every` episodes showing:
        - average steps
        - average tiles explored
        - success rate
        - current epsilon
        - average loss
    """

    env = PolytopiaEnv()
    # create agent - get obs_size from environment
    obs, _ = env.reset()
    agent = DQNAgent(
        obs_size=len(obs),
        num_actions=PolytopiaEnv.NUM_ACTIONS,
    )

    # tracking stats
    episode_steps    = []
    episode_explored = []
    episode_success  = []
    episode_losses   = []

    print(f"Training DQN agent for {num_episodes} episodes...\n")

    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        losses = []

        while not done:
            # select action
            action = agent.select_action(obs, env)

            # take step
            next_obs, reward, done, info = env.step(action)
            reward = max(-1.0, min(1.0, reward)) # clip reward

            # store experience
            agent.store_experience(obs, action, reward, next_obs, done)

            # train on a random batch
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)

            obs = next_obs
    
        # end of episode
        agent.decay_epsilon()
 
        episode_steps.append(info["steps"])
        episode_explored.append(info["explored"])
        episode_success.append(env.grid.all_explored())
        episode_losses.append(np.mean(losses) if losses else 0.0)
 
         # print progress every N episodes
        if (episode + 1) % print_every == 0:
            recent = slice(episode + 1 - print_every, episode + 1)
            avg_steps    = np.mean(episode_steps[recent])
            avg_explored = np.mean(episode_explored[recent])
            avg_loss     = np.mean(episode_losses[recent])
            success_rate = np.mean(episode_success[recent]) * 100
 
            print(
                f"  Episode {episode + 1:>5} | "
                f"Steps: {avg_steps:>6.1f} | "
                f"Explored: {avg_explored:>5.1f}/100 | "
                f"Success: {success_rate:>5.1f}% | "
                f"Epsilon: {agent.epsilon:.3f} | "
                f"Loss: {avg_loss:.4f}"
            )

    # save weights
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    agent.save(save_path)
    print(f"\nWeights saved to {save_path}")
    print("Training complete.")
 
    return agent
 
 
if __name__ == "__main__":
    train()
 