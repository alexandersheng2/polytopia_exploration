import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from training.gym_env import GymPolytopiaEnv


def train(total_timesteps: int = 300_000, save_path: str = "training/ppo_model"):
    """
    Train a PPO agent on the Polytopia exploration environment and save it.
    """
    env = Monitor(GymPolytopiaEnv())
    model = PPO("MlpPolicy", env, verbose=1)

    print(f"Training PPO agent for {total_timesteps} timesteps...\n")
    model.learn(total_timesteps=total_timesteps)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    model.save(save_path)
    print(f"\nModel saved to {save_path}.zip")

    return model


if __name__ == "__main__":
    train()
