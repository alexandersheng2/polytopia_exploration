import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class PPOAgent:
    """
    Wraps a Stable-Baselines3 PPO model behind the same
    select_action(obs, env) interface the other agents use.
    """

    def __init__(self, model_path: str):
        from stable_baselines3 import PPO
        self.model = PPO.load(model_path)

    def select_action(self, obs, env=None) -> int:
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)
