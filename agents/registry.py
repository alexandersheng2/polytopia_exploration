import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from env.game import PolytopiaEnv

REPO_ROOT = Path(__file__).resolve().parent.parent
DQN_WEIGHTS_PATH = REPO_ROOT / "training" / "dqn_weights.pth"
PPO_MODEL_PATH = REPO_ROOT / "training" / "ppo_model.zip"


@dataclass
class AgentSpec:
    id: str
    name: str
    description: str
    factory: Callable[[], object]  # zero-arg -> object with select_action(obs, env)
    public: bool = True


_REGISTRY: dict[str, AgentSpec] = {}


def register(spec: AgentSpec) -> None:
    _REGISTRY[spec.id] = spec


def get(agent_id: str) -> AgentSpec:
    return _REGISTRY[agent_id]


def list_public() -> list[AgentSpec]:
    return [s for s in _REGISTRY.values() if s.public]


def list_all() -> list[AgentSpec]:
    return list(_REGISTRY.values())


def _make_dqn():
    from agents.dqn_agent import DQNAgent
    obs, _ = PolytopiaEnv().reset()  # measure the real observation shape rather than hardcoding it
    agent = DQNAgent(obs_size=len(obs), num_actions=PolytopiaEnv.NUM_ACTIONS)
    agent.load(str(DQN_WEIGHTS_PATH))
    agent.epsilon = 0.0  # deterministic inference for the demo, no exploration noise
    return agent


def _make_greedy():
    from agents.greedy_agent import GreedyAgent
    return GreedyAgent()


def _make_random():
    from agents.random_agent import RandomAgent
    return RandomAgent(PolytopiaEnv.NUM_ACTIONS)


def _make_ppo():
    from agents.ppo_agent import PPOAgent
    return PPOAgent(str(PPO_MODEL_PATH))


register(AgentSpec(
    id="dqn",
    name="DQN (Deep Q-Network)",
    description=(
        "Trained from scratch with a replay buffer, target network, and "
        "epsilon-greedy exploration. Learns to explore purely from reward "
        "signal - watch it struggle on the last few tiles, a direct "
        "consequence of sparse rewards near full exploration."
    ),
    factory=_make_dqn,
    public=True,
))

register(AgentSpec(
    id="greedy",
    name="Greedy (BFS)",
    description=(
        "Recomputes a breadth-first search to the nearest unexplored tile "
        "every step, using perfect knowledge of the fog boundary. "
        "A strong, non-learned baseline."
    ),
    factory=_make_greedy,
    public=True,
))

register(AgentSpec(
    id="random",
    name="Random baseline",
    description="Picks a random action every step. The simplest possible baseline.",
    factory=_make_random,
    public=True,
))

register(AgentSpec(
    id="ppo",
    name="PPO (Proximal Policy Optimization)",
    description=(
        "Trained with Stable-Baselines3. Learns a direct policy rather than "
        "Q-values, which handles this task's sparse rewards far better than "
        "DQN - it explores the full map in about 40 steps on average, faster "
        "than the hand-coded Greedy baseline."
    ),
    factory=_make_ppo,
    public=True,
))
