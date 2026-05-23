# polytopia_exploration

A reinforcement learning project with the goal of achieving optimal map exploration in a Polytopia-inspired fog-of-war game environment.

Three agents are implemented and benchmarked against each other: a Random baseline, a Greedy agent, and a Deep Q-Network trained from scratch in PyTorch.

---

## RESULTS

| Agent   | Success Rate | Avg Steps | Avg Tiles Explored |
|---------|--------------|-----------|--------------------|
| Random  | 0%           | 120.0     | 72.0 / 100         |
| Greedy  | 100%         | 64.8      | 100.0 / 100        |
| DQN     | ~4%          | 118.2     | 86.1 / 100         |

The Greedy agent ends up dominating, having perfect knowledge of the fog map via BFS. The DQN agent learns some exploration strategies but struggles with sparse rewards in the final tiles

---

## Environment
 
- 10×10 grid with fog of war
- One warrior unit with movement 1 and 3×3 vision
- 8 possible actions (4 cardinal + 4 diagonal directions)
- Episode ends when all tiles are explored or 120 steps are reached
- Warrior spawns randomly (away from edges) each episode
**Observation space:** a flat array of 102 floats
- 100 values: fog map (10×10 flattened), 0.0 = unseen, 1.0 = seen
- 2 values: warrior (x, y) position normalized to [0, 1]
**Reward:**
- `+1` per new tile revealed
- `-0.05` for revisiting an explored tile
- `-0.1` for hitting the boundary

---

## Project Structure
 
```
polytopia_exploration/
├── env/
│   ├── __init__.py
│   ├── grid.py          # 10x10 grid, fog of war logic
│   ├── unit.py          # Warrior class, 8-directional movement
│   └── game.py          # Main environment, Gym-style interface
├── agents/
│   ├── base.py          # Shared run_episode and evaluate functions
│   ├── random_agent.py  # Baseline: random action each step
│   ├── greedy_agent.py  # BFS to nearest unexplored tile
│   └── dqn_agent.py     # Deep Q-Network with replay buffer
├── training/
│   └── train.py         # DQN training loop
├── play.py              # Manual play in the terminal
└── README.md
```
 
---
 
## Setup
 
```bash
git clone https://github.com/yourusername/polytopia_exploration
cd polytopia_exploration
python3 -m venv .venv
source .venv/bin/activate
pip install numpy torch
```
 
---
 
## Usage
 
**Play manually:**
```bash
python3 play.py
```
 
Controls:
```
q  w  e
a     d
z  x  c
```
 
**Run an agent:**
```bash
python3 agents/random_agent.py
python3 agents/greedy_agent.py
python3 agents/random_agent.py --episodes=50
python3 agents/greedy_agent.py --render
```
 
**Train the DQN agent:**
```bash
python3 training/train.py
```
 
---
 
## Implementation Details
 
### Greedy Agent
Uses breadth-first search (BFS) from the warrior's current position to find the nearest unexplored tile, then takes the first step of the shortest path toward it. BFS is recomputed every step to react to newly revealed tiles.
 
### DQN Agent
A neural network maps the 102-dimensional observation to Q values for all 8 actions. Training uses:
- **Replay buffer** (capacity 10,000) for decorrelated experience sampling
- **Target network** synced every 100 steps for stable Bellman targets
- **Epsilon-greedy exploration** decaying from 1.0 to 0.05 over 3000 episodes
- **Gradient clipping** (max norm 1.0) to prevent weight explosion
- **Reward clipping** to [-1, 1] for stable Q value estimation
Network architecture:
```
Linear(102 → 128) → ReLU → Linear(128 → 128) → ReLU → Linear(128 → 8)
```
 
---
 
## Why DQN Struggles
 
The Greedy agent is a strong baseline for this problem because it has direct access to the fog map and always moves optimally toward the nearest unseen tile. DQN must infer this behaviour from reward signal alone, which is difficult because:
 
- Rewards for the final unexplored tiles are sparse and delayed
- The agent rarely reaches "almost fully explored" states during training, so the network never learns what to do there
- Basic DQN is known to struggle with exploration-heavy problems — more advanced algorithms like PPO or curiosity-driven exploration would likely perform better
---
 
## Future Work

- Add multiple warrior units to start coordinated exploration
- Implement PPO via Stable Baselines 3 for comparison
- Add villages, star income, and troop spawning for a richer environment (likely less comparison between agents)
- Visualize training curves and agent behaviour with matplotlib or pygame
