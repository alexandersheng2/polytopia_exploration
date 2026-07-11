# polytopia_exploration

A reinforcement learning project with the goal of achieving optimal map exploration in a Polytopia-inspired fog-of-war game environment.

Four agents are implemented and benchmarked against each other: a Random agent, Greedy (BFS) agent, Deep Q-Network trained from scratch in PyTorch, and a PPO agent trained with Stable-Baselines3. A web demo lets you pick an agent and watch an animated playthrough in the browser.

---

## RESULTS

| Agent   | Success Rate | Avg Steps | Avg Tiles Explored |
|---------|--------------|-----------|--------------------|
| Random  | 0%           | 120.0     | 72.0 / 100         |
| DQN     | 12%          | 115.4     | 69.4 / 100         |
| PPO     | 91%          | 47.2 (40.0 on successes) | 99.6 / 100 |
| Greedy  | 100%         | 64.8      | 100.0 / 100        |

Greedy still wins on success rate, having direct access to the fog map and always moving toward the nearest unseen tile, which is close to unbeatable on a static map. PPO comes close on success rate and is actually **faster** than Greedy when it does succeed (~40 vs ~65 steps), having learned an efficient sweep pattern purely from reward signal. DQN was improved from an original ~4% success rate (see [Why DQN Struggles](#why-dqn-struggles)) but is still the weakest learned agent, consistent with its known difficulty on sparse-reward exploration tasks.

---

## Web Demo

A FastAPI backend runs a full episode server-side and returns it as a JSON trajectory; a vanilla JS/Canvas frontend animates it in the browser.

```bash
uvicorn server.main:app --reload
```

Then open `http://localhost:8000`, pick an agent from the dropdown, and hit Run.

---

## Environment

- 10×10 grid with fog of war
- One warrior unit with movement 1 and 3×3 vision
- 8 possible actions (4 cardinal + 4 diagonal directions)
- Episode ends when all tiles are explored or 120 steps are reached
- Warrior spawns randomly (away from edges) each episode

**Observation space:** a flat array of 110 floats
- 100 values: fog map (10×10 flattened), 0.0 = unseen, 1.0 = seen
- 2 values: warrior (x, y) position normalized to [0, 1]
- 8 values: one-hot encoding of the last action taken (all zero before the first step)

**Reward:**
- `+1` per new tile revealed
- `-0.05` for a valid move that reveals no new tiles (revisiting)
- `-0.1` for hitting the boundary

---

## Project Structure

```
polytopia_exploration/
├── env/
│   ├── __init__.py
│   ├── grid.py            # 10x10 grid, fog of war logic
│   ├── unit.py             # Warrior class, 8-directional movement
│   └── game.py              # Main environment, Gym-style interface
├── agents/
│   ├── base.py              # Shared run_episode and evaluate functions
│   ├── registry.py           # Agent registry (id -> name/description/factory) for the API
│   ├── random_agent.py       # Baseline: random action each step
│   ├── greedy_agent.py       # BFS to nearest unexplored tile
│   ├── dqn_agent.py          # Deep Q-Network with replay buffer, Double DQN
│   └── ppo_agent.py          # Wraps a trained Stable-Baselines3 PPO model
├── training/
│   ├── train.py               # DQN training loop
│   ├── dqn_weights.pth        # Trained DQN checkpoint
│   ├── gym_env.py             # gymnasium.Env adapter around PolytopiaEnv, for SB3
│   ├── train_ppo.py           # PPO training script (Stable-Baselines3)
│   └── ppo_model.zip          # Trained PPO checkpoint
├── server/
│   ├── main.py                 # FastAPI app: /api/agents, /api/run, static frontend mount
│   └── trajectory.py            # Runs one episode server-side, returns it as JSON
├── frontend/
│   ├── index.html               # Web demo page
│   ├── app.js                    # Fetches agents, runs episodes, animates on canvas
│   └── style.css
├── play.py                       # Manual play in the terminal
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone https://github.com/yourusername/polytopia_exploration
cd polytopia_exploration
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
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

**Run an agent from the command line:**
```bash
python3 agents/random_agent.py
python3 agents/greedy_agent.py
python3 agents/random_agent.py --episodes=50
python3 agents/greedy_agent.py --render
```

**Run the web demo:**
```bash
uvicorn server.main:app --reload
```

**Train the DQN agent:**
```bash
python3 training/train.py
```

**Train the PPO agent:**
```bash
python3 training/train_ppo.py
```

---

## Implementation Details

### Greedy Agent
Uses breadth-first search from the warrior's current position to find the nearest unexplored tile, then takes the first step of the shortest path toward it. BFS is recomputed every step to react to newly revealed tiles.

### DQN Agent
A neural network maps the 110-dimensional observation to Q values for all 8 actions. Training uses:
- **Replay buffer** (capacity 10,000) for decorrelated experience sampling
- **Target network** synced every 100 steps for stable Bellman targets
- **Double DQN updates** — the online network selects the next action, the target network evaluates it, reducing the Q-value overestimation bias of vanilla DQN
- **Epsilon-greedy exploration** decaying from 1.0 to 0.05 over 3000 episodes
- **Gradient clipping** (max norm 1.0) to prevent weight explosion
- **Reward clipping** to [-1, 1] for stable Q value estimation

Network architecture:
```
Linear(110 → 128) → ReLU → Linear(128 → 128) → ReLU → Linear(128 → 8)
```

### PPO Agent
Trained with [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) against `training/gym_env.py`, a thin `gymnasium.Env` adapter around `PolytopiaEnv` (SB3 requires a real Gymnasium environment; the core game logic is untouched). Unlike DQN, PPO learns a direct policy (a probability distribution over actions) rather than Q-values, using a clipped surrogate objective for stable on-policy updates. Trained for 500k timesteps with SB3's default `MlpPolicy` hyperparameters.

---

## Why DQN Struggles

The Greedy agent is a strong baseline for this problem because it has direct access to the fog map and always moves toward the nearest unseen tile. DQN must infer this behaviour from reward signal alone, which is difficult because:

- Rewards for the final unexplored tiles are sparse and delayed
- The agent rarely reaches "almost fully explored" states during training, so the network never learns what to do there
- Basic DQN is known to struggle with exploration-heavy problems

Investigating the original ~4%-success DQN's behaviour showed it frequently oscillating between two adjacent tiles rather than making progress. This was due to the observation having no memory of the agent's previous action. Thus, two already-explored adjacent tiles could produce near-identical inputs to the network, giving it no signal to tell "just came from there" from "haven't been here." Adding a last-action to the observation and moving the revisit penalty into the environment's actual reward function (previously applied as a training-script bolt-on), and switching to Double DQN roughly tripled the success rate! (4% → 12%). Training PPO on the same task and getting 91% success confirms the idea that this is fundamentally a sparse-reward exploration problem that a more modern policy-gradient method handles substantially better than vanilla DQN.

---

## Future Work (trying to make it closer to how Polytopia works)

- **Variable grid size** (roughly 5×5 to 20×20) — the environment already takes `width`/`height`, but DQN/PPO's fixed-size input layer would need either per-size retrained checkpoints or a size-invariant CNN encoder
- **Multiple warriors** (1–3, cooperatively controlled) — would require `PolytopiaEnv` to manage a list of units and agents to return one action per warrior
- **Island/water terrain generation** — Randomize the map generation to look more like islands. Could procedurally generate land/water with cellular automata + a flood-fill reachability check at generation time to guarantee all land is reachable.
- **Deploy the web demo publicly** (Render / Hugging Face Spaces) for use as a live interview/portfolio link
