"""
Manual play — test the environment by typing moves yourself.
 
Controls:
    w = up        e = up-right    d = right    c = down-right
    x = down      z = down-left   a = left     q = up-left
    (or type the action number 0-7 directly)
    quit / exit to stop
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.game import PolytopiaEnv
from env.unit import Warrior

KEY_MAP = {
    "w": 0, "x": 1, "a": 2, "d": 3,
    "q": 4, "e": 5, "z": 6, "c": 7,
}

def main():
    env = PolytopiaEnv()
    obs, info = env.reset()
    env.render()

    print("\nControls: w=↑  x=↓  a=←  d=→  q=↖  e=↗  z=↙  c=↘  |  'quit' to exit\n")

    while True:
        raw = input("Move: ").strip().lower()

        if raw in ("quit", "exit"):
            break
    
        if raw in KEY_MAP:
            action = KEY_MAP[raw]
        elif raw.isdigit() and int(raw) in range(8):
            action = int(raw)
        else:
            print(f"  invalid input '{raw}'. Use w/a/s/d/q/e/z/c or 0-7.")
            continue

        obs, reward, done, info = env.step(action)
        print()
        env.render()
        print(f"  → Action: {Warrior.ACTION_LABELS[action]}  Reward: {reward:+.1f}  "
              f"Explored: {info['explored']}/{info['total']}")
        
        if done:
            if env.grid.all_explored():
                print(f"\n All tiles explored in {info['steps']} steps")
            else:
                print(f"\n Max steps reached ({info['steps']}). "
                      f"Explored {info['explored']}/{info['total']} tiles.")
            break

if __name__ == "__main__":
    main()