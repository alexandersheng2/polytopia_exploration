class Warrior:
    """
    Warrior:
        Movement: 1 
        Vision: 1
    """
    
    ACTIONS = {
        0: (0, -1),   # up
        1: (0,  1),   # down
        2: (-1, 0),   # left
        3: (1,  0),   # right
        4: (-1, -1),  # up-left
        5: (1,  -1),  # up-right
        6: (-1,  1),  # down-left
        7: (1,   1),  # down-right
    }

    ACTION_LABELS = {
        0: "↑", 1: "↓", 2: "←", 3: "→",
        4: "↖", 5: "↗", 6: "↙", 7: "↘",
    }

    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y
        self.vision_radius = 1

    # position that unit would have after the action
    def get_next_position(self, action: int) -> tuple[int, int]:
        dx, dy = self.ACTIONS[action]
        return self.x + dx, self.y + dy

    # actually moving the unit
    def move(self, action: int):
        dx, dy = self.ACTIONS[action]
        self.x += dx
        self.y += dy

    def position(self) -> tuple[int, int]:
        return (self.x, self.y)