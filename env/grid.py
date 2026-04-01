import numpy as np

class Grid:
    """
    10 x 10 grid implementing fog of war
    tracks which tiles have been explored
    """

    def __init__(self, width: int = 10, height: int = 10):
        self.width = width
        self.height = height

        # True meaning tile has been explored
        self.explored = np.zeros((width, height), dtype = bool)

    #clear all explored tiles
    def reset(self):
        self.explored[:] = False

    # given a position (cx, cy), mark surrounding tiles with square of radius
    # vision_radius as explored
    def reveal(self, cx: int, cy: int, vision_radius: int = 1):
        for dy in range(-vision_radius, vision_radius + 1):
            for dx in range(-vision_radius, vision_radius + 1):
                nx, ny = cx + dx, cy + dy
                if self.in_bounds(nx, ny):
                    self.explored[ny][nx] = True

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height
    
    def explored_count(self) -> int:
        return int(self.explored.sum())
    
    def total_tiles(self) -> int:
        return self.width * self.height
    
    def all_explored(self) -> bool:
        return self.explored_count() == self.total_tiles()
    
