from game import Game
from abc import ABC, abstractmethod
from enums import Direction

class AlgorithmBase(ABC):
    def __init__(self, game: Game) -> None:
        self.game = game
    
    @abstractmethod
    def execute(self):
        pass

    def get_movement_array(self, direction = Direction.Up):
        directions = [Direction.Up, Direction.RU, Direction.Right, Direction.DR, Direction.Down, Direction.LD, Direction.Left, Direction.UL]
        index = directions.index(direction)
        
        return directions[index:] + directions[:index]
