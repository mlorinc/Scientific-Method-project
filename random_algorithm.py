import random
from algorithm import AlgorithmBase
from game import Game
from enums import Direction

class RandomAlgorith(AlgorithmBase):
    def __init__(self, game: Game) -> None:
        super().__init__(game)
    def execute(self):
        self.game.update(Direction(random.randint(0, 7)))