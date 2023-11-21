import random
from algorithm import AlgorithmBase
from game import Game
from enums import Direction, TileState

class SemiRandomAlgorith(AlgorithmBase):
    def __init__(self, game: Game) -> None:
        super().__init__(game)
    def execute(self):
        dir_array = self.get_movement_array(Direction.Up)
        valid_dir_array = [dir for dir in dir_array if self.game.can_move(dir)]
        neighbourhood_black_cells = [dir for dir in valid_dir_array if self.game.square_in_sight(dir, self.game.position) == TileState.UNVISITED]

        dir = None
        if neighbourhood_black_cells:
            dir = random.choice(neighbourhood_black_cells)
        else:
            dir = random.choice(valid_dir_array)

        self.game.update(dir)