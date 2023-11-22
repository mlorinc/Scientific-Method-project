import gymnasium
from gymnasium import spaces
import numpy as np
from typing import Tuple
from game import Game, get_move_vector
from enums import Direction, TileState

class GridEnvironment(gymnasium.Env):
    def __init__(self, game: Game):
        super(GridEnvironment, self).__init__()

        # Define action and observation spaces
        self.action_space = spaces.Discrete(len(Direction))
        self.observation_space = spaces.Box(low=0, high=2, shape=game.grid_size, dtype=np.int8)
        self.game = game
        self.steps_done = 0

    def reset(self):
        self.steps_done = 0
        output = self.game.reset()
        self.game.render()
        return output

    def step(self, action: Direction):
        # Perform the action and update the state
        tile = self.game.update(Direction(action))
        # Reward logic (customize based on your needs)
        reward = -0.1  # Small negative reward for each step
        reward -= 0.3 if Direction(action) in [Direction.DR, Direction.LD, Direction.RU, Direction.UL] else 0
        if tile == TileState.OBSTACLE:  # Obstacle
            reward -= 1.0  # Additional penalty for hitting an obstacle
        elif tile == TileState.VISITED:
            reward -= 0.5  # Additional penalty for revisiting a visited state
        elif tile == TileState.UNVISITED:
            reward += 1.0  # Positive reward for reaching the goal

        # Check if the episode is done
        done = self.game.unvisited_count == 0

        # Return the new state, reward, and done flag
        return self.game.grid.copy(), reward, done

    def render(self):
        # Print the current state (for visualization purposes)
        self.game.render(lazy=False)
