import pygame
import random
from enums import Algorithm, Map, Direction, TileState
import numpy as np
from queue import PriorityQueue
import time
from enums import Algorithm, Direction
from game import Game, get_move_vector, get_orentation_vector
from algorithm import AlgorithmBase
from abc import abstractmethod


class AStarAlgorithm(AlgorithmBase):
    def __init__(self, game: Game) -> None:
        super().__init__(game)
        self.path = []
    def execute(self):
        if not self.path:
            # Generate a new path
            self.path = self.navigate_to_closest_black_cell(self.game.position, self.game.current_direction)
        if self.path:
            next_step = self.path.pop()
            self.game.update(next_step)
  
    def heuristic(self, a, b):
        # Use the Manhattan distance heuristic
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def get_closest_black_cell(self, start, direction):
        queue = [start]
        visited = set()
        movement_array = self.get_movement_array(direction)

        # Iterate through the grid until a black cell is found
        while queue:
            current = queue.pop(0)
            if self.game.get_tile_state(current) == TileState.UNVISITED:
                return current
            visited.add(tuple(current))
            for direction in movement_array:
                if not self.game.can_move(direction, current):
                    continue

                move_vector = get_move_vector(direction)
                new_current = current + move_vector
                if tuple(new_current) not in visited:
                    queue.append(new_current)
                    visited.add(tuple(new_current))
        return None

    def astar(self, start, goal):
        open_set = PriorityQueue()
        open_set.put((0, np.array(start)))
        came_from = {}
        g_score = np.full(self.game.grid_size, np.inf)
        g_score[tuple(start)] = 0
        
        movement_array = self.get_movement_array()
        
        # Uncomment if you dont want to use diagonals
        # movement_array = [Direction.Up, Direction.Right, Direction.Down, Direction.Left]
    
        while not open_set.empty():
            current = np.array(open_set.get()[1])  # Get the node with the lowest total cost (f-score)

            if np.array_equal(current, goal):
                # Reconstruct the path from goal to start by traversing the 'came_from' chain
                path = []
                while tuple(current) in came_from:
                    path.append(came_from[tuple(current)][1])
                    current = came_from[tuple(current)][0]

                return path

            # Explore the neighboring nodes in all directions
            for direction in movement_array:
                if not self.game.can_move(direction, current):
                    continue

                move_vector = get_move_vector(direction)
                new_vector = current + move_vector

                tentative_g_score = g_score[tuple(current)] + 1 if direction in [Direction.Up, Direction.Right, Direction.Left, Direction.Down] else g_score[tuple(current)] + 1.44

                # Update the best path if this node provides a shorter path to the goal
                if tentative_g_score < g_score[tuple(new_vector)]:
                    came_from[tuple(new_vector)] = (current, direction) 
                    g_score[tuple(new_vector)] = tentative_g_score 
                    f_score = tentative_g_score + self.heuristic(goal, new_vector)
                    open_set.put((f_score, tuple(new_vector))) 
        return [] 

    # Using the BFS-based closest black cell logic in the pathfinding algorithm
    def navigate_to_closest_black_cell(self, start, direction):
        closest_black_cell = self.get_closest_black_cell(start, direction)

        if closest_black_cell is not None:
            return self.astar(start, closest_black_cell)
        
        return None
    
class AStarRandom(AStarAlgorithm):
    def __init__(self, game: Game) -> None:
        super().__init__(game)

    def get_closest_black_cell(self, start, direction: Direction):
        black_indexes = [(y, x) for y, row in enumerate(self.game.grid) for x, val in enumerate(row) if TileState(val) == TileState.UNVISITED]

        if not black_indexes:
            return None

        return random.choice(black_indexes)

class AStarSequential(AStarAlgorithm):
    def __init__(self, game: Game) -> None:
        super().__init__(game)

    def get_closest_black_cell(self, start, direction: Direction):
        return super().get_closest_black_cell(start, direction=Direction.Up)

class AStarOrientation(AStarAlgorithm):
    def __init__(self, game: Game) -> None:
        super().__init__(game)

    def get_closest_black_cell(self, start, direction: Direction):
        return super().get_closest_black_cell(start, direction)