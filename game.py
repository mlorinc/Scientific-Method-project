import pygame
import random
from enums import Algorithm, Map, Direction, TileState
import numpy as np
from queue import PriorityQueue
import time
from typing import List, Tuple

CYAN = (0, 255, 255)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)

color_mapping = {
   TileState.PLAYER: CYAN,
   TileState.UNVISITED: BLACK,
   TileState.VISITED: WHITE,
   TileState.OBSTACLE: RED
}

direction_vectors = {
        Direction.Up: np.array([0, 1]),
        Direction.Down: np.array([0, -1]),
        Direction.Left: np.array([-1, 0]),
        Direction.Right: np.array([1, 0]),
        Direction.UL: np.array([-0.5**0.5, 0.5**0.5]),
        Direction.LD: np.array([-0.5**0.5, -0.5**0.5]),
        Direction.DR: np.array([0.5**0.5, -0.5**0.5]),
        Direction.RU: np.array([0.5**0.5, 0.5**0.5]),
    }

def get_orentation_vector(direction):
  return direction_vectors[direction]

class Game(object):
    def __init__(self, screen_width: int=600, rotation_velocity: float = 180, velocity: float = 0.5) -> None:
        self.units_traveled = 0
        self.error = 0
        self.rotation_accumulator = 0
        self.current_direction = Direction.Right
        self.position = np.array([0, 0])
        self.ev3_path = []
        self._screen_width = screen_width
        self._rotation_velocity = rotation_velocity
        self._velocity = velocity
        self._render_queue = []
        pygame.init()
        pygame.display.set_caption("Vacuum bot")

    def get_tile_state(self, position: np.ndarray) -> TileState:
       return TileState(self.grid[tuple(position)])

    def set_tile_state(self, position: np.ndarray, state: TileState) -> None:
       self.grid[tuple(position)] = state.value 

    def prepare_map(self, grid_size: Tuple[int, int], start: Tuple[int, int], obstacles: List[Tuple[int, int]], grid_dimension: Tuple[int, int] = (30, 30)):
      self.grid_size = grid_size
      self.grid_dimension = grid_dimension
      self.start_position = start
      self.position = start
      self.grid = np.zeros(grid_size, dtype=np.int8)
      self.screen_dimension = np.array([grid_size[0] * grid_dimension[0] + self._screen_width, grid_size[1] * grid_dimension[1]])
      self.screen = pygame.display.set_mode(self.screen_dimension)
      self.set_tile_state(start, TileState.VISITED)
      self._obstacles = obstacles
      for obstacle in obstacles:
         self.set_tile_state(obstacle, TileState.OBSTACLE)
      self.unvisited_count = self.grid.size - 1 - len(obstacles)
      self.side_dimension = np.array([self._screen_width, self.screen_dimension[1]])
      self.side_start = self.screen_dimension - self.side_dimension

    def reset(self):
      grid = np.copy(self.grid)
      self.position = self.start_position
      self.grid = np.zeros(self.grid_size, dtype=np.int8)
      self.set_tile_state(self.start_position, TileState.VISITED)
      for obstacle in self._obstacles:
         self.set_tile_state(obstacle, TileState.OBSTACLE)
      self.unvisited_count = self.grid.size - 1 - len(self._obstacles)
      self.current_direction = Direction.Right
      return grid

    def update(self, direction) -> TileState:
        if not self.can_move(direction, self.position):
            return TileState.OBSTACLE

        move_vector = get_move_vector(direction)
        old_position = np.copy(self.position)
        self.position += move_vector

        self.rotation_accumulator += calculate_rotation(get_orentation_vector(self.current_direction), get_orentation_vector(self.current_direction))
        ev3_rotation = EV3Rotation(get_orentation_vector(self.current_direction), get_orentation_vector(self.current_direction))
        move_distance = 1 if not direction in [Direction.UL, Direction.LD, Direction.DR, Direction.RU] else 1.414
        result = None
        if self.get_tile_state(self.position) == TileState.UNVISITED:
            self.set_tile_state(self.position, TileState.VISITED)
            self.unvisited_count -= 1
            self._render_queue.append(self.position)
            self._render_queue.append(old_position)
            result = TileState.UNVISITED
        else:
            result = TileState.VISITED
            self.error += move_distance
            
        self.units_traveled += move_distance
        self.ev3_path.append((ev3_rotation, move_distance))
        
        self.current_direction = direction
        return result

    def draw_rectangle(self, position: np.ndarray, color=None):
      color = color or color_mapping[self.get_tile_state(position)]
      pygame.draw.rect(self.screen, color, (position[0] * self.grid_dimension[0], position[1] * self.grid_dimension[1], self.grid_dimension[0], self.grid_dimension[1]))

    def render(self, lazy=False):
        
        if not lazy:
          # Fill the screen with the grid
          self.screen.fill((0, 0, 0))
          for y in range(self.grid_size[1]):
              for x in range(self.grid_size[0]):
                  self.draw_rectangle((x, y))
        else:
           for change in self._render_queue:
              self.draw_rectangle(tuple(change))

        self.draw_rectangle(self.position, CYAN)
        
        # Draw the side screen
        pygame.draw.rect(self.screen, (128, 128, 128), (self.side_start[0], self.side_start[1], self.side_dimension[0], self.side_dimension[1]))
        
        font_size = 36
        font = pygame.font.Font(None, font_size)

        text_surface = font.render(f"Units traveled: {float('%.2f' % (self.units_traveled))}", True, (255, 255, 255))
        self.screen.blit(text_surface, (self.grid_size[0] - self._screen_width + 10, 10))

        text_surface = font.render(f"Error: {float('%.2f' % (self.error))}", True, (255, 255, 255))
        self.screen.blit(text_surface, (self.grid_size[0] - self._screen_width + 10, font_size + 10 + 10))

        text_surface = font.render(f"Rotations made: {float('%.2f' % (self.rotation_accumulator))}", True, (255, 255, 255))
        self.screen.blit(text_surface, (self.grid_size[0] - self._screen_width + 10, 2*font_size + 2*10 + 10))

        pygame.display.update()
    
    def square_in_sight(self, direction, position: np.ndarray):
        return self.get_tile_state(position + get_move_vector(direction))

    def can_move(self, direction, position = None):
        new_position = (position if position is not None else self.position) + get_move_vector(direction)
        return new_position[0] in range(0, self.grid_size[0]) and new_position[1] in range(0, self.grid_size[1]) and self.get_tile_state(new_position) != TileState.OBSTACLE

    def calculate_time_taken(self):
      time = 0
      for (rot, dist) in self.ev3_path:
        time += abs(rot) / self._rotation_velocity
        time += dist * self._velocity
      return time
  
    def data_to_tuple(self):
        return tuple([
          self.units_traveled,
          self.error,
          self.rotation_accumulator,
          self.calculate_time_taken(),
          self.start_position
        ])


def calculate_rotation(vector1, vector2):
  dot_product = np.dot(vector1, vector2)

  magnitude1 = np.linalg.norm(vector1)
  magnitude2 = np.linalg.norm(vector2)
  cosine_theta = dot_product / (magnitude1 * magnitude2)
  cosine_theta = max(min(cosine_theta, 1), -1) 
  angle_radians = np.arccos(cosine_theta)

  return np.degrees(angle_radians)

def EV3Rotation(vector1, vector2): 
  cross_product = np.cross(vector1, vector2)

  magnitude1 = np.linalg.norm(vector1) 
  magnitude2 = np.linalg.norm(vector2) 
  sine_theta = cross_product / (magnitude1 * magnitude2) 
  sine_theta = max(min(sine_theta, 1), -1) 
  angle_radians = np.arcsin(sine_theta)
  
  angle = np.degrees(angle_radians)

  if angle == 0 and not np.array_equal(vector1, vector2):
    angle = 180
  return angle

def get_move_vector(direction):
  player_y, player_x = 0, 0

  if direction == Direction.Up:
    player_y -= 1 
  elif direction == Direction.Down:
    player_y += 1
  elif direction == Direction.Left:
    player_x -= 1 
  elif direction == Direction.Right:
    player_x += 1
  elif direction == Direction.UL:
    player_x -= 1
    player_y -= 1
  elif direction == Direction.LD:
    player_x -= 1
    player_y += 1 
  elif direction == Direction.DR:
    player_x += 1
    player_y += 1
  elif direction == Direction.RU:
    player_x += 1
    player_y -= 1

  return np.array([player_x, player_y])