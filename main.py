import pygame
import random
from enum import Enum
import numpy as np

def Render():
  # Fill the screen with the grid
  screen.fill((0, 0, 0))
  for y in range(GRID_HEIGHT):
      for x in range(GRID_WIDTH):
          pygame.draw.rect(screen, grid[y][x], (x * GRID_SIZE, y * GRID_SIZE, GRID_SIZE, GRID_SIZE))

  # Draw the player
  pygame.draw.rect(screen, CYAN, (player_x * GRID_SIZE, player_y * GRID_SIZE, GRID_SIZE, GRID_SIZE))

  # Draw the side screen
  pygame.draw.rect(screen, (128, 128, 128), (WIDTH - SIDE_SCREEN_WIDTH, 0, SIDE_SCREEN_WIDTH, HEIGHT))
  
  text_surface = font.render(f"Units traveled: {float('%.2f' % (units_traveled))}", True, (255, 255, 255))
  screen.blit(text_surface, (WIDTH - SIDE_SCREEN_WIDTH + 10, 10))

  text_surface = font.render(f"Error: {float('%.2f' % (error))}", True, (255, 255, 255))
  screen.blit(text_surface, (WIDTH - SIDE_SCREEN_WIDTH + 10, font_size + 10 + 10))

  text_surface = font.render(f"Rotations made: {float('%.2f' % (rotation_accumulator))}", True, (255, 255, 255))
  screen.blit(text_surface, (WIDTH - SIDE_SCREEN_WIDTH + 10, 2*font_size + 2*10 + 10))

  pygame.display.update()

def calculate_rotation(vector1, vector2):
  dot_product = np.dot(vector1, vector2)

  magnitude1 = np.linalg.norm(vector1)
  magnitude2 = np.linalg.norm(vector2)
  cosine_theta = dot_product / (magnitude1 * magnitude2)
  cosine_theta = max(min(cosine_theta, 1), -1) 
  angle_radians = np.arccos(cosine_theta)

  return np.degrees(angle_radians)

def can_move(direction):
  if direction == Direction.Up:
    return player_y > 0 and not grid[player_y-1][player_x] == RED
  elif direction == Direction.Down:
    return player_y < GRID_HEIGHT - 1 and not grid[player_y+1][player_x] == RED
  elif direction == Direction.Left:
    return player_x > 0 and not grid[player_y][player_x-1] == RED
  elif direction == Direction.Right:
    return player_x < GRID_WIDTH - 1 and not grid[player_y][player_x+1] == RED
  elif direction == Direction.UL:
    return can_move(Direction.Up) and can_move(Direction.Left) and not grid[player_y-1][player_x-1] == RED
  elif direction == Direction.LD:
    return can_move(Direction.Left) and can_move(Direction.Down) and not grid[player_y+1][player_x-1] == RED
  elif direction == Direction.DR:
    return can_move(Direction.Down) and can_move(Direction.Right) and not grid[player_y+1][player_x+1] == RED
  elif direction == Direction.RU:
    return can_move(Direction.Right) and can_move(Direction.Up) and not grid[player_y-1][player_x+1] == RED
  
def get_orentation_vector(direction):
  if direction == Direction.Up:
    return np.array([0, 1])
  elif direction == Direction.Down:
    return np.array([0, -1])
  elif direction == Direction.Left:
    return np.array([-1, 0])
  elif direction == Direction.Right:
    return np.array([1, 0])
  elif direction == Direction.UL:
    return np.array([-0.5**0.5, 0.5**0.5])
  elif direction == Direction.LD:
    return np.array([-0.5**0.5, -0.5**0.5])
  elif direction == Direction.DR:
    return np.array([0.5**0.5, -0.5**0.5])
  elif direction == Direction.RU:
    return np.array([0.5**0.5, 0.5**0.5])
  
  raise ValueError("No such direction")

def manual_control(keys):
  global player_y, player_x, units_traveled, error

  if keys[pygame.K_UP] or keys[pygame.K_8] or keys[pygame.K_KP_8]:
    Update(Direction.Up)
  elif keys[pygame.K_DOWN] or keys[pygame.K_2] or keys[pygame.K_KP_2]:
    Update(Direction.Down)
  elif keys[pygame.K_LEFT] or keys[pygame.K_4] or keys[pygame.K_KP_4]:
    Update(Direction.Left)
  elif keys[pygame.K_RIGHT] or keys[pygame.K_6] or keys[pygame.K_KP_6]:
    Update(Direction.Right)
  elif keys[pygame.K_7] or keys[pygame.K_KP_7]:
    Update(Direction.UL)
  elif keys[pygame.K_1] or keys[pygame.K_KP_1]:
    Update(Direction.LD)
  elif keys[pygame.K_3] or keys[pygame.K_KP_3]:
    Update(Direction.DR)
  elif keys[pygame.K_9] or keys[pygame.K_KP_9]:
    Update(Direction.RU)

def move(direction):
  global player_y, player_x

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
  else:
     return

def Update(direction):
  global units_traveled, error, rotation_accumulator, current_direction

  if not can_move(direction):
     return

  move(direction)

  rotation_accumulator += calculate_rotation(get_orentation_vector(current_direction), get_orentation_vector(direction))
  current_direction = direction
     
  if grid[player_y][player_x] == BLACK:
      grid[player_y][player_x] = WHITE
  else:
    error += 1 if not direction in [Direction.UL, Direction.LD, Direction.DR, Direction.RU] else 1.414
  
  units_traveled += 1 if not direction in [Direction.UL, Direction.LD, Direction.DR, Direction.RU] else 1.414

pygame.init()

class Direction(Enum):
  Up = 0
  Left = 1
  Down = 2
  Right = 3
  UL = 4
  LD = 5
  DR = 6
  RU = 7

####### Add algorithms in this enumerator ##############
class Algorithm(Enum):
  Manual = 0  # control player with numpad
  Random = 1
######################################################## 

################### Adjust as needed ###################
GRID_WIDTH = 10
GRID_HEIGHT = 10
GRID_SIZE = 30 # size in pixels
########################################################

SIDE_SCREEN_WIDTH = 300
WIDTH, HEIGHT = GRID_WIDTH * GRID_SIZE + SIDE_SCREEN_WIDTH, GRID_HEIGHT * GRID_SIZE

CYAN = (0, 255, 255)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
font_size = 36
font = pygame.font.Font(None, font_size)
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Waka waka")

############## Initialize the player's position, we can make more tests using same map but different starting position ####
player_x = 0 # random.randint(0, GRID_WIDTH - 1)
player_y = 0 # random.randint(0, GRID_HEIGHT - 1)
###########################################################################################

grid = [[BLACK for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]

############## CREATE OBSTACLE MAPS HERE, we'll need like 3 I guess #######################
obstacles_1 = []
obstacles_2 = [(3, 5), (4, 5), (4, 6), (5, 6)] # [(y, x), ...]

obstacle_map = obstacles_2
###########################################################################################

for obstacle in obstacle_map:
    grid[obstacle[0]][obstacle[1]] = RED

grid[player_y][player_x] = WHITE
units_traveled = 0
error = 0
rotation_accumulator = 0
current_direction = Direction.Right

def Start():
  key_pressed = False
  running = True
  while running and any(BLACK in row for row in grid):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            key_pressed = True          

    if key_pressed:
        keys = pygame.key.get_pressed()
        if keys[pygame.K_q]:
          break

        if algorithm == Algorithm.Manual:
          manual_control(keys)
        
        key_pressed = False

    # add new algorythms here
    if algorithm == Algorithm.Random:
      Update(Direction(random.randint(0, 7)))

    Render()

# change here to test your algorithm
algorithm = Algorithm.Random
Start()
print(f"Distance traveled {units_traveled} units, error {error} units, total rotation {rotation_accumulator} deg")
pygame.quit()
