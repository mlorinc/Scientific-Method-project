import pygame
import random
from enum import Enum
import numpy as np
from queue import PriorityQueue

pygame.init()

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

def square_in_sight(direction, y, x):
  if direction == Direction.Up:
    return grid[y-1][x]
  elif direction == Direction.Down:
    return grid[y+1][x]
  elif direction == Direction.Left:
    return grid[y][x-1]
  elif direction == Direction.Right:
    return grid[y][x+1]
  elif direction == Direction.UL:
    return grid[y-1][x-1]
  elif direction == Direction.LD:
    return grid[y+1][x-1]
  elif direction == Direction.DR:
    return grid[y+1][x+1]
  elif direction == Direction.RU:
    return grid[y-1][x+1]

def can_move(direction, y, x):
  if direction == Direction.Up and not y > 0:
    return False
  elif direction == Direction.Down and not y < GRID_HEIGHT - 1:
    return False
  elif direction == Direction.Left and not x > 0:
    return False
  elif direction == Direction.Right and not x < GRID_WIDTH - 1:
    return False
  elif direction == Direction.UL:
    return can_move(Direction.Up, y, x) and can_move(Direction.Left, y, x)
  elif direction == Direction.LD:
    return can_move(Direction.Left, y, x) and can_move(Direction.Down, y, x)
  elif direction == Direction.DR:
    return can_move(Direction.Down, y, x) and can_move(Direction.Right, y, x)
  elif direction == Direction.RU:
    return can_move(Direction.Right, y, x) and can_move(Direction.Up, y, x)
  
  if square_in_sight(direction, y, x) == RED:
    return False
  
  return True
  
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

  return (player_y, player_x)


def Update(direction):
  global units_traveled, error, rotation_accumulator, current_direction, player_x, player_y
  if not can_move(direction, player_y, player_x):
     return

  dy, dx = move(direction)
  player_x += dx
  player_y += dy

  rotation_accumulator += calculate_rotation(get_orentation_vector(current_direction), get_orentation_vector(direction))
  current_direction = direction
     
  if grid[player_y][player_x] == BLACK:
      grid[player_y][player_x] = WHITE
  else:
    error += 1 if not direction in [Direction.UL, Direction.LD, Direction.DR, Direction.RU] else 1.414
  
  units_traveled += 1 if not direction in [Direction.UL, Direction.LD, Direction.DR, Direction.RU] else 1.414
  
def ReverseDir(direction):
  if direction == Direction.Up:
    return Direction.Down
  elif direction == Direction.Down:
    return Direction.Up
  elif direction == Direction.Left:
    return Direction.Right
  elif direction == Direction.Right:
    return Direction.Left
  elif direction == Direction.UL:
    return Direction.DR
  elif direction == Direction.LD:
    return Direction.RU
  elif direction == Direction.DR:
    return Direction.UL
  elif direction == Direction.RU:
    return Direction.LD

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
  AStar = 2
  AStarSequential = 3
  SemiRandom = 4
  AStartRandom = 5
######################################################## 

################### Adjust as needed ###################
GRID_WIDTH = 30
GRID_HEIGHT = 30
GRID_SIZE = 20 # size in pixels
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

grid = [[BLACK for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]

############## CREATE OBSTACLE MAPS HERE, we'll need like 3 I guess #######################
obstacle_tv = [(4,0),(3,0)]
obstacles_sofa = [(3,3), (4,3), (5,3), (3, 5), (4, 5), (6,3), (6,4), (6,5), (6,6), (5,5), (3, 6), (4, 6), (5, 6), (3, 4), (4, 4), (5, 4), (2,3), (2,4), (2,5), (2,6), (1,3), (1,4), (1,5), (1,6)]
obstacles_2 = [(10, 11), (10, 12), (10, 13), (10, 14), (11, 11), (12, 11), (13, 11), (13, 12), (13, 13), (11, 14), (12, 15)]
obstacle_kitchenwall = [(0, 19), (0, 20), (0,21), (0,22), (0,23), (0,24), (0,25), (0,26), (0,27), (0,28), (0,29), (1, 19), (1, 20), (1,21), (1,22), (1,23), (1,24), (1,25), (1,26), (1,27), (1,28), (1,29), (2,28), (2,29), (3,28), (3,29), (4,28), (4,29), (5,28), (5,29), (6,28), (6,29), (7,28), (7,29)]
obstacle_kitchenisland = [(3,19), (3,20), (4,19), (4,20), (5,19), (5,20), (6,19), (6,20), (7,19), (7,20), (7,21), (6,21), (7,22), (6,22), (7,23), (6,23), (7,24), (6,24), (7,25), (6,25), (7,26), (6,26)]
obstacle_table = [(18,17), (18,18), (18,19), (18,20), (18,21), (18,22), (19,21), (19,22), (19,17), (19,18), (19,19), (19,20), (20,21), (20,22), (20,17), (20,18), (20,19), (20,20)]
obstacle_chairs = [(16, 18), (16, 21), (22,18), (22,21)]
obstacle_bed = [(29,5), (29,6), (29,7), (29,8), (29,9), (29,10), (29,11), (29,12), (29,13), (28,5), (28,6), (28,7), (28,8), (28,9), (28,10), (28,11), (28,12), (28,13), (27,5), (27,6), (27,7), (27,8), (27,9), (27,10), (27,11), (27,12), (27,13), (26,5), (26,6), (26,7), (26,8), (26,9), (26,10), (26,11), (26,12), (26,13), (25,5), (25,6), (25,7), (25,8), (25,9), (25,10), (25,11), (25,12), (25,13)]
obstacle_cubboard2 = [(12,0), (13,0), (14,0), (15,0), (16,0), (12, 1), (13, 1), (14, 1), (15,1), (16,1), (12,2), (13,2), (14,2), (15,2), (16,2)]

obstacle_maze = [(3, 3), (3, 4), (3,5), (3,6), (3,7), (3,8), (3,9), (3,10), (3,11), (3,12), (3,13), (3,14), (3,15), (3,16), (3,17), (3,18), (3,19), (3,20), (3,21), (3,22), (3,23), (3,24), (3,25), (3,26),
                (4,26), (5,26), (6,26), (7,26), (8,26), (9,26), (10,26), (11,26), (12,26), (13,26), (14,26), (15,26), (16,26), (17,26), (18,26), (19,26), (20,26), (21,26), (22,26), (23,26), (24,26), (25,26), (26,26),
                (26,25), (26,24), (26,23), (26,22), (26,21), (26,20), (26,19), (26,18), (26,17), (26,16), (26,15), (26,14), (26,13), (26,12), (26,11), (26,10), (26,9), (26,8), (26,7), (26,6), (26,5), (26,4), (26,3),
                (25,3 ), (24,3), (23,3), (22,3), (21,3), (20,3), (19,3), (18,3), (17,3), (16,3), (15,3), (14,3), (13,3), (12,3), (11,3), (10,3), (9,3), (8,3), (7,3),
                (7,4), (7,5), (7,6), (7,7), (7,8), (7,9), (7,10), (7,11), (7,12), (7,13), (7,14), (7,15), (7,16), (7,17), (7,18), (7,19), (7,20), (7,21), (7,22),
                (8,22), (9,22), (10,22), (11,22), (12,22), (13,22), (14,22), (15,22), (16,22), (17,22), (18,22), (19,22), (20,22), (21,22), (22,22),
                (22,21), (22,20), (22,19), (22,18), (22,17), (22,16), (22,15), (22,14), (22,13), (22,12), (22,11), (22,10), (22,9), (22,8), (22,7),
                (21, 7), (20,7), (19,7), (18,7), (17,7), (16,7), (15,7), (14,7), (13,7), (12,7), (11,7), 
                (11,8), (11,9), (11,10), (11,11), (11,12), (11,13), (11,14), (11,15), (11,16), (11,17), (11,18),
                (12,18), (13,18), (14,18), (15,18), (16,18), (17,18), (18,18),
                (18,17), (18,16), (18,15), (18,14), (18,13), (18,12), (18,11),
                (17,11), (16,11), (15,11),
                (15,12), (15,13), (15,14),
                (16,14)]

#obstacle_map = obstacles_sofa + obstacle_table + obstacle_bed + obstacle_kitchenwall + obstacle_cubboard2 + obstacle_chairs + obstacle_tv + obstacle_kitchenisland
obstacle_map = obstacle_maze

###########################################################################################

for obstacle in obstacle_map:
    grid[obstacle[0]][obstacle[1]] = RED
    
############## Initialize the player's position, we can make more tests using same map but different starting position ####
player_x = random.randint(0, GRID_WIDTH - 1)
player_y = random.randint(0, GRID_HEIGHT - 1)

while grid[player_y][player_x] == RED:
  player_x = random.randint(0, GRID_WIDTH - 1)
  player_y = random.randint(0, GRID_HEIGHT - 1)
###########################################################################################

grid[player_y][player_x] = WHITE
units_traveled = 0
error = 0
rotation_accumulator = 0
current_direction = Direction.Right
spiral_state = 0



#################################### A* Algorithm #########################################
# Modify the heuristic to return 0 for black cells to encourage their exploration
def get_movement_array(direction = Direction.Up):
  directions = [Direction.Up, Direction.RU, Direction.Right, Direction.DR, Direction.Down, Direction.LD, Direction.Left, Direction.UL]
  index = directions.index(direction)
  
  return directions[index:] + directions[:index]
  
  

def heuristic(a, b):
    # Use the Manhattan distance heuristic
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# Function to determine the closest black cell using Breadth-First Search (BFS)
def get_closest_black_cell(start, direction):
    queue = [start]
    visited = set()
    movement_array = get_movement_array(direction)

    # Iterate through the grid using BFS until a black cell is found
    while queue:
        current = queue.pop(0)
        if grid[current[0]][current[1]] == BLACK:
            return current
        visited.add(current)
        for direction in movement_array:
          if not can_move(direction, current[0], current[1]):
            continue

          dy, dx = move(direction)
          row, col = current[0] + dy, current[1] + dx
          if(row, col) not in visited:
            queue.append((row, col))
            visited.add((row, col) )
    return None
  
def get_closest_black_cell_sequential(start):
    queue = [start]
    visited = set()
    
    movement_array = get_movement_array()

    # Iterate through the grid using BFS until a black cell is found
    while queue:
        current = queue.pop(0)
        if grid[current[0]][current[1]] == BLACK:
            return current
        visited.add(current)
        for direction in movement_array:
          if not can_move(direction, current[0], current[1]):
            continue

          dy, dx = move(direction)
          row, col = current[0] + dy, current[1] + dx
          if(row, col) not in visited:
            queue.append((row, col))
            visited.add((row, col) )
    return None
  
def get_closest_black_cell_random():
  black_indexes = [(y, x) for y, row in enumerate(grid) for x, val in enumerate(row) if val == BLACK]

  if not black_indexes:
      return None

  return random.choice(black_indexes)

def astar(start, goal):
    open_set = PriorityQueue()
    open_set.put((0, start))
    came_from = {}
    g_score = {spot: float("inf") for row in grid for spot in row}
    g_score[start] = 0
    
    movement_array = get_movement_array()
    
    # Uncomment if you dont want to use diagonals
    # movement_array = [Direction.Up, Direction.Right, Direction.Down, Direction.Left]
  
    while not open_set.empty():
        current = open_set.get()[1]  # Get the node with the lowest total cost (f-score)

        if current == goal:
            # Reconstruct the path from goal to start by traversing the 'came_from' chain
            path = []
            while current in came_from:
                path.append(came_from[current][1])
                current = came_from[current][0]

            return path

        # Explore the neighboring nodes in all directions
        for direction in movement_array:
            if not can_move(direction, current[0], current[1]):
              continue

            dy, dx = move(direction)
            row, col = current[0] + dy, current[1] + dx

            tentative_g_score = g_score[current] + 1 if direction in [Direction.Up, Direction.Right, Direction.Left, Direction.Down] else g_score[current] + 1.44

            # Update the best path if this node provides a shorter path to the goal
            if tentative_g_score < g_score.get((row, col), float("inf")):
                came_from[(row, col)] = (current, direction) 
                g_score[(row, col)] = tentative_g_score 
                f_score = tentative_g_score + heuristic(goal, (row, col))
                open_set.put((f_score, (row, col))) 
    return [] 


# Using the BFS-based closest black cell logic in the pathfinding algorithm
def navigate_to_closest_black_cell(start, direction):
  global algorithm

  closest_black_cell = None
  if algorithm == Algorithm.AStar:
    closest_black_cell = get_closest_black_cell(start, direction)
  elif algorithm == Algorithm.AStarSequential:
    closest_black_cell = get_closest_black_cell_sequential(start)
  elif algorithm == Algorithm.AStartRandom:
    closest_black_cell = get_closest_black_cell_random()

  if closest_black_cell:
      return astar(start, closest_black_cell)
  
  return None
###########################################################################################


################################ Semi random algorithm ####################################
def semi_random():
  dir_array = get_movement_array(Direction.Up)
  valid_dir_array = [dir for dir in dir_array if can_move(dir, player_y, player_x)]
  neighbourhood_black_cells = [dir for dir in valid_dir_array if square_in_sight(dir, player_y, player_x) == BLACK]
  
  dir = None
  if neighbourhood_black_cells:
      dir = random.choice(neighbourhood_black_cells)
  else:
      dir = random.choice(valid_dir_array)

  Update(dir)
###########################################################################################



def Start():
    key_pressed = False
    running = True
    path = []

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

        # add new algorithms here
        if algorithm == Algorithm.Random:
            Update(Direction(random.randint(0, 7)))
        elif algorithm in [Algorithm.AStar, Algorithm.AStarSequential, Algorithm.AStartRandom]:
            if not path:
                # Generate a new path
                path = navigate_to_closest_black_cell((player_y, player_x), current_direction)

            if path:
                next_step = path.pop()
                Update(next_step)

        elif algorithm == Algorithm.SemiRandom:
          semi_random()

        Render()


# change here to test your algorithm
algorithm = Algorithm.AStarSequential
Start()
print(f"Distance traveled {units_traveled} units, error {error} units, total rotation {rotation_accumulator} deg")

# running_display = True
# while running_display:
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             running_display = False
pygame.quit()
