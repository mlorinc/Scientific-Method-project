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
  SemiRandom = 3
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

############## Initialize the player's position, we can make more tests using same map but different starting position ####
player_x = 0 # random.randint(0, GRID_WIDTH - 1)
player_y = 0 # random.randint(0, GRID_HEIGHT - 1)
###########################################################################################

grid = [[BLACK for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]

############## CREATE OBSTACLE MAPS HERE, we'll need like 3 I guess #######################
obstacles_1 = []
obstacles_2 = [(3, 5), (4, 5), (4, 6), (5, 6)] # [(y, x), ...]
obstacles_3 = [(3, 5), (4, 5), (4, 6), (5, 6), (5, 5), (5, 3), (5, 4), (6, 4), (6, 5), (10, 11), (10, 12), (10, 13), (10, 14), (11, 11), (12, 11), (13, 11), (13, 12), (13, 13), (11, 14), (12, 15)]

obstacle_map = obstacles_3
###########################################################################################



for obstacle in obstacle_map:
    grid[obstacle[0]][obstacle[1]] = RED

grid[player_y][player_x] = WHITE
units_traveled = 0
error = 0
rotation_accumulator = 0
current_direction = Direction.Right
spiral_state = 0



#################################### A* Algorithm #########################################
# Modify the heuristic to return 0 for black cells to encourage their exploration
def heuristic(a, b):
    if grid[a[0]][a[1]] == BLACK:
        return 0
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# Function to determine the closest black cell using BFS
def get_closest_black_cell(start):
    queue = [start]
    visited = set()
    while queue:
        current = queue.pop(0)
        if grid[current[0]][current[1]] == BLACK:
            return current
        visited.add(current)
        for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            row, col = current[0] + dy, current[1] + dx
            if 0 <= row < len(grid) and 0 <= col < len(grid[0]) and grid[row][col] != RED and (row, col) not in visited:
                queue.append((row, col))
    return None

def astar(start, goal):
    # Initialize the priority queue to manage open set
    open_set = PriorityQueue()
    open_set.put((0, start))

    # Store parent nodes to reconstruct the path
    came_from = {}

    # Initialize scores for each spot in the grid
    g_score = {spot: float("inf") for row in grid for spot in row}
    g_score[start] = 0

    while not open_set.empty():
        current = open_set.get()[1]

        # If the goal is reached, reconstruct the path and return
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return path

        # Explore neighbors in four directions: right, down, left, up
        for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            row, col = current[0] + dy, current[1] + dx

            # Ensure the next step is within the grid and not an obstacle
            if 0 <= row < len(grid) and 0 <= col < len(grid[0]) and grid[row][col] != RED:
                # Calculate tentative g-score
                tentative_g_score = g_score[current] + 1

                # If the calculated g-score is less, update the g-score, add to came_from and open_set
                if tentative_g_score < g_score.get((row, col), float("inf")):
                    came_from[(row, col)] = current
                    g_score[(row, col)] = tentative_g_score
                    f_score = tentative_g_score + heuristic(goal, (row, col))
                    open_set.put((f_score, (row, col)))
    # If no path is found, return an empty list
    return []


# Using the BFS-based closest black cell logic in the pathfinding algorithm
def navigate_to_closest_black_cell(start):
    closest_black_cell = get_closest_black_cell(start)
    if closest_black_cell:
        return astar(start, closest_black_cell)
    return None
###########################################################################################




################################ Semi random algorithm ####################################
def semi_random():
    while BLACK in [cell for row in grid for cell in row]:
        # Get the adjacent cells for the current player position
        adj_cells = [
            (player_y - 1, player_x),  # Up
            (player_y + 1, player_x),  # Down
            (player_y, player_x - 1),  # Left
            (player_y, player_x + 1),  # Right
            (player_y - 1, player_x - 1),  # Upper Left
            (player_y + 1, player_x - 1),  # Lower Left
            (player_y + 1, player_x + 1),  # Lower Right
            (player_y - 1, player_x + 1),  # Upper Right
        ]

        # Filter out obstacles and out-of-bounds cells
        valid_adj_cells = [(y, x) for y, x in adj_cells if 0 <= y < GRID_HEIGHT and 0 <= x < GRID_WIDTH and grid[y][x] != RED]

        # Choose a black cell to move toward
        black_adj_cells = [(y, x) for y, x in valid_adj_cells if grid[y][x] == BLACK]
        if black_adj_cells:
            y, x = black_adj_cells[0]
        else:
            # If no black cells are adjacent, pick a random cell
            y, x = random.choice(valid_adj_cells)

        # Move the player towards the selected cell
        if y < player_y:
            Update(Direction.Up)
        elif y > player_y:
            Update(Direction.Down)
        elif x < player_x:
            Update(Direction.Left)
        elif x > player_x:
            Update(Direction.Right)
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
        elif algorithm == Algorithm.AStar:
            if not path:
                # Generate a new path
                goal = (random.randint(0, GRID_HEIGHT - 1), random.randint(0, GRID_WIDTH - 1))
                path = navigate_to_closest_black_cell((player_y, player_x))

            if path:
                next_step = path.pop()
                if next_step[0] < player_y:
                    Update(Direction.Up)
                elif next_step[0] > player_y:
                    Update(Direction.Down)
                elif next_step[1] < player_x:
                    Update(Direction.Left)
                elif next_step[1] > player_x:
                    Update(Direction.Right)

        elif algorithm == Algorithm.SemiRandom:
            semi_random()

        Render()


# change here to test your algorithm
algorithm = Algorithm.SemiRandom
Start()
print(f"Distance traveled {units_traveled} units, error {error} units, total rotation {rotation_accumulator} deg")

running_display = True
while running_display:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running_display = False
pygame.quit()
