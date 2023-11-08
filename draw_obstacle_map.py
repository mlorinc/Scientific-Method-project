import pygame

# Define some colors
BLACK = (0, 0, 0)
RED = (255, 0, 0)
WHITE = (255, 255, 255)

# Initialize Pygame
pygame.init()

# Set the width and height of each grid square and the grid size
GRID_SIZE = 30
GRID_WIDTH = 30
GRID_HEIGHT = 30
WIDTH = GRID_SIZE * GRID_WIDTH
HEIGHT = GRID_SIZE * GRID_HEIGHT

# Create the screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Grid Example")

# Create a 2D array representing the grid
grid_array = [[BLACK for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]  # Initialize with zeros (BLACK)

# Create a list to store the indices of red squares
red_indices = set([(2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (2, 9), (2, 10), (2, 11), (2, 12), (2, 13), (2, 14), (2, 15), (2, 16), (2, 17), (2, 18), (2, 19), (2, 20), (2, 21), (2, 22), (2, 24), (2, 25), (2, 26), (2, 27), (3, 2), (3, 20), (3, 24), (3, 27), (4, 2), (4, 4), (4, 10), (4, 12), (4, 14), (4, 20), (4, 22), (4, 23), (4, 24), (4, 27), (5, 2), (5, 7), (5, 8), (5, 10), (5, 12), (5, 14), (5, 16), (5, 17), (5, 18), (5, 20), (5, 22), (5, 27), (6, 2), (6, 4), (6, 5), (6, 6), (6, 7), (6, 10), (6, 12), (6, 14), (6, 20), (6, 22), (6, 23), (6, 24), (6, 25), (6, 26), (6, 27), (7, 2), (7, 4), (7, 7), (7, 9), (7, 10), (7, 12), (7, 14), (7, 16), (7, 17), (7, 18), (7, 20), (7, 27), (8, 2), (8, 4), (8, 5), (8, 7), (8, 12), (8, 14), (8, 20), (8, 22), (8, 23), (8, 25), (8, 27), (9, 2), (9, 4), (9, 7), (9, 8), (9, 9), (9, 10), (9, 23), (9, 27), (10, 2), (10, 4), (10, 9), (10, 10), (10, 11), (10, 12), (10, 13), (10, 14), (10, 15), (10, 16), (10, 17), (10, 18), (10, 19), (10, 20), (10, 21), (10, 22), (10, 23), (10, 24), (10, 25), (10, 27), (11, 2), (11, 4), (11, 6), (11, 7), (11, 9), (11, 15), (11, 17), (11, 21), (11, 25), (11, 27), (12, 2), (12, 4), (12, 7), (12, 12), (12, 17), (12, 19), (12, 21), (12, 23), (12, 25), (12, 27), (13, 2), (13, 4), (13, 5), (13, 7), (13, 8), (13, 9), (13, 10), (13, 11), (13, 12), (13, 13), (13, 14), (13, 16), (13, 17), (13, 19), (13, 21), (13, 27), (14, 2), (14, 4), (14, 7), (14, 14), (14, 17), (14, 19), (14, 21), (14, 22), (14, 23), (14, 24), (14, 25), (14, 27), (15, 2), (15, 4), (15, 6), (15, 7), (15, 9), (15, 10), (15, 11), (15, 12), (15, 13), (15, 14), (15, 15), (15, 19), (15, 25), (15, 27), (16, 2), (16, 4), (16, 15), (16, 16), (16, 17), (16, 23), (16, 25), (16, 27), (17, 2), (17, 4), (17, 5), (17, 6), (17, 7), (17, 8), (17, 10), (17, 11), (17, 12), (17, 13), (17, 15), (17, 17), (17, 18), (17, 19), (17, 21), (17, 25), (17, 27), (18, 2), (18, 4), (18, 8), (18, 10), (18, 15), (18, 21), (18, 22), (18, 23), (18, 24), (18, 25), (18, 27), (19, 2), (19, 4), (19, 6), (19, 8), (19, 10), (19, 12), (19, 13), (19, 14), (19, 15), (19, 17), (19, 18), (19, 21), (19, 27), (20, 2), (20, 4), (20, 6), (20, 8), (20, 9), (20, 10), (20, 14), (20, 18), (20, 21), (20, 23), (20, 24), (20, 25), (20, 26), (20, 27), (21, 2), (21, 4), (21, 6), (21, 12), (21, 14), (21, 16), (21, 17), (21, 18), (21, 21), (21, 27), (22, 2), (22, 4), (22, 6), (22, 7), (22, 8), (22, 9), (22, 10), (22, 12), (22, 14), (22, 18), (22, 21), (22, 22), (22, 23), (22, 24), (22, 25), (22, 27), (23, 2), (23, 6), (23, 10), (23, 11), (23, 12), (23, 13), (23, 14), (23, 15), (23, 16), (23, 18), (23, 25), (23, 27), (24, 2), (24, 4), (24, 5), (24, 6), (24, 7), (24, 8), (24, 12), (24, 13), (24, 18), (24, 20), (24, 21), (24, 22), (24, 25), (24, 27), (25, 2), (25, 4), (25, 8), (25, 10), (25, 13), (25, 15), (25, 16), (25, 17), (25, 18), (25, 20), (25, 22), (25, 23), (25, 24), (25, 25), (25, 27), (26, 2), (26, 6), (26, 10), (26, 11), (26, 13), (26, 15), (26, 20), (26, 27), (27, 2), (27, 3), (27, 4), (27, 5), (27, 6), (27, 7), (27, 8), (27, 9), (27, 10), (27, 11), (27, 12), (27, 13), (27, 15), (27, 16), (27, 17), (27, 18), (27, 19), (27, 20), (27, 21), (27, 22), (27, 23), (27, 24), (27, 25), (27, 26), (27, 27)])

for (y, x) in red_indices:
  grid_array[y][x] = RED

# Variables to track dragging
drawing = False

# Run the game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            x, y = event.pos
            grid_x = x // GRID_SIZE
            grid_y = y // GRID_SIZE
            if event.button == 1:  # Left mouse button
                drawing = True
                if 0 <= grid_x < GRID_WIDTH and 0 <= grid_y < GRID_HEIGHT:
                    grid_array[grid_y][grid_x] = RED  # Change the value to represent RED
                    red_indices.add((grid_y, grid_x))  # Append the red square indices to the list
            elif event.button == 3:  # Right mouse button
                drawing = True
                if 0 <= grid_x < GRID_WIDTH and 0 <= grid_y < GRID_HEIGHT:
                    grid_array[grid_y][grid_x] = BLACK  # Change the value back to represent BLACK
                    if (grid_y, grid_x) in red_indices:
                        red_indices.remove((grid_y, grid_x))  # Remove the indices from the red squares list
        elif event.type == pygame.MOUSEBUTTONUP:
            drawing = False
        elif event.type == pygame.MOUSEMOTION and drawing:
            x, y = event.pos
            grid_x = x // GRID_SIZE
            grid_y = y // GRID_SIZE
            if event.buttons[0]:  # Left mouse button
                if 0 <= grid_x < GRID_WIDTH and 0 <= grid_y < GRID_HEIGHT:
                    grid_array[grid_y][grid_x] = RED  # Change the value to represent RED
                    red_indices.add((grid_y, grid_x))  # Append the red square indices to the list
            elif event.buttons[2]:  # Right mouse button
                if 0 <= grid_x < GRID_WIDTH and 0 <= grid_y < GRID_HEIGHT:
                    grid_array[grid_y][grid_x] = BLACK  # Change the value back to represent BLACK
                    if (grid_y, grid_x) in red_indices:
                        red_indices.remove((grid_y, grid_x))  # Remove the indices from the red squares list

    # Fill the background with WHITE
    screen.fill(WHITE)

    # Draw the grid
    for y in range(GRID_HEIGHT):
      for x in range(GRID_WIDTH):
          pygame.draw.rect(screen, grid_array[y][x], (x * GRID_SIZE, y * GRID_SIZE, GRID_SIZE, GRID_SIZE))

    # Update the display
    pygame.display.flip()

# Print the indices of red squares after exiting the loop
print(sorted(red_indices))

# Quit Pygame
pygame.quit()
