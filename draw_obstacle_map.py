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
grid_array = [[0 for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]  # Initialize with zeros (BLACK)

# Create a list to store the indices of red squares
red_indices = []

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
                    grid_array[grid_y][grid_x] = 1  # Change the value to represent RED
                    red_indices.append((grid_y, grid_x))  # Append the red square indices to the list
            elif event.button == 3:  # Right mouse button
                drawing = True
                if 0 <= grid_x < GRID_WIDTH and 0 <= grid_y < GRID_HEIGHT:
                    grid_array[grid_y][grid_x] = 0  # Change the value back to represent BLACK
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
                    grid_array[grid_y][grid_x] = 1  # Change the value to represent RED
                    red_indices.append((grid_y, grid_x))  # Append the red square indices to the list
            elif event.buttons[2]:  # Right mouse button
                if 0 <= grid_x < GRID_WIDTH and 0 <= grid_y < GRID_HEIGHT:
                    grid_array[grid_y][grid_x] = 0  # Change the value back to represent BLACK
                    if (grid_y, grid_x) in red_indices:
                        red_indices.remove((grid_y, grid_x))  # Remove the indices from the red squares list

    # Fill the background with WHITE
    screen.fill(WHITE)

    # Draw the grid
    for y in range(GRID_HEIGHT):
        for x in range(GRID_WIDTH):
            color = RED if grid_array[y][x] == 1 else BLACK
            pygame.draw.rect(screen, color, (x * GRID_SIZE, y * GRID_SIZE, GRID_SIZE, GRID_SIZE))

    # Update the display
    pygame.display.flip()

# Print the indices of red squares after exiting the loop
print(red_indices)

# Quit Pygame
pygame.quit()
