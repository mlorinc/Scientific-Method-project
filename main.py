import pygame
import random
from enums import Algorithm, Map, Direction, TileState
from loader import dependent_variables, independent_variables
import numpy as np
from queue import PriorityQueue
import time
from game import Game
from maps import define_obstacles
import random_algorithm
import astar
import semi_random
from algorithm import AlgorithmBase
import pandas as pd
from typing import List, Tuple
import dqn_agent

def Start(start_position, algorithm, map, save = False):
    running = True

    rotational_speed = 180 # 180 deg/s
    velocity = 0.5 # 0.5 s/units

    game: Game = Game(rotation_velocity=rotational_speed, velocity=velocity)

    algorithm_map = {
      Algorithm.Random: random_algorithm.RandomAlgorith(game),
      Algorithm.SemiRandom: semi_random.SemiRandomAlgorith(game),
      Algorithm.AStarRandom: astar.AStarRandom(game),
      Algorithm.AStarSequential: astar.AStarSequential(game),
      Algorithm.AStarOrientation: astar.AStarOrientation(game)
    }

    ############ Load the selected obstacle map ############
    obstacle_map = []
    obstacle_room, obstacle_maze, obstacle_crazy, obstacle_smiley = define_obstacles()
    if map == Map.Empty:
      obstacle_map = []
    elif map == Map.Room:
      obstacle_map = obstacle_room
    elif map == Map.Spiral:
      obstacle_map = obstacle_maze
    elif map == Map.Maze:
      obstacle_map = obstacle_crazy
    elif map == Map.Smiley:
      obstacle_map = obstacle_smiley

    game.prepare_map((30, 30), start_position, obstacle_map, grid_dimension=(10, 10))
    game.render()
    # time.sleep(1)

    dqn_agent.create_agent(game)

    return game.data_to_tuple()

    algo: AlgorithmBase = algorithm_map[algorithm]

    while running and game.unvisited_count > 0:
        algo.execute()
        #time.sleep(0.25)
        game.render(lazy=True)

    time_taken = game.calculate_time_taken()
    print(f"Distance traveled {game.units_traveled} units, error {game.error} units, total rotation {game.rotation_accumulator} deg, time take {time_taken}s")

    if save:
      with open("ev3_path.txt", 'w') as file:
          file.write(str(game.ev3_path))
    ########################################################
    pygame.quit()
    return game.data_to_tuple()

#################### Loop to generate data for applying statistics ########################
def generate_data(algorithms: List[Algorithm], maps: List[Map], test_trials: int, positions: List[Tuple[int , int]]):
  data = []
  for algorithm in algorithms:
    for map in maps:
      for i in range(test_trials):
        position = positions[map][i]
        measurements = Start(position, algorithm, map)
        data.append(measurements)
  return pd.DataFrame(measurements)

def main():
  # algorithms = [Algorithm.AStarRandom, Algorithm.AStarSequential, Algorithm.AStarOrientation]
  algorithms = [Algorithm.AStarRandom]
  #algorithms = [Algorithm.SemiRandom, Algorithm.AStarRandom, Algorithm.AStar, Algorithm.AStarSequential]
  starting_positions = {
    Map.Empty: [(1, 0), (23, 27), (25, 3), (6, 19), (12, 14), (12, 27), (23, 7), (17, 16), (11, 3), (6, 11), (20, 17), (22, 5), (7, 0), (25, 23), (11, 1), (16, 6), (11, 25), (4, 27), (7, 16), (21, 7), (10, 20), (15, 15), (14, 5), (27, 3), (21, 6)],
    Map.Room: [(11, 1), (28, 21), (12, 21), (12, 29), (21, 23), (3, 25), (6, 11), (17, 1), (10, 5), (29, 2), (12, 26), (29, 22), (21, 0), (14, 12), (19, 7), (19, 10), (0, 14), (10, 16), (2, 10), (17, 21), (14, 22), (16, 28), (17, 2), (28, 24), (10, 8)],
    Map.Spiral: [(22, 23), (2, 28), (11, 0), (14, 0), (5, 2), (1, 28), (17, 9), (6, 29), (5, 15), (13, 17), (5, 2), (0, 22), (14, 24), (22, 0), (2, 15), (1, 15), (2, 5), (13, 20), (15, 15), (27, 20), (6, 7), (13, 20), (1, 10), (16, 8), (14, 4)],
    Map.Maze: [(26, 12), (29, 16), (3, 5), (16, 26), (3, 3), (23, 29), (1, 29), (28, 5), (29, 0), (28, 26), (21, 19), (18, 1), (3, 8), (22, 17), (12, 15), (11, 13), (26, 0), (10, 1), (6, 8), (28, 12), (17, 3), (17, 23), (14, 9), (15, 16), (24, 0)]
  }
  # maps = [Map.Empty, Map.Room, Map.Spiral, Map.Maze]
  maps = [Map.Room]
  test_trials = 1
  df = generate_data(algorithms, maps, test_trials, starting_positions)
  print(df)
  df.to_csv("test.csv")

if __name__ == "__main__":
  main()