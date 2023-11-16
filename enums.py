from enum import Enum

####### Add algorithms in this enumerator ##############
class Algorithm(Enum):
  Manual = 0  # control player with numpad
  Random = 1
  SemiRandom = 2
  AStarRandom = 3
  AStarSequential = 5
  AStarOrientation = 4
######################################################## 

# Human readable algorithm forms
algorithm_name_mapping = {
  Algorithm.Manual: "Manual",
  Algorithm.Random: "Random",
  Algorithm.SemiRandom: "Semi-random",
  Algorithm.AStarRandom: "A* Random",
  Algorithm.AStarOrientation: "A* Orientation",
  Algorithm.AStarSequential: "A* Sequential"
}

def get_algorithm_name(algo: Algorithm) -> str:
  return algorithm_name_mapping[algo]

algorithms = [
  Algorithm.Manual,
  Algorithm.Random,
  Algorithm.SemiRandom,
  Algorithm.AStarRandom,
  Algorithm.AStarOrientation,
  Algorithm.AStarSequential
]

############ Add maps in this enumerator ###############
class Map(Enum):
   Empty = 0
   Room = 1
   Spiral = 2
   Maze = 3
########################################################

class Direction(Enum):
  Up = 0
  Left = 1
  Down = 2
  Right = 3
  UL = 4
  LD = 5
  DR = 6
  RU = 7
