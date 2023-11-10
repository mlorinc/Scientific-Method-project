from enum import Enum

####### Add algorithms in this enumerator ##############
class Algorithm(Enum):
  Manual = 0  # control player with numpad
  Random = 1
  SemiRandom = 2
  AStartRandom = 3
  AStar = 4
  AStarSequential = 5
######################################################## 


############ Add maps in this enumerator ###############
class Map(Enum):
   Empty = 0
   Room = 1
   Maze = 2
   Crazy = 3
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
