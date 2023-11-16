# Scientific-Method-project

Step 1. Develop different full area exploration algorithms (here)

Step 2. Convert path traveled in simulation to path for the real robot and see how long it takes to drive that path

Step 3. Analyse the data and make a scientific report

# Install stats dependencies

conda create --name scm --file conda_requirements.txt

# Title: 
# Abstract


This is the implementation of a comparable analysis of a scientific project, which was conducted in November 2023 at the University of Southern Denmark (SDU) in Odense in the lecture Scientific Methods.



The study involved the implementation of a simulation resembling a grid-based game with dimensions of 30x30 cells. 
Initially, all cells were set to black, except for predefined red obstacles. Movement of a virtual "robot" resulted in the transformation of cells to white. 

This simulation aimed to emulate scenarios comparable to those encountered by a vacuum cleaner robot navigating a room with obstacles, represented as a grid. 
The primary objective was to optimize the robot's coverage of the entire area, minimizing the number of revisits to each grid cell. 



# Data
The data is in the following format:
{units_traveled},{error},{rotation_accumulator},{time_taken},{algorithm.value},{map.value}

The algorithm choices include:
0: Random
1: Semirandom
2: A* Random
3: A* Spiral
4: A* Sequential

The environmental maps comprised:
0: Empty
1: Room
2: Spiral maze
3: Complex maze

A description of these can be seen in the paper.





# This repository includes:
main.py:
    Main code that contains the implemented simulation as a pygame, all compared algorithms and the different maps.
    The code provides many comments that aim to help understanding the code. With this, it should be able to reproduce the results explained in the paper.

draw_obstacle_map.py:
    Code used to generate different environments. Here it is possible to select red grid cells with a mouse click. The output can be used in the main.py file to add a new obstacle_map.

requirements.txt:
    Contains the versions of the two libraries numpy and pygame that have been used.

ev3_path.txt:
    Textfile that contains the commands used to navigate the Lego Mindstorm robot EV3 and to test the movement of the virtual "robot" in the simulation.

data_0_0.txt, data_13_16.txt, data_23_24.txt:
    These data files are named by their respective starting position (data_row_column). The study utilized three distinct starting positions and subjected each algorithm to 25 test trials (with the exception of the random algorithm, which underwent testing only with the starting position (0,0) due to a heavy computation time). 

data_compare_astar.txt:
    These two algorithms were seperately compared with 25 different starting positions on every map (data_compare_astar.txt), because their results in the above files are the same for each test trial (because they don't use random movement).

# Plotting

` python ./stats.py graph:all . -e pdf`

# Hypothesis testing

`python ./stats.py hypothesis:generic . "A* Orientation" "A* Sequential" "Rotation accumulator"`

`python ./stats.py hypothesis:generic . "A* Orientation" "A* Random" "Rotation accumulator"`

`python ./stats.py hypothesis:generic . "A* Orientation" "A* Sequential" "Error"`

`python ./stats.py hypothesis:generic . "A* Orientation" "A* Sequential" "Time taken"`

