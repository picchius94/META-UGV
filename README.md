# Deep Meta-Learning Energy-Aware Path Planner for Mobile Robots in Unknown Environments
This repo is the adaptive path planner implementation of the paper: *"Deep Meta-Learning Energy-Aware Path Planner for Mobile Robots in Uknown Environments", Visca et al., 2021*.

## Effect of Terrain Transition Experiment
In this experiment, the adaptation performance of the adaptive meta-learning path planner are tested, when the robot moves on a new terrain.
Entry point for the experiment is `planning_simulation_trans_performance.py`.
All the entries of the dictionary `params` can be changed to vary terrain type, map size, initial robot position, etc..

## Effect of Heuristic Function Experiment
In this experiment, the effect of different heuristic functions for the meta-adaptive path planner are tested.
Entry point for the experiment is `planning_simulation_h_performance.py`.
All the entries of the dictionary `params` can be changed to vary terrain type, map size, initial robot position, etc..


### Neural Network Models
The meta-learning, and standard supervised learning models are given in `models.py`
The networks have been pretrained, and the weights are provided in the `log` folders.

### Dependencies
The following dependencies are required:
- numpy
- matplotlib
- pandas
- pychrono
- tensorflow
- opensimplex


