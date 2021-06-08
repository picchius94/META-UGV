# Deep Meta-Learning Energy-Aware Path Planner for Mobile Robots in Unknown Environments
This is the adaptive path planner implementation of the paper: *"Deep Meta-Learning Energy-Aware Path Planner for Mobile Robots in Uknown Environments", Visca et al., 2021*.

## Effect of Terrain Transition Experiment
In this experiment, the adaptation performance of the adaptive meta-learning path planner are tested, when the robot transitions on a new terrain.

Run `planning_simulation_trans_performance.py`.

All the entries of the dictionary `params` can be changed to modify terrain type, map size, initial robot position, etc..

## Effect of Heuristic Function Experiment
In this experiment, the effect of different heuristic functions for the meta-adaptive path planner are tested.

Run `planning_simulation_h_performance.py`.

All the entries of the dictionary `params` can be changed to modify terrain type, map size, initial robot position, etc..

## Miscellaneous
### Neural Network Models
The meta-learning, and separate supervised learning models are given in `models.py`
The networks have been pretrained, and the weights are provided in the `log` folders.

### Dependencies
The following dependencies are required:
- numpy
- matplotlib
- pandas
- pychrono
- tensorflow
- opensimplex


