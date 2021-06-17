# Deep Meta-Learning Energy-Aware Path Planner for Unmanned Ground Vehicles
This is the adaptive path planner implementation of the paper: *"Deep Meta-Learning Energy-Aware Path Planner for Unmanned Ground Vehicles", Visca et al., 2021*.

<img src="https://github.com/picchius94/META-UGV/blob/main/transition.gif" width="400"> <img src="https://github.com/picchius94/META-UGV/blob/main/transition2.gif" width="400">

## Effect of Terrain Transition Experiment
In this experiment, the performance of the meta-adaptive path planner are tested, when the vehicle transitions on a new terrain.

Run `planning_simulation_trans_performance.py`.

All the entries of the dictionary `params` can be changed to modify terrain type, map size, initial vehicle position, etc..

## Effect of Heuristic Function Experiment
In this experiment, the effect of different heuristic functions for the meta-adaptive path planner are tested.

Run `planning_simulation_h_performance.py`.

All the entries of the dictionary `params` can be changed to modify terrain type, map size, initial vehicle position, etc..

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


