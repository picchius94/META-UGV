# Deep Meta-Learning Energy-Aware Path Planner for Unmanned Ground Vehicles in Unknown Terrain
This is the implementation of the paper: *"Deep Meta-Learning Energy-Aware Path Planner for Unmanned Ground Vehicles in Unknown Terrain", Visca et al., 2021*.

<img src="https://github.com/picchius94/META-UGV/blob/main/Images/transition.gif" width="270"> <img src="https://github.com/picchius94/META-UGV/blob/main/Images/transition2.gif" width="270"> <img src="https://github.com/picchius94/META-UGV/blob/main/Images/transition3.gif" width="270">


1. ## Experiments
### Effect of Terrain Transition
In this experiment, the performance of the meta-adaptive path planner are tested, when the vehicle transitions on a new terrain.

Run `planning_simulation_trans_performance.py`.

All the entries of the dictionary `params` can be changed to modify terrain type, map size, initial vehicle position, etc..

### Effect of Heuristic Function
In this experiment, the effect of different heuristic functions for the meta-adaptive path planner are tested.

Run `planning_simulation_h_performance.py`.

All the entries of the dictionary `params` can be changed to modify terrain type, map size, initial vehicle position, etc..

### Note!
Line 37 in `my_chrono_simulator.py` must be changed with the correct local path to the Chrono Data directory.




1. ## Miscellaneous
### Terrain Types and SCM Parameters
Deformable terrains are modelled using the Project Chrono [[1]](#1) implementation of the Soil Contact Model (SCM) [[2]](#2). The complete list of implemented terrain types and respective terramechanical parameters is given in `terrain_list.py`.

<p align="center">
<img src="https://github.com/picchius94/META-UGV/blob/main/Images/terrain_types.png" width="700">
</p>

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




## References
<a id="1">[1]</a> 
A. Tasora, R. Serban, H. Mazhar, A. Pazouki, D. Melanz, J. Fleischmann, M. Taylor, H. Sugiyama, and D. Negrut, “Chrono: An open source multi-physics dynamics engine,” in International Conference on High Performance Computing in Science and Engineering. Springer, 2015, pp. 19–49.

<a id="2">[2]</a>
F. Buse, R. Lichtenheldt, and R. Krenn, “Scm-a novel approach for soil deformation in a modular soil contact model for multibody simulation”, IMSD2016 e-Proceedings, 2016.
