# This version add shots only with 'fastest', each time the terrain type changes, to be valid for all methods
import os
import A_star
import terrain_generator as tg
import my_chrono_simulator as mcs
from PIL import Image
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as patches
from datetime import datetime
random.seed(9)

os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(0)

VISUALISATION = False

path_data = "./H_performance/"
current_time = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
path_experiment = path_data + "Exp_{}/".format(current_time)
path_maps = path_experiment + "Simplex_Maps/"

params = {}
params["Version"] = "path_planning_experiment_h_performance"
params["map_size_x"] = 22
params["map_size_y"] = 7
params["x0"] = -8
params["y0"] = 0
params["yaw0"] = 0

params["target_speed"] = 1

params["meta_h_scaling"] = 1
params["goal_distance"] = 10.8

params["MAX_TIMEOUT_FASTEST"] = 5
params["MAX_TIMEOUT"] = 60

params["simplex_terrain_types"] = ["wavy","smooth","rough"]
params["simplex_types_probs"] = [0.65,0.25,0.1]
params["terrain_params_noise"] = 0.05

params["EXPERIMENT_ID"] = "./log_simplex_runs_model04_challenging_ids_comb_v3/"
params["META_MODEL"] = "model04_pos"

params["WHICH_MODELS_BY_NAME"] = []
params["WHICH_MODELS_BY_INDEX"] = [0,1,2,3,4,5,6,7,8,9]

# 'meta' should always be used, and put first
# if meta_inf is used, it should come after "meta" to have its maximum planning time as the solution of meta
params["WHICH_METHOD"] = ["meta","meta_0", "meta_inf"] 

params["SEPARATE_MODEL"] = "model_single"
separate_model_dir = "log_simplex_runs_single_terrain_with_var"
separate_model_dir_2 = "log_simplex_runs_single_terrain_challenging_with_var"
params["SEPARATE_MODEL_DIR"] = [separate_model_dir, separate_model_dir_2]
separate_models_weights = {}
separate_models_weights["0"] = "./{}/2021-03-05 17-46-09_tid_0/model_terrain_0.h5".format(separate_model_dir)
separate_models_weights["1"] = "./{}/2021-03-05 17-46-09_tid_1/model_terrain_1.h5".format(separate_model_dir)
separate_models_weights["3"] = "./{}/2021-03-05 17-46-09_tid_3/model_terrain_3.h5".format(separate_model_dir)
separate_models_weights["4"] = "./{}/2021-03-05 17-46-09_tid_4/model_terrain_4.h5".format(separate_model_dir)
separate_models_weights["9"] = "./{}/2021-03-05 17-46-09_tid_9/model_terrain_9.h5".format(separate_model_dir)
separate_models_weights["11"] = "./{}/2021-03-05 17-46-09_tid_11/model_terrain_11.h5".format(separate_model_dir)
separate_models_weights["13"] = "./{}/2021-03-05 17-46-09_tid_13/model_terrain_13.h5".format(separate_model_dir)
separate_models_weights["14"] = "./{}/2021-03-05 17-46-09_tid_14/model_terrain_14.h5".format(separate_model_dir)
separate_models_weights["15"] = "./{}/2021-03-05 17-46-09_tid_15/model_terrain_15.h5".format(separate_model_dir)
separate_models_weights["16"] = "./{}/2021-03-05 17-46-09_tid_16/model_terrain_16.h5".format(separate_model_dir)
separate_models_weights["17"] = "./{}/2021-03-05 17-46-09_tid_17/model_terrain_17.h5".format(separate_model_dir)
separate_models_weights["22"] = "./{}/2021-03-05 17-46-09_tid_22/model_terrain_22.h5".format(separate_model_dir)

separate_models_weights["2"] = "./{}/2021-03-07 08-42-36_tid_2/model_terrain_2.h5".format(separate_model_dir_2)
separate_models_weights["5"] = "./{}/2021-03-07 08-42-36_tid_5/model_terrain_5.h5".format(separate_model_dir_2)
separate_models_weights["7"] = "./{}/2021-03-07 08-42-36_tid_7/model_terrain_7.h5".format(separate_model_dir_2)
separate_models_weights["8"] = "./{}/2021-03-07 08-42-36_tid_8/model_terrain_8.h5".format(separate_model_dir_2)
separate_models_weights["10"] = "./{}/2021-03-07 08-42-36_tid_10/model_terrain_10.h5".format(separate_model_dir_2)
separate_models_weights["12"] = "./{}/2021-03-07 08-42-36_tid_12/model_terrain_12.h5".format(separate_model_dir_2)
separate_models_weights["18"] = "./{}/2021-03-07 08-42-36_tid_18/model_terrain_18.h5".format(separate_model_dir_2)
separate_models_weights["19"] = "./{}/2021-03-07 08-42-36_tid_19/model_terrain_19.h5".format(separate_model_dir_2)
separate_models_weights["20"] = "./{}/2021-03-07 08-42-36_tid_20/model_terrain_20.h5".format(separate_model_dir_2)
separate_models_weights["21"] = "./{}/2021-03-07 08-42-36_tid_21/model_terrain_21.h5".format(separate_model_dir_2)

#-------------------------Constants not to change-----------------------------------------------------------#
belly = 0.5
wheelbase = 1.688965*2
wheeltrack = 0.95*2
eps_base = 0.1075
eps_track = 0.0775
discr = 0.0625
y1_loc = -wheeltrack/2
x1_loc = wheelbase/2
y2_loc = wheeltrack/2
x2_loc = wheelbase/2
y3_loc = -wheeltrack/2
x3_loc = -wheelbase/2
y4_loc = wheeltrack/2
x4_loc = -wheelbase/2
DEM_size_x = int(params["map_size_x"]/discr +1)
DEM_size_y = int(params["map_size_y"]/discr +1)
DEM_Y_base = np.ceil((wheeltrack+eps_track)/discr).astype(int)
DEM_X_base = np.ceil((wheelbase+eps_base)/discr).astype(int)
x = np.linspace(-params["map_size_x"]/2,params["map_size_x"]/2,num=DEM_size_x)
y = np.linspace(-params["map_size_y"]/2,params["map_size_y"]/2,num=DEM_size_y)
Y , X = np.meshgrid(y,x)
def isint(x):
    try:
        int(x)
        return True
    except ValueError:
        return False
    
def generate_Simplex(map_size_x, map_size_y, path_image):
    discr = 0.0625
    DEM_size_x = int(map_size_x/discr +1)
    DEM_size_y = int(map_size_y/discr +1)
    # Create random simplex map
    simplex_terrain_type = random.choices(params["simplex_terrain_types"],
                                          weights=params["simplex_types_probs"],k=1)[0]
    simplex = tg.OpenSimplex_Map(map_size_x, map_size_y, discr, terrain_type = simplex_terrain_type, plot = False)
    simplex.sample_generator(plot=True)
    Z = simplex.Z
    minz = np.min(Z)      
    Z = Z-minz
    map_height = np.max(Z).item()
    # Save map as image
    Z_pixel = (Z/map_height*255).astype(np.uint8)
    im = Image.new('L', (Z_pixel.shape[1],Z_pixel.shape[0]))
    im.putdata(Z_pixel.reshape(Z_pixel.shape[0]*Z_pixel.shape[1]))
    im = im.rotate(90, expand=True)#
    im = im.resize((int(DEM_size_x/1),int(DEM_size_y/1)), Image.BILINEAR)
    im.save("{}".format(path_image))   
    
    return Z

def search_path(path_planner, start, goal, method):
    if method != "fastest":
        id_a, states, costs = path_planner.search(start, goal, optimization_criteria = 'energy', method = method)
    else:
        id_a, states, costs = path_planner.search(start, goal, optimization_criteria = 'distance', method = method)
    if id_a is None:
        print("Path not found")
    else:
        print("Path Found:")
        print("Actions: ", id_a)
        print("States: ", states)
        print("Costs: ", costs)
        print("Total Cost: ", sum(costs))
        print()
        
    return id_a, states, costs, path_planner.elapsed_time, path_planner.nodes_expanded
        

# Create Directories
if not os.path.exists(path_data):
    os.mkdir(path_data)
if not os.path.exists(path_experiment):
    os.mkdir(path_experiment)
if not os.path.exists(path_maps):
        os.makedirs(path_maps)
def main():
    # Create Experiment DIrectories and Files
    current_time = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    # Loop over Experiment Files
    dirs = os.listdir(params["EXPERIMENT_ID"])
    files = []
    if params["WHICH_MODELS_BY_NAME"]:
        for w in params["WHICH_MODELS_BY_NAME"]:
            files.extend([d for d in dirs if w in d])
    elif params["WHICH_MODELS_BY_INDEX"]:
        files = [dirs[w] for w in params["WHICH_MODELS_BY_INDEX"]]
    else:
        files = dirs
            
    # Open states file
    file = random.sample(files,1)[0]
    model_weights = params["EXPERIMENT_ID"] + file + '/model_best.h5'
    log_params = params["EXPERIMENT_ID"] + file + "/log_params_{}.txt".format(file)
    l = open(log_params, "r")
    cont = l.readlines()
    for c in cont:
        if "TERRAIN_IDS_TRAIN" in c:
            c = c[len("TERRAIN_IDS_TRAIN")+3:-2]
            id_train = [int(x) for x in c.split(",") if isint(x)]
        if "TERRAIN_IDS_VAL" in c:
            c = c[len("TERRAIN_IDS_VAL")+3:-2]
            id_val = [int(x) for x in c.split(",") if isint(x)]
    # Terrain type changes each tyep_freq traverses
    terrain_type = random.sample(id_val,1)[0]
            
    print("File: {}. Terrain Type: {}".format(file,terrain_type))
    # Generate new Map
    path_image = path_maps + '{}.bmp'.format(current_time)
    Z = generate_Simplex(params["map_size_x"], params["map_size_y"], path_image)
    
    max_time_meta_inf = params["MAX_TIMEOUT"]
    # Initialise Path Planner
    path_planner = A_star.A_star(params["WHICH_METHOD"])
    path_planner.meta_h_scaling = params["meta_h_scaling"]
    meta_method = False
    x_ref = {}
    y_ref = {}
    yaw_ref = {}
    x_ref["fastest"] = []
    y_ref["fastest"] = []
    yaw_ref["fastest"] = []
    en_true, en_pred, plan_time, plan_nodes, plan_actions = {},{},{},{},{}
    for method in params["WHICH_METHOD"]:
        x_ref[method] = []
        y_ref[method] = []
        yaw_ref[method] = []
        if "meta" in method and not meta_method:
            path_planner.set_meta_model(params["META_MODEL"], model_weights)
            meta_method = True
        if "separate" in method:
            path_planner.set_separate_model(params["SEPARATE_MODEL"], [separate_models_weights[str(id_t)] for id_t in id_train])
    path_planner.set_map(Z,params["map_size_x"], params["map_size_y"])
    
    
    for i_m, method in enumerate(params["WHICH_METHOD"]):
        xi, yi, yawi = params["x0"], params["y0"], params["yaw0"]*math.pi/180
        
        # Initial Fastest Path
        if not i_m:
            # path_planner.timeout = params["MAX_TIMEOUT_FASTEST"]
            # xg = xi + 5.4
            # yg = yi
            # print("Fastest Plan ...")
            # id_a_fastest,_,_,_,_ = search_path(path_planner,
            #                                 [xi, yi, yawi], [xg,yg], 
            #                                 method = 'fastest')
            id_a_fastest = random.sample([[0,4],[4,0],[1,3],[3,1]], 1)[0]
        print("Fast Path: ", id_a_fastest)
        # Little trick to set initial y so as to be at y=params["y0"], after initial fast trajectory
        _,yv,_ = path_planner.points((xi, yi, yawi), id_a_fastest)
        yi -= yv[-1]
        # Initialise simulator
        z0, roll0, pitch0 = path_planner.starting_pose(xi, yi, yawi)
        sim = mcs.simulator(path_image, (params["map_size_x"],params["map_size_y"],Z.max().item()), 
                            (xi,yi,z0), (roll0,pitch0,yawi), 
                            terrain_type, params["terrain_params_noise"], 
                            visualisation = VISUALISATION)
        # Each Action of the path is run and its energy (divided in segments) is measured after execution
        tot_segments = 0
        id_a = id_a_fastest
        n_actions = 0
        idx_ida = 0
        fail = False
        while idx_ida < len(id_a):                                            
            xv,yv,yaw_v = path_planner.points((xi, yi, yawi), [id_a[idx_ida]])
            zv = [z0]*len(xv)
            if sim.run((xv,yv,zv), params["target_speed"]):
                print("Goal Reached!")
            else:
                print("Failure")
                fail = True
                break
            if not i_m:
                x_ref["fastest"].extend(xv)
                y_ref["fastest"].extend(yv)
                yaw_ref["fastest"].extend(yaw_v)
            # Next initial state after single action
            xi, yi, yawi = xv[-1], yv[-1], yaw_v[-1]
            
            # Retrieving energy from executed actions and saving in path planner memory
            energy_segments, pitch_segments, roll_segments = path_planner.segments_stats(sim.data_run, (xv,yv,yaw_v))
            energy_action = 0
            for segment in range(len(energy_segments)):
                print("Segment ", segment)
                print(" Energy ", energy_segments[segment])
                print(" Pitch (mean,var) [{}, {}]".format(pitch_segments[0][segment], pitch_segments[1][segment]))
                print(" Roll (mean,var) [{}, {}]".format(roll_segments[0][segment], roll_segments[1][segment]))
                
                # Exclude Actions with big acceleration components
                if n_actions and not i_m: 
                    tot_segments += 1
                    # This version add shots only with 'fastest', to be valid for all methods
                    path_planner.add_memory_shot(tot_segments, energy_segments[segment], 
                                                 pitch_segments[0][segment], roll_segments[0][segment],
                                                 pitch_segments[1][segment], roll_segments[1][segment],
                                                 "fastest")
                energy_action += energy_segments[segment]
            print("Energy action: ", energy_action)    
            print()
            idx_ida += 1
            n_actions += 1
        
        # Method Planning
        print("Method {} Plan ...".format(method))
        if method == "meta_inf":
            path_planner.timeout = max_time_meta_inf
        else:
            path_planner.timeout = params["MAX_TIMEOUT"]
        xg = xi + params["goal_distance"]
        yg = yi
        id_a,states,costs,time,nodes = search_path(path_planner,
                                        [xi, yi, yawi], [xg,yg], 
                                        method = method)
        if method == "meta":
            max_time_meta_inf = time   
        energy_tot = 0
        pred_energy_tot = 0
        idx_ida = 0
        fail = False
        while idx_ida < len(id_a):                                            
            xv,yv,yaw_v = path_planner.points((xi, yi, yawi), [id_a[idx_ida]])
            zv = [z0]*len(xv)
            if sim.run((xv,yv,zv), params["target_speed"]):
                print("Goal Reached!")
            else:
                print("Failure")
                fail = True
                break
            x_ref[method].extend(xv)
            y_ref[method].extend(yv)
            yaw_ref[method].extend(yaw_v)
            # Next initial state after single action
            xi, yi, yawi = xv[-1], yv[-1], yaw_v[-1]
            
            # Retrieving energy from executed actions and saving in path planner memory
            energy_segments, pitch_segments, roll_segments = path_planner.segments_stats(sim.data_run, (xv,yv,yaw_v))
            energy_action = 0
            for segment in range(len(energy_segments)):
                print("Segment ", segment)
                print(" Energy ", energy_segments[segment])
                print(" Pitch (mean,var) [{}, {}]".format(pitch_segments[0][segment], pitch_segments[1][segment]))
                print(" Roll (mean,var) [{}, {}]".format(roll_segments[0][segment], roll_segments[1][segment]))
                
                energy_action += energy_segments[segment]
                energy_tot += energy_segments[segment]
            pred_energy_tot += costs[idx_ida]   
            print("Energy action: ", energy_action)    
            print()
            idx_ida += 1
            n_actions += 1
    
        sim.close()
        del sim
                
        print("Total energy: ", energy_tot)   
        print("Total energy prediction: ", pred_energy_tot)
        print("Planning Time: ", time)
        print("Planning Nodes: ", nodes)
        print("Actions: ", id_a)
        print()
        print()
        
        en_true[method] = energy_tot
        en_pred[method] = pred_energy_tot
        plan_time[method] = time
        plan_nodes[method] = nodes
        plan_actions[method] = id_a        
        
        
    for method in params["WHICH_METHOD"]:
        if method == "meta":
            print("Meta-Adaptive")
        elif method == "meta_0":
            print("Meta-Optimal")
        elif method == "meta_inf":
            print("Meta-ARA*")
        else:
            print(method)
        print("Planning Actions: ", plan_actions[method])
        print("Energy True: ", en_true[method])
        print("Energy Pred: ", en_pred[method])
        print("Planning Time: ", plan_time[method])
        print("Planning Nodes: ", plan_nodes[method])
        print()
        
    # Plot DTM and Trajectory
    fig = plt.figure(figsize=(45,15))
    ax = plt.gca()
    ax.set_aspect("equal")
    ax.tick_params(labelsize=40)
    ax.set_xlabel("[m]", fontsize = 35)
    ax.set_ylabel("[m]", fontsize = 35, rotation = 90, va= "bottom", labelpad = -25)
    im = ax.pcolormesh(Y,X,Z, cmap='Greys',shading='auto')
    # Draw a Circle around the target
    goal_type = "circle"
    goal_radius = 1 # if circle
    goal_depth = 1 # if rectangle
    goal_width = 6 # if rectangle
    goal_point = (params["x0"] + 5.4+ params["goal_distance"], params["y0"])
    if goal_type=="circle":
        circle1 = plt.Circle((goal_point[1], goal_point[0]), goal_radius, color='lightgray')
        ax.add_artist(circle1)
    elif goal_type=="rectangle":
        ax.add_patch(patches.Rectangle((goal_point[1]-goal_width/2,goal_point[0]-goal_depth/2), goal_width, goal_depth, angle = 0, alpha = 0.8, color='g'))
    # Plot reference and real trajectory
    list_traj = []
    plt.plot(y_ref["fastest"],x_ref["fastest"], 'orange')
    # plt.plot([0]*4, np.linspace(params["x0"]+5.4,params["x0"]+5.4-2.7,4), 'o', color='chocolate')
    traj, = plt.plot(y_ref["meta"],x_ref["meta"], 'red', linewidth=6, label="Meta-Adaptive")
    list_traj.append(traj)
    traj, = plt.plot(y_ref["meta_0"],x_ref["meta_0"], 'blue', label="Meta-Optimal")
    list_traj.append(traj)
    traj, = plt.plot(y_ref["meta_inf"],x_ref["meta_inf"], 'green', label="Meta-ARA*")
    list_traj.append(traj)
    plt.legend(handles=list_traj,fontsize=40,loc="upper left", bbox_to_anchor = (1,1))  
    # Plot Robot
    xbl, ybl = bottom_left_wheel((params["x0"] + 5.4, params["y0"], params["yaw0"]*math.pi/180), wheeltrack, wheelbase)
    ax.add_patch(patches.Rectangle((ybl,xbl), wheeltrack, wheelbase, angle = -params["yaw0"], alpha = 0.4))
    vy = np.sin(params["yaw0"]*math.pi/180)*0.8
    vx = np.cos(params["yaw0"]*math.pi/180)*0.8
    plt.arrow(params["y0"],params["x0"] + 5.4,vy,vx, head_width=0.3, head_length=0.6, color = 'red' )
    
    plt.show()
    
    return Z, x_ref, y_ref, yaw_ref
        
        
def bottom_left_wheel(state, width, length):
    (XC,YC,Theta) = state
    pcenter = np.array([[YC],[XC]])
    rcenter = np.matrix(((np.cos(Theta), np.sin(Theta)), (-np.sin(Theta), np.cos(Theta))))
    pbl=pcenter+rcenter*np.array([[-width/2],[-length/2]])
    xbl=pbl[1].item()
    ybl=pbl[0].item()
    return xbl,ybl   
      
                    
    
        
        
            
            
if __name__ == "__main__":
    main()
