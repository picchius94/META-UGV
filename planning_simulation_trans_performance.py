import numpy as np
import my_chrono_simulator as mcs
import terrain_generator as tg
from PIL import Image
import math
import random
import A_star
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as patches
from datetime import datetime
import os
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
random.seed(111)
#random.seed(55)
VISUALISATION = True
plot_graphs = True

path_terrains = "./Terrains/"

params = {}
params["Version"] = "path_planning_experiment_h_performance"
params["map_size_x"] = 50
params["map_size_y"] = 7
params["x0"] = -12.5-2.7*3
params["y0"] = 0
params["yaw0"] = 0

params["target_speed"] = 1
params["always_straight"] = False

params["meta_h_scaling"] = 1
params["goal_points"] = [(params["x0"]+2.7*1,0),(params["x0"]+2.7*4,0),(params["x0"]+2.7*7,0),
                         (params["x0"]+2.7*10,0),(params["x0"]+2.7*13,0),
                         (params["x0"]+2.7*16,0)]

params["MAX_TIMEOUT_FASTEST"] = 5
params["MAX_TIMEOUT"] = 60

params["simplex_terrain_types"] = ["wavy","smooth","rough"]
params["simplex_types_probs"] = [0.65,0.25,0.1]
params["terrain_params_noise"] = 0.05
params["n_terrain_types"] = 2
params["categories"] = [] # length must be equal to n_terrain_types or empty

params["EXPERIMENT_ID"] = "./log_simplex_runs_model04_challenging_ids_comb_v3/"
params["META_MODEL"] = "model04_pos"

params["WHICH_MODELS_BY_NAME"] = []
params["WHICH_MODELS_BY_INDEX"] = [0,1,2,3,4,5,6,7,8,9]

# 'meta' should always be used, and put first
# if meta_inf is used, it should come after "meta" to have its maximum planning time as the solution of meta
params["WHICH_METHOD"] = ["meta", "separate"] 

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
#------------------------------------------------------------------------------------------------------------#
def search_path(path_planner, start, goal, method, straight_flag = True):
    if method != "fastest":
        id_a, states, costs = path_planner.search(start, goal, optimization_criteria = 'energy', method = method, straight_flag = straight_flag)
    else:
        id_a, states, costs = path_planner.search(start, goal, optimization_criteria = 'distance', method = method, straight_flag = straight_flag)
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
def isint(x):
    try:
        int(x)
        return True
    except ValueError:
        return False
def main():
    current_time = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    if not os.path.exists(path_terrains):
        os.makedirs(path_terrains)
    # Create random simplex map
    path_image = path_terrains + '{}.bmp'.format(current_time)
    Z = generate_Simplex(params["map_size_x"], params["map_size_y"], path_image)
    
    
    # Select Model Weights and Train/Val terrain types from training files
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
    # Terrain type is selected from val
    # All terrains by macro-categories
    categories = {}
    # Very loose snow/sand
    categories["1"] = [18, 19, 20, 21]
    # Clay high moisture content
    categories["2"] = [7, 8, 10, 12]
    # Loose sand
    categories["3"] = [2, 5, 0, 22]
    # Sandy loams, moderately loose
    categories["4"] = [1, 3, 4, 13, 14, 15, 16, 17]
    # Compact clay
    categories["5"] = [9, 11]
    
    
    if params["categories"]:
        terrain_type = []
        n_t = 0
        while n_t < params["n_terrain_types"]:
            t = random.sample(id_val,1)[0]
            if t in categories[params["categories"][n_t]]:
                terrain_type.append(t)
                n_t += 1
            continue
    else:
        terrain_type = random.sample(id_val,params["n_terrain_types"])
            
    if len(terrain_type)==1:
        terrain_type = terrain_type[0]
    
    # Initialise Path Planner
    path_planner = A_star.A_star(params["WHICH_METHOD"])
    path_planner.meta_h_scaling = params["meta_h_scaling"]
    meta_method = False
    for method in params["WHICH_METHOD"]:
        if "meta" in method and not meta_method:
            path_planner.set_meta_model(params["META_MODEL"], model_weights)
            meta_method = True
        if "separate" in method:
            path_planner.set_separate_model(params["SEPARATE_MODEL"], [separate_models_weights[str(id_t)] for id_t in id_train])
    path_planner.set_map(Z,params["map_size_x"], params["map_size_y"])
    
    energy_true = {}
    energy_pred = {}
    x_real = {}
    y_real = {}
    for method in params["WHICH_METHOD"]:
        energy_true[method] = []
        energy_pred[method] = []
        x_real[method] = []
        y_real[method] = []
        # Initialise simulator
        xi, yi, yawi = params["x0"], params["y0"], params["yaw0"]*math.pi/180
        z0, roll0, pitch0 = path_planner.starting_pose(xi, yi, yawi)
        sim = mcs.simulator(path_image, (params["map_size_x"],params["map_size_y"],Z.max().item()), 
                            (xi,yi,z0), (roll0,pitch0,yawi), 
                            terrain_type, params["terrain_params_noise"], 
                            visualisation = VISUALISATION)
        
        # Loop over goal points
        energy_tot = 0
        energy_no_first = 0
        pred_energy_no_first = 0
        tot_segments = 0
        x_ref = []
        y_ref = []
        yaw_ref = []
        failure = False
        print()
        print()
        print("Terrain Types: ", terrain_type)
        for n_goal, goal_point in enumerate(params["goal_points"]):
            # Search Path
            print("Planning Path to {}".format(goal_point))
            if params["always_straight"] or not n_goal:
                straight_flag = True
            else:
                straight_flag = False
            id_a,states,costs,time,nodes = search_path(path_planner,
                                            [xi, yi, yawi], goal_point,
                                            method = method, straight_flag = straight_flag)
            if id_a is None:
                print("Path not found")
                sim.close()
                return -1
            if plot_graphs:
                xx_ref, yy_ref, yyaw_ref = path_planner.points((xi, yi, yawi), id_a)
                plot_planned_path(Z, xx_ref, yy_ref, yyaw_ref, xi, yi, yawi, [goal_point])
                x_ref.extend(xx_ref)
                y_ref.extend(yy_ref)
                yaw_ref.extend(yyaw_ref)
            if n_goal:
                pred_energy_no_first += sum(costs)
                energy_pred[method].extend(costs)
            else:
                energy_pred[method].extend([0]*len(costs))
            # Each Action of the path is run and its energy (divided in segments) is measured after execution
            print("Executing path ...")
            
            for ida in id_a:
                xv,yv,yaw_v = path_planner.points((xi, yi, yawi), [ida])
                zv = [z0]*len(xv)
                if sim.run((xv,yv,zv), params["target_speed"]):
                    print("Goal Reached!")
                else:
                    print("Failure")
                    failure = True
                    sim.close()
                    return -1
                # Next initial state after single action
                xi, yi, yawi = xv[-1], yv[-1], yaw_v[-1]
                #xi, yi, yawi = sim.data_run["X"].values[-1], sim.data_run["Y"].values[-1], sim.data_run["Yaw"].values[-1]*math.pi/180
                
                # Retrieving energy from executed actions and saving in path planner memory
                initial_speed = sim.data_run["FWD_Speed"].values[0]
                mean_speed = sim.data_run["FWD_Speed"].mean()
                # if n_goal and (initial_speed < 0.87 or mean_speed < 0.86):
                #     print("Failure")
                #     failure = True
                #     sim.close()
                #     return -1
                energy_segments, pitch_segments, roll_segments = path_planner.segments_stats(sim.data_run, (xv,yv,yaw_v))
                energy_action = 0
                for segment in range(len(energy_segments)):
                    print("Segment ", segment)
                    print(" Energy ", energy_segments[segment])
                    print(" Pitch (mean,var) [{}, {}]".format(pitch_segments[0][segment], pitch_segments[1][segment]))
                    print(" Roll (mean,var) [{}, {}]".format(roll_segments[0][segment], roll_segments[1][segment]))
                    
                    if tot_segments: # first ever segment contains initial acceleration (I exclude it)
                        path_planner.add_memory_shot(tot_segments, energy_segments[segment], 
                                                     pitch_segments[0][segment], roll_segments[0][segment],
                                                     pitch_segments[1][segment], roll_segments[1][segment],
                                                     method = method)
                    tot_segments += 1
                    energy_action += energy_segments[segment]
                    if n_goal:
                        energy_no_first += energy_segments[segment]
                energy_true[method].extend([energy_action])
                print("Energy action: ", energy_action)    
                print()
                energy_tot += np.sum(energy_segments)
                
            if failure:
                sim.close()
                return -1
            # Next initial state after single goal
            if params["always_straight"]:
                xi, yi, yawi = xv[-1], yv[-1], yaw_v[-1]
            else:
                xi, yi, yawi = sim.data_run["X"].values[-1], sim.data_run["Y"].values[-1], sim.data_run["Yaw"].values[-1]*math.pi/180
                
        
        print("Total energy: ", energy_tot)   
        print("Total energy without first action: ", energy_no_first)
        print("Total energy without first action pred: ", pred_energy_no_first)
        
        sim.close()
        
        stats = sim.data
        del sim 
        
        if failure:
            break
        if plot_graphs:
            dt = stats.Time.values[1]-stats.Time.values[0]
            len_stats = len(stats)
            stats["Motor_Power_kW"] = stats["Motor_Speed"]*stats["Motor_Torque"]/1000   
            energy_tot = 0
            energy_v = np.zeros(len_stats)
            for i in range(len_stats):
                energy_v[i] = energy_tot
                if stats["Motor_Power_kW"][i]>0:
                    energy_tot += stats["Motor_Power_kW"][i]*dt
            
            x_real[method] = stats["X"].values
            y_real[method] = stats["Y"].values
            plot_planned_path(Z,x_ref,y_ref,yaw_ref,stats["X"].values,stats["Y"].values,stats["Yaw"].values*math.pi/180, params["goal_points"])
            
            plt.figure()
            plt.plot(y_ref,x_ref)
            plt.plot(stats["Y"],stats["X"])
            plt.legend(["Reference","Real"])
            plt.show() 
            
            stats.plot(x="Time", y=["X"]) 
            stats.plot(x="Time", y=["Y"]) 
            stats.plot(x="Time", y=["I_Throttle","I_Braking"]) 
            stats.plot(x="Time", y=["I_Steering"]) 
            
            plt.figure()
            plt.plot(stats["Time"],stats["Motor_Torque"])
            plt.plot(stats["Time"],stats["Motor_Speed"])
            plt.plot(stats["Time"],stats["Motor_Power_kW"])
            plt.plot(stats["Time"],energy_v)
            plt.legend(["Motor Torque[Nm]", "Motor Speed [rad/s]", "Motor Power [kW]", "Energy [kJ]"])
            plt.show()
            
            plt.figure()
            plt.plot(stats["Time"],stats["FWD_Speed"])
            plt.plot(stats["Time"],[params["target_speed"]]*len_stats)
            plt.legend(["FWD Speed", "Target Speed"])
            plt.show()
            
            plt.figure()
            plt.plot(stats["Time"],stats["Roll"])
            plt.plot(stats["Time"],stats["Pitch"])
            plt.legend(["Roll", "Pitch"])
            plt.show()
    
    
    print()
    print()
    for method in params["WHICH_METHOD"]:
        print("Method {}".format(method))
        print("Energy true: ", energy_true[method])
        print("Energy pred: ", energy_pred[method])
        print("Tot Energy true: ", sum(energy_true[method]))
        print("Tot Energy pred: ", sum(energy_pred[method]))
        print("No first Energy true: ", sum(energy_true[method][1:]))
        print("No first Energy pred: ", sum(energy_pred[method][1:]))
        print("MSE: ", mean_squared_error(energy_true[method][1:], energy_pred[method][1:]))
        print("R2: ", r2_score(energy_true[method][1:], energy_pred[method][1:]))
        print()
    
    if plot_graphs:
        plot_models_paths(Z,x_real,y_real)
        plot_energy_graphs(energy_true, energy_pred)
        
    
def plot_energy_graphs(energy_true, energy_pred):   
    min_e = 1e5
    max_e = 0
    plt.figure(figsize=(35,15))
    ax = plt.gca()
    ax.tick_params(labelsize=50)
    ax.set_xlabel("Distance [m]", fontsize = 50)
    ax.set_ylabel("Energy [kJ]", fontsize = 50, rotation = 90, va= "bottom")
    
    list_traj = []
    for method in params["WHICH_METHOD"]:
        if method == 'meta':
            label = "Meta"
            color_true = 'royalblue'
            color_pred = 'slategray'
        else:
            label = "Sep.Model"
            color_true = 'darkorange'
            color_pred = 'saddlebrown'
        x_axis = np.linspace(0, 0 + 2.7*(len(energy_true[method])),(len(energy_true[method])),endpoint=False)
        en_true_plot = [0]
        en_true_plot.extend(energy_true[method][1:])
        en_pred_plot = [0]
        en_pred_plot.extend(energy_pred[method][1:])
        traj, = plt.plot(x_axis, en_true_plot, '--o', color=color_true, label="{} true".format(label), linewidth=4, markersize=17)
        list_traj.append(traj)
        traj, = plt.plot(x_axis, en_pred_plot, '--o', color=color_pred, label="{} pred.".format(label), linewidth=4, markersize=17)
        list_traj.append(traj)
        
        m_e = min(min(energy_true[method][1:]),min(energy_pred[method][1:]))
        M_e = max(max(energy_true[method][1:]),max(energy_pred[method][1:]))
        if m_e < min_e:
            min_e = m_e
        if M_e > max_e:
            max_e = M_e
    plt.legend(handles=list_traj,loc="best", fontsize=50)
    
    min_e  = 0
    
    xv = [0, 2.7*3, 2.7*6, 2.7*9, 2.7*12]
    plt.vlines(xv, min_e-2, max_e+2, colors='0.75', linestyles='dashed')
    x_bounds = ax.get_xlim()
    for xx in xv:
        ax.annotate(s='Plan.', fontsize=40, xy =(((xx-x_bounds[0])/(x_bounds[1]-x_bounds[0])),0.9), xycoords='axes fraction', verticalalignment='baseline', horizontalalignment='right' , rotation = 0)
    
    plt.show()
    
    

def plot_models_paths(Z,x_real,y_real):
    # Plot DTM and Trajectory
    fig = plt.figure(figsize=(15,15))
    ax = plt.gca()
    ax.set_aspect("equal")
    ax.tick_params(labelsize=40)
    ax.tick_params(axis='x')
    plt.xticks([-2.5,2.5], rotation=90)
    plt.yticks([-20,-10,0,10,20], rotation=90)
    ax.set_xlabel("[m]", fontsize = 35,rotation = 180)
    ax.set_ylabel("[m]", fontsize = 35, rotation = 90, va= "bottom")
    im = ax.pcolormesh(Y,X,Z, cmap="Greys",shading='auto')
    # cb = fig.colorbar(im, ax =ax)
    # cb.ax.tick_params(labelsize=40)
    # cb.set_label("[m]", fontsize=35, rotation = 90, va= "bottom", labelpad = 32)
    # Draw a Circle around the targets
    for goal_point in params["goal_points"]:
        circle1 = plt.Circle((goal_point[1], goal_point[0]), 1, color='lightgray')
        ax.add_artist(circle1)
    # Plot Robot
    x0, y0, yaw0 = params["x0"], params["y0"], params["yaw0"]*math.pi/180
    xbl, ybl = bottom_left_wheel((x0, y0, yaw0), wheeltrack, wheelbase)
    ax.add_patch(patches.Rectangle((ybl,xbl), wheeltrack, wheelbase, angle = -yaw0*180/math.pi, alpha = 0.4))
    vy = np.sin(yaw0)*0.8
    vx = np.cos(yaw0)*0.8
    plt.arrow(y0,x0,vy,vx, head_width=0.3, head_length=0.6, color = 'r' )
    # Plot reference and real trajectory
    list_traj = []
    for method in params["WHICH_METHOD"]:
        if method == 'meta':
            label = "Meta"
        else:
            label = "Sep.Model"
        traj, = plt.plot(y_real[method],x_real[method], label=label)
        list_traj.append(traj)
    plt.legend(handles=list_traj,fontsize = 35, loc='lower left', bbox_to_anchor = (1,1))
    plt.show()
    
def plot_models_paths_v2(Z,x_real,y_real):
    # Plot DTM and Trajectory
    fig = plt.figure(figsize=(15,15))
    ax = plt.gca()
    ax.set_aspect("equal")
    ax.tick_params(labelsize=40)
    ax.tick_params(axis='x')
    plt.xticks([-2.5,2.5], rotation=90)
    plt.yticks([-20,-10,0,10,20], rotation=90)
    ax.set_xlabel("[m]", fontsize = 35,rotation = 180)
    ax.set_ylabel("[m]", fontsize = 35, rotation = 90, va= "bottom")
    im = ax.pcolormesh(Y,X,Z, cmap="Greys",shading='auto')
    # cb = fig.colorbar(im, ax =ax)
    # cb.ax.tick_params(labelsize=40)
    # cb.set_label("[m]", fontsize=35, rotation = 90, va= "bottom", labelpad = 32)
    # Draw a Circle around the targets
    for goal_point in params["goal_points"]:
        circle1 = plt.Circle((goal_point[1], goal_point[0]), 1, color='lightgray')
        ax.add_artist(circle1)
    # Plot Robot
    x0, y0, yaw0 = params["goal_points"][0][0], params["goal_points"][0][1], 0
    xbl, ybl = bottom_left_wheel((x0, y0, yaw0), wheeltrack, wheelbase)
    ax.add_patch(patches.Rectangle((ybl,xbl), wheeltrack, wheelbase, angle = -yaw0*180/math.pi, alpha = 0.4))
    vy = np.sin(yaw0)*0.8
    vx = np.cos(yaw0)*0.8
    plt.arrow(y0,x0,vy,vx, head_width=0.3, head_length=0.6, color = 'r' )
    # Plot trajectories
    list_traj = []
    for method in params["WHICH_METHOD"]:
        if method == 'meta':
            label = "Meta"
        else:
            label = "Sep.Model"
        traj, = plt.plot(y_real[method],x_real[method], label=label)
        list_traj.append(traj)
    plt.legend(handles=list_traj,fontsize = 35, loc='lower left', bbox_to_anchor = (1,1))
    plt.show()
    
        
def plot_planned_path(Z,x_ref,y_ref,yaw_ref,x_real,y_real,yaw_real, goal_points):
    # Plot DTM and Trajectory
    fig = plt.figure(figsize=(15,15))
    ax = plt.gca()
    ax.set_aspect("equal")
    ax.set_title("Path on Map", fontsize = 35)
    ax.tick_params(labelsize=40)
    ax.tick_params(axis='x')
    ax.set_xlabel("[m]", fontsize = 35)
    ax.set_ylabel("[m]", fontsize = 35, rotation = 90, va= "bottom", labelpad = -25)
    im = ax.pcolormesh(Y,X,Z, cmap="Greys",shading='auto')
    cb = fig.colorbar(im, ax =ax)
    cb.ax.tick_params(labelsize=40)
    cb.set_label("[m]", fontsize=35, rotation = 90, va= "bottom", labelpad = 32)
    # Draw a Circle around the targets
    for goal_point in goal_points:
        circle1 = plt.Circle((goal_point[1], goal_point[0]), 1, color='lightgray')
        ax.add_artist(circle1)
    # Plot reference and real trajectory
    plt.plot(y_ref,x_ref)
    plt.plot(y_real,x_real)
    # Plot Robot
    if hasattr(x_real, "__iter__"):
        x0 = x_real[0]
        y0 = y_real[0]
        yaw0 = yaw_real[0]
        xf = x_real[-1]
        yf = y_real[-1]
        yawf = yaw_real[-1] 
    else:
        x0 = x_real
        y0 = y_real
        yaw0 = yaw_real  
    xbl, ybl = bottom_left_wheel((x0, y0, yaw0), wheeltrack, wheelbase)
    ax.add_patch(patches.Rectangle((ybl,xbl), wheeltrack, wheelbase, angle = -yaw0*180/math.pi, alpha = 0.4))
    vy = np.sin(yaw0)*0.8
    vx = np.cos(yaw0)*0.8
    plt.arrow(y0,x0,vy,vx, head_width=0.3, head_length=0.6, color = 'r' )
    if hasattr(x_real, "__iter__"):
        xbl, ybl = bottom_left_wheel((xf,yf,yawf), wheeltrack, wheelbase)
        ax.add_patch(patches.Rectangle((ybl,xbl), wheeltrack, wheelbase, angle = -yawf*180/math.pi, alpha = 0.4))
        vy = np.sin(yawf)*0.8
        vx = np.cos(yawf)*0.8
        plt.arrow(yf,xf,vy,vx, head_width=0.3, head_length=0.6, color = 'r' )
        plt.show()
    plt.show()

        
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