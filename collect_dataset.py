import os
import numpy as np
from PIL import Image
import math
import random
import pandas as pd
import my_chrono_simulator as mcs
import terrain_generator as tg
import heapq
from datetime import datetime
random.seed(0)

VISUALISATION = False
plot_maps = False

# Paths
path_data = "./Dataset/"
current_time = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
path_experiment = path_data + "Exp_{}/".format(current_time)
path_maps = path_experiment + "Simplex_Maps/"
path_output_file = path_experiment + "data.csv"

terrain_ids = []
terrain_ids.extend([0,1,3,4,9,11,13,14,15,16,17,22]) # less challenging
terrain_ids.extend([2,5,7,8,10,12,18,19,20,21])    # more challenging (6 is excluded)

num_runs_per_id = 150
simplex_terrain_types = ["wavy","smooth","rough"]
simplex_types_probs = [0.65,0.25,0.1] # proportions of the terrain types in the dataset 
terrain_params_noise = 0 # noise to add to the terramechanical parameters (percentage of each value)

map_size_x = 30
map_size_y = 7
x0 = -12.5
y0 = 0
yaw0 = 0

target_speed = 1

random_action_sequence = True
sequence_length = 9
sequence_id_a = [2]*9

num_actions = 5
length = 2.7
max_angle = 20*math.pi/180

segment_length = 0.9
n_points = 49
n_segments = int(length/segment_length)
segment_points = int((n_points-1)//n_segments)



# Constants not to change
# --------------------------------------------------------------------------------------------
belly = 0.5
wheelbase = 1.688965*2
wheeltrack = 0.95*2
eps_base = 0.1075
eps_track = 0.0775
discr = 0.0625
goal_radius = 0.5
y1_loc = -wheeltrack/2
x1_loc = wheelbase/2
y2_loc = wheeltrack/2
x2_loc = wheelbase/2
y3_loc = -wheeltrack/2
x3_loc = -wheelbase/2
y4_loc = wheeltrack/2
x4_loc = -wheelbase/2
DEM_size_x = int(map_size_x/discr +1)
DEM_size_y = int(map_size_y/discr +1)
DEM_Y_base = np.ceil((wheeltrack+eps_track)/discr).astype(int)
DEM_X_base = np.ceil((wheelbase+eps_base)/discr).astype(int)
x = np.linspace(-map_size_x/2,map_size_x/2,num=DEM_size_x)
y = np.linspace(-map_size_y/2,map_size_y/2,num=DEM_size_y)
Y , X = np.meshgrid(y,x)
y_base = np.linspace(-(wheeltrack+eps_track-discr)/2,(wheeltrack+eps_track-discr)/2,DEM_Y_base)
x_base = np.linspace(-(wheelbase+eps_base-discr)/2,(wheelbase+eps_base-discr)/2,DEM_X_base)
Y_base, X_base = np.meshgrid(y_base,x_base)
# Action space
forward_actions = []
curvature_list = np.linspace(-max_angle/length,max_angle/length,num_actions)
for curvature in curvature_list:
    if curvature:
        forward_actions.append((1/curvature,length*curvature*180/math.pi))
    else:
        forward_actions.append((length,0))
# --------------------------------------------------------------------------------------------

# Create Directories
if not os.path.exists(path_data):
    os.mkdir(path_data)
if not os.path.exists(path_experiment):
    os.mkdir(path_experiment)
if not os.path.exists(path_maps):
    os.mkdir(path_maps)

    
def bread_first_path(start):
    class PriorityQueue:
        def __init__(self):
            self.elements = []
        
        def empty(self):
            return len(self.elements) == 0
        
        def put(self, item, priority):
            heapq.heappush(self.elements, (priority, item))
        
        def get(self):
            return heapq.heappop(self.elements)[1]
    
    frontier = PriorityQueue()
    frontier.put(start,0)
    state_info = {}
    state_info[start] = (None,0,None)
        
    find = False
    actions = list(range(num_actions))
    while not frontier.empty() and not find:
        current_state = frontier.get()
        (xi,yi,yawi) = current_state
        (parent,distance,prev_ida) = state_info[current_state]
        
        random.shuffle(actions)
        
        for ida in actions:
            safe = True
            xi_ref, yi_ref, yawi_ref = action2traj(ida, xi, yi, yawi, n_points)
            for xxi, yyi, yyawi in zip(xi_ref, yi_ref, yawi_ref):
                p_wheels = all_wheels((xxi, yyi, yyawi), wheeltrack, wheelbase)
                for p_wheel in p_wheels:
                    if abs(p_wheel[0].item()) >= map_size_y/2-0.5 or abs(p_wheel[1].item()) >= map_size_x/2-0.5:
                        safe = False
                        break
                if not safe:
                    break
            if safe:
                next_state = (xi_ref[-1],yi_ref[-1],yawi_ref[-1])
                frontier.put(next_state,-(distance+1) + random.uniform(-.5,.5))
                state_info[next_state] = (current_state,distance+1,ida)
                if distance+1 == sequence_length:
                    find = True
                    final_state = next_state
                    break
    id_a = []          
    if find:
        while True:
            (parent,distance,prev_ida) = state_info[final_state]
            if parent is not None:
                id_a.append(prev_ida)
                final_state = parent
            else:
                break
        id_a.reverse()
    return id_a
        
def norm_angle(theta):
    if theta > math.pi:
        theta -= 2*math.pi
    elif theta < -math.pi:
        theta += 2*math.pi
    return theta

def action2traj(id_a,x0,y0,yaw0,n_cells):
    (r, dyaw) = forward_actions[id_a]
    dyaw = dyaw*math.pi/180
    yaw = np.linspace(yaw0,yaw0+dyaw,n_cells)
    yaw = np.array(list(map(norm_angle, yaw)))
    if dyaw:
        x = x0 - r*np.sin(yaw0) + r*np.sin(yaw);
        y = y0 + r*np.cos(yaw0) - r*np.cos(yaw);
    else:
        x = np.linspace(x0, x0 + r*np.cos(yaw0), n_cells);
        y = np.linspace(y0, y0 + r*np.sin(yaw0), n_cells);
    return x,y,yaw
def all_wheels(state, width, length):
    (XC,YC,Theta) = state
    pcenter = np.array([[YC],[XC]])
    rcenter = np.matrix(((np.cos(Theta), np.sin(Theta)), (-np.sin(Theta), np.cos(Theta))))
    pbl=pcenter+rcenter*np.array([[-width/2],[-length/2]])
    pbr=pcenter+rcenter*np.array([[width/2],[-length/2]])
    ptl=pcenter+rcenter*np.array([[-width/2],[length/2]])
    ptr=pcenter+rcenter*np.array([[width/2],[length/2]])
    return [pbl,pbr,ptl,ptr]

        
def starting_pose(x0,y0,yaw0,Z):
    # # Compute Wheels Position
    pcenter = np.array([[y0],[x0]])
    rcenter = np.matrix(((np.cos(yaw0), np.sin(yaw0)), (-np.sin(yaw0), np.cos(yaw0))))
    p1=pcenter+rcenter*np.array([[y1_loc],[x1_loc]])
    x1=p1[1].item()
    y1=p1[0].item()
    p2=pcenter+rcenter*np.array([[y2_loc],[x2_loc]])
    x2=p2[1].item()
    y2=p2[0].item()
    p3=pcenter+rcenter*np.array([[y3_loc],[x3_loc]])
    x3=p3[1].item()
    y3=p3[0].item()
    p4=pcenter+rcenter*np.array([[y4_loc],[x4_loc]])
    x4=p4[1].item()
    y4=p4[0].item()

    z_intorno = []
    for xi,yi in zip([x1,x2,x3,x4],[y1,y2,y3,y4]):
        ixl = [(np.floor((xi-(-DEM_size_x*discr/2))/discr)).astype(int)]
        iyl = [(np.floor((yi-(-DEM_size_y*discr/2))/discr)).astype(int)]
        
        ixl[0] = max(0,min(ixl[0],DEM_size_x -1))
        iyl[0] = max(0,min(iyl[0],DEM_size_y -1))
        
        ## Add some points in a 7x7 square
        for i in range(min(3,max(0,ixl[0]))):
            ixl.append(ixl[0]-i-1)
        for i in range(min(3,max(0,DEM_size_x -1-ixl[0]))):
            ixl.append(ixl[0]+i+1)
        for i in range(min(3,max(0,iyl[0]))):
            iyl.append(iyl[0]-i-1)
        for i in range(min(3,max(0,DEM_size_y -1-iyl[0]))):
            iyl.append(iyl[0]+i+1)
        zl = []
        for ix in ixl:
            for iy in iyl:
                zl.append(Z[ix,iy])
        z_intorno.append(np.array(zl).max())
    zt = np.array(z_intorno)
    ## Method3 points to fit
    points_x = [x1_loc, x2_loc, x3_loc, x4_loc]
    points_y = [y1_loc, y2_loc, y3_loc, y4_loc]
    points_z = z_intorno
    # Fit Plane which better approximates the points
    tmp_A = []
    tmp_b = []
    for i in range(len(points_x)):
        tmp_A.append([points_y[i], points_x[i], 1])
        tmp_b.append(points_z[i])
    b = np.matrix(tmp_b).T
    A = np.matrix(tmp_A)
    fit = (A.T * A).I * A.T * b 
    a = fit[0].item()
    b = fit[1].item()
    c = fit[2].item()
    # Compute Vectors in x and y directions
    z_vect = np.array([-a,-b,1])/np.linalg.norm([a,b,1])
    y_vect_y = math.sqrt(1/(1+(z_vect[0]/z_vect[2])**2))
    y_vect_z = -y_vect_y*z_vect[0]/z_vect[2]
    y_vect = np.array([y_vect_y, 0, y_vect_z])
    x_vect_x = math.sqrt(1/(1+(z_vect[1]/z_vect[2])**2))
    x_vect_z = -x_vect_x*z_vect[1]/z_vect[2]
    x_vect = np.array([0, x_vect_x, x_vect_z])
    # Compute roll and pitch angles
    roll0 = round(-np.arccos(np.dot(np.array([1,0,0]),y_vect))*180/math.pi,2)
    pitch0 = round(-np.arccos(np.dot(np.array([0,1,0]),x_vect))*180/math.pi,2)
    if y_vect_z>0:
        roll0 = -roll0
    if x_vect_z<0:
        pitch0 = -pitch0
    zw1 = a*y1_loc + b*x1_loc + c
    zw2 = a*y2_loc + b*x2_loc + c
    zw3 = a*y3_loc + b*x3_loc + c
    zw4 = a*y4_loc + b*x4_loc + c
    zw = np.array([zw1,zw2,zw3,zw4])
    dz = zt-zw
    z0 = round(c + dz.max(),3)+belly/np.cos(max(abs(roll0),abs(pitch0))*math.pi/180)
    #z0 = round(c + dz.max(),3)+belly
    return z0, roll0, pitch0

def segments_stats(data_run, ref_traj, Z, executed_actions = 1):
    dt = data_run.Time.values[1]-data_run.Time.values[0]
    stats = data_run.loc[data_run.Time >= 2,:]
    xv, yv, yaw_v = ref_traj
    # Estimated Pitch and Roll from initial poisiton using point clouds
    est_pitch_segments, est_roll_segments = estimate_pitch_roll(xv, yv, yaw_v, Z, executed_actions)
    # Splitting energy in segments from proprioceptive data
    energy_segments = np.zeros((n_segments*executed_actions))
    meas_speed_segments = np.zeros((2,n_segments*executed_actions))
    meas_roll_segments = np.zeros((2,n_segments*executed_actions))
    meas_pitch_segments = np.zeros((2,n_segments*executed_actions))
    power = np.array(stats["Motor_Speed"].values)*np.array(stats["Motor_Torque"].values)/1000   
    len_stats = len(stats)
    x_real, y_real, yaw_real = np.array(stats["X"].values),np.array(stats["Y"].values),np.array(stats["Yaw"].values)
    data_id = 0
    while data_id+1 < len_stats:
        xi,yi,yawi = x_real[data_id],y_real[data_id],yaw_real[data_id]
        id_min = np.sqrt(np.square(xi-xv)+np.square(yi-yv)).argmin()
        segment = int(id_min//segment_points)
        # Measured Variables from proprioceptive data
        energy_segment = 0
        meas_pitches = []
        meas_rolls = []
        meas_speed = []
        # Loop last until new segment is detected (or end file)
        while data_id < len_stats:
            energy_segment += power[data_id]*dt
            meas_pitches.append(stats["Pitch"].values[data_id])
            meas_rolls.append(stats["Roll"].values[data_id])
            meas_speed.append(stats["FWD_Speed"].values[data_id])
            if data_id+1<len_stats:
                data_id += 1
                xxi,yyi,yyawi = x_real[data_id],y_real[data_id],yaw_real[data_id]
                iid_min = np.sqrt(np.square(xxi-xv)+np.square(yyi-yv)).argmin()
                ssegment = int(iid_min//segment_points)
                if ssegment > segment:
                    if ssegment >= n_segments*executed_actions:
                        continue
                    else:
                        break
            else:
                break
        energy_segments[segment] = energy_segment
        meas_speed_segments[0,segment] = meas_speed[0] # initial speed
        meas_speed_segments[1,segment] = np.mean(meas_speed) # mean speed
        meas_pitch_segments[0,segment] = np.mean(meas_pitches)
        meas_pitch_segments[1,segment] = np.var(meas_pitches)
        meas_roll_segments[0,segment] = np.mean(meas_rolls)
        meas_roll_segments[1,segment] = np.var(meas_rolls)
        
    return energy_segments, est_pitch_segments, est_roll_segments, meas_pitch_segments, meas_roll_segments, meas_speed_segments 


def estimate_pitch_roll(xi_ref, yi_ref, yawi_ref, Z, executed_actions = 1):
    # Estimated Pitch and Roll from initial poisiton using point clouds
    mean_pitch_segments = np.zeros((n_segments*executed_actions))
    mean_roll_segments = np.zeros((n_segments*executed_actions))
    var_pitch_segments = np.zeros((n_segments*executed_actions))
    var_roll_segments = np.zeros((n_segments*executed_actions))
    for segment in range(n_segments*executed_actions):
        est_pitches = []
        est_rolls = []
        start_id = segment*segment_points
        end_id = start_id + segment_points
        for xx,yy,yyaw in zip(xi_ref[start_id:end_id],yi_ref[start_id:end_id],yawi_ref[start_id:end_id]):
            _, est_roll, est_pitch = starting_pose(xx, yy, yyaw, Z)
            est_pitches.append(est_pitch)
            est_rolls.append(est_roll)
        mean_pitch_segments[segment] = np.mean(est_pitches)
        mean_roll_segments[segment] = np.mean(est_rolls)
        var_pitch_segments[segment] = np.var(est_pitches)
        var_roll_segments[segment] = np.var(est_rolls)
    
    return (mean_pitch_segments, var_pitch_segments), (mean_roll_segments, var_roll_segments) 
         
def main():
    data = []
    for terrain_id in terrain_ids:
        for run in range(num_runs_per_id):
            print("Terrain id: {}, Run: {}".format(terrain_id,run))
            # Generate Simplex Map
            simplex_terrain_type = random.choices(simplex_terrain_types,weights=simplex_types_probs,k=1)[0]
            simplex = tg.OpenSimplex_Map(map_size_x, map_size_y, discr, terrain_type = simplex_terrain_type, plot = False)
            simplex.sample_generator(plot=plot_maps)
            Z = simplex.Z
            minz = np.min(Z)      
            Z = Z-minz
            map_height = np.max(Z)
            # Map is saved as image
            path_image = path_maps+"{0:02d}_{1:03d}_{2:.3f}.bmp".format(terrain_id,run,map_height)
            Z_pixel = (Z/map_height*255).astype(np.uint8)
            im = Image.new('L', (Z_pixel.shape[1],Z_pixel.shape[0]))
            im.putdata(Z_pixel.reshape(Z_pixel.shape[0]*Z_pixel.shape[1]))
            im = im.rotate(90, expand=True)
            im = im.resize((int(DEM_size_x/1),int(DEM_size_y/1)), Image.BILINEAR)
            im.save("{}".format(path_image)) 
            
            # Compute Initial Position
            z0, roll0, pitch0 = starting_pose(x0, y0, yaw0*math.pi/180, Z)
            prev_curv = 0
            # Define Action Sequence
            xi, yi, yawi = x0,y0,yaw0*math.pi/180
            if random_action_sequence:
                id_a = bread_first_path((xi,yi,yawi))
            else:
                id_a = sequence_id_a
                
            # Initialise Simulator
            sim = mcs.simulator(path_image, (map_size_x,map_size_y,map_height.item()), (x0,y0,z0), (roll0,pitch0,yaw0), terrain_id, terrain_params_noise, visualisation = VISUALISATION)
            # Execute Actions
            for ida in id_a:
                # Define points for path follower
                xv,yv,yaw_v = action2traj(ida, xi, yi, yawi, n_points)
                zv = [z0]*len(xv)
                # Run simulator
                if not sim.run((xv,yv,zv), target_speed):
                    print("Failure")
                    break
                # Retrieving statistics from executed action and adding to data
                energy_segments, est_pitch_segments, est_roll_segments, meas_pitch_segments, meas_roll_segments, meas_speed_segments  = segments_stats(sim.data_run, (xv,yv,yaw_v), Z)
                curv = curvature_list[ida]
                for segment in range(len(energy_segments)):
                    if not segment:
                        curv_tm1 = prev_curv
                    else:
                        curv_tm1 = curv
                    data.append({"terrain_id": terrain_id, "run": run,
                                 "segment": segment, "curvature": curv, "curvature_tm1": curv_tm1,
                                 "energy": energy_segments[segment],
                                 "mean_pitch_est": est_pitch_segments[0][segment], "mean_roll_est": est_roll_segments[0][segment],
                                 "mean_pitch_meas": meas_pitch_segments[0,segment], "mean_roll_meas": meas_roll_segments[0,segment],
                                 "var_pitch_meas": meas_pitch_segments[1,segment], "var_roll_meas": meas_roll_segments[1,segment],
                                 "var_pitch_est": est_pitch_segments[1][segment], "var_roll_est": est_roll_segments[1][segment],
                                 "initial_speed": meas_speed_segments[0,segment], "mean_speed": meas_speed_segments[1,segment]})
                # Next initial state
                xi, yi, yawi = xv[-1], yv[-1], yaw_v[-1]
                prev_curv = curv
            
            # Close simulation and save data in memory
            sim.close()
            pd.DataFrame(data).to_csv(path_output_file, index=False)
            
            
        
            

if __name__ == "__main__":
    main()

    
    
    





