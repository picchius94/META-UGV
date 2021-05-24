import numpy as np
import math
import heapq
import random
import time
#import ray
import psutil
import pandas as pd
import models


class PriorityQueue:
    def __init__(self):
        self.elements = []
    
    def empty(self):
        return len(self.elements) == 0
    
    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))
    
    def get(self):
        return heapq.heappop(self.elements)[1]
    
    def get_best_priority(self):
        if not self.empty():
            element = heapq.heappop(self.elements)
            heapq.heappush(self.elements, element)
            return element[0]
        else:
            return np.inf
    
class A_star:
    def __init__(self, methods):
        # Robot dims
        self.wheelbase = 1.688965*2
        self.wheeltrack = 0.95*2
        self.belly = 0.5
        # Action space
        self.num_actions = 5
        self.length = 2.7
        self.max_angle = 20*math.pi/180
        self.forward_actions = []
        if self.num_actions > 1:
            self.curvature_list = np.linspace(-self.max_angle/self.length,self.max_angle/self.length,self.num_actions)
        else:
            self.curvature_list = np.array([0])
        for curvature in self.curvature_list:
            if curvature:
                self.forward_actions.append((1/curvature,self.length*curvature*180/math.pi))
            else:
                self.forward_actions.append((self.length,0))
        # Sortying actions by increasing curvature
        self.actions = list(range(self.num_actions))
        self.actions_sorted = []
        while self.actions:
            id_min = abs(self.curvature_list[self.actions]).argmin()
            self.actions_sorted.append(self.actions.pop(id_min))
        # Goal setting
        self.goal_type = 'circle'
        self.goal_radius = 1
        self.goal_depth = 1
        self.goal_width = 6
        # Each action is segmented with follwoing params
        self.segment_length = 0.9
        self.n_points = 49
        self.n_segments = int(self.length/self.segment_length)
        self.segment_points = int((self.n_points-1)//self.n_segments)
        
        # Scaling of meta heuristic to enforce underestimate
        self.meta_h_scaling = 1
            
        # try:
        #     ray.init(num_cpus=psutil.cpu_count(logical=False))
        # except:
        #     ray.shutdown()
        #     ray.init(num_cpus=psutil.cpu_count(logical=False))
        self.log_time_period = 1
        self.timeout = 60
        
        # Memory shots
        self.all_methods = methods
        self.memory_list = {}
        self.memory = {}
        for method in methods:
            self.memory_list[method] = []
            self.memory[method] = pd.DataFrame()
        
        self.input_features = ["mean_pitch_est", "mean_roll_est", "std_pitch_est", "std_roll_est"]
        self.all_opt_criterias = ["energy", "pitch", "distance"]
    
    def set_meta_model(self, model_name, model_weights):
        self.model_n_shots = 3
        self.model_n_meta = 1
        self.model = models.get_model(model_name,self.model_n_shots,self.model_n_meta, n_features = len(self.input_features), summary=True)
        self.model.load_weights(model_weights)
    def set_separate_model(self, model_name, separate_models_weights):
        self.model_n_shots = 3
        separate_models = []
        for weight in separate_models_weights:
            separate_models.append(models.get_model(model_name,None,None,len(self.input_features), summary=False))
            separate_models[-1].load_weights(weight)
        self.separate_models = separate_models
        
        
    def set_map(self, Z, map_size_x, map_size_y):
        # Map setting
        self.Z = Z
        self.map_size_x = map_size_x
        self.map_size_y = map_size_y
        self.discr_x = 0.01
        self.discr_y = 0.01
        self.discr_th = math.pi/180
        self.discr = 0.0625
        self.y1_loc = -self.wheeltrack/2
        self.x1_loc = self.wheelbase/2
        self.y2_loc = self.wheeltrack/2
        self.x2_loc = self.wheelbase/2
        self.y3_loc = -self.wheeltrack/2
        self.x3_loc = -self.wheelbase/2
        self.y4_loc = self.wheeltrack/2
        self.x4_loc = -self.wheelbase/2
        self.DEM_size_x = int(self.map_size_x/self.discr +1)
        self.DEM_size_y = int(self.map_size_y/self.discr +1)
        self.x = np.linspace(-self.map_size_x/2,self.map_size_x/2,num=self.DEM_size_x)
        self.y = np.linspace(-self.map_size_y/2,self.map_size_y/2,num=self.DEM_size_y)
        self.Y , self.X = np.meshgrid(self.y,self.x)
        
    def add_memory_shot(self, segment, energy, mean_pitch, mean_roll, var_pitch, var_roll, method = None):
        if method is None:
            method = self.method
        if method == "fastest":
            for method in self.all_methods:
                self.memory_list[method].append({"segment": segment, "energy": energy,
                                    "mean_pitch_est": mean_pitch, "var_pitch_est": var_pitch,
                                    "mean_roll_est": mean_roll, "var_roll_est": var_roll})
        else:
            self.memory_list[method].append({"segment": segment, "energy": energy,
                                "mean_pitch_est": mean_pitch, "var_pitch_est": var_pitch,
                                "mean_roll_est": mean_roll, "var_roll_est": var_roll})
    def update_memory(self, method = None):
        if method is None:
            method = self.method
        if method == "fastest":
            for method in self.all_methods:
                if self.memory_list[method]:
                    self.memory[method] = pd.DataFrame(self.memory_list[method])
                    self.memory[method]["std_pitch_est"] = self.memory[method]["var_pitch_est"].pow(0.5)
                    self.memory[method]["std_roll_est"] = self.memory[method]["var_roll_est"].pow(0.5)
        else:
            if self.memory_list[method]:
                self.memory[method] = pd.DataFrame(self.memory_list[method])
                self.memory[method]["std_pitch_est"] = self.memory[method]["var_pitch_est"].pow(0.5)
                self.memory[method]["std_roll_est"] = self.memory[method]["var_roll_est"].pow(0.5)
    
    def norm_angle(self,theta):
        if theta > math.pi:
            theta -= 2*math.pi
        elif theta < -math.pi:
            theta += 2*math.pi
        return theta
    
    def action2traj(self,id_a,x0,y0,yaw0, n_points= 49):
        (r, dyaw) = self.forward_actions[id_a]
        dyaw = dyaw*math.pi/180
        yaw = np.linspace(yaw0,yaw0+dyaw,n_points)
        yaw = np.array(list(map(self.norm_angle, yaw)))
        if dyaw:
            x = x0 - r*np.sin(yaw0) + r*np.sin(yaw);
            y = y0 + r*np.cos(yaw0) - r*np.cos(yaw);
        else:
            x = np.linspace(x0, x0 + r*np.cos(yaw0), n_points);
            y = np.linspace(y0, y0 + r*np.sin(yaw0), n_points);
        return x,y,yaw
    
    def all_wheels(self, state, width, length):
        (XC,YC,Theta) = state
        pcenter = np.array([[YC],[XC]])
        rcenter = np.matrix(((np.cos(Theta), np.sin(Theta)), (-np.sin(Theta), np.cos(Theta))))
        pbl=pcenter+rcenter*np.array([[-width/2],[-length/2]])
        pbr=pcenter+rcenter*np.array([[width/2],[-length/2]])
        ptl=pcenter+rcenter*np.array([[-width/2],[length/2]])
        ptr=pcenter+rcenter*np.array([[width/2],[length/2]])
        return [pbl,pbr,ptl,ptr]
        

    def points(self, start, id_a, n_points = 49):
        x_ref = []
        y_ref = []
        yaw_ref = []
        xi, yi, yawi = start
        for ida in id_a:
            xi_ref, yi_ref, yawi_ref = self.action2traj(ida, xi, yi, yawi, n_points)
            if x_ref:
                xi_ref = xi_ref[1:]
                yi_ref = yi_ref[1:]
                yawi_ref = yawi_ref[1:]
            x_ref.extend(list(xi_ref))
            y_ref.extend(list(yi_ref))
            yaw_ref.extend(list(yawi_ref))
            xi = x_ref[-1]
            yi = y_ref[-1]
            yawi = yaw_ref[-1]
        return x_ref, y_ref, yaw_ref
    
    # Conversion from continuous to discrete:
    def cont2disc(self, s):
        (xc,yc,thc) = s
        xd = (np.floor((xc-(-self.map_size_x/2))/self.discr_x)).astype(int)
        yd = (np.floor((yc-(-self.map_size_y/2))/self.discr_y)).astype(int)
        if thc < -math.pi:
            thc += 2*math.pi
        elif thc > math.pi:
            thc -= 2*math.pi
        thd = (np.floor((thc-(-math.pi))/(self.discr_th))).astype(int)
        return (xd,yd,thd) 
    
    # Conversion from discrete to continuous    
    def disc2cont(self, s):
        (xd,yd,thd) = s
        xi = round(xd*self.discr_x- self.map_size_x/2, 4)
        yi = round(yd*self.discr_y - self.map_size_y/2, 4)
        th_i = round(thd*self.discr_th - math.pi, 4)
        return (xi, yi, th_i)
    
    def safety_check(self, xi_ref, yi_ref, yawi_ref):
        safe = True
        for xxi, yyi, yyawi in zip(xi_ref, yi_ref, yawi_ref):
            p_wheels = self.all_wheels((xxi, yyi, yyawi), self.wheeltrack, self.wheelbase)
            for p_wheel in p_wheels:
                if abs(p_wheel[0].item()) >= self.map_size_y/2-0.5 or abs(p_wheel[1].item()) >= self.map_size_x/2-0.5:
                    safe = False
                    return safe
            if self.goal_type == "rectangle" and xxi > self.end[0] + self.goal_depth/2:
                safe = False
                return safe
        return safe
    
    def goal_check(self, xi_ref, yi_ref, yawi_ref):
        next_state = (xi_ref[-1],yi_ref[-1],yawi_ref[-1])
        if self.goal_type == "circle":
            distance = np.sqrt(np.square(self.end[0]-next_state[0])+np.square(self.end[1]-next_state[1]))
            if distance < self.goal_radius:
                goal = True
            else:
                goal = False
        elif self.goal_type == "rectangle":
            if np.abs(self.end[0]-next_state[0]) <= self.goal_depth/2 and np.abs(self.end[1]-next_state[1]) <= self.goal_width/2:
                goal = True
            else:
                goal = False
        return next_state, goal
    
    def starting_pose(self, x0,y0,yaw0):
        # # Compute Wheels Position
        pcenter = np.array([[y0],[x0]])
        rcenter = np.matrix(((np.cos(yaw0), np.sin(yaw0)), (-np.sin(yaw0), np.cos(yaw0))))
        p1=pcenter+rcenter*np.array([[self.y1_loc],[self.x1_loc]])
        x1=p1[1].item()
        y1=p1[0].item()
        p2=pcenter+rcenter*np.array([[self.y2_loc],[self.x2_loc]])
        x2=p2[1].item()
        y2=p2[0].item()
        p3=pcenter+rcenter*np.array([[self.y3_loc],[self.x3_loc]])
        x3=p3[1].item()
        y3=p3[0].item()
        p4=pcenter+rcenter*np.array([[self.y4_loc],[self.x4_loc]])
        x4=p4[1].item()
        y4=p4[0].item()
    
        z_intorno = []
        for xi,yi in zip([x1,x2,x3,x4],[y1,y2,y3,y4]):
            ixl = [(np.floor((xi-(-self.DEM_size_x*self.discr/2))/self.discr)).astype(int)]
            iyl = [(np.floor((yi-(-self.DEM_size_y*self.discr/2))/self.discr)).astype(int)]
            
            ixl[0] = max(0,min(ixl[0],self.DEM_size_x -1))
            iyl[0] = max(0,min(iyl[0],self.DEM_size_y -1))
            
            ## Add some points in a 7x7 square
            for i in range(min(3,max(0,ixl[0]))):
                ixl.append(ixl[0]-i-1)
            for i in range(min(3,max(0,self.DEM_size_x -1-ixl[0]))):
                ixl.append(ixl[0]+i+1)
            for i in range(min(3,max(0,iyl[0]))):
                iyl.append(iyl[0]-i-1)
            for i in range(min(3,max(0,self.DEM_size_y -1-iyl[0]))):
                iyl.append(iyl[0]+i+1)
            zl = []
            for ix in ixl:
                for iy in iyl:
                    zl.append(self.Z[ix,iy])
            z_intorno.append(np.array(zl).max())
        zt = np.array(z_intorno)
        ## Method3 points to fit
        points_x = [self.x1_loc, self.x2_loc, self.x3_loc, self.x4_loc]
        points_y = [self.y1_loc, self.y2_loc, self.y3_loc, self.y4_loc]
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
        zw1 = a*self.y1_loc + b*self.x1_loc + c
        zw2 = a*self.y2_loc + b*self.x2_loc + c
        zw3 = a*self.y3_loc + b*self.x3_loc + c
        zw4 = a*self.y4_loc + b*self.x4_loc + c
        zw = np.array([zw1,zw2,zw3,zw4])
        dz = zt-zw
        z0 = round(c + dz.max(),3)+self.belly/np.cos(max(abs(roll0),abs(pitch0))*math.pi/180)
        return z0, roll0, pitch0
    
    def segments_stats(self, data_run, ref_traj, executed_actions = 1):
        dt = data_run.Time.values[1]-data_run.Time.values[0]
        stats = data_run.loc[data_run.Time >= 2,:]
        xv, yv, yaw_v = ref_traj
        # Estimated Pitch and Roll from initial poisiton using point clouds
        (mean_pitch_segments, var_pitch_segments), (mean_roll_segments, var_roll_segments) = self.estimate_pitch_roll(xv, yv, yaw_v, executed_actions)
        # Splitting energy in segments from proprioceptive data
        energy_segments = np.zeros((self.n_segments*executed_actions))
        power = np.array(stats["Motor_Speed"].values)*np.array(stats["Motor_Torque"].values)/1000   
        len_stats = len(stats)
        x_real, y_real, yaw_real = np.array(stats["X"].values),np.array(stats["Y"].values),np.array(stats["Yaw"].values)
        data_id = 0
        while data_id+1 < len_stats:
            xi,yi,yawi = x_real[data_id],y_real[data_id],yaw_real[data_id]
            id_min = np.sqrt(np.square(xi-xv)+np.square(yi-yv)).argmin()
            segment = int(id_min//self.segment_points)
            # Measured Variables (energy) from proprioceptive data
            energy_segment = 0
            # Loop last until new segment is detected (or end file)
            while data_id < len_stats:
                energy_segment += power[data_id]*dt
                if data_id+1<len_stats:
                    data_id += 1
                    xxi,yyi,yyawi = x_real[data_id],y_real[data_id],yaw_real[data_id]
                    iid_min = np.sqrt(np.square(xxi-xv)+np.square(yyi-yv)).argmin()
                    ssegment = int(iid_min//self.segment_points)
                    if ssegment > segment:
                        if ssegment >= self.n_segments*executed_actions:
                            continue
                        else:
                            break
                else:
                    break
            energy_segments[segment] = energy_segment
        return energy_segments, (mean_pitch_segments, var_pitch_segments), (mean_roll_segments, var_roll_segments) 


    def estimate_pitch_roll(self, xi_ref, yi_ref, yawi_ref, executed_actions = 1):
        # Estimated Pitch and Roll from initial poisiton using point clouds
        mean_pitch_segments = np.zeros((self.n_segments*executed_actions))
        mean_roll_segments = np.zeros((self.n_segments*executed_actions))
        var_pitch_segments = np.zeros((self.n_segments*executed_actions))
        var_roll_segments = np.zeros((self.n_segments*executed_actions))
        for segment in range(self.n_segments*executed_actions):
            est_pitches = []
            est_rolls = []
            start_id = segment*self.segment_points
            end_id = start_id + self.segment_points
            for xx,yy,yyaw in zip(xi_ref[start_id:end_id],yi_ref[start_id:end_id],yawi_ref[start_id:end_id]):
                _, est_roll, est_pitch = self.starting_pose(xx, yy, yyaw)
                est_pitches.append(est_pitch)
                est_rolls.append(est_roll)
            mean_pitch_segments[segment] = np.mean(est_pitches)
            mean_roll_segments[segment] = np.mean(est_rolls)
            var_pitch_segments[segment] = np.var(est_pitches)
            var_roll_segments[segment] = np.var(est_rolls)
        
        return (mean_pitch_segments, var_pitch_segments), (mean_roll_segments, var_roll_segments) 
        
    #@ray.remote
    def advance_state(self, ida, state):
        xi, yi, yawi = state
        xi_ref, yi_ref, yawi_ref = self.action2traj(ida, xi, yi, yawi, self.n_points)
        # Check Safety
        safe = self.safety_check(xi_ref, yi_ref, yawi_ref)
        # Check Goal
        next_state, goal = self.goal_check(xi_ref, yi_ref, yawi_ref)
        # Additional variables to compute cost
        if self.optimization_criteria == "distance":
            cost_vars = self.length + abs(self.curvature_list[ida]) # discourage curves
        elif self.optimization_criteria == "energy" or self.optimization_criteria == "pitch":
            pitch_segments, roll_segments = self.estimate_pitch_roll(xi_ref, yi_ref, yawi_ref)
            if self.optimization_criteria == "energy":
                cost_vars = (pitch_segments, roll_segments)
            elif self.optimization_criteria == "pitch":
                cost_vars = max(-5,-np.mean(pitch_segments[0]))
            
        return [safe, next_state, goal, ida, cost_vars]
    
    def separate_model_pred(self, n_shots, XG_SHOTS, XE_SHOTS, XG_META):
        if n_shots:
            # Identify most similar model based on shots
            error = np.empty((len(self.separate_models),XG_META.shape[0]))
            XG_SHOTS = XG_SHOTS[:,:n_shots]
            XE_SHOTS = XE_SHOTS[:,:n_shots]
            for i, separate_model in enumerate(self.separate_models):
                for b in range(XG_META.shape[0]):
                    ye = separate_model(XG_SHOTS[b])
                    ye = ye.numpy()
                    if not b:
                        YE_SHOTS = np.expand_dims(ye,axis=0)
                    else:
                        YE_SHOTS = np.concatenate([YE_SHOTS,np.expand_dims(ye,axis=0)],axis=0)
                error[i] = np.sum(np.square(YE_SHOTS-XE_SHOTS),axis=1).squeeze()
            best_m = np.argmin(error,axis=0)
            # Use it to make new prediction
            for i, b in enumerate(best_m):
                ye = self.separate_models[b](XG_META[i])
                ye = ye.numpy()
                if not i:
                    YE_P_META = np.expand_dims(ye,axis=0)
                else:
                    YE_P_META = np.concatenate([YE_P_META,np.expand_dims(ye,axis=0)],axis=0)
        else:
            # If zero shots, I use all models and take the average value
            for i in range(XG_META.shape[0]):
                ye = np.array([separate_model(XG_META[i]).numpy().item() for separate_model in self.separate_models])
                ye = np.array([np.mean(ye)])
                if not i:
                    YE_P_META = np.expand_dims(ye,axis=0)
                else:
                    YE_P_META = np.concatenate([YE_P_META,np.expand_dims(ye,axis=0)],axis=0)       
        return YE_P_META

    def create_shots_batch(self, XG_META):
        # Creating batch of past shots to inform the network
        # Note, if available shots are less than self.model_n_shots, 
        # position of XG_META is rearranged so as to enable prediction
        # with available shots (even if 0 shots are available)
        
        memory = self.memory[self.method]
        
        n_shots = min(self.model_n_shots,len(memory))
        for i in range(len(XG_META)):
            if n_shots:
                # Most recent shots are used
                xg_shots = memory.loc[:,self.input_features][-n_shots:].values
                xe_shots = memory.loc[:,["energy"]][-n_shots:].values
            for j in range(self.model_n_shots-n_shots):
                if not j:
                    if n_shots:
                        xg_shots = np.concatenate((xg_shots,XG_META[i]),axis=0)
                        xe_shots = np.concatenate((xe_shots,np.zeros((1,1))),axis=0)
                    else:
                        xg_shots = XG_META[i]
                        xe_shots = np.zeros((1,1))
                else:
                    xg_shots = np.concatenate((xg_shots,np.zeros((1,len(self.input_features)))),axis=0)
                    xe_shots = np.concatenate((xe_shots,np.zeros((1,1))),axis=0)                   
                    
            if not i:
                XG_SHOTS = np.expand_dims(xg_shots, axis=0)
                XE_SHOTS = np.expand_dims(xe_shots, axis=0)
            else:
                XG_SHOTS = np.concatenate([XG_SHOTS,np.expand_dims(xg_shots,axis=0)],axis=0)
                XE_SHOTS = np.concatenate([XE_SHOTS,np.expand_dims(xe_shots,axis=0)],axis=0)
        
        return n_shots, XG_SHOTS, XE_SHOTS
        
    def estimate_energy(self, outputs):
        # Creating batch of geometries of future trajectories
        xg_meta = []
        for i, output in enumerate(outputs):
            if output[0]:
                pitch_segment, roll_segment = output[4]
                mean_pitch = pitch_segment[0]
                std_pitch = np.sqrt(pitch_segment[1])
                mean_roll = roll_segment[0]
                std_roll = np.sqrt(roll_segment[1])
                if len(self.input_features) == 2:
                    for p, r in zip(mean_pitch, mean_roll):
                        xg_meta.append([p, r])
                elif len(self.input_features) == 4:
                    for p, r, vp, vr in zip(mean_pitch, mean_roll, std_pitch, std_roll):
                        xg_meta.append([p, r, vp, vr])
        if not len(xg_meta):
            return outputs
        XG_META = np.expand_dims(np.array(xg_meta),axis=1)
        n_shots, XG_SHOTS, XE_SHOTS = self.create_shots_batch(XG_META)
        # Predicting energy
        if self.method == "meta" or self.method == "meta_0" or self.method == "meta_inf":
            # Note, the whole sequence is predicted, while the future value is at the n_shots position
            energies = (self.model(XG_SHOTS, XE_SHOTS, XG_META)[:,n_shots,:]).numpy().squeeze()
        elif self.method == "separate":
            energies = self.separate_model_pred(n_shots, XG_SHOTS, XE_SHOTS, XG_META).squeeze()
        energies = np.clip(energies,0,None)
        
        final_output = []
        id_e = 0
        for output in outputs:
            if output[0]:
                output[4] = 0
                for i in range(self.n_segments):
                    output[4] += energies[id_e]
                    id_e += 1
            final_output.append(output)
        return final_output
        
    def min_goal(self, state):
        if self.goal_type == 'rectangle' and np.abs(self.end[1]-state[1]) <= self.goal_width/2:
            min_dist = np.abs(self.end[0]-self.goal_depth/2-state[0])
            min_zf = self.Z[(np.floor((self.end[0]-self.goal_depth/2-(-self.DEM_size_x*self.discr/2))/self.discr)).astype(int),
                            (np.floor((state[1]-(-self.DEM_size_y*self.discr/2))/self.discr)).astype(int)]
        else:
            min_dist = np.sqrt(np.square(self.end[0]-state[0])+np.square(self.end[1]-state[1]))
            min_zf = self.Z[(np.floor((self.end[0]-(-self.DEM_size_x*self.discr/2))/self.discr)).astype(int),
                            (np.floor((self.end[1]-(-self.DEM_size_y*self.discr/2))/self.discr)).astype(int)]  
        return min_dist, min_zf
        
    def heuristic(self, state):
        if self.optimization_criteria == "distance":
            distance = np.sqrt(np.square(self.end[0]-state[0])+np.square(self.end[1]-state[1]))
            if self.method != "fastest":
                return distance
            else:
                return distance*10            
        elif self.optimization_criteria == "energy" or self.optimization_criteria == "pitch":
            zs = self.Z[(np.floor((state[0]-(-self.DEM_size_x*self.discr/2))/self.discr)).astype(int),
                        (np.floor((state[1]-(-self.DEM_size_y*self.discr/2))/self.discr)).astype(int)]
            distance, zf = self.min_goal(state)
            if abs(distance) > self.discr:
                pitch = -np.arctan((zf-zs)/distance)*180/math.pi
            else:
                pitch = 0.0
            if self.optimization_criteria == "energy":
                if self.method == "meta" or self.method == "separate":
                    if len(self.input_features) == 2:
                        XG_META = np.array([[[pitch,0.0]]])
                    elif len(self.input_features) == 4:
                        XG_META = np.array([[[pitch,0.0,0.0,0.0]]])                    
                    n_shots, XG_SHOTS, XE_SHOTS = self.create_shots_batch(XG_META)
                if self.method == "meta":
                    energy = (self.model(XG_SHOTS, XE_SHOTS, XG_META)[:,n_shots,:]).numpy().squeeze().item()*self.meta_h_scaling
                elif self.method == "meta_0" or self.method == "meta_inf":
                    energy = max(0,-1.41*pitch-4)
                elif self.method == "separate":
                    energy = self.separate_model_pred(n_shots, XG_SHOTS, XE_SHOTS, XG_META).squeeze()
                    
                return energy*distance/self.segment_length
            elif self.optimization_criteria == "pitch":
                return max(0,-pitch*distance/self.length)
            
    
    def neighbours(self,state):
        state = self.disc2cont(state)
        #outputs = ray.get([self.advance_state.remote(self, ida, state) for ida in self.actions_sorted])
        if self.straight_flag:
            outputs = [self.advance_state(ida, state) for ida in [self.actions_sorted[0]]]
        else:
            outputs = [self.advance_state(ida, state) for ida in self.actions_sorted]
        if self.optimization_criteria == "distance" or self.optimization_criteria == "pitch":
            return outputs
        elif self.optimization_criteria == "energy":
            return self.estimate_energy(outputs)
            
    def update_OPEN(self, INCONS, eps):
        list_open = []
        min_fvalue = np.inf
        while not self.OPEN.empty():
            state = self.OPEN.get()
            if self.state_info[state][1][1]:
                fvalue = self.cost_so_far[state]
                priority = self.cost_so_far[state]
            else:
                fvalue = self.cost_so_far[state] + self.heuristic(self.disc2cont(state))
                priority = self.cost_so_far[state] + self.heuristic(self.disc2cont(state))*eps
            if fvalue < min_fvalue:
                min_fvalue = fvalue
            list_open.append((state, priority))
        for state in INCONS:
            if self.state_info[state][1][1]:
                fvalue = self.cost_so_far[state]
                priority = self.cost_so_far[state]
            else:
                fvalue = self.cost_so_far[state] + self.heuristic(self.disc2cont(state))
                priority = self.cost_so_far[state] + self.heuristic(self.disc2cont(state))*eps
            if fvalue < min_fvalue:
                min_fvalue = fvalue
            list_open.append((state, priority))
        for element in list_open:
            self.OPEN.put(element[0],element[1])
        
        return min_fvalue
                
        
    def Improve_Path(self, eps):
        CLOSED = []
        INCONS = []
        period_count = 0
        solution = None
        at_least_one_solution = False
        exception_time = False
        while solution is None:
            if not self.OPEN.empty():
                current_state = self.OPEN.get()
            else:
                break
            (parent,info) = self.state_info[current_state]
            if info[1]: # if goal is True
                solution = current_state
                break
            CLOSED.append(current_state)
            outputs = self.neighbours(current_state)
            goal_next_state = [None,np.inf]
            for output in outputs:
                safe, next_state, goal, ida, cost = output
                if safe:
                    new_cost = self.cost_so_far[current_state] + cost
                    h_cost = self.heuristic(next_state)
                    next_state = self.cont2disc(next_state)
                    if next_state not in self.cost_so_far or new_cost < self.cost_so_far[next_state]:
                        self.cost_so_far[next_state] = new_cost
                        self.state_info[next_state] = (current_state,(ida,goal,cost))
                        if goal:
                            at_least_one_solution = True
                            priority = new_cost
                            if priority < goal_next_state[1]:
                                goal_next_state = [next_state,priority]
                        else:
                            priority = new_cost + h_cost*eps
                        if next_state not in CLOSED:
                            self.OPEN.put(next_state, priority)
                        else:
                            INCONS.append(next_state)
            if goal_next_state[0] is not None:
                if goal_next_state[1] <= self.OPEN.get_best_priority():
                    solution = goal_next_state[0]
                
            self.elapsed_time = time.time() - self.start_time
            self.nodes_expanded += 1
            if solution is not None:
                break
            if self.elapsed_time > self.timeout:
                exception_time = True
                if at_least_one_solution or self.elapsed_time > 60:
                    break
            if self.elapsed_time >= self.log_time_period*period_count:
                period_count += 1
                print("Elapsed Time: {}s. "
                      "Nodes Expanded: {}. "
                      "Elements in Queue: {}".format(round(self.elapsed_time,3),self.nodes_expanded,len(self.OPEN.elements)))
        
        return INCONS, solution, exception_time
        
        
                    
    def retrieve_solution(self, final_state):
        id_a = []
        states = []
        costs = []
        s = self.disc2cont(final_state)
        states.append((s[0], s[1], round(s[2]*180/math.pi,1)))
        while True:
            (parent,info) = self.state_info[final_state]
            if parent is not None:
                id_a.append(info[0])
                costs.append(info[2])
                final_state = parent
                s = self.disc2cont(final_state)
                states.append((s[0], s[1], round(s[2]*180/math.pi,1)))
            else:
                break
        id_a.reverse()
        states.reverse()
        costs.reverse()
        return id_a, states, costs
        
    def search(self, start, end, optimization_criteria = 'energy', method = "meta", straight_flag=False):
        self.straight_flag = straight_flag
        if optimization_criteria not in self.all_opt_criterias:
            print("Non-valid optimization criteria")
            print("Options: ", self.all_opt_criterias)
            return None, None, None
        self.optimization_criteria = optimization_criteria
        if method not in self.all_methods and method != "fastest":
            print("Non-valid method")
            print("Options: ", self.all_methods+" fastest")
            return None, None, None
        self.method = method
        
        self.OPEN = PriorityQueue()
        INCONS = PriorityQueue()
        start = self.cont2disc(start)
        self.end = end
        self.OPEN.put(start,0)
        self.state_info = {}
        self.state_info[start] = (None,(None,False,None))
        self.cost_so_far = {}
        self.cost_so_far[start] = 0
        
        if self.optimization_criteria == "energy":
            self.update_memory()
            memory = self.memory[self.method]
            print("Available shots: ", len(memory))
            n_shots = min(self.model_n_shots,len(memory))
            if n_shots:
                print("Using xg shots {}:".format(self.input_features))
                print(memory.loc[:,self.input_features][-n_shots:].values)
                print("Using xe shots: ")
                print(memory.loc[:,["energy"]][-n_shots:].values)
        
            
        find = False
        self.start_time = time.time()
        self.elapsed_time = 0
        self.nodes_expanded = 0
        exception_time = False
        
        
        if self.method == "meta_inf":
            best_solution = None
            eps = 25
            eps_decay = 0.3
            print("Anytime Planner starting search, eps: ", eps)
            INCONS, solution, exception_time = self.Improve_Path(eps)
            if not exception_time and solution is not None:
                print("Solution Found!")
                print(self.retrieve_solution(solution))
                best_solution = solution
                min_fvalue = self.update_OPEN(INCONS, max(1,eps*eps_decay))
                eps1 = min(eps, self.cost_so_far[solution]/max(1e-1,min_fvalue))                
                print("Cost solution: ", self.cost_so_far[solution])
                print("Eps1: ", eps1)
            
                while eps1 > 1:
                    eps = max(1,eps*eps_decay)
                    print("Anytime Planner improve path, eps: ", eps)
                    INCONS, solution, exception_time = self.Improve_Path(eps)
                    if not exception_time and solution is not None:
                        print("Solution Found!")
                        print(self.retrieve_solution(solution))
                        if self.cost_so_far[solution] < self.cost_so_far[best_solution]:
                            best_solution = solution
                        min_fvalue = self.update_OPEN(INCONS, max(1,eps*eps_decay))
                        eps1 = min(eps, self.cost_so_far[solution]/max(1e-1,min_fvalue))
                        print("Cost solution: ", self.cost_so_far[solution])
                        print("Eps1: ", eps1)
                    else:
                        break
            
            if best_solution is not None:
                find = True
                final_state = best_solution
            if exception_time:
                print("Path planner run out of time! Retrieving best available solution...")
                self.update_OPEN(INCONS, 1)
                while not self.OPEN.empty():
                    current_state = self.OPEN.get()
                    if current_state in self.state_info:
                        (p,i) = self.state_info[current_state]
                        if i[1]:
                            if best_solution is None or self.cost_so_far[current_state] < self.cost_so_far[best_solution]:
                                find = True
                                final_state = current_state
                            break                
        else:        
            period_count = 0
            while not self.OPEN.empty() and not find:
                current_state = self.OPEN.get()
                (parent,info) = self.state_info[current_state]
                if info[1]: # if goal is True
                    find = True
                    final_state = current_state
                    break
                
                
                outputs = self.neighbours(current_state)
                for output in outputs:
                    safe, next_state, goal, ida, cost = output
                    if safe:
                        new_cost = self.cost_so_far[current_state] + cost
                        h_cost = self.heuristic(next_state)
                        next_state = self.cont2disc(next_state)
                        if next_state not in self.cost_so_far or new_cost < self.cost_so_far[next_state]:
                            self.cost_so_far[next_state] = new_cost
                            if goal:
                                if self.method == "fastest":
                                    find = True
                                    final_state = next_state
                                priority = new_cost
                            else:
                                priority = new_cost + h_cost
                        
                            self.OPEN.put(next_state, priority)
                            self.state_info[next_state] = (current_state,(ida,goal,cost))
                
                self.elapsed_time = time.time() - self.start_time
                self.nodes_expanded += 1
                if  self.elapsed_time > self.timeout:
                    exception_time = True
                    break
                elif self.elapsed_time >= self.log_time_period*period_count:
                    period_count += 1
                    if self.method != "fastest":
                        print("Elapsed Time: {}s. "
                              "Nodes Expanded: {}. "
                              "Elements in Queue: {}".format(round(self.elapsed_time,3),self.nodes_expanded,len(self.OPEN.elements)))
            if exception_time:
                if self.method != "fastest":
                    print("Path planner run out of time! Retrieving best available solution...")
                    # Look if a solution exists
                    while not self.OPEN.empty():
                        current_state = self.OPEN.get()
                        if current_state in self.state_info:
                            (p,i) = self.state_info[current_state]
                            if i[1]:
                                find = True
                                final_state = current_state
                                break
        
        if self.method != "fastest":
            self.elapsed_time = time.time() - self.start_time
            print("Elapsed Time: {}s. "
                "Nodes Expanded: {}. "
                "Elements in Queue: {}".format(round(self.elapsed_time,3),self.nodes_expanded,len(self.OPEN.elements)))            
                
        if find:
            return self.retrieve_solution(final_state)
        else:
            # print("Path not found")
            return None, None, None
    
        



