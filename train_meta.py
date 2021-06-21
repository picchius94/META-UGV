import os
import pandas as pd
import random
import numpy as np
import tensorflow as tf
from datetime import datetime
import models
from sklearn.metrics import r2_score
os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(0)
np.random.seed(0)
random.seed(0)

n_combs = 10 # number of random training-validation trials

params = {}

# To set with path to dataset
params["path_data"] = []
params["path_data"].append("./Dataset/Exp00/data.csv")

params["FREQ_VAL"] = 2
params["EPOCHS"] = 61
params["BATCH_SIZE"] = 32
params["N_ITERATIONS_TRAIN"] = 1
params["N_ITERATIONS_VAL"] = 0.5
params["MODEL"] = "model04_pos"
params["EXTRA_VAR"] = None

params["LOSS"] = 'mse'
params["LEARNING_RATE"] = 1e-4

params["N_SHOTS"] = 3
params["N_META"] = 1

params["SHOTS_WEIGHTS"] = [0,1,1,1]

params["TRAIN_PERC"] = 1

current_time = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
params["LOG_DIR"] = "./Training/Exp_{}/log_meta/".format(current_time)

params["INPUT_FEATURES"] = ["mean_pitch_est", "mean_roll_est", "std_pitch_est", "std_roll_est"]

CATEGORIES = []
CATEGORIES.append([18, 19, 20, 21]) # Very loose frictional
CATEGORIES.append([7, 8, 10, 12]) # Clay high moisture content
CATEGORIES.append([2, 5, 0, 22]) # Loose frictional
CATEGORIES.append([1, 3, 4, 13, 14, 15, 16, 17]) # Compact frictional
CATEGORIES.append([9, 11]) # Dry clay

params["TRAINING_DATASETS_PER_CATEGORY"] = [1, 1, 0, 1, 1]

# Create Directories
if not os.path.exists(params["LOG_DIR"]):
    os.makedirs(params["LOG_DIR"])

def batch_type(dfi):
    # Shots and meta are random
    row = dfi.sample(n=params["N_SHOTS"]+params["N_META"])
    row_shots = row.sample(n=params["N_SHOTS"])
    row_meta = row.drop(row_shots.index)
    
    return row_shots, row_meta
        
        

def create_batch(df):
    for i in range(params["BATCH_SIZE"]):
        terrain_id = random.sample(list(df["terrain_id"].drop_duplicates().values),1)
        dfi = df[df["terrain_id"].isin(terrain_id)]
        
        row_shots, row_meta = batch_type(dfi)
        
        xg_shots = row_shots.loc[:,params["INPUT_FEATURES"]].values
        xe_shots = row_shots.loc[:,["energy"]].values
        xg_meta = row_meta.loc[:,params["INPUT_FEATURES"]].values
        ye_meta = row_meta.loc[:,["energy"]].values
        
        if not i:
            XG_SHOTS = np.expand_dims(xg_shots,axis=0)
            XE_SHOTS = np.expand_dims(xe_shots,axis=0)
            XG_META = np.expand_dims(xg_meta,axis=0)
            YE_META = np.expand_dims(ye_meta,axis=0)
        else:
            XG_SHOTS = np.concatenate([XG_SHOTS,np.expand_dims(xg_shots,axis=0)],axis=0)
            XE_SHOTS = np.concatenate([XE_SHOTS,np.expand_dims(xe_shots,axis=0)],axis=0)
            XG_META = np.concatenate([XG_META,np.expand_dims(xg_meta,axis=0)],axis=0)
            YE_META = np.concatenate([YE_META,np.expand_dims(ye_meta,axis=0)],axis=0)
            
    return XG_SHOTS, XE_SHOTS, XG_META, YE_META


def tot_mape(y_true, y_pred):
    sum_true = np.sum(y_true)
    sum_pred = np.sum(y_pred)
    return abs((sum_true-sum_pred)/(sum_true+1e-6))
    
def main():
    df = pd.DataFrame()
    for path_data in params["path_data"]:
        dfi = pd.read_csv(path_data)
        df = pd.concat([df,dfi])
    ## Removing some of data:
    # Samples without low mean speed
    df = df[df.mean_speed>0.87]
    # Samples without low initial speed
    df = df[df.initial_speed>0.88]
    # Samples without rough pitch/roll variations
    df = df.loc[(df.var_pitch_est <=2.25) | (df.var_roll_est <=2.25)]
    df["std_pitch_est"] = df["var_pitch_est"].pow(0.5)
    df["std_roll_est"] = df["var_roll_est"].pow(0.5)
    
    for comb in range(n_combs):
        current_time = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        LOG_NAME = "{}/".format(current_time)
        TRAIN_LOG_DIR = params["LOG_DIR"]+LOG_NAME+"train/"
        TEST_LOG_DIR = params["LOG_DIR"]+LOG_NAME+"test/"
        
        # Selection of terrains for training and validation by category
        id_val = []
        id_train = []
        for cat in range(len(CATEGORIES)):
            t_ids = CATEGORIES[cat]
            id_train.extend(random.sample(t_ids, params["TRAINING_DATASETS_PER_CATEGORY"][cat]))
            id_val.extend([t for t in t_ids if t not in id_train])
        
        params["TERRAIN_IDS_TRAIN"] = id_train
        params["TERRAIN_IDS_VAL"] = id_val
        df_train = df[df["terrain_id"].isin(id_train)]
        df_val = df[df["terrain_id"].isin(id_val)]
        if params["TRAIN_PERC"] < 1:
            df_train = df_train.sample(frac=params["TRAIN_PERC"])
        
        print()
        print()
        print("Samples: {}".format(len(df)))
        print("Training Samples: {}".format(len(df_train)))
        print("Validation Samples: {}".format(len(df_val)))
        print("Training Terrains {}".format(id_train))
        print("Validation Terrains {}".format(id_val))
        
        
        # Loss Function and Metrics
        if params["LOSS"] == 'mse':
            loss_object = tf.keras.losses.MeanSquaredError()
        elif params["LOSS"] == 'mae':
            loss_object = tf.keras.losses.MeanAbsoluteError() 
        optimizer = tf.keras.optimizers.RMSprop(lr=params["LEARNING_RATE"])
        
        train_loss = tf.keras.metrics.Mean(name='train_loss')
        val_loss = []
        val_tot_mape = []
        val_r2_score = []
        for i in range(params["N_SHOTS"]+params["N_META"]+1):
            if params["LOSS"] == 'mse':
                val_loss.append(tf.keras.metrics.MeanSquaredError(name='val_loss_{}'.format(i)))
            else:
                val_loss.append(tf.keras.metrics.MeanAbsoluteError(name='val_loss_{}'.format(i)))  
            val_tot_mape.append(tf.keras.metrics.Mean(name='val_tot_mape_{}'.format(i)))
            val_r2_score.append(tf.keras.metrics.Mean(name='val_r2_score_{}'.format(i)))
        
        train_summary_writer = tf.summary.create_file_writer(TRAIN_LOG_DIR)
        test_summary_writer = tf.summary.create_file_writer(TEST_LOG_DIR)
        
        sample_weights = np.ones((params["BATCH_SIZE"],params["N_SHOTS"]+params["N_META"]))
        for i,w in enumerate(params["SHOTS_WEIGHTS"]):
            sample_weights[:,i] = np.ones((params["BATCH_SIZE"],))*w
        
        # Model Loading
        model = models.get_model(params["MODEL"],params["N_SHOTS"],params["N_META"],len(params["INPUT_FEATURES"]),params["BATCH_SIZE"], var = params["EXTRA_VAR"], summary=True)
        
        # Saving experiment hyperparams to txt file
        file = open(params["LOG_DIR"]+LOG_NAME+"log_params_{}.txt".format(current_time),"w+")
        for key, val in params.items():
            file.write("{}: {}\n".format(key,val))
        file.close()
        
        # Start Training Loop
        min_val_loss = 1e5
        for epoch in range(params["EPOCHS"]):
            ## Training
            train_loss.reset_states()
            for train_step in range(int(len(df_train)/params["BATCH_SIZE"]*params["N_ITERATIONS_TRAIN"])):
                XG_SHOTS, XE_SHOTS, XG_META, YE_META = create_batch(df_train)
                YE_TOT = np.concatenate((XE_SHOTS,YE_META),axis=1)
                with tf.GradientTape() as tape:
                    YE_P_TOT = model(XG_SHOTS, XE_SHOTS, XG_META, training=True)
                    L_train = loss_object(YE_TOT,YE_P_TOT, sample_weight=sample_weights)
                gradients = tape.gradient(L_train,model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                train_loss(L_train)
            print(
                f'Epoch {epoch + 1}, '
                f'Train Loss: {train_loss.result()}'
              )
            with train_summary_writer.as_default():
                tf.summary.scalar('train_loss', train_loss.result(), step=epoch)
            ## Validation
            if not epoch%params["FREQ_VAL"]:
                for i in range(len(val_loss)):
                    val_loss[i].reset_states()
                    val_tot_mape[i].reset_states()
                    val_r2_score[i].reset_states()
                for val_step in range(int(len(df_val)/params["BATCH_SIZE"]*params["N_ITERATIONS_VAL"])):
                    XG_SHOTS, XE_SHOTS, XG_META, YE_META = create_batch(df_val)
                    YE_TOT = np.concatenate((XE_SHOTS,YE_META),axis=1)
                    YE_P_TOT = model(XG_SHOTS, XE_SHOTS, XG_META, training=False)
                    for i in range(len(val_loss)):
                        if i < len(val_loss)-1:
                            val_loss[i](YE_TOT[:,i],YE_P_TOT[:,i])
                            val_tot_mape[i](tot_mape(YE_TOT[:,i],YE_P_TOT[:,i]))
                            val_r2_score[i](r2_score(tf.reshape(YE_TOT[:,i],[-1]),tf.reshape(YE_P_TOT[:,i],[-1])))
                        else:
                            val_loss[i](YE_TOT[:,1:],YE_P_TOT[:,1:])
                            val_tot_mape[i](tot_mape(YE_TOT[:,1:],YE_P_TOT[:,1:]))
                            val_r2_score[i](r2_score(tf.reshape(YE_TOT[:,1:],[-1]),tf.reshape(YE_P_TOT[:,1:],[-1])))
                for i in range(len(val_loss)):
                    if i < len(val_loss)-1:
                        print("Val loss {}: {}".format(i,val_loss[i].result()))
                        with test_summary_writer.as_default():
                            tf.summary.scalar('val_loss_{}'.format(i), val_loss[i].result(), step=epoch)
                        print("Val tot_mape {}: {}".format(i,val_tot_mape[i].result()))
                        with test_summary_writer.as_default():
                            tf.summary.scalar('val_tot_mape_{}'.format(i), val_tot_mape[i].result(), step=epoch)
                        print("Val r2_score {}: {}".format(i,val_r2_score[i].result()))
                        with test_summary_writer.as_default():
                            tf.summary.scalar('val_r2_score_{}'.format(i), val_r2_score[i].result(), step=epoch)
                    else:
                        print("Val loss Tot: {}".format(val_loss[i].result()))
                        with test_summary_writer.as_default():
                            tf.summary.scalar('val_loss_tot', val_loss[i].result(), step=epoch)
                        print("Val tot_mape Tot: {}".format(val_tot_mape[i].result()))
                        with test_summary_writer.as_default():
                            tf.summary.scalar('val_tot_mape_tot', val_tot_mape[i].result(), step=epoch)
                        print("Val r2_score Tot: {}".format(val_r2_score[i].result()))
                        with test_summary_writer.as_default():
                            tf.summary.scalar('val_r2_score_tot', val_r2_score[i].result(), step=epoch)
                        if val_loss[i].result() < min_val_loss:
                            min_val_loss = val_loss[i].result()
                            model.save_weights(params["LOG_DIR"]+LOG_NAME+"model_best.h5")
                
                
                
        
    


if __name__ == "__main__":
    main()
