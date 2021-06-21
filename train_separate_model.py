import os
import pandas as pd
import random
import numpy as np
import tensorflow as tf
from datetime import datetime
import models
os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(0)
np.random.seed(0)
random.seed(0)

params = {}

# To set with path to dataset
params["path_data"] = []
params["path_data"].append("./Dataset/Exp00/data.csv")

params["EPOCHS"] = 8000
params["MODEL"] = "model_single"

params["LOSS"] = 'mse'
params["LEARNING_RATE"] = 1e-3

current_time = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
params["LOG_DIR"] = "./Training/Exp_{}/log_separate_model/".format(current_time)

params["INPUT_FEATURES"] = ["mean_pitch_est", "mean_roll_est", "std_pitch_est", "std_roll_est"]

# Create Directories
if not os.path.exists(params["LOG_DIR"]):
    os.makedirs(params["LOG_DIR"])

def create_batch(df):
    batch = df.sample(n=params["BATCH_SIZE"])
    XG = batch.loc[:,params["INPUT_FEATURES"]].values
    YE = batch.loc[:,["energy"]].values
            
    return XG, YE
        
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
    
    
    terrain_ids = list(df["terrain_id"].drop_duplicates().values)
    for terrain_id in terrain_ids:
        params["TERRAIN_ID"] = terrain_id
        dfi = df[df["terrain_id"]==params["TERRAIN_ID"]]
        
        print("Samples: {}".format(len(dfi)))
        
        if params["LOSS"] == 'mse':
            loss_object = tf.keras.losses.MeanSquaredError()
        elif params["LOSS"] == 'mae':
            loss_object = tf.keras.losses.MeanAbsoluteError() 
        optimizer = tf.keras.optimizers.RMSprop(lr=params["LEARNING_RATE"])
        
        train_loss = tf.keras.metrics.Mean(name='train_loss')     
        
        LOG_NAME = "{}_tid_{}/".format(current_time,terrain_id)
        TRAIN_LOG_DIR = params["LOG_DIR"]+LOG_NAME+"train/"
        
        train_summary_writer = tf.summary.create_file_writer(TRAIN_LOG_DIR)
        
        params["BATCH_SIZE"] = len(dfi)
        model = models.get_model(params["MODEL"],None,None, len(params["INPUT_FEATURES"]), params["BATCH_SIZE"], summary=True)
        
        
        file = open(params["LOG_DIR"]+LOG_NAME+"log_params_{}.txt".format(current_time),"w+")
        for key, val in params.items():
            file.write("{}: {}\n".format(key,val))
        file.close()
        
        for epoch in range(params["EPOCHS"]):
            if epoch>300:
                optimizer = tf.keras.optimizers.RMSprop(lr=params["LEARNING_RATE"]/10)
            ## Training
            train_loss.reset_states()
            for train_step in range(int(len(dfi)/params["BATCH_SIZE"])):
                XG, YE = create_batch(dfi)
                
                with tf.GradientTape() as tape:
                    YE_P = model(XG, training=True)
                    L_train = loss_object(YE,YE_P)
                gradients = tape.gradient(L_train,model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                train_loss(L_train)
            print(
                f'Terrain {terrain_id}, '
                f'Epoch {epoch + 1}, '
                f'Train Loss: {train_loss.result()}'
              )
            with train_summary_writer.as_default():
                tf.summary.scalar('train_loss', train_loss.result(), step=epoch)
                
        model.save_weights(params["LOG_DIR"]+LOG_NAME+"model_terrain_" + str(params["TERRAIN_ID"]) +".h5")
            
            
                
        
    


if __name__ == "__main__":
    main()