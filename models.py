import tensorflow as tf
from tensorflow.keras.layers import Dense, Concatenate
from tensorflow.keras.layers import LSTM, TimeDistributed, Cropping1D


def get_model(model_name, n_shots, n_meta, n_features = 2, b_size = 16, var = None, summary=False):
    """ Return the network model
    """
    tf.keras.backend.set_floatx('float64')
    model = None
          
    if model_name=="model_single":
        xg_shape = (n_features,)
        class MyModel(tf.keras.Model):
            def __init__(self, name="model",**kwargs):
                super(MyModel, self).__init__(name=name,**kwargs)
                self.params = {}
                self.params['fc1'] = Dense(16,activation="relu", name = 'fc1')
                self.params['fc2'] = Dense(8,activation="relu", name = 'fc2')
                self.params['fc3'] = Dense(1,activation="linear", name = 'fc3')
            def call(self, x, training=False):
                x = self.params['fc1'](x)
                x = self.params['fc2'](x)
                x = self.params['fc3'](x)
                
                return x
        model = MyModel()
        model(tf.zeros((b_size,)+xg_shape))
    
    elif model_name == "model04_pos":
        xg_shot_shape = (n_shots,n_features)
        xe_shot_shape = (n_shots,1)
        xg_meta_shape = (n_meta,n_features)
        class MyModel(tf.keras.Model):
            def __init__(self, name="model",**kwargs):
                super(MyModel, self).__init__(name=name,**kwargs)
                self.params = {}
                self.params['LSTM'] = LSTM(128,return_sequences=True)
                self.params['fc2'] = Dense(64,activation="relu", name = 'fc2')
                self.params['fc3'] = Dense(1,activation="relu", name = 'fc3')
            def call(self, xg_shot, xe_shot, xg_meta, training=False):            
                x_shot_tot = Concatenate(axis=-1)([xg_shot,xe_shot])
                xg_shot_no_first = Cropping1D(cropping=(1,0))(xg_shot)
                xg_meta_tot = Concatenate(axis=1)([xg_shot_no_first,xg_meta])
                xe_0 = tf.zeros((tf.shape(xe_shot)[0],)+(1,1),dtype=tf.dtypes.float64)
                
                h = self.params['LSTM'](x_shot_tot)
                h_tot = Concatenate(axis=-1)([xg_meta_tot,h])
                x = TimeDistributed(self.params['fc2'])(h_tot)
                x = TimeDistributed(self.params['fc3'])(x)
                x = Concatenate(axis=1)([xe_0,x])
                
                return x
        model = MyModel()
        model(tf.zeros((b_size,)+xg_shot_shape),tf.zeros((b_size,)+xe_shot_shape),tf.zeros((b_size,)+xg_meta_shape))
     
    
    
        
        
        
    if model is None:
        print("Not valid model name")
        raise ValueError
    elif summary:
        model.summary()
    
    return model