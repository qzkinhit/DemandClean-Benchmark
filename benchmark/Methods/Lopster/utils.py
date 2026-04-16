import os
import numpy as np
import tensorflow as tf
import preprocess
from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.layers import Dense, Concatenate
from tensorflow.keras.optimizers import Adam
from latent_operators import LatentOperator
from transformation_in_x import apply_transformation_in_x
from latent_operator_train_loop import LatentTrainLoop


def create_and_train_LOP(train_dataset, val_dataset, x_dim, z_dim, K, K2, T, epochs= 100, lr = 0.001, model_name="test"):
 
    base_path = f'./MODELS/{model_name}/{T}_{x_dim}_{z_dim}_{K}_epochs_{epochs}/'

    input_tuple = Input(shape=(x_dim,))
    latent_vectors = []    
    for i in range(x_dim):
        z = Dense(z_dim, use_bias=False, activation="linear")(input_tuple)
        latent_vectors.append(z)
        
    encoder = Model(inputs = input_tuple, outputs = latent_vectors)

    column_operators = []
    for i in range(x_dim):
        #d_bt = Dense(128, activation = "relu")(latent_vectors[i])
        d_bt = Dense(128, activation = "tanh")(latent_vectors[i])
        
        d_out = Dense(1)(d_bt) #1 LOP per column
        column_operators.append(d_out)

    decoder = Model(inputs = latent_vectors,
                    outputs = column_operators,
                    name="my_latops")
    
    concatenation  =  Concatenate()(column_operators)
    latent_vectors =  Concatenate()(latent_vectors)
    interval_size = 1
    
    if not os.path.isdir(base_path):
        optimizer = Adam(learning_rate=lr, epsilon=1e-06)
        train_acc_metric = tf.keras.metrics.MeanSquaredError()
        val_acc_metric = tf.keras.metrics.MeanSquaredError()
        autoencoder = Model(inputs = input_tuple, outputs = [concatenation, latent_vectors])

        #MANUAL TRAIN LOOP======================================
        LOP = LatentOperator(K, x_dim, K2, interval_size)
        L = LatentTrainLoop(autoencoder, encoder, decoder, LOP, epochs, optimizer, train_acc_metric, val_acc_metric, T)
        L.train_loop(train_dataset, val_dataset)
        
        os.makedirs(base_path)
        #encoder.save_weights(f'{base_path}encoder.ckpt')
        #decoder.save_weights(f'{base_path}decoder.ckpt')

        encoder.save(f'{base_path}encoder.keras')
        decoder.save(f'{base_path}decoder.keras')
        print(encoder.summary())
        return encoder, decoder, LOP, float(L.train_acc), float(L.val_acc)
    else:
        encoder.load_weights(f'{base_path}encoder.keras')
        decoder.load_weights(f'{base_path}decoder.keras')
        
        LOP = LatentOperator(K, x_dim, K2, interval_size)
        return encoder, decoder, LOP, 0.0, 0.0


def create_and_train_classifier(train_dataset, val_dataset, inp_dim, z_dim, T, n_epochs= 100, lr = 0.001, model_name="test"):
    base_path = f'./MODELS/{model_name}/Classifier_{T}_{z_dim}_{n_epochs}/'
    nnreg_model = Sequential()
    nnreg_model.add(Dense(128, kernel_initializer='normal', activation='relu', input_dim = inp_dim))
    nnreg_model.add(Dense(128, kernel_initializer='normal', activation='relu'))
    nnreg_model.add(Dense(1))
    nnreg_model.compile(optimizer='adam',loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])

    if not os.path.isdir(base_path):
        os.makedirs(base_path)
        nnreg_model.fit(train_dataset, epochs = n_epochs, verbose = 0)
        nnreg_model.save(f'{base_path}classifier.keras')
    else:
        nnreg_model.load_weights(f'{base_path}classifier.keras')

    return nnreg_model    
