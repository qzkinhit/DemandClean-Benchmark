import os
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Reshape, Dropout, BatchNormalization, LeakyReLU, Conv2D, MaxPooling2D, Concatenate
from tensorflow.keras import Input, Model, Sequential
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd
from latent_operators import LatentOperator
from transformation_in_x import apply_transformation_in_x
from latent_operator_train_loop import LatentTrainLoop
from copy import deepcopy
from sklearn.metrics import mean_squared_error, accuracy_score

def predict_on_enhanced(x, LOP, encoder, decoder, _translate_all_columns_by_1, GT_K = None):

    K = LOP.n_rotations
    Zs = encoder(x)
    distances = []
    A = tf.transpose(Zs, [1,0,2])
   
    for sf in range(K):
        A = tf.transpose(A, [1,0,2])
        valuesA = tf.squeeze(decoder(tf.unstack(A, axis = 0)))
        distances.append(valuesA)
        A = tf.transpose(A, [1,0,2])
        A = tf.vectorized_map(_translate_all_columns_by_1, A) 
    rotations = (K - 1) -  tf.transpose(tf.argmax(distances, axis = 0), [1,0]) 
    Ks_predicted = rotations
    
    Z_ks = []

    for idx, k in enumerate(Ks_predicted): 
        aux = []
        for i in range(x.shape[1]):
            a = Zs[i][idx]
            aux.append(LOP.translate_operator_inverse(a, shift=k[i]))
        Z_ks.append(aux)
    Z_ks = tf.convert_to_tensor(Z_ks, dtype=tf.float32)
    return Z_ks, Ks_predicted

def eval_correctly(x_test, y_test, Zs, Ks, decoder, classifier):
    #"smart" = avoid decoding Zs when there is no error, just use the value
    xs = tf.squeeze(tf.transpose(decoder(tf.unstack(Zs, axis = 1)), [1,0,2]))
    x_p = tf.where(Ks == 0, x_test, xs)
 
    return classifier.evaluate(x_p, y_test)[1]

def eval_numeric_rmse(x_dirty, Zs, Ks, decoder):
    #"smart" = avoid decoding Zs when there is no error, just use the value
    xs = tf.squeeze(tf.transpose(decoder(tf.unstack(Zs, axis = 1)), [1,0,2]))
    x_p = tf.where(Ks == 0, x_dirty, xs)

    return x_p
