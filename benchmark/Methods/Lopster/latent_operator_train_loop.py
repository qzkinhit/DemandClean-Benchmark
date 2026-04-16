import os
import time
import tensorflow as tf
import numpy as np
from transformation_in_x import apply_transformation_in_x, include_errors_at_random
from tensorflow.keras.layers import Concatenate
import copy 

class LatentTrainLoop(object):

    def __init__(self, autoencoder, encoder, decoder, LOP, epochs, optimizer, train_acc_metric, val_acc_metric, transformation):

        self.autoencoder = autoencoder
        self.encoder = encoder
        self.decoder = decoder
        self.LOP = LOP
        self.epochs = epochs
        self.optimizer = optimizer
        self.train_acc_metric = train_acc_metric
        self.val_acc_metric = val_acc_metric
        self.transformation = transformation
        self.train_acc = 0.0
        self.val_acc = 0.0
        self.concatenation = Concatenate()
        self.loss_function = tf.keras.losses.MeanSquaredError()
    
    @tf.function
    def translate_operator_both_ways_indexed(self, inputs):
        zs , Ks_differences = inputs
        column = 0
        
        for k in Ks_differences:
            if k > 0:
                new_z = self.LOP.translate_operator_inverse(zs[column], abs(k))
            else:
                new_z = self.LOP.translate_operator(zs[column], abs(k))

            zs = tf.tensor_scatter_nd_update(zs, [column], new_z)
            column = column + 1

        return zs

    @tf.function
    def custom_loss(self,z,z_p):
        B_reconstruction_loss = self.loss_function(z, z_p)
        return B_reconstruction_loss
    
    @tf.function
    def _trans(self,z1,z2,z3):
        return tf.transpose(z1,[1,0,2]), tf.transpose(z2,[1,0,2]), tf.transpose(z3,[1,0,2])
    
    @tf.function
    def _decode(self,Z):
        return self.concatenation(self.decoder(tf.unstack(Z, axis = 0)))

    @tf.function
    def train_step(self, x, y):
        with tf.GradientTape() as tape:

            K = self.LOP.n_rotations
            K2 = self.LOP.n_left_shifts
            T = self.transformation

            #SHIFT BATCH-WISE====================================
            Z_vectors =  self.encoder(x)
            x_transformed, missing = include_errors_at_random(x, K, 2.0)
            Zs_transformed =  self.encoder(x_transformed)            
            Zs_B = self.encoder(x_transformed)
            
            x_transformed_B, missing_B = include_errors_at_random(x, K, 2.0)

            start = time.time()
            Zs_transformed, Z_vectors, Zs_B = self._trans(Zs_transformed, Z_vectors, Zs_B)
            diffs = missing - missing_B
            Zs_B = tf.vectorized_map(self.translate_operator_both_ways_indexed, [Zs_B, diffs])
            Zs_transformed, Z_vectors, Zs_B = self._trans(Zs_transformed, Z_vectors, Zs_B)
            logits_B  =  self._decode(Zs_B)

            loss_value = self.custom_loss(x_transformed_B, logits_B)

            start = time.time()
            grads = tape.gradient(loss_value, self.autoencoder.trainable_weights)

        start = time.time()
        self.optimizer.apply_gradients(zip(grads,  self.autoencoder.trainable_weights))
        self.train_acc_metric.update_state(x_transformed_B, logits_B)
    
        return loss_value


    @tf.function
    def test_step(self, x, y):
        outputs = self.autoencoder(x, training=False)
        val_logits = outputs[0]
        self.val_acc_metric.update_state(y, val_logits)

    def train_loop(self, train_dataset, val_dataset):
        for epoch in range(self.epochs):

            # Iterate over the batches of the dataset.
            for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
                try:                
                    loss_value = self.train_step(x_batch_train, y_batch_train)
                except tf.errors.InvalidArgumentError: 
                    pass

            # Display and reset metrics at the end of each epoch.
            self.train_acc = self.train_acc_metric.result()
            self.train_acc_metric.reset_state()

            #VALIDATE
            if epoch % 10 == 0:
                for x_batch_val, y_batch_val in val_dataset:
                    self.test_step(x_batch_val, y_batch_val)

                self.val_acc = self.val_acc_metric.result()
                tf.print(self.train_acc,"    ", self.val_acc)
                self.val_acc_metric.reset_state()
            
        for x_batch_val, y_batch_val in val_dataset:
            self.test_step(x_batch_val, y_batch_val)

        self.val_acc = self.val_acc_metric.result()
        self.val_acc_metric.reset_state()
