import tensorflow as tf
import numpy as np
import math

@tf.function
def apply_transformation_in_x(x, shift, shift2, K, K2, T):

    if T == 'tabular_scaling':
        return _tabular_scaling(x,shift)
    elif T == 'image_rotation':
        return _image_rotation(x, shift, K)
    elif T == 'missing_values':
        return _missing_values(x,shift, K)
    elif T == 'proportional_noise':
        return _proportional_noise(x, shift, shift2, K2)
    else:
        return _translate_only(x,shift)

def _translate_only(x,shift): pass

@tf.function
def _tabular_scaling(x,shift):

    if shift == 0: diagonal = [1.,1.,1.,1.,1.,1.,1.,1.]
    elif shift == 1: diagonal = [1.,1.6,1.4,1.,1.,1.,1.,1.]
    elif shift == 2: diagonal = [1.,1.,1.,1.,1.5,1.,1.,1.]
    elif shift == 3: diagonal = [1.,1.,1.,1.,1.,1.,1.3,1.3]
    elif shift == 7: diagonal = [1.,1.,1.,1.,1.,1.,1.70,1.70] # test for starting on a non anchor K
    else: diagonal = [1.,1.,1.,1.,1.,1.,1.,1.]
        
    return x * diagonal  

#The data is in the iterval 0,1 so -1 is a valid "NULL"
#To train on a Benchmark just remove the rows with nmissing values, train, then predict completion for the evaluation
@tf.function
def _missing_values(x, shift, K):
    if int(shift) == 0: 
        return x 
    else:   
        x_missing = tf.tensor_scatter_nd_update(x, [[shift-1]], [-1])
        return x_missing


#The data is in the iterval 0,1 so after nbosie it must stay in the same place
@tf.function
def _proportional_noise(x, shift, shift2, K2):

    if int(shift) == 0:
        return x 
    else:
        if shift2 < 5:
            a = 0.25 * (5 - float(shift2))
        else:
            a = 0.25 * (float(shift2) - 4)
            
        #half of K2 for negative and half for positive
        if int(shift2) <= int(K2/2):
            noise = -1 * (x[shift-1] * a)
        else: noise = (x[shift-1] * a)
       
        x_missing = tf.tensor_scatter_nd_add(x, [[shift-1]], [noise])
        return x_missing


@tf.function
def include_errors_at_random(x, K, factor):
    aux = np.ones((x.shape[0], K)) + 1
    aux[:,0] = 2.0 * factor #increase chance of no error to be factor fold
    logits = tf.math.log(aux)
    missing = tf.random.categorical(logits, x.shape[1], dtype = tf.int32)

    #enable to scal 0 valued columns
    x_transformed = tf.where(x == 0, x + 1e-10, x)

    #ideal intervals to avoid x = x*1,
    #can run into problems as soon as K >= 20
    increment  = 1 / (K/2)
    aux = 1
    
    for i in range(K):
        if i == K - 1:#last must be the MV position
            x_transformed = tf.where(missing == i, 3.0, x_transformed)
        else:
            step = (i + aux) * increment
            if 0.9 <= step <= 1.10:#too close to the original value
                aux = aux + 1
                step = (i + aux) * increment

            x_transformed = tf.where(missing == i + 1, x_transformed * step, x_transformed)

    #transforme back 0 valued columns
    x_transformed = tf.where(x == 0, x_transformed - 1e-10, x_transformed) #abs otherwise 0.5 - 1 = -0.5

    return x_transformed, missing

