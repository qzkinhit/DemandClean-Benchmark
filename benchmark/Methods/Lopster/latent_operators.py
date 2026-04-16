import tensorflow as tf
import numpy as np
from opt_einsum import contract


def _generate_shift_operator_matrices(n_rotations):
    """Generates family of shift operator matrices"""
    translation_matrix = np.zeros((n_rotations, n_rotations), dtype = np.int32) #Mk

    # fill Mk with 1 on the diagonal +1 position, upper shift matrix
    for i in range(n_rotations):
        translation_matrix[i, int((i + 1) % n_rotations)] = 1 

    translation_matrix = tf.convert_to_tensor(translation_matrix, dtype = tf.float32)

    #starts with the identity transform only
    translation_matrices = [tf.eye(n_rotations, n_rotations, dtype=tf.float32)] 

    # initiate the loop with the I x I
    T = tf.eye(n_rotations, n_rotations, dtype=tf.float32)
    
    #Operator_K, repeats Mk K-1 times
    #This loop applies the Translation Shift Operator K-1 times due to the identity
    #being the first. Next, it saves each incremental shift in a position of the
    #translation_matrices variable.
    for i in range(n_rotations):
        T =  tf.einsum("ij,jk->ik", translation_matrix, T)
        translation_matrices.append(T)

    #convert to tensorflow tensor
    _translation_matrices = tf.convert_to_tensor(translation_matrices, dtype=tf.float32)

    return _translation_matrices






class LatentOperator:
    """K2 must be lower than K, in order to reuse the matrices"""

    def __init__(self, K, n_columns, K2, latent_space_actual_size):

        self.interval = latent_space_actual_size
        self.n_rotations = K
        self.n_left_shifts = K2
        self.shift_matrices = _generate_shift_operator_matrices(K) #K > K2
        self.n_columns = n_columns

        # interval is the arbitrary size of each "bucket", i.e.,  the real size of the latent space
        self.left_shift_matrices = _generate_shift_operator_matrices(K2) #K > K2


    #N_rotation is the number of transformation K
    #SHIFT is the index of the generated transformations
    #Applied PER INSTANCE, not PER BATCH
    def translate_operator(self, z, shift, left_to_right = False):
        """Translate latent

        Args:
            z (1-dim tensor): latent vector
            shift (int): amount by which to shift.
                shift of 0 corresponds to the identity.
            left_to_right: go sideways instead of up and down translations.
        """

        # reshape into 2D tensor, n_rotations +1 to include identity transformation
        z_2d = tf.reshape(z,(self.n_rotations, -1))
        
        # if left_to_right:
        #     translation_matrix = self.left_shift_matrices[shift * self.interval]
            
            
        #     #z_2d_shifted = tf.einsum("ij,jk->ik", z_2d, translation_matrix, optimize=True) # Y RIGHT
        #     z_2d_shifted = tf.matmul(z_2d, translation_matrix) # Y RIGHT
            
        #     #PRINT transform
        #     # tf.print("\n=============\n", shift,"\n Z",
        #     #          z_2d[1,:] * 1000, "\n T",
        #     #          z_2d_shifted[1,:] * 1000,
        #     #          summarize = -1)


        # else:
        translation_matrix = self.shift_matrices[shift]
        #            z_2d_shifted = tf.einsum("ij,jk->ik", translation_matrix, z_2d, optimize=True) # Y UP
        z_2d_shifted = tf.matmul(translation_matrix, z_2d) # Y UP
          
            


        # OPs = self.matrices[1]
        # tf.print("===============\n"
        #         ,OPs[0],
        #         "\n-\n",OPs[1],
        #         "\n-\n",OPs[2],
        #         "\n-\n",OPs[3],
        #         "\n-\n",OPs[4],
        #         "\n-\n",OPs[5],
        # summarize = -1)

        
        # reshape back
        z_shifted = tf.reshape(z_2d_shifted,z.shape)
        
        return z_shifted



    def translate_operator_inverse(self, z, shift, left_to_right = False):
        # reshape into 2D tensor, n_rotations +1 to include identity transformation
        z_2d = tf.reshape(z,(self.n_rotations, -1))

        # if left_to_right:
        #     translation_matrix = self.left_shift_matrices[shift * self.interval]
            
        #     #z_2d_shifted = tf.einsum("ij,kj->ik",  z_2d, translation_matrix, optimize =True) # Y LEFT
        #     z_2d_shifted = tf.linalg.matmul(z_2d, translation_matrix, transpose_b=True) # Y LEFT
            

        #     # #PRINT transform
        #     # tf.print("\n=============\n", shift,"\n Z",
        #     #          z_2d[1,:] * 1000, "\n T",
        #     #          z_2d_shifted[1,:] * 1000,
        #     #          summarize = -1)
            

        # else:
        translation_matrix = self.shift_matrices[shift]
        #z_2d_shifted = tf.einsum("ji,jk->ik", translation_matrix, z_2d, optimize=True) # Y DOWN
        z_2d_shifted = tf.linalg.matmul(translation_matrix, z_2d, transpose_a=True) # Y DOWN
            

        z_shifted = tf.reshape(z_2d_shifted, z.shape)
        
        return z_shifted





#1D+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def translate_operator_1D(self, z, shift, left_to_right = False):

        z = tf.expand_dims(z, axis = 1)
        
        if left_to_right:
            translation_matrix =  self.left_shift_matrices[shift * self.interval]
            z_shifted = tf.matmul(z, translation_matrix) # Y RIGHT
        else:
            translation_matrix =  self.shift_matrices[shift]
            z_shifted = tf.matmul(translation_matrix, z) # Y UP

            
        return z_shifted



    def translate_operator_inverse_1D(self, z, shift, left_to_right = False):

        z = tf.expand_dims(z, axis = 1)
        
        if left_to_right:
            translation_matrix = self.left_shift_matrices[shift * self.interval]
            z_shifted = tf.linalg.matmul(z, translation_matrix, transpose_b=True) # Y LEFT
        else:
            translation_matrix = self.shift_matrices[shift]
            z_shifted = tf.linalg.matmul(translation_matrix, z, transpose_a=True) # Y DOWN

            
        return z_shifted
