import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import csv
import sys
import matplotlib
import matplotlib.pyplot as plt
import time
import tensorflow as tf
import cleaners as cln
import numpy as np
import argparse
import pandas as pd
import seaborn as sns
from pathlib import Path

from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.layers import Dense, Concatenate
from datasets import get_tf_database, load_regression, load_regression_dirty, load_features_and_data, reverse_categorical_columns, reverse_to_input_domain, prepare_data_subset
from latent_operators import LatentOperator
from transformation_in_x import apply_transformation_in_x, include_errors_at_random
from utils import create_and_train_LOP, create_and_train_classifier
from error_detection import predict_on_enhanced, eval_correctly, eval_numeric_rmse
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score, precision_score, recall_score
from copy import deepcopy
from plotter import maximize_plot

parser = argparse.ArgumentParser(description="Disentangled Latent Space Operator for Data Engineering")
parser.add_argument("--epochs", type=int, default=80)
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--latent", type=int, default=120)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--K", type=int, default=12)
parser.add_argument("--dataset", default='nasa')
parser.add_argument("--eval_tuples", type=int, default=-1)


args = parser.parse_args()

np.set_printoptions(threshold=sys.maxsize)
print("Is Eager execution?",tf.executing_eagerly())
LOP = None

@tf.function
def _translate_all_columns_by_1(inputs):
    zs  = inputs
    for column in range(zs.shape[0]):
        new_z = LOP.translate_operator(zs[column], 1)
        zs = tf.tensor_scatter_nd_update(zs, [column], new_z)
    return zs

def generate_cleaned_data(x_, Zs, Ks, decoder):
    #avoid decoding Zs when there is no error, just use the value
    xs = tf.squeeze(tf.transpose(decoder(tf.unstack(Zs, axis = 1)), [1,0,2]))
    x_p = tf.where(Ks == 0, x_, xs) 
    return x_p

@tf.function
def _translate_one_column_by_1(inputs, col):
    zs  = inputs
    for column in range(zs.shape[0]):
        if column == col:
            new_z = LOP.translate_operator(zs[column], 1)
        else:
            new_z = zs[column]            
        #replace the column values in the tensor
        zs =  tf.tensor_scatter_nd_update(zs, [column], new_z)
    return zs
    



#CONFIGS========================================================
ds = args.dataset
T_EXAMPLES = args.eval_tuples
V_EXAMPLES = args.eval_tuples 
T_EVAL_EXAMPLES = args.eval_tuples
V_EVAL_EXAMPLES = args.eval_tuples
K_EPOCHS = args.epochs
MISSING_REPLACE = '3.0'
T = 'missing_values'
K = args.K

#LOAD AIRBNB, NASA or BIKE=====================================
start = time.time()

#MAX AND MIN FOR THE ALREADY FILTERED SET + TARGET
x_clean_train, y_clean_train, x_clean, y_clean, MAX, MIN, SCALER, CAT_ENCODER = load_regression(ds, T_EVAL_EXAMPLES, V_EVAL_EXAMPLES, True, normalize_sklearn = True)
x_dirty_train, y_dirty_train, x_test, y_test = load_regression_dirty(ds, T_EVAL_EXAMPLES, V_EVAL_EXAMPLES, MISSING_REPLACE, SCALER, CAT_ENCODER, True, MAX, MIN, normalize_sklearn = True)

print(y_test.shape)
COLS = x_clean_train.shape[1]
print("# COLUMNS:", COLS)

train_dataset = get_tf_database(x_clean_train[0:T_EXAMPLES],
                                x_clean_train[0:T_EXAMPLES],
                                args.batch_size)

val_dataset = get_tf_database(x_clean[0:V_EXAMPLES],
                              x_clean[0:V_EXAMPLES],
                              args.batch_size)
print(time.time() - start)


# TRAIN MODELS========================================================
start = time.time()
encoder, decoder, LOP, t_acc, v_acc = create_and_train_LOP(train_dataset,
                                                           val_dataset,
                                                           COLS,
                                                           args.latent,
                                                           K,
                                                           1,
                                                           T,
                                                           epochs=args.epochs,
                                                           model_name=args.dataset)
print("LOP ",time.time() - start, "sec", "Train loss:", t_acc," VAL loss:" , v_acc)
  
concatenation_model = Concatenate()(decoder.output)


############ WRITE CLEAN DATA #################
headers,  target_name, dirty_data, _, dirty_data_with_y, _, clean_data, FULL_SCALER, CAT_ENCODER = load_features_and_data(ds, T_EVAL_EXAMPLES, V_EVAL_EXAMPLES, MISSING_REPLACE, SCALER, CAT_ENCODER, True, MAX, MIN, normalize_sklearn = True)
full_header = headers["full_header"]
header_with_y = headers["filtered_header_with_y"]
filtered_header = headers["filtered_header"]
numeric_header = headers["numeric_header"] 
categorical_header = headers["categorical_header"]


## REPAIR DATA ###############################################
def generate_qualitative_data(Zs, decoder, transpose = False):
    if transpose:
        Zs = tf.transpose(Zs, [1,0,2])
        return tf.transpose(tf.squeeze(decoder(tf.unstack(Zs, axis = 1))),[1,0])
    else:
        return tf.squeeze(decoder(tf.unstack(Zs, axis = 1)))

cols_to_change = [6, 7]
input_domain_data = []
n_columns = clean_data.shape[1]
z_list = []

#check if the experiment exists
quali_file = Path(f'./evaluation/repaired_data_{ds}_{len(cols_to_change)}.csv')
if not quali_file.is_file():
    #encode and decode the original data
    Zs = encoder(tf.convert_to_tensor(clean_data, dtype=tf.float32))
    repaired_data = generate_qualitative_data(Zs, decoder, True)
    repaired_data = pd.DataFrame(repaired_data.numpy(), columns = clean_data.columns)

    #moves a column by K steps#########################
    for c in range(clean_data.shape[1]):
        aux = []
        for r in range(clean_data.shape[0]):
            a = Zs[c][r]
            if c in cols_to_change:
                aux.append(LOP.translate_operator(a, shift=11))
            else:
                aux.append(a)
        z_list.append(tf.expand_dims(tf.convert_to_tensor(aux, dtype=tf.float32), axis = 0))

    Zs_shifted = tf.squeeze(tf.convert_to_tensor(z_list, dtype=tf.float32))

    #project the shifted data in the data space
    shifted_data = generate_qualitative_data(Zs_shifted, decoder, True)
    shifted_data = pd.DataFrame(shifted_data.numpy(), columns = clean_data.columns)

    #encode the shifted data again
    Zs_final = encoder(tf.convert_to_tensor(shifted_data[filtered_header], dtype=tf.float32))

    #get final results for the shifted data
    shifted_data = generate_qualitative_data(Zs_shifted, decoder, True)
    shifted_data = pd.DataFrame(shifted_data.numpy(), columns = clean_data.columns)

    shifted_data.to_csv(f'./evaluation/shifted_data_{ds}_{len(cols_to_change)}.csv', index= False)
    repaired_data.to_csv(f'./evaluation/repaired_data_{ds}_{len(cols_to_change)}.csv', index = False)


else:
    shifted_data = pd.read_csv(f'./evaluation/shifted_data_{ds}_{len(cols_to_change)}.csv')
    repaired_data = pd.read_csv(f'./evaluation/repaired_data_{ds}_{len(cols_to_change)}.csv')
###############################################################################

#get the actual values to compare
pd.set_option('display.max_columns', None)
print(reverse_to_input_domain(args.dataset, repaired_data, FULL_SCALER, CAT_ENCODER).head(), "\n\n", reverse_to_input_domain(args.dataset, shifted_data, FULL_SCALER, CAT_ENCODER).head())


#too many columns to show
#clean_data = clean_data.iloc[:, 1:8]
repaired_data = repaired_data.iloc[:, 1:8]
shifted_data = shifted_data.iloc[:, 1:8]

plt.rc('font', size=14)

fig, ax1 = plt.subplots()


if len(cols_to_change) == 2 :
    #separator for 2 shifts
    ax1.axvline(ax1.get_xticks()[2] + 4.1, color='k', linestyle =  '--')
elif len(cols_to_change) == 4 :
    #separator for 4 shifts
    ax1.axvline(ax1.get_xticks()[5] + 1.5, color='k', linestyle =  '--')

sns.boxplot(data = pd.concat([repaired_data, shifted_data], keys=('Original', 'Reconstructed')).stack().rename_axis(index=['Dataset', '', 'Column labels']).reset_index(level=[0,2], name='Column values'), x='Column labels', hue='Dataset', y='Column values', showfliers = False, palette = "Set2", width=0.3, linewidth= 0.8, showcaps = False, whis = 0, ax= ax1)

maximize_plot()
plt.setp(ax1.get_xticklabels(), rotation=30, horizontalalignment='right')

plt.tight_layout()
plt.autoscale()
plt.savefig(f'./evaluation/plots/qualitative_evaluation_{ds}_{len(cols_to_change)}.svg')
plt.show()












#clean_tuple = deepcopy(clean_data[filtered_header]).to_numpy()[10]
#Zs = encoder(tf.expand_dims(clean_tuple, axis=0))



# col, row = (0,0)
# input_domain_data = []
# n_columns = clean_data.shape[1]-1
# colors = np.full((n_columns, n_columns), "mintcream")

# for c in range(n_columns):
#     aux = []
#     for i in range(n_columns):
#         a = Zs[i][row]
#         #if i == col:
#         #    aux.append(LOP.translate_operator(a, shift=k))
#         if i < c:
#             aux.append(LOP.translate_operator(a, shift=11))
#             colors[c][i] = "lightgrey"
            
#         else:
#             aux.append(a)

#     #decode to the input domain
#     A = tf.expand_dims(tf.convert_to_tensor(aux, dtype=tf.float32), axis = 0)
#     input_domain_data.append(generate_qualitative_data(A, decoder))


# modified_data = deepcopy(clean_data[0:n_columns])
# modified_data[filtered_header] = input_domain_data

# #print(modified_data["race"])

# m = reverse_to_input_domain(args.dataset, modified_data, FULL_SCALER, CAT_ENCODER)


# ## ENCODE AND DECODE THE MODIFIED TUPLES #############################
# new_data = prepare_data_subset(m, args.dataset, MISSING_REPLACE, SCALER, CAT_ENCODER, normalize_y = True)[filtered_header].to_numpy()

# x_p = []

# for tup in new_data:
#     aux = encoder(tf.expand_dims(tup, axis=0))
#     x_p.append(generate_qualitative_data(tf.transpose(aux, [1,0,2]), decoder))

# final_data = deepcopy(dirty_data[0:n_columns])
# final_data[filtered_header] = x_p

# m = reverse_to_input_domain(args.dataset, final_data, FULL_SCALER, CAT_ENCODER)

# #####################################################################


# #tuples_to_show = deepcopy(dirty_data_with_y[filtered_header]).to_numpy()[0:10]
# #Zs_csv, Ks_csv = predict_on_enhanced(tuples_to_show, LOP, encoder, decoder, _translate_1)
# #cleaned = generate_cleaned_data(tuples_to_show, Zs_csv, Ks_csv, decoder)

# #df_cleaned = deepcopy(dirty_data[0:10])
# #df_cleaned[filtered_header] = cleaned


# #EVAL CATEGORICAL ACCURACY BY COLUMN###################################################
# #c = reverse_to_input_domain(args.dataset, clean_data[0:10], FULL_SCALER, CAT_ENCODER)
# #d = reverse_to_input_domain(args.dataset, dirty_data[0:10], FULL_SCALER, CAT_ENCODER)
# #l = reverse_to_input_domain(args.dataset, df_cleaned, FULL_SCALER, CAT_ENCODER)





# ### PLOT #######################################
# fig, axs = plt.subplots(1,1)

# def _plot_table(data, axs):
#     axs.axis('tight')
#     axs.axis('off')

#     data = data.round(0)
    
#     cell_text = []
#     for row in range(len(data)):
#         cell_text.append(data.iloc[row])

#     t = axs.table(cellText=cell_text, colLabels = data.columns, loc='center', cellColours = colors)
#     t.auto_set_font_size(False)
#     t.set_fontsize(9)
#     #t.scale(2, 2)

#     return t

# #interleaves clean, dirty, cleaned
# #interleaved_df = pd.concat([c, l, d]).sort_index().reset_index(drop=True)

# #get first 4 tuples all 3 forms
# #the_table = _plot_table(interleaved_df[0:12], axs)

# #remove the target
# m.drop(target_name, axis = 1, inplace= True)

# the_table = _plot_table(m[0:n_columns], axs)


# plt.show()
