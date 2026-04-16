import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import csv
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import argparse

from datasets import get_tf_database, load_regression, load_regression_dirty, load_features_and_data, reverse_categorical_columns, reverse_to_input_domain, get_date_columns
from latent_operators import LatentOperator
from utils import create_and_train_LOP
from error_detection import predict_on_enhanced
from copy import deepcopy

parser = argparse.ArgumentParser(description="Generalizable Data Cleaning of Tabular Data in Latent Space")
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--latent", type=int, default=120)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--K", type=int, default=12)
parser.add_argument("--dataset", default='adult')
parser.add_argument("--path", default="./")

args = parser.parse_args()

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


#=========================== CONFIGS ==========================================
ds = args.dataset
SAMPLE_SIZE = -1
MISSING_REPLACE = '3.0'
T = 'missing_values'

x_clean_train, y_clean_train, x_clean, y_clean, MAX, MIN, SCALER, CAT_ENCODER = load_regression(ds,
                                                                                                SAMPLE_SIZE,
                                                                                                SAMPLE_SIZE,
                                                                                                True,
                                                                                                normalize_sklearn = True,
                                                                                                path_to_dataset = args.path)

COLS = x_clean_train.shape[1]


#======================= TRAIN MODELS =========================================

train_dataset = get_tf_database(x_clean_train,x_clean_train,args.batch_size)
val_dataset = get_tf_database(x_clean,x_clean,args.batch_size)

encoder, decoder, LOP, t_acc, v_acc = create_and_train_LOP(train_dataset,
                                                           val_dataset,
                                                           COLS,
                                                           args.latent,
                                                           args.K,
                                                           1,
                                                           T,
                                                           epochs=args.epochs,
                                                           model_name=args.dataset)
                                                          



#======================= Data Cleaning =========================================
headers,  target_name, dirty_data, _, data_with_y, _, clean_data, FULL_SCALER, CAT_ENCODER = load_features_and_data(ds,
                                                                                                                    SAMPLE_SIZE,
                                                                                                                    SAMPLE_SIZE,
                                                                                                                    MISSING_REPLACE,
                                                                                                                    SCALER, CAT_ENCODER,
                                                                                                                    True, MAX, MIN,
                                                                                                                    normalize_sklearn = True,
                                                                                                                    path_to_dataset = args.path)
filtered_header = headers["filtered_header"]

X_dirty = deepcopy(data_with_y[filtered_header]).to_numpy()
Zs_csv, Ks_csv = predict_on_enhanced(X_dirty, LOP, encoder, decoder, _translate_all_columns_by_1)
clean_csv = generate_cleaned_data(X_dirty, Zs_csv, Ks_csv, decoder)

#=========== Replace the dirty data with the clean one =========================

#replace the cleaned columns, and reverse to input domain
# 将TensorFlow张量转换为numpy数组以兼容pandas
if hasattr(clean_csv, 'numpy'):
    clean_csv_np = clean_csv.numpy()
else:
    clean_csv_np = np.array(clean_csv)
dirty_data[filtered_header] = clean_csv_np
lop_data = reverse_to_input_domain(args.dataset, dirty_data, FULL_SCALER, CAT_ENCODER)

#recover the date columns
lop_data = get_date_columns(ds, lop_data, dirty_data)

#save a cleaned CSV
lop_data.to_csv(f'{args.path}/{args.dataset}/lopster.csv', index = False)
