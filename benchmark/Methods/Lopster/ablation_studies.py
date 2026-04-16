import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
#os.environ['CUDA_VISIBLE_DEVICES'] = '1' # choose a single GPU from the cluster
os.system('export PATH=/etc/alternatives/cuda/bin:$PATH')
os.system('export LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib64:$LD_LIBRARY_PATH')

import csv
import sys
import matplotlib.pyplot as plt
import time
import tensorflow as tf
from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import argparse
import pandas as pd

import cleaners as cln

from datasets import get_tf_database, reverse_categorical_columns, load_regression, load_regression_dirty, load_features_and_data, reverse_categorical_columns, reverse_to_input_domain
from latent_operators import LatentOperator
from transformation_in_x import apply_transformation_in_x, include_errors_at_random
from utils import create_and_train_LOP, create_and_train_classifier
from tensorflow.keras.layers import Concatenate
from error_detection import predict_on_enhanced, eval_correctly, eval_numeric_rmse
from multiprocessing.pool import ThreadPool as Pool
from sklearn.metrics import mean_squared_error, accuracy_score

parser = argparse.ArgumentParser(description="Disentangled Latent Space Operator for Data Engineering")
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--latent", type=int, default=240)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--K", type=int, default=4)
parser.add_argument("--dataset", default='nasa')
parser.add_argument("--ncols", type=int)
parser.add_argument("--experiment", default='vs_dirty')
parser.add_argument("--metric", default='rmse')
parser.add_argument("--training_tuples", type=int, default=-1)
parser.add_argument("--eval_tuples", type=int, default=40000)


args = parser.parse_args()
np.set_printoptions(threshold=sys.maxsize)
print("Is Eager execution?",tf.executing_eagerly())


#TODO: CREATE IT IN THE PROPER PLACE
LOP = None#LatentOperator(args.K, args.ncols, 0, 1)

@tf.function
def _translate_1(inputs):
    zs  = inputs
    column = 0
    #print(LOP.n_rotations)
    for i in range(zs.shape[0]):
        new_z = LOP.translate_operator(zs[column], 1)
        zs = tf.tensor_scatter_nd_update(zs, [column], new_z)
        column = column + 1
    return zs

def generate_cleaned_data(x_, Zs, Ks, decoder):
    #avoid decoding Zs when there is no error, just use the value
    xs = tf.squeeze(tf.transpose(decoder(tf.unstack(Zs, axis = 1)), [1,0,2]))
    x_p = tf.where(Ks == 0, x_, xs)
 
    return x_p



#CONFIGS========================================================
MODEL_EPOCHS = 200
K_EPOCHS = args.epochs
ds = args.dataset
T = 'missing_values'
K = args.K
K2 = 1

MISSING_REPLACE = '3.0'

x_clean_train, y_clean_train, x_clean, y_clean, MAX, MIN, SCALER, CAT_ENCODER = load_regression(ds, -1, -1, normalize_y = True)
_, _, x_test, y_test = load_regression_dirty(ds, -1, -1,
                                             MISSING_REPLACE, SCALER, CAT_ENCODER,
                                             True, MAX, MIN, True)

HEADERS,  target_name, dirty_data, x_p, data_with_y, y_p, clean_data, FULL_SCALER, CAT_ENCODER = load_features_and_data(ds, -1, -1, MISSING_REPLACE, SCALER, CAT_ENCODER, True, MAX, MIN, normalize_sklearn = True)


COLS = x_clean_train.shape[1]
ROWS = x_clean_train.shape[0] +  x_clean.shape[0]
print("#CLEAN TRAIN TUPLES", len(x_clean_train), "\n",  "#DIRTY TUPLES", len(x_test), "\n", "# COLUMNS:", COLS)


val_clean_dataset = get_tf_database(x_clean, x_clean, args.batch_size)



##################TRAIN ALL MODELS WITH VARYING Ks and N TUPLES#############################
list_of_ks = [1, 4, 5, 6, 10, 15, 20, 24, 30, 40, 60]
list_of_epochs = [10, 20, 40, 50, 60, 80,  100, 120, 180, 240, 360, 512]
list_of_latents = [12, 24, 36, 60, 72, 84, 120, 144, 180, 240, 324, 372, 480]#, 960]

if args.dataset == 'beers':
    list_of_training_size = [20, 50, 100, 200, 300, 600]
elif args.dataset == 'smart_factory':
    list_of_training_size = [20, 50, 100, 200, 300, 600, 1000, 1800,  3000, 4000, 5000, 7000, 10000, 14000]
else:
    list_of_training_size = [1000, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000]#, 45000]


list_of_models_by_tuples = {}
list_of_models_by_ks = {}
list_of_models_by_epoch = {}
list_of_models_by_latent = {}
list_of_models_proportional = {}
list_of_models_K_equal_latent = {}

train_times_k = []
train_times_latent = []


tuples_dataset = np.concatenate(([x_clean_train, x_clean]))

#Test on different training sample size
for n_tuples in list_of_training_size:

    print(n_tuples)

    if n_tuples > ROWS: break
    
    partial_dataset = get_tf_database(tuples_dataset[0:n_tuples], tuples_dataset[0:n_tuples], args.batch_size)
    encoder, decoder, LOP, t_acc, v_acc = create_and_train_LOP(partial_dataset,
                                                           val_clean_dataset,
                                                           COLS,
                                                           args.latent,
                                                           K,
                                                           K2,
                                                           T,
                                                           epochs=args.epochs,
                                                           model_name=f"{args.dataset}/{args.dataset}_{n_tuples}")
    

    for el in encoder.layers:
        el.trainable = False
    for el in decoder.layers:
        el.trainable = False

    list_of_models_by_tuples[f"model_{n_tuples}"] = {"encoder" : encoder,
                                    "decoder": decoder,
                                    "LOP": LOP,
                                    "t_acc": t_acc ,
                                    "v_acc" : v_acc}
    
    #print(LOP.n_rotations, LOP.interval, LOP.n_columns)



    

train_dataset = get_tf_database(x_clean_train[0:args.training_tuples], x_clean_train[0:args.training_tuples], args.batch_size)


for latent in list_of_latents:
    st = time.time()
    
    encoder, decoder, LOP, t_acc, v_acc = create_and_train_LOP(train_dataset,
                                                           val_clean_dataset,
                                                           COLS,
                                                           latent,
                                                           K,
                                                           K2,
                                                           T,
                                                           epochs=args.epochs,
                                                           model_name=args.dataset)
    

    for el in encoder.layers:
        el.trainable = False
    for el in decoder.layers:
        el.trainable = False

    list_of_models_by_latent[f"model_{latent}"] = {"encoder" : encoder,
                                    "decoder": decoder,
                                    "LOP": LOP,
                                    "t_acc": t_acc ,
                                    "v_acc" : v_acc}
    #print(LOP.n_rotations, LOP.interval, LOP.n_columns)
    if (time.time() - st) > 8: #avoid saving loading times, which are usually aroun 1 to 3 sec
        train_times_latent.append(time.time() - st)



for epoch in list_of_epochs:
    encoder, decoder, LOP, t_acc, v_acc = create_and_train_LOP(train_dataset,
                                                           val_clean_dataset,
                                                           COLS,
                                                           args.latent,
                                                           K,
                                                           K2,
                                                           T,
                                                           epochs=epoch,
                                                           model_name=args.dataset)
    

    for el in encoder.layers:
        el.trainable = False
    for el in decoder.layers:
        el.trainable = False

    list_of_models_by_epoch[f"model_{epoch}"] = {"encoder" : encoder,
                                    "decoder": decoder,
                                    "LOP": LOP,
                                    "t_acc": t_acc ,
                                    "v_acc" : v_acc}
    #print(LOP.n_rotations, LOP.interval, LOP.n_columns)
    

for k in list_of_ks:

    st = time.time()
    
    encoder, decoder, LOP, t_acc, v_acc = create_and_train_LOP(train_dataset,
                                                           val_clean_dataset,
                                                           COLS,
                                                           args.latent,
                                                           k,
                                                           K2,
                                                           T,
                                                           epochs=args.epochs,
                                                           model_name=args.dataset)
    

    for el in encoder.layers:
        el.trainable = False
    for el in decoder.layers:
        el.trainable = False

    list_of_models_by_ks[f"model_{k}"] = {"encoder" : encoder,
                                    "decoder": decoder,
                                    "LOP": LOP,
                                    "t_acc": t_acc ,
                                    "v_acc" : v_acc}

    if (time.time() - st) > 8: #avoid saving loading times, which are usually aroun 1 to 3 sec
        train_times_k.append(time.time() - st)
    #print(LOP.n_rotations, LOP.interval, LOP.n_columns)



for k_proportional in list_of_ks:
    
    encoder, decoder, LOP, t_acc, v_acc = create_and_train_LOP(train_dataset,
                                                           val_clean_dataset,
                                                           COLS,
                                                           k_proportional * 10,
                                                           k_proportional,
                                                           K2,
                                                           T,
                                                           epochs=args.epochs,
                                                           model_name=args.dataset)
    

    for el in encoder.layers:
        el.trainable = False
    for el in decoder.layers:
        el.trainable = False

    list_of_models_proportional[f"model_{k_proportional}"] = {"encoder" : encoder,
                                    "decoder": decoder,
                                    "LOP": LOP,
                                    "t_acc": t_acc ,
                                    "v_acc" : v_acc}
    #print(LOP.n_rotations, LOP.interval, LOP.n_columns)



    

for k_equal in list_of_latents:
    
    encoder, decoder, LOP, t_acc, v_acc = create_and_train_LOP(train_dataset,
                                                           val_clean_dataset,
                                                           COLS,
                                                           k_equal,
                                                           k_equal,
                                                           K2,
                                                           T,
                                                           epochs=args.epochs,
                                                           model_name=args.dataset)
    

    for el in encoder.layers:
        el.trainable = False
    for el in decoder.layers:
        el.trainable = False

    list_of_models_K_equal_latent[f"model_{k_equal}"] = {"encoder" : encoder,
                                    "decoder": decoder,
                                    "LOP": LOP,
                                    "t_acc": t_acc ,
                                    "v_acc" : v_acc}

    #print(LOP.n_rotations, LOP.interval, LOP.n_columns)















#Default Classifier MODEL========================================================================
t_dataset = get_tf_database(x_clean_train, y_clean_train, args.batch_size)
v_dataset = get_tf_database(x_clean, y_clean, args.batch_size)    

start = time.time()
nnreg_model = create_and_train_classifier(t_dataset,
                                          v_dataset,
                                          COLS,
                                          args.latent,
                                          T,
                                          n_epochs=MODEL_EPOCHS,
                                          model_name=args.dataset)
print("NN train ", time.time() - start, "sec")





#save training performance
df_time_k = pd.DataFrame(train_times_k, index = list_of_ks, columns = ['sec'])
df_time_k.index.name = "param"
if(df_time_k.shape[0] > 5):
    df_time_k.to_csv(f'./evaluation/ablation_studies/time_to_train_k_{args.dataset}.csv')

df_time_latent = pd.DataFrame(train_times_latent, index = list_of_latents, columns = ['sec'])
df_time_latent.index.name = "param"
if(df_time_latent.shape[0] > 5):
    df_time_latent.to_csv(f'./evaluation/ablation_studies/time_to_train_latent_{args.dataset}.csv')


####################################################################################################################################
print("RESULTS ONLY")

clean_score = nnreg_model.evaluate(x_clean, y_clean)[1]
n_score = nnreg_model.evaluate(x_test, y_test)[1] #dirty score

import psutil
process = psutil.Process()



def _evaluate_ablation_model_on_rmse(list_of_models, list_of_params, experiment_name):
    global LOP

    list_of_scores = []
    indexes_for_pandas = []
    dirty = dirty_data[HEADERS["filtered_header"]]#.iloc[:args.eval_tuples]
    #clean = clean_data[HEADERS["filtered_header"]]

    for m in list_of_params:
        start = time.time()
        
        model = list_of_models[f"model_{m}"]
        LOP = model["LOP"]
        Zs, Ks = predict_on_enhanced(dirty.to_numpy(), model["LOP"], model["encoder"], model["decoder"], _translate_1)

        d = eval_numeric_rmse(dirty, Zs, Ks, model["decoder"])
        dirty_data[HEADERS["filtered_header"]] = d

        
        lop_data = reverse_to_input_domain(args.dataset, dirty_data, FULL_SCALER, CAT_ENCODER)
        lop_data.to_csv(f'./DATASETS_REIN/{args.dataset}/LOP_ablation.csv', index = False)

        _, rmse_repaired = cln.evaluate(f"./DATASETS_REIN/{args.dataset}/clean.csv",
                                        f"./DATASETS_REIN/{args.dataset}/dirty01.csv",
                                        f"./DATASETS_REIN/{args.dataset}/LOP_ablation.csv",
                                        n_tuples = args.eval_tuples)

        print(rmse_repaired)
        
        list_of_scores.append({"lop_numeric": rmse_repaired, "inference_time": time.time() -  start, "inference_memory": process.memory_info().rss / 1000000.0})  # in MBs


        indexes_for_pandas.append(m)
        df_for_plots = pd.DataFrame.from_records(list_of_scores, index = indexes_for_pandas)
        df_for_plots.index.name = "param"
        df_for_plots.to_csv(f'./evaluation/ablation_studies/rmse_{experiment_name}.csv')







def _evaluate_ablation_model_on_domwnstream_task(list_of_models, list_of_params, experiment_name):
    global LOP
    
    list_of_scores = []

    for m in list_of_params:
        start = time.time()
        
        model = list_of_models[f"model_{m}"]

        LOP = model["LOP"]        
        concatenation_model = Concatenate()(model["decoder"].output)
        enhanced_model = Model(inputs=model["decoder"].input, outputs= nnreg_model(concatenation_model))
        enhanced_model.compile(optimizer='adam', loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])
        Zs, Ks = predict_on_enhanced(x_test, model["LOP"], model["encoder"], model["decoder"], _translate_1)
        detections = eval_correctly(x_test, y_test, Zs, Ks, model["decoder"], nnreg_model)

        list_of_scores.append({"clean": clean_score, "dirty": n_score, "lop":detections, "inference_time":time.time() -  start})

    df_for_plots = pd.DataFrame.from_records(list_of_scores, index = list_of_params)
    df_for_plots.index.name = "param"
    df_for_plots.to_csv(f'./evaluation/ablation_studies/downstream_{experiment_name}.csv')










#EXECUTE EVALUATION#####################################################################
if args.metric == "downstream":
    _evaluate_ablation_model_on_domwnstream_task(list_of_models_by_tuples, list_of_training_size, f'tuples_{args.dataset}')
    _evaluate_ablation_model_on_domwnstream_task(list_of_models_K_equal_latent, list_of_latents, f'equal_{args.dataset}')
    _evaluate_ablation_model_on_domwnstream_task(list_of_models_proportional, list_of_ks, f'proportional_{args.dataset}')
    _evaluate_ablation_model_on_domwnstream_task(list_of_models_by_epoch, list_of_epochs, f'epochs_{args.dataset}')
    _evaluate_ablation_model_on_domwnstream_task(list_of_models_by_latent, list_of_latents, f'latents_{args.dataset}')
    _evaluate_ablation_model_on_domwnstream_task(list_of_models_by_ks, list_of_ks, f'ks_{args.dataset}')

else:
    #_evaluate_ablation_model_on_rmse(list_of_models_K_equal_latent, list_of_latents, f'equal_{args.dataset}')
    #_evaluate_ablation_model_on_rmse(list_of_models_proportional, list_of_ks, f'proportional_{args.dataset}')

    #_evaluate_ablation_model_on_rmse(list_of_models_by_tuples, list_of_training_size, f'tuples_{args.dataset}')
    #print("tuples done.")
    _evaluate_ablation_model_on_rmse(list_of_models_by_epoch, list_of_epochs, f'epochs_{args.dataset}')
    print("epochs done.")
    _evaluate_ablation_model_on_rmse(list_of_models_by_latent, list_of_latents, f'latents_{args.dataset}')
    print("latent done.")
    _evaluate_ablation_model_on_rmse(list_of_models_by_ks, list_of_ks, f'ks_{args.dataset}')
    print("ks done.")
