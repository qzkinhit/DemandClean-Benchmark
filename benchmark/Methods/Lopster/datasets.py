import os
import tensorflow as tf
import numpy as np
import pandas as pd
import preprocess as pr
from sklearn import preprocessing
from copy import deepcopy
from sklearn.preprocessing import StandardScaler
import json

def _to_tf(X,Y, batchsize):
    _dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    _dataset = _dataset.cache().shuffle(100000, reshuffle_each_iteration=True).batch(batchsize, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    return _dataset

def get_tf_database(data, target, batchsize):
    return _to_tf(data, target , batchsize)

def _get_data_config():
    with open('dataset_configuration.json', 'r') as file:
        data = file.read()
    return json.loads(data)

def split_numerical_and_categorical_columns(ds, filtered_header):
    dataset_config = _get_data_config()
    numeric_header = [i for i in filtered_header if i not in dataset_config[ds]["cat_cols"]]
    # must do the loop to keep the order
    categorical_header = [i for i in filtered_header if i in dataset_config[ds]["cat_cols"]]
    return numeric_header, categorical_header

def _normalize_by_min_max(df, min_x, max_x):
    norm = (max_x - min_x) + 1e-10 #only breaks if MAX = MIN, so we need some noise
    normalized_data = pd.DataFrame(tf.where(min_x < 0, (df + min_x) / norm, (df - min_x) / norm).numpy(), columns = df.columns)
    return  normalized_data

def _normalize_by_mean(df, scaler):
    return  pd.DataFrame(scaler.transform(df), columns = df.columns)

def _fit_mean_scaler(df, scaler):
    normalized_data = scaler.fit_transform(df)
    return  pd.DataFrame(normalized_data, columns = df.columns), scaler

def _get_train_val_and_test(df, target, n_train_instances, n_test_instances):

    train, val, test = np.split(df.sample(frac=1), #shuffle
                                [int(.70*len(df)), int(.80*len(df))])

    X_train = train.to_numpy(dtype=np.float32)[:n_train_instances]#, df.columns != target ]
    X_val = val.to_numpy(dtype=np.float32)#[:, df.columns != target]
    X_test = test.to_numpy(dtype=np.float32)[:n_test_instances]#, df.columns != target]

    #must add a newaxis (empty)
    y_train = train.to_numpy(dtype=np.float32)[:n_train_instances, df.columns == target, np.newaxis]
    y_val = val.to_numpy(dtype=np.float32)[:,df.columns == target, np.newaxis]
    y_test = test.to_numpy(dtype=np.float32)[:n_test_instances,df.columns == target, np.newaxis]

    return  X_train, y_train, X_test, y_test, X_val, y_val
    
def _replace_nan(df, missing):
    df = df.replace('N/A', missing)
    df = df.replace('', missing)
    df = df.replace(' ', missing)
    df = df.replace('NaN', missing)
    df = df.fillna(missing)
    return df

def load_regression(ds, n_train_instances, n_test_instances, normalize_y = False, normalize_sklearn = True,  path_to_dataset = "DATASETS_REIN"):
    scaler = StandardScaler()
    dataset_config = _get_data_config()
    clean_dir = f"{path_to_dataset}/{ds}/"
    df = pd.read_csv(os.path.join(clean_dir, 'clean.csv'))

    df, CAT_ENCODER = pr.preprocess(df,
                                    dataset_config[ds]["date_cols"],
                                    dataset_config[ds]["id_cols"],
                                    dataset_config[ds]["cat_cols"])

    print("Unique Values per column:\n ", df.nunique())

    #drop rows that contains empty cells
    df.dropna(inplace = True)

    MIN = np.min(df,axis=0)
    MAX = np.max(df,axis=0)

    #non-normalized copy
    target = dataset_config[ds]['target']
    y_ = df[target].copy()

    if normalize_sklearn:
        df, scaler = _fit_mean_scaler(df, scaler)
    else:
        df = _normalize_by_min_max(df, MIN, MAX)

    # when all values of the column are equal it divides by zero and return NaN, so we need to replace by 0.0!!!
    df = df.fillna(0.0)
    
    if not normalize_y:
        df[target] = y_
    
    X_train, y_train, X_test, y_test, X_val, y_val = _get_train_val_and_test(df, target,  n_train_instances, n_test_instances)
    
    return X_train, y_train, X_test, y_test, MAX, MIN, scaler, CAT_ENCODER


def load_regression_dirty(ds, n_train_instances, n_test_instances, missing, scaler, CAT_ENCODER, normalize_y = False, MAX = None, MIN = None, normalize_sklearn = True, path_to_dataset = "DATASETS_REIN"):
    dataset_config = _get_data_config()
    clean_dir = f"{path_to_dataset}/{ds}/"
    # 支持dirty.csv和dirty01.csv两种命名方式
    dirty_path = os.path.join(clean_dir, 'dirty01.csv')
    if not os.path.exists(dirty_path):
        dirty_path = os.path.join(clean_dir, 'dirty.csv')
    df = pd.read_csv(dirty_path)
    df = _replace_nan(df, missing)

    #drop rows that still contains empty cells
    df,_ = pr.preprocess(df,
                         dataset_config[ds]["date_cols"],
                         dataset_config[ds]["id_cols"],
                         dataset_config[ds]["cat_cols"],
                         les = CAT_ENCODER)

    #non-normalized copy
    target = dataset_config[ds]['target']
    y_ = df[target].copy()
    
    if normalize_sklearn:
        df = _normalize_by_mean(df, scaler)
    elif MAX is not None:
        df = _normalize_by_min_max(df, MIN, MAX)
    else:
        df = _normalize_by_min_max(df, np.min(df,axis=0), np.max(df,axis=0))

    # For min/max normalization, when all values of the column are equal it divides by zero and return NaN,
    #so we need to replace by 0.0!!!
    df = df.fillna(0.0) 
    print(np.min(df,axis=0), "\n",  np.max(df,axis=0))

    if not normalize_y:
        df[target] = y_    

    X_train, y_train, X_test, y_test, X_val, y_val = _get_train_val_and_test(df, target,  n_train_instances, n_test_instances)

    return X_train, y_train, X_test, y_test

def load_features_and_data(ds, n_instances, n_test_instances, missing, scaler, CAT_ENCODER,  normalize_y = False, MAX = None, MIN = None, normalize_sklearn = True,  path_to_dataset = "DATASETS_REIN"):

    dataset_config = _get_data_config()

    clean_dir = f"{path_to_dataset}/{ds}/"
    # 支持dirty.csv和dirty01.csv两种命名方式
    dirty_path = os.path.join(clean_dir, 'dirty01.csv')
    if not os.path.exists(dirty_path):
        dirty_path = os.path.join(clean_dir, 'dirty.csv')
    df = pd.read_csv(dirty_path)
    df_clean = pd.read_csv(os.path.join(clean_dir, 'clean.csv'))

    # 只有当 n_instances > 0 时才做切片（避免 [0:-1] 丢失最后一行）
    if n_instances is not None and n_instances > 0:
        df = df.iloc[:n_instances]
        df_clean = df_clean.iloc[:n_instances]

    numeric_header = df_clean.select_dtypes(include="number").columns
    categorical_header = df_clean.select_dtypes(exclude="number").columns

    #check if it is not a timestamp
    for dtcol in dataset_config[ds]["date_cols"]:
        if  dtcol not in numeric_header :
            #remove date columns from the categorical set
            categorical_header = categorical_header.drop(dtcol)

    #save the full version for CSV write
    FULL = deepcopy(df)    
  
    df, _ = pr.preprocess(df,
                          dataset_config[ds]["date_cols"],
                          dataset_config[ds]["id_cols"],
                          dataset_config[ds]["cat_cols"],
                          les = CAT_ENCODER)

    df_clean, _ = pr.preprocess(df_clean,
                                dataset_config[ds]["date_cols"],
                                dataset_config[ds]["id_cols"],
                                dataset_config[ds]["cat_cols"],
                                les = CAT_ENCODER,
                                drop_columns = False)

    # #FIT THE SCALER FOR ALL COLUMNS##############################
    sc = StandardScaler()
    _, full_scaler = _fit_mean_scaler(df_clean, sc)
    # #################################################################

    FULL, _ = pr.preprocess(FULL,
                            dataset_config[ds]["date_cols"],
                            dataset_config[ds]["id_cols"],
                            dataset_config[ds]["cat_cols"],
                            les = CAT_ENCODER,
                            drop_columns = False)
    
    header = df.columns.tolist()
    full_header = df_clean.columns.tolist()


    #non-normalized copy
    target = dataset_config[ds]['target']
    y_ = df[target].copy()

    #Normalize    
    if normalize_sklearn:
        df = _normalize_by_mean(df, scaler)
        df_clean = _normalize_by_mean(df_clean, full_scaler)
        FULL = _normalize_by_mean(FULL, full_scaler)

    df_dirty = deepcopy(FULL)

    #replace NaNs as in REIN
    df_dirty = pd.DataFrame(np.nan_to_num(df_dirty), columns = df_dirty.columns)
    FULL = pd.DataFrame(np.nan_to_num(FULL), columns = FULL.columns)
    df_clean =  pd.DataFrame(np.nan_to_num(df_clean), columns = df_clean.columns)

    df_dirty = df_dirty.where(df_dirty >= -float(missing), float(missing))
    df_dirty = df_dirty.where(df_dirty <= float(missing), float(missing))

    if not normalize_y:
        df[target] = y_

    data_no_target = df.drop([target], axis = 1)
    filtered_header = header    #df.columns.tolist() #data_no_target.columns.tolist()
    Y = df[target].to_numpy() #dtype=np.float32)
    
    print("numeric:", numeric_header)
    print("categorical:", categorical_header)

    headers = {"full_header" : full_header ,
               "filtered_header_with_y": header,
               "filtered_header": filtered_header,
               "numeric_header": numeric_header,
               "categorical_header": categorical_header}
   
   

    return headers, target, FULL, data_no_target.to_numpy(), df_dirty, Y, df_clean, full_scaler, CAT_ENCODER



def reverse_categorical_columns(ds, data, label_encoder):
    dataset_config = _get_data_config()
    return pr.reverse_categorical_columns(ds, data, label_encoder, dataset_config)

def reverse_to_input_domain(ds, data, scaler, CAT_ENCODER):
    lop_data = pd.DataFrame(scaler.inverse_transform(data), columns = data.columns)


    #META INFORMATION
    dataset_config = _get_data_config()

    #negative columns go to zero if the column does not allow for negative values
    allow_negatives = dataset_config[ds]["allow_negatives"]
    min_zero_columns = lop_data.columns.difference(allow_negatives)   
    lop_data[lop_data[min_zero_columns] < 0] = 0
    
    #round integer columns to the closest integer
    non_integers = dataset_config[ds]["date_cols"] + dataset_config[ds]["id_cols"] +  dataset_config[ds]["cat_cols"] +  dataset_config[ds]["float_cols"]
    integer_columns = lop_data.columns.difference(non_integers)   
    lop_data[integer_columns] = lop_data[integer_columns].round(0).astype('int64')  
    ###########################################################
    
    
    return reverse_categorical_columns(ds, lop_data, CAT_ENCODER)


def prepare_data_subset(df, ds, missing, scaler, CAT_ENCODER, normalize_y = False):
    dataset_config = _get_data_config()
    df = _replace_nan(df, missing)

    #drop rows that still contains empty cells
    df,_ = pr.preprocess(df,dataset_config[ds]["target"],
                         dataset_config[ds]["date_cols"],
                         dataset_config[ds]["id_cols"],
                         dataset_config[ds]["cat_cols"],
                         les = CAT_ENCODER)

    #non-normalized copy
    target = dataset_config[ds]['target']
    y_ = df[target].copy()
    
    df = _normalize_by_mean(df, scaler)

    if not normalize_y:
        df[target] = y_    


    return df


def get_date_columns(ds, target_dataset, full_dataset):
    dataset_config = _get_data_config()
    dates = dataset_config[ds]["date_cols"]
    target_dataset[dates] = full_dataset[dates]
    return target_dataset
