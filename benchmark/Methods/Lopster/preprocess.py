from datetime import datetime
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import numpy as np
import os


#CHECK IF ALL DIRECTORIES HAVE BEEN CREATED########################

def _check_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

main_path = os.getcwd()
_check_dir(f'{main_path}/MODELS')
_check_dir(f'{main_path}/evaluation')
_check_dir(f'{main_path}/evaluation/plots')
_check_dir(f'{main_path}/evaluation/ablation_studies')
_check_dir(f'{main_path}/evaluation/ablation_studies/plots')


###################################################################



def _encode_cat(X_c):
    data = X_c.copy()
    nonulls = data.dropna().values
    #nonulls = data.values
    impute_reshape = nonulls.reshape(-1,1)
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    impute_ordinal = encoder.fit_transform(impute_reshape)
    data.loc[data.notnull()] = np.squeeze(impute_ordinal)
    #data = np.squeeze(impute_ordinal)
    return data, encoder

def _decode_cat(X_c, encoder):
    data = X_c.copy()
    nonulls = data.dropna().values.reshape(-1,1)
    #nonulls = data.values.reshape(-1,1)
    n_cat = len(encoder.categories_[0])
    nonulls = np.round(nonulls).clip(0, n_cat-1)
    nonulls = encoder.inverse_transform(nonulls)
    data.loc[data.notnull()] = np.squeeze(nonulls)
    #data = np.squeeze(nonulls)
    return data

def preprocess(df, datecols, id_cols, cat_cols, les=None, fill_na=True, drop_columns = True):
    data = df.copy(deep=True)
    # remove rows where target is missing
    #data.drop(data[data[target].isnull()].index, inplace=True)

    if drop_columns:
        data.drop(columns=id_cols, inplace=True)
        data.drop(columns=datecols, inplace=True)

    # for col, time_format in datecols:
    #     if time_format is not None:
    #         data[col] = data[col].apply(lambda x: datetime.strptime(x, time_format).timestamp() / 3600)
    #     else:
    #         data[col] = pd.to_datetime(data[col]).apply(lambda x: x.value) / 10 ** 9


    if les is None:
        les = dict()
        
    for cat_col in cat_cols:
        data[cat_col], les[cat_col] = _encode_cat(data[cat_col]) 
        
    #transform all to numeric
    for c in data.columns:
        data[c] = pd.to_numeric(data[c], errors = 'coerce')

    return data, les


def reverse_categorical_columns(ds, data, label_encoder, dataset_config):
    categorical_columns = dataset_config[ds]["cat_cols"]
    for cat_col in categorical_columns:
        data[cat_col] = _decode_cat(data[cat_col],label_encoder[cat_col])

    return data
