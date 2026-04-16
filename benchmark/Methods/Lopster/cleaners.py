################################################
# Benchmark: A collection of data repair methods
# Authors: Christian Hammacher, Mohamed Abdelaal
# Date: February 2021
# Software AG
# All Rights Reserved
################################################

################################################
import math
import numbers
import time
import argparse
from sklearn import preprocessing
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import KNNImputer as sklearnKNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer as sklearnIterativeImputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, accuracy_score
import sklearn.neighbors._base
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support

################################################

def evaluate(gtDF, dirtyDF, cleanedDF, n_tuples = -1):
        """
        Runs 3 Experiments for evaluating the cleaned dataset.
        1. Experiment: interprete all columns as categorical and calculate precision, recall, f1 relative to all error (actual_errors_dict)
        2. Experiment: get numerical columns from groundtruthDF and calulate RMSE for dirty dataset and for the repaired dataset (considers cells
                        where groundtruth, dirty and repaired dataset have numerical values). Uses StandardScaler
        3. Expriment: get categorical columns from groundtruthDF and calculate precision, recall, f1 relativ to all errors in the respective columns

        Arguments:
        detections -- dictionary - keys represent i,j of dirty cells & values are constant string "JUST A DUUMY VALUE"
        dirtyDF (dataframe) -- dirty dataframe
        cleanedDF (dataframe) -- dataframe that was repaired
        """

        def convert_to_float_or_nan(matrix):
            for (x,y), _ in np.ndenumerate(matrix):
                try:
                    matrix[x,y] = float(matrix[x,y])
                except ValueError:
                    matrix[x,y] = np.nan
            return matrix

        # Initialize a dictionary to pack the results
        evaluation_dict = {}


        groundTruthDF = pd.read_csv(f'{gtDF}', sep=",", encoding="utf-8")
        dirtyDF = pd.read_csv(f"{dirtyDF}", sep=",", encoding="utf-8")[:n_tuples]
        cleanedDF = pd.read_csv(f"{cleanedDF}", sep=",", encoding="utf-8")[:n_tuples]

        # Get numerical and categorical columns of the ground truth
        groundTruthDF = groundTruthDF.apply(pd.to_numeric, errors="ignore")
        gt_num_columns = groundTruthDF.select_dtypes(include="number").columns
        gt_cat_columns = groundTruthDF.select_dtypes(exclude="number").columns
        #print(gt_num_columns)

        y_groundtruth = groundTruthDF[gt_num_columns].to_numpy(dtype=float)
        y_cleaned = cleanedDF[gt_num_columns].to_numpy()
        y_dirty = dirtyDF[gt_num_columns].to_numpy()
                
        # convert each element in y_cleaned, y_dirty, and y_groundtruth to a float, and 
        # if it fails (due to the element not being a number), sets that element to NaN.
        y_cleaned = convert_to_float_or_nan(y_cleaned)
        y_dirty = convert_to_float_or_nan(y_dirty) 
        y_groundtruth = convert_to_float_or_nan(y_groundtruth)

        scaler = StandardScaler()
        """ y_groundtruth, y_cleaned, y_dirty have nan at the same positions
            thus nan values can be simply removed and the resulting arrays still fit """

        # scale, remove nan values
        scaler = scaler.fit(y_groundtruth) # with all samples
        y_true = scaler.transform(y_groundtruth[:n_tuples]).flatten().astype(float)
        y_true = np.nan_to_num(y_true) # replace nan with zero

        # scale, remove nan values and calculate rmse for repaired dataset
        y_pred = scaler.transform(y_cleaned).flatten().astype(float)
        y_pred = np.nan_to_num(y_pred)
        
        rmse_repaired = mean_squared_error(y_true, y_pred, squared=False)

        # scale, remove nan values and calculate rmse for dirty dataset
        y_pred2 = scaler.transform(y_dirty).flatten().astype(float)
        y_pred2 = np.nan_to_num(y_pred2)
        rmse_dirty = mean_squared_error(y_true, y_pred2, squared=False)


        return rmse_dirty, rmse_repaired
