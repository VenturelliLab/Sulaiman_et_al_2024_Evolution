import pandas as pd
import numpy as np
from scipy.stats import norm, linregress

import os
import sys
import time
import itertools

from glove_L1.glv3 import *

# get job ID
k = sys.argv[-1]

# range of L1 penalties to try
lmbdas = [0., 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.]

# dictionary to relate job ID to job settings
job_index = 0
job_settings = {}
for i in range(20):  # number of outer folds
    for j in range(10):  # number of inner folds
        for l1 in lmbdas:  # L1
            job_settings[job_index] = (i, j, l1)
            job_index += 1
outer, inner, lmbda = job_settings[int(k)]

# import train and test data
train_df = pd.read_csv(f"folds_CDanc/train_{outer}_{inner}.csv")
test_df = pd.read_csv(f"folds_CDanc/test_{outer}_{inner}.csv")

# determine species names
species = train_df.columns.values[2:]

# init model
model = gLV(dataframe=train_df,
            species=species,
            lmbda=lmbda)

# fit to data
f = model.fit_rmse(epochs=200)

# function to make predictions on test set
def predict_df(model, df):
    # save measured and predicted values
    mean_dfs = []

    for exp_name, exp_df in df.groupby("Treatments"):
        # make sure comm_data is sorted in chronological order
        exp_df.sort_values(by='Time', ascending=True, inplace=True)

        # get initial condition and evaluation times
        t_span = exp_df.Time.values
        Y_m = exp_df[species].values

        # predict
        y_mean = model.predict_point(Y_m, t_span)

        # save dataframe with predictions
        mean_df = pd.DataFrame()
        mean_df["Treatments"] = [exp_name] * y_mean.shape[0]
        mean_df['Time'] = t_span
        mean_df[species] = y_mean
        mean_dfs.append(mean_df)

    # concatenate list
    mean_df = pd.concat(mean_dfs)

    return mean_df


# plot fitness to data
pred_df = predict_df(model, test_df)

# save dataframe
pred_df.to_csv(f"CDanc_pred_{outer}_{inner}_{lmbda}.csv", index=False)
