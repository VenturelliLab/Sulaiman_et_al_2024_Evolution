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

# pandas dataframe with hyperparameter settings
hyper_df = pd.read_csv("CDevo_hyper_df.csv")
lmbda = hyper_df.iloc[hyper_df.k.values == int(k)].values[0][1:][0]

# import train and test data
train_df = pd.read_csv(f"folds_CDevo/train_{k}.csv")
test_df = pd.read_csv(f"folds_CDevo/test_{k}.csv")

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
pred_df.to_csv(f"CDevo_pred_{k}.csv", index=False)
