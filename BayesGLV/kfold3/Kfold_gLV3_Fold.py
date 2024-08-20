#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, linregress

import os
import time
import itertools
import sys

from glove.model3 import *

from sklearn.model_selection import KFold


# # fit gLV models

# In[3]:


def predict_df(df, species):

    # save measured and predicted values
    exp_names = []
    pred_species = []
    pred = []
    stdv = []
    true = []

    # pull just the community data
    test_data = process_df(df, species)

    # plot the results
    for exp, t_span, Y_m in test_data:

        # predict
        Y_p, Y_std = model.predict(Y_m, t_span)

        # set NaN to zero
        Y_p = np.nan_to_num(Y_p)
        Y_std = np.nan_to_num(Y_std)

        ### prediction results for species that were present ###
        inds_present = Y_m[0] > 0
        exp_names.append([exp]*sum(inds_present)*(Y_m.shape[0]-1))
        pred_species.append(np.tile(np.vstack(species)[inds_present], Y_m.shape[0]-1).T.ravel())
        true.append(Y_m[1:,inds_present].ravel())
        pred.append(Y_p[1:,inds_present].ravel())
        stdv.append(Y_std[1:,inds_present].ravel())

    # concatenate list
    exp_names = np.concatenate(exp_names)
    pred_species = np.concatenate(pred_species)
    true = np.concatenate(true)
    pred = np.concatenate(pred)
    stdv = np.concatenate(stdv)

    return exp_names, pred_species, true, pred, stdv


# In[ ]:

# get job ID
k = int(sys.argv[-1])

# strain
strain = sys.argv[-2]

# import train and test
train_df = pd.read_csv(f'{strain}_train_{k}.csv')
test_df = pd.read_csv(f'{strain}_test_{k}.csv')

# determine species names
species = train_df.columns.values[2:]

# instantiate gLV fit
model = gLV(species, train_df)

# fit to data
model.fit()

# predict test data
exp_names, pred_species, true, pred, stdv = predict_df(test_df, species)

# save prediction results to a .csv
kfold_df = pd.DataFrame()
kfold_df['Treatments'] = exp_names
kfold_df['species'] = pred_species
kfold_df['true'] = true
kfold_df['pred'] = pred
kfold_df['stdv'] = stdv
kfold_df.to_csv(f"{strain}_pred_{k}.csv", index=False)
