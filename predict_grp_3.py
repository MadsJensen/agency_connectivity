#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 19:20:31 2017

@author: au194693
"""

import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt

from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     permutation_test_score)
from sklearn.linear_model import (LogisticRegressionCV, LogisticRegression,
                                  RandomizedLogisticRegression)
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
plt.style.use("ggplot")

import os
os.chdir('/Users/au194693/projects/agency_connectivity/data')

# Create dataframe to extract values for X
pow_data = pd.read_csv("power_data_no-step_both_grps_all-freqs.csv")

pow_data = pow_data[pow_data.trial_status == True]
pow_data = pow_data[(pow_data.band != "theta") & (pow_data.band != "gamma2")]

pow_data_mean = pow_data.groupby(by=["subject", "group", "band",
                                     "label"]).mean().reset_index()
pow_data_mean = pow_data_mean.sort_values(
    by=["group", "subject", "band", "label"])

labels = list(pow_data.label.unique())
subjects = list(pow_data_mean.subject.unique())
pow_X = np.empty([len(pow_data_mean.subject.unique()), 84])

for i, sub in enumerate(subjects):
    fdata = pow_data_mean[pow_data_mean.subject == sub]
    pow_X[i, :] = fdata.r.values.reshape(-1)

ispc_data = pd.read_csv("itc_data_no-step_both_grps_all-freqs.csv")

ispc_data = ispc_data[ispc_data.trial_status == True]
ispc_data = ispc_data[(ispc_data.band != "theta") &
                      (ispc_data.band != "gamma2")]

ispc_data_mean = ispc_data.groupby(by=["subject", "group", "band",
                                       "label"]).mean().reset_index()
ispc_data_mean = ispc_data_mean.sort_values(
    by=["group", "subject", "band", "label"])

# Dataframe to generate labels
df = ispc_data_mean[ispc_data_mean.subject == "p21"][["band", "label"]]
df = df.append(df)
df["condition"] = "nan"
df["condition"][84:] = "ispc"
df["condition"][:84] = "power"
df["comb_name"] = df.condition + "_" + df.band + "_" + df.label

labels = list(df.comb_name.get_values())

ispc_X = np.empty([len(ispc_data_mean.subject.unique()), 84])

for i, sub in enumerate(subjects):
    fdata = ispc_data_mean[ispc_data_mean.subject == sub]
    ispc_X[i, :] = fdata.ISPC.values.reshape(-1)

# Concatenate into X and create y
X = np.concatenate((pow_X, ispc_X), axis=1)
y = np.concatenate((np.zeros(18), np.ones(18)))

# Scale X
pow_scl = scale(pow_X)
ispc_scl = scale(ispc_X)

X_scl = np.concatenate((pow_scl, ispc_scl), axis=1)

cv = StratifiedKFold(n_splits=9, shuffle=True)

# Logistic Regression with cross validation for C
scores = []
coefs = []
Cs = []
LRs = []

for train, test in cv.split(X, y):
    # clf = LogisticRegression(C=1)
    clf = LogisticRegressionCV()
    clf.fit(X[train], y[train])
    y_pred = clf.predict(X[test])

    scores.append(roc_auc_score(y[test], y_pred))
    coefs.append(clf.coef_)
    Cs.append(clf.C_)
    LRs.append(clf)

lr_mean = LogisticRegression()
lr_mean.coef_ = np.asarray(coefs).mean(axis=0)
lr_mean.C = np.asarray(Cs).mean()
lr_mean.intercept_ = np.asarray([est.intercept_ for est in LRs]).mean()

lr_coef_mean = np.asarray(coefs).mean(axis=0)
lr_coef_std = np.asarray(coefs).std(axis=0)

cv_scores = cross_val_score(
    lr_mean, X, y, scoring="roc_auc", cv=StratifiedKFold(9))

score_full_X, perm_scores_full_X, pvalue_full_X = permutation_test_score(
    lr_mean,
    X,
    y,
    scoring="roc_auc",
    cv=StratifiedKFold(9),
    n_permutations=2000,
    n_jobs=2)

# RandomizedLogisticRegression feature selection
for i in range(20):
    rlr = RandomizedLogisticRegression(
        n_resampling=5000, C=lr_mean.C, selection_threshold=0.6)
    rlr.fit(X, y)
    print(sum(rlr.get_support()))
    
    
X_rlr = rlr.transform(X)

cv_scores_rlr = cross_val_score(lr_mean, X_rlr, y, 
                                scoring="roc_auc", cv=StratifiedKFold(9))

score_rlr, perm_scores_rlr, pvalue_rlr = permutation_test_score(
    lr_mean,
    X_rlr,
    y,
    scoring="roc_auc",
    cv=StratifiedKFold(9),
    n_permutations=2000,
    n_jobs=2)

# printing results
print("score no reduction: %s (std %s)" % (cv_scores.mean(), cv_scores.std()))
print("rlr number of features: %s" % sum(rlr.get_support()))
print("score rlr: %s (std %s)" % (cv_scores_rlr.mean(), cv_scores_rlr.std()))

print("permutation result (full): %s, p-value: %s" %
      (score_full_X, pvalue_full_X))
print("permutation result (rlr): %s, p-value: %s" % (score_rlr, pvalue_rlr))

rlr_selected_lbl = dict()
for i, b in enumerate(rlr.get_support()):
    if b == True:
        rlr_selected_lbl[labels[i]] = rlr.scores_[i]

rlr_lr_coefs = dict()
for i, b in enumerate(rlr.get_support()):
    if b == True:
        rlr_lr_coefs[labels[i]] = lr_coef_mean[0][i]

# Plots
plt.figure()
plt.plot(list(rlr_selected_lbl.values()), 'ko')
plt.xticks(
    np.arange(0, len(rlr_selected_lbl), 1),
    rlr_selected_lbl.keys(),
    rotation='vertical')
plt.title("Randomized Logistic Regression selection scores")
plt.tight_layout()

plt.figure()
plt.plot(list(rlr_lr_coefs.values()), 'ko')
plt.xticks(
    np.arange(0, len(rlr_lr_coefs), 1),
    rlr_lr_coefs.keys(),
    rotation='vertical')
plt.title("Selected coefficients from mean logistic regression")
plt.tight_layout()
