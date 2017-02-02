#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 19:20:31 2017

@author: au194693
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import (StratifiedKFold, cross_val_predict,
                                     cross_val_score, permutation_test_score)
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import scale
from sklearn.grid_search import GridSearchCV

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

ispc_X = np.empty([len(ispc_data_mean.subject.unique()), 84])

for i, sub in enumerate(subjects):
    fdata = ispc_data_mean[ispc_data_mean.subject == sub]
    ispc_X[i, :] = fdata.ISPC.values.reshape(-1)

X = np.concatenate((pow_X, ispc_X), axis=1)
y = np.concatenate((np.zeros(18), np.ones(18)))

X_scl = scale(X)

cv = StratifiedKFold(n_splits=9, shuffle=True)
grid_estimators = []
scores_list = []
n_jobs = 2

scores = []
coefs = []
Cs = []

param_grid = {
    "n_estimators": np.arange(1, 50, 1),
    "learning_rate": np.arange(0.01, 1, 0.1)
}

for train, test in cv.split(X, y):
    # lr = LogisticRegression(C=1)
    # lr.fit(X[train], y[train])
    clf = AdaBoostClassifier()
    grid = GridSearchCV(
        clf,
        param_grid=param_grid,
        scoring="accuracy",
        verbose=1,
        n_jobs=n_jobs,
        cv=10)
    grid.fit(X[train], y[train])

    y_pred = grid.predict(X[test])

    grid_estimators.append(grid.best_estimator_)
    scores.append(roc_auc_score(y[test], y_pred))

ada_learning_rate = np.median(
    np.asarray([est.learning_rate for est in grid_estimators]))

ada_n_estimators = int(
    np.median(np.asarray([est.n_estimators for est in grid_estimators])))

ada_mean = AdaBoostClassifier(
    n_estimators=ada_n_estimators, learning_rate=ada_learning_rate)

cv_scores = cross_val_score(ada_mean, X, y, scoring="roc_auc", cv=cv)
cv_prob = cross_val_predict(ada_mean, X, y, cv=cv)

score, permutation_scores, pvalue = permutation_test_score(
    ada_mean, X, y, scoring="roc_auc", cv=cv, n_permutations=100, n_jobs=1)

feat_imp = np.asarray([est.feature_importances_ for est in grid_estimators])

scores = []
coefs = []
Cs = []

param_grid = {
    "n_estimators": np.arange(1, 50, 1),
    "learning_rate": np.arange(0.01, 1, 0.1),
    "max_depth": np.arange(1, 8, 1)
}

for train, test in cv.split(X, y):
    # lr = LogisticRegression(C=1)
    # lr.fit(X[train], y[train])
    clf = GradientBoostingClassifier()
    grid = GridSearchCV(
        clf,
        param_grid=param_grid,
        scoring="roc_auc",
        verbose=1,
        n_jobs=n_jobs,
        cv=6)
    grid.fit(X[train], y[train])

    y_pred = grid.predict(X[test])

    grid_estimators.append(grid.best_estimator_)
    scores.append(roc_auc_score(y[test], y_pred))

scores = []
coefs = []
Cs = []

for train, test in cv.split(ispc_X, y):
    # lr = LogisticRegression(C=1)
    # lr.fit(X[train], y[train])
    clf = LogisticRegressionCV()
    clf.fit(ispc_X[train], y[train])
    y_pred = clf.predict(ispc_X[test])

    scores.append(roc_auc_score(y[test], y_pred))
    coefs.append(clf.coef_)
    Cs.append(clf.Cs)

lr_mean = LogisticRegression()
lr_mean.coef_ = np.asarray(coefs).mean(axis=0)
lr_mean.C = np.asarray(Cs).mean()

lr_coef_mean = np.asarray(coefs).mean(axis=0)
lr_coef_std = np.asarray(coefs).std(axis=0)

score, permutation_scores, pvalue = permutation_test_score(
    lr_mean, X, y, scoring="accuracy", cv=cv, n_permutations=100, n_jobs=1)

matplotlib.rc('xtick', labelsize=5)
plt.figure()
plt.plot(lr_coef_mean.T, 'b', linewidth=1)
plt.plot(lr_coef_mean.T + lr_coef_std.T, 'b--', linewidth=1)
plt.plot(lr_coef_mean.T - lr_coef_std.T, 'b--', linewidth=1)
plt.xticks(np.arange(0, 168, 1), labels, rotation='vertical')

plt.margins(0.4)
# Tweak spacing to prevent clipping of tick-labels
plt.subplots_adjust(bottom=0.15)
