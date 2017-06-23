#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 19:20:31 2017

@author: au194693
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     permutation_test_score)
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import scale
from sklearn.grid_search import GridSearchCV

# import xgboost as xgb

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

# AdaBoost Classifier
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
        scoring="roc_auc",
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

score, permutation_scores, pvalue = permutation_test_score(
    ada_mean, X, y, scoring="roc_auc", cv=cv, n_permutations=100, n_jobs=1)

feat_imp = np.asarray([est.feature_importances_ for est in grid_estimators])

# XGB Classifier
scores = []
grid_estimators = []

param_grid = {
    "n_estimators": np.arange(1, 50, 1),
    "learning_rate": np.arange(0.01, 1, 0.1),
    "max_depth": np.arange(1, 8, 1)
}

for train, test in cv.split(X_scl, y):
    # lr = LogisticRegression(C=1)
    # lr.fit(X[train], y[train])
    clf = xgb.XGBClassifier()
    grid = GridSearchCV(
        clf,
        param_grid=param_grid,
        scoring="roc_auc",
        verbose=1,
        n_jobs=n_jobs,
        cv=10)
    grid.fit(X_scl[train], y[train])

    y_pred = grid.predict(X_scl[test])

    grid_estimators.append(grid.best_estimator_)
    scores.append(roc_auc_score(y[test], y_pred))

xgb_learning_rate = np.mean(
    np.asarray([est.learning_rate for est in grid_estimators]))

xgb_n_estimators = int(
    np.mean(np.asarray([est.n_estimators for est in grid_estimators])))

xgb_max_depth = int(
    np.median(np.asarray([est.max_depth for est in grid_estimators])))

xgb_mean = xgb.XGBClassifier(
    learning_rate=xgb_learning_rate,
    max_depth=xgb_max_depth,
    n_estimators=xgb_n_estimators)

xgb_cv_score = cross_val_score(xgb_mean, X_scl, y, scoring="roc_auc", cv=cv)
print("mean score: %s (std: %s)" % (xgb_cv_score.mean(), xgb_cv_score.std()))

score, permutation_scores, pvalue = permutation_test_score(
    xgb_mean,
    X_scl,
    y,
    scoring="roc_auc",
    cv=cv,
    n_permutations=2000,
    n_jobs=2)

# Logistic Regression with cross validation for C
scores = []
coefs = []
Cs = []
LRs = []

for train, test in cv.split(X_scl, y):
    clf = LogisticRegression(C=1)
    # lr.fit(X[train], y[train])
    #clf = LogisticRegressionCV()
    clf.fit(X[train], y[train])
    y_pred = clf.predict(X_scl[test])

    scores.append(roc_auc_score(y[test], y_pred))
    coefs.append(clf.coef_)
#    Cs.append(clf.C_)
    LRs.append(clf)

lr_mean = LogisticRegression()
lr_mean.coef_ = np.asarray(coefs).mean(axis=0)
# lr_mean.C = np.asarray(Cs).mean()
lr_mean.intercept_ = np.asarray([est.intercept_ for est in LRs]).mean()

lr_coef_mean = np.asarray(coefs).mean(axis=0)
lr_coef_std = np.asarray(coefs).std(axis=0)

cv_scores = cross_val_score(lr_mean, X_scl, y, scoring="roc_auc", cv=cv)


from sklearn.feature_selection import SelectPercentile, f_classif

selector = SelectPercentile(f_classif, percentile=5)
selector.fit(X, y)
scores = -np.log10(selector.pvalues_)
scores /= scores.max()
X_indices = np.arange(X_scl.shape[-1])
plt.bar(X_indices - .45, scores, width=.2,
        label=r'Univariate score ($-Log(p_{value})$)')

from sklearn.feature_selection import RFECV
rfecv = RFECV(estimator=lr_mean, step=1, cv=StratifiedKFold(9),
              scoring='roc_auc')
rfecv.fit(X_scl, y)

print("Optimal number of features : %d" % rfecv.n_features_)

score, permutation_scores, pvalue = permutation_test_score(
    lr_mean, X, y, scoring="roc_auc", cv=cv, n_permutations=2000, n_jobs=2)

lr_coef_or_mean = np.exp(lr_coef_mean)
lr_coef_or_std = np.exp(lr_coef_std)

plt.rc('xtick', labelsize=5)
plt.figure()
plt.plot(lr_coef_mean.T, 'b', linewidth=1)
plt.plot(lr_coef_mean.T + lr_coef_sem.T, 'b--', linewidth=1)
plt.plot(lr_coef_mean.T - lr_coef_sem.T, 'b--', linewidth=1)
plt.xticks(np.arange(0, 168, 1), labels, rotation='vertical')

plt.margins(0.4)
# Tweak spacing to prevent clipping of tick-labels
plt.subplots_adjust(bottom=0.15)
