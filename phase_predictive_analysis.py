import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import ensemble
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import (ShuffleSplit, cross_val_score,
                                      train_test_split)
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

results_all = pd.read_csv("/Users/au194693/projects/agency_connectivity/data/"
                          + "ispc_combinded_wide_beta.csv")

# combine condition to predict testing from learning
res_testing = results_all[results_all.condition == "testing"]
res_learning = results_all[results_all.condition == "learning"]

res_combined = res_testing.copy()

res_combined["ba_1_1_learning"] = res_learning["ba_1_1"].get_values()
res_combined["ba_4_4_learning"] = res_learning["ba_4_4"].get_values()
res_combined["ba_1_4_l_learning"] = res_learning["ba_1_4_l"].get_values()
res_combined["ba_1_4_r_learning"] = res_learning["ba_1_4_r"].get_values()
res_combined["trial_status_learning"] = res_learning[
    "trial_status"].get_values()

tmp = res_combined[res_combined.step == 8]
tmp = tmp[(tmp.trial_status == True) & (tmp.trial_status_learning == True)]
# tmp = tmp[tmp.subject == "p9"]

X = tmp[["ba_1_1", "ba_4_4", "ba_1_4_l", "ba_1_4_r", "ba_1_1_learning",
         "ba_4_4_learning", "ba_1_4_l_learning",
         "ba_1_4_r_learning"]].get_values()
y = tmp[["binding"]].get_values().squeeze()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

feature_names = ["ba_1_1", "ba_4_4", "ba_1_4_l", "ba_1_4_r", "ba_1_1_learning",
                 "ba_4_4_learning", "ba_1_4_l_learning", "ba_1_4_r_learning"]
# Grid search

clf = ensemble.GradientBoostingRegressor()
cv = ShuffleSplit(len(y), n_iter=5., test_size=0.25)
params = {"gradientboostingregressor__n_estimators": np.arange(1, 50, 1),
          "gradientboostingregressor__max_depth": np.arange(1, 7, 1),
          "gradientboostingregressor__learning_rate": np.arange(0.01, 1, 0.1)}

scaler_pipe = make_pipeline(StandardScaler(), clf)
grid = GridSearchCV(scaler_pipe, param_grid=params, verbose=1)

grid.fit(X_train, y_train)

params = {"learning_rate": 0.61, "n_estimators": 2, "max_depth": 1}

clf = ensemble.GradientBoostingRegressor(**params)

clf.fit(X_train, y_train)
mse = mean_squared_error(y_test, clf.predict(X_test))
print("MSE: %.4f" % mse)

###############################################################################
# Plot training deviance

# compute test set deviance
test_score = np.zeros((params['n_estimators'], ), dtype=np.float64)

for i, y_pred in enumerate(clf.staged_predict(X_test)):
    test_score[i] = clf.loss_(y_test, y_pred)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Deviance')
plt.plot(
    np.arange(params['n_estimators']) + 1,
    clf.train_score_,
    'b-',
    label='Training Set Deviance')
plt.plot(
    np.arange(params['n_estimators']) + 1,
    test_score,
    'r-',
    label='Test Set Deviance')
plt.legend(loc='upper right')
plt.xlabel('Boosting Iterations')
plt.ylabel('Deviance')

###############################################################################
# Plot feature importance
feature_importance = clf.feature_importances_
# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.subplot(1, 2, 2)
plt.barh(pos, feature_importance[sorted_idx], align='center')
#plt.yticks(pos, boston.feature_names[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()
