# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 12:49:53 2018

@author: Trey
"""

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

import pandas as pd
import numpy as np

data = pd.read_csv("Data/clean-salaries-by-college-type.csv")

X = data[["Starting Median Salary", "School Type_Engineering", "School Type_Ivy League",
          "School Type_Liberal Arts", "School Type_Party", "School Type_State"]]

y = data["Mid-Career Median Salary"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

rf_clf = RandomForestRegressor(n_estimators=100)

print("Cross Validation Scores:", 
      cross_val_score(rf_clf, X_train, y_train, cv=3))

rf_clf.fit(X_train, y_train)

y_predict = rf_clf.predict(X_test)
print("Test RMSE:", np.sqrt(mean_squared_error(y_test, y_predict)))