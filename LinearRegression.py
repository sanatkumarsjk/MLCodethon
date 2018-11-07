# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 11:32:08 2018

@author: Trey
"""

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

import pandas as pd
import numpy as np

# Trying to use Region and Starting Median Salary
# to predict Mid-Career Median Salary

print("Region + Starting Salary -> Mid-Career Salary")
print("-"*50)
data = pd.read_csv("Data/clean-salaries-by-region.csv")

scaler = StandardScaler()
scaler.fit_transform(data[["Starting Median Salary", "Mid-Career Median Salary"]].astype('float64'))

X = data[["Starting Median Salary", "Region_California", "Region_Midwestern", 
         "Region_Northeastern", "Region_Southern", "Region_Western"]]

y = data["Mid-Career Median Salary"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

linear_clf = LinearRegression()

print("Cross Validation Scores:", cross_val_score(linear_clf, X_train, y_train, cv=3))

linear_clf.fit(X_train, y_train)

y_predict = linear_clf.predict(X_test)

print("RMSE on Test Set:", np.sqrt(mean_squared_error(y_test, y_predict)))


# Trying to use School Type and Starting Median Salary
# to predict Mid-Career Median Salary

print("\nSchool Type + Starting Salary -> Mid-Career Salary")
print("-"*50)
data = pd.read_csv("Data/clean-salaries-by-college-type.csv")

scaler = StandardScaler()
scaler.fit_transform(data[["Starting Median Salary", "Mid-Career Median Salary"]].astype('float64'))

X = data[["Starting Median Salary", "School Type_Engineering", "School Type_Ivy League",
          "School Type_Liberal Arts", "School Type_Party", "School Type_State"]]

y = data["Mid-Career Median Salary"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

linear_clf = LinearRegression()

print("Cross Validation Scores:", 
      cross_val_score(linear_clf, X_train, y_train, cv=3))

linear_clf.fit(X_train, y_train)

y_predict = linear_clf.predict(X_test)
print("RMSE on Test Set:", np.sqrt(mean_squared_error(y_test, y_predict)))