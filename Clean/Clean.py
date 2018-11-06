# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 10:35:03 2018

@author: Trey
"""

import pandas as pd

# Function to drop features, convert string features
# of the form "$<number>.00" to ints of the form <number>,
# and one hot encode categorical features
def clean_data(filename, drop_features, string_features, cat_features):
    # Get data to be cleaned
    data = pd.read_csv("../Data/" + filename)

    # Drop features
    data = data.drop(drop_features, axis=1)

    # Convert string data to int 
    # Example: "$70,400.00" -> 70400
    for index, row in data.iterrows():
        for feature in string_features:
            data.at[index, feature] = int(row[feature][1:-3].replace(",", ""))
        
    # Encode categorical features
    data = pd.get_dummies(data, columns=cat_features)

    # Write back to file
    data.to_csv("../Data/clean-" + filename, index=False)

################### salaries-by-region.csv  ################################

# Dropping Mid-Career 10th Percentile and 90th Percentile
# because almost 15% of them are missing
drop_features = ["School Name", "Mid-Career 10th Percentile Salary", "Mid-Career 90th Percentile Salary"]

string_features = ["Starting Median Salary", "Mid-Career Median Salary",
                   "Mid-Career 25th Percentile Salary", "Mid-Career 75th Percentile Salary"]

cat_features = ["Region"]

clean_data("salaries-by-region.csv", drop_features, string_features,
           cat_features)

################### salaries-by-college-type.csv  ################################

# Dropping Mid-Career 10th Percentile and 90th Percentile
# because almost 15% of them are missing
drop_features = ["School Name", "Mid-Career 10th Percentile Salary", "Mid-Career 90th Percentile Salary"]

string_features = ["Starting Median Salary", "Mid-Career Median Salary",
                   "Mid-Career 25th Percentile Salary", "Mid-Career 75th Percentile Salary"]

cat_features = ["School Type"]

clean_data("salaries-by-college-type.csv", drop_features, string_features,
           cat_features)

################### degrees-that-pay-back.csv  ################################
drop_features = []

string_features = ["Starting Median Salary", "Mid-Career Median Salary", "Mid-Career 10th Percentile Salary",
                   "Mid-Career 25th Percentile Salary", "Mid-Career 75th Percentile Salary",
                   "Mid-Career 90th Percentile Salary"]

cat_features = []

clean_data("degrees-that-pay-back.csv", drop_features, string_features,
           cat_features)







