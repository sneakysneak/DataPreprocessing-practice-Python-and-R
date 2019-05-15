# for maths tools
import numpy as np
# plot charts so to display data
import matplotlib.pyplot as plt
# import and manage datasets
import pandas as pd

dataset = pd.read_csv('Data.csv')
# Take all the columns except the last one -1
X = dataset.iloc[:, :-1].values
# prints out only the 3rd column
Y = dataset.iloc[:, 3].values

# Splitting the dataset into Training set and Test set
from sklearn.model_selection import train_test_split
# test_size = 0.2 means its the 20% of the data, which is recommended as well
# because 20% of the size 8 observations of the train set and 2 observations of the test set should be
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.2, random_state = 0)
# print(X, Y, dataset, X_train, X_test, Y_train, Y_test)
# print('Test set X', X_test)

# Scaling dataset
# must put data to the same "range" "scale" in order to use the euclidean distance
# euclidean distance is the square root of the sum of the square's coordinates
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
# transform x train
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)