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
y = dataset.iloc[:, 3].values

# Take care of missing data
from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values ='NaN', strategy = 'mean')

# means of the column or rows so 0 or 1
# column 1 and 2 but needs from x to y that's why its 1:3
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
# so it works, prints the new updated csv
# print(X)

# encoding categorical data
# ex; yes, no, spain, france, germany
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X = LabelEncoder()

# not the whole matrix, just the first column
# labelEncoder_X.fit_transform(X[:, 0])
X[:, 0] = labelEncoder_X.fit_transform(X[:, 0])

# prevent the machine learning equations to not think germ greater than france and spain...
# DUMMY variables; try to make 3 diff columns and if france then 1 and the rest 0 0 sort of boolean
oneHotEncoder = OneHotEncoder(categorical_features= [0])
X = oneHotEncoder.fit_transform(X).toarray()
# print(X)

# Splitting the dataset into Training set and Test set
from sklearn.model_selection import train_test_split
# test_size§ = 0.2 means its the 20% of the data, which is recommended as well
# because 20% of the size 8 observations of the train set and 2 observations of the test set should be
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state = 0)
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
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)