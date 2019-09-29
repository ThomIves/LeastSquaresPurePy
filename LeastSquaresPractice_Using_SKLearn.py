from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import sys


# Importing the dataset
df = pd.read_csv('50_Startups.csv')

# Segregating the data set
X = df.iloc[:, :-1].values
Y = df.iloc[:, 4].values


# Functions to help encode text / categorical data
def get_one_hot_encoding_for_one_column(X, i):
    col = X[:, i]
    labels = np.unique(col)
    ohe = OneHotEncoder(categories=[labels])

    return ohe


def apply_one_hot_encoder_on_matrix(ohe, X, i):
    col = X[:, i]
    tmp = ohe.fit_transform(col.reshape(-1, 1)).toarray()
    X = np.concatenate((X[:, :i], tmp[:, 1:], X[:, i+1:]), axis=1)

    return X


# Use the two functions just defined above to encode cities
ohe = get_one_hot_encoding_for_one_column(X, 3)
X = apply_one_hot_encoder_on_matrix(ohe, X, 3)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=0)

# Solve for the Least Squares Fit
lin_reg_model = LinearRegression()
lin_reg_model.fit(X_train, Y_train)
skl_predictions = lin_reg_model.predict(X_test)
print(skl_predictions)

# Data produced from sklearn least squares model
SKLearnData = [103015.20159796, 132582.27760816, 132447.73845175,
               71976.09851258, 178537.48221056, 116161.24230165,
               67851.69209676, 98791.73374687, 113969.43533013,
               167921.06569551]
