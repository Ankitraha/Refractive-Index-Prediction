import tensorflow as tf
from tensorflow import keras
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer
from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.preprocessing import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.neighbors import KNeighborsRegressor
from statsmodels.regression.linear_model import OLS
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
import keras.backend as K
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


# Load the data
df_train = pd.read_csv('train_refractiveindex.csv')
df_train = df_train.drop(columns=["Molar Mass","Molar Volume","Name"])
df_train = df_train.sample(frac=1).reset_index(drop=True)
# Split the training data into features and labels
X_train = df_train.drop(columns=['Refractive Index'])
y_train = df_train['Refractive Index']

df_train.to_csv('submission_to_submit3.csv', index=False)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train)


from statsmodels.stats.stattools import durbin_watson

# Fit a linear regression model to the data
model = OLS(y_train, X_train).fit()

# Perform the Durbin-Watson test
dw_stat = durbin_watson(model.resid)

# Print the Durbin-Watson statistic
print("Durbin-Watson statistic: ", dw_stat)

# Check for autocorrelation
if dw_stat < 2 or dw_stat > 2:
    print("There is autocorrelation in the data, the model is likely non-linear")
else:
    print("There is no autocorrelation in the data, the model is likely linear")
# Define the model
model = GradientBoostingRegressor(n_estimators=500, learning_rate=0.01, max_depth=5, random_state=0)

# Fit the model to the training data
model.fit(X_train, y_train)


# Make predictions on the validation set
predictions = model.predict(X_val)

# Calculate the mean squared error
mse = mean_squared_error(y_val, predictions)

# Take the square root of the mean squared error to get the root mean squared error
rmse = np.sqrt(mse)
print("RMSE:", rmse)
# Calculate the R-squared score
r2 = r2_score(y_val, predictions)

# Print the R-squared score
print("R-squared:", r2)
df_test = pd.read_csv('test_ri.csv')
df_1=df_test['Name']
df_test = df_test.drop(columns=["Molar Mass","Molar Volume","Name"])
test_predictions = model.predict(df_test)
df_test['Refractive Index'] = test_predictions
df_test = df_test[['Refractive Index']]
# Merge the columns of the two dataframes
df_test = pd.concat([df_1, df_test], axis=1)
df_test.to_csv('ri.csv', index=False)
