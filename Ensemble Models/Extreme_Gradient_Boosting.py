import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from pprint import pprint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
#from sklearn.ensemble import xgboost
# Read in data
df = pd.read_csv(r"Mumbai.csv")
# Drop columns with low correlation
df = df.drop(['Area', 'Carpet Area', 'Balconies'], axis=1)
# Drop tuples with missing values
df = df.dropna()
# Drop locations with frequency count <= 10
df = df.groupby('Location').filter(lambda x: len(x) >= 10)
# One-hot encode the data using pandas get_dummies
df = pd.get_dummies(df)
# Labels are the values we want to predict
labels = np.array(df['Price'])
# Remove the labels from the features
# axis 1 refers to the columns
df = df.drop('Price', axis=1)
# Saving feature names for later use
feature_list = list(df.columns)
# Convert to numpy array
features = np.array(df)
# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.2,
                                                                            random_state=0)
# Look at the shape of all the data
print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)


# Function to evaluate the model
def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} lakhs.'.format(np.mean(errors)))
    #print('Accuracy = {:0.2f}%.'.format(accuracy))

    return accuracy


# Create the parameter grid based on the results of random search
param_grid = {
    'max_depth': [50,60,70],
    'learning_rate':[0.02,0.025,0.03],
    'min_child_weight':[4,5.6,7],
    'gamma':[0.4,0.5,0.6],
    'subsample':[0.5,0.6,0.7,0.8,0.9],
    'colsample_bytree':[0.7,0.8,0.9],
    'reg_alpha':[0, 0.005, 0.01],
    'n_estimators':[100,500,700]
}
# Create a based model
xgbmodel = XGBRegressor(scale_pos_weight=1)
# Instantiate the grid search model
grid_search = GridSearchCV(estimator=xgbmodel, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
# Fit the grid search to the data
grid_search.fit(train_features, train_labels)
# Display the best parameters
print(grid_search.best_params_)
# Evaluate the model obtained by grid search
best_grid = grid_search.best_estimator_
grid_accuracy = evaluate(best_grid, test_features, test_labels)
print("Grid Accuracy= {0.2f}%.",format(grid_accuracy))
# Check improvement w.r.t. base model
#print('Improvement of {:0.2f}%.'.format(100 * (grid_accuracy - base_accuracy) / base_accuracy))

