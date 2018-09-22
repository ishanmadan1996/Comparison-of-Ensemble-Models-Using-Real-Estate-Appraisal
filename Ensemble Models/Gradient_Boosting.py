import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from pprint import pprint
from sklearn.model_selection import RandomizedSearchCV
# Read in data
df = pd.read_csv("Mumbai.csv")
# Drop columns with low correlation
df = df.drop(['Area','Carpet Area','Balconies'], axis = 1)
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
df= df.drop('Price', axis = 1)
# Saving feature names for later use
feature_list = list(df.columns)
# Convert to numpy array
features = np.array(df)
# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.2, random_state = 0)
# Look at the shape of all the data
print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)
# Scale the attributes
scaler = StandardScaler()
# Fit on training set only
scaler.fit(train_features)
# Apply transform to both the training set and the test set
train_features = scaler.transform(train_features)
test_features = scaler.transform(test_features)
# Make an instance of the Model
pca = PCA(.90)
# Fit PCA on training set
pca.fit(train_features)
# Apply the mapping (transform) to both the training set and the test set
train_features = pca.transform(train_features)
test_features = pca.transform(test_features)
# Instantiate model with 100 decision trees
rf = GradientBoostingRegressor(n_estimators = 10, random_state = 0)
# Train the model on training data
rf.fit(train_features, train_labels)

# Function to evaluate the model
def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Mean Absolute Perccentage Error: {:0.4f}%.'.format(mape))
    print('Accuracy = {:0.4f}%.'.format(accuracy))
    
    return accuracy

# Evaluate the model
accuracy = evaluate(rf, test_features, test_labels)
# Begin hyperparameter tuning
# Look at parameters used by our current forest
print('Parameters currently in use:\n')
pprint(rf.get_params())
# Utility function to report best scores
def report(results, n_top=10):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
def main():
  learning_rate = [0.2, 0.3, 0.4, 0.5, 0.6]
  # Number of trees in random forest
  n_estimators = [int(x) for x in np.linspace(10, 110, num = 5)]
  # Number of features to consider at every split
  max_features = [0.3, 0.6, 0.9]
  # Maximum number of levels in tree
  max_depth = [int(x) for x in np.linspace(10, 50, num = 5)]
  # Minimum number of samples required to split a node
  min_samples_split = [2, 5, 10]
  # Minimum number of samples required at each leaf node
  min_samples_leaf = [1, 4, 7]
  # Create the random grid
  random_grid = {'learning_rate': learning_rate, 
                 'n_estimators': n_estimators,
                 'max_features': max_features,
                 'max_depth': max_depth,
                 'min_samples_split': min_samples_split,
                 'min_samples_leaf': min_samples_leaf,
                }
  pprint(random_grid)
  # Use the random grid to search for best hyperparameters
  # First create the base model to tune
  rf = GradientBoostingRegressor()
  # Random search of parameters, using 3 fold cross validation
  # Search across 100 different combinations, and use all available cores
  rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 150, cv = 3, verbose=2, random_state=0, n_jobs = -1, return_train_score = 'true')
  # Fit the random search model
  rf_random.fit(train_features, train_labels)
  # Print the results
  report(rf_random.cv_results_)
  # Display the best parameters
  print(rf_random.best_params_)
  # Save the results to a .csv file
  results_df = pd.DataFrame(rf_random.cv_results_)
  results_df.to_csv('gbr.csv')
  # Evaluate the base model
  base_model = GradientBoostingRegressor(n_estimators = 10, random_state = 0)
  base_model.fit(train_features, train_labels)
  base_accuracy = evaluate(base_model, test_features, test_labels)
  # Evaluate the model obtained by random search
  best_random = rf_random.best_estimator_
  random_accuracy = evaluate(best_random, test_features, test_labels)
  # Check improvement w.r.t. base model
  print('Improvement of {:0.2f}%.'.format( 100 * (random_accuracy - base_accuracy) / base_accuracy))
if __name__ == "__main__":
    main()
