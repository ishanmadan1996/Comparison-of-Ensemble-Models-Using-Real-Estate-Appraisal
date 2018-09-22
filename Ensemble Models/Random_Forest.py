import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from pprint import pprint
from sklearn.model_selection import GridSearchCV
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
rf = RandomForestRegressor(n_estimators = 10, random_state = 0)
# Train the model on training data
rf.fit(train_features, train_labels)
# Function to evaluate the model
def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} lakhs.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    
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
	# Number of features to consider at every split
	max_features = [0.5,0.6,0.7]
	# Maximum number of levels in tree
	max_depth = [35,40,45]
	# Minimum number of samples required to split a node
	min_samples_split = [2,3,4]
	# Create the random grid
	param_grid = {'max_features': max_features,
	              'max_depth': max_depth,
	              'min_samples_split': min_samples_split,
	             }
	pprint(param_grid)
	# Use the random grid to search for best hyperparameters
	# First create the base model to tune
	rf = RandomForestRegressor(n_estimators=110, min_samples_leaf=1)
	# Random search of parameters, using 3 fold cross validation
	# Search across 100 different combinations, and use all available cores
	grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, cv = 3, verbose=2, n_jobs = -1)
	# Fit the random search model
	grid_search.fit(train_features, train_labels)
	# Print the results
	report(grid_search.cv_results_)
	# Display the best parameters
	print(grid_search.best_params_)
	# Store results in a .csv file
	results_df = pd.DataFrame(grid_search.cv_results_)
	results_df.to_csv("forest_grid.csv")
	# Evaluate the base model
	base_model = RandomForestRegressor(n_estimators = 10, random_state = 0)
	base_model.fit(train_features, train_labels)
	base_accuracy = evaluate(base_model, test_features, test_labels)
	# Evaluate the model obtained by grid search
	best_grid = grid_search.best_estimator_
	grid_accuracy = evaluate(best_grid, test_features, test_labels)
	# Check improvement w.r.t. base model
	print('Improvement of {:0.2f}%.'.format( 100 * (grid_accuracy - base_accuracy) / base_accuracy))
if __name__ == "__main__":
    main()
