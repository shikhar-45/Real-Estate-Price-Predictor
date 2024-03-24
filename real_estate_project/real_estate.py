# Real Estate Price Predictor
# Aim is to make a model which would predict the value of a property with different features as the input to the model.

import pandas as pd

housing = pd.read_csv("data.csv")
housing.head()
housing.info()
housing['CHAS'].value_counts()
housing.describe()

import numpy as np

from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
print(f"Rows in train set : {len(train_set)}\nRows in test set : {len(test_set)}\n")

from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['CHAS']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

housing = strat_train_set.copy()

# Looking for Correlations

corr_matrix = housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False)

from pandas.plotting import scatter_matrix
attributes = ["MEDV", "RM", "ZN", "LSTAT"]
scatter_matrix(housing[attributes], figsize = (12,8))
housing.plot(kind="scatter", x="RM", y="MEDV", alpha=0.8)

# Trying out Attribute Combinations

housing["TAXRM"] = housing['TAX']/housing['RM']
housing.head()
corr_matrix = housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False)
housing.plot(kind="scatter", x="TAXRM", y="MEDV", alpha=0.8)
housing = strat_train_set.drop("MEDV", axis=1)
housing_labels = strat_train_set["MEDV"].copy()

# Missing Attributes

median = housing["RM"].median() # computing median
median
housing["RM"].fillna(median) # original housing dataframe remains unchanged
housing.shape
housing.describe() # Before we started filling missing attributes

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy = "median")
imputer.fit(housing)
imputer.statistics_
imputer.statistics_.shape
X = imputer.transform(housing)
housing_tr = pd.DataFrame(X, columns=housing.columns)
housing_tr.describe()

# Creating a Pipeline

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler()),
])
housing_num_tr = my_pipeline.fit_transform(housing)
housing_num_tr.shape

# Selecting a desired model for Real Estate Company

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
# model = LinearRegression()
# model = DecisionTreeRegressor()
model = RandomForestRegressor()
model.fit(housing_num_tr, housing_labels)
some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
prepared_data = my_pipeline.transform(some_data)
model.predict(prepared_data)
list(some_labels)

# Evaluating the model

from sklearn.metrics import mean_squared_error
housing_predictions = model.predict(housing_num_tr)
mse = mean_squared_error(housing_labels, housing_predictions)
rmse =  np.sqrt(mse)
mse

# Using better evaluation technique - Cross Validation

from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, housing_num_tr, housing_labels, scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)
rmse_scores

def print_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())
print_scores(rmse_scores)

# Saving the model

from joblib import dump, load
dump(model, 'Real_Estate.joblib')

# Testing the model on test data

X_test = strat_test_set.drop("MEDV", axis=1)
Y_test = strat_test_set["MEDV"].copy()
X_test_prepared = my_pipeline.transform(X_test)
final_predictions = model.predict(X_test_prepared)
final_mse = mean_squared_error(Y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
print(final_predictions, list(Y_test))
print(f"Final value of rmse = {final_rmse}")
# print(prepared_data[0])

def predict_price(features):
    return(model.predict(features)) 

# Using the model

# from joblib import dump, load
# import numpy as np
# model = load('Real_Estate.joblib')
# features = np.array([[-0.43942006,  3.12628155, -1.12165014, -0.27288841, -1.42262747,
#        -0.23979304, -1.31238772,  2.61111401, -1.0016859 , -0.5778192 ,
#        -0.97491834,  0.41164221, -0.86091034]])

# model.predict(features) 22.49