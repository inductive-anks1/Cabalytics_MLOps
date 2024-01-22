# train_model.py
import pathlib
import sys
import joblib
import mlflow
import pickle
import yaml
import os


import pandas as pd
from hyperopt import hp
from sklearn.model_selection import train_test_split
from hyperopt.pyll.base import scope
from sklearn.metrics import mean_squared_error
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, space_eval
from xgboost import XGBRegressor

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score,mean_absolute_error, mean_squared_error

from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor,ExtraTreesRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

def train_model(X_train, y_train, n_estimators, max_depth, max_samples, max_features, seed):
    # Define the preprocessing and model steps
    step1 = ColumnTransformer(transformers=[
        ('col_tnf', OneHotEncoder(sparse=False, drop='first'), [0, 1, 2, 3])
    ], remainder='passthrough')

    step2 = RandomForestRegressor(n_estimators=n_estimators,
                                  random_state=seed,
                                  max_samples=max_samples,
                                  max_features=max_features,
                                  max_depth=max_depth)

    # Create the pipeline
    pipe = Pipeline([
        ('step1', step1),
        ('step2', step2)
    ])

    # Fit the model
    pipe.fit(X_train, y_train)
    return pipe

def save_model(model, output_path):
    # Ensure the output path ends with '/'
    if not output_path.endswith('/'):
        output_path += '/'
    
    # Save the trained model to the specified output path
    with open(output_path + 'model.pkl', 'wb') as file:
        pickle.dump(model, file)

def drop_nan_values(X, y):
    # Check for NaN values in the target variable
    nan_mask = y.isna()
    if nan_mask.sum() > 0:
        print(f"Dropping {nan_mask.sum()} rows with NaN values in target variable")
        X = X[~nan_mask]
        y = y[~nan_mask]
    return X, y


def main():
    params_file = 'params.yaml'
    params = yaml.safe_load(open(params_file))["train_model"]

    # The input file path is provided as a command line argument
    # which should be "data/processed" based on the dvc.yaml cmd
    input_file = sys.argv[1]
    data_path = input_file  # This is already a relative path from the project root
    output_path = 'models'
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    
    # Join the data_path with the actual CSV filename
    train_features_path = os.path.join(data_path, 'Cab_Data_Train.csv')
    train_features = pd.read_csv(train_features_path)
    
    X_train = train_features.drop('Cab_Price', axis=1)
    y_train = train_features['Cab_Price']

    # Handle NaN values
    X_train, y_train = drop_nan_values(X_train, y_train)

    trained_model = train_model(X_train, y_train, params['n_estimators'], params['max_depth'], 
                                params['max_samples'], params['max_features'], params['seed'])
    
    save_model(trained_model, output_path)

if __name__ == "__main__":
    main()