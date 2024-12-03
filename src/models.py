import csv
import numpy as np
import pandas as pd
import os

#Define default features and target for UI implementation
features = ['Hours_Studied', 'Attendance']
target = 'Exam_Score'
selected_indices = [0]
target_index = [1]
#Init values for these measures
lasso_error, theta_lasso, ridge_error, theta_ridge, poly_error, theta_poly = 0.0, [], 0.0, [], 0.0, []
X_mean, X_std, y_mean, y_std = 1.0, 1.0, 1.0, 1.0
#X and y need to be global if we have functionality to add to them
X = np.zeros((1, len(features)))
y = np.zeros((1, 1))

#Also, need to be able to define when the models have actually been trained.
has_data = False


#ADDED BY CHELSEA - Updaters for global variables.
def update_xy(Xnew, ynew):
    global X, y
    X, y = Xnew, ynew
def update_indices(selected, target):
    global selected_indices, target_index
    selected_indices, target_index = selected, target
def update_stats(Xm, Xs, ym, ys):
    global X_mean, X_std, y_mean, y_std
    X_mean, X_std, y_mean, y_std = Xm, Xs, ym, ys
def update_modelresults(le, lt, re, rt, pe, pt):
    global lasso_error, theta_lasso, ridge_error, theta_ridge, poly_error, theta_poly
    lasso_error, theta_lasso, ridge_error, theta_ridge, poly_error, theta_poly = le, lt, re, rt, pe, pt

# Load CSV data into NumPy arrays
#Edited: 'replace' determines whether this dataset will be replacing the existing set,
#'predict' determines whether the file is used for prediction or training.
def load_data(file_path, replace=True, predicting=False, selected_columns=features, target_column=target):
    data = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        if predicting:
            headers = next(reader)  # Read the header row
            selected_indices = [headers.index(col) for col in selected_columns]
            for row in reader:
                data.append([float(row[i]) if row[i].isdigit() else row[i] for i in selected_indices])
        else:
            headers = next(reader)  # Read the header row
            selected_indices = [headers.index(col) for col in selected_columns]
            target_index = headers.index(target_column)
            update_indices(selected_indices, target_index)
        
            for row in reader:
                data.append([float(row[i]) if row[i].isdigit() else row[i] for i in selected_indices + [target_index]])
    data = np.array(data, dtype=object)
    
    if predicting:
        X = np.array(data, dtype=float)  # Prediction input
        return predict_all(X)
    else:
        X = np.array(data[:, :-1], dtype=float)  # Features
        y = np.array(data[:, -1], dtype=float)  # Target variable
        if replace:
            update_xy(X, y)
        else:
            concat_data(X, y)
    #return X, y

#ADDED BY CHELSEA - For combining new entries onto a data set
def concat_data(Xnew, ynew):
    global X, y
    if has_data:
        Xn = np.vstack((X, Xnew))
        yn = np.concatenate((y, ynew), axis=0)
    else:
        Xn = Xnew
        yn = ynew
    update_xy(Xn,yn)
    #return X, y

#ADDED BY CHELSEA - For reducing a single student record to the selected features/target,
#then adding to the overall sets
#Essentially the above but for a single row without a file source.
def load_single(input, predicting=False, selected_columns=features, target_column=target):
    if predicting:
        data = [[float(input[i]) if input[i].isdigit() else input[i] for i in selected_indices]]
        data = np.array(data, dtype=object)
        X_input = np.array(data, dtype=float)  # Prediction input
        return predict_all(X_input)
    else:
        data = [[float(input[i]) if input[i].isdigit() else input[i] for i in selected_indices + [target_index]]]
        data = np.array(data, dtype=object)
        Xnew = np.array(data[:, :-1], dtype=float)  # Features
        ynew = np.array(data[:, -1], dtype=float)  # Target variable
        concat_data(Xnew, ynew)

    

def normalize(X, y):
    """Normalize features and target to have mean 0 and standard deviation 1."""
    X_mean = np.mean(X, axis=0) # Mean of each column
    X_std = np.std(X, axis=0) # Standard deviation of each column
    y_mean = np.mean(y) # Mean of target
    y_std = np.std(y) # Standard deviation of target
    
    X_normalized = (X - X_mean) / X_std # Normalized features
    y_normalized = (y - y_mean) / y_std # Normalized target
    return X_normalized, y_normalized, X_mean, X_std, y_mean, y_std # Return the normalization parameters



def lasso_regression(X, y, alpha, iterations=1000, learning_rate=0.001):
    m, n = X.shape
    theta = np.zeros(n) # Initialize the weights
    
    for _ in range(iterations):
        predictions = X @ theta # Make predictions
        mse_loss = np.mean((y - predictions) ** 2)  # MSE
        gradient = (-2 / m) * (X.T @ (y - predictions)) + (alpha / m) * np.sign(theta) # L1 regularization, L1 norm
        theta -= learning_rate * gradient # Update the weights
    
    return theta


def ridge_regression(X, y, alpha, iterations=1000, learning_rate=0.001):
    m, n = X.shape
    theta = np.zeros(n)
    
    for _ in range(iterations):
        predictions = X @ theta
        mse_loss = np.mean((y - predictions) ** 2)  # MSE
        gradient = (-2 / m) * (X.T @ (y - predictions)) + (2 * alpha / m) * theta # L2 regularization, squared L2 norm
        theta -= learning_rate * gradient
    
    return theta

def polynomial_features(X, degree):
    """Generate polynomial features up to the given degree."""
    X_poly = X
    for d in range(2, degree + 1):
        X_poly = np.hstack((X_poly, X ** d)) # Add higher-order features
    return X_poly

def polynomial_regression(X, y, degree):
    """Train polynomial regression using the normal equation."""
    X_poly = polynomial_features(X, degree) # Generate polynomial features
    X_poly, _, _, _, _, _ = normalize(X_poly, y)  # Normalize expanded features
    theta = np.linalg.inv(X_poly.T @ X_poly) @ X_poly.T @ y # Normal equation
    return theta


def cross_validation_with_theta(X, y, model, model_params, folds=5, y_mean=0, y_std=1):
    fold_size = len(X) // folds
    errors = []
    best_theta = None
    
    for i in range(folds):
        # Split data into train and validation sets
        X_val = X[i * fold_size:(i + 1) * fold_size]
        y_val = y[i * fold_size:(i + 1) * fold_size]
        X_train = np.vstack((X[:i * fold_size], X[(i + 1) * fold_size:]))
        y_train = np.hstack((y[:i * fold_size], y[(i + 1) * fold_size:]))
        
        # Train the model
        if model == "lasso":
            theta = lasso_regression(X_train, y_train, **model_params)
        elif model == "ridge":
            theta = ridge_regression(X_train, y_train, **model_params)
        elif model == "polynomial":
            degree = model_params['degree']
            theta = polynomial_regression(X_train, y_train, degree)
        
        # Predict and calculate error (denormalize predictions)
        if model == "polynomial":
            X_val_poly = polynomial_features(X_val, model_params['degree'])
            X_val_poly, _, _, _, _, _ = normalize(X_val_poly, y_val)
            predictions = X_val_poly @ theta
        else:
            predictions = X_val @ theta
        
        predictions = predictions * y_std + y_mean  # Denormalize predictions
        error = np.mean((y_val * y_std + y_mean - predictions) ** 2)  # Calculate denormalized MSE
        errors.append(error)
        
        # Store the last theta (trained on the last fold)
        best_theta = theta
    
    #print(f"Model: {model}, Fold Errors: {errors}")
    return np.mean(errors), best_theta


#ADDED BY CHELSEA - Training function for a new dataset
#(Uses lots of stuff from the original/default training instructions in the if-statement)
def train_all():
    Xnorm, ynorm, X_mean, X_std, y_mean, y_std = normalize(X, y)
    update_stats(X_mean, X_std, y_mean, y_std)
    
    # Train and save theta for Lasso
    lasso_error, theta_lasso = cross_validation_with_theta(
        Xnorm, ynorm, model="lasso", model_params={"alpha": 0.1}, y_mean=y_mean, y_std=y_std
    )
    
    # Train and save theta for Ridge
    ridge_error, theta_ridge = cross_validation_with_theta(
        Xnorm, ynorm, model="ridge", model_params={"alpha": 0.1}, y_mean=y_mean, y_std=y_std
    )
    
    # Train and save theta for Polynomial Regression
    poly_error, theta_poly = cross_validation_with_theta(
        Xnorm, ynorm, model="polynomial", model_params={"degree": 2}, y_mean=y_mean, y_std=y_std
    )
    update_modelresults(lasso_error, theta_lasso, ridge_error, theta_ridge, poly_error, theta_poly)
    global has_data
    has_data = True


def predict(X_input, model, model_params, theta, X_mean, X_std, y_mean, y_std):
    """
    Predict the exam score given input features.
    
    Parameters:
    - X_input: Raw input features (not normalized)
    - model: The model type ("lasso", "ridge", or "polynomial")
    - model_params: The parameters used to train the model
    - theta: The trained parameters
    - X_mean, X_std: The mean and std of the features used during training
    - y_mean, y_std: The mean and std of the target variable during training
    
    Returns:
    - Predicted exam score (denormalized)
    """
    # Normalize the input features
    X_input_normalized = (X_input - X_mean) / X_std

    # For polynomial regression, expand features
    if model == "polynomial":
        degree = model_params['degree']
        X_input_normalized = polynomial_features(X_input_normalized, degree)

    # Compute predictions
    predictions_normalized = X_input_normalized @ theta
    
    # Denormalize the predictions
    predictions = predictions_normalized * y_std + y_mean
    return predictions

#Predict all 3 models in one go, generate results, and return to the UI.
def predict_all(X_input):
    predicted_lasso = predict(
        X_input,
        model="lasso",
        model_params={"alpha": 0.1},
        theta=theta_lasso,
        X_mean=X_mean,
        X_std=X_std,
        y_mean=y_mean,
        y_std=y_std,
    )
    predicted_ridge = predict(
        X_input,
        model="ridge",
        model_params={"alpha": 0.1},
        theta=theta_ridge,
        X_mean=X_mean,
        X_std=X_std,
        y_mean=y_mean,
        y_std=y_std,
    )
    predicted_poly = predict(
        X_input,
        model="polynomial",
        model_params={"degree": 2},
        theta=theta_poly,
        X_mean=X_mean,
        X_std=X_std,
        y_mean=y_mean,
        y_std=y_std,
    )
    return predicted_lasso, predicted_ridge, predicted_poly
    
