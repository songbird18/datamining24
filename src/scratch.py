import csv
import numpy as np
import pandas as pd

# Load CSV data into NumPy arrays
def load_data(file_path, selected_columns, target_column):
    data = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        headers = next(reader)  # Read the header row
        selected_indices = [headers.index(col) for col in selected_columns]
        target_index = headers.index(target_column)
        
        for row in reader:
            data.append([float(row[i]) if row[i].isdigit() else row[i] for i in selected_indices + [target_index]])
    data = np.array(data, dtype=object)
    X = np.array(data[:, :-1], dtype=float)  # Features
    y = np.array(data[:, -1], dtype=float)  # Target variable
    return X, y

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
    
    print(f"Model: {model}, Fold Errors: {errors}")
    return np.mean(errors), best_theta


if __name__ == "__main__":
    

    # File path and columns
    file_path = '/data/haoyang27/data_mining/StudentPerformanceFactors_Cleaned.csv'  # Replace with your file path
    selected_columns = ['Hours_Studied', 'Attendance', 'Family_Income']  # Include non-numerical columns
    target_column = 'Exam_Score'  # Replace with your target column

    # Load data into a pandas DataFrame
    data = pd.read_csv(file_path)

    # Perform label encoding for categorical columns
    categorical_columns = ['Family_Income']  # Adjust as needed
    for column in categorical_columns:
        label_map = {value: idx for idx, value in enumerate(sorted(data[column].unique()))}
        data[column] = data[column].map(label_map)
    
    # Select features and target
    X = data[selected_columns].values
    y = data[target_column].values

    # Normalize numerical columns (optional, categorical columns remain unaffected)
    X, y, X_mean, X_std, y_mean, y_std = normalize(X, y)
    
    # Train and save theta for Lasso
    lasso_error, theta_lasso = cross_validation_with_theta(
        X, y, model="lasso", model_params={"alpha": 0.1}, y_mean=y_mean, y_std=y_std
    )
    print("Lasso Regression Error:", lasso_error)
    
    # Train and save theta for Ridge
    ridge_error, theta_ridge = cross_validation_with_theta(
        X, y, model="ridge", model_params={"alpha": 0.1}, y_mean=y_mean, y_std=y_std
    )
    print("Ridge Regression Error:", ridge_error)
    
    # Train and save theta for Polynomial Regression
    poly_error, theta_poly = cross_validation_with_theta(
        X, y, model="polynomial", model_params={"degree": 2}, y_mean=y_mean, y_std=y_std
    )
    print("Polynomial Regression Error:", poly_error)


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


# Example input: Hours_Studied and Attendance for a new student
# X_input = np.array([[23, 84]])  # Replace with actual input features
X_input = np.array([
        [24, 98, 1],  # Student 1
        [15, 75, 1],  # Student 2
        [25, 90, 2],  # Student 3
    ]) 

# Predict using Lasso
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
print("Predicted Exam Score (Lasso):", predicted_lasso)

# Predict using Ridge
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
print("Predicted Exam Score (Ridge):", predicted_ridge)

# Predict using Polynomial
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
print("Predicted Exam Score (Polynomial):", predicted_poly)
