import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Open a text file to save the output
with open('svr_bidirectional_model_output.txt', 'w') as f:
    for part in ['hip']:
        # Load the two CSV files
        estimated_2D = pd.read_csv("data/MLData/estimated_2D_measurements copy.csv")  # File 1
        measurements_3D = pd.read_csv("data/MLData/metadata.csv")  # File 2

        # Initialize lists for matched x and y values
        x_values = []
        y_values = []

        # Match rows using photo_id
        for index, row in estimated_2D.iterrows():
            photo_id = row['photo_id']

            # Find the matching row in the second DataFrame
            matching_row = measurements_3D[measurements_3D['photo_id'] == photo_id]
            
            if not matching_row.empty:
                # Extract x and y values
                x_values.append(row[f'est_{part}_width'])  # 2D measurement
                y_values.append(matching_row[f'{part}'].values[0])  # Estimated 3D measurement

        # Convert lists to arrays
        x_values = np.array(x_values).reshape(-1, 1)
        y_values = np.array(y_values).reshape(-1, 1)

        # Split the data into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(x_values, y_values, test_size=0.35, random_state=42)

        # Feature Scaling (Standardization)
        scaler_x = StandardScaler()
        scaler_y = StandardScaler()

        # Scale the x and y values
        x_train_scaled = scaler_x.fit_transform(x_train)
        y_train_scaled = scaler_y.fit_transform(y_train)
        x_test_scaled = scaler_x.transform(x_test)
        y_test_scaled = scaler_y.transform(y_test)

        # Define hyperparameter grid for SVR
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 
            'gamma': [0.0001, 0.001, 0.01, 0.1, 1, 10], 
            'epsilon': [0.01, 0.1, 0.2, 0.3, 0.4, 0.5] 
        }

        # Train Forward Model: Predict y (3D) from x (2D)
        f.write(f"\nTraining forward model for {part}...\n")
        forward_svr = SVR(kernel='rbf')
        forward_grid_search = GridSearchCV(forward_svr, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=1)
        forward_grid_search.fit(x_train_scaled, y_train_scaled.ravel())

        best_forward_params = forward_grid_search.best_params_
        forward_model = forward_grid_search.best_estimator_

        f.write(f"Best Forward Model Hyperparameters for {part}: {best_forward_params}\n")

        # Train Reverse Model: Predict x (2D) from y (3D)
        f.write(f"\nTraining reverse model for {part}...\n")
        reverse_svr = SVR(kernel='rbf')
        reverse_grid_search = GridSearchCV(reverse_svr, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=1)
        reverse_grid_search.fit(y_train_scaled, x_train_scaled.ravel())

        best_reverse_params = reverse_grid_search.best_params_
        reverse_model = reverse_grid_search.best_estimator_

        f.write(f"Best Reverse Model Hyperparameters for {part}: {best_reverse_params}\n")

        # Evaluate Forward Model
        y_test_pred_scaled = forward_model.predict(x_test_scaled)
        y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled.reshape(-1, 1))
        y_test_actual = scaler_y.inverse_transform(y_test_scaled)

        forward_mse = mean_squared_error(y_test_actual, y_test_pred)
        forward_r2 = r2_score(y_test_actual, y_test_pred)
        f.write(f"Forward Model - MSE: {forward_mse:.4f}, R²: {forward_r2:.4f}\n")

        # Evaluate Reverse Model
        x_test_pred_scaled = reverse_model.predict(y_test_scaled)
        x_test_pred = scaler_x.inverse_transform(x_test_pred_scaled.reshape(-1, 1))
        x_test_actual = scaler_x.inverse_transform(x_test_scaled)

        reverse_mse = mean_squared_error(x_test_actual, x_test_pred)
        reverse_r2 = r2_score(x_test_actual, x_test_pred)
        f.write(f"Reverse Model - MSE: {reverse_mse:.4f}, R²: {reverse_r2:.4f}\n")

        # Save both models and scalers
        joblib.dump({
            'forward_model': forward_model, 
            'reverse_model': reverse_model, 
            'scaler_x': scaler_x, 
            'scaler_y': scaler_y
        }, f"svr_bidirectional_{part}_models.pkl")

        f.write(f"Models and scalers for {part} saved.\n")
