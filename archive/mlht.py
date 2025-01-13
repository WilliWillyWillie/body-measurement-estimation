import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Open a text file to save the output
with open('svr_model_output.txt', 'w') as f:
    for part in ['ankle', 'bicep', 'calf', 'forearm', 'hip', 'thigh', 'waist', 'wrist']:
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

        # Convert lists to DataFrame or array
        x_values = np.array(x_values).reshape(-1, 1)
        y_values = np.array(y_values).reshape(-1, 1)

        # Split the data into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(x_values, y_values, test_size=0.15, random_state=42)

        # Feature Scaling (Standardization) - SVR is sensitive to feature scaling
        scaler_x = StandardScaler()
        scaler_y = StandardScaler()

        # Scale the x and y values
        x_train_scaled = scaler_x.fit_transform(x_train)
        y_train_scaled = scaler_y.fit_transform(y_train)
        x_test_scaled = scaler_x.transform(x_test)
        y_test_scaled = scaler_y.transform(y_test)

        # Define hyperparameter grid for SVR (more precise tuning)
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 
            'gamma': [0.0001, 0.001, 0.01, 0.1, 1, 10], 
            'epsilon': [0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5] 
        }

        # Initialize SVR model
        svr_model = SVR(kernel='rbf')

        # Use GridSearchCV for hyperparameter tuning
        grid_search = GridSearchCV(svr_model, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=1)
        grid_search.fit(x_train_scaled, y_train_scaled.ravel())

        # Best hyperparameters found by GridSearchCV
        best_params = grid_search.best_params_
        
        # Save output to file
        f.write(f"\nBest Hyperparameters for {part} found by GridSearchCV: {best_params}\n")

        # Train the SVR model with the best hyperparameters
        svr_best_model = grid_search.best_estimator_

        # Evaluate the model on the test set
        y_test_pred_scaled = svr_best_model.predict(x_test_scaled)
        y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled.reshape(-1, 1))
        y_test_actual = scaler_y.inverse_transform(y_test_scaled)

        # Print actual vs predicted values
        f.write(f"\nActual vs Predicted Values for {part} (for the test set):\n")
        for i in range(len(y_test_actual)):
            f.write(f"Actual: {y_test_actual[i][0]:.6f}, Predicted: {y_test_pred[i][0]:.6f}\n")

        # Evaluation metrics
        test_mse = mean_squared_error(y_test_actual, y_test_pred)
        test_mae = mean_absolute_error(y_test_actual, y_test_pred)
        test_r2 = r2_score(y_test_actual, y_test_pred)

        # Compute Mean Absolute Percentage Error (MAPE) for average closeness
        mape = np.mean(np.abs((y_test_actual - y_test_pred) / y_test_actual)) * 100  # In percentage

        # Compute Root Mean Squared Error (RMSE)
        rmse = np.sqrt(test_mse)

        f.write(f"\n{part} - SVR Test Set Evaluation Metrics:\n")
        f.write(f"Mean Squared Error (MSE): {test_mse:.4f}\n")
        f.write(f"Mean Absolute Error (MAE): {test_mae:.4f}\n")
        f.write(f"R-squared (R²): {test_r2:.4f}\n")
        f.write(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%\n")
        f.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}\n")

        # Save the model and scalers
        joblib.dump({'model': svr_best_model, 'scaler_x': scaler_x, 'scaler_y': scaler_y}, f"svr_{part}_model.pkl")

        f.write(f"\nModel, scalers, and evaluation metrics for {part} saved.\n")
