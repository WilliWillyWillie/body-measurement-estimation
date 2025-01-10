import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# Load the two CSV files
estimated_2D = pd.read_csv("data/MLData/estimated_2D_measurements.csv")  # File 1
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
        x_values.append(row['est_wrist_width'])  # 2D measurement
        y_values.append(matching_row['wrist'].values[0])  # Estimated 3D measurement

# Convert lists to DataFrame or array
x_values = pd.DataFrame(x_values)
y_values = pd.DataFrame(y_values)

# Feature Scaling (Standardization) - SVR is sensitive to feature scaling
scaler_x = StandardScaler()
scaler_y = StandardScaler()

# Scale the x and y values
x_scaled = scaler_x.fit_transform(x_values)
y_scaled = scaler_y.fit_transform(y_values)

# Train the SVR model
svr = SVR(kernel='rbf')  # Using Radial Basis Function (RBF) kernel
svr.fit(x_scaled, y_scaled.ravel())  # Flatten y_values to match the model input format

# Make predictions
y_pred_scaled = svr.predict(x_scaled)

# Inverse transform the predictions to original scale
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))

# Plot the actual vs predicted values
plt.figure(figsize=(8, 6))
plt.scatter(x_values, y_values, color='blue', label='Estimated  3D vs 2D Measurements')
plt.scatter(x_values, y_pred, color='red', label='SVR Predicted Values')
plt.xlabel('2D Measurement (wrist width)')
plt.ylabel('Estimated 3D Measurement (wrist girth)')
plt.title('SVR Prediction: 2D Measurements vs. Estimated  3D Measurements')
plt.legend()
plt.grid(True)
plt.show()

def predict_3d_measurement(est_2d_value):
  """
  This function takes an estimated 2D measurement (ankle width) and predicts the corresponding 3D measurement using the trained SVR model.

  Args:
      est_2d_value (float): The estimated 2D measurement (ankle width) for which you want to predict the 3D measurement.

  Returns:
      float: The predicted 3D measurement (ankle width) based on the SVR model.
  """

  # Prepare the input data (scale it if necessary)
  new_x = pd.DataFrame([[est_2d_value]])
  new_x_scaled = scaler_x.transform(new_x)

  # Make the prediction using the trained SVR model
  predicted_scaled = svr.predict(new_x_scaled)

  # Inverse transform the prediction to get the original scale
  predicted_3d = scaler_y.inverse_transform(predicted_scaled.reshape(-1, 1))

  return predicted_3d[0][0]  # Extract the predicted value

# Example usage: predict 3D measurement for a new estimated 2D value
new_est_2d_value = 32.72033598364853  # Replace with your desired estimated 2D measurement (ankle width)
predicted_3d_value = predict_3d_measurement(new_est_2d_value)

print("Estimated 2D measurement:", new_est_2d_value)
print("Predicted 3D measurement:", predicted_3d_value)