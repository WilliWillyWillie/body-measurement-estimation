# Import necessary libraries

import joblib

def predict_3d_measurement(est_2d_value):
    """Predicts a 3D measurement given a 2D input using a Polynomial Regression model."""
    # Load the polynomial model and associated scalers
    model_data = joblib.load('poly_forearm_model.pkl')
    model, scaler_x, scaler_y, poly_features = (
        model_data['model'],
        model_data['scaler_x'],
        model_data['scaler_y'],
        model_data['poly_features'],
    )

    # Transform and scale the input
    new_x_poly = poly_features.transform([[est_2d_value]])
    new_x_scaled = scaler_x.transform(new_x_poly)

    # Predict and inverse-transform the output
    pred_scaled = model.predict(new_x_scaled)
    predicted_3d = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))

    return predicted_3d[0][0]

# Predict and print a new value
for i in range(1, 20):
    new_est_2d_value = i
    predicted_3d_value = predict_3d_measurement(new_est_2d_value)
    if predict_3d_measurement == 90:
        break
    print(f"Predicted 3D measurement for 2D input {new_est_2d_value}: {predicted_3d_value:.2f}")

