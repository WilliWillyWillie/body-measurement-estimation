# STEP 1: Import dependencies
import pandas as pd
import shutil

# STEP 2: Access traning CSVs
px_measured = pd.read_csv("data/MLData/2d_measurements.csv")
cm_measurements = pd.read_csv("data/MLData/metadata.csv")

# STEP 3: Convert 2D measured px height measurements into cm measurements
results = []
# Access the height column row by row
for index, row in px_measured.iterrows():
    px_height = row['height']
    photo_id = row['photo_id']

    # Find the matching row in the second DataFrame
    matching_row = cm_measurements[cm_measurements['photo_id'] == photo_id]
    
    actual_cm_height = matching_row['height'].values[0]

    # Convert each measurement from px to cm
    pxcm = actual_cm_height/px_height
    est_ankle_width = row['ankle_width'] * pxcm
    est_bicep_width = row['bicep_width'] * pxcm
    est_calf_width = row['calf_width'] * pxcm
    est_forearm_width = row['forearm_width'] * pxcm
    est_hip_width = row['hip_width'] * pxcm
    est_thigh_width = row['thigh_width'] * pxcm
    est_waist_width = row['waist_width'] * pxcm
    est_wrist_width = row['wrist_width'] * pxcm

    results.append({
        "photo_id": photo_id,
        "est_ankle_width": est_ankle_width,
        "est_bicep_width": est_bicep_width,
        "est_calf_width": est_calf_width,
        "est_forearm_width": est_forearm_width,
        "est_hip_width": est_hip_width,
        "est_thigh_width": est_thigh_width,
        "est_waist_width": est_waist_width,
        "est_wrist_width": est_wrist_width,
    })

# STEP 5: Create a DataFrame from the results
results_df = pd.DataFrame(results)

# STEP 6: Save the results to a new CSV
output_file = "data/MLData/estimated_2D_measurements.csv"
results_df.to_csv(output_file, index=False)

print(f"Estimated measurements saved to {output_file}")
