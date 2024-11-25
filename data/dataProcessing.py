# This program was used in BodyM_dataset which was removed in this repository
# The result was stored in "./training_data"

# STEP 1: Import dependencies
import pandas as pd
import shutil

# STEP 2: Access traning CSVs
metadata = pd.read_csv("bodyM_dataset/train/hwg_metadata.csv")
measurements = pd.read_csv("bodyM_dataset/train/measurements.csv")
subject_to_photo = pd.read_csv("bodyM_dataset/train/subject_to_photo_map.csv")

# STEP 3: Merge CSVs into one
merged = pd.merge(subject_to_photo, metadata, on="subject_id", how="right")
merged = pd.merge(merged, measurements, on="subject_id", how="left")

# STEP 4: Remove redundancies on the CSV
merged_cleaned = merged.drop_duplicates(subset='subject_id')

# STEP 5: Remove unwanted data on the CSV
merged_cleaned = merged_cleaned.drop(columns=['subject_id'])
merged_cleaned = merged_cleaned.drop(columns=['chest'])

# STEP 6: Remove PNG duplicates on bodyM_dataset/train/mask and store in a separate folder
for row in merged_cleaned.itertuples():
    shutil.copy(f"bodyM_dataset/train/mask/{row.photo_id}.png", "training_data")

# STEP 7: Output merged variable into a CSV
merged_cleaned.to_csv("metadata.csv", index=False)