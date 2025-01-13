# Step 1: Import dependencies.
import mediapipe as mp
import numpy as np
import cv2
import math
import joblib

# Step 2: Initialize configuration values for the Mediapipe Pose Landmarker.
model_path = "./model/pose_landmarker_lite.task"
num_poses = 1
min_pose_detection_confidence = 0.7
min_pose_presence_confidence = 0.7
min_tracking_confidence = 0.7
img_path = "./william.png"

# Optional: Initialize values for Mediapipe Pose Landmarker options.
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode


def main():
    # STEP 3: Create a PoseLandmarker object.
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE,
        num_poses=num_poses,
        min_pose_detection_confidence=min_pose_detection_confidence,
        min_pose_presence_confidence=min_pose_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
        output_segmentation_masks=True,
    )
    with PoseLandmarker.create_from_options(options) as landmarker:
        # STEP 4: Load the input image and get image info.
        image = cv2.imread(img_path)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        img_height, img_width, _ = image.shape

        # STEP 5: Detect pose landmarks from the input image.
        detection_result = landmarker.detect(mp_image)

        # STEP 6: Convert segmentation mask into a cv2 readable format
        segmentation_mask = detection_result.segmentation_masks[0].numpy_view()
        visualized_mask = (
            np.repeat(segmentation_mask[:, :, np.newaxis], 3, axis=2) * 255
        )

        # STEP 7: Normalize poselandmarks' x and y coords from detection result
        normalized_coords = normalize_poselandmarks(
            detection_result.pose_landmarks, img_height, img_width
        )

        # Step 8: Transform coordinates into 2d measurements in cm
        measurements = coords_to_measurements(normalized_coords, visualized_mask)

        print(measurements)

        for part in measurements:
            if part in ['ankle-width', 'bicep-width', 'calf-width', 'forearm-width', 'hip-width', 'thigh-width', 'waist-width', 'wrist-width']:
                measurements[part] = predict_3d_measurement(part.split('-')[0], measurements[part])
                part = f'{part.split('-')[0]}-girth'

        print(measurements)



def normalize_poselandmarks(pose_landmarks_list, height, width):
    coords = []
    for pose_landmark in pose_landmarks_list[0]:
        coords.append([round(pose_landmark.x * width), round(pose_landmark.y * height)])
    return coords


def coords_to_measurements(coords, mask):
    # Step 1: Create a dictionary for 13 body parts with correspoding coords
    actual_height = int(input("Enter Heigth in cm: "))

    measurement_points = {
        "ankle-width": coords[27],
        "arm-length": [coords[11], coords[15]],
        "bicep-width": [coords[11], coords[13]],
        "calf-width": [coords[25], coords[27]],
        "forearm-width": [coords[13], coords[15]],
        "height": [coords[0], coords[27]],
        "hip-width": [coords[23], coords[24]],
        "leg-length": [coords[23], coords[27]],
        "shoulder-breadth": [coords[11], coords[12]],
        "shoulder-to-crotch-length": [coords[11], coords[23], coords[24]],
        "thigh-width": [coords[23], coords[25]],
        "waist-width": [coords[11], coords[24]],
        "wrist-width": [coords[11], coords[15]],
    }

    i = 8


    upper_edge = searchVerticalEdge(measurement_points['height'][0], -i, mask)
    lower_edge = searchVerticalEdge(measurement_points['height'][1], i, mask)
    measurement_points['height'] = actual_height
    cm_per_px = actual_height / (lower_edge[1] - upper_edge[1])

    for part in measurement_points:
        if part == "ankle-width":
            left_edge = searchHorizontalEdge(measurement_points[part], -i, mask)
            right_edge = searchHorizontalEdge(measurement_points[part], i, mask)
            measurement_points[part] = (right_edge[0] - left_edge[0])

        elif part in ["arm-length", "leg-length"]:
            m = getSlope(measurement_points[part][0], measurement_points[part][1])
            upper_edge = measurement_points[part][0]
            lower_edge = searchDiagonalEdge(
                measurement_points[part][0], measurement_points[part][1], m, i, mask
            )
            measurement_points[part] = findPointDistance(upper_edge, lower_edge)

        elif part in [
            "bicep-width",
            "calf-width",
            "forearm-width",
            "thigh-width",
            "wrist-width",
        ]:
            if part == "bicep-width":
                traversing_point = getLowerPartialPoint(
                    measurement_points[part][0], measurement_points[part][1]
                )
            elif part == "calf-width":
                traversing_point = getUpperPartialPoint(
                    measurement_points[part][0], measurement_points[part][1]
                )
            elif part == "wrist-width":
                traversing_point = measurement_points[part][1]
            else:
                traversing_point = getDiagonalMidPoint(
                    measurement_points[part][0], measurement_points[part][1]
                )
                if part == "forearm-width":
                    traversing_point = getDiagonalMidPoint(
                        measurement_points[part][0], traversing_point
                    )
            mpen = getPerpendicularSlope(
                measurement_points[part][0], measurement_points[part][1]
            )
            upper_edge = searchDiagonalEdge(
                traversing_point, traversing_point, mpen, i, mask
            )
            lower_edge = searchDiagonalEdge(
                traversing_point, traversing_point, mpen, -i, mask
            )
            measurement_points[part] = findPointDistance(upper_edge, lower_edge)

        elif part in ["hip-width", "shoulder-breadth"]:
            left_edge = searchHorizontalEdge(measurement_points[part][1], -i, mask)
            right_edge = searchHorizontalEdge(measurement_points[part][0], i, mask)
            measurement_points[part] = (right_edge[0] - left_edge[0])

        elif part == "shoulder-to-crotch-length":
            # Lower edge gets the bottom most part of the crotch area
            lower_edge = searchVerticalEdge(
                getDiagonalMidPoint(
                    measurement_points[part][1], measurement_points[part][2]
                ), i, mask
            )
            m = getSlope(measurement_points[part][0], lower_edge)
            upper_edge = searchDiagonalEdge(
                measurement_points[part][0], lower_edge, m, i, mask
            )
            measurement_points[part] = findPointDistance(upper_edge, lower_edge)

        elif part == "waist-width":
            traversing_point = getLowerPartialPoint(
                measurement_points[part][0], measurement_points[part][1]
            )
            left_edge = searchHorizontalEdge(traversing_point, -i, mask)
            right_edge = searchHorizontalEdge(traversing_point, i, mask)
            measurement_points[part] = (right_edge[0] - left_edge[0])
    
    # Convert pixel to cm measurement and account for the 10% margin of error
    for part in measurement_points:
        if part in ['forearm-width', 'hip-width', 'thigh-width']:
            error_margin = .85
        elif part in ['waist-width']:
            error_margin = .9
        else:
            error_margin = 1
        if part != "height":
            measurement_points[part] = measurement_points[part] * cm_per_px * error_margin

    return measurement_points

def predict_3d_measurement(part, est_2d_value):
    if part in ['bicep', 'thigh', 'wrist', 'forearm']:
        """Predicts a 3D measurement given a 2D input."""
        model_data = joblib.load(f'./model/SVR/svr_{part}_model.pkl')
        model, scaler_x, scaler_y = model_data['model'], model_data['scaler_x'], model_data['scaler_y']

        # Prepare input
        new_x_scaled = scaler_x.transform([[est_2d_value]])
        pred_scaled = model.predict(new_x_scaled)
        predicted_3d = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))

        return predicted_3d[0][0]
    elif part in ['ankle', 'calf', 'hip', 'waist']:
        """Predicts a 3D measurement given a 2D input using a Polynomial Regression model."""
        # Load the polynomial model and associated scalers
        model_data = joblib.load(f'./model/SVR/poly_{part}_model.pkl')
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

def searchVerticalEdge(coords, step, mask):
    x, y = coords

    while True:
        while str(mask[int(y), int(x)]) == "[255. 255. 255.]":
            y += step
        y -= step
        step = step / 2
        if step % 1 != 0:
            return x, y


def searchHorizontalEdge(coords, step, mask):
    x, y = coords

    while step % 1 == 0:
        while str(mask[int(y), int(x)]) == "[255. 255. 255.]":
            x += step
        x -= step
        step = step / 2
    return x, y


def searchDiagonalEdge(coords1, coords2, m, step, mask):
    x1, y1 = coords1
    x2, y2 = coords2  # This would be the point used for searching the edge

    while step % 0.0625 == 0:
        while str(mask[round(y2), round(x2)]) == "[255. 255. 255.]":
            x2 += step
            y2_prev = y2
            y2 = m * (x2 - x1) + y1
        x2 -= step
        y2 = y2_prev
        step = step / 2
    return x2, y2


def getDiagonalMidPoint(coords1, coords2):
    x1, y1 = coords1
    x2, y2 = coords2

    return [((x1 + x2) / 2), ((y1 + y2) / 2)]


def getUpperPartialPoint(coords1, coords2):
    x1, y1 = coords1
    x2, y2 = coords2

    return [((2 * x1 + x2) / 3), ((2 * y1 + y2) / 3)]


def getLowerPartialPoint(coords1, coords2):
    x1, y1 = coords1
    x2, y2 = coords2

    return [((x1 + 2 * x2) / 3), ((y1 + 2 * y2) / 3)]


def getSlope(coords1, coords2):
    x1, y1 = coords1
    x2, y2 = coords2

    # Get slope between coords1 and coords2
    return (y2 - y1) / (x2 - x1)


def getPerpendicularSlope(coords1, coords2):
    x1, y1 = coords1
    x2, y2 = coords2

    # Get slope between coords1 and coords2
    return -1 / ((y2 - y1) / (x2 - x1))


def findPointDistance(coords1, coords2):
    x1, y1 = coords1
    x2, y2 = coords2

    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


if __name__ == "__main__":
    main()
