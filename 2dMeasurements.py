# Step 1: Import dependencies.
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2
import math

# Step 2: Initialize configuration values for the Mediapipe Pose Landmarker.
model_path = "./model/pose_landmarker_lite.task"
num_poses = 1
min_pose_detection_confidence = 0.7
min_pose_presence_confidence = 0.7
min_tracking_confidence = 0.7

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
        image = cv2.imread("william.png")
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
        measurements_2d = coords_to_measurements(normalized_coords, visualized_mask)


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

    for part in measurement_points:
        i = 8

        if part == "height":
            upper_edge = searchVerticalEdge(measurement_points[part][0], -i, mask)
            lower_edge = searchVerticalEdge(measurement_points[part][1], i, mask)
            measurement_points[part] = actual_height
            print(f"Height distance: {lower_edge[1] - upper_edge[1]}")
            cm_per_px = actual_height / (lower_edge[1] - upper_edge[1])

        elif part == "ankle-width":
            left_edge = searchHorizontalEdge(measurement_points[part], -i, mask)
            right_edge = searchHorizontalEdge(measurement_points[part], i, mask)
            measurement_points[part] = right_edge[0] - left_edge[0]
            print(
                f"[Part: {part}\nLeft Edge: {left_edge}\nRight Edge: {right_edge}\nLength: {measurement_points[part]}]"
            )

        elif part in ["arm-length", "leg-length"]:
            m = getSlope(measurement_points[part][0], measurement_points[part][1])
            upper_edge = measurement_points[part][0]
            lower_edge = searchDiagonalEdge(
                measurement_points[part][0], measurement_points[part][1], m, i, mask
            )
            measurement_points[part] = findPointDistance(upper_edge, lower_edge)
            print(
                f"[Part: {part}\nUpper Edge: {upper_edge}\nLower Edge: {lower_edge}\nLength: {measurement_points[part]}]"
            )

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
            print(
                f"[Part: {part}\nUpper Edge: {upper_edge}\nLower Edge: {lower_edge}\nLength: {measurement_points[part]}]"
            )

        elif part in ["hip-width", "shoulder-breadth"]:
            left_edge = searchHorizontalEdge(measurement_points[part][1], -i, mask)
            right_edge = searchHorizontalEdge(measurement_points[part][0], i, mask)
            measurement_points[part] = right_edge[0] - left_edge[0]
            print(
                f"[Part: {part}\nLeft Edge: {left_edge}\nRight Edge: {right_edge}\nLength: {measurement_points[part]}]"
            )

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
            print(
                f"[Part: {part}\nUpper Edge: {upper_edge}\nLower Edge: {lower_edge}\nLength: {measurement_points[part]}]"
            )

        elif part == "waist-width":
            traversing_point = getLowerPartialPoint(
                measurement_points[part][0], measurement_points[part][1]
            )
            left_edge = searchHorizontalEdge(traversing_point, -i, mask)
            right_edge = searchHorizontalEdge(traversing_point, i, mask)
            measurement_points[part] = right_edge[0] - left_edge[0]
            print(
                f"[Part: {part}\nLeft Edge: {left_edge}\nRight Edge: {right_edge}\nLength: {measurement_points[part]}]"
            )
    
    # Convert pixel to cm measurement and account for the 10% margin of error
    error_margin = .1 + 1
    for part in measurement_points:
        if part != "height":
            measurement_points[part] = measurement_points[part] * cm_per_px * error_margin
            print(f"{part}: {measurement_points[part]}")


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
