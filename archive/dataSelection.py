import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2
import os

model_path = "./model/pose_landmarker_lite.task"
num_poses = 1
min_pose_detection_confidence = 0.7
min_pose_presence_confidence = 0.7
min_tracking_confidence = 0.7

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode


def main():
    # STEP 2: Create an PoseLandmarker object.
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
        # STEP 3: Load the input image.
        for root, dirs, files in os.walk("./files"):
            for file in dirs:
                image = cv2.imread(f"files/{file}/front_img.jpg")
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

                # STEP 4: Detect pose landmarks from the input image.
                detection_result = landmarker.detect(mp_image)

                # STEP 5: Process the detection result. In this case, visualize it.
                segmentation_mask = detection_result.segmentation_masks[0].numpy_view()
                visualized_mask = (
                    np.repeat(segmentation_mask[:, :, np.newaxis], 3, axis=2) * 255
                )

                annotated_mask = draw_landmarks_on_image(visualized_mask, detection_result)
                cv2.imwrite(f"segmentation_landmark/{file}.jpg", annotated_mask)


def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        # Draw the pose landmarks.
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend(
            [
                landmark_pb2.NormalizedLandmark(
                    x=landmark.x, y=landmark.y, z=landmark.z
                )
                for landmark in pose_landmarks
            ]
        )
        mp.solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            mp.solutions.pose.POSE_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_pose_landmarks_style(),
        )
    return annotated_image


if __name__ == "__main__":
    main()
