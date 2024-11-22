import cv2
import mediapipe as mp
import numpy as np
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2


# Model Configurations
model_path = "./model/pose_landmarker_lite.task"
num_poses = 1
min_pose_detection_confidence = 0.7
min_pose_presence_confidence = 0.7
min_tracking_confidence = 0.7
# Capture Configuration
device_id = 0
fps = 10
# Window Configuration
window_name = "Pose Landmarker"
width = 640
height = 480
# Global Variables
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode


def main():
    # Create an PoseLandmarker object.
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE,
        num_poses=num_poses,
        min_pose_detection_confidence=min_pose_detection_confidence,
        min_pose_presence_confidence=min_pose_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
        output_segmentation_masks=True
    )

    with PoseLandmarker.create_from_options(options) as landmarker:
        # initiate VideoCapture(#)
        cap = cv2.VideoCapture(device_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        while cap.isOpened():
            success, frame = cap.read()
            if success:
                # Convert the frame received from OpenCV to a MediaPipe’s Image object.
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                timeframe_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
                # Detect pose landmarks from the input image.
                landmarker_result = landmarker.detect(mp_image)
                # Place landmarks on each frame
                annotated_image, segmentation_masks = draw_landmarks_on_image(mp_image.numpy_view(), landmarker_result)
                cv2.imshow("MediaPipe Pose", cv2.flip(segmentation_masks, 1))
                if cv2.waitKey(1) == ord("q"):
                    break
            else:
                exit("Ignoring empty camera frame.")
        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()


def draw_landmarks_on_image(rgb_image, detection_result):
    segmentation_mask = detection_result.segmentation_masks[0].numpy_view()
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
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style(),
        )
    return annotated_image, segmentation_mask


if __name__ == "__main__":
    main()