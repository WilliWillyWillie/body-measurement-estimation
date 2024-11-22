import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode


# Create a pose landmarker instance with the live stream mode:
def print_result(
    result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int
):
    cv2.imshow("landmarker", output_image.numpy_view())
    if cv2.waitKey(1) == ord("q"):
        exit()


options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="./model/pose_landmarker_lite.task"),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result,
    output_segmentation_masks=True,
)

with PoseLandmarker.create_from_options(options) as landmarker:
    # Use OpenCV’s VideoCapture to start capturing from the webcam.
    cap = cv2.VideoCapture(0)
    # Create a loop to read the latest frame from the camera using VideoCapture#read()
    while cap.isOpened():
        success, frame = cap.read()
        if success:
            # Convert the frame received from OpenCV to a MediaPipe’s Image object.
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            # Get VideoCapture's timeframe in ms
            timeframe_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            landmarker_result = landmarker.detect_async(mp_image, timeframe_ms)
        else:
            exit("Ignoring empty camera frame.")
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
