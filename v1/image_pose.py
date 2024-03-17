from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2


def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

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
    return annotated_image


def recognize_pose(image_path, result_path):
    base_options = python.BaseOptions(model_asset_path="pose_landmarker_lite.task")

    options = vision.PoseLandmarkerOptions(
        base_options=base_options, output_segmentation_masks=True
    )

    detector = vision.PoseLandmarker.create_from_options(options)

    image = mp.Image.create_from_file(image_path)

    detection_result = detector.detect(image)

    annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
    cv2.imwrite(f"{result_path}.jpg", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

    segmentation_mask = detection_result.segmentation_masks[0].numpy_view()
    visualized_mask = np.repeat(segmentation_mask[:, :, np.newaxis], 3, axis=2) * 255
    cv2.imwrite(f"{result_path}_mask.jpg", visualized_mask)


for image_number in range(1, 5):
    recognize_pose(
        image_path=f"images/before/image{image_number}.jpg",
        result_path=f"images/after/result{image_number}",
    )
