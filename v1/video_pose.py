import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

model_path = "pose_landmarker_full.task"

cap = cv2.VideoCapture(0)

with mp_pose.Pose(
    static_image_mode=False, model_complexity=1, min_detection_confidence=0.5
) as pose:
    while cap.isOpened():
        success, image = cap.read()

        if success:
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
                )

            cv2.imshow("MediaPipe Pose Detection", image)

            if cv2.waitKey(5) & 0xFF == 27:
                break

cap.release()
cv2.destroyAllWindows()
