import cv2
import mediapipe as mp
import numpy as np
import time
import os

cap = cv2.VideoCapture(1)

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands_detector = mp.solutions.hands.Hands()

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=30,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)




while cap.isOpened():
    ret, frame = cap.read()

    if cv2.waitKey(1) & 0xFF == ord('q') or not ret:
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_hands_detector.process(image_rgb)
    if results.multi_hand_landmarks is not None:
        for h in results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(frame, h)

    results = face_mesh.process(image_rgb)
    if results.multi_face_landmarks is not None:
        for face_landmarks in results.multi_face_landmarks:
            left_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
            right_eye_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

            image_rows, image_cols, _ = frame.shape

            left_eye_points = [(int(face_landmarks.landmark[idx].x * image_cols),
                                int(face_landmarks.landmark[idx].y * image_rows)) for idx in left_eye_indices]

            right_eye_points = [(int(face_landmarks.landmark[idx].x * image_cols),
                                 int(face_landmarks.landmark[idx].y * image_rows)) for idx in right_eye_indices]

            cv2.polylines(frame, [np.array(left_eye_points, dtype=np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.polylines(frame, [np.array(right_eye_points, dtype=np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)

    cv2.imshow('Eyes Detection', frame)


cap.release()
cv2.destroyAllWindows()
