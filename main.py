import cv2
import mediapipe as mp
import numpy as np
import os

# test
cap = cv2.VideoCapture(0)

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands_detector = mp.solutions.hands

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=30,
    refine_landmarks=True,
    min_detection_confidence=0.1,
    min_tracking_confidence=0.1
)

hands_detector = mp_hands_detector.Hands(
    static_image_mode=False,
    max_num_hands=60,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)



while cap.isOpened():
    ret, frame = cap.read()

    if cv2.waitKey(1) & 0xFF == ord('q') or not ret:
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(image_rgb)
    if results.multi_hand_landmarks is not None:
        for h in results.multi_hand_landmarks:
            mid_mcp = h.landmark[mp_hands_detector.HandLandmark.MIDDLE_FINGER_MCP]
            mid_tip = h.landmark[mp_hands_detector.HandLandmark.MIDDLE_FINGER_TIP]
            wrist = h.landmark[mp_hands_detector.HandLandmark.WRIST]



            if mid_tip.y < mid_mcp.y:
                os.startfile('alarm.mp3')
                mp.solutions.drawing_utils.draw_landmarks(frame, h, landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(circle_radius=5, color = (0,0, 255)))
            else:
                mp.solutions.drawing_utils.draw_landmarks(frame, h, landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(circle_radius=5, color=(0, 255, 0)))

    results = face_mesh.process(image_rgb)
    if results.multi_face_landmarks is not None:
        for face_landmarks in results.multi_face_landmarks:
            left_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
            right_eye_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

            left_pupil = [469, 470, 471, 472]
            right_pupil = [474, 475, 476, 477]

            image_rows, image_cols, _ = frame.shape

            left_eye_points = [(int(face_landmarks.landmark[idx].x * image_cols),
                                int(face_landmarks.landmark[idx].y * image_rows)) for idx in left_eye_indices]

            right_eye_points = [(int(face_landmarks.landmark[idx].x * image_cols),
                                 int(face_landmarks.landmark[idx].y * image_rows)) for idx in right_eye_indices]
            left_pupil = [(int(face_landmarks.landmark[idx].x * image_cols),
                                int(face_landmarks.landmark[idx].y * image_rows)) for idx in left_pupil]

            right_pupil = [(int(face_landmarks.landmark[idx].x * image_cols),
                           int(face_landmarks.landmark[idx].y * image_rows)) for idx in right_pupil]

            if face_landmarks.landmark[159].y + 0.001 >= face_landmarks.landmark[468].y:
                cv2.polylines(frame, [np.array(left_eye_points, dtype=np.int32)], isClosed=True, color=(0, 0, 255), thickness=2)
            else:
                cv2.polylines(frame, [np.array(left_eye_points, dtype=np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)

            if face_landmarks.landmark[386].y + 0.001 >= face_landmarks.landmark[473].y:
                cv2.polylines(frame, [np.array(right_eye_points, dtype=np.int32)], isClosed=True, color=(0, 0, 255), thickness=2)
            else:
                cv2.polylines(frame, [np.array(right_eye_points, dtype=np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)

            cv2.polylines(frame, [np.array(left_pupil, dtype=np.int32)], isClosed=True, color=(0, 255, 0),
                          thickness=2)
            cv2.polylines(frame, [np.array(right_pupil, dtype=np.int32)], isClosed=True, color=(0, 255, 0),
                          thickness=2)
    frame = np.fliplr(frame)
    cv2.imshow('SchoolHelper alpha beta version', frame)


cap.release()
cv2.destroyAllWindows()
