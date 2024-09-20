import cv2
import mediapipe as mp
from robot_dqn import TrackingDQN
# @@@@@@@@@@ 여기를 클래스로 만들고, app_test에서 import하자 20240918
# MediaPipe 포즈 모듈 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
tr = TrackingDQN()
# 웹캠 열기
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    # 이미지를 RGB로 변환
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # 포즈 추정
    results = pose.process(image)
    # 이미지 다시 BGR로 변환
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # 신체 랜드마크 그리기
    if results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        nose_landmark = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
        h, w, _ = image.shape
        if nose_landmark.x < 0.3:
            action = tr.select_action(-0.1)
        elif nose_landmark.x > 0.7:
            action = tr.select_action(0.1)
        else:
            action = 0
        if action == 1:
            position = "Right"
        elif action == 2:
            position = "Left"
        else:
            position = "None"
        cv2.putText(image, f'Position: {position}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Pose Detection', image)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
