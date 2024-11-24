import cv2
import mediapipe as mp
import numpy as np
import math
# MediaPipe 포즈 모듈 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# 웹캠 열기
cap = cv2.VideoCapture(0)
arm_length = []
start = True

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    
    # 이미지를 RGB로 변환
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 포즈 추정
    results = pose.process(image)
    
    # 이미지 다시 BGR로 변환
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # 신체 랜드마크 그리기
    if results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        # 팔의 좌표 추출
        left_shoulder = [
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y
        ]
        left_elbow = [
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].x,
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y
        ]
        right_shoulder = [
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y
        ]
        right_elbow = [
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].x,
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y
        ]

        if start:
            if left_shoulder is not None and left_elbow is not None:
                arm_length = np.linalg.norm(np.array(left_shoulder) - np.array(left_elbow))
                start = False
            else:
                continue
        left_arm_length = left_elbow[0] - left_shoulder[0]
        right_arm_length = right_elbow[0] - right_shoulder[0]
        new_arm_length = math.sqrt((right_elbow[0] - right_shoulder[0]) ** 2 + (right_elbow[1] - right_shoulder[1]) ** 2)
        if new_arm_length > arm_length: # 저장된 팔 길이보다 현재 측정된 팔 길이가 더 길 경우 교체
            arm_length = new_arm_length

        # 각도 계산
        left_shoulder_angle = np.arctan2(left_elbow[1] - left_shoulder[1], left_elbow[0] - left_shoulder[0])
        left_shoulder_angle = -left_shoulder_angle + np.radians(90)
        left_elbow_angle = (left_arm_length / arm_length) * np.radians(180)
        left_elbow_angle = -left_elbow_angle
        left_elbow_angle += np.radians(30)
        
        right_shoulder_angle = np.arctan2(right_shoulder[1] - right_elbow[1], right_shoulder[0] - right_elbow[0])
        right_shoulder_angle = -right_shoulder_angle - np.radians(90)
        right_elbow_angle = (right_arm_length / arm_length) * np.radians(180)
        right_elbow_angle = -right_elbow_angle
        right_elbow_angle -= np.radians(30)

        left_shoulder_angle = round(np.degrees(left_shoulder_angle) / 10) * 10
        left_elbow_angle = -round(np.degrees(left_elbow_angle) / 10) * 10
        right_shoulder_angle = -round(np.degrees(right_shoulder_angle) / 10) * 10
        right_elbow_angle = round(np.degrees(right_elbow_angle) / 10) * 10

        
        # 화면에 각도 표시
        cv2.putText(image, f'LS: {right_shoulder_angle} LE: {right_elbow_angle}', 
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(image, f'RS: {left_shoulder_angle} RE: {left_elbow_angle}', 
                    (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 결과 화면 출력
    cv2.imshow('Pose Detection', image)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
