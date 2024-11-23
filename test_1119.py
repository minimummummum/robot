import pybullet as p
import time
import pybullet_data
import cv2
import mediapipe as mp
import numpy as np

# MediaPipe 초기화
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# PyBullet 연결
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -10)
planeId = p.loadURDF("plane.urdf")

# 로봇 로드
robotStartPos = [0, 0, 0]
robotStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
robotId = p.loadURDF("C:/Users/msm16/Desktop/robot/robot11/robot11.urdf", robotStartPos, robotStartOrientation)

# 카메라 시점 설정 (로봇 뒤쪽 정면으로)
camera_distance = 0.5  # 카메라와 로봇 간 거리
camera_yaw = 180     # Y축 회전 (180은 로봇 뒤쪽)
camera_pitch = -30   # X축 회전 (negative 값으로 아래에서 보기)
camera_target = [0, 0, 0]  # 로봇의 위치

p.resetDebugVisualizerCamera(camera_distance, camera_yaw, camera_pitch, camera_target)

left_shoulder_joint_index = 4
left_elbow_joint_index = 7
right_shoulder_joint_index = 49
right_elbow_joint_index = 52
"""
# 로봇 관절 인덱스
shoulder_right = 4 # -10~190
shoulder_left = 49 # -190~10
hand_right = 7 # -65~55
hand_left = 52 # -55~65
leg_right = 15 # -20~20
leg_left = 31 # -20~20
foot_right = 20 # -90~90
foot_left = 36 # -90~90
max_velocity = 8 # 모터 속도
"""
# 카메라 설정
cap = cv2.VideoCapture(0)
arm_length = []
start = True
with mp_pose.Pose(static_image_mode=False, model_complexity=2, enable_segmentation=False, min_detection_confidence=0.5) as pose:
    while True:
        # 카메라에서 이미지 읽기
        ret, frame = cap.read()
        if not ret:
            break
        
        # 이미지 전처리
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 이미지 반전
        image = cv2.flip(image, 1)  # 1은 좌우 반전
        image.flags.writeable = False
        
        # 포즈 추정
        results = pose.process(image)
        
        # 이미지 다시 가시화
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # 포즈 데이터에서 좌표 추출
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

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


            # 로봇 조인트 각도 설정
            p.setJointMotorControl2(robotId, left_shoulder_joint_index, p.POSITION_CONTROL, left_shoulder_angle)
            p.setJointMotorControl2(robotId, left_elbow_joint_index, p.POSITION_CONTROL, left_elbow_angle)
            p.setJointMotorControl2(robotId, right_shoulder_joint_index, p.POSITION_CONTROL, right_shoulder_angle)
            p.setJointMotorControl2(robotId, right_elbow_joint_index, p.POSITION_CONTROL, right_elbow_angle)
            


        # 결과 이미지 출력
        cv2.imshow('Inverted Pose Detection', image)
        
        # 시뮬레이션 스텝
        p.stepSimulation()
        time.sleep(1. / 240.)

        # 종료 조건
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# 자원 해제
cap.release()
cv2.destroyAllWindows()
p.disconnect()
