import cv2
import mediapipe as mp
import numpy as np
from robot_dqn import TrackingDQN

class Robot_Action():
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.tr = TrackingDQN()
        self.arm_length = 0
        self.servo_state = [0, 0, 0, 0]

    def action(self, img, servo_state):
        try:
            self.servo_state = servo_state
            img_resized = cv2.resize(img, (320, 240))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            
            # MediaPipe 처리
            results = self.pose.process(img_rgb)
            img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                left_shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                                 landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y]
                left_elbow = [landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW].x,
                              landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW].y]
                right_shoulder = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                                  landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
                right_elbow = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW].x,
                               landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW].y]

                # 코 위치에 따른 동작 결정
                nose_landmark = landmarks[self.mp_pose.PoseLandmark.NOSE]
                # 화면 너무 끝이거나, 없을 때 -값 나와서 제한 걸어둠.
                if nose_landmark.x < 0.09 or nose_landmark.x > 0.91:
                    self.arm_length = 0
                    return "wait"
                if nose_landmark.x < 0.3:
                    action = self.tr.select_action(-0.1)
                elif nose_landmark.x > 0.7:
                    action = self.tr.select_action(0.1)
                else:
                    # 새로운 팔 길이 계산
                    new_arm_length = np.linalg.norm(np.array(left_shoulder) - np.array(left_elbow))
                    if new_arm_length > self.arm_length:
                        self.arm_length = new_arm_length
                    
                    # 팔꿈치 각도 계산
                    left_arm_x = left_elbow[0] - left_shoulder[0]
                    right_arm_x = right_elbow[0] - right_shoulder[0]

                    left_shoulder_angle = np.arctan2(left_elbow[1] - left_shoulder[1], left_elbow[0] - left_shoulder[0])
                    left_shoulder_angle = -left_shoulder_angle + np.radians(90)
                    left_elbow_angle = (left_arm_x / self.arm_length) * np.radians(180)
                    left_elbow_angle = -left_elbow_angle
                    left_elbow_angle += np.radians(30)

                    right_shoulder_angle = np.arctan2(right_shoulder[1] - right_elbow[1], right_shoulder[0] - right_elbow[0])
                    right_shoulder_angle = -right_shoulder_angle - np.radians(90)
                    right_elbow_angle = (right_arm_x / self.arm_length) * np.radians(180)
                    right_elbow_angle = -right_elbow_angle
                    right_elbow_angle -= np.radians(30)

                    self.servo_state[0] = round(np.degrees(left_shoulder_angle) / 10) * 10
                    self.servo_state[2] = -round(np.degrees(left_elbow_angle) / 10) * 10
                    self.servo_state[1] = -round(np.degrees(right_shoulder_angle) / 10) * 10
                    self.servo_state[3] = round(np.degrees(right_elbow_angle) / 10) * 10

                    return self.servo_state
                return action
            else:
                self.arm_length = 0
                return "wait"
        except Exception as e:
            pass
