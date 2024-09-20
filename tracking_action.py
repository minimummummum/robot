import cv2
import mediapipe as mp
from robot_dqn import TrackingDQN

class Tracking_Action():
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.tr = TrackingDQN()
    def tracking(self, img):
        try:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.pose.process(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            if results.pose_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(img, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                nose_landmark = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
                h, w, _ = img.shape
                if nose_landmark.x < 0.3:
                    action = self.tr.select_action(-0.1)
                elif nose_landmark.x > 0.7:
                    action = self.tr.select_action(0.1)
                else:
                    action = 0
                if action == 1:
                    position = "Right"
                elif action == 2:
                    position = "Left"
                else:
                    position = "None"
                cv2.putText(img, f'Position: {position}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Pose Detection', img)
            return action
        except Exception as e:
            #print("tracking_action.py error")
            pass
        