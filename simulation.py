import pybullet as p
import time
import pybullet_data
import numpy as np
class Simulation():
    def __init__(self):
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)
        p.loadURDF("plane.urdf")
        robotStartPos = [0, 0, 0.05]
        robotStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
        self.robotId = p.loadURDF("robot.urdf", robotStartPos, robotStartOrientation)
        camera_distance = 0.5
        camera_yaw = 180
        camera_pitch = -30
        camera_target = [0, 0, 0]
        p.resetDebugVisualizerCamera(camera_distance, camera_yaw, camera_pitch, camera_target)
        self.left_shoulder_joint_index = 4
        self.left_elbow_joint_index = 7
        self.right_shoulder_joint_index = 49
        self.right_elbow_joint_index = 52
        # 15 20 31 36 
    def simulation_run(self, data):
        for _ in range(50):
            p.resetBasePositionAndOrientation(self.robotId, [0, 0, 0.28], [0, 0, 0, 1])
            p.setJointMotorControl2(self.robotId, self.left_shoulder_joint_index, p.POSITION_CONTROL, np.radians(data[0]))
            p.setJointMotorControl2(self.robotId, self.left_elbow_joint_index, p.POSITION_CONTROL, np.radians(-data[2]))
            p.setJointMotorControl2(self.robotId, self.right_shoulder_joint_index, p.POSITION_CONTROL, np.radians(-data[1]))
            p.setJointMotorControl2(self.robotId, self.right_elbow_joint_index, p.POSITION_CONTROL, np.radians(data[3]))
            p.setJointMotorControl2(self.robotId, 15, p.POSITION_CONTROL, 0, 1)
            p.setJointMotorControl2(self.robotId, 20, p.POSITION_CONTROL, 0, 1)
            p.setJointMotorControl2(self.robotId, 31, p.POSITION_CONTROL, 0, 1)
            p.setJointMotorControl2(self.robotId, 36, p.POSITION_CONTROL, 0, 1)
            p.stepSimulation()
            time.sleep(1. / 240.)
    #if cv2.waitKey(5) & 0xFF == ord('q'):
        #break
#p.disconnect()
