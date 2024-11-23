import math
import numpy as np
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class MappingRobotEnv():
    def __init__ (self):
        super(MappingRobotEnv, self).__init__()
        self.action_space = 5 # 0: 대기 1: 어깨모터 +10, 2: 어깨모터 -10, 3: 팔꿈치모터 +10, 4: 팔꿈치모터 -10
        self.state_space = [0, 0] # 0: target x, target y, robot x, robot y, 어깨 각도, 팔꿈치 각도
        self.target_state_space = [0, 0]
        self.robot_state_space = [0, 0]
        self.robot_angle_space = [0, 0]
        self.arm_length = 1
        self.max_steps = 200
        self.reset()
    def random_target_state(self):
        angle_range1 = random.uniform(math.radians(-30), math.radians(50))
        angle_range2 = random.uniform(math.radians(0), math.radians(180))
        x = self.arm_length * math.cos(angle_range1)
        y = self.arm_length * math.sin(angle_range2)
        return x, y
    def reset(self):
        self.target_state_space = self.random_target_state() # 어깨가 원점이라 가정했을 때 팔꿈치 좌표, 나중엔 팔꿈치좌표 - 어깨좌표로 원점이동
        self.robot_state_space = [0, -self.arm_length]
        self.state_space = [self.state_check(self.target_state_space[0], self.robot_state_space[0]),
                            self.state_check(self.target_state_space[1], self.robot_state_space[1])]
        self.robot_angle_space = [0, 0] # 팔꿈치, 어깨
        self.score = 0
        self.step_count = 0
        self.done = False
        self.wait_sw = True
        self.reward = 0
        return self.state_space

    def state_check(self, a, b):
        c = 1 / (1 + abs(a - b))
        if c > 0.9:
            return 0
        elif (b - a) > 0:
            return 1
        else:
            return -1
        
    def step(self, action):
        self.step_count += 1
        if action == 1: # 1: 어깨모터  y+
            self.robot_angle_space[1] += 10
        elif action == 2: # 2: 어깨모터  y-
            self.robot_angle_space[1] -= 10
        elif action == 3: # 3: 팔꿈치모터  x+
            self.robot_angle_space[0] += 10
        elif action == 4: # 4: 팔꿈치모터  x-
            self.robot_angle_space[0] -= 10
        #self.robot_angle_space[1] = np.clip(self.robot_angle_space[1], 0, 180)
        #self.robot_angle_space[0] = np.clip(self.robot_angle_space[0], -30, 50)
        self.robot_state_space[0] = self.arm_length * math.sin(np.radians(self.robot_angle_space[0])) # 팔꿈치 x
        self.robot_state_space[1] = -(self.arm_length * math.cos(np.radians(self.robot_angle_space[1]))) # 어깨 y
        self.state_space = [self.state_check(self.target_state_space[0], self.robot_state_space[0]),
                            self.state_check(self.target_state_space[1], self.robot_state_space[1])]
        states = abs(self.state_space[0]) + abs(self.state_space[1])
        if states == 0:
            self.reward = 1000
        elif states == 1:
            self.reward = -10
        else:
            self.reward = -50
        if self.step_count >= self.max_steps:
            self.done = True
        return self.state_space, self.reward, self.done

    def render(self):
        pass


class DQN(nn.Module):
    def __init__(self, n_inputs, n_actions):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(n_inputs, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Load trained model
env = MappingRobotEnv()
n_actions = env.action_space
state_dim = 2

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load policy network
policy_net = DQN(n_inputs=state_dim, n_actions=n_actions).to(device)
policy_net.load_state_dict(torch.load("policy_net_mapping.pth"))
policy_net.eval()

def select_action(state):
    with torch.no_grad():
        return policy_net(state).max(1)[1].item()

# Run test episodes
num_episodes = 10
for i_episode in range(num_episodes):
    state = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    for t in range(env.max_steps):
        action = select_action(state)
        next_state, reward, done = env.step(action)
        
        state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)

        if done:
            print(f"Episode {i_episode+1} ended with total reward: {reward}")
            break
