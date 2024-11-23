import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple, deque
import matplotlib.pyplot as plt
import math
from itertools import count

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

        print(self.reward, action, self.step_count, self.state_space)
        if self.step_count >= self.max_steps:
            self.done = True
        return self.state_space, self.reward, self.done

    def render(self):
        pass

# Define DQN components
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

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
# Hyperparameters
BATCH_SIZE = 256
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TAU = 0.005
LR = 1e-4

# Initialize environment
env = MappingRobotEnv()

# Get number of actions from gym action space
n_actions = env.action_space
# Get the number of state observations
state = env.reset()

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

policy_net = DQN(n_inputs=2, n_actions=n_actions).to(device)
target_net = DQN(n_inputs=2, n_actions=n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].unsqueeze(0)
    else:
        return torch.tensor([[random.randint(0, env.action_space - 1)]], device=device, dtype=torch.long)

episode_durations = []
fig2, ax2 = plt.subplots()
def plot_durations(show_result=False):
    global ax2, fig2
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    
    ax2.clear()  # 축을 지우고 새로 그릴 준비
    if show_result:
        ax2.set_title('Result')
    else:
        ax2.set_title('Training...')
    
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Duration')
    ax2.plot(durations_t.numpy())  # 기존 축 위에 그래프 업데이트

    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        ax2.plot(means.numpy())
    try:
        plt.pause(0.001)  # 창을 즉시 업데이트하고 넘어감
    except Exception as e:
        print("plotting 오류")

def optimize_model():
    if len(memory) < BATCH_SIZE * 10:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    
    # Extract batch components
    state_batch = torch.cat([s for s in batch.state])
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    
    non_final_mask = torch.tensor([s is not None for s in batch.next_state], device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None], dim=0)
    
    # Compute state-action values
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    
    # Compute expected state-action values
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    
    # Compute loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 10)
    optimizer.step()

    # Update the target network
    for target_param, param in zip(target_net.parameters(), policy_net.parameters()):
        target_param.data.copy_(TAU * param.data + (1.0 - TAU) * target_param.data)

num_episodes = 1000
num_try = 0
for i_episode in range(num_episodes):
    num_try+=1
    state = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)  # (1,) -> (1, 1)으로 변환
    for t in count():
        action = select_action(state)
        next_state, reward, done = env.step(action.item())
        next_state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0) if not done else None  # 다음 상태 처리
        
        reward = torch.tensor([reward], device=device)
        memory.push(state, action, next_state, reward)
        optimize_model()
        state = next_state
        if done:
            episode_durations.append(reward)
            plot_durations()
            break
torch.save(policy_net.state_dict(), "policy_net_mapping.pth")
torch.save(target_net.state_dict(), "target_net_mapping.pth")
print('Training Complete')
#env.close()
