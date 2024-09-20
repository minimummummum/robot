import gym
from gym import spaces
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

# Define the BallChasing environment
class BallChasingEnv(gym.Env):
    def __init__(self):
        super(BallChasingEnv, self).__init__()
        self.observation_space = spaces.Box(low=1, high=1, shape=(1,), dtype=np.uint8)
        self.action_space = spaces.Discrete(3)  # 0: 가만히, 1: 오른쪽, 2: 왼쪽
        self.max_steps = 100
        self.step_count = 0
        self.score = 0
        self.max_score = 0
        self.ball_position = 0
        self.robot_view = 0
        self.sw = False
        self.reset()

    def reset(self):
        self.score = 0
        self.step_count = 0
        self.ball_position = 0
        self.robot_view = 0
        self.done = False
        return np.array([round(self.ball_position - self.robot_view, 2)], dtype=np.float32)

    def step(self, action):
        self.step_count += 1
        if action == 1:
            self.robot_view += 0.1
        elif action == 2:
            self.robot_view -= 0.1
        self.robot_view = round(self.robot_view, 2)
        self.robot_view = np.clip(self.robot_view, -1, 1)

        self.draw_info = np.zeros((100, 100, 1), dtype=np.uint8)
        observation = np.array([round(self.ball_position - self.robot_view, 2)], dtype=np.float32)
        ball_pixel = int((self.ball_position + 1) / 2 * 84)
        robot_pixel = int((self.robot_view + 1) / 2 * 84)  # 로봇 시야를 84x84 크기에 맞춤
        self.draw_info[20:24, ball_pixel:ball_pixel+4, 0] = 255  # 공을 이미지 상에 그리기
        self.draw_info[40:44, robot_pixel:robot_pixel+4, 0] = 128  # 로봇 시야를 이미지 상에 그리기

        reward = 1 - max(abs(self.ball_position - self.robot_view), 0)
        reward = round(reward, 2)
        self.score += reward
        self.score = round(self.score, 2)
        if self.score > self.max_score:
            self.max_score = self.score

        self.render()

        if reward <= 0.7:
            self.done = True
        if self.step_count >= self.max_steps:
            self.done = True
            
        self.ball_position += random.choice([-0.1, 0, 0.1])
        self.ball_position = round(self.ball_position, 2)
        self.ball_position = np.clip(self.ball_position, -1, 1)

        return observation, reward, self.done, False, {}

    def render(self, mode='human'):
        print(f"로봇 시야: {self.robot_view}, 공 위치: {self.ball_position}")
        print(f"현재 점수: {self.score}, 현재 시도: {num_try}")
        if num_try == 4500 and self.sw == False: # 일정 횟수 학습 후 시각 표시
            self.fig, self.ax = plt.subplots()
            self.sw = True
        if self.sw:
            if self.draw_info is not None and self.draw_info.size > 0:
                self.ax.clear()  # 축을 지우고 새로 그릴 준비
                self.ax.imshow(self.draw_info[:, :, 0], cmap='gray')  # 이미지 업데이트
                if self.done:
                    self.ax.set_title("Die.....!!")
                else:
                    self.ax.set_title(f"try: {num_try}, score: {self.score}, max_score: {self.max_score}")
                try:
                    plt.pause(0.001)  # 창을 즉시 업데이트하고 넘어감
                except Exception as e:
                    print("plt 에러")
            else:
                print("관찰 값 없음")


    def close(self):
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
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.5
EPS_END = 0.05
EPS_DECAY = 600
TAU = 0.005
LR = 1e-4

# Initialize environment
env = BallChasingEnv()

# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
state = env.reset()

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

policy_net = DQN(n_inputs=1, n_actions=n_actions).to(device)
target_net = DQN(n_inputs=1, n_actions=n_actions).to(device)
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
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

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

num_episodes = 700
num_try = 0
for i_episode in range(num_episodes):
    num_try+=1
    state = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)  # (1,) -> (1, 1)으로 변환
    for t in count():
        action = select_action(state)
        next_state, reward, done, _, _ = env.step(action.item())
        next_state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0) if not done else None  # 다음 상태 처리
        
        reward = torch.tensor([reward], device=device)
        memory.push(state, action, next_state, reward)
        optimize_model()
        
        state = next_state
        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break
torch.save(policy_net.state_dict(), "policy_net.pth")
torch.save(target_net.state_dict(), "target_net.pth")
print('Training Complete')
env.close()

# Plot the final results
plot_durations(show_result=True)
plt.show()
