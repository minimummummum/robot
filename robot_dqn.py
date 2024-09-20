import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
    
class TrackingDQN():
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_actions = 3 # 0: 가만히, 1: 오른쪽, 2: 왼쪽
        n_inputs = 1
        self.policy_net = DQN(n_inputs=n_inputs, n_actions=n_actions).to(self.device)
        self.policy_net.load_state_dict(torch.load("policy_net.pth"))
        self.policy_net.eval()
    def select_action(self, state):
        state = torch.tensor([state], dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return self.policy_net(state).max(1)[1].unsqueeze(0)