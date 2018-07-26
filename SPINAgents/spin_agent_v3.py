from collections import defaultdict
import queue
import random

import numpy as np
import pommerman
from pommerman import agents
from pommerman import constants
from pommerman import utility

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
import gym
from torch.autograd import Variable
import random
from collections import namedtuple

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

class DQN(nn.Module):
    def __init__(self, dueling=True):
        super().__init__()
        self.dueling = dueling
        self.conv1 = nn.Conv2d(4,4,3, padding=1)
        self.fc1 = nn.Linear(484, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 6)
        if dueling:
            self.v = nn.Linear(512, 1)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.view(-1, 484)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        if self.dueling:
            v = self.v(x)
            a = self.fc3(x)
            q = v + a
        else:
            q = self.fc3(x)
        return q
        
def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        
def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

class SPIN_3(agents.BaseAgent):

    def __init__(self, *args, **kwargs):#gamma=0.8, batch_size=128):
        super(SPIN_3, self).__init__(*args, **kwargs)
        self.target_Q = DQN()
        self.Q = DQN()
        self.gamma = 0.8
        self.batch_size = 128
        self.epsilon = 0.1
        hard_update(self.target_Q, self.Q)
        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=0.001)
        self.Loss = 0
    
    def prepInput(self, obs):

        board = np.array(list(obs['board'].copy()))
        layer1 = np.zeros(board.shape) # Walls & flames : place you can’t move to
        layer2 = np.zeros(board.shape) # Bombs : place you should move away from
        layer3 = np.zeros(board.shape) # Other agents
        layer4 = np.zeros(board.shape) # You

        # Add board to input
        for i in range(board.shape[0]):
            for j in range(board.shape[1]):
                if board[i][j] == 1 or board[i][j] == 2 or board[i][j] == 4: # Rigid Wall, Wooden Wall or flames
                    layer1[i][j] = 1
                if board[i][j] == 3: # Bombs
                    layer2[i][j] = 1
                if board[i][j] == 10 or board[i][j] == 11 or board[i][j] == 12: # Ennemies
                    layer3[i][j] = 1
          
        layer4[obs['position']] = 1
        
        # Concatenate input layers
        networkInput = np.reshape(layer1, (1, layer1.shape[0], layer1.shape[1]))
        networkInput = np.append(networkInput, layer2.reshape(1, layer2.shape[0], layer2.shape[1]), axis=0)
        networkInput = np.append(networkInput, layer3.reshape(1, layer3.shape[0], layer3.shape[1]), axis=0)
        networkInput = np.append(networkInput, layer4.reshape(1, layer4.shape[0], layer4.shape[1]), axis=0)

        # Prep input for convolution
        networkInput = networkInput.reshape(1, networkInput.shape[0], networkInput.shape[1], networkInput.shape[2])

        # print("SHAPE : ", networkInput.shape)
        # print(pd.DataFrame(networkInput[0][0]))
        # print(pd.DataFrame(networkInput[0][1]))
        # print(pd.DataFrame(networkInput[0][2]))
        # print(pd.DataFrame(networkInput[0][3]))
        # quit()
        
        return networkInput

    def act(self, obs, action_space):#self, x, epsilon=0.1):
        self.Input = Variable(torch.from_numpy(self.prepInput(obs)).type(torch.FloatTensor))
        x = self.Input
        p = random.uniform(0, 1)

        # Return random action with probability epsilon
        if p < self.epsilon :
            action = random.randint(0, 5)
            return action

        Q_sa = self.Q(x.data)
        argmax = Q_sa.data.max(1)[1]
        return argmax.data.numpy()[0]
    
    def backward(self, transitions):
        batch = Transition(*zip(*transitions))

        state = Variable(torch.cat(batch.state))
        action = Variable(torch.from_numpy(np.array(batch.action)))
        next_state = Variable(torch.cat(batch.next_state))
        reward = Variable(torch.cat(batch.reward))
        done = Variable(torch.from_numpy(np.array(batch.done)))

        Q_sa = self.Q(next_state).detach()
        target = self.target_Q(next_state).detach()

        _, argmax = Q_sa.max(dim=1, keepdim=True)
        target = target.gather(1, argmax)

        currentQvalues = self.Q(state).gather(1,action.unsqueeze(1)).squeeze()
        y = (reward.unsqueeze(1) + self.gamma * (target * (1-done.unsqueeze(1)))).squeeze()

        loss = F.smooth_l1_loss(currentQvalues,y)
        self.Loss = loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        soft_update(self.target_Q, self.Q, 0.995)
