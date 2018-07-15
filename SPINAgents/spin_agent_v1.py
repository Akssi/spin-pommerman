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
        self.conv1 = nn.Conv1d(3,1,1)
        self.fc1 = nn.Linear(169, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 6)
        if dueling:
            self.v = nn.Linear(512, 1)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.reshape(x.shape[0],x.shape[2])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        if self.dueling:
            v = self.v(x)
            a = self.fc3(x)
            # Why is tensor of lenght 1 ????? 
            # a = 1x6 and v = 1x1 so q = 1x6 ?????
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

class SPIN_1(agents.BaseAgent):

    def __init__(self, *args, **kwargs):#gamma=0.8, batch_size=128):
        super(SPIN_1, self).__init__(*args, **kwargs)
        self.target_Q = DQN()
        self.target_Q.cuda()
        self.Q = DQN()
        self.Q.cuda()
        self.gamma = 0.8
        self.batch_size = 128
        self.epsilon = 0.1
        hard_update(self.target_Q, self.Q)
        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=0.0001)
    
    def prepInput(self, obs):
        # Add board to input
        board = np.array(list(obs['board'].copy())).flatten()
        networkInput = board.reshape(board.size, 1)
        # Add bomb strength map
        bomb_blast_map = np.array(list(obs['bomb_blast_strength'].copy())).flatten()
        networkInput = np.append(networkInput, bomb_blast_map.reshape(bomb_blast_map.size, 1), axis=1)
        # Add position as a board map with '1' to denote player position
        positionBoard = obs['board'].copy()
        for i in range(len(obs['board'])):
            for j in range(len(obs['board'])):
                if (i,j) == obs['position']:
                    positionBoard[i][j] = 1
                else:
                    positionBoard[i][j] = 0
        positionBoard = np.array(list(positionBoard)).flatten()
        networkInput = np.append(networkInput, positionBoard.reshape(positionBoard.size, 1), axis=1)
        # Prep input for convolution
        networkInput = np.transpose(networkInput)
        networkInput = np.reshape(networkInput, (1, networkInput.shape[0], networkInput.shape[1]))

        return networkInput

    def act(self, obs, action_space):#self, x, epsilon=0.1):
        self.Input = Variable(torch.from_numpy(self.prepInput(obs)).type(torch.FloatTensor))
        x = self.Input.cuda()
        p = random.uniform(0, 1)
        if p < self.epsilon :
            action = int(np.round(random.uniform(-0.5, 5.5)))
            action = max(0, min(action, 5))
            return Variable(torch.Tensor([action])).type(torch.LongTensor).cpu()
        Q_sa = self.Q(x.data)
        argmax = Variable(Q_sa.data.max(0)[1]) 
        return argmax.cpu()
    
    def backward(self, transitions):
        batch = Transition(*zip(*transitions))

        state = Variable(torch.cat(batch.state)).cuda()
        for i in range(len(batch.action)):
            action = batch.action[i]
            if action.size > 1:
                batch.action[i] = action[0]
        action = Variable(torch.from_numpy(np.array(batch.action))).cuda()
        next_state = Variable(torch.cat(batch.next_state)).cuda()
        reward = Variable(torch.cat(batch.reward)).cuda()
        done = Variable(torch.from_numpy(np.array(batch.done))).cuda()

        Q_sa = self.Q(next_state).detach()
        target = self.target_Q(next_state).detach() 
        _, argmax = Q_sa.max(dim=1, keepdim=True)
        target = target.gather(1, argmax)

        currentQvalues = self.Q(state).gather(1,action.unsqueeze(1)).squeeze()
        y = (reward.unsqueeze(1) + self.gamma * (target * (1-done.unsqueeze(1)))).squeeze()

        loss = F.smooth_l1_loss(currentQvalues,y)  
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        soft_update(self.target_Q, self.Q, 0.995)
