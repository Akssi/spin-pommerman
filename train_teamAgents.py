import pommerman
from pommerman import agents
import SPINAgents

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

import sys, getopt

from tensorboardX import SummaryWriter


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def save_checkpoint(state, agent):
    filename = "Checkpoints/" + agent + '_game #' + str(state['epoch']) + ".pth"
    torch.save(state, filename)  

def load_checkpoint(agent, path):
    checkpoint = torch.load(path)
    agent.Q.load_state_dict(checkpoint['state_dict_Q'])
    agent.target_Q.load_state_dict(checkpoint['state_dict_target_Q'])
    agent.optimizer.load_state_dict(checkpoint['optimizer'])
    
    return checkpoint['epoch']

def main(argv):
    checkpointFilePath = ''
    alwaysRender = False
    useCuda = True
    try:
        opts, args = getopt.getopt(argv,"hrc:a:g:",["checkpoint=","agent=", "cuda="])
    except getopt.GetoptError:
        print('Error in command arguments. Run this for help:\n\ttrain_singleAgent.py -h')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-r':
            alwaysRender = True
        elif opt == '-h':
            print('train_singleAgent.py\n-c <checkpointfile> => Resume training from a saved checkpoint\n-a(--agent) <agent version> => Version of agent to train (default=0)\n-r => Always render')
            sys.exit()
        elif opt in ("-c", "--checkpoint"):
            checkpointFilePath = arg
        elif opt in ("-a", "--agent"):
            agentName = arg
        elif opt in ("-g","--cuda"):
            useCuda = arg 
   

    # Create a set of agents (exactly four)
    agent_list = [
        agents.SimpleAgent(),
        agents.SimpleAgent(),
        agents.SimpleAgent()
    ]
    if agentName == "2":
        agent_list.append(SPINAgents.SPIN_2())
    elif agentName == "1":
        agent_list.append(SPINAgents.SPIN_1())
    else:
        agent_list.append(SPINAgents.SPIN_0())


    # Make the "Team" environment using the agent list
    env = pommerman.make('PommeTeam-v0', agent_list)
    memory = ReplayMemory(100000)
    batch_size = 128
    epsilon = 1
    rewards = []
    start_epoch = 0

    # Writer to log data to tensorboard
    writer = SummaryWriter()

    if checkpointFilePath != '':
        start_epoch = load_checkpoint(agent_list[3], checkpointFilePath)

    # Run the episodes just like OpenAI Gym
    for i in range(start_epoch, 5750):
        state = env.reset()
        done = False
        total_reward = [0] * len(agent_list)
        epsilon *= 0.995
        while not done and agent_list[3]._character.is_alive:
            if i > 4990 or alwaysRender:
                env.render()
            # Set epsilon for our learning agent
            agent_list[3].epsilon = max(epsilon, 0.1)

            actions = env.act(state)
            agentAction = actions[3]
            actions[3] = actions[3].data.numpy()[0]
            obs_input = Variable(torch.from_numpy(agent_list[3].prepInput(state[3])).type(torch.FloatTensor))
            next_obs, reward, done, _ = env.step(actions)
            state = next_obs
            if not agent_list[3]._character.is_alive:
                reward[3] = -1
            # Fill replay memory for our learning agent
            memory.push(agent_list[3].Input, actions[3],
                torch.from_numpy(agent_list[3].prepInput(state[3])).type(torch.FloatTensor), torch.Tensor([reward[3]]),
                torch.Tensor([done]))
            total_reward = [x + y for x, y in zip(total_reward, reward)]
        rewards.append(total_reward)

        # Creates a dictionary with agent name and rewards to be displayed on tensorboard
        total_reward_list = []
        for j in range(len(total_reward)):
            total_reward_list.append((type(agent_list[j]).__name__+'('+str(j)+')', total_reward[j]))
        writer.add_scalars('data/rewards', dict(total_reward_list), i)
        writer.add_scalar('data/epsilon', agent_list[3].epsilon, i)
        writer.add_scalar('data/memory', memory.__len__(), i)

        print("Episode : ", i)
        if memory.__len__() > 10000:
            batch = memory.sample(batch_size)
            agent_list[3].backward(batch)
        if i > 0 and i % 750 == 0:
            save_checkpoint({
                    'epoch': i + 1,
                    'arch': 0,
                    'state_dict_Q': agent_list[3].Q.state_dict(),
                    'state_dict_target_Q': agent_list[3].target_Q.state_dict(),
                    'best_prec1': 0,
                    'optimizer' : agent_list[3].optimizer.state_dict(),
                }, agent_list[3].__class__.__name__)
    env.close()

    save_checkpoint({
            'epoch': 5000 + 1,
            'arch': 0,
            'state_dict_Q': agent_list[3].Q.state_dict(),
            'state_dict_target_Q': agent_list[3].target_Q.state_dict(),
            'best_prec1': 0,
            'optimizer' : agent_list[3].optimizer.state_dict(),
        }, agent_list[3].__class__.__name__)

    writer.close()

if __name__ == '__main__':
    main(sys.argv[1:])