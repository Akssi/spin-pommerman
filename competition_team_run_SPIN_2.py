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
    filename = agent + '_game #' + str(state['epoch']) + ".pth"
    torch.save(state, filename)  

def main():
    # Print all possible environments in the Pommerman registry
    print(pommerman.registry)

    # Create a set of agents (exactly four)
    agent_list = [
        agents.SimpleAgent(),
        agents.RandomAgent(),
        agents.SimpleAgent(),
        SPINAgents.SPIN_2()
        # agents.DockerAgent("pommerman/simple-agent", port=12345),
    ]
    # Make the "Team" environment using the agent list
    # env = pommerman.make('PommeTeam-v0', agent_list)
    env = pommerman.make('PommeTeamFast-v0', agent_list)
    memory = ReplayMemory(100000)
    batch_size = 128
    epsilon = 1
    rewards = []

    # Run the episodes just like OpenAI Gym
    for i in range(5000):
        state = env.reset()
        done = False
        total_reward = 0
        epsilon *= 0.99
        while not done and agent_list[3]._character.is_alive:
            if i > 4950:
                env.render()
            # Set epsilon for our learning agent
            agent_list[3].epsilon = max(epsilon, 0.1)

            actions = env.act(state)
            agentAction = actions[3]
            actions[3] = actions[3].data.numpy()[0]
            obs_input = Variable(torch.from_numpy(agent_list[3].prepInput(state[3])).type(torch.FloatTensor))
            next_obs, reward, done, _ = env.step(actions)
            state = next_obs
            
            # Fill replay memory for our learning agent
            memory.push(agent_list[3].Input, actions[3],
                torch.from_numpy(agent_list[3].prepInput(state[3])).type(torch.FloatTensor), torch.Tensor([reward[3]]),
                torch.Tensor([done]))
            total_reward += reward[3]
        rewards.append(total_reward)
        print("Episode : ", i)
        if memory.__len__() > 10000:
            batch = memory.sample(batch_size)
            agent_list[3].backward(batch)
        if i > 0 and i % 250 == 0:
            pd.DataFrame(rewards).rolling(50, center=False).mean().plot()
            plt.show()
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

    pd.DataFrame(rewards).rolling(50, center=False).mean().plot()
    plt.show()


if __name__ == '__main__':
    main()