import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import random
import numpy as np
import math
import util
import torch.nn.functional as F
from itertools import count
import os
import time
import evaluator
import requests
import json
import neuralevaluator

"""
DOES REINFORCEMENT LEARNING TO ATTACK A WEBSITE. CAN BE TRAINED BEFOREHAND BASED ON A PAYLOAD LIST SO THE ATTACKS CAN BE IMPROVED.
READ ALSO THE PAPER TO COMPREHEND WHAT THE NET DOES.
EXECUTABLE.
"""

config = json.load(open('config.json'))
eps_end = config["eps_end"]
eps_start = config["eps_start"]
eps_steps = config["eps_steps"]
batch_size = config["batch_size"]
batch_epochs = config["batch_epochs"]
gamma = config["gamma"]
hidden_size = config["hidden_size"]
num_lstm_layers = config["num_lstm_layers"]
dropout = config["dropout"]
optimizer = config["optimizer"]
learning_rate = config["learning_rate"]
living_reward = config["living_reward"]
loss_punishment = config["loss_punishment"]
win_factor = config["win_factor"]
weight_decay = config["weight_decay"]
use_rnns = config["use_rnns"] == "True"

capacity = config["capacity"]
num_epochs = config["num_epochs"]
base_reward = config["base_reward"]
filename = config["filename"]

#for unsupervised only
supervised = config["supervised"] == "True"
attack_url = config["attack_url"]
base_injection = config["base_injection"]

FloatTensor = torch.cuda.FloatTensor
LongTensor = torch.cuda.LongTensor
ByteTensor = torch.cuda.ByteTensor
Tensor = FloatTensor

total_wins = 0
total_losses = 0

class MyEmbedding(nn.Module):
    def __init__(self):
        super(MyEmbedding, self).__init__()
        self.embed = nn.Embedding(6, 7)

    def forward(self, input):
        res = []
        for i in input[0]:
            embd = self.embed(i).view(1, -1)
            res.append(embd)
        return torch.cat(res).unsqueeze_(0)

class Model(nn.Module):
    '''On Input of State s (String of previous payload) want to get a tensor with expected rewards for each char (as ONE tensor)'''
    def __init__(self, RNN = use_rnns):
        super(Model, self).__init__()
        self.use_rnns = RNN
        #takes input as (timestep, batch, data) with timestep[0] = first char...
        self.lstm = nn.LSTM(input_size=util.get_letters_num(), hidden_size=hidden_size, num_layers=num_lstm_layers, bias=True, dropout=dropout, batch_first=True)
        self.linear = nn.Linear(hidden_size, util.get_letters_num())
        self.rnn = nn.RNN(input_size=util.get_letters_num(), hidden_size=hidden_size, num_layers=num_lstm_layers, bias=True, dropout=dropout, batch_first=True)
        self.activation = nn.LogSoftmax(dim=0)
        self.lstms = nn.ModuleList([nn.LSTMCell(input_size=util.get_letters_num(), hidden_size=hidden_size).cuda() for _ in range(num_lstm_layers)])
        self.rnns = nn.ModuleList([nn.RNNCell(input_size=util.get_letters_num(), hidden_size=hidden_size).cuda() for _ in range(num_lstm_layers)])
        self.hidden_size = hidden_size
        for i in self.parameters():
            i.requires_grad = True

    def forward(self, x):
        #print(input.size())
        hidden = Variable(torch.zeros(1, self.hidden_size).cuda())
        full = x.transpose(0,1)
        if not self.use_rnns:
            x = self.lstm(x)
            x = x[0] #Dont care about hidden states
            #for char in full:
                #for i in self.lstms:
                    #hidden, char = i(char, (hidden, char))
                    #TODO currently in development, try later :(
        else:
            x = self.rnn(x)
            x = x[0]
        x = x[0][len(x[0])-1] #Output of LSTM: (Prediction for 2nd Char, Prediction for 3rd Char, ...) Only want the one for the next char after the end
        x = self.linear(x)
        return x


class Memory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.pos = 0

    def push(self, state, action, next_state, reward):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.pos] = (state, action, next_state, reward)
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class Agent:
    def __init__(self, eps_end, eps_start, eps_steps, batch_size, gamma, optimizer, filename = None, supervised=True, neuraleval = True):
        self.eps_end = eps_end
        self.eps_start = eps_start
        self.eps_steps = eps_steps
        self.batch_size = batch_size
        self.gamma = gamma
        self.state = " "
        self.done = 0
        self.model = Model().cuda()
        self.memory = Memory(capacity)
        self.supervised = supervised
        if filename is not None and os.path.isfile(filename):
            self.model = torch.load(filename)
        if optimizer == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        if optimizer == "rms":
            self.optimizer = optim.RMSprop(self.model.parameters(), weight_decay=weight_decay)
        self.filename = filename
        self.error = []
        self.epsilon = -1
        self.loss_fn = nn.MSELoss()
        if not supervised and not neuraleval:
            self.evaluator = evaluator.BasicEvaluator(url=attack_url, base_injection_dict=base_injection)
        elif not supervised:
            self.evaluator = neuralevaluator.NeuralEvaluator(evaluator.BasicEvaluator(url=attack_url, base_injection_dict=base_injection)) #TODO

    def train(self):
        for i in range(num_epochs):
            #print("###############################")
            print("######    EPOCH " + str(i) + " of " + str(num_epochs) + " (" + str(100*i/num_epochs) + "%)    ######")
            #print("###############################")
            global total_wins
            global total_losses
            game = []
            loss = -1
            for j in count():
                loss = 0
                for i in range(batch_epochs):
                    loss += self.backprop()
                loss /= batch_epochs
                if loss < 10000:
                    self.appendPlots(loss)
                else:
                    self.appendPlots(1000)

                #if loss > 0:
                    #print("Loss: " + str(loss))
                action = self.get_action()
                next_state, wins = self.take_action(action)
                game.append((self.state, action, next_state))

                if action == util.get_EOS_token() and self.state == " " and wins == -1:
                    self.promote_reward(wins*base_reward*base_reward, game)
                    print("Chose EOS and getting punished now")
                else:
                    self.promote_reward(wins*base_reward, game)

                self.state = next_state

                if wins >= 1:
                    print("ENDED GAME: " + self.state + " WON!")
                    total_wins = total_wins +1
                    self.epsilon = -1
                    break
                elif wins == -1*loss_punishment:
                    print("ENDED GAME: " + self.state + " LOST!")
                    total_losses = total_losses + 1
                    self.epsilon = -1
                    break
                elif action == util.get_EOS_token():
                    break
            self.state = " "
            self.saveModel()
        self.plotError()

    def backprop(self):
        if len(self.memory) < batch_size:
            return -1
        x = self.memory.sample(self.batch_size)#[(s, a, n, r)]

        batch = tuple(zip(*x))#((s1,s2,...),(a1,a2,...), (n1,n2,...), (r1,r2,...))

        assert len(batch[0]) == batch_size and len(batch[3]) == batch_size and len(batch[2]) == batch_size and len(batch[1]) == batch_size

        non_final = ByteTensor(tuple(map(lambda s: s is not None, batch[2])))
        non_final_next = Variable(torch.cat([util.example_to_tensor(s).cuda() for s in batch[2]]), volatile=True)
        state = Variable(torch.cat([util.example_to_tensor(s).cuda() for s in batch[0]]))
        state.requires_grad = True
        action = Variable(torch.cat([util.char_to_tensor(s).unsqueeze(0).cuda() for s in batch[1]]).type(LongTensor))
        reward = Variable(Tensor([s for s in batch[3]]))

        model_result = self.model(state)
        action_value = self.get_action_value(action, model_result)
        #next_value = Variable(torch.zeros(batch_size).cuda().type(FloatTensor))
        #prediction = self.model(non_final_next).max(1)[0]
        #next_value[non_final] = prediction.cuda()
        #next_value.volatile = False
        #target_action_value = (next_value * self.gamma) + reward
        target_action_value = reward

        loss = self.loss_fn(action_value, target_action_value)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.data[0]


    def promote_reward(self, base_reward, draws):
        reward = base_reward
        for (state, action, next_state) in reversed(draws):
            self.memory.push(state, action, next_state, reward)
            reward = reward * self.gamma


    def get_action_value(self, input_actions, model_output):
        action_values = []
        #output = self.model(states)
        output = model_output
        inp = input_actions.view(-1).data
        actions = Variable(LongTensor([i for i in range(len(inp)) if inp[i] == 1]))
        out = model_output.take(actions)
        return out


    def get_action(self):
        if self.supervised:
            return self.get_supervised_action()
        else:
            return self.get_unsupervised_action()

    def take_action(self, action):
        if self.supervised:
            return self.take_supervised_action(action)
        else:
            return self.take_unsupervised_action(action)

    def take_supervised_action(self, action):
        next_state = self.state
        # delete first character
        if action == util.get_EOS_token():
            if next_state[1:] in util.get_all_targets():
                # print("good ending with EOS")
                return next_state, win_factor
            else:
                return next_state, -1 * loss_punishment
        next_state = self.state + action
        if any(l.startswith(next_state[1:]) for l in util.get_all_targets()):
            return next_state, living_reward
        else:
            return next_state, -1 * loss_punishment

    def take_unsupervised_action(self, action):
        next_state = self.state
        if action != util.get_EOS_token():
            next_state += action
        resp = self.evaluator(next_state[1:], target="post")
        # delete first character
        if resp == True and action != util.get_EOS_token():
            return next_state, living_reward
        elif resp == False:
            return next_state, -1 * loss_punishment
        else:
            if resp == True:
                return next_state, -1 * loss_punishment
            return next_state, win_factor

    def get_supervised_action(self):
        if self.epsilon == -1:
            self.epsilon = random.random()
        threshold = 0.5 #eps_end + (eps_start - eps_end) * math.exp(-1. * self.done / eps_steps)
        self.done = self.done + 1
        if self.epsilon > threshold: #TODO
            possibles = [s for s in util.get_all_targets_copied() if s.startswith(self.state[1:])]
            pick = random.choice(possibles)
            #pick = possibles[0] #TODO besser???
            if len(self.state)-1 < len(pick):
                char = pick[len(self.state)-1:len(self.state)]
            else:
                #util.get_all_targets_copied().pop(0)
                #if len(util.get_all_targets_copied()) == 0:
                    #util.lines_copied = util.lines(util.targetstrings)#TODO
                char = util.get_EOS_token()
                self.epsilon = -1
            return char
        else:
            epsi = random.random()
            threshold = eps_end + (eps_start - eps_end) * math.exp(-1. * self.done / eps_steps)
            self.done = self.done + 1
            if epsi > threshold:
                state_tensor = util.example_to_tensor(self.state).cuda()
                input = Variable(state_tensor, volatile=True)
                data = self.model(input).data
                char = util.tensor_to_char(data)
                #print("picked " + char + " with expected reward of " + str(data.max()))
                return char
            else:
                return random.choice(util.get_letters())

    def get_unsupervised_action(self):
        epsilon = random.random()
        threshold = eps_end + (eps_start - eps_end) * math.exp(-1. * self.done / eps_steps)
        self.done = self.done + 1
        if epsilon > threshold:
            state_tensor = util.example_to_tensor(self.state).cuda()
            input = Variable(state_tensor, volatile=True)
            data = self.model(input).data
            char = util.tensor_to_char(data)
            #print("picked " + char + " with expected reward of "  + str(data.max()))
            return char
        else:
            return random.choice(util.get_letters())


    def saveModel(self):
        if self.filename is not None:
            if self.error[len(self.error)-1] == min(self.error):
                torch.save(self.model, self.filename)

    def appendPlots(self, loss):
        self.error.append(loss)

    def plotError(self):
        plt.figure()
        plt.plot(self.error)
        plt.show()

if __name__ == "__main__":
    start = time.time()
    agent = Agent(eps_end, eps_start, eps_steps, batch_size, gamma, optimizer, filename = None, supervised=supervised)
    agent.train()
    end = time.time()
    print("Total Training Time: " + str(end-start) + " for " + str(num_epochs*batch_epochs) + " training steps")
    print("Losses: " + str(total_losses))
    print("Wins: " + str(total_wins))