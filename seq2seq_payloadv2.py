import sys
from os import listdir
import unicodedata
import string
import torch
import torch.nn as nn
from torch.autograd import Variable
import random
import matplotlib.pyplot as plt
import os
import util
import numpy as np

'''
LEARNING! SEQ2SEQ Model using OWN RNNS, BETTER VERSION: VERSION 3
'''

letters = string.ascii_letters + string.digits + string.punctuation
letters_num = len(letters) + 1

def toAscii(s):
    return ''.join(
        char for char in unicodedata.normalize('NFD', s)
        if unicodedata.category(char) != 'Mn'
        and char in letters
    )

def lines(datei):
    f = open(datei, encoding='utf-8').read().split('\n')
    return [toAscii(l) for l in f]

def charToIndex(char):
    return letters.find(char)

def charToTensor(char):
    ret = torch.zeros(1, letters_num) #ret.size = (1, letters_num)
    ret[0][charToIndex(char)] = 1
    return ret

def passwordToTensor(name):
    ret = torch.zeros(len(name), 1, letters_num)
    for i, char in enumerate(name):
        ret[i][0][charToIndex(char)] = 1
    return ret

def targetToTensor(password):
    indizes = [letters.find(password[i]) for i in range(1,len(password))]
    indizes.append(letters_num - 1)
    return torch.LongTensor(indizes)

lines_file = lines('traindata.txt')

class Netz(nn.Module):
    def __init__(self, inputs, hiddens, outputs):
        super(Netz, self).__init__()
        self.hidden_size = hiddens

        self.input_to_output = nn.Linear(inputs + hiddens, outputs)
        self.input_to_hidden = nn.Linear(inputs + hiddens, hiddens)
        self.output_to_output = nn.Linear(hiddens + outputs, outputs)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), dim=1)
        hidden = self.input_to_hidden(combined)
        output = self.input_to_output(combined)
        out_combined = torch.cat((output, hidden), dim=1)
        output = self.output_to_output(out_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return Variable(torch.zeros(1, self.hidden_size).cuda())

def get_random_example():
    return random.choice(lines_file)

def get_random_train():
    pw = get_random_example()
    input_tensor = Variable(passwordToTensor(pw))
    target_tensor = Variable(targetToTensor(pw))
    return input_tensor, target_tensor


def tensor_to_char(tensor):
    idx = (np.where(tensor.cpu().numpy() == tensor.max()))[0][0]
    if idx >= letters_num - 1:
        return ""
    return letters[idx]

def tensor_to_string(tensor):
    output = ""
    if tensor.dim() == 3:
        for i in tensor[0]:
            output = output + tensor_to_char(i)
    elif tensor.dim() == 1:
        for idx in tensor:
            if idx < len(letters):
                output += letters[idx]
    else:
        for i in tensor:
            output = output + tensor_to_char(i)
    return output


criterion = nn.NLLLoss()
learning_rate = 0.01
def train(model, input_tensor, target_tensor):
    hidden = model.initHidden()
    model.zero_grad()
    loss = 0
    for i in range(input_tensor.size()[0]): #qwertz -> wertz[EOS]
        output, hidden = model(input_tensor[i], hidden)
        loss += criterion(output, target_tensor[i])
    loss.backward()
    torch.nn.utils.clip_grad_norm(model.parameters(), 1)
    for p in model.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss

def sample(start):
    input = Variable(passwordToTensor(start))
    hidden = model.initHidden()
    output = start
    for i in range(15):
        out, hidden = model(input[0], hidden)
        _, i = out.data.topk(1)
        i = i[0][0]
        if i == letters_num - 1:
            break
        else:
            output += letters[i]
        input = Variable(passwordToTensor(letters[i]))
    return output

best = (sys.maxsize,sys.maxsize, sys.maxsize)
for hiddens in range(1000,10000,100):
    for epochs in range(100, 10000,100):
        model = Netz(letters_num, hiddens, letters_num).cuda()

        if os.path.isfile('pwNet.pt'):
            model = torch.load('pwNet.pt').cuda()

        loss_sum = 0
        plots = []
        epochs = epochs
        loss_total = 0
        for i in range(epochs):
            input_tensor, target_tensor = get_random_train()
            input_tensor = input_tensor.cuda()
            target_tensor = target_tensor.cuda()
            output, loss = train(model, input_tensor, target_tensor)
            loss_sum += loss.data.cpu()[0]/input_tensor.size()[0]
            #if i % 1000 == 0:
             #   loss_total = loss_sum/1000
              #  print(100*i / epochs, '% made. Loss: ', loss_total)
               # plots.append(loss_total)
                #loss_sum = 0
                #torch.save(model, 'pwNet.pt')
        #plt.plot(plots)
        loss_sum = loss_sum / epochs
        if loss_sum < best[2]:
            best = (epochs, hiddens, loss_sum)
        print("Best: " + str(best))

print(best)
for i in letters:
    print(sample(i))
