import sys
import unicodedata
import string
import torch
import torch.nn as nn
from torch.autograd import Variable
import random
import matplotlib.pyplot as plt
import os
import numpy as np

'''
EXECUTABLE
CURRENT VERSION OF THE SEQ2SEQ NEURAL NETWORK TO GENERATE PAYLOADS FOR ATTACKING WEBSITES
'''


letters = string.ascii_letters + string.digits + string.punctuation + string.whitespace
letters_num = len(letters) + 1

def toAscii(s):
    """converts a unicode string to an ascii string"""
    return ''.join(
        char for char in unicodedata.normalize('NFD', s)
        if unicodedata.category(char) != 'Mn'
        and char in letters
    )

def lines(datei):
    if datei is not None and os.path.isfile(datei):
        f = open(datei, encoding='utf-8').read().split('\n')
        return [toAscii(l) for l in f]
    return []

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


class Netz(nn.Module):
    """The neural network for generating the payloads"""
    def __init__(self,input, hiddens, outputs, num_layers=2, dropout=0.8, is_cuda = True):
        super(Netz, self).__init__()
        self.hidden_size = hiddens
        self.num_layers = num_layers
        self.input = input
        self.output_size = outputs
        if is_cuda:
            self.lstms = nn.ModuleList([nn.LSTMCell(input_size=input, hidden_size=hiddens).cuda() for _ in range(num_layers)])
        else:
            self.lstms = nn.ModuleList([nn.LSTMCell(input_size=input, hidden_size=hiddens) for _ in range(num_layers)])
        #self.lstm = nn.LSTMCell(input_size=hiddens, hidden_size=outputs)
        self.final_layer = nn.Linear(hiddens, outputs)
        self.output_to_hidden = nn.Linear(outputs, hiddens)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.LogSoftmax(dim=1)
        self.is_cuda = is_cuda

    def forward(self, input, hidden):
        output, hidden = hidden
        #hidden, output = self.lstm(input, (hidden, output))
        output = self.output_to_hidden(output)
        for index, i in enumerate(self.lstms):
            hidden, output = i(input, (hidden, output))
            if index < len(self.lstms) - 1:
                output = self.dropout(output)
        output = self.softmax(self.final_layer(output))
        return output, hidden

    def initHidden(self):
        vars = torch.zeros(1, self.output_size)
        var2 = torch.zeros(1, self.hidden_size)
        if self.is_cuda:
            vars = vars.cuda()
            var2 = var2.cuda()
        return (Variable(vars), Variable(var2))

class Agent:
    """The wrapper for the neural network with lots of convenience functions"""
    def __init__(self, learning_rate=0.01, is_cuda = True, train_data_path = 'traindata.txt'):
        self.criterion = nn.NLLLoss()
        self.learning_rate = min(1, max(0,learning_rate))
        self.lines_file = lines(train_data_path)
        self.is_cuda = is_cuda if torch.cuda.is_available() else False

    def get_random_example(self):
        return random.choice(self.lines_file)

    def get_random_train(self):
        pw = self.get_random_example()
        input_tensor = passwordToTensor(pw)
        target_tensor = targetToTensor(pw)
        if self.is_cuda:
            input_tensor = input_tensor.cuda()
            target_tensor = target_tensor.cuda()
        return Variable(input_tensor), Variable(target_tensor)


    def tensor_to_char(self, tensor):
        idx = (np.where(tensor.cpu().numpy() == tensor.max()))[0][0]
        if idx >= letters_num - 1:
            return ""
        return letters[idx]

    def tensor_to_string(self, tensor):
        output = ""
        if tensor.dim() == 3:
            for i in tensor[0]:
                output = output + self.tensor_to_char(i)
        elif tensor.dim() == 1:
            for idx in tensor:
                if idx < len(letters):
                    output += letters[idx]
        else:
            for i in tensor:
                output = output + self.tensor_to_char(i)
        return output


    def train(self, model, input_tensor, target_tensor):
        output, hidden = model.initHidden()
        model.zero_grad()
        loss = 0
        for i in range(input_tensor.size()[0]): #qwertz -> wertz[EOS]
            output, hidden = model(input_tensor[i], (output, hidden))
            loss += self.criterion(output, target_tensor[i])
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 1)
        for p in model.parameters():
            if p.grad is not None:
                p.data.add_(-self.learning_rate, p.grad.data)
        return output, loss

    def sample(self, model, start, maxlength=15):
        input = passwordToTensor(start)
        if self.is_cuda:
            input = input.cuda()
        input = Variable(input)
        out, hidden = model.initHidden()
        output = start
        for i in range(maxlength):
            out, hidden = model(input[0], (out,hidden))
            _, i = out.data.topk(1)
            i = i[0][0]
            if i >= letters_num - 1:
                break
            else:
                output += letters[i]
            input = passwordToTensor(letters[i])
            if self.is_cuda:
                input = input.cuda()
            input = Variable(input)
        return output

    def plot(self, losses, name):
        plt.figure()
        plt.plot(losses)
        plt.savefig("errors/" + name + ".png")

    def start_training(self, hiddens, dropout, epochs, num_layers, path = None, plot = True):
        dropout = min(1, max(0, dropout))

        if path is not None and os.path.isfile(path):
            model = torch.load(path)

        if self.is_cuda:
            model = model.cuda()

        loss_sum = 0
        plots = []
        loss_total = 0
        for i in range(epochs):
            input_tensor, target_tensor = self.get_random_train()
            if self.is_cuda:
                input_tensor = input_tensor.cuda()
                target_tensor = target_tensor.cuda()
            output, loss = self.train(model, input_tensor, target_tensor)
            loss_sum += loss.data.cpu()[0] / input_tensor.size()[0]
            plots.append(loss.data.cpu()[0] / input_tensor.size()[0])

            print(agent.sample(model, "'"))
            # if i % 1000 == 0:
            #   loss_total = loss_sum/1000
            #  print(100*i / epochs, '% made. Loss: ', loss_total)
            # plots.append(loss_total)
            # loss_sum = 0
            # torch.save(model, 'pwNet.pt')
        # plt.plot(plots)
        loss_sum = loss_sum / epochs
        if plot:
            self.plot(plots,"epochs_" + str(epochs) + "_hiddens_" + str(hiddens) + "_dropout_" + str(dropout) + "_numlayers_" + str(num_layers))
        if path is not None:
            self.save(model, path)
        return model

    def save(self, model, path):
        if path is not None:
            torch.save(model, path)

    def load(self, path):
        if path is not None and os.path.isfile(path):
            model = torch.load(path)
            if self.is_cuda:
                model = model.cuda()
            return model
        return None

if __name__ == "__main__":
    agent = Agent(is_cuda=True)
    hiddens_list = [1000]
    for hiddens in hiddens_list:
        for dropout in range(20, 100, 200):
            epochs = 4000
            for num_layers in range(4,12,2):
                print("###\t   HIDDENS {} EPOCHS {} DROPOUT {} NUMLAYERS {}  \t###".format(hiddens, epochs, dropout, num_layers))
                drop = dropout/100
                model = agent.start_training(hiddens, drop, epochs, num_layers)
                for i in letters:
                    print(agent.sample(model, i))
