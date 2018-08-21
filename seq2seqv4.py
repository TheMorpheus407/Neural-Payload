import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import util
import os
import random
import unicodedata
import string
import numpy as np


'''
NOT LEARNING! SEQ2SEQ Model using LSTMCELL
'''

is_cuda = False
FloatTensor = torch.cuda.FloatTensor if is_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if is_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if is_cuda else torch.ByteTensor
Tensor = FloatTensor

hidden_size = 1000
num_layers = 4
is_cell = False

letters = string.ascii_letters + string.digits + string.punctuation + string.whitespace
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

def tensor_to_char(tensor):
    idx = (np.where(tensor.cpu().numpy() == tensor.max()))[0][0]
    if idx >= letters_num - 1:
        return ""
    return letters[idx]

def tensor_to_string(tensor):
    output = ""
    if tensor.dim() == 1:
        for idx in tensor:
            if idx < len(letters):
                output += letters[idx]
    else:
        for i in tensor:
            output = output + tensor_to_char(i)
    return output

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size = hidden_size, dropout = 0.3, is_cell = False):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        if is_cell:
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=False)
        else:
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.lstms = nn.ModuleList([nn.LSTMCell(input_size=input_size, hidden_size=hidden_size) for _ in range(num_layers)]) #TODO
        self.linear = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.is_cell = is_cell
        self.input_size = input_size


    def forward(self, x):
        #x = self.lstm(x)
        #x = x[0]  # Dont care about hidden states
        ret = []
        for char in range(x.size()[0]):
            hiddens, hiddens2 = self.initHidden()
            for count, i in enumerate(self.lstms):
                hiddens, hiddens2 = i(x[char], (hiddens, hiddens2))
            ret.append(hiddens)
            '''if self.is_cell:
                x = x[0][len(x[0]) - 1] # Output of LSTM: (Prediction for 2nd Char, Prediction for 3rd Char, ...) Only want the one for the next char after the end
                x = self.linear(x)'''
        x = torch.stack(ret)
        return self.relu(x)

    def initHidden(self):
        vars = torch.zeros(1, self.hidden_size)
        var2 = torch.zeros(1, self.hidden_size)
        if is_cuda:
            vars = vars.cuda()
            var2 = var2.cuda()
        return (Variable(vars), Variable(var2))

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers, input_size = hidden_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.linear = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, output_size, num_layers, batch_first=True, dropout=0.5)
        self.lstms = nn.ModuleList([nn.LSTMCell(input_size=input_size, hidden_size=output_size) for _ in range(num_layers)])
        self.sigmoid = nn.Sigmoid()

    def forward(self, encoded_input):
        #decoded_output, hidden = self.lstm(encoded_input)
        #decoded_output = self.sigmoid(decoded_output)
        ret = []
        for char in range(encoded_input.size()[0]):
            hiddens, hiddens2 = self.initHidden()
            for count, i in enumerate(self.lstms):
                hiddens, hiddens2 = i(encoded_input[char], (hiddens, hiddens2))
            ret.append(hiddens)
            '''if self.is_cell:
                x = x[0][len(x[0]) - 1] # Output of LSTM: (Prediction for 2nd Char, Prediction for 3rd Char, ...) Only want the one for the next char after the end
                x = self.linear(x)'''
        decoded_output = torch.stack(ret)
        return decoded_output

    def initHidden(self):
        vars = torch.zeros(1, self.output_size)
        var2 = torch.zeros(1, self.output_size)
        if is_cuda:
            vars = vars.cuda()
            var2 = var2.cuda()
        return (Variable(vars), Variable(var2))


#use if getting only lstm output of LAST state. Then only produce the next char
class DecoderCell(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers, input_size = hidden_size):
        super(DecoderCell, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.linear = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, output_size, num_layers, batch_first=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, encoded_input):
        encoded_input = encoded_input.view(1,1, -1)
        decoded_output, hidden = self.lstm(encoded_input)
        #decoded_output = self.sigmoid(decoded_output)
        return decoded_output

'''later needed for generation of payloads'''
class LSTMAutoEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, model_path = None, is_cell = is_cell):
        super(LSTMAutoEncoder, self).__init__()
        self.encoder = EncoderRNN(input_size, hidden_size, num_layers, is_cell = is_cell)
        self.decoder = DecoderRNN(hidden_size, input_size, num_layers)
        if is_cell:
            self.decoder = DecoderCell(hidden_size, input_size, num_layers) #TODO change back here?
        if is_cuda:
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()
        self.path = model_path
        self.hidden_size = hidden_size

    def forward(self, input, length):
        input = self.encoder(input)
        decoded_output = self.decoder(input)#, input.size()[1])
        decoded_output = nn.Dropout()(decoded_output)
        decoded_output = nn.LogSoftmax(dim=2)(decoded_output)
        return decoded_output

    def save_model(self):
        if self.train and self.path is not None:
            torch.save(self, self.path)

    def train_net_cell(self, lr=0.1):
        running_loss = 0.0
        criterion = nn.BCELoss()
        losses = []
        optimizer = optim.Adam(self.parameters(), lr)
        list = util.get_all_targets()
        random.shuffle(list)
        for i, item in enumerate(list):
            start = time.time()
            out = ""
            for i in range(1, len(item)):
                target_tensor = util.example_to_tensor((item + util.get_EOS_token())[i+1])
                #input_tensor = torch.zeros(target_tensor.size()[0], target_tensor.size()[1], self.hidden_size)
                input_tensor = util.example_to_tensor(item[0:i]) #[item[0] for _ in range(len(item))]
                if is_cuda:
                    target_tensor = target_tensor.cuda()
                    input_tensor = input_tensor.cuda()
                input_var =  Variable(input_tensor)
                target_var = Variable(target_tensor)
                optimizer.zero_grad()
                output = self(input_var, len(item))
                loss = criterion(output, target_var)
                out += util.tensor_to_string(output.data)
                #print("output: " + util.tensor_to_string(output.data))
                #print("Target: " + util.tensor_to_string(target_var.data))
                #print("input: " + util.tensor_to_string(input_var.data))
                loss.backward()
                torch.nn.utils.clip_grad_norm(self.parameters(), 1)
                optimizer.step()
                # print statistics
                running_loss += loss.data[0]
                #print('[%5d] loss: %.3f' % (i + 1, loss.data[0]))
            print("cell complete output: " + out)
            #losses.append(running_loss)
            #running_loss = 0.0
            self.save_model()
            #print("Epoch took " + str(time.time() - start) + " to complete")
        return running_loss/len(list)

    def train_net(self, lr=0.5):
        running_loss = 0.0
        criterion = nn.NLLLoss()
        losses = []
        optimizer = optim.Adam(self.parameters(), lr)
        list = util.get_all_targets()
        random.shuffle(list)
        for i, item in enumerate(list):
            start = time.time()
            target_tensor = targetToTensor(item)
            input_tensor = passwordToTensor(item)
            if is_cuda:
                target_tensor = target_tensor.cuda()
                input_tensor = input_tensor.cuda()
            input_var =  Variable(input_tensor)
            target_var = Variable(target_tensor)
            output = self(input_var, len(item))
            print("output: " + tensor_to_string(output.data))
            #print("Target: " + tensor_to_string(target_var.data))
            #print("input: " + tensor_to_string(input_var.data))
            #optimizer.zero_grad()
            self.zero_grad()
            loss = 0
            for i, out in enumerate(output):
                loss += criterion(out, target_var[i])
            #print(output)
            #exit()
            #for i, item in enumerate(output[0]):
            #    loss += criterion(item, target_var[0][i])
            loss.backward()
            for p in self.parameters():
                if p.grad is not None:
                    #print(p.grad.data)
                    p.data.add_(-lr, p.grad.data)
            #optimizer.step()

            # print statistics
            running_loss += loss.data[0]
            #print('[%5d] loss: %.3f' % (i + 1, loss.data[0]))
            #losses.append(running_loss)
            #running_loss = 0.0
            self.save_model()
            #print("Epoch took " + str(time.time() - start) + " to complete")
        return running_loss/len(list)

    def sample(self, length):
        input_tensor = torch.randn(1, length, self.hidden_size)
        if is_cuda:
            input_tensor = input_tensor.cuda()
        input_var = Variable(input_tensor)
        output = self(input_var)
        return tensor_to_string(output.data)


def load_aa_from_file(model_path, hidden_size=hidden_size, num_layers=num_layers):
    aa = LSTMAutoEncoder(letters_num, hidden_size, num_layers, model_path)
    if model_path is not None and os.path.isfile(model_path):
        aa = torch.load(model_path)
    return aa

def train(aa):
    losses = []
    epochs = 1000
    for i in range(epochs):
        start = time.time()
        #print("###############################")
        #print("########    EPOCH " + str(i) + "    ########")
        #print("###############################")
        if is_cell:
            loss = aa.train_net_cell()
        else:
            loss = aa.train_net()
        losses.append(loss)
        #print("Epoch took " + str(time.time()-start) + "s to complete")
        print('[%4d] loss: %.3f' % (i + 1, loss))
    aa.sample(7)
    f = open("losses_seq2seq_v0.04.txt", "w")
    json.dump(losses, f, indent=2)

if __name__ == "__main__":
    model_path = None #"seq2seq_model_v0.01.pt"
    aa = load_aa_from_file(model_path)
    if is_cuda:
        aa = aa.cuda()
    train(aa)