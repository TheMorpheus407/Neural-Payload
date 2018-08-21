import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import util
import os
import random
import requests


'''
NOT LEARNING! SEQ2SEQ Model using LSTMCELL
SEE VERSION 3 FOR A WORKING NETWORK.
'''

is_cuda = False
FloatTensor = torch.cuda.FloatTensor if is_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if is_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if is_cuda else torch.ByteTensor
Tensor = FloatTensor

hidden_size = 32
num_layers = 2
is_cell = True

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size = hidden_size, dropout = 0.3, is_cell = False):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        if is_cell:
            self.lstm = nn.LSTM(util.get_letters_num(), hidden_size, num_layers, dropout=dropout, batch_first=False)
        else:
            self.lstm = nn.LSTM(util.get_letters_num(), hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.is_cell = is_cell


    def forward(self, x):
        x = self.lstm(x)
        x = x[0]  # Dont care about hidden states
        if self.is_cell:
            x = x[0][len(x[0]) - 1] # Output of LSTM: (Prediction for 2nd Char, Prediction for 3rd Char, ...) Only want the one for the next char after the end
            x = self.linear(x)
        return self.relu(x)

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers, input_size = hidden_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.linear = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, output_size, num_layers, batch_first=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, encoded_input):
        decoded_output, hidden = self.lstm(encoded_input)
        decoded_output = self.sigmoid(decoded_output)
        return decoded_output

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
    def __init__(self, input_size, hidden_size, num_layers, model_path = None, train_model = True, is_cell = is_cell):
        super(LSTMAutoEncoder, self).__init__()
        self.encoder = EncoderRNN(input_size, hidden_size, num_layers, is_cell = is_cell)
        self.decoder = DecoderRNN(hidden_size, input_size, num_layers)
        if is_cell:
            self.decoder = DecoderCell(hidden_size, input_size, num_layers) #TODO change back here?
        if is_cuda:
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()
        self.path = model_path
        self.train_model = train_model
        self.hidden_size = hidden_size

    def forward(self, input, length):
        input = self.encoder(input)
        decoded_output = self.decoder(input)#, input.size()[1])
        decoded_output = nn.Softmax(dim=2)(decoded_output)
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

    def train_net(self, lr=0.1):
        running_loss = 0.0
        criterion = nn.BCEWithLogitsLoss()
        losses = []
        optimizer = optim.Adam(self.parameters(), lr)
        list = util.get_all_targets()
        random.shuffle(list)
        for i, item in enumerate(list):
            start = time.time()
            target_tensor = util.example_to_tensor(item)
            #input_tensor = torch.zeros(target_tensor.size()[0], target_tensor.size()[1], self.hidden_size)
            input_tensor = util.example_to_tensor(item) #[item[0] for _ in range(len(item))]
            if is_cuda:
                target_tensor = target_tensor.cuda()
                input_tensor = input_tensor.cuda()
            input_var =  Variable(input_tensor)
            target_var = Variable(target_tensor)
            output = self(input_var, len(item))
            print("output: " + util.tensor_to_string(output.data))
            print("Target: " + util.tensor_to_string(target_var.data))
            print("input: " + util.tensor_to_string(input_var.data))
            loss = 0
            optimizer.zero_grad()
            for i, item in enumerate(output[0]):
                loss += criterion(item, target_var[0][i])
            loss.backward()
            torch.nn.utils.clip_grad_norm(self.parameters(), 1)
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]
            print('[%5d] loss: %.3f' % (i + 1, loss.data[0]))
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
        return util.tensor_to_string(output.data)


def load_aa_from_file(model_path):
    aa = LSTMAutoEncoder(util.get_letters_num(), hidden_size, num_layers, model_path)
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
    f = open("losses_seq2seq_v0.01.txt", "w")
    json.dump(losses, f, indent=2)

if __name__ == "__main__":
    model_path = None #"seq2seq_model_v0.01.pt"
    aa = load_aa_from_file(model_path)
    if is_cuda:
        aa = aa.cuda()
    train(aa)