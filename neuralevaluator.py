#################################################
#
# This model prints out not IF the website is vulnerable, but WHERE it is e.g. the line of code
# Does NOT work well - indeed it only outputs zeroes.
# See neuralevaluator_cnn for a more basic but working alternative
# feel free to extend or improve here
#
#################################################

import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import util
import os
import requests

is_cuda = True
FloatTensor = torch.cuda.FloatTensor if is_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if is_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if is_cuda else torch.ByteTensor
Tensor = FloatTensor

hidden_size = 512
num_layers = 8

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size = hidden_size, dropout = 0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(util.get_letters_num(), hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.lstm(x)
        x = x[0]  # Dont care about hidden states
        #x = x[0][len(x[0]) - 1]  # Output of LSTM: (Prediction for 2nd Char, Prediction for 3rd Char, ...) Only want the one for the next char after the end
        x = self.linear(x)
        return self.relu(x)

'''later needed for generation of payloads'''
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers, input_size = hidden_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        #self.lstm = nn.LSTMCell(hidden_size, util.get_letters_num(), num_layers)
        self.linear = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, output_size, num_layers, batch_first=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, encoded_input):#, output_length):
        '''hx_tensor = torch.zeros(1, util.get_letters_num())
        cx_tensor = torch.zeros(1, util.get_letters_num())
        if is_cuda:
            hx_tensor = hx_tensor.cuda()
            cx_tensor = cx_tensor.cuda()
        hx = Variable(hx_tensor)
        cx = Variable(cx_tensor)
        output = []
        for i in range(output_length):
            hx,cx = self.lstm(encoded_input, (hx, cx))
            hx = self.sigmoid(hx)
            output.append(hx)
        return torch.cat(output)'''
        decoded_output, hidden = self.lstm(encoded_input)
        decoded_output = self.sigmoid(decoded_output)
        return decoded_output

'''later needed for generation of payloads'''
class LSTMAutoEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, model_path = None, train_model = True):
        super(LSTMAutoEncoder, self).__init__()
        self.encoder = EncoderRNN(input_size, hidden_size, num_layers)
        self.decoder = DecoderRNN(hidden_size, input_size, num_layers)
        if is_cuda:
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()
        self.path = model_path
        self.train_model = train_model

    def forward(self, input):
        encoded_input = self.encoder(input)
        decoded_output = self.decoder(encoded_input)#, input.size()[1])
        return decoded_output

    def save_model(self):
        if self.train and self.path is not None:
            torch.save(self, self.filename)

    def train_net(self, lr=0.001):
        running_loss = 0.0
        criterion = nn.BCEWithLogitsLoss()
        losses = []
        optimizer = optim.Adam(self.parameters(), lr)
        for i, item in enumerate(util.get_html()):
            start = time.time()
            input_tensor = util.example_to_tensor((item[0]+item[1])[:2])
            if is_cuda:
                input_tensor = input_tensor.cuda()
            input_var =  Variable(input_tensor)
            output = self(input_var)
            optimizer.zero_grad()
            #exit()
            loss = criterion(output, input_var)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%5d] loss: %.3f' % (i + 1, running_loss / 2000))
                losses.append(running_loss/2000)
                running_loss = 0.0
                self.save_model()
            print("Epoch took " + str(time.time() - start) + " to complete")
        return losses

'''
always queried - prints out WHERE!!! its vuln., if there is none, the website is safe
'''
class NeuralEvaluatorModel(nn.Module):
    def __init__(self, hidden_size = hidden_size, num_layers = num_layers, model_path = "neural_evaluator_model_v0.01.pt", train_model = True, intermediate_size=8):
        super(NeuralEvaluatorModel, self).__init__()
        self.website_encoder = EncoderRNN(util.get_letters_num(), hidden_size, num_layers, output_size=intermediate_size)
        self.payload_encoder = EncoderRNN(util.get_letters_num(), hidden_size, num_layers, output_size=intermediate_size)
        self.linear = nn.Linear(in_features=intermediate_size*2, out_features=1)
        self.sigmoid = nn.Sigmoid()
        if is_cuda:
            self.website_encoder = self.website_encoder.cuda()
            self.payload_encoder = self.payload_encoder.cuda()
        self.path = model_path
        self.train_model = train_model

    def forward(self, website, payload):
        website = self.website_encoder(website) #Size: (1,length,8)
        payload = self.payload_encoder(payload)[0]
        payload = payload[len(payload)-1].view(1,1,-1) #Size: (1,1,8)
        x_payload = torch.cat([payload for _ in range(website.size()[1])], 1) #Size: (1,length,8)
        x = torch.cat((website, x_payload), 2) #Size: (1,length,16)
        return self.sigmoid(self.linear(x)) #Size: (1,length,1)

    def save_model(self):
        if self.train and self.path is not None:
            torch.save(self, self.path)

    def train_net(self, lr=0.001):
        running_loss = 0.0
        criterion = nn.BCELoss()
        losses = []
        optimizer = optim.Adam(self.parameters(), lr)
        for i, (payload, website, target) in enumerate(util.get_website_attacks()): #TODO FULL payload (DICT!!), website: (header, text), target:(start idx von vuln., länge)
            if payload is None or website is None or target is None:
                continue
            start = time.time()
            payload = str(payload)
            website_tensor = util.example_to_tensor((website[0]+website[1]))
            payload_tensor = util.example_to_tensor(payload)
            target = util.generate_target_vuln(website[0], website[1], target)
            if is_cuda:
                website_tensor = website_tensor.cuda()
                payload_tensor = payload_tensor.cuda()
                target = target.cuda()
            website_var = Variable(website_tensor)
            payload_var = Variable(payload_tensor)
            output = self(website_var, payload_var) #Size: (1,length,1)
            optimizer.zero_grad()
            loss = criterion(output, Variable(target))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]
            print('[%5d] loss: %.3f' % (i + 1, loss.data[0]))
            losses.append(loss.data[0])
            self.save_model()
            #print("Epoch took " + str(time.time() - start) + "s to complete")
        return losses

class NeuralEvaluator():
    def __init__(self, base_injection_dict, url, evaluator = None):
        if evaluator == None:
            self.evaluator = NeuralEvaluatorModel()
        else:
            self.evaluator = evaluator
        if is_cuda:
            self.evaluator = self.evaluator.cuda()
        #self.callbacks = []
        self.url = url
        self.base_injection_dict = base_injection_dict


    #def register_callback_function(self, func):
        #self.callbacks.append(func)

    #def delete_callback_function(self, func):
        #self.callbacks.remove(func)

    def __call__(self, *args, **kwargs):
        if kwargs["target"] == "post":
            to_inject = util.payloaddict_to_string(self.base_injection_dict, args)
            resp = requests.post(self.url, data=to_inject)
            print("SENDING REQUEST TO " + self.url + " WITH PARAMS " + str(to_inject))
            headers = resp.headers
            if "date" in headers:
                headers.pop("date")
            if resp.status_code != 200:
                return False
            self.predict(resp, to_inject)
            return resp
        return False

    def predict(self, resp, to_inject):
        website = util.prepare_headers(resp.headers) + resp.text
        payload = str(to_inject)
        website_tensor = util.example_to_tensor(website)
        payload_tensor = util.example_to_tensor(payload)
        if is_cuda:
            website_tensor = website_tensor.cuda()
            payload_tensor = payload_tensor.cuda()
        website_var = Variable(website_tensor)
        payload_var = Variable(payload_tensor)
        output = self.evaluator(website_var, payload_var) #Size: (1,length,1)
        return output


def load_neuralevalmodel_from_file(model_path, hidden_size=hidden_size, num_layers = num_layers):
    aa = NeuralEvaluatorModel(hidden_size=hidden_size, num_layers = num_layers, model_path = model_path)
    if model_path is not None and os.path.isfile(model_path):
        aa = torch.load(model_path)
    return aa

def train(losses_path="losses_neural_evaluator.txt", model_path = "neural_evaluator_model.pt"):
    epochs = 10
    aa = load_neuralevalmodel_from_file(model_path)
    if is_cuda:
        aa = aa.cuda()
    losses = []
    for i in range(epochs):
        start = time.time()
        print("###############################")
        print("########    EPOCH " + str(i) + "    ########")
        print("###############################")
        losses.append(aa.train_net())
        print("Epoch took " + str(time.time()-start) + " s to complete")
    f = open(losses_path, "w")
    json.dump(losses, f, indent=2)

def predict():
    dict = {"message": "ZAP"}
    model = NeuralEvaluator(base_injection_dict=dict, url="http://localhost/Masterarbeit/xss.php")
    res = model("<script>alert('iCConsult')</script>", target="post").data.numpy()
    return res

if __name__ == "__main__":
    # Train the AA!
    train()
    #predict()


