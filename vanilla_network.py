import csv
from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import Levenshtein
import math
import pickle
import numpy as np
import copy
import sys
import wandb

start_token = 0
end_token = 1

class Encoder(nn.Module):
  def __init__(self, inp_vocab_size, embedding_size, n_layers, hl_size, dropout, cell_type, bidirectional):
    super(Encoder, self).__init__()
    self.vocab_size = inp_vocab_size
    self.embedding_size = embedding_size
    self.n_layers = n_layers
    self.hl_size = hl_size
    self.bidirectional = bidirectional
    self.cell_type = cell_type
    self.dropout = dropout

    if cell_type == 'RNN':
      self.cell = nn.RNN(self.embedding_size, self.hl_size, num_layers = self.n_layers, dropout = self.dropout, bidirectional = self.bidirectional).to(device)
    elif cell_type == 'GRU':
      self.cell = nn.GRU(self.embedding_size, self.hl_size, num_layers = self.n_layers, dropout = self.dropout, bidirectional = self.bidirectional).to(device)
    elif cell_type == 'LSTM':
      self.cell = nn.LSTM(self.embedding_size, self.hl_size, num_layers = self.n_layers, dropout = self.dropout, bidirectional = self.bidirectional).to(device)
    else:
      print('Wrong Cell Type.')
      exit()
    self.embedding_layer = nn.Embedding(self.vocab_size, self.embedding_size).to(device)
  
  def forward(self, input, hidden, c = 0):
    embedded = self.embedding_layer(input).view(1, 1, -1)
    if self.cell_type != 'LSTM':
      output, hidden = self.cell(embedded, hidden)
      
      return output, hidden

    else:
      output, (hidden, c) = self.cell(embedded, (hidden, c))

      return output, hidden, c
  
  def init_hidden(self):
    if self.bidirectional:
      return torch.zeros(self.n_layers*2, 1, self.hl_size, device = device)
    else:
      return torch.zeros(self.n_layers, 1, self.hl_size, device = device)

class DecoderVanilla(nn.Module):
  def __init__(self, out_vocab_size, embedding_size, n_layers, hl_size, dropout, cell_type, bidirectional):
    super(DecoderVanilla, self).__init__()
    self.vocab_size = out_vocab_size
    self.embedding_size = embedding_size
    self.n_layers = n_layers
    self.hl_size = hl_size
    self.softmax = nn.LogSoftmax(dim=1)
    self.cell_type = cell_type
    self.bidirectional = bidirectional
    self.dropout = dropout

    if self.bidirectional:
      self.linear = nn.Linear(2*self.hl_size, self.vocab_size).to(device)
    else:
      self.linear = nn.Linear(self.hl_size, self.vocab_size).to(device)

    if cell_type == 'RNN':
      self.cell = nn.RNN(self.embedding_size, self.hl_size, num_layers = self.n_layers, dropout = self.dropout, bidirectional = self.bidirectional).to(device)
    elif cell_type == 'GRU':
      self.cell = nn.GRU(self.embedding_size, self.hl_size, num_layers = self.n_layers, dropout = self.dropout, bidirectional = self.bidirectional).to(device)
    elif cell_type == 'LSTM':
      self.cell = nn.LSTM(self.embedding_size, self.hl_size, num_layers = self.n_layers, dropout = self.dropout, bidirectional = self.bidirectional).to(device)
    else:
      print('Wrong Cell Type.')
      exit()
    
    self.embedding_layer = nn.Embedding(self.vocab_size, self.embedding_size).to(device)
  
  def forward(self, input, hidden, c = 0):
    embedded = self.embedding_layer(input).view(1, 1, -1)
    output = F.relu(embedded)

    if self.cell_type != 'LSTM':
      output, hidden = self.cell(output, hidden)
      output = self.linear(output[0])
      output = self.softmax(output)
      return output, hidden
    else:
      output, (hidden, c) = self.cell(output, (hidden, c))
      output = self.linear(output[0])
      output = self.softmax(output)
      return output, hidden, c

class seq2seq_vanilla():
  def __init__(self, inp_language, out_language, embedding_size, n_layers, hl_size, dropout = 0.2, cell_type = 'LSTM', lr = 0.01, teacher_forcing_ratio = 0.5,bidirectional_flag = False):
    self.encoder = Encoder(inp_language.n_chars, embedding_size, n_layers, hl_size, dropout, cell_type, bidirectional = bidirectional_flag)
    self.decoder = DecoderVanilla(out_language.n_chars, embedding_size, n_layers, hl_size, dropout, cell_type, bidirectional = bidirectional_flag)
    self.lr = lr
    self.teacher_forcing = teacher_forcing_ratio
    self.max_length = out_language.max_size
    self.cell_type = cell_type

    self.encoder_optimizer = optim.SGD(self.encoder.parameters(), lr=self.lr)
    self.decoder_optimizer = optim.SGD(self.decoder.parameters(), lr = self.lr)

    self.loss_fn = nn.NLLLoss()

  def train_step(self, input, target):
    encoder_hidden = self.encoder.init_hidden()
    encoder_c = self.encoder.init_hidden()

    self.encoder_optimizer.zero_grad()
    self.decoder_optimizer.zero_grad()

    input_length = input.size(0)
    target_length = target.size(0)

    loss = 0

    for i in range(0, input_length):
      if self.cell_type != 'LSTM':
        encoder_output, encoder_hidden = self.encoder.forward(input[i], encoder_hidden)
      else:
        encoder_output, encoder_hidden, encoder_c = self.encoder.forward(input[i], encoder_hidden, encoder_c)
    
    decoder_input = torch.tensor([[start_token]], device=device)

    decoder_hidden = encoder_hidden
    decoder_c = encoder_c

    num = random.random()

    if num < self.teacher_forcing:
      #here, we use teacher forcing. 
      for j in range(0, target_length):
        if self.cell_type != 'LSTM':
          decoder_output, decoder_hidden = self.decoder.forward(decoder_input, decoder_hidden)
        else:
          decoder_output, decoder_hidden, decoder_c = self.decoder.forward(decoder_input, decoder_hidden, decoder_c)

        loss = loss + self.loss_fn(decoder_output, target[j])
        decoder_input = target[j]#.unsqueeze(0)

    else:
      #here, there is no teacher forcing. the predictions themselves are used. 
      for j in range(0, target_length):
        if self.cell_type != 'LSTM':
          decoder_output, decoder_hidden = self.decoder.forward(decoder_input, decoder_hidden)
        else:
          decoder_output, decoder_hidden, decoder_c = self.decoder.forward(decoder_input, decoder_hidden, decoder_c)

        loss = loss + self.loss_fn(decoder_output, target[j])
        value, index = decoder_output.topk(1)
        decoder_input = index.squeeze().detach()
        if decoder_input.item() == end_token:
          break
  
    loss.backward()
    self.encoder_optimizer.step()
    self.decoder_optimizer.step()

    return loss.item()/target_length
  
  def predict(self, input, target):
    #here, we use the model to inference. 
    with torch.no_grad():
      encoder_hidden = self.encoder.init_hidden()
      encoder_c = self.encoder.init_hidden()

      input_length = input.size(0)
      for i in range(0, input_length):
        if self.cell_type != 'LSTM':
          encoder_output, encoder_hidden = self.encoder.forward(input[i], encoder_hidden)
        else:
          encoder_output, encoder_hidden, encoder_c = self.encoder.forward(input[i], encoder_hidden, encoder_c)

      decoder_input = torch.tensor([[start_token]], device=device)
      decoder_hidden = encoder_hidden
      decoder_c = encoder_c

      outputs = []
      for i in range(0, self.max_length):
        if self.cell_type != 'LSTM':
          decoder_output, decoder_hidden = self.decoder.forward(decoder_input, decoder_hidden)
        else:
          decoder_output, decoder_hidden, decoder_c = self.decoder.forward(decoder_input, decoder_hidden, decoder_c)

        value, index = decoder_output.data.topk(1)
        decoder_input = index.squeeze().detach()
        outputs.append(decoder_input.item())
        if decoder_input.item() == end_token:
          break

      return outputs
  
  def predict_beam(self, input, beam_size):
    #this is used to predict b outputs, where b is the beam size. 
    with torch.no_grad():
      encoder_hidden = self.encoder.init_hidden()
      encoder_c = self.encoder.init_hidden()

      input_length = input.size(0)
      for i in range(0, input_length):
        if self.cell_type != 'LSTM':
          encoder_output, encoder_hidden = self.encoder.forward(input[i], encoder_hidden)
        else:
          encoder_output, encoder_hidden, encoder_c = self.encoder.forward(input[i], encoder_hidden, encoder_c)

      decoder_input = torch.tensor([[start_token]], device=device)
      decoder_hidden = encoder_hidden
      decoder_c = encoder_c

      possible_outputs = [] #a list of lists containing the top "beam_size" set of the best possible outputs at an instance
      next_inputs = [] #list containing the tensors that are to be fed as the next input
      decoder_hiddens = [] #list containing the hidden values corresponding to the given outputs
      decoder_cs = [] #list containing the cell states (in case we use an LSTM)

      if self.cell_type != 'LSTM':
        decoder_output, decoder_hidden = self.decoder.forward(decoder_input, decoder_hidden)
      else:
        decoder_output, decoder_hidden, decoder_c = self.decoder.forward(decoder_input, decoder_hidden, decoder_c)
        
      values, indices = decoder_output.data.topk(beam_size) #we unpack the top b values and their corresponding indices
      values = torch.exp(values) #we would like to deal with a probability instead of a log-probability, so we do e^val

      for j in range(0, beam_size):
        #we do on iteration and populate the above lists. we will then use these lists to iteratively get our best b guesses. 
        value = values[0, j]
        index = indices[0, j]
        possible_outputs.append([index.item()])
        next_inputs.append(index.squeeze().detach())
        decoder_hiddens.append(decoder_hidden)
        if self.cell_type == 'LSTM':
          decoder_cs.append(decoder_c)
                    
      for k in range(1, self.max_length):
        #now, we make predictions for the other timesteps
        temp_probabilities = np.zeros([beam_size, beam_size])
        temp_indices = []
        temp_hiddens = []
        temp_cs = []

        for i in range(0, beam_size):
          #we first iterate through all possible current inputs and their corresponding hidden states
          curr_input = next_inputs[i]
          hidden = decoder_hiddens[i]
          if self.cell_type == 'LSTM':
            c = decoder_cs[i]
          
          if self.cell_type != 'LSTM':
            decoder_output, decoder_hidden = self.decoder.forward(curr_input, hidden)
          else:
            decoder_output, decoder_hidden, decoder_c = self.decoder.forward(curr_input, hidden, c)
            temp_cs.append(decoder_c)
            
          #following this, we get the top b values for each of these inputs
          temp_hiddens.append(decoder_hidden)
          values, indices = decoder_output.data.topk(beam_size)
          values = torch.exp(values)

          temp_temp_indices = []
          for j in range(0, beam_size):
            #we now iterate through these b predictions correponding to a given input, and construct a probability table
            #this probability table gives the conditional probability of the output, given the input sequence. 
            value = values[0, j]
            index = indices[0, j]
            temp_probabilities[i][j] = value.item()
            temp_temp_indices.append(index.squeeze().detach())

            if curr_input.item() == end_token:
              temp_temp_indices.append(torch.tensor([[end_token]], device= device))
          
          temp_indices.append(temp_temp_indices)
        
        #we then choose the best b probabilities from these
        mat = np.array(temp_probabilities)
        idx = np.argpartition(mat, mat.size - beam_size, axis=None)[-beam_size:]
        results = np.column_stack(np.unravel_index(idx, mat.shape)) 

        updated_outputs = []

        decoder_hiddens = []
        decoder_cs = []
        next_inputs = []

        for i, result in enumerate(results):
          #we finally iterate through these best sequences and update our next_inputs, hiddens, and our possible_outputs. 
          x = result[0]
          y = result[1]
          next_inputs.append(temp_indices[x][y])
          decoder_hiddens.append(temp_hiddens[x])
          if self.cell_type == 'LSTM':
            decoder_cs.append(temp_cs[x])

          arr = copy.deepcopy(possible_outputs[x])
          if arr[-1] == end_token:
            pass
          else:
            arr.append(temp_indices[x][y].item())

          updated_outputs.append(arr)
        
        possible_outputs = copy.deepcopy(updated_outputs)

      #we finally return a set of predictions instead of one prediction.             
      return possible_outputs

  def evaluate_beam(self, data, beam_size, print_flag = True):
    #same as the evaluate fn, except we check for the multiple outputs generated 
    #by the beam based predict method, and choose the best one to get a good performance. 
    correct = 0
    character_wise = 0
    count = 0
    total_distance = 0

    for pair in data:
      input = pair[0]
      target = pair[1]
      predictions = self.predict_beam(input, beam_size)
      target = target.tolist()
      target = [t[0] for t in target]
      tar_word = decoded_word(tamil,target)

      mini = 2

      for pred in predictions:
        pred_word = decoded_word(tamil,pred)
        if pred_word == tar_word:
          correct = correct + 1
        
        dist = min((Levenshtein.distance(pred_word, tar_word)/max(len(tar_word),len(pred_word))), 1)
        if dist < mini:
          mini = dist
      
      if print_flag:
        if count%500 == 0:
          print([decoded_word(tamil, pred) for pred in predictions], tar_word)
        count = count + 1
      
      total_distance = total_distance + mini
    
    avg_distance = total_distance/len(data)
    char_acc = 1 - avg_distance
    acc = correct/len(data)
    return acc, char_acc


  def evaluate(self, data, print_flag = True):
    correct = 0
    character_wise = 0
    count = 0
    total_distance = 0

    for pair in data:
      input = pair[0]
      target = pair[1]
      pred = self.predict(input, target)
      target = target.tolist()
      target = [t[0] for t in target]

      if print_flag:
        if count%500 == 0:
          print(decoded_word(tamil,pred), decoded_word(tamil,target))
        count = count + 1

      pred_word = decoded_word(tamil,pred)
      tar_word = decoded_word(tamil,target)
      
      if pred_word == tar_word:
        correct = correct + 1
      
      total_distance = total_distance + min((Levenshtein.distance(pred_word, tar_word)/max(len(tar_word),len(pred_word))), 1)
    
    avg_distance = total_distance/len(data)
    char_acc = 1 - avg_distance
    acc = correct/len(data)

    return acc, char_acc