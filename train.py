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

def obtain_data(dir):
  x = []
  y = []

  with open(dir, 'r') as file:
    reader = csv.DictReader(file, fieldnames=['x', 'y'])
  
    for row in reader:
      x.append(row['x'])
      y.append(row['y'])
    
  return x, y

#First we define a Language class, which will be used to easily load our dataset.
class Language:
  def __init__(self, name):
    self.name = name
    self.char2index = {}
    self.index2char = {0: "SOS", 1: "EOS", 2: "unknown"} #mapping from index to character
    self.n_chars = 3  # Count SOS, EOS, and unknown. 
    self.max_size = 2 #to find the maximum length of the dataset we're training our model on

  def update_vocab(self, x):
    #this function creates the vocabulary using the data we feed it. 
    for word in x:
      if len(word) + 2 > self.max_size:
        self.max_size = len(word) + 2

      for letter in word:
        if letter not in self.char2index.keys():
          self.char2index[letter] = self.n_chars
          self.index2char[self.n_chars] = letter
          self.n_chars = self.n_chars + 1
  
  def get_index(self, character):
    #given an index, this function returns the corresponding character.
    if character in self.char2index.keys():
      return self.char2index[character]
    else:
      return 2
  
  def get_character(self, index):
    #given a character, this function returns the corresponding index
    if index == 2:
      return '$'
    else:
      return self.index2char[index]

def encoded_word(language, word):
  #this function takes a language and a word, and one-hot encodes it for our model. 
  coded = [language.get_index(letter) for letter in word]
  coded.append(end_token)
  return coded

def get_pairs(lang1, lang2, inputs, targets):
  #this function takes 2 languages, and the data (inputs and targets), encodes them, and returns them as a list of tensor tuples; ready to be fed directly into the model
  return [(torch.tensor(encoded_word(lang1, x), dtype=torch.long, device=device).view(-1, 1), torch.tensor(encoded_word(lang2, y), dtype=torch.long, device=device).view(-1, 1)) 
  for (x,y) in zip(inputs,targets)]

def decoded_word(language, encoded_word):
  #given an encoded word (an array of indice) and the language object, we return the decoded word. 
  if encoded_word[-1] == end_token:
    encoded_word = encoded_word[:-1]

  characters = [language.get_character(num) for num in encoded_word]
  decoded = ''.join(characters)
  return decoded

#Now, we define the class for the Encoder.
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

#Now, we define the vanilla decoder. We'll use the above encoder and the decoder to form our vanilla seq2seq model. 
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

#Now that we have our encoder and decoder, let us define the class for the seq2seq vanilla model. 
class seq2seq_vanilla():
  def __init__(self, inp_language, out_language, embedding_size = 128, n_layers = 3, hl_size = 128, decay_rate = 0, dropout = 0.2, cell_type = 'LSTM', lr = 0.01, teacher_forcing_ratio = 0.5,bidirectional_flag = False):
    self.encoder = Encoder(inp_language.n_chars, embedding_size, n_layers, hl_size, dropout, cell_type, bidirectional = bidirectional_flag)
    self.decoder = DecoderVanilla(out_language.n_chars, embedding_size, n_layers, hl_size, dropout, cell_type, bidirectional = bidirectional_flag)
    self.lr = lr
    self.teacher_forcing = teacher_forcing_ratio
    self.max_length = out_language.max_size
    self.cell_type = cell_type
    self.inp_lang = inp_language
    self.out_lang = out_language

    self.decay_rate = decay_rate
    self.encoder_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(self.encoder_optimizer, lr_lambda=self.decay)
    self.decoder_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(self.decoder_optimizer, lr_lambda=self.decay)

    self.encoder_optimizer = optim.SGD(self.encoder.parameters(), lr=self.lr)
    self.decoder_optimizer = optim.SGD(self.decoder.parameters(), lr = self.lr)

    self.loss_fn = nn.NLLLoss()
    print('Vanilla Model Initialized...')
  
  def decay(self, epoch):
    return self.decay_rate

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

    self.encoder_scheduler.step()
    self.decoder_scheduler.step()

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
    one_every = len(data)//10

    for pair in data:
      input = pair[0]
      target = pair[1]
      predictions = self.predict_beam(input, beam_size)
      target = target.tolist()
      target = [t[0] for t in target]
      tar_word = decoded_word(self.out_lang,target)

      mini = 2

      for pred in predictions:
        pred_word = decoded_word(self.out_lang,pred)
        if pred_word == tar_word:
          correct = correct + 1
        
        dist = min((Levenshtein.distance(pred_word, tar_word)/max(len(tar_word),len(pred_word))), 1)
        if dist < mini:
          mini = dist
      
      if print_flag:
        if count%one_every == 0:
          print([decoded_word(self.out_lang, pred) for pred in predictions], tar_word)
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
    one_every = len(data)//10

    for pair in data:
      input = pair[0]
      target = pair[1]
      pred = self.predict(input, target)
      target = target.tolist()
      target = [t[0] for t in target]

      if print_flag:
        if count%one_every == 0:
          print(decoded_word(self.out_lang,pred), decoded_word(self.out_lang,target))
        count = count + 1

      pred_word = decoded_word(self.out_lang,pred)
      tar_word = decoded_word(self.out_lang,target)
      
      if pred_word == tar_word:
        correct = correct + 1
      
      total_distance = total_distance + min((Levenshtein.distance(pred_word, tar_word)/max(len(tar_word),len(pred_word))), 1)
    
    avg_distance = total_distance/len(data)
    char_acc = 1 - avg_distance
    acc = correct/len(data)

    return acc, char_acc

#let us now define the decoder with attention class as follows:
class DecoderAttn(nn.Module):
  def __init__(self, out_vocab_size, max_length, embedding_size, hl_size, dropout, cell_type, bidirectional):
    super(DecoderAttn, self).__init__()
    self.vocab_size = out_vocab_size
    self.embedding_size = embedding_size
    self.hl_size = hl_size
    self.softmax = nn.LogSoftmax(dim=1)
    self.cell_type = cell_type
    self.bidirectional = bidirectional
    self.dropout = dropout
    self.max_length = max_length

    self.dropout_layer = nn.Dropout(self.dropout).to(device)


    if self.bidirectional:
      self.linear = nn.Linear(2*self.hl_size, self.vocab_size).to(device)
      self.attention = nn.Linear(2*self.hl_size + self.embedding_size, self.max_length).to(device)
      self.attn_combine = nn.Linear(2*self.hl_size + self.embedding_size, self.embedding_size).to(device)

    else:
      self.linear = nn.Linear(self.hl_size, self.vocab_size).to(device)
      self.attention = nn.Linear(self.hl_size + self.embedding_size, self.max_length).to(device)
      self.attn_combine = nn.Linear(self.hl_size + self.embedding_size, self.embedding_size).to(device)


    if cell_type == 'RNN':
      self.cell = nn.RNN(self.embedding_size, self.hl_size, num_layers = 1, bidirectional = self.bidirectional).to(device)
    elif cell_type == 'GRU':
      self.cell = nn.GRU(self.embedding_size, self.hl_size, num_layers = 1, bidirectional = self.bidirectional).to(device)
    elif cell_type == 'LSTM':
      self.cell = nn.LSTM(self.embedding_size, self.hl_size, num_layers = 1, bidirectional = self.bidirectional).to(device)
    else:
      print('Wrong Cell Type.')
      exit()
    
    self.embedding_layer = nn.Embedding(self.vocab_size, self.embedding_size).to(device)
  
  def forward(self, input, hidden, encoder_outputs, c = 0):
    embedded = self.embedding_layer(input).view(1, 1, -1)
    embedded = self.dropout_layer(embedded)

    if self.bidirectional:
      concatenated = torch.cat((embedded[0], hidden[0], hidden[1]), 1)
    else:
      concatenated = torch.cat((embedded[0], hidden[0]), 1)

    attn_weights = F.softmax(self.attention(concatenated), dim=1)
    attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))

    output = torch.cat((embedded[0], attn_applied[0]), 1)
    output = self.attn_combine(output).unsqueeze(0)

    if self.cell_type != 'LSTM':
      output, hidden = self.cell(output, hidden)
      output = self.linear(output[0])
      output = self.softmax(output)
      return output, hidden, attn_weights
    else:
      output, (hidden, c) = self.cell(output, (hidden, c))
      output = self.linear(output[0])
      output = self.softmax(output)
      return output, hidden, c, attn_weights

#and finally, we put everything together for the seq2seq with attention model. 
class seq2seq_attn():
  def __init__(self, inp_language, out_language, embedding_size = 128, hl_size = 128, decay_rate = 2**(math.log2(0.2)/30000),dropout = 0.2, cell_type = 'GRU', lr = 0.005, teacher_forcing_ratio = 0.5,bidirectional_flag = True):
    self.max_length = max(out_language.max_size,  inp_language.max_size)
    self.inp_lang = inp_language
    self.out_lang = out_language
    self.encoder = Encoder(inp_language.n_chars, embedding_size, 1, hl_size, dropout, cell_type, bidirectional = bidirectional_flag)
    self.decoder = DecoderAttn(out_language.n_chars, self.max_length, embedding_size, hl_size, dropout, cell_type, bidirectional = bidirectional_flag)
    self.lr = lr
    self.teacher_forcing = teacher_forcing_ratio
    self.cell_type = cell_type
    self.bidir = bidirectional_flag

    self.encoder_optimizer = optim.SGD(self.encoder.parameters(), lr=self.lr)
    self.decoder_optimizer = optim.SGD(self.decoder.parameters(), lr = self.lr)

    self.loss_fn = nn.NLLLoss()
    self.decay_rate = decay_rate
    self.encoder_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(self.encoder_optimizer, lr_lambda=self.decay)
    self.decoder_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(self.decoder_optimizer, lr_lambda=self.decay)
    print('Attention Model Initialized...')
  
  def decay(self, epoch):
    return self.decay_rate

  def train_step(self, input, target):
    encoder_hidden = self.encoder.init_hidden()
    encoder_c = self.encoder.init_hidden()

    self.encoder_optimizer.zero_grad()
    self.decoder_optimizer.zero_grad()

    input_length = input.size(0)
    target_length = target.size(0)

    loss = 0

    if self.bidir == True:
      encoder_outputs = torch.zeros(self.max_length, 2*self.encoder.hl_size, device=device)
    else:
      encoder_outputs = torch.zeros(self.max_length, self.encoder.hl_size, device=device)

    for i in range(0, input_length):
      if self.cell_type != 'LSTM':
        encoder_output, encoder_hidden = self.encoder.forward(input[i], encoder_hidden)
        encoder_outputs[i] = encoder_output[0, 0]
      else:
        encoder_output, encoder_hidden, encoder_c = self.encoder.forward(input[i], encoder_hidden, encoder_c)
        encoder_outputs[i] = encoder_output[0, 0]
    
    decoder_input = torch.tensor([[start_token]], device=device)

    decoder_hidden = encoder_hidden
    decoder_c = encoder_c

    num = random.random()

    if num < self.teacher_forcing:
      #here, we use teacher forcing. 
      for j in range(0, target_length):
        if self.cell_type != 'LSTM':
          decoder_output, decoder_hidden, decoder_attention = self.decoder.forward(decoder_input, decoder_hidden, encoder_outputs)
        else:
          decoder_output, decoder_hidden, decoder_c, decoder_attention = self.decoder.forward(decoder_input, decoder_hidden, encoder_outputs, decoder_c)

        loss = loss + self.loss_fn(decoder_output, target[j])
        decoder_input = target[j]#.unsqueeze(0)

    else:
      #here, there is no teacher forcing. the predictions themselves are used. 
      for j in range(0, target_length):
        if self.cell_type != 'LSTM':
          decoder_output, decoder_hidden, decoder_attention = self.decoder.forward(decoder_input, decoder_hidden, encoder_outputs)
        else:
          decoder_output, decoder_hidden, decoder_c, decoder_attention = self.decoder.forward(decoder_input, decoder_hidden, encoder_outputs, decoder_c)

        loss = loss + self.loss_fn(decoder_output, target[j])
        value, index = decoder_output.topk(1)
        decoder_input = index.squeeze().detach()
        if decoder_input.item() == end_token:
          break
      
            
    loss.backward()
    self.encoder_optimizer.step()
    self.decoder_optimizer.step()

    self.encoder_scheduler.step()
    self.decoder_scheduler.step()

    return loss.item()/target_length
  
  def predict(self, input, target):
    with torch.no_grad():
      encoder_hidden = self.encoder.init_hidden()
      encoder_c = self.encoder.init_hidden()

      if self.bidir == True:
        encoder_outputs = torch.zeros(self.max_length, 2*self.encoder.hl_size, device=device)
      else:
        encoder_outputs = torch.zeros(self.max_length, self.encoder.hl_size, device=device)

      input_length = input.size(0)
      for i in range(0, min(input_length, self.max_length)):
        if self.cell_type != 'LSTM':
          encoder_output, encoder_hidden = self.encoder.forward(input[i], encoder_hidden)
          encoder_outputs[i] = encoder_output[0, 0]
        else:
          encoder_output, encoder_hidden, encoder_c = self.encoder.forward(input[i], encoder_hidden, encoder_c)
          encoder_outputs[i] = encoder_output[0, 0]

      decoder_input = torch.tensor([[start_token]], device=device)
      decoder_hidden = encoder_hidden
      decoder_c = encoder_c

      outputs = []
      for i in range(0, self.max_length):
        if self.cell_type != 'LSTM':
          decoder_output, decoder_hidden, decoder_attention = self.decoder.forward(decoder_input, decoder_hidden, encoder_outputs)
        else:
          decoder_output, decoder_hidden, decoder_c, decoder_attention = self.decoder.forward(decoder_input, decoder_hidden, encoder_outputs, decoder_c)

        value, index = decoder_output.data.topk(1)
        decoder_input = index.squeeze().detach()
        outputs.append(decoder_input.item())
        if decoder_input.item() == end_token:
          break

      return outputs


  def evaluate(self, data, print_flag):
    correct = 0
    character_wise = 0
    count = 0
    total_distance = 0

    one_every = len(data)//10

    for pair in data:
      input = pair[0]
      target = pair[1]
      pred = self.predict(input, target)
      target = target.tolist()
      target = [t[0] for t in target]

      if print_flag:
        if count%one_every == 0:
            print(decoded_word(self.out_lang,pred), decoded_word(self.out_lang,target))
        count = count + 1

      pred_word = decoded_word(self.out_lang,pred)
      tar_word = decoded_word(self.out_lang,target)
      
      if pred_word == tar_word:
        correct = correct + 1
      
      total_distance = total_distance + min((Levenshtein.distance(pred_word, tar_word)/max(len(tar_word),len(pred_word))), 1)
    
    avg_distance = total_distance/len(data)
    char_acc = 1 - avg_distance
    acc = correct/len(data)

    return acc, char_acc

def main(argv):
  opts = []
  args = []
  params = {}

  for i in range(0, len(argv)):
    if i%2 == 0:
      opts.append(argv[i])
    else:
      args.append(argv[i])
  
  data_dir = ''
  language_prefix = ''
  n_iters = 75000
  attn_flag = False
  beam_size = 1
  print_flag = True
  n_layers = 1
  
  for opt, arg in zip(opts, args):
    if opt == '-d' or opt == '--dataset':
      data_dir = arg
    elif opt == '-l' or opt == '--language_prefix':
      language_prefix = arg
    elif opt == '-i' or opt == '--n_iterations':
      n_iters = int(arg)
    elif opt == '-a' or opt == '--attention':
      attn_flag = (arg=='y')
    elif opt == '-b' or opt == '--beam_size':
      beam_size = int(arg)
    elif opt == '-e' or opt == '--embedding_size':
      params['embedding_size'] = int(arg)
    elif opt == '-h' or opt == '--hidden_size':
      params['hl_size'] = int(arg)
    elif opt == '-nl' or opt == '--n_layers':
      pn_layers = int(arg)
    elif opt == '-c' or opt == '--cell_type':
      params['cell_type'] = arg
    elif opt == '-lr' or opt == '--learning_rate':
      params['lr'] = float(arg)
    elif opt == '-q' or opt == '--quiet':
      print_flag = (arg=='y')
    elif opt == '-d_rate' or opt == '--decay_rate':
      params['decay_rate'] = float(arg)
    elif opt == '-t' or opt == '--teacher_forcing':
      params['teacher_forcing_ratio'] = float(arg)
    elif opt == '-bi' or opt == '--bidirectional':
      params['bidirectional_flag'] = (arg == 'y')
    else:
      print('Follow the format to run the script.')
      sys.exit()
    
    if attn_flag == False:
      params['n_layers'] = n_layers
    
    #loading data
    x_train, y_train = obtain_data(data_dir + language_prefix + '_train.csv')
    x_test, y_test = obtain_data(data_dir + language_prefix + '_test.csv')
    x_val, y_val = obtain_data(data_dir + language_prefix + '_valid.csv')

    english = Language('eng')
    lang = Language('out_lang')
    english.update_vocab(x_train)
    lang.update_vocab(y_train)

    train_data = get_pairs(english, lang, x_train, y_train)
    test_data = get_pairs(english, lang, x_test, y_test)
    val_data = get_pairs(english, lang, x_val, y_val)

    training_pairs = [random.choice(train_data) for i in range(0, n_iters)]

    #initializing model
    if attn_flag:
      model = seq2seq_attn(inp_language = english, out_language = lang, **params)
    else:
      model = seq2seq_vanilla(inp_language = english, out_language = lang, **params)

    train_loss = 0

    one_every = n_iters//15

    for i in range(0, n_iters):
        training_pair = training_pairs[i]
        x = training_pair[0]
        y = training_pair[1]
        loss = model.train_step(x, y)
        train_loss = train_loss + loss

        if (i+1)%one_every == 0:
            print('------------------------------------------------')
            print('train loss is:', train_loss/one_every)
            if attn_flag:
              test_acc, char_acc = model.evaluate(val_data, print_flag)
            else:
              test_acc, char_acc = model.evaluate_beam(val_data, beam_size, print_flag)
            print(f'test accuracy is {test_acc} and character-wise accuracy is {char_acc}')
            train_loss = 0
    
    with open('model.pkl', 'wb') as file:
        pickle.dump(model, file)
        print('Model successfully saved.')
    
if __name__ == "__main__":
   main(sys.argv[1:])
    

