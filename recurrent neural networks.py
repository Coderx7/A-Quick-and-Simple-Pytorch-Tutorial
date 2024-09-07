#%% 
# In the name of God the most compassinate the most merciful
# introduction to RNNs in Pytorch
import os,sys
import numpy as np
import string

# to download files from internet! we can have several options
# we can use urllib.request
# its retrieve method will download the file and save it under a filename!
import urllib.request as request
# or use the requests module, to get remote files!(imagine this as wget!
# you have to save the file you downloaded!)
import requests
import matplotlib.pyplot as plt
import torch.version
%matplotlib inline 

import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim  

print(f'{sys.version=}')#3.11.9
print(f'{torch.__version__=}')#2.4.0+cu121

#%% RNN 
# In this section we are going to learn about RNNs in Pytorch, we will start with 
# Vaniall RNN and continue to see how GRU/LSTM works, and implement different project
# along the way to get a good grasp on the idea 
# we will first start with a simple example about time-series. as you know, we use 
# recurrent neural networks on series. we can use cnns as well, but here we will be 
# doing time-series and other sequential data using RNNs. 
# there are always 3 distinc steps to build a model. 
# 1. create the dataset or data to be fed to our model 
# 2. create our model in Pytorch 
# 3. run training/evaluation on the data 
# 4. visualize as needed 
# for our timesereies example, we need a dataset, to keep it simple, we create one for ourselves
# here is one way to do this : 
sequence_length = 30 
# using linspace, we create a series of points.(evenly spaced numbers over a specified interval.
# the series length is specified by the sequence length, which for our case, we defined it as 30. 
# using linspace the points are generated with fixed step-sizes. if we plot this 
# you'll note that they form a line! in order to give a structure/shape of some kind we alter these 
# points using sin().(we could use anything for his) 
# By doing this we are trying to create some kind of an underlying structure in our dataset
# and see whether our network can discover that underlying relationship and model it rather
# than just modeling a simple line!)
# please note that we could achieve this using other methods as well, for example
# generate some random numbers and then multiplying them by a theta, and network would
# need to learn what theta actually is to model the data. but for now lets use this 
sample = torch.linspace(start = 0, end=np.pi, steps=sequence_length)
sample_sin = sample.sin()
# now lets plot our data points and see how the look: 
plt.plot(sample,color='r')
plt.plot(sample_sin,color='g')
# we could do 
# dt = torch.randn(size=(30,))
# dt.mul_(5).add_(1.5)
# plt.plot(dt)

# so that was the sample, whats the label? 
# in time-series problems, we have different types of problems. 
# when our problem is a sequence to sequence and lets say we want to predict
# a value, we usually create the label from the sample itself, we just shift it 
# forward one timestep!
# so the sample and its label would be 
x = sample[:-1]
y = sample[1:] 
plt.plot(x, color='r' ,label='x')
plt.plot(y, color='y', label='y')

# now lets make a function for plotting make our life easier!
def plot_sample(x, y, fmt='g.', fmt2='y+', label1='x', label2='y'):

    # clears the figure, for this we have cla and clf. 
    # cla is used for clearing the current 'a'xes while clf
    # is used for clearing the whole 'f'igure
    plt.clf()
    # 
    # sidenote: 
    # plot([x], y, [fmt], *, data=None, **kwargs)
    # plot([x], y, [fmt], [x2], y2, [fmt2], ..., **kwargs)
    # The optional parameter fmt is a convenient way for 
    # defining basic formatting like color, marker and linestyle. 
    # It's a shortcut string notation described in the Notes section below.
    # plot(y, 'r+')     # ditto, but with red plusses
    # The following two lines are the same, the firts one uses [fmt] while
    # the second line uses proper arguments to create the same pattern!
    # >>> plot(x, y, 'go--', linewidth=2, markersize=12)  
    # >>> plot(x, y, color='green', marker='o', linestyle='dashed',
    #          linewidth=2, markersize=12
    plt.plot(x, fmt, label=label1)
    plt.plot(y, fmt2, label=label2)
    plt.legend()

# Anyway, what we just created and plotted was a single sample. a sample of 30 timesteps 
# we need more of these samples to train our model. 
# so we either need to create a dataset beforehand to use it during training, 
# or we can get away with it by just simulating that, how? by creating a generator!
# a generator that generates a sample each time it is called! 
# for our training, we need both a sample and its label! so we need to provide both.
def dataset_sample(i, seq_len=10, device='cuda'):
    # since we are creating new tensors, we can simply use a
    # device context manager, this way we dont have to use .to()
    with torch.device(device):
        # to create proper spacing/create new datapoint for each call
        # we incorporated i (index) into start and end points
        sample_raw = torch.linspace(i*np.pi, (i+1)*np.pi, seq_len+1)
        sample_data = sample_raw.sin()
        # since we plan on training, we add a batch-dimension for our toy example
        # we will be using batch of 1 here to keep everything simple, but later on
        # we see how we can create batches of larger sizes.
        data = sample_data[:-1].view(1,seq_len,1)
        label = sample_data[1:].view(1,seq_len,1)
        yield data, label

# lets test this
data, label = next(iter(dataset_sample(0)))
# flatten them to plot them
data = data.view(-1)
label =label.view(-1)
plot_sample(data.cpu(), label.cpu(), label1='data',label2='label')
# plt.clf()
# plt.plot(data.cpu(),'b+-', label='input, x')
#%%
# now lets create our actual RNN model.
# since some arguments are self explanetory, I'll only explain about the ones 
# that are new to us, and we havent covered so far like RNN module itself.
class RNN_Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1,
                 drpout=0.5, bidirect=False):
        super().__init__()

        self.input_size = input_dim
        self.hidden_size = hidden_dim
        self.output_size = output_dim
        # The first thing to pay attention to, when working with rnns in pytorch is
        # unlike normal vanilla rnn implementation, pytorchs rnns(grus and lstms),
        # will only return a tuple which includes, outputs and hiddenstates for each timestep. 
        # obviously the output is not the actual output that we want, for that we need
        # to add another layer after the rnn like an fc(linear) layer with sigmoid to 
        # achieve the actual output. (dont worry if its vague to you, we will get to this in a moment)
        # The second thing is num_layers, basically this allows you to create a stacked rnn
        # usually you may choose between 1-3 layers. 
        # citing from the documentation, 
        # 1.num_layers: Number of recurrent layers. E.g., setting num_layers=2 would mean 
        # stacking two RNNs together to form a stacked RNN, with the second RNN taking in
        # 'outputs' of the first RNN and computing the final results. Default is 1
        # 2.batch_first, means, if you have your data in batches(batch is the first dim), set this to true
        # 3.bidirectional, means whether you want your rnn to be bidirectional! 
        # by the way, the input_dim actually refers to the input size, e.g if you 
        # one_hot encoded your input, you feed the one_hot encoded dim.(we will see this
        # in a moment!)
        # note dropout will actually be relevant if we have more than 1 layers!
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers=num_layers,
                          batch_first=True,
                          dropout=drpout, 
                          bidirectional=bidirect)
        # as I just pointed out, for the actual outputs we need a new layer
        # we use fc layer, since we are going to predict values 
        # we need 30 values as output for each sample!
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden_state):
        # our input should have the shape : 
        #   (batch_size, seq_length, input_size)
        # our hidden states can be None or if not should have the shape: 
        #   (n_layers*direction, batch_size, hidden_dim)
        # and the output returned by rnn has the shape: 
        #   (batch_size, time_step, hidden_size)
        rnn_outputs, hidden_states = self.rnn(x, hidden_state)
        rnn_outputs = rnn_outputs.view(-1, self.hidden_size)
        output = self.fc(rnn_outputs)
        return output, hidden_states
        

batch_size=1
iteration = 80 
interval = 15 
sequence_length = 20
hidden_dim = 50
# we can have a stacked rnn, if we want one simply increase this number 
num_layers = 1
# uni or bidirection. ( 1 or 2)
direction = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# its initially zero, we could use None as well, and we usually do that
# but for now, I want you to see how a hidden state dimensions looks like
# under the hood so if during training something went wrong you know whats what!
# hidden_state = None
hidden_state = torch.zeros(size=(num_layers*direction, batch_size, hidden_dim)).to(device)
# input dim =1 means 1 character/ output dim 1 
# means 1 character (basically we are trying to 
# repilicate input (predict the next value))
model = RNN_Net(input_dim=1,
                hidden_dim=hidden_dim,
                output_dim=1,
                num_layers=num_layers).to(device)

# training 
# For the loss we simply use the mse 
criterion = nn.MSELoss()
# experiment with a large lr sucha s 0.1 and
# also a small lr like 0.0001 and see the changes
optimizer = optim.Adam(model.parameters(), lr=0.01)

# for debugging especially gradient related ones 
# we can use this. we'll talk aboutit more later
# torch.autograd.set_detect_anomaly(True)
print(model)

# note we dont have epochs here, just repeating the optimization process 
# several times. also note we are using a single sample each time which 
# is ok here, but not in real life as its very inefficient! 
for i in range(iteration):
    for (data, label) in dataset_sample(i, seq_len=sequence_length, device=device):
                                
        (output, hidden_state) = model(data, hidden_state)
        ## Representing Memory ##
        # make a new variable for hidden and detach the hidden state from its history
        # this way, we don't backpropagate through the entire history
        hidden_state = hidden_state.data
        # make sure the dims for output and label are the same
        # otherwise, you may not get an error, but a sckewed result
        # that may bogle your mind for quite sometime!
        # so we do this otherwise the assert will prevent us
        # for our specific case it doesnt cause an issue as this operation 
        # involving our output with the shape( 20,1) and our label with the
        # shape (1,20,1) will be broadcasted to (1,20,1) and everything will be fine!
        # but keep an eye nothetheless when you choose a different configuration!
        # assert output.shape == label.shape, f'output shape({output.shape}) and label shape({label.shape}) must match! or they will result in ({(output ==label).shape=})'
        loss = criterion(output, label)
        print(f'iter: {i} loss: {loss.item():.6f} ')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if i%interval==0:
        # sidenote 
        # since v1.9.0 flatten is available as a tensor operation so we can use it!
        # Unlike NumPy’s flatten, which always copies input’s data, 
        # this function may return the original object, a view, or copy. 
        # If no dimensions are flattened, then the original object input is returned. 
        # Otherwise, if input can be viewed as the flattened shape, then that view is returned.
        # Finally, only if the input cannot be viewed as the flattened shape is input’s data copied
        x = data.flatten().data.cpu().numpy()
        y = label.flatten().data.cpu().numpy()
        # or we could use the good old view(-1) for flattening!
        output = output.view(-1)
        
        plt.plot(x, 'g.', label='x-data')
        plt.plot(y,'b.', label='label')
        plt.plot(output.data.cpu().numpy(), 'r+',label='output')
        
        plt.legend()
        plt.show()
# Ok we see the network could easily learn the pattern and predict it. 
# now that we have seen and learned how to use a RNN module, lets scale 
# this up and use a more practical example.
#%% 
# Now using GRU and LSTM is the same, but we are going to use them 
# in a new architecture. lets delve into the realm of nlp and write
# a simple text generator! which is a bit more involved than our previous
# toy example. we will cover more concepts and hopefully get a better understanding
# concerning all of this.
# 
# sidenote: 
# the original version of this was written in 2018/2019 at which time
# rnns were still widely used and transformers werent as widespread yet.
# note that we dont use rnns to do these stuff anymore,
# we instead use newer architectures such as transformers which are amazingly good at these
# kinds of tasks and we will revisit all of these concepts(text generation, captioning, etc)
# with them in future sections as well.

# For our dataset, we use gutenberg opensource library which contains over 60K free ebooks,
# you can access this library using this link : http://www.gutenberg.org
# we use http://www.gutenberg.org/files/1399/1399-0.txt but feel free to use anything you
# you like !
# lets roll everybody!:)

# lets download the book and create a dataset out of it
# ref : https://stackoverflow.com/questions/7243750/download-file-from-web-in-python-3 
def download_ebook(url, file_name='corpus.txt', dir_name = 'data'):
    file_name_path = os.path.join(dir_name, file_name)
    
    if os.path.exists(file_name_path):
        return file_name_path
    
    if not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)
        
    request.urlretrieve(url, file_name_path)
    return file_name_path

url = 'http://www.gutenberg.org/files/1399/1399-0.txt'
# lets download and read it all!
with open(download_ebook(url),'r') as file: 
    corpus_raw = file.read()

# our text, contains more than pure alphabet characters.
# its filled with special characters as well. characters
# that are not visible, such as line breaks, escape characters, etc
# we use repr to see these special characters
# without it, the print statement will render them all and we wont see anything!
print(f'{repr(corpus_raw[:10])=}')
# now lets have a minimal preprocessing step in which 
# we remove all punctuation marks plus any characters 
# that are not printable, i.e. special characters.
# also lets make alll the characters lowercase! 
# this will help with the end result, there are more preprocessings, but
# for now its enough, we will see more in the next sections
# such as sentiment analysis and word embedding sections
# string module provides us a translate method which we can use to replace
# specific characters with some other characters. we can use it to replace 
# the ones we dont want. e.g the punctuation characters. 
# we are creating a translation dictionary, and replacing all punctuations with ''
# basically removing them with this line
corpus_raw = corpus_raw.translate(str.maketrans('','',string.punctuation))
# we can now filter out the other escape characters!
# by only selecting the printable ones!
corpus_raw = ''.join(x for x in corpus_raw if x in set(string.printable))
# and finally making all characters lower case
corpus_raw = ''.join([c.lower() for c in corpus_raw])
# now we get 'the projec' this time around!
print(repr(corpus_raw[:10]))

# We need to convert our input from textual format into a numerical one
# that can be used to train our model with.
# this is where tokenization comes into play. 
# simply put, tokenization refers to us selecting the basic building block
# of our input, which here is a character, and assign a numeric code to it.
# (we could use words instead of characters, instead, but thats more computationally
# demanding, but we get to it later on nonetheless). 
# note that tokenization is extremely important and the way its done has a critical impact
# on the end result. we will learn more about them in future sections but for now, we keep it simple)
# so our first step would be to tekenize our whole corpus 
# basically, we will tokenize our corpus of text
# and then digitize each letter/symbol, and then 
# use the encoded representation of these digitized
# words. 
# In order to encode them, we first need to 
# create a dictionary of each letter 
# first we find all unique symbols.
# using set, we can avoid duplicates
unique_chars = set(corpus_raw)
# lets sort it for better visualization
unique_chars = sorted(unique_chars)
# now lets create a char2int and int2char dictionaries!
int2char = dict(enumerate(unique_chars))
char2int = {c:d for d,c in int2char.items()}
print(f'{unique_chars=}')
print(f'{int2char=}')
print(f'{char2int=}')
print(f'unique characters: {len(unique_chars)} : \n {unique_chars}')
# lets convert our corpus to int
corpus_digitized = torch.tensor([char2int[char] for char in corpus_raw])
print(f'{corpus_digitized.shape=}')
print(corpus_digitized[:10])
print('after conversion: ')
print(repr(''.join([int2char[idx.item()] for idx in corpus_digitized[:10]])))

# ok so far so good. we now need to create a one-hot representation of our input
# our inputs are digits, each digit as you saw, represents a single letter/symbol
# so in order to train our net, we must one_hot encode them,
def one_hot(array, length=10):
    # our list conains several digits, we will create a one hot vector
    # for each digit. 
    # you might think to yourself, why using float32 while we can use unsigned-int8,
    # as we only deal with 0-1?! we can use uint8 and it helps alot in saving memory!
    # as it only takes 1 byte for each number where as sth like a float32/int32 takes 4 bytes.
    # but during training we have to cast this to float32 or otherwise the training will fail
    # so we may as well set it as float32 when we are making it in first place. 
    # to conserve memory and train with lower precision, there are other way we can use
    # such as halfprecision training which we will see in future chapters.
    one_hot_array = torch.zeros(size=(array.numel(), length), dtype=torch.float32)
    # instead of using a simple for loop, we use the vectorized version
    # sidenote:
    # the torch.arange() simply creates a list of indexes and uses it to 
    # pick values from the input-array which we flatten here, that value will
    # be used as index for one_hot_array ultimately. it simulates a loop in a
    # vectorized manner which is much faster!
    # note the indeces in pytorch must be long/int or byte. since our array maybe
    # int8 itself, we make sure the indexes are long().
    one_hot_array[torch.arange(array.numel()), array.flatten().long()] = 1
    # reshape the onehotvector to the original shape of input
    # this is to basically get sth like (b,t,c) 
    return one_hot_array.view(*array.shape, length)

# lets test this 
x = torch.tensor([0,1,9], dtype=torch.int32)
print(f'onehot encoded:\n{one_hot(x,10)}')
# sidenote: 
# In pytorch, the size of a data type is determined by the underlying numpy data type. 
# PyTorch uses the same data type sizes as NumPy.
# so torch.float32 and torch.int32, both have 4 bytes (they are 32bits after all)
# we can verify the size of these data types using the `torch.tensor.element_size` 
# attribute, which returns the size of each element of the tensor in bytes.
# 
# Check the element size
print('dtype sizes:')
print(f"Size of {x.dtype}: {x.element_size()} bytes")
print(f"Size of {x.float().dtype}: {x.float().element_size()} bytes")

# we can also use the `torch.finfo`(for floats) and `torch.iinfo`(for ints) functions to 
# get information about the data types, including their sizes.
print(f'using torch.finfo/torch.iinfo')
print(f"Size of {x.dtype}: {torch.finfo(torch.float32).bits // 8} bytes")
print(f"Size of {x.float().dtype}: {torch.iinfo(torch.int32).bits // 8} bytes")
# 
# which prints 4 bytes on my sysem. 
# Note that the sizes of data types can vary across different hardware platforms
# and Python installations. However, for `torch.float32` and `torch.int32`, the 
# sizes are typically 4 bytes on most modern systems.

# sidenote: you may ask, wait a second, why do we use characters and not words here? 
# we chose characters, becasue that would result in a small one-hot-encoded vector
# if we used words for example, and we wanted to one-hot encode them, that would be a several thousands
# elements vectors, that would take a huge amount of vram and computation! 
# later on we will use an embedding layer (we will see later on how a typical emebedding vector works)
# which takes way less memory and computation and is much much more efficient in terms of capturing 
# underlying relationships between words, etc. unlike one-hot-encoding, we can specify any dimension size
# for our embedding vector (in one-hot-encoding, we have to use vocab_length (total number of unique words
# in our text) for each vector and the absolute majority of that vector is just zeros which is extremely
# inefficient and wasteful). so thats why we first started using characaters, and then later on, 
# we will see how we can use words with embedding vectors.
# note that using words, comes with own issues aswell, which we will get to when we get to it.

#%%
# now we need to have batches! 
def get_next_batch(corpus_digitized, batch_size=1, seq_len=10):
    # lets create batches from our corpus, first lets see
    # how many batches we can get from our corpus
    char_count = corpus_digitized.size(0)
    # calculate how many characters can fit in a batch given the sequence-length
    each_batch_size = batch_size*seq_len
    # we can have this many batches
    batch_count = char_count // each_batch_size
    # now we should reshape our corpus data for easier access
    corpus = corpus_digitized[:batch_count * each_batch_size]
    # reshape it into having a batch-dim
    corpus = corpus.reshape(batch_size, -1)
    # read one batch for data and label, 
    # note our dtype is int becasue we are dealing with 0 and 1
    # we didnt choose int8, to conserve memory.
    # important note: 
    # note that since our whole character set is less than 256(its 38!), we can 
    # do this, otherwise we would face issues becasue x can not contain
    # values larger than 256! it will silently fail and you may spend a lot of time
    # to find the real culprit!
    # to be on the safe side, you can use torch.int32! but for our special case,
    # lets use uint8 (unsigned-int8) here!
    x = torch.zeros(size=(batch_size, seq_len), dtype=torch.uint8)
    y = torch.zeros_like(x)

    # preparing new input by reading seq_len from corpus each time
    # note the try block is there to fill the labels for the last batch
    for i in range(0, corpus.size(1), seq_len):
        # sidenote: 
        # x[...] is the same as x[:,:] which fills the x
        # if we used x instead, it would be a new variable of whatever type corpus
        # happened to be(i.e. int64). since we wanted to use uint8 to conserver 
        # memory, we use x[...], otherwise it really wouldnt have mattered to us
        # if we used int32/int64 as dtype for x reall.
        x[...] = corpus[:, i:i+seq_len]
        try :
            y[:, :-1] = x[:, 1:]
            y[:, -1] = corpus[:, i+seq_len]
        except:
            y[:, :-1] = x[:, 1:]
            y[:, -1] = corpus[:, 0]
        yield x,y

x,y = next(iter(get_next_batch(corpus_digitized,batch_size=3,seq_len=8)))
print(f'{x.dtype=}')
print(f'{y.dtype=}')
one_hot_x = one_hot(x, length=len(unique_chars))
one_hot_y = one_hot(y, length=len(unique_chars))
print(f'{x.shape=}')
print(f'{y.shape=}')
print(f'{x=}')
print(f'{y=}')
# print(f'{one_hot_x=}')
print(f'{one_hot_x.shape=}')
#%%

# Ok everything seems ok now. lets create our network 
# in order to be able to compare between different network types, 
# lets use all of them. lstm works the best as you will see
# followed by GRU and then RNN. as you can guess, RNN has the worst
# performance in terms if accuracy and text generation!

class lstm_char(nn.Module):
    def __init__(self, rnn_type='rnn', unique_chars=110, hidden_size=30,
                 num_layers=1, dropout=0.3, bidirection =False, act='tanh'):
        super().__init__()
        
        self.unique_char = unique_chars
        self.int2char = dict(enumerate(self.unique_char))
        self.char2int = {ch:ii for ii,ch in int2char.items()}
        self.input_size = len(unique_chars)
        self.output_size = self.input_size
        self.hidden_size = hidden_size
        self.drp = nn.Dropout(0.3)
        self.rnn_type = rnn_type.lower()

        self.direction = 2 if bidirection else 1

        if rnn_type.lower() == 'rnn':
            self.rnn = nn.RNN(self.input_size,
                            hidden_size,
                            num_layers,
                            batch_first=True,
                            dropout=dropout,
                            bidirectional=bidirection)
        
        elif rnn_type.lower() == 'gru':
            self.rnn = nn.GRU(self.input_size,
                            hidden_size,
                            num_layers,
                            batch_first=True,
                            dropout=dropout,
                            bidirectional=bidirection)
        else:
            self.rnn = nn.LSTM(self.input_size,
                            hidden_size,
                            num_layers,
                            batch_first=True,
                            dropout=dropout,
                            bidirectional=bidirection)
        
        self.fc = nn.Linear(hidden_size*self.direction, self.output_size)
    
    def forward(self, input, hidden_states):
        rnn_outputs, hidden_states = self.rnn(input,hidden_states)
        outputs = rnn_outputs.reshape(-1, self.hidden_size*self.direction)
        outputs = self.drp(outputs)
        outputs = self.fc(outputs)
        return outputs, hidden_states


# lets test our newtork 
seq_len = 30
data, labels = next(iter(get_next_batch(corpus_digitized, batch_size=2, seq_len=seq_len)))
print('initial data shape before one_hot encoding: ',data.shape)

print(f'{data.dtype=}')
print(f'{labels.dtype=}')
data = one_hot(data, length=len(unique_chars))
labels = one_hot(labels, length=len(unique_chars))

# data = torch.from_numpy(data)
# labels = torch.from_numpy(labels)

direction = True
num_layers = 1
model = lstm_char(rnn_type='lstm', 
                  unique_chars=unique_chars,
                  hidden_size=30,
                  num_layers=num_layers,
                  bidirection=direction )

print(f'rnn type : {model.rnn_type}')
print(f'our input(data).shape: {data.shape}')
outputs, hiddenstates = model(data,None)
print(f'model input size: {model.input_size}')
print(f'model output size: {model.output_size}')
# now our output may look weird sth like, remember that
# in order to get meaningful output we need to reshape it 
print(f'rnn output shape: {outputs.shape}')
# therefore the actual shape is 
print(f'output actual shape :{outputs.view(-1, seq_len, model.output_size).shape}')

#%%
# now lets train our model 
hidden_size = 512
layers_cnt = 2
bidirection = False
rnn_type = 'lstm'
device = 'cuda' if torch.cuda.is_available()  else 'cpu'
model = lstm_char(rnn_type=rnn_type,
                  unique_chars=unique_chars, 
                  hidden_size=hidden_size,
                  num_layers=layers_cnt, 
                  bidirection=bidirection,dropout=0.5)
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr = 0.01)
criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=20)

epochs =60
# in order to not face the exploding gradient in lstm
# we clip the gradients
clip = 5.
interval = 100 
batch_size = 128
label_length = len(unique_chars)
hidden_states = None 

val_ratio = 0.2
val_idx = int(corpus_digitized.numel() * (1-val_ratio))
train = corpus_digitized[:val_idx]
val = corpus_digitized[val_idx:]

print(f'running on device: {device}')
print(f'rnn_type: {model.rnn_type}')
print(f'corpus size: {corpus_digitized.numel():,}')
print(f'val size: {val.numel():,}')
print(f'train size: {train.numel():,}')
print(f'val + train: {val.numel() + train.numel():,}')
assert train.numel() + val.numel() == corpus_digitized.numel() ,'they must be equale!'

for e in range(epochs):
    model.train()
    total_loss = 0
    for i, (data, label) in enumerate(get_next_batch(train, batch_size, seq_len=seq_len), start=1):
        
        # one hot encode the data and then feed it to the model
        data = one_hot(data,length=label_length).to(device)
        # label is not one-hot-encoded, crossentropy can do this on its own
        label = label.to(device)

        output , hidden_states = model(data, hidden_states)
     
        if model.rnn_type == 'lstm':
            hidden_states = tuple(h.data for h in hidden_states)
        else:#RNN, GRU
            hidden_states = hidden_states.data 
        
        # We dont one hot encode our labels, the crossEntropy() layer will do this internally!
        # since for output, we reshaped it so that the batch_size and sequence length are fused
        # we do the same thing for labels, so it can be used in cossentropy!()
        # the crossentropy loss, will internally convert each digit to its corosponding one_hot
        # encoded vector. so basically we are just making sure, the shape between, output and 
        # label is the same 
        label = label.view(batch_size*seq_len).long()
        loss = criterion(output, label)
        total_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        # note the _, which indicates the inplace operation!
        torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=5.)
        optimizer.step()

        if i%interval==0:
            print(f'Epoch-Iter:: {e}/{epochs}-{i} | Loss: {total_loss/i:.4f} | LR: {scheduler.get_lr()[-1]:.6f}')
    
    # decay the lr per epochs
    scheduler.step()

    # test 
    hidden_states = None
    total_loss_val = 0
    for i, (data,label) in enumerate(get_next_batch(val, batch_size, seq_len),start=1):
        with torch.no_grad():
            model.eval()

            data = one_hot(data,label_length).to(device)
            label = label.to(device)

            output, hidden_states = model(data, hidden_states)
            
            if model.rnn_type == 'lstm':
                hidden_states = tuple(h.data for h in hidden_states)
            else:#RNN, GRU
                hidden_states = hidden_states.data 
            
            total_loss_val += criterion(output,label.view(batch_size*seq_len).long()).item()
            
            if i % interval ==0:
                print(f'  Loss-val: {total_loss_val/i:.4f}')
    print(f'  Loss-val-total: {total_loss_val/i:.4f}')
#...
#   Loss-val-total: 1.1561
# Epoch-Iter:: 58/60-100 | Loss: 1.0288 | LR: 0.000100
# Epoch-Iter:: 58/60-200 | Loss: 1.0290 | LR: 0.000100
# Epoch-Iter:: 58/60-300 | Loss: 1.0288 | LR: 0.000100
#   Loss-val-total: 1.1563
# Epoch-Iter:: 59/60-100 | Loss: 1.0305 | LR: 0.000100
# Epoch-Iter:: 59/60-200 | Loss: 1.0294 | LR: 0.000100
# Epoch-Iter:: 59/60-300 | Loss: 1.0288 | LR: 0.000100
#   Loss-val-total: 1.1565
# 
# After around 60 epochs we get to ~1.02 in training and 1.15 in val
# 
#%%
# sidenote: 
# there are many ways to generate good looking text, many such techniques
# were devised for LLMs in the past few years.
# we will see a few of them in transformer/llm sections

# now its time for sampling and generating new text.
# basically this boils down to feeding a random character and then 
# feed the next generated character to the next timestep as the next input
# and this goes on till we generate the whole text. 
# since our network produces distributions, we use softmax to get the probablity
# , for each timestep, we can choose the highest probable outcome, or just randomly
# choose one, you'll see how this is done in a moment. 
# first we need to create a fucntion that does one thing only! 
# feed one character and retreieve one character from our neytwork. 
# we then use this function to create more characters in a loop.
# thats basically it!
def predict(model, input, hidden_states=None, topk=5):
    model.eval()
    int2char = model.int2char
    char2int = model.char2int
    unique_chars = len(int2char)
    # print(char2int)
    # convert input string into corrosponding ids and add a batch dim
    input =  torch.tensor([char2int[input]]).reshape(1,-1)
    one_hot_vec = one_hot(input, unique_chars).to(device)
    output, hidden_states = model(one_hot_vec, hidden_states)

    output = output.softmax(dim=1)
    # now our output has probabilities for each sequence/timestep
    # we will choose the highest one here 
    probs, indexes = output.topk(k=topk, dim=1)
    indexes = indexes.cpu().data.squeeze()
    probs = probs.cpu().data.squeeze()
    
    # if we were to use numpy, we would have to write it like this, 
    # use indexes, and also provide the probablities for these 
    # char = np.random.choice(indexes.numpy(),p=probs.numpy()/probs.numpy().sum())
    # note we have to renormalize the probs so that each of these new probablities
    # also note that, we sample from the indexes, each index represents a character
    # so sampling from them means choosing between different characters based on their
    # probablities here.
    if probs.numel()==1: # if topk=1
        char = indexes.item()
    else:
        char = indexes[torch.multinomial(probs/probs.sum(dim=-1), num_samples=1, replacement=True)].item()
    return int2char[char], hidden_states

def sample(model, size=10, prompt='hello there',topk=5):
    
    # chars = [ch.lower() for ch in starter_message]
    h = None
    prompt = prompt.lower()
    # add the prompt/start message
    chars = list(prompt)
    # print(unique_chars)
    # now append the new generated text 
    # by first feeding the whole prompt/starter message to 
    # condition the model, and then the append the last output
    # (which is what we want) to the chars list
    for ch in prompt:
        o,h = predict(model, ch, h, topk=topk)
    chars.append(ch)

    # now start from the last newly generated character
    # we just got from previous loop, use it to generate
    # new characters, as many as the size dictates.
    # chars[-1] grabs the last freshly generated character
    # from previous attempt and feeds it to the network to
    # generate new one. the new one is then appended to the
    # chars list and this continues until we have generated
    # the right amount of characters specified by size 
    for _ in range(size):
        o, h = predict(model, chars[-1], h, topk=topk)
        chars.append(o) 
        
    # finally we convert all into a big string
    return ''.join(chars)

print(unique_chars)
print(repr(sample(model, size=200, prompt='\n',topk=5)))
# output: 
# \n\nand will be the creation was announced\n
# the storm why do you know when i would your\n
# form and she said smiling she went out that all over and he smiled him at the man to
# decide it in all its an answer t
 
# Heres a second output-with loss:1.056:
# anna and the carriage was sinking of another that he was almost successfully a sense of tears that he was a look
# \nwell thats a little thing i am there and i could not ask the sight of the powers of th
# 
# with loss ~= 1.0288 we get much better outputs like this: 
# yes i can go away and i have not seen it\n\n
# well then she thought\n\n
# yes to do with anna that you will come to me and then said levin and he was\n
# not all at once with him and so said he would have said all'
# 
# Now try with topk=1 which means to grab the most probable character all the time
# and compare it to the topk=5. try different prompts and see how it affects the output.
# 
# you see the network have learned words, and their positions in the sentence,
# and also a bit of grammar albeit not prefectly, it doesnt produce a lot of 
# meaningful pieces yet, but as loss gets lower, we can see results get better.
# with better training regime, better architecture, etc we can improve upon it.
# we dont spend too much time here, as we will return to this when we cover llms 
# and transformers in future sections. 
#%%
# NOte : 
# how to use bidirectional LSTM/RNN 
# In general if we want to create our own Bidirectional rnn network(be it rnn,lstm, gru), 
# we need to create two normal networks(lstms for example), and then feed one with 
# the normal input sequence, and the other with inverted input sequence. 
# After that, we just take the last states from both networks and concat them together 
# (or sum them concatenation seem to be the norm!) and thats all. 
# but using Pytorch we dont have to deal with this hassle 
# for using bidirectional version of the mentioned rnn networks, just look at the implementation
# that I provided in our example. 
# 
# todo: summarize/rephrase: 
# to create a bidirectional RNN from scratch, 
# we would need to create two separate RNN models (e.g., LSTM, GRU, or vanilla RNN).
# One model would process the input sequence in the normal order, 
# while the other model would process the input sequence in reverse order.
# after processing the input sequences with both models, we would typically concatenate
# (or sum) the final hidden states or outputs from both models. 
# Concatenation is more common, as it preserves the information from both directions.
#  
# 
# note that in our simple case of text generation, the bilstm wouldnt magically make everything better
# infact you may see the loss decreases much more, but the text generation is aweful! guess what
# is causing this? the bidirection defeats the purpose of serial nature of the text, the network
# cant learn/model the statistics of what comes next given certain characters.

#%%
#%% Attention Mechanism 
# good resources : 
# papers: 
# https://www.aclweb.org/anthology/D15-1166.pdf
# https://arxiv.org/pdf/1502.03044.pdf
# https://arxiv.org/pdf/1409.0473.pdf
# others:
# https://www.youtube.com/watch?v=yInilk6x-OY
# https://www.youtube.com/watch?v=W2rWgXJBZhU&t=607s
# https://blog.floydhub.com/attention-mechanism/

# https://towardsdatascience.com/
# https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/seq2seq-translation.ipynb
# https://colab.research.google.com/github/bastings/annotated_encoder_decoder/blob/master/annotated_encoder_decoder.ipynb#scrollTo=NL240dtrgDw6
# intuitive-understanding-of-attention-mechanism-in-deep-learning-6c9482aecf4f  
# https://github.com/thomlake/pytorch-attention 
# https://medium.com/intel-student-ambassadors/implementing-attention-models-in-pytorch-f947034b3e66 
# https://towardsdatascience.com/attention-seq2seq-with-pytorch-learning-to-invert-a-sequence-34faf4133e53  
# https://machinelearningmastery.com/how-does-attention-work-in-encoder-decoder-recurrent-neural-networks/
# https://medium.com/syncedreview/a-brief-overview-of-attention-mechanism-13c578ba9129  
# https://stackoverflow.com/questions/50571991/implementing-luong-attention-in-pytorch
# https://medium.com/@shashank7.iitd/understanding-attention-mechanism-35ff53fc328e
# https://medium.com/@bgg/seq2seq-pay-attention-to-self-attention-part-1-d332e85e9aad

# Now lets learn about Attention mechanism! 
# sidenote - old
# before we continue, for sequence to sequence models and translations, one of the good 
# resources is https://www.manythings.org/anki/ that we can download text corpus and use it
# for translations. OK now lets continue with attention mechanism 
# as you can see, I have posted a lot of resources that you can use, I myself read some of them
# and will quote from them. 

# So what is an attention mechanism and why do we even care? 
# Attention mechanism, as its name implies, is a mechanism which helps the network to pay more
# attention to specific parts of the input in order to produce more plausible outcome. 
# it was initially proposed for NMT or neural machine translation, where for example, you'd want
# to translate a sentence from one language to another.
# in a traditional case which we saw earlier in such cases, a seq2seq model is used, 
# that is, a model comprising of two networks, an encoder and a decoder, where the input sequence is fed to the encoder, a decoder ultimately recieves a 
# compressed representation , representing the input sequence, from the encoder part and then, 
# tries  to produce a sequence as the answer. 
# the problem with this procedure was/is that for a long
# sequence, we cant transfer the information from earlier time steps, its just simply not possible 
# (yes I know how we said about the lstm and gru gates, retaining earlier time step features, the 
# idea here, is even if lstm gates simply and ease the transfer of a specific feature from earlier
# time steps to the later timesteps, lots of such information will be lost becasue the last output
# is simply a finite fixed-size vector which can only accomated so much features, and mostly they 
# will be features from recent timesteps as apposed to earlier ones. please note that, there 
# are lots of relationships between each word, and also underlying concept in a given sequence, 
# suppose, in an optimal case, your sequence, had 40 underlying features, that needed to be identified 
# and used so a perefect output is created, however, since there is no mechanism to retain 'all' of 
# these features, and we face a fixed final vector, plus our training procedure is not perefect and 
# also have noise!, only a handful of such features get the chance to be transfered, features do get
# identified, but they cant be utilized as we dont have a mechanism to use them effectively so in the
# current procedure, they just get lost. 
# so what should we do then? we can use all of the states 
# from all previous timesteps instead of using only the last one. this way, we can provide much more
# information and this wealth of information at each timestep can help the network produce better result
# but how do we do that? surely not all hidden states, are equally important when it comes to producing 
# a translation, a new word e.g. here we can rank them based on how much they affect the outcome. 
# this way the network will gradually understand the relationship and focuses on the correct states
# when needed. this is the gist of attention. we simply use all hidden states from all timesteps in 
# the encoder and feed them to the decoder as input. 
# how do we pay attention more or less to a specific hidden state at a timestep ?
# we calculate a score between that hidden state and the current hidden state of our decoder (
# the current timestep in the decoder). that is, for every single timestep in our encoder, 
# we simply calculate a score beteen that timestep and the previous timestep in our decoder(
# that is used for creating the current output). 
# how do we do that? there are several ways for this that we will get to shortly. 
# when we calculated scores for all timesteps, we take the softmax. why do we do that? 
# we do that, so the cumulative score is 1, and we can treat these scores, as a conditioal factors
# a weight that specifies / shows the importance of a specific timestep. 
# as we said there are several ways for implementing attention mechanism. 
# two important methods exist among others. They are known as Bahdanau, and luoung , 
# which indicate the main authors of two papers that proposed these methods for attentions. 

#ref https://blog.floydhub.com/attention-mechanism/
# Bahdanau paper: https://arxiv.org/abs/1409.0473 
# 
# As we briefly mentioned a little before, one of the main issues of the traditional RNNs, when 
# it came to do sequence-to-sequence modeling (e.g. machine translation), were that they used a
# fixed length vector/representation. this works for small lengths, but when the sequence length
# becomes larger, it becomes extremeley inefficient. we simply cant retain a state from long time ago
# and therefore cant form new relationships, reveal underlying concepts, etc properly and adequatly.
# Bahdanau et al, therefore proposed a method to fix this issue. So in his work, he proposed to utilize
# all intermediate hiddenstates from encoder, as aposed to its final state, when we are generating new 
# tokens in the decoder. remember the idea was the encoder digests the input, creates a middle compressed
# representation, which then the decoder uses to generate a translation for. but in its simple form the encoder
# cant encode all the necessary information, therefore the so called compressed form is extremely lossy and
# cant be used to represent the input sequence adequatly and properly. this in turn leads to decoder not being
# able to do its job properly as it doesnt get the required information it needs. this mechansim however, 
# tries to fix this issue, by streaming more information from different timesteps in the encoder to the 
# decoder so it can access to more information and consequently does a better job.(and indeed it does)
# this is what later became known as, Bahdanau attention, or also Additive Attention.
# 
# The basic idea as we just saw, is to simply somehow connect the decoder's hidden states with the
# relevant input sentences which is nothing but the encoder's outputs(hidden states) for each timestep. 
# this connection is shown as a simple addition (hence the name addative attention) between the two. 
# Attention = v_scale * tanh(W_decoder * decoder_hiddens + W_encoder * encoder_hiddens)
# This is the gist of it, there are concepts such as context and other implmentation details which we 
# are going to get to in a moment. 
#
# sidenote: a lot of people implemented this mechanism differently! and many incorrectly, below is the
# correct way, in the sense that it is what the paper instructs to follow is presented. 
# the very basic idea is to use the attention weight (after going through softmax)and apply it on the
# encoder's hidden-states and get a context vector. it is then fed to the decoder to get new hiddenstates/outputs
# and this repeates until the decoder creates an output for all timesteps. 
# in the second version, the context vector is applied on the input(the same input which encoder digested)
# and then that is then fed to the decoder and its repeated until all timesteps are done.
#
# The actual overal steps according to the paper however, is as follows: 
# 1.Encoder produces hidden states of each element in the input sequence
# 2.Decoder recieves these hidden states and calculating an alignment score 
#   between its previous hidden state and each of the encoder’s hidden-states. 
#   (Note: The last encoder hidden-state can be used as the first hidden-state in the decoder)
# 3.The scores for each encoder hidden state are combined and represented in a single vector and subsequently softmaxed
# 4.The encoder hidden-states and their respective alignment scores are multiplied to form the context vector
# 5.The context vector is concatenated with the previous decoder output and fed into the Decoder RNN 
#   for that time step along with the previous decoder hidden-state to produce a new output
# 
# The process (steps 2-5) is repeated for each time-step of the decoder until a token is produced
# or output is past the specified maximum length.

# sidenote:
# remember in PyTorch, a RNN returns two outputs, the first one is the hidden-states for each timestep
# and the second one is the final hidden-state (for the last timestep). PyTorch refers to the first returned
# field, as outputs, and the the second one as hidden_states. just be aware of that.
# 
# (this is important and the outputs may not be the best name, as unlike the traditional sense, 
# its not gone through a linear layer with activation function (e.g. tanh in the case of simple RNN),
# as one might think of the output layer of a rnn like what we did in our RNN tutorial/implementation) 
# therefore, instead of using only the last hidden state for the final timestep, we’ll be carrying forward 
# all hidden-states (i.e. for all timesteps) produced by the encoder to the next step.

# After this step, we can start using the decoder to producethe outputs. 
# At each time step of the decoder, we have to calculate the alignment score of each encoder output
# with respect to the decoder input and hidden-state at that time step. 
# The alignment score is the essence of the Attention mechanism, as it quantifies the amount of 
# "Attention" the decoder will place on each of the encoder outputs when producing the next output.

# The alignment scores for Bahdanau Attention are calculated using the hidden state produced by
# the decoder in the previous timestep and the encoder outputs with the following equation:
# score_alignment = W_combined * tanh(W_decoder * H_decoder + W_encoder * H_encoder)

# as you can see, its basically the decoders hidden state plus the encoders hidden state which are
# being used in a tanh transformation function. the weights are basically going to specify how much 
# importantce each hidden state has. 

# The decoder hidden state and encoder outputs will be passed through their individual 
# Linear layer(that is we use nn.Linear without a bias since it simply does a W*input!
# and makes life easier for us, without it, we should define a new parameter W and multiply it
# by the decoder hidden state, its the same thing! but uglier! so thats why we simply use 
# a linear layer as a learnable parameter for (W_decoder*H_decoder)) and have their own 
# individual trainable weights.

# Lastly, the resultant vector from the previous few steps will undergo matrix multiplication with 
# a trainable vector, obtaining a final alignment score vector which holds a score for each encoder
# output.

# Note: As there is no previous hidden state or output for the first decoder step, the last encoder 
# hidden state and a Start Of String (<SOS>) token can be used to replace these two respectively.

# 3. Softmaxing the Alignment Scores
# After generating the alignment scores vector in the previous step, we can then apply a softmax on this
# vector to obtain the attention weights. The softmax function will cause the values in the vector to
# sum up to 1 and each individual value will lie between 0 and 1, therefore representing the weightage
# each input holds at that time step.

# 4. Calculating the Context Vector
# After computing the attention weights in the previous step, we can now generate the context vector by
# doing an element-wise multiplication of the attention weights with the encoder outputs.
# Due to the softmax function in the previous step, if the score of a specific input element is closer
# to 1 its effect and influence on the decoder output is amplified, whereas if the score is close to 0,
# its influence is drowned out and nullified.

# 5. Decoding the Output
# The context vector we produced will then be concatenated with the previous decoder output. 
# It is then fed into the decoder RNN cell to produce a new hidden state and the process repeats itself
# from step 2. The final output for the time step is obtained by passing the new hidden state through a
# Linear layer, which acts as a classifier to give the probability scores of the next predicted word.

# seems a lot of people implement attention based on the version explained in this paper: 
# https://www.cs.cmu.edu/~hovy/papers/16HLT-hierarchical-attention-networks.pdf
# basically this only looks at the encoders hidden states and doesnt include decoders
# hidden state in calculating the score and creating the context vector! 
# https://mlwhiz.com/blog/2019/03/09/deeplearning_architectures_text_classification/?utm_campaign=shareaholic&utm_medium=reddit&utm_source=news
# this is an excellent blog post. highly recommened it 
# https://srome.github.io/Understanding-Attention-in-Neural-Networks-Mathematically/

# good attention intro (for rnns): https://www.youtube.com/watch?v=B3uws4cLcFw&list=PLgtf4d9zHHO8p_zDKstvqvtkv80jhHxoE
# Now lets build our model.
# Heres how we are going to implement it
# we are going to define a separate module for encoder
# and an attention_decoder module
# finally we will use them in our RNN module with these two!
# of course we can build all of them in a single model but the first way is neater
# for encoder and decoder, we are going to use embedding layers.
#%%
import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers=1, bidirectional=False, dropout=0.3):
        super().__init__()
        # our vocabulary size is used as embedding_size
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout
        
        # instead of one-hot encoding our input sequence, we use an embedding layer
        # to give us representations that are way more informative than a one-hot
        # encoded representation!
        # In some implementations you may see, the encoder
        # accepts hidden-size as input-size, and subsequently
        # the embedding dim also have to become hidden-size.
        # but here, we simply use an embedding_dim to decouple it from
        # input-size/sequence-length or the hidden-size.
        self.embedding = nn.Embedding(self.vocab_size, embedding_dim)
        
        # we need a rnn to process the input sequence
        # we use an lstm for its superior performance, 
        # the paper uses GRU!
        self.lstm = nn.LSTM(input_size=embedding_dim, 
                            hidden_size=self.hidden_size, 
                            num_layers=self.num_layers,
                            batch_first=True, 
                            dropout=self.dropout, 
                            bidirectional=self.bidirectional)

    def forward(self, x, h):
        x = self.embedding(x)
        return self.lstm(x,h)
# lets test 
# enc = Encoder(input_sequence_size=30, embedding_dim=75,vocab_size=50, hidden_size=100)
# xt = torch.randint(0,50, size=(5,30))
# outputs,hidden_state = enc(xt,None)
# print(f'{outputs.shape=}\n{hidden_state[0].shape=}\n{hidden_state[1].shape=}')
# good now lets implement our decoder with the attention mechanism

class BahdanauAttentionDecoder(nn.Module):
    def __init__(self, input_sequence_length, output_sequence_length, vocab_size, embedding_dim, hidden_size, num_layers, bidirectional=False, dropout=0.5) -> None:
        super().__init__()
        # input size or sequence length!
        self.input_seq_len = input_sequence_length
        # since this is a seq2seq model, the
        # output size can be different
        self.output_seq_len = output_sequence_length
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = nn.Dropout(dropout)
        
        # we need an embedding layer to prepare our input instead of one-hotencoding it
        # we need an rnn for decoding. we can use a lstmcell or a lstm layer, we use the
        # latter as it can be more powerful (add more layers, etc)
        # but we should have no issue replacing it with a lstm cell (test this when all is done)
        #
        # we need 3 weights. 
        #   one for decoders state, 
        #   one for encoders state and
        #   one for the weight scale(w_v)
        # we can use input_sequence_size, or hiddensize, but I want to use embedding-dim!
        # to decouple it from inputsize and hiddensize.
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        
        # our decoder's input will be the concatenated input_t and context_vector
        # both of which will have the shape (batchsize, timesteps) or (batchsize, seq_length)
        # we are using a LSTM instead of a LSTMCell, becasue it makes our life easier. 
        # if we use LSTMCell, we have to use a few extra reshapes which is unncessary really. 
        # using an lstm layer lke this, makes it a oneliner! easy peasy!
        # 
        # Note: theres a problem here, during our inference, when we want to feed our decoder
        # we have to always feed input_seq_length words, otherwise it would fail! note that this
        # was primarily used for machine translation and it makes sense, but for word generation
        # this causes an issue where we cant simply feed it a single word and get a single word
        # as output, so we may either prepad our inputs with spaces and place our initial word
        # at the end of the sequence and then feed it to the network and carry on during evaluation
        # time.
        self.decoder = nn.LSTM(self.input_seq_len+self.embedding_dim,
                               hidden_size=self.hidden_size,
                               num_layers=self.num_layers,
                               batch_first=True,
                               dropout=dropout, 
                               bidirectional=self.bidirectional,
                               )
        
        # #TODO disable after test 
        # self.decoder = nn.LSTM(self.embedding_dim,
        #                        hidden_size=self.hidden_size,
        #                        num_layers=self.num_layers,
        #                        batch_first=True,
        #                        dropout=dropout, 
        #                        bidirectional=self.bidirectional,
        #                        )
        # we also implement the lstmcell version as well, when we implemented everything
        # we enable this and add the lstmcell related changes (its only the decoder part
        # see the explanation in codes)
        # self.decoder = nn.LSTMCell(self.input_seq_len+self.embedding_dim,
        #                            hidden_size=self.hidden_size)
        
        # weights for decoders hidden_states
        # we use a linear layer without bias 
        # which does exactly what we want!
        self.W_decoder = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        # weights for encoders hidden_states 
        self.W_encoder = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        # weights for the scaling which is (1,hidden_size) 
        # becasue its multiplied by the result of the previous
        # weights, for this we use nn.Parameter to create a 
        # learnable wight/parameter, some implementation only 
        # use a single neuron/scaler for W_v
        # the 1, in the shape is just to remind us we do the right thing so the correct broadcasting happens
        # see the later notes
        self.W_v =  nn.Parameter(torch.ones(1,hidden_size))
        # final classification layer to give us the final results
        self.classifier = nn.Linear(hidden_size, output_sequence_length)
    
    
    def forward(self, input_sequence, encoder_states, hidden_state):
        # since we want to feed our input to decoder as well
        # we run embedding on the input_sequence as well
        # for all input timesteps, we repeat this process
        # print(f'input_sequence.shape:{tuple(input_sequence.shape)}')
        # print(f'hidden-size:{self.hidden_size}')
        # print(f'embedding dim:{self.embedding_dim}')
        input_sequence = self.embedding(input_sequence)
        # print(f'after embedding: {tuple(input_sequence.shape)}')
        
        # #TODO plain decoder test, disable after test
        # self.direction = 2 if self.bidirectional else 1
        # outputs, hidden_state = self.decoder(input_sequence, hidden_state)
        # outputs = self.classifier(outputs.reshape(-1, self.hidden_size*self.direction))
        # return outputs, hidden_state
    
        # remember we also need to grab outputs for each timestep
        # so for a given input_sequence, we return an output_sequence 
        # (here of the same size, but it could be different in general)
        # we do the classificantion in the loop
        outputs = []
        for t in range(input_sequence.size(1)):
            # grab one timestep from input
            # print(f'------------------------')
            # print(f'timestep: {t}/{input_sequence.size(1)}')
            input_t = input_sequence[:,t]
            # note that pytorch requires our input to be in the form
            # of (batch, timestep, input_dim) to treat it as a batched input
            # so we need to have a dim dedicated to the timestep, since this is
            # a single timestep, we simply make the dim and that suffices.
            # likewise the hiddenstate shape will have the form 
            # (num_layers*bidirection, Batchsize, hiddensize) which in our case
            # will be (1, batchsize, hiddensize) as we have a single layer and no
            # bidirection.
            input_t = input_t[:,None,:]
            # print(f'{input_t.shape=}')
            # print(f'{input_t=}')
            # TODO enable after test
            output_t, hidden_state = self.forward_attention(input_t, encoder_states, hidden_state)
            # TODO disable after test -(plain decoder test)
            # output_t, hidden_state = self.decoder(input_t, hidden_state)
            hidden_state = tuple(h.detach()for h in hidden_state)
            # output_t is (2,1,7), so we need the classifier to get
            # the output of size (bs,output_size).
            # the reshape makes it (2,7) which is compatible with our classifier
            output_t = self.classifier(output_t.view(output_t.size(0),-1))
            # print(f'--{output_t.shape=}')
            #
            # To grab the actual outputs we have two ways. one way is to stack
            # the outputs along timesteps and return the final output as (batch, ts, outputsize)
            # and then use crossentropy to calculate loss. 
            # the second way is to use logsoftmax, use max and grab the top indexes for each timestep
            # and return the final output which would be (batch, ts) and use nllloss to calculate the loss!
            # method 1:
            # simply append the outputs and at the end
            # stackthem in timestep dim and return it
            outputs.append(output_t)
            
            # method 2: 
            # # calculate softmax
            # output_t = torch.log_softmax(output_t, dim=-1)
            # # print(f'---{output_t=}')
            # # now grab the maximum value (most likely word/class)
            # _, output_t_idx = output_t.max(dim=-1)
            # # print(f'---{output_t_idx=}')
            # # store them so we can stack them later as one full output sequence
            # outputs.append(output_t_idx.detach())
            
        # stack the outputs along timestep dim to get (bs, output-seq-len/timesteps)
        # they can now be used to compare with labels 
        # we used log_softmax so we need to use nlloss instead 
        # (not sure if I can still use crossentropy)
        outputs = torch.stack(outputs, dim=1)
        # print(f'++++{outputs.shape=}')
        return outputs, hidden_state

    def forward_attention(self, input_sequence_t, encoder_states, hidden_state):
        # calculate attention raw score (attn = w_v + tanh(W_D+W_E))
        # do a softmax, make the rawscores into conditional probablities
        # multiply the new weight/scores by the attention-states, get the context vector
        # feed the context vector to the decoder wither alone or concatenated with input
        # at timestep t, calculate new outputs, hiddenstates and return the outputs 
        # and hidden_state for this timestep
        # grab the long-term memory(h)
        (h,c) = hidden_state
        # print(f'{h.shape=}')              # (1, batch, hidden-size)
        # print(f'{encoder_states.shape=}') # (batch, timesteps, hidden-size)
        # our hidden state has the shape (num-layers, batch, hidden-size), since our num_layers=1
        # its (1, batchsize, hiddensize) here.
        # while our encoder's outputs has (batch, timesteps, hidden-size)
        # we need to reshape our hiddenstate to have (batchsize, 1, hiddensize)
        # the easiest way is to simply transpose/permute it:
        h = h.permute(1,0,2)
        
        # print(f'{h.shape=}')
        # after this we can simply add the W_d and W_e together, the W_d will be broadcasted
        # (will be repeated along dim=1, and these two will be added properly)
        W_d = self.W_decoder(h)
        W_e = self.W_encoder(encoder_states)
        # atten = W_v * tanh(W_d + W_e)
        # print(f'{W_d.shape=}') # will be (batch, 1, hidden-size)
        # print(f'{W_e.shape=}') # will be (batch, ts, hidden-size)
        weights_added = torch.tanh(W_d + W_e)
        # print(f'{weights_added.shape=}') # (batch, ts, hidden-size)
        # print(f'{self.W_v.shape=}') #(1,hidden-size)
        # likewise our W_v shape is (1,hidden-size), in order to multiply it by
        # our weights_added, we need to make it compatible. 
        # making it (1,1,hidden-size) should do the trick, and it will be broadcasted 
        # accordingly to match (batch, ts, hidden-size).
        # the good thing is, we dont need to do sth like W_v.data.unsqueez_(0)! for this
        # (1,hidden-size) will be automatically broadcasted.
        # TODO method 1 :
        attention_score = self.W_v * weights_added # (batch, ts, hidden-size)
        
        # note that we have two ways to get to the same shape, but only one of them is correct.
        # if we do that the result will also be (batch,ts,hidden-size)
        # we could also achieve this shape by doing a matmul
        # TODO use method 2:
        # attention_score = torch.matmul(weights_added ,self.W_v.data.t())
        
        # print(f'{attention_score.shape=}') # (batch,ts,1)
        # and we get (batchsize, ts, 1). 
        # this means the context vector at the end will endup (batchsize,ts,hiddensize)
        # so which way is correct? 
        # lets think about our choices here
        # TODO: test both of these methods and rewrite this after test: 
        # The implications of using `torch.matmul(weights_added, W_v.t())` instead of 
        # the element-wise multiplication would be different in terms of the mathematical
        # operation and the resulting tensor shape.
        # In the case of element-wise multiplication (the previous way), the operation is 
        # applied element-wise between the broadcasted tensors, and the resulting tensor 
        # has the same shape as the larger tensor (`wights_added` in this case).
        # On the other hand, when using `torch.matmul(weights_added, W_v.t())`, it performs
        # a batched matrix multiplication between `weights_added` and the transposed weight
        # matrix `W_v.t()`. This operation has different implications:
        # 1. **Linear Transformation**: The matrix multiplication applies a 
        #      linear transformation to each batch in `wights_added` using the weight matrix
        #      `w_v`. 
        #      This is a common operation in neural networks, 
        #      where the weight matrix is used to transform the input data (`wights_added`) 
        #      into a different representation (e.g., applying a fully connected layer).
        # 2. **Dimensionality Reduction**: The resulting tensor has a reduced dimensionality 
        #      compared to the input tensor `wights_added`. Specifically, the last dimension 
        #      is reduced to 1, which means that each batch in the output tensor is a vector.
        #    - If `wights_added` has shape `(2, 5, 7)` and `w_v` has shape `(1, 7)`, the output
        #      tensor will have shape `(2, 5, 1)`.
        #    - This dimensionality reduction is often desired in neural networks, where the 
        #      output of one layer (e.g., a fully connected layer) is used as input to the 
        #      next layer.
        # 3. **Weight Sharing**: By using the same weight matrix `w_v` for all batches in 
        #      `wights_added`, the linear transformation is shared across all batches. 
        #       This is a common technique in neural networks, where the same set of weights
        #       is applied to different inputs (batches) during training and inference.
        # 4. **Potential for Non-linearity**: In neural networks, the output of a linear 
        #       transformation (like matrix multiplication) is often passed through a 
        #       non-linear activation function (e.g., ReLU, sigmoid) to introduce 
        #       non-linearity, which is essential for learning complex patterns.
        # So, if `w_v` is a weight matrix, using `torch.matmul(wights_added, w_v.t())` 
        # instead of element-wise multiplication suggests that we are performing a 
        # linear transformation on the input tensor `wights_added` using the weight matrix `w_v`.
        # This is a common operation in neural networks, where weight matrices are learned during
        # training to transform the input data into a desired representation.
        # However, it's important to note that the specific implications depend on the context 
        # and the architecture of our neural network model. 
        # The choice between element-wise multiplication and matrix multiplication should
        # align with the intended mathematical operation and the desired transformation of
        # the input data.
        #
        # torch.set_printoptions(profile='default')
        # softmax to ensure both nonnegativity and normalization. 
        attention_weights = attention_score.softmax(dim=-1)
        # print(f'{attention_weights.shape=}')
        # print(f'{weights=}')
        
        # Like before, we also have two ways of calculating context vector
        # we can have an elementwise multiplication and then sum over the 
        # hidden dim or simply do a batch matrix multiply, where we multiply these two
        # and the result is (batchsize, ts,ts) (and then sum so the last dim is reduced)
        # TODO test with this (enable this and disable the matmul version)
        # context_vector = attention_weights*encoder_states
        # context_vector = context_vector.sum(dim=-1)
        # print(f'{context_vector.shape=}')
        # note that context is a weighted sum of the encoder hidden-states
        # that is why we need to sum along the hiddenstate dimension to ultimately
        # get (batchsize, ts)
        
        # print(f'-{context_vector.shape=}')
        # TODO:‌ use bmm or batched matrix multiply (basically matmul()) it should
        # give us (batchsize, ts,ts) which if summed gives us (batchsize, ts)!
        # this maybe the right operation, as it carries a dot product between the two
        # matrixes (batch multiplication happens between the two)
        # while for the first approach, its an elemntwise multiplication and summation
        # basically scaling and summing the last dim. 
        context_vector = torch.matmul(attention_weights, encoder_states.permute(0,2,1)).sum(dim=-1)
        # print(f'{context_vector.shape=}')
        # and at this point we have a context vector that shows the attention
        # of each input sequence, we can feed this directly to decoder
        # or concatenate it with the input at timestep_t and then feed this new
        # input to decoder.
        
        # I have seen some people where they multiply context by the encoder-states 
        # and then use that to concatenate with the input. 
        # this is called multiplicative attention. by the way its different than 
        # loung attention! which we will see in a moment
        # 
        # states = context*encoder_states
        # print(f'{states.shape=}')
        
        # print(f'{input_sequence_t.shape=}')
        
        # we can now use this to feed out decoder , 
        # but usually we concatentate our input_sequence with this
        # context and then feed it to our decoder 
        # since our input needs to be in the form (batchsize, timestep, features), we 
        # make context_vector to have a timestep dim as well. 
        decoder_input = torch.cat([input_sequence_t, context_vector[:,None,:]],dim=-1)
        # print(f'{decoder_input.shape=}')
        # make the hiddenstate the (layer, batch, hiddenstate) instead of (batch,layer,hiddensatet)
        # h = h.permute(1,0,2)
        # print(f'{h.shape=}')
        # lstm decoder expects the hiddensize to be a tuple so we can either use (h,h)
        # or use the previous hiddenstate altogether, (h is our longterm memory and c 
        # is the short term memory!and we have been using the long term one)
        # sidenote: to test the lstmcell version, comment this part and uncomment the lstmcell section!
        # outputs, hidden_state = self.decoder(decoder_input, (h,h))
        outputs, hidden_state = self.decoder(decoder_input, hidden_state)
        
        ############################## changes for using lstmcell ###################
        # # when we are using lstmcell, the documenations says our input needs to be 2D,
        # # that is (batch, features). this is becasue a lstmcell process batches of data
        # # for each timestep consecutively/serially. but since we are already dealing with
        # # single timestep inputs, our input shape is (batch, 1, features). so by simply 
        # # removing the second dim (timestep dimemsion), we get our (batch, features) shape
        # # and we can go on and feed it to our lstmcell.
        # decoder_input = decoder_input.squeeze(1)
        # print(f'{decoder_input.shape=}')
        # # the hiddenstate also needs to be 2D
        # hidden_state = tuple(h.squeeze(0) for h in hidden_state)
        # # print(f'{hidden_state[0].shape=}')
        # print('hidden_state shapes: ',*[tuple(h.shape) for h in hidden_state])
        
        # # # now if we wanted, we could use a for loop to process the inputs and it would
        # # # look like this:
        # # outputs = []
        # # # remember if we are using a loop for processing batches for each timestep, 
        # # # we must make the input-sequence to have the shape (time_steps, batch, input_size)
        # # # sidenote, make sure to commentout the decoder_input.squeeze(1) part before continuing!
        # # # so the shapes match! 
        # # # also remember to comment out the next line where we use self.decoder() in a single call!
        # # decoder_input = decoder_input.permute(1,0,2)
        # # print(f'{decoder_input.shape=}')
        # # for i in range(decoder_input.size(0)):
        # #     print(decoder_input[i].shape)
        # #     hidden_state = self.decoder(decoder_input[i], hidden_state)
        # #     outputs.append(hidden_state[0])
        # # outputs = torch.stack(outputs,dim=0)
        # # # change outputs to have the shape (batch, tx, features)
        # # outputs = outputs.permute(1,0,2)
        
        # # but since, we are already using a single timestep input, we can simply do
        # # sidenote: remember to comment this out to test the loop-version!
        # hidden_state = self.decoder(decoder_input, hidden_state)
         
        # # outputs is the first hiddenstate, but we need to make it as (bs, tx, features)
        # # so we add a new dimension for the timestep at dim=1
        # outputs = hidden_state[0].unsqueeze(1)
        # print(f'{outputs.shape=}')
        # # we also change hiddenstate shape to match the hiddenstate of encoder
        # # so everything works out and we dont need to change anything else anywhere
        # hidden_state = tuple(h.unsqueeze_(0) for h in hidden_state)
        # ####################end of lstmcell required change#######################
        
        # print(f'{outputs.shape=}')
        # print(f'{hidden_state[0].shape=}')
        return outputs, hidden_state

seq_length = 5
embedding_dim = seq_length
hidden_size = 7
batch_size=2
vocab_size = 50
enc = Encoder(vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_size=hidden_size)
xt = torch.randint(0, vocab_size, size=(batch_size, seq_length))
h = None
outputs,hidden_state = enc(xt,h)
decoder = BahdanauAttentionDecoder(input_sequence_length=seq_length, 
                                   output_sequence_length=vocab_size, 
                                   vocab_size=vocab_size,
                                   embedding_dim=embedding_dim,
                                   hidden_size=hidden_size,
                                   num_layers=1)

outputs,hidden_state = decoder(xt,outputs,hidden_state)
print(f'{outputs.shape=}')
print(f'{hidden_state[0].shape=}')
#%%
# Now lets create the whole model 
# we'll keep it simple (we can use different values for encoder/decoder but here we use only
# one set for both)
class LSTMBahdanau(nn.Module):
    def __init__(self, input_size, output_size, vocab_size,
                 embedding_dim, hidden_size, enc_num_layers=1,
                 enc_dropout=0.3, dec_num_layers=1, dec_dropout=0.5):
        super().__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.enc_dropout = enc_dropout
        self.dec_dropout = dec_dropout
        self.enc_num_layers = enc_num_layers
        self.dec_num_layers = dec_num_layers
        
        self.encoder = Encoder(vocab_size=vocab_size,
                               embedding_dim=embedding_dim,
                               hidden_size=hidden_size,
                               dropout=enc_dropout,
                               num_layers=enc_num_layers)
        
        self.decoder = BahdanauAttentionDecoder(input_sequence_length=input_size,
                                                output_sequence_length=output_size,
                                                vocab_size=vocab_size,
                                                embedding_dim=embedding_dim,
                                                hidden_size=hidden_size,                 
                                                num_layers=dec_num_layers,dropout=dec_dropout
                                                )
        self.drp = nn.Dropout2d(0.1)
        
    def forward(self, x, h):
        enc_outputs, hidden_state = self.encoder(x,h)
        # TODO: decoder needs to work with encoders output, the initial input for decoder
        # should be the starting token, the start token + hidden_state from encoder should
        # do thetrick and start the chain 
        output,hidden_state = self.decoder(x, enc_outputs, hidden_state)
        # output = self.drp(output)
        return output,hidden_state

vocab_size=50
batch_size=4
seq_length = 5
embedding_dim=30
hidden_dim=100

xt = torch.randint(0, vocab_size, size=(batch_size, seq_length))
h = None

model = LSTMBahdanau(seq_length, vocab_size, vocab_size, embedding_dim, hidden_dim)
out,hidden_state = model(xt,h)
print(f'{out.shape=}')
print(f'{hidden_state[0].shape=}')
#! Remember to test different methods for decoder multiplication
#%############### correct way of implementing the bahdanu attention 
#%%
import os,sys
import numpy as np
import string
import urllib.request as request
import requests
import matplotlib.pyplot as plt
import torch.version
%matplotlib inline 

import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim  

print(f'{sys.version=}')#3.11.9
print(f'{torch.__version__=}')#2.4.0+cu121

class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers=1, bidirectional=False, dropout=0.3):
        super().__init__()
        # our vocabulary size is used as embedding_size
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout
        
        # instead of one-hot encoding our input sequence, we use an embedding layer
        # to give us representations that are way more informative than a one-hot
        # encoded representation!
        # In some implementations you may see, the encoder
        # accepts hidden-size as input-size, and subsequently
        # the embedding dim also have to become hidden-size.
        # but here, we simply use an embedding_dim to decouple it from
        # input-size/sequence-length or the hidden-size.
        self.embedding = nn.Embedding(self.vocab_size, embedding_dim)
        
        # we need a rnn to process the input sequence
        # we use an lstm for its superior performance, 
        # the paper uses GRU!
        self.lstm = nn.LSTM(input_size=embedding_dim, 
                            hidden_size=self.hidden_size, 
                            num_layers=self.num_layers,
                            batch_first=True, 
                            dropout=self.dropout, 
                            bidirectional=self.bidirectional)

    def forward(self, x, h):
        x = self.embedding(x)
        return self.lstm(x,h)
# lets test 
# enc = Encoder(input_sequence_size=30, embedding_dim=75,vocab_size=50, hidden_size=100)
# xt = torch.randint(0,50, size=(5,30))
# outputs,hidden_state = enc(xt,None)
# print(f'{outputs.shape=}\n{hidden_state[0].shape=}\n{hidden_state[1].shape=}')
# good now lets implement our decoder with the attention mechanism

class BahdanauAttentionDecoder(nn.Module):
    def __init__(self, input_sequence_length, output_sequence_length, vocab_size, 
                 embedding_dim, hidden_size, int2word, word2int,
                 num_layers, bidirectional=False, dropout=0.5) -> None:
        super().__init__()
        # input size or sequence length!
        self.input_seq_len = input_sequence_length
        # since this is a seq2seq model, the
        # output size can be different
        self.output_seq_len = output_sequence_length
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = nn.Dropout(dropout)
        self.int2word = int2word
        self.word2int = word2int
        # we need an embedding layer to prepare our input instead of one-hotencoding it
        # we need an rnn for decoding. we can use a lstmcell or a lstm layer, we use the
        # latter as it can be more powerful (add more layers, etc)
        # but we should have no issue replacing it with a lstm cell (test this when all is done)
        #
        # we need 3 weights. 
        #   one for decoders state, 
        #   one for encoders state and
        #   one for the weight scale(w_v)
        # we can use input_sequence_size, or hiddensize, but I want to use embedding-dim!
        # to decouple it from inputsize and hiddensize.
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        
        # our decoder's input will be the concatenated input_t and context_vector
        # both of which will have the shape (batchsize, timesteps) or (batchsize, seq_length)
        # we are using a LSTM instead of a LSTMCell, becasue it makes our life easier. 
        # if we use LSTMCell, we have to use a few extra reshapes which is unncessary really. 
        # using an lstm layer lke this, makes it a oneliner! easy peasy!
        # 
        # Note: theres a problem here, during our inference, when we want to feed our decoder
        # we have to always feed input_seq_length words, otherwise it would fail! note that this
        # was primarily used for machine translation and it makes sense, but for word generation
        # this causes an issue where we cant simply feed it a single word and get a single word
        # as output, so we may either prepad our inputs with spaces and place our initial word
        # at the end of the sequence and then feed it to the network and carry on during evaluation
        # time.
        
        # FIXME
        # our decoder input is the contanted input_t + context_vector
        # input_t is embd_dim becasue we use the output of embedding layer
        # context_vector is as large as the hidden_state for our encoder
        self.decoder = nn.LSTM(self.hidden_size + self.embedding_dim,#self.input_seq_len+self.embedding_dim,
                               hidden_size=self.hidden_size,
                               num_layers=self.num_layers,
                               batch_first=True,
                               dropout=dropout, 
                               bidirectional=self.bidirectional,
                               )
        
        # we also implement the lstmcell version as well, when we implemented everything
        # we enable this and add the lstmcell related changes (its only the decoder part
        # see the explanation in codes)
        # self.decoder = nn.LSTMCell(self.input_seq_len+self.embedding_dim,
        #                            hidden_size=self.hidden_size)
        
        # weights for decoders hidden_states
        # we use a linear layer without bias 
        # which does exactly what we want!
        self.W_decoder = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        # weights for encoders hidden_states 
        self.W_encoder = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        # weights for the scaling which is (1,hidden_size) 
        # becasue its multiplied by the result of the previous
        # weights, for this we use nn.Parameter to create a 
        # learnable wight/parameter, some implementation only 
        # use a single neuron/scaler for W_v
        # the 1, in the shape is just to remind us we do the right thing so the correct broadcasting happens
        # see the later notes
        self.W_v =  nn.Parameter(torch.ones(1,hidden_size))
        # or we could simply use a linear layer! and we would use it like attention_scores = self.W_v(weights_added)
        # self.W_v = nn.Linear(hidden_size, 1)
        
        # final classification layer to give us the final results
        self.classifier = nn.Linear(hidden_size, output_sequence_length)
        
        # self.start token
        self.start_token = torch.tensor([self.word2int['<sos>']],dtype=torch.long).view(1,1)
    
    
    def forward(self, input_sequence, encoder_states, hidden_state):
        # since we want to feed our input to decoder as well
        # we run embedding on the input_sequence as well
        # for all input timesteps, we repeat this process
        # print(f'input_sequence.shape:{tuple(input_sequence.shape)}')
        # print(f'hidden-size:{self.hidden_size}')
        # print(f'embedding dim:{self.embedding_dim}')
        #
        # note we dont need input embedding in decoder, its managed by encoder
        # input_sequence = self.embedding(input_sequence)
        # print(f'after embedding: {tuple(input_sequence.shape)}')
        # but instead we need a single timestep input, the start token 
        # grab its embedding to use it for initial input 
        # 
        # self.start token
        self.start_token = torch.tensor([self.word2int['<sos>'] for _ in range(input_sequence.size(0))], dtype=torch.long).view(-1,1)
        input_t = self.embedding(self.start_token.to(input_sequence.device))
        
        # print(f'{input_t.shape=}')
        # print(f'after embedding: {tuple(input_sequence.shape)}')
        
        # #TODO plain decoder test, disable after test
        # self.direction = 2 if self.bidirectional else 1
        # outputs, hidden_state = self.decoder(input_sequence, hidden_state)
        # outputs = self.classifier(outputs.reshape(-1, self.hidden_size*self.direction))
        # return outputs, hidden_state
    
        # remember we also need to grab outputs for each timestep
        # so for a given input_sequence, we return an output_sequence 
        # (here of the same size, but it could be different in general)
        # we do the classificantion in the loop
        outputs = []
        timesteps = input_sequence.size(1)
        for t in range(timesteps):
            # grab one timestep from input
            # print(f'------------------------')
            # print(f'timestep: {t}/{input_sequence.size(1)}')
            # we dont use input_sequence! it doesnt make sense, what would you do in eval mode then?
            # input_t = input_sequence[:,t]
            # so instead for the first time we use a start token, for the next timesteps we use the
            # output of the decoder from previous timestep.
            # note that pytorch requires our input to be in the form
            # of (batch, timestep, input_dim) to treat it as a batched input
            # so we need to have a dim dedicated to the timestep, since this is
            # a single timestep, we simply make the dim and that suffices.
            # likewise the hiddenstate shape will have the form 
            # (num_layers*bidirection, Batchsize, hiddensize) which in our case
            # will be (1, batchsize, hiddensize) as we have a single layer and no
            # bidirection.
            # input_t = input_t[:,None,:]
            # print(f'{input_t.shape=}')
            # print(f'{input_t=}')
            output_t, hidden_state = self.forward_attention(input_t, encoder_states, hidden_state)
           
            hidden_state = tuple(h.detach()for h in hidden_state)
            # output_t is (2,1,7), so we need the classifier to get
            # the output of size (bs,output_size).
            # the reshape makes it (2,7) which is compatible with our classifier
            output_t = self.classifier(output_t.view(output_t.size(0),-1))
            # update the input_t for the next round
            # get the highest probablity one
            output_t_top_probs = output_t.softmax(dim=-1)
            # print(f'{output_t_top_probs=}')
            output_t_final = output_t_top_probs.max(dim=-1)[1]
            # print(f'{output_t_final=}')
            # get its embedding and use it as the new input_t 
            # and remember to make its shape right (i.e. (b,ts,features))
            input_t = self.embedding(output_t_final)[:,None,:]

            # print(f'--{output_t.shape=}')
            #
            # To grab the actual outputs we have two ways. one way is to stack
            # the outputs along timesteps and return the final output as (batch, ts, outputsize)
            # and then use crossentropy to calculate loss. 
            # the second way is to use logsoftmax, use max and grab the top indexes for each timestep
            # and return the final output which would be (batch, ts) and use nllloss to calculate the loss!
            # method 1:
            # simply append the outputs and at the end
            # stackthem in timestep dim and return it
            outputs.append(output_t)
            
            # method 2: 
            # # calculate softmax
            # output_t = torch.log_softmax(output_t, dim=-1)
            # # print(f'---{output_t=}')
            # # now grab the maximum value (most likely word/class)
            # _, output_t_idx = output_t.max(dim=-1)
            # # print(f'---{output_t_idx=}')
            # # store them so we can stack them later as one full output sequence
            # outputs.append(output_t_idx.detach())
            
        # stack the outputs along timestep dim to get (bs, output-seq-len/timesteps)
        # they can now be used to compare with labels 
        # we used log_softmax so we need to use nlloss instead 
        # (not sure if I can still use crossentropy)
        outputs = torch.stack(outputs, dim=1)
        # print(f'++++{outputs.shape=}')
        return outputs, hidden_state

    def forward_attention(self, input_sequence_t, encoder_states, hidden_state):
        # calculate attention raw score (attn = w_v * tanh(W_D+W_E))
        # do a softmax, make the rawscores into conditional probablities
        # multiply the new weight/scores by the attention-states, get the context vector
        # feed the context vector to the decoder wither alone or concatenated with input
        # at timestep t, calculate new outputs, hiddenstates and return the outputs 
        # and hidden_state for this timestep
        # grab the long-term memory(h)
        (h,c) = hidden_state
        # print(f'{h.shape=}')              # (1, batch, hidden-size)
        # print(f'{encoder_states.shape=}') # (batch, timesteps, hidden-size)
        # our hidden state has the shape (num-layers, batch, hidden-size), since our num_layers=1
        # its (1, batchsize, hiddensize) here.
        # while our encoder's outputs has (batch, timesteps, hidden-size)
        # we need to reshape our hiddenstate to have (batchsize, 1, hiddensize)
        # the easiest way is to simply transpose/permute it:
        h = h.permute(1,0,2)
        
        # print(f'{h.shape=}')
        # after this we can simply add the W_d and W_e together, the W_d will be broadcasted
        # (will be repeated along dim=1, and these two will be added properly)
        W_d = self.W_decoder(h)
        W_e = self.W_encoder(encoder_states)
        # atten = W_v * tanh(W_d + W_e)
        # print(f'{W_d.shape=}') # will be (batch, 1, hidden-size)
        # print(f'{W_e.shape=}') # will be (batch, ts, hidden-size)
        weights_added = torch.tanh(W_d + W_e)
        # print(f'{weights_added.shape=}') # (batch, ts, hidden-size)
        # print(f'{self.W_v.shape=}') #(1,hidden-size)
        # likewise our W_v shape is (1,hidden-size), in order to multiply it by
        # our weights_added, we need to make it compatible. 
        # making it (1,1,hidden-size) should do the trick, and it will be broadcasted 
        # accordingly to match (batch, ts, hidden-size).
        # the good thing is, we dont need to do sth like W_v.data.unsqueez_(0)! for this
        # (1,hidden-size) will be automatically broadcasted.
        # !FIXME TODO: the calculation for attention score seems wrong,
        # !FIXME TODO: the shape of the context-vector must match the hidden_state
        # !FIXME TODO: therefore check this and fix it
        # TODO method 1 :
        # This is wrong, becasue it gives us the wrong attention_score (alignment score (alpha score!))
        # attention_score = self.W_v * weights_added # (batch, ts, hidden-size)
        
        # note that we have two ways to get to the same shape, but only one of them is correct.
        # if we do that the result will also be (batch,ts,hidden-size)
        # we could also achieve this shape by doing a matmul
        # TODO use method 2:
        # instead this is the way to do it it seems
        attention_score = torch.matmul(weights_added ,self.W_v.data.t())
        # attention_score = self.W_v(weights_added)
        print(f'{attention_score.shape=}') # (batch,ts,1) which means each timestep has a 
        # single weight
        # and we get (batchsize, ts, 1). 
        # this means the context vector at the end will endup (batchsize,ts,hiddensize)
        # so which way is correct? 
        # lets think about our choices here
        # TODO: test both of these methods and rewrite this after test: 
        # The implications of using `torch.matmul(weights_added, W_v.t())` instead of 
        # the element-wise multiplication would be different in terms of the mathematical
        # operation and the resulting tensor shape.
        # In the case of element-wise multiplication (the previous way), the operation is 
        # applied element-wise between the broadcasted tensors, and the resulting tensor 
        # has the same shape as the larger tensor (`wights_added` in this case).
        # On the other hand, when using `torch.matmul(weights_added, W_v.t())`, it performs
        # a batched matrix multiplication between `weights_added` and the transposed weight
        # matrix `W_v.t()`. This operation has different implications:
        # 1. **Linear Transformation**: The matrix multiplication applies a 
        #      linear transformation to each batch in `wights_added` using the weight matrix
        #      `w_v`. 
        #      This is a common operation in neural networks, 
        #      where the weight matrix is used to transform the input data (`wights_added`) 
        #      into a different representation (e.g., applying a fully connected layer).
        # 2. **Dimensionality Reduction**: The resulting tensor has a reduced dimensionality 
        #      compared to the input tensor `wights_added`. Specifically, the last dimension 
        #      is reduced to 1, which means that each batch in the output tensor is a vector.
        #    - If `wights_added` has shape `(2, 5, 7)` and `w_v` has shape `(1, 7)`, the output
        #      tensor will have shape `(2, 5, 1)`.
        #    - This dimensionality reduction is often desired in neural networks, where the 
        #      output of one layer (e.g., a fully connected layer) is used as input to the 
        #      next layer.
        # 3. **Weight Sharing**: By using the same weight matrix `w_v` for all batches in 
        #      `wights_added`, the linear transformation is shared across all batches. 
        #       This is a common technique in neural networks, where the same set of weights
        #       is applied to different inputs (batches) during training and inference.
        # 4. **Potential for Non-linearity**: In neural networks, the output of a linear 
        #       transformation (like matrix multiplication) is often passed through a 
        #       non-linear activation function (e.g., ReLU, sigmoid) to introduce 
        #       non-linearity, which is essential for learning complex patterns.
        # So, if `w_v` is a weight matrix, using `torch.matmul(wights_added, w_v.t())` 
        # instead of element-wise multiplication suggests that we are performing a 
        # linear transformation on the input tensor `wights_added` using the weight matrix `w_v`.
        # This is a common operation in neural networks, where weight matrices are learned during
        # training to transform the input data into a desired representation.
        # However, it's important to note that the specific implications depend on the context 
        # and the architecture of our neural network model. 
        # The choice between element-wise multiplication and matrix multiplication should
        # align with the intended mathematical operation and the desired transformation of
        # the input data.
        #
        # torch.set_printoptions(profile='default')
        # softmax to ensure both nonnegativity and normalization. 
        attention_weights = attention_score.softmax(dim=-1)
        print(f'{attention_weights.shape=}')
        # print(f'{weights=}')
        
        # Like before, we also have two ways of calculating context vector
        # we can have an elementwise multiplication and then sum over the 
        # hidden dim or simply do a batch matrix multiply, where we multiply these two
        # and the result is (batchsize, ts,ts) (and then sum so the last dim is reduced)
        # TODO test with this (enable this and disable the matmul version)
        # context_vector = attention_weights*encoder_states
        # context_vector = context_vector.sum(dim=-1)
        # print(f'{context_vector.shape=}')
        # note that context is a weighted sum of the encoder hidden-states
        # that is why we need to sum along the hiddenstate dimension to ultimately
        # get (batchsize, ts)
        
        # print(f'-{context_vector.shape=}')
        # TODO:‌ use bmm or batched matrix multiply (basically matmul()) it should
        # give us (batchsize, ts,ts) which if summed gives us (batchsize, ts)!
        # this maybe the right operation, as it carries a dot product between the two
        # matrixes (batch multiplication happens between the two)
        # while for the first approach, its an elemntwise multiplication and summation
        # basically scaling and summing the last dim. 
        # print(f'{encoder_states.shape=}')
        # ok note that doing element multiplication and then summing over and axis is wrong!
        # 
        # context_vector = torch.matmul(attention_weights, encoder_states.permute(0,2,1)).sum(dim=-1)
        # this is also wrong! where we multiply elementwise and then sum over dim=1!
        # context_vector = attention_weights * encoder_states
        # note we are summing over timesteps(dim=1) and not the last dim which is hiddenstate dims,
        # so ultimately we have a single hidden-state vector
        # that incorporates all previous hiddenstates, this becomes our context_vector
        # context_vector_weighted_sum = context_vector.sum(dim=1)
        # print(f'{context_vector_weighted_sum.shape=}') # should be (batch, 1, hidden_size)
        # and at this point we have a context vector that shows the attention
        # of each input sequence, we can feed this directly to decoder
        # or concatenate it with the input at timestep_t and then feed this new
        # input to decoder.
        
        # instead we should use batch-matrix multiplication which 
        attention_weights = attention_weights.permute(0,2,1)
        context_vector = torch.bmm(attention_weights,encoder_states)
        print(f'{context_vector.shape=}')
        
        
        # Further explanation: 
        #         Sure! Let's break down how batch matrix multiplication (`torch.bmm`) computes the weighted sum of the encoder states step by step.

        # ### Step-by-Step Explanation:

        # 1. **Inputs:**
        #    - `attention_weights`: A tensor of shape `(batch_size, seq_len, 1)` representing the attention weights for each time step in the sequence.
        #    - `encoder_states`: A tensor of shape `(batch_size, seq_len, hidden_size)` representing the hidden states of the encoder for each time step.

        # 2. **Permute Attention Weights:**
        #    - Before performing batch matrix multiplication, we need to permute the `attention_weights` to match the dimensions required for multiplication.
        #    ```python
        #    attention_weights = attention_weights.permute(0, 2, 1)
        #    ```
        #    - After permutation, `attention_weights` will have the shape `(batch_size, 1, seq_len)`.

        # 3. **Batch Matrix Multiplication:**
        #    - Perform batch matrix multiplication between the permuted `attention_weights` and `encoder_states`.
        #    ```python
        #    context_vector = torch.bmm(attention_weights, encoder_states)
        #    ```
        #    - Here, `torch.bmm` computes the matrix product for each batch. The resulting `context_vector` will have the shape `(batch_size, 1, hidden_size)`.

        # 4. **Weighted Sum:**
        #    - The batch matrix multiplication effectively computes the weighted sum of the encoder states. Each element in the `context_vector` is the sum of the encoder states weighted by the corresponding attention weights.
        #    - Mathematically, for each batch \(i\), the context vector \(c_i\) is computed as:
        #      \[
        #      c_i = \sum_{j=1}^{\text{seq_len}} \text{attention_weights}_{ij} \cdot \text{encoder_states}_{ij}
        #      \]
        #    - This operation is performed for each batch in parallel.

        # ### Example:
        # Let's consider a simple example with a batch size of 1, sequence length of 3, and hidden size of 2.

        # - `attention_weights` (after permutation):
        #   \[
        #   \begin{bmatrix}
        #   [0.2, 0.3, 0.5]
        #   \end{bmatrix}
        #   \]
        # - `encoder_states`:
        #   \[
        #   \begin{bmatrix}
        #   [h_{11}, h_{12}], [h_{21}, h_{22}], [h_{31}, h_{32}]
        #   \end{bmatrix}
        #   \]

        # - Batch matrix multiplication:
        #   \[
        #   \text{context_vector} = \begin{bmatrix}
        #   [0.2, 0.3, 0.5]
        #   \end{bmatrix} \times \begin{bmatrix}
        #   [h_{11}, h_{12}], [h_{21}, h_{22}], [h_{31}, h_{32}]
        #   \end{bmatrix}
        #   \]

        # - Resulting `context_vector`:
        #   \[
        #   \begin{bmatrix}
        #   [0.2 \cdot h_{11} + 0.3 \cdot h_{21} + 0.5 \cdot h_{31}, 0.2 \cdot h_{12} + 0.3 \cdot h_{22} + 0.5 \cdot h_{32}]
        #   \end{bmatrix}
        #   \]

        # This `context_vector` is the weighted sum of the encoder states, where the weights are given by the attention weights.

        # I hope this helps! If you have any more questions or need further clarification, feel free to ask!
        
        # I have seen some people where they multiply context by the encoder-states 
        # and then use that to concatenate with the input. 
        # this is called multiplicative attention. by the way its different than 
        # loung attention! which we will see in a moment
        # 
        # states = context*encoder_states
        # print(f'{states.shape=}')
        
        # print(f'{input_sequence_t.shape=}')
        
        # we can now use this to feed out decoder , 
        # but usually we concatentate our input_sequence with this
        # context and then feed it to our decoder 
        # since our input needs to be in the form (batchsize, timestep, features), we 
        # make context_vector to have a timestep dim as well. 
        # decoder_input = torch.cat([input_sequence_t, context_vector[:,None,:]],dim=-1)
        # FIXME: OK heres the issue, we dont feed the whole input_sequence to our decoder
        # since this process is happening in a loop, we feed a single input timestep
        # each time, the initial input of the decoder is usually a start token, (usually for translation)
        # for the next inputs, we use the output of our decoder from previous step.
        # 
        # print(f'{context_vector_weighted_sum[:,None,:].shape=}')
        decoder_input = torch.cat([input_sequence_t, context_vector_weighted_sum[:,None,:]],dim=-1)
        # print(f'{decoder_input.shape=}')
        # print(f'{hidden_state[0].shape=}')
        # make the hiddenstate the (layer, batch, hiddenstate) instead of (batch,layer,hiddensatet)
        # h = h.permute(1,0,2)
        # print(f'{h.shape=}')
        # lstm decoder expects the hiddensize to be a tuple so we can either use (h,h)
        # or use the previous hiddenstate altogether, (h is our longterm memory and c 
        # is the short term memory!and we have been using the long term one)
        # sidenote: to test the lstmcell version, comment this part and uncomment the lstmcell section!
        # outputs, hidden_state = self.decoder(decoder_input, (h,h))
        outputs, hidden_state = self.decoder(decoder_input, hidden_state)
        # print(f'{outputs.shape=}')
        # print(f'-----------------------------------')
        ############################## changes for using lstmcell ###################
        # # when we are using lstmcell, the documenations says our input needs to be 2D,
        # # that is (batch, features). this is becasue a lstmcell process batches of data
        # # for each timestep consecutively/serially. but since we are already dealing with
        # # single timestep inputs, our input shape is (batch, 1, features). so by simply 
        # # removing the second dim (timestep dimemsion), we get our (batch, features) shape
        # # and we can go on and feed it to our lstmcell.
        # decoder_input = decoder_input.squeeze(1)
        # print(f'{decoder_input.shape=}')
        # # the hiddenstate also needs to be 2D
        # hidden_state = tuple(h.squeeze(0) for h in hidden_state)
        # # print(f'{hidden_state[0].shape=}')
        # print('hidden_state shapes: ',*[tuple(h.shape) for h in hidden_state])
        
        # # # now if we wanted, we could use a for loop to process the inputs and it would
        # # # look like this:
        # # outputs = []
        # # # remember if we are using a loop for processing batches for each timestep, 
        # # # we must make the input-sequence to have the shape (time_steps, batch, input_size)
        # # # sidenote, make sure to commentout the decoder_input.squeeze(1) part before continuing!
        # # # so the shapes match! 
        # # # also remember to comment out the next line where we use self.decoder() in a single call!
        # # decoder_input = decoder_input.permute(1,0,2)
        # # print(f'{decoder_input.shape=}')
        # # for i in range(decoder_input.size(0)):
        # #     print(decoder_input[i].shape)
        # #     hidden_state = self.decoder(decoder_input[i], hidden_state)
        # #     outputs.append(hidden_state[0])
        # # outputs = torch.stack(outputs,dim=0)
        # # # change outputs to have the shape (batch, tx, features)
        # # outputs = outputs.permute(1,0,2)
        
        # # but since, we are already using a single timestep input, we can simply do
        # # sidenote: remember to comment this out to test the loop-version!
        # hidden_state = self.decoder(decoder_input, hidden_state)
         
        # # outputs is the first hiddenstate, but we need to make it as (bs, tx, features)
        # # so we add a new dimension for the timestep at dim=1
        # outputs = hidden_state[0].unsqueeze(1)
        # print(f'{outputs.shape=}')
        # # we also change hiddenstate shape to match the hiddenstate of encoder
        # # so everything works out and we dont need to change anything else anywhere
        # hidden_state = tuple(h.unsqueeze_(0) for h in hidden_state)
        # ####################end of lstmcell required change#######################
        
        # print(f'{outputs.shape=}')
        # print(f'{hidden_state[0].shape=}')
        return outputs, hidden_state

seq_length = 5
embedding_dim = 6
hidden_size = 7
batch_size=2
vocab_size = 50
enc = Encoder(vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_size=hidden_size)
xt = torch.randint(0, vocab_size, size=(batch_size, seq_length))
h = None
outputs,hidden_state = enc(xt,h)
decoder = BahdanauAttentionDecoder(input_sequence_length=seq_length, 
                                   output_sequence_length=vocab_size, 
                                   vocab_size=vocab_size,
                                   embedding_dim=embedding_dim,
                                   hidden_size=hidden_size,
                                   num_layers=1,
                                   int2word=int2word,
                                   word2int=word2int)

outputs,hidden_state = decoder(xt,outputs,hidden_state)
print(f'{outputs.shape=}')
print(f'{hidden_state[0].shape=}')
#%%
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        print(f'hiddenstate: {query.shape=}')
        print(f'(encoder-outputs: {keys.shape=}')
        # this does the summation at the end as well!! and gives us (b,1,ts)
        # instead of (ba,h, ts) (we had to sum over timesteps which seems wrong!!)
        # becasue later when we multiply by the weights (scores*outputs) we get the correct shape!(2,7)
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        print(f'{scores.shape=}')
        scores = scores.squeeze(2).unsqueeze(1)
        print(f'{scores.shape=}')
        
        weights = F.softmax(scores, dim=-1)
        print(f'{weights.shape=}')
        context = torch.bmm(weights, keys)
        print(f'{context.shape=}')
        return context, weights
    
SOS_token=0
MAX_LENGTH=5
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1,device='cpu'):
        super(AttnDecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = BahdanauAttention(hidden_size)
        self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)
        self.device=device

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=self.device).fill_(SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []

        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions


    def forward_step(self, input, hidden, encoder_outputs):
        embedded =  self.dropout(self.embedding(input))

        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        input_gru = torch.cat((embedded, context), dim=2)

        output, hidden = self.gru(input_gru, hidden)
        output = self.out(output)

        return output, hidden, attn_weights

seq_length = 5
embedding_dim = 6
hidden_size = 7
batch_size=2
vocab_size = 50
enc = Encoder(vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_size=hidden_size)
xt = torch.randint(0, vocab_size, size=(batch_size, seq_length))
h = None
outputs,hidden_state = enc(xt,h)

decoder = AttnDecoderRNN( hidden_size=hidden_size,output_size=vocab_size)
decoder2 = BahdanauAttentionDecoder(input_sequence_length=seq_length, 
                                   output_sequence_length=vocab_size, 
                                   vocab_size=vocab_size,
                                   embedding_dim=embedding_dim,
                                   hidden_size=hidden_size,
                                   num_layers=1,
                                   int2word=int2word,
                                   word2int=word2int)

outputs1,hidden_state1,_ = decoder(outputs,hidden_state[0])
outputs2,hidden_state2 = decoder2(xt,outputs,hidden_state)
print(f'{outputs1.shape=}')
print(f'{hidden_state1[0].shape=}')
print(f'{outputs2.shape=}')
print(f'{hidden_state2[0].shape=}')

#%%
# Now lets create the whole model 
# we'll keep it simple (we can use different values for encoder/decoder but here we use only
# one set for both)
class LSTMBahdanau(nn.Module):
    def __init__(self, input_size, output_size, vocab_size, int2word, word2int,
                 embedding_dim, hidden_size, enc_num_layers=1,
                 enc_dropout=0.3, dec_num_layers=1, dec_dropout=0.5):
        super().__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.enc_dropout = enc_dropout
        self.dec_dropout = dec_dropout
        self.enc_num_layers = enc_num_layers
        self.dec_num_layers = dec_num_layers
        self.int2word = int2word
        self.word2int = word2int
        
        self.encoder = Encoder(vocab_size=vocab_size,
                               embedding_dim=embedding_dim,
                               hidden_size=hidden_size,
                               dropout=enc_dropout,
                               num_layers=enc_num_layers)
        
        self.decoder = BahdanauAttentionDecoder(input_sequence_length=input_size,
                                                output_sequence_length=output_size,
                                                vocab_size=vocab_size,
                                                embedding_dim=embedding_dim,
                                                hidden_size=hidden_size,                 
                                                num_layers=dec_num_layers,dropout=dec_dropout,
                                                int2word=self.int2word,
                                                word2int=self.word2int)
        self.drp = nn.Dropout2d(0.1)
        
    def forward(self, x, h):
        enc_outputs, hidden_state = self.encoder(x,h)
        # TODO: decoder needs to work with encoders output, the initial input for decoder
        # should be the starting token, the start token + hidden_state from encoder should
        # do thetrick and start the chain 
        output,hidden_state = self.decoder(x, enc_outputs, hidden_state)
        # output = self.drp(output)
        return output,hidden_state

#%%
# now lets grab a text and try to test our architecture. 
# we wanto keep it simple, and generate text, the same thing we did before
# but this time we are going to use words instead of characters. 
# previously we chose characters, becasue that would result in a small onehot encoded vector
# if we used words for example, and we wanted to onehot encode them, that would be a several thousands
# elements vectors, that would take a huge amount of vram and computation! but now we plan on using
# an embedding layer (we will see later on how a typical emebedding vector works)
# which takes way less memory and computation and is much much more efficient in terms of capturing 
# underlying relationships between words, etc. unlike one-hot-encoding, we can specify any dimension size
# for our embedding vector (if it was one-hot-encoding, we had to use vocab_length for each vector
# and the majority of that vector is just zero which is extremely inefficient). we'll see all of this
# in a moment
# note that since we are using words, we face new challanges based 
def download_ebook(url, file_name='corpus.txt', dir_name = 'data'):
    file_name_path = os.path.join(dir_name, file_name)
    
    if os.path.exists(file_name_path):
        return file_name_path
    
    if not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)
        
    request.urlretrieve(url, file_name_path)
    return file_name_path

url = 'http://www.gutenberg.org/files/1399/1399-0.txt'
# lets download and read it all!
# this time around lets grab all the words. we use split() to split the words
# based on white characters ('\n,\r,\t,\f)
with open(download_ebook(url),'r') as file: 
    corpus_words = file.read().split()

print(f'{len(corpus_words)=:,}') # 352,804 words!
print(f'{repr(corpus_words[:10])=}')
# as you can see, there are 352,804 words, most of which are duplicates.
# lets get rid of them and grab the unique ones.
# lets also make them all lowercase, this way we remove all variantions 
# of a word and only keep the simple lowercase form
corpus_words_lower = [word.lower() for word in corpus_words]
print(f'{len(corpus_words_lower)=:,}') # 352,804 words!
print(f'{repr(corpus_words_lower[:10])=}')
# now lets grab the unique ones if we use set(), we will get a 10x reduction
# i.e. we would get 28,151 words, but it will also destroy our input text structure!
# we want to retain the word order, so we can't simply use set() like when we used characters!
# corpus_words_unique = [word for word in set(corpus_words_lower)]
# print(f'{len(corpus_words_unique)=:,}') # 28,151 words!
# print(f'{repr(corpus_words_unique[:10])=}')
# we do this part when we want to create the int2word and word2int dictionaries.
# 
# recall that we are trying to create a dictionary for converting words to integer codes and vice versa
# so our preprocessing shouldnt alter the text structure, in doing so we lose some degree of cleanness
# like the words that are attached to puntuation marks, (like "child,", "hoped.", etc would not be removed
# which is not desigrable, but it suffices for now we try to just keep it simple despite having these issues
# we'll see how to get aroundthis issue later on).
# 
# note that if we were to do the same preprocessings we did on characters, on words here, 
# we would face a lot of issues, like for example our simple tokenization wouldnt handle a lot of 
# cases such as different tenses.
# verbs in different tenses or expressions would be destroyed by our simplistic preprocessing.
# like if we removed the unwanted-characters! we would have butched many words and expressions! 
# this has a direct impact on the final result. 
# in the future when we start working with transformers, we see how to get around this 
# (we will be familiarized with tokenization and how its done in the process. 
# for now lets accept this as a good enough result and carry on!)
#%%
# now lets create our word2int and int2word dictionaries
# note that we can use set to grab the unique words, here!
# this will give us a 10x reduction! i.e. 28,151 words!
# we start from 1 because we want to reserve 0 for special token <sos>
int2word = dict(enumerate(set(corpus_words_lower),start=1))
word2int = {w:c for c,w in int2word.items()}

# lets add the special tokens as well like sos
int2word[0] = '<sos>'
word2int['<sos>'] = 0

print(f'{len(int2word)=:,}')
print(f'{len(word2int)=:,}')
print(f'{int2word=}')
print(f'{word2int=}')
# lets convert our corpus to int
words_digitized = torch.tensor([word2int[word] for word in corpus_words_lower])
print(f'{words_digitized.shape=}')
print(words_digitized[:10])
print('after conversion: ')
# print(repr(' '.join([int2word[idx.item()] for idx in words_digitized[:10]])))
# lets make that into a function!
def convert_to_words(seq_int_list):
    return ' '.join([int2word[idx] for idx in seq_int_list])

print(convert_to_words(words_digitized[:10].tolist()))

#%%
url = 'http://www.gutenberg.org/files/1399/1399-0.txt'
# lets download and read it all!
with open(download_ebook(url),'r') as file: 
    corpus_raw = file.read()

print(f'{repr(corpus_raw[:10])=}')
corpus_raw = corpus_raw.translate(str.maketrans('','',string.punctuation))
# we can now filter out the other escape characters!
# by only selecting the printable ones!
corpus_raw = ''.join(x for x in corpus_raw if x in set(string.printable))
# and finally making all characters lower case
corpus_words_lower = ''.join([c.lower() for c in corpus_raw])
# now we get 'the projec' this time around!
print(repr(corpus_words_lower[:10]))

unique_chars = set(corpus_words_lower)
# lets sort it for better visualization
unique_chars = sorted(unique_chars)
# now lets create a char2int and int2char dictionaries!
# start from 1, becasue we reserve 0 for special token (sos)
int2word = dict(enumerate(unique_chars, start=1))
word2int = {c:d for d,c in int2word.items()}

# lets add the special tokens as well like sos
int2word[0] = '<sos>'
word2int['<sos>'] = 0

print(f'{unique_chars=}')
print(f'{int2word=}')
print(f'{word2int=}')
print(f'unique characters: {len(unique_chars)} : \n {unique_chars}')
# lets convert our corpus to int
words_digitized = torch.tensor([word2int[char] for char in corpus_words_lower])
print(f'{words_digitized.shape=}')
print(words_digitized[:10])
print('after conversion: ')
print(repr(''.join([int2word[idx.item()] for idx in words_digitized[:10]])))


#%%
# dataset needs work! we only have 14 words for our dataset? this cant be right!
# what to do now?
# now we need to have batches! 
def get_next_batch(words_digitized, batch_size=1, seq_len=10):
    char_count = words_digitized.size(0)
    each_batch_size = batch_size*seq_len
    batch_count = char_count // each_batch_size
    corpus = words_digitized[:batch_count * each_batch_size]
    corpus = corpus.reshape(batch_size, -1)
    # print(f'{corpus.shape=}')
    # note the dtype and our warning previously. now we have a much larger vocabsize
    # so our tensor must accomadate way more numbers than 256 of uint8! if you are not
    # careful and use the wrong/insufficient dtype here, you'll face a lot of headache later on
    # because of overflow:)). change this to uint8 and run this to see the difference 
    x = torch.zeros(size=(batch_size, seq_len), dtype=torch.long)
    y = torch.zeros_like(x)
    for i in range(0, corpus.size(1), seq_len):
        x[...] = corpus[:, i:i+seq_len]
        try :
            y[:, :-1] = x[:, 1:]
            y[:, -1] = corpus[:, i+seq_len]
        except:
            y[:, :-1] = x[:, 1:]
            y[:, -1] = corpus[:, 0]
        yield x,y

x,y = next(iter(get_next_batch(words_digitized, batch_size=3, seq_len=8)))
print(f'{x.shape=}')
print(f'{y.shape=}')
print(f'{x=}')
print(f'{y=}')
#training
#%%
# lets test our newtork 
seq_len = 5
data, labels = next(iter(get_next_batch(words_digitized, batch_size=2, seq_len=seq_len)))
print(f'{data.shape=}')
print(f'{labels.shape=}')

num_layers = 1
model = LSTMBahdanau(input_size=seq_len,
                     output_size=len(word2int),
                     vocab_size=len(word2int),
                     embedding_dim=100,
                     hidden_size=100,
                     int2word=int2word,
                     word2int=word2int)

print(f'our input(data).shape: {data.shape}')
outputs,hidden_state = model(data,None)
print(f'model input size: {model.input_size}')
print(f'model output size: {model.output_size}')
# now our output may look weird sth like, remember that
# in order to get meaningful output we need to reshape it 
print(f'rnn output shape: {outputs.shape}')
print(f'{outputs[:,:,:3]=}')
# therefore the actual shape is 
# print(f'output actual shape :{outputs.view(-1, seq_len, model.output_size).shape}')
print(*[convert_to_words(output_lst.tolist()) for output_lst in outputs.max(dim=-1)[1] ], sep='\n')

#%%
# now lets train our model
seq_len = 30
input_size = seq_len
# remember our output size is the same as
# the vocab_size, that is, any word in the vocab
# can be a likely valid output, see it as the number
# of classes we have in output (each word represents a single class)
output_size = len(word2int)
vocab_size = len(word2int)
embedding_dim = 100
hidden_size = 512
# layers_cnt = 1
# bidirection = False
device = 'cuda' if torch.cuda.is_available()  else 'cpu'
# device='cpu'
model = LSTMBahdanau(input_size=input_size,
                     output_size=output_size,
                     vocab_size=vocab_size,
                     embedding_dim=embedding_dim,
                     hidden_size=hidden_size,
                     int2word=int2word,
                     word2int=word2int)

model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr = 0.001)
criterion = nn.CrossEntropyLoss()#ignore_index=-100
scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=30)

epochs =30
# in order to not face the exploding gradient in lstm
# we clip the gradients
clip = 5.
interval = 1000
batch_size = 32
# label_length = len(word2int)
hidden_states = None

val_ratio = 0.2
val_idx = int(words_digitized.numel() * (1-val_ratio))
train = words_digitized[:val_idx]
val = words_digitized[val_idx:]

print(f'running on device: {device}')
print(f'input_size:     {input_size}')
print(f'output_size:    {output_size:,}')
print(f'hidden_size:    {hidden_size}')
print(f'embedding_dim:  {embedding_dim}')
print(f'vocab_size:     {vocab_size:,}')
print(f'word count:     {words_digitized.numel():,}')
print(f'val idx:        {val_idx:,}')
print(f'val size:       {val.numel():,}')
print(f'train size:     {train.numel():,}')
print(f'val + train:    {val.numel() + train.numel():,}')
assert train.numel() + val.numel() == words_digitized.numel() ,'they must be equale!'

for e in range(epochs):
    model.train()
    total_loss = 0
    for i, (data, label) in enumerate(get_next_batch(train, batch_size, seq_len=seq_len), start=1):
        
        # label is not one-hot-encoded, crossentropy can do this on its own
        data = data.to(device)
        label = label.to(device).long()
        # print(f'{data.shape=} {label.shape=}')
        output , hidden_states = model(data, hidden_states)
          
        hidden_states = tuple(h.data for h in hidden_states)
     
        # print(f'{label.shape=}')  # label.shape=torch.Size([32, 30])
        # print(f'{output.shape=}') # output.shape=torch.Size([32, 30, 28151])
        # label = label.view(batch_size*seq_len).long()
        # print(f'{label=}')
        loss = criterion(output.view(-1,vocab_size), label.view(-1))
            
        total_loss += loss.item()
        # print(f'{total_loss=}')
        
        optimizer.zero_grad()
        loss.backward()
        # note the _, which indicates the inplace operation!
        # like before this doesnt seem to matter much really! 
        # training loss decreases however the validation
        # lossincreases after some epochs! get this to work!
        torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=5.)
        optimizer.step()

        if i%interval==0:
            print(f'Epoch-Iter:: {e}/{epochs}-{i} | Loss: {total_loss/i:.4f} | LR: {scheduler.get_lr()[-1]:.6f}')
    
    # decay the lr per epochs
    scheduler.step()

    # test 
    hidden_states = None
    total_loss_val = 0
    for i, (data,label) in enumerate(get_next_batch(val, batch_size, seq_len),start=1):
        with torch.no_grad():
            model.eval()

            data = data.to(device)
            label = label.to(device).long()

            output, hidden_states = model(data, hidden_states)
            
            hidden_states = tuple(h.data for h in hidden_states)
            
            total_loss_val += criterion(output.view(-1,vocab_size), label.view(-1)).item()
            
            if i % interval ==0:
                print(f'  Loss-val: {total_loss_val/i:.4f}')
    print(f'  Loss-val-total: {total_loss_val/i:.4f}')
#...
# Epoch-Iter:: 56/60-100 | Loss: 0.8325 | LR: 0.000100
# Epoch-Iter:: 56/60-200 | Loss: 0.7869 | LR: 0.000100
#   Loss-val-total: 11.0800
# Epoch-Iter:: 57/60-100 | Loss: 0.8294 | LR: 0.000100
# Epoch-Iter:: 57/60-200 | Loss: 0.7825 | LR: 0.000100
#   Loss-val-total: 11.0806
# Epoch-Iter:: 58/60-100 | Loss: 0.8234 | LR: 0.000100
# Epoch-Iter:: 58/60-200 | Loss: 0.7790 | LR: 0.000100
#   Loss-val-total: 11.0897
# Epoch-Iter:: 59/60-100 | Loss: 0.8194 | LR: 0.000100
# Epoch-Iter:: 59/60-200 | Loss: 0.7761 | LR: 0.000100
#   Loss-val-total: 11.0936
# 

# TODO:
# validation goes up! training goes down!
# next attempt should be to disable attention and only 
# use an ordinary lstm and see how it performs!
# using characters only instead of words ran smoothly, both losses decreased without any issues!
# making me believe we face massive overfitting in our model when using words (as the number
# of classes/words are in thousands while when using chars its only 38!)
# 
# Epoch-Iter:: 57/60-1000 | Loss: 0.5964 | LR: 0.000100
#   Loss-val-total: 0.7419
# Epoch-Iter:: 58/60-1000 | Loss: 0.5949 | LR: 0.000100
#   Loss-val-total: 0.7413
# Epoch-Iter:: 59/60-1000 | Loss: 0.5944 | LR: 0.000100
#   Loss-val-total: 0.7415

# the text generation is awful, not sure its becasue of the generation procedure or the model
# TODO fix the generation / test with plain lstm and see how it goes
# plain lstm (encoder-plain decoder)
# Epoch-Iter:: 56/60-1000 | Loss: 1.1627 | LR: 0.000100
#   Loss-val-total: 1.3113
# Epoch-Iter:: 57/60-1000 | Loss: 1.1625 | LR: 0.000100
#   Loss-val-total: 1.3111
# Epoch-Iter:: 58/60-1000 | Loss: 1.1623 | LR: 0.000100
#   Loss-val-total: 1.3115
# Epoch-Iter:: 59/60-1000 | Loss: 1.1621 | LR: 0.000100
#   Loss-val-total: 1.3118
# the loss is much higher than the attention version, it didnt decrease as rapidly as the attention
# version, and the text generation is aweful nonetheless
# TODO use single classification instead of looping for plain lstm
# Epoch-Iter:: 57/60-1000 | Loss: 0.9309 | LR: 0.000100
#   Loss-val-total: 1.0851
# Epoch-Iter:: 58/60-1000 | Loss: 0.9295 | LR: 0.000100
#   Loss-val-total: 1.0852
# Epoch-Iter:: 59/60-1000 | Loss: 0.9281 | LR: 0.000100
#   Loss-val-total: 1.0849
#loss decreased when I used classification in a single call but the generation is still nonsensical and gibrish
# TODO I guess it could be the way we are generating the text, next change the way we generate the text
# first start by trainig sequence of 1 and testing eval like before, then use a nn.linear 
# to create an intermediate representation before feeding it to the decoder, but we need to
# somehow make it work with different channel numbers! thats the issue! here when sampling
#
# sequence 1 result: 
# Epoch-Iter:: 57/60-45000 | Loss: 1.8432 | LR: 0.000100
#   Loss-val-total: 1.9065
# Epoch-Iter:: 58/60-45000 | Loss: 1.8459 | LR: 0.000100
#   Loss-val-total: 1.9096
# Epoch-Iter:: 59/60-45000 | Loss: 1.8457 | LR: 0.000100
#   Loss-val-total: 1.9089
# text generation doesnt seem complete giberish anymore, they have a bit of structure, but
# not alot, could be due to insufficient training/higher loss, notet hat we used the old sampler
# that uses single character each time. 
# TODO: train a bit more and see if it helps with the output, if so, then switch to attention
# and training with seq=1 and see how it performs, then decide on the next move
# Epoch-Iter:: 86/90-45000 | Loss: 1.8145 | LR: 0.000100
#   Loss-val-total: 1.8954
# Epoch-Iter:: 87/90-45000 | Loss: 1.8144 | LR: 0.000100
#   Loss-val-total: 1.8954
# Epoch-Iter:: 88/90-45000 | Loss: 1.8144 | LR: 0.000100
#   Loss-val-total: 1.8954
# Epoch-Iter:: 89/90-45000 | Loss: 1.8143 | LR: 0.000100
#   Loss-val-total: 1.8954
# more training didnt change the loss that much, but the text generation seems a tiny bit better
# you can see words better formed, verbs better formed but the grammar, structure is still missing
# TODO next lets try with attention with seq of 1 
# ok its obismally bad! 
#TODO fix the attention mechanism. we shouldnt be feeding the input to the decoder like this
# changed the code to fix the attention mechanism part 1: 
# Epoch-Iter:: 86/90-1000 | Loss: 2.7969 | LR: 0.000100
#   Loss-val-total: 2.8033
# Epoch-Iter:: 87/90-1000 | Loss: 2.7969 | LR: 0.000100
#   Loss-val-total: 2.8029
# Epoch-Iter:: 88/90-1000 | Loss: 2.7966 | LR: 0.000100
#   Loss-val-total: 2.8028
# Epoch-Iter:: 89/90-1000 | Loss: 2.7971 | LR: 0.000100
#   Loss-val-total: 2.8033
# the loss decreased, but the convergence is slow, it need more training
# TODO next disable the gradient clipping part/try to train more with higher lr
# the gradient clipping removal didnt do anything, but lowering the lr to 0.001 resulted in
# much faster convergence and much lower loss both during training and validation:
# this was trained for 90 epochs which got worse and model diverged after epoch 30
# Epoch-Iter:: 28/90-1000 | Loss: 0.3809 | LR: 0.001000
#   Loss-val-total: 0.5979
# Epoch-Iter:: 29/90-1000 | Loss: 0.3810 | LR: 0.001000
#   Loss-val-total: 0.5953
# Epoch-Iter:: 30/90-1000 | Loss: 0.2737 | LR: 0.000010
#   Loss-val-total: 0.5775
# Epoch-Iter:: 31/90-1000 | Loss: 0.1969 | LR: 0.000100
#   Loss-val-total: 0.5857
# Epoch-Iter:: 32/90-1000 | Loss: 0.1596 | LR: 0.000100
#   Loss-val-total: 0.6120
# Epoch-Iter:: 33/90-1000 | Loss: 0.1384 | LR: 0.000100
#   Loss-val-total: 0.6330
# so I traind for the second time this time for 30 epochs only 
# Epoch-Iter:: 26/30-1000 | Loss: 0.4237 | LR: 0.001000
#   Loss-val-total: 0.6812
# Epoch-Iter:: 27/30-1000 | Loss: 0.4257 | LR: 0.001000
#   Loss-val-total: 0.6520
# Epoch-Iter:: 28/30-1000 | Loss: 0.4038 | LR: 0.001000
#   Loss-val-total: 0.6898
# Epoch-Iter:: 29/30-1000 | Loss: 0.3875 | LR: 0.001000
#   Loss-val-total: 0.6460
# the loss is very low, but the text generation is nonsense, all the classes are 1!
# which shouldnt happen, and therefore it keeps generating <sos> for all timesteps/characters
# TODO find and fix the issue 
#%%

def predict_nextword(model, inputs, hidden_states=None, topk=5,device='cuda'):
    model.eval()
    
    unique_chars = len(int2word)
    # print(char2int)
    # convert input string into corrosponding ids and add a batch dim
    print(f'{inputs=}')
    input_tensor =  torch.tensor([word2int[w] for w in inputs]).reshape(1,-1).to(device).long()
    # print(f'{input_tensor.shape=}')
    
    output, hidden_states = model(input_tensor, hidden_states)
    output = output.softmax(dim=1)
    # now our output has probabilities for each sequence/timestep
    # we will choose the highest one here 
    probs, indexes = output.topk(k=topk, dim=1)
    indexes = indexes.cpu().data.squeeze()
    probs = probs.cpu().data.squeeze()
    # print(f'{output=}')
    # print(f'{probs.shape=}')
    # print(f'{indexes.shape=}')
    probs = probs.view(-1)
    indexes = indexes.view(-1)
    # if we were to use numpy, we would have to write it like this, 
    # use indexes, and also provide the probablities for these 
    # char = np.random.choice(indexes.numpy(),p=probs.numpy()/probs.numpy().sum())
    # note we have to renormalize the probs so that each of these new probablities
    # also note that, we sample from the indexes, each index represents a character
    # so sampling from them means choosing between different characters based on their
    # probablities here.
    if probs.numel()==1: # if topk=1
        word = indexes.item()
    else:
        word = indexes[torch.multinomial(probs/probs.sum(dim=-1), num_samples=1, replacement=True)].item()
    return int2word[word], hidden_states

def sample(model, size=10, prompt='hello there',topk=5, seq_len=30):
    
    # chars = [ch.lower() for ch in starter_message]
    h = None
    prompt = prompt.lower()
    # add the prompt/start message
    # preprend spaces so it becomes seq_len big
    prompt_prepended = ' '*(seq_len - len(prompt.split(' ')))+prompt
    # words = prompt_prepended.split()
    words = prompt_prepended
    print(f'after prepending spaces: {len(words)}, {words}')
    
    # now append the new generated text 
    # by first feeding the whole prompt/starter message to 
    # condition the model, and then the append the last output
    # (which is what we want) to the words list
    # feed in sequences of seq_len big (30)
    # for w in words:
    #     o,h = predict_nextword(model, w, h, topk=topk)
    # words.append(w)

    # now start from the last newly generated character
    # we just got from previous loop, use it to generate
    # new characters, as many as the size dictates.
    # words[-1] grabs the last freshly generated character
    # from previous attempt and feeds it to the network to
    # generate new one. the new one is then appended to the
    # chars list and this continues until we have generated
    # the right amount of characters specified by size
    print(words[-len(words):30])
    for i in range(size):
        text = words[-seq_len:]
        # print(f'-- text fed: {-len(words)=} {text=}')
        o, h = predict_nextword(model, text, h, topk=topk)
        words+= o
        if words[0]==' ':
            words=words[0:]

    # finally we convert all into a big string
    return ' '.join(words)#words[seq_len-1:]
# unique_chars = len(int2word)
# print(unique_chars)
outputz = sample(model, size=30, prompt='\n',topk=5)
print(repr(outputz))
#%%
def predict(model, input, hidden_states=None, topk=5):
    model.eval()
    # int2word = int2word
    # word2int = word2int
    unique_chars = len(int2word)
    # print(char2int)
    # convert input string into corrosponding ids and add a batch dim
    input =  torch.tensor([word2int[input]]).reshape(-1,1).to('cuda')
    # input =  torch.tensor([word2int['<sos>']]).reshape(-1,1).to('cuda')
    output, hidden_states = model(input, hidden_states)

    output = output.softmax(dim=1)
    # print(f'{output=}')
    # now our output has probabilities for each sequence/timestep
    # we will choose the highest one here 
    probs, indexes = output.topk(k=topk, dim=1)
    indexes = indexes.cpu().data.squeeze()
    probs = probs.cpu().data.squeeze()
    # print(f'{probs.shape=}')
    # print(f'{indexes.shape=}')
    # if we were to use numpy, we would have to write it like this, 
    # use indexes, and also provide the probablities for these 
    # char = np.random.choice(indexes.numpy(),p=probs.numpy()/probs.numpy().sum())
    # note we have to renormalize the probs so that each of these new probablities
    # also note that, we sample from the indexes, each index represents a character
    # so sampling from them means choosing between different characters based on their
    # probablities here.
    if probs.numel()==1: # if topk=1
        char = indexes.item()
    else:
        char = indexes[torch.multinomial(probs/probs.sum(dim=-1), num_samples=1, replacement=True)].item()
    return int2word[char], hidden_states

def sample(model, size=10, prompt='hello there',topk=5):
    
    # chars = [ch.lower() for ch in starter_message]
    h = None
    prompt = prompt.lower()
    # add the prompt/start message
    chars = list(prompt)
    # print(unique_chars)
    # now append the new generated text 
    # by first feeding the whole prompt/starter message to 
    # condition the model, and then the append the last output
    # (which is what we want) to the chars list
    for ch in prompt:
        o,h = predict(model, ch, h, topk=topk)
    chars.append(ch)

    # now start from the last newly generated character
    # we just got from previous loop, use it to generate
    # new characters, as many as the size dictates.
    # chars[-1] grabs the last freshly generated character
    # from previous attempt and feeds it to the network to
    # generate new one. the new one is then appended to the
    # chars list and this continues until we have generated
    # the right amount of characters specified by size 
    for _ in range(size):
        o, h = predict(model, chars[-1], h, topk=topk)
        chars.append(o) 
        
    # finally we convert all into a big string
    return ''.join(chars)
outputz = sample(model, size=30, prompt='the projec',topk=1)
print(repr(outputz))

#%%
# #this is basically a decoder part -old should be deleted
# class BahdanauAttention(nn.Module):
#     def __init__(self):
#         super().__init__()
        
#         # todo: 
#         # i believe the lstm is decoder here, but I beleive it should have been lstm cell, which processes
#         # decoders input for each timestep and goes on untill all tokens are generated.
#         # with lstm, we feed the decoders input, get the whole output(all next hiddenstates) based on 
#         # this single input which I guess can be also correct in a way, but not entirly due to paper.
#         # also if we use lstm cell, we should not do softmax on the output, as thats only for the final
#         # layer to calculate the actual outputs for the last timestep (that is when the last timestep is
#         # reached, we get all previous hidden-states and do a softmax to get class probablities for each
#         # timestep.)
#         # but with current implementation, we use decoder-inputs, calculate outputs for all timesteps
#         # but only save the current timestep, then go for a second round, use the final hiddenstate of our
#         # decoder(lstm second output), and feed it to attention cell as the decoders previous hidden-state
#         # which is then used in the attention mechanisim with the encoders hiddenstate and the input values
#         # so its not wrong I guess becasue its doing it for each step!
#         # , but we should be able to do this using lstm cell as well. lets see
#         self.lstm = nn.LSTM(input_size=hidden_size * 2, hidden_size=hidden_size,num_layers=num_layers,
#                            batch_first=True)

#     # this forward, we assume , we get inputs of length 1 sequence 
#     # that means, our inputs are fed one timestep at a time 
#     # thats why we are using the lstm layer and it works!
#     # for this to work, we must have a loop in our training procesure
#     # where we feed one timestep at a time. 
#     def attention_cell(self, x_t, hidden_states_t, encoder_outputs):

#         # x_t = self.embedding(x_t)
#         # since lstm has two states, h and c, we only care about h!
#         (h_prev, c) = hidden_states_t
#         # lets first calculate the score, we use the previous hiddenstate
#         # of the decoder, which can be none or the last state of the encoder
#         # As we have already read there is no previous hidden state or output 
#         # for the first decoder step, the last encoder hidden state and a 
#         # Start Of String (<SOS>) token can be used to replace these
#         # two, respectively.

#         D = self.W_d(h_prev)
#         print(f'h_prev shape: {h_prev.shape}')
#         print(f'encoders.shape: {encoder_outputs.shape}')
#         print(f'W_e shape: {self.W_e.weight.shape}')
#         encoder_outputs = encoder_outputs.reshape(encoder_outputs.size(0),-1)
#         print(f'encoders.shape: {encoder_outputs.shape}')
#         E = self.W_e(encoder_outputs)
        
#         score_raw_part1 = F.tanh(D + E)
#         print(f'tanh(w_d*h_prev + w_e*encoders): {score_raw_part1.shape}')
#         # scale it , we use bmm which is batchmatrixmultiplication 
#         score_raw = score_raw_part1.bmm(self.W_v.unsqueeze(2))
#         print(f'score raw(W_v*tanh()): {score_raw.shape}')

#         # normalize it using softmax and get our attention weights
#         # these will be multiplied by our encoders states
#         weights = F.softmax(score_raw, dim=1)
#         print(f'attention weights: {weights.shape}')
#         # now create context vector which is attention_weights * encoder outputs(states)
#         # Multiplying the Attention weights with encoder outputs to get the context vector
#         # since we are dealing with not a sample but several samples at the same time, we
#         # use bmm. 
#         context_vector = torch.bmm(weights.unsqueeze(0), encoder_outputs.unsqueeze(0))
#         print(f'context_vector: {context_vector.shape}')

#         # Concatenating context vector with embedded input word
#         decoder_input = torch.cat((x_t, context_vector[0]), 1).unsqueeze(0)
#         print(f'output(concat(context_vector[0],x_t)): {decoder_input.shape}')
#         # Passing the concatenated vector as input to the LSTM cell
#         output, hidden = self.lstm(decoder_input, hidden)
#         # Passing the LSTM output through a Linear layer acting as a classifier
#         output = F.log_softmax(self.classifier(output[0]), dim=1)
#         return output, hidden, weights

#     def forward(self, x, hidden_states, encoder_outputs):

#         x = self.embedding(x)
#         print(f'x.shape after embedding:  {x.shape}')
#         # x = x.view(1, -1)
#         # print(f'x.shape after view(1,-1) : {x.shape}')

#         # we can set it to none 
#         # hidden_states = None

#         # since lstm has two states, h and c, we only care about h!
#         # x.shape = (batch, timestep, dims)
#         O = torch.zeros_like(x)
#         H = []
#         A = []
#         for t in range (x.size(1)):
#             O[:,t,:], hidden_states, weights = self.attention_cell(x[:,t,:], hidden_states,encoder_outputs)
#             H.append(hidden_states)
#             A.append(weights)
#         return O, torch.tensor(H), torch.tensor(A)

# # ok lets test our implementation so far and see if it works well
# # first lets create a dummy input 
# input_np = np.array([[1,2,3],[0,4,1]])
# input = one_hot(input_np,length=5)
# # print(f'input_one hot encoded: \n{input}')
# encoder = Encoder(5)
# model = BahdanauAttention(5)
# # batch, n_layer*bid
# input = torch.from_numpy(input)
# print(f'input: {input.shape}')
# d_input = torch.from_numpy(input_np).long()

# encoder_outputs, encoder_last_hidden_states = encoder(d_input, None)


# print(f'encoder output: {encoder_outputs.shape}')
# print(f'h from encoder(shape (n_layers * num_dir, batch, hidden_size)) \n{encoder_last_hidden_states[0].shape}')
# yz = model(d_input, encoder_last_hidden_states, encoder_outputs)

# print('results: ')
# print(f'input {input}')
# print(f'decoder output: {yz}')
#%%
# lets implement this one first and then we implement the luoungs version :






#%% sentiment analysis 
# lets do sentiment analysis . we read a bunch of reviews with their corrosponding labels
# here are the steps we need to take
# we know our label contains words, positive and neagative, so we convert them into numbers, 1, 0
# our reviews must be dgitized so we can feed them into our network  , but before that we need
# to do some preprocessings. the preprocessings include 
# 1. make everything lower case -not needed really
# 2. remove punctuations 
# 3. remove special characters such as \n 
# 4. split only words! we dont want to create text, so creating dictionaries of characters 
# as apposed to creating dictionary of words is not going to suit us, becasue we intend on 
# using word embedding, and relationship between words need to be found out! this is not possible
# with characters! 
# 5. thats nearly basically it(there are still some left), 
# we need to create two dictionaries for converting word2int and in2words ()
# there is one thing to note that, we start our intergers from 1 and not 0. we will be using 0 
# later on for padding the input so we can have batches of the same size. 
# 6.so we order our words based on  their frequency! the most frequent wants get to the top and
# the least freqyent one goes to the bottom of the list. 
# 7. an important step is to just normalize/standardize the length. since we want to use batch
# we need to pick a length. we cant use the bigest one, becasue we may waste alot of space 
# so we search and remove the larges and smallest ones. 
# 8. good lets go 
import numpy as np 
import string
import torch
import torch.nn as nn 
import torch.nn.functional as F 
import matplotlib.pyplot as plt
import torch.optim as optim  
%matplotlib inline 

# first lets read our files 
with open('/media/hossein/SSD1/code_dl/reviews.txt','r') as file: 
    reviews_raw = file.read().lower()
with open('/media/hossein/SSD1/code_dl/labels.txt', 'r') as file: 
    labels_raw = file.read().lower()
# lets see what we have here 
print(repr(reviews_raw[:2000]))
print(repr(labels_raw[:20]))
# ok, good, now lets remove punctuations and \ns
reviews_raw = ''.join([c for c in reviews_raw if c not in string.punctuation])
print(repr(reviews_raw[:2]))

# ok now lets remove \ns
reviews_raw_split = reviews_raw.split('\n')
# print(repr(reviews_raw[:200]))
# ok good, now lets create our wordlist from reviews
reviews_words_list = ''.join(reviews_raw_split).split()
print(reviews_words_list[:20])
#ok good, now lets get each words frequency and also sort them based on that 
# first get each words frequency in our list. we use Counter from collections for that
from collections import Counter
words_dict = Counter(reviews_words_list)
print(words_dict['the'])
# now lets sort these from highest to lowest 
words_sorted_list = sorted(words_dict,key=words_dict.get,reverse=True)
# print(words_dict_sorted)
int2word = dict(enumerate(words_sorted_list,1))
word2int = {word:idx for idx,word in int2word.items()}
print(f'int2word[1] {int2word[1]}')
print(f'word2int["the"] {word2int["the"]}')
# now lets digitize our label 
labels = [1 if word=='positive' else 0 for word in labels_raw.split('\n')]
print(f'labels[:50](raw): \n{labels_raw[:54]}')
print(f'labels[:50]: {labels[:6]}')

# now lets create the digitized version of the reviews 
reviews_digitized=[]
for each_review in reviews_raw_split:
    # create a temp list of words separated by space for each review by doing .split()
    reviews_digitized.append([word2int[w] for w in each_review.split()])

print(len(reviews_digitized))
print(reviews_raw_split[:2])
print(reviews_digitized[:2])
#%%
# ok the frequency is clear, they are sorted, lets find the biggest and smallest reviews
max_len = max([len(rev) for rev in reviews_digitized])
min_len = min([len(rev) for rev in reviews_digitized])
dic_rev_length = Counter(len(rev) for rev in reviews_digitized)
print(f'max length: {max_len} and min_len : {min_len}')
print(dic_rev_length[2])
print(f'max: {max(dic_rev_length)}')
print(f'min: {min(dic_rev_length)}')
print(f'max count {dic_rev_length[max_len]}')
print(f'min count {dic_rev_length[min_len]}')

# lets remove the ones with zero index 
# we dont this, since we also need to remove the corrosponding labels
# so we instead get the index and remove the reviews based on their index
# new_reviews = [review for review in reviews_digitized if len(review)!=0]
# new_reviews2 = [review for review in new_reviews if len(review)!=2514]
idxs = [idx for idx,review in enumerate(reviews_digitized) if len(review)==min_len]
print(f'idxs: {idxs[:10]}')
idxs += [idx for idx,review in enumerate(reviews_digitized) if len(review)==max_len]
new_reviews = [review for idx,review in enumerate(reviews_digitized) if idx not in (idxs)]
print(f'idxs: {idxs[:10]}')
print(len(reviews_digitized))
print(len(new_reviews))
# lets remove the labels 
new_labels = [label for idx, label in enumerate(labels) if idx not in (idxs)]
print(new_labels[:10])
#%%
# Ok now lets create a function for padding our input
# we do this so we can create batches and increase the performance
# Pytorch provides its own functions 
# anyway lets create a function that gets the digitized review and returns a 
# padded numpy array . we define a maximum length , and fill it from the end
def pad_input(new_reviews, max_length =200):
    padded_array = np.zeros(shape=(len(new_reviews), max_length),dtype=np.int32)
    for i,review in enumerate(new_reviews) : 
        padded_array[i,-len(review):] = review[:max_length]
    return padded_array 

reviews_digitized = pad_input(new_reviews, 150)
print(reviews_digitized[:1])
#%%
training_frac = 0.80
tr_idx = int(reviews_digitized.shape[0] * training_frac)
training_set, remaining_set = reviews_digitized[:tr_idx,:], reviews_digitized[tr_idx:,:]
val_frac = 0.5
val_idx = int(remaining_set.shape[0]*val_frac)
val_set = remaining_set[:val_idx,:]
test_set = remaining_set[val_idx:,:]
print(training_set.shape)
print(val_set.shape)
print(test_set.shape)

# now labels! 
training_set_label, remaining_set_label = np.array(new_labels[:tr_idx]), np.array(new_labels[tr_idx:])
val_set_label = remaining_set_label[:val_idx]
test_set_label = remaining_set_label[val_idx:]

print(training_set_label.shape)
print(val_set_label.shape)
print(test_set_label.shape)
print(training_set_label[:2])
# now lets go for the training 
# we can use the torchTensorDataset for this 

import torch 
import torch.utils.data 
import torch.nn as nn 
import torch.nn.functional as F 


batch_size = 50
# we feed our numpy data and its corrosponding labels , and it will create us a dataset!
train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(training_set),torch.from_numpy(training_set_label))
test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(test_set),torch.from_numpy(test_set_label))
val_dataset = torch.utils.data.TensorDataset(torch.from_numpy(val_set),torch.from_numpy(val_set_label))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,shuffle=False)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,shuffle=False)

# view a sample batch !

features, labels=next(iter(train_loader))
print(features.shape)
print(features[:2,:30])
print(labels[:2])
#%%
class SentimentLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,
                       embedding_dim=500, num_layers=1, dropout_ratio =0.5):
        super().__init__()

        vocab_size = input_size
        self.embedding = nn.Embedding(num_embeddings = vocab_size, embedding_dim = embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first = True, 
                            dropout=dropout_ratio)
        self.fc = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout2d(0.3)
        self.num_layers = num_layers

    def forward(self, x, hidden):
        x= x.long()
        # print(f'input: {x.shape}')
        embeddings = self.embedding(x)
        # if hidden != None:
        #     print(f'hiddens.shape: {hidden[0].shape} {hidden[1].shape}')   
        # print(f'input_embeddings: {embeddings.shape}')
        outputs, hidden = self.lstm(embeddings, hidden)

        outputs = self.dropout(outputs)
        # print(outputs.shape)
        outputs = outputs.contiguous().view(-1, self.hidden_size)
        # we want 0 or 1 
        outputs = F.sigmoid(self.fc(outputs))
        # print(f'before reshape: outputs.shape: {outputs.shape}')
        # lets make the output batch_first again 
        outputs = outputs.view(x.size(0), -1)
        # print(f'after reshape:  outputs.shape: {outputs.shape}')
        # print(f'outputs[:,-1].shape: {outputs[:,-1].shape}')
        # print(outputs.shape)
        # we need the last sequence output, so we get the last one using -1
        return outputs[:,-1], hidden

    def init_weights(self, batch_size, device):

        weight = next(self.parameters()).data
        hidden_states = (weight.new_zeros(self.num_layers,batch_size,self.hidden_size).to(device),
         weight.new_zeros(self.num_layers,batch_size,self.hidden_size).to(device))
        return hidden_states


# Instantiate the model w/ hyperparams
vocab_size = len(word2int)+1 # +1 for the 0 padding + our word tokens
output_size = 1
embedding_dim = 500 # embedding size, is important!
hidden_dim = 300 # this is important as well!
# with embed 1000, hd 300 : 
n_layers = 3#78.68 with 1 layer, 80% with 2 layers and 81.88 with 3 layers 

model = SentimentLSTM(input_size=vocab_size, hidden_size =hidden_dim,
                     output_size=output_size,embedding_dim=embedding_dim,num_layers=n_layers,
                      dropout_ratio=0.5)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

features, labels=next(iter(train_loader))
features = features.to(device)
print(model)
x,y = model(features, None)

criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 4 
val_interval = 100
clip_threshold = 5
counter=0
hidden_states = model.init_weights(batch_size,device)
for e in range(epochs):
    hidden_states = model.init_weights(batch_size,device)
    for features, labels in train_loader:
        model.train()
        #batch counter!
        counter+=1

        features = features.to(device)
        labels = labels.to(device)
  
        outputs, hidden_states = model(features, hidden_states)
        # print(f'hiddens.shape: {hidden_states[0].shape} {hidden_states[1].shape}')   
        hidden_states = tuple(h.data for h in hidden_states)
        
        optimizer.zero_grad()
        loss = criterion(outputs, labels.float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_threshold)
        optimizer.step()

        if counter%val_interval == 0: 
            hidden_states = model.init_weights(batch_size, device)
            val_losses = []
            for feats, labels in val_loader:
                with torch.no_grad():

                    model.eval()

                    feats = feats.to(device)
                    labels = labels.to(device)
                    outputs, hidden_states = model(feats, hidden_states)
                    # print(f'eval: hiddens.shape: {hidden_states[0].shape} {hidden_states[1].shape}')   
                    hidden_states = tuple(h.data for h in hidden_states)

                    val_loss = criterion(outputs, labels.float())
                    val_losses.append(val_loss.item())

            print('epoch: {}/{}'.format(e,epochs))
            print('loss: {}'.format(loss.item()))
            print('val-loss: {}'.format(np.mean(val_losses)))



#%%
#now lets test this on our test set, here calculate accuracy and loss@
hidden_states = model.init_weights(batch_size, device)
test_losses =[]
num_corrects = 0
i=0  

model.eval()
for features, labels in test_loader:
    with torch.no_grad():
        i+=1
        features = features.to(device)
        labels = labels.to(device)
        outputs, hidden_states = model(features, hidden_states)
        hidden_states = tuple(hidden.data for hidden in hidden_states)
        loss = criterion(outputs, labels.float())
        test_losses.append(loss.item())
        #acc
        preds = torch.round(outputs)
        if i==0: 
            print(f'preds.shape: {preds.shape}, labels.shape {labels.shape}')

        # results = preds.eq(labels.float().view_as(preds))
        # result = np.squeeze(results.cpu().numpy())
        # num_corrects += np.sum(result)
        results = (preds==labels.float()).sum()
        num_corrects += results.item()
        
        #print(num_corrects.item())

print(f'acc= {(num_corrects/len(test_loader.dataset))*100.0} %')
print(f'loss: {np.mean(test_losses)}')


#%%
# here test this on your own data! this is called inference ! give it some random text and see 
# if it can correctly classify it as positive or negative! 

def tokenize_input(review='damn you son of a gun. that was hell!'):
    from string import punctuation
    #first lower case all words 
    review = review.lower()
    #second remove all punctuations
    all_text_no_punc = ''.join(c for c in review if c not in punctuation)
    #third get all the words 
    words_list = all_text_no_punc.split()

    digitized=[]
    digitized.append(np.array([word2int[w] for w in words_list]))

    return digitized

test_review_neg = 'The worst movie I have seen; acting was terrible and I want my money back. This movie had bad acting and the dialogue was slow.'

review_digitized = tokenize_input(test_review_neg)
print(review_digitized)

review_padded = pad_input(review_digitized, 200)
print(review_padded.shape)

review_padded = torch.from_numpy(review_padded).to(device)
hidden_states = model.init_weights(1, device)
outputs, hidden_states = model(review_padded, hidden_states)

pred = torch.round(outputs)

if pred.item() == 0:
    print('negative')
else:
    print('postive')
#%%

# word embedding 
# for word embedding training we have several methods, word2vec is one of them
# here we will be using the skipgram model. we can train skipgram model with negative sampling
# we will implement both! 
# the skipgram model is simply an embedding layer with an fc layer followed by a logsoftmax
# the skipgram model with negative sampling, is two embedding layers, the embedding layer for input
# and output must be thesame, we feed uor word to input embedding get target words, feed those
# target words to output embedding and must get back the initial word that was fed into input 
# emebdding.  
#
# The first thing that we do is we need a body of text that we can use as our dataset 
# and learn the embeddings from. lets use the text8 in data dirctory
with open(r'data\text8','r') as file : 
    corpus_raw = file.read()
# now we have all the contents which include, words and punctuations and white spaces
# since we want to learn proper embeddings for 'words' we can remove the punctuations
# altogether. we also can remove less frequent words. we may also want to remove 'some'
# (emphasize on 'some') common and uncommon words as well. why? becasue words such as the, a
# are very common, but they dont really provide much insight to the surrounding words,
# so removing them can help us achieve better embeddings as these noises are removed. 
# how do we do that? we use a mikolove formula for that which we will get to shortly.
# but first lets do : 
# 1. remove punctuations , actually replacing them with proper symbol
# 2. remove less frequent words 
# 3. remove some common/uncommen words based on mikolove criteria
# removing punctuations 
def remove_punctuations(input_corpus):
    input_corpus = input_corpus.replace('.','<PERIOD>')
    input_corpus = input_corpus.replace('!','<EXCLAMATION>')
    input_corpus = input_corpus.replace('(','<LPAR>')
    input_corpus = input_corpus.replace(')','<RPAR>')
    input_corpus = input_corpus.replace('[','<LBRAC>')
    input_corpus = input_corpus.replace(']','<RBRAC>')
    input_corpus = input_corpus.replace('#','<HASH>')
    input_corpus = input_corpus.replace("'",'<SingleQuote>')
    input_corpus = input_corpus.replace('"','<DoubleQuote>')
    input_corpus = input_corpus.replace(':','<COLON>')
    input_corpus = input_corpus.replace('$','<DOLLAR>')
    input_corpus = input_corpus.replace('%','<PERCENT>')
    input_corpus = input_corpus.replace(';','<SEMICOLON>')
    input_corpus = input_corpus.replace('-','<DASH>')
    return input_corpus

corpus = remove_punctuations(corpus_raw)
# now remove less frequent words 
# sort the word list 
# create word2int int2word 
# create subsampling 
import math, random # used for sqrt and random respectively 
word_dic = Counter(corpus.split())
word_dic = {word:freq for word,freq in word_dic.items() if freq>5}
word_list = sorted(word_dic, key=word_dic.get, reverse=True)
# should be 'the'
print(word_list[0])
print(len(word_list))
# create word2int and int2word dicts
int2word = dict(enumerate(word_list))
word2int = {word:idx for idx,word in int2word.items()}
# now lets do subsampling, we remove some common and uncommon words. using
# mikolov formula w = sqrt(t/word_freq)
# lets calculate word frequencies 
temp_dic = Counter(word_list)
word_frq_dict = {word:1-math.sqrt(freq/len(word_list)) for word, freq in temp_dic.items()}
threshold = 1e-5 
word_list = [word for word in word_list if random.random() < word_frq_dict[word]]

print(word_list[0])
print(len(word_list))
# ok now we need to get the target words for each word. we define a function that 
# accepts a input list, index, windows size 
def get_target(word_list, idx , window_size=5):
    random_len = random.randint(1, window_size)
    start_idx = idx - random_len if (idx - random_len) > 0 else 0
    end_idx = idx + random_len if (idx + random_len) <len(word_list) else len(word_list)

    before_words = word_list[start_idx:idx]
    after_words = word_list[idx+1:end_idx+1]
    return before_words + after_words

# lets test 
get_target('Hello brother, howdy?', idx=10,window_size=5)
# now we need to create a batching mechanism
#test again
string = [i for i in range(10)]
idx = random.randint(1,5)
window = 5
print(f'input : {string}')
print (f'idx: {idx} window: {window}')
targets = get_target(string,  idx,  window)
print(targets)
# get digitized word
word_list = [word2int[word] for word in word_list]
# lets create a batch retriever 
def get_batch(word_list, batch_size=10, window_size=5):

    word_cnt = len(word_list)
    total_batches_cnt = word_cnt//batch_size
    word_lists = word_list[:total_batches_cnt * batch_size]
    # we need a X and Y which contains each xs targets 

    for idx in range(0, len(word_lists), batch_size):
        batch = word_lists[idx: idx + batch_size]
        X,Y = [],[] 
        for i in range(len(batch)):
            x = batch[i]
            y_target = get_target(batch, i, window_size)
            X.extend([x]*len(y_target))
            Y.extend(y_target)
        yield X, Y


w,t = next(iter(get_batch(word_list,3)))
print(w)
print(t)

# now lets define cosine similarity 
def cosine_similarity(word2int, embedding_layer, word, topk=5):
    # get word index
    word_idx = word2int[word]
    # get word embedding
    embeddings = embedding_layer(word_idx)
    embeddings = embeddings.unsqueeze(0) # add a batch dimension
    # now cosine similarity is word embedding 
    embeddings = torch.LongTensor(embeddings)
    magnitutes = embedding_layer.weight.pow(2).sum(dim=1).sqrt().unsqueeze()
    similarity = torch.mm(embeddings, embedding_layer.weight.t())/magnitutes 

    return similarity 

# lets create a cosine similarity for validation words, to see how certain words
# are doing. we create some random words, and take their cosine simlarity in the 
# embeddings. if their target words are plausible then we are good! lets do this 
import numpy as np 
def cosine_similarity_validation(word2int, embedding_layer, validation_size, window_size=100):
    # first lets create some random word indexes 
    # we get some common words and some uncommon words. if you recall, we sorted
    # our words based on their frequencies, so that the most frequent ones stay 
    # atop and less frequent ones stay at the bottom. 

    # random.sample(sequence, k)
    # Parameters:
    # sequence: Can be a list, tuple, string, or set.
    # k: An Integer value, it specify the length of a sample.
    common_words_idx = np.array( random.sample(range(0,window_size),validation_size//2) )
    uncommon_words_idx = np.array(random.sample(range(2000,2000+window_size),validation_size//2))

    val_words = common_words_idx + uncommon_words_idx
    val_words = torch.LongTensor(val_words)
    embeddings = embedidng_layer(val_words.unsqueeze(0))
    magnitutues = embedding_layer.weight.pow(2).sum(dim=1).sqrt().unsqueeze()

    similarity = torch.mm(embeddings,embedding_layer.weight.t())/magnitutes 

    return val_words, similarity 

# now ok. its time to create our model for embedding learning using skipgram! model
class SkipGram(nn.Module):
    def __ini__(self, vocab_size, embedding_size=300):
        super().__init__()

        self.embedding_layer = nn.Embedding(vocab_size, embedding_size)
        self.fc = nn.Linear(embedding_size, vocab_size)
        
    def forward(self, x):
        x = self.embedding_layer(x)
        x = self.fc(x)
        log_probs = F.log_softmax(x,dim=1)
        return log_probs

# ok, now lets create word emebedding using skipgram with negative sampling 
# basically what we do here is that we have two embeddings, we feed a word into
# first emebdding get couple of target words that are similar and then feed one of 
# those words into the second embedding and we should get the first word that was initially
# enetered. in doing so, we also feed some noise words, that we use to achieve our loss
# becasue simply trying every single words would impose huge burden. lets see how it is done
class SkipGramWithNegativeSampling(nn.Module):
    def __ini__(self, vocab_size, embedding_size, noise_dist=None):
        super().__init__()

        self.vocab_size = vocab_size
        self.input_embedding = nn.Embedding(vocab_size, embedding_size)
        self.output_emebedding = nn.Embedding(vocab_size,embedding_dim)
        self.noise_distribution = noise_dist
    
    # forward, input and output and noise embeddings
    def forward(self, x):
        input_embeddings_res = self.input_embedding(x)
        output_embeddings_res = self.output_embedding(x)
        return input_embeddings_res, output_embeddings_res

    def forward(self, batch_size, n_sample):
        if self.noise_distribution == None:
            distribution = torch.ones(self.vocab_size)
        else:
            distribution = self.noise_distribution
        
        noise_words = torch.multinomial(distribution, batch_size*n_sample,replacement=True)

        noise_embeddings = self.output_emebedding(noise_words).view(batch_size,n_sample,-1)
        return noise_embeddings

# now ok. now lets create a loss function for ourselves 
class SkipGramNegativeSamplingLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, input_embeddings, output_embeddings, noise_embedidngs):

        # here we will have two losses. the first one shows how much the input
        # and output emebeddings are like each other 
        # and the second loss is responsbile for creating very different embeddings
        # that is negative sampling part. 
        # in order to see how two embeddings are like each other, we simply multiply them
        # 1xembeddings 1xembedding , so we should have a 1x1 result.
        batch_size = input_embeddings.size(0) 
        input_embeddings = input_embeddings.view(batch_size, embedding_dim, 1)
        output_embeddings = output_embeddings.view(batch_size, 1, embedding_dim)
        # log(1) = 0, log(0)=1
        loss1 = torch.bmm(input_embeddings, output_embeddings).sigmoid().log().squeeze()

        # now for our noise 
        noise_embedidngs = noise_embedidngs
        loss2 = torch.bmm(noise_embedidngs.neg(),input_embeddings)
        loss2 = loss2.sum(dim=1)

        return torch.mean(loss1+loss2)



#%%
# TEXT CNN https://mlwhiz.com/blog/2019/03/09/deeplearning_architectures_text_classification/?utm_campaign=shareaholic&utm_medium=reddit&utm_source=news 


#%% transformers
# https://www.reddit.com/r/MachineLearning/comments/dlhcub/d_are_small_transformers_better_than_small_lstms/





#%% CTCloss
# https://github.com/BelBES/crnn-pytorch
# https://towardsdatascience.com/intuitively-understanding-connectionist-temporal-classification-3797e43a86c
# https://machinelearning-blog.com/2018/09/05/753/
# https://stats.stackexchange.com/questions/320868/what-is-connectionist-temporal-classification-ctc
# https://distill.pub/2017/ctc/
# https://github.com/cmudeeplearning11785/Fall2018-tutorials/tree/master/recitation-8




#%% image captioning
# 
#%% GRU
# sentiment analysis 


#%% word embedding 


#%%
# Transformers
# https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html
# 
# For many years, LSTMs has been state-of-the-art when it comes to NLP tasks.
# However, recent advancements in Attention-based models and Transformers have
# produced even better results. With the release of pre-trained transformer
# models such as Google’s BERT and OpenAI’s GPT, the use of LSTM has been 
# declining. 


#%% CTC loss 

#%% wordembedding 


#%% Attention /LSTM with attention 


#%% Image captioning!

#%% BERT, and new sota for nlp stuff!!!
# https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/
#%% rnn-autoencoder (sequence 2 sequence autoencoders),
# sequence to sequence (with bidirection) : https://towardsdatascience.com/understanding-bidirectional-rnn-in-pytorch-5bd25a5dd66 
# seq2seq_vae


#%%
# Named-Entity Recognition(NER). 
