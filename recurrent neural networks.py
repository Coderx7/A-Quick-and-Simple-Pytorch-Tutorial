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
        # This is called truncated backprogation through time (Truncated BPTT)
        # its handy for stable training but be aware that it hurts long dependencies
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
        # by this we merge the batch and sequence dims as one, so instead of (3,4,5)
        # we have (7,5), and the linear layer will process all sequences this way
        # later on we reshape back the output to get sequences back .e. (3,4,output_size)
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

        # note what we do here is effectively truncated backprogapation through time (truncated BPTT)
        # if we dont detach the hidden state the computational graph from previous batches
        # will still be connected and cause the backward pass to try to traverse through
        # the entire sequence (which is too long and not what we want) and so we 
        # will get a long error like this: 
        #  "RuntimeError: Trying to backward through the graph a second time 
        #   (or directly access saved tensors after they have already been freed). 
        #   Saved intermediate values of the graph are freed when you call .backward() or 
        #   autograd.grad(). Specify retain_graph=True if you need to backward through the 
        #   graph a second time or if you need to access saved tensors after calling backward."
        # 
        # we use the hidden state from the previous batch to initialize the current batch,
        # and we want to truncate the gradient flow across batches (but not within the sequence
        # of one batch we'll see more in attention section).
        # therefore we must detach the hidden state from the previous batch to 
        # prevent gradients from flowing back to them a second time (which would cause the graph
        # to be too deep and also would be incorrect).
        # when we do not detach the hidden state like this, the hidden state from the previous
        # batch will be part of the computational graph of the current batch. then when we 
        # compute the loss for the current batch and call loss.backward() the gradients would
        # try to flow back through the entire sequence of batches which as you can guess
        # result in a very deep computational graph (which is neither feasible nor what we want
        # here). furthermore if we try to do this for multiple batches without truncation,
        # we get the error because the graph from the previous backward pass has been freed!
        # unless we use retain_graph=True which the error message also clearly points out
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
# in a traditional case which we saw earlier/in such cases, a seq2seq model is used, 
# that is, a model comprising of two networks, an encoder and a decoder, where the input sequence
# is fed to the encoder, a decoder ultimately recieves a 
# compressed representation, representing the input sequence, from the encoder part and then, 
# tries  to produce a sequence as the answer. 
# the problem with this procedure was/is that for a long
# sequence, we cant transfer the information from earlier time steps, its just simply not possible 
# (yes I know how we said about the lstm and gru gates, retaining earlier time step features, the 
# idea here, is even if lstm gates simplify and ease the transfer of a specific feature from earlier
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
# so what should we do then? we can use all of the states from all previous timesteps instead of using
# only the last one. this way, we can provide much more information and this wealth of information at 
# each timestep can help the network produce better result but how do we do that? 
# surely not all hidden states are equally important when it comes to producing 
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
# this is important and the outputs may not be the best name, as unlike the traditional sense, 
# its not gone through a linear layer with activation function (e.g. tanh in the case of simple RNN),
# as one might think of the output layer of a rnn like what we did in our RNN tutorial/implementation) 
# therefore, instead of using only the last hidden state for the final timestep, we’ll be carrying forward 
# all hidden-states (i.e. for all timesteps) produced by the encoder to the next step.
# 
# after this step, we can start using the decoder to producethe outputs. 
# at each time step of the decoder, we have to calculate the alignment score of each encoder output
# with respect to the decoder input and hidden-state at that time step. 
# the alignment score is the essence of the Attention mechanism, because it quantifies the amount of 
# "Attention" the decoder will place on each of the encoder outputs when producing the next output.
# 
# the alignment scores for Bahdanau Attention are calculated using the hidden state produced by
# the decoder in the previous timestep and the encoder outputs with the following equation:
# score_alignment = W_combined * tanh(W_decoder * H_decoder + W_encoder * H_encoder)
# 
# as you can see, its basically the decoders hidden state plus the encoders hidden state which are
# being used in a tanh transformation function. the weights are basically going to specify how much 
# importantce each hidden state has. 
#
# the decoder's hidden state and encoder's outputs will be passed through their individual 
# Linear layer(that is we use nn.Linear without a bias since it simply does a W*input!
# and makes life easier for us, without it, we should define a new parameter W and multiply it
# by the decoder hidden state, its the same thing! but uglier! so thats why we simply use 
# a linear layer as a learnable parameter for (W_decoder*H_decoder)) and have their own 
# individual trainable weights.
# lastly the resultant vector from the previous few steps will undergo matrix multiplication with 
# a trainable vector, obtaining a final alignment score vector which holds a score for each encoder
# output.
#
# sidenote: 
# because there is no previous hidden state or output for the first decoder step, we use the last encoder 
# hidden state and a Start Of String (<SOS>) token to replace these two respectively.
# 
# after generating the alignment scores vector in the previous step, we can then apply a softmax on this
# vector to obtain the attention weights. The softmax is used so the vector values sum to 1 so each 
# individual value will lie between 0 and 1 and we can consider them as actual(normalized) weights for each input
# for that timestep.
#
# after computing the attention weights in the previous step, we can now generate the context vector by
# doing an element-wise multiplication of the attention weights with the encoder outputs.
# because we used softmax in the previous step, if the score of a specific input element is closer
# to 1 its effect and influence on the decoder output will be amplified and if the score is close to 0,
# its influence will be decreased accordingly(i.e. it'll be nullified).
# 
# the context vector we produced will then be concatenated with the previous decoder output. 
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
#%% fixed version
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
        x_emb = self.embedding(x)
        return self.lstm(x_emb, h)
# lets test 
# enc = Encoder(input_sequence_size=30, embedding_dim=75,vocab_size=50, hidden_size=100)
# xt = torch.randint(0,50, size=(5,30))
# outputs,hidden_state = enc(xt,None)
# print(f'{outputs.shape=}\n{hidden_state[0].shape=}\n{hidden_state[1].shape=}')
# good now lets implement our decoder with the attention mechanism

class BahdanauAttentionDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size,
                 num_layers, bidirectional=False, dropout=0.5) -> None:
        super().__init__()
        
        # since this is a seq2seq model, the
        # output length can be different than the input length,
        # but the size of output (we mean a single timestep) is vocab_size
        # which means, a single word in output (single timestep) can be any
        # word in our vocabulary thus the output size is the same as vocab_size
        # this is used in our linear layer at the end which we use for classification
        # 
        # now note that the following attribute, is the maximum output length,
        # i.e. the maximum number of words that can appear in the output, becasue this is 
        # a seq2seq model, a translation model, the source language and the destination language
        # can have different lengths. therefore, we can specify a maximum length for output
        # the same way, a maximum length is used for the input. but since in practice these two
        # maximum length are chosen to be the same, which means, input sequence and output sequence
        # have the same length, any language that has shorter enntry, will fill the remaining space
        # with EOS symbols signifying the actual data has ended. (its like padding) so we dont use
        # a separate variable/attribute for output sequence length anymore.
        # 
        # we simply use the same length from the input sequence.
        # this is used as a loop counter to keep track of output words before we
        # return the result.
        # self.output_sequence_length = output_sequence_length
                
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

        # we can use hiddensize, but I want to use embedding-dim!
        # to decouple it from hiddensize.
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        
        # our decoder's input will be the concatenated input_t and context_vector
        # both of which will have the shape (batchsize, embd_dim) or (batchsize, hiddensize)
        # we are using a LSTM instead of a LSTMCell, becasue it makes our life easier. 
        # if we use LSTMCell, we have to use a few extra reshapes which is unncessary really. 
        # using an lstm layer like this, makes it a oneliner! easy peasy!
        
        # our decoder input is the contanted input_t + context_vector
        # input_t is embd_dim becasue we use the output of embedding layer
        # context_vector is as large as the hiddensize for our encoder
        self.decoder = nn.LSTM(self.hidden_size + self.embedding_dim,
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
        # we need 3 weights. 
        #   one for decoders state, 
        #   one for encoders state and
        #   one for the weight scale(w_v)
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
        # the 1, in the shape is just to remind us we do the right thing so the correct
        # broadcasting happens see the later notes
        self.W_v =  nn.Parameter(torch.ones(1,hidden_size))
        # or we could simply use a linear layer! and we would use it like attention_scores = self.W_v(weights_added)
        # self.W_v = nn.Linear(hidden_size, 1)
        
        # final classification layer to give us the final results
        # when we define the classifier, we specify the output dim for a single
        # output timestep(word,token),the number of outputs, is determined by the
        # output sequence length (number of timesteps) which will be automatically
        # taken care of (by fusing the timestep dim with batch dim, basically treat
        # each timestep as another batch, which later we reshape back and get the
        # processed final output sequence)
        self.classifier = nn.Linear(hidden_size, self.vocab_size)
        
        # self.start token - we use 0 as the number representing the special start token
        # later on when we build our vocabulary, we inlcude it there!
        self.sos_symbol = 0
    
    def forward(self, encoder_states, hidden_state, teacher_forcing_input=None):
        
        # note teacher_forcing_input can be used to train faster and better, 
        # where we use the actual label/target for a given timestep, instead 
        # of the previously generated output by the decoder, and this way we
        # help the model the same way a teacher helps a student by sometimes 
        # giving a part of the answer. I'll add this when the main attention
        # functionality is implemented and model can be trained properly,then
        # we can enhance it by 2 lines of code and thats it we have teacher
        # forcing as well(note that teacher forcing is only done in training!)
        
        # note we dont need input embedding in decoder, its managed by the encoder
        # instead we need a single timestep input, the start token 
        # grab its embedding to use it for initial input
        # 
        # self.start token
        self.start_token = torch.tensor([self.sos_symbol for _ in range(encoder_states.size(0))], dtype=torch.long).view(-1,1)
        # or we could have done it using repeat!
        # self.start_token = torch.tensor([self.sos_symbol],dtype=torch.long).repeat(encoder_states.size(0)).view(-1,1)
        input_t = self.embedding(self.start_token.to(encoder_states.device))
    
        # remember we also need to grab outputs for each timestep
        # so for a given input_sequence, we return an output_sequence 
        # (here of the same size, but it could be different in general)
        # we do the classificantion in the loop as well.
        # note that had we used a normal decoder without attention, we would have done 
        # the classification in one go, like this
        # outputs, hidden_state = self.decoder(input_sequence, hidden_state)
        # outputs = self.classifier(outputs.reshape(-1, self.hidden_size*self.direction))
        # however, since now we are using attention, we need to apply attention iteratively
        # one step after another timestep and produce outputs one by one (because each output
        # is used as the input for the next output! therefore we have a loop here)
        outputs = []
        # we grab attention weights and stackthem 
        # so later on we can visualize attention
        # for a given input (it would be a (input_sequence x output_sequence) matrix when staked at
        # the end therefore you'd see which timesteps/words interact positively e.g.)
        attentions = []
        # the other thing to do is to specify a max-length for output characters/words/timesteps
        # becasue in translation, the source and destination can have different lengths, so we can
        # use a max_length to create out input/output sequences when we create our dataset. so since
        # the sequence length for both input and output is the same, we can use encoder_states.size(1)
        # in our loop as it has hiddenstate for each timestep. note that having the same sequence legth
        # for both input and output doesnt pose an issue for us, provided that, first its long enough to
        # accomedate all sample lengths for both langauges, and then we fill the unsued space for either
        # of them that are shorter with <EOS> (end of sentence) and that should do it
        max_length = encoder_states.size(1) # input_sequence.size(1)
        for t in range(max_length):
            # print(f'------------------------')
            # for the first time we use a start token(sos), for the next timesteps we use the
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
            output_t, hidden_state, attention_weights = self.forward_attention(input_t, encoder_states, hidden_state)
            
            # truncated BPTT, I initially enabled it to help with more stable training, 
            # and everything seemingly went smoothly. later on however I notice this 
            # hurts long term learning/dependencies! so I'm disabling it now!
            # update:
            # after disabling it, we got much faster convergence rate(at least 3x)
            # and lower loss. 
            # note that we can disable it here, because we are doing the whole operation
            # in a step by step fashion. in our previous examples, we didnt have a loop to 
            # get the outputs, the whole sequence was fed to the model forward at once and
            # we'd get the result. also we didnt need the hiddenstates gradients
            # from the previous batch so we truncated it after each batch! (the standard practice
            # for TBPTT is to detach the hidden state after each batch). if we didnt, we would 
            # be traversing the previous compulation graph (which was freed) a second time and face an error!
            # 
            # this is not the case hre, here we are generating the output sequence step by step,
            # which means we are still within the same batch(so no cross batch issues here), 
            # so we are safe not to detach the hidden state. if we want we can detach it at 
            # each step (or after some step-interval) (to avoid backpropagating through the entire generation loop)
            # but as we saw it will hurt the performance! so not detaching it is not only ok 
            # here but the absoluetly correct thing to do! as we are in the same batch!
            # 
            # (also the hidden state is reset for each batch already(each time forward is called 
            # for a new batch we start off with a new hiddenstate). therefore, we dont have the
            # issue of carrying the hidden state across batches like our previous examples anyway!)
            # 
            # so to cut a long story short, we were detaching the hidden state within the same
            # batch (across output timesteps) to avoid a very deep graph (and to simulate a 
            # truncated BPTT within the batch) but since we werent carrying the hidden state across 
            # batches (it is reset for each batch) we dont have the same cross-batch issue we
            # face in our previous examples (and also detaching it was a bad idea in first place
            # as our sequence length is not large to pose an issue!(>1oo would be when we think about using truncated bptt)
            # and even if it was detaching at every single step isnt right and will hurt the model performance) so it was not useful at all!
            # also I found out that the standard practice for seq2seq is NOT to detach within a batch anyway!
            # 
            # hidden_state = tuple(h.detach()for h in hidden_state)
            
            # output_t is (2,1,7) (7 is hidden-size here), so we need the classifier to give
            # the output of size (bs, output_size). (our classifier dim is (hiddensize, outputsize))
            # the reshape makes it (2,7) which is compatible with our classifier
            output_t = self.classifier(output_t.view(output_t.size(0),-1))
            # update the input_t for the next round
            # get the highest probablity one
            output_t_probs = output_t.softmax(dim=-1)
            # print(f'{output_t_top_probs=}')
            _,output_t_idx_max = output_t_probs.max(dim=-1)
            # print(f'{output_t_max=}')
            # get its embedding and use it as the new input_t 
            # and remember to make its shape right (i.e. (b,ts,features))
            input_t = self.embedding(output_t_idx_max)[:,None,:]
            
            # simply append the outputs and at the end
            # stackthem in timestep dim and return it
            outputs.append(output_t)
            # stack attentions as well
            attentions.append(attention_weights.detach())

        # now stack the outputs along timestep dim to get (bs, output-seq-len/timesteps)
        # they can now be used to compare with labels 
        outputs = torch.stack(outputs, dim=1)
        # do the same for attention weights
        attentions = torch.stack(attentions, dim=1)

        return outputs, hidden_state, attentions

    def forward_attention(self, input_sequence_t, encoder_states, hidden_state):
        # calculate attention raw score (attn = w_v * tanh(W_D+W_E))
        # do a softmax, make the raws_cores into conditional probablities
        # multiply the new weight/scores by the attention-states, get the context vector
        # feed the context vector to the decoder  concatenated with input
        # at timestep t,(initial input is <sos> token, the next inputs are the previous outputs
        # of the decoder (previous timestep output becomes next timestep input))
        # calculate new outputs, hiddenstates and return the outputs and hidden_state for this timestep
        # grab the long-term memory(h)
        (h,c) = hidden_state
        # print(f'{h.shape=}')              # (1, batch, hidden-size)
        # print(f'{encoder_states.shape=}') # (batch, timesteps, hidden-size)
        # our hidden state has the shape (num-layers, batch, hidden-size), 
        # since our num_layers=1 it becomes (1, batchsize, hidden-size) here.
        # while our encoder's outputs has (batch, timesteps, hidden-size)
        # we need to reshape our hiddenstate to have (batchsize, 1, hidden-size)
        # the easiest way is to simply transpose/permute it:
        h = h.permute(1,0,2)                # (batchsize, 1, hidden-size)
        # print(f'{h.shape=}')
        # after this we can simply add the W_d and W_e together, the W_d will be broadcasted
        # (will be repeated along dim=1, and these two will be added properly)
        W_d = self.W_decoder(h)
        W_e = self.W_encoder(encoder_states)
        weights_added = torch.tanh(W_d + W_e)
        # print(f'{weights_added.shape=}')  # (batch, ts, hidden-size)
        # print(f'{self.W_v.shape=}')       # (1, hidden-size)
        # likewise our W_v shape is (1, hidden-size), in order to multiply it by
        # our weights_added, we need to make it compatible. 
        #
        # Note that we have two ways to get to the same shape, but only one of them is correct.
        # one (wrong) way would be making w_v (1,1,hidden-size), so that it will be broadcasted 
        # accordingly to match (batch, ts, hidden-size) which we want our final weights to look like.
        # (1,hidden-size) will be automatically broadcasted so no need to to do sth like W_v.data.unsqueez_(0)
        # attention_score = self.W_v * weights_added # (batch, ts, hidden-size)
        # This is wrong, becasue it gives us the wrong attention_score (alignment score (alpha score!))
        # if we do that the result will also be (batch,ts,hidden-size) but we want it to be (batch, ts, 1)
        # signaling, theres a single weight for each timestep
        # (sidenote: the shape of the context-vector must match the hidden_state)
        # we could also achieve this shape by doing a matmul which is the correct way by the way
        # this is the way to do it as it gives us the desired shape of (batch, ts, 1)
        # (sidenote: the shape of the context-vector must match the hidden_state which happens
        # only like this, otherwise the context vector at the end would endup (batchsize, ts, hiddensize)
        # which is wrong, as it should be the weighted sum of the encoder-states (if we sum over ts axis and
        # get (batch, 1, hidden_size), it would still be wrong as we have done a different operation))
        attention_score = torch.matmul(weights_added ,self.W_v.t())
        # if we had implemented W_v as a linear layer instead of a nn.Parameter,
        # we could have also done this instead and get the exact same result:
        # attention_score = self.W_v(weights_added)
        # print(f'{attention_score.shape=}') # (batch,ts,1) which means each timestep has a 
        # single weight
        # 
        # torch.set_printoptions(profile='default')
        # softmax to ensure both nonnegativity and normalization.
        #sidenote: note that this is the weight matrix thats usually visualized
        # to show where the attention is focused in different tutorials you see
        # so if you want to visualize it, you can return the attention weights as well
        # and then concat them later along dim=1 and then go for visualizing it
        # note we calculate softmax along dim=1! otherwise we wouldnt be getting
        # a probablity along that dim obviously!
        attention_weights = attention_score.softmax(dim=1)
        # print(f'{attention_weights.shape=}') #(2,5,1)
        
        # note there are several ways to multiply attention_weights with encoder_states 
        # and get the same result, but not all are correct. Here are 3 ways to get the same result: 
        # 
        # method 1: matrix multiplication
        # context_vector = torch.matmul(attention_weights.permute(0,2,1), encoder_states)
        # 
        # method 2: elementwise multiplication followed by summation 
        # context_vector = (attention_weights * encoder_states).sum(dim=1)[:,None,:]
        # note we are summing over timesteps(dim=1) and not the last dim which is hiddenstate dims,
        # so ultimately we have a single hidden-state vector that incorporates 
        # all previous hiddenstates, this becomes our context_vector
        # print(f'{context_vector.shape=}') # should be (batch, 1, hidden_size)
        # we added a new empty dim at dim=1(by doing [:,None,:]) to make it match
        # the hidden-state shape so we can concat them to feed the decoder as the input.
        # 
        # method 3: (the correct way)
        # use batch-matrix multiplication which 
        # first permute the weights dim so they become compatible with encoder_states
        # and then carry on the multiplication which gives us the context_vector which 
        # is the weighted sum of the encoder states (each element in the context_vector
        # is the sum of the encoder states weighted by the corresponding attention weights
        # which is what we want!)
        # print(f'{encoder_states.shape=}')#(2,5,7)
        context_vector = torch.bmm(attention_weights.permute(0,2,1), encoder_states)
        # print(f'{context_vector.shape=}')
        
        # sidenote: now whats the deal here? whats the difference between these 3 methods
        # to give you the idea, lets first have a reminder 
        # - `torch.matmul` performs matrix multiplication. 
        #    When dealing with 3D tensors, it performs "batch" matrix multiplication.
        # - For 2D tensors, it is equivalent to the dot product.
        # - For higher dimensions, it follows the rules of broadcasting and performs
        #   the appropriate matrix multiplications.
        # next is torch.bmm: 
        # - `torch.bmm` is specifically designed for batch matrix multiplication of 3D tensors.
        # - It takes two 3D tensors of shapes `(b, n, m)` and `(b, m, p)` and returns a 3D tensor
        #   of shape `(b, n, p)`.
        # 
        # so to cut a long story short:
        # method 1 and 3 give us the correct result, because torch.matmul in method 1 performs a 
        # batch matrix multiplication just like method 3. 
        # obviously torch.matmul and torch.bmm are not the same but in this specific case 
        # they result in the same outcome because of the shapes and dimensions involved,
        # they compute the weighted sum of the encoder states.
        # element-wise multiplication followed by a sum is a different operation, although 
        # it can produce the same shape in specific cases like ours the outcome is wrong!
        # 
        # so to make the intend clear, we use torch.bmm otherwise if we use matmul, we may make
        # a mistake and go haywire as I previously did and wasted a lot of time debugging!
        # 
        # sidenote: 
        # (if its not clear yet, bmm simply means, we are dealing with several samples instead of 1
        # set aside the batch dimension for a moment and you'll noticed we endup with 5 numbers 
        # that act as weights for each timestep (recall our attention_weights shape was (2,5,1) 
        # and our encoder-states was (2,5,7), if you set aside the batch we get (5)(ignore the 1) 
        # and (5,7) respectively which again lets you know there are 5 numbers for 5 hiddenstates. 
        # now to multiply them we use simple matrix multiplication. 
        # when there are several samples, like in our case, we have a batch, so we do this
        # in a for loop, and do the multiplication for each sample individually and 
        # then stack the results hence how we actually do a batch-matrix-multiply behind the scene!
        # of course the actual behind the scene implementation of bmm uses specialized routine to
        # speed things up based on the hardware its executed but the idea stays the same regardless))
        # sidenote: 
        # the dimensions in explanation (like (2,5,7) etc) comes from my test example below which I 
        # used for debugging
        # 
        # 
        # at this point we have a context vector that shows the attention
        # of each input sequence, we can feed this directly to decoder
        # or concatenate it with the input at timestep_t and then feed this new
        # input to decoder (the second method is what we do)
        # 
        # I have seen some people where they multiply context by the encoder-states 
        # and then use that to concatenate with the input. 
        # this is called multiplicative attention. by the way its different than 
        # loung attention! which we will see in a moment
        # states = context*encoder_states
        # print(f'{states.shape=}')
        
        # print(f'{input_sequence_t.shape=}')
        
        # Note we dont feed the whole input_sequence to our decoder
        # since this process is happening in a loop, we feed a single input timestep
        # each time, the initial input of the decoder is usually a start token, 
        # for the next inputs, we use the output of our decoder from previous step.
        # 
        decoder_input = torch.cat([input_sequence_t, context_vector],dim=-1)
        # print(f'{decoder_input.shape=}')
        # print(f'{hidden_state[0].shape=}')
        # 
        # make the hiddenstate the (layer, batch, hiddenstate) instead of (batch,layer,hiddensatet)
        # h = h.permute(1,0,2)
        # print(f'{h.shape=}')
        # lstm decoder expects the hiddensize to be a tuple so we can either use (h,h)
        # or use the previous hiddenstate altogether, (h is our longterm memory and c 
        # is the short term memory! and we have been using the long term one)
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
        return outputs, hidden_state, attention_weights

seq_length = 5
embedding_dim = 6
hidden_size = 7
batch_size=2
vocab_size = 50
enc = Encoder(vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_size=hidden_size)
xt = torch.randint(0, vocab_size, size=(batch_size, seq_length))
h = None
outputs,hidden_state = enc(xt,h)

decoder = BahdanauAttentionDecoder(vocab_size=vocab_size,
                                   embedding_dim=embedding_dim,
                                   hidden_size=hidden_size,
                                   num_layers=1,
                                   )

outputs,hidden_state, attentions = decoder(outputs,hidden_state)
print(f'{outputs.shape=}')
print(f'{hidden_state[0].shape=}')
print(f'{attentions.shape=}')
#%%
torch.manual_seed(12)
seq_length = 5
embedding_dim = 6
hidden_size = 7
batch_size=2
vocab_size = 50
enc = Encoder(vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_size=hidden_size)
xt = torch.randint(0, vocab_size, size=(batch_size, seq_length))
h = None
outputs,hidden_state = enc(xt,h)

decoder = BahdanauAttentionDecoder(vocab_size=vocab_size,
                                    embedding_dim=embedding_dim,
                                    hidden_size=hidden_size,
                                    num_layers=1,
                                    )

outputs,hidden_state,attention = decoder(outputs,hidden_state)
print(f'{outputs.shape=}')
print(f'{hidden_state[0].shape=}')
print(f'{attention.shape=}\n')
print(f'{outputs[0][0]=}')
print(f'{attention[0,:,:,0]=}\n')
#%%
# Now lets create the whole model 
# we'll keep it simple (we can use different values for encoder/decoder but here we use only
# one set for both)
class LSTMBahdanau(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size, embedding_dim, hidden_size, enc_num_layers=1,
                 enc_dropout=0.3, dec_num_layers=1, dec_dropout=0.5):
        super().__init__()
        
        # input_vocab can be different than output vocab (french vs english)
        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.enc_dropout = enc_dropout
        self.dec_dropout = dec_dropout
        self.enc_num_layers = enc_num_layers
        self.dec_num_layers = dec_num_layers
        
        self.encoder = Encoder(vocab_size=input_vocab_size,
                               embedding_dim=embedding_dim,
                               hidden_size=hidden_size,
                               dropout=enc_dropout,
                               num_layers=enc_num_layers)
        
        self.decoder = BahdanauAttentionDecoder(vocab_size=output_vocab_size,
                                                embedding_dim=embedding_dim,
                                                hidden_size=hidden_size,                 
                                                num_layers=dec_num_layers,
                                                dropout=dec_dropout,
                                                )
        self.drp = nn.Dropout2d(0.1)

    def forward(self, x, h, target=None):
        enc_outputs, hidden_state = self.encoder(x, h)
        output, hidden_state, attentions = self.decoder(enc_outputs, hidden_state, target)
        # output = self.drp(output)
        return output, hidden_state, attentions

torch.manual_seed(12)
seq_length = 5
embedding_dim = 6
hidden_size = 7
batch_size=2
vocab_size = 50
xt = torch.randint(0, vocab_size, size=(batch_size, seq_length))
h = None
model = LSTMBahdanau(vocab_size,
                     vocab_size,
                     embedding_dim,
                     hidden_size=hidden_size)
o,h,a = model(xt,h)
print(f'{o.shape=}')
print(f'{h[0].shape=}')
print(f'{a.shape=}')
print(f'{a[0,:,:,0]=}')

#%%
import re
import unicodedata
from unidecode import unidecode
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
# note
# since our dataset is utf-8/unicode, there are non ascii characters
# which we are better off removing this simplifies the 
# text and keeps our vocab small.
# there are several ways you might think of to do this, 
# like using str.encode('ascii', 'ignore') to ignore 
# nonascii characters this is not good as it ruins 
# the words (for example "Héllo, Wörld!" would become Hllo, Wrld!)
# str.encode('ascii', 'replace').decode('ascii') is not better as
# it would replace all non-ascii characters! e.g. "Héllo, Wörld!"
# would become "H?llo, W?rld!". 
# but using unidecode.unidecode(), we can achieve what we are after
unicode_string = "Héllo, Wörld!"
# convert unicode to ascii, by ignoring non-ascii characters
print(f"ignoring non-ascii characters:  {unicode_string.encode('ascii', 'ignore').decode('ascii')}")
# convert unicode to ascii, by replacing non-ascii characters with a placeholder
print(f"replacing non-ascii characters: {unicode_string.encode('ascii', 'replace').decode('ascii')}")
# using ascii_string = unidecode(unicode_string)
print(f'converting using unidecode(): {unidecode(unicode_string)}')
# theres also this function from official pytorch docs that uses 
convert_to_ascii = lambda s: ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
# in which uses unicodedata.normalize('NFD', s) to decompose characters 
# into their base form and combining marks. For example, “é” becomes “e” + “´”.
# and later on, it removes those characters in another try. 
print(f'converting using unicodedata.normalize(): {convert_to_ascii(unicode_string)}')

dataset_filename ='/media/hossein/SSD1/A-Quick-and-Simple-Pytorch-Tutorial/data/data/eng-fra.txt'

with open(dataset_filename, encoding='utf-8') as f:
    lines = f.read().lower().strip().splitlines()

def normalize(unicode_str):
    # convert unicode to ascii characters
    ascii_str = unidecode(unicode_str)
    # lets also put a space before any punctuations (.!?)
    # this is to make words such as 'go.' and 'go' be treated the same
    # and prevent duplication of words in vocab becasue of 
    # the adjacent punctuation marks such as .!? (go and go. and go! all would be different words!)
    # str_normalized = re.sub(r"([.!?])", r" \1", ascii_str)
    # but a better choice is to check if theres already a space,
    # and only if theres none before punctuation marks, only then add a space before them!
    # (?<!\s) is a negative lookbehind assertion that makes sure
    # there is no space before any of our 3 punctuation marks(i.e. ([.!?]))
    str_normalized = re.sub(r"(?<!\s)([.!?])", r" \1",  ascii_str)
    # english, french sentences are separated by a \t (tab)
    # so lets split them!
    return str_normalized.strip().split('\t')

pairs_list = [normalize(line) for line in lines]
print(f'{len(pairs_list)=}')
print(f'{pairs_list[:5]}')

# now before we continue with the rest of code
# recall that, we need to use a max length for sequences
# so we can filter sequences based on their length and content
# this way we reduce even more words/vocabs.

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)

def filter_pair(pair, max_words_in_sentence):
    return len(pair[1].split(' ')) < max_words_in_sentence \
        and len(pair[0].split(' ')) < max_words_in_sentence and pair[0].startswith(eng_prefixes)

def filter_pairs(pairs,max_words_in_sentence):
    return [pair for pair in pairs if filter_pair(pair, max_words_in_sentence)]

# as you can see the number has decreased drastically!
# this is ok as we are just experimenting, we wouldnt want
# such drastic reduction in our dataset (the majority of reduction
# is becasue of the last check which only includes a specific type of sample
# with of which, we would have vocabs of 11K and 23K for english and french
# we later on test without this check
max_words_in_sentence = 10
pairs_list = filter_pairs(pairs_list, max_words_in_sentence)
print(f'{len(pairs_list)=}')
print(f'{pairs_list[:5]}')

# now lets create vocabs for each language separately
# dont forget to include the special symbols first!
en_word2int = {'<sos>':0, '<eos>':1}
fr_word2int = {'<sos>':0, '<eos>':1}

# to get unique words, we concat all samples for each language
# and then grab the unique words and then use that to fill the
# word2int dictionaries
en_samples = ' '.join(p[0] for p in pairs_list)
fr_samples = ' '.join(p[1] for p in pairs_list)

print(f'{len(en_samples)=}')
print(f'{en_samples[:10]}')

print(f'{len(fr_samples)=}')
print(f'{fr_samples[:10]}')

unique_words_en = list(set(en_samples.split()))
unique_words_fr = list(set(fr_samples.split()))

print(f'{len(unique_words_en)=:,}')
print(f'{unique_words_en[:10]=}')

print(f'{len(unique_words_fr)=:,}')
print(f'{unique_words_fr[:10]=}')

en_word2int.update({w:i for (i,w) in enumerate(unique_words_en, start=2)})
fr_word2int.update({w:i for (i,w) in enumerate(unique_words_fr, start=2)})
# now lets create the int2word dictionaries as well
en_int2word = {i:w for (w,i) in en_word2int.items()}
fr_int2word = {i:w for (w,i) in fr_word2int.items()}

print(f'{len(en_word2int)=:,}')
print(f'{en_word2int=}')
print(f'{en_int2word=}')

print(f'{len(fr_word2int)=:,}')
print(f'{fr_word2int=}')
print(f'{fr_int2word=}')

# now lets convert a sentence to int and vice versa!
def convert_to_int(sentence, lang, max_words, to_tensor=False):
    dic = en_word2int if lang == 'en' else fr_word2int
    int_sequence = [dic[s] for s in sentence.split()]
    # when done, add the <eos> symbol at the end signifying its end
    if len(int_sequence) < max_words:
        int_sequence.append(dic['<eos>'])
    else:
        int_sequence[-1] = dic['<eos>']
    return torch.tensor(int_sequence, dtype=torch.long).view(1, -1) if to_tensor else int_sequence

def convert_to_word(id_sequence, lang):
    dic = en_int2word if lang == 'en' else fr_int2word
    is_tensor = isinstance(id_sequence, torch.Tensor)
    return [dic[s.item() if is_tensor else s] for s in id_sequence]

sentence = pairs_list[0][0]
print(sentence)
print(convert_to_int(sentence,'en', max_words=max_words_in_sentence, to_tensor=True))
print(convert_to_word(convert_to_int(sentence,'en',max_words=max_words_in_sentence),'en'))

device = 'cuda' if torch.cuda.is_available()  else 'cpu'

# now lets create our dataloader!
def get_dataloader(batch_size, input_lang, output_lang, seq_length, 
                   pairs_list, en_word2int, fr_word2int, reverse=False, device='cuda'):

    if reverse:
        pairs_list = [(l2,l1) for l1,l2 in pairs_list]
        input_lang, output_lang = output_lang, input_lang
    
    # input word2int
    input_word2int = en_word2int if input_lang == 'en' else fr_word2int
    output_word2int = fr_word2int if output_lang == 'fr' else en_word2int
    
    print(f'{input_lang=}')
    print(f'{output_lang=}')
    print(f'{pairs_list[0]=}')
    # print(f"{input_word2int['i']=}")
    
    # make it so we all full batches
    sample_num = len(pairs_list)
    sample_num = (sample_num//batch_size) * batch_size
    print(f'{sample_num=:,}')

    input_ids = np.zeros((sample_num, seq_length), dtype=np.int32)
    target_ids = np.zeros((sample_num, seq_length), dtype=np.int32)

    for idx, (inp, tgt) in enumerate(pairs_list):
        # to ensure we get full batches all the time
        if idx>=sample_num:
            continue
        # inp_ids = indexesFromSentence(input_lang, inp)
        # tgt_ids = indexesFromSentence(output_lang, tgt)
        # print(f'{inp=}')
        inp_ids = convert_to_int(inp, input_lang, max_words=seq_length)
        tgt_ids = convert_to_int(tgt, output_lang, max_words=seq_length)
        
        # inp_ids.append(input_word2int['<eos>'])
        # tgt_ids.append(output_word2int['<eos>'])
        
        input_ids[idx, :len(inp_ids)] = inp_ids
        target_ids[idx, :len(tgt_ids)] = tgt_ids

    train_data = TensorDataset(torch.LongTensor(input_ids).to(device),
                               torch.LongTensor(target_ids).to(device))

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    return input_lang, output_lang, input_word2int, output_word2int, train_dataloader

input_lang, output_lang, input_w2i, output_w2i, dl = get_dataloader(32, 'en','fr', 10, pairs_list, en_word2int, fr_word2int,reverse=True)
print(f'{len(dl)=}')
print(f'{next(iter(dl))[0][:5]=}')
#%%
import numpy as np
from tqdm import tqdm
import re
import unicodedata
from unidecode import unidecode
from collections import namedtuple

import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

# from torch.utils.data import TensorDataset, DataLoader, RandomSampler

# so far so good, but thats very messy, lets tidy things up a bit so 
# we can actually make heads or tails of it!
# for this, in order not to send several variables each time to every functions!
# and write logic each time, lets encapsulate each language into a structure
# and access all related information all in one place through said structure.
# we can use a class for vocab, but I think a named tuple should suffice, as we plan
# on encapsulating everything in a single class to avoid repetition
# between language/vocab class and dataset class. (reading files/using pairs_list
# so instead we do it once here!)
# 
Vocab = namedtuple('Vocab', ['name', 'size', 'word2int', 'int2word'])

class Dataset():
    def __init__(self, dataset_filename, max_sequence_words=10, swap_languages=False, filter_long_sequences=False):
        # read the dataset, use the filename 
        # to identify the languages, and then
        # create a named tuple for each langauge
        # filename looks like: eng-fra.txt
        self.dataset_filename = dataset_filename
        self.max_sequence_words = max_sequence_words
        self.should_swap_languages = swap_languages
        self.should_filter_long_sequence = filter_long_sequences
        
        filename = os.path.basename(dataset_filename).split('.')[0]
        self.in_lang, self.out_lang = filename.split('-')
   
        with open(dataset_filename, encoding='utf-8') as f:
            self.lines = f.read().lower().strip().splitlines()
        
        self.pairs_list = [self._preprocess(line) for line in self.lines]
        
        if self.should_swap_languages:
            self.pairs_list = [(l2,l1) for l1,l2 in self.pairs_list]
            self.in_lang, self.out_lang = self.out_lang, self.in_lang

        if self.should_filter_long_sequence:
            self.pairs_list = self._filter_long_sequences()
        
        self.in_vocab, self.out_vocab = self._create_vocabs()

    def _preprocess(self, unicode_str):
        # convert unicode to ascii characters
        ascii_str = unidecode(unicode_str)
        # lets also put a space before any punctuations (.!?)
        # this is to make words such as 'go.' and 'go' be treated the same
        # and prevent duplication of words in vocab becasue of 
        # the adjacent punctuation marks such as .!? (go and go. and go! all would be different words!)
        # str_normalized = re.sub(r"([.!?])", r" \1", ascii_str)
        # but a better choice is to check if theres already a space,
        # and only if theres none before punctuation marks, only then add a space before them!
        # (?<!\s) is a negative lookbehind assertion that makes sure
        # there is no space before any of our 3 punctuation marks(i.e. ([.!?]))
        str_normalized = re.sub(r"(?<!\s)([.!?])", r" \1",  ascii_str)
        # english, french sentences are separated by a \t (tab)
        # so lets split them!
        return str_normalized.strip().split('\t')

    def _filter_long_sequences(self):
        # we are basically selecting a small subset of all the sentences 
        # by this filtering routine!
        eng_prefixes = ("i am ", "i m ",
                        "he is", "he s ",
                        "she is", "she s ",
                        "you are", "you re ",
                        "we are", "we re ",
                        "they are", "they re ")
        # french doesnt seem to be having many widly
        # used abbreviated forms unlike english!
        french_prefixes = ("je suis", "j'suis",
                     #     "il est",  "il est",
                     #     "elle est","elle est",
                           "tu es", "t'es",
                     #     "nous sommes", "nous sommes",
                     #     "vous etes", "vous etes",
                     #     "ils sont", "ils sont",
)
        prefixes = eng_prefixes if self.in_lang == 'eng' else french_prefixes
        
        pair_is_small = lambda pair: (len(pair[1].split(' ')) < self.max_sequence_words and
                                      len(pair[0].split(' ')) < self.max_sequence_words and
                                      pair[0].startswith(prefixes))
        return [pair for pair in self.pairs_list if pair_is_small(pair)] 

    def _create_vocabs(self):
        
        # now lets create vocabs for each language separately
        # dont forget to include the special symbols first!
        in_word2int = {'<sos>':0, '<eos>':1}
        out_word2int = {'<sos>':0, '<eos>':1}

        # to get unique words, we concat all samples for each language
        # and then grab the unique words and then use that to fill the
        # word2int dictionaries
        # a single loop is faster!
        in_samples, out_samples = [], []
        for p1,p2 in self.pairs_list:
            in_samples.append(p1)
            out_samples.append(p2)
        in_samples = ' '.join(in_samples)
        out_samples = ' '.join(out_samples)
        
        # set allows us to only grab the unique words!
        unique_words_in = list(set(in_samples.split()))
        unique_words_out = list(set(out_samples.split()))

        in_word2int.update({w:i for (i,w) in enumerate(unique_words_in, start=2)})
        out_word2int.update({w:i for (i,w) in enumerate(unique_words_out, start=2)})
        
        # now lets create the int2word dictionaries as well
        in_int2word = {i:w for (w,i) in in_word2int.items()}
        out_int2word = {i:w for (w,i) in out_word2int.items()}
        
        vocab_in = Vocab(name=self.in_lang, 
                         size=len(in_word2int), 
                         word2int=in_word2int, 
                         int2word=in_int2word)
        
        vocab_out = Vocab(name=self.out_lang, 
                          size=len(out_word2int), 
                          word2int=out_word2int, 
                          int2word=out_int2word)
        
        return vocab_in, vocab_out
    
    # now lets convert a sentence to int and vice versa!
    def convert_to_int(self, sentence, vocab, to_tensor=False):
        # only grab up to max seq words!
        int_sequence = [vocab.word2int[s] for s in sentence.split()][:self.max_sequence_words]
        # when done, add the <eos> symbol at the end signifying its end
        if len(int_sequence) < self.max_sequence_words:
            int_sequence.append(vocab.word2int['<eos>'])
        else:
            int_sequence[-1] = vocab.word2int['<eos>']
        return torch.tensor(int_sequence, dtype=torch.long).view(1, -1) if to_tensor else int_sequence

    def convert_to_word(self, int_sequence, vocab):
        is_tensor = isinstance(int_sequence, torch.Tensor)
        return [vocab.int2word[s.item() if is_tensor else s] for s in int_sequence]

    # now lets create our dataloader!
    def get_dataloader(self, batch_size, pin_memory=True, num_workers=8):
        # make it so we all full batches
        sample_num = len(self.pairs_list)
        sample_num = (sample_num//batch_size) * batch_size
        
        input_seq_ids = np.zeros((sample_num, self.max_sequence_words), dtype=np.int32)
        target_seq_ids = np.zeros((sample_num, self.max_sequence_words), dtype=np.int32)

        for i, (input_seq, target_seq) in enumerate(self.pairs_list):
            # to ensure we get full batches all the time
            if i>=sample_num:
                continue

            input_seq = self.convert_to_int(input_seq, self.in_vocab)
            target_seq = self.convert_to_int(target_seq, self.out_vocab)

            input_seq_ids[i, :len(input_seq)] = input_seq
            target_seq_ids[i, :len(target_seq)] = target_seq

        train_data = TensorDataset(torch.LongTensor(input_seq_ids), 
                                   torch.LongTensor(target_seq_ids))

        train_sampler = RandomSampler(train_data)

        train_dataloader = DataLoader(train_data, 
                                      sampler=train_sampler, 
                                      batch_size=batch_size,
                                      pin_memory=pin_memory, 
                                      num_workers=num_workers)
        
        return train_dataloader, self.in_vocab, self.out_vocab 

dt = Dataset('/media/hossein/SSD1/A-Quick-and-Simple-Pytorch-Tutorial/data/data/eng-fra.txt',
             max_sequence_words=10,swap_languages=0,filter_long_sequences=0)

print(dt.in_vocab.size)
print(dt.out_vocab.size)

dl,in_vocab,out_vocab = dt.get_dataloader(32)
print(f'{len(dl)=}')
print(f'{next(iter(dl))[0][:5]=}')

#%%

def train_epoch(dataloader, model, optimizer, criterion):
    model.train()
    total_loss = 0
    hidden_states = None
    for input_tensor, target_tensor in tqdm(dataloader):

        input_tensor = input_tensor.to(device)
        target_tensor = target_tensor.to(device)
        
        outputs, hidden_states, attentions = model(input_tensor, hidden_states, target_tensor)
        hidden_states = tuple(h.data for h in hidden_states)
        
        loss = criterion(outputs.view(-1, outputs.size(-1)), target_tensor.view(-1))
        # or if we had access to vocab_size,
        # loss = criterion(outputs.view(-1, out_vocab.size), target_tensor.view(-1))
        
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)

def train(train_dataloader, model, epochs, checkpoint_path,learning_rate=0.001, interval=100):
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    losses = []
    total_loss = 0
    for epoch in range(epochs):
        loss = train_epoch(train_dataloader, model, optimizer, criterion)
        total_loss += loss

        if epoch%interval == 0:
            loss_avg = total_loss / interval
            losses.append(loss_avg)
            total_loss = 0
            print(f'{epoch}/{epochs} | Loss: {loss_avg:.4f}')
            torch.save({"state_dict":model.state_dict(),
                        "epochs":epoch,
                        "loss":loss_avg,
                        "embedding_dim":model.embedding_dim,
                        "hidden_size":model.hidden_size}, checkpoint_path)
            
    plt.plot(losses)

embedding_dim = 256
hidden_size = 256
batch_size = 32

dataset_filename = '/media/hossein/SSD1/A-Quick-and-Simple-Pytorch-Tutorial/data/data/eng-fra.txt'
dt = Dataset(dataset_filename, 
             max_sequence_words=10,
             swap_languages=0,
             filter_long_sequences=0)

train_dataloader,in_vocab, out_vocab = dt.get_dataloader(batch_size)

model = LSTMBahdanau(input_vocab_size=in_vocab.size,
                    output_vocab_size=out_vocab.size,
                    embedding_dim=embedding_dim,
                    hidden_size=hidden_size)

model.to(device)
# note we didnt use any schedulers, we are using the bare minimum here
# in an actual usecase, we will create a much better network, with better 
# regularization, and optimization regime, but for now its suffices 
# we just want to see how it performs and whether our attention mechanism actually works!
# (it does :))
train(train_dataloader, model, epochs=80, interval=5, checkpoint_path='./weights/bahdanau_attention.pth')
# 65/80 | Loss: 0.5280
# 100%|██████████| 4245/4245 [00:50<00:00, 84.21it/s]
# 100%|██████████| 4245/4245 [00:50<00:00, 84.35it/s]
# 100%|██████████| 4245/4245 [00:50<00:00, 84.24it/s]
# 100%|██████████| 4245/4245 [00:50<00:00, 84.25it/s]
# 100%|██████████| 4245/4245 [00:51<00:00, 83.23it/s]
# 70/80 | Loss: 0.5204
# 100%|██████████| 4245/4245 [00:51<00:00, 82.71it/s]
# 100%|██████████| 4245/4245 [00:49<00:00, 84.91it/s]
# 100%|██████████| 4245/4245 [00:50<00:00, 84.63it/s]
# 100%|██████████| 4245/4245 [00:50<00:00, 84.56it/s]
# 100%|██████████| 4245/4245 [00:50<00:00, 84.55it/s]
# 75/80 | Loss: 0.5140
# 100%|██████████| 4245/4245 [00:50<00:00, 84.71it/s]
# 100%|██████████| 4245/4245 [00:50<00:00, 84.67it/s]
# 100%|██████████| 4245/4245 [00:49<00:00, 84.91it/s]
# 100%|██████████| 4245/4245 [00:50<00:00, 84.32it/s]
# second run:
# 65/80 | Loss: 0.5032
# 100%|██████████| 4245/4245 [00:51<00:00, 82.51it/s]
# 100%|██████████| 4245/4245 [00:51<00:00, 82.38it/s]
# 100%|██████████| 4245/4245 [00:51<00:00, 82.23it/s]
# 100%|██████████| 4245/4245 [00:49<00:00, 85.93it/s]
# 100%|██████████| 4245/4245 [00:49<00:00, 86.01it/s]
# 70/80 | Loss: 0.4949
# 100%|██████████| 4245/4245 [00:49<00:00, 85.58it/s]
# 100%|██████████| 4245/4245 [00:49<00:00, 85.11it/s]
# 100%|██████████| 4245/4245 [00:49<00:00, 85.52it/s]
# 100%|██████████| 4245/4245 [00:49<00:00, 85.32it/s]
# 100%|██████████| 4245/4245 [00:49<00:00, 85.59it/s]
# 75/80 | Loss: 0.4893
# 100%|██████████| 4245/4245 [00:49<00:00, 85.31it/s]
# 100%|██████████| 4245/4245 [00:49<00:00, 85.48it/s]
# 100%|██████████| 4245/4245 [00:49<00:00, 85.17it/s]
# 100%|██████████| 4245/4245 [00:49<00:00, 85.21it/s]
#%%
#%%
# torch.save({"state_dict":model.state_dict(),
#             "embedding_dim":model.embedding_dim,
#             "hidden_size":model.hidden_size}, './weights/bahdanau_attention.pth')
#%%
model.load_state_dict(torch.load('./weights/bahdanau_attention.pth')["state_dict"])
# previously I used a frozen W_v (the one with W_v.data.t() which I did by mistake during my
# debugging early in the implementation) yet the model trained seemingly fine. however
# by unfreezing it (i.e. using it normally) we get much better results. you can see
# both weights, the one that uses frozen_W_v is marked accordingly and the one that doesnt
# have any extra tags, is the correct and best model so far which uses the W_v in computation graph
# normally. 
# model.decoder.W_v
#%%
import random
def evaluate(model, sentence, dt):
    model.eval()
    hidden_states=None
    
    with torch.no_grad():
        input_tensor = dt.convert_to_int(sentence, dt.in_vocab, to_tensor=True).to(device)
        outputs, hidden_states, attentions = model(input_tensor, hidden_states)

        _, output_idxs = outputs.topk(1)
        output_idxs.squeeze_()

        output_words = []
        for idx in output_idxs:
            if idx.item() == dt.out_vocab.word2int['<eos>']:
                output_words.append('<eos>')
                break
            output_words.append(dt.out_vocab.int2word[idx.item()])
    return output_words, attentions

def evaluate_model(model, dt, sample_count=5):
    for _ in range(sample_count):
        pair = random.choice(dt.pairs_list)
        print(f'input  : {pair[0]}')
        print(f'target : {pair[1]}')
        output_words, _ = evaluate(model, pair[0], dt)
        print(f"output : {' '.join(output_words)}")

evaluate_model(model, dt, sample_count=10)
#%%

# now lets visualize the attention weights as well and see how they look 
def visualize_attention(input_sentence, output_words, attention_weights):
    _, ax = plt.subplots(figsize=(8, 6))
    # use seaborn heatmap
    sns.heatmap(attention_weights.cpu().numpy(), cmap='viridis', 
                ax=ax,
                cbar=True,
                xticklabels= input_sentence.split() , 
                yticklabels= output_words)
    
    # rotate y-axis labels for better readability
    plt.yticks(rotation=0)
    
    # move the x-axis labels to the top instead of the bottom
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    plt.show()

def evaluate_and_visualize_attention():
    # note if you get some keyerror, its because we limited the dataset so much
    # it couldnt create a big enough vocab for the language. limitting the sequence length
    # too much can result in this error.
    # also note we use lowercase letters in our vocab, so we can use upper case letters here
    # if we do we need to make them lower case before feeding them to our model (also remove dots as well)
    # todo: preprocess the text properly so the user is not bothered with anything except 
    # inputing the texts
    test_sentences = [("he is not as tall as his father", "il n'est pas aussi grand que son pere"),
                      ("I am too tired to drive", "je suis trop fatigue pour conduire"),
                      ("I am sorry if this is a silly question", "je suis desole si c'est une question idiote"),
                      ("I am really proud of you", "je suis reellement fiere de vous"),
                      ("I don't trust anybody", "Je ne me fie à personne"),
                      ("He can swim like a fish", "Il est capable de nager comme un poisson"),
                      ("Don't mind me Just keep doing what you were doing", "Ne fais pas attention à moi. Continue ce que tu étais en train de faire"),
                      # starting with larger sequences, we can see how trimming the sequences
                      # during dataset creating results in model outputs.
                      ("I can't believe that you aren't at least willing to consider the possibility of other alternatives", "Je n'arrive pas à croire que vous ne soyez pas tout au moins disposées à envisager d'autres possibilités"),
                      ("It may be impossible to get a completely error free corpus due to the nature of this kind of collaborative effort However if we encourage members to contribute sentences in their own languages rather than experiment in languages they are learning we might be able to minimize errors", "Il est peut-être impossible d'obtenir un Corpus complètement dénué de fautes, étant donnée la nature de ce type d'entreprise collaborative. Cependant, si nous encourageons les membres à produire des phrases dans leurs propres langues plutôt que d'expérimenter dans les langues qu'ils apprennent, nous pourrions être en mesure de réduire les erreurs"),
                      ]

    for en,fr in test_sentences:
        input_sentence = (en if dt.in_lang == 'eng' else fr).lower()
        expected_sentence = (fr if dt.in_lang == 'eng' else en).lower()
        
        output_words, attentions = evaluate(model, input_sentence, dt)
        attentions.squeeze_(3).squeeze_()
        print(f"input    = {input_sentence}")
        print(f"expected = {expected_sentence}")
        print(f"got      = {' '.join(output_words)}")
        visualize_attention(input_sentence, output_words, attentions[:len(output_words), :])

evaluate_and_visualize_attention()

#%%
# as you can see when the sentences are short, we can quickly get pretty good results!
# longer sentences on the other hand are not as good!(this seems to have been improved
# by disabling the truncated BPTT in our code earlier! lstms do not work well on 
# long bodies of texts in general but as we saw our change did infact improve our baseline!)
# anyway, the vocab and training part needs refactoring and we will hopefully do that in the
# next round. 
# meanwhile the official pytorch code example can be read/used as well. 
# I found the dataset that we used here from the official docs, and used their training loop
# as a starting point for our own training loop, I only needed to changed a few parts
# to make it compatible with our own codebase here. so its a decent source you may want to have
# a look at as well.
# with this we conclude our bahdanau attention section and go to the next section 
# which is sentiment analysis!)
# also note that we didnt pick the best model here, just trained for some epochs and then
# ran some quick tests to see if our implementation is correct and working as expected, so
# our intention wasnt to get the best out of these models. in a real situation we would spend
# a lot of time finetuning all aspects of our training in order to achieve the best performance
# so keep that in mind!
#%%

#%% sentiment analysis 
# lets do sentiment analysis . we read a bunch of reviews with their corrosponding labels
# here are the steps we need to take
# we know our label contains words, positive and neagative, so we convert them into numbers, 0, 1
# our reviews must be dgitized so we can feed them into our network  , but before that we need
# to do some preprocessings. the preprocessings include 
# 1. make everything lower case -not needed really
# 2. remove punctuations 
# 3. remove special characters such as \n 
# 4. split only words! we dont want to generate text, so creating dictionaries of characters 
# as apposed to creating dictionary of words is not going to suit us, becasue we intend on 
# using word embedding, and relationship between words need to be found out! this is not possible
# with characters!
# 5. thats nearly basically it(there are still more to this but we are good for now, keep reading!), 
# we need to create two dictionaries for converting word2int and in2words ()
# there is one thing to note though, we start our intergers from 1 and not 0. we will be using 0 
# later on for padding the input so we can have batches of the same size. (we can use any number really
# but 0 is convienet)
# 6.so we order our words based on their frequency! the most frequent ones get to the top and
# the least frequent ones go to the bottom of the list.(why? we'll see)
# 7. an important step is to just normalize/standardize the length. since we want to use batch
# we need to pick a length. we cant use the bigest one, becasue we may waste alot of space, 
# especially if the majority of our samples are way shorter than the biggest sample in the dataset! 
# so we search and remove the larges and smallest ones.
# 8. good lets go 
import numpy as np 
import string
from collections import Counter

import torch
import torch.nn as nn 
import torch.nn.functional as F 
import matplotlib.pyplot as plt
import torch.optim as optim  
%matplotlib inline 

# first lets read our files 
with open('/media/hossein/SSD1/A-Quick-and-Simple-Pytorch-Tutorial/data/reviews.txt','r') as file: 
    reviews = file.read().lower()
with open('/media/hossein/SSD1/A-Quick-and-Simple-Pytorch-Tutorial/data/labels.txt', 'r') as file: 
    labels = file.read().lower()
    
# lets see what we have here
# so we have many reviews that are each separated by a new line
# likewise, each line in the labels belongs to the corrosponding 
# review
print(f'{repr(reviews[:2000])=}')
print(f'{repr(labels[:20])=}')

# ok, good, now lets remove punctuations, this is so that words that are
# adjacent to punctuations are not detected as new words ('go' vs 'go.')
# and basically they are treated the same.
reviews = ''.join([c for c in reviews if c not in string.punctuation])
print(f'{repr(reviews[:200])=}')

# now lets separate each review using \n (line break)
reviews_list = reviews.split('\n')
print(f'{reviews_list[:2]=}')

#good, now lets create our words list from the reviews
reviews_words_list = reviews.split()
print(reviews_words_list[:20])

# now to create our vocabs(word2int/int2word dictionaries), we need to grab a list of unique words
# our reviews_words_list contains a lot of duplicates for each word, so we need to get rid of them
# this is infact to filter such duplicate words. The subsequent sorting is just there to give us back
# an ordered list of words! which we then can use to create our word2int/int2word dictionaries
# and also see what words are more frequently used and what are less frequently used.
# obviously we can use set() to remove duplicates if we dont want the word frequencies!
# and we can use a list comprehension, and not sort anything! but we did it anyway 
word_counts = Counter(reviews_words_list)
# so to make this not completely useless, lets see how good/bad are repeated in our dataset!
print(f"#times 'good' has been used: {word_counts['good']=:,}")
print(f"#times 'bad' has been used : {word_counts['bad']=:,}")
# note that these dont mean anything! these dont necessarily mean there are
# more positive reviews than negative ones, rather they show frequency of use
# a bad review may use several positive words in the review but ultimately show its bad
# but on the otherside, the existence of positive words, increases the chances of a 
# review to be positive. lets not get over our heads and let the model decide after the training:)

# sort these from highest to lowest count, and get the desired word list
# (check IDF -inverse document frequency (comes handy in rag systems/searching mechanism as one of the criterias))
words_sorted_list = sorted(word_counts, key=word_counts.get, reverse=True)

int2word = dict(enumerate(words_sorted_list,1))
word2int = {word:idx for idx,word in int2word.items()}

print(f'{len(word2int)=:,}')
print(f'{len(int2word)=:,}')
# the happens to be the most frequently used word in our dataset! interesting!!! lets move on!
print(f'{int2word[1]=}')
print(f'{word2int["the"]=}')

# now lets digitize our label
# instead of doing '1 if word=='positive' else 0' 
# I simplified it as int(word=='positive')
# if its true, it will be 1, and if its not, 
# it will be 0! its shorter and easier to read!
labels_digitized = [int(word=='positive') for word in labels.split('\n')]
print(f'labels[:50](raw): \n{labels[:54]}')
print(f'labels[:50]: {labels_digitized[:6]}')

# now lets create the digitized version of the reviews 
reviews_digitized=[]
for review in reviews_list:
    # create a temp list of words separated by space
    # for each review by doing .split()
    reviews_digitized.append([word2int[w] for w in review.split()])

print(f'{len(reviews_digitized)=:,}')
print(f'{reviews_list[0]=}')
print(f'{reviews_digitized[0]=}')

#%%
# lets find the longest and shortest reviews
# this allows us to better choose a max-length
# that suits us the best.
# sidenote: note that we are using reviews_digitazed instead of reviews_list
# becasue, reviews_list contains white characters as well which doesnt exist
# in reviews_digitized (we remove them using split()),
# this doesnt change the final outcome though, However it does make our approach
# using reviews_digitized much faster (the max length in review_list is 13,740,
# while in reviews_digitized its only 2,541!
min_len = min([len(r) for r in reviews_digitized])
max_len = max([len(r) for r in reviews_digitized])
print(f'min length: {min_len:,} and max_len: {max_len:,}')

review_length_counts = Counter(len(r) for r in reviews_digitized)
min_len_cnt = review_length_counts[min_len]
max_len_cnt = review_length_counts[max_len]
# this counter gives us the same min/max length as we just calculated 
print(f'min: {min(review_length_counts):,}')
print(f'max: {max(review_length_counts):,}')
# this also allows us to query and see, how many reviews 
# exist with certain length for example there are
# 1 review with length of 0! and 0 reviews with length of 2!
# this makes it easier for us to choose a more suitable max_length
# for our reviewes. 
print(f'reviews with length of 0 words: {review_length_counts[0]}')
print(f'reviews with length of 2 words: {review_length_counts[2]}')
print(f'reviews with length of 150 words: {review_length_counts[150]}')

print(f'max count {max_len_cnt:,}')
print(f'min count {min_len_cnt:,}')

# lets remove the ones with min/max lengths
# new_reviews = [review for review in reviews_digitized if not len(review) in (min_len,max_len)]
# but wait! we dont this, since we also need to remove the corrosponding labels
# so we instead get the index and remove the reviews based on their index
idxs_to_ignore = [idx for (idx,review) in enumerate(reviews_digitized) if len(review) in (min_len,max_len)]
# lets check wether the min_len,max_len are removed! since we had 
assert len(idxs_to_ignore) == min_len_cnt + max_len_cnt, 'min/max length are not removed!'
# now lets grab them reviews!
new_reviews = [review for (idx,review) in enumerate(reviews_digitized) if idx not in idxs_to_ignore]
print(f'{idxs_to_ignore=}')
print(f'{len(reviews_digitized)=:,}')
print(f'{len(new_reviews)=:,}')
# lets remove the labels
new_labels = [label for (idx, label) in enumerate(labels_digitized) if idx not in idxs_to_ignore]
print(f'{len(new_labels)=:,}')
#%%
# Ok now lets create a function for padding our input
# we do this so we can create batches and increase the performance
# Pytorch provides its own functions 
# anyway lets create a function that gets the digitized review and returns a 
# padded numpy array . we define a maximum length , and fill it from the end
def pad_input(new_reviews, max_length =200):
    padded_tensor = torch.zeros(size=(len(new_reviews), max_length), dtype=torch.int32)
    for i,review in enumerate(new_reviews):
        padded_tensor[i, -len(review):] = torch.tensor(review[:max_length])
    return padded_tensor 

reviews_digitized = pad_input(new_reviews, 150)
print(reviews_digitized[:1])
#%%
# now lets define a train/val/test split
training_frac = 0.80
tr_idx = int(reviews_digitized.shape[0] * training_frac)
training_set = reviews_digitized[:tr_idx,:]
remaining_set = reviews_digitized[tr_idx:,:]

val_frac = 0.5
val_idx = int(remaining_set.shape[0]*val_frac)
val_set = remaining_set[:val_idx,:]
test_set = remaining_set[val_idx:,:]
print(f'{training_set.shape=}')
print(f'{val_set.shape=}')
print(f'{test_set.shape=}')

# now labels! 
training_set_label = torch.tensor(new_labels[:tr_idx])
remaining_set_label = torch.tensor(new_labels[tr_idx:])
val_set_label = remaining_set_label[:val_idx]
test_set_label = remaining_set_label[val_idx:]

print(f'\n{training_set_label.shape=}')
print(f'{val_set_label.shape=}')
print(f'{test_set_label.shape=}')
print(f'{training_set_label[2]=}')
# now lets go for the training 
# we can use the torchTensorDataset for this 
#%%
import torch 
import torch.utils.data 
import torch.nn as nn 
import torch.nn.functional as F 


batch_size = 50
# we feed our data and its corrosponding labels , and it will create us a dataset!
train_dataset = torch.utils.data.TensorDataset(training_set, training_set_label)
test_dataset = torch.utils.data.TensorDataset(test_set, test_set_label)
val_dataset = torch.utils.data.TensorDataset(val_set, val_set_label)

# if we choose a batchsize that makes the last batch not full
# assuming we used a batch of 50 e.g., we get an error like: 
# RuntimeError: Expected hidden[0] size (3, 49, 300), got [3, 50, 300]
# which is basically complaining that the last batch wasnt 50
# so one easy way is to use drop_last=True, which ignores that incomplete batch of samples
# other ways like handling this during training are less effective as you have to 
# constantly check for this case which happens only once in an epoch but you check all the 
# time! so this is pretty much the best way unless we want to do the dataloadin part ourseleves
# like before and handle this there! doesnt really make sense to do that so lets just use drop_last;)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,shuffle=True, drop_last=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,shuffle=False)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,shuffle=False)

# view a sample batch !

features, labels=next(iter(train_loader))
print(features.shape)
print(features[:2,:30])
print(labels[:2])
#%%
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_size, output_size,
                       embedding_dim=500, num_layers=1, dropout_ratio =0.5):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=embedding_dim)
        
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first = True, 
                            dropout=dropout_ratio)
        self.fc = nn.Linear(hidden_size, output_size)
        self.lstm_output_dropout = nn.Dropout2d(0.3)
        self.embd_output_dropout = nn.Dropout2d(0.3)
        

    def forward(self, x, hidden):
        
        embeddings = self.embedding(x.long())
        embeddings = self.embd_output_dropout(embeddings)
        
        outputs, hidden = self.lstm(embeddings, hidden)
        outputs = self.lstm_output_dropout(outputs)

        # we want 0 or 1 
        outputs = F.sigmoid(self.fc(outputs.reshape(-1, self.hidden_size)))
        # lets make the output batch_first again 
        outputs = outputs.view(x.size(0), -1)
        # we need the last sequence output, so we get the last one using -1
        return outputs[:,-1], hidden

    # we can simply use None to initialize the hiddenstate instead of this!
    # I leave it here as a reminder though!
    def init_weights(self, batch_size, device):
        weight = next(self.parameters()).data
        hidden_states = (weight.new_zeros(self.num_layers, batch_size, self.hidden_size).to(device),
        weight.new_zeros(self.num_layers, batch_size, self.hidden_size).to(device))
        return hidden_states

# instantiate the model w/ hyperparams
vocab_size = len(word2int)+1 # +1 for the 0 padding 
output_size = 1
embedding_dim = 500 # embedding size, is important!
hidden_dim = 300 # this is important as well!
# with embed 1000, hd 300 : 
n_layers = 3#78.68 with 1 layer, 80% with 2 layers and 81.88 with 3 layers 

model = SentimentLSTM(vocab_size=vocab_size, 
                      hidden_size =hidden_dim,
                      output_size=output_size,
                      embedding_dim=embedding_dim,
                      num_layers=n_layers,
                      dropout_ratio=0.5)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

features, labels=next(iter(train_loader))
features = features.to(device)
print(model)
x,y = model(features, None)

# binary crossentropy becasue we have a single output, its either true of false (>0.5 or not)
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 4 
val_interval = 100
clip_threshold = 5
hidden_states = model.init_weights(batch_size,device)

for e in range(epochs):
    # simply do hidden_states = None!
    hidden_states = model.init_weights(batch_size,device)
    for i,(features, labels) in enumerate(train_loader):
        model.train()

        features = features.to(device)
        labels = labels.to(device)

        outputs, hidden_states = model(features, hidden_states)
        # print(f'hiddens.shape: {hidden_states[0].shape} {hidden_states[1].shape}')   
        hidden_states = tuple(h.data for h in hidden_states)
        
        optimizer.zero_grad()
        loss = criterion(outputs, labels.float())
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), clip_threshold)
        optimizer.step()

        if i%val_interval == 0:
            # see setting it to None works just as well. 
            # infact you should use this instead of the
            # other old way!
            hidden_states = None
            val_losses = []
            for feats, labels in val_loader:
                with torch.no_grad():
                    model.eval()

                    feats = feats.to(device)
                    labels = labels.to(device)

                    outputs, hidden_states = model(feats, hidden_states)
                    hidden_states = tuple(h.data for h in hidden_states)

                    val_loss = criterion(outputs, labels.float())
                    val_losses.append(val_loss.item())

            print(f'Epoch {e}/{epochs} | Loss: {loss.item():.4f} | Val Loss: {np.mean(val_losses):.4f}')

# now lets test this on our test set, here calculate accuracy and loss@
# hidden_states = model.init_weights(batch_size, device)
hidden_states = None
test_losses =[]
acc = 0

model.eval()
for i, (features, labels) in enumerate(test_loader):
    with torch.no_grad():
        features = features.to(device)
        labels = labels.to(device)
        outputs, hidden_states = model(features, hidden_states)
        hidden_states = tuple(hidden.data for hidden in hidden_states)
        loss = criterion(outputs, labels.float())
        test_losses.append(loss.item())
        
        # grab the output by rounding it (it becomes 0/1)
        preds = torch.round(outputs)
        acc += (preds==labels.float()).float().mean().item()
                
print(f'Accuracy= {(acc/len(test_loader))*100.0:.2f}')
print(f'Loss: {np.mean(test_losses):.4f}')

#%%
# lets test this on our own data! 
# this is called inference ! give it some random text and see 
# if it can correctly classify it as positive or negative! 
from string import punctuation
def tokenize_input(review='damn you son of a gun. that was hell!'):
    #first lower case all words 
    review = review.lower()
    # second remove all punctuations
    all_text_no_punc = ''.join(c for c in review if c not in punctuation)
    # third get all the words 
    words_list = all_text_no_punc.split()
    # digitized=[]
    digitized = [[word2int[w] for w in words_list]]
    return digitized

test_review_neg = 'The worst movie I have seen; acting was terrible and I want my money back. This movie had bad acting and the dialogue was slow.'

review_digitized = tokenize_input(test_review_neg)
review_padded = pad_input(review_digitized, 200)

# grab the device from a model parameter
device = next(model.parameters()).device
review_padded = review_padded.to(device)

hidden_states = None
outputs, hidden_states = model(review_padded, hidden_states)

pred = torch.round(outputs)

print(f"review: '{test_review_neg}'")
print(f"The review is {'negative' if pred.item() == 0 else 'positive'}")

# while this might work, this is in no way a good model, to get a decent performance we 
# would want to use a better/larger model/better regularization/optimization regime
# we usually dont bother using lstms for these kinds of tasks anymore, we now use transformers!
# transformers are our goto models when dealing with anything nlp! we will cover transformers in later chapters. 
#%%
# 
# Word Embedding - SkipGram
# Add the intro from my old word2vec.py
#
# For word embedding training we have several methods, word2vec is one of them
# here we will be using the skipgram model. we can train skipgram model with negative sampling
# we will implement both!
# the skipgram model is simply an embedding layer with a fullyconnected/linear layer 
# followed by a logsoftmax.
# the skipgram model with negative sampling, is two embedding layers, 
# the embedding layer for the input and output must be the same, 
# we feed our word to input embedding get target words, feed those
# target words to output embedding and must get back the initial word 
# that was fed into input emebdding.

# There are several methods for training word embeddings, Word2Vec is the most popular one.
# In Word2Vec, we have two main approaches/implementations: Skip-Gram and CBOW (Continuous Bag of Words). 
# Here, we’ll focus on the Skip-Gram model, and implement it with and without Negative Sampling.
#
# the normal skip-gram model works this way:
# given a center word (input word), the model predicts the context words i.e. the
# surrounding words. the model itself is pretty simple. its made of an embedding
# layer, to map input words to a vector representation then a fully connected layer/
# linear layer, which transforms the embeddings from previous stage to be fed into
# a log-softmax layer, to output the probabilities of said context words.
# The model is trained to maximize the probability of correct context words 
# appearing around a given input word.

# Skip-Gram with Negative Sampling
# In the Skip-Gram with Negative Sampling variant, we modify the architecture
# and the loss function to be more efficient and effective as well. the skipgram
# model incurs a lot of overhead simply because of the way it works. in this version
# we now have two embedding layers, one for the input words and one for the output 
# (context) words. the embeddings from these layers are adjusted separately.
# the training prcess goes like this: 
# we feed an input word into the input embedding layer to get its vector representation.
# we then use this representation to predict multiple target context words by/through
# the output embedding layer.
# instead of updating all context word probabilities (as in the full softmax like before),
# we use negative sampling in which we select a small number of positive pairs 
# (input word and actual context words) and then randomly sample several negative 
# pairs (input word and random/non-context words) from a noise distribution.
# this allows us to have high similarity between input and actual context words
# and low similarity between input and randomly sampled negative words.
# and it works much better and tranis much faster provided we chose the hyperparameters
# properly(otherwise it maybe the otherway around!see training logs)!

import random
import string
from collections import Counter

import numpy as np 
import pandas as pd
from tqdm import tqdm

# for visualizations that need dimensionality reduction
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

#
# The first thing that we do is we need a body of text that we can use as our dataset 
# and learn the embeddings from. lets use the text8 in data dirctory
dataset_path = '/media/hossein/SSD1/A-Quick-and-Simple-Pytorch-Tutorial/data/text8.txt'
with open(dataset_path,'r') as file : 
    corpus_raw = file.read()
    
# now we have all the contents which include, words and punctuations and white spaces
# since we want to learn proper embeddings for 'words' we can remove the punctuations
# altogether. we also can remove less frequent words. we may also want to remove 'some'
# (emphasize on 'some') common and uncommon words as well. why? becasue words such as the, a
# are very common, but they dont really provide much insight to the surrounding words,
# so removing them can help us achieve better embeddings as these noises are removed. 
# how do we do that? we use a mikolove formula for that which we will get to shortly.
# but first lets do : 
# 1. remove punctuations ,actually replacing them with proper symbols
# 2. remove less frequent words 
# 3. balance words based on mikolove criteria to address imbalanced data
#
# removing punctuations 
def remove_punctuations(input_corpus):
    # this is kind of a glorified version of replace()!
    # you can think of str.translate() as a more efficient and flexible version of replace().
    # when we need to perform multiple string replacements.
    # It allows us to create a translation table with str.maketrans() 
    # and apply all the replacements in a single pass, 
    # which can be faster and more concise than chaining multiple replace() calls.
    translation_table = str.maketrans({'.': '<PERIOD>',
                                       '!': '<EXCLAMATION>',
                                       '(': '<LPAR>',
                                       ')': '<RPAR>',
                                       '[': '<LBRAC>',
                                       ']': '<RBRAC>',
                                       '#': '<HASH>',
                                       "'": '<SingleQuote>',
                                       '"': '<DoubleQuote>',
                                       ':': '<COLON>',
                                       '$': '<DOLLAR>',
                                       '%': '<PERCENT>',
                                       ';': '<SEMICOLON>',
                                       '-': '<DASH>'})
    input_corpus = input_corpus.translate(translation_table)
    return input_corpus

corpus = remove_punctuations(corpus_raw)
# now remove less frequent words 
# calculate mikolove formula
# create subsampling and get final dataset
# create and sort vocab from word counts from final dataset
# create word2int int2word using vocab
# convert dataset to digitize for training

words_in_corpus = corpus.split() # should be 253,854 words
word_counts = Counter(words_in_corpus)
# total number of words in the corpus
total_count = len(words_in_corpus) # or we could also do sum(word_counts.values())
print(f'Total number of words in the corpus: {total_count:,}')
# now lets remove scarce/rarely used words!
min_word_count = 5
words_in_corpus = [word for word in words_in_corpus if word_counts[word] > min_word_count]
print(f'Total number after removing words with count<{min_word_count}: {len(words_in_corpus):,}')
# now lets get the new words count again
word_counts = Counter(words_in_corpus)
#update total_count based on latest words_in_corpus
total_count = len(words_in_corpus) # or we could also do sum(word_counts.values())

# subsampling
# now its time to apply the mikolove formula/method. 
# for this we need to first calculate the word
# frequencies with respect to the whole corpus
word_freqs = {word: freq/total_count for word, freq in word_counts.items()}
print(f'number of times "the" is repeated in the corpus: {word_freqs['the']:.4f}')
# this is the threshold, see the notes ahead 
# for how to choose the right value
threshold = 1e-4 # 1e-5
print(f'subsampling threshold: {threshold}')
# here we calculate the mikolov formula (1-sqrt(t/f(w)))
# where t is the threshold parameter and f(w_i) is the 
# frequency of the ith word (w_i) in the whole dataset.
# this is basically the probability that a word is discarded.
# as for the idea behind doing so, recall that frequent words such as "the",
# "of", "for", etc don't provide much context to the surrounding words. 
# Therefore discarding some of these words, can practically remove some 
# of the noise in our data. this will both speed up our training 
# and also improve the final representation.(more on this in a moment)
# This process is called subsampling by Mikolov. 
# For each word in the training set, we'll discard it with probability given by
# (1-sqrt(t/f(w)))
print(f'dataset before subsamping:')
print(f"{' '.join(words_in_corpus[:50])}")

# probablity_drop = {word: 1 - np.sqrt(threshold / word_freqs[word]) for word in word_counts}
# print(f'{probablity_drop=:}')
# the probablity of 1-probablity_drop (which means the probablity of being kept!)
# make our new word_list out of our original corpus
# word_list = [word for word in words_in_corpus if random.random() < (1-probablity_drop[word])]

# we can simply directly calculate keep_probablity! 
probablity_keep = {word: np.sqrt(threshold / word_freqs[word]) for word in word_counts}
# print(f'{probablity_keep=:}')
#todo: change word_list to dataset or corpus
word_list = [word for word in words_in_corpus if random.random() < probablity_keep[word]]
print(f'dataset after subsamping:')
print(f"{' '.join(word_list[:50])}")

# sidenote:
# concerning the value for threshold, heres a quick rule of thumb:
# we use threshold to control how aggressively frequent words are discarded
# its basically our attempt at balancing the words in our dataset so
# a lower threshold (1e-5) = more aggressive subsampling of common words
# a higher threshold (1e-3) = less aggressive subsampling
#
# the formula was 1-√(t/f(w)) which is the probablity of discarding a word,
# √(t/f(w)) therefore is the probablity of keeping word w (prob_keep(w)) so
# when f(w)>t it means the word is common so lower keep probability
# when f(w)<t it means the word is rare so higher keep probability 
# 
# what values are usually good to use?
# the default is 1e-5 which is mikolov's original recommendation in the paper
# and has been used on large datasets. for small datasets a value like 1e-4 to
# 1e-3 seems fine. for specialized vocabularies we need to experiment with 
# 1e-5 to 1e-3!
#
# here is a quick table showing what each value for threshold means in practice: 
#    Word                   Keep Prob   Keep Prob     Effect Change
#  Frequency                (t=1e-4)    (t=1e-3)
# Very common(f=0.05)         1.4%	       14%	      10× more common words kept
# Common(f=0.01)	          3.2%	       31.6%	  10× more kept
# Medium(f=0.001)	          10%	       100%    	  No subsampling
# Rare(f=0.0001)              100%	       100%	      No change
# Very rare(f=0.00002)	      100%	       100%	      No change
# 
# as you can see, the threshold value first and formost affects the very common
# wordsand for medium or rare/very rare words, they are all included basically no
# subsampling happens for them!
#
# so low threhsolds like 1e-5 is recommended for most applications,
# we especially use it when we need more aggressive subsampling
# especially when we are dealing with large datasets with extreme
# word frequency imbalances, not only it reduces the trainig time
# it will also improve representation of less common/medium-frequency
# words and result in a higher quality embeddings.
# 
# higher threshold such as 1e-3 on the other hand, are used for
# less aggressive subsampling process, for cases where the dataset
# is not that large and we want to keep more context words,
# its useful when common words carry important meaning
# and obviously it imposes more overhead on our training (it will
# take longer to train because there will be more words to work with!)
# we can always query the dataset before and after our subsampling and 
# have an indea where we're going with our changes!
#
#sidenote:
# since here we want to grab the words(keep them instead of discarding them),
# we use 1-prob, which means grab the words that are more probable than being
# discarded(grab less frequent words more often than words such as 'the', 'of',
# 'and' which are much more frequent)
# note that if we dont do this obviously we will be having a larger word_list, 
# and it would take much longer to train our model and have other serious implications 
# that is, no mikolov's subsampling is done which means frequent words are
# overrepresented and rare words are undersampled.
# not only this has a negative impact on training efficiency but worse results in 
# low embedding quality as well.
# (basically not doing this leads to overrepresentation of these words in our 
# training data and because they are too frequent they are less informative and
# thus contribute less to learning meaningful representations. 
# keeping them therefore only increases the computational overhead. moreover, 
# rare or contextually important/rich words (which provide valuable information
# for learning but are less frequent) will also be more likely to be dropped! 
# this impacts the diversity of our training data lowering it making it harder
# (or even impossible) for the model to learn meaningful embeddings and capture 
# meaningful semantic relationships for these words.
# after all word embeddings rely on the contextual diversity provided by different
# words and discarding rare words and keeping frequent ones like that, will lead 
# to low quality embeddings that perform badly/poorly on downstream tasks, such 
# as semantic similarity, word analogy tasks etc)

# now based on this filter, we now have 4,655,219 words! out of the original 17m words!
print(f'{len(word_list)=:,}')
# now lets get the new words count again
word_counts = Counter(word_list)
# and sort them in a descending fashion so that the 
# most frequent ones come first. the sorting comes 
# handy later on for visualization and evaluation purposes
#sidenote: 
# previously I had done a grave mistake and it was to assign
# this to word_list instead! this not only destroyed our training data
# making it 63k instead of 4.6m! but even worst it destroys the context
# the whole dataset was gone, and we only were left with 63 indivisual words!
vocab_words = sorted(word_counts, key=word_counts.get, reverse=True)
# and now lets create the dictionaries for word2int and int2word
int2word = dict(enumerate(vocab_words))
word2int = {word:idx for idx,word in int2word.items()}
print(f'{len(int2word)=:,}')
print(f'{len(word2int)=:,}')
# # get digitized word
# word_list_digitized = [word2int[word] for word in word_list]

#%%
# some common-uncommon words. 
# earlier words are more frequently used in our dataset
# while the later indexes are less frequent. we go from
# most frequent words to the least frequenet words!
# we can use these later to assess how our model is doing
base_common=250
base_uncommon=2000
print(f'int2word[250:260]: {[int2word[base_common+i] for i in range(10)]}')
print(f'int2word[2000:2010]{[int2word[base_uncommon+i] for i in range(10)]}')

# ok now we need to get the target words for each word. we define a function that 
# accepts a input list, index, windows size 
def get_target(word_list, current_idx , window_size=5):
    # select a random length between 1 and window_size
    # we are trying to get words around a given (word index)
    random_len = random.randint(1, window_size)
    start_idx = current_idx - random_len if (current_idx - random_len) > 0 else 0
    end_idx = current_idx + random_len if (current_idx + random_len) <len(word_list) else len(word_list)

    before_words = word_list[start_idx:current_idx]
    after_words = word_list[current_idx+1:end_idx+1]
    return before_words + after_words
#%%
# lets test with text instead of list of words to see if it works!
target = get_target('Hello brother, howdy?', current_idx=10, window_size=5)
# yup it does!
print(f'{target=}')
# now we need to create a batching mechanism
#test again
idx_list = [i for i in range(10)]
idx = random.randint(1,5)
window = 5
print(f'input : {idx_list}')
print (f'idx: {idx} window: {window}')
targets = get_target(idx_list,  idx,  window)
print(f'{targets=}')
# get digitized word
word_list_digitized = [word2int[word] for word in word_list]
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

w,t = next(iter(get_batch(word_list_digitized, 3)))
print(f'{w=}')
print(f'{t=}')

#%% 
# now ok. its time to create our model for embedding learning using skipgram model!
class SkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_size=300):
        super().__init__()

        self.embedding_layer = nn.Embedding(vocab_size, embedding_size)
        self.fc = nn.Linear(embedding_size, vocab_size)
        
    def forward(self, x):
        x = self.embedding_layer(x)
        x = self.fc(x)
        log_probs = F.log_softmax(x,dim=1)
        return log_probs
#%%
# lets create a cosine similarity for validation words, to see how certain words
# are doing. we create some random words, and take their cosine simlarity in the 
# embeddings. if their target words are plausible then we are good! lets do this 

def evaluate_embeddings(embedding_layer, window_size=100, validation_size=16,
                        common_start_index=0, uncommon_start_index=2000):
    # first lets create some random word indexes
    # we get some common words and some uncommon words. if you recall, we sorted
    # our vocab based on word frequencies, so that the most frequent ones stay 
    # at the very begining and the less frequent ones stay at the very end, therefore
    # choosing a smaller index(i.e. common_start_index) means choosing more frequently
    # used words, and a larger index(i.e. uncommon_start_index) means, choosing less
    # frequently used words.
    device = next(embedding_layer.parameters()).device
    # random.sample(sequence, k) accepts a list/tuple/string or set with
    # an integer value k as the length of a sample.
    common_words_idx = torch.tensor(random.sample(range(common_start_index, common_start_index+window_size), validation_size//2))
    uncommon_words_idx = torch.tensor(random.sample(range(uncommon_start_index, uncommon_start_index+window_size), validation_size//2))
    # append both more common and less common word ids together  
    val_words = torch.concat((common_words_idx ,uncommon_words_idx)).to(device)
    embeddings = embedding_layer(val_words) #(K,E)

    # calculate cosine similarity
    # the formula for cosine similarity is (A, B) = (A · B) / (||A|| * ||B||)
    # ||A|| means L2 norm (euclidian norm) or p2-norm(magnitude) for A,
    # so ||A||*||B|| means our denominator
    # is the multiplication of l2-norms of both tensors.
    # for calculating norms, we have a few options. we can use the old schoolway
    # which is embeddings.pow(2).sum(dim=1,keepdim=True).sqrt()
    # we use keepdim so the shape is retained for our next multiplication
    # we can also use torch.linalg.norm(embeddings, dim=1, keepdim=True) for this
    #
    # embeddings_norm = embeddings.pow(2).sum(dim=1,keepdim=True).sqrt() #(K,1)
    # all_embeddings_norm = embedding_layer.weight.pow(2).sum(dim=1,keepdim=True).sqrt() #(N,1)
    # 
    # our denominator (||A||*||B||) should have a shape (K,N) 
    # magnitutes = torch.mm(embeddings_norm, all_embeddings_norm.t())
    # 
    # and finally to calculate the similarity we put all of this together
    # note the epsilon(1e-8) is there for numerical stability in case magnitudes 
    # face underflow (due to tiny float numbers being multiplied!)
    # similarity1 = torch.mm(embeddings, embedding_layer.weight.t())/ (magnitutes + 1e-8)
    # 
    # or we can use pytorch's builtin cosine_similarity function
    # this is how we use a single embedding example using pytorch's function
    # since we have more than 1 embedding, we have to call it for 
    # each embedding in a loop
    # similarities = []
    # for embedding in embeddings:
    #     similarity_score = F.cosine_similarity(embedding.unsqueeze(0), embedding_layer.weight)
    #     similarities.append(similarity_score)
    # similarities = torch.stack(similarities)  # (N, vocab_size)
    # 
    # this is not efficent though!, we can do better!
    # F.cosine_similarity can also be used with vectors! it can compare 
    # a set of vectors to another set using broadcasting! so if we unsqueeze
    # the tensors to make their dimensions compatible for broadcasting, we can
    # have a oneliner!
    # for example if our embeddings shape is [16, 300] and 
    # mbedding_layer.weight shape is [60k, 300] we can do the broadcasting like this
    # embeddings:            [16 , 300] -> [16, 1, 300]
    # mbedding_layer.weight: [60k, 300] -> [1, 60k, 300]
    # now pytorch can broadcast them to a common shape [16, 60k, 300] and compute
    # the similarity along the last dimension (dim=2).
    # The result is [16, 60k] and is exactly what we want!
    # 
    similarity = F.cosine_similarity(embeddings.unsqueeze(1), embedding_layer.weight.unsqueeze(0), dim=2)
    # we can check our implementation is the same as pytorchs by comparing the two results
    # note because of numerical instabilities in floating point operations, our version
    # is only 1e-7 accurate and its ok for us if not we can use pytorchs version
    # assert torch.allclose(similarity1,similarity, atol=1e-7) ,'not the same!'
    return val_words, similarity 

# lets have an analogy test (i.e. a is to b as c is to d).
# like king to man is queen to woman, things like that
# its a simple arithmetic operation vec(a) - vec(b) + vec(c) = vec(d)
def check_analogy_test(word_a, word_b, word_c, embedding_layer, word2int, int2word, top_k=5):
    for w in [word_a, word_b, word_c]:
        if w not in word2int:
            print(f"Error: word '{w}' does NOT exist in the vocabulary!")
            return

    # we normalize the entire vocabulary embedding matrix once for efficiency
    # after this using a simple dotproduct we can get consine similarity!
    all_embeddings_norm = F.normalize(embedding_layer.weight, p=2, dim=1)
    
    # get the normalized vectors for our three words
    vec_a = all_embeddings_norm[word2int[word_a]] #e.g. king
    vec_b = all_embeddings_norm[word2int[word_b]] #e.g. man
    vec_c = all_embeddings_norm[word2int[word_c]] #e.g. woman
    
    # its anologous to d ≈ a-b + c
    result_vec =  vec_a - vec_b + vec_c
    
    # find the cosine similarity between our result vector and all words in the vocab
    # torch.mv does matrix-vector multiplication but 
    similarities = torch.mv(all_embeddings_norm, result_vec)
    
    # dont forget to exclude input words from top results (forexample woman
    # shouldnt be in the results if its in the query!)
    exclude_indices = [word2int[w] for w in [word_a, word_b, word_c]]
    # we disable them by assigning thme a very low number denoting their rank/similarity score
    similarities[exclude_indices] = -10.0
    
    # get the top k results
    top_scores, top_indices = torch.topk(similarities, k=top_k)
    
    print(f"Analogy: {word_a} - {word_b} + {word_c} = ?")
    print("Top results:")
    for i in range(top_k):
        idx = top_indices[i].item()
        word = int2word[idx]
        score = top_scores[i].item()
        print(f"  - {word:<15} (score: {score:.4f})")
    # print()

def evalualte_model_quality(embedding_layer, int2word, window_size, validation_size, common_start_index=250, uncommon_start_index=2000,topk=6):
    # get examples and their similarities 
    valid_examples, valid_similarities = evaluate_embeddings(embedding_layer,
                                                            window_size=window_size,
                                                            validation_size=validation_size,
                                                            common_start_index=common_start_index,
                                                            uncommon_start_index=uncommon_start_index)
    # get topk highest similar words
    _, closest_idxs = valid_similarities.topk(topk)
    valid_examples = valid_examples.cpu()
    closest_idxs =  closest_idxs.cpu()
  
    print(f' Validation similarity test:')
    for i, valid_idx in enumerate(valid_examples):
        closest_words = [int2word[idx.item()] for idx in closest_idxs[i]][1:]
        print(f"  -{int2word[valid_idx.item()]:<10}| {', '.join(closest_words)}")
    
    # run some analogy tests at the end, note since our dataset is very limited
    # we cant do many diverse tests for example I cant do Iran/Tehran, there are
    # only 4 sentences about Iran in the whole dataset, so is the case for many
    # many other things, like Japan, China, other countries, verb tenses, things, etc
    # so we have to keep it simple here. 
    # this should be enough for our evaluation/testing purposes.
    # we are checking analogies like a:b :: c:d
    # king:man :: queen:woman (we should get woman!)
    check_analogy_test('king', 'man', 'woman', embedding_layer, word2int, int2word)
    # we should get mother
    check_analogy_test('father', 'man', 'woman', embedding_layer, word2int, int2word)
    # we should get wife
    check_analogy_test('husband', 'man', 'woman', embedding_layer, word2int, int2word)

# model = SkipGram(len(word2int), 300)
# evaluate_embeddings(model.embedding_layer,
#                     window_size=5,
#                     validation_size=8,
#                     common_start_index=100,
#                     uncommon_start_index=2000)

#%%
# random.seed(10)
# np.random.seed(10)
seed = 10
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# now lets start the actual training!
device = 'cuda' if torch.cuda.is_available() else 'cpu'
embedding_size = 300
vocab_size = len(word2int)

model = SkipGram(vocab_size, embedding_size)
model.to(device)
# note we are using log_softmax, so we must use nllloss here,
criterion = nn.NLLLoss()  # or implement negative sampling (our next implementation)
optimizer = optim.Adam(model.parameters(), lr=0.003)

# Training loop
num_epochs = 50
batch_size = 512
window_size = 5
validation_size = 8
# with 4.6m words in our dataset after all normalization/cleaning,
# with a batchsize of 512, there will be around 9000 iterations
interval = 9000

for epoch in range(num_epochs):
    losses = []
    for i,(X, Y) in enumerate(get_batch(word_list_digitized, 
                                        batch_size=batch_size, 
                                        window_size=window_size), start=1):
        X = torch.LongTensor(X).to(device)
        Y = torch.LongTensor(Y).to(device)
        
        output = model(X)
        
        loss = criterion(output, Y)
        losses.append(loss.item())
        
        optimizer.zero_grad()        
        loss.backward()
        optimizer.step()
        
        if i%interval==0:
            evalualte_model_quality(model.embedding_layer, int2word, window_size, validation_size, 
                                    common_start_index=250, uncommon_start_index=2000)#4000

            print(f' -Epoch {epoch}/{num_epochs} | Iter {i} | Loss: {np.mean(losses):.4f}')

    print(f"Epoch {epoch}/{num_epochs}, Loss: {np.mean(losses):.4f}")
    torch.save({"state_dict":model.state_dict(),
                "epochs":epoch,
                "loss":np.mean(losses),
                "vocab_size":vocab_size,
                "embedding_size":embedding_size,
                "word2int": word2int,
                "int2word": int2word}, "./weights/skipgram_model.pth")

# after 5 epochs this is what we get: 
# the loss doesnt show it properly, but using similarity check
# we can clearly see the converging process where similar words
# start to show up eventually.
# also note the first test was done using a threshold =1e-5 
# while the second was done later using 1e-4. the second was
# is far more accurate as you can see. not only it achieves
# a lower loss in much fewer epochs(faster convergence rate),
# but it provides a much better/detailed embeddings
#  Validation similarity test:
#   -any       | peru, zodiacal, posed, encouraging, sociologists
#   -being     | vocals, caco, herbs, averted, ean
#   -english   | dissidents, taraza, eyre, pes, minicomputer
#   -city      | nicea, coherentism, grinnell, ampex, verdon
#   -wheel     | incision, destroys, casper, scone, metals
#   -viii      | swirling, metazoa, redefining, zionists, safl
#   -timeline  | rapeseed, pentagons, sentences, coronets, reset
#   -coined    | susa, auditory, punitive, casas, afdb
#  -Epoch 0/50 | Iter 100 | Loss: 11.0113
# and after afew more training: 
# we see it starts to learn some relationships among some words:
#  Validation similarity test:
#   -then      | provence, if, maximally, place, geometrically
#   -english   | distinctions, words, language, canadian, ngg
#   -city      | cities, tiers, suburbs, metropolitan, town
#   -any       | definition, disjoint, does, have, surah
#   -timeline  | profile, miscalculation, experienced, hodges, paper
#   -wheel     | incision, destroys, acrylic, costanza, casper
#   -coined    | auditory, geological, grimaldi, susa, linguist
#   -viii      | vii, pope, unwilling, staves, didier
#  -Epoch 1/50 | Iter 4300 | Loss: 9.9559
# and it gets better as training goes on . 
# interstingly loss despite decreasing, doesnt properly 
# shows the rate of change as clearly as we expect/want,
# the similarity shows the improvements much better!)
# -Epoch 5/50 | Iter 3500 | Loss: 9.3953
#  Validation similarity test:
#   -any       | if, be, otherwise, certain, must
#   -english   | french, scottish, british, welsh, dictionary
#   -being     | as, been, or, less, but
#   -then      | if, a, the, x, function
#   -wheel     | wheels, switches, rear, brakes, drive
#   -timeline  | sites, com, modern, links, timelines
#   -coined    | term, describe, phrase, popularized, synonymously
#   -graduate  | undergraduate, education, college, faculty, students
#  -Epoch 5/50 | Iter 3600 | Loss: 9.3952
#
# Validation similarity test:
#   -city      | downtown, cities, metropolitan, town, located
#   -being     | often, sometimes, so, very, be
#   -any       | not, if, be, can, apply
#   -english   | french, american, seven, irish, swedish
#   -viii      | vii, ix, iv, iii, england
#   -wheel     | wheels, brakes, rear, shaft, engine
#   -timeline  | external, com, references, online, faq
#   -coined    | describe, term, surrealism, popularized, myth
#  -Epoch 15/50 | Iter 9000 | Loss: 9.1761
# Epoch 15/50, Loss: 9.1774

# second run:
#  Validation similarity test:
#   -given     | infty, therefore, attributes, systemic, modernise
#   -place     | vena, pressuring, retracting, jsf, tape
#   -important | ieoh, harsher, josephson, subversive, arbitrage
#   -sometimes | linebacker, limon, igreja, ciii, luminosity
#   -algebra   | axiomatic, legg, riggs, grundgesetz, paine
#   -selected  | agreements, leda, identically, third, democr
#   -pop       | hall, dreamland, crazy, rediscovered, undocumented
#   -magnetic  | steered, cystitis, conjunct, addition, homomorphisms
# Analogy: king - man + woman = ?
# Top results:
#   - hurrian         (score: 0.3671)
#   - flamingo        (score: 0.3556)
#   - caddo           (score: 0.3429)
#   - wy              (score: 0.3373)
#   - neale           (score: 0.3366)
# Analogy: father - man + woman = ?
# Top results:
#   - entwistle       (score: 0.5342)
#   - burn            (score: 0.4166)
#   - sucessful       (score: 0.4037)
#   - diaconate       (score: 0.3947)
#   - frankland       (score: 0.3770)
# Analogy: husband - man + woman = ?
# Top results:
#   - origen          (score: 0.4130)
#   - witches         (score: 0.4119)
#   - comenius        (score: 0.4065)
#   - pair            (score: 0.4028)
#   - leroux          (score: 0.4010)
#  -Epoch 0/50 | Iter 9000 | Loss: 9.6313
# Epoch 0/50, Loss: 9.5002
#  Validation similarity test:
#   -place     | advantage, places, recharge, unbeaten, wtc
#   -sometimes | commonly, often, frequently, usually, as
#   -information| data, file, link, processing, database
#   -given     | is, gives, for, frac, consider
#   -pop       | rock, music, song, songs, album
#   -magnetic  | electrical, moving, vacuum, velocity, effect
#   -algebra   | logic, mathematical, notation, theorem, finite
#   -widespread| controversy, usage, modernization, popular, impact
# Analogy: king - man + woman = ?
# Top results:
#   - sweden          (score: 0.4037)
#   - daughter        (score: 0.4023)
#   - wife            (score: 0.3829)
#   - kings           (score: 0.3806)
#   - emperor         (score: 0.3804)
# Analogy: father - man + woman = ?
# Top results:
#   - mother          (score: 0.5696)
#   - wife            (score: 0.5379)
#   - married         (score: 0.5151)
#   - son             (score: 0.5079)
#   - daughter        (score: 0.4701)
# Analogy: husband - man + woman = ?
# Top results:
#   - daughter        (score: 0.5625)
#   - married         (score: 0.4869)
#   - wife            (score: 0.4720)
#   - wives           (score: 0.4681)
#   - children        (score: 0.4615)
#  -Epoch 1/50 | Iter 9000 | Loss: 9.1859
# Epoch 1/50, Loss: 9.1383
#  Validation similarity test:
#   -information| data, file, database, reports, message
#   -place     | advantage, places, seriously, preposition, event
#   -important | vital, importance, difficult, fundamental, crucial
#   -given     | gives, defined, denoted, derived, assigned
#   -magnetic  | electric, electrical, tape, electromagnetic, velocity
#   -algebra   | mathematics, theorem, mathematical, logic, algebraic
#   -widespread| occurred, caused, corruption, resulted, due
#   -selected  | clifton, included, none, chosen, collected
# Analogy: king - man + woman = ?
# Top results:
#   - queen           (score: 0.4261)
#   - daughter        (score: 0.4249)
#   - duke            (score: 0.4198)
#   - kingdom         (score: 0.4186)
#   - emperor         (score: 0.3869)
# Analogy: father - man + woman = ?
# Top results:
#   - mother          (score: 0.6239)
#   - son             (score: 0.5529)
#   - daughter        (score: 0.5371)
#   - wife            (score: 0.5330)
#   - parents         (score: 0.4974)
# Analogy: husband - man + woman = ?
# Top results:
#   - daughter        (score: 0.6514)
#   - married         (score: 0.6083)
#   - wife            (score: 0.5511)
#   - actress         (score: 0.5465)
#   - she             (score: 0.5313)
#  -Epoch 2/50 | Iter 9000 | Loss: 9.0267
# Epoch 2/50, Loss: 9.0002
#  Validation similarity test:
#   -given     | defined, gives, denoted, hence, derived
#   -place     | advantage, preposition, places, course, location
#   -sometimes | often, commonly, usually, frequently, generally
#   -information| data, access, server, users, database
#   -algebra   | algebraic, mathematics, theorem, mathematical, calculus
#   -magnetic  | electromagnetic, electric, electrical, velocity, tape
#   -selected  | none, collected, serve, pp, enlarged
#   -pop       | rock, album, songs, music, blues
# Analogy: king - man + woman = ?
# Top results:
#   - daughter        (score: 0.5039)
#   - monarch         (score: 0.4375)
#   - queen           (score: 0.4343)
#   - prince          (score: 0.4303)
#   - kings           (score: 0.4291)
# Analogy: father - man + woman = ?
# Top results:
#   - mother          (score: 0.6864)
#   - daughter        (score: 0.5761)
#   - wife            (score: 0.5742)
#   - son             (score: 0.5649)
#   - married         (score: 0.5223)
# Analogy: husband - man + woman = ?
# Top results:
#   - married         (score: 0.6493)
#   - daughter        (score: 0.6153)
#   - children        (score: 0.5752)
#   - female          (score: 0.5719)
#   - mother          (score: 0.5459)
#  -Epoch 3/50 | Iter 9000 | Loss: 8.9494
# Epoch 3/50, Loss: 8.9299
#  Validation similarity test:
#   -information| data, message, database, knowledge, description
#   -important | vital, prominent, crucial, importance, interesting
#   -place     | advantage, seriously, toll, places, finish
#   -sometimes | often, commonly, usually, frequently, occasionally
#   -magnetic  | electromagnetic, electrical, velocity, electric, magnet
#   -algebra   | algebraic, mathematical, mathematics, calculus, theorem
#   -selected  | edited, collected, prose, references, anthology
#   -pop       | rock, music, album, blues, hop
# Analogy: king - man + woman = ?
# Top results:
#   - monarch         (score: 0.5554)
#   - daughter        (score: 0.5490)
#   - queen           (score: 0.5093)
#   - kings           (score: 0.4459)
#   - duke            (score: 0.4258)
# Analogy: father - man + woman = ?
# Top results:
#   - mother          (score: 0.7330)
#   - daughter        (score: 0.6718)
#   - wife            (score: 0.6096)
#   - son             (score: 0.5755)
#   - parents         (score: 0.5358)
# Analogy: husband - man + woman = ?
# Top results:
#   - daughter        (score: 0.7325)
#   - married         (score: 0.5933)
#   - mother          (score: 0.5857)
#   - wife            (score: 0.5814)
#   - her             (score: 0.5792)
#  -Epoch 4/50 | Iter 9000 | Loss: 8.9005
# Epoch 4/50, Loss: 8.8854
#  Validation similarity test:
#   -important | vital, prominent, influential, importance, significant
#   -place     | advantage, finish, palace, bath, happens
#   -information| data, knowledge, message, database, file
#   -given     | gives, defined, denoted, calculated, implies
#   -pop       | rock, album, blues, music, songs
#   -selected  | select, assembled, elected, appendix, miscellaneous
#   -algebra   | algebraic, mathematical, theorem, algebras, mathematics
#   -widespread| occurred, suppression, due, led, looting
# Analogy: king - man + woman = ?
# Top results:
#   - queen           (score: 0.5448)
#   - daughter        (score: 0.5428)
#   - duke            (score: 0.4599)
#   - monarch         (score: 0.4526)
#   - hj              (score: 0.4381)
# Analogy: father - man + woman = ?
# Top results:
#   - mother          (score: 0.6799)
#   - daughter        (score: 0.6586)
#   - wife            (score: 0.6045)
#   - son             (score: 0.5552)
#   - her             (score: 0.5387)
# Analogy: husband - man + woman = ?
# Top results:
#   - daughter        (score: 0.6825)
#   - married         (score: 0.6263)
#   - wife            (score: 0.5818)
#   - her             (score: 0.5726)
#   - householder     (score: 0.5577)
#  -Epoch 5/50 | Iter 9000 | Loss: 8.8696
# Epoch 5/50, Loss: 8.8557
#  Validation similarity test:
#   -place     | palace, advantage, where, places, toll
#   -important | vital, interesting, valuable, prominent, fundamental
#   -information| data, file, knowledge, database, message
#   -sometimes | often, usually, commonly, typically, frequently
#   -selected  | appendix, presidents, concerning, assembled, translated
#   -pop       | rock, blues, album, hop, music
#   -algebra   | algebraic, theorem, mathematical, algebras, mathematics
#   -widespread| popular, growing, increasingly, prevalent, violence
# Analogy: king - man + woman = ?
# Top results:
#   - queen           (score: 0.5825)
#   - monarch         (score: 0.5471)
#   - daughter        (score: 0.4930)
#   - kings           (score: 0.4645)
#   - actress         (score: 0.4492)
# Analogy: father - man + woman = ?
# Top results:
#   - mother          (score: 0.6926)
#   - wife            (score: 0.6394)
#   - daughter        (score: 0.5925)
#   - son             (score: 0.5925)
#   - whom            (score: 0.5145)
# Analogy: husband - man + woman = ?
# Top results:
#   - daughter        (score: 0.6556)
#   - married         (score: 0.6190)
#   - wife            (score: 0.6097)
#   - mother          (score: 0.6082)
#   - she             (score: 0.5885)
#  -Epoch 6/50 | Iter 9000 | Loss: 8.8450
# Epoch 6/50, Loss: 8.8331
#  Validation similarity test:
#   -sometimes | often, commonly, usually, or, frequently
#   -information| data, message, messages, database, users
#   -given     | gives, is, defined, implies, exists
#   -important | crucial, interesting, fundamental, vital, principal
#   -pop       | rock, album, blues, hop, music
#   -algebra   | algebraic, algebras, theorem, mathematics, mathematical
#   -magnetic  | electromagnetic, dipole, electrical, coil, electric
#   -widespread| prevalent, increasingly, occurred, acceptance, popular
# Analogy: king - man + woman = ?
# Top results:
#   - monarch         (score: 0.5741)
#   - queen           (score: 0.5414)
#   - kings           (score: 0.5050)
#   - reign           (score: 0.4830)
#   - daughter        (score: 0.4809)
# Analogy: father - man + woman = ?
# Top results:
#   - mother          (score: 0.6654)
#   - son             (score: 0.6564)
#   - daughter        (score: 0.6298)
#   - wife            (score: 0.6089)
#   - sister          (score: 0.5547)
# Analogy: husband - man + woman = ?
# Top results:
#   - married         (score: 0.6473)
#   - daughter        (score: 0.6265)
#   - she             (score: 0.6069)
#   - wife            (score: 0.6037)
#   - mother          (score: 0.5830)
#  -Epoch 7/50 | Iter 9000 | Loss: 8.8289
# Epoch 7/50, Loss: 8.8175
#  Validation similarity test:
#   -given     | gives, defined, follows, obtain, is
#   -place     | where, advantage, places, brest, bath
#   -important | fundamental, crucial, vital, influential, interesting
#   -information| data, message, file, knowledge, users
#   -magnetic  | dipole, electromagnetic, electric, electrical, energy
#   -pop       | rock, album, hop, music, blues
#   -widespread| ubiquitous, increasingly, wider, considerable, controversy
#   -selected  | annotated, select, biographical, links, appendix
# Analogy: king - man + woman = ?
# Top results:
#   - kings           (score: 0.5331)
#   - monarchs        (score: 0.4898)
#   - queen           (score: 0.4822)
#   - duke            (score: 0.4785)
#   - daughter        (score: 0.4704)
# Analogy: father - man + woman = ?
# Top results:
#   - mother          (score: 0.6444)
#   - wife            (score: 0.6221)
#   - daughter        (score: 0.5774)
#   - son             (score: 0.5559)
#   - sister          (score: 0.5162)
# Analogy: husband - man + woman = ?
# Top results:
#   - daughter        (score: 0.6084)
#   - wife            (score: 0.6025)
#   - mother          (score: 0.5922)
#   - married         (score: 0.5849)
#   - actress         (score: 0.5512)
#  -Epoch 8/50 | Iter 9000 | Loss: 8.8162
# Epoch 8/50, Loss: 8.8055
#  Validation similarity test:
#   -given     | gives, implies, defined, follows, is
#   -place     | places, advantage, pains, where, opening
#   -information| data, knowledge, file, software, message
#   -important | crucial, fundamental, importance, prominent, significant
#   -algebra   | algebraic, algebras, mathematical, theorem, finite
#   -magnetic  | electromagnetic, dipole, electric, electrical, resonance
#   -widespread| prevalent, growing, due, popular, resulted
#   -pop       | music, rock, hop, album, blues
# Analogy: king - man + woman = ?
# Top results:
#   - queen           (score: 0.5420)
#   - daughter        (score: 0.5044)
#   - duke            (score: 0.4858)
#   - elizabeth       (score: 0.4770)
#   - monarch         (score: 0.4743)
# Analogy: father - man + woman = ?
# Top results:
#   - mother          (score: 0.6591)
#   - sister          (score: 0.6193)
#   - daughter        (score: 0.6163)
#   - wife            (score: 0.6097)
#   - son             (score: 0.5772)
# Analogy: husband - man + woman = ?
# Top results:
#   - wife            (score: 0.6377)
#   - married         (score: 0.6179)
#   - daughter        (score: 0.6102)
#   - marriage        (score: 0.6036)
#   - sister          (score: 0.5802)
#  -Epoch 9/50 | Iter 9000 | Loss: 8.8072
# Epoch 9/50, Loss: 8.7966
#  Validation similarity test:
#   -sometimes | often, usually, commonly, typically, generally
#   -place     | places, advantage, thursday, round, final
#   -given     | gives, follows, is, defined, any
#   -important | crucial, vital, fundamental, prominent, importance
#   -algebra   | mathematical, algebraic, algebras, differential, finite
#   -magnetic  | electromagnetic, dipole, electric, electrical, resonance
#   -pop       | hop, music, rock, band, jazz
#   -widespread| popular, popularity, due, prevalent, resulted
# Analogy: king - man + woman = ?
# Top results:
#   - queen           (score: 0.5453)
#   - daughter        (score: 0.5106)
#   - monarch         (score: 0.5043)
#   - kings           (score: 0.4998)
#   - monarchs        (score: 0.4834)
# Analogy: father - man + woman = ?
# Top results:
#   - mother          (score: 0.6280)
#   - wife            (score: 0.6160)
#   - son             (score: 0.5964)
#   - daughter        (score: 0.5861)
#   - sister          (score: 0.5399)
# Analogy: husband - man + woman = ?
# Top results:
#   - wife            (score: 0.6259)
#   - married         (score: 0.5991)
#   - daughter        (score: 0.5988)
#   - her             (score: 0.5633)
#   - marriage        (score: 0.5404)
#  -Epoch 10/50 | Iter 9000 | Loss: 8.7980
#%%
# load checkpoint
checkpoint = torch.load("./weights/skipgram_model_e9_1e-4.pth")
embedding_size = checkpoint['embedding_size']
vocab_size = checkpoint['vocab_size']
epoch = checkpoint['epochs']
loss = checkpoint['loss']
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = SkipGram(vocab_size, embedding_size)
model.to(device)
model.load_state_dict(checkpoint['state_dict'])

print(f'skipgram model loaded!')
print(f"{epoch=}")
print(f"{loss=}")
print(f"{embedding_size=}")
print(f"{vocab_size=:,}")

#%%
# lets run some checks on our model and see how well it works
# lets grab a few words and see if the model can correctly identify 
# any underlying relationship between them 
def check_semantic_analogy(test_words, embedding_layer, word2int,device):
    test_indices = [word2int[word] for word in test_words if word in word2int]
    # get their embeddings
    embeddings = embedding_layer(torch.tensor(test_indices).to(device))
    # to get cosine similarity we need to unit normalize the embeddings
    embeddings_norm = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    # since the vectors are normalized this dot product is exactly equal to
    # the cosine similarity
    similarities = torch.mm(embeddings_norm, embeddings_norm.t()).cpu().detach().numpy()
    
    # display similarity matrix
    df = pd.DataFrame(similarities, index=test_words, columns=test_words)
    print(df)
    print()

def check_nearest_neighbors(word, embedding_layer, word2int, int2word, topk=5):
    # check if its already in vocab
    if word.lower() not in word2int:
        print(f"Word '{word}' does NOT exist in vocabulary.")
        return
    # grab the index and embedding
    idx = word2int[word.lower()]
    embedding = embedding_layer(torch.tensor([idx]).to(device))
    # compute cosine similarity with all embeddings
    all_embeddings = embedding_layer.weight
    # print(f'{all_embeddings.shape=}')
    # print(f'{embedding.shape=}')
    # since we are dealing with one word, we can use pytorch's consine_similarity
    similarity = F.cosine_similarity(embedding, all_embeddings)
    # print(f'{similarity.shape=}')
    # get top-k similar words 
    # get k+1 to include the word itself
    closest_indices = similarity.topk(topk + 1).indices.cpu().numpy()
    closest_words = [int2word[i] for i in closest_indices if i != idx]
    
    print(f"Nearest neighbors for '{word}': {', '.join(closest_words)}")

# now lets also visualize the embeddings space
def visualize_embedding_space(embedding_layer, int2word, words_to_visualize=200, use_pca=False, figsize=(24,16)):
    # choose between pca or tsne for diminsionality reduction
    reducer = PCA(n_components=2) if use_pca else TSNE()
    embeddings = embedding_layer.weight.detach().cpu().numpy()
    embd_reduced = reducer.fit_transform(embeddings[:words_to_visualize, :])

    plt.figure(figsize=(24,16))
    plt.title(f'Visualizing Embedding Space Using {"PCA" if use_pca else "TSNE"}')
    for idx in range(words_to_visualize):
        plt.scatter(*embd_reduced[idx, :], color='steelblue')
        plt.annotate(int2word[idx], (embd_reduced[idx, 0], embd_reduced[idx, 1]), alpha=0.7)

#%%
evalualte_model_quality(model.embedding_layer,
                        int2word,
                        window_size=5,
                        validation_size=8,
                        common_start_index=256,
                        uncommon_start_index=2000)

# lets see if the model can correctly identify their relationships
test_words = ['king', 'queen', 'man', 'woman', 'prince', 'princess']
check_semantic_analogy(test_words, model.embedding_layer, word2int, device)
test_words2 = ['father', 'mother', 'man','woman','son','daughter']
check_semantic_analogy(test_words2, model.embedding_layer, word2int, device)

# should give us son and price
check_analogy_test('daughter', 'woman', 'man', model.embedding_layer, word2int, int2word)
check_analogy_test('princess', 'woman', 'man', model.embedding_layer, word2int, int2word)

# find nearest neighbors to each word
check_nearest_neighbors('king', model.embedding_layer, word2int, int2word, topk=5)
check_nearest_neighbors('Iran', model.embedding_layer, word2int, int2word, topk=5)
#%%
visualize_embedding_space(model.embedding_layer, int2word, words_to_visualize=200)
visualize_embedding_space(model.embedding_layer, int2word, words_to_visualize=200, use_pca=True)

#%%
# ok, now lets create word emebedding using skipgram with negative sampling 
# why? becasue negative sampling significantly reduces the computation of 
# softmax across the full vocabulary! which is instead of predicting all
# possible words, predict only positive and sampled negative words.
# basically what we do here is that we have two embeddings, we feed a word into
# first emebdding get couple of target words that are similar and then feed one of 
# those words into the second embedding and we should get the first word that was initially
# enetered. in doing so, we also feed some noise words, that we use to achieve our loss
# becasue simply trying every single words would impose huge burden. lets see how it is done

class SkipGramWithNegativeSampling(nn.Module):
    def __init__(self, vocab_size, embedding_size, noise_dist=None):
        super().__init__()
        self.input_embedding = nn.Embedding(vocab_size, embedding_size)
        self.output_embedding = nn.Embedding(vocab_size, embedding_size)
        self.noise_distribution = noise_dist

    def forward(self, input_words, target_words, n_samples):
        # Get embeddings for input and target words
        # TODO use positive words instead of targetwords 
        # TODO and use negative words instead of noise_words
        input_embeds = self.input_embedding(input_words)  # (batch_size, embedding_dim)
        target_embeds = self.output_embedding(target_words)  # (batch_size, embedding_dim)
        
        # Generate noise embeddings
        noise_words = torch.multinomial(self.noise_distribution, 
                                        input_words.size(0) * n_samples, 
                                        replacement=True).view(input_words.size(0),
                                                               n_samples)  # (batch_size, n_samples)
        noise_embeds = self.output_embedding(noise_words)  # (batch_size, n_samples, embedding_dim)
        
        return input_embeds, target_embeds, noise_embeds


# now ok. now lets create a loss function for ourselves 
class SkipGramNegativeSamplingLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_embeddings, output_embeddings, noise_embedidngs):

        # here we will have two losses. the first one shows how much the input
        # and output/target/context emebeddings are like each other 
        # and the second loss is responsbile for creating very different embeddings
        # that is negative sampling part. 
        # 
        # basically we have a positive/target embedding, and negative/noise/random embedding
        # we want the input embedding and our target/positive embedding to be as close as possible
        # while the input embedding and negative/noise/or random embeddings are as far away as pssible
        # 
        # in order to see how two embeddings are like each other, we simply multiply them
        # (1xembeddings) and (1xembedding) , so we should have a 1x1 result.
        
        batch_size = input_embeddings.size(0)
        embedding_size = input_embeddings.size(1)
        # reshape them so we can multiply them (or we can use unsqueeze()!)
        input_embeddings = input_embeddings.view(batch_size, embedding_size, 1)
        output_embeddings = output_embeddings.view(batch_size, 1, embedding_size)
        # reminder: log(1) = 0, log(0)=undefined! 
        # since batches are involved we simply use the bmm (bacth-matrix-multiply)
        # and because we want probablities, so we use sigmoid.
        # and for numerical stability we use log!
        #  
        # basically sigmoid maps our dot product (similarity) to a probability between 0 and 1
        # and log converts this probability into a log-probability, which makes it easier to
        # sum probabilities (log-sum trick) during optimization.
        # this overall allows us not only to have numerical stability by using the log-probability
        # (we avoid potential issues with small probabilities (as log values scale better))
        # but also more efficient training because the log-probability formulation simplifies 
        # gradient calculations and thus makes training more stable and efficient.
        # and finally maximizing the log probability corresponds to minimizing the negative 
        # log-likelihood, which is a common approach in probabilistic models.
        # sidenote: we could have used F.logsigmoid() fused operator as well!
        # see the simplified version below(after this implementation)
        # in order not to face issues, we must make sure log doesnt recieve 0!
        # otherwise we would get nans! the easy way is to use logsigmoid() but without it
        # we can simply clamp the 0 to a very small number epsillon and this way avoid the issue
        # epsilon =1e-8 seems ok, but the larger the epsilon gets the larger the bias becomes
        # we want to have the least amount of bias imposed here, (i.e. the closest to 0 we can get)
        # to prevent no clamping as much as possible and only clamp when otherwise it would
        # lead to nans (output of sigmoid becomes 0 and thus log(0) becons -inf and resulting in nans
        # in our loss!
        epsillon = 1e-10
        loss1 = torch.bmm(output_embeddings, input_embeddings).sigmoid().clamp(min=epsillon).log().squeeze()
        # print(f'{loss1.shape=}') #torch.Size([3030])
        # now for our noise/random/megative samples we simply do the same thing
        # but since the random samples and input embeddings should not be similar
        # we use a -1 sign in the operation to signal they must to be similar (a large positive number)
        # and finally we sum all the results to have a single number for loss
        # 
        # update:
        # I initially left out the squeeze() at the end, and it caused our loss2 to have an extra dim
        # (e.g. (3030,1))! this simple mistake, sent everything into oblivion! 
        # because when our loss1 is added to loss2, their shape is not the same 
        # so pytorch goes for a broadcast and therefore it reshapes loss1 to (1,3030)
        # while loss2 is (3030,1). so when it tries to add the two, we endup with a 
        # (3030,3030) tensor! basically it adds every entry from loss1 to every entery
        # from loss2, whereas, it should have been adding each entry with its respective 
        # entry in the other tensor. that is, idx1 from loss1 must be added to idx1 in loss2.
        # whereas here, idx1 is added to all entries in loss2, and each entry repeats 
        # this so instead of 3030 values for example, we end up with 3030x3030 values 
        # which leads to nonsensical value for loss(adding values like this is meaningless
        # (imagine trying to average ones score and instead of adding your best and worst scores,
        # you go and add your best score to everyones worst scores, and they do this with 
        # everyone elss (their best score is added to everyone elses worst score!)))
        loss2 = torch.bmm(noise_embedidngs.neg(), input_embeddings).sigmoid().clamp(min=epsillon).log().sum(dim=1).squeeze()
        # print(f'{loss2.shape=}') #torch.Size([3030])
        # and we add them both and try to minize the whole loss
        return -torch.mean(loss1+loss2)

# we could simplify our loss further like this
# this is a numerically stable version because of logsigmoid!
# and a bit consiser and clearer!
# class SkipGramNegativeSamplingLoss(nn.Module):
#     def __init__(self):
#         super().__init__()
#     def forward(self, input_embeds, target_embeds, noise_embeds):
#         pos_score = torch.bmm(target_embeds.unsqueeze(1), input_embeds.unsqueeze(2)).squeeze()  # (batch_size)
#         pos_loss = F.logsigmoid(pos_score)
#         neg_score = torch.bmm(noise_embeds.neg(), input_embeds.unsqueeze(2)).squeeze()  # (batch_size, n_samples)
#         neg_loss = F.logsigmoid(neg_score).sum(dim=1)  
#         return -(pos_loss + neg_loss).mean()

def create_noise_distribution(word_freqs, power=0.75):
    # The negative sampling requires sampling "noise words" (i.e. random words from
    # the vocabulary) to contrast with positive examples. 
    # so we need a noise distribution to sample from. 
    # In order to have a noise distribution, we need probablities! 
    # how do we create one? 
    # we can make a frequency distribution of words in the corpus and then 
    # normalized it to represent probabilities (i.e. a unigram distribution)
    # 
    # we can calculate it as:
    # unigram_dist[i] = count(w_i)/total_word_count
    # where count(w_i) is the number of occurrences of word w_i
    # in the corpus, and total_word_count is the total number of words.
    # we already have word_freqs dictionary, so we can easily do : 
    # word_freqs = [cnt for k,cnt in word_freqs.items()]
    # unigram_dist = word_freqs / sum(word_freqs)
    #
    # sidenote: refresher 
    # A unigram distribution is a probability distribution over individual words (or tokens)
    # in a corpus. It represents the relative frequency of each word, essentially giving the
    # likelihood of randomly picking a specific word from the corpus.
    # 
    # The prefix Uni in Uni-gram means one, likewise bi means two, and tri means three)
    # therefore, "Unigram" simply refers to single words or tokens, the same goes to
    # bigram, that is a "bigram" refers to pairs of consecutive words, 
    # and "trigram" refers to three-word sequences.
    #
    # The unigram distribution is a frequency-based distribution where each word's 
    # probability is proportional to how often it appears in the corpus.
    # 
    # How do we compute it then?
    # simpe! to calculate the unigram distribution, we simply count the occurrences 
    # of each word in the corpus and then normalize these counts by dividing by 
    # the total number of words in the corpus.
    # 
    # for a word w_i, the unigram probability P(w_i) is:
    # P(w_i) = count(w_i) / total_number_of_words_in_corpus
    # where count(w_i) is the number of times word w_i appears in the corpus.
    # total number of words in corpus is the sum of all word frequencies.
    #
    # but this wouldnt be enough, as not all the words are repeated equally
    # in our corpus. therefore to account for that we incorporate power 
    # in our formula like this:
    # 
    # final_distribution = (unigram_dist**power) / np.sum(unigram_dist**power) 
    # 
    # now the power in our new formula (unigram_dist ** power) raises the probabilities to a
    # certain power(more explanation in a moment), which adjusts the distribution to 
    # favor less frequent words while at the same time keeps frequent words relatively likely.
    # speaking of the power used, different powers have different implications and effects,
    # for example, 
    # If power=1, the distribution remains proportional to the unigram distribution
    # If power<1, it smooths the distribution, reducing the dominance of highly frequent words
    # If power>1, it amplifies the dominance of frequent words
    # 
    # now with these changes, our noise distribution can make sure frequent words 
    # are more likely to be selected as noise samples and the distribution can 
    # also be configured using the power to balance between very frequent and less 
    # frequent words.
    # 
    # sidenote: Why do we use power=0.75?
    # because its a value that has been widely used its shown empiracally to provide 
    # a good trade-off between frequent and rare words which improves the quality 
    # of our word embeddings. frequent words still appear often as negative samples
    # but their dominance is reduced compared to their true frequency in the corpus.
    # 
    # finally by doing np.sum(unigram_dist ** power), our adjusted probabilities are
    # summed to normalize the distribution. this is to make sure the sum of distribution
    # equals 1, making it a valid probability distribution.
    # this normalized_dist is then used to sample negative words during training.
    word_freqs = [cnt for _,cnt in word_freqs.items()]
    unigram_dist = torch.tensor(word_freqs / np.sum(word_freqs))
    noise_distribution = unigram_dist ** power/torch.sum(unigram_dist**power)
    return noise_distribution

#%%
# before we continue with training lets first see
# what words pop up at which indexes, this gives us
# a better idea what word to choose
# for our similarity validation process
# too small numbers are usually reserved for the,of,etc
# which are not what we want, remember we have 63K words
# so choose freely, we are not bound to 0, 100, 200 or even 2000!
# play with the numbers and better see the similar words
# 
start=200
window_size=5
print(f'words starting at {start}:')
for i in range(start,start+window_size):
    print(f'{i}: {int2word[i]}')
#%%
seed = 10
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# now lets start the actual training!
device = 'cuda' if torch.cuda.is_available() else 'cpu'

embedding_size = 300
vocab_size = len(word2int)
# initialize noise distribution
noise_dist = create_noise_distribution(word_freqs)
noise_dist = noise_dist.to(device)

model = SkipGramWithNegativeSampling(vocab_size, embedding_size, noise_dist)
model.to(device)
# note we are using log_softmax, so we must use nllloss here,
criterion = SkipGramNegativeSamplingLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

num_epochs = 20
batch_size = 512
window_size = 10#5
validation_size = 8
interval = 9000

print(f'training skipgram with negative sampling...')

for epoch in range(num_epochs):
    losses = []
    for i, (X, Y) in enumerate(get_batch(word_list_digitized, 
                                         batch_size=batch_size, 
                                         window_size=window_size), start=1):
        X = torch.LongTensor(X).to(device)
        Y = torch.LongTensor(Y).to(device)

        input_embeds, target_embeds, noise_embeds = model(X, Y, n_samples=5)
        loss = criterion(input_embeds, target_embeds, noise_embeds)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        
        losses.append(loss.item())
        
        if i % interval == 0:
            evalualte_model_quality(model.input_embedding,
                                    int2word,
                                    validation_size=8,
                                    window_size=window_size, 
                                    common_start_index=256, 
                                    uncommon_start_index=2000)

            print(f' -Epoch {epoch}/{num_epochs} | Iter {i} | Loss: {np.mean(losses):.4f}')
            
    print(f"Epoch {epoch}/{num_epochs}, Loss: {np.mean(losses):.4f}")
    # Save model
    torch.save({"state_dict":model.state_dict(),
                "epochs":epoch,
                "loss":np.mean(losses),
                "vocab_size":vocab_size,
                "embedding_size":embedding_size,
                "word2int": word2int,
                "int2word": int2word}, "./weights/skipgram_negativesampling_model.pth")
# 
# Epoch 15/50, Loss: 1.8525
# president : presidential, elected, executive, minister, elections
# political : politics, parties, social, party, government
# party     : democratic, coalition, election, vote, elections
# order     : orders, given, position, that, needed
# ahman     : emmitt, moriarty, mulligan, spinrad, geary
# theophylline: allosteric, metabolised, kaposi, allergen, benzoic
# esta      : greencine, cet, roca, mgm, coro
# shortlist : wrongdoing, incapacitated, sihanouk, steadfastly, bosnians
#  -Epoch 16/50 | Iter 7 | Loss: 1.8418
# Epoch 16/50, Loss: 1.8422
# order     : orders, given, knights, any, their
# party     : elections, democratic, coalition, opposition, parties
# president : elected, presidential, cabinet, executive, vice
# political : politics, democracy, parties, social, government
# theophylline: allosteric, allergen, theobromine, kaposi, guillain
# ahman     : spinrad, moriarty, wyche, emmitt, rookie
# shortlist : eurofighter, bosnians, nuseibeh, sihanouk, underscores
# whitgift  : arrigo, berengar, excommunicates, tunis, donati
#  -Epoch 17/50 | Iter 7 | Loss: 1.8327
# Epoch 17/50, Loss: 1.8331
#
# second test (using lr=0.001)
# training skipgram with negative sampling...
#  Validation similarity test:
#   -back      | on, in, towards, and, it
#   -central   | pakistan, border, s, comets, was
#   -age       | fantastical, earliest, keypad, organizers, population
#   -make      | immediately, complete, to, that, sept
#   -festival  | questions, supposed, joliet, frequent, slavs
#   -algebra   | whether, center, raiders, to, advocacy
#   -blues     | eruptions, initiated, pierre, spliced, communes
#   -selected  | was, museums, alive, naval, zero
# Analogy: king - man + woman = ?
# Top results:
#   - permitted       (score: 0.4091)
#   - indicator       (score: 0.4091)
#   - portrayed       (score: 0.4027)
#   - lle             (score: 0.3952)
#   - wheels          (score: 0.3939)
# Analogy: father - man + woman = ?
# Top results:
#   - aarhus          (score: 0.4941)
#   - dates           (score: 0.4401)
#   - whig            (score: 0.4378)
#   - spiritual       (score: 0.4204)
#   - islam           (score: 0.4092)
# Analogy: husband - man + woman = ?
# Top results:
#   - resolved        (score: 0.4447)
#   - compiler        (score: 0.4330)
#   - casta           (score: 0.3935)
#   - nihilist        (score: 0.3869)
#   - portrayed       (score: 0.3840)
#  -Epoch 0/20 | Iter 9000 | Loss: 20.2548
# Epoch 0/20, Loss: 14.7005
#  Validation similarity test:
#   -control   | its, for, an, to, the
#   -age       | he, children, years, s, his
#   -others    | of, to, as, in, that
#   -central   | east, north, largest, europe, south
#   -speakers  | paved, tournaments, magistracies, languages, secede
#   -pop       | hands, volunteer, originally, microphones, stories
#   -festival  | charles, andy, criticism, relies, repel
#   -alliance  | russia, policy, primarily, armed, reign
# Analogy: king - man + woman = ?
# Top results:
#   - one             (score: 0.6986)
#   - england         (score: 0.6928)
#   - five            (score: 0.6845)
#   - six             (score: 0.6840)
#   - nine            (score: 0.6822)
# Analogy: father - man + woman = ?
# Top results:
#   - she             (score: 0.6926)
#   - her             (score: 0.6766)
#   - one             (score: 0.6683)
#   - died            (score: 0.6641)
#   - england         (score: 0.6620)
# Analogy: husband - man + woman = ?
# Top results:
#   - newspaper       (score: 0.3659)
#   - retaliated      (score: 0.3622)
#   - townes          (score: 0.3280)
#   - arroyo          (score: 0.3202)
#   - outstretched    (score: 0.3171)
#  -Epoch 1/20 | Iter 9000 | Loss: 4.9935
# Epoch 1/20, Loss: 4.4391
#  Validation similarity test:
#   -others    | as, among, become, their, of
#   -case      | is, any, be, given, example
#   -st        | saint, seven, john, william, th
#   -man       | his, story, love, he, him
#   -magnetic  | electric, speed, field, objects, particle
#   -selected  | magazine, student, museum, york, news
#   -ft        | km, area, m, total, miles
#   -alliance  | democratic, leaders, party, government, elections
# Analogy: king - man + woman = ?
# Top results:
#   - son             (score: 0.9126)
#   - prince          (score: 0.8932)
#   - died            (score: 0.8914)
#   - queen           (score: 0.8754)
#   - born            (score: 0.8727)
# Analogy: father - man + woman = ?
# Top results:
#   - son             (score: 0.9187)
#   - wife            (score: 0.9122)
#   - married         (score: 0.9023)
#   - daughter        (score: 0.8975)
#   - mother          (score: 0.8811)
# Analogy: husband - man + woman = ?
# Top results:
#   - wife            (score: 0.6733)
#   - married         (score: 0.6701)
#   - she             (score: 0.6457)
#   - her             (score: 0.6281)
#   - daughter        (score: 0.6259)
#  -Epoch 2/20 | Iter 9000 | Loss: 3.1395
# Epoch 2/20, Loss: 2.9816
#  Validation similarity test:
#   -age       | children, years, male, birth, living
#   -st        | saint, william, seven, founded, th
#   -back      | up, off, away, down, behind
#   -central   | east, north, south, northern, largest
#   -widespread| largely, particularly, increasingly, been, especially
#   -ft        | km, miles, m, area, feet
#   -speakers  | spoken, languages, speaking, language, dialects
#   -selected  | s, review, published, student, magazine
# Analogy: king - man + woman = ?
# Top results:
#   - prince          (score: 0.9494)
#   - daughter        (score: 0.9295)
#   - married         (score: 0.9202)
#   - son             (score: 0.9146)
#   - brother         (score: 0.8934)
# Analogy: father - man + woman = ?
# Top results:
#   - married         (score: 0.9863)
#   - daughter        (score: 0.9713)
#   - mother          (score: 0.9460)
#   - son             (score: 0.9433)
#   - wife            (score: 0.9183)
# Analogy: husband - man + woman = ?
# Top results:
#   - married         (score: 0.9475)
#   - daughter        (score: 0.8827)
#   - mother          (score: 0.8464)
#   - wife            (score: 0.8405)
#   - her             (score: 0.8367)
#  -Epoch 3/20 | Iter 9000 | Loss: 2.5561
# Epoch 3/20, Loss: 2.4889
#  Validation similarity test:
#   -age       | years, children, birth, female, male
#   -others    | as, some, considered, have, what
#   -st        | saint, nd, th, founded, rd
#   -case      | any, if, be, instance, is
#   -selected  | review, press, published, introduction, publishing
#   -speakers  | spoken, languages, dialects, speaking, dialect
#   -algebra   | infinite, mathematics, notation, theorem, finite
#   -magnetic  | electric, particles, particle, speed, mechanical
# Analogy: king - man + woman = ?
# Top results:
#   - son             (score: 0.9049)
#   - prince          (score: 0.8950)
#   - daughter        (score: 0.8637)
#   - married         (score: 0.8578)
#   - throne          (score: 0.8527)
# Analogy: father - man + woman = ?
# Top results:
#   - married         (score: 0.9658)
#   - mother          (score: 0.9396)
#   - daughter        (score: 0.9325)
#   - son             (score: 0.9242)
#   - wife            (score: 0.9147)
# Analogy: husband - man + woman = ?
# Top results:
#   - married         (score: 0.9533)
#   - marriage        (score: 0.8764)
#   - mother          (score: 0.8711)
#   - child           (score: 0.8590)
#   - wife            (score: 0.8553)
#  -Epoch 4/20 | Iter 9000 | Loss: 2.2938
# Epoch 4/20, Loss: 2.2602
#  Validation similarity test:
#   -man       | love, my, story, she, girl
#   -st        | saint, nd, rd, louis, william
#   -make      | them, they, makes, without, give
#   -countries | nations, country, european, foreign, europe
#   -speakers  | spoken, dialects, languages, speaking, dialect
#   -ft        | km, miles, meters, feet, metres
#   -pop       | albums, songs, band, music, album
#   -widespread| increasingly, largely, especially, particularly, recent
# Analogy: king - man + woman = ?
# Top results:
#   - prince          (score: 0.9154)
#   - son             (score: 0.8859)
#   - married         (score: 0.8568)
#   - daughter        (score: 0.8508)
#   - throne          (score: 0.8425)
# Analogy: father - man + woman = ?
# Top results:
#   - married         (score: 0.9918)
#   - mother          (score: 0.9585)
#   - daughter        (score: 0.9476)
#   - wife            (score: 0.9347)
#   - marriage        (score: 0.9337)
# Analogy: husband - man + woman = ?
# Top results:
#   - married         (score: 0.9997)
#   - marriage        (score: 0.9300)
#   - child           (score: 0.9113)
#   - mother          (score: 0.8870)
#   - wife            (score: 0.8850)
#  -Epoch 5/20 | Iter 9000 | Loss: 2.1568
# Epoch 5/20, Loss: 2.1380
#  Validation similarity test:
#   -countries | nations, country, european, europe, foreign
#   -central   | east, south, west, north, region
#   -man       | love, portrayed, story, girl, said
#   -case      | any, may, instance, is, be
#   -festival  | festivals, celebrated, holiday, theatre, christmas
#   -blues     | jazz, band, pop, song, musicians
#   -ft        | meters, metres, km, miles, feet
#   -alliance  | party, leadership, democratic, communist, coalition
# Analogy: king - man + woman = ?
# Top results:
#   - prince          (score: 0.9312)
#   - married         (score: 0.8955)
#   - son             (score: 0.8925)
#   - daughter        (score: 0.8880)
#   - throne          (score: 0.8736)
# Analogy: father - man + woman = ?
# Top results:
#   - married         (score: 1.0549)
#   - mother          (score: 1.0211)
#   - daughter        (score: 1.0013)
#   - wife            (score: 0.9801)
#   - husband         (score: 0.9795)
# Analogy: husband - man + woman = ?
# Top results:
#   - married         (score: 1.0919)
#   - child           (score: 0.9874)
#   - marriage        (score: 0.9846)
#   - daughter        (score: 0.9668)
#   - mother          (score: 0.9628)
#  -Epoch 6/20 | Iter 9000 | Loss: 2.0772
# Epoch 6/20, Loss: 2.0660
#  Validation similarity test:
#   -age       | years, birth, children, female, older
#   -central   | east, west, south, north, region
#   -others    | some, considered, themselves, reasons, among
#   -man       | girl, love, woman, my, evil
#   -algebra   | theorem, finite, mathematics, mathematical, algebraic
#   -alliance  | party, leadership, communist, leaders, leader
#   -festival  | festivals, celebrated, theatre, holiday, summer
#   -pop       | albums, music, band, songs, album
# Analogy: king - man + woman = ?
# Top results:
#   - prince          (score: 0.8658)
#   - son             (score: 0.8433)
#   - daughter        (score: 0.8364)
#   - throne          (score: 0.8360)
#   - queen           (score: 0.8208)
# Analogy: father - man + woman = ?
# Top results:
#   - mother          (score: 1.0211)
#   - married         (score: 1.0057)
#   - daughter        (score: 0.9678)
#   - wife            (score: 0.9631)
#   - husband         (score: 0.9586)
# Analogy: husband - man + woman = ?
# Top results:
#   - married         (score: 1.0396)
#   - marriage        (score: 0.9965)
#   - mother          (score: 0.9694)
#   - daughter        (score: 0.9330)
#   - wife            (score: 0.9234)
#  -Epoch 7/20 | Iter 9000 | Loss: 2.0286
# Epoch 7/20, Loss: 2.0212
#  Validation similarity test:
#   -others    | some, among, reasons, considered, noted
#   -age       | years, birth, children, female, older
#   -back      | away, off, down, behind, up
#   -make      | so, them, they, making, give
#   -blues     | jazz, band, songs, guitar, pop
#   -selected  | review, pdf, publishing, edition, publication
#   -algebra   | finite, theorem, algebraic, mathematics, mathematical
#   -alliance  | party, leadership, leader, coalition, leaders
# Analogy: king - man + woman = ?
# Top results:
#   - son             (score: 0.8640)
#   - daughter        (score: 0.8534)
#   - married         (score: 0.8516)
#   - prince          (score: 0.8502)
#   - brother         (score: 0.8247)
# Analogy: father - man + woman = ?
# Top results:
#   - married         (score: 1.0658)
#   - mother          (score: 1.0378)
#   - wife            (score: 1.0038)
#   - daughter        (score: 0.9956)
#   - marriage        (score: 0.9503)
# Analogy: husband - man + woman = ?
# Top results:
#   - married         (score: 1.1016)
#   - marriage        (score: 1.0158)
#   - mother          (score: 0.9671)
#   - wife            (score: 0.9614)
#   - she             (score: 0.9433)
#  -Epoch 8/20 | Iter 9000 | Loss: 1.9960
# Epoch 8/20, Loss: 1.9912
#  Validation similarity test:
#   -case      | any, cases, example, may, instance
#   -age       | years, children, birth, female, older
#   -make      | them, makes, they, so, making
#   -man       | woman, girl, love, evil, story
#   -ft        | meters, metres, feet, miles, height
#   -alliance  | party, coalition, leadership, leader, democratic
#   -widespread| recent, widely, particularly, increasingly, largely
#   -algebra   | finite, theorem, algebraic, mathematics, mathematical
# Analogy: king - man + woman = ?
# Top results:
#   - daughter        (score: 0.8335)
#   - married         (score: 0.8202)
#   - son             (score: 0.8113)
#   - prince          (score: 0.8043)
#   - throne          (score: 0.7995)
# Analogy: father - man + woman = ?
# Top results:
#   - married         (score: 1.0564)
#   - mother          (score: 1.0443)
#   - wife            (score: 0.9912)
#   - daughter        (score: 0.9898)
#   - husband         (score: 0.9866)
# Analogy: husband - man + woman = ?
# Top results:
#   - married         (score: 1.1079)
#   - marriage        (score: 1.0030)
#   - mother          (score: 0.9727)
#   - wife            (score: 0.9517)
#   - daughter        (score: 0.9502)
#  -Epoch 9/20 | Iter 9000 | Loss: 1.9736
# Epoch 9/20, Loss: 1.9697
#  Validation similarity test:
#   -others    | some, many, among, noted, have
#   -back      | behind, off, down, away, up
#   -age       | years, female, birth, children, older
#   -st        | saint, nd, th, thomas, rd
#   -festival  | festivals, celebrated, celebration, holiday, theatre
#   -algebra   | algebraic, finite, theorem, mathematics, mathematical
#   -selected  | review, presented, published, pdf, general
#   -blues     | jazz, guitar, musicians, band, songs
# Analogy: king - man + woman = ?
# Top results:
#   - daughter        (score: 0.8117)
#   - married         (score: 0.8065)
#   - son             (score: 0.7994)
#   - queen           (score: 0.7963)
#   - prince          (score: 0.7758)
# Analogy: father - man + woman = ?
# Top results:
#   - married         (score: 1.0498)
#   - mother          (score: 1.0377)
#   - wife            (score: 1.0014)
#   - daughter        (score: 0.9919)
#   - husband         (score: 0.9782)
# Analogy: husband - man + woman = ?
# Top results:
#   - married         (score: 1.1202)
#   - marriage        (score: 1.0134)
#   - wife            (score: 0.9844)
#   - mother          (score: 0.9681)
#   - daughter        (score: 0.9670)
#  -Epoch 10/20 | Iter 9000 | Loss: 1.9573
# Epoch 10/20, Loss: 1.9544
#  Validation similarity test:
#   -control   | controlled, effectively, controls, controlling, allowing
#   -central   | east, west, south, region, north
#   -countries | nations, europe, eu, country, european
#   -back      | off, behind, away, down, when
#   -algebra   | algebraic, theorem, finite, mathematics, mathematical
#   -alliance  | party, coalition, leadership, leader, democratic
#   -speakers  | spoken, dialects, languages, speaking, dialect
#   -widespread| recent, widely, especially, particularly, largely
# Analogy: king - man + woman = ?
# Top results:
#   - married         (score: 0.8087)
#   - daughter        (score: 0.8037)
#   - son             (score: 0.7808)
#   - wife            (score: 0.7737)
#   - throne          (score: 0.7612)
# Analogy: father - man + woman = ?
# Top results:
#   - married         (score: 1.0675)
#   - mother          (score: 1.0370)
#   - wife            (score: 1.0188)
#   - daughter        (score: 0.9899)
#   - husband         (score: 0.9684)
# Analogy: husband - man + woman = ?
# Top results:
#   - married         (score: 1.1246)
#   - marriage        (score: 1.0401)
#   - wife            (score: 1.0041)
#   - her             (score: 0.9712)
#   - daughter        (score: 0.9639)
#  -Epoch 11/20 | Iter 9000 | Loss: 1.9444
# Epoch 11/20, Loss: 1.9419
#  Validation similarity test:
#   -age       | years, older, ages, males, female
#   -case      | cases, any, example, instance, if
#   -st        | saint, nd, th, rd, thomas
#   -countries | nations, europe, eu, country, belgium
#   -blues     | jazz, songs, musicians, band, guitar
#   -alliance  | coalition, party, leadership, leader, allies
#   -festival  | festivals, celebrated, celebration, holiday, celebrations
#   -pop       | album, band, rock, songs, albums
# Analogy: king - man + woman = ?
# Top results:
#   - son             (score: 0.7758)
#   - daughter        (score: 0.7743)
#   - married         (score: 0.7513)
#   - wife            (score: 0.7396)
#   - throne          (score: 0.7289)
# Analogy: father - man + woman = ?
# Top results:
#   - mother          (score: 1.0699)
#   - married         (score: 1.0312)
#   - wife            (score: 0.9905)
#   - daughter        (score: 0.9802)
#   - husband         (score: 0.9426)
# Analogy: husband - man + woman = ?
# Top results:
#   - married         (score: 1.1109)
#   - marriage        (score: 1.0095)
#   - wife            (score: 0.9848)
#   - her             (score: 0.9646)
#   - she             (score: 0.9626)
#  -Epoch 12/20 | Iter 9000 | Loss: 1.9329
# Epoch 12/20, Loss: 1.9313
#  Validation similarity test:
#   -make      | give, making, they, them, enough
#   -countries | nations, europe, eu, country, belgium
#   -case      | cases, instance, example, may, if
#   -man       | evil, woman, girl, my, life
#   -blues     | jazz, musicians, songs, guitar, band
#   -festival  | festivals, celebrated, celebration, holiday, celebrations
#   -algebra   | algebraic, theorem, finite, mathematics, mathematical
#   -magnetic  | electromagnetic, electrical, electric, electrons, optical
# Analogy: king - man + woman = ?
# Top results:
#   - daughter        (score: 0.7958)
#   - son             (score: 0.7888)
#   - married         (score: 0.7618)
#   - wife            (score: 0.7524)
#   - throne          (score: 0.7441)
# Analogy: father - man + woman = ?
# Top results:
#   - mother          (score: 1.0559)
#   - married         (score: 1.0444)
#   - daughter        (score: 1.0024)
#   - wife            (score: 1.0012)
#   - son             (score: 0.9319)
# Analogy: husband - man + woman = ?
# Top results:
#   - married         (score: 1.1266)
#   - wife            (score: 1.0101)
#   - marriage        (score: 1.0043)
#   - her             (score: 0.9882)
#   - daughter        (score: 0.9787)
#  -Epoch 13/20 | Iter 9000 | Loss: 1.9246
# Epoch 13/20, Loss: 1.9225
#  Validation similarity test:
#   -central   | east, west, north, southeast, region
#   -age       | years, older, males, ages, children
#   -others    | some, among, many, have, themselves
#   -case      | cases, instance, example, any, if
#   -selected  | review, bibliography, collected, serve, presented
#   -widespread| recent, widely, especially, resulted, been
#   -speakers  | spoken, dialects, speak, languages, dialect
#   -festival  | festivals, celebrated, celebration, holiday, celebrations
# Analogy: king - man + woman = ?
# Top results:
#   - daughter        (score: 0.7672)
#   - queen           (score: 0.7619)
#   - wife            (score: 0.7457)
#   - princess        (score: 0.7415)
#   - married         (score: 0.7356)
# Analogy: father - man + woman = ?
# Top results:
#   - mother          (score: 1.0566)
#   - married         (score: 1.0182)
#   - wife            (score: 1.0101)
#   - daughter        (score: 0.9923)
#   - son             (score: 0.9180)
# Analogy: husband - man + woman = ?
# Top results:
#   - married         (score: 1.1096)
#   - marriage        (score: 1.0223)
#   - wife            (score: 1.0159)
#   - daughter        (score: 0.9609)
#   - her             (score: 0.9598)
#  -Epoch 14/20 | Iter 9000 | Loss: 1.9159
# Epoch 14/20, Loss: 1.9140
#  Validation similarity test:
#   -age       | years, older, ages, males, children
#   -back      | off, behind, when, down, up
#   -case      | cases, any, instance, example, if
#   -make      | making, them, makes, way, enough
#   -magnetic  | electromagnetic, electrical, electric, electrons, optical
#   -selected  | review, bibliography, presented, collected, references
#   -speakers  | spoken, dialects, speak, languages, speaking
#   -blues     | jazz, musicians, guitar, rhythm, band
# Analogy: king - man + woman = ?
# Top results:
#   - daughter        (score: 0.7642)
#   - throne          (score: 0.7470)
#   - son             (score: 0.7299)
#   - queen           (score: 0.7256)
#   - married         (score: 0.7227)
# Analogy: father - man + woman = ?
# Top results:
#   - mother          (score: 1.0483)
#   - married         (score: 0.9981)
#   - wife            (score: 0.9847)
#   - daughter        (score: 0.9536)
#   - son             (score: 0.9085)
# Analogy: husband - man + woman = ?
# Top results:
#   - married         (score: 1.0787)
#   - marriage        (score: 0.9954)
#   - wife            (score: 0.9677)
#   - her             (score: 0.9441)
#   - daughter        (score: 0.9266)
#  -Epoch 15/20 | Iter 9000 | Loss: 1.9088
# Epoch 15/20, Loss: 1.9069
#  Validation similarity test:
#   -back      | off, behind, down, when, again
#   -countries | nations, europe, european, eu, belgium
#   -central   | east, west, south, north, southeast
#   -man       | my, girl, woman, evil, boy
#   -ft        | metres, meters, feet, height, miles
#   -widespread| especially, widely, recent, increasingly, resulted
#   -festival  | festivals, celebrated, celebration, holiday, celebrations
#   -selected  | bibliography, review, collected, presented, edited
# Analogy: king - man + woman = ?
# Top results:
#   - daughter        (score: 0.7732)
#   - queen           (score: 0.7447)
#   - throne          (score: 0.7318)
#   - princess        (score: 0.7115)
#   - wife            (score: 0.7037)
# Analogy: father - man + woman = ?
# Top results:
#   - mother          (score: 1.0395)
#   - wife            (score: 0.9857)
#   - married         (score: 0.9754)
#   - daughter        (score: 0.9684)
#   - son             (score: 0.9014)
# Analogy: husband - man + woman = ?
# Top results:
#   - married         (score: 1.0723)
#   - marriage        (score: 1.0133)
#   - her             (score: 0.9791)
#   - wife            (score: 0.9690)
#   - she             (score: 0.9533)
#  -Epoch 16/20 | Iter 9000 | Loss: 1.9023
# Epoch 16/20, Loss: 1.9008
#  Validation similarity test:
#   -make      | making, give, makes, them, way
#   -central   | east, west, north, south, southeast
#   -countries | nations, europe, belgium, united, european
#   -man       | woman, girl, my, boy, himself
#   -blues     | jazz, musicians, guitar, rhythm, band
#   -festival  | festivals, celebrated, celebration, holiday, celebrations
#   -widespread| widely, especially, recent, increasingly, popularity
#   -pop       | rock, album, songs, albums, music
# Analogy: king - man + woman = ?
# Top results:
#   - daughter        (score: 0.7341)
#   - queen           (score: 0.7173)
#   - throne          (score: 0.7163)
#   - son             (score: 0.6827)
#   - princess        (score: 0.6705)
# Analogy: father - man + woman = ?
# Top results:
#   - mother          (score: 1.0332)
#   - wife            (score: 0.9588)
#   - daughter        (score: 0.9406)
#   - married         (score: 0.9332)
#   - son             (score: 0.8914)
# Analogy: husband - man + woman = ?
# Top results:
#   - married         (score: 1.0267)
#   - marriage        (score: 0.9787)
#   - her             (score: 0.9685)
#   - wife            (score: 0.9391)
#   - she             (score: 0.9208)
#  -Epoch 17/20 | Iter 9000 | Loss: 1.8961
# Epoch 17/20, Loss: 1.8952
#  Validation similarity test:
#   -age       | years, older, males, ages, females
#   -control   | controlled, controlling, controls, effectively, power
#   -man       | woman, my, girl, boy, himself
#   -central   | east, west, north, south, region
#   -alliance  | coalition, party, allies, parties, faction
#   -blues     | jazz, musicians, rhythm, guitar, songs
#   -speakers  | spoken, dialects, languages, speak, speaking
#   -ft        | metres, meters, feet, height, miles
# Analogy: king - man + woman = ?
# Top results:
#   - daughter        (score: 0.7423)
#   - queen           (score: 0.7169)
#   - throne          (score: 0.7051)
#   - princess        (score: 0.6854)
#   - wife            (score: 0.6826)
# Analogy: father - man + woman = ?
# Top results:
#   - mother          (score: 1.0395)
#   - wife            (score: 0.9643)
#   - married         (score: 0.9339)
#   - daughter        (score: 0.9331)
#   - son             (score: 0.8815)
# Analogy: husband - man + woman = ?
# Top results:
#   - married         (score: 1.0486)
#   - marriage        (score: 0.9674)
#   - her             (score: 0.9502)
#   - wife            (score: 0.9494)
#   - she             (score: 0.9208)
#  -Epoch 18/20 | Iter 9000 | Loss: 1.8916
# Epoch 18/20, Loss: 1.8906
#  Validation similarity test:
#   -make      | making, makes, way, so, give
#   -back      | off, behind, down, when, again
#   -countries | nations, europe, belgium, united, european
#   -age       | years, older, males, ages, household
#   -widespread| widely, recent, popularity, especially, resulted
#   -blues     | jazz, musicians, guitar, rhythm, folk
#   -festival  | festivals, celebration, celebrated, holiday, celebrations
#   -selected  | bibliography, review, presented, each, papers
# Analogy: king - man + woman = ?
# Top results:
#   - daughter        (score: 0.7362)
#   - queen           (score: 0.7346)
#   - throne          (score: 0.7092)
#   - princess        (score: 0.6694)
#   - son             (score: 0.6652)
# Analogy: father - man + woman = ?
# Top results:
#   - mother          (score: 1.0157)
#   - daughter        (score: 0.9478)
#   - married         (score: 0.9284)
#   - wife            (score: 0.9217)
#   - son             (score: 0.8943)
# Analogy: husband - man + woman = ?
# Top results:
#   - married         (score: 1.0187)
#   - marriage        (score: 0.9636)
#   - her             (score: 0.9381)
#   - daughter        (score: 0.9375)
#   - wife            (score: 0.9078)
#  -Epoch 19/20 | Iter 9000 | Loss: 1.8869
# Epoch 19/20, Loss: 1.8861
#%%
# load checkpoint
checkpoint = torch.load("./weights/skipgram_negativesampling_model.pth")
embedding_size = checkpoint['embedding_size']
vocab_size = checkpoint['vocab_size']
epoch = checkpoint['epochs']
loss = checkpoint['loss']
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = SkipGramWithNegativeSampling(vocab_size, embedding_size)
model.to(device)
model.load_state_dict(checkpoint['state_dict'])
print(f'SkipGramWithNegativeSampling model loaded!')
print(f"{epoch=}")
print(f"{loss=}")
print(f"{embedding_size=}")
print(f"{vocab_size=:,}")
#%%
evalualte_model_quality(model.input_embedding,
                int2word,
                validation_size=8,
                window_size=window_size, 
                common_start_index=256, 
                uncommon_start_index=2000)
#%%

# lets see if the model can correctly identify their relationships
test_words = ['king', 'queen', 'man', 'woman', 'prince', 'princess']
check_semantic_analogy(test_words, model.input_embedding, word2int, device)
test_words2 = ['father', 'mother', 'man','woman','son','daughter']
check_semantic_analogy(test_words2, model.input_embedding, word2int, device)

# should give us son and price
check_analogy_test('daughter', 'woman', 'man', model.input_embedding, word2int, int2word)
check_analogy_test('princess', 'woman', 'man', model.input_embedding, word2int, int2word)

# find nearest neighbors to each word
check_nearest_neighbors('king', model.input_embedding, word2int, int2word, topk=5)
check_nearest_neighbors('Iran', model.input_embedding, word2int, int2word, topk=5)

#%%
visualize_embedding_space(model.input_embedding, int2word, words_to_visualize=200, use_pca=False)
visualize_embedding_space(model.input_embedding, int2word, words_to_visualize=200, use_pca=True)

# recap
# we used cosine similarity values to show high similarity for semantically related words.
# we saw that words with similar contexts cluster together in the embedding space.
# we also used embeddings can be manipulated mathematically and yet get meaningful 
# results. in addition to that we used nearest neighbors to see how the model produces
# meaningful and related words for a given word.
# 
# in order to improve our results we need to fine-tune our training regime, 
# from better hyperparamter selection (we saw one example using threshold but
# there are more to this such as e.g. choosing better learning rate and its
# decaying strategy, embeddingsize, etc) to training for longer periods.
# also in subsampling part, making sure frequent words are not overly dominant 
# improves the results as we saw first hand and finally its important to use a
# large enough validation set to assess embedding quality better. needless to say
# a larger decent dataset is always going to provide us with better embeddings.
# 


#%%
# TEXT CNN https://mlwhiz.com/blog/2019/03/09/deeplearning_architectures_text_classification/?utm_campaign=shareaholic&utm_medium=reddit&utm_source=news 


#%% CTCloss
# https://github.com/BelBES/crnn-pytorch
# https://towardsdatascience.com/intuitively-understanding-connectionist-temporal-classification-3797e43a86c
# https://machinelearning-blog.com/2018/09/05/753/
# https://stats.stackexchange.com/questions/320868/what-is-connectionist-temporal-classification-ctc
# https://distill.pub/2017/ctc/
# https://github.com/cmudeeplearning11785/Fall2018-tutorials/tree/master/recitation-8

#%%
# Named-Entity Recognition(NER). 
# a good video on positional encoding 2026 : https://www.youtube.com/watch?v=A-cgFIKDsPc
# %%
