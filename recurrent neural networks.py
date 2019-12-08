#%% 
# In the name of God the most compassinate the most merciful
# introduction to RNNs in Pytorch
import numpy as np 
import string
import torch
import torch.nn as nn 
import torch.nn.functional as F 
import matplotlib.pyplot as plt
import torch.optim as optim  
%matplotlib inline 


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
plt.plot(sample_sin,color='w')
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
    # plot([x], y, [fmt], *, data=None, **kwargs)
    # plot([x], y, [fmt], [x2], y2, [fmt2], ..., **kwargs)
    # The optional parameter fmt is a convenient way for 
    # defining basic formatting like color, marker and linestyle. 
    # It's a shortcut string notation described in the Notes section below.
    # plot(y, 'r+')     # ditto, but with red plusses
    # The following two lines are the same, the firts one uses [fmt]
    # >>> plot(x, y, 'go--', linewidth=2, markersize=12)  
    # >>> plot(x, y, color='green', marker='o', linestyle='dashed',
    #          linewidth=2, markersize=12
    plt.plot(x, fmt, label=label1)
    plt.plot(y, fmt2, label=label2)

# Any way, what we just created and plotted is a single sample. a sample of 30 timesteps 
# we need more of these samples to train our model. so we eaither need to create a datset
# beforehand and then during training, read from it, or we can just simulate that, by creating
# a generator which generates a sample each time it is called! 
# in a dataset we need both a sample and its label!
def dataset_sample(i, seq_len=10, device=torch.device('cuda')):
    sample_raw = torch.linspace(i*np.pi, (i+1)*np.pi, seq_len+1)
    sample_data = sample_raw.sin().to(device)
    data = sample_data[:-1].view(1,seq_len,1)
    label = sample_data[1:].view(1,seq_len,1)
    yield sample_raw.numpy(), data, label

#lets test this 
raw, data, label = next(iter(dataset_sample(0)))
data = data.view(-1)
label =label.view(-1)
plot_sample(data.cpu(), label.cpu())
# plt.clf()
plt.plot(data.cpu(),'r+-', label='input, x')
# now lets create our RNN. 
class RNN_Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1,
                 drpout=0.5, bidirect=False):
        super().__init__()

        self.input_size = input_dim
        self.hidden_size = hidden_dim
        self.output_size = output_dim
        # some arguments are self explanetory, so I explain about others
        # the first thing to pay attention when working with rnns in pytorch is
        # unlike normal vanilla rnn implementation, pytorchs rnns(grus and lstms)
        # are like this as well, will only return a tuple which includes, outputs
        # and hiddenstates for each timestep. obviously the output is not the actual
        # output that we want, for that we need to add another layer after the rnn
        # for example an fc layer with sigmoid to achive the actual output. (dont worry 
        # if its vague tt you, we will get to this in a moment)
        # the second thing is num_layers, basically this allows you to create a stacked rnn
        # usually you may choose between 1-3 layers. 
        # As it is said in the documentation, num_layers: Number of recurrent layers. E.g.,
        # setting num_layers=2 would mean stacking two RNNs together to form a stacked RNN,
        # with the second RNN taking in 'outputs' of the first RNN and computing the final 
        # results. Default is 1
        # batch_first, means, if you have your data in batches(batch is the first dim),
        # set this to true
        # bidirectional, means whether you want your rnn to be bidirectional! 
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
# its initially zero, we could use None as well, but I wanted you to see how 
# a hidden state dims looks like
# we can have a stacked rnn, if we want one, simply increase this number 
num_layers = 1
# uni or bidirection. ( 1 or 2)
direction = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# hidden_state = None
hidden_state = torch.zeros(size=(num_layers*direction, batch_size, hidden_dim)).to(device)
model = RNN_Net(input_dim=1,
                hidden_dim=hidden_dim,
                output_dim=1
                num_layers=num_layers).to(device)

# training 
criterion = nn.MSELoss()
# experiment with a large lr sucha s 0.1 and also a small lr like 0.0001 and see the changes
optimizer = optim.Adam(model.parameters(), lr=0.01)

# torch.autograd.set_detect_anomaly(True)
print(model)

for i in range(iteration):
    for (raw, data, label) in dataset_sample(i,
                                              seq_len=sequence_length,
                                              device=device):
                                
        (output, hidden_state) = model(data, hidden_state)

        ## Representing Memory ##
        # make a new variable for hidden and detach the hidden state from its history
        # this way, we don't backpropagate through the entire history
        hidden_state = hidden_state.data
        # make sure the dims for output and label is the same
        # otherwise, you may not get an error, but a sckewed result
        # that may bogle your mind for quite sometime!
        loss = criterion(output, label)
        print(f'iter: {i} loss: {loss.item():.6f} ')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if i%interval==0:
        x = data.view(-1).data.cpu().numpy()
        y = label.reshape(-1).data.cpu().numpy()
        output = output.view(-1)
        
        plt.plot(x, 'r.', label='x-data')
        plt.plot(y,'w.', label='label')
        plt.plot(output.data.cpu().numpy(), 'y+',label='output')
        
        plt.legend()
        plt.show()
#%% 
# Now using GRU and LSTM is the same, but we are goingto use them 
# in a new architecture. lets delve into the realm of nlp and write
# a simple text generator! 

# for our dataset, we use gutenberg opensource library which contains over 60K free ebooks,
# you can access  this library using this link : http://www.gutenberg.org
# we use http://www.gutenberg.org/files/1399/1399-0.txt but feel free to use anything you
# you like !
# lets roll everybody!:)

# lets download the book and create a dataset out of it
# ref : https://stackoverflow.com/questions/7243750/download-file-from-web-in-python-3 
def download_ebook(url='http://www.gutenberg.org/files/1399/1399-0.txt',
                   file_name='corpus.txt'):
    import urllib.request as request
    import os, sys

    dir_name = 'data'
    file_name_path = os.path.join(dir_name, file_name)

    if os.path.exists(file_name_path):
        return file_name_path
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    try:
        request.urlretrieve(url, file_name_path)
        return file_name_path
    except :
        print(f'exception occurec!: {sys.exc_info()[0]} ')
        return None

with open(download_ebook(),'r') as file: 
    corpus_raw = file.read()

# we use repr to see the special characters
# without it, the print statement will render them     
# but before that lets have a minimal preprocessing step
# lets lower case all characters! this will help with 
# the end result, there are more preprocessings, but
# for now its enough, we will see more in the next sections
# such as sentiment analysis and word embedding section
corpus_raw = corpus_raw.translate(str.maketrans('','',string.punctuation))
corpus_raw = ''.join(filter(lambda x: x in set(string.printable), corpus_raw))
corpus_raw = ''.join([h.lower() for h in corpus_raw])

print(repr(corpus_raw[:10]))

# our first step would be to tekenize our corpus 
# basically, we will tokenize our corpus of text
# and then digitize each letter/symbol, and then 
# use the encoded representation of these digitized
# words. in order to encode them, we first need to 
# create a dictionary of each letter 
# first we find all unique symbols.
# using set, we can avoid duplicates
unique_chars = set(corpus_raw)
# now lets create a char2int and int2char dictionaries!
int2char = dict(enumerate(unique_chars))
char2int = {c:d for d,c in int2char.items()}
# print(chars)
# print(int2char)
# print(char2int)
print(f'unique characters: {len(unique_chars)} : \n {unique_chars}')
# lets convert our corpus to int
corpus_digitized = np.array([char2int[char] for char in corpus_raw])
print(f'corpus digitized shape: {corpus_digitized.shape}')
print(corpus_digitized[:10])
print('after conversion: ')
print(repr(''.join([int2char[ii] for ii in corpus_digitized[:10]])))

# ok so far so good. we now need to create a one-hot representation of our input
# our inputs are digits, each digit as you saw, represents a single letter/symbol
# so in order to train our net, we must one_hot encode them,
def one_hot(input_array, length=10):
    # our list conains several digits, we will create a one hot vector
    # for each digit 
    one_hot_array = np.zeros(shape=(input_array.size, length), dtype=np.float32)
    one_hot_array[np.arange(input_array.size), input_array.flatten()] = 1
    return one_hot_array.reshape(*input_array.shape,length)

# lets test this 
print(one_hot(np.array([0,1,9]),10))
# now we need to have batches! 
def get_next_batch(corpus_digitized, batch_size=1, seq_len=10):
    # lets create batches from our corpus, first lets see
    # how many batches we can get from our corpus
    char_count = corpus_digitized.shape[0]
    each_batch_size = batch_size*seq_len
    batch_count = char_count // each_batch_size
    # now we should reshape our corpus data for easier access
    corpus_p = corpus_digitized[:batch_count * each_batch_size]
    # 
    corpus_p = corpus_p.reshape(batch_size, -1)
    # read one batch for data and label 
    x = np.zeros(shape=(batch_size, seq_len),dtype=np.int)
    y = np.zeros_like(x)

    for i in range(0, corpus_p.shape[1], seq_len):
        x[:,:] = corpus_p[:, i:i+seq_len]
        try : 

            y[:, :-1] = x[:, 1:]
            y[:,-1] = corpus_p[:,i+seq_len]
        except:
            y[:,:-1] = x[:,1:]
            y[:,-1] = corpus_p[:,0]
        yield x,y

x,y = next(iter(get_next_batch(corpus_digitized,batch_size=3,seq_len=8)))
print(x.shape)
print(y.shape)
print(x)
print(y)
print(one_hot(x,length=len(unique_chars)).shape)
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
data, labels = next(iter(get_next_batch(corpus_digitized,batch_size=2,seq_len=seq_len)))
print('initial data shape before one_hot encoding: ',data.shape)

data = one_hot(data, length=len(unique_chars))
labels = one_hot(labels, length=len(unique_chars))

data = torch.from_numpy(data)
labels = torch.from_numpy(labels)

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
device = torch.device('cuda' if torch.cuda.is_available()  else 'cpu')
model = lstm_char(rnn_type=rnn_type,
                  unique_chars=unique_chars, 
                  hidden_size=hidden_size,
                  num_layers=layers_cnt, 
                  bidirection=bidirection,dropout=0.3)
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr = 0.01)
criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=20)

epochs =60
# in order to not face the exploding gradient in lstm
# we clip the gradients
clip = 5.
interval = 1000 
batch_size = 128
label_length = len(unique_chars)
hidden_states = None 

val_ratio = 0.2
val_idx = int(corpus_digitized.size * (1-val_ratio))
train = corpus_digitized[:val_idx]
val = corpus_digitized[val_idx:]

print(f'rnn_type: {model.rnn_type}')
print(f'corpus size: {corpus_digitized.size}')
print(f'val size: {val.size}')
print(f'train size: {train.size}')
print(f'val + train: {val.size + train.size}')
assert train.size + val.size == corpus_digitized.size ,'they must be equale!'

for e in range(epochs):
    for i, (data, label) in enumerate(get_next_batch(train, batch_size,seq_len=seq_len)):

        #one hot encode 
        model.train()
        data = torch.from_numpy(one_hot(data,length=label_length)).to(device)
        label = torch.from_numpy(label).to(device)

        output , hidden_states = model(data, hidden_states)
        if model.rnn_type == 'lstm':
            hidden_states = tuple(h.data for h in hidden_states)
        else:#RNN, GRU
            hidden_states = hidden_states.data 
        
        
        # we dont one_hot_encode our labels, the crossEntropy() layer will do this internally!
        # since for output, we reshaped it so that the batch_size and sequence length are fused
        # we do the same thing for labels, so it can be used in cossentropy!()
        # the crossentropy loss, will internally convert each digit to its corosponding one_hot
        # encoded vector. so basically we are just making sure,  the shape between, output and 
        # label is the same 
        label = label.view(batch_size*seq_len).long()
        loss = criterion(output, label)
        optimizer.zero_grad()
        loss.backward()
        # note the _, which indicates the inplace operation!
        torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=5.)
        optimizer.step()
        if i%interval==0:
            print(f'epoch: {e}/{epochs} loss: {loss.item():.4f} lr: {scheduler.get_lr()[-1]:.6f}')
    scheduler.step()
# test 
print('test')
hidden_states = None
for i, (data,label) in enumerate(get_next_batch(val,batch_size,seq_len)):
    with torch.no_grad():
        model.eval()

        data = torch.from_numpy(one_hot(data,label_length)).to(device)
        label = torch.from_numpy(label).to(device)

        output, hidden_states = model(data,hidden_states)
        if model.rnn_type == 'lstm':
            hidden_states = tuple(h.data for h in hidden_states)
        else:#RNN, GRU
            hidden_states = hidden_states.data 
        loss = criterion(output,label.view(batch_size*seq_len).long())
        if i % interval ==0:
            print(f'loss: {loss.item():.4f}')
#%%
# now its time for sampling and generating new text. 
# basically this boils down to feeding a random character and then 
# feed the next generated character to the next timestep as the next input
# and this goes on till we generate the whole text. 
# since our network produces distributions , we use softmax to get the probablity
# , for each timestep, we can choose the highest probable outcome, or just randomly
# choose one, you'll see how this is done if this is vagueto you. 
# first we need to create a fucntion that does one thing only! 
# feed one character and retreieve one character from our neytwork. 
# we then use this function to create more characters in a loop. thats basically it!
def predict(model, input, unique_chars, hidden_states=None, topk=None):
    model.eval()
    int2char = model.int2char
    char2int = model.char2int 
    # print(char2int)
    # convert input string into corrosponding ids
    input =  np.array([char2int[input]]).reshape(1,-1)
    one_hot_vec = torch.from_numpy(one_hot(input, len(unique_chars))).to(device)
    output, hidden_states = model(one_hot_vec, hidden_states)

    output = torch.nn.functional.softmax(output,dim=1)
    # now our output has probabilities for each sequence/timestep
    # we will choose the highest one here 
    if topk==None:
        indexes = output.topk(np.arrange(len(unique_chars)))
    else:
        probs, indexes = output.topk(k=topk, dim=1)
        indexes = indexes.cpu().data.numpy().squeeze()
        probs = probs.cpu().data.numpy().squeeze()

    char = np.random.choice(indexes,p=probs/probs.sum())
    return int2char[char], hidden_states

def sample(model, size=10, prime='hello there'):
    
    chars = [ch.lower() for ch in prime]
    h = None
    prime = prime.lower()
    # print(unique_chars)
    for ch in prime:
        o,h = predict(model, ch, unique_chars,h,topk=5)
    chars.append(ch)

    for c in range(size):
        o, h = predict(model,chars[-1],unique_chars,h,topk=5)
        chars.append(o) 

    return ''.join(chars)

print(unique_chars)
print(sample(model, size=200,prime='The '))
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
# note that in our simple case of text generation, the bilstm wouldnt magically make everything better
# infact you may see the loss decreases much more, but the text generation is aweful! guess what
# is causing this? 


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
# before we continue, for sequence to sequence models and translations, one of the good 
# resources is https://www.manythings.org/anki/ that we can download text corpus and use it
# for translations. OK now lets continue with attention mechanism 
# as you can see, I have posted a lot of resources that you can use, I myself read some of them
# and will quote from them. so what is an attention mechanism and why do we even care? 
# Attention mechanism, as its name implies, is a mechanism which helps the network to pay more
# attention on specific parts of the input in order to produce more plausible outcome. 
# it was initially proposed for NMT or neural machine translation, where for example, you want
# to translate a sentence from one language to another. in a traditional case which we saw earlier
# in such cases, a seq2seq model is used, that is, a network comprising of two networks, an encoder
# and a decoder, where the input sequence is fed to the encoder, a decoder ultimately recieves a 
# compressed representation , representing the input sequence, from the encoder part and then, 
# tries  to produce a sequence as the answer. the problem with this procedure was/is that for a long
# sequence, we cant transfer the information from earlier time steps, its just simply not possible 
# (yes I know how we said about the lstm and gru gates, retaining earlier time step features, they 
# idea here, is even if lstm gates simply and ease the transfer of a specific feature from earlier
# time steps to the later timesteps, lots of such information will be lost becasue the last output
# is simply a finite fixed-size vector which can only accomated so much features, and mostly they 
# will be features from recent timesteps as apposed to earlier ones. please note that, there 
# are lots of relationships between each word, and also underlying concept in a given sequence, 
# suppose, in an optimal case, your sequence, had 40 underlying features, that needed to be identified 
# and used so a perefect output be created, however, since there is no mechanism to retain 'all' of 
# these features, and we face a fixed final vector, plus our training procedure is not perefect and 
# also have noise!, only a handful of such features get the chance to be transfered, features do get
# identified, but they cant be utilized as we dont have a mechanism to use them effectively so in the
# current procedure, they just get lost. so what should we do then? we can use all of t he states 
# from all previous timesteps instead of using only the last one. this way, we can provide much more
# information and this wealth of information at each timestep can help the network produce better result
# but how do we do that? surely not all hidden states, are equally important when it comes to producing 
# a translation, a new word e.g. here we can rank them based on how much they affect the outcome. 
# this way the network will gradually understand the relation ship and focuses on the correct states
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

# The first type of Attention, commonly referred to as Additive Attention, came from a paper by 
# Dzmitry Bahdanau, which explains the less-descriptive original name. The paper aimed to improve the
# sequence-to-sequence model in machine translation by aligning the decoder with the relevant input
# sentences and implementing Attention. The entire step-by-step process of applying Attention in
# Bahdanau’s paper is as follows:
# 1.Producing the Encoder Hidden States - Encoder produces hidden states of each element in the input
# sequence
# 2.Calculating Alignment Scores between the previous decoder hidden state and each of the encoder’s
# hidden states are calculated (Note: The last encoder hidden state can be used as the first hidden
# state in the decoder)
# 3.Softmaxing the Alignment Scores - the alignment scores for each encoder hidden state are combined 
# and represented in a single vector and subsequently softmaxed
# 4.Calculating the Context Vector - the encoder hidden states and their respective alignment scores 
# are multiplied to form the context vector
# 5.Decoding the Output - the context vector is concatenated with the previous decoder output and fed
# into the Decoder RNN for that time step along with the previous decoder hidden state to produce a 
# new output
# The process (steps 2-5) repeats itself for each time step of the decoder until an token is produced
# or output is past the specified maximum length

# For our first step, we’ll be using an RNN or any of its variants (e.g. LSTM, GRU) 
# to encode the input sequence. After passing the input sequence through the encoder RNN,
# a hidden state/output will be produced for each input passed in. Instead of using only the hidden
# state at the final time step, we’ll be carrying forward all the hidden states produced by the 
# encoder to the next step

# After obtaining all of our encoder outputs, we can start using the decoder to produce outputs. 
# At each time step of the decoder, we have to calculate the alignment score of each encoder output
# with respect to the decoder input and hidden state at that time step. The alignment score is the 
# essence of the Attention mechanism, as it quantifies the amount of “Attention” the decoder will 
# place on each of the encoder outputs when producing the next output.

# The alignment scores for Bahdanau Attention are calculated using the hidden state produced by the decoder in the previous time step and the encoder outputs with the following equation:

# score_alignment = W_combined * tanh(W_decoder * H_decoder + W_encoder * H_encoder)

# as you can see, its basically the decoders hidden state plus the encoders hidden state which are
# being used in a tanh transformation function. the weights are basically going to specify how much 
# importantce each hidden state has. 

# The decoder hidden state and encoder outputs will be passed through their individual 
# Linear layer(that is we use nn.Linear without a bias since it simply does a W*input! and makes life easier for us, without it, we should define a new parameter W and multiply it by the decoder hidden state, its the same thing! but uglier! so  thats why we simply use a linear layer as a learnable parameter for (W_decoder*H_decoder)) and have their own individual trainable weights.

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

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size=30, num_layers=1, bidirectional=False,drp=0.3):
        super().__init__()

        # **h_0** of shape (num_layers * num_directions, batch, hidden_size)
        self.encoder = nn.LSTM(input_size=hidden_size,hidden_size=hidden_size,num_layers=num_layers,
                              bidirectional=bidirectional, batch_first=True,dropout=drp)
        self.embedding = nn.Embedding(num_embeddings=input_size, embedding_dim = hidden_size)                      

    def forward(self, x, h):
        x = self.embedding(x)
        return self.encoder(x,h)

#this is basically a decoder part 
class BahdanauAttention(nn.Module):
    def __init__(self, output_size, hidden_size=30, num_layers =1,bidirectional=False, drp=0.3):
        super().__init__()
        
        self.input_size = hidden_size*2 
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional 
        self.drp = drp 

        # we need to calculate (score = W_v . tanh( W_d*H_d + W_e*H_e))
        # W_v shape is nx1, W_d is n×n and W_e n×2n these shapes can be found 
        # in the bahdanaus paper page 14.  
        # and the output of score must be a single number for each 
        # encoders hiddenstates(which means, seq_len since we have a 
        # n output (hiddenstate) for every sequence). 
        # always remeber, we only need the W and not bias! so we set them off
        self.W_d = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_e = nn.Linear(hidden_size, hidden_size, bias=False)
        # attention combine
        self.W_v = nn.Parameter(torch.FloatTensor(1, hidden_size))


        self.embedding = nn.Embedding(num_embeddings=output_size, embedding_dim = hidden_size)
        # input_size is hidden_size * 2, since unlike before, now 
        # in addition to the hidden_state we used to get from encoder, we now 
        # have the attention vector that gets concatenated together and fed to the decoder.
        #  
        self.lstm = nn.LSTM(input_size=hidden_size * 2, hidden_size=hidden_size,num_layers=num_layers,
                            bidirectional=bidirectional, batch_first=True,dropout=drp)
        
        # LSTMCell(input_size: int, hidden_size: int, bias: bool) -> None
        self.lstm_cell = nn.LSTMCell(hidden_size * 2, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)


    # this forward, we assume , we get inputs of length 1 sequence 
    # that means, our inputs are fed one timestep at a time 
    # thats why we are using the lstm layer and it works!
    # for this to work, we must have a loop in our training procesure
    # where we feed one timestep at a time. 
    def attention_cell(self, x_t, hidden_states_t, encoder_outputs):

        # x_t = self.embedding(x_t)
        # since lstm has two states, h and c, we only care about h!
        (h_prev, c) = hidden_states_t
        # lets first calculate the score, we use the previous hiddenstate
        # of the decoder, which can be none or the last state of the encoder
        # As we have already read there is no previous hidden state or output 
        # for the first decoder step, the last encoder hidden state and a 
        # Start Of String (<SOS>) token can be used to replace these
        # two, respectively.

        D = self.W_d(h_prev)
        print(f'h_prev shape: {h_prev.shape}')
        print(f'encoders.shape: {encoder_outputs.shape}')
        print(f'W_e shape: {self.W_e.weight.shape}')
        encoder_outputs = encoder_outputs.reshape(encoder_outputs.size(0),-1)
        print(f'encoders.shape: {encoder_outputs.shape}')
        E = self.W_e(encoder_outputs)
        
        score_raw_part1 = F.tanh(D + E)
        print(f'tanh(w_d*h_prev + w_e*encoders): {score_raw_part1.shape}')
        # scale it , we use bmm which is batchmatrixmultiplication 
        score_raw = score_raw_part1.bmm(self.W_v.unsqueeze(2))
        print(f'score raw(W_v*tanh()): {score_raw.shape}')

        # normalize it using softmax and get our attention weights
        # these will be multiplied by our encoders states
        weights = F.softmax(score_raw, dim=1)
        print(f'attention weights: {weights.shape}')
        # now create context vector which is attention_weights * encoder outputs(states)
        # Multiplying the Attention weights with encoder outputs to get the context vector
        # since we are dealing with not a sample but several samples at the same time, we
        # use bmm. 
        context_vector = torch.bmm(weights.unsqueeze(0), encoder_outputs.unsqueeze(0))
        print(f'context_vector: {context_vector.shape}')

        # Concatenating context vector with embedded input word
        decoder_input = torch.cat((x_t, context_vector[0]), 1).unsqueeze(0)
        print(f'output(concat(context_vector[0],x_t)): {decoder_input.shape}')
        # Passing the concatenated vector as input to the LSTM cell
        output, hidden = self.lstm(decoder_input, hidden)
        # Passing the LSTM output through a Linear layer acting as a classifier
        output = F.log_softmax(self.classifier(output[0]), dim=1)
        return output, hidden, weights

    def forward(self, x, hidden_states, encoder_outputs):

        x = self.embedding(x)
        print(f'x.shape after embedding:  {x.shape}')
        # x = x.view(1, -1)
        # print(f'x.shape after view(1,-1) : {x.shape}')

        # we can set it to none 
        # hidden_states = None

        # since lstm has two states, h and c, we only care about h!
        # x.shape = (batch, timestep, dims)
        O = torch.zeros_like(x)
        H = []
        A = []
        for t in range (x.size(1)):
            O[:,t,:], hidden_states, weights = self.attention_cell(x[:,t,:], hidden_states,encoder_outputs)
            H.append(hidden_states)
            A.append(weights)
        return O, torch.tensor(H), torch.tensor(A)

# ok lets test our implementation so far and see if it works well
# first lets create a dummy input 
input_np = np.array([[1,2,3],[0,4,1]])
input = one_hot(input_np,length=5)
# print(f'input_one hot encoded: \n{input}')
encoder = Encoder(5)
model = BahdanauAttention(5)
# batch, n_layer*bid
input = torch.from_numpy(input)
print(f'input: {input.shape}')
d_input = torch.from_numpy(input_np).long()

encoder_outputs, encoder_last_hidden_states = encoder(d_input, None)


print(f'encoder output: {encoder_outputs.shape}')
print(f'h from encoder(shape (n_layers * num_dir, batch, hidden_size)) \n{encoder_last_hidden_states[0].shape}')
yz = model(d_input, encoder_last_hidden_states, encoder_outputs)

print('results: ')
print(f'input {input}')
print(f'decoder output: {yz}')
#%%
# lets implement this one first and then we implement the luoungs version :






#%% sentiment analysis 
# lets do sentiment analysis . we read a bunch of reviews with their corrosponding labels
# here are the steps we need to take
# we know our label contains words, positive and neagative, so we convert them into numbers, 1, 0
# our reviews must be dgitized so we can feed them into our network  , but before that we need
# to do some preprocessings. the preprocessings include 
# 1. make everything lower case 
# 2. remove punctuations 
# 3. remove special characters such as \n 
# 4. split only words! we dont want to create text, so creating dictionaries of characters 
# as apposed to creating dictionary of words is not going to suite us, becasue we intend on 
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
with open(r'data\reviews.txt','r') as file: 
    reviews_raw = file.read().lower()
with open(r'data\labels.txt', 'r') as file: 
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
    padded_array = np.zeros(shape=(len(new_reviews), max_length),dtype=np.int)
    for i,review in enumerate(new_reviews) : 
        padded_array[i,-len(review):] = review[:max_length]
    return padded_array 

reviews_digitized = pad_input(new_reviews, 150)
print(reviews_digitized[:1])

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