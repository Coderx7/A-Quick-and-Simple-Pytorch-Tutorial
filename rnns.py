# %%
# in the name of God the most compassionate the most merciful 
# in this section Im going to review using rnns(rnn/gru/lstm)
# and use it to prediction/regression, generate text, classification, sentiment analysis
# we use pytorch
import numpy as np 
import torch
import torch.nn as nn
import matplotlib.pyplot as plt 
%matplotlib inline 
plt.rcParams["figure.figsize"] = (4,3)
print(f'torch version: {torch.__version__}')

# %%
# lets first get familiar with the rnn. 
# to start, lets create some dummy numbers, we use linspace to create
# evenly spaced numbers betwene two points. here we are going for 30
# numbers.
#  
xs = torch.linspace(0, torch.pi, 30)
xs_sin = torch.sin(xs)
# lets plot this and see what we are dealing here 
# plt.figure(figsize=(4,3))
plt.plot(xs, color='green', linestyle='--', label='raw')
plt.plot(xs_sin, 'go', label='sin')

# %%
# we did this to create somekind of underlying structure for the model to discover. 
# we could use a formula with random numbers to create that, like a regression problem
# plt.figure(figsize=(4,3))
XS = torch.randn(size=(30,))
XS.mul_(5).add_(-3)
plt.plot(XS)
# note that for our example linspace/sin is better, becasue we have to keep generating 
# a different value for our dataset and it should stay small, otherwise everything would
# go haywire!
# %%
# ok that was the sample for our timeseries prediction example, whats the label? 
# the label is usually the next token/item in the sequence 
xs = XS[:-1]
ys = XS[1:]
# we can use different linestyles such as 'solid', 'dashed','dotted' or 'None'
# or their shortcuts in fmt parameter, where we can specify the basic formattings
# like color, marker, linestyle using their shortcuts. for example  
# '-' or 'solid' : solid line
# '--' or 'dashed': Dashed line
# '-.' or 'dashdot': Dash-dot line
# ':' or 'dotted': Dotted line
# '' , ' ' or 'None': No line'
#--------------
# shortcut for markers
# '.': Point marker
# ',': Pixel marker
# 'o': Circle marker
# 'v': Triangle down marker
# '^': Triangle up marker
# '<': Triangle left marker
# '>': Triangle right marker
# '1': Tri down marker
# '2': Tri up marker
# '3': Tri left marker
# '4': Tri right marker
# 's': Square marker
# 'p': Pentagon marker
# '*': Star marker
# 'h': Hexagon1 marker
# 'H': Hexagon2 marker
# '+': Plus marker
# 'x': X marker
# 'D': Diamond marker
# 'd': Thin diamond marker
# '|': Vline marker
# '_': Hline marker
# g stands for green, r stands for red
plt.plot(xs,'g-D',label='data')
plt.plot(ys, 'r:', label='label')

# %%
# now lets create a dataset of these points. we need much more points for training our network
# one way is to create a dataset completely, or use generators to generate data on the fly 
# since we intend on using sin, we can easily do this by creating x numbers from a given i 
def generate_data_label(i, sequence_length=10):
    x = torch.linspace(i*torch.pi, i+1*torch.pi, sequence_length).sin()
    data = x[:-1]
    label = x[1:]
    yield data, label

# note a generator is already an iterator, so you don’t need to call iter() on it 
# (if it was an iterable like a list, tuple, then you had to use iter to make it iterator)
# but since a generator is already an iterator the next suffices
x,y = next(generate_data_label(0))
# newer version of pytorch offer many convinient functions such as tolist use them for small arrays
# for larger ones, use numpy to avoid needless overhead/copying
plt.plot(x.tolist())
plt.plot(y.tolist())

# %%
# but wait a second this is not right! we want a sequence of x timestep, this is one long 1d array! we want (batch,timestep,feature)
# lets fix this 
def generate_data_label(i, sequence_length=10):
    x = torch.linspace(i*torch.pi, (i+1)*torch.pi, sequence_length+1).sin()
    data = x[:-1].view(1,sequence_length,1)
    label = x[1:].view(1,sequence_length,1)
    yield data, label 
x,y = next(generate_data_label(0))
plt.plot(x.numpy().reshape(-1))
plt.plot(y.numpy().reshape(-1))    


# %%
# now lets create our RNN model 
class RNN_predictor(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers=1, dropout=0.0, is_bidirectional=False,device='cpu') -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.is_bidirectional = is_bidirectional
        self.direction = 2 if is_bidirectional else 1 
        # lets create a rnn layer 
        # our RNN requires inputsize, hiddensize, for output we need an extra fc layer. the pytorch 
        # however, generates the output for each timestep, and concats these with hiddenstates
        # and this is what we get. well see how to do that in a moment, 
        # and oh about the num_layers , it specifies the number of RNN layers, basically stack them
        # on top of another. num_layers=2 simply means two rnns stacked on top of eachother. 
        # the outputs/hiddenstates of the previous rnn goes to the next obviously!
        # bidirection, is set true, looks at the input in two mood, one the normal, and the other
        # in reverse. and the output would be two hiddenstates instead of 1, one for each mode of operation
        # the activation function is tanh by default as expected.
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                          batch_first=True, dropout=dropout, bidirectional=is_bidirectional,device=device)
        
        # for output, we take hiddenstate, and use that to generate the final output
        # note that The torch.nn.RNN class in PyTorch returns two outputs: outputs and hidden_state.
        # The 'outputs' are the hidden states at each time step. The outputs tensor 
        # has the shape (batch, seq_len, num_directions * hidden_size) if its batch_first=true.
        # (pytorch doesnt implement output part of rnn for RNN, it only returns the hiddenstates
        # at each timestep, so if anyone wants to do anything with them(like calculating the outputs by
        # mlutiplying them by a Woutput and running them through a softmax, they can do it, or if
        # they want to pass them to another rnn layer as is, they can do it as well. basically they
        # provide this flexibility for this reason.))
        # hidden_state: This is a tensor containing the hidden state for t = seq_len. In other words, 
        # this is the final hidden state output by the RNN. 
        # The hidden_state tensor has the shape (batch, num_layers * num_directions, hidden_size).
        # which one to use depends on our specific use case:
        # If we are interested in the hidden state at each time step 
        # (for example, if we are feeding the output of the RNN into another RNN, 
        # we should use output.
        # If we only care about the final output of the RNN 
        # (for example, in a sentiment analysis task where the entire sequence of words 
        # is used to determine a single output), we should use h_n.
        # if we use bidirection, we need twice the hiddensize
        self.fc = nn.Linear(hidden_size*self.direction, output_size)

    def forward(self, inputs, hidden_state=None):
        # print(f'{inputs.shape=}')
        # hidden_state in the begining can be None, or comefrom another source
        # obviously in subsequent calls, we always use the previous hidden_states
        rnn_all_hs_output, final_hidden_state = self.rnn(inputs, hidden_state)
        # print(f'{inputs.shape=}, {outputs.shape=}, {final_hidden_state.shape=}')
        # rnn class doesnt implement the output part of the RNN, so its on us 
        # to calculate the outputs, we can use the outputs part of the rnn, or its 
        # final_hidden_state. 
        # but lets use all the previous hidden_states, and create outputs for each of them
        # and then use this output as our model predictions for each timestep
        # we want to predict all the points in the sequence and not just 1, this is a 
        # sequence to sequence problem after all, we have a sequence in input and a sequence
        # in the output (we have t timsetsteps after all! (think of it as t characters in a string!
        # and each string is a what? yup a sequence!))
        # since our fc expects hidden_size, but we have b,t, hiddensize, we go ahead and merged the 
        # two dimensions (batch and timestep), we would get a large 2d matrix, which when fed to fc
        # will calculate the outputs for each timestep! if we need them(which in our case we do! cuz
        # our input is a sequence and our output is also a sequence (remember our label is our input
        # shifter to the right by one timestep!)), we just need to reshape
        # them back to the original shape (b,t,hiddensize) and use them the way we want!
        rnn_all_hs_output = rnn_all_hs_output.view(-1, self.hidden_size*self.direction)
        outputs = self.fc(rnn_all_hs_output)
        return outputs, final_hidden_state

# note that our input size here is 1, we just happen to have 10 of it (we have 10 timesteps here each 
# having input size of 1)
x,y = next(generate_data_label(1,20))
rnn = RNN_predictor(1,1, 40,1,is_bidirectional=False)
rnn(x)
# %%
# now lets create a training loop and see if our model can learn to predict sin!
# we need 
# a model 
# a criterion for loss 
# an optimizer for optimization 
# a learning rate scheduler for decaying the learning rate 
# thats pretty much it 
sequence_length = 20
hidden_size=50
input_size=1
output_size=1

model = RNN_predictor(input_size,output_size, hidden_size,num_layers=1,is_bidirectional=False)
device = "cuda" if torch.cuda.is_available() else "cpu"
optimizer = torch.optim.Adam(model.parameters(), lr = 0.1)
criterion = torch.nn.MSELoss()
# basically this is not needed, 0.1 is too much, 0.01 is prefect!
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,[1])
iteration = 100
batch_size=1
interval = 10
# we need an intial hiddenstate, it can be None or all zeros the shape based on the docs should be 
# (bidirection*num_layer, batch-size, hidden-size)
hidden_state = torch.zeros(size=((2 if model.is_bidirectional else 1)*model.num_layers, batch_size,model.hidden_size), device=device)
model = model.to(device=device)

for i in range(iteration):
    x,y = next(generate_data_label(i, sequence_length=sequence_length))
    x,y = tuple(t.to(device) for t in (x,y))
    outputs, hidden_state = model(x, hidden_state)
    # so we dont include it in computation graph
    hidden_state = hidden_state.data
    # lets make sure the shapes of both outputs and labels are the same
    loss = criterion(outputs, y.view(*outputs.shape))
    
    # do a backward but before it clearout the gradients 
    scheduler.step()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
        
    # print the loss 
    if i%interval==0:
        print(f'{i}) {loss=:.4f} ')
        # clear the figure 
        plt.clf()
        # lets also print the network prediction and compare it with the label 
        plt.plot(outputs.view(-1).detach().cpu().numpy(), 'r--*', label='outputs')
        plt.plot(y.view(-1).detach().cpu().numpy(), 'g-+', label='label')
        plt.legend()
        plt.show()

        
# %%
# ok that was RNNs, the lstms and GRUs are the same. now that we have a working knowledge 
# of RNNs, lets use a much better example, lets work on text generation, and use LSTM which 
# are way better than simple vanilla RNNs, LSTM are much better at retaining long depedencies
# than vanilla RNNs, (but still lacking compared to transformers) becasue of their architectureal 
# changes and the fact that they are much better than the rnns when it comes to vanishing gradients
# they are not immune to it, but pretty robust compared to RNNs.
# anyway lets dive in
# lets use a text dataset for this, we use http://www.gutenberg.org/files/1399/1399-0.txt'
import sys, os
print(f'{os.path.basename(__file__)}')
with open('/media/hossein/SSD1/code_dl/corpus.txt') as file: 
    corpus_raw = file.read()
    
print(f'{repr(corpus_raw[:100])}')
# in order to get better performance we need to do a few preprocessings, 
# one of them is to remove punctuations, the other is to make everything lowercase
# and the list goes on. lowercasing doesnt really do much, so 
# lets keep the letter cases as they are and only remove the punctuations.
# lets remove the punctuations to make things easier for our model 
# and get a better generation down the road, for this we use str.translate 
# with str.maketrans to remove all punctuations from the dataset. 
# The str.maketrans() function creates a translation table where when it’s 
# used with three arguments, each character in the third argument is mapped to None.
# this basically creates our table which the str.translate then uses to replace the punctuations
# with nothing (basically removing them)
# look at the str.maketrans('', '', string.punctuation), the first two arguments are empty 
# strings and the third argument is string.punctuation, which includes all punctuation characters.
import string
corpus_new = corpus_raw.translate(str.maketrans('','',string.punctuation))
print(f'{repr(corpus_new[:100])}')
# now lets tokenize our input for this we need to extract unqiue characters in our datset 
characters = set(corpus_new)
char_length = len(characters)
itoa = dict(enumerate(characters))
atoi = {c:i for i,c in itoa.items()}
# now we need to convert our dtaset from text representation to its digitized version 
# becasue ultimately we want to represent them as a tensor, and we need numbers. 
corpus_array = np.array([atoi[c] for c in corpus_new])
dataset_length = len(corpus_array)
print(f'{corpus_array[:100]}')
# now lets create a function that maps a given array of numbers returns the equivalent string
convert_to_str = lambda x : ''.join([itoa[i] for i in x])
print(f'{convert_to_str(corpus_array[:100])}')
# now we have one last step, we are not going to use embeddings here, so we cant simply 
# feed our decimal numbers as inputs, we instead one_hot encode them. we feed our model 
# with these one_hot_encoded values 
def one_hot_encode(num_array, length):
    # print(f'{num_array.shape=}')
    # we create a vector the same length as all the characters, 
    one_hot_array = np.zeros(shape=(len(num_array),length))
    # lets do this in one vectorized implementation, basically the first dimension
    # creates an array with values from 0- length of the array, basically acting like 
    # a for loop, it starts with 0, then goes and grabs the first element from num_array
    # which is given in the second dimension, and set it to 1. then goes for the next number
    # and does this for the second item in the list and then third and then fourth ... until
    # all is done.
    one_hot_array[np.arange(len(num_array)),num_array] = 1
    return one_hot_array
# lets test this 
batch_data = one_hot_encode(np.array([1,0,3,4,8]), 10)
print(batch_data)
# now lets test it with batched version : 
print(one_hot_encode(np.random.randint(0,10,size=(3,5)),10))
# but it fails, becasue the broadcasting fails (3,) vs (3,5)
# how can we fix this? we can flatten this 2d input and fix the issue!
#%%
def one_hot_encode(num_array, length):
    # note that since we are planning on accepting all array forms (1d,2d,etc)
    # we can no longer simply use len(), because len only returns the number of elements
    # in the first dimensions, previosly we had 1d array so len(array) would be fine as 
    # it would return the number of all elements (which were at the first dimension)
    # however, we now need to count all elements in all dimensions, we use .size now
    one_hot_array = np.zeros(shape=(num_array.size,length))
    # we need to do the same here, and also 
    # in order to prevent broadcasting issue
    # with 2d array, we simply flatten it
    one_hot_array[np.arange(num_array.size), num_array.reshape(-1)] = 1 
    # and before we send it back, we reshape our final result
    # note that here, we took the shape of the input array, and then used a -1
    # at the end, becasue we want to retain the input shape, if we get (3,5)
    # we want to get (3,5,10), not (3,50). because we want to preserved the 
    # timesteps, sequence length, (3,5) says, we have 3 sample sequence of 5 characters
    # we could write (3,5) as (3,5,1) becasue each character is represented as a single number
    # now that we represent each character as a vector of 10, we would obviously want (3,5,10)
    # thats why instead of one big vector for each sample i.e (3,50), we organize them properly
    # into different sequence so we ultimately have (batch, sequence, features)
    return one_hot_array.reshape(*num_array.shape, -1)

# now lets test it again 
print(one_hot_encode(np.array([1,0,3,4,8]), 10).shape)
print(one_hot_encode(np.random.randint(0,10,size=(3,5)),10).shape)
# and all works just fine
# if we wanted to use pytorch, one_hot_encode is already implemented 
print(torch.nn.functional.one_hot(torch.tensor([1,0,3,4,8]),10))
print(torch.nn.functional.one_hot(torch.randint(0,10,size=(3,5)),10))
# ok this works as expected. lets test it on corpus_digitized
print(f'one_hot_encode(corpus_array[:3], {char_length}):')
print(one_hot_encode(corpus_array[:3], len(characters)))
# ok so far so good, now we want to do the data loading part
# we need a function that gives us a new batch of data in the proper format
# the pytorch dataset doesnt really offer much for our case asfaras i know!
# we want to create a function that returns a batch of sequences. so we need
# to specify the sequence length, like how many characters do we want thisto 
# be? and then create a batch accordingly
def get_batch(corpus_array, batch_size=32, sequence_length=10):
    # lets see how many batches of data we can create out of our input
    # with the sepecifed sequence length. 
    # we need to generate batches of data and labels
    # labels are simply the data shifted to the right by one!
    length = len(corpus_array)
    batch_length = batch_size * sequence_length
    total_batch = length//(batch_length)
    # lets grab only the full batches
    cropped_size = total_batch * batch_length
    data = np.empty(shape=(batch_length))
    label = np.empty(shape=(batch_length))
    for index in range(0,cropped_size,batch_length):
        data = corpus_array[index:index+batch_length]
        # label is our data one character shifted to the right, so that 
        # our label always has one character after the corrosponding index in 
        # the data
        label[:-1] = data[1:]
        # for the last character, use the next character from the corpus,
        # if we are past the end of the corpus, read the first character
        # instead
        if (index+batch_length+1<cropped_size):
            label[-1:] = corpus_array[index+batch_length:index+batch_length+1]
        else:
            label[-1:] = corpus_array[0]
            
        yield data.reshape(batch_size, -1), label.reshape(batch_size,-1)

gen = get_batch(corpus_array=np.arange(20),batch_size=2,sequence_length=3)
for data, label in gen:
    print(f'{data=}')
    print(f'{label=}')
#%%
# now lets try this with our corpus 
def test_dataset(g):
    batch_data, batch_label = next(g)
    batch_data_str = ''.join(list(convert_to_str(s) for s in batch_data))
    batch_label_str = ''.join(list(convert_to_str(s) for s in batch_label))
    # print(batch_data_str)
    # print(batch_label_str)
    return batch_data_str+"\n"
    
g = get_batch(corpus_array=corpus_array,batch_size=5,sequence_length=5)
print(f'first 100 words: {convert_to_str(corpus_array[:100])}')
print(f'\nSeveral batches of data looks like this: ')
print(f"{''.join(list(test_dataset(g) for _ in range(4)))}")

# %%
# now its time to create our model. lets make our model dynamic so we can choose 
# to use rnn,gru or lstm as we wish, this way we can test different variants and see
# how they affect our results
class TextGenerator(nn.Module):
    def __init__(self,rnn_type,input_size,output_size,hidden_size,num_layers=1,
                 is_bidirectional=False,dropout=0.0,act='tanh',itoa={},atoi={},device='cpu') -> None:
        super().__init__()
        self.rnn_type = rnn_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.is_bidirectional = is_bidirectional
        self.device = device
        self.dropout_rate = dropout
        self.direction= 2 if is_bidirectional else 1 
        self.itoa = itoa
        self.atoi = atoi 
        self.dropout = nn.Dropout(dropout)
        if self.rnn_type == 'rnn':
            self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size,
                              bidirectional=is_bidirectional,
                              dropout=dropout, 
                              num_layers=num_layers,
                              nonlinearity=act,
                              batch_first=True)
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                              bidirectional=is_bidirectional,
                              dropout=dropout, 
                              num_layers=num_layers,
                              batch_first=True)
        else:
            self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                              bidirectional=is_bidirectional,
                              dropout=dropout, 
                              num_layers=num_layers,
                              batch_first=True)
            
        self.fc = nn.Linear(hidden_size*self.direction, output_size)
    
    def forward(self, x, hidden_state=None):
        all_states, final_hidden_state = self.rnn(x,hidden_state)
        # since we want to generate texts, we need all states to generate
        # an output for each timestep.
        # side note: 
        # if we use view we may face this issue:
        # RuntimeError: view size is not compatible with input tensor's size and stride
        # (at least one dimension spans across two contiguous subspaces). 
        # Use .reshape(...) instead.
        # The reason is in PyTorch, tensors have both a size and a stride. The size 
        # represents the dimensions of the tensor, while the stride represents the 
        # number of elements in memory that need to be skipped over to move to the 
        # next element along each dimension.
        # When you call .view(), PyTorch tries to return a new tensor that has the
        # same data but different size from the original tensor. However, this is 
        # only possible if the requested view is compatible with the original tensor’s 
        # size and stride.
        # If it’s not, you’ll get the error message you’re seeing. In such cases, you 
        # can use .reshape(), which returns a new tensor with the same data but different
        # size. Unlike .view(), .reshape() can handle tensors whose elements are not 
        # contiguous in memory.
        # but why wont view work sometimes?
        # The view() function in PyTorch works on contiguous tensors, that is the tensors 
        # where all the data is continuously stored in memory. 
        # However, some operations in PyTorch can cause a tensor to become non-contiguous.
        # thats when view() wont work as expected.
        # when we perform certain operations on a tensor, such as taking a transpose forexample,
        # it doesn’t rearrange the data in memory but only changes the way it is indexed, 
        # making the tensor non-contiguous.
        # In our case, the tensor all_states might be non-contiguous in memory. when we try
        # to use view() on it, we get a runtime error.
        # to make sure the tensor is contiguous, we can use contiguous() before view(), 
        # but what contiguous() does is it creates a new copy() and return a view from it,
        # so in fact its not any different than reshape, cuz reshape also creates a copy
        # thats why using reshape here is better than doing a contiguous().view()
        # 
        # also a reminder concerning view and reshape.
        # usually reshape migh be prefered, cuz it works on both contiguous and non-contiguous
        # tensors, while view doesnt(it only works on contiguous tensors). 
        # reshape function was introduced in version 0.4. it returns a tensor with the 
        # same data and number of elements as the input, but with the specified shape.
        # When possible, the returned tensor will be a view of the input. Otherwise, it 
        # will be a copy. This means that reshape may return a copy or a view of the 
        # original tensor, and we cannot predict whether it will return a view or a copy.
        # 
        # all_states = self.dropout(all_states)
        outputs = self.fc(all_states.reshape(-1, self.direction*self.hidden_size))
        return outputs,final_hidden_state
    
    @torch.no_grad()
    def sample_text_vanialla(self, max_length=100):
        # here we are simply generating text, we dont have any input, but for our model to generate
        # an output we need an output! so we give it a simple single character array of zero!
        # thats it, we can choose any number other than zero, like 1, 2, ...up to char_length
        # becasue this number is representing a single character after all
        prompt=torch.zeros(size=(1,1)).long()
        # or we could use this isntead, note the [[]], it adds 2 dimensions (make this 2d array with shape=(1,1)
        # prompt=torch.tensor([[0]]).long()
        # or use any other characters index like 10
        # prompt=torch.tensor([[10]]).long()
        # or even a character code directly! using atoi!
        # prompt=torch.tensor([[self.atoi[' ']]]).long()
        hidden_state = None
        char_list = []
        for i in range(max_length):
            # now we feed the prompt to the model, note that the input is a single character/digit for a single sequence
            # and keep repeating this to create new characters. we feed model a charcter and get a new character out!
            one_hot_prompt = torch.nn.functional.one_hot(prompt.to(self.device), self.input_size).float()
            outputs, hidden_state = self.forward(one_hot_prompt, hidden_state)
            hidden_state = tuple(s.data for s in (hidden_state)) if self.rnn_type =='lstm' else hidden_state.data
            # apply temperature to the logits (outputs)
            # outputs = outputs / temperature
            # we want probablities so lets use softmax 
            probs = outputs.softmax(dim=-1)
            # now we have bunch of probablities, so which one should we choose? 
            # we have fed a sequence of 1 0s, and expect the network to give us back a 
            # new character. so we have 85 classes or characters, 85 probablities to choose from
            # if we simply grab the highest probablity, that would be a greedy sampling and wouldnt
            # give us a good output. instead we use sampling, and based on the probablity of each class
            # try to sample from it, this way, higher probablity classes/characters are more likely to be
            # selected, while the least likely ones also do get a chance, albeit small, but they get their
            # chance as well and it will result in a much better output.
            # we fed our model a single sample of 1 character sequence, so we get (1,85) probablities. 
            # lets remove the first dimension, and feed it to torch.multinomial for sampling (torch.multinomial
            # wants 1d array! thats why we remove the first dimension)
            probs = probs.squeeze(0)
            # lets use this to sample, replacement=True, means the same character can be choosen again
            # basically torch.multinomial is the equivalent of numpy.random.choice (indexes, p=probs/probs.sum())
            idx = torch.multinomial(probs, 1, replacement=True)
            # now lets extract the character str and also feed this back to the model for the next round
            char_list.append(itoa[idx.item()])
            # now use this new idx/character to generate the next one 
            prompt = idx.unsqueeze(0) 
        print(''.join(char_list))
    
    @torch.no_grad()#added this version later, after I wrote the two other numpy based version below
    def sample_text_vanialla_with_temperature_with_topk_sampling(self, temperature=0.8, topk=5, max_length=100):
        # in this version, lets add temperature along with topk sampling and instead of a single probablity
        # we use top k probablities, basically the idea is, instead of sampling among all characters
        # /classes, lets grab the top k classes/characters and randomly choose one of them. this way
        # we should get much better results, becasue obviously we are choosing from most probably outcomes
        # than a bunch of likely and unlikely outcome whch happens by default when we look all the characters
        # /classes in the output
        # everything stays the same excep the probablities section, before we feed probablities to multinomial
        # so see there
        prompt=torch.zeros(size=(1,1)).long()
        # or we could use this isntead, note the [[]], it adds 2 dimensions (make this 2d array with shape=(1,1)
        # prompt=torch.tensor([[0]]).long()
        # or use any other characters index like 10
        # prompt=torch.tensor([[10]]).long()
        # or even a character code directly! using atoi!
        # prompt=torch.tensor([[self.atoi[' ']]]).long()
        hidden_state = None
        char_list = []
        for i in range(max_length):
            # now we feed the prompt to the model, note that the input is a single character/digit for a single sequence
            # and keep repeating this to create new characters. we feed model a charcter and get a new character out!
            one_hot_prompt = torch.nn.functional.one_hot(prompt.to(self.device), self.input_size).float()
            outputs, hidden_state = self.forward(one_hot_prompt, hidden_state)
            hidden_state = tuple(s.data for s in (hidden_state)) if self.rnn_type =='lstm' else hidden_state.data
            # apply temperature to the logits (outputs)
            outputs = outputs / temperature
            # we want probablities so lets use softmax 
            probs = outputs.softmax(dim=-1)
            # now we have bunch of probablities, so which one should we choose? 
            # we have fed a sequence of 1 0s, and expect the network to give us back a 
            # new character. so we have 85 classes or characters, 85 probablities to choose from
            # if we simply grab the highest probablity, that would be a greedy sampling and wouldnt
            # give us a good output. or previous try was to use sampling, and based on the probablity of 
            # each class try to sample from it, this way, higher probablity classes/characters are more likely to be
            # selected, while the least likely ones also do get a chance, albeit small, but they get their
            # chance as well and it will result in a much better output.
            # but right now, we want to sample from topk and not all 85 characters/classes. 
            # we fed our model a single sample of 1 character sequence, so we get (1,85) probablities. 
            # so lets grab top k probablities 
            probs, indexes = probs.topk(topk, dim=-1)
            # now we have our probs, what remains is to normalize these so that they all add to 1, making them
            # new probablities (i.e. imagine we grabed topk=3, and we got back 0.33, 0.21, 0.19, clearly they 
            # dont sum to 1, if we want to treat them as true probablities (they represent the original ouput probablities),
            # we need to normalize them, becasue we want to select from them based on their probability in this 
            # list (imagine they are raw logits and we want to convert them to probablity!) when we do this we 
            # see that the first item has 0.33/0.73=0.45, 0.21/0.73=0.28, 0.19/0.73=0.26, now we can easily sample
            # from this and it represents the original probablity much better) (otherwise they will be treated as weights
            # and create bias towards the larger numbers/weight(I explaned it in a moment)) 
            # lets remove the first dimension, and feed it to torch.multinomial for sampling (torch.multinomial
            # wants 1d array onlt! thats why we remove the first dimension)
            probs.squeeze_(0)
            # while we are at it lets remove the first dimension from indexes as well
            indexes.squeeze_(0)
            # normalize the probs 
            probs = probs/probs.sum(dim=-1)
            # now lets use this to sample, replacement=True, means the same character can be choosen again
            # basically torch.multinomial is the equivalent of numpy.random.choice (indexes, p=probs/probs.sum())
            # also note that, the output of multinomial is 0 to k, but we need their actual indexes they represent
            # so to get the actual class indexes, we use the indexes (i.e. the idx multinomial returns is relative
            # to the probablities/weights we feed it, to get the actual output indexes our model generated we need
            # to look into indexes)
            # side note: torch.multinomial, works with weights as well. that is we dont necessarily need to renormalize
            # our probablities, if we were to feed them as is, it would work as well and multinomial would treat them
            # as weights.(weights dont need to between 0 and 1, so decimal numbers like 10,12,3 can be considered as 
            # weights. however they must not be negative!)
            # but since we are dealing with probablities, its a good thing to do so. 
            # why? becasue when we are dealing with a subset of probabilities (the top-k probabilities here),
            # we are essentially creating a new probability distribution from these probabilities, and the sum of 
            # all probabilities in a probability distribution must equal 1.
            # If we didn't normalize the top-k probabilities, the torch.multinomial would still work, but the 
            # probabilities wouldnt be accurate. it would treat the unnormalized probabilities as weights, 
            # and the sampling would be biased towards the larger weights. by normalizing the probabilities, 
            # we ensure that the sampling is proportional to the original probabilities.
            idx = indexes[torch.multinomial(probs, 1, replacement=True)]
            # now lets extract the character str and also feed this back to the model for the next round
            char_list.append(itoa[idx.item()])
            # now use this new idx/character to generate the next one 
            prompt = idx.unsqueeze(0) 
        print(''.join(char_list))

    @torch.no_grad()
    def sample_text_better(self, prompt_str='hi', max_length=100):
        # basically preferably we want to start with an intial prompt, it can be anything, 
        # a single \n, ., or space, or an expression, and then we generate the text. 
        # here are the steps we need to take
        # convert input str to digits,
        # convert digits array to one-hot-encoded verctor
        # feed that to the model, and get its output
        # our model is seq-seq, which means, given an input seq, we 
        # generate a sequence in the output.
        # since we want to keep generating text, we do it "one character" at a time
        # feed the network one character, get the output, use that character to feed the
        # network and get the third character, and this goes on until we reach our max_length
        # so our initial prompt needs to be in a loop so we get its outputs 
        # then from there we continue with the max_length and generate text
        # prompt_digits = np.array([atoi[c] for c in prompt_str])
        hidden_state = None
        cstr=[]
        for c in prompt_str:
            idx = np.array([atoi[c]])
            idx_one_hot_vec = one_hot_encode(idx, self.input_size)
            idx_tensor = torch.from_numpy(idx_one_hot_vec).float().to(self.device)
            outputs, hidden_state = self.forward(idx_tensor, hidden_state)
            # add the prompt to our character list which will contain all the generated characters
            cstr.append(c)
            
        # the conditioning is done now, note that we didnt use the initial prompt
        # to choose a prediction, we just wanted to condition the model on the existing
        # sequence, and then when its finished, start generating the actual text
        for i in range(max_length):
            # to generate new text, we use the last character from our inital prompt
            # and kickoff the process. note that we are using cstr to grab the next 
            # character each time, we start with the last character inserted which 
            # shows the end of our initial-prompt, use that to generate a new character
            # which we decode and then store into cstr again, so that when we get the last
            # inserted character, we are basically using the very last character our model
            # generated, the hidden_state is also in sync as you can see, we use the hidden
            # state from previous generation as we are generating new characters. 
            idx = np.array([atoi[cstr[-1]]])
            idx_one_hot_vec = one_hot_encode(idx, self.input_size)
            idx_tensor = torch.from_numpy(idx_one_hot_vec).float().to(self.device)
            # note that this hidden state is comming from the last
            # character of the prompt, we just processed
            outputs, hidden_state = self.forward(idx_tensor, hidden_state)
            # now we need to calculate the probablities for each characters in the output
            preds = outputs.softmax(dim=-1)
            # now we have a list of probablities, to get good output, we grab a few of the
            # most probable characters and sample from them. 
            probs, indexes = preds.topk(5, dim=-1)
            # now that we have topk probs, lets randomly choose between them
            # before that lets convert them to numpy and use them for our job
            probs, indexes = tuple(t.cpu().squeeze().data.numpy() for t in (probs, indexes))
            # print(f'{indexes} {probs}')
            idx = np.random.choice(indexes, p=probs/probs.sum())
            # this is our next character!
            cstr.append(itoa[idx])
        print(''.join(cstr))
            
    # now lets use this with temperature sampling 
    # this technique allows us to control the randomness of the predictions by scaling the logits 
    # before applying softmax. Higher values (e.g., 1.0 or above) make the actions more random, 
    # while lower values (e.g., 0.2) make the actions more deterministic. 
    # lets create our sampler
    @torch.no_grad()
    def sample_text_with_temperature_sampling(self, prompt_str='hi', temperature=0.8, max_length=100, k=5):
        # like before we first feed the prompt to condition our model and then start generating 
        # the actual text
        hidden_state = None 
        # a list to store our generated text
        cstr = []
        if prompt_str is not None: 
            # lets add our prompt to it right now
            cstr = list(prompt_str)
            # now lets condition our model on the initial prompt
            for c in prompt_str:
                idx_tensor = torch.from_numpy(one_hot_encode(np.array([atoi[c]]), self.input_size)).float().to(self.device)
                _, hidden_state = model(idx_tensor, hidden_state)
        
        # now lets generate our own text 
        for i in range(max_length):
            # grab the last character from or character-list, if theres none use 0 as the begining
            idx_tensor = torch.from_numpy(one_hot_encode(np.array([atoi[cstr[-1] if cstr else itoa[0]]]), self.input_size)).float().to(self.device)
            outputs, hidden_state = self.forward(idx_tensor, hidden_state)
            # lets calculate the probablities , but before that we normalize the logits by the temperature
            outputs_scaled = outputs/temperature
            # now we use this scaled outputs to get the probs 
            preds = outputs_scaled.softmax(dim=-1)
            # now lets get the topk probablities and randomly choose between them 
            probs, indexes = preds.topk(k, dim=-1)
            # since we want to use numpy.random.choice, lets convert them to numpy arrays
            # also lets squeeze them so the random.choice doesnt complain about it.(they must be 1d arrays)
            probs, indexes = tuple(t.cpu().squeeze(0).data.numpy() for t in (probs, indexes))
            # choose an index, with the given probablity (we are making sure the probablities sum to 1)
            # also note the 'p=' which is for probablity (missing that migh give you headache!) 
            idx = np.random.choice(indexes, p=probs/probs.sum())
            # now this is our next character lets add it to our cstr list 
            cstr.append(itoa[idx])
        print(''.join(cstr))        

# lets test this rnn 

model = TextGenerator('rnn', input_size=char_length,output_size=char_length,hidden_size=5,num_layers=1,is_bidirectional=False, atoi=atoi, itoa=itoa)
print(model(torch.randn(size=(2,3,char_length)),None))
model.sample_text_vanialla()
model.sample_text_vanialla_with_temperature_with_topk_sampling(topk=5)
model.sample_text_better()
model.sample_text_with_temperature_sampling()
# %%
# now lets train this 
# for training we need 
# input_size, output_size, hidden_size, 
# model, loss, optimizer, scheduler
# 
# our input size is character_length, we feed our input in the form of one-hot-encoded
# vectors, obviously we cant feed decimal numbers, because that would mean our model
# needs to comeup with the exact index for our characters, 1,3,80,etc. that would be a 
# regression problem, but we want the probablities for our characters, to see given a 
# characters, what are the chances/probablity that the next one be character x for example
# so that we can later on sample from it and generate meaningful text.
# likewise, since we want the model to spit out probablities, we need as many outputs as
# we have inputs each representing as ingle character(class). 
# we could use embeddings which are way better than one-hot-encoded vectors for the input,
# but for the sake of simplicty we use once-hot-encoded vectors for now. 
input_size = char_length
output_size = char_length
hidden_size = 200
# rnn trains the slowest, and requires much more epoch to get low loss, and 
# requires gradient clipping after some seq_length k while
# lstm, trains much faster, requires less epoch, and no gradient clipping
# rnn got 4 runs to get a training loss=2.7461 and val loss=2.0000 while
#lstm gets it right the first time, its much more stable
# with seq_length=60, rnn tr/vl loss drops to 2.3894/2.0000 on the first run
# and on after 4 runs the best it gets is 2.1047/1.0000 (80 epochs)
# lstm gets 1.22/1.00 in 30 epochs!(1.19/1 in epoch 79), gru performs similarly
# to lstm but quickly diverges around epoch 20, signaling the lr might be too high so
# when decayed eevry 10 epochs(instead of 20) it got 1.37/1.00 @epoch30-all the way to 80
# gru works better than rnn, lstm works much better than gru but is heavier
rnn_type = 'lstm'
# we want sequences made of this many characters
sequence_length = 60
num_layers = 2
bidirectional = False 
direction = 2 if bidirectional else 1
dropout = 0.3
vanilla_rnn_act = 'tanh'
batch_size = 128
epochs = 80
interval = 400
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = TextGenerator(rnn_type=rnn_type, 
                      input_size=input_size, 
                      output_size=output_size, 
                      hidden_size=hidden_size,
                      num_layers=num_layers, 
                      is_bidirectional=bidirectional, 
                      dropout=dropout, 
                      act=vanilla_rnn_act,
                      atoi=atoi, 
                      itoa=itoa,
                      device=device)
# we treat this as a classification task, so each character is a calss of its own
# this allows us to make the model give us probablities for each characters
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
# each 20 step/epoch, lower the learning rate by 10 times
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# note that model.to() changes the model in place (basically any nn.Module)
# we can also do model = model.to(device), its prefectly fine.
# however remember when it comes to tensors, .to() is not inplace and instead returns a copy
# the rule of thumb is, tensors must follow strict rules when it comes to changing the
# underlying data.
model=model.to(device)
# we also need an initial hidden_state, we can initialize it with zeros, or set 
# it to None!, so we use the former way to remind ourselves that, when creating
# hidden-state, we should take bidirection and num_layers into account as well! 
# remember that, since we are using batches, our hidden_state should also have
# a batch dimension and it all should have the form (D*num_layers,Batch,hstate)
# which D is 2 if bidirection and 1 otherwise.
hidden_state = torch.zeros(size=(direction*num_layers, batch_size, hidden_size),device=device)
# since we also use lstm, lstm uses two hiddenstates, hstate and cstate. so we 
# need one more 
c_hidden_state = torch.zeros_like(hidden_state,device=device)
hidden_state = (hidden_state,c_hidden_state) if rnn_type=='lstm' else hidden_state
# it would have been easier to just set hidden_state=None and that would work
# for all variants of the rnn. anyway lets continue
# before we go we also need to have train/val splits so we can test our model
# performance. 
train_ratio = 0.8
train_length = int(dataset_length*train_ratio)
train_data = corpus_array[:train_length]
val_data = corpus_array[train_length:]

# before we go on lets calculate total batches again so we can use it in our
# trainig/val log  
def get_total_batches(dataset_length):
    return dataset_length//(batch_size*sequence_length)

train_batch_cnt = get_total_batches(len(train_data))
val_batch_cnt = get_total_batches(len(val_data))
print(f'{train_batch_cnt=} {val_batch_cnt=}')

total_train_loss = 0
total_val_loss = 0
for epoch in range(epochs):
    # since we want to preserve the hidden state across sequences during training, we can reset the hidden state 
    # at the start of each epoch, rather than at the start of each sequence (becasue we are using one big document
    # that sequences are relavnt toeachother). However, we would still need to handle the issue of trying to 
    # backpropagate through the same hidden state multiple times. 
    # One possible solution is to use truncated backpropagation through time (BPTT), where we only backpropagate 
    # through a fixed number of steps at a time.
    # this time lets use the easier one and simply use None to reset hidden state
    hidden_state = None

    # grab a batch of data
    model.train()
    for i, (data, label) in enumerate(get_batch(train_data, batch_size, sequence_length)):
        # lets prepare the data, one-hot-encode
        data = one_hot_encode(data, char_length)
        # intrestingly we dont need to one_hot_encode the labels
        # becasue we only need labels for loss calculation and 
        # cross_entropy itself manages everything, so feed the label
        # as is.
        # now lets convert them into torch tensors
        # note the .float(), our tensors need to be float for training
        data = torch.from_numpy(data).float().to(device=device)
        # note that since our label contains integer numbers, we need to convert it 
        # to long, otherwise crossentropy will error out.
        label = torch.from_numpy(label).long().to(device=device)
        # at this point you might ask, what didnt we use torch tensor instead of numpy
        # from the first place, we could! but using numpy allowed us to see how we use it
        # in torch!
        
        outputs, hidden_state = model(data, hidden_state)
        # detach hidden_state from the computation graph, why you may ask?
        # there are two reasons, one technical/implementation related and the other is conceptual.
        # from a technical point of view, by default, pytorch frees up some resources after we call 
        # .backward() to compute the gradients. This is usually what we want because it saves memory,
        # but it can lead to problems if we are trying to backpropagate through the same part of the
        # computation graph more than once.
        # in our case, we are trying to use the 'same' hidden_state across multiple iterations of our loop.
        # when we call .backward(), pytorch "forgets" some of the intermediate values used to compute 
        # hidden_state, so we cant use it for backpropagation again in the next iteration.
        # One way to fix this issue is to detach the hidden_state from its history at each step. 
        # This prevents pytorch from trying to backpropagate through the same hidden_state multiple times.
        # .detach() creates a tensor that shares the same data but does not require gradients, effectively
        # cutting off hidden_state from its history. This allows us to use the same hidden_state variable across 
        # multiple iterations without running into the error we would be seeing if we werent to do this.
        # we could also use .data attribute and get the same benifit. 
        # since lstm has two hiddenstates, we need to use both states data
        # 
        # 2. from the conceptual point of view, note that in the case of an LSTM (or any RNN really), detaching
        # the hidden state doesn't pose any issues in training.
        # why? simply becasue, if you remember, the very purpose of a hidden state in an LSTM is to maintain a kind of 
        # "memory" of the past inputs in the sequence. When we are training an LSTM (or any rnn for that matter), 
        # on a sequence, we typically don't need or want this memory to extend beyond the current sequence. 
        # By detaching the hidden state at the end of each sequence (or at each time step, depending on our usecase),
        #  we are essentially telling pytorch to treat each sequence (or time step) as independent from the others
        # during backpropagation. 
        # This allows the LSTM to update its weights based on each individual sequence, rather than trying to backpropagate
        # errors all the way back through every sequence it’s seen, which could lead to issues like exploding or vanishing
        # gradients.
        # so, while detach() does prevent the detached tensors from being modified during backpropagation, 
        # in this case, it’s actually what we want. It allows the LSTM to learn from each sequence independently, 
        # which is typically what we want when training on sequence data.
        # 
        # also note that we dont reset the hidden states at each iteration becasue our data is related. 
        # in other words,  in many sequence generation tasks, especially when the sequences are related 
        # or are part of the same text, like our case, it’s beneficial to preserve the hidden state across 
        # sequences. This allows the model to maintain some context or "memory" from one sequence to the
        # next. and hence why we are using this trick to retain the hiddenstate values from previous 
        # sequence, and use it for the next one. 
        # recap and notes:  
        # If our sequences are independent (e.g., different sentences or documents), it’s common to reset 
        # the hidden state at the start of each new sequence. This is because the context from the previous
        # sequence may not be relevant for the current sequence.
        # If our sequences are related (e.g., different parts of the same text), we might want to preserve
        # the hidden state across sequences. This allows the model to maintain some context from one sequence
        # to the next.
        # If our sequences are very long, we might run into issues with vanishing or exploding gradients. 
        # In this case, it can be beneficial to reset the hidden state periodically, even if the sequences 
        # are related.
        # 
        if rnn_type == 'lstm':
            hidden_state = tuple(state.detach() for state in hidden_state)
        else:
            hidden_state = hidden_state.data
        # calculate the loss 
        # print(f'{outputs.shape=}\n{label.shape=}')
        # if we print the outputs shape we'll see that the first two dimensions are merged
        # resulting in a (batchsize*sequence_length, char_length) so we need to flatten our
        # label so their shapes match.
        loss = criterion(outputs, label.flatten())
        total_train_loss += loss.item()
        # clear the gradients and do a backprop
        optimizer.zero_grad()
        loss.backward()
         # also since we might face gradient explosion, we might want to clip the gradients! to remedy that
         # but I havent faced one yet, with vanila rnn, this is more of a headache with rnn (lstm works well
         # by default usually but its susciptible as well, but much less compared to vanila rnn)
        #  anyway if that happened and we got nans/etc try this first, and note that gradient clipping itself
        # may hinder training if not set properly!
        # torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), max_norm=5)
        optimizer.step()
    
        if i%interval==0:
            print(f'Epoch: {epoch+1}/{epochs}, loss: {loss:.4f} lr:{scheduler.get_last_lr()[-1]:.6f}')
    
    # we decay the learning rate after each epoch
    # after version 1.1.0, we do call scheduler after optimizer, so 
    # optimizer gets the chance to work with the initial lr, before
    # its decayed using scheduler.step (prior to 1.1.0 we had to call
    # scheduler first so the lr would be ready to be consumed by optimizer
    # but not anymore!)
    scheduler.step()
    
    # do a val test 
    with torch.no_grad():
        h_state = None
        model.eval()
        total_val_loss=0
        for i, (data,label) in enumerate(get_batch(val_data, batch_size, sequence_length)):
            # lets prepare the data, one-hot-encode
            data = torch.from_numpy(one_hot_encode(data, char_length)).float().to(device=device)
            label = torch.from_numpy(label).long().to(device=device)
            outputs, final_state = model(data, h_state)
            # get the data only, we dont want this operation recorded in computaional graph!
            # since lstm has two hiddenstates, we need to use both states data
            if rnn_type == 'lstm':
                hidden_state = tuple(state.data for state in final_state)
            else:
                hidden_state = final_state.data
            # calculate the loss 
            loss = criterion(outputs, label.flatten())
            total_val_loss += loss.item()
            # calculate accuracy 

        print(f'val-loss: {total_val_loss//val_batch_cnt:.4f}')
                    
#%%
# model.sample_text_vanialla()
# model.sample_text_better(prompt_str='\n')
model.sample_text_with_temperature_sampling(prompt_str='hello',temperature=0.8)
model.sample_text_vanialla_with_temperature_with_topk_sampling(topk=5, temperature=0.8)
#%%
# ! fix this
# using beam search for text generationg 
# another improvement we can use to generate better text, is to use of a beam search strategy 
# unlike greedy decoding, which selects the most probable next token at each step, beam search
# maintains a set of the k most probable sequences at each step. This can often lead to better
# results, as it allows the model to avoid getting stuck in local optima.
# in our implementation below, beam_width is the number of sequences to keep at each step. 
# a larger beam_width will increase the chances of finding a good sequence, but it will also
# increase the computational cost.
@torch.no_grad()
def sample_text_with_beam_search(model, prompt_str='hi', max_length=100, k=5, beam_width=3):
    hidden_states = [None] * beam_width
    sequences = [[c] for c in prompt_str*beam_width]
    print(f'{sequences}')
    scores = torch.zeros(beam_width, 1).to(device)

    for i in range(max_length):
        all_candidates = []
        for j, seq in enumerate(sequences):
            idx_tensor = torch.from_numpy(one_hot_encode(np.array([atoi[seq[-1]]]), char_length)).float().to(device)
            outputs, hidden_states[j] = model(idx_tensor, hidden_states[j])
            logits = outputs.log_softmax(dim=-1)
            topk_log_probs, topk_indexes = logits.topk(k, dim=-1)
            topk_scores = scores[j] + topk_log_probs.squeeze(0)
            for l in range(k):
                all_candidates.append((topk_scores[l], topk_indexes[0][l], j, seq + [itoa[topk_indexes[0][l].item()]]))

        ordered = sorted(all_candidates, key=lambda tup:tup[0], reverse=True)
        sequences = []
        scores = torch.zeros(beam_width, 1).to(device)
        for j in range(beam_width):
            scores[j] = ordered[j][0]
            sequences.append(ordered[j][-1])
            if j != ordered[j][2]:
                hidden_states[j] = hidden_states[ordered[j][2]]

    print(''.join(sequences[0]))
    
sample_text_with_beam_search(model)
#%%
# test with temperature sampling 
# this technique allows us to control the randomness of the predictions by scaling the logits 
# before applying softmax.
def vanilla_sampling(model, prompt_array, temperature = 0.8, max_length=100):
    hidden_state = None
    char_list = []
    for i in range(max_length):
        # feed the input to the model, note that the input is a single character/digit for a single sequence
        one_hot_prompt = torch.nn.functional.one_hot(prompt_array, char_length).float()
        outputs, hidden_state = model(one_hot_prompt, hidden_state)
        hidden_state = tuple(s.data for s in (hidden_state)) if model.rnn_type =='lstm' else hidden_state.data
        # apply temperature to the logits (outputs)
        # outputs = outputs / temperature
        # we want probablities so lets use softmax 
        probs = outputs.softmax(dim=-1)
        # now we have bunch of probablities, so which one should we choose? 
        # we have fed a sequence of 1 0s, and expect the network to give us back a 
        # new character. so we have 85 classes or characters, 85 probablities to choose from
        # if we simply grab the highest probablity, that would be a greedy sampling and wouldnt
        # give us a good output. instead we use sampling, and based on the probablity of each class
        # try to sample from it, this way, higher probablity classes/characters are more likely to be
        # selected, while the least likely ones also do get a chance, albeit small, but they get their
        # chance as well and it will result in a much better output.
        # we fed our model a single sample of 1 character sequence, so we get (1,85) probablities. 
        # lets remove the first dimension, and feed it to torch.multinomial for sampling (torch.multinomial
        # wants 1d array! thats why we remove the first dimension)
        probs = probs.squeeze(0)
        # lets use this to sample, replacement=True, means the same character can be choosen again
        # basically torch.multinomial is the equivalent of numpy.random.choice (indexes, p=probs/probs.sum())
        idx = torch.multinomial(probs, 1, replacement=True)
        # now lets extract the character str and also feed this back to the model for the next round
        char_list.append(itoa[idx.item()])
        # now use this new idx/character to generate the next one 
        prompt_array = idx.unsqueeze(0) 
    print(''.join(char_list))
    
# lets create a single sequence single character/digit prompt and feed it to our network
prompt = torch.zeros(size=(1, 1),device=device).long()
vanilla_sampling(model, prompt)
#%%


#%%
# # use embedding

# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%



