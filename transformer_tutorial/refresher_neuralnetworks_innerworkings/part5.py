#%%
# in the name of God the most compassionate the most merciful
# in this section we will reveisit our previously nn with bn
# but this time with a trick that makes it much better. 
# we are going to try and implement wavenet, which is an old
# but very intresting model by google for nlp. 
# in that paper there are couple of tricks that can be very useful for us.
# among those tricks and tips, the batching tricks are specifically important
# and learning about them allows us to get new ideas on how to implement
# more efficiently what we have in mind. 
# so first lets implement the model in a more modular way, i.e. with layers, etc
# since we covered the finer details about how they work in previous parts.
# we explain every step of the way in the comments again to help clarify 
# any ambigues detail and/or recall previous information and hopefully make
# them stick for a very long time.
# 
# what was our problem again?
# we were trying to create a model that creates new names based on an existing
# dataset of names. we saw that we could build this model as simple as a bigram
# model where it only produces the next character given a character, (though it 
# wouldnt perform well) or we could also make a neural network, and model this
# bigram model, the power would be the same as bigram by default, but it gives
# us more flexibility in improving it further by adding more capacity. 
# e.g. we could increase the context size by taking more characters, instead of only 1.
# we then added batch normalization to improve the performance. 
# we now want to add heirarchy of information to our model as the next step. 
# so lets build the model.
# 
# first import the modules we are going to use 
import random 
# bunch of typing hints so our intellisense works properly
from typing import Any
from collections.abc import Iterable

import numpy as np
import matplotlib.pyplot as plt

import torch 
import torch.nn.functional as F 



# lets set seeds for determinstic output,
torch.manual_seed(255)
# the better way is to use a dedicated generator like previous parts
# but this should suffice for now.
# g = torch.Generator().manual_seed(255)
# np.random.seed(255)
random.seed(255)

# now lets read the dataset and build our dataset
names = open('./data/names.txt').read().splitlines()
# now lets create mappings to convert characters to numbers, since our network
# works with numbers only!
# first lets extract all the unique characters in our dataset
characters = sorted(set(''.join(names)))
# add special character to our character list for signifying the start and end
# of a name, we palce it at first for ease of use
characters = ['.']+characters
atoi = {ch:v for v,ch in enumerate(characters)}
# a reverse mapping for numbers to characters to later
itoa = {id:ch for ch,id in atoi.items()}
# lets print and see if they are ok!
print(f'{characters=}') 
print(f'{atoi=}')
print(f'{itoa=}')
#
# all look ok now lets create our dataset. we are going to use more characters
# together this time, so lets say we have 'mina', what we basically want is to
# provide n consecutive characters and get the next following character. so in
# other words, given 'min' the network gives us the next character which is a.
# 'min' would be considered our input example and 'a' would be its label.
# lets specify the number of consecutive characters as our context_size
context_size = 8
# now letus create our dataset 
def create_dataset(names, context_size):
    # a list for our input samples 
    X = []
    # a list for our labels 
    Y = []
    for name in names:
        # now create the initial input which is nothing but 3 empty characters
        # as we want the network to be able to generate any name from nothing! 
        input = [atoi['.']] * context_size
        # note that our names need to start and end with '.' but as our input
        # starts from 3 empty characters (i.e. '.') the first part is handled
        # automatically and we only need to add the trailing '.' here.
        for ch in name+'.':
            idx = atoi[ch]
            # add the new input to the dataset
            X.append(input)
            Y.append(idx)
            # update the input with the new character, which means discard one
            # character from the beginning and add the new one at the end,like
            # we are sliding from left to the right
            input = input[1:] + [idx]
    
    return torch.tensor(X), torch.tensor(Y)

# lets test the dataset 
X,Y = create_dataset(names[:2], context_size)
print(f'{X=}')
print(f'{Y=}')
# looks good lets create the whole dataset
X, Y = create_dataset(names, context_size)
# train/val/test splits for better evaluation
# before that we need to shuffle the data and then split
random.shuffle(X)
# now lets split train/test/val as 80,10,10
training_count = int(0.8*len(X))
validation_count = int(0.9*len(X))
# convert to torch tensor while we are at it
train_x, train_y = create_dataset(names[:training_count], context_size)
val_x, val_y = create_dataset(names[training_count:validation_count], context_size)
test_x, test_y = create_dataset(names[validation_count:], context_size)
# lets make sure the sizes match
assert len(X) == sum(map(len,[train_x, val_x, test_x])), 'sizes must match'
print(f'samples: {sum(map(len,[train_x, val_x, test_x])):,}')

# now lets implement the layers, we want
# we need an embedding layer, 
# we need a linear layer 
# we need a batchnorm layer
# and lets make them all into a model!!!
# before thats lets elaborate on whats differnt now:
# we are going to use hierarchy of features like the one presented in the wavenet
# paper (https://arxiv.org/pdf/1609.03499.pdf), that is we are going to group
# a few consecutive features together, that for example, each two adjacent features
# are seen as one group, the next make up another groupd and so on, the result of 
# these make up the next layers output, and this goes on till we reach the last layer
# effectively creating a larger respective field of the input for the latter neurons
# so how do we do that? there are differnet ways of doing this, lets have an example
# we know that our input is sth like (32,3) initially, this 32 examples, each having
# 3 characters. we then feed this input to an embedding layer which is basically a lookup
# table kind of thing, for each character, we get an embedding vector of sth like 10 dim 
# so we basically get (32,3,10). now imagine we increase that 3 (our context_size) to 
# something larger like 8, we would then endup with an input of shape (32,8,10). so far
# so good. now previously, we would flatten this input to 32,80 and pass it to our linear
# layer for further processing and so on. which is what we did previously. now this time
# we want to incorporate feature hiearchy of some kind. we want to group some features
# together and run the calculation in parallel on all of them. like if we have an array like
# a = [1,2,3,4,5,6,7,8,9,10], we would like to create groups like the following: 
# (1,2), (3,4), (5,6), (7,8), (9,10).
# our operations (whatever they happen to be) can be executed directly on all of them
# in parallel.as you can see, the next layers would deal with the result from these groups
# the result of (g1,g2) (that is (1,2) is g1 and (3,4) result is g2), (g2,g3), (g3,g4),
# (g4,g5) will be available for the latter layer and so on, as you can see the receptive field
# for deeper layer becomes larger and they can see more in the input. now that we have this
# basic undrestanding, lets see how we can split our input into groups like this.
# there are different ways for doing this
# one appraoch is to simply use steps in a tensor and extract different groups this way, 
# sth like this for example
a = torch.randn(size=(4,8,10)) 
# we could do a[:,::2,:] which means, get every two items, and ultimatley get (4,4,10) tensor.
b = a[:,::2,:]
# we can get the other half, as a second group by using an odd step and creating different pairs
c = a[:,1::2,:]
print(f'{a.shape=}\n{b.shape=}\n{c.shape=}')
# as you can see we divided the second dim from 8 into two groups and we can concat them
# achiving our goal, (the embedding dimension is flattened for being used in operations
# and the previous dimensions act like a batch if you will)
e = torch.concat([b,c],dim=2) # 4,4,20
# but theres a much easier way to do this and that is simply reshapeing! 
# which would result in the same outcome  
print('two tensors are equal = ', (a.view(4,4,20) == e).all())
# we said we flattened the embedding dimension to ultimatley be used in operations
# what exactly do we mean by that?
# we know we want our embeddings to be optimized! and we wanted groups of characters
# so the operations could happen in parallel, so for multiplication, only the last dimension matters
# for example a tensor of shape (3,10) x (10,4) will result in a new tensor of shape
# (3x4), now if we change this to (3,6,10) x (10,4), our multiplication would go 
# just fine and we would have (3,6,4) as the result. basically those dimensions 
# are treated like a batch dim! they just get repeated. 
a = torch.arange(0,30).view(6,5)
b = torch.arange(0,20).view(5,4)
print(f'{a.shape=} {b.shape}')
print(f'{a=}')
print(f'{b=}')
c = a@b
print(f'{c.shape=}')
print(f'{c}')
# now if we add more dims to the a 
a = a.view(3,2,5)
print(f'{a.shape=}')
print(f'{a=}')
d = a@b 
print(f'{d.shape=}')
print(f'{d=}')
# now as you can see, all the previous dimensions act like a batch dimension and get repeated
# so our linear layer can carry on its job on multi-dimension tensor as well, as long as the
# last dimension and the first dimension of the tensors involved are compatible, the operation
# goes through and we get the result.
# now we dont need to change anything for the linear layer, but for batchnorm we need to account
# for this, as the 3 dim tensors mean/var need to happen for the first two dimensions both batch
# and the next dimension after it, so we take into account all the numbers involved for each sample


from abc import ABC, abstractmethod
# lets create a base abstract module class that implements two basic operations our
# modules need to have
class Module(ABC):
    
    @abstractmethod
    def parameters(self):
        raise NotImplemented
    
    def requires_grad(self):
        for p in self.parameters():
            p.requires_grad = True
    
    def zero_grad(self) ->None:
        for p in self.parameters():
            p.grad = None

class Linear(Module):
    def __init__(self, in_features, out_features, bias=True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.randn(size=(in_features, out_features)) #* 1/in_features
        self.bias = torch.zeros(size=(1, out_features)) if bias else None
    
    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        out = X@self.weight
        if self.bias is not None:
            out += self.bias
        return out
    
    def parameters(self) -> list[torch.Tensor]:
        return [self.weight, self.bias] if self.bias is not None else [self.weight]

class BatchNorm1d(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True ) -> None:
        """Calculates BatchNorm1d

        Args:
            num_features (_type_): out_features from previous layer
            eps (_type_, optional): epsilon to be used for numerical stability. Defaults to 1e-6.
            momentum (float, optional): moementum used for creating running_stats. Defaults to 0.1.
            affine (bool, optional): whether to create and calculate gamma/beta parameters.
            or simple normalize the input. Defaults to True.
            track_running_stats (bool, optional): whether to calculate running_stats. Defaults to True.
        """
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momemntum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        # parameters updated during training
        self.gamma_bn_gain = torch.ones(size=(1,num_features))
        self.beta_bn_bias = torch.zeros(size=(1,num_features))
        self.training = True
        # buffers updated for test time
        self.running_mean = torch.zeros_like(self.beta_bn_bias)
        self.running_var = torch.ones_like(self.gamma_bn_gain)
        
    def __call__(self, X) -> Any:
        # if we are in training mode 
        if self.training:
            # before we go on, make sure, we have batch>1 during training, since
            # var() of a single number is nan!
            assert X.shape[0]>1, 'single sample with traning enable is not supported.'
            
            # we are adding support for our heiarchial form I'll explained earlier
            # basically we are creating the dims along which we calculate mean/var
            # if dim=2 which is normal usage, it will be dim=(0,), if its 3 it
            # will be dim=(0,1), and so on. althouhg we only either use 2 or 3
            # for input's dimensions.
            dims = tuple(range(X.ndim-1))
            # dont forget the keepdim=True,
            # print(f'{dim=}')
            xmean = X.mean(dim=dims, keepdim=True)
            xvar = X.var(dim=dims, keepdim=True)
        else:
            xmean = self.running_mean
            xvar = self.running_var

        if self.track_running_stats and self.training:
            # dont forget to put this in a no_grad() block otherwise it will be part of 
            # optimization!
            with torch.no_grad():
                self.running_mean = (1-self.momemntum)*self.running_mean + self.momemntum*xmean
                self.running_var = (1-self.momemntum)*self.running_var + self.momemntum*xvar
        
        x_hat = (X-xmean)/torch.sqrt(xvar + self.eps)
        if self.affine:
            return self.gamma_bn_gain * x_hat + self.beta_bn_bias
        else:
            return x_hat
    
    def parameters(self):
        if self.affine:
            return [self.gamma_bn_gain, self.beta_bn_bias]
        else:
            return []

class Embedding(Module):
    def __init__(self, vocab_size, embedding_size) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.weight = torch.randn(size=(self.vocab_size, self.embedding_size))
        
    def __call__(self, X:torch.Tensor) -> torch.Tensor:
        return self.weight[X]
    
    def parameters(self) ->list[torch.Tensor]:
        return [self.weight]

# now flatten needs to account for this change as well. 
class FlattenConsecutive(Module):
    """This layer is used to flatten the input tensor.

    Args:
        Module (_type_): _description_
    """
    def __init__(self, block_size) -> None:
        super().__init__()
        self.n = block_size
        
    def __call__(self,X) -> torch.Tensor:
        batch, sequence, channels = X.shape
        # create groups of two, if n==2 e.g, which means, we multiply channels by the same factor
        # see the explanation ahead, where I elaborated on this division mechanism and what its for
        out = X.view(batch, sequence//self.n, channels*self.n)
        # if after the division, the secquence dim is 1, then treat it like a 2d tensor
        if out.shape[1] == 1:
            return out.squeeze(1)
        else:
            return out
        
    def parameters(self)->list:
        return []

class Sequential(Module):
    def __init__(self, layers:Iterable) -> None:
        super().__init__()
        self.layers = layers
    
    def __call__(self, out:torch.Tensor) -> torch.Tensor:
            # out = self.layers[0](X)
            # print(f'{self.layers[0].__class__.__name__} : {tuple(out.shape)}')
            for layer in self.layers:
                out = layer(out)
                # print(f'{layer.__class__.__name__} : {tuple(out.shape)}')
            return out
    
    def parameters(self):
        return [param for layer in self.layers for param in layer.parameters()]
    
    @property
    def training(self):
        return all([layer.training for layer in self.layers if isinstance(layer, BatchNorm1d)])
    
    @training.setter
    def training(self, is_training):
        for layer in self.layers:
            if isinstance(layer, BatchNorm1d):
                layer.training = is_training
    

class Tanh(Module):
    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        return torch.tanh(X)

    def parameters(self):
        return []

# ok now lets create our model
vocab_size = 27 
embedding_size = 10
hidden_size = 68 # set this to 68 so it has the same number of parameters as the previous network
# and we want to see if our change actually improves the loss or not

# model = Sequential([Embedding(vocab_size, embedding_size, generator=g),
#                    # now if we use context_size here, this would be our previous version
#                    FlattenConsecutive(context_size),
#                    Linear(embedding_size*context_size, hidden_size,bias=False), BatchNorm1d(hidden_size), Tanh(),
#                    Linear(hidden_size, vocab_size),
#                    ])

# but since we want to create hierarcy, instead of the whole context size, we split them 
# into smaller groups, such as 2 for example  
model = Sequential([Embedding(vocab_size, embedding_size),
                   # now that we use the group of two, we must update the next layer as well
                   FlattenConsecutive(2), Linear(embedding_size*2, hidden_size,bias=False), BatchNorm1d(hidden_size), Tanh(),
                   # now lets create more layers like this, 
                   # at each level we are basically dividing 
                   # the embedding sequences in half
                   FlattenConsecutive(2), Linear(hidden_size*2, hidden_size,bias=False), BatchNorm1d(hidden_size), Tanh(),
                   FlattenConsecutive(2), Linear(hidden_size*2, hidden_size,bias=False), BatchNorm1d(hidden_size), Tanh(),
                   Linear(hidden_size, vocab_size),
                   ])

# lets make the last layer less confident
with torch.no_grad():
    model.layers[-1].weight *= 0.1 # this is not needed, the weights seem smalls enough!
    ty,tx = torch.histogram(model.layers[-1].weight.view(-1),bins=50,density=True)
    plt.plot(tx[:-1],ty)
    plt.show()
    print(f'{model.layers[-1].__class__.__name__} weight balanced!')

model.requires_grad()
# for p in model.parameters():
#     print(p.shape)

param_count = sum(p.nelement() for p in model.parameters())
print(f'param count: {param_count:,}')

max_iter = 200_000
losses=[]
for i in range(max_iter):
    # create a batch of indexes
    batch_idx = torch.randint(0, len(train_x), size=(32,))
    # read
    output_logits = model(train_x[batch_idx])
    # calculate probs , remember, we treat each row (27 columns) as probs, so we sum over dim=1
    # and divide by their total
    # probs = output_logits.softmax(dim=1)
    # calulate loss which is negative log likelihood (always remember we only calculate loss for the
    # samples we fed our network with, (I always seem to forget this the first time!))
    # loss = -probs[range(batch_idx.shape[0]), Y[batch_idx]].log().mean()
    # or use crossentropy which we know to be better 
    # (more numerically stable (for max subtraction from logits before softmax 
    # it uses internally to get probs)
    # and more efficient becasue of operator fusion)
    loss = F.cross_entropy(output_logits, train_y[batch_idx])

    losses.append(loss.log10().item())
    
    if i%10_000==0:
        print(f'{loss=}')

    # clear the grads 
    model.zero_grad()
    # for p in model.parameters():
    #     p.grad=None

    # for p in model.parameters():
    #     print(f'{p.grad}')
    
    # run backward pass
    loss.backward()
    # break

    lr = 0.1 if i<=150_000 else 0.01
    for param in model.parameters():
        # print(f'{i}, {param.grad}')
        param.data += -lr*param.grad


# lets plot the loss
# since we have 200_000 iteration, we can average each 1000 iterations as one epoch
# ultimately resulting in 200 values which demonstrates
if len(losses)>=1000:
    plt.plot(torch.tensor(losses).view(-1,1000).mean(1).tolist())
# which as you can see we achieve a much better loss compared to previous model which only
# used context_size of 3 (of course we have to test with context_size of 8 to see how it works
# and also take into account that we have multiple layers, and nonlinearity as apposed to two
# regardless of this, we learned some new techniques and tips and also intuitions concerning
# these under the hood stuffs)
#%%
# if we try to print out the output dimensions in each layer we would get this:
# Embedding : (32, 8, 10)
# FlattenConsecutive : (32, 4, 20)
# Linear : (32, 4, 68)
# BatchNorm1d : (32, 4, 68)
# Tanh : (32, 4, 68)
# FlattenConsecutive : (32, 2, 136)
# Linear : (32, 2, 68)
# BatchNorm1d : (32, 2, 68)
# Tanh : (32, 2, 68)
# FlattenConsecutive : (32, 136)
# Linear : (32, 68)
# BatchNorm1d : (32, 68)
# Tanh : (32, 68)
# Linear : (32, 27)
# as we can see, intially we start with an 8 character/emebdding resulting in 8*10 = 80 vector
# but we dont want to work exclusively on this specific sequence of characters, so we try to
# create combinations of smaller sequences and their results. how do we do that? we start with
# splitting the sequence in half, from the initial 8 down to 4, now we have 4 sequences of twice
# the embedding size, effectively compining two embeddings (this is as if we used context_size of 2)
# (sidenote: like imagine we had large a=[1,2,3,4,5,6,7,8], we now have 4 pairs (1,2) (3,4) (5,6) (7,8), 
# this is what we are doing effectively here (splitting the sequence, and flattening the last dim
# why would we do that? becasue we dont want to multiply 80 (8*10) numbers immediately in one go!
# instead we want to first multiply the each pair with the weight matrix and learn sth and then increase 
# the context size))
# which would give us 32,2,10 and ultimately 32,20) now we have 4 sequences of such pairs that are 
# operated on in parallel. next, we take this further, we again split the remaing sequences from 4
# down to 2, and double the output, which as you can see gives us (32,2,136) which when operated on
# a linear layer, gives us the (32,2,68) as we specified the output to be 68), what this does is 
# effectively, combine the result of the previous layer(block) operation, we basically combined
# 4 characters combinations, on the next round, we split the remaining sequences in half again, 
# doubling the output, ultimately resulting in basically combing the 8 character sequence together
# (first we had 4 pairs of two characters, these got processed
# second we had 2 pairs of four characters, these got processed as well
# finally we had 1 pair of eight characters)
# the difference is, now instead of using 1 single combination for each sample, we combined each
# characters, at different levels, getting more information about relations between characters, 
# and more possibilities they can offer. (this is akin to filling the blank given a sentence, 
# like if we only had sentences like a dog jumped etc, in our dataset (emphasis on 'a dog', if 
# we always seen 'a dog', when prompted with a task like the dog jumped __, or a cat jumped __
# we wouldnt be able to fill the blank, but if somehow we could create a connection between 
# a dog, the dog, and cat and the likes, we would infer the information required to fill the 
# blank despite never having seen the dog, or a cat, etc.) the idea is to create as many 
# connections between the available information we have so that we can infer existing/explicit
# relationships between them and hopefully utilize them to our advantage)
#%%

# %%
# lets sample !
# create initial empty input 
# feed it to the model 
# get the probs
# sample from the probs 
# repeat until encountered with a ending symbol
# before that put batchnorm into test mode 
# for layer in model.layers:
#     if isinstance(layer, BatchNorm1d):
#         layer.training = False
print(f'{model.training=}')
model.training = False
# create 10 names
for i in range(10):
    # empty input 
    input = [0]*context_size
    chstr =''
    while True:
        # since we need batch dimension, we can put input in []
        # and it will add a dimension!
        output = model(torch.tensor([input]))
        probs = output.softmax(dim=1)
        training_count = torch.multinomial(probs, num_samples=1, replacement=True).item()
        chstr += itoa[training_count]
        input = input[1:] + [training_count]
        if training_count == 0:
            print(f'{chstr=}')
            break
    
# %%
# what we just implemented can be easily(and is usually) implemented using a convolution layer. 
# basically convolution layer is nothing but a linear kernel(weight) being applied on patches of input
# basically a for loop that runs a linear layer on an input in other words. but the good thing about 
# convolution isthat, that looping mechanism is built in specialied libraries, such as cuda, etc, and
# are very efficient. 
# for example basically to do what we did above, for example to process a sequence of 8 characters
# we would do:
model.training = False
logits = torch.zeros(size=(8,27))
for i in range(8):
    logits[i] = model(train_x[[7+i]])
logits.shape
# which calls our model 8 times! but a conv layer does this in one forward pass! 
# basically doing the same but efficiently!
#