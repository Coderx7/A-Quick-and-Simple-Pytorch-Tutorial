#%%
# in the name of God the most compassionate the most merciful
# !talk about pitfalls of log and exp 
# log give -inf when 0 is met
# exp fails when there are large positive numbers in a tensor (cuz exp(x) simply calculates e**x and it quickly grows)
# exp(-5) is ok, exp(-2000) is also ok, exp(-3,2,0,-1) is ok but exp(-3,-1.2, 2, 100) creates inf! for 100! 
# and when divided, mult,etc causes nans!  
# now what is usually done to avoid this, is people do exp(tensor - max(tensor))
#! Explain THIS IN DETAIl see the latter part of this section 
# 
# in this section we are going to take another approach that is more sensible and practical. 
# we are going to implement a 2003 paper by bengio etal named A neural probablistic language model 
# for several reasons that Im about to explain in a moment. 
# basically the idea of this paper is that, knowing using contexts size larger 
# than 4 or 5 in an ngram model can impose a huge overhead (and we are talking about words by the way)
# to give you an idea, our character level bigram model had a context of 1 character, 


# lets read the words, create an embedding this time. 
# the input recieves 3 input character, and perdicts the 4th
# we will create a network that we feed 3 inputs, these inputs
# will be converted into an embedding, where each character is
# represented by a vector of length x (2,3 or whatever), then 
# that embedding output is fed to another layer (tanh e.g.)
# and finally the output is fed to the last layer which outputs
# a vector of 27 length, providing the probablities for each 
# character as being the next one!
import torch 
import matplotlib.pyplot as plt 
import torch.nn.functional as F
# %matplotlib inline # acually this line doesnt work inside vscode ipython 
# it only works on jupyter notebook and infact its not even needed there 
# anymore to have figures/plots inlined! but im just writing it here for
#muscle memory of mine!!!

names = open('./data/names.txt').read().splitlines()
# lets extract the alphabet and create atoi and itoa mappings
atoi={'.':0}
atoi.update({ch:i for i,ch in enumerate(sorted(set(''.join(names))),start=1)})

itoa = {v:k for k,v in atoi.items()}
# lets test 
print(f'{atoi=}')
print(f'{itoa=}')
# now we need to create dataset of inputs with 3 characters as input
# and the next one be the label
context = ''
X=[]
Y=[]
block_size=3
for name in names[:5]:
    context = [0]*block_size
    for ch in name+'.':
        idx = atoi[ch]
        X.append(context)
        Y.append(idx)
        context = context[1:]+[idx]
        print(context)

# lets create a generator to have determinstic behavior
# we use this on randomly initialized tensors (weights, embeddings etc)
g = torch.Generator().manual_seed(255)

X=torch.tensor(X)
Y=torch.tensor(Y)

print(f'{X.shape}')
print(f'{Y.shape}')
#  now that we have our data, let us create our embedding layer.
# embedding layer is simply an array of some vectors, basically
# each word/character is given a vector of length x, e.g. 2,10,30,etc
# depending on the dataset. so if we have 5 characters, we will have 5 vectors
# the idea here is to represent more information in these 'emebedings'
# each character can now be represented better and more freely
# lets create our embedding matrix with initial random values
# lets choose embeding size of 2 , each character is represented by 2 floats!
C=torch.randn(size=(X.shape[0],2), generator=g)
print(f'{C.shape=}')
# now how can we extract a specific emebding related to a specific input you say?
idx0 = X[20][0]
idx1 = X[20][1]
idx2 = X[20][2]
print(f'\nX[20]: {X[20]}')
print(f'C[{idx0}]{C[idx0]}')
print(f'C[{idx1}]{C[idx1]}')
print(f'C[{idx2}]{C[idx2]}')
# so each input contains 3 characters, so it contains 3 indexes 
# instead of separately fetching each inputs embd we can do it in one go!
print(f'\nC[X[20]]{C[X[20]]}')
#python fetches each value from X[20] and send it to C and get a result, 
# then do it for the next and the next until all the results are gathered
# then returns the result. basically pytorch internally does a forloop for
# us !
# likewise, we can do this with all values of X and not just a single index!
print(f'C[X].shape: {C[X].shape}\nC[X][1].shape: {C[X][:1].shape}\nC[X][1]:{C[X][:1]}')
# gives us the embeddings for all inputs (basically each sample has 3 embeddings for
# each character triplet) in total our result shape would be 32(number of samples) 
# x3(number of inputs which is 3 characters(make up a single sample) x 2(embedingsize)
# )
# now that we have our embedding lets create our nn
# the embedding layer doesnt need a W, because we want to optimize it ourselves 
# its a collection of vectors that are random at first but need to learn
# proper value just like a normal weight
# the second layer is the tanh layer which needs a W1 and b1
# the third and last layer needs to output probablities for all characters
# so it needs a W2,b2 to learn the wieghts for this
# so lets go we first start with a few samples and when the code is finished
# make it work for all samples
# W1 involves embedding layer, our embeding layer dim is (#,3,2)
# so our W1 needs to have the shape (3*2, whatever_neuron_count)
W1 = torch.randn(size=(6, 100),generator=g)
b1 = torch.ones(1)
# W2 deals with W1 so it has shape(100, 27)
W2= torch.randn(size=(100, 27),generator=g)
b2= torch.ones(1)

# lets create a list of parameters for easier manipulation later on when we do backprop
parameters = [W1,b1, W2,b2, C]
# lets make them all have gradients becasue we want them to be updated in backprop!
for param in parameters:
    param.requires_grad_(True)

#%%    
# now lets go
# lets get the embeddings corrosponding to our inputs
embd = C[X[:5]]
# do a forward 
# becasue our embd has the shape (x,3,2), we need to reshape it so 
# its compatible with our weights matrix. we can use concat to stackthem as 1
# but this needs to be done for every single sample, which requires a loop and
# this is not desirable to say the least(and inefficient), we can go one step further
# and remove the loop using torch.unbind which returns separate tuples out of a tensor
# given an axis to work based off of. basically if we give a=[[1,2,3],[4,5,6],[7,8,9]] and
# do a torch.unbind(a,dim=0), it will return a tuple of 3 rows : [1,2,3], [4,5,6] 
# and finally[7,8,9]. so for us we could do 
# method 1 cat 
# embd = torch.cat(embd.unbind(dim=2))
# we can use view() to change the view so no copy, etc is done
# or we could use torch.unbind to achieve this with copy and more operations! which is not reasonable here!
# embd.view(x.size()[0], 6)
# or we could make torch infer the other dim itself incase we later on change the embeddings/input size
print(f'{embd.size()=}') # prints embd.size()=torch.Size([5, 3, 2])
embd = embd.view(embd.size()[0], -1)
# now lets do the forward 
logits = (embd@W1 + b1).tanh() 
#lets check the logits shape 
# now lets go for the secodn layer
logits = logits@W2 + b2
print(f'{logits.shape=}') # prints logits.shape=torch.Size([5, 100])
# lets calculate the propbablities the old way
probs = logits/logits.sum(dim=1, keepdim=True)
# or simply do which is basically the very same thing!
probs = torch.softmax(logits, dim=1)
# now lets calculate the loss which is the nagtive of log-likelihood
# note that we want to calculate loss for what we just fed our nn
# so we only grab those probs here to calculate loss not the whole probs
loss = -probs[torch.arange(embd.size()[0]), Y[:5]].log().mean() 
loss_all = -probs.log().mean()
print(f'negative log_prob[32,Y].mean(): {loss:.4f}')# prints  13.3888
print(f'negative log_prob.mean()(wrong): {loss_all:.4f}') # prints 14.2780
# and lets calculate them the new way!
loss_n = torch.nn.functional.cross_entropy(logits, Y[:5])
print(f'loss_crossentropy: {loss_n:.4f}') # prints 13.3888
# why use cross_entropy()? well theres no need in reinvernting the wheel each time!
# when its already available ,and we know how it works under the hood (so no educational motivation here)
# and its numerically stable compared to our version. 
# lets see why its numerically stable and how it achieves this
# and what implications it has 
#!explain 
#%%
# suppose we have 
logits = torch.tensor([-2,-3,0,5])
counts = logits.exp()
probs = counts/counts.sum()
# and all goes well and our probs are just ok 
print(probs)
# prints 
# tensor([9.0466e-04, 3.3281e-04, 6.6846e-03, 9.9208e-01])
# but lets see what happens when one of the entries are much larger than others
# this can happen during optimization where one or many entries get very large positive values
# we face issues
logits = torch.tensor([-2,-3,0,100])
counts = logits.exp()
probs = counts/counts.sum()
print(f'{counts=}')
print(f'{probs=}')
# prints 
# counts=tensor([0.1353, 0.0498, 1.0000,    inf])
# probs=tensor([0., 0., 0., nan])
# the probs creates nan! now! and it stems from counts having the inf! as you can see 
# !this happens because we run out of range for these floating point numbers that represent these counts
# basically we are taking these numbers and exp() takes e to the power of these large numbers and thats why it becomes inf
# for example in our case, e**100 is 2.6881171418161212e+43 (yes thats more than 43 digits)
# note that exp doesnt have an issue with negative numbers so this is ok (cuz the more negative they just 
# get closer to 0! and basically after some numbers it just becomes 0! (test with -2000 and see))
logits = torch.tensor([-100,-3,0,1])
counts = logits.exp()
probs = counts/counts.sum()
print(f'{probs=}')
# prints 
# probs=tensor([9.8091e-45, 1.3213e-02, 2.6539e-01, 7.2140e-01])


# but large positive numbers cause nans
# now to fix this issue what is often done is to add or subtract one arbitrary number and the result would stay the same
logits = torch.tensor([-2,-3,0,5]) 
logits2 = torch.tensor([-2,-3,0,5]) +1
logits3 = torch.tensor([-2,-3,0,5]) -3
counts = logits.exp()
counts2 = logits2.exp()
counts3 = logits3.exp()
probs = counts/counts.sum()
probs2 = counts2/counts2.sum()
probs3 = counts3/counts3.sum()
print(f'{probs=}')
print(f'{probs2=}')
print(f'{probs3=}')
# prints
# probs=tensor([9.0466e-04, 3.3281e-04, 6.6846e-03, 9.9208e-01])
# probs2=tensor([9.0466e-04, 3.3281e-04, 6.6846e-03, 9.9208e-01])
# probs3=tensor([9.0466e-04, 3.3281e-04, 6.6846e-03, 9.9208e-01])
# as you can see the result stays the same 
# and since the negative numbers are ok and its only the large positive numbers that cause an issue, 
# whats being done for example in pytorch is to take the maximum in the tensor and subtract it from 
# that maximum like this 
logits = torch.tensor([-2,-3,0,5]) - 5 
counts = logits.exp()
probs = counts/counts.sum()
print(f'{probs=}')
# prints
# probs=tensor([9.0466e-04, 3.3281e-04, 6.6846e-03, 9.9208e-01])
# even if we use larger number its still well behaved 
logits = torch.tensor([-2,-3,0,100]) - 100 
counts = logits.exp()
probs = counts/counts.sum()
print(f'{probs=}')
# prints
# probs=tensor([5.6052e-45, 1.4013e-45, 3.7835e-44, 1.0000e+00])

# and all is fine now
#%%
# now that we have our loss lets do backprop
# but before that make sure to zero out the gradients
for param in parameters():
    param.grad=None # this is more efficient than setting grad=0! 
    
loss.backward()
# update the weights 
for param in parameters:
    param.data += -0.01*param.grad
#%%
# and we see the loss is decreasing
# we can now repeat this process for some iterations and on the whole dataset
context = ''
X=[]
Y=[]
block_size=3
for name in names:
    context = [0]*block_size
    for ch in name+'.':
        idx = atoi[ch]
        X.append(context)
        Y.append(idx)
        context = context[1:]+[idx]
        # print(context)
        
X=torch.tensor(X)
Y=torch.tensor(Y)

print(f'{X.shape}')
print(f'{Y.shape}')

C=torch.randn(size=(X.shape[0],2), generator=g)
W1 = torch.randn(size=(6, 100),generator=g)
b1 = torch.ones(1)
W2= torch.randn(size=(100, 27),generator=g)
b2= torch.ones(1)
# lets create a list of parameters for easier manipulation later on when we do backprop
parameters = [W1,b1, W2,b2, C]
# lets make them all have gradients becasue we want them to be updated in backprop!
for param in parameters:
    param.requires_grad_(True)
loss = loss2 = 0
for i in range(1000):
    # get the whole datasets embeddings in one go! and dont forget to reshape!
    embds = C[X].reshape(X.shape[0],-1)
    # make sure to use @ for matrix multiplication and Not * ! 
    logits = (embds@W1 +b1).tanh()
    logits = logits@W2+b2
    # produce probablities
    probs = logits.softmax(dim=1)
    # which is equivalent to 
    # probs = logits/logits.sum(dim=1, keepdim=True)
    # now lets calculate the loss which is negative loglikelihood
    loss = F.cross_entropy(logits, Y)
    # which is equiavalent to do 
    loss2 = -probs[torch.arange(probs.shape[0]), Y].log().mean()
    # print(f'{loss.item()=:.4f}, {loss2.item()=:.4f}')
    
    # now lets backprop
    #before doing a backward lets zero-out the gradients
    for param in parameters:
        param.grad = None
    loss.backward()
    
    for param in parameters:
        # I tested it and used 0.1 and then 1.0 and noticed 1.0 works better here!! see the later section
        # for a better solution on findig the right lr
        param.data += -1.0 * param.grad
        
print(f'{loss.item()=:.4f}, {loss2.item()=:.4f}') # prints loss.item()=2.4838, loss2.item()=2.4838
#%%
#now lets see if the embeddings learned actually make sense! 
# lets visualize them, since they are 2d we can plot them  
plt.figure(figsize=(8,8))
plt.scatter(C[:,0].data, C[:,1].data, s=200)
for i in range(C.shape[0]):
    plt.text(C[i,0].item(),C[i,1].item(),s=itoa[i], ha='center', va='center', color='white')
plt.grid('minor')
    
#%%
# but a problem exists here, it seems its really slow, can we speed it up? 
# yes we can lets do this

C=torch.randn(size=(X.shape[0],2), generator=g)
W1 = torch.randn(size=(6, 100),generator=g)
b1 = torch.ones(1)
W2= torch.randn(size=(100, 27),generator=g)
b2= torch.ones(1)
# lets create a list of parameters for easier manipulation later on when we do backprop
parameters = [W1,b1, W2,b2, C]
# lets make them all have gradients becasue we want them to be updated in backprop!
for param in parameters:
    param.requires_grad_(True)
loss = loss2 = 0
print(f'{X.shape[0]=}')
batch_size=32
for i in range(200000): # iterations * batchsize must cover the whole dataset (at least once, or more)
    # lets create batches of data and run this as several minibatches
    # instead of one huge batch of all the datasets 
    # in order to gett he whole samples, we randomly create sample indexes
    idxs = torch.randint(0,X.shape[0],size=(batch_size,))
    
    # get the whole datasets embeddings in one go! and dont forget to reshape!
    embds = C[X[idxs]].reshape(batch_size,-1)
    # print(embds.shape)
    logits = (embds@W1 +b1).tanh()
    logits = logits@W2+b2
    # produce probablities
    probs = logits.softmax(dim=1)
    # which is equivalent to 
    # probs = logits/logits.sum(dim=1, keepdim=True)
    # now lets calculate the loss which is negative loglikelihood
    loss = F.cross_entropy(logits, Y[idxs])
    # which is equiavalent to do 
    loss2 = -probs[torch.arange(batch_size), Y[idxs]].log().mean()
    # print(f'{loss.item()=:.4f}, {loss2.item()=:.4f}')
    
    # now lets backprop
    #before doing a backward lets zero-out the gradients
    for param in parameters:
        param.grad = None
    loss.backward()
    
    for param in parameters:
        param.data += -0.1 * param.grad
        
print(f'{loss.item()=:.4f}, {loss2.item()=:.4f}') # right away we got better loss : loss.item()=2.1308, loss2.item()=2.1308


#%%
# theres another problem how do we comeup with a learning rate? 
# this is one of the ways we can choose a learning rate
# the idea is simple, first we search for the lowest learning rate that gives us a somewhat
# good speed at convergence (loss decreases well), then we increase it and see up to what value
# it works and doesnt diverge, we increase it by 10x each, and then when it diverged, we try
# backing off a bit and find that nice spot. then with the lower bound and upperbound found,
# we basically sample from this range and use it to train our network (how we use it may differ
# like we may use each learning rate for each batchsize, or some iteration in between or even epochs
# based on the model and dataset and task at hand. 
# but another approach is to use this method to find the best lr and use that insead for the whole 
# training (which is not ideal as emperical experiments indicate), however it can be used to start
# from a somewhat better lr and quickly improve without spending too much time with smaller lrs e.g.

# to get a range between two bounds we can use nn.linspace 
lrs = torch.linspace(0.001,1, steps=1000)
# steps specifies the stepsize from the start to the end so that it gives us 1000 values
# print(f'{lrs=}')
# prints 
# lrs=tensor([0.0010, 0.0020, 0.0030, 0.0040, 0.0050, 0.0060, 0.0070, 0.0080, 0.0090,
#         0.0100, 0.0110, 0.0120, 0.0130, 0.0140, 0.0150, 0.0160, 0.0170, 0.0180,
#         0.0190, 0.0200, 0.0210, 0.0220, 0.0230, 0.0240, 0.0250, 0.0260, 0.0270,
#         0.0280, 0.0290, 0.0300, 0.0310, 0.0320, 0.0330, 0.0340, 0.0350, 0.0360,
#         0.0370, 0.0380, 0.0390, 0.0400, 0.0410, 0.0420, 0.0430, 0.0440, 0.0450,
#         0.0460, 0.0470, 0.0480, 0.0490, 0.0500, 0.0510, 0.0520, 0.0530, 0.0540,
# ...
#  0.9280, 0.9290, 0.9300, 0.9310, 0.9320, 0.9330, 0.9340, 0.9350, 0.9360,
#         0.9370, 0.9380, 0.9390, 0.9400, 0.9410, 0.9420, 0.9430, 0.9440, 0.9450,
#         0.9460, 0.9470, 0.9480, 0.9490, 0.9500, 0.9510, 0.9520, 0.9530, 0.9540,
#         0.9550, 0.9560, 0.9570, 0.9580, 0.9590, 0.9600, 0.9610, 0.9620, 0.9630,
#         0.9640, 0.9650, 0.9660, 0.9670, 0.9680, 0.9690, 0.9700, 0.9710, 0.9720,
#         0.9730, 0.9740, 0.9750, 0.9760, 0.9770, 0.9780, 0.9790, 0.9800, 0.9810,
#         0.9820, 0.9830, 0.9840, 0.9850, 0.9860, 0.9870, 0.9880, 0.9890, 0.9900,
#         0.9910, 0.9920, 0.9930, 0.9940, 0.9950, 0.9960, 0.9970, 0.9980, 0.9990,
#         1.0000])
# but this is not really that hepful and is very wasteful, if you look at the numbers
# they are very close to eachother and basically not good choices we can do better.
# in practice, what has shown to provide good performance is the exponential decay of the
# learning rate, there are different ways to use an exponential decaying lr (like by multiplying
# it repeatedly by 0.985 for example!)
# but here we are going to use torch.linspace to aid us in this
# but then again another approach that this is used for, is to actually find out the best lr
# for that we first create a set of exponents for our learning rates and then exponentiate them
# to be used for finding out the lr. see the example before your'll undrestand especiall the plots
lrs = torch.linspace(-3, 0, 1000)
# and instead use exponential!!! here 10**-3 means 0.001 so it starts from 0.001 and goes up to
# 10**0 which is 1 basically. 
lri = 10**lrs 
print(f'{lri}=')
# prints
# tensor([0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0011,
#         0.0011, 0.0011, 0.0011, 0.0011, 0.0011, 0.0011, 0.0011, 0.0011, 0.0011,
#         0.0011, 0.0011, 0.0011, 0.0012, 0.0012, 0.0012, 0.0012, 0.0012, 0.0012,
#         0.0012, 0.0012, 0.0012, 0.0012, 0.0012, 0.0012, 0.0013, 0.0013, 0.0013,
#         0.0013, 0.0013, 0.0013, 0.0013, 0.0013, 0.0013, 0.0013, 0.0013, 0.0014,
#         0.0014, 0.0014, 0.0014, 0.0014, 0.0014, 0.0014, 0.0014, 0.0014, 0.0014,
#         0.0015, 0.0015, 0.0015, 0.0015, 0.0015, 0.0015, 0.0015, 0.0015, 0.0015,
# ... 
#  0.8297, 0.8355, 0.8412, 0.8471, 0.8530, 0.8589, 0.8648, 0.8708, 0.8769,
#         0.8830, 0.8891, 0.8953, 0.9015, 0.9077, 0.9140, 0.9204, 0.9268, 0.9332,
#         0.9397, 0.9462, 0.9528, 0.9594, 0.9660, 0.9727, 0.9795, 0.9863, 0.9931,
#         1.0000])=

#%%
C=torch.randn(size=(X.shape[0],2), generator=g)
W1 = torch.randn(size=(6, 100),generator=g)
b1 = torch.ones(1)
W2= torch.randn(size=(100, 27),generator=g)
b2= torch.ones(1)
# lets create a list of parameters for easier manipulation later on when we do backprop
parameters = [W1,b1, W2,b2, C]
# lets make them all have gradients becasue we want them to be updated in backprop!
for param in parameters:
    param.requires_grad_(True)
loss = loss2 = 0
print(f'{X.shape[0]=}')
batch_size=32
losses=[]
# notice the iteration, this is for testing and finding out about the best lr 
# so we used 1000 iteration for our 1000 lrs!
for i in range(1000): # in reallife, iterations * batchsize must cover the whole dataset (at least once, or more)
    # lets create batches of data and run this as several minibatches
    # instead of one huge batch of all the datasets 
    # in order to gett he whole samples, we randomly create sample indexes
    idxs = torch.randint(0,X.shape[0],size=(batch_size,))
    
    # get the whole datasets embeddings in one go! and dont forget to reshape!
    embds = C[X[idxs]].reshape(batch_size,-1)
    # print(embds.shape)
    logits = (embds@W1 +b1).tanh()
    logits = logits@W2+b2
    # produce probablities
    probs = logits.softmax(dim=1)
    # which is equivalent to 
    # probs = logits/logits.sum(dim=1, keepdim=True)
    # now lets calculate the loss which is negative loglikelihood
    loss = F.cross_entropy(logits, Y[idxs])
    # which is equiavalent to do 
    loss2 = -probs[torch.arange(batch_size), Y[idxs]].log().mean()
    # print(f'{loss.item()=:.4f}, {loss2.item()=:.4f}')
    
    # now lets backprop
    #before doing a backward lets zero-out the gradients
    for param in parameters:
        param.grad = None
    loss.backward()
    
    lr = lri[i]
    for param in parameters:
        param.data += -lr * param.grad
    losses.append(loss.item())
    
print(f'{loss.item()=:.4f}, {loss2.item()=:.4f}') # right away we got better loss : loss.item()=2.1308, loss2.item()=2.1308

fig, (ax1,ax2) = plt.subplots(2)
fig.suptitle('lr and loss')
ax1.plot(lri, losses)
# and this shows us basically how our learning rate just works vs loss
# in order to get the exponent and see where exactly our best lr lies we can use the exponent lrs 
ax2.plot(lrs, losses)
# the second plot shows us that at basically around lr = 0.03 and lr=0.1 is the best decrease in loss
# and good choice 

#%%
# now lets test the new found lr on the whole dataset !
C=torch.randn(size=(X.shape[0],2), generator=g)
W1 = torch.randn(size=(6, 100),generator=g)
b1 = torch.ones(1)
W2= torch.randn(size=(100, 27),generator=g)
b2= torch.ones(1)
# lets create a list of parameters for easier manipulation later on when we do backprop
parameters = [W1,b1, W2,b2, C]
# lets make them all have gradients becasue we want them to be updated in backprop!
for param in parameters:
    param.requires_grad_(True)
loss = loss2 = 0
print(f'{X.shape[0]=}')
batch_size=32
losses=[]
# we use enough iterations to cover the whole dataset 
for i in range(200000): # iterations * batchsize must cover the whole dataset (at least once, or more)
    # lets create batches of data and run this as several minibatches
    # instead of one huge batch of all the datasets 
    # in order to gett he whole samples, we randomly create sample indexes
    idxs = torch.randint(0,X.shape[0],size=(batch_size,))
    
    # get the whole datasets embeddings in one go! and dont forget to reshape!
    embds = C[X[idxs]].reshape(batch_size,-1)
    # print(embds.shape)
    logits = (embds@W1 +b1).tanh()
    logits = logits@W2+b2
    # produce probablities
    probs = logits.softmax(dim=1)
    # which is equivalent to 
    # probs = logits/logits.sum(dim=1, keepdim=True)
    # now lets calculate the loss which is negative loglikelihood
    loss = F.cross_entropy(logits, Y[idxs])
    # which is equiavalent to do 
    loss2 = -probs[torch.arange(batch_size), Y[idxs]].log().mean()
    # print(f'{loss.item()=:.4f}, {loss2.item()=:.4f}')
    
    # now lets backprop
    #before doing a backward lets zero-out the gradients
    for param in parameters:
        param.grad = None
    loss.backward()
    
    # train until we platue and then we decay the lr and continue training
    # this is the simplest form of course, and acts just as a headsup 
    lr = 0.1 if i<150000 else 0.03
    for param in parameters:
        param.data += -lr * param.grad
    
print(f'{loss.item()=:.4f}, {loss2.item()=:.4f}') # loss.item()=2.1339, loss2.item()=2.1339

#%%
# theres another issue, how do we test the generalization? we split the dataset into train, val, test
# or train, dev, test with different ratios, like 80, 10, 10, or 70, 20,10, etc (its usually best to 
# have more training example so the network can better learn and utilize the samples)

# ok, now how do we test our model now? and test it
# we simply sample from the probablity distribution the network produces
# here it is 
# for testing we simply feed the network an input, and then sample from the last layer each time
# until we face the ending token/symbol of '.' in the output
for i in range (10):
    input = [0]*block_size
    input_tensor = torch.tensor(input).reshape(1,-1)
    # print(f'{input_tensor.shape=}')
    chsr = ''
    while True:
        embd = C[input_tensor].reshape(input_tensor.shape[0],-1)
        # print(f'{embd.shape=}')
        logits = (embd@W1+b1).tanh() 
        logits = logits@W2+b2
        # get probablities 
        probs = logits.softmax(dim=1)
        # print(f'{probs.shape=}')
        idx = torch.multinomial(probs.squeeze(0), num_samples=1, replacement=True, generator=g).item()
        # print(f'{idx=}')
        chsr += itoa[idx]
        input = input[1:]+[idx]
        input_tensor = torch.tensor(input).reshape(1,-1)
        
        if idx == 0:
            print(chsr)
            chsr=''
            break
    
# %%
# so to recap 
# we learned about exp issues with large positive numbers which creates inf
# logs with 0  which creates -inf
# and operations on them causes nans!
# we learned about embeddings and their lookup stratgey
# we learned how to narrow down a decent learning rate for training using torch.linsapce(-someexponent, someexponents, steps)
# and then exponanting them to create lrs to test them and find the right exp to use
# we also noticed why we should use crossentropy, we saw that to keep numerical stability we can add 
# a constant or subtract a constant from a tensor before doing exp, and it makes it numerically stable 
# without changing the outcome, and this is why in pytorchfor example, they usually subtract from the max
# in a tensor to avoid inf issue for large numbers  
# and we also saw we can use softmax() to get 
#probablities out of logits (basically counts/counts.sum(dim=1,keepdim=True) is the definition of softmax)
# we also did mini batch training, and separated training set, dev/val sets and test sets to better train our modle
# we also learned how to sample from our language model in this section 
# we also noted that if the val and traing loss are roughly equal, it means we are underfitting
# the network is not powerful enough to memorize the dataset/overfit. there are many other reasons 
# as well but we cover them later on inshaalah