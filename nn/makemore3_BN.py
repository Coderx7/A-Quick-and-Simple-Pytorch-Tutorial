# in the name of God the most compassionate the most merciful
# in this section lets improve upon our previous efforts 
# lets build some intutions about some fundamental concepts in neural networks
# vid around min 8 talks about: (logits must be close to zero!)
# Loss: how to calculate a default expected loss for the initial of our training
#       Learn what causes the initial loss to be very high and how to fix it 
#       Why we dont set all the weights to zero and why breaking symmertry (having some entropy) is desired
#       
# in order for us to see all of this unfold we use our previous imlementation here 
#%%
import sys, os
from typing import Any
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
%matplotlib inline

names = open('./names.txt').read().splitlines()
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

C=torch.randn(size=(X.shape[0],2), generator=g)
W1 = torch.randn(size=(6, 100),generator=g)
b1 = torch.ones(100)
W2= torch.randn(size=(100, 27),generator=g)
b2= torch.ones(27)
# lets create a list of parameters for easier manipulation later on when we do backprop
parameters = [W1,b1, W2,b2, C]
# lets make them all have gradients becasue we want them to be updated in backprop!
for param in parameters:
    param.requires_grad_(True)
loss = loss2 = 0
print(f'{X.shape[0]=}')
batch_size=32
losses = []
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
    # print the loss once in a while!
    if i %1000:
        print(f'{loss.item()=:.4f}, {loss2.item()=:.4f}')
        losses.append(loss.item())
        
    # now lets backprop
    #before doing a backward lets zero-out the gradients
    for param in parameters:
        param.grad = None
    loss.backward()
    
    for param in parameters:
        param.data += -0.1 * param.grad
    
    break # prints loss.item()=15.7703, loss2.item()=15.7703
print(f'{loss.item()=:.4f}, {loss2.item()=:.4f}') # loss.item()=0.2599, loss2.item()=0.2599
print(f'probs[0]: {probs[0]}')
# prints 
#probs[0]: tensor([7.7428e-08, 2.7481e-02, 6.2695e-03, 7.2357e-07, 1.7849e-07, 4.9687e-01,
#                  2.2755e-04, 1.3417e-06, 1.5987e-06, 5.3501e-04, 1.8916e-02, 3.1359e-10,
#                  7.6329e-07, 1.4384e-05, 2.3562e-06, 4.6414e-06, 3.3834e-06, 5.1464e-04,
#                  4.5700e-05, 1.0638e-03, 1.1809e-10, 6.7200e-11, 3.8618e-09, 3.8034e-01,
#                  3.4085e-06, 6.7508e-02, 1.9685e-04], grad_fn=<SelectBackward0>)


# now if we run this as you can see the loss is pretty high! 15.7703 to be exact in our case
# however, we know that by deafult, all classes are as likely and all probablities must be the same
# thus have a value of 1/27 or 0.0370! but we dont see this why? this happens because our network is wrongly very confident about and assigns
# large probablities to some or all of the classes, thus resulting a larger loss
# if we set all the weights to 0, we would get this behavior, but theres a problem
# first lets see this first hand 
#%%
# lets set the weight to 0 and see the result: 
# as we can see the probablities for all classes is now the same 1/27=0.370, and the loss
# is 3.2958, much smaller than what we used to get initially. but as we said theres an issue
#
C=torch.randn(size=(X.shape[0],2), generator=g)
W1 = torch.randn(size=(6, 100),generator=g)
b1 = torch.ones(100)
W2= torch.randn(size=(100, 27),generator=g) * 0
b2= torch.ones(27) * 0
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
    # print the loss once in a while!
    if i %10000:
        print(f'{loss.item()=:.4f}, {loss2.item()=:.4f}')
    
    # now lets backprop
    #before doing a backward lets zero-out the gradients
    for param in parameters:
        param.grad = None
    loss.backward()
    
    for param in parameters:
        param.data += -0.1 * param.grad
    
    break # prints loss.item()=3.2958, loss2.item()=3.2958
    
print(f'{loss.item()=:.4f}, {loss2.item()=:.4f}') # loss.item()=3.2958, loss2.item()=3.2958
print(f'probs[0]: {probs[0]}')
# prints 
# probs[0]: tensor([0.0370, 0.0370, 0.0370, 0.0370, 0.0370, 0.0370, 0.0370, 0.0370, 0.0370,
#         0.0370, 0.0370, 0.0370, 0.0370, 0.0370, 0.0370, 0.0370, 0.0370, 0.0370,
#         0.0370, 0.0370, 0.0370, 0.0370, 0.0370, 0.0370, 0.0370, 0.0370, 0.0370],
#        grad_fn=<SelectBackward0>)
# the issue is that need to break symetry so nearons are forced to learn different things
# when the initial values for all neurons is the same, the do the same thing with the same input
# so to speedup the learning process we'd like to have a bit of entropy and break symetry 
# thats why we dont use 0s for all the weights. setting weights to zero has other issues
# as well and its best to avoid this (this might not pose an issue in this simplistic example
# but in general and real world applications it has, (remember log issues with 0, when you
# multiply an input with 0, you get 0, especially if your biases are also 0! and down 
# the line, if you hvae log, (log(0)=-inf !)))
# so what do we do instead? we instead choose a small number close to zero to get the benifit
# of the both worlds,

#%%
# now lets make w2 and b2 have a much smaller values, one way is to simply initialize them
# with smaller numbers, a simple way would be to multiply them by a small constant
g=torch.manual_seed(255)
C=torch.randn(size=(X.shape[0],2), generator=g)
W1 = torch.randn(size=(6, 100),generator=g)
b1 = torch.ones(100)
W2= torch.randn(size=(100, 27),generator=g) * 0.01
b2= torch.ones(27) * 0.01
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
    # print the loss once in a while!
    if i %10000:
        print(f'{loss.item()=:.4f}, {loss2.item()=:.4f}')
    
    # now lets backprop
    #before doing a backward lets zero-out the gradients
    for param in parameters:
        param.grad = None
    loss.backward()
    
    for param in parameters:
        param.data += -0.1 * param.grad
    
    break # prints loss.item()=3.3146, loss2.item()=3.3146
    
print(f'{loss.item()=:.4f}, {loss2.item()=:.4f}') # loss.item()=3.3146, loss2.item()=3.3146
print(f'probs[0]: {probs[0]}')
#prints 
# probs[0]: tensor([0.0343, 0.0387, 0.0354, 0.0325, 0.0345, 0.0352, 0.0424, 0.0376, 0.0353,
#                   0.0339, 0.0368, 0.0396, 0.0369, 0.0330, 0.0396, 0.0406, 0.0389, 0.0390,
#                   0.0331, 0.0354, 0.0354, 0.0369, 0.0375, 0.0435, 0.0377, 0.0411, 0.0354],
#                   grad_fn=<SelectBackward0>)
# as you can see, we achieved a very close loss to our default one (when used 0 for W2),
# but we still retain some entropy and have broken symmetry. 
# intrestingly if remove the seed, and run this several times, we may very well see a loss
# lower than our default case, why is that?
# well its pretty simple, we would just get lucky and the network could get right for one or
# more classes, thus decreasing the overall loss evern further.
# 
# Now how does this benifit us really after all? 
# well, now the network can spend its times doing the actual optimization from the very begining
# rather than trying to compensate for the wrongly over confident network of ours and trying
# to lower the probablities and scales of the classes first! see previously network was very
# wrong, and it had to first come upw ith some sensible defaults, we provided that sensible
# default here (remember all classes are as likely to happen by default wheras previously
# this semantic was not enforced and network was overly confident that some classes were 
# more likely to happen than others (or less likely to happen than others, thus skewing
# the results for the worse)). after this change it can now do its job better and faster!
# if we plot the losses (making it bigger with log10 to see it better), we see that its 
# like a hokey stick initially but after this change, the whole loss values seem to be
# in the same range almost, as apposed to the past where initially loss values were much
# higher than the the ones at the end. 
#%%
# Now the issues with initialization is not yet over. infact if we look at the 
# previous layer which incorporates a tanh() nonlinearity, we notice some intresting
# behavior. lets plot the logits values and see what we see. in order to better see 
# the changes, I divided the tanh() layer operations into 3 separate one. 
# now before we go on, lets plot the network and learn sth new 
g=torch.manual_seed(255)
C=torch.randn(size=(X.shape[0],2), generator=g)
W1 = torch.randn(size=(6, 100),generator=g)
b1 = torch.ones(100)
W2= torch.randn(size=(100, 27),generator=g) *0.01
b2= torch.ones(27) *0.01
# lets create a list of parameters for easier manipulation later on when we do backprop
parameters = [W1,b1, W2,b2, C]
# lets make them all have gradients becasue we want them to be updated in backprop!
for param in parameters:
    param.requires_grad_(True)
loss = loss2 = 0
print(f'{X.shape[0]=}')
batch_size=32
losses=[]
for i in range(200000): # iterations * batchsize must cover the whole dataset (at least once, or more)
    # lets create batches of data and run this as several minibatches
    # instead of one huge batch of all the datasets 
    # in order to gett he whole samples, we randomly create sample indexes
    idxs = torch.randint(0,X.shape[0],size=(batch_size,))
    
    # get the whole datasets embeddings in one go! and dont forget to reshape!
    embds = C[X[idxs]].reshape(batch_size,-1)
    # print(embds.shape)
    preact = (embds@W1 +b1)
    h_output = preact.tanh()
    logits = h_output@W2+b2
    # produce probablities
    probs = logits.softmax(dim=1)
    # which is equivalent to 
    # probs = logits/logits.sum(dim=1, keepdim=True)
    # now lets calculate the loss which is negative loglikelihood
    loss = F.cross_entropy(logits, Y[idxs])
    # which is equiavalent to do 
    loss2 = -probs[torch.arange(batch_size), Y[idxs]].log().mean()
    # print the loss once in a while!
    if i %10000:
        print(f'{loss.item()=:.4f}, {loss2.item()=:.4f}')
    
    # lets make loss bigger for better visualization!
    losses.append(loss.log10().item())
    # now lets backprop
    #before doing a backward lets zero-out the gradients
    for param in parameters:
        param.grad = None
    loss.backward()
    
    for param in parameters:
        param.data += -0.1 * param.grad
    
    break

print(f'tanh outputs[0]: {h_output[0]}')
# prints 
# tanh outputs[0]: 
#          tensor([ 0.9855, -0.9916,  0.9604, -0.1495, -0.9690,  0.9938,  0.9999, -0.6229,
#                  -0.8590,  0.9285, -0.6467,  0.9943,  0.9074, -0.6842,  0.9527, -0.9176,
#                  -0.5147,  0.9518,  1.0000,  0.9999,  0.9997,  0.3540, -0.6063,  0.9547,
#                  -0.9942,  0.9956, -0.5982,  0.7840,  0.8188,  0.3558,  0.9999,  0.9999,
#                   0.1328,  0.1968, -0.9912,  0.9832,  0.6800, -0.5120,  0.9999,  0.9999,
#                   0.8716,  0.9729,  0.3500, -0.9511, -0.9439,  0.9125, -0.9999,  0.9866,
#                  -0.8578,  0.7306,  0.9988,  0.8713,  0.9511, -0.8800,  0.9990, -0.5973,
#                   0.9868, -0.2105, -0.6844, -0.7671,  1.0000,  1.0000, -0.9090,  0.9990,
#                  -0.9443,  0.8292, -0.2569,  1.0000, -0.8327,  0.7495, -0.8541,  0.8756,
#                   0.4267, -0.9174, -0.8280, -0.2208,  0.9995, -0.9977, -0.9686, -0.9995,
#                  -0.9284, -0.9577,  0.9998,  0.9354, -0.5837, -0.6537,  0.9965, -0.2443,
#                   0.9992,  0.9199,  0.9062,  0.9982,  0.9959,  0.9990,  0.9849, -0.9006,
#                  -0.9382,  0.9953, -0.4278, -0.7721], grad_fn=<SelectBackward0>)
# as you can see, there are lots of 1s and -1s which signifies the tanh is very active
# its basically is squashing many values to the 1 and -1 at the two extreme sides
# plotting these values as a histogram and visualizing it makes it more intuitive and 
# apparent whats going on and why
plt.hist(h_output.view(-1).tolist(),bins=50)
# side note this is the same as doing plt.hist(outputs.view(-1).detach().numpy(),bins=50)
# lets also plot the preactivations before the tanh is applied on them 
figure, axs = plt.subplots(2)
figure.suptitle('visualizing values after and before tanh')
axs[0].hist(h_output.view(-1).tolist(), bins=50)
axs[1].hist(preact.view(-1).tolist(), bins=50)
# as you can see, the outputs_preactivation base is very spread! it ranges from -7.5 - 10!
# which is too spread apart! as we saw previously, we want our values to be closer to 0
# this would allow the activation functions such as tanh(), not work much and dont create
# values at the very extreme ends (dont saturate the values at both ends!), basically hindering
# the network performance greatly, note that tanh in backward pass, basically shrinks the 
# gradients magnitude, making them smaller! the only way tanh doesnt shrink/kill our gradients
# is when the value is around 0 or exactly 0! (becasue recall that the backprop for tanh was :
# 1 - tanh(x)**2, so if the values are saturated at the extreme ends of 1 and -1(or very close
# to them), the grad would be (1-1)*grad = 0, basically no learning would happen at that
# point, regardless of what the grad is, we are just killing it there!)
# to further elaborate this, lets see how many neurons in that layer actually fire at the extremes
# for every single input in our batch!
plt.figure(figsize=(10,5))
plt.imshow(h_output.abs()>0.95, cmap='gray', interpolation='nearest')
# the white dots/blocks signify that the neurons at these places are very active and their output
# are at the extreme ends 1/-1, the balck ones are either in between, note that black here
# doesnt mean the neurons are dead, a dead neuron produces 0 regardless of input, so if e.g.
# in this image we have a column of black blocks, that would mean that specific neuron is dead
# it wouldnt activate for anyinput in our batch (we have 32 examples in our batch and 100 neurons)
# Important note: note how we create our condition, note that we are comparing against 0.95!
# if we instead try >0.99, well see that a lot of the blocks turn black! why? beucase any value
# other than this will be treated as black!
plt.figure(figsize=(10,6))# im changing the figsize slightly so it doesnt overwrite the previous plot!
plt.imshow(h_output.abs()>0.99, cmap='gray', interpolation='nearest')
# if we were to catch dead neurons a better way would be to look for 0 specifically!
# any way, so when it comes to squashing functions such as tanh, relu, sigmoid, etc
# we need to take extra precautions 
# now back to the issue at hand, we just observed that the values of preactivation are large
# so they cause tanh() to saturate, and thus create this issue. so we need to decrease
# their values , how do we do that? during initialization we can enfoce this as well!
#%% lets apply this enforcement here on w1 and b1 as well and see the result
g=torch.manual_seed(255)
C=torch.randn(size=(X.shape[0],2), generator=g)
W1 = torch.randn(size=(6, 100),generator=g) *0.01
b1 = torch.ones(100) *0.01
W2= torch.randn(size=(100, 27),generator=g) * 0.01
b2= torch.ones(27) * 0.01
# lets create a list of parameters for easier manipulation later on when we do backprop
parameters = [W1,b1, W2,b2, C]
# lets make them all have gradients becasue we want them to be updated in backprop!
for param in parameters:
    param.requires_grad_(True)
loss = loss2 = 0
print(f'{X.shape[0]=}')
batch_size=32
losses=[]
for i in range(200000): # iterations * batchsize must cover the whole dataset (at least once, or more)
    # lets create batches of data and run this as several minibatches
    # instead of one huge batch of all the datasets 
    # in order to gett he whole samples, we randomly create sample indexes
    idxs = torch.randint(0,X.shape[0],size=(batch_size,))
    
    # get the whole datasets embeddings in one go! and dont forget to reshape!
    embds = C[X[idxs]].reshape(batch_size,-1)
    # print(embds.shape)
    preact = embds@W1 +b1
    h_output = preact.tanh()
    logits = h_output@W2+b2
    # produce probablities
    probs = logits.softmax(dim=1)
    # which is equivalent to 
    # probs = logits/logits.sum(dim=1, keepdim=True)
    # now lets calculate the loss which is negative loglikelihood
    loss = F.cross_entropy(logits, Y[idxs])
    # which is equiavalent to do 
    loss2 = -probs[torch.arange(batch_size), Y[idxs]].log().mean()
    # print the loss once in a while!
    if i %10000:
        print(f'{loss.item()=:.4f}, {loss2.item()=:.4f}')
    
    # lets make loss bigger for better visualization!
    losses.append(loss.log10().item())
    # now lets backprop
    #before doing a backward lets zero-out the gradients
    for param in parameters:
        param.grad = None
    loss.backward()
    
    for param in parameters:
        param.data += -0.1 * param.grad
    
    # break to see the results of our initialization quickly! 
    break
# lets plot the outputs and the preactivation 
# lets see their histogram first 
fig, axs = plt.subplots(2)
axs[0].hist(h_output.view(-1).tolist(), bins=50); # semicolon means wont print the returned value!
axs[1].hist(preact.view(-1).tolist(), bins=50); # semicolon means wont print the returned value!
# now both the outputs and their preactivation counterparts are between -1,1 and 
# in a much smaller range!
# now lets visualize the neurons output for the batch as well 
plt.figure(figsize=(10,5))
# good now we see no values at the extreme ends which is what we want, the image is black 
# signifying no values in outputs is larger than 0.95 or almost 1 
plt.imshow(h_output.abs()>0.95,cmap='gray', interpolation='nearest')
#%% so now that we changed the weights, we should get a better loss 
# but also note that this is not necessarily needed for this simple example, as the 
# network may ultimately reach to the good ending as its very simple problem
# this is not the case for deeper network however.
# also we dont initialize layers like this manually, there is a rule to follow 
# that provides us with a guaranteed way of initializing each layer properly
# to understand this concept intuitvely lets have a very simply exacmple 
# imagine we have an input x and weight matrix w and we want y 
# (for now lets forget about bias and activation functions), we would have sth like this:
x = torch.randn(size=(1000,100))
w = torch.randn(size=(100,10))
y = x@w 
# now both our x and w are guassian, lets see their mean/std and also see y's as well
print(f'{x.mean()=:.4f}, std: {x.std():.4f}') # x.mean()=0.0019, std: 1.0031
print(f'{w.mean()=:.4f}, std: {w.std():.4f}') # w.mean()=0.0064, std: 1.0079
print(f'{y.mean()=:.4f}, std: {y.std():.4f}') # y.mean()=0.1327, std: 10.0168
# which basically shows the x, w having mean of 0 and std of 1, but y isnt following!
# lets plot them and see this in practice!
plt.subplot(311)
plt.hist(x.view(-1), bins=50,density=True);
plt.subplot(312)
plt.hist(w.view(-1), bins=50,density=True);
plt.subplot(313)
plt.hist(y.view(-1), bins=50,density=True);

# as you can see, the y is simply much more spread than x or w! and if we take a larger w
# it becomes evern broader at the base (std increases!)
x = torch.randn(size=(1000,100))
w = torch.randn(size=(100,10)) * 4 # imagine this to be an intermidiate layer or operation!
y = x@w 

w2 = torch.randn(size=(100,10)) * 0.2 # imagine this to be an intermidiate layer or operation!
y2 = x@w2 
# now both our x and w are guassian, lets see their mean/std and also see y's as well
print(f'{x.mean()=:.4f}, std: {x.std():.4f}') # x.mean()=0.0010, std: 1.0020
print(f'{w.mean()=:.4f}, std: {w.std():.4f}') # w.mean()=0.1039, std: 4.0100
print(f'{y.mean()=:.4f}, std: {y.std():.4f}') # y.mean()=0.2204, std: 40.1171

print(f'{w2.mean()=:.4f}, std: {w2.std():.4f}') # w2.mean()=-0.0000, std: 0.1921
print(f'{y2.mean()=:.4f}, std: {y2.std():.4f}') # y2.mean()=-0.0101, std: 1.9155

plt.subplot(311)
plt.hist(x.view(-1), bins=100,density=True,color='gray');
plt.subplot(312)
plt.hist(w.view(-1), bins=100,density=True,color='gray');
plt.subplot(313)
plt.hist(y.view(-1), bins=100,density=True,color='gray');

plt.subplot(312)
plt.hist(w2.view(-1), bins=100,density=True, color='orange');
plt.subplot(313)
plt.hist(y2.view(-1), bins=100,density=True, color='orange');


# the x stays the same, but both w and y's std has increased 4 times! 
# so as we increase the value, the values at y basically get more and more extreme values!
# likewise if we use a smaller value (coefficient like 0.2), we see that our guassian distribution
# becomes thiner and thiner, basically limitting the amount of values (ranges of values)
# or pool of values to sample from to shrink. (a broader base, means larger numbers to sample from which
# has many issues such as numerical instability caused by exp, e.g., and gradient saturation as we saw earlier) 
# so how do we choose a good value? 
# it turns out that in order to remain guassian or better said, to keep our activations throughout 
# the network (that can consist of many layers and activation functions) from expanding 
# to infinity or getting collapsed(i.e. shrink all the way to zero), and keep them wellbehaved we need to divide our weight matrix by
# (i.e. they have reasonable values throughout the network) 
# the fan_in squared (fan_in is the number of inputs to that weight matrix)
x = torch.randn(size=(1000,10))
w = torch.randn(size=(10,200)) /10**1/2
y = x@w 

print(f'mean(w):{w.mean():.4f} std(w):{w.std():.4f}') # mean(w):-0.0015 std(w):0.0492
print(f'mean(y):{y.mean():.4f} std(y):{y.std():.4f}') # mean(y): 0.0004 std(y):0.1564
fig, axs = plt.subplots(2)
fig.suptitle('mean and std after normalizing based on fan_in')
axs[0].hist(w.view(-1), bins=50); 
axs[1].hist(y.view(-1), bins=50); 
# now here, we just did whats known for linear layer, but when there are activation functions
# involved, this changes a bit, like for relu, its (2/fan_in)**0.5 , 2 beucase relu ignores
# half of the input (its max(0,x)). and is known as gain. different activation functions require 
# different gains, for relu its radical(2), for tanh is 5/3, and for linear layers its 1.
# it happens that we can also do this in the backward pass, and instead of scaling the activations
# we can scale the gradients instead, since these are the gradinets that we ultimately use to
# update our parameters. thats why in pytorch documentations, when dealing with initialization
# we can either choose 'fan_in' or 'fan_out' as modes of initialization, in practice this is
# not really that different and nearly always, fan_in is used.
# so the formula to calculate std is std = gain/radical(fan_in)
# torch.nn.init.kaiming_normal_(w, mode='fan_in',nonlinearity='relu')
# print(f'mean(w):{w.mean():.4f} std(w):{w.std():.4f}') # mean(w):-0.0012 std(w):0.0973
# axs[0].hist(w.view(-1), bins=50, color='cyan'); 

# why does activation functions such as relu and tanh require a gain? because they are contracting
# functions, they squash the input in a way. relu, just removes half of the values (those smaller than=0)
# and tanh, squashes the values in another way, (squashes at -1,1) so we need to take this 
# into account when initializing and address it accordingly.(to fight the squeezing in, we boost
# the weights a little bit (by the gain value), so we renormalize the values back to the 
# unit standard deviation) thats why we use gain!
# so we want to set our standard deviation (std) to be gain/(fan_in**0.5), which intuitivley
# means to do sth like this : 
print(f"std: {torch.randn(10000).std()=:.4f}")
# now look what happens when we multiply this by a factor like 0.2
print(f"std: {(torch.randn(10000) * 0.2).std()=:.4f}")
# this scaled down the gaussian and shrunk the std, it became the exact factor we use to 
# multiply our vector with!
# so when we were multiplying 0.01 or 0.2 with our weight matrixes before, we were in fact
# specifying their std there! which here means that when we do gain/(fan_in**0.5) or the likes
# we are actually setting the std!
# so to racap, in order to properly initialize our weights (especially the one with tanh)
# we need to do this : 
#%%
# note during tests I foundout that the fan_in of 6 for tanh doesnt just work properly!
# as you can see, it just creates a std of ~0.7 which is still not enough for solving the issue
# visualizing the histogram of output shows this clearly that the values are at the extreme ends
# signifying the need for smaller std to avoid saturation . 
# also its very important that we may very well have a guassian distribution for our w
# but during the course of training the weights shift to a nother distribution, 
# we can test this here as well and set a break at the first iteration and see the distibutions
# to be exactly guassion, and then remove the break and train the model to only see the output
# distribution diverged from the initial guassian, this is why we present sth called BN later on
# read on !
g=torch.manual_seed(255)
C=torch.randn(size=(X.shape[0],2), generator=g)
W1 = torch.randn(size=(6, 100),generator=g) * 0.01#((5/3)/6**0.5) # gain for tanh is 5/3
b1 = torch.ones(100) *0.01
W2= torch.randn(size=(100, 27),generator=g) * 0.01
b2= torch.ones(27) * 0.01
# lets create a list of parameters for easier manipulation later on when we do backprop
parameters = [W1,b1, W2,b2, C]
# lets make them all have gradients becasue we want them to be updated in backprop!
for param in parameters:
    param.requires_grad_(True)
loss = loss2 = 0
print(f'{X.shape[0]=}')
batch_size=32
losses=[]
for i in range(200000): # iterations * batchsize must cover the whole dataset (at least once, or more)
    # lets create batches of data and run this as several minibatches
    # instead of one huge batch of all the datasets 
    # in order to gett he whole samples, we randomly create sample indexes
    idxs = torch.randint(0,X.shape[0],size=(batch_size,))
    
    # get the whole datasets embeddings in one go! and dont forget to reshape!
    embds = C[X[idxs]].reshape(batch_size,-1)
    # print(embds.shape)
    preact = embds@W1 +b1
    h_output = preact.tanh()
    logits = h_output@W2+b2
    # produce probablities
    probs = logits.softmax(dim=1)
    # which is equivalent to 
    # probs = logits/logits.sum(dim=1, keepdim=True)
    # now lets calculate the loss which is negative loglikelihood
    loss = F.cross_entropy(logits, Y[idxs])
    # which is equiavalent to do 
    loss2 = -probs[torch.arange(batch_size), Y[idxs]].log().mean()
    # print the loss once in a while!
    if i %10000:
        print(f'{loss.item()=:.4f}, {loss2.item()=:.4f}')
    
    # lets make loss bigger for better visualization!
    losses.append(loss.item())
    # now lets backprop
    #before doing a backward lets zero-out the gradients
    for param in parameters:
        param.grad = None
    loss.backward()
    
    lr = 0.1 if i <100_000 else 0.01
    for param in parameters:
        param.data += -lr * param.grad
    # break
  
print(f'loss={sum(losses)/len(losses):.4f}') #prints loss=0.2547
fig, axs = plt.subplots(3)
fig.suptitle('weights and activation distributions')
axs[0].hist(preact.view(-1).tolist(), bins=50,density=True); # ; is used for suppressing the output
axs[1].hist(h_output.view(-1).tolist(), bins=50,density=True);
axs[2].hist(logits.view(-1).tolist(), bins=50,density=True);

#%%
# knowing these are good for intuition, but nowadays, initializing deep networkds dont reauire
# this maticulous approach of finetuning everything, by using normalization layers, such as 
# BatchNOrmalization, this issue is the issues of the past! BN greatly enhanced training 
# BatchNormalization is not alone, we have layer normalization, group normalization, instance
# normalization and also newer more sophesticated optimizers such as adam, rmsprop, to help 
# reduce the importance of initialization.
# so so far we noticed that we want our hidden states/layers output to be roughly guassian
# so why not trying to do so ourselves? making sure they stay guassian? this is the idea of 
# batchnormalization (the extend of this differs for each layer, how spread or how thin a 
# distribution should look like/be differs for each layer, so we take that into consideration
# in the BatchNormalization as well. lets see how to implement one)
# (we say roughly becasue, if the values are too small, tanh is basically inactve
# and if the values are too large, tanh saturates them, basically kills the gradients and
# learns nothing!)
# The batcn normalization formula consits of this: 
# calculate the mean sample wise: mu = sum(xi)/n
# calculate the std sample wise: var = (X-mu)**2/n
# normalize it all : X^ = X-mu/(var - eps)**0.5 ; eps=1e-6 so we dont divide by zero
# and finally output = alpha * X^ + beta ; so we basically can create identity if its not helping!
# side note: to get the std from variance, we simply take its square root (i.e. std=var**0.5)
# side note: the std(0) is nan!(we can calculate std on a list/tensor not a single number!)
# see torch.std(torch.tensor([1.])) 
# that would be 
print(preact.shape) # [32,100]
# averaging along the samples (batch would be )
# mu2 = preact.sum(dim=0, keepdim=True)/preact.shape[0]
# but easier way is to simply do .mean()
mu=preact.mean(dim=0, keepdim=True)# torch.Size([1, 100])
# print((mu==mu2).all())
# now lets calculate the rest
# calculate the var 
# var = preact.var(0, keepdim=True)
# eps = 1e-5
# preact_hat1 = (preact-mu)/torch.sqrt(var) # we can add eps to avoid division by zero
# but its easier to directly calculate std
std = preact.std(dim=0, keepdim=True) 
preact_hat = (preact - mu)/std
# print(torch.allclose(torch.sqrt(var),std)) #True
# print(torch.allclose(preact_hat, preact_hat1)) # True
# in order to able to control the distribution (how broad or thin it should get,etc)
# we multiply it by a gain (we just learned above previously) and plus a bias term which would be
# houtput = houtput*alpha_gain + bias
bn_gain_alpha=torch.ones_like(std); 
bias=torch.zeros_like(std)
# since we initialized our gain_alpha and bias to be 1 and zero, our std will be 1 and our mean
# will be 0, basically create a prefect guassian at the begining
result = bn_gain_alpha*preact_hat + bias 
# so now lets put this into our network!
#%%
g=torch.manual_seed(255)
C=torch.randn(size=(X.shape[0],2), generator=g)
W1 = torch.randn(size=(6, 100),generator=g) * 0.01#((5/3)/6**0.5) # gain for tanh is 5/3
b1 = torch.ones(100) *0.01
W2= torch.randn(size=(100, 27),generator=g) * 0.01
b2= torch.ones(27) * 0.01
# we have two new parameters bn_gain_alpha and bias 
# since we want the initial distribution to be unit guassian, we use ones for gain
# and zeros for bias 
bn_gain_alpha = torch.ones(size=(1, 100))
bias_beta = torch.zeros(size=(1, 100))
# and we make sure to include them in our paramers list to be optimized
# lets create a list of parameters for easier manipulation later on when we do backprop
parameters = [W1,b1, W2,b2, C, bn_gain_alpha, bias_beta]
# lets make them all have gradients becasue we want them to be updated in backprop!
for param in parameters:
    param.requires_grad_(True)
loss = loss2 = 0
print(f'{X.shape[0]=}')
batch_size=32
losses=[]
for i in range(200000): # iterations * batchsize must cover the whole dataset (at least once, or more)
    # lets create batches of data and run this as several minibatches
    # instead of one huge batch of all the datasets 
    # in order to gett he whole samples, we randomly create sample indexes
    idxs = torch.randint(0,X.shape[0],size=(batch_size,))
    
    # get the whole datasets embeddings in one go! and dont forget to reshape!
    embds = C[X[idxs]].reshape(batch_size,-1)
    # print(embds.shape)
    preact = embds@W1 +b1
    
    # calculate preact mean and std and normalize it
    # BatchNormalization
    mu = preact.mean(0, keepdim=True)
    std = preact.std(0, keepdim=True)
    preact_hat = (preact - mu)/std
    preact = bn_gain_alpha * preact_hat + bias_beta
    
    h_output = preact.tanh()
    logits = h_output@W2+b2
    # produce probablities
    probs = logits.softmax(dim=1)
    # which is equivalent to 
    # probs = logits/logits.sum(dim=1, keepdim=True)
    # now lets calculate the loss which is negative loglikelihood
    loss = F.cross_entropy(logits, Y[idxs])
    # which is equiavalent to do 
    loss2 = -probs[torch.arange(batch_size), Y[idxs]].log().mean()
    # print the loss once in a while!
    if i %10000:
        print(f'{loss.item()=:.4f}, {loss2.item()=:.4f}')
    
    # lets make loss bigger for better visualization!
    losses.append(loss.item())
    # now lets backprop
    #before doing a backward lets zero-out the gradients
    for param in parameters:
        param.grad = None
    loss.backward()
    
    lr = 0.8 if i <100_000 else 0.08
    for param in parameters:
        param.data += -lr * param.grad
    # break

print(f'loss={sum(losses)/len(losses):.4f}') #prints loss=0.2587
fig, axs = plt.subplots(3)
fig.suptitle('weights and activation distributions after BN')
axs[0].hist(preact.view(-1).tolist(), bins=50,density=True); # ; is used for suppressing the output
axs[1].hist(h_output.view(-1).tolist(), bins=50,density=True);
axs[2].hist(logits.view(-1).tolist(), bins=50,density=True);
#%%
# if we dont get improved results, it means the network is not powerful enough, we might see
# the improvement if we make it bigger (Next make it bigger and assess the changes)
# now back at the batchnormalization: 
# the intresting thing to note here is that, each sample is being affected by the samples it 
# accompanies in each batch! and it causes for the lack of better word, a jitter like effect
# in the statistics of the normalization we are doing using batchnormalization. 
# in other words, for every output in our logits (the hidden layer activation)
# it is no longer a function of the input and that neuron only, but also the whole bunch of
# samples in that batch!
# this in otherwords, means, BN actually introduces some noise in the training proceduer thus
# prevent the network from overfitting to each example exactly, because each time the mean/std
# statistics depends on the samples inside each batch (which change each time), so in a way
# it creates some kind of a regularization effect.
# in other words, this behavior creates a regularization effect and in a way it  
# kind of acts like data augmentation, BN also allows for using much larger learning rates!
#
# This was during training, we need to change this for test time! (we need to calculate the 
# mean and std on the dataset to be able to run this in tst mode or when we have only a single
# example and not a batch to calculate its mean/std! one way would be to calculate the mean/std
# of the whole dataset !
with torch.no_grad():
    embds = C[X].reshape(batch_size,-1)
    # print(embds.shape)
    preact = embds@W1 +b1
    # calculate the mean/std of the whole dataset
    dataset_mean = preact.mean(0, keepdim=True)
    dataset_std = preact.std(0, keepdim=True)
    
    print(f'dataset_mean: {dataset_mean}')
    print(f'dataset_std: {dataset_std}')
#%%
# but this is not desirable and we should be able to calculate this mean when calculating
# the mean/std in the training phase! how should we do that|? simple, we use a running mean/std
# to estimate the mean/std of the whole trainingset during training.
# we can go about calculating running mean/std with this form: 
# since we want mean=0, std=1 at the begining we initialize them accordingly 
running_mean = torch.zeros(size=(1,100))
running_std = torch.ones(size=(1,100))
# running mean would be get the current value plus a bit of the currnet mean for current batch
running_mean = 0.999*running_mean + (1-0.999)*mean 
# likewise, for running std, pay attention to the current std, and plus a bit to the current batch
running_std = 0.999*running_std + (1-0.999)*std
# thats it, now lets incorporate this into our training : 
#%%

g=torch.manual_seed(255)
C=torch.randn(size=(X.shape[0],2), generator=g)
W1 = torch.randn(size=(6, 100),generator=g) * 0.01#((5/3)/6**0.5) # gain for tanh is 5/3
b1 = torch.ones(100) *0.01
W2= torch.randn(size=(100, 27),generator=g) * 0.01
b2= torch.ones(27) * 0.01
# we have two new parameters bn_gain_alpha and bias 
# since we want the initial distribution to be unit guassian, we use ones for gain
# and zeros for bias 
bn_gain_alpha = torch.ones(size=(1, 100))
bias_beta = torch.zeros(size=(1, 100))

# lets add the running versions of mean and std, also note that we dont optimize these!
# we just calculate them outside of the computation graph! see below
running_mean = torch.zeros_like(bn_gain_alpha)
running_std = torch.ones_like(bn_gain_alpha)

# and we make sure to include them in our paramers list to be optimized
# lets create a list of parameters for easier manipulation later on when we do backprop
parameters = [W1,b1, W2,b2, C, bn_gain_alpha, bias_beta]
# lets make them all have gradients becasue we want them to be updated in backprop!
for param in parameters:
    param.requires_grad_(True)
loss = loss2 = 0
print(f'{X.shape[0]=}')
batch_size=32
losses=[]
for i in range(200000): # iterations * batchsize must cover the whole dataset (at least once, or more)
    # lets create batches of data and run this as several minibatches
    # instead of one huge batch of all the datasets 
    # in order to gett he whole samples, we randomly create sample indexes
    idxs = torch.randint(0,X.shape[0],size=(batch_size,))
    
    # get the whole datasets embeddings in one go! and dont forget to reshape!
    embds = C[X[idxs]].reshape(batch_size,-1)
    # print(embds.shape)
    preact = embds@W1 +b1
    # print(f'{b1.grad=}')
    # calculate preact mean and std and normalize it
    # BatchNormalization
    mu = preact.mean(0, keepdim=True)
    std = preact.std(0, keepdim=True)
    
    # calculate this snippet without any gradient house keeping/history etc!
    with torch.no_grad():
        # now these two values should be very close to the mean/std calculated on the whole dataset!
        # lets check these at the end
        running_mean = 0.999*running_mean + (1-0.999)* mu 
        running_std = 0.999*running_std + (1-0.999)* std
    
    preact_hat = (preact - mu)/std # we can add eps to std to prevent divide by zero!
    preact = bn_gain_alpha * preact_hat + bias_beta
    
    h_output = preact.tanh()
    logits = h_output@W2+b2
    # produce probablities
    probs = logits.softmax(dim=1)
    # which is equivalent to 
    # probs = logits/logits.sum(dim=1, keepdim=True)
    # now lets calculate the loss which is negative loglikelihood
    loss = F.cross_entropy(logits, Y[idxs])
    # which is equiavalent to do 
    loss2 = -probs[torch.arange(batch_size), Y[idxs]].log().mean()
    # print the loss once in a while!
    if i %10000:
        print(f'{loss.item()=:.4f}, {loss2.item()=:.4f}')
    
    # lets make loss bigger for better visualization!
    losses.append(loss.item())
    # now lets backprop
    #before doing a backward lets zero-out the gradients
    for param in parameters:
        param.grad = None
    loss.backward()
        
    lr = 0.8 if i <100_000 else 0.08
    for param in parameters:
        param.data += -lr * param.grad
    # break

print(f'loss={sum(losses)/len(losses):.4f}') #prints loss=0.2612
fig, axs = plt.subplots(3)
fig.suptitle('weights and activation distributions after BN')
axs[0].hist(preact.view(-1).tolist(), bins=50,density=True); # ; is used for suppressing the output
axs[1].hist(h_output.view(-1).tolist(), bins=50,density=True);
axs[2].hist(logits.view(-1).tolist(), bins=50,density=True);

with torch.no_grad():
    embds = C[X].reshape(batch_size,-1)
    # print(embds.shape)
    preact = embds@W1 +b1
    # calculate the mean/std of the whole dataset
    dataset_mean = preact.mean(0, keepdim=True)
    dataset_std = preact.std(0, keepdim=True)
    
    print(f'dataset_mean: {dataset_mean}')
    print(f'dataset_std: {dataset_std}')

print(f'running mean: {running_mean}')
print(f'running std:  {running_std}')
# 
# also note that the bias is sperious when using batch norm?why? becasue if you look closely
# you'll see that we take preact means right after the bias term is added, and then we 
# subtract the preact from the mean, removing any effect the addition of b had!
# in fact if we see the b's gradient we see that its zero!, and batchnormalizations own bias
# is in charge of what b1 was doing!
# so when we use batchnormalization, we remove the previous layer's bias, becasue its wasteful!
# and doesnt have any effect!
# to recap: 
# we use batchnormalization to control the statistcs of the activations in layers in our nn
# we usually use batchnormalization after layers where theres a multiplication (like linear layers
# or convolutional layers, etc). 
# batchnormalization internally, has parameters for gains and bias that are trained in backpropagation
# it also has two buffers (variables that are not trained in the network) for calculating the
# running mean/std for being used at test time.
# what batchnorm does is it takes the incoming batch, and tries to center that batch around
# unit mean/std and then its offseting and scaling it by the bias/gain that it learned so far
# and on top of that its keeping track of the running mean/std (with momentum) of the samples in the dataset
# so it can be used for inference time and there is no need for calculating the whole dataset
# mean/std in a separate operation! and also this allows us to forward single samples at test times
# also note that calculating the gain and bias is very important for test time as well
# its not just the mean/std purely, the gain matters alot as well
# note that momentum part is where we specify how much attention we want to give to the batch mean
# thats the part where we do a (1-0.999) which means our momentum here is 0.001! 
# if you look at pytorch implementation, youll see the default value is 0.1! 
# (and it swaps the order, meaning it does (1-momentum)*x^ + momentum*xt)
# where it calls the x^ the estimated value (i.e. running_mean for example, 
# and xt the new observed value (i.e. mean for example) but its the same thing. 
# how to choose a value for momentum? 
# basically when the batchsize is large, it means the mean/std is roughly the same for all batches
# so this means, we can use a larger momentum here(like 0.1 or more),but if the batchsize is small
# such as 32 in our case, the mean/std might take slightly different values, becasue we are only
# using 32 samples to calculate the mean/std, so they may change a lot , so using a mometum like 0.1
# might not allow this to settle down, and we reach the actual mean/std of the dataset, 
# and it might make running mean/std to thrash too much during training and do not converge!
#  also affine, keyword in pytroch implementation, refers to the gain and bias being available
# or not! which by default should always be the case! and track_running_stat specifies the 
# calculation of running_mean/std, by default its True and calculates the running_mean/std
# you can set this to False, and calculate the mean/std in a separate process if you want
# like how we did earlier, the choice is ours!
# # the usage of batchnormaliztion today seems to be discouraged becasue of the coupling it 
# does with the batch during training!
# # lets do a recap but this time implementation wise (lets create everything modular now!)
# and run some further tests
#%%
# lets create the layers individually
class Linear:
    def __init__(self, in_features, out_features, bias=True) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = bias
        self.weight = torch.randn(size=(in_features, out_features)) *1./ in_features**0.5
        self.bias = torch.zeros(size=(1, out_features)) if bias else None
    
    def __call__(self, X) -> torch.Tensor:
        output =  X @ self.weight 
        # self.out is for ease of use during plotting later on!
        self.out = output + self.bias if self.has_bias else output
        return self.out
    
    def parameters(self) -> list[torch.tensor]:
        return [self.weight, self.bias] if self.has_bias else [self.weight]

    def zero_grad(self):
        for p in self.parameters():
            p.grad=None

class BatchNorm1d:
    def __init__(self, out_features, track_running_stat=True, momentum=0.1, eps=1e-6, is_training=True) -> None:
        self.out_features = out_features
        self.track_running_stats = track_running_stat
        self.momentum = momentum
        self.eps = eps
        # check to whether use the running mean/std for test or use mean/std for batch in training
        self.is_training = is_training
        # buffers to keep running mean/std (one mean/var for each neuron basically!)
        self.running_mean = torch.zeros(size=(1, out_features))
        self.running_var = torch.ones(size=(1, out_features))
        
        # learnable parameters gamma/beta (gain and bias)
        # gain_gama for specifying the std of the distribution as the network trains(e.g. how broad or thin it should be)
        self.gamma = torch.ones(size=(1,out_features))
        # bias_beta for specifying how much to the sides the distribution should move to (e.g. where on the x axis it should be!)
        self.beta = torch.zeros(size=(1,out_features))
        
    def __call__(self, X:torch.Tensor) -> torch.Tensor:
        # note that at test time we may have 1 samples, and var(1) is nan!
        # so its crucial to first check if we are in training more or not
        # and then go on for calculating mean/var or using their estimations
        if self.is_training:
            mean = X.mean(dim=0, keepdim=True)
            var = X.var(dim=0, keepdim=True)
        else: 
            mean = self.running_mean
            var = self.running_var

        # calculate the running mean/var outside of computation graph
        if self.track_running_stats:
            with torch.no_grad():
                self.running_mean = (1-self.momentum)*self.running_mean + self.momentum* mean
                self.running_var = (1-self.momentum)* self.running_var + self.momentum*var
        
        X_hat = (X-mean)/torch.sqrt(var+self.eps)
        # for later ease of use when dealing with plots and stuff like that!
        self.out = self.gamma * X_hat + self.beta
        return self.out
    
    def parameters(self)-> list[torch.tensor]:
        return [self.gamma, self.beta]
    
    def zero_grad(self)->None:
        for p in self.parameters():
            p.grad = None

class Tanh:
    def __call__(self,X:torch.Tensor) -> torch.Tensor:
        # again for ease of use when doing plots later on
        self.out = torch.tanh(X)
        return self.out
    
    def parameters(self)->list:
        return []
    
    def zero_grad(self):
        pass

class Embedding:
    def __init__(self, in_features: int, embedding_dim: int) -> None:
        self.in_features = in_features
        self.embedding_dim = embedding_dim
        # embedding is basically a tensor with shape of [N, embedding_dim]
        # its just a lookup table! for our vocabs, basically it gives
        # one embedding per vocab(our characters are represented by 
        # a vector of two instead of a single number to capture
        # some more underlying relationships with other characters, etc)
        self.Embeddings = torch.randn(size=(in_features, embedding_dim))
    
    def __call__(self, X:torch.Tensor) -> torch.Tensor:
        return self.Embeddings[X]
    
    def parameters(self)-> list[torch.tensor]:
        return [self.Embeddings]
    
    def zero_grad(self)->None:
        for p in self.parameters():
            p.grad = None

class Flatten:
    def __init__(self, debug=False) -> None:
        self.debug = debug
        
    def __call__(self, X:torch.Tensor) -> torch.Tensor:
        out =  X.view(X.shape[0], -1)
        if self.debug:
            print(f'input.shape : {X.shape}')
            print(f'flattened input: {out.shape}')
        return out
    
    def parameters(self)->list[torch.tensor]:
        return []

    def zero_grad(self)->None:
        pass


# now lets create our exmaple here again
names = open('./names.txt').read().splitlines()
# now lets create itoa and atoi dictionaries 
atoi={'.':0}
atoi.update({ch:i for i,ch in enumerate(sorted(set(''.join(names))),1)})
itoa = {v:k for k,v in atoi.items()}
# print(atoi)
# print(itoa)
# now lets create the datasets 

# lets create train, dev and test splits here
import random
def build_dataset(names):
    X = []
    Y = []
    block_size = 3
    for name in names:
        context = [atoi['.']] * block_size
        for ch in name+'.':
            idx = atoi[ch]
            X.append(context)
            Y.append(idx)
            context = context[1:]+[idx]

    X_tensor = torch.tensor(X)
    Y_tensor = torch.tensor(Y)
    print(X_tensor.shape, Y_tensor.shape)
    return X_tensor, Y_tensor

random.seed(42)
# shuffle the names around
random.shuffle(names)
training_count = int(0.8*len(names))
validation_count = int(0.9*len(names))
X_tensor_tr, Y_tensor_tr = build_dataset(names[:training_count])
X_tensor_val, Y_tensor_val = build_dataset(names[training_count:validation_count])
X_tensor_test, Y_tensor_test = build_dataset(names[validation_count:])


embedding_size = 10
vocab_size = 27 # 27 characters 

torch.manual_seed(255)
torch.set_anomaly_enabled(True)

layers:list[Embedding|Linear|Flatten] = [
          Embedding(vocab_size, embedding_size),  
          # there are 3 numbers in each sample, so the output of our embedding layer
          # would be (...,3,2) so we need to flatten the last dimension so it can be
          # used by and fed to the next linear layer 
          Flatten(True),
          Linear(embedding_size * block_size, 100), Tanh(), 
          Linear(100, 100), Tanh(), 
          Linear(100, 100), Tanh(),
          Linear(100, 100), Tanh(),
          Linear(100, 100), Tanh(),
          Linear(100, vocab_size) ]

parameters:torch.tensor = [p for l in layers for p in l.parameters()]
n_parameters = sum(p.nelement() for p in parameters)
print(f'{n_parameters=:,}')

# remember this is not part of optimization so it must to take part in computational graph!
with torch.no_grad():
    # now we should be able to initialize all the layers however we want!
    # lets make the last layer less confident
    layers[-1].weight *= 0.1
    # grab all the layers except the last one that doesnt have a tanh!
    # and apply the corrosponding gain!
    for layer in layers[:-1]:
        if isinstance(layer, Linear):
            # initialize its weight with tanh gain
            # note that initially we simply initialized the linear layers 
            # as a 1/fan_in**0.5, but then here we increased its gain by 5/3
            # which is the exact number suggested when a tanh() nonlinearity
            # is used after a linear layer. now looking at the histogram
            # plot down below, (run this once (put a break!), visualize the plot
            # then comeback and change it and see the plot again with gain=1!)
            # youll see the initial layers are saturated, but later layers not so much
            # the std is roughly the same for all the layers, now remove the comment below
            # and apply the gain =1 and see the plot again, you'll see that the saturation 
            # has decreased drastically compared to the previous state where gain of 5/3 was 
            # used, however, its clear that the initial layers seem to have good spread of values
            # they seems to be fine, but as the layers get deeper and deeper, their std is
            # shrinking more and more (the bell curve gets thiner and thiner until it collapses), 
            # so we see that the use of the right gain (i.e. 5/3) kept the std of all the layers
            # roughly the same and prevented them from collapsing. 
            # Also note that, if we were to only use the linear layers, the gain of 1 would be ok!
            # however, its only after the use of tanh() nonlinearity that we need  to apply the new 
            # gain. why? becasue tanh() is a squashing function, and and basically what they do
            # is that they take a distribution and squashe it, therefore arises the need for
            # componsating this squashing force by a gain of 5/3, so it counters that force
            # by expanding it and thus keeps the distribution roughly well behaved 
            # its important to note that 5/3 is just a number that works well, a higher number
            # can e.g. retain the std much better, but also result in a greater ratio of saturated
            # neurons. (basically higher gain would result in more saturation which in turn means
            # lots of dead neurons/ impeded learning rate, wasted network capacity! 
            # so 5/3 is a good number for a series of linear layer with tanh(), it keeps the std
            # roughly the same(stabalizes the std), at a reasonable point (few saturated nerons))
            # test the values below and see the results
            # Also note that too small of a gain, also kills the activations,
            # (the activation shrink to zero) but also gradients as well(collapses the std). 
            # basically when the std collapses or gets close to zero, the pool of values 
            # for the layers get very limited obviously, they become very close to zero, 
            # and we know that functions like tanh, basically dont do much for values close 
            # to zero, they are basically inactive at that point. (use tanh(0.1), tanh(0.01)
            # and the likes and you see it just passes them trough!) and for gradients it means
            # basically they are 0 or very close to it, they are too weak at that point to convey
            # anything back and have any meaningful effect! if at all) so its very bad!
            #
            # now if we remove the tanh(), and only use the linear layers instead
            # we notice that activations become more defuse (as layers get deeper and deeper
            # ) as the std is expanded more rapidly,
            # becasue the std is expanded by a larger gain (there is not squashing done by tanh
            # anymore, so the std gets expanded especially for deeper layers as the effect is 
            # magnified)
            #
            # more over, the number of saturated neurons also drastically increases becasue
            # more defuse std, means larger numbers,( our check >0.97), whatever that means
            # for our network (basically multiple linear layer doesnt provide any higher abstractions
            # so its useless anyway, but to get the idea we created this example so yeah!)
            # and for the gradients, 
            # the gradients for the deeper layers become weaker and weaker on the other hand.
            # we see that the std shrinks for the gradients as the layer gets deeper, the
            # std goes towards collapsing! (ealier layer are more defuse, have broader range to
            # choose from, and thus pack a biger signal strength, but as the layers get deeper
            # the std collapses, and range of values become very limited, close to zero, dimnishing
            # their signal stregnth)
            # 
            # 
            # layer.weight *= 0.5 
            # layer.weight *= 1
            # layer.weight *= 3
            layer.weight *= 5/3

for param in parameters:
    param.requires_grad = True
    
# Now lets run a training loop
max_iter = 200_000
batch_size = 32 
losses=[]
update_ratios=[]
for i in range (max_iter):
    # grab a list of idx for our batch 
    idxs = torch.randint(0, X_tensor_tr.shape[0], size=(batch_size,))
    # do a forward pass
    x = X_tensor_tr[idxs]
    for layer in layers:
        x = layer(x)
    
    loss = F.cross_entropy(x, Y_tensor_tr[idxs])
    
    # zero out the previous grads to avoid gradient accumulation
    for layer in layers:
        if isinstance(layer, Tanh) or isinstance(layer, Linear):
            layer.out.retain_grad()
        # else:
        #     layer.zero_grad()
    
    for layer in layers:
        layer.zero_grad()
    
    # for p in parameters:
    #     p.grad = None

    # do a backward and caculate fresh gradients wrt new batch
    loss.backward()
    
    # now optimize the weights and biases 
    lr = 0.1 if i<10000 else 0.01
    for p in parameters:
        p.data += -lr*p.grad
    
    if i%10_000:        
        print(f'{loss=:.4f}') 
    
    if i%10_000==0:
        losses.append(loss.log10().item())
    
    with torch.no_grad():
        # basically we are comparing the update with the current values of the parameters
        # when we use log10, we are basically trying to show the exponents here for better
        # visualization
        update_ratio = [(lr*p.grad.std()/p.data.std()).log10().item() for p in parameters]
        update_ratios.append(update_ratio)
        
    if i>=1000:
        break

print(f'iter: {i} loss:{loss.item():.4f}') # iter: 1000 loss:2.2748

legends=[]
plt.figure(figsize=(20,4))
for i,layer in enumerate(layers[:-1]):
    # tanh is used becasue it has a finite range of -1,1 and visualizing it is easy
    if isinstance(layer, Tanh):
    # if isinstance(layer, Linear): # we test this when tanh layers are commented out
        output = layer.out
        layer_name = layer.__class__.__name__
        mean = output.mean()
        std = output.std()
        saturated = (output.abs()>0.97).float().mean()*100
        print(f'layer {i} ({layer_name:10}) mean: {mean:.4f} std: {std:.4f} saturated: {saturated:.2f}%')
        # now lets plot these , we use histogram to get the values 
        ty,tx = torch.histogram(output, density=True)
        plt.plot(tx[:-1].detach(), ty.detach())
        legends.append(f'layer{i}({layer_name})')
        
plt.legend(legends)
plt.title('activation distribution')
#%%
# we can do the same exact thing with the gradients instead of activations 
# what we are seeing here is that all the gradients for all the layers, roughly have 
# the same magnitude, (the gradients are roughly the same for every layer)
# but if we used the gain of 1, notice that the gradients of later layers are smaller
# than the gradients of the earlier layers,suggesting that as the layers get deeper
# their gradient strength gets weaker and weaker (use gain of 0.5 and see the result),
# and thus using a gain of 5/3 help us in equalizing the gradient strength for 
# the whole network. 
plt.figure(figsize=(20,5)) # the slight change in figuresize is to force matplotlib to not overwrite existing one and create one anew each time
legends= []
for i, layer in enumerate(layers[:-1]):
    if isinstance(layer, Tanh):
    # if isinstance(layer, Linear):
        output_grad = layer.out.grad
        layer_name = layer.__class__.__name__
        mean = output_grad.mean()
        std = output_grad.std()
        print(f'layer {i} ({layer_name:10}) mean: {mean:.4f} std: {std:.4f}')
        # now lets draw the histogram 
        hy,hx = torch.histogram(output_grad, density=True)
        # plot the points! note that plt doesnt know about torch tensors, so we give them numpy()
        # (.detach(), creates a numpy() copy of our tensor whcih matplotlib can use)
        plt.plot(hx.detach()[:-1], hy.detach(),)
        legends.append(layer_name)
plt.legend(legends)
plt.title('gradient distribution')
#%%
# now let us also visualize the parameters 
# one indicators that we can use to see if everything is ok or not is the ratio of 
# gradients over the data, if this number is large, then we are in trouble(especially if we
# use simple gradient descent update), why?
# lets look at the ratios and std, and see what they mean. 
# basically this information tells us whether all the layers are training at the same speed
# or not. when lets say for example our last layer std is twice as large as the layers
# before it (previous layers for example all had around 0.0002 but our last layer had
# std=0.002), this means, our last layer is training 10 times faster than the previous layers
# of course as we train more, the network tries to fix this issue and we infact can 
# see this if we train our network for 1000 iterations and then break (do this now!)
# well it happens that the gradient to data ratio is not really that important after all,
# whats more important is the update ratio! (of course they are important and they dont need
# to have whacky ratios otherwise that signals something is very wrong, but in general, a bit
# of difference is fine and network can manage to sort things out, unless it doesnt and now
# you know how to spot the issue regarding this!) 
# anyway, lets talk about update ratios now! jump to the next section below!
legends = []
plt.figure(figsize=(20,4)) # the slight change in figuresize is to force matplotlib to not overwrite existing one and create one anew each time
for i,p in enumerate(parameters):
    grad = p.grad
    shape = tuple(p.shape) # we could also write p.data.numpy().shape to get the pure shape but this is easier and less typing!
    if shape[0] > 1: # only plot weights (ignore biases)
        print(f'weight {shape} mean: {p.mean():.6f} std: {p.std():.6f} grad/data ratio: {grad.std()/p.std():.2f}')
        hy,hx = torch.histogram(grad)
        plt.plot(hx[:-1].detach(), hy.detach())
        legends.append(f'{i} {shape}')
plt.legend(legends)
plt.title('parameters statistics')

#%% 
# now lets plot the update ratios !
# first create new figure with different size! 
# when we plot this information, we see that they evolve overtime, initially they have 
# some values and they stablize as the network is trained,
# and then we are plotting a line, where roughly speaking, these updateratio values should
# be, and that value is roughly -3 here. basically what this means is that, there are some
# values in a tensor for example, and the update ratio, must be around 1/1000ths of the 
# the magnitude of values in that tensor (i.e. the update ratio must be 1/1000th of the data)
# so for example if the log10() of an update ratio for a parameter was -1, it means the 
# update ratio for that parameter is pretty high, the parameters are being updated alot
# one thing to note is the pink like (the last layer) where its an outlier compared to others
# and its because, the last layer has been artificially changed to stay small so softmax layer
# doesnt produce high confidence result at the begining, if we go back we see that we are 
# actually shrinking the last layers weight std for this reason (*= -0.1), 
# this made the last layers values really small, so the updates are larger to compensate 
# that but ultimately it stabalizes as you can see, but still different from others.
# thats why the rate of change for this layer and others are different
# so we our update ratios to be roughly -3 here, anything lower than -3 (like -3.5, -4, etc)
# means the parameters are not training fast enough. 
# we can test this with a smaller learning rate, (tets this with lr=0.001 now), and we notice
# that the update ratio drop to -6,-5, etc, signifying the slow training, and updates being
# way smaller than what they need to be (basically showing the size of updates are 10000/100000 times in 
# in magnitude to the size of numbers in that tensor) thus a symtomp of trainingto slowly
# and also our plot shows that we are using a bit higher learning rate, becasue we are
# around -2.5, but overall, its a good start, as everything looks well behaved.
# this plot allows us to see if something is not right/is miscalibrated, pretty quickly.
# like for example try messing the inital gains and see what happens, (remove the linear layers 
# fan_in fix, and run the test, we see that no only previous plots scream the issue
# but also here, the uprate ratios are not in sync, they are all over the place, signifying
# they are trained with different speeds, the earlier layers have huge update ratios, while
# latter layers have drastically smaller updates, resulting in a complete mess!)
# anything larger than that (-1., -1.5)
# are very large values, indicating the layers at which this update ratio is happening is
# training much faster compared to others. likewise, values smaller than -3, means
# we are training slowly for that parameter/layer
legends =[]
plt.figure(figsize=(20,5))
for i,p in enumerate(parameters):
    shape = tuple(p.shape)
    if shape[0]>1:
        # use only weights
        # plot!
        plt.plot([update_ratios[j][i] for j in range(len(update_ratios))])
        legends.append(f'param {i}')
# draw a line from 0 to the length of update ratios, to show where the ratios should ultimately reside     
# the ratios should be ~1e-3; indicate on plot
plt.plot([0, len(update_ratios)],[-3,-3], 'k') 
plt.legend(legends)

#%%
#
# now we saw that if we have an mlp (stack of linear layers), how we can calibrate it
# so everything looks good, however as we saw, its a tedious task, here lets use BatchNorm
# and see all the stats and plots look fine, even if we mess up the gain they comeout fine!
# 

layers:list[Embedding|Linear|Flatten] = [
          Embedding(vocab_size, embedding_size),  
          # there are 3 numbers in each sample, so the output of our embedding layer
          # would be (...,3,2) so we need to flatten the last dimension so it can be
          # used by and fed to the next linear layer 
          Flatten(True),
          Linear(embedding_size * block_size, 100), BatchNorm1d(100), Tanh(), 
          Linear(100, 100),                         BatchNorm1d(100), Tanh(), 
          Linear(100, 100),                         BatchNorm1d(100), Tanh(),
          Linear(100, 100),                         BatchNorm1d(100), Tanh(),
          Linear(100, 100),                         BatchNorm1d(100), Tanh(),
          Linear(100, vocab_size),  
          # !see the comment below when initializing the weights
          # BatchNorm1d(vocab_size)
          ]

parameters:torch.tensor = [p for l in layers for p in l.parameters()]
n_parameters = sum(p.nelement() for p in parameters)
print(f'{n_parameters=:,}')

# remember this is not part of optimization so it must to take part in computational graph!
with torch.no_grad():
    # !now when we use batchnormalization, we can use it before softmax as well
    # !and it shouldnt cause much issues (thoug i wouldnt do it (see last plot)), but since this time its the gamma that
    # !specifies the std of the output distribution, we use gamma (we dont have weight
    # !forthe last layer now, our last layer is batchnorm not linear!)
    layers[-1].weight *= 0.1
    # layers[-1].gamma *= 0.1
    
    # grab all the layers except the last one that doesnt have a tanh!
    # and apply the corrosponding gain!
    for layer in layers[:-1]:
        if isinstance(layer, Linear):
            # initialize its weight with tanh gain
            # changing gain here does have an impact on the output eventhough we use BN now
            # try 3 , 2 for example and see the last plot and read its comments
            # layer.weight *= 0.5 
            # layer.weight *= 1
            # layer.weight *= 3
            layer.weight *= 5/3

for param in parameters:
    param.requires_grad = True
    
# Now lets run a training loop
max_iter = 200_000
batch_size = 32 
losses=[]
update_ratios=[]
for i in range (max_iter):
    # grab a list of idx for our batch 
    idxs = torch.randint(0, X_tensor_tr.shape[0], size=(batch_size,))
    # do a forward pass
    x = X_tensor_tr[idxs]
    for layer in layers:
        x = layer(x)
    
    loss = F.cross_entropy(x, Y_tensor_tr[idxs])
    
    # zero out the previous grads to avoid gradient accumulation
    for layer in layers:
        if isinstance(layer, Tanh) or isinstance(layer, Linear):
            layer.out.retain_grad()
    
    for layer in layers:
        layer.zero_grad()
    
    # do a backward and caculate fresh gradients wrt new batch
    loss.backward()
    
    # now optimize the weights and biases 
    lr = 0.1 if i<10000 else 0.01
    for p in parameters:
        p.data += -lr*p.grad
    
    if i%10_000:        
        print(f'{loss=:.4f}') 
    
    if i%10_000==0:
        losses.append(loss.log10().item())
    
    with torch.no_grad():
        update_ratio = [(lr*p.grad.std()/p.data.std()).log10().item() for p in parameters]
        update_ratios.append(update_ratio)
        
    if i>=1000:
        break

print(f'iter: {i} loss:{loss.item():.4f}') 
# iter: 1000 loss:2.8939 
# for gain=3 we get iter: 1000 loss:2.3039
# for when we used batchnorm at the very end as the last layer : iter: 1000 loss:2.8343

legends=[]
plt.figure(figsize=(20,4))
for i,layer in enumerate(layers[:-1]):
    # tanh is used becasue it has a finite range of -1,1 and visualizing it is easy
    if isinstance(layer, Tanh):
    # if isinstance(layer, Linear): # we test this when tanh layers are commented out
        output = layer.out
        layer_name = layer.__class__.__name__
        mean = output.mean()
        std = output.std()
        saturated = (output.abs()>0.97).float().mean()*100
        print(f'layer {i} ({layer_name:10}) mean: {mean:.4f} std: {std:.4f} saturated: {saturated:.2f}%')
        # now lets plot these , we use histogram to get the values 
        ty,tx = torch.histogram(output, density=True)
        plt.plot(tx[:-1].detach(), ty.detach())
        legends.append(f'layer{i}({layer_name})')
        
plt.legend(legends)
plt.title('activation distribution')
#%%
# we can do the same exact thing with the gradients instead of activations 
plt.figure(figsize=(20,5)) # the slight change in figuresize is to force matplotlib to not overwrite existing one and create one anew each time
legends= []
for i, layer in enumerate(layers[:-1]):
    if isinstance(layer, Tanh):
    # if isinstance(layer, Linear):
        output_grad = layer.out.grad
        layer_name = layer.__class__.__name__
        mean = output_grad.mean()
        std = output_grad.std()
        print(f'layer {i} ({layer_name:10}) mean: {mean:.4f} std: {std:.4f}')
        # now lets draw the histogram 
        hy,hx = torch.histogram(output_grad, density=True)
        # plot the points! note that plt doesnt know about torch tensors, so we give them numpy()
        # (.detach(), creates a numpy() copy of our tensor whcih matplotlib can use)
        plt.plot(hx.detach()[:-1], hy.detach(),)
        legends.append(layer_name)
plt.legend(legends)
plt.title('gradient distribution')
#%%
# now let us also visualize the parameters 
legends = []
plt.figure(figsize=(20,4)) # the slight change in figuresize is to force matplotlib to not overwrite existing one and create one anew each time
for i,p in enumerate(parameters):
    grad = p.grad
    shape = tuple(p.shape) # we could also write p.data.numpy().shape to get the pure shape but this is easier and less typing!
    if shape[0] > 1: # only plot weights (ignore biases)
        print(f'weight {shape} mean: {p.mean():.6f} std: {p.std():.6f} grad/data ratio: {grad.std()/p.std():.2f}')
        hy,hx = torch.histogram(grad)
        plt.plot(hx[:-1].detach(), hy.detach())
        legends.append(f'{i} {shape}')
plt.legend(legends)
plt.title('parameters statistics')

#%% 
# now lets plot the update ratios !
# when we used batchnorm as the last layer, note that we kind of messed up things, 
# our first layer(blue) starts training slower, but ultimately catches up though, 
# and other layers, update faster(-2.25) nothing too bad,as they stabalize later on
# but compared to when we dont use batchnorm as the last layer, these are more whacky!
# and much more osiliation can be seen.
# so personally I wouldnt use batchnorm at the final layer, although as we said and saw
# it doesnt impose much problem(of course its specific to problem at hand! this maynot
# be the case for all the cases out there!)
#
# also note that, we might see that, even though we are using batchnorm, changing the gain does have an effect
# on the gradients update (and this is especially evident in this plot (try gain=2 while batchnorm is
# the last layer))
# this happens especially for example when batchnorm is the last layer, 
# we see that pretty much everywhere else, everything looks well behaved nontheless, which is good
# and in this plot especially, (when batchnorm is used as the last layer), we see the trainig speed 
# is impeded, so we need higher learning rate to compensate that it seems. (the update ratio is 
# around -3.6 and basically below the line, though it ultimately catches up and stabalizes
# but its janky with lots of ossilliations ), I didnt observe this behaviro when no batchnorm
# is used as the last layer, suggesting this doesnt happen at least as heavely when batcnorm
# is not used at the last layer, so overall the choice of gain doesnt matter as much
# so to recap, the -3 on logscale seems like a good value, anything higher or lower might prbably
# be too small or too big for update magnitude (but at the end its not carved in stone, 
# check this with your usecase, its an intuition after all and we should be able to work it out
# for ourseleves.)

legends =[]
plt.figure(figsize=(20,5))
for i,p in enumerate(parameters):
    shape = tuple(p.shape)
    if shape[0]>1:
        # use only weights
        # plot!
        plt.plot([update_ratios[j][i] for j in range(len(update_ratios))])
        legends.append(f'param {i}')
# draw a line from 0 to the length of update ratios, to show where the ratios should ultimately reside     
# the ratios should be ~1e-3; indicate on plot
plt.plot([0, len(update_ratios)],[-3,-3], 'k') 
plt.legend(legends)