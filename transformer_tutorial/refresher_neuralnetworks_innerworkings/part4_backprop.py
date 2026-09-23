# in the name of God the most compassionate the most merciful 
# in this section we will try to code the backward pass for our 
# initial part3 example, basically we will implement the backwardpass
# for a 3 layer nn (4 layer including an embedding layer)
# this is to get familiar with the tensor operations in a backward pass
# and get an intuitive idea of how stuff works under the hood
# things such as broadcasting, mean, max, etc will be worked out inshaallah.
#%% 
import random
import itertools
import torch

# we start off by reading the data and implementing our simple nn with emebdings
names = open('./data/names.txt').read().splitlines()
# we need to grab all the unique characters in the dataset
# and sort them for ease of use
character_list = sorted(set(''.join(names)))
print(f'{character_list=}')

# now we need to create integer representations of the characters becasue neural networks
# understand numbers only!
# but before that, for ease of use, lets put special character denoting the start/end of
# names at the very begining of mapping (assigne the first code, 0, to it), 
# we reserve the first code (0) for our symbol which is '.' 
character_list = ['.']+character_list
atoi={ch:i for i,ch in enumerate(character_list)}

# now lets create the itoa for getting back the characters from numerical codes
itoa = {v:k for k,v in atoi.items()}
# print(atoi,'\n',itoa)

# now lets create our dataset 
# we are going to create a model which accepts the first 3 characters, 
# and returns the 4th. Here we are creating context, given a context of 3 characters, 
# provide us with the next. The starting point would be an empty string, dentoting
# equal likelihood for any given name.
# 
# context_size = 3 
# # list for holding our dataset samples
# X = []
# # a list for the labels which contains the next characters for each sample in X
# Y = []
# for name in names: 
#     # we build the initial sample, and remember we need numbers, not characters
#     sample = [atoi['.']]*context_size
#     # we append an ending symbol at the end of each name to signify where it ends
#     # since our initial sample contains the initial empty symbol, we dont readd any here
#     for ch in name+'.':
#         idx = atoi[ch]
#         X.append(sample)
#         Y.append(idx)
#         # update sample with the new character, move one character forward
#         sample = sample[1:] + [idx]
# 
## now lets check X and Y
# print(X[:5], ''.join([itoa[c] for p in X[:5] for c in p]))
# print(Y[:5], ''.join([itoa[c] for c in Y[:5]]))

# now lets convert this into a function for easier use
# in case we decide to have different splits like training, val, test
def build_dataset(names:list[str], context_size:int):
    # list for holding our dataset samples
    X:list[int] = []
    # a list for the labels which contains the next characters for each sample in X
    Y:list[int] = []
    for name in names: 
        # we build the initial sample, and remember we need numbers, not characters
        sample = [atoi['.']]*context_size
        # we append an ending symbol at the end of each name to signify where it ends
        # since our initial sample contains the initial empty symbol, we dont readd any here
        for ch in name+'.':
            idx = atoi[ch]
            X.append(sample)
            Y.append(idx)
            # update sample with the new character, move one character forward
            sample = sample[1:] + [idx]
    return X, Y

context_size = 3
X,Y = build_dataset(names, context_size)

# now lets check X and Y
print(X[:5], ''.join([itoa[c] for p in X[:5] for c in p]))
print(Y[:5], ''.join([itoa[c] for c in Y[:5]]))

# now lets create tensors out of our lists 
# but before that lets create some splits first!
# 80% for training and 10% for val and test resspectively
len_dataset = len(names)
num1 = int(0.8*len_dataset)
num2 = int(0.9*len_dataset)

# to shuffle our lists we use random.shuffle to shuffle our X 
random.seed(255)
random.shuffle(names)
names_tr = names[:num1]
names_val = names[num1:num2]
names_test = names[num2:]

# or we could have also done 
# tr_cnt = int(0.8*len_dataset)
# val_cnt = len_dataset - tr_cnt//2
# names_tr = names[:tr_cnt]
# names_val = names[tr_cnt: tr_cnt+val_cnt]
# names_test = names[tr_cnt+val_cnt:]
# assert sum(map(len,[names_tr,names_val,names_test])) == len(names) , 'must match'
#
# now lets create our dataset using names_tr
X_tr, Y_tr = build_dataset(names_tr, context_size)

# now lets check X and Y
print(X[:5], ''.join([itoa[c] for p in X[:5] for c in p]))
print(Y[:5], ''.join([itoa[c] for c in Y[:5]]))

# now lets convert them to tensor
X_tr = torch.tensor(X_tr)
Y_tr = torch.tensor(Y_tr)

# now that we have our dataset and labels ready lets create our network
# we start with an embedding layer which is nothing but a lookup table 
# for our characters, so for each character we get an embedding

# before we go on, lets set a manual seed for deterministic outcome
# torch.manual_seed(255)
g = torch.Generator().manual_seed(255)
vocab_size = len(character_list)
embedding_size = 10

EMB = torch.randn(size = (vocab_size, embedding_size), generator=g)

# next we should have a linear layer after our embedding layer, to work on the embeddings
# since our input is 3 numbers, and each number has an embedding vector of size 10, our
# linear layer needs an in_features =30
hidden_size = 100
W1 = torch.randn(size=(context_size * embedding_size, hidden_size), generator=g)
b1 = torch.randn(size=(1,hidden_size), generator=g)
# we need a second layer to give us the final probablities for the next character
W2 = torch.randn(size=(hidden_size, vocab_size), generator=g)
b2 = torch.randn(size=(1,vocab_size), generator=g)
# we want to add a batchnorm layer, so we need to create the parameters for it as well
# batchnorm needs a gamma, a beta as the only two learnable parameters
# we usually start gamma_gain as ones, and beta_bias as zeros so the start off with 
# an initial guassian distribution, but here for unsmasking our possible errors we use
# randn
bn_gamma_gain = torch.randn(size=(1,hidden_size), generator=g)
bn_beta_bias = torch.randn(size=(1,hidden_size), generator=g)

# now lets create a parameters list to hold all the parameters for optimization
parameters = [EMB, W1, b1, W2, b2, bn_gamma_gain, bn_beta_bias]
# before starting the optimization lets enable their gradients
for param in parameters:
    param.requires_grad = True 
    
# OK now lets do a forward pass 
iter_max = 200_000
batch_size = 32
eps= 1e-6
losses=[]
for i in range (iter_max):
    # lets create a random batch of the input
    batch_idxs = torch.randint(0, len(X_tr), size=(batch_size,), generator=g)# shape: [32]
    
    # print(X_tr.shape) # X_tr.shape: (182535,3)
    
    # get the mini-batch
    x_batch = X_tr[batch_idxs]
    
    # get the embeddings
    embds = EMB[x_batch]
    
    # reshaping here, is actually concatenating the tensors, so for easier calculations
    # for manual gradient calculation, lets make this a separate operation
    embdscat = embds.view(embds.shape[0],-1)
    
    # now lets apply the first layer 
    out_preact = embdscat @ W1 + b1
    
    # now lets apply the batchnorm
    # to do this we need to calculate mean, var and output_hat 
    # and then use gamma and beta
    # calculate the mean/var across all samples, along 0th dimension
    mean = out_preact.mean(dim=0, keepdim=True)
     
    # since backpropagating from .var() is hard becasue its multipart, 
    # lets divide it into smaller chuncks that we can easily derive
    # 
    # var = out_preact.var(dim=0, keepdim=True)
    # var is basically 
    # bn_diff = out_preact - mean
    # bn_diff2 = bn_diff**2 
    # 
    # note from future: 
    # first see the code below and then come read these notes
    # note that here, we are dividing by 1/n-1 rather than 1/n which is in the paper
    # this is called an unbaised variance, infact in the BN paper, they do the biased variance
    # during training (which uses 1/n) but during inference they use the unbiased version
    # (that is the 1/(n-1) like what we have here) this is called the bessels correction
    # and it happens that when we actually do this, our numerical stability gets better
    # without this, for example db1 at the very end, would not be approximately the same
    # as what pytorch produces, but after I made this change, it just became True!
    # (update, it seems, while this helps, the b1 being approx True or not, is related to seed!
    # change the seed and find out! need to figureout why this is happening and whether this
    # is torch's bug!)
    # 
    # var = 1/(n-1)*sum(bndiff_sqaured)
    # (so it gives us better estimate on variance! always use this!)
    # 
    # side note: 
    # if we look at torch.var, we see that it also has an unbiased argument
    # which after this seems to be True by default! (torch.var(input: Tensor, dim: _size | _int | None, unbiased: _bool = True, keepdim: _bool = False, *, out: Tensor | None = None) )
    # 
    # side note2:
    # Pytorchs implementation of batchnormalization, does biased variance for training and
    # uses unbiased version for test time (just like the paper), and based on karpathys view
    # this is a bug. he himself always uses the unbiased version all the time. 
    # also pytorch documentation at the time of writting this, doesnt show a way to
    # specify whehther to use biased or unbiased variance in BN, so its biased during training and
    # uses unbiased variance duing test. also this doesnt pose much of a problem when the batchsize is larger
    # 
    bn_diff = out_preact - mean 
    bn_diff2 = bn_diff**2 
    bn_diff2sum = bn_diff2.sum(dim=0, keepdim=True)
    # unbaised variance
    var = 1/(out_preact.shape[0]-1) * bn_diff2sum
    
    # normalize the input /add eps to prevent division by zero
    # x_hat = (out_preact-mean)/(var+eps)**0.5
    # but lets divide it into more parts so calculating 
    # the gradients later on is easier
    out_normalized = out_preact - mean 
    var_sq = (var+eps)**0.5
    
    # from future: 
    # for some reason the division in pytorch causes our result not to
    # be the exact same one pytorch produces! 
    # they are approximately the same (the actual difference is extremely small),
    # but nonethe less its there so we change the division into power and multiplication.
    # I leave the backward calculation for division intact (just comment it)
    # x_hat = out_normalized/var_sq
    var_sq_inv = var_sq**-1 
    x_hat = out_normalized * var_sq_inv
    
    # finally apply the gamma_gain and bias on the x_hat
    out_bn = bn_gamma_gain * x_hat + bn_beta_bias
    
    # apply tanh
    out_tanh = torch.tanh(out_bn)
    
    # now lets apply the second layer
    logits = out_tanh @ W2 + b2
    
    # lets implement the crossentropy loss now
    # which is the negative log likelihood 
    # which means we must treat our output as logcounts
    # so we do exp() to make them non negative
    # and then normalize them by their total which would be softmax
    # but to make it numerically stable, we subtract them from their max
    # that would be our probabablities, and then we would look at the
    # predictions for for the right classes and take their log and mean
    # becasue we need the negative of log of (likelihood/probablity).
    # 
    # since we have several samples, we either need to sum or average
    # since average is more customary (because it results in a smaller loss)
    # we use that.
    # 
    # note that our logits shape is (32,27), 32 being the batch size, 
    # and 27 being the vocab size, or in other words, our number of classes. 
    # so when we want to calculate max, we need to have a max for each sample,
    # 
    # (note that we are going to treat each column as a class probablity, 
    # so their total needs to sum to 1, therefore we dont care about other rows.
    # when we are normalizing or doing anything, its sample wise and involves 
    # all classes for that row/sample), 
    #
    # we have 32 input examples, and therefor for each example, we see whats 
    # the maximum value among classes for that specific input. 
    # 
    # note that the keepdim=True is needed becasue it makes it a row vector (32,1),
    # which when broadcasted, would allow us to subtract the values
    # in each column from the maximum for that specific row!
    # 
    # also note that torch.max, or tensor.max, returns a tuple of indexes and values
    # but since we want the values only, we used .values (also without using .values 
    # property, we had to set the require_grads on logits_max explicityly otherwise
    # it wouldnt get the gradients)
    logits_max = logits.max(dim=1, keepdim=True).values
    # subtract from the max for each sample
    logitsnorm = logits - logits_max
    # treat them as log counts (we do a exp to make them all non-negative)
    logcounts = logitsnorm.exp()
    # normalize and get a probablity distribution
    # sum along the columns, so each value is normalized properly 
    # remember that we have 27 columns, each representing 1 class, 
    # so their total count must sum to 1, so we sum along the columns
    # to get the total and then each column divided by that total gives us
    # propabalities
    # probs = logcounts/logcounts.sum(dim=1, keepdim=True)
    # for easier calculation of gradients, lets split the operations into separate ones
    logcountsum = logcounts.sum(dim=1, keepdim=True)
    
    # note from future: 
    # initially this was the only section, but as I later, down below, explained during
    # backpropagation section, I faced an issue, where the exact bit when we use division
    # doesnt happen, basically our result has an eps difference with pytorchs output
    # and it seems as more operations are encountered, this epsilon gets larger and larger
    # until for some later backprop results down the road, it just becomes very large
    # so much so that even the results are no longer approximately the same so 
    # I had to convert the division into power and multiplication
    # so I comment this line here now and instead write its replacements 
    # probs = logcounts /logcountsum
    # instead of division, lets convert that into multiplication!
    logcountsum_inv = logcountsum**-1
    probs = logcounts * logcountsum_inv
    
    # calculate the final negative log of likelihoods
    # side note, note that we use arange, and not 'range', torch.range is deprecated
    # becasue its behavior is different from python's range, that is it returns [start,end]
    # instead of returning [start,end), again that means, if we write torch.range(0,32)
    # it will return 33 numbers (0 up and including 32 itself) whereas pythons range
    # and the new torch.arange, return only 32 numbers (0 up to including 31)
    # this behavior also causes an issue here, if you use torch.range here 
    # you'll get a not so obvious error : 
    # IndexError: tensors used as indices must be long, int, byte or bool tensors
    logprobs = probs.log()
    loss = -logprobs[torch.arange(0,len(batch_idxs)), Y_tr[batch_idxs]].mean()
    # or we could write this as 
    # logprobs = probs[torch.arange(0,len(batch_idxs)), Y_tr[batch_idxs]].log()
    # loss = -logprobs.mean()
    
    # zero out gradients before calculating the new grads
    for param in parameters:
        param.grad = None 
    
    # before we do a backwardpass, lets freeze the gradients for our test 
    # we retain the gradients for intermediate operations result as well 
    # to check our manual gradients against them.
    # disable for full training of course!
    for param in itertools.chain(parameters,[logprobs, probs, logcounts,logcountsum,
                                             logcountsum_inv,logitsnorm,
                                             logits, logits_max, out_bn,
                                             out_tanh,x_hat, out_preact,
                                             var, mean, out_preact,var_sq,var_sq_inv,
                                             bn_diff2sum,bn_diff2, bn_diff, out_normalized,
                                             embds, embdscat]):
        param.retain_grad()
    
    loss.backward()
    losses.append(loss.item())
    # break to be able to calculate the backward pass ourselves
    break

    if i%10000==0:
        print(f'{loss}')
        
    lr = 0.1 if i<=100_000 else 0.01
    for param in parameters:
        param.data += -lr * param.grad
print(f'loss: ',sum(losses)/len(losses))
# prints a full training results in  loss:  2.322884672728181
#%%
print(torch.__version__)
# lets create a function for comparing our gradients with pytorchs
def compare(label, our_grad, pytorch_tensor, atol=1e-8):
    """compares our manually calculated gradients 
    against the pytorchs automatically calculated 
    gradients.

    Args:
        str (_type_): a string to describe the tensors being compared
        our_grad (_type_): our manually calculated gradients
        pytorch_tensor (_type_): pytorch tensors for which we have calculated the gradient
        atol (_type_): absolute difference between two tensors that we accept as being equal.
    """
    pytorch_grad = pytorch_tensor.grad
    # check whther they are exactly equal
    exactly_same = torch.all(our_grad == pytorch_grad).item()
    # check whether they are approximately equal
    approximately_same = torch.allclose(our_grad, pytorch_grad, atol=atol)
    # lets also calcualte their difference
    difference = (our_grad - pytorch_grad).abs().max().item()
    print(f'{label:17}| exact: {str(exactly_same):5} | approx: {str(approximately_same):5} | diff: {difference}')

# now the first operation that we have is probs
# our first variable to get its grads is the -logprobs.mean()
# so we write 
# dlogprobs = ???
# so what should we write here? how to go about getting the grads for a mean() operation?
# one good tip to tackle these is to comeup with a much simpler example and see how
# we can get the deravitaive for that. 
# to this end, suppose we have sth like this:
# we know that mean is simply a sum of several numbers divided by their count, like this:
# y = -(a + b + c )/3
# 
# now if we were to calculate dd/da what would that be?
# to answer that we can simplify the previous line like this: 
# y= -1/3a + -1/3b + -1/3c 
# right? its the same previous expression, we just expanded it. 
# now once again what would be the dd/da? 
# dy/da = -1/3
# what about dd/db?
# dy/db = -1/3
# 
# and so on. so here we had 3 numbers, so it was -1/3, if we had more, we would write more
# so its basically -1/n, n being the number of all numbers involved. 
#
# so whats the n here? the shape of logpros is (32,27) 32 sample, and 27 classes
# and we know that we must calculate the gradient for every elements in our tensor
# 
# so what should do now? lets expand on the line that creates logprobs
# its loss=-logprobs[torch.arange(0,len(batch_idxs)), Y_tr[batch_idxs]].mean()
# what are we doing exactly? 
# logprobs[torch.arange(0,len(batch_idxs)), Y_tr[batch_idxs]] simply is 
# two part, the first part simply acts like a loop, at each time it selects a single row
# of logprobs, and the second part simply picks the correct column/class, together they
# give us the true class for each sample. 
# if we print Y_tr[batch_idxs] we will see sth like this: 
# 
# tensor([ 8, 14,  0, 12, 17,  9,  0, 25,  5,  0,  5,  0, 25,  1,  3,  2,  0, 11,
#         14,  0,  1, 26,  2,  9,  8,  9,  1, 14,  5,  9, 14,  1])
# 
# each of these are index to the correct class, what this is saying, is, 
# the correct class for sample 1 is 8, the correct class for sample 2 is 14, and so on
# 
# so we have 32 numbers, 32 is our n so the gradients for these locations in probs 
# is simply -1/n or -1/32 in our case. 
# now what about the rest of the logprobs elements? simple, since they didnt participate 
# in the training, their gradients will be zero.
# and since our gradinet is always the same shape of our input and also the majority 
# are zero, we can simply do sth like this 
dlogprobs = torch.zeros_like(logprobs)
# and now we just need to update the locations that participated in training so 
dlogprobs[range(0,len(batch_idxs)), Y_tr[batch_idxs]] = -1/len(batch_idxs)
# now lets see if our calculation is correct 
compare('dlogprobs',dlogprobs, logprobs)

# now if we were to calculated the logprobs like this : 
#logprobs = probs[torch.arange(0,len(batch_idxs)), Y_tr[batch_idxs]].log()
#loss = -logprobs.mean()
#logprobs would be [32] and we had to do 
# dlogprobs = torch.zeros_like(logprobs)
# dlogprobs[...] = -1/len(batch_idxs)
# and that would be it 
# compare('dlogprobs',dlogprobs, logprobs)
# and for the next round, we had to calculate the probs[...].log gradients
# 
# (side note: if we tried to print(probs[torch.arange(0,len(batch_idxs)), Y_tr[batch_idxs]].grad)
# we'll get this error : 
#  UserWarning: The .grad attribute of a Tensor that is not a leaf Tensor is being accessed. 
# Its .grad attribute won't be populated during autograd.backward(). If you indeed want the
# .grad field to be populated for a non-leaf Tensor, use .retain_grad() on the non-leaf Tensor.
# If you access the non-leaf Tensor by mistake, make sure you access the leaf Tensor instead. 
# See github.com/pytorch/pytorch/pull/30531 for more informations. 
# (Triggered internally at /croot/pytorch_1686931851744/work/build/aten/src/ATen/core/TensorBody.h:486.)
# basically that intermidiate operation(slicing, etc involved) is not available to us here, 
# because it was temporarily made and assigned.) 
# 
# so to see the gradients, we have to see the whole tensor.grad and then compare ours to it
# that should be fine.(because we need to calculate the grdients for all elements anyway,
# and for comparison that should suffice, however this seems like a bug to me, the grads are 
# already calculated, so pytorch should be able to return them, but instead it executes the logic
# that should only be called when calculating the gradients, or maybe its because I didnt call
# retain_grads on it?) anyway!
# but what should we do now? how can we calculate the gradients of log(x)?
# gradients of log(x) is 1/x (for reminder see : https://www.intmath.com/differentiation-transcendental/5-derivative-logarithm.php)
# so if we had for example 
# y = [log a , logb , logc] 
# dy/da = 1/a
# 
# so in our case it would be 
# dprobs[torch.arange(0,len(batch_idxs)), Y_tr[batch_idxs]] = 1/probs[torch.arange(0,len(batch_idxs)), Y_tr[batch_idxs]]
# and what about the rest of the elements? like before, they would be zero
# so we would have 
# dprobs = torch.zeros_like(probs)
# since we have chain rule, we multiple the previous gradients times this one 
# dprobs[torch.arange(0,len(batch_idxs)), Y_tr[batch_idxs]] = dlogprobs * 1/probs[torch.arange(0,len(batch_idxs)), Y_tr[batch_idxs]]
# compare('drpobs',dprobs, probs)
# and we see this is exactly the answer
# print(f'dprobs:{dprobs[torch.arange(0,len(batch_idxs)), Y_tr[batch_idxs]]}')
# print(f'probs:{probs.grad[torch.arange(0,len(batch_idxs)), Y_tr[batch_idxs]]}')
#
# now for dprobs, we already saw the second scenario, so we know hwat to do 
dprobs = torch.zeros_like(probs)
# 
# the derivative of log(x) is 1/x, and since its chain rule, we need to multiply
# the previous gradients here as well (also recall that each function has a local gradient
# here our function is log, and its local derivative is 1/probs, we need to multiply that
# to the gradient from previous operation.)
# 
# also on a sidenote: what this line is effectively doing, is its boosting the gradients 
# of the values that had low probabilities (recall that when probs are correct, its 1
# and 1/1 is 1 basically passing through the dlogprobs, but if the probs are wrong, they
# would be small and 1/small_num basically is a larger number, now being multiplied by
# dlogprobs, making them bigger)
dprobs[...] = (1/probs) * dlogprobs 
# lets compare the two ... which is 100% ok!
compare('dprobs',dprobs, probs)
# so far so good, now we reach to 
#probs = logcounts * logcountsum_inv
# now here we need to calculate dlogcounts and dlogcountsum_inv
# firs lets do dlogcounts:
# first we print the shapes: 
# print(f'{dprobs.shape=} {logcounts.shape=} {logcountsum_inv.shape=}')
# dprobs.shape:           (32, 27) # previous gradient
# logcounts.shape:        (32, 27) 
# logcountsum_inv.shape:  (32, 1)
# so we see the shapes of logcounts and logcountsum_inv differs, this means a
# broadcast is necessary which we need to take into consideration.
# for now the dlogcount needs to be the same shape as of logcounts which is (32,27)
# and its local derivative logcountsum_inv which is (32,1), it also needs to be 
# multiplied by the previous gradient which is dprob (32,27), so lets do this
dlogcounts = logcountsum_inv * dprobs
# the shape is 32,27 which means all is Ok (side note, note that we did an elemenwise
# muiltiplication here, note the * here, matrix multiplication (@) cant be used because
# dimensions dont match! always remember this!)
# now for dlogcountsum_inv, its shape must be the same of logcountsum_inv which is (32,1)
# but its local gradient is logcounts which's shape is (32,27), and we also need to 
# multiply by the previous gradient which is dprob with shape of (32,27), as you see
# the result would be 32,27, but we need it to be (32,1). this means we need to 
# somehow get this to become (32,1). we use sum over the columns to do this (cuz remember
# logcountsum_inv had to be broadcasted in first place to be 32,27, and it did so by
# replicating the same column to the right until 27 identical columns are created
# therefore, when taking the gradients, we sum all of these replicated cells as well
# to account for their true effect in the operation).
# also, if in doubt on which axis to chose remember that, if its a column vector, 
# replication is done columnwise, so the addition needs to be columnwise as well
# also this is the only way the shape becomes the same)
# logcounts and dprobs have the same shape so we elementwise multiply(hadamard product) them and the sum
dlogcountsum_inv = (logcounts * dprobs).sum(dim=1, keepdim=True)
# lets compare these two
# compare('dlogcounts-incom', dlogcounts, logcounts)
compare('dlogcountsum_inv', dlogcountsum_inv, logcountsum_inv)
# ok we see dlogcounts is not the same! but dlogcountsum_inv is
# why is that? if you look closely, you'll notice logcounts is also involved in another
# operation(two operations below), so since we havent accounted for its contribution there
# its not the same as pytorch's result. lets continue this 
#
# and next is logcountsum_inv = logcountsum**-1
# first lets print the shape , we just saw above logcountsum_inv is (32,1)
# so logcountsum shape is (32,1) as well, since its an elementwise power operation
# the shape is the same. lets calculate its local derivative which is pow*x**(pow-1)
# we also need to multiply this by the previous gradient which is dlogcountsum_inv
# with the same shape
dlogcountsum = -1*logcountsum**(-2) * dlogcountsum_inv
# 
# and next is logcountsum = logcounts.sum(dim=1, keepdim=True)
# as you can see, logcounts is being used here as well, and we need to account 
# for its role here (and route the gradient through it as well)
# logcounts shape is (32,27), so its dlogcounts shape must be the same (32,27)
# we need to calculate its local derivative and multiply it by the previous gradient
# but the logcountsum shape is (32,1). 
# now what should we do here? how should we go about this? 
# whats the local gradient for logcounts.sum()?
# whenever there is a sum operation, it means routing the gradient, basically
# the local gradients of the tensor on which the sum() is being applied, is 1
# so the local gradients for logcounts is simply a tensor of ones everywhere!
# but we also need to multiply the local gradients with the previous gradient 
# (becasue of chain rule and also routing the gradients in the graph as you know)
# but the previous gradient shape is (32,1). 
# it happens that this doesnt pose any issues, we can simply multiply them, and 
# the previous gradient will be broadcasted, its a column vector, so what happens
# is this column will be replicated 27 times to form a 32,27 matrix. which then
# these will be multiplied elemntwise by 1s and get routed.
# further explanation (i wrote for the previous version (down below)):
# lets see what we have here, we have logcounts which is (32,27)
# and is being summed over so that it results in logcountsum with shape (32,1)
# or a column vector (i.e. a vector that has only 1 full column!)
# lets see what happens here with a simple example 
# suppose we have a 3x3 tensor a, 
# and its being summed to a column vector b of shape 3x1, and 
# the b vector is mean by summing all the columns in each row:
# [a11 a12 a13] ---> [b1]   [a11 + a12 + a13]
# [a21 a22 a23] ---> [b2] = [a21 + a22 + a23]
# [a31 a32 a33] ---> [b3]   [a31 + a32 + a33]
# now we have dervivate with respect to this vector b, and now we want 
# the gradients with respect to the original tensor, 
# the gradients, basically we want to understand how bs depend on the as 
# whats the local deriviate of this operation,
# from our simple matrix-schematic above, we can see that each b is 
# dependent only on the as in the same row, so b1 depends on a11, a12, a13 only
# and likewise, b2 only depends on a21,a22,a23, and so on. 
# this observation, tells us, that b1 has no interaction with the a2x and a3x rows
# so it has no effect on them. therefore the deravative of b1 with respect to 
# all the elemenst in the rows 2 and 3 is basically zero, but its deravative with respect
# to the first row (a1), is basically 1 for each element. 
# so to finally calculate the gradients, we multiply the local gradients (which are 1s for each row for all columns)
# by the gradients from previous operation, 
# so we can create a (32,27) tensor and fill it accordingly, i.e. each row 
# will have the gradients of the corrosponding row in dlogcountsum, replicated all the way
# for all the columns in that row, and this repeats for all the rows. 
# one easy way for achiving this is to create a ones_tensor like logscount which is 32x27
# and multiply that with the dlogcountsum which is 32,1 (which would be broadcasted to
# 32x27 itself, and each row would replicate the first column 27 times, resulting in the
# final answer we are after.) that would be: 
# also note that since dlogcounts was once calculated, we need to add this to 
# the existing gradients.
dlogcounts += torch.ones_like(logcounts) * dlogcountsum
# now lets compare the dlogcountsum and dlogcounts now 
compare('dlogcountsum', dlogcountsum, logcountsum)
compare('dlogcounts', dlogcounts, logcounts)
#
# if we were to process this like and use / this is how its done (but for some reason
# pytorch seems to be having a bug, if we go this way, the exact equality becomes false
# meaning, our results would differ with a small diff and gradually until the end this
# number would increase, so for this reason we use the first method and I leave this part
# commented out)
# probs = logcounts/logcounts.sum(dim=1, keepdim=True)
# or the simplified version which is:
# logcountsum = logcounts.sum(dim=1, keepdim=True)
# probs = logcounts/logcountsum
# what do we have here? a form like y = a/b
# dy/da = 1/b
# and dy/db would be -a*b**-2 how?
# recall that we could write a/b as a*b**-1 and if we try to solve that 
# it would be a*-1*b**(-1-1) which would be -ab**-2
# which would ultimately be (-logcounts*(logcountsum**-2))
# but theres an issue here, the logcountsum shape(32,1) is different than the logscount(32,27)
# for this to work, logcountsum needs to be broadcasted, and each row replicates a single value
# and then an elementiwise multiplication is done, so we have two operations here, one replication
# and one multiplication. lets give an example 
# y = a * b 
# and suppose a is 3x3 and b is 3x1 
# it would look like this: 
# [a11  a12  a13]      [b1]    
# [a21  a22  a23]   *  [b2]    = ?
# [a31  a32  a33]      [b3]    
# which then is broadcasted like this : 
# [a11  a12  a13]      [b1  b1  b1]     [a11*b1 a12*b1 a13*b1]
# [a21  a22  a23]   *  [b2  b2  b2]  =  [a21*b2 a22*b2 a23*b2]
# [a31  a32  a33]      [b3  b3  b3]     [a31*b3 a32*b3 a33*b3]
# so whats happening here, a replication followed by multiplication is happening here
# if we were to treat b as a scaler for example, it would be easy, y=ab, dy/da =b 
# so that would be logscount, however, we need to take the replication into account as well
# if we look closely, we'll note that basically each row is replicated, so if there was 
# one b, that would be 1, but since its replicated n times, the operation is done n times
# meaning we need to sum them all along each row!
# alsonote that logcountsumshape was(32,1), but logcounts shape was(32,27)
# the dlogscountsum must have the same shape as the logscountsum, so we needed to make the (32,27)
# into (32,1) anyway, further helping us to note that we needed that sum
# so ultimately we would do (we would also keepdim=True) to have the 32,1 shape to make all work
# dlogcountsum = logcounts.sum(dim=1, keepdims=True) 
# now this was the local gradient, we need to multiply that with the previous gradient so
# ultimately we would endup doing this :
# in the dimension 
# print(f'{logcounts.shape=}, {logcountsum.shape=} {dprobs.shape}')
# dlogcounts = torch.zeros_like(logcounts)
# dlogcounts[...] = 1./logcountsum * dprobs
# note that since dprobs and logcounts had the same shape, we first multiply them and then sum the result
# dlogcountsum = (-(logcounts*(logcountsum**-2)) * dprobs).sum(dim=1, keepdims=True)
# now lets compare the results with pytorchs 
# compare('dlogcountsum', dlogcountsum, logcountsum)
# and now for logcounts we still need more to do as up there we have 
# logcountsum = logcounts.sum(dim=1, keepdim=True)
# lets see what we have here, we have logcounts which is (32,27)
# and is being summed over so that it results in logcountsum with shape (32,1)
# or a column vector (i.e. a vector that has only 1 full column!)
# 
# lets see what happens here with a simple example 
# suppose we have a 3x3 tensor a, 
# and its being summed to a column vector b of shape 3x1, and 
# the vector b is created by summing all the columns in each row:
# [a11 a12 a13] ---> [b1]   [a11 + a12 + a13]
# [a21 a22 a23] ---> [b2] = [a21 + a22 + a23]
# [a31 a32 a33] ---> [b3]   [a31 + a32 + a33]
# now we have dervivate with respect to this vector b, and now we want 
# the gradients with respect to the original tensor, 
# the gradients, basically we want to understand how bs depend on the as 
# whats the local deriviate of this operation,
# from our simple matrix-schematic above, we can see that each b is 
# dependent only on the as in the same row, so b1 depends on a11, a12, a13 only
# and likewise, b2 only depends on a21,a22,a23, and so on. 
# this observation, tells us, that b1 has no interaction with the a2x and a3x rows
# so it has no effect on them. therefore the deravative of b1 with respect to 
# all the elemenst in the rows 2 and 3 is basically zero, but its deravative with respect
# to the first row (a1), is basically 1 for each element. 
# so to finally calculate the gradients, we multiply the local gradients 
# (which are 1s for each row for all columns)
# by the gradients from previous operation, 
# so we can create a (32,27) tensor and fill it accordingly, i.e. each row 
# will have the gradients of the corrosponding row in dlogcountsum, replicated all the way
# for all the columns in that row, and this repeats for all the rows. 
# one easy way for achiving this is to create a ones_tensor like logscount which is 32x27
# and multiply that with the dlogcountsum which is 32,1 (which would be broadcasted to
# 32x27 itself, and each row would replicate the first column 27 times, resulting in the
# final answer we are after.) that would be: 
# also note that dlogcounts was once calculated, so this needs to be added to the previous result
# dlogcounts += torch.ones_like(logcounts) * dlogcountsum
# compare('dlogcounts', dlogcounts, logcounts)
#
# next we have logcounts = logitsnorm.exp()
# the deravative of logitsnorm.exp() will be logitsnorm.exp() or in otherwords
# the logcounts itself, so we may as well use that instead of recalculating it here again
# and since its the chainrule, we multiply our local gradient with the previous gradient
dlogitsnorm = logcounts * dlogcounts
compare('dlogitsnorm', dlogitsnorm, logitsnorm)
#
# next line is logitsnorm = logits - logits_max
# first lets check their shapes :
# print(f'{logits.shape=} , {logits_max.shape=}')
# and we notice their shape is different (
# logits.shape=[32, 27]
# logits_max.shape=[32, 1]
# so a broadcasting is being done here again lets do a simple example 
# [c11 c12 c13]   [a11 a12 a13]   [b1]
# [c21 c22 c23] = [a21 a22 a23] - [b2] 
# [c31 c32 c33]   [a31 a32 a33]   [b3]
# so for example we have c32 = a32 - b3
# and here we see that the derevative of a32 with respect to c32 is 1
# while the derevative of b3 with respect to c32 is -1
# so the derevative of all the input tensor is 1, while the derevative of 
# so the gradients from tensor C, just flows to the input tensor a,
# and it also flows to the column vector b (although it has -1), moreover
# vector b, gets broadcasted to do the operation, so we need to do a sum for it as well
# so we would have 
# basically copy the dlogitsnorm
dlogits = 1*dlogitsnorm
# and for logits_max 
dlogits_max = (-1*dlogitsnorm).sum(dim=1, keepdim=True) 
# compare('dlogits-incom', dlogits, logits)
compare('dlogits_max', dlogits_max, logits_max)

# side note1: that logits is involved in more operations, so its not yet complete
# side note2: we said earlier that the reason we take the max and subtract the logits from it
# is purely for the numerical stability, and it has no effect on the probablities, and 
# indeed it was the case, now what this means in terms of gradients and backpropagation
# is that the gradients for logits_max need to be zero or extremely small to not affect the probablity
# and if we look at the dlogits_max, we see this is indeed the case:
# print(f'{dlogits_max}') 
# prints 
# tensor([[ 0.0000e+00],
#         [-3.4503e-12],
#         [ 0.0000e+00],
#         [-4.0546e-10],
#         [-6.5379e-09],
#         [ 3.0216e-10],
#         [-1.1400e-09],
#         [ 1.7621e-09],
#         [-1.8626e-09],
#         [-1.9632e-09],
#         [-5.3842e-09],
#         [ 1.9791e-09],
#         [ 3.5725e-09],
#         [ 2.2119e-09],
#         [ 7.3342e-09],
#         [-2.7906e-09],
#         [ 6.0137e-09],
#         [-1.8490e-11],
#         [-5.8935e-10],
#         [ 1.8626e-09],
#         [-1.4012e-11],
#         [ 3.5414e-09],
#         [ 2.4594e-09],
#         [ 0.0000e+00],
#         [-1.9632e-09],
#         [ 3.7253e-09],
#         [ 1.0482e-09],
#         [-6.4898e-10],
#         [-2.3918e-11],
#         [-3.7253e-09],
#         [ 5.8790e-09],
#         [-1.4743e-09]], grad_fn=<SumBackward1>)
# shows very small numbers (basically zero)
#
# now next is logits_max = logits.max(dim=1, keepdim=True).values
# and we need to calculate dlogits with respect to max operation 
# that is, in other words, we want to route the gradients from dlogits_max 
# through the numbers in logits that were chosen as max, lets elaborate a bit on this
# and make this more clear, basically the max function, searches in the logits and picks
# some numbers and makes up the logits_max, therefore, now when the backward pass wants to
# route the gradients back to the begining, the gradients need to pass through each value
# in logits, creating the dlogits/ 
# so if we have the indexes of these max values, the local gradients would be 1 for them
# and we only then need to multiply it by the previous gradient which would be dlogits_max
# the indexes were the max didnt come from would obvisouly be zero!
# how do we do this now? 
# like before, we can first create a zero tensor with the same shape of logits
# and then fill up the locations that need to be one and the dothe multiplication
# with the previous gradient. 
# the good news is torch.max returns a tuple, an index and a value
# and we used the values for the maximum values. 
# dlogits was once calculated, so here we add to the previous gradients
# dlogits_max.shape=torch.Size([32, 1])
# this line is not correct, we only need to add the dlogits_max gradienst to specific columns
# and 0 otherwise. but for some reason this is so slightly different than the one_hot version
# which explictily sets all other elements to zero and only multiplies the max indexes to dlogits_max
# why?
# dlogits[range(0,len(logits_max)), logits.max(dim=1).indices] += 1*dlogits_max
dlogits += torch.nn.functional.one_hot(logits.max(dim=1).indices, num_classes=logits.shape[-1]) * dlogits_max
compare(f'dlogits', dlogits, logits)
# there is another way of doing this, and that would be using the one_hot_encoded method
# basically multiplying the right indexes by the dlogits would fill the dlogits propelry
# and left the rest at zero, basically like this
# dlogits += torch.nn.functional.one_hot(logits.max(dim=1).indices, num_classes=logits.shape[-1]) * dlogits_max
# compare(f'dlogits', dlogits, logits)
# if we try to visualize this one_hot encoded logits_max indices, we'll see that only the indices
# of max values are 1 and the rest are 0
# import matplotlib.pyplot as plt 
# plt.imshow(torch.nn.functional.one_hot(logits.max(dim=1).indices, num_classes=logits.shape[-1]).data)
#
# next is logits = out_tanh @ W2 + b2
# first lets see the shapes 
# out_tanh: (32,100)
# W2 : (100,27) and
# b2 : (1,27)
# print(f'{out_tanh.shape=}  {W2.shape=} {b2.shape=} {dlogits.shape=}')
# dout_tanh is W2 and dW2 is out_tanh, db2 would be 1
# (note the + ,whenever theres + the gradient is just routed)
# as always we multiply our local derivative with the previous gradients
# dout_tanh must have the same shape as out_tanh so it must be (32,100)
# but W2.shape is (100,27) and dlogits.shape is (32,27), so what do we do? 
# we do dlogits*W2.t()  
# print(f'{dlogits.shape=}, {W2.t().shape=}')
dout_tanh = dlogits@W2.T
compare('dout_tanh',dout_tanh, out_tanh)
#likewise, dW2 must be 100,27, out_tanh is (32,100) and dlogits is (32,27)
# to get this to work we need to have out_tanh.T*dlogits
dW2 = out_tanh.T@dlogits
compare('dW2', dW2, W2)
# again db2 shape must be (1,27) like b2, and it needs to be multiplied 
# by dlogits of (32,27), and we see we have a row vector (b2) or ones multiplyied by 32,27
# theres a broadcasting happening so we need to take that into account and as we know we sum
# over rows!
db2 = (1*dlogits).sum(dim=0,keepdim=True)
# now lets compare
compare('db2', db2, b2)
# next is out_tanh = torch.tanh(out_bn)
# first lets print the shape: 
# print(f'{dout_tanh.shape=}, {out_bn.shape=}')
# not surprisingly they both have the same shape which is (32,100)
# we need to calculate dout_bn, the derivative of tanh is 1-tanh(x)**2
# so its 1-(out_tanh)**2 * the previous gradient 
dout_bn = (1-out_tanh**2)* dout_tanh
#lets compare 
compare('dout_bn', dout_bn, out_bn)
#
# next line is out_bn = bn_gamma_gain * x_hat + bn_beta_bias
# first lets print their shapes
# print(f'{dout_bn.shape=} {bn_gamma_gain.shape=} {x_hat.shape=} {bn_beta_bias.shape=}')
# dout_bn.shape = (32,100) # previous gradients
# x_hat.shape=(32, 100) 
# bn_gamma_gain.shape=(1, 100) 
# bn_beta_bias.shape=(1, 100)
# and now we want dbn_gamma_gain,and  its shape must be the same as bn_gamma_gain(1,100)
# we also know that its derivative is x_hat but x_hat shape is (32,100), 
# so a broadcast must happen to give us out_bn of shape(32,100) and we need to sum along the rows
# so we are left with 1 row vector ultimately, lets dothis 
# we multiply our local gradient(x_hat) by previous gradient (dout_bn) 
# note that we multiply elementwise here and not matrix multiplication that requires transpose
# the two matrix have the same shape, so we can get their elementwise multiplication or
# (Hadamard product. - dont confuse this with dotproduct. 
# but since the shape needs to match, sum over rows to get a single row of (1,100)
# side note: elementwise multiplication is called hadamard product:
# withthe dot product, we multiply the corresponding components and add those products together. 
# With the Hadamard product (element-wise product) we multiply the corresponding components, 
# but do not aggregate by summation.
dbn_gamma_gain = (x_hat*dout_bn).sum(dim=0,keepdim=True)
# lets check 
compare('dbn_gamma_gain', dbn_gamma_gain, bn_gamma_gain)
# and likewise for x_hat we would have a dx_hat of shape (32,100)
# it would be bn_gamma_gain but it needs to be broadcasted, since its 
# (1,100), it will be replicated along the rows to gte (32,100)
# we let that be handled by the multiplication by previous gradient which is dout_bn
# which is has the shape of 32,100.
dx_hat = bn_gamma_gain*dout_bn
#lets compare 
compare('dx_hat', dx_hat, x_hat)
# next its bn_beta_bias, which as we know so far, + means rout the gradients
# bn_beta_bias, local gradients would be all 1s, and since its shape is different
# from x_hat(and out_bn), it must have been broadcasted, so we need to sum over rows!
# also it needs to be multiplied by previous gradient, and since dout_bn is (32,100)
# we need to sum this anyway for the shapes to workout (because again we know dbn_beta_bias
# has the same shape as of bn_beta_bias which is 1,100)
# dbn_beta_bias = torch.ones_like(bn_beta_bias)
dbn_beta_bias = (1*dout_bn).sum(dim=0,keepdim=True)
#lets compare
compare('dbn_beta_bias', dbn_beta_bias, bn_beta_bias)
#
# next is x_hat = (out_tanh-mean)/(var+eps)**0.5
# now this is multipart, it would have been much better if we separated them initially!
# ok so we separated them and now we have 
     
# so next in line is         x_hat = out_normalized * var_sq_inv
# first lets print the shapes involved: 
# print(f'{x_hat.shape=} {out_normalized.shape=} {var_sq_inv.shape=}')
# x_hat.shape=(32, 100) 
# out_normalized.shape=(32, 100) 
# var_sq_inv.shape=(1, 100)
# so we need to calculate the dout_normalized and dvar_sq_inv
# dout_normalized is 32,100, var_sq_inv which is its local gradient is (1,100) 
# and the previous gradient dx_hat is (32,100), so there shouldnt be a problem 
# as var_sq_inv will be broadcasted automatically when multiplied by dx_hat and
# the shape will come out just fine
dout_normalized = var_sq_inv*dx_hat
compare('dout_normalized', dout_normalized, out_normalized)
# now lets calculate the dvar_sq_inv
# its shape should be 1,100, its local gradient will be out_normalized which is 32,100
# and the previous gradient which is dx_hat is 32,100 as well, so the result of multiplication
# will be 32,100. in order to make it 1,100, we simply sum over rows.
dvar_sq_inv = (out_normalized*dx_hat).sum(dim=0, keepdim=True)
# now lets compare 
compare('dvar_sq_inv', dvar_sq_inv, var_sq_inv)
#
# and now                var_sq_inv = var_sq**-1
# var_sq shape is (1,100), the local gradient would be -1*var_sq**-2 and its previous
# gradient is dvar_sq_inv which is (1,100) so theres no issue lets calculate it 
dvar_sq = -1*var_sq**-2 * dvar_sq_inv
# now lets compare 
compare('dvar_sq', dvar_sq, var_sq)
#
# if we were to calculate the x_hat = out_normalized/var_sq
# this is how we would have done it: (note that in pytorch this creates a small eps difference
# between our result and pytorchs for dvar_sq beacsue its in a division!)
# lets see what we have, out_normalized,var_sq and we need to calculate their gradienst
# first lets print their shapes 
# print(f'{out_normalized.shape=}, {var_sq.shape=}, {dx_hat.shape=}')
# their shape is :
# out_normalized.shape=(32, 100) 
# var_sq.shape=(1, 100)
# dx_hat.shape=torch.Size([32, 100])
# and we know that dout_normalzied must have the shape of (32,100)
# but the var_sq is 1,100, and dx_hat is (32,100) so they are compatible in elementwise
# multiplication, since var_sq (1,100) will automatically be broadcasted and all is ok
# dout_normalized = dx_hat*(1.0/var_sq)
# lets compare
# compare('dout_normalized', dout_normalized, out_normalized)
# and for dvar which should have the shape(1,100), and also needs to be multiplied by
# the previous gradient (dx_hat) which is (32,100). 
# the result would be 32x100, so we need to sum over rows to get (1,100)
# dvar_sq = dx_hat*(-1*(var_sq**-2)).sum(dim=0,keepdim=True)
# compare('dvar_sq', dvar_sq, var_sq)
#
# next is                         var_sq = (var+eps)**0.5
#
# and now we need to calculate dvar 
# first print the shapes: 
# print(f'{var.shape=} {dvar_sq.shape=}')
# var.shape=(1, 100) 
# dvar_sq.shape=(1, 100)
# so dvar shape should be (1,100) as well, so we transpose one to get 100x100
# and then need to sum over rows to get 1,100
dvar = (dvar_sq * (0.5*(var+eps)**(0.5-1.0))).sum(dim=0, keepdim=True)
# print(f'{dvar.shape=}')
# lets compare
compare('dvar', dvar, var)
#
# and next is             out_normalized = out_preact - mean 
#
# lets print the shapes first: 
# print(f'{out_preact.shape=}, {mean.shape=}')
# out_preact.shape=(32, 100), 
# mean.shape=(1, 100)
# out_normalized.shape= (32,100)
# since we have subtraction, like addition, these route the gradients and their
# local gradients is just 1(for out_preact) and -1(for -mean). 
# note that dout_preact is involved in other operations as well (var and mean) 
# so this is not its final value!
dout_preact = torch.ones_like(out_preact) * dout_normalized
# and for mean (1,100), this is the same, we just need to sum over rows
dmean = -1*(torch.ones_like(mean)*dout_normalized).sum(dim=0, keepdim=True) 
# lets compare 
# compare('dout_preact-incom', dout_preact, out_preact)
compare('dmean', dmean, mean)
#
# next is       var = out_preact.var(dim=0, keepdim=True)
# which was hard so we split it into the smaller parts 
# so next is     
#               var = 1/(out_preact.shape[0]-1) * bn_diff2sum
# so here we needto calculate bn_diff2sum, 1/out_preact.shape[0] is simply a constant
# and doesnt need to be derived so we leave it.
# first print the shapes 
# print(f'{dvar.shape=} {bn_diff2sum.shape=}')
# they both have a shape 1,100
# the local derivate of bn_diff2 is 1/outpreact.shape[0] so we multiply that with the 
# previous gradient 
dbn_diff2sum = 1/(out_preact.shape[0]-1) * dvar
# next is            bn_diff2sum = bn_diff2.sum(dim=0, keepdim=True)
# also note that bndiff2 is being summed over, so we need to account for this as well
# we have a gradient, we need to see how they affect the whole bn_diff2 
# we saw when we have sum, we just route the previous gradients, since we are dealing with
# bn_diff2, its derivative shape is the same,(32,100) when multiplied by previous gradient
# it will be properly broadcasted and all will be fine, 
dbn_diff2 = torch.ones_like(bn_diff2) * dbn_diff2sum
#
# side note:
# if we wanted to claculate the 1/(out_preact.shape[0]-1) * bn_diff2.sum(dim=0, keepdim=True)
# without breaking it into two separate operations, we could do this. 
# !explain how to do it 
#
# dbn_diff2 = 1/(out_preact.shape[0]-1)* torch.ones_like(bn_diff2) * dvar
#
# next is               bn_diff2 = bn_diff**2 
# we need to calculate dbn_diff now, so 
# first lets print the shapes:
# print(f'{dbn_diff2.shape=} {bn_diff.shape=}')
# dbn_diff2.shape=(32, 100) 
# bn_diff.shape=(32, 100)
# so the dbn_diff needs to be 32,100. 
# the local gradient is 2*bn_diff which is 32,100, times the previous gradient 
# which is 32,100, the multipilcation would be fine because the shapes match.
dbn_diff = 2*bn_diff * dbn_diff2
#
# next is               bn_diff = out_preact - mean
# and now we are calculating the normalization once again, so lets calculate the
# gradients
# first their shapes: 
# print(f'{out_preact.shape=} {mean.shape=}')
# out_preact.shape=(32, 100) 
# mean.shape=(1, 100)
# their shape is not the same, so we need to take this into consideration.
# for dout_preact this isnot an issue, becasue dbn_diff is 32,100.
# by the way, since they were already once calculated, 
# we add this to existing gradients
# also note dout_preact is not yet finished!
dout_preact += 1.0* dbn_diff
# but for dmean, we need to sum over rows to get the right shape
dmean += -(dbn_diff).sum(dim=0, keepdim=True)
# now lets compare 
compare('dbn_diff2sum', dbn_diff2sum, bn_diff2sum)
compare('dbn_diff2',dbn_diff2,bn_diff2)
compare('dbn_diff',dbn_diff,bn_diff)
# compare('dout_preact-incom',dout_preact,out_preact)
compare('dmean',dmean,mean)
# next is        mean = out_preact.mean(dim=0, keepdim=True)
# now we need to calculate dout_preact here again and account for the mean operation here
# we are going to see how the gradient affects the whole out_preact tensor, and route 
# it properly. since out_preact is 32,100, its derivative is also 32,100. but 
# the dmean (previous gradient) is only 1,100.  the local gradient is simply 1/n
# this is where dout_preact is complete 
dout_preact += 1/len(out_preact) * dmean
# lets compare 
compare('dout_preact',dout_preact, out_preact)
# next is     out_preact = embdscat @ W1 + b1
# now the dW1, db1 and dembdscat needs to be calculated 
# first print shapes
# print(f'{dout_preact.shape=} {embdscat.shape=} {W1.shape}')
# dout_preact.shape= (32, 100)
# embdscat.shape= (32, 30)
# W1.shape = (30,100)
# so our dW1 shape is (30,100). therefore we need to Transpose embscat to get the right shape 
dW1 = embdscat.T @ dout_preact
# now lets do dembdcat, it should be 32,30. so dout_preact@W1.T should do  
dembdscat = dout_preact@W1.T
# and now lets do the db1 which is simply all ones routing the previous gradient
# since b1 shape is (1,100) we need to sum over rows
db1 = (1.0*dout_preact).sum(dim=0, keepdim=True)
# now lets compare 
compare('dw1', dW1, W1)
compare('dembdscat', dembdscat, embdscat)
compare('db1', db1, b1)
# next is           embdscat = embds.view(embds.shape[0],-1)
# now for this lets first print the shapes
# print(f'{embdscat.shape=} {embds.shape=}')
# embdscat.shape=(32, 30) 
# embds.shape=(32, 3, 10)
# as you can see, this is a matter of concatention.
# since we are dealing with view, it doesnt change anything, its just a matter of "view"
# so we can easily interpret it as the shape of embdscat! 
dembds = dembdscat.view(*embds.shape)
# thats it! we routed the gradients accordingly now lets compare
compare('dembds', dembds, embds)
# next is           embds = EMB[x_batch]
# and finally the embedding layer itself. 
# first lets print the shapes involved: 
# print(f'{EMB.shape=} {dembds.shape=} {x_batch.shape=}')
# EMB.shape=(27, 10) 
# dembds.shape=(32, 3, 10)
# x_batch.shape=(32, 3)
# we want dEMB, which should be 27,10, 
# lets also print some contents of x_batch
# print(x_batch[:5])
# tensor([[12,  9, 14],
#         [ 8,  5,  2],
#         [ 0,  0,  1],
#         [12,  9,  5],
#         [18, 13,  1]])
# now basically what is happening here is that we have a lookup table of 27,10
# which is our EMB, we have batch of idx, that each shows a character, 
# each character then gets represented as a 10 dim vector.
# each input has 3 characters, thus 3x 10 vector for each input
# on the other hand we have dembds which has a shape of (32,3,10)
# signifying we have gradients for all of the inputs, 32 examples, 3 characters
# each having the respective 10 dim vector.
# so what we need to do is to redirect these gradients back to the input, basically
# reversing the process in put, also note that we have repeated characters in the input
# (0 0, 9, etc appear multiple times therefore we need to take care of their gradients properly)
# so lets create our dEMB first
dEMB = torch.zeros_like(EMB)
# so we go for every input in our input batch
# grab the index, since each character is a unique index, 
# and also EMB is also made out of these character idxes as its index
# we can grab the respective gradient from embds and route it back to the dEMB
for i in range(x_batch.shape[0]):
    for j in range(x_batch.shape[1]):
        # get character index
        idx = x_batch[i,j]
        # use the index to grab the proper dembeding row for embedding, use the i,j indices
        # to grab the respective gradient (becasue edmbs.shape[i,j] (the first two dims) is 
        # the same as the input (x_batch, they are both (32,3)))
        # and we use += to account for repeated characters in the input
        dEMB[idx] += dembds[i,j]    

# lets compare:
compare('dEMB', dEMB, EMB)

# now this is the vectorized version : 
# basically whats happening is that our dembds has repeated enteries for each
# character, and what we need to do is to sum all of these repeated enteries
# in dembds, this way, since we have 27 unique characters, the final result
# would be 27 unique/final vectors of 10 length, and hence the dEMB 27,10 is
# created this way.
# in the semi-vectoriez implementation belowe, we first enumerate all the 
# indices (codes for our characters starting from 0-27)
# and then check in the dembds for all the entries that corrospond to this number
# and for that we use the x_batch, becasue it has the same shape as dembds
# so they are compatible. 
# we check in x_batch for the current index, and when found
# return the index(i,j), dembds use these (i,j)s and build a temporary tensor
# now containing all the vectors corrosponding to that character.
#(in fact, we create a boolean mask of (32,3), which when applied on dembds
# returns all the enteries that are set true, thus returning a list of vectors
# that we then sum over)
# in the next step, we simply sum all of these vectors together and 
# save it to the dEMB matrix under the corrosponding index!
dEMB2 = torch.zeros((27, dembds.size(2)))
for i, index in enumerate(range(27)):
    mask = (x_batch == index)
    # print(f'{mask=}')
    out = dembds[mask]
    dEMB2[i] = torch.sum( out, dim=0)

# print(f'{dEMB2.shape}')
compare('dEMB2', dEMB2, EMB)
import torch.nn.functional as F

# now the full vectorized version
dEMB3=torch.vstack([torch.sum(dembds[x_batch == i], dim=0) for i in range(27)])
# vectorized version 2 (a better version)
dEMB4=(F.one_hot(x_batch, num_classes=27).transpose(1, 2).float() @ dembds).sum(0)
# another version taken from 
dEMB5 = F.one_hot(x_batch,num_classes=27).float().view(-1, EMB.shape[0]).T @ dembds.view(-1, EMB.shape[1])
# another version
dEMB6 = torch.zeros_like(EMB).scatter_add_(0, x_batch.view(-1,1).repeat(1,dembds.shape[-1]),dembds.view(-1, dembds.shape[-1]))
#another version
dEMB7 = torch.zeros_like(EMB)
dEMB7.index_add_(0, x_batch.view(-1), dembds.view(-1, 10))
# print(dEMB3.shape)
# print(dEMB4.shape)
# if you see some of these report as not approx=True, its because of the set atol, 
# try with atol=1e-7 or 1e-6 and they all pass
compare('dEMB3', dEMB3, EMB)
compare('dEMB4', dEMB4, EMB)
compare('dEMB5', dEMB5, EMB)
compare('dEMB6', dEMB6, EMB)
compare('dEMB7', dEMB7, EMB)


# %%
# now we should be able to run optimization lets tidy things up and place them here
# to make a loop

import itertools
import random
import torch
import torch.nn.functional as F

# set the seed for determinstic output
random.seed(255)
g = torch.Generator().manual_seed(255)

names = open('./data/names.txt').read().splitlines()
character_list = sorted(set(''.join(names)))
print(f'{character_list=}')
character_list = ['.']+character_list
atoi={ch:i for i,ch in enumerate(character_list)}
# now lets create the itoa for getting back the characters from numerical codes
itoa = {v:k for k,v in atoi.items()}

def build_dataset(names:list[str], context_size:int) :
    # list for holding our dataset samples
    X:list[int] = []
    # a list for the labels which contains the next characters for each sample in X
    Y:list[int] = []
    for name in names: 
        # we build the initial sample, and remember we need numbers, not characters
        sample = [atoi['.']]*context_size
        # we append an ending symbol at the end of each name to signify where it ends
        # since our initial sample contains the initial empty symbol, we dont readd any here
        for ch in name+'.':
            idx = atoi[ch]
            X.append(sample)
            Y.append(idx)
            # update sample with the new character, move one character forward
            sample = sample[1:] + [idx]
    return X, Y

context_size = 3
X,Y = build_dataset(names, context_size)
# now lets check X and Y
print(X[:5], ''.join([itoa[c] for p in X[:5] for c in p]))
print(Y[:5], ''.join([itoa[c] for c in Y[:5]]))
# 80% for training and 10% for val and test resspectively
len_dataset = len(names)
num1 = int(0.8*len_dataset)
num2 = int(0.9*len_dataset)
# shuffle our lists we use random.shuffle to shuffle our X 
random.shuffle(names)
names_tr = names[:num1]
names_val = names[num1:num2]
names_test = names[num2:]
# now lets create our dataset using names_tr
#
X_tr, Y_tr = build_dataset(names_tr, context_size)
# now lets check X and Y
print(X[:5], ''.join([itoa[c] for p in X[:5] for c in p]))
print(Y[:5], ''.join([itoa[c] for c in Y[:5]]))
# now lets convert them to tensor
X_tr = torch.tensor(X_tr)
Y_tr = torch.tensor(Y_tr)

vocab_size = len(character_list)
embedding_size = 10

EMB = torch.randn(size = (vocab_size, embedding_size), generator=g)
hidden_size = 100
W1 = torch.randn(size=(context_size * embedding_size, hidden_size), generator=g)
b1 = torch.randn(size=(1,hidden_size), generator=g)
W2 = torch.randn(size=(hidden_size, vocab_size), generator=g)
b2 = torch.randn(size=(1,vocab_size), generator=g)
bn_gamma_gain = torch.randn(size=(1,hidden_size), generator=g)
bn_beta_bias = torch.randn(size=(1,hidden_size), generator=g)

# now lets create a parameters list to hold all the parameters for optimization
parameters = [EMB, W1, b1, W2, b2, bn_gamma_gain, bn_beta_bias]
    
# OK now lets do a forward pass 
iter_max = 200_000
batch_size = 32
eps= 1e-6
running_mean = torch.zeros_like(b1)
running_var = torch.ones_like(b1)
momentum = 0.1
losses = []
for i in range (iter_max):
    batch_idxs = torch.randint(0, len(X_tr), size=(batch_size,), generator=g)# shape: [32]
    x_batch = X_tr[batch_idxs]
    embds = EMB[x_batch]
    embdscat = embds.view(embds.shape[0],-1)
    out_preact = embdscat @ W1 + b1
    mean = out_preact.mean(dim=0, keepdim=True)
    bn_diff = out_preact - mean 
    bn_diff2 = bn_diff**2 
    bn_diff2sum = bn_diff2.sum(dim=0, keepdim=True)
    var = 1/(out_preact.shape[0]-1) * bn_diff2sum
    out_normalized = out_preact - mean 
    
    # calculating running mean and var for test time
    running_mean = (1-momentum)*running_mean + momentum*mean
    running_var = (1-momentum)*running_var + momentum*var
    
    var_sq = (var+eps)**0.5
    var_sq_inv = var_sq**-1 
    x_hat = out_normalized * var_sq_inv
    out_bn = bn_gamma_gain * x_hat + bn_beta_bias
    out_tanh = torch.tanh(out_bn)
    logits = out_tanh @ W2 + b2
    logits_max = logits.max(dim=1, keepdim=True).values
    logitsnorm = logits - logits_max
    logcounts = logitsnorm.exp()
    logcountsum = logcounts.sum(dim=1, keepdim=True)
    logcountsum_inv = logcountsum**-1
    probs = logcounts * logcountsum_inv
    logprobs = probs.log()
    loss = -logprobs[torch.arange(0,len(batch_idxs)), Y_tr[batch_idxs]].mean()

    # calculate the gradients manually 
    
    dlogprobs = torch.zeros_like(logprobs)
    dlogprobs[range(0,len(batch_idxs)), Y_tr[batch_idxs]] = -1/len(batch_idxs)
    dprobs = torch.zeros_like(probs)
    dprobs[...] = (1/probs) * dlogprobs 
    dlogcounts = logcountsum_inv * dprobs
    dlogcountsum_inv = (logcounts * dprobs).sum(dim=1, keepdim=True)
    dlogcountsum = -1*logcountsum**(-2) * dlogcountsum_inv
    dlogcounts += torch.ones_like(logcounts) * dlogcountsum
    dlogitsnorm = logcounts * dlogcounts
    dlogits = 1*dlogitsnorm
    dlogits_max = (-1*dlogitsnorm).sum(dim=1, keepdim=True) 
    dlogits += torch.nn.functional.one_hot(logits.max(dim=1).indices, num_classes=logits.shape[-1]) * dlogits_max
    dout_tanh = dlogits@W2.T
    dW2 = out_tanh.T@dlogits
    db2 = (1*dlogits).sum(dim=0,keepdim=True)
    dout_bn = (1-out_tanh**2)* dout_tanh
    dbn_gamma_gain = (x_hat*dout_bn).sum(dim=0,keepdim=True)
    dx_hat = bn_gamma_gain*dout_bn
    dbn_beta_bias = (1*dout_bn).sum(dim=0,keepdim=True)
    dout_normalized = var_sq_inv*dx_hat
    dvar_sq_inv = (out_normalized*dx_hat).sum(dim=0, keepdim=True)
    dvar_sq = -1*var_sq**-2 * dvar_sq_inv
    dvar = (dvar_sq * (0.5*(var+eps)**(0.5-1.0))).sum(dim=0, keepdim=True)
    dout_preact = torch.ones_like(out_preact) * dout_normalized
    dmean = -1*(torch.ones_like(mean)*dout_normalized).sum(dim=0, keepdim=True) 
    dbn_diff2sum = 1/(out_preact.shape[0]-1) * dvar
    dbn_diff2 = torch.ones_like(bn_diff2) * dbn_diff2sum
    dbn_diff = 2*bn_diff * dbn_diff2
    dout_preact += 1.0* dbn_diff
    dmean += -(dbn_diff).sum(dim=0, keepdim=True)
    dout_preact += 1/len(out_preact) * dmean
    dW1 = embdscat.T @ dout_preact
    dembdscat = dout_preact@W1.T
    db1 = (1.0*dout_preact).sum(dim=0, keepdim=True)
    dembds = dembdscat.view(*embds.shape)
    dEMB=torch.vstack([torch.sum(dembds[x_batch == i], dim=0) for i in range(27)])
    # gradients calculation

    # parameters = [EMB, W1, b1, W2, b2, bn_gamma_gain, bn_beta_bias]
    dparameters=[dEMB, dW1, db1, dW2, db2, dbn_gamma_gain, dbn_beta_bias]

    lr = 0.1 if i<=100_000 else 0.01
    for param,grad in zip(parameters, dparameters):
        param.data += -lr * grad
    
    losses.append(loss.item())
    if i%10000==0:
        print(f'{loss.item()}')
        
print(f'loss:', sum(losses)/len(losses))
# which prints 
# 11.806575775146484
# 2.535858154296875
# 2.4068679809570312
# 2.406609535217285
# 2.328763246536255
# 2.667058229446411
# 2.191136598587036
# 2.2679288387298584
# 2.591965913772583
# 1.9428260326385498
# 2.1402201652526855
# 2.166907548904419
# 2.3032281398773193
# 2.2943642139434814
# 2.177385091781616
# 2.4250869750976562
# 2.1525397300720215
# 2.0469741821289062
# 2.3654978275299072
# 2.33795428276062
# loss: 2.322884666481614
# which if we compare with the pytorch loss : 2.322884672728181 we see 
# they are nearly identical (the difference is 0.0000000062 or 6.246566819356758e-09
# which is basically zero)
#%%
# now lets sample from it 
# for sampling we would feed the '...' as input and keep creating 
# before going on, we need to calculate the datasetmean/var as well
for i in range (10):
    input = [0]*3
    input_tensor = torch.tensor(input).reshape(1,-1)
    # print(f'{input_tensor.shape=}')
    chsr = ''
    while True:
        embd = EMB[input_tensor].reshape(input_tensor.shape[0],-1)
        # print(f'{embd.shape=}')
        out_preact = embd @ W1 + b1
        # print(f'{out_preact.shape=}')
        # ############## batchnorm ##############
        # for test time, we need to calculate the mean/var of the the whole
        # dataset to use here 
        # mean = out_preact.mean(dim=0, keepdim=True)
        # bn_diff = out_preact - mean 
        # bn_diff2 = bn_diff**2
        # bn_diff2sum = bn_diff2.sum(dim=0, keepdim=True)
        # var = 1/(out_preact.shape[0]-1) * bn_diff2sum
        mean = running_mean
        var = running_var
        out_normalized = out_preact - mean 
        var_sq = (var+eps) ** 0.5
        var_sq_inv = var_sq ** -1 
        x_hat = out_normalized * var_sq_inv
        out_bn = bn_gamma_gain * x_hat + bn_beta_bias
        ########################################
        out_tanh = torch.tanh(out_bn)
        logits = out_tanh @ W2 + b2
        
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
