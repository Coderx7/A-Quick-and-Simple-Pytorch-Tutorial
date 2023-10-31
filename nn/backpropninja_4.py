# in the name of God the most compassionate the most merciful 
# in this section we will try to code the backward pass for our 
# initial makemore example, basically we will implement the backwardpass
# for a 3 layer nn (4 layer including an embedding layer)
# this is to get familiar with the tensor operations in a backward pass
# and get an intuitive idea of how stuff works under the hood
# things such as broadcasting, mean, max, etc will be worked out inshaallah.
#%% 
import itertools

# we start off by reading the data and implementing our simple nn with emebdings
names = open('./names.txt').read().splitlines()
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
# we are going to create a model which accepts the first 3 characters, and returns the 4th
# here we are creating context, given a context of 3 characters, provide us with the next
# the starting point would be an empty string, dentoting equal likelihood for any given name
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
## now lets check X and Y
# print(X[:5], ''.join([itoa[c] for p in X[:5] for c in p]))
# print(Y[:5], ''.join([itoa[c] for c in Y[:5]]))
# now lets convert this into a function for easier use in case we decided to have 
# different splits like training, val, test
def build_dataset(names:list[str], context_size:int) -> tuple(list[int]):
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
# now lets create tensors our of our lists 
# but before that lets create some splits! 
# 80% for training and 10% for val and test resspectively
len_dataset = len(names)
num1 = int(0.8*len_dataset)
num2 = int(0.9*len_dataset)
# shuffle our lists we use random.shuffle to shuffle our X 
import random 
random.shuffle(names)
names_tr = names[:num1]
names_val = names[num1:num2]
names_test = names[num2:]
# or we could do 
# tr_cnt = int(0.8*len_dataset)
# val_cnt = len_dataset - tr_cnt//2
# names_tr = names[:tr_cnt]
# names_val = names[tr_cnt: tr_cnt+val_cnt]
# names_test = names[tr_cnt+val_cnt:]
# assert sum(map(len,[names_tr,names_val,names_test])) == len(names) , 'must match'
#
# now lets create our dataset using names_tr
#
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
import torch
# before we go on, lets set a manual seed for deterministic outcome
torch.manual_seed(255)

vocab_size = len(character_list)
embedding_size = 10

EMB = torch.randn(size = (vocab_size, embedding_size))
# next we should have a linear layer after our embedding layer, to work on the embeddings
# since our input is 3 numbers, and each number has an embedding vector of size 10, our
# linear layer needs an in_features =30
hidden_size = 100
W1 = torch.randn(size=(context_size * embedding_size, hidden_size))
b1 = torch.randn(size=(1,hidden_size))
# we need a second layer to give us the final probablities for the next character
W2 = torch.randn(size=(hidden_size, vocab_size))
b2 = torch.randn(size=(1,vocab_size))
# we want to add a batchnorm layer, so we need to create the parameters for it as well
# batchnorm needs a gamma, a beta as the only two learnable parameters
# we usually start gamma_gain as ones, and beta_bias as zeros so the start off with 
# an initial guassian distribution, but here for unsmasking our possible errors we use
# randn
bn_gamma_gain = torch.randn(size=(1,hidden_size))
bn_beta_bias = torch.randn(size=(1,hidden_size))

# now lets create a parameters list to hold all the parameters for optimization
parameters = [EMB, W1, b1, W2, b2, bn_gamma_gain, bn_beta_bias]
# before starting the optimization lets enable their gradients
for param in parameters:
    param.requires_grad = True 
    
# OK now lets do a forward pass 
iter_max = 200_000
batch_size = 32
eps= 1e-6
for i in range (iter_max):
    # lets create a random batch of the input
    batch_idxs = torch.randint(0, len(X_tr), size=(batch_size,))# shape: [32]
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
    # calculate the mean/var for all samples individually i.e. along 0th dimension
    mean = out_preact.mean(dim=0, keepdim=True)
    var = out_preact.var(dim=0, keepdim=True)
    # normalize the input /add eps to prevent division by zero
    # x_hat = (out_preact-mean)/(var+eps)**0.5
    # but lets divide it into more parts so calculating the gradients later on is easier
    out_normalized = out_preact - mean 
    var_sq = (var+eps)**0.5
    x_hat = out_normalized/var_sq
    # finally apply the gamma_gain and bias on the x_hat
    out_bn = bn_gamma_gain * x_hat + bn_beta_bias
    # apply tanh
    out_tanh = torch.tanh(out_bn)
    # now lets apply the second layer
    logits = out_tanh @ W2 + b2
    # lets implement the crossentropy loss
    # which is the negative log likelihood 
    # which is we must treat our output as logcounts
    # so we do exp() to make them non negative
    # and then normalize them by their total which would be softmax
    # but to make it numerically stable, we subtract them from their max
    # that would be our probabablities, and then we would look at the
    # predictions for for the right classes and take their log and mean
    # becasue we need the negative of log of (likelihood/probablity) 
    # since we have several samples, we either need to sum or average
    # since average is more customary because it results in a smaller loss
    # we use that 
    # note that our logits shape is (32,27), 32 being the batch size, and 27 being the
    # vocab size, or in other words, our number of classes. so when we want to calculate 
    # max, we need to have a max for each sample, (note that we are going to treat
    # each column as a class probablity, so their total needs to sum to 1, therefore
    # we dont care about other rows, when we are normalizing or doing anything, its
    # sample wise and involves all classes for that row/sample), we have 32 input examples, 
    # and therefor for each example, we see, whats the maximum value among classes,
    # for that input. note that the keepdim=True is needed becasue it makes it a 
    # row vector (32,1), which when broadcasted, would allow us to subtract the values
    # in each column from the maximum for that specific row!
    # also note that torch.max, or tensor.max, returns a tuple of indexes and values
    # but since we want the values only, we used values (also without using .values 
    # property, we had to set the require_grads on logits_max explicityly otherwise
    # it wouldnt get the gradients)
    logits_max = logits.max(dim=1, keepdim=True).values
    # subtract from the max for each sample
    logitsnorm = logits - logits_max
    # treat them as log counts (we do a exp to make them all non-negative)
    logcounts = logitsnorm.exp()
    # normalize and get a probablity distribution
    # sum along the columns, so each value is normalized properly 
    # remember that we have 27 column, each representing 1 class, 
    # so their total count must sum to 1, so we sum along the columns
    # to get the total and then each column divided by that total gives us
    # propabalities
    # probs = logcounts/logcounts.sum(dim=1, keepdim=True)
    # for easier calculation of gradients, lets split the operations into separate ones
    logcountsum = logcounts.sum(dim=1, keepdim=True)
    # probs = logcounts /logcountsum
    logcountsum_inv = logcountsum**-1
    probs = logcounts * logcountsum_inv
    # instead of division, lets convert that into multiplication!
    # logcountsum_inv = logcountsum**-1
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
    # to check our manual gradients
    for param in itertools.chain(parameters,[logprobs, probs, logcounts,logcountsum,logcountsum_inv,logitsnorm,
                                             logits, logits_max, out_bn,out_tanh,x_hat, out_preact,
                                             var, mean, out_preact,var_sq,out_normalized, embds, embdscat]):
        param.retain_grad()
    
    loss.backward()
    # break to be able to calculate the backward pass ourselves
    break

    lr = 0.1
    for param in parameters:
        param.data += -lr * param.grad
    
#%%
# lets create a function for comparing our gradients with pytorchs
def compare(str, our_grads, ptensor):
    """compares our manually calculated gradients 
    against the pytorchs automatically calculated 
    gradients.

    Args:
        str (_type_): a string to describe the tensors being compared
        our_grads (_type_): our manually calculated gradients
        ptensor (_type_): pytorch tensors for which we have calculated the gradient
    """
    pgrad = ptensor.grad
    exactly_same = torch.all(our_grads == pgrad).item()
    approximately_same = torch.allclose(our_grads, ptensor.grad)
    difference = (our_grads - pgrad).abs().max().item()
    print(f'{str:12} | exact: {exactly_same} | approx: {approximately_same} | diff: {difference}')

# now the first operation that we have is probs
# our first variable to get its grads is the -logprobs.mean()
# so we write 
# dlogprobs = ???
# so what should we write here? how to go about getting the grads for a mean() operation?
# one good tip to tackle these is to comeup with a much simpler example and see how
# we can get the deravitaive for that. to this end, suppose we have sth like this:
# we know that mean is simply a sum of several numbers divided by their count, like this:
# y = -(a + b + c )/3
# now if we were to calculate dd/da what would that be?
# we can simplify the previous line like this: 
# y= -1/3a + -1/3b + -1/3c 
# right? we just expanded it. now what would be the dd/da? 
# dy/da = -1/3
# what about dd/db?
# dy/db = -1/3
# and so on. so here we had 3 numbers, so it was -1/3, if we had more, we would write more
# so its basically -1/n, n being the number of all numbers involved. 
# so whats the n here? the shape of logpros is (32,27) 32 sample, and 27 classes
# and we know that we must calculate the gradient for every elements in our tensor
# so what should do now? lets expand on the line that creates logprobs
# its loss=-logprobs[torch.arange(0,len(batch_idxs)), Y_tr[batch_idxs]].mean()
# what are we doing exactly? 
# logprobs[torch.arange(0,len(batch_idxs)), Y_tr[batch_idxs]] simply is 
# two part, the first part simply acts like a loop, at each time it selects a single row
# of logprobs, and the second part simply picks the correct column/class, together they
# give us the true class for each sample. 
# if we print Y_tr[batch_idxs] we will see sth like this: 
# tensor([ 8, 14,  0, 12, 17,  9,  0, 25,  5,  0,  5,  0, 25,  1,  3,  2,  0, 11,
#         14,  0,  1, 26,  2,  9,  8,  9,  1, 14,  5,  9, 14,  1])
# each of these are index to the correct class, what this is saying is, 
# the correct class for sample 1 is 8, the correct class for sample 2 is 14, and so on
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
# (side note: if we tried to print(probs[torch.arange(0,len(batch_idxs)), Y_tr[batch_idxs]].grad)
# we'll get this error : 
#  UserWarning: The .grad attribute of a Tensor that is not a leaf Tensor is being accessed. 
# Its .grad attribute won't be populated during autograd.backward(). If you indeed want the
# .grad field to be populated for a non-leaf Tensor, use .retain_grad() on the non-leaf Tensor.
# If you access the non-leaf Tensor by mistake, make sure you access the leaf Tensor instead. 
# See github.com/pytorch/pytorch/pull/30531 for more informations. 
# (Triggered internally at /croot/pytorch_1686931851744/work/build/aten/src/ATen/core/TensorBody.h:486.)
# basically that intermidiate operation(slicing, etc involved) is not available to us here, because it was temporarily
# made and assigned. 
# so to see the gradients, we have to see the whole tensor.grad and then compare ours to it
# that should be fine.(because we need to calculate the grdients for all elements anyway,
# and for comparison that should suffice, however this seems like a bug to me, the grads are 
# already calculated, so pytorch should be able to return them, but instead it executes the logic
# that should only be called when calculating the gradients) anyway!
# but what should we do now? how can we calculate the gradients of log(x)?
# gradients of log(x) is 1/x (for reminder see : https://www.intmath.com/differentiation-transcendental/5-derivative-logarithm.php)
# so if we had for example 
# y = [log a , logb , logc] 
# dy/da = 1/a
# so in our case it would be 
# dprobs[torch.arange(0,len(batch_idxs)), Y_tr[batch_idxs]] = 1/probs[torch.arange(0,len(batch_idxs)), Y_tr[batch_idxs]]
# and what about the rest of the elements? like before, they would be zero
# so we would have 
# dprobs = torch.zeros_like(probs)
# # since we have chain rule, we multiple the previous gradients times this one 
# dprobs[torch.arange(0,len(batch_idxs)), Y_tr[batch_idxs]] = dlogprobs * 1/probs[torch.arange(0,len(batch_idxs)), Y_tr[batch_idxs]]
# compare('drpobs',dprobs, probs)
# and we see this is exactly the answer
# print(f'dprobs:{dprobs[torch.arange(0,len(batch_idxs)), Y_tr[batch_idxs]]}')
# print(f'probs:{probs.grad[torch.arange(0,len(batch_idxs)), Y_tr[batch_idxs]]}')
#
# now for dprobs, we already saw the second scenario, so we know hwat to do 
dprobs = torch.zeros_like(probs)
# the derivative of log(x) is 1/x, and since its chain rule, we need to multiply
# the previous gradients here as well (also recall that each function has a local gradient
# here our function is log, and its local derivative is 1/probs, we need to multiply that
# to the gradient from previous operation.)
# also on a sidenote: what this line is effectively doing, is its boosting the gradients 
# of the values that had low probabilities (recall that when probs are correct, its 1
# and 1/1 is 1 basically passing through the dlogprobs, but if the probs are wrong, they
# would be small and 1/small_num basically is a larger number, now being multiplied by
# dlogprobs, making them bigger)
dprobs[...] = (1/probs) * dlogprobs 
# lets compare the two ... which is 100% ok!
compare('dprobs',dprobs, probs)
#probs = logcounts * logcountsum_inv
# lets calculate dlogcounts
# first shapes
# dprobs.shape=(32, 27) previous gradient
# logcounts.shape = (32,27)
# logcountsum_inv.shape =(32, 1)
dlogcounts = dprobs * logcountsum_inv
#
dlogcountsum_inv = (logcounts*dprobs).sum(dim=1, keepdim=True)
# print(f'{dlogcountsum_inv.shape=}')
#
#logcountsum_inv = logcountsum**-1
# now dlogcountsum, first shapes : 
# logcountsum (32,1)
# dprobs = (32,27)
# so the dlogcountsum must be (32,1) so we need to sum over columns
dlogcountsum = ((-1*logcountsum**-2)*dlogcountsum_inv).sum(dim=1,keepdim=True)

# so far so good, now we reach to 
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
compare('dlogcountsum', dlogcountsum, logcountsum)
# and now for logcounts we still need more to do as up there we have 
# logcountsum = logcounts.sum(dim=1, keepdim=True)
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
# also note that dlogcounts was once calculated, so this needs to be added to the previous result
dlogcounts += torch.ones_like(logcounts) * dlogcountsum
compare('dlogcounts', dlogcounts, logcounts)
# next we have logcounts = logitsnorm.exp()
# the deravative of logitsnorm.exp() will be logitsnorm.exp() or in otherwords
# the logcounts itself, so we may as well use that instead of recalculating it here again
# and since its the chainrule, we multiply our local gradient with the previous gradient
dlogitsnorm = logcounts * dlogcounts
compare('dlogitsnorm', dlogitsnorm, logitsnorm)
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
dlogits = 1.0*dlogitsnorm
# and for logits_max 
dlogits_max = (-1.0*dlogitsnorm).sum(dim=1, keepdim=True) 
compare('dlogits', dlogits, logits)
compare('dlogits_max', dlogits_max, logits_max)
# side note: we said earlier that the reason we take the max and subtract the logits from it
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
# now the next line is logits_max = logits.max(dim=1, keepdim=True).values
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
dlogits += torch.zeros_like(logits)
dlogits[range(0,len(logits_max)), logits.max(dim=1,keepdim=True).indices] += 1*dlogits_max
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
# (note the + ,whenever theres + the gradient is just rounted)
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
dout_bn = (1-(out_tanh**2))* dout_tanh
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
# but since the shape needs to match, we instead use x_hat.T to get (100,32)*(32,100)
# to get 100x100 and then sum over rows to get a single row of (1,100)
dbn_gamma_gain = (x_hat.T@dout_bn).sum(dim=0,keepdim=True)
# lets check 
compare('dbn_gamma_gain', dbn_gamma_gain, bn_gamma_gain)
# and likewise for x_hat we would have a dx_hat of shape (32,100)
# it would be bn_gamma_gain but it needs to be broadcasted, since its 
# (1,100), it will be replicated along the rows to gte (32,100)
# we let that be handled by the multiplication by previous gradient which is dout_bn
# which is has the shape of 32,100.
dx_hat = dout_bn@bn_gamma_gain.T
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
# x_hat = out_normalized/var_sq
# lets see what we have, out_normalized,var_sq and we need to calculate their gradienst
# first lets print their shapes 
# print(f'{out_normalized.shape=}, {var_sq.shape=}, {dx_hat.shape=}')
# their shape is :
# out_normalized.shape=(32, 100) 
# var_sq.shape=(1, 100)
# dx_hat.shape=torch.Size([32, 1])
# and we know that dout_normalzied must have the shape of (32,100)
# but the var_sq is 1,100, and dx_hat is (32,1) so we need to do:
dout_normalized = dx_hat@(1.0/var_sq)
# lets compare
compare('dout_normalized', dout_normalized, out_normalized)
# and for dvar which should have the shape(1,100), and also needs to be multiplied by
# the previous gradient (dx_hat) which is (32,1). the result would be 32x100, so we need
# to sum over rows to get (1,100)
dvar_sq = (dx_hat@(1.0/((var+eps)**0.5)**2)).sum(dim=0,keepdim=True)
compare('dvar_sq', dvar_sq, var_sq)
#
# next is var_sq = (var+eps)**0.5
# and now we need to calculate dvar 
# first print the shapes: 
# print(f'{var.shape=} {dvar_sq.shape=}')
# var.shape=(1, 100) 
# dvar_sq.shape=(1, 100)
# so dvar shape should be (1,100) as well, so we transpose one to get 100x100
# and then need to sum over rows to get 1,100
dvar = (dvar_sq.T @ (0.5*(var+eps)**(0.5-1.0))).sum(dim=0, keepdim=True)
# lets compare
compare('dvar', dvar, var)
#and next is  out_normalized = out_preact - mean 
#
#
# next is     var = out_tanh.var(dim=0, keepdim=True)
#
#
# next is     mean = out_tanh.mean(dim=0, keepdim=True)
#
#
# next is     out_tanh = torch.tanh(out)
#
#
# next is     out = embdscat @ W1 + b1
#
#
# next is   embdscat = embds.view(embds.shape[0],-1)
#
#
# next is embds = EMB[x_batch]