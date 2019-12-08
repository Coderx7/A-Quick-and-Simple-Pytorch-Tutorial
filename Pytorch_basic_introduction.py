#%% [markdown] 
# in the name of God the most compassionate the most merciful
# Pytorch basics : introduction on tensors
import torch 
import numpy as np 


# Here we are going to see what torch is and how it is similar to numpy!   
# Basically torch is a framework that is used for training and working with deep neural networks.
# What is Pytorch then? 
# PyTorch is a Python package that provides two high-level features:
#   1.Tensor computation (like NumPy) with strong GPU acceleration
#   2.Deep neural networks built on a tape-based autograd system
# Basically PYtorch is the python wrapper for torch! 
# You can reuse your favorite Python packages such as NumPy, SciPy and Cython to extend PyTorch when needed. 
# in this section we are going to have an introduction concerning torch as a numpy replacement! in working with tensors
# lets see how we can ue pytorch in this sense !
#%%
# what is a tensor? 
# a tensor is a general name given to arrays/vectors/matrices. 
# a vector is simply a 1 dimensional tensor. 
# a matrix is simply a 2 dimensional tensor. 
# and so on. 
# so if we have a matrix that has 3 channels(RGB for example), that is a 3d tensor
# I guess you get the idea. whenever we talk about tensor, remember its just another name
# for an array (we call a 1d array a vector, a 2d array a matrix so you can see tensor as an array!)
# Tensors are the fundamental data structure used for neural networks so in Pytorch we will be dealing with
# them a lot! (nearly all of the time!)

# creating new tensors 
# create a tensor of size (5) , (2, 2), (3, 5, 6) with all zeros, ones, and random values, no values
t = torch.Tensor(size=(1,))
t1_zeros = torch.zeros(size=(5,))
t1_ones = torch.ones(size=(2,2))
t1_rand = torch.rand(size=(2,2,2))
t1_empt = torch.empty(size=(2,2,2))


print(f'zeros: {t1_zeros}')
print(f'ones: {t1_ones}')
print(f'rand: {t1_rand}')
print(f'empt: {t1_empt}')
print(t)


# what if we want our tensors to have specific data! easy, we can send our data using list!
# or a numpy array! 
# here we are creating a tensor from a list of numbers (1, 2, 3, 4)!
data_1 = torch.tensor([1, 2, 3, 4])
print(f'{data_1}')

# using an numpy array 
array_np = np.random.rand(4)
data_2 = torch.tensor(array_np)
print(f'np: {array_np}')
print(f'{data_2}')


# there is a difference in the number of decimals, we can use printoptions to get what we want!
torch.set_printoptions(precision=8)
print(f'np: {array_np}')
print(f'{data_2}')

# lets go to defaul! we can use default profile and just go back to defaul!
# as it turns out, we can use other profiles (short, full) as well for our uses!
torch.set_printoptions(profile='default')
print(f'np: {array_np}')
print(f'{data_2}')

# by the way we can directly create a new tensor from a numpy array! like  this
tensor_from_numpy = torch.from_numpy(array_np)
print(f'tensor_from_numpy: {tensor_from_numpy}')


# intrestingly we can get the underlying numpy array from a tensor, using. numpy() method!
print(f'data_2(torch tensor): {data_2}')
print(f'data_2.numpy()(converted to numpy!): {data_2.numpy()}')
#%%

# Ok, so we just learnt how to create tensors. in the beginning we said we can leverage GPU! 
# so lets see how we can do that! but before that, we need to check if GPU support is available to us!
print(f'is GPU enabled? : {torch.cuda.is_available()}')
# so as it turns out, all tensors, can have two modes, they can either be on CPU or GPU
# the tensors we created so far are in CPU mode. to see on which device our tensors are created
# we can simply use the device property! 
print(f'data_2 is created on : {data_2.device.type}')

# so how do we move or define a new tensor or an existing one from one device to another?
# we can easily do that using to() or cpu() or cuda() methods. 
# here how we do it. 
# puts the tensor on gpu! 
data_2 = data_2.cuda()
print(f'data_2 device : {data_2.device.type}')

# puts the tnesor back to cpu!
data_2 = data_2.cpu()
print(f'data_2 device : {data_2.device.type}')

# we can do better, and based on the system for example decide if a tensor can use gpu or not!
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_2 = data_2.to(device)
print(f'data_2 device : {data_2.device.type}')
#
# now if we want to create a tensor on specific device in the definition we do 
# the 0 denotes the specific gpu on the system. if you have only 1, you can simply 
# use cuda, if you want to use a specific gpu, you use its respective index!
data_3 = torch.rand(size=(2,2), device='cuda:0') # or device = 0
print(f'data_3 device: {data_3.device}')
# how do we get how many gpus are available on the system? easy we do : 
from torch import cuda
gpu_count = cuda.device_count()
print(f'all gpus available : {gpu_count}')
print(f'gpu name : {cuda.get_device_name(0)}')
print(f'gpu capability : {cuda.get_device_capability(0)}')
# you can find all intresting methods for your use here https://pytorch.org/docs/stable/cuda.html


#%%

# Ok, now what if we have a tensor that is already on a specific device(it can be cpu or a gpu)
# and also has a specific datatype!( all of our tensors can have dtype! the default is float32!)
# in such cases, we can simply use the torch.*_new methods. lets see 
tensor_special = torch.rand(size=(2,2), device = 'cuda', dtype=torch.float16)
print(tensor_special)

# now lets create a new tensor from this one that is both on cuda and uses float16!!
new_tensor_ones = tensor_special.new_ones(size=(2,2))
print(new_tensor_ones)
# we have other options such as new_tensor, new_empty, new_full, new_zeros
new_tensor_full = tensor_special.new_full(size=(2,2), fill_value=.3)
print(new_tensor_full)
# we create a new tensor with the same dtype and device 
new_tensor_newtensor = tensor_special.new_tensor(np.random.uniform(-1,1,size=(2,2)))
print(new_tensor_newtensor)

#%%
# now  that we learnt how to create a new tensor, initialize a tensor, specify different dtypes, device, etc
# lets work on addition, subtraction, multiplication, negation, transpose, and the likes 
# for adding two tensors, 
# either a tensor should be  as scaler, or has 1 dimension in comon
t1 = torch.tensor([1., 2., 3., 4.])
t2 = torch.tensor([[10.,10.,10.,10.],[10.,10.,10.,10.]])
t3 = torch.tensor([0.5])
print(f't1 = {t1}')
print(f't2 = {t2}')
print(f't3 = {t3}')
print(f't1 + t2 =\n {t1 + t2}')
print(f't1 + t3 =\n {t1 + t3}')
#%%
# adding and subtracting is really obvious, but when it comes to multiplilication we have several options!
# mm, matmul, bmm 
# basically mm and matmul are kinda the same, they both do multipilication, the difference is, 
# the matmul does the broadcasting as well while the mm doesnt. 
# it is recommened to use mm, becasue if the dimensions dont match, you'll face an error and know where to fix!
# however, in matmul, when the dimensions dont match, it may broadcast and thus dont give you an error while
# the result may very well be wrong! so to be on the safe side, always try to use mm!
# bmm is mm with batches. basically if you do want to multiply several samples of two tensors 
# you can use bmm. we will see how this works later on so dont worry about it! 

# torch.matmul(tensor1, tensor2, out=None) → Tensor
# Matrix product of two tensors.
# The behavior depends on the dimensionality of the tensors as follows:
#    If both tensors are 1-dimensional, the dot product (scalar) is returned.
#    If both arguments are 2-dimensional, the matrix-matrix product is returned.
#    If the first argument is 1-dimensional and the second argument is 2-dimensional,
#        a 1 is prepended to its dimension for the purpose of the matrix multiply. 
#        After the matrix multiply, the prepended dimension is removed.
#    If the first argument is 2-dimensional and the second argument is 1-dimensional, 
#        the matrix-vector product is returned.
#    If both arguments are at least 1-dimensional and at least one argument is N-dimensional
#        (where N > 2), then a batched matrix multiply is returned. If the first argument is
#        1-dimensional, a 1 is prepended to its dimension for the purpose of the batched matrix
#        multiply and removed after. If the second argument is 1-dimensional, a 1 is appended to
#        its dimension for the purpose of the batched matrix multiple and removed after.
#        The non-matrix (i.e. batch) dimensions are broadcasted (and thus must be broadcastable).
#        For example, if tensor1 is a (j×1×n×m)(j \times 1 \times n \times m)(j×1×n×m) tensor and 
#        tensor2 is a (k×m×p)(k \times m \times p)(k×m×p) tensor, out will be an
#        (j×k×n×p)(j \times k \times n \times p)(j×k×n×p) tensor.
# Note
#     The 1-dimensional dot product version of this function does not support an out parameter.

data_1 = torch.rand(size=(2,3))
data_2 = torch.rand(size = (2,))
# pay careful attention to the dimensions and how the multiplication is carried out!
# data2 * data1
data_3 = torch.matmul(data_2, data_1)
print(f'data_2(2,) * data_1(2x3): {data_3}')
# as you just saw, the data_2 was broadcasted so it can be multiplied by data_1
# data_2 was 1d, and it was treated as (1,2) so the dimensions can between two tensors
# are valid. thus the output is a 1x3 tensor! 
# this is how we do transpose!
data_4 = torch.matmul(data_1.t(), data_2)
print(f'data_1.t()(3x2) * data_2(2,): {data_4}')
# now in this example, the data_2 again is broadcasted ans this time  
# it is treated as (2x1) tensor so the dimensions between tensors are valid 
# as you can see the output is a tensor of 3x1.

# we can do all of these using mm! 
print('using torch.mm:')
# mm is short for matrix multiply, so all dimensions must be specified!
# unlike matmul, there is no broadcasting going on here! we must specify all dimensions ourselevs
# thats why we used .view() to reshape our tensor to the form it needs to be to have a proper multiplication!
data_3_2 = torch.mm(data_2.view(1,2), data_1)
print(f'data_2(1x2) * data_1(2x3): {data_3_2}')
# this is how we do transpose!
data_4_2 = torch.mm(data_1.t(), data_2.view(2,1))
print(f'data_1.t()(3x2) * data_2(2x1): {data_4_2}')


# if you want to know more about boradcasting in Pytorch read more here : 
# https://pytorch.org/docs/stable/notes/broadcasting.html#broadcasting-semantics 

# In short, if a PyTorch operation supports broadcast, then its Tensor arguments
# can be automatically expanded to be of equal sizes (without making copies of the data).

# Two tensors are “broadcastable” if the following rules hold:
#     Each tensor has at least one dimension (like what we just saw in our example above!)
#     When iterating over the dimension sizes, starting at the trailing dimension,
#     the dimension sizes must either be equal, one of them is 1, or one of them does not exist.
x=torch.empty(5,7,3)
y=torch.empty(5,7,3)
# same shapes are always broadcastable (i.e. the above rules always hold)

x=torch.empty((0,))
y=torch.empty(2,2)
# x and y are not broadcastable, because x does not have at least 1 dimension

# can line up trailing dimensions
x=torch.empty(5,3,4,1)
y=torch.empty(  3,1,1)
# x and y are broadcastable.
# 1st trailing dimension: both have size 1
# 2nd trailing dimension: y has size 1
# 3rd trailing dimension: x size == y size
# 4th trailing dimension: y dimension doesn't exist

# but:
x=torch.empty(5,2,4,1)
y=torch.empty(  3,1,1)
# x and y are not broadcastable, because in the 3rd trailing dimension 2 != 3

# Backwards compatibility
# Prior versions of PyTorch allowed certain pointwise functions to execute on 
# tensors with different shapes, as long as the number of elements in each tensor was equal. 
# The pointwise operation would then be carried out by viewing each tensor as 1-dimensional. 
# PyTorch now supports broadcasting and the “1-dimensional” pointwise behavior is considered 
# deprecated and will generate a Python warning in cases where tensors are not broadcastable, 
# but have the same number of elements.
# Note that the introduction of broadcasting can cause backwards incompatible changes in the 
# case where two tensors do not have the same shape, but are broadcastable and have the same 
# number of elements. For Example:
# torch.add(torch.ones(4,1), torch.randn(4))
# would previously produce a Tensor with size: torch.Size([4,1]), but now produces a Tensor 
# with size: torch.Size([4,4]). In order to help identify cases in your code where backwards 
# incompatibilities introduced by broadcasting may exist, you may set torch.utils.backcompat.
# broadcast_warning.enabled to True, which will generate a python warning in such cases.
# For Example:
# torch.utils.backcompat.broadcast_warning.enabled=True
# torch.add(torch.ones(4,1), torch.ones(4))
# __main__:1: UserWarning: self and other do not have the same shape, but are broadcastable, 
# and have the same number of elements.
# Changing behavior in a backwards incompatible manner to broadcasting rather than viewing as 
# 1-dimensional.

# now that we have the multiplication covered, lets takl about how to change the shape of our tensors
# for this we have several options. 
# x.reshape():  this is like what we have in numpy, but there is a catche here. 
#               sometimes, reshape, just changes the shape and returns the very same data (x)
#               but sometimes, it returns a 'clone' of the data because of some internal operations!
#               (it copies the data to some other memory location and thus return a clone!!)  
# As it is explained in the docs : 
#               Returns a tensor with the same data and number of elements as input,
#               but with the specified shape. When possible, the returned tensor will
#               be a view of input. Otherwise, it will be a 'copy'. Contiguous inputs and
#               inputs with compatible strides can be reshaped without copying, but you 
#               should not depend on the copying vs. viewing behavior.
#
# view():       This is what we should be using nearly 100% of all times! view always returns the same
#               data(x). it works just like reshape, but with the benifit of returning the very same data!
#               (there is a note that we will get to later when we deal with rnns and lstms!)
# As we see in the docs (https://pytorch.org/docs/stable/tensors.html#torch.Tensor.view): 
#               Returns a new tensor with the same data as the self tensor but of a different shape.
#               The returned tensor shares the same data and must have the same number of elements, 
#               but may have a different size. 
#               For a tensor to be 'viewed', the new view size must be 'Compatible' with its 
#               original size and stride, i.e. each new view dimension must either:
#                   i.be a subspace of an original dimension, or
#                   ii.only span across original dimensions d,d+1,…,d+k that satisfy the following 
#                      contiguity-like condition that ∀i=0,…,k−1
#                                    stride[i] = stride[i+1] × size[i+1]
#               Otherwise, contiguous() needs to be called before the tensor can be viewed. 
#               See also: reshape(), which returns a view if the shapes are compatible, and copies
#               (equivalent to calling contiguous()) otherwise.

# resize_():    as the name implies it 'physically' resizes the tensor 'inplace' (note the '_' which denotes inplace
#               operation) there is a catch here as well!
#               if the new specified dimensions, result in a larger tensor, new uninitialized data will be 
#               resulted. similarly, if the new dimensions are less than the actual dimensions, data will be
#               lost! 
# so the best option as you can see is to use view unless, you specifically intend on using the other two!
# knowing their pitfals ! in which case is fine!!
# in our introductory tutorial, we will always be using view!
# 
#  


#%%
# inplace operations 
# tensors also provide inplace version of some operations such as mul, add, abs, cos, etc
# these inplace operations are denoted by and underscore or '_' at the end 
# add_
# mul_
a = torch.tensor([1])
print(f'a =  {a}')
a.mul_(2)
print(f'a*2 = {a}')
a.div_(2)
print(f'a/2 = {a}')
a.add_(2)
print(f'a+2 = {a}')
a.sub_(2)
print(f'a-2 = {a}')
#%%
# now let create a simple hidden layer with a weight and bias and input 
# lets imlement a simple 1 layer and then 2 layer neural network! 
# dont worry here we will keep it simple! 
# our network has 5 neurons in its hidden layer, gets an input with 7 data points
# and creates 1 output 
# lets write the calculation for 1 step only!(forward propagation only)
inputs = torch.randn(2,7)
W = torch.rand(size=(7,5)) 
b = torch.rand(size=(5,))
W_output = torch.rand(size=(5,1)) 
b_output = torch.rand(size=(1,))

def sigmoid(x):
    return 1/(1+torch.exp(-x))
output = sigmoid(torch.mm(inputs,W) + b)
output = sigmoid(torch.mm(output,W_output) + b_output)
print(output)

#%%


