#%% [markdown] 
# in the name of God the most compassionate the most merciful
# Pytorch basics : introduction on tensors
import torch 
import numpy as np 

# Here we are going to see what torch is and how similar it is to numpy!
# torch is a deep learning framework written in C/C++ that is used for 
# training and working with deep neural networks.
# 
# What is Pytorch then? 
# PyTorch is a Python package that provides two high-level features:
#   1.Tensor computation (like NumPy) with strong GPU acceleration
#   2.Deep neural networks built on a tape-based autograd system
# 
# Basically PYtorch is the python wrapper for torch! 
# You can reuse your favorite Python packages such as NumPy, SciPy and Cython to extend PyTorch when needed. 
# In this section we are going to have an introduction concerning torch as a numpy replacement in working with tensors
# 
# note that pytorch has added support for many numpy functions, 
# so if you are familiar with numpy, then you'll love torch!
# lets see how we can ue pytorch in this sense!
#%%
# What is a tensor? 
# simply put, a tensor is a general name given to arrays.
# we can think of a tensor as a multi-dimensional array, 
# that generalizes the concepts of vectors and matrices. 
# that is a tensor is really a fancy name for arrays 
# its a generalized way of representing data with multiple dimensions or modes.
# It is an extension of the concept of vectors (1D tensors) 
# and matrices (2D tensors) to higher dimensions.
# So, whenever we talk about tensors, remember that 
# it's simply a more flexible and encompassing term for arrays. 
#
# why do we care about them? 
# they are important to us for a few reasons. for one 
# they are used to represent multi-dimensional data, such
# as images, audio, text, etc in deeplearning (and machinelearning in general).
# why do we use them like that you ask? 
# becasue they enable us to efficiently do computation on large datasets and complex 
# models which are typical for deeplearning. this the second reason, 
# which happens to be one of the most crucial ones why we use them (becasue of parallel
# computation which allows us to quickly and efficiently run algorithms that would take a 
# huge amount of times if processed used normal cpu! we can use our gpus to run them which is
# a great help!) 
# 
# sidenote:
# a tensor can be 1 dimensional like a vector, 2 dimensional like a matrix or
# as we'll soon see, more dimensional which is usually 
# what we refer to as simply a [multi-dimensional] tensor!
# we dont have a specific name for higher dimensional arrays, 
# so instead we use the general term 'tensor' for them.
# 
# a tensor as you now know, can have any dimensions (1,2,3,...), but usually when you hear the term tensor,
# it may refer to 3 and higher dimensional arrays(becasue if its 1d or 2d, we usually refer to them as vectors/matrices)
#  
# 
# 
# Common tensor operations include element-wise operations like addition,
# multiplication, etc., matrix multiplications, convolutions, and more.
# Deep learning frameworks, such as PyTorch, Jax(replacement for tensorflow) and others,
# provide optimized implementations of tensor operations, making it easier 
# to build and train complex neural networks.
# Here we are going to have a very crude introduction to some of these operations in pytorch
# and familiarize ourselves with some features. 
# note that usually they may not make much sense, until later on when we actually try to do something
# meaningful with them, like implementing certain algorithms/operations/modules with them. 
# they would make much more sense then, but to get there we need to have a basic idea.
# this is essential for that basic idea!
#
#
# Creating new tensors 
# to create a new tensor we can use several appraoches. 
# we can use the Tensor() class, or use functions such as zeros,ones,rand and empty to say a few.
#  
# lets create a tensor of size (5) , (2, 2), (3, 5, 6) with all zeros, ones, random values and finally no values
t = torch.Tensor(size=(1,))
t1_zeros = torch.zeros(size=(5,))
t1_ones = torch.ones(size=(2,2))
t1_rand = torch.rand(size=(2,2,2))
t1_randn = torch.randn(size=(2,2,2))
t1_empt = torch.empty(size=(2,2,2))

print(f'zeros: {t1_zeros}')
print(f'ones: {t1_ones}')
print(f'rand: {t1_rand}')
print(f'empt: {t1_empt}')
print(t)

# we use torch.zeros() when we want a tensor to have zero values everywhere.
# likewise if we want to have a tensor with 1 as values we use torch.ones()
# we use torch.rand() to create a tensor with random values from a uniform distribution
# like numpy, torch offers other variants, such as as torch.randn for normal distribution sampling
# torch.randint to generate random integer numbers and much more. 
# we may also want to create a tensor quickly, without initializing it with anything really, in tihs case
# we use torch.empty() which creates an empty tensor in the sense that its not initialized so it has whatever
# values that happens to be on the memory where it points to. its not empty(as in having all values equals to 0)
# its empty in the sense, its not preinitialized. this is especially useful for cases where we want to fill a tensor
# with some calculations, and thus it doesnt make sense to initialize it with a value to only be replaced later, which
# would result in unnecessary computation overhead and slower speed! its equivalent to doing torch.Tensor()
# 

# what if we want our tensors to have specific data!
# like we have our own data and need to create a tensor for it how do we do that? 
# there are several ways to do this, but the simplest one is 
# to simply send our data using list or a numpy array! 
# here we are creating a tensor from a list of numbers (1, 2, 3, 4)!
tensor_1 = torch.tensor([1, 2, 3, 4])
print(f'{tensor_1}')

# using an numpy array 
array_np = np.random.rand(4)
tensor_2 = torch.tensor(array_np)
print(f'np array: {array_np}')
print(f'torch tensor: {tensor_2}')

# sidenote, 
# note that we are using torch.tensor() (lowercase function, and not the class Tensor())
# torch.Tensor is the main tensor class. 
# All tensors are instances of torch.Tensor. 
# When we call torch.Tensor(), we get an empty tensor without any data. 
# On the other hand, torch.tensor() is a function that constructs a tensor with data.
# and it infers the data type automatically. 
# For example consider the following examples:
print(f'{torch.Tensor(10)=}') #returns an uninitialized FloatTensor with 10 values.
print(f'{torch.tensor(10)=}') #returns a LongTensor containing a single value (10) 


# looking at the previous example we see that there is a difference in the number of decimals,
# we can use printoptions to get what we want!
torch.set_printoptions(precision=8)
print(f'np array: {array_np}')
print(f'torch tensor: {tensor_2}')

# how can we reset it back to the defaults? easy we can use default profile and just go back to defaul!
# as it turns out, we can use other profiles (short, full) as well for our uses!
torch.set_printoptions(profile='default')
print(f'np: {array_np}')
print(f'np array: {tensor_2}')

# by the way we can directly create a new tensor from a numpy array! 
# like  this
# unlike the previous way, this uses the same underlying numpy array so no copying takes place!
# this is the way to go for large numpy arrays to prevent massive overhead due to copying time!
tensor_from_numpy = torch.from_numpy(array_np)
print(f'tensor_from_numpy: {tensor_from_numpy}')

# intrestingly we can access the underlying numpy array from a tensor, using. numpy() method!
print(f'data_2(torch tensor): {tensor_2}')
print(f'data_2.numpy()(converted to numpy!): {tensor_2.numpy()}')
# if we look closely we can see that both the numpy and torch array point to the same memory location
# when we use torch.from_numpy() to create the tensor, 
# if we change the value in the torchtensor, the values in the numpy_array will change and vice versa,
# but this will not be the case when we use torch.tensor() which creates a copy from the given data!
print(f'torch.from_numpy() shares the underlying data')
tensor_from_numpy[0]=999
# now lets change a value in array_npy
array_np[0] = -999
print(f'{tensor_from_numpy[0]=}')
print(f'{array_np[0] == tensor_from_numpy[0]=}')
print(f'torch.tensor() creates a copy of the numpy data')
array_np[0] = 5
print(f'{array_np[0] == tensor_2[0]=}')

#%%

# Ok, so we just learnt how to create tensors. in the beginning we said we can leverage GPU! 
# so lets see how we can do that! but before that, we need to check if GPU support is available to us!
# for that we use torch.cuda.is_available() function. torch.cuda module offers a slew of goodies related
# to the gpu, (cuda is for nvidia cards, but it works for other cards such as AMDs that support rocm as well
# and you dont need to change anything. you can always check torch.cuda
# 
print(f'is GPU enabled? : {torch.cuda.is_available()}')
# so as it turns out, all tensors, can have two modes, they can either be on the CPU or the GPU
# the tensors we created so far are in CPU mode. to see on which device our tensors are created
# and will run we simply use the device property! 
print(f'data_2 is created on : {tensor_2.device.type}')

# so how do we move or define a new tensor or an existing one from one device to another?
# we can easily do that using .to(), .cpu() or .cuda() methods. 
# .cuda() as the name implies, puts the tensor on the GPU! 
tensor_2 = tensor_2.cuda()
print(f'data_2 device : {tensor_2.device.type}')

# similarly .cpu() puts the tnesor back to the cpu!
tensor_2 = tensor_2.cpu()
print(f'data_2 device : {tensor_2.device.type}')

# we can do better, and based on our system for example decide if a tensor can use gpu or not!
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# or simply just 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# and then use .to() method to transfer the data to the desired device
tensor_2 = tensor_2.to(device)
print(f'data_2 device : {tensor_2.device.type}')

# now if we want to create a tensor on specific device in the definition we simply
# set the device parameter! like device='cuda' or 'cuda:0'.
# the 0 here denotes the specific GPU on our system. if we have only 1, we can simply 
# use 'cuda', if we want to use a specific GPU, then we use its respective index!
data_3 = torch.rand(size=(2,2), device='cuda:0') # or device = 0
print(f'data_3 device: {data_3.device}')

# how do we get how many gpus are available on our system? 
# easy we can use cuda.device_count(). 
# how do we know which index belongs to which GPU then? 
# we simply use cuda.get_device_name(idx) for that!
from torch import cuda
gpu_count = cuda.device_count()
print(f'all gpus available : {gpu_count}')
print(f'gpu name : {cuda.get_device_name(0)}')
# to see a specific GPU's capabilities we can simply use cuda.get_device_capability(idx)
print(f'gpu capability : {cuda.get_device_capability(0)}')

# There are many more useful functions in cuda module. 
# to read and learn more about this check out https://pytorch.org/docs/stable/cuda.html
# we'll see more functions in later chapters but its a good idea to have a look at the docs anyway!

#%%

# Ok, now what if we have a tensor that is already on a specific device(it can be cpu or a gpu)
# and also has a specific datatype!( all of our tensors can have dtype! the default is float64! 
# (previously in older versions it was fp32))
# in such cases, we can simply use the torch.*_new methods. lets see 
tensor_special = torch.rand(size=(2,2), device = 'cuda', dtype=torch.float16)
print(f'{tensor_special=}')

# now lets create a new tensor from this one that is both on cuda and uses float16!!
new_tensor_ones = tensor_special.new_ones(size=(2,2))
print(f'{new_tensor_ones=}')
# we have other functions such as new_tensor, new_empty, new_full, new_zeros as well
new_tensor_zeros = tensor_special.new_zeros(size=(2,2))
print(f'{new_tensor_zeros=}')
# a new tensor full of 3 with the same dtype and device as tensor_special
new_tensor_full = tensor_special.new_full(size=(2,2), fill_value=0.3)
print(f'{new_tensor_full=}')
# uninitialized tensor with the same dtype and device as tensor_special
new_tensor_empty = tensor_special.new_empty(size=(2,2))
print(f'{new_tensor_empty=}')

# and finally if we have a data of our own, we can create 
# a new tensor with the same dtype and device as tensor_special as well
new_tensor_newtensor = tensor_special.new_tensor(np.random.uniform(-1,1,size=(2,2)))
print(f'{new_tensor_newtensor=}')
#
# why would we want something like that? how is that any benificial to us? 
# later on when you write modules, you'll notice that instead of checking for an input
# tensors dtype/device all the time and then creating the right combinations each time, 
# we can easily create a tensor this way, which transfers the dtype and device of that tensor
# automatically without us explicily checking and making a tensor for said dtype/device combo!
# its less code, less bug and more efficient!
#%%
# before we continue, its worth taking a bit of time and learn about generators
# and seeding. 
# sometimes we want to produce determinstic output, for various reasons, ranging
# from debugging purposes, to repreducibility purposes. 
# therefore in order to have a repreducible output, we need to take control of the 
# randomness in our operations. 
# whenever we work with operations that involve random numbers, setting the seed for
# the RNG (random number generator) to use the same seedm allows us to generate the 
# same sequence of numbers again and again.
# Nearly all libraries that involve such operations, offer ways to set the seed, including
# both numpy and pytorch.
# In pytorch we can simply specify a seed for the global RNG, by using `torch.manual_seed(number)`.
# interestingly there is `torch.seed()` that generates a random seed automatically sets the global RNG
# and then returns it! 
# 
# note that torch.manual_seed() both sets the seed for global RNG and returns a generator object!
# while torch.seed() only returns the random seed that was used to set the global RNG.
generator = torch.manual_seed(15)
# now if we try and create a random tensor, it will always have the same values
random_tensor = torch.randn(size=(2,2))
print(f'{random_tensor=}')
# we will always get 
# random_tensor=tensor([[-0.7056,  0.6741],
#                       [-0.5454,  0.9107]])
# 
# now if we want to see what seed was used initially 
# we can easily use `torch.initial_seed()` or `torch.random.initial_seed()`:
print(f'{torch.initial_seed()=}')

# When we set the seed using torch.manual_seed(), or torch.seed() or anyotherway 
# we can retrieve the random state as a tensor using either 
# `torch.get_rng_state()` or `torch.random.get_state()`.
# The state tensor contains information about the internal
# state of the random number generator (RNG). 
# this RNG state is returned as a torch.ByteTensor
# and it contains all the necessary bits to restore the RNG to
# a specific point in time. we can save this into a file
# and later on restore it and set it back using `torch.random.set_state(saved_state)`.
# but an easier way would be to simply use torch.initial_seed()
print(f'{torch.random.get_rng_state()=}')

# note that the rng_state doesnt only contain initial_seed, it has other information
# as well. so its not like a byte representation of a single seed number!
# if we convert the initial_seed() into a bytes array and look at it
# we can see our initial seed there, at the begining of the array but
# the rest will be zeros whereas in the actual rng_state they are nonzero
# values: 

# lets see this in action. 
# generate a random seed
seed = torch.seed()
# retrieve the initial seed used
init_seed=torch.initial_seed()
# now get the rng_state
rng_state = torch.get_rng_state()
print(f'{seed=}')
print(f'{init_seed=}')
# rng_state is a tensor of size torch.Size([5056])
print(f'{rng_state.shape=}')
# by default pytorch doesnt print all the elements and it might give us
# the impression that only the few starting elements are nonzero and the
# rest are zeros! this is obviouly wrong! see the rest
print(f'{rng_state=}')
# to convert our seed into bytes, we take the byte length as well
# our system is little endian, so we specify that as well otherwise, the
# result would be messed up (kind of flipped) due to cpu endian-ness!
seed_bytes = seed.to_bytes(rng_state.shape[0],'little')
# now create a numpy/torch array out of our bytes, 
# thanks to torch implementing numpy operations, they are identical here:
# seed_bytearray = np.frombuffer(seed_bytes, dtype=np.uint8)
seed_bytearray = torch.frombuffer(seed_bytes, dtype=torch.uint8)
# length checks out as well
print(f'{seed_bytearray.shape=}')
# seems pretty similar to rng_state right? not so fast
print(f'{seed_bytearray=}')
print(f'{rng_state=}')
# when we compare them we see that they are not equal!
print(f'{torch.equal(rng_state, seed_bytearray)=}')
# now lets try to see them in full
torch.set_printoptions(profile='full')
print(f'{seed_bytearray=}')
# while our rng_state is quit different after the few early elements
# which tells us it has more information other than a simple seed!
print(f'{rng_state=}')
# so the thing to remember is, to either store the seed-number, or the
# rng_state for resuming purposes later on. (we usually use seed only!)

# lets reset the printoptions back to its defaults
torch.set_printoptions(profile='default')


# we can create generators and instead of using a global one, use 
# separate seeds for separate sections of our code.
# this initializes the global rng, so we dont need to grab the return generator!
torch.manual_seed(15)
random_tensor_1 = torch.randn(size=(2,2))
# now lets create a new tensor this time using a generator
generator = torch.Generator(device='cpu').manual_seed(5)
random_tensor_2 = torch.randn(size=(2,2,), generator=generator)
random_tensor_3 = torch.randn(size=(2,2,), generator=generator)
print(f'{random_tensor_1=}')
print(f'{random_tensor_2=}')
print(f'{random_tensor_3=}')
# and we get : 
#random_tensor_1=tensor([[-0.7056,  0.6741],
#                        [-0.5454,  0.9107]])
# random_tensor_2=tensor([[-0.4868, -0.6038],
#                         [-0.5581,  0.6675]])
# random_tensor_3=tensor([[-0.1974,  1.9428],
#                         [-1.4017, -0.7626]])
# as you can see, the global generator is used with our first tensor 
# while for the other two we used an explicit generator and their results stay the same
# no matter how many times we run this!
# 
# sidenote:
# torch like numpy, has put all random related functionalities into random
# module, so while some of the main functionalities can be accessed through
# torch.* , we can always access them all at one place, which is torch.random
# just like numpy!

# note that to get true determinstic output, we usually need to set seed not only
# for torch, but python and numpy as well, especially if some other libraries we use 
# happen to use them. 
# so for For custom operators, we might need to set python seed as well.
# so we might want to do sth like this and seed the global numpy RNG as well as pythons: 
# as well: 
import random 
random.seed(15)
np.random.seed(15)
torch.random.manual_seed(15)
# as the official documentation says: 
# However, some applications and libraries may use NumPy Random Generator objects, 
# not the global RNG (https://numpy.org/doc/stable/reference/random/generator.html),
# and those will need to be seeded consistently as well.
# what does it mean really?
# this is refering to a recent change in numpy where it introduced a new random number generation
# system that provides more flexibility and features than the older global RNG.
# in the new system, instead of relying on the global state (as in `np.random`), 
# NumPy now encourages using explicit random generator objects (instances of `numpy.random.Generator`).
# These generator objects allow us to manage seeds, distributions, and other properties independently.
# 
# therefore if our code interacts with NumPy (directly or indirectly), it's now essential to 
# seed these generator objects consistently.
# also setting the global seed (e.g., `np.random.seed(0)`) won't necessarily affect these generator objects.
# and finally to ensure reproducibility, we would need to set the seed for both PyTorch 
# (using `torch.manual_seed(0)`) and NumPy (using `np.random.seed(0)`).
# basically when working with numpy alongside pytorch, we need to be aware of these 
# separate random generator objects and seed them consistently for reproducible results.

# I also need to mention something important that getting determinstic output
# when it comes to cuda is not always as plain and simple as the cpu version
# becasue of the nature of cuda, trying to get determinstic output will not be easy to say the least
# and some times not possible, and for the cases where its possible it may very well result 
# in degraded performance. read https://pytorch.org/docs/stable/notes/randomness.html
# TODO: explain more 
# see, the deterministic behavior in pytorch refers to ensuring that given the same input, 
# the same sequence of operations will produce the same output.
# However, achieving complete determinism across different releases, or various platforms 
# can be challenging to say the least especially when it comes to cuda.
# 
# note that the determinstic behavior we talk about here, is usually bound to software/hardware. 
# that is, we expect that given the same input, and same sequence of operations, when run on the
# same software and hardware, we always get the same output. 
# This is an important implication (we see why this is the case when something like cuda is involved)
# 
# **CuDNN and CUDA**: The cuDNN library, used by CUDA convolution operations, 
# can introduce nondeterminism. When a cuDNN convolution is called with new size parameters, 
# it runs multiple convolution algorithms to find the fastest one. 
# Due to benchmarking noise and different hardware, the benchmark may 
# select different algorithms on subsequent runs, even on the same machine(due to benchmarking noise as hardwre is the same here).
# Disabling the benchmarking feature with `torch.backends.cudnn.benchmark = False`
# causes cuDNN to deterministically select an algorithm, possibly at the cost of 
# reduced performance.(this is usually the case!)
#
# 
# this is not all, aside from this, using `torch.use_deterministic_algorithms()`
# we can configure PyTorch to use deterministic algorithms instead of nondeterministic ones
# where available, and to throw an error if an operation is known to be nondeterministic
# (and without a deterministic alternative). 
# we can find the list of such operations here : https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithms 
# 
# note that some of these operations are only determinstic when they are on CPU. 
# so going determinstic aside from negatively affecting the performance, may not always be feasible
# we can use set_deterministic_debug_mode() as an alternative interface for torch.use_deterministic_algorithms() 
# which allows us to specify what to do when it faces nondeterminstic operations (do nothig(0), warn(1), or error out(2!)
# 
# for example trying to run the nondeterministic CUDA implementation of 
# torch.Tensor.index_add_() will throw an error while it will run ok on CPU mode! 
#
#
# sidenote 1:
# we mentioned earlier that disabling CUDA convolution benchmarking ensures that 
# CUDA selects the same algorithm each time an application is run, however, that
# algorithm itself may be nondeterministic, unless either torch.use_deterministic_algorithms(True)
# or torch.backends.cudnn.deterministic = True is set. 
# The latter setting controls only this behavior, unlike torch.use_deterministic_algorithms() 
# which will make other PyTorch operations behave deterministically, too.
# so to make to make the whole process determinstic, torch.use_deterministic_algorithms(True)
# must be used (provided all our operations have determinstic implementations)
#
# sidenote 2: 
# note that for everything to go smoothly without a problem, 
# for CUDA versions 10.2 or greater, we should set the environment variable 
# `CUBLAS_WORKSPACE_CONFIG` according to CUDA documentation: https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility
# 
# sidenote 3:  
# In some versions of CUDA, RNNs and LSTM networks may have non-deterministic behavior. 
#
# sidenote 4:
# any operations that return a tensor with undefined values (from an uninitialized memory), 
# such as torch.empty(), which is then used as "an input" to some other operations can not 
# be used when determinstic behavior is needed simply becasue they introduce randomness this way.
# In order to get around this issue, pytorch offers, `torch.utils.deterministic.fill_uninitialized_memory()`
# which initializes all unintialized memories with a known value. 
# its set by default when torch.use_deterministic_algorithms(True)
# while this works, it comes with a huge price/overhead becasue of this extra initialization step.
# if we dont use such cases that involve using unintialized memory as 'an input' to an 
# operation, then this function can be set to False.
# 
# from official documentation: 
# Operations such as `torch.empty()`` and `torch.Tensor.resize_()` 
# can return tensors with uninitialized memory that contain undefined values.
# Using such a tensor as "an input" to another operation is invalid if determinism is required,
# because the output will be nondeterministic. 
# But there is nothing to actually prevent such invalid code from being run. 
# So for safety, `torch.utils.deterministic.fill_uninitialized_memory` is set to True
# by default, which will fill the uninitialized memory with a known value 
# if torch.use_deterministic_algorithms(True) is set. 
# This will prevent the possibility of this kind of nondeterministic behavior.
# However, filling uninitialized memory is detrimental to performance. 
# So if your program is valid and does not use uninitialized memory as the input 
# to an operation, then this setting can be turned off for better performance.
#
# sidenotes:
# Some PyTorch operations use random numbers internally. While `torch.manual_seed()` 
# helps control the RNG, certain operations (like `torch.svd_lowrank()`) may still 
# exhibit nondeterministic behavior.
# 
# when using libraries like NumPy, ensure consistent seeds for their random number generators as well.
# 
# Use `torch.manual_seed(0)` to seed the RNG for all devices (CPU and CUDA).
# For custom operators, set the Python seed with `random.seed(0)`.
# If relying on NumPy, use `np.random.seed(0)` (but be aware of NumPy Random Generator objects).
# Remember that complete reproducibility isn't guaranteed, but these steps limit sources of nondeterminism.

# Deterministic operations may be slower than nondeterministic ones, 
# but they facilitate experimentation, debugging, and regression testing.
# Be cautious when sacrificing performance for reproducibility.
# In summary, while PyTorch provides tools to enhance determinism, achieving perfect 
# reproducibility across all scenarios remains challenging.
#
# check dataloaders as well (since this is still too early, well cover this later when
# we talk about them.)
#
#%%
# now  that we've learnt how to create a new tensor, initialize it, specify different dtypes, device, etc
# lets work on addition, subtraction, multiplication, negation, transpose, and the likes 
# for adding two tensors, 
# either a tensor should be as scaler, or has 1 dimension in comon
t1 = torch.tensor([1., 2., 3., 4.])
t2 = torch.tensor([[10.,10.,10.,10.],
                   [10.,10.,10.,10.]])
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
# 
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

# note that we need a different seed for each cells in a jupyter notebook environment
# unless we use a single generator for all operations, which we dont do now becasue we are lazy!:d
torch.manual_seed(15)
tensor_1 = torch.rand(size=(2,3))
tensor_2 = torch.rand(size = (2,))
print(f'{tensor_1=}')
print(f'{tensor_2=}')
# pay careful attention to the dimensions and how the multiplication is carried out!
# data2 * data1
tensor_3 = torch.matmul(tensor_2, tensor_1)
print(f'tensor_2(2,) x tensor_1(2x3): {tensor_3}')
print(f'{tensor_3.shape=}')
# as you just saw, the tensor_2 was broadcasted so it can be multiplied by tensor_1
# tensor_2 was 1D, and it was treated as (1,2) so the dimensions between two tensors
# are valid. thus the output is a 1x3 tensor! 
# this is how we do transpose! using .t() method!
tensor_4 = torch.matmul(tensor_1.t(), tensor_2)
print(f'tensor_1.t()(3x2) x tensor_2(2,): {tensor_4}')
print(f'{tensor_4.shape=}')
# now in this example, the tensor_2 again is broadcasted and this time  
# it is treated as (2x1) tensor so the dimensions between tensors are valid 
# as you can see the output is a tensor of 3x1.

# note that, since one of our tensors is 1D, the result is also shown as 1D
# if we explictly make the tensor_2 2D, the output will follow suit as well
# here we get a row vector which is (1,3) (A row vector is a one-dimensional array (or vector) that has a single row and multiple columns)
tensor_3_2 = torch.matmul(tensor_2.view(1,2), tensor_1)
print(f'{tensor_3_2=}\n{tensor_3_2.shape=}')
# and likewise we get (3,1) or a column vector here
tensor_4_2 = torch.matmul(tensor_1.t(), tensor_2.view(2,1))
print(f'{tensor_4_2=}\n{tensor_4_2.shape=}')

# we can do all of these using mm! 
print('using torch.mm:')
# mm is short for matrix multiply, so all dimensions must be specified!
# unlike matmul, there is no broadcasting going on here!
# we must specify all dimensions ourselevs thats why we used .view() to reshape our tensor 
# to the form it needs to be to have a proper multiplication!
data_3_2 = torch.mm(tensor_2.view(1,2), tensor_1)
print(f'data_2(1x2) * data_1(2x3): {data_3_2}')
# this is how we do transpose!
data_4_2 = torch.mm(tensor_1.t(), tensor_2.view(2,1))
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
# incompatibilities introduced by broadcasting may exist, you may set :
# torch.utils.backcompat.broadcast_warning.enabled to True, which will generate a python 
# warning in such cases.
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
#               
#               When it is unclear whether a view() can be performed, it is advisable to use reshape(),
#               which returns a view if the shapes are compatible, and copies (equivalent to calling 
#               contiguous()) otherwise.
#
# resize_():    as the name implies it 'physically' resizes the tensor 'inplace' (note the '_' which denotes inplace
#               operation) there is a catch here as well!
#               if the new specified dimensions, result in a larger tensor, new uninitialized data will be 
#               resulted. similarly, if the new dimensions are less than the actual dimensions, data will be
#               lost! 
# 
#
# 
#
# sidenote: 
# using view() we can also view the array using other dtypes. that is, we can reinterpret_cast
# our tensor from one dtype to another!
# view(dtype) -> Tensor
# Returns a new tensor with the same data as the self tensor but of a different dtype.
# If the element size of dtype is different than that of self.dtype, 
# then the size of the last dimension of the output will be scaled proportionally. 
# For instance, if dtype element size is twice that of self.dtype, then each pair of elements
# in the last dimension of self will be combined, and the size of the last dimension of the output
# will be half that of self. 
# If dtype element size is half that of self.dtype, then each element in the last dimension of 
# self will be split in two, and the size of the last dimension of the output will be double that
# of self. For this to be possible, the following conditions must be true:
#   self.dim() must be greater than 0.
#   self.stride(-1) must be 1.
# Additionally, if the element size of dtype is greater than that of self.dtype, 
# the following conditions must be true as well:
#   self.size(-1) must be divisible by the ratio between the element sizes of the dtypes.
#   self.storage_offset() must be divisible by the ratio between the element sizes of the dtypes.
#   The strides of all dimensions, except the last dimension, must be divisible by the ratio between
#   the element sizes of the dtypes.
# If any of the above conditions are not met, an error is thrown.

# lets see this using an example: we have a float tensor
# which we want to reinterpret its values as int32 and uint8!
random_tensor = torch.randn(size=(2,2),dtype=torch.float32)
print(f'{random_tensor=}')
# it views the memory location asif it was int32
print(f'{random_tensor.view(torch.int32)=}')
# it views the memory location as a uint8,
# note that viewing a float tensor as uint8 is a reinterpretation, not a direct conversion
# therefore what we actually get, in this case, is a byte representation of the original float value!
print(f'{random_tensor.view(torch.uint8)=}')
# to test this theory we can easily do 
bytes_rep = random_tensor.view(torch.uint8)[0].numpy()
float_num = np.frombuffer(bytes_rep,dtype=np.float32)
print(f'{float_num=}')
print(f'{random_tensor[0]=}')
# which checks out !

# import sys
# print(f'{sys.getsizeof(random_tensor[0,0].item())=}') #24
# random_tensor=tensor([[ 1.0682,  0.1424],
#                       [-1.2754, -0.1769]])
# random_tensor.view(torch.int32)=tensor([[ 1065925280,  1041354631],
#                                         [-1079820441, -1103813929]], dtype=torch.int32)
# random_tensor.view(torch.uint8)=tensor([[160, 186, 136,  63, 135, 207,  17,  62],
#                                         [103,  63, 163, 191, 215,  34,  53, 190]], dtype=torch.uint8)
# 
# 
# 
# so the best option as you can see is to use view() unless, you specifically intend on using the other two!
# knowing their pitfals ! in which case is fine!!
# in our introductory tutorial, we will always be using view!
# 
#  
#
#


#%%
# inplace operations 
# tensors also provide inplace version of some operations such as mul, add, abs, cos, etc
# these inplace operations are denoted by and underscore or '_' at the end 
# add_
# mul_
a = torch.tensor([1.])
print(f'{a=}')
print(f'{a.mul_(2)=}')
print(f'{a.div_(2)=}')
print(f'{a.add_(2)=}')
print(f'{a.sub_(2)=}')
print(f'{a.tanh_()=}')
#%%
# now lets create a simple hidden layer with a weight and bias and input 
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

output = sigmoid(torch.mm(inputs, W) + b)
output = sigmoid(torch.mm(output, W_output) + b_output)
print(f'{output=}')

#%%


