#%% In the name of God the most compassionate the most merciful
# Pytorch basics : Introduction of tensors

import torch 
import numpy as np
import torch.version 

def print_header(title, width=40, fillchar='─', newline=True):
    if newline:
        print()
    print(f"{f' {title} '.center(width, fillchar)}")

def show_tensor(name,t,newline=True):
    print_header(f"{name}", newline=newline)
    print(f"shape={tuple(t.shape)!s:<10}")
    print(t)
#%% intro
# We are going to lean about torch and how we can use it to train neural networks. 
# We are going to see what torch is and how similar it is to numpy!
# 
# What is torch/Pytorch? 
# torch is a deep learning framework written in C/C++ and it is used for 
# training and working with deep neural networks.
# 
# sidenote:
# Historically torch was a lua based framework for deeplearning, later on
# it was reimplemented in C/C++ and was reintroduced as `Pytorch`, a python 
# framework for deeplearning.
#
# So PyTorch is the Python package that wraps around the torch library.
# and it provides two high-level features:
#   1.Tensor computation (like NumPy) with strong GPU acceleration
#   2.Deep neural networks built on a tape-based autograd system
# 
# Since its a normal python package, we can reuse our favorite 
# Python packages such as NumPy, SciPy and Cython to extend it when needed. 
# 
# In this section we are going to get familiar with and learn about
# torch and its echo system. since torch offers tensor computation
# you'll see a lot of similarities in terms of function parity with 
# numpy. 
# (In fact pytorch tries to follow numpy and has has added support 
# for many numpy functions, as its an stablished library and extensively
# used for tensor operations). this makes working with
# torch very pleasant if you already know Numpy, and also do porting
# very easy.
# 
# Before we dive into the details, it helps to have a holistic view
# of the PyTorch ecosystem. Think of PyTorch as a toolbox where each
# module has a specific responsibility. You don't need to learn all of
# them at once, but it is useful to know they exist and what problems
# they solve.
#
# At the center of everything is torch itself.
# This is where tensors live. It provides tensor creation, mathematical
# operations, indexing, linear algebra, random number generation,
# GPU support, and automatic differentiation. Basically the very foundation
# we need to build a deep learning model, hence why we start with this!
# If you already know NumPy, this is the part that will feel the most familiar.
#
# On top of torch sits torch.nn.
# This module provides the building blocks used to construct neural
# networks, such as Linear layers, Convolution layers, Recurrent layers,
# activation functions like ReLU, normalization layers, pooling layers,
# and loss functions. Instead of implementing these ourselves, we simply
# combine them to build a model.
#
# Once we have a model, we need a way to optimize it.
# That is the job of torch.optim.
# It implements optimization algorithms such as SGD, Adam, RMSProp,
# AdamW and many others. These optimizers know how to update the
# parameters of our neural network based on the gradients computed
# during backpropagation.
#
# Another important module is torch.utils.
# It contains utilities that make training much easier.
# The most commonly used ones are Dataset and DataLoader, which help
# us load, preprocess, shuffle and batch our data efficiently.
# Nearly every PyTorch project uses a DataLoader at some point.
#
# PyTorch also provides torch.autograd.
# This is the automatic differentiation engine.
# Instead of manually computing derivatives, autograd records the
# operations we perform on tensors and automatically computes gradients
# for us during backpropagation. This is one of the key technologies
# that makes deep learning practical.
#
# As models become larger, training efficiently becomes more important.
# Modules such as torch.cuda and torch.amp help here.
# torch.cuda gives us access to NVIDIA GPUs, while torch.amp
# (Automatic Mixed Precision) allows training with lower precision
# floating-point numbers, making training faster while using less memory.
#
# PyTorch also contains many specialized modules for particular tasks.
# For example, torch.linalg provides a rich collection of linear algebra
# routines, torch.fft implements Fast Fourier Transforms,
# torch.distributions contains probability distributions useful in
# probabilistic models and reinforcement learning, and torch.special
# provides many advanced mathematical functions.
#
# Beyond the core library, PyTorch has an ecosystem of domain-specific
# libraries built on top of it.
# For example `torchvision` is used for computer vision (datasets, image transforms,
# pretrained vision models) while `torchaudio` is used for audio and speech processing
# likewise `torchtext` is used for natural language processing and text datasets
# and `torchrec`  is used recommendation systems. these are just a few examples, there
# are more! 
# These libraries build upon the same tensor and autograd system,
# so once you understand the PyTorch fundamentals, learning these
# becomes much easier.
#
# So, if we summarize PyTorch, we can think of it in layers:
# tensors               -> represented by torch
# automatic gradients   -> handled by autograd
# neural network layers -> provided by torch.nn
# optimization          -> handled by torch.optim
# data loading          -> handled by torch.utils.data
# hardware acceleration -> torch.cuda and torch.amp
# specialized domains   -> torchvision, torchaudio, torchtext, etc.
#
# Fortunately, you don't need to master the entire ecosystem before
# becoming productive. Most PyTorch projects spend the majority of
# their time using only a handful of these modules. As we progress
# through this course, we'll build up this toolbox gradually, learning
# each module only when we actually need it. By the end, you'll not
# only know how to use PyTorch, but also where to look whenever you
# encounter a new problem.

#%% section intro
# we are going to cover a few  sections and by the end of this chapter
# you should have a basic understanding of how to use torch and tensors 
# and will be ready to learn more advanced concepts about deeplearning 
# training and implementation. 
#
# Section 1: What is a Tensor & How to Create Them
# Section 1.1: Shared Memory vs. Copying (Bridging PyTorch and NumPy)
# Section 2: Essential Tensor Attributes 
# Section 3: Tensor Indexing, Slicing, and Boolean Masking
# Section 4: Device Management & Custom Defaults
# Section 5: Precise Data-Type Control & Casting
# Section 6: Dimension Manipulation (Shape, Reshape, View, Resize, Squeeze, Unsqueeze & Permute)
# Section 7: Tensor Operations (Math, Broadcasting & Reductions)
# Section 8: Joining, Splitting and Repeating Tensors (Concatenation, Stacking & Repeating)
# Section 9: Seeding & Reproducibility (RNG Management)
# Section 10: Advanced Memory Management on CUDA
# Section 11: Quick detour - Learn some utility functions(torch.print_options)

#%% Section 1: What is a Tensor and How to create one
# We start with the Tensor, the most fundamental concept we will
# be dealing the most first.
#
# What is a tensor? 
# Simply put, a tensor is a general name given to arrays.
# We can think of a tensor as a multi-dimensional array, 
# that generalizes the concepts of scalers(0D), vectors(1D)
# ,matrices(2D) and higher dimensional structures. 
# (A tensor is really a fancy name for arrays its a generalized
# way of representing data with multiple dimensions or modes.)
# So, whenever we talk about tensors, remember that 
# it's simply a more flexible and encompassing term for arrays. 
# 
# We use tensors to represent all forms of data in 
# Deep Learning (images, audio, text embeddings, etc.)
#
# Why do we care so much about them? 
# Aside from the terminology nuiacense- whether we call these 
# data structures arrays or tensors- the real reason they
# are important to us is they are used to represent multi-dimensional data,
# such as images, audio, text, etc in deeplearning (and machinelearning in general)
# and they are implemented so efficiently, they allows us to use 
# hardware acceleration (i.e. GPUs on our systems), to efficiently 
# do computation on large datasets and complex models which are typical 
# for deeplearning.
# 
# This is the very reason, and the most important one why we use them.
# Different numerical computing libraries refer to multi-dimensioanl arrays,
# by different names, e.g. Torch and Tensorflow among others refer to them 
# as Tensors, while Numpy, Cupy and MxNet call it ndarray (n-dimensional array),
# Others like Jax may call it Array. Most deep learning libraries use Tensor though.

# sidenote:
# Unlike other libraries we mentioned here, Numpy doesnt offer GPU
# acceleration. CuPy (a cuda based gpu accelerated library) instead does,
# CuPy's interface is highly compatible with NumPy and SciPy; in most cases
# it can be used as a drop-in replacement.
# Also note that Both NumPy and CuPy do not support automatic differentiation
# they are just pure numerical computation libraries, where as others such as
# Torch, Tensorflow,Jax or MXNet are specificlly designed for deeplearning.
# 
# The parallel computation offered by torch allows us to quickly and efficiently
# run algorithms that would otherwise take a huge amount of time to complete! 
# We can use our GPUs to run operations on them and make deeplearning actually feasible!
# 
# There are several important operations that we can run on tensors.
# Common tensor operations include element-wise operations like addition,
# multiplication, etc., matrix multiplications, convolutions, and more.
# Deep learning frameworks, such as PyTorch, Jax(replacement for Tensorflow) 
# and others, provide optimized implementations of tensor operations, making
# it very easy to build and train complex neural networks.
#
# sidenote:
# torch is the most used deeplearining framework in the world, especially 
# among researchers a great number of new papers in the field publish their
# implementations using torch, hence you can run/experiment with sota works
# readily when you know torch! 
# 
# sidenote:
# a tensor can be 0 dimensional like a scaler(an ordinary number), 1 dimensional 
# like a vector, 2 dimensional like a matrix or
# as we'll soon see, more dimensional which is usually 
# what we refer to as simply a [multi-dimensional] tensor!
# we dont have a specific name for higher dimensional arrays, 
# so instead we use the general term 'tensor' for them.
#  
# sidenote:(exessive?obvious?)
# In torch/coding nomenclecture however, since tensor is the basic 
# building block on which the whole process is based on -- its the
# class that implements basically everything we use to train networks--
# any dimensional object thats inherited from Tensor is called tensor!
# so regardless of its dimensions we call them simply tensors!
#   
# 
# Here we are going to have a very crude introduction to some of 
# these operations in pytorch and familiarize ourselves with some features. 
# 
# note that usually they may not make much sense right now, 
# but later on when we actually try to do something meaningful
# with them, like implementing certain algorithms/operations/modules with them 
# they would make much more sense. To get there we need to have a basic idea.
# this is essential for that basic idea!
#
# Creating new tensors 
# to create a new tensor we can use several appraoches. 
# we can use the Tensor() class, or use factory functions such as zeros(),
# `ones()`,`rand()` and `empty()` to say a few.
#  
# lets create a few tensors with pre defined shapes and default values
# size (5), (2, 2), (3, 5, 6) using the functions we just learned about!
# 
t = torch.Tensor(size=(1,))              # Using uninitialized scaler 
t1_zeros = torch.zeros(size=(5,))        # using all zeros 
t1_ones = torch.ones(size=(2, 2))        # using all ones
t1_rand = torch.rand(size=(2, 2, 2))     # Using uniform distribution [0,1]
t1_randn = torch.randn(size=(2, 2, 2))   # Using normal distribution (mean=0, std=1)
t1_empty = torch.empty(size=(2, 2, 2))   # Using uninitialized memory

# TODO : put all tensors on a new line so they look nicer when printed,
print('Tensor((1,))')
print(t)

print('\nZeros((5,)):')
print(t1_zeros)

print('\nOnes((2,2)):')
print(t1_ones)

print('\nRand((2,2,2)):')
print(t1_rand)

print('\nEmpty((2,2,2)):')
print(t1_empty)

# We use `torch.zeros()` when we want a tensor to have zero values everywhere.
# likewise if we want to have a tensor with 1 as values we use `torch.ones()`
# They come handy when we want to do operations such as add, multiplication,
# etc or do masking which we will shortly see.
# 
# we use `torch.rand()` to create a tensor with random values from a uniform distribution
# like numpy, torch offers other variants, such as as `torch.randn` for normal distribution
# sampling, `torch.randint()` to generate random integer numbers and much more. 
# 
# We may also want to create a tensor quickly, without initializing it with anything really, in this case
# we use `torch.empty()` which creates an empty tensor in the sense that its not initialized so it has whatever
# values that happens to be on the memory where it points to. its not empty(as in having all values equals to 0)
# its empty in the sense, its not pre-initialized. this is especially useful for cases where we want to fill a tensor
# with some calculations, and thus it doesnt make sense to initialize it with a value to only be replaced later, which
# would result in unnecessary computation overhead and slower speed! its equivalent to doing `torch.Tensor()`
# 

# What if we want our tensors to have specific data!
# like we have our own data and need to create a tensor for it
# how do we do that? 
# There are several ways to do this, but the simplest one is 
# to simply send our data using a list or a numpy array! 
# 
# Here we are creating a tensor from a raw python list of numbers (1, 2, 3, 4)!
list_data = [1,2,3,4]
tensor_from_list = torch.tensor(list_data)
print(f'\nTensor from list:   {tensor_from_list}')

# using a numpy array 
array_np = np.random.rand(4)
tensor_from_np_copy = torch.tensor(array_np)
print(f'\nNumpy array:            {array_np}')
print(f'\nPytorch tensor(copied): {tensor_from_np_copy}')

# sidenote, 
# note that we are using `torch.tensor()` (lowercase function, and not the class Tensor())
# `torch.Tensor` is the base tensor class. 
# All tensors are instances of `torch.Tensor`. 
# When we call `torch.Tensor()`, we get an empty tensor without any data. 
# On the other hand, `torch.tensor()` is a factory function that constructs
# a tensor with the given data and it infers the data type automatically.
 
# For example consider the following examples:
print(f'\ntorch.Tensor(10): {torch.Tensor(10)}')  # Returns an uninitialized FloatTensor with 10 values.
print(f'\ntorch.tensor(10): {torch.tensor(10)}')  # Returns a LongTensor containing a single value (10)
 
# So `torch.Tensor` is the main class constructor when called with a shape/size, 
# it returns an uninitialized `FloatTensor` (equivalent to torch.empty).
# however `torch.tensor` is a factory function that expects data as its argument 
# and infers the type.

# Looking at the previous example we see that 
# there is a difference in the number of decimals,
# we can use printoptions to get what we want!
torch.set_printoptions(precision=8)
np.set_printoptions(precision=8)

print(f'\nNumpy Array:         {array_np}')
print(f'\nTorch Tensor-Copied: {tensor_from_np_copy}')

# How can we reset it back to the defaults? easy we can use default profile and
# just go back to defaul!
# As it turns out, we can use other profiles (short, full) as well!
torch.set_printoptions(profile='default')
# numpy offers the same routine, infact Pytorch took its set_printoptions from NumPy!
np.set_printoptions(precision=None)

print(f'\nNumpy Array:   {array_np}')
print(f'\nShared Tensor: {tensor_from_np_copy}')

# Section 2: Shared Memory vs. Copying (Bridging PyTorch and NumPy)
# We can directly create a new tensor from a numpy array! 
# unlike the previous way, this uses the same underlying
# numpy array so no copying takes place!
shared_tensor = torch.from_numpy(array_np)
print(f'\nShared tensor from numpy: {shared_tensor}\n')

# this is the way to go for large numpy arrays to prevent massive 
# overhead due to copying time!

# Intrestingly we can access the underlying numpy array
# from any tensor, using its numpy() method!

# this is how sharing the underlying memory looks:
print('\nBefore modification:')
print(f'  NumPy Array:   {array_np}')
print(f'  Shared tensor: {shared_tensor}\n')

# if we look closely we can see that both the numpy and
# torch array point to the same memory location when we 
# use `torch.from_numpy()` to create the tensor, 
# if we change the value in the torch tensor, the values
# in the numpy array will change also and vice versa.
# However this will not be the case when we use torch.tensor()
# which creates a copy from the given data!

# print('Shared tensor shares the underlying data with array_np')
shared_tensor[0] = 999
print('\nAfter modifying shared_tensor(PyTorch side):')
print(f'  NumPy Array:   {array_np}')
print(f'  Shared tensor: {shared_tensor}\n')

# now lets change a value in array_npy
array_np[0] = -999
print('\nAfter modifying array_np(NumPy side):')
print(f'  NumPy Array:   {array_np}')
print(f'  Shared Tensor: {shared_tensor}\n')

# we can see this by checking their storage address as well
numpy_address = array_np.ctypes.data
tensor_address = shared_tensor.data_ptr()

print('\nNumpy Data address vs Shared Tensor data address')
print(f'  NumPy data address:  {numpy_address}')
print(f'  Tensor data address: {tensor_address}')
print(f'  Shared memory ?      {numpy_address == tensor_address}\n')

# However as we mentioned earlier, torch.tensor() does not!
print('\nNumpy Data vs torch.tensor() data')
array_np[0] = 5
tensor_cpy_address = tensor_from_np_copy.data_ptr()
print('\nAfter setting array_np[0] = 5')
print(f'  NumPy Array:         {array_np}')
print(f'  Torch.tensor():      {tensor_from_np_copy}\n')

print('\nNumpy Data address vs torch.tensor() data address')
print(f'  NumPy data address:  {numpy_address}')
print(f'  Tensor data address: {tensor_cpy_address}')
print(f'  Shared memory ?      {numpy_address == tensor_cpy_address}\n')

# As we said earlier we can convert a tensor back to numpy
# and this operation also shares the underlying memory
print('\nSharing Memory between torch tensor and numpy with .numpy()')
tensor_to_convert = torch.ones((3,3))
np_from_tensor = tensor_to_convert.numpy() # shares memory!

# modify one to see the effect on both
tensor_to_convert *= 2
print(f'  Torch Tensor: {tensor_to_convert}')
print(f'  Numpy Array(using tensor.numpy()): {np_from_tensor}\n')
print(f'Is memory shared back to NumPy? {tensor_to_convert.data_ptr() == np_from_tensor.ctypes.data}')
#%% Section 2: Essential Tensor Attributes 
# Before we dive into the operations, we need to inspect what makes up a Tensor.
# Every PyTorch tensor carries metadata that describes how it is stored and
# how PyTorch should treat it during computation. 
# These are the several important attributes and methods that we'll 
# encounter constantly when we deal with training/inference of a neural
# network, but 4 of them are the most used:
# 
# shape (or .size()): it describes the dimensions of the tensor.
#      For example, a tensor with shape (3, 4) has 3 rows and 4 columns.
#      Shape determines whether operations such as addition, multiplication,
#      reshaping, and broadcasting are valid.
#
# dtype: specifies the type of values stored in the tensor (float32, int64,
#       bool, etc). The data type affects memory usage, numerical precision,
#       and which operations are permitted.
#
# device: indicates where the tensor resides: CPU, NVIDIA/AMD(through ROCM) GPU (CUDA),
#       Apple Silicon GPU (MPS), and so on. Operations can only be performed
#       between tensors on the same device.
#
# requires_grad: A boolean flag that tells PyTorch's Autograd engine whether to record
#       operations on this tensor. During neural network training, model
#       parameters usually have requires_grad=True so gradients can be computed
#       automatically during backpropagation.
#
 
tensor = torch.randn(size=(3, 4), requires_grad=True)
print(f'Tensor Shape (attribute): {tensor.shape}')
print(f'Tensor Size (method):     {tensor.size()}')
print(f'Tensor Data Type:         {tensor.dtype}')
print(f'Tensor Device:            {tensor.device}')
print(f'Requires Gradient?        {tensor.requires_grad}')

# There are other attributes and methods, that are as useful/important 
# and are extensively used during training/inference process.
# below we can see several of the mostly used ones:

# Total number of elements in the tensor.
# numel() and its alias nelement() return the total number
# of values in a tensor regardless of how those values are 
# arranged accross dimensions.
# this is useful for a variety of reasons such as :
# counting the total number of values in a tensor
# or checking whether a reshape operation is possible
# (ie. the number of elements must remain the same)
# another thing this method is used for is for counting
# model parameters.
print(f'Tensor elements count:                 {tensor.numel()}')
print(f'Tensor elements count(alias):          {tensor.nelement()}')

# TODO Use properties instead of attributes? or keep using attribute?in python im more accustomed to attribute myself so thats why I used them here
# but properties seem better

# Number of dimensions (also called the tensor's rank).
# This is one the most commonly used tensor attributes
# its especially useful when we want to check whether 
# a tensor has the expected number of dimension or when
# writting code that works with tensors of different ranks
# a scalar has 0 dimensions, a vector has 1, a matrix has 2, etc.
print(f'Number of dimensions (ndim):           {tensor.ndim}')
print(f'Number of dimensions (dim method):     {tensor.dim()}')

# The memory layout describes how tensor elements are stored.
# For almost all tensors(dense) we'll encounter, this will is torch.strided.
# Almost all dense tensors use `torch.strided`, which means they
# store elements in contiguous or strided memory. Other layouts,
# such as sparse layouts, exist for specialized use cases but are
# much less common.
print(f'Tensor layout:                         {tensor.layout}')

# Another attribute that we may encounter a lot especially when things go 
# wrong, in error messages is the contiguous attribute of a tensor. 
# A contiguous tensor is stored in one uninterrupted block of memory.
# Many PyTorch operations are faster on contiguous tensors, and some
# operations (such as view()) require contiguity otherwise we face error!
print(f'Is tensor contiguous?                  {tensor.is_contiguous()}')

# grad_fn attribute stores the operation that created this tensor.
# note that leaf tensors created directly by the user have grad_fn=None.
print(f'Gradient function(grad_fn):                     {tensor.grad_fn}')

# .grad attribute stores the computed gradients for leaf tensors, 
# after calling backward().
# tensor is a leaf node but since Backward() hasn't been called yet,
# this is currently None.
print(f'Gradient currently stored:             {tensor.grad}')

# is_leaf attribute specifies says whether the tensor is directly created 
# by us (like weights, biases, basically model parameters) or is an
# intermediate tensor as a result of an operation. 
# Pytorch only stores the gradients for leaf nodes -(nodes starting
# a graph) and it discards the gradients for non-leaf tensors.
# A leaf tensor is a normal tensor like any other, however the 
# distinctions is there purely out of a technicality in having
# more efficient vram usage during traings.
# We need to calculate gradients for all tensors in a network 
# that have requires_grad = True. simply storing all of the gradients
# for all tensors like that, leads to excessive amount of vram.
# Instead, In practice libraries such as PyTorch use a smart approach
# in which instead of storing the gradients for every single tensor 
# in the graph it only stores the gradients for the leaf nodes, that 
# the optimizers require for tuning and optimization. the rest of the
# intermediat/non-leaf nodes (that get created as the result of operations
# involved) will have their gradients calculated 
# dynamically/on the fly during backpropagation and then discared 
# to save memory (for non-leaf nodes Pytorch instead records how
# the tensor was created, i.e. stores the function that yielded
# that tensor in grad_fn and any other piece of information thats
# required for computing gradients and computes the gradients using
# that during backprop)
# 
# (so leaf tensors are the tensors we created that require gradients
# PyTorch stores .grad only for them by default because they are the
# tensors that optimizers update, while gradients of intermediate 
# (non-leaf) tensors are computed on the fly and discarded to save
# memory.) 
print(f'Is tensor a leaf node:                 {tensor.is_leaf}')

# When we run an operation and do a backward it gets filled 
# note upon entering 
out = tensor + 1 

# sidenote:
# since out is not a scalar, calling backward directly would raise an error
# like 'RuntimeError: grad can be implicitly created only for scalar outputs.'
# what we do during training is that we typically reduce the model's outputs
# to a single scalar loss (e.g. using sum or mean or a loss function, etc)
# this scalar serves as the starting point of the backpropagatopn process.
# Otherwise we need to provide the initial (upstream) gradients ourselves so
# that pytorch can then propagate this back and calculate all other gradients
# with respect to them. 
# since we are not training, and dont need a loss, we can use sum() to provide
# the initial gradients. you may also see, some people, prefer a more verbose/explicit
# approach by sending all ones with the same shape as the output tensor doing backward()
# and send that as gradinets to backward() that is out.backward(gradient=torch.ones_like(out)! 
# its the exact equivalent to out.sum().backward() since the gradient of a sum operation is ones!
# hence why we chose the shorter method using sum()
# 
loss = out.sum()
loss.backward()
# or equivalently 
# out.backward(gradient=torch.ones_like(out))

print('\nAfter forming a computation graph using tensor + 1')
print(f'  Is tensor Leaf node?                 {tensor.is_leaf}')
print(f'  Gradient for tensor:                 {tensor.grad}')
print(f'\n  Is out leaf node?                    {out.is_leaf}')
print(f'  Gradient for non-leaf node:          {out.grad}')
print(f'  Non-leaf grad_fn:                    {out.grad_fn}')

# sidenote:
# note that had we used an inplace operation on tensor, like tensor +=1
# we would have faced an error. we can not do inplace changes to leaf nodes
# that require grads. 
# 

# note2:
# To force Pytorch to store gradients for non-leaf(intermediate) nodes
# we can use retain_grad() function. note retain_grad() doesnt
# follow the the inplace naming convenion for tensors in using underscore(_) to 
# denote inplace changes, so it actually does change the tensor attribute 
# in place!( more explain in a moment)
# to query the grad retention status, we use the `retains_grad` attribute!

out2 = tensor + 1
out2.retain_grad()
out2.sum().backward()

print('\nForcing grad population for non-leaf node(out2) using retain_grad')
print(f'  Is out2 leaf node?                   {out2.is_leaf}')
print(f'  Does out2 retains gradients:         {out2.retains_grad}')
print(f'  Gradient for non-leaf node:          {out2.grad}')
print(f'  Non-leaf grad_fn:                    {out2.grad_fn}\n')

# Memory occupied by a single element.
# this method returns the size (amount of bytes) a dtype occupies
# for example, float32 occupies 4 bytes, float64 occupies 8 bytes,
# int64 occupies 8 bytes, and so on.
print(f'Element size (bytes):                  {tensor.element_size()}')

# we can use this to calculate the amount of memory a tensor's data buffer
# takes. note that there are more than just a simple data buffer in a tensor.
# PyTorch stores metadata for each tensor and theres also 
# Python object overhead, gradients, or allocator bookkeeping as well so 
# in reality the actual memory footprint is larger.
print(f'Total memory (bytes):                  {tensor.element_size() * tensor.numel()}')

# we can inspect a tensor's dtype directly (e.g. float32, int64, etc),
# but Pytorch also provides a conviniet method named is_floating_point()
# 
# rather than checking for a specific floating-point dtype directly, this
# method returns True for any floating-point tensor (float16, bfloat16, float32,...)
# This is useful because many mathamatical operations and layers expect floating point inputs!
print(f'Is floating-point tensor?              {tensor.is_floating_point()}')

# Also remember that tensor conversion methods always return new tensors.
# They never modify the original one inplace
# Here we convert the tensor from float32 (the default) to float64.
tensor_fp64 = tensor.double()

print("\nData type conversion:")
print(f'Original dtype:                        {tensor.dtype}')
print(f'Converted dtype:                       {tensor_fp64.dtype}')
print(f'Original tensor unchanged?             {tensor.dtype == torch.float32}')

# sidenote:
# Most Pytorch `tensor` operations are not inplace, instead they return a new tensor
# and leave the original unchanged.
# By convention, tensor methods that modify a tensor's *contents* inplace end
# with a trailing underscore(_) such as tensor.add_, tensor.zero_, etc
# 
# some methods such as retain_grad() modify the tensor's autograd behavior
# or its internal state (i.e. flags) rather than its data, so they do not 
# necessarily follow this naming convention
# 
# the underscore convention is about in-place tensor mutation only, 
# not "any method that changes anything about the object"!

# so in a nutshell: methods with an trailing underscore like foo_() modify
# the tensor's data/storage in place.
# the others usually dont modify the *data*, but may very well still change
# metadata or autograd state like retain_grad().
# note detach_() is underscored because it changes autograd state inplace, like requires_grad_()

# out.requires_grad_(True)   # modifies tensor state (underscore)
# out.add_(1)                # modifies tensor data (underscore)
# out.copy_(tensor)          # modifies tensor data (underscore)
# out.retain_grad()          # modifies autograd behavior (no underscore)
# 
# likewise, there are other *stateful* methods without underscores such as:
# 
# tensor.share_memory_()   # underscore (storage-related)
# module.cpu()             # changes module state, no underscore
# module.cuda()            # changes module state, no underscore
# module.train()           # changes module state, no underscore
# module.eval()            # changes module state, no underscore

#%% Section 3: Tensor Indexing, Slicing, and Boolean Masking
# When we are dealing with tensors and large datasets, we need 
# effective ways to carry out different tasks. we constantly
# find ourselves in scenarios where extracting, inspecting and 
# modifying specific regions of tensors are heavily involved.
# In this regard, Pytorch follows Python's indexing rules and
# extends them with NumPy-style advanced indexing, making it 
# both very intuitive and very powerful.

idx_tensor = torch.tensor([[10, 20, 30],
                           [40, 50, 60],
                           [70, 80, 90]])

print(f'Original 3x3 Tensor:\n{idx_tensor}')

# 4.1 Basic indexing 
# Indexing starts at 0, just like Python lists.
print(f'\nElement at row 1, column 2:          {idx_tensor[1, 2]}')

# Entire row
print(f'First row:                             {idx_tensor[0]}')

# Entire column
print(f'Last column:                           {idx_tensor[:, -1]}')

# Negative indexing counts from the end.
print(f'Bottom-right element:                  {idx_tensor[-1, -1]}')

# 4.2 Slicing
# The syntax is identical to Python lists:
# start : stop : step

print(f'\nFirst two rows:\n{idx_tensor[:2]}')
print(f'Last two rows:\n{idx_tensor[1:]}')
print(f'First two columns:\n{idx_tensor[:, :2]}')
print(f'Every other column:\n{idx_tensor[:, ::2]}')

# note slicing returns a view whenerver possible rather than
# copying the underlying data
sub_tensor = idx_tensor[:2, :2]

print(f'\nTop-left 2x2 block:\n{sub_tensor}')

# 4.3 Fancy Indexing
# Just like Numpy, Pytorch allows us to use arbitrary rows and columns 
# using integer tensors or Python lists. This makes many indexing
# operations concise and expressive, and easy to read.
# Not only that, because the indexing operation is performed by 
# PyTorch's optimized backend(i.e. C/C++) rather than by the 
# Python interpreter, its much faster!
# we will be using fancy indexing extensively throughout this book.

print(f'\nRows 0 and 2:\n{idx_tensor[[0, 2]]}')
print(f'Columns 0 and 2:\n{idx_tensor[:, [0, 2]]}')

# 4.4 Boolean Masking
# When it comes to [large] tensor based operations, we generally want to operate
# on entire tensors rather than individual elements. Vectorized tensor 
# operations are significantly faster than explicit Python loops because
# they are implemented in optimized C/C++ and can take advantage of 
# hardware acceleration.
#
# Boolean masking is one of the most useful vectorized techniques. It lets
# us select or modify only the elements satisfying a condition without
# writing loops. A few common examples include ignoring padded tokens, 
# selecting positive samples, filtering detections above a confidence
# threshold, and removing invalid values.

# This creates a tensor mask, with the same shape as `idx_tensor`
# where each entry that is greater than 45 will be set to `True`
# and `False` otherwise.
mask = idx_tensor > 45
print(f'\nBoolean mask (values > 45):\n{mask}')
# Now using this mask, we can extract only the values that satisfy our 
# condition i.e. > 45
filtered = idx_tensor[mask]

# note that boolean indexing always returns a 1-D tensor containing
# all selected elements.
print(f'Filtered elements:                     {filtered}')

# 4.5 Modifying values using masks
# Boolean masks can also be used for in-place modification.
# Instead of filtering elements, we can also modify them directly.
# This is a common technique for clipping values, removing invalid
# entries, or masking unwanted regions before further computation.
idx_tensor[idx_tensor < 40] = 0

print(f'\nTensor after replacing values < 40 with 0:\n{idx_tensor}')

# sidenote:
# Basic slicing usually returns a *view* of the original tensor,
# whereas advanced indexing (integer lists or boolean masks)
# returns a new tensor.






#%% Section 4: Device Management & Custom Defaults
# Earlier, we learned how to create tensors. Those tensors have all lived on 
# the CPU so far. However, one of PyTorch's biggest strengths is its ability 
# to execute tensor operations on hardware accelerators, such as GPUs. 
# Before we can move our tensors to an accelarator device, we first need to
# determine whether one is available.

# In this section, we'll learn how to check for accelerator support and inspect
# the device on which a tensor resides.

#sidenote:
# We used the term "accelerator" instead of "GPU" because unlike the early
# days of Pytorch when it only supported Nvidia GPUs, PyTorch now supports 
# several kinds of hardware designed to speed up tensor operations, aka accelerators!
# aside from CPU, which is the default device, Pytorch currently supports 
# the following accelerator devices:
#
# - CUDA (NVIDIA GPUs): The most common accelerator you'll see in PyTorch
#   tutorials and industery. NVIDIA GPUs are the most commonly used acceraltor
#   both on consumer level GPUs and Server GPUS. They have thousands 
#   processing cores that can execute many operations in
#   parallel, making them orders of magnitude faster than CPUs for 
#   training and running neural networks.
#
# - MPS (Apple Silicon): If you're using a recent Mac with an M-series chip,
#   PyTorch can use Apple's Metal Performance Shaders (MPS) backend to take
#   advantage of the integrated GPU.
#
# - ROCm (AMD GPUs): ROCm is AMD's equivalent of CUDA. If you have a supported
#   AMD GPU, PyTorch can use the ROCm platform to accelerate tensor operations
#   in much the same way that CUDA does on NVIDIA hardware. 
#
# - XPU (Intel GPUs): PyTorch also supports supported Intel GPUs through the
#   XPU device type. Under the hood, it uses Intel's oneAPI software stack,
#   but from your code, you simply move tensors to the "xpu" device.
#
# Besides these devices, PyTorch also includes optimized libraries such as
# XNNPACK, MKL, and cuDNN. These aren't separate devices, instead they make
# operations on a given device (such as the CPU or GPU) run faster behind the
# scenes.
#
# Moreover, the nice thing is that, in most cases, your PyTorch code barely 
# changes. you simply move your tensors (and later, your models) to whichever
# device is available, and PyTorch takes care of running the computations there.
# 
# Before we can use an accelerator, we first need to determine which ones are
# available on our machine. 

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"MPS available:  {torch.backends.mps.is_available()}")
print(f"XPU available:  {hasattr(torch, 'xpu') and torch.xpu.is_available()}")

# sidenote:
# One interesting detail is that ROCm is not a separate device type in PyTorch.
# PyTorch uses the same "cuda" device interface for both NVIDIA CUDA and AMD ROCm.
# On a ROCm build of PyTorch, torch.device("cuda") refers to an AMD GPU, not an 
# NVIDIA GPU.
#
# This is because PyTorch's GPU backend was originally built around the CUDA API,
# and the ROCm backend implements the same interface for compatibility. As a result,
# you still write device="cuda" in your code when using a ROCm-compatible AMD GPU.

# On my machine, it prints the following outputs:
# 
# > CUDA available: True
# > MPS available:  False
# > XPU available:  False

# To see on which device our tensors are created
# and run we simply use the `.device` property!
print(f'tensor is created on : {tensor.device}')

# On my machine it prints :
# > tensor is created on : cuda:0
#
# Notice that `.device` returns `cuda:0` instead of just `cuda`.
# The number identifies the specific accelerator being used.
# This is because some machines have more than one accelerator.
# For example, a workstation might have two NVIDIA GPUs, or a
# server might have eight or more GPUs for training large models.
# For this reason, Pytorch assigns an index to each accelerator:
#   cuda:0    # First GPU
#   cuda:1    # Second GPU
#   cuda:2    # Third GPU
#   ...
# We can choose a specific device by its index:
#
# device = torch.device("cuda:1")
# x = torch.randn(3, 3, device=device)
#
# we also simply write "cuda", and it will work because PyTorch 
# uses the first GPU ("cuda:0") by default.
#
# We'll use only a single accelerator throughout this book, since the vast
# majority of PyTorch code works the same regardless of how many GPUs are
# installed.
#
# To get the general device name without any index, we can use `.device.type`
# property. This will allow us to simply get the device *type* like 'cpu','cuda'
# instead of its specific device id, i.e. "cuda:0", "xpu:1", etc.

# how do we move or define a new tensor or an existing one
# from one device to another?
# We can easily do that using .to(), method. 
# PyTorch also provides convenience methods such as .cuda() and .cpu(). 
# `.cuda()` as the name implies, puts the tensor on the GPU and `.cpu()`
# does the same on CPU! note if we dont specify a device index to cuda(),
# it will use the first device.
# 
# Throughout this book, we'll prefer .to(device) because it works 
# regardless of whether you're using CUDA, MPS, XPU, or just the CPU.

tensor = tensor.cuda()
print(f'tensor device : {tensor.device.type}')

# similarly .cpu() puts the tnesor back to the cpu!
tensor = tensor.cpu()
print(f'tensor device : {tensor.device.type}')

# If we want to create a tensor on specific device in 
# the definition we simply set the device parameter! 
# like device='cuda' or 'cuda:0'.
tensor = torch.rand(size=(2,2), device='cuda:0') # or device = 0
print(f'tensor device: {tensor.device}')

# Note we can also use the index to the accelerator device
# without hardcoding the device type! 
tensor = torch.rand(size=(2,2), device=0) 
print(f'tensor device: {tensor.device}')

# We can do better, and based on our machine for example decide
# if a tensor can use hardware acceleration on GPU or not!

# We can specify a device using the torch.device explicitly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# or simply use the string counterpart 
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# A much better way is to make this device-agnostic and
# make the code dynamically chose whats available on the machine
 
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
elif hasattr(torch, "xpu") and torch.xpu.is_available():
    device = "xpu"
else:
    device = "cpu"

# and then use .to() method to transfer the data to the desired device
tensor = tensor.to(device)
print(f'tensor device : {tensor.device.type}')

# how do we get how many gpus are available on our system? 
# easy we can use cuda.device_count(). 
# how do we know which index belongs to which GPU then? 
# we simply use cuda.get_device_name(idx) for that!

gpu_count = torch.cuda.device_count()
print(f'all gpus available : {gpu_count}')
print(f'gpu name : {torch.cuda.get_device_name(0)}')
# to see a specific GPU's capabilities we can simply 
# use cuda.get_device_capability(idx)
print(f'gpu capability : {torch.cuda.get_device_capability(0)}')

# so to list all available GPUS and their capabilities we can simply do:
if torch.cuda.is_available():
    print(f'Available GPUs: {torch.cuda.device_count()}')

    for i in range(torch.cuda.device_count()):
        print(f'  #{i+1} GPU Name (cuda:{i}): {torch.cuda.get_device_name(i)}')
        print(f'  GPU capability: {torch.cuda.get_device_capability(i)}')

# sidenote:
# Not all accelerators may expose multiple devices. While some types
# such as CUDA/ROCm and XPU, do expose multiple devices, Apple MPS forexample 
# doesnt. It curently only supports one device.

# sidenote:
# Since PyTorch uses CUDA for AMD GPUS aswell, in order to know 
# exactly which backend our machine comes with(CUDA or ROCm),
# other than checking the device name we just went over, we can
# also get that information by checking the torch build information:
# The `toch.version` module contains the specific version information
# of each installed package, so by simply quering them, we can identify
# the available backend.

print(torch.version.cuda) # 13.0
print(torch.version.hip)  # None

# on my machine, it prints out 
# > 13.0
# > None
# signifying I'm using a CUDA Card.

# There are many more useful functions in cuda module. 
# to read and learn more about this check out
# https://pytorch.org/docs/stable/cuda.html
# we'll see more functions in later chapters but its
# a good idea to have a look at the docs anyway!

#%%TODO or should I use the new stgructure where theres a top-down order?
#%% Section 4: Device Management & Custom Defaults
# 1. What is an accelerator?
# Earlier, we learned how to create tensors. So far, all of them have lived on
# the CPU. However, one of PyTorch's biggest strengths is its ability to execute
# tensor operations on hardware accelerators, such as GPUs.
#
# Before we can use an accelerator, we first need to understand what they are
# and determine whether one is available on our machine.
#
# sidenote:
# Why do we say "accelerator" instead of "GPU"?
# We use the term accelerator instead of GPU because, unlike the early days of
# PyTorch when it only supported NVIDIA GPUs, PyTorch now supports several kinds
# of hardware designed to speed up tensor operations.
#
# Besides the CPU, which is the default device, PyTorch currently supports the
# following accelerator platforms:
#
# - CUDA (NVIDIA GPUs): The most common accelerator you'll encounter in PyTorch
# tutorials and industry. NVIDIA GPUs contain thousands of processing cores
# capable of executing many operations in parallel, making them significantly
# faster than CPUs for training and running neural networks.
#
# - MPS (Apple Silicon): If you're using a recent Mac with an M-series chip,
# PyTorch can use Apple's Metal Performance Shaders (MPS) backend to accelerate
# tensor operations on the integrated GPU.
#
# - ROCm (AMD GPUs): AMD's GPU platform. Supported AMD GPUs can accelerate
# PyTorch computations in much the same way CUDA does on NVIDIA hardware.
#
# - XPU (Intel GPUs): PyTorch also supports compatible Intel GPUs through the
# XPU device type. Under the hood it uses Intel's oneAPI software stack, but
# from your code you simply move tensors to the "xpu" device.
#
# Besides these devices, PyTorch also includes optimized libraries such as
# XNNPACK, MKL, and cuDNN. These are not separate devices. Instead, they optimize
# operations on the CPU or GPU behind the scenes.
#
# The nice thing is that, in most cases, your PyTorch code barely changes. You
# simply move your tensors (and later, your models) to whichever device is
# available, and PyTorch takes care of executing the computations there.
#
# 2. How do I know what accelerator my machine has?
# Before using an accelerator, we first need to determine which ones are
# available.

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"MPS available:  {torch.backends.mps.is_available()}")
print(f"XPU available:  {hasattr(torch, 'xpu') and torch.xpu.is_available()}")

# On my machine, this prints:
#
# CUDA available: True
# MPS available:  False
# XPU available:  False
#
# Once we know which accelerator is available, we can begin working with
# devices.

# 3. Inspecting a tensor's device
# Every tensor knows which device it lives on.
#
# We can inspect it using the .device property.
print(f"Tensor device: {tensor.device}")

# On my machine, this prints 'cuda:0'!
# 
# Notice that the output is cuda:0 instead of simply cuda.
# The first part (cuda) tells us the device type, while the number (0)
# identifies the specific accelerator being used.
#
# If we only care about the device type, we can use .device.type instead.
print(tensor.device.type)
# which prints 'cuda'!
# 
# The number becomes useful on machines that have multiple accelerators. For
# example, a workstation might contain two GPUs while a large training server
# may contain eight or more. PyTorch assigns each accelerator an index:
#
# cuda:0    # First GPU
# cuda:1    # Second GPU
# cuda:2    # Third GPU
# ...
#
# We'll revisit multiple GPUs later in this chapter.
# 
# 4. Moving tensors between devices
# Knowing where a tensor lives is useful, but eventually we'll want to move
# tensors from one device to another.
#
# The recommended way to do this is with the .to() method.
#
# First, let's choose the best device available on the current machine.

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
elif hasattr(torch, "xpu") and torch.xpu.is_available():
    device = "xpu"
else:
    device = "cpu"

# Now we can move tensors to that device.
tensor = tensor.to(device)
print(f"Tensor device: {tensor.device.type}")
# PyTorch also provides convenience methods such as .cuda() and .cpu().
tensor = tensor.cuda()
print(tensor.device.type)
# To move the tensor back to the CPU,
tensor = tensor.cpu()
print(tensor.device.type)
# Throughout this book we'll prefer .to(device) because it works regardless of
# whether you're using CUDA, MPS, XPU, or just the CPU.
#
# 5. Creating tensors directly on a device
# Instead of creating a tensor on the CPU and then moving it, we can create it
# directly on the desired device.
tensor = torch.rand((2, 2), device=device)
print(tensor.device)
# If you already know the exact device you want, you can also specify it
# explicitly.
tensor = torch.rand((2, 2), device="cuda:0")
# PyTorch also allows you to specify the device index directly.
tensor = torch.rand((2, 2), device=0)
# Although this works, using the device variable from earlier is generally
# preferred because it keeps your code portable across different machines.
#
# 6. Working with multiple GPUs
# Some machines contain more than one GPU.
# If we want to know how many GPUs are available, we can use
torch.cuda.device_count()

# To display every available GPU along with its name,
if torch.cuda.is_available():
    print(f"Available GPUs: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# To inspect a GPU's compute capability,
torch.cuda.get_device_capability(0)
# If we want to use a specific GPU, we simply specify its index.
# since I only have one GPU, I use 0!
device = torch.device("cuda:0") 
tensor = torch.randn(3, 3, device=device)
# If no index is specified,
device = "cuda"
# PyTorch automatically uses the first GPU (cuda:0).
#
# Throughout this book we'll use only a single accelerator since the
# overwhelming majority of PyTorch code is identical regardless of how many
# GPUs are installed.
#
# sidenote:
# Not every accelerator platform supports multiple devices.
#
# CUDA/ROCm and XPU can expose multiple accelerators, whereas Apple's MPS
# currently exposes only a single device.
# 
# 7. CUDA vs ROCm
# One interesting detail is that ROCm is not a separate device type inside
# PyTorch.
# Even on AMD GPUs, PyTorch still uses the "cuda" device interface.
#
# For example,
device = "cuda"
# may refer to either
# - an NVIDIA GPU (CUDA), or
# - an AMD GPU (ROCm),
#
# depending on which build of PyTorch is installed.
#
# This is because PyTorch's GPU backend was originally designed around CUDA,
# and the ROCm backend implements the same programming interface for
# compatibility.
#
# If you'd like to know which backend your installation was built with, you can
# inspect the version information.
print(torch.version.cuda)
print(torch.version.hip)
# On my machine, this prints
#
# 13.0
# None
#
# indicating that I'm using the CUDA backend. On a ROCm installation,
# torch.version.cuda would be None while torch.version.hip would contain the
# ROCm version.
#
# There are many more useful functions available in the torch.cuda module.
# We'll introduce more of them later in the book, but it's worth browsing the
# documentation if you're curious.
#

# Since pytorch 2.0 we have a `device` context manager which makes our lives easier
# by assigning a specified device to all *new* tensors that get created inside that
# context manager scope. 
# that is any tensors-inlcuding models (torch modules)-we create inside that context manager will 
# be assigned that device!

print('\nUsing context manager to set device(cuda)')
with torch.device('cuda'):
    # our model can be as simple as a linear layer (we'll learn about them 
    # in more details in future chapters)
    model = torch.nn.Linear(4,1)
    dummy_input = torch.rand(size=(1,4))
    dummy_output = model(dummy_input)

# note if we grab one of the parameters and check its device it shows cuda!
print(f' Model.device:        {next(model.parameters()).device}')
print(f' Dummy_input.device:  {dummy_input.device}')
print(f' Dummy_output.device: {dummy_output.device}')

# note that as of now(torch2.12), torch.device context manager does not change the device
# for tensors that already exist. For those we still need to have use .to() to
# move the data to a specific device

# If you have noticed, all tensors we create by default, have been on cpu. 
# we can change this behavior and make, by default, all tensors to be on a
# specific device like cuda globally!

# we use torch.set_default_device() for this purpose: 
print('\nMaking CUDA the default device')
torch.set_default_device('cuda')
# now from now on, all tensors, modules, etc will have their device='cuda' by default
dummy_input = torch.rand(size=(1,4))
print(f'  dummy_input.device:  {dummy_input.device}')
# since pytorch  2.3.0 we can also get the current default device
# using `torch.get_default_device()`
assert int("".join(torch.__version__.split('.')[:2])) > 23, 'pytorch 2.3.0+ is needed'
print(f"\nthe default device is now '{torch.get_default_device().type}'")

# revert back to CPU
torch.set_default_device("cpu")
# 
#%%TODO  we need to start each subsection with the problem first, not the technology
# that is, we start with what the problem is, and they explain our way and introduce 
# technologies. this should give us a much smoother read. 
# as an example, for a section taling about dtypes e.g. we shouldnt start with sth like
# pytorch supports x,y,z dtypes and methods! instead we should be saying sth like,
# "so far we have been using default dtypes, but .... " and then start explaining why 
# we need to be able to choose different dtypes and how we can do that. we dont just
# say stuff like "the problem is," thats terrible! and not a good starting way.
# anyway you get the idea, lets do that. 
#%% Section 5: Precise Data-Type Control & Casting
# Up until now all the tensors we've created have used Pytorch's default data types.
# For many applications thats prefectly fine. However, in practice, we'll often
# need more control. whether we're storing labels as integers, performing
# high-precision comuptation or training models with mixed-precision, choosing
# the appropriate data type is essential!

# A tensor's data type (`dtype`) determines how its values are represented in mmeory
# This directly affects numerical precisin, memory consumption and computational performance.
# Fortunately Pytorch makes it very easy to inspect, change and convert a tensors dtype
# whenever needed.

# Let's see how PyTorch handles different tensor data types.
# When constructing a tensor, PyTorch attempts to infer an appropriate
# data type from the values provided.

# The following line creates a tensor with `torch.int64` as data type.
tensor = torch.tensor([1, 2, 3])
print(f'Infered dtype for [1, 2, 3] : {tensor.dtype}') # torch.int64

# In the second example below, because the list contains a floating-point
# value, PyTorch promotes all elements to a floating-point dtype. 
tensor_float32 = torch.tensor([1., 2, 3])
print(f'Infered dtype for [1., 2, 3] : {tensor_float32.dtype}') # torch.float32

# There are several ways to convert a tensor from one data type to another.
# One option is to specify the desired dtype explicitly during construction.
# The following line casts the default `torch.int64` into `torch.in32` 
tensor_int32 = torch.tensor([1, 2, 3], dtype=torch.int32)
print(f'cast [1, 2, 3] to torch.int32: {tensor_int32.dtype}')

# We can also use the generic `.to()` method.
# Besides changing the dtype, `.to()` can also move tensors between devices,
# making it one of the most commonly used tensor conversion methods.
cast1 = tensor_int32.to(dtype=torch.float32)

# PyTorch also provides convenience methods for the most common conversions,
# such as `.float()`, `.double()`, `.long()`, `.half()`, and many others.
cast2 = tensor_int32.float()   # converts to float32
cast3 = tensor_float32.long()  # converts to int64 (used for index/target layers)

print(f'Cast 1: {cast1.dtype}')
print(f'Cast 2: {cast2.dtype}')
print(f'Cast 3: {cast3.dtype}')

# sidenote:
# The default dtype in PyTorch is float32 we can query the default dtype
# by calling `torch.get_default_dtype()`

print(torch.get_default_dtype()) # torch.float32

# Now what if we have a tensor that is already on a specific device
# (be it CPU, GPU,etc) and also has a specific datatype?
# In such cases, we can simply use the `torch.*_new` methods to 
# create tensors with the same exact device, dtype configuration!

# lets see 
tensor_special = torch.rand(size=(2,2), device = 'cuda', dtype=torch.float16)
print(f'{tensor_special=}')

# Now lets create a new tensor from this one that is both on cuda and uses float16!!
new_tensor_ones = tensor_special.new_ones(size=(2,2))
print(f'{new_tensor_ones=}')

# we have other functions such as new_tensor, new_empty, new_full, new_zeros as well
new_tensor_zeros = tensor_special.new_zeros(size=(2,2))
print(f'{new_tensor_zeros=}')

# a new tensor full of 0.3 with the same dtype and device as tensor_special
new_tensor_full = tensor_special.new_full(size=(2,2), fill_value=0.3)
print(f'{new_tensor_full=}')

# uninitialized tensor with the same dtype and device as tensor_special
new_tensor_empty = tensor_special.new_empty(size=(2,2))
print(f'{new_tensor_empty=}')

# Finally if we have a data of our own, we can create a new tensor with 
# the same dtype and device as tensor_special as well
new_tensor_newtensor = tensor_special.new_tensor(np.random.uniform(-1,1, size=(2,2)))
print(f'{new_tensor_newtensor=}')

# You may be puzzled and think to yourslef why would we want something like that? 
# How is that any benificial to us? 
# Later on when you write modules, you'll notice that instead of checking for an input
# tensors dtype/device all the time and then creating the right combinations each time, 
# you can easily create a tensor this way, which transfers the dtype and device of that
# tensor automatically without us explicily checking and making a tensor for said 
# dtype/device combo! its less code, less bug and more efficient!

# sidenote2: 
# Like torch.device, we have a way to specify a default dtype by using
# `torch.set_default_dtype()`.
# However note that, unlike what you might think at first, it doesn't 
# allow you to set just any dtype you like!
# It only supports `torch.float32`` and `torch.float64`` as inputs. 
# Other dtypes may be accepted without complaint but are not supported
# and are unlikely to work as expected.
# 
# When PyTorch is initialized its default floating point dtype
# is `torch.float32`, and the intent of `set_default_dtype(torch.float64)`
# is to facilitate NumPy-like type inference.
# 
# The default floating point dtype is used to :
#  1.To implicitly determine the default complex dtype. 
#    When the default floating point type is float32 
#    the default complex dtype is complex64, and 
#    when the default floating point type is float64
#    the default complex type is complex128.
#  2.To infer the dtype for tensors constructed using Python floats or complex Python
#    numbers. See examples below.
#  3.To determine the result of type promotion between bool and integer tensors and
#    Python floats and complex Python numbers.
print(f'{torch.tensor([1.2, 3]).dtype=}')

#%% Section 6: Dimension Manipulation (Shape, Reshape, View, Resize, Squeeze, Unsqueeze & Permute)
# The tensors we have created and experimented with so far had a pre-specified
# shape and we didn't need to change them. However, this is not always the case
# As we'll see shortly, different PyTorch operations and neural network layers 
# expect tensors to have specific dimensions. As a result, we'll frequently need
# to reshape tensors, add or remove dimensions, or rearrange the order of their 
# axes without changing the underlying data. This can range from simply adding
# a batch dimension into an input tensor during inference, reshape/merge the 
# dimensions in an input to make it suitable for the next layer like an LSTM or
# swap axis (permute) when implementing an attention module.
# 
# In this section, we'll explore the most common dimension manipulation
# operations and learn when to use each one.

# 6.1 Reshaping Tensors: view() vs. reshape() vs resize()
# Like Numpy Pytorch offers a `.reshape()` method that allows us to change
# a tensor's shape.

# using torch.arange(n) we create a 1D tensor from [0-n),
tensor = torch.arange(12)
# reshape the 1D tensor with 12 elements into 3 rows and 4 columns
tensor_3x4 = tensor.reshape(3,4)
# reshape the same 1D tensor into a 3D tensor
tensor_1x2x6 = tensor.reshape(1,2,6)
print (f'tensor:                {tensor}')
print (f'tensor.reshape(3,4):   {tensor_3x4}')
print (f'tensor.reshape(1,2,6): {tensor_1x2x6}')

# If we specify all dimensions except one, PyTorch can automatically
# compute the missing size so that the total number of elements remains unchanged.
# We can reshape a tensor into any shape, as long as the sizes for
# each dimension, result in the same total number of elements.

# this will automatically infer the last dim to be 2! so 2x3x2=12 total elements
tensor_inferred = tensor.reshape(2,3,-1)
print('Using -1 to automatically infer last dim:')
print (f'  tensor_inferred.shape:    {tensor_inferred.shape}')
print (f'  tensor_inferred:          {tensor_inferred}')

# Passing -1 as the *only* dimension tells PyTorch to infer that dimension,
# since its the only provided dimension, it effectively flattens the tensor
# into a one-dimensional vector.
tensor_flattened = tensor.reshape(-1)
print (f'tensor.reshape(-1):    {tensor_flattened}')

# Interestingly, in this example, `.reshape()` did not create 
# new copies of the tensors, it was able to return *views* of the 
# original tensor. That is all these tensors are different "views" of the
# same underlying storage hence all of these tensors share the same 
# underlying storage. Modifying one therefore, immediately affects
# the others.

tensor_flattened[1] *= 1000 
print('\nAfter changing tensor_flattened[1] *= 1000 ')
print (f'  tensor:                   {tensor}')
print (f'  tensor.reshape(3,4):      {tensor_3x4}')
print (f'  tensor.reshape(1,2,6):    {tensor_1x2x6}')
print (f'  tensor.reshape(2,3,-1):   {tensor_inferred}')
print (f'  tensor.reshape(-1):       {tensor_flattened}')

# This behavior comes from the fact that, the original tensor was stored as
# a contigeous chunk in memory or had compatible strides so reshape was able
# to rearrange the layout to achieve a certain view.
# Not all operations preserve this property. One common example is transposing
# a tensor.
# Transposing a tensor changes its strides, and produces a non-contiguous tensor.
# In this case, `.reshape()` transparaently allocates a new contigeous memory
# and returns a copy of the original tensor.

# since transpose is only meaningful for tensors with at least two dimensions,
# we'll use our 2-D tensor here.
tensorT = tensor_3x4.t().reshape(2,6) # .view(2,6) raises runtime error
print('\nBefore transposing:')
print(f'  tensor:             {tensor}')
print(f'  tensor transposed:  {tensorT}')

# Now if we change tensorT, to have the first row to be 999
# the original tensor wont be affected!
tensorT[0] = 999
print('\nAfter transposing and changing it:')
print(f'  tensor:             {tensor}')
print(f'  tensor transposed:  {tensorT}')

# Pytorch offers another method however, convineintly named as *view*. The 
# `.view()` behaves similarly to `reshape()`, with one important distinction,
# it only works on contiguous tensors and therefore never allocates new memory.

# In the previous example if we tried to use `.view()` we would haved faced 
# a runtime error, which would interestingly instruct us to use `.reshape()` instead!

# How can we tell whether a tensor is contiguous?
# Using `is_contiguous()` we can easily determine if a tensor is contigous or not:
print(f'tensor_3x4 contiguous:     {tensor_3x4.is_contiguous()}')      # True
print(f'tensor_3x4.t() contiguous: {tensor_3x4.t().is_contiguous()}')  # False

# `.reshape()` as you saw is more flexible than `.view()`. Like a `.view()` 
# it returns a view whenever possible, but if it can not rearrange the memory,
# it transparently allocates a new tensor and returns a copy instead.
#
# so if we specifically need to avoid a copy, we use `view()`. Otherwise,
# `reshape()` is often the more convenient choice. 
# 
# sidenote:
# If a tensor is not contigeous, we can use `.contigeous()` and make it continueous
# Note however, this means a copy occurs, so a `tensor.contigeous().view()` would
# be no different that using `tensor.reshape()`. 
# 
# This is why many suggest to use `.reshape()` instead, unless you want to make your
# intent clear by using `view()` signifying, the tensor being worked on uses contigeous
# memory and any views share the underlying storage. 
# The same semantic cant be conveyed about `.reshape()` as its not gauranteed to
# return views all the time.
print(f'tensor_3x4.t().contiguous().is_contiguous(): {tensor_3x4.t().contiguous().is_contiguous()}')  # False

# sidenote:
# For a tensor to be 'viewed', the new view size must be 'Compatible' with its 
# original size and stride, i.e. each new view dimension must either:
# i.be a subspace of an original dimension, or
# ii.only span across original dimensions d,d+1,…,d+k that satisfy the following 
# contiguity-like condition that ∀i=0,…,k−1
# stride[i] = stride[i+1] × size[i+1]
# Otherwise, contiguous() needs to be called before the tensor can be viewed. 

# sidenote:(extra?)
# Contiguous inputs and inputs with compatible strides can be reshaped without copying,
# but you should not depend on the copying vs. viewing behavior.

# sidenote:(extra?)
# When it is unclear whether a `view()` can be performed, it is advisable to use `reshape()`,
# which returns a view if the shapes are compatible, and copies (equivalent to calling 
# `contiguous()`) otherwise.

             
# resize_:
# As the name implies it 'physically' resizes the tensor 'inplace' (note the '_' which denotes inplace
# operation). 
# If the new specified dimensions, result in a larger tensor, new uninitialized data will be 
# resulted. Similarly, if the new dimensions are less than the actual dimensions, data will be
# lost! 


# sidenote:
# An Interesting detail about `.view()` is although its primarily used 
# to reshape tensors, it can also reinterpret a tensor's underlying bytes
# as a different data type!
# Unlike `.to()`, `.float()`, or `.long()`, `view(dtype)` does NOT perform
# a (mathimatical) conversion. Instead, it simply reinterprets the existing binary 
# representation using a different dtype.
#
# This is a low-level operation and is rarely needed in day-to-day deep
# learning. It is primarily useful when working with binary file formats,
# interfacing with external libraries, debugging low-level code, or
# inspecting the bit-level representation of numerical values.
# 
# the following example demonstrates this more clearly.
raw = torch.tensor([14,256], dtype=torch.int16)
print(f'\nraw tensor:          {raw}')
print(f'raw.view(torch.uint8): {raw.view(torch.uint8)}')

# On my machine this prints:
#
# > tensor([  14, 256], dtype=torch.int16)
# > tensor([14, 0, 0, 1], dtype=torch.uint8)
#
# Each int16 value occupies bytes, so 14 is
# 0x000E, which is stored as the bytes [0x0E, 0x00]
# that is [14, 0] on little-endian systems like the 
# one I'm running on.
# Likewise 256 is 0x0100 which is stored as [0x00, 0x01].

# To test this we can easily do 
bytes_rep = raw.view(torch.uint8).numpy()
raw_int16 = np.frombuffer(bytes_rep, dtype=np.int16)
print(f'Bytes interpreted as int16: {raw_int16}')
print()

# To reiterate, calling `.view(torch.uint8)` does not convert
# the values instead, it *reinterprets* the same block of memory as an
# array of `uint8`` values, so we are literally looking at the
# individual bytes that make up each 16-bit integer.
# 
# As mentioned before, this is a very low level operation and not needed in 
# day to day deeplearning chores, but it can come handy when doing
# things such as parsing binary file formats or reading image headers
# stuff like that!


# 6.2 Adding/Removing Dimensions: squeeze() & unsqueeze()
# Neural networks often expect tensors to have a specific number of
# dimensions. For example, a model may expect a batch dimension even
# when processing a single sample. Conversely, some operations leave
# behind dimensions of size 1 that are no longer needed.
#
# PyTorch provides `squeeze()` to remove dimensions of size 1 and
# `unsqueeze()` to insert them without changing the underlying data.

# if no dimension is provied to squeeze(), it removes all dimensions of size 1
tensor = torch.arange(12).reshape(1,3,4,1)
print('\nBefore .squeeze()')
print(f'  tensor.shape: {tensor.shape}')
print(f'  tensor: {tensor}')

# When no dimension is specified, squeeze() removes *all* dimensions
# whose size is 1.
print('\nAfter .squeeze()')
print(f'  tensor.shape: {tensor.squeeze().shape}')

# We can also remove a specific singleton dimension.
print('\nAfter .squeeze(0)')
print(f'  tensor.shape: {tensor.squeeze(0).shape}')

# sidenote:
# A dimension of size 1 is commonly called a singleton dimension.


# Adding dimensions
# 
# Unlike `.squeeze()`, `.unsqueeze()` always requires us to specify where
# the new dimension should be inserted. The inserted dimension always
# has size 1.

print('\nBefore .unsqueeze()')
print(f'  tensor.shape: {tensor.shape}')

print('\nAfter .unsqueeze(3)')
print(f'  tensor.shape: {tensor.unsqueeze(3).shape}')

# We can insert a dimension at any valid position.
print('\nAfter .unsqueeze(0)')
print(f'  tensor.shape: {tensor.unsqueeze(0).shape}')

# As with many PyTorch operations, squeeze() and unsqueeze() also have
# in-place variants ending with an underscore.
print(f"\nOriginal shape:                  {tensor.shape}")

tensor.squeeze_(0)
print(f'After squeeze_(0):                 {tensor.shape}')

tensor.unsqueeze_(0)
print(f'After unsqueeze_(0):               {tensor.shape}')

# While `squeeze()` and `unsqueeze()` only add or remove dimensions of size 1,
# `permute()` changes the *order* of a tensor's existing dimensions.
# 
# Reordering dimensions is especially common in deep learning because different
# libraries and models expect tensors in different layouts. For example, image
# data is often stored as (Height, Width, Channels), or HWC for short, whereas
# PyTorch's convolutional layers expect (Channels, Height, Width), or CHW.
#
# Besides API compatibility, certain dimension orders may also offer better
# performance on specific hardware and software backends (e.g., CUDA/cuDNN).
# We'll revisit this topic when discussing performance optimization.
#
# Unlike `squeeze()` and `unsqueeze()`, `permute()` requires us to specify the
# complete ordering of all dimensions.

print(f"\nOriginal shape:               {tensor.shape}")

# To use `permute()`, we list every dimension in the order we want them to appear.
# Here we swap dimensions 1 and 2 while leaving the remaining dimensions
# unchanged.
tensor = tensor.permute(0,2,1,3)
print(f'After permute(0,2,1,3):       {tensor.shape}')
print(tensor)

# Notice that `permute()` does not modify the values stored in the tensor.
# It only changes how those values are interpreted across dimensions.

# sidenote:
# The arguments passed to `permute()` describe where each *original*
# dimension should appear in the new tensor.
# For example: `permute(0, 2, 1, 3)` means:
#
# new dim 0 <- old dim 0
# new dim 1 <- old dim 2
# new dim 2 <- old dim 1
# new dim 3 <- old dim 3
#
# In other words, dimensions 1 and 2 are swapped while the remaining
# dimensions stay in the same order.


# `.permute()` appears throughout deep learning workflows For example, 
# it is commonly used to convert images between HWC and CHW layouts,
# prepare data for different neural network layers,or rearrange model
# outputs for visualization.
# We'll encounter many practical examples of it in later chapters.

# sidenote:
# If you're coming from NumPy, `.permute()` is roughly equivalent to
# numpy.transpose(), which also reorders an arbitrary number of dimensions.
#
# PyTorch also provides `.transpose()`, but unlike `.permute()`, it swaps only
# two dimensions at a time.

#%% Section 7: Tensor Operations (Math, Broadcasting)
#
# So far, we've covered a range of topics related to tensors, from 
# creating and inspecting them to managing them on different devices.
# But we intentionally left out one of the most important aspects
# of working with tensors, i.e. performing computations with them.
# lets talk about them.
#
# There are many tensor operations we can use, but in this 
# section we are going to only explore PyTorch's built-in mathematical
# operations, learn the difference between element-wise and matrix multiplication,
# and finally see how broadcasting allows tensors of different shapes to
# interact.
#
# These operations form the foundation of virtually every deep learning model,
# from simple linear regression to modern transformer architectures.

# 7.1 Element-wise/Pointwise Arithmetic
# Elemetwise/pointwise arithmetic operations refer to operations in which 
# only the corrosponding elements in respective tensors are being operated on,
# independant of other elements.
# As a result, both tensors must have the same shape (or be broadcastable, 
# which we'll discuss shortly).

# sidenote:
# we use lowercase identifiers to signify scalers, 
# capital idenitifiers to indicate Matrices and
# v for vectors. 

A = torch.tensor([[1., 2., 3.],
                  [4., 5., 6.]])

B = torch.tensor([[10., 20., 30.],
                  [40., 50., 60.]])

print("Tensor A:")
print(A)

print("\nTensor B:")
print(B)

print("\nAddition (A+B):")
print(A + B)

print("\nSubtraction (B-A):")
print(B - A)

print("\nMultiplication (A*B):")
print(A * B)

print("\nDivision (A/B):")
print(A / B)

print("\nPower (A**2):")
print(A ** 2)

# There are inplace variants where the name ends with an underscore(_)
show_tensor("Inplace Multiplication A.mul_(2)", A.mul_(2))
show_tensor("Inplace Division A.div_(2)", A.div_(2))
show_tensor("Inplace Addition A.add_(2)", A.add_(2))
show_tensor("Inplace Subtraction A.sub_(2)", A.sub_(2))

# 7.2 Matrix Multiplication
# Matrix multiplication is one of the most important operations in deep
# learning. In fact it is the most fundamental operation that is used in
# deep learning. 
# From simple modules like fully connected layers(also known as linear 
# layers in PyTorch), to embeddings to attention mechanisms, and many other
# modules, it plays a crucial and unrivaled role so mastering it is paramount!
#
# Elementwise multiplication that we just covered, like adding and subtracting,
# are self explanetory, there's no special case, or exception. The rule is 
# simple and straightforward. However, for multiplilication, we have several
# rules and depending on tensor's dimensions, the way the multiplication is 
# carried out changes.
# 
# sidenote
# To get the most out of this section without confusing rules, we start simple
# and cover the basics and later on we expand on them.

# sidenote2:
# We are going to see how to do linear algerba matrix multiplication. 
# This is what we usually mean by matrix multiplication.

# In PyTorch, just like Numpy, matrix multiplication can be performed using either
# the `@` operator or `torch.matmul()`. Both are equivalent/interchagable.

X = torch.tensor([[1., 2.],
                  [3., 4.]])

Y = torch.tensor([[5., 6.],
                  [7., 8.]])

print("\nTensor X:")
print(X)

print("\nTensor Y:")
print(Y)

print("\nElement-wise multiplication (X * Y):")
print(X * Y)

print("\nMatrix multiplication (X @ Y):")
print(X @ Y)

print("\nMatrix multiplication (torch.matmul(X, Y)):")
print(torch.matmul(X, Y))

# Vectors behave slightly differently.
# Multiplying two vectors with `@` computes their dot product.

v1 = torch.tensor([1., 2.])
v2 = torch.tensor([10., 20.])

print("\nVector v1:")
print(v1)

print("\nVector v2:")
print(v2)

print("\nElement-wise multiplication (v1 * v2):")
print(v1 * v2)

print("\nDot product (v1 @ v2):")
print(v1 @ v2)

# `torch.matmul()` is PyTorch's most general matrix multiplication method/routine.
# Besides vectors and matrices, it also supports batched matrix
# multiplication and automatically applies broadcasting when needed.
#
# PyTorch also provides `torch.mm()`, which is a specialized version that only
# accepts *two-dimensional* matrices. Some developers prefer using `torch.mm()`
# when they want to ensure that only *matrix-matrix* multiplication is allowed.
#
# We'll revisit batched matrix multiplication later when we work with batches
# of data and neural network models.

# sidenote:
# Think of `mm` in `torch.mm` as to *m*atrix-*m*atrix (hence `mm`) multplication,
# likewise, `torch.bmm` as the batched matrix-matrix multiplication method/routine.
# it should help you remember that they only work on matrices!

# deeper example review
# Below we revisit the example we glanced over just now and pay more attention
# to some details we might have missed, to get a better understanding on what
# is going on.

X = torch.arange(6.).view(2,3) + 1
Y = torch.arange(2.).view(2,) + 1

show_tensor("X",X)
show_tensor("Y",Y)

# Pay careful attention to the dimensions and how the multiplication is carried out!
# X * Y

Z = torch.matmul(Y, X)
show_tensor("Y @ X",Z)

# As you just saw, `Y` was broadcasted so it could be multiplied by `X`
# `Y` is 1D and it is treated as `(1,2)` so the dimensions between two tensors
# are valid. Thus the output becomes a `1x3` tensor! 

# now lets transpose X and see what changes! we use .t() for transposing!

Z = torch.matmul(X.t(), Y)
show_tensor("X.t() @ Y",Z)

# now in this example, the tensor_2 again is broadcasted and this time  
# it is treated as (2x1) tensor so the dimensions between tensors are valid 
# as you can see the output is a tensor of 3x1.

# note that, since one of our tensors is 1D, the result is also shown as 1D
# if we explictly make the tensor_2 2D, the output will follow suit as well
# here we get a row vector which is (1,3) (A row vector is a one-dimensional array (or vector) that has a single row and multiple columns)

Z = torch.matmul(Y.view(1,2), X)
show_tensor("X",X)
show_tensor("Y",Y)
show_tensor("Y.view(1,2) @ X",Z)

# and likewise we get (3,1) or a column vector here
Z = torch.matmul(X.t(), Y.view(2,1))
show_tensor("X.t()",X)
show_tensor("Y",Y)
show_tensor("t() @ Y.view(2,1)",Z)

# we can do all of these using mm! 
# print('\nUsing torch.mm:')
print_header('Using torch.mm')

# mm is short for matrix matrix multipliplication, so all dimensions must be specified!
# unlike torch.matmul, there is no broadcasting going on here!
# We must specify all dimensions ourselevs thats why we used `.view()` to
# reshape our tensor to the form it needs to be to have a proper multiplication!
Z = torch.mm(Y.view(1,2), X)
show_tensor("torch.mm(Y.view(1,2), X)",Z)

# now if we transposed X:
Z = torch.mm(X.t(), Y.view(2,1))
# print(f'data_1.t()(3x2) * data_2(2x1): {Z}')
show_tensor("X.t()",X)
show_tensor("Y",Y)
show_tensor("torch.mm(X.t(), Y.view(2,1))",Z)

# 7.3 Broadcasting
# Broadcasting is one of PyTorch's most convenient features. It allows tensors
# with compatible shapes to participate in the same operation without
# explicitly reshaping or copying data. 
# Broadcasting, simply put, means to broad cast dimensions in a tensor so 
# it becomes identical to the other tensor, so their operation can continue
# as if they were identical in shape from the begining.

matrix = torch.tensor([[1., 2., 3.],
                       [4., 5., 6.]])

bias = torch.tensor([10., 20., 30.])

show_tensor("Matrix",matrix)
show_tensor("Bias",bias)

# In this example, we have a single Bias vector while we have 2 rows 
# in the matrix, in order for the Addition operation to carry on 
# bias needs to be replicated once along the row dimension, so each
# row in matrix, has a corrosponding row in bias. 
# In practice no copying happens when broadcasting takes place! 
# because PyTorch performs this expansion logically, so it does not actually
# duplicate the underlying data in memory.

print_header("Broadcasted Addition")
show_tensor("Matrix + Bias", matrix + bias, newline=False)

# Scalars are broadcast as well in the same fashion. an scaler will be
# replicated for as many elements as there are in the matrix so the addition
# can go ahead without any issues.
print_header("Scalar Broadcasting",)
show_tensor("Matrix + 100", matrix + bias, newline=False)

# Note that, in order for the broadcasting to work, the two tensor must have 
# compatible dimensions. That is, each tensor must have at least one dimension
# and starting from the trailing dimensions, each pair of dimensions must either:
# be equal or one of them must be 1 or one of them must not exist.
# 
# In short, if a PyTorch operation supports broadcast, then its Tensor arguments
# can be automatically expanded to be of equal sizes (without making copies of the data).

# Deeper example review 
# lets review some of the points we discussed just now

X = torch.empty(5,7,3)
Y = torch.empty(5,7,3)
# Tensors with the same shapes are always broadcastable (i.e. the above rules always hold)

X = torch.empty((0,))
Y = torch.empty(2,2)
# x and y are not broadcastable, because x does not have at least 1 dimension

# can line up trailing dimensions
X = torch.empty(5,3,4,1)
Y = torch.empty(  3,1,1)
# x and y are broadcastable.
# 1st trailing dimension: both have size 1
# 2nd trailing dimension: y has size 1
# 3rd trailing dimension: x size == y size
# 4th trailing dimension: y dimension doesn't exist
# 
# but:
X = torch.empty(5,2,4,1)
Y = torch.empty(  3,1,1)
# x and y are not broadcastable, because in the 3rd trailing dimension 2 != 3

# sidenote about backwards compatibility:
# Early versions of PyTorch (i.e. <1.0) allowed certain *pointwise/elementwise* functions to 
# execute on tensors with different shapes, as long as the number of elements in each tensor
# was equal. 
# The pointwise operation would then be carried out by viewing each tensor as 1-dimensional. 
# PyTorch now supports broadcasting and the “1-dimensional” pointwise behavior is considered 
# deprecated and will generate a Python warning in cases where tensors are not broadcastable, 
# but have the same number of elements.
# Note that the introduction of broadcasting can cause backwards incompatible changes in the 
# case where two tensors do not have the same shape, but are broadcastable and have the same 
# number of elements.
# 
# For Example:
# torch.add(torch.ones(4,1), torch.randn(4))
# would previously produce a Tensor with size: torch.Size([4,1]), but now produces a Tensor 
# with size: torch.Size([4,4]).
# In order to help identify cases in your code where backwards incompatibilities introduced
# by broadcasting may exist, you may set :
# `torch.utils.backcompat.broadcast_warning.enabled` to `True`, which will generate a python 
# warning in such cases.
# For Example:
# torch.utils.backcompat.broadcast_warning.enabled=True
# torch.add(torch.ones(4,1), torch.ones(4))
# __main__:1: UserWarning: self and other do not have the same shape, but are broadcastable, 
# and have the same number of elements.
# Changing behavior in a backwards incompatible manner to broadcasting rather than viewing as 
# 1-dimensional.

# torch.utils.backcompat.broadcast_warning.enabled=True
A = torch.randn((4,1))
v = torch.randn((4,))

show_tensor("A", A)
show_tensor("v", v)

torch.utils.backcompat.broadcast_warning.enabled=True
show_tensor("A + v", A + v) 
# results in a 4x4 tensor

# this line fails:
# print(A @ v) 
show_tensor("v @ A", torch.matmul(v,A))
show_tensor("A @ v.view(1,-1)", torch.matmul(A, v.view(1,-1)))
# show_tensor("v.view(-1,1) @ A", torch.matmul(v.view(-1,1), A))

# 7.4 Reduction Operations
# Reduction operations summarize many values into fewer values (often a single
# value). These operations are extremely common when computing statistics and
# defining loss functions during neural network training.

# print("\nSum:")
# print(MATRIX.sum())
show_tensor("Sum",matrix.sum())

# print("\nMean:")
# print(MATRIX.mean())
show_tensor("Mean",matrix.mean())

# print("\nMaximum:")
# print(MATRIX.max())
show_tensor("Maximum",matrix.max())

# print("\nMinimum:")
# print(MATRIX.min())
show_tensor("Minimum",matrix.min())

# sidenote:
# To convert a Pytorch scaler into a Python number, we use .item()
print_header("Raw Pytorch Scalar")
print(f"Sum = {repr(matrix.sum())}")

print_header("Aftert using .item()")
print(f"Sum = {repr(matrix.sum().item())}")


# Reductions can also be performed along a specific dimension.
show_tensor("Column sum(dim=0)",matrix.sum(dim=0))

show_tensor("Column sum(dim=1)",matrix.sum(dim=1))

show_tensor("Column mean(dim=0)",matrix.mean(dim=0)) 
#%% 7.6 Advanced Matrix Multiplication
#
# Previously, we introduced matrix multiplication using the `@`
# operator and `torch.matmul()` and briefly talked about `torch.mm`.
# For most everyday PyTorch code, these are all you'll ever need.
#
# However, PyTorch actually provides several related matrix multiplication
# functions. Although they may appear redundant at first, each exists for a
# specific purpose and understanding their differences will help you read
# existing PyTorch code and avoid common shape-related errors.
#
# Throughout this section, pay close attention to the dimensions of each
# tensor and how they influence the multiplication being performed.

# 7.6.1 The @ Operator
# The @ operator was introduced in Python 3.5 specifically for matrix
# multiplication. In PyTorch, it is simply syntactic sugar for
# `torch.matmul().` Both produce identical results.

A = torch.tensor([[1., 2.],
                  [3., 4.]])

B = torch.tensor([[5., 6.],
                  [7., 8.]])

# print("Using the @ operator:")
# print(A @ B)
show_tensor("Using @ operator (A @ B)", A @ B)
show_tensor("Using torch.matmul (torch.matmul(A, B))", torch.matmul(A, B))

# 7.6.2 torch.matmul()
# As we already pointed out, `torch.matmul()` is PyTorch's most general 
# matrix multiplication routine. Depending on the dimensions of its inputs, 
# it automatically performs the appropriate type of multiplication.
#
# It supports:
#
#   • Vector x Vector        -> Dot Product
#   • Matrix x Vector
#   • Vector x Matrix
#   • Matrix x Matrix
#   • Batched Matrix x Matrix
#   • Broadcasting across batches
#
# For most applications, `torch.matmul()` (or the `@` operator) is the
# recommended choice.

# Vector x Vector
# Multiplying two vectors produces their dot product.

v1 = torch.tensor([1., 2., 3.])
v2 = torch.tensor([4., 5., 6.])

# print("\nVector x Vector:")
# print(torch.matmul(v1, v2))
show_tensor("Vector x Vector", torch.matmul(v1, v2))

# Matrix x Vector
# Every row of the matrix is multiplied with the vector.
M = torch.tensor([[1., 2., 3.],
                  [4., 5., 6.]])

v = torch.tensor([10., 20., 30.])

# print("\nMatrix x Vector:")
# print(torch.matmul(M, v))
show_tensor("Matrix x Vector", torch.matmul(M, v))

# Vector x Matrix
# The vector behaves as a row vector.
v = torch.tensor([10., 20.])

M = torch.tensor([[1., 2., 3.],
                  [4., 5., 6.]])

show_tensor("Vector x Matrix", torch.matmul(v, M))

# Matrix x Matrix
# This is the standard matrix multiplication most people are familiar with.

A = torch.tensor([[1., 2.],
                  [3., 4.]])

B = torch.tensor([[5., 6.],
                  [7., 8.]])

show_tensor("Matrix x Matrix", torch.matmul(A, B))

# Batched Matrix Multiplication
# One of `torch.matmul()`'s greatest strengths is its ability to multiply
# entire batches of matrices simultaneously.
#
# Instead of multiplying one pair of matrices, PyTorch multiplies every
# corresponding pair in the batch.

batch_a = torch.randn(2, 3, 4)
batch_b = torch.randn(2, 4, 5)
result = torch.matmul(batch_a, batch_b)
# print("\nBatched Matrix Multiplication:")
# print(result.shape)
show_tensor("Batch A", batch_a)
show_tensor("Batch B", batch_b)
show_tensor("Batched Matrix Multiplication", result)

# Conceptually this performs:
#
# batch_a[0] @ batch_b[0]
# batch_a[1] @ batch_b[1]
# ...
# batch_a[9] @ batch_b[9]

# Broadcasting
# `torch.matmul()` also supports broadcasting across batch dimensions.

batch_a = torch.randn(2, 3, 4)
shared_b = torch.randn(4, 5)
result = torch.matmul(batch_a, shared_b)
show_tensor("Batch A", batch_a)
show_tensor("shared_B", shared_b)
show_tensor("Broadcasted Matrix Multiplication", result)

# Here the same matrix is reused for every batch:
#
# batch_a[0] @ shared_b
# batch_a[1] @ shared_b
# ...
# batch_a[9] @ shared_b
#
# This behavior is extremely common in modern deep learning models.

# 7.6.3 torch.mm()
# `torch.mm()` is a specialized version of matrix multiplication that accepts
# only two-dimensional matrices.
#
# Unlike `torch.matmul()`, it does NOT support vectors, batches, or broadcasting.

A = torch.randn(2, 3)
B = torch.randn(3, 4)

show_tensor("A", A)
show_tensor("B", B)
show_tensor("torch.mm(A, B)", torch.mm(A, B))

# The following would raise an error because `torch.mm()` only accepts
# matrices:
#
# batch_a = torch.randn(2, 3, 4)
# batch_b = torch.randn(2, 4, 5)
#
# torch.mm(batch_a, batch_b)

# 7.6.4 torch.bmm()
# `torch.bmm()` performs batched matrix multiplication.
#
# Unlike `torch.matmul()`, it requires both tensors to already have matching
# batch dimensions and does NOT perform broadcasting.

batch_a = torch.randn(6, 3, 4)
batch_b = torch.randn(6, 4, 5)

show_tensor("batch_A", batch_a)
show_tensor("batch_B", batch_b)
show_tensor("torch.bmm(batch_A, batch_B)", torch.bmm(batch_a, batch_b))

# This works because both tensors contain 6 matrices.
# The following would fail because `torch.bmm` doesnt support broadcasting
# and the second tensor doesnt have a matching batch dimension either.
#
# batch_A = torch.randn(6, 3, 4)
# shared_B = torch.randn(4, 5)
#
# torch.bmm(batch_A, shared_B)

# 7.6.5 Which One Should we Use?
# For most PyTorch code, we use the `@` operator whenever possible.
# We use `torch.matmul()` if we prefer the functional API or need to call
# the operation programmatically.
# We only use `torch.mm()` if we intentionally want to restrict 
# our code to two-dimensional matrix multiplication.
# And finaly we use `torch.bmm()` when we are working with batches
# of matrices that already have matching batch dimensions and 
# we do not want broadcasting.
#
# In practice, you'll see `@` and `torch.matmul()` far more frequently
# than `torch.mm()` or `torch.bmm()`.

# sidenote:
#                   Matrix Multiplication Summary
#     Input Shapes              Output Shape           Result
# ------------------------------------------------------------------------------
# (n)       @   (n)        ->         ()        Vector x Vector -> Dot Product 
#
# (m, n)    @   (n)        ->         (m)            Matrix x Vector
#
# (n)       @   (n, p)     ->         (p)            Vector x Matrix
#
# (m, n)    @   (n, p)     ->        (m, p)          Matrix x Matrix
#
# (...,m,n) @   (...,n,p)  ->       (...,m,p)    Batched Matrix Multiplication
#
# `torch.matmul()` automatically handles all of the above cases.
# The `@` operator is simply shorthand for `torch.matmul()`.

# %% Section 8: Joining, Splitting and Repeating Tensors (Concatenation, Stacking and Repeating)
# 
# One of the core operations that we'll often find ourselves using repeatedly 
# is tensor joining and splitting. As our models become more complex, we'll 
# need to combine multiple tensors into a single larger tensor or split a 
# tensor into smaller pieces either purely performance motivated or because
# of an algorithic requirement.
# These operations are common when preparing datasets, building mini-batches,
# processing model outputs, or implementing neural network architectures.
#
# In this section, we'll learn how to concatenate, stack, split, and chunk
# tensors, and understand when each operation is the appropriate choice.
#
# PyTorch offers many functions to this end. torch.concatenate(), and its alias
# torch.cat(), joins two or more tensors together along the given axis.

A = torch.tensor([[1, 2],
                  [3, 4]])

B = torch.tensor([[5, 6],
                  [7, 8]])

show_tensor("A", A)
show_tensor("B", B)

# Concatenation joins tensors along an existing dimension.
# print("\nConcatenate along rows (dim=0):")
# print(torch.cat((A, B), dim=0))
show_tensor("Concatenate along rows (dim=0)", torch.cat((A, B), dim=0))


# print("\nConcatenate along columns (dim=1):")
# print(torch.cat((A, B), dim=1))
show_tensor("Concatenate along columns (dim=1)", torch.cat((A, B), dim=1))

# Stacking on the other hand creates a NEW dimension.
print("\nStack along dim=0:")
print(torch.stack((A, B), dim=0))
show_tensor("Stack along dim=0", torch.stack((A, B), dim=0))

# print("\nStack along dim=1:")
# print(torch.stack((A, B), dim=1))
show_tensor("Stack along dim=1", torch.stack((A, B), dim=1))

# Notice the difference:
# cat()   -> joins existing dimensions.
# stack() -> creates a brand-new dimension.

# sidenote:
# `torch.cat` is an alias for `torch.concatenate()`

# For splitting we use `torch.split()` to divide a tensor into smaller tensors.
# We can specify how many `splits` we want in the output tensor
# by specifying the number of splits directly, or specify the exact size for each
# split in the output using a list of tuple.
# note if the tensor can not be divided equally, the last split will be smaller.

tensor = torch.arange(12).reshape(3, 4)

show_tensor("Original tensor", tensor)

# split the tensor into two, along the column dimension (dim=1)
parts = torch.split(tensor, 2, dim=1)

print_header("Split into groups of 2 columns")
for i, part in enumerate(parts):
    print(f"Part {i}:")
    print(part)

# specify each split size individually
parts = torch.split(tensor, [1,2,1], dim=1)
print_header("differently sized splits along columns")
for i, part in enumerate(parts):
    show_tensor(f"Part {i}", part)

# We usually use `.torch.split` to specify each chunks size separately.

# Chunking
# `torch.chunk()` splits a tensor into *approximately equal-sized* pieces.
# Unlike `torch.split()` we specify the number of chunks not the size of each chunk.
# note we we said *approximately* equally-sized, because if an even split 
# is not possible, the chunks may have different sizes. 
# 

chunks = torch.chunk(tensor, chunks=3, dim=0)
print_header("Chunk into 3 equal row chunks")
for i, chunk in enumerate(chunks):
    print(f"Chunk {i}:")
    print(repr(chunk))
    # show_tensor(f"Chunk {i}", chunk)

tensor_odd = torch.arange(15).reshape(3, 5)
chunks = torch.chunk(tensor_odd, chunks=2, dim=0)
print_header("Chunk into 2 diffrenly sized chunks")
for i, chunk in enumerate(chunks):
    # print(f"Chunk {i}:")
    # print(repr(chunk))
    show_tensor(f"Chunk {i}", chunk)


# Also note when using .chunk, just like `.split()` the chunks 
# keep the dimension, that is chunk is (1,1,4), not just 4! 
# the distinction matters later on. 

chunks = torch.chunk(tensor.unsqueeze(0), chunks=3, dim=1)
print_header("Chunks keeps the dimension")
for i, chunk in enumerate(chunks):
    # print(f"Chunk {i}:")
    # print(repr(chunk))
    show_tensor(f"Chunk {i}", chunk)

# We usually use `torch.chunk` when we want to divide work into larger
# groups like e.g. split a batch across several GPUs or split features
# into several parts, where having differently sized chunks doesnt pose
# any issues.

# Unbinding
# `torch.unbind()` works similarly to split and .chunk() but with the difference
# we cant specify the number of splits, it returns exactly one tensor per element
# along the specified dimension and it removes the dimension when returning the split.

rows = torch.unbind(tensor, dim=0)
print_header("Rows returned by unbind(dim=0):")
for i, row in enumerate(rows):
    # print(repr(row))
    show_tensor(f"row {i}", row)

# note it returns all elements along the row dimension, each as a separate row
# the dimension is also removed and we get (4,) instead of (1,4) for each row!

# sidenote:
# when should we use these? 
# we usually use `torch.split()` when we want to specify the *size* of each split.
# we usually use `torch.chunk()` when we want to specify the *number* of chunks.
# Chunks are approximately equal-sized.
# we usually use `torch.unbind()` when we want to return one tensor for every slice
# along a dimension. its equivalent to chunking into single slices followed by 
# squeezing away the split dimension.

# Repeating and Expanding
# Another common operation is making a tensor larger. PyTorch provides two
# related operations for tensors:
# `Tensor.repeat()` and `Tensor.expand()`.
# 
# `.repeat()` physically copies the tensor's data.
# while `expand()` creates a larger *view* of the same data without copying it.
#
# Both can produce tensors with the same shape, but they have very different
# performance and memory characteristics and are intended for different use cases.

vector = torch.tensor([1, 2, 3])

show_tensor("Original vector", vector)

# `Tensor.repeat(*sizes)` accepts a list or tuple of sizes for each dimension
# and repeats the tensor along those dimension based on respective sizes.
# note the arguments specify *how many times* each dimension should be repeated,
# its not the final size.
#
# For a 1D tensor we provide one repeat factor.
show_tensor("vector.repeat(3)", vector.repeat(3))

# For higher-dimensional tensors we provide one repeat factor per dimension.
matrix = vector.unsqueeze(0)
show_tensor("matrix", matrix)

show_tensor("matrix.repeat(2, 1)", matrix.repeat(2, 1))
show_tensor("matrix.repeat(2, 2)", matrix.repeat(2, 2))

# Notice how each repeat factor affects one dimension independently.
#
# matrix.shape is (1, 3)
# when we do repeat(2, 1) we get shape(2, 3)
# when we do repeat(2, 2) we get shape(2, 6)
# only that dimension gets repeated that many times. 

# Unlike broadcasting, `repeat()` allocates new memory and physically copies
# every element into the output tensor. The repeated values are completely
# independent copies.

# Expanding
# `Tensor.expand(*sizes)` looks similar to repeat(), but the arguments have
# a completely different meaning.
# Instead of specifying repeat factors, we specify the desired *final size*
# of each dimension.
# Passing -1 as the size for a dimension means not changing the size of that
# dimension.

row = torch.tensor([[1, 2, 3]])
show_tensor("row", row)

expanded = row.expand(4, 3)
show_tensor("row.expand(4, 3)", expanded)

# Even though the result looks identical to `repeat()`, no copies were made.
# expand() returns a *view* of the original tensor by reusing the same memory.
# This makes it much more memory efficient than repeat().

# A dimension can only be expanded if its size is 1. Since there is only one
# row, PyTorch can safely pretend that same row exists multiple times.
#
# The following would raise an error because the first dimension already has
# size 2, so PyTorch cannot expand it without creating new data.
#
bad = torch.tensor([[1, 2, 3],
                    [4, 5, 6]])
try:
    show_tensor("bad",bad)
    bad.expand(4, 3)
except Exception as ex:
    print_header('bad.expand(4, 3) Failed')
    # print(ex.msg)
    
# Because expand() shares memory with the original tensor, modifying the
# original tensor changes every expanded view.

row[0, 0] = 99
show_tensor("Modified original row", row)
show_tensor("Expanded view reflects the change", expanded)

# sidenote:
# As we just saw because expand returns a view, more than one element of an
# expanded tensor may refer to a single memory location. 
# As a result, in-place operations (especially ones that are vectorized) may
# result in incorrect behavior.
# If you need to write to the tensors, please clone them first or use repeat()
# or expand_copy()!

# sidenote 
# summary:
# repeat(*sizes) -> the arguments specify how many times to repeat each
#                   dimension. A new tensor is allocated and the data is copied.
#
# expand(*sizes) -> the arguments specify the desired output size. No new
#                   memory is allocated; instead a broadcasted view is returned.
#
# We usually use repeat() when independent copies are required.
# We usually use expand() when broadcasting is sufficient and we want to avoid
# unnecessary memory allocations.


# Repeat Interleave
# Besides `Tensor.repeat()`, PyTorch also provides `torch.repeat_interleave()`.
#
# While `repeat()` duplicates entire dimensions, `repeat_interleave()` repeats
# individual elements along a dimension. its behave's like NumPy's `repeat()`.
#
# For the arguments, we specify the `repeats` argument, with either a single integer
# or a tensor specifying how many times each element should be repeated.
# the dim argument specifies the dimension along which elements are repeated. 
# If we use None, (i.e. dont fill it, omit it), the input tensor is first flattened.
# 
vector = torch.tensor([1, 2, 3])
show_tensor("Original vector", vector)

show_tensor("torch.repeat_interleave(vector, repeats=2)",
    torch.repeat_interleave(vector, repeats=2))

# Unlike `repeat()`, which repeats the entire tensor
# like e.g. 
# [1, 2, 3] -> repeats=2 -> [1, 2, 3, 1, 2, 3]
#`repeat_interleave()` repeats each individual element
# [1, 2, 3] -> repeats=2 -> [1, 1, 2, 2, 3, 3]

# We can also specify a different repeat count for every element by 
# specifying a tensor with counts for each element.
show_tensor("Different repeat counts",
    torch.repeat_interleave(vector, repeats=torch.tensor([1, 3, 2])))

# We usually use `repeat_interleave()` when duplicating labels, indices,
# or individual samples, rather than entire tensor dimensions.

# Expand Copy
# `Tensor.expand_copy()` behaves similarly to `expand()`, but instead of
# returning a view, it allocates new memory and copies the expanded result.
#
# Just like expand(), only dimensions whose size is 1 may be expanded.
# However, unlike expand(), the returned tensor owns its own storage.

row = torch.tensor([[1, 2, 3]])

show_tensor("Original row", row)

expanded = row.expand(4, 3)
expanded_copy = torch.expand_copy(row, [4, 3])

show_tensor("expand()", expanded)
show_tensor("expand_copy()", expanded_copy)

# The outputs look identical, but expand() shares memory with the original
# tensor while expand_copy() creates an independent tensor.

row[0, 0] = 99

show_tensor("Modified original row", row)
show_tensor("expand() shares storage", expanded)
show_tensor("expand_copy() owns its data", expanded_copy)

# sidenote
# summary
# expand() returns a broadcasted view without copying data.
# expand_copy() performs the same expansion but allocates new memory.
#
# We usually use expand() when a read-only broadcasted view is sufficient,
# and expand_copy() when we need an expanded tensor that can be modified
# independently of the original.

#%% Section 9: Seeding & Reproducibility (RNG Management)
# before we continue, its worth taking a bit of time and learn about generators
# and seeding. 
# sometimes we want to produce determinstic output, for various reasons, ranging
# from debugging purposes, to repreducibility purposes. 
# therefore in order to have a repreducible output, we need to take control of the 
# randomness in our operations. 
# whenever we work with operations that involve random numbers, setting the seed for
# the RNG (random number generator) to use the same seed allows us to generate the 
# same sequence of numbers again and again.
# 
# Nearly all libraries that involve such operations, offer ways to set the seed,
# including both numpy and pytorch.
# 
# By default, PyTorch uses a single global random number generator (RNG) which we
# can simply specify a seed for by using `torch.manual_seed(number)`.
# Interestingly there is `torch.seed()` that generates a random seed automatically 
# and sets the global RNG and then returns it!
# 
# note that `torch.manual_seed()` both sets the seed for global RNG and returns
# the global generator! while `torch.seed()` only returns the random seed that
# was used to set the global RNG.

seed = 15
global_rng_generator = torch.manual_seed(seed)

# now if we try and create a random tensor, it will always have the same values.
# its as if we specified generator=global_rng_generator below. 
random_tensor = torch.randn(size=(2,2))
show_tensor(f"Random tensor with manual seed({seed})", random_tensor)
# print(f'{random_tensor=}')
# we will always get 
# random_tensor=tensor([[-0.7056,  0.6741],
#                       [-0.5454,  0.9107]])
# 

# Now if we want to see what seed was used initially 
# We can easily use `torch.initial_seed()` or `torch.random.initial_seed()`:
print_header(f"Displaying Initial seed({seed})")
print(f'torch.initial_seed():  {torch.initial_seed()}')

# When we set the seed using `torch.manual_seed()`, or `torch.seed()`
# or any other way, we can retrieve the random state as a tensor using either 
# `torch.get_rng_state()` or `torch.random.get_state()`.
# The state tensor contains information about the internal state of the 
# random number generator (RNG). 
# This RNG state is returned as a `torch.ByteTensor` and it contains all 
# the necessary bits to restore the RNG to a specific point in time. 
# We can save this into a file and later on restore it and set it back 
# using `torch.random.set_state(saved_state)`.
# But an easier way would be to simply use `torch.initial_seed()`
# print(f'\n{torch.random.get_rng_state()=}')
show_tensor("Random RNG State", torch.random.get_rng_state())

# Note that the rng_state doesnt only contain the `initial_seed`, it has 
# other information as well. so its not like a byte representation of a 
# single seed number!
# 
# DeepDive Note:
# If we convert the `initial_seed()` into a bytes array and look at it
# we can see our initial seed there, at the begining of the array but
# the rest will be zeros whereas in the actual `rng_state` they are nonzero
# values: 
# 
# lets see this in action.
# generate a random seed
seed = torch.seed()
#
# retrieve the initial seed used
init_seed=torch.initial_seed()
#
# now get the rng_state
rng_state = torch.get_rng_state()
print(f'seed :          {seed}')
print(f'initial seed:   {init_seed}')
#
# rng_state is a tensor of size torch.Size([5056])
print(f'rng_state.shape:{rng_state.shape}')
#
# By default Pytorch doesn't print all the elements and it might give us
# the impression that only the few starting elements are nonzero and the
# rest are zeros! this is obviouly wrong! see the rest
# print(f'{rng_state=}')
show_tensor("rng_state",rng_state)
#
# to convert our seed into bytes, we take the byte length as well
# our system is little endian, so we specify that as well otherwise, the
# result would be messed up (kind of flipped) due to cpu endian-ness!
seed_bytes = seed.to_bytes(rng_state.shape[0],'little')
#
# now we create a numpy/torch array out of our bytes, 
# Thanks to torch implementing numpy operations, they are identical here:
# seed_bytearray = np.frombuffer(seed_bytes, dtype=np.uint8)
seed_bytearray = torch.frombuffer(seed_bytes, dtype=torch.uint8)
#
# length checks out as well
print(f'seed_bytearray.shape:   {seed_bytearray.shape}')
print(f'seed_bytearray:         {seed_bytearray}')
print(f'rng_state:              {rng_state}')
#
# seems pretty similar to rng_state right? not so fast
# when we compare them we see that they are not equal!
print(f'Is rng_state == seed_bytearray? {torch.equal(rng_state, seed_bytearray)}')
#
# now lets try to see them in their full glory!
torch.set_printoptions(profile='full')
# print(f'{seed_bytearray=}')
show_tensor("seed_bytearray", seed_bytearray)
#
# while our `rng_state` is quit different after the few early elements
# which tells us it has more information other than a simple seed!
# print(f'{rng_state=}')
show_tensor("rng_state", rng_state)
#
# so the thing to remember is, to either store the seed-number, or the
# `rng_state` for resuming purposes later on. (we usually use seed only!)
#
# lets reset the printoptions back to its defaults
torch.set_printoptions(profile='default')

# We can create generators and instead of using the global RNG, use 
# separate seeds for separate sections of our code.

# `torch.manual_seed` initializes the global rng, so we dont need 
# to grab the returned generator! this is what you see commonly used
# in many training tutorials and jupyter notebooks.
seed = 15
torch.manual_seed(seed)
random_tensor_1 = torch.randn(size=(2,2))
show_tensor(f"Using global random generator (seed={seed})", random_tensor_1)

# The issue with this approach however is, the global RNG is shared by all
# PyTorch random operations.
# 
# So every call to `torch.rand()`, `torch.randn()`, `torch.randint()`, etc.
# will use the global RNG and consume values from the same generator,
# basically changing the internal state and thus resulting in a different
# set of random values.
# 
# That is if we (or even another library we use) insert an
# additional random operation somewhere earlier in the program,
# after the `manual_seed()`, the global RNG advances and causes all
# subsequent random numbers to change.
#
# thats why we try to use a dedicated generator for any set of operations
# that we want determinstic behavior from.
# 
# A Generator object maintains its own RNG state, and is completely 
# independent of the global generator. By using our own generator
# and passing it explicitly, we guarantee that only the operations using
# that generator affect its state giving us locally deterministic behavior.
#  
# Other parts of our code are still free to use the global RNG without changing
# the random numbers produced by our generator.
#

torch.manual_seed(15)
random_tensor_1 = torch.randn(size=(2,2))
show_tensor("Using global RNG before nn.Linear ", random_tensor_1)

# Using the same seed, but with a new operation that uses randomness
# under the hood. 
torch.manual_seed(seed)

# Create a simple fully connected (Linear) layer.
# Even though we're only defining the layer, PyTorch randomly
# initializes its weights and bias.
_ = torch.nn.Linear(10, 5)
random_tensor_1 = torch.randn(size=(2,2))
show_tensor(f"Using global RNG After nn.Linear (seed={seed})", random_tensor_1)

# As you can see, simply creating a `torch.nn.Linear` layer changes
# the next random numbers produced by the global RNG.
#
# The reason is that `nn.Linear` randomly initializes its parameters
# (weights and bias) during construction, consuming values from the
# global random number generator.
#
# Note that this isn't limited to neural network layers. Many PyTorch
# operations use randomness internally. For example, creating a DataLoader
# with `shuffle=True` or applying random data augmentations or using layers
# such as Dropout will also consume random numbers.
#
# The important thing to remember (takeaway) is that not every operation 
# that advances the global RNG obviously "looks random". Because of this, 
# code elsewhere in a larger program or in a library we might happen to 
# use can unintentionally affect the sequence of random numbers our code receives.
#
# If we need a section of code to have its own isolated and reproducible
# stream of random numbers,i.e. determinstic output, we should create and
# use our own `torch.Generator`.

# Now lets create a new tensor this time using a dedicated generator.
# Generators can be built on any device, in fact, when using generators,
# the tensors and the generators assigned to them must live on the same device
# otherwise it would lead an error.

# lets create the generator on he same device as the tensors
device = 'cpu'
with torch.device(device) as device:
    generator = torch.Generator(device=device).manual_seed(5)
    random_tensor_2 = torch.randn(size=(2,2,), generator=generator)
    random_tensor_3 = torch.randn(size=(2,2,), generator=generator)
    show_tensor("random_tensor_1 (Global RNG)",random_tensor_1)
    show_tensor("random_tensor_2 (local generator)",random_tensor_2)
    show_tensor("random_tensor_3 (local generator)",random_tensor_3)
        
# and we get : 
#random_tensor_1=tensor([[-0.7056,  0.6741],
#                        [-0.5454,  0.9107]])
# random_tensor_2=tensor([[-0.4868, -0.6038],
#                         [-0.5581,  0.6675]])
# random_tensor_3=tensor([[-0.1974,  1.9428],
#                         [-1.4017, -0.7626]])

# As you can see, the global generator is used with our first tensor 
# while for the other two we used an explicit generator and their results 
# stay the same no matter how many times we run this!
# 
# sidenote:
# torch like numpy, has put all random related functionalities into random
# module, so while some of the main functionalities can be accessed through
# torch.* , we can always access them all at one place, which is torch.random
# just like numpy!

# note that to get true determinstic output, we usually need to set seed not only
# for torch, but python and numpy as well, especially if some other libraries we use 
# happen to use them. 
# so for custom operators, we might need to set python seed as well.
# we might want to do sth like this and seed the global numpy RNG as well as python's: 
# as well: 
import random 
seed = 15
random.seed(seed)
np.random.seed(seed)
torch.random.manual_seed(seed)
# and if we want to have dedicated generator we do 
generator = torch.Generator(device=device).manual_seed(5)

# sidenote:
# using the Generator returned by `torch.manual_seed()` is *not* the
# same as creating a new `torch.Generator()`.
#`torch.manual_seed()` seeds the global RNG and returns a reference
# to it. Since the global RNG is shared, any PyTorch operation that
# uses randomness can advance its state, changing the generated random numbers.
# To get an isolated stream of random numbers, we need to create our
# own Generator with `torch.Generator().manual_seed(seed)`.

#%% sidenote - appendix?:
# As the official documentation says: 
# Some applications and libraries may use NumPy Random Generator objects, 
# not the global RNG (https://numpy.org/doc/stable/reference/random/generator.html),
# and those will need to be seeded consistently as well.
# 
# what does it mean really?
# This is refering to a recent change in numpy where it introduced a new random 
# number generation system that provides more flexibility and features than the
# older global RNG.
# In the new system, instead of relying on the global state (as in `np.random`), 
# NumPy now encourages using explicit random generator objects (instances of `numpy.random.Generator`).
# These generator objects allow us to manage seeds, distributions, and other properties independently.
# 
# therefore if our code interacts with NumPy (directly or indirectly), it's now 
# essential to seed these generator objects consistently.
# also setting the global seed (e.g., `np.random.seed(0)`) won't necessarily affect 
# these generator objects.
# and finally to ensure reproducibility, we would need to set the seed for both PyTorch
# (using `torch.manual_seed(0)`) and NumPy (using `np.random.seed(0)`).
# basically when working with numpy alongside pytorch, we need to be aware of these 
# separate random generator objects and seed them consistently for reproducible results.

# I also need to mention something important that getting determinstic output
# when it comes to cuda is not always as plain and simple as the cpu version
# becasue of the nature of cuda, trying to get determinstic output will not be easy to
# say the least and some times not possible, and for the cases where its possible it
# may very well result in degraded performance. 
# read https://pytorch.org/docs/stable/notes/randomness.html

# see, the deterministic behavior in pytorch refers to ensuring that given the same 
# input, the same sequence of operations will produce the same output.
# However, achieving complete determinism across different releases, or various
# platforms can be challenging to say the least especially when it comes to cuda.
# 
# note that the determinstic behavior we talk about here, is usually bound to software/hardware. 
# that is, we expect that given the same input, and same sequence of operations, 
# when run on the same software and hardware, we always get the same output. 
# This is an important implication (we see why this is the case when something like
# cuda is involved)
# 
# The cuDNN library, used by CUDA convolution operations, can introduce nondeterminism.
# When a cuDNN convolution is called with new size parameters, it runs multiple 
# convolution algorithms to find the fastest one. 
# Due to benchmarking noise and different hardware, the benchmark may select different
# algorithms on subsequent runs, even on the same machine(due to benchmarking noise as
# hardwre is the same here).
# Disabling the benchmarking feature with `torch.backends.cudnn.benchmark = False`
# causes cuDNN to deterministically select an algorithm, possibly at the cost of 
# reduced performance.(this is usually the case!)
#
# this is not all, aside from this, using `torch.use_deterministic_algorithms()`
# we can configure PyTorch to use deterministic algorithms instead of nondeterministic ones
# where available, and to throw an error if an operation is known to be nondeterministic
# (and without a deterministic alternative). 
# we can find the list of such operations here : https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithms 
# 
# note that some of these operations are only determinstic when they are on CPU. 
# so going determinstic aside from negatively affecting the performance, may not always
# be feasible. we can use `set_deterministic_debug_mode()` as an alternative interface
# for `torch.use_deterministic_algorithms()` which allows us to specify what to do when
# it faces nondeterminstic operations (do nothig(0), warn(1), or error out(2!)
# 
# for example trying to run the nondeterministic CUDA implementation of 
# `torch.Tensor.index_add_()` will throw an error while it will run ok on CPU mode!
#
#
# sidenote 1:
# we mentioned earlier that disabling CUDA convolution benchmarking ensures that 
# CUDA selects the same algorithm each time an application is run, however, that
# algorithm itself may be nondeterministic, unless either `torch.use_deterministic_algorithms(True)`
# or `torch.backends.cudnn.deterministic = True` is set. 
# The latter setting controls only this behavior, unlike `torch.use_deterministic_algorithms()` 
# which will make other PyTorch operations behave deterministically, too.
# so to make the whole process determinstic, `torch.use_deterministic_algorithms(True)`
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
# any operations that return a tensor with undefined values (from an 
# uninitialized memory), such as `torch.empty()`, which is then used as
# "an input" to some other operations can not be used when determinstic
# behavior is needed simply becasue they introduce randomness this way.
# In order to get around this issue, pytorch offers, 
# `torch.utils.deterministic.fill_uninitialized_memory()` which initializes 
# all unintialized memories with a known value. 
# its set by default when `torch.use_deterministic_algorithms(True)` while 
# this works, it comes with a huge price/overhead becasue of this extra 
# initialization step.
# if we dont use such cases that involve using unintialized memory as 'an input'
# to an operation, then this function can be set to False.
# 
# from official documentation: 
# Operations such as `torch.empty()`` and `torch.Tensor.resize_()` 
# can return tensors with uninitialized memory that contain undefined values.
# Using such a tensor as "an input" to another operation is invalid if determinism
# is required, because the output will be nondeterministic. 
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
# when using libraries like NumPy, make sure consistent seeds for their random number 
# generators as well.
# 
# Use `torch.manual_seed(0)` to seed the RNG for all devices (CPU and CUDA).
# For custom operators, set the Python seed with `random.seed(0)`.
# If relying on NumPy, use `np.random.seed(0)` (but be aware of NumPy Random Generator
# objects).
# 
# Remember that complete reproducibility isn't guaranteed, but these steps 
# limit sources of nondeterminism.
#
# deterministic operations may be slower than nondeterministic ones, 
# but they facilitate experimentation, debugging, and regression testing.
# Be cautious when sacrificing performance for reproducibility.
# while PyTorch provides tools to enhance determinism, achieving perfect 
# reproducibility across all scenarios is still challenging.
# check dataloaders as well (since this is still too early, well cover this
# later when we talk about them.)
#

#%%
# now lets create a simple hidden layer with a weight and bias and input 
# lets imlement a simple 1 layer and then 2 layer neural network! 
# dont worry here we will keep it simple! 
# our network has 5 neurons in its hidden layer, gets an input with 7 data points
# and creates 1 output 
# lets write the calculation for 1 step only!(forward propagation only)
X = torch.randn(size=(2,7))
W = torch.rand(size=(7,5)) 
b = torch.rand(size=(5,))
W_output = torch.rand(size=(5,1)) 
b_output = torch.rand(size=(1,))

def sigmoid(x):
    return 1/(1+torch.exp(-x))

output = sigmoid(torch.mm(X, W) + b)
output = sigmoid(torch.mm(output, W_output) + b_output)
print(f'{output}')

# we could further simplify this by using the built-in functions!
output = ((X @ W) + b).sigmoid()
output = ((output @ W_output) + b_output).sigmoid()
show_tensor("output", output)
#%%
# Before we end our discussion here, I'd like to talk a bit about logging some extra
# information about pytorch, and basically our stack. 
# its always a good idea to log some amount of information about the current stack 
# that is being used to produce something. 
# things like what version of pytorch, python, cuda and other things we might be 
# interested in and have an effect on our result. 
# its always a good idea to log such information so we know what configuration was
# used in getting certain results.
# One of the first things we would want to log is the version of pytorch we are using
# we can follow the python convention and use `torch.__version__` to get torch version 
print(f'{torch.__version__}') # 2.2.0+cu118
# or use the `version` module to access versions for other modules involved in the package
# such as the version of cude currently being shipped with, the hip version if any is installed
# the git commit number for this release and finally whether this is a debug build!
print(f'{torch.version.cuda=}')
print(f'{torch.version.hip=}')
print(f'{torch.version.debug=}')
print(f'{torch.version.git_version=}')
# we can use torch.cuda to access a slew of information related to the cuda stack
# ranging from simple versions, gpu count in the system, each gpu's name/capabilities
# and their temperature/utilization. 
# lets display all of the GPUs on the system. 
print(f'{torch.cuda.device_count()=}')

# dummy tensor for taking up some vram so we can test with the memory stats below!
dummy_tensor = torch.randn(size=(1000,512,512), dtype=torch.float64, device='cuda')

for i in range(torch.cuda.device_count()):
    name = torch.cuda.get_device_name(i)
    compute_capability = torch.cuda.get_device_capability(i)
    properties = torch.cuda.get_device_properties(i)
    print(f'{i}) {name}')
    print(f"  - Compute capability: {'.'.join([str(cc) for cc in compute_capability])}")
    print(f'  - Total memory:       {properties.total_memory/(2**30):.0f}GB')
    print(f'  - SM count:           {properties.multi_processor_count}')
    current_temp = torch.cuda.temperature()
    current_util = torch.cuda.utilization()
    print(f'  - current temp:       {current_temp}C')
    print(f'  - current util:       {current_util}%')

    # of course the temperature and utilization doesnt make sense for the start 
    # of our log but they definitely come in handy for monitoring our system/gpu status,
    # for example low uitilization could imply our batchsize need to change, or 
    # we need more worker threads or a faster media to read form, or maybe some 
    # parts of our model is not efficiently executing and we are io bound! 
    # we can also check them for health checks so that we dont burn our gpu!!

    # we can also see how much memory is currently taken!
    # torch.cuda.mem_get_info() gives us free memory out of the whole vram
    memory_usage = torch.cuda.mem_get_info(i)
    memory_usage = tuple(m//2**20 for m in memory_usage)
    print(f'  - available memory:   {memory_usage[0]:,}/{memory_usage[1]:,}MB')
    
    memory_reserved = torch.cuda.memory_reserved(0)//2**20
    max_memory_reserved = torch.cuda.max_memory_reserved(0)//2**20
    print(f'  - reserved memory:    {memory_reserved:,}MB')
    print(f'  - max reserved memory:{max_memory_reserved:,}MB')
    
    # list_gpu_processes() gives us a string showing all python processes 
    # that are currently using a specific GPU vram. its a string and 
    # we can print it rightaway or parse it and extract relavent information
    processes = torch.cuda.memory.list_gpu_processes(i)
    print(f'  - processes taking vram:')
    print(f'  - \t{processes=}')
    
    # memory stats offer more information in the form of a dictionary. 
    # we can use any key such as 'active.all.allocated' to only grab the 
    # information we want.
    memory_stat = torch.cuda.memory.memory_stats(i)
    print(f'{memory_stat=}')

    # or we could use memory_summary and get a nice table displaying all 
    # relavent information
    # summary = torch.cuda.memory.memory_summary(i)
    # print(f'\n  - vram summary:       \n{summary}')
    
# I guess thats enough for now. before we call it a day, lets see how we can 
# free gpu memory, this is especially handy in jupyeter notebooks where memory
# can quickly get consumed after several cells execution. 
# to free up memory we can use empty_cache().
torch.cuda.empty_cache()
print(f'- available memory:   {torch.cuda.mem_get_info(0)[0]//2**20}MiB')
# it didnt free anything it seems!
# The reason is empty_cache() as the name suggests, frees all "unused cached memory"
# The occupied GPU memory by tensors can not be freed this way. 
# Therefore, we cant use this to increase the amount of GPU memory available for PyTorch.
# So what do we do? 
# In order to make this work for us, we need to delete the variables that take up
# vram or somehow make them refer to sth else so that the memory chunk they are 
# referring to can be reclaimed. 
#
# Note that sometimes this doesnt work either, when this happens, this is most 
# probably a case of memory leakage, mutiple variables pointing to the same 
# memory chunk, etc.
# So we need to watch out for these cases as well.(one of such cases that we 
# will get to later happens during training, like appending, adding loss, 
# total_loss += loss, where it should have been total_loss += loss.item() otherwise, 
# the whole computation graph is being added each time instead of the loss value!
# which leaks memory (and takes up more vram as trainibg continues)
# 
# Now back to what we were doing, by deleting the tensor or setting it to something
# like None, we mark that chunk of memory ready for being garbage collected! 
# (if its referenced only once) if not, and if we have two variables refering to 
# the same object, both of them needs to be deleted or made to point to sth else 
# (so the ref count becomes 0 and it can be freed))
# lets make a second variable to also refer to dummy_tensor to see this in action
dummy_tensor2 = dummy_tensor
# del dummy_tensor
dummy_tensor = None
# see if we only delete dummy_tensor, the mmeory wont be freed
dummy_tensor2 = None
# and following a empty_cache() call we may reclaim that memory!
torch.cuda.empty_cache()
print(f'- available memory:   {torch.cuda.mem_get_info(0)[0]//2**20}MB')

# also note that, sometimes this process gets a bit more involved,
# but the underlying issue stays the same, multiple references to the same memory
# chuncks, or memory leak. one of the cases where this may happen, (we will cover
# in later chapeters) could be trying to free memories taken by optimizers, or 
# our model.
# in such cases, we may need to need to explictly move all the tensors to cpu first,
# then delete the variable/instance followed by a gc.collect() to finally do a 
# empty_cache().
# this may happen when we want to e.g. delete our optimizer and free-up the memory
# it takes! however, it wouldnt work for some reason!
# the reason is the model's parameters, is also referenced by optimizers, so simply 
# deleting the model or setting it to None, wont do it. We need to also handle the
# optimizer we can delete them both, and this should free the memory. 
# if somehow we want to keep the model, and want to remove the optimizer, this is 
# what we would try first (move all the params to cpu, delete optimizer, gc.collect
# it and then try to empty_cache)
# again we'll see this later on. this was just a heads up!
# see : https://discuss.pytorch.org/t/how-can-we-release-gpu-memory-cache/14530/27

# sidenote: how empty_cache() works: 
# note when an object/variable is no longer referenced, its memory is set to be freed.
# this means its memory can be used to create new objects/tensors.
# the same way, deleting an object in python runtime, doesnt guarantee its given 
# back to the OS, the same thing applies in cuda runtime as well. 
# the memory is not released to the OS immediately and therefore when you query
# nvidia-smi it wont show any freed up memory!
# This is a typical behavior we often see when dealing with cuda/deeplearning training
# process this is caused by the pytorch allocator behavior, which keeps such these
# memory chuncks (as reserved) so it can do memory allocations much faster. 
# empty_cache() when called, forces the allocator to release these memories 
# that it's kept to allocate new tensors, back to the OS. 
# when this happens, the freed amount is reported in nvidia-smi.
# its noteworthy to mention that, these reserved memories, were already available to 
# allocator to create new tensors, so its not crucial to call empty_cache() to be 
# able to use such memories. (it makes a difference if we want to use them in a 
# separate process though)
# its just that it makes memory bookkeeping/logging on our side more clear!
# ref: https://discuss.pytorch.org/t/how-can-we-release-gpu-memory-cache/14530/4

# sidenote: concerning discrepency between nvidia-smi report vs pytorch's:
# PyTorch uses a caching memory allocator to speed up memory allocations. 
# This allows fast memory deallocation without device synchronizations. 
# However, the unused memory managed by the allocator will still show 
# as if used in nvidia-smi. 
# memory_allocated() and max_memory_allocated() can be used to monitor 
# memory occupied by tensors.
# memory_reserved() and max_memory_reserved() can be used to monitor the
# total amount of memory managed by the caching allocator. 
# torch.cuda.max_memory_allocated reported number can differ from the one 
# reported by nvidia-smi and may report a much smaller amount.   
# This discrepency is related to CUDA memory allocator. 
# The current CUDA memory allocator is a caching allocator, and it shows 
# more memory than is currently being occupied by tensors (the amount 
# reported by torch.cuda.max_memory_allocated()).
# a portion of this number in nvidia-smi belongs to "reserved" memory which
# is used to speed up future allocations/reclaim unused memory from garbage 
# collected tensors.
# The reserved memory amount (using torch.cuda.max_memory_reserved()) will be
# closer to  what is being reported by nvidia-smi.
# note that there will always be some additional overhead depending on 
# what operations/libraries are being used e.g. like cuDNN, cuBLAS, etc).
# this can become a substantial amount, as much a few hundreds of MB in many cases.
# so its prefectly normal to see higher values being reported in nvidia-smi
# ref: https://discuss.pytorch.org/t/pytorchs-torch-cuda-max-memory-allocated-showing-different-results-from-nvidia-smi/165706

# good to read docs: 
# https://pytorch.org/docs/stable/notes/cuda.html#cuda-memory-management
# https://pytorch.org/docs/stable/torch_cuda_memory.html#torch-cuda-memory 


# side sidenote:d
# note that we are using MiB instead of MB. but why?
# Originally, a kilobyte was defined as 1024 bytes (2^10). 
# However, in the SI metric system, "kilo" represents 1000 (10^3). 
# To avoid confusion, the International Electrotechnical Commission (IEC)
# introduced the term "KiB" (Kibibyte) to represent 1024 bytes. 
# So:
# 1 KB = 1000 bytes (SI convention)
# 1 KiB = 1024 bytes (binary convention)
# similarly, a megabyte was originally 1024 kilobytes (2^20), but the 
# SI metric system defines it as 1000 kilobytes. To maintain consistency:
# 1 MB = 1000 KB = 1,000,000 bytes (SI convention)
# 1 MiB(mebi byte) = 1024 KiB = 1,048,576 bytes (binary convention)
# Following the same pattern:
# 1 GB = 1000 MB = 1,000,000 KB = 1,000,000,000 bytes (SI convention)
# 1 GiB(gibi-byte) = 1024 MiB = 1,073,741,824 bytes (binary convention)
# IEC added these terms back in 1998! the hard drive manufacturers quickly used it!
# but nearly everyone else sticked to the good old definition!(including windows)
# until a few years ago when this slowly started to catch up and you probably see it
# here and there more often including in nvidia-smi reports.
# %% Section 11: Quick detour, utility functions (torch.set_printoptions)
# One small utility that is worth knowing early on is `torch.set_printoptions()`.
# As the name suggests, this function controls how tensors are displayed 
# when we print them. It does not change the actual values stored inside
# a tensor, only how those values appear on the screen.
# 
# This is identical to NumPy's `np.set_printoptions()`, infact it was taken
# from Numpy! so if you've used NumPy before, the idea should feel familiar.
#
# For example, imagine we have a very large tensor with thousands of elements.
# By default, PyTorch does not print every single value because that would 
# quickly flood the terminal with text. Instead, it prints only the beginning
# and end of the tensor, replacing the middle with "...".
# Likewise, floating-point numbers are displayed using a default precision 
# so the output remains readable. Sometimes, however, we may want more or 
# less detail. That's where `torch.set_printoptions()` comes in.
# One of the easiest ways to configure the printing behavior is
# by using one of PyTorch's predefined profiles.

# There are 3 default profiles we can use ['default', 'short', 'full']. 
# the 'default' profile argument tells PyTorch to use a predefined collection 
# of print settings. it restores PyTorch's normal printing behavior. This is
# especially useful if you previously changed the print options and want to 
# go back to the standard settings.

# 'full' profile as the name implies, displays the whole tensor without any
# summerization. 'short' on the other hand, is the opposite, it makes tensor
# printing more compact by showing fewer elements and using a smaller display format.
#
# We can also manually change different aspects related to tensor printing
# in the output. `torch.set_printoptions()` lets us customize 
# aspects such as:
# precision: 
#  Number of digits of precision for floating point output (default = 4).
# 
# threshold:
#  Total number of array elements which trigger summarization rather than full repr (default = 1000)
#  basically when to summarize large tensors using "...".
# 
# edgeitems:
#  Number of array items in summary at beginning and end of each dimension (default = 3)
# 
# linewidth:
#  The number of characters per line for the purpose of inserting line breaks
#  (default = 80). (i.e. maximum number of characters per printed line).
#  Thresholded matrices will ignore this parameter.
#  
# sci_mode:
#  For whether scientific notation should be used. That is instead of 0.001, it prints
#  in scientific notation form 1e-3. If None (default) is specified,
#  the value is defined by torch._tensor_str._Formatter. This value is automatically
#  chosen by the framework.

tensor = torch.randn(4,10)

torch.set_printoptions(precision=2, threshold=10, linewidth=80)
show_tensor("tensor with custom printoptions", tensor)

torch.set_printoptions(profile='default')
show_tensor("tensor with default values", tensor)

# sidenote:
# Numpy, beside set_printoption, also offers `np.printoptions` as
# a context manager. Unlike Numpy, Pytorch sadly doesnt offer one.
# If we have a context manager, instead of modifying the global 
# settings, we can apply custom print options only within a specific
# block of code so once we leave the block, the previous settings 
# are automatically restored.
# we can create one for ourselves like below:
from contextlib import contextmanager

@contextmanager
def printoptions(**kwargs):
    # save the options we care about
    defaults = {
        "precision": 4,
        "threshold": 1000,
        "edgeitems": 3,
        "linewidth": 80,
        "sci_mode": None,
        }

    torch.set_printoptions(**kwargs)
    try:
        yield
    finally:
        torch.set_printoptions(**defaults)

with printoptions(precision=1):
    show_tensor("tensor inside context manager", tensor)
   
# Outside the 'with' block, the original print settings
# are automatically restored.
show_tensor("tensor outside context manager", tensor)

# This is generally preferred when we only need custom formatting
# for debugging or inspecting a few tensors, since it avoids
# accidentally changing the print behavior for the rest of our
# program.
#
# We'll mostly keep the default settings throughout this course,
# but occasionally we'll change them to make tensors easier to
# inspect and understand as we did earlier in this chapter. 
# Whenever possible, we'll prefer the context manager since it
# keeps the changes local and makes our code easier to reason about.