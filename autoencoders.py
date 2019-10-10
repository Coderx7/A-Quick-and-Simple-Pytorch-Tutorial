#%%
# in the name of God the most compassionate the most merciful
# in this part, we are going to learn about autoencoders and 
# how we can implement them in Pytorch. 
# Autoencoders are a kind of networks that map their input
# to a new representation. this is usually refered to as 
# compressing the input into a latent space representation,
# this means, they accept the data, and then downsample it
# until they reach a specified/suitable size feature vector
# and then upsample that feature vector gradually until they
# reach to the original size, and then try to reconstruct the
# input(so our input image acts as a label as well!). 
# during this process of reconstructing the input data
# from the compressed representation, the new representation is
# developed and can be used for various applications. 
# the first part of the network that downsamples the input data
# into a feature vector is called an "Encoder", and the part that
# reconstructs the input from the mentioned feature vector is called
# a "Decoder". 
# when we have successfully trained our autoencoder, we can use its
# new representation instead of our new data. so we can use it for 
# dimensionality reduction just like PCA (if linear) and much more 
# powerful than that when using a deep nonlinear autoencoder! 
# we can use the new representation for lots of applications including
# sending /storing the reduced representation instead of the full input
# and reconstruct the input using the representation, this will result 
# in a considerable reduction in network traffic or space required to store
# the actual data. 
# The usage is not limited to such usescases, we can get fancy and creative 
# for example and make a black and white image , color again! or denoise our input
# reconstruct missing parts, create new data/images, visualizations, etc!
# there are lots and lots of use cases for autoencoders
# However, note that, the notion of compression spoke here is different than that of
# what you find in different media formats such as jpeg, mp3, etc. 
# Autoencoders do not work well on unseen data and thus usually have difficulties 
# generalizing well to unseen data. more on this later  
# There are different kinds of Autoencoders, they can be linear, or
# nonlinear, shallow, or deep, convolutional, or not, etc
# we will cover some of the most famous variants here. 
# lets start

# before we start lets get familiar with couple of concepts 

# note :
# https://www.statisticshowto.datasciencecentral.com/posterior-distribution-probability/
# Posterior probability is the probability an event will happen after all evidence or 
# background information has been taken into account. It is closely related to prior probability,
# which is the probability an event will happen before you taken any new evidence into account.
# You can think of posterior probability as an adjustment on prior probability:
#         Posterior probability = prior probability + new evidence (called likelihood).

# For example, historical data suggests that around 60% of students who start college will 
# graduate within 6 years. This is the prior probability. However, you think that figure is 
# actually much lower, so set out to collect new data. The evidence you collect suggests that
# the true figure is actually closer to 50%; This is the posterior probability.

# What is a Posterior Distribution?
# The posterior distribution is a way to summarize what we know about uncertain quantities in 
# Bayesian analysis. It is a combination of the prior distribution and the likelihood function,
# which tells you what information is contained in your observed data (the “new evidence”). 
# In other words, the posterior distribution summarizes what you know after the data has been 
# observed. The summary of the evidence from the new observations is the likelihood function.
# Posterior Distribution = Prior Distribution + Likelihood Function (“new evidence”)
# Posterior distributions are vitally important in Bayesian Analysis. They are in many ways 
# the goal of the analysis and can give you:
#     Interval estimates for parameters,
#     Point estimates for parameters,
#     Prediction inference for future data,
#     Probabilistic evaluations for your hypothesis.
# ------------------------------------------------------------------------------ 

# https://www.statisticshowto.datasciencecentral.com/likelihood-function/

# What is a prior probablity : 
# https://www.statisticshowto.datasciencecentral.com/prior-probability-uniformative-conjugate/

# Prior Probability: Uniformative, Conjugate
# Probability > Prior Probability: Uniformative, Conjugate

# What is Prior Probability?
# Prior probability is a probability distribution that expresses established beliefs about an 
# event before (i.e. prior to) new evidence is taken into account. When the new evidence is used
# to create a new distribution, that new distribution is called posterior probability.
# prior probability 
# For example, you’re on a quiz show with three doors. A car is behind one door, 
# while the other two doors have goats. You have a 1/3 chance of winning the car. This is the 
# prior probability. Your host opens door C to reveal a goat. Since doors A and B are the only 
# candidates for the car, the probability has increased to 1/2. The prior probability of 1/3 has 
# now been adjusted to 1/2, which is a posterior probability.
# In order to carry our Bayesian inference, you must have a prior probability distribution. 
# How you choose a prior is dependent on what type of information you’re working with. 
# For example, if you want to predict the temperature tomorrow, a good prior distribution 
# might be a normal distribution with this month’s mean temperature and variance.

# Uninformative Priors
# An uninformative prior gives you vague information about probabilities. It’s usually used when 
# you don’t have a suitable prior distribution available. However, you could choose to use an 
# uninformative prior if you don’t want it to affect your results too much.
# The uninformative prior isn’t really “uninformative,” because any probability distribution 
# will have some information. However, it will have little impact on the posterior distribution 
# because it makes minimal assumptions about the model. For the temperature example, 
# you could use a uniform distribution for your prior, with the minimum values at the record low 
# for tomorrow and the record high for the maximum.

# Conjugate Prior
# A conjugate prior has the same distribution as your posterior prior. For example, if you’re 
# studying people’s weights, which are normally distributed, you can use a normal distribution
#  of weights as your conjugate prior.

# lets start!

import datetime
import numpy as np 
import torch
import torchvision
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim
import matplotlib.pyplot as plt 
%matplotlib inline

# We will use MNIST dataset for our experiments here. 
# Lets get back to our discussion! 
# We mentioned couple of examples/usecases for autoencoders, but why do we have to
# shrink the size in the encoder part ? why do we gradually reduce the input size until
# we reach a feature vector of some size? 
# shrinking the size gradually, acts as a imposing a constraint on the input
# by doing so, we are forcing the network to choose the important features in 
# our input data, the features that has the essence of our input data and can later
# be used to reconstruct the input. This is why the new resuling representation works 
# very well and can be used instead of the input for some applications. The new repres
# -entation simply has the most important features in the input. 
# if such constraint was not present, the network would not be able to learn anything
# meaningful about the distribution of our input data and thus the resulting vector
# would be of no use to us. So we should be shrinking the input until we reach 
# a certain size of our liking(based on our usage) 
# Note that, these new features may not be individually interpretable specially in the
# case of nonlinear deep autoencoders. there are ways to see what and how a specific 
# feature in the resulting feature vector responds to different attributs present in
# an input but never make this mistake that e.g. the 10 features in the bottleneck layer
# (our feature vector) represents the exact attributes in your input. these features
# may represent a complex interactions between several features that define a 
# characteristic in you data. anyway we'll get to this later on. 

# Ok, enough talking lets get busy and have our first auto encoder. 
# before we continue, we should pickup a dataset. I chose MNIST as its simple enough
# to be used in different types of autoencoders with quick training time. 
# after we created our dataset, we will implement different types of AutoEncoders 
dataset_train = datasets.MNIST(root='MNIST',
                               train=True,
                               transform = transforms.ToTensor(),
                               download=True)
dataset_test  = datasets.MNIST(root='MNIST', 
                               train=False, 
                               transform = transforms.ToTensor(),
                               download=True)
batch_size = 128
num_workers = 0
dataloader_train = torch.utils.data.DataLoader(dataset_train,
                                               batch_size = batch_size,
                                               shuffle=True,
                                               num_workers = num_workers, 
                                               pin_memory=True)

dataloader_test = torch.utils.data.DataLoader(dataset_test,
                                               batch_size = batch_size,
                                               num_workers = num_workers,
                                               pin_memory=True)

# lets view a sample of our images 
def view_images(imgs, labels, rows = 4, cols =11):
    # images in pytorch have the shape (channel, h,w) and since we have a
    # batch here, it becomes, (batch, channel, h, w). matplotlib expects
    # images to have the shape h,w,c . so we transpose the axes here for this!
    imgs = imgs.detach().cpu().numpy().transpose(0,2,3,1)
    fig = plt.figure(figsize=(8,4))
    for i in range(imgs.shape[0]):
        ax = fig.add_subplot(rows, cols, i+1, xticks=[], yticks=[])
        # since mnist images are 1 channeled(i.e grayscale), matplotlib
        # only accepts these kinds of images without any channesl i.e 
        # instead of the shape 28x28x1, it wants 28x28
        ax.imshow(imgs[i].squeeze(), cmap='Greys_r')
        ax.set_title(labels[i].item())
    plt.tight_layout(pad=1,rect= (0, 0, 40, 40))

# now lets view some 
imgs, labels = next(iter(dataloader_train))
view_images(imgs, labels,13,10)

# good! we are ready for the actual implementation
#%% 
# The first autoencoder weare going to implement is the simplest one, 
# a linear autoencoder.
# creating an autoencoder is just like any other module we have seen so far, simply
# inherit from nnModule and define the needed layers and call them in the forward()
# method the way you should. lets do this :
class LinearAutoEncoder(nn.Module):
    def __init__(self, embedingsisze=32):
        super().__init__()
        # lets define our autoencoder we have two parts, an encoder 
        # and a decoder. 
        # the encoder shrinks the input gradually until it becomes
        # a certain size, and the decoder accepts that as input and
        # gradually upsamples it to reach the actual input size. 
        
        # The encoder part: 
        # So our encoder part simply is a linear 
        # layer, or a fully connected layer that
        # accepts the input. since this is a linear layer,
        # we have to flatten the input and our 28x28 image
        # will simply have 28x28=784 input features 
        # The simplest form can be an a one layered encoder
        # and a 1 layered decoder! of course we can add more
        # layers between them, but lets see how this performs
        self.fc1 = nn.Linear(28*28, embedingsisze)
        # our decoder part
        self.fc2 = nn.Linear(embedingsisze, 28*28)

    def forward(self, inputs):
        # our foward pass is nothing specially
        # simply feed these layers in order!
        # but before that, we must flatten our input!
        inputs = inputs.view(inputs.size(0), -1)
        # encoder part
        output = self.fc1(inputs)
        # decore part
        output = self.fc2(output)
        # since in the output we want an image not a flattened
        # evctor, we reshape our input again!
        output = output.view(-1, 1, 28, 28)
        return output 


model_linear_ae = LinearAutoEncoder()
print(model_linear_ae)
#%%
# now lets train our model. 
# since we compare the output of our network with our input
# we use MSELoss for this. 
criterion = nn.MSELoss()
def train(model, dataloader, optimizer, scheduler, epochs, device):
    for e in range(epochs):
        # we dont need label so we use _ as its a convention
        for i, (imgs,_) in enumerate(dataloader):
            imgs = imgs.to(device)
            reconstructed_images = model(imgs)
            loss = criterion(reconstructed_images, imgs)        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i% 2000==0:        
                print(f'epoch: ({e}/{epochs}) loss: {loss.item():.6f} lr:{scheduler.get_lr()}')
        scheduler.step()
    print('done')

# Now lets see the output of our autoencoder
def test(model,device):
    imgs, labels = next(iter(dataloader_test))
    imgs = imgs.to(device)
    outputs = model(imgs)
    view_images(outputs, labels)
#%%

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_linear_ae = model_linear_ae.to(device)
optimizer = optim.Adam(model_linear_ae.parameters(), lr = 0.1) 
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5)

train(model_linear_ae, dataloader_train, optimizer, scheduler, 20, device) 
test(model_linear_ae, device)    
# so this is the linear autoencoder! in order to make a vanila autoencoder
# which may refer to a version with nonlinear activation functions, you 
# only need to apply a transformation function .
# in the fowarad pass and in order  to get a good result, you need to add a few more 
# layers .(we do this in the next architecture )
# we can get better results with more epochs and decaying learnng rate,
#  but it wont make a drastic change! specially on more complex data, as its 
# just a linear model.
#%%
# in order to be able to capture more complex structures,... in  the input data
# one way is to add more hidden layers! so 
# Now lets create a multi layer auto encoder!
class MLPAutoEncoder(nn.Module):
    def __init__(self, embedingsisze=32):
        super().__init__()

        self.fc1 = nn.Linear(28*28, 64)
        self.fc2 = nn.Linear(64, embedingsisze)
        # our decoder part
        self.fc3 = nn.Linear(embedingsisze, 64)
        self.fc4 = nn.Linear(64, 28*28)


    def forward(self, inputs):
        inputs = inputs.view(inputs.size(0), -1)
        # encoder part
        output = F.relu(self.fc1(inputs))
        output = F.relu(self.fc2(output))
        # decore part
        output = F.relu(self.fc3(output))
        # since the output is images, values should 
        # be in the range [0, 1]!
        output = F.sigmoid(self.fc4(output))
        output = output.view(-1, 1, 28, 28)
        return output 

model_mlp_ae = MLPAutoEncoder().to(device)
print(model_mlp_ae)

# criterion = nn.MSELoss()
optimizer = optim.Adam(model_mlp_ae.parameters(), lr = 0.01) 
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5)
train(model_mlp_ae, dataloader_train, optimizer, scheduler, 20, device)    
test(model_mlp_ae,device)  
#%%
# While our mlp model is more powerful than the previous model, it is not suitable for data such as images
# for image like data, we use conv layers! and hence our new autoencoder is Convolutional AutoEncoder. 
# lets see how to implement this : 
# something that needs to be said is that, when the number of layers is increased, i.e. 
# your network gets deeper, you may see that your model may train sometimes and not the 
# other times and the loss may not decrease. when you see this, you should know this is
# happening becasue of the depth of your network.  use the batchnorm and all will be good. 
# thats why I created two functions for this very purpose. try creating your network with
# and without batchnormalization enabled and see the difference (try running for several 
# times with the one with no batchnormalization to see that sometimes it may work and some 
# times it will fail, but with batchnorm, it will always work!)
def conv_bn(in_,out_,k_size=3, s=2,pad=0,bias=False,batchnorm=True):
    layers = []
    layers.append(nn.Conv2d(in_,out_,kernel_size=k_size,stride=s,padding=pad,bias=bias))
    if batchnorm:
        layers.append(nn.BatchNorm2d(out_))
    return nn.Sequential(*layers)

def deconv_bn(in_,out_,k_size=4, s=2,pad=0,bias=False,batchnorm=True):
    layers = []
    layers.append(nn.ConvTranspose2d(in_,out_,kernel_size=k_size,stride=s,padding=pad,bias=bias))
    if batchnorm:
        layers.append(nn.BatchNorm2d(out_))
    return nn.Sequential(*layers)

class ConvAutoEncoder(nn.Module):
    def __init__(self, embedingsize=32):
        super().__init__()
        # for conv layers, since we are dealing with 3d featuremaps
        # we shrink the number of featuremaps as well as the spatial
        # dimensions. we do so until we reach a size that satifies us
        # 
        self.conv1 = conv_bn(1, 256, 3, 1) # each stride 2 downsamples the dimensions by half
        self.conv2 = conv_bn(256, 128, 3, 2) # 14 
        self.conv3 = conv_bn(128, 64, 3, 2)  # 7 
        self.conv4 = conv_bn(64, embedingsize, 3, 2) # output is 64x2x2
        # decoder 
        # now for decoder we have two options, we can simply use a conv layer 
        # followed by a upsample layer. or we can use a deconv layeror  a 
        # transposed convolution layer. the difference between them is that
        # using the transposedconv approach, results in a checkerboard effect
        # while thats not the case for upsample method!
        # we use k=2,s=2, as it upsamples the image 2x. 
        # from there we can use different kernel size, strides 
        # to reach to desired dimensions. 
        self.deconv5 = deconv_bn(embedingsize, 64, 2, 2) 
        self.deconv6 = deconv_bn(64, 128, 4, 2) 
        self.deconv7 = deconv_bn(128, 256, 5, 2)
        # and since our image is 1 channel, this last layer will produce a singe image!
        self.conv8 = deconv_bn(256, 1, 6, 1,0,True,False)
         

    def forward(self, x):
        output = F.relu(self.conv1(x))
        output = F.relu(self.conv2(output))
        output = F.relu(self.conv3(output))
        output = F.relu(self.conv4(output))
        output = F.relu(self.deconv5(output))
        output = F.relu(self.deconv6(output))
        output = F.relu(self.deconv7(output))
        # since we want an image, we use sigmoid
        output = F.sigmoid(self.conv8(output))
        return output


#%%
# now lets train it
model_c= ConvAutoEncoder()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
optimizer = optim.Adam(model_c.parameters(), lr =0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5)
model_c = model_c.to(device)
train(model_c, dataloader_train, optimizer, scheduler, 20, device)    
test(model_c, device)  
# As an excersize try to replace all ConvTranspose2d Layers with Conv2d+Upsample
# and see how the outputs turn out !
#%% 
# Now lets create more powerful Convolutional AutoEncoders. the vanial convolutional autoencoder
# is not that powerful. therefore we can use several variants such as:
# denoising autoencoder, Sparse autoencoder, variational autoencoder

# using denoising autoencoder, our archietcture needs to be deep enough because its
# a more complex taks. however, our previous convautoencoder is deep enough so we can 
# use that here as well. 
# basically in denoising autoencoder, we feed a noisy image and get a noise free image
# so what we will actually do in the training process is to add random noise to our image
# prior to feeding it to our model and then compare the reconstructed image with the actual
# original image which is noise free. in doing this, network will learn to remove noise from
# images. we will use the same criterion. nearly 99% of what we saw until now is the same 
# and we just will add a simplenoise lets see that 
noise_threshold = 0.5
epochs = 20
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# the quality and performance of our model in denoising will increase as we
# increase the embedding size. 
model = ConvAutoEncoder(1).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001) 
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5)

# before we go on lets view a sample of noisy images : 
imgs,labels = next(iter(dataloader_test))
imgs = imgs + noise_threshold * torch.rand_like(imgs)
imgs.clamp_(0,1)
view_images(imgs,labels)

print(model)
for e in range(epochs):
    loss_epoch = 0.0
    for imgs,_ in dataloader_train:
        imgs = imgs.to(device)

        #apply noise to our image 
        imgs_noisy = imgs + noise_threshold * torch.rand_like(imgs)
        # clip all values outside of 0,1 becasue our image values 
        # should be in this range!
        imgs_noisy = imgs_noisy.clamp(0,1)
        imgs_recons = model(imgs_noisy)

        loss = criterion(imgs_recons, imgs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_epoch += loss.item()
    print(f'epoch: {e}/{epochs} loss: {loss.item()} lr: {scheduler.get_lr()}')
    scheduler.step()

# lets see how the network does on noisy image!
imgs,labels = next(iter(dataloader_test))
imgs = imgs.to(device)
imgs = imgs + noise_threshold * torch.rand_like(imgs)

imgs.clamp_(0,1)
view_images(imgs,labels)
new_noise_free_imgs = model(imgs)
view_images(new_noise_free_imgs,labels)
#%%
# you may ask, so far we have been starting with a large number of channels, 
# and gradually decreased and at the same time shrunk the spatial extend, what if we do
# the opposite, we begin with few channels and large spatial extend
# and then gradually increase the channels and shrink the spatial dimension until
# you reach a large vector representation with little or no spatial extent. and in the 
# decoder, we do the opposite obviously! lets see how that performs! (tldr !it performs worse!)
class ConvolutionalAutoEncoder_v2(nn.Module):
    def __init__(self, embeddingsize=32):
        super().__init__()
        self.encoder = nn.Sequential(conv_bn(1, 32, 3, 1),
                                conv_bn(32, 64, 3, 2),
                                conv_bn(64, 128, 3, 2),
                                conv_bn(128, embeddingsize, 3, 2))

        self.decoder = nn.Sequential(deconv_bn(embeddingsize, 128, 2, 2),
                                deconv_bn(128, 64, 4, 2),
                                deconv_bn(64, 32, 5, 2),
                                deconv_bn(32, 1, 6, 1,batchnorm=False))
    def forward(self, inputs):
        output = self.encoder(inputs)
        return self.decoder(output)
                               

noise_threshold = 0.5
epochs = 20
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# the quality and performance of our model in denoising will increase as we
# increase the embedding size. 
model = ConvolutionalAutoEncoder_v2(32).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001) 
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5)

# before we go on lets view a sample of noisy images : 
imgs,labels = next(iter(dataloader_test))
imgs = imgs + noise_threshold * torch.rand_like(imgs)
imgs.clamp_(0,1)
view_images(imgs,labels)

print(model)
for e in range(epochs):
    loss_epoch = 0.0
    for imgs,_ in dataloader_train:
        imgs = imgs.to(device)

        #apply noise to our image 
        imgs_noisy = imgs + noise_threshold * torch.rand_like(imgs)
        # clip all values outside of 0,1 becasue our image values 
        # should be in this range!
        imgs_noisy = imgs_noisy.clamp(0,1)
        imgs_recons = model(imgs_noisy)

        loss = criterion(imgs_recons, imgs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_epoch += loss.item()
    print(f'epoch: {e}/{epochs} loss: {loss.item()} lr: {scheduler.get_lr()}')
    scheduler.step()

# lets see how the network does on noisy image!
imgs,labels = next(iter(dataloader_test))
imgs = imgs.to(device)
imgs = imgs + noise_threshold * torch.rand_like(imgs)

imgs.clamp_(0,1)
view_images(imgs,labels)
new_noise_free_imgs = model(imgs)
view_images(new_noise_free_imgs,labels)
#%%
# sparse autoencoder: these kinds of autoencoders simply use a regularizer term so that
# the features are more sparse! usually l1 loss is used! 
#  In the previous examples, the representations were only constrained by the size of the
# hidden layers. In such a situation, what typically happens is that the hidden layer is
# learning an approximation of PCA (principal component analysis).
# But another way to constrain the representations to be compact is to add a sparsity 
# contraint on the activity of the hidden representations, so fewer units would "fire" 
# at a given time.
# in order to have sparsity, we need to have overcomplete representations. so lets 
# implement a sparse autoencoder in this section and see how it performs. 
# as I said earlier, aside from the normal reconstruction loss, we need a new regularizer
# lets create this regularizer now. We are going to create a Function object that applies
# l1penalty we inherit from autograd.Function class for this. 
# good exlanation https://www.youtube.com/watch?v=7mRfwaGGAPg

import copy # sed for deep copy of our weights
from torch.autograd import Function  # used for implementing l1_lenalty 
class L1Penalty(Function):
    # we override the forward method with our own arguments (input, l1_weight)
    # input is the input obviously and l1_weight is the percentage of zero weights
    # that is 0.1 means, we want 10% of our weights to be zero (or near zero)
    # or sparsity ratio if you will!
    # In the forward pass, we simply save our input and l1_weight for use in backwardpass
    @staticmethod
    def forward(ctx, input, l1_weight):
        ctx.save_for_backward(input)
        ctx.l1_weight = l1_weight    
        return input

    #   backward must accept a context `ctx` as the first argument, followed by
    #   as many outputs did `forward` return, and it should return as many
    #   tensors, as there were inputs to `forward`. 
    #   Each argument is the gradient w.r.t the given output, and each returned
    #   value should be the gradient w.r.t. the corresponding input.

    #   The context can be used to retrieve tensors saved during the forward
    #   pass. It also has an attribute ctx.needs_input_grad` as a tuple
    #   of booleans representing whether each input needs gradient. E.g.,
    #   `backward` will have ``ctx.needs_input_grad[0] = True`` if the
    #   first input to `forward` needs gradient computated w.r.t. the
    #   output.
    @staticmethod
    def backward(ctx, grad_outputs):
        input, = ctx.saved_tensors
        # since we only need gradients with respect to the input
        # we need to explicitly say we dont need gradienst to be 
        # calculated for our second argument term in forward method 
        # i.e. l1_weight. so we return None for other arguemnst that
        # we dont want any gradient. 
        # this is a term that we apply in the backward pass, 
        # that is, we are enforcing the constraint by adding 
        # a new term to the gradient 
        grad_input = input.clone().sign().mul(ctx.l1_weight)
        grad_input +=grad_outputs
        # since we have two inputs in our foward pass, we need to
        # provide two gradients in the backward pass. but becasue
        # we only care about input and not the l1_weight, (we dont)
        # need any gradients for it becsaue we are not tuning that!
        # we return None
        return grad_input, None

# now lets create our architecture 
class SparseAutoEncoder(nn.Module):
    def __init__(self, embeddingsize=400, tied_weights = False):
        super().__init__()
        self. tied_weights = tied_weights

        self.encoder = nn.Sequential(nn.Linear(28*28, embeddingsize),
                                    nn.Sigmoid())# or relu
        self.decoder = nn.Sequential(nn.Linear(embeddingsize, 28*28),
                                    nn.Sigmoid())
        # you may see some people, use the shared weights between encoder
        # and decoder, i.e. decoder uses the transposed weightmatrix of the 
        # encoder. for doing this  there are couple of ways. 
        # one of way is to use the functional form and simply 
        # use one weight and its transpose like this 
        # weight = nn.Parameter(torch.rand(input_dim, output_dim))
        # self.encoder = F.linear(input, weight, bias=False)
        # self.decoder = F.linear(input, weight.t(), bias=False)
        # we can also simply define our new weight and assigne it to both modules
        if self.tied_weights:
            weights = nn.Parameter(torch.randn_like(self.encoder[0].weight))
            self.encoder[0].weight.data = weights.clone()
            self.decoder[0].weight.data = self.encoder[0].weight.data.t()
        

    def forward(self, input):
        input = input.view(input.size(0), -1)
        output_enc = self.encoder(input)
        rec_imgs = self.decoder(output_enc)
        rec_imgs = rec_imgs.view(input.size(0), 1, 28, 28)
        return output_enc, rec_imgs


def sparse_loss_function(outputs_enc, reconstructed_imgs, imgs, penalty_type=0, l1_weight=0.01, Beta=1):
    """
    penalty_type : 
    0: sparsity on activations 
    1: sparsity using l1 penalty using gradient enforcemet
    2: sparsity using kl divergence
    """
    criterion = nn.MSELoss()
    loss = criterion(reconstructed_imgs, imgs)

    if penalty_type == 0:
        sparsity_loss = torch.mean(abs(outputs_enc))
        return loss + sparsity_loss
    elif penalty_type == 1:
        # apply the l1penalty on the weights of our encoder
        # through added term in backpropagation
        output = L1Penalty.apply(outputs_enc, l1_weight)
        return loss
    else:
        # use kl divergence, calculate ro^ which is the
        # mean of activations in our hidden layer in which
        # we want sparsity
        # the idea here is that each neurons activation should be sparse
        # that means, its values need to be zero or close to zero. now 
        # how do we do that? we set a threshold, we call it ro and set it
        # to a value e.g. 0.05 and then check the mean of each neurons 
        # activations, and call it ro_hat, we compare our ro_hat against
        # our threshold which is ro! then we penalize all neurons that 
        # their ro_hat is larger than the threshold. but how do we compare 
        # them? we use kl divergence. why? we can model two distributions (bernolli)
        # being p and q with the probability of success ro and ro_hat respectively
        # the idea is, to ensure the predicted distribution is as close to the 
        # actual one and we can model this with kl divergence
        #  
        ro_hat = torch.mean(outputs_enc).to(imgs.device)
        ro = torch.ones_like(ro_hat).to(imgs.device) * l1_weight
        # ro and ro_hat must be probablities, what we have now is just logits
        # so we use softmax to turn our logits into probabilties
        # remember our activation function must be sigmoid 
        # print(ro.shape, ro_hat.shape)
        
        kl = torch.sum(ro * torch.log(ro / ro_hat) +
                      (1 - ro) * torch.log((1 - ro) / (1 - ro_hat)))
        return loss + (Beta * kl)


epochs = 50
penalty_type = 0
# ro 0.01 ~ 0.05 or l1_weight 
sparsity_ratio = 0.1
loss_type = 2
# at the end read the Cyclical Annealing Schedule section to get a very good idea about
# how you can achieve better result and why!
Beta = 3 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sae_model = SparseAutoEncoder(embeddingsize=400,                             
                              tied_weights=True).to(device)
optimizer = torch.optim.Adam(sae_model.parameters(), lr = 0.1) 
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10) 

print(sae_model)        
# lets save the weights of our encoder and decoders before we train them 
# and then compare them with the new weights after training and see how
# they changed!
init_weights_encoder = copy.deepcopy(sae_model.encoder[0].weight.data) 
init_weights_decoder = copy.deepcopy(sae_model.decoder[0].weight.data)
imgs_list =[]
# now lets start training ! 
for e in range(epochs):
    for imgs,_ in dataloader_train:
        imgs = imgs.to(device)
        output_enc, rec_imgs = sae_model(imgs)
        loss = sparse_loss_function(output_enc, rec_imgs, imgs, loss_type, sparsity_ratio, Beta)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'epoch: {e}/{epochs} loss: {loss.item():.6f} lr = {scheduler.get_lr()[-1]:.6f}')
    scheduler.step()
    # at each epoch, we sample one image and its reconstruction
    # for viewing later on to see how the training affects the
    # result we get
    imgs_list.append((imgs[0],rec_imgs[0]))
#%% 
# now lets first visualize the image/reconstruction pairs and how they look : 
def visualize(imgs_list, rows=5, cols=10):
    fig = plt.figure(figsize=(15,2))
    plt.subplots_adjust(wspace=0,hspace=0)
    
    print(f'number of samples: {len(imgs_list)}')
    for i in range(len(imgs_list)):
        img,recons = imgs_list[i]
        # print(img.shape,recons.shape)
        img = img.cpu()
        recons = recons.cpu().detach()
        ax = fig.add_subplot(rows, cols, i+1, xticks=[], yticks=[])
        x = torchvision.utils.make_grid([img,recons])
        ax.imshow(x.numpy().transpose(1,2,0))

visualize(imgs_list)

# Now lets visualize the weights and see how they look. 
# we had the initial weights saved so lets subtract them
# from the trained one and see the diffs , it will show us
# where the changes happened 

def visualize_grid(imgs, rows=20, cols=20):
    fig = plt.figure(figsize=(20, 20))
    imgs = imgs.cpu().numpy().transpose(0, 2, 3, 1).squeeze()
    plt.subplots_adjust(wspace=0, hspace=0)
    for i in range(imgs.shape[0]):
        ax = fig.add_subplot(rows, cols, i+1, xticks=[], yticks=[])
        ax.imshow(imgs[i], cmap='Greys_r')


def visualize_grid2(imgs, label, normalize=True):
    fig = plt.figure(figsize=(10, 10))
    imgs = imgs.cpu()  
    plt.subplots_adjust(wspace=0, hspace=0)
    x = torchvision.utils.make_grid(
        imgs, nrow=20, normalize=normalize).numpy().transpose(1, 2, 0)
    ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[])
    ax.imshow(x)
    ax.set_title(label)

trained_W_encoder = sae_model.encoder[0].weight.data.cpu(
).clone().view(sae_model.encoder[0].out_features, 1, 28, 28)
trained_W_decoder = sae_model.decoder[0].weight.data.cpu(
).clone().view(sae_model.decoder[0].in_features, 1, 28, 28)
init_weights_encoder = init_weights_encoder.view(
    sae_model.encoder[0].out_features, 1, 28, 28).cpu()
init_weights_decoder = init_weights_decoder.view(
    sae_model.decoder[0].in_features, 1, 28, 28).cpu()

w_diff_encoder = init_weights_encoder - trained_W_encoder
w_diff_decoder = init_weights_decoder - trained_W_decoder

w_decoders_transposed = sae_model.decoder[0].weight.data.cpu().clone().t()

# in order to see that decoders weight is infact the same as
# encoders, lets transpose it again and reshape it.
# here I show both the encoders, weight and our decoders weight
# transposed! 
print(trained_W_encoder.shape)
print(w_decoders_transposed.shape)
w_decoders_transposed = w_decoders_transposed.view(sae_model.encoder[0].out_features, 1, 28, 28)
# note that the decoder weights (in terms of original data) will be smoothed encoders weights
# (also in terms of original data)
# info from : https://medium.com/@SeoJaeDuk/arhcieved-post-personal-notes-about-contractive-auto-encoders-part-1-ef83bce72932 
# end of the page, in the ppt slide image

print(init_weights_encoder.shape)
visualize_grid2(init_weights_encoder, 'Initial weights')
visualize_grid2(trained_W_encoder, 'Trained weights(Encoder)')
visualize_grid2(w_diff_encoder, 'weights diff (Encoder)')
visualize_grid2(trained_W_decoder,'Trained Weights (Decoder)')
visualize_grid2(w_decoders_transposed,'Trained Weights (Decoder-transposed)')
# the black shows negative values, and white show positive values
# and the gray shows zero values.
# we start from a high positive and high negative values in our initial
# weights. and then after training and imposing sparsity we can see that
# we are mostly seeing gray colors which indicate the values are zero!
# and that is what we were after!
# if you look at the w_diff, you can see that there are lots of high and
# low (negative) values as well. this is becsaue  in order to make the
# weights have more reasonable weights, they had to be decreased/increased
#%% 
# the cool thing about autoencoders are that we can use them to pretrain
# our weights on our data and then use that for classification or etc. 
# this was actually done a lot back in the day until 2014/2015. 
# in that era, the use of xavier initialization algorithm accompanied by 
# batchnormalization killed the need for pretraining in this way. but lets 
# see how we can do this if the needs be. 
# its simple, just like finetuning, we may add/remove the layers we want
# here we will remove the decoder part and instead add a classifier
# lets remove the decoder 
layers_before_decoder = list(sae_model.children())[:-1]
sae_model2 = nn.Sequential(*layers_before_decoder)
# since we created a sequential model here, we should add a new module
# using add_module. because if we simplt do sth like : 
# sae_model2.classifier = nn.Linear(sae_model2[0].out_features, 10)
# classifier will be just an attribute, and for the forward pass we 
# need to do sth like 
# output=sae_model2.forward(input)
# output = sae_model2.classifier(output)
# so this is not ideal at all. therefore we do : 
sae_model2.add_module('classifier', nn.Linear(sae_model2[0][0].out_features, 10))
print(sae_model2)
#%% now that we have our model built lets run trainng and pay attention
# what is the first accuracy we get

criterion = nn.CrossEntropyLoss()
epochs = 20 
optimizer = torch.optim.SGD(sae_model2.parameters(), lr=0.1)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5)
acc = 0.0
sae_model2 = sae_model2.to(device)
for e in range(epochs):
    for i, (imgs, labels) in enumerate(dataloader_train):
        imgs = imgs.to(device)
        labels = labels.to(device)

        imgs = imgs.view(imgs.size(0),-1)
        output = sae_model2(imgs)
        loss = criterion(output, labels)
        _,class_idx = torch.max(output,dim=1)
        acc += torch.mean((class_idx.view(*labels.shape) == labels).float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    acc = acc/len(dataloader_train)
    print(f'epoch: ({e}/{epochs}) acc: {acc*100:.4f} loss: {loss.item():.6f} lr: {scheduler.get_lr():.6f}')
    scheduler.step()


# now you can try it without running the autoencoder training and 
# see how it performs.
# Important note : 
# There is a difference between sparsity on parameter and sparsity on representation.
# Sparse Autoencoder proposed by Andrew NG is able to learn a sparse representation 
# and it is well known that l1 regularization encourages sparsity on parameters.
# They are different lets explain this in more detail!

# Notes: 
# For imposing the sparsity constraint instead of l1 norm, we can
# also use KL divergance the principle is the same, where we took 
# the average of the activations at each layer that we want their 
# weights to be sparse, this time we calculate
# the kl-loss  which is like this : 
# def kl_divergence(p, p_hat):
#     funcs = nn.Sigmoid()
#     p_hat = torch.mean(funcs(p_hat), 1)
#     p_tensor = torch.Tensor([p] * len(p_hat)).to(device)
#     return torch.sum(p_tensor * torch.log(p_tensor) - p_tensor * torch.log(p_hat) + (1 - p_tensor) * torch.log(1 - p_tensor) - (1 - p_tensor) * torch.log(1 - p_hat))

# finally  this was a simple autoencoder, we can have several layers
# and also you can use batchnormalization, etc for your deep autoencoders as well


#%%
# -VAE (Variational Autoencoders) 
# -Creating MNIST Like digits 
# -The Reparametrization Trick
# Variational Autoencoders (VAEs) have one fundamentally unique property that 
# separates them from vanilla autoencoders, and it is this property that makes
# them so useful for generative modeling: their latent spaces are, by design,
# continuous, allowing easy random sampling and interpolation.

# It achieves this by doing something that seems rather surprising at first: 
# making its encoder not output an encoding vector of size n, rather, outputting
# two vectors of size n: a vector of means, μ, and another vector of standard
# deviations, σ
# They form the parameters of a vector of random variables of length n, with 
# the i-th element of μ and σ being the mean and standard deviation of the i-th
# random variable, X_i, from which we sample, to obtain the sampled encoding 
# which we pass onward to the decoder:
# This stochastic generation means, that even for the same input, while the mean
# and standard deviations remain the same, the actual encoding will somewhat vary
# on every single pass simply due to sampling.
# read more  : https://towardsdatascience.com/intuitively-understanding-variational-autoencoders-1bfe67eb5daf
# There are other resouces for this as well. its highly recommened to read them: 
# https://www.jeremyjordan.me/variational-autoencoders/
# https://jaan.io/what-is-variational-autoencoder-vae-tutorial/
# https://www.youtube.com/watch?v=uaaqyVS9-rM
# http://blog.shakirm.com/2015/10/machine-learning-trick-of-the-day-4-reparameterisation-tricks/


# we'll also have an example concerning words(in NLP domain) and see how we can 
# leverage VAEs in that domain as well. for now lets see how we can implement this
# for vision domain. i.e. on mnist dataset

# note: 
# For variational autoencoders, the encoder model is sometimes referred to as
# the 'recognition model' whereas the decoder model is sometimes referred to as 
# the 'generative model'.

# if you havent read the links I gave you, go read them all. each single one of them
# will help you grasp one aspect very good!
#  
# now lets define our VAE model . 
class VAE(nn.Module):
    def __init__(self, embedding_size=100):
        super().__init__()

        self.embedding_size = embedding_size
        # our encoder will give two vectors one for μ and another for σ.
        # using these two parameter, we sample our z representation vector
        # which is used by the decoder to reconstruct the input. 
        # So we can say that The encoder ‘encodes’ the data which is 784-dimensional
        # into a latent (hidden) representation space z, which is much less than 784
        # dimensions. This is typically referred to as a ‘bottleneck’ because the 
        # encoder must learn an efficient compression of the data into this 
        # lower-dimensional space. Let’s denote the encoder qθ(z∣x). 
        # We note that the lower-dimensional space is stochastic: 
        #>> the encoder outputs  parameters to qθ(z∣x), which is a Gaussian probability
        #   density. 
        # We can sample from this distribution to get noisy values of the 
        # representations z .
      
        self.fc1 = nn.Linear(28*28, 512)
        self.fc1_mu = nn.Linear(512, self.embedding_size) # mean
        # we use log since we want to prevent getting negative variance
        self.fc1_std = nn.Linear(512, self.embedding_size) #logvariance

        # our decoder will accept a randomly sampled vector using
        # our mu and std. 
        # The decoder is another neural net. Its input is the representation z,
        # it outputs the parameters to the probability distribution of the data,
        # and has weights and biases ϕ. The decoder is denoted by pϕ(x∣z). 
        # Running with the handwritten digit example, let’s say the photos are 
        # black and white and represent each pixel as 0 or 1. 
        # The probability distribution of a single pixel can be then represented 
        # using a Bernoulli distribution. The decoder gets as input the latent 
        # representation of a digit z and outputs 784 Bernoulli parameters,
        # one for each of the 784 pixels in the image. 
        # The decoder ‘decodes’ the real-valued numbers in z into 784 real-valued 
        # numbers between 0 and 1. Information from the original 784-dimensional 
        # vector cannot be perfectly transmitted, because the decoder only has 
        # access to a summary of the information 
        # (in the form of a less-than-784-dimensional vector z). 
        # How much information is lost? We measure this using the reconstruction 
        # log-likelihood logpϕ(x∣z) whose units are nats. This measure tells us how 
        # effectively the decoder has learned to reconstruct an input image x given
        # its latent representation z.
        self.decoder = nn.Sequential( nn.Linear(self.embedding_size, 512), 
                                      nn.ReLU(),
                                      nn.Linear(512, 28*28),
                                      # in normal situations we wouldnt use sigmoid
                                      # but since we want our values to be in [0,1]
                                      # we use sigmoid. for loss we will then have  
                                      # to use, plain BCE (and specifically not BCEWithLogits)
                                      nn.Sigmoid())



    # Rather than directly outputting values for the latent state as we would 
    # in a standard autoencoder, the encoder model of a VAE will output 
    # "parameters(mean μ,variance σ) describing a distribution for each dimension in 
    # the latent space". 
    # Since we're assuming that our prior follows a normal distribution, we'll output
    # two vectors describing the mean and variance of the latent state distributions.
    # If we were to build a true multivariate Gaussian model, we'd need to define a
    # covariance matrix describing how each of the dimensions are correlated. 
    # However, we'll make a simplifying assumption that our covariance matrix only 
    # has nonzero values on the diagonal, allowing us to describe this information 
    # in a simple vector.
    # Our decoder model will then generate a latent vector by sampling from these
    # defined distributions and proceed to develop a reconstruction of the original
    # input.
    # However, this sampling process requires some extra attention. When training 
    # the model, we need to be able to calculate the relationship of each parameter 
    # in the network with respect to the final output loss using backpropagation. 
    # However, we simply cannot do this for a "random sampling process". Fortunately,
    # we can leverage a clever idea known as the "reparameterization trick" which 
    # suggests that we randomly sample ε from a unit Gaussian, and then shift the 
    # randomly sampled ε by the latent distribution's mean μ and scale it by the 
    # latent distribution's variance σ.
    # With this reparameterization, we can now optimize the parameters of the distribution
    # while still maintaining the ability to randomly sample from that distribution.
    # Note: In order to deal with the fact that the network may learn negative values
    # for σ, we'll typically have the network learn log(σ) and exponentiate(exp)) this value 
    # to get the latent distribution's variance.
    def reparamtrization_trick(self, mu, logvar):
        # note : why do we really want the epsilon? 
        # what is the intuition behind it : 
        # watch this : https://youtu.be/9zKuYvjFFS8?t=415
        # we divide by two because we are eliminating the negative values
        # and we only care about the absolute possible deviance from standard.
        # read in depth technical reasons here : 
        # all answers contain great explanations 
        # https://stats.stackexchange.com/questions/199605/how-does-the-reparameterization-trick-for-vaes-work-and-why-is-it-important
        # https://stats.stackexchange.com/questions/429315/why-is-reparameterization-trick-necessary-for-variational-autoencoders
        # https://stats.stackexchange.com/questions/342762/how-do-variational-auto-encoders-backprop-past-the-sampling-step/342815#342815
        # https://blog.neurallearningdymaics.com/2019/06/variational-autoencoders-1-motivation.html
        # http://ruishu.io/2018/03/14/vae/
        # up until now, you show have been convinced that we use reparameterization trick solely 
        # because otherwise we couldnt backprop to random node! this however is not the whole story!
        # Kingma: This reparameterization is useful for our case since it can be used to rewrite an 
        # expectation w.r.t qϕ(z∣x) such that the Monte Carlo estimate of the expectation is 
        # differentiable w.r.t. ϕ.
        # The issue is not that we cannot backprop through a “random node” in any technical sense. 
        # Rather, backproping would not compute an estimate of the derivative. 
        # Without the reparameterization trick, we have no guarantee that sampling large numbers of z
        # will help converge to the right estimate of ∇θ.
        # read in more detail here: 
        # http://gregorygundersen.com/blog/2018/04/29/reparameterization/
        # if you want to know about expectation and what it is, this may help 
        # https://revisionmaths.com/advanced-level-maths-revision/statistics/expectation-and-variance)
        std = torch.exp(0.5*logvar)
        # epsilon sampled from normal distribution with N(0,1)
        eps = torch.randn_like(std)
        # How to sample from a normal distribution with known mean and variance?
        # https://stats.stackexchange.com/questions/16334/ 
        # (tldr: just add the mu , multiply by the var) . 
        # why we use an epsilon, ? 
        # you should know by now, if not read the former links I provided.
        # basically there are 2 main explanations, the first one (Which is not true) is
        # because without it, backprop wouldnt work.
        # for  the random part we sample from normal distribution N(0,1)
        # and treat this as a mere input. (like the images that are input and we dont 
        # calculate the gradients for) 
        # we then shift this new sample with the mean and std we have and effectively
        # reach the very same result. that is we add our mu and scale it by std 
        # (since our eps has 0 mean and std 1, adding it with mu, and scaling it by std
        # will make it N(mum std) which is what we want. our expression also now can be
        # easily backpropagated. 
        # also you need to know that, it is also said this reparameterization trick is only done for 
        # numerical stability and actually  the basic way can be done as well! 
        # and finally, the actual reason was given above, we actually do this to guarantee the right estimate 
        # of ∇θ. without this, we have no guarantee that sampling large numbers of z, will help convertence
        # to the right estimates of ∇θ.
        return mu + eps*std
    # 
    def encode(self, input):
        input = input.view(input.size(0), -1)
        output = F.relu(self.fc1(input))
        # we dont use activations here
        mu = self.fc1_mu(output)
        log_var = self.fc1_std(output)

        # In its original form, VAEs sample from a random node z which is 
        # approximated by the parametric model q(z∣ϕ,x) of the true posterior.
        # Backprop cannot flow through a random node. Introducing a new parameter 
        # ϵ allows us to reparameterize z in a way that allows backprop to flow 
        # through the deterministic nodes. this is called reparamerization trick
        z = self.reparamtrization_trick(mu, log_var)
        return z, mu, log_var

    def forward(self, input):
        
        # our encoder recieves the input and produces two vectors
        # mean and std. a normal autoencoder, creates a set of atttibutes
        # in its representation vectot(e.g attributes or features describing
        # concepts such as, eye, smile, beard, gender, has glasses etc) in 
        # an input image of faces. so in other words, An ideal autoencoder 
        # will learn descriptive attributes of faces such as skin color, 
        # whether or not the person is wearing glasses, is female, etc. in
        # an attempt to describe an observation(input image) in some compressed representation.
        # for example a vector could be like (gender:-0.73, smile:0.99, glasses: 0.002, etc )
        # In the example above, we've described the input image in terms of its latent 
        # attributes using a 'single value' to describe each attribute. 
        # However, we may prefer to represent each latent attribute as a 'range of possible values'.
        # For instance, what 'single value' would you assign for the smile attribute if you feed
        # in a photo of the Mona Lisa? Using a variational autoencoder, 
        # we can describe latent attributes in probabilistic terms.
        # the mean and variance that we produce here, is used exactly for this very reason
        # using them, we are learning a distrubution for each attribute and thus mu and variance
        # specify a range of values for each attribute.
        # [With this approach, we'll now represent each latent attribute for a given input 
        # as a probability distribution. When decoding from the latent state, we'll randomly
        # sample from each latent state distribution to generate a vector as input for our decoder.]
        # thats why later on, we use these means, variances along with an epsilon(act as a random variable)
        # to reconstruct the input image. 
        # So By constructing our encoder model to output a range of possible values 
        # (a statistical distribution) from which we'll randomly sample to feed into our decoder
        # , we're essentially enforcing a continuous, smooth latent space representation.
        # For any sampling of the latent distributions, we're expecting our decoder 
        # to be able to accurately reconstruct the input. 
        # Thus, values which are nearby to one another in latent space should correspond
        # with very similar reconstructions.
        # Intuitively, the mean vector controls where the encoding of an input should be 
        # centered around, while the standard deviation controls the “area”, how much from 
        # the mean the encoding can vary. As encodings are generated at random from anywhere
        # inside the “circle” (the distribution), the decoder learns that not only is a
        #  single point in latent space referring to a sample of that class, 
        # but all nearby points refer to the same as well. 
        # This allows the decoder to not just decode single, specific encodings in the 
        # latent space (leaving the decodable latent space discontinuous), but ones that 
        # slightly vary too, as the decoder is exposed to a range of variations of the
        # encoding of the same input during training. 
        # The model is now exposed to a certain degree of local variation by varying the 
        # encoding of one sample, resulting in smooth latent spaces on a local scale, that is,
        # for similar samples. Ideally, we want overlap between samples that are not very 
        # similar too, in order to interpolate between classes. 
        # However, since there are no limits on what values vectors μ and σ can take on,
        # the encoder can learn to generate very different μ for different classes, 
        # clustering them apart, and minimize σ, making sure the encodings themselves don’t
        # vary much for the same sample (that is, less uncertainty for the decoder). 
        # This allows the decoder to efficiently reconstruct the training data.
        # What we ideally want are encodings, all of which are as close as possible to each
        # other while still being distinct, allowing smooth interpolation, and enabling the
        # construction of new samples.
        # In order to force this, we introduce the Kullback–Leibler divergence 
        # (KL divergence[2]) into the loss function. The KL divergence between two probability
        # distributions simply measures how much they diverge from each other. 
        # Minimizing the KL divergence here means optimizing the probability distribution
        # parameters (μ and σ) to closely resemble that of the target distribution.
        # Intuitively, this loss encourages the encoder to distribute all encodings 
        # (for all types of inputs, eg. all MNIST numbers), evenly around the center 
        # of the latent space. If it tries to “cheat” by clustering them apart into 
        # specific regions, away from the origin, it will be penalized.
        # Now, using purely KL loss results in a latent space results in encodings densely 
        # placed randomly, near the center of the latent space, with little regard for 
        # similarity among nearby encodings. The decoder finds it impossible to decode 
        # anything meaningful from this space, simply because there really isn’t any meaning.
        # Optimizing the two together, however, results in the generation of a latent space
        # which maintains the similarity of nearby encodings on the local scale via clustering,
        # yet globally, is very densely packed near the latent space origin 
        # (compare the axes with the original).
        # Intuitively, this is the equilibrium reached by the cluster-forming nature of the
        # reconstruction loss, and the dense packing nature of the KL loss, forming distinct
        # clusters the decoder can decode. This is great, as it means when randomly generating,
        # if you sample a vector from the same prior distribution of the encoded vectors, N(0, I), 
        # the decoder will successfully decode it. And if you’re interpolating, there are 
        # no sudden gaps between clusters, but a smooth mix of features a decoder can understand.
        z, mu, logvar = self.encode(input)
        # decoder 
        reconstructed_img = self.decoder(z)
        return reconstructed_img, mu, logvar

# Note :
# In my experience working on the VAE, the KL annealer helps to train the model.
# To be more specific, when training your encoder and decoders right off the 
# bat variationally (KL-term constant) can lead to a lot of instability while 
# training. So a good first step is to train them as a AE and at some moment 
# slowly switch on the KL term. It allows the model to arrive to a 'decent' 
# spot (trained as AE) before going VAE. A similar pattern lies with ORGAN, 
# you need to train pre-train your generators so that they are in a decent 
# spot before competing with the discriminator. There are many ways of doing
# this, most are just engineering, hence MOSES's approach also works.
# For the original VAE i think around epoch 30 we start the KL annealing.


# Also read : https://github.com/jxhe/vae-lagging-encoder
# The code seperates optimization of encoder and decoder in VAE, and performs 
# more steps of encoder update in each iteration. This new training procedure 
# mitigates the issue of posterior collapse in VAE and leads to a better VAE 
# model, without changing model components and training objective.

# for calculating loss, we can have several options 
# 1. use mse for reconstruction loss 
# 2. use BCE for reconstruction loss   
# when using bce we have two options, we can use reduce='sum'
# or we can use reduce='mean'. 
# if we want to use BCE with reduce='sum' we only calculate the kl
# with sum. but when we want to use BCE with reduce='mean' or mse
# we use sume(,-1) and then use torch.mean(loss_recons+kl)
# we also need to normalize our reconstruction loss by the input dim
# ension size. 

def loss_function(outputs, inputs, mu, logvar, reduction ='mean', use_mse = False):
    if reduction == 'sum':
        criterion = nn.BCELoss(reduction='sum')
        reconstruction_loss = criterion(outputs, inputs)
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return reconstruction_loss + KL
    else:
        if use_mse:
            criterion = nn.MSELoss()
        else: 
            criterion = nn.BCELoss(reduction='mean')
        reconstruction_loss = criterion(outputs, inputs)
        # normalize reconstruction loss
        reconstruction_loss *= 28*28
        KL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), -1)
        return torch.mean(reconstruction_loss + KL)


#%%
# I set this option to see the full stack-trace when a weird error occurs
# its good practice to get accustomed to the debugging/profiing facilities provided
# by pytorch, I might dedicate a separate section for this later on
# torch.autograd.set_detect_anomaly(True)
# torch.set_printoptions(profile='full')
#%%
# now lets train :
epochs = 50

embeddingsize = 2
interval = 2000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VAE(embeddingsize).to(device)
reduction='mean'
optimizer = torch.optim.Adam(model.parameters(), lr =0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 50)

for e in range(epochs):
    for i, (imgs, labels) in enumerate(dataloader_train):
        imgs = imgs.to(device)
        preds,mu, logvar = model(imgs)

        loss = loss_function(preds, imgs, mu, logvar, reduction=reduction, use_mse=False)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 
        if i% interval ==0:
            loss = loss/len(img) if reduction=='sum' else loss
            print(f'epoch {e}/{epochs} [{i*len(imgs)}/{len(dataloader_train.dataset)} ({100.*i/len(dataloader_train):.2f}%)]'
                  f'\tloss: {loss.item():.4f}'
                  f'\tlr: {scheduler.get_lr()}')
    scheduler.step()

#%% 
# save the model
torch.save({"states":model.state_dict(),
            "embedding_size":model.embedding_size,
            "optimizer":optimizer.state_dict(),
            "scheduler":scheduler.state_dict()},
            f"vae_{model.embedding_size}_mean_bce.pth")
print('model saved!')
#%%
# load the model 
states = torch.load(f"vae_{model.embedding_size}_mean_bce.pth")
model.load_state_dict(state_dict=states['states'])
print('weights loaded')
#%%
# generate sth
from torchvision import utils
interval = 1000
embeddingsize = 2
sample = torch.randn(size=(32, embeddingsize)).to(device)
# sample *= 0.5+ 0.5
model.eval()
imgs = model.decoder(sample)
print(imgs.shape)
imgs = imgs.view(-1, 1, 28,28)
img = utils.make_grid(imgs,nrow=8,normalize=True).cpu().detach().numpy().transpose(1,2,0)
plt.imshow(img, cmap='Greys_r')
#%%
# test
test_set_size = len(dataloader_test.dataset)
img_pairs = []
losses = []
interval = 10
with torch.no_grad():
    for i, (imgs, labels) in enumerate(dataloader_test):
        imgs = imgs.to(device)
        preds, mu, logvar = model(imgs)
        loss = loss_function(preds, imgs, mu, logvar, reduction=reduction, use_mse=False)
        losses.append({'val_loss':loss.item()})
        
        print(f'[{i*len(imgs)} / {test_set_size} ({100.*i/len(dataloader_test):.2f}%)]'
            f'\tloss: {(loss).item():.4f}')

        if i%interval==0:
            reconstructeds = preds.cpu().detach().view(-1, 1, 28, 28)
            imgs = imgs[:20].cpu().detach().numpy()
            recons = reconstructeds[:20].numpy()
            pairs = np.array([np.dstack((img1,img2)) for img1, img2 in zip(imgs,recons)])
            img_pairs.append(pairs)
#%%
# plot the losses using pandas! 
# this actually is very neat and comes handy very often!
# we can have a list of dictionaries, where each value is 
# attributed by a key. this way, our keys will be used as
# legends and we have a simple plot with minimum hassle
import pandas as pd 
pd.DataFrame(losses).plot()

#%%
# lets plot the classes in the latent space!
batch_size = 10000
dataloader_test2 = torch.utils.data.DataLoader(dataset_test,
                                               batch_size = batch_size,
                                               num_workers = num_workers,
                                               pin_memory=True)
imgs, labels = next(iter(dataloader_test2))
imgs = imgs.to(device)
z_test,_,_ = model.encode(imgs)
z_test = z_test.cpu().detach().numpy()

plt.figure(figsize=(12,10))
print(z_test.shape)
plt.scatter(x=z_test[:,0],
            y=z_test[:,1],
            c=labels.numpy(),
            alpha=.4,
            s=3**2,
            cmap='viridis')
plt.colorbar()
plt.xlabel('Z[0]')
plt.ylabel('Z[1]')
plt.show()
#%%

#%%
# display a 2D manifold of the digits
embeddingsize = model.embedding_size
n = 20  # figure with 20x20 digits
digit_size = 28

z1 = torch.linspace(-2, 2, n)
z2 = torch.linspace(-2, 2, n)

z_grid = np.dstack(np.meshgrid(z1, z2))
z_grid = torch.from_numpy(z_grid).to(device)
z_grid = z_grid.reshape(-1, embeddingsize)

x_pred_grid = model.decoder(z_grid)
x_pred_grid= x_pred_grid.cpu().detach().view(-1, 1, 28,28)
x = make_grid(x_pred_grid,nrow=n).numpy().transpose(1,2,0)
plt.figure(figsize=(10, 10))
plt.xlabel('Z_1')
plt.ylabel('Z_2')
plt.imshow(x)
plt.show()

#%%
def display_imgs_recons(img_pairs, nrows=8, rows=20, cols=1):
    img_cnt = len(img_pairs)
    print(img_cnt)
    fig = plt.figure(figsize=(28, 28))
    for i in range(img_cnt):
        grid_imgs = make_grid(torch.from_numpy(img_pairs[i]),
                            nrow=nrows,
                            normalize=True)
        ax = fig.add_subplot(rows, cols, i+1, xticks=[],yticks=[])
        ax.imshow(grid_imgs.numpy().transpose(1,2,0))
        save_image(grid_imgs, f'results/imgs_{i}.jpg')

display_imgs_recons(img_pairs, nrows=10, rows=8, cols = 1)
#%% 
# now lets generate new images by stepping through the latent space
import matplotlib.animation as animation
fig = plt.figure()
ax = fig.add_subplot(111)

plt.rcParams["animation.convert_path"] = r"C:\Program Files\ImageMagick\convert.exe"
z = torch.randn(size = (30, model.embedding_size)).to(device)
model.eval()
def animate(i): 
    imgs = model.decoder(z*(i*0.03)+0.02)
    imgs2 = imgs.view(imgs.size(0), 1, 28, 28)
    new_img = make_grid(imgs2).cpu().detach().numpy().transpose(1,2,0)
    ax.clear()
    ax.imshow(new_img)

anim = animation.FuncAnimation(fig, animate, frames=100, interval=300, repeat=True,repeat_delay=1000)
anim.save('vis.gif', writer="imagemagick", extra_args="convert", fps=20)
plt.show()

#%% 
# Conditional VAE 
# in vanilla VAE, the image generation is a random process and we have no control over it
# in this version, we are going to create a conditional variation, so that we can create
# images for a specific class/creteria.
# Conditional Variational Autoencoder (CVAE) is an extension of Variational Autoencoder (VAE), a generative model that we have studied in the last post. We’ve seen that by formulating the problem of data generation as a bayesian model, we could optimize its variational lower bound to learn the model.
# However, we have no control on the data generation process on VAE. 
# This could be problematic if we want to generate some specific data. 
# As an example, suppose we want to convert a unicode character to handwriting. 
# In vanilla VAE, there is no way to generate the handwriting based on the character 
# that the user inputted. Concretely, suppose the user inputted character ‘2’, how 
# do we generate handwriting image that is a character ‘2’? We couldn’t.
# Hence, CVAE [1] was developed. Whereas VAE essentially models latent variables and 
# data directly, CVAE models lantent variables and data, both conditioned to some 
# random variables. 
# for this we use the labels as our conditional factor. lets see how it is done. 
class VAE_Conditional(nn.Module):
    def __init__(self, embedding_size=2, num_classes = 10):
        super().__init__()
        self.embedding_size = embedding_size
        # we use this as our conditional factor
        # note however that The conditional variable c could be anything. 
        # We could assume it comes from a categorical distribution expressing
        # the label of our data, gaussian expressing some regression target,
        # or even the same distribution as the data 
        # (e.g. for image inpainting: conditioning the model to incomplete image).
        # here we are using class labels 
        self.num_classes = num_classes
        # encoder 
        self.fc1 = nn.Linear(28*28 + num_classes, 512)
        # we are actually adding the one_hot encoded length here. 
        self.fc_mu = nn.Linear(512, embedding_size )
        self.fc_std = nn.Linear(512, embedding_size)
        
        # decoder 
        # our decoder will utilize our conditional factor along side our embedding
        # so unlike vanilla vae, the decoder has embedding_size + condition
        # dims and differs with the last layer of the encoder output dim
        self.decoder = nn.Sequential(nn.Linear(embedding_size + num_classes, 512),
                                    nn.ReLU(), 
                                    nn.Linear(512 , 28*28),
                                    nn.Sigmoid())

    def encode(self, x, y):
        # accepts input image, and outputs z using reparametrization 
        x = x.view(x.size(0), -1)
        # y is a one hot encoded vector which we fuse(add) with our input
        inputs = torch.cat((x,y),dim=1)
        output = F.relu(self.fc1(inputs))
        mu = self.fc_mu(output) 
        std = self.fc_std(output)
        z = self.reparametrization_trick(mu, std)
        return z, mu, std

    def decode(self, z, y):
        z_cond = torch.cat((z,y), dim=1)
        output = self.decoder(z_cond)
        output = output.view(z.size(0), 1, 28, 28)
        return output

    def reparametrization_trick(self, mu, logvar):
        # since we need positive variance we devide by 2
        std = torch.exp(logvar * 0.5)
        # sample from a normal distribution N(0,1)
        eps = torch.randn_like(std)
        # produce z using mu and logvar
        # shift it by mu and scale it by std 
        return mu + eps * std

    def forward(self, input, y):
        z, mu, std = self.encode(input, y)
        output = self.decode(z, y)
        return output, mu, std

def one_hot(input, num_classes=10):
    result = torch.zeros(size=(input.size(0), num_classes))
    result[range(0,input.size(0)), input[:]] = 1
    return result

# z = torch.randint(0,9, size=(5,))
# print(z)
# print(one_hot(z))
def loss_function(outputs, imgs, mu, logvar, reduction='mean', use_mse=False):
    # this loss has two parts, a construction loss and a KL divergence loss which
    # shows how much distance exists between two given distrubutions. 
    if reduction=='mean':
        if use_mse:
            criterion = nn.MSELoss()
        else:
            criterion = nn.BCELoss(reduction='mean')
        recons_loss = criterion(outputs, imgs)
        # normalize the reconstruction loss
        recons_loss *= 28*28
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        # when using mean, we always sum over the last dim
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), -1)
        return torch.mean(recons_loss + kl)
    else:
        criterion = nn.BCELoss(reduction='sum')
        recons_loss = criterion(outputs, imgs)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recons_loss + kl

#%%
# now lets train 
epochs = 50
embedding_size = 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VAE_Conditional(embedding_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=20)
print(datetime.datetime.now())

for e in range(epochs):
    for i, (imgs,labels) in enumerate(dataloader_train):
        imgs = imgs.to(device)
        labels = labels.to(device)

        one_hot_labels = one_hot(labels).to(device)
        output, mu, logvar = model(imgs, one_hot_labels)
        loss = loss_function(output, imgs, mu, logvar )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'epoch/epochs: {e}/{epochs} loss: {loss.item():.4f}')
    scheduler.step()

interval = 1000
for i,(imgs, labels) in enumerate(dataloader_test):
    model.eval()
    with torch.no_grad():
        imgs = imgs.to(device)
        labels = labels.to(device)

        one_hot_labels = one_hot(labels).to(device)
        outputs, mu, logvar = model(imgs, one_hot_labels)
        loss = loss_function(outputs, imgs, mu, logvar )
        if i % interval:
            print(f'iter: {i}/{len(dataloader_test)} loss: {loss.item():.4f}')
            reconstructeds = preds.cpu().detach().view(-1, 1, 28, 28)
            count = 20 if imgs.size(0)>20 else imgs.size(0)
            imgs = imgs[:count].cpu().detach().numpy()
            recons = reconstructeds[:count].numpy()
            pairs = np.array([np.dstack((img1,img2)) for img1, img2 in zip(imgs,recons)])
            img_pairs.append(pairs)

#%%
# create a 2d manifold for z1 
# for this we feed our models encoder our test data or whatever data we want to visualize its z 
# latent space distrubution. then we use plt.scatter to plot the points
batch_size = 10000
dataloader_test2 = torch.utils.data.DataLoader(dataset_test,
                                               batch_size = batch_size,
                                               num_workers = num_workers,
                                               pin_memory=True)
imgs, labels = next(iter(dataloader_test2))
imgs = imgs.to(device)
labels = labels.to(device)
one_hot_labels = one_hot(labels).to(device)
z, _,_ = model.encode(imgs, one_hot_labels)

z_= z.cpu().detach().numpy()        
plt.scatter(x=z_[:,0], y=z_[:,1], c=labels.cpu().numpy(), alpha=.4,
            s=3**2,cmap='viridis')
plt.colorbar()
plt.xlabel('Z[0]')
plt.ylabel('Z[1]')
plt.show()
# as you can see the shape this time looks really messy compared to the original
# VAE. its becasue we are really modelig P(z|c) which c==y . 
# https://wiseodd.github.io/techblog/2016/12/17/conditional-vae/
# http://ijdykeman.github.io/ml/2016/12/21/cvae.html
# To generate an image of a particular number, just feed that number into the decoder
# along with a random point in the latent space sampled from a standard normal distribution. 
# Even if the same point is fed in to produce two different numbers, the process will work 
# correctly, since the system no longer relies on the latent space to encode what number
# you are dealing with. Instead, the latent space encodes other information, like stroke 
# width or the angle at which the number is written.
#%%
# lets create new samples
z = torch.randn(size=(3, model.embedding_size)).to(device)
labels = torch.tensor([[1],[2],[1]])
labels = one_hot(labels).to(device)
preds = model.decode(z, labels).detach().cpu()
img = make_grid(preds)
plt.imshow(img.numpy().transpose(1,2,0),cmap='gray')
#%%
# now lets see the digits 2d manifold

# the normal interpolation that we used for vanila va wont work here
# as we dont want to blend each class to each other, this simply wont happen
# as each latent space is also conditioned on a class. by this class we are 
# explicitly asking the network to create digits like it. so there is no point
# in alterations like this. smaller alterations this way will distort the digit
# you can uncomment this section and see it for your self. 
def vanila_vae_digits_manifold(n=10):
    z1 = torch.linspace(start=-9,end=9, steps=n)
    z2 = torch.linspace(start=-9, end=9, steps=n)
    # lets create a grid out of these two variables 
    # we use np.meshgrid and we stack them using dstack
    grid = np.dstack(np.meshgrid(z1, z2))
    grid = torch.from_numpy(grid).to(device)
    grid = grid.view(-1, model.embedding_size)
    labels = torch.randint(0,9,size=(grid.size(0),1))
    # remmember labels must be in one_hot encoded form!
    labels_one_hot = one_hot(labels).to(device)
    print(grid.shape)
    print(labels)
    print()
    preds = model.decode(grid, labels_one_hot).cpu().detach()
    img = make_grid(preds,nrow=n)
    fig = plt.figure(figsize=(n,n))
    ax = fig.add_subplot(111)
    ax.imshow(img.numpy().transpose(1,2,0))

# image reconstruction 
display_imgs_recons(img_pairs,nrows=10,rows=86,cols=1)


# number of digits 
n = 10 
num_classes = 10
z = torch.randn(size=(n*num_classes, model.embedding_size)).to(device)
print(z.shape)

labels_grid = torch.tensor([[i] * n for i in range(num_classes)])
print(labels_grid.flatten())

labels_one_hot = one_hot(labels_grid.flatten()).to(device)
print(f'z: {z.shape} labels: {labels_one_hot.shape}')

preds = model.decode(z, labels_one_hot).cpu().detach()
img = make_grid(preds, nrow=n)

fig = plt.figure(figsize=(n,n))
ax = fig.add_subplot(111)
ax.imshow(img.numpy().transpose(1,2,0))

#%%
# Disentagled Variational Autoencoders or (β-VAE)
# good reads : https://towardsdatascience.com/disentanglement-with-variational-autoencoder-a-review-653a891b69bd 
# https://lilianweng.github.io/lil-log/2018/08/12/from-autoencoder-to-beta-vae.html#contractive-autoencoder
# https://openreview.net/forum?id=Sy2fzU9gl
# https://arxiv.org/pdf/1901.09415.pdf
# https://arxiv.org/abs/1606.05579

# The basic idea in disentagled vae is that, we want different neurons in our latent distribution
# to be uncorollated, they all try to learn something different about the input data. In order to implement 
# this, the only thing that needs to be added to the vanilla VAE, is a β term.
# previously for the vanilla VAE we had : 
#     L = E_q(z|X)[log_p(X|z)] - D_KL[q(z|X)||p(z))]
# Now for the disentagled version (β-VAE) we just add the β like this : 
#     L = E_q(z|X)[log_p(X|z)] - βD_KL[q(z|X)||p(z))]
# so to put it simply, in a disentagled vae (B-Vae) the autoencoder will only use a varable if it 
# its important 

def fc_batchnorm_act(in_, out_, use_bn=True, act=nn.ReLU()):
    return nn.Sequential(nn.Linear(in_,out_),
                         act,
                         nn.BatchNorm1d(out_) if use_bn else nn.Identity())
                         
class B_VAE(nn.Module):
    def __init__(self, embedding_size=5):
        super().__init__()
        self.embedding_size = embedding_size
        
        # self.fc1 = nn.Linear(28*28, 512)
        self.encoder_entry = nn.Sequential(fc_batchnorm_act(28*28,512),
                                           fc_batchnorm_act(512,256),
                                           fc_batchnorm_act(256,128),
                                           fc_batchnorm_act(128,64))
        self.fc_mu = nn.Linear(64, embedding_size)
        self.fc_std = nn.Linear(64, embedding_size)

        self.decoder = nn.Sequential(fc_batchnorm_act(embedding_size, 64),
                                     fc_batchnorm_act(64,128),
                                     fc_batchnorm_act(128,256),
                                     fc_batchnorm_act(256,512),
                                     fc_batchnorm_act(512, 28*28,False,nn.Sigmoid()))
        # self.decoder = nn.Sequential(nn.Linear(embedding_size, 512),
        #                             nn.ReLU(),
        #                             nn.Linear(512, 28*28),
        #                             nn.Sigmoid())

    def reparameterization_trick(self, mu, logvar):
        # divide by two, since we want positive deviation only
        std = torch.exp(logvar * 0.5)
        # sample epslion from N(0,1) 
        eps = torch.randn_like(std)
        # sampling now can be done by shifting the eps by (adding) the mean 
        # and scaling it by the variance. 
        return mu + eps * std

    def encode(self, imgs):
        imgs = imgs.view(imgs.size(0), -1)
        # output = F.relu(self.fc1(imgs))
        output = self.encoder_entry(imgs)
        # remember we dont use nonlinearities for mu and logvar!
        mu = self.fc_mu(output)
        logvar = self.fc_std(output)
        z = self.reparameterization_trick(mu, logvar)
        return z, mu, logvar

    def decode(self, z):
        reconstructed_imgs = self.decoder(z)
        reconstructed_imgs = reconstructed_imgs.view(-1, 1, 28, 28)
        return reconstructed_imgs

    def forward(self, x):
        # encoder 
        z, mu, logvar = self.encode(x)
        # decoder
        reconstructed_imgs = self.decode(z)
        return reconstructed_imgs, mu, logvar

def loss_disentagled_vae(outputs, imgs, mu, logvar, Beta, reduction='mean', use_mse=False):
    # this loss has two parts, a construction loss and a KL divergence loss which
    # shows how much distance exists between two given distrubutions. 
    if reduction=='mean':
        if use_mse:
            criterion = nn.MSELoss()
        else:
            criterion = nn.BCELoss(reduction='mean')
        recons_loss = criterion(outputs, imgs)
        # normalize the reconstruction loss
        recons_loss *= 28*28
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        # when using mean, we always sum over the last dim
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), -1)
        # we use beta and multiply it by our kl term. this is specific to 
        # disentagled vae and is actually the main reason why the disentaglement 
        # work
        return torch.mean(recons_loss + (Beta*kl))
    else:
        criterion = nn.BCELoss(reduction='sum')
        recons_loss = criterion(outputs, imgs)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recons_loss + (Beta*kl)    

epochs = 50

embeddingsize = 5
interval = 2000
reduction='mean'
# beta is a value biger than 1 
Beta = 5.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = B_VAE(embeddingsize).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr =0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 50)

for e in range(epochs):
    for i, (imgs, labels) in enumerate(dataloader_train):
        imgs = imgs.to(device)
        preds,mu, logvar = model(imgs)

        loss = loss_disentagled_vae(preds, imgs, mu, logvar, Beta= Beta, reduction=reduction, use_mse=False)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 
        if i% interval ==0:
            loss = loss/len(img) if reduction=='sum' else loss
            print(f'epoch {e}/{epochs} [{i*len(imgs)}/{len(dataloader_train.dataset)} ({100.*i/len(dataloader_train):.2f}%)]'
                  f'\tloss: {loss.item():.4f}'
                  f'\tlr: {scheduler.get_lr()}')
    scheduler.step()
#%% 
# test
test_set_size = len(dataloader_test.dataset)
img_pairs = []
losses = []
interval = 10
with torch.no_grad():
    for i, (imgs, labels) in enumerate(dataloader_test):
        imgs = imgs.to(device)
        preds, mu, logvar = model(imgs)
        loss = loss_disentagled_vae(preds, imgs, mu, logvar, Beta= Beta, reduction=reduction, use_mse=False)
        losses.append({'val_loss':loss.item()})
        
        print(f'[{i*len(imgs)} / {test_set_size} ({100.*i/len(dataloader_test):.2f}%)]'
            f'\tloss: {(loss).item():.4f}')

        if i%interval==0:
            reconstructeds = preds.cpu().detach().view(-1, 1, 28, 28)
            imgs = imgs[:20].cpu().detach().numpy()
            recons = reconstructeds[:20].numpy()
            pairs = np.array([np.dstack((img1,img2)) for img1, img2 in zip(imgs,recons)])
            img_pairs.append(pairs)
#%% 
# import pandas as pd 
# pd.DataFrame(losses).plot()
model.eval()
# create sample image
z = torch.randn(size=(3, model.embedding_size)).to(device)
preds = model.decode(z).cpu().detach()
img = make_grid(preds)
plt.imshow(img.numpy().transpose(1,2,0))
#%%
# visualize latent space
n = 1 
z = torch.randn(size=(n,model.embedding_size)).to(device)
print(z.shape)
#%%
fig = plt.figure()
ax = fig.add_subplot(111)
preds = model.decode(z).cpu().detach()
img_latent_space = make_grid(preds,nrow=5).numpy().transpose(1,2,0)
ax.imshow(img_latent_space)
#%%
def change_latentvariable(z, n=3, count = 3, dim=0):
    z_new = torch.zeros(size=(n, count, z.size(-1)))
    for i in range(count):
        z_new[:,i,:] = z[:, :]
        z_new[:,i, dim]=  z_new[:,i, dim] - (0.2*i)
    return z_new

def show_manifold(z, n, count, dim , device):
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111)
    latent_space_manifold = change_latentvariable(z, n, count, dim).to(device)
    print(latent_space_manifold.shape)
    preds = model.decode(latent_space_manifold.view(-1,model.embedding_size)).cpu().detach()
    img_latent_space_man = make_grid(preds,nrow=count).numpy().transpose(1,2,0)
    ax.imshow(img_latent_space_man)

show_manifold(z, n=n, count=5, dim=3, device=device)
show_manifold(z, n=n,  count=5, dim=1, device=device)
show_manifold(z, n=n,  count=5, dim=2, device=device)
show_manifold(z, n=n,  count=5, dim=3, device=device)
show_manifold(z, n=n,  count=5, dim=4, device=device)
# visualize the 2d manifold 
#%%
# variations over the latent variable :
z_dim = model.embedding_size
sigma_mean = 2.0*torch.ones((z_dim))
mu_mean = torch.zeros((z_dim))

# Save generated variable images :
nbr_steps = 8
gen_images = torch.ones( (nbr_steps, 1, 28, 28) )

for latent in range(z_dim) :
    #var_z0 = torch.stack( [mu_mean]*nbr_steps, dim=0)
    var_z0 = torch.zeros(nbr_steps, z_dim)
    val = mu_mean[latent]-sigma_mean[latent]
    step = 2.0*sigma_mean[latent]/nbr_steps
    print(latent, mu_mean[latent]-sigma_mean[latent], mu_mean[latent], mu_mean[latent]+sigma_mean[latent])
    for i in range(nbr_steps) :
        var_z0[i] = mu_mean
        var_z0[i][latent] = val
        val += step

    var_z0 = var_z0.to(device)


    gen_images_latent = model.decode(var_z0)
    gen_images_latent = gen_images_latent.cpu().detach()
    gen_images = torch.cat( [gen_images, gen_images_latent], dim=0)

img = make_grid(gen_images)
plt.imshow(img.cpu().numpy().transpose(1,2,0))
#%%
def plot_latentspace(num_rows,num_cols=9,figure_width=10.5,image_height=1.5):
    fig = plt.figure(figsize=(figure_width, image_height * num_rows))
    
    for i in range(num_rows):
        z_i_values = np.linspace(-3.0, 3.0, num_cols)
        z_i = z[0][i].detach().cpu().numpy()
        z_diffs = np.abs((z_i_values - z_i))
        j_min = np.argmin(z_diffs)
        for j in range(num_cols):
            z_i_value = z_i_values[j]
            if j != j_min:
                z[0][i] = z_i_value
            else:
                z[0][i] = float(z_i)
                
            x = model.decode(z).detach().cpu().numpy()
            
            ax = fig.add_subplot(num_rows, num_cols, i * num_cols + j + 1)
            ax.imshow(x[0][0], cmap='gray')
            
            if i == 0 or j == j_min:
                ax.set_title(f'{z[0][i]:.1f}')
            
            if j == j_min:
                ax.set_xticks([], [])
                ax.set_yticks([], []) 
                color = 'mediumseagreen'
                width = 8
                for side in ['top', 'bottom', 'left', 'right']:
                    ax.spines[side].set_color(color)
                    ax.spines[side].set_linewidth(width)
            else:
                ax.axis('off')
        z[0][i] = float(z_i)
        
    plt.tight_layout()
    fig.subplots_adjust(wspace=0.04)
num_rows = z.shape[-1]
plot_latentspace(num_rows)
#%% 
# # Contractive Autoencoder
# main paper : http://www.icml-2011.org/papers/455_icmlpaper.pdf
# ref1: https://wiseodd.github.io/techblog/2016/12/05/contractive-autoencoder/
# ref2: https://www.youtube.com/watch?v=BW7P1fvnAWk
# So we have seen many different flavors of a Autoencoders. However, there is one more autoencoding method on top of them, dubbed 
# Contractive Autoencoder (Rifai et al., 2011).
# The contractive autoencoder is categorized as a regularizier autoencoder. in other words
# This autoencoder specifically prevents an overcomplete autoencoder from learning identity
# which basically means, it prevents it from copying/memorizing the input as apposed to learning
# benificial features to reconstruct the input that matters for us. 
# using the regularization term that we will get to shortly, this can also be used on undercomlete
# encoders as well. 
# It will achieve this , by adding a new term to the weight matrix. which is as follows: 
#                  2 
# Ω(Ø) =  ‖ Jx(h) ‖p 
# lets expand this and see what this term actually is. 
#     2  
# ‖  ‖p  is called a norm. a Frobenius Norm. Frobenius Norm is like L2 norm (Eculidean norm)
# and is used on matrix and is calculated as the square root of the sum of the absolute squares of its elements
# which means, we simply sum all the elements in a mxn matrix and then take the square root of it.
# Now what is it applied on? it is applied on Jx(h). what is Jx(h) you may ask? Its the Jacobian
# Matrix. what is a Jacobian matrix. it simply a matrix of partial derivatives of all elements 
# with respect to all inputs. if you closely you can see that we have Jx(h), J, x and h . 
# x is our input, h is our parameters. Jx(h) means, a matrix of partial derivitives of all parameters
# with respect to x. How does it look like ? this is roughly how it looks like : 
# suppose, input has n dimensions and our hidden layer has h dimensions. 
# our resulting jabobian matrix will have n+k dimensions 
#         | dh1/dx1, dh1/dx2, dh1/dx3, ..., dh1/dxn|
# Jx(h) = | dh2/dx1, dh2/dx2, dh2/dx3, ..., dh2/dxn|
#         | dh3/dx1, dh3/dx2, dh3/dx3, ..., dh3/dxn|
#         |  ...      ...       ...    ...    ...
#         | dhk/dx1, dhk/dx2, dhk/dx3, ..., dhk/dxn| 
# As you can see, each column, shows the partial derivitives for all neurons with respect
# to a single input. for example, the first column, shows the partial derivitives for all
# neurons with respect to the "first" input, likewise, the second column, shows the partial
# derivative for all neurons in our hidden layer with respect to the second input and so on.
# So basically when we are taking the derivative of a vector with respect to another vector
# we get a matrix that you see above.  we can say each row belongs to one neuron and each col
# represents an the respective neurons gradient with respect to all inputs.
# So, what does all of this mean? what does each entry in the Jacobian matrix mean for us? 
# what can we infer from lets say element (i,l) of this matrix? 
# Thie (i,l)th element simply tells us, howmuch the h(l) changes with a change in x(i) 
# basically each entry in the jacobian matrix captures the variation in the output of
# the lth neuron with a small variation in the jth input. 
# OK, now what does the Frobenious norm capture here? 
# what do we get by adding all the lements absolute values and squaring them? 
# This basically shows, howmuch each of these lements vary with respect to the input 
# and we are taking the square of that (to make it more prounounced)
# So this whole term is added to the loss function and the loss gets minimized. 
# This means, we want our Frobenious norm to get minimized as well, which means we want
# it to idealy be zero or near zero. (we said ideally becasue in actuality it wont be zero
# as there is always a tradeoff between L(Ø) and Ω(Ø) (remember loss = L(Ø) + Ω(Ø)), 
# if the norm gets to zero, it means L(Ø) will be very high)
# 
# Lets get a better intuition on how this works : 
# imagine, for example, dh1/dx1 goes actually to zero(dh1/dx1=0). what would that mean? 
# It means, h1 is not sensitive to variations in x1!  
#  but what does the original concept mandates here? what did we want to capture? 
# we wanted the neurons to capture these important characteristics (variations in input)
# so if x1 changes, we want h1 to change as well .
# So we wanted to capture the important characteristics in the input by each neuron, but 
# now, we have added a contradictory condition that we dont want to capture these kinds of 
# variations! So what is happening here? 
# L(Ø) says we should be able to capture these variations ortherwise I will not be able to 
# reconstruct the input. if all of my h_i's are not sensitive to variations in x1, this means
# if I give it any x1, it will produce the same h_i. 
# Lets recap again, basically, when we add the frobenious norm to the loss, and want to minize
# this norm as well which means, going toward zero, which again means, all the dh_i's
# need to go toward zero so that (dh/dx) is zero or near zero. and this simply means
# h_i is not sensitive to variations in input. while clearly we said we want to capture such
# variations (using the L(Ø) part in our loss)! 
# Thats the catch here, we have two contradictory terms in our loss, one tries to capture 
# the important features, while the other one tries just the opposite. 
# L(Ø) says, capture the variations in the data while
# Ω(Ø) says, do not capture the variations in teh data!
# Whats the tradoff here? capture only the important variations in the data and 
# do not capture the ones that are not important.
# look at the following plot for example :
#                  Y
#             . %8.                                   
#     .  . .    S@ .  .  . .  .  . .  .  . .  .  . .  
#    .     . .  8X .       .       .       .       .
#      .      . @%   . .     . .     . .     . .    
#  .     .  . . 8t .     .       . .     .       .  
#    .  .     . 8% .  .   . .  .   .:. U1  . .  .   .
#   .     . .   8t .    .       ..8%.t .         .  
#     .       . 8%  .      . . .%@888.    . . .    
#    U2. .  .   8t .  .  .   .88 ;t.     .        . 
# .X@:.       . 8%  .   .  ..8S:..  .  .     .  .   
#  :X8X    .    @% .     .:SX...          .       . 
#    . 8  . .  .8t .  . . ;8;.   .  . .     . .     
#   .  %8@;.  . @%  . . %%8;.           .       .  .
#     . ..X8@. .@% . .8SS%:.   .  .  .    .   .   . 
#         . ;@8%8t :S@%:.        .     .    .   .   
#  .  .       .X @  8S.     . .    .     .          
#      . . . . .@888: . . .    . .  . . .  . . . . .
#   . X@@X@@@X@X888@@@X@@@X@@@@@@X@@@@@@X@@@@@@X@.;;
#              .:.                                X;
# 
# So This is how it goes, we have 2 dimensions u1 and u2 , of which u1 is more important
# as the data variation along the u1 dimension is something that we should care about
# What about the variations in u2? Not important, they seem like noises, becasue these 
# variations , they are not all laying up on the centeral line, they are slighly away from
# the line. here are some variations. but should we go out of our way to capture these
# variations? does it make sense to do that? no!     
# So it makes sense to maximize a neuron to be sensitive to variations along U1 
# but it does not make sense to make neuron sensitive for these variations along other
# dimension which is U2 . 
# So by doing so we balance the two conditions. one condition tries to capture all the
# important variations and says do this, but do it only for dimensions that only their features
# (variations) are important . the other condition says, dont capture information , it says
# do this, but only for the dimensions that are not important.
# This is like PCA  (unbder certain conditions, vanilla autoencoder is equivalent to PCA)
# the passage from "Representation Learning: A Review and New Perspectives" by Bengio,
# Courville, et al. states:
# "In the case of a linear auto-encoder (linear encoder and
# decoder) with squared reconstruction error, the basic autoencoder
# objective in Equation 19 is known to learn the same
# subspace13 as PCA. This is also true when using a sigmoid
# nonlinearity in the encoder (Bourlard and Kamp, 1988), but
# not if the weights W and W0 are tied (W0 = WT )." 
# 
# I read this as saying that even with with a sigmoid in the encoder, if
# the weights are untied, you still end up learning the same subspace as
# PCA (?) and if the weights _are_ tied you do not.  Why? 

# The reasons I'm aware of for using tied weights:
# 1. In the linear case the optimal solution is PCA, which can be
# obtained with tied weights.
# 2. It has a regularization effect:
#     2a. Less parameters to be optimized
#     2b. It can prevent degenerate solutions, in particular those with
# very small weights in encoder, compensated by very large weights in
# decoder (something that would allow for instance a near-linear
# solution to be found with tanh nonlineraities)
# 3. Less parameters to be stored (=> lower memory footprint)

# That being said, lots of people also use un-tied weights... there's no
# general rule that tied > untied (or the reverse) -- it depends on the
# architecture & data.

# For the last question, this is because the optimal representation of
# your data in R^d in the sense of linear reconstruction error in
# original space R^n (d < n) is when the R^d representation is obtained
# by a PCA projection (up to some invertible linear transform). So if
# the weights are untied you can in the encoding part learn this
# projection, and in the decoding part learn the linear reconstruction.
# While if the weights are tied, in general you can't do this with a
# nonlinear encoder (the weights to obtain the PCA projection with the
# nonlinearity transform won't be the transposed of the PCA linear
# reconstruction). 

# I don't know if people do it explicitly for this purpose. Now that I think about it, 
# another motivation for tied weights may also come from RBMs, where the "reconstruction" 
# weights to compute P(x|h) are the transposed version of the "encoding" weights 
# computing P(h|x).
# I don't think that in general you can claim that tied weights lead to more "interesting" 
# solutions than PCA in the context of dimensionality reduction. Also, the better auto-encoders
# usually have a nonlinear decoder, in which case PCA is no longer optimal and untying weights 
# may actually help.
# -=- Olivier

# Autoencoders with tied weights have some important advantages :

#     It's easier to learn.
#     In linear case it's equvialent to PCA - this may lead to more geometrically adequate coding.
#     Tied weights are sort of regularisation.

# But of course - they're not perfect : they may not be optimal when your data comes from 
# highly nolinear manifold. Depending on size of your data I would try both approaches - 
# with tied weights and not if it's possible.

# UPDATE :
# You asked also why representation which comes from autoencoder with tight weights might be 
# better than one without. Of course it's not the case that such representation is always 
# better but if the reconstruction error is sensible then different units in coding layer 
# represents something which might be considered as generators of perpendicular features which
#  are explaining the most of the variance in data (exatly like PCAs do). This is why such 
# representation might be pretty useful in further phase of learning.

# https://medium.com/@SeoJaeDuk/arhcieved-post-personal-notes-about-contractive-auto-encoders-part-1-ef83bce72932



# So in 
# The idea of Contractive Autoencoder is to make the learned representation to be 
# robust towards small changes around the training examples. It achieves that by 
# using different penalty term imposed to the representation.
# The loss function for the reconstruction term is similar to previous Autoencoders 
# that we have been seen, i.e. using ℓ2 loss(MSE or BCE). The penalty term, however is more 
# complicated: we need to calculate the representation’s jacobian matrix with 
# regards of the training data.
# so basically our loss in this case would be : 
# L = \lVert X - \hat{X} \rVert_2^2  + \lambda \lVert J_h(X) \rVert_F^2 
# in which
# \lVert J_h(X) \rVert_F^2 = \sum_{ij} \left( \frac{\partial h_j(X)}{\partial X_i} \right)^2
# that is, the penalty term is the Frobenius norm of the jacobian matrix, which is the 
# sum squared over all elements inside the matrix. We could think Frobenius norm as the
# generalization of euclidean norm.
# 
# In the loss above, clearly it’s the calculation of the jacobian that’s not 
# straightforward. Calculating a jacobian of the hidden layer with respect to 
# input is similar to gradient calculation. Recall than jacobian is the generalization
# of gradient, i.e. when a function is a vector valued function, the partial derivative
# is a matrix called jacobian.
# However, we use autograd instead of creating the jacobian matrix as it is not practical for
# complex architectures and it imposes a huge overhead!

# https://medium.com/@SeoJaeDuk/arhcieved-post-personal-notes-about-contractive-auto-encoders-part-1-ef83bce72932
# carefully created penalty term can result in extracting more useful and effective features
# that gives insight to the given data. The penalty term invented by the authors of this paper
# makes the auto-encoders learned features to be locally invariant without any preference for
# particular directions. (they obtain invariance in the directions that make sense in the 
# context of the given training data, i.e., the variations that are present in the data should
# also be captured in the learned representation, but the other directions may be contracted 
# in the learned representation.) And the description of the penalty term can be seen below.
# Additionally, one very interesting question the authors asked is the notion, how can we 
# extract robust features? (aka features that are robust to small changes in the given input).
# The way they did is by adding a penalty term that is sensitive to the given input, and as the
# network trains, it’s objective is to make that sensitivity smaller and smaller.

# Note when the auto encoders do not have an activation function the above loss function is
# same as having weight decay (L2 penalty). Additionally, the authors of this paper have 
# investigate the case where the weights are tied. Another interesting fact is 
# sparse auto-encoders that outputs many zeroed out activation units achieves highly 
# contractive mapping, even without an concrete objective functions. One difference between
# de-noising auto encoders to CAE is DAE makes the network robust to both encoders and decoders,
# CAE only makes the encoder portion robust. (CAE robust is achieved via analytical solution,
# while DAE is achieved stochastically. )

# From this ppt I learned that the proof that a single layer neural network was based on using 
# exponentially large number of neurons and hence it is not practical. Also weight sharing is a
# method to reuse the weights on different layers, they will differ since their gradient differs. 
# Finally, auto encoders with regularization learns to model the keep only sensitivity to 
# variations on the manifold. (Reconstruction → Forces variations on the manifold, 
# regularization → want to remove variations.)
# Additionally, with the contraction loss, the network is trying to find features that are
# robust to given input. And when we have tied weights auto-encoders we can see the 
# relationship between the decoder weights to encoder weights as smoothen weights. 
# (In terms of the original data.) Here are some links to why we might use tied 
# auto-encoders, https://stackoverflow.com/questions/36889732/tied-weights-in-autoencoder,
#  https://groups.google.com/forum/#!topic/theano-users/QilEmkFvDoE

# this is a good read if you want to know why I wrote the loss function the way i did :
# https://mc.ai/how-pytorch-backward-function-works/
def fc_batchnorm_act(in_, out_, use_bn=True, act=nn.ReLU()):
    return nn.Sequential(nn.Linear(in_,out_),
                         act,
                         nn.BatchNorm1d(out_) if use_bn else nn.Identity())

class Reshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape 

    def forward(self, input):
        return input.view(self.shape)

class Contractive_AutoEncoder(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()
        self.embedding_size = embedding_size
        self.encoder = nn.Sequential(Reshape(shape=(-1, 28*28)),
                                     fc_batchnorm_act(28*28, 400, False, nn.Sigmoid()))

        self.decoder = nn.Sequential(fc_batchnorm_act(400, 28*28, False, nn.Sigmoid()),
                                     Reshape(shape=(-1, 1, 28, 28)))     
                                                            
        # self.encoder = nn.Sequential(Reshape(shape=(-1, 28*28)),
        #                              fc_batchnorm_act(28*28,512),
        #                              fc_batchnorm_act(512,256),
        #                              fc_batchnorm_act(256,128),
        #                              # dont use batchnorm on the last layer of enc
        #                              fc_batchnorm_act(128, embedding_size, False))
        
        # self.decoder = nn.Sequential(fc_batchnorm_act(embedding_size,128),
        #                              fc_batchnorm_act(128,256),
        #                              fc_batchnorm_act(256,512),
        #                              fc_batchnorm_act(512, 28*28, False, nn.Sigmoid()),
        #                              Reshape(shape=(-1, 1, 28, 28)))

    def forward(self, input):
        # flatten the input
        # shape = input.shape
        # input = input.view(input.size(0), -1)
        # outputs_e = F.relu(self.encoder(input))
        # outputs = F.sigmoid(self.decoder(output_e))
        # outputs = output.view(*shape)
        outputs_e = self.encoder(input)
        outputs = self.decoder(outputs_e)
        return outputs_e, outputs

def loss_function(output_e, outputs, imgs, lamda = 1e-4, device=torch.device('cuda')):
 
    criterion = nn.MSELoss()
    assert outputs.shape == imgs.shape ,f'outputs.shape : {outputs.shape} != imgs.shape : {imgs.shape}'
    loss1 = criterion(outputs, imgs)

    output_e.backward(torch.ones(outputs_e.size()).to(device), retain_graph=True)    
    # Frobenious norm, the square root of sum of all elements (absolute value)
    # in a jacobian matrix 
    loss2 = torch.sqrt(torch.sum(torch.abs(imgs.grad)))
    imgs.grad.data.zero_()
    loss = loss1 + (lamda*loss2) 
    return loss 

# based on the manual calculation of gradients as 
# done here :
# https://github.com/avijit9/Contractive_Autoencoder_in_Pytorch/blob/master/CAE_pytorch.py
# remember this loss only works for a 2 layer net, that uses sigmoid! 
# I just included this here as a reference 
def loss_function2(W, x, recons_x, h, lam=1e-4):
    mse = F.mse_loss(recons_x, x)
    # Since: W is shape of N_hidden x N. So, we do not need to transpose it as
    # opposed to #1
    dh = h * (1 - h) # Hadamard product produces size N_batch x N_hidden
    # Sum through the input dimension to improve efficiency, as suggested in #1
    w_sum = torch.sum(W**2, dim=1)
    # unsqueeze to avoid issues with torch.mv
    w_sum = w_sum.unsqueeze(1) # shape N_hidden x 1
    contractive_loss = torch.sum(torch.mm(dh**2, w_sum), 0)
    return mse + contractive_loss.mul_(lam)

# torch.autograd.set_detect_anomaly(True)
epochs = 50 
interval = 2000
embedding_size = 5
lam = 1e-4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Contractive_AutoEncoder(embedding_size).to(device)
optimizer = optim.Adam(model.parameters(), lr =0.001)
print(model)

W = model.encoder[1][0].weight

for e in range(epochs):
    for i, (imgs, labels) in enumerate(dataloader_train):
        imgs = imgs.to(device)
        labels = labels.to(device)
        # note imgs is not a leaf node, so the gardients wouldnot be ratained
        # in order to ratain gradients for non leaf nodes, use retain_graph
        # .grad field is only populated for leaf Tensors. If you want it for other Tensors, 
        # you can use the imgs.retain_grad() function to get the .grad field populated 
        # for non-leaf Tensors. but I found it esaier to just enable/diable the grads
        # inside the training loop and thus outside of lossfunction. 
        # also imgs.retain_grad() shuold be called before doing forward() as it will
        # instruct the autograd to store grads into nonleaf nodes. 
        imgs.retain_grad()
        imgs.requires_grad_(True)
        
        outputs_e, outputs = model(imgs)
        loss = loss_function(outputs_e, outputs, imgs, lam,device)
        # loss = loss_function2(W, imgs, outputs, outputs_e, lam)

        imgs.requires_grad_(False)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'epoch/epochs: {e}/{epochs} loss: {loss.item():.4f}')

# test
for i, (imgs, labels) in enumerate(dataloader_test):
    model.eval()
    imgs = imgs.to(device)
    labels = labels.to(device)

    imgs.requires_grad_(True)
    outputs_e, outputs = model(imgs)

    loss = loss_function(outputs_e, outputs, imgs, lam, device)
    # loss = loss_function2(W, imgs, outputs, outputs_e, lam)
    imgs.requires_grad_(False)
    print(f'iter/iterss: {i}/{len(dataloader_test)}  loss: {loss.item():.4f}')

# reconstruction test 
imgs, labels = next(iter(dataloader_test))
outputs_e, outputs_img = model(imgs.to(device))
imgs_org = make_grid(imgs[:10].detach().cpu(),nrow=10).numpy().transpose(1,2,0)
img = make_grid(outputs_img[:10].detach().cpu(),nrow=10).numpy().transpose(1,2,0)
img = np.concatenate((imgs_org,img),axis=0)
plt.imshow(img)

#%% 

#%%



#%%


#%%
# to do : 
# Sequence-to-Sequence Autoencoder
# https://mc.ai/less-pain-more-gain-a-simple-method-for-vae-training-with-less-of-that-kl-vanishing-agony/
# There is growing interest in exploring the use of variational auto-encoders 
# (VAE), a deep latent variable model, for text generation. Compared to the 
# standard RNN-based language model that generates sentences one word at a time
# without the explicit guidance of a global sentence representation, VAE is 
# designed to learn a probabilistic representation of global language features
# such as topic, sentiment or language style, and makes the text generation 
# more controllable. For example, VAE can generate sentences with a specific 
# tense, sentiment or topic.

# However, training VAE on languages is notoriously difficult due to something called KL vanishing. 
# While VAE is designed to learn to generate text using both local context and global features, it 
# tends to depend solely on local context and ignore global features when generating text. When this 
# happens, VAE is essentially behaving like a standard RNN language model.
# In “Cyclical Annealing Schedule: A Simple Approach to Mitigating KL Vanishing,” to be presented at 
# 2019 Annual Conference of the North American Chapter of the Association for   Computational Linguistics 
# (NAACL), researchers at Microsoft Research AI and Duke University propose an extremely simple remedy to 
# KL vanishing as well as their proposal to make the code publicly available on Github. 
# The remedy is based on a new scheduling scheme called Cyclical Annealing Schedule. Intuitively, 
# during the course of VAE training, we periodically adjust the weight of the KL term in the objective 
# function, providing the model opportunities to learn to leverage the global latent variables in text 
# generation, thus encoding as much global information in the latent variables as possible. The paper 
# briefly describes KL vanishing and why it happens, introduces the proposed remedy, and illustrates the 
# VAE learning process using a synthetic dataset.
# What is KL vanishing and why does it happen?
# VAEs aim to learn probabilistic representations z of natural languages x, with an objective consisting 
# of two terms: (1) reconstruction to guarantee the inferred latent feature z can represent its corresponding
# observed sentence; and (2) KL regularization to leverage the prior knowledge to modulate language understanding.
# The two terms are balanced by a weighting hyper-parameter β:
# When applied on text corpora, VAEs typically employ an auto-regressive decoder, which sequentially generates 
# the word tokens based on ground-truth words in the previous steps, in conjunction with latent z. Recent work 
# has found that naïve training of VAEs (keeping constant β=1) leads to model degeneration—the KL term becomes 
# vanishingly small. This issue causes two undesirable outcomes: (1) the learned features are almost identical 
# to the uninformative Gaussian prior, for all observed languages; and (2) the decoder completely ignores the 
# latent feature, and the learned model reduces to a simpler neural language model. Hence, the KL vanishing issue.
# This negative result is so far poorly understood. We developed a two-path competition interpretation to shed 
# light on the issue. Let’s first look at the standard VAE in Figure 1 (a), below. The reconstruction of sequence 
# x=[x1 ,…,xT] depends only on one path passing through the encoder ϕ, latent representation z and decoder Θ. 
# However, when an auto-regressive decoder is used in a VAE, there are two paths from observed x to its 
# reconstruction, as shown in Figure 1(b). Path A is the same as that in the standard VAE, where z serves as 
# the global representation that controls the generation of x; Path B leaks the partial ground-truth information 
# of x at every time step of the sequential decoding. It generates xt conditioned on x<t=[x1,…,xt-1]. Therefore, 
# Path B can potentially bypass Path A to generate xt, leading to KL vanishing. From this perspective, 
# we hypothesize that the KL vanishing problem is related to the low quality of z at the beginning phase of 
# decoder training. This is highly possible when the naive constant schedule of β=1 is used, as the KL term 
# pushes z close to an uninformative prior, less representative of the corresponding observations. This lower 
# quality z introduces more difficulties in reconstructing x, and eventually blocks the information flow via Path A. 
# As a result, the model is forced to learn an easier solution to decoding—generating x via Path B only.
 

# Figure 1: Illustration of information flows on (a) one path in a standard VAE, and (b) two paths in a VAE with an auto-regressive decoder.
# Cyclical Annealing Schedule

# A simple remedy via scheduling β during VAE training was proposed by Bowman, et al, as shown in Figure 2(a). 
# It starts with β=0 at the beginning of training, and gradually increases β until β=1 is reached. This monotonic 
# schedule of β has become the de facto standard in training text VAEs, and has been widely adopted in many NLP 
# tasks. Why does it improve the performance empirically? When β<1, z is trained to focus more on capturing useful 
# information for reconstruction of x. When the full VAE objective is considered (β=1), z learned earlier can be 
# viewed as VAE initialization; such latent features are much more informative than the random start in constant 
# schedule and thus are ready for the decoder to use.
# Figure 2: Annealing β with (a) the monotonic schedule and (b) the cyclical schedule.
# Figure 2: Annealing β with (a) the monotonic schedule and (b) the cyclical schedule.
# Is there a better schedule? It is key to have meaningful latent z at the beginning of training the decoder, so 
# that Path A is utilized. The monotonic schedule under-weights the prior regularization when β<1; the learned z 
# tends to collapse into a point estimate. This underestimation can result in sub-optimal decoder learning. 
# A natural question concerns how one can get a better distribution estimate for z as initialization, while decoder 
# has the opportunity to leverage such z in learning.
# Our proposal is to use the latent z trained under the full VAE objective as initialization. 
# To learn to progressively improve z we propose a cyclical schedule for β that simply repeats the monotonic
# schedule multiple times as shown in Figure 2(b). We start with β=0, increase β at a fast rate, and then stay 
# at β=1 for subsequent learning iterations. This completes one period of monotonic schedule. It encourages the 
# model to converge towards the VAE objective, and infers its first raw full latent distribution. Unfortunately,
# β=1 gradually blocks Path A, forbidding more information from passing through z. Crucially, we then start the 
# second period of β annealing and training is continued at β=0 again. This perturbs the VAE objective, dislodges 
# it from the convergence, and reopens Path A. Importantly, the decoder now (1) has the opportunity to directly 
# leverage z, without obstruction from KL; and (2) is trained with the better latent z than point estimates, as 
# the full distribution learned in the previous period is fed in. We repeat this β annealing process several 
# times to achieve better convergences.
# Visualization of learning dynamics in the latent space

# To visualize the learning processes on an illustrative problem, let’s consider a synthetic dataset consisting 
# of 10 different sequences, as well as a VAE model with a 2-dimensional latent space, and an LSTM encoder and 
# decoder.
# We visualize the resulting division of the latent space for different training steps in Figure 3, where each 
# color corresponds to the latent probabilistic representation of a sequence. We observe that:
#     The constant schedule produces heavily mixed latent codes z for different sequences throughout the entire 
# training process.
#     The monotonic schedule starts with a mixed z, but soon divides the space into a mixture of 10 cluttered 
# Gaussians in the annealing process (the division remains cluttered in the rest of training).
#     The cyclical schedule behaves similarly to the monotonic schedule in the 1st cycle. But starting from 
# the 2nd cycle, much more divided clusters are shown when learning on top of the 1st period results. However, 
# β<1 leads to some holes between different clusters. This is alleviated at the end of the 2nd cycle, as the 
# model is trained with β=1. As the process repeats, we see clearer patterns in the 4th cycle than the 2nd 
# cycle for both β<1 and β=1. It shows that more structured information is captured in z, using the cyclical 
# schedule.
# Figure 3: The process of learning probabilistic representations in the latent space for three schedules.
# The learning curves for the VAE objective (ELBO), reconstruction error, and KL term are shown in Figure 4. 
# The three schedules share very similar ELBO values. However, the cyclical schedule provides substantially 
# lower reconstruction error and higher KL divergence. Interestingly, the cyclical schedule improves the 
# performance progressively; it becomes better than the previous cycle, and there are clear periodic patterns 
# across different cycles. This suggests that the cyclical schedule allows the model to use the previously 
# learned results as a warm-restart to achieve further improvement.
# Figure 4: Comparison of terms in VAE for three schedules.
# Improving performance on NLP tasks

# The new cyclical schedule has been demonstrated to be effective in improving probabilistic representations 
# of synthetic sequences on the illustrative example, but is it beneficial in downstream real-world natural 
# language processing (NLP) applications? We tested it on three tasks:
#     Language Modeling. On the Penn Tree-Bank dataset, the cyclical schedule can provide more informative 
# language representations (measured by the improved KL term), while retaining the similar perplexity. It is 
# significantly faster than existing methods, and can be combined to improve upon them.
#     Dialog response generation. It is key to have probabilistic representations for conversational context, 
# reasoning stochastically for different but relevant responses. On the SwitchBoard dataset, the cyclical 
# schedule generates highly diverse answers that cover multiple plausible dialog acts.
#     Unsupervised Language Pre-training. On the Yelp dataset, a language VAE model is first pre-trained to 
# extract features, then a classifier is fine-tuned with different proportions of labelled data. The cyclical 
# schedule provides robust distribution-based representations of sentences, yielding strong generalization on 
# testing datasets.

# We hope to see you at NAACL-HLT this June to discuss these approaches in more detail and we’ll look forward 
# to hearing what you think!
# Acknowledgements
# This research was conducted by Chunyuan Li, Hao Fu, Xiaodong Liu, Jianfeng Gao, Asli Celikyilmaz, and 
# Lawrence Carin. Additional thanks go to Yizhe Zhang, Sungjin Lee, Dinghan Shen, and Wenlin Wang for their 
# insightful discussion. The implementation in our experiments heavily depends on three NLP applications 
# published on Github repositories; we acknowledge all the authors who made their code public, which tremendously 
# accelerates our project progress.
# The post Less pain, more gain: A simple method for VAE training with less of that KL-vanishing agony appeared 
# first on Microsoft Research.

#%%
# Adversarial Autoencoder https://blog.paperspace.com/adversarial-autoencoders-with-pytorch/


#%% [markdown]
