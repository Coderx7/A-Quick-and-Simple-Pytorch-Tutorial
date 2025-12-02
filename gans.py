#%%
# in the name of God the most compassionate the most merciful 
# GAN
# in this sction we will be learning about GANs and implement 
# some of the prominent architectures a long the way
# we start off simple with a proof of concept and then go for
# more advanced architectures and hopefully get a good idea 
# about these types of generative networks.
#
#%%
# Here we are going to create a simple GAN network. a GAN network 
# consists of a generator network and a discriminator network. 
# the generator part's job is to get a vector of some length 
# and generate an image and the discriminators job is simply
# identfying if its a real image or not (that is is it generated or not).
# the catch here is the generator will try and ultimately create 
# real life looking iamges that can fool the discriminator! 
# this means, it will learn a latent space that the real images
# belong to and simply sampling from it can result in 
# real looking images. 
# so lets see how we can do this
import sys
import os
import math
import time
import random
import copy
from datetime import datetime
from pathlib import Path
import gc
import logging
import builtins
import yaml
from tqdm import tqdm
# for types.MethodType(), to assign our custom
# functions to existing instace
import types
from functools import partial


import numpy as np 

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils.data import DataLoader

from torchvision import datasets, transforms
from torchvision import utils, models

import matplotlib.pyplot as plt
# for creating gif files for animations later on
import matplotlib.animation as animation
%matplotlib inline

#%%
# TODO: 
# an introduction to GANS and tricks here? or later after the initial exposure? 
# Before we go on, make sure you DO read this :


#%%
# now lets define our models 
# we need to implement two separate networks. one for
# the discriminator and another for our generator network.
# we follow the first paper and only use linear layers beacuse
# it was a proof of concept to show this works, we later on
# improve upon this.
# for now, all we care about is to create a simple classifier!
# which's job is to identify whether a given image is real or 
# generated!
# 
# since discriminator is very important and its gradient is used
# to steer the generators update, we need to avoid sparse gradients
# or layers/opeations that result in sparse gradients (i.e. the
# majority of elements have grdient of 0 like relu!)
class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, act=nn.LeakyReLU(0.2)):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        # since we are going to classify a single thing, whether the 
        # given image is real or not, a single class suffices!
        self.output_size = 1
        
        self.net = nn.Sequential(nn.Linear(input_size, hidden_size*4),
                                 act,
                                 nn.Dropout(0.3),
                                 nn.Linear(hidden_size*4, hidden_size*2),
                                 act,
                                 nn.Dropout(0.3),
                                 nn.Linear(hidden_size*2, hidden_size),
                                 act,
                                 nn.Dropout(0.3),
                                 # we dont use any activation functions at the end
                                 nn.Linear(hidden_size, self.output_size),
                                )

    def forward(self, x:torch.Tensor):
        output = self.net(x.flatten(start_dim=1))
        return output
        

# now the generator network. its the same as our discriminator network but 
# becasue we are trying to create images, our outputs should produce values
# that are sensible for images. 
# we can use sigmoid, but it turns out that tanh works better! so our input
# should also be scaled between -1 and 1 instead of 0 anad 1! 
# lets create our generator network!!
# our generator will recieve a latent vector z of some size and then give us 
# back a vector of the same length as input, which after reshaping we treat
# it as an image
class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, act=nn.LeakyReLU(0.2)):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        # unlike discriminator network, we can use relu 
        # for generator just fine!
        self.act = act
        
        self.net = nn.Sequential(nn.Linear(input_size, hidden_size),
                                 act,
                                 nn.Dropout(0.3),
                                 nn.Linear(hidden_size, hidden_size*2),
                                 act,
                                 nn.Dropout(0.3),
                                 nn.Linear(hidden_size*2, hidden_size*4),
                                 act,
                                 nn.Dropout(0.3),
                                 nn.Linear(hidden_size*4, output_size),
                                )

    def forward(self, input):
        # if we use sigmoid here, regardless of scaling
        # our input between -1 and 1 we will not succeed!
        # the discriminators loss decreases well but generators goes up!
        return self.net(input.flatten(1)).tanh()
# lets test them 
imgs = torch.randn(size=(5,1,28,28))
latent_vectors = torch.randn(size=(5,100))

discriminatorcnn = Discriminator(28*28, 32)
generator = Generator(100, 32, 28*28)

dis_output = discriminatorcnn(imgs)
gen_output = generator(latent_vectors)

print(f'{dis_output.shape=}')
print(f'{gen_output.shape=}')
#%%
# for the loss criterion, we should know that our discriminator's job is to 
# successfuly recognize which image is fake and which image is real!
# so we should create labels for each image. the real images will have label=1
# becasue they are real! and the fake ones are the ones generated by our
# generator network which we consider 0!
# we will use Binary CrossEntropy with Logits (BCEntropyWithlogits)
# as our criterion since only one class is involved and we have raw logist
# (had we used sigmoid on our last layer, we would have needed to use nn.BCELoss)
# There is also one minor trick which helps our training go smoothly.
# instead of label =1.0 we actually smooth out our labels, that is we set 
# labels = 0.9 instead of 1.0 so the discriminator isnt over confident
# this helps prevent our model in several ways:
# for one, it makes our model to not pay attention to obvious features
# that define a class/object, and instead work harder to find more features
# that identify the said class/object (so it works when such obvious fetures
# are missing!) second, it prevents our model to become quickly an expert 
# which would hinder our generator in catching up to it, thus it will always
# know an image is fake so generator can not learn properly and create realistic
# images.
# sidenote:
# for smoothing people usually use a range of values instead of using just 0.9
# instead of 1. usually people choose a random number between (0.7-0.9) but I
# used .9 for simplicity. we will revisit this a bit longer in future.

def real_loss(disc_output, is_smoothed=False):
    criterion = torch.nn.BCEWithLogitsLoss()
    # create labels, all one just like the output
    labels = torch.ones_like(disc_output)
    # apply smoothing if required
    labels = labels*0.9 if is_smoothed else labels
    # and finally calculate the loss   
    real_loss = criterion(disc_output, labels)
    return real_loss

def fake_loss(disc_output):
    criterion = nn.BCEWithLogitsLoss()
    labels = torch.zeros_like(disc_output)
    fake_loss = criterion(disc_output, labels)
    return fake_loss
#%%
#training. the training proceduere is like this, 
# first we train our discriminator net, on real images, 
# then get its loss, then imiediately, we generate some images
# using our generator, and feed its output to our discrinimator,
# and get an output, now, we use the fake_loss and calculate the loss, 
# then we add these two losses, create a final loss, do backprop, update the weigts 
# now before going for the next batch of images, we turn to our generatornetwork,
# its his turn now! we again generate some random vector feed our generator, 
# get an output, feed our it to our Discriminator again, and this time for loss,
# we use the real labels since we want to see, how well we are doing in creating
# real looking images! and then do backward and update on generator network and
# go for thenext batches lets see how to do this in action!

# before we go for the training we need a dataset. we use mnist for two 
# main reasons. one because the original paper used mnist and 2 because
# anything other than MNIST wouldnt work. this simple architecture with
# our primitive setup only works with very simple datasets, anything complex
# and it wont work. There are simply a lot to tune for this to work. lets
# not get ahead of ourselves for now and use mnist!
# 
batch_size = 64
num_workers = 8

#create our transformer 
transform = transforms.ToTensor()
train_dataset = datasets.MNIST('./data/MNIST',train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

# lets see a sample bacth
def display_images(imgs, cols=8, title='',unnormalize=False, save_path=None, figsize=(12,6)):
    plt.figure(figsize=figsize)
    plt.title(title)
    # images that are generated by generator will have gradients
    # because they are normal tensors we treat them as images so
    # we get rid of their gradients here because we dont need it
    imgs = imgs.cpu().detach()
    # rescale back to 0-1 range from -1/1 range
    imgs = (imgs+1)/2 if unnormalize else imgs
    # when we unnormalize, some values will be a tiny bit lower or
    # higher than [0-1] range, so we clip those values here!
    imgs = imgs.clamp(0, 1)
    # print(f'{imgs.min()=}')
    # print(f'{imgs.max()=}')
    images = utils.make_grid(imgs,nrow=cols).cpu().numpy().transpose(1,2,0) # c,h,w -> h,w,c
    
    plt.imshow(images)
    if save_path:
        os.makedirs(os.path.split(save_path)[0], exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    
imgs, labels = next(iter(train_loader))
display_images(imgs, title='sample batch from mnist', save_path='./results/misc_visualizations/test.jpg')
#%%
# discriminators
# mnist images are 28x28x1 so our flattened
# image will be 784 dimensional hence 28x28!
disc_input_size = 28*28
disc_hidden_size = 32

# generator 
# input latent vector size
gen_input_size = 100
gen_output_size = 28*28
gen_hidden_size = 32

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

# now lets create our models 
discriminatorcnn = Discriminator(disc_input_size, disc_hidden_size, act=nn.LeakyReLU(0.2))
generator = Generator(gen_input_size, gen_hidden_size, gen_output_size, act=nn.LeakyReLU(0.2) )
# 
discriminatorcnn = discriminatorcnn.to(device)
generator = generator.to(device)

# optimizers, note we are using different lr's here!
# note adam optimizer is not random choice, its one of the optimizers
# that can get us a quick convergence without much hassle, especially in GANs
# it can also lower the possibility of mode collapse to some extend
disc_optimizer = torch.optim.Adam(discriminatorcnn.parameters(), lr=0.02)
gen_optimizer = torch.optim.Adam(generator.parameters(), lr=0.002)

epochs = 100
interval=1000
# number of images to generator for evaluation during training
num_samples = 32
# we use this fixed vector as our latent vector to generator 
# to see how our network works during training.
# fixed_z = np.random.uniform(-1,1, size=(n_sample, input_dim_G))
# fixed_z = torch.from_numpy(fixed_z).float().to(device)
# rand range is 0-1, to get -1,1 we do 2*rand-1
fixed_z = 2*torch.rand(size=(num_samples, gen_input_size), device=device) -1
# or we could have used
fixed_z = torch.distributions.Uniform(-1,1).sample(sample_shape=(num_samples,gen_input_size)).to(device)

losses=[]

for epoch in range(epochs):
    # set models in train mode
    discriminatorcnn.train()
    generator.train()
    # we dont need image's real labels because we are not trying to classify 
    # mnist didgits! we want to create images and we will create our own labels
    # as we explained earlier
    for i, (real_images,_) in enumerate(train_loader):
        
        # rescale input images from [0,1) to [-1, 1)
        real_images = (real_images*2 - 1).to(device)
        # discriminator needs to classify the real image as real
        # and fake images as fake. so we need to have both of them
        real_outputs = discriminatorcnn(real_images)
        real_images_loss = real_loss(real_outputs.cpu(), is_smoothed=True)
        
        # now we generate an image using generator and classify it as fake
        latent_vectors = torch.distributions.Uniform(-1,1).sample((real_images.size(0), gen_input_size)).to(device)
        fake_images = generator(latent_vectors)
        # the discreminator must classify all generated images as fake
        fake_outputs = discriminatorcnn(fake_images)
        fake_images_loss = fake_loss(fake_outputs.cpu())
        discriminator_loss = real_images_loss + fake_images_loss
        
        # optimize discriminator first 
        disc_optimizer.zero_grad()
        discriminator_loss.backward()
        disc_optimizer.step()

        # now its time to optimize generator so that the discriminator 
        # cant distinguish between the generated images and the real ones!
        latent_vectors = torch.distributions.Uniform(-1,1).sample((real_images.size(0), gen_input_size)).to(device)
        fake_images = generator(latent_vectors)
        # since we want the discriminator to treat these as real images,
        # we treat these as though they are real images so we assign real labels!
        # we wont be updating the discriminator cuz it will mess up its detection ability
        # however we will optimize the generator so that it can update its weights
        # to more accurately reconstruct the images to fool the discriminator(lower the loss)
        generated_outputs = discriminatorcnn(fake_images)
        generated_real_loss = real_loss(generated_outputs.cpu())

        gen_optimizer.zero_grad()
        generated_real_loss.backward()
        gen_optimizer.step()

        if i%interval ==0: 
            print(f'Epoch/Epochs: {epoch}/{epochs} | iter: {i} | Discriminator Loss : {discriminator_loss.item():.4f} | Generator loss: {generated_real_loss.item():.4f} ')

    losses.append((discriminator_loss.item(), generated_real_loss.item()))
    print(f'Epoch/Epochs: {epoch}/{epochs} | Discriminator Loss : {np.mean(np.array(losses)[:,0]):.4f} | Generator loss: {np.mean(np.array(losses)[:,1]):.4f} ')
    # generate some samples from our fixed latent vector to see
    # how well we are doing during trainig
    generator.eval()
    # reshape images back to 28x28x1
    generated_images = generator(fixed_z).view(-1,*real_images.shape[1:])
    display_images(generated_images, 
                   cols=num_samples//4,
                   title=f'Generated Images at Epoch {epoch}',
                   unnormalize=True)

#%%
losses = np.array(losses)
plt.plot(losses[:,0], label='Discriminator loss')
plt.plot(losses[:,1], label='Generator loss')
plt.title('Loss')
plt.legend()
plt.show()

#%%

#%%
#DCGAN - Unsupervised representation learning With deep convolutional Generative adversarial networks - ICLR 2016
# the digits in vanilla GAN doesnt look that good, we pointed out earlier
# that this is expected, as getting good generations is more involved than
# what our simple setup could possibly offer. so now we will be covering 
# a newer architecture called DCGAN that using several tricks and a cnn
# generated a much better quality outputs that was previously imposisble
# lets see whats its about
#%%
# up until now we have been using simple fully connected layers
# and using gray-scale/black and white/single channeled images like mnist!
# as it truns out, when dealing with images, Conv layers are a much better 
# choice as they take the image features such into consideration!
# a few tips and tricks in the DCGAN paper. 
# for now lets stick to a few rules:
# 1. normalize input to -1,1
# 2. use tanh at the final layer of generator so output range is also -1,1
# 3. use leaky_relu in discriminator 
# 4. initialize the model weights according to dcgan papepr for more stable trainig
# 5. use label smoothing (soft(as against hard(not i.e 0/1)) and noisy labels)
# 6. dont use relu, maxpool in discriminator(avoid sparse gradients)
# there are more tips but for now lets keep it simple we'll cover more in a moment
#%%
# now lets create our models with what we learned just now
# 
# for we use conv-bn-act as a block instead of repeating them
# in our model setup, later on we can improve this block and 
# our architecture wouldnt need to be changed at all!
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=2, padding=1, batch_norm=False, act_func=nn.LeakyReLU(0.2)):
        super().__init__()
         # we do subsamling by using stride=2
        self.block = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size,
                                             stride, padding, bias=not batch_norm),
                                   nn.BatchNorm2d(out_channels) if batch_norm else
                                   nn.Identity(),
                                   act_func)
    def forward(self, x):
        return self.block(x)

# for upsamling we use ConvTranspose2d (previously incorrectly known as deconvolution)
# we make our generator block more powerful so it has easier time learning
# how to generate better images otherwise it will be dominated by our discriminator!
class ConvTransBlock(nn.Module):
    def __init__(self,  in_channels, out_channels, kernel_size,
                 stride=2, padding=1, batch_norm=False, act_func=nn.ReLU(inplace=True)):
        super().__init__()
        self.block = nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                             stride, padding, bias=not batch_norm),
                                   nn.BatchNorm2d(out_channels) if batch_norm else
                                   nn.Identity(),
                                   act_func if act_func else nn.Identity())
        
        # add residual connection, its not part of DCGAN
        # but since we are doing on smaller datasets I decided
        # to give it a shot just to get better output. we should 
        # be able to get results without it but its for less headache!
        # here we grab the input and upsample it 
        # so it matches the shape of our convtrans block and
        # add them together 
        self.residual = nn.Sequential(nn.Upsample(scale_factor=stride, mode='nearest'),
                                      nn.Conv2d(in_channels=in_channels, out_channels=out_channels,kernel_size=1,
                                                stride=1, bias=not batch_norm),
                                      nn.BatchNorm2d(out_channels) if batch_norm else
                                      nn.Identity(),
                                      )
 
    def forward(self, x):
        out = self.block(x)
        # residual connection like this is not part of dcgan
        # but it can imporve our result. try it out and see how
        # ita ffects the whole process and final output. 
        # uncomment the following two lines to test it:
        x_res = self.residual(x)
        # using activation functions like relu on (out+x_res) will
        # completely destroy generation! so dont apply any activations
        out = out+x_res
        return out

# we also need to initialize the weights the same way DCGAN paper did
def weights_init_dcgan(module):
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(module.weight.data, 0.0, 0.02)
        if module.bias is not None:
            nn.init.constant_(module.bias.data, 0)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0)


class DiscriminatorCNN(nn.Module):
    def __init__(self, hidden_size, act=nn.LeakyReLU(0.2)):
        super().__init__()

        self.hidden_size = hidden_size
        self.act = act
        # we do subsamling using a stride>1 instead of using maxpool layer
        # note we dont use batch_norm for the first layer of our discrimnator so 
        # the mean/std of the input is not changed
        # we use kernel=4 because it makes downsampling easier
        # our input would shrink like this
        # C1-32x32 -> C2-16x16 -> C3-8x8 -> L-4x4 -> 
        # so when it reaches our last linear layer its a 4x4 featuremap
        #
        self.net = nn.Sequential(ConvBlock(3, hidden_size, 4, 2, 1, batch_norm=False, act_func=act),
                                 ConvBlock(hidden_size, hidden_size*2, 4, 2, 1, batch_norm=True, act_func=act),
                                 ConvBlock(hidden_size*2, hidden_size*4, 4, 2, 1, batch_norm=True, act_func=act),
                                 # flatten the input to linear layer
                                 nn.Flatten(),
                                 # classifier recieves a 4x4 featuremap, and
                                 # outputs a single number needed for classifying
                                 # real or fake images. 
                                 # since its a linear layer accepting 3d featuremap
                                 # to account for all elements we multiply
                                 # the previous layer's output_channels which
                                 # is hidden_size*4 by the featuremap size (4x4) 
                                 nn.Linear(hidden_size*4 * 4*4, 1),
                                 act)
        
        # initialize weights
        # to check if our initialization is done correctly
        # lets check mean/std before after init
        # print(f'Conv mean: {self.net[2].block[0].weight.data.mean():.4f} | std: {self.net[0].block[0].weight.data.std():.4f}')
        self.apply(weights_init_dcgan)
        # print(f'Conv mean: {self.net[0].block[0].weight.data.mean():.4f} | std: {self.net[0].block[0].weight.data.std():.4f}')
        
        
    def forward(self, x):
        return self.net(x)

class GeneratorCNN(nn.Module):
    def __init__(self, z_size, hidden_size,  act=nn.ReLU()):
        super().__init__()

        self.z_size = z_size
        self.hidden_size = hidden_size
        self.act = act
        # for generation we reverse the order we had in our discriminator for ease of use
        # we dont have to do this, we are free how to upsample from the given latent vector
        # in as many layers as we want. but to keep it simple we reverse the architecture
        # we used in our discriminator here. the nagative side of this is, it constrains us
        # dramatically because for no good reason we are limiting ourseleves to this tiny
        # crude architecture while we can build much powerful architectures.
        # though since we did the same for discriminator and used a simple architecture we
        # dont go overboard with this either!        
        self.net = nn.Sequential(nn.Linear(z_size, hidden_size*4 * 4*4),
                                 nn.BatchNorm1d(hidden_size*4* 4*4),
                                 nn.ReLU(inplace=True),
                                 # unflatten the output of linear layer back to 3d
                                 # shape to be fed to convtranspos2d. we use Unflatten()
                                 # specify the dim we want to unflatten which is 1 
                                 # (cuz linear is 2d (batch, dim)) and then reshape it to
                                 # (out_channels, h,w) 
                                 nn.Unflatten(dim=1, unflattened_size=(hidden_size*4, 4, 4)),
                                 ConvTransBlock(hidden_size*4, hidden_size*2, 4, batch_norm=True, act_func=act), #8x8
                                 ConvTransBlock(hidden_size*2, hidden_size, 4, batch_norm=True, act_func=act),   #16x16
                                 # disable batchnorm for last layer of generator so 
                                 # it doesnt normalize the image values!
                                 ConvTransBlock(hidden_size, 3, 4, batch_norm=False, act_func=nn.Tanh()),              #32x32
                                 )
        
        # initialize weights
        self.apply(weights_init_dcgan)

        
    def forward(self, x): 
        # we could have also done it in functional form for that we had to
        # make the fc stand alone and the reshape its output to be 3d
        # x = self.fc(x)
        # and unflatten the output back to 3d shape to be fed to convtranspos2d
        # we use the in_channels of the immediate first layer after linear layer
        # and multiply it by the featuremap size which is 4x4
        # print the generatorcnn model and you'll see how we accessed each layer
        # like this 
        # x = x.view(-1, self.net[0].block[0].in_channels, 4, 4)
        return self.net(x)

x = torch.randn((5,3,32,32))
z = torch.randn((5,100))
discriminatorcnn = DiscriminatorCNN(16)
generatorcnn = GeneratorCNN(100, 16)
# print(f'{generatorcnn}')
doutput = discriminatorcnn(x)
goutput = generatorcnn(z)
print(f'{doutput.shape=}')
print(f'{goutput.shape=}') 
#%%
# losses follow what we already attempted with the exception that we 
# now smooth it a bit better.
def real_loss(preds_real, smooth=True, strict_DCGAN=False, device='cuda'):
    criterion = nn.BCEWithLogitsLoss()
    # create labels of all ones to refer to real images
    labels = torch.ones_like(preds_real, device=device)
    if not strict_DCGAN:
        # smooth them if required
        # note: 
        # DCGAN itself used real labels and since they had large datasets
        # it worked pretty well. however others such as Salimans in 
        # Improved Techniques for Training GANs, 2016 offered smoothing tips
        # like this. it used 0.9 instead of 1 and it worked reall well.
        # other variations also were introduced like sampling from 0.7-0.9
        # to make it harder for discriminator to overfit and force it to learn
        # more robust features in practice however, I found .9 to work much better!
        # labels = labels * torch.distributions.Uniform(0.7,0.9).sample() if smooth else labels
        labels = labels * 0.9 if smooth else labels
        
    return criterion(preds_real, labels)

def fake_loss(preds_fake, smooth=False, strict_DCGAN=False, device='cuda'):
    criterion = nn.BCEWithLogitsLoss()
    labels = torch.zeros_like(preds_fake, device=device)
    if not strict_DCGAN:
        # smooth the fake labels with a random number in range (0,0.3),
        # this is infact noise injection essentially!
        # but random smoothing like this for fakes amy destabilize training
        # im implementing it for experimentation but in practice we dont use it
        labels = torch.ones_like(preds_fake, device=device) * torch.distributions.Uniform(0,0.3).sample() if smooth\
        else labels
    return criterion(preds_fake, labels)

#%% 
#%%
# for the dataset this time around we will be working with SVHN dataset, 
# short for Stree View Home Number this is a dataset for post number images!
# the images are 32x32x3 and it has 100k images.
# the original paper of DCGAN used a much larger dataset called LSUN with
# 3 million images of bedrooms. they also tested on 315K images of faces and
# on imagenet dataset which has 1.2 million images. LSUN dataset is more than
# 47GB so instead we go for svhn which in total has 600k images and is much 
# smaller.(it has 3 parts, by default the trainig is 73k, validation is 26k and 
# extra is 530k which is 1.3GB only and is what we are going to use).
# we want a dataset that is large enough to give us a decent result. the more data
# the better. there are other datasets out there like celeba, but svhn is both
# small and has plenty of images so yeah!
#
# sidenote: 
# if you still want to experiment with LSUN, you need to find it in kaggle.com
# or the lmdb versions other people put up later on because the princeton university
# that hosted the dataset no longer offers any download links.
# 
#
def get_dataloader(dataset_name="SVHN", split=None, resize_dims=(32,32), batch_size=128, num_workers=8, store_path="./data/"):
    dataset_name = dataset_name.lower()
    if dataset_name == 'svhn':
        split = 'extra' if not split else split
        transform = transforms.Compose([
        # transforms.RandomHorizontalFlip(),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor()])
        dataset = datasets.SVHN(os.path.join(store_path, dataset_name.upper()), split=split, transform=transform, download=True)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    
    elif dataset_name == 'celeba':
        split = 'train' if not split else split
        transform = transforms.Compose([transforms.Resize(resize_dims),transforms.ToTensor()])
        dataset = datasets.CelebA(os.path.join(store_path), split=split, transform=transform, download=True)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    
    else:
        raise ValueError(f"'{dataset_name}' is not a valid dataset name!")

    return data_loader


dataset_name = 'svhn'
train_loader = get_dataloader(dataset_name=dataset_name)
#visualize 
(imgs, labels) = next(iter(train_loader))
display_images(imgs, title=f'{dataset_name} samples',cols=16)

# we need to check the minimum and maximum 
# values of each pixel so we can scale them
# between -1and 1 
print(f'default min: {imgs.min()}')
print(f'default max: {imgs.max()}')
# how to do that?
# one way for scalig between -1 and 1 is to multiply
# our input by (max-min) and then + min which in our
# case are +1 and -1 so 
# imgs = imgs*(1-(-1))+ (-1)
# which if you remember is exactly like before (x*2 - 1!)  
imgs = imgs*2-1
# now lets see one example!
print(f'scaled min: {imgs.min()}')
print(f'scaled max:  {imgs.max()}')
print(f'{len(train_loader)=}')
#%% training!
# svhn or celeba
dataset_name='svhn'
# a larger batchsize provides more stablity
# decrease it and see the impact
batch_size = 128
train_loader = get_dataloader(dataset_name=dataset_name, batch_size=batch_size)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


disc_hidden_size = 32#16
gen_hidden_size = 64
# dcgan used 100 if I recall correctly
z_size = 100

epochs = 50 
interval = len(train_loader)//2

#discriminator
discriminatorcnn = DiscriminatorCNN(hidden_size=disc_hidden_size)
discriminatorcnn = discriminatorcnn.to(device)
#generator
generatorcnn = GeneratorCNN(z_size, hidden_size=gen_hidden_size)
generatorcnn = generatorcnn.to(device)
# DCGAN used [0.5,0.999] for betas for adams for better training stability
# we lower the lr for disciminator so it doesnt learn too fast!
disc_optimizer = torch.optim.Adam(discriminatorcnn.parameters(), 0.0001, [0.5, 0.999])
gen_optimizer = torch.optim.Adam(generatorcnn.parameters(), 0.0002, [0.5, 0.999])

gen_num_samples = 64
fixed_z = torch.distributions.Uniform(-1,1).sample((gen_num_samples,z_size)).to(device)

experiment_date = datetime.now().strftime("%Y%m%d%H%M%S")
losses = []

for epoch in range(epochs):

    discriminatorcnn.train()
    generatorcnn.train()

    for i, (imgs_real, _) in enumerate(train_loader):

        #scale input to [-1,1]
        imgs_real = (2*imgs_real-1).to(device)
        
        # before we go on lets add small gaussian noise to real images
        # I added this later when I noticed heavy mode collapse happining
        # we do this to both real and fake images to fight mode collapse
        # this is not needed in dcgan but having it enabled helps with 
        # training/generation we'll talk about this in future I'll leave this here
        # for now - see training remarks ahead for an intersting find!
        imgs_real += 0.05 * torch.randn_like(imgs_real)
               
        # train discriminator! 
        # real image predictions
        preds_real = discriminatorcnn(imgs_real)
        disc_real_loss = real_loss(preds_real, smooth=True, device=device)
        
        # generate an image using generator 
        z_vector = torch.distributions.Uniform(-1,1).sample((imgs_real.size(0), z_size)).to(device)
        # we detach the imgs_fake so the discriminator cant use the gradients
        # from the generator and quickly learn!
        imgs_fake = generatorcnn(z_vector).detach()
        
        # add noise to fake images as well(not needed for dcgan)
        imgs_fake += 0.05 * torch.randn_like(imgs_fake)
        
        preds_fake = discriminatorcnn(imgs_fake)
        disc_fake_loss = fake_loss(preds_fake, smooth=False, device=device)
        # calculate discrimiator loss out of real and fake losses
        disc_loss = disc_real_loss + disc_fake_loss
        
        # for debugging purposes
        # if disc_real_mean is a lot larger than disc_fake_mean (e.g. 2.0 vs -2.0) 
        # then it means our discriminator is strong but if both are near the same
        # value and the loss is low then it means our discriminator is confused
        # or is over-regularized.
        disc_real_mean = preds_real.mean().item()
        disc_fake_mean = preds_fake.mean().item()
        
        # and optimize discrimnator 
        disc_optimizer.zero_grad()
        disc_loss.backward()
        disc_optimizer.step()

        # now train genertor to create images that look real
        z_vector = torch.distributions.Uniform(-1,1).sample((imgs_real.size(0),z_size)).to(device)
        fake_imgs = generatorcnn(z_vector)
        preds_fake = discriminatorcnn(fake_imgs)
        
        # swap loss! treat fake images as real images
        gen_real_loss = real_loss(preds_fake, smooth=False, device=device)
        # 
        # we are seeing mode collapse it might be due to discriminator is
        # doing great too early so lets mess it up!
        # so occasionally (around 5% of the times) flip labels 
        # if torch.rand(1).item() < 0.05:
        #     gen_real_loss = fake_loss(preds_fake, device=device)
        # else:
        #     gen_real_loss = real_loss(preds_fake, smooth=False, device=device)
            
        # optimize generator
        gen_optimizer.zero_grad()
        gen_real_loss.backward()
        gen_optimizer.step()

        if i+1%interval==0:
            # append discriminator loss and generator loss
            losses.append((disc_loss.item(), gen_real_loss.item()))
            # print discriminator and generator loss
            print(f'Epoch/Epochs: {epoch}/{epochs} | Iter: {i}/{len(train_loader)} | Disc Loss: {disc_loss:6.4f} | Gen Loss: {gen_real_loss:6.4f}')

    losses.append((disc_loss.item(), gen_real_loss.item()))
    
    d_loss_mean = np.mean(np.array(losses)[:,0])
    g_loss_mean = np.mean(np.array(losses)[:,1])
    
    print(f'Epoch/Epochs: {epoch}/{epochs} | Disc Loss : {d_loss_mean:.4f} | Gen loss: {g_loss_mean:.4f} ')
    print(f" -- Discriminator's real mean: {disc_real_mean:.4f} | Discriminator's fake mean = {disc_fake_mean:.4f}")
    
    #save model weights at each epoch
    torch.save({"state_dict":generatorcnn.state_dict(),
                "hidden_size":gen_hidden_size,
                "z_size":z_size,
                "epoch":epoch,
                "losses":losses,
                "dataset_name":dataset_name,
                }, f"./weights/gan/dcgan_generatorcnn_{experiment_date}.pt")
    
    # generate some images mid training to evaluate our model's performance 
    with torch.no_grad():
        generatorcnn.eval()
        # reshape images back to 32x32x3
        generated_images = generatorcnn(fixed_z).view(-1,*imgs_real.shape[1:])
        display_images(generated_images, 
                    cols=gen_num_samples//8,
                    title=f'Generated Images at Epoch {epoch}',
                    unnormalize=True,
                    save_path=f'./results/gan/dcgan/{experiment_date}/epoch_{epoch}.jpg')
    
#%%
losses = np.array(losses)

plt.plot(losses[:,0],label="Discriminator's loss")
plt.plot(losses[:,1],label="Generator's loss")
plt.title('Loss')
plt.legend()
plt.show()

#%%
# load a checkpoint and lets run some experiments on latent space
states = torch.load('./weights/gan/dcgan_generatorcnn_20250812172504.pt',map_location="cpu", weights_only=False)
z_size = states["z_size"]
hidden_size = states["hidden_size"]
dataset_name = states["dataset_name"]
generatorcnn = GeneratorCNN(z_size, hidden_size)
generatorcnn.load_state_dict(states["state_dict"])
generatorcnn.eval()
print(f"Generator's weights for {dataset_name.upper()} loaded!")
# 
#%%
# remarks:
# ok early on we faced high generator's loss and mode collapse
# then we added data-aumentation and noise to images and beafed up
# our generator and now we noticed a dramatic decrease in generators
# loss but at the same time the discriminator's loss stayed constant 
# and stopped improving!
# obviously looking at the images we can say 100% thats not a sign of
# great training! just the contrary it seems.
# it seems our generator is producing outputs the discriminator finds
# easily as real, that is our discriminator is no longer giving useful
# feedback. so two things might be happening here:
# its either our discriminator that has collapsed as its loss is stuck 
# at a low number (~0.32 - 0.36) meaning its confident but probably wrong!
# its no longer pushing the generator toward generating realistic images
# meanwhile our generator is overfitting to a few patterns that the
# discriminator is consistently misclassifying as real!
# or it may be that we have too much regularization/augmentation!
# the heavy regularization is interfering with our discriminator's decision
# boundary in the latent space, and because of that, the generator 
# found an easy way around and kept at it without bothering with more
# complex samples/situations!
# # increased discrimnators lr to 2e-4 fro 1e-4 :
#      -- no visible change!
# # disabled the data-augmentation - increased discrimnators lr to 2e-4 fro 1e-4: 
#      -- no luck
# # disabled the data-augmentation - increased discrimnators lr to 2e-4 fro 2e-3: 
# #    -- generator loss stays at around 2.7 now but a lot of similar/repeated patterns 
# #    -- are being generated! discriminator loss is around 0.396 at epoch 20!
# #    -- but the gans loss is not decreasing, its stuck at 2.8x and fluctuattes aorund that
# # the previous config + now we initialized the weights according to dcgan:
# #    -- checking the disc_real_mean and disc_fake_mean shows they are far away
# #    -- signaling our discrimnator is strong! to be more exact we have 
#      -- Epoch/Epochs: 0/50 | Disc Loss : 0.4475 | Gen loss: 2.2510 
#           -- Discriminator's real mean: 3.1040 | Discriminator's fake mean = -2.7766
#      -- 3.1 and -2.7 are our discriminator raw logits (before sigmoid, remember
#      -- we have BCE as loss it applies the sigmoid internally and then it will 
#      -- affect the gradient (the gradient will be very small as you can imagine))
#      -- sigmoid(3)=~0.95 which means our discriminator is predicting around 95% 
#      -- real images as real! and sigmoid(-2.7)=0.06 meaning discriminator is 
#      -- predicting ~94.3% fake for generated images (and up to epoch 5 it becomes -5 
#      -- which makes it 99%). this means our generator is getting very small gradients 
#      -- because the discriminator's outputs are far into the saturated regions of
#      -- the sigmoid, where slope goes toward 0 and also we are not using the batchnorm
#      -- layer at the final layer of generator either because that would cause another issue
#      -- thats why by epoch 5 our generator loss is still around 3~4! the discriminator 
#      is winning too easily, starving the generator of signal.
#      -- so Im using stronger label smoothing (using 0.7-0.9 instead of 0.9 for real images)
#      -- that didnt solve the issue, reverted the smoothing back to 0.9 and instead
#      -- decreased discrimnator's capacity (hiddensize down to 16 instead of 32) and
#      -- that alone didnt do anythig, so I decreased the discriminators lr down to 0.0001
#      -- from 0.002. for the initial epochs, the mean became much lower (0.4 vs 2 / -0.2 vs -2.9) 
#      -- but as the training progrssed, they fluctuated to larger values like 1.7, 3,-2 etc
#      -- both discriminator and generator loss are different. discriminator loss is higher
#      -- compared to before (0.8 vs 0.3) and generators loss is larger (0.6 vs 1.4)
#      -- as the traiing progressed, the discriminators loss gets better (0.65) while
#      -- the generator's loss gets worse (1.76 epoch 24) 
#      -- so I increased the label swap rate from 5% to 10% and imediately noticed
#      -- the discriminator and generators loss stay roughly the same (0.986 vs 1~1.2)
#      -- the real and fake means for discriminator is also around 1 for both but
#      -- as training progressed, they took different values (2.42 for real and -1.5 for fake)
#      -- discrimnator's loss very slowly decreases and generator's increases
#      -- but the rate of change is very small 
#      -- disabling label swap for generator 
#      -- OK! I made a ridiculous mistake, when creating ConvTranspose, I missed
#      -- the nonlinearity! so we were dealing with a linear generator and that 
#      -- was the reason why we were having this much issues! after fixing that
#      -- and also disabling label swap for generator its doing way better! the rea;
#      -- and fake means are also now in a much better place (0.19 / -0.19) which 
#      -- signals the discriminator is struggling and it shows from its loss thats 
#      -- been stuck at 1.37 while generators loss has been stuck around 0.79
#      -- so we are going to increase discriminators capacity (even with these conditions)
#      -- we are seeing way better generations. very diverse outputs but low quality
#      -- unline before that we faced mode collapse all the time (many repeated patterns)
#      -- increased discriminator hidden_size from 16 to 32 (lr is still 1e-4) and right away image quality
#      -- got a lot better. the discriminators loss decreased to 1.28 from 1.37
#      --  and real and fake images mean are mostly in a good range (0.1 and -0.2) 
#      -- all showing good gradient flow.im satisfied with the result 
#      -- also note that we used residual connection in generator. without it
#      -- we can still get result but obviously residuals help well. 
#      -- I test both so you see the difference (having residual makes sharper 
#      -- and better well formed images)
#      -- I also disabled gaussian noise addition trick because its not needed for dcgan
#      -- I ran a test without it so we know the model would still be able to perform
#      -- without it, though having it enabled is good fortraining stability
#      -- (sidenote: I noticed when I disabled gausian noise trick, the generation
#      -- looked funny, like some adjacent images, would eerily look as if they are
#      -- connected as if they form one number. its intersting, you can see these images
#      -- in results/gan/dcgan/20250811144424 directory you can compare it against
#      -- the /20250811135652 directory that uses gausian noise and this doesnt happen
#      -- could be a coincidence but could also be due to added randomness that helps with
#      -- more diversity of samples in anycase its really intersting result)
#      -- update:
#      -- I did a bit more research and it seems our initial assesment was spot on. 
#      -- basically the effect we saw when we didnt use noise (adjacent images look as
#      -- if they form part of a bigger image) is a sign of latent space entanglement
#      -- that is caused by overfitting to a few discriminator weaknesses. 
#      -- meaning our generator probably overfitted to a small set of latent-output mappings and
#      -- the discriminator wasnt forcing variety).
#      -- 
#      quicknote:
#      -- (latent space entanglement simply means in our generator's latent space, a single latent
#      -- dimension controls many unrelated things at once! instead of just neatly controlling one
#      -- interpretable feature/elemnt of variation)
#      -- when we train a gan (or a vae or any generator model for that matter) the generator learns a
#      -- mapping from z -> image but by default nothing in our training procedure is forcing each element of z to 
#      -- correspond to a single disentangled property like for example hair color/length or smile intensity!
#      -- instead each latent dimension usually affects multiple features simultaneously because
#      -- first of all the generator is just trying to produce realistic outputs not to align dimensions with 
#      -- our human interpretable factors and second because the underlying data distribution itself can 
#      -- have correlations between factors (e.g. in our dataset beard might correlate with male and
#      -- short hair), so the model learns those entangled relationships aswell.
#      -- (during interpolations we'll see this when for example if we change only 
#      -- z7 (7th element in our z vector which we feed to our generator) we might expect just 
#      -- one property in the image (e.g. rotate head) to change but in practice we see several
#      -- things change altogether! (e.g. hair color the background the face, etc changes)
#      -- this is entanglement, one element/coordinate is tangled up with many factors.
#      -- 
#      -- in other words, the gaussian noise here acts like a regularizer and forces 
#      -- the generator to explore more of the distribution and in doing so it learns
#      -- more robust features/patterns so it makes adjacent samples less likely to 
#      -- be visually related. however without it the generator overfits to some simple/safe features 
#      -- that allows it to act real while discriminator is fine with it and doesnt force it
#      -- to change and improve its result (since for what discriminator is concerned, it just
#      -- cares about if it looks real or not, if it keeps repeating a simple yet realistic sample
#      -- it would be fine for the discriminator! but oviously not us! we want diversity!)
#      -- if you think about it it makes complete sense 
#      -- the discriminator sees crystal-clear images so it can and does find very sharp and easily 
#      -- deterministic features that separate real/fake especially early in training.
#      -- in turn, the generator goes and learns about a few safe patterns that consistently
#      -- pass as real and keeps using them! 
#      -- since our latent space interpolation isnt being pushed toward diverse modes,
#      -- small changes in the latent vector can produce outputs that look connected 
#      -- rather than independent! almost like they are tiles of a bigger texture!
#      -- (neighboring seeds that look too similar is a also symptom of poor mode coverage)
#      -- now with gaussian noise added, each real/fake image fed to discriminator has
#      -- a small random change/purterbation. this will force the discriminator to learn more 
#      -- robust/generalizable features instead of pixel perfect cues (since now both real and fake
#      -- iamges have noise they have in common so to separate between them it needs better features!).
#      -- at the same time the generator can no longer rely on one/few brittle patterns that easily fail, 
#      -- it must spread across more modes to fool discrimnitaor consistently.
#      -- this added randomness/stochasticity in the training signal helps to break up those connected
#      -- patterns in the latent space and ultimately lead to a more independent/varied generation.
#      -- 
#
#%%
# one interesting thing the authors of dcgan did was they showed the 
# latent space learned using GAN learned disentangled features which 
# we could for example just like wordembeddings, do some arithmetic
# operations and get meaningful outcomes. 
# for examples, if we subtracted the latent vectors for man with glasses
# from man without glasses and then add woman without glasses we would get
# woman with glasses! 
# in conditional version, labels are used to control the attributes, but we 
# should be able to play with attributes aswell though it maynot be much,
# in unconditional version where no labels are used, like ours here
# we need to comeup with a way to get images with the same attribute first 
# and then use those latent vectors to do the arthimetics.
# note we said images and not just a single image, because we need to average
# those latent vectors to get rid of their specific nuacenses and only capture the 
# essense of said attribute(i.e. the direction of change if you will), 
# otherwise it wouldnt work properly as not all features are disentangled prefectly.
# so we need a way to identify the existence of an attribute in an image,
# one way would be to use a classifier on our generator's outputs and save the
# z vectors that resulted in a specific attribute e.g. smile, and do this for 
# other attributes like having galsses, being bald, etc.
# since we dont have a specific classifier for stuff like this, we can also start 
# generating randomly and grab the vectors that result in certain attributes. 
# though we could also do it manually. if you recall in autoencoders section
# we had a similar problem back then and we saw we could use std/mean to specify
# the direction of change. we can do the same here and by playing with different
# std/mean find our vectors and use them in our experiment.
# to make it more managble we use opencv's picker so it becomes easy to change
# and see the outcomes.
#update:
# I gave it a try but not only finding proper attributes this way is hard but also ineffcient
# what we are doing here doesnt work properly simply because we are just biasing the 
# distribution we are sampling from which only changes style/noise intensity but not the semantic 
# attributes themselves, in order to get varied attributes ,we need large batchsizes, 
# and then play with the mean/std to get what we like and then pick the id of the said
# attribute manually and form a batch of that attribute from corrosponding latent vector
# and so on, its really not efficient! unlike our vae case, the mean/stds are not that 
# well defined so that by managing them we access different attributes. 
# so I gave up on the idea in part because we have other more streamlined ways to do this, 
# I have also seen (and experimented) with some newer approaches which I didnt get tge 
# desired result either because of our simplistic architecture and features not being properly
# developed I believe.
# the methods I tested include Sefa by Shen&Zhou 2020(https://arxiv.org/abs/2007.06600) and 
# clip where we use the clip model to identify latents that have the attributes we want by
# comparing the text embeddings of the attributes and their image embeddings, and grabing the
# image embeddings that have the highest similarity of the text embeddings of the attribute
# we are after (e.e. "person with eyeglasses").
# neigther of these methods worked well for me most probably because our model itself
# is not only not powerful by any means, the learned features are not that high quality(disentangled properly)
# and also the image size is really small compared to what clip vision backend for example requires
# and resizing it to 224 would introduce a lot of artifacts which would decrease the effectiveness 
# even more!
# for the record, the sefa was trained on stylegan2 architecture if I recall correctly, which
# is one of the best GAN architectures even today and features are very well disenangled compared
# to our simplistic model here. the paper shows it can work pretty well on stylegan see https://genforce.github.io/sefa/
# there are other papers as well but I didnt bother implementing them as it would be in vein
# we might come back to this when we cover stylegan papers, but for now we wont go any deeper!
#
# sidenote:
# concerning how sefa method works, basically this method sugggests we use 
# eigen decomposition of our generator's weights(linear layer weight) to get meaningful 
# directions without any data or labels and then use those to create the attributes we want.
# (why the linear layer? because its the layer that does the mapping!)
# its basically doing torch.linalg.eigh(W.T @ W) ,W being our linear layer's weights
# and then picking top k eigenvectors as attribute directions.
# to refresh your memories, eigenvectors show directions towards maximum variation (like pca),
# (informally speaking, variation means information(different values/ differences which itself conveys information to us),
# when we say an area has maximum variation it means datapoints there are very spread out 
# (imagine a chart and a line, if datapoints are not spreadout along the line, it means they
# are cramped into one thing, many datapoints are essentially duplicates or very close to eachother
# so when we say variations are spread out along a direcion we really mean datapoints being spread
# on that direction, so if we capture that area we basically have captured a lot of variation/information/datapoints
# there. this is how pca does its thing as well! hope this has made it obvious now!))
# so the top eigenvectors in our generator are the directions in the latent space
# that affect the output by a lot/strongly. (how much exactly? its specified by the eigenvalues!)
# these directions most of the time, corrolate with/corrospond to different attributes (liek pose, color, eyeglasses, hair,etc)
# so much so if we move along said directions, we can see different attributes change in the input.
# as we just said, the eigenvectors are (semantic) directions, and the eigenvalues are 
# their values/magnitude, they show us how strong they are, i.e. larger values mean more 
# impact on the output and viceversa.
# sidenote2:
# W.t() @ W is known as a gram matrix so if you faced this term it refers to this operation
# also as to why we are doing this in first place? so we can see the effect of each input 
# feature on all other input features. its basically done to capture correlations/co-variances
# of input features after they are passed through the linear layer of our generator.
# (each column in W corresponds to how one input dimension (i.e. latent feature) contributes 
# to all outputs. so gram matrix is essentially measuring how similar input features i and j are,
# in terms of their effect on the output).
# (we can think of this operation (gram matrix) as a way to summarize how the generator's 
# weights couple latent dimensions together so by factorizing it, we can uncover latent 
# axes/drections of variation that our GAN has learned!)
# see my implementation below (after opencv example)
# update2:
# I ended up writing a quick classifier for celeba to do this. see the section after 
# conditional version
#%%
#%%
# playing with attributes by changing std/mean 
# tldr this is hard this way see the next approach below
np.random.seed(66)
# update from future:
# this uses the default random generator and is wrong!
# see my update about random generator in stylegan2 experiment section at the end
# todo: remove this and use a separate private generator
torch.manual_seed(66)

@torch.no_grad()
def interpolate_latents(generator, z1, z2, steps=8, eps=1e-8, **kwargs):
    generator.eval()
    alphas = torch.linspace(0, 1, steps)
    grids = []
    for a in alphas:
        z_interp = (1-a)*z1 + a*z2 + eps
        imgs = generator(z_interp,**kwargs)
        imgs_grid = torch.cat((*imgs,), dim=1)
        # print(f'{imgs_grid.shape=}')
        grids.append(imgs_grid)
    return torch.cat((*grids,), dim=2)

def show_images(imgs, title, cols=8, figsize=(4,3)):
    if imgs.max()>1 or imgs.min()<0:
        imgs = ((imgs+1)/2).clamp(0,1)
        
    if imgs.ndim==4:
        imgs = utils.make_grid(imgs,nrow=cols)
        
    imgs = imgs.permute(1, 2, 0).cpu().numpy()
    plt.figure(figsize=figsize)
    plt.imshow(imgs)
    plt.axis("off")
    plt.title(title)
    plt.show()
#%%
# z1 = 2*torch.rand(generatorcnn.z_size)-1
# z2 = 2*torch.rand(generatorcnn.z_size)-1
steps = 8
num_samples=5
std = 0.6
mean = 0.00001
z1 = std * torch.randn(size=(num_samples, generatorcnn.z_size),generator=random_gen) + mean
z2 = std * torch.randn(size=(num_samples, generatorcnn.z_size),generator=random_gen) + mean

imgs = interpolate_latents(generatorcnn, z1, z2, steps=steps,eps=1e-8)
title =  " ".join([f"{a:.2f}" for a in torch.linspace(0, 1, steps)])
show_images(imgs, f'Latent Arithmetic: alphas {title}',figsize=(12,6))
#%%
import cv2
# define our simple gui 
def onchange(x):
    pass

@torch.no_grad()
def choose_meanstd(generator, num_samples,random_gen, ncols=8,**kwargs):
    generator.eval()
    
    cv2.namedWindow('std_mu_finder')
    
    # create trackbars for std and mean 
    # opencv trackbar only supports ints, 
    # so we specify our desired range as int
    # and then in code divide them to get fractions
    # std: 0.0 - 1.0
    # mean: 0.0 - 1
    cv2.createTrackbar('std', 'std_mu_finder', 500, 1000, onchange)
    cv2.createTrackbar('mean', 'std_mu_finder', 0, 1000, onchange)
    # to be able to sample new values we use this
    cv2.createTrackbar('resample', 'std_mu_finder', 0, 1, onchange)
    
    z = torch.randn(size=(num_samples, generator.z_size), generator=random_gen)
    old_std, old_mean, old_minus,old_resample=None,None,None,None
    imgs=None
        
    while True:
        std = cv2.getTrackbarPos('std','std_mu_finder',)
        mean = cv2.getTrackbarPos('mean','std_mu_finder')
        resample = cv2.getTrackbarPos('resample','std_mu_finder')
        
        frac_std = std/1000
        frac_mean = mean/1000
        
        if old_resample != resample:
            z = torch.randn(size=(num_samples, generator.z_size),generator=random_gen)
            print(f'New z sampled!')
            
        # only generate when values change so 
        # we dont waste too much cpu and hug the system!
        if old_std!=std or old_mean!=mean or old_resample!=resample:
            new_z = frac_std * z + frac_mean
            # update from future: added kwargs so
            # this can be used for future gans as
            # well which accept more arguments
            imgs = generator(new_z, **kwargqs)
                       
            img = utils.make_grid(imgs, nrow=ncols, normalize=True, value_range=(-1, 1))
            img = (img.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = cv2.resize(img, dsize=None, fx=2.0, fy=2.0)
            
            old_std, old_mean, old_resample = std, mean, resample
            print(f'Generated using std:{frac_std:5f} mu:{frac_mean:.5f}')
            
        if img is not None:
            cv2.imshow('std_mu_finder', img)
            
        # break loop when 'q' is pressed
        # also note we dont need to check this every 1 ms!
        # waiting every 30ms/90ms suffices, I chose 60ms
        if cv2.waitKey(60) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    return z, frac_std, frac_mean

seed=1
np.random.seed(seed)
random_gen = torch.manual_seed(seed)
steps = 8
num_samples=5
generatorcnn.to('cpu')
# simply grabing an attribute like women may not be easy
# for finetuned example, we need to use large batch and inidivually
# get the index to images that contain a specific attribute we like (like glasses)
# and then use those indexes to form a batch of latent vectors and feed it to
# our latent arithmetic function, or use a small batch that the majority have
# an attribute (like women)
#std/mu:220/200 -> women
z,std,mean = choose_meanstd(generatorcnn, num_samples=num_samples,random_gen=random_gen, ncols=8)
z_attr1 = z*std+mean
z_attr1 = z_attr1[:4]
#%%
# male!
seed=66
np.random.seed(seed)
random_gen = torch.manual_seed(seed)
num_samples=10
z2,std2,mean2 = choose_meanstd(generatorcnn, num_samples=num_samples,random_gen=random_gen, ncols=8)
z_attr2_all = z2*std2+mean2
ids = [1,2,3,5]
z_attr2 = z_attr2_all[ids]

#%%
@torch.no_grad()
def latent_arithmetic_unconditional(generator, z_with_attr, z_without_attr, z_base, alpha_values,ncols=8,**kwargs):
    # if its a single example, add batch dim
    if z_base.ndim==1:
        z_base.unsqueeze_(0)
    # getting the actual attribute (direction)
    direction = z_with_attr.mean(dim=0) - z_without_attr.mean(dim=0)
    # we normalize the vector so it only encodes the direction and not magnitudes,
    # this way all attribute directions will have unit length and will be comparable.
    # we can then use this fact and control the effect's strength(our desired direction/concept)
    # using a single number like alpha, like for example "move +2 in the smiling direction"
    # or "move -1.5 in the glasses direction" and because all directions are normalized
    # alpha will have a uniform meaning across attributes!
    direction = direction/direction.norm()
    print(f'{direction.shape=}')
    results = []
    # calculate new z based on new direction + add a bit of variety using alpha
    # to see other variations
    for alpha in alpha_values:
        # note: we must multiply alpha by direction not add them!
        # alpha * direction means "take alpha steps along this attribute axis"!
        # like our previous example "move +2 in the smiling direction" now
        # if we just add alpha, it means moving alpha steps in 
        # all directions at once, equally, which has no semantic
        # meaning its just a uniform shift of the latent vector.
        # so to make it exclusive for a specific attribute/direction
        # we only scale that direction by multiplying it exclusively
        z_new = z_base + (alpha * direction)
        # print(f'{z_new.shape=}')
        # update from future:
        # later on our generators take more parameters! so we pass future possible
        # arguments this way
        imgs = generator(z_new,**kwargs).cpu()
        imgs_grid = utils.make_grid(imgs,nrow=ncols)
        results.append(imgs_grid)
    return torch.stack(results)
#%%
seed = 10
np.random.seed(seed)
random_gen = torch.manual_seed(seed)
z_base = torch.randn(size=(z_attr1.size(0),generatorcnn.z_size),generator=random_gen)
print(f'{z_base.shape=}')
alphas = torch.linspace(-3,5,steps=24)
imgs = latent_arithmetic_unconditional(generatorcnn, z_attr1, z_attr2, z_base[1].unsqueeze(0), alpha_values=alphas)
show_images(imgs, f'latent arithmatic using manually selected attributes',figsize=(12,6))
# in order to get more accurate results we need more samples from each attribute
# but I guess this suffices for now. we'll see a much better example below (see after conditional version)
#%%
@torch.no_grad()
def sefa_linear_eigenvectors(generator:GeneratorCNN, topk=10):
    # first linear layer that maps z
    W = generator.net[0].weight.data
    gram_matrix = W.t() @ W
    eigen_vals, eigen_vecs = torch.linalg.eigh(gram_matrix)
    # by default the eigenvalues/vectors are orderer in an ascending manner
    # from smallest to the largest, however we flip the order to descending
    # so the strongest eigenvectors/directions come first and we then easily
    # do topk
    # print(f'{eigen_vals[0:3]}=')
    # print(f'{eigen_vals[-3:]}=')
    eigen_vals = eigen_vals.flip(0)
    eigen_vecs = eigen_vecs.flip(1)
    # print(f'after flipping:')
    # print(f'{eigen_vals[0:3]}=')
    # print(f'{eigen_vals[-3:]}=')
    # print(f'{eigen_vals.shape=}')#n
    # print(f'{eigen_vecs.shape=}')#nxn
    return eigen_vals[:topk], eigen_vecs[:, :topk]

@torch.no_grad()
def traverse(generator, z, eigen_vec, alphas):
    outs = []
    for a in alphas:
        z_mod = z + a * eigen_vec.unsqueeze(0)
        imgs = generator(z_mod)
        outs.append(imgs)
    return torch.cat(outs, dim=0)

eigen_vals, eigne_vectors = sefa_linear_eigenvectors(generatorcnn,topk=10)
# print(f'{eigne_vectors.shape=}') #(n,topk)
topk = eigne_vectors.size(1)
z = torch.randn(size=(8,generatorcnn.z_size),generator=random_gen)
# the original sefa was used on stylegan and these alphas work well
# however on our model they are simply too much so we use smaller ones
# even with that we dont see any meaningful change!
# alphas=(-3.0,-2.0,-1.0,0.0,1.0,2.0,3.0)
alphas=(-0.7,-0.3,-0.1,0,0.1,0.3,0.7)
for i in range(topk):
    print(f'trying direction {i}/{topk}:')
    eigen_vec = eigne_vectors[:,i]
    imgs = traverse(generatorcnn, z, eigen_vec, alphas)
    show_images(imgs,f'direction {i}/{topk}',figsize=(12,6))
   
#%%
# clip experiment
import math
import clip  # pip install ftfy regex tqdm && pip install git+https://github.com/openai/CLIP.git

@torch.no_grad()
def clip_normalize_image(imgs, size = 224) :
    imgs = (imgs+1.0)*0.5# or (imgs+1)/2
    device = imgs.device
    # since torchvision transforms doesnt support batches (e.g. Normalize() doesnt,
    # we have to do the preprocessing manually. we can use interpolate for
    # resizing, normalizng mean/std is straight forward we just need to get
    # the shape right!)
    imgs = F.interpolate(imgs, size=(size, size), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1,3,1,1)
    std  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1,3,1,1)
    return (imgs - mean) / std

@torch.no_grad()
def calculate_direction_using_clip(generator, text_positive, text_negative, num_samples,
    random_gen, batch_size=32, top_ratio=0.10, device="cpu", **kwargs):

    generator = generator.to(device).eval()

    print(f'available models: {clip.available_models()}')
    # available models:['RN50','RN101','RN50x4','RN50x16',
    # 'RN50x64','ViT-B/32','ViT-B/16','ViT-L/14','ViT-L/14@336px']
    clip_model, _ = clip.load("ViT-B/32", device=device)
    clip_model.eval()
    
    # tokenize our input and feed it to the clip to get the embeddings
    text_tokens = clip.tokenize([text_positive, text_negative]).to(device)
    text_embeddings = clip_model.encode_text(text_tokens)
    # normalize the embeddings so later we can use them for cosine similarity check
    # sidenote:
    # we explained this in vae as well but I repeat it again here as a quick reminder
    # you may see this normalization refered to as putting embeddings on the "unit sphere"
    # or having them have unit length, its a fancy term that says we are making our embedings
    # length to be 1 (i.e. scale each embedding so its length=1) or in some other words, 
    # all the points sit on the surface of a sphere with the radius of 1 or similar takes.
    # the idea is that, we imagine the origin of the embedding space as the center of a 
    # sphere and each embeddings as an arrow pointing outward toward the sphere surface
    # (this is why we say all embeddings lie on the surface of a sphere with radius 1 hence unit sphere!)
    # now by normalizing all of the embeddings to have length 1, the vectors length becomes
    # irrelavent because each and every one's length is the same(ie. 1), therefore for 
    # comparing them, we can only check their angles! the closer the angles, the more similar
    # the embedding vectors are.
    # this way we can quickly compare embeddings against each other using cosine similarity!
    # (quicknote2:
    # as to why we imagine this as a sphere and not anyother shapes? its simply because its
    # the only shape that makes it possible for everything to be equally scaled and therefore
    # comparisons will be purely about direction (concept) and not magnitude (strength)) 
    # quicknote3:
    # we could have also rephrased this operation as all embeddings being normalized so they
    # all become directions and not magnitudes! 
    # or we could have also said, since their length is now 1, they are very close to orgin 
    # that their distance from the origin no longer matters to us only their origentation/direction
    # can be used for comparison!
    # quicknote4:
    # its obvious but I'll mention it anyway, origin is d dimensional, the same as embeddings!
    # so if our embedding dim is 4 e.g. the origin will be (0,0,0,0) and the center of our sphere!
    # so origin that we talk about here is not 2d/3d! just wanted to put this out there in case
    # someone had any confusion concerning this!
    text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)  # [2,D]

    Z = torch.randn(num_samples, generator.z_size, generator=random_gen, device=device)
    margins = []
    for i in range(0, num_samples, batch_size):
        z = Z[i:i+batch_size]
        imgs = generator(z, **kwargs)
        imgs = clip_normalize_image(imgs)
        # grab image embeddings/features
        img_feats = clip_model.encode_image(imgs)
        # normalize them 
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
        # now we calculate the cosine similarity 
        # (since they are normalized the dotproduct gives us the cosine similarity )
        sim = img_feats @ text_embeddings.t()  # [B,2]
        # subtract the positive similarity from negative similarity
        # if the reuslt is >0 then it means our image is closer to 
        # positive text than negative
        margin = sim[:,0] - sim[:,1]
        margins.append(margin)
    margins = torch.cat(margins, dim=0)

    # grab the top most positive margines
    k = max(1, int(math.ceil(top_ratio * num_samples)))
    # grab the top picks
    top_idx = torch.topk(margins, k, largest=True).indices
    # grab the least similar (the closes to negative samples)
    bot_idx = torch.topk(margins, k, largest=False).indices

    z_pos = Z[top_idx]
    z_neg = Z[bot_idx]
    # calculate the direction from the two directions we calculated 
    direction = (z_pos.mean(dim=0) - z_neg.mean(dim=0))
    direction = direction / (direction.norm() + 1e-8)

    info = {"k": k, 
            "top_margin_mean": float(margins[top_idx].mean().cpu()),
            "bottom_margin_mean": float(margins[bot_idx].mean().cpu()),}
    return direction.cpu(), info

@torch.no_grad()
def apply_direction(generator, z, direction, alphas=(-3,-2,-1,0,1,2,3),**kwargs):
    direction = direction.to(z.device)
    imgs = []
    # print(f'direction vector = {direction}')
    for a in alphas:
        z_new = z + a * direction.unsqueeze(0)
        img = generator(z_new,**kwargs)
        imgs.append(img)
    return torch.cat(imgs, dim=0)
#%%
# calculate the direction for glasses attribute
direction, info = calculate_direction_using_clip(generatorcnn, 
                                               text_positive="wearing eyeglasses",
                                               text_negative="without eyeglasses",
                                               num_samples=1024,
                                               random_gen=random_gen,
                                               batch_size=64,
                                               top_ratio=0.10,
                                               device="cpu",)
print("CLIP direction stats:", info)

z = torch.randn(8, generatorcnn.z_size, generator=random_gen)
imgs = apply_direction(generatorcnn, z, direction, alphas=(3,4,5))
show_images(imgs,'CLIP direction',figsize=(12,6))

#%%
# Before we continue if you remember we said getting a GAN to work is 
# an involved effort and requires a few tips and tricks at the very least 
# to get it to work properly. when DCGAN came out, it provided a few of such 
# tips, we used some in our previous examples, one of the main authors later
# posted a list of such tricks in his github repository and it became one of the early
# sources we could get our hands on latest tips and tricks that work! 
# while some of these tips and tricks are still valid, some have gone obsolote
# in newer architectures, and some have also evolved. having said that, for now
# lets review these tips we expand on them later. 
# sidenote:
# this is from 2016 by the way
# https://github.com/soumith/ganhacks#16-discrete-variables-in-conditional-gans
# they directly affect how GANs are trained!
# 
# Quick summary of the points that we covered:
# dont use relu in discriminator! 
# use Guassian/normal distribution instead of uniform for sampling
# in batch use different batches for real and fake separately (especially if you use batchnorm!)
# use tanh for generator's last layer
# use label smoothing
# use adam for generator, and you can use sgd with discriminator but we used adam for both!
# (Adam also helps in lowering mode collapse)
# use larger batcsizes (it stabalizes the training)
# 
# and heres the whole list 
# Main post at github:  
# How to Train a GAN? Tips and tricks to make GANs work
# While research in Generative Adversarial Networks (GANs) continues to improve the fundamental stability of these models, we use a bunch of tricks to train them and make them stable day to day.
# Here are a summary of some of the tricks.
# Here's a link to the authors of this document
# If you find a trick that is particularly useful in practice, please open a Pull Request to add it to the document. If we find it to be reasonable and verified, we will merge it in.
# 1. Normalize the inputs
#     normalize the images between -1 and 1
#     Tanh as the last layer of the generator output
# 2: A modified loss function
# In GAN papers, the loss function to optimize G is min (log 1-D), but in practice folks practically use max log D
#     because the first formulation has vanishing gradients early on
#     Goodfellow et. al (2014)
# In practice, works well:
#     Flip labels when training generator: real = fake, fake = real
# 3: Use a spherical Z
#     Dont sample from a Uniform distribution
# cube.png
#     Sample from a gaussian distribution
# sphere.png
#     When doing interpolations, do the interpolation via a great circle, rather than a straight line from point A to point B
#     Tom White's Sampling Generative Networks ref code https://github.com/dribnet/plat has more details
# 4: BatchNorm
#     Construct different mini-batches for real and fake, i.e. each mini-batch needs to contain only all real images or all generated images.
#     when batchnorm is not an option use instance normalization (for each sample, subtract mean and divide by standard deviation).
# batchmix
# 5: Avoid Sparse Gradients: ReLU, MaxPool
#     the stability of the GAN game suffers if you have sparse gradients
#     LeakyReLU = good (in both G and D)
#     For Downsampling, use: Average Pooling, Conv2d + stride
#     For Upsampling, use: PixelShuffle, ConvTranspose2d + stride
#         PixelShuffle: https://arxiv.org/abs/1609.05158
# 6: Use Soft and Noisy Labels
#     Label Smoothing, i.e. if you have two target labels: Real=1 and Fake=0, then for each incoming sample, if it is real, then replace the label with a random number between 0.7 and 1.2, and if it is a fake sample, replace it with 0.0 and 0.3 (for example).
#         Salimans et. al. 2016
#     make the labels the noisy for the discriminator: occasionally flip the labels when training the discriminator
# 7: DCGAN / Hybrid Models
#     Use DCGAN when you can. It works!
#     if you cant use DCGANs and no model is stable, use a hybrid model : KL + GAN or VAE + GAN
# 8: Use stability tricks from RL
#     Experience Replay
#         Keep a replay buffer of past generations and occassionally show them
#         Keep checkpoints from the past of G and D and occassionaly swap them out for a few iterations
#     All stability tricks that work for deep deterministic policy gradients
#     See Pfau & Vinyals (2016)
# 9: Use the ADAM Optimizer
#     optim.Adam rules!
#         See Radford et. al. 2015
#     Use SGD for discriminator and ADAM for generator
# 10: Track failures early
#     D loss goes to 0: failure mode
#     check norms of gradients: if they are over 100 things are screwing up
#     when things are working, D loss has low variance and goes down over time vs having huge variance and spiking
#     if loss of generator steadily decreases, then it's fooling D with garbage (says martin)
# 11: Dont balance loss via statistics (unless you have a good reason to)
#     Dont try to find a (number of G / number of D) schedule to uncollapse training
#     It's hard and we've all tried it.
#     If you do try it, have a principled approach to it, rather than intuition
# For example
# 
# while lossD > A:
#   train D
# while lossG > B:
#   train G
# 
# 12: If you have labels, use them
#     if you have labels available, training the discriminator to also classify the samples: auxillary GANs
# 13: Add noise to inputs, decay over time
#     Add some artificial noise to inputs to D (Arjovsky et. al., Huszar, 2016)
#         http://www.inference.vc/instance-noise-a-trick-for-stabilising-gan-training/
#         https://openreview.net/forum?id=Hk4_qw5xe
#     adding gaussian noise to every layer of generator (Zhao et. al. EBGAN)
#         Improved GANs: OpenAI code also has it (commented out)
# 14: [notsure] Train discriminator more (sometimes)
#     especially when you have noise
#     hard to find a schedule of number of D iterations vs G iterations
# 15: [notsure] Batch Discrimination
#     Mixed results
# 16: Discrete variables in Conditional GANs
#     Use an Embedding layer
#     Add as additional channels to images
#     Keep embedding dimensionality low and upsample to match image channel size
# 17: Use Dropouts in G in both train and test phase
#     Provide noise in the form of dropout (50%).
#     Apply on several layers of our generator at both training and test time
#     https://arxiv.org/pdf/1611.07004v1.pdf
#%%
# now we have implemented some of these points already, lets implement a few more
# in our dcgan and see what happens!
# maybe lets add normal smapling to the existing trainig loop and let the rest be 
# covered in our next implementations? (like conditional gan?)
# OK, lets implement the conditional version 
# in conditional version, simply put, we include other form of information to affect
# or steer the generation. for example we can use a label and control the generation 
# process, only generating certain images for a specific label. we are not limited to
# labels, we can use text, even another image, or other forms of input to do this. 
# using labels is the simplest one because we already have access to it so lets do that
# In order to get this to work, we need to feed our labels to both discrimnator and generator
# the simplest way to do this for discriminator is either concatenate the label information 
# with the input and then feed them together to the network, or we can onehote encode the labels
# and fuse them later on in the network, before we flatten the featuremaps for classification
# or go advance and fuse this at different levels of the architecture for maximal affectiveness
# for generator the simplest way would be to concatenate the onehot encoded reprsenation of our
# labels and concatenate it with the z_vector that the generator recieves at the begining
# just like the discriminators case, there are more ways to improve this, like using embeddings
# and advance fusion techniques (some of which we explain in VAE section). for now 
# lets keep this simple. 
class DiscriminatorCNNConditional(nn.Module):
    def __init__(self, hidden_size, num_classes, act=nn.LeakyReLU(0.2)):
        super().__init__()

        self.hidden_size = hidden_size
        self.act = act
        self.num_classes = num_classes
        
        # we can use an embedding instead of one-hot encoded version which
        # is a much better choice and provide better results
        # self.embds_lables = nn.Embedding(num_classes, num_classes)
        # but since we want a bit more flexibility we can use a linear layer
        # instead and accept one-hotencoded input and pass it through a linear
        # layer to get new representation. 
        # this will help us with celeba dataset which has multilabels and wouldnt
        # work with our first approach without some adding more logic, the only 
        # catch now is, we need to send one-hot encoded labels for normal datasets
        # such as svhn. obviously we can do much better but in order to keep the
        # architecture as intact as possible this imho is simple workaround
        self.embds_lables = nn.Linear(num_classes, num_classes)
        
        # the input will be the image channels and the mebddesinsg concatenated
        # not the best method, but it works, we could make embeddinsg go through 
        # a transformation and then upsample/fuse it with input or we can do it in one go!
        self.net = nn.Sequential(ConvBlock(3+num_classes, hidden_size, 4, 2, 1, batch_norm=False, act_func=act),
                                 ConvBlock(hidden_size, hidden_size*2, 4, 2, 1, batch_norm=True, act_func=act),
                                 ConvBlock(hidden_size*2, hidden_size*4, 4, 2, 1, batch_norm=True, act_func=act),
                                 nn.Flatten(),
                                 nn.Linear(hidden_size*4 * 4*4, 1),
                                 act)
        
        # initialize weights
        self.apply(weights_init_dcgan)
                
        
    def forward(self, x, y):
        b,c,h,w = x.shape
        # add a nonlinearity on top for further improve results
        y_embds = F.relu(self.embds_lables(y.float()))
        # since we want to add these embeddings as channels
        # we need to expand them to have shape (b,num_classes,h,w)
        # so first we need to add two empty dims for hw
        # sidenote: didnt use inplace version because it causes 
        # error during backprop
        y_embds = y_embds.unsqueeze(2).unsqueeze(3) # (b,num_classes,1,1)
        # and then repeat those dimensions
        y_embds = y_embds.repeat((1,1,h,w)) # (b,num_classes,32,32)
        # print(f'{y_embds.shape=}')
        # concatenate the inputs 
        x = torch.concat([x,y_embds],dim=1)
        return self.net(x)

class GeneratorCNNConditional(nn.Module):
    def __init__(self, z_size, hidden_size, num_classes, act=nn.ReLU()):
        super().__init__()

        self.z_size = z_size
        self.hidden_size = hidden_size
        self.act = act
        # like discriminator we can use an embedding and get the labels instead
        # of onehot encoded version but to make it easier to use with other 
        # datasets such as celeba which are multilabel, we will instead use 
        # a linear layer instead of the embeddings, and use one_hotencoded
        # input for normal datasets like svhn, celeba doesnt need that obviously
        # self.embd_labels = nn.Embedding(num_classes, num_classes)
        self.embds_labels = nn.Linear(num_classes, num_classes)
        
        self.net = nn.Sequential(nn.Linear(z_size + num_classes, hidden_size*4 * 4*4),
                                 nn.BatchNorm1d(hidden_size*4* 4*4),
                                 nn.ReLU(inplace=True),
                                 nn.Unflatten(dim=1, unflattened_size=(hidden_size*4, 4, 4)),
                                 ConvTransBlock(hidden_size*4, hidden_size*2, 4, batch_norm=True, act_func=act), #8x8
                                 ConvTransBlock(hidden_size*2, hidden_size, 4, batch_norm=True, act_func=act),   #16x16
                                 ConvTransBlock(hidden_size, 3, 4, batch_norm=False, act_func=nn.Tanh()),              #32x32
                                 )
        
        # initialize weights
        self.apply(weights_init_dcgan)
        
    def forward(self, x, y): 
        # use a relu nonlinearity on top of our labels
        labels_embeddings = F.relu(self.embds_labels(y.float()))
        x = torch.concat([x,labels_embeddings],dim=1)
        return self.net(x)

dataset_name='celeba'
train_loader = get_dataloader(dataset_name)
imgs, labels = next(iter(train_loader))

if dataset_name=='svhn':
    labels = F.one_hot(labels, 10)
    num_classes = 10
else:
    num_classes = 40

z_vector = torch.randn(size=(imgs.size(0),100))
disc = DiscriminatorCNNConditional(32, num_classes=num_classes)
gen  = GeneratorCNNConditional(100, 32, num_classes=num_classes)
output_d = disc(imgs, labels)
output_g = gen(z_vector, labels)
print(f'{output_d.shape=}')
print(f'{output_g.shape=}')
#%%
# torch.autograd.set_detect_anomaly(True)
# now lets train our architecture again 
# svhn or celeba
dataset_name='celeba'
num_classes = 40 if dataset_name =='celeba' else 10

# a larger batchsize provides more stablity
batch_size = 128
train_loader = get_dataloader(dataset_name=dataset_name, batch_size=batch_size)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

disc_hidden_size = 32#16
gen_hidden_size = 64
# dcgan used 100 if I recall correctly
z_size = 100

# celeba requires more epochs 
epochs = 100#50 
interval = len(train_loader)//2

#discriminator
discriminatorcnn = DiscriminatorCNNConditional(hidden_size=disc_hidden_size, num_classes=num_classes)
discriminatorcnn = discriminatorcnn.to(device)
#generator
generatorcnn = GeneratorCNNConditional(z_size, hidden_size=gen_hidden_size, num_classes=num_classes)
generatorcnn = generatorcnn.to(device)
# DCGAN used [0.5,0.999] for betas for adams for better training stability
# we lower the lr for disciminator so it doesnt learn too fast!
disc_optimizer = torch.optim.Adam(discriminatorcnn.parameters(), 0.0001, [0.5, 0.999])
gen_optimizer = torch.optim.Adam(generatorcnn.parameters(), 0.0002, [0.5, 0.999])

gen_num_samples = 80
# use normal for sampling
fixed_z = torch.randn(size=(gen_num_samples, z_size), device=device)
# create a few samples for each class so we see different variations in each class
samples_count = gen_num_samples // num_classes
# fixed_labels = (torch.arange(0,num_classes).view(num_classes,1)*torch.ones(size=(1,gen_num_samples//num_classes))).view(-1)
fixed_labels = torch.arange(num_classes).repeat(samples_count, 1).t().flatten()
# print(f'{fixed_labels=}')
fixed_labels = F.one_hot(fixed_labels.long(), num_classes=num_classes).to(device)

# for celeba we need a bit more to do 
if dataset_name =='celeba':
    # first pick a few attribute indexes we want to vary
    # we can look at the list_attr_celeba.txt in our celeba folder
    # which contains 40 attribues which are as follows: 
    attributes = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes','Bald', 
                  'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',  
                  'Blurry','Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 
                  'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 
                  'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 
                  'Oval_Face','Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 
                  'Sideburns','Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 
                  'Wearing_Hat','Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']
    # to make it easier lets create a dictionary and pick the attributes that way!
    atrributes_dict = {name:i for i,name in enumerate(attributes)}
    attr_selection = ["Male",'Bald',"Eyeglasses","Smiling",'Young','Wearing_Hat','Black_Hair','Mustache']
    samples_count = gen_num_samples//len(attr_selection)
    fixed_labels = torch.zeros(size=(gen_num_samples, num_classes),device=device)
    # ids = [torch.tensor(atrributes_dict[a]).repeat(samples_count) for a in attr_selection]
    # ids = torch.cat(tuple(ids), dim=0)
    ids = torch.repeat_interleave(torch.tensor([atrributes_dict[a] for a in attr_selection]),samples_count)
    fixed_labels[torch.arange(gen_num_samples), ids] = 1
    # print(f'{fixed_labels.shape=}')
    # torch.set_printoptions(profile="full")
    # print(f'{fixed_labels=}')

experiment_date = datetime.now().strftime("%Y%m%d%H%M%S")
losses = []

for epoch in range(epochs):

    discriminatorcnn.train()
    generatorcnn.train()

    for i, (imgs_real, labels) in enumerate(train_loader):

        #scale input to [-1,1]
        imgs_real = (2*imgs_real-1).to(device)
        labels = labels.to(device)
        # for normal single label datasets such as svhn,cifar10,etc 
        # convert to onehot encoded representation
        if dataset_name != 'celeba':
            labels = F.one_hot(labels, num_classes=num_classes).to(device)
        
        # for celeba, when we go conditional, this will hurt the performance
        # drastically, as there are many labels, and the netowrk needs to
        # comeup with robust features to disentangle them properly, this is
        # hard by itself, given our simple architectures, so adding noise like his
        # will complicate this further. 
        # enable this and see what happens!
        # imgs_real += 0.05 * torch.randn_like(imgs_real)
               
        # train discriminator! 
        # real image predictions
        preds_real = discriminatorcnn(imgs_real, labels)
        disc_real_loss = real_loss(preds_real, smooth=True, device=device)
        
        # generate an image using generator 
        z_vector = torch.randn(size=(imgs_real.size(0), z_size),device=device)
        # we detach the imgs_fake so the discriminator cant use the gradients
        # from the generator and quickly learn!
        imgs_fake = generatorcnn(z_vector, labels).detach()
        
        # add noise to fake images as well(not needed for dcgan)-
        # imgs_fake += 0.05 * torch.randn_like(imgs_fake)
        
        preds_fake = discriminatorcnn(imgs_fake, labels)
        disc_fake_loss = fake_loss(preds_fake, smooth=False, device=device)
        # calculate discrimiator loss out of real and fake losses
        disc_loss = disc_real_loss + disc_fake_loss
        
        # for debugging purposes
        # if disc_real_mean is a lot larger than disc_fake_mean (e.g. 2.0 vs -2.0) 
        # then it means our discriminator is strong but if both are near the same
        # value and the loss is low then it means our discriminator is confused
        # or is over-regularized.
        disc_real_mean = preds_real.mean().item()
        disc_fake_mean = preds_fake.mean().item()
        
        # and optimize discrimnator 
        disc_optimizer.zero_grad()
        disc_loss.backward()
        disc_optimizer.step()

        # now train genertor to create images that look real
        z_vector = torch.randn(size=(imgs_real.size(0),z_size),device=device)
        fake_imgs = generatorcnn(z_vector, labels)
        preds_fake = discriminatorcnn(fake_imgs, labels)
        
        # swap loss! treat fake images as real images
        gen_real_loss = real_loss(preds_fake, smooth=False, device=device)
        # 
        # we are seeing mode collapse it might be due to discriminator is
        # doing great too early so lets mess it up!
        # so occasionally (around 5% of the times) flip labels 
        # if torch.rand(1).item() < 0.05:
        #     gen_real_loss = fake_loss(preds_fake, device=device)
        # else:
        #     gen_real_loss = real_loss(preds_fake, smooth=False, device=device)
            
        # optimize generator
        gen_optimizer.zero_grad()
        gen_real_loss.backward()
        gen_optimizer.step()

        if i+1%interval==0:
            # append discriminator loss and generator loss
            losses.append((disc_loss.item(), gen_real_loss.item()))
            # print discriminator and generator loss
            print(f'Epoch/Epochs: {epoch}/{epochs} | Iter: {i}/{len(train_loader)} | Disc Loss: {disc_loss:6.4f} | Gen Loss: {gen_real_loss:6.4f}')

    losses.append((disc_loss.item(), gen_real_loss.item()))
    
    d_loss_mean = np.mean(np.array(losses)[:,0])
    g_loss_mean = np.mean(np.array(losses)[:,1])
    
    print(f'Epoch/Epochs: {epoch}/{epochs} | Disc Loss : {d_loss_mean:.4f} | Gen loss: {g_loss_mean:.4f} ')
    print(f" -- Discriminator's real mean: {disc_real_mean:.4f} | Discriminator's fake mean = {disc_fake_mean:.4f}")
    
    #save model weights at each epoch
    torch.save({"state_dict":generatorcnn.state_dict(),
                "hidden_size":gen_hidden_size,
                "z_size":z_size,
                "epoch":epoch,
                "losses":losses,
                "dataset_name":dataset_name,
                }, f"./weights/gan/dcgan_generatorcnnconditional_{experiment_date}.pt")
    
    # generate some images mid training to evaluate our model's performance 
    with torch.no_grad():
        generatorcnn.eval()
        # reshape images back to 32x32x3
        generated_images = generatorcnn(fixed_z, fixed_labels).view(-1,*imgs_real.shape[1:])
        display_images(generated_images, 
                    cols=samples_count,
                    title=f'Generated Images at Epoch {epoch}',
                    unnormalize=True,
                    save_path=f'./results/gan/dcgan/{experiment_date}/epoch_{epoch}.jpg')

#%%
losses = np.array(losses)
plt.plot(losses[:,0],label="Discriminator's loss")
plt.plot(losses[:,1],label="Generator's loss")
plt.title('Loss')
plt.legend()
plt.show()
#%%
# load a checkpoint and lets run some experiments on latent space
states = torch.load('./weights/gan/dcgan_generatorcnnconditional_20250813125606.pt',
                    map_location='cpu',
                    weights_only=False)

epoch = states["epoch"]
z_size = states["z_size"]
hidden_size = states["hidden_size"]
dataset_name = states["dataset_name"]
num_classes = 40 if dataset_name =='celeba' else 10

losses = states.get("losses",[0])
losses = np.array(states["losses"])


generatorcnn_conditional = GeneratorCNNConditional(z_size, hidden_size, num_classes=num_classes)
generatorcnn_conditional.load_state_dict(states["state_dict"])
generatorcnn_conditional.eval()

print(f"Generator's weights for {dataset_name.upper()} loaded!")
print(f'z_size: {z_size}')
print(f'hidden_size: {hidden_size}')
print(f'epoch: {epoch}')
print(f'DLoss: {losses[:,0].mean():.4f} | GLoss: {losses[:1].mean():.4f}')
#%%
# to do some latent space arithmetic on a conditional version, we need to
# have the latent vectors that have specific attributes, simply having labels 
# wouldnt do us any favor, as we are after the vector z, for calculating the 
# directions not labels which are simple onehot encoded vectors! 
# so it seems we have no other choice but to use classifier! I googled for a
# celeba classifier, there are a few repos, but they are either just training scripts
# without actual wieghts or their accuracy is not that good, so I decided to 
# quickly train a simple one myself. 

from torchvision import models
class CelebAClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = models.resnet18(models.ResNet18_Weights.DEFAULT)
        # since our generator uses 32x32, but resent is trained on 224x224
        # images on imagenet, we either have to resize our input to match
        # resent input dims, or modify its architecture so it doesnt
        # downsaple too much!
        # use normal conv instead of stride2 which downsamples the input
        self.net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # also remove the first pooling this should avoid too much downsampling
        # of the input early on which was intended for images with 224x224
        # with these changes, each block will be working with larger input volume
        # quicknote:
        # note that by doing these changes we messup the architectures weights
        # and it needs to tune the weights during training so its expected to
        # see degraded performance at first compared to intact architecture! bu overall
        # we should see an improved performance at the end (i.e. higher acc) 
        self.net.maxpool = nn.Identity()
        self.net.fc = nn.Linear(self.net.fc.in_features, 40)
    
    def forward(self, x):
        # well use bcewithlogits so no need for sigmoid here
        return self.net(x)

    def classify(self,x):
        if x.size(-1)>32:
            x = F.interpolate(x, size=(32,32))
        return self(x).sigmoid()

# quick check to see how small the input gets with our changes applied
def check_network_inputs(model):
    hooks = []
    def print_shape_hook(module, input, output):
        classname = module.__class__.__name__
        print(f"{classname}:")
        # input is a tuple and depending on how many inputs the
        # model recieves in forward() will have as many items!
        # since we only send in imgs, then it will only have a
        # single item! the output by default is sent as is
        print(f"  input shape: {input[0].shape}")
        print(f"  output shape: {output.shape}")
        print(f'-------')
        
    # register the hook on each block and the first convlayer
    for name, layer in model.net.named_children():
        if isinstance(layer, (nn.Sequential, nn.Conv2d)):
            hook = layer.register_forward_hook(print_shape_hook)
            hooks.append(hook)        
    return hooks

classifier = CelebAClassifier()
hooks = check_network_inputs(classifier)
x = torch.randn(size=(5,3,32,32))
out = classifier(x)
print(f'{out.shape=}')
# remove hooks its good practice 
# when we are done to remove them
for hook in hooks:
    hook.remove()

# update from future: 
# moved this up here so so we can use them more easily for other gans as well
# these will come in handy in training! we'll use them for label/attribute 
celeba_attribute_names = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes','Bald', 
                  'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',  
                  'Blurry','Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 
                  'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 
                  'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 
                  'Oval_Face','Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 
                  'Sideburns','Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 
                  'Wearing_Hat','Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']
# to make it easier lets create a dictionary and pick the attributes that way!
celeba_attr_word2idx = {name:i for i,name in enumerate(celeba_attribute_names)}
celeba_attr_idx2word = {i:name for name,i in celeba_attr_word2idx.items()}
#%%
# now training
batch_size = 128
train_loader = get_dataloader(dataset_name='celeba',batch_size=batch_size)
val_loader = get_dataloader(dataset_name='celeba', split='valid', batch_size=batch_size)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

celeba_classifier = CelebAClassifier()
celeba_classifier.to(device)

optimizer = torch.optim.AdamW(celeba_classifier.parameters(), lr=0.0001)
criterion = nn.BCEWithLogitsLoss()

epochs = 5
# with batch=128, its 1272
num_batches = len(train_loader)
intervals = num_batches//2 + 1

# show top and bottom 5 attribute accuracies
topk=10
#%%
print(f'Training {CelebAClassifier.__name__}...')

for epoch in range(epochs):
    celeba_classifier.train()
    losses=[]
    train_accuracies = []
    for i, (imgs, labels) in enumerate(train_loader):
        
        imgs,labels = imgs.to(device), labels.to(device).float()
        
        preds = celeba_classifier(imgs)
        loss = criterion(preds, labels)
        losses.append(loss.item())
        
        # apply sigmoid and threshold to get preds
        preds = preds.sigmoid() > 0.5
        accuracy = ((preds == labels).sum().item()/labels.numel())*100
        train_accuracies.append(accuracy)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1)%intervals==0:
            print(f'Epoch: {epoch}/{epochs} | Iter: {i}/{num_batches} | Accuracy: {accuracy:.2f}% | Loss: {loss:.4f}')
    
    # at the end of epoch calculate train accuracy and val accuracy
    train_loss = np.mean(losses)
    train_accuracy = np.mean(train_accuracies)

    with torch.no_grad():
        celeba_classifier.eval()
        val_losses=[]
        val_accuracies = []
        per_attr_accuracies = []
        for imgs, labels in val_loader:
            imgs,labels = imgs.to(device), labels.to(device).float()

            preds = celeba_classifier(imgs)
            loss = criterion(preds, labels)
            
            val_losses.append(loss.item())

            # apply sigmoid and threshold to get preds
            preds = preds.sigmoid() > 0.5
            accuracy = ((preds == labels).sum().item()/labels.numel())*100
            val_accuracies.append(accuracy)

            # per-attribute accuracy
            per_attr_accuracy = ((preds == labels.bool()).sum(dim=0)/imgs.size(0))*100
            per_attr_accuracies.append(per_attr_accuracy.cpu().numpy())
            
        val_loss = np.mean(val_losses)
        val_accuracy = np.mean(val_accuracies)
        # add batch dim to attr_accuracies by stacking them up and then averaging them
        val_per_attr_accuracy = np.mean(np.stack(per_attr_accuracies), axis=0)
       
    print(f'Epoch: {epoch}/{epochs} | Train Acc: {train_accuracy:.2f} | Train Loss: {train_loss:.4f} | Val Acc: {val_accuracy:.2f} | VAL Loss: {val_loss:.4f}')
    # attribute accuracies
    print(f'  -- Val Accuracy per attributes:')
    num_cols = 5
    num_attr = len(celeba_attribute_names)
    num_rows = (num_attr + num_cols - 1) // num_cols
    for row in range(num_rows):
        text = ""
        for col in range(num_cols):
            idx = row + (col * num_rows)
            if idx < num_attr:
                name = celeba_attr_idx2word[idx]
                acc = val_per_attr_accuracy[idx]
                text += f"   {name:<18}: {acc:.2f}   "
        print(text)
    
    # display best and worse accuracies among attributes
    # create list of attributes with their accuracies
    attr_acc_pairs = [(celeba_attr_idx2word[i], acc) for i, acc in enumerate(val_per_attr_accuracy)]
    # sor the accuracies from lowest to highest
    # this way we can easily take topk and bottomk
    # which shows us the best and worse attributes
    attr_acc_pairs.sort(key=lambda x: x[1], reverse=True)

    print(f"\n  -- Top Attributes: {' '*10} -- Bottom Attributes:")
    for ((best_name, best_acc), (worse_name,worse_acc)) in zip(attr_acc_pairs[:topk], attr_acc_pairs[-topk:]):
        print(f"   {best_name:<18}: {best_acc:.2f} {' '*4} {worse_name:<18}: {worse_acc:.2f}")

    torch.save({"state_dict":celeba_classifier.state_dict(),
                "epoch":epoch,
                "loss":train_loss,
                "val_loss":val_loss,
                "train_accuracy":train_accuracy,
                "val_accuracy":val_accuracy,
                "val_per_attr_accuracy":val_per_attr_accuracy,
                },"./weights/cebela_classifier.pt")
#%% load celeba classifier's weights
# good now lets test this
checkpoint = torch.load("./weights/cebela_classifier.pt", map_location="cpu",weights_only=False)
celeba_classifier = CelebAClassifier()
celeba_classifier.load_state_dict(checkpoint.pop("state_dict"))
celeba_classifier.eval()

epoch = checkpoint["epoch"]
train_loss = checkpoint["loss"]
val_loss = checkpoint["val_loss"]
train_accuracy = checkpoint["train_accuracy"]
val_accuracy = checkpoint["val_accuracy"]
val_per_attr_accuracy = checkpoint["val_per_attr_accuracy"]
topk=10
attr_acc_pairs=None
for k,v in checkpoint.items():
    if 'attr' not in k:
        print(f'{k}: {v}')
    else:
        attr_acc_pairs = [(celeba_attr_idx2word[i], acc) for i, acc in enumerate(val_per_attr_accuracy)]
        attr_acc_pairs.sort(key=lambda x: x[1], reverse=True)
        print(f"\n  -- Top Attributes: {' '*10} -- Bottom Attributes:")
        for ((best_name, best_acc), (worse_name,worse_acc)) in zip(attr_acc_pairs[:topk], attr_acc_pairs[-topk:]):
            print(f"   {best_name:<18}: {best_acc:.2f} {' '*4} {worse_name:<18}: {worse_acc:.2f}")
#%%
# create an image using generator
device = 'cpu'
num_samples = 8
with torch.device(device):
    z = torch.randn(size=(num_samples,generatorcnn_conditional.z_size))
    labels = torch.zeros(size=(num_samples, len(celeba_attribute_names)))
    ids = torch.tensor([celeba_attr_word2idx["Eyeglasses"]]).repeat_interleave(num_samples)
    labels[torch.arange(num_samples), ids] = 1

    generatorcnn_conditional.to(device)
    imgs = generatorcnn_conditional(z, labels)
    show_images(imgs, 'generated imgs with galsses', figsize=(12,6))
    
# imgs,lbls = next(iter(val_loader))
# lst=[]
# lst = [imgs[i] for i in range(len(lbls)) if lbls[i][celeba_atrr_word2idx["Eyeglasses"]]==1]
# imgs = torch.stack(lst)
# show_images(imgs[:8],'val dl sample with glasses',figsize=(12,6))
#%%
keyword = 'Eyeglasses'
# keyword = 'Attractive'
# keyword = 'Male'
idx = celeba_attr_word2idx[keyword]
# classify them using our classifier!
imgs_normalized = ((imgs+1)/2).clamp(0,1)
show_images(imgs_normalized,'val dl imgs normalized',figsize=(12,6))
preds = celeba_classifier(imgs_normalized).sigmoid()
# pytorch 2.4 has a bug where it may randomly go for a 
# broken __format__ when it shouldnt and thus results in ValueError!
# its fixed in 2.5 though! the reason im casting to numpy is this!
print(f"raw preds for {keyword}:\n",preds[:,idx].detach().cpu().numpy())
# grab only the high confidence ones
preds_thresh = preds>0.8

for i, (pred,acc) in enumerate(zip(preds_thresh,preds)):
    predicted_attrs = pred.nonzero(as_tuple=True)[0]
    accs = preds[i,predicted_attrs]
    print(f'image {i}: {" ".join([f"{celeba_attr_idx2word[a.item()]}({acc*100:.2f}%) | " for a,acc in zip(predicted_attrs,accs)])}')
#%%
# now lets use it to do some arithemetic in latent space
# for the conditional version, the attributes are controled using labels
# so they reside in label space! we can specify any attribute or combinations of
# them simply by flipping their respective label index and thats all
# we can still play with the latent vector z for some tuning as well but
# its not used for attribute change, just pose/style changes
# for the unconditional version, the attributes live in the latent space
# itself, so by changing latent z we can change everything from one attribute
# to another with other varying pose/style/etc changes. the issue is finding the
# latent codes that does what we want. in our previous attempts, we used sefa/clip
# to find those attributes, but since our model wasnt powerful enough, the features
# werent as developd/disentangled so sefa didnt work properly. the clip also didnt 
# work as expected (but still better than sefa) because for 1 the image resolution
# didnt match, and resizing it to 224 caused lots of artifacts, affecting the clip
# performance adversly. that leaves us here, I had to build a quick classifier so 
# we can use that to identify and get afew samples for a specific attribute and then 
# use that for our latent arithmetic test. 
# so lets create a few samples and see how many attributes we can fish form it
# we start by unconditional version and go to conditional version after it
# uncondition 

@torch.no_grad()
def get_samples_for(generator:GeneratorCNN, classifier:CelebAClassifier,
    attr_name, celeba_attr_word2idx, num_samples, random_generator, 
    threshold=0.5, device='cuda', **kwargs):
    
    generator.to(device)
    classifier.to(device)
    
    generator.eval()
    classifier.eval()
    
    z = torch.randn(size=(num_samples,generator.z_size), device=device, generator=random_generator)
    # update: later added **kwargs so in future models such as progans/stylegan we send
    # additional arguments to the generator, this way we can use this with those archs
    # as well
    imgs = generator(z, **kwargs)
    # classify the images 
    preds = classifier.classify(imgs)
    preds = preds>threshold
    # grab the images with attribute 
    # attribs_indexes = [i for i in range(len(preds)) if preds[i][celeba_attr_word2idx[attr_name]]==1]
    # other_indexes = [i for i in range(len(preds)) if preds[i][celeba_attr_word2idx[attr_name]]!=1]
    # or we can do this more efficiently by calculating a mask and grab the indexes
    # for both
    attr_mask = preds[:,celeba_attr_word2idx[attr_name]] == 1
    attribs_indexes = attr_mask.nonzero(as_tuple=True)[0].tolist()
    other_indexes = (~attr_mask).nonzero(as_tuple=True)[0].tolist()
    
    # grab images and corrosponding latents
    imgs_with_attr = imgs[attribs_indexes]
    latents_with_attr = z[attribs_indexes] 
    # now lets also grab afew images that dont have this attribue
    # so we can use it in our arithmetic test
    imgs_no_attr = imgs[other_indexes]
    latents_no_attr = z[other_indexes]
    
    return (imgs_with_attr,latents_with_attr), (imgs_no_attr,latents_no_attr)
#%%
seed = 66
device = 'cuda'

np.random.seed(seed)
random_gen = torch.cuda.manual_seed(seed) if device =='cuda' else torch.manual_seed(seed)
# note that not all atributes exist in dataset equally
# in fact the attributes are very imbalanced, for example
# trying to get Bald results in fewer success than using Male
# or eyeglasses!
attr_name = "Male"

(imgs,zs),(imgs_other,zs_other) = get_samples_for(generatorcnn,
                                                  celeba_classifier,
                                                  attr_name, 
                                                  celeba_attr_word2idx,
                                                  num_samples=256, 
                                                  random_generator=random_gen, 
                                                  threshold=0.5, 
                                                  device=device)

show_images(imgs,f'Generated with attr {attr_name}',figsize=(12,6))
show_images(imgs_other[:imgs.size(0)],f'Generated without attr({attr_name})',figsize=(12,6))
imgs_gen = generatorcnn(zs)
show_images(imgs_gen,f'Regenerated with latents({attr_name})',figsize=(12,6))
# now that we know everything works, lets do some arrithmetics 
#%%
# for "Eyeglasses" increase the batchsize 
# or else you'll get an error! because it
# may not find any samples with galsses!
# now before we blindly do some arithmetic in order to get
# good results we need to pay attention to a few things
# 1.our z_base may contain the attribute we want to remove already
# and subtracting the direction doesnt do much, or it may have the
# attribute and the arithmatic manipulation further comound its effect
# so the best way to know it actually works and we are doing everything
# correctly is to make sure our z_base is neutral
# 2.the number of samples for with attribute and without attribute
# must match, if our latent samples are a few it will result in a noisy mean
# and wouldnt work properly, using a higher threshold like 0.9 may give us 
# a few very highly confident samples, but leave many lower confidence ones out
# which we could use to get a more accurate mean! so we need to get as many samples
# as we can get our hands on. 
@torch.no_grad()
def get_neutral_latents(generator:GeneratorCNN, classifier:CelebAClassifier,
                        celeba_attr_word2idx, attr_name, num_samples, random_gen,
                        device, threshold=0.1, **kwargs):
    generator.eval()
    generator.to(device)
    classifier.eval()
    classifier.to(device)
    
    z_base = torch.randn(size=(num_samples, generator.z_size), device=device, generator=random_gen)
    imgs = generator(z_base,**kwargs)
    preds = classifier.classify(imgs)
    # now we want to grab all the samples that have 
    # the lowest confidence for selected attribute
    preds_with_attr = preds[:,celeba_attr_word2idx[attr_name]]
    # grab the lowest confidences ids
    ids = (preds_with_attr<threshold).nonzero(as_tuple=True)[0]
    # print(f'{ids=}')
    # and finally grab the latens that dont have that attribute
    z_base = z_base[ids]
    return z_base
#%%
attr_name = 'Smiling'
z = get_neutral_latents(generatorcnn,celeba_classifier, 
                    celeba_attr_word2idx,
                    attr_name,num_samples=32,random_gen=random_gen,device='cpu',threshold=0.05)
imgs = generatorcnn(z)
show_images(imgs,f'neutral images (no {attr_name})',figsize=(12,6))
#%%
attr_name = 'Male'
# z_base = torch.randn(size=(num_samples,generatorcnn.z_size),device=device,generator=random_gen)
z_base = get_neutral_latents(generatorcnn,celeba_classifier, celeba_attr_word2idx,attr_name, 
                            num_samples=32,random_gen=random_gen,device=device,
                            threshold=0.1)

(ims1,zs1),(ims2,zs2) = get_samples_for(generatorcnn, celeba_classifier, attr_name,
                celeba_attr_word2idx=celeba_attr_word2idx,
                num_samples=256,
                random_generator=random_gen,
                # increase the confidence level to 
                # get more accurate results
                threshold=0.7,
                device=device)

show_images(ims1,f'Regenerated with latents({attr_name})',figsize=(12,6))
show_images(ims2,f'Regenerated without latents({attr_name})',figsize=(12,6))
# from women to male!(woman gradually loses feminity and turns into male)
imgs = latent_arithmetic_unconditional(generatorcnn, zs1, zs2[:zs.size(0)], z_base[0],alpha_values=torch.linspace(-3,7,steps=24))
show_images(imgs, 'latent arithmetic(female to male)',figsize=(12,6))
# from male to female!(maleness decreases at each step)
imgs = latent_arithmetic_unconditional(generatorcnn, zs2[:zs.size(0)], zs1, z_base[0],alpha_values=torch.linspace(-3,7,steps=24))
show_images(imgs, 'latent arithmetic(male to female)',figsize=(12,6))

#%% now e can grab different attributes, like men with hairs
# and males, and subtract them to get bald people! but since our generator
# doesnt generate prefect images, our classifier may have difficulty accurately classify them
# so lets stick to the attributes that the model handles better than others for now!
attr_name = "Smiling"
z_base = get_neutral_latents(generatorcnn,celeba_classifier, celeba_attr_word2idx,attr_name, 
         num_samples=32,random_gen=random_gen,device=device, threshold=0.1)

(ims_s,zs_s),(ims_ns,zs_ns) = get_samples_for(generatorcnn, celeba_classifier, attr_name,
                celeba_attr_word2idx=celeba_attr_word2idx,
                num_samples=256,
                random_generator=random_gen,
                # too much confidence can ignore many correct 
                # but lower confidence samples and therefore
                # result in less accurate direction and 
                # ultimately worse result! 
                threshold=0.8,
                device=device)

show_images(ims_s, f'batch for {attr_name}',figsize=(12,6),cols=16)
show_images(ims_ns, f'batch for no {attr_name}',figsize=(12,6),cols=16)
#%%
# not smiling to smiling (gradually smiling is increased)
alphas = torch.linspace(-3,7,steps=24)
img_results = latent_arithmetic_unconditional(generatorcnn, zs_s, zs_ns, z_base[0],alpha_values=alphas)
show_images(img_results, 'latent arithmetic(not smiling to smiling)',figsize=(12,6))
# from smilig to no smiling(the smile gradually fades into anger/crying)
# sidenote: 
# if samples with smiles are less than 13/14, this might not be the case!
# larger samples result in more accurate/stable direction calculation and better 
# end result(play with threshold for get_samples_for and see the result)
img_results = latent_arithmetic_unconditional(generatorcnn, zs_ns, zs_s, z_base[0],alpha_values=alphas)
show_images(img_results, 'latent arithmetic(smiling to nosmiling/angry/crying)',figsize=(12,6))
#%%
# male notsmiling to women smiling!
# gradually maleness decreases, while 
# femaleness and smile increases!
img_results = latent_arithmetic_unconditional(generatorcnn, zs_s, zs1, z_base[0],alpha_values=alphas)
show_images(img_results, 'latent arithmetic',figsize=(12,6))
#%%
# now for the conditional version, since the labels do the majority of work,
# we may think we cant do much but we should still be able to affect attributes, 
# because not all attributes are like in the labels, only 40 are labeled, but 
# many other explicit attributes also exist in the image so we should be able
# to affect it to some extend.(note that our conditional version is obviously worse
# than the unconditional version because in order to get accurate result we need to
# have good fusion of label information into the architecture which we do not
# and the model itself needs to be better configured, all said, we just want to
# get a sense of stuff works and not get the best possible result because we have
# more better changes ahead. anyway we should still get some meaningful changes thisway
# )
def get_labels(attr_names, celeba_attr_word2idx, num_samples,device):
    labels = torch.zeros(size=(num_samples,40),device=device)
    attr_names = [attr_names] if isinstance(attr_names,str) else attr_names
    ids = [celeba_attr_word2idx[n] for n in attr_names]
    labels[:,ids] = 1
    return labels

# l = get_labels("Bald", celeba_attr_word2idx, 4,'cpu')
# print(l)

@torch.no_grad()
def get_samples_for_confitional(generator:GeneratorCNNConditional, labels, num_samples, random_generator, device='cuda'):
    generator.eval()
    generator.to(device)
    labels.to(device)
    
    z = torch.randn(size=(num_samples,generator.z_size), device=device, generator=random_generator)    
    imgs = generator(z,labels)
    return imgs, z

@torch.no_grad()
def latent_arithmetic_conditional(generator, z_with_attr, z_without_attr, z_base, alpha_values, labels=None,ncols=8):
    # grab device from model parameters
    device = next(generator.parameters()).device
    
    # if its a single example, add batch dim
    if z_base.ndim==1:
        z_base.unsqueeze_(0)
    
    # create a neutral label
    if labels is None:
        labels = torch.zeros(size=(z_base.size(0),40),device=device)
        
    # get the direction
    direction = z_with_attr.mean(dim=0) - z_without_attr.mean(dim=0)
    # we normalize the vector so it only encodes the direction and not magnitudes,
    # this way all attribute directions will have unit length and will be comparable.
    # we can then use this fact and control the effect's strength(our desired direction/concept)
    # using a single number like alpha, like for example "move +2 in the smiling direction"
    # or "move -1.5 in the glasses direction" and because all directions are normalized
    # alpha will have a uniform meaning across attributes!
    direction = direction/direction.norm()
    print(f'{direction.shape=}')
    results = []
    # calculate new z based on new direction + add a bit of variety using alpha
    # to see other variations
    for alpha in alpha_values:
        # note: we must multiply alpha by direction not add them!
        # alpha * direction means "take alpha steps along this attribute axis"!
        # like our previous example "move +2 in the smiling direction" now
        # if we just add alpha, it means moving alpha steps in 
        # all directions at once, equally, which has no semantic
        # meaning its just a uniform shift of the latent vector.
        # so to make it exclusive for a specific attribute/direction
        # we only scale that direction by multiplying it exclusively
        z_new = z_base + (alpha * direction)
        # print(f'{z_new.shape=}')
        imgs = generator(z_new, labels).cpu()
        imgs_grid = utils.make_grid(imgs,nrow=ncols)
        results.append(imgs_grid)
    return torch.stack(results)

@torch.no_grad()
def get_neutral_latents_conditional(generator:GeneratorCNNConditional, classifier:CelebAClassifier,
                                    celeba_attr_word2idx, attr_name, num_samples, random_gen,
                                    device,threshold=0.1):
    generator.eval()
    generator.to(device)
    classifier.eval()
    classifier.to(device)
    
    z_base = torch.randn(size=(num_samples, generator.z_size), device=device, generator=random_gen)
    # neutral labels
    labels = torch.zeros(size=(num_samples,40),device=device)
        
    imgs = generator(z_base,labels)
    preds = classifier(imgs).sigmoid()
    # now we want to grab all the samples that have 
    # the lowest confidence for selected attribute
    preds_with_attr = preds[:,celeba_attr_word2idx[attr_name]]
    # grab the lowest confidences ids
    ids = (preds_with_attr<threshold).nonzero(as_tuple=True)[0]
    # print(f'{ids=}')
    # and finally grab the latens that dont have that attribute
    z_base = z_base[ids]
    return z_base

attr_name ='Eyeglasses'
z_base = get_neutral_latents_conditional(generatorcnn_conditional, celeba_classifier,celeba_attr_word2idx,
                                attr_name=attr_name,
                                num_samples=32,
                                random_gen=random_gen,
                                device=device,threshold=0.1)

labels = torch.zeros(size=(z_base.size(0),40),device=device)
imgs = generatorcnn_conditional(z_base,labels)
show_images(imgs, f'generated neutral not having {attr_name}', figsize=(12,6))
#%%

'5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes','Bald', 
'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',  
'Blurry','Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 
'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 
'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 
'Oval_Face','Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 
'Sideburns','Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 
'Wearing_Hat','Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young'

num_samples=8
attr_name1 = "Male"
attr_name2 = "Black_Hair"

z_base = get_neutral_latents_conditional(generatorcnn_conditional, celeba_classifier,
                                         celeba_attr_word2idx,
                                         attr_name=attr_name1,
                                         num_samples=32,
                                         random_gen=random_gen,
                                         device=device,threshold=0.1)

labels1 = get_labels(attr_name1, celeba_attr_word2idx, num_samples=8,device=device)
labels2 = get_labels(attr_name2, celeba_attr_word2idx, num_samples=8,device=device)

imgs1,z1 = get_samples_for_confitional(generatorcnn_conditional, labels1, num_samples,random_gen,device) 
imgs2,z2 = get_samples_for_confitional(generatorcnn_conditional, labels2, num_samples,random_gen,device) 

show_images(imgs1, f'generated images for {attr_name1}',figsize=(12,6), cols=8)
show_images(imgs2, f'generated images for {attr_name2}',figsize=(12,6), cols=8)

alphas = torch.linspace(-3,10,steps=24)
# we use a neutral label so we can see
# the outcome of our manipulation
labels = None
imgs_out = latent_arithmetic_conditional(generatorcnn_conditional, z1,z2,z_base[2], alpha_values=alphas)
show_images(imgs_out, f'generated images for {attr_name1} & {attr_name2}',figsize=(12,6), cols=8)
# the results arent good compared to our unconditional version, its because our model isnt
# doing a great job at generating images, we will revisit this in future again with more powerful
# architecture and hopefully by then we will get much better results!
#%%
# excellent articles on GANs:
# https://jonathan-hui.medium.com/gan-gan-series-2d279f906e7b
# tips and tricks for GANs:
# https://www.reddit.com/r/MachineLearning/comments/i085a8/d_best_gan_tricks/
#
# 
# as we pointedout at the start of this chapter, there are far better architectures
# for GANs than the vanilla version or the DCGAN. lets move ahead and talk about
# these architectures/improvements. 
# initially I was going to cover cyclegan and then wgangp which are the ones I 
# covered back in 2019. but since Im doing tihs all over again, I may as well 
# include a few more architectures and explain what each brought to the table.
# I first try to use papers or methods that can be incorporated
# into our existing architectures so far, like changes related to training regime
# like losses(lsgan,wgangp) and then go to papers that propose a new improved 
# architecture altogether (sgan,progan,stylegan,etc) and finally I'll hope to
# talk about intersting papers in terms of applications (like cylegan,pix2pix, etc)
# that can give us new intuitions about their wide usecases. 
# also I guess I wont explain everything in detail at first, I'd like to have a breif
# inftroduction, enough to get a picture of what we are dealing with and then during 
# implementation, add more details if the needs be. with this out of the way, lets read on!
#  
# LSGAN and WGAN(GP) 
# the papers we are going to talk about now are LSGAN (https://arxiv.org/abs/1611.04076)
# and WGAN/WGAN-GP(https://arxiv.org/abs/1701.07875 / https://arxiv.org/abs/1704.00028). 
# they came after DCGAN. LSGAN came sooner in 2016 and WGAN papers followed the next year.
# these two papers focused on the training aspect and proposed new loss functions to
# improve the performance of GAN architectures.
# LSGAN is short for Least-squares GANs which uses least square loss to avoid vanishing gradients,
# its essentially the DCGAN architecture plus a new loss function (least-square).
# The WGAN paper also proposed to use a different loss function, they used something called a
# Wasserstein distance (EM) for loss and claimed it prevents mode collapse and stablizes the training
# (it indeed makes trainig much more tsable than vanila GAN and doesnt display mode collapse at least
# in datasets tried by the authors, I didnt test it extensively but in the few experiments I have
# done on datasets like cifar10/celeba it performs really well! the wgan papers that followed shortly
# provided way better results).
# 
#
# sidenote: 
# from the wgan-gp paper:
# EM is short for the Earth-Mover (also called Wasserstein-1) distance. EM distance (i.e. W(q, p))
# is informally defined as the minimum cost of transporting mass in order to transform the 
# distribution q into the distribution p (where the cost is mass times transport distance).
# Under mild assumptions, W(q, p) is continuous everywhere and differentiable almost everywhere
# you can think of it as a weaker equivalent to JS and KL losses with the difference its differentiabale
# nearly everywhere, this is explained in the original wgan paper."
# 
# so its a way of measuring how different two probability distributions
# are just like KL and JS, but with the advantage that it is still meaningful and 
# gives useful gradients even when q and p dont overlap at all (where KL and JS completely fail)
# this is why W is said to be differentiable almost everywhere, as explained in the original wgan paper.
# the wgangp paper then added a gradient penalty trick to make sure the critic network(discriminator)
# satisfies the required smoothness (i.e. the 1-Lipschitz constraint) which makes training much more stable.
#  
# also from page 9 of wgangp paper:
# ...The KL divergences between two such distributions are infinite,
# and so the JS divergence is saturated. Although GANs do not literally minimize these divergences
# [16], in practice this means a discriminator might quickly learn to reject all samples that don’t lie
# on Vᵀₙ(sequences of one-hot vectors) and give meaningless gradients to the generator. However,
# it is easily seen that the conditions of Theorem 1 and Corollary 1 of [2] are satisfied even on this
# non-standard learning scenario with X = ∆ᵀₙ. This means that W(Pr, Pg) is still well defined,
# continuous everywhere and differentiable almost everywhere, and we can optimize it just like in any
# other continuous variable setting. The way this manifests is that in WGANs, the Lipschitz constraint
# forces the critic to provide a linear gradient from all ∆ᵀₙ towards towards the real points in Vᵀₙ.
#
# quick note:
# concerning Wasserstein-1 being weaker equivalent as stated in the paper, its not weaker as in less powerufl
# its as in less stricter than KL and JS. in the KL divergence for example if the two distributions 
# dont overlap at all the KL will go to infinity and it punishes the mismatches very harshly.
# likewise, in JS divergence if the two distributions dont overlap the JS will be stuck at log(2)(saturates),
# so gradients vanish! it just refuses to give useful feedback. now compare it to Wasserstein-1
# even if the two distributions dont overlap at all, it still gives a finite and meaningful distance
# (how far youd need to "move mass" to match them!) so both KL and JS give either infinite or zero 
# gradients when distributions are far way and dont overlap at all making them useless when the 
# generator is performig badly but wasserstein-1 would still give us good gradients!
#
# sidenote 2:
# The 1-Lipcshitz constraint basically means the critic function(our discriminator) must not 
# change too fast. if we move the input a little, the output can only move by at most the same amount. 
# in other words, the critic's slope/gradient everywhere must be <=1.
# this smoothness condition is required by the math behind the wasserstein distance 
# (via Kantorovich–Rubinstein duality). without it, the critic could "cheat" and give meaningless
# distance estimates!
#
# quicknote:
# implementing a k-Lipcshitz constraint via "weight clipping" biases the critic towards much simpler
# functions. that is if we implement k-Lipshitz by weight clipping, we will be forcing all the 
# critic's weights to stay within a fixed range (e.g. [-0.01, 0.01]). This was the original 
# trick in the wgan paper to *approximately* enforce the 1-Lipschitz condition. (see note below)
# but clipping makes the critic too simple (low capacity), so it cant learn rich functions well 
# and often leads to poor training. the wgangp fixed this by replacing weight clipping with a 
# gradient penalty, which enforces the Lipschitz condition in a smoother more flexible way.
# 
# sidenote:
# k in k-Lipschitz means its slope (rate of change) is bounded by k, but because we are dealing
# with Wasserstein-1 distance, the math requires it to be k=1, if we go higher than 1, the critc
# will just scale the distance several times up to k times! not only this breaks the definition
# and means we are no longer computing the real Earth-Mover distance, but a scaledand incorrect 
# version of it but also since its scaledup, it will also affect the learning rates, making them
# several times larger which in turn will cause instability in training. unless of course we 
# carefully tune the hyperparameters to account for this cahnge!
# similarly but conversly if k becomes smaller than 1,it will shirnk the learning rtae adn slowdown
# the training!
# 
# this was further improved by WGAN-GP paper which was a very good addition since it results in 
# higher quality images than wgan. (also another good charactestic both WGAN and WGANGP have is 
# their loss roughly correlates with sample quality so if loss decreases it shows the sample quality is
# better this is not the case for previous cases)
#
# so to recap, 
# basically in WGAN, and WGAN-GP, the loss functions are changed. for WGAN, in our discriminator we 
# no longer use an activation function (sigmoid e.g), and simply clip the discriminators weights.
# (as we later see in wgangp paper, this clipping is the reason that makes training unstable in wgan!
# it was fixed in wgan-gp paper by using gradient penalty instead (i.e. penalized the norm of 
# gradient of the discriminator with respect to its input, we'll see this in a moment).
# 
# the reason these losses were proposed was because with both DCGAN and vanilla GAN, when the 
# discriminator works best while the generator is far behind struggling, the gradients will 
# dimnishe and generator will never learn. they were created to address this issue and stablize
# the training.
# moreover we also know that this is a prevalent case, that is discriminator nearly always
# gets optimized easier than the generator in a GAN framework, and minimizing the gan objective
# function with an optimimal discriminator is equivalent to the js-divergence. the problem is
# since this happens and generator is far behind, the generated image distribution (p) will be far 
# away from the actual sample/groundtruth distribution (p) and therefore the gradient will 
# be next to nothing and thus the generator hardly learns anything!
# unless both perform relatively well, generator can not get good gradients from discriminator
# even though it has good gradients due to JS issue with non-overlaping distributions as we
# pointed out!
# 
# for the actual loss calculation for wasserstein-1, a 1-lipschitz function is needed.
# but instead of writting one, we can make the network learn it itself. to do that we
# only need to change the discriminator, so that at the end it doesnt use a nonlinearity
# function like sigmoid, it would therefore learn a scalar value /score instead of a probablity
# which can then be used to interpret as how real the input images are. 
# 
# sidenote:
# theres a similar function in reinforcement learning which is called value-function where 
# it measures how good a state (i.e. input) is. this is where the term critic comes from. 
# because of this, people discriminators in wgan(gp) are refered to as a critic to reflect
# this underlying change and it would be more accurate termonoly as we dont deal with 
# probablity like before and the loss and what takes palce is different.
# 
# OK I guess its enough for now, lets get to the implementation and then at the end we do another 
# recap if necessary
# 
#
#%%
class DiscriminatorCNN(nn.Module):
    def __init__(self, hidden_size, use_batchnorm=True, act=nn.LeakyReLU(0.2)):
        super().__init__()

        self.hidden_size = hidden_size
        self.act = act
        self.use_batchnorm = use_batchnorm
        self.net = nn.Sequential(ConvBlock(3, hidden_size, 4, 2, 1, batch_norm=False, act_func=act),
                                 ConvBlock(hidden_size, hidden_size*2, 4, 2, 1, batch_norm=use_batchnorm, act_func=act),
                                 ConvBlock(hidden_size*2, hidden_size*4, 4, 2, 1, batch_norm=use_batchnorm, act_func=act),
                                 nn.Flatten(),
                                 nn.Linear(hidden_size*4 * 4*4, 1),)
        
        self.apply(weights_init_dcgan)
                
    def forward(self, x):
        return self.net(x)

class GeneratorCNN(nn.Module):
    def __init__(self, z_size, hidden_size,  act=nn.ReLU()):
        super().__init__()

        self.z_size = z_size
        self.hidden_size = hidden_size
        self.act = act
    
        self.net = nn.Sequential(nn.Linear(z_size, hidden_size*4 * 4*4),
                                 nn.BatchNorm1d(hidden_size*4* 4*4),
                                 nn.ReLU(inplace=True),
                                 nn.Unflatten(dim=1, unflattened_size=(hidden_size*4, 4, 4)),
                                 ConvTransBlock(hidden_size*4, hidden_size*2, 4, batch_norm=True, act_func=act), #8x8
                                 ConvTransBlock(hidden_size*2, hidden_size, 4, batch_norm=True, act_func=act),   #16x16
                                 ConvTransBlock(hidden_size, 3, 4, batch_norm=False, act_func=nn.Tanh()),              #32x32
                                 )
        
        # initialize weights
        self.apply(weights_init_dcgan)
        
    def forward(self, x): 
        return self.net(x)

# lsgan is nothing except we replace bce with mse
# some people use sigmoid at final layer of discriminator to get [0-1]
# range but since it affects the gradients (i.e. it weakens them
# since we dont use batchnorm at final layer), we dont do that  
# mse loss can work with raw unbounded logits/scores just fine
def _lsgan_real_loss(preds_real, smooth=False):
    labels = torch.ones_like(preds_real, device=preds_real.device)
    # LSGAN is more stable than DCGAN so we dont need smoothing as much as we used to
    # but it can still be benificial like before (in managing models overconfidence)
    labels = labels * 0.9 if smooth else labels
    return F.mse_loss(preds_real, labels)

def _lsgan_fake_loss(preds_fake, smooth=False):
    device = preds_fake.device
    labels = torch.zeros_like(preds_fake, device=device)
    labels = torch.ones_like(preds_fake, device=device) * torch.distributions.Uniform(0,0.3).sample()\
             if smooth else labels
    return F.mse_loss(preds_fake, labels)

def lsgan_discriminator_loss(preds_real, preds_fake, smooth=False):
    # treat real as real and fake as fake for discriminator
    return _lsgan_real_loss(preds_real, smooth) + _lsgan_fake_loss(preds_fake, smooth)

def lsgan_generator_loss(preds_fake):
    # treat generator output as real
    return _lsgan_real_loss(preds_fake, smooth=False)

# for wgan we use our discriminator/critic raw logits like before
# and the loss is simply the average of fake-real values
# we then need to clip the model weights after each discriminator 
# optimizer step.
# 
# sidenote:
# unlike typical losses we see this one is not positive, 
# usually the loss is a positive number where we try to minize it
# but here, it goes toward negative range (we want real average score/preds_real.mean()
# to be positive and larger than preds_fake), and we can see the 
# same in generator loss as well! the reason is in wgan its reallly
# much better to think of the critic/discriminator's function
# as an objective function needs to be maximized and not as a 
# loss/cost function that needs to be minimized.
# thats why we define the loss here this way, so it becomes a minimization
# which in effect maximizes what we want. our optimizers can only minimize
# so flipping the sign makes it a maximization)
# 
# as we said the discriminator/critics's job is to make the score of 
# real images go up toward +infity, and the score of fake images the other way, 
# i.e. down towards -infinity! basically we(the discriminator) want 
# to maximize this value objective which is preds_real.mean() - preds_fake.mean()
# now imagine for example preds_real is around 50 and preds_fake is around -50!
# the objective would then be 50-(-50)=100. we (the discriminator) want
# to make this number as large as possible. the optimizers we use implement 
# gradient descent, and optimizer.step() in pytorch optimizers likewise is 
# designed to minimize a function by using the negative gradients, so to 
# make it maximize, we simply minize its negative, i.e. -objective! so
# the more/larger negative loss for discriminator (and by extension for generator aswell)
# is a good thing for us! 
def wgan_critic_loss(preds_real, preds_fake):
    # or we could also do -(preds_real.mean() - preds_fake.mean())
    return preds_fake.mean() - preds_real.mean()

def wgan_generator_loss(preds_fake):
    return -preds_fake.mean()

def gradient_penalty(discriminator, imgs_real, imgs_fake):
    batch_size = imgs_real.size(0)
    device = imgs_real.device
    
    # we need to make critic smooth. that is we need to make sure
    # its outputs do not change sharply for small changes in the input images.
    # this is the very definition of 1-lipschitz functions (our model/critic
    # is a function of inputs to outputs).(technically speaking 1-lipschitz 
    # means the critics slope (i.e. gradient wrt input images) is never bigger
    # than 1 which means the output cant change faster than the input moves which
    # in turn means 1-lipschitz critic = the critic's output changes at most 1
    # unit for each unit change in its input image or it doesnt change faster than input
    # or as we said at the begining dont change sharply for small changes in the input!)
    # so how do we do that?
    # we can see/imagine real image and fake image distributions as two separate islands
    # and then bridge the gap between them with smooth values.
    # we can simply do this by interpolating between real and
    # fake images so our discriminator/critic is 1-lipschitz
    # in the space between the two distributions.
    # note that enforcing this only at real or fake points
    # wont be enough simply because the critic could be very 
    # steep in between(i.e. values in between change sharply! or in our example case 
    # if we only checked the islands (real or fake samples only),
    # the bridge could look nice at the ends but have a huge bump or cliff in the middle)
    # so having real images or fake ones be smooth is not enough everything
    # between them needs to also be smooth.
    # we can do this two ways, either randomly pick some numbers each time
    # or use torch.linspace to have fixed points in between. 
    # we use the random way, and pick a random epsillon between [0,1] 
    # simply because random sampling gives stochastic coverage 
    # across training but torch.linspace is fixed which would either be 
    # too sparse which is bad coverage (we might miss bumps in other places)
    # or too dense(i.e. alot of points) which would be too slow to process.
    # so randomly picking numbers is a much better choice as it allows us to
    # cover more ground so to speak(we work with random spots each time we train
    # which over the course of training, after many iterations, we end up 
    # checking all over our so called bridge but without wasting time and much faster as well!)
    # 
    # so to recap:
    # the critic needs to be smooth/1lipschitz and being smooth means its output values dont 
    # change too sharply for small changes in the input image.
    # 1-lipschitz means the critic's slope/rate of change (gradient with respect to input images)
    # is never bigger than 1.
    # and finallt we interpolate between real and fake so we can test and enforce
    # this slope constraint in the space/points between the two distributions, not just at real/fake points.
    #
    # create a random eps with shape (batch, 1,1,1) so it has h,w,c dims
    # so when we multiply it by inputs, its broadcast to have the right dimensions
    eps = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolated_input = eps * imgs_real + (1 - eps) * imgs_fake
    interpolated_input.requires_grad_(True)
    
    inter_preds = discriminator(interpolated_input)
    grad = torch.autograd.grad(outputs=inter_preds,
                               inputs=interpolated_input,
                               grad_outputs=torch.ones_like(inter_preds),
                               create_graph=True,
                               retain_graph=True,
                               only_inputs=True,)[0]
    # caculate l2-norm of gradients
    grad_norm = grad.view(batch_size, -1).norm(2, dim=1)
    # make sure the gradient norm with respect to inputs is almost equal to 1
    # any deviation from norm = 1 is therefore penalized
    penalty = ((grad_norm - 1) ** 2).mean()
    return penalty

def wgangp_critic_loss(critic:DiscriminatorCNN, imgs_real, imgs_fake, lambda_factor=10):
    # the actual loss for wgangp is the wgan loss + the gp
    # the gp replaces the weight clipping part only so the 
    # actual loss stays the same
    # disc_loss= E[disc(fake)] - E[disc(real)] + lambda⋅GP
    # lambda is usally 10.
    wgan_loss = wgan_critic_loss(critic(imgs_real), critic(imgs_fake))
    gp = gradient_penalty(critic, imgs_real, imgs_fake)
    return wgan_loss + (lambda_factor*gp)

x = torch.randn((5,3,32,32))
z = torch.randn((5,100))
discriminatorcnn = DiscriminatorCNN(16)
generatorcnn = GeneratorCNN(100, 16)
# print(f'{generatorcnn}')
doutput = discriminatorcnn(x)
goutput = generatorcnn(z)
print(f'{doutput.shape=}')
print(f'{goutput.shape=}') 
#
# 
#%%
# add FID/IFD/Incepcionscore metric inception score
# up until now, we have checked the training results visually by 
# looking at the generated images during training and telling if everything
# is going as expected. we couldnt decide the image quality simply based on 
# the loss values so now before we go for the actual trainig, lets also implement
# the metrics that will allow us in solving this issue, something thats used in 
# GAN papers to convey how the model is doing! 
# other researchers like us came to the very same decision, that simply reporting
# the loss doesnt convey much as with GANs, losses arent a good representation of
# how the final image looks like, the quality, its variety/diversity of samples
# so in 2016 Inception score was introduced in Improved Techniques for Training GAN
# (paper: ) to do exactly that and it became the standard in gan papers
# the idea behind inception score was that, a good gan needs to produce high quality
# images, that would mean each image should look like a clear object from some class.
# and also it needs to have good diversity, which would mean the the set of generated 
# images should span many classes not just one.
# in order to capture these two creteria, the authors came up with the following thinking:
# first feed generated images into InceptionV3 model(cuz they came from google brain!)
# and get a conditional label distribution for each image p(y∣x) (probability of class y given this image x)
# then average all those class probability vectors in that batch of generated images, this is
# known as marginal distribution p(y), its basically for telling us if we randomly
# pick one generated image whats the overall chance it looks like a cat, dog, car, etc?
# if the generator only generates cats, the marginal distribution might for example 
# look like [0.99 cat, 0.01 everything else] which means low diversity! but if on the other hand
# it generates many classes, the marginal would be more balanced and all classes would 
# roughly be the same (i.e. cat = dog = car = around the same value) which means high diversity!
# and then calculate the kl divergence between the two (i.e. for each image, compare 
# p(y∣x) (sharp prediction for that image) to p(y) (diverse predictions across dataset)
# and this gives us the score. 
# the idea is that to check if an image sharp/has a high quality, we see if its clear
# e.g. "its definitely a dog", therefore p(y∣x) will have low entropy (peak at one class)
# similarliy for analyzing diversity, we check if all generated images are varied 
# (dogs, cats, cars…), if so p(y) has high entropy (spread across classes).
# and the KL divergence rewards this combo. however since its in log form, the numbers
# can be negative and positive, so the authors used exp() on it to make it all positive
# and easier to work with (starts from 1 and goes up so bigger number == better)
#
# the problem with inception score is that it can be fooled, that is, if the generator
# collapses to a few classes but still produces very sharp images, the score can still 
# be high which is not robust. moreover it also depends on imagenet labels, so it can  
# not reflect "quality" in domains that are very different from imagenet (e.g. medical images).
# so a new metric was proposed in 2017 called Frechet Inception Distance (FID) in "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium"
# paper (paper: ) and became the golden standard ever since (since 2018)!
# the paper says the issue with inception score is that it only evaluates relative entropy
# between individual predictions and the marginal distribution, not distributional similarity.
# that is, it only cares about how its predictions fair against each otehr, rather than comparing
# the generated images distribution against the real image distributions which is a much better
# thing to do! FID therefore instead compares the real vs. generated distributions in Inception feature space,
# using means and covariances (like Fréchet distance and hence the name!)
# this captures both quality and diversity more robustly so FID replaced IS since 2018 
# and is used in BigGAN, StyleGAN, other generative models like diffusion models (which 
# we'll see in diffusion chapter).
# so we will be using FID but will also have IS and compare them 
# 
# update: 
# added support for dataloader so we can get more accurate estimates 
# I happened to need them for easier debugging later on (progan section)

from scipy import linalg
class IS_FID_Calculator():
    def __init__(self, device='cpu', cache_dir='./weights/'):
        
        self.device = device
        # to save/cache the score for real images so we dont 
        # compute them over and over 
        self.cache_dir = cache_dir
        self._cach_dirname = "IS_FID_cache"
        # in order to preserve data integrity between experiments
        # we clear the caches whenever we create a new instance
        # and build new ones for that instance, this way if an
        # experiment uses some type of dataaugmentation or resizing
        # etc on the dataset x, the other datasets that dont use
        # those augmentations/resizes or do something of their own
        # will have the wrong cache! so we clear the caches and build
        # new ones each time. so during each experiment we always
        # use fresh caches prepared specificially for that experiment
        self._clear_cache()
        
        weights=models.Inception_V3_Weights.IMAGENET1K_V1
        self.model = models.inception_v3(weights=weights).to(device)
        # we need imagenet mean/std for input preprocessing
        # print(f'{weights.transforms()=}')
        self.mean = torch.tensor(weights.transforms().mean, device=device).view(1,3,1,1)
        self.std  = torch.tensor(weights.transforms().std,  device=device).view(1,3,1,1)
        self.model.eval()
        
        # save for switching between is and fid metrics
        # for FID we remove the classifier but for IS we
        # actually use the classifier!
        self.fc = self.model.fc
    
    def _clear_cache(self):
        dir_path = os.path.join(self.cache_dir, self._cach_dirname)
        cache_files = os.listdir(dir_path)
        if len(cache_files) > 0:
            print("Clearning FID/IS Cache...")
        for f in cache_files:
            if "fid_stats" in f and os.path.splitext(f)[-1]==".pt":
                file_path = os.path.join(dir_path,f)
                os.remove(file_path)
                print(f"  --{f} removed!")
            
    def _preprocess(self, imgs):
        # images to inception must be in [0-1] range since ours
        # is in -1,1 we normalize it back to 0-1!
        imgs = ((imgs+1)/2).clamp(0,1)
    
        # inception expects input size of 299x299 and normalized
        # the original transformation was resizing to 342 on the
        # smallest dim and then center-cropping 299x299m but since
        # our data is 32x32, resizing to 342 would introduce massive
        # pixelation/blurriness, and the scores wouldnt be reliable
        # so instead i decided to go for 299 as before, the better
        # way is to use larger resolutions or use an inceptionv3 
        # model trained on 32x32 imagenet!
        #
        # h, w = imgs.shape[2:]
        # # resize the small side to 342, so we keep aspect ratio
        # small_side = min(h,w)
        # scale = 342 / small_side
        # new_h = int(round(h * scale))
        # new_w = int(round(w * scale))
        # imgs = F.interpolate(imgs, size=(new_h, new_w), mode='bilinear')
        # # center crop to 299x299
        # top = (new_h - 299) // 2
        # left = (new_w - 299) // 2
        # imgs = imgs[:, :, top:top+299, left:left+299]
        # for cifar10 use bicubic to get smoother images?
        # 'bilinear' is the default for inception though!
        imgs = F.interpolate(imgs, size=(299, 299), mode='bicubic')
        imgs = (imgs - self.mean) / self.std
        
        # sidenote: 
        # we could use weights.transform() but it would only
        # run on cpu and we have to manually process each image
        # first converting the image tenstors back to pil and then
        # run it through the transforms() and then stack the results
        # and send to gpu! thats why we used interpolate and manual mean/std
        # normalziation, it runs all on the gpu and is vectorized!
        return imgs
    
    # def _forward(self, imgs):
    #     self.model(imgs)
        
    @torch.no_grad()
    def compute_IS(self, inputs, splits=10):
        # calculate IS = exp(Ex​[KL(p(y∣x) ∥ p(y))]), p(y)=Ex​[p(y∣x)]
        # Ex is expectation of x
        # assign back the classifier in case FID was called
        self.model.fc = self.fc
        
        # if we have a batch of images run them quickly
        if isinstance(inputs, torch.Tensor):
            inputs = inputs.to(self.device)
            processed_imgs = self._preprocess(inputs)
            preds = self.model(processed_imgs).softmax(dim=-1)
        
        else:
            # otherwise we have a dataloader, because we need to process 
            # at least 10k! or more images to get an accurate estimate!
            preds = []
            for imgs,_ in inputs:
                imgs = imgs.to(self.device)
                processed_imgs = self._preprocess(imgs)
                out = self.model(processed_imgs).softmax(dim=-1)
                preds.append(out)
            preds = torch.cat(preds)
            
        # to calculate the IS score, we can do it in one go
        # or do it in splits as the authors did. if we do
        # this in one go, we would only get a single score
        # which already shows how diverse/confident the model's 
        # predictions are which is fine, but it does not tell
        # us how stable that score is with respect to different
        # subsets of data. IS can give varying scores for small
        # datasets so its that robust! on the other hand if
        # we go the other way and do split, by splitting the 
        # dataset and computing IS per split and then reporting
        # mean ± std, we can capture how much the score fluctuates!
        # this will provid us with a measure of reliability 
        # (error bars) that makes model comparisons much easier
        # and more meaningful!
        
        # batchsize must be >= split
        batch_size = preds.size(0)
        assert batch_size>=splits, f'batch_size({batch_size}) must be >= splits{splits}'
        split_size = batch_size // splits
        scores = []

        for i in range(splits):
            # p(y∣x)
            preds_split = preds[i*split_size:(i+1)*split_size, : ]
            # take the average of classes (marginal distribution)
            py = preds_split.mean(dim=0)
            # print(f'{py.shape=}')
            # calculate kl divergence between p(y∣x) (sharp prediction for that image)
            # with p(y) (diverse predictions across dataset) (kl term per class)
            # KL(p(y∣x) ∥ p(y)) = ∑ p(y∣x) log(p(y∣x)/p(y)) 
            # note: log(a/b) = log(a)-log(b)
            kl = preds_split * (preds_split.log() - py.log())
            # sum over classes so we get per-sample KL. we do 
            # this so we get a single value of divergence for
            # each sample, then do a mean() to get the kl mean
            # of the whole split
            kl_mean = kl.sum(dim=1).mean()
            score = kl_mean.exp()
            scores.append(score.item())
        # now instead of having a single vlaue, we calculate the mean/std of
        # the split socres 
        return float(np.mean(scores)), float(np.std(scores))

    @torch.no_grad()
    def _get_FID_mean_covariance(self, inputs):
        # if inputs is a batch of images
        if isinstance(inputs, torch.Tensor):
            inputs = inputs.to(self.device)
            processed_imgs = self._preprocess(inputs)
            features = self.model(processed_imgs)
        else:
            preds = []
            for imgs,_ in inputs:
                imgs = imgs.to(self.device)
                processed_imgs = self._preprocess(imgs)
                features = self.model(processed_imgs)
                preds.append(features)
            features = torch.cat(preds)

        mean = features.mean(dim=0)
        covariance = torch.cov(features.T)
        return mean, covariance
    
    def _get_fname(self, dataset_name, split):
        # create a filename to save/load fid stats to/from disk
        parts = ['fid_stats']
        if dataset_name: parts.append(dataset_name)
        if split: parts.append(split)
        fname = "_".join(parts)+".pt"
        dir_path = os.path.join(self.cache_dir, self._cach_dirname)
        os.makedirs(dir_path, exist_ok=True)
        return os.path.join(dir_path, fname)
    
    def _stats_exists(self, dataset_name, split):
        fname = self._get_fname(dataset_name, split)
        return os.path.exists(fname)
    
    def _read_existing_file(self, dataset_name, split):
        fname = self._get_fname(dataset_name, split)
        return torch.load(fname, weights_only=False)["mean_cov"]
    
    @torch.no_grad()
    def compute_FID(self, real_imgs, fake_imgs, dataset_name=None, split=None):
        #
        # compute FID=∥ μ_r - μ_f ​∥² + Tr(Σr​ + Σf​ - 2 * (Σr​Σf​)1/2)
        # mu is mean and sigma is covariance matrix
        # Tr(M) means sum of diagonal elements of the input matrix(M)
        # its torch.Trace() 
        
        # remove the classifier, we want 2048 features
        self.model.fc = nn.Identity()
        
        # for real dataset if we have already calculated mean_cov use them
        # we need to check both the name and split, some datasets such as
        # celeba have diferent splits like train, extra, test so if user
        # tries to use different splits, we should be able to handle it
        if dataset_name and self._stats_exists(dataset_name, split):
            print(f'Using FID stats for {dataset_name}-{split} from cache...')
            real_mean, real_cov = self._read_existing_file(dataset_name, split)
        else:
            real_mean,real_cov = self._get_FID_mean_covariance(real_imgs)
            # save to disk for future reuse
            fname = self._get_fname(dataset_name, split)
            torch.save({"mean_cov":(real_mean, real_cov)}, fname)
            
        fake_mean,fake_cov = self._get_FID_mean_covariance(fake_imgs)
        
        mean_diff = real_mean - fake_mean
        # squared L2 norm
        mean_diff_squared = mean_diff.dot(mean_diff)#.to(self.device)
        # take matrix square root of covariance matrixes
        # to see how similar the two are 
        #
        # sidenote: we cant use dot product for example to get similarity
        # betwene the covariance matrixes, simply because that would give
        # us the angular similarity and ignore the geometry of probability
        # distributions. we want the geometric similarity as well.
        # the FID distance is in fact also known as wasserstein-2 distance
        # and the sqrtm(squre root trace) comes from that! so the dot product would
        # measure raw similarity of numbers but the sqrtm would measure how
        # close the distributions are in terms of optimal transport!
        # geometrically covariance matrices represent ellipses in the feature
        # space. the matrix square root aligns these ellipses in a way that 
        # accounts for scale and orientation. the trace with the square root
        # gives the true minimal cost of transporting one Gaussian distribution
        # into the other. to get a better mental image, imagine these two 
        # covariance matrixes as two ellipses, one is horizontal, and the 
        # other is vertical, now if we dot product the two, we might get a 
        # large value, since the varaince is high, however they are not 
        # the same distributions, they do not look the same, one is horizontal
        # the other is vertical. the trace squre root captures exactly this 
        # differences and correctly tells us they are not similar at all!
        # pytorch doesnt offer sqrtm function (tf does by the way!) so we
        # have to use scipy for sqrtm.
        cov_prod_sqrt = linalg.sqrtm(real_cov.cpu().numpy() @ fake_cov.cpu().numpy())
        cov_prod_sqrt = torch.from_numpy(cov_prod_sqrt).to(self.device)
        # if the result contains imaginary components, get rid of it!
        if torch.is_complex(cov_prod_sqrt):
            cov_prod_sqrt = cov_prod_sqrt.real
        
        # finally calculating the score!
        # sidenote: the higher the fid score the worse the results are. for example 
        # if its beyond 200, it means the output is garbage and the generated images 
        # are very blurry, not formed properly, basically far from the real distribution!
        # scores around 100-150 are much better, images should be clear as in sharp! but
        # they are malformed, some seem good but the majority have a lot of artifact and issues
        # as we get close to 100 (117 and lower) thing start to get much better though!(still with visible artifacts!)
        # scores around 50-100 are much better than previous cases but are considered 
        # low quality! simply because you can still clearly see artifcats in them
        # lower scores, around 20-50 are considered fine-ish! the generated images are 
        # more similar to the real ones but they still have noticeable issues/flaws. 
        # dcgan,wgan architectures are in this category of scores!(DCGAN baseline is 
        # ~40-50, WGAN-GP around ~25-30) even lower scores like the ones around 10-20 
        # are considered good! they look pretty realistic overall! but you could still 
        # notice some issues in the images and tell they are generated below 10 is 
        # considered really good! images are nearly indistinguishable from real ones!
        # lower than that like 5 and below is just amazingly good! this is the score the 
        # state of the art architectures ahcieved like e.g. BigGAN, StyleGAN2! and you can barely
        # tell them from the real ones if at all!
        # sidenote2:
        # these scores might change based on the dataset we work with, for example you may see
        # celeba have pretty good looking images with FID 110, but cifar10 looks aweful! but
        # the overal trend stays the same, the lower the better!
        # sidenote3:
        # we need more than 10K images for a reliable/stable FID score! so during training with
        # small batchesizes like ours (128) we can get a rough idea about where the training is
        # going, but it wont be a robust metric as the number of samples is just too low!
        # FID and IS are distribution-level metrics not per-batch metrics!
        # so to get real score we need to use large number of images!
        fid = mean_diff_squared + torch.trace(real_cov+fake_cov-2 * cov_prod_sqrt)
        return fid.item()
#%%
imgs = torch.randn(size=(10,3,32,32))
metric = IS_FID_Calculator()
iss = metric.compute_IS(imgs)
fids = metric.compute_FID(imgs,imgs)
# note we are feeding noise! so we should get (1,0)
print(f'{iss=}')
# note we are feeding noise as well, but we compare them 
# against eachother, so our fid should show us they are 
# very close so we should get a low number around 0!
print(f'{fids=}')
del metric
gc.collect()
#%%
# test with dataloaders and cuda
dataset_name = 'celeba'
split='train' # train, test, extra(for celeba)
batch_size=64
loader = get_dataloader(dataset_name=dataset_name, split=split, batch_size=batch_size)
loader2 = iter(loader)
metric = IS_FID_Calculator(device='cuda')
iss = metric.compute_IS(loader)
fids = metric.compute_FID(loader,loader2, dataset_name=dataset_name, split=split)
# note when we used a real dataset we get a much larger mean and a small std
# (for cifar10 I get mu=7.44 and std=0.07. which if we look back we can see the mean is
# much larger than the random noise which was N(1,0) really. the mean/std can be differnt
# for different datasets (for example for celeba-train its iss=(2.8332451581954956, 0.012351234134362015)
# but nonetheless this goes to show if we get small means, as low as 1 or around it, 
# it means our generator has collapsed!
# note that this number is only valid if we used a large number of images, like around
# 10k and more. cifar10 training has 50k images! and if we try the test split we get
# a bit different score (we get slightly different numbers: note we get larger std for
# test as the number of samples is smaller
# iss(train)=(7.4428346157073975, 0.07107214257352347)
# iss(test )=(7.255614423751831, 0.2497227828365655)
print(f'{iss=}')
# looking athe fid we can also see, the score is extremely small, for me I get -0.001
# which shows giving absolutely identical images (we used  the same loader by coping 
# only the iterator) we get that small number. if we use cifar10 test we get slightly
# larger number, so larger images give more accurate estimate
# fids(train)=-0.0010211613262072206
# fids(test )=-0.0026117772795259953
print(f'{fids=}')
del metric
gc.collect()
#%%
# lets add a few more datasets 
def get_transforms(resize_dims, data_augmentation, normalize):
    trans_list = []
    
    if data_augmentation:
        trans_list.append(transforms.RandomHorizontalFlip())
        
    if resize_dims is not None:
        trans_list.append(transforms.Resize(resize_dims))
   
    # always do ToTensor() at the very end
    trans_list.append(transforms.ToTensor())
    
    # normalize to [-1,1]
    if normalize:
        trans_list.append(transforms.Normalize(mean=[0.5]*3, std=[0.5]*3))
    
    return transforms.Compose(trans_list)
    
def get_dataloader(dataset_name="SVHN", split=None, resize_dims=(32,32), batch_size=128, num_workers=8, store_path="./data/", data_augmentation=False, normalize=False):
    dataset_name = dataset_name.lower()
    transform = get_transforms(resize_dims, data_augmentation, normalize)
    dataset=None
    
    if isinstance(split, str):
        split = split.lower()
    
    if dataset_name.lower() == 'mnist':
        is_train = split in [None, True, 'train']
        dataset = datasets.MNIST(os.path.join(store_path, dataset_name.upper()),
                                 train=is_train,transform=transform,download=True)
            
    elif 'cifar' in dataset_name.lower():
        is_train = split in [None, True, 'train']
        dataset = datasets.CIFAR10(os.path.join(store_path, dataset_name.upper()),
                                   train=is_train, transform=transform, download=True)
    
    elif dataset_name == 'svhn':
        split = split if split else 'extra'
        dataset = datasets.SVHN(os.path.join(store_path, dataset_name.upper()),
                                split=split, transform=transform, download=True)
    
    elif dataset_name == 'celeba':
        split = split if split else 'train'
        dataset = datasets.CelebA(os.path.join(store_path),split=split, 
                                  transform=transform, download=True)

    elif dataset_name == 'celeba_hq':
        # 2.7G https://www.kaggle.com/datasets/lamsimon/celebahq
        split = split if split else 'train'
        dataset = datasets.ImageFolder(root=f'{store_path}/{dataset_name}/{split}',
                                       transform=transform)

    elif dataset_name == 'ffhq':
        # the path looks like ./data/ffhq_128/thumbnails128x128/[images are here]
        # but we give ./data/ffhq_128/ so thumbnail128x128 is treated as a single class
        # since we dont need labels, so we can treat all images as one class to use iwth Imagefolder
        dataset = datasets.ImageFolder(root=f'{store_path}/{dataset_name}',
                                       transform=transform)

    else:
        raise ValueError(f"'{dataset_name}' is not a valid dataset name!")
    
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers, pin_memory=True)
    
    return data_loader

# dataset_name = 'celeba'
# train_loader = get_dataloader(dataset_name=dataset_name, resize_dims=(128,128),
#                               batch_size=16, data_augmentation=True,normalize=False)
# #visualize 
# (imgs, labels) = next(iter(train_loader))
# display_images(imgs, title=f'{dataset_name} samples',cols=4)

#%%
def training_loop(discriminator, generator, train_loader, disc_optimizer:torch.optim.Adam, gen_optimizer,
                  epochs, interval, gen_update_interval, dataset_name, loss_type, 
                  lambda_factor=10, gen_num_samples = 64, use_batchnorm=False, wgan_range=(-0.01, 0.01),
                  noise_addition=False,device='cuda', weights_save_dir='./weights/gan', images_save_dir='./results/gan'):
    
    lr_d = [p['lr'] for p in disc_optimizer.param_groups]
    lr_g = [p['lr'] for p in gen_optimizer.param_groups]
    
    print(f'Dataset:                   {dataset_name}')
    print(f'Loss type:                 {loss_type}')
    print(f'Discriminator LR:          {lr_d}')
    print(f'Generator LR:              {lr_g}')
    print(f'Epochs:                    {epochs} ')
    print(f'Interval:                  {interval} ')
    print(f'Generator update interval: {gen_update_interval}')
    print(f'WGAN weight cliping range: {wgan_range}')
    print(f'Noise addition to input:   {noise_addition}')
    print(f'WGAN-GP Lambda factor:     {lambda_factor}')
    print(f'gen_num_samples:           {gen_num_samples}')
    print(f'Checkpoint Directory:      {weights_save_dir}')
    print(f'Images Directory:          {images_save_dir}')
    
    metric = IS_FID_Calculator(device)

    fixed_z = torch.randn((gen_num_samples,generator.z_size)).to(device)

    experiment_date = datetime.now().strftime("%Y%m%d%H%M%S")
    
    print(f'Training on {dataset_name} with loss={loss_type} in {experiment_date}')

    for epoch in range(epochs):
        discriminator.train()
        generator.train()
        
        losses = []
        epoch_scores = []
        for i, (imgs_real, _) in enumerate(train_loader):
            #scale input to [-1,1]
            imgs_real = (2*imgs_real-1).to(device)
            
            # if adding noise makes trainig more stable and we get
            # better looking images it means our discriminator is
            # too powerful that messing the signal up and making it
            # harder for it, improves our result! it acts as a regularizer
            # (in terms of distribution impact, adding noise increases the variance
            # for both real/fake images so the discriminator cant prefectly memorize
            # the training data or latch onto a single fake mode!)
            if noise_addition:
                imgs_real += 0.05 * torch.randn_like(imgs_real)
               
            # train discriminator/critic! 
            # real image predictions
            preds_real = discriminator(imgs_real)
            # disc_real_loss = real_loss(preds_real, smooth=True, device=device)
            # generate an image using generator 
            z_vector = torch.randn((imgs_real.size(0), z_size)).to(device)
            # we detach the imgs_fake so the discriminator cant use the gradients
            # from the generator and quickly learn!
            imgs_fake = generator(z_vector).detach()
        
            # add noise to fake images as well(not needed for dcgan)
            if noise_addition:
                imgs_fake += 0.05 * torch.randn_like(imgs_fake)
        
            preds_fake = discriminator(imgs_fake)
            # disc_fake_loss = fake_loss(preds_fake, smooth=False, device=device)
            # calculate discrimiator loss out of real and fake losses
            if loss_type =='lsgan':
                disc_loss = lsgan_discriminator_loss(preds_real, preds_fake)
            elif loss_type =='wgan':
                disc_loss = wgan_critic_loss(preds_real, preds_fake)
            elif loss_type =='wgangp':
                disc_loss = wgangp_critic_loss(discriminator, imgs_real, imgs_fake, lambda_factor=lambda_factor)
            else:
                raise ValueError(f"Invalid loss type:{loss_type} entered!")
        
            # for debugging purposes
            # if disc_real_mean is a lot larger than disc_fake_mean (e.g. 2.0 vs -2.0) 
            # then it means our discriminator is strong but if both are near the same
            # value and the loss is low then it means our discriminator is confused
            # or is over-regularized.
            disc_real_mean = preds_real.mean().item()
            disc_fake_mean = preds_fake.mean().item()
            
            # store average scores for real and fake images
            epoch_scores.append((disc_real_mean, disc_fake_mean))
            
            # and optimize discrimnator 
            disc_optimizer.zero_grad()
            disc_loss.backward()
            disc_optimizer.step()
        
            # dont forget to clip discriminator's weights in wgan
            if loss_type=='wgan':
                for p in discriminator.parameters():
                # keep it roughly 1-lipschitz smaller than this
                # restricts the discriminator/critic too much!
                # larger values might be ok only if things go south!
                    p.data.clip_(*wgan_range)

            # now train genertor to create images that look real
            z_vector = torch.randn((imgs_real.size(0),z_size)).to(device)
            fake_imgs = generator(z_vector)
            preds_fake = discriminator(fake_imgs)
        
            # generator loss
            # swap loss! treat fake images as real images
            if loss_type=='lsgan':
                gen_real_loss = lsgan_generator_loss(preds_fake)
            
            elif 'wgan' in loss_type: #wgan-wgangp
                gen_real_loss = wgan_generator_loss(preds_fake)
        
            else:
                raise ValueError(f"losstype {loss_type} not detected!")
            
            # optimize generator
            # update generator with a delay, usually update per 5 critic update
            # seems to make convergence faster
            if (i+1)%gen_update_interval == 0:
                gen_optimizer.zero_grad()
                gen_real_loss.backward()
                gen_optimizer.step()
        
            if (i+1)%interval==0:
                # append discriminator loss and generator loss
                # losses.append((disc_loss.item(), gen_real_loss.item()))
                # print discriminator and generator loss
                print(f'[Epoch {epoch}/{epochs} | Iter: {i}/{len(train_loader)}] Disc Loss: {disc_loss:.6f} | Gen Loss: {gen_real_loss:.6f}')
                print(f" -- Batch-{i}: Disc's real mean: {disc_real_mean:.4f} | Disc's fake mean = {disc_fake_mean:.4f}")
                
            losses.append((disc_loss.item(), gen_real_loss.item()))
    
        d_loss_mean = np.mean(np.array(losses)[:,0])
        g_loss_mean = np.mean(np.array(losses)[:,1])

        average_score_real_mean = np.mean(np.array(epoch_scores)[:,0])
        average_score_fake_mean = np.mean(np.array(epoch_scores)[:,1])

        # calculate is/fid scores-
        # we simply use a single batch to get a rough idea
        # how things are, they cannot be used to compare with
        # whats reported in papers, for that we need to run on
        # a large number of images(real and fake)
        IS_score = metric.compute_IS(imgs_fake)
        FID_score = metric.compute_FID(imgs_real, imgs_fake)
    
        print(f" -- Last Batch : Disc's real mean: {disc_real_mean:.4f} | Disc's fake mean: {disc_fake_mean:.4f}")
        print(f" -- Epoch's Avg: Disc's real mean: {average_score_real_mean:.4f} | Disc's fake mean: {average_score_fake_mean:.4f}")
        print(f'[Epoch {epoch}/{epochs}] Disc Loss-Avg: {d_loss_mean:.6f} | Gen loss-Avg: {g_loss_mean:.6f} | IS: (μ:{IS_score[0]:.4f}, σ²:{IS_score[1]:.4f}) | FID: {FID_score:.2f}')
        
        #save model weights at each epoch
        torch.save({"state_dict":generator.state_dict(),
                "hidden_size":generator.hidden_size,
                "z_size":generator.z_size,
                "lr_d":lr_d,
                "lr_g":lr_g,
                "use_batchnorm":discriminator.use_batchnorm if hasattr(discriminator,'use_batchnorm') else False,
                "noise_addition":noise_addition,
                "wgan_range":wgan_range,
                "epoch":epoch,
                "gen_update_interval":gen_update_interval,
                "loss_type":loss_type,
                "FID":FID_score,
                "IS":IS_score,
                "d_loss_mean":d_loss_mean,
                "g_loss_mean":g_loss_mean,
                "dataset_name":dataset_name,
                }, f"{weights_save_dir}/dcgan_generatorcnn_{loss_type}_{experiment_date}.pt")
    
        # generate some images mid training to evaluate our model's performance 
        with torch.no_grad():
            generator.eval()
            # reshape images back to 32x32x3
            generated_images = generator(fixed_z).view(-1,*imgs_real.shape[1:])
            display_images(generated_images, 
                    cols=gen_num_samples//8,
                    title=f'Using {loss_type.upper()} at Epoch {epoch} FID:{FID_score:.2f} (dLoss:{d_loss_mean:.6f} | gLoss:{g_loss_mean:.6f})',
                    unnormalize=True,
                    save_path=f'{images_save_dir}/dcgan_{loss_type}/{experiment_date}/epoch_{epoch}.jpg')
    print("training is complete!")

#%%    
# now lets train 
#'lsgan'
#'wgan'
#'wgangp'
#sidenote: lsgan and wgangp give the best results
loss_type = 'wgangp'
# for wgangp /gradient penalty scaler lambda
lambda_factor=10

dataset_name = 'celeba'
batch_size=128
train_loader = get_dataloader(dataset_name=dataset_name, split='train',batch_size=batch_size)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

disc_hidden_size = 32#16
gen_hidden_size = 64
z_size = 100
# wgan and wgangp dont use batchnorm because it messes with the 1-lipschitz constraint
# as it introduces sample coupling(sample to sample coupling) while 1-lipschitz constraint
# requires each and every sample to conform to this. I however trained with batchnorm and 
# it seemed completely fine! though it may not work on complex datasets, or we might see 
# mode collapse or other weirdness later on, I havent digged too much though cifar10/celeba seem fine!
# without bn, the convergence rate slows down drastically!(also wgangp gives better results
# than wgan when no bn is used. when bn is used their (wgan and wgangp )results seem the same)
# update:
# in small resolution (32x32) and short trainig it seems having
# batchnorm works, but without batchnorm, which should be the default
# wgangp provides much better images much faster (see next experiment with 64x64 input
# and larger network!) the wgan is just not that stable! without batchnorm it just doesnt
# perform well! see my explanation ahead!
# use_batchnorm = True
use_batchnorm = loss_type=='lsgan'


epochs = 50
num_batches = len(train_loader)
interval = num_batches//2+1
# every 5 discriminator/critic updates, update the generator
# sidenote:
# during training wgan I faced the discriminators loss
# became 0! as early as the first few epochs! that meant the discriminator
# had collapsed completely! after a bit of digging and trying lsgan and wgangp
# and noticing they work well, it was clear the discriminator wasnt the issue
# the trainig wasnt either, so it had to be wgan loss. it turns out this was thecase
# the weight clipping keeps value in a very narrow range, restriciting its capcity
# on the other hand, I set generator update to take palce evry 5 discriminator update
# so the discrimnator quickly found a way to identify all generator's as fake
# and saturated! generator stopped recieiing good gradients and got stuck at generating
# nonsense! our learning rates were too low as well and it contributed as welll I guess
# cuz it couldnt distinguish between real/fake images quickly so generator wouldnt get any gradients!
# so either way we had an issue! its an inherent issue of wgan and our training regime issue
# for fixing it, we can make generator get updated as quickly as the discrimnator or use larger lrs
# I first reverted the lrs back to 0.0001/0.0002 instead of (0.001/0.002) this helped
# I no longer got absolute mess! the images however very extremely blurry the loss stayed
# around 0.0001 ish up until the very end. didnt improve much. next I increased the 
# generators update frequency, made it as fast as the discriminator(gen_update_interval=1)
# this fixed the other issue, now I have images that look way better and trainig seems to be
# gooing smoothly (although the loss is still low, and training goes slowly but its working now!)
# by enabling batchnorm for wgan, it gets much better though as we know it shouldnt be used
# but it seems this t ime around it actually contributes in more stable/powerful gradient signal
# and it shows in discriminators loss (its around 100x larger with batcnorm!)
# wgangp on the other hand is way more stable! it works fine wih and without batchnorm
# with gen_update_interval 1 and 5! lsgan too lsgan works fine with1
# the convergence rate with 1 is much faster for wgan and wet lower IFD score
gen_update_interval = 1 #5 if "wgan" in loss_type else 1 

#discriminator
discriminatorcnn = DiscriminatorCNN(hidden_size=disc_hidden_size, use_batchnorm=use_batchnorm)
discriminatorcnn = discriminatorcnn.to(device)
#generator
generatorcnn = GeneratorCNN(z_size, hidden_size=gen_hidden_size)
generatorcnn = generatorcnn.to(device)

# the paper says [0.5,0.999] diverges in wgan/wgangp
betas = [0.5, 0.999] if loss_type=='lsgan' else [0, 0.9]

# for wgangp/lsgan I used 0.001/0.002(1e-4/2e-4 also work but are slower)
# for wgan I used 0.0001/0.0002 nothing larger works properly (even with batchnorm)
if loss_type=='wgan':
    lr_d, lr_g  = 0.0001, 0.0002
else:
    # lsgan and wgangp(with and without batchnorm) work with 
    # both 1e-3/2e-3 and 1e-4/2e-4.
    lr_d, lr_g = 0.001, 0.002

# disc_optimizer = torch.optim.RMSprop(discriminatorcnn.parameters(), lr=5e-5) # for wgan
disc_optimizer = torch.optim.Adam(discriminatorcnn.parameters(), lr_d, betas=betas)
gen_optimizer = torch.optim.Adam(generatorcnn.parameters(), lr_g, betas=betas)


training_loop(discriminatorcnn, 
              generatorcnn, 
              train_loader=train_loader,
              disc_optimizer=disc_optimizer,
              gen_optimizer=gen_optimizer, 
              epochs=epochs, 
              interval=interval,
              gen_update_interval=gen_update_interval, 
              dataset_name=dataset_name,
              loss_type=loss_type, 
              lambda_factor=lambda_factor,
              use_batchnorm=use_batchnorm,
              device=device)

#%%
# note that the scores we get here is expected to be low as we have a very 
# simple architecture and training regime aside from the fact that our images
# are 32x32 which when resized will still be blurry and pixelated which will
# lead to lower fid score, beefing the network up, and having a better traiing
# regime and better hyper parameters will increase our score but we are not 
# going to stick here for long! we have more architectures to cover!
# we will test with larger imagesze and abit larger network though! see ahead!
#%%
# 
# load models 
# dcgan_generatorcnn_wgangp_20250830150142
# dcgan_generatorcnn_wgangp_20250901143328.pt
# try models for sep 1 (20250901) after 17 which I applied the latest changes!
checkpoint = torch.load("./weights/gan/dcgan_generatorcnn_wgangp_20250901143328.pt",
                        map_location="cpu",
                        weights_only=False)

epoch = checkpoint["epoch"]
z_size = checkpoint["z_size"]
hidden_size = checkpoint["hidden_size"]
dataset_name = checkpoint["dataset_name"]
loss_type = checkpoint["loss_type"]
losses = np.array(checkpoint.pop("losses"))

generatorcnn = GeneratorCNN(z_size,hidden_size)
generatorcnn.load_state_dict(checkpoint.pop("state_dict"))
generatorcnn.eval()

for k,v in checkpoint.items():
    print(f'{k}: {v}')
    
print(f'DLoss: {losses[:,0].mean():.4f} | GLoss: {losses[:1].mean():.4f}')
#%%
# now lets retest again 
def run_latent_arithmatic(attr_name, 
                          generator:GeneratorCNN,
                          classifier:CelebAClassifier,
                          word2idx, 
                          random_gen,
                          showcase_one_sample=True,
                          num_samples=32, 
                          attribute_pool_size=256,
                          maximum_prob_for_neutral_confidence=0.1,
                          attribute_confidence_rate=0.7,
                          alpha_values: torch.Tensor | None = None,
                          device='cpu',
                          topk=10,
                          **kwargs):

    if alpha_values is None:
        alpha_values = torch.linspace(-3, 7, steps=24)

    z_base = get_neutral_latents(generator, classifier, word2idx, attr_name, num_samples,
                                random_gen, device,
                                # use lower probs to get more accurate results, 
                                maximum_prob_for_neutral_confidence,**kwargs)

    with_attrs, without_attrs = get_samples_for(generator, classifier, attr_name,
                                                word2idx,
                                                num_samples=attribute_pool_size,
                                                random_generator=random_gen,
                                                # increase the confidence level to 
                                                # get more accurate results too much
                                                # confidence can ignore many correct 
                                                # but still lower confidence samples and
                                                # therefore result in less accurate direction
                                                # and ultimately worse result! 
                                                threshold=attribute_confidence_rate,
                                                device=device,
                                                **kwargs)

    (ims_attr, latents_attrs) = with_attrs
    (ims_no_attrs, latents_no_attrs) = without_attrs
    
    print(f'{ims_attr.shape=}')
    print(f'{ims_no_attrs.shape=}')
    
    col_count1 = math.ceil(math.sqrt(ims_attr.size(0)))
    col_count2 = math.ceil(math.sqrt(ims_no_attrs.size(0)))
    
    show_images(ims_attr, f'latents with ({attr_name})', cols=col_count1, figsize=(12,6))
    show_images(ims_no_attrs, f'latents without({attr_name})', cols=col_count2, figsize=(12,6))
    
    latents = z_base[0] if showcase_one_sample else z_base
    
    #! why did I do this?zs.size(0)? was I trying to use more or elss attributes to 
    #! see how that affects the result? add this as argument
    # I wrote topk instead of zs.size(0)!
    # e.g. from women to male!(woman gradually loses feminity and turns into male) 
    imgs = latent_arithmetic_unconditional(generator, latents_attrs, latents_no_attrs[:topk], latents, alpha_values,**kwargs)
    show_images(imgs, f'latent arithmetic(opposite toward {attr_name})', figsize=(12,6))
    
    # e.g. from male to female!(maleness decreases at each step)
    imgs = latent_arithmetic_unconditional(generator, latents_no_attrs[:topk], latents_attrs, latents, alpha_values,**kwargs)
    show_images(imgs, f'latent arithmetic({attr_name} toward the opposit)',figsize=(12,6))
    print(f'done!')
#%%
run_latent_arithmatic(attr_name='Male', 
                      generator=generatorcnn, 
                      classifier=celeba_classifier, 
                      word2idx=celeba_attr_word2idx,
                      random_gen=random_gen,
                      showcase_one_sample=True,
                      num_samples=64,
                      attribute_pool_size=256,
                      maximum_prob_for_neutral_confidence=0.1,
                      attribute_confidence_rate=0.8,
                      alpha_values=torch.linspace(-3,7,steps=24),
                      device='cpu')

#%%
run_latent_arithmatic(attr_name='Smiling', 
                      generator=generatorcnn, 
                      classifier=celeba_classifier, 
                      word2idx=celeba_attr_word2idx,
                      random_gen=random_gen,
                      showcase_one_sample=True,
                      num_samples=32,
                      attribute_pool_size=256,
                      maximum_prob_for_neutral_confidence=0.01,
                      attribute_confidence_rate=0.8,
                      alpha_values=torch.linspace(-3,7,steps=24),
                      device='cpu')
#%%
run_latent_arithmatic(attr_name='Eyeglasses', 
                      generator=generatorcnn, 
                      classifier=celeba_classifier, 
                      word2idx=celeba_attr_word2idx,
                      random_gen=random_gen,
                      showcase_one_sample=True,
                      num_samples=32,
                      attribute_pool_size=256,
                      maximum_prob_for_neutral_confidence=0.01,
                      attribute_confidence_rate=0.8,
                      alpha_values=torch.linspace(-3,7,steps=24),
                      device='cpu')

#%% 
#%%
# ok so we saw lsgan did pretty good and trainig was a breeze as we could 
# use batcnorm unlike wgan which was a head in the neck! and we had a lot 
# of issues trainig it and getting a somewhat decent output. 
# wgangp on the otherhand proved to be way more stable than wgan, and without
# batchnorm could create very good looking iamges. it gave way better/larger/stabler
# gradient signal (unlike wgan which was extremely low) and it also worked well 
# with batchnorm enabled!
# before we go to other architectures, lets create a more powerful version of our network!
# and see how much of a difference it makes!
# 
# There are several issues with our previous implementation, if we want 
# to get any imporvements we need to address every single one of them. 
# first we cant go ahead and simply do anything with our blocks
# GANs are extremely fragile/sensitive to work with, even a slight 
# seemingly okish change can result in complete failure!
# to make this more practical, lets create a version first and then
# analyze it and improve it. 
# 
# sidenote: 
# The following version causes massive instability in trainig!
# we can only train porperly with large lr and even then we dont get
# good results! the lsgan completely fails with severe mode collapse!
# the wgan also fails, and only wgangp manages to produce something intersting!
# (In GANs we need to have simple discriminator anything complex
# powerful complicates things! having said that lets continue and 
# later see explanations ahead!)
class ConvBlock2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=2, padding=1, batch_norm=False, act_func=nn.LeakyReLU(0.2)):
        super().__init__()
         # we do subsamling by using stride=2
        self.block = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size,
                                             stride, padding, bias=not batch_norm),
                                   nn.BatchNorm2d(out_channels) if batch_norm else
                                   nn.Identity(),
                                   act_func,
                                   # 1x1
                                   nn.Conv2d(out_channels, out_channels//2, kernel_size=1,
                                             stride=1, padding=0, bias=not batch_norm),
                                   nn.BatchNorm2d(out_channels//2) if batch_norm else
                                   nn.Identity(),
                                   act_func,
                                   nn.Conv2d(out_channels//2, out_channels, kernel_size=1,
                                             stride=1, padding=0, bias=not batch_norm),
                                   nn.BatchNorm2d(out_channels) if batch_norm else
                                   nn.Identity(),
                                   act_func,)
    def forward(self, x):
        return self.block(x)

# convtranspose lets add more layers to our block and use bottleneck design
# so we can learn better representations by forcing the network that way!
class ConvTransBlock2(nn.Module):
    def __init__(self,  in_channels, out_channels, kernel_size,
                 stride=2, padding=1, batch_norm=False, act_func=nn.ReLU(inplace=True)):
        super().__init__()
        self.block = nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                             stride, padding, bias=not batch_norm),
                                   nn.BatchNorm2d(out_channels) if batch_norm else
                                   nn.Identity(),
                                   act_func if act_func else nn.Identity(),
                                   
                                   nn.Conv2d(out_channels, out_channels//2, kernel_size=1,
                                             stride=1, padding=0, bias=not batch_norm),
                                   nn.BatchNorm2d(out_channels//2) if batch_norm else
                                   nn.Identity(),
                                   act_func if act_func else nn.Identity(),
                                   
                                   nn.Conv2d(out_channels//2, out_channels, kernel_size=1,
                                             stride=1, padding=0, bias=not batch_norm),
                                   nn.BatchNorm2d(out_channels) if batch_norm else
                                   nn.Identity(),
                                   act_func if act_func else nn.Identity())
        
        
        self.residual = nn.Sequential(nn.Upsample(scale_factor=stride, mode='nearest'),
                                      nn.Conv2d(in_channels=in_channels, out_channels=out_channels,kernel_size=1,
                                                stride=1, bias=not batch_norm),
                                      nn.BatchNorm2d(out_channels) if batch_norm else
                                      nn.Identity(),
                                      )
 
    def forward(self, x):
        out = self.block(x)
        x_res = self.residual(x)
        # using activation functions like relu on (out+x_res) will
        # completely destroy generation! so dont apply any activations
        out = out+x_res
        return out

# this time lets train on 64x64 images!
class DiscriminatorCNN64(nn.Module):
    def __init__(self, hidden_size=32, use_batchnorm=True, act=nn.LeakyReLU(0.2)):
        super().__init__()

        self.hidden_size = hidden_size
        self.act = act
        self.use_batchnorm = use_batchnorm
        # the ConvBlock2 does not even work, so to get this out of the way lets use convblock
        # and carry on with the original version
        self.net = nn.Sequential(ConvBlock(3, hidden_size, 4, 2, 1, batch_norm=False, act_func=act),#32x32
                                 ConvBlock(hidden_size, hidden_size*2, 4, 2, 1, batch_norm=use_batchnorm, act_func=act),#16x16
                                 ConvBlock(hidden_size*2, hidden_size*4, 4, 2, 1, batch_norm=use_batchnorm, act_func=act),#8x8
                                 ConvBlock(hidden_size*4, hidden_size*4, 4, 2, 1, batch_norm=use_batchnorm, act_func=act),#4x4
                                 nn.Flatten(),
                                 nn.Linear(hidden_size*4 * 4*4, 1),)
        # the weight initt is still very important,
        # the dcgan weight init makes things more stable!
        self.apply(weights_init_dcgan)

    def forward(self, x):
        return self.net(x)

class GeneratorCNN64(nn.Module):
    def __init__(self, z_size, hidden_size,  act=nn.ReLU()):
        super().__init__()

        self.z_size = z_size
        self.hidden_size = hidden_size
        self.act = act
        self.net = nn.Sequential(nn.Linear(z_size, hidden_size*4 * 4*4),
                                 nn.BatchNorm1d(hidden_size*4 * 4*4),
                                 nn.ReLU(inplace=True),
                                 nn.Unflatten(dim=1, unflattened_size=(hidden_size*4, 4, 4)),
                                 ConvTransBlock2(hidden_size*4, hidden_size*2, 4, batch_norm=True, act_func=act), #8x8
                                 ConvTransBlock2(hidden_size*2, hidden_size*2, 4, batch_norm=True, act_func=act), #16x16
                                 ConvTransBlock2(hidden_size*2, hidden_size, 4, batch_norm=True, act_func=act), #32x32
                                 ConvTransBlock2(hidden_size, 3, 4, batch_norm=False, act_func=nn.Tanh()),      #64x64
                                 )
        
        self.apply(weights_init_dcgan)
        
    def forward(self, x): 
        return self.net(x)

x = torch.randn((5,3,64,64))
z = torch.randn((5,100))
discriminatorcnn = DiscriminatorCNN64(16)
generatorcnn = GeneratorCNN64(100, 16)
# print(f'{generatorcnn}')
doutput = discriminatorcnn(x)
goutput = generatorcnn(z)
print(f'{doutput.shape=}')
print(f'{goutput.shape=}')
#%%
# lsgan constantly faces severe mode collapse(powerful discriminator)
# update: 
# we talk about this in full detail in the next part for now this suffices
# to know lsgan here doesnt work properly and faces mode collapse, for now
# and I tried to overcome it by playing with lr and discriminator capacity
# see experiment 20250913135137 we ultimately achieve fid 156 and no apparent mode collapse)
# wgangp however always does a better job! havent seen mode collapse in nearly
# 100 tests! this shows how stable wgangp is! to get the lsgan to
# not fail, we have to constrain the discriminator 
# wgan also fails for some reason! it was working need to see what
# I had changed!
# update:
# it seems only wgangp trains well. both wgan/lsgan face a lot of instablity
# and flatout fail by default! I guess to getthem to work one easier? way (not sure!)
# would be to use spectral norm to help keep 1lipcshitz condition during training!
# todo: check spectralnorm and see if it helps!
loss_type = 'wgangp'
# for wgangp /gradient penalty scaler lambda
lambda_factor=10#10 #5
# (with wgangp) for cifar10 up until epoch 17 we had many severe distortions
# blobs of colors, but eventually as more epochs elapsed we got better(using 
# smaller lambda(5) seems better but im not sure i need more experiments)
# celeba is much easier to train compared to ihghlt diverse cifar10!
# wgan faces the same issues, but do not recover from it up to the very end
# lsgan completely fails!
dataset_name = 'celeba'
batch_size=128
train_loader = get_dataloader(dataset_name=dataset_name, split='train',resize_dims=(64,64),batch_size=batch_size)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# since the generator is more powerful, and discriminator is
# less powerful we quickly face severe mode collpase, so 
# for lsgan to work we need to beefup the discriminator!
disc_hidden_size = 16 if loss_type=='lsgan' else 32
gen_hidden_size = 64
z_size = 100
# wgan and wgangp dont use batchnorm because it messes with the 1-lipschitz constraint
# as it introduces sample coupling(sample to sample coupling) while 1-lipschitz constraint
# requires each and every sample to conform to this. I however trained with batchnorm and 
# it seemed completely fine!
# sidenote:
# in small resolution (32x32) and short trainig it seems having
# batchnorm works, but without batchnorm, which should be the default
# it provides much better images much faster
use_batchnorm = loss_type=='lsgan'

epochs = 50
num_batches = len(train_loader)
interval = num_batches//2+1
# every 5 discriminator/critic updates, update the generator
# wgangp works fine with 1! wgan seems not!
gen_update_interval = 1 if loss_type == "wgan" else 1

#discriminator
discriminatorcnn64 = DiscriminatorCNN64(hidden_size=disc_hidden_size, 
                                        use_batchnorm=use_batchnorm)
discriminatorcnn64 = discriminatorcnn64.to(device)
#generator
generatorcnn64 = GeneratorCNN64(z_size, hidden_size=gen_hidden_size)
generatorcnn64 = generatorcnn64.to(device)

# the paper says [0.5,0.999] diverges in wgan/wgangp
# with bn=True especially for wgan/wgangp, lr must be larger (0.001/0.002)
# otherwise it will take a lot to get there!
# lsgan fails with large lrs(1e-3/2e-3)
betas = [0.5, 0.999] if loss_type=='lsgan' else [0, 0.9]

if loss_type=='lsgan':
    # by lowering the discriminator capacity I found these two work well forlsgan
    # so we dont face mode collapse
    lr_d, lr_g = 0.0001, 0.0004
else:
    lr_d, lr_g = 0.001, 0.002
    
disc_optimizer = torch.optim.Adam(discriminatorcnn64.parameters(), lr_d, betas=betas)
gen_optimizer = torch.optim.Adam(generatorcnn64.parameters(), lr_g, betas=betas)

training_loop(discriminatorcnn64, 
              generatorcnn64, 
              train_loader=train_loader,
              disc_optimizer=disc_optimizer,
              gen_optimizer=gen_optimizer, 
              epochs=epochs, 
              interval=interval,
              gen_update_interval=gen_update_interval, 
              dataset_name=dataset_name,
              loss_type=loss_type, 
              lambda_factor=lambda_factor,
              use_batchnorm=use_batchnorm,
              noise_addition=True,
              device=device)
#%%
# for lsgan 20250913135137
# load models 
checkpoint = torch.load("./weights/gan/dcgan_generatorcnn_wgangp_20250901064034.pt",
                        map_location="cpu",
                        weights_only=False)

epoch = checkpoint["epoch"]
z_size = checkpoint["z_size"]
hidden_size = checkpoint["hidden_size"]
dataset_name = checkpoint["dataset_name"]
loss_type = checkpoint["loss_type"]
losses = np.array(checkpoint.pop("losses"))

generatorcnn64 = GeneratorCNN64(z_size,hidden_size)
generatorcnn64.load_state_dict(checkpoint.pop("state_dict"))
generatorcnn64.eval()

for k,v in checkpoint.items():
    print(f'{k}: {v}')
    
print(f'DLoss: {losses[:,0].mean():.4f} | GLoss: {losses[:1].mean():.4f}')
#%%
run_latent_arithmatic(attr_name='Male', 
                      generator=generatorcnn64, 
                      classifier=celeba_classifier, 
                      word2idx=celeba_attr_word2idx,
                      random_gen=random_gen,
                      showcase_one_sample=True,
                      num_samples=64,
                      attribute_pool_size=256,
                      maximum_prob_for_neutral_confidence=0.1,
                      attribute_confidence_rate=0.7,
                      alpha_values=torch.linspace(-3,7,steps=24),
                      device='cpu')

#%%
run_latent_arithmatic(attr_name='Smiling', 
                      generator=generatorcnn64, 
                      classifier=celeba_classifier, 
                      word2idx=celeba_attr_word2idx,
                      random_gen=random_gen,
                      showcase_one_sample=True,
                      num_samples=32,
                      attribute_pool_size=256,
                      maximum_prob_for_neutral_confidence=0.1,
                      attribute_confidence_rate=0.8,
                      alpha_values=torch.linspace(-3,7,steps=24),
                      device='cpu')
#%%
run_latent_arithmatic(attr_name='Eyeglasses', 
                      generator=generatorcnn64, 
                      classifier=celeba_classifier, 
                      word2idx=celeba_attr_word2idx,
                      random_gen=random_gen,
                      showcase_one_sample=True,
                      num_samples=32,
                      attribute_pool_size=256,
                      maximum_prob_for_neutral_confidence=0.01,
                      attribute_confidence_rate=0.8,
                      alpha_values=torch.linspace(-3,7,steps=24),
                      device='cpu')
#%% as we can see we got much better images using wgangp and traiing 
# is much more stable than the other two methods! we also see better 
# feature entaglement, that is the individual stays the same and only
# a specific feature changes! not several ones. (like male2female or
# glasses vs no glasses, etc) compairng to previous attempt this one
# seems much better and more accurate the direction corrosponds to one
# concept overal it seems (more accurate/less noisy that is)
# 
#%%as we see, not only things havent improved, we have faced severe issues
# to the point nearly all methods either fail completeley or require a lot 
# of adjustment to make it work! 
# now lets correct our mistakes in previous sections and hopefully get a much
# better result. 
# the first issue is the bottleneck design in our ConvBlock2, we intended on
# getting a richer representation, but made it worse! our bottleneck causes
# massive information loss in the discriminator. the discriminator needs all
# the information it can get to be able to find subtle artifacts in the generated
# images so the feedback works properly! however, by squeezing the featuremaps
# using a bottleneck(i.e. out_channels//2) we are forcing the network to discard
# alot of information(very high frequency/subtle details) that can help distinguish betwene
# the real and fake images. this is why the normal/wider convolution(ConvBlock) works
# much much better as it does not throw away any bits of information!
# the discriminator's job is to detect artifacts and alot of common ganartifacts (e.g. repeating patterns,
# unnatural textures, blurry areas where there should be sharp details, sparkly noise, etc)
# are inherently high frequency! so rule number 1 Do not use any bottlenecks in our design!
# 
# sidenote:
# a good way to internalize this is to imagine each channel/featuremap as a pattern detector
# suppose its a word describing a concept (its not a prefect analogy but keep going it makes snese i promise),
# and a complex image, would therefore require a large bag/vocabulary of these pattern detectors
# to be fully and properly described. the more words/detectors/experts we have at our disposal,
# the better we can describe our input. for example 
# channel 1 might learn to activate for simple things like horizontal lines
# channel 2 might learn to activate for a more specific thing like green, leafy textures
# channel 3 might learn to activate for fluffy textures (like dog fare, hairs, etc)
# ....
# channel 256 might learn to activate on subtle, noise-like artifacts that is common in fake images!
# and so on. as you can see, each channel can be regarded as an expert of sort in detecting very specific
# thins in the input. its as if we have reports coming from a commitee of experts! and each have written 
# a one-page report (i.e. a feature map) on the subject (i.e. input image).
# some reports are about low-frequency information like "overall, the subject is a human face" (these are experts/specialists in macro-shapes for example).
# many other reports are about high-frequency information like "there is a sharp reflection of light/glint on the left pupil!"
# or "the hair texture shows unnatural repetition in this specific area" or "the edge between the chin and the background
# is slightly blurry" (these are experts/specialits in fine details, textures (high frequency information) forexample).
# so when we discard some of these channels in a bottlenck we are effectively discaring finetuned pattern detectors/experts/words
# in our vocabulary which would directly hurt our ability to describe the input image properly. 
# noe that we are doing this in a very crude and inefficent way as well which compounds/exacerates/intensifies/ the issue further!
# 
# sidenote2:
# the whole job of a discriminator/critic is to be an expert in finding issues in the input image (an expert forensic!).
# so it must find the absolute smallest almost impreciptible falws/issues in the generator's output!
# these flaws are almost always highfrquency artifacts which include:
# slightly incorrect textures, unnatural sharpness or blurriness on an edge, checkerboard patterns, repetitive noise!
# and many more!
# 
# sidenote3:refresher
# low frequency represents the slow changes in pixel values across the image.
# things like large structures, overall shapes, general lighting conditions, 
# broad color gradients, smooth transitions, things of this nature where change is gradual and smooth
# basically there are no sudden/abrupt changes(like the overall shading of a face or the smooth blue of the sky)
# conversly high frequency represents the rapid/sudden changes in pixel values over "short" distances!
# thingslike edges, textures, fine details, noise, sharp transitions, things like that which change
# suddenly in few pixels apart (e.g. an edge is basically few bright pixels next to few darker pixels, 
# we see a sudden change in a short distance). a lot of high frequency stuff/components are abrupt 
# (like for example the sharp edge between an object and its background) but not all of them!
# not all high frequency details are necessarily abrupt in the sense that a hard edge is!
# things like fine textures like hair strands, fabric weaves, or skin pores! are all examples of this. they are 
# not necessarily abrupt like an edge, but they involve rapid, small changes in pixel values over a small area
# these are the critical "subtle details"! we want! anothe example is the film grain/noise, the the subtle noise 
# or the film grain are also high frequency! likewise small imprefictions, like a tiny spec, a small wrinkle,
# a subtle glint, stuff like these are all rapid changes in intensity of pixel values in a small area.
#
# note we can have subtle high frequency or subtle low frequency details, itsnot like high frequency refers to subtle details
# or vice versa! 
# for example subtle high frequency details can be like the slight variations in the texture of a brick wall,
# or the shimmer on a piece of silk, or the fine lines on an aged face! these are subtle, and they are high 
# frequency as well because they involve rapid changes over small regions!
# likewise subtle low frequency details are like a very slight, almost imperceptible shift in the ambient lighting 
# across the entire scene, or a very gentle broad color gradient that adds depth! as you can see these are also subtle
# but they are low frequency because they change gradually over large regions!
#
# when we try to compress the input using a bottleneck, the network needs to priorize what to keep and what to discard 
# and it usually prioritizes the most important/noticeable features for distinguishing real images from the fakes. 
# while this includes hard edges its the finest high frequency textures and subtle variations that are often the first
# to be lost or poorly represented! as they require more specific detectors/channels to capture.
# conversely for the generator to produce realistic images it needs to synthesize these same high-frequency details
# beautifully and naturally. If the discriminator isnt sensitive enough to them, the generator wont be sufficiently 
# penalized for missing them, leading to smoother, less detailed, or artifact-ridden outputs.

# GAN training by nature is very unstable, and we adding multiple layers of convolution with batchnorm in specific 
# way (like bottleneck,residual etc) into a single block doesnt help either. not only that but by adding many more
# moving parts (i.e. need to be adjudsted) we create complex and very prolematic/abnormal gradient dynamics that will
# be very hard for the optimizer to deal with. by problematic gradient dynamics I mean, exploding/vanishing gradients
# stuff of that nature(because we disable batchnorm this will complicate things see debug log ahead).
# this is why simplicity is much favored!
# 
# The other glaring issue that we see in our ConvBlock botth the original and the second version, is the use of normalization
# more specifically BachNorm! although we designed it so we can omiit the Batchnorm, simply because it messes with the
# 1lipschitz constraint thats required for successful training of gan especially for WGAN/WGANGP. (batchnorm normalizes across
# the whole batch and creates dependencies between samples but we want per sample 1lipschitz constraint to be enforced so it
# the 1lipschitz constraint), but its not going to help much as we have already seen. 
# some simply use other forms of per sample normalization like InstanceNorm or LayerNorm isntead
# of batchnorm, but that itself is not going to work much better! the best way thats 
# been found to help with this is to use Spectral Normalization to keep 1lipschitz conditions all the time at all layers!
# with this simple change we shouldbe able to have a way better training and end result.
# so rule number two is to use Spectral Norm instead of Batchnorm in the discriminator.
# 
# sidenote:
# we said earlier that discriminator needs to be simple or otherwise it will dominate and we will face mode collapse
# but alsonote that when we also may face depective mode collapse which is for when generator is more powerful and it
# creates high quality images yet without diversity of low diversity, in this case we need to have a potent/powerful 
# disciminator to be able to distinuish tiny differces and force the generator to explore more and have more diversity!
# we will talk about these in more detail in the next part. 
#
# quicknote:
# when we apply spectral norm, the training will get a massive hit in performance as this imposes a heavy(but constant)
# overhead. because pytorch needs to run an iterative algorithm called Power Iteration at least once(default is n_power_iterations=1)
# for every layer wrapped in spectral_norm in every forward pass, to estimate the largest singular value of that layer's weight matrix.
# This involves extra matrix-vector multiplications that are not part of the standard forward pass but are
# required for the normalization process!
# 
# these two changes should rectify our discriminators issues. nowlets look at our generators ConvTransBlock blocks.
# The first obvious issue is the bottleneck design, it needs to go. we already covered it. the second obvious reason
# the way the residual is added to the main path. we already saw if we apply activation functions like relu to the mix
# it just destroys the generation! the reason is we have done it incorrectly! we have applied the activation
# function on the main path (see the self.block) and then add the residual information to the output. 
# suppose the main path contains negative values, when we incorporate relu for example, it destroys the negative values,
# zeros them out, then we add the residual nformation, it would be like 0 + residual, the gradient cant flow back
# throuh the main path in this case, so we effectively created a broken pathway! when we add a relu on top it, it 
# exacerbates the issue even further and generation fails completely. so the idea is to not apply activation function
# before we incorporate the residual information. that is the self.block in our ConvTransBlock shouldnt have activation
# functions, only conv+bn, and then the raw logits needs to be added to the residuals and then carry on with another
# activation on top. 
# todo recheck 
# note that if even doing that we dont get proper issue, its probably because the two inputs are somehow so differently
# processed that when we add them we get large negative values, or if not, its because the distribution
# is far from zero in which case, relu is not a good fit and would destroy this information. 
# also the addition to residual connection for the final layer will destroy the generation so we dont dothat for 
# the final layer

# the other issue, a bit minor compared to previous ones, is the use of ConvTranspose2d with strides of 2 and kernel size thats 
# divisble by the stride (i.e. stride=2 and kernel_size=4). this creates a checkerboard artifact in the generated images. we can 
# get around this by simply replacing ConvTranspose2d with Upsample layer and a normal conv layer!
# Having all of this said, now lets apply the changes
# 
class DiscConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=2, padding=1, use_spectral_norm=True, act_func=nn.LeakyReLU(0.2), dropout_rate=0.2):
        super().__init__()
        
        # add spectral norm to keep 1-lipschitz constrain everywhere, 
        # its crucial for wgan/wgangp but all other agns also benifit as well!
        # its a must have!
        self._spectral_norm = lambda m: nn.utils.spectral_norm(m)
        conv = nn.Conv2d(in_channels=in_channels,
                         out_channels=out_channels,
                         kernel_size=kernel_size,
                         stride=stride,
                         padding=padding)
        self.conv = self._spectral_norm(conv) if use_spectral_norm else conv
        self.act = act_func
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout2d(dropout_rate)
    
    def forward(self, x):
        # this is to constrain t he discriminator so it doesnt overpower the generator!
        return self.dropout(self.act(self.conv(x)))
        
class DiscriminatorImproved64(nn.Module):
    def __init__(self, hidden_size=32, act=nn.LeakyReLU(0.2), no_spec_norm_list=None, dropout_rate=0.2):
        super().__init__()

        self.hidden_size = hidden_size
        self.act = act
        self.dropout_rate = dropout_rate
        # list of layer indexes specifying which
        # layer gets its spectral normalization disabled!
        # by default all affine layers (here 5) are normalized 
        # for maximum stability. but sometimes its not needed 
        # so this is to exempt some layers so we get better performance
        # see notes ahead!
        if no_spec_norm_list is not None:
            self.spec_norm_list = [not (i in no_spec_norm_list) for i in range(5)]
        else:
            self.spec_norm_list = [True]*5
        assert len(self.spec_norm_list) == 5, f'spectral_norm_list doesnt match network layer count({len(spec_norm_list)}!=5)'
        print(f'{self.spec_norm_list=}')
        
        # test with no spectral norm and see how it goes
        # update:
        # it seems applying spectral norm to every layer doesnt necessarily
        # result in better output! at least for wgangp! (no_spec_norm_list=[1,2])
        # okay, we can omit the spectal normalization from the first and last (linear) 
        # layers, without much worry. I did the opposite! kept the first and the last
        # two layers with spectral norm and ommit the others. training went well and I
        # got much better result (well formed clear images) and lower FID using wgangp
        # basically if the training is stable, we can try ommiting spectral normalization
        # from layers. conv with spectral norm, seem to be doing enough regularization that
        # doesnt need all layers to be exactly 1lipschitz! 
        # I need to test more and I'll update my notes
        # update2:
        # the last layer is highly recommened to have spectral norm so the network output is
        # 1lipschitz. but if the training is stable it can be omited (the leading convlayer
        # with spectral norm usually take care of it well enough)
        # the idea is first to apply spectral norm to all layers since this provides the most 
        # stability, and when you get a baseline, try removing some to get more speed etc
        #update3:
        # for lsgan the config that worked for wgangp doesnt work!imediately we see mode collapse!
        # in epoch1, it continues in epoch 4 and images are hellish!
        # its a disaster for wgan as well! up until epoch 3 we have very bad (blurry/malformed
        # some have dark blobs in them not good basically!
        self.net = nn.Sequential(DiscConvBlock(3, hidden_size, 4, 2, 1, act_func=act, use_spectral_norm=self.spec_norm_list[0], dropout_rate=dropout_rate),#32x32
                                 DiscConvBlock(hidden_size*1, hidden_size*2, 4, 2, 1, act_func=act, use_spectral_norm=self.spec_norm_list[1], dropout_rate=dropout_rate),#16x16
                                 DiscConvBlock(hidden_size*2, hidden_size*4, 4, 2, 1, act_func=act, use_spectral_norm=self.spec_norm_list[2], dropout_rate=dropout_rate),#8x8
                                 DiscConvBlock(hidden_size*4, hidden_size*4, 4, 2, 1, act_func=act, use_spectral_norm=self.spec_norm_list[3], dropout_rate=dropout_rate),#4x4
                                 nn.Flatten(),)
        
        self.fc = nn.Linear(hidden_size*4 * 4*4, 1)
        # we must also normalize the linear layer as well.(its highly recommended)
        # basicaly all layers that perform major affine transformations
        # like Conv and Linear should be normalized. see notes below
        # self._spectral_norm = lambda m: nn.utils.spectral_norm(m)
        if self.spec_norm_list[4]:
            self.fc = nn.utils.spectral_norm(self.fc)
        
        # the weight initt is still very important,
        # the dcgan weight init makes things more stable!
        self.apply(weights_init_dcgan)

    def forward(self, x):
        output = self.net(x)
        return self.fc(output)
    
# and for generator
class UpsampleBlock(nn.Module):
    def __init__(self,  in_channels, out_channels, kernel_size,
                 stride=1, padding=1, batch_norm=False, act_func=nn.ReLU(inplace=True)):
        super().__init__()
        
        self.act_func = act_func
        # by using upsample we will see very sharp/glossy images!
        # note the changes result in higher quality images regardless
        # of using upsample vs convtranspose. that is our improvement 
        # is not only due to using upsample. if you try to use convtranspose
        # you'll see we get roughly the same performance
        self.block = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'),
                                   nn.Conv2d(in_channels, out_channels, kernel_size,
                                             stride=stride, padding=padding, bias=not batch_norm),
                                   nn.BatchNorm2d(out_channels) if batch_norm else
                                   nn.Identity(),
                                   # no activation for the block! when we wantto incorporate residuals!
                                   )

        self.residual = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'),
                                      nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                                stride=1, padding=0, bias=not batch_norm),
                                      nn.BatchNorm2d(out_channels) if batch_norm else
                                      nn.Identity(),
                                      )
 
    def forward(self, x):
        out = self.block(x)
        x_res = self.residual(x)
        # now we have raw logits for both, so we add and then apply activation
        # update: adding stylegan factor (sqrt(2))
        # todo remove the residual connection here and add it in the generator
        # between different blocks (create feature pyramid in fact!)
        out = self.act_func(out + x_res) #* (1 / 1.414)
        return out

class GeneratorImproved64(nn.Module):
    def __init__(self, z_size, hidden_size,  act=nn.ReLU()):
        super().__init__()

        self.z_size = z_size
        self.hidden_size = hidden_size
        self.act = act
        self.net = nn.Sequential(nn.Linear(z_size, hidden_size*4 * 4*4),
                                 nn.BatchNorm1d(hidden_size*4 * 4*4),
                                 # note since our generator isnt traversed more than once, having inplace operation is ok
                                 # later on we will see cases where this is not the case e.g. in cyclegan! just wanted to
                                 # put this notice here before I forgetit!
                                 nn.ReLU(inplace=True),
                                 nn.Unflatten(dim=1, unflattened_size=(hidden_size*4, 4, 4)),
                                 UpsampleBlock(hidden_size*4, hidden_size*2, 3, batch_norm=True, act_func=act), #8x8
                                 UpsampleBlock(hidden_size*2, hidden_size*2, 3, batch_norm=True, act_func=act), #16x16
                                 UpsampleBlock(hidden_size*2, hidden_size*1, 3, batch_norm=True, act_func=act), #32x32
                                 )
        # our final layer/block doesnt have any residual to mess with the final output!
        self.final_layer = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'), #64x64
                                         nn.Conv2d(hidden_size, 3, 3, stride=1, padding=1),
                                         nn.Tanh())
        
        self.apply(weights_init_dcgan)
        
    def forward(self, x): 
        output = self.net(x)
        return self.final_layer(output)

x = torch.randn((5,3,64,64))
z = torch.randn((5,100))
disci64 = DiscriminatorImproved64(16, no_spec_norm_list=[0])
# print(f'{disci64.spec_norm_list=}')
geni64 = GeneratorImproved64(100, 16)
# print(f'{disci64=}')
doutput = disci64(x)
goutput = geni64(z)
print(f'{doutput.shape=}')
print(f'{goutput.shape=}')
#%% 
# with the new changes in our architecture, we see much sharper/glossier images
# which are due to using upsample layer. the trainig is a bit more stable
# overall even without spectralnorm. with spectral norm it gets more stable but
# it wont provide realistic images right off the bat! its the job of the architecture
# and training regime!
#
# update:
# compared to our previous attempt, it seems the wgan/lsgan did massively better
# when we use Batchnorm! which we know is wrong! but then again when we used spectral norm
# things didnt get better! the wgan images are not formed, have black blobs, blurry images
# and it plagues all images!
# this means our discriminator is way more powerful than our generator! the bn works because
# it indirectly acts as a powerful regularizer and it blurs the signal by accumulating
# the statistics of all samples in a batch!( it makes the the statistics of the features
# to be dependent on the whole batch which blurs the signal and artificially slows the 
# discriminator down and prevents it from winning too quickly, bn also allows for considerablt
# larger lr and convergence gets faster if things go smoothly! thats why without it even picking a lr
# and even the range of values for the weights become issues and things to tune! which complicates things
# further) hence we get better results there than here!
# so I added a dropout layer after each conv layer to make discriminator more constrained!
# update2:
# after the change this is what I got at epoch 1:
# Epoch/Epochs: 1/50 | Iter: 636/1272 | Disc Loss: 0.101219 | Gen Loss: -1.977101
# Epoch/Epochs: 1/50 | Disc Loss : -1.102075 | Gen loss: 0.781875 | IS: (μ:2.3388, σ²:0.2658) | FID: 222.38
#  -- Discriminator's real mean: 0.0014 | Discriminator's fake mean = -0.0431
# the discriminators mean for real images and fake images look very bad 
# the wgan is supposed to learn to assign big positive numbers(big scores not huge 
# (1 or 2 digit are ok but larger or 3 digits and more are bad)) to real images and
# much smaller numbers to fake ones(the difference between these two numbers must be large enough to have good distinction distance between real/fake)
# but here we see real mean(average score for real images) = 0.0014 and fake mean = -0.0431 which means 
# the difference between the two (real_mean - fakemean = 0.0014 + 0.0431= 0.0445 which is practically zero! 
# in other words the discriminator is telling us it has absolutely no idea which image is real 
# and which one is fake! now because the output of the discriminator is nearly identical for both
# real and fake images, the gradients it passes back to the generator are extremely small! therefore
# the generator has practically no strength or will to change or improve! also the black blobs and warped faces
# we kept seeing in images is a visual sign/manifistation of generator reciving zero gradient signals! 
# the reason is simply because how wgan works. this is one of the reason wgan weight cliping is
# deprecated and wgangp is used. weight clipping can either cause vainishing gradient or exploding gradient
# depending on the value we choose for clipping! in our case it seems we are clipping a small number
# and maybe using a larger one might help! 
# update:
# I increates the clipping range from -0.01,0.01 to -0.05,0.05 and immediately got better results!
# the discriminators average score became much larger (Discriminator's real mean: 5.0224 | Discriminator's fake mean = 4.4442)
# there were still some black blobs, but the image qualy became 10x better! when I 
# disabled the spectral norm the quality decreased but the training seems stable.
# 
# (quicknote: 
# note that is score/fid are very noisy(because we didnt run them on large number of images, we just
# used single batch each time to get a rough idea) so for detailed debugging we need to look at other
# clues to be sure of what is ok and whats not)
# 
# Epoch/Epochs: 2/50 | Iter: 636/1272 | Disc Loss: -15.749696 | Gen Loss: -12.988724
# Epoch/Epochs: 2/50 | Disc Loss : -20.169931 | Gen loss: -12.215222 | IS: (μ:2.0913, σ²:0.4107) | FID: 290.18
#  -- Discriminator's real mean: -10.7830 | Discriminator's fake mean = -30.3853
# 
# as you can see the average score and the difference is good! toward the end of the training the scores
# improved, suggesting with more training we will get better
# results:
# poch/Epochs: 49/50 | Iter: 636/1272 | Disc Loss: 0.044365 | Gen Loss: 23.883381
# Epoch/Epochs: 49/50 | Disc Loss : -4.688194 | Gen loss: 35.643920 | IS: (μ:1.9881, σ²:0.3252) | FID: 153.03
#  -- Discriminator's real mean: 3.3240 | Discriminator's fake mean = 0.7945
# 
# Ive heard we shouldnt use spectral norm with wgan! they are both different regularizer so for wgan tests 
# we need to disable spectral normalization! but in my experiments it seems having it around helps!
# and it shows in the average score we get, but I need more experiments to say it forsure)
# (sidenote: switching to rmsprop with gen_interval=5 did a good job while adam kept failing see
# my explanation ahead)
# 
# sidenote: 
# we need to use lower rangers (i.e. (-0.02,0.02),(-0.03,0.03)) if both work, we usually go for 
# the smaller range! because larger values can result in unstable rtaining (cause exploding gradients)
# see wgan debugging section ahead where I disected the training to findout the issue which was directly
# related to large wgan clipping range (i.e. -0.05,0.05) and large lr!
#
#  
# lsgan faces mode collapse here which means the discriminator is still more powerful. see the debugging
# explanation ahead to see how we fix this issue.
# 
# sidenote:
# concerning mode collapse, we can have an analogy like this, a strict art teacher (i.e. the discriminator) 
# is checking the student's assignment. the teacher says everything you've done is terrible, except
# for this one simple sketch of a cat! that one is almost okay! the student who is desperate for passing
# the grade! stops trying to draw and paint anything else (like landscape or portrays etc) and instead
# just keeps drawing many copies of the same cat sketch that the teacher found ok! this is what sums the mode
# collapse we witness. 
# basically this happens when the generator is weak and hasnt learned the full data distribution yet. 
# the generator therefore collapses on a half formed image because its the only thing the pwoerful/strict
# discriminator doesnt immediately reject! most of the times this happens early in the training as the 
# discriminator quickly learns whats real and whats not basically establishing its dominance overpowering the generator completely!
# the generator therefore will be punished into a corner where it only produces a limited number of things
# that the discriminator doest reject! (so this issue comes from failure of exploration by generator which
# is driven by (fear of) discriminator rejection!). note that it maynot be that the discriminator prefectly
# knows whats real/fake, rather it might be it has become overconfident and hyper-specific about certain things
# and it has learned to reject everything except the one/few mediocre modes. 
#
# powerful generator causes deceptive collapse(i.e. high quality repeative image later in training):
# now as I previously pointed out, a powerful discriminator can cause mode collapse, but sometimes, if
# the generator is too powerful, the dynamic changes! its like a con artists/expert forgerer(i.e. powerful generator)
# goes to a pawn shop owner(discriminator) and shows him a fake rolex! the shop owner is completely fooled
# and buys it! the con artists/forgerer sees how easy it is to fool the owner, and doesnt bother creating any
# other brands, directly goes and keeps creating the same fake rolex and sells it to the shop owner every day!
# and the shop owner falls for it every single day! 
# in other words the generator is powerful and has learned to create very good images. it collapses because
# it found a perfect forgery and has no incentive to create other different perfect forgeries.
# this is the second case where we can face mode collapse.
# so when the generator is too powerful, the samples are high quality yet repeatative, and it can happen
# later in the training after the generator has had time to explore and then finds its winning,
# repeatable strategy whereas when the disciminator is too powerful we have low quality repeative samples
# happening early in the training! this happens because the generator exploits a weakness! 
# its also a failure of exploration but this time its driven by finding a lazy/easy win!(i.e. the generator
# gets stuck because it has found a winning strategy against a weak opponent and has no incentive to diversify!)
# (the discriminator is underpowered here and is consistently fooled and can not learn to detect the 
# repeated high quality forgery!)
# 
# so to recap: if we see blurry/malformed or simplistic faces/images being repeated over and over our
# first guess should be an overpowered/powerful discriminator and our first step should be to to weaken it
# by adding dropout, lower its learning rate, introduce noise or make the  generator more powerful!(lower disciminator optimization round compared to generator!)
# on the other hand if we see sharp beautiful faces/images being repeated it means we have a weak/underpowered
# discriminator or maybe its powerful but learning very slowly for some reason (small lr e.g. requires more otimization updates against generators).
# therefore the first step would be to try make the discriminator more powerful by increasing its 
# capacity(layer/hidden_size,etc) or make it faster by increasing its learning rate or optimization iterations
# so it can detect the generators fake images and force it to to learn new capabilities and explore
# more and hopefully comeup with more diverse outputs/generations!
# 
# wgan is not recommened at all! just go with wgangp! 
#

print(f'Training Improved versions of Discriminator and Generator')
loss_type = 'wgangp'
lambda_factor=10
dataset_name = 'celeba'
batch_size=128
train_loader = get_dataloader(dataset_name=dataset_name, split='train',resize_dims=(64,64),batch_size=batch_size)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
disc_hidden_size = 32
gen_hidden_size = 64
z_size = 128

epochs = 50
num_batches = len(train_loader)
interval = num_batches//2+1
# every 5 discriminator/critic updates, update the generator
# wgangp works fine with 1! wgan seems not! wgan needs more
# updates so it learns what what real and fake images are 
# in practice though,when i set this to 5, wgan goes nuts!
# and it fails spectacularly! this could be due to large lr(0.001)
# so I need to try lower lr as well, it could also be due to adam momentum
# ruining it, as the main authors uses rmsprop! but on the other hand
# in my previous experiments,we had much better luck!(using bn of course!)
# (using larger clipping range (-0.05-0.05) stablized adam. I also reenabled
# noise addition, aside from adding dropout to everylayer in critic/discriminator
# to get this to work! see updates above!)
# 
# I disabled noise addition to see how much impact it had on our result ignoring
# all other factors(larger clipping range and more constrained discriminator)
# the first attempt failed completely and the model couldnt recover!(experiment 20250911142646)
# the second time it did a tiny bit better, but still the outcome is terrible! (experiment 20250911175046)
# third experiment 20250911200109: the same things. (we might see a bit better runs but overall they are garbage!(20250912070341))
# here is the logs: 
# 
# Epoch/Epochs: 2/50 | Iter: 636/1272 | Disc Loss: -73.066040 | Gen Loss: -239.430008
# -- Current Batch: Discriminator's real mean: 216.3880 | Discriminator's fake mean = 143.3220
# Epoch/Epochs: 2/50 | Disc Loss-Avg: -15.158274 | Gen loss-Avg: -40.854735 | IS: (μ:1.8912, σ²:0.1763) | FID: 337.50
# -- Last Batch : Disc's real mean: 100.5570 | Disc's fake mean: 52.4385
# -- Epoch's Avg: Disc's real mean: 72.6963 | Disc's fake mean: 57.5380
# 
# Epoch/Epochs: 5/50 | Iter: 636/1272 | Disc Loss: -69.714111 | Gen Loss: -331.979858
# -- Current Batch: Discriminator's real mean: 431.1464 | Discriminator's fake mean = 361.4323
# Epoch/Epochs: 5/50 | Disc Loss-Avg: -25.437134 | Gen loss-Avg: -144.851538 | IS: (μ:1.9750, σ²:0.2361) | FID: 349.54
# -- Last Batch : Disc's real mean: 261.0208 | Disc's fake mean: 303.9453
# -- Epoch's Avg: Disc's real mean: 185.7275 | Disc's fake mean: 160.2903
# 
# Epoch/Epochs: 8/50 | Iter: 636/1272 | Disc Loss: -142.536316 | Gen Loss: 32.370567
# -- Current Batch: Discriminator's real mean: 12.7788 | Discriminator's fake mean = -129.7575
# Epoch/Epochs: 8/50 | Disc Loss-Avg: -20.678199 | Gen loss-Avg: -78.778972 | IS: (μ:2.0957, σ²:0.2550) | FID: 307.89
# -- Last Batch : Disc's real mean: 455.5588 | Disc's fake mean: 103.0827
# -- Epoch's Avg: Disc's real mean: 112.3258 | Disc's fake mean: 91.6476
# 
# Epoch/Epochs: 15/50 | Iter: 636/1272 | Disc Loss: 61.262222 | Gen Loss: 351.622009
# -- Current Batch: Discriminator's real mean: -144.0047 | Discriminator's fake mean = -82.7425
# Epoch/Epochs: 15/50 | Disc Loss-Avg: -43.587865 | Gen loss-Avg: 115.796932 | IS: (μ:2.2103, σ²:0.3161) | FID: 314.68
# -- Last Batch : Disc's real mean: -848.5926 | Disc's fake mean: -647.2932
# -- Epoch's Avg: Disc's real mean: -49.0232 | Disc's fake mean: -92.6111
# 
# Epoch/Epochs: 29/50 | Iter: 636/1272 | Disc Loss: -646.508789 | Gen Loss: 7939.584473
# -- Current Batch: Discriminator's real mean: -7306.2427 | Discriminator's fake mean = -7952.7515
# Epoch/Epochs: 29/50 | Disc Loss-Avg: -127.489522 | Gen loss-Avg: 1746.615550 | IS: (μ:2.0884, σ²:0.1529) | FID: 281.29
# -- Last Batch : Disc's real mean: -2222.5659 | Disc's fake mean: -1898.1831
# -- Epoch's Avg: Disc's real mean: -1534.8222 | Disc's fake mean: -1662.3117
# 
# Epoch/Epochs: 30/50 | Iter: 636/1272 | Disc Loss: 237.401855 | Gen Loss: -2123.584473
# -- Current Batch: Discriminator's real mean: 3024.0740 | Discriminator's fake mean = 3261.4758
# Epoch/Epochs: 30/50 | Disc Loss-Avg: -187.184406 | Gen loss-Avg: 1911.559015 | IS: (μ:2.1577, σ²:0.2889) | FID: 299.63
# -- Last Batch : Disc's real mean: 45.1980 | Disc's fake mean: -3266.8750
# -- Epoch's Avg: Disc's real mean: -1674.6901 | Disc's fake mean: -1861.8746
# 
# Epoch/Epochs: 31/50 | Iter: 636/1272 | Disc Loss: -967.682251 | Gen Loss: 156.541016
# -- Current Batch: Discriminator's real mean: 758.2507 | Discriminator's fake mean = -209.4315
# Epoch/Epochs: 31/50 | Disc Loss-Avg: -205.252820 | Gen loss-Avg: 1846.286319 | IS: (μ:2.0962, σ²:0.2626) | FID: 286.81
# -- Last Batch : Disc's real mean: -2306.6543 | Disc's fake mean: -1883.2412
# -- Epoch's Avg: Disc's real mean: -1587.8289 | Disc's fake mean: -1793.0817
# 
# what stands out in our log, consistently, is that the average scores are in in hunderds and thousands!
# which is crazy! we basically want 1 digit or two digit average scores (real image mean/fake image mean)
# but we get huge numbers! in wgan, the numbers should be close to the clipping values, but we are way off!
# this means we are facing a violant exploding gradients and as a result complete divergence!
# this is the other side of wgan with weight clipping, we now saw both vanishing gradient and exploding ones!
# 
# as early as epoch 2 we see the loss is high, and the average score we get is 72 vs 57! this is very high as
# we just mentioned we expect the values to be close to the clipping range, at most 1 digit and small 2 digits
# at epoch 5, we see the last batch has average score 261 vs 303! which should be the opposite!
# the network has flipped and assigned higher score to the fake images than the real one!
# this is a sign of a bad update that has pushed it into a nonsensical state for that batch. 
# the overall epoch still looks normal(epochs average score 185 vs 160). we continue to epoch 8 
# where we see seemingly the model has done a good job it scored 12 for real and -129 for fake images! 
# but for the last batch we see 455 vs 103! these are huge positive numbers!!
# this means the discriminator's internal state is changing wildly that goes from small correct scores
# (i.e. healthy) to an unhealthy huge explosive scores all with in the same epoch!
# this means the the optimizer is taking very large (and uncontrolled) steps (but its not the only cause behind this)
# a few epochs in, at 15, we can see the discriminator has another major failure, and we have
# -144 vs -82 which went on to become -848 vs -647 in the last batch! its so massive the average 
# is now negative as well! the optimizer's update/reaction has thrown all the scores deep into negative
# territory. the whole system is swaying from one extreme to another! 
# in epoch 29, things get worse, the scores are -7306 vs -7952.7515 the order is correct but
# they are now in the thousands! thousands! which is insane! the gradients are so large at this point
# that the training process loses its meaning! the generator is getting a gradient signal proportional 
# to +7952 which is an insanely massive update that will throw the weights into a completely random
# new state. the gradients should be small enough so they can cause healthy changes in weights not massive 
# updates that scramble everything! things get even worse, in epoch 30 we see the discriminator flips again
# and gives 3024 vs 3261 which then at the last epoch flips back to 45 vs -3266 which is wild! it seems 
# as though the discriminator is changng its opinion on real vs fake on a whim! the ossiliation is also wild! 
# the next epoch shows we go from 3000 in epoch30 to around -3000 in epoch 31! 
# all of this shows we have very bad and violent gradient explosions. the gradient values are so massively 
# large they can no more be used to have meaningful updates, they look like massive floods that ruine/flip
# everything upside down! the optimizer also takes massive uncontrolled steps because of these massive gradients
# (the lr is large, but the massivegrdients compounds this) this is what causes the scores to swing 
# from +3000 to -3000 for example.
# all of this points to discriminator not being satble at all, in fact far from it so much so it never
# has a chance to learn a consistent and meaningful distance function because its weights are constanly
# changed/scrambled completely by these massive updates!
# the generator therefore also rceives useless information because the gradients it receives are not going 
# to guide it toward improvement i.e. making better faces! they are just chaotic noise at this point that 
# only tell it to react to the discriminators latest random state! whatever it happens to be at the moment!
# the FID and IS scores also show us tha no learning is happening! but we needed debugging details to
# know whats going wrong! we now fully know what has gone wrong!so how do we fix this? the simplest thing
# would be to use a much lower learning rate and use smaller range of values for clipping and ultimately 
# use noise addition!
# 
# update: 
# by switching to wgan_range=(-0.02, 0.02) and keeping everytihng else the same as before. we have much
# stable training. no sign of gradient explosion! and average scores are 1 digit and small two digits!
# but the results are still garbage! the network still confuses real vs fake but no explsion as of yet
# simply lowering the lr alone without lowering the clipping range wouldnt work. clipping range must
# be set lower to not face exploding gradients!(this reminds of of the early days of dl where bn wasnt
# introduced, we had a lot of issues like!)
# next we are going to lower the lr to 1e-5 with wgan_range=(-0.02,0.02): the avergae scores are one 1 digit
# as the should be, and everything seems stable. but since the lr is very low, the convergence rate is
# understandably very slow as well and the default 50 epochs wont be enough to get good results(experiment 20250912092645)
# next we are goingto increase the lr back to 0.001, keep the range small (-0.02,0.02) but enable noise
# addition( experiment 20250912112621) the values are larger than the previous experiment, and we see large
# average scores (in 100/200, not larger but still is very bad. we can see the swinging here as well, 
# the initial batch has e.g. an average score of 286, and in last batch it goes down to 3! the jump is too 
# large which suggest the update step is very large so large it doesnt allow the discriminator to settle on
# a proper weight. it jumps around hit the limit(clipping) gets thrown to ther otherway harshly and this
# continues. basically the original issue but a bit milder because the clipping range is much lower than 
# before. therefore the result is not good at all! reverting back the lr back to 1e-5 made
# stuff better but at the same time convergence speed is slow!(experiment 20250912124800). 
# remove noise addition will make the resuls worse so having it around is good(20250912144741): 
# next switching to rmspropm with lr=5e-5 (wih no noise addition): performs kindof the same but I guess
# maybe a bit better?!
# now trying with spectral norm on all layers (we shouldnt do this but lets do it anyway):(20250912173350)
# itimproved the results but the result overall is very bad (fid 200+!)
# next enable noise addition, spectnorm and use larger clipping weights (-0.05,0.05) with rmsprop:(20250912191703)
# now with no spectral norm: it fails. so spectral norm keeps the training stable but the result
# is no way near as good as wgangp! also with larger range the quality is a tad better than before
# all in all, wgan really isnt worth spending our time. its extremeley inefficient and tricky to get
# to work and even then it doesnt produce a decent output! especially when we can use wgangp!
#  
# this means after we removed the noise addition, the clipping range is just too large, so large 
# that it doesnt enforce 1lipschitz and causes gradient explosion!
#
# switching to rmsprop actually did improve things with interval=5
# and it got better as training went (got fid 167 in epoch 50)
# but I still need to check other things! (like lower interval etc)
# using gen_interval=1 gives clearer images, but I see mode collapse!
# as earl as epoch 3 and it gets more obvious in epoch 6/7 so for wgan
# and average score shows this clearly as well :
# Epoch/Epochs: 7/50 | Iter: 636/1272 | Disc Loss: 0.032342 | Gen Loss: -0.381364
# Epoch/Epochs: 7/50 | Disc Loss : -0.022526 | Gen loss: -0.224257 | IS: (μ:1.6361, σ²:0.1471) | FID: 160.45
#  -- Discriminator's real mean: 0.4809 | Discriminator's fake mean = 0.2593
# discriminator cant provide good gradients for generator because it cant distinuish
# properly between real and fake, going gen_int=3 didnt do much, gent_int=5 looks much better
# 
# Lsgan: 
# test with lsgan(wt spect norm):(20250912212420) lsgan results in mode collapse, this means
# like before, the discriminator is more powerful than the generator, it quickly learns whats
# fake and whats real, the ones that are closer to the real image, will therefore have stronger gradients
# and the generator will be focused on them, resulting in mode collapse (images that look the same)
# we already are using dropout and noise addition to make the discriminator more constrained! so something
# else needs to be done. the other thing that can contribute to this is that the generator has a larger lr
# compared to the discriminator, therefore the generator can quickly rush towards the first minimum 
# it finds thus this manifests itself as the mode collapse (repeated images!) so we need to use a smaller
# learning rate for the generator and a larger/faster one for the discriminator. lets go with lr_d = 0.0004
# and lr_g = 0.0001 for now. the idea here is to allow the discriminator to be able to quickly change and 
# adapt to whatever new things the generator comes up with and keep forcing it to explore and improve itself
#  this fixed the mode collapse in lsgan and we have a stable training so far!
gen_update_interval = 5 if loss_type == "wgan" else 1

# by using
# [1,2] works only for wgangp, others fail miserably!
# wgan fails with all layers specto normalized when gen_update_interval=5
# with gen_update_interval=1 its still trash!
# lsgan keeps failing with mode collapse (repeative images)! 
no_spec_list = [0,5]#[]# list(range(5)) #[] #[1,2]
#discriminator
discriminator_I64 = DiscriminatorImproved64(hidden_size=disc_hidden_size,
                                           no_spec_norm_list=no_spec_list,
                                           # we face mode collapse toward the end when using wgangp
                                           # so I had to set a higher dropout and compensate with 
                                           # larger lr for disciminator to keep it balanced and not face 
                                           # mode collapse early on or later on(both form of mode collapses
                                           # occur if either overpower the other)
                                           dropout_rate=0.25 if loss_type=='wgangp' else 0.2)
discriminator_I64 = discriminator_I64.to(device)
#generator
generator_I64 = GeneratorImproved64(z_size, hidden_size=gen_hidden_size)
generator_I64 = generator_I64.to(device)

betas = [0.5, 0.999] if loss_type=='lsgan' else [0, 0.9]

if loss_type=='lsgan':
    lr_d, lr_g = 0.0004, 0.0001
elif loss_type == 'wgan':
    # wgan requires way smaller lr like 1e-5, 2e-5
    # and a small clipping range like -0.02,0.02 
    # also rmsprop seems do to much better than adam!
    lr_d,lr_g = 0.00001, 0.00002#1e-5, 2e-5
else:#wgangp
    lr_d, lr_g = 0.002, 0.001#0.001, 0.002

# disc_optimizer = torch.optim.RMSprop(discriminatorI64.parameters(), lr=5e-5) # for wgan
disc_optimizer = torch.optim.Adam(discriminator_I64.parameters(), lr_d, betas=betas)
gen_optimizer = torch.optim.Adam(generator_I64.parameters(), lr_g, betas=betas)

training_loop(discriminator_I64, 
              generator_I64, 
              train_loader=train_loader,
              disc_optimizer=disc_optimizer,
              gen_optimizer=gen_optimizer, 
              epochs=epochs, 
              interval=interval,
              gen_update_interval=gen_update_interval, 
              dataset_name=dataset_name,
              loss_type=loss_type, 
              lambda_factor=lambda_factor,
              use_batchnorm=False,
              wgan_range=(-0.02, 0.02), #(-0.02, 0.02) (-0.05, 0.05)
              noise_addition=True,
              device=device)
#%%
# 20250913174046
# load models 
checkpoint = torch.load("./weights/gan/dcgan_generatorcnn_wgangp_20250913174046.pt",
                        map_location="cpu",
                        weights_only=False)

epoch = checkpoint["epoch"]
z_size = checkpoint["z_size"]
hidden_size = checkpoint["hidden_size"]
dataset_name = checkpoint["dataset_name"]
loss_type = checkpoint["loss_type"]
# losses = np.array(checkpoint.pop("losses"))

generator_I64 = GeneratorImproved64(z_size,hidden_size)
generator_I64.load_state_dict(checkpoint.pop("state_dict"))
generator_I64.eval()

for k,v in checkpoint.items():
    print(f'{k}: {v}')
    
# print(f'DLoss: {losses[:,0].mean():.4f} | GLoss: {losses[:1].mean():.4f}')
#%%
run_latent_arithmatic(attr_name='Male', 
                      generator=generator_I64, 
                      classifier=celeba_classifier, 
                      word2idx=celeba_attr_word2idx,
                      random_gen=random_gen,
                      showcase_one_sample=True,
                      num_samples=64,
                      attribute_pool_size=256,
                      maximum_prob_for_neutral_confidence=0.1,
                      attribute_confidence_rate=0.7,
                      alpha_values=torch.linspace(-3,7,steps=24),
                      device='cpu')

#%%
run_latent_arithmatic(attr_name='Smiling', 
                      generator=generator_I64, 
                      classifier=celeba_classifier, 
                      word2idx=celeba_attr_word2idx,
                      random_gen=random_gen,
                      showcase_one_sample=True,
                      num_samples=32,
                      attribute_pool_size=256,
                      maximum_prob_for_neutral_confidence=0.1,
                      attribute_confidence_rate=0.8,
                      alpha_values=torch.linspace(-3,7,steps=24),
                      device='cpu')
#%%
run_latent_arithmatic(attr_name='Eyeglasses', 
                      generator=generator_I64, 
                      classifier=celeba_classifier, 
                      word2idx=celeba_attr_word2idx,
                      random_gen=random_gen,
                      showcase_one_sample=True,
                      num_samples=32,
                      attribute_pool_size=256,
                      maximum_prob_for_neutral_confidence=0.01,
                      attribute_confidence_rate=0.8,
                      alpha_values=torch.linspace(-3,7,steps=24),
                      device='cpu')


#%%
# we improved our results but we had a lot of issues dealing with wgan and to a lesser degress lsgan
# wgangp proved to be a really great addition to our gan architecture. 
# we saw that using carefully tuned hyperparameters we can achieve really good looking images.
# however the issue is that smaller sizes result in better training stability and final output
# we could successfully train 64x64 images, and with a bit more effort get 128x128, but beyond
# that it seems its impossible to get a decent result, training becomes vert unstable and we face
# mode collapse left and right!
# the reason is when we start training with high resolution images (>128) the discriminator faces
# a huge number of fine-grained details like wrinkles hair stands, textures, etc right away!
# and compared to the generator thats just starting to make sense of things, and comming up with
# some samples, it quickly learns and rejects all generated samples, this in turn means the gradient
# will zero or extremely small for generator to be able to have any improvements and thus will be 
# stuck with ugly low res generations and therefore collapse! we may be able to fight this to some
# extend and the generator starts generating some high quality images, but it will be limited and
# with low diversity(one or a few images/concepts get repeated) which is the other form of mode collapse
# so the generator can not using random weights produce highly detailed images that the discriminator
# sees. this is where progan comes into play. the authors thought we as humans learn to draw from corase to fine
# details, so why not try that with GANs? and like that they decided to train GANs from low res to high
# res in several steps. the idea is, start training on low res input like 4x4, then when the training is
# stable, go up one level and try 8x8, and so on. this way both the generator and the discriminator get
# to first learn global structures like faces, body outlines, color distributions, and then gradually 
# learn about finer details like eyes, hairs, skin textures,etc. since this happens at each stage, 
# and both discriminator and generator get to see and learn at the same scale, the training stays balanced
# and we dont face wild ossiliations that we'd face normally when one has more information while the other
# doesnt! this way they could train up t 1024x1024 resolution whcih was insane back then (2017) when 
# the paper came out! they used wgangp for loss, and aside from that and the fact that we have 
# multistage upsampling/processing, they had two new changes in the architcture, they used pixelnorm
# in the generator only and for the discriminator they used something called equalized learning rate
# (i.e. scaling weights at runtime) and minibatch standard deviation layer which they added near the
# end of the discrmintaor to to make the generator generate more diverse images and avoid/fight mode collapse.
#
# lets implement progan! 
# we need two blocks one for our discriminator and the other for our generator
# we wont be using any residuals this time around, its not needed. it will be
# a simple 2 conv layer block with leakyrelu and this tiem around we will be using aveagepooling
# to downsize the input instead of a larger stride

# update:
# Initially when I first tried to implement this I completely forgot Equalized Learning Rate
# and faced a lot of issues and instablity during traing especially from 32x32 resolution
# and higher! as I found out the hardway, this is crucial to have for a stable training!
# see the debug logs at the end
#
# now whats the idea behind equalize learning rate? the idea behind this was to make all 
# conv layers learn at the same or similar consistent speed. both similar and consistent
# are important here. in other words, to decouple the learning rate from the magnitudes 
# of weights in the conv layers.(or we could also say to make the learning rate independant 
# of a layers input size). if you look at the dbeug logs ahead, you'll see we faced a lot of
# issues when we passed certain threshold(32x32 res), we were doing fine until all of sudden
# everything would go south!
# this would happen because normally, the (bad) initializations can cause the magnitude (dynamic range)
# of the activations to explode or vanish as they go through different layers in the network 
# which means some layers might have massive gradients while others might have very small ones.
# this obviously will make the optimizers job extremely difficult. we faced this when BatchNorm
# couldnt be used, we saw this in our previous experiments as well where we had to do careful
# initialization to get things to work. we now face the same issue here. 
# to fix this we cannot obviously use BN, and if we dont use it as you already know we face a 
# lot of issues so we try to have equalized learing rates! this way we can keep the magnitude 
# of all features and gradients consistent throughout the whole network and therefore make 
# training several times more stable and less sensative to our choice of learning rate!
# (a large part of the  issues we have been having like exploding gp, discriminator/generator imbalance 
# all stem from this fact and this technique should fix that for us!)
#
# to implement this equalization/normalization technique, instead of using a standard/normal weight
# initialization algorithm like Xavier, or Kaiming He (the authors used kaminghe) "once", we 
# rescale every convolutional layers weights at every single forward pass! the scaling factor
# that we use is the same as the one used in KamingHe initialization algotrithm( i.e. sqrt(2/fanin)).
# and thats it!
# also note the paper uses bias=False but I included for experimentation!
class EqualizedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        # initialize the weights 
        self.conv.weight.data.normal_(0,1)
        
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

        # scaler is sqrt(2/fan-in)
        # instead of math.sqrt we could also do 
        # self.scaler = (2/(in_channels * kernel_size* kernel_size))**0.5
        # note since scaler can be computed at runtime, theres no need to have
        # self.register_buffer("scaler",...) to preserve it for inference!
        self.scaler = math.sqrt(2/(in_channels * kernel_size* kernel_size))
        
    def forward(self, x):
        # scale the conv weights, note that in backprop, the gradients are calculated
        # with respect to self.conv.weight normally, and our scaler here, a python scaler
        # mind you!, just acts as a scaler (obviously) and scales the gradients so they stay
        # uniformly scaled!!
        scaled_weights = self.conv.weight * self.scaler
        return F.conv2d(x, scaled_weights, self.bias, stride=self.conv.stride, padding=self.conv.padding)


class DiscBlockProGAN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()
        # update:
        # the original paper disabled bias (I trained with bias=True just fine)
        # update2: we use EqualizedConv2d instead of conv2d in all layers
        self.block = nn.Sequential(EqualizedConv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
                                   nn.LeakyReLU(0.2),
                                   EqualizedConv2d(out_channels, out_channels, kernel_size, stride, padding, bias=bias),
                                   nn.LeakyReLU(0.2),
                                   # update from future: 
                                   # previously I downsampled in forward-pass, but since this is
                                   # only used in block, to make things more efficient I add
                                   # the pooling here! so each discganblock downsamples the input 
                                   nn.AvgPool2d(2),
                                  )
        
    def forward(self, x):
        return self.block(x)
 
# we also need to implemenet that mini batch standard deviation, the idea is to
# check the standard deviation of the inputs so we can see how far values lie from the mean
# (basically measure how much variation existis within the batch of inputs)
# and the standard deviation does exactly that. this in turn will help us understand
# whether we are facing mode collapse or not because if the generator is producing identical
# or nearly identical outputs then the std for those outputs (batch) will be very small and 
# this way the discriminator can catch that! and no more fail to notice that generated images
# in the batch all look too simialr! and hence detect mode collapse and force generator to diversify!
# how does it do? we averge all the stds that we got for every feature/pixel in the whole batch
# and concatenate it to the input batch as a standalone channel, this way, the generator needs 
# to comeup with the same values or close enough that matches the real input avberage std.
class AddBatchStdDev(nn.Module):
    def forward(self, x:torch.Tensor):
        # calculate the input std(σ)
        # we could also do std = torch.sqrt(x.var(dim=0,unbiased=False)+1e-8)
        # but its numerically less stable than the dedicated std() pytorhc offers
        # simply because here we are first calculating the variance and then take
        # its squre root to get std, but if the variance happens to be very large 
        # or the oppositive, very small, this two stage calculation may accumulate
        # more rounding errors
        std = torch.std(x, dim=0, unbiased=False)
        # print(f'{std.shape=}')
        # we dont need individual stds for each pixel vector, so we average that to
        # get a single number across the whole batch and concatenate it to input as
        # new channel that shows us the situation
        std_mean = torch.mean(std).view(1,1,1,1)
        b,c,h,w = x.shape
        # out = torch.cat([x, std_mean.repeat(b,1,h,w)],dim=1) #(b,c+1,h,w)
        # expand is more memory efficient than repeat so lets use that
        out = torch.cat([x, std_mean.expand(b, 1, h, w)], dim=1)
        return out
    
# for generator block, we need PixelNorm which simply is calculating l2 norm(in fact root mean square)
# for each pixel across channels!and normalize the input by that! that is do x/sqrt(mean(x²)+eps) 
# x being the input and epsilon(eps) for numerical stability.
# the idea here is to decouple the feature vector magnitude from its direction in the generation process,
# we are familiar with this as we have been doing it ourselves in latent space artithmatic experiments 
# to capture the essense(direction) of certain concepts we liked to play with.
# the easiest way is to implement it as a simple layer!

# sidenote:
# we used mean instead of sum here so the values dont become huge when the number of channels is large, which might make
# us vulnarable to overflow(especially when we want to go half precision or lower and have large number of channeels e.g.)
# also if we didnt use mean(rms), it would make the gradient scale dependant on the number of channels!
# by simply using mean instead of sum, we make it much more numerically stable, and gradients scale
# wont be depadant on number of channels. the only inconvience is that the vector length will be sqrt(c)
# instead of the 1 (true unit vector) but this isnt an issue in our training anyway)
# 
# quicknote:
# (l2norm is sqrt(sum(x^2)) and having mean instead of sum doesnt pose an issue here simply because they
# are proportional (l2norm = sqrt(mean(x²) * sqrt(c) c being the number of channels in x. after all 
# mean(x²)=sum(x²)/C and c is the number of channels! its obvious!)
#
# todo: too excessive? cuz its obvious!!
# uncomment this section and see for yourself if you are not convinced!
# c=64
# x = torch.randn(4, c, 32, 32)
# #pixelnorm using rms
# output_tensor = x / torch.sqrt(torch.mean(x**2, dim=1, keepdim=True)+1e-8)
# #pixel norm using l2norm
# output_tensor2 = x / torch.sqrt(torch.sum(x**2, dim=1, keepdim=True)+1e-8)
# # if we grab a single pixel vector from the output_tensor that used rms
# # and calculate its l2norm, we will see its length equals the sqrt of channels
# pixel_vector = output_tensor[0, :, 0, 0]
# pixel_vector2 = output_tensor2[0, :, 0, 0]
# # calculate l2norm on both of these vectors, we'l see the pixel_vector2 is
# # an scaled version of pixel_vector2! by exactly sqrt(channels)
# pixel_l2_norm = torch.linalg.norm(pixel_vector)
# pixel2_l2_norm = torch.linalg.norm(pixel_vector2)
# print(f"{pixel_vector.shape=}")
# print(f'pixel_vector length(l2norm) by rms: {pixel_l2_norm.item():.1f}')
# print(f'pixel_vector2 length(l2norm) by sum : {pixel2_l2_norm.item():.1f}')
# print(f"math.sqrt(c) -> c={c}: {math.sqrt(c)}")
# print(f'pixel_vector length(l2norm) by rms / sqrt(c): {pixel_l2_norm/math.sqrt(c):.1f}')
# # and we see the l2norm is indeed proportional to rms*sqrt(c)!
# out = torch.allclose(output_tensor2, output_tensor/math.sqrt(64))
# print(f'both yield the same result: {out}')
# 
class PixelNorm(nn.Module):
    def forward(self, x, eps=1e-8):
        # keepdim is to retain input shape so we get (b,1,h,w) and pytorch
        # broadcast it properly to the whole input volume
        # we could also use l2norm and still get the same effect but rms is more stable
        # return x / torch.sqrt(torch.sum(x**2, dim=1, keepdim=True)+eps)
        # note for fp16 in order to be safe its a good idea to set eps to a
        # larger value like 1e-5 so we dont face numerical instability! 
        return x / torch.sqrt(torch.mean(x**2, dim=1, keepdim=True)+eps)

# just like the discriminator generator block will be simple 2 layer conv blocks
# with leakyrelu(0.2) and pixel norm! the authors said using leakyrelu in generator
# as well as disciminator made training much more stable, because it allows much better
# gradient flow. so we do the same here. since we have stages, we do the upsampling later in code
class GenBlockProGAN(nn.Module):
    #update: like discriminator the paper uses bias=False (I disabled it the second time im reviewing this)
     def __init__(self,in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
         super().__init__()
         
         self.block = nn.Sequential(# include the upsampling layer for efficieny so we dont
                                    # do it in forward pass!
                                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                    EqualizedConv2d(in_channels, out_channels, kernel_size,
                                              stride, padding, bias=bias),
                                    nn.LeakyReLU(0.2),
                                    PixelNorm(),
                                    EqualizedConv2d(out_channels, out_channels, kernel_size,
                                              stride, padding, bias=bias),
                                    nn.LeakyReLU(0.2),
                                    PixelNorm())
     def forward(self, x):
        return self.block(x)


# now for the discriminator we build the architectures in steps
# the initial block will process a small dimension, e.g. 4x4
# then for the next stage, we merge the smaller size into the
# larger size (using alpha to gradually do the transition from
# low res to higher res, otherwise it wont learn properly and
# trainingw ill be very unstable).
class DiscriminatorProGAN(nn.Module):
    def __init__(self, max_steps=6, starting_base=2):
        super().__init__()
        
        # in order to be able to properly resume checkpoints
        # we need to build the model properly at resume aswell
        # so lets move all the logic into setup layers so we 
        # can call it easily whenever we like to reinitialze the network
        self.setup_layers(max_steps, starting_base)
    
    def setup_layers(self, max_steps, starting_base):
        self.max_steps = max_steps
        # this is to control the channel size
        self.starting_base = starting_base
        # create some channels for our layers
        # we can choose any number for our channels
        # we like, but to keep things simple and like
        # the original paper, we go in power of 2.
        # to specify the minimum number of channels 
        # we simply startthe power at that desired size
        # i.e. if we want the smallest number of channels
        # be 16 we simply use 2^3 = 8!
        # so its 2^max_steps all the way down to 2^3=8
        # i.e. [512, 256, 128, 64, 32, 16, 8]
        # also note we go from lowest res with highest number
        # of channels to the highest res with the lowest
        # number of channels. in other words, 512 is for
        # lowest res image we work with like 4x4 and 16 
        # is for the highest res e.g 512x512 or whatever we
        # choose as the highest resolution!
        # Initially I went with 1024 as the l argest channel count
        # but it proved to be extremeley taxing on my vram!(rtx3080)
        # and it also took a lot of time(with max_steps=7)
        # so lets use smaller channels!
        channels = [ 2**(i+starting_base) for i in range(max_steps,0,-1)]
        print(f'{channels=}')
        # we have to build 3 blocks, one is used for input images
        # and the other for the rest of the processing and a final one
        # for the final output. this way managing things becomes so much easier
        # note we have to use nn.modulelist or otherwise these wont be registered
        # as submodules of this module and wont be found/discovered by other pytorch calls (.to(), .parameters() etc)
        #
        # like the name implies this part deals with image input exclusively. 
        # for each stage/step/depth of the network, we assign a dedicated 
        # imageprocessor (layer thataccepts images) and produces the output
        # with proper number of channels for the next block to porcess
        #
        # quicknote:
        # we are basically creating several subnetworks dynamically, each subnetwork 
        # needs to get an input image, downsamples it and send it to final layer to 
        # get a pridiction.
        # to make things easy, we decided our final layer accepts 4x4 input and produces
        # the pridiction by a pooling at the end.
        # the first subnetwork therefore only has one layer that accepts the image
        # the first image is 4x4, so no further processing will be needed, we directly send the
        # output to final layer and get a pridiction. 
        # the next subnetwork will work with 8x8 images, so it has an input layer of its own
        # that accepts an 8x8 image, but since its 8x8, we need to run more processing to get
        # good result, so we add an additional block to process it, since we
        # want pridiction, and final layer accepts 4x4, we downsample it to 4x4 but before we 
        # send that 4x4 to final layer, we must merge it with the previous lower res output,
        # to get the previous res output, we downsample our 8x8 input image to 4x4 and feed
        # it to the previous subnetwork image layer, so it process it only and the res is retained,
        # we now have the output from previous layer, we merge our downsampled output with this
        # one, and get a new output, we need to run some processing on this one and finally
        # feed the result to final layer for prediction.
        # the next network will work with 16x16 images, so this needs to do the same thing, 
        # process it and downsample it to 8x8, now to merge the output with the previous 
        # subnetwork's output, we do the same, downsample the 16x16 input image to 8x8 image,
        # feed it to the previous (low res) subnetwork imagelayer, get 8x8 image, merged it
        # with the new downampled high res output, process it and feed it to the final layer.
        # so basically every subnetwork uses larger input, has an intial processing, then 
        # downsamples it to lower res, feed it to previous lower res, get the initial processing
        # and then merges the highres downsampled to lowres from previous step and does some
        # processing on it and sends it to final layer!
        # the generator does the same thing but in reverse!
        # we can create these subnetworks individually and call them as subnetworks
        # at proper steps, or we can build them like this dynamically
        #          
        # update: 
        # using simple 1x1conv with leakyrelu seems to be the norm my version seems
        # to be abit too complex! using simple conv1x1-leakyrelu made everything much better!
        # self.fromImgs = nn.ModuleList([DiscBlockProGAN(3, channels[i]) for i in range(max_steps)])
        # update2:
        # I later added downsampling to DiscBlockProgan and removed the downsampling from forwardpass
        # so if you eevr wanted to retest this, remember not to downsample again! you dont need to now!
        # (see the code in commit 7fbaef5c7e96333c30fb06d7a56ce59e9e55aedb which is just before I change this
        # the architecture and explanations make much more snese incase you get confused!)
        self.fromImgs = nn.ModuleList([nn.Sequential(EqualizedConv2d(3, channels[i], kernel_size=1),
                                                     # update: damn it I mistyped 0.02 as 0.2 and it 
                                                     # created so much issues, this practically lowered
                                                     # the gradient flow 10x!
                                                     nn.LeakyReLU(0.2)) for i in range(max_steps)])
        # print(f'{self.fromImgs=}')
        
        # and this part deals with the rest of processing, each block belongs to separate stage/step/depth
        # here we only specify the channel configurations, the actual spatial size will be determined
        # and handled in the forward pass. that is after each step, we halve the output like 32x32 -> 16x16
        # ans so on 
        self.blocks = nn.ModuleList([DiscBlockProGAN(channels[i],channels[i-1]) for i in range(1,max_steps)])
        # print(f'{self.blocks=}')
        
        # update:
        # to make things more efficeint and faster, I replaced this loop
        # from the forward pass. this is used to process the input for the
        # remainig layers/blocks after we processed the step and step-1 blocks.
        # for i in range(step-2,-1,-1):
        #     out = self.blocks[i](out)
        #     # at each step we lower the res for the next stage/block
        #     out = F.avg_pool2d(out, 2)
        self.remaining_blocks = nn.ModuleList()
        for step in range(self.max_steps):
            remaining = []
            for i in range(step-2,-1,-1):
                remaining.append( self.blocks[i])
            # each step needs its own sequence, so we store each in a list to call when needed
            self.remaining_blocks.append(nn.Sequential(*remaining) if remaining else nn.Identity())
         
        # and the final block that grabs the final processed 4x4 output from previous processings(self.blocks)
        # and gives us a final 1 output. note since we will be using wgangp we dont use any activation functions
        # also note since we are working at the lowest rest (4x4), the channel configuration will be the first
        # one, i.e. 512 in our case (higher channel count goes with lowest res and viceversa this is so
        # when we recieve a high res image( i.e. with stage/step/depth>0), it starts with low channel count, and as
        # the spatial size decreases the number of channels increases so the representational capacity is not 
        # hindered just like any normal cnn!)
        self.final = nn.Sequential(# calculate and add average stddev to input samples
                                   AddBatchStdDev(),
                                   # our final layer works on the lowest res, so the channels[0]
                                   # is what we want here!
                                   EqualizedConv2d(channels[0]+1, channels[0], kernel_size=3, padding=1),
                                   nn.LeakyReLU(0.2),
                                   # make the final 4x4 volume 1x1 at the end
                                   EqualizedConv2d(channels[0], 1, kernel_size=4, stride=1, padding=0))
    
    def forward(self, x, alpha, step):
        # if we at the lowest res/depth/step simply
        # return the result
        if step == 0:
            out = self.fromImgs[0](x)
            # print(f'{out.shape=}')
            out = self.final(out)
            return out.view(-1,1)
        
        # otherwise, if we have high res image process the input
        # and merge it with the previous step output
        new_input_out = self.fromImgs[step](x)
        new_input_out = self.blocks[step-1](new_input_out)
        # downsample so we can merge it with the previous lower res output
        # update: each block downsamples its input so no need to downsample again!
        # new_input_out = F.avg_pool2d(new_input_out,2)
        
        # to get the previous result, we need to downsample the input to get the proper size
        x_downsampled = F.avg_pool2d(x, 2)
        previous_input_out = self.fromImgs[step-1](x_downsampled)
        
        # now merge the two so we have smooth transition between blocks (fade-in path)
        out = alpha * new_input_out + (1-alpha)*previous_input_out
        
        # and finally process the rest of the blocks from high res
        # to the last layer. after merging we are at depth-1, so we
        # need to continue from depth-2 to the end
        # for i in range(step-2,-1,-1):
        #     out = self.blocks[i](out)
        #     # at each step we lower the res for the next stage/block
        #     out = F.avg_pool2d(out, 2)
        # precomputed the layer configurations so we get much faster training!
        out = self.remaining_blocks[step](out)
        
        # and finally get the output
        out = self.final(out)
        # print(f'{out.shape=}')
        return out.view(-1,1)
 
 # like convlayers, we need to implement Equalized learning rate fo convtransposed aswell
class EqualizedConvTrans(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0,bias=False):
        super().__init__()
        
        self.conv_trans = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride,
                                             padding, output_padding, bias=bias)
        # initialize the weights with normal distribution
        self.conv_trans.weight.data.normal_(0,1)
        
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        # sqrt(2/fan-in)        
        self.scaler = math.sqrt(2/(in_channels * kernel_size* kernel_size))
        
    def forward(self, x):
        scaled_weights = self.conv_trans.weight * self.scaler
        return F.conv_transpose2d(x, scaled_weights, self.bias, stride=self.conv_trans.stride,
                                  padding=self.conv_trans.padding,
                                  output_padding=self.conv_trans.output_padding)

class GeneratorProGAN(nn.Module):
    def __init__(self, z_size, max_steps=6, starting_base=2):
        super().__init__()  
           
        # lets do the same thing for generator
        self.setup_layers(z_size, max_steps,starting_base)
        
    def setup_layers(self, z_size, max_steps,starting_base):
        self.z_size = z_size
        self.max_steps = max_steps
        self.starting_base = starting_base
        # generators like the discriminator but the oppiste!
        # [512,256,128,64,32,16] we go from low res with high 
        # channel count(i.e 512) to high res with low channel count(i.e. 16)
        # update: used max_steps=7 but its too large for me and my 
        # rtx3080! it took me more than 5 hours and 9.84Gb vram 
        # to get to 32x32, so im halving the channels to make it more managebale!
        # so instead of i+3, we go with i+2
        channels = [ 2**(i+starting_base) for i in range(max_steps,0,-1)]
        print(f'{channels=}')
        # this is the first layer we use to get latent vector and build a 4x4 initial output which
        # is then processed by the blocks and the rest.
        self.initial = nn.Sequential(EqualizedConvTrans(z_size, channels[0], 4, 1, 0),
                                    # we could also do 
                                    # nn.Linear(z_size, channels[0]* 4*4),
                                    # nn.Unflatten(dim=1,unflattened_size=(z_size,4,4)),
                                    # to get 4x4 fmap from the given latent vector, but 
                                    # not nn.upsample, if we were to use upsample it would
                                    # have simply copy the input to a 4x4 fmap basically
                                    # instead of learning 4x4 pixels, it would copy 1 into
                                    # 4x4 which would make it harder for the network to learn
                                    # structures from inpyts. we can learn it during training,
                                    # but not now well see how to do this in stylegan!
                                     nn.LeakyReLU(0.2),
                                     PixelNorm(),
                                     EqualizedConv2d(channels[0], channels[0], kernel_size=3,padding=1),
                                     nn.LeakyReLU(0.2),
                                     PixelNorm(),)
        
        # we use this block to get image output(final layer)
        # update: forgot tanh!
        # update2: like discriminator only a single 1x1 conv layer should be enough my version(paper uses only 1)
        # seems to be too complex and this might be one of the reasons why Im having so much
        # difficulty post 32x32 resolution. see debug log at the end for more information!
        # self.toImgs = nn.ModuleList([nn.Sequential(GenBlockProGAN(channels[i], 3), nn.Tanh()) for i in range(max_steps)])
        # 
        # sidenote from future: 
        # while implementing the stylegan1, I noticed neither progan nor styleganv1 which follows
        # progan, use tanh in toImgs layer (they call it toRGB). the reason being it can
        # slow down the convergence as its saturating function and can cause mode collapse or washedout colors
        # if it saturates! so I leave it be , but test without it as well
        # TODO: test the implementation without Tanh()
        self.toImgs = nn.ModuleList([nn.Sequential(EqualizedConv2d(channels[i], 3, kernel_size=1),
                                                   nn.Tanh()
                                                   )for i in range(max_steps)])
        # print(f'{self.img_output=}')
        
        # and this to do the rest of processing. like before we only do chanel configs here and the
        # actual upsampling for each step/stage/depth is done during forward pass
        # update: like in discriminator , I included the upsampling in GenBlockProGAN to make
        # things more efficient and much faster! and remove the upsampling part from forward pass below
        self.blocks = nn.ModuleList([GenBlockProGAN(channels[i-1],channels[i]) for i in range(1,max_steps)])
        # print(f'{self.blocks=}')
    
    
    def forward(self, z, alpha, step):
        if z.dim()==2:
            # reshape z to be 4d so convtranspose works properly
            z = z.view(z.size(0),-1,1,1)

        out = self.initial(z) #4x4
        
        # the base case, if we are at step 0, stop and return the image
        # since there are no previous lyaer before 0! and we dont need
        # to do fadein! (cuz theres nothing other than this to merge with!)
        if step==0:
            return self.toImgs[step](out)
        
        # otherwise, upsample the low res output we got so far and
        # process it through the first block up to current step
        # to make it high res.
        previous_output = None
        for i in range(step):
            # keep the previous output so later on we can use it to get previous step output
            previous_output = out
            # upsample the current res to next step res, 
            # update: removed, blocks now does downsampling aswell
            # out = F.interpolate(out, scale_factor=2, mode='bilinear')
            out = self.blocks[i](out)
        
        # now get the image out for this step
        out_img_new = self.toImgs[step](out)
        # now we need to merge the current image with the previous lower res image
        out_img_old = self.toImgs[step-1](previous_output)
        # now we need to upsample it so we can incorporate alpha for the final merge
        out_img_old = F.interpolate(out_img_old, scale_factor=2,mode='bilinear')
        # now like some people like to call it, we merge the high res path(out_img_new) with the low res path(out_img_old)
        final_img = alpha * out_img_new + (1-alpha)* out_img_old
        return final_img


# x = torch.randn(size=(5,3,256,256))
# z = torch.randn(size=(5,100))
max_steps = 7
disc = DiscriminatorProGAN(max_steps=max_steps, starting_base=2)
gen = GeneratorProGAN(100,max_steps=max_steps, starting_base=2)
# test all the stages/steps
for i in range(max_steps):
    H= W = 2**i*4
    x = torch.randn(size=(5,3,H,W))
    z = torch.randn(size=(5,100))
    disc_out = disc(x, alpha=1, step=i)
    print(f'disc_out.shape: {tuple(disc_out.shape)}')
    gen_out = gen(z, alpha=1, step=i)
    print(f'gen_out.shape : {tuple(gen_out.shape)}')
#%% progan loss functions and related functions for trainig
# now training part!
# before we write the training loop, there are a few things we need to be aware of
# first we are dealing with different resolutions, so we need a separate dataloader
# for each resolution so we can create one during the training. 
# second, since we have different resolutions, we cant use the same batchsize for all
# when the res is low, we can use larger batchsize, but as we get to higher res we maynot
# be bale to fit all into our vram, so its best to create a separate batchsize for each resolution!
# and third, we also need to take care of alpha, we need to grdaually increase it from
# the initial 0 toward the final 1 which signals to use the new output. so we can specify
# certain number of iterations before we go for alpha change and therefore fadein process.
# and finally the epochs, it would be a good idea to have at last a few epochs for each res
# so the network doesnt quit that res prematurely and learns porperly.
# I guess I said it all lets write the training loop, if anything is left out I explain it 
# in code
# todo: see page 13 for training details : https://research.nvidia.com/sites/default/files/pubs/2017-10_Progressive-Growing-of/karras2018iclr-paper.pdf
# also at the last page shows different results for different setups (like how each addition
# (pixelnorm,minibatchstddev, etc)improves the training )
# TODO talk about the Sliced Wasserstein Distance (SWD) that the authors used for evaluating
# the images at different resolutions. (section 4.2)
# (its used to measure the similarity between the real and generated image patches at different
# resulotions/scales. its very efficient and an approximation of the full wasserstein. as
# its an efficient way to check adn see how well the generator is doing and whether it captures
# the image statistics (i.e. textures, structures, etc) accuractely.
# I completely missed this and sice we used FID/IS in previous experiments, went with that!
# so add this as well!
#
# minor change in our wgangp loss, because ourt discriminator/critic needs alpha and step
# we need to add these as well. we could go back and add an additional args to the original
# implementation so the discriminator that needs additinal arguments can use that but I 
# thought to keep things simple! so the flow of things stays intact and simple hence I 
# reimplemented (copied them!) them here instead
def gradient_penalty_progan(discriminator:DiscriminatorProGAN, imgs_real, imgs_fake, *args):
    batch_size = imgs_real.size(0)
    device = imgs_real.device
    eps = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolated_input = eps * imgs_real + (1 - eps) * imgs_fake
    interpolated_input.requires_grad_(True)
    
    inter_preds = discriminator(interpolated_input, *args)
    grad = torch.autograd.grad(outputs=inter_preds,
                               inputs=interpolated_input,
                               grad_outputs=torch.ones_like(inter_preds),
                               create_graph=True,
                               retain_graph=True,
                               only_inputs=True,)[0]
    # caculate l2-norm of gradients
    grad_norm = grad.view(batch_size, -1).norm(2, dim=1)
    # make sure the gradient norm with respect to inputs 
    # is almost equal to 1. any deviation from norm = 1 
    # is therefore penalized.
    penalty = ((grad_norm - 1) ** 2).mean()
    return penalty

def wgangp_critic_loss_progan(critic:DiscriminatorProGAN, imgs_real, imgs_fake, lambda_factor, *args):
    real_preds = critic(imgs_real, *args)
    fake_preds = critic(imgs_fake, *args)
    wgan_loss = wgan_critic_loss(real_preds, fake_preds)
    gp = gradient_penalty_progan(critic, imgs_real, imgs_fake, *args)
    # gp shouldnt be large! the normal range is 1 at most 10! anything
    # larger it means gradient explosion is underway!
    if gp>100:
        print(f'WARNING: High Gradient Policy: {gp.item():.2f}')
        # returning gp is a good idea cause allows us to log it
        # and check it to see whats happening during training! 
    return wgan_loss + (lambda_factor*gp), gp.item()

# update: 
# added this later to make reading logs much easier
# todo: check values and tune them more accurately
def get_status(score, higher_is_better=True, high_threshold=0.8, mid_threshold=0.4, low_threshold=0):
    if higher_is_better:
        # for real score we want high positive numbers
        # any positive number is good!
        if score >=high_threshold:
            return "😎"
        
        # ok, but worrisome, we want large positive nubers
        # nothing close to 0!
        elif score >=mid_threshold: 
            return "😟"
        
        elif score >=mid_threshold/2: 
            return "😰"
        # if its smaller than 0.4 we are in trouble! discriminator
        # /critic is not learning! it has no idea what is real and
        # whats not and is not giving high score to real images!
        else:
            return "😵"
    else:
        # fake_score needs to be close to 0 or less!
        # so any negative number for fake is good!
        if score <= low_threshold:
            return "😎"
        # if its larger its okish, but its worrisome it needs to
        # get lower and lower, otherwise it means discriminator/
        # critic has no idea about fake/real and generator may be
        # winning
        elif score <= mid_threshold:
            return "😟"
        
        # its larger than 0.6! and the generator may be wining!
        elif score <= mid_threshold/2:
            return "😰"
        
        else:
            # larger than that its not good!
            return "😵"

def get_overall_status(real_mean, fake_mean, IS_score=None, 
                       min_mu=1.2, max_fake_threshold=0, min_disance=1.0):
    
    # using IS_score we can also quickly show if we have mode collapse
    # or not. the mean must be larger than 1, the more the better, if
    # its close to 1 (e.g. 1.1) or 1, the generator has collapsed! we
    # cant have that! 
    # the std must obviously be low, a high std ( or worst >1),
    # means our generator is very unstable and creates very different images
    # this is not about diversity! we want to match our original dataset and
    # our IS should reflect its closeness to the dataset. 
    # 
    # sidenote:
    # the fake_images std can be both low and high and be okay and not ok at the same time!
    # since its complicated (I explained this in detail in debugging section)
    # I therefore remove the std check, as it can give false positive/false negative
    # depending on the situation which defeats the purpose of its use here!
    #
                       
    # IS_std measures consitency, lower value is better it means generator
    # is doing a good job creating the same quality images, but it needs
    # to be calculated over a large amounts of images (e.g. 10k) to be reliable
    mu_is, std_is = None, None
    if IS_score is not None:
        mu_is, std_is = IS_score
    
    # if the discriminator cant decide what real is and assings higher
    # score to the fake image, we have a serious issue!
    # the discriminator may have collapsed or be going to! 
    # (we have flipped discriminator instead of giving + to real its treating
    # fakes as real and giving them higher scores than it gives to the real ones)‼️
    if real_mean <= fake_mean:
        return "‼️"
    
    # check for mode collapse
    # 
    # the mean must be larger than 1 anything close to 1, could simply
    # mean collapsed generator (it creates bad images (1,0) e.g. means terribe score)
    # the std like the mean can very from dataset to dataset, more than
    # it does for the mean. so I'm not going to check for it here
    # 0.3 could be too much, while much lower std can also show critical issues
    # depending on the dataset and number of smaples used to calculate IS
    # so lets ignore it for now I need to include more information along side
    # this to make it somewhat accurate if I want to report it automatcally
        
    # if the fake_score is larger than 0 regardless of the real_score, 
    # we may be going toward mode collapse! fake_score needs to be a small
    # number, the smaller the better, or more accurately the more negative the better!
    # so when its not negative, and its not early in the training, 
    # the discriminator is either weak or the generator has already
    # collpased or has found a loop hole(a pattern e.g.) to exploit discriminators
    # weakness because, it has managed to fool the discrimnator and get 
    # a high positive number. obviously we dont want that to happen.so when
    # this happens it means something has gone very wrong. either generator
    # overpowered the discriminator (or we can say discriminator is weak)
    # or the generator has collapsed and keeps repeating something that gets
    # it identified as real, both of which results in horrible output which
    # we dont want! we can also figure this out with more certainity using IS score.
    # if the IS std is too large it means the generator is all over the place, 
    # it produces wildly different values that shows it hasnt learned properly
    # and is generating nonsense noise!
    elif fake_mean >= max_fake_threshold or (mu_is is not None and mu_is <= min_mu):
        return "😱"

    # real score needs to be larger than fake score thats the base line but
    # they also need to be far away from eachother. the discriminator is supposed
    # to give large positive scores to the real image and tiny score to the fakes
    # it needs to be negative as we said just now! so if they are both positive,
    # theres something wrong! if we are at the start of training its ok, but they
    # need to be a large gap between them that ultimately leads to fakes becoming
    # negative and real becoming more positive. 
    # here we say if the difference is less than 1, then they are too close, 
    # and we might be having a problem! but its not as bad as the previous one
    # when the fake_score is a large positive number! but if they are both large
    # positive numbers, then its game over! (one can be 20 the other can be 19! i
    # t would still be catasrophic!)
    elif (real_mean - fake_mean) <= min_disance:
        return "🫤"

    else:
        # everything should be fine!
        return "😀"

@torch.no_grad()   
def get_IS_FID_score(metric:IS_FID_Calculator, gen:GeneratorProGAN, data_loader,
                     dataset_name, split, batch_size=64, num_samples=10_000, **kwargs):
    # instead of going over the whole trainig sets, we can choose
    # a portion of them instead grab a subset of the dataloader as many as num_samples
    # and generate as many as num_samples fake_loader
    # update:
    # to make sure we get all classes, and not just create a dataloader out of one class
    # by mistake shuffle the indexes! this important because somedatasets are sorted
    # so our previous version which uses (range(num_samples)) would grab the first
    # class or classes, and we would be comparing e.g. airplanes with everything else!
    # so this fixes that as well.
    total_sample_count = len(data_loader.dataset)
    indexes = torch.randperm(total_sample_count)[:num_samples]
    subset = torch.utils.data.Subset(data_loader.dataset, indexes)
    real_loader = DataLoader(subset, batch_size, shuffle=False, num_workers=8, pin_memory=True)
    
    real_batch_cnt = len(real_loader)
    fake_batch_cnt = math.ceil(num_samples/batch_size)
    
    assert real_batch_cnt == fake_batch_cnt, f'real and fake images count must match ({real_batch_cnt} vs {fake_batch_cnt})'    
    
    def fake_loader(num_samples, batch_size, **kwargs):
        device = next(gen.parameters()).device
        
        for i in range(fake_batch_cnt):
            # so if there are some left that dont fit a prefect batch only
            # grab as many as they are not more
            current_batch_size = min(batch_size, num_samples - i*batch_size)
            z = torch.randn(size=(current_batch_size, gen.z_size), device=device)
            # update from future (for stylegan2)
            fakes,*_ = gen(z, **kwargs)
            #update:
            # to be sure the range is valid -1,1 especially for stylegans
            # we apply a tanh, obviously this isnt a problem as we are not 
            # training! just to be sure FID recieves proper range!
            # return a tuple to mimic an actual dataloader!
            yield fakes.tanh(), torch.zeros((current_batch_size,1))
    
    IS_score = metric.compute_IS(fake_loader(num_samples, batch_size, **kwargs))
    FID_score = metric.compute_FID(real_loader,
                                   fake_loader(num_samples, batch_size, **kwargs),
                                   dataset_name=dataset_name,
                                   split=f'{split}_{num_samples//1000}K')
    return IS_score, FID_score

# like the paper, we can use an exponential moving average of weights for
# the generator and get a much better output (the paper always used this
# for inference/visualization, whereas we always used the generator itself
# im adding this at the end when I got great results normally, im just
# adding this for the sake of completeness cuz I wasnt sure if my changes
# were good enough to give me good results initially (see debug logs at the end))
@torch.no_grad()
def update_ema_generator(g:GeneratorProGAN, g_ema:GeneratorProGAN, decay=0.999):
    # sidenote, we only update the parameters we dont touch buffers (we dont have
    # any, but if we had like batchnorm, we wouldnt touch them as it would have
    # destroyed their stats!)
    for ema_p,p in zip(g_ema.parameters(),g.parameters()):
        ema_p.data.mul_(decay).add_(p.data, alpha=1-decay)

#%% progan trainig loop

def training_loop_progan(discriminator:DiscriminatorProGAN, generator:GeneratorProGAN, disc_optimizer:torch.optim.Adam, 
                         gen_optimizer:torch.optim.Adam, epoch_list, batch_size_list, gen_update_interval, dataset_name,
                         split, loss_type='wgangp', lambda_factor=10, gen_num_samples = 64, wgan_range=(-0.01, 0.01),
                         noise_addition=False, use_ema_inference=False, ema_warmup_images_threshold=1_500_000,
                         keep_raw_generations=True, quick_and_noisy_IS_FID=False, device='cuda', resume=False, decay_step=3, 
                         weights_save_dir='./weights/gan', images_save_dir='./results/gan', checkpoint_path=None,):
        
    # lr_d = [p['lr'] for p in disc_optimizer.param_groups][0]
    # lr_g = [p['lr'] for p in gen_optimizer.param_groups][0]
    # since we want the default lr/betas we set when we created optimizers 
    # we can use defaults dictionary. these as the name implies are our
    # default values. the ones in parameter_groups are the updated ones
    # if e.g. we use a scheduler, the lr in parameter_groups will change
    # but the defaults will be intact! we dont use a scheduler for now so
    # its the same for us here!(note param_groups can be more than 1,
    # and its saved with satet_dict, but defaults is not!)
    # to be sure we catch this if later on we used more lrs for optimizers
    # heres an assert!
    assert len(disc_optimizer.param_groups)==1 and\
           len(gen_optimizer.param_groups)==1, 'We expect to have only one param_groups!'
    
    lr_d = disc_optimizer.param_groups[0]["lr"]
    lr_g = gen_optimizer.param_groups[0]["lr"]
    betas_d = disc_optimizer.defaults["betas"]
    betas_g = gen_optimizer.defaults["betas"]
    
    # get a copy of the generator so we can calculate the exponential moving average
    # for its weights to be used for better visualization/inference
    # we need to update this after each iteration
    # sidenote: 
    # during my initial experiments I noticed whenever I tried to use ema version for
    # visualization I would get solid grays for images. it would start off fine in the
    # 4x4 step, but gradually fade into solid gray and from there it was solid gray for
    # the subsequent steps. turns out, my ema is ok but since at the start the model
    # is just random weights, the updates are eratic / random as well. when we use the
    # original generator the images are ok because they show the weighst at that exact 
    # time, but when we use the ema version, its the smoothed out version from the begining
    # each random update that was applied on our ema_version, gradually made it go toward
    # 0! some random weights were higher (e.g. +0.05) but then there were other random upades
    # (e.g. -0.04) and as you can see, if we average these random values, we endup with 0 
    # or very close to it, hence why we get the gray images! 
    # so what the authors did was they only enabled ema after some warm up steps/epochs?
    # and this way, they jumped over the random weights. 
    # so we dont want these initial junk/meaningless updates to destroy our ema because going
    # with decay rate like 0.999 which is our default, it will take a lot to replace the effect
    # of these initial updates( our original generator may very well learn alot but the ema
    # will be far far behind).
    # so the idea is to wait until the updates become reasonable, and generator actually learns
    # something and then enable ema. we can do this in a few ways. one way is to have a counter
    # and track how many real images have been seen in the training, like e.g. 10k/100k images,
    # and only after this value we start the ema process, which is to initialize it with the
    # weights of generator and from carry on! 
    # the other way is to simply start this at a later step like 16x16 res, when the generator
    # has learned something. its much easier as well but the issue with the second method is 
    # that, we start at a later state, the 16x16 weights themselevs may be unstable, and 
    # averaging them might not help much. or it gives us the smooth transition we can have 
    # with the previous method. so having started before would help more. also we can set 
    # the image number a higher number and achive the same result, moreover, 16x16 might work
    # for some datasets and not for others, so I guess we do the first method! 
    # for that we select a threshold for the number of real images that the network need to
    # see before we go for ema, and a global counter to count that!
    # sidenote: 10k images may not be enough for all datasets for example in my experiments,
    # 10K was nothing! and I would get gray images, even 100k wasnt enough, I used 1M! and then
    # things started to look normal (i.e. not solid grays!) but then they also transitioned to
    # other solid colors like maron(brown/redish color) that is it was still too early! also it
    # could be caused by our large lr! so I guess the second method would be as effective 
    # because im not storin bunch of variables in the checkpoint! just start from step x and 
    # carry on! ok I tested that and the second method doesnt work! first it limits us, that is
    # imagine we want to start the ema at the middle of step2 (16x16 res) because right at the
    # very start of step 2, it doesnt work, its still very unstable! we cant do anything! cuz
    # we can only specify the step! moreover, if we start at step=2 e.g. we need to keep checking
    # if the outer most loop, the step loop itself, and until we reach the specific step(i.e. e.g. 2)
    # we use the generators weights, and if we do that, its simply reseting the ema at each step
    # and we update that at each iteration which would result in solid gray images! 
    # if we dont update by having an if statement to see if we are passed that warmup step(i.e.2)
    # that would be two checks! so we dont want the second method, its inefficient and not good
    # the first method gives us more flexibility
    # update: found the issue, and fixed it (see debug log tldr add_() was missing!)
    # instead of copy.deepcopy we could instantiate a new copy adn simply do 
    # load_statedict() on it. i.e. do 
    # ema_generator = GeneratorProGAN(generator.z_size, generator.max_steps).requires_grad_(False).eval()
    # ema_generator.to(device)
    # ema_generator.load_state_dict(generator.state_dict())
    ema_generator = copy.deepcopy(generator).requires_grad_(False).eval()
    
    # real images seen so far during training
    ema_warmup_images_seen = 0
    
    assert discriminator.max_steps == generator.max_steps, 'max_steps for generator and discriminator/critic must be equal!'
    
    metric = IS_FID_Calculator(device)

    fixed_z = torch.randn((gen_num_samples, generator.z_size)).to(device)

    experiment_date = datetime.now().strftime("%Y%m%d%H%M%S")

    starting_step = 0
    starting_epoch = 0
    last_training_step_counter = 0
    max_steps = discriminator.max_steps
    z_size = generator.z_size
    
    # check for resuming from a checkpoint
    if resume:
        if checkpoint_path:
            checkpoint_filename = os.path.split(checkpoint_path)[-1]
        else:
            checkpoints_dirs = sorted([subdir for subdir in os.listdir(weights_save_dir)\
                                       if os.path.isdir(os.path.join(weights_save_dir, subdir))])
            #grab the last checkpoint/most recent one
            checkpoint_dirpath = os.path.join(weights_save_dir, checkpoints_dirs[-1])
            # grab the latest checkpoint 
            checkpoint_files = sorted([f for f in os.listdir(checkpoint_dirpath) if f.endswith(".ckpt")])
            checkpoint_filename = checkpoint_files[-1]
            checkpoint_path = os.path.join(checkpoint_dirpath, checkpoint_filename)
        
        if not os.path.exists(checkpoint_path):
            raise ValueError("The Path is not valid")
        
        # load the stuff
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        max_steps = checkpoint["max_steps"]
        discriminator.setup_layers(max_steps)
        discriminator.load_state_dict(checkpoint["disc_state_dict"])
        discriminator = discriminator.to(device)
        
        z_size = checkpoint["z_size"]
        generator.setup_layers(z_size, max_steps)
        generator.load_state_dict(checkpoint["gen_state_dict"])
        generator = generator.to(device)
        
        # load the ema version, dont forget to also load the images seen so far!
        ema_warmup_images_threshold = checkpoint["ema_warmup_images_threshold"]
        ema_warmup_images_seen = checkpoint["ema_warmup_images_seen"]
        ema_generator.load_state_dict(checkpoint["gen_ema_state_dict"])
        ema_generator = ema_generator.to(device)
        
        # if we are not in the last epoch, then we still have epochs to process
        # therefore the optimizers state must be loaded otherwise everything will
        # go haywire!
        # we dont want the optimizer states to to be available when we go for the 
        # new step with new learning rate cuz probably the internal stats will not 
        # go well with our manual lr change and we may verywell face huge update 
        # or viceversa depending on what we are doing! so we only load optimizers
        # when we are at the middle of the processing of a step! for each step that
        # requires manual lr, we create a brand new optimizer there! so we can load
        # the optimizer's state here just fine
        disc_optimizer.load_state_dict(checkpoint["disc_optimizer"])
        gen_optimizer.load_state_dict(checkpoint["gen_optimizer"])

        # grab the initial lrs
        lr_d = checkpoint["lr_d"]
        lr_g = checkpoint["lr_g"]
        betas_d = disc_optimizer.defaults["betas"]
        betas_g = gen_optimizer.defaults["betas"]

        loss_type = checkpoint["loss_type"]
        starting_step = checkpoint["step"]
        last_training_step_counter = checkpoint["training_step_counter"]
        decay_step = checkpoint.get("decay_step", decay_step)
        epoch_list = checkpoint["epoch_list"]
        starting_epoch = checkpoint["epoch"]+1
        batch_size_list = checkpoint["batch_size_list"]
        wgan_range = checkpoint["wgan_range"]
        gen_update_interval = checkpoint["gen_update_interval"]
        lambda_factor = checkpoint["lambda_factor"]
        noise_addition = checkpoint["noise_addition"]
        
        # we want to start the next step when we resume. 
        # remember during checkpoint saving, we save the finished epoch, so on resume we start 
        # with epoch+1. if this is total number of epochs, then that step is done so we increase
        # the step by 1 as well, if not, the step stays intact and we resume from the next epoch
        # as normal.
        if starting_epoch == epoch_list[starting_step]:
            starting_step += 1
            # also reset the initial epoch for the new step
            starting_epoch = 0
                    
    # store training log for each step  
    all_training_losses = [[] for _ in range(max_steps)]
    all_gradient_penalties = [[] for _ in range(max_steps)]
  
    print(f'ProGAN Training on {dataset_name} with loss={loss_type} in {experiment_date}')
    if resume:
        print(f'--Resume:                  {"N/A" if not resume else checkpoint_filename}'
            f'\n  --From Step:             {starting_step}'
            f'\n  --From Epoch:            {starting_epoch}'
            f'\n  --Checkpoint Path:       {checkpoint_path}'
            f'\n  --Last FID:              {checkpoint["FID"]}'
            f'\n  --Last IS:               {checkpoint["IS"][0]:.4f} ± {checkpoint["IS"][1]:.4f}')
          
    print(f'--Disc Param Count:          {sum([p.numel() for p in discriminator.parameters()]):,}')
    print(f'--Genr Param Count:          {sum([p.numel() for p in generator.parameters()]):,}')
    print(f'--Dataset:                   {dataset_name}-{split}')
    print(f'--Loss type:                 {loss_type}')
    print(f'--Discriminator LR:          {lr_d}')
    print(f'--Generator LR:              {lr_g}')
    print(f'--Max Step:                  {discriminator.max_steps}')
    print(f'--Decay Step:                {decay_step}')
    print(f'--Epochs:                    {epoch_list} ')
    print(f'--Batch-Sizes:               {batch_size_list} ')
    print(f'--ema_warmup_image_threshold:{ema_warmup_images_threshold:,} ')
    print(f'--ema_real_images_seen:      {ema_warmup_images_seen:,} ')
    print(f'--Generator update interval: {gen_update_interval}')
    print(f'--WGAN weight cliping range: {wgan_range}')
    print(f'--Noise addition to input:   {noise_addition}')
    print(f'--WGAN-GP Lambda factor:     {lambda_factor}')
    print(f'--gen_num_samples:           {gen_num_samples}')
    print(f'--Checkpoint Directory:      {weights_save_dir}')
    print(f'--Images Directory:          {images_save_dir}')
    
    for step in range(starting_step, max_steps):
        
        # reset the optimizer for each new step, so the gradients from previous step doesnt 
        # interfere with new, if we were at the middle of training continue with resumed
        # optermizer states, otherwise create brand new ones for each step
        if starting_epoch == 0:
            disc_optimizer = torch.optim.Adam(discriminator.parameters(),lr=lr_d, betas=betas_d)
            gen_optimizer = torch.optim.Adam(generator.parameters(),lr=lr_g, betas=betas_g)
           
        # if we resumed from a half trained model, we continue from the statring_epoch to 
        # the end, however after that we need to reset the starting_epoch for the rest of
        # the steps so they go start from 0
        if step>starting_step:
            starting_epoch = 0
        
        # if step==ema_starting_step:
            # start the ema after some steps so generator weights are not complete random!
            # ema_generator = copy.deepcopy(generator).requires_grad_(False).eval()
        
        # specify resolutions
        # specify batchsizes for each resolution
        # create dataloader for each res
        # specify fadein transition iteration count
                
        batch_size = batch_size_list[step]
        epochs = epoch_list[step]
        # 4 is the lowest res so we want 8,16 etc
        res = 2**step*4
        train_loader = get_dataloader(dataset_name, split=split, resize_dims=(res,res), batch_size=batch_size)
        # calculate how often fadein should kick in we give a bit of leeway
        # so the model learns abit about the old/previous/lowres output and
        # then increase alpha a bit so ultimately during the whole epochs for
        # that res, its gradually increased without hurting the training
        num_batches = len(train_loader)
        interval = num_batches//2+1
        # for alpha we need to take the whole training into account
        # so we need to see how many epochs we have and how many iterations
        # so that we can specify at what point we want to tune it. 
        # usually its gradually increased for the first half of training steps
        # for that specific resolution, and then for the second half we use alpha=1
        # I messed it up the first time and from step=1 I had terrible generations (totla mode collapse!)
        # so getting alpha right is very important
        total_number_of_steps = epochs*num_batches
        # fadein_steps = total_number_of_steps//2
        #update:
        # instead of having half of steps to old res, 
        # lets make it 80% so the other part gets to learn more
        # not sure how it affects early layers need to test this once 
        # for the full training (didnt work, discriminator got more powerful in epoch4+)
        # lets go with 0.4 this time - didnt work! resetting this to default
        fadein_steps = int(total_number_of_steps*0.5)
        # if we are in brand new step, set it to 0 otherwise if we are resuming
        # use the last training step counter so alpha is restored properly!
        training_step_counter = 0 if starting_epoch==0 else last_training_step_counter
        alpha=0
        
        #4,8,16,32,64,128,256
        if step>=decay_step:
            # halve the learing rate for larger steps because 
            # as we get to larger resolutions, it becomes much more
            # sensive and to keep the training stable we need to use
            # very small lr!
            # update after the fixes, decaying lr is not needed see
            # debug log for more information
            
            # e.g. 0.5 goes to 0.25 to 0.125 etc each time we halve the previous one
            decay = 0.5**(step-2)
            
            # update:
            # when I decayed the lrs at 32x32, we faced mode collapsed 
            # because we made discriminator too slow to function/react
            # and generator went heywire and faced mode collapse!
            # so this time I want to use a higher lr for discriminator
            # see update logs at the end.
            
            # instead of changing the optimizers lr manually
            # lets reset the optimizer so changing lr doesnt mess up anything
            disc_optimizer = torch.optim.Adam(discriminator.parameters(),lr=lr_d*decay, betas=betas_d)
            gen_optimizer = torch.optim.Adam(generator.parameters(),lr=lr_g*decay, betas=betas_g)

        current_lr_d = [p['lr'] for p in disc_optimizer.param_groups][0]
        current_lr_g = [p['lr'] for p in gen_optimizer.param_groups][0]

        print(f' Step: {step}/{max_steps} -> Training on [{res}x{res}]')
        print(f'  --Epochs:                      {epochs} ')
        print(f'  --BatchSize:                   {batch_size} ')
        print(f'  --Number of Batches:           {num_batches} ')
        print(f'  --Interval:                    {interval} ')
        print(f'  --Fade-in Steps:               {fadein_steps} ')
        print(f'  --Last training Step taken:    {training_step_counter} ')
        print(f'  --Current Discriminator LR:    {current_lr_d}')
        print(f'  --Current Generator LR:        {current_lr_g}')
        print(f'  --Current Discriminator Betas: {betas_d}')
        print(f'  --Current Generator Betas:     {betas_g}')

        
        for epoch in range(starting_epoch, epochs):
            discriminator.train()
            generator.train()

            losses = []
            step_all_gps = []
            epoch_scores = []
            for i, (imgs_real, _) in enumerate(train_loader):
                #scale input to [-1,1]
                imgs_real = (2*imgs_real-1).to(device)
                
                # track how many real images the network has seen
                ema_warmup_images_seen += imgs_real.size(0)
                
                # if adding noise makes trainig more stable and we get
                # better looking images it means our discriminator is
                # too powerful that messing the signal up and making it
                # harder for it, improves our result! it acts as a regularizer
                # (in terms of distribution impact, adding noise increases the variance
                # for both real/fake images so the discriminator cant prefectly memorize
                # the training data or latch onto a single fake mode!)
                if noise_addition:
                    imgs_real += 0.05 * torch.randn_like(imgs_real)
                
                # train discriminator/critic! 
                # real image predictions
                preds_real = discriminator(imgs_real,alpha,step)
                # disc_real_loss = real_loss(preds_real, smooth=True, device=device)
                # generate an image using generator 
                z_vector = torch.randn((imgs_real.size(0), z_size)).to(device)
                # we detach the imgs_fake so the discriminator cant use the gradients
                # from the generator and quickly learn!
                imgs_fake = generator(z_vector, alpha, step).detach()
            
                # add noise to fake images as well(not needed for dcgan)
                if noise_addition:
                    imgs_fake += 0.05 * torch.randn_like(imgs_fake)
            
                preds_fake = discriminator(imgs_fake, alpha, step)
                # disc_fake_loss = fake_loss(preds_fake, smooth=False, device=device)
                # calculate discrimiator loss out of real and fake losses
                if loss_type =='lsgan':
                    disc_loss = lsgan_discriminator_loss(preds_real, preds_fake)
                elif loss_type =='wgan':
                    disc_loss = wgan_critic_loss(preds_real, preds_fake)
                elif loss_type =='wgangp':
                    disc_loss,gp_value = wgangp_critic_loss_progan(discriminator, imgs_real, imgs_fake, lambda_factor, alpha, step)
                else:
                    raise ValueError(f"Invalid loss type:{loss_type} entered!")
            
                # for debugging purposes
                # if disc_real_mean is a lot larger than disc_fake_mean (e.g. 2.0 vs -2.0) 
                # then it means our discriminator is strong but if both are near the same
                # value and the loss is low then it means our discriminator is confused
                # or is over-regularized.
                disc_real_mean = preds_real.mean().item()
                disc_fake_mean = preds_fake.mean().item()
                
                # keep track of gp trends during training 
                # can help us choose a better lambda and 
                # see if we are using too strong of a lambda
                # that constrains/limits our discriminator too much!
                step_all_gps.append(gp_value)

                # we can also check the fake_images std and understand if everything is OK or not
                # the fake_images std can be both low and high and be okay and not ok at 
                # the same time!
                # I explained this in detail in debugging section. see that
                # note that we dont use the images rather we get the fake_preds because we 
                # dont want to work in pixel space, but rather in features space that have
                # semantic and fake_preds give us that.
                # calculating its mean/std gives us the info we want.
                # 
                # note:
                # first we need to detach the preds so our mean()/std() operations
                # are not recorded in computational graph. its not
                # part of training and we dont want to optimize anything
                # we just want to get some stats.
                preds_detached = preds_fake.detach()
                # we have the mean already so we just get std
                disc_fake_std = preds_detached.std()
                # now that we calculated the std for fakes, lets do that for real
                # we can now better compare them!
                disc_real_std = preds_real.detach().std()

                # store average scores for real and fake images
                epoch_scores.append((disc_real_mean, disc_fake_mean))
                
                # and optimize discrimnator 
                disc_optimizer.zero_grad()
                disc_loss.backward()
                disc_optimizer.step()
            
                # dont forget to clip discriminator's weights in wgan
                if loss_type=='wgan':
                    for p in discriminator.parameters():
                    # keep it roughly 1-lipschitz smaller than this
                    # restricts the discriminator/critic too much!
                    # larger values might be ok only if things go south!
                        p.data.clip_(*wgan_range)

                # now train genertor to create images that look real
                # todo put this in gen_update_interval check so we only run this
                # when we want to optimize, but since currently im doing wgangp
                # and its 1:1 that check is really not needed. also I check preds_fake
                # in loss, so lets leave it be for now, until we get this working!
                z_vector = torch.randn((imgs_real.size(0),z_size)).to(device)
                fake_imgs = generator(z_vector, alpha, step)
                preds_fake = discriminator(fake_imgs, alpha, step)
                        
                # generator loss
                # swap loss! treat fake images as real images
                if loss_type=='lsgan':
                    gen_real_loss = lsgan_generator_loss(preds_fake)
                
                elif 'wgan' in loss_type: #wgan-wgangp
                    gen_real_loss = wgan_generator_loss(preds_fake)
            
                else:
                    raise ValueError(f"losstype {loss_type} not detected!")
                
                # optimize generator
                # update generator with a delay, usually update per 5 critic update
                # seems to make convergence faster
                if (i+1)%gen_update_interval == 0:
                    gen_optimizer.zero_grad()
                    gen_real_loss.backward()
                    gen_optimizer.step()
                    # only update the ema when the main is also updated
                    # otherwise if generator is not updated but we update
                    # the ema, we are practically reapplying old ema again
                    # and agaian (depending on gen_update_interval). for 
                    # our default case which is wgangp, this interval is 1
                    # so it doesnt make any difference if its inside ifblock
                    # or outside, but for other loss types since this can change
                    # it can cause issues, so we better do it properly here!
                    # also until we have not seen as many images as we'd like
                    # dont starting the actual ema update! instead use the 
                    # generators weights!
                    if ema_warmup_images_seen < ema_warmup_images_threshold:
                        ema_generator.load_state_dict(generator.state_dict())
                    else:
                        update_ema_generator(generator, ema_generator)
                
                # we want high positive score/average number for real_mean and
                # lower positive or <real for fake mean (its basically 
                # like this for them, the real_mean means, on average 
                # how good(confident) is the discriminator at detecting real images
                # and for fake_mean it means on average how good is the generator
                # at fooling the discriminator. as you can see we want a high
                # positive average number for discriminator and a low average
                # number(preferably 0 or less) for generator.
                status_r = get_status(disc_real_mean, higher_is_better=True)
                status_f = get_status(disc_fake_mean, higher_is_better=False)
                status_o = get_overall_status(disc_real_mean, disc_fake_mean)
                # included the std for real and fake so it makes it much clearer 
                # to compare and see where we are standing   
                d_real_stat_str = f"D_real_avg: {status_r} {disc_real_mean:+.4f} ± {disc_real_std:+.4f} 📈"
                d_fake_stat_str = f"D_fake_avg: {status_f} {disc_fake_mean:+.4f} ± {disc_fake_std:+.4f} 📉"

                if (i+1)%interval==0:
                    print(f'[{res}x{res}][Epoch {epoch}/{epochs} | Iter: {i}/{len(train_loader)}] Disc Loss: {disc_loss:.4f} | Gen Loss: {gen_real_loss:.4f}')
                    print(f" -- {status_o} Batch-{i}:  {d_real_stat_str}| {d_fake_stat_str}")
                    
                losses.append((disc_loss.item(), gen_real_loss.item()))
                
                # after a good number of training steps (usually half)
                # we switch to alpha=1 until then we gradually increase it
                # at every training step for the specific step/resolution 
                # we currenly are at
                if training_step_counter<fadein_steps:
                    alpha = training_step_counter/fadein_steps
                else:
                    alpha = 1
                # update the training steps
                training_step_counter += 1
        
            d_loss_mean = np.mean(np.array(losses)[:,0])
            g_loss_mean = np.mean(np.array(losses)[:,1])

            # always update the last step per epoch 
            all_training_losses[step].append((d_loss_mean, g_loss_mean))
            gp_mean_epoch = float(np.mean(step_all_gps))
            all_gradient_penalties[step].append(gp_mean_epoch)
            
            # real
            average_score_real_mean = np.mean(np.array(epoch_scores)[:,0])
            average_score_real_std = np.mean(np.array(epoch_scores)[:,0])
            # fake
            average_score_fake_mean = np.mean(np.array(epoch_scores)[:,1])
            average_score_fake_std = np.std(np.array(epoch_scores)[:,1])
            
            # calculate is/fid scores
            # this is only for debugging/
            # it gives us a very noisy estimate of is/fid and I dont use it normally
            # I added it for cifar10 experiments, because the full acuurate version
            # took more time than the whole epoch, and I wanted to do quick hyperparameter
            # tuning, I added it to have a somewhat noisy fid/is score so loging goes as usually
            # whenever I need acurate is/fid I use the full version
            if quick_and_noisy_IS_FID:
                IS_score = metric.compute_IS(imgs_fake)
                FID_score = metric.compute_FID(imgs_real, imgs_fake)
            else:
                IS_score, FID_score = get_IS_FID_score(metric, generator, train_loader, dataset_name, split, alpha=alpha, step=step)

            status_avg_r = get_status(average_score_real_mean, higher_is_better=True)
            status_avg_f = get_status(average_score_fake_mean, higher_is_better=False)
            status_avg_o = get_overall_status(average_score_real_mean,
                                              average_score_fake_mean,
                                              IS_score=IS_score,
                                              min_mu=1.2)
           
            real_stats_avg_str = f"D_real_avg: {status_avg_r} {average_score_real_mean:>+.4f} ± {average_score_real_std:<+.4f} 📈"
            fake_stats_avg_str = f"D_fake_avg: {status_avg_f} {average_score_fake_mean:>+.4f} ± {average_score_fake_std:<+.4f} 📉"

            dloss_avg_str = f"DLoss(Avg): {d_loss_mean:.4f}"
            gloss_avg_str = f"GLoss(Avg): {g_loss_mean:.4f}"

            is_score_str = f"IS: {IS_score[0]:.4f} ± {IS_score[1]:.4f})"
            fid_score_str = f"FID: {FID_score:.2f}"

            gp_str = f"GP[avg]: {gp_mean_epoch:.2f}"
            
            summary = f"{dloss_avg_str} | {gloss_avg_str} | {is_score_str} | {fid_score_str} | {gp_str}"
            
            print(f" -- {status_o} Last Batch : {d_real_stat_str} | {d_fake_stat_str}")
            print(f" -- {status_avg_o} Epoch's Avg: {real_stats_avg_str} | {fake_stats_avg_str}")
            print(f'[{res}x{res}][Epoch {epoch}/{epochs}] {summary}')
            
            #save model weights at each epoch
            checkpoint_dir = f"{weights_save_dir}/progan_{dataset_name}_{loss_type}_{experiment_date}"
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            torch.save({"disc_state_dict":discriminator.state_dict(),
                        "gen_state_dict":generator.state_dict(),
                        "gen_ema_state_dict":ema_generator.state_dict(),
                        "disc_optimizer":disc_optimizer.state_dict(),
                        "gen_optimizer":gen_optimizer.state_dict(),
                        "z_size":generator.z_size,
                        "lr_d":lr_d,
                        "lr_g":lr_g,
                        "max_steps":discriminator.max_steps,
                        "decay_step":decay_step,
                        "noise_addition":noise_addition,
                        "wgan_range":wgan_range,
                        "step":step,
                        "training_step_counter":training_step_counter,
                        "ema_warmup_images_threshold":ema_warmup_images_threshold,
                        "ema_warmup_images_seen":ema_warmup_images_seen,
                        "epoch":epoch,
                        "epoch_list":epoch_list,
                        "batch_size_list":batch_size_list,
                        "lambda_factor":lambda_factor,
                        "gen_update_interval":gen_update_interval,
                        "loss_type":loss_type,
                        "FID":FID_score,
                        "IS":IS_score,
                        "d_loss_mean":d_loss_mean,
                        "g_loss_mean":g_loss_mean,
                        "all_training_losses":all_training_losses,
                        "all_gradient_penalties":all_gradient_penalties,
                        "dataset_name":dataset_name,
                        "split":split,
                    }, f"{checkpoint_dir}/checkpoint_step_{step}_{experiment_date}.ckpt")
        
            # generate some images mid training to evaluate our model's performance 
            with torch.no_grad():
                gen = ema_generator.eval() if use_ema_inference else generator.eval()
                
                generated_images = gen(fixed_z, alpha, step)
                
                ema_marker_str = "[EMA]_" if use_ema_inference else ""
                loss_str = f"(dLoss:{d_loss_mean:.6f} | gLoss:{g_loss_mean:.6f}"
                lrs_str = f"{current_lr_d:.0e},{current_lr_g:.0e}"
                title_str = f"Step {step} [{res}x{res}, α={alpha:.2f}] with {loss_type.upper()} @ Epoch {epoch} FID:{FID_score:.2f} {loss_str} [{lrs_str}]"
                save_path=f'{images_save_dir}/progan_{loss_type}/{dataset_name}_{experiment_date}/{ema_marker_str}step_{step}_{res}x{res}_epoch_{epoch}.jpg'
                
                display_images(generated_images, 
                               cols=gen_num_samples//8,
                               title=f"{ema_marker_str}{title_str}",
                               unnormalize=True,
                               save_path=save_path,
                               figsize=(16,8))
                
                # save the original images only when ema is enable, 
                # otherwise its already being saved/displayed
                if keep_raw_generations and use_ema_inference:
                    generated_images = generator(fixed_z, alpha, step)
                    display_images(generated_images, 
                                   cols=gen_num_samples//8,
                                   title=title_str,
                                   unnormalize=True,
                                   save_path=save_path.replace(ema_marker_str,""),
                                   figsize=(16,8))
    
    print("ProGAN training is complete!")


#%%progan training
print(f'Training PROGAN!')
loss_type = 'wgangp'
# initially set to 10, but during trainig since 64x64,
# discriminator was not performing well and constantly
# failed, this might be due to strong/large lambda factor!
# so im using a smaller value for now!
# update: that wasnt the issue 10 is ok, 5 is ok as well!
lambda_factor=5
# cifar10 is a lot harder than celeba. try celeba first
# and then cifar10 if you like
dataset_name = 'celeba'
split = 'train'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# original paper uses 512
z_size = 512
# 7 means 4x4 up to 256x256
max_steps = 7
# update:
# my initial implementation used a loop, and this somehow caused memory leakage
# at each step in jupyternotebook (it wouldnt release someo of the previously 
# allocated vram and this cuased the vram to quickly run out!
# additinally we have started with 1024 as the largest channel size which took
# a huge amount of time for training (5+hours only up tp 32x32 and 9.8GB vram,
# (64x64 without goingout of memory but the training started to destablize becasue
# of large lr at that stage so I decided to end the trainig.)
# for the new round I decided to halve the channel numbers. so now we are going with
# [512,256,128,64,32,16,8] 
 
# up to 64x64 it takes around 6.4~7GB, 128 aroud 7.4 basically its below 10G
# batchsize is extremely important, small batchsizes directly mess up the result
# see page 26 in the paper for a few examples.(also I faced this first hand!)
# so to get the best results always use the largest batch we can!
BATCH_SIZES = [128,128,128,128,64,32,16]
# the more epochs the better result we get, despite the FID 
# that may flucturate but the image quality 100% gets better
# with more epochs.
# cifar10 is way harder than celeba, so more epochs for each 
# step is required. I used EPOCHS = [10,20,50,100,50,70,100]
# and got somewhat decent results. I didnt use more epochs for
# 64x64 and larger res because I thought image quality wouldnt be good
# but in 64x64 the quality actually improved (images became more welformed
# and detailed, despite the fact that the original cifar10 is 32x32 and
# when upsampled the output will be very blurry. but when you zoomout 
# the results you'll notice the differen. use more epochs for
# 5th and other steps as well to possibly get even better results,
# though I myself would allocate more epochs to 32x32 and then 64x64
# 
# for celeba this is what I used, use more to get better
# EPOCHS = [10,10,10,20,40,40,40]
EPOCHS = [10,10,10,20,40,40,40]
gen_update_interval = 5 if loss_type == "wgan" else 1

#discriminator
discriminator_progan = DiscriminatorProGAN(max_steps)
discriminator_progan = discriminator_progan.to(device)
#generator
generator_progan = GeneratorProGAN(z_size, max_steps)
generator_progan = generator_progan.to(device)

#sidenote:
# in Adam optimizer beta1(the first value for betas) controls the momentum,
# (i.e. it controls the exponential moving average of the gradients) setting
# it to 0 essentially disables it.
# beta2 on the other hand controls the exponential moving average of the 
# squared gradients which represents its scale or variance(of the gradients)
# a lower beta2 makes the optimizer react faster to recent gradient magintudes and likewise
# a higher beta2 would smooth things out and hence slow its reaction to recent gradient magnitudes!
#
# lets make it a bit more clear, imagine for example we chose beta2=0.9 this is now like
# we gave the optimizer a very short memory! since it has a short memory it now reacts very
# quickly to the scale of the gradients it faces in the last few batches and if we for example
# get a few batches with small gradients the optimizer's internal scaling factor can shrink 
# and therefore cause the next step to be huge! which obviously will lead to an overshhoot 
# and huge updates!(i.e. large parameter updates)
# (Adam devides the learning rate by sqrt(v_t), so when v_t quickly shriks because of the 
# small gradienst and short memory the denominator becomes small which in turn would increase
# the impact of the learning rate resulting in a large/huge optimizer step!)
# if we chose a larger beta2 like 0.99, it will provide a much longer memory, therefore
# the estimates of the gradient variance will be much smoother and more stable.
# the optimizer therefore wont take sudden massive steps anymore which would otherwise make 
# training very unstable and make the gradient penalty very high! hence why larger beta2 
# like 0.99 make training more stable and gradients magnitudes more smooth.
# (so too small beta2 can cause instability (huge weight updates, ossiliations) and 
# too large values can also make updates very slow and slow the convergence)
#
# update:
# change betas from 0.99 to 0.999 to make 32x32 stage stable
# need to change this if things went south! 
# changed it back to 0.99 as that didnt fix the issue
# update: after introducing several fixes in the implementation
# and adding equalized learning rate, 0, 0.99 is the right choice
# and works great. see debug log at the end.
# I have not tested the lsgan/wgan though so their numbers
# are from previous tests, in paper, the lsgan used the same
# betas as wgangp, plus noise addition. 
# see debug log for information
betas = [0.5, 0.999] if loss_type=='lsgan' else [0, 0.99]

if loss_type=='lsgan':
    lr_d, lr_g = 0.0004, 0.0001
elif loss_type == 'wgan':
    lr_d,lr_g = 0.00001, 0.00002#1e-5, 2e-5
else:#wgangp
    # lrs like 0.001/0.0003 all result in huge gradient penalties (gp)
    # which would result in huge losses! this meant we had huge updates
    # and unstale training so I hd to reduce the learning rate drastically!
    # we need to tune this during training for the next steps.
    # the first step is very senstive so we had to keep lr low
    # the larger resolutions require careful tuning as well as we get to
    # later steps, the resolution becomes larger, it becomes way harder to
    # get things right and discrimnator/genertaor clash can destablize trainng
    # really quickly (see the logs below)
    # update: 
    # part of this issue was I missed tanh from toimg layers in generator
    # when I fixed that lr=0.0001 for both disc and gen lead to generator
    # constantly having lower loss than discriminator (which wasnt the case
    # before. so I increased the lr for discriminator to see how it goes)
    # ok no matter what I do the the generator gets lower loss (I used x7
    # larger lr for disciminator but still no luck(I even got large gp which is bad
    # so I need to change it. reverted it back to 0.0001 for both.
    # see debug log ahead!)
    # update:
    # started the training with larger lr for discriminator (3e-3) than 
    # the generator (2e-3), until we hit 64x64, at which point I noticed
    # discriminator constantly overpowers the generator enough to not
    # allow the generator create more fine details, even more epochs wouldnt
    # help, the generator loss would stay around the same thing, and images
    # look smeared. I went to 128x128 and this didnt resolve. so I resumed
    # from stage4, increased its epoch to 35 (it was end of step4 with 30 epochs)
    # and set the lr_d and lr_g back to 0,0001 for both and set decay_step=4
    # as well, lowering and then halving the learning rates. then resumed the
    # training. by this we finally managed to get rid of those smear like patterns
    # liquaady pattern around the hair,face which can be seen in experiments (celeba_20250925180621,
    # celeba_20250926072732 and celeba_20250926192919).
    # update:
    # start with lrs = 3e-4/2e-4 or 2e-4/2e-4 up until 64x64
    # for 64x64 decrease lr_d and increase lr_g so generator
    # only a tiny bit so it doeesnt lose and doesnt overpower
    # the discriminator!
    # update:
    # after implementing equalized leanring rate layer, we 
    # can use large lrs like 0.001 for both and not decay at all
    # we will get very decent images! see debug logs ahead!
    # 
    # update for cifar10:
    # for cifar10 0.001 is too much lr_d, lr_g = 0.0005, 0.00052
    # seems to be stable (close but gen is a bit larger this makes 
    # them roughly equale with a bit of leeway for generator to be
    # able to catchup when it gets hard in highe res. 
    # this is by the way for default 7m size for cifar10 we probably
    # need to use a much smaller network cuz the data is much less
    # compared to celeba (60k vs 160k)) we could also use more epochs
    # cuz there are fewer images in cifar10 and it could lead to instablity
    # in later epochs becasue previous ones arent trained enough!
    # update: 
    # cifar10 is a lot harder than celeba and I mean a lot alot harder
    # not only the images are small themselves, they are only afew tens
    # of thousands of images.
    # using a smaller network can be ok, because too complex of an architecture
    # can make things harder, (make discriminator much powerful, overfitting issues, etc
    # so a moderate size/ is ok. usually discriminator can be made simpler
    # and generator might need a bit more help but you need to find that out 
    # by trial and error, first use the same arch for both and establish a baseline
    # like what I did, and then try different hyper parameter. I noticed the 7m
    # gives better faster convergence than the 400k version (but I could be wrong
    # cause I didnt test the 400k with more epochs, so I guess I should do that later!)) 
    # so to get a somewhat decent output for cifar 10 the rule I found is
    # 1.try as many epochs for each step as possible, higher res require more
    # if you dont train the previous step properly, the next step will face issues!
    # 2.use large lr as long as its possible. it makes convergence much faster
    # only if you see mode collapse or no improvements for an extended period 
    # of time decay the lr. starting with lower lrs makes things much much harder
    # It might seem its the right choice, but then you quickly need to deal with
    # one overpowering the other, or if not tolerate the slow convergence. 
    # what I found is that the generator needs to be a tiny bit larger than discrimnaor
    # (for example like my previous test lr_d, lr_g = 0.0005, 0.00052, this creates 
    # a healthy feedback between them and always allows the generator to have leewayto
    # improve so if it came down to choose alower lr, follow this unless you see it needs otherwise
    # which is rare in our case with cifar or any hard dataset where generator may be struggling more)
    # 3. you must get 16x16 to a decent level, things should be formed at that
    # level. if you look at my example experiments you see that, of course they
    # are extrelemly low res, but if you zoom out in matplot lib, you should easily
    # see the objects patterns are there but very low res. 
    # sidenote:
    # during training even at 16x16 you will see weird patterns and may thnk we have mode collapse
    # dont end the trainng, let it train fo rmore, the network is trying its best to
    # add more details around the low res objects it found. if you look closely youll see
    # initially I thought i was hitting mode collpase and would end until one time I said
    # left and when I cameback I noticed things had improved drastically, there I found about ths
    # tip!)
    # 
    # from 16x16 to higher res may take a lot of time. that is 32x32 maynot give us the crisper version.
    # at least for my experiments I didnt get it. I got more detailed versions in 64x64
    # (i.e. when you zoome out you can clearly see a clearer detailed objects, 
    # but up close they are blurry cuz the original images are 32x32 themselevs
    # and their upsampled versions cant be that great!) for me, horses, dogs
    # ships and sometimes trucks/cars were visible if zoomed out, birds werent
    # always well formed or detailed, (im writting this and watching epoch 27/50 in 64x64
    # so given more epochs we should get more detailed and better formed images.(at 37/50 things
    # got indeed better and I can see birds and cars, frogs much better! even at 128x128 we
    # are improving! more well formed/defined objects are clear now)
    # for cifar10 my lrs are lr_d, lr_g = 0.001, 0.001 and I decay_step=4 (initially I
    # also tested with decay_step=2, but 4 seems ok. the experiment I wrote about here
    # is 20251003124237 
    # sidenote2:
    # the fake_mean can be better, but given the quality im ok! we can definitely spend 
    # more time on cifar and get more decent /welformed images, but im satisfied with
    # the current result! also I almost forgot! always let the model train, dont just
    # stop the training the moment you see some 1 digit positive fake scores,
    # let it train! if you dont see mode collapse, ugly artifacts, let it train and you'll
    # see it improves if not you know up until where it was ok with what lr, and you can
    # decide what to do accordingly.(also look at the std aswell see debug logs ahead for
    # more ifnormation read and then comeback here))
    lr_d, lr_g = 0.001, 0.001 #0.0003, 0.0002

# no need to decay now!
# for celeba we dont decay, so we use decay_step=7
# but for cifar10 I used decay_step=4
decay_step = 7#4#3#2

# use ema of generator weights for inference/ mid traiing visualization
# note this is only for visualization/reporting, we calculate ema and 
# save it in the checkpoints anyway! note choosing ema_warmup_images_threshold
# is crucial to get a working ema! otherwise it will become solid grays!
# warmup is different dataset to dataset! I explained this in trainig loop
# read that if you missed!
# update : fixed it see debug log 
use_ema_inference = True

# disc_optimizer = torch.optim.RMSprop(discriminatorI64.parameters(), lr=5e-5) # for wgan
disc_optimizer = torch.optim.Adam(discriminator_progan.parameters(), lr_d, betas=betas)
gen_optimizer = torch.optim.Adam(generator_progan.parameters(), lr_g, betas=betas)

#compile the models for faster training!
# update: it doesnt work for models that have double backwardpass!
# discriminator_progan.compile()
# generator_progan.compile()
#
# todo: next rampup the lr to 0.001 and dont decrease for any steps
# just like the paper, see why ours fail! it shouldnt fail
# if it failes, then use conv3x3! it might be generator needs that
# power to work well with ihgher lr! the original paper arch is 23m
# while ours is 7m!
# ok everything went fine adn got great results! 
#
training_loop_progan(discriminator_progan,
                     generator_progan, 
                     disc_optimizer=disc_optimizer,
                     gen_optimizer=gen_optimizer, 
                     epoch_list=EPOCHS, 
                     batch_size_list=BATCH_SIZES,
                     gen_update_interval=gen_update_interval, 
                     dataset_name=dataset_name,
                     split=split,
                     loss_type=loss_type, 
                     lambda_factor=lambda_factor,
                     wgan_range=(-0.02, 0.02),
                     noise_addition=False,
                     device=device,
                     resume=False,
                     # paper uses ema enable if you like and it 
                     # must only be enabled for fresh training 
                     # not resumes!
                     use_ema_inference=use_ema_inference,
                    # ema_warmup_images_threshold=3000_000, for cifar10, celeba may need adifferent value
                    keep_raw_generations=True,
                    # quick_and_noisy_IS_FID=False,
                    # checkpoint_path="./weights/gan/progan_celeba_wgangp_20250930091608/checkpoint_step_4_20250930091608.ckpt",
                     decay_step=decay_step)

#%%
# change some paratemers during experimental resumes!(like add more epochs, change lambda_factor, etc)
# checkpoint_path ='./weights/gan/progan_celeba_wgangp_20250926072732/checkpoint_step_4_20250926072732.ckpt'
# checkpoint_path ='./weights/gan/progan_celeba_wgangp_20250927174948/checkpoint_step_3_20250927174948.ckpt'
# checkpoint_path ='./weights/gan/progan_celeba_wgangp_20250929080324/checkpoint_step_3_20250929080324.ckpt'
# checkpoint_path ='./weights/gan/progan_celeba_wgangp_20250929131133/checkpoint_step_4_20250929131133.ckpt'
# checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
# for k,v in checkpoint.items():
#     if not isinstance(v,dict):
#         print(f'{k:<15} {v}')
#     elif "param_groups" in v.keys():
#         print(f'{k:<15} {v["param_groups"]}')
# 
# checkpoint["lr_d"] = 0.00004
# checkpoint["lr_g"] = 0.000042
# checkpoint["lambda_factor"] = 10
# # since we changed the epochs, lr_d/lr_g wont take effect and instead
# # we need to change the optimizers lr!
# checkpoint["epoch_list"]=[10, 10, 10, 40, 45, 50, 50]
# checkpoint["disc_optimizer"]["param_groups"][0]["lr"] = 0.00004
# checkpoint["gen_optimizer"]["param_groups"][0]["lr"] = 0.000042
# #%%
# torch.save(checkpoint,checkpoint_path)
# for k,v in checkpoint.items():
#     if not isinstance(v,dict):
#         print(f'{k:<15} {v}')
#     elif "param_groups" in v.keys():
#         print(f'{k:<15} {v["param_groups"]}')
# #%%
# all_losses = checkpoint["all_training_losses"]
# all_gps = checkpoint["all_gradient_penalties"]
# idx=5
# plt.plot(np.array(all_losses[idx]))
# plt.show()
# plt.plot(np.array(all_gps[idx]))
# plt.show()
#%% progan debug log
# sidenote:
#
# reminder before going over for debugging:
# first remember that in gan training the discriminator and generator 
# can not possibly dominate the other one. they should and always be
# in a never ending competition to overcome eachother. so we cant 
# have one get an extremely low loss, while the other has not!
# its basically a cat and mouse game, if anyone wins its over for
# the other one, for example if the discrimnator becomes prefect
# and always identifies fakes 100% of the times, then its gradients 
# vanish and the generator stops learning! its over!(so no prefect model!)
# so instead it should be like this, the generator does a bit better
# then the discriminator needs to catch up, then discriminator does
# better and identifies fakes well and generator needs to catch up
# and this should continue, this is what we say, they need to be in 
# a state of constant equiliberium!(i.e. neither one wins)
# having said that, both can have decreasing loss. wild ossiliations/
# fluctuations is a bad sign, but a downward loss for both is good.
# if one loss is nearly constant while the other one is changing it
# means one model has stopped learning and its a bad sign as well.
# 
# d_real_mean-> we want high positive number that shows on averge how
#               confident the discriminator is in detecting real images.
#               we want high positive score (high confidence for real image detections)
#
# d_fake_mean-> we want small number (even negative) that shows on average
#               how many times has the generator fooled the discriminator.
# d_fake_std:-> std can be low and high and means its a good sign for generator
#               or bad! it depends on the context. for example we
#   high good-> can say a high std is good sign for fake images because it 
#               shows diversity since it shows the generator is generating 
#               very different images that discriminator finds different!
#               and assigns different scores to them!(which we could then
#               interperet as having different quality or types of images
#               which is a good sign) or the otherway around!
#               to be more specific, in the beginning of the training, 
#               since the generator and discrimator are not yet trained
#               properly, we can see high std for fake images because the
#               generator is producing very different images and it can 
#               differ widly from class/image to class/image(i.e. diverse outputs)
#    high bad-> however, as the training continues and both the discriminator and generator
#               are trained more, a high std could also mean either the discriminator
#               has issues (is weak, etc that gives different scores to generators outputs)
#               or the generator faces instability and keeps generating very different 
#               images that are not consistent.(i.e. is weak, lr is high it fluctuates,
#               or it can generate only some images better while some others worse, etc).
#               so a high std can also mean instablity in either discriminator/generator
#               or both if its not at the begining stages of the training.
# medium good-> we want as training continues, the generator produces diverse but
#               consistently high quality images which may lead to a lower std,
#               not high not too low, somewhere in between.
#     low bad-> if on the other hand, the std becomes very low (e.g. near zero)
#               it can mean a mode collapse has occured, because either the generator
#               is producing nearly identical images so discriminator keeps
#               assiging the same score to them hence low std. or it could also mean
#               the discriminator has gone nuts and keeps giving the same scores
#               to all images, which means the trainig is just gone off the rails!
#               this should be visible in the disc_loss, gen_loss as well along with
#               other ways like visualizing generators outputs, or even looking at IS score!
#               so  high std at beginning -> its normal, models are still learning.
#               if  high std at the mid/end -> its unstable discriminator or generator
#               if very low std at the beginning -> its mode collapse, (generator keeps making the same images)
#               if very low std at the mid/end -> its mode collapse
#               if not high not very low -> training is going ok, generator is generating high quality images
#               note, we can have high/low stds during training, but if they presist for
#               a considerable time, then we can say for sure.
#               
#               as you can see, we cant say much without taking other info into account
#               so use other metrics as well that may be easier and more straight forward
#               to get an idea whats going on.
#
# d_real_std:-> like before, the std for real images is a bit nuasanced here aswell
#               because both low and high can be good or bad! for example since the images
#               in datasets are fixed, we cant simply say it shows the diversity
#               of images in our dataset because they are fixed, we dont generate
#               them, so its not about the data itself, rather its about the
#               discriminator's action. therefore we can say it can show the 
#               variable scoring of the discriminator for real images.
#               if its high, then it means the discriminator is giving a lot of
#               different scores,(a wide range of scores) to the real images
#               which could mean either we are at the begining of the trainig so 
#               the discriminator hasnt properly learnt to consistently give 
#               high scores to real images so it has hit and miss here and there,
#               or it simply shows the variety of images(diversity) in the dataset
#               itsel(basically shows the dataset has very different types of images,
#               that some are harder/some are easier for the discriminator to identify
#               hence the different score it gives each. for example in our celeba
#               datasets we have different images with different lightinig conditions,
#               poses, etc that can conttibute to this). 
#               so high score either comes from inefficent/weak discriminator(early in training)
#               or from very different images that make discriminator give different scores!
#               However when the model is trained properly and has learned to consistently
#               give high scores to real images, then the std should be low, espcially
#               if the dataset is uniform and the discriminator assigns a similar
#               score to all real images(which brings the std down) or it could simply 
#               reflect the diversity of images in the dataset. when we have low diversity,
#               and the discriminator learns well to identify and score them all very 
#               highly and that lowers the std down. 
#               
# IS_score   -> we want a mean larger than 1 and a std very low close to 0.
#               if mean is 1 or close to it, its basically bad quality((1,0))
#               std or standard deviation, shows the consitency of the final
#               inception score. it shows the stability of the generator's
#               overall performance through different samples it generated 
#               over time.(remember we split the input into k(10) splits
#               and calculate a score for each split, the std shows the 
#               deviation among these scores. so a small deviation means 
#               we consistently achieved the same score for several splits
#               of its input (basically different subset of our generators output))
#               
# D_loss     -> remember the major part of the loss is simply preds_fake.mean() - preds_real.mean()
#               we want negative loss for discrimnator. this means discriminator
#               is identifying fakes very well. if its positive it means the discrimnator
#               has given larger positive score to fake images than it has given
#               to the real images. so it means the discriminator has flipped!
#               and is doing the opposite of what it should be doing!
#
# G_loss      -> remember the loss is simply -preds_fake.mean() for the generator so
#                we want the loss for generated images be negative. it means
#                the generator has created realistic images that discriminator 
#                is fooled and it has assigned a positive score to it(D(fake)).
#                the generators goald is to maximize D(fake), ie creates realistic
#                images that fools discriminator and get a high positive score, which
#                will be reflected as negative number(because it -D(fake)). 
#                so the more negative gloss the better.
#                if we get a positive gloss, it means, the generator has failed to 
#                come up with good images, it either has collapsed or the discriminator
#                is more powerful and identifies all images as fakes hence a very negative
#                number is given to D(fake) which when -(-D(fake)) becomes a positive loss)
#                at the begining of the training, the discrimnator that cannot
#                identify real/fake well, we might get negative or positive loss
#                but as the trainig goes we want the generator loss to be negative.
#                note if the loss fluctuates its ok as long as we see images getting
#                improved, the gloss and dloss need to be in constant back and forth so
#                we cant see one be prefect, we cant have both of them being prefect
#                just good enough for the images to get improved, see our results at the
#                end you'll understand!
# debugging: 
# initialy I started with lr=0.002/0.001, the first step(0) 
# went on, didnt notice much, until step=1 started and noticed
# the loss was insanely huge! in the hundereds of thousands to millions!
# went back and noticed it started from the begining of the training
# then noticed I had a misake in generator (had PixelNorm in first layer
# where we accept input latent vector) removed it but the problem still
# existed. then noticed the betas in adams, 0.9 was small, it meant
# as I explained before, if we get a few batches of small gradients, adam
# would take huge steps and we ould have huge parameter updates that would
# cause massive instablity in training. made it 0.99 to make it much smoother
# still I was getting massive loss (but a lot smaller now but still huge)
# this time I checked and saw the issue was comming from the gradient penalty
# i.e. gp was huge! which when we added it to our loss our loss would be huge
# as well. this meant again we had massive gradients, and the only source that
# would contribute to this would be the learning rates, I set it to 0.0003 to
# both discriminator and generators, but no luck, the loss dropped to a much smalle
# r value but still it would go to huge magnitues. made it 0.0001 and it worked!
# since the first stage is extremely sensitive , we have to use a smaller lr
# we can then increase it for the following steps, which we did use 0.001!
# which didnt work for step1, turned it down back to 0.0001! and train till the end
# and then we decide what values to experiment with for each step!
# things went smoothly up until 32x32 , when we hit 64x64 resolution, 
# for the first two epochs, the image quality became much better, sharp and detailed
# however as more epochs passed, the images became worse and the loss showed as well:
# 
# [64x64][Epoch 0/10 | Iter: 1272/2544] Disc Loss: -14.984601 | Gen Loss: 51.437641
#  -- Batch-1272: Disc's real mean: -17.3619 | Disc's fake mean = -33.0078
#  -- Last Batch : Disc's real mean: -28.3425 | Disc's fake mean: -32.9185
#  -- Epoch's Avg: Disc's real mean: -58.1945 | Disc's fake mean: -93.2741
# [64x64][Epoch 0/10] Disc Loss-Avg: -25.013562 | Gen loss-Avg: 94.841881 | IS: (μ:1.0000, σ²:0.0000) | FID: 333.02
# 
# [64x64][Epoch 1/10 | Iter: 1272/2544] Disc Loss: -5.097756 | Gen Loss: 0.344297
#  -- Batch-1272: Disc's real mean: 7.5136 | Disc's fake mean = 2.2513
#  -- Last Batch : Disc's real mean: -4.6169 | Disc's fake mean: -6.2919
#  -- Epoch's Avg: Disc's real mean: 3.9837 | Disc's fake mean: -1.3474
# [64x64][Epoch 1/10] Disc Loss-Avg: -4.625465 | Gen loss-Avg: 2.729605 | IS: (μ:1.0000, σ²:0.0000) | FID: 241.13
# 
# [64x64][Epoch 2/10 | Iter: 1272/2544] Disc Loss: -4.328329 | Gen Loss: 22.177746
#  -- Batch-1272: Disc's real mean: -31.3004 | Disc's fake mean = -36.4820
#  -- Last Batch : Disc's real mean: -24.6972 | Disc's fake mean: -30.4969
#  -- Epoch's Avg: Disc's real mean: -10.4536 | Disc's fake mean: -15.6216
# [64x64][Epoch 2/10] Disc Loss-Avg: -4.539772 | Gen loss-Avg: 16.904588 | IS: (μ:1.0000, σ²:0.0000) | FID: 222.88
# 
# [64x64][Epoch 3/10 | Iter: 1272/2544] Disc Loss: -14.829409 | Gen Loss: 44.549088
#  -- Batch-1272: Disc's real mean: -31.8848 | Disc's fake mean = -50.9633
#  -- Last Batch : Disc's real mean: -100.3516 | Disc's fake mean: -153.9123
#  -- Epoch's Avg: Disc's real mean: -36.4500 | Disc's fake mean: -57.4129
# [64x64][Epoch 3/10] Disc Loss-Avg: -16.055016 | Gen loss-Avg: 58.809311 | IS: (μ:1.0000, σ²:0.0000) | FID: 257.49
#
# [64x64][Epoch 4/10 | Iter: 1272/2544] Disc Loss: -82.065460 | Gen Loss: 286.170471
#  -- Batch-1272: Disc's real mean: -244.0014 | Disc's fake mean = -383.1243
#  -- Last Batch : Disc's real mean: -313.4012 | Disc's fake mean: -542.5110
#  -- Epoch's Avg: Disc's real mean: -193.2854 | Disc's fake mean: -317.3992
# [64x64][Epoch 4/10] Disc Loss-Avg: -81.167684 | Gen loss-Avg: 319.669814 | IS: (μ:1.0000, σ²:0.0000) | FID: 336.45

# it seems as we get to larger resolutions, it becomes more sensitive because 
# there are mo details to get right. the discriminator therefore can quickly 
# learn whats missing or is out of place or not just right and reject it as fake
# but the generator cant keepup as it gets small or no gradient from discrimnator
# this is obvious by looking athe discrimnator loss and generator loss. the 
# discriminator loss is small while the generator loss keeps going up. the average
# scorer for real/fake also shows this, and we go from 3.9/-1, to -10/-15 which is 
# still good/healthy but the next epoch -36/-57 shows things started to go south
# and we see the next epoch we have -193/-317 which shows a much worse penalty for
# divergence. at this stage we can see images have completely been destroyed with
# ugly artifacts. so what do we do? the learning rate seems large for this larger 
# resolutions. it seems as we get to larger adn larger resolutions it still is very
# sesitive and everything can quickly destablize. we can lower the learning rate again
# it seems one rule thats been used by people is to halve the learning rate for each
# new resolution to keep things stable. since we have been mostly ok up to 16x16
# we can start halving the lr either at res 16x16 or 32x32. looking at the loss it seems
# 16x16 could use some help(we should halve the lr for 32x32 and most definitely for everything
# after ward). 
# update:
# did that but we faced disgusting images as early as res 32x32. we faced mode collapse!
# looking at the log we can clearly see the discriminator is impeded by the drastic learning
# rate reduction! and this caused the generator to take advantage and try to score lower by
# repeating something that discriminator confusingly accepts! the discriminator ultimately
# catches up but by that time our generator is long gone, stuck in a bad minima and collapsed!
# as we can see in the log before, in 16x16 things are relatively healthy and everything 
# is stable.
# 
# [16x16][Epoch 7/10 | Iter: 636/1272] Disc Loss: -22.291618 | Gen Loss: 49.185047
# -- Batch-636: Disc's real mean: -28.7658 | Disc's fake mean = -59.9654
# -- Last Batch : Disc's real mean: -28.1403 | Disc's fake mean: -57.5411
# -- Epoch's Avg: Disc's real mean: -26.2675 | Disc's fake mean: -54.0882
# [16x16][Epoch 7/10] Disc Loss-Avg: -21.439875 | Gen loss-Avg: 54.244800 | IS: (μ:2.3518, σ²:0.3569) | FID: 214.65
# 
# [16x16][Epoch 8/10 | Iter: 636/1272] Disc Loss: -21.416265 | Gen Loss: 54.202488
# -- Batch-636: Disc's real mean: -25.8276 | Disc's fake mean = -53.0298
# -- Last Batch : Disc's real mean: -25.6820 | Disc's fake mean: -51.4843
# -- Epoch's Avg: Disc's real mean: -26.6747 | Disc's fake mean: -54.6126
# [16x16][Epoch 8/10] Disc Loss-Avg: -21.534471 | Gen loss-Avg: 54.770717 | IS: (μ:2.4247, σ²:0.2343) | FID: 206.51
# 
# [16x16][Epoch 9/10 | Iter: 636/1272] Disc Loss: -21.474728 | Gen Loss: 57.242680
# -- Batch-636: Disc's real mean: -26.2436 | Disc's fake mean = -52.3254
# -- Last Batch : Disc's real mean: -27.3929 | Disc's fake mean: -57.1398
# -- Epoch's Avg: Disc's real mean: -26.7484 | Disc's fake mean: -54.7311
# [16x16][Epoch 9/10] Disc Loss-Avg: -21.575619 | Gen loss-Avg: 54.875121 | IS: (μ:2.3588, σ²:0.1684) | FID: 204.83
# 
# but the moment we go to the next step and lower the learning rate, the discriniator's loss
# gets closer to zero (see loss-avg -12 to -2) and generators loss decreases suddenly as well
# the scores for real and fake images also bcome close (from -34/-50 to -17/-20 in the next epoch)
# and this continues for the next epoch, in 4th epoch, discriminator figures out whats goingon
# and starts rejecting heavily, this is where the discriminator's feedback gets a massive gradient
# for the absolutely horrendeous patterns generator has been producing. this is why our losses 
# (for both disciminator and generator) gets huge all of a sudden in the 4th epoch. 
# the generator tries to react to this masive gradient but it fails and the whole thing goes 
# down the abyss and it diverges and its loss increases once more. at this point generator is
# long gone, and has collapsed!
#
# Files already downloaded and verified
# Step: 3/7 -> Training on [32x32]
# --Epochs:                    10
# --BatchSize:                 128
# --Interval:                  637
# --Fade-in Steps:             6360
# --Current Discriminator LR:  5e-05
# --Current Generator LR:      5e-05
# [32x32][Epoch 0/10 | Iter: 636/1272] Disc Loss: -10.688290 | Gen Loss: 48.339836
# -- Batch-636: Disc's real mean: -34.1420 | Disc's fake mean = -46.8792
# -- Last Batch : Disc's real mean: -23.2028 | Disc's fake mean: -27.7775
# -- Epoch's Avg: Disc's real mean: -34.5333 | Disc's fake mean: -50.6039
# [32x32][Epoch 0/10] Disc Loss-Avg: -12.739969 | Gen loss-Avg: 50.921864 | IS: (μ:2.0661, σ²:0.2811) | FID: 265.04
# 
# [32x32][Epoch 1/10 | Iter: 636/1272] Disc Loss: -3.317345 | Gen Loss: 22.969427
# -- Batch-636: Disc's real mean: -16.7480 | Disc's fake mean = -20.5004
# -- Last Batch : Disc's real mean: -8.4562 | Disc's fake mean: -10.6902
# -- Epoch's Avg: Disc's real mean: -17.4643 | Disc's fake mean: -20.5115
# [32x32][Epoch 1/10] Disc Loss-Avg: -2.776609 | Gen loss-Avg: 20.925267 | IS: (μ:2.2124, σ²:0.2590) | FID: 193.58
# 
# [32x32][Epoch 2/10 | Iter: 636/1272] Disc Loss: -2.580442 | Gen Loss: 17.744585
# -- Batch-636: Disc's real mean: -8.2727 | Disc's fake mean = -10.9937
# -- Last Batch : Disc's real mean: -11.4209 | Disc's fake mean: -15.2118
# -- Epoch's Avg: Disc's real mean: -10.2747 | Disc's fake mean: -13.2658
# [32x32][Epoch 2/10] Disc Loss-Avg: -2.736482 | Gen loss-Avg: 13.715730 | IS: (μ:2.3364, σ²:0.3356) | FID: 187.93
# 
# [32x32][Epoch 3/10 | Iter: 636/1272] Disc Loss: -7.093531 | Gen Loss: 17.955961
# -- Batch-636: Disc's real mean: -8.9597 | Disc's fake mean = -17.0247
# -- Last Batch : Disc's real mean: -12.4538 | Disc's fake mean: -33.4308
# -- Epoch's Avg: Disc's real mean: -9.4872 | Disc's fake mean: -18.2752
# [32x32][Epoch 3/10] Disc Loss-Avg: -7.362884 | Gen loss-Avg: 18.680397 | IS: (μ:2.3080, σ²:0.2140) | FID: 208.76
# 
# [32x32][Epoch 4/10 | Iter: 636/1272] Disc Loss: -30.003349 | Gen Loss: 79.969521
# -- Batch-636: Disc's real mean: -61.7957 | Disc's fake mean = -106.9964
# -- Last Batch : Disc's real mean: -80.1722 | Disc's fake mean: -138.7608
# -- Epoch's Avg: Disc's real mean: -52.8957 | Disc's fake mean: -95.7904
# [32x32][Epoch 4/10] Disc Loss-Avg: -31.294711 | Gen loss-Avg: 96.287702 | IS: (μ:2.5499, σ²:0.3617) | FID: 247.23
# 
# [32x32][Epoch 5/10 | Iter: 636/1272] Disc Loss: -49.656693 | Gen Loss: 195.336334
# -- Batch-636: Disc's real mean: -93.0573 | Disc's fake mean = -157.1307
#
# the balck circles and repeated patterns as I said, is a sign the generator is trying to
# win by repeating what it has found to fool the discriminator and it tells us the discriminator
# has been impeded by slow update(it takes a long time to react to generators wrong behavior)
# when we decreased its learing rate. so what do we do? not what we just did! i.e. we need to
# keep discriminators lr the same like the last time that worked or tune it properly so its
# not too small so generator doesnt take advantge!
# I set the discriminator to have a bit larger lr lets see how this goes
# update:
# I multiplied discriminator lr by 1.5 it was good for 1 epoch and then it overpowered
# the generator!lets make the difference a bit lower like 1.2 and see how it goes
# it didnt work and generator loss went south in 3 epochs. going with 1.1 didnt change
# anything it was insignificant and lead to mode collapse. trying to decay lr in step4
# isntead and see how that goes. that didnt work either. using default lr again didnt
# work either. 
# update:
# disabled optimizer states when manually setting new decayed lr : by doing this I noticed
# when we resumed, the gp warnings quickly disapeared at each batch the gp magnitude decreased
# from the intial 950! to 627 to 444 and ater a few other batches down to below 100! so the
# optimizer state reset actually did something!
# that was fixed by reseting the optimizer state, but the original issue of mode collapse 
# in 32x32 (epoch 3) stays. I tred a larger beta2=0.999 didnt solve the issue. it hit me
# that it could be the alpha or the short number of epochs for that resolution. 
# I changed the alpha decay rate and instead of 50/50% of the whole trainig iteration
# I allocated more (i.e. 80%) to the lower res so we have more time to learn properly
# at step3 altha stays at 0.50 previously it was 0.80, and the loss still is low but I can
# see the images quality sometimes drop but goes up the next epoch nonetheless.
# I guess I dont decay the lr anymore and instead work on the more epochs for higher 
# resolutions and see whether it fixes the issue (I might not need to change alpha,
# just more epochs might do it!) ok simpl in creasing the alpha ratio for low res didnt work
# the generator loss started increasing in epoch 4+, which means discriminator learnt whats
# fake and overpowered the generator.lets do the opposite lower alpha for low res and see how
# it goes. Im facing the same issue. using more epochs and by extension extending alpha
# seemed to work at first but later on I faced the same issue, the loss for both started to
# get really large (i,e, 140/190 etc) so I changed again and tried running this with noise=True
# and more epochs for 8x8 and larger res to see if that helps
# update:
# I noticed I missed tanh from toimg layers in generator which output images!
# I also set bias=False for all conv layers as the paper did the same!
# when I started training again, I noticed now the generator has lower loss than 
# the discriinator! it consitently was lower than discriminator from the very begiing
# up to the 16x16 res epoch 5 where I ended the experiment.
# now I want to increase discriminators loss to 0.0002 so its larger than generator!
# that didntg help. even going as high as 0.0007 didnt help. the generator loss would
# still be lower and we would also face exploding gp! it was bad andtrainig wsant going
# smoothly. so I ended it.
# update:
# I simplified my fromImg and ToImg layers, and instead use a simple 1x1 conv and leaklyrelu
# for discriminator(fromImg) and 1x1conv with tanh for toImg. this stablized the training a lot!
# however the loss for discriminator and generator although very good/healthy (1 digit loss)
# are very close, very close to each other, and it went on for many epochs up unti 16x16 epoch 13
# which I ended the trainig. none of them could overpower the other, or get the higher hand!
# and it seemed we were going at a very slow pace if at all!
# update2:
# I noticed I missed a crucial part of Progan which was Equalized learning rate, basically
# dynamically scaling conv layers weights so the gradients are uniform and we dont face 
# exploding or vanishing gradients issue. 
# update:
# after adding Equalized learning rate modules, the loss has drastically decreased! and the
# trainig has been way more stable!
# update:
# starting with 64x64, we can see images become much sharper, but they are malformed or smeared
# looking, as if someone draw a paint brush, that kind of smearness!, this is seemingly normal
# as the netowkr has learned general structures in the image and it hasnot yet known or figured 
# out the fine details, like exact hair strands, colors, face elements, eyes, basically all the
# fine details! and from there it tries to learn those in higher resolutions. this is directly
# realted to how training goes, if we choose proper settings, images develop with much better
# details early on (my own experience), but none the less the transition from lower to higher 
# res seems to always have these kinds of artifacts to some extend (depending on res and trainig
# condition.) also sometimes the generator loss becomes positive, while loss is negative, let it
# train for more, usualyy it learns a long the way and things improve otherwise when it grows! then
# its time to end the traiing we dont want large positive loss for generator (or discriminator!)
# update:
# I couldnt get rid of the artifacts, no matter what! I tried more epochs, different lrs,..
# when we got to higher res, we either faced mode collapse with severely artifact driven images!
# or when mode collapse wasnt happenig, the severe artifacts and bad quality persisted! itwasntjust working!
# until it dawned on me to reset optimizers for each step and treat each step as a seperate 
# thing! I had done it previously when I wanted to lower the lr, but not for each step and with
# that training never went smoothly, it was never about high lr(it was, but the actual underlying
# issue was the large gradients and moving average from previous step that messed up the new more
# sensitive step). 
# update:
# after I reset the optimizers for each step, and increased epochs for larger res like 32x32,
# I started getting massively better image quality at 32x32! images formed properly
# but less detailed obviously!(due to being 32x32!) the training became much more stable, 
# losses became so much more well behaved. it seems obvious now, but the gradient magnitudes 
# and moving average of the previous stage would hurt the new higher resolution stage and 
# make it go haywire completely! when I reset the optimizers for each stage, it became so 
# much better! we now get fid of 36! previously I couldnt imagine this we were hovering in
# 200s, 300s, or at best 150s! we had to do this all along!
# we could have really not faced any of previous issues had we done this! (in retrospect 
# aside from several days of training with different hyperparameters, we got to do a lot of
# debugging and learn a lot as well which is a good thing!)
# update:
# started the training with larger lr for discriminator (3e-3) than 
# the generator (2e-3), until we hit 64x64, at which point I noticed
# discriminator constantly overpowers the generator enough to not
# allow the generator create more fine details, even more epochs wouldnt
# help, the generator loss would stay around the same thing, and images
# look smeared. I went to 128x128 and this didnt resolve. so I resumed
# from stage4, increased its epoch to 35 (it was end of step4 with 30 epochs)
# and set the lr_d and lr_g back to 0,0001 for both and set decay_step=4
# as well, lowering and then halving the learning rates. then resumed the
# training. by this we finally managed to get rid of those smear like patterns
# liquaady pattern around the hair,face which can be seen in experiments (celeba_20250925180621,
# celeba_20250926072732 and celeba_20250926192919).
# update:
# things didnt go as I expected for 128x128. 32x32 was really good, 64x64 really improved
# but 128x128 not at all.
# update:
# spotted a bug in discriminator! I had used slop="0.02" instead of 0.2 for fromImgs!which
# had weakened the gradient flow 10x! fixed it and im not trying to test if this helps!
# update:
# after the fix tried training with lr=2e-4 for both discriminatr and generator. 
# it went kind of smoothly up to 32x32, we achieved roughly the same FID and quality
# that we achieved previously with lr_d=3e-4 and lr_g=2e-4, a bit worse this time.
# the 64x64 started and itw as roughly the same, maybe a bit worse cuz we used lower lr
# this time. however the 128x128, not only didnt improve, but it got worse, we went from
# FID 34 in 32x32 to fid 79 in 64x64 to 160 in epoch 21/30 for 128x128! something needs
# to change! starting from 16x16 we need to be getting good quality, 4x4 and 8x8 we dont
# have images so we cant say much, FID cant be used and its in 200/300s,but when we get
# to 16x16 we should be having proper general form of images, and by 32x32, we should have
# decent looking images, a lbeigt low resi! but prefectly identifiable, and likewise 64x64
# needs to make it much clearer looking. so if we are not improving and seeing good quality
# images at any of these steps something is wrong! also the changes I did caused our memory
# consumption to drop drastically! (removing loops in discriminator) previously when we hit
# 128x128, we would have hit 9.6GB but now its 6.5!
# update:
# im trying to resume from 32x32 and use lr=3e-4 and lr2e-4 to see how it goes and I wont
# go beyond 64x64 until I get decent images, when I do then im sure 128x128 shouldnt be
# an issue! I have spent too much time on this cuz it takes too long to train!
# update:
# resumed from the step3 (begiing of 64x64 step4) with the lr that i mentioned just now
# (3e-4/2e-4) initially everything seems to be going well, the fid was around 40/41 
# but as training went, each epoch it got worse! it fake got positive stores which
# says discriminator cant identify fake images properly and struggles to do that! 
# since we already used a larger lr for discriminator, we can increase its rate of update
# by seting gen_update_interval to something like 2 or more, but I guess this is wrong
# and will cause discriminator to dominate the generator. another thing that could be
# happening is that we might have been using the wrong lambda_factor, i.e. its too large
# and limits/constrains the discriminator too much (remember the weight initiatlization
# issue we had for wgan?) this might be it. so now im going to use a much smaller gp and
# see how it goes!
# update:
# I printed the average gp to see if the lambda was large, its around 0.1~0.15 which is great
# it shows its stable and not exploding like before we introduce equalized learning rate and
# other stuff we did. so far so good however, starting with 64x64 generator starts to struggle
# the fake_mean has become positive, a large positive number, which means generator has failed 
# to fool discriminator repeatedly(discriminator asssigned negative scores to signify its fake!
# and did it with a good confidence! -fake_mean thus becomes a large positive number!) and it shows
# the FID quickly drops from 40 to 68 to 78 etc.(also I lowered the lambda_factor to 5 it seems
# this makes it more stable? i.e. I get lower FID/better fake-mean when resuming with lower lambda_factor)
# the balance between discriminator and generator was great up until now, but now we need to make
# generator a bit more powerful so it can deal with all the new missing detailes from new higher res!
# this is the resume log by the way :
# Step: 4/7 -> Training on [64x64]
# [64x64][Epoch 0/30 | Iter: 1272/2544] Disc Loss: -3.2945 | Gen Loss: -1.5743
# -- 😱 Batch-1272:  D_real_avg: 😎 +14.0373 ± +2.9058 📈| D_fake_avg: 😵 +10.5777 ± +1.9900 📉
# Using FID stats for celeba-train_10K from cache...
# -- 😱 Last Batch : D_real_avg: 😎 +7.2324 ± +3.9085 📈 | D_fake_avg: 😵 +4.1415 ± +2.5747 📉
# -- 😱 Epoch's Avg: D_real_avg: 😎 +7.3060 ± +7.3060 📈 | D_fake_avg: 😵 +3.4679 ± +6.6141 📉
# [64x64][Epoch 0/30] DLoss(Avg): -3.5450 | GLoss(Avg): -2.5923 | IS: 2.7384 ± 0.0461) | FID: 40.64 | GP[avg]: 0.06
# 
# [64x64][Epoch 1/30 | Iter: 1272/2544] Disc Loss: -3.7887 | Gen Loss: -6.4448
# -- 😱 Batch-1272:  D_real_avg: 😎 +7.5593 ± +2.7882 📈| D_fake_avg: 😵 +3.3704 ± +2.2962 📉
# Using FID stats for celeba-train_10K from cache...
# -- 😱 Last Batch : D_real_avg: 😎 +9.5385 ± +2.6258 📈 | D_fake_avg: 😵 +4.2515 ± +1.4386 📉
# -- 😱 Epoch's Avg: D_real_avg: 😎 +8.6388 ± +8.6388 📈 | D_fake_avg: 😵 +3.7770 ± +6.8083 📉
# [64x64][Epoch 1/30] DLoss(Avg): -4.2924 | GLoss(Avg): -2.8783 | IS: 2.6073 ± 0.0502) | FID: 41.06 | GP[avg]: 0.11
# 
# [64x64][Epoch 2/30 | Iter: 1272/2544] Disc Loss: -4.9872 | Gen Loss: -7.5516
# -- 😱 Batch-1272:  D_real_avg: 😎 +6.2007 ± +2.3118 📈| D_fake_avg: 😟 +0.3874 ± +1.5063 📉
# Using FID stats for celeba-train_10K from cache...
# -- 😱 Last Batch : D_real_avg: 😎 +8.7716 ± +2.5128 📈 | D_fake_avg: 😟 +3.4392 ± +2.1746 📉
# -- 😱 Epoch's Avg: D_real_avg: 😎 +7.6186 ± +7.6186 📈 | D_fake_avg: 😵 +1.9932 ± +7.6857 📉
# [64x64][Epoch 2/30] DLoss(Avg): -4.8152 | GLoss(Avg): -1.0651 | IS: 2.6001 ± 0.0313) | FID: 68.66 | GP[avg]: 0.16
# 
# [64x64][Epoch 3/30 | Iter: 1272/2544] Disc Loss: -4.2191 | Gen Loss: 11.4249
# -- 😀 Batch-1272:  D_real_avg: 😵 -0.4766 ± +3.5924 📈| D_fake_avg: 😎 -5.3919 ± +3.7549 📉
# Using FID stats for celeba-train_10K from cache...
# -- 😀 Last Batch : D_real_avg: 😵 +9.7364 ± +1.9480 📈 | D_fake_avg: 😎 +4.9699 ± +1.6249 📉
# -- 😱 Epoch's Avg: D_real_avg: 😎 +6.9414 ± +6.9414 📈 | D_fake_avg: 😵 +1.5484 ± +6.9933 📉
# [64x64][Epoch 3/30] DLoss(Avg): -4.6348 | GLoss(Avg): -0.7618 | IS: 2.7645 ± 0.0490) | FID: 78.97 | GP[avg]: 0.15
# 
# [64x64][Epoch 4/30 | Iter: 1272/2544] Disc Loss: -3.3210 | Gen Loss: 7.5638
# -- 😱 Batch-1272:  D_real_avg: 😎 +7.3455 ± +2.8731 📈| D_fake_avg: 😵 +3.3457 ± +2.5463 📉
# update:
# lets use lr_d=0.0002 and lr_g = 0.00025 (decrease discriminator abit and increase generators a bit)
# ok it was good for 1 epoch! we started with FID37, but the next epoch it went up to 47! 
# I guess we need to crank the lr more for generator! or better make discrinator slower to
# response
# update:
# set lr_d = 0.0001, and keep lr_g=0.0002 (with lambda_factor=5)
# that made discriminator stronger! it seems lower lr made it less jumpy! and do better
# so instead now Im doing the opposite, lowering lr_g to 0.0001 and lr_d=0.0002!
# ok it made it much worse! generator needs to be made more powerful!
# update:
# make it more powerful. decied to increase 32x32 step by 10 epochs with 
# lr_d=0.0002 and lr_g=0.0003 and see how it affects the 32x32 step and whether
# training more in previous step would help the next step
# update:
# the combo lr_d=0.0002 and lr_g=0.0003 didnt work and generator quickly overpowered the discrinator
# and we got two consecutive bad epochs which I then ended.made it lr_d=0.0002 and lr_g=0.00025
# instead to see how it goes.
# update:
# this time around the generator constantly overpowered the discrimnator!
# update:
# resume the new 64x64(checkpoint_step_3_20250929080324) with the same lr = 2e-4 failed with discriminator overpowering geneator
# in 2~3 epochs.
# update:
# resume the new 64x64(checkpoint_step_3_20250929080324) with lr_d=0.0002 and lr_g=0.00022:
# this actually worked and after several epochs, generator finally pulled ahead and seems on par
# with the discriminator (things seems balanced now), and we improved FID ever since.
# we might be able to make convergence faster by using larger lr_g, like 0.00023 or even 0.00025
# without hurting discriminator that much. at this rate, I guess we need more epochs for 64x64
# and 30 epochs is not enough(@25 we have FID=68.62) also looking at later epochs, we can see
# we have ossiliations, we jump to FID 68 to 70, back to 67 and ... so I guess maybe we instead
# need to lower the lr a bit for both but keep the ratio intact?(if we remove the lr_d too much
# we may make it more powerful like our previous test! so maybe lets continue abit like this
# for some more epochs and then decide?)for now im waiting it til 128x128 ends and see how it
# performs there as well(the lrs are shrunk to 2.5e-05/2.75e-05). the image quality is obviously
# a lot better than before. we are 100% on the right track. the 128x128 also shows very good images
# compared to before which were infested with weird artifacts. we still have artifacts but we
# clearly see some images very well formed and with minial artifacts, we sitll need work to do
# but it shows good progress.we see ossiliations here (128x128) as well, one epoch we are down
# 75 the next epoch it jumps to 90! then down to 87 adn then back to 91 and then 97,105,...
# the lr maybe high, also generator seems to be struggling and needs a push at this step aswell
# 
# update:
# increase the previous 64x64 checkpoint (checkpoint_step_4_20250929131133) training with
# additional 20 epochs, did that, lowered the lr to 0.00004 and 0.000042, but the same ossilications
# exist and the convergence rate became really slow! so Im scrapping all of this and starting
# new
# 
# update:
# started with lr=0.001 just like the paper and see how it goes, we should see something different
# cuase I have had a few bugs back when we initially used this. if this doesnt work, and discrimnator
# dominates the generator (I can use noise like before, but as paper also says, it affects the
# image quality, so using small amounts might not do much and if we use more affects the image
# we can use this with lsgan just like the paper did, but I want to achieve what the paper has achieved
# aswell as close as I can) so it means the generator is weaker, and I will use conv3x3 for both
# this time ramping both up to 23m and recheck for final time. currently we achieve good results
# fid that previously we couldnt, and we can continue improving it with careful lr, but it takes too much time
# and I cant have that! so we are going full beast after this!
# 
# update:(experiment 20250930091608)
# thank God! so far as of epoch 5 of 32x32 (alpha=0.4) we are down to FID 35 which is pretty good!
# the high learning rate that previously kept messing up, after using equalized learnng rate
# seems to be fine and give us a fast convergence. at epoch 19 we are at FID 21! it seems we
# can get better result by decaying the lr after epoch 15! cuz we see ossilications every other
# epochs, we go from 35-27-32-30-34-27-26-28-23 etc)
# ok starting from 64x64 I'm seeing the results are getting worse, each epoch!
# ok they are not, infact the FID fluctuates from epch to epoch, but as more epochs are passed
# images get much detailed and better, especially when we get to alpha=1 (e.g. at epoch 20 we
# our FID is 73, its much more clear and good than its epoch 15 with FID=64!(alpha=0.8)
# the thing is, the loss is decreasing for both discriminator and generator! and that counts!
# the fake_avg becomes positive but hovers around +2/+3 at max but mostly around 0 or negative
# but I guess as long as loss is decreasing its good, which have been our case so far!)
# so its going good.(progan_celeba_wgangp_20250930091608)
# however I just noticed I havent implemented the the ema for generator's weights for 
# inference/generation. this should give us a much lower FID/higher quality 
# (ema always gives better result based on my experience) so after im done 
# in this part, I'll do the ema and see how far we can get our scores!
# im pretty satisfied with the results so far, so Im going to call it a day and
# finish our progan saga here. the image quality is very good the diversity is 
# verygood without ema! (I havent tained with ema) and we did this with 7m! not 23
# (of course we havent gone to 1024x1024 like the original, butalso we dont have 
# the gpus they had! and we know wha to do to get higher res now! its only a mater of
# time and maybe a bit of lr tuning!)
# I endted the training at the start of 128x128 and Im satisfied with t he resulys
# it just took too much . 
# 
# update:
# the reason ema didnt work was because I missed add_ and instead used add! this meant
# we never updated the weights with the new information in the generator and ema_gen
# would get multiplied by 0.999 each time, basically exponentially decaying to zero! 
# and hence why we kept getting solid grays!
#
# the weights and logs are available here. jupyternotebook 11 are the ones having
# training experiments weights and results: 
# url: https://mega.nz/folder/zoRxRSbQ#5cvLQtlRHvnmk7oQo8BTlA
# 
# 
#%%
# full log 1 - the initial log when we started (with bugs in our implemenation)
# ProGAN Training on celeba with loss=wgangp in 20250919075830
# --Discriminator channels:      [1024, 512, 256, 128, 64, 32, 16]
# --Generators channels:         [1024, 512, 256, 128, 64, 32, 16]
# --Dataset:                   celeba
# --Loss type:                 wgangp
# --Discriminator LR:          [0.0001]
# --Generator LR:              [0.0001]
# --Max Step:                  7
# --Epochs:                    [10, 10, 10, 10, 10, 10, 10]
# --Generator update interval: 1
# --WGAN weight cliping range: (-0.02, 0.02)
# --Noise addition to input:   False
# --WGAN-GP Lambda factor:     10
# --gen_num_samples:           64
# --Checkpoint Directory:      ./weights
# --Images Directory:          ./results/gan
# Files already downloaded and verified
# Step: 0/7 -> Training on [4x4]
# --Epochs:                    10
# --BatchSize:                 128
# --Interval:                  637
# --Fade-in Steps:             6360
# --Current Discriminator LR:  [0.0001]
# --Current Generator LR:      [0.0001]
# [4x4][Epoch 0/10 | Iter: 636/1272] Disc Loss: -4.925677 | Gen Loss: 8.639502
# -- Batch-636: Disc's real mean: -3.4278 | Disc's fake mean = -9.0515
# -- Last Batch : Disc's real mean: -3.5601 | Disc's fake mean: -9.2121
# -- Epoch's Avg: Disc's real mean: -3.0610 | Disc's fake mean: -8.4252
# [4x4][Epoch 0/10] Disc Loss-Avg: -4.617619 | Gen loss-Avg: 8.577881 | IS: (μ:1.3896, σ²:0.0953) | FID: 93.27
# [4x4][Epoch 1/10 | Iter: 636/1272] Disc Loss: -4.865841 | Gen Loss: 8.279886
# -- Batch-636: Disc's real mean: -3.3368 | Disc's fake mean = -9.0052
# -- Last Batch : Disc's real mean: -3.1052 | Disc's fake mean: -8.5130
# -- Epoch's Avg: Disc's real mean: -3.1897 | Disc's fake mean: -8.5223
# [4x4][Epoch 1/10] Disc Loss-Avg: -4.781563 | Gen loss-Avg: 8.551735 | IS: (μ:1.4813, σ²:0.0960) | FID: 98.20
# [4x4][Epoch 2/10 | Iter: 636/1272] Disc Loss: -4.744717 | Gen Loss: 8.948160
# -- Batch-636: Disc's real mean: -2.8070 | Disc's fake mean = -7.7893
# -- Last Batch : Disc's real mean: -3.2985 | Disc's fake mean: -8.8349
# -- Epoch's Avg: Disc's real mean: -3.0226 | Disc's fake mean: -8.4156
# [4x4][Epoch 2/10] Disc Loss-Avg: -4.842244 | Gen loss-Avg: 8.434741 | IS: (μ:1.4694, σ²:0.1286) | FID: 103.95
# [4x4][Epoch 3/10 | Iter: 636/1272] Disc Loss: -4.845220 | Gen Loss: 8.880974
# -- Batch-636: Disc's real mean: -2.8261 | Disc's fake mean = -7.9490
# -- Last Batch : Disc's real mean: -3.1340 | Disc's fake mean: -8.3969
# -- Epoch's Avg: Disc's real mean: -2.9363 | Disc's fake mean: -8.3838
# [4x4][Epoch 3/10] Disc Loss-Avg: -4.889210 | Gen loss-Avg: 8.399119 | IS: (μ:1.4381, σ²:0.1449) | FID: 102.72
# [4x4][Epoch 4/10 | Iter: 636/1272] Disc Loss: -5.092498 | Gen Loss: 8.336432
# -- Batch-636: Disc's real mean: -2.8242 | Disc's fake mean = -8.6238
# -- Last Batch : Disc's real mean: -3.0536 | Disc's fake mean: -8.7346
# -- Epoch's Avg: Disc's real mean: -2.8819 | Disc's fake mean: -8.3612
# [4x4][Epoch 4/10] Disc Loss-Avg: -4.921630 | Gen loss-Avg: 8.374806 | IS: (μ:1.3715, σ²:0.1099) | FID: 100.50
# [4x4][Epoch 5/10 | Iter: 636/1272] Disc Loss: -4.762464 | Gen Loss: 7.676002
# -- Batch-636: Disc's real mean: -3.1710 | Disc's fake mean = -8.7792
# -- Last Batch : Disc's real mean: -2.7949 | Disc's fake mean: -8.2154
# -- Epoch's Avg: Disc's real mean: -2.8177 | Disc's fake mean: -8.3175
# [4x4][Epoch 5/10] Disc Loss-Avg: -4.938248 | Gen loss-Avg: 8.329852 | IS: (μ:1.4978, σ²:0.2215) | FID: 98.05
# [4x4][Epoch 6/10 | Iter: 636/1272] Disc Loss: -5.119781 | Gen Loss: 8.698465
# -- Batch-636: Disc's real mean: -2.4154 | Disc's fake mean = -7.9340
# -- Last Batch : Disc's real mean: -2.8108 | Disc's fake mean: -8.5038
# -- Epoch's Avg: Disc's real mean: -2.7774 | Disc's fake mean: -8.2924
# [4x4][Epoch 6/10] Disc Loss-Avg: -4.952495 | Gen loss-Avg: 8.301826 | IS: (μ:1.4222, σ²:0.0776) | FID: 96.33
# [4x4][Epoch 7/10 | Iter: 636/1272] Disc Loss: -4.991845 | Gen Loss: 7.971615
# -- Batch-636: Disc's real mean: -2.9745 | Disc's fake mean = -8.8734
# -- Last Batch : Disc's real mean: -2.6837 | Disc's fake mean: -8.0777
# -- Epoch's Avg: Disc's real mean: -2.7869 | Disc's fake mean: -8.3084
# [4x4][Epoch 7/10] Disc Loss-Avg: -4.956729 | Gen loss-Avg: 8.317844 | IS: (μ:1.4367, σ²:0.1053) | FID: 105.24
# [4x4][Epoch 8/10 | Iter: 636/1272] Disc Loss: -5.075427 | Gen Loss: 8.320007
# -- Batch-636: Disc's real mean: -2.7875 | Disc's fake mean = -8.4707
# -- Last Batch : Disc's real mean: -2.9384 | Disc's fake mean: -8.6286
# -- Epoch's Avg: Disc's real mean: -2.7941 | Disc's fake mean: -8.3234
# [4x4][Epoch 8/10] Disc Loss-Avg: -4.964680 | Gen loss-Avg: 8.330603 | IS: (μ:1.4715, σ²:0.1896) | FID: 107.10
# [4x4][Epoch 9/10 | Iter: 636/1272] Disc Loss: -4.868393 | Gen Loss: 8.235387
# -- Batch-636: Disc's real mean: -2.8559 | Disc's fake mean = -8.2226
# -- Last Batch : Disc's real mean: -2.8510 | Disc's fake mean: -8.4673
# -- Epoch's Avg: Disc's real mean: -2.7859 | Disc's fake mean: -8.3241
# [4x4][Epoch 9/10] Disc Loss-Avg: -4.971841 | Gen loss-Avg: 8.331405 | IS: (μ:1.4361, σ²:0.1627) | FID: 103.42
# Files already downloaded and verified
# Step: 1/7 -> Training on [8x8]
# --Epochs:                    10
# --BatchSize:                 128
# --Interval:                  637
# --Fade-in Steps:             6360
# --Current Discriminator LR:  [0.0001]
# --Current Generator LR:      [0.0001]
# [8x8][Epoch 0/10 | Iter: 636/1272] Disc Loss: -3.942824 | Gen Loss: 4.006044
# -- Batch-636: Disc's real mean: -3.3105 | Disc's fake mean = -8.3451
# -- Last Batch : Disc's real mean: -1.1180 | Disc's fake mean: -3.5323
# -- Epoch's Avg: Disc's real mean: -2.5742 | Disc's fake mean: -6.8915
# [8x8][Epoch 0/10] Disc Loss-Avg: -3.793555 | Gen loss-Avg: 6.953594 | IS: (μ:1.5331, σ²:0.1087) | FID: 98.50
# [8x8][Epoch 1/10 | Iter: 636/1272] Disc Loss: -0.976770 | Gen Loss: 4.483275
# -- Batch-636: Disc's real mean: -3.3820 | Disc's fake mean = -4.4443
# -- Last Batch : Disc's real mean: -1.4105 | Disc's fake mean: -2.1686
# -- Epoch's Avg: Disc's real mean: -2.7687 | Disc's fake mean: -4.1819
# [8x8][Epoch 1/10] Disc Loss-Avg: -1.237222 | Gen loss-Avg: 4.318108 | IS: (μ:1.6119, σ²:0.1315) | FID: 78.51
# [8x8][Epoch 2/10 | Iter: 636/1272] Disc Loss: -0.713874 | Gen Loss: 3.783400
# -- Batch-636: Disc's real mean: -2.1031 | Disc's fake mean = -3.0995
# -- Last Batch : Disc's real mean: -2.5537 | Disc's fake mean: -3.6629
# -- Epoch's Avg: Disc's real mean: -2.7358 | Disc's fake mean: -3.6460
# [8x8][Epoch 2/10] Disc Loss-Avg: -0.715368 | Gen loss-Avg: 3.809226 | IS: (μ:1.7053, σ²:0.1853) | FID: 90.17
# [8x8][Epoch 3/10 | Iter: 636/1272] Disc Loss: -2.171912 | Gen Loss: 5.163008
# -- Batch-636: Disc's real mean: -2.6180 | Disc's fake mean = -4.9433
# -- Last Batch : Disc's real mean: -7.7876 | Disc's fake mean: -12.5876
# -- Epoch's Avg: Disc's real mean: -4.8014 | Disc's fake mean: -7.1733
# [8x8][Epoch 3/10] Disc Loss-Avg: -2.053780 | Gen loss-Avg: 7.317423 | IS: (μ:2.2320, σ²:0.2663) | FID: 126.10
# [8x8][Epoch 4/10 | Iter: 636/1272] Disc Loss: -6.651464 | Gen Loss: 14.208398
# -- Batch-636: Disc's real mean: -10.7196 | Disc's fake mean = -19.4104
# -- Last Batch : Disc's real mean: -8.1940 | Disc's fake mean: -19.2125
# -- Epoch's Avg: Disc's real mean: -8.8186 | Disc's fake mean: -16.8516
# [8x8][Epoch 4/10] Disc Loss-Avg: -6.892580 | Gen loss-Avg: 16.944392 | IS: (μ:2.2503, σ²:0.1996) | FID: 142.88
# [8x8][Epoch 5/10 | Iter: 636/1272] Disc Loss: -9.578485 | Gen Loss: 23.625065
# -- Batch-636: Disc's real mean: -7.0493 | Disc's fake mean = -17.3150
# -- Last Batch : Disc's real mean: -9.1310 | Disc's fake mean: -22.0471
# -- Epoch's Avg: Disc's real mean: -8.6074 | Disc's fake mean: -20.4968
# [8x8][Epoch 5/10] Disc Loss-Avg: -9.963724 | Gen loss-Avg: 20.564052 | IS: (μ:2.4033, σ²:0.4934) | FID: 138.27
# [8x8][Epoch 6/10 | Iter: 636/1272] Disc Loss: -10.238967 | Gen Loss: 18.904112
# -- Batch-636: Disc's real mean: -8.4896 | Disc's fake mean = -21.1637
# -- Last Batch : Disc's real mean: -8.5351 | Disc's fake mean: -21.2374
# -- Epoch's Avg: Disc's real mean: -8.0157 | Disc's fake mean: -20.0544
# [8x8][Epoch 6/10] Disc Loss-Avg: -10.078539 | Gen loss-Avg: 20.116750 | IS: (μ:2.4288, σ²:0.3593) | FID: 152.85
# [8x8][Epoch 7/10 | Iter: 636/1272] Disc Loss: -10.101956 | Gen Loss: 16.769264
# -- Batch-636: Disc's real mean: -9.0578 | Disc's fake mean = -22.5658
# -- Last Batch : Disc's real mean: -7.1267 | Disc's fake mean: -18.1838
# -- Epoch's Avg: Disc's real mean: -7.8162 | Disc's fake mean: -19.9497
# [8x8][Epoch 7/10] Disc Loss-Avg: -10.146060 | Gen loss-Avg: 20.010850 | IS: (μ:2.1634, σ²:0.2896) | FID: 147.63
# [8x8][Epoch 8/10 | Iter: 636/1272] Disc Loss: -10.350794 | Gen Loss: 21.058495
# -- Batch-636: Disc's real mean: -7.0868 | Disc's fake mean = -18.8944
# -- Last Batch : Disc's real mean: -7.9366 | Disc's fake mean: -20.4258
# -- Epoch's Avg: Disc's real mean: -7.7232 | Disc's fake mean: -19.9228
# [8x8][Epoch 8/10] Disc Loss-Avg: -10.197758 | Gen loss-Avg: 19.979316 | IS: (μ:2.2553, σ²:0.3744) | FID: 139.80
# [8x8][Epoch 9/10 | Iter: 636/1272] Disc Loss: -10.080774 | Gen Loss: 17.941248
# -- Batch-636: Disc's real mean: -8.1024 | Disc's fake mean = -20.9515
# -- Last Batch : Disc's real mean: -6.8041 | Disc's fake mean: -18.5868
# -- Epoch's Avg: Disc's real mean: -7.6544 | Disc's fake mean: -19.8999
# [8x8][Epoch 9/10] Disc Loss-Avg: -10.232327 | Gen loss-Avg: 19.956750 | IS: (μ:2.2126, σ²:0.3614) | FID: 136.53
# Files already downloaded and verified
# Step: 2/7 -> Training on [16x16]
# --Epochs:                    10
# --BatchSize:                 128
# --Interval:                  637
# --Fade-in Steps:             6360
# --Current Discriminator LR:  [0.0001]
# --Current Generator LR:      [0.0001]
# [16x16][Epoch 0/10 | Iter: 636/1272] Disc Loss: -7.659537 | Gen Loss: 14.488880
# -- Batch-636: Disc's real mean: -13.5219 | Disc's fake mean = -23.4782
# -- Last Batch : Disc's real mean: -6.6747 | Disc's fake mean: -10.8735
# -- Epoch's Avg: Disc's real mean: -9.6144 | Disc's fake mean: -19.0800
# [16x16][Epoch 0/10] Disc Loss-Avg: -7.970980 | Gen loss-Avg: 19.321590 | IS: (μ:1.6438, σ²:0.2244) | FID: 197.63
# [16x16][Epoch 1/10 | Iter: 636/1272] Disc Loss: -2.020536 | Gen Loss: 6.696161
# -- Batch-636: Disc's real mean: -6.0554 | Disc's fake mean = -8.2295
# -- Last Batch : Disc's real mean: -4.9554 | Disc's fake mean: -6.2148
# -- Epoch's Avg: Disc's real mean: -4.0256 | Disc's fake mean: -6.3547
# [16x16][Epoch 1/10] Disc Loss-Avg: -2.111051 | Gen loss-Avg: 6.702969 | IS: (μ:1.8410, σ²:0.2926) | FID: 128.10
# [16x16][Epoch 2/10 | Iter: 636/1272] Disc Loss: -1.739925 | Gen Loss: 3.560276
# -- Batch-636: Disc's real mean: -4.9438 | Disc's fake mean = -6.8329
# -- Last Batch : Disc's real mean: -6.5064 | Disc's fake mean: -8.5992
# -- Epoch's Avg: Disc's real mean: -3.7870 | Disc's fake mean: -5.6142
# [16x16][Epoch 2/10] Disc Loss-Avg: -1.595649 | Gen loss-Avg: 6.049932 | IS: (μ:2.0744, σ²:0.2274) | FID: 144.43
# [16x16][Epoch 3/10 | Iter: 636/1272] Disc Loss: -3.705430 | Gen Loss: 13.437008
# -- Batch-636: Disc's real mean: -8.5909 | Disc's fake mean = -13.2413
# -- Last Batch : Disc's real mean: -14.0865 | Disc's fake mean: -22.1837
# -- Epoch's Avg: Disc's real mean: -10.8148 | Disc's fake mean: -15.5159
# [16x16][Epoch 3/10] Disc Loss-Avg: -4.039606 | Gen loss-Avg: 15.904802 | IS: (μ:2.3125, σ²:0.4237) | FID: 180.10
# [16x16][Epoch 4/10 | Iter: 636/1272] Disc Loss: -14.294024 | Gen Loss: 35.716000
# -- Batch-636: Disc's real mean: -23.3948 | Disc's fake mean = -43.2292
# -- Last Batch : Disc's real mean: -18.3042 | Disc's fake mean: -40.5495
# -- Epoch's Avg: Disc's real mean: -21.7162 | Disc's fake mean: -39.6128
# [16x16][Epoch 4/10] Disc Loss-Avg: -14.260592 | Gen loss-Avg: 39.969417 | IS: (μ:2.0524, σ²:0.2302) | FID: 199.34
# [16x16][Epoch 5/10 | Iter: 636/1272] Disc Loss: -21.822086 | Gen Loss: 51.630539
# -- Batch-636: Disc's real mean: -22.9608 | Disc's fake mean = -51.1459
# -- Last Batch : Disc's real mean: -27.5397 | Disc's fake mean: -60.3107
# -- Epoch's Avg: Disc's real mean: -23.1056 | Disc's fake mean: -51.1131
# [16x16][Epoch 5/10] Disc Loss-Avg: -21.496181 | Gen loss-Avg: 51.352162 | IS: (μ:2.3834, σ²:0.3537) | FID: 210.10
# [16x16][Epoch 6/10 | Iter: 636/1272] Disc Loss: -22.095404 | Gen Loss: 54.631283
# -- Batch-636: Disc's real mean: -21.4014 | Disc's fake mean = -48.5897
# -- Last Batch : Disc's real mean: -25.7336 | Disc's fake mean: -56.9287
# -- Epoch's Avg: Disc's real mean: -22.5394 | Disc's fake mean: -50.8309
# [16x16][Epoch 6/10] Disc Loss-Avg: -21.714162 | Gen loss-Avg: 51.049461 | IS: (μ:2.4055, σ²:0.2742) | FID: 219.43
# [16x16][Epoch 7/10 | Iter: 636/1272] Disc Loss: -22.313515 | Gen Loss: 47.001690
# -- Batch-636: Disc's real mean: -25.0881 | Disc's fake mean = -55.9551
# -- Last Batch : Disc's real mean: -25.4260 | Disc's fake mean: -57.7063
# -- Epoch's Avg: Disc's real mean: -22.7652 | Disc's fake mean: -51.3375
# [16x16][Epoch 7/10] Disc Loss-Avg: -21.930832 | Gen loss-Avg: 51.536230 | IS: (μ:2.6507, σ²:0.3991) | FID: 209.44
# [16x16][Epoch 8/10 | Iter: 636/1272] Disc Loss: -21.818005 | Gen Loss: 54.625275
# -- Batch-636: Disc's real mean: -22.1112 | Disc's fake mean = -49.2159
# -- Last Batch : Disc's real mean: -24.8057 | Disc's fake mean: -53.4536
# -- Epoch's Avg: Disc's real mean: -23.0335 | Disc's fake mean: -51.7901
# [16x16][Epoch 8/10] Disc Loss-Avg: -22.060185 | Gen loss-Avg: 51.988842 | IS: (μ:2.3966, σ²:0.3689) | FID: 203.22
# [16x16][Epoch 9/10 | Iter: 636/1272] Disc Loss: -21.474421 | Gen Loss: 42.068031
# -- Batch-636: Disc's real mean: -27.8928 | Disc's fake mean = -59.5297
# -- Last Batch : Disc's real mean: -21.0027 | Disc's fake mean: -47.1649
# -- Epoch's Avg: Disc's real mean: -23.6127 | Disc's fake mean: -52.5325
# [16x16][Epoch 9/10] Disc Loss-Avg: -22.175056 | Gen loss-Avg: 52.723336 | IS: (μ:2.3401, σ²:0.4043) | FID: 205.07
# Files already downloaded and verified
# Step: 3/7 -> Training on [32x32]
# --Epochs:                    10
# --BatchSize:                 128
# --Interval:                  637
# --Fade-in Steps:             6360
# --Current Discriminator LR:  [0.0001]
# --Current Generator LR:      [0.0001]
# [32x32][Epoch 0/10 | Iter: 636/1272] Disc Loss: -9.742870 | Gen Loss: 22.850704
# -- Batch-636: Disc's real mean: -31.4601 | Disc's fake mean = -45.6997
# -- Last Batch : Disc's real mean: -9.6557 | Disc's fake mean: -15.0042
# -- Epoch's Avg: Disc's real mean: -24.8806 | Disc's fake mean: -41.3923
# [32x32][Epoch 0/10] Disc Loss-Avg: -12.990907 | Gen loss-Avg: 41.992856 | IS: (μ:2.1172, σ²:0.2746) | FID: 292.92
# [32x32][Epoch 1/10 | Iter: 636/1272] Disc Loss: -3.576055 | Gen Loss: 9.675308
# -- Batch-636: Disc's real mean: -9.8570 | Disc's fake mean = -13.9727
# -- Last Batch : Disc's real mean: -2.8534 | Disc's fake mean: -5.3507
# -- Epoch's Avg: Disc's real mean: -6.6131 | Disc's fake mean: -10.1418
# [32x32][Epoch 1/10] Disc Loss-Avg: -3.124855 | Gen loss-Avg: 10.824034 | IS: (μ:2.1128, σ²:0.2201) | FID: 140.89
# 
# [32x32][Epoch 2/10 | Iter: 636/1272] Disc Loss: -3.419496 | Gen Loss: 4.430109
# -- Batch-636: Disc's real mean: -8.3010 | Disc's fake mean = -12.1431
# -- Last Batch : Disc's real mean: -4.7530 | Disc's fake mean: -7.6684
# -- Epoch's Avg: Disc's real mean: -4.3094 | Disc's fake mean: -7.4291
# [32x32][Epoch 2/10] Disc Loss-Avg: -2.788034 | Gen loss-Avg: 8.243693 | IS: (μ:2.1323, σ²:0.2053) | FID: 150.22
# 
# [32x32][Epoch 3/10 | Iter: 636/1272] Disc Loss: -6.667298 | Gen Loss: 22.475258
# -- Batch-636: Disc's real mean: -6.3903 | Disc's fake mean = -13.2792
# -- Last Batch : Disc's real mean: -24.2167 | Disc's fake mean: -43.2305
# -- Epoch's Avg: Disc's real mean: -12.7801 | Disc's fake mean: -22.0853
# [32x32][Epoch 3/10] Disc Loss-Avg: -7.668428 | Gen loss-Avg: 22.880436 | IS: (μ:2.1997, σ²:0.2628) | FID: 172.00
# 
# [32x32][Epoch 4/10 | Iter: 636/1272] Disc Loss: -29.124699 | Gen Loss: 96.169762
# -- Batch-636: Disc's real mean: -49.6919 | Disc's fake mean = -87.8773
# -- Last Batch : Disc's real mean: -82.3294 | Disc's fake mean: -164.1101
# -- Epoch's Avg: Disc's real mean: -52.1123 | Disc's fake mean: -96.0587
# [32x32][Epoch 4/10] Disc Loss-Avg: -31.948336 | Gen loss-Avg: 97.059814 | IS: (μ:2.3290, σ²:0.2181) | FID: 196.82
# 
# [32x32][Epoch 5/10 | Iter: 636/1272] Disc Loss: -53.967762 | Gen Loss: 129.994873
# -- Batch-636: Disc's real mean: -86.0678 | Disc's fake mean = -172.2634
# -- Last Batch : Disc's real mean: -88.3117 | Disc's fake mean: -171.4049
# -- Epoch's Avg: Disc's real mean: -76.8118 | Disc's fake mean: -150.8211
# [32x32][Epoch 5/10] Disc Loss-Avg: -51.307398 | Gen loss-Avg: 151.665356 | IS: (μ:2.2485, σ²:0.3168) | FID: 200.52
# 
# [32x32][Epoch 6/10 | Iter: 636/1272] Disc Loss: -52.603218 | Gen Loss: 148.264023
# -- Batch-636: Disc's real mean: -81.8048 | Disc's fake mean = -160.9394
# -- Last Batch : Disc's real mean: -72.4470 | Disc's fake mean: -140.2893
# -- Epoch's Avg: Disc's real mean: -78.9150 | Disc's fake mean: -153.0085
# [32x32][Epoch 6/10] Disc Loss-Avg: -51.385977 | Gen loss-Avg: 153.785579 | IS: (μ:2.3602, σ²:0.2871) | FID: 202.67
# 
# [32x32][Epoch 7/10 | Iter: 636/1272] Disc Loss: -53.027206 | Gen Loss: 157.106827
# -- Batch-636: Disc's real mean: -91.4071 | Disc's fake mean = -170.6373
# -- Last Batch : Disc's real mean: -63.2842 | Disc's fake mean: -121.8838
# -- Epoch's Avg: Disc's real mean: -84.0880 | Disc's fake mean: -158.4912
# [32x32][Epoch 7/10] Disc Loss-Avg: -51.570314 | Gen loss-Avg: 159.210803 | IS: (μ:2.3455, σ²:0.2675) | FID: 202.04
# 
# [32x32][Epoch 8/10 | Iter: 636/1272] Disc Loss: -50.969788 | Gen Loss: 184.474548
# -- Batch-636: Disc's real mean: -79.6008 | Disc's fake mean = -146.3249
# -- Last Batch : Disc's real mean: -93.2223 | Disc's fake mean: -171.6383
# -- Epoch's Avg: Disc's real mean: -88.7116 | Disc's fake mean: -163.3899
# [32x32][Epoch 8/10] Disc Loss-Avg: -51.760759 | Gen loss-Avg: 164.070204 | IS: (μ:2.3821, σ²:0.4678) | FID: 221.44
# 
# [32x32][Epoch 9/10 | Iter: 636/1272] Disc Loss: -53.924160 | Gen Loss: 168.099167
# -- Batch-636: Disc's real mean: -93.9407 | Disc's fake mean = -173.9554
# -- Last Batch : Disc's real mean: -86.6145 | Disc's fake mean: -158.3096
# -- Epoch's Avg: Disc's real mean: -91.5164 | Disc's fake mean: -166.5711
# [32x32][Epoch 9/10] Disc Loss-Avg: -52.006154 | Gen loss-Avg: 167.229953 | IS: (μ:2.3043, σ²:0.1846) | FID: 212.40
# 
# Files already downloaded and verified
# Step: 4/7 -> Training on [64x64]
# --Epochs:                    10
# --BatchSize:                 64
# --Interval:                  1273
# --Fade-in Steps:             12720
# --Current Discriminator LR:  [0.0001]
# --Current Generator LR:      [0.0001]
# [64x64][Epoch 0/10 | Iter: 1272/2544] Disc Loss: -14.984601 | Gen Loss: 51.437641
# -- Batch-1272: Disc's real mean: -17.3619 | Disc's fake mean = -33.0078
# -- Last Batch : Disc's real mean: -28.3425 | Disc's fake mean: -32.9185
# -- Epoch's Avg: Disc's real mean: -58.1945 | Disc's fake mean: -93.2741
# [64x64][Epoch 0/10] Disc Loss-Avg: -25.013562 | Gen loss-Avg: 94.841881 | IS: (μ:1.0000, σ²:0.0000) | FID: 333.02
# [64x64][Epoch 1/10 | Iter: 1272/2544] Disc Loss: -5.097756 | Gen Loss: 0.344297
# -- Batch-1272: Disc's real mean: 7.5136 | Disc's fake mean = 2.2513
# -- Last Batch : Disc's real mean: -4.6169 | Disc's fake mean: -6.2919
# -- Epoch's Avg: Disc's real mean: 3.9837 | Disc's fake mean: -1.3474
# [64x64][Epoch 1/10] Disc Loss-Avg: -4.625465 | Gen loss-Avg: 2.729605 | IS: (μ:1.0000, σ²:0.0000) | FID: 241.13
# [64x64][Epoch 2/10 | Iter: 1272/2544] Disc Loss: -4.328329 | Gen Loss: 22.177746
# -- Batch-1272: Disc's real mean: -31.3004 | Disc's fake mean = -36.4820
# -- Last Batch : Disc's real mean: -24.6972 | Disc's fake mean: -30.4969
# -- Epoch's Avg: Disc's real mean: -10.4536 | Disc's fake mean: -15.6216
# [64x64][Epoch 2/10] Disc Loss-Avg: -4.539772 | Gen loss-Avg: 16.904588 | IS: (μ:1.0000, σ²:0.0000) | FID: 222.88
# [64x64][Epoch 3/10 | Iter: 1272/2544] Disc Loss: -14.829409 | Gen Loss: 44.549088
# -- Batch-1272: Disc's real mean: -31.8848 | Disc's fake mean = -50.9633
# -- Last Batch : Disc's real mean: -100.3516 | Disc's fake mean: -153.9123
# -- Epoch's Avg: Disc's real mean: -36.4500 | Disc's fake mean: -57.4129
# [64x64][Epoch 3/10] Disc Loss-Avg: -16.055016 | Gen loss-Avg: 58.809311 | IS: (μ:1.0000, σ²:0.0000) | FID: 257.49
# [64x64][Epoch 4/10 | Iter: 1272/2544] Disc Loss: -82.065460 | Gen Loss: 286.170471
# -- Batch-1272: Disc's real mean: -244.0014 | Disc's fake mean = -383.1243

#
#
#
#
# now our final log after fixing all bugs 
# and issues (look at the scale of loss/means)
# note that the smilies for fake_mean is not accurate (I set the low threshold very harshly
# and if you look you'll see its around 0, and when it gets to +1~+2 which is still okish
# the std shows massive variance. if the variance was low that would be an issue but high
# variance means the generator is comming up with very diverse images and is trying well)
# 
# Training PROGAN!
# channels=[512, 256, 128, 64, 32, 16, 8]
# channels=[512, 256, 128, 64, 32, 16, 8]
# ProGAN Training on celeba with loss=wgangp in 20250930091608
# --Disc Param Count:          7,092,584
# --Genr Param Count:          7,092,584
# --Dataset:                   celeba-train
# --Loss type:                 wgangp
# --Discriminator LR:          0.001
# --Generator LR:              0.001
# --Max Step:                  7
# --Decay Step:                7
# --Epochs:                    [10, 10, 10, 30, 40, 40, 40] 
# --Batch-Sizes:               [128, 128, 128, 128, 64, 32, 16] 
# --Generator update interval: 1
# --WGAN weight cliping range: (-0.02, 0.02)
# --Noise addition to input:   False
# --WGAN-GP Lambda factor:     5
# --gen_num_samples:           64
# --Checkpoint Directory:      ./weights/gan
# --Images Directory:          ./results/gan
# Files already downloaded and verified
#  Step: 0/7 -> Training on [4x4]
#   --Epochs:                      10 
#   --BatchSize:                   128 
#   --Number of Batches:           1272 
#   --Interval:                    637 
#   --Fade-in Steps:               6360 
#   --Last training Step taken:    0 
#   --Current Discriminator LR:    0.001
#   --Current Generator LR:        0.001
#   --Current Discriminator Betas: [0, 0.99]
#   --Current Generator Betas:     [0, 0.99]
# [4x4][Epoch 0/10 | Iter: 636/1272] Disc Loss: -0.2637 | Gen Loss: 0.7397
#  -- 🫤 Batch-636:  D_real_avg: 😵 -0.3313 ± +0.3181 📈| D_fake_avg: 😎 -0.6711 ± +0.2464 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 🫤 Last Batch : D_real_avg: 😵 +0.0063 ± +0.2137 📈 | D_fake_avg: 😎 -0.1435 ± +0.1836 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😵 -0.1635 ± -0.1635 📈 | D_fake_avg: 😎 -0.8983 ± +1.0369 📉
# [4x4][Epoch 0/10] DLoss(Avg): -0.6027 | GLoss(Avg): 1.0167 | IS: 1.1468 ± 0.0046) | FID: 326.88 | GP[avg]: 0.03
# [4x4][Epoch 1/10 | Iter: 636/1272] Disc Loss: -0.1480 | Gen Loss: 0.2829
#  -- 🫤 Batch-636:  D_real_avg: 😵 +0.0329 ± +0.2893 📈| D_fake_avg: 😎 -0.1553 ± +0.2336 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😱 Last Batch : D_real_avg: 😵 +0.1147 ± +0.2916 📈 | D_fake_avg: 😟 +0.0323 ± +0.3174 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😵 +0.0517 ± +0.0517 📈 | D_fake_avg: 😎 -0.1023 ± +0.1598 📉
# [4x4][Epoch 1/10] DLoss(Avg): -0.1033 | GLoss(Avg): 0.1412 | IS: 1.1617 ± 0.0054) | FID: 321.46 | GP[avg]: 0.01
# [4x4][Epoch 2/10 | Iter: 636/1272] Disc Loss: -0.0992 | Gen Loss: 0.0070
#  -- 😱 Batch-636:  D_real_avg: 😵 +0.1551 ± +0.3857 📈| D_fake_avg: 😟 +0.0035 ± +0.3081 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 🫤 Last Batch : D_real_avg: 😵 -0.1071 ± +0.4185 📈 | D_fake_avg: 😎 -0.1707 ± +0.2789 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😵 +0.1437 ± +0.1437 📈 | D_fake_avg: 😟 +0.0131 ± +0.1593 📉
# [4x4][Epoch 2/10] DLoss(Avg): -0.0796 | GLoss(Avg): 0.0204 | IS: 1.1619 ± 0.0042) | FID: 322.94 | GP[avg]: 0.01
# [4x4][Epoch 3/10 | Iter: 636/1272] Disc Loss: -0.0961 | Gen Loss: -0.0755
#  -- 😱 Batch-636:  D_real_avg: 😰 +0.2300 ± +0.2761 📈| D_fake_avg: 😟 +0.0904 ± +0.2167 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 🫤 Last Batch : D_real_avg: 😵 -0.1629 ± +0.4184 📈 | D_fake_avg: 😎 -0.2334 ± +0.3547 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😵 +0.1419 ± +0.1419 📈 | D_fake_avg: 😟 +0.0164 ± +0.1306 📉
# [4x4][Epoch 3/10] DLoss(Avg): -0.0769 | GLoss(Avg): 0.0142 | IS: 1.1615 ± 0.0036) | FID: 321.73 | GP[avg]: 0.01
# [4x4][Epoch 4/10 | Iter: 636/1272] Disc Loss: -0.0468 | Gen Loss: -0.0218
#  -- 🫤 Batch-636:  D_real_avg: 😵 +0.0095 ± +0.3608 📈| D_fake_avg: 😎 -0.0866 ± +0.2892 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 🫤 Last Batch : D_real_avg: 😵 +0.0231 ± +0.3782 📈 | D_fake_avg: 😎 -0.0752 ± +0.3137 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😵 +0.1369 ± +0.1369 📈 | D_fake_avg: 😟 +0.0122 ± +0.1381 📉
# [4x4][Epoch 4/10] DLoss(Avg): -0.0763 | GLoss(Avg): 0.0135 | IS: 1.1624 ± 0.0041) | FID: 319.47 | GP[avg]: 0.01
# [4x4][Epoch 5/10 | Iter: 636/1272] Disc Loss: -0.0832 | Gen Loss: -0.0306
#  -- 😱 Batch-636:  D_real_avg: 😰 +0.3013 ± +0.3035 📈| D_fake_avg: 😟 +0.1657 ± +0.2733 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 🫤 Last Batch : D_real_avg: 😵 +0.0479 ± +0.2912 📈 | D_fake_avg: 😎 -0.0306 ± +0.2595 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😵 +0.1397 ± +0.1397 📈 | D_fake_avg: 😟 +0.0183 ± +0.1539 📉
# [4x4][Epoch 5/10] DLoss(Avg): -0.0742 | GLoss(Avg): 0.0061 | IS: 1.1632 ± 0.0043) | FID: 321.61 | GP[avg]: 0.01
# [4x4][Epoch 6/10 | Iter: 636/1272] Disc Loss: -0.1235 | Gen Loss: -0.1953
#  -- 😱 Batch-636:  D_real_avg: 😰 +0.2288 ± +0.3014 📈| D_fake_avg: 😟 +0.0584 ± +0.2759 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😱 Last Batch : D_real_avg: 😵 +0.1004 ± +0.2650 📈 | D_fake_avg: 😟 +0.0291 ± +0.2185 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😵 +0.1174 ± +0.1174 📈 | D_fake_avg: 😎 -0.0005 ± +0.1574 📉
# [4x4][Epoch 6/10] DLoss(Avg): -0.0710 | GLoss(Avg): 0.0214 | IS: 1.1616 ± 0.0041) | FID: 320.06 | GP[avg]: 0.01
# [4x4][Epoch 7/10 | Iter: 636/1272] Disc Loss: -0.0021 | Gen Loss: 0.0462
#  -- 🫤 Batch-636:  D_real_avg: 😵 +0.0418 ± +0.3210 📈| D_fake_avg: 😎 -0.0224 ± +0.2712 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😱 Last Batch : D_real_avg: 😰 +0.2154 ± +0.3457 📈 | D_fake_avg: 😟 +0.0501 ± +0.2006 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😵 +0.1012 ± +0.1012 📈 | D_fake_avg: 😎 -0.0146 ± +0.1304 📉
# [4x4][Epoch 7/10] DLoss(Avg): -0.0686 | GLoss(Avg): 0.0375 | IS: 1.1653 ± 0.0040) | FID: 319.07 | GP[avg]: 0.01
# [4x4][Epoch 8/10 | Iter: 636/1272] Disc Loss: -0.0110 | Gen Loss: 0.1256
#  -- 🫤 Batch-636:  D_real_avg: 😵 -0.0470 ± +0.4295 📈| D_fake_avg: 😎 -0.1039 ± +0.3213 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 🫤 Last Batch : D_real_avg: 😵 +0.0521 ± +0.3049 📈 | D_fake_avg: 😎 -0.0215 ± +0.3186 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😵 +0.0761 ± +0.0761 📈 | D_fake_avg: 😎 -0.0372 ± +0.1469 📉
# [4x4][Epoch 8/10] DLoss(Avg): -0.0658 | GLoss(Avg): 0.0584 | IS: 1.1587 ± 0.0027) | FID: 324.20 | GP[avg]: 0.01
# [4x4][Epoch 9/10 | Iter: 636/1272] Disc Loss: -0.0445 | Gen Loss: -0.0962
#  -- 😱 Batch-636:  D_real_avg: 😵 +0.1792 ± +0.2247 📈| D_fake_avg: 😟 +0.0989 ± +0.2071 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 🫤 Last Batch : D_real_avg: 😵 +0.1321 ± +0.3128 📈 | D_fake_avg: 😎 -0.0009 ± +0.3535 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😵 +0.0665 ± +0.0665 📈 | D_fake_avg: 😎 -0.0415 ± +0.1381 📉
# [4x4][Epoch 9/10] DLoss(Avg): -0.0609 | GLoss(Avg): 0.0626 | IS: 1.1677 ± 0.0030) | FID: 321.05 | GP[avg]: 0.01
# Files already downloaded and verified
#  Step: 1/7 -> Training on [8x8]
#   --Epochs:                      10 
#   --BatchSize:                   128 
#   --Number of Batches:           1272 
#   --Interval:                    637 
#   --Fade-in Steps:               6360 
#   --Last training Step taken:    0 
#   --Current Discriminator LR:    0.001
#   --Current Generator LR:        0.001
#   --Current Discriminator Betas: [0, 0.99]
#   --Current Generator Betas:     [0, 0.99]
# [8x8][Epoch 0/10 | Iter: 636/1272] Disc Loss: -1.4576 | Gen Loss: -0.3277
#  -- 😀 Batch-636:  D_real_avg: 😎 +1.2772 ± +1.0212 📈| D_fake_avg: 😎 -0.3468 ± +0.5478 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😱 Last Batch : D_real_avg: 😎 +1.7041 ± +0.7461 📈 | D_fake_avg: 😵 +1.1932 ± +0.7825 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +1.7663 ± +1.7663 📈 | D_fake_avg: 😵 +0.5964 ± +0.5807 📉
# [8x8][Epoch 0/10] DLoss(Avg): -1.0227 | GLoss(Avg): -0.5101 | IS: 1.7498 ± 0.0275) | FID: 252.08 | GP[avg]: 0.03
# [8x8][Epoch 1/10 | Iter: 636/1272] Disc Loss: -0.2365 | Gen Loss: 0.6164
#  -- 😱 Batch-636:  D_real_avg: 😎 +1.1550 ± +0.5990 📈| D_fake_avg: 😵 +0.8748 ± +0.3766 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😱 Last Batch : D_real_avg: 😰 +0.3898 ± +0.4860 📈 | D_fake_avg: 😟 +0.0723 ± +0.5324 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +0.8224 ± +0.8224 📈 | D_fake_avg: 😵 +0.4308 ± +0.6690 📉
# [8x8][Epoch 1/10] DLoss(Avg): -0.3381 | GLoss(Avg): -0.2518 | IS: 1.8398 ± 0.0162) | FID: 237.82 | GP[avg]: 0.01
# [8x8][Epoch 2/10 | Iter: 636/1272] Disc Loss: -0.3928 | Gen Loss: -0.2920
#  -- 😱 Batch-636:  D_real_avg: 😟 +0.6953 ± +0.5276 📈| D_fake_avg: 😟 +0.2330 ± +0.4417 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😱 Last Batch : D_real_avg: 😰 +0.2190 ± +0.6184 📈 | D_fake_avg: 😟 +0.0753 ± +0.6187 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😟 +0.5082 ± +0.5082 📈 | D_fake_avg: 😟 +0.2243 ± +0.4580 📉
# [8x8][Epoch 2/10] DLoss(Avg): -0.2317 | GLoss(Avg): -0.0205 | IS: 1.9827 ± 0.0250) | FID: 246.65 | GP[avg]: 0.01
# [8x8][Epoch 3/10 | Iter: 636/1272] Disc Loss: -0.3901 | Gen Loss: 0.2850
#  -- 😱 Batch-636:  D_real_avg: 😟 +0.6061 ± +0.6391 📈| D_fake_avg: 😟 +0.1602 ± +0.5669 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😱 Last Batch : D_real_avg: 😟 +0.7216 ± +0.4983 📈 | D_fake_avg: 😵 +0.6259 ± +0.3808 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😟 +0.4913 ± +0.4913 📈 | D_fake_avg: 😟 +0.2682 ± +0.3911 📉
# [8x8][Epoch 3/10] DLoss(Avg): -0.1820 | GLoss(Avg): -0.0614 | IS: 1.9779 ± 0.0293) | FID: 247.36 | GP[avg]: 0.01
# [8x8][Epoch 4/10 | Iter: 636/1272] Disc Loss: -0.0616 | Gen Loss: -0.0945
#  -- 😱 Batch-636:  D_real_avg: 😟 +0.6809 ± +0.4455 📈| D_fake_avg: 😵 +0.5875 ± +0.4418 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😱 Last Batch : D_real_avg: 😟 +0.4865 ± +0.3512 📈 | D_fake_avg: 😟 +0.2868 ± +0.3185 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😰 +0.3781 ± +0.3781 📈 | D_fake_avg: 😟 +0.1800 ± +0.3677 📉
# [8x8][Epoch 4/10] DLoss(Avg): -0.1616 | GLoss(Avg): 0.0310 | IS: 1.9988 ± 0.0295) | FID: 245.54 | GP[avg]: 0.01
# [8x8][Epoch 5/10 | Iter: 636/1272] Disc Loss: -0.1111 | Gen Loss: -0.0302
#  -- 😱 Batch-636:  D_real_avg: 😵 +0.1661 ± +0.5042 📈| D_fake_avg: 😟 +0.0206 ± +0.4234 📉
# Using FID stats for celeba-train_10K from cache...
#  -- ‼️ Last Batch : D_real_avg: 😟 +0.7617 ± +0.4295 📈 | D_fake_avg: 😵 +0.9282 ± +0.3697 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😰 +0.2393 ± +0.2393 📈 | D_fake_avg: 😟 +0.0741 ± +0.3246 📉
# [8x8][Epoch 5/10] DLoss(Avg): -0.1327 | GLoss(Avg): 0.1152 | IS: 2.0626 ± 0.0292) | FID: 241.29 | GP[avg]: 0.01
# [8x8][Epoch 6/10 | Iter: 636/1272] Disc Loss: -0.1085 | Gen Loss: 0.0298
#  -- 🫤 Batch-636:  D_real_avg: 😵 -0.0150 ± +0.3219 📈| D_fake_avg: 😎 -0.1494 ± +0.3251 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 🫤 Last Batch : D_real_avg: 😵 -0.1711 ± +0.4231 📈 | D_fake_avg: 😎 -0.3154 ± +0.3866 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😰 +0.2100 ± +0.2100 📈 | D_fake_avg: 😟 +0.0730 ± +0.3167 📉
# [8x8][Epoch 6/10] DLoss(Avg): -0.1081 | GLoss(Avg): 0.0941 | IS: 1.9326 ± 0.0264) | FID: 244.56 | GP[avg]: 0.01
# [8x8][Epoch 7/10 | Iter: 636/1272] Disc Loss: -0.1162 | Gen Loss: -0.0734
#  -- 😱 Batch-636:  D_real_avg: 😰 +0.2823 ± +0.3283 📈| D_fake_avg: 😟 +0.1356 ± +0.2357 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😱 Last Batch : D_real_avg: 😰 +0.2803 ± +0.4583 📈 | D_fake_avg: 😟 +0.0942 ± +0.4144 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😰 +0.2126 ± +0.2126 📈 | D_fake_avg: 😟 +0.0798 ± +0.2894 📉
# [8x8][Epoch 7/10] DLoss(Avg): -0.1052 | GLoss(Avg): 0.0707 | IS: 2.0704 ± 0.0235) | FID: 241.79 | GP[avg]: 0.01
# [8x8][Epoch 8/10 | Iter: 636/1272] Disc Loss: -0.1329 | Gen Loss: -0.5302
#  -- 😱 Batch-636:  D_real_avg: 😎 +0.8480 ± +0.3319 📈| D_fake_avg: 😵 +0.6935 ± +0.3151 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😱 Last Batch : D_real_avg: 😟 +0.4432 ± +0.3448 📈 | D_fake_avg: 😟 +0.2893 ± +0.3459 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😰 +0.2009 ± +0.2009 📈 | D_fake_avg: 😟 +0.0745 ± +0.2864 📉
# [8x8][Epoch 8/10] DLoss(Avg): -0.0998 | GLoss(Avg): 0.0635 | IS: 2.0692 ± 0.0281) | FID: 246.39 | GP[avg]: 0.01
# [8x8][Epoch 9/10 | Iter: 636/1272] Disc Loss: -0.2325 | Gen Loss: 0.6407
#  -- 🫤 Batch-636:  D_real_avg: 😵 -0.2819 ± +0.2937 📈| D_fake_avg: 😎 -0.5358 ± +0.3255 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 🫤 Last Batch : D_real_avg: 😵 -0.1488 ± +0.3815 📈 | D_fake_avg: 😎 -0.2810 ± +0.3117 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😵 +0.1829 ± +0.1829 📈 | D_fake_avg: 😟 +0.0616 ± +0.2737 📉
# [8x8][Epoch 9/10] DLoss(Avg): -0.0950 | GLoss(Avg): 0.0628 | IS: 2.0104 ± 0.0168) | FID: 243.27 | GP[avg]: 0.01
# Files already downloaded and verified
#  Step: 2/7 -> Training on [16x16]
#   --Epochs:                      10 
#   --BatchSize:                   128 
#   --Number of Batches:           1272 
#   --Interval:                    637 
#   --Fade-in Steps:               6360 
#   --Last training Step taken:    0 
#   --Current Discriminator LR:    0.001
#   --Current Generator LR:        0.001
#   --Current Discriminator Betas: [0, 0.99]
#   --Current Generator Betas:     [0, 0.99]
# [16x16][Epoch 0/10 | Iter: 636/1272] Disc Loss: -1.6957 | Gen Loss: -1.3111
#  -- 😱 Batch-636:  D_real_avg: 😎 +5.5809 ± +1.1949 📈| D_fake_avg: 😵 +3.6694 ± +0.8042 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😱 Last Batch : D_real_avg: 😎 +5.4075 ± +1.1244 📈 | D_fake_avg: 😵 +4.5114 ± +0.7527 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +2.0893 ± +2.0893 📈 | D_fake_avg: 😵 +0.8669 ± +1.6463 📉
# [16x16][Epoch 0/10] DLoss(Avg): -1.0787 | GLoss(Avg): -0.4270 | IS: 2.5254 ± 0.0405) | FID: 205.54 | GP[avg]: 0.03
# [16x16][Epoch 1/10 | Iter: 636/1272] Disc Loss: -0.5668 | Gen Loss: 0.1282
#  -- 🫤 Batch-636:  D_real_avg: 😵 +0.0358 ± +0.6200 📈| D_fake_avg: 😎 -0.5675 ± +0.5959 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😱 Last Batch : D_real_avg: 😎 +2.8472 ± +0.6349 📈 | D_fake_avg: 😵 +2.2517 ± +0.5536 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +1.1790 ± +1.1790 📈 | D_fake_avg: 😵 +0.6332 ± +1.4976 📉
# [16x16][Epoch 1/10] DLoss(Avg): -0.5065 | GLoss(Avg): -0.2252 | IS: 2.7223 ± 0.0275) | FID: 126.94 | GP[avg]: 0.01
# [16x16][Epoch 2/10 | Iter: 636/1272] Disc Loss: -0.3604 | Gen Loss: 2.3163
#  -- 🫤 Batch-636:  D_real_avg: 😵 -0.8462 ± +0.6164 📈| D_fake_avg: 😎 -1.2259 ± +0.6466 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😱 Last Batch : D_real_avg: 😎 +2.4722 ± +1.0725 📈 | D_fake_avg: 😵 +2.2629 ± +1.0466 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😟 +0.6340 ± +0.6340 📈 | D_fake_avg: 😟 +0.2803 ± +1.0039 📉
# [16x16][Epoch 2/10] DLoss(Avg): -0.3279 | GLoss(Avg): 0.1194 | IS: 2.7263 ± 0.0597) | FID: 113.29 | GP[avg]: 0.01
# [16x16][Epoch 3/10 | Iter: 636/1272] Disc Loss: -0.1763 | Gen Loss: 0.7588
#  -- 🫤 Batch-636:  D_real_avg: 😵 -0.1146 ± +0.5819 📈| D_fake_avg: 😎 -0.3131 ± +0.4780 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😱 Last Batch : D_real_avg: 😟 +0.7771 ± +0.5570 📈 | D_fake_avg: 😵 +0.4243 ± +0.4130 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😟 +0.6181 ± +0.6181 📈 | D_fake_avg: 😟 +0.3078 ± +0.8407 📉
# [16x16][Epoch 3/10] DLoss(Avg): -0.2879 | GLoss(Avg): 0.0765 | IS: 2.8125 ± 0.0304) | FID: 125.60 | GP[avg]: 0.00
# [16x16][Epoch 4/10 | Iter: 636/1272] Disc Loss: -0.4061 | Gen Loss: 0.5118
#  -- 😱 Batch-636:  D_real_avg: 😟 +0.5907 ± +0.4273 📈| D_fake_avg: 😟 +0.1635 ± +0.3792 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 🫤 Last Batch : D_real_avg: 😰 +0.3471 ± +0.4665 📈 | D_fake_avg: 😎 -0.2026 ± +0.4244 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😟 +0.5199 ± +0.5199 📈 | D_fake_avg: 😟 +0.1900 ± +0.8223 📉
# [16x16][Epoch 4/10] DLoss(Avg): -0.3041 | GLoss(Avg): 0.1682 | IS: 2.7698 ± 0.0395) | FID: 136.20 | GP[avg]: 0.01
# [16x16][Epoch 5/10 | Iter: 636/1272] Disc Loss: -0.3806 | Gen Loss: 0.5930
#  -- 🫤 Batch-636:  D_real_avg: 😵 -0.1718 ± +0.5092 📈| D_fake_avg: 😎 -0.5715 ± +0.4476 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 🫤 Last Batch : D_real_avg: 😵 -0.8277 ± +0.8382 📈 | D_fake_avg: 😎 -1.0742 ± +0.9041 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😰 +0.3509 ± +0.3509 📈 | D_fake_avg: 😟 +0.0013 ± +0.8271 📉
# [16x16][Epoch 5/10] DLoss(Avg): -0.3238 | GLoss(Avg): 0.3292 | IS: 2.5848 ± 0.0276) | FID: 126.24 | GP[avg]: 0.01
# [16x16][Epoch 6/10 | Iter: 636/1272] Disc Loss: -0.3106 | Gen Loss: -0.4958
#  -- 🫤 Batch-636:  D_real_avg: 😵 -0.0463 ± +0.5847 📈| D_fake_avg: 😎 -0.3976 ± +0.4725 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 🫤 Last Batch : D_real_avg: 😰 +0.2484 ± +0.4992 📈 | D_fake_avg: 😎 -0.0994 ± +0.4024 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😰 +0.3740 ± +0.3740 📈 | D_fake_avg: 😟 +0.0356 ± +0.7752 📉
# [16x16][Epoch 6/10] DLoss(Avg): -0.3127 | GLoss(Avg): 0.2583 | IS: 2.8236 ± 0.0421) | FID: 140.11 | GP[avg]: 0.01
# [16x16][Epoch 7/10 | Iter: 636/1272] Disc Loss: -0.2459 | Gen Loss: 0.4088
#  -- 🫤 Batch-636:  D_real_avg: 😵 -0.1897 ± +0.5413 📈| D_fake_avg: 😎 -0.4573 ± +0.4820 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 🫤 Last Batch : D_real_avg: 😵 -0.3622 ± +0.4673 📈 | D_fake_avg: 😎 -0.6549 ± +0.4061 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😰 +0.3841 ± +0.3841 📈 | D_fake_avg: 😟 +0.0618 ± +0.7023 📉
# [16x16][Epoch 7/10] DLoss(Avg): -0.2977 | GLoss(Avg): 0.2133 | IS: 2.9116 ± 0.0853) | FID: 123.96 | GP[avg]: 0.00
# [16x16][Epoch 8/10 | Iter: 636/1272] Disc Loss: -0.3340 | Gen Loss: -0.2769
#  -- 🫤 Batch-636:  D_real_avg: 😵 +0.1737 ± +0.4660 📈| D_fake_avg: 😎 -0.1835 ± +0.3934 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😱 Last Batch : D_real_avg: 😎 +1.1004 ± +0.7620 📈 | D_fake_avg: 😵 +0.7897 ± +0.6406 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😰 +0.3905 ± +0.3905 📈 | D_fake_avg: 😟 +0.0813 ± +0.6914 📉
# [16x16][Epoch 8/10] DLoss(Avg): -0.2848 | GLoss(Avg): 0.1744 | IS: 2.8451 ± 0.0615) | FID: 117.93 | GP[avg]: 0.00
# [16x16][Epoch 9/10 | Iter: 636/1272] Disc Loss: -0.3294 | Gen Loss: -0.7759
#  -- 😱 Batch-636:  D_real_avg: 😎 +1.0814 ± +0.4496 📈| D_fake_avg: 😵 +0.7289 ± +0.4362 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😱 Last Batch : D_real_avg: 😎 +0.8952 ± +0.4993 📈 | D_fake_avg: 😵 +0.5611 ± +0.5205 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😰 +0.3884 ± +0.3884 📈 | D_fake_avg: 😟 +0.0912 ± +0.7115 📉
# [16x16][Epoch 9/10] DLoss(Avg): -0.2732 | GLoss(Avg): 0.1466 | IS: 2.9151 ± 0.0369) | FID: 117.43 | GP[avg]: 0.00
# Files already downloaded and verified
#  Step: 3/7 -> Training on [32x32]
#   --Epochs:                      30 
#   --BatchSize:                   128 
#   --Number of Batches:           1272 
#   --Interval:                    637 
#   --Fade-in Steps:               19080 
#   --Last training Step taken:    0 
#   --Current Discriminator LR:    0.001
#   --Current Generator LR:        0.001
#   --Current Discriminator Betas: [0, 0.99]
#   --Current Generator Betas:     [0, 0.99]
# [32x32][Epoch 0/30 | Iter: 636/1272] Disc Loss: -1.2415 | Gen Loss: 1.2311
#  -- 😱 Batch-636:  D_real_avg: 😎 +3.2820 ± +0.8886 📈| D_fake_avg: 😵 +1.9968 ± +0.6764 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😀 Last Batch : D_real_avg: 😎 +1.6370 ± +1.9188 📈 | D_fake_avg: 😎 -0.3086 ± +1.9319 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +3.3526 ± +3.3526 📈 | D_fake_avg: 😵 +1.9043 ± +2.8738 📉
# [32x32][Epoch 0/30] DLoss(Avg): -1.3134 | GLoss(Avg): -1.0944 | IS: 2.7931 ± 0.0405) | FID: 91.49 | GP[avg]: 0.03
# [32x32][Epoch 1/30 | Iter: 636/1272] Disc Loss: -2.2392 | Gen Loss: -1.6712
#  -- 😱 Batch-636:  D_real_avg: 😎 +4.7975 ± +1.7515 📈| D_fake_avg: 😵 +2.1521 ± +1.0200 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😱 Last Batch : D_real_avg: 😎 +4.8652 ± +1.0632 📈 | D_fake_avg: 😵 +2.4115 ± +0.7919 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +4.6619 ± +4.6619 📈 | D_fake_avg: 😵 +2.2233 ± +4.5768 📉
# [32x32][Epoch 1/30] DLoss(Avg): -2.0911 | GLoss(Avg): -1.4029 | IS: 2.8740 ± 0.0921) | FID: 61.93 | GP[avg]: 0.07
# [32x32][Epoch 2/30 | Iter: 636/1272] Disc Loss: -0.7903 | Gen Loss: -0.0256
#  -- 😱 Batch-636:  D_real_avg: 😎 +7.4767 ± +1.8407 📈| D_fake_avg: 😵 +6.5758 ± +1.3343 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 🫤 Last Batch : D_real_avg: 😵 -1.3233 ± +1.5201 📈 | D_fake_avg: 😎 -1.5672 ± +1.3964 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +3.3923 ± +3.3923 📈 | D_fake_avg: 😵 +2.3048 ± +3.1573 📉
# [32x32][Epoch 2/30] DLoss(Avg): -1.0026 | GLoss(Avg): -1.7433 | IS: 2.8791 ± 0.0504) | FID: 48.39 | GP[avg]: 0.02
# [32x32][Epoch 3/30 | Iter: 636/1272] Disc Loss: -0.3220 | Gen Loss: 2.7174
#  -- 😱 Batch-636:  D_real_avg: 😎 +4.5638 ± +1.2720 📈| D_fake_avg: 😵 +4.1975 ± +1.0616 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 🫤 Last Batch : D_real_avg: 😵 -0.2821 ± +1.4367 📈 | D_fake_avg: 😎 -1.0777 ± +1.5884 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +1.7225 ± +1.7225 📈 | D_fake_avg: 😵 +0.8940 ± +2.3195 📉
# [32x32][Epoch 3/30] DLoss(Avg): -0.7738 | GLoss(Avg): -0.3705 | IS: 2.7227 ± 0.0367) | FID: 43.49 | GP[avg]: 0.01
# [32x32][Epoch 4/30 | Iter: 636/1272] Disc Loss: -0.8247 | Gen Loss: -1.7893
#  -- 😱 Batch-636:  D_real_avg: 😎 +2.0981 ± +1.1602 📈| D_fake_avg: 😵 +1.2248 ± +0.9431 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😀 Last Batch : D_real_avg: 😵 +0.0243 ± +1.0373 📈 | D_fake_avg: 😎 -1.0651 ± +0.8500 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +1.2037 ± +1.2037 📈 | D_fake_avg: 😵 +0.4122 ± +2.1329 📉
# [32x32][Epoch 4/30] DLoss(Avg): -0.7436 | GLoss(Avg): 0.0863 | IS: 3.0275 ± 0.0940) | FID: 47.10 | GP[avg]: 0.01
# [32x32][Epoch 5/30 | Iter: 636/1272] Disc Loss: -0.5505 | Gen Loss: 0.7406
#  -- 😱 Batch-636:  D_real_avg: 😟 +0.7330 ± +1.4294 📈| D_fake_avg: 😟 +0.1132 ± +1.4256 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😱 Last Batch : D_real_avg: 😎 +3.1216 ± +1.2034 📈 | D_fake_avg: 😵 +2.8845 ± +1.4065 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +1.1855 ± +1.1855 📈 | D_fake_avg: 😵 +0.4501 ± +1.8358 📉
# [32x32][Epoch 5/30] DLoss(Avg): -0.6927 | GLoss(Avg): 0.0460 | IS: 2.8536 ± 0.0589) | FID: 35.30 | GP[avg]: 0.01
# [32x32][Epoch 6/30 | Iter: 636/1272] Disc Loss: -0.6134 | Gen Loss: 2.2153
#  -- 🫤 Batch-636:  D_real_avg: 😵 +0.0506 ± +0.9454 📈| D_fake_avg: 😎 -0.6048 ± +0.9557 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 🫤 Last Batch : D_real_avg: 😵 -1.2289 ± +1.8000 📈 | D_fake_avg: 😎 -1.6221 ± +1.7056 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +1.1055 ± +1.1055 📈 | D_fake_avg: 😵 +0.4144 ± +1.7439 📉
# [32x32][Epoch 6/30] DLoss(Avg): -0.6516 | GLoss(Avg): 0.0580 | IS: 2.9178 ± 0.0593) | FID: 33.56 | GP[avg]: 0.01
# [32x32][Epoch 7/30 | Iter: 636/1272] Disc Loss: -0.6638 | Gen Loss: 2.9274
#  -- 😱 Batch-636:  D_real_avg: 😎 +0.8025 ± +1.0183 📈| D_fake_avg: 😟 +0.1146 ± +0.8499 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 🫤 Last Batch : D_real_avg: 😵 -0.0947 ± +1.0736 📈 | D_fake_avg: 😎 -0.9884 ± +0.6895 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +1.0212 ± +1.0212 📈 | D_fake_avg: 😟 +0.3426 ± +1.6567 📉
# [32x32][Epoch 7/30] DLoss(Avg): -0.6401 | GLoss(Avg): 0.1031 | IS: 3.0165 ± 0.0843) | FID: 37.53 | GP[avg]: 0.01
# [32x32][Epoch 8/30 | Iter: 636/1272] Disc Loss: -0.8619 | Gen Loss: -0.0956
#  -- 🫤 Batch-636:  D_real_avg: 😎 +0.8038 ± +0.9285 📈| D_fake_avg: 😎 -0.0908 ± +0.8900 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😱 Last Batch : D_real_avg: 😎 +1.9116 ± +0.9160 📈 | D_fake_avg: 😵 +1.2660 ± +0.6584 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +0.9924 ± +0.9924 📈 | D_fake_avg: 😟 +0.3460 ± +1.5652 📉
# [32x32][Epoch 8/30] DLoss(Avg): -0.6092 | GLoss(Avg): 0.1074 | IS: 2.9905 ± 0.0372) | FID: 35.09 | GP[avg]: 0.01
# [32x32][Epoch 9/30 | Iter: 636/1272] Disc Loss: -0.4993 | Gen Loss: -0.0859
#  -- 😱 Batch-636:  D_real_avg: 😟 +0.5621 ± +0.7032 📈| D_fake_avg: 😟 +0.0139 ± +0.7481 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😱 Last Batch : D_real_avg: 😎 +3.6655 ± +0.9580 📈 | D_fake_avg: 😵 +2.8334 ± +0.7183 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +0.9931 ± +0.9931 📈 | D_fake_avg: 😟 +0.3593 ± +1.5462 📉
# [32x32][Epoch 9/30] DLoss(Avg): -0.5972 | GLoss(Avg): 0.0733 | IS: 2.9283 ± 0.0741) | FID: 27.42 | GP[avg]: 0.01
# [32x32][Epoch 10/30 | Iter: 636/1272] Disc Loss: -0.5938 | Gen Loss: 0.2614
#  -- 🫤 Batch-636:  D_real_avg: 😵 -1.6225 ± +1.1940 📈| D_fake_avg: 😎 -2.2421 ± +1.0547 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😱 Last Batch : D_real_avg: 😎 +2.2091 ± +1.1076 📈 | D_fake_avg: 😵 +1.8263 ± +0.9922 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +0.9693 ± +0.9693 📈 | D_fake_avg: 😟 +0.3469 ± +1.5726 📉
# [32x32][Epoch 10/30] DLoss(Avg): -0.5867 | GLoss(Avg): 0.0691 | IS: 2.9123 ± 0.0728) | FID: 32.15 | GP[avg]: 0.01
# [32x32][Epoch 11/30 | Iter: 636/1272] Disc Loss: -0.4922 | Gen Loss: 1.5231
#  -- 🫤 Batch-636:  D_real_avg: 😵 -1.3257 ± +0.9523 📈| D_fake_avg: 😎 -1.8373 ± +1.0139 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😱 Last Batch : D_real_avg: 😎 +3.2527 ± +1.1900 📈 | D_fake_avg: 😵 +2.4109 ± +1.1352 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +0.9774 ± +0.9774 📈 | D_fake_avg: 😟 +0.3584 ± +1.4098 📉
# [32x32][Epoch 11/30] DLoss(Avg): -0.5838 | GLoss(Avg): 0.0386 | IS: 3.0923 ± 0.0623) | FID: 30.00 | GP[avg]: 0.01
# [32x32][Epoch 12/30 | Iter: 636/1272] Disc Loss: -0.4460 | Gen Loss: 0.2487
#  -- 😱 Batch-636:  D_real_avg: 😎 +1.8363 ± +0.8561 📈| D_fake_avg: 😵 +1.3628 ± +0.7475 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😱 Last Batch : D_real_avg: 😎 +0.9999 ± +0.8563 📈 | D_fake_avg: 😟 +0.3194 ± +0.8979 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +0.9320 ± +0.9320 📈 | D_fake_avg: 😟 +0.3135 ± +1.4616 📉
# [32x32][Epoch 12/30] DLoss(Avg): -0.5828 | GLoss(Avg): 0.0783 | IS: 3.1266 ± 0.0642) | FID: 34.79 | GP[avg]: 0.01
# [32x32][Epoch 13/30 | Iter: 636/1272] Disc Loss: -0.6977 | Gen Loss: 1.3642
#  -- 🫤 Batch-636:  D_real_avg: 😵 -0.5987 ± +0.8572 📈| D_fake_avg: 😎 -1.3282 ± +0.8520 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😱 Last Batch : D_real_avg: 😎 +1.6376 ± +1.3089 📈 | D_fake_avg: 😵 +1.3252 ± +0.9650 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +0.9237 ± +0.9237 📈 | D_fake_avg: 😟 +0.3046 ± +1.3019 📉
# [32x32][Epoch 13/30] DLoss(Avg): -0.5829 | GLoss(Avg): 0.0775 | IS: 2.7303 ± 0.0655) | FID: 27.46 | GP[avg]: 0.01
# [32x32][Epoch 14/30 | Iter: 636/1272] Disc Loss: -0.4140 | Gen Loss: 2.9280
#  -- 😱 Batch-636:  D_real_avg: 😟 +0.5215 ± +0.9950 📈| D_fake_avg: 😟 +0.0788 ± +0.8784 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😱 Last Batch : D_real_avg: 😎 +1.8319 ± +0.9070 📈 | D_fake_avg: 😵 +1.3725 ± +1.1905 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +0.9072 ± +0.9072 📈 | D_fake_avg: 😟 +0.2912 ± +1.3243 📉
# [32x32][Epoch 14/30] DLoss(Avg): -0.5793 | GLoss(Avg): 0.0901 | IS: 2.9889 ± 0.0612) | FID: 26.29 | GP[avg]: 0.01
# [32x32][Epoch 15/30 | Iter: 636/1272] Disc Loss: -0.5717 | Gen Loss: -1.8330
#  -- 😱 Batch-636:  D_real_avg: 😎 +1.9164 ± +0.7670 📈| D_fake_avg: 😵 +1.3263 ± +0.7602 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😱 Last Batch : D_real_avg: 😎 +0.9776 ± +0.8548 📈 | D_fake_avg: 😟 +0.2628 ± +0.7911 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +0.8887 ± +0.8887 📈 | D_fake_avg: 😟 +0.2744 ± +1.3714 📉
# [32x32][Epoch 15/30] DLoss(Avg): -0.5785 | GLoss(Avg): 0.0908 | IS: 2.9324 ± 0.0598) | FID: 28.09 | GP[avg]: 0.01
# [32x32][Epoch 16/30 | Iter: 636/1272] Disc Loss: -0.3912 | Gen Loss: -1.0041
#  -- 😱 Batch-636:  D_real_avg: 😎 +1.4663 ± +0.8790 📈| D_fake_avg: 😵 +1.0173 ± +0.7189 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 🫤 Last Batch : D_real_avg: 😵 -1.4241 ± +0.7353 📈 | D_fake_avg: 😎 -2.1096 ± +0.7244 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +0.9013 ± +0.9013 📈 | D_fake_avg: 😟 +0.3093 ± +1.3239 📉
# [32x32][Epoch 16/30] DLoss(Avg): -0.5581 | GLoss(Avg): 0.0604 | IS: 2.6926 ± 0.0658) | FID: 23.75 | GP[avg]: 0.01
# [32x32][Epoch 17/30 | Iter: 636/1272] Disc Loss: -0.5939 | Gen Loss: 1.2384
#  -- 🫤 Batch-636:  D_real_avg: 😵 -0.1352 ± +0.7595 📈| D_fake_avg: 😎 -0.7596 ± +0.6301 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 🫤 Last Batch : D_real_avg: 😟 +0.7376 ± +0.7065 📈 | D_fake_avg: 😎 -0.0775 ± +0.6680 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +0.8834 ± +0.8834 📈 | D_fake_avg: 😟 +0.2974 ± +1.3471 📉
# [32x32][Epoch 17/30] DLoss(Avg): -0.5530 | GLoss(Avg): 0.0506 | IS: 2.7085 ± 0.0553) | FID: 27.85 | GP[avg]: 0.01
# [32x32][Epoch 18/30 | Iter: 636/1272] Disc Loss: -0.5757 | Gen Loss: -0.1095
#  -- 🫤 Batch-636:  D_real_avg: 😰 +0.2166 ± +1.2918 📈| D_fake_avg: 😎 -0.3887 ± +1.4000 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 🫤 Last Batch : D_real_avg: 😟 +0.5816 ± +0.8916 📈 | D_fake_avg: 😎 -0.1902 ± +0.9276 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +0.8896 ± +0.8896 📈 | D_fake_avg: 😟 +0.3137 ± +1.2571 📉
# [32x32][Epoch 18/30] DLoss(Avg): -0.5442 | GLoss(Avg): 0.0347 | IS: 2.7845 ± 0.0777) | FID: 25.36 | GP[avg]: 0.01
# [32x32][Epoch 19/30 | Iter: 636/1272] Disc Loss: -0.3702 | Gen Loss: 0.5716
#  -- 😱 Batch-636:  D_real_avg: 😎 +2.2336 ± +0.8430 📈| D_fake_avg: 😵 +1.8136 ± +0.7803 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 🫤 Last Batch : D_real_avg: 😰 +0.3305 ± +0.8058 📈 | D_fake_avg: 😎 -0.6500 ± +0.6904 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +0.8715 ± +0.8715 📈 | D_fake_avg: 😟 +0.3006 ± +1.2489 📉
# [32x32][Epoch 19/30] DLoss(Avg): -0.5395 | GLoss(Avg): 0.0381 | IS: 2.7540 ± 0.0876) | FID: 21.90 | GP[avg]: 0.01
# [32x32][Epoch 20/30 | Iter: 636/1272] Disc Loss: -0.5681 | Gen Loss: -0.4281
#  -- 😱 Batch-636:  D_real_avg: 😎 +2.8541 ± +0.8903 📈| D_fake_avg: 😵 +2.2388 ± +0.6918 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 🫤 Last Batch : D_real_avg: 😟 +0.7119 ± +0.8407 📈 | D_fake_avg: 😎 -0.0528 ± +0.5734 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +0.8683 ± +0.8683 📈 | D_fake_avg: 😟 +0.3021 ± +1.2771 📉
# [32x32][Epoch 20/30] DLoss(Avg): -0.5356 | GLoss(Avg): 0.0254 | IS: 2.7466 ± 0.0707) | FID: 22.63 | GP[avg]: 0.01
# [32x32][Epoch 21/30 | Iter: 636/1272] Disc Loss: -0.5507 | Gen Loss: -1.6498
#  -- 😱 Batch-636:  D_real_avg: 😎 +1.1822 ± +0.8738 📈| D_fake_avg: 😵 +0.6029 ± +0.9107 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😱 Last Batch : D_real_avg: 😟 +0.7627 ± +0.9516 📈 | D_fake_avg: 😟 +0.3380 ± +0.9860 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +0.8749 ± +0.8749 📈 | D_fake_avg: 😟 +0.3143 ± +1.3247 📉
# [32x32][Epoch 21/30] DLoss(Avg): -0.5306 | GLoss(Avg): 0.0085 | IS: 3.0025 ± 0.0567) | FID: 28.10 | GP[avg]: 0.01
# [32x32][Epoch 22/30 | Iter: 636/1272] Disc Loss: -0.5689 | Gen Loss: 0.9175
#  -- 🫤 Batch-636:  D_real_avg: 😟 +0.4471 ± +0.7179 📈| D_fake_avg: 😎 -0.1477 ± +0.5155 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😱 Last Batch : D_real_avg: 😟 +0.6908 ± +0.7218 📈 | D_fake_avg: 😟 +0.0820 ± +0.5564 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +0.8578 ± +0.8578 📈 | D_fake_avg: 😟 +0.3015 ± +1.3132 📉
# [32x32][Epoch 22/30] DLoss(Avg): -0.5270 | GLoss(Avg): 0.0062 | IS: 2.9091 ± 0.0557) | FID: 22.99 | GP[avg]: 0.01
# [32x32][Epoch 23/30 | Iter: 636/1272] Disc Loss: -0.3891 | Gen Loss: 1.1437
#  -- 🫤 Batch-636:  D_real_avg: 😵 -0.7894 ± +0.7378 📈| D_fake_avg: 😎 -1.1993 ± +0.6694 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😱 Last Batch : D_real_avg: 😎 +1.8940 ± +0.8243 📈 | D_fake_avg: 😵 +1.0989 ± +0.5710 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +0.8517 ± +0.8517 📈 | D_fake_avg: 😟 +0.3029 ± +1.2704 📉
# [32x32][Epoch 23/30] DLoss(Avg): -0.5200 | GLoss(Avg): 0.0041 | IS: 2.8218 ± 0.0553) | FID: 21.76 | GP[avg]: 0.01
# [32x32][Epoch 24/30 | Iter: 636/1272] Disc Loss: -0.4252 | Gen Loss: 0.5874
#  -- 😱 Batch-636:  D_real_avg: 😎 +2.2106 ± +0.9274 📈| D_fake_avg: 😵 +1.7404 ± +0.9802 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😱 Last Batch : D_real_avg: 😎 +1.6306 ± +0.7557 📈 | D_fake_avg: 😵 +1.2126 ± +0.5428 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +0.8462 ± +0.8462 📈 | D_fake_avg: 😟 +0.3018 ± +1.2588 📉
# [32x32][Epoch 24/30] DLoss(Avg): -0.5159 | GLoss(Avg): 0.0011 | IS: 2.9249 ± 0.0659) | FID: 23.95 | GP[avg]: 0.01
# [32x32][Epoch 25/30 | Iter: 636/1272] Disc Loss: -0.6005 | Gen Loss: -2.5432
#  -- 🫤 Batch-636:  D_real_avg: 😰 +0.2494 ± +0.7393 📈| D_fake_avg: 😎 -0.3779 ± +0.6238 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😱 Last Batch : D_real_avg: 😎 +1.3033 ± +0.9440 📈 | D_fake_avg: 😵 +0.6487 ± +0.8208 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +0.8580 ± +0.8580 📈 | D_fake_avg: 😟 +0.3159 ± +1.2066 📉
# [32x32][Epoch 25/30] DLoss(Avg): -0.5137 | GLoss(Avg): -0.0188 | IS: 2.8837 ± 0.0407) | FID: 24.54 | GP[avg]: 0.01
# [32x32][Epoch 26/30 | Iter: 636/1272] Disc Loss: -0.7327 | Gen Loss: 1.9403
#  -- 🫤 Batch-636:  D_real_avg: 😰 +0.2124 ± +0.6102 📈| D_fake_avg: 😎 -0.5344 ± +0.6471 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 🫤 Last Batch : D_real_avg: 😵 -0.6423 ± +0.6164 📈 | D_fake_avg: 😎 -1.2752 ± +0.6255 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +0.8414 ± +0.8414 📈 | D_fake_avg: 😟 +0.3013 ± +1.1002 📉
# [32x32][Epoch 26/30] DLoss(Avg): -0.5119 | GLoss(Avg): -0.0139 | IS: 2.8261 ± 0.0652) | FID: 23.27 | GP[avg]: 0.01
# [32x32][Epoch 27/30 | Iter: 636/1272] Disc Loss: -0.5652 | Gen Loss: -2.8197
#  -- 😱 Batch-636:  D_real_avg: 😎 +1.2503 ± +0.8081 📈| D_fake_avg: 😵 +0.6629 ± +0.6213 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😱 Last Batch : D_real_avg: 😎 +2.0472 ± +0.8724 📈 | D_fake_avg: 😵 +1.4286 ± +0.8351 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +0.8645 ± +0.8645 📈 | D_fake_avg: 😟 +0.3285 ± +1.2486 📉
# [32x32][Epoch 27/30] DLoss(Avg): -0.5076 | GLoss(Avg): -0.0381 | IS: 2.6699 ± 0.0596) | FID: 22.98 | GP[avg]: 0.01
# [32x32][Epoch 28/30 | Iter: 636/1272] Disc Loss: -0.1303 | Gen Loss: 0.6649
#  -- 😱 Batch-636:  D_real_avg: 😎 +2.2225 ± +1.2938 📈| D_fake_avg: 😵 +2.0476 ± +1.3143 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 🫤 Last Batch : D_real_avg: 😵 +0.0162 ± +0.7318 📈 | D_fake_avg: 😎 -0.4953 ± +0.7081 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +0.8419 ± +0.8419 📈 | D_fake_avg: 😟 +0.3076 ± +1.2777 📉
# [32x32][Epoch 28/30] DLoss(Avg): -0.5060 | GLoss(Avg): -0.0212 | IS: 2.8846 ± 0.0574) | FID: 23.35 | GP[avg]: 0.01
# [32x32][Epoch 29/30 | Iter: 636/1272] Disc Loss: -0.5694 | Gen Loss: 0.9096
#  -- 🫤 Batch-636:  D_real_avg: 😟 +0.6047 ± +0.7055 📈| D_fake_avg: 😎 -0.0061 ± +0.6057 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😱 Last Batch : D_real_avg: 😎 +2.4808 ± +0.8470 📈 | D_fake_avg: 😵 +1.7664 ± +0.8116 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +0.8405 ± +0.8405 📈 | D_fake_avg: 😟 +0.3043 ± +1.0644 📉
# [32x32][Epoch 29/30] DLoss(Avg): -0.5085 | GLoss(Avg): -0.0355 | IS: 2.8701 ± 0.0643) | FID: 23.52 | GP[avg]: 0.01
# Files already downloaded and verified
#  Step: 4/7 -> Training on [64x64]
#   --Epochs:                      40 
#   --BatchSize:                   64 
#   --Number of Batches:           2544 
#   --Interval:                    1273 
#   --Fade-in Steps:               50880 
#   --Last training Step taken:    0 
#   --Current Discriminator LR:    0.001
#   --Current Generator LR:        0.001
#   --Current Discriminator Betas: [0, 0.99]
#   --Current Generator Betas:     [0, 0.99]
# [64x64][Epoch 0/40 | Iter: 1272/2544] Disc Loss: -1.4680 | Gen Loss: -2.1855
#  -- 😱 Batch-1272:  D_real_avg: 😎 +2.4003 ± +1.5967 📈| D_fake_avg: 😵 +0.8773 ± +1.4818 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😱 Last Batch : D_real_avg: 😎 +15.0507 ± +3.3533 📈 | D_fake_avg: 😵 +13.1487 ± +3.2024 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +3.3896 ± +3.3896 📈 | D_fake_avg: 😵 +1.8741 ± +3.3982 📉
# [64x64][Epoch 0/40] DLoss(Avg): -1.4212 | GLoss(Avg): -1.1424 | IS: 2.7524 ± 0.0453) | FID: 28.14 | GP[avg]: 0.02
# [64x64][Epoch 1/40 | Iter: 1272/2544] Disc Loss: -3.0494 | Gen Loss: 1.1169
#  -- 😱 Batch-1272:  D_real_avg: 😎 +9.4841 ± +2.1930 📈| D_fake_avg: 😵 +5.9894 ± +1.4849 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😱 Last Batch : D_real_avg: 😎 +9.7822 ± +3.3100 📈 | D_fake_avg: 😵 +8.7852 ± +3.5171 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +5.5693 ± +5.5693 📈 | D_fake_avg: 😵 +2.8582 ± +6.0780 📉
# [64x64][Epoch 1/40] DLoss(Avg): -2.3609 | GLoss(Avg): -1.8870 | IS: 2.6446 ± 0.0516) | FID: 49.29 | GP[avg]: 0.07
# [64x64][Epoch 2/40 | Iter: 1272/2544] Disc Loss: -1.5936 | Gen Loss: -1.8544
#  -- 😱 Batch-1272:  D_real_avg: 😎 +3.4493 ± +1.5209 📈| D_fake_avg: 😵 +1.7603 ± +1.3438 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😀 Last Batch : D_real_avg: 😵 +0.0381 ± +1.9450 📈 | D_fake_avg: 😎 -1.4863 ± +1.3510 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +5.6077 ± +5.6077 📈 | D_fake_avg: 😵 +3.8337 ± +4.0341 📉
# [64x64][Epoch 2/40] DLoss(Avg): -1.5798 | GLoss(Avg): -3.2073 | IS: 2.7108 ± 0.0531) | FID: 55.37 | GP[avg]: 0.04
# [64x64][Epoch 3/40 | Iter: 1272/2544] Disc Loss: -1.2476 | Gen Loss: -1.5275
#  -- 😀 Batch-1272:  D_real_avg: 😟 +0.7618 ± +1.4616 📈| D_fake_avg: 😎 -0.6022 ± +1.6931 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😱 Last Batch : D_real_avg: 😎 +11.3602 ± +2.6425 📈 | D_fake_avg: 😵 +8.4061 ± +1.4889 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +3.5265 ± +3.5265 📈 | D_fake_avg: 😵 +2.0708 ± +3.3875 📉
# [64x64][Epoch 3/40] DLoss(Avg): -1.3256 | GLoss(Avg): -1.4850 | IS: 2.8746 ± 0.0689) | FID: 52.29 | GP[avg]: 0.03
# [64x64][Epoch 4/40 | Iter: 1272/2544] Disc Loss: -1.2363 | Gen Loss: -1.0228
#  -- 😱 Batch-1272:  D_real_avg: 😎 +5.6925 ± +1.9085 📈| D_fake_avg: 😵 +4.3455 ± +1.5079 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😱 Last Batch : D_real_avg: 😎 +6.2164 ± +1.2220 📈 | D_fake_avg: 😵 +4.9725 ± +1.6367 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +2.1503 ± +2.1503 📈 | D_fake_avg: 😵 +0.7459 ± +3.4379 📉
# [64x64][Epoch 4/40] DLoss(Avg): -1.2954 | GLoss(Avg): -0.1379 | IS: 2.8162 ± 0.0752) | FID: 62.12 | GP[avg]: 0.02
# [64x64][Epoch 5/40 | Iter: 1272/2544] Disc Loss: -1.9901 | Gen Loss: 1.0257
#  -- 😀 Batch-1272:  D_real_avg: 😎 +1.8335 ± +1.4198 📈| D_fake_avg: 😎 -0.2302 ± +1.2452 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😱 Last Batch : D_real_avg: 😎 +0.9981 ± +1.4415 📈 | D_fake_avg: 😵 +0.4003 ± +1.0066 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +2.0487 ± +2.0487 📈 | D_fake_avg: 😵 +0.7102 ± +3.1348 📉
# [64x64][Epoch 5/40] DLoss(Avg): -1.2405 | GLoss(Avg): -0.0954 | IS: 2.6971 ± 0.0491) | FID: 62.25 | GP[avg]: 0.02
# [64x64][Epoch 6/40 | Iter: 1272/2544] Disc Loss: -0.7203 | Gen Loss: -1.4223
#  -- 😱 Batch-1272:  D_real_avg: 😎 +3.0558 ± +1.8790 📈| D_fake_avg: 😵 +2.2162 ± +1.6737 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😱 Last Batch : D_real_avg: 😎 +2.0796 ± +1.8789 📈 | D_fake_avg: 😵 +1.1729 ± +1.4523 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +2.0084 ± +2.0084 📈 | D_fake_avg: 😵 +0.7010 ± +3.1554 📉
# [64x64][Epoch 6/40] DLoss(Avg): -1.2140 | GLoss(Avg): -0.1099 | IS: 2.6950 ± 0.0620) | FID: 59.30 | GP[avg]: 0.02
# [64x64][Epoch 7/40 | Iter: 1272/2544] Disc Loss: -1.9917 | Gen Loss: -0.3455
#  -- 😱 Batch-1272:  D_real_avg: 😎 +2.1676 ± +1.5663 📈| D_fake_avg: 😟 +0.1179 ± +1.2565 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😀 Last Batch : D_real_avg: 😵 -2.0632 ± +1.8170 📈 | D_fake_avg: 😎 -3.9134 ± +2.4764 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +1.9545 ± +1.9545 📈 | D_fake_avg: 😵 +0.6722 ± +3.0503 📉
# [64x64][Epoch 7/40] DLoss(Avg): -1.1917 | GLoss(Avg): -0.0946 | IS: 2.7984 ± 0.0547) | FID: 64.37 | GP[avg]: 0.02
# [64x64][Epoch 8/40 | Iter: 1272/2544] Disc Loss: -1.1614 | Gen Loss: -3.4301
#  -- 😀 Batch-1272:  D_real_avg: 😟 +0.4851 ± +1.4345 📈| D_fake_avg: 😎 -0.7412 ± +1.1555 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😱 Last Batch : D_real_avg: 😎 +4.1431 ± +2.2417 📈 | D_fake_avg: 😵 +2.6022 ± +1.9255 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +1.9273 ± +1.9273 📈 | D_fake_avg: 😵 +0.6729 ± +2.8438 📉
# [64x64][Epoch 8/40] DLoss(Avg): -1.1674 | GLoss(Avg): -0.1167 | IS: 2.6318 ± 0.0581) | FID: 65.52 | GP[avg]: 0.02
# [64x64][Epoch 9/40 | Iter: 1272/2544] Disc Loss: -1.1499 | Gen Loss: -1.6860
#  -- 😱 Batch-1272:  D_real_avg: 😎 +1.9679 ± +1.3505 📈| D_fake_avg: 😵 +0.7485 ± +1.1892 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😀 Last Batch : D_real_avg: 😟 +0.7394 ± +1.5605 📈 | D_fake_avg: 😎 -0.9630 ± +1.1569 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +1.8176 ± +1.8176 📈 | D_fake_avg: 😵 +0.5834 ± +2.8270 📉
# [64x64][Epoch 9/40] DLoss(Avg): -1.1498 | GLoss(Avg): -0.0403 | IS: 2.5871 ± 0.0510) | FID: 60.82 | GP[avg]: 0.02
# [64x64][Epoch 10/40 | Iter: 1272/2544] Disc Loss: -1.3226 | Gen Loss: 1.1982
#  -- 😀 Batch-1272:  D_real_avg: 😎 +1.3234 ± +1.3425 📈| D_fake_avg: 😎 -0.0876 ± +1.0847 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😱 Last Batch : D_real_avg: 😎 +2.6630 ± +1.3270 📈 | D_fake_avg: 😵 +1.1062 ± +2.2158 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +1.7576 ± +1.7576 📈 | D_fake_avg: 😵 +0.5423 ± +2.7147 📉
# [64x64][Epoch 10/40] DLoss(Avg): -1.1329 | GLoss(Avg): -0.0032 | IS: 2.9608 ± 0.0881) | FID: 58.74 | GP[avg]: 0.02
# [64x64][Epoch 11/40 | Iter: 1272/2544] Disc Loss: -1.1577 | Gen Loss: -1.3166
#  -- 😱 Batch-1272:  D_real_avg: 😎 +4.5795 ± +1.7538 📈| D_fake_avg: 😵 +3.3292 ± +1.6106 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😱 Last Batch : D_real_avg: 😎 +1.3502 ± +1.9997 📈 | D_fake_avg: 😵 +0.6703 ± +1.9375 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +1.6948 ± +1.6948 📈 | D_fake_avg: 😵 +0.4808 ± +2.5454 📉
# [64x64][Epoch 11/40] DLoss(Avg): -1.1317 | GLoss(Avg): 0.0296 | IS: 2.6625 ± 0.0492) | FID: 64.52 | GP[avg]: 0.02
# [64x64][Epoch 12/40 | Iter: 1272/2544] Disc Loss: -1.2384 | Gen Loss: -2.0209
#  -- 😱 Batch-1272:  D_real_avg: 😎 +1.4528 ± +1.9141 📈| D_fake_avg: 😟 +0.1309 ± +1.3352 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😱 Last Batch : D_real_avg: 😎 +2.7457 ± +1.1857 📈 | D_fake_avg: 😵 +0.9853 ± +1.0823 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +1.6751 ± +1.6751 📈 | D_fake_avg: 😵 +0.4757 ± +2.4823 📉
# [64x64][Epoch 12/40] DLoss(Avg): -1.1186 | GLoss(Avg): 0.0285 | IS: 2.7383 ± 0.0629) | FID: 67.88 | GP[avg]: 0.02
# [64x64][Epoch 13/40 | Iter: 1272/2544] Disc Loss: -1.3775 | Gen Loss: -1.7569
#  -- 😱 Batch-1272:  D_real_avg: 😎 +1.6858 ± +1.4092 📈| D_fake_avg: 😟 +0.2274 ± +1.4578 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😀 Last Batch : D_real_avg: 😵 -1.0747 ± +1.3434 📈 | D_fake_avg: 😎 -2.8030 ± +1.5862 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +1.6921 ± +1.6921 📈 | D_fake_avg: 😵 +0.5103 ± +2.7070 📉
# [64x64][Epoch 13/40] DLoss(Avg): -1.1017 | GLoss(Avg): -0.0014 | IS: 2.6536 ± 0.0450) | FID: 63.38 | GP[avg]: 0.02
# [64x64][Epoch 14/40 | Iter: 1272/2544] Disc Loss: -0.7095 | Gen Loss: 1.3063
#  -- 😱 Batch-1272:  D_real_avg: 😎 +3.8443 ± +1.3784 📈| D_fake_avg: 😵 +3.0536 ± +1.4700 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😱 Last Batch : D_real_avg: 😎 +1.0896 ± +2.6750 📈 | D_fake_avg: 😟 +0.3350 ± +1.9076 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +1.6687 ± +1.6687 📈 | D_fake_avg: 😵 +0.4823 ± +2.5251 📉
# [64x64][Epoch 14/40] DLoss(Avg): -1.1062 | GLoss(Avg): -0.0015 | IS: 2.7843 ± 0.0554) | FID: 62.94 | GP[avg]: 0.02
# [64x64][Epoch 15/40 | Iter: 1272/2544] Disc Loss: -0.5022 | Gen Loss: -1.4597
#  -- 😱 Batch-1272:  D_real_avg: 😎 +3.6165 ± +1.2689 📈| D_fake_avg: 😵 +3.0659 ± +0.9862 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😀 Last Batch : D_real_avg: 😰 +0.2178 ± +1.5204 📈 | D_fake_avg: 😎 -1.8475 ± +1.2804 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +1.6543 ± +1.6543 📈 | D_fake_avg: 😵 +0.4803 ± +2.5515 📉
# [64x64][Epoch 15/40] DLoss(Avg): -1.0953 | GLoss(Avg): 0.0132 | IS: 2.7547 ± 0.0464) | FID: 64.63 | GP[avg]: 0.02
# [64x64][Epoch 16/40 | Iter: 1272/2544] Disc Loss: -0.7733 | Gen Loss: -0.2229
#  -- 😱 Batch-1272:  D_real_avg: 😎 +1.8142 ± +1.7137 📈| D_fake_avg: 😵 +0.9535 ± +1.5001 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😱 Last Batch : D_real_avg: 😎 +4.9696 ± +2.0064 📈 | D_fake_avg: 😵 +3.0314 ± +1.6491 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +1.6644 ± +1.6644 📈 | D_fake_avg: 😵 +0.4837 ± +2.5728 📉
# [64x64][Epoch 16/40] DLoss(Avg): -1.0999 | GLoss(Avg): 0.0012 | IS: 2.8617 ± 0.0685) | FID: 72.26 | GP[avg]: 0.02
# [64x64][Epoch 17/40 | Iter: 1272/2544] Disc Loss: -1.1121 | Gen Loss: -1.5329
#  -- 😱 Batch-1272:  D_real_avg: 😎 +4.9105 ± +1.8222 📈| D_fake_avg: 😵 +3.6734 ± +1.4905 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😀 Last Batch : D_real_avg: 😵 -1.0104 ± +1.5599 📈 | D_fake_avg: 😎 -2.2319 ± +1.1933 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +1.6362 ± +1.6362 📈 | D_fake_avg: 😵 +0.4622 ± +2.4045 📉
# [64x64][Epoch 17/40] DLoss(Avg): -1.0940 | GLoss(Avg): 0.0111 | IS: 2.7002 ± 0.0410) | FID: 63.93 | GP[avg]: 0.02
# [64x64][Epoch 18/40 | Iter: 1272/2544] Disc Loss: -1.2856 | Gen Loss: -4.3384
#  -- 😱 Batch-1272:  D_real_avg: 😎 +5.5401 ± +1.5721 📈| D_fake_avg: 😵 +4.1940 ± +1.2313 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 🫤 Last Batch : D_real_avg: 😵 -2.1552 ± +1.5232 📈 | D_fake_avg: 😎 -2.7384 ± +1.7321 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +1.6719 ± +1.6719 📈 | D_fake_avg: 😵 +0.4943 ± +2.5349 📉
# [64x64][Epoch 18/40] DLoss(Avg): -1.0968 | GLoss(Avg): -0.0055 | IS: 2.7701 ± 0.0490) | FID: 66.20 | GP[avg]: 0.02
# [64x64][Epoch 19/40 | Iter: 1272/2544] Disc Loss: -1.1490 | Gen Loss: -0.2216
#  -- 😀 Batch-1272:  D_real_avg: 😵 +0.0847 ± +1.7079 📈| D_fake_avg: 😎 -1.1738 ± +1.7451 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😀 Last Batch : D_real_avg: 😟 +0.7417 ± +1.5284 📈 | D_fake_avg: 😎 -0.7976 ± +2.0880 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +1.6484 ± +1.6484 📈 | D_fake_avg: 😵 +0.4567 ± +2.5672 📉
# [64x64][Epoch 19/40] DLoss(Avg): -1.1090 | GLoss(Avg): 0.0121 | IS: 2.6828 ± 0.0564) | FID: 72.66 | GP[avg]: 0.02
# [64x64][Epoch 20/40 | Iter: 1272/2544] Disc Loss: -1.1262 | Gen Loss: 1.3220
#  -- 😀 Batch-1272:  D_real_avg: 😟 +0.7735 ± +1.3282 📈| D_fake_avg: 😎 -0.3828 ± +1.5260 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😱 Last Batch : D_real_avg: 😎 +3.9808 ± +1.8234 📈 | D_fake_avg: 😵 +2.9636 ± +1.3665 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +1.6335 ± +1.6335 📈 | D_fake_avg: 😵 +0.4453 ± +2.3810 📉
# [64x64][Epoch 20/40] DLoss(Avg): -1.1066 | GLoss(Avg): 0.0295 | IS: 2.6436 ± 0.0534) | FID: 73.87 | GP[avg]: 0.02
# [64x64][Epoch 21/40 | Iter: 1272/2544] Disc Loss: -1.3338 | Gen Loss: 0.4703
#  -- 😱 Batch-1272:  D_real_avg: 😎 +2.4117 ± +1.7538 📈| D_fake_avg: 😵 +0.9858 ± +1.3789 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😱 Last Batch : D_real_avg: 😎 +5.9494 ± +1.1312 📈 | D_fake_avg: 😵 +4.4915 ± +1.3889 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +1.6310 ± +1.6310 📈 | D_fake_avg: 😵 +0.4436 ± +2.5189 📉
# [64x64][Epoch 21/40] DLoss(Avg): -1.1059 | GLoss(Avg): 0.0224 | IS: 2.7577 ± 0.0532) | FID: 70.83 | GP[avg]: 0.02
# [64x64][Epoch 22/40 | Iter: 1272/2544] Disc Loss: -0.8035 | Gen Loss: -3.8237
#  -- 😱 Batch-1272:  D_real_avg: 😎 +2.8213 ± +2.3307 📈| D_fake_avg: 😵 +1.9438 ± +1.6167 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 🫤 Last Batch : D_real_avg: 😵 -1.0342 ± +2.0293 📈 | D_fake_avg: 😎 -1.8232 ± +1.1917 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +1.6394 ± +1.6394 📈 | D_fake_avg: 😵 +0.4552 ± +2.5084 📉
# [64x64][Epoch 22/40] DLoss(Avg): -1.1036 | GLoss(Avg): 0.0111 | IS: 2.8048 ± 0.0733) | FID: 66.42 | GP[avg]: 0.02
# [64x64][Epoch 23/40 | Iter: 1272/2544] Disc Loss: -1.3526 | Gen Loss: 4.7033
#  -- 😀 Batch-1272:  D_real_avg: 😵 -0.8260 ± +1.6711 📈| D_fake_avg: 😎 -2.2267 ± +1.6573 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😀 Last Batch : D_real_avg: 😟 +0.4253 ± +1.9015 📈 | D_fake_avg: 😎 -1.4591 ± +1.4891 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +1.6459 ± +1.6459 📈 | D_fake_avg: 😵 +0.4541 ± +2.3760 📉
# [64x64][Epoch 23/40] DLoss(Avg): -1.1104 | GLoss(Avg): 0.0020 | IS: 2.7448 ± 0.0476) | FID: 70.90 | GP[avg]: 0.02
# [64x64][Epoch 24/40 | Iter: 1272/2544] Disc Loss: -0.8441 | Gen Loss: -0.3029
#  -- 😱 Batch-1272:  D_real_avg: 😎 +2.4673 ± +1.7338 📈| D_fake_avg: 😵 +1.4926 ± +1.3777 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😱 Last Batch : D_real_avg: 😎 +5.1134 ± +1.6552 📈 | D_fake_avg: 😵 +3.7855 ± +1.4905 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +1.6478 ± +1.6478 📈 | D_fake_avg: 😵 +0.4569 ± +2.4012 📉
# [64x64][Epoch 24/40] DLoss(Avg): -1.1097 | GLoss(Avg): -0.0026 | IS: 2.7690 ± 0.0637) | FID: 70.81 | GP[avg]: 0.02
# [64x64][Epoch 25/40 | Iter: 1272/2544] Disc Loss: -1.1616 | Gen Loss: -3.3647
#  -- 😱 Batch-1272:  D_real_avg: 😎 +4.0747 ± +1.5615 📈| D_fake_avg: 😵 +2.7990 ± +1.3285 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😱 Last Batch : D_real_avg: 😎 +7.1875 ± +2.3366 📈 | D_fake_avg: 😵 +7.1117 ± +1.7681 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +1.6348 ± +1.6348 📈 | D_fake_avg: 😵 +0.4404 ± +2.4153 📉
# [64x64][Epoch 25/40] DLoss(Avg): -1.1122 | GLoss(Avg): 0.0043 | IS: 2.8623 ± 0.0500) | FID: 65.48 | GP[avg]: 0.02
# [64x64][Epoch 26/40 | Iter: 1272/2544] Disc Loss: -0.6399 | Gen Loss: 0.0119
#  -- 😱 Batch-1272:  D_real_avg: 😎 +3.5809 ± +2.2635 📈| D_fake_avg: 😵 +2.8199 ± +1.8428 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😱 Last Batch : D_real_avg: 😎 +3.2424 ± +1.8967 📈 | D_fake_avg: 😵 +2.8901 ± +2.1480 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +1.6685 ± +1.6685 📈 | D_fake_avg: 😵 +0.4794 ± +2.5140 📉
# [64x64][Epoch 26/40] DLoss(Avg): -1.1078 | GLoss(Avg): -0.0199 | IS: 2.4825 ± 0.0422) | FID: 62.21 | GP[avg]: 0.02
# [64x64][Epoch 27/40 | Iter: 1272/2544] Disc Loss: -0.8322 | Gen Loss: 1.3733
#  -- 🫤 Batch-1272:  D_real_avg: 😵 -1.4148 ± +1.4159 📈| D_fake_avg: 😎 -2.3187 ± +1.7800 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😱 Last Batch : D_real_avg: 😎 +1.9128 ± +1.2990 📈 | D_fake_avg: 😵 +0.9161 ± +1.1795 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +1.6277 ± +1.6277 📈 | D_fake_avg: 😵 +0.4248 ± +2.4125 📉
# [64x64][Epoch 27/40] DLoss(Avg): -1.1201 | GLoss(Avg): 0.0129 | IS: 2.7417 ± 0.0342) | FID: 63.10 | GP[avg]: 0.02
# [64x64][Epoch 28/40 | Iter: 1272/2544] Disc Loss: -1.3597 | Gen Loss: 4.0865
#  -- 😀 Batch-1272:  D_real_avg: 😟 +0.7780 ± +1.3353 📈| D_fake_avg: 😎 -0.6369 ± +1.3319 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😀 Last Batch : D_real_avg: 😵 -1.3351 ± +1.4300 📈 | D_fake_avg: 😎 -3.2884 ± +1.2542 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +1.6634 ± +1.6634 📈 | D_fake_avg: 😵 +0.4516 ± +2.4879 📉
# [64x64][Epoch 28/40] DLoss(Avg): -1.1280 | GLoss(Avg): -0.0130 | IS: 2.8361 ± 0.0689) | FID: 61.72 | GP[avg]: 0.02
# [64x64][Epoch 29/40 | Iter: 1272/2544] Disc Loss: -0.5370 | Gen Loss: -1.6042
#  -- 😱 Batch-1272:  D_real_avg: 😎 +4.1694 ± +1.7309 📈| D_fake_avg: 😵 +3.5315 ± +1.5231 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😀 Last Batch : D_real_avg: 😟 +0.7688 ± +1.3687 📈 | D_fake_avg: 😎 -0.2932 ± +1.8096 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +1.6611 ± +1.6611 📈 | D_fake_avg: 😵 +0.4470 ± +2.2621 📉
# [64x64][Epoch 29/40] DLoss(Avg): -1.1306 | GLoss(Avg): -0.0234 | IS: 2.6785 ± 0.0441) | FID: 63.02 | GP[avg]: 0.02
# [64x64][Epoch 30/40 | Iter: 1272/2544] Disc Loss: -0.8870 | Gen Loss: -1.9219
#  -- 🫤 Batch-1272:  D_real_avg: 😟 +0.6829 ± +1.6271 📈| D_fake_avg: 😎 -0.2705 ± +1.8295 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😀 Last Batch : D_real_avg: 😎 +0.9034 ± +1.5272 📈 | D_fake_avg: 😎 -0.5978 ± +1.4874 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +1.6701 ± +1.6701 📈 | D_fake_avg: 😵 +0.4610 ± +2.4661 📉
# [64x64][Epoch 30/40] DLoss(Avg): -1.1255 | GLoss(Avg): -0.0343 | IS: 2.7626 ± 0.0587) | FID: 67.20 | GP[avg]: 0.02
# [64x64][Epoch 31/40 | Iter: 1272/2544] Disc Loss: -1.0201 | Gen Loss: -1.0743
#  -- 😱 Batch-1272:  D_real_avg: 😎 +1.3069 ± +1.3054 📈| D_fake_avg: 😟 +0.2049 ± +1.1819 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😱 Last Batch : D_real_avg: 😎 +2.6120 ± +1.6965 📈 | D_fake_avg: 😵 +1.5320 ± +1.8375 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +1.6643 ± +1.6643 📈 | D_fake_avg: 😵 +0.4471 ± +2.4843 📉
# [64x64][Epoch 31/40] DLoss(Avg): -1.1326 | GLoss(Avg): -0.0198 | IS: 2.6865 ± 0.0552) | FID: 64.18 | GP[avg]: 0.02
# [64x64][Epoch 32/40 | Iter: 1272/2544] Disc Loss: -1.4386 | Gen Loss: -1.9351
#  -- 😱 Batch-1272:  D_real_avg: 😎 +1.9995 ± +1.5577 📈| D_fake_avg: 😵 +0.5028 ± +1.0163 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😀 Last Batch : D_real_avg: 😵 -1.8169 ± +1.6093 📈 | D_fake_avg: 😎 -2.9354 ± +1.4647 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +1.6904 ± +1.6904 📈 | D_fake_avg: 😵 +0.4680 ± +2.3313 📉
# [64x64][Epoch 32/40] DLoss(Avg): -1.1373 | GLoss(Avg): -0.0427 | IS: 2.6816 ± 0.0542) | FID: 65.28 | GP[avg]: 0.02
# [64x64][Epoch 33/40 | Iter: 1272/2544] Disc Loss: -1.1647 | Gen Loss: -0.2473
#  -- 😀 Batch-1272:  D_real_avg: 😎 +1.0494 ± +1.6574 📈| D_fake_avg: 😎 -0.1691 ± +1.3020 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😀 Last Batch : D_real_avg: 😎 +1.0518 ± +2.7385 📈 | D_fake_avg: 😎 -0.7368 ± +1.5590 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +1.6811 ± +1.6811 📈 | D_fake_avg: 😵 +0.4548 ± +2.4107 📉
# [64x64][Epoch 33/40] DLoss(Avg): -1.1411 | GLoss(Avg): -0.0362 | IS: 2.6672 ± 0.0502) | FID: 66.99 | GP[avg]: 0.02
# [64x64][Epoch 34/40 | Iter: 1272/2544] Disc Loss: -1.3061 | Gen Loss: 0.5820
#  -- 😱 Batch-1272:  D_real_avg: 😎 +2.1261 ± +1.5664 📈| D_fake_avg: 😵 +0.7468 ± +1.5362 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😀 Last Batch : D_real_avg: 😎 +1.8459 ± +2.3533 📈 | D_fake_avg: 😎 -0.2083 ± +2.0369 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +1.6821 ± +1.6821 📈 | D_fake_avg: 😵 +0.4581 ± +2.3987 📉
# [64x64][Epoch 34/40] DLoss(Avg): -1.1392 | GLoss(Avg): -0.0451 | IS: 2.8234 ± 0.0751) | FID: 73.28 | GP[avg]: 0.02
# [64x64][Epoch 35/40 | Iter: 1272/2544] Disc Loss: -1.5757 | Gen Loss: 3.8951
#  -- 😀 Batch-1272:  D_real_avg: 😵 -0.6295 ± +1.3246 📈| D_fake_avg: 😎 -2.2673 ± +1.1880 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😱 Last Batch : D_real_avg: 😎 +4.1509 ± +1.9250 📈 | D_fake_avg: 😵 +3.3511 ± +1.0124 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +1.6876 ± +1.6876 📈 | D_fake_avg: 😵 +0.4619 ± +2.3642 📉
# [64x64][Epoch 35/40] DLoss(Avg): -1.1407 | GLoss(Avg): -0.0516 | IS: 2.6717 ± 0.0356) | FID: 71.45 | GP[avg]: 0.02
# [64x64][Epoch 36/40 | Iter: 1272/2544] Disc Loss: -1.4851 | Gen Loss: 0.7726
#  -- 😀 Batch-1272:  D_real_avg: 😎 +1.1314 ± +1.6542 📈| D_fake_avg: 😎 -0.4439 ± +1.7364 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😱 Last Batch : D_real_avg: 😎 +4.0320 ± +1.9193 📈 | D_fake_avg: 😵 +3.1639 ± +1.4059 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +1.6580 ± +1.6580 📈 | D_fake_avg: 😵 +0.4295 ± +2.4211 📉
# [64x64][Epoch 36/40] DLoss(Avg): -1.1431 | GLoss(Avg): -0.0113 | IS: 2.9291 ± 0.0692) | FID: 68.56 | GP[avg]: 0.02
# [64x64][Epoch 37/40 | Iter: 1272/2544] Disc Loss: -1.0914 | Gen Loss: -1.8732
#  -- 😱 Batch-1272:  D_real_avg: 😎 +3.9382 ± +1.5822 📈| D_fake_avg: 😵 +2.7625 ± +1.5888 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😀 Last Batch : D_real_avg: 😵 -2.4442 ± +1.5921 📈 | D_fake_avg: 😎 -3.7682 ± +1.6987 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +1.6939 ± +1.6939 📈 | D_fake_avg: 😵 +0.4621 ± +2.3620 📉
# [64x64][Epoch 37/40] DLoss(Avg): -1.1465 | GLoss(Avg): -0.0418 | IS: 2.6212 ± 0.0328) | FID: 63.46 | GP[avg]: 0.02
# [64x64][Epoch 38/40 | Iter: 1272/2544] Disc Loss: -1.2898 | Gen Loss: -0.5412
#  -- 😱 Batch-1272:  D_real_avg: 😎 +3.6361 ± +1.6790 📈| D_fake_avg: 😵 +2.2847 ± +1.5175 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😱 Last Batch : D_real_avg: 😎 +2.9484 ± +1.1174 📈 | D_fake_avg: 😵 +1.4045 ± +0.9940 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +1.6563 ± +1.6563 📈 | D_fake_avg: 😵 +0.4193 ± +2.5003 📉
# [64x64][Epoch 38/40] DLoss(Avg): -1.1517 | GLoss(Avg): -0.0200 | IS: 2.6789 ± 0.0568) | FID: 64.99 | GP[avg]: 0.02
# [64x64][Epoch 39/40 | Iter: 1272/2544] Disc Loss: -1.7353 | Gen Loss: 0.8213
#  -- 😱 Batch-1272:  D_real_avg: 😎 +2.7084 ± +1.9584 📈| D_fake_avg: 😵 +0.8666 ± +1.6017 📉
# Using FID stats for celeba-train_10K from cache...
#  -- ‼️ Last Batch : D_real_avg: 😎 +4.8756 ± +1.5475 📈 | D_fake_avg: 😵 +5.0822 ± +1.8732 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +1.7018 ± +1.7018 📈 | D_fake_avg: 😵 +0.4629 ± +2.2425 📉
# [64x64][Epoch 39/40] DLoss(Avg): -1.1527 | GLoss(Avg): -0.0608 | IS: 2.7978 ± 0.0363) | FID: 74.90 | GP[avg]: 0.02
# Files already downloaded and verified
#  Step: 5/7 -> Training on [128x128]
#   --Epochs:                      40 
#   --BatchSize:                   32 
#   --Number of Batches:           5087 
#   --Interval:                    2544 
#   --Fade-in Steps:               101740 
#   --Last training Step taken:    0 
#   --Current Discriminator LR:    0.001
#   --Current Generator LR:        0.001
#   --Current Discriminator Betas: [0, 0.99]
#   --Current Generator Betas:     [0, 0.99]
# [128x128][Epoch 0/40 | Iter: 2543/5087] Disc Loss: -3.6623 | Gen Loss: -13.6196
#  -- 😱 Batch-2543:  D_real_avg: 😎 +9.3962 ± +5.2220 📈| D_fake_avg: 😵 +5.4109 ± +2.1279 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😱 Last Batch : D_real_avg: 😎 +10.9558 ± +4.4700 📈 | D_fake_avg: 😵 +6.5109 ± +2.5163 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +5.6352 ± +5.6352 📈 | D_fake_avg: 😵 +2.3892 ± +7.6095 📉
# [128x128][Epoch 0/40] DLoss(Avg): -2.8983 | GLoss(Avg): -1.3319 | IS: 2.6605 ± 0.0547) | FID: 75.32 | GP[avg]: 0.07
# [128x128][Epoch 1/40 | Iter: 2543/5087] Disc Loss: -3.4595 | Gen Loss: -10.3281
#  -- 😱 Batch-2543:  D_real_avg: 😎 +5.3306 ± +2.9100 📈| D_fake_avg: 😵 +1.3468 ± +2.2137 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😱 Last Batch : D_real_avg: 😎 +18.1006 ± +4.7497 📈 | D_fake_avg: 😵 +15.6530 ± +3.8944 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +7.1562 ± +7.1562 📈 | D_fake_avg: 😵 +3.2913 ± +7.7395 📉
# [128x128][Epoch 1/40] DLoss(Avg): -3.3117 | GLoss(Avg): -2.3767 | IS: 2.9790 ± 0.0566) | FID: 96.23 | GP[avg]: 0.11
# [128x128][Epoch 2/40 | Iter: 2543/5087] Disc Loss: -3.6503 | Gen Loss: -13.1799
#  -- 😱 Batch-2543:  D_real_avg: 😎 +13.2722 ± +5.6368 📈| D_fake_avg: 😵 +9.0895 ± +3.5566 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😀 Last Batch : D_real_avg: 😵 -3.6726 ± +3.1654 📈 | D_fake_avg: 😎 -8.0478 ± +3.5224 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +5.9970 ± +5.9970 📈 | D_fake_avg: 😵 +2.7421 ± +7.0842 📉
# [128x128][Epoch 2/40] DLoss(Avg): -2.8353 | GLoss(Avg): -1.8405 | IS: 2.8663 ± 0.0644) | FID: 103.55 | GP[avg]: 0.08
# [128x128][Epoch 3/40 | Iter: 2543/5087] Disc Loss: -2.5269 | Gen Loss: 15.5859
#  -- 😀 Batch-2543:  D_real_avg: 😵 -6.7520 ± +4.9024 📈| D_fake_avg: 😎 -9.6355 ± +5.2051 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😱 Last Batch : D_real_avg: 😎 +3.0505 ± +4.2614 📈 | D_fake_avg: 😵 +1.0004 ± +3.0838 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +4.9127 ± +4.9127 📈 | D_fake_avg: 😵 +1.7451 ± +7.0426 📉
# [128x128][Epoch 3/40] DLoss(Avg): -2.7800 | GLoss(Avg): -0.8432 | IS: 2.8136 ± 0.0353) | FID: 115.48 | GP[avg]: 0.08
# [128x128][Epoch 4/40 | Iter: 2543/5087] Disc Loss: -3.1733 | Gen Loss: -6.0760
#  -- 😱 Batch-2543:  D_real_avg: 😎 +8.1544 ± +5.5077 📈| D_fake_avg: 😵 +4.5455 ± +3.9312 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😱 Last Batch : D_real_avg: 😎 +15.5785 ± +4.4691 📈 | D_fake_avg: 😵 +12.7505 ± +5.6919 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +4.3047 ± +4.3047 📈 | D_fake_avg: 😵 +1.2544 ± +6.4945 📉
# [128x128][Epoch 4/40] DLoss(Avg): -2.6902 | GLoss(Avg): -0.3743 | IS: 2.8781 ± 0.0877) | FID: 121.63 | GP[avg]: 0.07
# [128x128][Epoch 5/40 | Iter: 2543/5087] Disc Loss: -4.0735 | Gen Loss: 12.6269
#  -- 😀 Batch-2543:  D_real_avg: 😵 -1.9196 ± +5.4732 📈| D_fake_avg: 😎 -6.3278 ± +4.1829 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😀 Last Batch : D_real_avg: 😵 -6.7113 ± +4.6847 📈 | D_fake_avg: 😎 -10.9621 ± +4.9942 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +4.1813 ± +4.1813 📈 | D_fake_avg: 😵 +1.1837 ± +6.4514 📉
# [128x128][Epoch 5/40] DLoss(Avg): -2.6504 | GLoss(Avg): -0.2928 | IS: 2.7739 ± 0.0622) | FID: 115.36 | GP[avg]: 0.07
# [128x128][Epoch 6/40 | Iter: 2543/5087] Disc Loss: -3.8139 | Gen Loss: -3.3123
#  -- 😱 Batch-2543:  D_real_avg: 😎 +4.2449 ± +4.9179 📈| D_fake_avg: 😟 +0.0687 ± +4.0927 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😀 Last Batch : D_real_avg: 😎 +3.2889 ± +3.2488 📈 | D_fake_avg: 😎 -1.4138 ± +2.6062 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +4.1653 ± +4.1653 📈 | D_fake_avg: 😵 +1.1639 ± +6.3639 📉
# [128x128][Epoch 6/40] DLoss(Avg): -2.6565 | GLoss(Avg): -0.2927 | IS: 3.0194 ± 0.0507) | FID: 120.42 | GP[avg]: 0.07
# [128x128][Epoch 7/40 | Iter: 2543/5087] Disc Loss: -2.9172 | Gen Loss: -6.6332
#  -- 😱 Batch-2543:  D_real_avg: 😎 +5.2605 ± +3.1757 📈| D_fake_avg: 😵 +2.0555 ± +2.5172 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😱 Last Batch : D_real_avg: 😎 +5.7253 ± +2.3663 📈 | D_fake_avg: 😵 +3.5811 ± +3.6588 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +4.1400 ± +4.1400 📈 | D_fake_avg: 😵 +1.1336 ± +6.3453 📉
# [128x128][Epoch 7/40] DLoss(Avg): -2.6620 | GLoss(Avg): -0.2659 | IS: 2.9958 ± 0.0524) | FID: 122.16 | GP[avg]: 0.07
# [128x128][Epoch 8/40 | Iter: 2543/5087] Disc Loss: -0.8883 | Gen Loss: -4.4606
#  -- 😱 Batch-2543:  D_real_avg: 😎 +6.4813 ± +3.6347 📈| D_fake_avg: 😵 +5.2158 ± +4.2558 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😀 Last Batch : D_real_avg: 😵 -1.1576 ± +2.6307 📈 | D_fake_avg: 😎 -3.3454 ± +3.0013 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +4.1129 ± +4.1129 📈 | D_fake_avg: 😵 +1.1533 ± +6.4465 📉
# [128x128][Epoch 8/40] DLoss(Avg): -2.6242 | GLoss(Avg): -0.2843 | IS: 2.8897 ± 0.0386) | FID: 116.15 | GP[avg]: 0.07
# [128x128][Epoch 9/40 | Iter: 2543/5087] Disc Loss: -4.1808 | Gen Loss: 12.0230
#  -- 😀 Batch-2543:  D_real_avg: 😎 +0.8834 ± +4.5494 📈| D_fake_avg: 😎 -3.6988 ± +4.6390 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😀 Last Batch : D_real_avg: 😵 -4.7443 ± +3.5597 📈 | D_fake_avg: 😎 -7.2245 ± +3.7767 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +4.0204 ± +4.0204 📈 | D_fake_avg: 😵 +1.0538 ± +6.2646 📉
# [128x128][Epoch 9/40] DLoss(Avg): -2.6323 | GLoss(Avg): -0.1977 | IS: 2.9071 ± 0.0385) | FID: 115.59 | GP[avg]: 0.07
# [128x128][Epoch 10/40 | Iter: 2543/5087] Disc Loss: -3.4377 | Gen Loss: 3.4703
#  -- 😀 Batch-2543:  D_real_avg: 😵 -0.3324 ± +3.2622 📈| D_fake_avg: 😎 -4.0988 ± +3.8194 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😀 Last Batch : D_real_avg: 😎 +3.2430 ± +4.7138 📈 | D_fake_avg: 😎 -0.2335 ± +5.5938 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +4.0372 ± +4.0372 📈 | D_fake_avg: 😵 +1.0575 ± +6.3287 📉
# [128x128][Epoch 10/40] DLoss(Avg): -2.6443 | GLoss(Avg): -0.2131 | IS: 2.6315 ± 0.0637) | FID: 121.86 | GP[avg]: 0.07
# [128x128][Epoch 11/40 | Iter: 2543/5087] Disc Loss: -2.8691 | Gen Loss: 0.2289
#  -- 😱 Batch-2543:  D_real_avg: 😎 +8.2346 ± +2.8506 📈| D_fake_avg: 😵 +4.9681 ± +4.2211 📉
# Using FID stats for celeba-train_10K from cache...
#  -- ‼️ Last Batch : D_real_avg: 😎 +5.9909 ± +3.9288 📈 | D_fake_avg: 😵 +6.4053 ± +5.7140 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +4.0063 ± +4.0063 📈 | D_fake_avg: 😵 +1.0243 ± +6.0501 📉
# [128x128][Epoch 11/40] DLoss(Avg): -2.6467 | GLoss(Avg): -0.1964 | IS: 2.7597 ± 0.0600) | FID: 118.19 | GP[avg]: 0.07
# [128x128][Epoch 12/40 | Iter: 2543/5087] Disc Loss: -3.9330 | Gen Loss: 2.1010
#  -- 😀 Batch-2543:  D_real_avg: 😵 -2.6460 ± +2.9215 📈| D_fake_avg: 😎 -6.8104 ± +5.4011 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😱 Last Batch : D_real_avg: 😎 +10.1640 ± +3.0329 📈 | D_fake_avg: 😵 +5.3806 ± +3.3514 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +3.9215 ± +3.9215 📈 | D_fake_avg: 😵 +0.9106 ± +6.1020 📉
# [128x128][Epoch 12/40] DLoss(Avg): -2.6732 | GLoss(Avg): -0.1075 | IS: 2.7729 ± 0.0334) | FID: 120.86 | GP[avg]: 0.07
# [128x128][Epoch 13/40 | Iter: 2543/5087] Disc Loss: -3.2112 | Gen Loss: -4.3590
#  -- 😱 Batch-2543:  D_real_avg: 😎 +5.0430 ± +5.2885 📈| D_fake_avg: 😵 +1.3084 ± +3.1120 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😱 Last Batch : D_real_avg: 😎 +9.1091 ± +5.1502 📈 | D_fake_avg: 😵 +5.8672 ± +4.8443 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +3.9793 ± +3.9793 📈 | D_fake_avg: 😵 +0.9614 ± +6.0594 📉
# [128x128][Epoch 13/40] DLoss(Avg): -2.6794 | GLoss(Avg): -0.1567 | IS: 2.8800 ± 0.0699) | FID: 125.72 | GP[avg]: 0.07
# [128x128][Epoch 14/40 | Iter: 2543/5087] Disc Loss: -2.1034 | Gen Loss: 0.8310
#  -- 😱 Batch-2543:  D_real_avg: 😎 +8.4638 ± +2.6837 📈| D_fake_avg: 😵 +5.9544 ± +3.2493 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😀 Last Batch : D_real_avg: 😎 +2.9446 ± +4.2682 📈 | D_fake_avg: 😎 -0.2689 ± +3.3653 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +4.0210 ± +4.0210 📈 | D_fake_avg: 😵 +1.0216 ± +5.9901 📉
# [128x128][Epoch 14/40] DLoss(Avg): -2.6648 | GLoss(Avg): -0.2143 | IS: 2.7318 ± 0.0354) | FID: 120.37 | GP[avg]: 0.07
# [128x128][Epoch 15/40 | Iter: 2543/5087] Disc Loss: -3.8334 | Gen Loss: -3.9069
#  -- 😱 Batch-2543:  D_real_avg: 😎 +6.4906 ± +4.3235 📈| D_fake_avg: 😵 +2.3719 ± +2.9386 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😱 Last Batch : D_real_avg: 😎 +6.6116 ± +3.5445 📈 | D_fake_avg: 😵 +2.0624 ± +3.4311 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +4.0307 ± +4.0307 📈 | D_fake_avg: 😵 +1.0008 ± +5.4763 📉
# [128x128][Epoch 15/40] DLoss(Avg): -2.6912 | GLoss(Avg): -0.2137 | IS: 2.9563 ± 0.0519) | FID: 122.89 | GP[avg]: 0.07
# [128x128][Epoch 16/40 | Iter: 2543/5087] Disc Loss: -1.5929 | Gen Loss: 5.7114
#  -- 😀 Batch-2543:  D_real_avg: 😵 -3.7380 ± +3.8466 📈| D_fake_avg: 😎 -5.5356 ± +3.8074 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😱 Last Batch : D_real_avg: 😎 +8.2828 ± +4.3467 📈 | D_fake_avg: 😵 +5.5052 ± +4.7894 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +4.0216 ± +4.0216 📈 | D_fake_avg: 😵 +0.9783 ± +5.9762 📉
# [128x128][Epoch 16/40] DLoss(Avg): -2.7031 | GLoss(Avg): -0.1704 | IS: 2.9255 ± 0.0771) | FID: 117.45 | GP[avg]: 0.07
# [128x128][Epoch 17/40 | Iter: 2543/5087] Disc Loss: -4.0397 | Gen Loss: 6.4271
#  -- 😀 Batch-2543:  D_real_avg: 😵 -1.3896 ± +5.2689 📈| D_fake_avg: 😎 -5.7682 ± +3.0744 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😱 Last Batch : D_real_avg: 😎 +6.7212 ± +2.6733 📈 | D_fake_avg: 😵 +3.5877 ± +2.3812 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +3.8895 ± +3.8895 📈 | D_fake_avg: 😵 +0.8370 ± +5.9780 📉
# [128x128][Epoch 17/40] DLoss(Avg): -2.7141 | GLoss(Avg): -0.0819 | IS: 2.9992 ± 0.0678) | FID: 123.45 | GP[avg]: 0.07
# [128x128][Epoch 18/40 | Iter: 2543/5087] Disc Loss: -1.0116 | Gen Loss: 1.0766
#  -- 😱 Batch-2543:  D_real_avg: 😎 +5.6020 ± +3.7023 📈| D_fake_avg: 😵 +4.2588 ± +3.4826 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😀 Last Batch : D_real_avg: 😵 -0.2922 ± +3.8711 📈 | D_fake_avg: 😎 -3.0014 ± +2.0987 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +3.9259 ± +3.9259 📈 | D_fake_avg: 😵 +0.8646 ± +5.7461 📉
# [128x128][Epoch 18/40] DLoss(Avg): -2.7241 | GLoss(Avg): -0.0897 | IS: 2.8871 ± 0.0677) | FID: 117.72 | GP[avg]: 0.07
# [128x128][Epoch 19/40 | Iter: 2543/5087] Disc Loss: -2.4476 | Gen Loss: 2.5083
#  -- 😱 Batch-2543:  D_real_avg: 😎 +4.1571 ± +3.5606 📈| D_fake_avg: 😵 +1.4771 ± +3.0439 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😱 Last Batch : D_real_avg: 😎 +5.9212 ± +3.9744 📈 | D_fake_avg: 😵 +2.7760 ± +3.4972 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +3.8830 ± +3.8830 📈 | D_fake_avg: 😵 +0.8438 ± +5.8593 📉
# [128x128][Epoch 19/40] DLoss(Avg): -2.7011 | GLoss(Avg): -0.0473 | IS: 2.8862 ± 0.0682) | FID: 124.77 | GP[avg]: 0.07
# [128x128][Epoch 20/40 | Iter: 2543/5087] Disc Loss: -3.6148 | Gen Loss: -9.9046
#  -- 😱 Batch-2543:  D_real_avg: 😎 +12.1721 ± +4.7174 📈| D_fake_avg: 😵 +8.2988 ± +3.6505 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😀 Last Batch : D_real_avg: 😟 +0.6579 ± +2.5957 📈 | D_fake_avg: 😎 -2.4760 ± +2.7168 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +3.8727 ± +3.8727 📈 | D_fake_avg: 😵 +0.8236 ± +5.7697 📉
# [128x128][Epoch 20/40] DLoss(Avg): -2.7054 | GLoss(Avg): -0.0592 | IS: 2.7446 ± 0.0899) | FID: 117.31 | GP[avg]: 0.07
# [128x128][Epoch 21/40 | Iter: 2543/5087] Disc Loss: -2.7524 | Gen Loss: 11.1109
#  -- 😀 Batch-2543:  D_real_avg: 😵 -0.9532 ± +2.4933 📈| D_fake_avg: 😎 -3.9435 ± +4.4649 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😱 Last Batch : D_real_avg: 😎 +11.0629 ± +6.0542 📈 | D_fake_avg: 😵 +5.6241 ± +3.3537 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +3.9204 ± +3.9204 📈 | D_fake_avg: 😵 +0.8497 ± +5.4859 📉
# [128x128][Epoch 21/40] DLoss(Avg): -2.7235 | GLoss(Avg): -0.0828 | IS: 2.7281 ± 0.0420) | FID: 125.72 | GP[avg]: 0.07
# [128x128][Epoch 22/40 | Iter: 2543/5087] Disc Loss: -3.0803 | Gen Loss: -1.8272
#  -- 😱 Batch-2543:  D_real_avg: 😎 +6.3588 ± +5.7723 📈| D_fake_avg: 😵 +2.9853 ± +2.8228 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😱 Last Batch : D_real_avg: 😎 +7.3246 ± +2.8110 📈 | D_fake_avg: 😵 +5.1916 ± +3.0450 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +3.9530 ± +3.9530 📈 | D_fake_avg: 😵 +0.8831 ± +5.5138 📉
# [128x128][Epoch 22/40] DLoss(Avg): -2.7221 | GLoss(Avg): -0.1331 | IS: 2.7190 ± 0.0732) | FID: 128.88 | GP[avg]: 0.07
# [128x128][Epoch 23/40 | Iter: 2543/5087] Disc Loss: -2.9533 | Gen Loss: 1.8466
#  -- 😀 Batch-2543:  D_real_avg: 😎 +2.2108 ± +2.2627 📈| D_fake_avg: 😎 -0.9476 ± +2.7863 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😀 Last Batch : D_real_avg: 😵 -1.2684 ± +6.5055 📈 | D_fake_avg: 😎 -3.4393 ± +4.3539 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +3.9451 ± +3.9451 📈 | D_fake_avg: 😵 +0.8694 ± +5.4191 📉
# [128x128][Epoch 23/40] DLoss(Avg): -2.7284 | GLoss(Avg): -0.1210 | IS: 2.8747 ± 0.0722) | FID: 126.05 | GP[avg]: 0.07
# [128x128][Epoch 24/40 | Iter: 2543/5087] Disc Loss: -2.8412 | Gen Loss: 4.4408
#  -- 😀 Batch-2543:  D_real_avg: 😰 +0.2728 ± +4.0992 📈| D_fake_avg: 😎 -2.8013 ± +3.8053 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😀 Last Batch : D_real_avg: 😎 +1.9038 ± +5.7948 📈 | D_fake_avg: 😎 -0.5312 ± +3.5215 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +3.9521 ± +3.9521 📈 | D_fake_avg: 😵 +0.8638 ± +5.4361 📉
# [128x128][Epoch 24/40] DLoss(Avg): -2.7391 | GLoss(Avg): -0.1180 | IS: 2.7205 ± 0.0636) | FID: 110.63 | GP[avg]: 0.07
# [128x128][Epoch 25/40 | Iter: 2543/5087] Disc Loss: -2.2267 | Gen Loss: -6.6081
#  -- 😱 Batch-2543:  D_real_avg: 😎 +12.5836 ± +4.3913 📈| D_fake_avg: 😵 +9.8933 ± +4.2538 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😀 Last Batch : D_real_avg: 😵 -3.7671 ± +3.2930 📈 | D_fake_avg: 😎 -9.9924 ± +4.3833 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +3.8964 ± +3.8964 📈 | D_fake_avg: 😵 +0.8228 ± +5.8011 📉
# [128x128][Epoch 25/40] DLoss(Avg): -2.7269 | GLoss(Avg): -0.0676 | IS: 2.8337 ± 0.0646) | FID: 110.79 | GP[avg]: 0.07
# [128x128][Epoch 26/40 | Iter: 2543/5087] Disc Loss: -1.6448 | Gen Loss: 1.6342
#  -- 😀 Batch-2543:  D_real_avg: 😵 -2.0032 ± +3.5273 📈| D_fake_avg: 😎 -3.9701 ± +2.8513 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😱 Last Batch : D_real_avg: 😎 +8.2374 ± +4.5530 📈 | D_fake_avg: 😵 +6.1964 ± +3.9273 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +3.8230 ± +3.8230 📈 | D_fake_avg: 😵 +0.7484 ± +5.6307 📉
# [128x128][Epoch 26/40] DLoss(Avg): -2.7269 | GLoss(Avg): -0.0257 | IS: 2.9005 ± 0.1094) | FID: 123.26 | GP[avg]: 0.07
# [128x128][Epoch 27/40 | Iter: 2543/5087] Disc Loss: -1.4830 | Gen Loss: -11.1008
#  -- 😱 Batch-2543:  D_real_avg: 😎 +28.0224 ± +11.7509 📈| D_fake_avg: 😵 +25.9262 ± +8.6641 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😀 Last Batch : D_real_avg: 😎 +2.2418 ± +2.9054 📈 | D_fake_avg: 😎 -1.4668 ± +3.3335 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +3.9504 ± +3.9504 📈 | D_fake_avg: 😵 +0.8478 ± +5.6885 📉
# [128x128][Epoch 27/40] DLoss(Avg): -2.7534 | GLoss(Avg): -0.1196 | IS: 2.7332 ± 0.0457) | FID: 125.22 | GP[avg]: 0.07
# [128x128][Epoch 28/40 | Iter: 2543/5087] Disc Loss: -4.1584 | Gen Loss: 1.1575
#  -- 😱 Batch-2543:  D_real_avg: 😎 +5.3082 ± +4.9732 📈| D_fake_avg: 😵 +0.8128 ± +3.8049 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😱 Last Batch : D_real_avg: 😎 +7.5551 ± +4.9080 📈 | D_fake_avg: 😵 +2.3310 ± +4.5039 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +3.9363 ± +3.9363 📈 | D_fake_avg: 😵 +0.8210 ± +5.4625 📉
# [128x128][Epoch 28/40] DLoss(Avg): -2.7633 | GLoss(Avg): -0.1154 | IS: 2.8380 ± 0.0742) | FID: 117.33 | GP[avg]: 0.07
# [128x128][Epoch 29/40 | Iter: 2543/5087] Disc Loss: -1.6728 | Gen Loss: -2.3600
#  -- 😀 Batch-2543:  D_real_avg: 😵 -1.9460 ± +4.5955 📈| D_fake_avg: 😎 -3.9038 ± +3.0960 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😀 Last Batch : D_real_avg: 😵 -1.3679 ± +4.8526 📈 | D_fake_avg: 😎 -5.6232 ± +5.6918 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +4.0107 ± +4.0107 📈 | D_fake_avg: 😵 +0.8988 ± +5.4643 📉
# [128x128][Epoch 29/40] DLoss(Avg): -2.7609 | GLoss(Avg): -0.1790 | IS: 2.7662 ± 0.0694) | FID: 119.43 | GP[avg]: 0.07
# [128x128][Epoch 30/40 | Iter: 2543/5087] Disc Loss: -5.0278 | Gen Loss: 15.3333
#  -- 😀 Batch-2543:  D_real_avg: 😵 -4.5254 ± +4.8401 📈| D_fake_avg: 😎 -9.8075 ± +6.1109 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😱 Last Batch : D_real_avg: 😎 +4.1270 ± +4.6156 📈 | D_fake_avg: 😟 +0.2314 ± +1.8921 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +4.0362 ± +4.0362 📈 | D_fake_avg: 😵 +0.8894 ± +5.4733 📉
# [128x128][Epoch 30/40] DLoss(Avg): -2.7897 | GLoss(Avg): -0.1893 | IS: 2.9480 ± 0.0658) | FID: 109.34 | GP[avg]: 0.07
# [128x128][Epoch 31/40 | Iter: 2543/5087] Disc Loss: -2.4112 | Gen Loss: 2.7771
#  -- 😱 Batch-2543:  D_real_avg: 😎 +6.0828 ± +4.7530 📈| D_fake_avg: 😵 +3.3677 ± +5.5942 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😱 Last Batch : D_real_avg: 😎 +3.4503 ± +2.5071 📈 | D_fake_avg: 😵 +0.4274 ± +4.1579 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +4.0863 ± +4.0863 📈 | D_fake_avg: 😵 +0.9405 ± +5.3881 📉
# [128x128][Epoch 31/40] DLoss(Avg): -2.7873 | GLoss(Avg): -0.2366 | IS: 2.7932 ± 0.0628) | FID: 119.84 | GP[avg]: 0.07
# [128x128][Epoch 32/40 | Iter: 2543/5087] Disc Loss: -1.6677 | Gen Loss: 9.4166
#  -- 😀 Batch-2543:  D_real_avg: 😎 +1.2541 ± +3.9665 📈| D_fake_avg: 😎 -0.7563 ± +3.6081 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😱 Last Batch : D_real_avg: 😎 +12.2794 ± +5.1852 📈 | D_fake_avg: 😵 +9.5969 ± +5.7240 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +4.1613 ± +4.1613 📈 | D_fake_avg: 😵 +1.0002 ± +5.7857 📉
# [128x128][Epoch 32/40] DLoss(Avg): -2.8017 | GLoss(Avg): -0.3208 | IS: 2.8081 ± 0.0421) | FID: 124.01 | GP[avg]: 0.07
# [128x128][Epoch 33/40 | Iter: 2543/5087] Disc Loss: -3.0821 | Gen Loss: -4.8644
#  -- 😱 Batch-2543:  D_real_avg: 😎 +8.1394 ± +3.1508 📈| D_fake_avg: 😵 +4.7019 ± +3.0339 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😱 Last Batch : D_real_avg: 😎 +8.1370 ± +4.7789 📈 | D_fake_avg: 😵 +6.6281 ± +4.1563 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +4.2458 ± +4.2458 📈 | D_fake_avg: 😵 +1.0730 ± +5.4149 📉
# [128x128][Epoch 33/40] DLoss(Avg): -2.8102 | GLoss(Avg): -0.3933 | IS: 2.8083 ± 0.0647) | FID: 117.48 | GP[avg]: 0.07
# [128x128][Epoch 34/40 | Iter: 2543/5087] Disc Loss: -1.4611 | Gen Loss: 1.9477
#  -- 😱 Batch-2543:  D_real_avg: 😎 +4.3782 ± +5.4579 📈| D_fake_avg: 😵 +2.3972 ± +5.5922 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😱 Last Batch : D_real_avg: 😎 +4.4003 ± +4.4851 📈 | D_fake_avg: 😵 +1.6388 ± +4.4123 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +4.3633 ± +4.3633 📈 | D_fake_avg: 😵 +1.2118 ± +5.3674 📉
# [128x128][Epoch 34/40] DLoss(Avg): -2.7929 | GLoss(Avg): -0.4969 | IS: 2.7895 ± 0.0603) | FID: 120.64 | GP[avg]: 0.07
# [128x128][Epoch 35/40 | Iter: 2543/5087] Disc Loss: -3.5766 | Gen Loss: -3.4147
#  -- 😱 Batch-2543:  D_real_avg: 😎 +5.8128 ± +4.3729 📈| D_fake_avg: 😵 +1.6266 ± +4.2631 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😱 Last Batch : D_real_avg: 😎 +7.6714 ± +3.3716 📈 | D_fake_avg: 😵 +5.1147 ± +2.9043 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +4.3407 ± +4.3407 📈 | D_fake_avg: 😵 +1.1714 ± +5.2483 📉
# [128x128][Epoch 35/40] DLoss(Avg): -2.8066 | GLoss(Avg): -0.4731 | IS: 3.0997 ± 0.0579) | FID: 114.40 | GP[avg]: 0.07
# [128x128][Epoch 36/40 | Iter: 2543/5087] Disc Loss: -0.9440 | Gen Loss: 0.5591
#  -- 😀 Batch-2543:  D_real_avg: 😵 -0.0762 ± +4.1351 📈| D_fake_avg: 😎 -1.4109 ± +3.2882 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😀 Last Batch : D_real_avg: 😵 -8.1008 ± +2.9726 📈 | D_fake_avg: 😎 -11.9384 ± +3.1537 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +4.3803 ± +4.3803 📈 | D_fake_avg: 😵 +1.2148 ± +5.4842 📉
# [128x128][Epoch 36/40] DLoss(Avg): -2.8031 | GLoss(Avg): -0.5266 | IS: 2.7153 ± 0.0499) | FID: 115.62 | GP[avg]: 0.07
# [128x128][Epoch 37/40 | Iter: 2543/5087] Disc Loss: -3.9054 | Gen Loss: -1.3970
#  -- 😱 Batch-2543:  D_real_avg: 😎 +4.6845 ± +3.7527 📈| D_fake_avg: 😵 +0.4726 ± +3.5772 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😀 Last Batch : D_real_avg: 😵 -1.4074 ± +3.4660 📈 | D_fake_avg: 😎 -5.1017 ± +5.1334 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +4.4133 ± +4.4133 📈 | D_fake_avg: 😵 +1.2356 ± +5.1814 📉
# [128x128][Epoch 37/40] DLoss(Avg): -2.8145 | GLoss(Avg): -0.5629 | IS: 2.8299 ± 0.0650) | FID: 120.05 | GP[avg]: 0.07
# [128x128][Epoch 38/40 | Iter: 2543/5087] Disc Loss: -5.2086 | Gen Loss: -1.5967
#  -- 😱 Batch-2543:  D_real_avg: 😎 +6.3489 ± +2.8745 📈| D_fake_avg: 😵 +0.7870 ± +2.4554 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😱 Last Batch : D_real_avg: 😎 +12.1366 ± +4.1341 📈 | D_fake_avg: 😵 +8.1835 ± +2.7635 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +4.4379 ± +4.4379 📈 | D_fake_avg: 😵 +1.2347 ± +5.4123 📉
# [128x128][Epoch 38/40] DLoss(Avg): -2.8344 | GLoss(Avg): -0.5694 | IS: 2.7458 ± 0.0426) | FID: 116.01 | GP[avg]: 0.07
# [128x128][Epoch 39/40 | Iter: 2543/5087] Disc Loss: -4.1290 | Gen Loss: 3.0675
#  -- 😀 Batch-2543:  D_real_avg: 😎 +4.0782 ± +3.1308 📈| D_fake_avg: 😎 -0.3996 ± +3.9610 📉
# Using FID stats for celeba-train_10K from cache...
#  -- 😀 Last Batch : D_real_avg: 😎 +2.4379 ± +3.7642 📈 | D_fake_avg: 😎 -0.0778 ± +3.5729 📉
#  -- 😱 Epoch's Avg: D_real_avg: 😎 +4.3907 ± +4.3907 📈 | D_fake_avg: 😵 +1.2159 ± +5.3316 📉
# [128x128][Epoch 39/40] DLoss(Avg): -2.8113 | GLoss(Avg): -0.5210 | IS: 2.8186 ± 0.0480) | FID: 117.65 | GP[avg]: 0.07
# Files already downloaded and verified
#  Step: 6/7 -> Training on [256x256]
#   --Epochs:                      40 
#   --BatchSize:                   16 
#   --Number of Batches:           10174 
#   --Interval:                    5088 
#   --Fade-in Steps:               203480 
#   --Last training Step taken:    0 
#   --Current Discriminator LR:    0.001
#   --Current Generator LR:        0.001
#   --Current Discriminator Betas: [0, 0.99]
#   --Current Generator Betas:     [0, 0.99]
 
#%%
# now lets load the last experiment and see the result
device = "cuda" if torch.cuda.is_available() else "cpu"
# cifar10(the fid/is is garbage you need to calculate it again if you like)
# checkpoint_path ='./weights/gan/progan_cifar10_wgangp_20251003124237/checkpoint_step_5_20251003124237.ckpt'
# celeba
# checkpoint_path ='./weights/gan/progan_celeba_wgangp_20250927174948/checkpoint_step_3_20250927174948.ckpt'
# checkpoint_path ='./weights/gan/progan_celeba_wgangp_20250929080324/checkpoint_step_3_20250929080324.ckpt'
# checkpoint_path ='./weights/gan/progan_celeba_wgangp_20250929131133/checkpoint_step_4_20250929131133.ckpt'
# good models using high lr
# 64x64
# checkpoint_path ='./weights/gan/progan_celeba_wgangp_20250930091608/checkpoint_step_4_20250930091608.ckpt'
#128x128
checkpoint_path ='./weights/gan/progan_celeba_wgangp_20250930091608/checkpoint_step_5_20250930091608.ckpt'

checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
# print(checkpoint.keys())
last_step = checkpoint["step"]
all_training_losses = checkpoint.pop("all_training_losses")
all_gradient_penalties = checkpoint.pop("all_gradient_penalties")

generator_pgan = GeneratorProGAN(checkpoint["z_size"], checkpoint["max_steps"])
generator_pgan.load_state_dict(checkpoint["gen_state_dict"])
generator_pgan = generator_pgan.eval()
generator_pgan = generator_pgan.to(device)

for k,v in checkpoint.items():
    if not isinstance(v,dict):
        print(f'{k:<15} {v}')
    elif "param_groups" in v.keys():
        print(f'{k:<15} {v["param_groups"]}')

#%%
# lets generator some images
num_samples=36
z = torch.randn((num_samples, generator_pgan.z_size),device=device)
dim = 2**(last_step+1)
res=f"{dim}x{dim}"
with torch.no_grad():
    imgs = generator_pgan(z, alpha=1, step=last_step)
    display_images(imgs, 
                   cols=6, 
                   title=f'Step {last_step} [{res}]',
                   unnormalize=True, 
                   figsize=(16,8))

#%%
# good now lets implement stylegan architectures.
# the authors of progan introduced stylegan in 2019 and followed it up with the second and third
# versions in the following years each improving upon the previous one and becoming the sota of
# image geenration using GANs (until diffusion models were introduced later in 2021/2022)
# we start off with stylegan 1 and see what the changes/novelities were and then go to implement
# it

# StyleGan1: https://arxiv.org/pdf/1812.04948
# The first StyleGAN paper(A Style-Based Generator Architecture for Generative Adversarial Networks)
# was introduced to address one of the biggest issues of PROGAN architecture.
# the main issues of the progan architecture is that as long as we want unconditional images, it 
# works very good. however, when we try to go conditional and control the features/styles, it becomes
# very hard. PROGAN doesnt do a good job in feature disentanglement (it has high feature entanglment!)
# so we cant have finegrained control in this regard.
# StyleGAN came to address this issue by doing a much better job at feature disentanglement!
# previously we would simply start our generator by an input latent vector(z) and then gradually upsample
# it to get the final image. the latent vector z had to use the simple/fixed gaussian distribution, 
# this all changes in stylegan! in stylegan, we now first create a middle representation from the 
# input latent vector and then feed the resulting vector that to the generator.
# that is, we make our input latent vector z go thtough a nonlinear mapping (i.e. feed it to an MLP) 
# to get a new representation and use this new latent vector w instead in our generator.
# 
# by doing this our new latent vector w will be no longer constarined like z, so it can learn the actual 
# complex/nonlinear distribution of image features. the new latent space(W) therefore will
# be much more disentangled than the latent space Z(gaussian). 
# this means its elements/components will represent different distinct features/attributes much better
# and corospond more cleanly to different attributes (i.e. e.g. one part controls pose, the other
# controls the hair textures, the next one color, etc)
# 
# sidenote:
# the reason we do this is not that w learns the actual distribution of training image features really,
# rather, when we apply the nonlinear mapping, the mlp can choose a distribution for w that aligns well
# with what we want, i.e. making generating data easier and more disenangled!
# in otherwords, this allows the network to learn a mapping (from Z) to a new space (W) thats unwarped
# so to speak (compared to our initial fixed gaussian distribution) and is therefore much better for 
# representing different factors or elements of variantions in our training data)
# 
# because of these changes, the authors decided to call the generator, the synthesis network, because it 
# now starts with a learned constant tensor(as if its a blank canvas! a tensor of 1s e.g.!) instead of
# a random latent vector. (note we dont feed the w as input to generator) all the information about 
# an image is injected at each layer,(that is these operations occure in each block) and the 
# latent vector w will also be used to extract the styles from.
# basically we feed w to two linear layers to get two separate components, the scale and bias(basically std and mean!)
# with which we direct the generation process towards the style we want.
# This happens in the AdaIN(adaptive instance normalization) layer, we get the scale/bias and add them 
# to each featuremap at each level. AdaIN normalizes each featuremap to have zero mean and unit variance(σ=1) (basically it 
# removes the current style in the image) so we can then use the new scale(std)/bias(mean) from w to 
# modulate/style the normalized featuremap (i.e direct the generation process by enforcing our specific
# scale/bias. I used std/mean, because effectively we are applying them instead of the previous std/mean
# if you recall our autoencoder chapter, we did a similar thing back then to get specific styles as well!).
# 
# sidenote2:
# in literature people use modulate/style when applying scale/bias, I initially used normalize, but since
# it might give an incorrect meaning, I chose to use module/style as well. normalize or normalization is
# usually used to convey we're transforming or forcing the data into a standard state like mean=0/std=1, but here
# we arent doing that, so calling it normalizing may not be correct thats why I guess people tend to call 
# this kind of normalization, modulation/styling instead!(the paper itself calls it "style" in section 2
# where it details the process, and says they called it so because previous works used similar networks to 
# do style transfer. in official tf implementation though they call it modulation (style modulation))
#
# (quicknote: it seems people in electronics/signal processing refer to the expressions such as
# "scale*input + bias" as modulation! so thats why they decided to call this modulation! also if
# we imagine the input as a canvas that the network is trying to draw something on, in thatcase
# normalizing it wouldnt make sense, rather styling it would make much more sense hence why some
# called it style!(honestly its a personal thing, cuz if you plan on using analogies, normalizing the
# first time shouldnt be called normalizing but reseting canvas! so I guess modulate is a better
# technical term and style is a better analogy, and normalize would be mathimatically correct because
# thats what we are doing essentially, but it'd be confusing nonetheless because of what we just
# discussed! so modulate it is!))
#
# this way we apply a global style in a scale specific manner, so for example at lower resolutions 
# (i.e. 4-8) it controls the high level features like face shapes and pose, basically corase features,
# while at mid resolutions (i.e. 16-32) it controls the mid level features like facial shape and style
# and at high resolutions like 64 to 1024, it controls the fine details in the image like colors(hair/eye,lighting),
# and other microstructures (i.e. skin textures,etc). I used facial features because prgan was mainly
# trained on celeba, and its more intuitive (it was trained on other datasets as well like lsun/cifar10
# but with celeba its much easier for analogies and examples)
# 
# in addition to that, to introduce random variations(Stochastic variation as the paper calls it),
# in the image like hair placement, freckles, skin pores, etc, the authors decided to inject guassion
# noise to each featuremap aswell. 
# this was done to prevent the network from generating psuedorandom patterns from determinstic 
# inputs and therefore reduce artifacts like repition that is seen in other GAN architectures.
# (instead of leting the network try to insert such small variations into the w as well, and 
# hence waste its capacity on learning them and make all of this process harder than it is! 
# they decided to do this so the network has an easier time doing its job! and it immediatly shows! 
# the paper explains it better than me, 
# from section 3.2. Stochastic variation: 
# "...Let us consider how a traditional generator implements stochastic variation. 
# Given that the only input to the network is through the input layer, the network needs to
# invent a way to generate spatially-varying pseudorandom numbers from earlier activations 
# whenever they are needed. This consumes network capacity and hiding the periodicity of
# generated signal is difficult — and not always successful, as evidenced by commonly seen
# repetitive patterns in generated images. Our architecture sidesteps these issues altogether
# by adding per-pixel noise after each convolution."
# from Figure 5:
# "Figure 5. Effect of noise inputs at different layers of our generator. 
# (a) Noise is applied to all layers. (b) No noise. (c) Noise infine layers only (64² – 1024²)
# (d) Noise in coarse layers only(4² – 32²)
# We can see that the artificial omission of noise leads to featureless “painterly” look.
# Coarse noise causes large-scale curling of hair and appearance of larger background features,
# while the fine noise brings out the finer curls of hair, finer background detail, and
# skin pores." read the paper now, this is beautifully shown in page 5) 
#
# also relevant read
# from paper - section 3.3 Separation of global effects from stochasticity:
# "... while changes to the style have global effects
# (changing pose, identity, etc.), the noise affects only
# inconsequential stochastic variation (differently combed
# hair, beard, etc.). This observation is in line with style transfer 
# literature, where it has been established that spatially
# invariant statistics (Gram matrix, channel-wise mean, variance, etc.)
# reliably encode the style of an image [20, 39]
# while spatially varying features encode a specific instance.
# In our style-based generator, the style affects the entire
# image because complete feature maps are scaled and biased with the same values. 
# Therefore, global effects such as pose, lighting, or background style 
# can be controlled coherently. Meanwhile, the noise is added independently to
# each pixel and is thus ideally suited for controlling stochastic variation. 
# If the network tried to control, e.g., pose using
# the noise, that would lead to spatially inconsistent decisions
# that would then be penalized by the discriminator. Thus the
# network learns to use the global and local channels appropriately,
# without explicit guidance."
#
# another notable trick the authors used in the paper was to use a technique called style mixing or 
# mixing regularization, in which during training they mix two latent vectors w1,w2 with a probablity (50%-90%) 
# so two styles are swapped/switched at a random point! this was done so the localized style effects 
# are encouraged(encourage stylize to localize) and also it decorolates adjacent scales and improves the model(that is if
# there are certain images with a certain shapes in corase stage, and certain hair placement is finer stage
# swithicng like that, forces the network to comeup with new styles! and not just those coupled from
# the same w. (one w has black hair, while the other has another color e.g. pink this makes the network
# to create more diverse combinations of features found in the training data))
# 
# From paper:
# "... a given percentage of images are generated using two random latent codes instead of one
# during training. When generating such an image, we simply switch from one latent code to another
# — an operation we refer to as style mixing— at a randomly selected point in the synthesis network.
# To be specific, we run two latent codes z1, z2 through the mapping network, and have the corresponding
# w1, w2 control the styles so that w1 applies before the crossover point and w2 after it. 
# This regularization technique prevents the network from assuming that adjacent styles are correlated."
# 
# Figure 3 page 5:
# "Figure 3. Two sets of images were generated from their respective latent codes (sources A and B);
# the rest of the images were generated by copying a specified subset of styles from source B and
# taking the rest from source A. 
# Copying the styles corresponding to coarse spatial resolutions (4²– 8²) brings high-level aspects
# such as pose, general hair style, face shape, and eyeglasses from source B, while all colors
# (eyes, hair, lighting) and finer facial features resemble A. 
# If we instead copy the styles of middle resolutions (16² – 32²) from B, we inherit smaller scale
# facial features, hair style, eyes open/closed from B, while the pose, general face shape, and 
# eyeglasses from A are preserved. 
# Finally, copying the fine styles (64² – 1024²) from B brings mainly the color scheme and microstructure."
# see the figure in the paper.
#
# another trick they used was that they found out if they sample w closer to the average population
# they get higher quality, more typical images, at the cost of decreased variety! this is called truncation trick!
# 
# aside from the tricks, loss also changes in StyleGAN1. the authors did use WGAN-GP loss in one
# of the experiments (CelebA dataset, configuration A in table1), but for the rest of the experiments
# that provide the best results, they used a different loss. they used non-saturating logistic loss which
# is a variation of the original GAN loss, a long with a new R1 regularizer that replaces WGAN-GP.
# that is for the discriminator maximize the D(real) and minimize the D(G(z)) and for the generator 
# maximize the D(G(z)). in practice this is implemented using the sofplus function which is a 
# smooth approximation of relu (i.e. log(1+exp(x)))
# so the discriminator loss becomes: D_loss = E[softplus(-D(real)) + softplus(D(G(z)))] 
# (the first term(softplus(-D(real))) gets smaller as D(real) becomes more positive(i.e. accuratly identifies real images)
# and the second term becomes smaller as D(G(z)) becomes more negative (i.e. correctly identifies fake images)
# and the generator: G_loss = E[softplus(-D(G(z)))] it becomes smaller as D(G(z)) becomes more possitive
# (i.e. the generator fools the discriminator well)
# 
# quicknote: E is Expectation here(i.e. average over the batch)
#
# now for the regularizer part, since WGAN-GP calculates gradients on many random points
# between the real and fake image distributionsthe its computationally expeisnve and
# it can also be unstable sometimes therefore the authors tried something simpler. 
# they decided, instead of calculating the gradients on many random points like before,
# only calculate and penalize the discriminators gradients with respect to its inputs (real images!)
# the intuition behind it is the same as wgangp/1lipshitz condition, i.e. the discriminator
# shouldnt be too sensitve to small changes in input. if the gradient of the discriminators
# output with respect to its input is very large, then it means the discriminator's
# decision surface is very spiky/sharp(lots of high peaks/deep valleys). this is not good
# because it provides unstable/unhelpful gradients to the generator. we want a smooth surface
# so the trainig is more stable. this method helps us achieve that.
# 
# the R1 formula therefore is :
# L_R1 = (gamma/2) * E[ || ∇D(x_real) ||² ]
# where (gamma/2) is the multiplier for specifying the penalty strength.
# the || ∇D(x_real) ||² is squared L2 norm(sum of squares) of the gradients of discriminator's
# output with respect to the inputs(real images). we add this penalty term to the discriminator's loss.
# so the discriminators loss becomes: 
# D_loss = E[softplus(-D(real)) + softplus(D(G(z)))] + (γ/2 * E[||∇D(x_real)||²])
# 
#quicknote: the gradient operator ∇, is called del
#quicknote2: ||.|| is l2norm! dont mistake it with ||.||² which is l2norm suqared!(obviously!)
# 
#
# sidenote:
# why do we write γ/2(gamma/2) and not just gamma or lambda without the fraction?
# its a multiplier after all and we can call it lambda(λ) or gamma(γ) or anything else and it'd just work as well!
# yes thats right, however the reason for using a different name like gamma and using 1/2 factor
# instead of just lambda alone like we did back in wgan loss section, is because having 1/2 makes 
# gradient calculation much easier and simpler. I used gamma because the paper calls it gamma and
# the distinction is important as well.
# when we calculate the gradients, this 1/2 will cancel out the squared gradient norm power(the 2)
# (2 in g² or ||∇D(x_real)||²) and make the gradient calculation clean and simple (mathimatically!) 
# that is, it'd become d/dg => 1/2*g² => 1/2*2g = g!
# thats it! so as a multiplier/scaler, we can call it lambda or any other name, and assign any 
# number to specify the strength of the actual penalty term(|| ∇D(x_real) ||²), 
# however for clearer gradient calculation (mathimatics/formula wise), calling it gamma/2,
# makes it clear that the 1/2 is there for cleaner gradient calculation, and the 
# when the value of gamma is reported as e.g. 10 in the paper, as the strength of penalty used in training, 
# everyone knows the actual effective strength is infact 5! 
# so those who simplify the formula and directly go for lambda*penalty term e.g. will be using
# twice the actual amount if they follow the paper details on training hyperpameters and not realize it!
# this is actually what I fell for in the progan section. so I thought I added this note here!
# (so we can use lambda simply as a scaler for penalty term, like l*penalty, like before, 
# to distinuish it from the gamma/2 and thus use different number accordinly, but it makes it 
# more confusing and made us make mistake before, so to be aligned with the paper and other 
# implementations I also use gamma/2 and dont simplify the formula so we dnt make any mistakes
# and can easily follow the paper and other implementations hyperparameter choices!)
#
# sidenote:
# as the paper states in table1, the FID score they report is the lowest score they could
# achieve during the whole training!(involving 50K of training images by the way) so this
# doesnt mean, the images they got at ihghest resolution were giivng in fact the lowest FID!
# we saw that in progan ourselevs that we could get to around 8 or 11 in 32x32 res, but not
# in the higher resolutions(because its more complex and takes time to get good) so yeah! just
# be aware of that
#
# reminder:
# if you havent read the paper yet, read it, then come back and continue
#
# these were all the changes the stylegan paper had compared to progan. 
# so all other progan related novelities are still valid, we can safely say stylegan1
# is an improved version of progan!
# having said all of this, lets now implemenet it and see how it works!

# add blur so discriminator doesnt quickly win! 
# we add it to generator to mask checkermarks
# as well especially in mid level resolutions 
# so the discriminator doesnt get ahead quickly
# this is also called as anti-aliasing in other
# implementations (the paper also calls it that)
class Blur(nn.Module):
    def __init__(self):
        super().__init__()
        # create a simple 3x3 bluring kernel
        kernel = torch.tensor([1,2,1])
        # by multiplying it this way we get 3x3 kernel we want
        # i.e. [1,2,1]
        #      [2,4,2]
        #      [1,2,1]
        kernel = kernel.view(1,-1) * kernel.view(-1,1)
        # normalize by 16 so we dont change the magnitude
        # of the featuremaps we apply our kernel on (the default
        # kernel we made adds 16 to each patch its applied on
        # i.e it increases the brightness by 16, by dividing it
        # by 16 we simply get the blur we want without messing
        # up the patch brightness!)
        #[1/16, 2/16, 1/16]
        #[2/16, 4/16, 2/16]
        #[1/16, 2/16, 1/16]
        kernel = kernel/kernel.sum()
        self.register_buffer("kernel", kernel.view(1,1,3,3))

    def forward(self, x):
        in_channels = x.size(1)
        kernel = self.kernel.repeat(in_channels, 1,1,1)
        # since the 3x3 conv natually shrinks the spatial size by 1, 
        # we add padding=1 to the
        # input itself so when we apply the kernel everything falls in
        # to its rightful place, we dont use the paddig in conv2d() as
        # because we want to pad it ourselves. conf2d always uses 0 to
        # pad, this creates a black edge around the input, we dont want
        # black (or white) pixels around the image, instead we want the
        # edge pixels of the image itself to make it more natural. so we
        # use F.pad with 'reflect' to have full control over this. we use
        # conv2d to just slide on the input and apply our kernel!
        x = F.pad(x,[1,1,1,1],mode='reflect')
        # apply the kernel using the conv operation
        return F.conv2d(x, kernel, groups=in_channels)

# sidenote:
# I initially tried sqrt(2/fan-in) for equalizedlinear, and bias was optional as well
# but I found out other implementations dont use the vanilla version like us, they
# do it a bit differently, they instead do (1/sqrt(fan_in))*lr_mult. the lr_mult is 
# there to be able to tune the lr dynamically during training and they usually set it
# to 0.01 for mapping_network layer(other times its set to 1).
# its both used in the scaler itself and is also applied on the bias! to make
# training more stable (so both weights and bias are scaled by lr_mult)
# todo: check and set proper lr_mult in the architecture, we need to use 0.01 in mapping 
# only and the rest should use lr_mult=1)
# when we do this we dont need to set a seprate learning rate for mapping network durng
# training. I did this mistake and this makes the mapping network 10000x slower and to
# fail in higher resolutions like 32x32 and larger.

class EqualizedLinear(nn.Linear):
    def __init__(self, in_features, out_features, lr_mult=1, device=None, dtype=None):
        super().__init__(in_features, out_features, bias=True, device=device, dtype=dtype) 
        self.lr_mult = lr_mult

        # initalize the weight
        # initially I did normal_(0,1) but this caused the outputs to become too small 
        # when we used lrmult=0.01 i.e. we effectively scaling the weights twice! once 
        # by 1/sqrt(fan_in) and a second time by lr_mult.(i.e. weights * 1/√fan_in * lr_mult)
        # so to preserve the layer's output variance, we need to first scale the weights up 
        # at initialization by 1/lr_mult, (i.e. normal(0,1/lr_mult) so that after multiplying
        # the weights by lr_mult at runtime, the effective scale doesnt change anymore.
        # (i.e. (weights/lr_mult) * (1/√fan_in * lr_mult))
        # 
        # sidenote :
        # linear layer is simply y=Wx. so the output variance is obviously affected by/directly dependant on
        # the input and weights! that is Var[y]=Var[Wx] which roughly equals fan_in*Var(Wx) or fan_in*Var[Wij]*Var[x]
        # and since Var[Wij]=1, then Var[y] = fan_in*Var[x]
        # as you can see this grows linearly with fan_in(i.e. number of input_features)! 
        # this is obviously very problematic for us since as the network gets deeper, if 
        # the variance gets larger, then activations get larger(will explode) and if it 
        # gets smaller, it will shrink the activations and therefore lead to either 
        # exploding or vanishing gradients! basiclly an unsatble signal propagation which
        # leads to slow convergence at best or failed training otherwise
        # 
        # thats why we scale the weights by 1/√fan_in so we get Var[y]=fan_in*(1/fan_in)*Var[x]=Var[x]
        # now the output variance equals the input variance. the kaminghe/xavier did this
        # only at initialization to normalize the variance of activations on all layers and
        # have a more stable training. later on BatchNorm was introduced to maintain actiations
        # mean and variance during training especially with larger lrs, but since stylegan doesnt use
        # BatchNorm or any other types of normalization layers, to achieve the same effect 
        # they tried the weight scaling dynamically at every forward pass! instead of just at 
        # initialization this way using different learning rates (especially larger ones) isnt
        # an issue anymore and everything works out well. (without this we had to maticulously
        # choose a small enough learning rate thats not too tiny to take ages to train or 
        # face vanishing gradients or be too large to face exploding gradient or just diverge)
        # quicknote:why not use batchnorm?
        # version 1: 
        # there are two main reasons for that. first generally speaking, since batchnorm
        # uses the batch stats(mean/variance) to normalize all samples! this introduces
        # noise and instability in training and makes the gradients batch dependant. 
        # a large batch behaves/performs differently than a smaller one. moreover, this
        # makes them dependenat on each other(each sample depends on the rest of the samples
        # in that batch!) this is problematic because now, when the generator uses latent
        # code z1 e.g. to create an image, the output is also depandant on whats in the
        # rest of the batch! this will result in inconsistent images that depend on the batch
        # and not just the latent code!
        # second, in stylegan we need to preserve each samples mean/variance because they
        # carry the style information, details such as colors, textures, contrast, color balance,
        # brightness, etc all come from samples mean/variance! batchnorm distorts/destorys 
        # these information and makes it very hard for the model to accurately control the style!
        # hence why batchnorm is not used!
        #
        # version2:
        # batchnorm uses the batch statistics(mean/variance) to normalize all the samples in batch,
        # i.e. it throws away individual mean/variance for each sample! this is a problem because
        # in generator we want to preserve the signal for each sample the best we can, we dont want
        # other samples to distort the statistics of another sample! we dont want other samples to 
        # affect another one in any way. all samples in the batch, in batchnorm, use the same mean/varaince!
        # if we use batchnorm, then the generator's output for one latent code z1 would depend on
        # whats in the rest of the samples in that batch! it wont be independant
        # and determinstic anymore!(i.e. depend only on latent code z1 e.g.) this destroys the
        # independece we need to generate determinstic outputs(with features we want)! 
        # aside from that, this introduces noise and instability as well, because gradients now
        # become batch dependant! a larger batch performs differently than a smaller one!
        # and more importantly, this is exclusive to stylegan1, batchnorm forces the feature activations
        # to have zero mean and variance 1(unit variance) across the batch, but in order to 
        # capture the style of the image, we need to preserve them as they carry the style information!
        # everything such as colors, textures, brightness, contrast, color balance, etc all 
        # come from these individual mean/variance. batchnorm distorts or destroys these 
        # information and makes it much harder for the model to control the style accurately 
        # and result in inconsistent images which depend on the batch!
        # 
        # apart from that, sometimes we want a specific layer train faster or slower, like our case
        # in mapping_network. in this case, we can use a specific learning rate for that layer 
        # and multiply it by the main learning rate! but we need to keep/preserve the variance
        # how do we do that? simple, we only need to scale up the weights before we apply that
        # learning rate!(i.e. lr_mult.) hence why we do self.weight.normal(0,1/lr_mult)
        # now we can easily multiply the weights by the lr_mult at runtime which is the effectively
        # the same as multiplying the main learning rate and our layer specific learning rare(lr_mult)
        # at runtime without multiplying/affecting the output variance!
        # 
        # sidenote2:
        # note that when we talk about magnitutdes, like scaling the weights, we use std (standard deviation)
        # which is linear! but when we talk about propagation! we use variance which is std²!(its squared!)
        # thats why when we scale the weights, we use 1/√fan_in(i.e. 1/sqrt(fan_in)) but when we
        # talk about variance propagatin (e.g. variance of the layers outputs) i.e. for analysis, we use variance.
        # and how the 1/sqrt(fan_in) came into being? this is from xavier init paper in 2010. 
        # the authords(glorot and bengio) said lets pick a variance for w so both the forward and backward passes
        # keep the same variance! for the forward pass they did:
        # Var[y] = fan_in*Var[Wij]*Var[x] , for Var[y]=Var[x], Var[Wij]= 1/fan_in so they cancel eachother out 
        # so forward variance for W is 1/fan_in. for backward pass this is the same we get:
        # Var[δx] = fan_out*Var[Wij]*Var[δy] for the variance of output with respect to input be the same (Var[δx]=Var[δy])
        # again Var[Wij] needs to be 1/fan_out ! so the output variance is 1/fan_out. 
        # and then they wanted to hold both conditions approximtaley so the they average them together
        # and the variance became 2/(fan_in + fan_out). therefore the std becomes sqrt(2/fan_in+fan_out)
        # 
        # since we now only care about the forward pass stability (in convolutinal layers or in generators)
        # we only take the first part i.e. var=1/fan_in and its std=1/sqrt(fan_in) which we actually use in
        # multiplication!(as we are dealing with magnitude!)
        #
        # quicknote:
        # He etal later came and extended this reasoning and said since relu drops half of its
        # activations (i.e. sets them to zero), on average, then the variance should be doubled
        # i.e. 2/fan_in to account for the lost values! 
        # (and in stylegan1 the authords instead of applying this at initialization only, applied
        # it dynamically at each forward pass so we get equalized learing rate throughout training!
        # and since we didnt use relu, we went with var=1/fan_in)
        #
        #quotenote2:
        # to get an understanding how bad this was, with N(0,1) the std is 1, so at runtime 
        # when we want to calculate weights*(1/√fan_in)*lr_mult, with lr_mult=0.01 the 
        # effective std will be 0.01 as well (W_std_effective = std_init*lr_mult = 1*0.01=0.01
        # so as a result the layer output variance will also be shrunk by lr_mult²:
        # Var[y]=fan_in*Var[Wij]*Var[x] = fan_in * (lr_mult²)*Var[x]
        # that is the variance will be 10000x smaller!
        # reminder:
        # for a scaler and a random variable x we have: Var[a.X] = a².Var[X] so 
        # so when we have W_effective = W_init * 1/√fan_in * lr_mult, with N(0,1)
        # the variance for W_init will be 1 so the effective variance becomes:
        # Var[W_effective] = (1/√fan_in * lr_mult)² * Var[W_init] = lr_mult²/(1/fan_in)*1 = lr_mult²/fan_in 
        # the output variance shrinks by 1000x (0.01*0.01=0.0001)!
        self.weight.data.normal_(0,1/lr_mult)
        # some implementations use 1/sqrt(fanin), since we dont use relu
        # √2 overcompensates and makes activations larger than they should
        # and can lead to training instability so the official impl used 1/sqrt(fanin)
        self.scaler = (1/math.sqrt(self.in_features))*self.lr_mult
        
    def forward(self, x):
        # note we are scaling the bias as well
        return F.linear(x, self.weight*self.scaler, self.bias*self.lr_mult)

# class EqualizedConv2d(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
#         super().__init__()
        
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
#         # initialize the weights 
#         self.conv.weight.data.normal_(0,1)
        
#         self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

#         # scaler is sqrt(2/fan-in)
#         # instead of math.sqrt we could also do 
#         # self.scaler = (2/(in_channels * kernel_size* kernel_size))**0.5
#         # note since scaler can be computed at runtime, theres no need to have
#         # self.register_buffer("scaler",...) to preserve it for inference!
#         self.scaler = math.sqrt(2/(in_channels * kernel_size* kernel_size))
        
#     def forward(self, x):
#         # scale the conv weights, note that in backprop, the gradients are calculated
#         # with respect to self.conv.weight normally, and our scaler here, a python scaler
#         # mind you!, just acts as a scaler (obviously) and scales the gradients so they stay
#         # uniformly scaled!!
#         scaled_weights = self.conv.weight * self.scaler
#         return F.conv2d(x, scaled_weights, self.bias, stride=self.conv.stride, padding=self.conv.padding)

# implemented based on official tf impl
class EqualizedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, lr_mult=1.0, bias=True):
        super().__init__()

        self.bias = bias
        self.stride = stride
        self.padding = padding
        self.lr_mult = lr_mult

        # initialize the weights 
        self.weight = nn.Parameter(torch.randn(size=(out_channels, in_channels, kernel_size, kernel_size)))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        
        # the official impl uses 1/sqrt(fan_in) because the √2 is there to 
        # compensate for the variance drop when we use relu
        # which zeros half of the activations! 
        # therefore when we use the leakyrelu which doesnt zero out 
        # as much, the activation variance also wont drop by half
        # so if we use √2 with leakyrelu, it will overcompensate and
        # and make activations too large, and lead to training instability
        # it could be one of the reasons why I faced so much issues in training
        # past some resolutions.(but ultimately we managed to get good results
        # so I'm not sue how much of an impact this is gonna have on us
        # to be on the safe side I do as the official tf impl does this time!)
        # also the whole thing is further scaled it by lr_mult,though
        # for conv layer its 1! but I added it anyway!
        # 
        fan_in = in_channels * kernel_size * kernel_size
        # self.scaler = (1/math.sqrt(fan_in)) * lr_mult
        self.scaler = math.sqrt(2/(fan_in)) * lr_mult
        
    def forward(self, x):
        # scale the conv weights, note that in backprop, the gradients are calculated
        # with respect to self.conv.weight normally, and our scaler here, a python scaler
        # mind you!, just acts as a scaler (obviously) and scales the gradients so they stay
        # uniformly scaled!!
        scaled_weights = self.weight * self.scaler
        # in the official code bias is also scaled
        scaled_bias = self.bias * self.lr_mult if self.bias is not None else None
        return F.conv2d(x, scaled_weights, scaled_bias, stride=self.stride, padding=self.padding)

class DiscBlockStyleGAN1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()
        self.blur = Blur()
        # update:
        # the original paper disabled bias (I trained with bias=True just fine)
        # update2: we use EqualizedConv2d instead of conv2d in all layers
        self.block = nn.Sequential(EqualizedConv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
                                   nn.LeakyReLU(0.2),
                                   EqualizedConv2d(out_channels, out_channels, kernel_size, stride, padding, bias=bias),
                                   nn.LeakyReLU(0.2),
                                   # before we downsample, we blur the input
                                   Blur(),
                                   nn.AvgPool2d(2),
                                  )
        
    def forward(self, x):
        return self.block(x)
    
# the discriminator from progan stays nearly the same except for the first res
# that starts at 8x8 instead of 4x4 and the final layer that uses two fc layers
# instead of 1 conv layer. its almost the same!
class DiscriminatorStyleGAN1(nn.Module):
    def __init__(self, max_steps=7, channels=[512,512,512,256,128,64,32]):
        super().__init__()
        assert max_steps == len(channels), f'number of channels({len(channels)}) must match max_steps({max_steps})'
        self.setup_layers(max_steps, channels)
    
    def setup_layers(self, max_steps, channels):
        self.max_steps = max_steps
        # self.starting_base = starting_base
        # update:
        # see my note in generator. this loop is too simplistic 
        # self.channels = [ 2**(i+starting_base) for i in range(max_steps,0,-1)]
        # [512,256,128,64,32,16,8]
        self.channels = channels
        # print(f'{self.channels=}')

        self.fromImgs = nn.ModuleList([nn.Sequential(EqualizedConv2d(3, self.channels[i], kernel_size=1),
                                                     nn.LeakyReLU(0.2)) for i in range(max_steps)])
        self.blocks = nn.ModuleList([DiscBlockStyleGAN1(self.channels[i],self.channels[i-1]) for i in range(1,max_steps)])

        self.remaining_blocks = nn.ModuleList()
        for step in range(self.max_steps):
            remaining = []
            for i in range(step-2,-1,-1):
                remaining.append( self.blocks[i])
            self.remaining_blocks.append(nn.Sequential(*remaining) if remaining else nn.Identity())
        
        # StyleGAN Discriminator has 2 Linear layer at the end
        self.final = nn.Sequential(AddBatchStdDev(),
                                   EqualizedConv2d(self.channels[0]+1, self.channels[0], kernel_size=3, padding=1),
                                   nn.LeakyReLU(0.2),
                                   #2 FC layers ?(FC-LReLU-FC)?
                                   nn.Flatten(),
                                   # update: while the paper says they use 8x8! 
                                   # the official code I checked used 4x4! so we
                                   # use 4x4 to be safe (explained more in generator section below)
                                   EqualizedLinear(self.channels[0]*4*4, self.channels[0]),
                                   nn.LeakyReLU(0.2),
                                   EqualizedLinear(self.channels[0],1))
                               
    
    def forward(self, x, alpha, step):
        if step == 0:
            out = self.fromImgs[0](x)
            out = self.final(out)
            return out.view(-1,1)

        new_input_out = self.fromImgs[step](x)
        new_input_out = self.blocks[step-1](new_input_out)
        
        x_downsampled = F.avg_pool2d(x, 2)
        previous_input_out = self.fromImgs[step-1](x_downsampled)
        out = alpha * new_input_out + (1-alpha)*previous_input_out
        out = self.remaining_blocks[step](out)

        out = self.final(out)
        return out.view(-1,1)
 
# The generator however as we discussed is different. we now need a mapping network, AdaIN, and
# noise injection. lets implement them
 
# Mapping Network is a 8 layer mlp. but since its linear layer, like conv layers before
# we need to use the equalized version

# Mappingnetwork is an 8 layer mlp wtih leakyrelu as nonlinearity
# it accepts the latent vector z and gives us the latent vector w
# sidenote: the paper says that increasing the depth of the 
# mapping network tends to make the training unstable with 
# high learning rates so they use a much lower lr(λ= 0.01)
# specifically for mapping network optimization
class MappingNetwork(nn.Module):
    def __init__(self, z_dim=512, w_dim=512, num_layers=8):
        super().__init__()
        layers = [] 
        for i in range(num_layers):
            # to encourage slower/smoother updates in the latent space mapping we use a much smaller lr_mult
            layers.append(EqualizedLinear(z_dim if i==0 else w_dim, w_dim, lr_mult=0.01))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.net = nn.Sequential(*layers)
        self.pixel_norm = PixelNorm()

    def forward(self, z):
        # normalize the latent vector before feeding
        # it to mapping network
        z_normalized = self.pixel_norm(z)
        return self.net(z_normalized)


# AdaIN is also a simple module that normalizes the input to have mean=0,std=1
# and then adds the scaler/beta we learn from w!
# the official tf impl only has the modulation part. the normalization
# doesnt happen in adaIN, adaIn only recieves the x, w and works on w
# and applies the styles on x. x is normalized beforehand in styleconv
# but I did it here
# todo move instance_normalization out of adain and into styleconv so
# its more in line with the paper official impl
class AdaIN(nn.Module):
    def __init__(self, channels, w_dim=512, eps=1e-8):
        super().__init__()
        # nor normalization numerical stability
        self.eps = eps
        self.fc_style = EqualizedLinear(w_dim, channels)
        self.fc_bias = EqualizedLinear(w_dim, channels)
        # instead of creating two separate linear layer to get scale and beta
        # separately, we use one linear layer and then split it to get the two
        # self.fc_style = EqualizedLinear(w_dim, channels*2)
                        
    def forward(self, x, w):
        # normalize the input x to have zero mean/unit variance
        # we can  x = (x-x.mean())/x.std() or simply 
        # use F.isnatnce_norm()
        x_norm = F.instance_norm(x, eps=self.eps)
        # now get the scale and beta from w
        scale = self.fc_style(w).unsqueeze(2).unsqueeze(3)
        bias = self.fc_bias(w).unsqueeze(2).unsqueeze(3)
        # style = self.fc_style(w).unsqueeze(2).unsqueeze(3)
        # scale, bias = style.chunk(2,dim=1)
        
        # add 1 for inital identity like behavior!
        # this trick is from the official nvidia implementation, 
        # since at the start, the initial weights of fc_style produce scales close to zero
        # we do this so the adain layer start closer to an identity function and 
        # it results in more stability(we dont start off with std ~0! things are 
        # kept in a normal range!)
        return (1+scale) * x_norm + bias

# noise injection
# for noise injection, we want to apply noise to each pixels in each featuremap
# so the shape should be 4d, 1,channels,1,1, we also dont want to randomly
# inject noise, after all this is supposed to add random attributes like freckles
# etc, so the network needs to be able to control this, so we add a weight paramter
# to make this a learned operation rather than a completelyy random operation
class NoiseInjection(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1,channels,1,1))

    def forward(self, x, noise=None):
        # for debugging purposes we can send in a fixed noise
        # otherwise we dont need that.(we need it for paper experiments aswell)
        # I explained more in the experiment section after the training.
        # we dont really need the noise argument in training, its mostly
        # for those experiments.
        # update:
        # ok I never actually got to use this for debugging during training
        # however during inference time, to create different experiments
        # that we see in the paper, I had to disable noise injection
        # because it acts as a source of randomness, so I could experiment
        # and show the effects of truncation trick and or style mixing like
        # the paper! 
        # for the experiment, I had to manually set the weighst to zero
        # because I couldnt use the noise argument here properly because
        # of shape mismatch.(getting the right size in loop becomes cumbersome
        # unless we changethe forwardpass toa ccount for this which I dont like
        # I want to keep it as straight forward as I can. so I guess
        # to make this easier I can simply add the logic here! simply
        # create the zero tensor here if noise needs to be constant/disabled!)
        # todo keep noise=None as default so we dont change anything in code by default
        # but add new case to dynamically update the spataial dim of the noise here
        # so we dont get shape mismatch! anymore! for constant/zero case this
        # works 100% and for anytihng other like a fixed tensor of some value, 
        # this should still work!
        if noise is None:
            noise = torch.randn(size=(x.size(0), 1, x.size(2), x.size(3)), device=x.device)
        else: # or if we use a fixed something, we make sure its dims match!
            noise = F.interpolate(noise, size=(x.size(2), x.size(3)))
        return x+(self.weight*noise)

# StyleBlock
# unlike progan, since we have new operations, we need to make a new block
# we need AdaIN, EqualizedConv, and NoiseInjection with leakyrelu 
# the description of each block is given in figure1 page 2 of the paper
# but the official tf implementation differs from it! in official code
# the order is conv>noise>bias>lrelu>adain but paper says conv>noise>lrelu>adain!
# 
# update:
# I kept facing a lot of instability during training especially starting
# with 32x32 resolution. the discriminator kept overpowering the generator
# so as a last resort, I swapped the order so adain is first applied and 
# then did lrelu! after this I instantly faced massive improvement!
# and generator took the lead!
# I guess the reason behind this improvement is that when we apply the
# nonlinearity(leakyrelu) before adain we are warping the distribution
# of the featuremaps in a nonlinear way! which then when adaIN tries to
# normalize this already rectified distribution, it gets weird! its now 
# skewed in a sense! 
# we dont want that! as doing this weakens the adain's ability to control
# the style properly! (come back to our intial intution,
# we wanted to normalize the input directly not after going through a 
# nonlinearity that changes the whole landscape somewhat drastically!)
# to fix this obviously we need to apply adain right after the convolution 
# and noise injection but before the activation!
# as I said before doing this no matter what I did at 32x32 res, the disc
# would get much lower loss (e.g. 0.7vs1.2) and image quality would not improve! 
# this would be compunded in later res. when I swapped the order and 
# applied adain before lrelu the issue was fixed and d_loss became 
# 1.35 vs 0.72 of g_loss. basically the roles are reversed and gen
# is now doing much better than disc!
# note that in all of the tests Ive done the discriminator and generator had
# the same exact channel configurations.
#update2:
# I ultimately found the initial bug that made disc overpower generator 32x32 
# onward! there was a bug in EqualizedLinear implementation. I explained this
# in detail in debug log and EqualizedLinear layer aswell. but the point is the
# order swap still applicapble and much better than the original order. In fact
# I noticed this is in line with the findings in the second version of the paper 
# (stylegan2) and they did pretty much what we did here as well (plus more obviously)
# but the part we did here stays valid nonetheless.
#
class StyleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True,
                 w_size=512, upsample=False, eps=1e-8, swap_adaIN_order=False, apply_conv=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        # the official tf implementations applies bias separately after conv
        # so I include bias instead!
        self.bias = bias
        self.w_size = w_size
        self.upsample = upsample
        self.eps = eps
        # see note below
        self.swap_adaIN_order = swap_adaIN_order
        # for 4x4 input we dont apply conv the first time, only noise, lrelu/adain
        # but the second time we do the whole thing with conv
        # not sure how much of a difference it makes initially as conv is a linear
        # operation
        self.apply_conv = apply_conv
        # sidenote: 
        # to use my simplified version, set bias=bias,
        # and comment out self.bias related code in forward(), 
        # and in generator use apply_conv=True for
        # the first layer aswell! this way we will have
        # conv/noise/lrelu/adain order which is
        # my first implmentation and works just as well.
        # however by setting bias=False and use a 
        # seprate bias(self.bias below), we basically follow
        # the official tf impl. see debug log to know why I did this!
        self.conv = EqualizedConv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.noise_inject = NoiseInjection(out_channels)
        self.bias = nn.Parameter(torch.zeros(out_channels,))
        self.adain = AdaIN(out_channels, w_size, eps)
        self.blur = Blur()

    def forward(self, x, w, noise=None):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            # blur the output to hide checker marks effects this is especially important
            # for 32x32 and higher res so the discriminator doesnt win too quickly!
            x = self.blur(x)

        # note from furture:
        # my initial simplification:
        # conv/noise/lrelu/adaIn works just fine!
        # and my only issue was the bug in equalizedlinear layer
        # but since I've already added the separate bias, and trained
        # a few models, I'll keep these changes so we can easily load
        # the checkpoints for later use. 
        # just know, the separate bias and no conv for the first layer
        # of 4x4 isnt a hard implenetation choice and we can easily use
        # the simplified version (which I originally did, i.e. use the
        # EqualizedConv2d's own bias, and apply conv and then noiseinjection)
        # works just as good. 
        # you can always go to the nov 7, commit 6c59bcb2d76ed1cdfd8dc13d0def012d80d16482
        # and see what I mean (however, note we had several serious bug fixes after it
        # like equalizedlinear layer bug and separate lr for mapping_network in optimizer)
                
        if self.apply_conv:
            out = self.conv(x)
        else:
            # in official tf impl, the first layer of
            # 4x4 res doesnt use any conv layer!
            out = x
            
        # inject noise into the output featuremaps,
        out = self.noise_inject(out, noise)
        
        # apply a separate bias!
        # note: we can omit this completely and use the bias in conv layer
        # and there would be no issues! 
        # i.e. doing conv(+bias)>noise_injection>lrelu>adain
        # works as well, but I leave it as is cuz its how the
        # official impl did it (see debug log for extensive explanation)
        if out.ndim==2:
            out += self.bias
        else:
            out += self.bias.view(1,-1,1,1)
        
        # apply the styles from w on the output
        # update: 
        # do adain before lrelu !
        # applying it after activation results in disc overpowering gen!
        # at higher res (starting from 32x32!)
        # so the order must be conv>adain>activation since adain normalizes
        # the featuremap and it must happen before nonlinear activation warps
        # the statisics of the featuremap!
        # the official code uses the original order i.e. conv-noise-act-adain
        # but for me this swapped oreder works otherwise I couldnt get good
        # images starting 32x32 res! after I swapped it the convergence became
        # very fast!
        # todo understand why!
        # update:
        # I digged deeper and tested more configurations, I noticed nearly every
        # implmenetation I saw online had incorporated stylegan2 tricks into the paper
        # to varying extend.
        # also the original order (i.e. lrelu>adain) should work on FFHQ with proper
        # channel capacity(channel capacity and epoch directly affect the result but
        # since it takes a huge amount of time to test I gave up on it and used smaller
        # channels in early tests.later on with fp16(now thatI know our implementation works
        # I tried and got good results(see the test section)))
        # anyway just wanted to add that here
        # update2: 
        # that was not the case, the reason disc would always dominate gen post 32x32
        # was becasue of a bug in equalizedlinear, after I fixed that, the original 
        # order worked flawlessly. however, my findings was still great, it makes training
        # much more stable and easier! and while the other implementations use stylegan2
        # tricks, its not why they worked, they still implemented stylegan1 and didnt have
        # a bug like me! when I fixed mine, we were as good as them, even better!
        if self.swap_adaIN_order:
            # my change: 
            # normalize pre-activation
            out = self.adain(out, w)
            out = F.leaky_relu(out, negative_slope=0.2)
            return out
        else:
            # original order: first nonlinearity then adain
            # normalzie post-activation
            out = F.leaky_relu(out, negative_slope=0.2)
            out = self.adain(out, w)
            return out

# its much easier for us to implement style mixing here inside generator
# than implement it as a standalone function and use it during training!
# doing it here is much more straightforward and easier
# basically with a probablity e.g 90% chance we use two latent vector z
# instead of just 1, and combine their ws together and feed that to our
# generator. we only do that in training as we explained before
class GeneratorStyleGAN1(nn.Module):
    def __init__(self, z_size=512, w_size=512, max_steps=7, 
                 mn_num_layers=8, channels=[512,512,512,256,128,64,32],
                 style_mixing_prob=0.9, ema_w_beta=0.995,
                 swap_adaIN_order=False, ):
        super().__init__()
        
        assert max_steps == len(channels), f'number of channels({len(channels)}) must match max_steps({max_steps})'
        # lets do the same thing for generator
        self.setup_layers(z_size, w_size, max_steps, mn_num_layers,
                          channels, style_mixing_prob, 
                          ema_w_beta, swap_adaIN_order)

                
    def setup_layers(self, z_size, w_size, max_steps, mn_num_layers,
                     channels, style_mixing_prob, ema_w_beta,
                     swap_adaIN_order):
        
        self.z_size = z_size
        self.w_size = w_size
        self.max_steps = max_steps
        self.mn_num_layers = mn_num_layers
                
        self.style_mixing_prob = style_mixing_prob
        # truncation trick! in order to get higher quality generations
        # like the paper says, we need to use trunkation trick 
        # to do that we need a moving average of our w 
        # without doing this we get more varied images but also with
        # more artifacts, the average w, increases the quality at
        # the expense of lower diversity
        self.ema_w_beta = ema_w_beta
        self.register_buffer("ema_w",torch.zeros(size=(1,w_size)))
        
        # swap the order of AdaIN/LeakyReLU in styleconv block
        self.swap_adaIN_order = swap_adaIN_order
        
        # for our simple tests this is ok, but if we wanted to go for higher res
        # we can repeat some channels for adjacent res to get better result
        # i.e. instead of doubling each time we can have e.g. [512,512,512,256,128,64...]
        # update:
        # this was a bad idea, this way of specifying channels isnt efficient
        # its too simplistic and we can get carried away and forget that we set up a stupid
        # channel count later on and waste too much time figuring out what is missing or
        # implemented incorrectly and we cant get good results! 
        # the original used much larger channels so here we go!
        # self.channels = [ 2**(i+starting_base) for i in range(max_steps,0,-1)]
        self.channels = channels
        # print(f'{self.channels=}')
        
        # unlike progan, we start with a learned constant tensor, as if its a blank canvas
        # and little by little draw on it! we start with a 1x512x8x8 block(the paper says so
        # in page 9 Hyperparameters and training details! however the the official code uses
        # 4x4! https://github.com/NVlabs/stylegan/blob/master/training/networks_stylegan.py#L504)
        #
        # so I'll go with the official implementation and use 4x4 instead to be on the safe side!
        # sidenote:
        # I use torch.ones as its what the paper says in training details section
        # and it aligns better with my initial analogy. however in stylegan2 they used randn
        # instead (for better starting variance I guess, im not sure) so I added that aswell.
        # 
        self.const_input = nn.Parameter(torch.ones(size=(1, self.channels[0], 4, 4)))
        # this is from stylegan2 (page 11 section B implementation details, generator redesign)
        # might be a good idea to test it later! for now stick to the torch.ones version!
        # until we get a fully working implementation then we can test!
        # self.const_input = nn.Parameter(torch.randn(size=(1, channels[0], 4, 4)))
        
        self.mapping_network = MappingNetwork(z_size, w_size, self.mn_num_layers)
        # unlike progan, the official tensorflow implementation doesnt use tanh, and
        # outputs are unbounded.(see https://github.com/NVlabs/stylegan/blob/master/training/networks_stylegan.py#L524)
        # the idea is, we know tanh might help initially, but since its saturating, it can
        # lead to slow convergence/limit the expressiveness of the network. without
        # it the network should(and will) be able to learn the proper range itself
        # so theres no issue if we normalize the dataset images to -1,1 during training,
        # it can handle it just fine (sidenote: if tanh gets saturated, we see washed out
        # colors or even mode collapse!) so I remove the tanh here. (now that I think about it
        # I guess progan also didnt use tanh! but I did! need to remove that aswell!)
        self.toImgs = nn.ModuleList([EqualizedConv2d(self.channels[i], 3, kernel_size=1) for i in range(max_steps)])
        
        # unlike the progan version, the stylegan paper uses two layers for each res
        self.blocks = nn.ModuleList()
        #4x4
        # I initially used one styleconvblock for 4x4 res, but the paper used two. 
        # using one layer, means the network cant stylize the input strongly, so the base
        # would lack refienments therefore we would face low convergence because 
        # the network has less ability in injecting diverse styles early on!
        # also the first layer doesnt use conv but directly injects noise and adds bias
        # followed by lrelu activation and adaIn application! we could simply apply conv
        # anyway and it works, but to follow the official impl I later decided to 
        # do the same thing here.
        self.blocks.append(StyleConvBlock(self.channels[0], self.channels[0], w_size=w_size, upsample=False, apply_conv=False,swap_adaIN_order=self.swap_adaIN_order))
        self.blocks.append(StyleConvBlock(self.channels[0], self.channels[0], w_size=w_size, upsample=False, swap_adaIN_order=self.swap_adaIN_order))
        for i in range(1, max_steps):
            self.blocks.append(StyleConvBlock(self.channels[i-1], self.channels[i], w_size=w_size, upsample=True,swap_adaIN_order=self.swap_adaIN_order))
            self.blocks.append(StyleConvBlock(self.channels[i], self.channels[i], w_size=w_size, upsample=False,swap_adaIN_order=self.swap_adaIN_order))

    @torch.no_grad()
    def _update_ema_w(self, w_batch):
         if self.training:
             # beta*old_avg + (1-beta)*new_avg
             # note we use the whole batch of w so we need to avergae
             # it. doing this will include the information from
             # multiple smaples in each update round.(its smoother
             # than per sample verion).
             # the idea behind this is that by calculating the mean
             # of w in the W space (i.e. the center of mass)
             # we can access typical/orinary styles as apposed to 
             # more different/varied ones that are farther from the
             # normal (i.e. mean).therefore these have higher chances
             # of being much better developed than the more rare/less 
             # frequent ones.averaging over the batches approximates 
             # this for us without needing to storing the whole past!
             self.ema_w.mul_(self.ema_w_beta).add_(w_batch.mean(0), alpha=1-self.ema_w_beta)
    
   
    def forward(self, z, alpha, step, psi=None, noise=None):
        # stylemixing during training
        # initially I used a single w for all layers, but as the paper
        # says, the w must be per layer. that is duing the mixing process
        # we need to provide a different w for each AdaIN layer in each block
        # so we need to calculate the number of styles (number of styleconvblocks)
        # since we have 2 layers per resolution for each step we will have:
        num_styles = 2*(step+1)
        #convert the latent z into latent w
        w = self.mapping_network(z)
        
        # update the ema_w for truncation trick
        if self.training:
            self._update_ema_w(w)

        # we use truncation trick during inference only
        # sidenote: we could also do this in one go, post trainig
        # by generating a few thousands sample ws using our 
        # mapping network, and then aveage those ws! i.e.:
        # zs = torch.randn(size=(50_000,self.z_size))
        # ws_avg = self.mapping_network(zs).mean(dim=0,keepdim=True)
        # w = ws_avg+psi*(w-ws_avg)
        # that would give us the average w! the lager the sample size
        # the more accurate our estimate will be but we can also learn
        # it as we go during training so we opted to do this instead!
        # sidenote2:
        # why do we do thta? because in the z space, the are areas of low probablity
        # that sometimes for some reasons, map to strange/low quality/weird samples
        # images, either because some features are not developed properly because the
        # training samples are not enough(i.e. they are underepresented), or the network 
        # is not capable enough, for whatever reason, they just happen to be much different
        # than normal images! now if we instead try to sample ws closer to aveage face(average w), 
        # the chances of getting more high quality, but typical/average looking images increases.
        # so we are trading between higher quality vs diversity, cuz the closer we get to avegae
        # the more uniform/the same we look! too much far away from average therefore means
        # somethin is drastically different than the average!
        if not self.training and psi:
            ema_w_batch = self.ema_w.repeat(w.size(0),1)
            # (1-psi)*ema_w + psi*w or simplified as
            # ema_w+psi*(w-ema_w)
            w = ema_w_batch + psi * (w - ema_w_batch)
        
        if self.training and random.random() <self.style_mixing_prob:
            # grab a second z, calculate the w
            z2 = torch.randn(size=z.size(), device=z.device)
            w2 = self.mapping_network(z2)
            # I specifically dont update ema_w with w2 because first
            # the paper only uses w1, and second if we do use w2, it
            # actually goes more toward diverse outputs which we dont
            # want here 
            # self._update_ema_w(w2)
            # 
            # now we need to pick a crossover point to stich half of
            # each w with the other one and form a new w!
            # this is done so the images that get generated have their
            # coarse features (early layers) come from from one style 
            # and their fine details(later layers) from another.
            # but we have 1 w for each sample! after all mapping network
            # produces one w for eac z! i.e. we have 1 w for each sample
            # to make this work, we simply repeat it for as many layers
            # as we have, now we have a w for each layer and the corssover
            # point is calculated based on number of layers here
            # sidenote:
            # if we are repeating the same w for all layers, then why not
            # sharing the single w with all of them like before? the reason is
            # this way, each layer gets to independently participate and have
            # a say, and we get a separate gradient path for each layer now,
            # its as if we have several outlets throughout the house that share
            # the same electricity cable(or facets that use the
            # same main water pipe) with this every room has access to it and
            # can use flow as it needs. using this each layer can now contribute
            # differently to the final style/image, and we also get to control
            # affect the style by changing or playing with each layers specific
            # w (the underlying storgae is the same, but final gradient sum is also
            # the same, but this way we take a different route of optimization and
            # and using the mixing here get a whole different path! put in 
            # otherwords this is our attempt to make W more disentangled usinf this
            # trick(it enforces better dientanglement and locality in the
            # learned representations. we can still generate images without doing this
            # (remember progan!) however, the whole issue was the enganglement! so this 
            # helps with that! I guess its now too obvious!)
            #
            # sidenote: we could use expand but have to clone it, because expand uses 
            # view under the hood, so it doesnt actually copy anything, just points to
            # the underlying data so to get correct result we need to clone it, but 
            # repeat copies the data! which makes it prefect for us here!
            # 
            # so we could have also repeated for the whole layers by default and later
            # here only grab up to the current step layers! that would have worked as well
            # todo do this instead!
            crossover_point = random.randint(1, num_styles-1)
            # repeat along the channels dim so we get (b,num_styles,w_dim)
            # one w for each layer!
            # to avoid an extra concat, we assign the first part to w
            # and change the other half with w2 so we dont do 
            # something like (w = torch.cat([w1[:,:crossover_point], w2[:,crossover_point:]],dim=1))
            w = w.unsqueeze(1).repeat(1,num_styles,1)# (b,num_styles,w_dim)
            w[:, crossover_point:,:] = w2.unsqueeze(1).repeat(1,num_styles-crossover_point,1)
            # print(f'{crossover_point=}')
        else:
            # if no style mixing or training
            # make w per layer like before so each adaIN gets
            w = w.unsqueeze(1).repeat(1,num_styles,1)

        # set the batchsize forr const_input/canvas
        x = self.const_input.repeat(z.size(0), 1,1,1)
        # the process goes like this:
        # process the inputs from lowest res to the current res
        # if its the first step, no fadein required, convert to
        # image and return. 
        # otherwise, process the input from the lowest res to the
        # current res and go for fadein at the end.
        
        if step == 0: #4x4
            # we need to pass a separate w for each layer accordingly
            x = self.blocks[0](x, w[:,0,:], noise)
            x = self.blocks[1](x, w[:,1,:], noise)
            return self.toImgs[step](x)
        
        # now feed the inputs x and w to blocks
        # for each step, we start off from the very begining
        # and go all the way to the current step.
        # since we want the previous_output as well we keep
        # one copy like before
        previous_image = None
        for i in range(2*step+2): # since we have 2 layers per res
            # first process the input through 
            # if we do this after the following if block, previous_image
            # would be the raw const_input that has not been stylized by adain
            # and noiseinjection is not applied!
            x = self.blocks[i](x, w[:,i,:],noise)
            # save a copy of previous step output
            if i == (2*step-1) and alpha<1:
                # the current x is the output for previous res so 
                # convert it to image and upsample it so it matches
                # the next res
                previous_image = self.toImgs[step-1](x)
                previous_image = F.interpolate(previous_image, scale_factor=2, mode='nearest')
                
        # now x is the final output, i.e. the highest res
        # so convert to image
        new_image = self.toImgs[step](x)
        if previous_image is not None:
            final_image = alpha * new_image + (1-alpha)*previous_image
        else:
            final_image = new_image
        return final_image
    
# x = torch.randn(size=(5,3,256,256))
# z = torch.randn(size=(5,100))
max_steps = 7
channels=[512,256,128,64,32,16,8]
disc = DiscriminatorStyleGAN1(max_steps=max_steps,channels=channels)
gen = GeneratorStyleGAN1(100,100,max_steps=max_steps,channels=channels)

for m in [disc, gen]:
    print(f'channels: {m.channels}')
    
# test all the stages/steps
for i in range(0,max_steps):
    # start off with 4x4
    H = W = 2**i*4
    x = torch.randn(size=(5,3,H,W))
    z = torch.randn(size=(5,100))
    n = torch.zeros((5,1,H,W))
    disc_out = disc(x, alpha=1, step=i)
    print(f'disc_out.shape: {tuple(disc_out.shape)}')
    gen_out = gen(z, alpha=1, step=i,noise=n)
    print(f'gen_out.shape : {tuple(gen_out.shape)}')

#%%
# now for training the loop stays the same with minor changes
# before we go for training we need a few more things to implement.
# the mixing regularization and the loss function
def r1_penalty(d_preds, x_real, gamma=10):
    assert x_real.requires_grad, 'discriminators inputs(x_real) must have requires_grad enabled!'
    
    # here we are going to penalize large gradients this will effectively preventing 
    # the gradients from exploding. the idea is, since we are using a non-saturing loss
    # and we dont limit the discriminator;s output, discriminator can produce infintly large
    # scores. this will therefore result in huge gradients and later gradient explosion.
    # to fix this we simply try to prevent the gradient from getting too large, 
    # or in other words prevent the discriminator's function from becoming too steep
    # and therefore keeping its amplifying rate/gain low! when we do that, in effect it
    # will slow down the discriminator's learning, so it effectively prevents the discriminator
    # from becoming too good too fast!
    # 
    # sidenote:
    # we can do d_preds.sum() and remove torch.ones_like(d_preds) below
    # .sum() aggregates the discriminators outputs into a scaler 
    # and tells the grad() what to differntiate. sum as you recall distributes
    # the gradients to every single element equally, so its the same as starting
    # the gradient calculation process by providing all ones for every d_preds elements
    # (i.e. torch.ones_like(d_preds)). so its basically a simple trick
    # that saves us from typing a few characters more! 
    # I prefer our version better because its clear what and why im doing this(sum is
    # aswell, but ours is straightforward)
    grads = torch.autograd.grad(outputs=d_preds,
                                inputs=x_real,
                                grad_outputs=torch.ones_like(d_preds),
                                create_graph=True)[0]
    
    # take squared l2norm (l2norm is sqrt(sum(x²)) so squaring it becomes: sqrt(sum(x²))²
    # we coluld do: grads_l2norm_squared = grads.view(d_preds.size(0),-1).norm(2,dim=1).pow(2).mean()
    # but our second version is a bit faster becaue sqrt and pow operations are not 
    # used (they cancel eachother out anyway so theres no need to calculate them)
    grads_l2norm_squared = grads.pow(2).view(d_preds.size(0),-1).sum(1).mean()
    penalty = gamma/2 * grads_l2norm_squared
    return penalty

def discriminator_loss_stylegan1(d_preds_real, x_real, d_preds_fake, gamma):
    # D_loss = E[softplus(-D(x_real)) + softplus(D(G(z)))]+ r1_penalty
    # softplus is log(1+exp(x)) but since pytorch offers a numerically
    # stable version, we use the builtin one
    
    
    # softplus, is a non-saturating loss, like WGAN/GP. it is a smoothed version of
    # relu (i.e. log(1+exp(x))) and is not bounded, so we are basically getting raw
    # outputs/logits so the output can get as large as it wants!
    # 
    # the sofplus+r1_penalty is there so we dont face vanishing/exploding
    # gradients! the softplus prevents the vanishing gradients, by being non-saturating
    # but not the exploding gradients!
    # in fact because its unbounded its very susceptible to gradient explosions!
    # by switching to a non-saturating loss, the discriminator's output can grow indefinitly!
    # (it can learn to produce huge scores for tiny changes in input images! be it real or fake
    # for real huge positive scores, for fakes, huge negative scores!)
    # which means the gradient of the discriminator with respect to its input becomes huge!
    # all of this means, the discriminator acts as a massive amplifier! 
    # so when the generator backpropagates its loss, our discriminator which acts as this
    # huge gradient amplifier causes the gradient signal to explode!
    # 
    # to fix the explosion part we use the r1_penalty term. its job is to make sure 
    # discriminator doesnt get too good too fast! therefore the discriminator itself
    # rarely faces gradient explosion if at all but the generator on the other hand 
    # that only uses softplus(D(x)) is very prune/susciptible to gradient explosions.
    # see the explanations ahead.
    #
    loss = (F.softplus(-d_preds_real) + F.softplus(d_preds_fake)).mean()
    # update:
    #  after adding fp16, I noticed r1_penalty needs to be done 
    #  in full precision mode (i.e. fp32) otherwise we get nans
    with torch.amp.autocast(device_type="cuda", enabled=False):
        penalty = r1_penalty(d_preds_real.float(), x_real.float(), gamma)
    
    return loss + penalty

def generator_loss_stylegan1(d_preds_fake):
    # G_loss = E[softplus(-D(G(z)))]
    
    # sidenote from future/reminder:
    # I originally wrote this for dbeug log when I faced gradient explosion at some point
    # but I thought having it here as a reminder works better so here it is:
    #
    # note, softplus is a non-saturating loss, just like the wgan/wgangp loss.
    # this matters because in our previous experiments we saw when discriminator gets really 
    # good, the gradients vanish and thats why generator cant improve, cuz it doesnt get 
    # proper feedback. 
    # however this behavior depends on the loss function we use. if we use a staurating loss
    # functions like BCE, that uses sigmoid, when the model gets good, (the real images e.g.)
    # it produces values very close to 0, so the the gradients become extremely tiny and 
    # therefore the generator cant improve with those tiny gradients. 
    # on the otherhand, when we use non-saturating losses, that are not bounded,(like softplus/WGAN), 
    # i.e. we dont use any activations for the final layer of discirminator, no sigmoid is used,
    # the network can learn to produce very large scores for the real images and very small
    # ones (i.e. very large negative numbers) for fake images! 
    # so when we calculate the generator's loss by softplus(-disc(fake_images)).mean() and 
    # the discriminator produces large negative numbers then the gradients will be massive 
    # as well! this will lead to gradient explosion!
    #
    # note the softplus uses log, and any large number given to log will be small, so loss 
    # itself isnt going to explode, the gradient will! and its the generator that goes down
    # the hill!(i.e. the explosion happens in the generator not the discriminator 
    # because the gradient is propegated through several layers in the generator! because of 
    # repeated multiplications that follow!
    # for example imagine this:
    # if our discriminator is very good and assigns a large positive number to real images,
    # i.e. D(fake_imgs)=40 the gradient of softplus(y) with respect to its input(y) will be 
    # dLG/dsotfplus(y) = 1/(1+e^-y) or in other words simply sigmoid(y), now the gradients
    # with respect to discriminator will be dLG/dD_fake = -simoid(-40) = -4*10^-18 essentially 0!
    # the gradient will be very tiny and its almost nothing!  
    # if model wasnt good and produced a large positive number for fake images, then there 
    # would be no issues again! cause the gradient would be nearly 0 and at most generator wouldnt
    # change much) however, when the discriminator assignes a large negative number to fake images,
    # confidently identifying it as fake, i.e. D(fake_images)=-40, we then have (dervaitive with
    # respect to D): softplus(y)=(-(-40)) -> softplus(40) -> -sigmoid(40)=-1.
    # now if we have lets say 10 layers, and imagine their weights to be a value like 3 for 
    # the sake of our example, then we will have -1*3^10=-59049! when we reach the first layer!
    # as you can see the gradient magnitude just exploded through several layers of multiplications! 
    # 
    # note:3^10 is just an analogy in place of the actual multiplications that happen after each layer
    # because that would also grow exponentially, I simply replaced it with an example with exponential growth like that!
    # 
    # (note the sign doesnt matter here the magnitude does! when updating the weights,
    # we move in the opposite direction of the gradients (i.e. delta_w=-lr*grad=0.001*(-59049)=~59 e.g.
    # so w_new = w_old+delta_w! now the new weights will also start to get large(explode) and
    # in the next forward pass cause the activations to explode and result in nans!
    # this is why we cant have disc get good quickly at all, if it does, we can face gradient explosion!
    # and vanishing gradient depending which will result in exploding gradient ultimately!
    # 
    return F.softplus(-d_preds_fake).mean()

# we have implemented the disc/gen
# we have implemented the losses
# so lets do the training loop

@torch.no_grad()
def update_ema_generator(g:GeneratorStyleGAN1, g_ema:GeneratorStyleGAN1, decay=0.999):
    # sidenote, we only update the parameters we dont touch buffers 
    # as it would have destroyed their stats!)
    # this dynamic decay is from stylegan2 if I dont get any better 
    # results will go back to the old version!
    # decay = min(1 - 1 / (warmup_images_seen / 1000 + 1), decay_rate)
    for ema_p,p in zip(g_ema.parameters(), g.parameters()):
        ema_p.data.mul_(decay).add_(p.data, alpha=1-decay)
    # copy the ema_w over
    g_ema.ema_w.copy_(g.ema_w)
   
   
def training_loop_stylegan(discriminator:DiscriminatorStyleGAN1, generator:GeneratorStyleGAN1, disc_optimizer:torch.optim.Adam, 
                         gen_optimizer:torch.optim.Adam, epoch_list, batch_size_list, gen_update_interval, dataset_name,
                         split, data_augmentation=False, normalize=True, use_fp16=False, r1_penalty_interval=16, gamma=10, psi=0.7, gen_num_samples = 64, 
                         use_ema_inference=False, ema_warmup_images_threshold=500_000,
                         keep_raw_generations=True, quick_and_noisy_IS_FID=False, device='cuda', resume=False,
                         decay_step=3, weights_save_dir='./weights/gan', images_save_dir='./results/gan', checkpoint_path=None,
                         decay_func = lambda step:0.5**(step-2)):
    
    experiment_date = datetime.now().strftime("%Y%m%d%H%M%S")
    current_experiment_name = f"stylegan1_{dataset_name}_{experiment_date}"

    lr_d = disc_optimizer.param_groups[0]["lr"]
    lr_g = gen_optimizer.param_groups[0]["lr"]
    # print(f'{lr_g=}')
    betas_d = disc_optimizer.defaults["betas"]
    betas_g = gen_optimizer.defaults["betas"]
    
    ema_generator = copy.deepcopy(generator).requires_grad_(False).eval()
    
    # real images seen so far during training
    ema_warmup_images_seen = 0
    
    assert discriminator.max_steps == generator.max_steps, 'max_steps for generator and discriminator/critic must be equal!'
    
    metric = IS_FID_Calculator(device)

    fixed_z = torch.randn((gen_num_samples, generator.z_size)).to(device)

    starting_step = 0
    starting_epoch = 0
    last_training_step_counter = 0
    max_steps = discriminator.max_steps
    z_size = generator.z_size
    
    scaler = torch.amp.grad_scaler.GradScaler(device, enabled=use_fp16)
    
    # check for resuming from a checkpoint
    if resume:
        if checkpoint_path:
            checkpoint_filename = os.path.split(checkpoint_path)[-1]
        else:
            checkpoints_dirs = sorted([subdir for subdir in os.listdir(weights_save_dir)\
                                       if os.path.isdir(os.path.join(weights_save_dir, subdir))])
            #grab the last checkpoint/most recent one
            checkpoint_dirpath = os.path.join(weights_save_dir, checkpoints_dirs[-1])
            # grab the latest checkpoint 
            checkpoint_files = sorted([f for f in os.listdir(checkpoint_dirpath) if f.endswith(".ckpt")])
            checkpoint_filename = checkpoint_files[-1]
            checkpoint_path = os.path.join(checkpoint_dirpath, checkpoint_filename)
        
        if not os.path.exists(checkpoint_path):
            raise ValueError("The Path is not valid")
        
        # load the stuff
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        max_steps = checkpoint["max_steps"]
        channels_d = checkpoint["channels_d"]
        discriminator.setup_layers(max_steps,channels_d)
        discriminator.load_state_dict(checkpoint["disc_state_dict"])
        discriminator = discriminator.to(device)
        
        z_size = checkpoint["z_size"]
        w_size = checkpoint["w_size"]
        mn_nlayer = checkpoint["mn_nlayer"]
        channels_g = checkpoint["channels_g"]
        style_mixing_prob = checkpoint["style_mixing_prob"]
        swap_adaIN_order = checkpoint["swap_adaIN_order"]
        ema_w_beta = checkpoint["ema_w_beta"]
        
        generator.setup_layers(z_size, w_size, max_steps, mn_nlayer, channels_g,
                               style_mixing_prob=style_mixing_prob,
                               ema_w_beta=ema_w_beta,
                               swap_adaIN_order=swap_adaIN_order)
        generator.load_state_dict(checkpoint["gen_state_dict"])
        generator = generator.to(device)
        
        scaler.load_state_dict(checkpoint.get("scaler_state_dict", scaler.state_dict()))
        
        # load the ema version, dont forget to also load the images seen so far!
        ema_warmup_images_threshold = checkpoint["ema_warmup_images_threshold"]
        ema_warmup_images_seen = checkpoint["ema_warmup_images_seen"]
        if use_ema_inference:
            ema_generator.load_state_dict(checkpoint["gen_ema_state_dict"])
            ema_generator = ema_generator.to(device)
        
        disc_optimizer.load_state_dict(checkpoint["disc_optimizer"])
        gen_optimizer.load_state_dict(checkpoint["gen_optimizer"])

        # grab the initial lrs
        lr_d = checkpoint["lr_d"]
        lr_g = checkpoint["lr_g"]
        betas_d = disc_optimizer.defaults["betas"]
        betas_g = gen_optimizer.defaults["betas"]

        data_augmentation = checkpoint["data_augmentation"]
        normalize = checkpoint["normalize"]
        use_fp16 = checkpoint["use_fp16"]
        starting_step = checkpoint["step"]
        last_training_step_counter = checkpoint["training_step_counter"]
        decay_step = checkpoint.get("decay_step", decay_step)
        epoch_list = checkpoint["epoch_list"]
        starting_epoch = checkpoint["epoch"]+1
        batch_size_list = checkpoint["batch_size_list"]
        gen_update_interval = checkpoint["gen_update_interval"]
        r1_penalty_interval = checkpoint["r1_penalty_interval"]
        gamma = checkpoint["gamma"]
        # noise_addition = checkpoint["noise_addition"]
        
        if starting_epoch == epoch_list[starting_step]:
            starting_step += 1
            # also reset the initial epoch for the new step
            starting_epoch = 0

    # store training log for each step  
    all_training_losses = [[] for _ in range(max_steps)]
  
    print(f'StyleGAN1 Training on {dataset_name} in {experiment_date}')
        
    if resume:
        print(f'--Resume:                  {"N/A" if not resume else checkpoint_filename}'
            f'\n  --From Step:             {starting_step}'
            f'\n  --From Epoch:            {starting_epoch}'
            f'\n  --Checkpoint Path:       {checkpoint_path}'
            f'\n  --Last FID:              {checkpoint["FID"]}'
            f'\n  --Last IS:               {checkpoint["IS"][0]:.4f} ± {checkpoint["IS"][1]:.4f}')
          
    print(f'--Disc Param Count:          {sum([p.numel() for p in discriminator.parameters()]):,}')
    print(f'--Genr Param Count:          {sum([p.numel() for p in generator.parameters()]):,}')
    print(f'--MNetwork numlayers:        {generator.mn_num_layers}')    
    print(f'--style_mixing_prob:         {generator.style_mixing_prob}')
    print(f'--swap_adaIN_order:          {generator.swap_adaIN_order}')
    print(f'--Dataset:                   {dataset_name}-{split}')
    print(f'--DataAugmentation:          {data_augmentation}')
    print(f'--Normalize[-1,1]:           {normalize}')
    print(f'--Use Half-Precision:        {use_fp16}')
    print(f'--Discriminator LR:          {lr_d}')
    print(f'--Generator LR:              {lr_g}')
    print(f'--Max Step:                  {discriminator.max_steps}')
    print(f'--Decay Step:                {decay_step}')
    print(f'--Epochs:                    {epoch_list} ')
    print(f'--Batch-Sizes:               {batch_size_list} ')
    print(f'--ema_warmup_image_threshold:{ema_warmup_images_threshold:,} ')
    print(f'--ema_real_images_seen:      {ema_warmup_images_seen:,} ')
    print(f'--Generator update interval: {gen_update_interval}')
    print(f'--R1 Penalty Interval:       {r1_penalty_interval}')
    print(f'--Gama factor:               {gamma}')
    print(f'--PSI:                       {psi}')
    print(f'--gen_num_samples:           {gen_num_samples}')
    print(f'--Checkpoint Directory:      {weights_save_dir}')
    print(f'--Images Directory:          {images_save_dir}')
    
    if (use_fp16 and (lr_d>0.001 or lr_g>0.001)):
        print(f"⚠️ Warning! ⚠️ Large LR({lr_d},{lr_g}) for FP16 can lead to Nan! Decrease it for a stable training!")
        
    for step in range(starting_step, max_steps):
        if starting_epoch == 0:
            disc_optimizer = torch.optim.Adam(discriminator.parameters(),lr=lr_d, betas=betas_d)
            gen_optimizer = torch.optim.Adam(generator.parameters(), lr=lr_g, betas=betas_g)

        if step>starting_step:
            starting_epoch = 0
                
        batch_size = batch_size_list[step]
        epochs = epoch_list[step]
        # 4 is the lowest res so we want 8,16 etc
        res = 2**step*4
        train_loader = get_dataloader(dataset_name, split=split, resize_dims=(res,res),
                                      batch_size=batch_size, 
                                      data_augmentation=data_augmentation,
                                      normalize=normalize)
        num_batches = len(train_loader)
        interval = num_batches//2+1
        total_number_of_steps = epochs*num_batches
        fadein_steps = int(total_number_of_steps*0.5)
        training_step_counter = 0 if starting_epoch==0 else last_training_step_counter
        alpha=0
        
        #4,8,16,32,64,128,256
        if step>=decay_step:
            
            # e.g. 0.5 goes to 0.25 to 0.125 etc each time we halve the previous one
            # decay = 0.5**(step-2)
            # update: using a lambda/callable we can control the decay 
            # decrease it a certain amount or increase it
            decay = decay_func(step)
            
            disc_optimizer = torch.optim.Adam(discriminator.parameters(),lr=lr_d*decay, betas=betas_d)
            gen_optimizer = torch.optim.Adam(generator.parameters(),lr=lr_g*decay, betas=betas_g)

        current_lr_d = [g['lr'] for g in disc_optimizer.param_groups]
        current_lr_g = [g['lr'] for g in gen_optimizer.param_groups]

        print(f' Step: {step}/{max_steps} -> Training on [{res}x{res}]')
        print(f'  --Epochs:                      {epochs} ')
        print(f'  --BatchSize:                   {batch_size} ')
        print(f'  --Number of Batches:           {num_batches} ')
        print(f'  --Interval:                    {interval} ')
        print(f'  --R1-Interval:                 {r1_penalty_interval} ')
        print(f'  --Fade-in Steps:               {fadein_steps} ')
        print(f'  --Last training Step taken:    {training_step_counter} ')
        print(f'  --Current Discriminator LRs:   {current_lr_d}')
        print(f'  --Current Generator LRs:       {current_lr_g}')
        print(f'  --Current Discriminator Betas: {betas_d}')
        print(f'  --Current Generator Betas:     {betas_g}')

        
        for epoch in range(starting_epoch, epochs):
            discriminator.train()
            generator.train()

            losses = []
            # step_all_gps = []
            epoch_scores = []
            for i, (imgs_real, _) in enumerate(train_loader):
                
                imgs_real = imgs_real.to(device)
                # enable gradients for the images
                # our inputs are detached leaf nodes that dont have gradients by default
                # the autograd system doesnt track gradients for them, so to get gradients,
                # we need to enable their gradients
                imgs_real.requires_grad_(True)
                
                #scale input to [-1,1]
                #update:
                # its no more needed, the data are normalized 
                # to [-1,1] during dataloading process now
                # imgs_real = (2*imgs_real-1).to(device)
                
                # track how many real images the network has seen
                ema_warmup_images_seen += imgs_real.size(0)
                
                # I specifically dont use noise_addition here because the original stylegan1
                # didnt use any and we need to achieve the same result usin the same setup!
                # and as Im writing this we achieved it! so I comment this out!
                # if noise_addition:
                #     imgs_real += 0.05 * torch.randn_like(imgs_real)
                    
                with torch.amp.autocast(device_type="cuda", enabled=use_fp16):
                    # train discriminator/critic! 
                    # real image predictions
                    preds_real = discriminator(imgs_real, alpha, step)
                    # generate an image using generator 
                    z_vector = torch.randn((imgs_real.size(0), z_size)).to(device)
                    # we detach the imgs_fake so the discriminator cant use the gradients
                    # from the generator and quickly learn!
                    imgs_fake = generator(z_vector, alpha, step).detach()
                
                    # add noise to fake images as well(not needed for dcgan)
                    # if noise_addition:
                    #     imgs_fake += 0.05 * torch.randn_like(imgs_fake)
                
                    preds_fake = discriminator(imgs_fake, alpha, step)
                
                # calculate discrimiator loss out of real and fake losses
                # sidenote:
                #   for fp16 we need to calculate r1_penalty part
                #   in full precision mode(fp32) otherwise we'll get nans!
                #   we could also do it here , that is do:
                #   
                #   with torch.amp.autocast(device_type="cuda", enabled=False):
                #       disc_loss = discriminator_loss_stylegan1(preds_real.float(), imgs_real.float(), preds_fake.float(), gamma)
                #   
                #   and it works, but the issue is this will be slow! because it forces
                #   all of the operations involved in loss calculaton, such as 
                #   F.softplus() and mean() take place in fp32 while they can prefectly 
                #   run in fp16 and give us a boost! and it wont pose a numerical instability
                #   either. r1_penalty on the other hand needs to be in fp32 so we only
                #   run that portion in fp32. 
                disc_loss = discriminator_loss_stylegan1(preds_real, imgs_real, preds_fake, gamma)
                
                # for debugging purposes
                # if disc_real_mean is a lot larger than disc_fake_mean (e.g. 2.0 vs -2.0) 
                # then it means our discriminator is strong but if both are near the same
                # value and the loss is low then it means our discriminator is confused
                # or is over-regularized.
                disc_real_mean = preds_real.mean().item()
                disc_fake_mean = preds_fake.mean().item()
                
                # note:
                # first we need to detach the preds so our mean()/std() operations
                # are not recorded in computational graph. its not
                # part of training and we dont want to optimize anything
                # we just want to get some stats.
                preds_detached = preds_fake.detach()
                # we have the mean already so we just get std
                disc_fake_std = preds_detached.std()
                # now that we calculated the std for fakes, lets do that for real
                # we can now better compare them!
                disc_real_std = preds_real.detach().std()

                # store average scores for real and fake images
                epoch_scores.append((disc_real_mean, disc_fake_mean))
                
                # and optimize discrimnator 
                disc_optimizer.zero_grad()
                # disc_loss.backward()
                # disc_optimizer.step()
                # fp16
                scaler.scale(disc_loss).backward()
                
                # clip gradients >1 so we dont hit infs because of possible overflows!
                # if we face nans, it means underflow or an invalid opeation occured
                # (e.g. division by zero, sqrt(negative), log(negative), etc usually
                # when we underflow we may face nans as well)
                # 
                # quick reminder:
                #   overflow -> inf
                #   undeflow -> 0 or subnormal(see note in training) that eventually becomes 0
                #   1/0 -> inf (division by zero)
                #   0/0 -> nan (division by zero)
                #   inf-inf -> nan: 
                #   other invalid math ops like log(-1),sqrt(-1) etc -> nan
                #   unstable activations cane cause nans as well(e.g. if they are not
                #   normalized properly).
                #   division by zero in (custom) loss functions 
                #   or the grad scaler not handling the scaling well either not being enabled
                #   or even disabled! (we faced this when fp_16=false, scaler was always true!
                #   scaling the gradients incorrectly all the time) so its important to 
                #   make sure the proper flag to use fp16 is passed to scaler properly.also
                #   checking its value can give good insight into what it uses and whether we
                #   need to tune it ourselves at some point during training.
                #
                # we souldnt be needing this for discriminator cuz we arent doing anything
                # special in it!(we might though! because of r1-penalty part we might!)
                # 
                # update: 
                #   noticed clipping at 1 causes issues down the road and some implementations
                #   used 10 so I use that as well-the issue was sth else.
                # update2:
                #   to fight nans, we use a larger eps for adam. it works much better 
                # update3:
                #   r1_penalty needed to be calculaed in full precision. did that and fp16 is
                #   working ok now.
                #
                # scaler.unscale_(disc_optimizer)
                # nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1)
                
                scaler_out_d = scaler.step(disc_optimizer)
                            
                # now train genertor to create images that look real
                with torch.amp.autocast(device_type="cuda", enabled=use_fp16):
                    # todo put this in gen_update_interval check so we only run this
                    # when we want to optimize, but since currently im doing wgangp
                    # and its 1:1 that check is really not needed. also I check preds_fake
                    # in loss, so lets leave it be for now, until we get this working!
                    z_vector = torch.randn((imgs_real.size(0),z_size)).to(device)
                    fake_imgs = generator(z_vector, alpha, step)
                    preds_fake = discriminator(fake_imgs, alpha, step)
                
                    # generator loss
                    # swap loss! treat fake images as real images
                    gen_real_loss = generator_loss_stylegan1(preds_fake)

                    # optimize generator
                    # update generator with a delay, sylegan1 uses 1:1 update ratio
                    if (i+1)%gen_update_interval == 0:
                        gen_optimizer.zero_grad()
                        # gen_real_loss.backward()
                        # mn_grad_norm = torch.norm(next(generator.mapping_network.parameters()).grad)
                        # gen_optimizer.step()
                        
                        # fp16
                        scaler.scale(gen_real_loss).backward()
                        # update: 
                        #   monitoring gradient norm allows us to quickly know if something
                        #   is going wrong. the norm must not exceed 100! if that happens 
                        #   something is very wrong. a healthy norm is much lower than 100
                        #   (1~10) anything higher is a tell tale sign of gradient explosion
                        #   so to monitor for a gradient explosion, 100 is good choice imo.
                        #   when setting mapping_network lr, during fp16 we face nans!
                        #   that was fixed, it was because of extremeley small variance in 
                        #   equalizedlinear I believe. 
                        # update:
                        #   we face nans in fp16, this time after trying small batches=16 with
                        #   lr=0.003. larger batchsizes didnt cause any issues but when I 
                        #   sstarted using small batchsizes, I started to get nans again!
                        #   I also noticed we got large gradient norms early on. that means
                        #   we have gradient explosion!(reported norm for mapping_network
                        #   was above 100!(i.e. 119)!). thinking about it, it makes sense
                        #   we introcude more unstable/noisy gradients by smaller batchsize,
                        #   so we are more susceptible to nans (i.e. with smaller batchsizes
                        #   we get much more noiseier and higher-variance estimate of the
                        #   actual/true gradients. that is we can get an unsually/abnormally
                        #   large gradients which can then result in exploding gradients
                        #   if used with large enough lr!) that is, when we have exploding
                        #   gradients, it results in a large, so large numbers the 
                        #   fp16 cant represent properly and it overflows to inf(fp16 can
                        #   only represent up to 65504! anything larger and we get inf!)
                        #   now if we have sth like inf-inf, or 0 * inf, or anything*inf 
                        #   it will result in a nan. 
                        #   (we dont do 0 * sth, by that I mean we might have an underflow 
                        #   happening somewhere aswell, resulting to loss of precision and
                        #   ending up 0! which then gets multiplied by another bad result 
                        #   from exploding gradient (inf) and ultimately get nan! 
                        #   also our learning rate may be too large for that batchsize
                        #   and be the reason why it fails!(large lr * large gradients->inf)
                        # update: 
                        #   it was the learning rate, reduced it to 0.001 and we no more 
                        #   got nans! (this probably went sth like this, initially when we 
                        #   start training our weights are random, so the first few predictions
                        #   are bound to be wrong, this results in a large loss and consequently
                        #   a large gradient. this goes on for a few epochs and the network over
                        #   corrects and cant settle on a good spot,(takes large leaps in the 
                        #   loss landscape(remember zig-zag movement!)) making more wrong 
                        #   predictions so it exacerbates the issue further, we also use a 
                        #   small batch, so some samples can be too off the rails, so they get
                        #   abnormally large gradients, our large learning rate isnt helping
                        #   at all and leads us down the abyss. we get huge gradients, huge 
                        #   weight updates and after a few epochs boom! two huge number's 
                        #   multiplication e.g. goes over 65504 and we get inf!
                        #  update: 
                        #   we also had to run r1_penalty in fp32 otherwise we'd get nans
                        #   after some epochs.
                        #
                        # monitor the norm (one of the params is enough)
                        mn_grad_norm = torch.norm(next(generator.mapping_network.parameters()).grad)
                         
                        # tried gradient clipping(1) to prevent explosion but didnt help
                        # 
                        # update2:
                        #   stylegan1 didnt use any gradient clipping, they only trained in fp32
                        #   other implementations that did use this, either like us used fp16
                        #   or simply did this for stable training (in which case they used a
                        #   larger norm like 10. we dont do that, and after our latest fixes 
                        #   it trains just fine without it)
                        # update3:
                        #   when we use small batchsizes, like 16, whith fp16 enabled, we get
                        #   nans! that can be caused by undeflow/overflow due to fp16 small range
                        #   we clip the gradients to see if it prevents the explosion
                        #   we might have other issues (i.e. loss calculation may need to be
                        #   done in fp32 to not get nans/inf itself!)
                        # update4:
                        #   it was the large lr with small batchsize. since small batchsize gave
                        #   noisier and higher variance estimate of the actucal gradients 
                        #   (some are unusually large) when model makes the early wrong 
                        #   predictions, with huge gradients, we get huge weight updates, 
                        #   and therefore, after a few epochs, weights values grow large,
                        #   result in larger norms. add in large lr and the fact that if
                        #   we go >65504 boom! we get inf!
                        # 
                        # norm of 1 to 10 is considered healthy, anything drastically higher
                        # is a bad sign!(i.e exploding gradients)(note in our current setup
                        # this is the case, for other types of networks, much deeperones, etc
                        # the norm can be different but for us right now this is the case)
                        scaler.unscale_(gen_optimizer)
                        nn.utils.clip_grad_norm_(generator.parameters(), max_norm=10)
                        
                        scaler_out_g = scaler.step(gen_optimizer)
                        
                        if mn_grad_norm>100:
                            print(f'Warning! mapping_network_grad_norm={mn_grad_norm.item():.4f}')# '{scaler_out_g=}')
                        
                        if ema_warmup_images_seen < ema_warmup_images_threshold:
                            ema_generator.load_state_dict(generator.state_dict())
                        else:
                            update_ema_generator(generator, ema_generator)
                
                # update scaler only once
                # we dont update the scale for individual models,
                # we only update it once when we ran the optimization
                # step for all of them. otherwise, the scaler will
                # be in a constant loop of increasing/decreasing the
                # scale and cause instability during training. instead
                # we do it once, if a model needs larger scale it will take
                # care of it otherwise it will use the same scale until one
                # requires otherwise. its more efficient
                scaler.update()
                    
                status_r = get_status(disc_real_mean, higher_is_better=True)
                status_f = get_status(disc_fake_mean, higher_is_better=False)
                status_o = get_overall_status(disc_real_mean, disc_fake_mean)
                # included the std for real and fake so it makes it much clearer 
                # to compare and see where we are standing   
                d_real_stat_str = f"D_real_avg: {status_r} {disc_real_mean:+.4f} ± {disc_real_std:+.4f} 📈"
                d_fake_stat_str = f"D_fake_avg: {status_f} {disc_fake_mean:+.4f} ± {disc_fake_std:+.4f} 📉"

                if (i+1)%interval==0:
                    print(f'[{res}x{res}][Epoch {epoch}/{epochs} | Iter: {i}/{len(train_loader)}] Disc Loss: {disc_loss:.4f} | Gen Loss: {gen_real_loss:.4f}')
                    print(f" -- {status_o} Batch-{i}:  {d_real_stat_str}| {d_fake_stat_str}")
                    
                losses.append((disc_loss.item(), gen_real_loss.item()))
                
                if training_step_counter<fadein_steps:
                    alpha = training_step_counter/fadein_steps
                else:
                    alpha = 1
                # update the training steps
                training_step_counter += 1
        
            d_loss_mean = np.mean(np.array(losses)[:,0])
            g_loss_mean = np.mean(np.array(losses)[:,1])

            # always update the last step per epoch 
            all_training_losses[step].append((d_loss_mean, g_loss_mean))
            
            # real
            average_score_real_mean = np.mean(np.array(epoch_scores)[:,0])
            average_score_real_std = np.mean(np.array(epoch_scores)[:,0])
            # fake
            average_score_fake_mean = np.mean(np.array(epoch_scores)[:,1])
            average_score_fake_std = np.std(np.array(epoch_scores)[:,1])
            
            if quick_and_noisy_IS_FID:
                IS_score = metric.compute_IS(imgs_fake)
                FID_score = metric.compute_FID(imgs_real, imgs_fake)
            else:
                IS_score, FID_score = get_IS_FID_score(metric, generator, train_loader, dataset_name, split, alpha=alpha, step=step)

            status_avg_r = get_status(average_score_real_mean, higher_is_better=True)
            status_avg_f = get_status(average_score_fake_mean, higher_is_better=False)
            status_avg_o = get_overall_status(average_score_real_mean,
                                              average_score_fake_mean,
                                              IS_score=IS_score,
                                              min_mu=1.2)
           
            real_stats_avg_str = f"D_real_avg: {status_avg_r} {average_score_real_mean:>+.4f} ± {average_score_real_std:<+.4f} 📈"
            fake_stats_avg_str = f"D_fake_avg: {status_avg_f} {average_score_fake_mean:>+.4f} ± {average_score_fake_std:<+.4f} 📉"

            dloss_avg_str = f"DLoss(Avg): {d_loss_mean:.4f}"
            gloss_avg_str = f"GLoss(Avg): {g_loss_mean:.4f}"

            is_score_str = f"IS: {IS_score[0]:.4f} ± {IS_score[1]:.4f})"
            fid_score_str = f"FID: {FID_score:.2f}"
            
            # mintor gradient norm for mapping_network to better
            # tune hyper parameters, epsecially when it comes to fp16!
            mn_grad_norm_str = f"GradNorm: {mn_grad_norm:.2f}"
            
            summary = f"{dloss_avg_str} | {gloss_avg_str} | {is_score_str} | {fid_score_str} | {mn_grad_norm_str}"
            
            print(f" -- {status_o} Last Batch : {d_real_stat_str} | {d_fake_stat_str}")
            print(f" -- {status_avg_o} Epoch's Avg: {real_stats_avg_str} | {fake_stats_avg_str}")
            print(f'[{res}x{res}][Epoch {epoch}/{epochs}] {summary}')
            
            #save model weights at each epoch
            checkpoint_dir = f"{weights_save_dir}/{current_experiment_name}"
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            state_dicts = {"disc_state_dict":discriminator.state_dict(),
                           "gen_state_dict":generator.state_dict(),
                           "gen_ema_state_dict":ema_generator.state_dict() if use_ema_inference else None,
                           "disc_optimizer":disc_optimizer.state_dict(),
                           "gen_optimizer":gen_optimizer.state_dict(),
                           "scaler_state_dict":scaler.state_dict()
                           }
            settings = {
                        "z_size":generator.z_size,
                        "w_size":generator.w_size,
                        "mn_nlayer":generator.mn_num_layers,
                        "channels_d":discriminator.channels,
                        "channels_g":generator.channels,
                        "style_mixing_prob":generator.style_mixing_prob,
                        "swap_adaIN_order":generator.swap_adaIN_order,
                        "ema_w_beta":generator.ema_w_beta,
                        "use_fp16":use_fp16,
                        "lr_d":lr_d,
                        "lr_g":lr_g,
                        "max_steps":discriminator.max_steps,
                        "decay_step":decay_step,
                        "step":step,
                        "training_step_counter":training_step_counter,
                        "ema_warmup_images_threshold":ema_warmup_images_threshold,
                        "ema_warmup_images_seen":ema_warmup_images_seen,
                        "epoch":epoch,
                        "epoch_list":epoch_list,
                        "batch_size_list":batch_size_list,
                        "gamma":gamma,
                        "psi":psi,
                        "gen_update_interval":gen_update_interval,
                        "r1_penalty_interval":r1_penalty_interval,
                        "FID":FID_score,
                        "IS":IS_score,
                        "dataset_name":dataset_name,
                        "data_augmentation":data_augmentation,
                        "normalize":normalize,
                        "split":split,
                    }
            loss_dicts = {"d_loss_mean":d_loss_mean,
                          "g_loss_mean":g_loss_mean,
                          "all_training_losses":all_training_losses
                         }
            # all_settings = {**state_dicts,**settings}
            all_settings = state_dicts | settings | loss_dicts
            torch.save(all_settings, f"{checkpoint_dir}/checkpoint_step_{step}_{experiment_date}.ckpt")
            
            # generate some images mid training to evaluate our model's performance 
            with torch.no_grad():
                gen = ema_generator.eval() if use_ema_inference else generator.eval()
                
                generated_images = gen(fixed_z, alpha, step, psi=psi)
                
                ema_marker_str = "[EMA]_" if use_ema_inference else ""
                loss_str = f"(dLoss:{d_loss_mean:.6f} | gLoss:{g_loss_mean:.6f}"
                lrs_str = f"{current_lr_d[0]:.0e},{current_lr_g[0]:.0e}"
                title_str = f"Step {step} [{res}x{res}, α={alpha:.2f}] @ Epoch {epoch} FID:{FID_score:.2f} {loss_str} [{lrs_str}]"
                img_store_dir_path = f'{images_save_dir}/stylegan1/{dataset_name}_{experiment_date}'
                img_filename = f'{ema_marker_str}step_{step}_{res}x{res}_epoch_{epoch}.jpg'
                save_path= os.path.join(img_store_dir_path, img_filename)
                                
                display_images(generated_images, 
                               cols=gen_num_samples//8,
                               title=f"{ema_marker_str}{title_str}",
                               unnormalize=True,
                               save_path=save_path,
                               figsize=(16,8))
                
                # save the original images only when ema is enable, 
                # otherwise its already being saved/displayed
                if keep_raw_generations and use_ema_inference:
                    generated_images = generator(fixed_z, alpha, step, psi=psi)
                    display_images(generated_images, 
                                   cols=gen_num_samples//8,
                                   title=title_str,
                                   unnormalize=True,
                                   save_path=save_path.replace(ema_marker_str,""),
                                   figsize=(16,8))
                    
                # save the settings that achieved this aswell
                settings_path = os.path.join(img_store_dir_path,'settings.yaml')
                with open(settings_path, "w") as f:
                    yaml.dump(settings,f, sort_keys=False)

    print("SttyleGAN1 training is complete!")

def get_dataset_size(name,split='train'):
    dl = get_dataloader(name,split,batch_size=1)
    return len(dl)

#%% training stylegan1
print(f'Training StyleGAN1')
gamma=10#10
# cifar10 is a lot harder than celeba
# simply becasue its a small multiclass dataset
# that has only 50k training samples. basically
# 10k for every class which is very low to get decent
# result. on  the other hand celeba and FFHQ datasets 
# are simply human faces and this makes it extremely easy
# to get it right compared to cifar10! they also have much 
# larger number of samples!
# so to test and evalualte we always try celeba first
# and then cifar10 if we like
dataset_name = 'celeba'
split = 'train'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

use_fp16=False

# original paper uses 512
z_size = 512
w_size = 512
# 7 means 4x4 up to 256x256
max_steps = 7#3 if dataset_name=="cifar10" else 7

# batchsize extremely matters, the larger the batchsize the
# better the performance. 
# see https://github.com/NVlabs/stylegan2-ada-pytorch?tab=readme-ov-file#expected-training-time
# it gives a pretty good insight into the importantce of batchsize 
# (also sidenote: see the the lowest FID is achieved in smaller res
# the larger the res, the larger the FID score gets)
# log-vram usage 
# fp32: with Batch_sizes=[128,128,128,64,32,16] and w_size=z_size=512
#  and channels=[512,512,512,512,256,128] (23vs25m) the vram usage is
#  up to 6799MB @ 32x32 and up 9637MB @ 64x64. since im using my integrated
#  GPU for display, I can easily train up to 10236MB! (xorg takes 4MB!)
#  With channels=[512,256,128,64,32,16,8] (11mvs11m) the vram usage is
#  up 3015MB @ 64x64 @ 128x128 its 3565MB.
# 
# update:
#  with batchsize like 16/32 we face exploding gradients when we reach 8x8
#  both at fp16 and fp32!
#  I got this issue for both celeba and ffhq dataset. this shouldnt happen 
#  because we are using equalizedconv/equzlizedlinear so the learning rate
#  should be equalized but for some reason this happens. even clipping the
#  gradients to 1 and even 10 doesnt fix the issue. (larger batchsize doesnt
#  exibit this issue though!). this is probably a bug in my code honestly 
#  but for now im not going to bother! im too exausted right now!
# 
# update: 
#  it was in fact a bug in equalizedlinear. it would incorrectly make
#  the variance extremely low, resuling in very small numbers, which would then
#  be compounded and result in huge gradients in adam optimization and result
#  in gradient explosions and this happens at both fp32 and fp16.
#  (i.e. that is since adam optimizer sees extremely small gradients, it
#  increases its adaptive learning rate, but when the activations get larger
#  (because of multiple layers in between all affecting the output) this leads
#  to gradient explosion even though our initial numebrs are extremely small
#  now if we use a large enough batch, the gradients will be much smoother 
#  and uniform, and we wouldnt face this issue, we would only suffer from 
#  slow convergence (and also unstable training because discriminator trains
#  faster than generator because of this issue). 
#  however, when we use a small batchsize, it means we get noisier gradients,
#  and therefore the adam optimzier's adaptive statistics also become less 
#  stable/accurate/useful. therefore whenever the parameters magnitudes 
#  and adam's effective learning rates are mismatched, we can expect all hell breaking loose!
#  the instability will be compounded and reveal itslef in the form of gradient explosions.
#  gradient clipping also didnt help until I fixed the issue.
#
# update:
#  after that fix I reran the experiement in fp32 with bs=16
#  autocast machinery was there, and it resulted in gradient explosion!
#  with fp16 I would get nans! after commenting the autocast machinery
#  retraining with fp32 and bs=16, the issue was gone. 
#  note that I had used use_fp16=False, yet still I'd face gradient explosion
#  for some reason! only after commenting the code relaetd to fp16/half preciison training
#  the issue was gone! so its 100% related to autocast (could be a pytorch bug
#  or my fp16 implementation!)
# update:
#  found the bug! the scaler was always True by default, so when I disabled the fp16
#  the scaler would still scale the gradients and lead to explosion!
#  the fp16 getting nans is next to solve (probably related to loss/r1penalty)
#  I guess we need to calculate it in fp32 probably but lets see
# update: 
#  it was the large lr(0.003) with small batchsize that caused the fp16 nans.
#  lowering the lr down to 0.001 fixed the issue. didnt need to do gradient clipping
#  but enabled it anyway beacuse smaller batchsize can yield abnormally huge gradients
#  (its very noisy) and eventually build up larger weight norm(because of large updates!)
#  until at some point we hit the 65504 limit of fp16 and overflow! I explained it
#  in details severla times throughout the code! so I dont bother repeating it here again!
# 
# note if using fp16 and smaller batchsize(like 16),
# use smaller lr! otherwise might get nans!
#
if use_fp16:    #res from 4 up to 256 (7 steps)
    BATCH_SIZES = [128,128,128,128,64,32,16]
else:
    BATCH_SIZES = [128,128,128,64,32,16,8]#,4,2]

# sidenote: 
#  in pro/stylegans usually kimage is used as the metric for
#  how long the training should go on. each epoch means one
#  round of full dataset consumption during training, but since
#  this can mean different number of images for different datasets
#  and using data-augmentations this can further get nuisansed for 
#  every dataset the authors talk about k real images (thousand images)
#  the discriminator needs to see. its a much better metric to keep track
#  of if you think about it.
#  they mentioned they used 1600 kimages per resolution. 800kimages for
#  fadein phase and 800kimages for stabilization (i.e. in total 1.6million images 
#  per resolution which if we use batchsize=128 its 12500 iterations
#  or for cifar10 it would be 32 epochs for each res! 
#  for stylegan1 this number is 12000kimage to 25000kimage (12m to 25m)
#  depending on the dataset which is 240 epochs in total to get the best results. 
#  (some other datasets like lsun seems to have been trained with
#  40000kimage to 70000kimages in total. i.e. for the subsets like bedrooms 
#  at 256x256 70m images and for cars at 512x384 46m images were used)
#  they also do it a bit more efficiently than us, they switch to the 
#  next res when FID plateues! The FID itself is not calculated every epoch
#  they calculate it after some kimages seen. we havent done this
#  so far, but im writing this for future references. todo for later! 
#  Having done all of this, note that I could not get a decent outcome until
#  I swapped adain/lrelu. only then I got good results! so this is where my
#  implementation differs the most.
#  for celeba this is what I used, use more to get better
# EPOCHS = [10,10,10,20,40,40,40]
# EPOCHS = [10,10,20,30,50,60,70]
# for cifar10, since our max res is 32x32, we need to allocate 
# more epochs especially to the 32x32 res. even after doing that
# the FID may not change, but the image quality 100% improves, however
# the images may be malformed but sharp.there are some classes that
# are better than others like horses, some cars, fish, but cars, or
# more complex things are malformed usually(at least based on my experiments
# with limited epochs) and this is directly related to small number of images
# for each class and how many epochs the model is trained onviously!
# also the droplet effect is (water smudge effects in images) are expected
# this is the stylegan1 issue which will be fixed in stylegan2!
# EPOCHS = [10,10,20,50,30,60,70]
# kimages = int(math.ceil(1_600_000 / get_dataset_size(dataset_name,split)))
# print(f'{kimages=:,}')
# 
# update:
#  I ran a lot of tests, when I fixed my bugs, I noticed with much number of 
#  epochs we can achieve pretty good results. so while kimages are a good 
#  measure to have, the good old epochs would do well as well. 
#  also the number of epochs needless to say is directly dependant on the model size
#  the larger the model, the less epochs per res is needed and vice versa. 
#  these are the new configs I came up with (initially used one of the pytorch impls)
#  ffhq128 [8,16,32,32,64,64], celeba [4,8,16,16,32,48]
#  I got great results with [8,16,32,32,64,64] with both 23/25m and 11m models
#  see debug logs for more information
#  for cifar10 we use more epochs for 32x32 [16,24,48,64,64,64]
EPOCHS = [8,16,32,32,64,64]# [4,8,16,16,32,48]

gen_update_interval = 1
# no where in the paper or official code they apply
# r1_penalty after 16 iterations each time! so I 
# instead use 1 here to match the official impl
# I completely removed this and everything works great
r1_penalty_interval = 1#1#16
style_mixing_prob = 0.9
# truncation rate
psi = 0.7
# whether to use original order or not
swap_adaIN_order = False
# I've got my best results with [512,512,512,512,256,128,64] for both
# discriminator and generator(in ffhq128) but it takes ~2 hours to train
# a single epoch in 64x64!!(17min for 32x32) the models become 23m/25m.
# for the record, using [512,256,128,64,32,16,8] for both models, they become 11m/11m
# and 32x32 takes only 5 mins! and 64x64 takes 15mins! and results are gorgeous as well!
# at least for ffhq128!
channels_d = [512,256,128,64,32,16,8]
channels_g = [512,256,128,64,32,16,8]
#discriminator
discriminator_stylegan1 = DiscriminatorStyleGAN1(max_steps,channels=channels_d)
discriminator_stylegan1 = discriminator_stylegan1.to(device)
#generator
mn_nlayer = 8#8
generator_stylegan1 = GeneratorStyleGAN1(z_size, w_size, max_steps, mn_nlayer,
                                         channels_g, style_mixing_prob,
                                         swap_adaIN_order=swap_adaIN_order)

generator_stylegan1 = generator_stylegan1.to(device)

betas = [0, 0.99]
# 0.003 works great for fp32 but
# for fp16 use smaller lrs like 0.001 otherwise we get nans!
lr_d = 0.003 #0.0015 onlyfor res>64, 0.002 for res>256 and 0.003 for res> 512
# log:
# -I faced mode collapse in 64x64, then I noticed the mapping network lr 
# was too low(1.5e-7!). this is around 10,000 times smaller than the rest
# of the generator parameters. this happened because of a bug in training_loop
# where at each step I had incorrectly kept decaying the lr by 0.01 when I was
# creating new optimizers all while the learning rate was already set before
# training by the initial optimizer).
# 
# -In fp16 using lr=1.5e-5 resulted in nans! its apparently too large! to fix
# this we can either use a larger eps for adam optimizer and or also use gradient 
# clipping to prevent this. I first try the eps and if it doesnt work I also 
# clip gradienst becasue default eps(1e-8) is too small for fp16 anyway.
#  --update: larger eps so far did the trick! and we are getting better images
#    so far in 16x16 res (used to be very bad, but now they look better though
#    they are still extremly low res (16x16)) 
#  -- ok at 64x64 we faced mode collpase! disc had much lower loss(0.4) vs gen(3.0).
#     paper uses 100x smaller lr for mapping network because it has 8 layers!
#     the more layers the more unstability! so they multiply it by 0.01! I did that
#     im not sure why disc is overpowering generator!
# -update: it was a bug in equalizedlinear see debug log
# -update: the fp16 can use lr=0.003 just fine given large batches, only if we go
#  smaller batchsize(like 16/32) we face nans, in which case using lr=0.001 solves
#  the issue. 
#  the initial comment (1.5e-5 being too large) is simply incorrect beause
#  it was caused by several bugs in the code at the time. one was buggy equalizedlinear
#  and the other was a separate lr for mapping network that would make effective lr
#  100x smaller! compounded by another bug, shariking that another 100x at every step
#  during training! so the gradients were too tiny. using eps 1e-8 would therefore 
#  not help as it was too small for fp16, leading to underflow and ultimaly
#  0! remember the adams update formula is w = w - lr * m_hat / (sqrt(v_hat)+eps)
#  so not only v_hat goes toward 0(very small), the eps thats being added is also practically 0,
#  so it can underflow and become exactly 0, making the denominator practically zero
#  and lead to division by zero and thus inf which in subsequent operations can 
#  lead to nan! if it doesnt underflow and become 0, its a tiny number leading to
#  large effective learnng rate, leading to gradient explosions and ultimatly nan!
#  increasing the eps masked the first issue, made it so we dont face nans
#  asap but the training was flawed until those bugs were fixed. moreover, later using
#  smaller batchsizes, lead to nans again, because of noisier gradients which ultimately
#  required smaller lr (and optionally gradient clipping) to safely prevent the nans
# 
# we can use much larger lr than 0.0015 
# and get a much faster convergence
# 0.003 works great with large batches!
# however use smaller lrs like 0.001 when doing fp16!
lr_g = 0.003#0.0015 is used for res>64

# update:
#  made the lr 1000x larger than the normal case
#  this was the first thing I did when I got 
#  nans during fp16 training with lr 1.5e-5 for
#  mapping network, didnt work!
#  note:with gradient clipping nan issue is also gone but
#  by using larger eps(1e-5) the loss decreases twice as much
#  compared to the default eps(1e-8). the result is much better
#  with no gradient clipping with eps=1e-5
# 
# 
# sidenote: informative but not that relavent really!
# I faced a bug in fp16/fp32 where I would get nans in fp16 and if I disabled
# it, I'd face gradient explosion, I couldnt figure out why I was facing this
# I though my stylegan implementation must have an issue, for the time being
# I disabled the autocast related code and later found a bug in equalizedlinear
# after I fixed that, trained successfully, I came back to fp16 weirdness and thought
# that bug must have caused those nans/explosions and wrote this explanation. 
# that wasnt right, the issue was entirely something else but the information 
# about fp16 I wrote here while trying to fix it is too good to be removed, 
# so I leave it here for a reminder/refresher for myself:
# 
# the old explanation:
# -I think the issue was caused because of equalizedlinear bug and mapping_network using 
# a second 100x smaller lr! and adam optimizer's adaptive lr getting large as a
# result(since the denominator sqrt(v)+eps will be dominated by the eps (the adam's 
# update formula was w = w - lr * m_hat / (sqrt(v_hat)+eps)
# so obviously if v (and thus sqrt(v)) goes to zero because of the tiny gradients then
# the denominator practically becomes the eps! thus smaller eps -> smaller denominator
# and smaller denominator leads to larger updates which in turn can lead to instablity 
# in training and nans or gradient explosions.
# 
# in fp16 this is even worse because the smallest positive denormal/subnormal number that
# can be represented is 5.96e-8 (or 2^-24) and eps is 1e-8(i.e. 2^-26) which is much smaller
# than 5.96e-8 (1e-8 < 5.96e-8) so it underflows to 0 and we face divide by zero and thus nans!
# larger eps therefore doesnt lead to that.
# for example consider 1e-5 which we used for fp16 to get rid of nans during training, it 
# can be represented as 2^-17 which is roughly 7.6e-6) and will therefore be safe from 
# underflow.
# why do we care about subnormals in fp16 and not normal numbers? 
# tldr: 
# because the range of normal number is very limited in fp16, and if we were to use normal
# numbers, we would lose all numbers smaller than it to 0! denormals are there to prevent
# that, instead of just setting all smaller numbers to 0, fewer significant bits are used
# to the precision of the number is decreased, this way we can show/represent much smaller
# numbers that we normally would be able usin just normal numbers in floating point systems.
#
# more explanation:
# because in practice, during training, if a value like our optimizers eps here,
# becomes smaller than the smallest denormal/subnormal(e.g. 1e-8 <5.96e-8) it will underflow
# to zero and result in division by zero and nans!
# fp32 has different limit its smallest normal is around -1.175e-38 and smallest denormla
# is 1.4e-45.
# in floating point systems(like IEEE 754 thats used for fp16/32), normal numbers have a 
# minimum positive magnitude (e.g. ~6.1x10^-5 in fp16).
# anything smaller than that cant be represented as normal because then the exponent cant
# go lower while keeping the implicit leading 1 in mantisa.
# therefore without denormals, if a calculation produced a result below this threshold 
# (e.g. sth like 0.001x0.001=10^-6) it would have underflowed directly to zero! 
# this is called "flush to zero"(FTZ) by the way.
# this is obviously bad because it creates a huge gap between the smallest normal and zero
# and if we clamp all numbers in between to zero we lose a huge amount of information. 
# we have seen this issue first hand in training, that how lack of precision causes all sorts 
# of issues from complete failure to extremely slow convergence.
# (also properties like associativiy break (a+b)+c will not be equal to a+(b+c) if small 
# values get zeroed out unevenly)
# therefore denormals are there to fill that gap and allow the mantisa to have leading zeros
# (i.e. no implicit 1 anymore) which essentially, extends the range down to much smaller
# values (e.g. 5.96x10^-8 as we saw in fp16).
# this creates a gradual underflow, as numbers get smaller, precision is decreased smoothly
# (i.e. we use fewer significant bits to represent numbers) but we dont lose the value entirely
# until we truly hit zero!
#
# quicknote:
# 1e-5 is 10^-5 and to see what 2^x can represent (since floating points are stored
# as ±(1+m)×2ᵉ where m is the fractional part(i.e.mantisa) and e is exponent obviously
# therefore powers of 2 are exactly represenable without any rounding if they are in limit
# our 1e-5 is a decimal number, and not exactly power of 2, the closest number to one is
# 2^-17 which in decimal form is 7.6e-6. (to calculate that we simply need to take its log₂
# so log₂(10^-5) is -5*log₂(10). log₂(10) is ~3.321928 so -5*3.321928=-16.60964 or ~-17.
# so 2^-16.60964 or simply 2^-17 (7.6e-6) is the number that represents 1e-5) 
# 
# update:
# my fp16 is broken 100%. even after the equalizedlinear fix, enabling fp16 causes nans
# if I set use_fp16=False, and train with bs=16, I face gradient explosion!
# lowering the lrs down to 0.0001 doesnt help! after commenting out autocast
# related code, the gradient explosion stopped even with small batchsize=16
# it could be a bug in pytorch or my own!
# 
# update:
# found the bug! I had forgotton to pass the use_fp16 to GradScaler()
# aswell and the scaler was always enabled, so when I disabled the fp16
# the scaler would still scale the gradients and lead to explosion!
# 
eps = 1e-5 if use_fp16 else 1e-8
# no need to decay now!
decay_step = 7#4#3#2
# 0:4,1:8,2:16,3:32,4:64,5:128,6:256
# increase the lr after step3
def decay_func(step):
    if step==4:
        return 1.5
    elif step>=5:
        return 2
    else:
        return 1
# log:
# at epoch 2 of 8x8 it suddenly goes all solid grays
# up to that point (i.e. all 4x4s, up until epoch 2 of 8x8 it looked normal!
# so update ema needs some work!
use_ema_inference = True

disc_optimizer = torch.optim.Adam(discriminator_stylegan1.parameters(), lr=lr_d, betas=betas)
# mapping network is already using 100 times smaller learnng rate
# through EqualizedLinear! 
# so we are not decreasing it 100x more! by 0.01 again! This damn thing
# was the reason I faced so many issues in 32x32 res and higher!(generator would 
# be dominated by the discriminator becasue gmapping was extremely slow to update!
# especially in higher res that needed more details!)
# mapping_params = list(generator_stylegan1.mapping_network.parameters())
# gen_other_params = [p for p in generator_stylegan1.parameters() if p not in set(mapping_params)]
# gen_optimizer = torch.optim.Adam([{'params':mapping_params,'lr':lr_g},
#                                   {'params':gen_other_params,'lr':lr_g}
#                                  ], betas=betas, eps=eps)
gen_optimizer = torch.optim.Adam(generator_stylegan1.parameters(), lr=lr_g, betas=betas, eps=eps)

training_loop_stylegan(discriminator_stylegan1,
                     generator_stylegan1, 
                     disc_optimizer=disc_optimizer,
                     gen_optimizer=gen_optimizer, 
                     epoch_list=EPOCHS, 
                     batch_size_list=BATCH_SIZES,
                     gen_update_interval=gen_update_interval,
                     r1_penalty_interval=r1_penalty_interval,
                     dataset_name=dataset_name,
                     split=split,
                     data_augmentation=True,
                     gamma=gamma,
                     psi=psi,
                     use_fp16=use_fp16,
                     device=device,
                     resume=False,
                     use_ema_inference=use_ema_inference,
                     ema_warmup_images_threshold=1_000_000,
                     keep_raw_generations=True,
                     quick_and_noisy_IS_FID=False,
                    #  checkpoint_path="./weights/gan/stylegan1_ffhq_20251107210907/checkpoint_step_2_20251107210907.ckpt",
                     decay_step=decay_step,
                     decay_func=decay_func)

# - cleanup comments/explanations in styleconvblock/generator/training section
# - test latent space
#  
# todo: remember to include dataset sizes e.g. celeba_hq is only 30K highres
# celeba is around 200k, and ffhq_128 is around 70k. we have all of them so 
# we can test them and hopefully get decent results (after we got the right
# hyperparameters!)
#

#%% change_checkpoint_fields
def change_checkpoint_fields(checkpoint_path, updated_fields, display_fields=True, save=False):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    if display_fields:
        print(f'Read checkpoint:')
        for k,v in checkpoint.items():
            if not isinstance(v,dict):
                print(f'{k:<15} {v}')
            elif "param_groups" in v.keys():
                print(f'{k:<15} {v["param_groups"]}')

    for k,v in updated_fields.items():
        checkpoint[k] = v
    
    if display_fields:
        print('#'*30)
        print('Updated Checkpoint:')
        for k,v in checkpoint.items():
            if not isinstance(v,dict):
                print(f'{k:<15} {v}')
            elif "param_groups" in v.keys():
                print(f'{k:<15} {v["param_groups"]}')
    
    if save:
        torch.save(checkpoint,checkpoint_path)
        print(f'new checkpoint saved to disk!')
    
    return checkpoint
    
# change some paratemers during experimental resumes!(like add more epochs, change lambda_factor, etc)
# checkpoint_path ="./weights/gan/stylegan1_ffhq_20251113102258/checkpoint_step_4_20251113102258.ckpt"
# updated_fields = {"data_augmentation":True,
#                   "normalize":True}
# chkpnt = change_checkpoint_fields(checkpoint_path, updated_fields,
#                                   display_fields=True, 
#                                   save=False)
#%% load_checkpoints
def load_checkpoints(checkpoint_path, device="cuda"):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    dataset_name = checkpoint["dataset_name"]
    last_step = checkpoint["step"]
    FID_score = checkpoint["FID"]

    z_size = checkpoint["z_size"]
    w_size = checkpoint["w_size"]
    max_steps = checkpoint["max_steps"]
    mn_layers = checkpoint["mn_nlayer"]
    channels = checkpoint["channels_g"]
    style_mix = checkpoint["style_mixing_prob"]
    swap_ordr = checkpoint["swap_adaIN_order"]
    ema_wb = checkpoint["ema_w_beta"]

    generator_st1 = GeneratorStyleGAN1(z_size, w_size,max_steps,mn_layers,channels, style_mix, ema_wb, swap_ordr)
    generator_st1.load_state_dict(checkpoint["gen_state_dict"])
    generator_st1 = generator_st1.eval()
    generator_st1 = generator_st1.to(device)

    print(f'Loading checkpoint for {dataset_name} @ step={last_step} FID={FID_score}')
    return generator_st1, last_step

device = 'cuda'
num_samples=36
#cifar10
# checkpoint_path = './weights/gan/stylegan1_cifar10_20251114082051/checkpoint_step_3_20251114082051.ckpt'
#not as good as previous one, but interesting nonethelss.
# checkpoint_path = './weights/gan/stylegan1_cifar10_20251115223041/checkpoint_step_3_20251115223041.ckpt'
# 23/25m model-not trained as long! the next one
# checkpoint_path = './weights/gan/stylegan1_ffhq_20251109035618/checkpoint_step_4_20251109035618.ckpt'
# gives the best results so far (11/11m models)
checkpoint_path = './weights/gan/stylegan1_ffhq_20251110095733/checkpoint_step_4_20251110095733.ckpt'
generator_style1, last_step = load_checkpoints(checkpoint_path, device='cuda')

z = torch.randn((num_samples, generator_style1.z_size), device=device)
dim = 2<<last_step+1
res=f"{dim}x{dim}"
with torch.no_grad():
    imgs = generator_style1(z, alpha=1, step=last_step, psi=0.7)
    display_images(imgs, 
                   cols=6, 
                   title=f'Step {last_step} [{res}]',
                   unnormalize=True, 
                   figsize=(16,8))
#%% create_interpolation_animation
@torch.no_grad()
def create_interpolation_animation(imgs_tensor, filename='vis', interval=300, repeat=True, repeat_delay=1000):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    def animate(i):
        img_tensor = imgs_tensor[i].squeeze(0)
        frame = utils.make_grid(img_tensor, normalize=True).cpu().detach().numpy().transpose(1,2,0)
        ax.clear()
        ax.imshow(frame)

    anim = animation.FuncAnimation(fig, 
                                   animate,
                                   frames=imgs_tensor.size(0),
                                   interval=interval, 
                                   repeat=repeat, 
                                   repeat_delay=repeat_delay)
    # save the git using pillow
    anim.save(f'{filename}.gif', writer="pillow")
    plt.show()

# unlike our previous architectures, we use w to control the image generation, instead of z
# so our experiments will be targeting w space instead. but since our generator's forward
# pass only accepts z, we cant use that forward pass, we need to separate that logic. 
# instead of going back and change the generator class and separate them there, we can 
# easily do that here. becasue we dont do any fusion at inference, we don need progressive
# growing, or style mixing, all of that can be removed. this makes the forward pass 
# very simple and straight forward, just get the get the w, run through the layers 
# and get the final image! 
# (todo: definitely add this to the class when I refactor it later for easier handeling!)
#
# sidenote, to make this more seemless, we can monkey patch our class, so
# lets write this asif its a method of ourGeneratorStyleGan1 class!

#update from future:
# lets calculate average w post training and see if its the same
# as ema_w or close to it and use it in truncation trick
@torch.no_grad()
def calculate_w_avg(self, batch_size, sample_size=500_000, random_gen=None):
    
    assert batch_size<=sample_size,f'{batch_size=} can not be greater than {sample_size=}!'
    num_batches = sample_size//batch_size
        
    device = next(self.parameters()).device
    w_avg_sum = torch.zeros(size=(1,self.w_size), device=device)
    
    if sample_size<=500_000:
        zs = torch.randn((sample_size, self.z_size), device=device, generator=random_gen)
        return self.mapping_network(zs).mean(dim=0, keepdim=True)
    
    for _ in tqdm(range(num_batches)):
        zs = torch.randn((batch_size, self.z_size),device=device, generator=random_gen)
        w_avg_sum += self.mapping_network(zs).mean(dim=0,keepdim=True)

    return w_avg_sum/num_batches

# now lets implement the main part which is forward with w:
def forward_from_w(self, w, step, psi=None, w_avg=None, constant_noise=False):
    num_styles = 2*(step+1)
    if not self.training and psi is not None:
        if w_avg is None:
            ema_w_batch = self.ema_w.repeat(w.size(0),1)
        else:
            ema_w_batch = w_avg.repeat(w.size(0),1)
        w = ema_w_batch + psi * (w - ema_w_batch)
        
    # we must broadcast it after truncation or
    # otherwise truncation wont work
    # if we broadcast before truncation
    # then every single style layer would recieve
    # a different latent code!
    # since w now has shape the (b,num_styles,512) while ema_w_batch
    # is (b,512) so it will be broadcast to to (b, num_styles, 512)
    # as well which then when we do w-ema_w_batch and then multiply by psi
    # it gets applied independently to every copy of the repeated w!
    w = w.unsqueeze(1).repeat(1,num_styles,1)
    
    # set the batchsize for const_input/canvas
    x = self.const_input.repeat(w.size(0), 1,1,1)

    # disable noise injection by setting noise weights to all zero
    # so we remove all aspect of variations so we can properly test
    # truncation impact!
    # sidenote:
    # initially every time I tried different psi ratios, nothing 
    # would change it was as if psi=None! my manual w_avg didnt
    # do anything! I tried everything until I found out there's 
    # another source of randomness in our model and thats noise_injection!
    # I though that cant be, cuz its supposed to only affect fine details
    # but after I set it to all zeros, effectively disabling it
    # now different psi rates show their effect. now we can see 
    # with psi=0 e.g. we get single image replicated for all images
    # as its using the averge w! now w shows its impact!
    noise = None
    if constant_noise:
        noise = torch.zeros((x.size(0),1,x.size(2),x.size(3)),device=x.device)    
    
    for i in range(2*step+2):
        x = self.blocks[i](x, w[:,i,:], noise)
    
    return self.toImgs[step](x)

# since we already have our loaded instance we simply assign it
# if we hadnt loaded, we could do 
# GeneratorStyleGAN1.custom_forward = custom_forward
# and then instantaite and laod the weights
# but now we simply add this as a new method
generator_style1.forward_from_w = types.MethodType(forward_from_w, generator_style1)
generator_style1.calculate_w_avg = types.MethodType(calculate_w_avg, generator_style1)

# now lets write the actual experiments on w
# lets do a latent space exploration like before 
@torch.no_grad()
def interpolate_w(generator:GeneratorStyleGAN1, z1, z2, step, 
                  psi=None, w_avg=None, constant_noise=False,
                  alphas=None, interp_steps=60, device='cuda'):
    generator = generator.to(device)
    generator.eval()
        
    z1,z2 = tuple(z.to(device) for z in (z1,z2))
    w1 = generator.mapping_network(z1)
    w2 = generator.mapping_network(z2)
    
    # interpolate
    if alphas is None:
        alphas = torch.linspace(0, 1, interp_steps).to(device)
    else:
        alphas = alphas.to(device)
        
    imgs = []
    for a in alphas:
        # interpol = (1-a)*w1 + a*w2 #i.e. w1+a(w2-w1)
        # we can also use lerp which simply does the same thing
        # i.e. start + weight*end-start
        interpol = torch.lerp(w1,w2,a)
        img = generator.forward_from_w(interpol,step,psi,w_avg,constant_noise).cpu()
        imgs.append(img)
    output_imgs = torch.cat(imgs, dim=0)
    return output_imgs

# for w_avg, too few samples, and we fail
# to get a close estimate but with enough
# samples we get good results
def get_w_avg(sample_size=100_000, random_gen=None):
    if sample_size>100_000:
        batch_size = int(0.1*sample_size)
    else:
        batch_size = math.ceil(0.5*sample_size)
    return generator_style1.calculate_w_avg(batch_size, sample_size, random_gen)

sample_size = 100_000
w_avg = get_w_avg(sample_size=sample_size)

#to see how close our w_avg is to ema_w
# print(f'ema_w norm: {generator_style1.ema_w.norm()}')
# print(f'manual w_avg norm: {w_avg.norm()}')

# now to get a better overall view of noise_injection
# lets keep everything the same except constant_noise(disable noise injection)
def run_interpolation_test(step, w_avg_sample_size, constant_noise, psi_rates=None,
                           alphas=None, num_samples=36, make_gifs=False,
                           gif_dir='./results/gan/stylegan1/gifs',interval=100,random_gen=None):
    
    if psi_rates is None:
        psi_rates = [0, 0.3, 0.7, 1]

    # since when we do constant_noise=True, 
    # we physically set all the weights to 0
    # in order to be able to work on the same
    # instantce of our model we make a copy
    # here so the original weights are not 
    # altered and we can run other tests!
    # update: no more needed! 
    # generator_copy = copy.deepcopy(generator_style1)

    w_avg = get_w_avg(sample_size=w_avg_sample_size,random_gen=random_gen)
    
    # show the norm difference to see how close they are
    print(f'self.ema_w norm: {generator_style1.ema_w.norm().item()}')
    print(f'manual w_avg norm: {w_avg.norm().item()}')
    
    for rate in psi_rates:
        for weight in [None, w_avg]:
            interpolated_images = interpolate_w(generator_style1,
                                                z1, 
                                                z2,
                                                step,
                                                psi=rate,
                                                w_avg=weight,
                                                constant_noise=constant_noise,
                                                alphas=alphas,
                                                interp_steps=num_samples)
            
            if make_gifs:
                msg = "ema_w" if weight is None else "w-avg"
                fname =f"interpolation_psi_{rate}_{msg}_const_noise_{constant_noise}"
                # make sure directory exists, if not create it
                os.makedirs(gif_dir, exist_ok=True)
                fpath = os.path.join(gif_dir, fname)
                create_interpolation_animation(interpolated_images, fpath, interval=interval)
            else:
                #display as a grid
                msg = "ema_w" if weight is None else f"w-avg(sz={w_avg_sample_size:,})"
                display_images(interpolated_images, title=f'psi={rate} + {msg} | constant_noise:{constant_noise}', cols=cols, unnormalize=True)

# now lets try 
num_samples = 36
cols = int(num_samples**0.5)
seed=66
# seed global pytorch rng
torch.manual_seed(seed)
# seed cuda rng on all gpus
torch.cuda.manual_seed_all(seed)
# and finally get a fixed random generator.
fixed_randg = torch.Generator(device=device).manual_seed(seed)
# sidenote:
# previously I would use the generator returned from manual_seed*()
# but I noticed it doesnt give me determinstic outputs properly!
# turns out the manual_seed/manual_seed_all()
# return the default_generator which is used
# whenever a random generator is not passed to randn e.g.
# the issue is that, if we rely on this, and 
# somewhere in our code or a library we use,
# calls torch.rand*()(i.e. randn,randint,etc)
# without an explicit random generator, it 
# will use and thus change the default_generator internal state
# and therefore change our randomness! breaking our determinstic output
# extra-note:
# the Random Number Generator pytorch uses is
# known as a stateful pseduo-random number generator (PRNG for short)
# note the stateful part. it works just like
# a tape with an infinit sequence of random numbers
# written on it. it has a read head that points to
# the next number that we havent used yet and 
# each time we call a rand function (e.g. randn, randint,etc)
# it advances the read head on that sequence/stream.
# effectively updating the internal state and calculating
# the new random number from the new sequence of numbers.
# when we use a separate generator, we get a private stream
# so when we call rand* function, it gives us determinstic
# outputs. because its not shared, so each number in 
# the sequence of numbers gets used by the same order
# we call rand* functions in our cdoe and we always get
# the same random numbers in succession.
# but when we use the default/global random generator
# each time a rand function is called, regardless from where
# our codebase, or a thirdparty library code, the stream is
# advanced and the internal state changes, this is because
# it is shared by everyone, so any function/code in 
# our codebase or libraries that use the default generator
# causes the head to move forward, so our rand* functions
# that get called each time, they get a different randomnumber
# hence why we get different results sometimes!
# (note things like other layers like dropout, dataloaders,
# cuda kernels, they are all sources of changing the default
# generator state because they all use randomness. even the 
# order of operations can affect this so to get 100% determinstic
# output we must use a dedicated random generator and not use
# the global/default generator)
# to make this click! imagine this to be our tape
# tape an infinitly large list [0,1,2,3,4,...]
# the read head points to the next unused number as random value
# each call to rand/randn/etc uses a number and thus 
# moves the head forward! when we use a private generator
# we have a private list of numbers, we first call rand1
# it uses the first number and moves to the next in sequence
# when we use rand function 2 e.g., it uses the new (second) number
# when we use rand function 3 e.g. it uses the next one. 
# if we repeat this, it starts all over again. each rand uses
# a new/different random number, but our results will be the same
# because if the first got 0, second got 1, third got2, they 
# will always be that as long as the order is the same. 
# (i.e. rand1 is called first, rand2 called second and rand3 called last!)
# todo: simplify and make it shorter its too long

z1 = torch.randn(size=(1, generator_style1.z_size), device=device, generator=fixed_randg)
z2 = torch.randn(size=(1, generator_style1.z_size), device=device, generator=fixed_randg)

run_interpolation_test(last_step, 
                       sample_size,
                       #psi_rates=[0,0.5,1.2,3,5],
                       constant_noise=True,
                       num_samples=num_samples,
                       random_gen=fixed_randg)
# with constant noise, we can truly see the average 
# and images close to average are exceptionally
# clear and well developed!
# with psi rate we can see how better results though less diverse depending
# on which value we pick
#%% run_interpolation_test
# now lets do this with noise injection i.e. constant_noise=False
run_interpolation_test(last_step, 
                       sample_size,
                       #psi_rates=[0,0.5,1.2,3,5],
                       constant_noise=False,
                       num_samples=num_samples)
#%% making some gifs
# now lets also make some gifs as well!
alphas = torch.linspace(0,1,25)
for noise_status in [True]:
    run_interpolation_test(last_step, sample_size,
                           psi_rates=[0.7],
                           constant_noise=noise_status,
                           alphas=alphas,
                           num_samples=36,
                           make_gifs=True,
                           gif_dir='./results/gan/stylegan1/gifs',
                           interval=100,
                           random_gen=fixed_randg)
#%% truncation and stylemixing tests
# now lets do some style mixing, use one z for the coarse features
# and another one for style change. 

def apply_truncation(self, w, psi=None):
    if not self.training and psi is not None:
        if w_avg is None:
            ema_w_batch = self.ema_w.repeat(w.size(0),1)
        else:
            ema_w_batch = w_avg.repeat(w.size(0),1)
        w = ema_w_batch + psi * (w - ema_w_batch)
    return w

def forward_from_w_simple(self, w, step, constant_noise=False):
    # set the batchsize for const_input/canvas
    x = self.const_input.repeat(w.size(0), 1,1,1)
    
    noise = None
    if constant_noise:
        noise = torch.zeros((x.size(0),1,x.size(2),x.size(3)),device=x.device) 
    
    for i in range(2*step+2):
        x = self.blocks[i](x, w[:,i,:], noise)
    
    return self.toImgs[step](x)

# update:
# for latent related experiments we need to use constant_noise
# to properly see the impact of other factors like stylemixing at
# different levels. truncation also helps a lot
@torch.no_grad()
def style_mix(self, z_source, z_style, layer_indx_for_crossover,
              step, psi_src=None, psi_sty=None, constant_noise=True):
    
    num_layers = 2*self.max_steps
    assert 0<layer_indx_for_crossover<num_layers, f'layer index{layer_indx_for_crossover} must be < {num_layers}'
    
    device = next(self.parameters()).device
    
    z_source = z_source.to(device)
    z_style = z_style.to(device)   
    
    w_source = self.mapping_network(z_source)
    w_style = self.mapping_network(z_style)
    
    # apply truncation
    w_source = self.apply_truncation(w_source, psi_src)
    w_style = self.apply_truncation(w_style, psi_sty)
    
    # expand to match shape
    w_source = w_source.unsqueeze(1).repeat(1,num_layers,1)
    w_style = w_style.unsqueeze(1).repeat(1,num_layers,1)
    
    # we are going to need both source and style ws for later
    # reference so lets use it to create our result w_mixed
    w_mixed = w_source.clone()
    w_mixed[:,layer_indx_for_crossover:,:] = w_style[:,layer_indx_for_crossover:,:]
    
    img_source = self.forward_from_w_simple(w_source,step,constant_noise).cpu()
    img_style = self.forward_from_w_simple(w_style,step,constant_noise).cpu()
    img_mixed = self.forward_from_w_simple(w_mixed,step,constant_noise).cpu()
    
    return img_source, img_style, img_mixed

# lets add them to our instance 
generator_style1.apply_truncation =  types.MethodType(apply_truncation, generator_style1)
generator_style1.forward_from_w_simple = types.MethodType(forward_from_w_simple, generator_style1)
generator_style1.style_mix = types.MethodType(style_mix, generator_style1)

z_source = torch.randn(1, generator_style1.z_size, device=device, generator=fixed_randg)
z_style = torch.randn(1, generator_style1.z_size, device=device, generator=fixed_randg)

# different layers affect different details experiment
# with all and see the result. e.g. starting from 4 
# we can see more drastic style transfers. if we go
# lower, like 1, 2 , we are basically seeing second z!
# the majority of values belong to second z, but starting
# from 3,4, we can see the first image is structually there
# and styles start to transfer.(skin tone, colors, are obvious) 
layer_crossover = 5
img_src, img_style, img_mix = generator_style1.style_mix(z_source, 
                                                         z_style, 
                                                         layer_crossover, 
                                                         step=last_step,
                                                         psi_src=0.7,
                                                         psi_sty=0.8,
                                                         constant_noise=True)
imgs = torch.cat([img_src,img_style,img_mix])
display_images(imgs, title='images source|style|mix', unnormalize=True,figsize=(8,6))
# as we can see, the structure of the source image stays the same, but the 
# texture/style of the style image is transfered. lets see how each layer affects
# the result 
#%% change styles
def change_styles(z_source, z_style, step, psi_src=0.8, psi_sty=0.8):
    num_layers = 2*generator_style1.max_steps-1
    imgs_all = []
    for layer in range(1,num_layers):
        img_src, img_style, img_mix = generator_style1.style_mix(z_source, 
                                                                 z_style, 
                                                                 layer, 
                                                                 step=step,
                                                                 psi_src=psi_src,
                                                                 psi_sty=psi_sty,
                                                                 constant_noise=True)
        
        imgs = torch.cat([img_src,img_style,img_mix],dim=0)
        # print(f'{imgs.shape=}')
        
        row = utils.make_grid(imgs, nrow=3, normalize=True)
        display_images(row, title=f'{layer} source|style|mix',
                       unnormalize=False,  figsize=(8,6))
        # print(f'{row.shape=}')
        
        imgs_all.append(row.unsqueeze(0))
    
    ims = torch.cat(imgs_all, dim=0)
    # print(f'{ims.shape=}')
    display_images(ims,
                   title='Source | Style | Mix',
                   unnormalize=False,
                   cols=1,
                   figsize=(16,16))

change_styles(z_source,z_style,last_step,psi_src=0.7, psi_sty=0.7)
#%%
# now lets see the default method with constant noise
# we should get much better results this  time 

torch.no_grad()
def run_simple_gen_with_const_noise(constant_noise=True, num_samples=36):
    
    z = torch.randn((num_samples, generator_style1.z_size), device=device)
    noise = torch.zeros((num_samples,1,dim,dim),device=device) if constant_noise else None
    # None is basically psi=1, but im including it for the sake of including it!
    for i in [None, 0.4,0.7,1]:
        imgs = generator_style1(z, alpha=1, step=last_step, psi=i, noise=noise)
        
        display_images(imgs, 
                        cols=6, 
                        title=f'Step {last_step} [{res}] - [PSI: {i}] - [Inject Noise: {not constant_noise}]',
                        unnormalize=True, 
                        figsize=(16,8))
        
run_simple_gen_with_const_noise()
# as we can see we get much better results, but less diverse as well when we use constant_noise
# note that without constant_noise, we cant properly see the impact of psi/stylemix as we
# mentioned earlier. also note we havent trained our model very well! just enough to get
# decent looking images. it would take a lot of time (I wasted two weeks on experimenting
# with my buggy implementation unitl I fixed them! so I didnt let it train for too much when
# things got working!) so by trainig more our default mode should start looking great as well
# but can always use truncation trick with constant noise to get much better results if we need to!
# also the most important part, we can clearly see the slow and smooth trnasition when interpolating
# which shows the features are disentangled much better than progan version. (add more tests/
# make it more obvious (train with celeba, use celeba classifier so we can do more experiments))
#%% stylegan1 debug logs:
# debug logs:
# note I want to get this to work organically, 
# that is just like the original paper. for that
# reason we should be able to get a working example
# with the same hyperparameters and only what is
# said in the paper, so we are not going to do
# anything extra, like noise addition that we did
# in earlier gan experiments. for that reason I 
# remove those options so we can only focus on 
# getting to the root of the issues and hopefully
# implement everything accuractely.
# 
# starting the training with fp32 we had a bug
# which made our mapping network to have lr=1.5e-7
# this lead to mode collapse in 64x64 as the generator
# was far behind(slow compared to) dicriminator.
# switched to fp16 so we can use larger batches.
# using fp16 and lr=1.5e-5 lead to nans early on
# to solve the issue we used larger eps for adams
# it worked very well and we no more got any nans
# the 16x16 res looked specially good. both of the
# discriminator and generator losses were balanced
# both were around 1. but switching to 32x32, the
# discriminator got the lead (down to .7 vs 1.8 gen)
# the 32x32 didnt become as good as I expected, it 
# was blury I didnt like it. when it got to 64x64
# the discriminator loss became 0.4 and generator
# became 3. the images had clear artifacts and malformed
# so next we are going to apply gradient clipping
# before that though I'd like to train in fp32 to
# see much of this comes from fp16 and how much from
# the actual traiing!
# 
# update: faced mode collapse 0.4 vs 3.26! discriminaror
# massivly does better than generator
# 
# update:
# usef fp32 this time, and the loss didnt change, upto16²
# everything is very good, but starting with 32² it becomes
# bad and in 64² it becomes worse, the loss is like fp16
# so the fp16 wasnt the issue.
# 
# update(stylegan1_celeba_20251030125518): 
# reverted back the adam eps=1e-8 and instead used gradient clipping
# as expected we no more get nans in 8x8 res like before, however
# I noticed the loss for both discriminator and generator
# is much larger (5.88 va 10.10 vs 0.7 vs 1.2 in 32²) the quality
# of generation is also much worse! we also faced partial model collapse
# as early as e16@32²!
# 
# update:(20251030161059)
# trying with fp32 and mn_lr=1e-4:‌ up to 16x16 it went great
# d_loss and g_loss both around 1 and overall images look good
# however starting 32x32, d_loss=0.7 but g_loss=1.6, as expected
# this didnt turn out any better either!
# 
# update(20251030201508):
# trying with fp32 and mn_lr=1e-3: absolutely no difference! 32x32 disc
# gets lower loss(0.76) and gloss goes 1.76! at e3 of 64x64 we faced
# abnormally large gradients in mapping network (started as 114,125~164
# in the next few epochs went up in thousands, by epoch12 it was in hunderds
# of thousands (406k!) and by e16 it was in in millions(7m!) it became so bad
# in epoch31 the gradient norm was 8.8239e+12! ) so the mn_lr=1e-3 is just 
# too much.we need to dial it back to 1e-5 and see whats giving generator a
# hard time here!
# 
# update(20251031065944):
# I reread the paper and had another look at the official tf impl
# there was no sign of scaling the r1_penalty, infact there was
# not any intermittent appliance of r1_penalty! they applied it
# normally! to test the effect however, I disabled the scaling by
# iteration count, to see the effect on the training. 
# in 8x8 the discriminator loss that used to be larger than generator
# is now lower! in 16x16 epoch 0, many white dots artifact are vsibile
# now that dont happen when we scale! also the d_loss is also much lower
# than the generator, and g_loss went up(2.5)!(it used to be the same for both around 1.1!)
# and obviously the image quality goes down hill!
# so not applying the r1_penalty makes discriminator do better! 
# and make gens work harder and hence do worse!
# in other words, applying it less frequently even with scaling
# can only keep up to 16x16, after than the discriminator overpwoers
# the generator. so next we will apply r1_penalty all the time!
# 
# update(20251031082850):
# did nothing! my understanding seems wrong! need to read the whole official imp
# and see what im doing wrong!
# 
# update:
# im trying different parts now. siwtched to cifar10 for quicker experiments
# 
# (20251031164619):
# I swapped the EqualizedConv2d with the old one we had for progan, and trained
# the results were very bad, many grayish images, especially at 32², the losses
# however were pretty close , like 0.89/1.4 but the outcome was very noisy.
# 
# (20251031185610):
# swapped back to our stylegan specific version and so far its much more colorful
# clearer (even at 16²) the losses are close (1.15 vs 0.989 @16²)
# but when it comes to 32² it becomes blurry and discriminator gets the lower loss(0.8vs1.59)
# 
# (20251031220917, 20251101072321):
# use larger gamma to make disc struggle abit more so generator can breathe abit!
# at larger res like 32x32: ok this didnt help at all we still face the same issue
# this might be due to model capacity itself. that is we may have used too few channels
# for later resolutions that requrie more processings! 
# 
# update(20251101094159):
# so lets increase channel counts like the original paper: the problem still exists
# I trained with more channels, with less channels and the problem still exists, starting
# with 32x32 the disc just overwhelms the generator, its loss quickly (i.e. epoch0!) gets
# down(0.7 vs 1.2). trained with cifar10, celeba didnt make any difference, they show the
# same exact symtopm! 
# 
# update(stylegan1_celeba_20251102091252):
# I noticed some implementations include a blur module (anti-aliasing) while the official imp
# has it implemented but didnt use it if I recall correctly. this was meant to fix the checkerboard
# issues that happens when we go into higher res where its prominent and disc can easily find it
# and flag the images as fake! I used that but the still didnt change anything!
# 
# update(stylegan1_celeba_20251102123140):
# I went back to the begining and changed the order of adaIn and lrelu in StyleConvBlock.
# I swapped the ordr and im currently training again hope this fixes the issue!(blur is also active in this experiement)
# 
# update: it actually did work! now in 32x32, the d_loss is 1.35 vs g_loss=0.728!
# so it was the damn order all this time!! the convergence is much much faster now! and
# we achieve very low FID as well (29.54 in epoch7@32²!)given this I guess we can only
# use way fewer epochs! starting with epoch 4 @64x64 we got warnings for grad_norms exceedingly
# getting larger for mapping network! the loss was still not affected, but I ended the 
# training at epoch 6 so I can address it properly. I previously had gradient_clipping
# for fp16 trainig but commented it out because I used larger eps for adams to make it
# 
# stable. now I will uncommented it and enable it as default as fp32 also seems to require it
# update:(cifar10_20251103154849):
# training cfar10 with small number of epochs and channels. the model fails to properly
# form images at higher resolutions.insufficient training at lower res(only 10 epochs for
# early resolutions (4²,8²,16²) with insufficient channels made 32x32 completely fail although
# the loss seems okayish! but its completely random blobs! if you look closer we can see
# early res were developing good siluhetts, low res versions of images, but before they converge
# we started the next resolution/fade in process which ruined everything.
# 
# update:(cifar10_20251103185721)
# with more epochs and more channels for each res, we get a better result. early in 32x32 we
# can see the same issue as before, temporarily the images turn into bloby mess, but given more
# epochs, they get more formed into actual images. horses, cars, are more clearly formed.but
# overall the quality isnot good. using larger batchisze, more data helps. also the discriminator
# is struggling and this is why the quality is getting worse! the generator is overpowering
# the discriminator. need to fix that to get decent images. however im really tired! and it
# is taking too much time to train! 
#
# update:(stylegan1_ffhq_20251104101011)
# test last experiment with lrelu>adain (original order) with ffhq:didnt change, still at
# 32x32 we faced eventual mode collpase because discrimnators loss was lower than generators
# now at this point i guess its because of batchsize we are using, because the paper uses
# small batches per gpu (16 x8gpus) the gradients get accumulated as if its batchsize=128
# but layer wise its not the same. note we have minibatchstddev when we use smaller batchsize
# the stats will be much noisier than we use batchsize=128. I guess the paper authors relied
# on this and specifically chose smaller batchsizes for this exact reason.unlike previous gans
# progressive gans are relient on these kind of noise for stabliziation I guess. 
# lets check this again with smaller batch size(for now lets just use smaller batchsize
# we dont accumulate gradients to mimic the x8 for now!)(update:wrong! the batchsize was ok! the issue is sth else)
# update:(20251105054505)
# use smaller batchsizes, drastically longer trainig epochs per res:
# 
# update: 
# refactored code a bit trying a few more experiments before calling it a day for good!
# experiment 1(stylegan1_celeba_20251106112914): fp32 - large batches
# use original order + lower disc_channels + more gen_channels + more epochs per res
# interestingly I started getting warnings(mapping_network_grad_norm exceeding 100)
# at epoch 21 at 16x16 resolution. it got worse so much by epoch 26 we hit 3000+, i.e.
# gradient explosion. a bit of digging and it seems the problem is not in the code but
# the hyperparameters I used. this time around I made discrimnator much thinner than the
# generator (11m vs 23m). it seems the discriminator got good really well at that point
# and created a massive gradient feedback for the fake images, the gradients ended up at
# mapping_network which we happen to check for and we see the warning. also starting with
# this we see the d_loss is a bit lower than g_loss (1.02 vs 1.1945) and images are getting
# worse, weird artifacts are visible, so this might be the reason! I let it train until we 
# reach 32x32 to see how it ends up! it got worse at 32x32 from iteration 0 epoch 0! 
# its now in 20ks!(update: wrong see later updates what was the cause there was actually a bug 
# in code that manifested itself with the specific hyperparametrs like that)
# quicknote:
# we remember that previously we said when discriminator gets really good, the gradients vanish
# not explode, and thats why generator cant improve, cuz it doesnt get proper feedback. 
# but here we are saying the complete opposite. the reason we say this here is because 
# this behavior depends on the loss function we use. if we use staurating loss functions
# like bce, that use sigmoid, when the model gets good, (the real images e.g.) it gives
# values very close to 0, so the the gradients become extremely tiny and therefore the
# generator cant improve with those tiny gradients. on the otherhand, when we use other
# types of losses, like WGAN, we dont use any activations for the final layer of discirminator
# no sigmoid is used, so the network can learn to produce very large scores for the correct
# images(real ones) and very small ones (large negative numbers), so when we calculate the
# genertor loss by -d(fake_images).mean() and the disc produces large negative numbers then 
# the gradients will be massive as well!
# now the intersting part is, in stylegan1 we are using softplus+r1_penalty, which is a
# non-saturating loss, that looks like the WGAN-GP. softplus is a smoothed version of relu
# (i.e. log(1+exp(x))) and is not bounded, so we are basically getting raw outputs/logits 
# so the output can get as large as it wants! the sofplus like in wgangp is there so we 
# dont face vanishing gradients!(discriminator doesnt get too good too fast!)
# but not the exploding gradients! in fact its very susceptible to gradient explosion
# to fix that r_penalty is used. but the generator doesnt use that term so it faces explosion
# as we can see! (update the loss functions with new information, can remove this explanation)
# 
# 
# experiments results/weights are stored at https://mega.nz/folder/zRcUHSJR#kjV_qY5LuinhdwugJZYL0A
# quicklog 
# seems d is overpowering g
# lets remove less epoch so it doesnt overtrain
# see if this fixes the mn gradient explosion
# 
# stylegan1_celeba_20251106163711:
#  - cant resume properly from previous experiment that 
#    had autocast(scaler errors out on resume saying:
#    ...
#    --> 363 scaler_out_d = scaler.step(disc_optimizer)
#    ... 
#    AssertionError: No inf checks were recorded for this optimizer.)
#    despite fp_16=False!
# 
# stylegan1_celeba_20251106164141:
#  - so disabled the whole autocast machinery and resumed but the images
#    are extremely distorted I give up and start fresh!
# 
# stylegan1_celeba_20251106165912:
#  - start fresh with no autocast. 
#    ok 32x32 and still no gradient explosion
#    but we are seeing severely artifacted images
#    the dloss is down to 0.71 at epoch 1 while gloss is 1.98!
#    the d is clearly overpowering g! and we have partial
#    mode collapse let it train more and see if we get
#    any gradient explosion again - ended at epoch 6 
#    if not train with 30 epochs(kimages*3) to see if this was
#    really an overtraining issue or simple a pytorch bug because
#    of autocast because in previous exp, I disabled autocast completely!
#    not just disable it, I commented out all machinery (scaler.*)
# 
# stylegan1_celeba_20251106212921: 
#  - running the test with kimage*3 to see if we get gradient expo a 16x16
#    no gradient explosion happedn we are at epoch 20 at 32x32
# 
# stylegan1_celeba_20251107070546:
#  - now lets increase lr=0.002 to see if it fixes anything(doubt it but lets try)
#    so far no explosions, the loss is dloss=1.00 vs gloss=1.24 at 16x16 e28
#    at 32x32e0 we have 0.63 vs 2.14 which is very high im going to let it run more
#    and see how it turns out. ended it at epoch 2 as there are many artifacts poping up!
# 
# stylegan1_celeba_20251107134153: 
#  - now lets try the conv/noise/bias/lrelu/adain and see how it does! lrs=0.001
#    fp32, γ=10, kimages*2(i.e. 20 epochs per res), autocast still disabled
#    lets see how this combo does.the dloss=1.03 vs gloss=1.16 at 16x16 e19
#    we see the same issue we had starting with 32x32 the dloss=0.7 vs gloss=1.89
#    the changes I made doesnt seem to affect the outcome much really so my previous
#    implementation that was conv/noise/lrelu/adin was a good approximation
#
# stylegan1_celeba_20251107170458:   
#  - now resuming with increased gamma(20) to see if its salvagable!
#    it has a tiny impact. 0.85 vs 1.5 at epoch 1 (32x32). 
#
# stylegan1_ffhq_20251107210907:
# - now lets try one of the pytorch implementations configs and see
#    how that goes! : https://github.com/huangzh13/StyleGAN.pytorch/
#    the default configs is at https://github.com/huangzh13/StyleGAN.pytorch/blob/master/config.py
#    and specific settings for each dataset is https://github.com/huangzh13/StyleGAN.pytorch/tree/master/configs
#    notable changes are 1. mapping network is 4 layers here.
#    the lr is also set as 0.003! the depth is 6(up to 128x128) fo ffhq128
#    lets train with no data-augmentation the loss is the same as our previous
#    experiments, 0.8 vs 1.68 at e0 32x32. also 0.73 vs 1.74 at e25 32x32
#    the fid is terrible so I end the training
#
# stylegan1_ffhq_20251108072704:    
# - now apply augmentation and see how that changes anything if at all
#   we resume from 32x32 e0: epoch 10 0.75 vs 1.80 didnt imporve so I stopped 
#   the training!
#
# stylegan1_ffhq_20251108110259:
# - next use the same lr(0.003) for mapping_network as well! first resume
#   ok no apparent change so far, might be because in previous epoch mapping
#   network hasnt caught up yet due to slow update rate. lets start fresh
#
# stylegan1_ffhq_20251108124531:
# - start with lr=0.003 for the whole generator (including mapping_network)
#   the same issue occured again!e2 0.79 vs 1.69 in 32x32 and I see green/purple
#   blobs on the images, basically output is artifact riddent .the image quality
#   however is not abysmal! or I may be halucinating! but the overal image structure
#   looks ok, however there are discoloration as I said. lets let it train a bit
#   I also found a tiny bug in EqualizedLinear thats used in mapping network aswell,
#   I think the discoloration we see might be attributed to number of mapping layers,
#   we used 4 instead of 8! so the network might not have the necessary capacity to
#   comeup with proper styles by the current res! 
#   ok, but the green blob is gone in epoch 3! so I think my reasoning might have 
#   been correct!
#   ok! the training is going on, but I digged deeper and the bug I found is not that tiny
#   and insignificant as I thought, infact its huge!! and looking at it I guess this 
#   is why I couldnt train properly!
#   in EqualizedLinear, I hadnt initialized the weights properly (I should have done 
#   normal(0,1/lrmult) but instead had done(N(0,1) and didnt think of it as being problematic at all!
#   oh I was so wrong! so wrong! looking at the formula, (I explained in detail in EqualizedLinear)
#   this seemingly innocent mistake! caused the output variance to shrink extremely bad (10000x!) 
#   when we applied the 0.01 lrmult, so its why we kept seeing network struggle post
#   32x32! and ultimately fail!(I looked at official impl and noticed they did that
#   but ignored it thinking it wasnt needed as the paper didnt say anything about it,
#   I thought it was one of those simplification/or over enginnering etc, and the 
#   dynamically scaling we have should work! because the equalizedconv2d was similar
#   and I had not had any issues with it, only after all of these issues when I 
#   revisted it again, I found how wrong I initially was!)
#   now imagine we have been applying 0.01 to mapping network on top of that! 
#   making the variance 1000,000x times smaller!!! no wonder nothing worked! damn!!)
#   initially I checked mapping network to see why we are not training differenly 
#   than before even though I'm using much larger lr now(i.e. 0.003), then I noticed
#   EqualziedLinear and went to check it again! and the rest is history!
#   fast forwrad and after finding the bug! it completely makes sense that that 
#   EqualizedLinear bug caused extrmeley slow convergence.
#   even without the fix, with the increased lr (removing the excessive 0.01 from mapping_network
#   and using 0.003) we can see improvements! in epoch 18/19 we can cearly see images have
#   improved a lot since alpha=1, but we can still see different artifacts, especially 
#   discolarations that can be attributed to extremely slow convergence. 
#   it can also be attributed to smaller number of mapping layers i said earlier but I
#   beleive at this point its a cause of EquzliedLinear bug which is now fixed, but
#   since this experiment is ongoing I'll let this round go and see the final 
#   results.(update I ended this at the end of 32x32 and started a new experiment with the
#   fix see below)
#
# stylegan1_ffhq_20251109035618:
# - test with new EqualizedLinear fix! increase the mapping network layers to 8 again:
#   massively improved at 16x16 e15 1.37 vs 0.71 and images are miles better! way better!
#   they are very detailed for 16x16! see the results and compare it to previous examples
#   to understand what I mean also in 32x32 although we have 1.37 vs 0.74 and image quality
#   are way way better!
#   yup that was the issue after all! the thing is EqualizedLinear was the last thing
#   that I could ever think had issues! I was 100% Igot it right! and here we are 3 weeks
#   in to only find the culprit was the EqualizedLinear all this time!(of course my gaffe
#   for appling 0.01 on mapping network in optimizer made this evern worse! but we learned
#   alot and found a whole new way to get more stable training as well! so not bad I guess!)
#   the images at epoch 27 are gorgeous! detailed well formed most of the time. its working
#   prefectly! note during training images may get deformed or discolored early on but as training goes
#   and alpha goes to 1 it gets better and better.
#   the training in 64x64 is extremely slow, each epoch now takes nearly two hours!! to complete!!
#   two hours! (i.e.1:40') and I am using an rtx3080! with 12700k! this is insane!
#   9677MB vram is being used but the image quality is gorgeous! unlike the previous res
#   like 32x32 that the image gets distorted and then gradually as it reaches alpha=1 it 
#   becomes high quality, at 64x64, epoch 0, images are really sharp, but also with artifacts
#   the artifacts are noticeable when you view the image up close. by epoch 2 these are 
#   nearly gone, but there are still some weirdness, like eyes are all black suddenly!
#   like demonic black!I'll end the training at epoch 10 so I can start other experiments!
#   the results are improving epoch after epoch but it takes a lot of time and I cant wait
#   that long (around 2 days worth of training remains till we finish the 64 epochs of 64x64!
#   and we are at epoch 9 since yesterday's afternoon (16:15)!)
#
# stylegan1_ffhq_20251110095733:
# - since it takes ages to train especially in 64x64+ lets use a small number of channels
#   for each resolution so we get a good idea how this impact the results and to what extend.
#   use [512,256,128,64,32,16,8] for both channels now. the convergence rate is lower but
#   image quality is still very good. however, the previous larger config gave us much better
#   looking iamges much earlier, for example, images very nearly structurally prefect in 
#   epoch3@64x64 though there were some noticeable artifacts, but the images from afar looked
#   completely fine(at epoch 10 they looked way better, afew seemed prefect!)! but here, 
#   even in epoch 15@64x64, you can still see considerable deformity in a lot of the
#   images from afar(by that I mean in vscode interactive pane where the sample generation is
#   scaled down to fit properly, and by up close I mean when I click on Expand image to 
#   view them in full resolution). but since it runs several times faster, we can wait and 
#   see ultimately how well the images get at what resolution adn whethere its worth it.
#   compared to our previous 23m/25m models, this trains much much faster as I just mentioned
#   for example it takes us only 5 mins to do an epoch at 32x32 res, and only 15mins to do
#   an epoch at 64x64! whereas before, it took 17mins for32x32 and 1 hour and 45 mins! for
#   a single epoch in 64x64! at epoch 17, it starts to look decent (like epoch 6 in larger model!)
#   but at epoch 29(when α is nearly 1) we are getting much better images! at e61 finally 
#   most of the images are artifact free and look decent. so using smaller models,with
#   more epochs we can get good results. however I guess going larger with less epoch should
#   take us there faster. when I look at it, the previous model used many epochs, we could
#   have decreased the epochs so fadin could start faster and get better results quicker!
#   also at 128x128, it takes 35mins to train a single epoch! still way faster than the
#   previous large config. I end this experiment at e2@128x128 so I can continue the rest 
#   of the experiments.(note: the generator used 0.001 instead of 0.003! but despite that
#   we got great results!this was a mistake I found out after 4 experiments!)
# 
# stylegan1_ffhq_20251111075258:
# - previous experiment now with smaller number of epochs: try with [4,8,8,8,16,16]:
#   the epochs are not enough, I guess at least they need to be doubled! this might
#   work for the large config, but definitely not the 11m config we used in the previous
#   experiment.(note: the generator used 0.001 instead of 0.003!))
#
#  stylegan1_ffhq_20251111130402:
# - revert back the last changes in stylegan (separate bias, etc) and see with the
#   newly fixed EqualizedLinear, how our previous implementation works
#   disabled bias in styleconv2d, enabled bias in styleconv2d(equalizedconv2d)
#   basically the old order but now with this epochs : [8,16,16,16,32,32]
#   as expected my implementation works just fine. so the separate bias/ plain conv for
#   fist layer of 4x4 res, specific order conv/noise/bias, none of that matters really!
#   im going to rever the changes anyway, since our previous experiments were done
#   using that configuration so we can easily load the checkpoints. 
#   the higher number of epochs made the images look much better than previous experiment
#   that used fewer channels, however this is not obviously enough, 32x32 needs more
#   to look decent like the other previous experiments that used [8,16,32,32,64,64]
#   I guess that might be the sweet spot for the 11m model.anywa we train the 64x64
#   as well and call it a day and go for the next experiment which is adaIn-lrelu 
#   order swapping! stopped at epoch 5@64x64. the results were what we expected#
#   now lets continue with the next experiment!(note: the generator used 0.001 instead of 0.003!))
#
# stylegan1_ffhq_20251111182532:
# - train with swapped_adaIN_order aswell see if it indeed is better choice than original:
#   in order to get an accurate comparison, we use the stylegan1_ffhq_20251110095733 settings
#   and only swap_adaIN_order: the outputs are roughly the same, after I fixed the main bug
#  it seems the order swap really doesnt do much. before the fix however, it was way different
#  we could actually train after the order swap! but now they seem the same really! ok
#  I made a mistake here, the swap_adaIN_order is not applied! we wasted our time! 
#  damn it!(not only that the generator used 0.001 instead of 0.003!)
#
# stylegan1_ffhq_20251112142815:
# - I just noticed in the generator I forgot to send the swap_adaIn_order argument to
#   each styleconvblock! basically our previous experiment used the original order!
#   this is the actual experiment with the adain_order actually swapped!: the quality
#   seems to be better than the original order. the loss is also slightly lower, the 
#   FID is also slighyly better than! in higher res, the loss is much lower. ok I just
#   noticed I had another mistake! when I was refactoring generator optimizer, I forgot
#   to add lr_g back! thus the generator was using 0.001 instead of 0.003!(disc was using0.003)
#   even with generator having much slower update, we got great results!
# 
# stylegan1_ffhq_20251113102258
#  - train with proper lr for generator to see how it works but this time lets do it 
#    up to 32x32 so we dont waste too much time: the image quality is much better much
#    quicker now. at epoch 14@64x64, they look way better. by e19 images are gorgeous!
#    the losses are dloss=1.370 vs gloss=0.720. at 63@64x64 1.36 vs 0.722 and image quality
#    is the best among all previous experiments. more clear, vibrant and look high res!
#
# stylegan1_cifar10_20251114082051
# -  train cifar10 
#    we use larger epochs for early layers because we are going to only train up to 32x32
#    epochs=[16, 24, 48, 64, 64, 64] with swapped_adain_order. as we saw in our earlier 
#    experiments, some classes are well formed better than others, some look prefect while
#    others are very malformed, not artifacty, but malformed, either missing parts, or
#    warped in unnatural way. im not sure yet if training for more would fix that or we just
#    need more data per class.(looking at earlier epochs that these malformed samples were
#    more abundant, and as we trained, especially towards the end, they got better
#    I guess training for longer might give us better results but we need to test 
#   to be sure)
#
# stylegan1_cifar10_20251114140050:
# - train cifar10-fp16-small batches-eps=1e-5:
#   to see if the autocast implementation is ok and the issues of exploding gradients
#   were truly due to our EqualizedLinear bug, we train with fp16 and batches=16
#   I also use the original order. everything else stays the same as our previous cifar10
#   experiment: faced the same issue, nans during training this time early on in 4x4 res
# 
# stylegan1_cifar10_20251114140050/
# stylegan1_cifar10_20251114142351/
# stylegan1_cifar10_20251114144112/
# stylegan1_cifar10_20251114145725/
# stylegan1_cifar10_20251114150815
# -train cifar10-fp32 small batches eps=1e-8:
#  disabled the fp16, everything else intact from previous experiment. faced gradient
#  explosion this time.test with smaller lr down to 0.0001 didnt do anything at fp16 
#  we get nans, if we set use_fp16=False, we get gadient explosion!
# 
# stylegan1_cifar10_20251114151527:
# - commented out all autocast related code, withfp32, now with small bs=16, all is fine!
#   the fp16/autocast code has some issues! found the issue with autocast.scaler
#   was always true even with use_fp16=False, that would keep scaling the gradients
#   and lead to greadient explosion!
#
# stylegan1_cifar10_20251114164657:
# - fix autocast or remove it completely: fixed scaler being True despite use_fp16=False
#   now we dont get gradient explosion. I cant believe I missed that part!
#
# stylegan1_cifar10_20251114183323:
# - test fp16 now with small batches so we have more unstable gradients to exacerbates
#   the issue further so we can better identify and fix it. right off the bat grad_norm
#   shot up to 119! we didnt touch anything in the code, we are just recording what
#   happens and then go for the fix.ok faced nans! but in FID_score section!
#   cov_prod_sqrt = linalg.sqrtm(real_cov.cpu().numpy() @ fake_cov.cpu().numpy())
#   baghie farda enshalaah
# 
# stylegan1_cifar10_20251115083009:
# - removed fid_score calculation to reveal the underlying issue easily. 
#   the configs are intact, bs=16, lr=0.003 and fp16=True, ok in epoch 9
#   we get nans for loss. more specifically we can D_fake_avg is nan! (D_real_avg
#   is very small, but for this res and cifar10, its normal). 
#   
# stylegan1_cifar10_20251115100058:
# - set gradient_clipping to prevent gradient explosion with norm=10: at epoch4 the
#   nans happened again.
#
# stylegan1_cifar10_0251115102555  
# - lowered the lr = 0.001 and it seems to have fixed it!
#      
# stylegan1_cifar10_20251115124520:
# stylegan1_cifar10_20251115131747:
# - with lower lr=0.001, disabled gradient clipping to see how stable the trainig is
#   and whether we still need gradinet clipping because we use small batches
#   which give noisy gradients: so far the gradient norm is very stable and <1
#   but it'd be a good idea to have in place for cases where we increase lr abit
#   or made changes in other hyperparameters. with gradnorm now part of training log
#   it can be easily spotted what is amiss!
# 
# stylegan1_cifar10_20251115144021:
#   training cifar10 with fp16, normal batchsize(like stylegan1_cifar10_20251114082051)
#   lr=0.003: no nans as of epoch4@16x16, but the fp32 is considerably faster at convergence
#   e.g. the 16x16 had to go nearly 30 epochs to show some images that resemble cifar10
#   objects, until then it looked mostly blobs of color!even at epoch 40, the objects dont
#   look properly formed where as in fp32 experiment they were looking good at early epochs!
#   it shows, we definitely need to optimize the training somehow/fp16 cant be used like this
# 
# stylegan1_cifar10_20251115182244:
# - wrapped generator in autocast previously only the loss section was put in autocast section
#   wrapped the discriminator loss calculation part with autocast enabled=False:
#   faced Nans at epoch 9 @4x4 !
# 
# - cast all important tensors to float both in discriminator and generator and disabled
#   autocast for generator loss calculation, this fixed the nan errors that we got
#   when we wrapped generator in autocast. now it trains well, the convergence is also
#   much faster.next remove excessive casts and see which part specifically needs fp32 

# stylegan1_cifar10_20251115214132/
# stylegan1_cifar10_20251115223041:
# - removed all casts, only r1_penalty needed to be done in fp32, so the rest of the 
#   code isnt changed. this should now give us a bit more speed as unlike before,
#   only r1_penalty part is done in fp32 not the whole discriminator's loss.
#   the images are not looking good, at least as not as the fp32 counterpart.
#   they are malformed, at least it seems it might need more epochs! the fp32
#   is considerably better formed espcially in last 4~5 epochs of 32x32 res.
#   also I have not noticed any drastic speedup ups until 64x64. at 64x64 each epoch
#   takes 7mins to complete(in fp32 it takes 15mins)
#
# - a quick test on ffhq to affirm everything is working properly: faced nans in e10@4x4
#   lowered the lr = 0.001 to get rid of it. his happenes probably because ffhq
#   has a much more complex distirbution than cifar10, so earlyon we can get huge gradients
#   even though we have semi large batches, however having large lr like 0.003
#   will cause issues as we said fp16 has quit limited range and can easily overflow
#   add to that the adam optimizer.(sidenote cifar10 has more semantic complexity, i.e.
#   more classes, but from pixel level complexity, ffhq is more complex.(the network 
#   needs to get everything right for an image to look right, lighnig, textures,shapes, 
#   statical corrolations between pixels,etc)
#   otherwise we will reject the result as bad, however, in cifar10, even if the class
#   in question, a dog e.g. is malformed, its still a dog and at that low resolution 
#   we dont care much, details are limited compared to ffhq. so I guess thats the cause
#   and we fixed the issue as well with lower lr!)
# 

#
# %% styleGAN2
# Stylegan2 Analyzing and Improving the Image Quality of StyleGAN https://arxiv.org/pdf/1912.04958
# we had great success with stylegan1 and also learned a lot!
# now lets go for the second paper, stylegan2 and see what its baout!
# before we continue I strongly recommend to read the paper. 
# espcially have a look at the pages 11 and onward where it talks about
# implementation details and other interesting stats like gpu hours
#  
# the paper sums up the issues in the styelgan1 as two major design falws
# one steming from the way adaIN normalization was done (i.e. the blob artifact
# or water drop artifact that was present in some images, the ones that didnt
# show it, also had it in their featuremaps!) and the other
# related to the progressive nature of the architecture that lead to
# some artifacts of its own (i.e. caused 'phase' artifact see fig6 page 6)
#  The key issue is that the progressively grown generator appears to have a strong location
# basically the issue seems to be, the progressively grown generator has a 
# location prereference for details, for example features like teeths or eyes
# may remain stuck(remain the same) when we move smoothly over the image (i.e. interpolating
# smoothly) before suddenly jumping to the next preferred location. see 
# the acompayning video https://www.youtube.com/watch?v=c-NJtV9Jvp0
# which demonstrates this). logically they should move smoothly over the image
# the authors add that they believe the problem is that in the progressive growing
# "...each resolution serves momentarily as the output resolution, forcing it to
# generate maximal frequency details, which then leads to the trained network to
# have excessively high frequencies in the intermediate layers, compromising
# shift invariance". for this reason the dropped the progressive growing and instead
# opted to use MSG-GAN inspired architecture which they modified further by
# adding residual connections that would ultimately make it similar to LAPGAN!
# without the per-resolution discriminators!
# they add that the residual connections in discriminator's were helpful whereas
# they were harmful in generators!(with an exception for lsun dataset experiments)
# so at the end they used a skip generator and a residual discriminator without
# progressive growing.
# 
# from paper:
# "... The key aspect of progressive growing, which we would
# like to preserve, is that the generator will initially focus on
# low-resolution features and then slowly shift its attention to
# finer details. The architectures in Figure 7 make it possible
# for the generator to first output low resolution images that
# are not affected by the higher-resolution layers in a significant way, 
# and later shift the focus to the higher-resolution layers as the
# training proceeds. Since this is not enforced in any way, 
# the generator will do it only if it is beneficial. 
# To analyze the behavior in practice, we need to quantify how
# strongly the generator relies on particular resolutions over
# the course of training.
# Since the skip generator (Figure 7b) forms the image by
# explicitly summing RGB values from multiple resolutions,
# we can estimate the relative importance of the corresponding 
# layers by measuring how much they contribute to the
# final image. In Figure 8a, we plot the standard deviation of
# the pixel values produced by each tRGB layer as a function
# of training time. We calculate the standard deviations over
# 1024 random samples of w and normalize the values so that
# they sum to 100%."
#
# ...At the start of training, we can see that the new skip
# generator behaves similar to progressive growing — now
# achieved without changing the network topology. It would
# thus be reasonable to expect the highest resolution to 
# dominate towards the end of the training. The plot, however,
# shows that this fails to happen in practice, which indicates
# that the generator may not be able to “fully utilize” the target resolution.
# To verify this, we inspected the generated images manually and
# noticed that they generally lack some of the pixel-level detail
# that is present in the training data — the images could be described
# as being sharpened versions of 512² images instead of true 10242 images.
# This leads us to hypothesize that there is a capacity problem in our networks,
# which we test by doubling the number of feature maps in the highest-resolution 
# layers of both networks .
# This brings the behavior more in line with expectations:
# Figure 8b shows a significant increase in the contribution of the highest-resolution layers,
# and Table 1, row F shows that FID and Recall improve markedly.
# The last row shows that baseline StyleGAN also benefits from additional
# capacity, but its quality remains far below StyleGAN2.
# ..."
# 
# Projection of images to latent space
# Inverting the synthesis network g, i.e. manipulating a given image in the latent space:
#
# "...Manipulating a given image in the latent feature space requires finding a matching
# latent code w for it first. Previous research [1, 10] suggests
# that instead of finding a common latent code w, the results
# improve if a separate w is chosen for each layer of the generator. 
# The same approach was used in an early encoder implementation [32]. 
# While extending the latent space in this fashion finds a closer match to a given image,
# it also enables projecting arbitrary images that should have no latent representation.
# 
# Instead, we concentrate on finding latent codes in the original, unextended latent space,
# as these correspond to images that the generator could have produced.
# Our projection method differs from previous methods in two ways. 
# First, we add ramped-down noise to the latent code during optimization in order to explore
# the latent space more comprehensively. 
# Second, we also optimize the stochastic noise inputs of the StyleGAN generator, regularizing
# them to ensure they do not end up carrying coherent signal. 
# The regularization is based on enforcing the autocorrelation coefficients of the noise maps
# to match those of unit Gaussian noise over multiple scales. Details of our projection method
# can be found in Appendix"
#
# Attribution of generated images: 
# "Detection of manipulated or generated images is a very
# important task. At present, classifier-based methods can
# quite reliably detect generated images, regardless of their
# exact origin [29, 45, 40, 51, 41]. However, given the rapid
# pace of progress in generative methods, this may not be a
# lasting situation. Besides general detection of fake images,
# we may also consider a more limited form of the problem:
# being able to attribute a fake image to its specific source [2].
# With StyleGAN, this amounts to checking if there exists a
# w ∈ W that re-synthesis the image in question.
# We measure how well the projection succeeds by computing the LPIPS [50] 
# distance between original and resynthesized image as DLPIPS[x, g(˜g−1 (x))], 
# where x is the image being analyzed and g˜−1 denotes the approximate projection operation.
# Figure 10 shows histograms of these distances for LSUN CAR and FFHQ datasets
# using the original StyleGAN and StyleGAN2, and Figure 9 shows example projections.
# The images generated using StyleGAN2
# can be projected into W so well that they can be almost
# unambiguously attributed to the generating network. However, with the original StyleGAN,
# even though it should
# technically be possible to find a matching latent code, it appears that the mapping from W 
# to images is too complex for this to succeed reliably in practice. 
# We find it encouraging that StyleGAN2 makes source attribution easier even though the
# image quality has improved significantly"
    
# they also noted that both FID and P&R(i.e. Precision and Recall) metrics that were used
# to quantitvely analyze the outputs, focus on textures rather than shapes and thus do not
# accurately capture all aspects of image quality. they also found out the Perceptual Path
# Length(PPL) metric that they origibnally introduced for estimating the quality of lating
# space interpolations does a much better job and corrolates with consitency and stablity of
# shapes hence why they decided to regularize the synthesis netwrk(generator) so it
# favors smooths mappings and as consequnetly achieved a clear improvement in quality.
# 
# they also added they found "that projection of images to the latent space W worked 
# significantly better with the new, pathlength regularized StyleGAN2 generator
# than with the original StyleGAN. This makes it easier to attribute a genertaed image
# to its source!"
#
# implementation wise, the stylegan2 closely resembles stylegan1, the basic building blocks
# are still the same, BatchSTDDev, MappingNetwork, r1penaly, non-saturating logistic loss,,
# euqlized learing rate, stylemixing, truncation trick, ema of generators weights, nearly 
# everything is intact except a few items like the network architecture itself and the
# normalization process which now insead of normalizing the pixel features using adaIn, 
# we now normalize the weights of the convolution layer . this preserves the statistics (mean/std)
# while preventing the droplet artifact that was caused by instance normalization in adaIN module. this is refered to as weight modulation.
#
# (they also uses bilinear upsampling this time around)

#sidenote: 
# the paper syas the value for gamma γ is considerably different for different datasets
# for example for ffhq its 10 but for lsun they used γ=100!
# "...We have found that the optimal choices
# for the training length and R1 regularization weight γ tend
# to vary considerably between datasets and configurations.
# We use γ = 10 for all training runs except for configuration E in Table 1,
# as well as LSUN CHURCH and LSUN HORSE in Table 3, where we use γ = 100."
#

#%%
# blur() is not used in this version and instead the authors used a cuda kernel
# for the upsample/downsample called upfirdn2d!
# I initially was going to stick to simple upsample/downsample and not
# bother with this detail, but later changed my mind and went to
# implement it after I noticed the actual thing requires custom cuda
# kernels being compiled and then used, I didnt like the hassle and
# asked chatgpt to give me a faithful implementation but in python!
# update:
# the chatgpt version was not only buggy but extremely slow
# and on top of that it used view, pad and permute a lot. 
# permute itself breaks tensor contiguity and forces pytorch
# to copy the entire memory block so it slows things down considerably!
# so I forgot about it and instead went for a better implementation.
# the new implementation uses conv_transpose2d and conv2d
# that have optimized cuda kernels, and therefore are significantly 
# faster than maual indexing in previous impl. unlike the first implenetation
# this new one calculates the specific cropping/padding thats needed to
# match the exact pixel alignment in original/official stylegan2 accurately
# (the code is from gemini):
# update fournd a good explanation for upfirdn2d:
# read https://ppeetteerrsx.com/post/cuda/stylegan_cuda_kernels/
# update2: 
# I trained the model without upfirdn2d, using the old upsample/downsample
# we had, and the training went smoothly, I didnt see any differences
# with the same config, the output looked the same to me. maybe the
# effects it has will be more appraent when we do some interpolation?
# im not sure, see the debug log for the future updates!
def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)
    if k.ndim == 1:
        k = k[:, None] * k[None, :]
    k = k / k.sum()
    return k

def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
    """
    Fast pure-torch implementation of StyleGAN2 upfirdn2d.
    
    Args:
        input: (B, C, H, W)
        kernel: (kH, kW) - separable or 2d kernel
        up: int, upsampling factor
        down: int, downsampling factor
        pad: tuple (pad0, pad1) represents (pad_x, pad_y). 
             If you have 4 pad values, pass them accordingly.
    """
    out = input
    
    # Prepare kernel
    # (kH, kW) -> (1, 1, kH, kW)
    kernel = kernel.to(input.device, dtype=input.dtype).unsqueeze(0).unsqueeze(0)
    
    # If the kernel is small (like [1,3,3,1]), flipping doesn't matter much 
    # as it is symmetric, but strictly speaking StyleGAN flips it.
    kernel = torch.flip(kernel, [2, 3])

    # input_height = input.shape[2]
    # input_width = input.shape[3]
    
    # --------------------------------------
    # 1. UPSAMPLE (using ConvTranspose2d)
    # --------------------------------------
    if up > 1:
        # Reshape kernel for Grouped ConvTranspose: (C, 1, kH, kW)
        # We repeat the kernel for every channel, and use groups=C
        # This keeps channels independent.
        w = kernel.repeat(input.shape[1], 1, 1, 1)
        
        # Scaling factor is required because conv_transpose spreads energy
        w = w * (up ** 2) 
        
        # Calculate padding
        # Standard StyleGAN2 padding logic for upsampling:
        # pad_x0 = (kW - up) // 2
        # pad_x1 = (kW - up + 1) // 2
        # pad_y0 = (kH - up) // 2
        # pad_y1 = (kH - up + 1) // 2
        
        # However, ConvTranspose2d output size logic is:
        # H_out = (H_in - 1) * stride - 2 * padding + kernel_size
        # We manipulate F.pad on the input to achieve exact alignment.
        
        # Simply pad input to simulate the "valid" region
        p_x0 = (kernel.shape[3] - up) // 2
        p_x1 = (kernel.shape[3] - up + 1) // 2
        p_y0 = (kernel.shape[2] - up) // 2
        p_y1 = (kernel.shape[2] - up + 1) // 2
        
        # Apply user supplied pad if it exists (StyleGAN2 passes pad args)
        if len(pad) == 4:
            p_x0 += pad[0]
            p_x1 += pad[1]
            p_y0 += pad[2]
            p_y1 += pad[3]
        elif len(pad) == 2:
            p_x0 += pad[0]
            p_x1 += pad[0]
            p_y0 += pad[1]
            p_y1 += pad[1]

        # In ConvTranspose, 'padding' argument crops the output. 
        # But it's easier to verify by using F.conv_transpose2d with padding=0
        # and cropping manually or padding input carefully.
        
        # Let's use the standard "Rosinality" approach which is robust:
        # 1. Pad input manually
        out = F.pad(out, (0, 0, 0, 0)) # Placeholder if we needed pre-padding
        
        # 2. Run Transpose Conv
        # We let the conv expand it, then we crop the edges to match StyleGAN math
        out = F.conv_transpose2d(out, w, stride=up, padding=0, groups=out.shape[1])
        
        # 3. Crop to remove the extra pixels introduced by kernel width
        # The output of conv_transpose is larger than we want.
        # We need to crop (p_x0, p_x1, p_y0, p_y1) from the edges.
        
        # Compute dimensions
        h_new = out.shape[2]
        w_new = out.shape[3]
        
        out = out[:, :, p_y0 : h_new - p_y1, p_x0 : w_new - p_x1]

    # --------------------------------------
    # 2. DOWNSAMPLE (using Conv2d with stride)
    # --------------------------------------
    elif down > 1:
        # Reshape kernel for Grouped Conv2d: (C, 1, kH, kW)
        w = kernel.repeat(input.shape[1], 1, 1, 1)
        
        # Calculate padding
        # Standard StyleGAN2 padding logic for downsampling:
        # pad_x0 = (kW - down + 1) // 2
        # pad_x1 = (kW - down) // 2
        
        p_x0 = (kernel.shape[3] - down + 1) // 2
        p_x1 = (kernel.shape[3] - down) // 2
        p_y0 = (kernel.shape[2] - down + 1) // 2
        p_y1 = (kernel.shape[2] - down) // 2

        if len(pad) == 4:
            p_x0 += pad[0]
            p_x1 += pad[1]
            p_y0 += pad[2]
            p_y1 += pad[3]
            
        # Pad the input
        out = F.pad(out, (p_x0, p_x1, p_y0, p_y1))
        
        # Apply Strided Convolution
        out = F.conv2d(out, w, stride=down, padding=0, groups=out.shape[1])

    # --------------------------------------
    # 3. KERNEL ONLY (No scaling, just filtering)
    # --------------------------------------
    else:
        # Just a standard depthwise convolution with padding to maintain size
        w = kernel.repeat(input.shape[1], 1, 1, 1)
        
        p_x = (kernel.shape[3] - 1) // 2
        p_y = (kernel.shape[2] - 1) // 2
        
        if len(pad) == 4:
            out = F.pad(out, pad)
        else:
            out = F.pad(out, (p_x, p_x, p_y, p_y)) # Symmetric padding if not specified
            
        out = F.conv2d(out, w, stride=1, padding=0, groups=out.shape[1])

    return out

# WRAPPERS
FIR_KERNEL = make_kernel([1, 3, 3, 1])

def upsample_2d(x, kernel=FIR_KERNEL, factor=2):
    # Wrapper handles the default kernel
    return upfirdn2d(x, kernel, up=factor)

def downsample_2d(x, kernel=FIR_KERNEL, factor=2):
    # Wrapper handles the default kernel
    return upfirdn2d(x, kernel, down=factor)
    
class Blur(nn.Module):
    def __init__(self):
        super().__init__()
        kernel = torch.tensor([1,2,1])
        kernel = kernel.view(1,-1) * kernel.view(-1,1)
        kernel = kernel/kernel.sum()
        self.register_buffer("kernel", kernel.view(1,1,3,3))

    def forward(self, x):
        in_channels = x.size(1)
        kernel = self.kernel.repeat(in_channels, 1,1,1)
        x = F.pad(x,[1,1,1,1],mode='reflect')
        return F.conv2d(x, kernel, groups=in_channels)

class EqualizedLinear(nn.Linear):
    def __init__(self, in_features, out_features, lr_mult=1, device=None, dtype=None):
        super().__init__(in_features, out_features, bias=True, device=device, dtype=dtype) 
        self.lr_mult = lr_mult
        self.weight.data.normal_(0,1/lr_mult)
        self.scaler = (1/math.sqrt(self.in_features))*self.lr_mult
        
    def forward(self, x):
        return F.linear(x, self.weight*self.scaler, self.bias*self.lr_mult)

class EqualizedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, lr_mult=1.0, bias=True):
        super().__init__()

        self.bias = bias
        self.stride = stride
        self.padding = padding
        self.lr_mult = lr_mult

        # initialize the weights 
        self.weight = nn.Parameter(torch.randn(size=(out_channels, in_channels, kernel_size, kernel_size)))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        
        fan_in = in_channels * kernel_size * kernel_size
        self.scaler = math.sqrt(2/(fan_in)) * lr_mult
        
    def forward(self, x):
        scaled_weights = self.weight * self.scaler
        scaled_bias = self.bias * self.lr_mult if self.bias is not None else None
        return F.conv2d(x, scaled_weights, scaled_bias, stride=self.stride, padding=self.padding)

# the discriminator block now uses residual connections
# so we need to add that here as well
class DiscBlockStyleGAN2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, bias=False, use_upfirdn2d=True):
        super().__init__()
        self.use_upfirdn2d = use_upfirdn2d
        self.blur = Blur()
        self.block = nn.Sequential(EqualizedConv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
                                   nn.LeakyReLU(0.2),
                                   EqualizedConv2d(out_channels, out_channels, kernel_size, stride, padding, bias=bias),
                                   nn.LeakyReLU(0.2),
                                   # instead of blur and downsample we use the new downsample_2d
                                   # in forward
                                   # Blur(),
                                   #nn.AvgPool2d(2),
                                  )
        # for residual connection we can simply downsample the input and add it to output
        # but having a linear transformation like conv doesnt hurt and usually is usuful
        # nonetheless
        self.skip = EqualizedConv2d(in_channels, out_channels, kernel_size=1, bias=False)
        
        if use_upfirdn2d:
            self.downsample_fn = downsample_2d
        else:
            avg_pool = partial(F.avg_pool2d, kernel_size=2, stride=2)
            self.downsample_fn = avg_pool
            #or use blur like stylegan1?
            # self.downsample_fn = lambda x : avg_pool(self.blur(x))
        
    def forward(self, x):
        skip = self.skip(x)
        # skip = F.avg_pool2d(skip,2,2)
        skip = self.downsample_fn(skip)
        out = self.block(x)
        out = self.downsample_fn(out)
        return out+skip
    
class DiscriminatorStyleGAN2(nn.Module):
    def __init__(self, channels=[512,512,512,256,128,64,32], use_upfirdn2d=True):
        super().__init__()

        self.setup_layers(channels,use_upfirdn2d)
    
    def setup_layers(self, channels, use_upfirdn2d):
        self.channels = channels
        # whether to use the upfrdn2d or normal avgpool for downsampling
        self.use_upfirdn2d = use_upfirdn2d
        
        self.fromImgs = nn.Sequential(EqualizedConv2d(3, self.channels[-1], kernel_size=1),
                                      nn.LeakyReLU(0.2))
        
        blocks = []
        for i in range(len(self.channels)-1,0,-1):
            blocks.append(DiscBlockStyleGAN2(self.channels[i],self.channels[i-1], use_upfirdn2d=use_upfirdn2d))
            
        self.blocks = nn.Sequential(*blocks)
        
        self.final = nn.Sequential(AddBatchStdDev(),
                                   EqualizedConv2d(self.channels[0]+1, self.channels[0], kernel_size=3, padding=1),
                                   nn.LeakyReLU(0.2),
                                   nn.Flatten(),
                                   EqualizedLinear(self.channels[0]*4*4, self.channels[0]),
                                   nn.LeakyReLU(0.2),
                                   EqualizedLinear(self.channels[0],1))

    
    def forward(self, x):
        out = self.fromImgs(x)
        out = self.blocks(out)
        out = self.final(out)
        return out.view(-1,1)
 
 # in mapping network we dont use any normalization either (no pixelnorm)
class MappingNetwork2(nn.Module):
    def __init__(self, z_dim=512, w_dim=512, num_layers=8):
        super().__init__()
        layers = [] 
        for i in range(num_layers):
            layers.append(EqualizedLinear(z_dim if i==0 else w_dim, w_dim, lr_mult=0.01))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.net = nn.Sequential(*layers)
        
    def forward(self, z):
        # we dont use any pixelnorm in mapping network anymore
        return self.net(z)

# since weight modulation is really a simple linear layer applied on w
# we implement it as a part of the conv module instead of a seprate one

# this was supposed to be applied on EqualizedConv2d weights, but since our
# modulation operation cancels out the scaling part of the equalized conv2d
# then we dont need to bother bother writting that down! also we need the plain
# EqualizedConv2d for discriminator, so we implement this as a new module
class ModulatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, w_dim=512, demodulate=True, eps=1e-8):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size= kernel_size
        self.padding = padding
        self.stride = stride
        self.w_dim = w_dim
        self.demodulate = demodulate
        self.eps = eps
        
        self.weight = torch.nn.Parameter(torch.randn(size=(1,out_channels, in_channels, kernel_size, kernel_size)))
        # weight modulation is simply fc layer transforming w and acts as a scale
        self.fc_style = EqualizedLinear(w_dim, in_channels)
        # initialize the bias/scale to 1
        self.fc_style.bias.data.fill_(1)

    def forward(self, x, w):
        b,c,img_h,img_w = x.shape
        # shape:(batch,1,in_channel,1,1)
        style = self.fc_style(w).view(-1,1,c,1,1)
        weights = self.weight * style 
        
        if self.demodulate:        
            # weight demodulation.(remember sqrt/rsqrt accept only positive numbers, any negatives results in nan!)
            demod = torch.rsqrt(weights.pow(2).sum((2,3,4))+self.eps)
            weights = weights * demod.view(-1,self.out_channels,1,1,1)

        # reshape x for group convolution trick
        # to handle different weighst for each batch sample,
        # we put the batch dim into output channels and use groups=batch
        x = x.view(1, b*c,img_h,img_w)
        weights = weights.view(b*self.out_channels, c, self.kernel_size, self.kernel_size)
        
        # apply the conv operationg using the weights
        out = F.conv2d(x, weights, stride=self.stride, padding=self.padding, groups=b)
        # reshape back to original shape
        out = out.view(b, self.out_channels, img_h, img_w)
        return out
                
class NoiseInjection(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1,channels,1,1))

    def forward(self, x, noise=None):
        if noise is None:
            noise = torch.randn(size=(x.size(0), 1, x.size(2), x.size(3)), device=x.device)
        else: 
            noise = F.interpolate(noise, size=(x.size(2), x.size(3)))
        return x+(self.weight*noise)

class StyleConvBlock2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True,
                 w_size=512, upsample=False, eps=1e-8, demodulate=True, use_upfirdn2d=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.w_size = w_size
        self.upsample = upsample
        self.eps = eps
        self.demodulate = demodulate
        # whether to choose upfirdn2d or normal interpolate to upsample
        self.use_upfirdn2d = use_upfirdn2d

        self.conv = ModulatedConv2d(in_channels, out_channels, kernel_size, stride, padding, w_size, demodulate,eps)
        self.noise_inject = NoiseInjection(out_channels)
        self.bias = nn.Parameter(torch.zeros(out_channels,))
        self.blur = Blur()
        
        if use_upfirdn2d:
            self.upsample_fn = upsample_2d
        else:
            interp = partial(F.interpolate, scale_factor=2, mode="bilinear", align_corners=False)
            self.upsample_fn = interp
            #or use blur like stylegan1?
            # self.upsample_fn = lambda x : self.blur(interp(x))


    def forward(self, x, w, noise=None):
        # sg2 does conv then upsample?
        if self.upsample:
            # instead of upsample+blur, we now use upsample2d(from upfirdn2d)
            # we can use upsample/blur but we may not get the exact results 
            # although if I recall correctly, this was done only for gaining 
            # more performance and less vram usage.
            # anyway lets be faithful to the original impl as much as we can
            # when we got our results, we can always switch back to this again and
            # see how much of an impact it has on our results
            # also if we were to approximate this without writting the upfirdn2d,
            # we could use f.interpolate(mode="bicubic") for upsample since its 
            # closer to [1,3,3,1] than bilinear and for downsample conv2d with 
            # guassian blur kernel.(the kernel is exactly the outer product of 
            # the [1,3,3,1] with itself, normalized so sum is 1, it'd be a 4x4 kernel. 
            # sample code (from gemini):
            # def get_stylegan_blur_kernel(channels, device='cuda'):
            #     # Define the base vector [1, 3, 3, 1]
            #     k = torch.tensor([1, 3, 3, 1], dtype=torch.float32, device=device)
            #    
            #     # Create the 2D matrix (Outer Product)
            #     # Shape becomes [4, 4]
            #     k2d = torch.outer(k, k)
            #    
            #     # Normalize (Sum must be 1.0 to preserve brightness)
            #     k2d = k2d / k2d.sum()
            #    
            #     # Reshape for PyTorch Conv2d: [Out_Channels, In_Channels/Groups, H, W]
            #     # For a depthwise convolution (applying blur to each channel independently),
            #     # In_Channels/Groups must be 1.
            #     kernel = k2d.view(1, 1, 4, 4)
            #    
            #     # Repeat for every channel in your image
            #     return kernel.repeat(channels, 1, 1, 1)
            # 
            # # example:
            # channels = x.shape[1]
            # blur_kernel = get_stylegan_blur_kernel(channels, device=x.device)
            # # stride=2 for downsampling
            # # padding=1 for making sure the output size is exactly half (64 -> 32)
            # F.conv2d(x, weight=gaussian_blur_kernel, stride=2, padding=1, groups=channels))
            
            # we can now experiment with the old method as well     
            x = self.upsample_fn(x)

        out = self.conv(x, w)
            
        # inject noise into the output featuremaps,
        out = self.noise_inject(out, noise)
        
        if out.ndim==2:
            out += self.bias
        else:
            out += self.bias.view(1,-1,1,1)

        out = F.leaky_relu(out, negative_slope=0.2)
        return out

# now unlike the previous version, we no more use progressive growing, 
# so now each block generates the image and adds it to the previous upsampled image
# (the old image is upsampled and then added to the new image from current resolution)
# and the final image will be the summed up version of all previous images. 
class GeneratorStyleGAN2(nn.Module):
    def __init__(self, z_size=512, w_size=512, mn_num_layers=8,
                 channels=[512,512,512,256,128,64,32],
                 style_mixing_prob=0.9, ema_w_beta=0.995, 
                 use_upfirdn2d=True, eps=1e-8):
        super().__init__()
        
        self.setup_layers(z_size, w_size, mn_num_layers,
                          channels, style_mixing_prob, 
                          ema_w_beta, use_upfirdn2d, eps)
                
    def setup_layers(self, z_size, w_size, mn_num_layers,
                     channels, style_mixing_prob, ema_w_beta, use_upfirdn2d, eps):
        
        self.z_size = z_size
        self.w_size = w_size
        self.mn_num_layers = mn_num_layers
        
        self.style_mixing_prob = style_mixing_prob
        self.ema_w_beta = ema_w_beta
        self.register_buffer("ema_w",torch.zeros(size=(1,w_size)))
        # whether to use the upfirdn2d for upsampling or not
        self.use_upfirdn2d = use_upfirdn2d
        # for fp16 tests in case we faced instabilityies that require larger eps
        self.eps = eps
        
        self.channels = channels
        
        # in the second version we use randn instead of just ones
        self.const_input = nn.Parameter(torch.randn(size=(1, self.channels[0], 4, 4)))
        
        self.mapping_network = MappingNetwork2(z_size, w_size, self.mn_num_layers)
        
        self.blocks = nn.ModuleList()
        self.toImgs = nn.ModuleList()
        
        self.blocks.append(StyleConvBlock2(self.channels[0], self.channels[0], w_size=w_size, upsample=False,eps=eps,use_upfirdn2d=use_upfirdn2d))
        self.blocks.append(StyleConvBlock2(self.channels[0], self.channels[0], w_size=w_size, upsample=False,eps=eps,use_upfirdn2d=use_upfirdn2d))
        self.toImgs.append(ModulatedConv2d(self.channels[0], 3, kernel_size=1, w_dim=w_size, demodulate=False,eps=eps))
        for i in range(1, len(channels)):
            self.blocks.append(StyleConvBlock2(self.channels[i-1], self.channels[i], w_size=w_size, upsample=True,eps=eps,use_upfirdn2d=use_upfirdn2d))
            self.blocks.append(StyleConvBlock2(self.channels[i], self.channels[i], w_size=w_size, upsample=False,eps=eps,use_upfirdn2d=use_upfirdn2d))
            self.toImgs.append(ModulatedConv2d(self.channels[i], 3, kernel_size=1, w_dim=w_size, demodulate=False,eps=eps))

    @torch.no_grad()
    def _update_ema_w(self, w_batch):
         if self.training:
             self.ema_w.mul_(self.ema_w_beta).add_(w_batch.mean(0), alpha=1-self.ema_w_beta)
    
    # this time lets separate the forward pass using w
    # to make it easier for calculating path length regularization
    # and future interpolation experimets
    def forward_from_w(self, w, noise=None):
        if w.ndim==2:
            w = w.unsqueeze(1).repeat(1, len(self.blocks),1)
        
        # set the batchsize forr const_input/canvas
        x = self.const_input.repeat(w.size(0), 1,1,1)
        
        #4x4
        x = self.blocks[0](x, w[:,0,:], noise)
        x = self.blocks[1](x, w[:,1,:], noise)
        # grab the first image(i.e old image )
        img = self.toImgs[0](x, w[:,1,:])
        
        # main loop, skip connections
        for i in range(1, len(self.channels)):
            idx1 = 2*i
            # upsample the previous img so we can add it to the new image(current resolution)
            img = F.interpolate(img, scale_factor=2, mode='bilinear', align_corners=False)
            #process the input for this resolution
            x = self.blocks[idx1](x, w[:, idx1,:], noise)
            x = self.blocks[idx1+1](x, w[:, idx1+1,:], noise)
            # create the image for current resolution and add it to the previous one
            img = img + self.toImgs[i](x, w[:, idx1+1,:])
            
        return img

    def apply_truncation(self, w, psi):
        ema_w_batch = self.ema_w.repeat(w.size(0),1)
        w = ema_w_batch + psi * (w - ema_w_batch)
        return w
           
    def forward(self, z, psi=None, noise=None):
        num_layers = len(self.blocks)
        w = self.mapping_network(z)
        
        if self.training:
            self._update_ema_w(w)

        if not self.training and psi:
            w = self.apply_truncation(w, psi)
        
        if self.training and random.random() <self.style_mixing_prob:
            # grab a second z, calculate the w
            z2 = torch.randn(size=z.size(), device=z.device)
            w2 = self.mapping_network(z2)
            # in stylegan2 we use all the layers,as
            # theres no steps!/prograssive growing!
            crossover_point = random.randint(1, num_layers-1)
            w = w.unsqueeze(1).repeat(1, num_layers,1)# (b,num_styles,w_dim)
            w[:, crossover_point:,:] = w2.unsqueeze(1).repeat(1, num_layers-crossover_point,1)
        else:
            w = w.unsqueeze(1).repeat(1, num_layers,1)

        img = self.forward_from_w(w, noise)
        # since we want to calculate path length regularization,
        # we need w for each generated image so we can see how much
        # an image changes when w changes (calculate its gradient with
        # respect to w so aside from making our forward to work with w
        # we can simply return w with the generated images as well and
        # make our life easier!)
        return img,w
    
channels=[512,256,128,64,32,16,8]
use_upfirdn2d=True
disc = DiscriminatorStyleGAN2(channels=channels,use_upfirdn2d=use_upfirdn2d)
gen = GeneratorStyleGAN2(100,100,channels=channels,use_upfirdn2d=use_upfirdn2d)

for m in [disc, gen]:
    print(f'channels: {m.channels}')
H=W=2**(len(channels)+1)
x = torch.randn(size=(5,3,H,W))
z = torch.randn(size=(5,100))
n = torch.zeros((5,1,H,W))
disc_out = disc(x)
print(f'disc_out.shape: {tuple(disc_out.shape)}')
gen_out,ws = gen(z, noise=n)
print(f'gen_out.shape:\nimgs:{tuple(gen_out.shape)} ws:{tuple(ws.shape)}')
    
#%%
# the losses stay the same, we just need to apply the r1_penalty at some interval
# we can still do it all the time, but its inefficient and we can get a little perf
# boost by doing it intermittently!

# update:
# I separate the main loss and r1_penalty part 
# so we can be more accurate and faster see training loop log ahead
def discriminator_main_loss_stylegan2(d_preds_real,d_preds_fake):
    loss = (F.softplus(-d_preds_real) + F.softplus(d_preds_fake)).mean()
    return loss

# must run in a fp32 graph
def discriminator_r1_penalty_loss_stylegan2(discriminator, x_real, gamma=10):
    x_real.requires_grad_(True)
    d_preds_real = discriminator(x_real)
    
    grads = torch.autograd.grad(outputs=d_preds_real,
                                inputs=x_real,
                                grad_outputs=torch.ones_like(d_preds_real),
                                retain_graph=True,
                                create_graph=True)[0]
    grads_l2norm_squared = grads.pow(2).view(d_preds_real.size(0),-1).sum(1).mean()
    penalty = gamma/2 * grads_l2norm_squared

    x_real.requires_grad_(False)
    return penalty

def generator_loss_stylegan2(d_preds_fake):
    # G_loss = E[softplus(-D(G(z)))]
    return F.softplus(-d_preds_fake).mean()


# we also need Path length regularization loss for the generator
def path_length_regularization_loss(fake_imgs,
                                    w_latents,
                                    mean_path_length, 
                                    decay=0.01, 
                                    pl_weight=2):
    
    # get the noise (y) , random noise images
    # but we can use randn with the same shape as images aswell
    # noise/sqrt(img_h*img_w)
    noise = torch.randn_like(fake_imgs)/math.sqrt(math.prod(fake_imgs.shape[2:]))    
    # now we calculate the gradients of the image*noise w.r.t w latents
    # this measures how much the image changes when w changes
    # we use sum() so we dont do .mean at the end
    grad = torch.autograd.grad(outputs=(fake_imgs*noise).sum(),
                               inputs=w_latents,
                               create_graph=True)[0]
    
    path_lengths = torch.sqrt(grad.pow(2).sum(dim=2).mean(dim=1))
    # update the moving average (no grad)
    path_mean = mean_path_length + decay*(path_lengths.mean() - mean_path_length)
    # now calculate the penalty
    path_penalty = (path_lengths - path_mean).pow(2).mean()*pl_weight
    
    return path_penalty, path_mean.detach()


@torch.no_grad()
def update_ema_generator(gen:GeneratorStyleGAN2, 
                         gen_ema:GeneratorStyleGAN2, 
                         current_batch_size, 
                         images_seen,
                         ema_kimg=10_000, 
                         rampup_rate=0.05):
    # sidenote, we only update the parameters we dont touch buffers 
    # as it would have destroyed their stats!)
    # 
    # in stylegan2, the decay is calculated automatically based on
    # how many kimages the model has seen so far so we dont need to
    # keep copying weights manually until some kimages and then start the 
    # actual ema process. (imagine it as length of memory)
    # this is done all automcatically it goes like this, we use a rampup
    # rate which is 1/2 of the images seen so far and calculate the effective decay
    # like this decay = 0.5^(batch_size/half_life)
    # batchsize is there so we tune the decay automatically based on 
    # how many steps (updates) is required, forexample if we want 
    # the history to fade after 10k images, and we have batchsize=10
    # thats 1000 update steps,so we need to decay accordingly in this
    # 1000 steps, likewise if the batchsize is larger,e.g. 1000, then we 
    # need to decay in 10 steps! if we dont use this, when we train using 
    # a larger batchsize, it would result in an ema thats too fast/jittery
    # and when using smaller batchsize, it would result in an ema thats too slow
    # to update!
    
    # kimages here controls how far back we can remember, its called half-life as well
    # the model gives 50% importance to the last kimages seen and another 50% to 
    # everything that happened before that! so choosing the right value is very important
    # forexample if we chose a high value like 20/50, it would make the ema to change 
    # very slowly, it will be very smooth and stable but at the same time if the model
    # learns a new feature (e.g. like sunglasses,etc) it will treat it as temporary jitter
    # and the ema wont immediatley show it, it will take a long time to show it!
    # conversly, if we chose a small value like 1, the ema will therefore change much quickly
    # basically following the main model, it will reflect the main model, so we wont be
    # getting the averging/smoothing effect out of it! the same artifacts, etc will be there too
    # kimg = 10 was the sweetspot the authords found for 256x256, for higher resolutions like 1024x1024
    # they used 20. since we are training to128x128 in fp32, we choose 10 as well but if it 
    # doesnt turn out ok we can play with it!
    nimgs = ema_kimg * 1000
    # rampup is there so we can shorten the so called memory initially 
    # when the initial weights are garbage and then linearly increase it
    # until we hit the kimgs target, i.e. let the training progresses and weights
    # become more sensible!
    # if we dont do this, early on when weights are random, we will be
    # accumuliating them into ema and it will take a long time for them
    # to disappear(for the ema to forget those garbage weights).
    # so the result to look good will take a long time
    if rampup_rate is not None:
        nimg = min(nimgs, images_seen*rampup_rate)
        
    # 1e-8 is there to avoid division by zero
    decay = 0.5**(current_batch_size/max(nimg,1e-8))
    
    for ema_p,p in zip(gen_ema.parameters(), gen.parameters()):
        ema_p.data.mul_(decay).add_(p.data, alpha=1-decay)
    # # copy the ema_w over
    gen_ema.ema_w.copy_(gen.ema_w)
    
    
def training_loop_stylegan2(discriminator:DiscriminatorStyleGAN2, generator:GeneratorStyleGAN2, disc_optimizer:torch.optim.Adam, 
                         gen_optimizer:torch.optim.Adam, epochs, batch_size, dataset_name,
                         split, data_augmentation=False, normalize=True, use_fp16=False, 
                         path_length_interval=4, r1_penalty_interval=16, gamma=10, psi=0.7,
                         gen_num_samples = 64, use_ema_inference=False, kimg=10,
                         keep_raw_generations=True, quick_and_noisy_IS_FID=False, device='cuda',
                         resume=False, eps=1e-8, weights_save_dir='./weights/gan',
                         images_save_dir='./results/gan', checkpoint_path=None, ):
    
    experiment_date = datetime.now().strftime("%Y%m%d%H%M%S")
    current_experiment_name = f"stylegan2_{dataset_name}_{experiment_date}"

    lr_d = disc_optimizer.param_groups[0]["lr"]
    lr_g = gen_optimizer.param_groups[0]["lr"]
    # print(f'{lr_g=}')
    betas_d = disc_optimizer.defaults["betas"]
    betas_g = gen_optimizer.defaults["betas"]
    
    ema_generator = copy.deepcopy(generator).requires_grad_(False).eval()
    
    # real images seen so far during training
    ema_warmup_images_seen = 0
    
    metric = IS_FID_Calculator(device)

    fixed_z = torch.randn((gen_num_samples, generator.z_size)).to(device)

    starting_step = 0
    starting_epoch = 0
    last_training_step_counter = 0
    z_size = generator.z_size
    
    scaler = torch.amp.grad_scaler.GradScaler(device, enabled=use_fp16)
    
    # check for resuming from a checkpoint
    if resume:
        if checkpoint_path:
            checkpoint_filename = os.path.split(checkpoint_path)[-1]
        else:
            checkpoints_dirs = sorted([subdir for subdir in os.listdir(weights_save_dir)\
                                       if os.path.isdir(os.path.join(weights_save_dir, subdir))])
            #grab the last checkpoint/most recent one
            checkpoint_dirpath = os.path.join(weights_save_dir, checkpoints_dirs[-1])
            # grab the latest checkpoint 
            checkpoint_files = sorted([f for f in os.listdir(checkpoint_dirpath) if f.endswith(".ckpt")])
            checkpoint_filename = checkpoint_files[-1]
            checkpoint_path = os.path.join(checkpoint_dirpath, checkpoint_filename)
        
        if not os.path.exists(checkpoint_path):
            raise ValueError("The Path is not valid")
        
        # load the stuff
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        channels_d = checkpoint["channels_d"]
        disc_use_upfirdn2d = checkpoint.get("disc_use_upfirdn2d",True)
        discriminator.setup_layers(channels_d, disc_use_upfirdn2d)
        discriminator.load_state_dict(checkpoint["disc_state_dict"])
        discriminator = discriminator.to(device)
        
        z_size = checkpoint["z_size"]
        w_size = checkpoint["w_size"]
        mn_nlayer = checkpoint["mn_nlayer"]
        channels_g = checkpoint["channels_g"]
        style_mixing_prob = checkpoint["style_mixing_prob"]
        ema_w_beta = checkpoint["ema_w_beta"]
        gen_use_upfirdn2d = checkpoint.get("gen_use_upfirdn2d",True)
        
        generator.setup_layers(z_size, w_size, mn_nlayer, channels_g,
                               style_mixing_prob=style_mixing_prob,
                               ema_w_beta=ema_w_beta,
                               use_upfirdn2d=gen_use_upfirdn2d,
                               eps=eps)
        generator.load_state_dict(checkpoint["gen_state_dict"])
        generator = generator.to(device)
        
        scaler.load_state_dict(checkpoint.get("scaler_state_dict", scaler.state_dict()))
        
        # load the ema version, dont forget to also load the images seen so far!
        kimg = checkpoint["kimg"]
        ema_warmup_images_seen = checkpoint["ema_warmup_images_seen"]
        if use_ema_inference:
            ema_generator.load_state_dict(checkpoint["gen_ema_state_dict"])
            ema_generator = ema_generator.to(device)
        
        disc_optimizer.load_state_dict(checkpoint["disc_optimizer"])
        gen_optimizer.load_state_dict(checkpoint["gen_optimizer"])

        # grab the initial lrs
        lr_d = checkpoint["lr_d"]
        lr_g = checkpoint["lr_g"]
        betas_d = disc_optimizer.defaults["betas"]
        betas_g = gen_optimizer.defaults["betas"]

        data_augmentation = checkpoint["data_augmentation"]
        normalize = checkpoint["normalize"]
        use_fp16 = checkpoint["use_fp16"]
        last_training_step_counter = checkpoint["training_step_counter"]
        epochs = checkpoint["epochs"]
        starting_epoch = checkpoint["epoch"]+1
        batch_size = checkpoint["batch_size"]
        path_length_interval = checkpoint["path_length_interval"]
        r1_penalty_interval = checkpoint["r1_penalty_interval"]
        gamma = checkpoint["gamma"]

        
    # store training log
    all_training_losses = []
  
    print(f'StyleGAN1 Training on {dataset_name} in {experiment_date}')
        
    if resume:
        print(f'--Resume:                  {"N/A" if not resume else checkpoint_filename}'
            f'\n  --From Step:             {starting_step}'
            f'\n  --From Epoch:            {starting_epoch}'
            f'\n  --Checkpoint Path:       {checkpoint_path}'
            f'\n  --Last FID:              {checkpoint["FID"]}'
            f'\n  --Last IS:               {checkpoint["IS"][0]:.4f} ± {checkpoint["IS"][1]:.4f}')
          
    print(f'--Disc Param Count:          {sum([p.numel() for p in discriminator.parameters()]):,}')
    print(f'--Genr Param Count:          {sum([p.numel() for p in generator.parameters()]):,}')
    print(f'--MNetwork numlayers:        {generator.mn_num_layers}')    
    print(f'--style_mixing_prob:         {generator.style_mixing_prob}')
    print(f'--Disc use_upfirdn2d:        {discriminator.use_upfirdn2d}')
    print(f'--Genr use_upfirdn2d:        {generator.use_upfirdn2d}')
    print(f'--Dataset:                   {dataset_name}-{split}')
    print(f'--DataAugmentation:          {data_augmentation}')
    print(f'--Normalize[-1,1]:           {normalize}')
    print(f'--Use F16:                   {use_fp16}')
    print(f'--Discriminator LR:          {lr_d}')
    print(f'--Generator LR:              {lr_g}')
    print(f'--Epochs:                    {epochs}')
    print(f'--Batch-Size:                {batch_size}')
    print(f'--ema_kimage:                {kimg:,} ')
    print(f'--ema_real_images_seen:      {ema_warmup_images_seen:,} ')
    print(f'--Path Length Reg interval:  {path_length_interval}')
    print(f'--R1 Penalty Interval:       {r1_penalty_interval}')
    print(f'--Gama factor:               {gamma}')
    print(f'--PSI:                       {psi}')
    print(f'--gen_num_samples:           {gen_num_samples}')
    print(f'--Checkpoint Directory:      {weights_save_dir}')
    print(f'--Images Directory:          {images_save_dir}')
    
    if (use_fp16 and (lr_d>0.001 or lr_g>0.001)):
        print(f"⚠️ Warning! ⚠️ Large LR({lr_d},{lr_g}) for FP16 can lead to Nan! Decrease it for a stable training!")
    
    res = 2<<len(generator.channels)    
    train_loader = get_dataloader(dataset_name, split=split, 
                                  resize_dims=(res,res),
                                  batch_size=batch_size, 
                                  data_augmentation=data_augmentation,
                                  normalize=normalize)
    num_batches = len(train_loader)
    interval = num_batches//2+1
    training_step_counter = 0 if starting_epoch==0 else last_training_step_counter

    current_lr_d = [g['lr'] for g in disc_optimizer.param_groups]
    current_lr_g = [g['lr'] for g in gen_optimizer.param_groups]

    print(f'Training StyleGAN2 on [{res}x{res}]')
    print(f'  --Epochs:                      {epochs}')
    print(f'  --BatchSize:                   {batch_size}')
    print(f'  --Number of Batches:           {num_batches}')
    print(f'  --Interval:                    {interval}')
    print(f'  --Path Length Interval:        {path_length_interval}')
    print(f'  --R1-Penalty Interval:         {r1_penalty_interval}')
    print(f'  --Last training Step taken:    {training_step_counter}')
    print(f'  --Current Discriminator LRs:   {current_lr_d}')
    print(f'  --Current Generator LRs:       {current_lr_g}')
    print(f'  --Current Discriminator Betas: {betas_d}')
    print(f'  --Current Generator Betas:     {betas_g}')

    path_length_mean = 0
    for epoch in range(starting_epoch, epochs):
        discriminator.train()
        generator.train()

        losses = []
        epoch_scores = []
        for i, (imgs_real, _) in enumerate(train_loader):
            
            imgs_real = imgs_real.to(device)
            # we do it inside r1_penalty
            # imgs_real.requires_grad_(True)
            # track how many real images the network has seen for ema_generator
            ema_warmup_images_seen += imgs_real.size(0)

            # update: 
            # I initially did this and calculate the main loss+r1_penalty in one go
            # this was problematic as the r1penalty was done on a fp16 graph even doing
            # casting to float doesnt change the fact that the whole graph was already 
            # in fp16, so when we are calculating the gradients of gradients (what r1penalty
            # is basically doning) it involves squaring and summing gradients and we can
            # easily face numerical issues (we havent so far, but the fp16 experiments
            # have been visibly way inferiro to the fp32, they objects malformed etc 
            # if you see debug log for stylegan1 e.g.) I guess this could have likely caused it
            # anyway, the other issue is that this diminshes the speedup we should get in fp16
            # we cant do much about peak vram usage (nvidia-smi shows peack ram by the way)
            # as the r1_penaty and later plr loss need to be calculated in fp32 anyway so that
            # causes the vram to peak, but aside from that we should be able to get a modest speed boost!
            # so we first calculate the main loss, do backward() so it frees the graph,vram
            # then go for the r1_penalty, calculate it do a backward pass, the gradients will be
            # accumulated with the previous/existing onces and then do a optimizer step!
            #
            # with torch.amp.autocast(device_type="cuda", enabled=use_fp16):
            #     preds_real = discriminator(imgs_real)
            #     z_vector = torch.randn((imgs_real.size(0), z_size)).to(device)
            #     imgs_fake = generator(z_vector)[0].detach()
            #     preds_fake = discriminator(imgs_fake)
            # disc_loss = discriminator_loss_stylegan2(preds_real, imgs_real, preds_fake, gamma, i, r1_penalty_interval)
            #
            with torch.amp.autocast(device_type="cuda", enabled=use_fp16):
                preds_real = discriminator(imgs_real)
                z_vector = torch.randn((imgs_real.size(0), z_size)).to(device)
                imgs_fake = generator(z_vector)[0].detach()
                d_preds_fake = discriminator(imgs_fake)
                
                disc_loss_main = discriminator_main_loss_stylegan2(preds_real, d_preds_fake)
                
                disc_optimizer.zero_grad()
                # this frees the graph we can now go for r1_penalty part
                scaler.scale(disc_loss_main).backward()
                
            # unlike stylegan1, we dont need to apply penalty all the time!
            with torch.amp.autocast(device_type="cuda", enabled=False):
                # only apply penalty intermittently
                if (i%r1_penalty_interval)==0:
                    r1_penalty_term = discriminator_r1_penalty_loss_stylegan2(discriminator, imgs_real)
                    # remember to scale the penalty so on average 
                    # the same amount of regularization is applied at the end
                    r1_penalty_term *= r1_penalty_interval
                    # add the gradients to existing ones from main loss
                    scaler.scale(r1_penalty_term).backward()
            # and finally we do take an optimizer step
            scaler_out_d = scaler.step(disc_optimizer)
            
            # now train genertor to create images that look real
            with torch.amp.autocast(device_type="cuda", enabled=use_fp16):
                z_vector = torch.randn((imgs_real.size(0),z_size)).to(device)
                fake_imgs,_ = generator(z_vector)
                preds_fake = discriminator(fake_imgs)
            
                # generator loss
                # swap loss! treat fake images as real images
                gen_real_loss = generator_loss_stylegan2(preds_fake)

                # optimize generator
                # update generator with a delay, sylegan1 uses 1:1 update ratio
                gen_optimizer.zero_grad()
                scaler.scale(gen_real_loss).backward()
                
            # apply path length regularization every 4 iterations
            if i%path_length_interval==0: 
                # since we already did .backward()once, the computational graph is freed
                # we can do (backward(keep_graph=true)) but it takes vram so instead
                # we do a second foward pass and grab the fake_imgs, ws we want and a
                # active computational graph which we can use to calculate plr! 
                # also like r1_penalty we need to calculate this in fp32
                with torch.amp.autocast(device_type="cuda", enabled=False):
                    fake_imgs,w_latents = generator(z_vector)
                    plr_loss, path_length_mean = path_length_regularization_loss(fake_imgs,
                                                                                w_latents,
                                                                                path_length_mean)
                    # scale the plr_loss so on average the regularization stays the same
                    plr_loss*=path_length_interval
                # note since we havent done optimizer.step(), all backward()s
                # will accumulate the gradients as normal so we are ok!
                scaler.scale(plr_loss).backward()
                
            # note: the unscaling part must be done at the very end,when we are done
            # doing backward passes, just before we do step(), otherwise it will messup
            # the gradient accumulation (some will be scaled and some will be unscaled)
            # therefore when we take a step, it will result in a huge update, and either
            # we diverge asap or get nans. (we diverged! cuz I added plr later, after 
            # norm calculation and this made a whole mess. only after moving the norm calculation
            # after it everything became ok!)
            # monitor the norm (one of the params is enough)
            mn_grad_norm = torch.norm(next(generator.mapping_network.parameters()).grad)
            scaler.unscale_(gen_optimizer)
            nn.utils.clip_grad_norm_(generator.parameters(), max_norm=10)
            
            # take optimizer step
            scaler_out_g = scaler.step(gen_optimizer)
                
            if mn_grad_norm>100:
                print(f'Warning! mapping_network_grad_norm={mn_grad_norm.item():.4f}')# '{scaler_out_g=}')
            
            update_ema_generator(generator,
                                 ema_generator,
                                 imgs_real.size(0),
                                 ema_warmup_images_seen,
                                 ema_kimg=kimg)
            
            # update the scaler for the next round
            scaler.update()

            # for debugging purposes
            disc_loss = disc_loss_main+r1_penalty_term
            
            disc_real_mean = preds_real.mean().item()
            disc_fake_mean = d_preds_fake.mean().item()
            
            preds_detached = d_preds_fake.detach()
            disc_fake_std = preds_detached.std().cpu()
            disc_real_std = preds_real.detach().std().cpu()
            
            epoch_scores.append((disc_real_mean, disc_fake_mean))
                
            status_r = get_status(disc_real_mean, higher_is_better=True)
            status_f = get_status(disc_fake_mean, higher_is_better=False)
            status_o = get_overall_status(disc_real_mean, disc_fake_mean)

            d_real_stat_str = f"D_real_avg: {status_r} {disc_real_mean:+.4f} ± {disc_real_std:+.4f} 📈"
            d_fake_stat_str = f"D_fake_avg: {status_f} {disc_fake_mean:+.4f} ± {disc_fake_std:+.4f} 📉"

            if (i+1)%interval==0:
                print(f'[Epoch {epoch}/{epochs} | Iter: {i}/{len(train_loader)}] Disc Loss: {disc_loss:.4f} | Gen Loss: {gen_real_loss:.4f} | PLR: {plr_loss.item():.4f}')
                print(f" -- {status_o} Batch-{i}:  {d_real_stat_str}| {d_fake_stat_str}")
                
            losses.append((disc_loss.item(), gen_real_loss.item()))
            
            # update the training steps
            training_step_counter += 1
    
        d_loss_mean = np.mean(np.array(losses)[:,0])
        g_loss_mean = np.mean(np.array(losses)[:,1])

        all_training_losses.append((d_loss_mean, g_loss_mean))
        
        # real
        average_score_real_mean = np.mean(np.array(epoch_scores)[:,0])
        average_score_real_std = np.mean(np.array(epoch_scores)[:,0])
        # fake
        average_score_fake_mean = np.mean(np.array(epoch_scores)[:,1])
        average_score_fake_std = np.std(np.array(epoch_scores)[:,1])
        
        if quick_and_noisy_IS_FID:
            IS_score = metric.compute_IS(imgs_fake)
            FID_score = metric.compute_FID(imgs_real, imgs_fake)
        else:
            IS_score, FID_score = get_IS_FID_score(metric, generator, train_loader, dataset_name, split)

        status_avg_r = get_status(average_score_real_mean, higher_is_better=True)
        status_avg_f = get_status(average_score_fake_mean, higher_is_better=False)
        status_avg_o = get_overall_status(average_score_real_mean,
                                            average_score_fake_mean,
                                            IS_score=IS_score,
                                            min_mu=1.2)
        
        real_stats_avg_str = f"D_real_avg: {status_avg_r} {average_score_real_mean:>+.4f} ± {average_score_real_std:<+.4f} 📈"
        fake_stats_avg_str = f"D_fake_avg: {status_avg_f} {average_score_fake_mean:>+.4f} ± {average_score_fake_std:<+.4f} 📉"

        dloss_avg_str = f"DLoss(Avg): {d_loss_mean:.4f}"
        gloss_avg_str = f"GLoss(Avg): {g_loss_mean:.4f}"

        is_score_str = f"IS: {IS_score[0]:.4f} ± {IS_score[1]:.4f})"
        fid_score_str = f"FID: {FID_score:.2f}"
        
        # mintor gradient norm for mapping_network to better
        # tune hyper parameters, epsecially when it comes to fp16!
        mn_grad_norm_str = f"GradNorm: {mn_grad_norm:.4f}"
        
        plr_mean_str = f"Path Length mean: {path_length_mean.item():.4f}"
        
        summary = f"{dloss_avg_str} | {gloss_avg_str} | {is_score_str} | {fid_score_str} | {mn_grad_norm_str} | {plr_mean_str}"
        
        print(f" -- {status_o} Last Batch : {d_real_stat_str} | {d_fake_stat_str}")
        print(f" -- {status_avg_o} Epoch's Avg: {real_stats_avg_str} | {fake_stats_avg_str}")
        print(f'[Epoch {epoch}/{epochs}] {summary}')
        
        #save model weights at each epoch
        checkpoint_dir = f"{weights_save_dir}/{current_experiment_name}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        state_dicts = {"disc_state_dict":discriminator.state_dict(),
                        "gen_state_dict":generator.state_dict(),
                        "gen_ema_state_dict":ema_generator.state_dict() if use_ema_inference else None,
                        "disc_optimizer":disc_optimizer.state_dict(),
                        "gen_optimizer":gen_optimizer.state_dict(),
                        "scaler_state_dict":scaler.state_dict()
                        }
        settings = {
                    "z_size":generator.z_size,
                    "w_size":generator.w_size,
                    "mn_nlayer":generator.mn_num_layers,
                    "channels_d":discriminator.channels,
                    "channels_g":generator.channels,
                    "disc_use_upfirdn2d":discriminator.use_upfirdn2d,
                    "gen_use_upfirdn2d":generator.use_upfirdn2d,
                    "style_mixing_prob":generator.style_mixing_prob,
                    "ema_w_beta":generator.ema_w_beta,
                    "eps":generator.eps,
                    "use_fp16":use_fp16,
                    "lr_d":lr_d,
                    "lr_g":lr_g,
                    "training_step_counter":training_step_counter,
                    "kimg":kimg,
                    "ema_warmup_images_seen":ema_warmup_images_seen,
                    "epoch":epoch,
                    "epochs":epochs,
                    "batch_size":batch_size,
                    "gamma":gamma,
                    "psi":psi,
                    "path_length_interval":path_length_interval,
                    "r1_penalty_interval":r1_penalty_interval,
                    "FID":FID_score,
                    "IS":IS_score,
                    "dataset_name":dataset_name,
                    "data_augmentation":data_augmentation,
                    "normalize":normalize,
                    "split":split,
                }
        loss_dicts = {"d_loss_mean":d_loss_mean,
                        "g_loss_mean":g_loss_mean,
                        "all_training_losses":all_training_losses
                        }
        # all_settings = {**state_dicts,**settings}
        all_settings = state_dicts | settings | loss_dicts
        torch.save(all_settings, f"{checkpoint_dir}/checkpoint_step_{experiment_date}.ckpt")
        
        # generate some images mid training to evaluate our model's performance 
        with torch.no_grad():
            gen = ema_generator.eval() if use_ema_inference else generator.eval()
            
            generated_images,_ = gen(fixed_z, psi=psi)
            
            ema_marker_str = "[EMA]_" if use_ema_inference else ""
            loss_str = f"(dLoss:{d_loss_mean:.6f} | gLoss:{g_loss_mean:.6f}"
            lrs_str = f"{current_lr_d[0]:.0e},{current_lr_g[0]:.0e}"
            title_str = f"Epoch {epoch} FID:{FID_score:.2f} {loss_str} [{lrs_str}]"
            img_store_dir_path = f'{images_save_dir}/stylegan2/{dataset_name}_{experiment_date}'
            img_filename = f'{ema_marker_str}epoch_{epoch}.jpg'
            save_path= os.path.join(img_store_dir_path, img_filename)
                            
            display_images(generated_images, 
                            cols=gen_num_samples//8,
                            title=f"{ema_marker_str}{title_str}",
                            unnormalize=True,
                            save_path=save_path,
                            figsize=(16,8))
            
            # save the original images only when ema is enable, 
            # otherwise its already being saved/displayed
            if keep_raw_generations and use_ema_inference:
                generated_images,_ = generator(fixed_z, psi=psi)
                display_images(generated_images, 
                                cols=gen_num_samples//8,
                                title=title_str,
                                unnormalize=True,
                                save_path=save_path.replace(ema_marker_str,""),
                                figsize=(16,8))

            # save the settings that achieved this aswell
            settings_path = os.path.join(img_store_dir_path,'settings.yaml')
            with open(settings_path, "w") as f:
                yaml.dump(settings,f, sort_keys=False)

    print("SttyleGAN2 training is complete!")

#%% training stylegan2
print(f'Training StyleGAN2')
# gamma value can change from dataset to dataste
# for ffhq I guess they used 10 but for lsun they used 100!
gamma=10
dataset_name = 'cifar10'
split = 'train'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

use_fp16=True

# original paper uses 512
z_size = 512
w_size = 512
# with 128x128, fp32 with 64 bs -> vram 6033mb
BATCH_SIZES = 32 if use_fp16 else 32

EPOCHS = 100

path_length_interval = 4
r1_penalty_interval = 16
style_mixing_prob = 0.9
# truncation rate
psi = 0.7
#up to 128x128 
# with bs=32 [512,256,128,64,32,16] trains well 10/11m
# it takes 13/14mins per epoch. 
# for quick tests [256,128,64,32,16,8] should be good
# it takes 4/5mins per epoch.
# channels_d = [256,128,64,32,16,8]
# channels_g = [256,128,64,32,16,8]
#celeba-ffhq
# channels_d = [512,256,128,64,32,16]#,8]
# channels_g = [512,256,128,64,32,16]#,8]

#cifar10 32x32 only
channels_d = [512,256,128,64]#32,#16]#,8]
channels_g = channels_d

# whether to use upfirdn2d or normal upsample/downsample
use_upfirdn2d = True # True

#discriminator
discriminator_stylegan2 = DiscriminatorStyleGAN2(channels=channels_d,
                                                 use_upfirdn2d=use_upfirdn2d)
discriminator_stylegan2 = discriminator_stylegan2.to(device)
#generator
mn_nlayer = 8
generator_stylegan2 = GeneratorStyleGAN2(z_size, w_size, mn_nlayer,
                                         channels_g, style_mixing_prob, 
                                         use_upfirdn2d=use_upfirdn2d)
generator_stylegan2 = generator_stylegan2.to(device)

betas = [0, 0.99]
# 0.003
lr_d = 0.001 if not use_fp16 else 0.001
lr_g = 0.001 if not use_fp16 else 0.001

eps = 1e-5 if use_fp16 else 1e-8

use_ema_inference = True

disc_optimizer = torch.optim.Adam(discriminator_stylegan2.parameters(), lr=lr_d, betas=betas)
gen_optimizer = torch.optim.Adam(generator_stylegan2.parameters(), lr=lr_g, betas=betas, eps=eps)

training_loop_stylegan2(discriminator_stylegan2,
                     generator_stylegan2,
                     disc_optimizer=disc_optimizer,
                     gen_optimizer=gen_optimizer, 
                     epochs=EPOCHS,
                     batch_size=BATCH_SIZES,
                     path_length_interval=path_length_interval,
                     r1_penalty_interval=r1_penalty_interval,
                     dataset_name=dataset_name,
                     split=split,
                     data_augmentation=True,
                     gamma=gamma,
                     psi=psi,
                     use_fp16=use_fp16,
                     device=device,
                     resume=False,
                     use_ema_inference=use_ema_inference,
                     kimg=10,
                     keep_raw_generations=True,
                     quick_and_noisy_IS_FID=False,
                     )
#%% clear vram
for m in [discriminator_stylegan2,
          generator_stylegan2,
          disc_optimizer,
          gen_optimizer]:
    m=None

def get_human_readable_size(size_byte):
    unites = ["B","KiB","MiB","GiB"]
    for unit in unites:
        if size_byte <1024:
            return f"{size_byte:,.2f} {unit}"
        else:
            size_byte/=1024
    return f"{size_byte:,.2f} {unit[0]}"

total_vram_usage_bytes = 0
for obj in gc.get_objects():
    try:
        if torch.is_tensor(obj) and obj.device == torch.device("cuda:0"):
            # get size in bytes
            size_bytes = obj.nelement()*obj.element_size()
            total_vram_usage_bytes += size_bytes
            sz_str = get_human_readable_size(size_bytes)
            print(f'{obj.__class__.__name__}({tuple(obj.size())}) takes {sz_str}')
            obj=None
    except Exception as ex:
        print(f'{ex}')
        
print(f'in total {get_human_readable_size(total_vram_usage_bytes)} worth of tensor took up vram')

gc.collect()
torch.cuda.empty_cache()
#%%
# debug log:
# stylegan2_ffhq_20251127152002:
# - everything is going smoothly alhamdolelah! we are at epoch 53, with 2m/4m models
#   the configuration is [256,128,64,32,16,8]. with batchsize=64, we are using 6G of
#   vram! the training is fast with this config in 5 hours we have done 54 epochs so 
#   10 epochs per hour! image quality is still not good, but its vibrant and overall
#   wellformed for the most part. I was thinking if we can relax the mapping network
#   reduced lr a bit to see if it can make the convergence faster. the overal training
#   seems absolutely ok. but I would like to give that a try. I guess I'll do that after
#   I increased the model capacity a bit to see how it works. the results after 100 epochs
#   turned out not bad considering we're only using 2m/4m model and only 100 epochs. 
#   it did improve so provided we trained it more we'd get better results. it took 
#   around 10 hours to finish 100 epochs by the way.(it took 5/6mins per epoch)
#
# stylegan2_ffhq_20251128073036:
# - updated both discriminator and generator to work with and without upfirdn2d. we can 
#   now easily switch and experiment with the normal upsample/downsample and see itsimpatc
#   the first time im going to use no blur just upsampe and downsample. ok it went smoothly
#   at least seemingly until I test the interpolations and see how the results looks
#   we need to train a more powerful model to be able to better test its features. these two
#   models are not there yet.(by the way it took 4 mins per epoch)
#
# stylegan2_ffhq_20251128162446:
# - with the new channel config of [512,256,128,64,32,16], we have 11/10m models, with bs=32
#   vram usage of 9500MB and use_upfirdn2d = True, we started the training hopefully we get
#   much better results. each epoch takes 13/14mins! the image quality is much better but 
#   I guess we could improve more by more training or a larger network. 
# 
# stylegan2_ffhq_20251130084938:
# fixed ema update bug, updated the decay formula for ema update and now we are trainig 
# ffhq with the same config as stylegan2_ffhq_20251127152002 to see how the ema affects
# our results. I also need to fix FID calculation, its been failing for sometime now!
# not sure whats causing it. the ema really improves the image generation quality
# the ema version at 44 looks like the 100 of normal model. the end result in ema
# has less artifacts than the ordinary/main model. so its a must have during training
# however it does not substitude a stronger model obviously. while our 4m config performed
# much better this way, I still prefer the 11/10m version better!
#
# stylegan2_celeba_20251130203936:
# - fixed FID cache bug, use [512,256,128,64,32,16] bs=32,use_upfirdn2d = True:at e18 we are
#   at FID:27 and everything looks good!I guess ourlr might be a bit too much using 1e-3 next time
#   (after a few more epochs(i.e. 21) during which the loss seemed stagnant and not improving, we
#   did improve, and loss dropped from 0.9376 down to 0.90! fid went to 26.10 in e24 but still
#   fluctuating to 34 in following epochs(i.e.24/25).I ended the training at epoch 31. I guess its
#   good enough!

# stylegan2_cifar10_20251201190515:
# testing fp16:
# test cifar10 with fp16 up to 32x32  model 11/10m- it seems fp16 is broken! in 13 epochs we got noise!
# pure noise! generator loss shot up to 3 while discriminator 0.07! complete collapse right from the
# begining!! the weird thing is the fp32 takes less vram than fp16!
# 
# stylegan2_cifar10_20251201203050:
# with fp32 now.trains normally in epoch 15 ema shows very good result. fid is 48 at e16
# 
# stylegan2_cifar10_20251202064345:
# training with lower lr=0.001 ok but it seems 0.003 initially speeds things up we can dial down
# the lr later imho that would be better.
# 
# fp16 training fixed: 
# vram wise fp16 doesnt benifit us at all, as crucial parts are still being done in fp32! so fp32
# is much better
#%%
#%% load_checkpoints
def load_checkpoints(checkpoint_path, device="cuda", use_ema=False):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    dataset_name = checkpoint["dataset_name"]
    epoch = checkpoint["epoch"]
    FID_score = checkpoint["FID"]

    z_size = checkpoint["z_size"]
    w_size = checkpoint["w_size"]
    mn_layers = checkpoint["mn_nlayer"]
    channels = checkpoint["channels_g"]
    style_mix = checkpoint["style_mixing_prob"]
    eps = checkpoint["eps"]
    use_upfirdn2d = checkpoint.get("use_upfirdn2d",True)
    ema_wb = checkpoint["ema_w_beta"]
    ema_weights = checkpoint["gen_ema_state_dict"]
    res = 2<<(len(channels))
    
    generator_st2 = GeneratorStyleGAN2(z_size, w_size,mn_layers,channels, style_mix, ema_wb, use_upfirdn2d,eps)
    
    if use_ema and ema_weights is not None:
        generator_st2.load_state_dict(ema_weights)
        print('ema weights loaded!')
    else:
        generator_st2.load_state_dict(checkpoint["gen_state_dict"])
        print('normal weights loaded!')
    
    generator_st2 = generator_st2.eval()
    generator_st2 = generator_st2.to(device)

    print(f'Loading checkpoint for {dataset_name} [{res}x{res}] @ Epoch {epoch} FID={FID_score}')
    return generator_st2, res, dataset_name

device = 'cuda'
num_samples=6
use_ema = True
# checkpoint_path = './weights/gan/stylegan2_ffhq_20251127152002/checkpoint_step_20251127152002.ckpt'
# checkpoint_path = './weights/gan/stylegan2_ffhq_20251128073036/checkpoint_step_20251128073036.ckpt'
# 11/10m 
# checkpoint_path = './weights/gan/stylegan2_ffhq_20251128162446/checkpoint_step_20251128162446_e53.ckpt'
# checkpoint_path = './weights/gan/stylegan2_ffhq_20251128162446/checkpoint_step_20251128162446_e90.ckpt'
# checkpoint_path = './weights/gan/stylegan2_ffhq_20251128162446/checkpoint_step_20251128162446.ckpt'
# ffhq with ema -small model (4m)
# checkpoint_path = './weights/gan/stylegan2_ffhq_20251130084938/checkpoint_step_20251130084938.ckpt'
# celeba with ema - 11m model
checkpoint_path = './weights/gan/stylegan2_celeba_20251130203936/checkpoint_step_20251130203936.ckpt'
generator_stylegan2, dim, dataset_name = load_checkpoints(checkpoint_path, device='cuda', use_ema=use_ema)

z = torch.randn((num_samples, generator_stylegan2.z_size), device=device)
res=f"{"EMA" if use_ema else ""} {dim}x{dim}"
with torch.no_grad():
    imgs,ws = generator_stylegan2(z, psi=0.7)
    display_images(imgs, 
                   cols=2, 
                   title=f'StyleGAN2 [{res}]',
                   unnormalize=True, 
                   figsize=(16,8))
#%% create_interpolation_animation
@torch.no_grad()
def create_interpolation_animation(imgs_tensor, filename='vis', interval=300, repeat=True, repeat_delay=1000):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    def animate(i):
        img_tensor = imgs_tensor[i].squeeze(0)
        frame = utils.make_grid(img_tensor, normalize=True).cpu().detach().numpy().transpose(1,2,0)
        ax.clear()
        ax.imshow(frame)

    anim = animation.FuncAnimation(fig, 
                                   animate,
                                   frames=imgs_tensor.size(0),
                                   interval=interval, 
                                   repeat=repeat, 
                                   repeat_delay=repeat_delay)
    # save the git using pillow
    anim.save(f'{filename}.gif', writer="pillow")
    plt.show()


# sidenote:
# for interpolation we can either use lerp or slerp
# the most straight one is linear interpolation (i.e. lerp)
# which means we move in a straight line in our latent space
# from vector v1 to vector v2 each time taking a step from v1 toward v2.
# basically v_out = (1-t)*v1 + t*v2 (t is a value between 0 and 1)
# so when t=0 we are at v1 and when t=1 we are at v2 and all the other
# values in between therefore give us points between these two ends. 
# so far so good. the issue however is that, it sometimes results
# in less natural looking/distorted transitions because the straight path
# might(and usually do) go through areas in the latent space that dont 
# corrospond to well formed or realistic faces/objects(i.e. areas of malformed/underdeveopled/faulty samples!)
# and therefore cause artifacts or a less smooth change in appearance.
# for this reason, slerp is used instead. slerp is simply spherical linear interpolation
# and it means we move between the two vectors along the arc of a great circle, on a hypersphere.
# this has two important implications, one because we use the same angle
# between the two vectors, we maintain a constant rate of change, i.e. we'll 
# have a more uniform and natural looking transformation in our interpolation.
# this is not the case for lerp as it can have a non-uniform rate of change when
# its projected onto the hyperspher.
# second and more importantly, the distribution of meaningful latent vectors is usually
# more like a hypersphere, so staying on this surface will give us more chance of remaining
# within areas/regions of latent space that produce high-quality/realstic faces/objects.
# 
def slerp(w1, w2, t, threshold=0.9995):
    # normalize the input vectors so we get work with directions
    w1norm = w1 / torch.linalg.norm(w1)
    w2norm = w2 / torch.linalg.norm(w2)

    dot = torch.sum(w1norm * w2norm)

    # if the two vectors are close enough then we
    # can probably use lerp without any issues
    # the two vectors are idential if their dot product
    # is 1!
    if torch.abs(dot)>threshold:
        return torch.lerp(w1,w2,t)

    # caulate the angle between the two vectors
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)

    # moving from w1 to w2 along the arc between the two
    a = torch.sin((1.0 - t) * theta) / sin_theta
    b = torch.sin(t * theta) / sin_theta
    return a * w1 + b * w2

# now lets write the actual experiments on w
# lets do a latent space exploration like before 
@torch.no_grad()
def interpolate_w(generator:GeneratorStyleGAN2, z1, z2, psi=None, constant_noise=False,
                  alphas=None, use_slerp=True, interp_steps=60, device='cuda'):
    generator = generator.to(device)
    generator.eval()
        
    z1,z2 = tuple(z.to(device) for z in (z1,z2))
    w1 = generator.mapping_network(z1)
    w2 = generator.mapping_network(z2)
    
    # interpolate
    if alphas is None:
        alphas = torch.linspace(0, 1, interp_steps).to(device)
    else:
        alphas = alphas.to(device)
    
    noise = None
    if constant_noise:
        noise = torch.zeros((z1.size(0),1,1,1),device=device)    
        
    imgs = []
    for a in alphas:
        if use_slerp:
            # slerp does a much smoother interpolation
            # 
            interpol = slerp(w1,w2,a)
        else:
            interpol = torch.lerp(w1,w2,a)
        interpol = generator.apply_truncation(interpol,psi)
        img = generator.forward_from_w(interpol,noise).cpu()
        imgs.append(img)
    output_imgs = torch.cat(imgs, dim=0)
    return output_imgs

def run_interpolation_test(generator, constant_noise, psi_rates=None,
                           alphas=None, use_slerp=True, num_samples=36, make_gifs=False,
                           gif_dir='./results/gan/stylegan2/gifs',interval=100,):
    
    if psi_rates is None:
        psi_rates = [0, 0.3, 0.7, 1]
    
    mode = "slerp" if use_slerp else "lerp"
    for rate in psi_rates:
        interpolated_images = interpolate_w(generator,
                                            z1, 
                                            z2,
                                            psi=rate,
                                            constant_noise=constant_noise,
                                            alphas=alphas,
                                            use_slerp=use_slerp,
                                            interp_steps=num_samples)
            
        if make_gifs:
            fname =f"interpolation_{dataset_name}_psi_{rate}_const_noise_{constant_noise}_using_{mode}"
            # make sure directory exists, if not create it
            os.makedirs(gif_dir, exist_ok=True)
            fpath = os.path.join(gif_dir, fname)
            create_interpolation_animation(interpolated_images, fpath, interval=interval)
        else:
            display_images(interpolated_images, title=f'{dataset_name}_psi={rate} | constant_noise:{constant_noise} using {mode}', cols=cols, unnormalize=True)

# now lets try 
num_samples = 36
cols = int(num_samples**0.5)
seed=66
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
fixed_randg = torch.Generator(device=device).manual_seed(seed)

z1 = torch.randn(size=(1, generator_stylegan2.z_size), device=device, generator=fixed_randg)
z2 = torch.randn(size=(1, generator_stylegan2.z_size), device=device, generator=fixed_randg)

run_interpolation_test(generator_stylegan2,
                       #psi_rates=[0,0.5,1.2,3,5],
                       constant_noise=True,
                       num_samples=num_samples,)

#%% run_interpolation_test
# now lets do this with noise injection i.e. constant_noise=False
run_interpolation_test(generator_stylegan2, 
                       #psi_rates=[0,0.5,1.2,3,5],
                       constant_noise=False,
                       num_samples=num_samples,)
#%% making some gifs
# now lets also make some gifs as well!
alphas = torch.linspace(-2,2,60)#-1,1
for noise_status in [True]:
    run_interpolation_test(generator_stylegan2,
                           psi_rates=[0.7],
                           constant_noise=noise_status,
                           alphas=alphas,
                           use_slerp=True,
                           num_samples=36,
                           make_gifs=True,
                           gif_dir='./results/gan/stylegan2/gifs',
                           interval=100,)
#%%
@torch.no_grad()
def style_mix(generator:GeneratorStyleGAN2, z_source, z_style, layer_indx_for_crossover,
              psi_src=None, psi_sty=None, constant_noise=True):
    
    num_layers = 2*len(generator.channels)
    assert 0<layer_indx_for_crossover<num_layers, f'layer index{layer_indx_for_crossover} must be < {num_layers}'
    
    device = next(generator.parameters()).device
    
    z_source = z_source.to(device)
    z_style = z_style.to(device)   
    
    w_source = generator.mapping_network(z_source)
    w_style = generator.mapping_network(z_style)
    
    # apply truncation
    w_source = generator.apply_truncation(w_source, psi_src)
    w_style = generator.apply_truncation(w_style, psi_sty)
    
    # expand to match shape
    w_source = w_source.unsqueeze(1).repeat(1,num_layers,1)
    w_style = w_style.unsqueeze(1).repeat(1,num_layers,1)
    
    # we are going to need both source and style ws for later
    # reference so lets use it to create our result w_mixed
    w_mixed = w_source.clone()
    w_mixed[:,layer_indx_for_crossover:,:] = w_style[:,layer_indx_for_crossover:,:]
    
    noise = None
    if constant_noise:
        noise = torch.zeros((w_source.size(0),1,1,1),device=device)
    
    img_source = generator.forward_from_w(w_source, noise).cpu()
    img_style = generator.forward_from_w(w_style, noise).cpu()
    img_mixed = generator.forward_from_w(w_mixed, noise).cpu()
    
    return img_source, img_style, img_mixed

z_source = torch.randn(1, generator_stylegan2.z_size, device=device, generator=fixed_randg)
z_style = torch.randn(1, generator_stylegan2.z_size, device=device, generator=fixed_randg)

# different layers affect different details experiment
# with all and see the result. e.g. starting from 4 
# we can see more drastic style transfers. if we go
# lower, like 1, 2 , we are basically seeing second z!
# the majority of values belong to second z, but starting
# from 3,4, we can see the first image is structually there
# and styles start to transfer.(skin tone, colors, are obvious) 
layer_crossover = 5
img_src, img_style, img_mix = style_mix(generator_stylegan2,
                                        z_source, 
                                        z_style, 
                                        layer_crossover, 
                                        psi_src=0.7,
                                        psi_sty=0.8,
                                        constant_noise=True)
imgs = torch.cat([img_src,img_style,img_mix])
display_images(imgs, title='images source|style|mix', unnormalize=True,figsize=(8,6))
# as we can see, the structure of the source image stays the same, but the 
# texture/style of the style image is transfered. lets see how each layer affects
# the result 
#%% change styles
def change_styles(z_source, z_style, psi_src=0.8, psi_sty=0.8):
    num_layers = 2*len(generator_stylegan2.channels)
    imgs_all = []
    for layer in range(1,num_layers):
        img_src, img_style, img_mix = style_mix(generator_stylegan2,
                                                z_source, 
                                                z_style, 
                                                layer, 
                                                psi_src=psi_src,
                                                psi_sty=psi_sty,
                                                constant_noise=True)
        
        imgs = torch.cat([img_src,img_style,img_mix],dim=0)
        # print(f'{imgs.shape=}')
        
        row = utils.make_grid(imgs, nrow=3, normalize=True)
        display_images(row, title=f'{layer} source|style|mix',
                       unnormalize=False,  figsize=(8,6))
        # print(f'{row.shape=}')
        
        imgs_all.append(row.unsqueeze(0))
    
    ims = torch.cat(imgs_all, dim=0)
    # print(f'{ims.shape=}')
    display_images(ims,
                   title='Source | Style | Mix',
                   unnormalize=False,
                   cols=1,
                   figsize=(16,16))

change_styles(z_source,z_style,psi_src=0.7, psi_sty=0.7)
#%%
# grab the modulation weights for closed form factorization
# we are trying to extract meaninful directions fromw eights
# so we can use them to generate images along those directions
# creating images with certain attributes.
# for best performance the ema version of the generator weights are 
# used
weights = []
for k,v in generator_stylegan2.state_dict().items():
    # print(k)
    # grab only fc_style weights in blocks that deal with style
    if "fc_style.weight" in k and "blocks" in k:
        # print(f'{k}')
        weights.append(v)

Ws = torch.cat(weights)
# now using svd we get eigen vectors(unique directions)
eigen_vecs = torch.svd(Ws).V.to("cpu")
#sidenote:reminder
# svd or singular value decompision basically breaks down our input matrix into its fundamental/principal building blocks!
# we want the V matrix, because its columns (or rows of Vᵀ) are the right singular vectors
# i.e. these vectors form an orthonormal basis for the input space! that is in other words,
# each of these vectors represent the principal directions of variations in our input! 
# in our case, they are the directions in w latent space that cause the most significant and
# also independent changes in the final style of the image.
# the first singular vector is the direction of maximum change, the second one is the next most
# significant direction orthonormal to the first one and so on and so forth!
# 
# sidenote2:
# this is obvious but im saying it anyway what we have here are singular vectors, 
# (which are eigenvectors of WsᵀWs) but since its pretty common to call this eigenvector
# I do this aswell. the important matter is we know the distinction!
#%%
# and now we can apply these new directions to our latent vectors ws and get the result
# lets test this
@torch.no_grad()
def interpolate_w_with_direction(generator:GeneratorStyleGAN2, z1, direction,
                                 psi=None, constant_noise=False,
                                 alphas=None, interp_steps=60, device='cuda',
                                 make_gifs=False, interval=100,
                                 gifs_dir='./results/gan/stylegan2/gifs/',
                                 msg=""):
    generator = generator.to(device)
    generator.eval()
    
    direction = direction.to(device) 
    w = generator.mapping_network(z1)
    # we apply the truncation to base w, initially I did it inside the loop
    # this was wrong clearly as truncation moves the w toward the average
    # so if we applied a direction on our w and then do truncation it would
    # remove the said direction! having it here is as if we are making the
    # canvas ready (start from average w/face) and then apply the direcion
    w = generator.apply_truncation(w, psi)
            
    # interpolate
    if alphas is None:
        # initially used 0-1, but -3,3 gives much more pronounced effects
        alphas = torch.linspace(-3, 3, interp_steps).to(device)
    else:
        alphas = alphas.to(device)
    
    noise = None
    if constant_noise:
        noise = torch.zeros((z1.size(0),1,1,1),device=device)    
        
    imgs = []
    for a in alphas:
        w_new = w+a*direction
        img = generator.forward_from_w(w_new, noise).cpu()
        imgs.append(img)
    
    output_imgs = torch.cat(imgs, dim=0)
    
    if make_gifs:
        fname =f"{msg}_interpolation_{dataset_name}_psi_{psi}_const_noise_{constant_noise}"
        # make sure directory exists, if not create it
        os.makedirs(gifs_dir, exist_ok=True)
        fpath = os.path.join(gifs_dir, fname)
        create_interpolation_animation(output_imgs, fpath, interval=interval)
    else:
        display_images(output_imgs, 
                       title=f'{msg}_{dataset_name}_psi_{psi}| constant_noise:{constant_noise}', 
                       cols=math.ceil(math.sqrt(interp_steps)), 
                       unnormalize=True)
    return output_imgs

seed=66
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
fixed_randg = torch.Generator(device=device).manual_seed(seed)
z1 = torch.randn(size=(1, generator_stylegan2.z_size), device=device, generator=fixed_randg)

for i in range(eigen_vecs.size(0)):
    imgs_with_directions = interpolate_w_with_direction(generator_stylegan2, 
                                                        z1, 
                                                        eigen_vecs[i],
                                                        psi=0.7,
                                                        constant_noise=True,
                                                        alphas=None,
                                                        interp_steps=30,
                                                        device='cuda',
                                                        make_gifs=True,
                                                        msg=f"dir_{i}")
    # display_images(imgs_with_directions, 
    #            title=f'direction {i} applied',
    #            cols=5, 
    #            unnormalize=True)
    
#%%
#%%
import cv2
# define our simple gui 
def onchange(x):
    pass

@torch.no_grad()
def choose_meanstd(generator, num_samples,random_gen, ncols=8,**kwargs):
    generator.eval()
    
    cv2.namedWindow('stylegan2 latent experiment ')
    
    # create trackbars for std and mean 
    # opencv trackbar only supports ints, 
    # so we specify our desired range as int
    # and then in code divide them to get fractions
    # alpha: -3 - 3
    # mean: 0.0 - 1
    # to be able to sample new values we use this
    cv2.createTrackbar('low', 'low_finder', 1, 20, onchange)
    cv2.createTrackbar('high', 'high_finder', 1, 20, onchange)
    cv2.createTrackbar('alpha', 'alpha_selector', 500, 1000, onchange)
    z = torch.randn(size=(num_samples, generator.z_size), generator=random_gen)
    old_std, old_mean, old_minus,old_resample=None,None,None,None
    imgs=None
        
    while True:
        std = cv2.getTrackbarPos('std','std_mu_finder',)
        mean = cv2.getTrackbarPos('mean','std_mu_finder')
        resample = cv2.getTrackbarPos('resample','std_mu_finder')
        
        frac_std = std/1000
        frac_mean = mean/1000
        
        if old_resample != resample:
            z = torch.randn(size=(num_samples, generator.z_size),generator=random_gen)
            print(f'New z sampled!')
            
        # only generate when values change so 
        # we dont waste too much cpu and hug the system!
        if old_std!=std or old_mean!=mean or old_resample!=resample:
            new_z = frac_std * z + frac_mean
            # update from future: added kwargs so
            # this can be used for future gans as
            # well which accept more arguments
            imgs = generator(new_z, **kwargqs)
                       
            img = utils.make_grid(imgs, nrow=ncols, normalize=True, value_range=(-1, 1))
            img = (img.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = cv2.resize(img, dsize=None, fx=2.0, fy=2.0)
            
            old_std, old_mean, old_resample = std, mean, resample
            print(f'Generated using std:{frac_std:5f} mu:{frac_mean:.5f}')
            
        if img is not None:
            cv2.imshow('std_mu_finder', img)
            
        # break loop when 'q' is pressed
        # also note we dont need to check this every 1 ms!
        # waiting every 30ms/90ms suffices, I chose 60ms
        if cv2.waitKey(60) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    return z, frac_std, frac_mean

seed=1
np.random.seed(seed)
random_gen = torch.manual_seed(seed)
steps = 8
num_samples=5
generatorcnn.to('cpu')
#%%
# a detour to something fun CycleGAN