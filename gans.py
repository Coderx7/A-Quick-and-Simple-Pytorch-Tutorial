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
import os
from datetime import datetime

import torch 
import numpy as np 

import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils.data import DataLoader

from torchvision import datasets, transforms
from torchvision import utils

import matplotlib.pyplot as plt
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

discriminator = Discriminator(28*28, 32)
generator = Generator(100, 32, 28*28)

dis_output = discriminator(imgs)
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
def display_images(imgs, rows=8, title='',unnormalize=False, save_path=None):
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
    images = utils.make_grid(imgs,nrow=rows).cpu().numpy().transpose(1,2,0) # c,h,w -> h,w,c

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
discriminator = Discriminator(disc_input_size, disc_hidden_size, act=nn.LeakyReLU(0.2))
generator = Generator(gen_input_size, gen_hidden_size, gen_output_size, act=nn.LeakyReLU(0.2) )
# 
discriminator = discriminator.to(device)
generator = generator.to(device)

# optimizers, note we are using different lr's here!
disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.02)
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
    discriminator.train()
    generator.train()
    # we dont need image's real labels because we are not trying to classify 
    # mnist didgits! we want to create images and we will create our own labels
    # as we explained earlier
    for i, (real_images,_) in enumerate(train_loader):
        
        # rescale input images from [0,1) to [-1, 1)
        real_images = (real_images*2 - 1).to(device)
        # discriminator needs to classify the real image as real
        # and fake images as fake. so we need to have both of them
        real_outputs = discriminator(real_images)
        real_images_loss = real_loss(real_outputs.cpu(), is_smoothed=True)
        
        # now we generate an image using generator and classify it as fake
        latent_vectors = torch.distributions.Uniform(-1,1).sample((real_images.size(0), gen_input_size)).to(device)
        fake_images = generator(latent_vectors)
        # the discreminator must classify all generated images as fake
        fake_outputs = discriminator(fake_images)
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
        generated_outputs = discriminator(fake_images)
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
                   rows=num_samples//4,
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

#%%
# move after DCGAN impl
# Before we continue if you remember we said getting a GAN to work is 
# an involved effort and requires a few tips and tricks at the very least 
# to work properly. when DCGAN came out, it provided many of such tips and 
# the authors posted a list of them in their github repo. 
# while some of these tips and tricks are still valid, some have gone obsolote
# in newer architectures, and some have also evolved. having said that, for now
# lets review these tips we expand on them later. 
# sidenote:
# this is from 2016 by the way
# https://github.com/soumith/ganhacks#16-discrete-variables-in-conditional-gans
# they directly affect how GANs are trained!
# 
# quick summary of the points:
# dont use relu in discriminator! 
# use Guassian/normal distribution instead of uniform in weight initialization!
# in batch use different batches for real and fake separately (especially if you use batchnorm!)
# use tanh for generator's last layer
# use label smoothing
# use adam for generator, and you can use sgd with discriminator!
# 
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
# a larger batchsize provides more stablity
batch_size = 128
num_workers = 8
# check what happens if we use augmentations here?! aka us transforms.Compose
transform = transforms.Compose([
    # transforms.RandomHorizontalFlip(),
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor()
])

train_dataset = datasets.SVHN('./data/SVHN', split='extra', transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

#visualize 
(imgs, labels) = next(iter(train_loader))
display_images(imgs, title='svhn samples',rows=16)

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
                "epoch":epoch,
                "loss":g_loss_mean
                }, f"./weights/dcgan_generatorcnn_{experiment_date}.pt")
    
    # generate some images mid training to evaluate our model's performance 
    generatorcnn.eval()
    # reshape images back to 32x32x3
    generated_images = generatorcnn(fixed_z).view(-1,*imgs_real.shape[1:])
    display_images(generated_images, 
                   rows=gen_num_samples//8,
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
states = torch.load('./weights/dcgan_generatorcnn_20250811195800.pt')
generatorcnn.load_state_dict(states["state_dict"])

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
# our first training log where we had 
# mode collapse and high generators loss
# Epoch/Epochs: 0/50 | Iter: 0/8299 | Discriminator Loss: 1.3818 | Generator Loss: 0.7014
# Epoch/Epochs: 0/50 | Iter: 5000/8299 | Discriminator Loss: 0.3742 | Generator Loss: 4.8500
# Epoch/Epochs: 0/50 | Discriminator Loss : 0.7194 | Generator loss: 3.6977 
# Epoch/Epochs: 1/50 | Iter: 0/8299 | Discriminator Loss: 0.4485 | Generator Loss: 2.7390
# Epoch/Epochs: 1/50 | Iter: 5000/8299 | Discriminator Loss: 0.3596 | Generator Loss: 10.5628
# Epoch/Epochs: 1/50 | Discriminator Loss : 0.5544 | Generator loss: 4.9144 
# Epoch/Epochs: 2/50 | Iter: 0/8299 | Discriminator Loss: 0.3457 | Generator Loss: 7.2209
# Epoch/Epochs: 2/50 | Iter: 5000/8299 | Discriminator Loss: 0.4086 | Generator Loss: 4.7951
# Epoch/Epochs: 2/50 | Discriminator Loss : 0.4964 | Generator loss: 5.0215 
# Epoch/Epochs: 3/50 | Iter: 0/8299 | Discriminator Loss: 0.3463 | Generator Loss: 7.7827
# Epoch/Epochs: 3/50 | Iter: 5000/8299 | Discriminator Loss: 0.3952 | Generator Loss: 7.4056
# Epoch/Epochs: 3/50 | Discriminator Loss : 0.4632 | Generator loss: 5.6637 
# Epoch/Epochs: 4/50 | Iter: 0/8299 | Discriminator Loss: 0.3533 | Generator Loss: 6.6564
# Epoch/Epochs: 4/50 | Iter: 5000/8299 | Discriminator Loss: 0.3868 | Generator Loss: 3.9747
# Epoch/Epochs: 4/50 | Discriminator Loss : 0.4462 | Generator loss: 5.6578 
# Epoch/Epochs: 5/50 | Iter: 0/8299 | Discriminator Loss: 0.4926 | Generator Loss: 3.4306
# Epoch/Epochs: 5/50 | Iter: 5000/8299 | Discriminator Loss: 0.3742 | Generator Loss: 6.7072
# Epoch/Epochs: 5/50 | Discriminator Loss : 0.4460 | Generator loss: 5.4157 
# Epoch/Epochs: 6/50 | Iter: 0/8299 | Discriminator Loss: 0.5397 | Generator Loss: 3.2851
# Epoch/Epochs: 6/50 | Iter: 5000/8299 | Discriminator Loss: 0.3896 | Generator Loss: 3.3385
# Epoch/Epochs: 6/50 | Discriminator Loss : 0.4458 | Generator loss: 5.1422 
# Epoch/Epochs: 7/50 | Iter: 0/8299 | Discriminator Loss: 0.4388 | Generator Loss: 3.9077
# Epoch/Epochs: 7/50 | Iter: 5000/8299 | Discriminator Loss: 0.4192 | Generator Loss: 3.1753
# Epoch/Epochs: 7/50 | Discriminator Loss : 0.4408 | Generator loss: 5.0697 
# Epoch/Epochs: 8/50 | Iter: 0/8299 | Discriminator Loss: 0.3691 | Generator Loss: 5.5397
# Epoch/Epochs: 8/50 | Iter: 5000/8299 | Discriminator Loss: 0.4798 | Generator Loss: 3.4502
# Epoch/Epochs: 8/50 | Discriminator Loss : 0.4408 | Generator loss: 5.0244 
# Epoch/Epochs: 9/50 | Iter: 0/8299 | Discriminator Loss: 0.3967 | Generator Loss: 3.7294
# Epoch/Epochs: 9/50 | Iter: 5000/8299 | Discriminator Loss: 0.3992 | Generator Loss: 3.8892
# Epoch/Epochs: 9/50 | Discriminator Loss : 0.4386 | Generator loss: 4.9025 
# Epoch/Epochs: 10/50 | Iter: 0/8299 | Discriminator Loss: 0.4434 | Generator Loss: 3.0252
# Epoch/Epochs: 10/50 | Iter: 5000/8299 | Discriminator Loss: 0.4099 | Generator Loss: 3.3124
# Epoch/Epochs: 10/50 | Discriminator Loss : 0.4366 | Generator loss: 4.7645 
# Epoch/Epochs: 11/50 | Iter: 0/8299 | Discriminator Loss: 0.4566 | Generator Loss: 3.3419
# Epoch/Epochs: 11/50 | Iter: 5000/8299 | Discriminator Loss: 0.4031 | Generator Loss: 3.6082
# Epoch/Epochs: 11/50 | Discriminator Loss : 0.4346 | Generator loss: 4.6863 
# Epoch/Epochs: 12/50 | Iter: 0/8299 | Discriminator Loss: 0.3736 | Generator Loss: 4.5408
# Epoch/Epochs: 12/50 | Iter: 5000/8299 | Discriminator Loss: 0.3928 | Generator Loss: 4.0038
# Epoch/Epochs: 12/50 | Discriminator Loss : 0.4314 | Generator loss: 4.6957 
# Epoch/Epochs: 13/50 | Iter: 0/8299 | Discriminator Loss: 0.3939 | Generator Loss: 4.5121
# Epoch/Epochs: 13/50 | Iter: 5000/8299 | Discriminator Loss: 0.3907 | Generator Loss: 3.9596
# Epoch/Epochs: 13/50 | Discriminator Loss : 0.4278 | Generator loss: 4.6483 
# Epoch/Epochs: 14/50 | Iter: 0/8299 | Discriminator Loss: 0.4002 | Generator Loss: 3.7172
# Epoch/Epochs: 14/50 | Iter: 5000/8299 | Discriminator Loss: 0.3666 | Generator Loss: 5.4941
# Epoch/Epochs: 14/50 | Discriminator Loss : 0.4245 | Generator loss: 4.6557 
# Epoch/Epochs: 15/50 | Iter: 0/8299 | Discriminator Loss: 0.4076 | Generator Loss: 3.7309
# Epoch/Epochs: 15/50 | Iter: 5000/8299 | Discriminator Loss: 0.4487 | Generator Loss: 3.0717
# Epoch/Epochs: 15/50 | Discriminator Loss : 0.4235 | Generator loss: 4.6138 
# Epoch/Epochs: 16/50 | Iter: 0/8299 | Discriminator Loss: 0.4148 | Generator Loss: 3.3697
# Epoch/Epochs: 16/50 | Iter: 5000/8299 | Discriminator Loss: 0.3830 | Generator Loss: 3.7170
# Epoch/Epochs: 16/50 | Discriminator Loss : 0.4215 | Generator loss: 4.5914 
# Epoch/Epochs: 17/50 | Iter: 0/8299 | Discriminator Loss: 0.3547 | Generator Loss: 4.8107
# Epoch/Epochs: 17/50 | Iter: 5000/8299 | Discriminator Loss: 0.3798 | Generator Loss: 3.5328
# Epoch/Epochs: 17/50 | Discriminator Loss : 0.4190 | Generator loss: 4.5633 
# Epoch/Epochs: 18/50 | Iter: 0/8299 | Discriminator Loss: 0.3932 | Generator Loss: 3.5611
# Epoch/Epochs: 18/50 | Iter: 5000/8299 | Discriminator Loss: 0.3501 | Generator Loss: 5.9975
# Epoch/Epochs: 18/50 | Discriminator Loss : 0.4161 | Generator loss: 4.5836 
# Epoch/Epochs: 19/50 | Iter: 0/8299 | Discriminator Loss: 0.3643 | Generator Loss: 6.1004
# Epoch/Epochs: 19/50 | Iter: 5000/8299 | Discriminator Loss: 0.3477 | Generator Loss: 6.0416
# Epoch/Epochs: 19/50 | Discriminator Loss : 0.4139 | Generator loss: 4.6642 
# Epoch/Epochs: 20/50 | Iter: 0/8299 | Discriminator Loss: 0.3797 | Generator Loss: 5.4030
# Epoch/Epochs: 20/50 | Iter: 5000/8299 | Discriminator Loss: 0.3783 | Generator Loss: 4.4055
# Epoch/Epochs: 20/50 | Discriminator Loss : 0.4117 | Generator loss: 4.7086 
# Epoch/Epochs: 21/50 | Iter: 0/8299 | Discriminator Loss: 0.3521 | Generator Loss: 6.0425
# Epoch/Epochs: 21/50 | Iter: 5000/8299 | Discriminator Loss: 0.3771 | Generator Loss: 5.4637
# Epoch/Epochs: 21/50 | Discriminator Loss : 0.4096 | Generator loss: 4.7307 
# Epoch/Epochs: 22/50 | Iter: 0/8299 | Discriminator Loss: 0.3609 | Generator Loss: 3.6636
# Epoch/Epochs: 22/50 | Iter: 5000/8299 | Discriminator Loss: 0.3611 | Generator Loss: 3.8999
# Epoch/Epochs: 22/50 | Discriminator Loss : 0.4074 | Generator loss: 4.7055 
# Epoch/Epochs: 23/50 | Iter: 0/8299 | Discriminator Loss: 0.3523 | Generator Loss: 5.6232
# Epoch/Epochs: 23/50 | Iter: 5000/8299 | Discriminator Loss: 0.3531 | Generator Loss: 5.1195
# Epoch/Epochs: 23/50 | Discriminator Loss : 0.4051 | Generator loss: 4.7278 
# Epoch/Epochs: 24/50 | Iter: 0/8299 | Discriminator Loss: 0.3555 | Generator Loss: 4.7999
# Epoch/Epochs: 24/50 | Iter: 5000/8299 | Discriminator Loss: 0.3376 | Generator Loss: 5.1868
# Epoch/Epochs: 24/50 | Discriminator Loss : 0.4038 | Generator loss: 4.7352 
# Epoch/Epochs: 25/50 | Iter: 0/8299 | Discriminator Loss: 0.3545 | Generator Loss: 5.9432
# Epoch/Epochs: 25/50 | Iter: 5000/8299 | Discriminator Loss: 0.4008 | Generator Loss: 4.7830
# Epoch/Epochs: 25/50 | Discriminator Loss : 0.4025 | Generator loss: 4.7671 
# Epoch/Epochs: 26/50 | Iter: 0/8299 | Discriminator Loss: 0.3459 | Generator Loss: 5.1820
# Epoch/Epochs: 26/50 | Iter: 5000/8299 | Discriminator Loss: 0.3469 | Generator Loss: 5.1300
# Epoch/Epochs: 26/50 | Discriminator Loss : 0.4004 | Generator loss: 4.7628 
# Epoch/Epochs: 27/50 | Iter: 0/8299 | Discriminator Loss: 0.3515 | Generator Loss: 4.4014
# Epoch/Epochs: 27/50 | Iter: 5000/8299 | Discriminator Loss: 0.3486 | Generator Loss: 4.8296
# Epoch/Epochs: 27/50 | Discriminator Loss : 0.3988 | Generator loss: 4.7638 
# Epoch/Epochs: 28/50 | Iter: 0/8299 | Discriminator Loss: 0.3418 | Generator Loss: 5.4003
# Epoch/Epochs: 28/50 | Iter: 5000/8299 | Discriminator Loss: 0.3619 | Generator Loss: 7.6773
# Epoch/Epochs: 28/50 | Discriminator Loss : 0.3970 | Generator loss: 4.8114 
# Epoch/Epochs: 29/50 | Iter: 0/8299 | Discriminator Loss: 0.3570 | Generator Loss: 5.5610
# Epoch/Epochs: 29/50 | Iter: 5000/8299 | Discriminator Loss: 0.3540 | Generator Loss: 5.2090
# Epoch/Epochs: 29/50 | Discriminator Loss : 0.3956 | Generator loss: 4.8174 
# Epoch/Epochs: 30/50 | Iter: 0/8299 | Discriminator Loss: 0.3555 | Generator Loss: 5.5297
# Epoch/Epochs: 30/50 | Iter: 5000/8299 | Discriminator Loss: 0.3408 | Generator Loss: 5.3928
# Epoch/Epochs: 30/50 | Discriminator Loss : 0.3941 | Generator loss: 4.8391 
# Epoch/Epochs: 31/50 | Iter: 0/8299 | Discriminator Loss: 0.3472 | Generator Loss: 4.7879
# Epoch/Epochs: 31/50 | Iter: 5000/8299 | Discriminator Loss: 0.3685 | Generator Loss: 6.2318
# Epoch/Epochs: 31/50 | Discriminator Loss : 0.3928 | Generator loss: 4.8645 
# Epoch/Epochs: 32/50 | Iter: 0/8299 | Discriminator Loss: 0.3520 | Generator Loss: 5.4744
# Epoch/Epochs: 32/50 | Iter: 5000/8299 | Discriminator Loss: 0.3889 | Generator Loss: 3.7217
# Epoch/Epochs: 32/50 | Discriminator Loss : 0.3920 | Generator loss: 4.8626 
# Epoch/Epochs: 33/50 | Iter: 0/8299 | Discriminator Loss: 0.3474 | Generator Loss: 4.8153
# Epoch/Epochs: 33/50 | Iter: 5000/8299 | Discriminator Loss: 0.3539 | Generator Loss: 5.7329
# Epoch/Epochs: 33/50 | Discriminator Loss : 0.3907 | Generator loss: 4.8721 
# Epoch/Epochs: 34/50 | Iter: 0/8299 | Discriminator Loss: 0.3354 | Generator Loss: 5.6520
# Epoch/Epochs: 34/50 | Iter: 5000/8299 | Discriminator Loss: 0.3589 | Generator Loss: 6.0369
# Epoch/Epochs: 34/50 | Discriminator Loss : 0.3896 | Generator loss: 4.9051 
# Epoch/Epochs: 35/50 | Iter: 0/8299 | Discriminator Loss: 0.3423 | Generator Loss: 5.5350
# Epoch/Epochs: 35/50 | Iter: 5000/8299 | Discriminator Loss: 0.3553 | Generator Loss: 3.7807
# Epoch/Epochs: 35/50 | Discriminator Loss : 0.3887 | Generator loss: 4.9045 
# Epoch/Epochs: 36/50 | Iter: 0/8299 | Discriminator Loss: 0.3349 | Generator Loss: 5.3008
# Epoch/Epochs: 36/50 | Iter: 5000/8299 | Discriminator Loss: 0.3452 | Generator Loss: 4.7669
# Epoch/Epochs: 36/50 | Discriminator Loss : 0.3878 | Generator loss: 4.9115 
# Epoch/Epochs: 37/50 | Iter: 0/8299 | Discriminator Loss: 0.3737 | Generator Loss: 4.7639
# Epoch/Epochs: 37/50 | Iter: 5000/8299 | Discriminator Loss: 0.3928 | Generator Loss: 2.7415
# Epoch/Epochs: 37/50 | Discriminator Loss : 0.3874 | Generator loss: 4.8838 
# Epoch/Epochs: 38/50 | Iter: 0/8299 | Discriminator Loss: 0.3481 | Generator Loss: 2.8839
# Epoch/Epochs: 38/50 | Iter: 5000/8299 | Discriminator Loss: 0.3440 | Generator Loss: 4.6328
# Epoch/Epochs: 38/50 | Discriminator Loss : 0.3865 | Generator loss: 4.8668 
# Epoch/Epochs: 39/50 | Iter: 0/8299 | Discriminator Loss: 0.3822 | Generator Loss: 5.8024
# Epoch/Epochs: 39/50 | Iter: 5000/8299 | Discriminator Loss: 0.4669 | Generator Loss: 2.3760
# Epoch/Epochs: 39/50 | Discriminator Loss : 0.3869 | Generator loss: 4.8567 
# Epoch/Epochs: 40/50 | Iter: 0/8299 | Discriminator Loss: 0.3510 | Generator Loss: 6.6908
# Epoch/Epochs: 40/50 | Iter: 5000/8299 | Discriminator Loss: 0.3430 | Generator Loss: 6.2795
# Epoch/Epochs: 40/50 | Discriminator Loss : 0.3859 | Generator loss: 4.8776 
# Epoch/Epochs: 41/50 | Iter: 0/8299 | Discriminator Loss: 0.3502 | Generator Loss: 6.1670
# Epoch/Epochs: 41/50 | Iter: 5000/8299 | Discriminator Loss: 0.3504 | Generator Loss: 6.2387
# Epoch/Epochs: 41/50 | Discriminator Loss : 0.3850 | Generator loss: 4.9123 
# Epoch/Epochs: 42/50 | Iter: 0/8299 | Discriminator Loss: 0.3311 | Generator Loss: 5.6953
# Epoch/Epochs: 42/50 | Iter: 5000/8299 | Discriminator Loss: 0.3554 | Generator Loss: 5.9011
# Epoch/Epochs: 42/50 | Discriminator Loss : 0.3842 | Generator loss: 4.9358 
# Epoch/Epochs: 43/50 | Iter: 0/8299 | Discriminator Loss: 0.3394 | Generator Loss: 6.0122
# Epoch/Epochs: 43/50 | Iter: 5000/8299 | Discriminator Loss: 0.3325 | Generator Loss: 6.4785
# Epoch/Epochs: 43/50 | Discriminator Loss : 0.3831 | Generator loss: 4.9670 
# Epoch/Epochs: 44/50 | Iter: 0/8299 | Discriminator Loss: 0.3555 | Generator Loss: 6.6303
# Epoch/Epochs: 44/50 | Iter: 5000/8299 | Discriminator Loss: 0.3529 | Generator Loss: 5.0026
# Epoch/Epochs: 44/50 | Discriminator Loss : 0.3824 | Generator loss: 4.9875 
# Epoch/Epochs: 45/50 | Iter: 0/8299 | Discriminator Loss: 0.3768 | Generator Loss: 5.8738
# Epoch/Epochs: 45/50 | Iter: 5000/8299 | Discriminator Loss: 0.3433 | Generator Loss: 4.3444
# Epoch/Epochs: 45/50 | Discriminator Loss : 0.3829 | Generator loss: 4.9778 
# Epoch/Epochs: 46/50 | Iter: 0/8299 | Discriminator Loss: 0.3613 | Generator Loss: 4.2595
# Epoch/Epochs: 46/50 | Iter: 5000/8299 | Discriminator Loss: 0.3372 | Generator Loss: 5.3025
# Epoch/Epochs: 46/50 | Discriminator Loss : 0.3820 | Generator loss: 4.9928 
# Epoch/Epochs: 47/50 | Iter: 0/8299 | Discriminator Loss: 0.3356 | Generator Loss: 6.1092
# Epoch/Epochs: 47/50 | Iter: 5000/8299 | Discriminator Loss: 0.3525 | Generator Loss: 3.9497
# Epoch/Epochs: 47/50 | Discriminator Loss : 0.3812 | Generator loss: 4.9977 
# Epoch/Epochs: 48/50 | Iter: 0/8299 | Discriminator Loss: 0.3389 | Generator Loss: 5.8874
# Epoch/Epochs: 48/50 | Iter: 5000/8299 | Discriminator Loss: 0.3448 | Generator Loss: 5.8812
# Epoch/Epochs: 48/50 | Discriminator Loss : 0.3805 | Generator loss: 5.0053 
# Epoch/Epochs: 49/50 | Iter: 0/8299 | Discriminator Loss: 0.3380 | Generator Loss: 4.2348
# Epoch/Epochs: 49/50 | Iter: 5000/8299 | Discriminator Loss: 0.3422 | Generator Loss: 6.3342
# Epoch/Epochs: 49/50 | Discriminator Loss : 0.3798 | Generator loss: 5.0151 
# 
# second try with noise addition/data agugmentation 
# and larger generator: 
# 
# Epoch/Epochs: 0/50 | Iter: 0/8299 | Discriminator Loss: 1.3796 | Generator Loss: 0.6428
# Epoch/Epochs: 0/50 | Iter: 5000/8299 | Discriminator Loss: 0.4917 | Generator Loss: 2.1808
# Epoch/Epochs: 0/50 | Discriminator Loss : 0.8358 | Generator loss: 0.9456 
# Epoch/Epochs: 1/50 | Iter: 0/8299 | Discriminator Loss: 0.6822 | Generator Loss: 0.0559
# Epoch/Epochs: 1/50 | Iter: 5000/8299 | Discriminator Loss: 0.5547 | Generator Loss: 0.7232
# Epoch/Epochs: 1/50 | Discriminator Loss : 0.7084 | Generator loss: 0.7354 
# Epoch/Epochs: 2/50 | Iter: 0/8299 | Discriminator Loss: 0.4608 | Generator Loss: 0.3528
# Epoch/Epochs: 2/50 | Iter: 5000/8299 | Discriminator Loss: 0.4051 | Generator Loss: 0.1116
# Epoch/Epochs: 2/50 | Discriminator Loss : 0.6175 | Generator loss: 0.9389 
# Epoch/Epochs: 3/50 | Iter: 0/8299 | Discriminator Loss: 0.4187 | Generator Loss: 0.1164
# Epoch/Epochs: 3/50 | Iter: 5000/8299 | Discriminator Loss: 0.4851 | Generator Loss: 0.4418
# Epoch/Epochs: 3/50 | Discriminator Loss : 0.5727 | Generator loss: 0.7746 
# Epoch/Epochs: 4/50 | Iter: 0/8299 | Discriminator Loss: 0.3884 | Generator Loss: 0.1763
# Epoch/Epochs: 4/50 | Iter: 5000/8299 | Discriminator Loss: 0.4051 | Generator Loss: 0.2293
# Epoch/Epochs: 4/50 | Discriminator Loss : 0.5359 | Generator loss: 0.6685 
# Epoch/Epochs: 5/50 | Iter: 0/8299 | Discriminator Loss: 0.4432 | Generator Loss: 0.1180
# Epoch/Epochs: 5/50 | Iter: 5000/8299 | Discriminator Loss: 0.3688 | Generator Loss: 0.3176
# Epoch/Epochs: 5/50 | Discriminator Loss : 0.5115 | Generator loss: 0.5889 
# Epoch/Epochs: 6/50 | Iter: 0/8299 | Discriminator Loss: 0.3572 | Generator Loss: 0.1740
# Epoch/Epochs: 6/50 | Iter: 5000/8299 | Discriminator Loss: 0.3309 | Generator Loss: 0.0688
# Epoch/Epochs: 6/50 | Discriminator Loss : 0.4873 | Generator loss: 0.5261 
# Epoch/Epochs: 7/50 | Iter: 0/8299 | Discriminator Loss: 0.3373 | Generator Loss: 0.4239
# Epoch/Epochs: 7/50 | Iter: 5000/8299 | Discriminator Loss: 0.3464 | Generator Loss: 0.1138
# Epoch/Epochs: 7/50 | Discriminator Loss : 0.4696 | Generator loss: 0.4903 
# Epoch/Epochs: 8/50 | Iter: 0/8299 | Discriminator Loss: 0.3348 | Generator Loss: 0.1290
# Epoch/Epochs: 8/50 | Iter: 5000/8299 | Discriminator Loss: 0.3407 | Generator Loss: 0.0633
# Epoch/Epochs: 8/50 | Discriminator Loss : 0.4547 | Generator loss: 0.4485 
# Epoch/Epochs: 9/50 | Iter: 0/8299 | Discriminator Loss: 0.3283 | Generator Loss: 0.0771
# Epoch/Epochs: 9/50 | Iter: 5000/8299 | Discriminator Loss: 0.3641 | Generator Loss: 0.6683
# Epoch/Epochs: 9/50 | Discriminator Loss : 0.4434 | Generator loss: 0.4354 
# Epoch/Epochs: 10/50 | Iter: 0/8299 | Discriminator Loss: 0.3435 | Generator Loss: 0.0830
# Epoch/Epochs: 10/50 | Iter: 5000/8299 | Discriminator Loss: 0.3538 | Generator Loss: 0.0962
# Epoch/Epochs: 10/50 | Discriminator Loss : 0.4342 | Generator loss: 0.4106 
# Epoch/Epochs: 11/50 | Iter: 0/8299 | Discriminator Loss: 0.3334 | Generator Loss: 0.2468
# Epoch/Epochs: 11/50 | Iter: 5000/8299 | Discriminator Loss: 0.3316 | Generator Loss: 3.0445
# Epoch/Epochs: 11/50 | Discriminator Loss : 0.4289 | Generator loss: 0.4870 
# Epoch/Epochs: 12/50 | Iter: 0/8299 | Discriminator Loss: 0.3672 | Generator Loss: 0.8241
# Epoch/Epochs: 12/50 | Iter: 5000/8299 | Discriminator Loss: 0.3357 | Generator Loss: 0.1275
# Epoch/Epochs: 12/50 | Discriminator Loss : 0.4225 | Generator loss: 0.4794 
# Epoch/Epochs: 13/50 | Iter: 0/8299 | Discriminator Loss: 0.3435 | Generator Loss: 0.3437
# Epoch/Epochs: 13/50 | Iter: 5000/8299 | Discriminator Loss: 0.3278 | Generator Loss: 0.0659
# Epoch/Epochs: 13/50 | Discriminator Loss : 0.4162 | Generator loss: 0.4567 
# Epoch/Epochs: 14/50 | Iter: 0/8299 | Discriminator Loss: 0.5264 | Generator Loss: 0.0307
# Epoch/Epochs: 14/50 | Iter: 5000/8299 | Discriminator Loss: 0.3273 | Generator Loss: 0.0589
# Epoch/Epochs: 14/50 | Discriminator Loss : 0.4148 | Generator loss: 0.4299 
# Epoch/Epochs: 15/50 | Iter: 0/8299 | Discriminator Loss: 0.3303 | Generator Loss: 0.1005
# Epoch/Epochs: 15/50 | Iter: 5000/8299 | Discriminator Loss: 0.3600 | Generator Loss: 2.8284
# Epoch/Epochs: 15/50 | Discriminator Loss : 0.4103 | Generator loss: 0.4716 
# Epoch/Epochs: 16/50 | Iter: 0/8299 | Discriminator Loss: 0.3402 | Generator Loss: 0.3626
# Epoch/Epochs: 16/50 | Iter: 5000/8299 | Discriminator Loss: 0.3335 | Generator Loss: 0.3848
# Epoch/Epochs: 16/50 | Discriminator Loss : 0.4058 | Generator loss: 0.4607 
# Epoch/Epochs: 17/50 | Iter: 0/8299 | Discriminator Loss: 0.3613 | Generator Loss: 0.0929
# Epoch/Epochs: 17/50 | Iter: 5000/8299 | Discriminator Loss: 0.3282 | Generator Loss: 0.0538
# Epoch/Epochs: 17/50 | Discriminator Loss : 0.4025 | Generator loss: 0.4389 
# Epoch/Epochs: 18/50 | Iter: 0/8299 | Discriminator Loss: 0.3261 | Generator Loss: 0.0782
# Epoch/Epochs: 18/50 | Iter: 5000/8299 | Discriminator Loss: 0.3324 | Generator Loss: 0.2204
# Epoch/Epochs: 18/50 | Discriminator Loss : 0.3991 | Generator loss: 0.4229 
# Epoch/Epochs: 19/50 | Iter: 0/8299 | Discriminator Loss: 0.3296 | Generator Loss: 0.0993
# Epoch/Epochs: 19/50 | Iter: 5000/8299 | Discriminator Loss: 0.3280 | Generator Loss: 2.2213
# Epoch/Epochs: 19/50 | Discriminator Loss : 0.3956 | Generator loss: 0.4426 
# Epoch/Epochs: 20/50 | Iter: 0/8299 | Discriminator Loss: 0.3275 | Generator Loss: 0.1223
# Epoch/Epochs: 20/50 | Iter: 5000/8299 | Discriminator Loss: 0.3455 | Generator Loss: 0.3317
# Epoch/Epochs: 20/50 | Discriminator Loss : 0.3926 | Generator loss: 0.4306 
# Epoch/Epochs: 21/50 | Iter: 0/8299 | Discriminator Loss: 0.3372 | Generator Loss: 0.0628
# Epoch/Epochs: 21/50 | Iter: 5000/8299 | Discriminator Loss: 0.3269 | Generator Loss: 0.0592
# Epoch/Epochs: 21/50 | Discriminator Loss : 0.3898 | Generator loss: 0.4144 
# Epoch/Epochs: 22/50 | Iter: 0/8299 | Discriminator Loss: 0.3290 | Generator Loss: 0.1799
# Epoch/Epochs: 22/50 | Iter: 5000/8299 | Discriminator Loss: 0.3324 | Generator Loss: 0.3507
# Epoch/Epochs: 22/50 | Discriminator Loss : 0.3872 | Generator loss: 0.4057 
# Epoch/Epochs: 23/50 | Iter: 0/8299 | Discriminator Loss: 0.3283 | Generator Loss: 0.1559
# Epoch/Epochs: 23/50 | Iter: 5000/8299 | Discriminator Loss: 0.3261 | Generator Loss: 0.0584
# Epoch/Epochs: 23/50 | Discriminator Loss : 0.3847 | Generator loss: 0.3923 
# Epoch/Epochs: 24/50 | Iter: 0/8299 | Discriminator Loss: 0.3257 | Generator Loss: 0.0387
# Epoch/Epochs: 24/50 | Iter: 5000/8299 | Discriminator Loss: 0.3329 | Generator Loss: 0.2434
# Epoch/Epochs: 24/50 | Discriminator Loss : 0.3826 | Generator loss: 0.3814 
# Epoch/Epochs: 25/50 | Iter: 0/8299 | Discriminator Loss: 0.3330 | Generator Loss: 0.0755
# Epoch/Epochs: 25/50 | Iter: 5000/8299 | Discriminator Loss: 0.3304 | Generator Loss: 0.0804
# Epoch/Epochs: 25/50 | Discriminator Loss : 0.3806 | Generator loss: 0.3703 
# Epoch/Epochs: 26/50 | Iter: 0/8299 | Discriminator Loss: 0.3265 | Generator Loss: 0.1570
# Epoch/Epochs: 26/50 | Iter: 5000/8299 | Discriminator Loss: 0.3319 | Generator Loss: 0.0708
# Epoch/Epochs: 26/50 | Discriminator Loss : 0.3786 | Generator loss: 0.3611 
# Epoch/Epochs: 27/50 | Iter: 0/8299 | Discriminator Loss: 0.3311 | Generator Loss: 0.1208
# Epoch/Epochs: 27/50 | Iter: 5000/8299 | Discriminator Loss: 0.3282 | Generator Loss: 0.1310
# Epoch/Epochs: 27/50 | Discriminator Loss : 0.3769 | Generator loss: 0.3527 
# Epoch/Epochs: 28/50 | Iter: 0/8299 | Discriminator Loss: 0.3273 | Generator Loss: 0.1171
# Epoch/Epochs: 28/50 | Iter: 5000/8299 | Discriminator Loss: 0.3507 | Generator Loss: 0.1679
# Epoch/Epochs: 28/50 | Discriminator Loss : 0.3755 | Generator loss: 0.3452 
# Epoch/Epochs: 29/50 | Iter: 0/8299 | Discriminator Loss: 0.3279 | Generator Loss: 0.1068
# Epoch/Epochs: 29/50 | Iter: 5000/8299 | Discriminator Loss: 0.3298 | Generator Loss: 0.2010
# Epoch/Epochs: 29/50 | Discriminator Loss : 0.3739 | Generator loss: 0.3487 
# Epoch/Epochs: 30/50 | Iter: 0/8299 | Discriminator Loss: 0.3377 | Generator Loss: 0.3742
# Epoch/Epochs: 30/50 | Iter: 5000/8299 | Discriminator Loss: 0.3375 | Generator Loss: 0.2570
# Epoch/Epochs: 30/50 | Discriminator Loss : 0.3726 | Generator loss: 0.3459 
# Epoch/Epochs: 31/50 | Iter: 0/8299 | Discriminator Loss: 0.3273 | Generator Loss: 0.1375
# Epoch/Epochs: 31/50 | Iter: 5000/8299 | Discriminator Loss: 0.3599 | Generator Loss: 0.7176
# Epoch/Epochs: 31/50 | Discriminator Loss : 0.3716 | Generator loss: 0.3551 
# Epoch/Epochs: 32/50 | Iter: 0/8299 | Discriminator Loss: 0.3398 | Generator Loss: 1.1232
# Epoch/Epochs: 32/50 | Iter: 5000/8299 | Discriminator Loss: 0.3258 | Generator Loss: 0.0987
# Epoch/Epochs: 32/50 | Discriminator Loss : 0.3704 | Generator loss: 0.3592 
# Epoch/Epochs: 33/50 | Iter: 0/8299 | Discriminator Loss: 0.3291 | Generator Loss: 0.2774
# Epoch/Epochs: 33/50 | Iter: 5000/8299 | Discriminator Loss: 0.3288 | Generator Loss: 0.1186
# Epoch/Epochs: 33/50 | Discriminator Loss : 0.3692 | Generator loss: 0.3529 
# Epoch/Epochs: 34/50 | Iter: 0/8299 | Discriminator Loss: 0.3263 | Generator Loss: 3.2918
# Epoch/Epochs: 34/50 | Iter: 5000/8299 | Discriminator Loss: 0.3514 | Generator Loss: 0.0371
# Epoch/Epochs: 34/50 | Discriminator Loss : 0.3682 | Generator loss: 0.3756 
# Epoch/Epochs: 35/50 | Iter: 0/8299 | Discriminator Loss: 0.3296 | Generator Loss: 0.1227
# Epoch/Epochs: 35/50 | Iter: 5000/8299 | Discriminator Loss: 0.4661 | Generator Loss: 0.1242
# Epoch/Epochs: 35/50 | Discriminator Loss : 0.3684 | Generator loss: 0.3733 
# Epoch/Epochs: 36/50 | Iter: 0/8299 | Discriminator Loss: 0.3359 | Generator Loss: 0.4327
# Epoch/Epochs: 36/50 | Iter: 5000/8299 | Discriminator Loss: 0.3276 | Generator Loss: 0.1187
# Epoch/Epochs: 36/50 | Discriminator Loss : 0.3674 | Generator loss: 0.3705 
# Epoch/Epochs: 37/50 | Iter: 0/8299 | Discriminator Loss: 0.3273 | Generator Loss: 0.1299
# Epoch/Epochs: 37/50 | Iter: 5000/8299 | Discriminator Loss: 0.3326 | Generator Loss: 0.2645
# Epoch/Epochs: 37/50 | Discriminator Loss : 0.3664 | Generator loss: 0.3687 
# Epoch/Epochs: 38/50 | Iter: 0/8299 | Discriminator Loss: 0.3678 | Generator Loss: 0.4906
# Epoch/Epochs: 38/50 | Iter: 5000/8299 | Discriminator Loss: 0.3311 | Generator Loss: 0.0654
# Epoch/Epochs: 38/50 | Discriminator Loss : 0.3659 | Generator loss: 0.3650 
# Epoch/Epochs: 39/50 | Iter: 0/8299 | Discriminator Loss: 0.3276 | Generator Loss: 0.1085
# Epoch/Epochs: 39/50 | Iter: 5000/8299 | Discriminator Loss: 0.3274 | Generator Loss: 0.1476
# Epoch/Epochs: 39/50 | Discriminator Loss : 0.3651 | Generator loss: 0.3605 
# Epoch/Epochs: 40/50 | Iter: 0/8299 | Discriminator Loss: 0.3345 | Generator Loss: 0.2792
# Epoch/Epochs: 40/50 | Iter: 5000/8299 | Discriminator Loss: 0.3276 | Generator Loss: 0.0541
# Epoch/Epochs: 40/50 | Discriminator Loss : 0.3642 | Generator loss: 0.3549 
# Epoch/Epochs: 41/50 | Iter: 0/8299 | Discriminator Loss: 0.3434 | Generator Loss: 0.0634
# Epoch/Epochs: 41/50 | Iter: 5000/8299 | Discriminator Loss: 0.3268 | Generator Loss: 0.0858
# Epoch/Epochs: 41/50 | Discriminator Loss : 0.3635 | Generator loss: 0.3481 
# Epoch/Epochs: 42/50 | Iter: 0/8299 | Discriminator Loss: 0.3258 | Generator Loss: 0.0578
# Epoch/Epochs: 42/50 | Iter: 5000/8299 | Discriminator Loss: 0.3689 | Generator Loss: 2.0997
# Epoch/Epochs: 42/50 | Discriminator Loss : 0.3630 | Generator loss: 0.3571 
# Epoch/Epochs: 43/50 | Iter: 0/8299 | Discriminator Loss: 0.3363 | Generator Loss: 0.0501
# Epoch/Epochs: 43/50 | Iter: 5000/8299 | Discriminator Loss: 0.3264 | Generator Loss: 0.1211
# Epoch/Epochs: 43/50 | Discriminator Loss : 0.3622 | Generator loss: 0.3508 
# Epoch/Epochs: 44/50 | Iter: 0/8299 | Discriminator Loss: 0.3271 | Generator Loss: 0.0656
# Epoch/Epochs: 44/50 | Iter: 5000/8299 | Discriminator Loss: 0.3971 | Generator Loss: 0.0756
# Epoch/Epochs: 44/50 | Discriminator Loss : 0.3620 | Generator loss: 0.3444 
# Epoch/Epochs: 45/50 | Iter: 0/8299 | Discriminator Loss: 0.3262 | Generator Loss: 0.0580
# Epoch/Epochs: 45/50 | Iter: 5000/8299 | Discriminator Loss: 0.3265 | Generator Loss: 0.2380
# Epoch/Epochs: 45/50 | Discriminator Loss : 0.3612 | Generator loss: 0.3396 
# Epoch/Epochs: 46/50 | Iter: 0/8299 | Discriminator Loss: 0.3282 | Generator Loss: 0.0776
# Epoch/Epochs: 46/50 | Iter: 5000/8299 | Discriminator Loss: 0.3345 | Generator Loss: 0.0745
# Epoch/Epochs: 46/50 | Discriminator Loss : 0.3633 | Generator loss: 0.3337 
# Epoch/Epochs: 47/50 | Iter: 0/8299 | Discriminator Loss: 0.5611 | Generator Loss: 0.0289
# Epoch/Epochs: 47/50 | Iter: 5000/8299 | Discriminator Loss: 0.3264 | Generator Loss: 0.1181
# Epoch/Epochs: 47/50 | Discriminator Loss : 0.3641 | Generator loss: 0.3281 
# Epoch/Epochs: 48/50 | Iter: 0/8299 | Discriminator Loss: 0.3256 | Generator Loss: 0.0465
# Epoch/Epochs: 48/50 | Iter: 5000/8299 | Discriminator Loss: 0.3264 | Generator Loss: 0.0961
# Epoch/Epochs: 48/50 | Discriminator Loss : 0.3633 | Generator loss: 0.3232 
# Epoch/Epochs: 49/50 | Iter: 0/8299 | Discriminator Loss: 0.3286 | Generator Loss: 0.1229
# Epoch/Epochs: 49/50 | Iter: 5000/8299 | Discriminator Loss: 0.3398 | Generator Loss: 0.6905
# Epoch/Epochs: 49/50 | Discriminator Loss : 0.3627 | Generator loss: 0.3244 
#

#%%

#%%
# back to improvements new architecture 
# WGANGP 
#
#%%
# progan?
# Stylegan2/3?
#%%
# a detour to something fun CycleGAN