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
def display_images(imgs, rows=8, title='',unnormalize=False):
    plt.title(title)
    # images that are generated by generator will have gradients
    # because they are normal tensors we treat them as images so
    # we get rid of their gradients here because we dont need it
    imgs = imgs.cpu().detach()
    # rescale back to 0-1 range from -1/1 range
    imgs = (imgs+1)/2 if unnormalize else imgs
    # print(f'{imgs.min()=}')
    # print(f'{imgs.max()=}')
    images = utils.make_grid(imgs,nrow=rows).cpu().numpy().transpose(1,2,0) # c,h,w -> h,w,c
    plt.imshow(images, cmap='gray')
    plt.show()
    
imgs, labels = next(iter(train_loader))
display_images(imgs, title='sample batch from mnist')
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
# dont use relu! 
# use Guassian distribution instead of uniform!
# in batch use different batches for real and fake separately (especially if you use batchnorm!)
# use tanh
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
                 stride=2, padding=1, batch_norm=False):
        super().__init__()
        self.block = nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                             stride, padding, bias=not batch_norm),
                                   nn.BatchNorm2d(out_channels) if batch_norm else
                                   nn.Identity())
        # add residual connection, grab the input upsample it 
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
        x_res = self.residual(x)
        # used relu on (out+x_res) and it completely destroys generatioN!
        out = out+x_res
        # print(f'{out.shape=}')
        return out

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
                                 # unflatten the output of linear layer back to 3d
                                 # shape to be fed to convtranspos2d. we use Unflatten()
                                 # specify the dim we want to unflatten which is 1 
                                 # (cuz linear is 2d (batch, dim)) and then reshape it to
                                 # (out_channels, h,w) 
                                 nn.Unflatten(dim=1, unflattened_size=(hidden_size*4, 4, 4)),
                                 ConvTransBlock(hidden_size*4, hidden_size*2, 4, batch_norm=True), #8x8
                                 ConvTransBlock(hidden_size*2, hidden_size, 4, batch_norm=True),   #16x16
                                 # disable batchnorm for last layer of generator so 
                                 # it doesnt normalize the image values!
                                 ConvTransBlock(hidden_size, 3, 4, batch_norm=False),              #32x32
                                 nn.Tanh())   
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
batch_size = 64
num_workers = 8
# check what happens if we use augmentations here?! aka us transforms.Compose
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor()
])

train_dataset = datasets.SVHN('./data/SVHN', split='extra', transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

#visualize 
(imgs, labels) = next(iter(train_loader))
display_images(imgs, title='svhn samples')

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

#%% training!

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

disc_hidden_size = 32
gen_hidden_size = 64
z_size = 128

epochs = 50 
interval = 5000

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

losses = []

for epoch in range(epochs):

    discriminatorcnn.train()
    generatorcnn.train()

    for i, (imgs_real, _) in enumerate(train_loader):

        #scale input to [-1,1]
        imgs_real = (2*imgs_real-1).to(device)
        
        # before we go on lets add small gaussian noise to real images
        # I add this later when I noticed heavy mode collapse happining
        # we do this to both real and fake images to fight mode collapse
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
        
        # add noise to fake images as well
        imgs_fake += 0.05 * torch.randn_like(imgs_fake)
        
        preds_fake = discriminatorcnn(imgs_fake)
        disc_fake_loss = fake_loss(preds_fake, smooth=False, device=device)
        # calculate discrimiator loss out of real and fake losses
        disc_loss = disc_real_loss + disc_fake_loss
        # and optimize discrimnator 
        disc_optimizer.zero_grad()
        disc_loss.backward()
        disc_optimizer.step()

        # now train genertor to create images that look real
        z_vector = torch.distributions.Uniform(-1,1).sample((imgs_real.size(0),z_size)).to(device)
        fake_imgs = generatorcnn(z_vector)
        preds_fake = discriminatorcnn(fake_imgs)
        
        # swap loss! treat fake images as real images
        # gen_real_loss = real_loss(preds_fake, smooth=False, device=device)
        # ok that doesnt work properly and we are seeing mode collapse so
        # instead occasionally (around 5% of the times) flip labels 
        if torch.rand(1).item() < 0.05:
            gen_real_loss = fake_loss(preds_fake, device=device)
        else:
            gen_real_loss = real_loss(preds_fake, smooth=False, device=device)
            
        # optimize generator
        gen_optimizer.zero_grad()
        gen_real_loss.backward()
        gen_optimizer.step()

        if i% interval==0:
            # append discriminator loss and generator loss
            losses.append((disc_loss.item(), gen_real_loss.item()))
            # print discriminator and generator loss
            print(f'Epoch/Epochs: {epoch}/{epochs} | Iter: {i}/{len(train_loader)} | Discriminator Loss: {disc_loss:6.4f} | Generator Loss: {gen_real_loss:6.4f}')

    losses.append((disc_loss.item(), gen_real_loss.item()))
    print(f'Epoch/Epochs: {epoch}/{epochs} | Discriminator Loss : {np.mean(np.array(losses)[:,0]):.4f} | Generator loss: {np.mean(np.array(losses)[:,1]):.4f} ')
    # generate some images mid training to evaluate our model's performance 
    generatorcnn.eval()
    # reshape images back to 32x32x3
    generated_images = generatorcnn(fixed_z).view(-1,*imgs_real.shape[1:])
    display_images(generated_images, 
                   rows=gen_num_samples//8,
                   title=f'Generated Images at Epoch {epoch}',
                   unnormalize=True)
    
    
    
    
    
#%%
losses = np.array(losses)

plt.plot(losses[:,0],label="Discriminator's loss")
plt.plot(losses[:,1],label="Generator's loss")
plt.title('Loss')
plt.legend()
plt.show()

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