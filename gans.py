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
from pathlib import Path

import torch 
import numpy as np 

import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils.data import DataLoader

from torchvision import datasets, transforms
from torchvision import utils, models

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
                }, f"./weights/dcgan_generatorcnn_{experiment_date}.pt")
    
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
states = torch.load('./weights/dcgan_generatorcnn_20250812172504.pt', weights_only=False)
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
random_gen = torch.manual_seed(66)

@torch.no_grad()
def interpolate_latents(generator, z1, z2, steps=8, eps=1e-8):
    generator.eval()
    alphas = torch.linspace(0, 1, steps)
    grids = []
    for a in alphas:
        z_interp = (1-a)*z1 + a*z2 + eps
        imgs = generator(z_interp)
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
def choose_meanstd(generator, num_samples,random_gen, ncols=8,):
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
            imgs = generator(new_z)
                       
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
def latent_arithmetic_unconditional(generator, z_with_attr, z_without_attr, z_base, alpha_values,ncols=8):
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
        imgs = generator(z_new).cpu()
        imgs_grid = utils.make_grid(imgs,nrow=ncols)
        results.append(imgs_grid)
    return torch.stack(results)

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
    random_gen, batch_size=32, top_ratio=0.10, device="cpu"):

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
        imgs = generator(z)
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
def apply_direction(generator, z, direction, alphas=(-3,-2,-1,0,1,2,3)):
    direction = direction.to(z.device)
    imgs = []
    # print(f'direction vector = {direction}')
    for a in alphas:
        z_new = z + a * direction.unsqueeze(0)
        img = generator(z_new)
        imgs.append(img)
    return torch.cat(imgs, dim=0)

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
                }, f"./weights/dcgan_generatorcnnconditional_{experiment_date}.pt")
    
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
states = torch.load('./weights/dcgan_generatorcnnconditional_20250813125606.pt',
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
#%%
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
    attr_name, celeba_attr_word2idx, num_samples, random_generator, threshold=0.5, device='cuda'):
    
    generator.to(device)
    classifier.to(device)
    
    generator.eval()
    classifier.eval()
    
    z = torch.randn(size=(num_samples,generator.z_size), device=device, generator=random_generator)
    imgs = generator(z)
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
                        celeba_attr_word2idx, attr_name, num_samples, random_gen, device,threshold=0.1):
    generator.eval()
    generator.to(device)
    classifier.eval()
    classifier.to(device)
    
    z_base = torch.randn(size=(num_samples, generator.z_size), device=device, generator=random_gen)
    imgs = generator(z_base)
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
# and the loss is simply the everage of fake-real values
# we then need to clip the model weights after each discriminator 
# optimizer step.
def wgan_critic_loss(preds_real, preds_fake):
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
    # or as we said at the begining dont change sharply for small changes in input!)
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
from scipy import linalg
class IS_FID_Calculator():
    def __init__(self, device='cpu'):
        
        self.device = device
        
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
    def compute_IS(self, images, splits=10):
        # calculate IS = exp(Ex​[KL(p(y∣x) ∥ p(y))]), p(y)=Ex​[p(y∣x)]
        # Ex is expectation of x
        # assign back the classifier in case FID was called
        self.model.fc = self.fc
        images = self._preprocess(images)
        preds = self.model(images).softmax(dim=-1)
                
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
    def _get_FID_mean_covariance(self, imgs):
        imgs = self._preprocess(imgs)
        features = self.model(imgs)
        mean = features.mean(dim=0)
        covariance = torch.cov(features.T)
        return mean, covariance
    
    @torch.no_grad()
    def compute_FID(self, real_imgs, fake_imgs):
        #
        # compute FID=∥ μ_r - μ_f ​∥² + Tr(Σr​ + Σf​ - 2 * (Σr​Σf​)1/2)
        # mu is mean and sigma is covariance matrix
        # Tr(M) means sum of diagonal elements of the input matrix(M)
        # its torch.Trace() 
        
        # remove the classifier, we want 2048 features
        self.model.fc = nn.Identity()
        real_mean,real_cov = self._get_FID_mean_covariance(real_imgs)
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
        cov_prod_sqrt = torch.from_numpy(cov_prod_sqrt).to(real_imgs.device)
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

imgs = torch.randn(size=(10,3,32,32))
metric = IS_FID_Calculator()
iss = metric.compute_IS(imgs)
fids = metric.compute_FID(imgs,imgs+torch.randn_like(imgs))
print(f'{iss=}')
print(f'{fids=}')
#%%
# lets add a few more datasets 
def get_dataloader(dataset_name="SVHN", split=None, resize_dims=(32,32), batch_size=128, num_workers=8, store_path="./data/"):
    dataset_name = dataset_name.lower()

    if dataset_name.lower() == 'mnist':
        if isinstance(split, str):
            split = 'train' in split.lower()
        else:
            split = True if (not split or 'train') else False
                
        transform = transforms.Compose([
        transforms.Resize(resize_dims),
        # transforms.RandomHorizontalFlip(),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor()])
        dataset = datasets.MNIST(os.path.join(store_path, dataset_name.upper()), train=split, transform=transform, download=True)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    
    elif 'cifar' in dataset_name.lower():
        if isinstance(split, str):
            split = 'train' in split.lower()
        else:
            split = True if (not split or 'train') else False
        transform = transforms.Compose([
        transforms.Resize(resize_dims),
        # transforms.RandomHorizontalFlip(),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor()])
        dataset = datasets.CIFAR10(os.path.join(store_path, dataset_name.upper()), train=split, transform=transform, download=True)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    
    elif dataset_name == 'svhn':
        split = 'extra' if not split else split
        transform = transforms.Compose([
        transforms.Resize(resize_dims),
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

#%%
def training_loop(discriminator, generator, train_loader, disc_optimizer, gen_optimizer,
                  epochs, interval, gen_update_interval, dataset_name, loss_type, 
                  lambda_factor=10, gen_num_samples = 64, use_batchnorm=False, wgan_range=(-0.01, 0.01),
                  noise_addition=False,device='cuda', weights_save_dir='./weights', images_save_dir='./results/gan'):
    
    print(f'Datset:                    {dataset_name}')
    print(f'Loss type:                 {loss_type}')
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

        # calculate is/fid scores
        IS_score = metric.compute_IS(imgs_fake)
        FID_score = metric.compute_FID(imgs_real, imgs_fake)
    
        print(f" -- Last Batch : Disc's real mean: {disc_real_mean:.4f} | Disc's fake mean: {disc_fake_mean:.4f}")
        print(f" -- Epoch's Avg: Disc's real mean: {average_score_real_mean:.4f} | Disc's fake mean: {average_score_fake_mean:.4f}")
        print(f'[Epoch {epoch}/{epochs}] Disc Loss-Avg: {d_loss_mean:.6f} | Gen loss-Avg: {g_loss_mean:.6f} | IS: (μ:{IS_score[0]:.4f}, σ²:{IS_score[1]:.4f}) | FID: {FID_score:.2f}')
        
        #save model weights at each epoch
        torch.save({"state_dict":generator.state_dict(),
                "hidden_size":generator.hidden_size,
                "z_size":generator.z_size,
                "lr_d":[p['lr'] for p in disc_optimizer.param_groups],
                "lr_g":[p['lr'] for p in gen_optimizer.param_groups],
                "use_batchnorm":discriminator.use_batchnorm if hasattr(discriminator,'use_batchnorm') else False,
                "noise_addition":noise_addition,
                "wgan_range":wgan_range,
                "epoch":epoch,
                "gen_update_interval":gen_update_interval,
                "loss_type":loss_type,
                "FID":FID_score,
                "IS":IS_score,
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
# mode collapse later on, I havent digged too much though cifar10/celeba seem fine!
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
checkpoint = torch.load("./weights/dcgan_generatorcnn_wgangp_20250901143328.pt",
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
def run_latent_arithmatic(attr_name, generator:GeneratorCNN, classifier:CelebAClassifier,
                          word2idx, 
                          random_gen, 
                          showcase_one_sample=True,
                          num_samples=32, 
                          attribute_pool_size=256,
                          maximum_prob_for_neutral_confidence=0.1,
                          attribute_confidence_rate=0.7,
                          alpha_values=torch.linspace(-3, 7, steps=24),
                          device='cpu'):

    z_base = get_neutral_latents(generator, classifier, word2idx, attr_name, num_samples,
                                random_gen, device,
                                # use lower probs to get more accurate results, 
                                maximum_prob_for_neutral_confidence)

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
                                                device=device)

    (ims_attr, latents_attrs) = with_attrs
    (ims_no_attrs, latents_no_attrs) = without_attrs
    
    print(f'{ims_attr.shape=}')
    print(f'{ims_no_attrs.shape=}')
    
    col_count1 = math.ceil(math.sqrt(ims_attr.size(0)))
    col_count2 = math.ceil(math.sqrt(ims_no_attrs.size(0)))
    
    show_images(ims_attr, f'latents with ({attr_name})', cols=col_count1, figsize=(12,6))
    show_images(ims_no_attrs, f'latents without({attr_name})', cols=col_count2, figsize=(12,6))
    
    latents = z_base[0] if showcase_one_sample else z_base
    
    # from women to male!(woman gradually loses feminity and turns into male)
    imgs = latent_arithmetic_unconditional(generator, latents_attrs, latents_no_attrs[:zs.size(0)], latents, alpha_values)
    show_images(imgs, f'latent arithmetic(opposite toward {attr_name})', figsize=(12,6))
    
    # from male to female!(maleness decreases at each step)
    imgs = latent_arithmetic_unconditional(generator, latents_no_attrs[:zs.size(0)], latents_attrs, latents, alpha_values)
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
# GANs are extremely finiky/fragile to work with, even a slight 
# seemingly okish change can result in complete failure!
# to make this more practical, lets create a version first and then
# analyze it and improve it. 
# sidenote: 
# The following version causes massive instability in trainig!
# we can only train porperly with large lr and even then we dont get
# good results! the lsgan completely fails with severe mode collapse!
# the wgan also fails, and only wgangp manages to produce something intersting!
# (In GANs we need to have simple discriminator anything complex
# powerful complicates things!see explanations ahead!)
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
# lsgan constantly faces severe mode collapse since the gen
# is more powerful than the discriminator! wgangp however
# always does a better job! havent seen mode collapse in nearly
# 100 tests! this shows how stable wgangp is! to get the lsgan to
# not fail, we have to beefup the discriminator 
# wgan also fails for some reason! it was working need to see what
# I had changed!
# update:
# it seems only wgangp trains well. both wgan/lsgan face a lot of instablity
# and flatout fail! I guess to getthem to work one easier? way (not sure!)
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
dataset_name = 'cifar10'
batch_size=128
train_loader = get_dataloader(dataset_name=dataset_name, split='train',resize_dims=(64,64),batch_size=batch_size)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# since the generator is more powerful, and discriminator is
# less powerful we quickly face severe mode collpase, so 
# for lsgan to work we need to beefup the discriminator!
disc_hidden_size = 32
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
discriminatorI64 = DiscriminatorCNN64(hidden_size=disc_hidden_size, 
                                        use_batchnorm=use_batchnorm)
discriminatorI64 = discriminatorI64.to(device)
#generator
generatorI64 = GeneratorCNN64(z_size, hidden_size=gen_hidden_size)
generatorI64 = generatorI64.to(device)

# the paper says [0.5,0.999] diverges in wgan/wgangp
# with bn=True especially for wgan/wgangp, lr must be larger (0.001/0.002)
# otherwise it will take a lot to get there!
# lsgan fails with large lrs(1e-3/2e-3)
betas = [0.5, 0.999] if loss_type=='lsgan' else [0, 0.9]

if loss_type=='lsgan':
    lr_d, lr_g = 0.0001, 0.0002
else:
    lr_d, lr_g = 0.001, 0.002
    
disc_optimizer = torch.optim.Adam(discriminatorI64.parameters(), lr_d, betas=betas)
gen_optimizer = torch.optim.Adam(generatorI64.parameters(), lr_g, betas=betas)

training_loop(discriminatorI64, 
              generatorI64, 
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
# load models 
checkpoint = torch.load("./weights/dcgan_generatorcnn_wgangp_20250901064034.pt",
                        map_location="cpu",
                        weights_only=False)

epoch = checkpoint["epoch"]
z_size = checkpoint["z_size"]
hidden_size = checkpoint["hidden_size"]
dataset_name = checkpoint["dataset_name"]
loss_type = checkpoint["loss_type"]
losses = np.array(checkpoint.pop("losses"))

generatorI64 = GeneratorCNN64(z_size,hidden_size)
generatorI64.load_state_dict(checkpoint.pop("state_dict"))
generatorI64.eval()

for k,v in checkpoint.items():
    print(f'{k}: {v}')
    
print(f'DLoss: {losses[:,0].mean():.4f} | GLoss: {losses[:1].mean():.4f}')
#%%
run_latent_arithmatic(attr_name='Male', 
                      generator=generatorI64, 
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
                      generator=generatorI64, 
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
                      generator=generatorI64, 
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
# stuff of that nature. this is why simplicity is much favored!
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
    def __init__(self, hidden_size=32, act=nn.LeakyReLU(0.2), no_spec_norm_list=None):
        super().__init__()

        self.hidden_size = hidden_size
        self.act = act
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
        self.net = nn.Sequential(DiscConvBlock(3, hidden_size, 4, 2, 1, act_func=act, use_spectral_norm=self.spec_norm_list[0]),#32x32
                                 DiscConvBlock(hidden_size*1, hidden_size*2, 4, 2, 1, act_func=act, use_spectral_norm=self.spec_norm_list[1]),#16x16
                                 DiscConvBlock(hidden_size*2, hidden_size*4, 4, 2, 1, act_func=act, use_spectral_norm=self.spec_norm_list[2]),#8x8
                                 DiscConvBlock(hidden_size*4, hidden_size*4, 4, 2, 1, act_func=act, use_spectral_norm=self.spec_norm_list[3]),#4x4
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
# it wont provide realistic images right off the bat!
#
# update:
# compared to our previous attempt, it seems the wgan/lsgan did massively better
# when we use Batchnorm! which we know is wrong! but then again when we used spectral norm
# things didnt get better! the wgan images are not formed, have black blobs, blurry images
# and it plagues all images!
# this means our discriminator is way more powerful than our generator! the bn works because
# it indirectly acts as a powerful regularizer and it blurs the signal by accumulating
# the statistics of all samples in a batch!
#!(it forces the statistics of the features to be dependent on the entire batch,
# which "blurs" the signal and artificially slows the discriminator down, 
# preventing it from winning so easily) hence we get better results there than here!
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
# todo: check no drpout aswell see if that impacts the same after using larger range for clipping
# wgan is not recommened at all! just go with wgangp! 
#

print(f'Training Improved versions of Discriminator and Generator')
loss_type = 'wgan'
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
#  
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
gen_update_interval = 5 if loss_type == "wgan" else 1

# by using
# [1,2] works only for wgangp, others fail miserably!
# wgan fails with all layers specto normalized when gen_update_interval=5
# with gen_update_interval=1 its still trash!
# lsgan keeps failing with mode collapse (repeative images)! 
no_spec_list = list(range(5)) #[] #[1,2]
#discriminator
discriminatorI64 = DiscriminatorImproved64(hidden_size=disc_hidden_size,
                                           no_spec_norm_list=no_spec_list)
discriminatorI64 = discriminatorI64.to(device)
#generator
generatorI64 = GeneratorImproved64(z_size, hidden_size=gen_hidden_size)
generatorI64 = generatorI64.to(device)

betas = [0.5, 0.999] if loss_type=='lsgan' else [0, 0.9]

if loss_type=='lsgan':
    lr_d, lr_g = 0.0001, 0.0002
elif loss_type == 'wgan':
    # wgan requires way smaller lr like 1e-5, 2e-5
    # and a small clipping range like -0.02,0.02 
    # also rmsprop seems do to much better than adam!
    lr_d,lr_g = 0.00001, 0.00002#1e-5, 2e-5
else:
    lr_d, lr_g = 0.001, 0.002

disc_optimizer = torch.optim.RMSprop(discriminatorI64.parameters(), lr=1e-5) # for wgan
# disc_optimizer = torch.optim.Adam(discriminatorI64.parameters(), lr_d, betas=betas)
gen_optimizer = torch.optim.Adam(generatorI64.parameters(), lr_g, betas=betas)

training_loop(discriminatorI64, 
              generatorI64, 
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
              wgan_range=(-0.02, 0.02), #(-0.05, 0.05)
              noise_addition=True,
              device=device)
#%%
# load models 
checkpoint = torch.load("./weights/dcgan_generatorcnn_wgangp",
                        map_location="cpu",
                        weights_only=False)

epoch = checkpoint["epoch"]
z_size = checkpoint["z_size"]
hidden_size = checkpoint["hidden_size"]
dataset_name = checkpoint["dataset_name"]
loss_type = checkpoint["loss_type"]
losses = np.array(checkpoint.pop("losses"))

generatorI64 = GeneratorImproved64(z_size,hidden_size)
generatorI64.load_state_dict(checkpoint.pop("state_dict"))
generatorI64.eval()

for k,v in checkpoint.items():
    print(f'{k}: {v}')
    
print(f'DLoss: {losses[:,0].mean():.4f} | GLoss: {losses[:1].mean():.4f}')
#%%
run_latent_arithmatic(attr_name='Male', 
                      generator=generatorI64, 
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
                      generator=generatorI64, 
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
                      generator=generatorI64, 
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
# progan?stackgan?
# Stylegan2/3?
#%%
# a detour to something fun CycleGAN
# 