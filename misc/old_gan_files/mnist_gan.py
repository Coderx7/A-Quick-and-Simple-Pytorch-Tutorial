#%% 
# in the name of God the most compassionate the most merciful
# 
# sidenote:
# I wrote these back in 2018/2019 much of the information here are now
# either considered completely obsolete or evolved in a way, only a fraction
# still apply unchanged. we will have a look at these later on as we have 
# much more powerful methods for generations than these early architectures
# and tips and tricks around them that were needed to get the most out of them!
# we will cover some newer GAN architectures that vastly improved upon these
# early architectures. having said that, I only retain them for historical references.
#  
# Here we are going to create a simple GAN network. a GAN network 
# consists of a generator network and a discriminator network. 
# the generator part's job is to get a vector of some length 
# and generate an image and the discriminators job is simply
# identfying if its a real image or not (that is is it generated or not).
# the catch here is the generator will try and ultimately create 
# real life looking iamges that can fool discriminator! 
# this means, it will learn a latent space that the real images
# belong to and simply sampling from it can result in 
# real looking images. 
# so lets see how we can do this 
import torch 
import numpy as np 
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import torch.nn as nn 
import torch.nn.functional as F 

%matplotlib inline

batch_size = 64
n_workers = 2

#create our transformer 
transform = transforms.ToTensor()
train_dataset = datasets.MNIST('../data/MNIST',True, transform=transform, download=True)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# lets see a sample bacth 
imgs, labels = next(iter(train_dataloader))
fig = plt.figure(figsize=(4,3))
ax = fig.add_subplot(111)
image = imgs.numpy()[0].squeeze()
ax.imshow(image, cmap='gray')

# Before we go on, make sure you DO read this :
#    https://github.com/soumith/ganhacks#16-discrete-variables-in-conditional-gans
# they directly affect how GANs are trained!
# dont use relu! use guassian distribution instead of uniform!
# in batch use different batches for real and fake separately (especially if you use batchnorm!)
# use tanh
# use label smoothing
# use adam for generator, and you can use sgd with discriminator! 

# How to Train a GAN? Tips and tricks to make GANs work
# While research in Generative Adversarial Networks (GANs) continues to improve 
# the fundamental stability of these models, we use a bunch of tricks to train 
# them and make them stable day to day.
# Here are a summary of some of the tricks.
# Here's a link to the authors of this document
# If you find a trick that is particularly useful in practice, please open a Pull Request to add it to the document. 
# If we find it to be reasonable and verified, we will merge it in.
# 1. Normalize the inputs
#     normalize the images between -1 and 1
#     Tanh as the last layer of the generator output
# 2: A modified loss function
# In GAN papers, the loss function to optimize G is min (log 1-D), but in practice
# folks practically use max log D because the first formulation has vanishing gradients
# early on Goodfellow et. al (2014)
# In practice, works well:
#     Flip labels when training generator: real = fake, fake = real
# 3: Use a spherical Z
#     Dont sample from a Uniform distribution
# cube.png
#     Sample from a gaussian distribution
# sphere.png
#     When doing interpolations, do the interpolation via a great circle, rather than
#     a straight line from point A to point B
#     Tom White's Sampling Generative Networks ref code https://github.com/dribnet/plat has more details
# 4: BatchNorm
#     Construct different mini-batches for real and fake, i.e. each mini-batch needs to 
#     contain only all real images or all generated images.
#     when batchnorm is not an option use instance normalization (for each sample, 
#     subtract mean and divide by standard deviation).
# batchmix
# 5: Avoid Sparse Gradients: ReLU, MaxPool
#     the stability of the GAN game suffers if you have sparse gradients
#     LeakyReLU = good (in both G and D)
#     For Downsampling, use: Average Pooling, Conv2d + stride
#     For Upsampling, use: PixelShuffle, ConvTranspose2d + stride
#         PixelShuffle: https://arxiv.org/abs/1609.05158
# 6: Use Soft and Noisy Labels
#     Label Smoothing, i.e. if you have two target labels: Real=1 and Fake=0, 
#     then for each incoming sample, if it is real, then replace the label with
#     a random number between 0.7 and 1.2, and if it is a fake sample, replace 
#     it with 0.0 and 0.3 (for example). Salimans et. al. 2016
#     make the labels the noisy for the discriminator: occasionally flip the labels
#     when training the discriminator
# 7: DCGAN / Hybrid Models
#     Use DCGAN when you can. It works!
#     if you cant use DCGANs and no model is stable, use a hybrid model : KL + GAN or VAE + GAN
# 8: Use stability tricks from RL
#     Experience Replay
#         Keep a replay buffer of past generations and occassionally show them
#         Keep checkpoints from the past of G and D and occassionaly swap them 
#         out for a few iterations
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

# while lossD > A:
#   train D
# while lossG > B:
#   train G

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
#sidenot:
# this implements the early version of GAN. obviously using linear
# layers is not anywhere near optimal choice when it comes to iamges
# however, in order to keep things simple and replicate the early architectures
# we used linear layers here. we see much improvements using dcgan ahead which
# utilizes cnn and few other enhancements to get much better results!
#
# now lets define our models 
# the discriminator first, its a simple normal network! 
# accepts something and says if its something legit or not!!!
# the catch is, we seem to need to use leaky relu only!!!!!! 
class DiscriminatorNet (torch.nn.Module):
    def __init__(self, input_dim, hidden_size, output_dim, act = nn.LeakyReLU(0.2)):
        super().__init__()

        self.act = act
        self.fc1 = nn.Linear(input_dim, hidden_size*4)
        self.fc2 = nn.Linear(hidden_size*4, hidden_size*2)
        self.fc3 = nn.Linear(hidden_size*2, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_dim)

        self.dropout = nn.Dropout(0.3)

    def forward(self, x:torch.Tensor):

        x = x.flatten(start_dim=1)
        output = self.act(self.fc1(x))
        output = self.dropout(output)
        output = self.act(self.fc2(output))
        output = self.dropout(output)
        output = self.act(self.fc3(output))
        output = self.dropout(output)
        # raw scores!
        output = self.fc4(output)
        return output

# now the generator network. its the same as our discriminator network but 
# becasue we are trying to create images, our outputs should produce values
# that are sensible for images. 
# we can use sigmoid, but it turns out that tanh works better! so our input
# should also be scaled between -1 and 1 instead of 0 anad 1! 
# lets create our generator network!!

class GeneratorNet(nn.Module):
    def __init__(self, input_dim, hidden_size, output_size, act=nn.LeakyReLU(0.2)):
        super().__init__()

        self.act = act
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size*2)
        self.fc3 = nn.Linear(hidden_size*2, hidden_size*4)
        self.fc4 = nn.Linear(hidden_size*4, output_size)
        self.dropout = nn.Dropout(0.3)

    def forward(self, input):
        # the choice of activation function seems not really that 
        # decisive, I mean, relu works fine as well! (for mnist
        # and this architecture, but when dataset becomes larger 
        # and more complex things change quite a bit!)
        output = self.act(self.fc1(input))
        output = self.dropout(output)
        output = self.act(self.fc2(output))
        output = self.dropout(output)
        output = self.act(self.fc3(output))
        output = self.dropout(output)
        # if we use sigmoid here, regardless of scaling
        # our input between -1 and 1 we will not succeed!
        # the discriminators loss decreases well but generators goes up!
        output = F.tanh(self.fc4(output))
        return output


#%%
# for the loss criterion, we shuod know that our Discriminators job is to 
# successfuly recognize which image is fake and which image is real!
# so we should create labels for each image. the real images will have label=1
# becasue they are real! duh?!! and the fake ones are the ones generated by our
# generator network!
# we will use Binary CrossEntropy with Logitsc (BCEntropyWithlogits)
# as our criterion. There is also one minor trick. 
# instead of label =1.0 we actually smooth out our labels, meaning we set 
# labels = 0.9 instead of 1.0 so the discriminator has easier time!
# why does that work? I dont know yet! lets find out!

def real_loss(Discriminators_output, is_smoothed=False):
    batch_size = Discriminators_output.size(0)
    if is_smoothed: 
        labels = torch.ones(batch_size)*0.9
    else : 
        labels = torch.ones(batch_size)
    
    criterion = torch.nn.BCEWithLogitsLoss()
    real_loss = criterion(Discriminators_output.squeeze(), labels)
    return real_loss

def fake_loss(Discriminators_output):
    batch_size = Discriminators_output.size(0)
    labels = torch.zeros(batch_size)
    criterion = nn.BCEWithLogitsLoss()
    fake_loss = criterion(Discriminators_output.squeeze(), labels)
    return fake_loss

#%%
#%% model hyper paramters
#Discriminators 
input_dim_D = 28*28
output_size_D = 1
hidden_size_D = 32 

#Generator 
input_dim_G = 100
output_size_G = 28*28
hidden_size_G = 32

# Now lets create our models 
D = DiscriminatorNet(input_dim_D, hidden_size_D, output_size_D, act=nn.LeakyReLU(0.2))
G = GeneratorNet(input_dim_G, hidden_size_G, output_size_G, nn.LeakyReLU(0.2) )

print(D)
print()
print(G)


#otimizers = 
optimizer_D = torch.optim.Adam(D.parameters(), lr=0.02)
optimizer_G = torch.optim.Adam(G.parameters(), lr=0.002)



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

epochs = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
D = D.to(device)
G = G.to(device)

interval=1000
n_sample = 16
# we use this fixed vector to see how our network works
fixed_z = np.random.uniform(-1,1, size=(n_sample, input_dim_G))
fixed_z = torch.from_numpy(fixed_z).float().to(device)

losses=[]
samples = []

for e in range(epochs):
    D.train()
    G.train()
    for i, (real_images,_) in enumerate(train_dataloader):

        batch_size = real_images.size(0)

        real_images = real_images*2 - 1  # rescale input images from [0,1) to [-1, 1)
        real_images = real_images.to(device)
        real_ouput = D(real_images)
        real_img_loss = real_loss(real_ouput.cpu(), True)

        # now generate an image! 
        fake_images = np.random.uniform(-1,1, size =(batch_size, input_dim_G))
        fake_tensor = torch.from_numpy(fake_images).float().to(device)
        fake_images = G(fake_tensor)
        fake_output = D(fake_images)
        fake_img_loss = fake_loss(fake_output.cpu())
        d_loss = real_img_loss + fake_img_loss

        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        # now the generator part!
        z = np.random.uniform(-1,1,size=(batch_size,input_dim_G))
        z_tensor = torch.from_numpy(z).float().to(device)
        fake_image = G(z_tensor)
        D_fake_output = D(fake_image)
        g_loss = real_loss(D_fake_output.cpu())

        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()

        if i%interval ==0: 
            print(f'iter/epoch: {i}/{e}) D loss : {d_loss.item():.4f} G loss: {g_loss.item():.4f} ')

    losses.append((d_loss.item(),g_loss.item()))
    G.eval()
    fake_images = G(fixed_z)
    samples.append(fake_images)

import pickle as pkl 
from Cython.Shadow import inline
with open('../weights/samples.pkl', 'wb') as file : 
    pkl.dump(samples, file)
#%%
fig, axes = plt.subplots()
losses = np.array(losses)
print(losses)

plt.plot(losses.T[0], label='D loss')
plt.plot(losses.T[1], label='G loss')
plt.title('loss')
plt.legend()
plt.show()


#%%
# lets visualize our samples in each epoch
def vis_samples(samples, title):
    fig, axes = plt.subplots(4, 4, sharex=True, sharey=True)
    samples = samples.cpu().detach().numpy().squeeze()
    for ax, img in zip(axes.flatten(), samples) :
        img = img/2 +1 
        ax.imshow(img.reshape(28,28),cmap='gray')
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False) 
        ax.set_title(title)
with open('samples.pkl','rb') as f: 
    samples = pkl.load(f)

for i in range(len(samples)):    
    vis_samples(samples[i],str(i))
#%%



#%%
