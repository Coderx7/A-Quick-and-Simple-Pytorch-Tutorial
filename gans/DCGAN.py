#%%
#in the name of God
# here we will be creating DCGAN or Deep Convolutional GAN! 
# up until now we have been using simple fully connected layers
# and using gray-scale aka black and white or single channeled images like mnist!
# as it truns out, when dealing with images, Conv layers do wonders! so lets test
# that as well here! 
import torch 
import numpy as np 
import torch.nn as nn 
import torch.nn.functional as F 
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt 
import pickle as pkl
%matplotlib inline 

# we will be working with SVHN dataset, which is a dataset for post number images! 

batch_size = 128
# check what happens if we use augmentations here?! aka us transforms.Compose
transform = transforms.ToTensor()
train_dataset = datasets.SVHN('SVHN', split='train', transform=transform, download=True)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

#visualize 
(imgs, labels) = next(iter(train_dataloader))
#images for pytorch are  tensors, should be converted into numpy! and
# transposed, since their channel is swapped! 
imgs=imgs.numpy().transpose(0,2,3,1)
plt.imshow(imgs[0])


fig = plt.figure(figsize=(10,2))

for i in range (20):
    ax = fig.add_subplot(2, 10, i+1, xticks=[], yticks=[])
    ax.imshow(imgs[i])
    ax.set_title(labels[i].item())

# we need to check the minimum and maximum values of each pixel so we can scale them between -1and 1
# how to do that ? lets see
# 

print(f'min: {imgs[0].min()}')
print(f'max: {imgs[0].max()}')

img0 = imgs[0]*(1-(-1))+ (-1)

# one way for scalig between -1 and 1 is to multiply our input by (max-min) and then + min
# which in our case are +1 and -1 so 
imgs = imgs*(1-(-1))+ (-1) # which if you remember is exactly like before (x*2 - 1!)  

# now lets see one example!
print(imgs.min())
print(imgs.max())

print(img0.min())
print(img0.max())



#
 # if we wanted to work a batch, and calculate min/max for all images in 
 # a batch we would simply do!
# ims_min = imgs.reshape(-1,32*32*3).min(1)
# ims_max = imgs.reshape(-1,32*32*3).max(1)
# print('all min values in imags: ',ims_min)
# print('all max values in imags: ',ims_max)
# minimum_in_whole_batch = ims.min()
# maximum_in_whole_batch = ims.max()
# print(f'minimum value in the whole batch: {minimum_in_whole_batch}')
# print(f'maximum value in the whole batch: {maximum_in_whole_batch}')

# Before we go, make sure you DO read this :
#    https://github.com/soumith/ganhacks#16-discrete-variables-in-conditional-gans
# they directly affect how GANs are trained!
# dont use relu! use guassian distribution instead of uniform!
# in batch use different batches for real and fake separately (especially if you use batchnorm!)
# use tanh
# use label smoothing
# use adam for generator, and you can use sgd with discriminator! 

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
# cube
#     Sample from a gaussian distribution
# sphere
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
def conv_batch(in_, out_, kernelsize, stride=2, padding=1, batchnorm=False):
    layers = []
    # we do subsamling by using st ride =2 as well!
    conv = nn.Conv2d(in_, out_, kernelsize,stride, padding)
    layers.append(conv)
    if batchnorm:
        layers.append(nn.BatchNorm2d(out_))
    return nn.Sequential(*layers)

def conv_transpose_batch(in_, out_, kernel_, stride=2, padding=1, batchnorm=False):
    layers = nn.ModuleList()

    conv_trans = nn.ConvTranspose2d(in_, out_, kernel_, stride, padding)
    layers.append(conv_trans)

    if batchnorm:
        layers.append(nn.BatchNorm2d(out_))
    return nn.Sequential(*layers)    


class D_Net(nn.Module):
    def __init__(self, input_dim, output_dim=1, act=nn.LeakyReLU(0.2)):
        super().__init__()

        self.act = act 
        # we do subsamling as well!
        self.conv1 = conv_batch(3, input_dim, 4, 2, 1, batchnorm=False)
        self.conv2 = conv_batch(input_dim, input_dim*2, 4, 2, 1, batchnorm=True)
        self.conv3 = conv_batch(input_dim*2, input_dim*4, 4, 2, 1, batchnorm=True)
        self.fc1 = nn.Linear(input_dim*4*4*4, output_dim)

    def forward(self, input_image_batch):

        output = self.act(self.conv1(input_image_batch))
        output = self.act(self.conv2(output))
        output = self.act(self.conv3(output))

        output = output.view(-1, self.fc1.in_features)
        output = self.act(self.fc1(output))
        return output



class G_Net(nn.Module):
    def __init__(self, z_size, conv_fmap_n,  act=nn.ReLU()):
        super().__init__()

        self.act = act
        self.conv_dim = conv_fmap_n
        
        self.fc1 = nn.Linear(z_size, conv_fmap_n*4*4*4) #its depth, h, w 
        self.convt1 = conv_transpose_batch(conv_fmap_n*4, conv_fmap_n*2, 4, batchnorm=True) #8x8
        self.convt2 = conv_transpose_batch(conv_fmap_n*2, conv_fmap_n, 4, batchnorm=True)#16x16
        self.convt3 = conv_transpose_batch(conv_fmap_n, 3, 4, batchnorm=False)#32x32


    def forward(self, input_vector): 
        output = self.fc1(input_vector)
        output = output.view(-1, self.conv_dim*4, 4, 4) # batch, depth, h,w
        
        output = self.act(self.convt1(output))
        output = self.act(self.convt2(output))

        output = self.convt3(output)
        output = F.tanh(output)

        return output


#%%
# losses! 
def real_loss(input_, smooth=True, device = torch.device('cuda')):
    batch_size = input_.size(0)
    if smooth: 
        # or the simpler mode, torch.ones(batch_size)*0.9
        labels = torch.ones(batch_size) * 0.9
        # seems the simple 0.9 works better overall at leas for this example and number of epochs(50!)
        #labels = torch.ones(batch_size) * ((0.9 - 0.7) * np.random.random_sample() + 0.7)
    else: 
        labels = torch.ones(batch_size)

    labels = labels.to(device)
    criterion = nn.BCEWithLogitsLoss()

    return criterion(input_.squeeze(), labels)

def fake_loss(input, smooth=False, device=torch.device('cuda')):
    batch_size = input.size(0)
    if smooth:
        # smooth the fake labels with a random number in range (0,0.3)
        labels = torch.ones(batch_size).to(device) * ((0.3 - 0.0) * np.random.random_sample() + 0.0)
    else:
        labels = torch.zeros(batch_size).to(device)
    criterion = nn.BCEWithLogitsLoss()

    return criterion(input.squeeze(), labels)

#%% 
#training!

#discriminator
input_size = 32
output_size = 1
D = D_Net(input_size, output_dim=output_size)

#generator
z_size = 100
conv_fmap_n = input_size 
G = G_Net(z_size, conv_fmap_n)

print(D)
print(G)

epochs = 50 
interval = 300

beta1=0.5
beta2=0.999 # default value
optimizer_d = torch.optim.Adam(D.parameters(), 0.0002, [beta1, beta2])
optimizer_g = torch.optim.Adam(G.parameters(), 0.0002, [beta1, beta2])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
D = D.to(device)
G = G.to(device)

losses = []
samples = []

z_vec_fixed = np.random.uniform(-1,1,size=(16, z_size))
z_tensor_fixed = torch.from_numpy(z_vec_fixed).float().to(device)

for e in range(epochs):

    D.train()
    G.train()

    for i, (imgs, _) in enumerate(train_dataloader):

        batch_size = imgs.size(0)

        #scale input*(max-min) + min
        imgs = imgs *(1-(-1))+ (-1)

        imgs = imgs.to(device)
        # train discriminator! 

        outputs = D(imgs)
        real_loss_ = real_loss(outputs, False, device)
        # generate an image using generator 
        z_vector = np.random.uniform(-1,1, size=(batch_size, z_size))
        z_tensor = torch.from_numpy(z_vector).float().to(device)
        image_output_g = G(z_tensor)
        if i==0:
            print(image_output_g.shape)
        outputs_fake = D(image_output_g)
        fake_loss_ = fake_loss(outputs_fake, False, device)
        d_loss = real_loss_ + fake_loss_

        optimizer_d.zero_grad()
        d_loss.backward()
        optimizer_d.step()

        # train genertor !
        z_vector = np.random.uniform(-1,1,size=(batch_size, z_size))
        z_tensor = torch.from_numpy(z_vector).float().to(device)
        fake_image_g = G(z_tensor)
        fake_output = D(fake_image_g)
        # swap loss! 
        real_loss_g = real_loss(fake_output, smooth=False, device=device)

        optimizer_g.zero_grad()
        real_loss_g.backward()
        optimizer_g.step()

        if i% interval==0:
            # append discriminator loss and generator loss
            losses.append((d_loss.item,real_loss_g.item()))
            # print discriminator and generator loss
            print('Epoch [{:5d}/{:5d}] | d_loss: {:6.4f} | g_loss: {:6.4f}'.format(
                    e, epochs, d_loss.item(), real_loss_g.item()))
    
    G.eval()
    images = G(z_tensor_fixed)
    samples.append(images)
with open('dcgan_images.pkl','wb') as f: 
    pkl.dump(samples, f)

#%%
#visualization!
fig, axes = plt.subplots()
losses = np.array(losses)
print(losses)

plt.plot(losses.T[0],label='D loss')
plt.plot(losses.T[1],label='G loss')
plt.title('loss')
plt.legend()
plt.show()


#%%
# lets visualize our samples in each epoch
def vis_samples(samples, title):
    fig, axes = plt.subplots(4, 4, sharex=True, sharey=True)
    samples = samples.cpu().detach().numpy().transpose(0,2,3,1)
    
    for ax, img in zip(axes.flatten(), samples) :
        img = ((img +1)*255 / (2)).astype(np.uint8) # rescale to pixel range (0-255)
        ax.imshow(img.reshape((32,32,3)))
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False) 
        ax.set_title(title)
with open('dcgan_images.pkl','rb') as f: 
    samples = pkl.load(f)

for i in range(len(samples)):    
    vis_samples(samples[i],str(i))

#%%
