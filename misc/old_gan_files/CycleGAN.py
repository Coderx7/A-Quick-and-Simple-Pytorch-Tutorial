#%%
# in the name of God the most compassionate the most merciful 
# here we will be creating a CycleGAN, a type of GAN that 
# works on image 2 image translation, which means, in simple terms
# does style transfer! (kind of!), it gives us the ability to map 
# one image from anothe rimages domain (turn a summer image, into 
# a winder looking one!!)

import torch 
import numpy as np 
import torchvision
from torchvision import datasets, transforms
import torch.nn as nn 
import torch.nn.functional as F 
import matplotlib.pyplot as plt
import os 

%matplotlib inline 

# read images ( we use yosmite dataset! its in the current director!)
# we have winter and summer subfolders and another folder for test which
# is prefixed with test_. lets create a function for easily reading them
# and making dataloaders we need for our trainings 
def get_data (root='.\summer2winter-yosemite', image_type='summer', batch_size=16, size=128 ):
    transform = transforms.Compose([transforms.Resize(size),
                                     transforms.ToTensor()])
    
    n_workers=2
    data_train_dir = os.path.join(root, image_type)
    train_dir = os.path.join(data_train_dir, image_type)

    data_test_dir = os.path.join(root, 'test_'+image_type)
    test_data = os.path.join(data_test_dir, image_type)
    
    train_dataset = datasets.ImageFolder(data_train_dir, transform=transform)
    test_dataset = datasets.ImageFolder(data_test_dir, transform=transform)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=n_workers,shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,batch_size=batch_size, num_workers=n_workers,)

    return train_dataloader, test_dataloader

#test 
train_dataloader_summer, test_dataloader_summer = get_data(image_type='summer')
train_dataloader_winter, test_dataloader_winter = get_data(image_type='winter') 


#%%

imgs, labels = next(iter(train_dataloader_summer))
#plt.imshow(imgs[0].numpy().transpose(1,2,0))
def visualize(imgs, labels):
    fig = plt.figure(figsize=(10,3))
    #fig.set_figheight(3)
    for i in range(16):
        ax = fig.add_subplot(2, 8, i+1, xticks=[], yticks=[])
        ax.imshow(imgs[i].numpy().transpose(1,2,0))
        ax.set_title(labels[i].item())
def visualize_image(img):
    plt.imshow(img.numpy().transpose(1,2,0))

#visualize(imgs, labels)

imgs, lables = next(iter(train_dataloader_winter))
grid_images = torchvision.utils.make_grid(imgs)
visualize_image(grid_images)
print(grid_images.shape)

#%%
# since we are dealing with GANs, we know we should scale our images in range -1, 1
def scale (image,feature_range=(-1,1)):

    (min, max) = feature_range
    # img*2 - 1
    img = (max-min) * image + min
    return img


#%%
def conv(in_, out_, k_size, stride, pad, batchnorm=True):
    # we can use [] (normal list), but modulelist is a much better choice
    # since all modules will have their attributes and one can use them!!
    layers = nn.ModuleList()

    conv = nn.Conv2d(in_, out_, k_size, stride, pad)
    layers.append(conv)

    if batchnorm:
        layers.append(nn.BatchNorm2d(num_features=out_))
        
    return nn.Sequential(*layers)

class Discriminator(nn.Module):
    def __init__(self, conv_depth, out_dim=1):
        super().__init__()

        self.conv_depth = conv_depth 
        # our input image is 128x128. and the output fmap is calculated like this:
        #(w-k)+2p/s + 1 = (128-4)+2/2 +1 = 64
        self.conv1 = conv(3, conv_depth, k_size=4, stride=2, pad=1, batchnorm=False)#65x65
        #(w-k)+2p/s + 1 = (65-4)+2/2 +1 = 32
        self.conv2 = conv(conv_depth, conv_depth*2, k_size=4, stride=2, pad=1, batchnorm=True)#32x32
        #(w-k)+2p/s + 1 = (32-4)+2/2 +1 = 16
        self.conv3 = conv(conv_depth*2, conv_depth*4, k_size=4, stride=2, pad=1, batchnorm=True)#16x16
        #(w-k)+2p/s + 1 = (16-4)+2/2 +1 = 8
        self.conv4 = conv(conv_depth*4, conv_depth*8, k_size=4, stride=2, pad=1, batchnorm=True)#8x8
        #(w-k)+2p/s + 1 = (8-4)+2/1 +1 = 4
        self.conv5 = conv(conv_depth*8, out_dim, k_size=4, stride=1, pad=1, batchnorm=False)
    
    def forward(self, input):

        output = F.relu(self.conv1(input))
        output = F.relu(self.conv2(output))
        output = F.relu(self.conv3(output))
        output = F.relu(self.conv4(output))
        #used for classification!
        output = self.conv5(output)

        return output


class ResBlock(nn.Module):
    def __init__(self, conv_dim):
        super().__init__()

        self.conv1 = conv( conv_dim,  conv_dim,  3,  1,  1, True)
        self.conv2 = conv( conv_dim,  conv_dim,  3,  1,  1, True)
    
    def forward(self, input): 

        output = F.relu(self.conv1(input))
        output = input + F.relu(output)
        return output

def conv_transpose(in_, out_, k_size, stride=2, pad=1, batchnorm=True):

    layers=nn.ModuleList()

    layers.append(nn.ConvTranspose2d(in_, out_,k_size, stride, pad))
    if batchnorm: 
        layers.append(nn.BatchNorm2d(out_))
    return nn.Sequential(*layers)

class CycleGenerator(nn.Module):
    def __init__(self, conv_fmap=64, n_resblock=6):
        super().__init__()

        # here we have an encoder, couple of n_resblocks and then a decoder
        # which is a made of several deconv layer(transpose conv layers)
        self.conv1 = conv(3, conv_fmap, 4, 2, 1, False) #64
        self.conv2 = conv(conv_fmap, conv_fmap*2, 4,2,1)#32
        self.conv3 = conv(conv_fmap*2, conv_fmap*4, 4,2,1)#16

        layers=[]
        for i in range(n_resblock):
            layers.append(ResBlock(conv_fmap*4))

        self.resblocks = nn.Sequential(*layers)

        self.deconv1 = conv_transpose(conv_fmap*4, conv_fmap*2, k_size=4)#32
        self.deconv2 = conv_transpose(conv_fmap*2, conv_fmap, k_size=4)#64
        self.deconv3 = conv_transpose(conv_fmap, 3, k_size=4)#128

    def forward(self, input):
        
        #encoder
        output = F.relu(self.conv1(input))
        output = F.relu(self.conv2(output))
        output = F.relu(self.conv3(output))

        output = F.relu(self.resblocks(output))
        #decoder 
        output = F.relu(self.deconv1(output))
        output = F.relu(self.deconv2(output))
        # final image!
        output = F.tanh(self.deconv3(output))
        return output

def create_models(conv_fmap_g=64, conv_fmap_d=64,
                 n_resblocks=6,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):


    G_XtoY = CycleGenerator(conv_fmap=conv_fmap_g,n_resblock=n_resblocks).to(device)
    G_YtoX = CycleGenerator(conv_fmap=conv_fmap_g,n_resblock=n_resblocks).to(device)

    D_X = Discriminator(conv_depth=conv_fmap_d).to(device)
    D_Y = Discriminator(conv_depth=conv_fmap_d).to(device)

    return  G_XtoY, G_YtoX, D_X, D_Y

G_XtoY, G_YtoX, D_X, D_Y = create_models()
print(G_XtoY)
print('-'*40)
print(G_YtoX)
print('\n'+'-'*40)
print(D_X)
print('-'*10)
print(D_Y)




#%%
# Ok, before we go on, lets explain something. 
# we have two networks for each network type. i.e two generators and two discriminators
# what is different here is the way our generators work, previously, we would feed a random
# vector and upsample it until we reach to the image size we desire and then change it so it 
# would resemble our real images. 
# Here, as you can see, our generators accept images rather than a simple vector of random value!
# in fact they are an autoencoder which reconstruct the input image!. What they are doing is actually
# getting an image for winter, down sample it until they reach a feature vector , and then reconstruct it@
# the important thing here is, we feed a winter image, and ask to reconstruct it as if it was a summer image
# our next generator does exactly the opposite, it will recieve a summer picture and reconstruct the winter version
# of it! the dicriminators therefore are for this very task! 
# after the images are reconstructed, we need to know if they are similar/close to the actual real image
# so we have a cyclic loss that checks if a reconstructed image is the same as the real one. 
# apart from that, for our discriminators, we no longer use sigmoid, this time we will be using simle least squared
# error as it is shown to perform better. 
# for exaample for calculating the real loss (label is 1 or close to 1),
# we would do (torch.mean(output_d2 - 1)**2) and 
# for the fake one we would do (torch.mean(output_d - 0)**)
# so in total we will have 3 lossses. lets write them down: 
def real_loss(output_d):
    return torch.mean(output_d - 1 ) **2

def fake_loss(output_d):
    return torch.mean(output_d)**2 

def cyclic_loos(real_image, reconstructed_image, lamda_weight):
    loss = torch.mean(torch.abs(real_image - reconstructed_image) )
    return loss* lamda_weight

#%%
# optimizers 
# since we want to train generators, together, we 
# combine the parameters of these networks together and train them with one optimizer
# we said earlier, that these generators are going to work together, so it makes sense
# the parameters are trained together

lr = 0.0002
beta1 = 0.5
beta2 = 0.999

g_pramas = list(G_XtoY.parameters()) + list(G_YtoX.parameters())
optimizer_g = torch.optim.Adam(g_pramas, lr=lr, betas=[beta1, beta2])

optimizer_d_x = torch.optim.Adam(D_X.parameters(), lr=lr, betas=[beta1, beta2])
optimizer_d_y = torch.optim.Adam(D_Y.parameters(), lr=lr, betas=[beta1, beta2])


#%%
# before goingto the training lets write a save/snapshot function that
# saves our models (generators) to the disk
import os
import pdb
import pickle
import argparse

import warnings
warnings.filterwarnings("ignore")

# import torch
import torch


# numpy & scipy imports
import numpy as np
import scipy
import scipy.misc


def checkpoint(iteration, G_XtoY, G_YtoX, D_X, D_Y, checkpoint_dir='checkpoints_cyclegan'):
    """Saves the parameters of both generators G_YtoX, G_XtoY and discriminators D_X, D_Y.
        """
    G_XtoY_path = os.path.join(checkpoint_dir, 'G_XtoY.pkl')
    G_YtoX_path = os.path.join(checkpoint_dir, 'G_YtoX.pkl')
    D_X_path = os.path.join(checkpoint_dir, 'D_X.pkl')
    D_Y_path = os.path.join(checkpoint_dir, 'D_Y.pkl')
    torch.save(G_XtoY.state_dict(), G_XtoY_path)
    torch.save(G_YtoX.state_dict(), G_YtoX_path)
    torch.save(D_X.state_dict(), D_X_path)
    torch.save(D_Y.state_dict(), D_Y_path)


def merge_images(sources, targets, batch_size=16):
    """Creates a grid consisting of pairs of columns, where the first column in
        each pair contains images source images and the second column in each pair
        contains images generated by the CycleGAN from the corresponding images in
        the first column.
        """
    _, _, h, w = sources.shape
    row = int(np.sqrt(batch_size))
    merged = np.zeros([3, row*h, row*w*2])
    for idx, (s, t) in enumerate(zip(sources, targets)):
        i = idx // row
        j = idx % row
        merged[:, i*h:(i+1)*h, (j*2)*h:(j*2+1)*h] = s
        merged[:, i*h:(i+1)*h, (j*2+1)*h:(j*2+2)*h] = t
    merged = merged.transpose(1, 2, 0)
    return merged
    

def to_data(x):
    """Converts variable to numpy."""
    if torch.cuda.is_available():
        x = x.cpu()
    x = x.data.numpy()
    x = ((x +1)*255 / (2)).astype(np.uint8) # rescale to 0-255
    return x

def save_samples(iteration, fixed_Y, fixed_X, G_YtoX, G_XtoY, batch_size=16, sample_dir='samples_cyclegan'):
    """Saves samples from both generators X->Y and Y->X.
        """
    # move input data to correct device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    fake_X = G_YtoX(fixed_Y.to(device))
    fake_Y = G_XtoY(fixed_X.to(device))
    
    X, fake_X = to_data(fixed_X), to_data(fake_X)
    Y, fake_Y = to_data(fixed_Y), to_data(fake_Y)
    
    merged = merge_images(X, fake_Y, batch_size)

    path = os.path.join(sample_dir, 'sample-{:06d}-X-Y.png'.format(iteration))
    
    scipy.misc.imsave(path, merged)
    print('Saved {}'.format(path))
    
    merged = merge_images(Y, fake_X, batch_size)

    path = os.path.join(sample_dir, 'sample-{:06d}-Y-X.png'.format(iteration))

    scipy.misc.imsave(path, merged)
    print('Saved {}'.format(path))

#%%
# training 
epochs = 8000 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# lets read some test images and use them for testing our network
test_iter_x = iter(test_dataloader_summer)
test_iter_y = iter(test_dataloader_winter)
# we specify a fixed image so we can see how our network is performing
fixed_image_x = next(test_iter_x)[0].to(device)
fixed_image_y = next(test_iter_y)[0].to(device)
#scale them between -1 , 1 
fixed_image_x = scale(fixed_image_x)
fixed_image_y = scale(fixed_image_y)


iter_x = iter(train_dataloader_summer)
iter_y = iter(train_dataloader_winter)

print(f'summer pics number: {len(train_dataloader_summer.dataset)}')
print(f'winter pics number:: {len(train_dataloader_winter.dataset)}')
# in case the length is not the same, use the smaller batchsize
# and see howmany batches we can get for the specified epochs
batch_per_epoch = min(len(iter_x),len(iter_y))

print()

print_interval = 10
snapshot_interval = 100
losses=[]
for e in range(epochs):

    G_XtoY.train()
    G_YtoX.train()
    # this means, if we run out of images, lets start again from the beginnig
    if e%batch_per_epoch == 0:
        iter_x = iter(train_dataloader_summer)
        iter_y = iter(train_dataloader_winter)
        
    Images_X, _ = next(iter_x)
    Images_Y, _ = next(iter_y)

    Images_X = scale(Images_X).to(device)
    Images_Y = scale(Images_Y).to(device)

    if e==0:
        print(Images_Y.shape)

    # D_X, here we are going to make D_X identify fake images from real ones
    # D_X must identify which image is a real X image, and which one is fake (reconstructed)
    # the real loss means, D_X identifies real image_x 
    # and the fake loss means, D_X identifies reconstructed image (using YtoX(image_y)) is 
    # fake x (reconstructed form image_y) 
    output_x = D_X(Images_X)
    real_loss_dx = real_loss(output_x)
    # now we generate an x image and D_X should recognize its fake! 
    fake_x = G_YtoX(Images_Y).to(device)
    if e ==0:
        print(fake_x.shape)
    output_fake_dx = D_X(fake_x)
    fake_loss_dx = fake_loss(output_fake_dx)
    loss_dx = fake_loss_dx + real_loss_dx 

    optimizer_d_x.zero_grad()
    loss_dx.backward()
    optimizer_d_x.step()

    # D_Y, now we will do this the opposite way we work with images_y here but generate x images!
    output_dy = D_Y(Images_Y)
    real_loss_dy = real_loss(output_dy)
    # now generate a Y image using XtoY generator and and X (After all we want to get x and make it look like y)
    # and vice versa!
    fake_y = G_XtoY(Images_X).to(device)
    output_dy = D_Y(fake_y)
    fake_loss_dy = fake_loss(output_dy)
    loss_dy = real_loss_dy + fake_loss_dy

    optimizer_d_y.zero_grad()
    loss_dy.backward()
    optimizer_d_y.step()


    # now its time for the generators to be trained. 
    # we simply feed each generator the oposite image and make them act as if they are real!
    optimizer_g.zero_grad()
    
    g_fake_image_x = G_YtoX(Images_Y).to(device)
    fake_image_output_x = D_X(g_fake_image_x)
    loss_gytox = real_loss(fake_image_output_x)
    #recostruct from fake image 
    reconstructed_y = G_XtoY(g_fake_image_x).to(device)
    cycle_reconstructed_loss_y = cyclic_loos(Images_Y, reconstructed_y, lamda_weight=10)

    # X_image ro begir bego in Y_image e!!
    g_fake_image_y = G_XtoY(Images_X).to(device)
    fake_image_output_y = D_Y(g_fake_image_y)
    loss_gxtoy = real_loss(fake_image_output_y)
    #reconstruct x!
    reconstructed_x = G_YtoX(g_fake_image_y).to(device)
    cycle_reconstructed_loss_x = cyclic_loos(Images_X, reconstructed_x, 10)

    loss_total_g = loss_gxtoy + loss_gytox + cycle_reconstructed_loss_x + cycle_reconstructed_loss_y

    loss_total_g.backward()
    optimizer_g.step()

    if e%print_interval == 0:
        losses.append((loss_dx.item(), loss_dy.item(), loss_total_g.item()))
        print('epochs: [{:5d}/{:5d}]\tloss_dx: {:6.4f}\t loss_dy: {:6.4f}\t loss_g: {:6.4f} '.format(e,
                                                                                              epochs,
                                                                                              loss_dx.item(),
                                                                                              loss_dy.item(),
                                                                                              loss_total_g.item()))

    if e % snapshot_interval == 0:

        G_XtoY.eval()
        G_YtoX.eval()
        with torch.no_grad():
            samples_Y = G_XtoY(fixed_image_x).to(device)
            samples_X = G_YtoX(fixed_image_y).to(device)

            save_samples(e, fixed_image_x, fixed_image_y, G_XtoY, G_YtoX, batch_size=16)



#%%






#%%






#%%





#%%





#%%
