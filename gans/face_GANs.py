#%% [markdown]
# in the name of Allah the most compassionate the most merciful 
# lets create a Face GAN 
#%% [markdown]
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torchvision import datasets, models, transforms
import numpy as np  
import matplotlib.pyplot as plt 
import torchvision
import pickle as pkl
# what to do : 
# 1. create a dataloader
# 2. view a batch of images 
# 3. preprocess them as in rescaling between -1,1
# 4. create a discriminator and a generator 
# 5. create losses for the training 
# 6. train 
# 7. view some new samples 

# 1. create dataloader 
def get_dataloader(root='./processed_celeba_small/celeba/', batch_size=64, resize=(32,32)):
    
    trans = transforms.Compose([transforms.Resize(size=resize),
                                transforms.ToTensor()])    
    dataset = datasets.ImageFolder(root=root, transform=trans)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=2)
    return dataloader

batch_size=64
#folder = 'E:\\DeepLearning\\Pytorch_Udacity_Course\\Codes\\gans\\processed_celeba_small\\600_folder\\'
# G:\Tensorflow_section\dataset\Old_images_bkup\test_gan
#folder = 'G:\\Tensorflow_section\\dataset\\Old_images_bkup\\test_gan'
folder = 'G:\\Tensorflow_section\\dataset\\gantest'
dataloader  = get_dataloader(batch_size=batch_size, resize=(64,64))
print(f'dataset size: {len(dataloader.dataset)}')
imgs, labels = next(iter(dataloader))
#%%
def visualize_images(imgs): 
    fig = plt.figure(figsize=(8,8))
    for i in range(imgs.size(0)):
        ax = fig.add_subplot(8, 8, i+1, xticks=[], yticks=[])
        img = imgs[i].numpy().transpose(1,2,0)
        ax.imshow(img)

visualize_images(imgs)
#preprocess and rescaling 
def rescale(imgs, range_=(-1,1)):
    (min, max) = range_
    rescaled_images = imgs * (max-min) + min 
    return rescaled_images

#check if it works : 
print('before rescaling min, max:', imgs[0].min(), imgs[0].max())
imgs = rescale(imgs)
print('after rescaling min, max:', imgs[0].min(), imgs[0].max())
visualize_images(imgs)


#%%
# seems discriminator is very crucial! the fewer the number of layers the better
# and also the kernel size =4 is the way to do it! both for its easier and also it seems
# to be better!
def conv_batch(in_dim, out_dim, kernel_size, stride, padding, batch_norm=True):
    layers = nn.ModuleList()

    conv = nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias=False)
    layers.append(conv)
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_dim))
    return nn.Sequential(*layers)

class Discriminator(nn.Module):
    def __init__(self, conv_dim=32, act = nn.ReLU(), mode=0):
        super().__init__()
        
        self.mode = mode
        self.conv_dim = conv_dim 
        self.act = act
        self.conv1 = conv_batch(3, conv_dim, 4, 2, 1, False)
        self.conv2 = conv_batch(conv_dim, conv_dim*2, 4, 2, 1)
        self.conv3 = conv_batch(conv_dim*2, conv_dim*4, 4, 2, 1)
        self.conv4 = conv_batch(conv_dim*4, conv_dim*8, 4, 1, 1)
        self.conv5 = conv_batch(conv_dim*8, conv_dim*10, 4, 2, 1)
        self.conv6 = conv_batch(conv_dim*10, conv_dim*10, 3, 1, 1)

        self.fc1 = nn.Linear(64*64*3, 64*10)
        self.fc2 = nn.Linear(64*10, 64*10)
        self.fc3 = nn.Linear(64*10,1)
        
        # self.conv5 = conv_batch(conv_dim*4, conv_dim*5, 3, 2, 1)
        # self.conv6 = conv_batch(conv_dim*5, conv_dim*6, 3, 2, 1)
        self.drp = nn.Dropout(0.5)
        self.fc = nn.Linear(conv_dim*10*3*3, 128) # it seems, larger fmaps prvide better results?!
        self.fc4 = nn.Linear(128,1)

    def forward(self, input):
        if self.mode == 0:
            batch = input.size(0)
            output = self.act(self.conv1(input))
            #print(f'conv1: {output.shape}')
            output = self.act(self.conv2(output))
            #print(f'conv2: {output.shape}')
            # output = F.max_pool2d(output,kernel_size=2)
            output = self.act(self.conv3(output))#16
            #print(f'conv3: {output.shape}')
            output = self.act(self.conv4(output))#16
            #print(f'conv4: {output.shape}')
            output = self.act(self.conv5(output))#16
            #print(f'conv5: {output.shape}')
            output = self.act(self.conv6(output))#16
            #print(f'conv6: {output.shape}')
            # this works with conv_dim*3 in conv1 and doubles of it in the remaining layers.
            # this shows, the larger discriminator works better, but overfitting makes images darker
            # also shallower nets seem to perform better!
            # output = F.dropout2d(self.act(self.conv2(output)),p=0.01)
            # # output = F.max_pool2d(output,kernel_size=2)
            # output = F.dropout2d(self.act(self.conv3(output)),p=0.01)#16
            # output = F.dropout2d(self.act(self.conv4(output)),p=0.1)#16

            # # output = F.max_pool2d(output,kernel_size=2)
            # output = self.act(self.conv5(output))#8
            # # output = F.max_pool2d(output,kernel_size=2)
            # output = self.act(self.conv6(output))#4
            # output = F.max_pool2d(output,kernel_size=2)
            # pred
            #print(output.shape)
            output = output.view(batch, self.fc.in_features)
            output = self.fc(output)
            output = self.drp(output)
            output = self.fc4(output)
        else:
            output = input.view(-1, 64*64*3)
            output = self.drp(self.fc1(output))
            output = self.drp(self.fc2(output))
            output = self.drp(self.fc3(output))
        return output

def deconv_convtranspose(in_dim, out_dim, kernel_size, stride, padding, batchnorm=True):

    layers = []
    deconv = nn.ConvTranspose2d(in_dim, out_dim, kernel_size = kernel_size, stride=stride, padding=padding)
    layers.append(deconv)
    if batchnorm:
        layers.append(nn.BatchNorm2d(out_dim))

    return nn.Sequential(*layers)

class Generator(nn.Module):
    def __init__(self, z_size=100, conv_dim=32, mode=0): 
        super().__init__()
        self.conv_dim = conv_dim
        #make the 1d input into a 3d output of shape (conv_dim*4, 4, 4 )
        self.fc = nn.Linear(z_size, z_size*4)#4x4
        self.fc2 = nn.Linear(z_size*4, conv_dim*4*4*4)#4x4
        
        # conv and deconv layer work on 3d volumes, so we now only need to pass the number of fmaps and not the
        # input volume size (its h,w which is 4x4!)
        self.mode = mode
        self.drp = nn.Dropout(0.5)
        if mode == 0: 
            self.deconv1 = deconv_convtranspose(conv_dim*4, conv_dim*3, kernel_size=3, stride=2, padding=1)#7x7
            self.deconv2 = deconv_convtranspose(conv_dim*3, conv_dim*2, kernel_size=3, stride=2, padding=1)#13x13
            self.deconv3 = deconv_convtranspose(conv_dim*2, conv_dim, kernel_size=3, stride=2, padding=1)#25x25
            self.deconv4 = deconv_convtranspose(conv_dim, conv_dim, kernel_size=4, stride=1, padding=0)#16x16
            self.deconv5 = deconv_convtranspose(conv_dim, 3, kernel_size=5, stride=1, padding=0,batchnorm=False)
        elif mode ==1:
            self.deconv1 = deconv_convtranspose(conv_dim*4, conv_dim*3, kernel_size =4, stride=2, padding=1)#10x10
            self.deconv2 = deconv_convtranspose(conv_dim*3, conv_dim*2, kernel_size =4, stride=2, padding=1)#20x20
            self.deconv3 = deconv_convtranspose(conv_dim*2, conv_dim, kernel_size =4, stride=2, padding=1)#40x40
            self.deconv4 = deconv_convtranspose(conv_dim, conv_dim, kernel_size =3, stride=2, padding=1)#63x63
            self.deconv5 = deconv_convtranspose(conv_dim, 3, kernel_size =4, stride=1, padding=1, batchnorm=False)#64x64
        elif mode==2:
            self.deconv1 = deconv_convtranspose(conv_dim*4, conv_dim*2, kernel_size =4, stride=2, padding=1)#7x7
            self.deconv2 = deconv_convtranspose(conv_dim*2, conv_dim, kernel_size =4, stride=2, padding=1)#14x14
            # self.deconv3 = deconv_convtranspose(conv_dim*2, conv_dim, kernel_size =4, stride=1, padding=1)#15x15
            self.deconv4 = deconv_convtranspose(conv_dim, conv_dim, kernel_size =4, stride=2, padding=1)#16x16
            self.deconv5 = deconv_convtranspose(conv_dim, 3, kernel_size =4, stride=2, padding=1, batchnorm=False)#32x32

    def forward(self, input):
        output = self.fc(input)
        output = self.drp(output)
        output = self.fc2(output)
        output = self.drp(output)
        
        output = output.view(-1, self.conv_dim*4, 4, 4)
        # print(output.shape)
        if self.mode != 2: 
            output = F.relu(self.deconv1(output))
            #print(f'd1: {output.shape}')
            output = F.relu(self.deconv2(output))
            #print(f'd2: {output.shape}')
            output = F.relu(self.deconv3(output))
            #print(f'd3: {output.shape}')
            output = F.relu(self.deconv4(output))
            #print(f'd4: {output.shape}')
            # we create the image using tanh!
            output = F.tanh(self.deconv5(output))
            #print(f'd5: {output.shape}')
        else:
            output = F.relu(self.deconv1(output))
            #print(f'd1: {output.shape}')
            output = F.relu(self.deconv2(output))
            output = F.relu(self.deconv4(output))
            #print(f'd2: {output.shape}')
            # we create the image using tanh!
            output = F.tanh(self.deconv5(output))
            
        return output


dd = Discriminator(mode=0)
zd = np.random.rand(2,3,64,64)
zd = torch.from_numpy(zd).float()
# print(dd)
print(dd(zd).shape)

gg = Generator(mode=1)
z = np.random.uniform(-1,1,size=(2,100))
z = torch.from_numpy(z).float()
print(gg(z).shape)

#%%
# def conv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
#     """Creates a convolutional layer, with optional batch normalization.
#     """
#     layers = []
#     conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
#                            kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
    
#     layers.append(conv_layer)

#     if batch_norm:
#         layers.append(nn.BatchNorm2d(out_channels))
#     return nn.Sequential(*layers)
# class Discriminator(nn.Module):

#     def __init__(self, conv_dim=32):
#         """
#         Initialize the Discriminator Module
#         :param conv_dim: The depth of the first convolutional layer
#         """
#         super(Discriminator, self).__init__()

#         # complete init function
#         self.conv_dim = conv_dim

#         # Define all convolutional layers
#         # Should accept an RGB image as input and output a single value
#         self.conv1 = conv(3, conv_dim, 4, batch_norm=False) # x, y = 64 depth = 3
#         self.conv2 = conv(conv_dim, conv_dim * 2, 4) # x, y = 32 depth = 64
#         self.conv3 = conv(conv_dim * 2, conv_dim * 4, 4) # x, y = 16 depth = 128
        
#         self.fc = nn.Linear(conv_dim*4*4*4, 1)
#         self.out = nn.Sigmoid()
#         self.dropout = nn.Dropout(0.5)
        

#     def forward(self, x):
#         """
#         Forward propagation of the neural network
#         :param x: The input to the neural network     
#         :return: Discriminator logits; the output of the neural network
#         """
#         # define feedforward behavior
#         x = F.leaky_relu(self.conv1(x), 0.2)
#         # x = self.dropout(x)
#         x = F.leaky_relu(self.conv2(x), 0.2)
#         # x = self.dropout(x)
#         x = F.leaky_relu(self.conv3(x), 0.2)
#         # x = self.dropout(x)
        
#         x = x.view(-1, self.conv_dim*4*4*4)
        
#         x = self.fc(x)
#         x = self.dropout(x)
        
#         # x = self.out(x)
        
#         return x
# def deconv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
#     """Creates a transpose convolutional layer, with optional batch normalization.
#     """
#     layers = []
#     # append transpose conv layer
#     layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False))
#     # optional batch norm layer
#     if batch_norm:
#         layers.append(nn.BatchNorm2d(out_channels))
#     return nn.Sequential(*layers)
# class Generator(nn.Module):
    
#     def __init__(self, z_size=100, conv_dim=32):
#         """
#         Initialize the Generator Module
#         :param z_size: The length of the input latent vector, z
#         :param conv_dim: The depth of the inputs to the *last* transpose convolutional layer
#         """
#         super(Generator, self).__init__()

#         # complete init function
#         self.conv_dim = conv_dim
        
#         self.fc = nn.Linear(z_size, conv_dim*4*4*4)
        
#         self.t_conv1 = deconv(conv_dim*4, conv_dim*2, 4 )
#         self.t_conv2 = deconv(conv_dim*2, conv_dim, 4)
#         self.t_conv3 = deconv(conv_dim, 3, 4, batch_norm=False)
#         self.dropout = nn.Dropout(0.5)
        

#     def forward(self, x):
#         """
#         Forward propagation of the neural network
#         :param x: The input to the neural network     
#         :return: A 32x32x3 Tensor image as output
#         """
#         # define feedforward behavior
        
#         x = self.fc(x)
#         x = self.dropout(x)
        
#         x = x.view(-1, self.conv_dim*4, 4, 4)
        
#         x = F.relu(self.t_conv1(x))
#         # x = self.dropout(x)
#         x = F.relu(self.t_conv2(x))
#         # x = self.dropout(x)
#         x = F.tanh(self.t_conv3(x))
        
#         return x
#%%    
# gg = Generator(mode=2)
# z = np.random.uniform(-1,1,size=(2,100))
# z = torch.from_numpy(z).float()
# print(gg(z).shape)
# dd = Discriminator(mode=0)
# zd = np.random.rand(2,64,64,3)
# zd = torch.from_numpy(zd).float()
# print(dd(zd).shape)
#%%

def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight.data,0.0, std=0.02)



#%% 
# create losses 
# these are based on LSGAN paper
def real_loss(output_, smooth, device):
    batch_size = output_.size(0)
    if smooth: 
        labels = torch.ones(batch_size)*0.9
    else: 
        labels = torch.ones(batch_size)
    labels = labels.to(device)

    criterion = nn.BCEWithLogitsLoss()
    return criterion(output_.squeeze(), labels.squeeze())

def fake_loss(output_, smooth, device):
    batch_size = output_.size(0)
    if smooth: 
        (a,b) = (0, 0.3)
        labels = torch.ones(batch_size) * (b - a ) * np.random.random_sample() + a
    else: 
        labels = torch.zeros(batch_size)
    labels = labels.to(device)
    criterion = nn.BCEWithLogitsLoss()
    return criterion(output_.squeeze(), labels)

#%% 
# 1. disable bias!
# 2. initialize model weights manually normal distribution 
#    with mean = 0, std dev = 0.02.
# 2. use kernel_sizes of 4 ! and see the result  (done, seems a bit lower, but not that much!)
# 3. upsample rapiddly (z_size*4*8*8 instead of z_size*4*2*2) forexample. z_size means
#    the input vector with size_z which we feed to our fc layer at the beginning
#  4.use leaky relu in discriminator
#  5. use deeper architectures
#  6. use shallower architectures  

#%% 
# now training 
# create models first 
conv_dim = 64
z_size = 100
batch_size = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

D = Discriminator(conv_dim=conv_dim,act=nn.LeakyReLU(0.2),mode=0)
G = Generator(z_size=z_size, conv_dim=conv_dim,mode=1)
print(D)
print(G)

D.apply(init_weights)
G.apply(init_weights)

# for probability density read this : http://www.engineeredsoftware.com/nasa/density.htm
# https://www.reddit.com/r/explainlikeimfive/comments/cjp9d4/eli5_what_is_a_probability_density/
# https://www.youtube.com/watch?v=1xQ4r2gcW3c

# Generative Adversarial Networks (GAN) [8] are com-posed of two models
# that are alternatively trained to com-pete with each other. The generator G
# is optimized to re-produce the true data distribution 'pdata' by generating 
# images that are difficult for the discriminator D to differentiate from real
# images.

# Meanwhile, D is optimized to distinguish real images and synthetic images 
# generated by G. Overall,the training procedure is similar to a two-player
# min-maxgame with the following objective function,
# min_G_max_D_V(D, G)=E_x∼pdata[logD(x)] + E_z∼pz[log(1−D(G(z)))],(1)
# where x is a real image from the true data distribution pdata,and z is a noise
# vector sampled from distribution pz (e.g.,uniform or Gaussian distribution).
# Conditional GAN [7,19]
# is an extension of GAN where both the generator and discriminator
# receive additional conditioning variables 'c', yielding G(z,c) and D(x, c). 
# This formulation allows G to generate images conditioned on variables c

# https://lilianweng.github.io/lil-log/2017/08/20/from-GAN-to-WGAN.html
# As a result, training a GAN faces a dilemma:
#     If the discriminator behaves badly, the generator does not have accurate feedback and the loss function cannot represent the reality.
#     If the discriminator does a great job, the gradient of the loss function drops down to close to zero and the learning becomes super slow or even jammed.
# This dilemma clearly is capable to make the GAN training very tough.

# ref https://medium.com/@jonathan_hui/gan-a-comprehensive-review-into-the-gangsters-of-gans-part-1-95ff52455672
# GANs Problems
# Training GAN is not easy. GAN models may suffer the following problems:
#     Mode collapse: the generator produces limited varieties of samples,
#     Diminished gradient: the discriminator gets too successful that the gradients vanish and the generator learns nothing,
#     Non-convergence: the model parameters oscillate, destabilize and never converge,
#     Unbalance between the generator and discriminator causes overfitting, and
#     Highly sensitive to hyperparameters.
# Mode
# Mode collapses when generated images converge to the same image (the same optimal point).
# Source
# Full mode collapse is not common. But partial collapse happens often. About half of the images below have one similar image.
# Non-convergence
# GAN is a game where your opponent always counteracts your actions. The optimal solution is known as Nash equilibrium which is hard to find. Gradient descent is not necessarily a stable method for finding such equilibrium. When mode collapses, the training turns into a cat-and-mouse game in which the model will never converge. Just another thought, maybe the nature of the game makes GANs hard to converge.

# Other problems
# The non-convergence and mode collapse is often interpreted as an imbalance between the discriminator and the generator. The discriminator may overwhelm the other (or vice versa). There are many attempts at addressing the problem but not much progress has been made in the first few years. Some researchers believe that this is not a feasible or a desirable goal since a good discriminator gives good feedback. However, some progress have been made lately with a more dynamic scheme in balancing their training.

# Measurement
# GAN’s objective functions measure the competition between the generator and the discriminator. However, these metrics do not reflect the image quality and not suitable for model comparison, progress monitor and performance tuning. In the figure below, the generator cost increases even the image quality improves.

# In early research, we compare model results visually which are strongly biased. Many “state-of-the-art” claims from early research papers are hard to verify or overstated. To address that, Inception Score (IS) is developed to measure the image quality and the diversity. If we can label generated images correctly while the generated images are evenly distributed among different object classes, we give them a high IS score.

# Fréchet Inception Distance (FID) measures the statistical difference between the features of the real and generated images extracted by an Inception network. A low FID distance indicates the generated images are natural with similar diversity as the real images. To learn more about the definitions of IS and FID, and their weakness, we provide another article in measuring GAN performance.




# How do StackGans work? (those that create images from text?)
# Conditioning AugmentationAs shown in Figure2, the text descriptiont is 
# first encoded by an encoder, yielding a text embedding 't'.  In previous
# works [26,24], the text embedding is nonlinearly transformed to generate
# conditioning latent variables as the input of the generator.
# However, latent space for the text embedding is usually high dimensional 
# (>100dimen-sions). With limited amount of data, it usually causes dis-continuity
# in the latent data manifold, which is not desirable for learning the generator.
# To mitigate this problem, we introduce a Conditioning Augmentation technique to
# produce additional conditioning variables 'c'. In contrast to the 
# fixed conditioning text variable 'c' in [26,24], we randomly sample the 
# latent variables ˆc from an independent Gaussian distribution N(μ(φt),Σ(φt)),
# where the mean μ(φt) and diagonal covariance matrix Σ(φt) are functions of the
# text embedding φt. The proposed Conditioning Augmentation yields more training pairs
# given a small number of image-text pairs, and thus encourages robustness to small 
# perturbations along the conditioning manifold. To further enforce the smoothness over
# the conditioning manifold and avoid overfitting [6,14], we add the following 
# regularization term to the objective of the generator during training
#             D_KL(N(μ(φt),Σ(φt))|| N(0,I))    ,(2)
# which is the Kullback-Leibler divergence (KL divergence) between the standard Gaussian 
# distribution and the conditioning Gaussian distribution. The randomness introduced in
# the Conditioning Augmentation is beneficial for modeling text to image translation as
# the same sentence usually corresponds to objects with various poses and appearances.

#Important Note, KL divergence is not a good candidate for distribution comparisions,
# the FE or inception score or something like that is being used now! read WGAN or WGAN-GP 
# paper for more information! 

#https://ajolicoeur.wordpress.com/cats/ 

# I experimented with generating faces of cats using Generative adversarial networks (GAN).
# I wanted to try DCGAN, WGAN and WGAN-GP in low and higher resolutions.
# I used the CAT dataset (yes this is a real thing!) for my training sample.
# This dataset has 10k pictures of cats. I centered the images on the kitty
# faces and I removed outliers (I did this from visual inspection, 
# it took a couple of hours…). I ended up with 9304  images bigger 
# than 64 x 64 and 6445 images bigger than 128 x 128.

# DCGAN
# The DCGAN generator converges to very realistic pictures in about 2-3 hours with only 209 
# epochs but some mild tweaking is necessary for proper convergence. You must choose separate
# learning rates for D and G so that neither G or D become way better than the other,
# it’s a very careful balance but once you got it, you’re set for convergence!
# With 64 x 64 images, the sweet spot is using .00005 for the Discriminator learning rate
# and .0002 for the Generator learning rate. There’s no apparent mode collapse and we end up
# with really cute pictures!
#All my initial attempts at generating cats in 128 x 128 with DCGAN failed. 
# However, simply by replacing the batch normalizations and ReLUs with SELUs,
# I was able to get slow (6+ hours) but steady convergence with the same learning rates as before.
# SELUs are self-normalizing and thus remove the need for batch normalization.
# SELUs are extremely new so very little research has been done on SELUs with GANs but from what 
# I observed, they seem to greatly increase GANs stability. The cats are not as good looking as
# the previous ones and there is a noticeable lack of variety (lots of black cats with similar faces).
# This is mostly explained by the fact that the sample size is N=6445 rather than N=9304 
# (I only trained the models on images bigger than 128×128). Still, some cats are pretty 
# good looking and they are in higher resolution than before so I still consider this a success!

# WGAN
# ref: https://medium.com/@jonathan_hui/gan-wasserstein-gan-wgan-gp-6a1a2aa1b490
# The WGAN generator converges very slowly (took 4-5h, 600+ epochs) and only when using 64 hidden nodes.
# I could not make the generator converge with 128 hidden nodes. With DCGAN, you have to tweak the learning rates
# a lot but you are able to see quickly if it’s not going to converge (If Loss of D goes to 0 or if loss of G goes 
# to 0 at the start) but with WGAN, you need to let it run for many epochs before you can tell.
# Visually, there is some pretty striking mode collapse here; many cats have heterochromia,
# one eye closed and one eye open or a weird nose. Overall the results are not as impressive as
# with DCGAN but then it could be because the neural networks are less complex so this might 
# not be a fair comparison. It also seems to get stuck into a local optimum. So far, WGAN is disappointing.
# WGAN-GP (An improved version of WGAN with regularization instead of weight clipping) might be able 
# to deal with these issues. In the paper by Gulrajani et al. (2017), they were able to train a 101
# layers neural network to produce pictures! So I doubt that training a cat generator with 5 layers
# and 128 hidden nodes would be much of a problem. The Adam optimizer also has some properties which
# lower the risk of mode collapse and getting stuck into a bad local optimum (ref: https://arxiv.org/abs/1706.08500).
# This is likely contributing to the problem with WGAN because it doesn’t use Adam
# while DCGAN and WGAN-GP both use it. 
# basically in WGAN, and WGAN-GP, the loss functions are changed. for WGAN, in our discriminator we 
# no longer use an activation function (sigmoid e.g), and simply clip the discriminators weights. the
# problem with DCGAN and normal GAN, is that, when D works optimally but the generator network is simply
# far off! the gradients will dimnishe and generator will never learn. 
# From the ref: 
# "In practice, GAN can optimize the discriminator easier than the generator. 
# Minimizing the GAN objective function with an optimal discriminator is equivalent to 
# minimizing the JS-divergence (proof). As illustrated above, if the generated image has 
# distribution q far away from the ground truth p, the generator barely learns anything"
# An optimal discriminator produces good information for the generator to improve. 
# But if the generator is not doing a good job yet, the gradient for the generator diminishes 
# and the generator learns nothing (the same conclusion we just explain).

# Calculating Wasserstein distance:
# So to calculate the Wasserstein distance, we just need to find a 1-Lipschitz function. 
# Like other deep learning problem, we can build a deep network to learn it. 
# Indeed, this network is very similar to the discriminator D, 
# just without the sigmoid function and outputs a scalar score rather than a probability.
# This score can be interpreted as how real the input images are. 
# In reinforcement learning, we call it the value function which measures 
# how good a state (the input) is. We rename the discriminator to critic to reflect its new role.

# Loss : 
# Correlation between loss metric and image quality

# Also In GAN, the loss measures how well it fools the discriminator rather than a measure of the image quality. 
# As shown below, the generator loss in GAN does not drop even the image quality improves. 
# Hence, we cannot tell the progress from its value. We need to save the testing images and 
# evaluate it visually. 
# On the contrary, WGAN loss function reflects the image quality which is more desirable.

# Good points of WGAN against GAN : (no model collapse, generator learns well)
# Improve training stability:
# Two significant contributions for WGAN are
#     it has no sign of mode collapse in experiments, and
#     the generator can still learn when the critic perform well.
# As shown below, even though we remove the batch normalization in DCGAN, WGAN can still perform.

# (for WGAN, it may need to train the discriminator more than the generator. )
# 
# The WGAN-GP is the same network, with the difference that weght clipping is dismissed becasue
# it reduces the model capability fit properly the complex data. instead they impose/enforce the 1-Lipschitz 
# constraint using the gradients. 

# "The weight clipping behaves as a weight regulation. 
# It reduces the capacity of the model f and limits the capability to model complex functions. 
# In the experiment below, the first row is the contour plot of the value function estimated by WGAN. 
# The second row is estimated by a variant of WGAN called WGAN-GP. 
# The reduced capacity of WGAN fails to create a complex boundary to surround the 
# modes (orange dots) of the model while the improved WGAN-GP can."

# enforce the Lipschitz constraint
# Quote from the research paper: Weight clipping is a clearly terrible way to enforce a Lipschitz constraint. 
# If the clipping parameter is large, then it can take a long time for any weights to reach their limit, 
# thereby making it harder to train the critic till optimality. 
# If the clipping is small, this can easily lead to vanishing gradients when the number of layers is big, 
# or batch normalization is not used (such as in RNNs) … 
# and we stuck with weight clipping due to its simplicity and already good performance.

# The difficulty in WGAN is to enforce the Lipschitz constraint. 
# Clipping is simple but it introduces some problems. 
# The model may still produce poor quality images and does not converge, 
# in particular when the hyperparameter c is not tuned correctly.
# The model performance is very sensitive to this hyperparameter. 
# In the diagram below, when batch normalization is off, 
# the discriminator moves from diminishing gradients to exploding gradients when c increases from 0.01 to 0.1.
# w = w.clip(-c,c)

# Also batchnormalization in"DISCRIMINATOR" is dismissed in WGAN-GP:
# Batch normalization is avoided for the critic (discriminator). 
# Batch normalization creates correlation between samples in the same batch. 
# It impacts the effectiveness of the gradient penalty which is confirmed by experiments.
# By design or not, some new cost functions add gradient penalty to the cost function. 
# Some is purely based on empirical observation that models misbehaves when the gradient increases. 
# However, gradient penalty adds computational complexity that may not be 
# desirable but it does produce some higher-quality images.

# WGAN-GP (Improved WGAN)
# The WGAN-GP generator converges very slowly (more than 6+ hours) but it does so with pretty much any settings.
# It’s working directly out-of-the-box without any tweaking necessary. You can increase or decrease the learning
# rate by a lot without causing many problems. So for this, WGAN-GP really has my appreciation.
# The cats are very diverse and there is no apparent mode collapse so this is a major improvement on WGAN.
# On the other hand, the cats are very blurry looking, kind of as if you were looking at up-scaled versions
# of low resolutions pictures and I’m not sure why that is. This might be a peculiarity with the Wasserstein loss.
# I assume that using different learning rates and architectures would help. Further attempts at this need to be made,
# it certainly has a lot of potential.

# LSGAN (Least Squares GAN)
# LSGAN is a slightly different approach where we try to minimize the squared distance between the Discrimination 
# output and its assigned label; they recommend using: 1 for real images, 0 for fake images in Discriminator update
# and then 1 for fake images in Generator update.
# A paper by Hejlm et al. (2017) suggests using instead: 1 for real images, 0 for fake images in Discriminator update
# but .50 for fake images in Generator update to seek the boundary instead.
# I didn’t have the time to make some full runs with it yet but it seems to be quite stable overall and to output
# nice looking cats. Although it is generally stable, one time, the loss and gradients exploded and things went
# from cats to nonsense. You can see epoch 31 and 32:
# So it’s not completely stable, it can break down really bad. 
# Choosing better hyper-parameters for the Adam optimizer would help prevent that.
# You don’t need to tweak the learning rate as with DCGAN though and when it doesn’t break down
# (this might be rare), it seems to lead to good-looking cats.
# Edit: The first author of LSGAN, Xudong Mao, sent me an example of LSGAN generating cats in 
# 128×128 which shows that this approach can create reasonably good samples.
#  You can see their results here:https://github.com/AlexiaJM/Deep-learning-with-cats
# https://ajolicoeur.wordpress.com/RelativisticGAN/


# a great exlanation concernng different GANs , read it all 
# https://www.gwern.net/Faces
# https://www.gwern.net/Faces#why-dont-gans-work

# Why Don’t GANs Work?
# Why does StyleGAN work so well on anime images while other GANs worked not at all or slowly at best?
# The lesson I took from “"Are GANs Created Equal? A Large-Scale Study"”, Lucic et al 2017, is that 
# CelebA/CIFAR10 are too easy, as almost all evaluated GAN architectures were capable of occasionally
# achieving good FID if one simply did enough iterations & hyperparameter tuning.
# Interestingly, I consistently observe in training all GANs on anime that clear lines & sharpness & 
# cel-like smooth gradients appear only toward the end of training, after typically initially blurry 
# textures have coalesced. This suggest an inherent bias of CNNs: color images work because they provide some
# degree of textures to start with, but lineart/monochrome stuff fails because the GAN optimization dynamics
# flail around. This is consistent with Geirhos et al 2018’s findings—which uses style transfer to construct 
# a data-augmented/transformed “"Stylized-ImageNet"”—showing that ImageNet CNNs are lazy and, because the tasks
# can be achieved to some degree with texture-only classification (as demonstrated by several of Geirhos et al 2018’s
# authors via “"BagNets"”), focus on textures unless otherwise forced. So while CNNs can learn sharp lines & shapes
#  rather than textures, the typical GAN architecture & training algorithm do not make it easy. Since CIFAR10/CelebA
#  can be fairly described as being just as heavy on textures as ImageNet (which is not true of anime images),
#  it is not surprising that GANs train easily on them starting with textures and gradually refining into good 
# samples but then struggle on anime.
# This raises a question of whether the StyleGAN architecture is necessary and whether many GANs might work, 
# if only one had good style transfer for anime images and could, to defeat the texture bias, generate many 
# versions of each anime image which kept the shape while changing the color palette? (Current style transfer
# methods like the AdaIN PyTorch implementation used by Geirhos et al 2018, do not work well on anime images, 
# ironically enough, because they are trained on photographic images, typically using the old VGG model.)

# https://github.com/tdrussell/IllustrationGAN
# The model is based on DCGANs, but with a few important differences:
#  No strided convolutions. The generator uses bilinear upsampling to upscale a feature blob
#       by a factor of 2, followed by a stride-1 convolution layer. The discriminator uses a stride-1
#       convolution followed by 2x2 max pooling.
#  Minibatch discrimination. See Improved Techniques for Training GANs for more details.
#  More fully connected layers in both the generator and discriminator. In DCGANs, both networks 
#       have only one fully connected layer.
#  A novel regularization term applied to the generator network. Normally, increasing the number 
#       of fully connected layers in the generator beyond one, triggers one of the most common failure 
#       modes when training GANs: "the generator "collapses" the z-space and produces only a very small
#       number of unique examples."
#       In other words, very different z vectors will produce nearly the same generated image. To fix this,
#       I add a small 'auxiliary' z-predictor network that takes as input the output of the last fully
#       connected layer in the generator, and predicts the value of z.
#       In other words, it attempts to learn the inverse of whatever function the generator fully connected
#       layers learn. The z-predictor network and generator are trained together to predict the value of z.
#       This forces the generator fully connected layers to only learn those transformations that preserve
#       information about z. The result is that the aformentioned collapse no longer occurs,
#       and the generator is able to leverage the power of the additional fully connected layers.


# https://arxiv.org/pdf/1905.01164.pdf

lr_d = 0.005 #0.002
lr_g = 0.004
# optimizer_d = torch.optim.Adam(D.parameters(), lr = lr)
# optimizer_g = torch.optim.Adam(G.parameters(), lr = lr)

beta1=0.1 #momentum
beta2=0.5 # default value
optimizer_d = torch.optim.Adam(D.parameters(), lr_d , [beta1, beta2])
optimizer_g = torch.optim.Adam(G.parameters(), lr_g , [beta1, beta2])


D = D.to(device)
G = G.to(device)
# create a fixed_vector for evaluating our network performance 
z_vec_fixed = np.random.uniform(-1, 1, size=(batch_size, z_size))
# it has to be float! rememer that!
z_vec_tensor_fixed = torch.from_numpy(z_vec_fixed).float().to(device)

# or simply do 
#z_vec_tensor_fix = torch.Tensor(size=(batch_size, z_size)).uniform_(-1,1).float().to(device)

epochs = 30
losses =[]
samples = []
interval= 500

with_detach=False

print(f'data loader length : {len(dataloader.dataset)}')
i=0
for e in range(epochs): 

    D.train()
    G.train()
    for imgs, _ in dataloader:
        i+=1
        imgs = imgs.to(device)
        #rescale images 
        imgs = rescale(imgs)
       
        # feed the discriminator 
        output_score_real = D(imgs)
        loss_real_d = real_loss(output_score_real, True, device)
        # now generate a new image and calculate the fake loss
        batch_size = imgs.size(0)
        z_vec = np.random.uniform(-1,1,size=(batch_size,z_size))
        z_tensor = torch.from_numpy(z_vec).float().to(device)
        fake_img = G(z_tensor)
        # print('fake',fake_img.shape)
        output_score_fake = D(fake_img)
        loss_fake_d = fake_loss(output_score_fake, False, device)
        loss_d = loss_real_d + loss_fake_d 

        optimizer_d.zero_grad()
        loss_d.backward()
        optimizer_d.step()

        # now the generator  
        # create a new image 
        z_vec = np.random.uniform(-1,1,size=(batch_size,z_size))
        z_tensor = torch.from_numpy(z_vec).float().to(device)
        
        fake_img = G(z_tensor)

        if i==1:
            print(f'g: {fake_img.shape}')
        #lets classify by D 
        output_score_fake_tolooklikereal = D(fake_img)
        # now lets make D think it was real, 
        loss_g = real_loss(output_score_fake_tolooklikereal, True, device)

        optimizer_g.zero_grad()
        loss_g.backward()
        optimizer_g.step()

        if i % interval ==0 : 
            print(f'epoch {e}/{epochs} iter: {i} loss_d: {loss_d.item()} loss_g: {loss_g.item()} ')
            break

    losses.append((loss_d.item(), loss_g.item()))
    with torch.no_grad():
        G.eval()
        sample = G(z_vec_tensor_fixed)
        samples.append(sample)

#save the model to the disk    
with open(f'face_gan_{epochs}_{z_size}_{conv_dim}.t', 'wb') as f : 
    states = {"epochs":epochs,
             "z_size":z_size,
             "conv_dim": conv_dim, 
             "states_g": G.state_dict(),
             "lr":lr_g}    
    torch.save(states, f)    
#%%    
#sampe images as pickle files 
with open(f'face_imgs_gan_{epochs}_{z_size}_{conv_dim}.pkl','wb' ) as f: 
    pkl.dump(samples, f)
    
#%% 
losses = np.array(losses)
plt.plot(losses.T[0],label='blue_d')
plt.plot(losses.T[1],label='green_g')
plt.legend()
plt.show()


#%%
# lets visualize our samples in each epoch
def vis_samples(samples, title):
    fig, axes = plt.subplots(4, 4, sharex=True, sharey=True)
    samples = samples.cpu().detach().numpy().transpose(0,2,3,1)
    
    for ax, img in zip(axes.flatten(), samples) :
        img = ((img +1)*255 / (2)).astype(np.uint8) # rescale to pixel range (0-255)
        #print(img.shape)
        ax.imshow(img.reshape((64,64,3)))
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False) 
        ax.set_title(title)

# with open(f'face_imgs_gan_{epochs}_{z_size}_{conv_dim}.pkl','rb') as f: 
    # samples = pkl.load(f)

for i in range(len(samples)):    
    vis_samples(samples[i],str(i))
#%%
# morphing test
def morph_sample(G, z_vec_numpy, count=10, row =2, col=10 , 
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    
    fig = plt.figure(figsize=(col,row))
    plt.subplots_adjust(wspace=0,hspace=0)
    for i in range(count):
        #what we subtract or add here, to the latent sapce
        # directly results in new image. for example 0.5 seems to add the glasses
        # and the multiplier factor seems to rotate the image to sides!!! im not sure!
        # if we add or subtract vectors, I guess we can get much better results! this means embeddingsin images!
        z = z_vec_numpy + 0.5 * (i*0.2)
        z = torch.from_numpy(z).float().to(device)
        imgs = G(z)
        imgs = imgs.detach().to('cpu').numpy().squeeze().transpose(1,2,0)
        # ((imgs - min)*255 // (max-min))
        imgs = ((imgs +1)*255 /2 ).astype(np.uint8)
        ax = fig.add_subplot(row,col, i+1, xticks=[], yticks=[])
        ax.imshow(imgs)

        
        #plt.show()
z = np.random.uniform(-1,1, size=(1,100))
G.eval()
morph_sample(G, z, count=40, row=4, col=10)
#%%
# Observations : 
# The discriminator is very very important . the generator loss is actually the discriminator loss
# you can decrease it by only using a fully connected network as discriminator, however, when you use
# a fully connected network for discriminator, you'll noticed the generator loss will decrease, but the
# discriminator loss will go up! and when you visualize the images, you'll notice that the images 
# are actually mean of the dataset! So one must craft discriminator network with utmost care
#  it seems, creating a shallow discriminator provides the best results. 3 or mostly 4 layers! 
# also the use of kernel_size of 4 just makes life easier, as it needs less layers to downsample
# from any image size. the use of LeakyReLU is not mandetory and it seems always the ReLU provides better results
# the use of dropout also is imortant. if the data overfits, the images will get darker, as the overfitting continues
# we can see the image clarity decreases and goes towards darkness, earlier epochs create more vivid images and as
# training goes on, the images, get darker and darker (though the output looks fine in later epochs, only darker!)
# the larger the discriminator the better the results look! 
# leaky relu in descriminator seems to be fine. at least better than relu. we can use rmsprop as well!
# all in all DCGAN is not good and we have other variants such as WGAN and prograssive GANs as well that we should
# also check! generator. a deeper generator decreased error and provided better resuts
# the significance of z_size is that thats exactly the embedding. later on when the network is trained
# by meddling with different number of the input latent vector, we can change different features. so 
# bigger latent space means more features!
# note 2: if we use detach on G()'s output, we will noticed that the generators loss will not decrease like before
# it will be as high as 8 at the beginning and starto to increase to like around 11 and 12 and stay there. 
# meanwhile, the discriminator's loss will be as low as 0.3!! in the beginning and not change much, BUT!
# the images generated by the generator, look just like complete noise! while before
# i.e. not using detach after G(z), they generated good images
# I could train with 300 images, but the quality was very poor, and they were not diverse at all.
#  600 was much better but still was in adequate. faces had more pronounced details and were more diverese 
# it was not enough though but at the very minimum I generated some images. images looked like 
# scketechs and were not clear like images, there were lots of distortions in colors  

#%%


#%%


#%%
