#%% 
# بسم الله الرحمن الرحیم 
# lets create a gan, or a generative adversarial network 
# GAN by itself refers to a network comprised of fc layers
# in  this configuration, the use of leaky relu in descrtiminator is
# very important . 
# lets begin 
# we will first download and use the mnist dataset 
# we will create t wo networks, D and G for discriminator and Generator respectively 
# the D network is simply a normal network that does the classification 
# the G network however, starts with a randomly initialized vector of some length and gradually expands at each layer until it reconstructs the image
# we create two optimizers for each network. 
# we use adam for generator, and sgd for discriminator (we can adam fo rboth!) 
# remember  to scale/normalize the input image between 1 and -1 and use tanh at the last layer of generetor 
# create two losses for real labels and fake labels, you can do smoothing as well. 
# train!
# note:
# we may use batch norm as well. but not for the first layer of discriminator and ot the last layer of geerator!
import torch 
from torchvision import models, datasets, transforms
import torch.nn.functional as F 
import torch.nn as nn 
import torch.utils as utils 
import matplotlib.pyplot as plt 
%matplotlib inline
import numpy as np 
# create the transform 
trans = transforms.ToTensor()
train_dataset = datasets.MNIST('data',True, transform=trans,download=True)
test_dataset = datasets.MNIST('data',False, transform=trans,download=True)

#craete dataloaders 
batch_size = 128
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size= batch_size, shuffle=True, num_workers = 0)
test_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size= batch_size, num_workers = 0)

# visualize a sample image
imgs, labels = next(iter(train_dataloader))
img = imgs[0]
print(img.min())
print(img.max())
imgs = imgs * 2 - 1
print(img.shape)
plt.imshow(img.numpy().transpose(1,2,0).squeeze(),cmap='Greys')
#%%
# Ok, now lets continue we now need to create two networks 
class  Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 1)
        self.drp = nn.Dropout(0.3)
    def forward(self, x):
        x= x.view(x.size(0), -1)
        output = self.drp(F.leaky_relu(self.fc1(x)))
        output = self.drp(F.leaky_relu(self.fc2(output)))
        output = self.drp(F.leaky_relu(self.fc3(output)))
        return output

d = Discriminator()
x = torch.rand(size=(5,1,28,28))
output = d(x)
print(output)

# now lets create our generator 
# star with a vector of somelength, upsample to reach image dims 
class Generator(nn.Module):
    def __init__(self, z_size=100):
        super().__init__()

        self.fc1 = nn.Linear(100, 200)
        self.fc2 = nn.Linear(200,500)
        self.fc3 = nn.Linear(500, 784)#28*28
        self.drp = nn.Dropout(0.5)

    def forward(self, x):
        x = x.view(x.size(0),-1)
        output = self.drp(F.leaky_relu(self.fc1(x)))
        output = self.drp(F.leaky_relu(self.fc2(output)))
        # force outputs to -1/1 
        output = F.tanh(self.fc3(output))
        return output.view(x.size(0),1,28,28)

# lets test our generator 
g = Generator()
x = torch.from_numpy(np.random.uniform(-1,1,size=(2,100))).float()
output = g(x)
print(output.shape) 

# now lets create losses for our networks 
def real_loss(outputs, smooth = False):
    batch_size = outputs.size(0)
    if smooth:
        labels = torch.ones(batch_size) *0.9
    else : 
        labels = torch.ones(batch_size)

    criterion = nn.BCEWithLogitsLoss()
    return criterion(outputs.view(*labels.shape).cpu(), labels)

def fake_loss (outputs, smooth=False):
    batch_size = outputs.size(0)
    if smooth:
        a,b=(0.0,0.3)
        labels = torch.zeros(batch_size) * (b - a ) * np.random.random_sample() + a
    else:
        labels = torch.zeros(batch_size)
    criterion = nn.BCEWithLogitsLoss()
    return criterion(outputs.view(*labels.shape).cpu(),labels)

#%%

# OK, now lets create our models. 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
d = Discriminator().to(device)
z_size = 100
g = Generator(z_size).to(device)

# optimizers = 
d_opt = torch.optim.Adam(d.parameters(), lr = 0.01)
g_opt = torch.optim.Adam(g.parameters(), lr = 0.002)

epochs =20
interval = 2
fixed_z = torch.from_numpy(np.random.uniform(-1,1,size=(6, z_size))).float()
lst=[]
for e in range(epochs):

    d.train()
    g.train()
    for i,(imgs,labels) in enumerate(train_dataloader):

        imgs = imgs.to(device)
        # first  the discriminator 
        output = d(imgs)
        loss11 = real_loss(output, True)
        # now create a random vec and feed generator 
        z_var = torch.from_numpy(np.random.uniform(-1,1,size=(imgs.size(0), z_size))).float()

        output_fake = g(z_var.to(device))

        output12 = d(output_fake)
        loss12 = fake_loss(output12, False)

        loss1 = loss11 + loss12 

        d_opt.zero_grad()
        loss1.backward()
        d_opt.step()

        # now the generator 
        z_var = torch.from_numpy(np.random.uniform(-1,1,size=(imgs.size(0),z_size))).float()
        output21 = g(z_var.to(device))
        output22 = d(output21)
        loss2 = real_loss(output22)

        g_opt.zero_grad()
        loss2.backward()
        g_opt.step()

    print(f'loss1: {loss1.item():0.4f} loss2: {loss2.item():.4f}')
    g.eval()
    imgs=g(fixed_z.to(device))
    lst.append(imgs)
#%%
from torchvision import utils as v_utils
#visualize the outputs 
def visualize(samples, nrows=4,ncols=4):

    print(len(samples))     
    assert len(samples) == nrows*ncols, 'they must match!'
    fig = plt.figure(figsize=(8,5))
    fig, axes = plt.subplots(nrows=nrows,ncols=ncols,sharex=False,sharey=False)
    print(axes.flatten().shape)

    for ax,img in zip(axes.flatten(),samples):
        # print(img.shape)
        # img = v_utils.make_grid(img.cpu().detach(), 6) 
        # img = img.numpy().transpose(1,2,0) 
        img = img[0].cpu().detach().numpy().transpose(1,2,0).squeeze()
        img = (img +1 )/ 2
        # img = ((img +1)*255 / (2)).astype(np.uint8)
        ax.imshow(img, cmap='Greys')
        # plt.show()


visualize(lst, 4, 5)


#%% 
# now in this section we want to test DCGAN, thats simply the acronim for 
# Deep Convolutional GAN! simply replace fc layers with conv layers! 
# and in generator we use upsample or convtranspose!

import torch 
from torchvision import datasets,models,transforms
import torch.nn as nn
import torch.nn.functional as F 
import matplotlib.pyplot as plt
%matplotlib inline 
import numpy as np 

# # create dataset 
dataset_train = datasets.MNIST(root='MNIST',train=True, transform=transforms.ToTensor())
# dataset_train = datasets.CIFAR10(root='cifar10',train=True, transform=transforms.ToTensor())
# dataset_train = datasets.SVHN(root='SVHN',split='train', transform=transforms.ToTensor())

batch_size=128
dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, num_workers=0)

def conv_batch(in_,out_,k_size,padding, stride, bias=False, act = nn.LeakyReLU(), batchnorm=True):
    return nn.Sequential(nn.Conv2d(in_,out_,k_size,stride,padding=padding,bias=bias),
                        nn.BatchNorm2d(out_) if batchnorm else nn.Identity(),
                        act)
# lets create models 
class misc(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        print(f'shape: {input.shape}')
        return input

class D(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(conv_batch(1, 32, 3, 1, 1, batchnorm=False),#32x32
                                #  misc(),
                                 conv_batch(32,64, 3, 1, 2, batchnorm=True),#16x16
                                #  misc(),
                                 nn.MaxPool2d(2,2),#8x8
                                #  misc(),
                                nn.Dropout2d(0.2),
                                 conv_batch(64, 128, 3, 1, 2,batchnorm=True),#4x4
                                #  misc(),
                                nn.Dropout2d(0.2),
                                 conv_batch(128, 10, 3, 1, 2,batchnorm=True),#2x2
                                #  misc(),
                                nn.Dropout2d(0.2),
                                 nn.Flatten(),
                                #  misc(),
                                 nn.Linear(10*2*2, 1))
    def forward(self, input):
        output = self.net(input)
        return output

def deconv(in_,out_,k,s,p,bias=False,batchnorm=True,act=nn.LeakyReLU()):
    return nn.Sequential(nn.ConvTranspose2d(in_, out_, k, s ,padding=p, bias=bias),
                  nn.BatchNorm2d(out_) if batchnorm else nn.Identity(),
                  act)
class G(nn.Module):
    def __init__(self, z_size):
        super().__init__()
        conv = 100
        # adding more fc layers help drastically but it may also cause model collapse
        self.fc1 = nn.Linear(z_size, z_size*2)
        self.fc2 = nn.Linear(z_size*2, 400 *4*4)# 4x4 depth, h,w
        self.deconv1 = deconv(400, 300, 4, 2, 1)
        self.deconv2 = deconv(300, 200, 4, 2, 2)
        self.deconv3 = deconv(200, 1,4, 2, 1, act=nn.Tanh(), batchnorm=False)

        self.drp = nn.Dropout2d(0.3)

    def forward(self, input):
        shape = input.shape
        #input = input.view(input.size(0), -1)
        output = self.drp (F.relu(self.fc1(input)))
        output = self.drp (F.relu(self.fc2(output)))

        output = output.view(input.size(0), 400, 4, 4)
        
        output = self.drp(self.deconv1(output))
        output = self.drp(self.deconv2(output))
        output = self.deconv3(output)
        # print(output.shape)
        return output

d = D()
print(f'Discriminator: {d}')
randx = torch.rand(size=(2,1,32,32))
out1 = d(randx)
print(f'd.out: {out1.shape}')

g = G(z_size=100)
randx2 = torch.from_numpy(np.random.uniform(-1, 1, size=(2,100))).float()
out2 = g(randx2)
print(f'g.out: {out2.shape}')

# now lets create the losses and then training loop 
def real_loss(outputs, smooth=False):
    batch = outputs.size(0)
    if smooth: 
        labels = torch.ones(batch) * 0.9  # or torch.ones(batch)* (0.9 - 0.7) * np.random_sample()+0.7
    else:
        labels = torch.ones(batch)
    criterion = nn.BCEWithLogitsLoss()
    return criterion(outputs.view(*labels.shape), labels)

def fake_loss(outputs, smooth=False):
    batch=outputs.size(0)
    labels = torch.zeros(batch)
    criterion = nn.BCEWithLogitsLoss()
    return criterion(outputs.view(*labels.shape), labels)

#%% 
# now lets train 
device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
d = D().to(device)
g = G(100).to(device)

opt_d = torch.optim.Adam(d.parameters(), lr=0.01)
opt_g = torch.optim.Adam(g.parameters(), lr=0.004)

epochs = 50 
interval=200
z_fix = torch.from_numpy(np.random.uniform(-1,1,size=(10,100))).float().to(device) 
lst2=[]
for e in range(epochs):
    d.train()
    g.train()
    for i, (imgs,labels) in enumerate(dataloader_train):
        imgs = imgs.to(device)

        #rescale 
        imgs = imgs * 2 + -1

        outputs_real = d(imgs)
        loss_real = real_loss(outputs_real.cpu(),True)
        z_vec = torch.from_numpy(np.random.uniform(-1,1,size=(imgs.size(0),100))).float().to(device)
        output_fake = g(z_vec)
        output_fake_d = d(output_fake)
        loss_fake_d = fake_loss(output_fake_d.cpu())
        loss1  = loss_real + loss_fake_d 

        opt_d.zero_grad()
        loss1.backward()
        opt_d.step()

        z_vec = torch.from_numpy(np.random.uniform(-1,1,size=(imgs.size(0),100))).float().to(device)
        fake_to_real_img = g(z_vec)
        fake_to_real_output = d(fake_to_real_img)
        fake_real_loss = real_loss(fake_to_real_output.cpu(),True)

        opt_g.zero_grad()
        fake_real_loss.backward()
        opt_g.step()

    print(f'({e}/{epochs}) loss1: {loss1.item():0.4f} loss2: {fake_real_loss.item():.4f}')
    g.eval()
    outputs = g(z_fix)
    lst2.append(outputs)

#%%
from torchvision import utils as v_utils
#visualize the outputs 
def visualize(samples, nrows=4,ncols=4):

    print(len(samples))     
    assert len(samples) == nrows*ncols, 'they must match!'
    fig = plt.figure(figsize=(8,5))
    fig, axes = plt.subplots(nrows=nrows,ncols=ncols,sharex=False,sharey=False)
    print(axes.flatten().shape)

    for ax,img in zip(axes.flatten(),samples):
        # print(img.shape)
        # img = v_utils.make_grid(img.cpu().detach(), 6) 
        # img = img.numpy().transpose(1,2,0) 
        img = img[0].cpu().detach().numpy().transpose(1,2,0).squeeze()
        img = (img +1 )/ 2
        # img = ((img +1)*255 / (2)).astype(np.uint8)
        ax.imshow(img, cmap='Greys')
        # plt.show()


visualize(lst2[30:], 4,5)