#%% 
# in  the name of God the most compassionate the most merciful 

# papers: https://arxiv.org/abs/1503.03832
# refs : 
# http://bamos.github.io/2016/01/19/openface-0.2.0/
# https://omoindrot.github.io/triplet-loss 
# https://github.com/adambielski/siamese-triplet/blob/master/Experiments_MNIST.ipynb
# https://hackernoon.com/facial-similarity-with-siamese-networks-in-pytorch-9642aa9db2f7

# read the refs both of them . 
# Here we will be creating a siamese network for face verification
# basically to cut a longs tory short, we are trying to create a 
# network that can distinguish between different individuals. 
# there are many different ways to tackle this, one way is using
# a siamese network which is(was) two networks that share the same para
# meters,(now only one network!) and feed two images (similar and disimilar images)
# and then use their corrosponding features, to say if they are
# the same or not(actually 3 images if anchor is used). 
# we use triple loss which includes an anchor image
# (or simply the actual image) a positive image (which is an another
# image (the anchor and positive image belong to the same person!))
# and a negative image, which is someone else's image. 
# the loss is like this : 
# D(A, P) << D(A,N), which reads, the distance between anchor and pos
#itive image must be way lower than the distance metric between the anchor 
# and a negative image . So the loss can be reformulated as : 
# D(A,P) + D(A,N) = 0 , while this reformulation seems plausible
# it is not perfect, we add another term , (called margin) 
# to the D(A,P) so we have D(A,P) + alpha < D(A,N). 
# This means D(A,P) needs to be much (specified by alpha (e.g =0.2))
# lower than the D(A,D). this intuitievely makes our network be more
# accurate and not randomly mistake a wrong person with another one. 
# OK, there are other losses as well! like, we can simply use BCE 
# and treat correct and incorrect images simply as 1 and 0s. 
# this was explained in deepface paper back in 2014! 
# However, the actual implementation of triple loss is more complicated
# basically there are two ways to achieve it, using offline mining and
# online mining. the offline mining refers to the fact that the triplets
# (the 3 image-pairs we use in our triple loss) are precomputed form the
# dataset and then used in the training. the online mining refers to creating
# the triplets on the fly. this simply means, we create triplets from each batch
# while training. the second link explains this very well. 
# for the sake of brevity we first use the lame(naive) implementation! which is 
# not efficient at all. and later on implement the online mining method 
#   
import torch 
from PIL import Image 
import numpy as np 
import os
import torchvision 
from torchvision import datasets, transforms, utils 
import matplotlib.pyplot as plt 
%matplotlib inline 
from collections import defaultdict
import random 
import torch.nn as nn 
import torch.nn.functional as F 

plt.imshow(Image.open('/media/hossein/SSD1/A-Quick-and-Simple-Pytorch-Tutorial/Face verification Siamese Networks/data/faces/training/s8/9.pgm'),cmap='Greys_r')

# ok, lets create a dataset for reading the images triples easily !
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, root='data/faces/',is_train=True, trans=transforms.ToTensor()):
        super().__init__()

        self.is_train = is_train
        self.transform = trans
        if self.is_train:
            self.root = os.path.join(root,'training')
        else:
            self.root = os.path.join(root,'testing')
        # since our dataset is already in  the form of classes (s1,s2,etc)
        # and each class has its own folder. we can simply use ImageFolder
        # and ease ourselevs
        self.dataset = datasets.ImageFolder(self.root, trans)
        self.class_indexes = [idx for c,idx in self.dataset.class_to_idx.items()]
        # here we use a default dict and using that we create a dictionary
        # with the class idx as its key and a list of all images belonging 
        # to that class as its value. the good thing about defaultdict is
        #  that, when we first start, and there is no class, key, it wont
        # give an error, it simply creates the new key and voila, we add 
        # the items to it (we specified its default type as a list)
        self.class_idxs= defaultdict(list)
        for (image_path, class_idx) in self.dataset.imgs:
            self.class_idxs[class_idx].append(image_path)
        # print(f'img path: {self.dataset.imgs[0]}')

    def __getitem__(self, index):
        
        # we need to return anchor, positive and negative
        # the catch is, we must implement it in a way that 
        # nearly 50% of the times, positive samples are chosen
        # so basically we are not using the index argument here
        
        # here we are going to choose an anchor, a positive
        # and a negative example and return them! 
        # lets find an anchor first, we randomly choose a class idx
        anchor_class = random.choice(self.class_indexes)
        # print(f'anchor class: {anchor_class}')
        # now lets pick the anchor image and another random positive sample!
        anchor_img, positive_img = random.choices(self.class_idxs[anchor_class], k=2)

        # now lets choose another class for our negative example
        neg_class = random.choice([num for num in self.class_idxs if num != anchor_class])
        negative_img = random.choice(self.class_idxs[neg_class])
        
        anchor = self.transform(Image.open(anchor_img).convert('L'))
        img0 = self.transform(Image.open(positive_img).convert('L'))
        img1 = self.transform(Image.open(negative_img).convert('L'))
        
        # here we didnt use triple loss, for triple loss we need to 
        # get the anchor in addition to the positive and negative samples

        return anchor, img0, img1 

    def __len__(self):
        length = len(self.dataset)
        return length


transform = transforms.Compose([transforms.ToTensor()])
dataset_train = MyDataset(is_train=True,trans=transform)
dataset_test = MyDataset(is_train=False,trans=transform)

imgs = dataset_train[1]
print(imgs[0].shape)
def show_img(imgs):
    fig = plt.figure(figsize=(6, 4))
    for i,img in enumerate(imgs): 
        ax = fig.add_subplot(1, 3, i+1, xticks=[], yticks=[])
        ax.imshow(img.numpy().transpose(1,2,0).squeeze(), cmap='Greys_r')

show_img(imgs)
# now lets create the data loaders 
dataloader_train = torch.utils.data.DataLoader(dataset_train,batch_size=32,num_workers=0)
dataloader_test = torch.utils.data.DataLoader(dataset_test,batch_size=32,num_workers=0)




#now that we have done this, lets create our network and then our loss function!
def conv_batch(in_,out_,k_size, stride,padding, bias=False, act = nn.ReLU(), batchnorm=True):
    return nn.Sequential(nn.Conv2d(in_,out_,k_size,stride,padding=padding,bias=bias),
                        nn.BatchNorm2d(out_) if batchnorm else nn.Identity(),
                        act)

class view(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        print(f'shape: {input.shape}')
        return input

class global_max_pooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        # kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False,return_indices=False
        output = F.max_pool2d(input, input.size()[2:])
        output = output.view(output.size(0), -1)
        # output = F.adaptive_max_pool2d(input,1)
        return output

class printshape(nn.Module):
    def __init__(self):
        super().__init__()  

    def forward(self, input):
        print(input.shape)
        return input        

class siamese_net(nn.Module):
    def __init__(self,embedding_size=128):
        super().__init__()

        self.embedding_size = embedding_size
        self.model = nn.Sequential( conv_batch(1,12,3,2,0,act=nn.ReLU()),
                                    nn.Dropout2d(0.2),
                                    conv_batch(12,32,3,2,0,act=nn.ReLU()),
                                    nn.Dropout2d(0.2),
                                    conv_batch(32,64,3,2,0,act=nn.ReLU()),
                                    nn.Dropout2d(0.2),
                                    conv_batch(64,64,3,2,0,act=nn.ReLU()),
                                    nn.Dropout2d(0.2),
                                    conv_batch(64,128,3,2,0,act=nn.ReLU()),
                                    nn.Dropout2d(0.2),
                                    # instead of flatten we could simply use
                                    # the global pooling but I didnt feel like
                                    # it! ok let me code it! done!
                                    # printshape(),
                                    global_max_pooling(), 
                                    # printshape(),
                                    # nn.Flatten(),
                                    nn.Linear(128,embedding_size))

        # this is the naive method. the network is replicated
        # (here we use 3 forward passes) to get 3 emebddings
        # for each input(anchor,positive,negative images)
        # another way is to just create triplets from the
        # the samebatch of images. meaning, we simply feed 
        # the network with n images (just like a normal
        # classification) and get 3 embeddings at the end
        # and then use those embeddings to create triplets
        # This is called online triplet mining
        # read more here :
        # https://omoindrot.github.io/triplet-loss                            
    def forward(self, input1,input2,input3):
        vec1 = self.model(input1)
        vec2 = self.model(input2)
        vec3 = self.model(input3)
        return vec1, vec2, vec3

img1,img2,img3 = next(iter(dataloader_train))
print(img1.shape)
m = siamese_net()
vecs = m(img1,img2,img3)
print(vecs[0].shape)

# now let us write down our loss function! 
def loss_function(embanc, embpos,embneg, margin=0.2):
    # we can use euclidean distance, or anything like it
    # to get the distance between two embedding. here we will
    # use pairwise distance function in Pytorch
    # p=2 means eucleadian distance! 
    distance = nn.PairwiseDistance(p=2)
    d1 = distance(embanc,embpos)
    d2 = distance(embanc,embneg)
    
    loss = d1 - d2 + margin
    loss = torch.max(loss, torch.zeros_like(loss))
    loss = torch.sum(loss)
    return loss

epochs = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = siamese_net().to(device)
optimizer=torch.optim.Adam(model.parameters(), lr = 0.01)

for e in range(epochs):
    losses=0.0
    for imgs in dataloader_train:
        imgs = tuple(img.to(device) for img in imgs)

        img_ebmbeddings = model(*imgs)
        loss = loss_function(*img_ebmbeddings, margin=0.2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses += loss.item()
    print(f'loss: {losses:.4f}')
#%% 
# OK now lets test!
img1,img2,img3 = next(iter(dataloader_test))
print(img1[0].shape)
img1 = img1.to(device)
img2=img2.to(device)
img3=img3.to(device)

model.eval()
embed1, embed2, embed3 = model(img1,img2,img3)

distance = nn.PairwiseDistance(p=2)
# print(embeds[0][0])
# print(embeds[0][1])
id=3
d1 = torch.sub(torch.pow(embed1[id],2), torch.pow(embed3[id],2))
print(d1.shape)

# distance = F.pairwise_distance(embeds[0][0], embeds[0][1], p=2, eps=1e-6, keepdim=True)
img = torch.cat((img1[id],img3[id]), 2)
print(img.shape)
img = utils.make_grid(img,nrow=1)
plt.imshow(img.cpu().numpy().transpose(1,2,0))
plt.text(75,8,f'similarity: {torch.mean(d1).item():.4f}',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})

# %%
