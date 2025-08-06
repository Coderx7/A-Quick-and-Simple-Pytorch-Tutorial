#%% 
# In the name of God 
import torch 
import torch.nn.functional as F 
import torchvision 
from torchvision import datasets, transforms, models
import torch.nn as nn 
import numpy as np 
import matplotlib.pyplot as plt 

# we are going to use GANs to train a semi supervised fashion. 
# we will be using svhn dataset. and we will be using a fraction
# of the dataset in a supervised manner (with label) and use the 
# rest of the dataset in an unsupervised manner in a GAN network
# lets see how this is done . 
# first here are the steps : 
# 1. define a new dataset class for svhn for ease of use(getting a specified number of samples only)
# 2. create a normal discriminator 
# 3. create a normal generator 
# 4. create two losses (real and fake losses)
# 5. create a class for ganlogits for ease of use 
# 6. create a new loss that uses feature similarity for our generator 
# 7. train. 
# we will expand on all of these when we try to implemet them. 
# lets begin 
# DataSet for our SVHN. 
# basically we already have a dataset dedicated to SVHN dataset in Pytorch. 
# but in order to be able to have more control and also prevent code clutter
# we create one here. this is how we do it 
class SVHN_Dataset(torch.utils.data.Dataset):
    def __init__(self, img_size, split='train', label_mask_size = 1000):
        super().__init__()
        self.split = split.strip().lower()
        self.img_size = img_size
        self.mask_size = label_mask_size
        trans = transforms.Compose([transforms.Resize(img_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))])
        self.dataset = datasets.SVHN(root='SVHN',split=self.split, transform = trans, download=True)
        self._create_label_mask()
    
    def is_train_split(self):
        return True if self.split == 'train' else False
    
    def _create_label_mask(self):
        if not self.is_train_split():
            self.mask = None
        mask = np.zeros(shape=(len(self.dataset)))    
        mask[0 : self.mask_size] = 1
        np.random.shuffle(mask)
        self.mask = torch.FloatTensor(mask)
        

    def __len__(self):
        return len(self.dataset) 
    
    def __getitem__(self, idx):
        imgs, labels = self.dataset.__getitem__(idx)
        if self.is_train_split():
            return imgs, labels, self.mask[idx]
        return imgs, labels


#%%
# get data loaders and view a batch of images 
def get_loaders(batch_size=32, image_size=32, mask_size=1000):
    data_loader_train = torch.utils.data.DataLoader(SVHN_Dataset(image_size,split='train',label_mask_size=mask_size),
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    num_workers = 0)
                                                    
    data_loader_test = torch.utils.data.DataLoader(SVHN_Dataset(image_size, split='test'),
                                                   batch_size=batch_size, num_workers=0)
    return data_loader_train, data_loader_test

dataloader_train, dataloader_test = get_loaders(batch_size=32, image_size=32, mask_size=1000) 

def visualize(imgs, rows=3, cols=11):
    fig = plt.figure(figsize=(cols,rows))
    plt.subplots_adjust(wspace=0,hspace=0)
    imgs = imgs.detach().cpu().numpy().transpose(0,2,3,1)
    count = imgs.shape[0]
    print(f'imgs count {count}')
    for i in range(count):
        ax = fig.add_subplot(rows, cols, i+1, xticks=[], yticks=[])
        #denormalize 
        img = ((imgs[i] - imgs[i].min())*255 / (imgs[i].max() - imgs[i].min())).astype(np.uint8)
        
        ax.imshow(img)
    plt.show()

imgs, labels, masks = next(iter(dataloader_train))
visualize(imgs)

imgs, labels = next(iter(dataloader_test))
visualize(imgs)


#%%
# create a discriminator
def conv_batchnorm(in_, out_, kernel_size, stride, padding, batch_norm=False, bias=False):
    layers = nn.ModuleList()
    layers.append(nn.Conv2d(in_, out_, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
    if batch_norm: 
        layers.append(nn.BatchNorm2d(out_))
    return nn.Sequential(*layers)

class Discriminator(nn.Module):
    def __init__(self, conv_fmaps=32, num_classes=10, act=nn.ReLU()):
        super().__init__()

        self.conv_fmaps = conv_fmaps
        self.num_classes = num_classes
        self.act = act
        self.conv1 = conv_batchnorm(3, conv_fmaps, kernel_size=3, stride=1, padding=1,bias=True)#32
        self.conv2 = conv_batchnorm(conv_fmaps, conv_fmaps, kernel_size=3, stride=1, padding=1,batch_norm=True)#32
        # nn.MaxPool2d(2,2),#16
        self.conv3 = conv_batchnorm(conv_fmaps, conv_fmaps*2, kernel_size=3, stride=1, padding=1,batch_norm=True)#16
        self.conv4 = conv_batchnorm(conv_fmaps*2, conv_fmaps*2, kernel_size=3, stride=1, padding=1,batch_norm=True)#16
        #nn.MaxPool2d(2,2),#8
        self.conv5 = conv_batchnorm(conv_fmaps*2, conv_fmaps*3, kernel_size=3, stride=1, padding=1,batch_norm=True)#8
        self.conv6 = conv_batchnorm(conv_fmaps*3, conv_fmaps*3, kernel_size=3, stride=1, padding=1,batch_norm=True)#8
         #nn.MaxPool2d(2,2),#4
        self.conv7 = conv_batchnorm(conv_fmaps*3, conv_fmaps*4, kernel_size=3, stride=1, padding=1,batch_norm=True)#4
         #nn.MaxPool2d(2,2))#2
        #when using average pooling
        self.fc = nn.Linear(conv_fmaps*4*1*1, num_classes + 1)
        self.avgpool = nn.AvgPool2d(2,2)
        self.drp = nn.Dropout2d(p=0.2)

    def forward(self, input):
        output = self.act(self.conv1(input))
        output = self.drp(self.act(self.conv2(output)))
        output = F.max_pool2d(output,2,2)

        output = self.drp(self.act(self.conv3(output)))
        output = self.drp(self.act(self.conv4(output)))
        output = F.max_pool2d(output,2,2)

        output = self.drp(self.act(self.conv5(output)))
        output = self.drp(self.act(self.conv6(output)))
        output = F.max_pool2d(output,2,2)

        output = self.drp(self.act(self.conv7(output)))
        output = F.max_pool2d(output,2,2)

        # use average pooling 
        # for features we dont use batchnormalization , since it will normalize them and this is not what we want!
        features = self.avgpool(output)
        features = features.view(input.size(0), -1)
        class_logits = self.fc(features)

        #ganlogits 
        # gives ten logits in each sample, to real_class_logits and the next remaining logit to fake_class_logit
        real_class_logits, fake_class_logits = torch.split(class_logits, self.num_classes, dim=1)
        # since this is only a single number , lets remove the 1, dimension (1,1) becomes, (1)
        fake_class_logits = fake_class_logits.squeeze()
        # for stability sake, we use a numerically more stable class logits here 
        max_logits,_ = torch.max(class_logits, dim=1, keepdim=True)
        stable_real_class_logits = class_logits - max_logits
        max_logits = max_logits.squeeze()
        # now simply calculate the gan logits 
        # Set gan_logits such that P(input is real | input) = sigmoid(gan_logits).

        # Keep in mind that class_logits gives you the probability distribution over all the real
        # classes and the fake class. You need to work out how to transform this multiclass softmax
        # distribution into a binary real-vs-fake decision that can be described with a sigmoid.
        # Numerical stability is very important.
        # You'll probably need to use this numerical stability trick:
        # log sum_i exp a_i = m + log sum_i exp(a_i - m).
        # This is numerically stable when m = max_i a_i.
        # (It helps to think about what goes wrong when...
        #   1. One value of a_i is very large
        #   2. All the values of a_i are very negative
        # This trick and this value of m fix both those cases, but the naive implementation and
        # other values of m encounter various problems)
        
        # basically what we have at hand, is real class logits, which are probability distributions over all real 
        # and fake classes. we need  to find a way to transform this probablity distribution into a binary realvsfake
        # decision that can be described with a sigmoid! 
        # we can do this using log(sum(exp())), but this needs to be stable numerically, as we may have
        # 1.a very negative number, and 2. value of a_i is very large.

        # It turns out that in parametrizing a lot of machine learning problems, 
        # you actually have to perform this operation of taking the log of the sum of exponentials.
        # For example, this comes up when calculating the log-likehood.
        # So in these cases, it’s nice to have a single function which performs the whole series of operations
        # for you.
        # Using the log-sum-exp function, with something called the “log-sum-exp trick”,
        # can help you prevent underflow/overflow errors. The “log-sum-exp” trick is essentially
        # just exploiting a mathematical identity to reduce underflow/overflow when you use the log-sum-exp function.
        #  See this helpful blog post for more details.
        # read
        # https://thoughtsrecurring.com/posts/log-sum-exp-trick/
        # https://blog.feedly.com/tricks-of-the-trade-logsumexp/
        # https://medium.com/@chrisxu94/a-one-minute-intro-to-the-log-sum-exp-function-7b165bdc13f4
        # read : https://www.xarg.org/2016/06/the-log-sum-exp-trick-in-machine-learning/ 

        gan_logits = torch.log(torch.sum(torch.exp(stable_real_class_logits),1)) + max_logits - fake_class_logits

        # later use softmax_log instead of softmax and see how it goes
        class_probs = F.softmax(class_logits)
        return class_probs, class_logits, gan_logits, features


def deconv_(in_, out_, kernel_size, stride, padding, batchnorm = False, bias = False):  
    layers = []
    layers.append(nn.ConvTranspose2d(in_, out_, kernel_size=kernel_size,stride=stride, padding=padding,bias=bias))
    if batchnorm:
        layers.append(nn.BatchNorm2d(out_))
    return nn.Sequential(*layers)

class Generator(nn.Module):
    def __init__(self, z_size=100, conv_fmaps=32):
        super().__init__()   
        self.initial_hw = 4
        self.fc = nn.Linear(z_size, conv_fmaps*8 * self.initial_hw*self.initial_hw)
        self.conv_fmaps = conv_fmaps

        self.G = nn.Sequential(
            deconv_(conv_fmaps*8 , conv_fmaps*8, kernel_size=4, stride=2, padding=1,batchnorm=True, bias=False),#8x8
            nn.ReLU(),
            deconv_(conv_fmaps*8, conv_fmaps*4, kernel_size=3, stride=1, padding=1, batchnorm=True, bias=False),#8x8
            nn.ReLU(),
            deconv_(conv_fmaps*4, conv_fmaps*2, kernel_size=4, stride=2, padding=1, batchnorm=True, bias=False),#16x16
            nn.ReLU(),
            deconv_(conv_fmaps*2, conv_fmaps, kernel_size=3, stride=1, padding=1, batchnorm=True, bias=False),#16x16
            nn.ReLU(),
            deconv_(conv_fmaps, 3, kernel_size=4, stride=2, padding=1, batchnorm=False, bias=True)#32x32
        )
    def forward(self, input):
        output = F.relu(self.fc(input))
        output = output.view(-1, self.conv_fmaps* 8, self.initial_hw, self.initial_hw)
        output = self.G(output)
        output = F.tanh(output)
        return output


dd = Discriminator(conv_fmaps=32,num_classes=10)
img = np.random.randn(2,3,32,32)
img = torch.from_numpy(img).float()
cls_probs, cls_logits, gan_logits, features = dd(img)
print('D: ',cls_probs.shape, cls_logits.shape, gan_logits.shape, features.shape)

gg = Generator(z_size=100, conv_fmaps=32)
z = np.random.uniform(-1,1,size=(2,100))
z = torch.from_numpy(z).float()
imgs = gg(z)
print(f'G:{imgs.shape}')

#%%
# losses 
def real_loss(output, device, smooth=True): 
    batch_size = output.size(0)
    if smooth:
        labels = torch.ones(batch_size) * 0.9
    else:
        labels = torch.ones(batch_size)
    criterion = nn.BCEWithLogitsLoss()
    return criterion(output.squeeze(), labels.to(device).float())

def fake_loss(output, device):
    batch_size = output.size(0)
    labels = torch.zeros(batch_size).to(device)
    criterion = nn.BCEWithLogitsLoss()
    return criterion(output, labels.float())

def one_hot(input, num_classes=11):
    arr = np.zeros(shape=(input.size(0), num_classes))
    input = input.detach().cpu().numpy()
    arr[np.arange(input.shape[0]), input]=1
    return torch.from_numpy(arr).float()

print(one_hot(torch.tensor([1,2,9])))

def scale(imgs):
    max_,_ = imgs.max(dim=0)
    min_,_ = imgs.min(dim=0)
    print(max_.shape)
    print(min_.shape)
    imgs = imgs * (max_ - min_) + min_
    return imgs

#%% 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

z_size = 100
conv_fmaps = 32

D = Discriminator(conv_fmaps=conv_fmaps, num_classes = 10).to(device)
G = Generator(z_size=z_size, conv_fmaps=conv_fmaps).to(device)

lr = 0.0002
beta1 = 0.5
beta2 = 0.99

optim_D = torch.optim.Adam(D.parameters(), lr = lr)
optim_G = torch.optim.Adam(G.parameters(), lr = lr)

epochs = 80
interval = 600 

rows = 4
cols = 10

for e in range(epochs):
    D.train()
    G.train()
    masked_corrects=0
    num_samples = 0
    for i, (imgs, labels, masks) in enumerate(dataloader_train):

        # normalize the image (to be between -1 and 1)
        # feed it to the discriminator. 
        # imgs = scale(imgs)
        batch_size = imgs.size(0)
        # here is the story, previously in normal gans we would be 
        # reading a real image, feeding it to D, get the real_loss
        # then feed a fake image from generator and get the fake_loss
        # 2. then for the generator we would generate a new image 
        # feed it to discriminator and treat it as if it was real 
        # and there for calculate real_loss. 
        # How ever, in this new case of ours, i.e. semi-supervised training
        # we have very few labeled data, and lots of unlabeled data.
        # 
        # so we must find a way to specify which gan image belongs t o 
        # to the correct class and which doesnt (apart from looking like the real classes repsectively)
        # so how do we do this? 
        # we first feed our image to the discriminator. 
        # then we get, outputs, class_logits, gan_logits and features
        # here we first are going to deal with our gan_logits. 
        # treat them as real (since they are being read from our real dataset)
        # so we use real_loss_g here for it, then we ourselevs create a fake image
        # and feed it to our discriminator, this time, the new gan_logits from our fake image
        # is used for our fake_loss_g (since it doesnt exists, we created it) thus labeling it fake
        # lets do this part and calculate our gan_loss here   
        imgs = imgs.to(device)
        labels = labels.to(device).long()

        output, class_logits, gan_logits, features = D(imgs)
        # we need two losses, real loss and fake loss. 
        # since this data is fed from real images, our gan_logits refer to real images
        # so we consider and treat them as real!  
        d_real_loss_g = real_loss(gan_logits, device, smooth=True)
        # we now need a fake loss, lets create a fake image using generator 
        # for this we first create a latent vector z 
        z = torch.from_numpy(np.random.uniform(-1,1,size=(batch_size, z_size))).float().to(device)
        fake_image_g = G(z)
        # now the fake loss for D, we feed this fake image and tell it its fake! 
        # remeber we use detach to avoid backprop for netG here
        output_fake, class_logits_fake, gan_logits_fake, features_fake = D(fake_image_g.detach())
        fake_loss_g = fake_loss(gan_logits_fake, device)
        # now our gan_loss will be 
        real_loss_g = d_real_loss_g + fake_loss_g

        # now that was the unsupervised section(gan) that dealt with gans
        # now we have labeled data as well, lets do this section 
        # we have fed our real images before, so no need to refeed our Discriminator
        # we now need to use our normal output lets do it  :
        # we had masks for our labels, to know which label we have and which we dont
        # i.e. artificially acting as if our number of labels are limited! 
        # we do this to better caclulate the accuracy for our labeled data!

        one_hot_labels = one_hot(labels).to(device)
        
        # print('one_hot_labels: ',one_hot_labels.dtype)
        # print('output: ',output.dtype)
        d_class_entropy_loss = -torch.sum(one_hot_labels * torch.log(output),dim=1)
        d_class_entropy_loss = d_class_entropy_loss.squeeze()

        masks = masks.to(device).float()
        # print('masks: ',masks.dtype,masks.device)
        # print('entropy loss: ', d_class_entropy_loss.dtype, d_class_entropy_loss.device)
        
        delim = torch.max(torch.Tensor([1, torch.sum(masks.data)]))
        d_class_loss = torch.sum(masks * d_class_entropy_loss) / delim
        # (simply put, previously we had one use/application for our gan, and)
        # that was creating fake images, here, our gan plays two roles not one
        # it both acts as real images and fake images. so we have a real_loss and
        # fake_loss for our gan here as well)
        d_loss = d_class_loss + real_loss_g

        optim_D.zero_grad()
        # this is a special case as we are not done with the discriminator here
        # we do a backward pass here, but retain our graph since 
        # in the next step we must also take into account the features!
        d_loss.backward(retain_graph=True)
        optim_D.step()

        # generator 
        # now we feed our "previously" generated fake image to the discriminator! this time
        # note that, the discriminator has been updated once! and we are getting new features
        # 
        _,_,_,features_d = D(fake_image_g)
        # now featuremathcing part!
        d_features_first_mean = torch.mean(features,dim=0).squeeze()
        d_features_second_mean = torch.mean(features_d,dim=0).squeeze()
        g_loss= torch.mean(torch.abs(d_features_first_mean - d_features_second_mean))
        
        optim_G.zero_grad()
        g_loss.backward()
        optim_G.step()

        # calculate the number of correct predictions 
        _, preds_idxs = torch.max(class_logits,1)
        eq = torch.eq(labels, preds_idxs)
        correct = torch.sum(eq.float())
        masked_corrects += torch.sum(eq.float() * masks)
        num_samples += torch.sum(masks)

        if i%interval ==0: 
            print('epoch/iter: {}/{}   d_loss: {}   g_loss: {}'.format(e, i, d_loss.item(), g_loss.item()))
            #visualize(imgs,rows, cols)
            #visualize(fake_image_g,rows, cols)
        
        
    accuracy = masked_corrects.item() / max(1, num_samples.item())
    print(f'training_epoch  : {e} accuracy : {accuracy}')

    # test 
    D.eval()
    corrects_t = 0
    num_samples = 0
    for imgs, labels in dataloader_test: 
        imgs = imgs.to(device)
        labels = labels.to(device)
        output,_,_,_ = D(imgs)
        _,preds_idxs = torch.max(output,1)
        eq = torch.eq(labels, preds_idxs)
        corrects_t += torch.sum(eq.float())
        num_samples += len(labels)
    acc = corrects_t/max(1.0,1.0* num_samples)
    print('Test:\tepoch {}/{}\taccuracy {}'.format(epochs,e, acc))  


#%%




#%%
class custom_x(nn.Module):
    def __init__(self, shape):
        super().__init__()

        self.var1 = nn.Parameter(torch.zeros(shape))
        self.var2 = nn.Parameter(torch.rand(shape))
        # print(self.var1, self.var2)

    def forward(self):
        weight_matrix= self.var1 + self.var2 
        return weight_matrix

    def __str__(self):
        return f'{self.var1.grad} \n {self.var2.grad}'    



class snet(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 2, 1, 0)
        shape = self.conv1.weight.shape
        self.var1 = nn.Parameter(torch.ones(shape))
        self.var2 = nn.Parameter(torch.ones(shape))
        # method 2 
        self.cs = custom_x(shape=(shape))

        self.conv2 = nn.Conv2d(6, 6, 5, 1, 0)
        self.fc = nn.Linear(6*11*11, num_classes)

    def some_method(self):
        """
            Suppose this is a custom method, tasked with
            producing values for each entry in the weight matrix
            of a convolutional layer. for simplicity we used 
            addition here
        """
        res = self.var1 + self.var2
        return res

    def forward(self, input):
        #weight = self.some_method()
        weight = self.cs()
        output = F.conv2d(input, weight )
        output = self.conv2(output)
        output = output.view(input.size(0), -1)
        output = self.fc(output)
        #print('var1 grad',self.var1.grad)
        print('cs.var1 ',self.cs.var1)
        print('cs.var1 grad',self.cs.var1.grad_fn)
        
        return output


n = snet(num_classes=3)
fake_dataset = torchvision.datasets.FakeData(100,
                                             image_size=(3, 16, 16),
                                             num_classes=3,
                                             transform=transforms.ToTensor())
fake_dataloader = torch.utils.data.DataLoader(fake_dataset,
                                              batch_size=20)
criterion = nn.CrossEntropyLoss()


opt = torch.optim.Adam(n.parameters(), lr=0.01)
for imgs, labels in fake_dataloader:
    p = n(imgs)
    loss = criterion(p, labels)
    opt.zero_grad()
    loss.backward()
    opt.step()
    print(loss.item())
                