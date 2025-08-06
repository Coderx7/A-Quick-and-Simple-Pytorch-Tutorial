# %% [markdown]
# in the name of Allah the most compassionate the most merciful
# lets create a Face GAN
# %% [markdown]
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


def get_dataloader(root='./processed_celeba_small', batch_size=64, resize=(32, 32)):

    trans = transforms.Compose([transforms.Resize(size=resize),
                                transforms.ToTensor()])
    dataset = datasets.ImageFolder(root=root+'/celeba/', transform=trans)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=2)
    return dataloader


batch_size = 64
dataloader = get_dataloader(batch_size=batch_size, resize=(64, 64))
imgs, labels = next(iter(dataloader))
# %%


def visualize_images(imgs):
    fig = plt.figure(figsize=(8, 8))
    for i in range(imgs.size(0)):
        ax = fig.add_subplot(8, 8, i+1, xticks=[], yticks=[])
        img = imgs[i].numpy().transpose(1, 2, 0)
        ax.imshow(img)


visualize_images(imgs)
#preprocess and rescaling


def rescale(imgs, range_=(-1, 1)):
    (min, max) = range_
    rescaled_images = imgs * (max-min) + min
    return rescaled_images


# check if it works :
print('before rescaling min, max:', imgs[0].min(), imgs[0].max())
imgs = rescale(imgs)
print('after rescaling min, max:', imgs[0].min(), imgs[0].max())
visualize_images(imgs)


# %%
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
    def __init__(self, conv_dim=32, act=nn.ReLU(), mode=0):
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
        self.fc3 = nn.Linear(64*10, 1)
        # self.conv5 = conv_batch(conv_dim*4, conv_dim*5, 3, 2, 1)
        # self.conv6 = conv_batch(conv_dim*5, conv_dim*6, 3, 2, 1)
        self.drp = nn.Dropout(0.5)
        # it seems, larger fmaps prvide better results?!
        self.fc = nn.Linear(conv_dim*10*3*3, 1)

    def forward(self, input):
        if self.mode == 0:
            batch = input.size(0)
            output = self.act(self.conv1(input))
            #print(f'conv1: {output.shape}')
            output = self.act(self.conv2(output))
            #print(f'conv2: {output.shape}')
            # output = F.max_pool2d(output,kernel_size=2)
            output = self.act(self.conv3(output))  # 16
            #print(f'conv3: {output.shape}')
            output = self.act(self.conv4(output))  # 16
            #print(f'conv4: {output.shape}')
            output = self.act(self.conv5(output))  # 16
            #print(f'conv5: {output.shape}')
            output = self.act(self.conv6(output))  # 16
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
            # print(output.shape)
            output = output.view(batch, self.fc.in_features)
            output = self.fc(output)
            output = self.drp(output)
        else:
            output = input.view(-1, 64*64*3)
            output = self.drp(self.fc1(output))
            output = self.drp(self.fc2(output))
            output = self.drp(self.fc3(output))
        return output


def deconv_convtranspose(in_dim, out_dim, kernel_size, stride, padding, batchnorm=True):

    layers = []
    deconv = nn.ConvTranspose2d(
        in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=padding)
    layers.append(deconv)
    if batchnorm:
        layers.append(nn.BatchNorm2d(out_dim))

    return nn.Sequential(*layers)


class img2vec(nn.Module):
    def __init__(self, z_size, n_fmaps):  # image input is 64x64
        super().__init__()
        self.conv1 = conv_batch(
            in_dim=3, out_dim=32, kernel_size=3, stride=1, padding=1, batch_norm=True)  # 64x64
        self.conv2 = conv_batch(
            in_dim=32, out_dim=64, kernel_size=3, stride=2, padding=1, batch_norm=True)  # 32x32
        self.conv3 = conv_batch(
            in_dim=64, out_dim=64, kernel_size=3, stride=2, padding=1, batch_norm=True)  # 16x16
        self.conv4 = conv_batch(
            in_dim=64, out_dim=64, kernel_size=3, stride=1, padding=1, batch_norm=True)  # 16x16
        self.conv5 = conv_batch(
            in_dim=64, out_dim=128, kernel_size=3, stride=2, padding=1, batch_norm=True)  # 8x8
        self.conv6 = conv_batch(in_dim=128, out_dim=n_fmaps,
                                kernel_size=3, stride=2, padding=1, batch_norm=True)  # 4x4

        self.embedding = nn.Linear(
            self.conv6._modules['0'].out_channels*4*4, z_size)
        #self.embedding = nn.Linear(self.conv6._modules['0'].out_channels*4*4 ,z_size)

    def forward(self, input):
        batch_size = input.size(0)
        output = self.conv1(input)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = self.conv5(output)
        output = self.conv6(output)
        # flatten the input
        output = output.view(batch_size, -1)
        # raw scores
        output = self.embedding(output)
        # log probablities
        #output = F.log_softmax(output, dim=1)
        return output


class Generator(nn.Module):
    def __init__(self, z_size=100, conv_dim=32, mode=0):
        super().__init__()
        self.conv_dim = conv_dim
        # make the 1d input into a 3d output of shape (conv_dim*4, 4, 4 )
        self.fc = nn.Linear(z_size, conv_dim*4*4*4)  # 4x4
        print('fc dims', self.fc.weight.shape)
        self.ie = img2vec(z_size, n_fmaps=conv_dim*4)
        print('ie dims', self.ie._modules['embedding'].weight.shape)
        # conv and deconv layer work on 3d volumes, so we now only need to pass the number of fmaps and not the
        # input volume size (its h,w which is 4x4!)
        self.mode = mode
        self.drp = nn.Dropout(0.5)
        if mode == 0:
            self.deconv1 = deconv_convtranspose(
                conv_dim*4, conv_dim*3, kernel_size=3, stride=2, padding=1)  # 7x7
            self.deconv2 = deconv_convtranspose(
                conv_dim*3, conv_dim*2, kernel_size=3, stride=2, padding=1)  # 13x13
            self.deconv3 = deconv_convtranspose(
                conv_dim*2, conv_dim, kernel_size=3, stride=2, padding=1)  # 25x25
            self.deconv4 = deconv_convtranspose(
                conv_dim, conv_dim, kernel_size=4, stride=1, padding=0)  # 16x16
            # batchnorm, may corrupt the output!
            self.deconv5 = deconv_convtranspose(
                conv_dim, 3, kernel_size=5, stride=1, padding=0)
        elif mode == 1:
            self.deconv1 = deconv_convtranspose(
                conv_dim*4, conv_dim*3, kernel_size=4, stride=2, padding=1)  # 10x10
            self.deconv2 = deconv_convtranspose(
                conv_dim*3, conv_dim*2, kernel_size=4, stride=2, padding=1)  # 20x20
            self.deconv3 = deconv_convtranspose(
                conv_dim*2, conv_dim, kernel_size=4, stride=2, padding=1)  # 40x40
            self.deconv4 = deconv_convtranspose(
                conv_dim, conv_dim, kernel_size=3, stride=2, padding=1)  # 63x63
            self.deconv5 = deconv_convtranspose(
                conv_dim, 3, kernel_size=4, stride=1, padding=1, batchnorm=False)  # 64x64
        elif mode == 2:
            self.deconv1 = deconv_convtranspose(
                conv_dim*4, conv_dim*2, kernel_size=4, stride=2, padding=1)  # 7x7
            self.deconv2 = deconv_convtranspose(
                conv_dim*2, conv_dim, kernel_size=4, stride=2, padding=1)  # 14x14
            # self.deconv3 = deconv_convtranspose(conv_dim*2, conv_dim, kernel_size =4, stride=1, padding=1)#15x15
            self.deconv4 = deconv_convtranspose(
                conv_dim, conv_dim, kernel_size=4, stride=2, padding=1)  # 16x16
            self.deconv5 = deconv_convtranspose(
                conv_dim, 3, kernel_size=4, stride=2, padding=1, batchnorm=False)  # 32x32

    def forward(self, input):
        output1_fc_raw = self.fc(input)
        output = self.drp(output1_fc_raw)
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

        output2_embedding_raw = self.ie(output)
        return output, output1_fc_raw, output2_embedding_raw


dd = Discriminator(mode=0)
zd = np.random.rand(2, 3, 64, 64)
zd = torch.from_numpy(zd).float()
# print(dd)
print('D: ', dd(zd).shape)

gg = Generator(mode=1)
z = np.random.uniform(-1, 1, size=(2, 100))
z = torch.from_numpy(z).float()
img, fc_raw, embedding_raw = gg(z)
print('G', img.shape, fc_raw.shape, embedding_raw.shape)

imvec = img2vec(z_size=100, n_fmaps=128)
output = imvec(img)
print('Emb', output.shape)
# %%

# %%
# gg = Generator(mode=2)
# z = np.random.uniform(-1,1,size=(2,100))
# z = torch.from_numpy(z).float()
# print(gg(z).shape)
# dd = Discriminator(mode=0)
# zd = np.random.rand(2,64,64,3)
# zd = torch.from_numpy(zd).float()
# print(dd(zd).shape)
# %%


def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight.data, 0.0, std=0.02)


# %%
# create losses
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
        (a, b) = (0, 0.3)
        labels = torch.ones(batch_size) * (b - a) * \
            np.random.random_sample() + a
    else:
        labels = torch.zeros(batch_size)
    labels = labels.to(device)
    criterion = nn.BCEWithLogitsLoss()
    return criterion(output_.squeeze(), labels)

# %%
# 1. disable bias!
# 2. initialize model weights manually normal distribution
#    with mean = 0, std dev = 0.02.
# 2. use kernel_sizes of 4 ! and see the result  (done, seems a bit lower, but not that much!)
# 3. upsample rapiddly (z_size*4*8*8 instead of z_size*4*2*2) forexample. z_size means
#    the input vector with size_z which we feed to our fc layer at the beginning
#  4.use leaky relu in discriminator
#  5. use deeper architectures
#  6. use shallower architectures


# %%
# now training
# create models first
conv_dim = 64
z_size = 100
batch_size = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

D = Discriminator(conv_dim=conv_dim, act=nn.LeakyReLU(0.2), mode=0)
G = Generator(z_size=z_size, conv_dim=conv_dim, mode=1)
#IE = img2vec(z_size=100)
# print(D)
# print(G)
# print(IE)

criterion = nn.L1Loss()

D.apply(init_weights)
G.apply(init_weights)
lr = 0.0005  # 0.002
# optimizer_d = torch.optim.Adam(D.parameters(), lr = lr)
# optimizer_g = torch.optim.Adam(G.parameters(), lr = lr)

beta1 = 0.1  # momentum
beta2 = 0.99  # default value
optimizer_d = torch.optim.Adam(D.parameters(), lr, [beta1, beta2])
optimizer_g = torch.optim.Adam(G.parameters(), lr, [beta1, beta2])
# to test this we have several ways.
# 1. train generator and imvect together - jointly(parameters are fused together)!
# 2. train generator and imvect separately
#   1.the 3 network gets trained jointly all together
#   2. the generator is trained first and then in a second training loop gets trained with imvect
# ievec will learn the probablity distribution
# the good thing about 1 and 2.1 is that the discriminator
# acts as the true/false label specifier
# the bad thing is I guess it will be hard, but tavakol bar khoda, lets go
# 3. the imvvect is inside generator and will  try to reassmble the input from
# the generated image back to the first thing. this in theory should give us an embedding lookup
# that spits out embedding! when we give it an image!
# optimizer_gie = torch.optim.Adam(G.parameters() + IE.parameters(), lr , [beta1, beta2])
# optimizer_ie = torch.optim.Adam(IE.parameters(), lr , [beta1, beta2])

D = D.to(device)
G = G.to(device)
# create a fixed_vector for evaluating our network performance
z_vec_fixed = np.random.uniform(-1, 1, size=(batch_size, z_size))
# it has to be float! rememer that!
z_vec_tensor_fixed = torch.from_numpy(z_vec_fixed).float().to(device)

epochs = 50
losses = []
samples = []
interval = 500
print(f'data loader length : {len(dataloader.dataset)}')
i = 0
for e in range(epochs):

    D.train()
    G.train()
    for imgs, _ in dataloader:
        i += 1
        imgs = imgs.to(device)
        # rescale images
        imgs = rescale(imgs)

        # feed the discriminator
        output_score_real = D(imgs)
        loss_real_d = real_loss(output_score_real, True, device)
        # now generate a new image and calculate the fake loss
        batch_size = imgs.size(0)
        z_vec = np.random.uniform(-1, 1, size=(batch_size, z_size))
        z_tensor = torch.from_numpy(z_vec).float().to(device)
        fake_img, fc_raw, emd_raw = G(z_tensor)
        # print('fake',fake_img.shape)
        output_score_fake = D(fake_img)
        loss_fake_d = fake_loss(output_score_fake, False, device)
        loss_d = loss_real_d + loss_fake_d

        optimizer_d.zero_grad()
        loss_d.backward()
        optimizer_d.step()

        # now the generator
        # create a new image
        z_vec = np.random.uniform(-1, 1, size=(batch_size, z_size))
        z_tensor = torch.from_numpy(z_vec).float().to(device)
        fake_img, fc_raw, emd_raw = G(z_tensor)

        if i == 1:
            print(f'g: {fake_img.shape}')

        # lets classify by D
        output_score_fake_tolooklikereal = D(fake_img)
        # now lets make D think it was real,
        loss_g = real_loss(output_score_fake_tolooklikereal, True, device)

        # log_fc = F.log_softmax(fc_raw)
        # log_emb = F.log_softmax(embedding_raw)
        #loss_ie  = (torch.mm(G.ie._modules['embedding'].weight, G.fc.weight ).mean().log())
        # here we try to minize the output discrepency /difference between the input_latent space
        # and the latent space provided by the emd_raw. if they become, one,
        # this means they have resided to the same embeddings in their weights!
        # I should also freeze the fc.weight somehow! this may prevent from corrupting
        # the weights we are trying to maintaine!
        # but fc also needs to be updated as the image should be generated and look real!
        loss_ie = criterion(z_tensor.detach(), emd_raw)
        if i == 1:
            print(emd_raw.shape)
            print(z_tensor.shape)
            print(f'{loss_ie}')

        #loss_ie = criterion(log_fc, log_emb)
        # or multiply them by together and mean that as loss as the should be the same!
        loss_g_e = loss_g + loss_ie

        optimizer_g.zero_grad()
        loss_g_e.backward()
        optimizer_g.step()

        if i % interval == 0:
            print(f'epoch {e}/{epochs} iter: {i} loss_d: {loss_d.item()} loss_g: {loss_g.item()} loss_ie: {loss_ie.item()} loss_g_ed: {loss_g_e.item()}')
            break

    losses.append((loss_d.item(), loss_g.item(),
                   loss_ie.item(), loss_g_e.item()))
    with torch.no_grad():
        G.eval()
        sample, fc_raw, emb_raw = G(z_vec_tensor_fixed)
        samples.append(sample)

# save the model to the disk
with open(f'embd_gan_{epochs}_{z_size}_{conv_dim}.t', 'wb') as f:
    states = {"epochs": epochs,
              "z_size": z_size,
              "conv_dim": conv_dim,
              "states_g": G.state_dict(),
              "lr": lr}
    torch.save(states, f)
# sampe images as pickle files
with open(f'embd_imgs_gan_{epochs}_{z_size}_{conv_dim}.pkl', 'wb') as f:
    pkl.dump(samples, f)
print('training done!')
# %%
losses = np.array(losses)
plt.plot(losses.T[0], label='loss_d')
plt.plot(losses.T[1], label='loss_g')
plt.plot(losses.T[2], label='loss_em')
plt.plot(losses.T[3], label='loss_g_e')
plt.legend()
plt.show()


# %%
# lets visualize our samples in each epoch
def vis_samples(samples, title):
    fig, axes = plt.subplots(4, 4, sharex=True, sharey=True)
    samples = samples.cpu().detach().numpy().transpose(0, 2, 3, 1)

    for ax, img in zip(axes.flatten(), samples):
        # rescale to pixel range (0-255)
        img = ((img + 1)*255 / (2)).astype(np.uint8)
        # print(img.shape)
        ax.imshow(img.reshape((64, 64, 3)))
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.set_title(title)


with open(f'embd_imgs_gan_{epochs}_{z_size}_{conv_dim}.pkl', 'rb') as f:
    samples = pkl.load(f)

for i in range(len(samples)):
    vis_samples(samples[i], str(i))
# %%
# morphing test


def morph_sample(G, z_vec_numpy, count=10, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):

    fig = plt.figure(figsize=(10, 2))

    for i in range(count):
        # what we subtract or add here, to the latent sapce
        # directly results in new image. for example 0.5 seems to add the glasses
        # and the multipicant factor seems to rotate the image to sides!!! im not sure!
        # if we add or subtract vectors, I guess we can get much better results! this means embeddingsin images!
        z = z_vec_numpy - 0.5 * (i*0.2)
        z = torch.from_numpy(z).float().to(device)
        imgs, _, _ = G(z)
        imgs = imgs.detach().to('cpu').numpy().squeeze().transpose(1, 2, 0)
        imgs = ((imgs + 1)*255 / 2).astype(np.uint8)
        ax = fig.add_subplot(2, count//2, i+1, xticks=[], yticks=[])
        ax.imshow(imgs)


        # plt.show()
z = np.random.uniform(-1, 1, size=(1, 100))
morph_sample(G, z, count=20)

# %%


def denormal(img):
    img = img.detach().to('cpu').numpy().squeeze()
    img = ((img + 1)*255 / 2).astype(np.uint8).transpose(1, 2, 0)
    return img


def gen_tensor(factor=1):
    z = np.random.uniform(-1, 1, size=(1, 100))
    z = torch.from_numpy(z).float().to(device)
    return z*factor


def vis(img_list, row=1, col=3):

    fig = plt.figure()
    for i, img in enumerate(img_list):
        ax = fig.add_subplot(row, col, i+1, xticks=[], yticks=[])
        img = denormal(img)
        ax.imshow(img)


def tsne_vis(vec1):
    # now lets visualize these embeddings in tsne
    import matplotlib.pyplot as plt
    # %matplotlib inline
    %config InlineBackend.figure_format = 'retina'

    from sklearn.manifold import TSNE
    # getting the embeddings
    #embeddings_numpy = model_10.embeddingLayer.weight.to('cpu').detach().numpy()
    vec1 = vec1.detach().to('cpu').numpy()
    vis_words = 100
    tsne = TSNE()
    embed_tsne = tsne.fit_transform(vec1)

    fig, ax = plt.subplots(figsize=(16, 16))
    ax.set_facecolor("white")
    for idx in range(vis_words):
        plt.scatter(*embed_tsne[idx, :], color='blue')
        plt.annotate(
            idx, (embed_tsne[idx, 0], embed_tsne[idx, 1]), color='black', alpha=0.9)


def vis_latentspace(G, z_size, row=10, col=10, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    fig = plt.figure(figsize=(10, 10))
    for i in range(z_size):
        z = torch.zeros(size=(1, z_size)).to(device)
        z[0, i] = 1
        img, _, _ = G(z)
        ax = fig.add_subplot(row, col, i+1, xticks=[], yticks=[])
        img = denormal(img)
        ax.imshow(img)


vis_latentspace(G, z_size)

# %%


def get_sum(em):
    return np.round(em.detach().cpu().numpy().sum())


def round_em(em):
    return np.round(em.detach().cpu().numpy())


def vis_latentspace2(G, z, row=10, col=10, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    fig = plt.figure(figsize=(col+4, row+4))
    old_value = None
    for i in range(z.shape[1]):
        #z = torch.zeros(size=(1, z_size)).to(device)
        old_value = z[0, i].item()
        z[0, i] = 1
        z_sum = get_sum(z)
        img, _, em = G(z)
        z[0, i] = old_value
        ax = fig.add_subplot(row, col, i+1, xticks=[], yticks=[])
        img = denormal(img)
        ax.imshow(img)
        ax.set_title(str(get_sum(em))+':'+str(z_sum))


z3 = torch.zeros(size=(1, 100)).to(device)
z3[0, 0] = 1

z1 = gen_tensor(0)
z2 = gen_tensor(-0.5)
z3 = gen_tensor(0.5)
vis_latentspace2(G, z1, row=10, col=10)
vis_latentspace2(G, z2, row=10, col=10)

img1, fc1, em1 = G(z1)
img2, fc2, em2 = G(z2)
img3, fc3, em3 = G(z3)

print(z1.sum())
print(np.round(em1.detach().cpu().numpy().sum()))
#img1, fc1, em1 = G(z1)
# z2=gen_tensor()
#img2, fc2, em2 = G(z2)
vis([img1, img2, img3])

img1, fc3, em1 = G(em1)
img2, fc2, em2 = G(em2)
img3, fc3, em3 = G(em3)
vis([img1, img2, img3])

em4 = em3 - em1

print('em1', round_em(em1), get_sum(em1))
print('em3', round_em(em3), get_sum(em3))
print('em4', round_em(em4), get_sum(em4))

img4, fc4, em4 = G(em4)
vis([img1, img3, img4])

# tsne_vis(z1)
# tsne_vis(em1)

# مشکلی که داریم اینجا اینه که فیچرها انتنگلد هستند و جدا از هم نیستن
# باید بجای اینکه یه لایه فولی کانکتد اول باشه و فیچرها قرو قاطی باشن چندتا داشته باشیم .
# ضمنا اون لتند وکتور که اول به جنریتور میدیم اون فیچرها نیست!
# در اصل اون 2048 که تو بعد دوم هست و بعد ریشیپ میکنیم بعنوان یه توده 4در4  اون فیچرهای اصلیه
# اون 100 ورودی رو 100 سطر در نظر بگیر  که هر سطرش یه فیچر با اندازه 2048 داره
# و اون یه کانسپت یا مجموعه ای از چند کانسپت رو کنترل میکنه
# برای همین وقتی مثلا یه وکتور 0 بدیم و فقط یه المانش 1 باشه
# یعنی سطر متناظرش اون فیچرها فعال باشن
# به همین شکل وقتی کلا صفر باشه میبینیم یه تصویر یکسان داریم یا میانگین
# و وقتی مقادیر رندوم باشن یعنی هر سطر به یک میزان دخالت داده میشه
# اگه ما بخواییم یه امبدینگ درست بکنیم
# باید بجای اینکه 100 سطر داشته باشیم مثل استایل گن کار کنیم
# یعنی چندتا فولی کانکتد داشته باشیم که هر کدوم مثلا یه ویژگی یا مجموعه ویژگی رو
# بما بدن  که بتونیم براحتی با فعال غیرفعال کردناونها به چیزی که میخواییم برسیم
# همینطور بجای اینکه 100 بزاریم مثلا میزاریم 1 در فلان
# تا ببینیم میتونیم هر کدوم رو  اینطور یه امبدینگ کنیم یا نه.
# ضمنا مقاله 302.pdf
# حتما خونده بشه عالیه نکاتش. یکبار خوندم ولی باید نوت برداری بشه.
# نکات عالی در مورد ترینینگ گن و بحث مین مکس میگه
# هر وقت جنریتور خوب کار بکنه لاسش بیاد پایین دیسکرممنیتور لاسش میره بالا
# و همینطور وقتی لاس دیکسرمنیتور میره بالا یعنی باید تغییر کنه و اشتباه تشخیص داده .
# این باعث میشه دوباره لاس جنریتور بره بالا و این سیکل ادامه پیدا میکنه
# و کم کم لاس هر دو کم میشه ولی این مین ماکس هست تا جایی که
# جنریتور انقدر خوب بشه که دیسکرمنتیور نتونه تشخیص بدیه .
# بعد میشه تشخیص رندوم یک بار درست تشخیص میده یکبار غلط
# چون فرقی بین خروجی جنریتور و داده اصلی نیست و لاسش بین .0.3
# و همین حدودا قرار میگیره
# اگه لاس دیسکرمنیتور 0 شد یا نیزدیک اون یعنی بدرستی داره همه چیز رو تشخیص میده یعنی
# جنریتور داغونه و یه مشکلی داره که دیسکرمنیتور هی میفهمه این فیکه . باید اونو اکی کرد.
# ضمنا گرادیان از دیکسرمنیتور میاد پس وجود لیکی رلو تو دیسکرمنیتور خیلی کمک کننده اس
# تو لایه اول دیسکرمنیتور نباید بچ نرمالیزیشن گذاشت چون مین و واریانس تصویر ورودی رو خراب میکنه
# به همین صورت لایه اخر جنریتور هم نباید گذاشت چون تصویر تولید میشه و بچ نرمالیزیشن خرابش میکنه
#  اون مقاله خونده بشه حتما!!!


# %%


# %%
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


# %%
#

# %%
# %%
