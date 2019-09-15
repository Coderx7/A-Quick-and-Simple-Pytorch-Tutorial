#%%
# in the name of God the most compassionate the most merciful
# in this part, we are going to learn about autoencoders and 
# how we can implement them in Pytorch. 
# Autoencoders are a kind of networks that map their input
# to a new representation. this is usually refered to as 
# compressing the input into a latent space representation,
# this means, they accept the data, and then downsample it
# until they reach a specified/suitable size feature vector
# and then upsample that feature vector gradually until they
# reach to the original size, and then try to reconstruct the
# input(so our input image acts as a label as well!). 
# during this process of reconstructing the input data
# from the compressed representation, the new representation is
# developed and can be used for various applications. 
# the first part of the network that downsamples the input data
# into a feature vector is called an "Encoder", and the part that
# reconstructs the input from the mentioned feature vector is called
# a "Decoder". 
# when we have successfully trained our autoencoder, we can use its
# new representation instead of our new data. so we can use it for 
# dimensionality reduction just like PCA (if linear) and much more 
# powerful than that when using a deep nonlinear autoencoder! 
# we can use the new representation for lots of applications including
# sending /storing the reduced representation instead of the full input
# and reconstruct the input using the representation, this will result 
# in a considerable reduction in network traffic or space required to store
# the actual data. 
# The usage is not limited to such usescases, we can get fancy and creative 
# for example and make a black and white image , color again! or denoise our input
# reconstruct missing parts, create new data/images, visualizations, etc!
# there are lots and lots of use cases for autoencoders
# However, note that, the notion of compression spoke here is different than that of
# what you find in different media formats such as jpeg, mp3, etc. 
# Autoencoders do not work well on unseen data and thus usually have difficulties 
# generalizing well to unseen data. more on this later  
# There are different kinds of Autoencoders, they can be linear, or
# nonlinear, shallow, or deep, convolutional, or not, etc
# we will cover some of the most famous variants here. 
# lets start
import numpy as np 
import torch
import torchvision
from torchvision import datasets, transforms
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim
import matplotlib.pyplot as plt 
%matplotlib inline


#%%
# We will use MNIST dataset for our experiments here. 
# Lets get back to our discussion! 
# We mentioned couple of examples/usecases for autoencoders, but why do we have to
# shrink the size in the encoder part ? why do we gradually reduce the input size until
# we reach a feature vector of some size? 
# shrinking the size gradually, acts as a imposing a constraint on the input
# by doing so, we are forcing the network to choose the important features in 
# our input data, the features that has the essence of our input data and can later
# be used to reconstruct the input. This is why the new resuling representation works 
# very well and can be used instead of the input for some applications. The new repres
# -entation simply has the most important features in the input. 
# if such constraint was not present, the network would not be able to learn anything
# meaningful about the distribution of our input data and thus the resulting vector
# would be of no use to us. So we should be shrinking the input until we reach 
# a certain size of our liking(based on our usage) 
# Note that, these new features may not be individually interpretable specially in the
# case of nonlinear deep autoencoders. there are ways to see what and how a specific 
# feature in the resulting feature vector responds to different attributs present in
# an input but never make this mistake that e.g. the 10 features in the bottleneck layer
# (our feature vector) represents the exact attributes in your input. these features
# may represent a complex interactions between several features that define a 
# characteristic in you data. anyway we'll get to this later on. 

#%% Ok, enough talking lets get busy and have our first auto encoder. 
# before we continue, we should pickup a dataset. I chose MNIST as its simple enough
# to be used in different types of autoencoders with quick training time. 
# after we created our dataset, we will implement different types of AutoEncoders 
dataset_train = datasets.MNIST(root='MNIST',
                               train=True,
                               transform = transforms.ToTensor(),
                               download=True)
dataset_test  = datasets.MNIST(root='MNIST', 
                               train=False, 
                               transform = transforms.ToTensor(),
                               download=True)
batch_size = 32
num_workers = 2
dataloader_train = torch.utils.data.DataLoader(dataset_train,
                                               batch_size = batch_size,
                                               shuffle=True,
                                               num_workers = num_workers)

dataloader_test = torch.utils.data.DataLoader(dataset_test,
                                               batch_size = batch_size,
                                               num_workers = num_workers)

# lets view a sample of our images 
def view_images(imgs, labels, rows = 3, cols =11):
    # images in pytorch have the shape (channel, h,w) and since we have a
    # batch here, it becomes, (batch, channel, h, w). matplotlib expects
    # images to have the shape h,w,c . so we transpose the axes here for this!
    imgs = imgs.detach().cpu().numpy().transpose(0,2,3,1)
    fig = plt.figure(figsize=(11,4))
    for i in range(imgs.shape[0]):
        ax = fig.add_subplot(rows, cols, i+1, xticks=[], yticks=[])
        # since mnist images are 1 channeled(i.e grayscale), matplotlib
        # only accepts these kinds of images without any channesl i.e 
        # instead of the shape 28x28x1, it wants 28x28
        ax.imshow(imgs[i].squeeze(), cmap='Greys_r')
        ax.set_title(labels[i].item())


# now lets view some 
imgs, labels = next(iter(dataloader_train))
view_images(imgs, labels)

,# good! we are ready for the actual implementation
#%% 
# The first autoencoder weare going to implement is the simplest one, 
# a linear autoencoder.
# creating an autoencoder is just like any other module we have seen so far, simply
# inherit from nnModule and define the needed layers and call them in the forward()
# method the way you should. lets do this :
class LinearAutoEncoder(nn.Module):
    def __init__(self, embedingsisze=32):
        super().__init__()
        # lets define our autoencoder we have two parts, an encoder 
        # and a decoder. 
        # the encoder shrinks the input gradually until it becomes
        # a certain size, and the decoder accepts that as input and
        # gradually upsamples it to reach the actual input size. 
        
        # The encoder part: 
        # So our encoder part simply is a linear 
        # layer, or a fully connected layer that
        # accepts the input. since this is a linear layer,
        # we have to flatten the input and our 28x28 image
        # will simply have 28x28=784 input features 
        # The simplest form can be an a one layered encoder
        # and a 1 layered decoder! of course we can add more
        # layers between them, but lets see how this performs
        self.fc1 = nn.Linear(28*28, embedingsisze)
        # our decoder part
        self.fc2 = nn.Linear(embedingsisze, 28*28)

    def forward(self, inputs):
        # our foward pass is nothing specially
        # simply feed these layers in order!
        # but before that, we must flatten our input!
        inputs = inputs.view(inputs.size(0), -1)
        # encoder part
        output = self.fc1(inputs)
        # decore part
        output = self.fc2(output)
        # since in the output we want an image not a flattened
        # evctor, we reshape our input again!
        output = output.view(-1, 1, 28, 28)
        return output 


model_linear_ae = LinearAutoEncoder()
print(model_linear_ae)
#%%
# now lets train our model. 
# since we compare the output of our network with our input
# we use MSELoss for this. 
criterion = nn.MSELoss()
def train(model, dataloader, optimizer, scheduler, epochs, device):
    for e in range(epochs):
        # we dont need label so we use _ as its a convention
        for i, (imgs,_) in enumerate(dataloader):
            imgs = imgs.to(device)
            reconstructed_images = model(imgs)
            loss = criterion(reconstructed_images, imgs)        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i% 2000==0:        
                print(f'epoch: ({e}/{epochs}) loss: {loss.item():.6f} lr:{scheduler.get_lr()}')
        scheduler.step()
    print('done')

# Now lets see the output of our autoencoder
def test(model,device):
    imgs, labels = next(iter(dataloader_test))
    imgs = imgs.to(device)
    outputs = model(imgs)
    view_images(outputs, labels)
#%%

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_linear_ae = model_linear_ae.to(device)
optimizer = optim.Adam(model_linear_ae.parameters(), lr = 0.1) 
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5)

train(model_linear_ae, dataloader_train, optimizer, scheduler, 20, device) 
test(model_linear_ae, device)    
# so this is the linear autoencoder! in order to make a vanila autoencoder
# which may refer to a version with nonlinear activation functions, you 
# only need to apply a transformation function .
# in the fowarad pass and in order  to get a good result, you need to add a few more 
# layers .(we do this in the next architecture )
# we can get better results with more epochs and decaying learnng rate,
#  but it wont make a drastic change! specially on more complex data, as its 
# just a linear model.
#%%
# in order to be able to capture more complex structures,... in  the input data
# one way is to add more hidden layers! so 
# Now lets create a multi layer auto encoder!
class MLPAutoEncoder(nn.Module):
    def __init__(self, embedingsisze=32):
        super().__init__()

        self.fc1 = nn.Linear(28*28, 64)
        self.fc2 = nn.Linear(64, embedingsisze)
        # our decoder part
        self.fc3 = nn.Linear(embedingsisze, 64)
        self.fc4 = nn.Linear(64, 28*28)


    def forward(self, inputs):
        inputs = inputs.view(inputs.size(0), -1)
        # encoder part
        output = F.relu(self.fc1(inputs))
        output = F.relu(self.fc2(output))
        # decore part
        output = F.relu(self.fc3(output))
        # since the output is images, values should 
        # be in the range [0, 1]!
        output = F.sigmoid(self.fc4(output))
        output = output.view(-1, 1, 28, 28)
        return output 

model_mlp_ae = MLPAutoEncoder().to(device)
print(model_mlp_ae)

# criterion = nn.MSELoss()
optimizer = optim.Adam(model_mlp_ae.parameters(), lr = 0.01) 
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5)
train(model_mlp_ae, dataloader_train, optimizer, scheduler, 20, device)    
test(model_mlp_ae,device)  
#%%
# While our mlp model is more powerful than the previous model, it is not suitable for data such as images
# for image like data, we use conv layers! and hence our new autoencoder is Convolutional AutoEncoder. 
# lets see how to implement this : 
# something that needs to be said is that, when the number of layers is increased, i.e. 
# your network gets deeper, you may see that your model may train sometimes and not the 
# other times and the loss may not decrease. when you see this, you should know this is
# happening becasue of the depth of your network.  use the batchnorm and all will be good. 
# thats why I created two functions for this very purpose. try creating your network with
# and without batchnormalization enabled and see the difference (try running for several 
# times with the one with no batchnormalization to see that sometimes it may work and some 
# times it will fail, but with batchnorm, it will always work!)
def conv_bn(in_,out_,k_size=3, s=2,pad=0,bias=False,batchnorm=True):
    layers = []
    layers.append(nn.Conv2d(in_,out_,kernel_size=k_size,stride=s,padding=pad,bias=bias))
    if batchnorm:
        layers.append(nn.BatchNorm2d(out_))
    return nn.Sequential(*layers)

def deconv_bn(in_,out_,k_size=4, s=2,pad=0,bias=False,batchnorm=True):
    layers = []
    layers.append(nn.ConvTranspose2d(in_,out_,kernel_size=k_size,stride=s,padding=pad,bias=bias))
    if batchnorm:
        layers.append(nn.BatchNorm2d(out_))
    return nn.Sequential(*layers)

class ConvAutoEncoder(nn.Module):
    def __init__(self, embedingsize=32):
        super().__init__()
        # for conv layers, since we are dealing with 3d featuremaps
        # we shrink the number of featuremaps as well as the spatial
        # dimensions. we do so until we reach a size that satifies us
        # 
        self.conv1 = conv_bn(1, 256, 3, 1) # each stride 2 downsamples the dimensions by half
        self.conv2 = conv_bn(256, 128, 3, 2) # 14 
        self.conv3 = conv_bn(128, 64, 3, 2)  # 7 
        self.conv4 = conv_bn(64, embedingsize, 3, 2) # output is 64x2x2
        # decoder 
        # now for decoder we have two options, we can simply use a conv layer 
        # followed by a upsample layer. or we can use a deconv layeror  a 
        # transposed convolution layer. the difference between them is that
        # using the transposedconv approach, results in a checkerboard effect
        # while thats not the case for upsample method!
        # we use k=2,s=2, as it upsamples the image 2x. 
        # from there we can use different kernel size, strides 
        # to reach to desired dimensions. 
        self.deconv5 = deconv_bn(embedingsize, 64, 2, 2) 
        self.deconv6 = deconv_bn(64, 128, 4, 2) 
        self.deconv7 = deconv_bn(128, 256, 5, 2)
        # and since our image is 1 channel, this last layer will produce a singe image!
        self.conv8 = deconv_bn(256, 1, 6, 1,0,True,False)
         

    def forward(self, x):
        output = F.relu(self.conv1(x))
        output = F.relu(self.conv2(output))
        output = F.relu(self.conv3(output))
        output = F.relu(self.conv4(output))
        output = F.relu(self.deconv5(output))
        output = F.relu(self.deconv6(output))
        output = F.relu(self.deconv7(output))
        # since we want an image, we use sigmoid
        output = F.sigmoid(self.conv8(output))
        return output


#%%
# now lets train it
model_c= ConvAutoEncoder()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
optimizer = optim.Adam(model_c.parameters(), lr =0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5)
model_c = model_c.to(device)
train(model_c, dataloader_train, optimizer, scheduler, 20, device)    
test(model_c, device)  
# As an excersize try to replace all ConvTranspose2d Layers with Conv2d+Upsample
# and see how the outputs turn out !
#%% 
# Now lets create more powerful Convolutional AutoEncoders. the vanial convolutional autoencoder
# is not that powerful. therefore we can use several variants such as:
# denoising autoencoder, Sparse autoencoder, variational autoencoder

# using denoising autoencoder, our archietcture needs to be deep enough because its
# a more complex taks. however, our previous convautoencoder is deep enough so we can 
# use that here as well. 
# basically in denoising autoencoder, we feed a noisy image and get a noise free image
# so what we will actually do in the training process is to add random noise to our image
# prior to feeding it to our model and then compare the reconstructed image with the actual
# original image which is noise free. in doing this, network will learn to remove noise from
# images. we will use the same criterion. nearly 99% of what we saw until now is the same 
# and we just will add a simplenoise lets see that 
noise_threshold = 0.5
epochs = 20
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# the quality and performance of our model in denoising will increase as we
# increase the embedding size. 
model = ConvAutoEncoder(1).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001) 
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5)

# before we go on lets view a sample of noisy images : 
imgs,labels = next(iter(dataloader_test))
imgs = imgs + noise_threshold * torch.rand_like(imgs)
imgs.clamp_(0,1)
view_images(imgs,labels)

print(model)
for e in range(epochs):
    loss_epoch = 0.0
    for imgs,_ in dataloader_train:
        imgs = imgs.to(device)

        #apply noise to our image 
        imgs_noisy = imgs + noise_threshold * torch.rand_like(imgs)
        # clip all values outside of 0,1 becasue our image values 
        # should be in this range!
        imgs_noisy = imgs_noisy.clamp(0,1)
        imgs_recons = model(imgs_noisy)

        loss = criterion(imgs_recons, imgs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_epoch += loss.item()
    print(f'epoch: {e}/{epochs} loss: {loss.item()} lr: {scheduler.get_lr()}')
    scheduler.step()

# lets see how the network does on noisy image!
imgs,labels = next(iter(dataloader_test))
imgs = imgs.to(device)
imgs = imgs + noise_threshold * torch.rand_like(imgs)

imgs.clamp_(0,1)
view_images(imgs,labels)
new_noise_free_imgs = model(imgs)
view_images(new_noise_free_imgs,labels)
#%%
# you may ask, so far we have been starting with a large number of channels, 
# and gradually decreased and at the same time shrunk the spatial extend, what if we do
# the opposite, we begin with few channels and large spatial extend
# and then gradually increase the channels and shrink the spatial dimension until
# you reach a large vector representation with little or no spatial extent. and in the 
# decoder, we do the opposite obviously! lets see how that performs! (tldr !it performs worse!)
class ConvolutionalAutoEncoder_v2(nn.Module):
    def __init__(self, embeddingsize=32):
        super().__init__()
        self.encoder = nn.Sequential(conv_bn(1, 32, 3, 1),
                                conv_bn(32, 64, 3, 2),
                                conv_bn(64, 128, 3, 2),
                                conv_bn(128, embeddingsize, 3, 2))

        self.decoder = nn.Sequential(deconv_bn(embeddingsize, 128, 2, 2),
                                deconv_bn(128, 64, 4, 2),
                                deconv_bn(64, 32, 5, 2),
                                deconv_bn(32, 1, 6, 1,batchnorm=False))
    def forward(self, inputs):
        output = self.encoder(inputs)
        return self.decoder(output)
                               

noise_threshold = 0.5
epochs = 20
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# the quality and performance of our model in denoising will increase as we
# increase the embedding size. 
model = ConvolutionalAutoEncoder_v2(32).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001) 
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5)

# before we go on lets view a sample of noisy images : 
imgs,labels = next(iter(dataloader_test))
imgs = imgs + noise_threshold * torch.rand_like(imgs)
imgs.clamp_(0,1)
view_images(imgs,labels)

print(model)
for e in range(epochs):
    loss_epoch = 0.0
    for imgs,_ in dataloader_train:
        imgs = imgs.to(device)

        #apply noise to our image 
        imgs_noisy = imgs + noise_threshold * torch.rand_like(imgs)
        # clip all values outside of 0,1 becasue our image values 
        # should be in this range!
        imgs_noisy = imgs_noisy.clamp(0,1)
        imgs_recons = model(imgs_noisy)

        loss = criterion(imgs_recons, imgs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_epoch += loss.item()
    print(f'epoch: {e}/{epochs} loss: {loss.item()} lr: {scheduler.get_lr()}')
    scheduler.step()

# lets see how the network does on noisy image!
imgs,labels = next(iter(dataloader_test))
imgs = imgs.to(device)
imgs = imgs + noise_threshold * torch.rand_like(imgs)

imgs.clamp_(0,1)
view_images(imgs,labels)
new_noise_free_imgs = model(imgs)
view_images(new_noise_free_imgs,labels)
#%%
# sparse autoencoder: these kinds of autoencoders simply use a regularizer term so that
# the features are more sparse! usually l1 loss is used! 
#  In the previous examples, the representations were only constrained by the size of the
# hidden layers. In such a situation, what typically happens is that the hidden layer is
# learning an approximation of PCA (principal component analysis).
# But another way to constrain the representations to be compact is to add a sparsity 
# contraint on the activity of the hidden representations, so fewer units would "fire" 
# at a given time.
# in order to have sparsity, we need to have overcomplete representations. so lets 
# implement a sparse autoencoder in this section and see how it performs. 
# as I said earlier, aside from the normal reconstruction loss, we need a new regularizer
# lets create this regularizer now. We are going to create a Function object that applies
# l1penalty we inherit from autograd.Function class for this. 
import copy # sed for deep copy of our weights
from torch.autograd import Function  # used for implementing l1_lenalty 
class L1Penalty(Function):
    # we override the forward method with our own arguments (input, l1_weight)
    # input is the input obviously and l1_weight is the percentage of zero weights
    # that is 0.1 means, we want 10% of our weights to be zero (or near zero)
    # or sparsity ratio if you will!
    # In the forward pass, we simply save our input and l1_weight for use in backwardpass
    @staticmethod
    def forward(ctx, input, l1_weight):
        ctx.save_for_backward(input)
        ctx.l1_weight = l1_weight    
        return input

    #   backward must accept a context `ctx` as the first argument, followed by
    #   as many outputs did `forward` return, and it should return as many
    #   tensors, as there were inputs to `forward`. 
    #   Each argument is the gradient w.r.t the given output, and each returned
    #   value should be the gradient w.r.t. the corresponding input.

    #   The context can be used to retrieve tensors saved during the forward
    #   pass. It also has an attribute ctx.needs_input_grad` as a tuple
    #   of booleans representing whether each input needs gradient. E.g.,
    #   `backward` will have ``ctx.needs_input_grad[0] = True`` if the
    #   first input to `forward` needs gradient computated w.r.t. the
    #   output.
    @staticmethod
    def backward(ctx, grad_outputs):
        input, = ctx.saved_tensors
        # since we only need gradients with respect to the input
        # we need to explicitly say we dont need gradienst to be 
        # calculated for our second argument term in forward method 
        # i.e. l1_weight. so we return None for other arguemnst that
        # we dont want any gradient. 
        # this is a term that we apply in the backward pass, 
        # that is, we are enforcing the constraint by adding 
        # a new term to the gradient 
        grad_input = input.clone().sign().mul(ctx.l1_weight)
        grad_input +=grad_outputs
        # since we have two inputs in our foward pass, we need to
        # provide two gradients in the backward pass. but becasue
        # we only care about input and not the l1_weight, (we dont)
        # need any gradients for it becsaue we are not tuning that!
        # we return None
        return grad_input, None

# now lets create our architecture 
class SparseAutoEncoder(nn.Module):
    def __init__(self, embeddingsize=400, l1_weight =0.1, use_l1_penalty = False, tied_weights = False):
        super().__init__()
        self.l1_weight = l1_weight
        self.use_l1_penalty = use_l1_penalty
        self. tied_weights = tied_weights

        self.encoder = nn.Sequential(nn.Linear(28*28, embeddingsize),
                                    nn.Tanh())# or relu
        self.decoder = nn.Sequential(nn.Linear(embeddingsize, 28*28),
                                    nn.Sigmoid())
        # you may see some people, use the shared weights between encoder
        # and decoder, i.e. decoder uses the transposed weightmatrix of the 
        # encoder. for doing this  there are couple of ways. 
        # one of way is to use the functional form and simply 
        # use one weight and its transpose like this 
        # weight = nn.Parameter(torch.rand(input_dim, output_dim))
        # self.encoder = F.linear(input, weight, bias=False)
        # self.decoder = F.linear(input, weight.t(), bias=False)
        # we can also simply define our new weight and assigne it to both modules
        if self.tied_weights:
            weights = nn.Parameter(torch.randn_like(self.encoder[0].weight))
            self.encoder[0].weight.data = weights.clone()
            self.decoder[0].weight.data = self.encoder[0].weight.data.transpose(0, 1)
        

    def forward(self, input):
        input = input.view(input.size(0), -1)
        sparsity_loss = 0.0
        output = self.encoder(input)
        sparsity_loss += torch.mean(abs(output))
        # apply the l1penalty on the weights of our encoder
        # through added term in backpropagation
        if self.use_l1_penalty:
            output = L1Penalty.apply(output, self.l1_weight)
        output = self.decoder(output)
        #sparsity_loss += torch.mean(abs(output))
        output = output.view(input.size(0), 1, 28, 28)
        return output, sparsity_loss

# important note: 
# Please note that here we have two way for enforcing the sparsity constraint
# using the L1Penalty Loss, we are directly enforcing the sparsity constraint
# on the weights using the gradients. This is has a more prominent effect as 
# you will see in the weight visualizations below. 
# However, calculating the sparsity loss from layers output, enforces the 
# constraint on the activations of the model and then indirectly affects the
# weights. 
# becasue of this, the way we optimize our network will be different in the sense
# that we need to use different hyper parameters to get similar outcomes. 

#%%
epochs = 50
use_l1_penalty = False
sparsity_ratio = 0.1
criterion = nn.MSELoss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sae_model = SparseAutoEncoder(embeddingsize=400,
                             l1_weight=sparsity_ratio,
                             use_l1_penalty=use_l1_penalty,
                             tied_weights=True).to(device)
optimizer = torch.optim.Adam(sae_model.parameters(), lr = 0.1) 
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10) 

print(sae_model)        
# lets save the weights of our encoder and decoders before we train them 
# and then compare them with the new weights after training and see how
# they changed!
init_weights_encoder = copy.deepcopy(sae_model.encoder[0].weight.data) 
init_weights_decoder = copy.deepcopy(sae_model.decoder[0].weight.data)
imgs_list =[]
# now lets start training ! 
for e in range(epochs):
    for imgs,_ in dataloader_train:
        imgs = imgs.to(device)
        output, sparsity_loss = sae_model(imgs)
        loss = criterion(output, imgs)
        if use_l1_penalty:
            loss_f = loss
        else:
            loss_f = loss + (sparsity_ratio * sparsity_loss)
        optimizer.zero_grad()
        loss_f.backward()
        optimizer.step()
    print(f'epoch: {e}/{epochs} loss_f: {loss_f.item():.6f} loss: { loss.item():.6f}'\
          f' sparsity loss: {sparsity_loss.item():.6f} lr = {scheduler.get_lr()}')
    scheduler.step()
    # at each epoch, we sample one image and its reconstruction
    # for viewing later on to see how the training affects the
    # result we get
    imgs_list.append((imgs[0],output[0]))
#%% 
# now lets first visualize the image/reconstruction pairs and how they look : 
def visualize(imgs_list, rows=5, cols=10):
    fig = plt.figure(figsize=(15,2))
    plt.subplots_adjust(wspace=0,hspace=0)
    
    print(f'number of samples: {len(imgs_list)}')
    for i in range(len(imgs_list)):
        img,recons = imgs_list[i]
        # print(img.shape,recons.shape)
        img = img.cpu()
        recons = recons.cpu().detach()
        ax = fig.add_subplot(rows, cols, i+1, xticks=[], yticks=[])
        x = torchvision.utils.make_grid([img,recons])
        ax.imshow(x.numpy().transpose(1,2,0))

visualize(imgs_list)

# Now lets visualize the weights and see how they look. 
# we had the initial weights saved so lets subtract them
# from the trained one and see the diffs , it will show us
# where the changes happened 

def visualize_grid(imgs, rows=20, cols=20):
    fig = plt.figure(figsize=(20, 20))
    imgs = imgs.cpu().numpy().transpose(0, 2, 3, 1).squeeze()
    plt.subplots_adjust(wspace=0, hspace=0)
    for i in range(imgs.shape[0]):
        ax = fig.add_subplot(rows, cols, i+1, xticks=[], yticks=[])
        ax.imshow(imgs[i], cmap='Greys_r')


def visualize_grid2(imgs, label, normalize=True):
    fig = plt.figure(figsize=(10, 10))
    imgs = imgs.cpu()  
    plt.subplots_adjust(wspace=0, hspace=0)
    x = torchvision.utils.make_grid(
        imgs, nrow=20, normalize=normalize).numpy().transpose(1, 2, 0)
    ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[])
    ax.imshow(x)
    ax.set_title(label)


trained_W_encoder = sae_model.encoder[0].weight.data.cpu(
).clone().view(sae_model.encoder[0].out_features, 1, 28, 28)
trained_W_decoder = sae_model.decoder[0].weight.data.cpu(
).clone().view(sae_model.decoder[0].in_features, 1, 28, 28)
init_weights_encoder = init_weights_encoder.view(
    sae_model.encoder[0].out_features, 1, 28, 28).cpu()
init_weights_decoder = init_weights_decoder.view(
    sae_model.decoder[0].in_features, 1, 28, 28).cpu()

w_diff_encoder = init_weights_encoder - trained_W_encoder
w_diff_decoder = init_weights_decoder - trained_W_decoder

print(init_weights_encoder.shape)
visualize_grid2(init_weights_encoder, 'Initial weights')
visualize_grid2(trained_W_encoder, 'Trained weights(Encoder)')
visualize_grid2(w_diff_encoder, 'weights diff (Encoder)')
visualize_grid2(trained_W_decoder,'Trained Weights (Decoder)')
# the black shows negative values, and white show positive values
# and the gray shows zero values.
# we start from a high positive and high negative values in our initial
# weights. and then after training and imposing sparsity we can see that
# we are mostly seeing gray colors which indicate the values are zero!
# and that is what we were after!
# if you look at the w_diff, you can see that there are lots of high and
# low (negative) values as well. this is becsaue  in order to make the
# weights have more reasonable weights, they had to be decreased/increased
#%% 
# the cool thing about autoencoders are that we can use them to pretrain
# our weights on our data and then use that for classification or etc. 
# this was actually done a lot back in the day until 2014/2015. 
# in that era, the use of xavier initialization algorithm accompanied by 
# batchnormalization killed the need for pretraining in this way. but lets 
# see how we can do this if the needs be. 
# its simple, just like finetuning, we may add/remove the layers we want
# here we will remove the decoder part and instead add a classifier
# lets remove the decoder 
layers_before_decoder = list(sae_model.children())[:-1]
sae_model2 = nn.Sequential(*layers_before_decoder)
# since we created a sequential model here, we should add a new module
# using add_module. because if we simplt do sth like : 
# sae_model2.classifier = nn.Linear(sae_model2[0].out_features, 10)
# classifier will be just an attribute, and for the forward pass we 
# need to do sth like 
# output=sae_model2.forward(input)
# output = sae_model2.classifier(output)
# so this is not ideal at all. therefore we do : 
sae_model2.add_module('classifier', nn.Linear(sae_model2[0][0].out_features, 10))
print(sae_model2)
#%% now that we have our model built lets run trainng and pay attention
# what is the first accuracy we get

criterion = nn.CrossEntropyLoss()
epochs = 20 
optimizer = torch.optim.SGD(sae_model2.parameters(), lr=0.1)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5)
acc = 0.0
sae_model2 = sae_model2.to(device)
for e in range(epochs):
    for i, (imgs, labels) in enumerate(dataloader_train):
        imgs = imgs.to(device)
        labels = labels.to(device)

        imgs = imgs.view(imgs.size(0),-1)
        output = sae_model2(imgs)
        loss = criterion(output, labels)
        _,class_idx = torch.max(output,dim=1)
        acc += torch.mean((class_idx.view(*labels.shape) == labels).float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    acc = acc/len(dataloader_train)
    print(f'epoch: ({e}/{epochs}) acc: {acc*100:.4f} loss: {loss.item():.6f} lr: {scheduler.get_lr():.6f}')
    scheduler.step()


# now you can try it without running the autoencoder training and 
# see how it performs.

# Notes: 
# For imposing the sparsity constraint instead of l1 norm, we can
# also use KL divergance the principle is the same, where we took 
# the average of the activations at each layer that we want their 
# weights to be sparse, this time we calculate
# the kl-loss  which is like this : 
# def kl_divergence(p, p_hat):
#     funcs = nn.Sigmoid()
#     p_hat = torch.mean(funcs(p_hat), 1)
#     p_tensor = torch.Tensor([p] * len(p_hat)).to(device)
#     return torch.sum(p_tensor * torch.log(p_tensor) - p_tensor * torch.log(p_hat) + (1 - p_tensor) * torch.log(1 - p_tensor) - (1 - p_tensor) * torch.log(1 - p_hat))

# finally  this was a simple autoencoder, we can have several layers
# and also you can use batchnormalization, etc for your deep autoencoders as well


#%%
# -VAE (Variational Autoencoders) 
# -Creating MNIST Like digits 
# -The Reparametrization Trick
# Variational Autoencoders (VAEs) have one fundamentally unique property that 
# separates them from vanilla autoencoders, and it is this property that makes
# them so useful for generative modeling: their latent spaces are, by design,
# continuous, allowing easy random sampling and interpolation.

# It achieves this by doing something that seems rather surprising at first: 
# making its encoder not output an encoding vector of size n, rather, outputting
# two vectors of size n: a vector of means, μ, and another vector of standard
# deviations, σ
# They form the parameters of a vector of random variables of length n, with 
# the i-th element of μ and σ being the mean and standard deviation of the i-th
# random variable, X_i, from which we sample, to obtain the sampled encoding 
# which we pass onward to the decoder:
# This stochastic generation means, that even for the same input, while the mean
# and standard deviations remain the same, the actual encoding will somewhat vary
# on every single pass simply due to sampling.
# read more  : https://towardsdatascience.com/intuitively-understanding-variational-autoencoders-1bfe67eb5daf
# There are other resouces for this as well. its highly recommened to read them: 
# https://www.jeremyjordan.me/variational-autoencoders/
# https://jaan.io/what-is-variational-autoencoder-vae-tutorial/
# https://www.youtube.com/watch?v=uaaqyVS9-rM
# http://blog.shakirm.com/2015/10/machine-learning-trick-of-the-day-4-reparameterisation-tricks/


# we'll also have an example concerning words(in NLP domain) and see how we can 
# leverage VAEs in that domain as well. for now lets see how we can implement this
# for vision domain. i.e. on mnist dataset

# note: 
# For variational autoencoders, the encoder model is sometimes referred to as
# the 'recognition model' whereas the decoder model is sometimes referred to as 
# the 'generative model'.

# if you havent read the links I gave you, go read them all. each single one of them
# will help you grasp one aspect very good!
#  
# now lets define our VAE model . 
class VAE(nn.Module):
    def __init__(self, embedding=100):
        super().__init__()

        # our encoder will give two vectors one for μ and another for σ.
        # using these two parameter, we sample our z representation vector
        # which is used by the decoder to reconstruct the input. 
        # So we can say that The encoder ‘encodes’ the data which is 784-dimensional
        # into a latent (hidden) representation space z, which is much less than 784
        # dimensions. This is typically referred to as a ‘bottleneck’ because the 
        # encoder must learn an efficient compression of the data into this 
        # lower-dimensional space. Let’s denote the encoder qθ(z∣x). 
        # We note that the lower-dimensional space is stochastic: 
        #>> the encoder outputs  parameters to qθ(z∣x), which is a Gaussian probability
        #   density. 
        # We can sample from this distribution to get noisy values of the 
        # representations z .
        self.fc1 = nn.Linear(28*28, 400)
        self.fc1_mu = nn.Linear(400, embedding) # mean
        # we use log since we want to prevent getting negative variance
        self.fc1_std = nn.Linear(400, embedding) #logvariance

        # our decoder will accept a randomly sampled vector using
        # our mu and std. 
        # The decoder is another neural net. Its input is the representation z,
        # it outputs the parameters to the probability distribution of the data,
        # and has weights and biases ϕ. The decoder is denoted by pϕ(x∣z). 
        # Running with the handwritten digit example, let’s say the photos are 
        # black and white and represent each pixel as 0 or 1. 
        # The probability distribution of a single pixel can be then represented 
        # using a Bernoulli distribution. The decoder gets as input the latent 
        # representation of a digit z and outputs 784 Bernoulli parameters,
        # one for each of the 784 pixels in the image. 
        # The decoder ‘decodes’ the real-valued numbers in z into 784 real-valued 
        # numbers between 0 and 1. Information from the original 784-dimensional 
        # vector cannot be perfectly transmitted, because the decoder only has 
        # access to a summary of the information 
        # (in the form of a less-than-784-dimensional vector z). 
        # How much information is lost? We measure this using the reconstruction 
        # log-likelihood logpϕ(x∣z) whose units are nats. This measure tells us how 
        # effectively the decoder has learned to reconstruct an input image x given
        # its latent representation z.
        self.fc2 = nn.Linear(embedding, 400)
        self.fc2_2 = nn.Linear(400, 28*28)


    # Rather than directly outputting values for the latent state as we would 
    # in a standard autoencoder, the encoder model of a VAE will output 
    # "parameters(mean μ,variance σ) describing a distribution for each dimension in 
    # the latent space". 
    # Since we're assuming that our prior follows a normal distribution, we'll output
    # two vectors describing the mean and variance of the latent state distributions.
    # If we were to build a true multivariate Gaussian model, we'd need to define a
    # covariance matrix describing how each of the dimensions are correlated. 
    # However, we'll make a simplifying assumption that our covariance matrix only 
    # has nonzero values on the diagonal, allowing us to describe this information 
    # in a simple vector.
    # Our decoder model will then generate a latent vector by sampling from these
    # defined distributions and proceed to develop a reconstruction of the original
    # input.
    # However, this sampling process requires some extra attention. When training 
    # the model, we need to be able to calculate the relationship of each parameter 
    # in the network with respect to the final output loss using backpropagation. 
    # However, we simply cannot do this for a "random sampling process". Fortunately,
    # we can leverage a clever idea known as the "reparameterization trick" which 
    # suggests that we randomly sample ε from a unit Gaussian, and then shift the 
    # randomly sampled ε by the latent distribution's mean μ and scale it by the 
    # latent distribution's variance σ.
    # With this reparameterization, we can now optimize the parameters of the distribution
    # while still maintaining the ability to randomly sample from that distribution.
    # Note: In order to deal with the fact that the network may learn negative values
    # for σ, we'll typically have the network learn log(σ) and exponentiate(exp)) this value 
    # to get the latent distribution's variance.
    def reparamtrization_trick(self, mu, logvar):
        if self.training:
            # we divide by two because we are eliminating the negative values
            # and we only care about the absolute possible deviance from standard.
            std = logvar.mul(0.5).exp_()
            #epsilon sampled from normal distribution with N(0,1)
            eps = torch.tensor(std.data.new(std.size()).normal_(0,1))
            # How to sample from a normal distribution with known mean and variance?
            # https://stats.stackexchange.com/questions/16334/ 
            # (tldr: just add the mu , multiply by the var) . why we use an epsilon, ? 
            # because without it, backprop wouldnt work.
            return eps.mul(std).add(mu)
        else:
            # During the inference, we simply return the mean of the
            # learned distribution for the current input.  We could
            # use a random sample from the distribution, but mu of
            # course has the highest probability.
            return mu

    def forward(self, input):
        output = input.view(input.size(0), -1)
        # our encoder recieves the input and produces two vectors
        # mean and std. a normal autoencoder, creates a set of atttibutes
        # in its representation vectot(e.g attributes or features describing
        # concepts such as, eye, smile, beard, gender, has glasses etc) in 
        # an input image of faces. so in other words, An ideal autoencoder 
        # will learn descriptive attributes of faces such as skin color, 
        # whether or not the person is wearing glasses, is female, etc. in
        # an attempt to describe an observation(input image) in some compressed representation.
        # for example a vector could be like (gender:-0.73, smile:0.99, glasses: 0.002, etc )
        # In the example above, we've described the input image in terms of its latent 
        # attributes using a 'single value' to describe each attribute. 
        # However, we may prefer to represent each latent attribute as a 'range of possible values'.
        # For instance, what 'single value' would you assign for the smile attribute if you feed
        # in a photo of the Mona Lisa? Using a variational autoencoder, 
        # we can describe latent attributes in probabilistic terms.
        # the mean and variance that we produce here, is used exactly for this vert reason
        # using them, we are learning a distrubution for each attribute and thus mu and variance
        # specify a range of values for each attribute.
        # [With this approach, we'll now represent each latent attribute for a given input 
        # as a probability distribution. When decoding from the latent state, we'll randomly
        # sample from each latent state distribution to generate a vector as input for our decoder.]
        # thats why later on, we use these means, variances along with an epsilon(act as a random variable)
        # to reconstruct the input image. 
        # So By constructing our encoder model to output a range of possible values 
        # (a statistical distribution) from which we'll randomly sample to feed into our decoder
        # , we're essentially enforcing a continuous, smooth latent space representation.
        # For any sampling of the latent distributions, we're expecting our decoder 
        # to be able to accurately reconstruct the input. 
        # Thus, values which are nearby to one another in latent space should correspond
        # with very similar reconstructions.
        # Intuitively, the mean vector controls where the encoding of an input should be 
        # centered around, while the standard deviation controls the “area”, how much from 
        # the mean the encoding can vary. As encodings are generated at random from anywhere
        # inside the “circle” (the distribution), the decoder learns that not only is a
        #  single point in latent space referring to a sample of that class, 
        # but all nearby points refer to the same as well. 
        # This allows the decoder to not just decode single, specific encodings in the 
        # latent space (leaving the decodable latent space discontinuous), but ones that 
        # slightly vary too, as the decoder is exposed to a range of variations of the
        # encoding of the same input during training. 
        # The model is now exposed to a certain degree of local variation by varying the 
        # encoding of one sample, resulting in smooth latent spaces on a local scale, that is,
        # for similar samples. Ideally, we want overlap between samples that are not very 
        # similar too, in order to interpolate between classes. 
        # However, since there are no limits on what values vectors μ and σ can take on,
        # the encoder can learn to generate very different μ for different classes, 
        # clustering them apart, and minimize σ, making sure the encodings themselves don’t
        # vary much for the same sample (that is, less uncertainty for the decoder). 
        # This allows the decoder to efficiently reconstruct the training data.
        # What we ideally want are encodings, all of which are as close as possible to each
        # other while still being distinct, allowing smooth interpolation, and enabling the
        # construction of new samples.
        # In order to force this, we introduce the Kullback–Leibler divergence 
        # (KL divergence[2]) into the loss function. The KL divergence between two probability
        # distributions simply measures how much they diverge from each other. 
        # Minimizing the KL divergence here means optimizing the probability distribution
        # parameters (μ and σ) to closely resemble that of the target distribution.
        # Intuitively, this loss encourages the encoder to distribute all encodings 
        # (for all types of inputs, eg. all MNIST numbers), evenly around the center 
        # of the latent space. If it tries to “cheat” by clustering them apart into 
        # specific regions, away from the origin, it will be penalized.
        # Now, using purely KL loss results in a latent space results in encodings densely 
        # placed randomly, near the center of the latent space, with little regard for 
        # similarity among nearby encodings. The decoder finds it impossible to decode 
        # anything meaningful from this space, simply because there really isn’t any meaning.
        # Optimizing the two together, however, results in the generation of a latent space
        # which maintains the similarity of nearby encodings on the local scale via clustering,
        # yet globally, is very densely packed near the latent space origin 
        # (compare the axes with the original).
        # Intuitively, this is the equilibrium reached by the cluster-forming nature of the
        # reconstruction loss, and the dense packing nature of the KL loss, forming distinct
        # clusters the decoder can decode. This is great, as it means when randomly generating,
        # if you sample a vector from the same prior distribution of the encoded vectors, N(0, I), 
        # the decoder will successfully decode it. And if you’re interpolating, there are 
        # no sudden gaps between clusters, but a smooth mix of features a decoder can understand.
     
        output = F.relu(self.fc1(output))
        # we dont use activations here
        output_mu = self.fc1_mu(output)
        output_std = self.fc1_std(output)
        # In its original form, VAEs sample from a random node z which is 
        # approximated by the parametric model q(z∣ϕ,x) of the true posterior.
        # Backprop cannot flow through a random node. Introducing a new parameter 
        # ϵ allows us to reparameterize z in a way that allows backprop to flow 
        # through the deterministic nodes. this is called reparamerization trick
        z = self.reparamtrization_trick(output_mu, output_std)

        # decoder 
        output = F.relu(self.fc2(z))
        # since we are using bce, we dont use sigmoid for numerical stability
        # in normal situations, we would use that to make values in range [0,1]
        output = self.fc2_2(output)
        return output, output_mu, output_std


# Note :
# In my experience working on the VAE, the KL annealer helps to train the model.
# To be more specific, when training your encoder and decoders right off the 
# bat variationally (KL-term constant) can lead to a lot of instability while 
# training. So a good first step is to train them as a AE and at some moment 
# slowly switch on the KL term. It allows the model to arrive to a 'decent' 
# spot (trained as AE) before going VAE. A similar pattern lies with ORGAN, 
# you need to train pre-train your generators so that they are in a decent 
# spot before competing with the discriminator. There are many ways of doing
# this, most are just engineering, hence MOSES's approach also works.
# For the original VAE i think around epoch 30 we start the KL annealing.


# Also read : https://github.com/jxhe/vae-lagging-encoder
# The code seperates optimization of encoder and decoder in VAE, and performs 
# more steps of encoder update in each iteration. This new training procedure 
# mitigates the issue of posterior collapse in VAE and leads to a better VAE 
# model, without changing model components and training objective.



# now lets train :
epochs = 20 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.BCEWithLogitsLoss()
interval = 1000
model = VAE().to(device)

for e in range(epochs):
    for i, (imgs, labels) in enumerate(dataloader_train):
        imgs = imgs.to(device)
        preds,mu, logvar = model(imgs)
        # for loss we simply add the reconstruction loss +kl divergance
        loss_recons = criterion(preds, imgs.view(imgs.size(0), -1))
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # I guess 0.5 is the beta (a multiplier that specifies how large the distribution 
        # should be)
        # - D_{KL} = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl = 0.5 * torch.sum(1+ logvar - mu.pow(2) - logvar.exp())
        # Normalise by same number of elements as in reconstruction
        kl/=imgs.size(0) * (28*28)

        loss = loss_recons + kl 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 
        if i% interval ==0:
            print(f'epoch ({e}/{epochs}) loss: {loss.item():.6f} KL: {kl.item():.6f} recons {loss_recons.item():.6f}')

#%% 
# Contractive Autoencoder

#%%
# Sequence-to-Sequence Autoencoder
# https://mc.ai/less-pain-more-gain-a-simple-method-for-vae-training-with-less-of-that-kl-vanishing-agony/
# There is growing interest in exploring the use of variational auto-encoders 
# (VAE), a deep latent variable model, for text generation. Compared to the 
# standard RNN-based language model that generates sentences one word at a time
# without the explicit guidance of a global sentence representation, VAE is 
# designed to learn a probabilistic representation of global language features
# such as topic, sentiment or language style, and makes the text generation 
# more controllable. For example, VAE can generate sentences with a specific 
# tense, sentiment or topic.


# However, training VAE on languages is notoriously difficult due to 
# something called KL vanishing. While VAE is designed to learn to generate
# text using both local context and global features, it tends to depend 
# solely on local context and ignore global features when generating text. 
# When this happens, VAE is essentially behaving like a standard RNN language 
# model.



#%%
# Adversarial Autoencoder https://blog.paperspace.com/adversarial-autoencoders-with-pytorch/
# Conditinoal VAE 
