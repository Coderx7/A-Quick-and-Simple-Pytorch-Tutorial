#%%
# in the name of God the most compassionate the most merciful
#!edit add original papers for reference
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

# sidenote2: # !EDIT this - rewrite it 
# in other words, the encoder actually 'encodes' the input data with large dimensions, 
# into a latent (hidden) representation space (usually called z),
# with much smaller dimensions than the original dimensions of the data
# This type of design is typically referred to as a 'bottleneck',
# as the encoder needs to learn an efficient and unique representation,
# to compress data from the original higher-dimensional space into this lower-dimensional space.

# !EDIT this - rewrite it 
# sidenote: 
# its like a typical network we have already seen, a typical CNN,
# it takes in an image (e.g. a 3d tensor of size(28,28,1)), 
# and converts it to a much more compact and denser representation at the end
# (eg. 1d tensor of size 100). This dense representation is then
# used by a classifier (can be a single fc layer, or multiple layers/ablock/etc)
# to classify the image.
# now the encoder does pretty much the same thing, 
# it takes in an input and produces a much smaller representation (the encoding)), 
# like in a cnn, this new dense representation needs to contain useful/necessary data
# for the classifier to properly does it job.
# the difference is that, instead of a classifer at the end, 
# theres another network that does something else (in our case reconstructiong the input data
# from that dense representation) so as you can see this is not something weird!
# 
# 
# 
# during this process of reconstructing the input data
# from the compressed representation, the new representation is
# developed and can be used for various applications. 
# the first part of the network that downsamples the input data
# into a feature vector is called an "Encoder", and the part that
# reconstructs the input from the mentioned feature vector is called
# a "Decoder". 
# when we have successfully trained our autoencoder, we can use its
# new representation instead of our data. we can use it for 
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
# there are lots and lots of use cases for autoencoders(and in general generative models)
# However, note that, the notion of compression spoken here is different than that of
# what you find in different media formats such as jpeg, mp3, etc. 
# Autoencoders do not work well on unseen data and thus usually have difficulties 
# generalizing well to unseen data.(more on this later) so the techniques and nature of
#! work is different here (explain better!!)
# 
# There are different kinds of Autoencoders, they can be linear, or
# nonlinear, shallow, or deep, convolutional, or not, etc
# we will cover some of the most famous variants here. 
# lets start

# before we start lets get familiar with couple of concepts 

# note :
# https://www.statisticshowto.datasciencecentral.com/posterior-distribution-probability/
# Posterior probability is the probability an event will happen after all evidence or 
# background information has been taken into account. It is closely related to prior probability,
# which is the probability an event will happen before you taken any new evidence into account.
# You can think of posterior probability as an adjustment on prior probability:
#         Posterior probability = prior probability + new evidence (called likelihood).

# For example, historical data suggests that around 60% of students who start college will 
# graduate within 6 years. This is the prior probability. However, you think that figure is 
# actually much lower, so set out to collect new data. The evidence you collect suggests that
# the true figure is actually closer to 50%; This is the posterior probability.

# What is a Posterior Distribution?
# The posterior distribution is a way to summarize what we know about uncertain quantities in 
# Bayesian analysis. It is a combination of the prior distribution and the likelihood function,
# which tells you what information is contained in your observed data (the “new evidence”). 
# In other words, the posterior distribution summarizes what you know after the data has been 
# observed. The summary of the evidence from the new observations is the likelihood function.
# Posterior Distribution = Prior Distribution + Likelihood Function (“new evidence”)
# Posterior distributions are vitally important in Bayesian Analysis. They are in many ways 
# the goal of the analysis and can give you:
#     Interval estimates for parameters,
#     Point estimates for parameters,
#     Prediction inference for future data,
#     Probabilistic evaluations for your hypothesis.
# ------------------------------------------------------------------------------ 

# https://www.statisticshowto.datasciencecentral.com/likelihood-function/

# What is a prior probablity : 
# https://www.statisticshowto.datasciencecentral.com/prior-probability-uniformative-conjugate/

# Prior Probability: Uniformative, Conjugate
# Probability > Prior Probability: Uniformative, Conjugate

# What is Prior Probability?
# Prior probability is a probability distribution that expresses established beliefs about an 
# event before (i.e. prior to) new evidence is taken into account. When the new evidence is used
# to create a new distribution, that new distribution is called posterior probability.
# prior probability 
# For example, you’re on a quiz show with three doors. A car is behind one door, 
# while the other two doors have goats. You have a 1/3 chance of winning the car. This is the 
# prior probability. Your host opens door C to reveal a goat. Since doors A and B are the only 
# candidates for the car, the probability has increased to 1/2. The prior probability of 1/3 has 
# now been adjusted to 1/2, which is a posterior probability.
# In order to carry our Bayesian inference, you must have a prior probability distribution. 
# How you choose a prior is dependent on what type of information you’re working with. 
# For example, if you want to predict the temperature tomorrow, a good prior distribution 
# might be a normal distribution with this month’s mean temperature and variance.

# Uninformative Priors
# An uninformative prior gives you vague information about probabilities. It’s usually used when 
# you don’t have a suitable prior distribution available. However, you could choose to use an 
# uninformative prior if you don’t want it to affect your results too much.
# The uninformative prior isn’t really “uninformative,” because any probability distribution 
# will have some information. However, it will have little impact on the posterior distribution 
# because it makes minimal assumptions about the model. For the temperature example, 
# you could use a uniform distribution for your prior, with the minimum values at the record low 
# for tomorrow and the record high for the maximum.

# Conjugate Prior
# A conjugate prior has the same distribution as your posterior prior. For example, if you’re 
# studying people’s weights, which are normally distributed, you can use a normal distribution
#  of weights as your conjugate prior.

# lets start!

import datetime
import numpy as np 
import torch
import torchvision
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim
import matplotlib.pyplot as plt 
%matplotlib inline

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

# Ok, enough talking lets get busy and have our first auto encoder. 
# before we continue, we should pickup a dataset. I chose MNIST as its simple enough
# to be used in different types of autoencoders with short training time. 
# after we created our dataset, we will implement different types of AutoEncoders 
dataset_train = datasets.MNIST(root='MNIST',
                               train=True,
                               transform = transforms.ToTensor(),
                               download=True)
dataset_test  = datasets.MNIST(root='MNIST', 
                               train=False, 
                               transform = transforms.ToTensor(),
                               download=True)
batch_size = 128
num_workers = 0
dataloader_train = torch.utils.data.DataLoader(dataset_train,
                                               batch_size = batch_size,
                                               shuffle=True,
                                               num_workers = num_workers, 
                                               pin_memory=True)

dataloader_test = torch.utils.data.DataLoader(dataset_test,
                                               batch_size = batch_size,
                                               num_workers = num_workers,
                                               pin_memory=True)

# lets view a sample of our images 
def view_images(imgs, labels, rows = 12, cols =11, figsize=(12,16), dpi=100, normalized=False, mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5], fname_to_save_as=None, title=None, title_top_margine=0.99,title_fontsize=12):
    # images in pytorch have the shape (channel, h,w) and since we have a
    # batch here, it becomes, (batch, channel, h, w). matplotlib expects
    # images to have the shape h,w,c . so we transpose the axes here for this!
    imgs = imgs.detach().cpu().numpy().transpose(0,2,3,1)
    
    if normalized:
        #unnormalized the image
        #normalization is imgs-mean/std
        #unnormalizing is imgs*std+mean
        imgs = imgs * std + mean
        # clip to [0-1]
        imgs = imgs.clip(0,1) 
    
    # sidenote: note that if we use a large figsize, with a high dpi
    # we may get an error complaining the image size is too big! it 
    # refers to the whole matplotlib figure on which we are drawing 
    # our images! so make sure you set the right numbers here!
    # also note the figsize row,cols, if you use the wrong size
    # there might not be enough space to display the labels at the top!
    # (try (6,4) and see the result!)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    if title:
        fig.suptitle(title, fontsize=title_fontsize, y=title_top_margine)
    # plt.title('View Images') 
    
    max_plots = rows*cols
    # make sure we don't face an error for trying to
    # creating more subplots than available
    if imgs.shape[0]<max_plots:
        num_plots = imgs.shape[0] 
    else:
        num_plots = max_plots
        print(f'Warning, number of images({imgs.shape[0]}) exceed figures plots({max_plots}). '
              f'Only displaying the first {max_plots} images. (Hint: Increase rows/cols ())')
    
    for i in range(num_plots):
        ax = fig.add_subplot(rows, cols, i+1, xticks=[], yticks=[])
        # since mnist images are 1 channeled(i.e grayscale), matplotlib
        # only accepts these kinds of images without any channesl i.e 
        # instead of the shape 28x28x1, it wants 28x28,for color images
        # the squeeze and cmap will be ignored
        ax.imshow(imgs[i].squeeze(), cmap='Greys_r')
        lbl = labels[i]
        lbl = lbl.item() if isinstance(lbl,torch.Tensor) else lbl
        ax.set_title(lbl)
    
    # plt.subplots_adjust(top=0.90)    
    # we can use plt.tight_layout(pad=1,rect= (0, 0, 2, 2)) to have nice
    # compact figure, we could also simply use tight_layout and let 
    # matplotlib handle the padding, and scaling, but in this case lets
    # use rect to scale our images so they are larger in the plot!
    # (try numbers like 0.8, 1, 2, 20!)
    # sidenote, when using large numbers here, we may get an error if
    # we have used a large figuresize with a large dpi, I made that
    # clear just a few lines back, these are related!
    # note: I couldnt get this to work for figsize(6,8) when we add a title
    # to the figure. without a title, figsize(6,8) works great and tight_layout
    # like below does the job, however, when we add the title, it messes up the
    # title position, if I use tight_layout(), the figsize(6,8) doesnt look well
    # anymore! so I have to increase the figsize 2x!, it works, but it makes the
    # images larger, and is has more overhead! using plt.
    if figsize == (6,8):
        plt.tight_layout(pad=1,rect= (0, 0, 2, 2))
    else:
        # tight layout works well for larger fig_szes like (8,12)
        plt.tight_layout()

    # save the figure to the disk, this comes handy when we want to
    # keep track of our progress, or later on create a gif out of these
    # images. (basically keep traking how the model is doing in terms of
    # reconstruction is essential when it comes to generative models!)
    if fname_to_save_as:
        # bbox_inches='tight' makes matplotlib to include all of 
        # the elements of the figure even those that extend outside
        # the default bounding box, without this only a portion of 
        # our figure will be saved!
        plt.savefig(fname_to_save_as, bbox_inches='tight')
    
    plt.show()

# now lets view some 
imgs, labels = next(iter(dataloader_train))
view_images(imgs, labels,13,10)
randns = torch.rand(size=(imgs.size(0),3,32,32))
view_images(imgs=randns, 
            labels=[f'num_{l.item()}' for l in labels], 
            rows=13,
            cols=10,
            fname_to_save_as='./results/randomtest.jpg',
            # figsize=(12,16),
            title='Random images')

# good! we are ready for the actual implementation
#%% 
# The first autoencoder weare going to implement is the simplest one, 
# a linear autoencoder.
# creating an autoencoder is just like any other module we have seen so far, simply
# inherit from nn.Module and define the needed layers and call them in the forward()
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
        # The simplest form can be a one layered encoder
        # and a 1 layered decoder! of course we can add more
        # layers between them, but lets see how this performs
        self.fc1 = nn.Linear(28*28, embedingsisze)
        # our decoder part
        self.fc2 = nn.Linear(embedingsisze, 28*28)

    def forward(self, inputs):
        # our foward pass is nothing special
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
                print(f'epoch: ({e}/{epochs}) loss: {loss.item():.6f} lr:{scheduler.get_lr()[-1]:.6f}')
        scheduler.step()
    print('done')

# Now lets see the output of our autoencoder
def test(model,device,rows,cols):
    imgs, labels = next(iter(dataloader_test))
    imgs = imgs.to(device)
    outputs = model(imgs)
    view_images(outputs, labels,rows=rows,cols=cols)
#%%

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_linear_ae = model_linear_ae.to(device)
optimizer = optim.Adam(model_linear_ae.parameters(), lr = 0.1) 
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5)

train(model_linear_ae, dataloader_train, optimizer, scheduler, 20, device) 
test(model_linear_ae, device,rows=9,cols=5)
# so this is the linear autoencoder! in order to make a vanila autoencoder
# which may refer to a version with nonlinear activation functions, you 
# only need to apply a transformation function in the fowarad pass and 
# in order  to get a good result, you need to add a few more 
# layers .(we do this in the next architecture )
# we can get better results with more epochs and decaying learnng rate,
#  but it wont make a drastic change! especially on more complex data, as its 
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

    # lets create encoder/decoder methods separately this time
    # so we can use them easier later (for visualization etc) 
    def encoder(self, inputs):
        # encoder part
        inputs = inputs.view(inputs.size(0), -1)
        output = F.relu(self.fc1(inputs))
        output = F.relu(self.fc2(output))
        return output
    
    def decoder(self, inputs):
        # decore part
        output = F.relu(self.fc3(inputs))
        # since our output is image, values should 
        # be in the range [0, 1]!
        #sidenote: note that unlike our previous example,
        # we are now using a sigmoid transformation function here.
        # this is needed as we have used transformation/activaion
        # functions on several layers before, hence not linear
        # anymore. using sigmoid gives us a clear image! as it
        # enforces the values to be in range valid for images!
        # removing the ghosting and other alike artifacts from the image.
        # try removing sigmoid and running the example again
        output = F.sigmoid(self.fc4(output))
        output = output.view(-1, 1, 28, 28)
        return output
    
    def forward(self, inputs):
        output = self.encoder(inputs)
        output = self.decoder(output)
        return output 

model_mlp_ae = MLPAutoEncoder(32).to(device)
print(model_mlp_ae)

# criterion = nn.MSELoss()
optimizer = optim.Adam(model_mlp_ae.parameters(), lr = 0.01) 
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5)
train(model_mlp_ae, dataloader_train, optimizer, scheduler, 20, device)    
test(model_mlp_ae,device,rows=13,cols=10)
# note, the loss sometimes doesnt decrease which is expected 
# rerun the experiment to get a better result!
#%%
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
# lets also visualize the encodings/features learned by our encoder
# and see whether/how well these features are separated. 
# this kind of visualization specifically becomes intersting when we
# start implementing other types of autoencoders such as VAE. 
# when we get there we'll explain this further. 
# ok, to do this, one way is to use scatter plot and display
# each sample that way that is, we feed our images to the encoder,
# grab the feature vector and then display it in a scatterplot.
# since we are going to use scatter plot, our feature vector must be 2D
# (that is it needs to have 2 numbers!) if its not, we need to use pca or tsne
# to project them into 2d.
def plot_embedding_clusters(model, dataloader_train, title='',use_pca=False):
    model.eval()
    # grab the device from model parameter
    device = next(model.parameters()).device
    # grab all the features, because tsne needs to be applied to 
    # the whole dataset all atonce not batch by batch
    all_features = []
    all_labels = []

    with torch.no_grad():
        for imgs, lbls in dataloader_train:
            imgs = imgs.to(device)
            # Get feature vectors
            feature_vectors = model.encoder(imgs).cpu().view(imgs.size(0), -1).numpy()
            all_features.append(feature_vectors)
            all_labels.append(lbls.numpy())

    # concatenate all batches
    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    if use_pca:
        reducer = PCA(n_components=2)
        # since pca is sensitive to the scale of features and 
        # if the features are not properly scaled (e.g. mean-centered and variance-normalized),
        # it can produce poor projections we scale the features here!
        scaler = StandardScaler()
        all_features = scaler.fit_transform(all_features)
    else:
        reducer = TSNE(n_components=2, random_state=66, perplexity=30)

    plt.figure(figsize=(10, 8))
    # print(f'{all_features[0].shape[-1]}')
    
    if all_features[0].shape[-1] >2 :
        # features2d are coordinates showing where each datapoint is
        features2d = reducer.fit_transform(all_features)
        # print(f'{features2d[:5]}')
    else:
        features2d = all_features
    # tab10, is a colormap inwhich it has 10 colors, therefore its a prefect choice for us    
    scatter = plt.scatter(features2d[:, 0], features2d[:, 1], c=all_labels, cmap='tab10', alpha=0.6)

    # add class labels to each cluster for better visualization
    # to do this we need t o calculate the centeroid(i.e. mean) of each cluster
    # which is basically taking the average of all the points for that cluster
    # and then use plt.text to add class numbers
    
    # note we dont need all the labels, just one for each cluster!
    for label in list(range(10)):
        # find the centroid of each cluster
        # note that the values in features2d are coordinates(when using tsne),
        # which are the 2D positions of the data points
        # since our data are stored sequentially we know each row(class label) 
        # in all_labels belong to a corresponding data point in features2d.
        # that is for example, if all_labels[0] = 0, it means the first data point
        # in features2d belongs to class 0.
        # we use this to grab all the points belonging to a specific label one at a time 
        centroid = np.mean(features2d[all_labels == label], axis=0)
        # annotate the centroid with the class label
        plt.text(centroid[0], centroid[1], str(label), fontsize=12, fontweight='bold',
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.3'))

    title = f"\n{title}" if title else ''
    plt.title(f"{'PCA' if use_pca else 'TSNE'} Projection to 2D{title}")
    plt.colorbar(scatter, label='Class Label')
    plt.show()
#%%
plot_embedding_clusters(model_mlp_ae, dataloader_train, use_pca=False)
# 
# if we used embedding_dim=2 in our previous examples, we would get a drastically different image
# try that and see the difference. 
# TODO: note explain why tsne is a better choice here when our feature dim >2D
# note that we use PCA, when we are dealing with linear relationships
# which is not the case here (we are not doing a simple linear transformation here)
# it would also tend to produce more overlapping clusters,(apposed to distinct/wellseparated ones)
# when the data has complex, non-linear relationships (which is our case try use_pca=True))
# because of this, tsne is the right choice here as its specifically designed 
# for highdimensioal data. (it preserves local structures in high-dimensional data 
# that is the relationships between nearby points is preserved
# and it tries to keep points that are nearby in high-dimensional
# space close together in the lower dimension (our 2D projection).
# and its used extensively for visualizing clusters/groups in high dimensional data
# (compared to pca, it produces more distinct and well-separated clusters 
# in the 2D projection.
# (also pca focuses on preserving global structures (i.e., the overall variance in the data).
# and its less effective at preserving local relationships, which can make clusters less distinct
# in the 2D projection.)

# sidenote2: 
# tsne hyperparameters like perplexity and learning rate, control 
# the balance between preserving local and global structures.
# so tuning them can improve the visualization.

#%%
# While our mlp model is more powerful than the previous model, it is not suitable for data such as images
# for image like data, we use conv layers! and hence our new autoencoder is Convolutional AutoEncoder. 
# lets see how to implement this : 
# something that needs to be said is that, when the number of layers is increased, i.e. 
# your network gets deeper, you may see that your model may train sometimes and not the 
# other times and the loss may not decrease. when you see this, you should know this is
# happening becasue of the depth of your network.  use the batchnorm and all will be good. 
# thats why I created two functions for this very purpose. 
# try creating your network with and without batchnormalization enabled and see the difference
# (try running for several times with the one with no batchnormalization to see that sometimes 
# it may work and some times it will fail (the loss doesnt decrease it fluctuates around loss: 0.1xxx),
# but with batchnorm, it will always work!(the loss decreases 100x more (around 0.001xx)))
# remember to enable/disable batchnorm for both conv_bn() and deconv_bn()
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
        # note we disable batchnorm for the last layer
        # sidenote when using batchnorm, theres no need for a bias anymore! it becomes redundant!
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
test(model_c, device,rows=13,cols=10)
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
# and we just will add a simplenoise lets see that.

# here we specify how noisy our images become
noise_intensity_threshold = 0.5
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
# To add noise to our images, we create a random tensor with the same shape 
# as our image batch so we can easily add them together. we used torch.rand_like(), 
# which generates random values from a uniform distribution between 0 and 1. 
# we could have also used torch.randn_like(), (which generates random values 
# from a normal distribution (Gaussian) with a mean of 0 and a standard deviation of 1). 
# before adding them together though, we
# use a number to specify how much noise we want to apply to our images
# the smaller the threshold number, the fainter the noise (values) become
# and therefore the less our image is affected, the larger the threshold
# number, the stronger/heavier/more noticeable the noise becomes and therefore
# our image is more affected. 
# note that I named that number noise_intensity_threshold to make it apparent 
# that it only affects the magnitude of our noise tensor. it does not specify
# what precentage of the image is applied with the noise!(or how many pixels are affected)
# rather it only specifies "how much" "every single pixel" in our images are affected
# by the noise.
# also after we added the noise to our images, we need to normalzie them so the images 
# contain only valid values (values betwen 0-1). thats why we clamp the data afterward.

# sidenote:
# we said both of these methods(uniform and normal distributions) allow us to add noise, but they produce different types of noise,
# and you may ask, why would we want to choose one over the other? or whats the difference between them? 
# choosing between these two distributions, has different implications. 
# like for example, uniform noise is evenly spread across a range, while  
# Gaussian noise tends to cluster around the mean with some outliers.
# this in turn means a few things:
# For one, if we use a uniform distribution to generate noise,
# it means we plan on generating noise where every value within a specified range (e.g., 0 to 1) 
# is equally likely.
# This results in noise that is evenly spread across the range, it doesnt favor any part
# more than others, every part/range has the same importance, therefore creating a 
# flat/consistent perturbation across the image.
# Uniform noise is therefore useful for simulating random, unbiased distortions, 
# such as sensor noise or quantization errors but it looks more "artificial" and is evenly distributed.
# we also use uniform noise when we have no idea about the underlying distribution, 
# and want to avoid introducing bias that could heavily affect the posterior distribution.
# 
# Unlike uniform distribution, we use normal distribution to generate noise from 
# a Gaussian (normal) distribution with a mean of 0 and a standard deviation of 1.
# This means the noise values are more likely to be close to the mean (0), with 
# fewer extreme values (outliers). In other words, the noise favors values 
# around the mean more than those farther away.
# Gaussian noise is often used to simulate natural noise, such as thermal noise 
# in electronic systems or subtle variations in lighting. Gaussian noise is thus 
# more natural and resembles real-world noise.
# 
# and finally to answer the question of which one to use: use whatever suites the job!
# we usually use gaussian noise by default unless theres a reason to use uniform or other
# types of noise.
# Gaussian noise was and still is the most widely used type of noise in denoising autoencoders.
# because it is still our best choice for modeling natural noise, and many real-world noise
# sources (e.g., camera sensor noise, audio noise) are well-approximated by Gaussian distributions.
# in applications such as image denoising, audio denoising, and signal processing, 
# Gaussian noise is the default choice.
# 
# sidenote3:
# it should be obvious that if we use real world noise instead of gaussian noise, we may
# see a good improvement. but catching real world noise is not always an easy task, and
# gausian noise does a pretty good job, so thats why we dont see a lot of papers doing it
# however, there are several cases that do such as : 
# DnCNN: Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising 2017
# CBDNet: Toward Convolutional Blind Denoising of Real Photographs 2019
# RIDNet: Real Image Denoising with Feature Attention 2019
# etc 
# there are more papers that tried to use realworld noise. but how do you capture real world noise?
# to capture real world noise, we take photos or videos in noisy conditions 
# (low-light conditions or with high ISO settings where noise is more pronounced).
# we capture several images of a static scene (e.g., a blank wall or a dark room) 
# using the same camera settings. we then take the mean image to estimate the clean signal.
# and subtract it from each individual image and save result which is the noise for each sample.
# the steps nearly the same for audio or prety much anything else. 
# for example for audio: 
# we record audio in environments where the noise is present (e.g., a busy street, a crowded room).
# record as many samples as we need in the said environment,
# use a filtering or signal processing (spectral analysis) to isolate the noise component and 
# save the noise samples.
# and then during training, use these noise samples and add them to clean data. 
# note that clean data may not be that clean, (unless you make sure it is, either synthetically generated
# or generated in a noise free environment whatever the case is)

#%%
imgs = imgs + (noise_intensity_threshold * torch.rand_like(imgs))
imgs.clamp_(0,1)
view_images(imgs,labels)
# sidenote: TODO: (shorten the long explanations and stop repeating the same thing over and over again!)
# we usually use smaller noise thresholds (e.g., 0.1 or 0.2) for tasks like denoising, 
# where our goal is to remove subtle noise, we can use more intense noise as well, but 
# but the likelihood of removing fine details in the images during the denoising process 
# increases drastically.
# The noise threshold determines the level at which noise is separated from the true signal.
# so smaller thresholds are used when the noise level is low. this ensures that the denoising 
# process doesnt mistakenly remove fine image details or important structures, 
# which could otherwise be interpreted as noise and removed consequently.
# For autoencoders, introducing smaller noise levels during training 
# (e.g., Gaussian noise scaled with small thresholds like 0.1) can improve the denoising
# performance on low-noise images.
# we use larger noise thresholds (e.g., 0.5) for other usecases such as data-augmentation, 
# but not excessivly large (e.g. .7, 0.9, 1.0).
# larger values (e.g., 1.0+) are usually used for specific usecase like for example to test the robustness
# of our models against heavily corrupted samples.
# 
# sidenote2: 
# What we described here is known as noise scaling and its usually done 
# for controlling the intensity or strength of the noise, and not the proportion/precentage
# of the image that is affected.
# 
# When we scale the noise by a factor like 0.5, we are controlling the magnitude
# of the noise values, not the percentage of the image that is affected. 
# To make this a bit more clear lets step back a bit, and see how we create random values
# and what implications follow. 
# 
# To create a random value, we usually either use a uniform distribution or a normal distribution
# (we briefly talked about them in basic pytorch introduction chapter, 
# and we know there are many other distributions, but for what we are dealing with here,
# these are the two distributions that we normally use(rand/randn)). 
# 
# In pytorch we either use torch.rand_like(imgs) or torch.randn_like(imgs) to create a
# random tensor with the same shape as our input tensor.  
# torch.rand_like(imgs) generates random values uniformly distributed between 0 and 1.
# while torch.randn_like(imgs) generates random values from a Gaussian (normal) distribution 
# with a mean of 0 and a standard deviation of 1.
# 
# Now, when we multiply the noise by a number like 0.5, we are in fact scaling the magnitude
# of the noise values, which for the uniform noise, the noise values would now range between 0 and 0.5.
# and for gaussian/normal noise, the standard deviation of the noise would become 0.5 (it shrinks by half
# !explain more).
# 
# when we add this scaled noise to the original image, this means every pixel in the image
# is affected by the noise, but the strength of the noise depends on the scaling factor.

# The scaling factor (noise_intensity_threshold) determines how much the noise affects
# the image, i.e.if we use a smaller value (e.g., 0.1), the noise will be subtle and less noticeable
# and the image remains mostly intact, with only slight variations introduced by the noise.
# (i.e. noise_intensity_threshold = 0.1 adds very faint noise)
# whereas if we use a larger value (e.g., 0.5 or 1.0) the noise will be much stronger and
# more noticeable, and the image becomes significantly affected/distorted, with more pronounced 
# variations.(i.e. noise_intensity_threshold = 0.5 adds moderate noise,
# while noise_intensity_threshold = 1.0 adds a strong noise)

# so the scaling of the noise does not affect the percentage of the image that is noisy.
# rather,Every pixel in the image is affected by the noise and the scaling factor 
# only determines how much each pixel is altered, not how many pixels are altered.
# For example: If noise_intensity_threshold = 0.5, every pixel in the image will have 
# noise added, but the noise values will range between 0 and 0.5.
# If noise_threshold = 1.0, every pixel will still have noise added, but the noise values 
# will range between 0 and 1.0.
# we can visualize this effect easily as well
# grab an image and apply different levels of noise threshold/intensity
imgs = next(iter(dataloader_train))[0][0].unsqueeze(0)
noise_thresholds = [0.1, 0.2, 0.5, 0.7, 1.0, 2.0]
fig, axes = plt.subplots(1, len(noise_thresholds), figsize=(16,4))
for i, threshold in enumerate(noise_thresholds):
    noisy_imgs = imgs + threshold * torch.rand_like(imgs)
    noisy_imgs = noisy_imgs.clamp(0, 1)
    axes[i].imshow(noisy_imgs[0, 0], cmap='gray')
    axes[i].set_title(f'Noise Threshold = {threshold}')
    axes[i].axis('off')
    # plt.tight_layout(pad=1,rect=[0,0,2,2])
plt.show()

# to make things tidier lets create a simple function to do the job
def add_noise(imgs, noise_intensity_threshold=0.5, uniform_distribution=False):
    noise_tensor = torch.rand_like(imgs) if uniform_distribution else torch.randn_like(imgs)
    return imgs + (noise_intensity_threshold * noise_tensor)

noise_threshold = 0.5
uniform_dist = True

print(f'Training with {noise_threshold=:.4f} and {"uniform" if uniform_dist else "normal"} distribution')
print(model)
for e in range(epochs):
    loss_epoch = 0.0
    for imgs,_ in dataloader_train:
        imgs = imgs.to(device)

        #apply noise to our image 
        imgs_noisy = add_noise(imgs, noise_threshold, uniform_dist)
        # clip all values outside of 0,1 becasue our image values 
        # should be in this range!
        imgs_noisy = imgs_noisy.clamp(0,1)
        imgs_recons = model(imgs_noisy)

        loss = criterion(imgs_recons, imgs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_epoch += loss.item()
    print(f'epoch: {e}/{epochs} loss: {loss.item():.4f} lr: {scheduler.get_lr()[-1]:.6f}')
    scheduler.step()

# lets see how the network does on noisy image!
imgs,labels = next(iter(dataloader_test))
imgs = imgs.to(device)
imgs = add_noise(imgs, noise_threshold, uniform_dist)
imgs.clamp_(0,1)
view_images(imgs,labels)
new_noise_free_imgs = model(imgs)
view_images(new_noise_free_imgs,labels)
# note that we can improve our results with a better training regime(optimizer/shceduler/architecture)
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
uniform_dist = True
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
imgs = add_noise(imgs, noise_threshold, uniform_dist)
imgs.clamp_(0,1)
view_images(imgs,labels)

print(model)
for e in range(epochs):
    loss_epoch = 0.0
    for imgs,_ in dataloader_train:
        imgs = imgs.to(device)

        #apply noise to our image 
        imgs_noisy = add_noise(imgs, noise_threshold, uniform_dist)
        # clip all values outside of 0,1 becasue our image values 
        # should be in this range!
        imgs_noisy = imgs_noisy.clamp(0,1)
        imgs_recons = model(imgs_noisy)

        loss = criterion(imgs_recons, imgs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_epoch += loss.item()
    print(f'epoch: {e}/{epochs} loss: {loss.item():.4f} lr: {scheduler.get_lr()[-1]:.6f}')
    scheduler.step()

# lets see how the network does on noisy image!
imgs,labels = next(iter(dataloader_test))
imgs = imgs.to(device)
imgs = add_noise(imgs, noise_threshold, uniform_dist)

imgs.clamp_(0,1)
view_images(imgs,labels)
new_noise_free_imgs = model(imgs)
view_images(new_noise_free_imgs,labels)
#%%
plot_embedding_clusters(model, dataloader_train, use_pca=False)
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
# we implement a sparse autoencoder in this section and see how it performs. 
# as I said earlier, aside from the normal reconstruction loss, we need a new regularizer
# lets create this regularizer now. 
# We are going to create a Function object that applies
# l1penalty we inherit from autograd.Function class for this.
# good exlanation 
# andrew ng standford classnotes 2011: https://web.stanford.edu/class/cs294a/sparseAutoencoder_2011new.pdf
# a good video worth watching: https://www.youtube.com/watch?v=7mRfwaGGAPg

import copy # used for deep copy of our weights
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
    def __init__(self, embeddingsize=400, tied_weights = False):
        super().__init__()
        self. tied_weights = tied_weights

        self.encoder = nn.Sequential(nn.Flatten(),# instead of flattening the input in forward, we do it in encoder!
                                     nn.Linear(28*28, embeddingsize),
                                     nn.Sigmoid())# or relu
        self.decoder = nn.Sequential(nn.Linear(embeddingsize, 28*28),
                                     nn.Sigmoid())
        # you may see some people, use the shared weights between encoder
        # and decoder, i.e. decoder uses the transposed weightmatrix of the 
        # encoder. for doing this  there are couple of ways.
        # one of way is to use the functional form and simply 
        # use one weight and its transpose like this 
        # weight = nn.Parameter(torch.rand(input_dim, output_dim))
        # self.encoder = F.linear(input, weight, bias=bias_param)
        # self.decoder = F.linear(input, weight.t(), bias=bias_param2)
        # we can also simply define our new weight and assigne it to both modules
        # note that this nonfunction method is nuisanced! and you need to be aware
        # of that. (see my explanations ahead)
        if self.tied_weights:
            self.weights = nn.Parameter(torch.randn_like(self.encoder[1].weight))
            # note we use .data, so we directly link the underlying storage
            # for encoder weight to our parameter storage. if we dont use .data
            # we'll get an error saying we have to use nn.Parameter()!
            # or we will have to use the functional form instead.
            self.encoder[1].weight.data = self.weights
            # note that if we use id() we see they are different, 
            # however, this is expected as this is a just a view, 
            # not a new parameter, the actual underlying data is the same
            # and we can see this during training and after it
            # when we visualize the weights 
            # see the explanation ahead where I gave a more in depths explanation to ptove
            # this!
            self.decoder[0].weight.data = self.weights.t()
            # print(f'{id(self.weights)=}\n{id(self.weights.t())=}')
    
    # if we were to use the functional form
    # we would have these instead of the linear modules
    # def encoder(self, input):
    #     return F.sigmoid(F.linear(input, weight=self.weights,bias=encoder_bias))
    
    # def decoder(self, input):
    #     return F.sigmoid(F.linear(input, weight=self.weights.t(),bias=decoder_bias))
    
    def forward(self, input, apply_gradient_constraint=False, l1_weight=0):
        # input = input.view(input.size(0), -1) # replaced it with flatten in encoder
        output_enc = self.encoder(input)
        # we apply the L1penalty during forward pass
        # we have to do this in order for the altered gradients
        # to take effect in training, during loss calculation we simply
        # just use the reconstruction loss
        if apply_gradient_constraint:
            output_enc = L1Penalty.apply(output_enc, l1_weight)
        
        rec_imgs = self.decoder(output_enc)
        rec_imgs = rec_imgs.view(input.size(0), 1, 28, 28)
        return output_enc, rec_imgs

#%%
# heres a test to show that our way of sharing weights is actually correct
# and is the same as using the functional form! 
# sidenote/tldr:
# both functional and nonfunctional forms share the weights and they both work
# prefectly fine. however theres a catch here, in our nonfunctional method, we 
# bypass pytorch's autograd system (gradient tracking), but as I explain later, 
# this doesnt pose an issue for us in this case. 
# but it causes some inconsitencies which are not desired
# (such as wasted parameters). itd be safer to use functional form especially if 
# we plan on working something more complex! see the explanation at the end
# 
class SharedWeightsAE(nn.Module):
    def __init__(self, input_dim=4, embedding_dim=2):
        super().__init__()
        self.encoder = nn.Linear(input_dim,embedding_dim)
        self.decoder = nn.Linear(embedding_dim,input_dim)
        # define a single weight and assign it to both encoder and decoder
        self.shared_weight = nn.Parameter(torch.randn(embedding_dim, input_dim))
        # note we use .data to directly access the underlying storage and link
        # shared weight parameter's underlying storage with encoder/decoder's together
        # note that, by doing this, we are bypassing pytorchs autograd system ,
        # and causes it not to be able to track this operation and therefor track
        # the gradients. This will in-turn make the gradients for the shared_weight
        # to be None!
        # this however doesnt pose an issue for us, as the grad property for each module will
        # be populated properly during training (though the shared_weight wont have any gradients
        # for this reason, but since the underlying storage is linked, the changes will take
        # place in the same storage and everything will be fine, see my final explanation at the end)
        self.encoder.weight.data = self.shared_weight
        self.decoder.weight.data = self.shared_weight.t()
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

# heres the functional version
class SharedWeightsAEFunctional(nn.Module):
    def __init__(self, input_dim=4, embedding_dim=2):
        super().__init__()
        # a single weight parameter is used for both encoder and decoder
        self.shared_weight = nn.Parameter(torch.randn(embedding_dim, input_dim))
        # since we use the functional form of linear layer, 
        # we also prepare a separate bias parameter for 
        # the encoder and decoder as well(they are not shared obviously!)
        self.encoder_bias = nn.Parameter(torch.zeros(embedding_dim))
        self.decoder_bias = nn.Parameter(torch.zeros(input_dim))

    # instead of a module, we now create a method to easily call them
    # just like the previous version
    def encoder(self, x):
        return F.linear(x, self.shared_weight, self.encoder_bias)

    def decoder(self, x):
        return F.linear(x, self.shared_weight.t(), self.decoder_bias)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

torch.manual_seed(5)
def main(use_functional=True):
    print('-'*40)
    print(f"Using {'Functional' if use_functional else 'Non-Functional'} Form")

    if use_functional:
        model = SharedWeightsAEFunctional(input_dim=4, embedding_dim=2) 
    else:
        model = SharedWeightsAE(input_dim=4, embedding_dim=2)

    # our input
    x = torch.randn(3, 4)

    # forward pass
    _, decoded = model(x)

    # lets check weight sharing before we directly update the weights
    print('\nBefore update:')
    encoders_weight = model.shared_weight if use_functional else model.encoder.weight
    decoders_weight = model.shared_weight.t() if use_functional else model.decoder.weight
    print(f'Encoders Weight:\n {encoders_weight.detach().numpy()}')
    # note that since transposing(calling .t()) creates a temporary view
    # the id() will be different (values order are obviously different because
    # the shape is different after transposing!) so to better show that the 
    # underlying data is indeed the same, we transpose it back!
    # to get the same view as the original shared_weight used by encoder
    print(f'Decoders Weight(transposed):\n {decoders_weight.t().detach().numpy()}')

    # now lets update the shared weight directly!
    # this should reflect in both the encoder and decoder weights
    model.shared_weight.data += 1.0
    # model.encoder.weight.data += 1.0
    # model.decoder.weight.data += 1.0

    print('\nAfter direct update:')
    print(f'Encoders weight:\n {encoders_weight.detach().numpy()}')
    print(f'Decoders weight(transposed):\n {decoders_weight.t().detach().numpy()}')
    # Heres another check to make sure they all match!
    assert torch.eq(encoders_weight, decoders_weight.t()).all(),'They must match!'

    # lets see how gradients are affected/properly accumulated
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss = F.mse_loss(decoded, x)
    loss.backward()

    print('\nGradient check:')
    # shared_weight only has grads when using functional form,
    # in nonfunctional form its grads are None!
    print(f'shared_weight Gradients:\n{model.shared_weight.grad}')
    if not use_functional:
        # in nonfunctional form, the gradients are accumulated properly for 
        # respective parameters as they are part of linear layer and autograd
        # system handles it normally
        print(f'Encoder Gradients:\n{encoders_weight.grad}')
        print(f'Decoder Gradients:\n{decoders_weight.grad.t()}')
        
    # now lets take one sgd step and see how the shared weights
    # are affected. this shows us whether they are truly shared or not!
    optimizer.step()

    print('\nAfter the optimizer update:')
    print(f'Encoders weight:\n {encoders_weight.detach().numpy()}')
    print(f'Decoders weight(transposed):\n {decoders_weight.t().detach().numpy()}')
    
    print(f'Weight Norms:')
    print(f' shared_weight:   {model.shared_weight.norm()}')
    print(f' encoders_weight: {encoders_weight.norm()}')
    print(f' decoders_weight: {decoders_weight.t().norm()}')
    
    # Heres another check to make sure they all match!
    assert torch.eq(encoders_weight, decoders_weight.t()).all(),'They must match!'

    # note the difference in param count between the two methods
    # this is another of those nuisaunses we face when we bypass the autograd system!
    print(f'\nmodel param count: {sum(p.numel() for p in model.parameters()):,}')
    for name,param in model.named_parameters():
        print(f'{name}:{id(param)} {tuple(param.shape)}')

main(use_functional=True)
main(use_functional=False)

# ! edit
# Ok! so to recap here
# by doing self.encoder.weight.data = self.shared_weight directly we assign 
# the storage of self.shared_weight to self.encoder.weight and as a result
# both self.encoder.weight and self.shared_weight reference the same 
# underlying memory so updates to one will reflect in the other aswell.
# the same rule applies to our decoder's weight (self.decoder.weight) 
# and self.shared_weight.t() (.t() just creates a temporary view, 
# the underlying stoage is the same hence why theres no issue in using transposing in our .data trick!)
# we saw that by doing so Pytorchs autograd system doesnt see/track this manual
# .data assignment, and therefore wont be able to do certain things properly like before
# like tracking these manual operations involved and their gradients however,
# this doesnt pose any issues as gradients are computed independently 
# for self.encoder.weight and self.decoder.weight during backpropagation(because they are
# part of linear module, and autograd system knows them and properly does its job there).
# self.shared_weight.grad remains None though because self.shared_weight 
# isnt directly part of the computation graph anymore (because of .data assignment we did)
# but the encoder and decoder gradients accumulate correctly in self.encoder.weight.grad
# and self.decoder.weight.grad anyway since they are tracked as parameters of their 
# respective layers.
# we also used another check to make sure the weights were shared
# which was the encoder, decoder, and shared weight norms match because
# their storage is shared.
# updates to any one of these will reflect in the others.
# (when optimizer.step() is called, the optimizer updates self.encoder.weight and 
# self.decoder.weight using their respective gradients. since these weights share 
# the same storage as self.shared_weight, the shared weight is implicitly updated as well.)
# 
# recap of recap!:d
# so using .data to share weights allows value synchronization but in doing so bypasses 
# the autograd system, which leads to:
# gradients not being computed for self.shared_weight.
# unlike functional form, we will have independent gradients for self.encoder.weight and
# self.decoder.weight.
# 
# this way, gradients for self.shared_weight are effectively distributed between
# self.encoder.weight.grad and self.decoder.weight.grad.
# If we need gradients for self.shared_weight, we use the functional form or 
# explicitly ensure self.shared_weight is part of the computation graph.
# all things said, itd be better to basically try to avoid .data assignment trick 
# for weight sharing beucase it can lead to weird behaviors, especially in more 
# complex architectures
#%%
# Todo: 
# !edit make this short, and move the full explanation to after the code
# so it doesnt clutter the whole thing!
# also theres a lot of repition and this really needs to be addressed!

# now lets get back to what we were doing and write the loss function.
# but before we commit to that, we need to understand there are two types of sparsity
# when it comes to implementation details.  
# its either sparsity on parameters(weights) or sparsity on representations(activations)
#  
# each of these types serve different purposes and are achieved differently
# sparsity on parameters (parameter/weight sparsity) as the name suggests targets
# the weights of the network and aims to set many of the wights to exactly zero. 
# this is done by using L1 regularization as an additional penalty term 
# alongside the reconstruction loss (e.g. MSE loss) in our loss function.
# L1 regurlarization term penalizes the absolute values of the weights and
# makes the network try to favor more important features, and make other 
# less important ones to go towards zero during training.
# 
# sidenote1:
# we also have sparsity on activation where instead of weights, we use activations values,
# while some of the effects can overlap, they are not the same, and their goals 
# and mechanisms differ.
# we will see this in a moment when we talk about sparsity on representation(more explanation in a moment)  

# this will result in a model with fewer effective connections which help 
# the model to generalize better by focusing only on the important
# features instead of memorizing everything. it also helps save memory since 
# fewer weights need to be stored, and will also reduce computation overhead
# becasue fewer weights need calculations.
# it also makes the model more interpretable because a sparse model is obviously
# simpler now and naturally focuses on the most important connections, making it
# easier to identify which features or patterns(relationships/connections) the 
# model relies on. 
# 
#!edit sidenote2: 
# This is why, we can say, in many cases sparsity effectively performs implicit 
# feature selection. (by eliminating irrelevant or redundant features. (e.g. weights connected to unimportant
# input features may be pruned, which highlights the critical variables that influence 
# the models predictions.)

# therefore parameter sparsity is very useful for things like model compression,
# where we want our models to be light and efficient.
# (also visualizing and analyzing the learned relationships/weight connections will
# be much better/easier as there are fewer interactions to analyze,
# which makes its also useful from the interpretability and analysis of the model point of view
# (give example about llm usgae (like https://transformer-circuits.pub/2024/scaling-monosemanticity/)))
# 
# sparsity on representation (or sparse representation/activation) on the other hand,
# aims to make sure only a small number of neurons in the hidden layers are active for
# a given input.
# like the previous method, this is also done by adding an extra term for sparsity constraint
# to the loss function. 
# this term is usually based on KL divergence and tries to keep the activations low on average(
# each neuron only fire for a subset of inputs. more explanation later on).
# this is done to force the network to focus on capturing the most important features 
# while ignoring redundant stuff.
# 
# sidenote 4:
# note that neurons with sparse activations usually end up having weights that are 
# specialized for certain inputs or patterns, but this doesnt necessarily mean 
# the weights themselves are sparse. for example, a single neuron may very well 
# have dense weights (i.e. non-zero connections to many input features) but activate 
# only for specific patterns in the input.) so sparsity of activations doesnt necessarily
# mean sparsity in weights (although we might see some sparsity there, but its a sideeffect
# not the explicit /direct/intentional effect of this type of sparsity)
# 
# this kind of sparsity therefore is useful for tasks like dimensionality
# reduction, feature extraction, or unsupervised learning when we are trying to learn
# compact and meaningful representations.

# !todo remove 
# sidenote : (from andrewng's standford classnotes on sparse autoencoders 2011)
# ...we will think of a neuron as being "active" (or as "firing")
# if its output value is close to 1, or as being "inactive" if its output value is
# close to 0. We would like to constrain the neurons to be inactive most of the
# time)
# 
# the sparse Autoencoder proposed by Andrew NG() 
# is able to learn a sparse representation and it is well known that l1 regularization
# encourages sparsity on parameters.

#
# ok to recap what we have just covered:
# in sparsity on activations the goal is to make the neuron activations sparse, 
# ensuring that only a small subset of neurons in a layer are active (i.e., non-zero)
# for a given input.
# This is achieved by adding a sparsity term like KL divergence to the
# loss function, which encourages neurons to have low average activation
# (which using sigmoid means fire only for a few samples in the batch (explained more in detail ahead!)).
# 
# Neurons with sparse activations often end up with weights that are specialized
# for certain inputs or patterns, but this doesn’t necessarily mean the weights 
# themselves are sparse.for example, a single neuron may have dense weights 
# (non-zero connections to many input features) but activate only for specific 
# patterns in the input.
#
# in sparsity on parameters however, the goal is to directly make the weights sparse,
# setting many of them to exactly zero(or very close to zero making them practically inactive(i.e. zero!)), 
# regardless of the activations.
# This is achieved by explicitly penalizing the absolute values of weights (using L1 regularization).
# furthermore, sparse weights can indirectly lead to sparse activations because if many 
# connections are pruned (set to zero), the input to some neurons will also 
# reduce. However, this is not guaranteed nor is it the primary goal of sparsity on parameters.
# their primary goal is to lead to fewer effective connections in the model.
#
# moreover, sparsity on activations targets the outputs (neurons' responses), 
# while sparsity on parameters targets the weights (connections).
# Sparsity on activations may result in some weights becoming redundant 
# (effectively sparse), but it doesn't explicitly enforce this while 
# sparsity on parameters directly enforces zero weights but may or may not result in
# sparse activations.
#
# sparsity on activations helps in learning compact, meaningful representations, 
# especially useful in dimensionality reduction and feature extraction tasks.
# sparsity on parameters on the other hand reduces model size, computational cost, 
# and memory usage, making it suitable for resource-constrained environments like 
# mobile or edge devices.
# though today we have other means to make models suitable for such environments, 
# post trainig quantizations and pruning are two examples we will also cover in a 
# later chapter inshaallah)
#
# can sparsity on activations imply sparsity on parameters?
# sometimes yes it does. in cases where the sparsity on activations heavily 
# constrains the neurons, weights connected to consistently inactive neurons 
# may become unnecessary and could be pruned or driven to zero. 
# This can lead to sparsity in parameters as a secondary effect.
# its worth reiterating that this is not always the case and sparse activations 
# may still use dense weights, especially when those weights are necessary to 
# achieve selective neuron activation.
# therefore while sparsity on activations and sparsity on parameters can influence 
# each other, they are quite different and are used to achieve different goals.

# TODO summarize our explanation - its too long!!! 

# now that we know a bit about how this works, lets implement these cases here 
# we will be implementing both the sparsity on parameter and activations. 
# using l1 regurlarization,  gradient sparsity and we also implement kl divergence
# version as well which should give us the best result

#TODO this is ugly as hell, use proper keywords, and better merge this with the actual
# architecture (model) so we dont have seaprate bits and pieces scattered all over!

# we need model to access its parameters as well, so we add model as parameter here
def sparse_loss_function(model, outputs_enc, reconstructed_imgs, imgs, penalty_type=0, l1_weight=0.01, Beta=1):
    """
    penalty_type : 
    0: sparsity on parameter
    1: sparsity on activations 
    2: sparsity using l1 penalty using gradient enforcemet
    3: sparsity using kl divergence
    """
    
    # in all losses we have the basic reconstruction loss, for sparsity
    # we add an additional term.
    criterion = nn.MSELoss()
    reconstruction_loss = criterion(reconstructed_imgs, imgs)
    
    if penalty_type == 0: # sparsity on parameter
        # we enforce a constrain on the model weights/parameters
        # we add all the trainable parameters magnitudes 
        parameters_sum = sum(torch.sum(torch.abs(p)) for p in model.parameters() if p.requires_grad)
        # we can normalize the result so the number of parameters doesnt
        # skew our result (our choice of lambda/Beta)
        # param_count = sum(p.numel() for p in model.parameters())
        sparsity_loss = parameters_sum #/param_count
        # print(f'{reconstruction_loss:.6f} {sparsity_loss=:.6f} {parameters_sum}')
        return reconstruction_loss + (Beta*sparsity_loss)

    elif penalty_type == 1: # sparsity on parameter-using gradient enforcement
        # apply the l1penalty on the weights of our encoder
        # through added term in backpropagation during forward pass
        # here we simply grab the reconstruction loss
        # Compute gradients of encoder output w.r.t. input
        # gradients = torch.autograd.grad(outputs_enc.sum(), model.encoder[0].weight, create_graph=True)[0]
        # print(f'{output.shape=}') # (128,400)
        return reconstruction_loss
    
    elif penalty_type == 2: # sparsity on activation
        sparsity_loss = torch.mean(abs(outputs_enc))
        return reconstruction_loss + sparsity_loss
    
    elif penalty_type == 3:# sparsity on representation/activation
        # for this loss we need to use KL divergence, and
        # calculate what we refer to here as ro^ (ro_hat) which is the
        # mean of activations in our hidden layer (in fact any layer we want sparsity to be
        # enabeled/enforced) and then compare it with a threshold and if its larger than that we penalize the neurons.
        # basically the idea here is that each neuron's activation should be sparse (that is 
        # the activation values need to be close to zero most of the time, but not always(obviously!) and only a few of them be active)
        # this makes/encourages the model to learn and detect more distinct and meaningful features in our traing data.
        # and it goes like this, we first specify a sparsity level/threshold, 
        # which we call ro(ρ) (we choose this threshold (e.g 0.05 to specify 
        # the ratio of sparsity) and it represents the ideal probability of a neuron
        # being active (non-zero). that is we like our neurons to be active 5% of the 
        # times(or 5% of the inputs in ourbatch) and for the remaining 95% of the inputs, 
        # its output should be close to zero.
        #
        # (sidenote: 
        # when we say a neuron is active, we mean that the neuron's output (after the activation function, 
        # i.e sigmoid in our case) is significantly greater than zero.
        # in other words, the neuron is firing (values close to 1) and
        # contributing to the learning process for a particular input. 
        # by setting ro to 0.05, we are basically saying that, on average, each neuron 
        # should be active for only 5% of the inputs or in other words, it should average 
        # to 0.05 across all inputs in the batch which. 
        # (note that averging to 0.05 and being active for 5% of neurons can only be synomous if 
        # two assumptions hold here. first our activation function outputs near-zero values for 
        # most inputs and second, the non-zero activations are significantly larger and sparse (e.g. sigmoid values near 1, by sparse we mean they are few here as the rest are nearly zero! when we have few activations near 1, then by defnition its sparse!).
        # only then the mean (0.05) aligns with the proportion of inputs for which the 
        # neuron is active. 
        # (again think about it this way, if the neurons activation in our batch
        # averages to be around 0.05, it means the absolute majority of its activations have been really
        # tiny (near zero), but a few of them had very large values (close to 1) that averaging them all 
        # resulted in 0.05. if we consider sigmoid here, which outputs 0-1, and can be treated as a probability
        # then we can also say, our neuron here, was active for 5% of the inputs in our batch, hopefully
        # this is clear now!) 
        # this encourages the neuron to be selective in its responses, firing strongly for specific inputs
        # while remaining close to zero for most others.
        #
        # sidenote2: 
        # what if instead of sigmoid we used something liek relu(an unbounded activation 
        # function that doesnt result in probablities)
        # what happens then? as we know in relu, activations are non-negative and unbounded (0 to infinity)
        # in this case, ρ/ro no longer represents a probability but instead it reflects 
        # the desired average activation magnitude across all inputs in the batch.
        # for example, if we set ρ(ro)=0.05, it means the neuron should output values 
        # whose mean is around 0.05, even though individual activations might vary 
        # greatly (some large, some small, many zeros). the sparsity constraint still 
        # works similarly and it encourages most activations to be small or zero,
        # while occasionally allowing higher values.
        # in practice, however, KL divergence is more naturally suited to bounded, 
        # probabilistic outputs (i.e. when we are dealing with probablities and hence using sigmoid/softamx outputs)
        # for other cases, we use other methods for indirectly enforcing sparsity on weights
        # which we covered before (like l1 regularization).
        #  
        # next, we calculate the actual mean activation of each neuron across 
        # the batch. we call this ro_hat (ρ̂).
        # the idea is to compare ro_hat to our target ro. if ro_hat deviates 
        # from ro, we apply a penalty. neurons with ro_hat much larger than ro 
        # are overactive and need to be penalized to push their activations 
        # closer to the target sparsity. 
        # but how do we compare ro_hat and ro? this is where KL divergence comes in. 
        # KL divergence measures the difference between two probability distributions
        # we treat ro as the "true" distribution and ro_hat as the "predicted" 
        # distribution. we want to minimize the KL divergence between them to 
        # ensure that ro_hat is as close to ro as possible.
        # notice ro_hat is a scalar since torch.mean reduces all elements to a single value
        ro_hat = torch.mean(outputs_enc).to(imgs.device)
        # ro is also a scaler, or a weight/threshold
        # previously I had done # ro = torch.ones_like(ro_hat).to(imgs.device) * l1_weight
        # which was unnecessary as ro_hat was a scaler all along!
        ro = l1_weight
        # ro and ro_hat must be probablities, what we have now is just logits
        # so we use softmax to turn our logits into probabilties
        # remember our activation function must be sigmoid 
        # print(ro.shape, ro_hat.shape)
        # now that we have ro and ro_hat as probabilities, 
        # we calculate the KL divergence between them.
        # remember, the KL divergence for Bernoulli distributions is defined as:
        # KL(p || q) = p * log(p / q) + (1 - p) * log((1 - p) / (1 - q))
        # where p is ro and q is ro_hat. this tells us how "far" ro_hat 
        # is from ro. ideally, we want this value to be as small as possible.
        # note that we need to sum up the KL divergence across all neurons, as we want 
        # the overall sparsity penalty for the layer, not just for individual neurons.
        kl = torch.sum(ro * torch.log(ro / ro_hat) +
                      (1 - ro) * torch.log((1 - ro) / (1 - ro_hat)))
        # this sparsity penalty can now be added to the total loss,
        # alongside the reconstruction loss
        # this ensures that the network not only learns its primary task but also 
        # maintains sparsity in the hidden layer.
        return reconstruction_loss + (Beta * kl)
    else:
        raise Exception(f'Unknown penalty type ({penalty_type}) entered!')
    
#%%
# now lets train!
epochs = 20
# penalty_type = 0
# ro 0.01 ~ 0.05 or l1_weight 
# for gradient based constrained the ratio
# needs to be small for our example around 0.0001~1e-4
sparsity_ratio = 1e-4
loss_type = 3
tied_weights = True
# at the end read the Cyclical Annealing Schedule section to get a very good idea about
# how you can achieve better result and why!
# for sparsity on parameters lambda = 1e-6 (loss=0)
# for sparsity on representation(kl divergance) 2
Beta = 1e-6
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sae_model = SparseAutoEncoder(embeddingsize=400,                             
                              tied_weights=tied_weights).to(device)
optimizer = torch.optim.Adam(sae_model.parameters(), lr = 0.01) 
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10) 

print(sae_model)
print(f'param count: {sum(p.numel() for p in sae_model.parameters()):,}')
# lets save the weights of our encoder and decoders before we train them 
# and then compare them with the new weights after training and see how
# they changed!
init_weights_encoder = copy.deepcopy(sae_model.encoder[1].weight.data) 
init_weights_decoder = copy.deepcopy(sae_model.decoder[0].weight.data)
imgs_list =[]
# now lets start training!
for e in range(epochs):
    for imgs,_ in dataloader_train:
        imgs = imgs.to(device)
        output_enc, rec_imgs = sae_model(imgs,
                                         apply_gradient_constraint=(loss_type==1),
                                         l1_weight=sparsity_ratio)
        loss = sparse_loss_function(sae_model,
                                    output_enc,
                                    rec_imgs, 
                                    imgs, 
                                    penalty_type=loss_type,
                                    l1_weight=sparsity_ratio,
                                    Beta=Beta)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'epoch: {e}/{epochs} loss: {loss.item():.6f} lr = {scheduler.get_lr()[-1]:.6f}')
    scheduler.step()
    # at each epoch, we sample one image and its reconstruction
    # for viewing later on to see how the training affects the
    # result we get
    imgs_list.append((imgs[0],rec_imgs[0]))
#%%
plot_embedding_clusters(sae_model, dataloader_train, use_pca=False)
#%%
# note that using tied weights we get a better result and much lower loss, as this acts as a regularizer on
# its own which is not the case when weights are independant and require more trainig/regularization
# also note that the parameter count is not decreased even though we are using shared weights
# this is another nuasce of the nonfunctional method where autograd system is bypassed!
# 
# now lets see how sparse our weights have become
# to calculate this we can simply get the number of zero weights
# and divide them by the total number of weights!
def calculate_sparsity(model, tolerance=1e-5):
    # total parameter count
    total_weights_cnt = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # treat values within our tolerence as zero
    # (i.e. values too close to zero are treated as zero)
    # we do this because of floating point number funkiness! (we dont get exact matches)
    zero_weights_cnt = sum(torch.sum(torch.abs(param) < tolerance).item() for param in model.parameters())
    sparsity_percentage = (zero_weights_cnt / total_weights_cnt) * 100
    return sparsity_percentage

# to make it a bit more detailed, lets show them in a layerwise fashion
def display_layer_wise_sparsity(model, tolerance=1e-5):
    for name, module in model.named_children():
        sparsity_precentage = (calculate_sparsity(module, tolerance))
        print(f'Layer {name}: Sparsity: {sparsity_precentage:.4f}%')

tolerance=1e-4
sparsity_precentage = calculate_sparsity(sae_model, tolerance)
print(f'sparsity_precentage={sparsity_precentage:.6f}')
display_layer_wise_sparsity(sae_model,tolerance)

# now lets first visualize the sparsity, image/reconstruction pairs and how they look : 
# lets simply show a histogram of our models weights this should give us a good idea 
# about how sparse the weights have become  
def plot_weight_distribution(model):
    all_weights = list(p.detach().cpu().numpy().flatten() 
                       for p in model.parameters() 
                       if p.requires_grad)    
    all_weights = np.concatenate(all_weights)
    plt.hist(all_weights, bins=100, range=(-0.4, 0.4))
    plt.title("Weight Distribution")
    plt.xlabel("Weight Value")
    plt.ylabel("Frequency")
    plt.show()

plot_weight_distribution(sae_model)
#%%
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
#%%
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# def visualize_grid(imgs, label, rows=20, cols=20):
#     fig = plt.figure(figsize=(10, 10))
#     imgs = imgs.cpu().numpy().transpose(0, 2, 3, 1).squeeze()
#     plt.title(label)
#     for i in range(imgs.shape[0]):
#         ax = fig.add_subplot(rows, cols, i+1, xticks=[], yticks=[])
#         img = imgs[i]
#         # normalize to 0-1 range
#         # Add small epsilon to avoid division by zero
#         img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)
#         # print(f'{img.min()=:.4f} {img.max()=:.4f}')
#         ax.imshow(img, cmap='Greys_r')
# 
# lets make it a bit better and add a colorbar so 
# we can make out the color values
def visualize_grid(imgs, label, rows=20, cols=20):
    fig = plt.figure(figsize=(10, 10))
    imgs = imgs.cpu().numpy().transpose(0, 2, 3, 1).squeeze()
    plt.title(label)
    # normalize the colorbar scales based on the image min/max values
    # since we need to work with 1 min/max we use the global min/max
    # but here we inidividually normalize the images int 0-1 so we
    # we can ignore this
    # global_min = np.min(imgs)
    # global_max = np.max(imgs)
    # norm = Normalize(vmin=global_min, vmax=global_max)
    for i in range(imgs.shape[0]):
        ax = fig.add_subplot(rows, cols, i+1, xticks=[], yticks=[])
        img = imgs[i]
        # normalize to 0-1 range (already normalized globally using `norm`)
        # if we were to use the global min/max we would do 
        # img = norm(img)
        # but now we do this like before
        img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)
        ax.imshow(img, cmap='gray')

    # Add a single color bar to the right side (left, bottom, width, height)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  
    # we would have used norm, here if we used the global norm, but since we didnt
    # we can simply use None!
    cbar = plt.colorbar(ScalarMappable(norm=None, cmap='gray'), cax=cbar_ax)
    cbar.set_label('Pixel Value Range')  # Label for the color bar
    plt.subplots_adjust(wspace=0.0, hspace=0.0, right=0.9)  # Adjust space to fit color bar
    plt.show()

# we could combine all images and get a final image, everything stays the same!
def visualize_grid0(imgs, label, rows=20, cols=20, normalize=True):
    # normalize the images
    imgs = imgs.cpu().numpy().transpose(0, 2, 3, 1).squeeze()
    # we add a + 1e-8 so in case we have 0 in the denominator, we dont face any errors
    imgs = [(img - img.min()) / (img.max() - img.min() + 1e-8) for img in imgs]
    # combine all the images into a single big image
    height, width = imgs[0].shape[:2]
    
    # the placeholder for our larger image which contains all our images
    big_img = np.zeros((height * rows, width * cols), dtype=np.float32)
    for idx, img in enumerate(imgs):
        if idx >= rows * cols:
            break
        row, col = divmod(idx, cols)
        big_img[row * height:(row + 1) * height, col * width:(col + 1) * width] = img

    # plot the image
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[])
    ax.imshow(big_img, cmap='gray')
    ax.set_title(label)
    # images are already normalized so no need for a normalizer here
    # sidenote: 
    # the colormap Greys is not the same as gray!
    # using the gray colormap (cmap='gray'), by default maps 
    # lower intensity values (0) to black and higher intensity values (1) to white.
    # In Greys, the color mapping is reversed, with lower intensity values (0) mapped
    # to white and higher intensity values (1) mapped to black.
    # when using the outcome may look the same, but the interpertation differs)
    cbar = plt.colorbar(ScalarMappable(norm=None, cmap='gray'), ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Pixel intensity (Normalized)' if normalize else 'Pixel Intensity')


# we can also do this using opencv and can easily see the pixel values
# by zooming in!(press q to close the current image window and see the next ones)
import cv2
def visualize_grid_cv2(imgs, label, rows=20, cols=20):
    # normalize the images
    imgs = imgs.cpu().numpy().transpose(0, 2, 3, 1).squeeze()
    imgs = [(img - img.min()) / (img.max() - img.min() + 1e-8) for img in imgs]
    # combine all the images into a single large image
    height, width = imgs[0].shape[:2]
    big_img = np.zeros((height * rows, width * cols), dtype=np.float32)
    for idx, img in enumerate(imgs):
        if idx >= rows * cols:
            break
        row, col = divmod(idx, cols)
        big_img[row * height:(row + 1) * height, col * width:(col + 1) * width] = img

    # convert to 0-255 (uint8)
    big_img = (big_img * 255).astype(np.uint8)
    
    cv2.imshow(label, big_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# and finally we could have also used pytorch's makegrid to do the same thing
# note that the image is different because of the way the images are normalized
# in pytorch using global normalization, ie. unlike what we have done so far,
# it first combines the images, creating a big image, and then normalizes that
# big image with the min/max of it which is the min and max among all images
# this is obviously different than normalizing each image individually using its onw
# min/max lvaues. 
def visualize_grid0(imgs, label, normalize=True):
    fig = plt.figure(figsize=(10, 10))
    imgs = imgs.cpu()  
    plt.subplots_adjust(wspace=0, hspace=0)
    
    img_grid = torchvision.utils.make_grid(
        imgs, nrow=20, normalize=normalize).numpy().transpose(1, 2, 0)
    # print(img_grid.min(), img_grid.max())
    # normalize the colorbar scales based on the image min/max values
    if normalize:
        norm = Normalize(vmin=0, vmax=1)
    else:
        norm = Normalize(vmin=img_grid.min(), vmax=img_grid.max())
        
    ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[])
    ax.imshow(img_grid,'gray')
    ax.set_title(label)
    
    cbar = plt.colorbar(ScalarMappable(norm=norm, cmap='gray'), ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Pixel intensity (Normalized)' if normalize else 'Pixel Intensity')

trained_W_encoder = sae_model.encoder[1].weight.data.cpu().clone().reshape(sae_model.encoder[1].out_features, 1, 28, 28)
trained_W_decoder = sae_model.decoder[0].weight.data.cpu().clone().reshape(sae_model.decoder[0].in_features, 1, 28, 28)
init_weights_encoder = init_weights_encoder.reshape(sae_model.encoder[1].out_features, 1, 28, 28).cpu()
init_weights_decoder = init_weights_decoder.reshape(sae_model.decoder[0].in_features, 1, 28, 28).cpu()

w_diff_encoder = init_weights_encoder - trained_W_encoder
w_diff_decoder = init_weights_decoder - trained_W_decoder

w_decoders_transposed = sae_model.decoder[0].weight.data.cpu().clone().t()

# in order to see that decoders weight is infact the same as
# encoders, lets transpose it again and reshape it.
# here I show both the encoders weight and our decoders weight
# transposed! 
print(f'{trained_W_encoder.shape=}')
print(f'{w_decoders_transposed.shape=}')
w_decoders_transposed = w_decoders_transposed.view(sae_model.encoder[1].out_features, 1, 28, 28)
# note that the decoder weights (in terms of original data) will be smoothed encoders weights
# (also in terms of original data)
# info from : https://medium.com/@SeoJaeDuk/arhcieved-post-personal-notes-about-contractive-auto-encoders-part-1-ef83bce72932 
# end of the page, in the ppt slide image

print(f'{init_weights_encoder.shape=}')
visualize_grid(init_weights_encoder, 'Initial weights')
visualize_grid(trained_W_encoder, 'Trained weights(Encoder)')
visualize_grid(w_diff_encoder, 'weights diff (Encoder)')
visualize_grid(trained_W_decoder,'Trained Weights (Decoder)')
visualize_grid(w_decoders_transposed,'Trained Weights (Decoder-transposed)')
# after normalization, the white spots denote 1/255, and black areas denote 0 (close to 0)
# anything in between (i.e. gray) shows the numbers in between.
# (if unnormalized the black shows negative values, and white show positive values
# and the gray shows zero values.)
# we start from a high positive and high negative values in our initial
# weights. and then after training and imposing sparsity we can see that
# we are mostly seeing gray colors which indicate the values are zero!
# and that is what we were after!(unnormalized visualization)
# if you look at the w_diff, you can see that there are lots of high and
# low (negative) values as well. this is becsaue  in order to make the
# weights have more reasonable weights, they had to be decreased/increased
#%%
#! edit choose different types and see which one gives us the best pretraiing result
#! this should give us a better intuition as which one is best for this if we had the right intuition before(explanation in loss section)
# 
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
# using add_module. because if we simplt do sth like:
# sae_model2.classifier = nn.Linear(sae_model2[0].out_features, 10)
# classifier will be just an attribute, and for the forward pass we
# need to do sth like
# output=sae_model2.forward(input)
# output = sae_model2.classifier(output)
# so this is not ideal at all. therefore we do:
sae_model2.add_module('classifier', nn.Linear(sae_model2[0][1].out_features, 10))
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
    print(f'epoch: ({e}/{epochs}) acc: {acc*100:.4f} loss: {loss.item():.6f} lr: {scheduler.get_lr()[-1]:.6f}')
    scheduler.step()


# now you can try it without running the autoencoder training and 
# see how it performs.
# without pretraining (i.e. trainig sparseautoencoder first)
# this is what we get:
# epoch: (0/20) acc: 70.3653 loss: 0.887669 lr: 0.100000
# epoch: (1/20) acc: 79.6976 loss: 0.591737 lr: 0.100000
# epoch: (2/20) acc: 82.3894 loss: 0.516348 lr: 0.100000
# epoch: (3/20) acc: 84.0965 loss: 0.648719 lr: 0.100000
# epoch: (4/20) acc: 85.3295 loss: 0.411724 lr: 0.100000
# epoch: (5/20) acc: 85.9312 loss: 0.439958 lr: 0.001000
# epoch: (6/20) acc: 86.0341 loss: 0.512782 lr: 0.010000
# epoch: (7/20) acc: 86.1232 loss: 0.448194 lr: 0.010000
# epoch: (8/20) acc: 86.2333 loss: 0.557446 lr: 0.010000
# epoch: (9/20) acc: 86.3046 loss: 0.428177 lr: 0.010000
# epoch: (10/20) acc: 86.3592 loss: 0.440234 lr: 0.000100
# epoch: (11/20) acc: 86.3648 loss: 0.394169 lr: 0.001000
# epoch: (12/20) acc: 86.3776 loss: 0.358289 lr: 0.001000
# epoch: (13/20) acc: 86.3904 loss: 0.359354 lr: 0.001000
# epoch: (14/20) acc: 86.3877 loss: 0.469507 lr: 0.001000
# epoch: (15/20) acc: 86.3971 loss: 0.460794 lr: 0.000010
# epoch: (16/20) acc: 86.3977 loss: 0.474029 lr: 0.000100
# epoch: (17/20) acc: 86.4016 loss: 0.335743 lr: 0.000100
# epoch: (18/20) acc: 86.4005 loss: 0.519619 lr: 0.000100
# epoch: (19/20) acc: 86.4055 loss: 0.347018 lr: 0.000100
# but if we first trained our sparseautoencoder and then ran the classification
# we would get 
# epoch: (0/20) acc: 87.8931 loss: 0.334116 lr: 0.100000
# epoch: (1/20) acc: 90.2050 loss: 0.276905 lr: 0.100000
# epoch: (2/20) acc: 91.3398 loss: 0.350305 lr: 0.100000
# epoch: (3/20) acc: 92.1424 loss: 0.247515 lr: 0.100000
# epoch: (4/20) acc: 92.6760 loss: 0.159887 lr: 0.100000
# epoch: (5/20) acc: 92.9753 loss: 0.202914 lr: 0.001000
# epoch: (6/20) acc: 93.0326 loss: 0.191703 lr: 0.010000
# epoch: (7/20) acc: 93.1244 loss: 0.269451 lr: 0.010000
# epoch: (8/20) acc: 93.1579 loss: 0.239176 lr: 0.010000
# epoch: (9/20) acc: 93.2074 loss: 0.252368 lr: 0.010000
# epoch: (10/20) acc: 93.2336 loss: 0.353381 lr: 0.000100
# epoch: (11/20) acc: 93.2353 loss: 0.301774 lr: 0.001000
# epoch: (12/20) acc: 93.2347 loss: 0.390216 lr: 0.001000
# epoch: (13/20) acc: 93.2347 loss: 0.303082 lr: 0.001000
# epoch: (14/20) acc: 93.2336 loss: 0.321064 lr: 0.001000
# epoch: (15/20) acc: 93.2375 loss: 0.203758 lr: 0.000010
# epoch: (16/20) acc: 93.2370 loss: 0.294605 lr: 0.000100
# epoch: (17/20) acc: 93.2347 loss: 0.250669 lr: 0.000100
# epoch: (18/20) acc: 93.2364 loss: 0.270309 lr: 0.000100
# epoch: (19/20) acc: 93.2370 loss: 0.173221 lr: 0.000100
# not only we started with a much higher accuracy(~20% higher), 
# we also achieved higher accuracy at the end.obviously we used
# the bareminimum, using better architecture, better training regime
# the results can get better.


# finally  this was a simple autoencoder, we can have several layers
# and also you can use batchnormalization, etc for your deep autoencoders as well
# needless to say, from coding prespective we have a horrible code base
# which can be improved a lot! but for now it suffices as we were after the
# core concepts of the autoencoders for a real world scenario we take our time
# and code properly so it is maintainable and easy to understand and follow!
# !TODO: refactor codes, and make them more presentable while keeping it simple

#%%
# -VAE (Variational Autoencoders)
# -Creating MNIST Like digits
# -The Reparametrization Trick


# intori shoro konim
# I guess if we started the introduction with intuitions it would be better
# When talking about VAEs, we come accross two view points.
# (there are two common themes when you search for vae explantions)
# one that involves the underlying differences between VAEs and other types of autoencoders, 
# and the other, a somewhat higher level prespective which is more involved 
# in terms of how it works from distribution point of view. 
# I'll be explaining these two common view points, and hopefully at the end
# we will have an in-depth and rigiours understanding of VAE fundamentals and their inner workings
# this should give us a much better understanding that should come in handy later in 
# our researches. 
# 
# TLDR-brief introduction:
# The Variational Auto Encoders(VAE) are different with conventional autoencoders in that,
# the encoder does not create a single latent vector representation, instead, it creates
# two! 
# one for mean and another for standard deviation. these are in fact parameters that are 
# used to sample from a normal/gaussian distribution from which we get the latent vector from 
# which the decoder accepts as input and tries to reconstructs the input from.
# The encoder creates different mean/stds for each class by which we can generate
# samples similar to said classes. more importantly, because of the way VAE is built, its
# possible to go from one class to another in a gradual manner, which means we can actually
# have new variations in input that does not exist in the dataset explicitly.
# with this overal and coarse introduction of the VAE out of the way, lets get to the finer
# details!
# 
# In depth explanation: 
# VAEs, clustering of latent representation/spaces? 
# Where does a VAE concept comes from? 
# from an application point of view, initially we wanted to create 
# random images just like the ones in our datasets this was usually to create 
# synthetic data for training purposes, or extracting somewhat meaningful features
# or pretraining our model before doing the actual training 
# (back in the day most of the time as training was very hard 
# due to vanishing/exploding gradient issues at the time. its still used as 
# well especially in llm domain! though not for the mentioned issues, but 
# having a basic knowledge base for further manipulation)
# a bit later we found that, creating random data(images mostly at first) isnt 
# really that attractive, and we can actually do much more and much better, 
# for one, what about experimenting with controling the generation process 
# and attemping to create all sorts of things?!
# this becomes especially useful/important if we can change or alter the data we 
# already have! for example, imagine adding beard to your image, retouch it, see how
# you look with glasses on, etc all sorts of things, as you can imagine, this is a 
# lot more useful, and has a lot of real-world applications.
# VAEs and the likes (conditional VAEs, other types of generative models for that matter)
# have come for this goal!(sort of!)

#!EDIT
# what does make VAE especial you may ask? 
# So far we have implemented and trained different types of autoencoders, regardless of their
# main differences, (sparse/denoising/etc) one thing that they had in common was that
# when we look closer at their latent representations, we notice the encoder latent representation
# (encodings) formed distinctly clustered subspaces for each class. 
# if you think about it, this makes prefect sense, as distinct encodings for each image 
# type(or any data really) makes it much easier for the decoder to decode it
# it also aligns very well with our goal of replicating the same images.
# However, when we decide to build a generative model, where we want to create
# different 'variations' of the same image class or data, we dont just want to generate
# the same image we find in our dataset. 
# to this end, we would like to be able to generate variations on an input image, variations
# from the whole dataset, that does not explicitly show up in one image, 
# it would be great if we could, combine different features from different classes, 
# and still have a pretty realistic outcome. 
# this means from a technical prespective, to be able to move smoothly in the latent space,
# and be able to sample from any part of it. 
# sampling like this means, we could generate completely novel images that dont exist explicitly
# in our dataset, depending on where we sample from in our latent space, between which latent clusters.
#   
# our latent space therefore needs to be continuous otherwise, if it has
# gaps between clusters or in other words, discontinuities, and we try to generate a 
# variation from there part, the decoder will simply generate an unrealistic output, 
# because it does not know how to deal with that region of the latent space 
# since during training, it never saw encoded vectors coming from that region of latent space.
# this is why having a 'continuous' latent space is crucial here. 
# This is what that differntiates VAEs from conventional autoencoders (basically any generative 
# model for that matter). this is the core idea behind a VAE.
#
#! sidenote: edit this and add a plot to make it concrete
# you can think of continuous in its raw form, imagine it and contrast it against the discrete!
# like, imagine if we have a 2d plot, and we plot different samples, lets say dogs, in point a 
# and point b, a cat in point c, a car in point e, etc when we say we want a continuouse space,
# it means, all the points between point a and point b, should be valid points, and result in
# dogs as well(thats interpolating gives us valid samples all the way from a to b)! in descrete mode, its dog a, nothing, and then suddenly dog b, likewise, from 
# point b to c we should see inifint points from b up to c,which shows us dogs, all the way
# to a cat (as we get close to c, dogs look more like cats, until ultimately its just our cat
# we plotted in point c! sampling allows us to do this! does it now make sense?)
#  
#
# VAEs offer an intersting approach here, instead of mapping inputs to fixed points in
# the latent space (like traditional autoencoders), they say lets map inputs to probability
# distributions! specifically, a Gaussian/Normal distribution.
# This way we ensure the latent space is continuous and smooth, without any gaps or discontinuities.
# Therefore, when we randomly sample from the latent space, the decoder can generate 
# realistic outputs, even for points it has not explicitly seen during training.
# as the decoder has learned to generalize across the entire latent space, rather than just
# memorizing specific points.
#
# That is the gist of the vae, the actual process is very simple for the most part, 
# during training, the encoder doesnt just output a single latent vector, instead, 
# it predicts two vectors, the mean(μ) and standard deviation(σ) of a Gaussian distribution
# for each input(x_i).
# The latent representation vector z_i, is then sampled from this distribution. 
# (in practice however we need to use a process called the reparameterization trick 
# to get around a technical detail we will be getting into in a moment other than that
# this is pretty much it!).
# This ensures that the latent space is smooth and continuous, as each point in the 
# latent space corresponds to a valid potential data point.
# 
# sidenote: (edit)
# The sampling process (also refered to as stochastic generation by some reasearchers)
# means, the actual encoding will be different slightly at each forward pass,
# even for the same input, with the same mean and standard deviation, hence the name!
# we see the implication of this and why this is desirable for us in a moment)
#  
# what we do here, is in fact a form of regularization. this regulariziation allows VAEs
# to have meaningful interpolation and sampling. for example, if we move smoothly between
# two points in the latent space, the generated output transitions naturally between the
# two corresponding data points.
# also randomly sampling points from the latent space produces realistic variations,
# as every region of the space has been trained during the models learning process 
# (this works even if the combination of some attributes does not exist in our dataset
# explicitly, infact this is the actual case here, this is what we were after all along!).
# 
# (sidenote: this happens if and only if the model is trained prefectly, we can see this 
# in easy/simple datasets more easily, but for complex datasets it becomes very hard, we'll
# see this in our experiments first hand!)
#
# This regularization effect is achieved using a KL divergence loss, which encourages the
# learned latent distributions to remain close to a standard Gaussian prior (i.e., 
# a standard normal distribution). (informally speaking, this means the latent variables 
# cluster around the center of the space (around 0), resembling the properties of a 
# standard normal distribution)
# 
# (why? see the explanation in implementation below)
# This ensures latent space is well organized and nearby points in the latent space 
# correspond to similar outputs. This not only avoids gaps in the latent space but also
# encourages the model to generalize better when generating new data.
#
# sidenote- second prespective (application prespective/underthehood - explained more):
# lets view this from another angle, why does this makes sense?
# why would we want to have a distribution instead of fixed points, what do we get by doing it?
# lets make this more tangible by an example.
# remember we said earlier we want to be able to control variations in our input data? 
# like we want to make for example a person smile, or we want to add a mustache to 
# someones face. having a distribution instead of a fixed point allows us to 
# have different smiles, different mustaches and not just a single one.
# like mona lisa is also smiling, marlyn monroe(add her pic!) is also smiling, they are clearly 
# different smiles so a distibution for smile, would allow us to sample different samples of smiles
# for the lack of a better word and for our mustache example, we can specify different kinds of 
# mustaches small, big, fancy, etc and this applies to just everything and the great thing about it is, 
# there does not have to be an explicit image in our dataset for it! 
# imagine mona lisa sporting a mustache!:))
# the mustache is in the dataset, there are many images of men having mustaches of different kinds
# but no mona lisa!(or women for that matter hopefully!) or imagine glasses, hats, beard, etc! you get the point. 
# this happens because, as we previously mentioned, the latent space is smooth and continuous and 
# the decoder has also learned to generalize across the entire latent space, instead of just memorizing
# specific points in said latent space. add these to the fact that each point in the 
# latent space also corresponds to a valid potential data point, and you get the ability to roam that sapce
# and sample from it! all forms of variations can be achieved using this, gradually moving from one thing
# in one subspace toward another thing(subspace), and yet have a somewhat sensible output is what this gives us!
# 
# (sidenote: in practice however, this is extremely hard to achieve for anything mildly complex! we have much
# better chocies than vae, but from theorectical point of view, this is what we expect given the
# concept and whats involved. (the training can be notouuriously hard to get things working, lets not
# ahead of ourselevs, and for now lets keep building intuition for now))
# now back to the main point: 


# I find jeremy's phenamonal writeup on vaes to be especially great: 
# ref https://www.jeremyjordan.me/variational-autoencoders/
#
# so a second summary: 
# our encoder recieves the input and produces two vectors
# one for mean and another for std(in fact our encoder creates log variance which we
# then convert to standard deviation to then use for sampling, so technically
# speaking it creates mu and logvar in the network, but for doing its job it creates std!).
# this is in contrast to how a traditional autoencoder works.
# a traditional autoencoder creates a set of atttibutes 
# in its final representation vector(e.g attributes or features describing
# concepts such as, eye, smile, beard, gender, has glasses etc) in 
# an input image of faces.
# the idea here is the autoencoder (hopefully) learns descriptive attributes 
# of the input(in the case of faces this may be skin color, whether or not the person
# is wearing glasses, is female, etc) to describe an observation(i.e. our input image)
# in a compressed representation.
# 
# for example one such latent vector could be something like (gender:-0.73, smile:0.99, glasses: 0.002, etc )
# which is basically describing the input image in terms of its latent attributes,
# each of which are described by a 'single value'.(note this, its important for our discussion)
#
# (sidennote:(edit maybe its better if I add them as footnote?! or atleast some of these sidenotes are btter off as footnotes?!)
# in reality, however, we can not rely on this intuition that a single feature directly
# describes a single feature in the input, most often, its the combinations of several features
# that specifies the existence of a certain feature in the input data, but for the sake of explanation
# imagine this is the case so we can convey the idea behind it)
# 
# However, this may not reflect the variety/dynamic range of our input properly(this may be very limiting), 
# because the compression by nature limits the amount of attributes we can encode.
# to get a broader range, we would need to have a larger number of attributes, and that
# would mean less compression, and in turn less desired(varied) output, soon we will 
# be standing before a decision, to what extend can we compress and what features(diversity)
# can we have? do we use smaller number of attibutes and be limted?
# or use a larger number and face more training issues(issues, explain like what edit)?
# 
# therefore we may prefer to represent each latent attribute as a 'range of possible values'
# instead of simply a 'single one'.
# this would relax the previous limitation, as it can now encode a broader range of 
# variation/retain dynamicity! all while the number of attributes per say would remain intact!/unchanged!
#
# For instance, suppose we want to assign a value for the smile attribute for the image
# of mona lisa, what 'single value' should we assign to reflect her smile? its there, but
# at the same time its very underdefined! as if shes not smiling at all!
# if we were to use attributes that indicate the existance of a feature in input, her smile,
# would expectedly get a small value, and be treated as non existant) if we assigne somewhat
# higher value, then it would mess with the existing established rule/attributes that rightfully
# detect images that have defined smiles and would lead the network to wrongly classify similar 
# features as smile!
# hence why having a range of values would be very benificial to us where we can describe
# different types of an attribute (here smile e.g.). 
# we can achieve this by using probabilistic terms in our work.
# 
# the mean and variance that we produce in a vae encoder, is used exactly for this very 
# reason. using them, we are learning 'a distrubution' for 'each attribute' and thus 
# mu and variance specify a range of values for each attribute.
# 
# [With this approach, we'll now be able to represent each latent attribute for a given
# input as a probability distribution. when decoding from the latent representation we'll
# randomly sample from each latent attribute distribution to generate a vector as input
# for our decoder.]
# !edit(excessive or misplaced?)
# thats why later on, we use these means, variances(actually std) along with an epsilon(act as a random variable)
# to reconstruct the input image.
# 
# this simply means by producing probablity distribution for each latent attribute, 
# "we're essentially/practically enforcing a continuous, smooth latent space representation."
# This means the decoder should be able to accuractly reconstruct the input by sampling from
# these latent distributions. This also implies that the values that are close
# to eachother, in latent space will correspond to very similar reconstructions.(i.e. 
# should result in similar reconstructions)
# 
# all of this is is made possible by using the mean and variance produced in the encoder. 
# the mean controls where the encoding (value) for an input should be centered around, while
# the standard deviation controls/specifies the (valid) area (of change) around it, i.e. 
# how much from/how far from the mean the encoding can vary.
# sampling using mean and std is akin to randomly generating the encodings inside a circle (distribution)
# which causes the decoder to learn that not only a single point in the latent space 
# refers to a sample of a calss, but also all nearby points (all the points close to it) do as well!
# not only this allows the decodeer to decode single, specific encodings in the latent space 
# but also the ones that slightly vary too(i.e. the ones close to it), as the decoder is 
# exposed to a range of variations of the encoding of the same input during training 
# (each time we feedforward a specific sample, the sampling process introduces a slightly
# different value using the same mu,std (it wont be the same number) although the input 
# sample is the same.)
# This exposes the model to a certain degree of local variations, resulting in a smooth latent 
# space locally(i.e. on a local scale), (that is for similar samples) but at the same time leavs
# the decodable latent space discontinuous so different classes can form their own subspaces, otherwise if 
# they all are mushed up, it would become meaningless! and nearly impossible for the decoder to reconstruct
# accurately (more on this later))
#
# aside from that/moreover, we'd also want overlap between samples that are not very similar aswell, 
# in order to interpolate between classes.
# 
# However, since by default there is no limit/constraint enforcing mean(μ) and std(σ) vectors 
# to have specific values, the encoder can learn to generate different means μ for different classes, 
# clustering them apart, and at the same time minimize std(σ), leading to the encodings that don’t
# vary much for the same sample (which translates to less uncertainty for the decoder and thus 
# easier decoding). 
# This allows the decoder to efficiently/easily reconstruct the training data,
# but it is not desirable for us, as we discussed before we want the encodings to be as close as 
# possible yet be still distinct, allowing smooth interpolation between them, creating new samples.
# Therefore in order to prevent this, we introduce the KL divergence and use it in
# the loss function. The KL divergence measures how much two probablity distributions diverge
# (differ) from each other.
# !edit
# Minimizing it means the probability distribution parameters (μ and σ) need to closely resemble
# that of the target distribution(i.e. our original input data).
# that is they need to be as close as possible (basically resemeble/match? the original data)
# 
# from a visualization point of view, (if we try to visualize the encodings spaces we see) 
# it encourages the encoder to distribute all encodings (for all types of inputs,), evenly 
# around the center of the latent space (this makes the encodings to be distributed evenly 
# around the center of latent space (visually speaking) (edit: basically to have mean 0, (which means a normal distribution,
# which again because natural images follow normal distribution so it makes sense!) hence why they cluster at the center)).
# the encoder will therefore be penalized when/if it tries to cluster them apart into specific regions, 
# away from the origin.
# 
# However, in practice, with this change, the decoder will have a very hard time to get reconstructions right!
# if any atall, simply because the encodings are now simply densely placed randomly, near 
# the center of the latent space, with little to no regard for similarity among nearby encodings.
# to the decoder, this simply doesnt make much sense! based on our previous intuitions, 
# nearby points in latent space should resemble similar inputs, but now, after such enforcement,
# they are being placed at random places! where they have no bussiness being!)
# !edit
# therefore, we use another term in our loss function to circumvent/to get rid of/address this issue. 
# the reconstruction loss, like standard autoencoders will be made of both the BCE loss(because it treats
# pixels as probabilities and prevents blurry outputs and usually results in better performance.) 
# and the kl loss (constraining term). this results in [the generation of] a latent space that 
# addresses both of our concerns and fullfills them both(edit!), 
# maintaining the similarity of nearby encodings locally (on the local scale) by clustering,
# and yet globally, densely packing them near the latent space origin (see visualization).
# 
# sidenote:
# we use BCE because the original paper uses BCE, but some implementations started using MSE
# BCE is preferred in cases where we are dealing with normalized images between 0 and 1
# like binary or grayscale images, where the goal is to predict whether a pixel is closer to
# 0 (black) or 1 (white).(because in BCE each pixel value is seen as probabilities, it makes it especially suitable)
# MSE however, assumes continuous values, meaning it penalizes small differences more harshly,
# which may lead to blurry reconstructions. for example if we had an image where a 
# pixel was 0.9 and the predicted value was 0.8, BCE would penalize the small difference in
# a way that maintains a sharp reconstruction however, MSE, might have lead to an average of
# multiple possible outputs, causing blurry reconstructions. (we see this in our trainig)
# 
# having this said, you can see MSE being used with color images(especially in GANs), especially the ones that
# are not normalized in 0-1 (they are either unbounded, or are normalized [-1,1] 
# it produces smoother but sometimes blurrier reconstructions.)
# 
# so MSE tends to work better for smooth images, while BCE works well when pixel values behave
# like probabilities (high contrast regions, thresholded images, etc).
# !EDIT !EDIT !EDIT
# (we used mse with cifar10 dataset and with images in range (0-1) so its not a hard requirement
# though it might be a good idea to follow and get good result, )
#
# this is the equilibrium/fine balance reached by the cluster-forming nature of the
# reconstruction loss, and the dense packing nature of the KL loss, which forms distinct
# clusters that the decoder can decode.
# This means when randomly generating, if we sample a vector from 
# the same prior distribution of the encoded vectors, N(0, I),(natural images have
# normal distribution (unit normal distribution? applies to them as well)) 
# the decoder will successfully decode it. And if we're interpolating, there are 
# no sudden gaps between clusters, but a smooth mix of features a decoder can understand.

#! add edits from the second part of explanations, where I talka bout posterior distribution(q(z|x)
# to make the explanations here clearer for everyone.())
############################
    # recap of recap (more technical explanation):
    # our encoder(denoted as qθ(z∣x) (i.e. given this input data x, what is the
    # probability distribution of the latent variable z (i.e. whats the mu,var)(note its in log form!but anyway lets carry on(add this as footnote))) 
    # will return two vectors one for μ(mu) and another for standard deviation σ(sigma).
    # using these two parameters, we sample our z representation vector(latent vector)
    # which will be used by the decoder to reconstruct the input.
    # 
    # sidenote:
    # you may see phrases such as "The lower-dimensional space is stochastic" or 
    # "the latent representation is stochastic", these and similar phrases 
    # simply refer to the fact that that the representation in the
    # lower-dimensional space(z) is not deterministic or fixed as we already discussed
    # instead, it involves randomness/uncertainty because its modeled probabilistically.
    # our encoder doesnt directly output z it outputs the parameters of the probability
    # distribution qθ(z∣x) which we then use to sample from to produce the latent vector z 
    # hence the phrase stochastic, because sampling is involved and it changes each time
    # (it changes each time even for the same input because we use a random variable along side thme!)
    #

    # new edit:
    # The decoder (denoted as pϕ(x∣z)) will take a latent vector z,
    # sampled using the mean (mu) and standard deviation (std) from the previous step (encoder's output).
    # The decoder output is the parameters of the probability distribution of the reconstructed data.
    # that is, the decoder outputs parameters (i.e. probabilities) for each pixel in the image.
    # to make this more intuitive and easier to understand, consider the MNIST dataset
    # as an example. MNIST images are grayscale(.i.e. balck and white), and each pixel
    # is represented as a value between 0 and 1. 
    # the probability distribution of a single pixel can then be modeled as a bernoulli
    # distribution. (becasue we have two outcomes (its either 0 or 1))
    # furthermore, MNIST images are 28x28x1, meaning each image has 784 pixels in total, which
    # translates to an input dimensionality of 28x28x1 = 784.
    # The decoder takes the latent representation z as input and ultimately outputs a vector 
    # of size 784. This vector represents 784 bernoulli parameters, one for each pixel in the image.
    # simply put, the decoder 'decodes' the numbers in vector z into 784 numbers between 0 
    # and 1 in the output, where each number corresponds to the probability of a pixel being 
    # "on" (1) or "off" (0).

    # sidenote:
    # The information from the original input (784-dimensional vector in our case) can not be
    # perfectly preserved, because the decoder only has access to a compressed summary of the
    # original data represented as the lower-dimensional vector z.
    # This lossy compression expectedly leads to some loss of details (depending on the amount
    # of compression of course), as z is designed to only capture the most essential features 
    # of the input and discard the less important/ less critical ones.
    # Therefore the quality of this representation depends on how well the encoder-decoder
    # pair is trained to balance reconstruction accuracy with the constraints of the 
    # lower-dimensional space (i.e the right choice for the amount of compression (size of vector z,
    # as too few parameters may very well be insufficient to yield the desired output)
    # 
    #! sidenote2:!Edit - excessive remove
    # we can measure the quality of the reconstruction process and see how well
    # our model is doing by using the log-likelihood logpϕ(x∣z), which quantifies how well
    # the decoder has learned to map the latent representation z back to the original input x. (use latent vector z instead?)
    # The units of logpϕ(x∣z) are nats(its measure of information content).
    # Higher values mean the reconstructed data closely matches the original, signifying 
    # the decoder is capturing the underlying structure of the data effectively.
    # in the same fashion, the lower values imply more information is lost during
    # the compression and reconstruction process.
    # 
    #!edit sidenote3:(too excessive ?)
    # technically speaking, logpϕ(x∣z) measures how probable the original data x is under 
    # the distribution parameterized by the decoder, given z.
    # Higher log-likelihood means the decoder effectively captures the structure of 
    # x from z while lower values indicates greater reconstruction loss.

    # sidenote4:
    # note that in this approach we assume the features in the latent space are independent, 
    # that is, each dimension of z(each feature) contributes independently to the decoded 
    # output. this way, we are effectively reducing our model complexity (i.e. the complexity
    # of the relationships between features) and no more need to model complex
    # relationships between each feature. this simplifies the whole process of sampling 
    # and reconstruction as we will see in a moment)
    # 
    # it should be obvious/its a given that his assumption may not fully capture the true structure 
    # of the data but its a practical trade-off we are willing to pay in order to have 
    # much better computational efficiency in training the decoder.
    # without this we have to face a huge computation burden and a complex sampling process. 
    # 
    # why do we need this simplification? 
    # if we do not use this simplification we have to undertake heavy calculations and overhead!
    # technically speaking, what we are doing here, is assuming a diagonal covariance matrix in
    # a multivariate Gaussian distribution(that is features are independent of each other hence
    # diagonal values and everything else is 0)
    # 
    # This assumption impacts both the computational costs involved and how sampling is done in 
    # 2 major ways:
    # First of all a full covariance matrix in an n-dimensional multivariate Gaussian distribution 
    # has n^2 elements, because it includes both variances (n diagonal elements) and covariances 
    # (n(n-1)/2 off-diagonal elements). 
    # without the simplification, the encoder would need to estimate all n^2 parameters
    # of the covariance matrix, which includes variances and covariances.
    # however, with the simplification, only n parameters (the variances) need to be learned 
    # because covariances are assumbed to be zeros.
    # this dramatically reduces the number of parameters the model has to estimate, especially 
    # for high dimensional data (take our simple MNIST example e.g. with latent dimensions n=50 vs n^2=2500 parameters and imagine
    # what would be the cost for larger/more complex datasets).
    # 
    # moreover, the multivariate Gaussian's log-likelihood involves the inverse of the covariance matrix(i.e.the term 1/Sigma).
    # computing the inverse of a `nxn` covariance matrix has a computational complexity 
    # of O(n^3) while for our simplifed case (a diagonal covariance matrix), it is trivial,
    # its just O(n) (we just need to take the reciprocal of each diagonal element).
    # 
    # second of all, and more importantly, sampling from a general multivariate Gaussian requires decomposing 
    # the covariance matrix to generate correlated samples. this decomposition operation is also O(n^3).
    # for our simplified case however, no decomposition is needed, as the features(dimensions) are independent. 
    # sampling is also as simple as generating univariate Gaussian samples for each dimension,
    # which is O(n).
    # 
    # sidenote:
    # to be more specific, to generate a sample from a general multivariate Gaussian, 
    # the covariance matrix Sigma is used to create correlations between dimensions.
    # that is after generating uncorrelated Gaussian samples, they are transformed 
    # into correlated samples using the decomposition result.
    # 
    # while for our simple case, since each dimension of z is modeled as an independent
    # Gaussian distribution with its own mean mu_i and variance sigma_i^2 sampling is as simple as:
    # generating a sample from N(mu_i, sigma_i^2) independently for each dimension i. 
    # since theres no correlation between dimensions, no additional transformations 
    # are needed.
    # 
    # Why is this useful in Variational Autoencoders (VAEs)?
    # so to cut a long story short, it boils down to 
    # learning fewer parameters and easier sampling .
    # its fewer parameters because (only mu and diagonal Sigma is used)(less work for forward/backward passes)
    # and it avoids overfitting by simplifying the model, especially when working with limited data.
    # and easier sampling because, the encoder predicts mu(mean vector) and sigma(standard deviation vector)
    # from it, derived from diagonal variances) and sampling from N(mu, sigma) is done directly.
    # 
    # 
    # sidenote:
    # reminder univariate vs multivariate gaussian distribution 
    # "multivariate" in multivariate gaussian distribution means the distribution
    # involves more than one variable (or feature)
    # it generalizes the concept of a univariate Gaussian distribution (which is a
    # normal distribution with a single variable) to cases where there are 
    # multiple variables that may or may not be correlated.
    # in mathematical terms, a multivariate Gaussian distribution for n-dimensional
    # data is defined by a mean vector and a covariance matrix(sigma).
    # the mean vector (mu) is an n-dimensional vector, where each element represents
    # the mean of one variable.
    # the covariance matrix (Sigma) is an nxn matrix, where the diagonal elements (Sigma_{ii})
    # represent the variance of each variable and the off-diagonal elements (Sigma_{ij}) 
    # represent the covariance between pairs of variables.
    # 
    # a uivariate Gaussian distribution on the other hand, is simply a normal distribution 
    # with a single variable.
    # its defined by a single mu(mean) and a single sigma^2 (variance).(σ^2)
    # while a multivariate Gaussian distribution is a distribution with two or more variables.
    # and its defined by a mean 'vector' mu and a covariance 'matrix' Sigma.
    # 
    # for a multivariate Gaussian, the number of dimensions corresponds to the 
    # number of variables/features. (that is for example, a 2D Gaussian involves
    # two variables and has a 2x2 covariance matrix, a 3D Gaussian involves three 
    # variables and has a 3x3 covariance matrix and so on)
    # 
    # moreover, The covariance matrix Sigma determines how the variables are correlated
    # If Sigma is diagonal, the variables are uncorrelated (no covariance).
    # If Sigma has non-zero off-diagonal elements, the variables are correlated.
    # 
    # The shape of the probability density function depends on the covariance matrix,
    # in 2D, if the variables are uncorrelated (diagonal covariance matrix),
    # the contours of the distribution are circular or elliptical.
    # If the variables are correlated, the contours are tilted ellipses.
    #
    # Univariate Gaussian:A distribution of a single variable x.
    # p(x) = (1/sqrt(2*pi*sigma^2))*exp(-((x-mu)^2)/(2*sigma^2))
    # 
    # Multivariate Gaussian (2D case):
    # p(x) = (1/((2*pi)^(n/2)*|Sigma|^(1/2)))*exp(-0.5*(x-mu)^T * Sigma^(-1) * (x-mu))
    # Here, x is a vector (e.g., [x1, x2]), mu is the mean vector (e.g., [mu1, mu2]),
    # and Sigma is the covariance matrix.
    
    # recap:
    # the term "multivariate" in "multivariate Gaussian distribution" means that the
    # distribution models more than one variable.
    # it describes both the individual behavior of each variable (via the mean vector mu)
    # and their relationships (via the covariance matrix Sigma).

    # In simpler terms:
    # in a typical multivariate Gaussian distribution, we would need to define both variances
    # and covariances (how different features are related to each other).
    # However, by assuming the features are independent (diagonal covariance matrix), we only
    # need to define the variances of each feature, which simplifies the model significantly.
    # The decoder samples from this simplified Gaussian distribution and uses the latent vector
    # to generate a reconstruction of the input data.
    # Why is this assumption useful?
    # Using a diagonal covariance matrix (i.e., independent features)
    # reduces the complexity of the model. We don't need to estimate the covariances between 
    # features, which would require more parameters and computation.
    # also assuming independence between features makes the latent space easier to interpret, 
    # as each dimension of the latent vector corresponds to an independent
    # variable.
    # This assumption is common in practice, (especially in VAEs) because
    # it allows for easier training and implementation while still capturing useful underlying 
    # structure in the data.
    
    # As we briefly pointed out before, this sampling process wont work as is, and it 
    # requires a clever trick to work as expected.
    # When training the model, we need to be able to calculate the relationship 
    # of each parameter in the network with respect to the final output loss using backpropagation. 
    # 
    # However, we simply can not do this for a "random sampling process". (not that we cant, we absolutly can,
    # but it doesnt calculate what we want! which is computing an estimate of the derivative!)
    # this is where we have to use the previously mentioned trick, commonly known as reparametrization trick
    # its selfexplanetory once you get the idea behind it:d,
    # basically it says that we randomly sample ε from a unit Gaussian, and then shift the 
    # randomly sampled ε by the latent distribution's mean μ and scale it by the 
    # latent distribution's variance σ, which is effectively the same as using our initial mean and variance,
    # with the exception now, that, the epsilon itself is treated as a mere input (identity) and wont need backprogapation
    # (the same way you dont backpropagate to the input images) and the mu,variances will be treated as paramaters
    # and will be correctly incorporated into the computational graph and backpropagated properly. hence the name re-parameter-ziation. got it?)
    #
    # With this reparameterization, we can now optimize the parameters of the distribution
    # while still maintaining the ability to randomly sample from that distribution.
    # as to why this doesnt give us the proper estimate, think of it as the difference betweeen
    # stochastic gradient descent vs gradient decent, the stochastic part referts to the mini batches,
    # instead of the full training set as one batch, which each of them(mini batches) give an estimate
    # of the actual gradients, so if the estimate of these mini batches are not close, as we continue,
    # we get further away from the actual direction of the changes and fail to converge.
    # kingma argues in paper that, this is why this change, makes the model to have the right estimate
    # for mu/variances (more on this later)

##############################

# !edit I should split them in separate parts because its got too large!
# 


# 
#! edit add note to use BCE for reconstruction loss instead of MSE as it has better performance
# especially for larger datasets
#

# read more  : https://towardsdatascience.com/intuitively-understanding-variational-autoencoders-1bfe67eb5daf
# There are other resouces for this as well. its highly recommened to read them: 
# https://www.jeremyjordan.me/variational-autoencoders/
# https://jaan.io/what-is-variational-autoencoder-vae-tutorial/
# https://www.youtube.com/watch?v=uaaqyVS9-rM 
# http://blog.shakirm.com/2015/10/machine-learning-trick-of-the-day-4-reparameterisation-tricks/
# https://www.reddit.com/r/MLQuestions/comments/dl7mya/a_few_more_questions_about_vaes/

# we'll also have an example concerning words(in NLP domain) and see how we can 
# leverage VAEs in that domain as well. for now lets see how we can implement this
# for vision domain. i.e. on mnist dataset

# note: 
# in variational autoencoders, the encoder part is sometimes referred to as
# the recognition part while the decoder part is referred to as the generative part/model.

# if you havent read the links I gave you, go read them all. each single one of them
# will help you grasp one aspect very good!

# Viewppoint 2! what is VAE and how does it work? why was it created? whats the intuition behind it?
# explain 
# 

# sidenote: a much clearer implementation which I wrote for pytorch examples repo at the time: 
# https://github.com/Coderx7/examples/blob/vae-example-branch/vae/main.py
# maybe not!
#
# 
# read this https://deepai.org/machine-learning-glossary-and-terms/manifold-hypothesis 
# before revising the whole thing. gives a very good picture of the whole thing imho
# 


# note : why do we really want the epsilon in reparameterization trick? 
# what is the intuition behind it : https://youtu.be/9zKuYvjFFS8?t=415
# read in depth technical reasons here :
# all answers contain great explanations 
# https://stats.stackexchange.com/questions/199605/how-does-the-reparameterization-trick-for-vaes-work-and-why-is-it-important
# https://stats.stackexchange.com/questions/429315/why-is-reparameterization-trick-necessary-for-variational-autoencoders
# https://stats.stackexchange.com/questions/342762/how-do-variational-auto-encoders-backprop-past-the-sampling-step/342815#342815
# https://blog.neurallearningdymaics.com/2019/06/variational-autoencoders-1-motivation.html
# http://ruishu.io/2018/03/14/vae/
# 
# reading this links up until now, you show have been convinced 
# that we use reparameterization trick solely 
# because otherwise we couldnt backprop to random node! 
# this however is not the whole story!
# from Kingma: 
# This reparameterization is useful for our case since it can be used to rewrite an 
# expectation w.r.t qϕ(z∣x) such that the Monte Carlo estimate of the expectation is 
# differentiable w.r.t. ϕ. 
# The issue is not that we cannot backprop through a “random node” in any technical sense. 
# Rather, backproping would not compute an estimate of the derivative. 
# Without the reparameterization trick, we have no guarantee that sampling large numbers of z
# will help converge to the right estimate of ∇θ.(i.e. its there to avoid a very bad (high variance) estimate.))
# 
# read in more detail here: 
# http://gregorygundersen.com/blog/2018/04/29/reparameterization/
# if you want to know about expectation and what it is, this may help 
# https://revisionmaths.com/advanced-level-maths-revision/statistics/expectation-and-variance)


# now lets implement our VAE 

# first lets define conv and deconv blocks,
# we use two simple functions to this!

def conv(in_dim, out_dim, kernel_size=3, stride=1, padding=1, batch_norm=True, bias=False, act=nn.ReLU()):
    return nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias=bias),
                            nn.BatchNorm2d(out_dim) if batch_norm else nn.Identity(),
                            act)

def deconv(in_dim, out_dim, kernel_size=3, stride=2, padding=1, act = nn.ReLU(), batch_norm=True, bias=False):
    return nn.Sequential(nn.ConvTranspose2d(in_dim, out_dim, kernel_size, stride, padding, bias=bias),
                            nn.BatchNorm2d(out_dim) if batch_norm else nn.Identity(),
                            # important note for the last layer there should be no relu
                            # even if you put a sigmoid after the relu, it wont work!
                            act)

# a simplistic res module
class conv(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, stride=1, padding=1, batch_norm=True, bias=False,act=nn.LeakyReLU(0.2)):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(out_dim) if batch_norm else nn.Identity(),
            act
        )
        # residual connection needs input and output dimensions to match
        self.residual_connection = (in_dim == out_dim and stride == 1)

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual_connection:
            out += x
        return out

class deconv(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, stride=2, padding=1, act=nn.LeakyReLU(0.2), batch_norm=True, bias=True):
        super().__init__()
        self.deconv_block = nn.Sequential(
            nn.ConvTranspose2d(in_dim, out_dim, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(out_dim) if batch_norm else nn.Identity(),
            # nn.GroupNorm(1,out_dim) if batch_norm else nn.Identity(),
            act
        )
        # residual connection needs input and output dimensions to match
        self.residual_connection = (in_dim == out_dim and stride == 1)

    def forward(self, x):
        out = self.deconv_block(x)
        if self.residual_connection:
            out += x  
        return out

# instead of deconv, for getting better result, it doesnt work for me! i keep getting cuda error
# I guess its because of my choice of kernels! i need to get this to work!
# class PixelShuffleBlock(nn.Module):
#     def __init__(self, in_dim, out_dim, upscale_factor=2, act=nn.LeakyReLU(0.2), batch_norm=True):
#         super().__init__()
#         self.block = nn.Sequential(
#             # for pixelshuffle to work, we multiply the outdim by upscalefactor
#             # and then feed the result to pixelshuffle with the upscalefactor
#             # it will rearange the channels,upsample the image with the original outdim
#             # so the networks outputdim stays the same
#             nn.Conv2d(in_dim, out_dim * (upscale_factor ** 2), kernel_size=3, padding=1),
#             # use PixelShuffle to rearrange channels into spatial upsampling.
#             nn.PixelShuffle(upscale_factor),
#             nn.BatchNorm2d(out_dim) if batch_norm else nn.Identity(),
#             act
#         )
#         self.residual_connection = (in_dim == out_dim)

#     def forward(self, x):
#         out = self.block(x)
#         if self.residual_connection:
#             out += x
#         return out

#since I might disable batchnorm for decoder, I enable bias by default
# otherwise id leave it at false!
class upconv(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, scale_factor=2, padding=1, 
                 act=nn.LeakyReLU(0.2), batch_norm=True, bias=True):
        super().__init__()
        self.block = nn.Sequential(nn.Upsample(scale_factor=scale_factor, mode='nearest'),
                                   nn.Conv2d(in_dim, out_dim, kernel_size, stride=1, padding=padding, bias=bias),
                                   nn.BatchNorm2d(out_dim) if batch_norm else nn.Identity(),
                                   act)
        # residual connection only makes sense when no spatial change is applied
        self.residual_connection = (in_dim == out_dim and scale_factor == 1)

    def forward(self, x):
        out = self.block(x)
        if self.residual_connection:
            out += x
        return out

# the overall structure of the VAE is roughly the same it consits of an encoder section 
# and a decoder section. lets implement them, well explain each part when implementing them
class VAE(nn.Module):

    def __init__(self, embedding_size=100, input_channels=1):
        super().__init__()
        self.embedding_size = embedding_size
        # number of input channels
        self.input_channels = input_channels

        self.encoder = nn.Sequential(conv(self.input_channels, 32),#28x28
                                     conv(32,64,stride=2),#14x14
                                     conv(64,96,stride=2),#7x7
                                     conv(96,128,stride=2),#3x3
                                     conv(128,256,stride=2),#2x2
                                     # note: its best not to shrink too much and at least 
                                     # retain some spatial dimensions (like 2x2,4x4 (in some cases based on the network even 7x7 is good))
                                     conv(256,self.embedding_size,stride=2,padding=1),#1x1
                                     # nn.Linear(28*28, self.embedding_size)
                                    )
        
        # 1x1 is the spatial dims of the output of the last encoder layer
        # we can use an extra fc layer to get the flattened encoderoutput
        # and use the 1d output of this extra fc layer and decouple them!
        # but I simply didnt do that here!(i simply forgot when I was trying to get this to work initially!)
        bottleneck_size = self.embedding_size*1*1 
        # mean
        self.fc1_mu = nn.Linear(bottleneck_size, self.embedding_size) 
        # we use log since we want to prevent getting negative variance
        #logvariance
        self.fc1_logvar = nn.Linear(bottleneck_size, self.embedding_size) 

        #! calculate the logptheta(x|z) as well?
        #!
        # mnist?!
        # we can use dropout/bn to have better training!
        # sidenote: 
        # if we start our decoder with a linear layer,
        # we need to note 2 things:
        #1. preferably do not shrink too much in decoder, 
        # like at least retain some spatial dimension (like 4x4)
        # if we didnt, then in the decoder start with a larger spatial dim
        # we add the desired spatial dim in form of multiplication 
        # like (nn.Linear(self.embedding_size, 128*4*4)) 4*4 being the spatial dims
        # (and is a good choice usually dont go smaller unless you know what youre doing)
        # next we need to reshape the output properly so the next deconv layers get the
        # proper input.
        # we can do this in several ways, but one way would be to do this in forwardpass in
        # feed the flattened z from encoder to the first layer of decoder (Which is our linear layer)
        # and then reshape it so the z has the proper 4d shape
        # z = self.decoder[0](z).reshape(input.size(0),-1,1,1)
        # then feed the new z to the rest of the layers
        # reconstructed_img = self.decoder[1:](z)
        # which is not ideal so we use a simple deconv layer!
        # so instead we simply use nn.UnFlatten() which makes our lives much easier!
        # we needed to reshape to deconv layer had proper 4d input tensor
        # sidenote 2: 
        # during upsampling stage, we can use different kernel sizes ranging from 
        # 2 and up. larger kernels can capture more spatial information 
        # but may introduce artifacts, 
        # upsampling from very small spatial dimensions (like 1x1) can also result 
        # in visible artifacts in the reconstructed image. and might lead to 
        # blurry outputs unless carefully tuned so keep this in mind!
        # sidenote3 :
        # to calculate the deconvs output at each stage we use this formula
        # h is the input dimension (in our case is 2 (our input is 2x2))
        # ((h-1)*stride)+(kernel_size-2)*padding
        # (h=1,k=4,s=2,p=1) 1-1*2+4-2*1)=2/
        # (h=2,k=4,s=2,p=1) 2-1*2+4-2*1)=4/
        # (h=4,k=4,s=2,p=1) 4-1*2+4-2*1=8/ 
        # (h=8,k=2,s=2,p=1) 8-1*2+2-2*1=14/ 
        # (h=14,k=4,s=2,p=1)14-1*2+4-2*1=28/ 
        self.decoder = nn.Sequential(nn.Linear(self.embedding_size, 256*1*1),
                                     nn.ReLU(),
                                    #  nn.Dropout(0.1),
                                     nn.Unflatten(1,(256,1,1)),
                                     deconv(256,256,kernel_size=4),#2
                                    #  nn.Dropout(0.05),
                                     deconv(256,128,kernel_size=4),#4
                                    #  nn.Dropout(0.01),
                                     deconv(128,64,kernel_size=4),#8
                                     deconv(64,32,kernel_size=2),#14
                                     # remember we dont use batchnorm at the last later
                                     # beacuse it will destroy the image by trying to normalize it!
                                     # more importantly dont use relu! even though you use a sigmoid at the end
                                     # it will prevent the loss to go down. 
                                     # this simple mistake took a lot of my time! because
                                     # I simply forgot to check deconv!
                                     deconv(32,self.input_channels,kernel_size=4,batch_norm=False,act=nn.Sigmoid()),#28
                                     # in normal situations we wouldnt use sigmoid
                                     # but since we want our values to be in [0,1]
                                     # we use sigmoid. for loss we will then have  
                                     # to use, plain BCE (and specifically not BCEWithLogits)
                                    #  nn.Sigmoid()
                                    )
    
    # Note: 
    # In order to deal with the fact that the network will also learn negative values
    # for σ, we'll have the network learn log(σ) and exponentiate(exp) it 
    # to get the latent distribution's variance.
    def reparamtrization_trick(self, mu, logvar):
        # !edit combine them in one paragraph, we have too many sidenotes that we can 
        # !incorporate into the actual text I guess! 
        # torch.exp() converts logvar(log(variance) which our network produces) back to
        # variance (sigma^2) but note that here, we have the multiplication by 0.5 and
        # then exponentiation.
        # this is equivalent to computing the square root of the variance (its 
        # asif we wrote exp(0.5*log(σ^2))) which gives us back the standard deviation
        # remember log(a^b) = b.log(a) so log(√𝜎^2)=log((𝜎^2)^0.5)=0.5.log⁡(𝜎^2)
        # since we have logvar and not var, we simply exponantiate it with 0.5 multiplied
        # so it becomes variance.
        # 
        # sidenote:
        # variance(σ^2) must always be positive because it represents squared differences.
        # we dont directly optimize σ^2 or σ instead we work with log(σ^2) (logvar), 
        # which ensures that the computed variance (σ^2=exp(logvar)) is always positive,
        # even if logvar takes negative values. (exp() returns positive)
        # 
        # The factor 0.5 in exp(0.5*logvar) comes from the mathematical process of 
        # computing the standard deviation (σ) from log(σ^2):
        # σ = sqrt(σ^2) = exp(0.5 * logvar)
        # 
        # sidenote2:
        # why do we use logvariance instead of variance?
        # because variance (also standard deviation) can take very small or large values, 
        # that can lead to overflow or underflow in floating-point computations, therefore
        # representing it as log⁡(σ^2) avoids that issue.
        # 
        # sidenote2:
        # The standard deviation represents the 'scale' of the distribution 
        # in the same units as the data.(z = μ+σ⋅ϵ, ϵ∼N(0,I))
        # note here for sampling we use the standard deviation (σ) not variance!
        # because multiplying by variance (σ^2) wouldn't make sense dimensionally
        # and it would lead to an incorrect scaling.
        # we use variance (σ^2) in the KL divergence term during optimization though
        # (when representing the overall spread of a distribution).
        #
        std = torch.exp(0.5*logvar)
        # epsilon sampled from normal distribution with N(0,1)
        # we use epsilon so we put the stochasity/randomness in the epslilon itself
        # so we dont need to calculte gradient for it, we treat it as an input and
        # this way only optimize mu/std parameters
        eps = torch.randn_like(std)
        # How to sample from a normal distribution with known mean and variance?
        # https://stats.stackexchange.com/questions/16334/ 
        # (tldr: just add the mu , multiply by the var) . 
        
        # why we use an epsilon?
        # you should know by now, if not read the former links I provided.
        # basically there are 2 main explanations, the first one (Which is not accurate) is
        # because without it, backprop wouldnt work atall, its impossible to backprop!(which 
        # is not really the accurate, its possible and it works, but with a caveat!).
        # for the random part we sample from normal distribution N(0,1)
        # and treat this as a mere input. (like the images that are input and we dont 
        # calculate the gradients for) 
        # we then shift this new sample with the mean and std we have and effectively
        # reach the very same result. that is we add our mu and scale it by std 
        # (since our eps has 0 mean and std 1, adding it with mu, and scaling it by std
        # will make it N(mum std) which is what we want. our expression also now can be
        # easily backpropagated. 
        # also you need to know that, it is also said this reparameterization trick 
        # is only done for numerical stability and actually the basic way can be done as well!
        # and finally, the actual reason was given above, we actually do this to guarantee 
        # the right estimate of ∇θ. without this, we have no guarantee that sampling large 
        # numbers of z, will help convertence to the right estimates of ∇θ.(think about 
        # stochastic gradident decent vs gradient decent and how the former gives us an estimate
        # for the latter (a good estimate) and if it fails to do so,it would no more represent
        # the gradient decent/the actual gradient.)
        return mu + eps*std
    # 
    def encode(self, input):
        output = self.encoder(input).view(input.size(0),-1)
        # note we dont use activations for mu/std
        mu = self.fc1_mu(output)
        log_var = self.fc1_logvar(output)
        z = self.reparamtrization_trick(mu, log_var)
        return z, mu, log_var

    def forward(self, input):
        z, mu, logvar = self.encode(input)
        reconstructed_img = self.decoder(z)
        # print(f'{reconstructed_img.shape=}')
        return reconstructed_img, mu, logvar

# test the vae and the output shape, making sure 
# we didnt mess sth up in encoder/decoder
input_channels=1
model = VAE(embedding_size=100, input_channels=input_channels)
img_re, _,_ = model(torch.randn(size=(5,input_channels,28,28)))
print(f'{img_re.shape=}')

# Note :
# for proper training, dont incorporate kl term at the begining. 
# first train with the reconstruction loss, and then gradually introduce the kl term
# this should allow the model to arrive at a decent spot! otherwise it wont work properly
# (except maybe somehow account for the scale of the kl term, which if you do add a scaler 
# term, would essentially become a disentangled vae which is an improvement over the
# this (vanilla) version, but would still face some issues such as posterior collapse.
# we'll explain this in amoment, but before that, lets keep this simple for now.

# Also read : https://github.com/jxhe/vae-lagging-encoder
# The code seperates optimization of encoder and decoder in VAE, and performs 
# more steps of encoder update in each iteration. This new training procedure 
# mitigates the issue of posterior collapse in VAE and leads to a better VAE 
# model, without changing model components and training objective.

# for calculating loss, we can have several options 
# 1. use mse for reconstruction loss 
# 2. use BCE for reconstruction loss   
# when using bce we have two options, we can use reduce='sum'
# or we can use reduce='mean'. 
# if we want to use BCE with reduce='sum' we only calculate the kl
# with sum. but when we want to use BCE with reduce='mean' or mse
# we use sum(,-1) and then use loss_recons+torch.mean(kl)
# we also need to normalize our reconstruction loss by the input dim
# ension size. 
#!edit lets not use beta here, and show how hard it can get, so after it we
#! introduce beta and other techniques to fight the issues?
def loss_function(outputs, inputs, mu, logvar, reduction ='mean', use_mse = False, normalize=True):
    outputs = outputs.view(*inputs.shape)
    #! beta belongs to entangled vae, the normal vae doesnt have beta scaler
    if reduction == 'sum':
        criterion = nn.BCELoss(reduction='sum')
        reconstruction_loss = criterion(outputs, inputs)
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        # since we are using sum as reduction for our reconstruction loss (all samples loss sum)
        # our kl loss needs to be summed over all dimensions and all batch 
        # which gievs us a single scalar value.
        # the bad thing is, since its summed over batch, the batchsize affects the training
        # we need to use different lr for different batchsizes
        # because the gradients also scale with the batchsize, 
        # therefore learning rate needs to be ajusted accordingly)
        # also this means more instability as its harder to balance the two terms like this
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return reconstruction_loss + kl_loss
    else:
        if use_mse:
            criterion = nn.MSELoss()
        else:
            criterion = nn.BCELoss(reduction='mean')
        reconstruction_loss = criterion(outputs, inputs)
        # here for the kl loss we only sum over the latent dimensions,
        # this gives us a single loss for each sample, 
        # we need to take the mean of the whole batch and this makes it independent of 
        # the batchsize and should give us a more stable loss, this is more aligned with
        # our reconstruction loss which we do the same thing (take the mean of the whole batch (i.e. reduction=mean))
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), -1)
        # we also need to normalize the reconstruction/kl loss otherwise kl will overpower it!
        # and we would get nonsens as output, (since the image is averaged pixelwise, but
        # kl is summed for each sample its not balanced properly)
        # we need to either divide kl loss by the image dimensions, 
        # or multiply reconstruction loss by the image dimensions to scale it up a bit
        # todo use input dims instead of hardcoded dims
        # scaler = 28*28 if normalize else 1
        _,h,w,c = inputs.shape
        scaler = h*w*c if normalize else 1
        # note since we sumed over the latent dimension, we will have batchsize of losses
        # which we need to average to get a single loss value
        # this is a bit more stable than our previous version, but it will still be hard
        # to train properly, we will see in a moment how that goes. to fix it, a quick way
        # would be to use the beta variant, which we will cover shortly.
        return scaler*reconstruction_loss + kl_loss.mean()
#%%
#
# I set this option to see the full stack-trace when a weird error occurs
# its good practice to get accustomed to the debugging/profiing facilities provided
# by pytorch, I might dedicate a separate section for this later on
# torch.autograd.set_detect_anomaly(True)
# torch.set_printoptions(profile='full')
#
# now lets train :
# mnist dataset is a very simple dataset, and using embsize=2 we get good results right of the bat
# but this is not the case all the time, if we use more complex datasets, we quickly see 
# no matter how much we try we dont get good results with this implementation, its expected, 
# (as the original vae came in 2013(https://arxiv.org/pdf/1312.6114) and was only used on two
# datasets mnist and frey face dataset both of which are grayscale and very simple datasets.)
# but for now, lets not get ahead of ourselves, and stick to mnist for now, just try 
# different embeddingsize and hyperparamters to see how far you can get. even with mnist
# we will be facing issues here, we'll be discussing the issues we face here shortly and fix them all
# I also added cifar10 example (we made our model so it can handle both 1 and 3 input channels
# I just resized cifar10 so the changes is minimal here ))
epochs = 50

dataset_train = datasets.MNIST('MNIST', train=True, download=True,transform=transforms.ToTensor())
dataset_test = datasets.MNIST('MNIST', train=False, download=True,transform=transforms.ToTensor())

## uncomment these lines to test with cifar10 
## (only do this after you ave experimented with mnist)
# transformations = transforms.Compose([transforms.Resize(28), transforms.ToTensor()])
# dataset_train = datasets.CIFAR10('CIFAR10', train=True, download=True,transform=transformations)
# dataset_test = datasets.CIFAR10('CIFAR10', train=False, download=True,transform=transformations)

#TODO
# display the manifold for encoder encodings during training and make a gif out of it?
dataloader_train = torch.utils.data.DataLoader(dataset_train,batch_size=128,shuffle=True)
dataloader_test = torch.utils.data.DataLoader(dataset_test,batch_size=128,shuffle=False)

# imgs, lbls = next(iter(dataloader_train))
# print(f'{imgs.shape=}')
# make sure the images are in range(0,1)
# print(f'range = ({imgs.min()},{imgs.max()})')

# if its mnist use 1 if its cifar10 use 3 for input channel
input_channel = 1 if isinstance(dataset_train,datasets.MNIST) else 3
embeddingsize = 2#2,10
reduction='mean'#mean
# to see how it affects our result, when using using reduction='mean'
# set normalization to False, without normalization we wont learn 
# anything meaningful! (reduction='sum' doesnt use normalization)
normalize = True #False
# whether to use mse instead of bce in our loss
use_mse = False
interval = 2000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = VAE(embeddingsize, input_channel).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr =0.01,weight_decay=1e-4)#1e-4
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [5,10,25,45,50])

for e in range(epochs):
    for i, (imgs, labels) in enumerate(dataloader_train):
        imgs = imgs.to(device)
        preds,mu, logvar = model(imgs)
        loss = loss_function(preds, imgs, mu, logvar, reduction=reduction, use_mse=use_mse, normalize=normalize)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i% interval ==0:
            print(f'epoch {e}/{epochs} [{i*len(imgs)}/{len(dataloader_train.dataset)} ({100.*i/len(dataloader_train):.2f}%)]'
                  f'\tloss: {loss.item():.4f}'
                  f'\tlr: {scheduler.get_lr()[-1]}')
    scheduler.step()

#%% 
# save the model
torch.save({"states":model.state_dict(),
            "epochs":epochs,
            "embedding_size":model.embedding_size,
            "reduction":reduction,
            "normalize":normalize,
            "use_mse":reduction,
            "optimizer":optimizer.state_dict(),
            "scheduler":scheduler.state_dict()},
            f"vae_{model.embedding_size}_{reduction}_{'normalized' if normalize else 'not-normalized'}_{'mse' if use_mse else 'bce'}.pth")
print('model saved!')
#%%
# load the model 
states = torch.load(f"vae_{model.embedding_size}_{reduction}_{'normalized' if normalize else 'not-normalized'}_{'mse' if use_mse else 'bce'}.pth")
model.load_state_dict(state_dict=states['states'])
print('weights loaded')
#%%
#
#  
# now lets write some functions for visualization 
# and see how our model does
# 
# generate random images by randomly sampling from a simple normal distribution!
@torch.no_grad()
def generate_random_images(model:VAE, count:int=32, rows:int=8, img_shape=(1,28,28)):
    c = img_shape[0]
    # simply randomly sampling from a normal distribution will
    # give us random classes.
    sample = torch.randn(size=(count, model.embedding_size)).to(device)
    # we can further influence our generation by imposing different means/stds
    # sample *= 0.5 + 0.5
    model.eval()
    imgs = model.decoder(sample)
    # print(f'{imgs.shape=}')
    imgs = imgs.view(-1, *img_shape)
    img = make_grid(imgs,nrow=rows,normalize=True).cpu().detach().numpy().transpose(1,2,0)
    # only use cmap=Greys_r for grayscale images
    plt.imshow(img, cmap='Greys_r' if c ==1 else None)
    plt.title('randomly sampled generation')

# generate_random_images(model, count=32)

# lets now display the original images next to their reconstruction
# to see the quality of reconstruction
def display_imgs_recons(img_pairs, title='testset reconstruction', save_result= True, save_dir='results',nrows=8, rows=20, cols=1):
    img_cnt = len(img_pairs)
    rows = img_cnt//cols +1 if img_cnt>rows*cols else rows
    # print(f'{rows=} {cols=}')
    # print(f'{img_cnt=}')
    # print(f'{rows=} {cols=}')
    fig = plt.figure(figsize=(64, 64))
    
    if save_result:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    for i in range(img_cnt):
        grid_imgs = make_grid(torch.from_numpy(img_pairs[i]),
                            nrow=nrows,
                            normalize=True)
        ax = fig.add_subplot(rows, cols, i+1, xticks=[],yticks=[])
        ax.imshow(grid_imgs.numpy().transpose(1,2,0))
        ax.set_title(f'{title}-{i}')

        if save_result:
            save_image(grid_imgs, f'{save_dir}/imgs_{i}.jpg')
    plt.show()

@torch.no_grad()
def evaluate_on_testset(model, dataloader_test, sample_count=20, img_shape=(1,28,28)):
    test_set_size = len(dataloader_test.dataset)
    img_pairs = []
    losses = []
    interval = 10
    model.eval()

    for i, (imgs, labels) in enumerate(dataloader_test):
        imgs = imgs.to(device)
        preds, mu, logvar = model(imgs)
        loss = loss_function(preds, imgs, mu, logvar, reduction=reduction, use_mse=False)
        losses.append({'val_loss':loss.item()})
        
        print(f'[{i*len(imgs)} / {test_set_size} ({100.*i/len(dataloader_test):.2f}%)]'
            f'\tLoss: {(loss).item():.4f}')

        if i%interval==0:
            reconstructeds = preds.cpu().view(-1, *img_shape)
            # grab the first few images and their reconstructions
            # sidenote: when we use no_grad, theres no gradients, so no need for .detach()!
            imgs = imgs[:sample_count].cpu().numpy()
            recons = reconstructeds[:sample_count].numpy()
            pairs = np.array([np.dstack((img1,img2)) for img1, img2 in zip(imgs,recons)])
            img_pairs.append(pairs)

    # plot the losses using pandas! 
    # this actually is very neat and comes handy very often!
    # we can have a list of dictionaries, where each value is 
    # attributed by a key. this way, our keys will be used as
    # legends and we have a simple plot with minimum hassle
    import pandas as pd
    ax= pd.DataFrame(losses).plot()
    ax.set_title('testset loss')
    plt.show()
    
    display_imgs_recons(img_pairs, nrows=10, rows=8, cols = 1)

# evaluate_on_testset(model, dataloader_test)

#! edit choose better function names! 
# lets plot the latent space encodings and see
# how the encoded representations of our data
# look in the latent space
@torch.no_grad()
def plot_latent_space_encodings(model, batch_size = 10000):

    dataloader_test2 = torch.utils.data.DataLoader(dataset_test,
                                                batch_size = batch_size,
                                                num_workers = num_workers,
                                                pin_memory=True)
    imgs, labels = next(iter(dataloader_test2))
    imgs = imgs.to(device)
    z_test,*_ = model.encode(imgs)
    # since we are using torch.nograd, 
    # theres no gradients so we dont need to use .detach()
    # otherwise we had to use it here
    z_test = z_test.cpu().numpy()

    plt.figure(figsize=(12,10))
    print(z_test.shape)
    plt.scatter(x=z_test[:,0],
                y=z_test[:,1],
                c=labels.numpy(),
                alpha=.4,
                s=3**2,# point size, the biggger the value, the larger the points on the canvas
                cmap='viridis')
    plt.colorbar()
    plt.xlabel('Z[0]')
    plt.ylabel('Z[1]')
    plt.title('Latent space encodings of 2 dimensions')
    plt.show()

# lets now see how the latent space looks like with tsne
# the previous version plot_embedding_cluster would look at
# the encoders output, before they were used to create the
# latent vector z. hopefully this gives us a better picture
@torch.no_grad()
def plot_latentspace_clusters(model, dataloader_train, title='', use_pca=False):
    model.eval()
    # grab the device from model parameter
    device = next(model.parameters()).device
    # grab all the features, because tsne needs to be applied to 
    # the whole dataset all atonce not batch by batch
    all_features = []
    all_labels = []

    for imgs, lbls in dataloader_train:
        imgs = imgs.to(device)
        # Get feature vectors
        latent_feature_vectors,*_ = model.encode(imgs)
        all_features.append(latent_feature_vectors.cpu().view(imgs.size(0), -1).numpy())
        all_labels.append(lbls.numpy())

    # concatenate all batches
    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    if use_pca:
        reducer = PCA(n_components=2)
        # since pca is sensitive to the scale of features and 
        # if the features are not properly scaled (e.g. mean-centered and variance-normalized),
        # it can produce poor projections we scale the features here!
        scaler = StandardScaler()
        all_features = scaler.fit_transform(all_features)
    else:
        reducer = TSNE(n_components=2, random_state=66, perplexity=30)

    plt.figure(figsize=(10, 8))
    # print(f'{all_features[0].shape[-1]}')
    
    if all_features[0].shape[-1] >2 :
        # features2d are coordinates showing where each datapoint is
        features2d = reducer.fit_transform(all_features)
        # print(f'{features2d[:5]}')
    else:
        features2d = all_features
    # tab10, is a colormap inwhich it has 10 colors, therefore its a prefect choice for us    
    scatter = plt.scatter(features2d[:, 0], features2d[:, 1], c=all_labels, cmap='tab10', alpha=0.6)

    # add class labels to each cluster for better visualization
    # to do this we need t o calculate the centeroid(i.e. mean) of each cluster
    # which is basically taking the average of all the points for that cluster
    # and then use plt.text to add class numbers
    
    # note we dont need all the labels, just one for each cluster!
    for label in list(range(10)):
        # find the centroid of each cluster
        # note that the values in features2d are coordinates(when using tsne),
        # which are the 2D positions of the data points
        # since our data are stored sequentially we know each row(class label) 
        # in all_labels belong to a corresponding data point in features2d.
        # that is for example, if all_labels[0] = 0, it means the first data point
        # in features2d belongs to class 0.
        # we use this to grab all the points belonging to a specific label one at a time 
        centroid = np.mean(features2d[all_labels == label], axis=0)
        # annotate the centroid with the class label
        plt.text(centroid[0], centroid[1], str(label), fontsize=12, fontweight='bold',
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.3'))

    title = f"\n{title}" if title else ''
    plt.title(f"{'PCA' if use_pca else 'TSNE'} Projection to 2D{title}")
    plt.colorbar(scatter, label='Class Label')
    plt.show()

#! edit fix this with embd>2
@torch.no_grad()
def generate_latent_space_grid(model, n=20,lower_bound=-2, upper_bound=2, img_shape=(1,28,28)):
    # lets see if the transition in our latent space is smooth
    # that is we should be able to smoothly transition from one
    # class to the other, at least this is what we are tryting 
    # to see.
    # we create a vector of equally spaced values, and try to
    # visualize these vectors, (they act as our latent vector z)
    # since they are equally spaced, we can see how they change
    # gradually, ideally we want them to have a smooth transition
    # from one class to another. 
    # so lets see how our interpolation turns out
    # n means we want a figure with nxn digits
    model.eval()
    # we are basically creating a z vector, with n, equally spaced value
    # starting from lowerbound, up until upperbound (e.g from -2 to 2)
    # we create 2 such vectors, so we can create a grid of numbers
    # treating one z for xaxis and another for the yaxis. 
    z1 = torch.linspace(lower_bound, upper_bound, n)
    z2 = torch.linspace(lower_bound, upper_bound, n)
    # using np.meshgrid, we create our grid, meshgrid, simply 
    # expands z1 and z2 into 2D grids, by first repeats z1 values in
    # x-axis (rows) and then repeats the z2 values in y-axis(columns),
    # and finally using np.dstack, they are combined and the result 
    # will be a 3dgrid where each xy is made up of z1 and z2 values.
    # (test with a small example like linspace(-2,2,5), and see how it goes)
    z_grid = np.dstack(np.meshgrid(z1, z2))
    z_grid = torch.from_numpy(z_grid).to(device)
    z_grid = z_grid.reshape(-1, model.embedding_size)
    # print(f'{z_grid.shape=}')
    x_pred_grid = model.decoder(z_grid)
    x_pred_grid= x_pred_grid.cpu().view(-1, *img_shape)
    x = make_grid(x_pred_grid,nrow=n).numpy().transpose(1,2,0)
    plt.figure(figsize=(20, 20))
    plt.xlabel('Z_1')
    plt.ylabel('Z_2')
    plt.imshow(x)
    plt.title(f'latent space grid of numbers({n}x{n})')
    plt.show()
#%%
img_shape=(1,28,28)
generate_random_images(model, count=32, img_shape=img_shape)
evaluate_on_testset(model, dataloader_test, img_shape=img_shape)
plot_latent_space_encodings(model)
plot_embedding_clusters(model, dataloader_train, title='Encoder embedding',use_pca=False)
plot_latentspace_clusters(model, dataloader_train, title='Full latent clusters',use_pca=False)
generate_latent_space_grid(model,n=10,lower_bound=-2,upper_bound=2, img_shape=img_shape)
generate_latent_space_grid(model,n=20,lower_bound=-2,upper_bound=2, img_shape=img_shape)
#%%%
 
# now if you try to play with parameters, you'll see its really 
# hard to get it working! and our results can quickly get either blury
# or too similar/generic. so lets talk about these issues we are facing 
# and try to address them and hopefully get them fixed!
#
# During training we may face something called posterior collapse,
# it can happen for several reasons, but primarily, it happens
# when the decoder is more powerful than the encoder(or encoder is too weak!)
# and the latent space stops encoding meaningful information and the decoder ignores
# the latent features/variables during reconstruction.
# in its extreme form, the decoder wont even rely on the encoded representation
# and will only rely on the prior itself (i.e. the learned latent distribution
# collapses to the prior distribution (i.e. q(z|x) ≈ p(z) i.e. they almost are the same!))
# as a result, the latent features/variables will contain little to no useful information,
# leading to reconstructions that are too generic or blurry.
# 
# its worth noting that sometimes, you get clear and near prefect reconstructions, at train
# test time, but random generation is just nonsense. so it would be more accurate to say
# we will have blurry, nonsensical (blobs of color!/random noise) randomly sampled outputs
# because this is a sign of meaningless latent space! as we will see in our experiments shortly
# we can get great reconstrutions(like by using skipcon) but absolutely awful generations!(vaes by nature can not 
# produce crystal clear/sharp images when it comes to compelx datasets! so a blurry output
# is what we can hope for as best, (the amount of blurryness can be improved but dont expect much!)) 
# having said this, when the posterior collapse happens, both reconstructions and generations
# suffer greatly! theoutput is just bad! details minimal or nonexistant, images are eaither not
# formed, or are malformed(miss parts), etc. 
#
# (posterior collapse occurs when the encoder ignores the latent space,
# in a way that the learned latent distribution becomes close to the prior
# distribution (e.g. a standard normal distribution (N(0,I)), regardless 
# of the input images. 
# in other words, the encoder ignores the input data! therefore the decoder 
# reconstructs the data primarily from its learned prior or noise, rather than 
# utlizing meaningful information encoded in the latent space. 
# (it uses the patterns it learns from the prior distribution))
# #! edit, this needs more refining,
# sidenote:
# to refresh our memory :
# the posterior distribution (q(z|x)) represents the distribution 
# of the latent variable(vector) (z) conditioned on the input data (x).
# in other words, it means given this input data x, what is the
# probability distribution of the latent variable (z) (i.e. whats the (mu,sigma)?
# the encoder  approximates this posterior (q(z|x)) using a
# learned distribution, parameterized as a Gaussian distribution 
# that is q(z|x) = N(mu(x), sigma(x))
# where the encoder learns the mean (mu(x)) and variance (sigma(x)) for each input (x).
# 
# now, the prior distribution p(z) is a simple, predefined distribution 
# over the latent space (z). we chose the prior to be p(z) = N(0, I)
# which if you remember means z is assumed to come from a multivariate
# Gaussian distribution with mean 0 and identity covariance (diagnol covariance/independent dimensions! see the previous discussion!)
# 
# the kl term enforces a regularization on the latent space, ensuring that
# 1. q(z|x) i.e. posterior distribution does not deviate too far from the simple prior p(z).
# 2. the latent space stays smooth and meaningful, making it easier to sample from.
# 
# Why is it bad if the Posterior becomes too close to the prior?
# if it gets too close to the prior distribution it means mu(x) ≈ 0
# and sigma(x) ≈ I for all inputs, which is when we say it collapses
# to exactly match the prior (p(z)). 
# basically it says the inputs (X) are ignored!(the whole point was to learn mu,sig for each x
# and now, it treats as if they dont exist at all! it simply models N(0,I), i.e. noise!)
# 
# When this happens technically speaking we say:
# the latent space becomes uninformative because q(z|x) no longer depends
# on the input x. 
# it means the encoder gives up on learning meaningful latent representations, 
# and the decoder reconstructs the data purely from noise sampled from (p(z) = N(0,I)),
# or directly learns shortcuts from the reconstruction loss(this happens whne we us skip con forexample).
# The vae essentially fails to use the latent space for encoding useful information
# about the data.
# 
# !edit
# This happens because the KL divergence is minimized too quickly and thus it
# overpowers the reconstruction loss from the begining.
# 
# we want the posterior q(z|x) to be "close enough" to the prior p(z) for regularization,
# but not so close that it practically makes the model ignore x.
# The reconstruction loss job is to make sure the posterior q(z|x) encodes 
# meaningful information about the input x, while the KL divergence's job is 
# to make sure the latent space remains smooth and aligned with the prior.
# (we dont want our posterior to deviate from the prior, because
# we assumed given our prior we can regenerate the samples that look like our input)
# 
# now you know why we dont just minimize the KL loss alone cuz it would collapse q(z|x)
# to p(z), causing no meaningful relationship between x and z (posterior collapse)
# and also it would cause the decoder to reconstruct data from noise or directly 
# minimize reconstruction loss without using the latent space.
#
# so we use both losses together and balance reconstruction and KL loss:
# loss = reconstruction_loss + beta . kl_loss(q(z|x) | p(z))
# this ensures the latent space is regularized (via KL),
# and the encoder learns meaningful encodings of the data (via reconstruction loss).
# 
# # !edit
# recap
# so for short: 
# The prior distribution p(z) is fixed and simple (N(0, I)).
# The posterior distribution q(z|x) is learned and depends on the data x using reconstruction loss in encoder.
# The goal of the KL divergence is not to minimize it to zero, 
# but to balance it with the reconstruction loss to maintain a useful latent space.
# posterior collapse happens when the KL divergence dominates, leading the encoder 
# to ignore (x) and match the prior directly.
#
# prior distribution p(z): 
# is a predefined distribution over z (e.g. N(0, I))
# regularizes the latent space to stay simple and smooth
# independent of the data x.
# fixed by design (e.g., Gaussian)
# 
# posterior distribution q(z|x) 
# is the distribution of z given data x, learned by the encoder
# encodes meaningful information about x into z.
# depends on the input x
# learned by the vae during training.
# 
# 
# !edit
# If the encoder is too simple (e.g. insufficient capacity, few layers, 
# or too small latent dimensions), it may fail to encode meaningful 
# representations of the input. this makes it easy for the latent space
# to drift toward the prior, as the KL divergence loss 
# (minimizing the distance between posterior and prior)
# dominates over reconstruction loss.
# 
# note that posterior collapse can happen for several reasons, a simple or underpowered 
# encoder is one of the possible causes. However, its often the result of an interplay 
# of factors rather than just the simplicity of the encoder.
# 
# to be more precise, this happens when the kl divergence term dominates the loss.
# kl divergence job is to ensur the latent space follows a prior (i.e. a Gaussian N(0, I))  
# but when the kl term is too strong, the model learns to set q(z|x) ≈ p(z) )(i.e., 
# the posterior collapses to the prior), making z uninformative.
# as we pointed out this usually happens when we use a powerful decoder
# that can reconstruct the data directly from the prior distribution,without 
# needing latent variables.
# 
#edit: obvious?/excessive? 
# if the decoder is too powerful, it can learn to reconstruct x without 
# relying on z at all which means even if z contains no useful information,
# the decoder can still reconstruct well, leading to collapsed latents.

# thats not the only reason though, if we use a large scaler/factor to normalize the loss 
# (the kl term and reconstruction loss (its especially the case in beta-vaes(distenagled vaes) 
# which we will also cover)), a large scaler in the kl term forces the latent distribution 
# too close to the prior, increasing the risk of posterior collapse.
# when the scaler is too high, the vae prioritizes regularization over 
# learning meaningful latent representations.
# 
# the issue could also stem from the reparameterization trick,
# if you recall, the reparameterization trick job was to introduce 
# randomness when sampling from (q(z|x)), if the model learns to 
# reduce this randomness (e.g. by making standard deviation very small),
# the latent space may become degenerate. its worth noting that if the 
# latent space is too small, it may also be forced to collapse.

# so it could be several things that can contribute to this issue, altogether or alone. 
# likewise, there are several solutions/techniques that can help mitigate this issue
# and in a way they all do this by balancing the reconstruction quality and latent space 
# learning properly. 
# 
# the first and most obvious one is to reduce the kl scaler/weighting if its
# set too high, if this is not the case, and we still face issues during training, 
# then we can use a gradual approach, that is instead of a fixed scaler, 
# gradually increase it over time (e.g. use a kl annealing schedule).
# this should prevent the model from collapsing too early and should give meaningful
# latent encodings(i.e. we start with a very small beta (e.g like 0.0001) and increase it slowly to beta=1)
# practically starting with no kl constraint, and gradually increasing it little by litlle
# so we get to a good spot)
#
# we can also instead of minimizing kl loss entirely, enforce a minimum kl value per
# latent dimension (e.g. 0.1) this forces the model to use latent encodings
# even when kl regularization is high. (we dont want our kl term to be 0 or near zero
# so enforcing a minimum value of kl for each dimension essentially means we are making
# that dimension to do somework and contribute a bit so cllectively, the latent space
# gets to have at least some useful information for the decoder to utilize)
# 
# using a less powerful decoder is another obvious choice, since if its too powerful,
# it may learn to ignore latent vector z altogether. this is straightforward
# we just start using fewer layers or smaller networks or use a stronger bottleneck
# constraint.(this is only the case if we know for sure our encoder is working properly
# and its not the simplistic one between the two, because obviously if its encoder that
# needs more capacity, reducing decoders capacity wouldnt help!)
# also increasing latent space size(if its too small) can also help distribute 
# information across more dimensions and fix the issue.(usually reduces the severity of the issue)
# 
# we can also add skip connections between the encoder and decoder so
# that reconstruction does not fully rely on latent vector z.
# by doing so, we allow some direct flow of information
# from the encoder to the decoder which reduces the decoder's 
# reliance on a potentially collapsed z, encouraging meaningful latents)
# the idea is if the decoder still gets useful low-level 
# features even if z is uninformative, we can prevent posterior collapse!
# however, most often than not, if we dont tune this properly, it will cause 
# a posterior collapse itself! as it will make the decoder completely ignore the
# latent variables, and therefore they would not get the chance to get properly 
# tuned! it will result in getting near prefect reconstructions, but the random generation
# would be terrible, a sign of garbage/uninformative/collapsed latent space!
#
# 
# sidenote:
# !EDIT include the paper names/urls/refs
# this doesnt really belong here, because it belongs to heirarchial vaes
# but since the idea makes sense, I guess I include it here. (it really should 
# be explained in its own section). anyway lets explain this as well: 
#
# Assuming the latent vector z follows a simple Gaussian prior,
# p(z)=N(0,I) (i.e. all latent dimensions are independent and normally distributed
# around 0 with unit variance) as we already discussed, 
# can be too simplistic for complex data such as faces, natural images, sentences etc,
# if we relax this constraint by using a more structured latent space(e.g 
# hierarchical priors), we should get a much better result, 
# that is instead of assuming a single Gaussian prior ,
# we introduce a structured or hierarchical latent representation.
# this allows latent variables to be dependent on each other, 
# leading to a more flexible and powerful model.
# (that is, instead of just one latent variable z, we introduce multiple
# latent levels, higher level latents control more abstract/global 
# features, while lower levels refine details.
# !edit paper link
# sidenote:
# the idea comes from hierarchical vae paper,which proposed instead of one 
# latent vector z we use multiple latent vectors! the latters depending on the previous ones.
# (assuimg we use 2 latent vectors, the second mu,logvar would use the first latent vector z 
# to create the second set of mu and logvar which we would then use to create latent vector z2
# using reparameterization trick! and ultimately use this second z to reconstrcut the image)
# the idea was having multiple layers of latent variables with
# dependencies between them would improve expressiveness of
# the latent space, and help the network to model complex multimodal distributions
# and more importanty reduce posterior collapse, as higher layers 
# retain the global structure while lower layers capture 
# the finer details(z1 learns higher level global features and z2 
# learns lowerlevel fine-grained features ideally! in practice we can have multiple
# and therefore this lowlevel/highlevel featres can get a bit nuisaunced! but you get the idea
# multi-level of features/independent of others, practically trying to get arund the simplistic
# assumption in our default vae imple,enttaion without introducing the overhead we discussed earlier!)
# (we dont bother going this route though!)
# !Edit add refs/papers- check papers/refs
# ref https://arxiv.org/abs/1705.07120
# a similar approach was introduced by VampPrior (Variational Mixture of Gaussians)
# which said, instead of a single Gaussian prior, use a mixture of Gaussians.
# this captures multi-modal distributions (e.g. different facial expressions in images)
# the abstract reads: 
# Many different methods to train deep generative models have been introduced in the past. 
# In this paper, we propose to extend the variational auto-encoder (VAE) framework with a
# new type of prior which we call "Variational Mixture of Posteriors" prior, 
# or VampPrior for short. The VampPrior consists of a mixture distribution 
# (e.g., a mixture of Gaussians) with components given by variational posteriors 
# conditioned on learnable pseudo-inputs. 
# We further extend this prior to a two layer hierarchical model and show that this
# architecture with a coupled prior and posterior, learns significantly better models.
# The model also avoids the usual local optima issues related to useless latent dimensions 
# that plague VAEs. 
# We provide empirical studies on six datasets, namely, static and binary MNIST, OMNIGLOT, 
# Caltech 101 Silhouettes, Frey Faces and Histopathology patches, and show that applying the
# hierarchical VampPrior delivers state-of-the-art results on all datasets in the unsupervised
# permutation invariant setting and the best results or comparable to SOTA methods for the 
# approach with convolutional networks. 
# 
# we can also use VQ-VAE (Vector Quantized VAEs) which came to solve the vae issues (like posterior collapse)
# it replaces the continuous latent space with discrete latent embeddings, making the
# model less prone to collapse. and it works much much better than vaes, and has been
# very influential well cover this after we are done with vae!

# so now lets rewrite our vae, this time with the enhancements
#

#!edit use this instead of the above? or merge or use as recapt?
# Posterior collapse can happen for several reasons, and yes, a simple 
# or underpowered encoder is one of the possible causes. 
# However, its often the result of an interplay of factors 
# rather than just the simplicity of the encoder. 
# 
# 
# Reasons for Posterior Collapse:
# 
# Overly Powerful Decoder:
# if the decoder is too powerful (for the dataset/ or compared to encoder)
# it can learn to reconstruct the data directly from the prior distribution(N(0,I)) 
# or even from noise. the encoder then has no incentive to learn meaningful 
# latent representations leading to collapse.

# Simple or Underpowered Encoder:
# if the encoder is too simple (or it has too small latent dimensions),
# it may fail to encode meaningful representations of the input data
# this makes it easy for the latent space to drift toward the prior, 
# as the kl loss dominates over the reconstruction loss.
#
# the kl term job is to make the posterior align with the prior distribution (minimize the distance between them)
# but if this term is given too much weight (e.g with a large beta), 
# the encoder will prioritize minimizing kl term over learning a meaningful posterior
# this forces the latent space to collapse to the prior.

# Poor Training Dynamics (Learning Rate, Warm-Up):
# at the begining of the training, the decoder may dominate because it learns faster
# than the encoder this can result in posterior collapse because the 
# encoder gets stuck in a local minimum where it ignores the latent space entirely.
# without a kl warmup schedule (gradually increasing the weight of kl term during training,
# the kl term can overwhelm the reconstruction loss early on.

# insufficient Regularization in Latent Space:
# if theres no mechanism to ensure meaningful latent representations
# (e.g., free-bits regularization, disentanglement techniques), 
# the encoder might collapse to the simplistic/trivial solution of aligning the
# posterior with the prior.

# if the dataset is simple (e.g., small or low-dimensional), 
# the decoder may easily reconstruct data without requiring meaningful latent codes.
# this can be seen in datasets like MNIST where the decoder can perform well using 
# only prior information.

# Signs That the Encoder is Too Simple:
# The kl term quickly drops to zero during training, even for complex data.
# the reconstruction loss may improve, but the latent space doesn't 
# encode useful information (latent codes are random or meaningless).
# increasing the capacity of the encoder (e.g., deeper layers, more neurons)
# significantly improves performance.

# How to Fix Posterior Collapse (When the Encoder is Too Simple):
# Increase Encoder Capacity:
# add more layers or neurons to the encoder.
# use techniques like residual connections or
# attention to make the encoder more expressive.

# Regularize the Decoder:
# reduce the decoder's capacity to prevent it from
# "cheating" and relying on the prior.
# add dropout or other regularization techniques to 
# the decoder.

# KL Warm-Up Schedule:
# gradually increase the weight of the kl term during 
# training so that the encoder learns meaningful representations 
# before being forced to match the prior.

# Free-Bits Regularization:
# enforce a minimum KL loss for each latent dimension to ensure 
# that the encoder uses the latent space effectively.

# reduce beta as a high beta value can over amplify/over priortize the kl term.

# Change the Prior Distribution:
# use a more expressive prior (e.g., hierarchical or structured priors)
# that better matches the data distribution, so the encoder doesnt
# collapse to a simple normal distribution.

# A simple encoder can contribute to posterior collapse, but its not the sole reason.
# The issue typically arises from a combination of:
# an expressive decoder,
# overweighting of the KL term,
# poor training dynamics, or
# simple data.
# by addressing these factors holistically, we can prevent posterior collapse 
# and ensure the model learns meaningful latent representations.


class Print(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def forward(self, outputs):
        print(f'{outputs.shape=}')
        return outputs

class VAE(nn.Module):
    def __init__(self, embedding_size=100, input_channel=1, skip_connection=False, add_extra_noise=False, noise_weight=0.1, ema_decay=0.99):
        super().__init__()
        
        self.embedding_size = embedding_size
        self.input_channel = input_channel
        # whether to use skip connection from encoder to decoder
        self.use_skip_con = skip_connection
        # update: ok this moving average thing didnt work! Im not sure
        # If I implemented this incorrectly or what! I leave it be
        # for reference though, ill get back to it later
        # 
        # if we use skip connections between encoder and decoder
        # we will not be able to generate images using simple sampling
        # because we would need encoder outputs to concat with latent vectorz
        # and since there will be no image at first(cuz we are trying to generate some!
        # ourselevs from latent vector z!) we will face issue.
        # one way would be to use a image and use its encoderoutputs with our
        # latent varibles, but this is not desired!, the other way would be
        # to use the mean of our dataset! or a large batch and use that instead
        # of any random image! this is not really desired either or 
        # the third way, to have a running average of our encodersoutput
        # during training, and use this, when theres no images. 
        # this might give us a btter output. lets try this (I dont know if it works! im just trying!)
        # we define self.ema_skipcon at the end, when we defined our network and 
        # all sizes are determined!
        self.ema_decay = ema_decay        
        
        # whether to use extra noise in latent vector z
        # to make images more diverse(in fact prevent them from posterior collapse)
        self.add_extra_noise = add_extra_noise
        # a simple weight to control the amount of noise applied on our z
        self.noise_weight = noise_weight
        
        self.encoder = nn.Sequential(conv(self.input_channel,32),#28x28
                                     conv(32,64,stride=2),#14x14
                                     conv(64,96,stride=2),#7x7
                                     conv(96,128,stride=2),#3x3
                                     conv(128,256,stride=1),#2x2 # for 4x4: 1
                                     # set stride to 1 so the final output dim is 2x2
                                     # it helps for more complex datasets, but for mnist
                                     # a simple network would work, even a single fc layer!
                                     # so I decided to add a few more convs so we can experiment
                                     # with cifar as well
                                     # Initially I would get blury outputs, so I removed batchnorm
                                     # form the last layer of encoder and first layer of decoder
                                     # and could finally train and get sharp reconstructions
                                     # however, I also used larger spatial dms in encoders last layer
                                     # I just used the batchnorm again and noticed I got good results
                                     # I guess it was my choice of hypaerparameters as well.
                                     # so my verdict is, if things dont work, after you tried everything
                                     # try disabling batchnorm and see if it fixes the issue.
                                     # using batchnorm for other layers is ok, but for this layer
                                     # and encoders first layer might pose an issue (i'll edit this when my results are finalized)
                                     # ok bn can sometimes introduce weirdness into logvar! so if we get
                                     # huge values for mean/logvar (causes our kl loss to go nan!)
                                     # bn could be at fault (we need to start initilizing them properly
                                     # and see if it goes away if not disable bn and test again! see notes for logvar below)
                                     conv(256, self.embedding_size,stride=1,padding=1,batch_norm=True),#1x1 #for 4x4:1 # for 2x2:1 #for 1x1:2
                                    )
        # retaining some spatial dimensions such as 2x2/4x4 helps
        # when the dataset is more complex.
        self.bottleneck_size = self.embedding_size*4*4
        self.fc_mu = nn.Linear(self.bottleneck_size, self.embedding_size) 
        self.fc_logvar = nn.Linear(self.bottleneck_size, self.embedding_size)
        
        self.drp = nn.Dropout(0.1)
        # update:
        # ok during training with certain choices of hyperparameters
        # I noticed my kl loss goes to nan! 
        # looking closer I found out the logvar value was extremely high (like 1198352117964.0283!),
        # this happens because the slightest divergance
        # from proper range in logvar shoots us in the foot by exp!
        # so a common practice is to initialize the bias of the layer 
        # that produces logvar to a small negative value (for example,-1 or-2 or 
        # any small negative value we want for that matter)
        # so that initially exp(0.5xlogvar) is close to 1. 
        # for example using -1 forces the logvar to be std=exp(0.5x(−1))≈exp(−0.5)≈0.6065) 
        # self.fc_logvar.bias.data.fill_(-1)
        # ok this wasnt the issue! im clueless at this point! im removing bn now
        
        decoder_in_dim = self.embedding_size + self.bottleneck_size if self.use_skip_con else self.embedding_size
        # we use the followng formula to determine the output size here
        # ((h-1)*stride)+(kernel_size-2)*padding
        # h is the height for encoders output dim (here 1x1)
        # k is kernel , s is stride and p is for padding
        # (h=1,k=4,s=2,p=1)
        self.decoder = nn.Sequential(nn.Linear(decoder_in_dim, 256*4*4),
                                     # this bn seems crucial for stabalizing large embdsizes (like 400+)
                                     # without it loss shootsup alot! and reconstructions will be blurry
                                     # 
                                     nn.BatchNorm1d(256*4*4),
                                    #  nn.GroupNorm(1,256*4*4),
                                    # using leakyrely early on makes the model more unstable!loss shoots us quickly!
                                    # but other layers it seems ok to use leakyrelu
                                     nn.ReLU(),
                                    #  nn.Dropout(0.1),
                                     nn.Unflatten(1,(256,4,4)),
                                     # in encoderpart, relu seems to work better
                                     # but in decoder, leakyrelu works better it seems
                                     # batchnorm is especially important for large embdsizes
                                     # also except the last layer, the later layers having
                                     # bn makes somewhat sharper generations
                                     deconv(256,128,kernel_size=2,stride=2,batch_norm=True),#4,2 #for 4x4: 2,2  #for 1x1:4,2
                                    #  Print(),
                                    #  conv(128,128,kernel_size=3,stride=1,batch_norm=True),
                                    #  Print(),
                                     deconv(128,96,kernel_size=4,stride=1,batch_norm=True),#4   #for 4x4: 4,1  #for 1x1:4,2
                                    #  Print(),
                                    #  conv(96,96,kernel_size=3,stride=1,batch_norm=True),
                                    #  Print(),
                                     deconv(96,64,kernel_size=4,stride=2,batch_norm=True),#8    #for 4x4: 4,2  #for 1x1:4,2
                                    #  Print(),
                                    #  conv(64,64,kernel_size=3,stride=1,batch_norm=True),
                                    #  Print(),
                                     deconv(64,32,kernel_size=2,stride=1,batch_norm=True),#14    #for 4x4: 2,1  #for 1x1:2,2
                                    #  Print(),
                                    #  conv(32,32,kernel_size=3,stride=1,batch_norm=True),
                                    #  Print(),
                                     # while we use sigmoid here with bce, for more complex dataset
                                     # using tanh with mse seems to give better result, but
                                     # note that, the input needs to be normalized as well (to -1,1)
                                     # for our case we go with sigmoid anyway
                                     deconv(32,self.input_channel,kernel_size=6,batch_norm=False,act=nn.Sigmoid()),#28 #for 4x4:6 # for 1x1:4
                                    )
        
        # not that different from deconv when all bn is used, but upsample avoids checkermarks
        # self.decoder = nn.Sequential(nn.Linear(decoder_in_dim, 256*4*4),
        #                              nn.BatchNorm1d(256*4*4),
        #                              nn.ReLU(),
        #                              nn.Dropout(0.1),
        #                              nn.Unflatten(1,(256,4,4)),
        #                              # in encoderpart, relu seems to work better
        #                              # but in decoder, leakyrelu works better it seems
        #                              upconv(256,128,kernel_size=2,scale_factor=2,batch_norm=True),#4,2 #for 4x4: 2,2  #for 1x1:4,2
        #                              upconv(128,96,kernel_size=4,scale_factor=1,batch_norm=True),#4   #for 4x4: 4,1  #for 1x1:4,2
        #                              upconv(96,64,kernel_size=4,scale_factor=2,batch_norm=True),#8    #for 4x4: 4,2  #for 1x1:4,2
        #                              upconv(64,32,kernel_size=4,scale_factor=2,batch_norm=True),#14    #for 4x4: 2,1  #for 1x1:2,2
        #                              # while we use sigmoid here with bce, for more complex dataset
        #                              # using tanh with mse seems to give better result, but
        #                              # note that, the input needs to be normalized as well (to -1,1)
        #                              # for our case we go with sigmoid anyway
        #                              upconv(32,self.input_channel,kernel_size=4,scale_factor=1,batch_norm=False,act=nn.Sigmoid()),#28 #for 4x4:6 # for 1x1:4
        #                             )
        
        # self.decoder = nn.Sequential(nn.Linear(decoder_in_dim, 256*4*4),
        #                              nn.BatchNorm1d(256*4*4),
        #                              nn.ReLU(),
        #                              nn.Dropout(0.1),
        #                              nn.Unflatten(1,(256,4,4)),
        #                              PixelShuffleBlock(256,128,upscale_factor=2,batch_norm=True),#4,2 #for 4x4: 2,2  #for 1x1:4,2
        #                             #  Print(),
        #                              PixelShuffleBlock(128,96,upscale_factor=1,batch_norm=True),#4   #for 4x4: 4,1  #for 1x1:4,2
        #                             #  Print(),
        #                              PixelShuffleBlock(96,64,upscale_factor=2,batch_norm=True),#8    #for 4x4: 4,2  #for 1x1:4,2
        #                             #  Print(),
        #                              PixelShuffleBlock(64,32,upscale_factor=2,batch_norm=True),#14    #for 4x4: 2,1  #for 1x1:2,2
        #                             #  Print(),
        #                              # while we use sigmoid here with bce, for more complex dataset
        #                              # using tanh with mse seems to give better result, but
        #                              # note that, the input needs to be normalized as well (to -1,1)
        #                              # for our case we go with sigmoid anyway
        #                              nn.Conv2d(32, self.input_channel, kernel_size=7, padding=1),#28 #for 4x4:6 # for 1x1:4
        #                             # Print(),
        #                             )
        

        # now lets define our ema_skipcon 
        if self.use_skip_con:
            # we use self.register_buffer so ema_skipcon is saved when we save our model
            # and also its not included in computational graph
            self.register_buffer("ema_skipcon",torch.zeros(size=(1,self.bottleneck_size)))

    def reparamtrization_trick(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        z = mu + eps*std
        if self.add_extra_noise:
            # if the latent space is too smooth, it will create generic images(not varied enough)
            # by making the latent encodings more random/adding more randomness, we introduce more
            # diversity(the decoder should be able to create more diverse/different images (hopefully!))
            # adding noise to latent vector forces latent space to be used aswll, (making it 
            # more random makes decoder try harder and pay more attention to the latentspace
            # to also model the noise, otherwise, it could follow a simple normal distribution and hence
            #! ignore latent space altogether! (edit check my explanation/reasoning))
            # I added a noiseweight so we can have finer control over the added noise!
            z += self.noise_weight*torch.randn_like(z)
        return z
    
    def calculate_ema(self, encoder_outputs):
        # take the average of the whole batch
        outputs_mean = encoder_outputs.mean(dim=0)
        # this is standard ema calculation and works well, but 
        # since we use 0s at first, it has a bias towards zero
        # and it takes time to get it to accurate result
        # self.ema_skipcon = (self.ema_decay * self.ema_skipcon) + ((1-self.ema_decay)*outputs_mean)
        # by the way since we are using buffer, we cant use direct assignment 
        # so instead we use copy_ to have inplace operation to retain the buffer nature
        # of ema_skipcon, otheriwse it will be replaced by a tensor!
        # we could also use inplace ops like mul_,add_, etc as well but I guess copy_ is easier
        # the formula is readable and we get the job done!)
        self.ema_skipcon.copy_(self.ema_decay * self.ema_skipcon + (1 - self.ema_decay) * outputs_mean)
    
    def encode(self, input):
        # todo: add heirarchical z
        output = self.encoder(input)
        # print(f'{output.shape=}')
        output = output.view(input.size(0),-1)
        # dont use drpout on mu, it makes everything worse!
        mu = self.fc_mu(output)
        log_var = self.fc_logvar(output)
        # log_var=self.drp(log_var)
        z = self.reparamtrization_trick(mu, log_var)
        # if we use skip connection, lets update the moving average
        if self.use_skip_con:
            self.calculate_ema(output)
        
        return z, output, mu, log_var

    def decode(self, z, encoder_output):
        if self.use_skip_con:
            # use a moving average if theres no image/encoder-output
            if encoder_output is None:
                # since ema_skipcon is one vector, we need to repeat it
                # for the whole batch, so we can concat each row!
                # using repeat() function we can specify the repetition factor
                # for each dimension positionally, since in our case we have
                # a 2d vector, we use z.size(0) for the first dim as the batch dim
                # and use 1 for the second dim meaning we dont want to touch it!
                # leave it be as is!
                encoder_output = self.ema_skipcon.repeat(z.size(0),1)
            z = torch.cat([z, encoder_output], dim=-1)
        reconstructed_img = self.decoder(z)
        return reconstructed_img
    
    def forward(self, input):
        z, encoder_output, mu, logvar = self.encode(input)
        reconstructed_img = self.decode(z, encoder_output)
        return reconstructed_img, mu, logvar

    def calculate_loss(self, outputs, inputs, mu, logvar, beta, reduction='sum', use_mse=False, use_freebits=False, min_kl=0, normalize=True):
        b,h,w,c = inputs.shape
        outputs = outputs.view(*inputs.shape)
        criterion = nn.MSELoss(reduction=reduction) if use_mse else nn.BCELoss(reduction=reduction)
        reconstruction_loss = criterion(outputs, inputs)
        # weight for reconstruction loss 
        # we apply it only when reduction='mean'(I explaned below)
        scaler = 1
        # !edit
        # free bits regularization technique from https://arxiv.org/abs/1611.02731
        # ref https://stats.stackexchange.com/questions/267924/explanation-of-the-free-bits-technique-for-variational-autoencoders
        # enforcing a minimum kl loss value for each latent dimension  
        # prevents the kl term from going below min_kl value, essentially
        # making sure each latent dimension contributes at least some fixed
        # amount of information(at least some information is stored in latent space)
        # and therefore prevents the encoder from collapsing all 
        # latent dimensions to zero variance
        # # freebits ensures each latent dimension carries some information, clamp KL loss to min value
        if use_freebits:
            #! check if my implementation is correct/ if explanation is correct
            # calculates kl term for each dimensions (shape: (batch, latent_dim)
            kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
             # enforce minimum kl per each dimension for the whole batch (does it make it wosre?!)
            # kl_loss = torch.sum(torch.clamp(kl_per_dim, min=min_kl))
            # or we can only sum over the dimensions only and average that!?(which one?)
            kl_loss = torch.clamp(kl_per_dim, min=min_kl).sum(dim=-1).mean()
            # scale reconstructions?
            scaler = h*w*c if normalize else 1
        else:
            if reduction == 'sum':
                # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
                # since we are using sum as reduction for our reconstruction loss (all samples loss sum)
                # our kl loss needs to be summed over all dimensions and all samples in the batch 
                # which gievs us a single scalar value.
                # the bad thing is, since its summed over batch, the batchsize affects the training
                # we need to use different lr for different batchsizes because the gradients also
                # scale with the batchsize, therefore learning rate needs to be ajusted accordingly)
                # also this means more instability as its harder to balance the two terms like this
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            else: #reduction == 'mean'
                # here for the kl loss we only sum over the latent dimensions,
                # this gives us a single loss for each sample, 
                # we need to take the mean of the whole batch and this makes it independent of 
                # the batchsize and should give us a more stable loss, this is more aligned with
                # our reconstruction loss which we do the same thing (take the mean of the whole batch (i.e. reduction=mean))
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), -1)
                # we also need to normalize the reconstruction/kl loss otherwise kl will overpower it!
                # and we would get nonsens as output, (since the image is averaged pixelwise, but kl is summed for each sample
                # its not balanced properly)
                # we need to either divide kl loss by the image dimensions, 
                # or multiply reconstruction loss by the image dimensions to scale it up a bit
                # todo apply scaler to the freebits as well?
                scaler = h*w*c if normalize else 1
                # reconstruction_loss *= scaler
                # note since we sumed over the latent dimension, we will have batchsize of losses
                # which we need to average to get a single loss value
                kl_loss = kl_loss.mean(dim=0)

        # having a large weight for kl term (i.e. beta>1) can encourage a
        # structured and more meaningful latent space, but we need careful tuning
        total_loss = (scaler*reconstruction_loss) + (beta*kl_loss)
        return total_loss, reconstruction_loss, kl_loss

# test the vae and the output shape, making sure 
input_channel=3
test_model = VAE(embedding_size=100, input_channel=input_channel)
img_re, *_ = test_model(torch.randn(size=(5,input_channel,28,28)))
print(f'{img_re.shape=}')
#%%
# lets train our model again
# but this time, lets make things a bit tiddier!
# todo make this for epochs I guess thats better
def plot_training_metrics(mu_list, std_list, kl_losses, losses):
    epochs = range(len(mu_list))
    lists = (mu_list,std_list,kl_losses,losses)
    labels = ('Mean (μ)',"Standard Deviation (σ)", "KL Loss" , "Total Loss")
    colors = ['blue','orange','green','red']
    fig = plt.figure(figsize=(12, 8))
    for i,(lst,label) in enumerate(zip(lists, labels)):
        ax = fig.add_subplot(2, 2, i+1,)
        ax.plot(epochs, lst, color=colors[i], label=label)
        ax.set_title(f'{label} Over Time')
        ax.set_xlabel('Iterations')
        ax.set_ylabel(label)
    plt.legend()
    plt.tight_layout()
    plt.show()

import math
# we use this schedule to gradually increase beta
def beta_schedule(beta_max, epoch, total_epochs, midpoint=0.50, k=15):
    # scale epoch to [0,1]
    progress = epoch / total_epochs  
    # value goes from near 0 to near 1 around the midpoint.
    # by subtracting midpoint(0.5), we're shifting the range so that when progress is 0.5 
    # (i.e. halfway through training), the argument to the exponential is zero.
    # at this point, the sigmoid function yields half of its maximum value 
    # (when scaled by beta_max). this effectively means that the increase in 
    # beta is centered around the middle of training, with a slow start, 
    # a rapid increase around the halfway mark, and then a plateau as 
    # training approaches the end.
    current_beta = beta_max / (1 + math.exp(-k * (progress - midpoint)))
    return current_beta
# print(f'{[beta_schedule(0.5,i,100,k=10) for i in range(100)]}')
# k=15 seems ok!
# print(f'{[(i,beta_schedule(0.5,i,100,midpoint=0.5, k=15)) for i in range(100)]}')
# print(f'{[(i,beta_schedule(0.5,i,100,midpoint=0.85, k=15)) for i in range(100)]}')
# print(f'{[(i,beta_schedule(0.5,i,100,midpoint=0.85, k=10)) for i in range(100)]}')

def train(model:VAE, dataloader_train, optimizer, scheduler, device, epochs, beta, reduction, normalize, use_mse, interval, kl_anealing, use_freebits, min_kl=0):

    # a clear sign of posterior collapse is an extremely low kl term.
    # so if kl loss is close to zero, its a sign of collapse.  
    # so we keep track of it
    kl_losses=[]
    losses = []
    # also if all latent dimensions have almost zero variance,
    # it means they are not encoding useful information 
    # (they are roughly zero which means no learning is going on!!)
    # so by checking their values we can also get a hint!
    mu_list = []
    std_list = []
    
    print(f'Date:            {datetime.datetime.now().strftime("%H:%M:%S - %Y/%m/%d")}')
    print(f'Dataset:         {"CIFAR10" if model.input_channel==3 else "MNIST"}')
    print(f'Epochs:          {epochs}')
    print(f'embedding_size:  {model.embedding_size}')
    print(f'use_skip_con:    {model.use_skip_con}')
    print(f'add_extra_noise: {model.add_extra_noise}')
    print(f'beta:            {beta}')
    print(f'reduction:       {reduction}')
    print(f'normalize:       {normalize}')
    print(f'use_mse:         {use_mse}')
    print(f'kl_anealing:     {kl_anealing}')
    print(f'use_freebits:    {use_freebits}')
    print(f'min_kl:          {min_kl}')
    print(f'optimizer:       {optimizer}')
    print(f'scheduler:       {scheduler.state_dict()}')
    print(f'interval:        {interval}')
    
    model.train()
    for e in range(epochs):
        for i, (imgs, labels) in enumerate(dataloader_train):
            imgs = imgs.to(device)
            reconst_imgs,mu, logvar = model(imgs)
            
            # kl annealing prevents kl loss from overwhelming early training,
            # so we increase its beta gradually
            if kl_anealing:
                # a large kl term will cause posterior collapse, especially at begining
                # because before the model gets the chance to learn meaningful features
                # to reconstruct properly, the kl term had already forced it
                # to match the simplistic normal distribution(i.e. q(z|x) = (p(z)),)
                # so we start with a small scaler/beta and gradually increase it at 
                # each epoch this should allow our model to first learn reconstruction 
                # and then gradually apply the kl term (which mind you is a regulariziation term)
                # without it dominating the whole loss
                # linear one wasnt helping much, so I got a new one!
                # beta = min(beta,e/epochs)
                beta_current = beta_schedule(beta,e,epochs,k=15)
            else:
                beta_current = beta
            loss, recon_loss, kl_loss = model.calculate_loss(reconst_imgs, imgs, 
                                                             mu, logvar,
                                                             beta=beta_current, 
                                                             reduction=reduction,
                                                             use_mse=use_mse,
                                                             use_freebits=use_freebits,
                                                             min_kl=min_kl,
                                                             normalize=normalize
                                                             )
            
            losses.append(loss.item())
            # ideally, the kl loss should be balanced 
            # not too small and not too large
            # and defninetly nothing close to 0!
            kl_losses.append(kl_loss.item())

            # grab mean/stds 
            # if the standard deviation becomes too small and 
            # goes to nearly zero, the model isnt 
            # effectively using its latent space,
            # we should have a diverse range of latent activations
            # (again zero/close to zero, is a sign of 
            # not learning/contributing much to the whole process)
            # as for what we should be expecting, 
            # if the mean is always near 0 and std is near 1,
            # this means the model ignores the latent space 
            # (it means it has learned the prior (normal distribution N(0,I))!)
            mu_list.append(mu.mean().item())
            std_list.append(torch.exp(logvar*0.5).mean().item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step() 
            if i% interval ==0:
                print(f'Epoch {e}/{epochs} [{i}/{len(dataloader_train)}]'
                    f' | Loss: {np.mean(losses):.4f}'
                    f' | KL-Loss: {np.mean(kl_losses):.4f}'
                    f' | (μ,σ): ({np.mean(mu_list):.4f} , {np.mean(std_list):.4f})'
                    f' | lr: {scheduler.get_last_lr()}')
        scheduler.step()
    # plot mu/std, klloss and see how they behaved
    plot_training_metrics(mu_list, std_list, kl_losses, losses)    

def save_model(modelname, kwargs):
    try:
        torch.save(kwargs, modelname)
        print(f"{modelname} saved!")
    except Exception as ex:
        print(f'An Exception has occured: {str(ex)}')

def load_model(model, modelname, key='states'):
    # load the model 
    states = torch.load(modelname)
    model.load_state_dict(state_dict=states[key])
    print(f"{modelname}'s {key} loaded")
    # no need to return it, but do it in case we assign it
    return model

#! edit merge these two together? since we can display the diff
# next to plots as well!?
# some introspection functions to see if our model has collapsed!
def check_latent_representation_diversity(model:VAE, dataloader):
    # if we compare two different inputs latent vectors
    # and they are nearly the same, it means our model 
    # has collapsed! the encodings for different classes
    # must be very different.
    # if the difference is close to 0, the latent space 
    # is collapsing! there should be noticeable variation
    # between different images
    device = next(model.parameters()).device
    imgs,labels = next(iter(dataloader))
    # grab two random classes
    classes = torch.randint(0,10,size=(2,))
    # get the indexes for said classes
    indices = torch.where((labels == classes[0]) | (labels == classes[1]))[0]
    # and pick only two images for comparison
    imgs = imgs[indices[:2]]
    imgs = imgs.to(device)
    zs,_,mus,logvars = model.encode(imgs)
    # could do: 
    # difference = mus.diff(dim=0).abs().mean().item()
    # but since we have 2 its easier to read we do:
    difference = (mus[0]-mus[1]).abs().mean().item()
    print(f'difference between two latent images: {difference:4f}')
    # could use math.close(difference,0,abs_tol=1e-6) aswell
    # but this should do it as well (doesnt need an extra import!)
    collapsing = abs(difference)<1e-6
    print(f'Collapsing!!!' if collapsing else 'No collapsing. all is OK!')
    
# check if our latent space is smooth and gives us 
# smooth iterpolation between classes
def check_laten_representation_interpolation(model:VAE, dataloader, interpolation_steps=10):
    device = next(model.parameters()).device
    imgs,labels = next(iter(dataloader))
    # grab two random classes
    classes = torch.randint(0,10,size=(2,))
    # get the indexes for said classes
    indices = torch.where((labels == classes[0]) | (labels == classes[1]))[0]
    # and pick only two images for comparison
    imgs = imgs[indices[:2]]
    imgs = imgs.to(device)
    latent_vectors,encoder_outputs, *_ = model.encode(imgs)
    
    #lets create a nicely stepped vector of values
    # and use it to feed our decoder to see how our
    # our decoder interpolates between the latentvectors
    # with these values and whether the interpolation is 
    # smooth and final result looks good
    alphas = torch.linspace(0, 1, steps=interpolation_steps).to(device)
    interpolated_z = torch.lerp(latent_vectors[0], latent_vectors[1], alphas[:, None])
    # generate 10 interpolated images between our given latent vectors
    interpolated_images = model.decoder(interpolated_z)
    grid = make_grid(interpolated_images, nrow=interpolation_steps, normalize=True)
    plt.imshow(grid.cpu().numpy().transpose(1, 2, 0))
    plt.title("Latent Space Interpolation")
    plt.axis("off")
    plt.show()

@torch.no_grad()
def evaluate_on_testset(model:VAE, dataloader_test, sample_count=20, img_shape=(1,28,28), **kwargs):#beta=1, reduction='mean', use_mse=False, use_freebits=False, min_kl=0, normalize=True):
    test_set_size = len(dataloader_test.dataset)
    img_pairs = []
    losses = []
    interval = 10
    model.eval()

    for i, (imgs, labels) in enumerate(dataloader_test):
        imgs = imgs.to(device)
        labels = labels.to(device)
        reconst_imgs, mu, logvar = model(imgs)
        loss,*_ = model.calculate_loss(reconst_imgs, imgs, mu, logvar, beta, reduction, use_mse,use_freebits,min_kl,normalize)
        losses.append({'val_loss':loss.item()})
        
        print(f'[{i*len(imgs)} / {test_set_size} ({100.*i/len(dataloader_test):.2f}%)]'
            f'\tLoss: {(loss).item():.4f}')

        if i%interval==0:
            reconstructeds = reconst_imgs.cpu().view(-1, *img_shape)
            # grab the first few images and their reconstructions
            # sidenote: when we use no_grad, theres no gradients, so no need for .detach()!
            imgs = imgs[:sample_count].cpu().numpy()
            recons = reconstructeds[:sample_count].numpy()
            pairs = np.array([np.dstack((img1,img2)) for img1, img2 in zip(imgs,recons)])
            img_pairs.append(pairs)
    
    print(f'Testset Loss: {np.mean([entry["val_loss"] for entry in losses]):.4f}')
    # plot the losses using pandas! 
    # this actually is very neat and comes handy very often!
    # we can have a list of dictionaries, where each value is 
    # attributed by a key. this way, our keys will be used as
    # legends and we have a simple plot with minimum hassle
    import pandas as pd
    ax= pd.DataFrame(losses).plot()
    ax.set_title('testset loss')
    plt.show()
    
    display_imgs_recons(img_pairs, nrows=10, rows=8, cols = 1)

@torch.no_grad()
def generate_latent_space_grid(model:VAE, n=20,lower_bound=-2, upper_bound=2, img_shape=(1,28,28), img:torch.Tensor=None):
    print(f'using {lower_bound=} and {upper_bound=}')
    # lets see if the transition in our latent space is smooth
    # that is we should be able to smoothly transition from one
    # class to the other, at least this is what we are tryting 
    # to see.
    # we create a vector of equally spaced values, and try to
    # visualize these vectors, (they act as our latent vector z)
    # since they are equally spaced, we can see how they change
    # gradually, ideally we want them to have a smooth transition
    # from one class to another. 
    # so lets see how our interpolation turns out
    # n means we want a figure with nxn digits (note we assume our latent vector dim is 2)
    model.eval()
    # we are basically creating a z vector, with n, equally spaced value
    # starting from lowerbound, up until upperbound (e.g from -2 to 2)
    # we create 2 such vectors, so we can create a grid of numbers
    # treating one z for xaxis and another for the yaxis. 
    z1 = torch.linspace(lower_bound, upper_bound, n)
    z2 = torch.linspace(lower_bound, upper_bound, n)
    # using np.meshgrid, we create our grid, meshgrid, simply 
    # expands z1 and z2 into 2D grids, by first repeating z1 values in
    # x-axis (rows) and then repeating the z2 values in y-axis(columns),
    # we finally using np.dstack, combined them and the result 
    # will be a 3dgrid where each xy is made up of z1 and z2 values.
    # (test with a small example like linspace(-2,2,5), and see how it goes
    # visualizing it gives you a pretty good idea whats happening here)
    z_grid = np.dstack(np.meshgrid(z1, z2))
    z_grid = torch.from_numpy(z_grid).to(device)
    # print(f'{z_grid.shape=}')# nxnx2
    z_grid = z_grid.reshape(-1, model.embedding_size)
    print(f'{z_grid.shape=}')#(nxn, embdsize) 
    # 
    # todo think about skipconnection visualization
    # we cant simply use zeros for fake encoder output, 
    # because decoder is codnitioned on it
    # and it must have valid values, because it relies 
    # upon some extra information present in it, 
    # so one way is to maintain a running_mean/average 
    # of all encoder outputs during training and use that mean
    # during testing for generating purposes, I tested it
    # and it doesnt work! we get nonsense (the code is 
    # commented out in our model for future references (maybe im doing it wrong here!))
    # the other way that came to my mind was to use an 
    # actual image, get its encoders output and use that instead.
    # this gives us the image, but theres no variation happeing!
    # another way that came to mind was to use several images,
    # a batch of images to be precise! that is take a batch of images,
    # their mean, feed this to the encoder and use its outputs
    # with our generation! this doesnt work either!
    # 
    # imgs,_ = next(iter(dataloader))
    # imgs = imgs.to(device)
    # img_mean = imgs.mean(0)
    # z,output,*_ = model.encode(imgs[0].unsqueeze(0))
    # print(f'{z_grid.shape=} {output.shape=} ')
    # how to concat? z_grid is 10x10x2, ours is 1xd (we need to repeat ours 10x10 times!
    # so they have the same batch dim and then concat them)
    # we need to reshape zgrid to have (-1,embdsize)
    # by default since our embdsize=2, it aligns prefectly
    # with the default 10x10x2 which is 100x2, but when embdsz
    # is bigger than 2, it falls apart. 
    # when we have more, we have to align them properly as z1,z2,z3,etc form. 
    # so if we want square, we need to take sqroot of embdsize
    # I guess (that wouldnt be possible though, because a grid is by nature 2d, xy and yx
    # anything larger than that doesnt make sense, because we cant have xyz, xzy, zxy,yzx, etc)
    # you get the idea, unless we create separate 2d grids for each combo which is nuts!
    # leaving us to use an embd size that when reshaped, aligns prefectly, ie. 
    # imgs,_ = next(iter(dataloader))
    # imgs = imgs.to(device)
    # z,output,*_ = model.encode(imgs[0].unsqueeze(0))
    # img_mean = imgs.mean(0)
    # z,output,*_ = model.encode(img_mean.unsqueeze(0))
    output=None
    if model.use_skip_con:
        z,output,*_ = model.encode(img)
        output = output.repeat(z_grid.size(0),1)
        # print(f'{z_grid.shape=} {output.shape=} ')
    # ema_skipcon = model.ema_skipcon.expand(z_grid.shape[0], -1)
    x_pred_grid = model.decode(z_grid,output)
    x_pred_grid= x_pred_grid.cpu().view(-1, *img_shape)
    x = make_grid(x_pred_grid,nrow=n).numpy().transpose(1,2,0)
    plt.figure(figsize=(32, 24))
    plt.xlabel('Z_1')
    plt.ylabel('Z_2')
    plt.imshow(x)
    plt.title(f'latent space grid of numbers({n}x{n})')
    plt.show()

# lets see what each classes mean/std looks like
# each have their own different mean,
# the mean is drastically different than other classes
# though otherwise it shows the model has not been trained properly
# we can use this to generate as many images we want for each class
# we can create as many 0s, 1s or any classes we want! using their mean/std
# (as you will see in a moment the variation needs some work but overall
# lets see how they look!)
# ! edit make samples more varied by altering std a bit
@torch.no_grad()
def generate_similar_images(model:VAE, input_img:torch.Tensor, count:int=64, rows:int=8):
    c,h,w=input_img.shape
    if len(input_img.shape) == 3:
        input_img = input_img.unsqueeze(0)

    # grab the device from our model parameters
    device = next(model.parameters()).device
    model.eval()

    input_img = input_img.to(device)
    # grab the mu/logvar for the image class
    z0, enc_outputs, mu, logvar = model.encode(input_img)
    # convert the logvariance to std
    std = torch.exp(0.5*logvar)
    # create latent vectorz by sampling using the mu/std
    # using random noise(epsillon) to create randomness in output
    epsillon = torch.randn_like(std)
    z_random = mu + epsillon * std
    # how mu+std sample looks like
    z_plain = mu + std
    # instead of a single epsilon, we can create as many as
    # we like, and therefore generate as many images. just
    # make sure the size matches
    epsillons = torch.randn(size=(count, mu.shape[-1]), device=device)
    # by changing the std, we can generate slightly different variatations
    # a higher std introduces more randomness, leading to more diverse outputs,
    # a lower value generates outputs closer to the mean(mu) which means less variation
    # we can change this in steps and create a morphing effect
    # from one image into another. (we will be implementing this in a moment)
    # scaler = torch.linspace(0.01, 0.03, count).to(device).view(count,1)
    # print(f'{scaler=}')
    # epsillons *= scaler
    # print(f'{epsillons=}')
    z_batch = mu + epsillons * std

    z_random, z_plain,z_batch = (z.to(device) for z in (z_random, z_plain, z_batch))
    # generate images for each latent vector
    img_random, img_plain, img_batch = (model.decoder(z) for z in (z_random,z_plain,z_batch))
    # reshape the decoder outputs to the proper image dims
    img_random, img_plain, img_batch = (img.view(-1,c,28,28) for img in (img_random, img_plain, img_batch))
    # combine the images as one so we can display them as one big image
    # imgs_combined = torch.concat([input_img,img_random,img_plain],dim=3)
    imgs_combined = torch.dstack([input_img,img_random,img_plain])
    # combine all images as one so we can better visualize and inspect them
    img_batch_grid = make_grid(img_batch, nrow=rows, normalize=True)
    
    mu = mu.cpu().numpy().flatten()
    std = std.cpu().numpy().flatten()

    plt.figure(figsize=(8, 4))#(12,8)
    
    plt.subplot(2,3,1)
    plt.plot(mu, label="Mean (μ)")
    plt.title("Mean (μ)")
    plt.xlabel("Latent dimension")
    plt.ylabel("Value")
    plt.legend()

    plt.subplot(2,3,2)
    plt.plot(std, label="Std (σ)", color="orange")
    plt.title("Std (σ)")
    plt.xlabel("Latent dimension")
    plt.ylabel("Value")
    plt.legend()
    
    plt.subplot(2,3,4)
    imgs_combined = imgs_combined.squeeze().cpu().numpy()
    imgs_combined = imgs_combined if c ==1 else imgs_combined.transpose(1,2,0)
    plt.imshow(imgs_combined, cmap="gray")
    plt.title("Input image")
    plt.axis("off")
        
    plt.subplot(2,3,5)
    plt.imshow(img_batch_grid.squeeze().cpu().numpy().transpose(1,2,0), cmap="gray")
    plt.title("Similar images")
    plt.axis("off")
        
    plt.tight_layout()
    plt.show()

# now lets generate new images by stepping through the latent space
import matplotlib.animation as animation
@torch.no_grad()
def create_interpolation_animation(model:VAE, filename='vis', sample_count=30, fps=30,mu=0.02,std=0.02):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    z = torch.randn(size=(sample_count, model.embedding_size)).to(device)
    model.eval()
    def animate(i):
        # change the latent vector at each step so we get different image
        # and ultimately a cool animation showing each image morphing into another!
        # note that by choosing a larger std(0.03 vs 0.01), we increase the randomness
        # so it changes faster. the more farther away from mean, the more different
        # it becomes from that image
        imgs = model.decoder(z*(i*std)+mu)
        b,c,h,w = imgs.shape
        imgs2 = imgs.view(imgs.size(0), c, h, w)#1x28x28 or 3x28x28
        new_img = make_grid(imgs2).cpu().detach().numpy().transpose(1,2,0)
        ax.clear()
        ax.imshow(new_img)

    anim = animation.FuncAnimation(fig, animate, frames=100, interval=300, repeat=True, repeat_delay=1000)
    # save the git using pillow
    anim.save(f'{filename}.gif', writer="pillow", fps=fps)
    plt.show()

#%%
# before we start our training lets have a quick review:
# if kl loss is too small we can use kl annealing or Free Bits  
# if latent space is unstructured we can use beta>1  
# if the decoder is too strong we need to reduce decoder capacity(large dropout,fewer layers etc)
# if all outputs look the same, we can add noise to z to fix that
# now lets start training!

# whenever you face CUDA error we try to debug it using this
# if we are using jupyter notebook, otherwise we can simply execute our script
# in terminal like this: 
# CUDA_LAUNCH_BLOCKING=1 python ourscript.py 
# but since we are in jupyternotebook environment, we do this in code:
# import os
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def select_dataset(dataset_name='mnist', batch_size=128, size=28):
    if dataset_name.lower() == 'mnist':
        dataset_train = datasets.MNIST('MNIST', train=True, download=True,transform=transforms.ToTensor())
        dataset_test = datasets.MNIST('MNIST', train=False, download=True,transform=transforms.ToTensor())
    
    elif dataset_name.lower() in ['cifar','cifar10']:
        # for cifar10 a better architecture and training regime is required
        transformations_tr = transforms.Compose([transforms.Resize(size),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.ToTensor(),])
        transformations = transforms.Compose([transforms.Resize(size), transforms.ToTensor(),])
        dataset_train = datasets.CIFAR10('CIFAR10', train=True, download=True,transform=transformations_tr)
        dataset_test = datasets.CIFAR10('CIFAR10', train=False, download=True,transform=transformations)
   
    else:
        raise Exception(f'the input dataset {dataset_name} is not supported! choose between (mnist or cifar10)')
    
    dataloader_train = torch.utils.data.DataLoader(dataset_train,batch_size=batch_size,shuffle=True)
    dataloader_test = torch.utils.data.DataLoader(dataset_test,batch_size=batch_size,shuffle=False)

    return dataset_train, dataset_test, dataloader_train, dataloader_test

dataset = 'mnist'
# # dataset = 'cifar10'
# batch_size = 128
dataset_train, dataset_test, dataloader_train, dataloader_test = select_dataset(dataset_name=dataset,
                                                   batch_size=batch_size)
imgs,lbls=next(iter(dataloader_train))
view_images(imgs,lbls)
#%%
# mnist test 
dataset = 'mnist'
batch_size = 128
dataset_train, dataset_test, dataloader_train, dataloader_test = select_dataset(dataset_name=dataset, batch_size=batch_size)

epochs = 50#50,100
interval = 2000

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# if its mnist use 1 if its cifar10 use 3 for input channel
input_channel = 1 if dataset =='mnist' else 3
# for mnist, 2 works(of course its not optimal, but works nonetheless), try different 
# embedding sizes here and see their effects for yourself(learning rate also affects the results so its not just the embeddingsize its the whole package!!)
# choose something even, it makes visualization easier(especially 
# for generate_latent_space_grid function since we use 10x10/20x20
# by default, but you can use larger or smaller grids as well, just 
# make sure the dims match up with embeddingsize so visualization 
# works well.(ultimately 10x10 is 100 and for visualization we do (-1, embdsize)
# and this specifies the number of samples, when ebdsize=2, it gives us 10x10 grid
# but when embdsize is larger, as you can see, the number of samples and grid follow
# our reshape (-1,embdsize) you get the idea) 
embedding_size = 2#,10,20,50
# in theory beta>1 forces the model to
# use latent space more efficiently,
# but larger beta may make the image 
# blurrier! so we use smaller beta
# try different betas and see their effect on
# both interpolation quality and latent space formation
# (the choice of beta becomes challanging with more complex datasets i.e. cifar10!)
beta=1 #0.001,1,2,4,
# reduction mean works much better than sum, 
# its batch invariant and is much more stable
reduction='mean'
use_mse=True
normalize = True # for reduction='mean'
kl_anealing=False
use_skipconnection=False
add_extra_noise=False
use_freebits=False
min_kl=0.5

model = VAE(embedding_size, input_channel, use_skipconnection, add_extra_noise).to(device)

# note high loss like the ones in the thousands(when using sum e.g.),
# will be compounded by large lr (it will cause the model to make 
# too large of updates causing it to never learn)
# this may not show itself that much in mnist, but when switching to other more 
# complex datasets it will definitely show, so one must use a much much lower lr!
# 
# remember too of a large lr will make the network diverge, 
# it will show itself as mean going toward 0 and std to 1,
# the kl loss will also be around 0, all showing 100% collapse.
# so if our training goes properly, and we see loss decrease properly we're fine!
# this shows itself in more complex datasets such as cifar. we talked about
# the sign to know which part needs attention, dont forget about rudimentary things like lr, and 
# other proper techniques in training!
# 
lr =0.01
weight_decay = 1e-3
scheduler_steps = [20,30,40,45]#[20,45,65,85] # [20,35,45,49]
optimizer = torch.optim.Adam(model.parameters(), lr =lr, weight_decay=weight_decay)#1e-4
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, scheduler_steps)

train(model, dataloader_train, optimizer=optimizer, 
      scheduler=scheduler,
      device=device,
      epochs=epochs, 
      beta=beta,
      reduction=reduction,
      normalize=normalize,
      use_mse=use_mse,
      interval=interval,
      kl_anealing=kl_anealing,
      use_freebits=use_freebits,
      min_kl=min_kl)

kwargs = {"states": model.state_dict(),
          "epochs": epochs,
          "embedding_size":model.embedding_size,
          "use_skipconnection":use_skipconnection,
          "beta":beta,
          "kl_anealing":kl_anealing,
          "use_freebits":use_freebits,
          "min_kl":min_kl,
          "reduction":reduction,
          "normalize":normalize,
          "use_mse":reduction,
          "optimizer":optimizer.state_dict(),
          "scheduler":scheduler.state_dict()}

timestamp = datetime.datetime.now().strftime("%H_%M_%S_%Y_%m_%d")
modelname = f"vae_{"cifar10" if input_channel==3 else "mnist"}_{model.embedding_size}_{reduction}_{'normalized' if normalize else 'not-normalized'}_{'mse' if use_mse else 'bce'}_{timestamp}.pth"
save_model(modelname=modelname, kwargs=kwargs)
#%%
# load the model to make sure we are dealing with the right model!
load_model(model, modelname=modelname)
img_shape=(input_channel,28,28)
kwargs = {"img_shape":img_shape,
          "beta":beta,
          "reduction":reduction,
          "use_mse":reduction,
          "use_freebits":use_freebits,
          "min_kl":min_kl,
          "kl_anealing":kl_anealing,          
          "normalize":normalize}
check_latent_representation_diversity(model, dataloader_train)
# fix these two for skipcon version
check_laten_representation_interpolation(model, dataloader_train, interpolation_steps=10)#check5,10,20
generate_random_images(model, count=32,img_shape=img_shape)
evaluate_on_testset(model, dataloader_test, **kwargs)
# this generation is broken for skipcon for now,
# we can send an input img, for the sake of running it 
# without errors, but the output doesnt work as we expect it
# I need to fix it!
generate_latent_space_grid(model,n=10,lower_bound=-2,upper_bound=2,img_shape=img_shape,img=None)
generate_latent_space_grid(model,n=20,lower_bound=-2,upper_bound=2,img_shape=img_shape,img=None)
plot_latent_space_encodings(model)
plot_embedding_clusters(model, dataloader_train, title='Encoder embedding',use_pca=False)
plot_latentspace_clusters(model, dataloader_train, title='Full latent clusters',use_pca=False)
# lets view some images and generate
# some only for a specific class
# note that our current approach only
# gives us little control over this kind of
# generation, in order to get diverse output
# even for the same class, we need to put an effort!
imgs,labels = next(iter(dataloader_test))
view_images(imgs,labels)
# mu/std slightly changes for different instance of a class
# but overall they are roughly the same, they
# however change dirastically from class to class
generate_similar_images(model, imgs[3])#0
generate_similar_images(model, imgs[13])
generate_similar_images(model, imgs[25])
generate_similar_images(model, imgs[2])#1
generate_similar_images(model, imgs[5])
generate_similar_images(model, imgs[14])
generate_similar_images(model, imgs[1])#2
generate_similar_images(model, imgs[32])#3
generate_similar_images(model, imgs[4])#4
generate_similar_images(model, imgs[15])#5
generate_similar_images(model, imgs[11])#6
generate_similar_images(model, imgs[0])#7
generate_similar_images(model, imgs[8])#8
generate_similar_images(model, imgs[7])#9
# the animation maynot work inside jupyternotebook, but the gif file works
create_interpolation_animation(model, filename=f'mnist_{timestamp}')

#%%
# torch.autograd.set_detect_anomaly(False)
# import os
# os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

# cifar10 test
dataset = 'cifar10'
batch_size = 32
dataset_train, dataset_test, dataloader_train, dataloader_test = select_dataset(dataset_name=dataset,
                                                                                batch_size=batch_size,
                                                                                )

epochs = 100#50,100
interval = 2000

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# if its mnist use 1 if its cifar10 use 3 for input channel
input_channel = 1 if dataset =='mnist' else 3

# 50 seems a fair choice for cifar10 example,
# larger values need more regularization though
# good choice is 384, but for visualization 200 is better for now
embedding_size = 600 #2,10,20,50,70,90,100,128,256,384,512
# in theory beta>1 forces the model to
# use latent space more efficiently
# but for our quick tests here, especially in cifar10,
# we set it to 0.001 for this to work for us (otherwrise
# using higher values would give us blurrier images which
# again signals us we need to work more on the hyperparameters!)
# at least in my experiments so far this has been the case.
# note:
# that tiny beta is actually cumbersome, it doesnt allow us
# to have proper generation/interpolation. we need to
# have a healthy amount of beta, cuz that small amount 
# means the kl term is effectively nonexistant! or very weak
# 
# larger beta values result in blurier output (try beta=1 vs 0.001)
# when you experiment with hyperparameters, only change one parameter at a time
# otherwise you wont beable to know what each parameters effects are on your model
# 
# if images are blury then we need to increase the embeddingsize
# 
# sidenote:
# Initially I tested no batchnorm for the
# last layer of encoder and first layer of decoder, because
# I couldnt get sharp reconstrcutions, if we dont use batchnorm
# for the last layer of encoder and first layer of decoder 
# we will get nans, and to get around this issue, 
# we need to make sure to lower the lr (0.001 seems ok)
# or use freebits. 
# However, I usually try a few times before I decide its too much 
# this is especially the case when no bn is used
# because when bn is used the training is stable,
# so as a rule of thumb I retry a few times, and 
# only if it doesnt work I lower the lr again.
# usually it works on the 3rd or 4th attempt and 
# it starts conveging without spitting out any nans!(or huge kl values!) 
# this way the results are sharp and clearer than others so far.
# 
# However, after many more experiments I noticed I 
# didnt have to remove bn for those layers, 
# I had to do some other modifications
# which I'll be pointout in a moment)
# if we use bn for last layer of encoder this wont happen, 
# but beta still has effect on the output and makes it blurrier. 
# this tiny value causes the latent space to be squashed around 0(center)
# as a tiny mass,all encodings squashed together 
# (see model vae_cifar10_100_mean_normalized_mse_09_40_42_2025_02_10.pth
# load it and visualize the latentspace to see what im talking about)
# and compare it with vae_cifar10_100_mean_normalized_mse_09_49_53_2025_02_10.pth
# where beta=1 is used, see how the latentspace is more expanded!
# 

# 
beta=0.2#0.0001 #0.0001, 0.001,1,2,4,
# reduction mean works much better for both mnist and cifar,
# its much more stable! especially for cifar10
reduction='mean' #mean
# mse seems to work better for cifar if skipcon isused
# but during our tests, bce gives sharper images!
use_mse=False #True
normalize = True # True for reduction='mean'
# when we remove batchnorm from layers,
# especially the last layer of encoder
# the loss can become really unstable
# kl annealing can help, but it depends
# on other parameters as well(lr formost,
# then skipcon and beta value)
# using bce, it doesnt help,
kl_anealing=True #False
# skipcon is especially effecive when training 
# cifar10 for example(without kl-annealing)
# (especially if encoding has spatial dims>1 like 2x2 or 4x4
# infact we dont get very blury reconstructions if we use 1x1
# using 2x2 dims for the encoder outupt improved the result,
# but it was 4x4 that made it for me, gave me the sharpest reconstructions)
# the problem with skipconnection is, it prevents us from easily
# creating generations, because we dont use any encoders, and thus
# theres no encoder output to incorporate into latentvector z!
# also it can make the network ignore latent vector!
# we can easily see that using skipconnection with mnist can 
# result in extreme posterior collapse! 
# I had to completely turn off kl to get somewhat working output!
# we couldnt reconstruct properly
# (its expected if you think about it, 
# using skipcon the decoder can ignore the z completely, and
# reconstruct the input, therefore when we try to generate 
# something using sampling it will be garbage! cuz they were
# not trained properly to have meaningful values)
# 
# skipcon is necessary for getting sharp/clear images, 
# without it we will get very blury images
# also the training will be more unstable. so for 
# more stable training and sharper reconstructions we 
# enable skipcon
use_skipconnection=False #True
# adding extra noise didnt do much for me, at least
# in my experiments, I had the most luck with other 
# techniques though
add_extra_noise=False #False
# with beta values larger than 0.01(like 1), using freebits 
# make training more stable, it makes images somewhat
# better, but not that much by itself only. 
# I still prefer beta=0.001 without freebits.
# and I need to enable skipcon regardless of this option
# when enabled using freebits, makes images a bit blurier
# especially with high min_kl values
use_freebits=True #False
# by using bce, and activiating freebits
# we get sharper image(0.4 initially seems sharper than 0.5)
min_kl=0.4 #0.5

# sidenote from past (before bn)
# for cifar10 these are the best settings so far
# we might face nans a few times(if we dont use bn),
# but try running and it will hopefully converge!
# the curcial things is to have larger featuremaps at the
# end of the encoder (4x4 in our case) and not using bn for
# last layer of encoder and first layer of decoder. 
# forget it, bn was not the issue, infact using bn 
# makes training more stable, and theres no issues
# in using it!
# embedding_size = 50
# beta=0.001 # beta=1 works, but the result is a bit blurier and less detailed. beta 0.001 gives the best details so far
# reduction='mean'
# use_mse=True
# normalize = True # for reduction='mean'
# kl_anealing=True # # it seems disabling klanealing makes training more stable with our current settings!
# use_skipconnection=True
# lr =0.002
# weight_decay = 1e-3
# scheduler_steps = [35,45,49]
#!use more embeddings withou skipcon?
# using bce gives us sharper images compared to mse!
# 
model = VAE(embedding_size, input_channel, use_skipconnection, add_extra_noise).to(device)

#0.01 when bn is used, 0.001/0.002 
# when bn is not used. (it works when bn is used as well)
# also using large betas (betas>1)
# will also make training unstable 
# and you need to lower lr further!(if no bn is used!)

# lastnote:
# using bce,lr=0.001,beta=0.0001, no skipcon, and
# no kl-annealing and spatial dim=4x4 we got much better
# result than mse. also our latent vectors are being used
# previously it was horrible, it wont use anything and mean/std was always 0-1
# with current settings we are learning but we still need improvement
# both in scheduler, embdsize and proper architecture. I noticed in my
# previous test, by mistake decoder was using 256x1x1, instead of 256x4x4
# which constrained the network. after fixing it we got good improvements
# next test can be using larger fmaps in decoder (instead of 4x4 lets go 8x8)
# (without changing encoders output dim!)
# then try using larger inputs, like 32/38/42/64 instead of 28! and see how that affects it
# then try using larger embeddings,
# or use -1/1 with mse and see if that helps
# or now use skipcon with bce and see if that works ths time with moving average trick!
# use larger batch instead of 32!
# update increasing the embdsz from 50 to 70 decreased our loss from 1370 to 1361 
# and images became sharper! embds=90 made it 1355 and images are more formed(and sharper)
# using embdsz=100, got us to 1353! but I decided to work on embds=90 and with change to 
# epochs=100 and scheduler (30,35,55,75) it got down to 1353! with [30,35] 
# its down to 1351, but I guess the ebmdssz=100 gives more detailed images (although with
# the same loss, at least I guess so!) with [30,50] its 1351 as well(loss/klloss) are more smooth!
# compared to [30,35]!
# with embdsz=120 and [30,50] we get 1345. jumping to embdsz=256 the loss stayed at 1345 but
# images now have much more details.
# now we increase beta to 0.001 and see how that affects it (loss becomes 1393!) and makes it(reconstructions) worse
# so beta remains at 0.0001(0.0002 is also a bit worse, so increasing beta is no brainer at this point).
# lowering it to 0.00001 is also not improving thins, it increases the loss initially to 3000 and then
# gradually decreases, but diverges quickly, shooting the loss to 10000! and then trying to lower it down
# basically it fluctuates badly and doesnt improve! ultimately the final loss is 13014! and the results
# are blury as hell)
# also disabling freebits ruins the results making them very blury
# so freebits(0.4) is absolutely necessary (with beta 0.0001). 
# freebits=0.8 lowers our loss to 1335! but I guess the image quality is so so, not very different
# than the 0.4 version one! but I might be wrong. trying freebits=0.2 gives us 1335 as well!
# the result isnot good (that is not better than 0.4) (sidenote, from time to time, the loss incresaed
# very high due to very high kl loss, but normally the start in 1400s! and decrease). ok
# freebits=0.4 also achieved 1336! so I guess higher minkls may get higher values afterall 
# (using embdsz=200, we get 1335/1338/1339 (*3 runs) by the way, images are sharp but not sharper than 256, 
# * 1335 was achieved after disabling bn for first layer of decoder
# 
# !edit move this part down
# but random generation doesnt produce good images yet (images are blurry and 
# interpolation is not smooth yet),but using mean/std we can replicate the samples!)
# sidenote( I guess its because our kl weight is too small!
# cuz the reconstructions look okay on real data but random samples from 
# the prior (normal distribution we sample from) look blurry or nonsensical,
# it usually means the learned latent space is not well aligned with 
# the standard normal distribution we sample from. 
# In other words, the vae can reconstruct images by relying on learned 
# posterior distributions for the training data, but unconditional generation
# (using samples from N(0,I)) does not match how the models encoder actually 
# encodes real images)
# 
#
# using embdsz=512 we get a loss=1332, the images are sharper, but not by a lot, lets do [30,50,50]
# and see if it improves further,(with increasing embds im seeing more diverging, loss shoots up at the
# very begining , and I have to restart trainig so it starts from a good place (usually restarting training fixes it)
# ok with new schedules, we got 1333 and results are abit blurry I think!
# trying embds=400, we get a loss=1334, the results are like before. 
# 
# trying embdsz=384 we got 1334 the quality is a bit better( a second try its 1333)
# with beta=0.001 and we got 1335! the quality is not that different!(though its blurier! but still not bad! pretty legible!)
# at this point I guess its enough, more effort can be put and make the results improve
# we covered the principles and main factors and saw their effects.
# the random generation is not ideal, but we can see images formed, although they are heavily
# blured, and its as if we are looking into old, moldy image frames from 1800s! Iguess trainig
# longer should fix this
# 
# before we finish lets talk about random generation issues
# i noticed we get somewhat dark with visible checkerboard patterns
# they are not visible in reconstructed image, just in randomly generated ones
# it might be from the transposed conv (Deconv) because it introduces checkerboard patterns
# because of the way different kernels/paddings/strides are used. to avoid it we can use
# conv2d-upsample combo, or use pixelshuffle. 
# test this as well: 
# also not using batchnorm in decoder might help. I guess we first try no bn in decoder
# and see if that works.(ok it was bn for the decoders first layer! removing it we now only
# get a green overlay on images! )
# we can also try mse and skipcon at the very end as well
# ok I removed bn from first layer of decoder! with embdsz=200 and beta=0.0001 lets see 
# how random generation and recnstruction looks- the loss decreased and the weird blackness
# is also gone.but theres a green hue everywhere! maybe its the right track?
# use embdsz=384 and try random generation for seeing if our changes work, thisway we make sure
# recons are sharp, and network learns a good latent, instead of using already blurry 200embds 
# version (user larger betas? check to make latent space well formed? separated)
# !add classification loss to the bunch and see if that helps in separating things!?
# ! also edit the above
# !see info from chatgpt
# starting with embdsz=384 and no bn in decoder: it made loss to shoot out to 400k!
# the reconstructions are expectedly very blurry, the generated images however, look more
# colorful, but very very rough, you can see the images, but they are heavily distorted with noise
# and discoloration.
# next im going to enbale some bns for middle layers in decoder 
# to stabalize loss a bit: with the last three two layers (except the last layer) of decoder
# with bn=true, the loss seems to be back to normal range we got 1340, larger than our previous
# 1333, the reconstructions are better now, but not as good as all bn version obviously, the
# generation is still not good, but seems much less crazy! but still not clear or legible
# note for bigger embdsz like 400+ we need to bn for all layers (except the last layer)
# otherwise, loss will shoot to 20K-200k+!
# 
# i noticed we have been using leakyrely with decoder, I changed to relu with all bns and see
# how it does: we got 1338, so leaky relue seems better in decoder!(1333 vs 1338)
# using conv-upsample: with all bns enabled, we got a loss=1331! (with beta=0.001 its 1333)
# which is better than default deconv, but the reconstructions seem more blury than before (dconv version)
# also the generation didnt change from before! still greenish, blury noisy images that i cant
# make anything out of them really unless im paying a lot of attention to make out a shape!
#
# im using embds=400 instead of 384 from now on, because its close to 384 and also multiple of 200
# which we can easily use with our visualizations. the generation with all bns are very blurry
# and lots of black spots, but theres no green overlay! 
# 
# disabling bn for last 3 layers od decoder: this made reconstructions less detailed, more blurry
# it also worsened the generation, I cant no longer identify anything, they are brown/blackish blurry
# blobs now. I guess bn for later layers helps a lot lets enable them back!
# 
# disable only the second layer, all others except the last layer have bn=True: loss=1334, generation is aweful, checkermarks
# are very apparent, details are illegible, its basically blurry blobs, its almost like tghe previous test, its as bad
#  maybe a bit less, but with more checkermarks, and completely useless, so the first layers are
# important!
# using upconv again this time with all bns enabled except last layer: we got 1333, reconstruction
# isnt any different than deconv, and the generation is brown blurry blobs(probably because batchnorm
# steers/biased towards the average color/brightness in out dataset, hence the darker/brownish color I guess),
# so we use deconv lets use pixelshuffle! it doesnt work, it crashes the cuda!
# 
# removing drp from first layer of decoder: embds=400,got loss=1331, more details in reconstructions
# but random generations are still like before
# 
# use (-1,1) and tanh and horizontalflip : cudaerror! its getting ridiculous! torch 2.5 is buggy as hell!
# use mse -skipped because of cuda-device-side-error (after a few kernel resets, it started working!)
# use 0-1,sigmoid and horizontalflip: loss got 1331, but reconstruction is roughly the same
# as without using horizionalflip.the random sampling is checkermarked and blurry blobs like before
# using mse, doesnt change anything , the loss magnitude decreases, but the overall its the same
# range(-1,1) mse and tanh: loss:34, everything else seems like before
# 
# adding more convs between deconvs didnt change anything loss:1332, and aside from that nothing
# changed!
# use groupnorm(1,outdim) insteadof batchnorm in deconvs: loss is 1332, reconstructions blurrier than when using bn
# the random generation however, is now a colorful mess! there are lots of vibrant colored blobs
# I cant detect any images, seems like pure blurry bloby mess to me!
# use groupnorm(4,outdim) insteadof batchnorm in deconvs:1337 still the same,
# use groupnorm(outdim,outdim) instead of batchnorm for all layers: loss doesnt decrease!
#
# revert back to deconv,sigmoid,(0-1),embds=400, now this time remove dropout
# after mu in encoder: the loss is down to 1323! the images are now much sharper(nearly prefect I'd say?! details are visible unlike before)!
# silly me completey forgot about removing it! however, the generations are still blurry messes
# they are too blurry infact!
# using crossentropy loss with other losses didnt change much, got 1325! it made reconst blurrier
# so its not good.
#
# using beta=0.001 its roughly the same, the loss is 1326, the colors are a bit fader compared to 
# beta=0.0001, the generation is now shows a lot more checkermarks, black checkermarks. 
# still very blurry, with a tiny faint touch of images, its really faint, but if I look closely
# I can see its an image, a heavily, blured, noisy image, still cant properly tell what they all
# maybe a plane? its obviously somewhat more detailed than the previous beta(0.0001).
# 
# using beta=0.01 loss=1334(kl loss is now in 1300s),recons-images are considerably blurrier, no significant change in latent space
# is visible,(i.e. no visible improvements in formation of subspaces) but random generations
# now show more refined images/still heavily blurred with chekermarks, but its clear they 
# are images, you can tell they are heavily noisy/distorted images.
# 
# using beta=0.1 ,loss=1368 (kl loss is now in 200s, its inversly related to beta value, smaller beta means larger kl loss!)
# expectedly image recons are getting blurrier, but at the same time, random
# generations are getting better, now I can see blurry but colorful, images, no checkermarks or
# black bars in the images are visible. the images are very blurry though
# 
# using beta=1 (might be time to use klanealing so image recons is not affected that much!): loss
# 1536(kl loss 166 nearly intact the wholetimme), it decreased from 1900s, but didnt go down 
# much. the img recons are very very blurry!its very bad, some are not even formed properly.random generations
# is got better as well, but images like recons are still blurry, but they are better formed,with
# vibrant colors
# 
# trying beta=1, klanealing:  added new beta schedule instead of linear one which used too high
# values! now with beta=1, klanealing loss is 1441 (klloss 443) and it seems we have over fitted
# image recons is awful! expectly, many images are not even formed properly and the whole image is
# is very blurry. random regeneation is not good either, cant say its better than previous case!
# 
# using beta=0.5, klannealing: loss is 1394 (kl loss is 618). we overfitted half way!
# and mean got close to 0 and std close to 1 which is not good, it needs more regularization I guess
# as for the outputs, the image reconsts is much better than previous test(beta=1), but as expected
# due to higher beta at the end of training, images are blurry and less detailed, but much detailed
# and less blurry than previous case!(you get the idea) so slight segmentations can be noticed
# in latent space, not much, but its defnitely there).random generations are better than before
# some images clearly show the objects, though very blurry but its like the img recons quality
# while others are more jumbled!
# 
# using beta=0.5, kl anealing, usin midpoint=0.85 ineats of 0.5:loss is 1349(kl loss is5048)
# the mean is close to 0, and we see sign of overfitting, the image recon is much better than
# before, (before they looked like smudged abd blurry!) but this has structures, and details
# but obviously its very blurry, and images are not formed completely, and they lack details
# but still better than before. the generation is not good either, cant say its better than before
# it seems we need to hit a balance between kl and recon using
# precise annealing to get a good output
# using beta=0.5, kl anealing, usin midpoint=0.70:loss=1368(kl loss=1450), we got overfitted badly
# mean is nearly 0 which is bad! image recons is bad as well, blurry, smudged, very rough, not detailed!
# this is not good for us
# 
# using beta=0.3 no kl annealing: loss 1414 (klloss=192), imag recons are blurry, colors are not
# accurate, and shapes are not formed properly, but is worse than previous one!random generation is also 
# not good and very blurry!but some images seem well-formed, but very very blurry! (i can spot
# a car, a dogs head, a cat, a horse, but its exretemely blurry!) ran again, its pretty much the same thing!
# mean is near zero, and std is around 0.7, both of which show not ideal situation(nearning collapse).
#
# using beta=0.3, klannealing(midpoint=0.5,k=15): loss is 1373(klloss=677 ), mean=0.std=0.4
# we have overfitted it seems again, the image recons though still very blurry, is better than
# before, we still have not formed images,very blurry images, but compared to previous, its better
# random generation is not that different from before. but i could see interpolations improved
# it seems we just need to play with hyperparameters at this point! which means Im fine 
# ive already put a lot of effort into this, so we can do parameter tuning nexttiem
# 
# using beta=0.1 klannealing(midpoint=0.5,k=15):loss is 1347(klloss=1074),image recons is obviously
# way better as we used a smaller beta, still blurry, but way better,random generation is a bit
# chaotic, you can spot the object, like horse, car, but they are clutered blurry images
# using beta=0.2 klannealing(midpoint=0.5,k=15):loss is 1361(klloss=823), image recons are a bit worse
# than before, but id say acceptable given what we haveseen so far, the random generation 
# is better than before, images are less clutered and can be spotted more easily (though still
# blurry and need a long way to get better. but better than before). i stop here its good enugh!
# vae dont create sharp images like that! we need to us emore powerful variants!
# 
# try with embds=400, disabled drpout after logvar: (forgot to remove it when
# I removed mu dropout!): loss:1356(kllos:677) img recons details are overall better imho
# generations are much better now, still a long way to crisp images. however, the interpolation
# and random generation youcan more easily see the object, for example in interpolation we can
# see a car, morphing into another car, its blurry and rough but its there and its much 
# clearer than before, using higher embdsz(like 600) can give betetr result, for exampe
# with embds=600 the loss got 1364, but the generation and interpolation was good (like e.g. in -1,1 range)
# i conclude this here, we can improve this using other techniques we talked about like heirarchical latent variables
# but I dont want to spend too much time on it , as we can get much better results using other methods
#  
# for future refrence, I first started with embdsz=50 and everythin set to False
# except normalize, then tried with mse with basically every options, it only worked
# with skipcon enabled and beta=0.001 Iguess. I got near prefect reconstruction but
# the latent space was smushed into a tiny little circle around 0, and I couldnt interpolate
# or create meaningful reconstructions using sampling. one because we have to use something
# to get encoding output for our decoder becasue of skipcon(the latent vector was conditioned on it)
# I tried moving average of all encoders outputs to use duirng test time, it didnt work, if I used
# any image and use its encodings in decoder with a random latent vector z, it would only create 
# the said image, if I used a mean of a batch of images, I would get back the mean. basically the
# encodings would dictate the outcome, skipcon, had made latent representation to be ignored completely
# and hence it couldnt play any role. thats why I switched to bce this time. upon switching to bce
# I got way more clearer images than mse. and from there continued and enabled freebits and noticed
# the images got much better and clearer, playing with lr a bit and a bit of schedules, I got good results quickly
# from there when improvements stopped, I increased the embdsz and got great results. noticed the loss
#persay doesnt tell exactly if the output is good, it rouhghly saysso a loss of 1300 has a very good
# reconstruction.  also increasing the spatial dim both in encoder and decoder (both start from 4x4)
# positively affects the reconstruction and loss. I also made a mistake once, encoder output was 4x4
# but decoder started at 256x1x1, it adversly affected the result, when i made it 256x4x4 (everything else intact)
# it got good result, showing decoder starting with larger fmaps helps as well.
# over all we can improve this a lot hopefully


lr =0.001#0.001 0.002
weight_decay = 1e-3
scheduler_steps = [30,50]#,55,75]#[20,45,65,85] # [20,35,45,49]
optimizer = torch.optim.Adam(model.parameters(), lr =lr, weight_decay=weight_decay)#1e-4
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, scheduler_steps)

train(model, dataloader_train, optimizer=optimizer, 
      scheduler=scheduler,
      device=device,
      epochs=epochs, 
      beta=beta,
      reduction=reduction,
      normalize=normalize,
      use_mse=use_mse,
      interval=interval,
      kl_anealing=kl_anealing,
      use_freebits=use_freebits,
      min_kl=min_kl)

kwargs = {"states": model.state_dict(),
          "epochs": epochs,
          "embedding_size":model.embedding_size,
          "use_skipconnection":use_skipconnection,
          "beta":beta,
          "kl_anealing":kl_anealing,
          "use_freebits":use_freebits,
          "min_kl":min_kl,
          "reduction":reduction,
          "normalize":normalize,
          "use_mse":reduction,
          "optimizer":optimizer.state_dict(),
          "scheduler":scheduler.state_dict()}

timestamp = datetime.datetime.now().strftime("%H_%M_%S_%Y_%m_%d")
modelname = f"vae_{"cifar10" if input_channel==3 else "mnist"}_{model.embedding_size}_{reduction}_{'normalized' if normalize else 'not-normalized'}_{'mse' if use_mse else 'bce'}_{timestamp}.pth"
save_model(modelname=modelname, kwargs=kwargs)
#%%
timestamp2 = timestamp
print(f'{timestamp2=}')
# modelname='vae_cifar10_400_mean_normalized_bce_11_53_36_2025_02_17.pth'
# save_model(modelname=modelname, kwargs=kwargs)
# load the model to make sure we are dealing with the right model!
load_model(model, modelname=modelname)
img_shape=(input_channel,28,28)
kwargs = {"img_shape":img_shape,
          "beta":beta,
          "reduction":reduction,
          "use_mse":reduction,
          "use_freebits":use_freebits,
          "min_kl":min_kl,
          "kl_anealing":kl_anealing,
          "normalize":normalize}
check_latent_representation_diversity(model, dataloader_train)
plot_latent_space_encodings(model)
evaluate_on_testset(model, dataloader_test, **kwargs)
plot_embedding_clusters(model, dataloader_train, title='Encoder embedding',use_pca=False)
plot_latentspace_clusters(model, dataloader_train, title='Full latent clusters',use_pca=False)
# fix these two for skipcon version
if not model.use_skip_con:
    # check_laten_representation_interpolation(model, dataloader_train, interpolation_steps=10)#check5,10,20
    generate_random_images(model, count=32,img_shape=img_shape)
    # imgs,labels = next(iter(dataloader_train))
    # view_images(imgs,labels,rows=12,cols=11)
    # imgs = imgs.to(device)
    # img_t = imgs[0].unsqueeze(0)
#%%
# Use multiple of embdsz to get better output for exampe for embds=600 we can use 60/120etc
# for 400 its 20,40,etc, play with range as well, higher number(-10/10) may make the output worse
# use different ranges to inspect the output (-1,1,-2,2,-0.9,0.9 etc)
    generate_latent_space_grid(model,n=40,lower_bound=-2,upper_bound=2,img_shape=img_shape,img=None)
    # shows a fade horse at the end,still blurry need to focus to spot it
    generate_latent_space_grid(model,n=40,lower_bound=-1,upper_bound=1,img_shape=img_shape,img=None)
    generate_latent_space_grid(model,n=120,lower_bound=-1,upper_bound=1,img_shape=img_shape,img=None)
    generate_latent_space_grid(model,n=120,lower_bound=-0.9,upper_bound=0.9,img_shape=img_shape,img=None)

#%%
    # lets view some images and generate
    # some only for a specific class
    # note that our current approach only
    # gives us little control over this kind of
    # generation, in order to get diverse output
    # even for the same class, we need to put an effort!
    imgs,labels = next(iter(dataloader_test))
    view_images(imgs,labels)
    # mu/std slightly changes for different instance of a class
    # but overall they are roughly the same, they
    # however change dirastically from class to class
    generate_similar_images(model, imgs[3])#0
    generate_similar_images(model, imgs[10])#0
    generate_similar_images(model, imgs[21])#0
    generate_similar_images(model, imgs[6])#1
    generate_similar_images(model, imgs[9])#1
    # generate_similar_images(model, imgs[14])
    generate_similar_images(model, imgs[25])#2
    generate_similar_images(model, imgs[0])#3
    generate_similar_images(model, imgs[22])#4
    generate_similar_images(model, imgs[12])#5
    generate_similar_images(model, imgs[4])#6
    generate_similar_images(model, imgs[13])#7
    generate_similar_images(model, imgs[1])#8
    generate_similar_images(model, imgs[23])#9

    # the animation maynot work inside jupyternotebook, but the gif file works
    create_interpolation_animation(model, filename=f'cifar10_{timestamp}',mu=0.2,std=0.01)
#%%
# save the model
# timestamp = datetime.datetime.now().strftime("%H_%M_%S")
# modelname = f"vae_{"cifar10" if input_channel==3 else "mnist"}_{model.embedding_size}_{reduction}_{'normalized' if normalize else 'not-normalized'}_{'mse' if use_mse else 'bce'}_{timestamp}.pth"
# torch.save({"states": model.state_dict(),
#             "epochs": epochs,
#             "embedding_size":model.embedding_size,
#             "use_skipconnection":use_skipconnection,
#             "beta":beta,
#             "kl_anealing":kl_anealing,
#             "use_freebits":use_freebits,
#             "min_kl":min_kl,
#             "reduction":reduction,
#             "normalize":normalize,
#             "use_mse":reduction,
#             "optimizer":optimizer.state_dict(),
#             "scheduler":scheduler.state_dict()},
#             modelname)
# print('model saved!')
#%%
# load the model 
# states = torch.load(modelname)
# model.load_state_dict(state_dict=states['states'])
# print('weights loaded')

# img_shape=(input_channel,28,28)
# check_latent_representation_diversity(model, dataloader_train)
# # fix these two for skipcon version
# # check_laten_representation_interpolation(model, dataloader_train, interpolation_steps=10)#check5,10,20
# # generate_random_images(model, count=32,img_shape=img_shape)
# #todo use a kwargs for easier manipulation!
# evaluate_on_testset(model, dataloader_test, img_shape=img_shape, beta=beta, reduction=reduction,use_mse=use_mse,use_freebits=use_freebits,min_kl=min_kl,normalize=normalize)
# generate_latent_space_grid(model,n=10,lower_bound=-2,upper_bound=2,img_shape=img_shape,dataloader=dataloader_train)
# generate_latent_space_grid(model,n=20,lower_bound=-2,upper_bound=2,img_shape=img_shape,dataloader=dataloader_train)
# plot_latent_space_encodings(model)
# plot_embedding_clusters(model, dataloader_train, title='Encoder embedding',use_pca=False)
# plot_latentspace_clusters(model, dataloader_train, title='Full latent clusters',use_pca=False)
# wont work with skipconnection=True, todo: fix it

# cifar10 loss
# Files already downloaded and verified
# Files already downloaded and verified
# Date:            09:28:27 - 2025/02/10
# Dataset:         CIFAR10
# Epochs:          50
# embedding_size:  50
# use_skip_con:    True
# add_extra_noise: False
# beta:            0.001
# reduction:       mean
# normalize:       True
# use_mse:         True
# kl_anealing:     False
# use_freebits:    False
# min_kl:          0.4
# optimizer:       Adam (
# Parameter Group 0
#     amsgrad: False
#     betas: (0.9, 0.999)
#     capturable: False
#     differentiable: False
#     eps: 1e-08
#     foreach: None
#     fused: None
#     initial_lr: 0.002
#     lr: 0.002
#     maximize: False
#     weight_decay: 0.001
# )
# scheduler:       Counter({15: 1, 25: 1, 35: 1, 45: 1})
# interval:        2000
# Epoch 0/50 [0/391] | Loss: 184.5746 | KL-Loss: 6.4882 | (μ,σ): (0.0427 , 1.0200) | lr: 0.002
# /home/hossein/miniconda3/lib/python3.12/site-packages/torch/optim/lr_scheduler.py:595: UserWarning: To get the last learning rate computed by the scheduler, please use `get_last_lr()`.
#   _warn_get_lr_called_within_step(self)
# Epoch 1/50 [0/391] | Loss: 40.8017 | KL-Loss: 276.7021 | (μ,σ): (-0.0077 , 0.4125) | lr: 0.002
# Epoch 2/50 [0/391] | Loss: 29.7259 | KL-Loss: 157.6318 | (μ,σ): (-0.0066 , 0.3913) | lr: 0.002
# Epoch 3/50 [0/391] | Loss: 24.4113 | KL-Loss: 116.6159 | (μ,σ): (-0.0056 , 0.3828) | lr: 0.002
# Epoch 4/50 [0/391] | Loss: 21.1453 | KL-Loss: 96.5729 | (μ,σ): (-0.0042 , 0.3716) | lr: 0.002
# Epoch 5/50 [0/391] | Loss: 18.8720 | KL-Loss: 85.3616 | (μ,σ): (-0.0030 , 0.3576) | lr: 0.002
# Epoch 6/50 [0/391] | Loss: 17.2047 | KL-Loss: 78.9367 | (μ,σ): (-0.0024 , 0.3417) | lr: 0.002
# Epoch 7/50 [0/391] | Loss: 15.9369 | KL-Loss: 75.1898 | (μ,σ): (-0.0019 , 0.3260) | lr: 0.002
# Epoch 8/50 [0/391] | Loss: 14.9096 | KL-Loss: 72.9856 | (μ,σ): (-0.0018 , 0.3114) | lr: 0.002
# Epoch 9/50 [0/391] | Loss: 14.0579 | KL-Loss: 71.6381 | (μ,σ): (-0.0015 , 0.2984) | lr: 0.002
# Epoch 10/50 [0/391] | Loss: 13.3564 | KL-Loss: 70.9018 | (μ,σ): (-0.0013 , 0.2866) | lr: 0.002
# Epoch 11/50 [0/391] | Loss: 12.7576 | KL-Loss: 70.5410 | (μ,σ): (-0.0012 , 0.2761) | lr: 0.002
# Epoch 12/50 [0/391] | Loss: 12.2373 | KL-Loss: 70.3561 | (μ,σ): (-0.0012 , 0.2670) | lr: 0.002
# Epoch 13/50 [0/391] | Loss: 11.7811 | KL-Loss: 70.2461 | (μ,σ): (-0.0011 , 0.2591) | lr: 0.002
# Epoch 14/50 [0/391] | Loss: 11.3775 | KL-Loss: 70.1804 | (μ,σ): (-0.0010 , 0.2521) | lr: 0.002
# Epoch 15/50 [0/391] | Loss: 11.0276 | KL-Loss: 70.2354 | (μ,σ): (-0.0009 , 0.2458) | lr: 2e-05
# Epoch 16/50 [0/391] | Loss: 10.6327 | KL-Loss: 68.8279 | (μ,σ): (-0.0008 , 0.2460) | lr: 0.0002
# Epoch 17/50 [0/391] | Loss: 10.2681 | KL-Loss: 66.5190 | (μ,σ): (-0.0008 , 0.2529) | lr: 0.0002
# Epoch 18/50 [0/391] | Loss: 9.9409 | KL-Loss: 64.2191 | (μ,σ): (-0.0007 , 0.2611) | lr: 0.0002
# Epoch 19/50 [0/391] | Loss: 9.6455 | KL-Loss: 62.0366 | (μ,σ): (-0.0007 , 0.2697) | lr: 0.0002
# Epoch 20/50 [0/391] | Loss: 9.3726 | KL-Loss: 59.9974 | (μ,σ): (-0.0006 , 0.2782) | lr: 0.0002
# Epoch 21/50 [0/391] | Loss: 9.1224 | KL-Loss: 58.1055 | (μ,σ): (-0.0006 , 0.2864) | lr: 0.0002
# Epoch 22/50 [0/391] | Loss: 8.8944 | KL-Loss: 56.3752 | (μ,σ): (-0.0006 , 0.2940) | lr: 0.0002
# Epoch 23/50 [0/391] | Loss: 8.6854 | KL-Loss: 54.7983 | (μ,σ): (-0.0005 , 0.3009) | lr: 0.0002
# Epoch 24/50 [0/391] | Loss: 8.4898 | KL-Loss: 53.3354 | (μ,σ): (-0.0005 , 0.3074) | lr: 0.0002
# Epoch 25/50 [0/391] | Loss: 8.3102 | KL-Loss: 51.9912 | (μ,σ): (-0.0005 , 0.3133) | lr: 2.0000000000000003e-06
# Epoch 26/50 [0/391] | Loss: 8.1346 | KL-Loss: 50.6132 | (μ,σ): (-0.0005 , 0.3205) | lr: 2e-05
# Epoch 27/50 [0/391] | Loss: 7.9700 | KL-Loss: 49.1474 | (μ,σ): (-0.0005 , 0.3299) | lr: 2e-05
# Epoch 28/50 [0/391] | Loss: 7.8157 | KL-Loss: 47.7302 | (μ,σ): (-0.0004 , 0.3397) | lr: 2e-05
# Epoch 29/50 [0/391] | Loss: 7.6717 | KL-Loss: 46.3787 | (μ,σ): (-0.0004 , 0.3494) | lr: 2e-05
# Epoch 30/50 [0/391] | Loss: 7.5377 | KL-Loss: 45.0928 | (μ,σ): (-0.0004 , 0.3590) | lr: 2e-05
# Epoch 31/50 [0/391] | Loss: 7.4114 | KL-Loss: 43.8685 | (μ,σ): (-0.0004 , 0.3684) | lr: 2e-05
# Epoch 32/50 [0/391] | Loss: 7.2929 | KL-Loss: 42.7034 | (μ,σ): (-0.0004 , 0.3777) | lr: 2e-05
# Epoch 33/50 [0/391] | Loss: 7.1814 | KL-Loss: 41.5948 | (μ,σ): (-0.0004 , 0.3867) | lr: 2e-05
# Epoch 34/50 [0/391] | Loss: 7.0755 | KL-Loss: 40.5385 | (μ,σ): (-0.0004 , 0.3955) | lr: 2e-05
# Epoch 35/50 [0/391] | Loss: 6.9756 | KL-Loss: 39.5326 | (μ,σ): (-0.0004 , 0.4040) | lr: 2.0000000000000004e-07
# Epoch 36/50 [0/391] | Loss: 6.8801 | KL-Loss: 38.5736 | (μ,σ): (-0.0003 , 0.4124) | lr: 2.0000000000000003e-06
# Epoch 37/50 [0/391] | Loss: 6.7904 | KL-Loss: 37.6644 | (μ,σ): (-0.0003 , 0.4203) | lr: 2.0000000000000003e-06
# Epoch 38/50 [0/391] | Loss: 6.7048 | KL-Loss: 36.7987 | (μ,σ): (-0.0003 , 0.4279) | lr: 2.0000000000000003e-06
# Epoch 39/50 [0/391] | Loss: 6.6244 | KL-Loss: 35.9735 | (μ,σ): (-0.0003 , 0.4353) | lr: 2.0000000000000003e-06
# Epoch 40/50 [0/391] | Loss: 6.5475 | KL-Loss: 35.1865 | (μ,σ): (-0.0003 , 0.4423) | lr: 2.0000000000000003e-06
# Epoch 41/50 [0/391] | Loss: 6.4740 | KL-Loss: 34.4351 | (μ,σ): (-0.0003 , 0.4491) | lr: 2.0000000000000003e-06
# Epoch 42/50 [0/391] | Loss: 6.4047 | KL-Loss: 33.7182 | (μ,σ): (-0.0003 , 0.4557) | lr: 2.0000000000000003e-06
# Epoch 43/50 [0/391] | Loss: 6.3379 | KL-Loss: 33.0329 | (μ,σ): (-0.0003 , 0.4619) | lr: 2.0000000000000003e-06
# Epoch 44/50 [0/391] | Loss: 6.2739 | KL-Loss: 32.3772 | (μ,σ): (-0.0003 , 0.4680) | lr: 2.0000000000000003e-06
# Epoch 45/50 [0/391] | Loss: 6.2129 | KL-Loss: 31.7499 | (μ,σ): (-0.0003 , 0.4737) | lr: 2.0000000000000007e-08
# Epoch 46/50 [0/391] | Loss: 6.1540 | KL-Loss: 31.1492 | (μ,σ): (-0.0003 , 0.4793) | lr: 2.0000000000000004e-07
# Epoch 47/50 [0/391] | Loss: 6.0986 | KL-Loss: 30.5740 | (μ,σ): (-0.0003 , 0.4846) | lr: 2.0000000000000004e-07
# Epoch 48/50 [0/391] | Loss: 6.0450 | KL-Loss: 30.0226 | (μ,σ): (-0.0003 , 0.4897) | lr: 2.0000000000000004e-07
# Epoch 49/50 [0/391] | Loss: 5.9936 | KL-Loss: 29.4936 | (μ,σ): (-0.0003 , 0.4946) | lr: 2.0000000000000004e-07
#%%
#! make two segments, one for mnist test
#! and another for cifar10, so both results can be seen one after another
# ok, now we got both mnist and cifar to work, for getting sharper outputs we need
# a better model/training regime, but for our case it suffices
# thankfully, we could replicate all scanrios and see how each issue could be solved
# some issues wouldnt happen in simple datasets such as mnist, but when we used cifar10
# we could clearly see the output and their effectiveness.
# we can test with different embeddingsizes with larger epoch and lower epoch 
# (with anealing and without, so the effect of epoch shows itself
# (basically gradual decrease shows its potential when properly used not in small epochs (we could also use batches!))
# withskipconnection and without
# and show that simply one metric (like mean) doesnt show the full extend of the issue
# and using several clues make it much easier to know whats wrong!
# compare the visualizations, should give very good intuitions
# check decoder with huge dropouts to simulate weaker version
# use noise if images are similar
# recap the info - with solutions for each issue 

#recap
# ok lets quickly recap what changes we included this time and why:
# we said there are several issues that can cuz a posterior collapse
# one sign was overly generic images or blury ones. 
# 
# another clear sign is an extremely low kl term.
# if kl loss is close to zero, its a sign of collapse.  
# (if it starts high and drops to near zero, its probably
# a collapse, because means its too easy for the model to set q(z|x)
# close to p(z)) a nonzero value is what we want during training.
# freebits regularization would help because it makes sure 
# each dimension is at least doing something!(contributing positively and
# latent space actually does have some information!)
# note that a large beta is not bad per say, its just that too large of 
# a value especially at the begining hinders the model learning.
# a properly large beta can force more structured latent space
# and lead to meaningful separation in the latent space as well
# (it makes the kl loss larger, the recostruction loss needs to do
# a better job at separation to lower the loss otherwise everything goes south fast!).
#
# the third sign is, if the latent encodings standard deviation gets 
# nearly zero, it means the model isnt using its 
# latent space effectively, 
# we should have a diverse range of latent activations
# if the mean is always near 0 and std is near 1, 
# it means the model ignores the latent space(its using the p(z) only!).
# 
# forth, we can check the encodings and see if the
# latent encodings are almost identical(for different inputs)
# or not, if they are, then it means the model isnt
# using the latent space.
# (simply encode two different images 
# and compare their latent encodings,
# if the difference is close to 0, the latent space is collapsing
# there should be noticeable variation between different images)
#
# fifth, we can visualizing the latent space with t-sne(pca is not good, its linear and wont work properly for nonlinear relationships which is ourcase)
# if all the points cluster together, it's collapsed,
# (a good latent space should separate different categories, otherwise reconstruction shows how bad it is)
#
# we can check the generated samples and tell if osmething is wrong!
# if the generated images are nearly identical, regardless of input changes,
# its a sign that the latent space is underutilized.
# if all images look the same, posterior collapse is likely happening.  
# a good VAE should generate diverse samples
#(we can check how reconstruction changes with latent space,
# a properly trained VAE should smoothly interpolate between 
# different points in latent space.not being able to do this means,
# theres something wrong, depending on the severity, it could be a collapsed posterior,
# or simply a bad training regime (needs more trainig, inefficent model, etc (well talk more about this))
# if interpolation doesnt produce meaningful transitions, the latent space
# isnt being used effectively.)
#

# recap of our recap!
# why do we face posterior collapse? it happens when the encoder ignores the latent space 
# and learns a simplestic/trivial distribution, making the decoder reconstruct only from noise. 
# kl loss must not be close to 0 , it should be balanced (have nonzero values)
# the variance must not be close to 0, we should have non-zero variance
# different encodings must not be (nearly) identical, all encodings must be distinc
# generated images  must not be identical, obviously we must have diverse generations/outputs
# when using t-sne the latent space must not have a single cluster,  we must see  well-separated clusters
# when interpolating we must not see abrupt/sudden/weird/unmeaningful changes, we must see smooth transitions from one class into another
# 
# as we saw in our experiments, detecting posterior collapse often 
# requires checking kl term value, the latent space variance, 
# generated outputs and interpolation behavior.
# The best way to avoid posterior collapse is to carefully tune the kl loss, 
# use beta scaler, with kl anealing and avoid an overly powerful decoder(or a simple encoder!),
# other techniques such as freebits regularization, noise addtion, skipconnection come next.


#%%
# lets see what each class'es mean/std looks like
# each have their own different mean ,
# the mean is drastically different than other classes though otherwise it shows
# the model has not been trained properly
# we can use this to generate as many images we want for each class
# we can create as many 0s, 1s or any classes we want! using their mean/std
# #! edit make samples more varied by altering std a bit
# @torch.no_grad()
# def generate_similar_images(model:VAE, input_img:torch.Tensor, count:int=64, rows:int=8):
    
#     if len(input_img.shape) == 3:
#         input_img = input_img.unsqueeze(0)

#     # grab the device from our model parameters
#     device = next(model.parameters()).device
#     model.eval()

#     input_img = input_img.to(device)
#     # grab the mu/logvar for the image class
#     z0, enc_outputs, mu, logvar = model.encode(input_img)
#     # convert the logvariance to std
#     std = torch.exp(0.5*logvar)
#     # create latent vectorz by sampling using the mu/std
#     # using random noise(epsillon) to create randomness in output
#     epsillon = torch.randn_like(std)
#     z_random = mu + epsillon * std
#     # how mu+std sample looks like
#     z_plain = mu + std
#     # instead of a single epsilon, we can create as many as
#     # we like, and therefore generate as many images. just
#     # make sure the size matches
#     epsillons = torch.randn(size=(count, mu.shape[-1]), device=device)
#     # by changing the std, we can generate slightly different variatations
#     # a higher std introduces more randomness, leading to more diverse outputs,
#     # a lower value generates outputs closer to the mean(mu) which means less variation
#     # we can change this in steps and create a morphing effect
#     # from one image into another. (we will be implementing this in a moment)
#     # scaler = torch.linspace(0.01, 0.03, count).to(device).view(count,1)
#     # print(f'{scaler=}')
#     # epsillons *= scaler
#     # print(f'{epsillons=}')
#     z_batch = mu + epsillons * std

#     z_random, z_plain,z_batch = (z.to(device) for z in (z_random, z_plain, z_batch))
#     # generate images for each latent vector
#     img_random, img_plain, img_batch = (model.decoder(z) for z in (z_random,z_plain,z_batch))
#     # reshape the decoder outputs to the proper image dims
#     img_random, img_plain, img_batch = (img.view(-1,1,28,28) for img in (img_random, img_plain, img_batch))
#     # combine the images as one so we can display them as one big image
#     # imgs_combined = torch.concat([input_img,img_random,img_plain],dim=3)
#     imgs_combined = torch.dstack([input_img,img_random,img_plain])
#     # combine all images as one so we can better visualize and inspect them
#     img_batch_grid = make_grid(img_batch, nrow=rows, normalize=True)
    
#     mu = mu.cpu().numpy().flatten()
#     std = std.cpu().numpy().flatten()

#     plt.figure(figsize=(8, 4))#(12,8)
    
#     plt.subplot(2,3,1)
#     plt.plot(mu, label="Mean (μ)")
#     plt.title("Mean (μ)")
#     plt.xlabel("Latent dimension")
#     plt.ylabel("Value")
#     plt.legend()

#     plt.subplot(2,3,2)
#     plt.plot(std, label="Std (σ)", color="orange")
#     plt.title("Std (σ)")
#     plt.xlabel("Latent dimension")
#     plt.ylabel("Value")
#     plt.legend()
    
#     plt.subplot(2,3,4)
#     plt.imshow(imgs_combined.squeeze().cpu().numpy(), cmap="gray")
#     plt.title("Input image")
#     plt.axis("off")
        
#     plt.subplot(2,3,5)
#     plt.imshow(img_batch_grid.squeeze().cpu().numpy().transpose(1,2,0), cmap="gray")
#     plt.title("Similar images")
#     plt.axis("off")
        
#     plt.tight_layout()
#     plt.show()
    
# imgs,labels = next(iter(dataloader_test))
# view_images(imgs,labels)
# # mu/std slightly changes for different instance of a class
# # but overall they are roughly the same, they
# # however change dirastically from class to class
# generate_similar_images(model, imgs[3])#0
# generate_similar_images(model, imgs[13])
# generate_similar_images(model, imgs[25])
# generate_similar_images(model, imgs[2])#1
# generate_similar_images(model, imgs[5])
# generate_similar_images(model, imgs[14])
# generate_similar_images(model, imgs[1])#2
# generate_similar_images(model, imgs[32])#3
# generate_similar_images(model, imgs[4])#4
# generate_similar_images(model, imgs[15])#5
# generate_similar_images(model, imgs[11])#6
# generate_similar_images(model, imgs[0])#7
# generate_similar_images(model, imgs[8])#8
# generate_similar_images(model, imgs[7])#9

#%%
# # now lets generate new images by stepping through the latent space
# import matplotlib.animation as animation

# fig = plt.figure()
# ax = fig.add_subplot(111)
# z = torch.randn(size = (30, model.embedding_size)).to(device)
# model.eval()
# def animate(i):
#     # change the latent vector at each step so we get different image
#     # and ultimately a cool animation showing each image morphing into another!
#     # note that by choosing a larger std(0.03 vs 0.01), we increase the randomness
#     # so it changes faster. the more farther away from mean, the more different
#     # it becomes from that image
#     imgs = model.decoder(z*(i*0.02)+0.02)
#     imgs2 = imgs.view(imgs.size(0), 1, 28, 28)
#     new_img = make_grid(imgs2).cpu().detach().numpy().transpose(1,2,0)
#     ax.clear()
#     ax.imshow(new_img)

# anim = animation.FuncAnimation(fig, animate, frames=100, interval=300, repeat=True,repeat_delay=1000)
# # save the git using pillow
# anim.save('vis.gif', writer="pillow", fps=30)
# plt.show()
#%%





#%% 
# Conditional VAE 
# in a vanilla VAE, the generation process is stochastic, we sample from a latent distribution 
# (usually Gaussian), which means the output images are generated randomly without explicit 
# control although we could try to steer the generation (for example, by manipulating the 
# learned mu and std for each class, this approach is indirect and as we also saw is not precise.
# in this version, we are going to implement a Conditional Variational Autoencoder (CVAE),
# which allows us to generate images for a specific class or concept.
# The key advantage of a CVAE is that it allows us to incorporates conditional information 
# (like e.g. class labels) into both the encoder and the decoder, which enables more targeted 
# and controlled generation.
# usually labels are used as the conditional factor, but we're not limited to just that. 
# for example, we can use textual descriptions (e.g. "a red sports car") to generate images
# that match the detailed description, or we could even use another image as a condition to
# guide the style or content of the generated output. Other types of data/attributes 
# (such as color, texture, or any domain-specific features) can also be employed.
# the implementation is very simple, for our case, all we need to do is to encode the label
# and feed it to the encoder and decoder in the form of one_hot encoded array. this allows
# both of them to be conditioned on the specific label, and later on, generate data based on
# a given class.
# unlike the previous implementation lets keep this simple
# we now have a pretty good idea how to extend this if we want, 
# so theres no need to extra details for now. 
# lets see how its done
class VAE_Conditional(nn.Module):
    def __init__(self, embedding_size=2, num_classes = 10):
        super().__init__()
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        # encoder 
        self.fc1 = nn.Linear(28*28 + num_classes, 512)
        # we are actually adding the one_hot encoded length here. 
        self.fc_mu = nn.Linear(512, embedding_size )
        self.fc_logvar = nn.Linear(512, embedding_size)
        
        # decoder 
        # our decoder also uses our conditional factor along side the embedding
        # so it has embedding_size + condition dims and differs with the last 
        # layer of the encoder output dim
        self.decoder = nn.Sequential(nn.Linear(embedding_size + num_classes, 512),
                                    nn.ReLU(), 
                                    nn.Linear(512 , 28*28),
                                    nn.Sigmoid())

    def encode(self, x, y):
        x = x.view(x.size(0), -1)
        # y is a one hot encoded vector which we concat with our input
        # y is used as the conditioning factor!
        inputs = torch.cat((x,y),dim=1)
        output = F.relu(self.fc1(inputs))
        mu = self.fc_mu(output) 
        logvar = self.fc_logvar(output)
        z = self.reparametrization_trick(mu, logvar)
        return z, mu, logvar

    def decode(self, z, y):
        z_cond = torch.cat((z,y), dim=1)
        output = self.decoder(z_cond)
        output = output.view(z.size(0), 1, 28, 28)
        return output

    def reparametrization_trick(self, mu, logvar):
        std = torch.exp(logvar * 0.5)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, input, y):
        z, mu, logvar = self.encode(input, y)
        output = self.decode(z, y)
        return output, mu, logvar

def one_hot(input, num_classes=10):
    # previously in earlier versions of pytorch we had to do this
    #result = torch.zeros(size=(input.size(0), num_classes))
    #result[range(0,input.size(0)), input[:]] = 1
    # but in modern pytorch we can simply use pytorchs builtin one_hot
    # just note that we have to return float for labels
    return F.one_hot(input, num_classes).float()

# z = torch.randint(0,9, size=(5,))
# print(z)
# print(one_hot(z))
def loss_function(outputs, imgs, mu, logvar, reduction='mean', use_mse=False):
    b,h,w,c=imgs.shape
    if reduction=='mean':
        criterion = nn.MSELoss(reduction=reduction) if use_mse else nn.BCELoss(reduction=reduction)
        recons_loss = criterion(outputs, imgs)
        # normalize the reconstruction loss
        recons_loss *= h*w*c
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        # when using mean, we always sum over the last dim 
        # so we get a batch so we ultimately average the batch!
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), -1)
        return recons_loss + kl.mean()
    else:
        criterion = nn.BCELoss(reduction='sum')
        recons_loss = criterion(outputs, imgs)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recons_loss + kl

#%%
# now lets train 
epochs = 50
embedding_size = 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VAE_Conditional(embedding_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=20)
print(datetime.datetime.now())
img_pairs=[]
for e in range(epochs):
    for i, (imgs,labels) in enumerate(dataloader_train):
        imgs = imgs.to(device)
        labels = labels.to(device)

        one_hot_labels = one_hot(labels).to(device)
        output, mu, logvar = model(imgs, one_hot_labels)
        loss = loss_function(output, imgs, mu, logvar )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'epoch/epochs: {e}/{epochs} loss: {loss.item():.4f}')
    scheduler.step()

interval = 1000
for i,(imgs, labels) in enumerate(dataloader_test):
    model.eval()
    with torch.no_grad():
        imgs = imgs.to(device)
        labels = labels.to(device)

        one_hot_labels = one_hot(labels).to(device)
        outputs, mu, logvar = model(imgs, one_hot_labels)
        loss = loss_function(outputs, imgs, mu, logvar )
        if i % interval:
            print(f'iter: {i}/{len(dataloader_test)} loss: {loss.item():.4f}')
            reconstructeds = outputs.cpu().view(-1, 1, 28, 28)
            count = 20 if imgs.size(0)>20 else imgs.size(0)
            imgs = imgs[:count].cpu().numpy()
            recons = reconstructeds[:count].numpy()
            pairs = np.array([np.dstack((img1,img2)) for img1, img2 in zip(imgs,recons)])
            img_pairs.append(pairs)

#%%
# create a 2d manifold for z1
@torch.no_grad()
def plot_latent_space_with_labels(dataset_test):
    # for this we feed our models encoder our test data 
    # or whatever data we want to visualize its z 
    # latent space distrubution. 
    # then we use plt.scatter to plot the points
    batch_size = 10000
    dataloader_test2 = torch.utils.data.DataLoader(dataset_test,
                                                batch_size = batch_size,
                                                num_workers = num_workers,
                                                pin_memory=True)
    imgs, labels = next(iter(dataloader_test2))
    imgs = imgs.to(device)
    labels = labels.to(device)
    one_hot_labels = one_hot(labels).to(device)
    z, _,_ = model.encode(imgs, one_hot_labels)

    z_= z.cpu().numpy()        
    plt.scatter(x=z_[:,0], y=z_[:,1], c=labels.cpu().numpy(), alpha=.4,
                s=3**2,cmap='viridis')
    plt.colorbar()
    plt.xlabel('Z[0]')
    plt.ylabel('Z[1]')
    plt.show()

plot_latent_space_with_labels(dataset_test)
# as you can see the shape this time looks really messy compared to the original
# VAE. its becasue we are really modelig P(z|c) which c==y . 
# https://wiseodd.github.io/techblog/2016/12/17/conditional-vae/
# http://ijdykeman.github.io/ml/2016/12/21/cvae.html
# To generate an image of a particular number, just feed that number into the decoder
# along with a random point in the latent space sampled from a gaussian distribution. 
# Even if the same point is fed in to produce two different numbers, the process will work 
# correctly, since the system no longer relies on the latent space to encode what number
# you are dealing with. Instead, the latent space encodes other information, like stroke 
# width or the angle at which the number is written.

#%%
# lets create new samples
z = torch.randn(size=(8, model.embedding_size)).to(device)
# labels = torch.randint(0,10,size=(8,))
labels = torch.tensor([1,2,1,3,7,9,4,5])
print(f'{labels.shape=}')
labels = one_hot(labels).to(device)
preds = model.decode(z, labels).detach().cpu()
img = make_grid(preds)
plt.imshow(img.numpy().transpose(1,2,0),cmap='gray')
#%%
import os
# lets display them a longside the original ones
def display_imgs_recons(img_pairs, nrows=8, rows=20, cols=1):
    img_cnt = len(img_pairs)
    print(img_cnt)
    fig = plt.figure(figsize=(32,24))
    for i in range(img_cnt):
        grid_imgs = make_grid(torch.from_numpy(img_pairs[i]),
                            nrow=nrows,
                            normalize=True)
        ax = fig.add_subplot(rows, cols, i+1, xticks=[],yticks=[])
        ax.imshow(grid_imgs.numpy().transpose(1,2,0))
        ax.set_title(f'cvae testset reconstruction-{i}')
        
        if not os.path.exists('results'):
            os.makedirs('results')
        save_image(grid_imgs, f'results/cvae_imgs_{i}.jpg')

# image reconstruction 
display_imgs_recons(img_pairs,nrows=10,rows=23,cols=4)

# now lets see the digits 2d manifold
# we cant have the interpolation we used for vanila vae, because for one
# we dont want to blend a class into another since each latent space is 
# conditioned on a class now. by this conditioning we are basically 
# explicitly asking the network to create digits like it. 
# so there is no point in interpolations like in vae.
# if go ahead and try that, we see smaller changes this way
# will distort the output
@torch.no_grad()
def vanila_vae_digits_manifold(n=10):
    z1 = torch.linspace(start=-9,end=9, steps=n)
    z2 = torch.linspace(start=-9, end=9, steps=n)
    
    grid = np.dstack(np.meshgrid(z1, z2))
    grid = torch.from_numpy(grid).to(device)
    grid = grid.view(-1, model.embedding_size)
    labels = torch.randint(0,9,size=(grid.size(0),))
    
    # remmember labels must be in one_hot encoded form!
    labels_one_hot = one_hot(labels).to(device)
    print(f'{grid.shape=}')
    print(f'{labels=}')
    
    preds = model.decode(grid, labels_one_hot).cpu().detach()
    img = make_grid(preds,nrow=n)
    fig = plt.figure(figsize=(n,n))
    ax = fig.add_subplot(111)
    ax.imshow(img.numpy().transpose(1,2,0))

vanila_vae_digits_manifold()

def generate_samples(n=10, num_classes=10):
    # number of digits 
    # n = 10 
    # num_classes = 10
    z = torch.randn(size=(n*num_classes, model.embedding_size)).to(device)
    print(z.shape)

    labels_grid = torch.tensor([[i] * n for i in range(num_classes)])
    print(labels_grid.flatten())

    labels_one_hot = one_hot(labels_grid.flatten()).to(device)
    print(f'z: {z.shape} labels: {labels_one_hot.shape}')

    preds = model.decode(z, labels_one_hot).cpu().detach()
    img = make_grid(preds, nrow=n)

    fig = plt.figure(figsize=(n,n))
    plt.title(f'Generate {n} samples for each class({num_classes})')
    ax = fig.add_subplot(111)
    ax.imshow(img.numpy().transpose(1,2,0))

generate_samples()

#%%
# Disentagled Variational Autoencoders or (β-VAE)
# good reads : https://towardsdatascience.com/disentanglement-with-variational-autoencoder-a-review-653a891b69bd 
# https://lilianweng.github.io/lil-log/2018/08/12/from-autoencoder-to-beta-vae.html#contractive-autoencoder
# https://openreview.net/forum?id=Sy2fzU9gl
# https://arxiv.org/pdf/1901.09415.pdf
# https://arxiv.org/abs/1606.05579

# now here it is, the disentangled vae! we already talked about it in our vanilla vae section
# we saw that the only difference was we added a factor/beta to the kl loss and that was it!
# but whats the idea behind it?
# The basic idea in disentagled vae is that, we want different neurons in our latent 
# distribution to be uncorollated, they all try to learn something different about 
# the input data. In order to implement this, the only thing that needs to be added
# to the vanilla VAE, is a β term.
# previously for the vanilla VAE we had : 
#     L = E_q(z|X)[log_p(X|z)] - D_KL[q(z|X)||p(z))]
# Now for the disentagled version (β-VAE) we just add the β like this : 
#     L = E_q(z|X)[log_p(X|z)] - βD_KL[q(z|X)||p(z))]
# so to put it simply, in a disentagled vae (B-Vae) the autoencoder will only 
# use a varable if it its important.

def fc_batchnorm_act(in_, out_, use_bn=True, act=nn.ReLU()):
    return nn.Sequential(nn.Linear(in_,out_),
                         act,
                         nn.BatchNorm1d(out_) if use_bn else nn.Identity())
                         
class B_VAE(nn.Module):
    def __init__(self, embedding_size=5):
        super().__init__()
        self.embedding_size = embedding_size
        
        # self.fc1 = nn.Linear(28*28, 512)
        self.encoder_entry = nn.Sequential(fc_batchnorm_act(28*28,512),
                                           fc_batchnorm_act(512,256),
                                           fc_batchnorm_act(256,128),
                                           fc_batchnorm_act(128,64))
        self.fc_mu = nn.Linear(64, embedding_size)
        self.fc_std = nn.Linear(64, embedding_size)

        self.decoder = nn.Sequential(fc_batchnorm_act(embedding_size, 64),
                                     fc_batchnorm_act(64,128),
                                     fc_batchnorm_act(128,256),
                                     fc_batchnorm_act(256,512),
                                     fc_batchnorm_act(512, 28*28,False,nn.Sigmoid()))
        # self.decoder = nn.Sequential(nn.Linear(embedding_size, 512),
        #                             nn.ReLU(),
        #                             nn.Linear(512, 28*28),
        #                             nn.Sigmoid())

    def reparameterization_trick(self, mu, logvar):
        std = torch.exp(logvar * 0.5)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, imgs):
        imgs = imgs.view(imgs.size(0), -1)
        output = self.encoder_entry(imgs)
        # remember we dont use nonlinearities for mu and logvar!
        mu = self.fc_mu(output)
        logvar = self.fc_std(output)
        z = self.reparameterization_trick(mu, logvar)
        return z, mu, logvar

    def decode(self, z):
        reconstructed_imgs = self.decoder(z)
        reconstructed_imgs = reconstructed_imgs.view(-1, 1, 28, 28)
        return reconstructed_imgs

    def forward(self, x):
        # encoder 
        z, mu, logvar = self.encode(x)
        # decoder
        reconstructed_imgs = self.decode(z)
        return reconstructed_imgs, mu, logvar

def loss_disentagled_vae(outputs, imgs, mu, logvar, Beta, reduction='mean', use_mse=False):
    # this loss has two parts, a construction loss and a KL divergence loss which
    # shows how much distance exists between two given distrubutions. 
    if reduction=='mean':
        if use_mse:
            criterion = nn.MSELoss()
        else:
            criterion = nn.BCELoss(reduction='mean')
        recons_loss = criterion(outputs, imgs)
        # normalize the reconstruction loss
        recons_loss *= 28*28
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        # when using mean, we always sum over the last dim
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), -1)
        # we use beta and multiply it by our kl term. this is specific to 
        # disentagled vae and is actually the main reason why the disentaglement 
        # work
        return torch.mean(recons_loss + (Beta*kl))
    else:
        criterion = nn.BCELoss(reduction='sum')
        recons_loss = criterion(outputs, imgs)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recons_loss + (Beta*kl)    

epochs = 50

embeddingsize = 5
interval = 2000
reduction='mean'
# beta is a value biger than 1 
Beta = 5.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = B_VAE(embeddingsize).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr =0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 50)

for e in range(epochs):
    for i, (imgs, labels) in enumerate(dataloader_train):
        imgs = imgs.to(device)
        preds,mu, logvar = model(imgs)

        loss = loss_disentagled_vae(preds, imgs, mu, logvar, Beta= Beta, reduction=reduction, use_mse=False)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 
        if i% interval ==0:
            loss = loss/len(img) if reduction=='sum' else loss
            print(f'epoch {e}/{epochs} [{i*len(imgs)}/{len(dataloader_train.dataset)} ({100.*i/len(dataloader_train):.2f}%)]'
                  f'\tloss: {loss.item():.4f}'
                  f'\tlr: {scheduler.get_lr()}')
    scheduler.step()
#%% 
# test
test_set_size = len(dataloader_test.dataset)
img_pairs = []
losses = []
interval = 10
with torch.no_grad():
    for i, (imgs, labels) in enumerate(dataloader_test):
        imgs = imgs.to(device)
        preds, mu, logvar = model(imgs)
        loss = loss_disentagled_vae(preds, imgs, mu, logvar, Beta= Beta, reduction=reduction, use_mse=False)
        losses.append({'val_loss':loss.item()})
        
        print(f'[{i*len(imgs)} / {test_set_size} ({100.*i/len(dataloader_test):.2f}%)]'
            f'\tloss: {(loss).item():.4f}')

        if i%interval==0:
            reconstructeds = preds.cpu().view(-1, 1, 28, 28)
            imgs = imgs[:20].cpu().numpy()
            recons = reconstructeds[:20].numpy()
            pairs = np.array([np.dstack((img1,img2)) for img1, img2 in zip(imgs,recons)])
            img_pairs.append(pairs)
#%% 
# import pandas as pd 
# pd.DataFrame(losses).plot()
model.eval()
# create sample image
z = torch.randn(size=(8, model.embedding_size)).to(device)
reconstructed_imgs = model.decode(z).cpu().detach()
img = make_grid(reconstructed_imgs)
plt.imshow(img.numpy().transpose(1,2,0))
plt.title('random generation')
#%%
# n = 1
# z = torch.randn(size=(n,model.embedding_size)).to(device)
# print(f'{z.shape=}')
# fig = plt.figure()
# ax = fig.add_subplot(111)
# preds = model.decode(z).cpu().detach()
# img_latent_space = make_grid(preds,nrow=5).numpy().transpose(1,2,0)
# ax.imshow(img_latent_space)
#%%
def change_latentvariable(z, n=3, steps=3, scaler=0.2, dim=0):
    z_new = torch.zeros(size=(n, steps, z.size(-1)))
    #! edit fix this!  
    for i in range(steps):
        # we are basically creating a n(series) x steps x z tensor and in each
        # step, we fill one row of this tensor until all of them are filled
        # we use the same initial z, but each time we ever so slightly change it
        # so each row is different from the previous one, basically we are tryng
        # to have smooth interpolation between these by introducing fixed steps
        # into the latent vector.
        z_new[:,i,:] = z
        print(f'{z_new=}')
        print(f'{z_new.shape=}')
        # scale the latent vector by small amount
        # so they are different in each step
        z_new[:,i, dim] = z_new[:,i, dim] - (scaler*i)
    return z_new

def show_manifold(z, n, steps, dim , device):
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111)
    latent_vectors = change_latentvariable(z, n, steps, dim).to(device)
    print(latent_vectors.shape)
    preds = model.decode(latent_vectors.view(-1,model.embedding_size)).cpu().detach()
    img_latent_space_man = make_grid(preds,nrow=steps).numpy().transpose(1,2,0)
    ax.imshow(img_latent_space_man)
n=1
show_manifold(z, n=n, steps=5, dim=3, device=device)
show_manifold(z, n=n,  steps=5, dim=1, device=device)
show_manifold(z, n=n,  steps=5, dim=2, device=device)
show_manifold(z, n=n,  steps=5, dim=3, device=device)
show_manifold(z, n=n,  steps=5, dim=4, device=device)
# visualize the 2d manifold 
#%%
# variations over the latent variable :
z_dim = model.embedding_size
sigma_mean = 2.0*torch.ones((z_dim))
mu_mean = torch.zeros((z_dim))

# Save generated variable images :
nbr_steps = 8
gen_images = torch.ones(size=(nbr_steps,1,28,28) )

for latent in range(z_dim) :
    #var_z0 = torch.stack( [mu_mean]*nbr_steps, dim=0)
    var_z0 = torch.zeros(nbr_steps, z_dim)
    val = mu_mean[latent]-sigma_mean[latent]
    step = 2.0*sigma_mean[latent]/nbr_steps
    print(latent, mu_mean[latent]-sigma_mean[latent], mu_mean[latent], mu_mean[latent]+sigma_mean[latent])
    for i in range(nbr_steps) :
        var_z0[i] = mu_mean
        var_z0[i][latent] = val
        val += step
    var_z0 = var_z0.to(device)
    gen_images_latent = model.decode(var_z0)
    gen_images_latent = gen_images_latent.cpu().detach()
    gen_images = torch.cat( [gen_images, gen_images_latent], dim=0)
img = make_grid(gen_images)
plt.imshow(img.cpu().numpy().transpose(1,2,0))
#%%
#! remove these visualizations, dont need them really I guess
# here we create a grid of images where each row corresponds to one latent dimension. 
# Within each row, the latent dimension is varied across a range of values (from -3 to 3), 
# while the rest of the latent vector is kept constant. 
# One column in each row shows the output when the latent value is near its original value
# (highlighted with a green border), and the other columns show how shifting that latent
# dimension affects the generated image. 
# This provides an intuitive way to understand and visualize the influence of each latent 
# dimension in the model.
@torch.no_grad()
def latent_space_walk(num_rows,num_cols=9,figure_width=10.5,image_height=1.5):
    fig = plt.figure(figsize=(figure_width, image_height * num_rows))
    
    for i in range(num_rows):
        z_i_values = np.linspace(-3.0, 3.0, num_cols)
        z_i = z[0][i].cpu().numpy()
        z_diffs = np.abs((z_i_values - z_i))
        j_min = np.argmin(z_diffs)
        for j in range(num_cols):
            z_i_value = z_i_values[j]
            if j != j_min:
                z[0][i] = z_i_value
            else:
                z[0][i] = float(z_i)
                
            x = model.decode(z).cpu().numpy()
            
            ax = fig.add_subplot(num_rows, num_cols, i * num_cols + j + 1)
            ax.imshow(x[0][0], cmap='gray')
            
            if i == 0 or j == j_min:
                ax.set_title(f'{z[0][i]:.1f}')
            
            if j == j_min:
                ax.set_xticks([], [])
                ax.set_yticks([], []) 
                color = 'mediumseagreen'
                width = 8
                for side in ['top', 'bottom', 'left', 'right']:
                    ax.spines[side].set_color(color)
                    ax.spines[side].set_linewidth(width)
            else:
                ax.axis('off')
        z[0][i] = float(z_i)
        
    plt.tight_layout()
    fig.subplots_adjust(wspace=0.04)
    
num_rows = z.shape[-1]
latent_space_walk(num_rows)
#%% 
# VQ-VAE
# ref : paper: Neural Discrete Representation Learning
# there are two variants, the first paper came out in 2017 
# and a followup work with some improvements came in 2019 
# paper1?: https://arxiv.org/abs/1711.00937 2017
# paper2 (pixelcnn, required for generation part-the conditional version) https://arxiv.org/abs/1606.05328
# paper3(updated version): https://arxiv.org/abs/1906.00446 2019
#! pytorch implementation from Aäron van den Oord (author of pixelcnn), but overall the vq-vae explanation part is ok!): 
# https://colab.research.google.com/github/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb
# a good introductory video (doesnt get deep, doesnt cover all aspects of it(so reading the whole paper is a must), but overall is good for a brief introduction) https://www.youtube.com/watch?v=VZFVUrYcig0
#
# as exciting as the idea behind VAEs are, they are prune to posterior collapse
# and we saw that first hand, we tried different methods to improve upon our basic
# vae, but the outcome left a lot to be desired really! we saw that the basic vae was
# originally only used with simple datasets such as mnist and frey face dataset, and for
# anything more complex it wouldnt work.(rememeber the simplification we did in vae, its one of 
# the issues that prevent us from performing well on complex datastes)
# we saw that to get around these issues, different variants were proposed, we used some 
# of them in our work and got much better results. so there are other variants which improve upon it, 
# we didnt cover all of them here, because there are many. so I try to only use the ones
# that had substantially more novelties and improvements. the next variant we are going to
# cover is the VQ-VAE, a very influential paper, that came to resolve vae issues
# and was used in many high profile papers afterward (imagen, vqgan, etc)
# VQ-VAE uses the same basic idea of the vae, however, it changes it in somewhat fundamental way
# for one, it uses discrete latent representation instead of continuous one and rightfully argues 
# that its a more natural appraoch toward modeling what we are dealing with in the real world
# mostly because many important real world objects are discrete.
# for example, we can imagine classes like a Cat or a Car, and notice that it really may not 
# make sense to want to interpolate between these classes!(though we can find many other examples
# that this makes prefect sense, I just wanted to also have some counter examples and see how that
# helps us reach a new deduction) on top of that, discrete representations are
# also easier to model unlike their continous counterparts where we'd need to learn the
# dependencies between different variables which could be very complex(again there are defnitely
# good cases/solutions to this like using hierarchical latents to address complex dependencies, and
# the likes, but we cant hit everything with a hammer as not everything is a nail! we'll see this
# shortly how this new understanding/assumpution allows us to get drastically better results).
# !edit say vqvae is not generative itself, and it needs another model for pior? cite it from paper
# directly or say it in our words?(use the following (at the end) explanations here?)
# ! ADD introduction from paper, it says it good! 
#
# The abstract reads: 
# Learning useful representations without supervision remains a key challenge in 
# machine learning. In this paper, we propose a simple yet powerful generative model
# that learns such discrete representations. 
# Our model, the Vector Quantised-Variational AutoEncoder (VQ-VAE), differs from VAEs
# in two key ways: the encoder network outputs discrete, rather than continuous, codes;
# and the prior is learnt rather than static. 
# In order to learn a discrete latent representation, we incorporate ideas from vector 
# quantisation (VQ). Using the VQ method allows the model to circumvent issues of 
# "posterior collapse" -- where the latents are ignored when they are paired with a 
# powerful autoregressive decoder -- typically observed in the VAE framework. 
# Pairing these representations with an autoregressive prior, the model can generate 
# high quality images, videos, and speech as well as doing high quality speaker conversion
# and unsupervised learning of phonemes, providing further evidence of the utility of the
# learnt representations. 
# 

# how the model works?
# we have 3 parts in a vq-vae architecture(vqvae itself, we need more than just vqvae to generate images, more on this in a moment),
# an encoder, a quantizer and a decoder.
# the encoder gets an image and maps it to a sequence of discrete latent variables
# the decoder takes these latent sequences and tries to reconstruct the input
# now what about the quantizer part you may ask?
# well to be more specific, our encoder gets an rgb image, and outputs some outputs ( lets call it E(x))
# we also have an embedding layer where we make sure the encoder's ouput channels are the same 
# as the dimentionality of this embedding layers.
# to calculate the actual discrette latent variables, we need to use a trick, which is we 
# instead of feeding the encoders output directly to decoder, we find the nearest embedding vector
# and the encoders output, and grab its index.(grab the index of the embedding layer where its closes 
# to our encoders outputs) we use this index, and grab the corrosponding
# embedding vector from our embedings, and feed that to our decoder, and decoder uses it to 
# reconstruct the input. 
# since our comparsion here (the neigherst neighbor between our encoder and embeddings) doesnt
# have a real gradient, we cant backprop through it, (we cant have backward pass for that comparsion),
# so instead we simply pass the gradients from the decoder to the encoder without changing them.
# this is why we made our encoder output channels the same as embedding size, so we can compare with it
# and use the decoders gradients for encoder.
# the idea behind this is that since the encoders output representation and the input to decoder(which is the embeddings)
# share the same channel dimensial space, the gradients contain useful information for how the 
# encoder has to change its output to lower reconstruction error.
# basically if the encoders output is close to embeddings, then we should be able to treat them
# as the same and hence we can use embeddings gradients for the outputs as well. 
# we do this in quantizer part beftween encoer and decoder

# !edit
# note that the vqvae by itself is not a generative model, we cant generate anything with it.
# It's for learning useful "discrete" representations/ good embeddings if you will.
# to generate new data(we are not bound to only images, we can generate all sorts of data, image, audio,etc),
# as its stated in the paper, we need to pair it with an autoregressive 
# model like PixelCNN which the paper used or a transformer that can model the prior 
# distribution over the discrete latent codes. 
# (in the paper they used a PixelCNN architecture over the discrete latents to 
# generate new sequences of codes, which are then fed to the decoder to produce images.)
# so to generate new images, we train the vqvae model to get a good embeddings and encoder-decoder.
# if you remember vq-vae uses the vae framework, but so far we didnt specify any prior, or the likes
# this is where the second model comes in, unlike the normal vae, where we specified a prior, 
# here we learn it from data! so in our next step, we train a separate prior model (like PixelCNN) 
# on the discrete latent codes. and finally for the actual generation process, we now simply sample 
# from the prior model we just trained and get a sequence of codes which we then use to decode them 
# into an image.
# we can condition our generation the same way we did for normal vae, and generate images for certain class, 
# to do this we'd need the prior model to be conditioned on the class label, that is we train the 
# prior model to generate codes conditioned on the label, then decode those codes.() we can make it
# more intersting by conditioning it on a piece of text, so by describing what we want, we get an image)
# 
#@
# excessive?
# as we said, the vqvae model itself doesnt include the prior model in its training. The prior 
# is learned separately. So the vqvae focuses on reconstruction and learning the embedding(coodbook),
# while the prior model handles the generation of the latent codes. 
# This separation allows the vqvae to be used in different ways, depending on the prior model
# used.
# 
# Another thing to consider is that since the latent space is discrete, it might be more efficient
# for certain types of data, like speech or music, where discrete representations are more natural.
# For images, though, using a discrete latent space with a powerful decoder can still lead to 
# high-quality reconstructions and generations when paired with a good prior model.
# 
# Also, in terms of training, VQ-VAE uses a straight-through estimator to handle the non-differentiable 
# quantization step during backpropagation. This allows the model to learn the codebook entries effectively.
# The commitment loss(beta) also helps to ensure that the encoder commits to the codebook entries.
# But coming back to generation: without a prior model, can VQ-VAE generate new images? Probably not,
# because the encoder only maps inputs to discrete codes, and the decoder maps codes back to data. 
# To generate new data, you need new codes that weren't derived from an input, which requires a 
# prior model. 
# 
# the VQ-VAE alone isn't a generative model, but when combined with a prior model over the discrete 
# latents, it can generate new samples.
# In short, the use cases for VQ-VAE include learning discrete representations for data compression,
# unsupervised learning of useful features, and enabling controllable generation when paired with 
# an appropriate prior model. To generate new images, we need that prior model to sample plausible 
# sequences of discrete codes, which the VQ-VAE decoder can then turn into images

# a simplistic res module
class conv(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, stride=1, padding=1, batch_norm=True, bias=False,act=nn.LeakyReLU(0.2)):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(out_dim) if batch_norm else nn.Identity(),
            act,
            # nn.Conv2d(out_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=bias),
            # nn.BatchNorm2d(out_dim) if batch_norm else nn.Identity(),
            # act
        )
        # residual connection needs input and output dimensions to match
        self.residual_connection = (in_dim == out_dim and stride == 1)

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual_connection:
            out += x
        return out

class deconv(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, stride=2, padding=1, act=nn.LeakyReLU(0.2), batch_norm=True, bias=True):
        super().__init__()
        self.deconv_block = nn.Sequential(
            nn.ConvTranspose2d(in_dim, out_dim, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(out_dim) if batch_norm else nn.Identity(),
            # nn.GroupNorm(1,out_dim) if batch_norm else nn.Identity(),
            act,
            # nn.Conv2d(out_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=bias),
            # nn.BatchNorm2d(out_dim) if batch_norm else nn.Identity(),
            # act
        )
        # residual connection needs input and output dimensions to match
        self.residual_connection = (in_dim == out_dim and stride == 1)

    def forward(self, x):
        out = self.deconv_block(x)
        if self.residual_connection:
            out += x  
        return out

class Quantizer(nn.Module):
    def __init__(self, num_embd, embd_size, beta_weight, use_ema=True, decay_rate=0.99, epsilon=1e-5):
        super().__init__()
        
        self.num_embd = num_embd
        self.embd_size = embd_size
        self.beta_weight = beta_weight
               
        self.embeddings = nn.Embedding(self.num_embd, self.embd_size)
        # lets normalize the weights uniformly
        self.embeddings.weight.data.uniform_(-1/self.num_embd, 1/self.num_embd)
                
        # using exponential moving average to update the embedding
        # vectors instead of an auxillary loss can seemingly improve
        # the convergence a lot and prevents low preprelxity (more about this in a moment!)
        # so we implement it as well and see how it works!
        # from Aäron van den Oord implementation: 
        #  using ema has the advantage that the embedding updates are
        #  independent of the choice of optimizer for the encoder, 
        #  decoder and other parts of the architecture. 
        #  For most experiments the EMA version trains faster than the non-EMA version.
        self.use_ema = use_ema
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        # as we briefly explained, in vq-vae each embedding is actually acting as a prototype 
        # or a representative point in the latent space which during the quantization process, 
        # we assign the encoder outputs to the nearest of such embeddings. if you think about it
        # its just like how we assign data points to the closest centroid in a typical clustering algorithm 
        # like k-means. 
        # here we plan on keeping track of this process, basically how often each cluster is assigned,
        # in other words, we are basically tracking how many times each embedding (or cluster center) 
        # is chosen during the assignment process.(see more explanation a head)
        # we use self.register_buffer so ema_cluster_size is saved when we save our model
        # and also its not included in computational graph
        # sidenote: instead of ema_cluster_size, ema_assignment_frequency, or ema_cluster_frequency could be better!
        self.register_buffer('ema_cluster_size', torch.zeros(self.num_embd))
        self.ema_w = nn.Parameter(torch.Tensor(self.num_embd, self.embd_size))
        self.ema_w.data.normal_()

        
    def forward(self, encoder_outputs:torch.Tensor):
        # print(f'{encoder_outputs.shape=}')
        # we need to change our encoder_outputs shape from bchw to bhwc 
        # basically moving the channel to the last dim, so that when we
        # flatten the whole thing, we get each separate channels,
        # so if our input shape is [16,64,32,32] we ultimately get
        # [16*32*32, 64] or [16384,64] which means we are quantizing 
        # each 16384 vectors independently, in otherwords, the channels
        # are used as space in which it gets quantized (so it matches embedding size)
        encoder_outputs = encoder_outputs.permute(dims=(0,2,3,1)).contiguous()
        encoder_outputs_shape = encoder_outputs.shape
        encoder_outputs_flatten = encoder_outputs.view(-1, self.embd_size)
        
        # print(f'{encoder_outputs_flatten.shape=}')
        # print(f'{self.embeddings.weight.shape=}')
        # lets calculate the distance between them (squared euclidean distance)
        # which is Σᵢ(zᵢ - eᵢ)² which if we expand it will be 
        # d(z,e) = ∣∣z∣∣² + ∣∣e∣∣² − 2z⋅eᵀ
        # that is taking the l2 norm of the encoders output and embeddings which are (N,D)
        # N being our sample count(note its not batchsize, as we flattened it) and D being 
        # the embedding size followed by a dotproduct between the two (equation 1 from paper)
        # note we want an (N,num_embds) to be our final tensor shape, so when summing
        # we keepdim=True for encoderoutputs so it gives us a tensor of shape [N,1]
        # so when we add it to sum of embeddings which will be [embd_num], its broadcast to
        # the final shape of [N,embd_num] which simply says, we have embd_num for each sample!
        distances = (torch.sum(encoder_outputs_flatten**2, dim=1,keepdim=True) + 
                    torch.sum(self.embeddings.weight**2, dim=1) -
                    2*torch.matmul(encoder_outputs_flatten, self.embeddings.weight.t()))
        # or we could use torch.cdist
        # distances = torch.cdist(encoder_outputs_flatten, self.embeddings.weight, p=2) ** 2

        # now that we have the distances, lets grab the min indexes 
        min_indexes = torch.argmin(distances, dim=1).unsqueeze(1)
        # create a onehot encoded tensor for encodings
        encodings = torch.zeros(size=(min_indexes.shape[0], self.num_embd), device=encoder_outputs.device)
        # set each encoding to 1 where the indexes specify
        encodings.scatter_(dim=1, index=min_indexes,value=1)
        # now to get the actual embeddings, we simply multiply our encodings tensor
        # which is a one hot encoded vector by the embeddings. where-ever we have 1,
        # we will get the respective embeddings and the rest will be 0s. 
        # this is basically a selection based on indexes now our quantized tensor will
        # have embeddings at the specified indexes, and 0s everywhere else.
        # this is zₑ(x)
        quantized_z_ex = torch.matmul(encodings, self.embeddings.weight).view(encoder_outputs_shape)
        
        if not self.use_ema:
            # now to calculate the loss which is equation 3, we need to calculate the last two terms
            # as the paper says: 
            # The decoder optimises the first loss term only, the encoder optimises 
            # the first and the last loss terms, and the embeddings are optimised 
            # by the middle loss term.
            # we need quantized.detach() (stop_gradient operator in paper(written as sg[]))
            e_loss = F.mse_loss(quantized_z_ex.detach(), encoder_outputs)
            # this is z\_qx the 3rd term in equation 3, we stop gradient on encoders_output this time
            q_loss = F.mse_loss(quantized_z_ex, encoder_outputs.detach())
            # and finally we use the beta scaler for the second term
            # we later add the reconstruction loss(log) from decoder to this 
            loss = q_loss + self.beta_weight*e_loss
        
        else:# Use EMA to update the embedding vectors
            if self.training:
                # note that we are only using ema during training to stabilize and
                # update the embeddings and we dont need it during inference.
                # the whole point of of this process is to make training more stable,
                # by directly influencing the embeddings weight instead of using 
                # a second loss term (q_loss) like the normal form
                self.ema_cluster_size = self.ema_cluster_size * self.decay_rate\
                                         + (1 - self.decay_rate) * torch.sum(encodings, 0)

                # Laplace smoothing of the cluster size
                # Laplace (or additive) smoothing simply means adding a
                # small constant (i.e. epsilon) to each count so we dont face zero value! if 
                # we dont do this, and an embedding (cluster) is never used (e.g. ema_cluster_size_i = 0
                # we'll face division byzero in ema_w / ema_cluster_size)
                # this means num_emb*epsilon, but as you see, we are doing more than that here!
                # all this jazz is for normalizing the ema_cluster_size!
                total_assignments = torch.sum(self.ema_cluster_size.data)
                # we do this so that every cluster has a minimum nonzero value that 
                # prevents division by zero issues when we update the embeddings.
                # after we add the epsilon to all of the clusters we use total_assignments
                # in the denominator to normalize it.at the end once again multiply the whole result
                # by total_assignments again to scale it back to the original total.
                # we do this so the laplace smoothing doesn't change the  total number of assignments,
                # which is important for maintaining the scale when updating the embeddings. 
                # so the division by (total + num_embds*epsilon) and multiplication by total at the end
                # redistributes the added epsilon across the clusters while keeping the sum the same.
                #
                # (to have a better picture imagine we have only two clusters (num_embd=2), and epsilon=1. 
                # now suppose our total_assignments=10 and ema_cluster_size = [3,7] before adding epsilon,
                # then adding epsilon gives [4,8], sum is 12. Denominator is 10 + 2*1=12. 
                # So (4/12, 8/12) *10 = (4*10/12, 8*10/12) = (3.333, 6.666). So the total is 
                # preserved as 10, but the epsilon is distributed proportionally. 
                # this way, the relative sizes are adjusted but the total remains the same as before
                # adding the epsilon. (remember we dont want to change the total count, we just
                # want to prevent any cluster from being zero).)
                # 
                # long story short!:
                # we first add epsilon to each cluster (num_embd clusters total addition K*epsilon)
                # then normalize by (total + K*epsilon) to get proportions.
                # then multiply by original total to maintain the same total assignments but with smoothed counts.
                # all of this to prevent any cluster from having zero count, thus avoiding division
                # by zero when computing the average (ema_w / ema_cluster_size).
                self.ema_cluster_size = ((self.ema_cluster_size + self.epsilon)/(total_assignments + self.num_embd * self.epsilon) * total_assignments)
                # dw is the sum of encoder outputs assigned to each embedding, which is used 
                # to update the embeddings.(when we transpose encodings and multiply it by 
                # encoder_outputs_flatten, we're essentially summing the encoder outputs that
                # correspond to each embedding entry, this becomes the new value for our embeddings)
                dw = torch.matmul(encodings.t(), encoder_outputs_flatten)
                # update for the embeddings vectors. we use ema_w is to stabilize training 
                # by gradually updating the embeddings based on the recent assignments.
                # decay_rate determines how much of the old average is kept versus the new data(dw)
                # so here, ema_w is being updated by decaying the previous value and adding a 
                # portion of the new dw.
                # 
                # !edit:
                # ema_w controls how much historical data is retained vs. new contributions.
                # This smooths the codebook updates over time, preventing abrupt changes and 
                # stabilizing training.
                # ema_w tracks the cumulative weighted sum of encoder outputs assigned to each
                # codebook entry.
                
                self.ema_w = nn.Parameter(self.ema_w * self.decay_rate + (1 - self.decay_rate) * dw)
                # and finally to normalize the EMA of the embeddings vectors we divide ema_w by the cluster size 
                # as we saw the ema_cluster_size tracks how many times each embedding has been assigned, 
                # so dividing by this would average the summed encoder outputs (dw) by the number of 
                # assignments, effectively updating the embeddings to be the average of the encoder outputs
                # assigned to them. (note the unsqueeze(1) which we use to make the dimensions match for broadcasting)
                # 
                # sidenote: 
                # by using EMA twice for both the cluster sizes and the embedding vectors(ema_w), 
                # we make sure we (i.e. the model) avoid relying only on the current batchs assignments, 
                # which might be noisy. this should lead to a more stable training.
                
                # !edit: add relavent bits:
                #sidenote:
                # The `ema_cluster_size` was addressed earlier with Laplace smoothing to prevent division by zero. Now, combining that with `ema_w`, which is the EMA of the summed encoder outputs, and then normalizing by the smoothed cluster sizes gives the updated embeddings. This way, even if a cluster hasn't been assigned in the current batch, the EMA ensures it still gets updated based on historical data, smoothed by the epsilon.
                # I need to check if the dimensions make sense. `encodings` is probably a matrix of shape [batch*latent_dim, num_embeddings], where each row is a one-hot vector indicating the chosen embedding. Transposing it would make [num_embeddings, batch*latent_dim], and multiplying by `encoder_outputs_flatten` (which is [batch*latent_dim, embedding_dim]) gives a [num_embeddings, embedding_dim] matrix `dw`. Each row in `dw` is the sum of encoder outputs assigned to that embedding.
                # Then `ema_w` is a [num_embeddings, embedding_dim] parameter that's updated with EMA. Dividing by `ema_cluster_size.unsqueeze(1)` (which is [num_embeddings, 1]) gives the average encoder output per embedding, scaled by their smoothed counts. This updates the codebook embeddings to be the moving average of the encoder outputs assigned to them.
                # So, the key idea is to use EMA to smoothly update the codebook entries based on their usage over time, which helps in training stability. The division by `ema_cluster_size` ensures that each embedding vector is the average of all encoder outputs that were assigned to it, adjusted for the smoothing to avoid division by zero and ensure all embeddings get updated even if rarely used.
                # But why not just use the current batch's assignments directly? Because EMA reduces variance in the updates, making training more stable. Especially with discrete assignments, which can be noisy, using an EMA helps the codebook converge more smoothly.
                # Also, since `encodings` might be soft assignments (like in the original VQ-VAE with straight-through estimator), the EMA would still work by accumulating the weighted contributions from each encoder output to each codebook entry. The Laplace smoothing on cluster sizes ensures that even rarely used embeddings have a non-zero count, so their EMA doesn't get stuck or cause numerical issues when updating.
                # Putting it all together, these lines are responsible for updating the codebook embeddings in a stable, smoothed manner using EMA, which is crucial for effective training of VQ-VAEs by preventing codebook collapse and ensuring all embeddings are utilized.
                # 
                #Why These Steps?
                # Stability via EMA:
                #     Directly using per-batch assignments would cause noisy updates. EMA smooths these updates over time, critical for avoiding codebook collapse (where most embeddings go unused).
                #     Example: If a codebook entry is rarely used, its ema_cluster_size remains small, but EMA ensures it still receives gradual updates from dw.
                # Normalization by Cluster Size:
                #     Dividing by ema_cluster_size converts the summed encoder outputs (ema_w) into an average of the encoder outputs assigned to each codebook entry.
                #     Without this, embeddings would grow disproportionately large based on how frequently they’re used.
                # Alignment with Laplace Smoothing:
                #     The earlier Laplace smoothing of ema_cluster_size ensures no division by zero, even for unused codebook entries.
                #     Smoothing also ensures rarely used embeddings still receive small updates (via epsilon), preventing them from becoming "dead" units.
                #
                # Key Takeaways
                # EMA Smoothing: Stabilizes codebook updates by prioritizing historical consistency over noisy per-batch assignments.
                # Normalization: Ensures embeddings represent the average of assigned encoder outputs, not their sum.
                # Laplace Symbiosis: The earlier smoothing of ema_cluster_size ensures numerical stability and gradual updates for all codebook entries, even rarely used ones.
                
                self.embeddings.weight = nn.Parameter(self.ema_w / self.ema_cluster_size.unsqueeze(1))
            
            e_loss = F.mse_loss(quantized_z_ex.detach(), encoder_outputs)
            loss = self.beta_weight * e_loss
        
        # this is the Straight-Through Estimation (STE) part, which allows the gradients to 
        # flow through our discrete operation(i.e. choosing the nearest embedding vector(argmin)
        # which is non-differentiable.
        # basically using the detach trick here prevents the gradients from flowing through 
        # (quantized_z_ex - encoder_outputs) so, during backpropagation, in the forward pass we
        # use the quantized_z_ex, the model sees the actual quantized values and the in 
        # backward pass, its as if the encoder_outputs were used, which allows the gradients
        # to propagate through the encoder.
        quantized_z_ex = encoder_outputs + (quantized_z_ex - encoder_outputs).detach()
        # lets also calculate the average selection probablity. 
        # this tells us how often an embedding vector is selected. 
        # its important because it can tell us whether some vectors are rarely selected
        # or not, if so then it might indicate embedding space/codebook collapse!
        # its a common issue where only a few embeddings are used constantly and 
        # therefore it results in wasted model capacity.
        avg_probs = encodings.mean(dim=0)
        # now using the average selection probablity, we can calculate the preplexity
        # of the embedding space utilization.
        # the prepelxity measures how evenly the embedding vectors are being used.
        # The formula is based on Shannon entropy H = − ∑pᵢ * logpᵢ 
        # taking exp(H) gives us the perplexity, which ranges from 1 if only one embedding vector
        # is used, meaning no diversity, to num_embds if all embedding vectors/codebook vectors
        # are used equally.
        # basically a higher perplexity means better embedding space/codebook utilization
        # a low perplexity tells us that the model is underutilizing the embedding space, 
        # which may lead to worse performance.
        #
        # sidenote:
        # using a exponential moving average should prevent low prepexlity!
        # !edit add moving average!
        # 
        # sidenote:
        # !edit
        # Perplexity is in fact a measure of uncertainty or diversity in a probability distribution.
        # It is commonly used in language modeling, information theory, and in Vector Quantization 
        # (VQ) to measure how well an embedding space (or a codebook as many call it) is being utilized.
        # and as we said before it comes from Shannon entropy, which measures the amount of uncertainty
        # in a probability distribution.
        # The formula for entropy is:
        #  H(p) = - Σp_i * log(p_i)
        # where p_i represents the probability of selecting the i-th element from a distribution
        # The perplexity is simply the exponentiation of entropy:
        #  PPL = exp(H(p)) = e^(- Σ p_i * log(p_i))
        #This measures how uncertain or diverse a probability distribution is. In VQ-VAE,
        # it helps evaluate how well embedding space is being utilized.

        # sidenote: 
        # in many implementations you may see people refering to embeddings(the whole embedding vectors/ditionary of embedding vectors!)
        # as codebooks! 
        # so embedding vectors and codebook vectors are the same thing! if you read the paper
        # you'll see the authors always used embedding space/embedding vectors and the likes to
        # address embedding vectors! but many people started calling it codebook because it kindaof
        # looks like it (a ditionary of embedding vectors! thus a codebook!), 
        # anyway I like the embeddings better so I keep using that
        # but at the same time I add the codebook as well so you get familiar with t hat term as well
        # its usually specific to vq-vae implementations.
        #
        #
        prepelexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs+ 1e-10)))
        
        return loss, quantized_z_ex.permute(dims=(0,3,1,2)).contiguous(), prepelexity #, encodings

# lets now add the main model 
#! make it conditional so we can create different types of images?!
class VQVAE(nn.Module):
    def __init__(self, input_channels, embd_num, embd_size, beta, use_ema):
        super().__init__()
    
        self.input_channels = input_channels
        self.embd_num = embd_num
        self.embd_size = embd_size
        self.beta = beta
        self.use_ema = use_ema
        # indeces_shape which is fed to decoder (encoder output)
        self.enc_output_shape = []
        # well use the same encoder/decoder from previous architectures
        self.encoder = nn.Sequential(conv(self.input_channels,32),#28x28
                                     conv(32,64,stride=2),#14x14
                                     conv(64,96,stride=2),#7x7
                                     conv(96,96, kernel_size=3),
                                     conv(96,128,stride=1),#3x3 #for 7x7, stride=1, otherwise set=2
                                     conv(128,128, kernel_size=3),
                                     conv(128,256,stride=1),#2x2 # for 4x4: 1
                                     conv(256,256, kernel_size=3),
                                     conv(256, self.embd_size, stride=1,padding=1,batch_norm=True),#1x1 #for 4x4:1 # for 2x2:1 #for 1x1:2
                                    # Print(),
                                    )
        # self.drp = nn.Dropout(0.1)
        # we use the followng formula to determine the output size here
        # ((h-1)*stride)+(kernel_size-2)*padding
        # (h=1,k=4,s=2,p=1)
        # !if encoder output ix 7x7 -simply increasing dims to see if we get clearer images,
        # !this is not ideal, we are just doing it for testing purposes, we get better result
        # with sharper images, but the architecture is clearly weak. also larger fmaps may
        # not result in good codebooks as we leardned in vae, (we need to test this with decoder(prior model)
        # and see if this is the case here as well. (yeah I can confirm the larger fmaps defnitely
        # result in cleared images,)
        # next buff up the architecture)
        # 07-1*2+2-2*1=12
        # 12-1*1+4-2*1=13
        # 13-1*2+4-2*1=26
        # 26-1*1+4-2*1=27
        # 27-1*1+4-2*1=28
        # since we no longer dealing with flattened input, we simply use a conv layer!
        self.decoder = nn.Sequential(conv(self.embd_size,256, kernel_size=3),
                                    conv(256,256, kernel_size=3),
                                    #  Print(),
                                    deconv(256,128,kernel_size=2,stride=2,batch_norm=True),#for 4x4: 2,2  #for 1x1:4,2  #for 7x7:2,2
                                    conv(128,128, kernel_size=3),
                                    #  Print(),
                                    deconv(128,96,kernel_size=4,stride=1,batch_norm=True), #for 4x4: 4,1  #for 1x1:4,2  #for 7x7:4,1
                                    conv(96,96, kernel_size=3),
                                    #  Print(),
                                    deconv(96,64,kernel_size=4,stride=2,batch_norm=True),  #for 4x4: 4,2  #for 1x1:4,2  #for 7x7:4,2
                                    conv(64,64, kernel_size=3),
                                    #  Print(),
                                    deconv(64,32,kernel_size=4,stride=1,batch_norm=True),  #for 4x4: 2,1  #for 1x1:2,2  #for 7x7:4,1
                                    conv(32,32, kernel_size=3),
                                    conv(32,32, kernel_size=3),
                                    #  Print(),
                                    deconv(32,self.input_channels, kernel_size=4,stride=1,batch_norm=False,act=nn.Sigmoid()),#28 #for 4x4:6,2 # for 1x1:4 #for 7x7: 4,1
                                    # Print(),
                                    )
        
        self.quantizer = Quantizer(self.embd_num, self.embd_size,beta_weight=self.beta, use_ema=self.use_ema)

    def forward(self, inputs):
        outputs = self.encoder(inputs)
        # print(f'{outputs.shape=}')
        # we can use this later on with our prior model 
        # which we use for generating new imaegs, it comes handy
        # when we change input image size, and it changes, we simply
        # use this instead of remembering the encoder outputshape!
        if not self.enc_output_shape:
            self.enc_output_shape = outputs.shape[2:]
        
        loss, quantized, prepelexity = self.quantizer(outputs)
        # print(f'{quantized.shape=}')
        recons = self.decoder(quantized)
        return loss, recons, prepelexity

model_test = VQVAE(input_channels=3, embd_num=100, embd_size=64, beta=0.2, use_ema=True)
x = torch.randn(size=(10,3,32,32)) # our model works with 28x28 and 32x32 just fine!
vq_loss,rec,perp = model_test(x)
print(f'{vq_loss.item()=:.4f} {rec.shape=}, {perp=} {model_test.enc_output_shape=}')

#%%

def select_dataset(dataset_name='mnist', batch_size=128, size=28, limited_samples=False, train_samplesize=1000, test_samplesize=100):
    if dataset_name.lower() == 'mnist':
        dataset_train = datasets.MNIST('MNIST', train=True, download=True,transform=transforms.ToTensor())
        dataset_test = datasets.MNIST('MNIST', train=False, download=True,transform=transforms.ToTensor())
    
    elif dataset_name.lower() in ['cifar','cifar10']:
        # for cifar10 a better architecture and training regime is required
        transformations_tr = transforms.Compose([transforms.Resize(size),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.ToTensor(),])
        transformations = transforms.Compose([transforms.Resize(size), transforms.ToTensor(),])
        dataset_train = datasets.CIFAR10('CIFAR10', train=True, download=True,transform=transformations_tr)
        dataset_test = datasets.CIFAR10('CIFAR10', train=False, download=True,transform=transformations)
    
    elif dataset_name=='celeba':
        transformations_tr = transforms.Compose([transforms.Resize(size),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.ToTensor(),])
        transformations = transforms.Compose([transforms.Resize(size),transforms.ToTensor(),])
        dataset_train = torchvision.datasets.CelebA('./data/','train',download=True,transform=transformations_tr)
        dataset_test = torchvision.datasets.CelebA('./data/','valid',download=True,transform=transformations)
    
    elif dataset_name=='anime':
        dataset_train = datasets.ImageFolder('./data/anime_characters/raw_dirs/')
        train_size = int(0.8 * len(dataset_train))
        test_size = len(dataset_train) - train_size
        # split the dataset
        dataset_train, dataset_test = torch.utils.data.random_split(dataset_train, [train_size, test_size])
        
        transformations_tr = transforms.Compose([transforms.Resize(size),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.Lambda(lambda img: img.convert("RGB")),  # convert to RGB
                                                 transforms.ToTensor(),])
        transformations = transforms.Compose([transforms.Resize(size),
                                              transforms.Lambda(lambda img: img.convert("RGB")),  # convert to RGB
                                              transforms.ToTensor(),])
        # subsets have dataset property, so we access it to assign transformations!
        dataset_train.dataset.transform = transformations_tr
        dataset_test.dataset.transform = transformations
    else:
        raise Exception(f'the input dataset {dataset_name} is not supported! choose between (mnist or cifar10)')
    
    if limited_samples:
        # temporary test to see how many samples may be insuffiecient to train 
        # a vqvae properly from scratch, we use Subset() to specify and grab the
        # number of suitable samples we want for our test
        dataset_train = torch.utils.data.Subset(dataset_train, list(range(train_samplesize)))
        dataset_test = torch.utils.data.Subset(dataset_test, list(range(test_samplesize)))
    
    dataloader_train = torch.utils.data.DataLoader(dataset_train,batch_size=batch_size,shuffle=True)
    dataloader_test = torch.utils.data.DataLoader(dataset_test,batch_size=batch_size,shuffle=False)

    return dataset_train, dataset_test, dataloader_train, dataloader_test

#train
def train(model:VQVAE, dataset_name, lr, epochs,batch_size, interval, device, img_size, save_recons_dir=None,limited_samples=False, train_samplesize=60_000, test_samplesize=10_000):
    
    timestamp_str = datetime.datetime.now().strftime("%H:%M:%S - %Y/%m/%d")
    timestamp = timestamp_str.replace(":","_").replace("/","_")
    model_checkpoint_name = f'vqvae_{dataset_name.upper()}_{"x".join(map(str, img_size))}_{timestamp}.ckpt'
    if save_recons_dir:
        ext = os.path.splitext(model_checkpoint_name)[-1]
        dir_name = model_checkpoint_name.replace(ext,"")
        recons_dir = os.path.join(save_recons_dir,dir_name)
    
    dataset_train, dataset_test, dataloader_train, dataloader_test = select_dataset(dataset_name=dataset_name, 
                                                                                    batch_size=batch_size,
                                                                                    size=img_size,
                                                                                    limited_samples=limited_samples,
                                                                                    train_samplesize=train_samplesize,
                                                                                    test_samplesize=test_samplesize)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, amsgrad=False)
    
    # improved learning rate scheduler
    warmup_epochs = 5
    total_steps = len(dataloader_train) * epochs
    warmup_steps = len(dataloader_train) * warmup_epochs
    
    # added later to see how much improvement we can get out of our curent model!
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))
    
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    
    if warmup_steps > 0:
        # use 1e-8 as a very small start_factor to avoid potential issues with exactly 0
        scheduler_warmup = torch.optim.lr_scheduler.LinearLR(optimizer,
                                                             start_factor=1e-8,
                                                             end_factor=1.0,
                                                             total_iters=warmup_steps) 
        # decay to 1% of peak LR
        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                      T_max=total_steps - warmup_steps,
                                                                      eta_min=lr * 0.01)
        # combine schedulers
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_cosine], milestones=[warmup_steps])
        print(f"Using Linear Warmup ({warmup_steps} steps) + Cosine Annealing ({total_steps - warmup_steps} steps) scheduler.")
    else:
        # Only Cosine Annealing if no warmup
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=lr * 0.01)
        print(f"Using Cosine Annealing ({total_steps} steps) scheduler (no warmup).")


    # without this loss will decrease a lot but the result isnt as good as when
    # we normalize the loss, and take the whole dataset into account!
    #! whats really ahppening here? why do I need to take the whole dataset into account like this?
    # what does this do?
    # min_v = dataloader_train.dataset.data.min()
    # max_v = dataloader_train.dataset.data.max()
    # print(f'{min_v=}')
    # print(f'{max_v=}')
    
    # if isinstance(dataloader_train.dataset.data, np.ndarray):
    #     data_variance_train,data_variance_val = [np.var(loader.dataset.data/max_v) for loader in [dataloader_train,dataloader_test]] 
    # else:
    #     data_variance_train,data_variance_val = [torch.var(loader.dataset.data/max_v).item() for loader in [dataloader_train,dataloader_test]]
    # test of normalization efficacy!
    data_variance_train = data_variance_val =1
    
    # pixel_values = []
    # pixel_values = [img.numpy() for img,_ in dataset_train]
    # pixel_values = np.concatenate([img.flatten() for img in pixel_values])
    # data_variance = np.var(pixel_values)  

    print(f'Experiment Date:     {timestamp_str}')
    print(f'Checkpoint:          {model_checkpoint_name}')
    print(f'Dataset:             {dataset_name.upper()}')
    print(f'Limited Samples:     {"\033[91m" + str(limited_samples) + "\033[0m" if limited_samples else limited_samples}')# make it red so it stands out!
    print(f'Train size:          {len(dataloader_train.dataset):,}')
    print(f'Test size:           {len(dataloader_test.dataset):,}')
    print(f'Epochs:              {epochs}')
    print(f'BatchSize:           {batch_size}')
    print(f'embeddings_num:      {model.embd_num}')
    print(f'embedding_size:      {model.embd_size}')
    print(f'use_ema:             {model.use_ema}')
    print(f'beta/commmitment:    {model.beta}')
    print(f'optimizer:           {optimizer}')
    print(f'scheduler:           {scheduler.state_dict()}')
    print(f'interval:            {interval}')
    print(f'data variance[train]:{data_variance_train:.4f}')
    print(f'data variance[val]:  {data_variance_val:.4f}')
    
    total_reconstruction_errors = []
    total_vqlosses = []
    total_perplexities = []
    total_losses=[]
    total_val_losses=[]
    best_loss = float("inf")
    
    for epoch in range(epochs):
        model.train()
        reconstruction_errors = []
        vqlosses = []
        perplexities = []
        losses=[]
        val_losses=[]
        for i, (imgs, _) in enumerate(dataloader_train):
            imgs = imgs.to(device)
            vq_loss, imgs_rec, perplexity = model(imgs)
            
            #! normalzie the loss
            reconstruction_error = F.mse_loss(imgs_rec, imgs) / data_variance_train
            # reconstruction_error = F.binary_cross_entropy(imgs_rec, imgs) / data_variance_train
            loss = reconstruction_error + vq_loss
            
            losses.append(loss.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
              
            reconstruction_errors.append(reconstruction_error.item())
            vqlosses.append(vq_loss.item())
            perplexities.append(perplexity.item())

            if i%interval == 0:
                print(f'Epoch: {epoch}/{epochs}'
                      f' | Loss: {np.mean(losses):.4f}'
                      f' | Recons-Error: {np.mean(reconstruction_errors):.4f}'
                      f' | VQ-Loss: {np.mean(vqlosses):.4f}'
                      f' | Perplexity: {np.mean(perplexities):.4f}'
                      f' | LR: {scheduler.get_last_lr()[-1]:.6f}')
    
        # scheduler.step()
       
        with torch.no_grad():
            model.eval()
            for imgs, _ in dataloader_test:
                imgs = imgs.to(device)
                vq_loss, imgs_rec, perplexity = model(imgs)
                val_reconstruction_error = F.mse_loss(imgs_rec, imgs) / data_variance_val
                # val_reconstruction_error = F.binary_cross_entropy(imgs_rec, imgs) / data_variance_val
                val_loss = val_reconstruction_error + vq_loss
                val_losses.append(val_loss.item())

        # display reconstruction performance!
        fname=None
        if save_recons_dir:
            # ext = os.path.splitext(model_checkpoint_name)[-1]
            # dir_name = model_checkpoint_name.replace(ext,"")
            # recons_dir = os.path.join(save_recons_dir,dir_name)
            if not os.path.exists(recons_dir):
                os.makedirs(recons_dir)
            fname = f'{recons_dir}/recons_{epoch}.jpg'

        view_reconstructions(model, dataloader_test, fname=fname)
        
        # keep track of the stat for each epoch as well
        mean_loss = np.mean(losses)
        mean_val_loss = np.mean(val_losses)
        mean_reconstruction_errors = np.mean(reconstruction_errors)
        mean_vqloss = np.mean(vqlosses)
        mean_perplexity = np.mean(perplexities)
        
        total_losses.append(mean_loss)
        total_val_losses.append(mean_val_loss)
        total_reconstruction_errors.append(mean_reconstruction_errors)
        total_vqlosses.append(mean_vqloss)
        total_perplexities.append(mean_perplexity)

        print(f'Epoch: {epoch}/{epochs}'
              f' | Loss: {mean_loss:.4f}'
              f' | Val-Loss: {mean_val_loss:.4f}'
              f' | Recons-Error: {mean_reconstruction_errors:.4f}'
              f' | VQ-Loss: {mean_vqloss:.4f}'#mean_loss-mean_reconstruction_errors
              f' | Perplexity: {mean_perplexity:.4f}'
              f' | LR: {scheduler.get_last_lr()[-1]:.6f}')

        #! use a validation instead?
        if mean_val_loss < best_loss:
            best_loss = mean_val_loss
            # TODO: remove checkpoint related info such as optimizer/scheduler state_dicts
            # and save only the model weights, rename to .pt so it takes less space!
            torch.save({'dataset':dataset_name,
                        'batchsize':batch_size,
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        # 'optimizer': optimizer.state_dict(),
                        # 'scheduler':scheduler.state_dict(),
                        'val_loss': best_loss,
                        'train_loss': mean_loss,
                        'perplexity':mean_perplexity,
                        'enc_output_shape':model.enc_output_shape,
                        'img_size':img_size,
                        'limited_samples':limited_samples,
                        'train_samplesize':train_samplesize,
                        'test_samplesize':test_samplesize,
                        'model_config':{
                          'beta':model.beta,
                          'use_ema':model.use_ema,
                          'embd_num':model.embd_num,
                          'embd_size':model.embd_size,
                          'input_channels':model.input_channels,
                          },
                       }, model_checkpoint_name.replace('.ckpt','_best.pt'))
            print(f'Best model with loss={best_loss:.4f} saved at epoch {epoch}!')
        
        # save the last model
        torch.save({'dataset':dataset_name,
                    'batchsize':batch_size,
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler':scheduler.state_dict(),
                    'val_loss': best_loss,
                    'train_loss': mean_loss,
                    'perplexity':mean_perplexity,
                    'enc_output_shape':model.enc_output_shape,
                    'img_size':img_size,
                    'limited_samples':limited_samples,
                    'train_samplesize':train_samplesize,
                    'test_samplesize':test_samplesize,
                    'model_config':{
                      'beta':model.beta,
                      'use_ema':model.use_ema,
                      'embd_num':model.embd_num,
                      'embd_size':model.embd_size,
                      'input_channels':model.input_channels,
                      },
                  }, model_checkpoint_name)
    
    # create gifs from recons
    create_gifs(recons_dir)
    return total_losses,total_val_losses, total_reconstruction_errors, total_perplexities

@torch.no_grad()
def view_reconstructions(model:VQVAE, dataloader, fname=None):
    model.eval()
    (imgs, labels) = next(iter(dataloader))
    imgs = imgs.to(device)
    vq_encoder_output = model.encoder(imgs)
    _, quantize, _ = model.quantizer(vq_encoder_output)
    reconstructions = model.decoder(quantize)
    # for celeba only
    if labels[0].ndimension()>0:
       labels = ['N/A' for _ in range(imgs.size(0))]
    # view_images(imgs, labels, normalized=False,fname_to_save_as=None)
    # extract epoch from fname and use it to mark each image
    epoch = os.path.splitext(fname)[0].split('_')[-1]
    view_images(reconstructions, labels, normalized=False,fname_to_save_as=fname,title=f'Epoch {int(epoch):0}')

# lets also make a gif_creator!
import re
# to create gifs, we need our images to be ordered!
# we cant use sorted(), because it cant sort properly 
# when we have numbers, it will go frm 3 to 30!
# and 4 to 40, etc because it compares character by character!
# so we need to write a custom filter based on numbers!
def numerical_sort_key(filename, _re=re.compile(r'(\d+)')):
    # first we check if the fname has number in it extract it
    # if it doesnt come with a number, we take the text itself
    # and sorted() compares it alphabatically with others 
    return [int(text) if text.isdigit() else text.lower()
            for text in _re.split(filename)]

# needed for our gif display
from PIL import Image
import IPython.display as ipd

def is_notebook():
    try:
        shell = get_ipython().__class__.__name__
         # check if we are executing from jupyter notebook or an ipython terminal
        return shell in ['ZMQInteractiveShell', 'Shell'] 
    except NameError:
        return False  # Probably standard Python interpreter

# create gifs
def create_gifs0(dir_path, frame_interval=90, loop=0):
    imgs = [Image.open(os.path.join(dir_path,fname)) 
            for fname in sorted(os.listdir(dir_path),key=numerical_sort_key) if fname.endswith('.jpg')]
    
    dirname = os.path.basename(os.path.normpath(dir_path))
    gif_path = os.path.join(dir_path, f'{dirname}.gif')
    
    imgs[0].save(gif_path, format='GIF', append_images=imgs[1:], 
               save_all=True, duration=frame_interval, loop=loop)
    
    # display the gif inside jupyernotebook
    if is_notebook():
        ipd.display(ipd.Image(filename=gif_path))
    else:
        print(f'gif created successfully!')
        plt.show()

import matplotlib.animation as animation
# matplotlib animation module does a better job, it takes less space, so I guess I'll use this one
# than my previous implemetation-fps=600, and interval=90 results seem to have roughly the same speed 
# (because 1/600 = 0.0016 second(or 1.6 milliseconds) for each frame when using fps,
# likewise to get 600 fps with interval we need 1/1.6 = 600 fps !
# however, in my experience interval=90 feels like fps=600! so you might want to go for that!
#!todo choose one over the other!
# )
def create_gifs(dir_path, frame_interval=90, repeat_delay=1000, loop=True, fps=None, figsize=(6,8)):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    imgs = [Image.open(os.path.join(dir_path,fname)) 
            for fname in sorted(os.listdir(dir_path),key=numerical_sort_key) if fname.endswith(('.jpg', 'jpeg','.png'))]
    dirname = os.path.basename(os.path.normpath(dir_path))
    gif_path = os.path.join(dir_path, f'{dirname}.gif')
    def animate(i):
        ax.set_title(str(i))
        ax.clear()
        ax.axis('off')
        ax.imshow(imgs[i])
    
    fig.tight_layout()
    anim = animation.FuncAnimation(fig, animate, frames=len(imgs),
                                   interval=frame_interval, 
                                   repeat=loop, 
                                   repeat_delay=repeat_delay)
    if fps and frame_interval:
        print(f'Warning, frame_interval wont be used for gif creation!'
              f'either use FPS or frame_interval for gif creation (set one to None!)'
              'frame_interval is used for delay inside jupyternotebook'
              'while FPS is used for gif creation. Only if FPS=None,frame_interval is used'
              'otherwise, FPS superceeds frame_interval in gif creation')
        # print(f'Note: FPS is used for gif creation while ')
    
    # save the git using pillow backedn
    anim.save(gif_path, writer="pillow",fps=fps)
    # display the gif inside jupyernotebook
    if is_notebook():
        ipd.display(ipd.Image(filename=gif_path))
    else:
        print(f'gif created successfully!')
        plt.show()
    
# dataset = 'anime'
dataset = 'cifar10'
# batch_size = 128
dataset_train, dataset_test, dataloader_train, dataloader_test = select_dataset(dataset_name=dataset,
                                                                                batch_size=batch_size,
                                                                                size=(64,64))
(imgs, labels) = next(iter(dataloader_test))
view_images(imgs,labels, rows=13,cols=10, title=f'{dataset}')
#%%
# we first start with mnist to see if our implementation is ok 
# (sidenote: I actually faced quit a lot of issues and switching to mnist helped a lot. 
# I was initially using with cifar10 directly. after mnist, I used celeba, because its 
# simpler than cifar10 and one can more easily identify patterns, and whether
# you are dealing with blobs! or meaningful patterns. because celeba is basically aligned and cropped
# images of faces, its way easier to spot issues than tiny cifar10 where different classes can be
# very hard to see, and cant decide which part of thenetwork is faulty! (more on this later))

#todo why doesnt anime work!? it gives me blurry blacknwhite recons!!?
# our anime dataset is just too small to work! try cifar10 or other datasets with
# limited_samples=True and see the resulut (basically aroudn 1000 samples wont
# work if we train a model from scratch!) asign of small/inadequate training set
# is that the images will be discolored, almost black and white, becoming monocolors
# lots of yellow, brownish colors, and needless to say images are very blury
# test with limited samples and you'll see what I mean!
dataset = 'cifar10' #anime # celeba #cifar10
#! enshaallah tomorrow, run cifar1032x32, mnist64x64, celeba64x64 and call it a day!
img_size=(64,64)# larger image sizes, result in more detailed generations!
# whether to use limited samples (for testing purposes)
# to see how the model performs with different number of samples!
# based on my prelimenary tests, a training size of 10K seems to be bare minimum
# to give us some what borderline reconstructions, 5K would be insufficient and 
# colors wont form, we would get discolored, monocolor, black/yellow/brownish colors,
# with 7000 images, we get some more colors, images are a bit more visible/ but still
# theres a lot of monocolors, yellow, brown, blue, gray colors, not vivid colors at all
# images are not formed properly, but for some classes, we can see the objects sillohet!
# (we get Epoch: 99/100 | Loss: 0.1285 | Val-Loss: 0.1179 | Recons-Error: 0.0079 | VQ-Loss: 0.1207 | Perplexity: 3.4678 | LR: 0.000010)
# starting with 8000, we see normal colors being back, and images start to look normal!
# (by normal I mean compared to previous cases, not all are still well formed! but its much
# better than before, but it still lacking by a large extend!)
# we get Epoch: 99/100 | Loss: 0.0355 | Val-Loss: 0.0322 | Recons-Error: 0.0040 | VQ-Loss: 0.0315 | Perplexity: 10.4210 | LR: 0.000010
# note that im not saying 10k is enough to get something nice, not at all, what im saying
# 10k seems to be the limit that doesnt give us crappy images that cant even be bothered with!
# the colors seem accurate, objects are formed properly for the most part, details are kinda there
# but do not much, cuz images are very blurry still and look smudged! the image size definitely matters here,
# im working with 64x64 here! (aside from the nature of the images, obviously complex images will be tougher
# and simpler images/concepts should require much less effort to get this right! in fact mnist seems to 
# be ok with 1000 samples! cifar10 on the other hand is much more complex and is composed of natural images 
# so it obviously requires way more training data!) with 10k cifar10 this is what I get:
# Epoch: 99/100 | Loss: 0.0182 | Val-Loss: 0.0176 | Recons-Error: 0.0026 | VQ-Loss: 0.0156 | Perplexity: 13.1167 | LR: 0.000010
limited_samples=True
training_samplesize=8000
test_samplesize=100

input_channels = 1 if dataset=='mnist' else 3

device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size=128
epochs = 100#100
interval = 1000
# when learning rate is too large, we usually see artifacts in early stages
# of training (which tells us lr might be high!)
lr=0.001 #0.001
milestones=[120]
embd_num=512
embd_size=256#128
# commitment loss beta/weight
# lower values result in worse loss and recons error! (I tried 0.2)
beta=0.25 #0.25
# use ema for quantzier embeddings update
use_ema=False

#!edit 
#!use_ema doesnt make any difference on quality of recons 
# apparently when model is weak?!

model = VQVAE(input_channels=input_channels, embd_num=embd_num, embd_size=embd_size, beta=beta, use_ema=use_ema)
model.to(device)
# optimizer = optim.AdamW(model.parameters(), lr=lr, amsgrad=False)
# scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones)

train_losses, val_losses, train_recons_errors, train_perplexities = train(model,
                                                                        dataset,
                                                                        lr,
                                                                        # optimizer, 
                                                                        # scheduler,
                                                                        epochs, 
                                                                        batch_size,
                                                                        interval,
                                                                        device,
                                                                        img_size,
                                                                        save_recons_dir='./results',
                                                                        limited_samples=limited_samples,
                                                                        train_samplesize=training_samplesize,
                                                                        test_samplesize=test_samplesize)

#%%
# load the model
# TODO: remove the old models cuz they take up space!
# ckpt_name = 'vqvae_MNIST_12_45_15 - 2025_03_16.ckpt'
# ckpt_name = 'vqvae_CIFAR10_20_40_21 - 2025_03_17.ckpt' # this was trained with wrong variance normalization!
# ckpt_name = 'vqvae_CIFAR10_22_34_52 - 2025_03_17.ckpt'
# ckpt_name = 'vqvae_CIFAR10_10_13_46 - 2025_03_18.ckpt'# with beefed up resblock!
# ckpt_name = 'vqvae_CIFAR10_10_53_09 - 2025_03_18.ckpt'# with beefed up deconv-overfiitng-colors not accurate-very dimmed and undersaturated!(eg.g red is brown!!)
# ckpt_name = 'vqvae_CIFAR10_11_28_22 - 2025_03_18.ckpt'#AdamW
# ckpt_name = 'vqvae_CIFAR10_11_22_30 - 2025_03_19.ckpt'
# ckpt_name = 'vqvae_CIFAR_11_11_22 - 2025_03_23.ckpt' # no variance normalization
# ckpt_name = 'vqvae_CIFAR_11_42_17 - 2025_03_23.ckpt'
# ckpt_name = 'vqvae_CELEBA_17_32_39 - 2025_03_24.ckpt'#celeb32
# ckpt_name = 'vqvae_CELEBA_12_45_12 - 2025_03_25.ckpt'#celeb64, very good result!
# ckpt_name = 'vqvae_CIFAR10_20_17_56 - 2025_03_25.ckpt'# cifar64x64 (codesize =16x16) works great!
# codesize=8x8 - to see if codesize effects the latent variables
# because previously when we trained with 32x32, the simple generation would create
# somewhat meanigful outputs, like for celeba, the faces could be easily identified
# though they were caricaturish!, while the normal prior based generation sucked (because
# of our bug!) but when we used 64x64, the normal generation got decent but simple 
# generation seemed just like pure noise! here im trying to see if the codebook size
# has something to do with this, that is, by itself, codebook of 8x8 is trained/developed
# much better than 16x16 codebook, consequently resulting in maningful generation
# (non autoregressive), whereas for the 16x16 one, the codebook was low quality and
# we needed a prior model to make it work! if this is the case, it proves our initial 
# point we observed in vae before, that is, larger encoder output results in better
# reconstructions, but may not be as developed (this is related to our architecture 
# of course, as we dont have more layers working on certain featuremap sizes. 
# probably if we use better architecture, use more layers for each featuremap size,
# we wont see this issue for larger fmaps. but lets see how this goes!)
# ok it seems our estimate is correct. 32x32 gives somewhat identifiable non-autoregressive 
# generations, not complete noise! trying with 64x64 to see how it goes again
# I cant replicate this anymore! I dont know why I cant get meanigful generations out of
# simple_generation function! 
#
# ckpt_name = 'vqvae_CIFAR10_32x32_20_31_27 - 2025_03_26.ckpt'
# ckpt_name = 'vqvae_CIFAR10_64x64_23_21_12 - 2025_03_26.ckpt'
# ckpt_name = 'vqvae_CIFAR10_64x64_22_00_58 - 2025_04_01.ckpt'
# ckpt_name = 'vqvae_CIFAR10_32x32_08_21_28 - 2025_04_03.ckpt'
# ckpt_name = 'vqvae_CIFAR10_64x64_10_52_50 - 2025_04_03.ckpt'
# ckpt_name = 'vqvae_CIFAR10_64x64_10_52_50 - 2025_04_03_best.ckpt'
# ckpt_name = 'vqvae_CIFAR10_32x32_10_14_55 - 2025_04_03.ckpt'
# ckpt_name = 'vqvae_CELEBA_64x64_12_25_31 - 2025_04_03.ckpt'
# ckpt_name = 'vqvae_CELEBA_32x32_13_51_11 - 2025_04_03.ckpt'
# ckpt_name = 'vqvae_MNIST_64x64_15_58_57 - 2025_04_03.ckpt'
# ckpt_name = 'vqvae_MNIST_32x32_15_20_19 - 2025_04_03.ckpt'

# performs very very good ! increased embdsz actually results in way smaller loss
# abd BPD! I noticed the perplexity is much much lower though! but the generation
# nonetheless is much much better!
ckpt_name = 'vqvae_CIFAR10_64x64_14_24_34 - 2025_04_04.ckpt' #with embd=256
# ckpt_name = 'vqvae_CIFAR10_64x64_14_24_34 - 2025_04_04_best.ckpt' #with embd=256

# ckpt_name = 'vqvae_CELEBA_64x64_20_10_23 - 2025_04_04.ckpt' # with embd=256,64x64
# ckpt_name = 'vqvae_CELEBA_64x64_20_10_23 - 2025_04_04_best.pt'

#todo add train and test sizes so the rest of the pipeline also use the same
# number of samples for prior training. 

checkpoint = torch.load(ckpt_name, weights_only=False)
model_config = checkpoint['model_config']
dataset = checkpoint['dataset']
limited_samples = checkpoint.pop('limited_samples', False)
img_size = checkpoint.pop('img_size',None)
if not img_size:
    img_size = tuple(int(n) for n in ckpt_name.split('_')[2].split('x'))
# enc_output_shape = model_config.pop('enc_output_shape',None)
enc_output_shape = checkpoint['enc_output_shape']
perplexity = checkpoint.pop('perplexity',None)

device = 'cuda'
model = VQVAE(**model_config)
model.to(device)

model.load_state_dict(checkpoint['state_dict'])
model.enc_output_shape = enc_output_shape

print(f'dataset    : {checkpoint['dataset'].upper()}')
print(f'Epoch      : {checkpoint['epoch']}')
print(f'img_size   : {img_size}')
print(f'Limited samples    : {limited_samples}')
print(f'Encoder output size: {tuple(model.enc_output_shape)}')

for k,v in checkpoint['model_config'].items():
    print(f'{k:<10} : {v}')

print(f'\ntrain_loss : {checkpoint['train_loss']:.6f}')
print(f'val_loss   : {checkpoint['val_loss']:.6f}')
print(f'perplexity : {perplexity:.6f}' if perplexity else 'perplexity : N/A')

# dataset_train, dataset_test, dataloader_train, dataloader_test = select_dataset(dataset_name=checkpoint['dataset'],
#                                                                  batch_size=checkpoint['batchsize'])

#%%
# create gifs and show images
dirpath='./results/vqvae_CIFAR10_64x64_13_31_17 - 2025_04_06/'
dirpath = './results/vqvae_MNIST_64x64_17_23_16 - 2025_04_07'
create_gifs0(dirpath,
            delay=90,
            # figsize=(12,16)
            )
#%%
dirpath='./results/vqvae_CIFAR10_64x64_13_31_17 - 2025_04_06/'
dirpath = './results/vqvae_MNIST_64x64_17_23_16 - 2025_04_07'
create_gifs(dirpath,
            interval=90,
            delay=300,
            fps=None,
            # figsize=(12,16)
            )
#%%
import pandas as pd
# for logs in [train_losses, val_losses, train_recons_errors, train_perplexities]:
#     pd.DataFrame(logs).plot()
#     plt.show()
# pd.DataFrame(train_recons_errors).plot()
# plt.show()
# pd.DataFrame(train_perplexities).plot()
# plt.show()

# print(f'{len(train_losses)=}')
# print(f'{len(val_losses)=}')
# print(f'{len(train_recons_errors)=}')
# print(f'{len(train_perplexities)=}')
def display_logs(train_losses, val_losses, train_recons_errors, train_perplexities):
    for label, logs in zip(["Train Loss", "Val Loss", "Train Recon Error", "Train Perplexity"],
                            [train_losses, val_losses, train_recons_errors, train_perplexities]):
        df = pd.DataFrame(logs)
        ax = df.plot()  # Create the plot and get the axis
        ax.set_xlabel("Epochs")  # Label x-axis
        ax.set_ylabel("Value")  # Label y-axis
        ax.set_title(label)  # Set title
        plt.show()

display_logs(train_losses, val_losses, train_recons_errors, train_perplexities)
#%%
# taken from Aäron van den Oord implementation (link given before)
# use umap for latent space visualization, umap is better than tsne
#! explain a bit more
from umap.umap_ import UMAP # pip install umap-learn
@torch.no_grad()
def view_results(model,train_dataloader, val_dataloader):
    model.eval()
    for name, dataloader in zip(['training data','validation data'],[train_dataloader, val_dataloader]):
        print(f'Using {name}:')
        (imgs, labels) = next(iter(dataloader))
        imgs = imgs.to(device)
        vq_encoder_output = model.encoder(imgs)
        _, quantize, _ = model.quantizer(vq_encoder_output)
        reconstructions = model.decoder(quantize)
        # print(f'{reconstructions.shape=}')
        # print(f'{labels.shape=} {labels.ndimension()=}')
        # for celeba only
        if labels[0].ndimension()>0:
            labels = torch.ones((imgs.size(0),1))
        view_images(reconstructions, labels, normalized=False)
        view_images(imgs, labels)

    proj = UMAP(n_neighbors=3,
                min_dist=0.1,
                metric='cosine').fit_transform(model.quantizer.embeddings.weight.data.cpu())
    plt.scatter(proj[:,0], proj[:,1], alpha=0.3)

_, _, dataloader_train, dataloader_test = select_dataset(dataset_name=dataset,
                                                             batch_size=batch_size,
                                                             size=img_size)
view_results(model, dataloader_train, dataloader_test)
# now alhamdolelah finally we got pretty great reconstructions with 64x64
# image dimensions. the actual reason behind this is the larger encoder output shape
# that is in the encoder, if we use much larger featuremaps, the decoder can
# much more easily reconstruct the image with much more details. initially
# used 7x7 fmaps, and it gave us blurry images, no matter what we did, we couldnt
# improve it, but when we simply increased the image size to 64x64, we got wayyy 
# better result. this is in line with our previous observations in vanila vae, where
# the larger fmaps would result in way better reconstructions (the issue there though was
# that larger fmaps wouldnt allow the network to learn good features and generation
# was very bad (we faced posterior collapse. but partly that was due to our simplistic 
# architecture, I wonder if we see improvements by simply using this architecture here!
# all in all, we know for a fact that larger fmaps in encoder is key to get sharp images!))
#%%
# now to be able to generate images, as we stated before, we need a prior model
# todo: explain 

def get_latent_codes(model, dataloader):
    model.eval()
    all_indices = []
    all_labels = []
    with torch.no_grad():
        for data, labels in dataloader:
            data = data.to(device)
            encoder_output = model.encoder(data)
            
            encoder_output = encoder_output.permute(0,2,3,1).contiguous()
            encoder_flatten = encoder_output.view(-1, model.embd_size)
            distances = torch.cdist(encoder_flatten, model.quantizer.embeddings.weight)
            indices = torch.argmin(distances, dim=1)
            
            # reshape to (batch_size, H, W)
            indices = indices.view(data.shape[0], encoder_output.shape[1], encoder_output.shape[2])
            all_indices.append(indices.cpu())
            # for use in conditional generation
            all_labels.append(labels)
            
    all_indices =  torch.cat(all_indices, dim=0)  # Shape: [num_samples, H, W]
    all_labels = torch.cat(all_labels, dim=0)
    return all_indices,all_labels

# prior model i.e. pixelcnn (we can use a transformer based model as well, 
# (we could also use gan for this!))
#
# heres our pixelcnn model (works very bad, see the next model!)
class PixelCNN_old(nn.Module):
    def __init__(self, num_embds, embedding_size=128):
        super().__init__()
        self.num_embds = num_embds
        # self.H, self.W = input_shape
        self.embedding_size = embedding_size
        
        # Embedding layer: converts indices to dense vectors
        self.embedding = nn.Embedding(num_embds, embedding_size)
        
        # Masked convolutions now operate on embeddings
        # the 3 layer version performs worse, when increased the layers
        # it got better, so prior network design is also very critical
        # on getting a good generation.
        self.layers = nn.Sequential(
            MaskedConv2d('A', embedding_size, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # nn.Dropout(0.01),
            MaskedConv2d('B', 64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # nn.Dropout(0.01),
            MaskedConv2d('B', 128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # nn.Dropout(0.01),
            MaskedConv2d('B', 256, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            # nn.Dropout(0.02),
            MaskedConv2d('B', 384, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # nn.Dropout(0.02),
            nn.Conv2d(512, self.num_embds, kernel_size=1)  # Output logits over codebook indices
        )

    def forward(self, x):
        # x: (batch_size, H, W) containing integer indices
        x = self.embedding(x)  # (batch_size, H, W, embedding_size)
        x = x.permute(0, 3, 1, 2)  # (batch_size, embedding_size, H, W)
        logits = self.layers(x)  # (batch_size, num_embds, H, W)
        return logits

class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, in_channels, out_channels, kernel_size=3, padding=1, dilation=1):
        super().__init__(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        
        self.register_buffer('mask', torch.ones_like(self.weight))
        
        _, _, h, w = self.weight.size()
        center_h, center_w = h // 2, w // 2
        
        # we need to create two masks, A, and B like the paper.
        # the idea is, where are trying to create an autoregressive cnn layer
        # where a pixel can not see its current value or future one, only the 
        # previous values, and the current and future values are predicted using
        # previous values (hence the name autoregressive)
        # to do this on an image, we actually implement masks on the kernels themselevs
        # so when we apply them, this logic is applied on all pixels in an image.
        # back to our implementation, so for our first mask, we need to make sure 
        # a pixel does not see itself so we zero out the current pixel and all the 
        # future pixels (to the right and below).
        # we must use this mask on the first layer only. for subseuqent layers we
        # must use the second mask, the B mask.
        # For the secodn mask, we need to allow access to the current pixel but 
        # block future pixels. the difference from mask A is that the center pixel is allowed here.
        # lets visualize it ,suppose this is our mask initially
        # 1 1 1
        # 1 1 1
        # 1 1 1 
        # now to make it work as Mask A, we must change it so current pixel
        # and future pixels are disabled. in other words, they cant be used
        # to calculate the value of current pixels (when we apply it on an image)
        # so the center of our mask needs to be set to 0, so we would have
        # mask is 3x3, so 3//2=1, 3//2=1 so row 1 and col1 and all future pixels in last
        # row(i.e 2) are set to 0
        # 1 1 1 
        # 1 0 0 
        # 0 0 0
        # this leaves us with previous pixels only.
        # now for MasK B, we want to have previous pixels plus the current pixel
        # but still future pixels must be disabled. 
        # if this is our initial mask state for Mask B: 
        # 1 1 1
        # 1 1 1
        # 1 1 1 
        # we want sth like this:
        # 1 1 1
        # 1 1 0
        # 0 0 0
        # which basically means we want to do just like Mask A, except that now
        # we want to have current pixel as well. 
        # so we disable all following rows so they dont affect current pixel
        
        # make sure the mask is all 1s, so we can selectively disable parts of it
        self.mask.fill_(1)
        #sidenote: note that our mask is 4D and not just 2D, as we need to apply this
        # on a conv ouput which has input and output dims.
        
        if mask_type == 'A':
            # zero out the current pixel (at position (center_h, center_w)) and all pixels
            # to its right in the same row.
            # basically prevent future pixels in the same row from influencing the current pixel
            # (i.e. when predicting a pixel, the network cannot see the current pixel 
            # or any pixel that comes after it)
            self.mask[:, :, center_h, center_w:] = 0
            # prevent pixels below from influencing the current pixel (zero out rows>=2 )
            self.mask[:, :, center_h+1:, :] = 0
        elif mask_type == 'B':
            # zero out only the pixels to the right of the current pixel in the same row,
            # keeping the current pixel itself active.(i.e. prevent future pixels in the 
            # same row from influencing the current pixel)
            self.mask[:, :, center_h, center_w+1:] = 0
            # prevent pixels below from influencing the current pixel
            self.mask[:, :, center_h+1:, :] = 0

    def forward(self, x):
        # apply the mask to weights during forward pass
        return self._conv_forward(x, self.weight * self.mask, self.bias)
        # this was wrong and caused in total failure of the model, i would get
        # sold colors, like red, blue, whenever i wanted to decode and generate
        # an image!
        # self.weight.data *= self.mask  # Apply mask
        # return super().forward(x)
#sidenote: 
# a very good introduction on pixelcnn and maskedconvolutions:
# https://github.com/pilipolio/learn-pytorch/blob/master/201708_ToyPixelCNN.ipynb
# https://www.codeproject.com/Articles/5061271/PixelCNN-in-Autoregressive-Models
# https://jrbtaylor.github.io/conditional-pixelcnn/
# https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial12/Autoregressive_Image_Modeling.html
# !edit,  it needs more work, its not complete yet!
# see visualize_maskedcnn.py to see this in action

# temp-test improved architecture: 
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0, mask_type='B',dilation=1):
        super().__init__()
        self.block = nn.Sequential(MaskedConv2d(mask_type, in_channels, out_channels, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(),
                                  # nn.Dropout2d(dropout_rate),
                                  # pad=2,dialation=2 should give us a larger receptive field (5x5)
                                  # lets see if it helps! cuz theoretically a larger receptive field
                                  # should help the model capture long-range dependencies across the
                                  # image plane/spatial domain. the default is dialation=1.
                                  # (my initial tests didnt show any changes, need more experiments!)
                                  # !explain more why thats the case
                                   MaskedConv2d(mask_type,
                                                out_channels,
                                                out_channels, 
                                                kernel_size=3,
                                                #!(since we only want pad=2 dilation=2,
                                                #! I set padding to dilation because we only do dilation=2!in our tests)
                                                padding=dilation, 
                                                dilation=dilation),
                                   
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(),
                                   nn.Dropout2d(dropout_rate))
        
        #! lets test with no conv,bn on residuals. 
        self.skip = nn.Sequential()
        if in_channels != out_channels:
            self.skip = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1),
                                      # interestingly having bn lowers the loss pretty well!
                                      # who would have thought!!!!
                                      nn.BatchNorm2d(out_channels)
                                      )
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        residual = x
        # x = F.relu(self.bn1(self.conv1(x)))
        # x = self.bn2(self.conv2(x))
        # output = x+self.skip(residual)
        # output = F.relu(output)
        # remove the bn from residual
        # add the residual and block and then add a bn and a relu at the end!
        output = self.block(x) + self.skip(residual)
        #! OK the result s worse now, i dont know its beause of this change
        # or because I change optimizer from Adamw to adam! 
        # next test with the line below uncommented and if it fails
        # then revert back to adamw! (check why adamw works better in that case)-adamw is better
        output = F.relu(output)# using this, perforamnce is a bit better! not much though!
        return output 

class PixelCNN(nn.Module):
    def __init__(self, num_embds, embedding_size=128, num_class=10, make_conditional=True, dropout_rate=0.1):
        super().__init__()
        self.num_embds = num_embds
        # initialize it from input so its set dynamically!
        #!i guess i'll be using the vqvae model instead of this
        # because for generation we will be using vqvae model anyway
        # moreover, for this to have meaningful values, our prior must
        # do at least one forward pass, which again for generation
        # we simply dont do, because we need H,W before that! so I guess
        # I remove this alrogether?!
        self.input_shape = []
        # self.H, self.W = input_shape
        self.embedding_size = embedding_size
        # number of classes, used to condition generation on the class
        self.num_class = num_class
        # make model conditional 
        self.make_conditional = make_conditional
        
        self.fc_label_embedding = nn.Linear(num_class, embedding_size)
        
        self.embedding = nn.Embedding(num_embds, embedding_size)
        
        self.conv_input_size = self.embedding_size*2 if make_conditional else self.embedding_size
        
        # the first layer/block must be type A, the rest are B
        # basically mask A blocks the current pixel and all future pixels, 
        # while mask B allows the current pixel but blocks future ones.
        self.initial_conv = nn.Sequential(MaskedConv2d('A', self.conv_input_size, 128, kernel_size=7, padding=3),
                                          nn.BatchNorm2d(128),
                                          nn.ReLU(),
                                         #nn.Dropout2d(dropout_rate)
                                          )
        
        # we can use larger dilation for increased receptive field and improved performance
        # but so far no luck! we'll sticking to the dilation=1 (default)
        self.res_blocks = nn.ModuleList([ResidualBlock(128, 128, dropout_rate=0.00, dilation=1),
                                         ResidualBlock(128, 128, dropout_rate=0.00, dilation=1),
                                         ResidualBlock(128, 256, dropout_rate=0.00, dilation=1),
                                         ResidualBlock(256, 256, dropout_rate=0.00, dilation=1),
                                         ResidualBlock(256, 512, dropout_rate=0.00, dilation=1),
                                         ResidualBlock(512, 512, dropout_rate=0.00, dilation=1),
                                        ])
        
        # skip connections from all layers (feature pyramids)
        focount = 32
        self.skip_convs = nn.ModuleList([nn.Conv2d(128, focount, kernel_size=1),
                                         nn.Conv2d(128, focount, kernel_size=1),
                                         nn.Conv2d(256, focount, kernel_size=1),
                                         nn.Conv2d(256, focount, kernel_size=1),
                                         nn.Conv2d(512, focount, kernel_size=1),
                                         nn.Conv2d(512, focount, kernel_size=1)
                                        ])
        
        # last layers after concatenating skip connections
        self.final_layers = nn.Sequential(nn.Conv2d(focount*len(self.skip_convs), 512, kernel_size=1),
                                          nn.BatchNorm2d(512),
                                          nn.ReLU(),
                                          nn.Dropout2d(dropout_rate),
                                          nn.Conv2d(512, 256, kernel_size=1),
                                          nn.BatchNorm2d(256),
                                          nn.ReLU(),
                                          nn.Dropout2d(dropout_rate),
                                          nn.Conv2d(256, num_embds, kernel_size=1)
                                          )

    def forward(self, input_indices, labels=None):
        # input shape: (batch, h, w)
        if not self.input_shape:
            self.input_shape = input_indices.shape[1:]
        
        input_indices = self.embedding(input_indices) # (batch, h,w,embd)
        # (batch, embd, h, w)
        input_indices = input_indices.permute(0, 3, 1, 2)

        if self.make_conditional and labels is not None:
            # since we want to concat input and labels together, 
            # they must match in shape, we need to reshape our labels
            # accordingly. all we need to do is to add 2 new dimensions
            # to labels, and then repeat those two!
            # label to match input which is (batchsize, h,w)
            # butsince label is onehot encoded, we need to make it 4d
            # and also add a channel dim to x so they match!
            # print(f'{labels.shape=}')
            labels = self.fc_label_embedding(labels.float())# (batch,embd)
            # print(f'{labels.shape=}')
            labels = labels.view(labels.shape[0], labels.shape[1], 1, 1)
            labels = labels.expand(-1, -1, input_indices.shape[2], input_indices.shape[3])
            # print(f'{labels.shape=}')
            input_indices = torch.cat([input_indices,labels],dim=1)
        
        output = self.initial_conv(input_indices)
        skips = []
        for res_block, skip_conv in zip(self.res_blocks, self.skip_convs):
            output = res_block(output)
            skips.append(skip_conv(output))
        
        # combine skip connections into (batch, 192, 7, 7)
        combined = torch.cat(skips, dim=1)
        # print(f'{combined.shape=}')
        
        # we need logits so we can turn into probablities
        # for sampling in generation process
        logits = self.final_layers(combined)
        return logits

############################

# Gated activation as used in the original PixelCNN++
class GatedActivation(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        # Split the channels in half
        a, b = torch.chunk(x, 2, dim=1)
        # Apply gated activation: tanh(a) ⊙ sigmoid(b)
        return torch.tanh(a) * torch.sigmoid(b)


# Improved residual block with gated activation
class GatedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.1, dilation=1):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Double the output channels for gated activation
        gated_channels = out_channels * 2
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, gated_channels, kernel_size=1),
            nn.BatchNorm2d(gated_channels),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate)
        )
        
        self.conv2 = nn.Sequential(
            MaskedConv2d('B', gated_channels, gated_channels, kernel_size=3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(gated_channels),
            GatedActivation(),
            nn.Dropout2d(dropout_rate)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(gated_channels // 2, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )
        
        # Skip connection
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        
        self.activation = nn.ReLU()
    
    def forward(self, x):
        residual = x
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        return self.activation(x + self.skip(residual))

class Print(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def forward(self, outputs):
        print(f'{outputs.shape=}')
        return outputs


class GatedConv2d(nn.Module):
    """Gated Masked Convolution Layer"""
    def __init__(self, mask_type, in_channels, out_channels, kernel_size, padding, dilation=1):
        super().__init__()
        self.conv_f = MaskedConv2d(mask_type, in_channels, out_channels, kernel_size, padding, dilation=dilation)
        self.conv_g = MaskedConv2d(mask_type, in_channels, out_channels, kernel_size, padding, dilation=dilation)

    def forward(self, x):
        f = torch.tanh(self.conv_f(x))
        g = torch.sigmoid(self.conv_g(x))
        return f * g


# this block ignores the autoregressive nature and causes the model to cheat
# I kept getting extremely low validation loss with this, but generation was
# awful, extremely bad! basically garbage! then I swapped it with simple conv2d
# which I again got the same behavior cuz my dumbass forgot conv2d also breaks
# the autoregressive nature of the process, and causes the network to be baleto
# see the whole pixels! then I swapped it with normal resblock with maskedconv2d
# and it basically performed just like the old architecture, I then changed the 
# attention layer (see the next one ahead) to make it causal (basically the normal
# attention used in transformers) and this time it behaved like our resblock with maskedconv2d
# overall the changes in this architecture didnt do anything! so I guess the issue lies
# somewhere else, possibly in the vqvae base model itself, maybe I need to add the
# improvements in t he second paper to get good results or buff up the architetcure
# even more!
class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.query = nn.Conv2d(channels, channels // 8, 1)
        self.key = nn.Conv2d(channels, channels // 8, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, H, W = x.size()
        query = self.query(x).view(batch_size, -1, H * W).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, H * W)
        value = self.value(x).view(batch_size, -1, H * W)
        attn = torch.bmm(query, key)
        attn = F.softmax(attn, dim=-1)
        attn_out = torch.bmm(value, attn.permute(0, 2, 1))
        attn_out = attn_out.view(batch_size, C, H, W)
        return self.gamma * attn_out + x

class MaskedAttentionBlock(nn.Module):
    """Self-Attention with Causal Masking for Autoregressive Models"""
    def __init__(self, channels, H, W): # Need spatial dims for mask
        super().__init__()
        self.channels = channels
        self.H = H
        self.W = W
        # Use channels // 8 or some other factor, ensure it's not zero if channels < 8
        self.head_dim = max(1, channels // 8)
        self.query = nn.Conv2d(channels, self.head_dim, 1)
        self.key = nn.Conv2d(channels, self.head_dim, 1)
        self.value = nn.Conv2d(channels, channels, 1) # Value uses full channels
        self.gamma = nn.Parameter(torch.zeros(1))

        # Create causal mask
        mask = torch.tril(torch.ones(H * W, H * W)) # Shape (HW, HW)
        # Register as buffer
        self.register_buffer('causal_mask_base', mask) # Store the base mask


    def forward(self, x):
        batch_size, C, H, W = x.size()
        assert H == self.H and W == self.W, \
            f"Input spatial dims ({H},{W}) don't match block's expected dims ({self.H},{self.W})"

        HW = H * W
        query = self.query(x).view(batch_size, self.head_dim, HW).permute(0, 2, 1) # B, HW, C'
        key = self.key(x).view(batch_size, self.head_dim, HW) # B, C', HW
        value = self.value(x).view(batch_size, C, HW).permute(0, 2, 1) # B, HW, C

        # Attention Scores: B, HW, HW
        attn = torch.bmm(query, key) # Q * K^T
        attn = attn / (self.head_dim ** 0.5) # Scale

        # --- Apply the causal mask (Corrected) ---
        # Get the base mask and select the relevant part
        current_mask = self.causal_mask_base[:HW, :HW] # Shape (HW, HW)
        # Create the boolean condition for masked_fill
        mask_condition = (current_mask == 0) # Shape (HW, HW)
        # Apply to attn (B, HW, HW) - broadcasts mask_condition correctly
        masked_attn = attn.masked_fill(mask_condition, float('-inf'))
        # --- End Correction ---

        # Softmax over keys (last dimension)
        # attn_softmax shape will now be (B, HW, HW)
        attn_softmax = F.softmax(masked_attn, dim=-1)

        # --- Dtype Correction ---
        # Ensure dtypes match for bmm by casting value to attn_softmax's dtype
        # this is for when we use autocast! explain!
        value_casted = value.to(attn_softmax.dtype)
        # --- End Correction ---

        # print("attn_softmax shape:", attn_softmax.shape, "dtype:", attn_softmax.dtype)
        # print("value_casted shape:", value_casted.shape, "dtype:", value_casted.dtype)

        # Weighted sum of values: B, HW, C
        attn_out = torch.bmm(attn_softmax, value_casted) # Attn * V (Corrected)

        # Reshape back: B, C, H, W
        attn_out = attn_out.permute(0, 2, 1).view(batch_size, C, H, W)

        return self.gamma * attn_out + x

class ResidualBlock(nn.Module):
    """Gated Residual Block with Conditional BatchNorm"""
    def __init__(self, in_channels, out_channels, dropout_rate=0.1, dilation=1):
        super().__init__()
        self.block = nn.Sequential(
            MaskedConv2d('B', in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate),
            MaskedConv2d('B', out_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(out_channels)
        )
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        return F.relu(self.block(x) + self.skip(x))

class ImprovedPixelCNN(nn.Module):
    def __init__(self, num_embds, embedding_size=128, num_class=10, make_conditional=True, dropout_rate=0.1, H=8,W=8):
        super().__init__()
        self.num_embds = num_embds
        self.embedding_size = embedding_size
        self.num_class = num_class
        self.make_conditional = make_conditional
        # indecex dimensions required for attention
        self.H = H
        self.W = W
        self.embedding = nn.Embedding(num_embds, embedding_size)
        self.fc_label_embedding = nn.Linear(num_class, embedding_size)
        
        self.conv_input_size = self.embedding_size * 2 if make_conditional else self.embedding_size
        self.initial_conv = nn.Sequential(
            MaskedConv2d('A', self.conv_input_size, 128, kernel_size=11, padding=5),#k=11,p=5 ->8x8
            nn.BatchNorm2d(128),
            nn.ReLU(),
          )

        # Define blocks with their respective output channel sizes:
        self.res_blocks = nn.ModuleList([
            ResidualBlock(128, 128, dropout_rate=dropout_rate, dilation=1),
            ResidualBlock(128, 128, dropout_rate=dropout_rate, dilation=1),
            # MaskedAttentionBlock(128,H=H,W=W),#16x16 for 64x64 imgsz
            ResidualBlock(128, 256, dropout_rate=dropout_rate, dilation=1), 
            ResidualBlock(256, 256, dropout_rate=dropout_rate, dilation=1),
            # MaskedAttentionBlock(256,H=H,W=W),#16x16 for 64x64 imgsz
            ResidualBlock(256, 256, dropout_rate=dropout_rate, dilation=1),
            ResidualBlock(256, 512, dropout_rate=dropout_rate, dilation=1),
        ])

        # Corrected skip connections: one per block, matching output channels.
        self.skip_convs = nn.ModuleList([
            nn.Conv2d(128, 64, kernel_size=1),  # For Block 1 (128 -> 128)
            nn.Conv2d(128, 64, kernel_size=1),  # For Block 2 (128, 128)
            nn.Conv2d(256, 64, kernel_size=1),  # For Block 3 (128 -> 256)
            nn.Conv2d(256, 64, kernel_size=1),  # For Block 4 (256 -> 256)
            nn.Conv2d(256, 64, kernel_size=1),  # For Block 5 (256, 256)
            nn.Conv2d(512, 64, kernel_size=1)   # For Block 6 (256 -> 512)
        ])

        self.final_layers = nn.Sequential(
            nn.Conv2d(64 * len(self.skip_convs), 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(512, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_embds, kernel_size=1)
        )

    def forward(self, input_indices, labels=None):
        # Embed and prepare conditional input.
        input_indices = self.embedding(input_indices).permute(0, 3, 1, 2)
        if self.make_conditional and labels is not None:
            labels = self.fc_label_embedding(labels.float())
            labels = labels.view(labels.shape[0], labels.shape[1], 1, 1).expand(-1, -1, input_indices.shape[2], input_indices.shape[3])
            input_indices = torch.cat([input_indices, labels], dim=1)

        # print(f'{input_indices.shape=}')
        output = self.initial_conv(input_indices)
        skips = []
        for res_block, skip_conv in zip(self.res_blocks, self.skip_convs):
            output = res_block(output)
            skips.append(skip_conv(output))
        combined = torch.cat(skips, dim=1)
        logits = self.final_layers(combined)
        return logits



#########################
def train_prior(prior:PixelCNN, 
                vqvae_model:VQVAE,
                # latent_codes, 
                # latent_labels,
                dataloader,
                dataset_name:str, 
                num_classes=None,
                epochs=50,
                batchsize=32,
                lr=1e-3,
                weight_decay=1e-5,
                selected_label=9,
                sample_size=64,
                temperature=1,
                rows=9,
                cols=8,
                save_recons_dir=None,
                generation_device='cuda',
                figsize=(3,4),
                seed=66):
    
    train_datetime = datetime.datetime.now().strftime("%H_%M_%S_%Y_%m_%d")
    is_conditional = "Conditional_" if prior.make_conditional else ""
    model_checkpoint_name = f'vqvae_prior_{dataset_name.upper()}_embd{prior.embedding_size}_{is_conditional}{train_datetime}.ckpt'
    device = next(prior.parameters()).device
    
    recons_dir=None
    if save_recons_dir:
        ext = os.path.splitext(model_checkpoint_name)[-1]
        dir_name = model_checkpoint_name.replace(ext,"")
        recons_dir = os.path.join(save_recons_dir,dir_name)
    
    # H,W = prior.input_size
    # H, W = latent_codes.shape[1:]
    
    optimizer = torch.optim.Adam(prior.parameters(), lr=lr)    
    
    # After training vqvae, we need to grab the training set's encodings
    # and use these encodings to train our prior model
    latent_codes,latent_labels = get_latent_codes(vqvae_model, dataloader)

    # combine latent codes and labels
    dataset = torch.utils.data.TensorDataset(latent_codes, latent_labels)

    # dataloader_train = torch.utils.data.DataLoader(latent_codes, batch_size=batchsize, shuffle=True)
    # we can also split our latents into train/val and better keep track of our training
    # but I noticed for our simple case, its really not needed
    
    val_split = 0.1 
    dataset_size = len(dataset)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size
    dataset_train, dataset_val = torch.utils.data.random_split(dataset,[train_size, val_size])
    
    dataloader_train = torch.utils.data.DataLoader(dataset_train, 
                                                   batch_size=batchsize, 
                                                   shuffle=True, 
                                                   pin_memory=True, 
                                                   num_workers=8, 
                                                   drop_last=True)
    
    dataloader_val = torch.utils.data.DataLoader(dataset_val, 
                                                 batch_size=batchsize, 
                                                 shuffle=False,
                                                 pin_memory=True,
                                                 num_workers=8,
                                                 drop_last=True)
    
    #AdamW works much better than Adam! by a long shot! with adam we got loss=4.0, while
    # with AdamW with the same architecture we got down to 2 for mnist!
    optimizer = torch.optim.AdamW(prior.parameters(), lr=lr, weight_decay=weight_decay,)
    
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
    #                                                        factor=0.5, patience=3,
    #                                                        min_lr=1e-6)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=epochs)
    
    # new learning rate scheduler
    warmup_epochs = 5
    total_steps = len(dataloader_train) * epochs
    warmup_steps = len(dataloader_train) * warmup_epochs
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    
    # if warmup_steps > 0:
    #     # use 1e-8 as a very small start_factor to avoid potential issues with exactly 0
    #     scheduler_warmup = torch.optim.lr_scheduler.LinearLR(optimizer,
    #                                                          start_factor=1e-8,
    #                                                          end_factor=1.0,
    #                                                          total_iters=warmup_steps) 
    #     # decay to 1% of peak LR
    #     scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
    #                                                                   T_max=total_steps - warmup_steps,
    #                                                                   eta_min=lr * 0.01)
    #     # combine schedulers
    #     scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_cosine], milestones=[warmup_steps])
    #     print(f"Using Linear Warmup ({warmup_steps} steps) + Cosine Annealing ({total_steps - warmup_steps} steps) scheduler.")
    # else:
    #     # Only Cosine Annealing if no warmup
    #     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=lr * 0.01)
    #     print(f"Using Cosine Annealing ({total_steps} steps) scheduler (no warmup).")
    
    losses_epoch=[]
    losses_val_epoch=[]
    BPD_epoch=[]
    BPD_epoch_val=[]
    best_val_loss = float('inf')
    
    param_cnt = sum(p.numel() for p in prior.parameters())    
    
    print(f'Experiment Date:  {train_datetime}')
    print(f'Dataset Name   :  {dataset_name.upper()}')
    print(f'Embedding Size :  {prior.embedding_size}')
    print(f'Parameter Count:  {param_cnt:,}')
    print(f'Checkpoint Name:  {model_checkpoint_name}')

    
    for epoch in range(epochs):
        losses=[]
        bpds_training = []
        prior.train()
        for latents, labels in dataloader_train:
            latents = latents.to(device)
            
            if num_classes:
                if dataset_name.lower() =='celeba':
                    # use labels as is, except we make sure we filter all -1s as 0s!
                    labels = (labels == 1).float().to(device)
                else:
                    # otherwise convert to one_hot encoded
                    labels = F.one_hot(labels,num_classes=num_classes).to(device)
            else:
                labels = None
            
            logits = prior(latents,labels)
            # targets = latents.long()
            loss = F.cross_entropy(logits, latents.long())
            #! calculate bits per dimension: needs excessive edits
            # bits per dimension(bpd) is used to evaluate generative models, 
            # (especially those that work with images). for example , the original pixelcnn achieves bpd
            # of 2.29 for cifar10. and everyone who trains or works in this section uses it. 
            # because it makes comparing different models across different image sizes (or datasets) easier,
            # its essentially measuring how well the model compresses the data. 
            # the lower the value the better, it means the model is better at predicting the data distribution, 
            # as it requires fewer bits per dimension (or per pixel in our case) to encode the image
            # (since images are high-dimensional data, we normalize the negative log-likelihood by 
            # the number of dimensions (pixels) in the image)
            # the actual formula is BPD = (NLL) / (number of pixels * log(2))
            # our model outputs probabilities for each pixel and each pixel is modeled as a discrete distribution
            # (i.e. like 256 possible values for each color channel in an 8-bit image) 
            # The loss is the negative log-likelihood of the true pixel values given the model's predicted distribution
            # we sum the NLL over all pixels in the image, then average over the batch. This gives us the average
            # NLL per image.
            # we then divide this average NLL by the number of pixels per image (i.e. 32*32*3=3072) to get NLL per dimension (pixel).
            # then convert from nats to bits by dividing by log(2). 
            # note that since log base e (natural log) is used in the NLL calculation, if we divide by log(2) it converts it 
            # to bits (since log2(x) = ln(x)/ln(2)).
            # so, the formula would be:
            # BPD = (average NLL per image) / (number of pixels per image) / log(2)
            # or :
            # BPD = (NLL_total / (num_images * num_pixels)) / log(2)
            # but since we used crossentropy here,  it's already averaged over the batch and the elements.
            # so if the loss is computed as the average NLL per pixel, then we just need to convert that
            # average to bits by dividing by log(2) and dont need to divide it by n_dims here!
            #  
            
            #correct explanation : (revised by google):
            # Bits Per Dimension (BPD) is a standard metric used to evaluate the performance of generative models, particularly those modeling high-dimensional data like images (e.g., PixelCNN achieving ~2.92 BPD on CIFAR-10 is a common benchmark reference, though numbers vary slightly).
            # Purpose: BPD quantifies how well a model predicts the data distribution. It essentially measures the average number of bits required to encode each dimension (e.g., each pixel value or sub-pixel value) of the data, assuming an ideal compression scheme based on the model's predicted probabilities. Lower BPD indicates a better model (better compression, closer fit to the true data distribution). It allows for standardized comparison across models and datasets, normalizing for dimensionality.
            # Underlying Calculation: BPD is derived from the negative log-likelihood (NLL) of the data under the model. The NLL is typically calculated using the natural logarithm (base e), resulting in units of "nats".
            # Formula: The fundamental definition is:
            # BPD = Average NLL per Dimension (in nats) / ln(2)
            # or equivalently:
            # BPD = Average NLL per Dimension (in nats) * log2(e)
            # where ln(2) is the natural logarithm of 2 (approx 0.693) and log2(e) is the base-2 logarithm of e (approx 1.443). The division by ln(2) or multiplication by log2(e) converts the units from nats to bits.
            # Role of F.cross_entropy: When modeling discrete data (like pixel values 0-255), F.cross_entropy is commonly used as the loss function.
            # It calculates the NLL for each individual element (pixel/sub-pixel) based on the model's predicted probabilities (logits) and the true target value (latents.long()).
            # Crucially, with the default reduction='mean', F.cross_entropy averages these NLL values (in nats) over all elements across the entire batch.
            # Therefore, the output loss = F.cross_entropy(logits, latents.long()) directly gives you the Average NLL per Dimension (in nats).
            # Calculating BPD from F.cross_entropy Loss: Since loss.item() already represents the average NLL per dimension in nats:
            # # loss = F.cross_entropy(logits, latents.long()) # Assumes reduction='mean'
            # nats_per_dim = loss.item()
            # # Convert nats per dimension to bits per dimension
            # # Using np.log(2) which is ln(2):
            # bpd = nats_per_dim / np.log(2)
            # # Or using np.log2(np.e):
            # # bpd = nats_per_dim * np.log2(np.e)
            # Use code with caution.
            # Python
            # You do not need to divide by the number of dimensions (n_dims or np.prod(latents.shape[1:])) again, because the cross-entropy loss with mean reduction has already performed that averaging.
                
            # n_dims = np.prod(latents.shape)
            bpd = loss.item() * np.log2(np.e) #/ n_dims
            
            optimizer.zero_grad()
            loss.backward()
            # clip gradients to prevent exploding gradients
            # torch.nn.utils.clip_grad_norm_(prior.parameters(), max_norm=1.0)
            optimizer.step()
            # when using lambdalr/cosinelr
            scheduler.step()
            
            losses.append(loss.item())
            bpds_training.append(bpd)

        with torch.no_grad():
            print('validation...')
            prior.eval()
            losses_val=[]
            bpds_val=[]
            for latents, labels in dataloader_val:
                latents = latents.to(device)
                
                if num_classes:
                    if dataset_name.lower() =='celeba':
                        # use labels as is, except we make sure we filter all -1s as 0s!
                        labels = (labels == 1).float().to(device)
                    else:
                        # otherwise convert to one_hot encoded
                        labels = F.one_hot(labels,num_classes=num_classes).to(device)
                else:
                    labels = None
                    
                logits = prior(latents, labels)
                # targets = latents.long()
                loss = F.cross_entropy(logits, latents.long())
                bpd = loss.item() * np.log2(np.e)
                # store them for plots
                losses_val.append(loss.item())
                bpds_val.append(bpd)
                
        avg_loss = np.mean(losses)
        avg_bpd = np.mean(bpds_training)
        avg_val_loss = np.mean(losses_val)
        avg_val_bpd = np.mean(bpds_val)

        losses_epoch.append(avg_loss)
        losses_val_epoch.append(avg_val_loss)
        BPD_epoch.append(avg_bpd)
        BPD_epoch_val.append(avg_val_bpd)
        
        # scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'state_dict': prior.state_dict(),
                # 'optimizer': optimizer.state_dict(),
                # 'scheduler':scheduler.state_dict(),
                'loss': avg_loss,
                'val_loss': best_val_loss,
                'bpd': avg_bpd,
                'bpd_val':avg_val_bpd,
                'model_config': {
                    'num_embds': prior.num_embds,
                    'embedding_size': prior.embedding_size,
                    'num_class': prior.num_class,
                    'make_conditional': prior.make_conditional
                }
            }, model_checkpoint_name.replace('.ckpt','_best.pt'))
            
            print(f'best model saved with val-loss: {best_val_loss:.6f}')
        
        # save the last epoch 
        torch.save({
                'epoch': epoch,
                'state_dict': prior.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler':scheduler.state_dict(),
                'loss': avg_loss,
                'val_loss': avg_val_loss,
                'bpd': avg_bpd,
                'bpd_val':avg_val_bpd,
                'model_config': {
                    'num_embds': prior.num_embds,
                    'embedding_size': prior.embedding_size,
                    'num_class': prior.num_class,
                    'make_conditional': prior.make_conditional
                }
            }, model_checkpoint_name)
        
        print(f'Epoch: {epoch}/{epochs}  | Loss: {avg_loss:.6f} | Val-Loss: {avg_val_loss:.6f} | BPD: {np.mean(bpds_training):.6f} |  BPD_VAL: {np.mean(bpds_val):.6f} | LR:{scheduler.get_last_lr()[-1]:.6f}')
        
        # display reconstruction performance!
        fname=None
        if save_recons_dir:
            if not os.path.exists(recons_dir):
                os.makedirs(recons_dir)
            fname = f'{recons_dir}/generated_{epoch}.jpg'
        
        display_generated_samples(vqvae_model=vqvae_model,
                                  prior_model=prior,
                                  dataset=dataset_name,
                                  num_classes=num_classes,
                                  selected_label=selected_label,
                                  batch_size=sample_size,
                                  temperature=temperature,
                                  device=generation_device,
                                  rows=rows,
                                  cols=cols,
                                  figsize=figsize,
                                  seed=seed,
                                  fname=fname)
    
    # create gifs out of all generated samples
    create_gifs(dir_path=recons_dir)
    
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(losses_epoch, label='Training Loss')
    plt.plot(losses_val_epoch, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    # Plot bits per dimension
    plt.subplot(1, 3, 2)
    plt.plot(BPD_epoch)
    plt.xlabel('Epoch')
    plt.ylabel('Bits per Dimension')
    plt.title('Model Efficiency[BPD](Lower is Better)')
    
    plt.subplot(1, 3, 3)
    plt.plot(BPD_epoch_val)
    plt.xlabel('Epoch')
    plt.ylabel('Bits per Dimension')
    plt.title('Model Efficiency[BPD-Val](Lower is Better)')
    
    plt.tight_layout()
    plt.show()
    
    print('training prior model complete!')
    # pd.DataFrame(losses_epoch).plot()
    # plt.plot()
    return prior, model_checkpoint_name


from tqdm import tqdm

def train_improved_prior(prior, latent_codes, latent_labels, dataset_name, num_classes=None, 
                        epochs=100, batchsize=64, lr=3e-4, weight_decay=1e-4):
    
    train_datetime = datetime.datetime.now().strftime("%H_%M_%S_%Y_%m_%d")
    is_conditional = "Conditional_" if prior.make_conditional else ""
    model_checkpoint_name = f'improved_vqvae_prior_{dataset_name.upper()}_embd{prior.embedding_size}_{is_conditional}{train_datetime}.ckpt'
    device = next(prior.parameters()).device
    
    # Create dataset with latent codes and labels
    dataset = torch.utils.data.TensorDataset(latent_codes, latent_labels)
    
    val_split= 0.1
    dataset_size = len(dataset)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size 
    
    generator = torch.Generator().manual_seed(42)
    dataset_train, dataset_val = torch.utils.data.random_split(dataset,[train_size, val_size],
                                                                        generator=generator)
    
    # Create data loaders with proper augmentation
    dataloader_train = torch.utils.data.DataLoader(dataset_train, 
                                                   batch_size=batchsize, 
                                                   shuffle=True, 
                                                   pin_memory=True, 
                                                   num_workers=8, 
                                                   drop_last=True)
    
    dataloader_val = torch.utils.data.DataLoader(dataset_val, 
                                                 batch_size=batchsize, 
                                                 shuffle=False,
                                                 pin_memory=True,
                                                 num_workers=8,
                                                 drop_last=False)
    
    optimizer = torch.optim.AdamW(prior.parameters(), lr=lr, weight_decay=weight_decay,)
    
    # Improved learning rate scheduler
    warmup_epochs = 5
    total_steps = len(dataloader_train) * epochs
    warmup_steps = len(dataloader_train) * warmup_epochs
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    
    # if warmup_steps > 0:
    #     # use 1e-8 as a very small start_factor to avoid potential issues with exactly 0
    #     scheduler_warmup = torch.optim.lr_scheduler.LinearLR(optimizer,
    #                                                          start_factor=1e-8,
    #                                                          end_factor=1.0,
    #                                                          total_iters=warmup_steps) 
    #     # decay to 1% of peak LR
    #     scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
    #                                                                   T_max=total_steps - warmup_steps,
    #                                                                   eta_min=lr * 0.01)
    #     # combine schedulers
    #     scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_cosine], milestones=[warmup_steps])
    #     print(f"Using Linear Warmup ({warmup_steps} steps) + Cosine Annealing ({total_steps - warmup_steps} steps) scheduler.")
    # else:
    #     # Only Cosine Annealing if no warmup
    #     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=lr * 0.01)
    #     print(f"Using Cosine Annealing ({total_steps} steps) scheduler (no warmup).")

    
    # Evaluation metrics
    losses_train = []
    losses_val = []
    bpd_train = []
    bpd_val = []
    lr_history = []
    
    # Early stopping parameters
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    # Model info
    param_cnt = sum(p.numel() for p in prior.parameters() if p.requires_grad)
    
    print(f'====== Training Details ======')
    print(f'Experiment Date  : {train_datetime}')
    print(f'Dataset          : {dataset_name.upper()}')
    print(f'Embedding Size   : {prior.embedding_size}')
    print(f'Parameter Count  : {param_cnt:,}')
    print(f'Batch Size       : {batchsize}')
    print(f'Learning Rate    : {lr}')
    print(f'Weight Decay     : {weight_decay}')
    print(f'Dataset Sizes    : Train {train_size}, Val {val_size}')
    print(f'Checkpoint Name  : {model_checkpoint_name}')
    print(f'===========================')
    
    # Add AMP for mixed precision training
    scaler = torch.amp.GradScaler()
    
    for epoch in range(epochs):
        # Training phase
        prior.train()
        train_loss = 0.0
        train_bpd = 0.0
        
        progress_bar = tqdm(dataloader_train, desc=f'Epoch {epoch+1}/{epochs}')
        
        for latents, labels in progress_bar:
            latents = latents.to(device)
            labels = labels.to(device)
            
            # Convert labels to one-hot if needed and the model is conditional
            if prior.make_conditional and num_classes:
                if labels.dim() == 1 or (labels.dim() == 2 and labels.shape[1] == 1):
                    labels_onehot = F.one_hot(labels.squeeze(), num_classes=num_classes).float()
                else:
                    labels_onehot = labels.float()
            else:
                labels_onehot = None
            
            optimizer.zero_grad()
            
            # Use mixed precision training
            with torch.amp.autocast(device_type="cuda"):
                logits = prior(latents, labels_onehot)
                loss = F.cross_entropy(logits, latents.long())
            
            # since we are using F.cross_entropy with the default reduction='mean' 
            # it means it averages the loss over all the individual elements (dimensions) already
            # and we dont need to divide it by n_dims again (normalize it again!)
            # n_dims = np.prod(latents.shape)
            bpd = loss.item() * np.log2(np.e)
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            # torch.nn.utils.clip_grad_norm_(prior.parameters(), max_norm=1.0)
            
            # Update weights with gradient scaling
            scaler.step(optimizer)
            scaler.update()
            
            # Update learning rate
            scheduler.step()
            
            # Update metrics
            train_loss += loss.item()
            train_bpd += bpd
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'bpd': f'{bpd:.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.6f}'
            })
        
        # Calculate average metrics for the epoch
        train_loss /= len(dataloader_train)
        train_bpd /= len(dataloader_train)
        
        # Validation phase
        prior.eval()
        val_loss = 0.0
        val_bpd = 0.0
        
        with torch.no_grad():
            for latents, labels in dataloader_val:
                latents = latents.to(device)
                labels = labels.to(device)
                
                # Convert labels to one-hot if needed and the model is conditional
                if prior.make_conditional and num_classes:
                    if labels.dim() == 1 or (labels.dim() == 2 and labels.shape[1] == 1):
                        labels_onehot = F.one_hot(labels.squeeze(), num_classes=num_classes).float()
                    else:
                        labels_onehot = labels.float()
                else:
                    labels_onehot = None
                
                logits = prior(latents, labels_onehot)
                loss = F.cross_entropy(logits, latents.long())
                
                # Calculate bits per dimension
                # n_dims = np.prod(latents.shape)
                bpd = loss.item() * np.log2(np.e)
                
                val_loss += loss.item()
                val_bpd += bpd
        
        # Calculate average metrics for validation
        val_loss /= len(dataloader_val)
        val_bpd /= len(dataloader_val)
        
        # Store metrics
        losses_train.append(train_loss)
        losses_val.append(val_loss)
        bpd_train.append(train_bpd)
        bpd_val.append(val_bpd)
        lr_history.append(scheduler.get_last_lr()[0])
        
        # Print epoch summary
        print(f'Epoch: {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Train BPD: {train_bpd:.6f} | Val BPD: {val_bpd:.6f} | LR: {scheduler.get_last_lr()[0]:.6f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            #! make it match the old training script
            torch.save({'epoch': epoch,
                        'state_dict': prior.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'val_loss': best_val_loss,
                        'bpd':train_bpd,
                        'bpd_val': val_bpd,
                        'model_config': {
                            'num_embds': prior.num_embds,
                            'embedding_size': prior.embedding_size,
                            'num_class': prior.num_class,
                            'make_conditional': prior.make_conditional
                        }
            }, model_checkpoint_name.replace(".ckpt","_best.ckpt"))
            
            print(f'✅ Best model saved with val-loss: {best_val_loss:.6f}')
        else:
            patience_counter += 1
            print(f'⚠️ Validation loss did not improve. Patience: {patience_counter}/{patience}')
                
        # save all the models
        torch.save({'epoch': epoch,
                    'state_dict': prior.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'val_loss': best_val_loss,
                    'bpd':train_bpd,
                    'bpd_val': val_bpd,
                    'model_config': {
                        'num_embds': prior.num_embds,
                        'embedding_size': prior.embedding_size,
                        'num_class': prior.num_class,
                        'make_conditional': prior.make_conditional
                        }
        }, model_checkpoint_name.replace(".ckpt","_best.ckpt"))

        # Early stopping
        # if patience_counter >= patience:
        #     print(f'⛔ Early stopping triggered after {epoch+1} epochs')
        #     break
    
    # 
    # prior.eval()
    # test_loss = 0.0
    # test_bpd = 0.0
    
    # with torch.no_grad():
    #     for latents, labels in dataloader_test:
    #         latents = latents.to(device)
    #         labels = labels.to(device)
            
    #         if prior.make_conditional and num_classes:
    #             if labels.dim() == 1 or (labels.dim() == 2 and labels.shape[1] == 1):
    #                 labels_onehot = F.one_hot(labels.squeeze(), num_classes=num_classes).float()
    #             else:
    #                 labels_onehot = labels.float()
    #         else:
    #             labels_onehot = None
            
    #         logits = prior(latents, labels_onehot)
    #         loss = F.cross_entropy(logits, latents.long())
            
    #         # n_dims = np.prod(latents.shape)
    #         bpd = loss.item() * np.log2(np.e)
            
    #         test_loss += loss.item()
    #         test_bpd += bpd
    
    # test_loss /= len(dataloader_test)
    # test_bpd /= len(dataloader_test)
    
    # print(f'📊 Test Results | Loss: {test_loss:.6f} | BPD: {test_bpd:.6f}')
    
    # Plot training history
    plt.figure(figsize=(18, 6))
    
    # Loss plot
    plt.subplot(1, 3, 1)
    plt.plot(losses_train, label='Training Loss')
    plt.plot(losses_val, label='Validation Loss')
    # plt.axhline(y=test_loss, color='r', linestyle='--', label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    # BPD plot
    plt.subplot(1, 3, 2)
    plt.plot(bpd_train, label='Training BPD')
    plt.plot(bpd_val, label='Validation BPD')
    # plt.axhline(y=test_bpd, color='r', linestyle='--', label='Test BPD')
    plt.xlabel('Epoch')
    plt.ylabel('Bits per Dimension')
    plt.legend()
    plt.title('Model Efficiency (Lower is Better)')
    
    # Learning rate plot
    plt.subplot(1, 3, 3)
    plt.plot(lr_history)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(f'training_history_{dataset_name.upper()}_{train_datetime}.png')
    plt.show()
    
    print('✨ Training complete!')
    
    return prior, model_checkpoint_name


def generate(model:VQVAE, prior:PixelCNN, labels, num_classes, batch_size=1, temperature=1.0, device="cuda", seed=66):
    # todo use seed so we get the same images each time! for comparison purposes!
    generator = torch.Generator(device).manual_seed(seed)
    
    prior.eval()
    # it must match our latent space shape from our vqvae model
    # print(f'{model.enc_output_shape=}')
    H, W = model.enc_output_shape
    # print(f'{H=},{W=}')
    assert labels.size(0) == batch_size, 'classes count must batch batches!'
    
    # !edit explanation
    # if we conditioned our images on 0s, first then we need to start with zeros
    # but we didnt so we use randint
    # codes = torch.zeros((batch_size, H, W), dtype=torch.long, device=device)
    codes = torch.randint(0, model.embd_num, size=(batch_size, H, W), dtype=torch.long, device=device, generator=generator)
    labels = F.one_hot(labels,num_classes=num_classes).to(device)
    # print(f'{labels.shape=}')
    
    with torch.no_grad():
        for i in range(H):
            for j in range(W):
                # predict logits for the current latent position
                # higher temps means more diversity in output!
                # (i.e. it controls randomness in our output, lower temp=less randomness in ouput)
                logits = prior(codes,labels)[:, :, i, j] / temperature
                # convert to probability distribution
                probs = F.softmax(logits, dim=-1)
                # sample from the predicted distribution
                sampled_codes = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)
                # update the latent code map
                codes[:, i, j] = sampled_codes  

    # print(f'{codes.shape=}')
    # convert latent codes to embeddings and reshape for decoding
    quantized = model.quantizer.embeddings(codes.flatten()).view(batch_size, H, W, -1)
    # print(f'{quantized.shape=}')
    quantized = quantized.permute(0, 3, 1, 2).contiguous()
    # print(f'{quantized.shape=}')
    # decode the quantized representations into images
    generated = model.decoder(quantized)
    # print(f'{generated.shape=}')
    return generated

def display_generated_samples(vqvae_model:VQVAE, prior_model:PixelCNN, 
                              dataset, num_classes=10, selected_label=9,
                              batch_size=64, temperature=1, device='cuda', 
                              rows=9, cols=8, figsize=(3,4),seed=66, fname=None):

    if 'cifar' in dataset:
        class_names = {0:'airplanes', 1:'cars', 2:'birds', 3:'cats', 4:'deer',
                    5:'dogs', 6:'frogs', 7:'horses', 8:'ships',9:'trucks'}

    elif dataset =='mnist':
        class_names = {0:'zeros', 1:'ones', 2:'twos', 3:'threes', 4:'fours',
                    5:'fives', 6:'sixes', 7:'sevens', 8:'eigths',9:'nines'}
    else:#celeba
        class_names = {i:'N/A' for i in range(num_classes)}

    print(f'Generating images of {class_names[selected_label]}')
    # todo: create proper label for celeba!
    labels = torch.ones(size=(batch_size,),dtype=torch.long)*selected_label
    # due to a bug in my code (I hardcoded the encoder outputs shape/indexces shape)
    # I would get weird generations! when I icnreased the image size form 32 to 64 and
    # retired, the reconstructions got much better, but generation seemed cropped! looked
    # closer and noticed my bug and fixed it and now images are way better. they are very good
    # a bit deformed which is relaetd to overfitting , but overall it seems alright!
    generated_image = generate(vqvae_model,
                               prior_model,
                               labels=labels,
                               num_classes=num_classes,
                               batch_size=batch_size,
                               # when using conditional, using smaller values 
                               # for temperature, give us weireder images/really 
                               # simplestic images! like with way less details!
                               temperature=temperature,
                               device=device,
                               seed=seed)
    view_images(generated_image, labels, rows=rows, cols=cols, figsize=figsize, fname_to_save_as=fname) 


#!edit add more explanation
# another way to generate images, instead of using prior model
# we directly sample from code frequency, it shows if our model
# has good features or not (whether the problem lies in prior model/its training
# or vqvae features itself. the images may not look good! more explanation ahead)
def generate_simple(model, latent_codes, num_samples=1):
    # compute code frequencies from training data
    counts = torch.bincount(latent_codes.flatten())
    probs = counts / counts.sum()
    # sample indices from the frequency distribution
    H, W = latent_codes.shape[1:] #7x7
    indices = torch.multinomial(probs, num_samples * H * W, replacement=True)
    indices = indices.view(num_samples, H, W).to(device)
    # print(f'{indices.shape=}')
    # decode the indices
    quantized = model.quantizer.embeddings(indices)  # (num_samples, H, W, embd_size)
    # print(f'{quantized.shape=}')
    quantized = quantized.permute(0, 3, 1, 2)  # (num_samples, embd_size, H, W)
    # print(f'{quantized.shape=}')
    generated = model.decoder(quantized)
    # print(f'{generated.shape=}')
    return generated

def sample_from_prior(prior:PixelCNN, model:VQVAE, num_samples=16, temperature=1.0, class_label=None, 
                      num_classes=10, shape=None, top_k=0, top_p=0.9, device='cuda'):
    """
    Generate samples from the trained prior model
    
    Args:
        prior: Trained PixelCNN prior model
        vqvae: Trained VQVAE model for decoding latent codes
        num_samples: Number of samples to generate
        temperature: Temperature for sampling (higher = more diverse, lower = more conservative)
        class_label: Class label for conditional generation (None for unconditional)
        num_classes: Number of classes in the dataset
        shape: Shape of latent space (height, width)
        top_k: If > 0, only sample from the top k most likely tokens
        top_p: If < 1.0, only sample from the smallest set of tokens whose cumulative probability exceeds p
        device: Device to generate samples on
        
    Returns:
        Tensor of generated samples (batch_size, channels, height, width)
    """
    prior.eval()
    model.eval()
    
    # Determine latent shape if not provided
    if shape is None:
        shape = prior.input_shape
    
    # Create empty latent codes
    latent_shape = (num_samples, *shape)
    latents = torch.zeros(latent_shape, dtype=torch.long, device=device)
    
    # Prepare conditional labels if needed
    if prior.make_conditional and class_label is not None:
        if isinstance(class_label, int):
            # Single class for all samples
            labels = torch.full((num_samples,), class_label, dtype=torch.long, device=device)
            labels_onehot = F.one_hot(labels, num_classes=num_classes).float()
        else:
            # Different class for each sample
            labels = torch.tensor(class_label, dtype=torch.long, device=device)
            labels_onehot = F.one_hot(labels, num_classes=num_classes).float()
    else:
        labels_onehot = None
    
    # Auto-regressive generation
    with torch.no_grad():
        for h in tqdm(range(shape[0]), desc="Generating rows"):
            for w in range(shape[1]):
                # Get predictions for current pixel
                logits = prior(latents, labels_onehot)
                logits = logits[:, :, h, w]
                
                # Apply temperature
                if temperature != 1.0:
                    logits = logits / temperature
                
                # Convert to probabilities
                probs = F.softmax(logits, dim=-1)
                
                # Apply top-k sampling if specified
                if top_k > 0:
                    # Zero out all values below the top-k values
                    values, indices = torch.topk(probs, top_k, dim=-1)
                    min_values = values[:, -1].unsqueeze(-1).expand_as(probs)
                    probs = torch.where(probs < min_values, torch.zeros_like(probs), probs)
                    # Renormalize probabilities
                    probs = probs / probs.sum(dim=-1, keepdim=True)
                
                # Apply top-p (nucleus) sampling if specified
                if top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    
                    # Create a mask for probs that exceed the cumulative probability threshold
                    mask = cumulative_probs <= top_p
                    # Ensure at least one token is kept
                    mask[:, 0] = True
                    
                    # Convert mask back to original indices
                    batch_indices = torch.arange(num_samples).unsqueeze(-1).expand_as(sorted_indices)
                    mask_indices = torch.zeros_like(probs, dtype=torch.bool)
                    mask_indices.scatter_(-1, sorted_indices, mask)
                    
                    # Apply the mask and renormalize
                    probs = torch.where(mask_indices, probs, torch.zeros_like(probs))
                    probs = probs / probs.sum(dim=-1, keepdim=True)
                
                # Sample from the distribution
                pixel_samples = torch.multinomial(probs, 1).squeeze(-1)
                
                # Update latent codes with sampled value
                latents[:, h, w] = pixel_samples
    
    with torch.no_grad():
        reconstructions = model.decoder(latents)
    
    return reconstructions, latents



#%%
#todo move get_latent_codes inside training because they are tightly coupled!
# # After training vqvae, we need to grab the trainingset's encodings
# # and use these encodings to train our prior model
# latent_codes,latent_labels = get_latent_codes(model, dataloader_train)

# train our PixelCNN prior
# embdsize=256 results in a very decent generation 
# compared to 128 even with 32x32 imgsize
# 
# for celeba use the nonconditional version because it comes with 40 attributes for
# each individual image. we can choose to incorporate them or at least condition our
# models on one of these attributes, but for now we just ignore them and choose the
# unconditional version
# ok the unconditional generation works great, but when we start using the labels on celeba, 
# it will make the trainig harder, and images start worse than the unconditional version
# most probably because of the way we are incorporating the labels in our archiecture
# since its a multilabel case, a much better fusion strategy is needed. there are many 
# ways we can go about it, from fusing at multiple levels in our architecture so the
# labels semantic are transfered properly throughout the features in the model, to simply
# using several layers on embeddings to get better representation/or using summing/etc the
# list goes on!

conditional = True 
num_classes = 40 if dataset=='celeba' else 10

prior = PixelCNN(num_embds=model.embd_num, embedding_size=256,
                 num_class=num_classes,
                 make_conditional=conditional,
                 dropout_rate=0.1,).to(device)

prior, ckptname = train_prior(prior=prior,
                              vqvae_model=model,
                              dataloader=dataloader_train,
                              dataset_name=dataset,# for logging purposes only!
                              num_classes=num_classes, 
                              epochs=120,
                              batchsize=64,
                              lr=0.001,
                              weight_decay=1e-2,
                              selected_label=9,
                              sample_size=64,
                              temperature=1,
                              rows=9,
                              cols=8,
                              generation_device='cuda',
                              figsize=(3,4),
                              seed=66,
                              save_recons_dir='./results/')

# prior = ImprovedPixelCNN(num_embds=model.embd_num, 
#                          embedding_size=256,
#                          num_class=10,
#                          make_conditional=True,
#                          dropout_rate=0.1,).to(device)

# prior, checkpoint_path = train_improved_prior(prior,
#                                             latent_codes,
#                                             latent_labels,
#                                             dataset_name=dataset,
#                                             num_classes=10,
#                                             epochs=120,
#                                             batchsize=64,
#                                             lr=3e-4,
#                                             weight_decay=1e-2)

#sidenote: 
# starting with small lr leads to crazy overfitting! especially with 32x32 imgsize!
# sidenote: starting with smaller lr=(1e-3) overfitted badly, but the generation was
# waaaaay better. I increased embdsize for prior though, need to check 
# it with higher lr and see if it gets better if it doesnt oevrfit badly!
# note: when I changed the model from 32x32 to 64x64 in my second test, I didnt
# rerun the get_latent_codes, I guess this is the reason why I kept getting weird
# output all these times!
# check if this is the case using a second round of tests!
#%%
# vqvae_18_28_36_2025_03_25.ckpt shows very strange generations for celeba64x64!!!
# ok it was for wrong encoding size ( I used 7x7 when I had increased img size to 64x64 
# instead of 32x32 and it would mess up the generation! see git log info)
# when I fixed it it became ok. eventhough loss is around 4.xx the generation is miles
# better than than before!(when we used 32x32 versions!)

# ckptname='vqvae_18_28_36_2025_03_25.ckpt'
# cifar10 unconditional
# ckptname = 'vqvae_23_13_38_2025_03_25.ckpt'
ckptname = 'vqvae_prior_CIFAR10_embd256_Conditional_20_22_25_2025_04_02.ckpt'#64x64
ckptname = 'vqvae_prior_CIFAR10_embd256_Conditional_20_22_25_2025_04_02_best.ckpt'#64x64
# I noticed, running more epochs at the expense of lower BPD or worse val loss, results in
# better generation usually! so try both checkpoints (the last one and the best one) and 
# compare the results

ckptname = 'vqvae_prior_MNIST_embd256_Conditional_16_41_50_2025_04_03.ckp'#32
# Ok it seems, the val loss/val bpd doesnt mean the best result! especially if we
# get that in early epochs. the smalles training loss/bpd has a much better result
# than the our best val/bpd values! makes me wonder if having a validation set even
# matters!
ckptname = 'vqvae_prior_MNIST_embd256_Conditional_18_20_55_2025_04_03.ckpt'#64
ckptname = 'vqvae_prior_MNIST_embd256_Conditional_18_20_55_2025_04_03_best.ckpt'

ckptname = 'vqvae_prior_CIFAR10_embd256_Conditional_19_41_59_2025_04_03.ckpt'#64
ckptname = 'vqvae_prior_CIFAR10_embd256_Conditional_19_41_59_2025_04_03_best.ckpt'#64
# not good. I lowered the dropout ratio and it I believe it make it worse than before!
ckptname = 'vqvae_prior_CIFAR10_embd256_Conditional_09_36_43_2025_04_04.ckpt'#۳۲
ckptname = 'vqvae_prior_CIFAR10_embd256_Conditional_09_36_43_2025_04_04_best.ckpt'#۳۲
# for celeba because the dataset is much larger, we have far b etter generations!
# obviously having a better vqvae and prior models with better training can yield
# much better result. but for us this siffuces and shows given more data, with the
# same architecture, we can achieve pretty good results.
# ckptname = 'vqvae_prior_CELEBA_embd256_10_40_59_2025_04_04.ckpt'#64
# ckptname = 'vqvae_prior_CELEBA_embd256_10_40_59_2025_04_04_best.ckpt'#64

# train cifar10 x64x64 with embd=256 for vqvae and see if that changes anythinG!
# clean and git push to privae repo first

# ok increasing the embedding for vqvae model resultted in way better generations!
# both loss and BPD dropped from 5 to 1!! and the gap between training and val became
# way less steep! so we learned the vqvae is crucial to getting great reconstructions
# and simple reconstruction results in vqvae doesnt mean theres an issue in prior models
# secotion if our loss doesnt decrease! it may very well be vqvae needs to be tuned (buffed)
# more! in our case it was to simply use larger embedding dim (256)!
# I need to train others with the new embd_size for vqvae to see how they perform :)
# test these 3 models to see how they fair against each other
ckptname = 'vqvae_prior_CIFAR10_embd256_Conditional_16_02_21_2025_04_04.ckpt'#emb256/256 x64
# ckptname = 'vqvae_prior_CIFAR10_embd256_Conditional_16_02_21_2025_04_04_e46.ckpt'#emb256/256 x64
# ckptname = 'vqvae_prior_CIFAR10_embd256_Conditional_16_02_21_2025_04_04_best.ckpt'#emb256/256 x64
#
# like before with the increased embd, the generation is near prefect!
ckptname = 'vqvae_prior_CELEBA_embd256_10_32_36_2025_04_05.ckpt' # ebmbd256/256 64x64
# ckptname = 'vqvae_prior_CELEBA_embd256_10_32_36_2025_04_05_e55.ckpt' # ebmbd256/256 64x64
# ckptname = 'vqvae_prior_CELEBA_embd256_10_32_36_2025_04_05_best.pt' # ebmbd256/256 64x64


ckptname = 'vqvae_prior_CELEBA_embd256_Conditional_15_23_55_2025_04_05.ckpt'#embd256/256/64x64
ckptname = 'vqvae_prior_CELEBA_embd256_Conditional_15_23_55_2025_04_05_best.pt'#embd256/256/64x64

ckpt = torch.load(ckptname)
prior.load_state_dict(ckpt["state_dict"])
print(f'Epoch       : {ckpt["epoch"]}')
print(f'train_Loss  : {ckpt['loss']:.4f} | BPD: {ckpt['bpd']:.4f}')
print(f'val_Loss    : {ckpt['val_loss']:.4f}   | BPD: {ckpt['bpd_val']:.4f}')
#%%
# Generate new image
if 'cifar' in dataset:
    class_names = {0:'airplanes', 1:'cars', 2:'birds', 3:'cats', 4:'deer',
                   5:'dogs', 6:'frogs', 7:'horses', 8:'ships',9:'trucks'}

elif dataset =='mnist':
    class_names = {0:'zeros', 1:'ones', 2:'twos', 3:'threes', 4:'fours',
                   5:'fives', 6:'sixes', 7:'sevens', 8:'eigths',9:'nines'}
else:
    class_names = {i:str(i) for i in range(40)}

seed=12
batch_size = 64
num_classes=40
selected_label = 9
print(f'Generating images of {class_names[selected_label]}')
labels = torch.ones(size=(batch_size,),dtype=torch.long)*selected_label
# due to a bug in my code (I hardcoded the encoder outputs shape/indexces shape)
# I would get weird generations! when I icnreased the image size form 32 to 64 and
# retired, the reconstructions got much better, but generation seemed cropped! looked
# closer and noticed my bug and fixed it and now images are way better. they are very good
# a bit deformed which is relaetd to overfitting , but overall it seems alright!
generated_image = generate(model,
                           prior,
                           labels=labels,
                           num_classes=num_classes,
                           batch_size=batch_size,
                           # when using conditional, using smaller values 
                           # for temperature, give us weireder images/really 
                           # simplestic images! like with way less details!
                           temperature=1,
                           seed=seed)
view_images(generated_image,labels,rows=9,cols=8,figsize=(3,4))
#%%
generated_image1 = generate_simple(model, latent_codes,num_samples=64)
# print(f'{generated_image1.shape=}')
view_images(generated_image1,torch.ones(generated_image1.size(0),1),rows=8,cols=8)

#%%
samples, latents = sample_from_prior(prior,
                                    model,
                                    num_samples=16,
                                    temperature=0.8,
                                    class_label=3,  # Optional: specific class to generate
                                    top_p=0.9  # Use nucleus sampling
                                )
#%%
# check to see if our codebook has collapsed
# if only a few codes are used here (e.g. 1-2 codes dominate), our VQ-VAE codebook has collapsed
# despite the perplexity of 180 (which may be misleading if embd_num is large).
counts = torch.bincount(latent_codes.flatten())
print("Top 10 codes:", torch.topk(counts, 10).indices)
print("Code usage ratio:", len(counts.nonzero()) / model.embd_num)

#%%
# to see if our decoder is faulty, we can bypass the prior and
# manually set embeddings/codes to test the decoder.
# if this outputs a solid color(like our initial tests which we got red and then blue),
# it means our decoder is broken (though since our reconstructions was good this is unlikely).
# if it outputs diverse patterns, the prior is faulty.
#
# create random codes (replace 0 with other indices to test, using all zeros we get blue pattens)
# test_codes = torch.zeros((1, 32, 32), dtype=torch.long, device=device)
# using random codes, we get different patterns which shows our initial assumption
# that decoder is ok but prior has the issue is right!
test_codes = torch.randint(0, model.embd_num, (3, 64, 64), device=device)

quantized = model.quantizer.embeddings(test_codes)
quantized = quantized.permute(0, 3, 1, 2).contiguous()
generated_image = model.decoder(quantized)
view_images(generated_image,torch.ones(generated_image.size(0),1),rows=1,cols=1)
#%%
# The most frequent code maps to a vector that decodes to blue.
# the solution is to inspect the dominant codes embedding:
# if this is blue, the decoder associates this code with blue. 
# we need to retrain the VQ-VAE with a larger codebook (embd_num=512)
# or lower the commitment cost (beta=0.25) or do Codebook reset strategy https://arxiv.org/abs/1711.00937.
dominant_code = torch.argmax(counts).item()
print("Dominant code embedding:", model.quantizer.embeddings.weight[dominant_code])
# lets force decode this code:
test_codes = torch.full((3, 32, 32), dominant_code, device=device)
quantized = model.quantizer.embeddings(test_codes)
generated_image = model.decoder(quantized.permute(0, 3, 1, 2))
view_images(generated_image,torch.ones(generated_image.size(0),1),rows=1,cols=1)
#%%
# Case 2: the generate_simple() works but PixelCNN fails
# issue is PixelCNN isnt learning the code distribution.
# to solve this, we need to simplify the prior i.e. use a smaller PixelCNN or train longer.
# add dropout to prevent overfitting and or check input normalization.
# In PixelCNN.forward, ensure embeddings are normalized
# x = self.embedding(x)
# x = F.layer_norm(x, [self.emb_dim])  # Add this line
#%%
#side notes for debugging: 
# make sure the final layer uses nn.Sigmoid() for [0, 1] images
# verify quantized_z_ex includes gradients (quantized_z_ex = encoder_outputs + (quantized_z_ex - encoder_outputs).detach())
#

#%%
# # Contractive Autoencoder
# main paper : http://www.icml-2011.org/papers/455_icmlpaper.pdf
# ref1: https://wiseodd.github.io/techblog/2016/12/05/contractive-autoencoder/
# ref2: https://www.youtube.com/watch?v=BW7P1fvnAWk
# 
# So we have seen several different flavors of a autoencoders so far(the classic ones of course there are many newer variants!). 
# However, there is one more autoencoding method on top of them, known as 
# Contractive Autoencoder (Rifai et al., 2011).
# The contractive autoencoder is categorized as a regularizier autoencoder. in other words
# This autoencoder specifically prevents an overcomplete autoencoder from learning the identity function
# which basically means, it prevents it from copying/memorizing the input as apposed to learning
# benificial features to reconstruct the input that matters for us. 
# using the regularization term that we will get to shortly, this can also be used on undercomlete
# encoders as well. 
# It will achieve this, by adding a new term to the weight matrix. which is as follows: 
#                  2 
# Ω(Ø) =  ‖ Jx(h) ‖p 
# lets expand this and see what this term actually is. 
#     2  
# ‖  ‖p  is called a norm. a Frobenius Norm. Frobenius Norm is like L2 norm (Eculidean norm)
# and is used on matrix and is calculated as the square root of the sum of the absolute squares of its elements
# which means, we simply sum all the elements in a mxn matrix and then take the square root of it.
# Now what is it applied on? it is applied on Jx(h). 
# what is Jx(h) you may ask? Its the Jacobian Matrix. what is a Jacobian matrix?
# it simply a matrix of partial derivatives of all elements with respect to all inputs.
# if you look closely you can see that we have Jx(h), J, x and h.
# x is our input, h is our parameters. Jx(h) means, a matrix of partial derivitives of all parameters
# with respect to x. How does it look like ? this is roughly how it looks like: 
# suppose, input has n dimensions and our hidden layer has h dimensions. 
# our resulting jabobian matrix will have n+k dimensions
#         | dh1/dx1, dh1/dx2, dh1/dx3, ..., dh1/dxn|
# Jx(h) = | dh2/dx1, dh2/dx2, dh2/dx3, ..., dh2/dxn|
#         | dh3/dx1, dh3/dx2, dh3/dx3, ..., dh3/dxn|
#         |  ...      ...       ...    ...    ...
#         | dhk/dx1, dhk/dx2, dhk/dx3, ..., dhk/dxn| 
# As you can see, each column, shows the partial derivitives for all neurons with respect
# to a single input. for example, the first column, shows the partial derivitives for all
# neurons with respect to the "first" input, likewise, the second column, shows the partial
# derivative for all neurons in our hidden layer with respect to the second input and so on.
# So basically when we are taking the derivative of a vector with respect to another vector
# we get a matrix that you see above.  we can say each row belongs to one neuron and each col
# represents all neurons gradient with respect to a single input.
# So, what does all of this mean? what does each entry in the Jacobian matrix mean for us? 
# what can we infer from lets say element (i,l) of this matrix(i being inputs index as in 
# x_i and l being neurons index, being n_l)? 
# The (i,l)th element simply tells us, howmuch the h(l) changes with a change in x(i) 
# basically each entry in the jacobian matrix captures the variation in the output of
# the lth neuron with a small variation in the ith input.
# OK, now what does the Frobenious norm capture here? 
# what do we get by adding all the elments absolute values and squaring them? 
# This basically shows, howmuch each of these elements vary with respect to the input 
# and we are taking the square of that (to make it more prounounced)
# So this whole term is added to the loss function and the loss gets minimized. 
# This means, we want our Frobenious norm to get minimized as well, which means we want
# it to idealy be zero or near zero. (we said ideally becasue in actuality it wont be zero
# as there is always a tradeoff between L(Ø) and Ω(Ø) (remember loss = L(Ø) + Ω(Ø)), 
# if the norm gets to zero, it means L(Ø) will be very high)
# 
# Lets get a better intuition on how this works : 
# imagine, for example, dh1/dx1 goes actually to zero(dh1/dx1=0). what would that mean? 
# It means, h1 is not sensitive to variations in x1!  
#  but what does the original concept mandate here? what did we want to capture? 
# we wanted the neurons to capture these important characteristics (variations in input)
# so if x1 changes, we want h1 to change as well.
# So we wanted to capture the important characteristics in the input by each neuron, but 
# now, we have added a contradictory condition that we dont want to capture these kinds of 
# variations! So what is happening here? 
# L(Ø) says we should be able to capture these variations ortherwise I will not be able to 
# reconstruct the input. if all of my h_i's are not sensitive to variations in x1, this means
# if I give it any x1, it will produce the same h_i. 
# Lets recap again, basically, when we add the frobenious norm to the loss, and want to minize
# this norm as well which means, going toward zero, which again means, all the dh_i's
# need to go toward zero so that (dh/dx) is zero or near zero. and this simply means
# h_i is not sensitive to variations in input. while clearly we said we want to capture such
# variations (using the L(Ø) part in our loss)! 
# Thats the catch here, we have two contradictory terms in our loss, one tries to capture 
# the all features, while the other one tries just the opposite. 
# L(Ø) says, capture the variations in the data while
# Ω(Ø) says, do not capture the variations in teh data!
# Whats the tradoff here? capture only the important variations in the data and 
# do not capture the ones that are not important.
# look at the following plot for example :
#                  Y
#             . %8.                                   
#     .  . .    S@ .  .  . .  .  . .  .  . .  .  . .  
#    .     . .  8X .       .       .       .       .
#      .      . @%   . .     . .     . .     . .    
#  .     .  . . 8t .     .       . .     .       .  
#    .  .     . 8% .  .   . .  .   .:. U1  . .  .   .
#   .     . .   8t .    .       ..8%.t .         .  
#     .       . 8%  .      . . .%@888.    . . .    
#    U2. .  .   8t .  .  .   .88 ;t.     .        . 
# .X@:.       . 8%  .   .  ..8S:..  .  .     .  .   
#  :X8X    .    @% .     .:SX...          .       . 
#    . 8  . .  .8t .  . . ;8;.   .  . .     . .     
#   .  %8@;.  . @%  . . %%8;.           .       .  .
#     . ..X8@. .@% . .8SS%:.   .  .  .    .   .   . 
#         . ;@8%8t :S@%:.        .     .    .   .   
#  .  .       .X @  8S.     . .    .     .          
#      . . . . .@888: . . .    . .  . . .  . . . . .
#   . X@@X@@@X@X888@@@X@@@X@@@@@@X@@@@@@X@@@@@@X@.;;
#              .:.                                X;
# shape: "./contractive_autoencoder_u1_u2_axis_plot.png"
# 
# So This is how it goes, we have 2 dimensions u1 and u2 , of which u1 is more important
# as the data variation along the u1 dimension is something that we should care about
# What about the variations in u2? Not important, they seem like noises, becasue these 
# variations , they are not all laying up on the centeral line, they are slighly away from
# the line. here are some variations. but should we go out of our way to capture these
# variations? does it make sense to do that? no!     
# So it makes sense to maximize a neuron to be sensitive to variations along U1 
# but it does not make sense to make neuron sensitive for these variations along other
# dimension which is U2 . 
# So by doing so we balance the two conditions. one condition tries to capture all the
# important variations and says do this, but do it only for dimensions that only their features
# (variations) are important. 
# the other condition says, dont capture information, it says
# do this, but only for the dimensions that are not important.
# This is like PCA  (unbder certain conditions, vanilla autoencoder is equivalent to PCA)
# the passage from "Representation Learning: A Review and New Perspectives" by Bengio,
# Courville, et al. states:
# "In the case of a linear auto-encoder (linear encoder and
# decoder) with squared reconstruction error, the basic autoencoder
# objective in Equation 19 is known to learn the same
# subspace13 as PCA. This is also true when using a sigmoid
# nonlinearity in the encoder (Bourlard and Kamp, 1988), but
# not if the weights W and W0 are tied (W0 = WT )." 
# 
# I read this as saying that even with with a sigmoid in the encoder, if
# the weights are untied, you still end up learning the same subspace as
# PCA (?) and if the weights _are_ tied you do not.  Why? 

# The reasons I'm aware of for using tied weights:
# 1. In the linear case the optimal solution is PCA, which can be
# obtained with tied weights.
# 2. It has a regularization effect:
#     2a. Less parameters to be optimized
#     2b. It can prevent degenerate solutions, in particular those with
# very small weights in encoder, compensated by very large weights in
# decoder (something that would allow for instance a near-linear
# solution to be found with tanh nonlineraities)
# 3. Less parameters to be stored (=> lower memory footprint)

# That being said, lots of people also use un-tied weights... there's no
# general rule that tied > untied (or the reverse) -- it depends on the
# architecture & data.

# For the last question, this is because the optimal representation of
# your data in R^d in the sense of linear reconstruction error in
# original space R^n (d < n) is when the R^d representation is obtained
# by a PCA projection (up to some invertible linear transform). So if
# the weights are untied you can in the encoding part learn this
# projection, and in the decoding part learn the linear reconstruction.
# While if the weights are tied, in general you can't do this with a
# nonlinear encoder (the weights to obtain the PCA projection with the
# nonlinearity transform won't be the transposed of the PCA linear
# reconstruction). 

# I don't know if people do it explicitly for this purpose. Now that I think about it, 
# another motivation for tied weights may also come from RBMs, where the "reconstruction" 
# weights to compute P(x|h) are the transposed version of the "encoding" weights 
# computing P(h|x).
# I don't think that in general you can claim that tied weights lead to more "interesting" 
# solutions than PCA in the context of dimensionality reduction. Also, the better auto-encoders
# usually have a nonlinear decoder, in which case PCA is no longer optimal and untying weights 
# may actually help.
# -=- Olivier

# Autoencoders with tied weights have some important advantages :
#     It's easier to learn.
#     In linear case it's equvialent to PCA - this may lead to more geometrically adequate coding.
#     Tied weights are sort of regularisation.

# But of course - they're not perfect : they may not be optimal when your data comes from 
# highly nolinear manifold. Depending on size of your data I would try both approaches - 
# with tied weights and not if it's possible.

# UPDATE :
# You asked also why representation which comes from autoencoder with tight weights might be 
# better than one without. Of course it's not the case that such representation is always 
# better but if the reconstruction error is sensible then different units in coding layer 
# represents something which might be considered as generators of perpendicular features which
#  are explaining the most of the variance in data (exatly like PCAs do). This is why such 
# representation might be pretty useful in further phase of learning.

# https://agustinus.kristia.de/blog/contractive-autoencoder/



# So in 
# The idea of Contractive Autoencoder is to make the learned representation to be 
# robust towards small changes around the training examples. It achieves that by 
# using different penalty term imposed to the representation.
# The loss function for the reconstruction term is similar to previous Autoencoders 
# that we have been seen, i.e. using ℓ2 loss(MSE or BCE). The penalty term, however is more 
# complicated: we need to calculate the representation’s jacobian matrix with 
# regards of the training data.
# so basically our loss in this case would be : 
# L = \lVert X - \hat{X} \rVert_2^2  + \lambda \lVert J_h(X) \rVert_F^2 
# in which
# \lVert J_h(X) \rVert_F^2 = \sum_{ij} \left( \frac{\partial h_j(X)}{\partial X_i} \right)^2
# that is, the penalty term is the Frobenius norm of the jacobian matrix, which is the 
# sum squared over all elements inside the matrix. We could think Frobenius norm as the
# generalization of euclidean norm.
# 
# In the loss above, clearly it’s the calculation of the jacobian that’s not 
# straightforward. Calculating a jacobian of the hidden layer with respect to 
# input is similar to gradient calculation. Recall than jacobian is the generalization
# of gradient, i.e. when a function is a vector valued function, the partial derivative
# is a matrix called jacobian.
# However, we use autograd instead of creating the jacobian matrix as it is not practical for
# complex architectures and it imposes a huge overhead!

# https://medium.com/@SeoJaeDuk/arhcieved-post-personal-notes-about-contractive-auto-encoders-part-1-ef83bce72932
# carefully created penalty term can result in extracting more useful and effective features
# that gives insight to the given data. The penalty term invented by the authors of this paper
# makes the auto-encoders learned features to be locally invariant without any preference for
# particular directions. (they obtain invariance in the directions that make sense in the 
# context of the given training data, i.e., the variations that are present in the data should
# also be captured in the learned representation, but the other directions may be contracted 
# in the learned representation.) And the description of the penalty term can be seen below.
# Additionally, one very interesting question the authors asked is the notion, how can we 
# extract robust features? (aka features that are robust to small changes in the given input).
# The way they did is by adding a penalty term that is sensitive to the given input, and as the
# network trains, it’s objective is to make that sensitivity smaller and smaller.

# Note when the auto encoders do not have an activation function the above loss function is
# same as having weight decay (L2 penalty). Additionally, the authors of this paper have 
# investigate the case where the weights are tied. Another interesting fact is 
# sparse auto-encoders that outputs many zeroed out activation units achieves highly 
# contractive mapping, even without an concrete objective functions. One difference between
# de-noising auto encoders to CAE is DAE makes the network robust to both encoders and decoders,
# CAE only makes the encoder portion robust. (CAE robust is achieved via analytical solution,
# while DAE is achieved stochastically. )

# From this ppt I learned that the proof that a single layer neural network was based on using 
# exponentially large number of neurons and hence it is not practical. Also weight sharing is a
# method to reuse the weights on different layers, they will differ since their gradient differs. 
# Finally, auto encoders with regularization learns to model the keep only sensitivity to 
# variations on the manifold. (Reconstruction → Forces variations on the manifold, 
# regularization → want to remove variations.)
# Additionally, with the contraction loss, the network is trying to find features that are
# robust to given input. And when we have tied weights auto-encoders we can see the 
# relationship between the decoder weights to encoder weights as smoothen weights. 
# (In terms of the original data.) Here are some links to why we might use tied 
# auto-encoders, https://stackoverflow.com/questions/36889732/tied-weights-in-autoencoder,
#  https://groups.google.com/forum/#!topic/theano-users/QilEmkFvDoE

# this is a good read if you want to know why I wrote the loss function the way i did :
# https://mc.ai/how-pytorch-backward-function-works/
def fc_batchnorm_act(in_, out_, use_bn=True, act=nn.ReLU()):
    return nn.Sequential(nn.Linear(in_,out_),
                         act,
                         nn.BatchNorm1d(out_) if use_bn else nn.Identity())

class Reshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape 

    def forward(self, input):
        return input.view(self.shape)

class Contractive_AutoEncoder(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()
        self.embedding_size = embedding_size
        self.encoder = nn.Sequential(Reshape(shape=(-1, 28*28)),
                                     fc_batchnorm_act(28*28, 400, False, nn.Sigmoid()))

        self.decoder = nn.Sequential(fc_batchnorm_act(400, 28*28, False, nn.Sigmoid()),
                                     Reshape(shape=(-1, 1, 28, 28)))     
                                                            
        # self.encoder = nn.Sequential(Reshape(shape=(-1, 28*28)),
        #                              fc_batchnorm_act(28*28,512),
        #                              fc_batchnorm_act(512,256),
        #                              fc_batchnorm_act(256,128),
        #                              # dont use batchnorm on the last layer of enc
        #                              fc_batchnorm_act(128, embedding_size, False))
        
        # self.decoder = nn.Sequential(fc_batchnorm_act(embedding_size,128),
        #                              fc_batchnorm_act(128,256),
        #                              fc_batchnorm_act(256,512),
        #                              fc_batchnorm_act(512, 28*28, False, nn.Sigmoid()),
        #                              Reshape(shape=(-1, 1, 28, 28)))

    def forward(self, input):
        # flatten the input
        # shape = input.shape
        # input = input.view(input.size(0), -1)
        # outputs_e = F.relu(self.encoder(input))
        # outputs = F.sigmoid(self.decoder(output_e))
        # outputs = output.view(*shape)
        outputs_e = self.encoder(input)
        outputs = self.decoder(outputs_e)
        return outputs_e, outputs

def loss_function(output_e, outputs, imgs, lamda = 1e-4, device=torch.device('cuda')):
 
    criterion = nn.MSELoss()
    assert outputs.shape == imgs.shape ,f'outputs.shape : {outputs.shape} != imgs.shape : {imgs.shape}'
    loss1 = criterion(outputs, imgs)

    output_e.backward(torch.ones(outputs_e.size()).to(device), retain_graph=True)    
    # Frobenious norm, the square root of sum of all elements (absolute value)
    # in a jacobian matrix 
    loss2 = torch.sqrt(torch.sum(torch.abs(imgs.grad)))
    imgs.grad.data.zero_()
    loss = loss1 + (lamda*loss2) 
    return loss 

# based on the manual calculation of gradients as 
# done here :
# https://github.com/avijit9/Contractive_Autoencoder_in_Pytorch/blob/master/CAE_pytorch.py
# remember this loss only works for a 2 layer net, that uses sigmoid! 
# I just included this here as a reference 
def loss_function2(W, x, recons_x, h, lam=1e-4):
    mse = F.mse_loss(recons_x, x)
    # Since: W is shape of N_hidden x N. So, we do not need to transpose it as
    # opposed to #1
    dh = h * (1 - h) # Hadamard product produces size N_batch x N_hidden
    # Sum through the input dimension to improve efficiency, as suggested in #1
    w_sum = torch.sum(W**2, dim=1)
    # unsqueeze to avoid issues with torch.mv
    w_sum = w_sum.unsqueeze(1) # shape N_hidden x 1
    contractive_loss = torch.sum(torch.mm(dh**2, w_sum), 0)
    return mse + contractive_loss.mul_(lam)

# torch.autograd.set_detect_anomaly(True)
epochs = 50 
interval = 2000
embedding_size = 5
lam = 1e-4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Contractive_AutoEncoder(embedding_size).to(device)
optimizer = optim.Adam(model.parameters(), lr =0.001)
print(model)

W = model.encoder[1][0].weight

for e in range(epochs):
    for i, (imgs, labels) in enumerate(dataloader_train):
        imgs = imgs.to(device)
        labels = labels.to(device)
        # note imgs is not a leaf node, so the gardients wouldnot be ratained
        # in order to ratain gradients for non leaf nodes, use retain_graph
        # .grad field is only populated for leaf Tensors. If you want it for other Tensors, 
        # you can use the imgs.retain_grad() function to get the .grad field populated 
        # for non-leaf Tensors. but I found it esaier to just enable/diable the grads
        # inside the training loop and thus outside of lossfunction. 
        # also imgs.retain_grad() shuold be called before doing forward() as it will
        # instruct the autograd to store grads into nonleaf nodes. 
        imgs.retain_grad()
        imgs.requires_grad_(True)
        
        outputs_e, outputs = model(imgs)
        loss = loss_function(outputs_e, outputs, imgs, lam,device)
        # loss = loss_function2(W, imgs, outputs, outputs_e, lam)

        imgs.requires_grad_(False)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'epoch/epochs: {e}/{epochs} loss: {loss.item():.4f}')

# test
for i, (imgs, labels) in enumerate(dataloader_test):
    model.eval()
    imgs = imgs.to(device)
    labels = labels.to(device)

    imgs.requires_grad_(True)
    outputs_e, outputs = model(imgs)

    loss = loss_function(outputs_e, outputs, imgs, lam, device)
    # loss = loss_function2(W, imgs, outputs, outputs_e, lam)
    imgs.requires_grad_(False)
    print(f'iter/iterss: {i}/{len(dataloader_test)}  loss: {loss.item():.4f}')

# reconstruction test 
imgs, labels = next(iter(dataloader_test))
outputs_e, outputs_img = model(imgs.to(device))
imgs_org = make_grid(imgs[:10].detach().cpu(),nrow=10).numpy().transpose(1,2,0)
img = make_grid(outputs_img[:10].detach().cpu(),nrow=10).numpy().transpose(1,2,0)
img = np.concatenate((imgs_org,img),axis=0)
plt.imshow(img)

#%% 

#%%



#%%


#%%
# to do : 
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

# However, training VAE on languages is notoriously difficult due to something called KL vanishing. 
# While VAE is designed to learn to generate text using both local context and global features, it 
# tends to depend solely on local context and ignore global features when generating text. When this 
# happens, VAE is essentially behaving like a standard RNN language model.
# In “Cyclical Annealing Schedule: A Simple Approach to Mitigating KL Vanishing,” to be presented at 
# 2019 Annual Conference of the North American Chapter of the Association for   Computational Linguistics 
# (NAACL), researchers at Microsoft Research AI and Duke University propose an extremely simple remedy to 
# KL vanishing as well as their proposal to make the code publicly available on Github. 
# The remedy is based on a new scheduling scheme called Cyclical Annealing Schedule. Intuitively, 
# during the course of VAE training, we periodically adjust the weight of the KL term in the objective 
# function, providing the model opportunities to learn to leverage the global latent variables in text 
# generation, thus encoding as much global information in the latent variables as possible. The paper 
# briefly describes KL vanishing and why it happens, introduces the proposed remedy, and illustrates the 
# VAE learning process using a synthetic dataset.
# What is KL vanishing and why does it happen?
# VAEs aim to learn probabilistic representations z of natural languages x, with an objective consisting 
# of two terms: (1) reconstruction to guarantee the inferred latent feature z can represent its corresponding
# observed sentence; and (2) KL regularization to leverage the prior knowledge to modulate language understanding.
# The two terms are balanced by a weighting hyper-parameter β:
# When applied on text corpora, VAEs typically employ an auto-regressive decoder, which sequentially generates 
# the word tokens based on ground-truth words in the previous steps, in conjunction with latent z. Recent work 
# has found that naïve training of VAEs (keeping constant β=1) leads to model degeneration—the KL term becomes 
# vanishingly small. This issue causes two undesirable outcomes: (1) the learned features are almost identical 
# to the uninformative Gaussian prior, for all observed languages; and (2) the decoder completely ignores the 
# latent feature, and the learned model reduces to a simpler neural language model. Hence, the KL vanishing issue.
# This negative result is so far poorly understood. We developed a two-path competition interpretation to shed 
# light on the issue. Let’s first look at the standard VAE in Figure 1 (a), below. The reconstruction of sequence 
# x=[x1 ,…,xT] depends only on one path passing through the encoder ϕ, latent representation z and decoder Θ. 
# However, when an auto-regressive decoder is used in a VAE, there are two paths from observed x to its 
# reconstruction, as shown in Figure 1(b). Path A is the same as that in the standard VAE, where z serves as 
# the global representation that controls the generation of x; Path B leaks the partial ground-truth information 
# of x at every time step of the sequential decoding. It generates xt conditioned on x<t=[x1,…,xt-1]. Therefore, 
# Path B can potentially bypass Path A to generate xt, leading to KL vanishing. From this perspective, 
# we hypothesize that the KL vanishing problem is related to the low quality of z at the beginning phase of 
# decoder training. This is highly possible when the naive constant schedule of β=1 is used, as the KL term 
# pushes z close to an uninformative prior, less representative of the corresponding observations. This lower 
# quality z introduces more difficulties in reconstructing x, and eventually blocks the information flow via Path A. 
# As a result, the model is forced to learn an easier solution to decoding—generating x via Path B only.
 

# Figure 1: Illustration of information flows on (a) one path in a standard VAE, and (b) two paths in a VAE with an auto-regressive decoder.
# Cyclical Annealing Schedule

# A simple remedy via scheduling β during VAE training was proposed by Bowman, et al, as shown in Figure 2(a). 
# It starts with β=0 at the beginning of training, and gradually increases β until β=1 is reached. This monotonic 
# schedule of β has become the de facto standard in training text VAEs, and has been widely adopted in many NLP 
# tasks. Why does it improve the performance empirically? When β<1, z is trained to focus more on capturing useful 
# information for reconstruction of x. When the full VAE objective is considered (β=1), z learned earlier can be 
# viewed as VAE initialization; such latent features are much more informative than the random start in constant 
# schedule and thus are ready for the decoder to use.
# Figure 2: Annealing β with (a) the monotonic schedule and (b) the cyclical schedule.
# Figure 2: Annealing β with (a) the monotonic schedule and (b) the cyclical schedule.
# Is there a better schedule? It is key to have meaningful latent z at the beginning of training the decoder, so 
# that Path A is utilized. The monotonic schedule under-weights the prior regularization when β<1; the learned z 
# tends to collapse into a point estimate. This underestimation can result in sub-optimal decoder learning. 
# A natural question concerns how one can get a better distribution estimate for z as initialization, while decoder 
# has the opportunity to leverage such z in learning.
# Our proposal is to use the latent z trained under the full VAE objective as initialization. 
# To learn to progressively improve z we propose a cyclical schedule for β that simply repeats the monotonic
# schedule multiple times as shown in Figure 2(b). We start with β=0, increase β at a fast rate, and then stay 
# at β=1 for subsequent learning iterations. This completes one period of monotonic schedule. It encourages the 
# model to converge towards the VAE objective, and infers its first raw full latent distribution. Unfortunately,
# β=1 gradually blocks Path A, forbidding more information from passing through z. Crucially, we then start the 
# second period of β annealing and training is continued at β=0 again. This perturbs the VAE objective, dislodges 
# it from the convergence, and reopens Path A. Importantly, the decoder now (1) has the opportunity to directly 
# leverage z, without obstruction from KL; and (2) is trained with the better latent z than point estimates, as 
# the full distribution learned in the previous period is fed in. We repeat this β annealing process several 
# times to achieve better convergences.
# Visualization of learning dynamics in the latent space

# To visualize the learning processes on an illustrative problem, let’s consider a synthetic dataset consisting 
# of 10 different sequences, as well as a VAE model with a 2-dimensional latent space, and an LSTM encoder and 
# decoder.
# We visualize the resulting division of the latent space for different training steps in Figure 3, where each 
# color corresponds to the latent probabilistic representation of a sequence. We observe that:
#     The constant schedule produces heavily mixed latent codes z for different sequences throughout the entire 
# training process.
#     The monotonic schedule starts with a mixed z, but soon divides the space into a mixture of 10 cluttered 
# Gaussians in the annealing process (the division remains cluttered in the rest of training).
#     The cyclical schedule behaves similarly to the monotonic schedule in the 1st cycle. But starting from 
# the 2nd cycle, much more divided clusters are shown when learning on top of the 1st period results. However, 
# β<1 leads to some holes between different clusters. This is alleviated at the end of the 2nd cycle, as the 
# model is trained with β=1. As the process repeats, we see clearer patterns in the 4th cycle than the 2nd 
# cycle for both β<1 and β=1. It shows that more structured information is captured in z, using the cyclical 
# schedule.
# Figure 3: The process of learning probabilistic representations in the latent space for three schedules.
# The learning curves for the VAE objective (ELBO), reconstruction error, and KL term are shown in Figure 4. 
# The three schedules share very similar ELBO values. However, the cyclical schedule provides substantially 
# lower reconstruction error and higher KL divergence. Interestingly, the cyclical schedule improves the 
# performance progressively; it becomes better than the previous cycle, and there are clear periodic patterns 
# across different cycles. This suggests that the cyclical schedule allows the model to use the previously 
# learned results as a warm-restart to achieve further improvement.
# Figure 4: Comparison of terms in VAE for three schedules.
# Improving performance on NLP tasks

# The new cyclical schedule has been demonstrated to be effective in improving probabilistic representations 
# of synthetic sequences on the illustrative example, but is it beneficial in downstream real-world natural 
# language processing (NLP) applications? We tested it on three tasks:
#     Language Modeling. On the Penn Tree-Bank dataset, the cyclical schedule can provide more informative 
# language representations (measured by the improved KL term), while retaining the similar perplexity. It is 
# significantly faster than existing methods, and can be combined to improve upon them.
#     Dialog response generation. It is key to have probabilistic representations for conversational context, 
# reasoning stochastically for different but relevant responses. On the SwitchBoard dataset, the cyclical 
# schedule generates highly diverse answers that cover multiple plausible dialog acts.
#     Unsupervised Language Pre-training. On the Yelp dataset, a language VAE model is first pre-trained to 
# extract features, then a classifier is fine-tuned with different proportions of labelled data. The cyclical 
# schedule provides robust distribution-based representations of sentences, yielding strong generalization on 
# testing datasets.

# We hope to see you at NAACL-HLT this June to discuss these approaches in more detail and we’ll look forward 
# to hearing what you think!
# Acknowledgements
# This research was conducted by Chunyuan Li, Hao Fu, Xiaodong Liu, Jianfeng Gao, Asli Celikyilmaz, and 
# Lawrence Carin. Additional thanks go to Yizhe Zhang, Sungjin Lee, Dinghan Shen, and Wenlin Wang for their 
# insightful discussion. The implementation in our experiments heavily depends on three NLP applications 
# published on Github repositories; we acknowledge all the authors who made their code public, which tremendously 
# accelerates our project progress.
# The post Less pain, more gain: A simple method for VAE training with less of that KL-vanishing agony appeared 
# first on Microsoft Research.

#%%
# Adversarial Autoencoder https://blog.paperspace.com/adversarial-autoencoders-with-pytorch/


#%% [markdown]
