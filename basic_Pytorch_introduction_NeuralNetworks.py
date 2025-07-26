#%% [markdown]
# In the name of God the most compassionate the most merciful 
import torch 
import numpy as np 
import matplotlib.pyplot as plt 
%matplotlib inline

#%%
# Here we are going to see how we can create a neural network and train/test it in Pytorch
# We will see how we can augment our data, create our datasets, etc and alot more.  
# In Pytorch we can use the torchvision module, for reading existing datasets that Pytorch offers,
# or create a dataset out of our existing folder of images. 
# It also provides a fakedataset for images, which we can use for benchmarking, or debugging.  
# We also do augmentation using this module. 
# This module also provides several well known achitectures such as AlexNet, VGGNet, 
# ResNet, MobileNet, VIT, Conext, efficientnets, etc
# 
# ok. lets see how to use all of these! 
# here lets import datasets for using the dataset capabilities
# use transforms for data-augmentation, and
# models for using existing models
import torchvision
from torchvision import datasets, transforms, models

# Our first step is to create a dataset. lets create MNIST dataset for the start
# before we create our dataset, we should know that upon creating our dataset
# we need to specify at least 1 transformation and that is ToTensor(). 
# ToTensor() actually converts the input images into torch tensors and this is a must have
# transformation is not limited to ToTensor only, it includes a range of transformation from
# resize, flipping, etc. 
# so we do it this way 
transformations = transforms.ToTensor()
# but what if we want to do more transformations, such as padding the image, or filliping it, or resizing it?
# we can simply compose as many tansformations as we wish!
# the Compose method, takes a list of transformations so we can simply instead do : 
transformations = transforms.Compose([transforms.ToTensor(),
                                      # for normalizing note we used 0.5, that ',' 
                                      # is needed since mean and std require a tuple
                                      # and since mnist is grayscale(black n white)
                                      # / has 1 channel images only! we use 1
                                      # number only for mean and std.
                                      # sidenote: we dont need to normalize mnist as its already
                                      # in the 0-1 range. doing the following normalization will
                                      # takes it to [-1,1] range!
                                      # that is (img-0.5)/0.5 = (img)*2-1
                                      #transforms.Normalize(mean=(0.5,),std=(0.5,))
                                      ])
# 
dataset_train = datasets.MNIST(root='./data/MNIST/', train=True, transform=transformations, download=True)
dataset_test = datasets.MNIST(root='./data/MNIST/', train=False, transform=transformations, download=True)

# now we have our datasets. but as you know, usually we dont load the whole dataset all atonce!
# instead we read in batches! in Pytorch we do this using a dataloader. using a dataloader
# we can easily specify a batchsize, and other performance related options such as number of workers
# which is the number of threads used to read from the dataset and thus provide a more efficient data
# loading! lets see how to create a dataloader!
# the Dataloader resides in torch.utils.data !
import torch.utils.data as data
dataloader_train = data.DataLoader(dataset_train, batch_size=32, shuffle=True, num_workers=2)
# we do the same thing for test 
dataloader_test = data.DataLoader(dataset_test, batch_size=32, shuffle=False,num_workers=2)
print(f'test dataloader size: {len(dataloader_test)}')
# there is a note here. whenever you get weird errors concerning dataloader/dataset, the first thing 
# you do is to set num_workers = 0. usually when you have a probelm in your dataset (specially
# when you create your own dataset) this will help you see the issue immediaitly. becasue setting num_workers=0
# uses the same thread to run everything, and thus will catch your error, whereas when you set it to any number
# greater than 0, you will get a completely different error which is concerning threads failing!
# so have this in mind!


# OK, we defined our datasets, and dataloaders. lets inspect our data and see how they look ! 
imgs, labels = next(iter(dataloader_train))
# lets check the value range in our images: 
print(f'imgs.min()= {imgs.min().item()}')
print(f'imgs.max()= {imgs.max().item()}')
# remember our images, are not tensors, and in order to display them using matplotlib
# we must convert them back to normal numpy arrays and again since we normalized them 
# we must unnormalize them for visualization as well. 
# lets write a function that accepts a batch of images with their labels and displays them! 
def visualize_imgs(imgs, labels, row=3, cols=11):
    # images in pytorch have their axes swapped! 
    # so we first fix their orders firts! 
    # the dim is batch, c, h, w, but we should be having 
    # batch, h, w, c  
    # we used detach to clone images (we dont have to do this, but its good practice for later)
    imgs = imgs.detach().numpy().transpose(0, 2,3,1)
    # now we need to unnormalize our images but since 
    # we didnt normalize our images, we dont need to 
    # do anything here!
    # imgs = (imgs+1)/2
    # figsize takes two arguments, which specify the column(width) and row(height)
    fig = plt.figure(figsize=(20,5))
    for i in range(imgs.shape[0]):
        ax = fig.add_subplot(row, cols, i+1, xticks=[], yticks=[])
        # since our mnist images are 1 channel only, we remove the channel dimension
        # so that matplot lib can work with it (we changed 28x28x1 to 28x28!)
        ax.imshow(imgs[i].squeeze(), cmap='gray')
        ax.set_title(labels[i].item())
    plt.show()

visualize_imgs(imgs, labels)

# OK, lets see an image in more details!
# lets write a function that recieves a tensor image
# and after converting it to a normal image, draws it
# in a way, each pixel of the image shows its pixel intensity.
# this works for grayscale images. 
# to do this we use matplotlib's annotate() function
# which takes in a text and displays it at the given x,y coordinate
# in a plot.
def visualize_img(img):
    img = img.numpy().transpose(1,2,0).squeeze()
    fig = plt.figure(figsize=(28,28))
    ax = fig.add_subplot(1,1,1)
    # or 
    #ax = plt.subplot(111)
    ax.imshow(img, cmap='gray')
    threshold = 0.5
    w,h = img.shape
    # just loop through the image pixel values, and individually
    # print their brightness value at each location.
    for i in range(h): 
        for j in range(w):
            ax.annotate('{:.2f}'.format(img[i,j]),xy=(j,i),
            horizontalalignment='center',
            verticalalignment='center',
            color='white' if img[i,j]<threshold else 'black')

visualize_img(imgs[0])

#%%
# Before we continue with training a model, I'd like to point out a useful feature here
# note 1 concerning new transformations. 
# previously we just saw how to augment our data using torchvision.transforms module
# we saw there are many transformations that we can use. what about something new? 
# how can we add our own transformations? 
# the transformations that we saw and used such as transforms.ToTensor(), transforms.Resize()
# etc are called functors. you can create new functors and add them to the compose list!
# a functor is simply a class with a __call__(self, *args) method. in Python terminalogy 
# it is called as 'Callable'! for example, the ToTensor() funtcor can be implemented 
# like this : 
import torchvision.transforms.functional as F
class ToTensor(object):
    def __call__(self, pic):
        return F.to_tensor(pic)

# or a funcor that accepts parameters like resize for example, can be as simple as : 
class Resize(object):
    def __init__(self, size, interpolation):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, input):
        return F.resize(input, self.size, self.interpolation)

# as you can see you can write as many custom transformations as you like!
to_tensor_f = ToTensor()
import PIL.Image as Image
resize_f = Resize(6, Image.BILINEAR)
numpy_tensor = np.random.rand(3, 3, 3)
result_tensor = to_tensor_f(numpy_tensor)
print(result_tensor)
#resize this tensor!
x=resize_f(result_tensor)
print(x)

# I'll be covering the dataset related chores in detail in the upcomming parts, but for 
# now lets cover this. 
# Suppose we dont have separate sets for our trainning (such as training set, validation set, test set)
# and we want to create these by ourseleves, what should we do? 
# lets take MNIST as our example. as it only has a training set and a test set but no validation set. 
# as you will see what we describe here will be applicable to just anything.
# in Pytorch we have something called a sampler that as the name implies, samples! 
# basically a sampler defines the strategy to draw samples from a dataset.
# we have different kind of samplers. what we are after is a sampler called 'SubsetRandomSampler'
# we can access it from 'torch.utils.data' module .
# This class samples elements randomly from a given list of indices, without replacement.
# without replacement simply means the values are unique.
# what this class needs is a list of indeces. lets create ourselevs a list of indeces : 

dataset_train = datasets.MNIST(root='./data/MNIST/', train=True, transform=transformations, download=True)
dataset_test = datasets.MNIST(root='./data/MNIST/', train=False, transform=transformations, download=True)

train_num_samples = len(dataset_train)
# this simply gives us a list of indexes starting from 0 - 59999
train_indexes = list(range(train_num_samples))
print(f'some training indexes[:5] : {train_indexes[:5]}')
# now let us shuffle our indexes, shuffle is an inplace operation
# so it changes the list items orders.
np.random.shuffle(train_indexes)
print(f'some training indexes[:5] : {train_indexes[:5]}')

# now we have a list of indexes. lets specify our validation ratio from this list. 
# here we are specifying that 20% of our data is reserved for validation, 
val_ratio = 0.2
val_end = int(train_num_samples * 0.2)
validation_split_indexes = train_indexes[0:val_end]
training_split_indexes = train_indexes[val_end:]
# lets view the changes 
print(f'training size before split: {train_num_samples}')
print(f'validation size: {len(validation_split_indexes)}')
print(f'training   size: {len(training_split_indexes)}')
# make sure the splits are actually correctly done! 
assert len(validation_split_indexes) + len(training_split_indexes) == train_num_samples ,'they must match!'

# Now we have our list of indexes, what remains is to create a sampler that samples from 
# these lists
sampler_train = torch.utils.data.SubsetRandomSampler(training_split_indexes)
sampler_val = torch.utils.data.SubsetRandomSampler(validation_split_indexes)

# and the last step is just to create a dataloader.
# note that we dont use shuffle here. as they are mutually exclusive (i.e. they can not
# come together!) with samplers.
dataloader_train = torch.utils.data.DataLoader(dataset_train,
                   batch_size=32,
                   sampler=sampler_train,
                   num_workers=2)
dataloader_val = torch.utils.data.DataLoader(dataset_train,
                  batch_size=32,
                  sampler=sampler_val,
                  num_workers=2)
# check some samples 
imgs, labels = next(iter(dataloader_val))
visualize_imgs(imgs, labels)
# sidenote: torchvision also offers many utility functions 
# for example we can display a batch of tensor images using make_grid()!
plt.imshow(torchvision.utils.make_grid(imgs).permute((1,2,0)).numpy())
# but our function is more versatile and includes labels as well, so we stick
# to our own function for visualization. 
#
# sidenote 2: 
# torchvision.io also hosts functions to read image/video files into tensors 
# or save them to disk directly. torchvision has added many functionalities 
# since 5/6 years ago!
#
# so if you have a folder of images, you can use datasets.ImageFolder() 
# and then use SubsetRandomSampler() to split your data into
# different sets. here we only created a validation set out of our train
# -ing set. but you can do as many split as you like. 
# we will cover ImageFolder in later sections

#%%
# OK now that we have done this lets see how we can train a model ! 
# before that we need a model 
# we can create one or use an existing one, 
# first lets see how to use an existing model ! 
model = models.AlexNet(10)
# thats it, now we have a model the that we can use for training! 
# however there is a catch here, like always!
# this is alexnet and alexnet was trained for imagenet challange. 
# in imagenet chalange the image size that were used to train this 
# model were 224x224x3. so this model accepts 224x224x3 images as input only!
# while our input is a 28x28x1 images! this wont simply work!!! 
# either we have to resize our images from 28x28 to 224x24! (using resize() in transformers up there!)
# and also make images 3 channeled instead of 1! 
# this is too much work for now! instead lets create our own model first and 
# then later on see how we can use these models on new data!

# for creating a model, we simply define a new class that inherits from torch.nn.Module() 
# and then define our layers and ultimately specify the sequence of how these layers are
# used in forward() method. 
# lets see how all of this can be implemented! (this is very easy!)
import torch.nn as nn 
# torch.nn.functional module(usually imported into the F namespace by convention)
# is a module which contains activation functions, loss functions, etc, as well 
# as non-stateful versions of layers such as convolutional and linear layers, among others.
import torch.nn.functional as F 

class ourNetwork(torch.nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # lets create a simple multilayer preceptron (a 4 fully connected layer network!)
        # we have several ways for creating layers. the simplest form is like this
        # we define many layers we need here and then in the forward() we specify their order
        # nn.Linear() gives us a fully connected layer. since our image is 28x28,
        # and we are using fully connected layers, this means our input dimension is 28x28
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        
        # this method is called in our forward phase
        # we can recieve any number of parameters here,
        # but the default is one parameter which is the input batch!
        # we can also have none! and we will see an example concerning this
        # later on! for now, our forward method takes one argument and that 
        # is a batch of images 
    def forward(self, input_images):
        # since we have images, but our first layer
        # is fully connected, we have to flatten our input images
        # i.e. instead of being a matrix, we convert them into vector!
        # we will not change the underlying data, we are only going to change
        # how it looks (thus change its shape!) instead of seeing them in rows columns
        # (28 rows, 28 columns ), suppose we have one row and 28x28 columns! 
        # as you remember, we use view or reshape for this (also resize_ but we chose
        # view becasue we explained why previously!
        # for flattening we do this in two ways!
        # 1. we directly enter the size since we know the dims. 
        # input_images = input_images.view(-1, 28*28)
        # the bad thing about this is that if later we change our image dims
        # we have to comeback here and change the dims to match the new dims (e.g. 32x32)
        # a better way would be to specify the batchsize and then let the dims be automatically
        # infered like this : 
        batch_size = input_images.size(0)
        input_images = input_images.view(batch_size, -1)
        
        # sidenote, initially the pytorch team religiously refused to add a 
        # flatten layer, but later on after years of requests, they finally 
        # added a Flatten() layer to nn modules. we'll visit this layer later on.
        
        # to see how it changed! uncomment this
        # print(input_images.shape)
        output = self.fc1(input_images)
        # now we need to use a nonlinearity! we can use
        output = self.relu(output)
        # or we could also use the functional form which is 
        output = F.relu(self.fc2(output))
        output = F.relu(self.fc3(output))
        output = F.relu(self.fc4(output))
        return output

# thats it! thats all it needed to create a model. 
# now before we continue lets test our model and see if 
# we implemented everything corectly and we have no error whatsoever!
# lets create a dummy batch of fake images (which are just tensors with random vlaues!)
# the shape representes, batch, h,w,c
fake_images = torch.rand(size=(3,28,28,1))
our_model = ourNetwork(num_classes=10)
output = our_model(fake_images)
print(f'output: {output}')
# so far so good!  
#%% 
# now lets go for the next part which is training /testing. 
# we will need 
# 1. an optimizer such as sgd or adam, etc 
# 2. a criterion (a loss function ) 
# 3. thats all. 
# all optimizers reside in torch.optim 
# from torch import optim 
# our optimizers, take at least two paramters:
# 1. model parameters and 
# 2. the learning rate 
# they can take, weight decay, momentum etc as well, but the model parameters,
# and lr are the very minimum requirements. 
# remember if you set too of a high learning rate, you will see your loss
# will not decrease and may also not increase!! it may very well get stuck
# to or around a value . therefore your val_acc would also perform similarly.
# When you use BatchNormalization, you may not see this happening as often, 
# as it allows you to use considerably higher learning rates but knowing this is 
# important anyway. (we'll talk about normalization effects and BatchNorm specifically later on.)
# set this learning rate to 0.1 forexample and see the outcome. then reset it to 0.001 
# and rerun the training loop and witness the change.
optimizer = torch.optim.Adam(our_model.parameters(), lr = 0.001)
# for our loss function, its customery to use the name criterion, 
# you can use any name you like, but criterion is overwelmingly popular! 
# different loss functions can be found under torch.nn module 
# we use crossentropy for our classification task! 
criterion = nn.CrossEntropyLoss()

# now we are ready to start our training loop. 
# before that lets write code that allows us to 
# utilize GPU if its present 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
our_model = our_model.to(device)

epochs = 20
# at this interval we display our training loss or other informations we like
# for example run our model on test data to see the validation acc/loss or pretty much
# anything else we would like. you name it!
interval = 500 
# a counter! 
i = 0
training_losses = []
validation_losses = []
for e in range(epochs) :

    for i, (imgs, labels) in enumerate(dataloader_train, start=1): 
        # we set our model to train mode 
        # this is specially needed if we use dropout or batchnormalization
        # or pretty much any layers that during training and testing have different behaviors!
        our_model.train()
        
        # instead of creating a manual index, we used enumerate() functionality
        # this is more pythonic!
        # i+=1
                
        # in case we have access  to gpu, we must move 
        # all images, labels to gpu. so we should do 
        imgs = imgs.to(device)
        labels = labels.to(device)

        # now feed our images to the model and get the predictions 
        preds = our_model(imgs)
        # we can also write 
        # preds = our_model.forward(imgs)
        # however, the first style is more popular you can choose either of the two!
        # now lets see the loss 
        loss = criterion(preds, labels)
        # always do this before backpropagating the loss 
        # this zeros the gradient from previousl steps 
        optimizer.zero_grad()
        # backpropagate the loss 
        loss.backward()
        # now update the weights (take one step of the optimzier) 
        optimizer.step()

        if i % interval == 0 : 
            # since loss is a tensor, in order to see a python scaler, we use item()
            # else  the print will show the verbose tensor information becasue its a tensor!
            print(f'Epoch [{e}/{epochs} | Iter: {i}/{len(dataloader_train)}] Training-Loss: {loss.item():4f}')

    # lets run evaluations on test set after each epoch :
    total_corrects = 0
    total_loss_val = 0.0
    acc_val = 0.0
    # create a dictionary to keep track of each class correct predictions
    # its like doing torch.zeros(size=(10,), dtype=torch.int32)
    class_correct_counter = {k: 0 for k in range(10)}
    class_total_samples = {k: 0 for k in range(10)}
    total_corrects2 = 0
    total_corrects_top5=0
    incorrect_imgs_predicted_list = []
    for imgs, labels in dataloader_test:
        our_model.eval()
        
        imgs = imgs.to(device)
        labels = labels.to(device)
        preds = our_model(imgs)
        loss_val = criterion(preds, labels)
        # now lets see how the predictions looks like and what accuracy we get 
        # we can calculate the accuracy using a different ways! while doing this 
        # will also get familiar with some gotchas and hopefully learn to avoid them.
        # we'll see 3 ways of doing it, we start with the easiest/normalest way first. 
        # first method : 
        # in the first method, we get the index of highest probabilities in our predictions 
        # and compare them to the labels. our comparsion will give us a series of
        # true/false values which indicate, where our predictions were inline with our labels
        # and where they were different. 
        # we can then simply convert true /false to 1 and 0, and
        # then by suming all of them we get the total number of correct predictions. 
        # then dividing this number by the total number of samples in our test set, 
        # we get our accuracies.
        # we use torch.max() to get the highes values in a tensor.  
        values, class_indexes = torch.max(preds,dim=1)
        # not only max returns the highest values (maximum values)
        # it also gives us the index of those values. We use the index
        # to see which class(index) was predicted as correct by the netwok 
        # and compare it agains the labels (our true class)
        result = (class_indexes == labels)
        # or we could also use torch.eq(a, b) (and NOT torch.equal()) for this. 
        # Remember that torch.equal(a,b) says whether the two tensors are 'equal' or 'not' 
        # while torch.eq(a,b) says whether 'each' element is the same or not 
        # thus torch.equal() will return a 'single' True or False value as result,
        # while torch.eq() will return a matrix of boolean values each indicating if the 
        # respective elements were the same or not! therefor we use torch.eq becasue we 
        # want to check all elements in tensors agains each other: 
        # result = torch.eq(class_indexes, labels)
        
        # to see its shape or content you can uncomment this
        # print(result.shape)
        # result is a matrix of boolean values. lets sum them
        # before that lets convert True/False to 1 and 0s
        # this way we will treat True as 1 and False as 0. the sum therefore
        # will give us the number of true classes that were 
        # predicted correctly .
        # since we read in batches, we add all the correct predictions 
        # in each batch and then divide it by the total number of samples 
        # and get our accuracy! https://pytorch.org/docs/stable/torch.html#torch.eq
        total_corrects += result.float().sum()

        # 2. method 
        # There is a second way we can calculate the accuracy. and that is using topk!
        # suppose, we want to get the accuracy for top5 (like in imagenet)
        # how could we do it ? there are many ways but one of the easiest ways is using topk
        # lets see how we can do that: 
        # each tensor has a method named topk, topk accepts an argument k, which we specify
        # if we want top5 we set k=5, if we want top1 (which is normal accuracy) we set k=1. 
        # topk, like max, returns two results. the first is the top values and the second is
        # their respective index. so lets see it in action.
        # theres a slight catch here for k>1. for now lets work out the k=1 case
        top_values, indexes = preds.topk(k=1, dim=1)
        # print(f'{indexes.shape=}, {labels.shape=}')
        # we use squeeze on indexes so its shape becomes [32] instead of [32x1]
        result2 = torch.eq(indexes.squeeze(), labels).float()
        # remember since we are using topk, our 'indexes' has  a shape of [32, 1]
        # (1, becasue we are taking topk=1), while our label shape is [32]. 
        # this is important, because if we compare these two as is, in 
        # newer versions of Pytorch (0.4+) this will be broadcasted and the result will
        # be a tensor of [32,32] while it should have been [32]. (32 is our batch size/ number of samples)
        # If we do (indexes == labels) or torch.eq(indexes, labels) 
        # results will have the shape (32, 32).
        # What it's actually doing is comparing the one element in each row of indexes
        # with each element in labels which returns 32 True/False boolean values for each row.
        # therefore in order to not face this issue (which will not give you any error,
        # but mess up your result!), always make sure what you are comparing with and 
        # make sure the the tensors involved have the same exact shape. 
        # one way is to use squeeze, which we just saw! 
        result2 = (indexes.squeeze() == labels)
        # the other way is to make sure the 
        # first tensor has the same shape as the second one : 
        result2 = (indexes.view(*labels.shape) == labels)
        # uncomment to see the shapes 
        # print(f'indexes: {indexes.shape} result(eq) {result.shape} result2(==) {result2.shape}')
        
        total_corrects2 += result2.float().sum()
        
        # now lets do a top5: 
        _, indexes_top5 = preds.topk(k=5, dim=1)
        # now for k>1, like k=5, we would get a shape (32,5) while our label is (32,)
        # simply trying to compare them like before wouldnt obviously work as the shapes dont match
        # and more importantly we need to compare all 5 predictions against the groundtruth
        # and if any of those 5 predictions matched the groundtruth, we mark it a match. 
        # how do we do that? a simple but crude way would be to manually compare each predictions
        # with labels, and then sum the results. like [32,5] means instead of 1 predictions
        # we have 5 predictions, so we do the comparasion 5 times. since the result is boolean
        # simply adding them together should give us the right result.(we are looking for
        # correct predictions, and correct predictions are True or 1, the wrong ones will be 0
        # so if we add all 5 results, if one of the predictions were correct, the final matrix
        # will have 1.ok?)
        # but this is not efficient!
        result2_top5 = np.sum([torch.eq(indexes_top5[:,c], labels).float().cpu().numpy() for c in range(5)])
        # we can do this much better: 
        result2_top5 = (indexes_top5 == labels.view(-1, 1)).any(dim=1).float().sum().item()
        # print(f'{(indexes_top5 == labels.view(-1, 1)).shape=}')
        # since on one side we have 5 predictions for each row/sample, while
        # on the other side, have a single label for each row/sample and we want
        # to check all of those predictions and see if any of them were correct, 
        # to mark the result as correct, we can use the broadcasting to our advantage here.
        # to allow for the broadcasting to happen, we need to prepare labels to have the
        # shape (batch_size,1) so that the dim=1 (i.e. the label dimension) gets replicated to match
        # the shape of the other tensor. 
        # therefore (indexes == labels.view(-1, 1)) makes labels to be broadcasted and have the
        # shape (32,5), basically repeating the column 5 times. 
        # after that it becomes a simple elementwise comparison between the two tensors of the same shape.
        # our result will also be (32,5), we need (32,1)/(32,). 
        # so to get there, we use .any(dim=1) which scans along the dim=1 (dim=0 is batchsize), 
        # and upon encountering the first True(or match), it returns True for that row/sample, 
        # thus giving us the final (32) shape!
        # .float().sum() calculates the total number of correct predictions.
        total_corrects_top5 += result2_top5
        
        # 3rd Method  :
        # now we just found out about two ways of calculating the accuracy , so far we counted
        # the number of correct samples and then divided them by the total number of samples 
        # at the end and got our accuracy. we could also calculate accuracy per batch and then 
        # add all these accuracies and then divide them by the total number of batches. 
        # this is easily done as well. lets see how we can do this : 
        # we have our predictions, and labels, using max, or topk we can get the indexes
        # of the highest predictions. 
        # so if we add and average all of them it should give us the accuracy per batch! 
        # but there is a catch here, you must make sure the dimensions of labels, and predictions
        # indexes are the same. if they are not the same, and e.g. one is [N,1] while the other is [N]
        # it will be broadcased and the result will be a [N,N] tensor and will mess up your result completely!
        # so here just like before, we make sure the two tensors has the same shape 
        # (if we used max, the indexes and labels both would have the same shape and this wasnt necessary
        # but its good practice to havethis in mind and always make sure the shapes match exactly so we dont
        # spend a lot of time debugging something that may waste a lot of our time)
        result3 = (indexes.view(*labels.shape) == labels)
        acc_val += torch.mean(result3.float())
        # we can get number of batches, simply by doing len(dataloader_test) !
        # note we are using loss.item() to get the number, otherwise it would
        # accumulate the whole computation graph, and you'd have a memory leak! 
        total_loss_val += loss_val.item()

        # lets see a more detailed analysis concerning our model performance
        # that is, lets see, which classes are being predicted better than others 
        # for this, we will count the number of samples for each class and also its 
        # correct predictions by model. 
        # this can be easily done. 
        # we need two arrays/list which has as many elements as our class numbers 
        # lets do this 
        for i in range (labels.size(0)):
            class_total_samples[labels[i].item()] +=1 
            class_correct_counter[labels[i].item()] += (indexes[i] == labels[i]).item()
            # now lets get a bit more fancy and also save all the images that were uncorrectly
            # classified. its logical to only when we trained our model and in evaluation time
            # we are trying to see which images were hard for the network and get an idea and 
            # hopefully comeup with some solutions to fix the issue. during training this is not
            # recommended becasue it imposes a lot of over head!
            lbl = (indexes[i] == labels[i]).item()
            if lbl == 0 :
                # we save a tuple containing the image, the predicted class, and the actual class
                # this way later on we can not only see the wrongly predicted image, but 
                # the wrong class that it mispredicted it as 
                incorrect_imgs_predicted_list.append((imgs[i],indexes[i],labels[i]))

    for i in range(10):
        print(f'Class {i} ({class_correct_counter[i]}/{class_total_samples[i]}):'\
                f' class acc: {class_correct_counter[i]/class_total_samples[i]:.4f}')
    
    print(f'test set samples: {len(dataloader_test)}')
    print(f'accuracy_val(total corrects//dataset_size): {total_corrects/len(dataloader_test.dataset):.4f}'\
          f'\naccuracy_val(topk(k=1)): {total_corrects2/len(dataloader_test.dataset):.4f}'\
          f'\naccuracy_val(topk(k=1)-per batch acc): {acc_val/len(dataloader_test):.4f}'\
          f'\naccuracy_val(topk(k=5)): {total_corrects_top5/len(dataloader_test.dataset):.4f}'\
          f'\nloss_val: {total_loss_val/len(dataloader_test.dataset):.6f}')

#%%
# ok we learned a lot, now lets write the training loop one more time, 
# but this time in much clearer fashion so we can see whats going on clearly!
# to train from scratch instantiate the model again
model = ourNetwork(num_classes=10)
# lets instantiate the optimizer as well 
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
k=5

for epoch in range(epochs):
    
    train_accuracy = 0
    train_loss = 0
    val_accuracy = 0
    val_accuracy_top5 = 0
    val_loss = 0
    class_correct_count = {k:0 for k in range(10)}
    total_class_samples = {k:0 for k in range(10)}
    incorrectly_predicted_images = []
    for i, (imgs, labels) in enumerate(dataloader_train, start=1):
        model.train()
        
        imgs = imgs.to(device)
        labels = labels.to(device)
        
        # forward
        preds = model(imgs)
        
        # calculate loss
        loss = criterion(preds, labels)
        
        # do backward pass, 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # lets calculate training accuracy 
        train_accuracy += (preds.max(dim=1)[1].squeeze() == labels).float().mean().item() * 100
        train_loss += loss.item()
        
        # print the progress so far
        if i%interval==0:
            print(f'Epoch [{epoch:02d}/{epochs} - {i:04d}/{len(dataloader_train)}] | Loss: {train_loss/i:.4f} | Train-Acc: {train_accuracy/i:.2f}%')
        
    # run validation for each epoch
    for imgs, labels in dataloader_val:
        # Disable gradient calculation to preserve vram
        # Disabling gradient calculation is useful for inference,
        # when you are sure that you will not call Tensor.backward().
        # It will reduce memory consumption for computations that 
        # would otherwise have requires_grad=True.
        # In this mode, the result of every computation will have 
        # requires_grad=False, even when the inputs have requires_grad=True.
        # There is an exception! All factory functions, or functions that
        # create a new Tensor and take a requires_grad kwarg, will NOT be 
        # affected by this mode
        with torch.no_grad():
            model.eval()
            
            imgs = imgs.to(device)
            labels = labels.to(device)
            
            val_preds = model(imgs)
            val_loss += criterion(val_preds, labels).item()
            
            # val accuracy
            _, indexes = val_preds.max(dim=1)
            val_accuracy += (indexes.squeeze() == labels).float().mean().item()*100
            val_accuracy_top5 += (val_preds.topk(k,dim=1)[1] == labels.view(-1,1)).any(dim=1).float().mean().item()*100
            
            # lets calculate some statistics:
            for i in range(labels.size(0)):
                total_class_samples[labels[i].item()] += 1
                label_value = (indexes[i] == labels[i]).item()
                class_correct_count[labels[i].item()] += label_value
                
                # if label was wrong
                if not label_value:
                    incorrectly_predicted_images.append((imgs[i], indexes[i], labels[i]))
            
            # test vectorized 
                
    print(f' -- Val-Loss: {val_loss/len(dataloader_val):.4f} | Val Acc: {val_accuracy/len(dataloader_val):.2f} | Val-Top5: {val_accuracy_top5/len(dataloader_val):.2f}')
    
    # display class stats
    for c in range(10):
        total_samples = total_class_samples[c]
        correct_preds = class_correct_count[c]
        class_level_acc = (correct_preds/total_samples) * 100
        print(f'    -- class {c}: {correct_preds:04d}/{total_samples:04d} | class-level accuracy: {class_level_acc:.2f}')
    
print(f'Training Finished!')
# its now a bit better than the previous one but we still have a lot to improve here
# we can improve just about everything here, infact, in future sections and chapters, 
# we will be adding more to this training loop and making it better/ more maintainable 
# hopefully at the end it would look much better.
#%%
# lets visualize our wrongly predicted images and their wrong/true classes 
def visualize_wrongly_predicted(imgs_list, cols = 10):
    count = len(imgs_list)
    rows = int(np.ceil(count/cols))
    print(count,rows)
    fig = plt.figure(figsize=(cols, rows))
    plt.subplots_adjust(wspace=1, hspace=1)
    for i in range(count):
        ax = fig.add_subplot(rows, cols, i+1, xticks=[], yticks=[])
        (img, pred_class, true_class) = imgs_list[i]
        # convert from tensor to numpy image. squeeze because its 28x28x1 and 
        # matplotlib accepts 28x28 for grayscale images!
        img = img.detach().cpu().numpy().squeeze()
        ax.imshow(img,cmap='Greys_r')
        ax.set_title(f'({pred_class.item()} | {true_class.item()})')
    
    # plot the highest wrong class

visualize_wrongly_predicted(incorrectly_predicted_images)
#%%
# OK great we created a network from scratch and trained/tested it successfully . 
# you probably have couple of questions(ok many questions!). one of them is probabaly
# concerning the way we created our network. if you look back again, you'll see we didnt use 
# any softmax layer at the end? why? 
# as you know, in classification problems, we usually use a softmax layer at the very end
# of the network to get probablities for our predictions. 
# we also know that we usually use softmax with crossentropy loss. 
# Yet here we used crossEntropy without softmax? why is it?

# we can use softmax with crossentropy but if you read the documentations,
# you'll notice that when we used crossentropyloss, it acually requires 
# 'logits' or 'raw scores' as inputs. 
# it then applies a 'logsoftmax' on its input (logits), and then applies 
# a negative log likelihood afterward.
# So we can have softmax as our last layer, but for calculating loss, we 
# must send the logits.

# But Why do we use logits and not softmax itself? 
# we use logits and not softmax, becasue softmax gives us probablities, 
# which are floatingpoint numbers ranging from 0. to 1. and the critical
# issue with floatingpoint numbers is that they can not accurately 
# represent numbers close to 0 or 1 and thus we face numerical instabilities.
# especially when there is multiplication involved. imagine what would happen
# if we were to multiple several already tiny floatingpoint numbers! the result
# would become too small too quickly!
# 
# OK, but what about log_softmax? why do we use log_softmax instead of softmax then?
# there are several reasons for this. one of them is that log_softmax is numerically
# stable and doesnt have the problems associated with softmax. it also plays nicer with
# crossentropy (https://datascience.stackexchange.com/questions/40714/what-is-the-advantage-of-using-log-softmax-instead-of-softmax) 
# so in short, having log probabilities, help both in numerical stabilty and optimization
# performance.

# From wikipedia  :
#  A log probability is simply a logarithm of a probability. The use of log probabilities means 
# representing probabilities on a 'logarithmic scale', instead of the standard [0,1] unit interval.

# Since the probability of independent events multiply, and logarithms convert multiplication to addition,
# log probabilities of independent events add. 
# Log probabilities are thus practical for computations, and have an intuitive interpretation 
# in terms of information theory: 
# the negative of the log probability is the information content of an event. 
# Similarly, likelihoods are often transformed to the log scale, and the corresponding log-likelihood
# can be interpreted as the degree to which an event supports a statistical model. 
# The log probability is widely used in implementations of computations with probability, and is studied
# as a concept in its own right in some applications of information theory, such as natural language processing. 

# Representing probabilities in this way has several practical advantages:
#    Speed: Since multiplication is more expensive than addition, taking the product of a 
#           high number of probabilities is often faster if they are represented in log form.
#           (The conversion to log form is expensive, but is only incurred once.) 
#           Multiplication arises from calculating the probability that multiple independent
#           events occur: the probability that all independent events of interest occur is 
#           the product of all these events' probabilities.
#    Accuracy: The use of log probabilities improves numerical stability, when the probabilities
#              are very small, because of the way in which computers approximate real numbers.
#    Simplicity: Many probability distributions have an exponential form. Taking the log of 
#                these distributions eliminates the exponential function, unwrapping the exponent.
#                For example, the log probability of the normal distribution's PDF
#                is -(x-m_{x}/\sigma _{m})^{2}+C} instead of C_{2}\exp(-(x-m_{x}/\sigma _{m})^{2}).
#                Log probabilities make some mathematical manipulations easier to perform

# Ok a quick not so related question : 
# whats the difference between sigmoid and softmax by the way?

# softmax is an extension of sigmoid. sigmoid 
# is used for binary classification or multi-label classification where there arent mutually exclusive 
# classes. softmax is used when classes are mutually exclusive and  there is only "ONE" correct class
# also remember that the log-softmax has a 
# https://stats.stackexchange.com/questions/233658/softmax-vs-sigmoid-function-in-logistic-classifier

# recap from before : softmax : 
# Exercise: Implement a function softmax that performs the softmax calculation 
# and returns probability distributions for each example in the batch. Note that
# you'll need to pay attention to the shapes when doing this. If you have a 
# tensor a with shape (64, 10) and a tensor b with shape (64,), doing a/b will
# give you an error because PyTorch will try to do the division across the 
# columns (called broadcasting) but you'll get a size mismatch. The way to think
# about this is for each of the 64 examples, you only want to divide by one value,
# the sum in the denominator. So you need b to have a shape of (64, 1). 
# This way PyTorch will divide the 10 values in each row of a by the one value in
# each row of b. Pay attention to how you take the sum as well. You'll need to 
# define the dim keyword in torch.sum. Setting dim=0 takes the sum across the rows
# while dim=1 takes the sum across the columns.

# so the moral of the story : use logsoftmax instead of softmax 
# if you used logsoftmax in your network, use negativeloglikelyhood or NLLLoss
# to get probablities at test time.
# when you used logs0ftmax, just use torch.exp() on your result to get probabalities 
# and you re done. that was all for now! 

#%%
import torch
# now in this section we are going to learn how to do some initializations 
# there are several ways you can do this . 
# 1. using torch.nn.init module which provides lots of initialization algorithms 
#    such as xavier, rmsa, etc
# 2. using model.apply() 
# 3. directly initializing the weights and biases. 


# lets start with the direct method first. 
# as you know, each module may contain a weight and a bias. 
# we can access them directly. each weight has a data and a grad attribute. 
# the data as the name implies, contains the data, while grad contains the weights 
# gradients. 
# this is the case for bias as well, it has a data and a grad attribute as well. 
# lets create a simple fully-connected layer and initialize its weights and biases
# in different ways:
fc = torch.nn.Linear(1,1)
# initialize using normal distribution 
# since data is an oridinary tensor, it has access to all methods available to any tensors
# including the inplace normal_ method which samples from a normal distribution 

print(f'default: {fc.weight.data}')
fc.weight.data.normal_(mean=0.5, std=1)
print(f'normal_: {fc.weight.data}')

# initialize using uniform distribution [0,1] 
print(f'default:_ {fc.weight.data}')
fc.weight.data.uniform_(0.01,0.1)
print(f'uniform:_ {fc.weight.data}')

# for initializing bias we can simply do : 
print(f'bias(defaul): {fc.bias.data}')
fc.bias.data.normal_(0.5,1)
print(f'bias(normal): {fc.bias.data}')
# or more famously just initialize all by zero or 1 or etc 
fc.bias.data.fill_(1)
print(f'bias(1): {fc.bias.data}')
# now since there is no operation done yet there is no gradients either yet! 
print(f'grad: {fc.weight.grad}')

# the second way by which we can initialize a modules weights and biases
# is using torch.nn.init module. basically this is what we use nearly 99%
# of the times. 
# using it is straight forward 
torch.nn.init.normal_(fc.weight, mean = 0.0, std=1.0)
# or 
torch.nn.init.uniform_(fc.weight, a= -1.0, b = 1.0)
# initializing using a constant
torch.nn.init.constant_(fc.weight, 0.1234)
torch.nn.init.constant_(fc.bias, 0.9999)

print(f'fc.weight.data: {fc.weight.data}, \nfc.bias.data:   {fc.bias.data}')

# using xavier initialization 
torch.nn.init.xavier_normal_(fc.weight, gain=torch.nn.init.calculate_gain('relu'))


# now suppose we want to initialize all layers using a specific initialization scheme, 
# how do we do it?
# its easy we do it like this : 
#first lets create a simple dummy model 
import torch.nn as nn 
model = torch.nn.Sequential(*[nn.Linear(1,1),
                              nn.ReLU(),
                              nn.Conv1d(1,1,1),
                              nn.ReLU()])
# we iterate over all 'modules' and initialize
# their paramertes (weights and biases) 
for m in model.modules():
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight, torch.nn.init.calculate_gain('relu'))
        torch.nn.init.constant_(m.bias, 1)
        print('linear initialized!')
    elif isinstance(m, nn.Conv1d):
        torch.nn.init.kaiming_normal_(m.weight, a = 0)
        torch.nn.init.constant_(m.bias, 1)
        print('conv initialized!')


# we can define this in our architecture and initialize all modules. 
# but what if we have a model, and after its creation we want to change the initialization!
# here we use model.apply(). 
# we first write down a function that does the intialization and then apply this function to
# the model
# this is how it is done. 
# first write a function and do your thing!
def initialize_some_layers(m):
    if isinstance(m, nn.Linear):
        nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, 0)

# showing the first modules weight before the new initializtaion
print('before: ',model._modules['0'].weight.data)
model.apply(initialize_some_layers)
# after the initialization :
# quick note. our model has a .modules() method which is a generator 
# each time we call module(), it returns one module form _modules. 
# we here however used the underlying _modules which is a dictionary 
# since when creating our modules, we didnt specify a name, numbers 
# specify each module, thats why we used '0', we will learn about this 
# more in the next section.  
print('after: ',model._modules['0'].weight.data)
#thats it 
#%%

# so that was it!! you now know how to initialize a tensor/ a modules parameter in different ways. 
# in this section we will learn several other ways of creating networks . 
# previously we just learnt one way of creating a network which was simply by creating 
# some layers and then use them in the order we wanted. here we will learn more ways of doing this.
# 
# 
#  
# we will create networks in several different ways that are as follows: 
# 1. using the simple layer definition (what we just saw)
# 2. using nn.Sequential() to create a sequential model 
# 3. using ordered_dict to create named modules. 
# 4. using ordered_list to create a list of layers. 
# 5. using new modules 
# This list probably doesnt make any sense to you, but when we try implementing them 
# you'll understand why we covered each here. 
# lets start with the first method: 
# 1. In the first method, we first define all of the layers we need and then in forward() 
# we simply call them and use them in any order we like. here is a simple example 
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(6, 12, 3)
        self.pool1 = nn.MaxPool2d(2,2)
        self.relu2 = nn.ReLU() 
        
    def forward(self, x):
        output = self.conv1(x)
        output = self.relu1(output)
        output = self.conv2(output)
        output = self.relu2(output)
        output = self.pool1(output)
        return output

# as you can see we defined an attribute for each layer we wanted to use. 
# we can do better than this. for example , we can avoid defining two relu layers
# and instead use the functional form instead. 
# we can access the functional forms from torch.nn.functional module.  
import torch.nn.functional as F 
class SimpleNet2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.conv2 = nn.Conv2d(6, 12, 2)
        self.pool1 = nn.MaxPool2d(2,2)

    def forward(self, x):
        output = F.relu(self.conv1(x))
        output = F.relu(self.conv2(x))
        return self.pool1(output)

# as you may have guessed we can use the functional form for any of these layers, its up to you
# when and where to use either of them. usually we use activation functions this way as
# its more consise, easy to use and more readible.
# Also as you know, you can have control over which layer gets executed, so you can have
# different if clauses in your foward() as well. 

# 2. using nn.Sequential() class
# If we want to create a plain network/module which is just a series of layers in succession, 
# we can ease ourseleves using the nn.Sequential() class. this class as the name implies
# runs a series of layers in succesion. 
# nn.Sequential class is in fact a sequential container. 'Modules' will be added to it in
# the order they are passed in the constructor. 
# Alternatively, an ordered dict of modules can also be passed in.
# As you see, there are several ways we can create a network using nn.Sequential() class. 
# below I'll show some ways that comes to my mind: 
# first lets see what the : 
# "'Modules' will be added to it in the order they are passed in the constructor. "
# mean. this means, the layers order are specified when we specify them in the constructor
# see the example below to understand this :  

class SequentialNet1(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.net = nn.Sequential(nn.Conv2d(3, 6, 3), 
                                 nn.ReLU(), 
                                 nn.Conv2d(6, 12, 3),
                                 nn.ReLU(),
                                 nn.MaxPool2d(2,2)
                                )
    
    def forward(self, x):
        output = self.net(x)
        return output

# We can also use an ordered_dict for this as well
# this is how we do it:
from collections import OrderedDict
class SequentialNet2(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(OrderedDict([("conv1", nn.Conv2d(3, 6, 3)),
                                              ("relu1", nn.ReLU()),
                                              ("conv2", nn.Conv2d(6, 12, 3)),
                                              ("relu2", nn.ReLU()),
                                              ("maxpool1", nn.MaxPool2d(2, 2))
                                             ]))

    def forward(self, inputs):
        return self.net(inputs)

# we can also use a 'list' of layers with our nn.Sequetntial class. 
# here we use a list of layers with nn.Sequential
class SequentialNet3(nn.Module):
    def __init__(self):
        super().__init__()
        # nn.Sequential() can take a list of layers as well
        # so first lets create a list
        layers = []
        # add each layer you like to the list
        layers.append(nn.Conv2d(3, 6, 3))
        layers.append(nn.ReLU()) 
        layers.append(nn.Conv2d(6, 12, 3))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(2,2))   
        # and now we use nn.Sequential()
        # since Sequential does not accept a list,
        # we decompose it by using the * operator.
        self.net = nn.Sequential(*layers)

    # and now in forward, we just need to write one line of code!
    def forward(self, x):
        return self.net(x)

# You may see this a lot in many architectures in Pytorch. however, 
# you are better not use the Python list. and instead use Pytorchs ModuleList:
# becasue : 
# 1. the parameters (weights, biases) of modules inside of a Python list will be missing 
# and if you use it in the training they will not be updated, unless, you manually pass those parameters
# to the optimizer as well.  
# This means, when you do 'model.parameters()', the parameters of layers inside of a python list wont be
# returned becasue Pytorch doesnt look for modules (i.e. layers) in a python list.
# However, when you do use a ModuleList, there is no problem and everything is fine.  
# 2. Even if we pass those modules(that are in a python list) manually, when saving models using
# 'model.state_dict()', the parameters of modules inside of a Python list will not be saved. 
# So always stick to ModuleList 
# using the ModuleList is no different than the normal List, simply swap these together and thats it!: 
# 

# torch.nn.ModuleList : 
# torch.nn.ModuleList can be indexed like a regular Python list, but modules it contains are properly registered,
# and will be visible by all ~torch.nn.Module methods.
# here is an example showing how to use nn.ModuleList
class SequentialNet4(nn.Module):
    def __init__(self):
        super().__init__()
        # nn.Sequential() can take a list of layers as well
        # so first lets create a list
        layers = nn.ModuleList()
        # add each layer you like to the list
        layers.append(nn.Conv2d(3, 6, 3))
        layers.append(nn.ReLU()) 
        layers.append(nn.Conv2d(6, 12, 3))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(2,2))   
        # and now we use nn.Sequential()
        # since Sequential does not accept a list,
        # we decompose it by using the * operator.
        self.net = nn.Sequential(*layers)

    # and now in forward, we just need to write one line of code!
    def forward(self, x):
        return self.net(x)

# Since Sequnetial is a module container and respects the order of insertion,
# (which means internally maintains an ordered_dict), we can add different modules 
# using the add_module() method. this way we can provide a unique name
# for each module and later on access them using this very name!
# also when priniting the model, the names that we gave our modules, 
# makes the architecture more readable! 
# This is how we do it  
class SequentialNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential()
        self.net.add_module('conv1', nn.Conv2d(3, 6, 3))
        self.net.add_module('relu1', nn.ReLU())
        self.net.add_module('conv2', nn.Conv2d(6, 12, 3)) 
        self.net.add_module('relu2', nn.ReLU())
        self.net.add_module('maxpool2', nn.MaxPool2d(2,2))

    def forward(self, inputs):
        return self.net(inputs)

# Similar to what we saw with ModuleList, we can have ModuleDict
# which is an 'ordered' dictionary (unlike Python dict which doesnt
# preserve the order of items when inserted. (since python 3.7 
# dictionaries retain the order aswell)). 
# So we can also use a ModuleDict with Sequential() class . 
# # here is an example
# nn.ModuleDict holds submodules in a dictionary.
# torch.nn.ModuleDict can be indexed like a regular Python dictionary,
# but modules it contains are properly registered, and will be visible
# by all torch.nn.Module methods.
# torch.nn.ModuleDict is an **ordered** dictionary that respects the order of insertion,
# and in torch.nn.ModuleDict.update, the order of the merged OrderedDict or 
# another torch.nn.ModuleDict (the argument to torch.nn.ModuleDict.update).
# Note that torch.nn.ModuleDict.update with other unordered mapping types 
# does not preserve the order of the merged mapping.

# important note: the forward pass for this wont happen!
# becasue there is no forwardpass implemented for moduledict
# its just a container. Seqeuntial however, does support forward()
#
class SequentialNet6(nn.Module):
    def __init__(self):
        super().__init__()
        # the order of insertion is preserved
        self.net = nn.ModuleDict()
        # we can use the add_module
        self.net.add_module('conv1', nn.Conv2d(3, 6, 3))
        # or simply use the ordinary way!
        self.net['relu1'] = nn.ReLU()
        self.net['conv2'] = nn.Conv2d(6, 12, 3)
        self.net['relu2'] = nn.ReLU()
        self.net['maxpool1'] = nn.MaxPool2d(2, 2)

    def forward(self, inputs):
        # simply doing : 
        # return self.model(inputs)
        # wont work as moduledict dosnt implement forward()
        # so you must manually forward all layers here. i.e do 
        out = inputs
        # named_children() iterates over all higherlevel blocks/modules only
        # we dont use modules as biases and weights are also modules which dont
        # implement forward!!
        for n,m in self.net.named_children():
            out =  m(out)
        return out

# as you can see, our foward function has become a one liner! 
# whats the benifit of doing this ? 
# there are several benifits for using nn.Sequential. 
# 1. we can simply define our whole network in one line, as you can see
# and have everything in a consise manner. 
# 2. we can create different parts of our networks this way
# for example, we can define a feature extractor part and
# a classifier or a regressor, etc for our network. 
# and call them separately and this gives us a lot of flexibility.
# lets see some of these usecases:
# example 1 : 
class SequentialNet7(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 6, 3), 
                                      nn.ReLU(), 
                                      nn.Conv2d(6, 12, 3),
                                      nn.ReLU(),
                                      nn.MaxPool2d(2,2)
                                      )
        self.classifer = nn.Linear(12, 2)

    def forward(self, x):
        output = self.features(x)
        output = self.classifer(output)
        return output
    
# now using this scheme, we can easily, later on, swap any part of the network, like the classifer
# or easily use the feature extractor of our network. for example when finetuning, we can create a new
# classifer and assign it to our model.classifier and retrain our model without 
# even changing one line of our architecture. we will see this in action when we do finetuning.

# another benifit of using Sequential is that we can create different building blocks 
# for our networks. we just saw an example, lets expand on that.  

# here we will create a function for creating a convolution layer with batchnorm and 
# relu. here we create a list of layers and then send this list to nn.Sequential

def convlayer(input_dim, output_dim, kernel_size=3, stride=1, padding=1, batchnorm=False):
    # we could use a plain list here as well, 
    # as we are using it as a temporary container
    # for transfering our layers to a nn.sequential.
    # see the note at the end.
    layers = nn.ModuleList()
    conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, padding)
    layers.append(conv)
    if batchnorm: 
        layers.append(nn.BatchNorm2d(output_dim))
    layers.append(nn.ReLU())

    return nn.Sequential(*layers)

# sidenote:
# note that when we said not to use ordinary python lists to hold modules
# we mean, not to use it as an attribute of the model! 
# becasue that way, when you train the model, or anything that requires the
# access to submodules, a list doesnt have such attributes. remember the 
# weight initialization how it actually works by searching among all the modules
# in the model. thats why they cant be used, becasue they mask the actual modules
# they contain and no pytorch builtin machinery can thus work with them like that.
# you can however, use it just fine if all you want to do is to ultimately send it 
# to a nn.sequential to create the actual layer/module. 
# 

class SequentialNet8(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = convlayer(3, 6, 3)
        self.conv2 = convlayer(6, 12, 3, batchnorm=True)
        self.conv3 = convlayer(12, 12, 3, batchnorm=True)
        self.classifer = nn.Linear(12, 2)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2(x)
        output = self.conv3(x)
        output = self.classifer(output)
        return output

# as you can see, we made our code much more consise !! and much more readable!
# we can make this even better and turn this into a one liner like before!
# lets see 
class SequentialNet9(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
                                    convlayer(3, 6, 3, stride=2), 
                                    convlayer(6, 12, 3, stride=2, batchnorm=True),
                                    convlayer(12, 12, 3, stride=2, batchnorm=True)
                                    )
        self.classifer = nn.Linear(12, 2)

    def forward(self, x):
        output = self.features(x)
        # since we have a fc layer, we reshape our output
        # so the fc doesnt complain about the dimensions!
        output = output.view(x.size(0), -1)
        output = self.classifer(output)
        return output

# now lets talk about orderedlist and ordered dictionary! 
# before we continue lets create a model and print them 
# to see how the look 
simple_net1 = SimpleNet()
simple_net2 = SimpleNet2()
sequential_net1 = SequentialNet1()
sequential_net2 = SequentialNet2()
sequential_net3 = SequentialNet3()
sequential_net4 = SequentialNet4()
sequential_net5 = SequentialNet5()
sequential_net6 = SequentialNet6()
sequential_net7 = SequentialNet7()
sequential_net8 = SequentialNet8()
sequential_net9 = SequentialNet9()

print(f'{simple_net1=}')
print(f'{simple_net2=}')
print(f'{sequential_net1=}')
print(f'{sequential_net2=}')
print(f'{sequential_net3=}')
print(f'{sequential_net4=}')
print(f'{sequential_net5=}')
print(f'{sequential_net6=}')
print(f'{sequential_net7=}')
print(f'{sequential_net8=}')
print(f'{sequential_net9=}')

# take a look at how each version looks, 
# the last one can be further imporved
# if we use a separate module instead of
# a function. we will cover this in a moment
#%% 
# to access the modules in our model, we can modules() method.
# modules() method returns all modules from top to the bottom,
# starting from the parent module, down to its sub-modules(defined as its attribute)
# and their submodules if they have any(and this goes on recursively until all are met)
print("\nsimple_net1's modules: ")
for m in simple_net1.modules():
    print(m)
# as you can see, it also printed our model, and then went to print its layers.
# using this we can access each and every layer we like. 
# we saw this in action before, when we wanted to initialize layers weights and biases. 
# we also can use _modules which is an ordered dict! and access any layer by their name!
print(f'\n{simple_net1._modules=}') 
# get the conv1 number of input channels: 
print(f"conv1 input channels : {simple_net1._modules['conv1'].in_channels}") 

# but if we used a more complex architecture with more submodules, 
# it would get a bit messy!
# consider this:
print(f"\nsequential_net9's modules:")
for m in sequential_net9.modules():
    print(m)
# it shows all parent modules, and then their submodules.
# however, sometimes we want to access a layer/block, the higher representation
# as apposed to accessing individual building blocks of a layer/block.
# there are other, better suited alternatives when it comes to accessing 
# higher level building blocks in our network.
# these include: children and its named counterpart: named_children()
# while children only returns the high level modules, the named_children
# returns their corrosponding key-name/module as well.
print(f'\nChildren() only returns the high level modules:\n')
for module in simple_net1.children():
    print(f'{module.__class__}')

print(f'\nnamed_children() returns the name as well as the modules:')
for name, module in simple_net1.named_children():
    print(f'layer/block name: {name} {module.__class__}')

# we can use them repeatedly to access deeper blocks as well
print('\naccessing 2-level children of sequential_net9:')
for name, module in sequential_net9.named_children():
    print(f'--{name=}')
    for k,m in module.named_children():
        print(f'  {k}')
        
# we can recursively access them all!
def print_children(module, level=0):
    for name,child in module.named_children():
        print(f"{level*' '}-{name} {child.__class__.__name__}")
        print_children(child,level+1)
        
print(f'\nshowing all children in sequential_net9:')
print_children(sequential_net9)
print(f'\nshowing all children in sequential_net_2:')
print_children(sequential_net2)

# we'll see more usecases in future chapters.

# sidenote
# earlier we saw to access the parameters (i.e. weights and biases) of our modules 
# we can use .parameters() (recall when we were creating our optimizers and pass the
# model.parameters() to it?)
# like the children, it also has a named equivalent which is named_parameters()
# by default, parameters/named_parameters() will recursively go through all sub modules
# and their paramters!
# lets see the gradients for our model:
for name,param in sequential_net2.named_parameters(recurse=False):
    print(f'{name=}, {param.dtype=}')
# didnt print anything, becasue we dont have any immediate attribute which is of type nn.Parameter
# we have layers of type nn.Module!

print(f'with curse=True')
# now lets use with recurs
for name,param in sequential_net2.named_parameters(recurse=True):
    print(f'{name=}, dtype={param.dtype} type={param.__class__.__name__}')
# now when we enabled the recurse, we can now go to each layer, and see if they
# have any parameters of their own, which happens to be the case here, each of those
# layers have a weight and a bias parameter.

# we also have named_buffers() which are the fields that we define in our models
# which are not optimized/trained during the training process, but are crucial for
# the model to work and when we save the model, they will be present in the 
# resulting state_dict. an example of this would be the running_average in the
# BatchNormalization layer, which are not optimized during training, but are required
# for models correct behavior after training, so whenever we save the model's weights
# these are saved as buffers as well, and upon restoring they will be restored automatically 
# as well (they are named this way to create a distinction with weights/biases 
# which are optimized during training)
# we will talk about this in more depth in later chapters where we implement BatchNorm ourselevs

#%%
# now that we learnt how to create a neural network, lets see how we can
# create a new layer or module in Pytorch! 
# As always there are many ways to do this but here I'll try to demonstrate
# something simple and easy to follow. 
# lets create a flatten layer for Pytorch that we can use in our nn.Sequential
# so that we can flatten a convolution layers output so we can use it in a
# fully connected layer (nn.Linear). 
# As you have seen, before we can add a fc layer after a conv layer
# the output needs to be flattened.
# 
# sidenote:
# there was no layer for flattening in Pytorch as I was writting this for Pytorch 0.4 and later 1.0.
# Later versions however, starting from Pytoch 1.6, Flatten() layer was added, which can be used as well!)
# 
# up until now, we have been using something like : 
# output = output.view(x.size(0), -1)
# in our forward() method to flatten an output .and we couldnt do this in a nn.sequential!
# lets create a new module(layer in Pytorch)!
#  
# For creating a new module/layer, we inherit from nn.Module. 
# the rest is self explanatory. lets see how it is done 
class Flatten(nn.Module): 
    # our flatten layer doesnt need any argument, as it only 
    # flattens the input!
    def __init__(self):
        super().__init__()
        
    def forward(self, x): 
        # uncomment to see the changes if you like
        # print(f'before flattening: {x.shape}')
        x = x.view(x.size(0), -1)
        # print(f'after flattening: {x.shape}')
        return x
    
# thats all ! 
# our Flatten layer doesnt need any arguements, nor does it need any layer
# all that this layer does is to reshape the input featuremap, into flat
# representation. lets use this in action ! 
# lets create a simple network for classification that accepts input of size 32x32x3 
# and has 3 classes! 
class MyNet(nn.Module): 
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(nn.Conv2d(3, 6, 3, 1, 1), 
                                   nn.ReLU(), 
                                   nn.MaxPool2d(2, 2), # 16x16
                                   nn.Conv2d(6, 12, 3, 1, 1),
                                   nn.ReLU(),
                                   nn.MaxPool2d(2, 2), # 8x8 
                                   nn.Conv2d(12, 24, 3, 1, 1),
                                   nn.ReLU(),
                                   nn.MaxPool2d(2, 2), # 4x4
                                   nn.Conv2d(24, 32, 3, 1, 1),
                                   nn.ReLU(), 
                                   nn.MaxPool2d(2, 2), # 2x2
                                   Flatten(), 
                                   nn.Linear(128, 3)   # 32 * 2 * 2 = 128 
                                   )
    def forward(self, x):
        return self.model(x)

# now lets test the output 
fake_input = torch.rand(2, 3, 32, 32)
mynet = MyNet()
logits = mynet(fake_input)
print(logits)

# sidenote 2: 
# torch.nn.Flatten() is a bit more powerful than our version here, as it allows us
# to flatten specific dimes which give us more flexibility.
# it flattens a contiguous range of dims into a tensor. it takes two parameters:
# start_dim and end_dim. 
# the first one specifies the first dim to flatten (default is 1 (0 is batchsize))
# the second one specifies the last dim to flatten (default = -1), all the dims in between
# will be flattened. 
# basically it flattens all dimensions starting from the second dimension (index 1) by default.
# 


# before that let us try to create a ResNet architecture. 
# Resnet as you know is one of the mostly used architectures out there, and the 
# main building of this network is something called a resblock which is just a block 
# with residual connection(there are many variants out there now, this is the simplest 
# form of a residual block)!
# let us create this network. 
# first we create the resblock. 
# a resblock is simply one( or more conv layers) that gets applied on an input
# and at the end, the input is summed with the output of the conv layer(s)
# basically, we are trying to do (h(x) = x + f(x) where x is the input to resblock
# f(x) is the function learned using the conv layers inside resblock and h(x) is the
# function learnt at the end. 
# 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
class ResBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size, stride=stride, padding=padding),
                                   nn.BatchNorm2d(out_dim))                               

    def forward(self, x):
        output = self.conv1(x)
        output += x
        return F.relu(output)

# sidenote: 
# usually when we say resblock, it contains more than a single layer of convnet
# this is the bare minimum obviously!

# now lets use this layer/module/block in a new network! 
class Resnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(nn.Conv2d(3,  6, 3, 1, 1),
                                    ResBlock(6, 12, 3, 1, 1),
                                    nn.MaxPool2d(2, 2), #16x16 
                                    ResBlock(12, 24, 3, 1, 1),
                                    nn.MaxPool2d(2, 2), #8x8
                                    ResBlock(24, 32, 3, 1, 1),
                                    nn.MaxPool2d(2, 2), #4x4 
                                    Flatten(),
                                    nn.Linear(32 * 4*4 , 3))
    def forward(self, input):
        return self.model(input)

# thats it!
resnet = Resnet()
print(resnet)
# thats it! thats the basic resnet. there are other variants as well, which incorporate bottleneck!
# as well. you can easily add that too !  
# Now you should have a good idea how powerful and flexible Pytorch is and how you can 
# just create anything you have in your mind! 
        
#%%
# Lets see how we can use finetuning in Pytorch! 
# but before that let me tell you how you can save/load your models in Pytorch ! 
# in order to save your model, all you need to do is to use torch.save() 
# and pass your model.state_dict() which is a dictionary containing your model parameters: 
torch.save(model.state_dict(),'./weights/mymodel_weights.pth')
# sidenote: 
# we can use any extensions we like (like.t, .pt, .pth, etc),
# but a general trend/convention is to use 'pth' for model weights to refer its from pytorch!
 
# we can get fancier and take advantage of the fact that torch.save() accepts a dictionary 
# and thus add more items to be saved! 
# For example, we can save the parameters we used to instantiate our model, optimizer, schedulers,
# save the loss, the best acc, epochs passed, etc. basically everything that matters to us. 
# Just create a dictionary and add what you want and use this dictionary instead to save your 
# model. 
# this is what we call a checkpoint! so we can use to resume training at a later time.
# example input!
e = 10 
acc_val = 0.82
acc_train = 0.95
loss_train  = 0.1025
model = Resnet()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)

settings = {'state_dict': model.state_dict(),
            'optimizer':optimizer.state_dict(),
            'epoch': e, 
            'acc_val': acc_val,
            'acc_train': acc_train, 
            'loss_train': loss_train
            } 
# like before, we can use any extensions we like, but the general trend has been to
# use a .ckpt to show this includes more than the raw model weights, it includes other
# informations needed to resume training like optimizer and scheduler settings as well as
# any other setting or piece of config thats needed. 
torch.save(settings, './weights/ourmodel.ckpt')
#  thats it 
print('save done!')
#%%
# now to load the settings 
# we use  torch.load but thats not all! lets see how it works!
# we first load the whole dictionary from our model!
# note, its always a good idea to try to load the checkpoint in cpu mode first!
# TODO: explain more
model_settings_dict = torch.load('./weights/ourmodel.ckpt', map_location='cpu')
# now that we have our dictionary of settings, lets load the values 
# before we continue lets create the variables with garbage values so
# we know for sure the loading is done successfully 
epoch = None
acc_val = None
acc_train = None
loss_train = None
model = Resnet()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.555555)
# now lets start loading! 
epoch = model_settings_dict['epoch']
acc_val = model_settings_dict['acc_val']
acc_train = model_settings_dict['acc_train']
loss_train = model_settings_dict['loss_train']
# now for the optimizer and our model, we cant simply do this, instead
# we use a specific method in our model/optimizer called load_state_dict
# to load the contents properly. 
model.load_state_dict(model_settings_dict['state_dict'])
optimizer.load_state_dict(model_settings_dict['optimizer'])
print('load done! ')
print('some loaded variables: ')
print(f'{optimizer=}')
print(f'{epoch=}')
print(f'{acc_val=}')
print(f'{acc_train=}')

#%%
# OK we are ready to see how fine-tuning works in Pytorch 
# there are several models in models repository that we can use
# lets choose one but before that lets see what we have at our disposal 
from torchvision import models
import torch.nn as nn 

# previously to see what models we had access to through torchvisions.models we
# had to do dir(models), but since torchvision 0.14, we have models.list_models()
# to give us the list of the available models through torchvision.models:
print(f'{dir(models)=}')
print(f'{models.list_models()=}')

# lets use resnet18! by setting pretrained = True, we'll also be down-
#loading the imagenet weights. we dont need it for now!
resnet18 = models.resnet18(pretrained=True)
# lets print the model 
print(f'\nORIGINAL MODEL : \n{resnet18}\n')

# by looking at the architecure, we notice :
# (fc): Linear(in_features=512, out_features=1000, bias=True) 
# In order to retrain this network for our usecase
# we need to alter this layer. this was trained on imagenet
# which had 1000 classes. lets train this for cifar10 which
# has 10 classes. all we need to do is just defining a new
# fully connected (fc) layer and assigning it back to  
# resnet18.fc attribute!
# resnet18.fc = nn.Linear(512, 10)
# but wait we can do better!, 
# now instead of hardcoding the 512 which we saw by looking at the 
# printed version of our model, we can simply use the 
# 'in_features' attribute of the fc layer! and write : 
resnet18.fc = nn.Linear(resnet18.fc.in_features, 10)

print(f'\nNEW MODEL(after adding the new fc layer): \n{resnet18}')
# now before we dive in, to train our network we should first 
# freeze all layers except this new one, and train for several epochs, 
# so that it converges to a reasonable set of weights
# then we unfreeze all previous layers and train the whole net
# altogether again. 
# 
# So lets freeze all layers before this fc layer!
# we could do: 
# for module in resnet18.modules():
#     # search for every layer other than nn.Linear layer (our new classifer layer)
#     # and set its require_grad to False to effectively prevent it from updating its wieghts
#     if module._get_name() != nn.Linear.__name__:
#         print('layer: ',module._get_name())
#         for param in module.parameters():
#              param.requires_grad_(False)
    # this else clause is necessary becasue weight and bias for fc layer
    # are also modules and thus their name is not 'Linear' obviously!
    # So this last part is needed        
    # elif module._get_name() == nn.Linear.__name__:
    #     for param in module.parameters():
    #         param.requires_grad_(True)

# we could do better and instead use the isinstance!: 
# for module in resnet18.modules():
#     if not isinstance(module, nn.Linear):
#         for param in module.parameters():
#             param.require_grad = False
#     if isinstance(module, nn.Linear):
#         for param in module.paramertes():
#             param.require_grad = True
#
# again we could do better, by using parameters()!: 
# and split the process into two 
# for param in resnet18.parameters():
    # param.requires_grad = False 
# and for fc 
# for param in resnet18.fc.parameters():
    # param.requires_grad = True
# in case we want to have the parameters name, we can use named_parameters!
# print('Freezing All layers')
# for name, param in resnet18.named_parameters():
#     param.requires_grad = False
#     print(f'name: {name} requires_grad : {param.requires_grad}')        
# 
# print('\nUnfreezing the new FC layer')
# for name, param in resnet18.fc.named_parameters():
#     param.requires_grad = True      
#     print(f'name: {name} requires_grad : {param.requires_grad}')

# but we could do even better!, we can grab all layers except the last one
# and set all of their paramters gradient to False
for layer in list(resnet18.children())[:-1]:
    for p in layer.parameters():
        p.requires_grad_(False)

# check all layers status again!
for name, param in resnet18.named_parameters():
    print(f'{name} : {param.requires_grad}')            

#%%
# first version ! 
# lets create a training and testing functions for our case 
def validation(model, dataloader_test, criterion, k, device):
    loss_total = 0.0
    batch_accs = 0.0
    with torch.no_grad(): 
        # actiate evaluation mode 
        model.eval()
        for imgs, labels in dataloader_test:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs)
            loss_val = criterion(preds, labels)
            _, indexes = preds.topk(k, dim=1)
            batch_accs += (indexes == labels.view(-1,1)).any(dim=1).float().mean()*100
            loss_total += loss_val.item()
        
        accuracy_final = batch_accs/len(dataloader_test)
        loss_final = loss_total/len(dataloader_test)
    return loss_final, accuracy_final

def training(model, dataloader_train, dataloader_test, epochs, criterion, optimizer, k=1, interval=500, device='cuda'):
    model = model.to(device)
    training_acc_losses = []
    val_acc_losses = []
    trainig_batch_count = len(dataloader_train)
    test_batch_count = len(dataloader_test)

    for e in range(epochs):
        batch_accs = 0.0
        training_loss = 0.0
        
        # activate trainig mode
        model.train()
        for i, (imgs, labels) in enumerate(dataloader_train,start=1):
            
            imgs = imgs.to(device)
            labels = labels.to(device)
            
            preds = model(imgs)
            loss = criterion(preds, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            training_loss += loss.item()
            # calculate training accuracy 
            _, indexes= preds.topk(k=k, dim=1)
            batch_accs += (indexes == labels.view(-1,1)).any(dim=1).float().mean()*100

            if i % interval == 0 :
                print(f'(Epoch/Iter): {e}-{epochs}/{i}-{trainig_batch_count} | Train-loss: {loss.item():.6f} | Train-acc: {batch_accs/i:.4f} ')
        
        # accumulate accuracies and losses per epoch
        training_acc_losses.append( (batch_accs/ trainig_batch_count, training_loss/ trainig_batch_count))
        
        # run validation test at every epoch!
        val_acc_loss = validation(model, dataloader_test, criterion, k=k, device=device)
        val_acc_losses.append(val_acc_loss)
        print(f'Val_loss : {val_acc_loss[0]:.4f} | Val_acc: {val_acc_loss[1]:.4f}')

    return training_acc_losses, val_acc_losses

#%%
import copy
# lets create a copy of our model for future use
model = copy.deepcopy(resnet18)

# now lets try to train our resnet18
epochs = 10
k = 1
batch_size = 32
interval = 1000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.001)

transformations_train = transforms.Compose([transforms.Resize(224),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                                 std=(0.229, 0.224, 0.225))
                                            ])

transformations_test = transforms.Compose([transforms.Resize(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                                std=(0.229, 0.224, 0.225))
                                           ])

dataset_train = datasets.CIFAR10('./data/CIFAR10', train=True, transform = transformations_train, download=True)
dataset_test = datasets.CIFAR10('./data/CIFAR10', train=False, transform = transformations_test, download=True)

dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size = batch_size, shuffle=True, num_workers = 2)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size = batch_size, shuffle=False, num_workers = 2)

#%%
train_info, val_info = training(model, dataloader_train, dataloader_test, epochs, criterion, optimizer, k, interval, device)
# and we got 78% accuracy after 10 epochs! not bad, but not good either!
# (Epoch/Iter): 0-10/1000-1563 | Train-loss: 1.177670 | Train-acc: 49.9688 
# Val_loss : 1.0779 | Val_acc: 70.5871
# (Epoch/Iter): 1-10/1000-1563 | Train-loss: 1.115183 | Train-acc: 71.1281 
# Val_loss : 0.8626 | Val_acc: 74.2212
# (Epoch/Iter): 2-10/1000-1563 | Train-loss: 0.823192 | Train-acc: 73.8125 
# Val_loss : 0.7865 | Val_acc: 75.3095
# (Epoch/Iter): 3-10/1000-1563 | Train-loss: 0.783242 | Train-acc: 74.7625 
# Val_loss : 0.7307 | Val_acc: 76.8171
# (Epoch/Iter): 4-10/1000-1563 | Train-loss: 0.649752 | Train-acc: 75.8813 
# Val_loss : 0.7051 | Val_acc: 77.2065
# (Epoch/Iter): 5-10/1000-1563 | Train-loss: 0.787717 | Train-acc: 76.2594 
# Val_loss : 0.6867 | Val_acc: 77.5958
# (Epoch/Iter): 6-10/1000-1563 | Train-loss: 0.641532 | Train-acc: 76.4813 
# Val_loss : 0.6651 | Val_acc: 78.2149
# (Epoch/Iter): 7-10/1000-1563 | Train-loss: 0.570518 | Train-acc: 77.1656 
# Val_loss : 0.6579 | Val_acc: 78.1949
# (Epoch/Iter): 8-10/1000-1563 | Train-loss: 0.653849 | Train-acc: 77.3375 
# Val_loss : 0.6410 | Val_acc: 78.7740
# (Epoch/Iter): 9-10/1000-1563 | Train-loss: 0.484139 | Train-acc: 77.4281 
# Val_loss : 0.6388 | Val_acc: 78.8438
#%%
# now unfreeze all layers and retrain the whole network
for param in model.parameters():
    param.requires_grad = True

optimizer = torch.optim.SGD(model.parameters(), lr = 0.001)
train_info, val_info = training(model, dataloader_train, dataloader_test, epochs, criterion, optimizer, k, interval, device)

# we can decrease the learning rate at some intervals and have better convergance
# optimizer = torch.optim.SGD(model.parameters(), lr = 0.0001)
# train_info, val_info = training(model, dataloader_train, dataloader_test, epochs, criterion, optimizer, k, interval, device)

# optimizer = torch.optim.SGD(model.parameters(), lr = 0.00001)
# train_info, val_info = training(model, dataloader_train, dataloader_test, epochs, criterion, optimizer, k, interval, device)

# and this time around we got:
# we got a high accuracy!, 94.50% accuracy which is much better, but still we can do better
# you also must have noted that the training takes considerably longer compared 
# to the previous attempt. its expected as we are now training the whole network
# and not just the last layer! so it takes more vram and more computation as it 
# needs to compute gradients for more parameters, and update them accordingly!
# (Epoch/Iter): 0-10/1000-1563 | Train-loss: 0.186048 | Train-acc: 84.3156 
# Val_loss : 0.3088 | Val_acc: 89.6466
# (Epoch/Iter): 1-10/1000-1563 | Train-loss: 0.214217 | Train-acc: 90.0969 
# Val_loss : 0.2416 | Val_acc: 91.9030
# (Epoch/Iter): 2-10/1000-1563 | Train-loss: 0.311374 | Train-acc: 92.1875 
# Val_loss : 0.2158 | Val_acc: 92.7716
# (Epoch/Iter): 3-10/1000-1563 | Train-loss: 0.339803 | Train-acc: 92.9906 
# Val_loss : 0.2004 | Val_acc: 93.1310
# (Epoch/Iter): 4-10/1000-1563 | Train-loss: 0.171796 | Train-acc: 94.0125 
# Val_loss : 0.1890 | Val_acc: 93.6601
# (Epoch/Iter): 5-10/1000-1563 | Train-loss: 0.071380 | Train-acc: 94.8000 
# Val_loss : 0.1787 | Val_acc: 93.9796
# (Epoch/Iter): 6-10/1000-1563 | Train-loss: 0.245795 | Train-acc: 94.9344 
# Val_loss : 0.1782 | Val_acc: 94.1494
# (Epoch/Iter): 7-10/1000-1563 | Train-loss: 0.061658 | Train-acc: 95.4500 
# Val_loss : 0.1699 | Val_acc: 94.1593
# (Epoch/Iter): 8-10/1000-1563 | Train-loss: 0.155709 | Train-acc: 95.7156 
# Val_loss : 0.1648 | Val_acc: 94.2991
# (Epoch/Iter): 9-10/1000-1563 | Train-loss: 0.139559 | Train-acc: 96.2594 
# Val_loss : 0.1606 | Val_acc: 94.4988

#%%
# At this rate you must be thinking this is rediculous! cant we just create a loop where 
# we iteratively/periodiclly decrease the learning rate? 
# and the answer is yes it is! a simple form, for this would be :
lr=0.001
for i in range(3):
    lr =  lr * 0.1
    # change the learning rate 
    # method 1 :create a new instance of sgd optimizer with a new lr
    optimizer = torch.optim.SGD(model.parameters(), lr = lr)
    # method 2 change an existing sgd optimizer's lr
    # before changing the lr
    # print([p['lr'] for p in optimizer.param_groups])
    # for p in optimizer.param_groups:
    #     p['lr'] = p['lr'] * 0.1
    # # after changing the learning rate    
    # print([p['lr'] for p in optimizer.param_groups])
    train_info, val_info = training(model, dataloader_train, dataloader_test, epochs, criterion, optimizer, k, interval, device)
#%%
# but the best method would be to use MultiStepLR from 
# torch.optim.lr_schedule module.
# There are other schedulers as well which you can use, much better ones Id say
# but for now, multistepLR does the trick for us.
# Lets reimplement our training loop, this time in a better way! 
# if you look back, you can see there is a lot of duplication that
# we can get rid of! 
#### IMPORTANT NOTE  ####
# Prior to PyTorch 1.1.0, the learning rate scheduler was expected 
# to be called before the optimizer’s update; 1.1.0 changed this 
# behavior in a BC-breaking way. 
# If you use the learning rate scheduler (calling scheduler.step())
# before the optimizer’s update (calling optimizer.step()), 
# this will skip the first value of the learning rate schedule.
# If you are unable to reproduce results after upgrading to PyTorch 1.1.0,
# please check if you are calling scheduler.step() at the wrong time.


def train_validation_loop(model, dataloader, optimizer, criterion, is_training,
                          device, topk=1, interval=1000 ):
    
    batch_losses = 0.0
    batch_accuracies = 0.0
    total_batches = len(dataloader)
    status = 'Training' if is_training else 'Validation' 
    
    for i, (imgs, labels) in enumerate(dataloader,start=1):
        imgs = imgs.to(device)
        labels = labels.to(device)
        
        if is_training:
            model.train()
            preds = model(imgs)
            loss = criterion(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            model.eval()
            with torch.no_grad():
                preds = model(imgs)
                loss = criterion(preds, labels)

        batch_losses += loss.item()
        _, class_idxs = preds.topk(topk, dim=1)
        batch_accuracies += (class_idxs == labels.view(-1,1)).any(dim=1).float().mean()*100
        
        if i % interval == 0: 
            print(f'{status} Loss/Accuracy(so far): {batch_losses/i:.6f}/{batch_accuracies/i:.2f}%')

    # calculate the loss/accuracy after each epoch        
    loss_final = batch_losses/total_batches 
    accuracy_final = batch_accuracies/total_batches 
    return loss_final, accuracy_final

# now our main loop 
def training_loop(model, dataloader_train, dataloader_test, optimizer,
                  criterion, scheduler, device, epochs=10, topk=1, interval=1000):
    for e in range(epochs):
        train_loss, train_acc = train_validation_loop(model, dataloader_train, optimizer, criterion, is_training=True, device=device, topk=topk)
        test_loss, test_acc   = train_validation_loop(model, dataloader_test, optimizer, criterion, is_training=False, device=device, topk=topk)
        scheduler.step()
        print(f'(Epoch) {e}: lr: {scheduler.get_last_lr()[0]:.8f} \n\tTraining loss/accuracy: {train_loss:.6f}/{train_acc:.2f}\n'
        f'\tTesting loss/accuracy: {test_loss:.6f}/{test_acc:.2f}%')
        # we can specify at which epoch or learnng rate
        # reactivate/unfreeze all previous layers 
        # using sth like 
        # if e == 3:
        #     for param in model.paramertes():
        #         param.requires_grad = True
        
#%% 
from torch import optim
# use the clean model so we can see properly see it gives us the same result
# if not better, but this time in a much more consise and cleaner manner!
model = copy.deepcopy(resnet18)
optimizer = optim.SGD(model.parameters(), lr = 0.01)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[8, 19], gamma = 0.1)
criterion = nn.CrossEntropyLoss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print (f'initial lr: {scheduler.get_lr()}')

training_loop(model, dataloader_train, dataloader_test,
              optimizer, criterion, scheduler, device, epochs)

#unfreeze all layers and retrain 
print('unfreezing all layers now!')
[p.requires_grad_(True) for p in model.parameters()]

print (f'initial lr: {scheduler.get_lr()}')
training_loop(model, dataloader_train, dataloader_test,
              optimizer, criterion, scheduler, device, epochs)

# good! we could easily achieve 94.57% accuracy in several epochs with 
# some very primitive techniques but more consise!
# we can get much higher accuracy if we invest in 
# hyper parameter tuning, etc.  
# for now this is enough as we just wanted to 
# demonstrate how things can be done in Pytorch!
# There is still a lot to imporve which we will cover inshaallah in future chapters
# note: 
# the better way is to encapsulate the lr decay regime inside the training loop
# so that given the right number of epochs, we decay the lr, unfreeze the layers
# etc at the right time.
# also note that we set the scheduler to decay twice, one at epoch 8 and the other
# at 19. note that since we run the training twice, each for 10 epochs, but
# the scheduler and model are the same, we get the same effect. 
# sidenote 2: 
# since pytorch 1.13 we have had torch.compile() which compiles our code and allows us
# to achieve a good speedup in training e.g. it should work on most models as of (2.3.0)
# we will cover this in more details in future chapters as well inshaallah
#%%
# Lets see couple of other architectures and how they can be finetuned! 
# https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html is 
# a very good resource. have a look at it now!
# AlexNet 
alexnet = models.alexnet()
# lets print it firts 
print(f'alexnet model: {alexnet}')
# as we can see in the output, we have a features and a classifier
# attribute which signify their roles. 
# we know for a finetuning we should 90% of the times only replace the classifier
# we can do this here as well and swap classifier which our new one. however
# we can also swap the last layer which is responsible for the classification
# lets see how we can do that? 
alexnet.classifier[6] = nn.Linear(alexnet.classifier[6].in_features, 10)
# as you can see, we classifier is an ordered dictionary and we can easily 
# access any item using its key!  
print(f'alexnet model after change: {alexnet}')
#%% 
# VGG models have been retrained using Batchnorm and they achieve a much higher accuracy
# here we can use the new version!
vgg16 = models.vgg16_bn()
print(f'vgg16 model: {vgg16}')
# this is like alexnet, it has features, and classifier attributes we can use
# for feature extraction or finetuning, etc
vgg16.classifier[6] = nn.Linear(vgg16.classifier[6].in_features, 10)
print(f'vggnet16 model after change: {vgg16}')

#%% 
# Next is SqueezeNet, 
# squeezenet is a very small architecture that provides alexnet level
# accuracy with 50x less parameters there are  two versions 1_0 and 1_1
# that provides a bit more accuracy! than the former one!
squeezenet = models.squeezenet1_1()
print(f'squeezenet model: {squeezenet}')
# similar to all previous architectures so far, it has a features and a
# classifier attribute. it uses global average pooling to get the final
# scores and is fully convolutonal!
# we can swap the whole classifier, but we can also take advantage of 
# the fact that its a fully convolutional neural network and we can accept
# images of any size without any issues unlike the former architectures that
# were trained on a fixed size images and could only use fixed-size
# images at test time.
# here, we change the last convoloution layer output channels and make it equal 
# to the number of our new classes.
in_channels = squeezenet.classifier[1].in_channels
kernel_size = squeezenet.classifier[1].kernel_size
stride = squeezenet.classifier[1].stride
squeezenet.classifier[1] = nn.Conv2d(in_channels = in_channels, out_channels = 10, kernel_size = kernel_size, stride = stride) 
print(f'squeezenet model after change: {squeezenet}')
#%% 
# Now lets see how we can work with inception architecture. its a bit different than the others so lets 
# see how to treat it. 
inceptionv3 = models.inception_v3()
print(f'inceptionv3 model: {inceptionv3}')
# ... Inception v3 was first described in Rethinking the Inception Architecture for Computer Vision.
# This network is unique because it has two output layers when training.
# The second output is known as an auxiliary output and is contained in the AuxLogits part of the network.
# The primary output is a linear layer at the end of the network.
# Note, when testing we only consider the primary output.
# The auxiliary output and primary output of the loaded model are printed as:

#  (AuxLogits): InceptionAux(
#     (conv0): BasicConv2d(
#       (conv): Conv2d(768, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#       (bn): BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#     )
#     (conv1): BasicConv2d(
#       (conv): Conv2d(128, 768, kernel_size=(5, 5), stride=(1, 1), bias=False)
#       (bn): BatchNorm2d(768, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#     )
#     (fc): Linear(in_features=768, out_features=1000, bias=True)
#   )
# and 
#  (fc): Linear(in_features=2048, out_features=1000, bias=True)
# 

# in order to finetune this we need to change both of these layers. 
inceptionv3.AuxLogits.fc = nn.Linear(inceptionv3.AuxLogits.fc.in_features, 10)
inceptionv3.fc = nn.Linear(inceptionv3.fc.in_features, 10)
print(f'inceptionv3 model after change: {inceptionv3}')
#%%
# These were ancient architectures, There are newer ones 
# lets see some of the newer ones 
# mobilenet cameout in 2017 and achieved vgg16 accuracy(top1 69-70% acc) with 6/7 million parameters
# it was a huge success, it used depth separable convolutions to lower the computation
# and thus it achieved great result. however, becasue of this, it was extremely slow
# in inference, until the depth-wise-separable convolution operation has been optimized
# so prior to 2020, mobilenet was extremely slow despite having a small number of paramters
# the way the architecture was designed also lead to these issues. nonetheless v2
# version cameout with new design choices and again smaller flops and parameter counts(3m)
# in 2018 it achived 71.8% top1, and later in 2022/2023 with enhanced training it achived 72.15% top1
# v3 came out in 2019 it uses the Inverted Residual Block (also called Inverted Bottleneck)
# that was introduced in the earlier MobileNet-V2 model. 
# The IR consists of a 1x1 pointwise (PW) expansion convolution, followed by a depthwise (DW)
# convolution (3x3 or 5x5), and finally a 1x1 pointwise linear (PWL, no activation) convolution 
# in the residual path. Unlike most other residual blocks at the time, the wide part is in the 
# middle of the block at the depthwise conv, instead of at the start/end, and the output of the 
# block has no activation (linear output), hence 'Inverted'.  
# the v4 is also out in 2024 which achieves 73% top1 along with other versions up to 37m
# achieving 84% top1! this is not availble through torchvision models, but we can use it 
# through timms! which we will see later on. (https://huggingface.co/blog/rwightman/mobilenetv4)
# mobilenetv3-small can be a good choice so we usethat
mobilenet = models.mobilenet_v3_small()
print(f'{mobilenet=}')
# changing the classifier should now be easy for you
mobilenet.classifier[-1] = nn.Linear(mobilenet.classifier[-1].in_features, out_features=10,bias=True)
print(f'{mobilenet=}')
# the v2 is the same and this makes life much easier! :)
#
# next is convnext which comes with several variants, tiny, small, base and large
# ConvNeXt came out in 2020 in a paper named 'A ConvNet for the 2020s'. 
# it revitalized the cnns after the transformers ground breaking success previously,
# the paper's introduction puts it very well stating a tiny summary of what it does: 
# 
# The "Roaring 20s" of visual recognition began with the introduction of Vision Transformers (ViTs),
# which quickly superseded ConvNets as the state-of-the-art image classification model.
# A vanilla ViT, on the other hand, faces difficulties when applied to general 
# computer vision tasks such as object detection and semantic segmentation. 
# It is the hierarchical Transformers (e.g., Swin Transformers) that reintroduced 
# several ConvNet priors, making Transformers practically viable as a generic vision
# backbone and demonstrating remarkable performance on a wide variety of vision tasks.
# However, the effectiveness of such hybrid approaches is still largely credited to 
# the intrinsic superiority of Transformers, rather than the inherent inductive biases of
# convolutions. 
# In this work, we reexamine the design spaces and test the limits of
# what a pure ConvNet can achieve. We gradually "modernize" a standard ResNet toward 
# the design of a vision Transformer, and discover several key components that contribute
# to the performance difference along the way. The outcome of this exploration is a 
# family of pure ConvNet models dubbed ConvNeXt. 
# Constructed entirely from standard ConvNet modules,ConvNeXts compete favorably with
# Transformers in terms of accuracy and scalability, achieving 87.8% ImageNet top-1
# accuracy and outperforming Swin Transformers on COCO detection and ADE20K segmentation,
# while maintaining the simplicity and efficiency of standard ConvNets.
# 
# they created a v2 version in 2022(https://arxiv.org/pdf/2301.00808), which was self-supervised
# and achived a high accuracy by enhancing the architecture further using a new global response 
# normaliation layer and a fully convolutional masked autoencoder framework: 
# While convext models were originally designed for supervised learning with ImageNet labels,
# they can also potentially benefit from self-supervised learning techniques such as
# masked autoencoders (MAE) [31].
# However, we found that simply combining these two approaches leads to subpar performance.
# In this paper, we propose a fully convolutional masked autoencoder framework and a new 
# Global Response Normalization (GRN) layer that can be added to the ConvNeXt architecture
# to enhance inter-channel feature competition. 
# This co-design of self-supervised learning techniques and architectural improvement results
# in a new model family called ConvNeXt V2, which significantly improves the performance of pure
# ConvNets on various recognition benchmarks, including ImageNet classification, COCO detection,
# and ADE20K segmentation. 
# We also provide pre-trained ConvNeXt V2 models of various sizes, ranging from an efficient 3.7M 
# parameter Atto model with 76.7% top-1 accuracy on ImageNet, to a 650M Huge model that achieves 
# a state-of-the-art 88.9% accuracy using only public training data (i.e. they pretrained it and then finetuned it on imagenet!)
# the v2 is not availble in torchvision, it is however in timms again which we can use
# for now lets see how to use convext!
convext = models.convnext_base()
print(f'before-{convext=}')
# it follows the same trend which is 
convext.classifier[-1] = nn.Linear(convext.classifier[-1].in_features, out_features=10,bias=True)
print(f'after-{convext=}')

# next is SwinTransformers, which is a great architcture that uses transformers like vit
# but adds several induction biases that exist it cnn( i.e. priors) such as sliding window (attention)
# it came out in 2021 in the paper https://arxiv.org/abs/2103.14030. 
# theres a second version which cameout in 2022 : https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_Swin_Transformer_V2_Scaling_Up_Capacity_and_Resolution_CVPR_2022_paper.pdf
# which ramped everything up! 
# there are 3 variants, swin_t,swin_b and swin_s available in pytorch,
# swin_t is a 28.28m parameter transformer network which achieves top1 accuacy of 81.44%
# swin_s is a 49.60m parameter transformer network which achieves top1 accuracy of 83.19%
# swin_b is a 87.76m parameter transformer netwrk which achieves top1 accuracy of 83.58%
# the second version improved the results
# swin_v2_t is a 28.32m parameter transformer network which achieves top1 accuacy of 82.07%
# swin_v2_s is a 49.73m parameter transformer network which achieves top1 accuracy of 83.71%
# swin_v2_b is a 87.93m parameter transformer network which achieves top1 accuracy of 84.11%
# the second versions achieved slightly better results but got heavier as well. they are
# on the heavier side of models. the smallest version is 4~5 GFLOPs. 
# for comparison mobilenetsv3-large's FLOPS is 0.22GFLOPs! and it achieves 75% top1.
swinb = models.swin_b()
print(f'{swinb=}')
# for swin as you can see its much easier! just swap the head!
swinb.head = nn.Linear(swinb.head.in_features,10,bias=True)
#%%
import requests
from PIL import Image
# Using the pre-trained models
# Ok, this is all good and we learned how to make a network ready for a new tasks!
# but theres more to preparing a pre-trained network for a specific task. 
# changing the classifier accordingly is only a basic prerequisit. 
# we also need to use the same transformations that was used initially during model training. 
# things such as resizing with the right resolution/interpolation, 
# applying the proper inference transformations, rescaleing the values etc are a few 
# to point out. these need to be taken into account in order to get the same level of accuracy,
# and performance otherwise failing to do so most often leads to decreased accuracy or incorrect outputs.
# 
# This is where torchvision comes in handy. it offers some features that help us a lot
# to get this done quickly!
# as you may know, different models use different configurations, both in terms of transformations, 
# and hyperparamters among other things. even different variants of the same family of models 
# have different configurations. so in order to run these models, we need to know in advance
# how to prepare their inputs. 
# 
# lucky for us, since v0.14, torchvision offers several ways to get a model prepared and
# ready to use.
# We can access all necessary information needed for inference using each model's weights obj.
# Each weights object (which is an enum by the way) contains several attributes such as
# meta, transforms and url and DEFAULT among others. 
# its DEFAULT attribute points to the default Weights(configuration) as there may exist more
# than one Weights/configurations. usually these weights are labeled like IMAGENET1K_V1, 
# or IMAGENET1K_V2.
# so to access the inference transforms we have to choose a Weights/configuration first 
# (IMAGENET1K_V1 or IMAGENET1K_V2 or simply use DEFAULT to select the default Weights) 
# and then simply use `.transforms()`:
# 
# lets see how we can do this using an example: 
# grab the default Weights(configuration sounds much better than weights! as its really a config!)
weights = models.ResNet18_Weights.DEFAULT
# grab the transforms
preprocess = weights.transforms()
# get an image
img_url = 'https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg'
img = Image.open(requests.get(url=img_url,stream=True).raw)
# and apply it and get the preprocessed image:
img_transformed = preprocess(img)
# we can also use weights/config to initialize the model as well!
# Initialize model using its functional form
model = models.resnet18(weights=weights)
# set model to eval mode
model.eval()
# and then carry on with forward() and the rest
# but whats the difference between this and simply using 'pretrained=True'
# when instantiating our model? the simple answer is, using weights we can decide
# which weights to load, if there is only one weight, the pretrained=True does the job
# but if there are more, we can specify it this way. furthermore, we created the weights
# to get the transform, so we might as well, use it to instantiate our model anyway:)
# lets do a quick inference here:
def run_inference(model, img):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device ='cpu'
    model = model.to(device)
    img = img.to(device).unsqueeze(0)
    assert img.size(0) == 1, 'single image supported only'
    with torch.no_grad():
        model.eval()
        output = model(img)
        # The output has unnormalized scores (logits) 
        # To get probabilities, we need softmax().
        probs = output[0].softmax(dim=0)
        # grab the class names using weights' meta key!
        categories = weights.value.meta["categories"]
        # Show top 5 categories per image
        top5_prob, top5_idx = probs.topk(5,dim=0)
        print(f'-'*10)
        for i in range(top5_prob.size(0)):
            print(f'{categories[top5_idx[i]]}: {top5_prob[i].item()*100:.2f}%')
        print(f'-'*10)
run_inference(model, img_transformed)

# there are other ways to retrieve models and their corresponding weights:
# 1.Using `get_model(name, **config)` we can grab a prepared model by specifying
# the model name(model_builder_method i.e. the function's name we use to build models like resnet18()!),
# and weights and other paramters. note using models.ResNet18_Weights works as it selects the DEFAULT!
model = models.get_model('resnet18', weights=models.ResNet18_Weights)
run_inference(model, img_transformed)
# to initialize a raw model we can use get_model with no weight! (or simply use the builder function with pretrained=False!)
# this is a better choice than the builder_function, (like using models.resnet18())
# when you want to be able to test/work with multiple models. 
# this way you can quickly change a name from configfile and be ready to go without changng the code!
model_uninitialized = models.get_model("resnet18", weights=None)
# we can also use 'DEFAULT' to refer to the default weights for each model
# which again makes it easy to test several models without changing the code!
# using DEFAULT has the benifit of always using the up-to-date config(weights)
# it always points to the best weights. if this is not what you want, then you
# must use the specific weights name (like IMAGENET1K_V1 for example or whatever
# it is for your model)
model = models.get_model("resnet18", weights="DEFAULT")
run_inference(model, img_transformed)

# to get the said weights or configs using model's name only, 
# we can also use `get_model_weights(name)` the name is the function 
# we use to build the model (e.g. resnet18, mobilenet_v3_small, etc)
# (in torchvison documentation, its called 'builder_method' name!
# resnet18() is a builder_method!)
model_weights = models.get_model_weights('resnet18')
assert model_weights == models.ResNet18_Weights
print(f'{model_weights=}')
# then we can use it to get a fully prepared model!
model = models.resnet18(weights=model_weights)
run_inference(model, img_transformed)

# if we know the exact weight's name (i.e ResNet18_Weights.IMAGENET1K_V1),
# we can use 'get_weight()' to get the weights enum class (the class itself!)
model_weight_class = models.get_weight('ResNet18_Weights.IMAGENET1K_V1')
assert weights == models.ResNet18_Weights.IMAGENET1K_V1.DEFAULT
print(f'{model_weight_class=}')
# and we can use it and get a fully prepared model.
model = models.resnet18(weights=model_weight_class)
run_inference(model, img_transformed)

# note:
# we can use quantized versions as well
print(f'quantized_resnet18:')
model = models.get_model('resnet18', weights=models.quantization.ResNet18_QuantizedWeights)
run_inference(model, img_transformed)
# to be able to use the 'DEFAULT' as weights, we just need to alter the model name
# and prefix it with 'quantized_' to instruct the get_model to grab the quantized version
# is the same as doing
model = models.get_model('quantized_resnet18', weights='DEFAULT')
run_inference(model, img_transformed)
# I'll talk about quantization in a future chapter. for now, it suffices to know that
# quantization is a form of model compression where you decrease a model size and overhead
# (modelsize,memory/calculation overhead depending on the model/quantization type used)
# by using a lower precision (quantizing the weights e.g.) through a process known as quantization.
# instead of fp32, the weights will be in int8 e.g. massively decreasing model size, and also
# resuling in much faster inference speed. pytorch added early surrpot for quantization in 1.5
# and continually improved upon it(its easier and more hassle free compared to then!)
# the catch is, not all models react the same to quantization, some models such as resnet can 
# be quantized statically verywell with minimal reduction in accuracy, whereas models such as
# mobilenets are very sensitive to the process and can lose a lot of accuracy, these models 
# usually get a much better result when they are quantized during training.(known as QAT). 
# for now just know these models are good for inference
# train your models using the full weights, then for production quantize them. 
# we'll get to this later on in more depth inshaallah.

#%%
# to get a list of available models we can simply use list_models()
all_models = models.list_models()
# to get a list of models specific to classification or detection we simply
# use their modules!
classification_models = models.list_models(module=torchvision.models)
detection_models = models.list_models(module=torchvision.models.detection)
print(f'count: {len(all_models)}, {all_models=}')
print(f'count: {len(classification_models)}, {classification_models=}')
print(f'count: {len(detection_models)}, {detection_models=}')

# Using models from Hub
# 
# Most pre-trained models in torchvision can also be accessed directly via 
# PyTorch Hub without even having TorchVision installed:
# Option 1: passing weights param as string
model = torch.hub.load("pytorch/vision", "resnet18", weights="IMAGENET1K_V1")

# Option 2: passing weights param as enum
weights = torch.hub.load("pytorch/vision", "get_weight", weights="ResNet18_Weights.IMAGENET1K_V1")
model = torch.hub.load("pytorch/vision", "resnet18", weights=weights)

# You can also retrieve all the available weights of a specific model via PyTorch
# Hub by doing:

weight_enum = torch.hub.load("pytorch/vision", "get_model_weights", name="resnet18")
print([weight for weight in weight_enum])

# The only exception to the above are the detection models included on
# torchvision.models.detection. 
# These models require TorchVision to be installed because they depend on 
# custom C++ operators.

#%%
# torchvision models is not the only way we can load pretrained models. 
# one of the other ways we can access a huge list of models is through 
# torch.hub as we breifly discussed before. there are many more models 
# that are availble through hub, that are not in the torchvision models module.
# one of them is SimpleNetV1.
# lets start with SimpleNetv1. 
# This architecture came out back in 2016, and achieved sota in cifar10/mnist
# beating much larger networks at the time such as vggnet, resnet, etc
# it was several times faster in training and inference and consumed a lot less vram
# even less than mobilenetsv1/v2 which came after it( its tiny variants also perfromed
# way better than small architectures such as squeezenet which are availble in torchvison models). 
# The imagenet models were released in late 2022! 6 years after the first release of the paper
# and they outperform VGGNet and resnet variants such as resnet18,resnet34, and variants of 
# mobilenets and several other new architectures. 
# SimpleNet can be used as replacement for vggnet with a fraction of overhead. 
# There are much better architectures than simplenet today, but to get a sense of 
# how it performs and how we can use torch.hub to use it here it is : 
import torch
# use the latest master
# You are about to download and run code from an untrusted repository. 
# In a future release, this won't be allowed. 
# To add the repository to your trusted list, 
# change the command to {calling_fn}(..., trust_repo=False) and
# a command prompt will appear asking for an explicit confirmation of trust, 
# or load(..., trust_repo=True), which will assume that the prompt is to be 
# answered with 'yes'. 
# You can also use load(..., trust_repo='check') which will only prompt for confirmation if the repo is not already trusted. This will eventually be the default behaviour
# !wget 'https://raw.githubusercontent.com/Coderx7/SimpleNet_Pytorch/master/imagenet/simplenet.py'
#%%
model = torch.hub.load("coderx7/simplenet_pytorch", "simplenetv1_5m_m1", pretrained=True, trust_repo=True)
# or any of these variants at the moment
# model = torch.hub.load("coderx7/simplenet_pytorch:v1.0.0", "simplenetv1_5m_m2", pretrained=True, trust_repo=True)
# model = torch.hub.load("coderx7/simplenet_pytorch:v1.0.0", "simplenetv1_9m_m1", pretrained=True, trust_repo=True)
# model = torch.hub.load("coderx7/simplenet_pytorch:v1.0.0", "simplenetv1_9m_m2", pretrained=True, trust_repo=True)
# model = torch.hub.load("coderx7/simplenet_pytorch:v1.0.0", "simplenetv1_small_m1_05", pretrained=True, trust_repo=True)
# model = torch.hub.load("coderx7/simplenet_pytorch:v1.0.0", "simplenetv1_small_m2_05", pretrained=True, trust_repo=True)
# model = torch.hub.load("coderx7/simplenet_pytorch:v1.0.0", "simplenetv1_small_m1_075", pretrained=True, trust_repo=True)
# model = torch.hub.load("coderx7/simplenet_pytorch:v1.0.0", "simplenetv1_small_m2_075", pretrained=True, trust_repo=True)
model.eval()

# Download an example image from the pytorch website
import urllib
url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename)

# sample execution (requires torchvision)
from PIL import Image
from torchvision import transforms
input_image = Image.open(filename).convert('RGB')
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

# move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

with torch.no_grad():
    output = model(input_batch)
# Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
# print(output[0])
# The output has unnormalized scores. To get probabilities, you can run a softmax on it.
probabilities = output.softmax(dim=1)[0]
# print(probabilities)

# Download ImageNet labels
# !wget https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt


# Read the categories
with open("./data/imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]
# Show top categories per image
top5_prob, top5_catid = probabilities.topk(5,dim=0)
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], f'{top5_prob[i].item():.4f}')

# Thats pretty much it! you now should have a basic idea of how major chores can be done in Pytorch.
# there are more than what has been said here obviously, but for the start this should get you going
# see you in the next part!
# God bless all of you 
# %%
