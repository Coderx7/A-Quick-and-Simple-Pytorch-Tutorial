#%% [markdown]
# In the name of God the most compassionate the most merciful 
import torch 
import numpy as np 
import matplotlib.pyplot as plt 
%matplotlib inline

#%%
# here we are going to see how we can create a neural network and train and test it
# we will see how we can augment our data, create our datasets, etc . lets go 
# in Pytorch we can use the torchvision module, for reading existing datasets or create our own
# we also do augmentation using this module. this module also provides several wellknown achitectures
# such as AlexNet, VGGNet, ResNet, MobileNet, DenseNet, etc
# enough talking lets see how to use it 
# here lets import datasets for using the dataset capabilities
# use transforms for data-augmentation, and
# models for using existing models
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
                                      # for normalizing note we used 0.5, that , 
                                      # is needed since mean and std requires a tuple
                                      # and since mnist is grayscale(black n white)
                                      # or has 1 channel images only! we use 1
                                      # number only for mean and std.       
                                      transforms.Normalize(mean=(0.5,),std=(0.5,))]) 
# 
dataset_train = datasets.MNIST(root='MNIST', train=True, transform=transformations, download=True)
dataset_test = datasets.MNIST(root='MNIST', train=False, transform=transformations, download=True)

# now we have our datasets. but as you know, usually we dont load the whole dataset all atonce!
# instead we read in batches! in Pytorch we do this using a dataloader. using a dataloader
# we can easily specify a batchsize, and other performance related options such as number of workers
# which is the number of threads used to read from the dataset and thus provide a more efficient data
# loading!!! enough talking lets see how to create a dtaloader!
# the Dataloader resides in torch.utils.data !
import torch.utils.data as data
dataloader_train = data.DataLoader(dataset_train, batch_size=32, shuffle=True, num_workers=2)
# we do the same thing for test 
dataloader_test = data.DataLoader(dataset_test, batch_size=32, shuffle=False,num_workers=2)
print(f'test dataloader size: {len(dataloader_test)}')
# there is a note here. whenever you get wierd errors concerning dataloader/dataset, the first thing 
# you do first is  to set num_workers = 0. usually when you have a probelm in your dataset (specially
# when you create your own dataset) this will help you see the issue immediaitly. becasue setting num_workers=0
# uses the same thread to run everything, and  thus will catch your error, whereas when you set it to any number
# greater than 0, you will get a completely different error which is concerning threads failing! so have this in mind!


# OK, we defined our datasets, and dataloaders. lets inspect our data and see how they look ! 
imgs, labels = next(iter(dataloader_train))

# remember our images, are not tensors, and in order  to display them using matplotlib
# we must convert them back to normal numpy! arrays . and again since we normalized them 
# we must unnormalize them for visualization as well. lets write a function that accepts a batch of images
# with their labels and displays them! 
def visualize_imgs(imgs, labels, row=3, cols=11,):
    # images in pytorch have their axes swapped! 
    # so we first fix their orders firts! 
    # the dim is batch, c, h, w, but we should be having 
    # batch, h, w, c  
    # we used detach to clone images (we dont have to do this, but its good practice for later)
    imgs = imgs.detach().numpy().transpose(0, 2,3,1)
    # now we need to unnormalize our images. 
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
def visualize_img(img):
    img = img.numpy().transpose(1,2,0).squeeze()
    fig = plt.figure(figsize=(28,28))
    ax = fig.add_subplot(1,1,1)
    # or 
    #ax = plt.subplot(111)
    ax.imshow(img, cmap='gray')
    threshold = 0.5
    w,h = img.shape
    for i in range(h): 
        for j in range(w):
            ax.annotate('{:.2f}'.format(img[i,j]),xy=(j,i),
            horizontalalignment='center',
            verticalalignment='center',
            color='white' if img[i,j]<threshold else 'black')

visualize_img(imgs[0])

#%%
# OK now that we have done  this lets  see how we train a model ! 
# before that we need a model 
# we can create one or use an existing one, 
# first lets see how to use an existing model ! 
model = models.AlexNet(10)
# thats it now we have a model the that we can use for training! however there is a catch like always!
# this is alexnet. and alexnet was trained for imagenet challange. in imagenet chalange the image size
# that were used to train this model were 224x224x3. so this model accepts 224x224x3 images as input!
# while our input is a 28x28x1 images!! this wont simply work!!! 
# either we have to resize our images from 28x28 to 224x24! (using resize() in transformers up there!)
# and also make images 3 channeled instead of 1! this is too much work for now! instead  
# so lets create our own model first and then later on see how we can use these models on new data!

# for creating a model, we simply define a new class that inherits from torch.nn.Module() 
# and then define our layers and ultimately specify the sequence of how these layers are used in 
# forward() method. lets see how all of this can be implemented! (this is very easy!)
import torch.nn as nn 
import torch.nn.functional as F 
class ourNetwork(torch.nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # lets create a simple multilayer preceptron (4 fully connected layer network!)
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
        # for flattening we do this two way!
        # 1. we directly enter the size since we know the dims. 
        # input_images = input_images.view(-1, 28*28)
        # the bad thing about this is that if later we change our image dims
        # we have to comeback here and change the dims to match the new dims (e.g. 32x32)
        # a better way would be to specify the batchsize and then let the dims be automatically
        # infered like this : 
        batch_size = input_images.size(0)
        input_images = input_images.view(batch_size, -1)
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
# now lets go for the last part which is training /testing. 
# we will need 
# 1. an optimizer such as sgd or adam, etc 
# 2. a criterion (a loss function ) 
# 3. thats all. 
# all optimizers reside in torch.optim 
# from torch import optim 
# our optimizers, take at least two paramters, 1. model parameters
# and 2. the learning rate (they can take, weight decay, momentum etc as well)
# but the model parameters, and lr are the very minimum requirements 
# remember if you set too of a high learning rate, you will see your loss
# will not decrease and may also not increase!! it may very well get stuck
# to or around a value . therefore your val_acc would also perform similarly
# when you use batchnormalization, you maynot see this as you can use considerably
# higher learning rate. but knowing this is important. set this learning rate to 0.1 
# forexample and see the outcome. then reset it to 0.001 and rerun the training loop
# and witness the change
optimizer = torch.optim.Adam(our_model.parameters(), lr = 0.001)
# for our loss function, its customery to use the name criterion, 
# you can use anyname you like, but criterion is overwelmingly popular! 
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

    for imgs, labels in dataloader_train: 
        # we set our model to train mode 
        # this is specially needed if we use dropout or batchnormalization
        # or pretty much any layers that during training and testing have different behaviors!
        our_model.train()
        i+=1
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
            print(f'epoch/iter: {e}/{i} training-loss: {loss.item()}')

    # lets run evaluations on test set after each epoch :
    total_corrects = 0
    total_loss_val = 0.0
    acc_val = 0.0
    class_correct_counter = torch.zeros(size=(10,), dtype=torch.int32)
    class_total_samples = torch.zeros(size=(10,), dtype=torch.int32)
    total_corrects2 = 0
    uncorrect_imgs_predicted_list = []
    for imgs, labels in dataloader_test:
        our_model.eval()
        
        imgs = imgs.to(device)
        labels = labels.to(device)
        preds = our_model(imgs)
        loss_val = criterion(preds, labels)
        # now lets see how the predictions looks like and what accuracy we get 
        # we can calculate the accuracy using a multitudes of ways! 
        # we'll see 3 ways of doing it. we start with 
        # the easiest/normalest way first. 
        # first method : 
        # in the first method, we get the index of highest probabilities in our predictions 
        # and compare them with our labels. our comparsion will give us a series of
        # true/false values which indicate, where our predictions were inline with our labels
        # and where they were different. we can then simply convert true /false to 1 and 0, and
        # then by suming all of them we get the total number of correct predictions. then dividing
        # this number by the total number of samples in our test set, we get our accuracies.
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
        # respective elements were the same or not! there for we use torch.eq becasue we 
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
        # for example suppose, we want to get the accuracy for top5 (like in imagenet)
        # how could we do it ? there are many ways but one of the easiest ways is using topk
        # lets see how we can do that : 
        # each tensor has a method named topk, topk accepts an argument k, which we specify
        # if we want top5 we set k=5, if we want top1 (which is normal accuracy) we set k=1. 
        # topk, like max, returns two results. the first is the top values and the second is their 
        # respective index. so lets see it in action . 
        top_values, indexes = preds.topk(k=1, dim=1)
        # print(indexes.shape, labels.shape)
        # we use squeeze on indexes so its shape becomes [32] instead of [32x1]
        result2 = torch.eq(indexes.squeeze(), labels).float()
        # remember since we are using topk, our 'indexes' has  a shape of [32, 1] while our label is 
        # [32]. this is important, becsaue if we compare these two, in newver versions of Pytorch (0.4+)
        # this will be broadcasted and the result will be a tensor of [32,32] while it should be [32]
        # If we do (indexes == labels) or torch.eq(indexes, labels) 
        # results will have shape (32, 32), What it's doing is comparing the one element in each row of
        # indexes with each element in labels which returns 32 True/False boolean values for each row.
        # therefore in order to not face this issue (which will not give you any error, but mess up your result!)
        # always make sure what you are comparing with have the same exact shape. 
        # one way is to use squeeze, which we just saw! 
        result2 = (indexes.squeeze() == labels)
        # the other way is to do sth like this : 
        result2 = (indexes.view(*labels.shape) == labels)
        # uncomment to see the shapes 
        # print(f'indexes: {indexes.shape} result(eq) {result.shape} result2(==) {result2.shape}')
        
        total_corrects2 += result2.float().sum()
        
        # 3rd Method  :
        # now we just found out about two ways of calculating the accuracy , so far we counted
        # the number of correct samples and then divided them by the total number of samples 
        # at the end and got our accuracy. we could also calculate accuracy per batch and then 
        # add all these accuracies and then divide them by the to tal number of batches. 
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

        total_loss_val += loss_val.item()


        # lets see a more detailed analysis concerning our model performance
        # that is, lets see, which classes are being predicted better  than others 
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
                uncorrect_imgs_predicted_list.append((imgs[i],indexes[i],labels[i]))

    for i in range(10):
        print(f'class {i} : Total Samples : {class_total_samples[i].item()} / {class_correct_counter[i].item()}'\
                f' acc: {class_correct_counter[i].item()/class_total_samples[i].item()}')

    
    print(f'test set samples: {len(dataloader_test)}')
    print(f'accuracy_val(total corrects//dataset_size): {total_corrects/len(dataloader_test.dataset):.4f}'\
          f'\naccuracy_val(topk(k=1)): {total_corrects2/len(dataloader_test.dataset):.4f}'\
          f'\naccuracy_val(topk(k=1)-per batch acc): {acc_val/len(dataloader_test):.4f}'\
          f'\nloss_val: {total_loss_val/len(dataloader_test.dataset):.6f}')

      

#%%
# lets visualize our wrongly predicted images and their wrong/true classes 
def visualize_wrongly_predicted(imgs_list, cols = 10):
    count = len(imgs_list)
    rows = np.ceil(count/cols)
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

visualize_wrongly_predicted(uncorrect_imgs_predicted_list)
#%%
# OK great we created a network from scratch and trained/tested it successfully . 
# you probably have couple of questions(ok many questions!). one of them is probabaly
# concerning the way we created our network. if you look back again, you'll see we didnt use 
# any softmax layer at the end? why? 
# as you know, in classification problems, we usually use a softmax layer at the very end
# of the network to get probablities for our predictions. 
# we also know that we usually use softmax with crossentropy loss. 
# Yet here we used crossEntropy without softmax? what is it?  

# we can use softmax with crossentropy but if you read the documentations,
# you'll notice that when we used crossentropyloss, it acually requires 
# 'logits' or 'raw scores' as inputs. 
# it then applies a 'logsoftmax' on its input (logits), and then applies 
# a negative log likelihood afterward.
# So we can have softmax as our last layer, but for calculating loss, we 
# must send the logits.  

# But Why do we use logits and not softmax itself ? 
# we use logits and not softmax, becasue softmax gives us probablities, which are floating
# point numbers ranging from 0. to 1. and the critical issue with floating point numbers is 
# that floating point numbers can not accurately represent numbers close to 0 or 1 and thus we 
# face numerical instabilities. Therefore we use raw scores or logits.
# 
# OK, but what about log_softmax? why do we use log_softmax instead of softmax then?
# there are several reasons for this. one of them is that log_softmax is numerically stable and doesnt
# have the problems associated with softmax. it also plays nicer with crossentropy 
# (https://datascience.stackexchange.com/questions/40714/what-is-the-advantage-of-using-log-softmax-instead-of-softmax) 
# so in short, having log probabilities, help both in numerical stabilty and optimization performance 

# From wikipedia  :
#  A log probability is simply a logarithm of a probability. The use of log probabilities means 
# representing probabilities on a 'logarithmic scale', instead of the standard [0,1] unit interval.

# Since the probability of independent events multiply, and logarithms convert multiplication to addition,
# log probabilities of independent events add. Log probabilities are thus practical for computations,
# and have an intuitive interpretation in terms of information theory: the negative of the log probability
# is the information content of an event. Similarly, likelihoods are often transformed to the log scale, 
# and the corresponding log-likelihood can be interpreted as the degree to which an event supports a
# statistical model. The log probability is widely used in implementations of computations with
# probability, and is studied as a concept in its own right in some applications of information theory,
# such as natural language processing. 

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
# to get probablities at test time, when you used logs0ftmax, just use torch.exp() on 
# your result and you re done. 
# that was it! 

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
# we can access them directlt. each weight has a data and grad attributes 
# the data as the name sounds, contains the data, while grad contains the weights 
# gradients. 
# this is thecase for bias as well, it has a data and a grad attribute. 
# lets create a simple fc layer and initialize its weights and biases in different way 
fc = torch.nn.Linear(1,1)
# initialize using normal distribution 
# since data is a normal tensor, it has access to all methods available to any  tensors
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
# now to since there is no operation done yet there is no gradients yet! 
print(f'grad: {fc.weight.grad}')

# the second way we can initialize a modules weights and biases
# is using torch.nn.init module. basically  this is what we use nearly 99%
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


# now suppose we want to initialize all layers using a specific initialization scheme, how do we do it?
# its easy we do it like this : 
#first lets create a simple dummy model 
import torch.nn as nn 
model = torch.nn.Sequential(*[nn.Linear(1,1), nn.ReLU(),
                             nn.Conv1d(1,1,1), nn.ReLU()])
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
# we first write down a function that does the intialization and then apply this function on the model
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
# we wil create networks using different ways that are as follows: 
# 1. using the simple layer definition (what we just saw)
# 2. using nn.Sequential() to create a sequential model 
# 3. using ordered_dict to create named modules. 
# 4. using ordered_list to create a list of layers. 
# 5. using new modules 
# This list probably doesnt make any sense to you, but when we try implementing them you'll understand 
# why we covered each here. 
# lets start wil the first method: 
# 1. In the first method, we first define all of the layers we need and then in forward() 
# we simply call them and use them in any order we like. here is a simple example 
class simple_net(nn.Module):
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
# and instead use  the functional form like this. we can access the functional forms
# from torch.nn.functional module.  
import torch.nn.functional as F 
class simple_net2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.conv2 = nn.Conv2d(6, 12, 2)
        self.pool1 = nn.MaxPool2d(2,2)

    def forward(self, x):
        output = F.relu(self.conv1(x))
        output = F.relu(self.conv2(x))
        return self.pool1(output)

# as you may have guessed we can use the functional form for any of this, its up to you
# when and where to use either of them. usually we use activation functions, this way as
# its more consise, easy to use and more readible. 
# Also as you know, you can have control over which layer gets executed, so you can have
# different if clauses in your foward() as well. 

# 2. using nn.Sequential() class
# If we want to create a plain network which is just a series of layers in succession, 
# we can ease ourseleves using the nn.Sequential() class. this class as the name implies
# applies a series of layers in succesion. 
# nn.Sequential class is in fact a sequential container. 'Modules' will be added to it in
# the order they are passed in the constructor. 
# Alternatively, an ordered dict of modules can also be passed in.
# As you see, there are several ways we can create a network using nn.Sequential() class. 
# below I'll show some ways that comes to my mind: 
# first lets see what the : 
# "'Modules' will be added to it in the order they are passed in the constructor. "
# mean. this means, the layers order are specified when we specify them in the constructor
# see the example below to understand this :  

class sequential_net1(nn.Module):
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
class sequential_net2(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(OrderedDict([
                                                "conv1", nn.Conv2d(3, 6, 3),
                                                "relu1", nn.ReLU(),
                                                "conv2", nn.Conv2d(6, 12, 3),
                                                "relu2", nn.ReLU(),
                                                "maxpool1", nn.MaxPool2d(2, 2)
                                                ]))

    def forward(self, inputs):
        return self.model(inputs)

# we can also use a 'list' of layers with our nn.Sequetntial class. 
# here we use a list of layers with nn.Sequential
class sequential_net3(nn.Module):
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
class sequential_net4(nn.Module):
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
class sequential_net5(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential()
        self.model.add_module('conv1', nn.Conv2d(3, 6, 3))
        self.model.add_module('relu1', nn.ReLU())
        self.model.add_module('conv2', nn.Conv2d(6, 12, 3)) 
        self.model.add_module('relu2', nn.ReLU())
        self.model.add_module('maxpool2', nn.MaxPool2d(2,2))

    def forward(self, inputs):
        return self.model(inputs)

# Similarly to what we saw with ModuleList, we can have ModuleDict
# which is an 'ordered' dictionary (unlike Python dict which doesnt
# preserve the order of items when inserted). 
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
# (e.g., Python's plain dict) does not preserve the order of the merged mapping.

class sequential_net6(nn.Module):
    def __init__(self):
        super().__init__()
        # the order of insertion is preserved
        self.model = nn.ModuleDict()
        # we can use the add_module
        self.model.add_module('conv1', nn.Conv2d(3, 6, 3))
        # or simply use the ordinary way!
        self.model['relu1'] = nn.ReLU()
        self.model['conv2'] = nn.Conv2d(6, 12, 3)
        self.model['relu2'] = nn.ReLU()
        self.model['maxpool1'] = nn.MaxPool2d(2, 2)

    def forward(self, inputs):
        return self.model(inputs)

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
class sequential_net7(nn.Module):
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
# now using this scheme, we can easily later on swap any part of the network, like the classifer
# or easily use the feature extractor of our network. for example when finetuning, we can create a new
# classifer and assign it to our model.classifier and retrain our model without 
# even changing one line of our architecture. we will see this in action when we do finetuning.

# another benifit of using Sequential is that we can create different building blocks 
# for our networks. we just saw an example, lets expand on that.  

# here we will create a function for creating a convolution layer with batchnorm and 
# relu. here we create a list of layers and then send this list to nn.Sequential
def convlayer(input_dim, output_dim, kernel_size=3, stride=1, padding=1, batchnorm=False):
    layers = nn.ModuleList()
    conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, padding)
    layers.append(conv)
    if batchnorm: 
        layers.append(nn.BatchNorm2d(output_dim))
    layers.append(nn.ReLU())

    return nn.Sequential(*layers)

class sequential_net8(nn.Module):
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

# as you can see, we made our code much more consise !! and much more readdable!
# we can make this even better and turn this into a 1 liner like before! 
# lets see 
class sequential_net9(nn.Module):
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
simple_net1 = simple_net()
simple_net2 = simple_net2()
sequentialnet1 = sequential_net1()
# sequentialnet2 = sequential_net2()
sequentialnet3 = sequential_net3()
sequentialnet4 = sequential_net4()
sequentialnet5 = sequential_net5()
sequentialnet6 = sequential_net6()
sequentialnet7 = sequential_net7()
sequentialnet8 = sequential_net8()
sequentialnet9 = sequential_net9()

print(simple_net1)
print(simple_net2)

print(sequentialnet1)
# print(sequentialnet2)

print(sequentialnet3)
print(sequentialnet4)

print(sequentialnet5)
print(sequentialnet6)

print(sequentialnet7)
print(sequentialnet8)
print(sequentialnet9)


#%% 
# now when we want to access as pecific layer(module) in our model, as you can see
# based on how we defined our model, it will be different.
# for example for a simple_nets we can easily access each module using
# model.modules() generator like this : 
for m in simple_net1.modules():
    print(m)
# and access each and every layer we like. we saw this in action when we wanted
# to initialize layers weights and biases. 
# we also can use _modules which is an ordered dict! and access any layer by their name!
print(simple_net1._modules) 
# get the conv1 number of input channels: 
print(f"conv1 input channels : {simple_net1._modules['conv1'].in_channels}")
#%%
# now that we learnt how to create a neural network, lets see how we can create a new layer 
# or module in Pytorch! 
# As always there are many ways to do this but here I'll try to demonstrate something simple 
# and easy to follow . 
# lets create a flatten layer for Pytorch that we can use in our nn.Sequential
# so that we can flatten a convolution layers output so that we can use it in a
# fully connected layer (nn.Linear). 
# As you have seen , before we can add a fc layer after a conv layer
# the output needs to be flattened and there is no layer for flattening in Pytorch as 
# I'm writting this. (Pytorch 1.0)
# up until now, we have been using something like : 
# output = output.view(x.size(0), -1)
# in our forward() method to flatten an output .and we couldnt do this in a nn.sequential!
# lets create a new module(layer in Pytorch)! 
# for creating a new module/layer, we inherit from nn.Module. 
# the rest is self explanatory. lets see how it is done 
class Flatten(nn.Module): 
    # our flatten layer doesnt need any argument, as it only 
    # flattens the input!
    def __init__(self):
        super().__init__()
        
    def forward(self, x): 
        # uncomment to see the changes if you like
        print(f'before flattening: {x.shape}')
        x = x.view(x.size(0), -1)
        print(f'after flattening: {x.shape}')
        return x 
# thats all ! 
# our flatten layer doesnt need any arguements, nor does it need any layer
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

# before that let us try to create a ResNet architecture. 
# Resnet as you know is one of the mostly used architectures out there, and the 
# main building of this network is something called a resblock which is just a block 
# # with residual connection! let us create this network. 
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
        self.conv1 = nn.Sequential(nn.Conv2d(in_dim,
                                             out_dim,
                                             kernel_size, 
                                             stride=stride,
                                             padding=padding),
                                    nn.BatchNorm2d(out_dim))                               
        

    def forward(self, x):
        output = self.conv1(x)
        output += x
        F.relu(output)
        return output
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
torch.save(model.state_dict(),'mymodel.t')
# you can get fancier and take advantage of the fact that torch.save() accepts a dictionary 
# and thus add more items to be saved! for example, you can save the parameters you used to
# instantiate your model, optimizer, schedulers, save the loss, the best acc, epochs passed, etc. 
# just create a dictionary and add what you want and use this dictionary instead to save your 
# model. 
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
torch.save(settings, 'ourmodel.t')
#  thats it 
print('save done!')
#%%
# now to load the settings 
# for this we use  torch.load! but thats not all! lets see how it works!
# we first load the whole dictionary from our model! 
model_settings_dict = torch.load('ourmodel.t')
# now that we have our dictionary of settings, lets load the values 
# before we continue lets create the variables with garbage values so
# we know for sure the loading is done successfully 
e = -111111
acc_val = -0.00000
acc_train = 0.00000
loss_train = 0.00000
model = Resnet() 
optimizer = torch.optim.Adam(model.parameters(), lr = 0.555555)
# now lets start loading! 
e = model_settings_dict['epoch']
acc_val = model_settings_dict['acc_val']
acc_train = model_settings_dict['acc_train']
loss_train = model_settings_dict['loss_train']
# now for optimizer and our model, we can simply do this, instead
# we use a specific method in our model/optimizer called load_state_dict 
model.load_state_dict(model_settings_dict['state_dict'])
optimizer.load_state_dict(model_settings_dict['optimizer'])
print('load done! ')
print('some loaded variables: ')
print(optimizer)
print(e)
print(acc_val)
print(acc_train)

#%%
# OK we are ready to see how fine-tuning works in Pytorch 
# there are several models in models repository that we can use
# lets choose one but before that lets see what we have at our disposal 
from torchvision import models
import torch.nn as nn 

print(dir(models))
# lets use resnet18! by setting pretrained = True, we'll also be down-
#loading the imagenet weights. 
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
resnet18.fc = nn.Linear(512, 10)
# instead of hardcoding the 512 which we saw by looking at the 
# printed version of our model, we can simply use the 
# 'in_features' attribute of the fc layer! and write : 
# resnet18.fc = nn.Linear(resnet18.fc.in_features, 10)

print(f'\nNEW MODEL(after adding the new fc layer): \n{resnet18}')
# now before we dive in to train our network we should first 
# freeze all layers except this new one, and train for several epochs, 
# so that it converges to a reasonable set of weights
# then we unfreeze all previous layers and train the whole net
# altogether again. 
# So lets freeze all layers before this fc layer!
# for module in resnet18.modules():
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

# we could also use the isinstance!: 
# for module in resnet18.modules():
#     if not isinstance(module, nn.Linear):
#         for param in module.parameters():
#             param.require_grad = False
#     if isinstance(module, nn.Linear):
#         for param in module.paramertes():
#             param.require_grad = True

# However, a better way is to just use parameters 
# and split it into two 
for param in resnet18.parameters():
    param.requires_grad = False 
# and for fc 
for param in resnet18.fc.parameters():
    param.requires_grad = True
# in case we want to have the parameters name, we can use named_parameters!
# print('Freezing All layers')
# for name, param in resnet18.named_parameters():
#     param.requires_grad = False
#     print(f'name: {name} requires_grad : {param.requires_grad}')        

# print('\nUnfreezing the new FC layer')
# for name, param in resnet18.fc.named_parameters():
#     param.requires_grad = True      
#     print(f'name: {name} requires_grad : {param.requires_grad}')


# check all layers status again!
for name, param in resnet18.named_parameters():
    print(f'{name} : {param.requires_grad}')            

#%%
# first version ! 
# lets create a training and testing functions for our case 
def validation(model, dataloader_test, criterion, k, device):
    loss_total = 0.0
    acc_perbatch = 0.0
    # this disables gradient acumulation 
    # Context-manager that disabled gradient calculation.
    # Disabling gradient calculation is useful for inference,
    # when you are sure that you will not call Tensor.backward().
    # It will reduce memory consumption for computations that would
    # otherwise have requires_grad=True. 
    # In this mode, the result of every computation will have requires_grad=False,
    # even when the inputs have requires_grad=True.
    with torch.no_grad(): 
        for imgs, labels in dataloader_test:
            imgs, labels = imgs.to(device), labels.to(device)
            # actiate evaluation mode 
            model.eval()
            preds = model(imgs)
            loss_val = criterion(preds, labels)
            _, indexes = preds.topk(k, dim=1)
            results = (indexes.view(*labels.shape) == labels).float()
            acc_perbatch += torch.mean(results)
            loss_total += loss_val.item()
        
        acc = acc_perbatch/len(dataloader_test)
        loss_final = loss_total/len(dataloader_test)
    return loss_final, acc

def training(model, dataloader_train, dataloader_test, epochs, criterion, optimizer, k, interval, device):

    model = model.to(device)
    # activate trainig mode
    model.train()
    training_acc_losses = []
    val_acc_losses = []
    trainig_batch_count = len(dataloader_train)
    test_batch_count = len(dataloader_test)

    for e in range(epochs):
        
        acc_per_batch = 0.0
        training_loss = 0.0
        for i, (imgs, labels) in enumerate(dataloader_train):
            imgs = imgs.to(device)
            labels = labels.to(device)

            preds = model(imgs)
            loss = criterion(preds, labels)
            
            # calculate training accuracy 
            _, class_indexes= preds.topk(k=1, dim=1)
            results = (class_indexes.view(*labels.shape) == labels).float()
            acc_per_batch += torch.mean(results)

            training_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % interval == 0 : 
                print(f'(epoch/iter): {e}/{i} train-loss: {loss.item():.6f} train-acc: {acc_per_batch/trainig_batch_count:.4f} ')
                
        
        # accumulate accuracies and losses per epoch
        training_acc_losses.append( (acc_per_batch/ trainig_batch_count, training_loss/ trainig_batch_count))
        # run validation test at every epoch! 
        val_acc_loss = validation(model, dataloader_test, criterion, k=1, device=device)
        val_acc_losses.append(val_acc_loss)
        print(f'val_loss : {val_acc_loss[0]:.4f} val_acc: {val_acc_loss[1]:.4f}')

    return training_acc_losses, val_acc_losses

#%%
epochs = 10
k = 1
batch_size = 32
interval = 1000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(resnet18.parameters(), lr = 0.001)

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

dataset_train = datasets.CIFAR10('CIFAR10', train=True, transform = transformations_train, download=True)
dataset_test = datasets.CIFAR10('CIFAR10', train=False, transform = transformations_test, download=True)

dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size = batch_size, shuffle=True, num_workers = 2)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size = batch_size, shuffle=False, num_workers = 2)

#%%
train_info, val_info = training(resnet18, dataloader_train, dataloader_test, epochs, criterion, optimizer, k, interval, device)

#%%
# now unfreeze all layers and retrain the whole network
for param in resnet18.parameters():
    param.requires_grad = True

optimizer = torch.optim.SGD(resnet18.parameters(), lr = 0.001)
train_info, val_info = training(resnet18, dataloader_train, dataloader_test, epochs, criterion, optimizer, k, interval, device)

# we can decrease the learning rate at some intervals and have better convergance
# optimizer = torch.optim.SGD(resnet18.parameters(), lr = 0.0001)
# train_info, val_info = training(resnet18, dataloader_train, dataloader_test, epochs, criterion, optimizer, k, interval, device)

# optimizer = torch.optim.SGD(resnet18.parameters(), lr = 0.00001)
# train_info, val_info = training(resnet18, dataloader_train, dataloader_test, epochs, criterion, optimizer, k, interval, device)

#%%
# At this rate you must be thinking this is rediculous! cant we just create a loop where we iteratively/periodiclly 
# decrease the learning rate? I should say yes it is. a  simple form for this would be :     
lr=0.001
for i in range(3):
    lr =  lr * 0.1
    optimizer = torch.optim.SGD(resnet18.parameters(), lr = lr)
    train_info, val_info = training(resnet18, dataloader_train, dataloader_test, epochs, criterion, optimizer, k, interval, device)
#%%
# but the best method would be to use MultiStepLR from 
# torch.optim.lr_schedule module.
# There are other counterparts as well which you can use
# Lets reimplement our training loop, this time in a better way! 
# if you look back, you can see there is a lot of duplication that
# we can get rid of! 
def train_validation_loop(model, dataloader, optimizer, criterion, is_training,
                          device, topk=1, interval=1000 ):
    
    preds = None 
    loss = 0.0
    loss_total = 0.0
    accuracy_total = 0.0
    total_batches = len(dataloader)
    status = 'training' if is_training else 'validation' 
    
    for i, (imgs, labels) in enumerate(dataloader):
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

        loss_total += loss.item()
        _, class_idxs = preds.topk(k, dim=1)
        results = (class_idxs.view(*labels.shape) == labels).float()
        accuracy_total += torch.mean(results)

        if i % interval == 0: 
            _, class_idxs = preds.topk(k, dim=1)
            results = (class_idxs.view(*labels.shape) == labels).float()
            accuracy_per_batch = torch.mean(results)
            print(f'{status} loss/accuracy(per batch): {loss:.6f} / {accuracy_per_batch:.4f}')

    # calculate the loss/accuracy after each epoch        
    final_loss = loss_total/total_batches 
    final_accuracy = accuracy_total/total_batches
    return final_loss, final_accuracy

# now our main loop 
def training_loop(model, dataloader_train, dataloader_test, optimizer,
                  criterion, scheduler, device, epochs=10, topk=1, interval=1000):
    for e in range(epochs):
        train_loss, train_acc = train_validation_loop(model, dataloader_train, optimizer, criterion, True, device)
        test_loss, test_acc   = train_validation_loop(model, dataloader_test, optimizer, criterion, False, device)
        scheduler.step()
        print(f'(epoch) {e}: lr: {scheduler.get_lr()[0]:.8f} \n\ttraining loss/accuracy: {train_loss:.6f} / {train_acc:.4f}\n'
        f'\ttesting loss/accuracy: {test_loss:.6f} / {test_acc:.4f}')
        # we can specify at which epoch or learnng rate
        # reactivate/unfreeze all previous layers 
        # using sth like 
        # if e == 3:
        #     for param in model.paramertes():
        #         param.requires_grad = True
        
#%% 
from torch import optim
optimizer = optim.SGD(resnet18.parameters(), lr = 0.01)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2, 3, 5, 8], gamma = 0.1)
criterion = nn.CrossEntropyLoss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet18 = resnet18.to(device)
print (f'initial lr: {scheduler.get_lr()}')

training_loop(resnet18, dataloader_train, dataloader_test,
              optimizer, criterion, scheduler, device, epochs)

#unfreeze all layers and retrain 
print('unfreezing all layers now!')
[p.requires_grad_(True) for p in resnet18.parameters()]

scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2, 4], gamma = 0.1)
print (f'initial lr: {scheduler.get_lr()}')
training_loop(resnet18, dataloader_train, dataloader_test,
              optimizer, criterion, scheduler, device, epochs)

# impressive! we could easily achieve 95.53% accuracy in several epochs with 
# some very primitive techniques. we can get much higher accuracy if we invest in 
# hyper parameter tuning, etc.  for now this is enough as we just wanted to demonstrate
# how things can be done in Pytorch!
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
# VGG ? 
vgg16 = models.vgg16()
print(f'vgg16 model: {vgg16}')
# this is like alexnet, it has features, and classifier attributes we can use
# for feature extraction or finetuning, etc
vgg16.classifier[6] = nn.Linear(vgg16.classifier[6].in_features, 10)
print(f'vggnet16 model after change: {vgg16}')

#%% 
# Next is SqueezeNet, 
# squeezenet is a very small architecture that provides alexnet level accuracy with 50x less parameters
# there are  two versions 1_0 and 1_1 that provides a bit more accuracy! than the former one!
squeezenet = models.squeezenet1_1()
print(f'squeezenet model: {squeezenet}')
# similar to all previous architectures so far, it has a features and a classifier attribute
# it uses global average pooling to get the final scores and is fully convolutonal !
# we can swap the whole classifier , but we can also take advantage of the fact that its
# a fully convolutional neural network and we can accept images of an size without any issues
# unlike the former architectures that were trained on a fixed size images and could only use fixed-size
# images at test time. here, we change the last convoloution layer output channels and make it equal 
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
# Thats pretty much it! you now should have a basic idea of how major chores can be done in Pytorch.
# there are more than what has been said here obviously, but for the start this should get you going
# see you in the next part!
# God bless all of you 