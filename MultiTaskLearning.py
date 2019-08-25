#%% 
# In the name of God the most compassionate the most merciful
# In this part we are going to see how we can do multi-task learning in Pytorch
# we may have two parts but I'm not sure yet. 
# in the first example, we will build a multitask model that will do multi-label
# classification among its task. I was thinking to dedicate a whole part to multilabel classification,
# but I'm not sure yet, knowing that we'll be implementing one here. 
# lets see how this goes. If at the end of this part, I see we need a
# separate session for multi label classification I'll create one. 
# lets start 
# first let us import the basic modules 
import torch
import torch.nn as nn 
import torch.nn.functional as F 
from torch import optim 
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt 
from sklearn import metrics
%matplotlib inline

#%%
# Before we continue lets talk about our dataset and what we want to do.
# I searched for a multi task dataset that has a multilabel classification as well
# but couldnt find any, I nearly gave up untill luckily, found and donwloaded a small dataset
# of anime images from https://github.com/sugi-chan/pytorch_multitask, the dataset is not perefect
# there are duplicates, and its not great, but for the sake of learning its good. 
# and we are going to classify each image into several categories, for example, 
# we want to know the fighting style gender, region, image colors, etc
# our dataset contains 406 images for training and 51 images for testing(not validation!)
# which means, there are no labels for them! if we want, we can create a validation set
# using  torch.utils.data.SubsetRandomSampler() class. 
# our data is provided in two folders one containing training samples and the other tests
# There is also an accompanying .csv file which specifies the labels for each image. 
# since our labels are in a csv file, and this is not a simple multi-class classification
# we cant use ImageFolder() class we previously used in our examples. we instead
# will be creating our own Dataset! 
# we will inherit from torch.utils.data.Dataset and implement __getitem__ and __len__ methods. 
# thats all it needs. however, we will also add couple of more methods to our newly to be created
# dataset that will aid us in the process. (for example, we will save the label names, so later on
# when we want to see how our model performs, we can easily print the actual label names otherthan
# their crude form of a tensor containing 0s and 1s or etc!)
# but before that, lets see how our dataset looks like! that is lets have a look at our .csv file!
# that hosts our labels and path to our images
# This is how it looks 
# ,full_path,image_name,name,white,red,green,black,blue,purple,gold,silver,gender_Female,gender_Male,region_Asia,region_Egypt,region_Europe,region_Middle East,fighting_type_magic,fighting_type_melee,fighting_type_ranged,alignment_CE,alignment_CG,alignment_CN,alignment_LE,alignment_LG,alignment_LN,alignment_NE,alignment_NG,alignment_TN
# 1,images/014 Atalante 2.png,014 Atalante 2.png,atalante,1.0,0.0,1.0,0,0.0,0,0.0,0.0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,1,0,0
# 2,images/0144.jpg,0144.jpg,atalante,0.0,0.0,1.0,1,0.0,0,1.0,0.0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,1,0,0
# 25,images/1132.jpg,1132.jpg,xuanzang,1.0,0.0,0.0,0,0.0,0,1.0,0.0,1,0,1,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0
# 26,images/1133.jpg,1133.jpg,xuanzang,1.0,1.0,0.0,0,0.0,0,1.0,0.0,1,0,1,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0
# it has a header that specifies different columns role. 
# looking at our data we see, we have several categories : colors , genders, regions, and fighting styles,
# etc among these categories, only colors can have more than 1 value ( that is their values are not mutually
# exclusive. we can have both black ,blue gold and white at the same time) so color is multilabel. 
# for a normal single label classfication, we use crossentropy and in Pytorch, we simply use the index of 
# the correct class and do not feed the one hot encoded representation of the true class. 
# for a multi label case, we use BCE (BinaryCrossEntropy) and use the one hot encoded representation of 
# labels. 
# when building our dataset class, we need to provide labels in the proper form as well. 
# so lets get busy!

#%%
# we use csv for reading csv file
import csv
# we use PIL.Image for reading an image
import PIL.Image as Image
# for working with path, files and folders
import os

class AnimeMTLDataset(torch.utils.data.Dataset):
    def __init__(self, image_folder, csv_file_path, transformations, is_training_set = True) :
        super().__init__()
        
        self.path = csv_file_path 
        self.transforms = transformations
        self.is_training_set = is_training_set
        self.image_folder = image_folder
        self.dataset = {}
        self.column_names = None

        if self.is_training_set:
            # read the csv file into a dictionary
            with open(csv_file_path, 'r') as csv_file :
                # now we have a generator that when called
                # will read one line. 
                csv_reader = csv.reader(csv_file)
                # to skip header we simply do next(csv_reader)
                # but since column names can be useful for us
                # later on, we take advatnage of this and also
                # save the header!
                self.column_names = next(csv_reader)
                # read each record into our dictionary
                # each record(line) is a list containing all columns
                for i, line in enumerate(csv_reader):
                    self.dataset[i] = line
        else:
            self.image_folder = os.path.join(self.image_folder, 'test')
            for i, img_path in enumerate(os.listdir(self.image_folder)):
                self.dataset[i] = img_path

    def _format_input(self, input_str, one_hot=False):
        one_hot_tensor = torch.tensor([float(i) for i in input_str])
        if one_hot: 
            return one_hot_tensor 
        if one_hot_tensor.size(0) > 1 : 
            return torch.argmax(one_hot_tensor)
        else:
            return one_hot_tensor[0].int()
        
    # lets create the corsponding labels for each category
    def _parse_labels(self, input_str):
        # white,red,green,black,blue,purple,gold,silver
        colors = self._format_input(input_str[4:12], True)            
        # gender_Female,gender_Male
        genders = self._format_input(input_str[12:14])        
        # region_Asia, region_Egypt, region_Europe, region_Middle East  
        regions = self._format_input(input_str[14:18])        
        # fighting_type_magic, fighting_type_melee, fighting_type_ranged
        fighting_styles = self._format_input(input_str[18:21])          
        # alignment_CE, alignment_CG, alignment_CN, alignment_LE,
        # alignment_LG, alignment_LN, alignment_NE, alignment_NG, alignment_TN
        alignments = self._format_input(input_str[21:])  
        return colors, genders, regions, fighting_styles, alignments


    # in getitem, we retrieve one item based on the input index
    # thats why we used a ditionary to make it easier to fectch
    # images
    def __getitem__(self, index):
        if self.is_training_set:
            # we can access each category using a its corrosponding index
            # each record is simply a list and therefore accessing is trivial
            img_path = self.dataset[index][1]
            # to get labels in proper form, we use a helper method here
            labels = self._parse_labels(self.dataset[index])
        else:
            img_path = self.dataset[index]
            labels = -1
        # image files must be read as bytes so we use 'rb' instead of simply 'r' 
        # which is used for text files
        with open(os.path.join(self.image_folder, img_path), 'rb') as img_file:
            # since our datasets include png images, we need to make sure
            # we read only 3 channels and not more!
            img = Image.open(img_file).convert('RGB')
            # apply the transformations 
            img = self.transforms(img)
            return img, labels

    def __len__(self):
        return len(self.dataset)

    def Label_names(self):
        #remove the _in names (i.e gender_male becomes male)
        self.column_names = [name.split('_')[-1] if '_' in name else name\
                            for name in self.column_names ]
        # white,red,green,black,blue,purple,gold,silver
        color_names = self.column_names[4:12]
        # gender_Female,gender_Male
        gender_names = self.column_names[12:14]
        # region_Asia, region_Egypt, region_Europe, region_Middle East  
        region_names = self.column_names[14:18]        
        # fighting_type_magic, fighting_type_melee, fighting_type_ranged
        fighting_names = self.column_names[18:21]          
        # alignment_CE, alignment_CG, alignment_CN, alignment_LE,
        # alignment_LG, alignment_LN, alignment_NE, alignment_NG, alignment_TN
        alignment_names = self.column_names[21:]  
        return (color_names, gender_names, region_names, fighting_names, alignment_names)




# these are the imagenet data-augmentations done when training on imagenet dataset
transforms_train = transforms.Compose([transforms.Resize((256,256)),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])
                                      ])
anime_dataset = AnimeMTLDataset(image_folder = 'mtl_dataset',
                                csv_file_path = r'mtl_dataset\fgo_multiclass_labels.csv',
                                transformations=transforms_train)
#%%
# lets test our dataset class and see if it works ok: 
# but before that lets create some utility functions for 
# displaying our images
#unnormalize
def unnormalize(img):
    img = img.cpu().detach().numpy().transpose(1,2,0)
    return img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406] 

def show_imgs( imgs, rows=3, cols = 11):
    fig = plt.figure(figsize=(cols,rows))
    for i in range(imgs.size(0)):
        ax = fig.add_subplot(rows, cols, i+1, xticks=[], yticks=[])
        img = unnormalize(imgs[i])
        ax.imshow(img)
    plt.show()

#training: 
print('dataset size: {}'.format(len(anime_dataset)))
img, labels = anime_dataset[0]
plt.imshow(unnormalize(img))


#%%
transforms_val = transforms.Compose([transforms.Resize((224,224)),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])
                                      ])

anime_dataset_test = AnimeMTLDataset(image_folder = 'mtl_dataset',
                                csv_file_path = r'mtl_dataset\fgo_multiclass_labels.csv',
                                transformations=transforms_val, 
                                is_training_set =False)

print('Test dataset test : ')
print('dataset size: {}'.format(len(anime_dataset_test)))
img, _ = anime_dataset_test[3]
plt.imshow(unnormalize(img))
#%%
# now lets create a dataloader and carry on!
# lets create a validation and training set as well
import numpy as np
import torch.utils.data as data


samples_count = len(anime_dataset)
all_samples_indexes = list(range(samples_count))
np.random.shuffle(all_samples_indexes)

val_ratio = 0.2
val_end = int(samples_count * 0.2)
val_indexes = all_samples_indexes[0:val_end]
train_indexes = all_samples_indexes[val_end:]
assert len(val_indexes) + len(train_indexes) == samples_count , 'the split is not valid' 

sampler_train = data.SubsetRandomSampler(train_indexes)
sampler_val = data.SubsetRandomSampler(val_indexes)

# always start with 0 workers to be able to easily catch the errors
# in your code, when you solved all issues, you can increase this number for 
# a better and more efficient IO
num_workers=0
dataloader_train = data.DataLoader(anime_dataset, batch_size = 32, sampler = sampler_train, num_workers=num_workers)
dataloader_val = data.DataLoader(anime_dataset, batch_size = 32, sampler = sampler_val, num_workers=num_workers)

dataloader_test = data.DataLoader(anime_dataset_test, batch_size = 32, num_workers=num_workers)
# test 
print('training samples test')
imgs, labels = next(iter(dataloader_train))
show_imgs(imgs)

print('test samples test')
imgs, _ = next(iter(dataloader_test))
show_imgs(imgs)
# test dataloader
#%%
# Now lets create our architecture. 
# we will be using a pretrained model but since we need to add several classification heads
# we will create a new class and carry on.
# when we want to create a new class, we have two options, we can inherit from the architecture
# that we want to use as pretrained model, and rewrite the forward method the way we like. 
# or create a new class, instantiate an object from
# the class we want and use any part from that. 
# this is the first way 

# from torchvision.models.resnet import ResNet, BasicBlock
# class CustomResNet18_MultiTaskNet(ResNet):
#     def __init__(self):
#         super().__init__(BasicBlock, [2, 2, 2, 2])
#         #define  the layers as we want 

#     def forward(self, x):
#         # write the custom forward as we like
#         x = self.conv1(x)
#         # ....
#         return 

# our second method is nearly the same, that is what ever we are doing here
# we can do in method 1, with a slight difference. lets see how 
# we can actually do this using the second way
class Resnet18_multiTaskNet(nn.Module):
    def __init__(self, pretrained=True, frozen_feature_layers = False):
        super().__init__()
        
        resnet18 = models.resnet18(pretrained=pretrained)
        self.is_frozen = frozen_feature_layers
        # here we get all the modules(layers) before the fc layer at the end
        # note that currently at pytorch 1.0 the named_children() is not supported
        # and using that instead of children() will fail with an error
        self.features = nn.ModuleList(resnet18.children())[:-1]
        # this is needed because, nn.ModuleList doesnt implement forward()
        # so you cant do sth like self.features(images). therefore we use 
        # nn.Sequential and since sequential doesnt accept lists, we 
        # unpack all items and send them like this
        self.features = nn.Sequential(*self.features)

        if frozen_feature_layers:
            self.freeze_feature_layers()

        # now lets add our new layers 
        in_features = resnet18.fc.in_features
        # it helps with performance. you can play with it
        # create more layers, play/experiment with them. 
        self.fc0 = nn.Linear(in_features, 512)
        self.bn_pu = nn.BatchNorm1d(in_features, eps = 1e-5)
        # our five new heads for 5 tasks we have at hand!
        self.fc_color = nn.Linear(in_features, 8) 
        self.fc_gender = nn.Linear(in_features, 2) 
        self.fc_region = nn.Linear(in_features, 4) 
        self.fc_fighting = nn.Linear(in_features, 3)
        self.fc_alignment = nn.Linear(in_features, 9)

        # initialize all fc layers to xavier
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight, gain = 1)


    def forward(self, input_imgs):
        output = self.features(input_imgs)
        output = output.view(input_imgs.size(0), -1)
        output = self.bn_pu(F.relu(self.fc0(output)))
        # since color is multi label we should use sigmoid
        # but since we want a numerical stable one, we use
        # nn.BCEWithLogitsloss, as a loss which itself applies sigmoid
        # and thus accepts logits. so we wont use sigmoid here for that matter
        # its much stabler than sigmoid+BCE
        prd_color = self.fc_color(output)
        prd_gender = self.fc_gender(output)
        prd_region = self.fc_region(output)
        prd_fighting = self.fc_fighting(output)
        prd_alingment = self.fc_alignment(output)
        
        return prd_color, prd_gender, prd_region, prd_fighting, prd_alingment
    
    def _set_freeze_(self, status):
        for p in self.features:
            p.requires_grad = status

    def freeze_feature_layers(self):
        self._set_freeze_(True)

    def unfreeze_feature_layers(self):
        self._set_freeze_(False)


model = Resnet18_multiTaskNet(True, True)
print(model)


#%%
# now lets train our model 
# we can have different optimizers for each head or a single one for the whole model
# also if we want to unfreeze all layers, we need to have a different learing rate for features part
# and a different one for heads as they have random weights in the beginning. 
# we will see this both
# we need 5 losses, but since 4 out of 5 task use crossentropy we can use one for all of them
# except the color!
# for color is a multilabel problem and BCEWithlogit is numerically more stable than plain BCE+sigmoid
# so we use BCEWithLogitsLoss
criterion_1 = nn.BCEWithLogitsLoss()
# for gender, region, fighting, and alignment
criterion_2 = nn.CrossEntropyLoss()

#%%
def train_val(model, dataloader, optimizer, criterion_1, criterion_2, is_training, device, topk, interval):

    batch_cnt = len(dataloader)
    fields = ['color', 'gender', 'region', 'fighting', 'alignment']
    # this simply means create a list with len(fields) rooms.
    # it will create a list of 5 empty room. (ie. = [0.0, 0.0, 0.0, 0.0, 0.0])
    accuracies = [0.0]*len(fields)
    status = 'Training' if is_training else 'validation'

    # using set_grad_enabled() we can enable or disable
    # the gardient accumulation and calculation, this is specially
    # good for conserving more memory at validation time and higher performance
    with torch.set_grad_enabled(is_training):    
        
        model.train() if is_training else model.eval()
        
        for i, (imgs, labels) in enumerate(dataloader):
            imgs = imgs.to(device)
            labels = [lbl.to(device) for lbl in labels]
            (lbl_clr, lbl_gdr, lbl_rgn, lbl_ftn, lbl_algn) = labels 
            
            preds = model(imgs)
            (prd_clr, prd_gdr, prd_rgn, prd_ftn, prd_algn) = preds
            
            loss_c = criterion_1(prd_clr, lbl_clr)
            loss_gdr = criterion_2(prd_gdr, lbl_gdr)
            loss_rgn = criterion_2(prd_rgn, lbl_rgn)
            loss_ftn = criterion_2(prd_ftn, lbl_ftn)
            loss_algn = criterion_2(prd_algn, lbl_algn)
            
            loss_final = loss_c + loss_gdr + loss_rgn + loss_ftn + loss_algn
            # accuracies 
            _, indxs_gdr = prd_gdr.topk(topk,dim=1)
            _, indxs_rgn = prd_gdr.topk(topk,dim=1)
            _, indxs_ftn = prd_gdr.topk(topk,dim=1)
            _, indxs_algn = prd_gdr.topk(topk,dim=1)

            # for a multilabel problem there are different ways to calculate the accuracy
            # and other metrics. usually hamming loss is used, here we opted for a simplistic
            # method. I probably explain this in more detail in the multilabel classification
            # tutorial later on. 
            accuracies[0] += torch.mean((torch.round(prd_clr) == lbl_clr).float())
            accuracies[1] += torch.mean((indxs_gdr.view(*lbl_gdr.shape) == lbl_gdr).float())
            accuracies[2] += torch.mean((indxs_rgn.view(*lbl_rgn.shape) == lbl_rgn).float())
            accuracies[3] += torch.mean((indxs_ftn.view(*lbl_ftn.shape) == lbl_ftn).float())
            accuracies[4] += torch.mean((indxs_algn.view(*lbl_algn.shape) == lbl_algn).float())

            if is_training:
                optimizer.zero_grad()
                loss_final.backward() 
                optimizer.step()

        if i%interval==0:
            accs = [acc/batch_cnt for acc in accuracies]
            print(f'[{status}] iter: {i} loss: {loss_final.item():6f}')
            print (' ,'.join(list(f'{f}: {x:.4f}' for f, x in zip(fields, accs))))

 

def train_loop(model, epochs, dataloader_train, dataloader_val,
               optimizer, lr_scheduler, criterion_1, criterion_2, interval=10):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    for e in range(epochs):
        lrs = [f'{lr:.6f}' for lr in lr_scheduler.get_lr()]
        print(f'epoch {e} : lrs : {" ".join(lrs)}')
        train_val(model, dataloader_train, optimizer, criterion_1, criterion_2, True, device, 1, interval)
        train_val(model, dataloader_val, optimizer, criterion_1, criterion_2, False, device, 1, 1)
        lr_scheduler.step()

#%%
model = Resnet18_multiTaskNet(True)

epochs = 10
lr = 0.0001

# Previously we said that we have different options when it 
# comes to finetuning a pretrained model on a new dataset,
# we can freeze the feature part(i.e the pretrained layers),
# train new layers and then unfreeze all layers and train again.
# we can also chose not to freeze any layer, and instead
# use different learning rates for different layers. for example
# a much lower learning rate for pretrained layers and a much higher
# one for new layers.
# inPytorch Optimizers also support specifying per-parameter options.
# which allows us to do exactly this.(specifying different parameeters
# for different sections/layers of our network)
# To do this, instead of passing an iterable of Variable s,
# we pass in an iterable of dict s. Each of them will define a
# separate parameter group, and should contain a params key,
# containing a list of parameters belonging to it. Other keys
# should match the keyword arguments accepted by the optimizers,
# and will be used as optimization options for this group.
#TLDR 
# 
# we can specify different parameters in a list, but since we want
# to have different learning rates for each layer separately, we use
# a parameter group which basically is a dictionary for each layers parameters
# that can contain different options(lr, weight_decay, etc).
# we can still pass options as keyword arguments. 
# They will be used as defaults, in the groups that didn’t override them.
# This is useful when we only want to vary a single option, 
# while keeping all others consistent between parameter groups.
# like this: 
# optimizer = torch.optim.Adam(
#     # our first parameter group specifies our resnet part parameters
#     # note that we used key named 'params' as we are sending parameters
#     [{"params":model.features.parameters()},
#     # while our second parameter group, also specifies a learning rate 
#     # which means, I am overriding the default learning rate here
#      {"params":model.fc_color.parameters(), "lr": 0.1},
#      {"params":model.fc_gender.parameters(), "lr": 0.1},
#      {"params":model.fc_region.parameters(), "lr": 0.1},
#      {"params":model.fc_fighting.parameters(), "lr": 0.1},
#      {"params":model.fc_alignment.parameters(), "lr": 0.1},
#     ],
#     # we can still specify our options as keyword arguments  
#     # to be used as defaults. for example, this learning rate
#     # will be used for parameter groups that didn't specify the lr
#     # keyword, such as the first one here, they will use 
#     # the defaults that we specify here
#     lr=lrmain)

# we could also do this as well using the add_param_group
# This can be useful when fine tuning a pre-trained network
# as frozen layers can be made trainable and added to the Optimizer
# as training progresses.
# first we can specify the defaults 
optimizer = torch.optim.SGD(model.features.parameters(), lr = lr)
# and then add the needed parameter groups
# optimizer.add_param_group({"params": model.fc0.parameters(), "lr": lrlast})
# optimizer.add_param_group({"params": model.fc1.parameters(), "lr": lrlast})
optimizer.add_param_group({"params": model.fc_color.parameters(), "lr": 0.1})
optimizer.add_param_group({"params": model.fc_gender.parameters(), "lr": 0.1})
optimizer.add_param_group({"params": model.fc_region.parameters(), "lr": 0.1})
optimizer.add_param_group({"params": model.fc_fighting.parameters(), "lr": 0.1})
optimizer.add_param_group({"params": model.fc_alignment.parameters(), "lr": 0.1})
# lets decay/decrease the learning rate each 10 epochs!
# you can experiment with different schedulers, and you 
# are suggested to do so to learn more. I just chose the 
# simplest possible for  the sake of simplicity!
lrsched = torch.optim.lr_scheduler.StepLR(optimizer, 10)
train_loop(model, epochs, dataloader_train, dataloader_val, optimizer, lrsched, criterion_1, criterion_2, 5)
#%%
# or you can freeze the net, train for some epoch, unfreeze and retrain
# resetting the learning rates to their default values
# for p in optimizer.param_groups:
#     p['lr'] = 0.1
# optimizer.param_groups[0]['lr'] = 0.1

# # #%%
# model.unfreeze_feature_layers()
# train_loop(model, epochs, dataloader_train, dataloader_val, optimizer, lrsched, criterion_1, criterion_2, 5)


#%%
# now lets see how it performs on test set 
# lets create functions that show the predictions better!
# the best way for you to understand whats going on here
# (in case you dont know, ) is to use debugging and step in
# one line at a time and view the values. here I'm converting 
# the array values/label values into their corrosponding names
# for example gender[0 1] will be male, and like that!
def parse_predictions(names, preds):
    lst_names = []
    
    (colors, genders,regions, fightings, alignments) = names
    (clr_prd, gdr_prd, rgn_prd, ftn_prd, aln_prd) = preds
    # color names
    colornames = torch.round(clr_prd)
    for i in range(colornames.size(0)):
        clr = ' '.join([name for name, idx in zip(colors, colornames[i]) if idx ==1])
        gdr = genders[torch.argmax(gdr_prd[i]).item()]
        rgn = regions[torch.argmax(rgn_prd[i]).item()]
        ftn = fightings[torch.argmax(ftn_prd[i]).item()]
        aln = alignments[torch.argmax(aln_prd[i]).item()]
        lst_names.append((clr,gdr, rgn, ftn,aln))
    return lst_names

def show_predictions(imgs, preds, rows=32, cols=1):
    fig = plt.figure(figsize=(224,224))
    plt.subplots_adjust(hspace=0.2)
    preds_name = parse_predictions(anime_dataset.Label_names() ,preds)
    for i, (img, preds) in enumerate(zip(imgs, preds_name)):
        img = unnormalize(img)
        ax = fig.add_subplot(rows, cols, i+1,xticks=[], yticks=[])
        
        ax.imshow(img)
        (clr, gdr, rgn, ftn, aln )= preds
        str_info = f'color: {clr}\ngender: {gdr} region: {rgn} ' \
                   f'fighting style : {ftn} alignment: {aln}'
        ax.set_title(str_info)
        
        # print(f'color: {clr}')
        # print(f'gender: {gdr}')
        # print(f'region: {rgn}')
        # print(f'fighting style: {ftn}')
        # print(f'alignment : {aln}')
        #plt.show()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
for imgs, _ in dataloader_test:
    imgs = imgs.to(device)
    model.eval()
    preds = model(imgs)
    # print(preds)
    show_predictions(imgs, preds)
    

#%%
