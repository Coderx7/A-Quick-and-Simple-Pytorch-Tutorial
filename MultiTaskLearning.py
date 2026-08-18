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
# from sklearn import metrics
%matplotlib inline

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
                                       # note the scale, by default its (0.08,1) which may very well
                                       # hinder the leaning as it will use patches unrelaetd to the
                                       # actual concept in the sample and therefore you get a lot of
                                       # mistmatches and ultimately a poor performance (imagine you want to 
                                       # look at one's face and you instead only take a piece of their clothes!
                                       # play wih this value to understand this visually)
                                      transforms.RandomResizedCrop(224,scale=(0.8,1)),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])
                                      ])
mtl_dataset = '/media/hossein/SSD1/A-Quick-and-Simple-Pytorch-Tutorial/data/fgo_multitask_dataset'
anime_dataset = AnimeMTLDataset(image_folder = mtl_dataset,
                                csv_file_path = f'{mtl_dataset}/fgo_multiclass_labels.csv',
                                transformations=transforms_train)
#%%
# lets test our dataset class and see if it works ok: 
# but before that lets create some utility functions for 
# displaying our images
#unnormalize
def unnormalize(img):
    img = img.cpu().detach().numpy().transpose(1,2,0)
    img_normalized = img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
    return img_normalized.clip(0,1)

def show_imgs( imgs, rows=3, cols = 11):
    fig = plt.figure(figsize=(cols,rows))
    for i in range(imgs.size(0)):
        ax = fig.add_subplot(rows, cols, i+1, xticks=[], yticks=[])
        img = unnormalize(imgs[i])
        ax.imshow(img)
    plt.show()

#training: 
print('dataset size: {}'.format(len(anime_dataset)))
img, labels = anime_dataset[10]
plt.imshow(unnormalize(img))


#%%
transforms_val = transforms.Compose([transforms.Resize((224,224)),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])
                                      ])

anime_dataset_test = AnimeMTLDataset(image_folder = mtl_dataset,
                                csv_file_path = f'{mtl_dataset}/fgo_multiclass_labels.csv',
                                transformations=transforms_val, 
                                is_training_set =False)

print('Test dataset test : ')
print('dataset size: {}'.format(len(anime_dataset_test)))
img, _ = anime_dataset_test[20]
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
        for n,p in self.features.named_parameters():
            p.requires_grad = status
        # for m in self.features.children():
        #     for p in m.parameters():
        #         p.requires_grad=status    


    def freeze_feature_layers(self):
        self._set_freeze_(False)

    def unfreeze_feature_layers(self):
        self._set_freeze_(True)


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
    fields = ['gender', 'region', 'fighting', 'alignment','color']
    colors_names =['white','red','green','black','blue','purple','gold','silver']
    # this simply means create a list with len(fields) rooms.
    # it will create a list of 5 empty room. (ie. = [0.0, 0.0, 0.0, 0.0, 0.0])
    accuracies = [0.0]*len(fields)
    status = 'Training' if is_training else 'validation'
    per_label_acc_avg=0
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

            accuracies[0] += torch.mean((indxs_gdr.view(*lbl_gdr.shape) == lbl_gdr).float())
            accuracies[1] += torch.mean((indxs_rgn.view(*lbl_rgn.shape) == lbl_rgn).float())
            accuracies[2] += torch.mean((indxs_ftn.view(*lbl_ftn.shape) == lbl_ftn).float())
            accuracies[3] += torch.mean((indxs_algn.view(*lbl_algn.shape) == lbl_algn).float())
            # for a multilabel problem there are different ways to calculate the accuracy
            # and other metrics. there are usually two methods that are used often. 
            # subset accuracy and perlabel accuracy. 
            # subset accuracy means all labels for a sample must match exactly and
            # per-label accuracy means the accuracy is measured label-wise. 
            # the subset accuracy where absolutely all labels must match is used for critical
            # applications where we wantto make sure everything is accurate and hence can be 
            # much more challanging than the per-label accuracy. 
            # here we opted for a simplistic method which is per label accuracy.
            # I probably explain this in more detail in the multilabel classification
            # tutorial later on. 
            # 
            # sidenote: 
            # torch.mean((torch.round(prd_clr.sigmoid()) == lbl_clr).float()) is not the per-label-accuracy
            # its a global accuracy, that is, it flattens the tensor and takes its mean which
            # is equivalent to calculating the per-label-accuracy and then take the average 
            # of all the accuracies for a sample (basically turning a tensor of accuracies into
            # a single accuracy thats the average of them for easier understanding of how the whole
            # thing is doing generally (grossly, it doesnt give an accurate or detailed overview though!)).
            # while it can give us a rough estimate on how well the model is doing in each batch,
            # it doesnt tell us much on how well our model is doing for each label specifically! 
            color_predictions = torch.round(prd_clr.sigmoid())
            accuracies[4] += torch.mean((color_predictions == lbl_clr).float())
            # note the .all() where we indicate all predictions and their target labels
            # for each sample need to be true for that sample lable to pass as a match
            subset_acc = torch.mean((color_predictions == lbl_clr).all(dim=1).float())
            # Per-label accuracy
            per_label_acc = torch.mean((color_predictions == lbl_clr).float(), dim=0)
            # the same as accuracies[4] 
            per_label_acc_avg += torch.mean(per_label_acc) 

            if is_training:
                optimizer.zero_grad()
                loss_final.backward() 
                optimizer.step()

        if i%interval==0:
            accs = [acc/batch_cnt for acc in accuracies]
            print(f'[{status}] iter: {i} loss: {loss_final.item():6f}')
            print(f'Accuracies:')
            print ('\n'.join(list(f' --{f:<15}: {x*100:.2f}%' for f, x in zip(fields, accs))))
            print(f'{'per-label-acc-avg':<15}: {(per_label_acc_avg.item()/batch_cnt)*100:.2f}')
            print(f'Per-Label Accuracies:')
            print('\n'.join([f'  --{color:<7}: {acc*100:.2f}%' for color, acc in zip(colors_names, per_label_acc.tolist())]))
            
 

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

epochs = 20#10
lr = 0.001

optimizer = torch.optim.Adam(model.features.parameters(), lr = lr,weight_decay=1e-5)
# and then add the needed parameter groups
# optimizer.add_param_group({"params": model.fc0.parameters(), "lr": lrlast})
# optimizer.add_param_group({"params": model.fc1.parameters(), "lr": lrlast})
optimizer.add_param_group({"params": model.fc_color.parameters(), "lr": 0.002})
optimizer.add_param_group({"params": model.fc_gender.parameters(), "lr": 0.01})
optimizer.add_param_group({"params": model.fc_region.parameters(), "lr": 0.001})
optimizer.add_param_group({"params": model.fc_fighting.parameters(), "lr": 0.001})
optimizer.add_param_group({"params": model.fc_alignment.parameters(), "lr": 0.0001})
# lets decay/decrease the learning rate each 10 epochs!
# you can experiment with different schedulers, and you 
# are suggested to do so to learn more. I just chose the 
# simplest possible for  the sake of simplicity!
lrsched = torch.optim.lr_scheduler.MultiStepLR(optimizer, [5,10,13,18,20])
train_loop(model, epochs, dataloader_train, dataloader_val, optimizer, lrsched, criterion_1, criterion_2, 5)
# which made us achieve, the following results, which we can improve with better
# training regime and model improvements, and cleaning the dataset!
# epoch 18 : lrs : 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000
# [validation] iter: 2 loss: 2.629042
# Accuracies:
#  --gender         : 75.35%
#  --region         : 6.25%
#  --fighting       : 36.11%
#  --alignment      : 8.33%
#  --color          : 73.48%
# per-label-acc-avg: 73.48
# Per-Label Accuracies:
#   --white  : 33.33%
#   --red    : 100.00%
#   --green  : 100.00%
#   --black  : 100.00%
#   --blue   : 66.67%
#   --purple : 66.67%
#   --gold   : 66.67%
#   --silver : 33.33%
# epoch 19 : lrs : 0.000000 0.000000 0.000001 0.000000 0.000000 0.000000
# [validation] iter: 2 loss: 2.116157
# Accuracies:
#  --gender         : 87.50%
#  --region         : 28.47%
#  --fighting       : 23.96%
#  --alignment      : 5.21%
#  --color          : 79.17%
# per-label-acc-avg: 79.17
# Per-Label Accuracies:
#   --white  : 100.00%
#   --red    : 100.00%
#   --green  : 66.67%
#   --black  : 100.00%
#   --blue   : 100.00%
#   --purple : 100.00%
#   --gold   : 100.00%
#   --silver : 33.33%
#%%
torch.save(model.state_dict(),'./weights/mtl_animefighting.pt')
#%%
model.load_state_dict(torch.load('./weights/mtl_animefighting.pt'))
#%%
# or you can freeze the net, train for some epoch, unfreeze and retrain
# resetting the learning rates to their default values
# note lr=0.1 will lead to the loss explosion! so we use a much lower lr!
for p in optimizer.param_groups:
    p['lr'] = 1e-5
optimizer.param_groups[0]['lr'] = 1e-5

# # #%%
model.unfreeze_feature_layers()
train_loop(model, epochs, dataloader_train, dataloader_val, optimizer, lrsched, criterion_1, criterion_2, 5)


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
    # we didnt use sigmoid on our fc_color, so the logits are unbounded
    # therefore, we use sigmoid here to make the values be in 0-1 range
    # and then round the color predictions so we get either 0 or 1 for each color
    color_preds = torch.round(clr_prd.sigmoid())
    # print(f'{color_preds.shape=}') # shape: (32x8)
    # print(f'{color_preds.size(0)=}') # shape:32
    # print(f'{colors=}')#8 colors
    for i in range(color_preds.size(0)):
        # print(f'{color_preds[i].cpu().detach().numpy()}')
        # look at all the color predictions for the current sample and grab their 
        # name only if their predicted value (label) is 1 implying the color is
        # present in the image
        clr = ' '.join([color for color, label in zip(colors, color_preds[i]) if label==1])
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
#%% 
# In the name of God the most compassionate the most merciful
# sidenote:
# this is a second version I wrote a few years later 
# with a new dataset and a bit more explanation. I left the original tutorial
# until the new one covers all the points. go on ahead and read this aswell!
# its the same tutorial but with a new dataset and a bit of new information
# i'll merge the two later probably!)

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
# from sklearn import metrics
%matplotlib inline

# OK, I had to create a dataset myself. I used https://www.animecharactersdatabase.com/ to 
# create a simple dataset of anime characters with different attributes.
# you can download the dataset from here https://github.com/Coderx7/tiny_anime_hair_outfit_multiclass_multilabel_dataset
# we are going to classify each image into several categories, for example, 
# we want to know the gender, adulthood status, hair colors, stuff like that.
# our dataset contains 949 images. theres no separate validation/test set, so we will have to
# use torch.utils.data.SubsetRandomSampler() class to make up for that!

# but before that, lets see how our dataset looks like! that is lets have a look at our .csv file!
# that hosts our labels and path to our images
# This is how it looks (animelist.csv):
# 
# ID, FileName, Gender, Adulthood,  Hair_Length,    Hair_Color, Outfit_Colors
# 2,  0001_female_teen_short_yellow_blue_white_red_yellow_black_purple.png, female, teen, short, yellow, blue,white,red,yellow,black,purple
# 3,  0002_female_teen_short_pink_green_cream.jpg,  female, teen,  short, pink, green,cream
# 4,  0003_female_teen_short_white_red_white_gray_black_brown.jpg, female, teen, short, white, red,white,gray,black,brown
# 5,  0004_female_teen_short_black_red_white_gray_purple.png,   female, teen,   short, black, red,white,gray,purple
# ...
# it has a header that specifies different columns role. 
# looking at our data we see, we have several categories : hair color , genders, adulthood, and outfit colors,
# etc among these categories, only outfit colors can have more than 1 value (that is their values are not mutually
# exclusive. we can have both black ,blue gold and white at the same time) so outfit color is multilabel. 
# for a normal single label classfication, we use crossentropy and in Pytorch, we simply use the index of 
# the correct class and do not feed the one hot encoded representation of the true class. 
# for a multi label case, we use BCE (BinaryCrossEntropy) and use the one hot encoded representation of 
# labels. 
# when building our dataset class, we need to provide labels in the proper form as well. 
# so lets get busy!

# for working with path, files and folders
import os
# to read csv file we use csv module but we can
# also use pandas, but for this specific example
# its overkill so we stick to csv to keep it simple
import csv
# import pandas as pd

# We use PIL.Image for reading an image! 
# note that pytorch now also offers facilities
# for reading images internally using torchvision module torchvision.io)
# but it returns tensors which we dont need rightnow!
import PIL.Image as Image
# from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import MultiLabelBinarizer

class AnimeMTLDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file_path, transformations) :
        super().__init__()
        # to get the dirname we could also use Pathlib.Path,
        # but os.path.dirname does just fine aswell.
        self.image_folder = os.path.dirname(csv_file_path)
        self.transforms = transformations
        self.column_names = None
        
        # we can read csv files in several ways, 
        # lets do it using the simplest form which is 
        # using the python's built-in module csv!
        with open(csv_file_path,'r') as f:
            csv_reader = csv.reader(f)
            # grab the header and use it as column names
            self.column_names = next(csv_reader)
            # now we keep reading the remainig rows
            # before that we need a dictionary to store the images/labels
            self.dataset = {}
            # we also want to grab colors and replace each color
            # with the corrosponding index, but it requires a separate
            # loop over all the rows. we can simply use the hardcoded
            # list of them and save ourselves some time!
            self.colors_list={'white': 0, 'black': 1, 'brown': 2, 'blue': 3, 'red': 4, 'yellow': 5,
                    'gray': 6, 'green': 7, 'purple': 8, 'pink': 9, 'cream': 10, 'orange': 11}
            # likewise we need to do this for all string based values for our classes
            self.gender = {'male':0,'female':1}
            self.adulthood = {'teen':0,'adult':1}
            self.length = {'short':0,'long':1}
            for i,row in enumerate(csv_reader):
                # now we simply transform the labels 
                img_file_name = row[1].strip()
                gender_label = self.gender[row[2].strip()]
                adulthood_label = self.adulthood[row[3].strip()]
                length_label = self.length[row[4].strip()]
                haircolor_label = self.colors_list[row[5].strip()]
                # note that our haircolor is a single value just like gender/length/adulthoold
                # however, this is not the case for outfitcolors, as there can be any number of
                # colors, so we instead need to have a onehot encoded kind of array that shows
                # what colors exist for each sample.
                # since we have more than one color, and they are separated with comma, we grab
                # all of them in a list and check which ones exist for a given sample.
                # this will give us a list of true/false values which we can later use to form
                # our onehot encoded like array for labels for the outfitcolors!
                current_outfit_colors = row[6:]
                outfit_colors_encoded = [int(outfit_color in current_outfit_colors) for outfit_color in self.colors_list]
                # print(f'{img_file_name} | {current_outfit_colors=} | {outfit_colors_encoded=}')
                # now we fill our data source with the proper information!
                outfit_colors_encoded = torch.tensor(outfit_colors_encoded)
                self.dataset[i] = [row[0], img_file_name, gender_label, adulthood_label,length_label,haircolor_label,outfit_colors_encoded]
                
     # in getitem, we retrieve one item based on the input index
    # thats why we used a ditionary to make it easier to fectch
    # images
    def __getitem__(self, index):
        # we can access each category using a its corrosponding index
        # each record is simply a list and therefore accessing is trivial
        img_path = self.dataset[index][1]
        # to get labels in proper form, we use a helper method here
        labels = self.dataset[index][2:]
        
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

# these are the imagenet data-augmentations done when training on imagenet dataset
transforms_train = transforms.Compose([transforms.Resize((256,256)),
                                      transforms.RandomResizedCrop(224,scale=(0.8,1)),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])
                                      ])

transforms_val = transforms.Compose([transforms.Resize((224,224)),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])
                                      ])


# /media/hossein/SSD1/A-Quick-and-Simple-Pytorch-Tutorial/data/anime_characters/anime_dataset
mtl_dataset = '/media/hossein/SSD1/A-Quick-and-Simple-Pytorch-Tutorial/data/anime_characters/anime_dataset'
anime_dataset = AnimeMTLDataset(csv_file_path = f'{mtl_dataset}/animelist.csv',
                                transformations=transforms_train)

#%%
# lets test our dataset class and see if it works ok:
# but before that lets create some utility functions for
# displaying our images
#unnormalize
def unnormalize(img):
    img = img.cpu().detach().numpy().transpose(1,2,0)
    img_normalized = img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
    return img_normalized.clip(0,1)

def show_imgs( imgs, rows=3, cols = 11):
    fig = plt.figure(figsize=(cols,rows))
    for i in range(imgs.size(0)):
        ax = fig.add_subplot(rows, cols, i+1, xticks=[], yticks=[])
        img = unnormalize(imgs[i])
        ax.imshow(img)
    plt.show()

#training:
print('dataset size: {}'.format(len(anime_dataset)))
img, labels = anime_dataset[125]
print(f'{labels}')
plt.imshow(unnormalize(img))

#%%
# now lets create a dataloader and carry on!
# lets create a validation and training set as well
import numpy as np
import torch.utils.data as data

# lets split dataset into train, validation, and test sets
samples_count = len(anime_dataset)
all_samples_indexes = list(range(samples_count))
np.random.shuffle(all_samples_indexes)

val_ratio = 0.2
test_ratio = 0.1
val_end = int(samples_count * val_ratio)
test_end = int(samples_count * test_ratio) + val_end

val_indexes = all_samples_indexes[0:val_end]
test_indexes = all_samples_indexes[val_end:test_end]
train_indexes = all_samples_indexes[test_end:]

# make sure we didnt mess up! and all splits are valid
assert len(val_indexes) + len(train_indexes) + len(test_indexes) == samples_count, 'The split is not valid'

sampler_train = data.SubsetRandomSampler(train_indexes)
sampler_val = data.SubsetRandomSampler(val_indexes)
sampler_test = data.SubsetRandomSampler(test_indexes)

# always start with 0 workers to be able to easily catch the errors
# in your code, when you solved all issues, you can increase this number for 
# a better and more efficient IO
num_workers=8
dataloader_train = data.DataLoader(anime_dataset, batch_size = 32, sampler = sampler_train, num_workers=num_workers)
dataloader_val = data.DataLoader(anime_dataset, batch_size = 32, sampler = sampler_val, num_workers=num_workers)
dataloader_test = data.DataLoader(anime_dataset, batch_size = 32, sampler = sampler_test, num_workers=num_workers)
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
        self.bn_pu = nn.BatchNorm1d(512, eps = 1e-5)
        # our five new heads for 5 tasks we have at hand!
        # 
        # note that we can buff up our classifiers, like
        # use one or more conv layers.
        # 
        # just a headsup, if you use conv layer, remember
        # they are 3d/4d depending on conv1d/2d (cuz our input is 2d here 
        # following a linear layer) and therefore you need to do the needed
        # reshaping.
        # 
        # reminder:
        # recall that a conv1d layer accepts a 3d input (batch_size, channels, sequence_length)
        # while a conv2d requires a 4d shape (batchsize, channels, height, width)
        # conv1d applies the convolution filters over the sequence_length dimension (spatial/temporal dimension)
        # while the conv2d does that over the height and width dimension.
        # after all conv layers are designed to slide convolutional kernels
        # along some dimensions. 
        # also make sure the input has the right spatial/sequential structure,
        # or else, the convolution operation will fail and we'll get a
        # runtime error stating weights and input dont match (e.g.
        # 'RuntimeError: Given groups=1, weight of size [512, 512, 1], 
        # expected input[1, 2, 512] to have 512 channels, but got 2 channels instead'
        # so reshape the input when feeding them to conv layers
        self.fc_gender = nn.Linear(in_features, 2)
        self.fc_adulthood = nn.Linear(in_features, 2)
        self.fc_length = nn.Linear(in_features, 2)
        self.fc_haircolor = nn.Linear(in_features, 12)
        self.fc_outfitcolors = nn.Linear(in_features, 12)

        # initialize all fc layers with xavier initialization algorithm
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight, gain = 1)

    def forward(self, input_imgs):
        output = self.features(input_imgs)
        # print(f'{output.shape=}')
        output = output.view(input_imgs.size(0), -1)
        # print(f'{output.shape=}')
        output = F.relu(self.bn_pu(self.fc0(output)))
        # print(f'{output.shape=}')
        # since outfitcolor is multi label we should use sigmoid
        # but since we want a numerical stable one, we use
        # nn.BCEWithLogitsloss, as a loss which itself applies sigmoid
        # and thus accepts logits. so we wont use sigmoid here for that matter
        # its much more stable than sigmoid+BCE
        # also we can use BCE for other classes as well 
        # since they are binary classes themselves, and 
        # using a sigmoid and threshold we should be able to
        # determine which one is which during inference time!
        # however, we can also treat them as a multi-class 
        # problem where each have two classes so we can also
        # use crossentropy! thats what we do here
        prd_gender = self.fc_gender(output)
        prd_adulthood = self.fc_adulthood(output)
        prd_length = self.fc_length(output)
        prd_haircolor = self.fc_haircolor(output)
        prd_outfitcolors = self.fc_outfitcolors(output)
        return prd_gender, prd_adulthood, prd_length, prd_haircolor,prd_outfitcolors
    
    def _set_grads(self, is_grad_required):
        for n,p in self.features.named_parameters():
            p.requires_grad = is_grad_required
        # for m in self.features.children():
        #     for p in m.parameters():
        #         p.requires_grad=status    

    def freeze_feature_layers(self):
        self._set_grads(False)

    def unfreeze_feature_layers(self):
        self._set_grads(True)


model = Resnet18_multiTaskNet(True, True)
print(model)
x = model(torch.randn(size=(2,3,224,224)))

#%%
# now lets train our model 
# we can have different optimizers for each head or a single one for the whole model
# also if we want to unfreeze all layers, we need to have a different learing rate for features part
# and a different one for heads as they have random weights in the beginning. 
# we will see this both
# we need 5 losses, but since 4 out of 5 task use crossentropy we can use one for all of them
# except the color!
# for gender, hairlength, haircolor
criterion_1 = nn.CrossEntropyLoss()
# for color is a multilabel problem and BCEWithlogit is numerically more stable than plain BCE+sigmoid
# so we use BCEWithLogitsLoss
criterion_2 = nn.BCEWithLogitsLoss()


#%%
# now before we write the training loop, lets first see how to 
# calculate accuracy for our multilabel part.
# we said before that there are several ways to calculate such metrics.
# here we calculate two such ways which are subset accuracy and 
# per-label accuracy.
# subset accuracy means all labels for a sample must match exactly and
# per-label accuracy means it measures the accuracy label-wise.
# 
# sidenote: 
# Subset Accuracy
# Subset accuracy is a strict metric that checks if the entire set of predicted labels 
# matches the entire set of true labels for a sample. 
# In other words, all the labels must be correctly predicted for a sample to 
# be counted as accurate.
# so the formula will be: 
# Number of samples where all labels are correctly predicted / Total number of samples
# 
# for example if we have three labels, and only two samples out of four samples
# have all 3 labels match exactly with the true labels, then our accuracy will be 2/4 = 0.5,
# will be 50%! and if a single label out of 3 is wrong/misclassified, the whole label for 
# that sample is treated as complely wrong!
# therefore this means the subset accuracy can be very low for a multilabel problem,
# especially if there are many labels, as a single misclassification for any label 
# will make the entire sample be rendered as incorrect.
# However, its useful if we need all predictions to be correct for a task, 
# such as medical diagnosis where all conditions must be correctly identified.

# Per-label Accuracy
# Contrary to the subset accuracy, Per-label accuracy calculates the accuracy for each
# individual label across all samples. It measures how well the model predicts each 
# label independently of others.
# to calculate the accuracy for a label (i) we simply divide its correct labels and divided
# it by number of samples!
# Per-label Accuracy(i) = Number of correct predictions for Label i/Total number of samples
#
# for example using the same example as before, imagine our labels are like this: 
# 2 samples out of 4 samples have all 3 labels correctly predicted and we saw a 50% accuracy
# for subset accuracy approach. now to calculate the accuracy for each label,
# we treat each label (each one of three) as a single label, as if they were separate classes
# this means we will have 3 accuracy: 

# | Sample | True Labels  | Predicted Labels | Label 1  | Label 2  | Label 3  |
# |        |              |                  | correct? | correct? | correct? |
# |--------|--------------|------------------|----------|----------|----------|
# |   1    |  [1, 0, 1]   |    [1, 0, 1]     |   Yes    |  Yes     |   Yes    |
# |   2    |  [0, 1, 0]   |    [0, 1, 1]     |   Yes    |  Yes     |   No     |
# |   3    |  [1, 1, 0]   |    [1, 1, 0]     |   Yes    |  Yes     |   Yes    |
# |   4    |  [0, 0, 1]   |    [0, 0, 0]     |   Yes    |  Yes     |   No     |

# accuracy for Label 1 (all samples predicted correctly)= 4/4 = 1.0 
# accuracy for Label 2 (all samples predicted correctly)= 4/4 = 1.0 
# accuracy for Label 3 (only two samples predicted correctly)= 2/4 = 0.5

# As we can see per-label accuracy allows us to see how well the model performs 
# with respect to each label and it is specifically helful in situations where we
# have imbalanced data in our dataset by identifying which labels the model 
# struggles with the most.
# also this approach gives a better overview of the overall results, and 
# averages across all labels, which is useful for summarizing results.

# 
# so we can use subset accuracy when exact matches for all labels are required
# such as for example a robot's action plan where every step must be correct.
# and in the same vein, we can use per-label accuracy for diagnosing model performance
# on individual labels for example for multi-attribute prediction in images 
# (e.g. age, gender, emotion, etc)

def calculate_accuracy(y_pred, y_true, threshold=0.5):
    y_pred = torch.sigmoid(y_pred) > threshold
    # check the exact matchs (subset accuracy)
    subset_acc = torch.mean((y_pred == y_true).all(dim=1).float())
    # Per-label accuracy
    per_label_acc = torch.mean((y_pred == y_true).float(), dim=0)
    return subset_acc, per_label_acc

def train_val(model, dataloader, optimizer, criterion_1, criterion_2, is_training, device, topk, interval):

    batch_cnt = len(dataloader)
    # used to display in accuracy section for each class
    fields = [ 'gender','adulthood', 'hair length', 'hair color', 'outfit color']
    # used to display each color with its accuracy
    color_names = {idx:name for (name,idx) in anime_dataset.colors_list.items()}
    
    # this simply means create a list with len(fields) rooms.
    # it will create a list of 5 empty rooms (ie. = [0.0, 0.0, 0.0, 0.0, 0.0])
    accuracies = [0.0]*len(fields)
    status = 'Training' if is_training else 'validation'

    # using set_grad_enabled() we can enable or disable
    # the gardient accumulation and calculation, this is specially
    # good for conserving more memory at validation time and higher performance
    with torch.set_grad_enabled(is_training):    
        
        model.train() if is_training else model.eval()
        top_acc = 0
        for i, (imgs, labels) in enumerate(dataloader):
            imgs = imgs.to(device)
            labels = [lbl.to(device).long() for lbl in labels]
            (lbl_gdr, lbl_adlt, lbl_len, lbl_hclr, lbl_oclr) = labels 
            
            preds = model(imgs)
            (prd_gdr, prd_adlt, prd_len, prd_hclr, prd_oclr) = preds
            
            loss_gdr = criterion_1(prd_gdr, lbl_gdr)
            loss_adlt = criterion_1(prd_adlt, lbl_adlt)
            loss_len = criterion_1(prd_len, lbl_len)
            loss_hclr = criterion_1(prd_hclr, lbl_hclr)
            loss_c = criterion_2(prd_oclr, lbl_oclr.float())
            # penalize lossc to getbett
            loss_final = loss_gdr + loss_adlt + loss_len + loss_hclr + (loss_c)
            # accuracies 
            _, indxs_gdr = prd_gdr.topk(topk,dim=1)
            _, indxs_adlt = prd_adlt.topk(topk,dim=1)
            _, indxs_len = prd_len.topk(topk,dim=1)
            _, indxs_hclr = prd_hclr.topk(topk,dim=1)

            # for a multilabel problem there are different ways to calculate the accuracy
            # and other metrics. usually hamming loss is used, here we opted for a simplistic
            # method. I probably explain this in more detail in the multilabel classification
            # tutorial later on. 
            accuracies[0] += torch.mean((indxs_gdr.view(*lbl_gdr.shape) == lbl_gdr).float()).item()
            accuracies[1] += torch.mean((indxs_adlt.view(*lbl_adlt.shape) == lbl_adlt).float()).item()
            accuracies[2] += torch.mean((indxs_len.view(*lbl_len.shape) == lbl_len).float()).item()
            accuracies[3] += torch.mean((indxs_hclr.view(*lbl_hclr.shape) == lbl_hclr).float()).item()
            accuracies[4] += torch.mean((torch.round(prd_oclr.sigmoid()) == lbl_oclr).float()).item()
            
            subset_accuracy, perlabel_accuracy = calculate_accuracy(prd_oclr,lbl_oclr)
            perlabel_accuracy_avg = torch.sum(perlabel_accuracy)/len(perlabel_accuracy)*100
            
            all_accs_avg = np.mean(accuracies) + perlabel_accuracy_avg
            
            if top_acc<all_accs_avg:
                top_acc = all_accs_avg
                torch.save({'weights':model.state_dict(),
                            'all_accs_avg':all_accs_avg,
                            'accuracies':accuracies,
                            'subset_accuracy':subset_accuracy,
                            'perlabel_accuracy':perlabel_accuracy},"./weights/mtl_animestyles.pt")
            
            if is_training:
                optimizer.zero_grad()
                loss_final.backward() 
                optimizer.step()

        if i%interval==0:
            accs = [acc/batch_cnt for acc in accuracies]
            print(f'[{status}] Iter: {i} | Loss: {loss_final.item():6f}')
            print(f'Accuracies:')
            print ('\n'.join(list(f' --{f:<12}: {acc*100:.2f}%' for f, acc in zip(fields, accs))))
            print(f'Outfit colors:')
            print(f' --Sub-set  Acc:      {subset_accuracy.item()*100:.2f}%')
            print(f' --Perlabel Acc[Avg]: {perlabel_accuracy_avg:.2f}')
            print(f' --Perlabel Acc:    \n{'\n'.join([f'    {color_names[i]}:\t{p.item()*100:.2f}%' for i,p in enumerate(perlabel_accuracy)])}')
            

def train_loop(model, epochs, dataloader_train, dataloader_val,
               optimizer, lr_scheduler, criterion_1, criterion_2, interval=10):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    for e in range(epochs):
        lrs = [f'{lr:.6f}' for lr in lr_scheduler.get_lr()]
        print(f'Epoch {e} : lrs : {" ".join(lrs)}')
        train_val(model, dataloader_train, optimizer, criterion_1, criterion_2, True, device, 1, interval)
        train_val(model, dataloader_val, optimizer, criterion_1, criterion_2, False, device, 1, 1)
        lr_scheduler.step()


#%%
model = Resnet18_multiTaskNet(True)

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
epochs = 20
lr = 0.001
optimizer = torch.optim.Adam(model.features.parameters(), lr = lr)
# and then add the needed parameter groups
# optimizer.add_param_group({"params": model.fc0.parameters(), "lr": lrlast})
# optimizer.add_param_group({"params": model.fc1.parameters(), "lr": lrlast})
optimizer.add_param_group({"params": model.fc_gender.parameters(), "lr": 0.1})
optimizer.add_param_group({"params": model.fc_adulthood.parameters(), "lr": 0.1})
optimizer.add_param_group({"params": model.fc_length.parameters(), "lr": 0.1})
optimizer.add_param_group({"params": model.fc_haircolor.parameters(), "lr": 0.1})
optimizer.add_param_group({"params": model.fc_outfitcolors.parameters(), "lr": 0.1})
# lets decay/decrease the learning rate each 10 epochs!
# you can experiment with different schedulers, and you 
# are suggested to do so to learn more. I just chose the 
# simplest possible for  the sake of simplicity!
lrsched = torch.optim.lr_scheduler.MultiStepLR(optimizer, [5,10,13,18,20])
train_loop(model, epochs, dataloader_train, dataloader_val, optimizer, lrsched, criterion_1, criterion_2, 5)
#%%
torch.save(model.state_dict(),'./weights/mtl_animestyles.pt')
#%%
model.load_state_dict(torch.load('./weights/mtl_animestyles.pt'))
#%%
# or you can freeze the net, train for some epoch, unfreeze and retrain
# resetting the learning rates to their default values
# 1e-3 (0.001)
for p in optimizer.param_groups:
    p['lr'] = 1e-3
optimizer.param_groups[0]['lr'] = 1e-3

# # #%%
model.unfreeze_feature_layers()
train_loop(model, 20, dataloader_train, dataloader_val, optimizer, lrsched, criterion_1, criterion_2, 5)
#%%
torch.save(model.state_dict(),'./weights/mtl_animestyles2.pt')
#%%
# now lets see how it performs on test set 
# lets create functions that show the predictions better!
# the best way for you to understand whats going on here
# (in case you dont know, ) is to use debugging and step in
# one line at a time and view the values. here I'm converting 
# the array values/label values into their corrosponding names
# for example gender[0 1] will be male, and like that!
def parse_predictions(dataset:AnimeMTLDataset, preds):
    lst_names = []
    
    # (genders,adulthood, hairlength, haircolor,colors) = names
    (gdr_prd, adl_prd, len_prd, hclr_prd, oclr_prd) = preds
    # color names
    colornames = {idx:name for (name,idx) in dataset.colors_list.items()}
    gendernames = {idx:name for (name,idx) in dataset.gender.items()}
    adulthoodnames = {idx:name for (name,idx) in dataset.adulthood.items()}
    lengthnames = {idx:name for (name,idx) in dataset.length.items()}
    # apply sigmoid followed by rounding operation so we get the 0-1 range for our colors
    outfit_colors = torch.round(oclr_prd.sigmoid())
    # print(f'{outfit_colors[0:3]=}')
    for row in range(outfit_colors.size(0)):
        current_outfit_color = outfit_colors[row]
        # print(f'{current_outfit_color.cpu().detach().numpy()}')
        oclr = ' '.join([colornames[idx] for idx,label in enumerate(current_outfit_color) if label==1])
        gdr = gendernames[torch.argmax(gdr_prd[row]).item()]
        adl = adulthoodnames[torch.argmax(adl_prd[row]).item()]
        len = lengthnames[torch.argmax(len_prd[row]).item()]
        hclr = colornames[torch.argmax(hclr_prd[row]).item()]
        lst_names.append((gdr, adl, len, hclr,oclr))
    return lst_names

def show_predictions(imgs, preds, rows=32, cols=1):
    fig = plt.figure(figsize=(224,224))
    plt.subplots_adjust(hspace=0.2)
    preds_name = parse_predictions(anime_dataset ,preds)
    for i, (img, preds) in enumerate(zip(imgs, preds_name)):
        img = unnormalize(img)
        ax = fig.add_subplot(rows, cols, i+1,xticks=[], yticks=[])
        
        ax.imshow(img)
        (gdr, adl, len, hclr, oclr )= preds
        str_info = f'gender: {gdr} | adulthood: {adl} |' \
                   f'hair length: {len} | hair color: {hclr}\noutfit color:{oclr}'
        ax.set_title(str_info)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
# load the best model first 
statedicts = torch.load('./weights/mtl_animestyles.pt')
model.load_state_dict(statedicts['weights'])
model.to(device)

print(f'overall average accuracy: {statedicts['all_accs_avg']:.2f}')
for imgs, _ in dataloader_test:
    imgs = imgs.to(device)
    model.eval()
    preds = model(imgs)
    # print(preds)
    show_predictions(imgs, preds)
    
# %%
# now as you can see the results are not perfect, neither is our architecture and
# training regime. in order to improve upon our results we need t o have a better
# architecture. for example utilize larger featuremap sizes so the color informations
# can be obtained more accurately and more easily and use block consisting several 
# layers instead of a single linear layer for each head. 
# we can then use different weights for each label if they are imbalanced
# among other things.
# I might continue this and further improve these examples but for now I believe
# it suffices our usecase which is get an idea about how stuff works and how we can
# improve them.
