# in the name of God the most compassionate the most merciful 
# in this section we will be looking at the image captioning 
# and see how we can implement this using an LSTM and later on 
# a transformer model. 
# to give you a short summary of what we are dealing with here,
# image captioning is a subfield of artificial intelligence that 
# intersects computer vision and natural language processing. 
# The objective is to develop models that can generate human-like
# textual descriptions for visual input, such as images or videos. 
# This task is non-trivial, as it requires the model to understand 
# both the content and context of the visual input, and express that
# understanding in natural language.
# we have come a long way since the early attempts at this task. 
# early attempts relied on template-based methods and manual feature 
# engineering. However, the advent of deep learning revolutionized the field.
# Convolutional Neural Networks (CNNs) became the standard for image feature
# extraction, while Recurrent Neural Networks (RNNs), particularly 
# Long Short-Term Memory (LSTM) networks, were used to generate the 
# corresponding captions. This combination, often referred to as an 
# encoder-decoder framework, has been the backbone of many state-of-the-art
# image captioning models.
# in recent years, the trend has shifted towards end-to-end trainable models 
# and the use of attention mechanisms. Attention allows the model to weigh the
# importance of different features when generating each word in the caption, 
# leading to more accurate and contextually relevant captions. The introduction 
# of Transformer architectures, which rely entirely on self-attention mechanisms,
# further improved the performance of image captioning systems.
# The latest trend in image captioning is the use of vision-language pre-training.
# In this approach, models are pre-trained on large-scale image-text datasets, and
# then fine-tuned for the image captioning task. This has led to state-of-the-art 
# performance on several benchmark datasets.
# in short: 
# This is an intriguing area of artificial intelligence that intersects computer vision
# and natural language processing. The goal is to create models that can generate 
# human-like textual descriptions for visual input, such as images or videos.
# This task is challenging as it requires the model to understand both the content
# and context of the visual input, and express that understanding in natural language. 
# From early template-based methods to the current state-of-the-art Transformer models,
# the journey of image captioning has been marked by significant milestones and continues
# to be an active area of research.
# so now that we have the introduction out of the way, we need to implement one. 
# we need datasets. we can create one ourselves, or we can use the common ones out there
# we chose the latter one, to both familiarize ourseleves with such datasets and then
# with the new information we can build ours the way we want it, if these datasets dont cut it!
# MS-COCO with 120K images, Conceptual Captions with 3.3 million images scarped from the web,
# and The Remote Sensing Image Captioning Dataset (RSICD) with 10K+ images are some of the most
# commond atasest used for image captioning. 
# we will be using MS-COCO as its easier and can be used offline. ConceptualCaptions dataset
# has image-url, description pair, which works fine if you have stable internet connection,
# or want to save 5m images (2.2 labeled, and 3.3 unlabled) images on you hdd. 
# the dataset is also available on huggingface, so using it is pretty easy!
# 
# for image captioning we need two models usually palced in an encoder-decoder form
# one for extracting image features, and the other for generating textual descriptions.
# so a cnn for the images, and an lstm for the descrption part would do the trick.
# later on we will go transformers and use transformer based models for both parts.
# lets import the required modules
#%%
import math 
import random
import time
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence 
# since we want to use vision models 
import torchvision
from torchvision import models, transforms
# lets use bert tokenizer and skip creating the tokenization process ourselves!
from transformers import BertTokenizer
# lets import PIL to read images compatibe with torchvision
from PIL.Image import Image
import matplotlib.pyplot as plt 
import tqdm

# our first step is to prepare the dataset
# note that since coco dataset is relatively large (more than 20+GB), we instead use the mini version
# which can be downloaded from : https://github.com/giddyyupp/coco-minitrain
# or directly from : https://ln5.sync.com/dl/0324da1d0/rmi7abjx-2dj4ktii-d9jcwgc5-s7fwwrb7
# its around 4.6GB
#
# we need to normalize the images the way the base model was trained
# and we also need to normalize our descriptions/text, tokenize them etc
class CustomDataset(Dataset):
    def __init__(self, root, split='train') -> None:
        super().__init__()
        
    
    def __getitem__(self, index):
        pass

    def __len__(self):
        pass 

#now that our dataset is ok, lets attend to our model. 
# the idea is simple, we need a model to get the image features,
# we then feed these features to an lstm and that lstm produces 
# the sequence.
# !there are two ways of doing this. 
# the first way you may see is to use image features as the initial hidden-state,
# that is, our image features, act as the first hiddent_state to the lstm, 
# followed by a start token so that the lstm starts generating the actual
# description sequence token by token until end token is met for example.
# you may also see, some people use a linear projection before feeding the cnn features
# to the lstm hidden_state.(that is use alinear layer before lstm, this way we can decouple
# the dimensions of our cnn features, and hidden_state size which is a good thing)
# its a good technique and we use it here as well.
# the second way is that the image-features are fed as the first timestep of the input
# description, and then fed the result to the lstm. in this case, the imagefeature
# is simply prepended to the sequence, and the last token is also removed so the number
# of tokens match the original token.
# in this scenario, a linear layer is also used to decouple the imagefeature size
# from the embedding size(thats used to encode the tokens). 
# we will implement both methods and see how they differ in results and which one 
# is better.
# sidenote, usually lstm with attention is used to maximize the performance, but
# we only use the lstm to keep things simple, as ultimately we will be testing with
# transformers that have superior performance compared to lstm variant anyway.
# I order to have better management over our code, its better to separate our encoder
# and decoder parts and then use them as standalone modules in our actual model. 
# this allows us to be able to use different implementations/strategies for either of them
# and easily test them, play with them.
#! write encoder 
class Encoder(nn.Module):
    def __init__(self, embd_size, projection_size=4096) -> None:
        super().__init__()
        
        # our encoder is a vision model, pretrained on imagenet, we remove the classifier and
        # feed the features/flattened of course, to our lstm
        self.model = models.resnet50(pretrained=True)
        # before we remove the last layer, lets grab the penultimate dimension which will become
        # our input_size for our lstm model
        self.penultimate_dim = self.model.fc.in_features
        # lets remove the classifier at the end, the output is (1,2048,1,1) for a batch of 1 image
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        if projection_size:
            # instead of using a simple linear layer to match the lstm embdsize/hiddensize
            # in the decoder, we go for a linear projection to get better performance 
            self.ln = nn.Sequential(nn.Linear(self.penultimate_dim, projection_size),
                                    nn.Linear(projection_size, embd_size))
        else:
            self.ln = nn.Linear(self.penultimate_dim, embd_size)
    
    def forward(self, imgs):
        # feed the input and flatten the features
        features = self.model(imgs).view(imgs.size(0), -1)
        features = self.ln(features)
        return features

#! write decoder
class Decoder(nn.Module):
    def __init__(self, vocab_size, embd_size, hidden_size, num_layers=1, bidirectional=False, dropout=0.0, itow={}, wtoi={}, method='INPUT') -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embd_size = embd_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.direction = 2 if bidirectional else 1
        # dictionaries for int to word and word to int tokens.
        self.itow = itow
        self.wtoi = wtoi
        # which method to use, use imagefeatures as input, or initial hidden_state
        self.method = method
        # lets create an embedding layer first 
        self.embd = nn.Embedding(vocab_size, embd_size)
        # now the lstm
        self.decoder = nn.LSTM(input_size=embd_size,
                               hidden_size=hidden_size, 
                               num_layers=num_layers,
                               dropout=dropout,
                               bidirectional=bidirectional,
                               batch_first=True)
        # and a final classifier, note that since we may be using bidirectional lstm
        # we need to make sure the first dimension takes that into account as well.
        self.fc = nn.Linear(self.direction*self.hidden_size, self.vocab_size)

    def forward(self, img_features, sequences, hidden_states=None):
        # feed the sequences to embds 
        embds = self.embd(sequences)
        if self.method.lower() == 'input':
            # add the img_features to the sequence as the first timestep/token(start token)
            # and remove the end token, 
            # extra-explanation: 
            # we add a single dimension at dim=1 so we have the
            # (batch,seq,features) for our img_features instead of (batch, features), we then 
            # concat it along the second dim (i.e. the sequence/timestep dim) which is 1, and
            # since its a single token its 1 obviously!
            input_seq = torch.cat([img_features.unsqueeze(1), embds[:,:-1,:]], dim=1)
            outputs, final_hiddenstate = self.decoder(input_seq, hidden_states)
        else:
            # use the img_features as the initial hidden_state
            # since we may be using bidirectional and more than 1 layers, we must make the
            # hidden_states match the shape ((D*num_layers),Batch,Features)
            # if we didnt use use multilayer or bidirectional lstm, sth as simple as 
            # hiddenstate = (img_features.unsqueeze(0), torch.zeros_like(img_features).unsqueeze(0))
            # would work becasue it satisfies the (1,b,f) features as (numlayers=1 and direction=1)
            # anyway, the following codesnippet works for all cases nevertheless.
            h_0 = torch.stack([img_features for _ in range(self.direction*self.num_layers)])
            c_0 = torch.stack([img_features.new_zeros(*img_features.shape) for _ in range(self.direction*self.num_layers)])
            hidden_states = (h_0, c_0)
            outputs, final_hiddenstate = self.decoder(embds,hidden_states)
        # and finally lets calculate the class probablities, also note that we need
        # to reshape outputs so the dims are compatible with our fc. 
        # note that, -1 merges the batch and sequence dimensions, and the out_features dim
        # becomes compatible with our fc layer 
        outputs = self.fc(outputs.reshape(-1, self.hidden_size*self.direction)).softmax(dim=-1)
        # and finally reshape the output back to (batch, seq, features) form
        outputs = outputs.view(*sequences.shape,-1)
        return outputs, final_hiddenstate

enc = Encoder(512)
dec = Decoder(vocab_size=100, embd_size=512, hidden_size=514,num_layers=2, bidirectional=True, method='input')
x_img = torch.randn(size=(2,3,224,224))
x_des = torch.randint(0,100,size=(2,30))
out_feats = enc(x_img)
outputs,_ = dec(out_feats, x_des)
print(f'{out_feats.shape=}')
print(f'{outputs.shape=}')
#%%
class EncoderDecoderCaptionist(nn.Module):
    def __init__(self, embd_size, hidden_size, projection_size, num_layers, bidirectional, lstm_drpout ) -> None:
        super().__init__()
        
        self.embd_size = embd_size
        self.hidden_size = hidden_size
        self.projection_size = projection_size
        self.num_layers = num_layers
        # bidirectional is not used for imagecaptioning usually, as the words depend on the previous
        # ones not the ones that come after, but I included so you can see how it performs in practice
        self.bidirectional = bidirectional
        self.lstm_dropout = lstm_drpout
        self.direction = 2 if bidirectional else 1
        # we use this to tokenize (encoding/decoding) using gpt2 scheme
        # we usually would want to check the captions in our dataset and create a vocabulary out of
        # that, and then normalize it, but since we dont have access to the data right now, im using
        # tiktoken tokenizer to get things started!
        # i had issues accessing tiktoken vocabs so I instead went for transformers tokenizers
        # self.tokenizer = tiktoken.get_encoding('gpt2')
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        # our encoder is a vision model, pretrained on imagenet, we remove the classifier and
        # feed the features/flattened of course, to our lstm
        self.encoder = models.resnet50(pretrained=True)
        # before we remove the last layer, lets grab the penultimate dimension which will become
        # our input_size for our lstm model
        encoder_output_dim = self.encoder.fc.in_features
        # lets remove the classifier at the end, the output is (1,2048,1,1) for a batch of 1 image
        self.encoder = nn.Sequential(*[child for child in self.encoder.children()][:-1])
        # now lets add the decoder part
        # we may need a vocab, tokenizer to convert our text to tensors and train the lstm
        # so we are going to use tiktoken for tokenization and use its vocab
        # this means, we need an embedding layer
        # the input of our embedding_layer is ofcourse vocab_size
        self.embd = nn.Embedding(self.tokenizer.vocab_size, embd_size)
        self.decoder = nn.LSTM(input_size = embd_size,
                               hidden_size=hidden_size,
                               num_layers=num_layers,
                               bidirectional=bidirectional,
                               batch_first=True,
                               dropout=lstm_drpout
                               )
        # decouple cnn features from lstm hidden-state size
        self.linear_projection = nn.Sequential(nn.Linear(encoder_output_dim, projection_size),
                                               nn.Linear(projection_size, hidden_size))
        # we seem to be on the right track, lets create the final classes,
        # also note that we need to take bidirection into account (I always forget this!)
        self.fc = nn.Linear(self.hidden_size*self.direction, self.tokenizer.vocab_size)
        
    def forward(self, imgs, descriptions):
        
        # Note: 
        # note that in image captioning the forward pass in our case is independent for each training
        # iteration, like we dont need the hidden_state from the previous iteration in the new one, 
        # becasue each sample is different and completely independent, its not like we are training 
        # to generate text, where the next batch of texts may get some clue from previous batch hidden state
        # (as a paragraph or sentence, may be split into different consecutive batches (becasu we dont shuffle
        # the text come after each other so it makes sense), but then again, for the cases where each sample was
        # independent we used hidden_state=None for each iteration.) so this applies here as well and 
        # we start off with zero hidden sate, and use the input image feature to create the initial 
        # hidden state and feed that to the network.
        # Note 2: 
        # if you forgot, lstm has two memory states, thats why its hidden_state is a tuple and thats why
        # we use the img_features as the first item and make the second all zeros.
        
        # get the embeddings for the input text
        embds = self.embd(descriptions)
        # get the image features 
        imgs_vectors = self.encoder(imgs).view(imgs.size(0),-1)
        # print(f'{imgs_vectors.shape=}')
        # project the feature-vectors so we can feed it as the hidden_state
        imgs_features = self.linear_projection(imgs_vectors)
        # note that we used new_zeros so the second part inherits the device, and 
        # other attributes of img_features so they match in that sense and we dont 
        # error out! due to device mismatch or type mismatch!
        # print(f'{imgs_features.shape=}')
        # since hidden_states need to be 3d, that is (direction*num_layers,batch,D)
        # we have the batch,D part, so we need to add the first dimension. note that doing
        # hidden_state = (imgs_features.unsqueeze(0), imgs_features.new_zeros(*imgs_features.shape).unsqueeze(0))
        # only works for single lstm layer, and for multi layer lstm or bidirectional lstm wont work
        # a better way is to do this instead which covers all cases
        h_0 = torch.stack([imgs_features for _ in range(self.direction*self.num_layers)])
        c_0 = torch.stack([imgs_features.new_zeros(*imgs_features.shape) for _ in range(self.direction*self.num_layers)])
        hidden_state = (h_0,c_0)
        # feed the embeddings to the lstm
        print(f'{hidden_state[0].shape=}')
        outputs, final_hiddenstate = self.decoder(embds, hidden_state)
        print(f'{final_hiddenstate[0].shape=}')
        # lets calculate the classes
        outputs = self.fc(outputs.reshape(-1, self.hidden_size * self.direction))
        # reshape to b,seq,feats
        outputs = outputs.softmax(dim=-1).view(*descriptions.shape,-1) 
               
        # final_hiddenstate is not needed, but we pass it anyway!
        return outputs, final_hiddenstate

    def stoi(self, text_list):
        result = self.tokenizer(text_list,
                                is_split_into_words=False, 
                                padding=True, 
                                truncation=True, 
                                return_tensors="pt")
        return result["input_ids"]
    
    def itos(self, ids_batch:torch.Tensor):
        # ids may have padding so remove them
        ids_batch = ids_batch[ids_batch!=0]
        return self.tokenizer.decode(ids_batch.tolist())

# now lets test this 
text = ["this is a test thats something random!?!.","second text"]
model = EncoderDecoderCaptionist(embd_size=300, hidden_size=512,
                                 projection_size=2186,
                                 num_layers=2,
                                 bidirectional=True,
                                 lstm_drpout=0.3)
x = torch.randn(size=(2,3,224,224))
tokens = model.stoi(text)
output,_ = model(x,tokens)
print(f'{output.shape}')
argmax = output.argmax(-1)
print(argmax)
model.itos(argmax)
#%%























