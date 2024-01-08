#%% in the name of God the most compassionate the most merciful 
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
import math 
import random
import time
import os
import glob
import pathlib
import numpy as np
import json
import string
import itertools
from collections import Counter
# lets import PIL to read images compatibe with torchvision
import PIL.Image as Image
import matplotlib.pyplot as plt 
from tqdm import tqdm


import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
# since we want to use vision models 
import torchvision
from torchvision import io, models, transforms
# lets use bert tokenizer and skip creating the tokenization process ourselves!
from transformers import BertTokenizer

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
    
    def __init__(self, vocab_size, embd_size, hidden_size, num_layers=1, bidirectional=False, dropout=0.0, method='INPUT') -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embd_size = embd_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.direction = 2 if bidirectional else 1
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
        outputs = self.fc(outputs.reshape(-1, self.hidden_size*self.direction))
        # and finally reshape the output back to (batch, seq, features) form
        # note that we dont use softmax here, as we are planning to use crossentropy
        # and crossentropy expects logits, and applies the softamx itself
        outputs = outputs.view(*sequences.shape,-1)
        # return the outputs logits and final_hiddensate
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
# now lets create our main model and use these blocks 
class EncoderDecoderImageCaption(nn.Module):
    
    def __init__(self,vocab_size, encoder_projection_size=4096, embd_size=512, hidden_size=512,
                 num_layers=1, decoder_dropout=0.0, bidirectional=False, method='input') -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embd_size = embd_size
        self.hidden_size = hidden_size
        self.encoder_projection_size = encoder_projection_size
        self.num_layers = num_layers
        self.decoder_dropout = decoder_dropout
        self.bidirectional = bidirectional
        self.method = method
        self.encoder = Encoder(self.embd_size, self.encoder_projection_size)
        self.decoder = Decoder(self.vocab_size, 
                               self.embd_size,
                               self.hidden_size, 
                               self.num_layers,
                               self.bidirectional,
                               self.decoder_dropout, 
                               self.method)
    
    def forward(self, imgs, captions, hidden_states):
        image_features = self.encoder(imgs)
        outputs, hidden_states = self.decoder(image_features, captions, hidden_states)
        return outputs, hidden_states
# lets test
x_img = torch.randn(size=(2,3,224,224))
x_captions = torch.randint(0,100,size=(2,30))
model = EncoderDecoderImageCaption(encoder_projection_size=4096,
                                   vocab_size=100,
                                   embd_size=512,
                                   hidden_size=514,
                                   num_layers=2,
                                   decoder_dropout=0.0,
                                   bidirectional=True,
                                   method='input')

outputs,_ = model(x_img, x_captions,None)
print(f'{outputs.shape=}')
#%%
# this is the initial model I wrote when I didnt have the dataset yet and wanted to use 
# a off the shelf tokenizer and vocab. everythings is the same except for the tokenizer
class EncoderDecoderImageCaption2(nn.Module):
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
        # reshape the outputs logits to b,seq,feats
        outputs = outputs.view(*descriptions.shape,-1) 
        # return the outputs logits. final_hiddenstate is not needed, but we pass it anyway!
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
single_text = ["this is a test thats something random!?!.","second text"]
model = EncoderDecoderImageCaption2(embd_size=300, hidden_size=512,
                                 projection_size=2186,
                                 num_layers=2,
                                 bidirectional=True,
                                 lstm_drpout=0.3)
x = torch.randn(size=(2,3,224,224))
tokens = model.stoi(single_text)
output,_ = model(x,tokens)
print(f'{output.shape}')
argmax = output.argmax(-1)
print(argmax.shape)
print(model.itos(argmax))

#%%
# now lets use our training set and create our vocab, and the dataset itself.
# if you look at the mscoco (which by  the way stands for MicroSoft Common Objects in COntext)
# anyway what we are intrested in is captions. the captions are json files, and organized into
# several fields, under the info field, we have, description, license, images and annotations,
# sub fields respectively. 
# under images, we have informations related to each image, and under annotations, we have captions
# for each image organized in a list of dictionary items, each entery has a image_id, id and caption keys
# with respective information. 

coco_root = '/media/hossein/SSD/mscoco_dataset/'
annotation_dir = 'annotations_trainval2017/annotations'
captions_train_fname = 'captions_train2017.json'
captions_val_fname = 'captions_val2017.json'
# image folders
train_dir = 'train2017'
val_dir = 'val2017'
with open(os.path.join(coco_root,annotation_dir,captions_train_fname),'r') as f:
    # note we use load() to load the file, not loads, s stands for string, its used
    # to read a json in string form and parses it to a python object. so when we work
    # with files, we always use load!
    annotations_train = json.load(f)
with open(os.path.join(coco_root,annotation_dir,captions_val_fname),'r') as f:
    annotations_val = json.load(f)
    
# we could also do this oneliner instead. either way its fine.(note that the first approach is prefered
# becsaue if an exception occurs, it gracefully closes the file handle, whereas in our second approach 
# below, if we were to face an exception, the file handler would stay open indefinitely!)
# captions_train = json.loads(open(os.path.join(coco_root,annotation_dir,captions_train_fname)).read())
# captions_val = json.loads(open(os.path.join(coco_root,annotation_dir,captions_val_fname)).read())
captions_train = annotations_train["annotations"]
captions_val = annotations_val["annotations"]
print(f'{captions_train[:3]}')
print(f'{captions_val[:3]}')
# now to create our vocabulary we need to read both of these texts and extract the words.
# insetad of simpling merging the two, we can use itertools.chain!
# all_captions = captions_train+captions_val
# print(len(all_captions))
#

# print(string.punctuation)
all_captions = []
for row in itertools.chain(captions_train,captions_val):
    caption_normalized = row['caption'].translate(str.maketrans('','',string.punctuation))
    all_captions.append(caption_normalized)
    
assert len(all_captions) == len(captions_train) + len(captions_val), 'size mismatch'
print(f'{len(all_captions)}')
# lets view couple of captions 
print(*all_captions[:3],sep='\n')
# good. now lets tokenize our captions and build our vocab and dictionaries 
words = set(word 
            for caption in all_captions 
            for word in caption.split())

print(f'{len(words):,}')
# now lets create our dictionaries
# start off with some special tokens
itow = dict(enumerate(["<start>","<end>","<unk>"]))
# and then add the rest
itow.update(enumerate(words,start=3))
wtoi = {v:k for k,v in itow.items()}
print(f'{itow=}')
print(f'{wtoi=}')
# now lets create conversion methods for ease of use 
def words_to_idxs(input_text):
    normalized = input_text.translate(str.maketrans('','',string.punctuation))
    #add special tokens to our input 
    normalized = f"{itow[0]} {normalized} {itow[1]}"
    # if a word is not in our vocabulary, encode it with <unk> symbol
    return [wtoi.get(word, wtoi["<unk>"]) for word in normalized.split()]

def idxs_to_words(input_idxs):
    return [itow[idx] for idx in input_idxs]

print(words_to_idxs("Hello world! this is a test baby"))
print(idxs_to_words(words_to_idxs("Hello world! this is a test baby")))
#%%
# now lets consolidate all of that as a single class for easier use!
# also since we are dealing with sequences with different length we 
# would want to add the logic to our tokenizer so we can use that 
# anywhere we require it

class Tokenizer():
           
    def __init__(self, train_captions, val_captions) -> None:
        
        all_captions = []
        for row in itertools.chain(train_captions, val_captions):
            caption_normalized = row['caption'].translate(str.maketrans('','',string.punctuation))
            all_captions.append(caption_normalized)
        
        assert len(all_captions) == len(train_captions) + len(val_captions), 'size mismatch'
        
        self.min_length, self.max_length, self.seq_stats = self._calculate_caption_statistics(all_captions)
        
        words = set(word 
                    for caption in all_captions 
                    for word in caption.split())
        # since we plan on padding our input with 0s, it would be better to set 0 as the end
        # so we dont mess up the semantics
        # special symbols for normalizing our sequences. we define them like this so in case
        # we wanted to change them anywhere in our code, we would be able to easily do so 
        # without any issues.
        self._start = '<start>'
        self._end = '<end>'
        self._unknown = '<unk>'
        self.itow = dict(enumerate([self._end, self._start, self._unknown]))
        self.itow.update(enumerate(words, start=3))
        self.wtoi = {v:k for k,v in self.itow.items()}
        self.vocab_size = len(self.wtoi)

    def __len__(self):
        return len(self.wtoi)
        
    def encode(self, input_text):
        normalized = input_text.translate(str.maketrans('','',string.punctuation))
        normalized = f"{self.itow[self.wtoi[self._start]]} {normalized} {self.itow[self.wtoi[self._end]]}"
        # if a word is not in our vocabulary, encode it with <unk> symbol
        return [self.wtoi.get(word, self.wtoi[self._unknown]) for word in normalized.split()]

    def batch_encode(self, input_text_batch):
        return [self.encode(text) for text in input_text_batch]
    
    def decode(self, input_idxs):
        return [self.itow[idx] for idx in input_idxs]
    
    def batch_decode(self, input_idxs_batch):
        return [self.decode(idx) for idx in input_idxs_batch]
        
    def _calculate_caption_statistics(self, all_captions):
        # lets calculate max and min seq_length
        all_captions_length = [len(caption.split()) for caption in all_captions]
        min_len = min(all_captions_length)
        max_len = max(all_captions_length)
        seq_lengths = Counter(all_captions_length)
        return min_len, max_len, seq_lengths
    
tokenizer = Tokenizer(captions_train, captions_val)
single_text = "Hello world! this is a test baby"
batch_text = ["this wasnt a dog in a park!", "that was definitely a dog in the park!"]

print(tokenizer.encode(single_text))
print(tokenizer.decode(tokenizer.encode(single_text)))

print(*tokenizer.batch_encode(batch_text), sep='\n')
print(*tokenizer.batch_decode(tokenizer.batch_encode(batch_text)), sep='\n')

idxs = tokenizer.encode(single_text)
print(f'{idxs}')
idxs_padded = F.pad(torch.tensor(idxs), pad=[0,100-len(idxs)],mode='constant',value=0)
print(f'{idxs_padded}')

print(f'{tokenizer.min_length=}')
print(f'{tokenizer.max_length=}')
print('most common lengths:\n(length : # of sequences)',*tokenizer.seq_stats.most_common(10),sep='\n')
# displaying them gives us a better understanding of which length is more common
# we see that the overwhelming majority of sequences have 8-10/11 length
# sidenote: 
# also note that here we used plt.bar function to do this 
# becasue the plt.hist is used to plot the histogram of 
# our dataset, which automatically divides the data into
# bins and counts the number of data points in each bin.
# However, in our case, we already have the counts of each
# sequence length (from the Counter object), so we don’t 
# need to count the data points again.
# The plt.bar function is more appropriate in this case 
# because it allows us to create a bar plot from two lists:
# one representing the x-coordinates (sequence lengths) and
# the other representing the heights of the bars (counts).
# if we were starting with a raw list of sequence lengths like
# all_captions_length list in our _calculate_caption_statistics
# and we wanted to count how many sequences have each length 
# (i.e., we didn’t already have a Counter object), then 
# plt.hist would be the right tool to use. 
# It would automatically sort the sequence lengths into bins
# (which we could specify), count the sequences in each bin, 
# and plot the histogram.
# 
# !side note 2:
# related to english by the way
# the phrase “how many sequences have each length” might be a bit confusing. 
# at least it was to me and I said to myself, would it not be better to have 
# had writen it as "how many sequences have the same length?
# it happens it wouldnt mean the same. you know the first one is obviously 
# intended to mean "for each possible length, how many sequences have that length".
# in other words, it’s counting the number of sequences that have a length of 1,
# then the number of sequences that have a length of 2, and so on.
# the second phrase "how many sequences have the same length” is slightly different.
# This would typically be used in a context where we have a specific length in mind
# and we want to know how many sequences are that long.
# For example, if we have a length of 5 in mind, we would say/ask
# "how many sequences have the same length" to find out how many sequences are 5 units long.
# But if we want to know the counts for all possible lengths, we would ask 
# "how many sequences have each length".
# 
plt.bar(list(tokenizer.seq_stats.keys()), tokenizer.seq_stats.values())
plt.xlabel('sequences length')
plt.ylabel('counts')
# if we were to use histogram it would look like sth like this 
# plt.hist(all_seq_lengths, bins=(range(1, max(all_seq_lengths)+2)), align='left', rwidth=0.8)
# plt.xlabel('sequence length')
# plt.ylabel('count')
# the first argument is self explanetory, bins=range(1, max(seq_lengths)+2) ensures 
# that there is a bin for each integer length from 1 to the maximum sequence length,
# and align='left' aligns the bins with their left edges for a better more intuitive display.
# and finally the rwidth=0.8 parameter makes the bars slightly narrower than the bins
# for a better visual effect.
#%%
# we created our tokenizer, dictionaries, conversion functions for wtoi and itow.
# we now need to create our dataset and then start training! so lets consolidate everything
# in our dataset and call it a day
class COCODataset(nn.Module):
        
    def __init__(self, coco_root, 
                 annotation_dir,
                 train_imgs_dir='train2017',
                 val_imgs_dir='val2017',
                 split='train',
                 tokenizer=None,
                 # simply used resize(224,224) for testing purposes, see the test you'll see
                 transformations=transforms.Compose([transforms.Resize((224,224)),
                                                     transforms.ToTensor()]
                                                    )) -> None:
        super().__init__()
        self._coco_root = coco_root
        self._annotation_dir = annotation_dir
        self._captions_train_fname = 'captions_train2017.json'
        self._captions_val_fname = 'captions_val2017.json'
        self._train_dir = train_imgs_dir
        self._val_dir = val_imgs_dir
        self.split = split
        self.tokenizer = tokenizer
        self.transformations = transformations
        
        # captions_fname = captions_train_fname if 'train' in split else captions_val_fname
        self.img_list = []
        if split.lower() == 'train':
            captions_fname = self._captions_train_fname
            self.imgs_folder = self._train_dir
        elif split.lower() == 'val':
            captions_fname = self._captions_val_fname
            self.imgs_folder = self._val_dir
        else:
            raise Exception(f"unknown split'{split}' entered!")
        
        with open(os.path.join(coco_root,annotation_dir,captions_fname),'r') as f:
                self.annotations = json.load(f)
                # self.caption is a list of dictionaries that each belong to an image
                # we are intersted in the caption and image-id fields of each dictionary
                # in this list
                self.captions = self.annotations["annotations"]

        # instead of reading the images into a list like this
        # self.img_list = list(glob.glob(os.path.join(self._coco_root, f"{self.imgs_folder}/*.jpg")))
        # instead we create a dictionary so we can grab the image by name, becasue our captions and
        # images are related, and the annotation dictionary, has a field for image-id and caption text
        # therefore we grab the image-id from annotations dictionary and look it up in the images dictionary
        self.img_dict = {int(pathlib.Path(f).stem):f for f in glob.glob(os.path.join(self._coco_root, f"{self.imgs_folder}/*.jpg"))}
    
    def __getitem__(self, index):
        img_id = self.captions[index]["image_id"]
        img = Image.open(self.img_dict[img_id]).convert('RGB')
        img = self.transformations(img)
        # tokenize the caption and return the padded, numpy/torch version
        caption = self.captions[index]['caption']
        caption_idxs = torch.tensor(tokenizer.encode(caption))
        # note that usually we dont return the padded/truncated sequence from the dataset
        # its the dataloader's job to create a batch of sequences, and if some 
        # have different lengths, to make them work using sth like padding/truncation.
        # we do that using the colate_fn argument and pass a function that handles
        # these kinds of stuff, so the dataset need to return the actual data it contains,
        # batching chores are offloaded to the dataloader.
        return img, caption_idxs 
            
    def __len__(self):
        return len(self.img_dict)

dt_train = COCODataset(coco_root,annotation_dir=annotation_dir, tokenizer=tokenizer, split='train')
dt_val = COCODataset(coco_root,annotation_dir=annotation_dir, tokenizer=tokenizer, split='val')

def show_image(img,caption):
    plt.imshow(img.permute(1,2,0).numpy())
    plt.title(tokenizer.decode(caption.tolist()))

print(f'{len(dt_train)=}')
print(f'{len(dt_val)=}')
img,caption = dt_train[1]
img_val,caption_val = dt_val[1]
show_image(img, caption)
show_image(img_val, caption_val)

#%%
# now lets create our colate_fn function to do sequence management, padd,trunctaion etc 
def normalize_sequences(image_caption_list):
    # note that we could go on and add truncation and padding to our tokenizer
    # so it gave us the truncated,padded sequence when encoding. truncation for
    # sequences that exceed a specific max_length we specify based on our findings
    # about our dataset statistics, and padding for sequences that are below our
    # max_length, so ultimately we have sequences of the same length.
    # but pytorch offers functions called pad_sequence and pack_padded_sequence
    # using these we can padd our sequences, and pack them as a batch, pytorch
    # automatically sorts everything our, it will padd the sequences based on the 
    # longest sequence, and later on using pack_padded_sequence, it allows the pytorch
    # to only process the actual sequences, by remving the paddings and hence we dont 
    # need to do much. 
    # since theres a huge discrepency between the maximum sequence length and the majority
    # of sequences, we have to do some truncation, so a single sequence or a select few dont
    # result in lots of paddings for the rest of the sequences. but thene again if pack_padded_sequence
    # allows torch to remove the padding and only porcess the actual tokens, then this should not
    # matter, and without truncation we should be good(the only reason for truncation would be to
    # preserve memory then and not performance hit (when the number of padding increases too much
    # and overwhemels the actual data, it hinders the learning. so if the padding length is the same
    # as the actual data or close to it, then this is not good for the model and we need to remove
    # the padding somehow. this seems not to be an issue when using pack_padded_sequence though))
    # 
    # Now lets implement this function 
    # the output of our dataset is an image,caption pair, and dataloader grabs couple of them
    # as a list, so the input to our colate_function is a list of whatever our dataset returns
    # we need to separate the images and create a batch for images, and a separate batch for the
    # captions. 
    # before that, we need to take care of our captions, since our image, captions are related and
    # are a pair, we cant simply grab the images, make a batch of it first and then get captions, 
    # becasue we need to do some processing which involves sorting the sequences first
    # in order to use pad_sequence feature in pytorch. it requires the sequences to be sorted 
    # in decending order based on their length. 
    image_caption_list.sort(key=lambda data: len(data[1]), reverse=True)
    # now lets split the images and captions 
    images, captions = zip(*image_caption_list)
    # now lets create a batch of images (simply stack them all)
    images = torch.stack(images,dim=0)
    # now lets padd our seqeunces 
    # our caption is simply a list of numbers, so we need to convert it to a tensor 
    # not only that, we also need to provide all the captions as list, so we simply
    # create a list of tensors representing our captions. now varying length of each
    # caption doesnt pose an error (becasue we are using a python list) and the pad_sequence
    # takes care of padding the tensors and making them all the same size.
    captions = [torch.tensor(caption) for caption in captions]
    captions = pad_sequence(captions, batch_first=True, padding_value=0)
    return images, captions
# lets test 
dl_train = DataLoader(dt_train, 5, shuffle=True, pin_memory=True, num_workers=0,collate_fn=normalize_sequences)
dl_val = DataLoader(dt_val, 5, pin_memory=True, num_workers=0,collate_fn=normalize_sequences)

print(f'{len(dl_train)=}')
print(f'{len(dl_val)=}')
imgs,captions = next(iter(dl_train))
imgs_val,captions_val = next(iter(dl_val))
plt.imshow(torchvision.utils.make_grid(imgs).permute(1,2,0).numpy())
plt.show()
plt.imshow(torchvision.utils.make_grid(imgs_val).permute(1,2,0).numpy())
print(f'{captions.shape}')
print(f'{captions=}')
print(f'{captions_val.shape}')
print(f'{captions_val=}')
# as you can see, the caption tensors are padded based on the largest sequence in that batch
# which is a great feature to have as sequences will usually have the least amount of padding
# and it changes dynamically based on each batch!
# %%
# now we have everything in place lets write our training loop
# we need 
# model
# optimizer
# criterion 
# scheduler
project_size = 4096
embd_size = 512 
hidden_size = 512
num_layers = 1
dropout = 0.0
bidirectional=False
method = 'input' # or hidden_state
device = 'cuda' if torch.cuda.is_available() else 'cpu'

epochs = 1
interval = 100
batch_size = 32
num_workers = 8

# data loaders
dl_train = DataLoader(dt_train, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers, collate_fn=normalize_sequences)
dl_val = DataLoader(dt_val, batch_size=batch_size, pin_memory=True, num_workers=num_workers, collate_fn=normalize_sequences)

model = EncoderDecoderImageCaption(tokenizer.vocab_size, 
                                   project_size, 
                                   embd_size=embd_size,
                                   decoder_dropout=dropout, 
                                   bidirectional=bidirectional,
                                   method=method)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=20, gamma=0.1)
criterion = nn.CrossEntropyLoss()



model.to(device)

print(f'{device=}')
print(f'{epochs=}')
print(f'{len(dl_train)=}')
print(f'{len(dl_val)=}')

for epoch in range(epochs):
    
    model.train()
    hidden_states = None
    losses = []
    accs = []
    for i,(imgs,captions) in tqdm(enumerate(dl_train)):
        imgs,captions = tuple(t.to(device) for t in (imgs, captions))
        outputs,_ = model(imgs, captions, hidden_states)
        # print(f'{outputs.argmax(dim=-1).shape=}')
        # print(f'{captions.shape=}')
        # print(f'{captions}')
        # 
        # since we have our input in the form of (Batch,Timesteps,Classes),
        # and crossentropy expects (Batch,Classes,Timesteps), we need to permute
        loss = criterion(outputs.permute(0,2,1), captions)
        # we could also do a reshape and offer both outputs as 2d tensors (and therefore
        # had to flatten the captions to make it 1d) by default our outputs tensor is 3d
        # it contains (B,T,C) and our captions/labels contains (B,T).
        # so in other words, outputs.view(-1, outputs.size(-1)) reshapes the outputs tensor
        # to be 2D with shape (batch_size * sequence_length, vocab_size), and captions.view(-1)
        # reshapes the captions tensor to be 1D with shape (batch_size * sequence_length,). 
        # This is necessary because as we just said CrossEntropyLoss expects the input tensor 
        # to be of shape (minibatch, C) and the target tensor to be of shape (minibatch,)
        # if we are doing multi-class classification problem (which we are, but with sequences)
        # loss = criterion(outputs.view(-1, outputs.size(-1)), captions.view(-1))
        # print(f'{loss=}')
        losses.append(loss.item())
        accs.append((outputs.argmax(dim=-1)==captions).float().mean().item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i%interval==0:
            print(f'{epoch}/{epochs} loss: {np.mean(losses):.4f} Accuray: {np.mean(accs)*100:.2f}')
        
    scheduler.step()
    with torch.no_grad():
        model.eval()
        losses=[]
        accs=[]
        for i,(imgs,captions) in tqdm(enumerate(dl_val)):
            imgs,captions = tuple(t.to(device) for t in (imgs, captions))
            outputs,_ = model(imgs, captions, hidden_states)
            # since we have our input in the form of (Batch,Timesteps,Classes),
            # and crossentropy expects (Batch,Classes,Timesteps), we need to permute 
            loss = criterion(outputs.permute(0,2,1), captions)
            losses.append(loss.item())
            accs.append((outputs.argmax(dim=-1)==captions).float().mean().item())
            
        print(f'{epoch}/{epochs} val-loss: {np.mean(losses):.4f} val-Accuray: {np.mean(accs)*100:.2f}')

# %%
