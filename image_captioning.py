#%% in the name of God the most compassionate the most merciful 
# in this section we will be looking at the image captioning 
# and see how we can implement this using an LSTM and later on 
# a transformer model. 
# to give you a short summary of what we are dealing with here,
# I need to say that image captioning is a subfield of ai that 
# intersects computer vision and natural language processing. 
# The objective here is to develop a model that can generate human-like
# textual descriptions for visual input, such as images or videos. 
# This is a hard task as it requires the model to understand 
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
# in recent years, however, the trend has shifted towards end-to-end trainable models 
# and the use of attention mechanisms. Attention if you recall, allows the model to weigh the
# importance of different features when generating each word in the caption, 
# leading to more accurate and contextually relevant captions. 
# The introduction of Transformer architectures, which rely entirely on
# self-attention mechanisms, further improved the performance of image captioning systems.
# The latest trend in image captioning is the use of vision-language pre-training.
# In this approach, models are pre-trained on large-scale image-text datasets, and
# then fine-tuned for the image captioning task. This has led to state-of-the-art 
# performance on several benchmark datasets.
# in short: 
# so now that we have the introduction out of the way, we need to implement one. 
# we need datasets. we can create one ourselves, or we can use the common ones out there
# we chose the latter one, to both familiarize ourseleves with such datasets and then
# with the new information we can build ours the way we want it, if these datasets dont cut it!
# MS-COCO with 120K images(118k,5k for trainval), Conceptual Captions with 3.3 million images 
# scarped from the web, Flicker8K, PascalVOC(very old), and The Remote Sensing Image Captioning 
# Dataset (RSICD) with 10K+ images are some of the most common datasest used for image captioning. 
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
from typing import Any
import numpy as np
import json
import string
import itertools
from collections import Counter
# lets import PIL to read images compatibe with torchvision
import PIL.Image as Image
import matplotlib.pyplot as plt 
from tqdm import tqdm
import pickle

# for BLEU score
import nltk
from nltk.translate.bleu_score import sentence_bleu,corpus_bleu
# evaluate is much better it offers more metrics like rogue bleu, etc as well.
import evaluate

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, unpack_sequence, unpad_sequence
# since we want to use vision models 
import torchvision
from torchvision import io, models, transforms
# lets use bert tokenizer and skip creating the tokenization process ourselves!
from transformers import BertTokenizer

from simplenet import simplenetv1_9m_m1,simplenetv1_9m_m2,simplenetv1_5m_m1,simplenetv1_5m_m2

# our first step is to prepare the dataset
# note that since coco dataset is relatively large (more than 20+GB), we instead use the mini version
# which can be downloaded from : https://github.com/giddyyupp/coco-minitrain
# or directly from : https://ln5.sync.com/dl/0324da1d0/rmi7abjx-2dj4ktii-d9jcwgc5-s7fwwrb7
# its around 4.6GB
# !edit: ok, I downloaded the coco dataset and used the 2017 version here. 
# the minicoco was for detection I guess!

# lets first see how the model should work.
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
# its a good technique and we use it here as well. this approach gives us the best results by far.
# 
# the second(older) way is that the image-features are fed as the first timestep of the input
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
# In order to have better management over our code, its better to separate our encoder
# and decoder parts and then use them as standalone modules in our actual model. 
# this allows us to be able to use different implementations/strategies for either of them
# and easily test them, play with them.
#! write encoder 
class Encoder(nn.Module):
    def __init__(self, backend_name, embd_size, projection_size=4096) -> None:
        super().__init__()
        
        self.backend_name = backend_name.lower()
        # our encoder is a vision model, pretrained on imagenet, we remove the classifier and
        if 'res' in self.backend_name:
            # feed the features/flattened of course, to our lstm
            self.backend = models.resnet50(pretrained=True)
            # before we remove the last layer, lets grab the penultimate dimension which will become
            # our input_size for our lstm model
            self.penultimate_dim = self.backend.fc.in_features
            # lets remove the classifier at the end, the output is (1,2048,1,1) for a batch of 1 image
            self.backend = nn.Sequential(*list(self.backend.children())[:-1])
        
        elif 'simp' in self.backend_name:
            # self.encoder = torch.hub.load("coderx7/simplenet_pytorch", "simplenetv1_9m_m1", pretrained=True)
            self.backend = simplenetv1_5m_m2(pretrained=True)
            # before we remove the last layer, lets grab the penultimate dimension which will become
            # our input_size for our lstm model
            self.penultimate_dim = self.backend.classifier.in_features*49
            # self.backend = nn.Sequential(self.backend.features,
            #                             nn.AdaptiveMaxPool2d(7))
            self.backend = nn.Sequential(*list(self.backend.children())[:-1])
        else:
            raise Exception('unknown model')

        if projection_size:
            # instead of using a simple linear layer to match the lstm embdsize/hiddensize
            # in the decoder, we go for a linear projection to get better performance 
            self.ln = nn.Sequential(nn.Linear(self.penultimate_dim, projection_size),
                                    nn.Linear(projection_size, embd_size),
                                   )
        else:
            self.ln = nn.Linear(self.penultimate_dim, embd_size)
    
    def forward(self, imgs):
        # feed the input and flatten the features
        features = self.backend(imgs).view(imgs.size(0), -1)
        features = self.ln(features)
        return features

#! write decoder
class Decoder(nn.Module):
    
    def __init__(self, vocab_size, embd_size, hidden_size, num_layers=1, bidirectional=False, dropout=0.0, method='HIDDEN_STATE') -> None:
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
        
        # self.drpout = nn.Dropout(self.dropout)
        # and a final classifier, note that since we may be using bidirectional lstm
        # we need to make sure the first dimension takes that into account as well.
        self.fc = nn.Linear(self.direction*self.hidden_size, self.vocab_size)
        # didnt affect or affects the performance by a little bit only-removed for faster training
        # self.ln= nn.LayerNorm(self.direction*self.hidden_size)

    def forward(self, img_features, sequences, hidden_states=None):
        # feed the sequences to embds 
        embds = self.embd(sequences)
        if self.method.lower() in ['in','inp','input']:
            # add the img_features to the sequence as the first timestep/token(start token)
            # and remove the end token, 
            # extra-explanation: 
            # we add a single dimension at dim=1 so we have the
            # (batch,seq,features) for our img_features instead of (batch, features), we then 
            # concat it along the second dim (i.e. the sequence/timestep dim) which is 1, and
            # since its a single token its 1 obviously!
            embds = torch.cat([img_features.unsqueeze(1), embds[:,:-1,:]], dim=1)
            # we can get fancy and do other operations on embds like use an mlp to 
            # fuse the features before being fed into the lstm for further processing
            # but for now we stick to the simplest form and wont spend much time here
            # as our intention is to use transformers later on
        
        elif self.method.lower() in ['h','ht','hidden','hidden_state']:
            # note that since at test time we want to be able to generate description
            # we need to keep generating tokens to produce the final sentence. 
            # recall that lstm is nothing but a loop over input sequences
            # so if we feed it an input of a single sequence, it will gives us
            # a single sequence output, one token in, one token out!
            # if we want a sentence, it means either we need to feed it
            # an input sentnce or keep feeding it a new token and get a
            # new output until we get our sentence. 
            # obviously when we are doing eval at test, we dont have any input
            # sentence, we have a single image, and we want to create a
            # sentence for the given image. so we give it the start token
            # and use the networks output which is a single token, feed
            # it again as the new token, along with the hidden_state from
            # the previous step, and produce a new token, and keep repeat
            #ing this until we  reach our sentence length. this is why here we check
            # and only use the image-features wheh hidden_state is None signifying
            # its the initial hidden_state, for the case where its not None, 
            # it means, we are at test time and trying to generate a description
            # for a given image, and we already passed the input image, once
            # and we are in the process of generating the next tokens.
            # dont wory, the inclusion of this 'if statement' doesnt have much
            # impact on the performance.
            if hidden_states is None:
                # use the img_features as the initial hidden_state.
                # instead of feeding the image features as the first input to the LSTM, we
                # can use the image features to initialize the hidden state of the LSTM. 
                # this allows the LSTM to carry the image information through its internal state.
                # since we may be using bidirectional and more than 1 layers, we must make the
                # hidden_states match the shape ((D*num_layers),Batch,Features)
                # if we didnt use use multilayer or bidirectional lstm, sth as simple as 
                # hiddenstate = (img_features.unsqueeze(0), torch.zeros_like(img_features).unsqueeze(0))
                # would work becasue it satisfies the (1,b,f) features as (numlayers=1 and direction=1)
                # anyway, the following codesnippet works for all cases nevertheless.
                h_0 = torch.stack([img_features for _ in range(self.direction*self.num_layers)])
                c_0 = torch.stack([img_features for _ in range(self.direction*self.num_layers)])
                # works better than simply using zeros
                # c_0 = torch.stack([img_features.new_zeros(*img_features.shape) for _ in range(self.direction*self.num_layers)])
                hidden_states = (h_0, c_0)
            # outputs, final_hiddenstate = self.decoder(embds, hidden_states)

        else:
            raise Exception(f"unknown method used ({self.method})")
        
        outputs, final_hiddenstate = self.decoder(embds, hidden_states)
        # outputs = self.ln(outputs.reshape(-1, self.hidden_size*self.direction))
        # and finally lets calculate the class probablities
        # note that, -1 merges the batch and sequence dimensions, and the out_features dim
        # becomes compatible with our fc layer
        # why not simply using 
        # outputs = self.fc(outputs)
        #! why did I do all of this? why?
        # note that pytorch's nn.Linear layers can accept inputs of more than two dimensions. 
        # they apply the linear transformation to the last dimension and consider all other 
        # dimensions as part of the batch.
        # for example, if we have an input tensor of shape (batch_size,seq_len, num_features),
        # nn.Linear will apply the same linear transformation to every feature vector across 
        # both the batch_size and seq_len dimensions.
        # However, the reshaping operation outputs.reshape(-1, self.hidden_size*self.direction)
        # is often used when we want to connect the output of an LSTM to a fully connected layer
        # and we care about the individual outputs at each time step.
        # Without the reshape, nn.Linear would apply the same transformation to the outputs at
        # each time step, effectively treating each time step as part of the batch. 
        # This is fine if we only care about the final output of your LSTM, but if we want to 
        # make predictions based on each time step (like in sequence-to-sequence models), we 
        # would need to reshape our outputs before passing them to the fully connected layer.
        # as for our specific case, in image captioning, our target is a sequence of words 
        # (the caption), and we are using a word at each time step as the target for our 
        # models prediction at that time step. If we dont reshape the outputs, the fully 
        # connected layer (nn.Linear) will treat the sequence length dimension as part of
        # the batch, and it will output a prediction for each time step. This is fine if we
        # are only interested in the final output of our LSTM, but in image captioning, we 
        # are typically interested in the output at each time step.
        # by reshaping the LSTM outputs to be 2D (with shape (batch_size * seq_len, hidden_dim)),
        # we are treating each time step as a separate data point. 
        # This allows our model to make a separate prediction for each word in the caption, 
        # which is what we want in a task like image captioning. if you comment the following 
        # two lines and instead use a outputs =self.fc(outputs) and train the model you'll see
        # the convergance rate is slower and we get lower accuracy compared to now (reshaping here)
        # note that its true if we reshape the outputs to 2D, pass them through the fc layer,
        # and then reshape them back to 3D, the final shape of the outputs will be the same 
        # as if we had directly passed the 3D outputs through the fc layer.
        # However, the values within the tensor may not be the same. This is because the fully connected layer applies a linear transformation, and this transformation is applied independently to each 2D slice of the input when the input is 3D. When you manually reshape the tensor to 2D before passing it through the fully connected layer, you’re changing which elements of the tensor are grouped together in these 2D slices.
        # So while the shape of the output tensor may be the same in both cases, the semantics of the operation are different. In the first case (reshaping to 2D and then back to 3D), you’re treating each time step of each sample as a separate data point. In the second case (directly using the 3D outputs), you’re treating each time step as part of the batch.
        # lets make this more elaborate with an intuiitive example 
        # let’s consider an image captioning scenario.
        # suppose we have a batch of 2 images, and we're using an LSTM to generate captions
        # for these images. lets say the LSTM generates the following output sequences 
        # (after reshaping and passing through the fully connected layer):
        # image 1: "A cat on a sofa", "A dog in a park"
        # image 2: "A man riding a bike", "A woman holding an umbrella"
        # if we reshape these outputs to 2D before passing them through the fc layer
        # (outputs.reshape(-1, self.hidden_size*self.direction)), the fully connected 
        # layer will treat each time step of each sample as a separate data point. 
        # so, it will make a separate prediction for each word in the caption. 
        # This is what we want here.
        # however, if we directly use the 3D outputs in the linear layer without reshaping,
        # the linear layer will apply the same transformation to the outputs at each time step,
        # effectively treating each time step as part of the batch. This could potentially mix
        # information between different images and their corresponding captions in the batch,
        # which is not desirable.
        # in other words, reshaping the outputs to 2D before passing them through the fully 
        # connected layer ensures that the model generates a separate caption for each image,
        # without mixing information between different images and captions in the batch. 
        # ! needs another test to verify this is the case
        outputs = self.fc(outputs.reshape(-1, self.hidden_size*self.direction))
        # and finally reshape the output back to (batch, seq, features) form
        # note that we dont use softmax here, as we are planning to use crossentropy
        # and crossentropy expects logits, and applies the softamx itself
        outputs = outputs.view(*sequences.shape,-1)
        # return the outputs logits and final_hiddensate
        return outputs, final_hiddenstate


enc = Encoder('simple',512)
dec = Decoder(vocab_size=100, 
              embd_size=512,
              hidden_size=512,
              num_layers=2, 
              bidirectional=True, 
              method='in')
x_img = torch.randn(size=(2,3,224,224))
x_des = torch.randint(0,100,size=(2,30))
out_feats = enc(x_img)
outputs,_ = dec(out_feats, x_des, None)
print(f'{out_feats.shape=}')
print(f'{outputs.shape=}')
#%%
# now lets create our main model and use these blocks 
class EncoderDecoderImageCaption(nn.Module):
    
    def __init__(self,vocab_size, encoder_backend='simplenet', encoder_projection_size=4096, embd_size=512, hidden_size=512,
                 num_layers=1, decoder_dropout=0.0, bidirectional=False, method='input') -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.encoder_backend = encoder_backend.lower()
        self.embd_size = embd_size
        self.hidden_size = hidden_size
        self.encoder_projection_size = encoder_projection_size
        self.num_layers = num_layers
        self.decoder_dropout = decoder_dropout
        self.bidirectional = bidirectional
        self.method = method
        self.encoder = Encoder(self.encoder_backend,
                               self.embd_size, 
                               self.encoder_projection_size)
        
        self.decoder = Decoder(self.vocab_size, 
                               self.embd_size,
                               self.hidden_size, 
                               self.num_layers,
                               self.bidirectional,
                               self.decoder_dropout, 
                               self.method)
        # disable gradients for the encoder part, becasue its job is to give us img features
        # and it doesnt need to change, the lstm/decoder part however, needs to be updated though
        # so lets effectively freeze our encoder!(this consumes less vram and improves performance as well)
        for module in self.encoder.modules():
            module.requires_grad_(False)
    
    def forward(self, imgs, captions, hidden_states):
        image_features = self.encoder(imgs)
        outputs, hidden_states = self.decoder(image_features, captions, hidden_states)
        return outputs, hidden_states
    
    def generate_caption(self, pil_image, transformations_val, max_length, tokenizer_obj, topk=3):
        with torch.no_grad():
            model.eval()
            hidden_states = None
            output_str = []
            caption_str = "a"
            device = next(self.parameters()).device
            image_features = self.encoder(transformations_val(pil_image).unsqueeze(0).to(device))
            for i in range(max_length):
                caption = torch.tensor(tokenizer_obj.encode(caption_str, False), device=device).view(1,-1).long()
                outputs, hidden_states = model.decoder(image_features, caption, hidden_states)
                # grab topk words
                probs, indexes = outputs.softmax(dim=-1).topk(k=topk, dim=-1)
                probs = probs.view(probs.size(-1))
                indexes = indexes.view(indexes.size(-1))
                output_k = indexes[torch.multinomial(probs/probs.sum(dim=-1),1, replacement=True)]
                caption_str = tokenizer_obj.decode([output_k.item()])[0]
                output_str.append(caption_str)
                # print(caption_str)
        return ' '.join(output_str)


# lets test
x_img = torch.randn(size=(2,3,224,224))
x_captions = torch.randint(0,100,size=(2,30))
model = EncoderDecoderImageCaption(encoder_backend='simplenet', 
                                   encoder_projection_size=4096,
                                   vocab_size=100,
                                   embd_size=512,
                                   hidden_size=512,
                                   num_layers=2,
                                   decoder_dropout=0.0,
                                   bidirectional=True,
                                   method='hidden_state')

outputs,_ = model(x_img, x_captions,None)
print(f'{outputs.shape=}')
#%%
# this is the initial model I wrote when I didnt have the dataset yet and wanted to use 
# a off the shelf tokenizer and vocab. everythings is the same except for the tokenizer
class EncoderDecoderImageCaption2(nn.Module):
    def __init__(self, encoder_modelname, embd_size, hidden_size, projection_size, num_layers, bidirectional, lstm_drpout ) -> None:
        super().__init__()
        self.encoder_modelname = encoder_modelname
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
        if 'resnet' in self.encoder_modelname.lower():
            self.encoder = models.resnet50(pretrained=True)
            # before we remove the last layer, lets grab the penultimate dimension which will become
            # our input_size for our lstm model
            encoder_output_dim = self.encoder.fc.in_features
        
        elif 'simplenet' in self.encoder_modelname.lower():
            # self.encoder = torch.hub.load("coderx7/simplenet_pytorch", "simplenetv1_9m_m1", pretrained=True)
            self.encoder = simplenetv1_5m_m2(pretrained=True)
            # before we remove the last layer, lets grab the penultimate dimension which will become
            # our input_size for our lstm model
            encoder_output_dim = self.encoder.classifier.in_features*7*7

        else:
            raise Exception('unknown model')
            
        # lets remove the classifier at the end, the output is (1,2048,1,1) for a batch of 1 image
        self.encoder = nn.Sequential(*[child for child in self.encoder.children()][:-1])
        # lets set the encoder's gradient to 0 
        for module in self.encoder.modules():
            module.requires_grad_(False)
            
        # print(self.encoder(torch.randn(size=(1,3,224,224))).shape)
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
model = EncoderDecoderImageCaption2(encoder_modelname='simplenet',embd_size=300, hidden_size=512,
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
from functools import reduce
class Tokenizer():
           
    def __init__(self, train_captions, val_captions, use_lower=True) -> None:
        
        all_captions = []
        for row in itertools.chain(train_captions, val_captions):
            caption_normalized = row['caption'].translate(str.maketrans('','',string.punctuation))
            all_captions.append(caption_normalized)
        
        assert len(all_captions) == len(train_captions) + len(val_captions), 'size mismatch'
        # we can select words based on their frequency as well, like for example, we include
        # the words that are repeated in the dataset more than n times (e.g. 3)
        # we dont do that now!but we can if we want!
        self.min_length, self.max_length, self.seq_stats = self._calculate_caption_statistics(all_captions)
        #! make the whole words lowercase before creating the vocabs! this should drastically
        #! lower our vocabsize and thus model parameters and also improve performance hopefully!
        # by default we have 37,937 words, but if
        # we use .lower() we'll get 29,629 words.
        # thats around 8k fewer words/classes
        words = set(word.lower() if use_lower else word
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
        # since we later pad our input, we want to make sure we can differentiate between
        # our actual values and pads(initially I used 0 for end, and pads, but then changed
        # my mind to make it more obvious)
        self._pad = '<pad>'
        self.special_tokens = (self._pad, self._start, self._end, self._unknown)
        
        self.itow = dict(enumerate(self.special_tokens))
        # incase we add new special tokens, lets dynamically update the counts
        self.itow.update(enumerate(words, start=len(self.special_tokens)))
        self.wtoi = {v:k for k,v in self.itow.items()}
        self.vocab_size = len(self.wtoi)
        # save itow,wtoi and words to file for later use
        with open('vocabs.pkl','wb') as f:
            pickle.dump({"itow":self.itow, "wtoi":self.wtoi, "words":words}, f)

    def __len__(self):
        return len(self.wtoi)

    def encode(self, input_text, add_special_tokens=True):
        # if we recieve any special tokens, just return their code, they are probably
        # the initial token for text-generation at test time
        if input_text in self.special_tokens:
            return self.wtoi[input_text]
        
        normalized = input_text.translate(str.maketrans('','',string.punctuation))
        if add_special_tokens:
            normalized = f"{self.itow[self.wtoi[self._start]]} {normalized} {self.itow[self.wtoi[self._end]]}"
        # if a word is not in our vocabulary, encode it with <unk> symbol
        return [self.wtoi.get(word, self.wtoi[self._unknown]) for word in normalized.split()]

    def batch_encode(self, input_text_batch, add_special_tokens=True):
        return [self.encode(text,add_special_tokens) for text in input_text_batch]
    
    def decode(self, input_idxs, to_str=False, remove_special_tokens=False):
        token_list = self.special_tokens if remove_special_tokens else []
        output_list = [word for idx in input_idxs if (word:=self.itow[idx]) not in token_list]
        return ' '.join(output_list) if to_str else output_list
    
    def batch_decode(self, input_idxs_batch, to_str=False, remove_special_tokens=False):
        return [self.decode(idx, to_str, remove_special_tokens) for idx in input_idxs_batch]
                
    def _calculate_caption_statistics(self, all_captions):
        # lets calculate max and min seq_length
        all_captions_length = [len(caption.split()) for caption in all_captions]
        min_len = min(all_captions_length)
        max_len = max(all_captions_length)
        seq_lengths = Counter(all_captions_length)
        return min_len, max_len, seq_lengths
    
tokenizer = Tokenizer(captions_train, captions_val)
print(f'{tokenizer.itow=}')
single_text = "Hello world! this is a test baby"
batch_text = ["this wasnt a dog in a park!", "that was definitely a dog in the park!"]
print(f'{tokenizer.vocab_size=:,}')
print(tokenizer.encode(single_text))
print(tokenizer.decode(tokenizer.encode(single_text)))

print(*tokenizer.batch_encode(batch_text), sep='\n')
print(*tokenizer.batch_decode(tokenizer.batch_encode(batch_text),to_str=True, remove_special_tokens=False), sep='\n')

idxs = tokenizer.encode(single_text)
print(f'{idxs}')
# pad the input with max-length 0f 100. i.e. anything less than 100, will be padded with 0s
idxs_padded = F.pad(torch.tensor(idxs), pad=[0,100-len(idxs)],mode='constant',value=0)
print(f'{idxs_padded}')

print(f'{tokenizer.min_length=}')
print(f'{tokenizer.max_length=}')
top_num=15
print(f'most common lengths({top_num}):\n(length : # of sequences)',
      *tokenizer.seq_stats.most_common(top_num),sep='\n')
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
        # note that there are multiple captions for the same image! which we need to account for!
        img_id = self.captions[index]["image_id"]
        img = Image.open(self.img_dict[img_id]).convert('RGB')
        img = self.transformations(img)
        # tokenize the caption and return it
        caption = self.captions[index]['caption']
        caption_idxs = tokenizer.encode(caption)
        # note that usually we dont return the padded/truncated sequence from the dataset
        # its the dataloader's job to create a batch of sequences, and if some 
        # have different lengths, to make them work using sth like padding/truncation.
        # we do that using the colate_fn argument and pass a function that handles
        # these kinds of stuff, so the dataset need to return the actual data it contains,
        # batching chores are offloaded to the dataloader.
        return img, caption_idxs 

    def __len__(self):
        # note that there are several captions per image, so we use captions length
        return len(self.captions)

dt_train = COCODataset(coco_root,annotation_dir=annotation_dir, tokenizer=tokenizer, split='train')
dt_val = COCODataset(coco_root,annotation_dir=annotation_dir, tokenizer=tokenizer, split='val')

def show_image(img,caption):
    plt.imshow(img.permute(1,2,0).numpy())
    plt.title(tokenizer.decode(caption,True,True))

print(f'{len(dt_train)=:,}')
print(f'{len(dt_val)=:,}')
img,caption = dt_train[1]
img_val,caption_val = dt_val[1]
show_image(img, caption)
show_image(img_val, caption_val)

#%%
# now lets create our colate_fn function to do sequence management, padd,trunctaion etc 
# since our colate_fn has a new argument, we simply use a lambda to apply our argument
# like collate_fn=lambda batch: normalize_sequences(batch))
# 
# side note: lambda doesnt work for DDP(distributed data parallel training) and instead we
# can use a functor. we can also simply create a class with a __call__() method and use the
# class object instead. 
# since we have more arguments now, like truncation length, end token, and maybe more later
# lets implemenet this as a functor! 
class OurColateFN():
    def __init__(self, truncation_length, end_token) -> None:
        self.truncation_length = truncation_length
        self.end_idx = end_token

    def __call__(self, batch):
        return self._normalize_sequences(batch)
    
    def _normalize_sequences(self, caption_list):
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
        caption_list.sort(key=lambda data: len(data[1]), reverse=True)
        # now lets split the images and captions 
        images, captions = zip(*caption_list)
        # now lets create a batch of images (simply stack them all)
        images = torch.stack(images,dim=0)
        # now lets padd our seqeunces 
        # our caption is simply a list of numbers, so we need to convert it to a tensor 
        # not only that, we also need to provide all the captions as list, so we simply
        # create a list of tensors representing our captions. now varying length of each
        # caption doesnt pose an error (becasue we are using a python list) and the pad_sequence
        # takes care of padding the tensors and making them all the same size.
        # 
        # important note: note that our labels/captions are always shifted one token to the left
        # so the network learns the sequence one after the other, other wise, at generation
        # it will suck! while at training it seemingly achieves 100% really quick! so 
        # in nlp tasks where we create text as output, the labels are always one token ahead!
        # if we were to only create the label, like this and remove the first token, we would have
        # an issue. because we are creating the caption tensors with torch.tensor(caption[1:]).
        # This will remove the first token from each caption. If the first token is a special 
        # start-of-sequence token (which is <start> ), then we're effectively removing it. 
        # This could be a problem because our model needs this token to know where each caption
        # begins. so instead we directly create, inputs and labels here and then pad them
        
        # lets grab up to truncation_length items only, the smaller sequences will later be padded
        # and the larger ones wont have any paddings, and thus all will have the same length hopefully
        input_captions  = [torch.tensor(caption[:self.truncation_length][:-1]) for caption in captions]
        # note that for targets, we need to have the <end> at the end of our sequence
        # since we are truncating now, we need to assure the last token is <end>
        # target_captions = [caption[:truncation_length][1:] for caption in captions]
        target_captions = [torch.tensor(caption[:self.truncation_length][1:-1] + [self.end_idx])
                           if caption[:self.truncation_length][-1] != self.end_idx 
                           else torch.tensor(caption[:self.truncation_length][1:]) 
                           for caption in captions]
        
        input_captions  = pad_sequence(input_captions, batch_first=True, padding_value=0)
        target_captions = pad_sequence(target_captions, batch_first=True, padding_value=0)
        return images, input_captions, target_captions
    
# now lets test 
trunc_len = 15
end_token = tokenizer.encode(tokenizer._end)
dl_train = DataLoader(dt_train, 5, shuffle=True, pin_memory=True, num_workers=0,collate_fn=OurColateFN(trunc_len,end_token))
dl_val = DataLoader(dt_val, 5, pin_memory=True, num_workers=0,collate_fn=OurColateFN(trunc_len,end_token))

print(f'{len(dl_train)=}')
print(f'{len(dl_val)=}')
imgs, captions, targets = next(iter(dl_train))
imgs,captions,targets = next(iter(dl_val))
plt.imshow(torchvision.utils.make_grid(imgs).permute(1,2,0).numpy())
plt.show()
plt.imshow(torchvision.utils.make_grid(imgs).permute(1,2,0).numpy())
print(f'{captions.shape,targets.shape}')
print(f'{captions,targets=}')
print(f'{captions.shape}')
print(f'{captions=}')
print(*tokenizer.batch_decode(captions.tolist(),remove_special_tokens=False),sep='\n')
print(*tokenizer.batch_decode(targets.tolist(),remove_special_tokens=False),sep='\n')
# as you can see, the caption tensors are padded based on the largest sequence in that batch
# which is a great feature to have as sequences will usually have the least amount of padding
# and it changes dynamically based on each batch!

#%%
# now we have everything in place lets write our training loop
# we need 
# model
# optimizer
# criterion 
# scheduler
# a measure/score for how good our captions is (i.e use BLEU)
# enable debugging
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# side note concerining the quality of the description and its relationship with encoder/decoder
# How to know how to imporve the result further? 
# when we see the model creates gramatically correct yet semantically wrong descriptions
# it means the language modeling part(i.e. decoder part) is developed properly but the 
# image features part(i.e. encoder part) or the utilization of it is not good enough
# so we would pay more attention to the encoder part, using better model, enhancing its
# quality, etc, or the merging of the features, how we feed those features to our decoder
# etc. if we see our decoder gives us hints about the image yet the grammer is wrong or not
# formed properly then the issue lies at the decoder and language modeling part, we eighter
# need more embedding features/dims, hidden_state size, or even data  to begin with , 
# we might need dropout if we overfit there as well.

tokenizer = Tokenizer(captions_train, captions_val, use_lower=False)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
encoder_backend = 'resnet' # resent50 works the best, simplenet is just there as another test model
project_size = 2048 # this affects the output a lot!
# when using hidden_state the embedding_size and hidden_size must be the same
# as our image_features are used as initial hidden_states so should match
# for input mode, they can be different as they are fed as the initial token of
# the embeddings.
end_token=tokenizer.encode(tokenizer._end)
embd_size = 512
hidden_size = 512
num_layers = 1
dropout = 0.1
bidirectional=False
# (hidden_state gets 35% while input achieves 24.0% without bidirectional,
# if you enable bidirectional, then input's accuracy goes to 50/60%)
# the input version has a hardtime learning unless we enable bidirection
# but this doesnt mean its learning properly, it does reduce the loss drastically
# but at the expense of generating nonsense at test time. so we use no bidirectional
# in image captioning unless maybe if we use it in the middle, in an encoder type of thing for
# the decoder section(imagine our decoder is split into an encoder/decoder submodule itself)
# otherwise we will face issues during training and really not learn much useful features.
# bidirection leads to 3 kinds of issues:
# Exposure Bias: During training, our model is fed the ground truth captions, 
# so it always has the correct previous words when predicting the next word. 
# But during inference, the model feeds its own predictions as input for the 
# next time step. If the model makes a mistake, this error can propagate and
# affect the prediction of subsequent words, leading to poor performance.
# Inconsistency between training and inference: in training, a bi-LSTM can use 
# future information (i.e., words that come later in the caption). But during 
# inference, when generating a caption word by word, future words are unknown.
# This discrepancy can lead to a drop in performance at test time.
# Overfitting: BiLSTMs have more parameters than unidirectional LSTMs, which can
# lead to overfitting, especially if the amount of training data is limited.
# also note that the image encoder section is extremely important, so is the imagefeature
# fusion with the caption. if we have problem in any of this two parts we wont be getting
# a satisfactory outcome. 
# !for example the 'input' method has a hard time developing a proper language model, probably
# !becasue the way image features are fused/utilized with other input-langauge related
# !features, the scale difference, distribution difference,etc might make this hard for the 
# lstm to utilize the information properly.
# so we get the best performance usng image-features as initial hidden_states. (it achieves 
# 36% in ht mode, while it only achieves 25% for input mode)
# so simply using a better base vision model can improve our results (test with resent50/simplenet5m)
# and better fusion enhances the results as well. 
# finally proper truncation drastically improves our results and failing to do so adversly affects it
# keeping these in mind helps us have a good outcome inshaallah.
# here are some experiments I did and their results.
#! without relu on the decoders output we achieved 26%acc
#! used relu before outputs comes out of lstm, 20.68, bad text genertion and semantic understanding
#! > no relu + with layernorm(took 4:19:53 to finish 20 epochs.) -> achieved 25.27%
#! use 29k vocab with default/or best config to see how much .lower affects performance
#! now lets use truncation and set it based on the majority of sequence length > 35.44
# the truncation not only boosted our accuracy by nearly 10%! (from 25 to 35)
# it also lowered our vram consumption from 9Gig down to 3.9gig! 
# and it lowered the training time (down to ~4:00 hours)
#! now using smaller vocab by using .lower() as the default ->
# the vram usage is at 3.7Gb, the trainig is faster becasue the # params is 10m less!
#(43m vs 34.5m! ), and the accuracy is higher right off the bat, (at 2 epochs 34.15 vs 35.52)
# ultimately it finished at 36.79% at 20 epochs. (time elapsed: 03:57:13.65)
# However, the description generation quality is not as good as the cased version(that is when we use all cases) the uncased version(.lower() version)
# in my opinion, it starts all the sentences with <unk>, which seems to be needing early stopping
# or a bit more regularization or training time.
# by the way this old paper is a good read https://openaccess.thecvf.com/content_cvpr_2018/papers/Cui_Learning_to_Evaluate_CVPR_2018_paper.pdf 
#! up till now we use 2 lstm layers, try it with 1 lstm layer.
# using truncation we can improve our results drastically especially if theres a relatively
# large difference between sequence lengths. we can see this in action in our case, although
# we used pytorches pack-padding features, so each batch is padded based on the largest sequence length
# we still face difficulty in training as theres large discrepency between average sequence length and
# the top few percent of long input seuqences in our dataset, this leads to the majority of the sequences
# have more paddings than the actual data and thus hurt the performance drastically. 
# therefore if we use a truncation length that better provides a sinal to noise ratio, we get
# much better results.
# based on our tokenization statistics, and bar plot, we chose 15 as a length that covers most of
# our samples. smaller trunc-len, like 12 may very well give us better accuracy, but 15 seems ok, we can always test
# also note that, when dealing with truncation, we need to pay attention to the majority of samples
# so we cover as many samples as we can without hurting the performance.
trunc_len=15
method = 'ht' # hidden_state or input 
epochs = 20
interval = 500
batch_size = 64
num_workers = 8


transformations_train = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])])

transformations_val = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

dt_train = COCODataset(coco_root,annotation_dir,
                       tokenizer=tokenizer, split='train',
                       transformations=transformations_train)

dt_val = COCODataset(coco_root,annotation_dir,
                     tokenizer=tokenizer, split='val',
                     transformations=transformations_val)

# data loaders
dl_train = DataLoader(dt_train, batch_size=batch_size, shuffle=True, 
                      pin_memory=True, num_workers=num_workers, 
                      collate_fn=OurColateFN(trunc_len,end_token))

dl_val = DataLoader(dt_val, batch_size=batch_size, pin_memory=True, 
                    num_workers=num_workers, 
                    collate_fn=OurColateFN(trunc_len,end_token))

model = EncoderDecoderImageCaption(vocab_size=tokenizer.vocab_size,
                                   encoder_backend=encoder_backend,
                                   encoder_projection_size=project_size, 
                                   embd_size=embd_size,
                                   hidden_size=hidden_size,
                                   num_layers=num_layers,
                                   decoder_dropout=dropout, 
                                   bidirectional=bidirectional,
                                   method=method)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=5, gamma=0.1)
# ignore the paddings in the loss 
# sidenote:
# the ignore_index argument in CrossEntropyLoss allows us to specify a target value that is
# ignored when computing the loss. basically when calculating the loss, any target value that
# equals the ignore_index is not considered. this is particularly useful in tasks like sequence
# generation or segmentation, where our sequences or images can be of different lengths or sizes,
# and we've used padding to make all sequences or images the same size.
# For example, in our case which is image captioning, we often pad your sequences with a special 
# <pad> token so that all sequences in a batch have the same length. However, we don’t want our 
# model to learn to predict the <pad> token, so you can set ignore_index to the index of the <pad>
# token in our vocabulary. This way, the <pad> tokens will be ignored when calculating the loss.
# and hopefully we get better results cuz our model can now pay more attention to what matters!
# side note 2 concerinig pack_padded_sequence
# even if we're using pack_padded_sequence with our LSTM, we still want to use ignore_index with
# our loss function to make sure we are not computing the loss with respect to the <pad> tokens
# in our target sequence. 
# when we use pack_padded_sequence before feeding our sequences into an LSTM, pytorch internally
# ignores the padding while computing the outputs and hence the gradients. 
# This means that the LSTM won’t consider the <pad> tokens during backpropagation, which is exactly
# what we want.
# However, when we compute the loss using something like CrossEntropyLoss, we're comparing the LSTM's
# output at each time step to our target sequence. If our target sequence has <pad> tokens 
# (which it likely will, since our input sequence had them), then we'll be computing the loss
# with respect to these <pad> tokens unless we tell our loss function to ignore them. 
# sidenote 3: also note that, when we add this, we may see our accuracy drop drastically
# what used to be around 50,60% at the first few iterations (let alone epochs) now is down
# to 19,22%. 
# the drop in accuracy after adding ignore_index is due to the change in the way accuracy 
# is calculated. Before, when we didn’t use ignore_index, the <pad> tokens were likely considered 
# in the accuracy calculation. Since the model often correctly predicts <pad> tokens (because there 
# are so many of them), this can artificially inflate the accuracy.
# When we added ignore_index, the <pad> tokens were ignored in the accuracy calculation. 
# This means only the "real" tokens are considered, which can lead to a more realistic 
# (and often lower) accuracy.
# So, while it might seem concerning to see a drop in accuracy, this new accuracy is likely a
# more accurate reflection of our model's performance. 
# It’s important to remember that accuracy isn't everything, especially in tasks like language
# modeling or image captioning. Other metrics like BLEU or METEOR can provide a more holistic 
# view of your model’s performance.
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.wtoi[tokenizer._pad])

# calculate blue score
def calculate_bleu_score(ref_caps, gen_caps):
    # convert list of idxs to list of words in each batch
    refs = tokenizer.batch_decode(ref_caps,remove_special_tokens=True)
    gens = tokenizer.batch_decode(gen_caps,remove_special_tokens=True)
    # calculate the bleu 
    return corpus_bleu([[ref] for ref in refs], [gen for gen in gens])


rogue = evaluate.load("rouge")
bleu = evaluate.load('bleu')
# now lets define the actual function for measuring these scores 
def calculate_scores(outputs, targets, tokenizer):
    
    # first convert them into string sequences
    predicted_texts = tokenizer.batch_decode(outputs, to_str=True ,remove_special_tokens=True)
    reference_texts = tokenizer.batch_decode(targets, to_str=True ,remove_special_tokens=True)

    # now lets use our evaluate objects to calculate the scores as a dictionary
    # where each key contains a rogue score (rogue-1, rogue-2 and rogueL respectively)
    rogue_scores = rogue.compute(predictions=predicted_texts, references=reference_texts)
    # multiply by 100
    rogue_scores = {k: (v*100) for k,v in rogue_scores.items()}
    # now lets calculate the bleu score 
    bleu_scores = bleu.compute(predictions=predicted_texts, references=reference_texts)
    
    return {**rogue_scores,
            "bleu":bleu_scores["bleu"]*100,
            "gen_len":bleu_scores["translation_length"]//len(targets)
           }

print(f'{device=}')
print(f'{epochs=}')
print(f'{trunc_len=}')
print(f'{model.method=}')
print(f'{model.encoder_backend=}')
print(f'{len(dl_train)=:,}')
print(f'{len(dl_val)=:,}')
print(f'{tokenizer.vocab_size=:,}')
print(f'num_layers = {model.num_layers}')
print(f'{model.embd_size=}')
print(f'{model.hidden_size=}')
print(f'params = {sum(p.numel() for p in model.decoder.parameters()):,}')
start = time.time()
#
# sidenote:
# for evaluating the accuracy of our model, we can use accuracy, but thats not really
# helpful, instead a metric called BLEU which stands for (Bilingual Evaluation Understudy)
# is used. 
# the BLEU score measures the similarity between the generated caption and the 
# reference captions, which are the ground truth. It does this by calculating 
# the n-gram overlap between the generated caption and the reference captions.
# An n-gram is a contiguous sequence of n items from a given sample of text or
# speech.
# The BLEU score lies between 0 and 1. A score of 1 means that the generated 
# caption perfectly matches one of the reference captions, while a score of 0 
# means there’s no overlap. A score of 0.6 or 0.7 is considered very good, as
# even two humans would likely come up with different sentence variants for a
# problem, and would rarely achieve a perfect match

for epoch in range(epochs):
    
    model.train()
    losses_train = []
    accs_train = []
    bleu_scores = []
    hidden_states = None
    for i,(imgs,captions,targets) in tqdm(enumerate(dl_train)):
        imgs, captions, targets = tuple(t.to(device) for t in (imgs, captions,targets))
        outputs,_ = model(imgs, captions, hidden_states)
        # print(f'{outputs.argmax(dim=-1).shape=}')
        # print(f'{captions.shape=}')
        # print(f'{captions}')
        # 
        # since we have our input in the form of (Batch,Timesteps,Classes),
        # and crossentropy expects (Batch,Classes,Timesteps), we need to permute
        # print(f'{outputs.shape=} {captions.shape=}')
        loss = criterion(outputs.permute(0,2,1), targets)
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
        losses_train.append(loss.item())
        accs_train.append((outputs.softmax(dim=-1).argmax(dim=-1)==targets).float().mean().item())
        bleu_scores.append(calculate_bleu_score(targets.tolist(), outputs.softmax(dim=-1).argmax(dim=-1).tolist()))
        #! train with this and see the scores
        # results = calculate_scores(targets.tolist(), outputs.argmax(dim=-1).tolist())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i%interval==0:
            results = calculate_scores(outputs.softmax(dim=-1).argmax(dim=-1).tolist(), targets.tolist(),tokenizer=tokenizer)
            print(f'[{epoch}/{epochs} iter:{i}/{len(dl_train)}] loss: {np.mean(losses_train):.4f} Accuray: {np.mean(accs_train)*100:.2f} lr: {scheduler.get_last_lr()[-1]:.1e}')
            # print(f'BLEU score: {np.mean(bleu_scores):.4f}')
            print('metrics: ', results)
    # update the lr    
    scheduler.step()

    with torch.no_grad():
        model.eval()
        losses_val=[]
        accs_val=[]
        bleu_scores_val = []
        hidden_states = None
        for i,(imgs, captions, targets) in tqdm(enumerate(dl_val)):
            imgs, captions, targets = tuple(t.to(device) for t in (imgs, captions,targets))
            outputs,_ = model(imgs, captions, hidden_states)
            # since we have our input in the form of (Batch,Timesteps,Classes),
            # and crossentropy expects (Batch,Classes,Timesteps), we need to permute 
            loss = criterion(outputs.permute(0,2,1), targets)
            losses_val.append(loss.item())
            accs_val.append((outputs.softmax(dim=-1).argmax(dim=-1)==targets).float().mean().item())
            # print(f'labels: {tokenizer.batch_decode(target_cap_val[:3].tolist(),remove_special_tokens=False)}')
            # print(f'output:{tokenizer.batch_decode(outputs[:3].argmax(dim=-1).tolist(),remove_special_tokens=False)}')
            
            # only calculate on validation, becasue its an expensive/time-consuming operation!
            bleu_scores_val.append(calculate_bleu_score(targets.tolist(), outputs.softmax(dim=-1).argmax(dim=-1).tolist()))

        print(f'{epoch}/{epochs} '
            f'train-loss/acc: {np.mean(losses_train):.4f}/{np.mean(accs_train)*100:.2f} '
            f'val-loss/acc: {np.mean(losses_val):.4f}/{np.mean(accs_val)*100:.2f}')
        # print(f'BLEU scores: train-bleu: {np.mean(bleu_scores):.4f} val-bleu: {np.mean(bleu_scores_val):.4f}')
        results = calculate_scores(outputs.softmax(dim=-1).argmax(dim=-1).tolist(), targets.tolist(),tokenizer=tokenizer)
        print('metrics-val: ', results)
        
hours, rem = divmod(time.time() - start, 3600)
minutes, seconds = divmod(rem, 60)
print(f"time elapsed: {int(hours):0>2}:{int(minutes):0>2}:{seconds:05.2f}")
# %%
with torch.no_grad():
    model.eval()
    losses_val=[]
    accs_val=[]
    bleu_scores_val = []
    hidden_states = None
    for i,(imgs, captions, targets) in tqdm(enumerate(dl_val)):
        imgs, captions, targets = tuple(t.to(device) for t in (imgs, captions,targets))
        outputs,_ = model(imgs, captions, hidden_states)
        # since we have our input in the form of (Batch,Timesteps,Classes),
        # and crossentropy expects (Batch,Classes,Timesteps), we need to permute 
        loss = criterion(outputs.permute(0,2,1), targets)
        losses_val.append(loss.item())
        accs_val.append((outputs.argmax(dim=-1)==targets).float().mean().item())
        print(f'labels: {tokenizer.batch_decode(targets[:3].tolist(),to_str=True, remove_special_tokens=False)}')
        print(f'output:{tokenizer.batch_decode(outputs[:3].argmax(dim=-1).tolist(),to_str=True, remove_special_tokens=False)}')
        
        # only calculate on validation, becasue its an expensive/time-consuming operation!
        bleu_scores_val.append(calculate_bleu_score(targets.tolist(), outputs.argmax(dim=-1).tolist()))

    print(f'{epoch}/{epochs} '
        f'train-loss/acc: {np.mean(losses_train):.4f}/{np.mean(accs_train)*100:.2f} '
        f'val-loss/acc: {np.mean(losses_val):.4f}/{np.mean(accs_val)*100:.2f}')
    print(f'BLEU scores: train-bleu: {np.mean(bleu_scores):.4f} val-bleu: {np.mean(bleu_scores_val):.4f}')


hours, rem = divmod(time.time() - start, 3600)
minutes, seconds = divmod(rem, 60)
print(f"time elapsed: {int(hours):0>2}:{int(minutes):0>2}:{seconds:05.2f}")

#%%
# so using the image features as the initial hidden_states, 
# we managed to achiev a very high accuracy as shown below:
# 13/20 train-loss: 1.0000 train-Accuracy: 100.00 val-loss: 0.0468 val-Accuray: 99.67
# it was also considerably faster to converge than using the image features
# as the first sequence of the input.
# 
# lets test this and see how it works 
def generate_caption(model, img_list, trans, max_length, tokenizer:Tokenizer,topk=3 ):
    fig,axs = plt.subplots(1,len(img_list),sharex='none',sharey='none')
    for i in range(len(img_list)):
        axs[i].imshow(img_list[i])
    plt.show()
    img_captions=[]
    
    with torch.no_grad():
        for img in img_list:
            model.eval()
            hidden_states = None
            output_lst = []
            caption_str = tokenizer._start
            device = next(model.parameters()).device

            image_features = model.encoder(trans(img).unsqueeze(0).to(device))
            caption = torch.tensor(tokenizer.encode(caption_str, False), device=device).view(1,-1).long()

            for i in range(max_length):
                outputs, hidden_states = model.decoder(image_features, caption, hidden_states)
                # grab topk words
                probs, indexes = outputs.softmax(dim=-1).topk(k=topk,dim=-1)
                probs = probs.view(probs.size(-1))
                indexes = indexes.view(indexes.size(-1))
                idx = indexes[torch.multinomial(probs.softmax(dim=-1),1, replacement=True)]
                caption_str = tokenizer.decode([idx.item()])[0]
                # if caption_str != tokenizer._start: 
                output_lst.append(caption_str)
                caption = torch.tensor(idx.item(), device=device).view(1,-1).long()
                if caption_str == tokenizer._end:
                    break
            img_captions.append(' '.join(output_lst))

        print(*img_captions,sep='\n')
        return img_captions

img1 = './pretty_mage1.jpeg'
img2 = './pretty_mage2.jpeg'
img3 = './pretty_mage3.jpeg'
#A couple of baseball player standing on a field.
img4='/media/hossein/SSD/mscoco_dataset/val2017/000000000872.jpg'
img5='/media/hossein/SSD/mscoco_dataset/val2017/000000001000.jpg'

trans = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

images = [Image.open(img) for img in [img1, img2,img3,img4,img5]]
generate_caption(model, images, trans, max_length=200, tokenizer=tokenizer,topk=3)
#%%

#%%
# %%
# now lets use transformers as the decoder part and see how much of a difference it makes
#tldr : it only performs best if we have a lot of data, or we use a pretrained transformer model!
# but for the sake of compeleteness , lets see how we can create a quick and simly transformer
# instead of our lstm!
# !also note that 
#! this is not the right way to do this, but here is it anyway, until I write it properly and
# remove this section
# there could be several reasons why our accuracy is low after replacing LSTM with a Transformer
# decoder in our model:
# Model Complexity: Transformers are more complex than LSTMs. 
# They have more parameters and thus require more data to train effectively.
# If your dataset is small, the Transformer might overfit to the training data, 
# leading to poor performance on the validation or test data.
# Long-Range Dependencies: Transformers are designed to handle long-range dependencies better
# than LSTMs. However, in some tasks like image captioning, the dependencies between elements
# might not be very long-range. In such cases, LSTMs might perform better.
# Training Stability: Training Transformers can be less stable than training LSTMs.
# we might need to adjust our learning rate, batch size, or other hyperparameters to stabilize 
# the training.
# Positional Encoding: Transformers use positional encodings to capture the order of the elements
# in the sequence. If these are not implemented correctly, it could affect the performance of the
# model.
# Incorrect Usage: Transformers work differently than LSTMs. If they are not used correctly, 
# it could lead to poor performance. For example, you need to ensure that the masks are applied
# correctly during training.
# Remember, replacing LSTM with Transformer is not a guaranteed way to improve performance. 
# It depends on various factors like the complexity of the task, the amount of data available, 
# etc
class EncoderDecoderImageCaption(nn.Module):
    
    def __init__(self,vocab_size, 
                 encoder_backend='simplenet',
                 decoder_backend='trans',
                 encoder_projection_size=4096, 
                 embd_size=512, 
                 hidden_size=512,
                 num_layers=1, 
                 decoder_dropout=0.0, 
                 bidirectional=False, 
                 method='ht',
                 nhead=4, 
                 num_tdlayers=4, 
                 num_telayers=4,
                 pad_idx=0) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.encoder_backend = encoder_backend.lower()
        self.decoder_backend = decoder_backend
        self.embd_size = embd_size
        self.hidden_size = hidden_size
        self.encoder_projection_size = encoder_projection_size
        self.num_layers = num_layers
        self.decoder_dropout = decoder_dropout
        self.bidirectional = bidirectional
        self.method = method
        self.nhead = nhead
        self.num_tdlayers = num_tdlayers
        self.num_telayers = num_telayers
        # this number is used to pad the input sequences
        # and we need it to create attention-mask for our
        # transformer layer so it doesnt atten to padded sections
        self.pad_idx = pad_idx
        self.encoder = Encoder(self.encoder_backend,
                               self.embd_size, 
                               self.encoder_projection_size)
        
        self.decoder = Decoder(self.vocab_size, 
                               self.embd_size,
                               self.hidden_size, 
                               self.num_layers,
                               self.bidirectional,
                               self.decoder_dropout, 
                               self.method)
        
        self.embd = nn.Embedding(self.vocab_size, embd_size)
        #! we may need to send the attention_mask to transformers to make it work properly!
        # passing in an `attention_mask` since our inputs are padded. 
        # https://huggingface.co/docs/transformers/troubleshooting#incorrect-output-when-padding-tokens-arent-masked. 
        self.transformer = nn.Transformer(d_model=self.embd_size,
                                          nhead=self.nhead,
                                          num_decoder_layers=self.num_tdlayers,
                                          num_encoder_layers=self.num_telayers,
                                          batch_first=True,
                                          #! setting this to true improves accuracy to
                                          # only face nans in loss.
                                          # seems this is not the proper way to do it.
                                          # so we start doing it the right way inshallah
                                          # tomorrow.
                                          norm_first=False)
        
        
        self.fc = nn.Linear(self.embd_size, vocab_size)
        # disable gradients for the encoder part, becasue its job is to give us img features
        # and it doesnt need to change, the lstm/decoder part however, needs to be updated though
        # so lets effectively freeze our encoder!(this consumes less vram and improves performance as well)
        for module in self.encoder.modules():
            module.requires_grad_(False)
    
    def forward(self, imgs, captions, hidden_states):
        image_features = self.encoder(imgs)
        if self.decoder_backend in ['t','tr','trans','transformer']:
            hidden_states = None
            # our TransformerDecoder in pytorch takes two inputs: the target sequence(target captions)
            # and the output from the last layer of the encoder (memory). 
            # Target Sequence (captions): This is the sequence that the decoder will use to generate 
            # the output. during training, this is our target sequence (i.e., the correct captions).
            # during inference, this would be the sequence generated so far.(this is a bit different than our lstm version)
            # Memory (memory/ or in ourcase imagefeatures): This is the usually output from the last layer of the 
            # encoder. it represents the encoded version of the input sequence 
            # (i.e., the image features in our case). and it acts like using image-features as hiddenstates in lstm
            # the TransformerDecoder uses these two inputs to generate the output sequence. 
            # It does this by applying attention mechanisms that allow it to focus on different parts 
            # of the input sequence when generating each word in the output sequence.
            # In our code, captions is the target sequence and img_feats is the memory. 
            # So, the line outputs = self.transformer.decoder(embds, img_feats.unsqueeze(1))
            # is using the Transformer decoder to generate the output sequence based on the captions
            # and the encoded image features.
            # we can do another round of processing using the encoder part(may not be effective
            # as when we use actual tokens.)
            # sidenote: 
            # for a transformer to perform well we need lots of data, much much more than when we
            # have lstm. so we wont get better performance like ths, unless we have a lot more data
            # or use a pretrained transformer model. we will see in a moment
            # img_feats = self.transformer.encoder(image_features)
            embds = self.embd(captions)
            # by the way we need to unsqueeze our img_feats to be compatible(be in 3d shape not 2d so 
            # we add an extra time-step dim to the Batch,Features and making it, B,1,F instead)
            # Flatten the features and add an extra dimension for the sequence length
            image_features = image_features.view(image_features.size(0), 1, -1)
            # print(f'{image_features.shape=}')
            # Repeat the features for each word in the caption
            image_features = image_features.repeat(1, captions.size(1),1)
            # print(f'{image_features.shape=}')
            outputs = self.transformer.decoder(embds, image_features)
            # and finally we need predictions so we have our classifier!
            outputs = self.fc(outputs)
        else:
            outputs, hidden_states = self.decoder(image_features, captions, hidden_states)
        return outputs, hidden_states


# lets test
x_img = torch.randn(size=(2,3,224,224))
x_captions = torch.randint(0,100,size=(2,30))
model = EncoderDecoderImageCaption(encoder_backend='simplenet',
                                   decoder_backend='trans',
                                   encoder_projection_size=4096,
                                   vocab_size=100,
                                   embd_size=512,
                                   hidden_size=512,
                                   nhead=4,
                                   num_tdlayers=8,
                                   num_telayers=8,
                                   num_layers=2,
                                   decoder_dropout=0.0,
                                   bidirectional=True,
                                   method='hidden_state')

outputs,_ = model(x_img, x_captions,None)
print(f'{outputs.shape=}')
      
# %% transformer training section
tokenizer = Tokenizer(captions_train, captions_val, use_lower=False)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
encoder_backend = 'resnet'
decoder_backend ='transformer'
num_tdlayers = 4
num_telayers = 4
nhead = 8
project_size = 2048 
end_token=tokenizer.encode(tokenizer._end)
embd_size = 512
hidden_size = 512
num_layers = 1
dropout = 0.1
bidirectional=False

trunc_len=15
method = 'ht' # hidden_state or input 
epochs = 20
interval = 500
batch_size = 64
num_workers = 8


transformations_train = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])])

transformations_val = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

dt_train = COCODataset(coco_root,annotation_dir,
                       tokenizer=tokenizer, split='train',
                       transformations=transformations_train)

dt_val = COCODataset(coco_root,annotation_dir,
                     tokenizer=tokenizer, split='val',
                     transformations=transformations_val)

# data loaders
dl_train = DataLoader(dt_train, batch_size=batch_size, shuffle=True, 
                      pin_memory=True, num_workers=num_workers, 
                      collate_fn=OurColateFN(trunc_len,end_token))

dl_val = DataLoader(dt_val, batch_size=batch_size, pin_memory=True, 
                    num_workers=num_workers, 
                    collate_fn=OurColateFN(trunc_len,end_token))

model = EncoderDecoderImageCaption(vocab_size=tokenizer.vocab_size,
                                   encoder_backend='simplenet',
                                   decoder_backend='trans',
                                   nhead=nhead,
                                   num_tdlayers=num_tdlayers,
                                   num_telayers=num_telayers,
                                   encoder_projection_size=project_size, 
                                   embd_size=embd_size,
                                   hidden_size=hidden_size,
                                   num_layers=num_layers,
                                   decoder_dropout=dropout, 
                                   bidirectional=bidirectional,
                                   method=method)

model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=5, gamma=0.1)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.wtoi[tokenizer._pad])

# calculate blue score
def calculate_bleu_score(ref_caps, gen_caps):
    # convert list of idxs to list of words in each batch
    refs = tokenizer.batch_decode(ref_caps,remove_special_tokens=True)
    gens = tokenizer.batch_decode(gen_caps,remove_special_tokens=True)
    # calculate the bleu 
    return corpus_bleu([[ref] for ref in refs], [gen for gen in gens])

print(f'{device=}')
print(f'{epochs=}')
print(f'{trunc_len=}')
print(f'{model.method=}')
print(f'{model.encoder_backend=}')
print(f'{len(dl_train)=:,}')
print(f'{len(dl_val)=:,}')
print(f'{tokenizer.vocab_size=:,}')
print(f'num_layers = {model.num_layers}')
print(f'{model.decoder_backend=}')
print(f'{model.nhead=}')

print(f'{decoder_backend=}')
print(f'{model.embd_size=}')
print(f'{model.hidden_size=}')
print(f'params = {sum(p.numel() for p in model.parameters()):,}')
start = time.time()
#
for epoch in range(epochs):
    
    model.train()
    losses_train = []
    accs_train = []
    bleu_scores = []
    hidden_states = None
    for i,(imgs,captions,targets) in tqdm(enumerate(dl_train)):
        imgs, captions, targets = tuple(t.to(device) for t in (imgs, captions,targets))
        outputs,_ = model(imgs, captions, hidden_states)
        loss = criterion(outputs.permute(0,2,1), targets)
        losses_train.append(loss.item())
        accs_train.append((outputs.argmax(dim=-1)==targets).float().mean().item())
        bleu_scores.append(calculate_bleu_score(targets.tolist(), outputs.argmax(dim=-1).tolist()))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i%interval==0:
            print(f'[{epoch}/{epochs} iter:{i}/{len(dl_train)}] loss: {np.mean(losses_train):.4f} Accuray: {np.mean(accs_train)*100:.2f} lr: {scheduler.get_last_lr()[-1]:.1e}')
            print(f'BLEU score: {np.mean(bleu_scores):.4f}')
    # update the lr    
    scheduler.step()

    with torch.no_grad():
        model.eval()
        losses_val=[]
        accs_val=[]
        bleu_scores_val = []
        hidden_states = None
        for i,(imgs, captions, targets) in tqdm(enumerate(dl_val)):
            imgs, captions, targets = tuple(t.to(device) for t in (imgs, captions,targets))
            outputs,_ = model(imgs, captions, hidden_states)
            # since we have our input in the form of (Batch,Timesteps,Classes),
            # and crossentropy expects (Batch,Classes,Timesteps), we need to permute 
            loss = criterion(outputs.permute(0,2,1), targets)
            losses_val.append(loss.item())
            accs_val.append((outputs.argmax(dim=-1)==targets).float().mean().item())
            # print(f'labels: {tokenizer.batch_decode(target_cap_val[:3].tolist(),remove_special_tokens=False)}')
            # print(f'output:{tokenizer.batch_decode(outputs[:3].argmax(dim=-1).tolist(),remove_special_tokens=False)}')
            
            # only calculate on validation, becasue its an expensive/time-consuming operation!
            bleu_scores_val.append(calculate_bleu_score(targets.tolist(), outputs.argmax(dim=-1).tolist()))

        print(f'{epoch}/{epochs} '
            f'train-loss/acc: {np.mean(losses_train):.4f}/{np.mean(accs_train)*100:.2f} '
            f'val-loss/acc: {np.mean(losses_val):.4f}/{np.mean(accs_val)*100:.2f}')
        print(f'BLEU scores: train-bleu: {np.mean(bleu_scores):.4f} val-bleu: {np.mean(bleu_scores_val):.4f}')

hours, rem = divmod(time.time() - start, 3600)
minutes, seconds = divmod(rem, 60)
print(f"time elapsed: {int(hours):0>2}:{int(minutes):0>2}:{seconds:05.2f}")
# %% validation test section
with torch.no_grad():
    model.eval()
    losses_val=[]
    accs_val=[]
    bleu_scores_val = []
    hidden_states = None
    for i,(imgs, captions, targets) in tqdm(enumerate(dl_val)):
        imgs, captions, targets = tuple(t.to(device) for t in (imgs, captions,targets))
        outputs,_ = model(imgs, captions, hidden_states)
        # since we have our input in the form of (Batch,Timesteps,Classes),
        # and crossentropy expects (Batch,Classes,Timesteps), we need to permute 
        loss = criterion(outputs.permute(0,2,1), targets)
        losses_val.append(loss.item())
        accs_val.append((outputs.argmax(dim=-1)==targets).float().mean().item())
        print(f'labels: {tokenizer.batch_decode(targets[:3].tolist(),to_str=True, remove_special_tokens=False)}')
        print(f'output:{tokenizer.batch_decode(outputs[:3].argmax(dim=-1).tolist(),to_str=True, remove_special_tokens=False)}')
        
        # only calculate on validation, becasue its an expensive/time-consuming operation!
        bleu_scores_val.append(calculate_bleu_score(targets.tolist(), outputs.argmax(dim=-1).tolist()))

    print(f'{epoch}/{epochs} '
        f'train-loss/acc: {np.mean(losses_train):.4f}/{np.mean(accs_train)*100:.2f} '
        f'val-loss/acc: {np.mean(losses_val):.4f}/{np.mean(accs_val)*100:.2f}')
    print(f'BLEU scores: train-bleu: {np.mean(bleu_scores):.4f} val-bleu: {np.mean(bleu_scores_val):.4f}')


hours, rem = divmod(time.time() - start, 3600)
minutes, seconds = divmod(rem, 60)
print(f"time elapsed: {int(hours):0>2}:{int(minutes):0>2}:{seconds:05.2f}")

#
#%% using huggingface for doing image captioning - pretrained models
# now lets use huggingface pipeline for imagecaptioning and the more indepth one 
# which we ourselevs create a model to do image captioning
import os
# to download files/images off of internet
import requests
# to parse urls and see if they are valid for added check
from urllib import parse
import numpy as np
from tqdm import tqdm
from PIL import Image

import evaluate

import torch
import matplotlib.pyplot as plt 

from datasets import load_dataset
import transformers as trans
from transformers import VisionEncoderDecoderModel, GPT2TokenizerFast, ViTImageProcessor\
                         ,Seq2SeqTrainingArguments,Seq2SeqTrainer

# introduction to huggingface: 4 hours
# https://www.youtube.com/watch?v=6NTHvcXAl90
#  Data processing for Causal Language Modeling 
# https://www.youtube.com/watch?v=ma1TrR7gE7I


# set device to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"

# load a fine-tuned image captioning model and corresponding tokenizer and image processor
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning").to(device)
tokenizer = GPT2TokenizerFast.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# lets create a few helper functions to grab an image using a url
# it comes handy when we dont have an image locally at the moment!

def get_image(url):
    return Image.open(requests.get(url=url,stream=True).raw)

# lets define a generator to create diverse description instead of simply using greedy sampling
# we can use the built-in generator, but lets see how we can create our own generator.
def generate(model, image, tokenizer, max_iter=1, topk=3):
    img_tensor = image_processor(images=image,return_tensors='pt').to(device)
    decoder_inputIdx = torch.zeros((1,1),dtype=torch.long, device=device)
    idx_lst = []
    for i in range(max_iter):
        predictions = model(pixel_values=img_tensor["pixel_values"], decoder_input_ids=decoder_inputIdx)
        # grab the last timestep logits
        logits = predictions['logits'][:,-1,:]
        probs, indexes = logits.topk(dim=-1, k=topk)
        probs, indexes = tuple(t.view(-1) for t in (probs, indexes))
        idx = torch.multinomial(input=probs.softmax(dim=-1), num_samples=1, replacement=True)
        actual_idx = indexes[idx]
        idx_lst.append(actual_idx.item())
        decoder_inputIdx = torch.cat((decoder_inputIdx, actual_idx.unsqueeze(0)), dim=-1)
                
    # now lets convert the idxs to words 
    output = tokenizer.decode(idx_lst, skip_special_tokens=True)
    return output

url = "https://static.thehoneycombers.com/wp-content/uploads/sites/6/2022/03/anime-demon-slayer-900x643.jpeg"
plt.imshow(get_image(url))

# image_processor returns a dictionary containing 'pixel_values' which is what we want
img = image_processor(get_image(url),return_tensors="pt").pixel_values.to(device)
# num_beams=1 is greedy sampling, by increasing num_beams, we can create more diverse description
# or using do_sample=True, top_k=3 we can create diverse outputs!
output = model.generate(img, max_length=100, do_sample=True, top_k=3)
caption = tokenizer.batch_decode(output, skip_special_tokens=True)
# now lets use our version
caption2 = generate(model, get_image(url), tokenizer, max_iter=10, topk=3)

print(*caption, sep='\n')
print(caption2, sep='\n')

# Huggingface also provides a family of easy to use Classes with the name AutoModelFor..
# which infers the application from the pretrained checkpoints, like AutoModelForImageClassification
# and give it a pretrained checkpoint for a classification model and it will instantaite it for us
# there is none for imagecaptioning so we use this method instead which is easy eough
#! check to see if Im missing something regarding this
#%%
# now lets see how we can finetune these models on our own and hopefully get a better model
# for our needs
# there are several encoders/decoders we can use to create our imagecaptioning model. 
# huggingface allows us easily use any transformer based models it hosts to create our model
# side note: A "base-sized" model is usually smaller than "large" or "huge" models, 
# but larger than "small" or "tiny" models. The exact number of parameters in a "base-sized" model can vary depending on the 
# specific architecture of the model.
# good to read: https://huggingface.co/docs/transformers/model_doc/vit
#
# light models: 
# "google/vit-base-patch16-224" is a ViT model pre-trained on ImageNet-21k and fine-tuned on ImageNet 2012.
# "google/vit-base-patch16-224-in21k"
#  google vit-tiny and vit-small variants: 
# "WinKawaks/vit-small-patch16-224"
# "WinKawaks/vit-tiny-patch16-224" 
# "facebook/deit-tiny-patch16-224" a tiny-sized DeiT (Data-efficient Image Transformers) model, which is a distilled version of the ViT model.
# "facebook/deit-small-patch16-224" a small-sized DeiT model.
# 
# Heavier Models:
# "facebook/deit-base-patch16-224" 
# "facebook/deit-base-patch16-384" 
# "google/vit-base-patch16-224-in21k" (pre-trained on ImageNet-21k).
#  
# and we also need to choose a decoder, and like the encoder we have a lot of choices here.
# here are a few well-known models we can use : 

# GPT Models:
# 'gpt2': GPT-2 is a very commonly used model and has been used in a variety of tasks, including image captioning which is what we are after here.
# 'gpt2-medium': This is a larger variant of GPT-2.
# 'gpt2-large': This is one of the largest GPT-2 variants we can use.
# 'gpt2-xl': This is the largest GPT-2 variant.
# 'gpt3': GPT-3 models are not publicly available for fine-tuning, however, there are 
# smaller variants like 'gpt3-small', 'gpt3-medium', and 'gpt3-large' that can be used.
# 
#
# BERT Models: paper https://arxiv.org/pdf/1810.04805.pdf
# Tiny Model:
# "prajjwal1/bert-tiny" :converted the official bert-tiny from tf to pytorch
# Base Models:
# "bert-base-uncased": This is a base-sized BERT model with uncased text(i.e all lower cased)
# "bert-base-cased": This is a base-sized BERT model with cased text(i.e. has capital letters aswell).
# "bert-base-multilingual-uncased": This is a base-sized BERT model that can handle text in multiple languages(104, farsi is supported aswell-dont use this model use the cased one instead by the way).
# "bert-base-multilingual-cased": This is a base-sized BERT model that can handle text in multiple languages and respects casing.
# Large Models:
# "bert-large-uncased": This is a large-sized BERT model with uncased text.
# "bert-large-cased": This is a large-sized BERT model with cased text.
# 
# RoBERTa models:(RoBERTa: A Robustly Optimized BERT Pretraining Approach. https://arxiv.org/abs/1907.11692)
# RoBERTa (short for "Robustly Optimized BERT Pretraining Approach") is a variant of the BERT model,
# which was developed by researchers at Facebook AI. 
# Like BERT, RoBERTa is a transformer-based language model that uses self-attention to process input 
# sequences and generate contextualized representations of words in a sentence.
# One key difference between RoBERTa and BERT is that RoBERTa was trained on a much larger dataset 
# and using a more effective training procedure. 
# In particular, RoBERTa was trained on a dataset of 160GB of text, which is more than 10 times larger 
# than the dataset used to train BERT. Additionally, RoBERTa uses a dynamic masking technique during 
# training that helps the model learn more robust and generalizable representations of words.
# RoBERTa has been shown to outperform BERT and other state-of-the-art models on a variety of 
# natural language processing tasks, including language translation, text classification, and 
# question answering. It has also been used as a base model for many other successful NLP models 
# and has become a popular choice for research and industry applications.
# The RoBERTa model was proposed in "RoBERTa: A Robustly Optimized BERT Pretraining Approach". It was released on October 29, 2019¹.
# refs
# RoBERTa - Hugging Face. https://huggingface.co/docs/transformers/model_doc/roberta.
# There are several RoBERTa models available on the Hugging Face Model Hub that are commonly used for various tasks⁴. Some of them include:
# "roberta-base": the base variant of the RoBERTa model.
# "roberta-large": the large variant of the RoBERTa model.
# "roberta-large-mnli": This is the large variant of the RoBERTa model fine-tuned on the MultiNLI dataset.
# Both BERT and RoBERTa are highly popular and widely used in the field of natural language processing. 
# RoBERTa builds upon BERT by modifying key hyperparameters, 
# removing the next-sentence pretraining objective, and training with much larger mini-batches 
# and learning rates. 
# These changes allow RoBERTa to often outperform BERT on a range of benchmark tasks. 
# However, the choice between BERT and RoBERTa can depend on factors like the availability 
# of computational resources, the size of the training data, and the specific requirements 
# of the task.
#
# good to read refs:
# What is the difference between BERT and Roberta. https://datascience.stackexchange.com/questions/97310/what-is-the-difference-between-bert-and-roberta.
# GPT-3, BERT, and RoBERTa | AI Model Analysis & Comparison. https://medium.com/@livajorge7/gpt-3-bert-and-roberta-ai-model-analysis-comparison-7dfab049367d.
# A review of pre-trained language models: from BERT, RoBERTa, to ELECTRA .... https://tungmphung.com/a-review-of-pre-trained-language-models-from-bert-roberta-to-electra-deberta-bigbird-and-more/.
# Fine-tuning RoBERTa for Topic Classification with Hugging Face ... - Medium. https://medium.com/@achillesmoraites/fine-tuning-roberta-for-topic-classification-with-hugging-face-transformers-and-datasets-library-c6f8432d0820.
# What's difference RobertaModel, RobertaSequenceClassification (hugging .... https://stackoverflow.com/questions/64383443/whats-difference-robertamodel-robertasequenceclassification-hugging-face.
# https://moon-ci-docs.huggingface.co/docs/transformers/pr_25830/en/model_doc/roberta.
# https://towardsdatascience.com/bert-roberta-distilbert-xlnet-which-one-to-use-3d5ab82ba5f8.

# sidenote:
# BERT and GPT are two different types of transformer-based models.
# BERT (Bidirectional Encoder Representations from Transformers) is a pre-training language 
# representation model created by Google in 2018. Unlike other NLP models that use unidirectional 
# attention flow, BERT uses bidirectional flow, which allows it to use context from both directions
# during processing. This allows the model to understand the meaning of words in context and, 
# in turn, better comprehend language structures.
# On the other hand, GPT (Generative Pre-trained Transformer) models, developed by OpenAI, 
# are unidirectional and rely on the decoder part of the transformer architecture to generate text.
# GPT models are autoregressive, meaning they generate sequences by predicting the next token in a 
# sequence given the previous tokens.
# So, while both BERT and GPT are transformer-based models and are used in natural language processing
# tasks, they have different architectures and use different strategies for understanding and generating
# text.
#
# to cut a long story short: BERT is a Transformer encoder, while GPT is a Transformer decoder
# Bert creates its output all atonce, while GPT generates one token at a time(outpout sequentially)
# because its autoregressive)
# this stackoverflow answer says it well (for training):
# BERT is a Transformer encoder, which means that, for each position in the input, 
# the output at the same position is the same token (or the [MASK] token for masked tokens), 
# that is the inputs and output positions of each token are the same.
# GPT is a Transformer 'decoder', which means that it is meant for autoregressive inference. 
# This means that the tokens in the input are shifted one position to the right with respect 
# to the output, that is, if the output is [the, dog, is, brown, </s>], the input is 
# [<s>, the, dog, is, brown, </s>]. (note the star/end tokens are the same here)
#
# also note that:
# both the models — GPT-3 and BERT have been relatively new for the industry, 
# but their state-of-the-art performance has made them the winners among other 
# models in the natural language processing field. 
# However, being trained on 175 billion parameters, GPT-3 becomes 470 times bigger
# in size than BERT-Large.
# Secondly, while BERT requires an elaborated fine-tuning process where users have to 
# gather data of examples to train the model for specific downstream tasks, 
# GPT-3’s text-in and text-out API allows the users to reprogram it using instructions
# and access it. Case in point — for sentiment analysis or question answering tasks, 
# to use BERT, the users have to train the model on a separate layer on sentence encodings.
# However, GPT-3 uses a few-shot learning process on the input token to predict the output 
# result.
# On general NLP tasks like machine translation, answering questions, complicated arithmetic calculations or learning new words, GPT-3 works perfectly by conditioning it with a few examples — few-shot learning. Similarly, for text generation as well, GPT-3 works on a few prompts to quickly churn out relevant outputs, with an accuracy of approximately 52%. OpenAI, simply, by increasing the size of the model and its training parameters created a mighty monster of a model.
# to understand the context of the word, BERT is trained on mask language model tasks, 
# where it randomly masks 15% of words in each sequence to predict the outcome. 
# Similarly, for sentence prediction, BERT is fed with a pair of sentences as input 
# and then gets trained on an added auxiliary task for prediction. 
# Here it processes both sentences involved to predict a binary label of the sentence prediction.
# taken from the link below:
# this link does a greta job at explaining the differences and highly recommend reading it 
# https://analyticsindiamag.com/gpt-3-vs-bert-for-nlp-tasks/
#
# !check the correctness of all of these claims/points below becsuae gpts are very good at nearly everything
# though I havent tested with bert, but we need to fact check these.
# and this link : https://www.makeuseof.com/gpt-vs-bert/
# While both are highly versatile NLP models, their architectural differences set them apart in
# a few ways. For instance, BERT is far more capable for the following use cases:
# Sentiment Analysis: BERT can better understand the overall sentiment of a given text 
# as it analyzes words in either direction.
# Named Entity Recognition: BERT is capable of recognizing different entities in a specific 
# piece of text, including locations, people, or organizations.
# Answering Questions: Because of its superior comprehension capabilities, 
# BERT is more capable of extracting information from text and answering questions accurately.
# 
# The GPT learning model is no slouch, either. While sentiment analysis might not be its forte, 
# GPT excels in several other applications:
# Content Creation: If you've used ChatGPT, you probably know about this already. 
# When it comes to content creation, GPT outsmarts most other models. 
# Just write a prompt, and it'll churn out a perfectly coherent (though not always accurate) response.
# Summarizing Text: Just copy-paste a large block of text in ChatGPT and ask it to summarize it. 
# It's capable of summarizing text while maintaining the core information.
# Machine translation: GPT can be fine-tuned for translating text from one language to another, 
# thanks to its ability to generate text based on context.

# good for reading/refs
# (1) GPT vs. BERT: What Are the Differences Between the Two Most ... - MUO. https://www.makeuseof.com/gpt-vs-bert/.
# (2) GPT-3 vs. BERT: Comparing the Two Most Popular Language Models - InvGate. https://blog.invgate.com/gpt-3-vs-bert.
# (3) BERT vs GPT: Comparison of Two Leading AI Language Models - 360DigiTMG. https://360digitmg.com/blog/gpt-vs-bert.
# (4) GPT-3 Vs BERT For NLP Tasks - Analytics India Magazine. https://analyticsindiamag.com/gpt-3-vs-bert-for-nlp-tasks/.
# (5) What is the difference between GPT blocks and BERT blocks. https://datascience.stackexchange.com/questions/87637/what-is-the-difference-between-gpt-blocks-and-bert-blocks.
#
# BART Models by facebook 
# (BART or Bidirectional and Auto-Regressive Transformers came in 2019
# by the paper "BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension".
# dont mistake  this with google Bard!:
# 'facebook/bart-base': This is the base variant of the BART model.
# 'facebook/bart-large': This is the large variant of the BART model.

# T5 Models by google:
# T5, short for Text-to-Text Transfer Transformer, is a natural language processing (NLP) model developed by Google
# obviously its It’s based on the Transformer architecture, which is highly effective in NLP tasks.
# T5 uses a text-to-text approach, where every task – including translation, question answering, and
# classification – is cast as feeding the model text as input and training it to generate some target text.
# The T5 model was presented in the paper "Exploring the Limits of Transfer Learning with a Unified 
# Text-to-Text Transformer" in 2019 .
# T5 comes in different sizes: t5-small, t5-base, t5-large, t5-3b, and t5-11b3. 
# Based on the original T5 model, Google has released some follow-up works: 
# T5v1.1, which is an improved version of T5 with some architectural tweaks, 
# and is pre-trained on C4 only without mixing in the supervised tasks.
# 't5-small': This is the small variant of the T5 model.
# 't5-base': This is the base variant of the T5 model.
# 't5-large': This is the large variant of the T5 model.
# 't5-3b': This is a larger variant of the T5 model.
# 't5-11b': This is the largest T5 variant.
# 
# There are much more uptodate models, in 2024 but these suffice for our introductory tutorial imho
# to see the list of more models see this : https://huggingface.co/docs/transformers/main/en/model_doc/gpt2
#
# refs, 1/20/2024
# undefined. https://huggingface.co/ankur310794.
# nlpconnect/vit-gpt2-image-captioning · Hugging Face. https://huggingface.co/nlpconnect/vit-gpt2-image-captioning.
# Image captioning - Hugging Face. https://huggingface.co/docs/transformers/main/en/tasks/image_captioning.
# What is Image-to-Text? - Hugging Face. https://huggingface.co/tasks/image-to-text.
# undefined. https://ankur3107.github.io/assets/images/image-captioning-example.png.
# undefined. https://twitter.com/ankur310794.
# undefined. http://github.com/ankur3107.
# undefined. https://www.linkedin.com/in/ankur310794.
# undefined. https://huggingface.co/datasets/Narsil/image_dummy/resolve/main/parrots.png.
#  Fine-tune a pretrained model - Hugging Face. https://huggingface.co/docs/transformers/training.
#  How to Train the Hugging Face Vision Transformer On a Custom Dataset. https://blog.roboflow.com/how-to-train-vision-transformer/.
#  kalpesh22-21/Image_Captioning_using_Hugging_Face - GitHub. https://github.com/kalpesh22-21/Image_Captioning_using_Hugging_Face.
#  The Illustrated Image Captioning using transformers. https://ankur3107.github.io/blogs/the-illustrated-image-captioning-using-transformers/.

# lets use ms swin vit which was trained on 14m images(imagenet21k)
encoder_model = "microsoft/swin-base-patch4-window7-224-in22k"
decoder_model = "gpt2" # "bert-base-uncased"

# 
# now in order to create our model, we need to somehow combine our encoder and decoder models
# we use VisionEncoderDecoderModel class to do this, along with from_encoder_decoder_pretrained()
# we simply enter the name of the models from huggingface hub, and we have our final model ready
# to go
model = trans.VisionEncoderDecoderModel.from_encoder_decoder_pretrained(encoder_model,
                                                                  decoder_model).to(device)
# next we need to create our tokenizer and image preprocessors for our decoder and encoders 
# respectively
# since our decoder model isknown, we can simply use AutoTokenizer and pass the decoder name
# and fetch the tokenizer. but the AutoTokenizer is very slower compared to the GPTTokenizerFast
# since we are using gpt2, we go on and use the faster version.
# tokenizer = trans.AutoTokenizer.from_pretrained(decoder_model)
tokenizer = trans.GPT2TokenizerFast.from_pretrained(decoder_model)
# so if we know what model we are going to use, if theres a fast version we go for that!
# now lets grab the image_processor for our vision encoder model:
image_processor = trans.ViTImageProcessor.from_pretrained(encoder_model)

# we need to make sure our models decoder_start_token_id and pad_token_id are initialized properly
# becasue the gpt2 model we are using, doesnt have decoder_start_token_id and pad_token_id instead
# it has bos_token_id and eos_token_id (short for begining of sequence and end of sequence)
if 'gpt2' in decoder_model:
    # Padding tokens were not used during the pre-training of GPT and GPT-2, therefore they have none
    # so we need to add them ourseleves.
    # sidenote: Because GPT2 and GPT are causal LM you don't need to pad shorter sentences in batches.
    # It is important though that the loss on these "unnecessary" tokens is not calculated.
    # sidenote2: 
    # "Causal Language Model" (LM) refers to a type of language model that generates a 
    # probability distribution for the next token in the sequence based on the tokens 
    # that have come before it. This is also known as "autoregressive" language modeling.
    # GPT-2 and GPT are examples of causal LMs.
    # In the context of training these models, we dont need to pad shorter sentences in 
    # batches because these models generate the next token based on the previous ones, 
    # and they do not need to consider the future tokens. Therefore, the length of the 
    # sentences doesnt need to be the same, unlike in some other types of models where
    # you need to pad the input sequences to ensure they have the same length.
    # sidenote2: 
    # we already knew that padding is used in batch processing of sequences to make all 
    # sequences in a batch the same length for efficient computation and this is particularly
    # important for models that are not autoregressive, like BERT, where the model looks 
    # at the entire sequence at once.
    # However, for autoregressive models like GPT-2, each token is predicted one at a time
    # based on the previous tokens. Therefore, these models technically dont require sequences
    # to be of the same length. Thats why we said that "we dont need to pad shorter sentences
    # in batches" for GPT/GPT-2 models.
    # But in practice, when we're preparing batches of sequences for training in a practical 
    # implementation, we often still pad sequences for efficiency reasons. 
    # The key is to ensure that the padding tokens are ignored during the computation of the 
    # loss function as we briefly experimented with in our previous example using lstms.
    # 
    # now back to our main issue here, since tokenizer doesnt have pad_token_id, 
    # we use eos_token_id as pad_token_id, if we dont do this we get this error 
    # message during trainig: 
    # ValueError: Asking to pad but the tokenizer does not have a padding token. 
    # Please select a token to use as `pad_token` `(tokenizer.pad_token = tokenizer.eos_token e.g.)` 
    # or add a new pad token via `tokenizer.add_special_tokens({'pad_token': '[PAD]'})`.
    # side note from : https://github.com/ludwig-ai/ludwig/pull/3735/files 
    # Notes:
    #- (geoffrey): gpt2 has no pad token. Recommendation is to use eos token instead.
    #    - https://github.com/huggingface/transformers/issues/2630#issuecomment-1290809338
    #    - https://github.com/huggingface/transformers/issues/2648#issuecomment-616177044
    #- (Justin): Using the EOS token in place of the pad token causes an issue with HF model.generate() when
    #    there are multiple examples in the batch.
    #    - https://github.com/facebookresearch/llama/issues/380#issuecomment-1716832417
    #    - Recommendation is to set a separate '[PAD]' or '<pad>' token.
    # but i guess the best way if the following doesnt work is to use add_special_tokens
    # and do a tokenizer.padding_side = "right" (cuz - GPT-2 is a model with absolute position embeddings so it’s usually advised to pad the inputs on the right rather than the left.) 
    # https://github.com/huggingface/transformers/issues/2630#issuecomment-1637601289
    # 
    tokenizer.pad_token =  tokenizer.eos_token
    # now lets initialize the other configs
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.decoder_start_token_id = tokenizer.bos_token_id
else:
    # For other decoders such as bert, we use cls_token_id instead as the start_token_id
    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    
# to check if everything is as expected lets print them 
print("Tokenizer pad_token:", tokenizer.pad_token)
print("Tokenizer pad_token_id:", tokenizer.pad_token_id)
print("Model config pad_token_id:", model.config.pad_token_id)

#%%
import torch.nn as nn
# our model is built, our preprocessors and tokenizers for encoder and decoders are now initialized
# and ready , what remains is the dataset
# we can use the cocodataset from huggingface, but since we already have it lets use our own dataset
# theres no point in redownloading 20+ gigabyte of data again when we have it already!
# lets define our dataset, it only needs two minor changes, one for transformations
# which we will use image_processor, and the other for tokenizer encoder section
# since we already implemented this, I remove the comments here
class COCODataset(nn.Module):
        
    def __init__(self, coco_root, 
                 annotation_dir,
                 train_imgs_dir='train2017',
                 val_imgs_dir='val2017',
                 split='train',
                 transformations=image_processor) -> None:
        super().__init__()
        self._coco_root = coco_root
        self._annotation_dir = annotation_dir
        self._captions_train_fname = 'captions_train2017.json'
        self._captions_val_fname = 'captions_val2017.json'
        self._train_dir = train_imgs_dir
        self._val_dir = val_imgs_dir
        self.split = split
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
        
        self.img_dict = {int(pathlib.Path(f).stem):f for f in glob.glob(os.path.join(self._coco_root, f"{self.imgs_folder}/*.jpg"))}
    
    def __getitem__(self, index):
        img_id = self.captions[index]["image_id"]
        img = Image.open(self.img_dict[img_id]).convert('RGB')
        img = self.transformations(img, return_tensors="pt").pixel_values
        # tokenize the caption and return it
        caption = self.captions[index]['caption']
        # note that usually we dont return the padded/truncated sequence from the dataset
        # its the dataloader's job to create a batch of sequences, and if some 
        # have different lengths, to make them work using sth like padding/truncation.
        # we do that using the colate_fn argument and pass a function that handles
        # these kinds of stuff, so the dataset need to return the actual data it contains,
        # batching chores are offloaded to the dataloader.
        return img.squeeze(0), caption 

    def __len__(self):
        # note that there are several captions per image, so we use captions length
        return len(self.captions)

dt_train = COCODataset(coco_root,annotation_dir=annotation_dir, split='train', transformations=image_processor)
dt_val = COCODataset(coco_root,annotation_dir=annotation_dir, split='val', transformations=image_processor)

print(f'{len(dt_train)=:,}')
print(f'{len(dt_val)=:,}')
img,caption = dt_train[1]
img_val,caption_val = dt_val[1]
print(f'{img_val.shape=}')
# now lets create our Colate_FN class! 
class OurColateFN():
    def __init__(self, truncation_length,tokenizer) -> None:
        self.truncation_length = truncation_length
        self.tokenizer = tokenizer

    def __call__(self, batch):
        # separate the images and captions
        images, captions = zip(*batch)
        # normalize our captions
        targets = self.tokenizer([caption for caption in captions], 
                             max_length=self.truncation_length,
                             padding="max_length",
                             truncation=True,
                             return_tensors="pt")

        # and finally create batches for images and labels
        imgs = torch.stack([image for image in images])
        labels = torch.stack([x for x in targets["input_ids"]])
        # return them as a dictionary as its customary in hf
        return {'pixel_values': imgs, 'labels': labels }

# now lets test 
trunc_len = 15
dl_train = DataLoader(dt_train, 5, shuffle=True, pin_memory=True, num_workers=0,collate_fn=OurColateFN(trunc_len,tokenizer=tokenizer))
dl_val = DataLoader(dt_val, 5, pin_memory=True, num_workers=0,collate_fn=OurColateFN(trunc_len,tokenizer=tokenizer))

print(f'{len(dl_train)=:,}')
print(f'{len(dl_val)=:,}')
# reminder!
# note that we are dealing with dictionaries now, simply iterating a dictionary will give us the keys only
# so doing sth like imgs, labels = next(iter(dl_train)) would only return the keys and store them into
# imgs and labels respectively.
data = next(iter(dl_train))
# to actually get the key-value pair, we simply need to use .items() and then we are good to go!
imgs, targets = next(iter(dl_val)).items()
print(f'{data["labels"]=}')
print(f'{targets=}')
#%%
# now everything seems to be complete, except the fact that we need to have something for
# evaluating our model output. previoulsy we simply used accuracy which is not a good metric
# for this kind of work, as different sequences/text can be equally correct but not identical
# necessarily, this would result in lower accuracy becasue the output is not identical to the label
# but infact is pretty good. so we need to use a better metric.
# we used belu but there are more. lets see what we can use here:
# There are a lot of metrics that are currecnly used in the nlp realm 
# but the most common ones are as follows :
# BLEU: It evaluates the n-gram overlap between the reference caption and the generated caption
#       and gives a more balanced evaluation of content similarity and fluency. 
#       It is calculated by computing the precision of the generated text with 
#       respect to the reference text, and then taking the geometric average of
#       the n-gram precisions (such as unigram, bigram, 3-gram, 4-gram, etc.). 
#       The most common version is BLEU-4, which considers the unigram to 4-gram
#       overlap average. It is widely used in the machine translation task. 
#       Check this quick YouTube video to learn more about it https://www.youtube.com/watch?v=M05L1DhFqcw&ab_channel=HuggingFace or 
#       this tutorial : https://thepythoncode.com/article/bleu-score-in-python
#       
# ROUGE: It calculates the percentage of common tokens between the generated text and 
#        the reference text, with longer sequences given more weight. Like BLEU, 
#        the score is between 0 and 1, where 1 is the perfect match and 0 is the
#        poorer match. ROUGE can be calculated using different n-gram orders, 
#        such as ROUGE-1 (unigrams, or just single token), ROUGE-2 (bigrams), 
#        or ROUGE-L (longest common subsequence). 
#        It is also common in machine translation and text summarization tasks. 
#        The most common version we'll use for image captioning is ROUGE-L. 
#        Check this YouTube video to learn more about it : https://www.youtube.com/watch?v=TMshhnrEXlg&ab_channel=HuggingFace
#
# METEOR: A combination of ROUGE and BLEU, which also considers word alignments, 
#         synonymy, and other factors.
# CIDEr: A metric that measures similarity between the generated text and reference
#        texts using a consensus-based approach that takes into account the agreement
#        among multiple human annotators.
# SPICE: A semantic-based metric that computes a graph-based representation of the 
#        captions and compares them based on their content. This metric is invented
#        for image captioning specifically.
# 
# we can use evaluate python module to calculate these scores easily
# we will be using rogue and bleu metrics here 
rogue = evaluate.load("rouge")
bleu = evaluate.load('bleu')
# now lets define the actual function for measuring these scores 
def calculate_scores(outputs, targets, tokenizer):
    
    # first convert them into string sequences
    predicted_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    reference_texts = tokenizer.batch_decode(targets, skip_special_tokens=True)

    # now lets use our evaluate objects to calculate the scores as a dictionary
    # where each key contains a rogue score (rogue-1, rogue-2 and rogueL respectively)
    rogue_scores = rogue.compute(predictions=predicted_texts, references=reference_texts)
    # multiply by 100
    rogue_scores = {k: (v*100) for k,v in rogue_scores.items()}
    # now lets calculate the bleu score 
    bleu_scores = bleu.compute(predictions=predicted_texts, references=reference_texts)
    
    return {**rogue_scores,
            "bleu":bleu_scores["bleu"]*100,
            "gen_len":bleu_scores["translation_length"]//len(targets)
           }

# for trainer class, the result ofthe model is a EvalPredictions object, so we have this wrapper
def calculate_scores2(eval_predictions):
    return calculate_scores(eval_predictions.predictions, eval_predictions.label_ids, tokenizer=tokenizer)
#%% we have everything in place and we can now commence training!!
# for training just like what we saw previously in rnn section/text generation part
# we can use the huggingface trainer class which makes it a breeze to train or use 
# good old pytorch training loop. 
# we first train using trainer class and then also see how we can train this using 
# pytorch and whether its any different! 
from functools import partial

batch_size = 16
epochs=2
interval=2000
max_length = 15
num_workers=8
training_args = trans.Seq2SeqTrainingArguments(output_dir='./results_imgcaptioning-swin-gpt2',
                                               do_train=True,
                                               do_eval=True,
                                               per_device_train_batch_size=batch_size,
                                               per_device_eval_batch_size=batch_size,
                                               num_train_epochs=epochs,
                                               predict_with_generate=True,
                                               # evaluate the model at each eval_steps
                                               # other options are ['no', 'steps', 'epoch']
                                               evaluation_strategy='steps',
                                               eval_steps=interval,
                                               logging_steps=interval,
                                               save_steps=interval,  #save the model at intervals
                                               logging_dir='./results/logs',)

# now lets train 
trainer = trans.Seq2SeqTrainer(model = model, 
                               args=training_args,
                               train_dataset=dt_train,
                               eval_dataset=dt_val,
                               data_collator=OurColateFN(truncation_length=max_length, tokenizer=tokenizer),
                               tokenizer=tokenizer,
                               compute_metrics=calculate_scores2
                               )
# lets swap the dataloaders 
# dl_train = DataLoader(dt_train, batch_size, shuffle=True, pin_memory=True, num_workers=num_workers,collate_fn=OurColateFN(max_length, tokenizer))
# dl_val = DataLoader(dt_val, batch_size, pin_memory=True, num_workers=num_workers,collate_fn=OurColateFN(max_length, tokenizer))
# create a simple lambda that returns a dataloader for each
# trainer.get_train_dataloader = lambda: dl_train
# trainer.get_eval_dataloader = lambda: dl_val
# now lets train!
trainer.train()

# %%
# now to train it ourselves we would need a few more things including a criterion/loss
# an optimizer, and dataloaders. 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = trans.VisionEncoderDecoderModel.from_encoder_decoder_pretrained(encoder_model,decoder_model).to(device)
tokenizer = trans.GPT2TokenizerFast.from_pretrained(decoder_model)
image_processor = trans.ViTImageProcessor.from_pretrained(encoder_model)

if 'gpt2' in decoder_model:
    tokenizer.pad_token =  tokenizer.eos_token
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.decoder_start_token_id = tokenizer.bos_token_id
else:
    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    
# to check if everything is as expected lets print them 
print("Tokenizer pad_token:", tokenizer.pad_token)
print("Tokenizer pad_token_id:", tokenizer.pad_token_id)
print("Model config pad_token_id:", model.config.pad_token_id)

criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
optimizer = torch.optim.AdamW(model.parameters(), lr = 0.00001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=1, gamma=0.1)
epochs = 2
batch_size = 8
num_workers=8
max_length =15
interval = 5000
dl_train = DataLoader(dt_train, batch_size, shuffle=True, pin_memory=True, num_workers=num_workers,collate_fn=OurColateFN(max_length, tokenizer))
dl_val = DataLoader(dt_val, batch_size, pin_memory=True, num_workers=num_workers,collate_fn=OurColateFN(max_length, tokenizer))

for epoch in range(epochs):
    model.train()
    for i, (data) in tqdm(enumerate(dl_train),leave=False):
        imgs, labels = data["pixel_values"], data["labels"]
        imgs,labels = tuple(t.to(device) for t in (imgs, labels))
        # note that predictions contains the calculated loss as well, but 
        # try to do everything ourseleves here.
        predictions = model(pixel_values=imgs, labels=labels)
        logits = predictions["logits"]
        # we could also write 
        # loss = predictions.loss
        loss = criterion(logits.permute(0,2,1) , labels)
        metrics = calculate_scores(logits.softmax(dim=-1).argmax(dim=-1), labels,tokenizer=tokenizer)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i%interval==0:
            print(f'{loss.item():.4f} {metrics}')
        
    scheduler.step()
    
    with torch.nograd():
        model.eval()
        losses = []
        for i,data in tqdm(enumerate(dl_val),leave=False):
            imgs, labels = data["pixel_values"], data["labels"]
            imgs,labels = tuple(t.to(device) for t in (imgs, labels))
            predictions = model(pixel_values=imgs,labels=labels)
            logits = predictions["logits"]
            loss = criterion(logits.permute(0,2,1) , labels)
            metrics = calculate_scores(logits.softmax(dim=-1).argmax(dim=-1), labels,tokenizer=tokenizer)
            losses.append(loss.item())
            print(f'{np.mean(losses):.4f} {metrics}')
            
            
# %%
