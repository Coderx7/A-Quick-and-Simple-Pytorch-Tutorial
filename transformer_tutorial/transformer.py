#%%
# In the name of God the most compassionate the most merciful
# in this section we will have a look at how a transformer model works,
# in this section we will implement the attention mechanism
# which is the foundation of transformers, and see 
# how it works and lean more about it (self-attention, masked-attention, 
# scaled-attention, cross-attention, multi-head attention, etc)
# 
# before we delve into the implementation details, we have to take a detour
# and discuss some underlying concepts and techniques involved.
# 
# so how are we going about this? we need to create a language model
# and then progressively imporve it with attention mechanism.
# we will basically be creating a kind of chatgpt (minus its chat capability and
# its obviously great perormance :)) 
# to keep this as simple as possible and not lose track of the 
# important concepts involved, we can use our initial bigram model
# as the base model and work on improving that with attention.
# so lets start
#
# first lets import the basic stuff 

# for type hints
from collections.abc import Iterable
import math
import random 
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# lets setup the manual seeds or determinstic output
torch.manual_seed(255)
np.random.seed(255)
random.seed(255)

# first lets read the dataset, we are going to use tinyshalespear which
# is around 1million characters (around 1MB in size), and our model is
# supposed to create texts resembling this dataset.
dataset = []
with open('./data/tiny_shakespear.txt','r') as file:
    # this time we read the whole text as one big str
    dataset = file.read()
    
print(f'{dataset[:100]=}') 
# prints:
# dataset[:100]='First Citizen:\nBefore we proceed any further, hear me speak.\n\nAll:\nSpeak, speak.\n\nFirst Citizen:\nYou'
# 
# now lets create our atoi and itoa dictionaries for mapping
# before that lets create our unique character list, which is
# infact our vocabulary or vocab for short
vocab_list = sorted(set(''.join(dataset)))
vocab_size = len(vocab_list)
# lets create our mapping dictionaries now, we are basically going to use them
# for tokenization, converting our input into tokens which here are characters 
# and ultimately their integer representations
# so tokenization simply put, refers to converting an input into a list of numbers
# based on some creteria. here, we just use a simple index number to map our 
# vocabulary characters, which represent all characters our model can use to
# generate text, to integer representation. 
# for real-world applications, we usually use libraries such as sentecepeace 
# by google which works on a subword level which means, it doesnt encode the 
# whole word, neither does it encode based on individual characters, it use sth in between. 
# (from its github repo: https://github.com/google/sentencepiece )
# SentencePiece is a re-implementation of sub-word units, an effective way to 
# alleviate the open vocabulary problems in neural machine translation. 
# SentencePiece supports two segmentation algorithms, byte-pair-encoding 
# (BPE) [Sennrich et al.] and unigram language model [Kudo.]. 
#
# sidenote:--------------------------------------------------------------------------
# (For the reference:  
# Byte Pair Encoding (BPE) is a subword tokenization technique used in natural
# language processing (NLP) and text processing tasks. It is a data compression
# algorithm that splits words into subword units. 
# BPE is commonly used in tasks such as machine translation, text generation, 
# and language modeling. The basic idea behind BPE is to iteratively merge 
# the most frequent pairs of characters or subword units in a corpus to 
# create a new subword vocabulary. This merging process is based on the 
# statistical properties of the corpus, specifically the frequency of character
# or subword pairs.
#
# Here's roughly how the BPE algorithm works:
# first we initialize the vocabulary with all the characters or subwords in the corpus
# then we calculate the frequency of each character or subword in the corpus,
# after that until we reach the desired vocabulary size or a maximum number of
# iterations we keep doing the following steps:
#  1. Find the most frequent pair of characters or subwords in the corpus.
#  2. Merge the pair into a new subword unit by concatenating them.
#  3. Update the corpus by replacing occurrences of the merged pair with 
#     the new subword unit.
#  4. Update the vocabulary and frequency counts based on the new corpus.
# The final vocabulary is then the set of subword units obtained after the desired
# number of iterations or vocabulary size is reached.
#
# Why do we care about BPE? 
# BPE allows for the representation of both known and unknown words in a corpus.
# It is effective in handling out-of-vocabulary (OOV) words and reducing the 
# vocabulary size, which can improve the efficiency and performance of NLP models.
# By breaking down words into subword units, BPE can capture morphological and 
# semantic information more effectively, especially for languages with complex word 
# formations and agglutinative structures.
# 
# tiktokenize repository explains BPE in a rather friendlier way: 
# Models don't see text like you and I, instead they see a sequence of numbers
# (known as tokens). 
# Byte pair encoding (BPE) is a way of converting text into tokens. It has a 
# couple desirable properties:
# It's reversible and lossless, so you can convert tokens back into the original text
# It works on arbitrary text, even text that is not in the tokeniser's training data
# It compresses the text: the token sequence is shorter than the bytes corresponding
# to the original text. 
# On average, in practice, each token corresponds to about 4 bytes.(i.e. 4 characters!)
# It attempts to let the model see common subwords. For instance, "ing" is a common
# subword in English, so BPE encodings will often split "encoding" into tokens like 
# "encod" and "ing" (instead of e.g. "enc" and "oding"). Because the model will then
# see the "ing" token again and again in different contexts, it helps models generalise
# and better understand grammar.
#
# Whats a Unigram lanugage model?
# A unigram language model however, is a type of statistical language model that 
# predicts the probability of each word in a sequence independently, based solely
# on the frequency of occurrence of individual words in the training data. 
# It does not consider the context or the order of the words in the sequence.
# In a unigram language model, the probability of a particular word is estimated
# by counting the frequency of that word in the training corpus and normalizing it
# by the total number of words in the corpus. 
# The probability of a sequence of words is then calculated by multiplying the 
# probabilities of each individual word in the sequence.
# For example, consider the sentence "I love to eat pizza." In a unigram language model,
# the probability of this sentence would be calculated as the product of the probabilities
# of each word which would be: 
# P(I) * P(love) * P(to) * P(eat) * P(pizza).
# 
# Unigram models are the simplest form of language models and do not capture any 
# contextual information or dependencies between words. They are often used as a 
# baseline or reference model in natural language processing tasks. 
# While unigram models are not very accurate in capturing the complexities of natural 
# language, they can be computationally efficient and useful in tasks where the context
# is less important, such as certain text classification tasks or language generation tasks
# where only the frequency of individual words matters.
# 
# Openai/chatgpt for example uses its own tokenizer, called tiktoken 
# (https://github.com/openai/tiktoken) there are other tokenizers as well.
# huggingface has a good article on tokenizers (recommend it): 
# https://huggingface.co/docs/transformers/main/tokenizer_summary 
# 
# -------------------------------------------------------------------------------
#
# its a given that, each model only works with the tokenizer by which it was 
# used to train. so BERT, DistilBERT, and Electra only work with wordpiece, 
# while chatgpt uses titoken and we! use simple characters-indexes! 
# also note that the choice of tokenizer obviously affects our vocab size and
# model overhead/performance as well. 
# for example lets first implement our tokenizer and then compare it with sth
# like tiktoken module
atoi = {c:n for n,c in enumerate(vocab_list)}
itoa = {n:c for c,n in atoi.items()}
print(f'{vocab_size=}, {vocab_list=}')
print(f'{atoi=}')
print(f'{itoa=}')
# prints 
# vocab_size=65, vocab_list=['\n', ' ', '!', '$', '&', "'", ',', '-', '.', '3', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
# atoi={'\n': 0, ' ': 1, '!': 2, '$': 3, '&': 4, "'": 5, ',': 6, '-': 7, '.': 8, '3': 9, ':': 10, ';': 11, '?': 12, 'A': 13, 'B': 14, 'C': 15, 'D': 16, 'E': 17, 'F': 18, 'G': 19, 'H': 20, 'I': 21, 'J': 22, 'K': 23, 'L': 24, 'M': 25, 'N': 26, 'O': 27, 'P': 28, 'Q': 29, 'R': 30, 'S': 31, 'T': 32, 'U': 33, 'V': 34, 'W': 35, 'X': 36, 'Y': 37, 'Z': 38, 'a': 39, 'b': 40, 'c': 41, 'd': 42, 'e': 43, 'f': 44, 'g': 45, 'h': 46, 'i': 47, 'j': 48, 'k': 49, 'l': 50, 'm': 51, 'n': 52, 'o': 53, 'p': 54, 'q': 55, 'r': 56, 's': 57, 't': 58, 'u': 59, 'v': 60, 'w': 61, 'x': 62, 'y': 63, 'z': 64}
# itoa={0: '\n', 1: ' ', 2: '!', 3: '$', 4: '&', 5: "'", 6: ',', 7: '-', 8: '.', 9: '3', 10: ':', 11: ';', 12: '?', 13: 'A', 14: 'B', 15: 'C', 16: 'D', 17: 'E', 18: 'F', 19: 'G', 20: 'H', 21: 'I', 22: 'J', 23: 'K', 24: 'L', 25: 'M', 26: 'N', 27: 'O', 28: 'P', 29: 'Q', 30: 'R', 31: 'S', 32: 'T', 33: 'U', 34: 'V', 35: 'W', 36: 'X', 37: 'Y', 38: 'Z', 39: 'a', 40: 'b', 41: 'c', 42: 'd', 43: 'e', 44: 'f', 45: 'g', 46: 'h', 47: 'i', 48: 'j', 49: 'k', 50: 'l', 51: 'm', 52: 'n', 53: 'o', 54: 'p', 55: 'q', 56: 'r', 57: 's', 58: 't', 59: 'u', 60: 'v', 61: 'w', 62: 'x', 63: 'y', 64: 'z'}

# lets also create two helper functions to convert a list of these to the other part
def encode(characters:Iterable[str] ):
    return [atoi[c] for c in characters]

def decode(token_lst:Iterable[int]):
    return [itoa[n] for n in token_lst]

# lets test these 
print(f'{encode(dataset[:10])}')
print(f'{decode(encode(dataset[:10]))}')
# prints 
# [18, 47, 56, 57, 58, 1, 15, 47, 58, 47]
# ['F', 'i', 'r', 's', 't', ' ', 'C', 'i', 't', 'i']

# now lets try tiktoken
try: 
    import tiktoken
except:
    import os 
    os.system('pip install tiktoken')
    import tiktoken
    
# lets use the tokenizer for gpt2 model, this line downloads the gpt2 tokenizer and allows us to use it
encoder_gpt = tiktoken.get_encoding('gpt2')
# lets view its vocab size:
print(f'{encoder_gpt.n_vocab=:,}') # prints encoder_gpt.n_vocab=50,257
# now lets see how the encoding looks like here
print(f'{encoder_gpt.encode(dataset[:10])=}') # prints [5962, 327, 8846] 
# which is very intresting! while for us its 10 numbers/tokens for 10 characters(vocab size 65),
# this is 3 here with vocab size of 50,000!
# so we can have a short vocab size, at the expense of a larger sequence size, 
# or a large vocabsize and smaller sequence size.
# so thats why in practice, these subwords tokenizers are used for real world applications. 
# but for our case we stick to our primitive tokenizer to keep things as simple as possible. 
# 
# Ok, so far so good. now we need to create a dataset, like before we need to have
# an input/label pair our input is a series of characters, (a sequence of some length),
# and our label is the next character sicne we are using a simple bigram model,
# given a single character, we want the probablity of what comes next. 
# but, we also want to incorporate attention, and we want a context for our prediction,
# we want to be able to look at the past, the past characters, and based on that do sth. 
# this is the essence of attention (although this is not accurate, but for now let assume
# this is the case, we will elaborate on this and expand this metaphor and reasoning inshallah)
# 
# lets first tokenize the whole dataset or corpus as its usually called in nlp nomenclature(well corpus 
# usually is made of several bodies of text! but anyway you get the idea!)!
data = encode(dataset)
# since we are using pytorch lets convert that to a tensor
data_tensor = torch.tensor(data)
# as usual lets inspect the data
print(data_tensor[:100])
# now lets create a train/val split 
train_length = int(0.9 * len(data_tensor))
# note that we do not shuffle the data here like before, because we are not dealing
# with a list of samples!
# like names, here we are dealing with the whole text, and if we shuffle it like that,
# we just destroy it making it into a batch of random characters! which would not be 
# useable for us anymore. we are trying to learn the underlying semantic and relationships
# hidden in our data, and randomizing them like that just destroys those information! 
# to see this in action try this
_data_copy = data.copy()
random.shuffle(_data_copy)
print(''.join(decode(_data_copy[:100])))
# prints:
# Osetow Pr 
# oa geEeM rhnalelrt   aAdt Ktsw pTpeGxli
# oasEliMe
# vsieeose
# etttbSdnctiras  hnr:yhtayiuCv g
# so we simple divide the data normally
train_data = data_tensor[:train_length]
val_data = data_tensor[train_length:] 
# note that since we are planning on creating a simple transformer model, 
# we usually dont feed the whole dataset becaue its prohibitevly computation intensive,
# instead, what happens in practice is that we, grab chunks of data from the dataset
# and feed it to the model. these chunks, of course have a length, what length?
# we usually specify a maximum_length for the input on which our transformer model works.
# this maximum_length is usually refered to as block_size or more famously context_size.
# we had previously used different context_sizes and this is not really that different,
# lets for example define a context_size of 8
# block_size and context_size are interchangable
block_size = context_size = 8 
print(f'{train_data[:block_size]=}')
# which prints:
# train_data[:block_size]=tensor([18, 47, 56, 57, 58,  1, 15, 47])
# 
# one intresting observation we can make here is that, as simple as this seemingly ordindary 
# list of numbers looks, this sequence of numbers, actually contains several examples. 
# if you think about it, each number, is a token, representing a character (or word, subword, etc),
# and it shows what comes after what. basically not only it shows several pairs so to speak, it also
# shows, which characters are more likely to come, before a specific character comes later. 
# what we are actually going to do is that, we are going to train all of these characters 
# simultaneously!
# notice that in this example, we have 7 examples in a sequence of 8 characters:
# lets elaborate on this more. consider [18, 47, 56, 57, 58,  1, 15, 47]) as input:
# 1-in the context of 18, the next character is 47
# 2-in the context of 18,47, the next character is 56
# 3-in the context of 18,47,56, the next character is 57
# 4-in the context of 18,47,56,57 the next character is 58
# 5-in the context of 18,47,56,57,58 the next character is 1
# 6-in the context of 18,47,56,57,58,1 the next character is 15
# 7-and finally, in the context of 18,47,56,57,58,1,15 the next character is 47
# so this is infact 8 7 individual example embedded in a single context, 
# since we want the context_size to be 8, then we should grab one more character to have 
# 8 contexts
# lets visualize this example in code
# lets have a typical sequence of size 8 
x = train_data[:block_size]
y = train_data[1:block_size+1]
# what would be our label? we want the next character so it would be the previous locations +1
# basically offset the input by 1!
# thats why we started from 1, becasue the input started from 0, and since the input
# ended at block_size its label(next character) would obviously be block_size+1, 
# pretty obvious right?
#
# lets print this for better understanding 
for pos in range(block_size):
    # note that since we are using slice, x[:i+1], gives us 0 up to ith element (inclusive)
    # this should be obvious! but I said it in case you forgot!!
    print(f'input is {x[:pos+1].tolist()} label is {y[pos]}')
# prints 
# input is [18] label is 47
# input is [18, 47] label is 56
# input is [18, 47, 56] label is 57
# input is [18, 47, 56, 57] label is 58
# input is [18, 47, 56, 57, 58] label is 1
# input is [18, 47, 56, 57, 58, 1] label is 15
# input is [18, 47, 56, 57, 58, 1, 15] label is 47
# input is [18, 47, 56, 57, 58, 1, 15, 47] label is 58
# so as you can see we started from context_size of 1 up to context_size of 8.
# note that we do not do this simply for the efficancy aspect of it, 
# but also, when we feed these to our transformer model, we are making sure 
# that the transformer model gets used to see all combinations of our input 
# as well (from the context size of 1 up tp the context size of 8). 
# this way not only it sees the whole context_size as we initially expected
# but also all sequences before it, and basically what consituted to make the sample. 
# this allows us to later on, at test time be able to create sequences as small
# as context_size of only 1 up to the max_length which is our context_size of 8
# in our case(and more).
# so by doing this, the model can learn how to predict/create/genrate text up to
# context_size, and after it reached that, we have to truncate it, becausse the 
# model never recieves more than the context_size as input when its predicting 
# the next character.(we can continue generating infintely, but really, what the
# model does, is to always generate the next character based on the 'last' context_size 
# number of tokens. if context size is 8, only the last 8 characters are taken into 
# account for creating the next one. 
# this matters if you think about it, the model can only memorize 8 tokens! all that came 
# before is just gone! the model cant use any of them to infer new information! 
# it can only use the last context_size tokens! but nevertheless as we later see, 
# we can continue to generate infinit characters though they may not make sense 
# if the context size is small as you can guess!)
# 
# so far what we covered here was the time dimension of our input. 
# we have a sequence, and each entery basically denotes a time t dimension,
# at which, a character is introduced. 
# another imporatnt aspect we need to take care of is the batch dimension, 
# cuz we are going to feed multiple examples at once, we use this to harness
# the gpu parallilization capalibity in pytorch as without it, training this 
# simple model would take a lot of time on cpu! 
#
# so lets create a function that gives us a batch of the inputs rather than a single list/tensor!
# 
batch_size = 4 
# our block_size
context_size = 8 
# 
def get_batch(split, batch_size):
    #lets grab the data basedo on the split
    data = train_data if split =='train' else val_data
    # we want a batch of 4 of 8 tokens/characters (context-size). 
    # to make a batch we can grab 4 random indices as input and then expand them
    # by adding the next 8(i.e. context_size) characters to them, 
    # in order not to go past the last index, we subtract the length
    # of data(last valid index) from context size 
    idxs = torch.randint(low=0, high=len(data)-context_size, size=(batch_size,))
    # we have our indices, so lets create our samples
    # x = [data[i:i+context_size] for i in idxs]
    # and for labels we do the same thing but offset it by 1 so each characters label
    # becomes the next character
    # y = [data[i+1:i+context_size+1] for i in idxs]
    #
    # print(f'{x=}')
    # print(f'{y=}')
    # prints: 
    # x=[tensor([58, 39, 49, 43,  1, 51, 63,  1]), tensor([53, 59, 41, 46,  5, 42,  1, 61]), tensor([47, 53, 52,  1, 39, 57,  1, 63]), tensor([39, 57, 58,  1, 51, 63,  1, 50])]
    # y=[tensor([39, 49, 43,  1, 51, 63,  1, 54]), tensor([59, 41, 46,  5, 42,  1, 61, 47]), tensor([53, 52,  1, 39, 57,  1, 63, 53]), tensor([57, 58,  1, 51, 63,  1, 50, 53])]
    #
    # as you can see we have a list of tensors. since we want a batch, 
    # we just stack them or concat them
    # on top of each other
    # using stack!
    # x = torch.stack(x)
    # y = torch.stack(y)
    # stack() is intrestingly implemented by concat(), so if we want, 
    # we can do the same using concat!
    #
    # by default conact, concatenates all the tensors as one large tensor!
    # we need a reshape/view to get the right shape
    # since we are using view, no copy,etc is done, and its an efficient operation
    # x = torch.concat(x,dim=0).view(batch_size, context_size)
    # y = torch.concat(y,dim=0).view(batch_size, context_size)
    # 
    # however if we add an extra dim to our inputs we can remove the extra reshape/view
    # x = torch.concat([t.unsqueeze(0) for t in x], dim=0)
    # y = torch.concat([t.unsqueeze(0) for t in y], dim=0)
    #
    # By unsqueezing, we ensure that the tensors have the same number of dimensions before 
    # concatenating them with torch.cat/concat/concatenate.
    # (side tip: concat and concatenate are aliases for cat) 
    # This effectively replicates the behavior of torch.stack.
    # see torch.cat concatenates tensors along an existing dimension, and we only have 1 dimensional
    # tensors, so it will concatenate them along that, effectively making one
    # large 1 dimensional vector however, when we use unsqueeze on each tensor,
    # using torch.unsqueeze(0), we are adding a new dimension(dim 0) to that tensor,
    # making it 1,x instead of the original shape of (x,).  
    # So, by unsqueezing each tensor along the desired dimension, which for us is 
    # the 0ths dimension (or row dim) and then concatenating them with torch.cat,
    # we achieve the same result as torch.stack.
    # 
    # in practice we'd like to do this all in one go!
    x = torch.stack([data[i:i+context_size] for i in idxs])
    y = torch.stack([data[i+1:i+context_size+1] for i in idxs])
    
    return x,y

x,y  = get_batch('train',batch_size=4)
print(f'{x.shape=}\n{y.shape=}')
print(f'{x=}\n{y=}')
# which prints 
# x.shape=torch.Size([4, 8])
# y.shape=torch.Size([4, 8])
# tensor([[58, 39, 49, 43,  1, 51, 63,  1],
#         [53, 59, 41, 46,  5, 42,  1, 61],
#         [47, 53, 52,  1, 39, 57,  1, 63],
#         [39, 57, 58,  1, 51, 63,  1, 50]])
# tensor([[39, 49, 43,  1, 51, 63,  1, 54],
#         [59, 41, 46,  5, 42,  1, 61, 47],
#         [53, 52,  1, 39, 57,  1, 63, 53],
#         [57, 58,  1, 51, 63,  1, 50, 53]])
#
# now this is our batch of data, 4 samples with 8 characters, 
# bascially 32 examples, lets see that as well
for b in range(batch_size):
    # this is our time dimension
    for t in range(context_size):
        print(f'{x[b,:t+1]} --> {y[b,t:t+1].item()}')
#        
# prints : 
# tensor([58]) --> 39
# tensor([58, 39]) --> 49
# tensor([58, 39, 49]) --> 43
# tensor([58, 39, 49, 43]) --> 1
# tensor([58, 39, 49, 43,  1]) --> 51
# tensor([58, 39, 49, 43,  1, 51]) --> 63
# tensor([58, 39, 49, 43,  1, 51, 63]) --> 1
# tensor([58, 39, 49, 43,  1, 51, 63,  1]) --> 54
# tensor([53]) --> 59
# tensor([53, 59]) --> 41
# tensor([53, 59, 41]) --> 46
# tensor([53, 59, 41, 46]) --> 5
# tensor([53, 59, 41, 46,  5]) --> 42
# tensor([53, 59, 41, 46,  5, 42]) --> 1
# tensor([53, 59, 41, 46,  5, 42,  1]) --> 61
# tensor([53, 59, 41, 46,  5, 42,  1, 61]) --> 47
# tensor([47]) --> 53
# tensor([47, 53]) --> 52
# tensor([47, 53, 52]) --> 1
# tensor([47, 53, 52,  1]) --> 39
# tensor([47, 53, 52,  1, 39]) --> 57
# tensor([47, 53, 52,  1, 39, 57]) --> 1
# tensor([47, 53, 52,  1, 39, 57,  1]) --> 63
# tensor([47, 53, 52,  1, 39, 57,  1, 63]) --> 53
# tensor([39]) --> 57
# tensor([39, 57]) --> 58
# tensor([39, 57, 58]) --> 1
# tensor([39, 57, 58,  1]) --> 51
# tensor([39, 57, 58,  1, 51]) --> 63
# tensor([39, 57, 58,  1, 51, 63]) --> 1
# tensor([39, 57, 58,  1, 51, 63,  1]) --> 50
# tensor([39, 57, 58,  1, 51, 63,  1, 50]) --> 53 
#
# so now that we have the data sorted out, lets create our model. 
# as we said, we are going to use a bigram model as our language 
# model here and later add attention mechanism to it. 
# to make things easier and more self contained, lets add all the required logic to this model
# like when we do a forward, we be able to calculate loss as well if we are given the targets
#
# side note:-------------------------------------------------------------------------------
# review/reminder for some key terms: language model, bi/n-gram models, seq-to-seq models:
# A language model is a probabilistic model of a natural language. It's used to predict 
# the likelihood of a sequence of words or tokens(characters,etc). 
# Language models are used in a variety of tasks, including speech recognition, 
# machine translation, natural language generation, optical character recognition,
# handwriting recognition, grammar induction, and information retrieval.
# A bigram is a type of language model where the probability of the next word in a
# sequence depends only on the previous word. It's a sequence of two adjacent elements
# from a string of tokens, which are typically letters, syllables, or words. 
# Bigrams, along with other n-grams, are used in most successful language models for
# tasks like speech recognition. 
#
# A sequence-to-sequence (Seq2Seq) model is used in sequence prediction tasks, such 
# as language modeling and machine translation. 
# The idea is to use one LSTM (Long Short-Term Memory), the encoder, to read the input
# sequence one timestep at a time, to obtain a large fixed dimensional vector representation
# (a context vector), and then to use another LSTM, the decoder, to extract the output 
# sequence from that vector. the second LSTM is essentially a recurrent neural network 
# language model except that it is conditioned on the input sequence. 
# So, in essence, language models form the foundation of sequence-to-sequence models. 
# They provide the mechanism for predicting the next element in a sequence, which is 
# a key component of sequence-to-sequence models.
#
# !explain this better later
# side note about Autoregressive models : https://www.youtube.com/watch?v=vwG3KWzuACo
# the class of models we are tyring to implement here is called autoregressive and they 
# are shown to outperform recurrent networks (rnns/lstm/gru/etc)
# an autoregressive model is a feedforward model that predicts the next variable xt in 
# a time series based on k previous variables (xt-1, xt-2,...) 
# in RNNs, the parameters are shared across time (same function (f(.) at different t))
# while Autoregressive models, make a strong conditional independence assumption. 
# watch the video to have a clue!
# -------------------------------------------------------------------------------
#
# so lets go
class BigramModel(nn.Module):
    def __init__(self, vocab_size) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        # our bigram model which we built in refresher section, was nothing more 
        # than a 2d array of vocab_size, we can achieve the same behavior using 
        # a single weight matrix or torch.Embedding. we use torch.Embedding to 
        # not reinvent the wheel! and also cuz all new language models use word-embeddings!
        # so its a good choice for our base model anyway!
        self.token_embedding = torch.nn.Embedding(vocab_size, vocab_size)
    
    # since we want to be able to calculate loss, if there are labels, we get it as well
    # sidenote:---------------------------------------------------------------------------
    # note, we are using __call__ to be in line with our previous examples/implementations
    # from refresher section. in practice, we use forward() because we are using torch 
    # nn.Module from now on. this allows us to use hooks if we ever need to. 
    # ------------------------------------------------------------------------------------
    def __call__(self, inputs:torch.Tensor, labels:torch.Tensor=None) -> torch.Tensor:
        logits = self.token_embedding(inputs)
        loss = None
        if labels is not None:
            # calculate loss 
            # print(f'{logits.shape=}') # prints (4,8,65)
            # note that both inputs and labels have the shape (B,T)
            # but logits has the shape (B,T,C) which C here equals vocab_size 
            # this is an issue for torch.crossentropy, as it expects the input
            # to be in the form of (B,C,T), that is, the channels/embeddings dimension
            # need to be right after the batch dimension or otherwise it wont work.
            # so we need to account for it.
            # we can go on permuting the dimensions, like this and rectify the issue:
            # logits = logits.permute((0,2,1))
            # however, if for some reason the output of logits in the form of (4,8,65)
            # is not ideal for us, and instead we want the usual form of (B,C) e.g,
            # we can easily do it that way instead, 
            # think about it, what it really is, is 4*8=32 examples arranged in a (4,8) shape and 
            # if we rather a normal 2d tensor of shape (32,65), we can simply merge the first two dims!  
            # lets do just that, flatten the two dimensions into one and carry on!
            # we are basically concatenating the batch and time dimensions
            # into one dimension (effectively, stacking samples on top of each other), 
            # instead of having four compartments, each having 8 segments, we are going to have
            # 1 long compartment with 32 segments/rows for the lack of better words!!
            # so lets first get the shapes 
            # B,T,C = logits.shape
            # logits = logits.view(B*T,C)
            # 
            # and we also need to do the same for the labels 
            # labels = labels.view(B*T)
            # we could also do,
            # labels = labels.view(-1)
            # 
            # but thats not really needed, so we just simply permute logits temporarily 
            # so the logits shape stays the same regardless of calculating the loss or not :-)
            loss = F.cross_entropy(logits.permute((0,2,1)), labels)
        return logits, loss
    
    # now lets also add a generate method to generate texts
    def generate(self, idx, max_token_count):
        # before we implement this lets review what we expect this method to do
        # we want this to generate some characters, given an initial character
        # we also want to be able to control the length of the generated text
        # so we take a max_token_count.
        # we take the initial index, 
        # feed it to the model, 
        # get the logits,
        # turn logits to probs,
        # use torch.multinomial to sample from our probablity distribution
        # get the new character index, and add it to a list to gradually 
        # create our final text output
        # so lets go 
        # our idx should have the shape [B,T]
        assert len(idx.shape) == 2, f'idx.shape({idx.shape}) should have (b,t) form.'
        #
        # lets generate as many characters/idx as max_token_count specifies
        # this is basically specifies the time/sequence dimensions
        for i in range(max_token_count):
            # since we are defining a class method, to call the callable, we simply use self()
            logits, _ = self(idx)
            # to get the probablities 
            # usually we would simply do 
            # probs = torch.softmax(logits, dim=1)
            # but here, we want to only focus on the next character because this is what comes next
            # each time obviously, so we only take the last timestep to see what the model predicted.
            # sidenote: 
            # you might ask we have 7 choices, why are we choosing the last character/token?
            # can we choose any of those 7 time dimensions? like 1,2,3...,7 as well?
            # in this case it really doesnt matter and the loss stays the same, as its just 
            # a bigram model!
            # if we chose the last token its not like, the model cares about all the context_size,
            # no, its just meaningless to it, it does not have the capability to utilize the context_size
            # at all! 
            # so for this specific case, we can select any dimension other than 0 obviously and 
            # the loss wont change!
            # but to be consistent with our future changes, and not changing the codebase as much as
            # possible, and the fact that when generating text with the starting token (zeros(1,1) 
            # as we will see in a moment)
            # we have a single time dimension (i.e. 0) so any other value (except -1) would result
            # in an error, we dont hardcode a specific dim, and instead use -1 to refer to the last
            # dim whatever it happens to be at the time of execution. 
            # also note that while the choice of dim here doesnt affect the loss, it 'does' affect 
            # the text generation. 
            # try different dimensions when trying to generate text after you trained the model and
            # see for yourself you can also use different dimensions as we talked about during
            # training and see the loss wont change! 
            # (provided you set seeds so the output becomes determinstic)
            # 
            logits = logits[:,-1,:] # this now becomes (B,C) instead of the initial (B,T,C)
            probs = torch.softmax(logits, dim=-1) 
            # now lets sample from it! what! why? you may ask!
            # see genrating output really has nothing to do with the model, in a sense that,
            # comming up with what we consider satisfactory can be more than just selecting 
            # the entry with highest probablity.
            # we can do all sorts of things to direct the generation process toward what we 
            # find satisfactory. we can use temperature, top-p, etc to do just that. 
            # Using sampling using here is for this very reason, to this end, we use torch.multinomial
            # for generating the next token, this is a technique known as sampling! 
            # instead of simply choosing the token with the highest probability 
            # (which is known as greedy decoding by the way), sampling selects the next token randomly
            # according to the probability distribution produced by the model.
            # the reason we use sampling instead of the greedy decoding method (i.e. the highest probability way!)
            # is to introduce more diversity and randomness into the generated output. 
            # If we always choose the token with the highest probability, the generated text can becomes
            # repetitive and deterministic, especially over long sequences. By introducing some 
            # randomness like this, we can generate more diverse and interesting output.
            # note that this can sometimes generate less probable (and potentially less coherent)
            # sequences as well (obviously!). 
            # The balance between diversity and coherence is a common challenge in text generation,
            # and  different decoding strategies (like greedy decoding, sampling, beam search, 
            # using different temperatures, or top-p, etc) offer different trade-offs.
            # also replacement if true, means, if something is selected, it can be selected again!
            # (basically its place will be filled/replaced and ready to be used again(reloaded), 
            # like if e.g. you chose/pickedup an apple! another apple will be repalce the old one,
            # so you always have apple!)
            # basically if a token is picked one, replacement=True, means it can be selected again
            # (I'd like to call it replacemen = reloading!)
            idx_next_char = torch.multinomial(probs, num_samples=1, replacement=True) # shape is (B,1)
            # now lets add this to the next input to be fed to the model 
            # since idx has the shape(batch, T), we should add this tothe second dimension
            # to the time dimension/ or sequence dimension. this as the loop goes on, 
            # creates the shape (B,T+1) 
            #this doesnt make sense for this particular model, becasue we are always checking
            # the next character given the previous one, so all the concatenation we are doing
            # is just useless now. the reason we are implementing this like this, as I pointed 
            # out earlier is to create a base foundation, so that we can improve upon it when 
            # we add attention later on which will use the history of previous characters.
            idx = torch.cat((idx, idx_next_char), dim=1) 
        
        # and finally when all is done return the idx which by now should have the whole output
        return idx
    
model = BigramModel(vocab_size)
out,loss = model(x,y)
print(f'{out.shape} {loss}')
# prints:
# torch.Size([32, 65]) 4.574285984039307
# the loss is good, we learned previously that we can evaluate a base loss provided 
# our number of classes since we have 65 classes (our vocab_size or number of characters involved)
# the uniform probability for each class would be 1/65 =0.015384615, which if we take
# its negative log would turn out to be -ln(1/65) = 4.17438727, 
# which is pretty close to the loss we'v got here, signifying its a pretty decent value to begin with.
# also lets see the generate method at work
# lets create a dummy input, basically a batch of 1 and sequence of 1 of zero!
# we do this to generate a text from scratch
inputs = torch.zeros(size=(1,1),dtype=torch.int32)
output = model.generate(inputs, max_token_count=100)
print(f'{output=}')
print(''.join(decode(output.squeeze(0).tolist())))
# prints 
# wUbN:i :kLnOqHCQTG;.MbupOWH Xfi!MSUalaQppNNqIoWfmuSIIfmuZqf-3NLt-YkOc3vC-YkBfEHr3pJodwVY.nw.,r!&,Clt
# which is expected since our model is not trained yet! 
# lets train the model now and see how it works

batch_size = 32
context_size = 8
vocab_size = len(vocab_list)
max_iter = 20000

model = BigramModel(vocab_size)
optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-3)
device = 'cpu'#'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device=device)

print(f'{torch.__config__.show()}')
print(f'{device=}')
print(f'{model=}')

model.train()
for pos in range(max_iter):
    x,y = get_batch('train', batch_size=batch_size)
    x=x.to(device)
    y=y.to(device)
    logits, loss = model(x,y)
    # zero out the grad
    model.zero_grad(True)
    # do a backward pass
    loss.backward()
    # and do a single optimizer update
    optimizer.step()
    
    if pos%1000==0:
        print(f'{loss}')
# prints 
# 4.760574817657471
# 3.727213144302368
# 3.0139858722686768
# 2.6749324798583984
# 2.5884087085723877
# 2.5262675285339355
# 2.4420430660247803
# 2.4787800312042236
# 2.4680707454681396
# 2.564662456512451
# 2.51802659034729
# 2.562812328338623
# 2.422783613204956
# 2.5023772716522217
# 2.501049518585205
# 2.411632537841797
# 2.4028165340423584
# 2.4135568141937256
# 2.50952410697937
# 2.574878215789795

# relying on a single batch loss is not a good idea to measure the performance of our model.
# moreover relying on training loss, is not good either, so it would be much better if we 
# considered more batches for loss and even better we could also investivate the models 
# performance on our validation set. 
# so lets do just this and define a function that calculates loss for training and 
# validation sets alike but considers more batches for loss calculation

# this decorator signals torch not to calculate gradients for its operations!
@torch.no_grad()
def evaluate_loss (iterations, device=None):
    results={}
    # before calculating the loss, lets switch to eval mode,
    # although for our specific case this doesnt matter
    # but its good practice, as later on, we will add layers
    # that their behavior do change depending on traing/val mode.  
    model.eval()
    if device is None:
        # use the device assigned to what model params are assigned
        device = next(model.parameters()).device
    # since we already used no_grad decorator, we dont need to use
    # no_grad context manager here
    # with torch.no_grad():
    for split in ['train','val']:
        losses = torch.zeros(size=(iterations,), device=device)
        for i in range(iterations):
            x,y = get_batch(split, batch_size)
            x,y = tuple(t.to(device) for t in (x,y))
            logits, loss = model(x,y)
            losses[i] += loss
        results[split] = losses.mean(0)
    # since we want to use this inside training loop, make sure we set the model
    # back to train mode incase we use layers such as batchnorm,etc that the 
    # require being trained!
    model.train()
    return results

loss= evaluate_loss(200)
print(f'{loss["train"]=} {loss["val"]=}')
# so now that our function seems to be working lets use it in the training loop :
model = BigramModel(vocab_size)
optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-3)
model = model.to(device=device)
model.train()
for pos in range(max_iter):
    x,y = get_batch('train', batch_size=batch_size)
    x=x.to(device)
    y=y.to(device)
    logits, loss = model(x,y)
    # zero out the grad
    model.zero_grad(True)
    # do a backward pass
    loss.backward()
    # and do a single optimizer update
    optimizer.step()
    
    if pos%1000==0:
        # now instead of simply printing loss, lets use our new function!
        loss= evaluate_loss(200)
        print(f'train_loss: {loss["train"].item():.4f},  val_loss: {loss["val"].item():.4f}')
# which results in :
# train_loss: 4.7005,  val_loss: 4.6906
# train_loss: 3.7171,  val_loss: 3.7120
# train_loss: 3.1180,  val_loss: 3.1345
# train_loss: 2.8052,  val_loss: 2.8015
# train_loss: 2.6346,  val_loss: 2.6489
# train_loss: 2.5593,  val_loss: 2.5725
# train_loss: 2.5153,  val_loss: 2.5390
# train_loss: 2.4909,  val_loss: 2.5152
# train_loss: 2.4886,  val_loss: 2.5018
# train_loss: 2.4715,  val_loss: 2.4955
# train_loss: 2.4784,  val_loss: 2.5003
# train_loss: 2.4568,  val_loss: 2.4900
# train_loss: 2.4595,  val_loss: 2.4769
# train_loss: 2.4592,  val_loss: 2.4763
# train_loss: 2.4545,  val_loss: 2.4833
# train_loss: 2.4608,  val_loss: 2.4802
# train_loss: 2.4556,  val_loss: 2.4844
# train_loss: 2.4638,  val_loss: 2.4741
# train_loss: 2.4557,  val_loss: 2.4766
# train_loss: 2.4509,  val_loss: 2.4851
#
#now lets check its output again after some training 
inputs = torch.zeros(size=(1,1), device=device, dtype=torch.int32)
output = model.generate(inputs, max_token_count=500)
print(''.join(decode(output.squeeze(0).tolist())))
# prints :
# The oerire aruse tta tele hasidet thend tor w is, is A:
# Anganu nthiroure ghe fr'Cate ir he, chas, thto icithives ilaed s dl owaurd b.
# Whu uasm threw oud foube, bearss:
# When

# Mey bomo then blcee aprerod wis CUKEO IOMaurey.
# Siscaritheare, m tath kerd

# Fong on, rrw ckne t ath,
# OFoue tyerey pankee,
# Bul y trof pp
# Thaswer
# Ye.
# O:
# PO:
# Fo oulicoweny bl soth ne g ethe be t:
# PUCEEEStof whaigaven sldd gle y hadellag ans'd.
# And wethaing se her my ve icanool:
# cthasectofer
# Wed ina ik y RY wacr-the amar hes tha 
# 
# which looks much better than the initial random output we got earlier.
#%% 
torch.manual_seed(255)
# now that we have our base model, lets add the attention mechanism to it. 
# The attention mechanism at its core is nothing but a communication mechanism, and what it does is
# it tries to take advantage of the rich information embedded in the input/sequence. 
# for our case here, this is specifically about the past history (i.e. past tokens and how they
# are related or show/affect the next token probablity)
# but in general attention can utilize both past and future connections/sequence tokens. 
# what we just described here is not the whole story, but its enough to gives us a foundation 
# to build our intuition as we continue on. 
# we will elaborate more of course and hopefully gradually improve our definition and 
# understanding of the attention mechanism.
# we said attention is a kind of communication mechanism, but what does that mean? 
# how do tokens communicate? to put it simply, by communication, we mean to somehow 
# involve the value of one or more tokens in the operation so that they can play a 
# role in the final output. we can think of several ways of implementing this concept.
# one simple way could be to just sum all the values/weights assosiated with each
# token (previous tokens), and use that to determine the value/weight for the current token!
# another way could be to use the average instead of the sum, of the said tokens!
# so we are after finding a way, a good one, to include/incorporate every values/weights
# for the tokens involved, so we get a better/more accurate value/weight for the current token.
#
# now lets expand on this and get an intuitive undrestanding what all of this means.
# before we venture any further into the crux of the matter!(big words:-)), 
# let us learn about a technique thats used to efficiently implement attention mechanism, 
# after this you should get a good idea about all of this. we need this to understand the whole
# thing! so stay with me.
# For this purpose,lets imagine we have a simple input like the following: 
# lets create an input of the following shape
B,T,C = (4,8,2)
# to make it more intuitive lets make a tensor with known numbers and then reshape it
x = torch.arange(0, 64, dtype=torch.float).view(B,T,C)
print(f'x={x}')
# imagine we have an input like what we encountered previously in our examples. 
# in this sample input, we have a batch of 4 samples, each having 8 sequences 
# with each sequence having a vector of 2 values.
# what we are planning to do is to provide a way by which each token can communicate
# with other tokens. 
# we have 8 tokens in our sequence. so we want our tokens to be able to communicate with
# all the previous tokens that came before them. 
# The reason we are only looking in the past token is simply becasue we are trying to 
# perdict the future, so it only makes sense to look at the past and current timestamp 
# and infer on what to do for the future.  
# as we said earlier, one of the easiest ways we could come up to implement such communication
# mechanism between tokens could be to sum or average the values of all previous tokens 
# as a way of taking into account their contribution to the [final] answer/output.
# that is, lets say if we are currently at token 5, we take the average of 
# the current token and all previous tokens before it, effectively making a feature vector that
# reflects our current status of the sequence so far, having taken all previous tokens/steps up to now.
# note that as you may also have guessed, summing/averaging arent the best way to model 
# such interations. in fact they are an extremely weak form of interaction between tokens,
# this kind of communicating is extremely lossy so to speak, that is we lose a great deal 
# of information concerning the underlying relationships between tokens, their arrangements,
# their implicit interactions, semantics, etc. but for now this is ok. we will later on see 
# how we can fix this issue.
# 
# so now what we want to do, is to calculate the sum or average of all tokens up to the current
# token in all batches at the same time.
# a naive way for calculating the average would be to do sth like this using a for loop:
results = torch.zeros(size=(B,T,C))
for b in range(B):
    for t in range(T):
        results[b,t] = x[b,:t+1].mean(0)
print(f'{results=}')
# which prints 
# results=tensor(
#        [[[ 0.,  1.],
#          [ 1.,  2.],
#          [ 2.,  3.],
#          [ 3.,  4.],
#          [ 4.,  5.],
#          [ 5.,  6.],
#          [ 6.,  7.],
#          [ 7.,  8.]],

#         [[16., 17.],
#          [17., 18.],
#          [18., 19.],
#          [19., 20.],
#          [20., 21.],
#          [21., 22.],
#          [22., 23.],
#          [23., 24.]],

#         [[32., 33.],
#          [33., 34.],
#          [34., 35.],
#          [35., 36.],
#          [36., 37.],
#          [37., 38.],
#          [38., 39.],
#          [39., 40.]],

#         [[48., 49.],
#          [49., 50.],
#          [50., 51.],
#          [51., 52.],
#          [52., 53.],
#          [53., 54.],
#          [54., 55.],
#          [55., 56.]]])
#
# you may notice that, some people/researchers refer to what we did here(averaging) as BoW, 
# or bag of words. 
# We are effectively averaging embeddings here and averaging embeddings can be considered a form of 
# Bag of Words (BoW) representation. In BoW, the focus is on the occurrence and frequency of words, 
# rather than their order or structure. 
# By averaging embeddings, we are essentially treating each 'word' as an independent feature and
# capturing its representation in the form of a numerical vector.
#
# Although what we have is basically embeddings and averaging embeddings does not capture the exact 
# frequency of each word, it does however, capture the overall distribution and semantic information
# present in the text. 
# Like BoW, our approach here disregards word order and focuses on the presence and representation
# of words. 
# also note that averaging embeddings may preserve some semantic relationships between words, which
# BoW representations might not capture as effectively.
# 
# sidenote: -------------------------------------------------------------------------------
# Bag of Words (BoW) is a commonly used technique in natural language processing (NLP) for 
# representing text as a numerical feature vector. It disregards the order and structure of
# words in a document and focuses only on their occurrence and frequency.
# In the BoW model, a document or a piece of text is represented as a "bag" (unordered set) 
# of words, where each word is treated as an independent feature. The presence or absence of
# words in the document is encoded as a binary value (0 or 1), and the frequency of each word
# is often used as the value in the feature vector.
# 
# for the reference, this is how a BoW process works:
# First, we have the tokenization phase, in which the text is split into individual words or tokens. 
# punctuation marks, whitespace, and other special characters are usually removed or treated as
# separate tokens.
# Then, a vocabulary is created by taking all unique words from the entire corpus (collection of documents). 
# each unique word is assigned a unique index or position in the vocabulary. this is the vocabulary 
# creation phase!
# 
# next each document is represented as a feature vector, typically a as one-hot encoded vector or
# a count vector. (as the name suggests, in the one-hot encoding scheme, each word in the vocabulary
# corresponds to a binary feature, and the vector contains 1s in the positions where the word occurs
# and 0s elsewhere. while in a count vector, the value at each position represents the frequency of 
# the corresponding word in the document)
#
# and finally, the resulting feature vectors can be used as input to out models for tasks such as
# text classification, sentiment analysis, document clustering, information retrieval, etc. 
# Needless to say, BoW has some limitations as well. It does not capture the semantic meaning or 
# context of the words, as it treats each word independently. It also ignores the grammar and word order. 
# Having all these said, BoW is simple, efficient, and can be a useful baseline representation
# for various NLP tasks.
# -------------------------------------------------------------------------------
#
# so to recap one more time, we are basiaclly treating each timestep/sequence dimension, as a 'word',
# so we have 8 tokens/words, and we are averaging them (we are infact averaging their embeddings, 
# but thats obvious!) so we got ourselves a bow representation!
# now back to our discussion, if we try to visualize the results we get 
print(x[0])
print(results[0])
# prints:
# x[0]=
# tensor([[ 0.,  1.],
#         [ 2.,  3.],
#         [ 4.,  5.],
#         [ 6.,  7.],
#         [ 8.,  9.],
#         [10., 11.],
#         [12., 13.],
#         [14., 15.]])
# results[0]=
# tensor([[0., 1.],
#         [1., 2.],
#         [2., 3.],
#         [3., 4.],
#         [4., 5.],
#         [5., 6.],
#         [6., 7.],
#         [7., 8.]])
# if you look closely, you'll notice that each row, contains the mean(average) of all the rows before
# it. consider the first 3 rows in x[0], 
# tensor([[ 0.,  1.],
#         [ 2.,  3.],
#         [ 4.,  5.],
# now refer to the 3rd row in results[0] which is :  [2., 3.],
# (for the 3 rows in X[0] we have [0+2+4, 1+3+5] -> [6/3, 9/3] -> [2,3])
# likewise, consider the last row in results which contains the average for all the rows in x:
# 0+2+4+6+8+10+12+14 = 56 which when divided by their count, 8, results in 7, 
# and this is the same for the second column, thus we get:
# results[0,7] = [7., 8.]])
# 
# this is all good but the problem is using for loops to calculate this is very inefficient, 
# it happens that this operation can be efficiently calculated using matrix multiplication.
# 
# lets learn this trick using an example: 
# suppose we have the following as the input: 
a = torch.ones(size=(3, 3))
b = torch.randint(0,10,size=(3, 2)).float()
c = a@b
print(f'{a=}')
print(f'{b=}')
print(f'{c=}\n----')
# prints
# a=tensor(
#        [[1., 1., 1.],
#         [1., 1., 1.],
#         [1., 1., 1.]])
# b=tensor(
#        [[9., 3.],
#         [5., 5.],
#         [6., 5.]])
# c=tensor(
#        [[20., 13.],
#         [20., 13.],
#         [20., 13.]])
# nothing fancy here, we have matrix multiplication, the first row of 'a' is dot-producted by
# first col of 'b', then sumed, it makes up the first col of first row in c. likewise the 
# first row of 'a' dot the second col of 'b', then summed the results, makes up the second col
# of first row in c. and this goes on for the rest of the matrixes. 
# this is what we learned back in highschool, but what is it exactly that we are learning here
# exactly?!
# if you look closely, you'll notice that, the c cols are actually the sum of all the rows in b!
#        [9]
# b[:,0]=[5] 
#        [6]
# 9+5+6 is 20! likewise, 
#        [3]
# b[:,1]=[5] 
#        [5]
# 3+5+5 is 13!
# hence c = [20., 13.] which is repeated obviously because the second and third rows of a are 
# all 1s as well.
#           [20., 13.]  
#           [20., 13.] 
# I guess you are now starting to get where we are going with this, if we can some how alter the
# 'a' matrix, we may very well be able to achieve our goal! 
# how you may ask? the answer is using torch.tril!
# torch.tril() is a function that returns a matrix from a given tensor, so that half of it set to zero,
# basially it creates a triangular tensor, where the right half is just zeros! 
# lets see how it works, lets apply it on 'a'
a_tril = torch.tril(a)
print(f'a_tril:\n{a_tril}')
# it prints
# a_tril=tensor(
#        [[1., 0., 0.],
#         [1., 1., 0.],
#         [1., 1., 1.]])
# as you can see the the right half is set to zero and we are left with a triangle shape of 1s on 
# the left side.
# now if we do a@b this time we get:
c = a_tril@b 
print(f'b:\n{b}')
print(f'c:\n{c}')
# a_tril:
# tensor([[1., 0., 0.],
#         [1., 1., 0.],
#         [1., 1., 1.]])
# b:
# tensor([[9., 3.],
#         [5., 5.],
#         [6., 5.]])
# c:
# tensor([[ 9.,  3.],
#         [14.,  8.],
#         [20., 13.]])
# now if you look closely, you'll notice that, this time, each row in c, is effectively the sum of
# the previous rows in b, like the first row of 'a' is only 1 in the 0ths column, so the first row
# of b is copied in c intact (workout the math and see why). 
# the second row in 'a', now has two 1s in col 0 and 1 respectively, which effectively translates 
# to summing the first two rows in b. (9+5 =14, 3+5=8). likewise, the third row in 'a' is all 1s, 
# signfigying all rows in b will be summed which gives us (9+5+5=20, 3+5+5=13). 
# 
# so basically we are doing sums here, becasue our tensor a is all ones. so if we want to somehow 
# calculate the average, instead of sum, we can easily change 'a' by normalizing it so that the 
# each row sums to 1 (i.e. all cols sum to 1), this way the end result will be the average 
# (becasue the 'b' is multiplied by a fraction/scale and then summed)
# so if we scale 'a' by the sum of all its columns, we should get average instead
a = torch.ones(size=(3,3))
a = torch.tril(a)
a = a/a.sum(dim=1, keepdim=True) # this looks familiar doesn it? yup its softmax!
print(f'a:\n{a}')
# a:
# tensor([[1.0000, 0.0000, 0.0000],
#         [0.5000, 0.5000, 0.0000],
#         [0.3333, 0.3333, 0.3333]])
#
# note that, now each row, sums to 1. the first row, the first element is 1, because the rest are 0s
# but in the second row, as there are two 1s, the probabality is divided between the two, each being 0.5
# likewise, in the third row, as there are 3 1s, the probablity is divided between all of them, 
# making each to have the value 0.33. 
# and now if we try to multiply them, we get average as the result:
c = a@b 
print(f'calculating average:')
print(f'b:\n{b}')
print(f'c:\n{c}')
# we get 
# calculating average:
# b:
# tensor([[9., 3.],
#         [5., 5.],
#         [6., 5.]])
# c:
# tensor([[9.0000, 3.0000],
#         [7.0000, 4.0000],
#         [6.6667, 4.3333]])
# 
# we see that, each row in c, is the average of all the rows before it. 
#
# so using this trick, we can take the incremental average of any matrix we like. 
# now that we learned the trick, lets go back and implement the bows for loops using this techique !
# previously we had : 
# results = torch.zeros(size=(B,T,C))
# for b in range(B):
#     for t in range(T):
#         results[b,t] = x[b,:t+1].mean(0)
# which calculated the average for sequence dimensions (tokens) incrimentally
# lets do this now 
# we want a TxT weight becasue we want to average T timestep/tokens
weight =  torch.tril(torch.ones(size=(T,T)))
# remember to set keepdim=True, or otherwise, as we sum along dim=1, we lose that dim
# (it collapses, and then the broadcast will be wrong, it will infact make each column 
# have one probablity instead of each row, which is the exact opposite of what we want)
weight = weight/weight.sum(dim=1, keepdim=True)
print(f'weight:\n{weight}')
# and now we need to multiply this by x! 
bow_results= weight@x
print(f'bow_results:\n{bow_results}') 
# which prints 
# bow_results:
# shape: torch.Size([4, 8, 2])
# tensor([[[ 0.0000,  1.0000],
#          [ 1.0000,  2.0000],
#          [ 2.0000,  3.0000],
#          [ 3.0000,  4.0000],
#          [ 4.0000,  5.0000],
#          [ 5.0000,  6.0000],
#          [ 6.0000,  7.0000],
#          [ 7.0000,  8.0000]],

#         [[16.0000, 17.0000],
#          [17.0000, 18.0000],
#          [18.0000, 19.0000],
#          [19.0000, 20.0000],
#          [20.0000, 21.0000],
#          [21.0000, 22.0000],
#          [22.0000, 23.0000],
#          [23.0000, 24.0000]],

#         [[32.0000, 33.0000],
#          [33.0000, 34.0000],
#          [34.0000, 35.0000],
#          [35.0000, 36.0000],
#          [36.0000, 37.0000],
#          [37.0000, 38.0000],
#          [38.0000, 39.0000],
#          [39.0000, 40.0000]],

#         [[48.0000, 49.0000],
#          [49.0000, 50.0000],
#          [50.0000, 51.0000],
#          [51.0000, 52.0000],
#          [52.0000, 53.0000],
#          [53.0000, 54.0000],
#          [54.0000, 55.0000],
#          [55.0000, 56.0000]]])
# which gives us the same results as we expected.
# 
# side note: -----------------------------------------------------------------------
# one more thing before we continue on, note that the weight matrix is TxT while 
# the input is (BxTxC). (T,T) and (B,T,C) are not compatible, so what happens is
# that (T,T) is reshaped and a batch dimension is added to (T,T),making it (1,T,T)
# and then this is broadcasted along the batch dimension (replicated) to become 
# (B,T,T), then this will be multiplied by the (B,T,C)( note that at this stage
# a@b will be a batch multiplication operation, if you set the batch dimension aside,
# youll see that the rest of the dimensions match up, we have (T,T) and (T,C) which will
# result in (T,C). now if we add the batches back in, we will endup with (B,T,C)
# so in practice, the multiplication is done B times and then results are stacked.
# the first batches will multiply each other (T,T)x(T,C) = (T,C)
# the second batches will then multiply each other as well, getting another (T,C)
# and this goes on until we get (B,T,C))
#
a = torch.arange(0,9).view(3,3)
# a = torch.ones((3,3)).long()
b = torch.arange(0,30).view(2,3,5)
print(f'a:\n{a}')
print(f'b:\n{b}')
c = a@b 
print(f'c=a@b:\n{c}')
# is equivalent to 
# add a batch dimension to a
a = a.view(1,*a.shape) # or a.unsqueeze(0)
print(f'a with batch dim:\n{a.shape=}')
# replicate along the batch dimension
a = torch.cat(tuple(a.clone() for i in range(len(b))), dim=0)
print(f'{a.shape=}')
print(f'a(after replication along dim=0):\n{a}')
print(f'b:\n{b}')
# and now we have a case of batch-multiplication, the batch is the same 
# and sub tensors also are compatible, we have (T,T) and (T,C) so we now
# individually multiply each sub-tensor
c2 = torch.zeros_like(b)
for pos in range(b.shape[0]):
    c2[pos,...] = a[pos]@b[pos] # (T,T) x (T,C) -> (T,C)
# and ultimatley the result will have the shape (B,T,C)
print((c==c2).all())
# so to recap this 
# When performing the matrix multiplication `c = a @ b` with the given tensors, 
# several broadcasting steps occur to align the dimensions properly. 
# Here's a how it happens:
# 1. Tensor a has the (shape: 3, 3):
# 2. Tensor b has the (shape: 2, 3, 5):
# 3. They are not compatible so we broadcast tensor a to match the shape of b. 
# tensor a is expanded to (1, 3, 3) to have a batch dimension, and replicated 
# along dim 0 to result in (2, 3, 3). now both tensors have a batch of 2, and 
# if we put aside the batch dimension for a second, we'll notice that the rest
# of the shapes are compatible (3,3) and (3,5). so when we bring back the batch 
# dimension, we see we have a case of batch multiplication, that is we have 2
# sets of compatible tensors that need to be multiplied together. so we use a 
# simple for loop, to do multiplication, and stack the results and we are done!
# -------------------------------------------------------------------------------
#
# now back to our discussion. as we just saw, the traingular form in our weighted
# sum matrix, allows that each token at dimension t, can only interact with the tokens
# before it. 
# now you might think its a cool trick for taking average for a given matrix like 'a'
# but what if we dont want the simple average, and something more powerful? 
# there is another way of implementing the same thing but a bit differently
# if you look closely you can see that, we are dealing with probablities, so
# we may verywell use softmax to simplify this further.
# first lets create our triangular weight matrix
tril_tensor = torch.tril(torch.ones(size=(T,T)))
# now lets create a mask
weight = torch.zeros_like(tril_tensor)
# now lets mask all the zeros to -inf, so when we do softmax, all those -infs
# become 0. (recall that exp^-inf is 0! while exp^inf is inf! so its important to 
# set -inf (and also exp^0 is 1))
# so effectively what happens here is that, we're setting each entery in trail_tensor
# with zero to -inf, and leave the rest as zeros. when this tensor goes through softmax
# the 0s will be 1s and -infs will be 0s before they are normalized, when they are normalized
# the probablity of 1 will be split between all the enteries with the value of 1.
# so the first row has a single 1, so it will be 1.0, the second row has two 1s, so
# each one will take 0.5, the third will have 3 1s, so they each will become 0.333
# and so on.
weight=weight.masked_fill(tril_tensor==0, -torch.inf)
# now lets calculate the probs for each row, treating all cols as probs so their sum is 1
weight = weight.softmax(dim=-1)
bow_results2 = weight@x
print(f'{torch.all(bow_results==bow_results2)}')
# so you might ask, why would we want to do this? why make things more complicated?
# only to achieve what we already achieved pretty efficiently before? it doesnt make any sense!
# The answer is, this approach provides us with a flexibility that the previous ones
# wouldnt provide us with. think about it for a moment, what would we do if we didnt 
# want uniform probbablity for all the tokens in a row? 
# currently as you can see, we have uniform probablities in each row, we used that 
# to calculate the average, becasue that was what we were after initially! 
# but we said it earlier that average is a very weak form of communication! we lose
# a lot of information if we only use simple averaging like this. 
# if you think a bit more, you'll notice that its really the uniform probablity that
# each token gets, uniform probablity means they are basically the same, in terms of
# impact on the output. 
# thats like they are as important as the other tokens in every single situation! 
# this assumption is simply and clearly not true at all. again to understand why 
# this is the case remember the weights matrices in our models? if our model predicts
# all the classes the same, like at the very begining, when we havent started the training,
# the model is practically useless! until we train the model, and the weights get 
# different values. remember trained weights vs raw weights! so this is the same here really.)
# now if you see the previous implementation vs the newer one, you'll see that we basically 
# the 'weight' matrix can essentially be anything and not just zeros! 
# in other words, we have decoupled the 'weight' part from the constraint part (i.e. tril),
# which's job is to set the right half of the tensor to zero, so we actually get the expected
# behavior later on. (which is setting a constraint really (more later on))
# if you havent yet figured it out, the 'wieght' matrix, can be truly a weight matrix
# which can show different strengths for each token, basically it can learn the interations
# between tokens and manifest them. currently it is us who set it to all zeros, so we get
# uniform probablities, which inturn manifest itself as an average. but what if instead of 
# all zeros, we actually learn the values from the data itself? that would make sense, and 
# make it so that each token, can have a different connection(strength) to any other tokens
# now, and thus build semantic/meaningful relation. 
# This is infact what we are after. we want for tokens to learn associations and relations 
# with other tokens based on the data present in the dataset, having uniform weight like what
# we initially did, is really a far cry from what we intend, and therefore, we opt in to use 
# this new approach that allows us to actually exploit this new capability. 
# This is infact the problem that attention solves, that is gathering information from the 
# past but in a data driven manner. 
# (side note/reminder we are not limited to 'past' information only per say, in this example,
# this is the case however, we will explain this in more detail)), we will see how attention 
# does this exactly in a moment. 
#
# but before we jump into attention implementation also note that the tril part, infact is a
# hard-constrain here that prevents tokens from the past from interacting with the tokens 
# from the future. 
# so to recap here, basically the idea is, this triangular form, allows us to have 
# weighted aggregations of past elements. each element in the lower triangular part, 
# specifies, the degree by which it plays a rule in the said outcome. 
# that is how much of each element gets to get incorporated into the result of this
# specific position(i.e. current token's)
#
#
# Attention does its job by using two vectors called, key and query.
# basically this mean every single token, emits two vectors called key and query, 
# the query verctor as the name suggests, implies, what we are looking for, and 
# the key vetcor, again as the name suggets, implies, the contents, what it contains. 
# note that "emit" here refers to the process of generating or producing something. 
# When we say a token "emits" a key and a query vector, we mean that these vectors are produced 
# or generated from the token through some transformation (usually a learned linear transformation). 
# how exactly you may ask? the way we get our 'weights' for these tokens is like this:
# we simply dotproduct them together, 
# the ones that yield a high output/value, signify they are related positively.
# so our query is dotproducted with all the token's(keys) and the outputs reveal their 
# closeness/relevancy/similarity so to speak(if they align so to speak, they result in larger number)
# so effectively, the ones with higher number, are more similar to the query, the highest, 
# obviously having the most relavancy/similarity to the query.
#
# side note: --------------------------------------------------------------------------
# it might be intresting to include another point of view regarding tril-matrix and its 
# interaction with query & key. 
# From this point of view, the value of 1 at the position p, means the 'query' can attend
# to the 'key' at that position, and the value of 0 means otherwise (i.e. query can not 
# attend to the key at that position). this in turn means : 
# 1.each row corresponds to a query (the word we're predicting) and
# 2.each column corresponds to a key (the words we’re attending to).
# which once again shows, how we are evaluatiing multiple queries against multiple keys 
# at the same time, which is a key feature of the attention mechanism in transformers.
# furthermore,it shows that tril-matrix is simply a constraint on how query and key are 
# prevented from communicating/attending and how simple and straight forward the overall
# interaction between query and key is!
#--------------------------------------------------------------------------------------
#
# its now time to implement a single attention head, we can later use this to create 
# multi-attention-head which is basically several single attention heads working in 
# parallel. 
# first lets create some random input for our experiments while implementing attention 
torch.manual_seed(255)
# lets specify the batch, context_size and channel_num/embedding_size
B,T,C = 4,8,32
# lets create an input 
x = torch.randn(size=(B,T,C))
# we said that attention works with two vectors, key and query, lets implement them then!
# we can use nn.Linear to implement them but beore that, what are the dims for these vectors?,
# hint: recall that we said 'every token' emits/produces a vector through a transformation,
# since we are dealing with text, and nearly always take our inputs in the form of [word] embeddings
# the first dim would be our embedding size. the next dim, is sth called a head_sizes, which 
# specifies the size of the key/query output (or key/query vectory length really). its usually set
# as the same value as our embedding_size, but we can set it any value we like.)  
# lets define one
head_size = 16
# lets not forget to set bias=False, so what it does is exactly dotproduct 
# (actually, some people still leave the bias enabled! 
# so we will test both cases later and see if it actually matters! and if so to what extend! ) 
key = torch.nn.Linear(C, head_size, bias=False)
query = torch.nn.Linear(C, head_size, bias=False)
# now lets get the output
k = key(x)      # shape : 4,8,16 or (B,T, head_size)
q = query(x)    # shape : 4,8,16 or (B,T, head_size)
 # so the dims arent compatible for batch-multiplication, so we need a transpose
 # k.T wouldnt work becasue we have a batch-dim, so instead we use .transpose and
 # explictily specify the dims we want to be transposed. 
 # this will result in 4,8,16 by 4,16,8 which would give us 4,8,8 which is (B,T,T)
 # really as the result
weight_raw = q@k.transpose(2,1)
# now lets for a moment think about what is happening here, the key and query are applied
# on the input and each return an output of (B,T,head_size), they are in fact, processing
# all the tokens in the input, individually, simultaneously, all at the same time. so each 
# token is both a query, and a key, and when we do a dotproduct, we are basically telling 
# it to reveal the relation/similarity/relevance of every token with every other tokens in
# the same input and as we explained earlier, this is infact our weight matrix (which was 
# initially all zeros) but is now learned from the data!
# print(f'weight_raw\n{weight_raw}')
# now we can apply the constraint on it so that tokens can only communicate with the past 
# so we use the tril trick now!
tril_constrain = torch.tril(torch.ones(size=(T,T)))
# apply the tril on our weight_matrix, so we thanos snaped:d the right half values so each
# token can only talk to its past! (convinietly our weightmatrix is exactly the same dims 
# as our tril! we see why!)
raw_wieghts_masked = weight_raw.masked_fill(tril_constrain==0, float('-inf'))
# apply sotmax to get probablity for each token,
# remember weight is (B,T,T) and we use the last dim, we could use -1 as well, 
# but I wanted to make it explicitly clear here!
weight = raw_wieghts_masked.softmax(dim=2) 
# and finally we can apply our weight on the input (we called it raw for a reason, read on)
# (by the way this is also called self-attention(without tril/masking part) and 
# masked self-attention with the tril/masking part!)
bow_raw = weight@x 
# lets print raw_weights and weights and have some intuitive observations 
print(f'weight_raw\n{weight_raw}')
print(f'raw_weights_masked: {raw_wieghts_masked}')
print(f'weight\n{weight}')
# prints
# raw_weights(unconstrained)
#weight_raw
# tensor([[[ 1.0082e+00,  6.7622e-01, -6.4624e-02,  6.1701e-01, -2.7026e-01,  5.0988e-01,  1.6494e-01,  3.3523e-02],
#          [ 1.3140e+00, -1.1289e-01,  3.8299e-01, -1.1825e+00, -1.5816e+00,  3.9671e-01, -4.1703e-01,  7.0794e-02],
#          [-1.9066e+00,  7.9071e-01, -7.3821e-01,  8.2834e-01,  6.7349e-01,  1.3126e+00, -3.1857e-01, -5.0728e-01],
#          [ 3.0213e+00,  1.9722e+00,  2.0852e-01, -1.2651e+00,  7.8265e-01, -9.2628e-01,  1.1426e+00, -3.1167e-01],
#          [-2.5292e+00, -7.1544e-01,  9.2379e-01,  3.6982e-03,  9.7670e-01, -2.1753e-02, -5.0537e-01, -6.1707e-02],
#          [ 9.9968e-01,  2.9016e+00,  1.2989e+00, -7.0930e-01, -8.8080e-01, -2.6495e-01,  5.8941e-01, -1.2738e+00],
#          [-7.1386e-01,  1.0064e+00, -5.4975e-01, -2.6423e-01, -1.8767e+00,  1.2148e+00, -8.6248e-01,  1.7111e-01],
#          [-5.8812e-01, -1.1953e+00,  5.5418e-01, -2.3265e+00,  8.7663e-01, -9.5643e-01,  3.2523e-01, -4.3643e-01]],

#         [[ 1.0570e+00, -1.7903e+00,  1.4650e-01, -1.4147e+00, -1.7452e+00, -4.5249e+00, -1.9763e+00,  1.8833e+00],
#          [-1.3490e+00,  9.4076e-01, -6.9690e-01,  7.9933e-01,  4.7634e-01,  4.1861e-01,  2.3663e-01,  4.5217e-01],
#          [-4.1173e-01,  7.0261e-01, -4.0204e-01,  9.0092e-01,  1.4083e+00,  2.5488e+00,  1.6350e+00,  1.0048e+00],
#          [-2.8030e+00, -1.9162e+00,  4.6460e-01,  9.0675e-01, -1.1213e+00,  2.7562e+00,  7.0136e-02, -2.0337e+00],
#          [ 4.6742e-01, -2.6938e+00,  2.9156e+00,  2.5135e+00,  7.9011e-01,  4.9236e+00,  3.9545e+00, -2.2600e-01],
#          [ 1.0721e+00,  1.0566e+00,  1.4205e+00, -8.5941e-01,  7.1559e-01,  9.4798e-01, -1.5979e+00,  8.4627e-01],
#          [ 1.7478e-01, -1.3748e+00, -1.6473e+00, -1.9683e+00, -2.7323e-01, -6.0714e+00, -2.3567e+00,  2.2338e+00],
#          [ 5.0234e-01,  2.5284e+00, -2.0990e+00, -9.8971e-01,  1.9566e+00, -4.6640e+00, -1.0950e+00,  7.8260e-01]],

#         [[ 1.2140e+00, -2.1323e+00, -1.3948e+00, -5.6973e-01, -2.7986e-01,  4.3579e+00,  4.8477e-01, -9.7559e-01],
#          [ 8.7786e-01, -2.3352e+00, -2.2988e+00,  1.0322e+00, -1.4231e-01,  5.5233e+00,  1.0819e+00, -2.4722e-01],
#          [-1.0629e+00, -6.2475e-01, -3.5850e-01,  2.9387e-02, -5.6397e-01, -2.4025e-01, -4.2512e-02, -8.4302e-01],
#          [-4.1248e-01,  2.9397e+00,  1.9516e+00,  8.3461e-01, -4.0429e+00, -1.0676e+00, -3.0694e+00, -1.6156e+00],
#          [-2.7064e-01,  2.5970e+00,  4.7168e-01, -2.2572e+00,  1.5132e+00, -4.5593e+00, -3.1755e-01, -1.0504e+00],
#          [ 1.2678e+00, -7.5552e-01, -5.5975e-01,  1.1335e+00, -7.4317e-01,  2.8864e+00, -3.5616e-01, -2.6951e+00],
#          [ 5.6708e-01, -1.3799e+00, -1.1335e+00, -1.7761e+00,  1.7449e+00,  5.7658e-01,  2.0551e+00,  4.9077e-02],
#          [-1.0118e+00, -5.8753e-01,  3.9512e-01,  9.0220e-01, -1.7156e+00,  9.8603e-01, -4.4348e-01,  1.9781e+00]],

#         [[-3.6428e-01, -4.7902e-01,  6.7590e-01, -2.3153e-02,  1.9763e-01,  4.4385e-01,  7.2986e-01, -3.8846e-01],
#          [-6.2810e-01, -8.3713e-01,  1.3468e-01, -5.1328e-01, -2.6653e-02,  3.4325e-01, -1.0015e+00, -1.0334e+00],
#          [-2.3925e-01,  1.6580e+00,  6.6220e-01, -1.9702e+00, -1.0641e+00,  1.7792e-01,  8.0194e-01, -1.5265e-01],
#          [-1.2114e-01, -8.1565e-01, -5.0441e-01,  1.0567e+00,  3.6759e-01,  1.0552e+00, -5.2949e-01, -1.7715e+00],
#          [ 3.3470e-01, -5.7964e-01, -1.1165e+00,  7.8448e-01,  7.9756e-01, -3.0057e+00,  1.3784e+00, -1.2091e+00],
#          [-4.0079e-01, -5.7271e-01,  4.4317e-01,  4.7532e-01,  5.2594e-01,  2.2570e-02, -1.4040e+00,  2.3786e+00],
#          [ 1.3713e-01,  5.4308e-01,  1.6898e+00, -1.6946e+00, -6.3463e-01,  5.4102e-01,  1.3484e+00, -1.6068e+00],
#          [ 1.9584e-01,  4.7083e-01,  3.8171e-01, -3.8883e-01,  2.3697e-01,  8.6615e-01, -3.8587e-01,  2.5004e+00]]],
#        grad_fn=<UnsafeViewBackward0>)
#
# raw_masked_wieghts(constrained):
# tensor([[[ 1.0082e+00,        -inf,        -inf,        -inf,        -inf,        -inf,        -inf,        -inf],
#          [ 1.3140e+00, -1.1289e-01,        -inf,        -inf,        -inf,        -inf,        -inf,        -inf],
#          [-1.9066e+00,  7.9071e-01, -7.3821e-01,        -inf,        -inf,        -inf,        -inf,        -inf],
#          [ 3.0213e+00,  1.9722e+00,  2.0852e-01, -1.2651e+00,        -inf,        -inf,        -inf,        -inf],
#          [-2.5292e+00, -7.1544e-01,  9.2379e-01,  3.6982e-03,  9.7670e-01,        -inf,        -inf,        -inf],
#          [ 9.9968e-01,  2.9016e+00,  1.2989e+00, -7.0930e-01, -8.8080e-01, -2.6495e-01,        -inf,        -inf],
#          [-7.1386e-01,  1.0064e+00, -5.4975e-01, -2.6423e-01, -1.8767e+00,  1.2148e+00, -8.6248e-01,        -inf],
#          [-5.8812e-01, -1.1953e+00,  5.5418e-01, -2.3265e+00,  8.7663e-01, -9.5643e-01,  3.2523e-01, -4.3643e-01]],

#         [[ 1.0570e+00,        -inf,        -inf,        -inf,        -inf,        -inf,        -inf,        -inf],
#          [-1.3490e+00,  9.4076e-01,        -inf,        -inf,        -inf,        -inf,        -inf,        -inf],
#          [-4.1173e-01,  7.0261e-01, -4.0204e-01,        -inf,        -inf,        -inf,        -inf,        -inf],
#          [-2.8030e+00, -1.9162e+00,  4.6460e-01,  9.0675e-01,        -inf,        -inf,        -inf,        -inf],
#          [ 4.6742e-01, -2.6938e+00,  2.9156e+00,  2.5135e+00,  7.9011e-01,        -inf,        -inf,        -inf],
#          [ 1.0721e+00,  1.0566e+00,  1.4205e+00, -8.5941e-01,  7.1559e-01,  9.4798e-01,        -inf,        -inf],
#          [ 1.7478e-01, -1.3748e+00, -1.6473e+00, -1.9683e+00, -2.7323e-01, -6.0714e+00, -2.3567e+00,        -inf],
#          [ 5.0234e-01,  2.5284e+00, -2.0990e+00, -9.8971e-01,  1.9566e+00, -4.6640e+00, -1.0950e+00,  7.8260e-01]],

#         [[ 1.2140e+00,        -inf,        -inf,        -inf,        -inf,        -inf,        -inf,        -inf],
#          [ 8.7786e-01, -2.3352e+00,        -inf,        -inf,        -inf,        -inf,        -inf,        -inf],
#          [-1.0629e+00, -6.2475e-01, -3.5850e-01,        -inf,        -inf,        -inf,        -inf,        -inf],
#          [-4.1248e-01,  2.9397e+00,  1.9516e+00,  8.3461e-01,        -inf,        -inf,        -inf,        -inf],
#          [-2.7064e-01,  2.5970e+00,  4.7168e-01, -2.2572e+00,  1.5132e+00,        -inf,        -inf,        -inf],
#          [ 1.2678e+00, -7.5552e-01, -5.5975e-01,  1.1335e+00, -7.4317e-01,  2.8864e+00,        -inf,        -inf],
#          [ 5.6708e-01, -1.3799e+00, -1.1335e+00, -1.7761e+00,  1.7449e+00,  5.7658e-01,  2.0551e+00,        -inf],
#          [-1.0118e+00, -5.8753e-01,  3.9512e-01,  9.0220e-01, -1.7156e+00,  9.8603e-01, -4.4348e-01,  1.9781e+00]],

#         [[-3.6428e-01,        -inf,        -inf,        -inf,        -inf,        -inf,        -inf,        -inf],
#          [-6.2810e-01, -8.3713e-01,        -inf,        -inf,        -inf,        -inf,        -inf,        -inf],
#          [-2.3925e-01,  1.6580e+00,  6.6220e-01,        -inf,        -inf,        -inf,        -inf,        -inf],
#          [-1.2114e-01, -8.1565e-01, -5.0441e-01,  1.0567e+00,        -inf,        -inf,        -inf,        -inf],
#          [ 3.3470e-01, -5.7964e-01, -1.1165e+00,  7.8448e-01,  7.9756e-01,        -inf,        -inf,        -inf],
#          [-4.0079e-01, -5.7271e-01,  4.4317e-01,  4.7532e-01,  5.2594e-01,  2.2570e-02,        -inf,        -inf],
#          [ 1.3713e-01,  5.4308e-01,  1.6898e+00, -1.6946e+00, -6.3463e-01,  5.4102e-01,  1.3484e+00,        -inf],
#          [ 1.9584e-01,  4.7083e-01,  3.8171e-01, -3.8883e-01,  2.3697e-01,  8.6615e-01, -3.8587e-01,  2.5004e+00]]],
#        grad_fn=<MaskedFillBackward0>)
# 
# weight (final- normalized)
# tensor([[[1.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
#          [8.0641e-01, 1.9359e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
#          [5.2476e-02, 7.7872e-01, 1.6880e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
#          [7.0222e-01, 2.4596e-01, 4.2162e-02, 9.6594e-03, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
#          [1.1816e-02, 7.2474e-02, 3.7333e-01, 1.4877e-01, 3.9362e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00],
#          [1.0348e-01, 6.9321e-01, 1.3957e-01, 1.8735e-02, 1.5783e-02, 2.9217e-02, 0.0000e+00, 0.0000e+00],
#          [5.7514e-02, 3.2129e-01, 6.7772e-02, 9.0167e-02, 1.7979e-02, 3.9571e-01, 4.9572e-02, 0.0000e+00],
#          [7.3912e-02, 4.0274e-02, 2.3164e-01, 1.2995e-02, 3.1978e-01, 5.1140e-02, 1.8424e-01, 8.6020e-02]],
#
#         [[1.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
#          [9.1976e-02, 9.0802e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
#          [1.9773e-01, 6.0261e-01, 1.9966e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
#          [1.4181e-02, 3.4422e-02, 3.7221e-01, 5.7918e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
#          [4.6023e-02, 1.9501e-03, 5.3236e-01, 3.5612e-01, 6.3550e-02, 0.0000e+00, 0.0000e+00, 0.0000e+00],
#          [1.9494e-01, 1.9195e-01, 2.7619e-01, 2.8253e-02, 1.3648e-01, 1.7219e-01, 0.0000e+00, 0.0000e+00],
#          [4.5214e-01, 9.6007e-02, 7.3108e-02, 5.3035e-02, 2.8887e-01, 8.7621e-04, 3.5963e-02, 0.0000e+00],
#          [6.8046e-02, 5.1605e-01, 5.0474e-03, 1.5304e-02, 2.9133e-01, 3.8823e-04, 1.3775e-02, 9.0057e-02]],
#
#         [[1.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
#          [9.6132e-01, 3.8675e-02, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
#          [2.1870e-01, 3.3895e-01, 4.4235e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
#          [2.2894e-02, 6.5398e-01, 2.4345e-01, 7.9674e-02, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
#          [3.7332e-02, 6.5690e-01, 7.8427e-02, 5.1206e-03, 2.2222e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00],
#          [1.3611e-01, 1.7995e-02, 2.1887e-02, 1.1900e-01, 1.8219e-02, 6.8680e-01, 0.0000e+00, 0.0000e+00],
#          [9.8946e-02, 1.4120e-02, 1.8065e-02, 9.5010e-03, 3.2130e-01, 9.9891e-02, 4.3817e-01, 0.0000e+00],
#          [2.3306e-02, 3.5621e-02, 9.5163e-02, 1.5801e-01, 1.1530e-02, 1.7183e-01, 4.1141e-02, 4.6340e-01]],
#
#         [[1.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
#          [5.5207e-01, 4.4793e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
#          [9.8708e-02, 6.5816e-01, 2.4313e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
#          [1.8422e-01, 9.1987e-02, 1.2557e-01, 5.9822e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
#          [2.0870e-01, 8.3642e-02, 4.8896e-02, 3.2723e-01, 3.3154e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00],
#          [9.4141e-02, 7.9271e-02, 2.1893e-01, 2.2608e-01, 2.3782e-01, 1.4376e-01, 0.0000e+00, 0.0000e+00],
#          [7.8726e-02, 1.1815e-01, 3.7190e-01, 1.2607e-02, 3.6387e-02, 1.1790e-01, 2.6434e-01, 0.0000e+00],
#          [5.6646e-02, 7.4575e-02, 6.8217e-02, 3.1568e-02, 5.9024e-02, 1.1073e-01, 3.1662e-02, 5.6757e-01]]], grad_fn=<SoftmaxBackward0>)
#
# as you can see, our weight is initialized for each batch based on the input data, and each token
# has its own weight, that is they are not uniform!, to get a better understanding lets consider 
# the first batch of raw-weights[0], raw_masked_wieghts[0] and weight[0]:
#
# raw_weights[0]
# tensor([[[ 1.0082e+00,  6.7622e-01, -6.4624e-02,  6.1701e-01, -2.7026e-01,  5.0988e-01,  1.6494e-01,  3.3523e-02],
#          [ 1.3140e+00, -1.1289e-01,  3.8299e-01, -1.1825e+00, -1.5816e+00,  3.9671e-01, -4.1703e-01,  7.0794e-02],
#          [-1.9066e+00,  7.9071e-01, -7.3821e-01,  8.2834e-01,  6.7349e-01,  1.3126e+00, -3.1857e-01, -5.0728e-01],
#          [ 3.0213e+00,  1.9722e+00,  2.0852e-01, -1.2651e+00,  7.8265e-01, -9.2628e-01,  1.1426e+00, -3.1167e-01],
#          [-2.5292e+00, -7.1544e-01,  9.2379e-01,  3.6982e-03,  9.7670e-01, -2.1753e-02, -5.0537e-01, -6.1707e-02],
#          [ 9.9968e-01,  2.9016e+00,  1.2989e+00, -7.0930e-01, -8.8080e-01, -2.6495e-01,  5.8941e-01, -1.2738e+00],
#          [-7.1386e-01,  1.0064e+00, -5.4975e-01, -2.6423e-01, -1.8767e+00,  1.2148e+00, -8.6248e-01,  1.7111e-01],
#          [-5.8812e-01, -1.1953e+00,  5.5418e-01, -2.3265e+00,  8.7663e-01, -9.5643e-01,  3.2523e-01, -4.3643e-01]],
#
#
# raw_masked_wieghts[0]
# tensor([[[ 1.0082e+00,        -inf,        -inf,        -inf,        -inf,        -inf,        -inf,        -inf],
#          [ 1.3140e+00, -1.1289e-01,        -inf,        -inf,        -inf,        -inf,        -inf,        -inf],
#          [-1.9066e+00,  7.9071e-01, -7.3821e-01,        -inf,        -inf,        -inf,        -inf,        -inf],
#          [ 3.0213e+00,  1.9722e+00,  2.0852e-01, -1.2651e+00,        -inf,        -inf,        -inf,        -inf],
#          [-2.5292e+00, -7.1544e-01,  9.2379e-01,  3.6982e-03,  9.7670e-01,        -inf,        -inf,        -inf],
#          [ 9.9968e-01,  2.9016e+00,  1.2989e+00, -7.0930e-01, -8.8080e-01, -2.6495e-01,        -inf,        -inf],
#          [-7.1386e-01,  1.0064e+00, -5.4975e-01, -2.6423e-01, -1.8767e+00,  1.2148e+00, -8.6248e-01,        -inf],
#          [-5.8812e-01, -1.1953e+00,  5.5418e-01, -2.3265e+00,  8.7663e-01, -9.5643e-01,  3.2523e-01, -4.3643e-01]],
# 
# weight[0]
# tensor([[[1.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
#          [8.0641e-01, 1.9359e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
#          [5.2476e-02, 7.7872e-01, 1.6880e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
#          [7.0222e-01, 2.4596e-01, 4.2162e-02, 9.6594e-03, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
#          [1.1816e-02, 7.2474e-02, 3.7333e-01, 1.4877e-01, 3.9362e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00],
#          [1.0348e-01, 6.9321e-01, 1.3957e-01, 1.8735e-02, 1.5783e-02, 2.9217e-02, 0.0000e+00, 0.0000e+00],
#          [5.7514e-02, 3.2129e-01, 6.7772e-02, 9.0167e-02, 1.7979e-02, 3.9571e-01, 4.9572e-02, 0.0000e+00],
#          [7.3912e-02, 4.0274e-02, 2.3164e-01, 1.2995e-02, 3.1978e-01, 5.1140e-02, 1.8424e-01, 8.6020e-02]],
# 
# take the last token, which is 8.6020e-02, notice this token, not only knows its content and own 
# position in the sequence(its the 8th token after all!) but also knows which tokens comes before it
# and how much its related to any of them.(basically when it knows which token comes before it, it 
# creates its own query so to speak, and talks to every single previous token to find about which ones
# are more or less relavent to it and to what extend.) 
#
# lets expand on this a bit more using an example, how could we interpret these numbers?  
# obviously our input is random, and the relationship between each token is random as well. 
# but if for a moment we imagine these to be the values for a model at the begining of the training, 
# we may find some intersting intuitions. obviously this would be much more coherent and clear when 
# working on an already trained set of weights. 
# nonetheless, imagine these weights to belong to a sentence like "The cat sat on the red mat." 
# (which we just made up!) our tokens would be ["The", "cat", "sat", "on", "the", "red","mat", "."].
# by looking at the weights we may infer: 
#
# 1.The first row would correspond to the word "The". The attention is entirely on itself (1.0000),
# and not on any other words (0.0000), which makes sense as there are no previous words to attend to.
# 
# 2.The second row would correspond to the word "cat". It attends mostly to "The" (0.8064) and a bit
# to itself "cat" (0.1936). This could be because the model is learning that "The" often precedes a 
# noun.
#
# 3.The third row would correspond to "sat". It attends mostly to "cat" (0.7787), then to "sat" (0.1688),
# and a bit to "The" (0.0525). This could be because "sat" is a verb that is often associated with the 
# subject "cat".
#
# 4.The fourth row would correspond to "on". It attends mostly to "The" (0.7022), then to "cat" (0.2459), 
# a bit to "sat" (0.0422), and very little to "on" (0.0097). This could be because prepositions like "on" 
# often relate to the subject and verb in a sentence, hence higher number for "The" and "cat".
# 
# 5.The sixth row would correspond to "mat". It attends mostly to "red" (0.3957), then to "cat" (0.3213),
# a bit to "on" (0.0902) and "sat" (0.0678). This could be because "mat" is the object where the "cat"
# "sat" "on". and intrestingly the "red" is highly related to "mat", overall conveying all the important
# semantics and relationships.
# 
# as you noticed I didnt include other tokens as they wouldnt make much sense considering they are
# random but we could comeup with an example nevertheless that could give us an intuive understanding
# of whats going on in a typical attention weight matrix.
# below is a simple heatmap that shows the same wieghts which hopefully give you an evern better mental
# image:

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

tokens = ["The", "cat", "sat", "on", "the", "red", "mat", "."]
plt.figure(figsize=(8, 8))

# we can use this oneliner using seaborn and create a heatmap plot.
# sns.heatmap(weight[0].detach().numpy(), annot=True, cbar=True, fmt=".4f", xticklabels=tokens, yticklabels=tokens, cmap='hot')
# or use the old way using matplotlib:
plt.imshow(weight[0].detach().numpy(), cmap='Blues', interpolation='nearest')
plt.colorbar()
plt.title("weight[0]")
plt.xticks(np.arange(len(tokens)), tokens, rotation=45)
plt.yticks(np.arange(len(tokens)), tokens)
# Add the attention weights on each cell
for pos in range(len(tokens)):
    for i in range(len(tokens)):
        text = plt.text(i, pos, round(weight[0].detach().numpy()[pos, i], 4),
                       ha="center", va="center", color="w")
plt.show()
#
# now back at our own example, lets visualize the weights for better understanding
def plot_heatmap(weight, label, cmap='plasma', annotate=False):
    # lets draw the heatmap for each tensor 
    plt.figure(figsize=(12,6))
    sns.heatmap(weight.numpy(),cmap=cmap, annot=annotate,)
    plt.title(label)
    # Add the attention weights on each cell
    # if annotate:
        # plt.imshow(weight.numpy(),cmap=cmap)
        # plt.colorbar()
        # for i in range(weight.shape[0]):
            # for j in range(weight.shape[1]):
                # plt.text(j, i, round(weight.numpy()[i, j], 4),
                            # ha="center", va="center", color="w")
    plt.show()
    
    
plot_heatmap(tril_constrain.detach(), label='tril_constrain', cmap='Blues', annotate=True)
# sidenote:
# note that in attention, samples do not interact with each other at all. when we have a batch of 4, 
# each sample is processed in isolation, but in parallell to other samples.
# therefore since we have a batch here, basically 4 samples, we flatten the first two dims 
# so we can plot all as a 2d heatmap and see this visually. 
plot_heatmap(raw_wieghts_masked.view(-1,8).detach(), label='raw_wieghts_masked-all batches',cmap='RdBu')
# now lets look at the first batch in more detail
plot_heatmap(raw_wieghts_masked[0].detach(), label='raw_wieghts_masked-first batch',cmap='RdBu')
# since the colors are not that defined, maybe gray cmap shows this better! but either are ok
plot_heatmap(weight[0].detach(),'weight-first batch',cmap='Blues')
plot_heatmap(bow_raw[0].detach(),'bow_raw-first batch')

# So far we learned that when we get a high relevancy scale/response in the weights, when we do a 
# softmax it will assign a large probablity to them, informing the network that we need more 
# information from them, effectively allowing for aggregating a lot of their information into our
# position and we happen to learn more about them this way.
# However, in practice, we are not intrested in aggregating the 'inputs' raw values per say, rather 
# we want their information, so instead of just using the raw values of our input(x), we instead use
# a representation of them. 
# this is achieved using a third vector known as, 'value' and is the last vector we use. 
# just like the key and query, we set its bias to False, so we only get a simple vector,
# and a dotproduct output.(again this is the intuition, but how much adding a bias would
# affect this we will see later, for now we stick to the default no bias version)
#
# produces (B,T,head_size) just like the previous two key,query vectors
value = torch.nn.Linear(C,head_size, bias=False) 
x_processed = value(x)
# so 
bow_final = weight@x_processed
# we are not done yet.
# scaled-attention:
# so far we implemented the attention based on the original transformer paper(minus some differences we 
# get to later on, note that the attention mechanism predates transformer and was first intruced in 2014
# by bahdanu etal)
# except the part where we need to divide by the sqrt of the head_size. that is called scaled-attention
# so lets talk about this, and see why its needed.
# the reason is simple, without it, the probablities will be saturated.
# when we add this term to the calculations, it will make the 'weight' matrix to be 'unit variance', 
# when Q and K are unit variance. this in turn allows softamx to stay diffuse and not saturate too much.
# in other words, if we simply multiply key and query like that, the variance of the resulting weight matrix 
# will be around the head_size instead of 1 which is bad and makes optimization really hard.
# to see this effect consider the following example
k = torch.randn(size=(B,T,head_size))
q = torch.randn(size=(B,T,head_size))
w_unscaled = q@k.transpose(-2, -1) 
print(f'{k.var()=}')
print(f'{q.var()=}')
print(f'{w_unscaled.var()=}')
# now add the 1/sqrt(head_size)
w_scaled = q@k.transpose(-2, -1) * head_size**-0.5 
print(f'after applying 1/sqrt(head_size)')
# makes the weight variance 1!
print(f'{w_scaled.var()=}')
# why is it important? if you recall, the weight matrix is fed into softmax, so its really important 
# that weight matrix be fairly diffuse especially during initialization, if we look at the weight matrix here, 
# we'll notice that they are fairly diffuse now. 
print(f'weight_scaled[0]:\n{w_scaled[0]}')
# prints 
# weight_scaled[0]:
# tensor([[ 0.3972, -2.0957, -0.5396, -1.4860, -0.8393,  0.6273, -0.0157,  0.6284],
#         [ 1.3588, -0.0440,  2.1040, -0.2796,  1.7797,  1.4362,  1.3843,  0.0368],
#         [ 0.8109,  1.7400,  1.3579,  0.2097,  1.7152, -1.1242, -0.4349, -0.5690],
#         [ 0.0513,  2.8303,  0.7332,  0.0041,  0.9688, -1.6174, -1.4255,  0.2869],
#         [-0.7819, -0.6691, -1.4017, -0.4155, -0.8568, -0.2704, -0.9453,  0.3763],
#         [ 0.4092, -0.3079,  0.8472, -1.1753, -0.7699,  0.5091,  0.6385,  0.9677],
#         [ 0.3043, -2.1402, -2.2582, -0.3573, -1.2838, -0.0260, -0.0513,  0.9151],
#         [ 0.4180, -2.8353, -0.6435,  0.4842, -3.0202,  1.7690,  1.8837, -0.4393]])
#
# when softmax is applied:
masked_w_scaled_probs = w_scaled.masked_fill(torch.tril(torch.ones(T,T))==0,float('-inf')).softmax(dim=-1)[0]
print(masked_w_scaled_probs)
# tensor([[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.8026, 0.1974, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.1901, 0.4814, 0.3285, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.0499, 0.8038, 0.0987, 0.0476, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.1989, 0.2226, 0.1070, 0.2869, 0.1845, 0.0000, 0.0000, 0.0000],
#         [0.2148, 0.1049, 0.3329, 0.0440, 0.0661, 0.2374, 0.0000, 0.0000],
#         [0.3027, 0.0263, 0.0233, 0.1562, 0.0618, 0.2176, 0.2121, 0.0000],
#         [0.0901, 0.0035, 0.0312, 0.0962, 0.0029, 0.3478, 0.3901, 0.0382]])
#
# now compare it with the unscaled weight : 
print(f'weight_unscaled[0]:\n{w_unscaled[0]}')
# weight_unscaled[0]:
# tensor([[  1.5889,  -8.3829,  -2.1583,  -5.9440,  -3.3573,   2.5092,  -0.0629,  2.5135],
#         [  5.4350,  -0.1759,   8.4159,  -1.1183,   7.1189,   5.7446,   5.5373,  0.1472],
#         [  3.2435,   6.9600,   5.4317,   0.8388,   6.8607,  -4.4969,  -1.7396, -2.2761],
#         [  0.2051,  11.3214,   2.9327,   0.0165,   3.8754,  -6.4696,  -5.7019,  1.1475],
#         [ -3.1278,  -2.6763,  -5.6069,  -1.6621,  -3.4272,  -1.0816,  -3.7811,  1.5053],
#         [  1.6368,  -1.2317,   3.3888,  -4.7012,  -3.0795,   2.0364,   2.5541,  3.8710],
#         [  1.2171,  -8.5610,  -9.0327,  -1.4293,  -5.1353,  -0.1038,  -0.2053,  3.6603],
#         [  1.6719, -11.3411,  -2.5740,   1.9369, -12.0810,   7.0760,   7.5347, -1.7573]])
#
# when softmax is applied:
masked_w_unscaled_probs = w_unscaled.masked_fill(torch.tril(torch.ones(T,T))==0,float('-inf')).softmax(dim=-1)[0]
print(masked_w_unscaled_probs)
# tensor([[1.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
#         [9.9636e-01, 3.6442e-03, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
#         [1.9592e-02, 8.0566e-01, 1.7475e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
#         [1.4865e-05, 9.9975e-01, 2.2738e-04, 1.2309e-05, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
#         [1.2943e-01, 2.0328e-01, 1.0848e-02, 5.6050e-01, 9.5940e-02, 0.0000e+00, 0.0000e+00, 0.0000e+00],
#         [1.2012e-01, 6.8207e-03, 6.9264e-01, 2.1236e-04, 1.0748e-03, 1.7913e-01, 0.0000e+00, 0.0000e+00],
#         [6.3261e-01, 3.5856e-05, 2.2371e-05, 4.4856e-02, 1.1023e-03, 1.6883e-01, 1.5254e-01, 0.0000e+00],
#         [1.7351e-03, 3.8710e-09, 2.4850e-05, 2.2616e-03, 1.8472e-09, 3.8572e-01, 6.1020e-01, 5.6236e-05]])
#
# as you can see, some values are very negative, while others are very positive, for example, we have both 11 and -11 
# in the unscaled weight matrix and this is a recipe for disaster! 
# The reason it is a problem is that, with softamx, when numbers take very positive
# and nagative values, softmax actually converges towards one-hot-vector. 
# basically the values shrink towards the max value.
plot_heatmap(masked_w_unscaled_probs.view(-1,8).detach(),'unscaled masked-weight-probs(probs look more like onehot)',cmap='hot',annotate=True)
plot_heatmap(masked_w_scaled_probs.view(-1,8).detach(),'scaled masked-weight-probs(probs are more uniform)',cmap='hot',annotate=True)
# note that in the unscaled weights, the majority of values are either black(close to zero) or very bright
# where as in the scaled version, all values seem to be fairly uniform (they are all shades of red).
# compre the last two rows in these two weight matrixes, and you'll notice the unscaled one seems its convering
# towards a one-hot vector inwhich nearly all values are close to zero except one or two value having a much
# larger value.
# to further expand on this, consider the following example, it should demonstrate it more clearly and solidfy this concept for you.
# lets apply softmax on a list of numbers that are close to each other, we see the probablities are difuse and normal
print(f'{torch.softmax(torch.tensor([0.1,0.5,-0.3,-0.2]),dim=-1)}')
# we get a diffused probablity out of softmax
# tensor([0.2562, 0.3822, 0.1717, 0.1898])
# however if we increase the magnitude of the numbers(sharpen them) (and thus also the difference between them) 
# by multiplying by a number like 8 e.g. (just to simulate the effect here and compare it with the previous case)
print(f'{torch.softmax(torch.tensor([0.1,0.5,-0.3,-0.2])*8, dim=-1)}')
# we see the softmax starts to sharpen towards the max, shrinking other values except the largest number, and
# effectively converging toward a one-hot-encoded vector.
# tensor([0.0390, 0.9559, 0.0016, 0.0035])
#
# Therefore we dont want these values to be extreme, especially during initialization or otherwise, softmax will
# be way too picky!/gravitated toward them which means, we are basically aggregating the information from a 
# single node/token instead of multiple ones (becasue one has the largest value, its as if our sequence length
# is 1 or all the previous tokens are just 0s! and like that, we lose access to the wealth of information 
# the past history offers)
# so we want the probablities to be diffuse/not peaked like the second example here.
# so this scaling is used to retain the variance at a good value especially at initialization.
# so now that we are finally finished the self attention head, lets implement it as a module and incorporate everything
# we just discussed here.
#
class AttentionHead(nn.Module):
    def __init__(self, context_size, embd_size, head_size, use_bias=False) -> None:
        super().__init__()
        # we need context_size or block_size for creating the tril constrain
        self.context_size = context_size
        # we want this as the input dim for our key,query and value, this is 
        # infact the input_dim, since we plan on using the embeddings, we named
        # it embd_size
        self.embd_size = embd_size
        # head_size is the output dimension of our attnetion
        # note that usually the head_size is equal to embd_size 
        # and this is one of the reasons the transformers
        # overhead increases rapidly!
        self.head_size = head_size
        self.key = nn.Linear(embd_size, head_size, bias=use_bias)
        self.query = nn.Linear(embd_size, head_size, bias=use_bias)
        self.value = nn.Linear(embd_size, head_size, bias=use_bias)

        # use buffer for trail, since this is a buffer, we can use the self.register_buffer which is
        # inherited from nn.Module class, to add it as buffer to the module, so it can be saved in the
        # state_dict when we save the model.  tril doesnt change, and is not updated(its required_grad is false),
        # so it being saved in state_dict doesnt matter to us, but to demonstrate this feature of pytorch, 
        # we are using it, and its a good practice, since pytorch knows how to deal with it and wont include it
        # in the computaion graph anyway!
        # 
        # This is typically used to register a buffer that should not to be considered a model parameter. 
        # For example, BatchNorm's running_mean is not a parameter, but is part of the module's state. 
        # Buffers, by default, are persistent and will be saved alongside parameters. This
        # behavior can be changed by setting persistent to False. The only difference between a persistent
        # buffer and a non-persistent buffer is that the latter will not be a part of this module's state_dict.
        # tril allows us to impose constrain by creating a lower traingualr matrix and setting
        # the rest entries to zero, effectively preventing tokens of the future from communicating with the past
        # each token can only communicate with the previous tokens that came before it.
        self.register_buffer('tril', torch.tril(torch.ones(context_size,context_size)))

    def __call__(self, inputs:torch.Tensor) -> torch.Tensor:
        B,T,C = inputs.shape # (4,8,16)
        # create the weight by using k,q,v
        k = self.key(inputs)    #! (B,T,C)  e.g. (4,8,16)
        q = self.query(inputs)  #! (B,T,C)  e.g. (4,8,16)
        v = self.value(inputs)  #! (B,T,C)  e.g. (4,8,16)
        # create the weight matrix and scale it by 1/sqrt(head_size) to keep weight unit variance 
        weight = q@k.transpose(-2,-1)* self.head_size**-0.5 # !(B,T,T)
        # apply the tril constraint - (this makes this suitable for a decoder block only!)
        # important note: notice we used tril[:T,:T] and not simply tril
        # this is because, when the input has a small context_size < self.context_size
        # like when we want to generate inputs with sth like zeros((1,1)) which says
        # there is a single token (context_size 1) of 0 as the begining of the sequence,
        # when this is input, the tril by default makes a (context_size, context_size) matrix
        # which will be different than the input context_size which is 1 e.g. or 2 e.g.
        # and it will fail becasue our weight would be (1,1,1) or (1,2,2), but trail is (1,8,8)
        # and clearly this will cause an error. For this reason, we always create the 
        # tril check dynamcally by explicitly specifying the context_size based on the 
        # current input context_size so in case the context_size is smaller, tril is resized
        # dynamically accordingly.
        # to see this try uncommenting these lines and use an input with context_size of 1,2
        # basically smaller than context_size and see how it fails if you dont use :T,:T notion here
        # print(f'tril[:T,:T]==0: {(self.tril[:T,:T]==0).shape}')
        # print(f'tril==0: {(self.tril==0).shape}')
        weight = weight.masked_fill(self.tril[:T,:T]==0,float('-inf'))
        # weight2 = weight.masked_fill(self.tril==0,float('-inf'))
        # print(f'{weight.shape=}')
        # print(f'{weight2.shape=}')
        # note that we are using batch, so instead of hardcoding 2,
        # we simply use -1 to refer to the last dim
        weight = weight.softmax(dim=-1)
        # finally apply the weight on the v
        bow = weight@v # (B,context_sz,head_sz)
        return bow        

# now lets test our attention module now: 
at  = AttentionHead(8,16,18)
x = torch.randn(size=(4,8,16))
print(at(x).shape)
# 
#%%
import torch, torch.nn as nn
# side-quest!:
# TODO:  add the efficient/fused version 
# before we add this to our base model, lets check
# and see if fusing kqv can improve our speed!
# we might think its not really that efficient to
# provided that usually when we have multiple operations, 
# its much better to create 1 larger operation than 
# having several smaller ones, and when it comes to 
# multiplication we can do better. for example, 
# we can calculate k,q,v in one go! and be more efficient!
# for this we need to merge their weights, do the calcs,
# and then split the result! 
# well, lets first see how its done and then
# do a simple benchmark to see if it actually is any better!
# to give you an intuitive undrestanding try the following example,
# suppose we have our k,q,v layers
# key = nn.Linear(5,5, bias=False)
# query = nn.Linear(5,5, bias=False)
# value = nn.Linear(5,5, bias=False)
# x = torch.randn(size = (3,2,5))
# and we want to calculate their outputs. 
# before we do the multiplication, lets make sure their weights
# are something fixed so we can easily see for ourselves whats going on. 
# since the key,query and values' weights are (5,5), we have 25 values for each, 
# lets initilize them with 0-25! and then reshape them to the proper form.
# key.weight.data   =  torch.arange(25).view(5,5).float()
# query.weight.data =  torch.arange(25).view(5,5).float()
# value.weight.data =  torch.arange(25).view(5,5).float()
# now lets calculte their outputs :
# k,q,v = [m(x) for m in (key, query, value)]
# now we said that we can do a single multiplication 
# instead of 3 by merging the weights of key,query and
# value how do we do that? 
# we simply create a linear layer with 3 times the output size of the initial size
# used for k,q,v layers. 
# kqv = nn.Linear(5,5*3, bias=False)
# to work with the same values, with repeate the weights 3 times and then reshape
# kqv.weight.data = torch.arange(25).repeat(3).view(5*3,5).float()
# now if we calculate the output of kqv, we should see the outputs match
# kqv_output = kqv(x)
# but the shape is (3,2,15)! shapes dont match! how are we going to compare them then? 
# no problem we split the last dimension into 3 seprate tensors!
# k2,q2,v2 = kqv_output.split(5, dim=-1)
# and now if we do 
# print(torch.allclose(k,k2), torch.allclose(q,q2), torch.allclose(v,v2))
# we get True,True,True , signifying they are actually doing the very same thing!
# note that sometimes this may return false, due to numerical instability in floats
# but these two operations are equivalent 100%. 
# to be certain you can use int instead and see the output of these two sets
# of operations are identical.
# key = nn.Linear(5,5, bias=False)
# query = nn.Linear(5,5, bias=False)
# value = nn.Linear(5,5, bias=False)
# x = torch.randint(30, size=(3,2,5))
# 
# #we set requires_gradient to false becasue a tensor that 
# requires gradients must be floating point/complex dtype
# [m.requires_grad_(False) for m in (key, query, value)]
# key.weight.data   =  torch.arange(25).view(5,5)
# query.weight.data =  torch.arange(25).view(5,5)
# value.weight.data =  torch.arange(25).view(5,5)
# k,q,v = [m(x) for m in (key, query, value)]
# 
# # same here, becasue we are setting the weights value to int,
# we set require_gradients to false
# kqv = nn.Linear(5,5*3, bias=False).requires_grad_(False)
# kqv.weight.data = torch.arange(25).repeat(3).view(5*3,5)
# kqv_output = kqv(x)
# k2,q2,v2 = kqv_output.split(5, dim=-1)
# print(*[(o1==o2).all() for o1,o2 in zip((k,q,v),(k2,q2,v2))])
# 
# now that we know how to implement this, lets implement our attention head using this new trick!
# and see if its any faster! 

class AttentionHeadFused(nn.Module):
    def __init__(self, context_size, embd_size, head_size, use_bias=False) -> None:
        super().__init__()
        self.context_size = context_size
        self.embd_size = embd_size
        # headsize is not needed as its value is the same as embd!
        self.head_size = head_size
        self.use_bias = use_bias
        # now instead of having 3 separate linear layers for key,query and value with the same shape
        # we create 1 large linear layer to do all the ops in one go instead of 3
        self.kqv = nn.Linear(embd_size, head_size*3,bias=False)
        # since we are in an autoregressive model we need to impose a constrain so that a token
        # can only interact with the previous tokens. we use torch.tril to create a lower triangle
        # that contains the numbers, wile the upper triangle is all zeros.
        # since this wont be optimized, we mark it as a buffer and register accordingly 
        # when we do this, pytorch adds this to our model state_dict, and does not include it in
        # the optimization process!
        self.register_buffer('tril',torch.tril(torch.ones(context_size,context_size)))
        
    
    def forward(self, inputs):
        kqv_out = self.kqv(inputs)
        # since we need some info about the inputs shape such as sequence length 
        # we extract them here for ease of use
        B,T,C  = inputs.shape
        k,q,v = kqv_out.split(self.head_size, dim=-1)
        # now lets calculate our weight matrix from the data
        weight = q@k.transpose(-2,-1) * self.head_size**-0.5
        # we mask the weights based on tril, and set all enetries that are 0 to -inf
        # so when doing softmax, -infs become 0 and everything comes to place!
        # note that our input sequence may be as small as 1 up to context_size
        # so we need to account for this as well (the case of sequence length of 1
        # is when at test time we decide to generate some text using the starting token
        # which can be of any length, 1 or more. this way we dont face errors when token
        # size in input is 1!)
        # this is what makes our attention head, (self)masked-attention!
        weight = weight.masked_fill(self.tril[:T, :T]== 0, float('-inf'))
        # and now do a softmax so we get probablities 
        weight = weight.softmax(dim=-1)
        # now we have weights lets apply it on our values (representation of inputs)
        bow = weight@v
        return bow 
    
cs = 128
es = 128
hs  = 128
torch.backends.cudnn.benchmark = True
at  = AttentionHead(cs,es,hs).cuda()
at2 = AttentionHeadFused(cs,es,hs).cuda()
x = torch.randn(size=(128,cs,es)).cuda()
# import timeit
%timeit -n 100 at(x) 
%timeit -n 100 at2(x) 

print(f"Number of parameters in original layers: {sum(p.numel() for p in at.key.parameters()) +sum(p.numel() for p in at.query.parameters()) +sum(p.numel() for p in at.value.parameters()):,}")
print(f"Number of parameters in fused layer: {sum(p.numel() for p in at2.kqv.parameters()):,}")

# cs=es=hs=64 - cuda
# 71.3 µs ± 1.9 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
# 57.9 µs ± 1.5 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
# larger size., cs=ds=es=3096 batch:2
# 36.7 ms ± 10.5 ms per loop (mean ± std. dev. of 7 runs, 100 loops each)
# 33.8 ms ± 2.96 ms per loop (mean ± std. dev. of 7 runs, 100 loops each)
# larger size., cs=ds=es=3096 batch:8
# 136 ms ± 50.9 ms per loop (mean ± std. dev. of 7 runs, 100 loops each)
# 140 ms ± 4.04 ms per loop (mean ± std. dev. of 7 runs, 100 loops each)
# larger size., cs=ds=es=3096 batch:4
# 69.1 ms ± 25.9 ms per loop (mean ± std. dev. of 7 runs, 100 loops each)
# 76.7 ms ± 4.69 ms per loop (mean ± std. dev. of 7 runs, 100 loops each)
# Number of parameters in original layers: 28,755,648
# Number of parameters in fused layer: 28,755,648

# we get seemingly contradictory results but why?
# Fusing several operations into a single operation
# does not always result in improved performance.
# the improvement is directly related to the
# implementations both software wise and more importantly hardware wise.
# you can see fused operations are heavily used in many codebases/frameworks,
# take forexample pytorch, it heavily uses this, especially when it comes to 
# quantization or computation graph optimization) but if thats the case why
# we are getting this outcome?let me elaborate a bit more.
# you see in general, fusing multiple operations into one larger one,
# improves performance by reducing memory access and intermediate memory writes
# and the overhead associated with those separate operations.
# but to what extent we dont know, and benchmarking or consulting
# the documentations both for the software(library we use)
# and/or lowlevel hardware details are usually the only way we can know for sure.
# for example in Pytorch and CUDA related optimizations its 
# often the case that multiple separate kernels/operations can
# be fused into one, which can reduce kernel launch overhead and 
# more importantly reduce the amount of data that needs to be moved around in memory.
# but as we said this depends on whether the underlying compiler/backend
# and layer supports it.
# anyway, meddling around with matrix multiplication of random sizes,
# like what we have been doing here, as we can see, may not always translate
# into what we intuitively expect simply because cuda kernels and libraries 
# are usually heavily optimized for certain matrix dimensions, memory layouts,
# data types and alignments.
# some sizes may fit particularly well with the underlying hardware,
# tiling strategies or things such as tensor cores.
# 
# so a matrix being larger does not necessarily mean it performs worse,
# and a smaller one does not necessarily mean it performs better either.
# performance can vary quite significantly depending on the exact sizes
# and whether they align well with what the hardware and library are optimized for.
# 
# haivng all this said, sometimes having several smaller operations 
# can perform much better than a single larger and more complex fused operation.
# a larger fused kernel may increase register pressure, affect cache behavior
# or prevent the use of highly optimized specialized kernels so fusion can
# also introduce its own problems.
# 
# Additionally, our code may behave quite differently depending on the hardware.
# something that performs very well on a server GPU may perform differently
# on a consumer level GPU due to differences in compute capabilities,
# memory bandwidth, cache sizes, available vram and the overall architecture.
# 
# to recap in many cases it can significantly improve performance,
# especially by reducing memory traffic and kernel launch overhead,
# but it depends heavily on the operation, tensor sizes,
# software implementation, compiler/backend and the hardware itself.
# at the end of the day, benchmarking the actual workload
# is usually the only way to know what performs better.
# Thereore It's always recommended to benchmark and compare the performance 
# of different approaches on our specific hardware and input size to determine
# the most efficient solution.
#

#%%
#%% skip
# side quest 2 : test with unscaled/unmasked/noposition attention
# lets create a few arguments to our attentionhead so we can easily 
# test different options and see how they fair against eachother! 
# and whether what we said stays correct!
# remember our attentionhead currently doesnt use any positional information!
# note that simply testing a single attentionhead does not show a 
# significant difference between any of the mentioned changes.
# in order to see the actual difference, we need to test them in a better testcase
# which involves several attentionsheads/and layers to get a realistic sense of the differences. 
# we will do this at the end inshaalah.
# class AttentionHeadNoPos(nn.Module):
#     def __init__(self, context_size, embd_size, head_size, use_bias=False, scale=True, masking=True) -> None:
#         super().__init__()
#         self.context_size = context_size
#         self.embd_size = embd_size
#         self.head_size = head_size
#         self.scale = scale
#         self.masking = masking
#         self.key = nn.Linear(embd_size, head_size, bias=use_bias)
#         self.query = nn.Linear(embd_size, head_size, bias=use_bias)
#         self.value = nn.Linear(embd_size, head_size, bias=use_bias)
#         self.register_buffer('tril', torch.tril(torch.ones(context_size, context_size)))

#     def forward(self, inputs:torch.Tensor) -> torch.Tensor:
#         B,T,C = inputs.shape 
#         k = self.key(inputs)  
#         q = self.query(inputs)
#         v = self.value(inputs)
#         # lets check the effects of scaling/unscaling in practice
#         if self.scale:
#             weight = q@k.transpose(-2,-1)* self.head_size**-0.5
#         else:
#             weight = q@k.transpose(-2,-1)
#         # lets check the masking effects as well
#         if self.masking:
#             weight = weight.masked_fill(self.tril[:T,:T]==0,float('-inf'))
#         weight = weight.softmax(dim=-1)
#         bow = weight@v
#         return bow
    
# class BigramWithAttentionNoPos(nn.Module):
#     def __init__(self, vocab_size, context_size, embd_size, head_size, scale=True, masking=True, use_bias=False) -> None:
#         super().__init__()
#         self.vocab_size = vocab_size
#         self.embd_size = embd_size
#         self.context_size = context_size
#         self.embeddings = torch.nn.Embedding(vocab_size, embd_size)
#         # add a self-attention head
#         self.attnhead = AttentionHeadNoPos(context_size, 
#                                        embd_size, 
#                                        head_size, 
#                                        scale=scale, 
#                                        masking=masking, 
#                                        use_bias=use_bias)
#         self.fc = torch.nn.Linear(head_size, vocab_size)
        
#     def forward(self, inputs:torch.Tensor, labels=None) -> torch.Tensor:
#         token_embeddings = self.embeddings(inputs) 
#         out_attention = self.attnhead(token_embeddings)
#         logits = self.fc(out_attention)
#         loss = None
#         if labels is not None:
#             loss = F.cross_entropy(logits.permute(0,2,1), labels)
#         return logits, loss
    
#     def generate(self, idxs, max_token_count)-> list[torch.Tensor]:
#         for _ in range(max_token_count):
#             assert idxs.ndim >1, f"idx.shape '({tuple(idxs.shape)})' is invalid({idxs.ndim}). it must have the form (B,T)"
#             idxs_cropped = idxs[:, -self.context_size:]
#             logits,_ = self(idxs_cropped)
#             logits = logits[:,-1,:]
#             probs = logits.softmax(dim=-1)
#             new_idx = torch.multinomial(probs, num_samples=1, replacement=True)
#             idxs = torch.cat((idxs,new_idx), dim=-1)
#         return idxs
    
# # now lets test this and see if it works 
# def train(scaling, masking, bias, lr, batch_size, vocab_size, max_iter, device):
    
#     model = BigramWithAttentionNoPos(vocab_size=vocab_size,
#                                      context_size=32,
#                                      embd_size=128,
#                                      head_size=128,
#                                      scale=scaling,
#                                      masking=masking,
#                                      use_bias=bias)
    
#     param_count = sum([p.nelement() for p in model.parameters()])
#     print(f'param count:     {param_count:,}')
#     print(f'scaling:         {model.attnhead.scale}')
#     print(f'masking:         {model.attnhead.masking}')
#     print(f'positional info: -NO-')
#     print(f'head size:       {model.attnhead.head_size}')
#     print(f'embd size:       {model.embd_size}')
#     print(f'context size:    {model.context_size}')

#     with torch.no_grad():
#         x,y = get_batch('train',4)
#         _, loss = model(x,y)
#         print(f'loss before training: {loss:.4f}')
    
#     optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
#     model = model.to(device)
#     # set model to train mode explicitly
#     model.train()
#     for i in range(max_iter):
#         # get the batch
#         x,y = get_batch('train', batch_size=batch_size)
#         x,y = tuple(t.to(device) for t in (x,y))
#         # feed the model and get the logits
#         logits, loss = model(x,y)
#         # evaluate the model
#         if i%1000==0:
#             losses=evaluate_loss(100, device)
#             print(f'train: {losses["train"]:.4f}  val: {losses["val"]:.4f}')
#         # zero-out grads
#         model.zero_grad(True)
               
#         loss.backward() 
#         optimizer.step()
#     print(f'done!')

#     # now lets try its output
#     input = torch.zeros(size=(1,1)).int()
#     output = model.generate(input, 500).squeeze(0).tolist()
#     print(f"{''.join(decode(output))}")
#     print(f'-'*25)

# # and now we can train this : 
# device='cpu'
# batch_size = 32
# vocab_size = len(vocab_list)
# max_iter = 25000
# # attention requires much lower lr compared to plain bigram model
# lr = 0.001
# # attention with scaling - default
# train(scaling=True, masking=True, bias=False, lr=lr, batch_size=batch_size, vocab_size=vocab_size, max_iter=max_iter, device=device)
# # attention without scaling 
# train(scaling=False, masking=True, bias=False, lr=lr, batch_size=batch_size, vocab_size=vocab_size, max_iter=max_iter,device=device)
# # attention without masking 
# train(scaling=True, masking=False, bias=False, lr=lr, batch_size=batch_size, vocab_size=vocab_size, max_iter=max_iter, device=device)
# # attention without scaling and without masking
# train(scaling=False, masking=False, bias=False, lr=lr, batch_size=batch_size, vocab_size=vocab_size, max_iter=max_iter, device=device)
# # attention model with bias enabled
# train(scaling=True, masking=True, bias=True, lr=lr, batch_size=batch_size, vocab_size=vocab_size, max_iter=max_iter, device=device)
#! check if its ok to include them now or at the very end. because the changes may not be evident here!
#

# %%
#
# Earlier we mentioned that our implementation of attention so far, is refered to as self-attention,
# becasue the key, query and value vectors are applied on the same input (they use the same source!),
# hence the name, self attention.
# Moreover, we also saw that for our specific case, tokens/nodes can not communicate with the future
# nodes becasue we are developing an autoregressive language model, which by definition requires us
# to make predictions solely based on what has come thus far. 
# However, in general this constraint can be removed when it's advantageous for all tokens to interact,
# such as in usecases like sentiment analysis and machine translation among other examples. 
# Take sentiment analysis for example, inwhich we 'want', all the tokens, to able to communicate with
# eachother for an accurate analysis. we dont care if a previous token looks at a fture one or not,
# infact we welcome all interactions between tokens so that it maximizes the chances of revealing
# as much information as possible to ultimately reach to the right conclusion which determins the
# overal sentiment.
# In these cases, we do this by employing an encoder block, which is similar to our current setup 
# but without the constraint section.
#
# What we have implemented so far (with the tril, masking) is refered to as a decoder block,
# in which we are trying to generate a new token given the previous ones.
#
# side note:----------------------------------------------------------------------------------
# (loosly speaking, decoder is synomous with generation, just as the encoder is with encoding!
# you can imagine the decoder as a block that generates/produces something, as apposed to the
# encoder part/block which its job is to create a highlevel represenation of the input, 
# (usually refered to as "latent space"/"encoding"), to be consumed by others (like decoders!).
# this process is known as encoding and this is where the encoder gets its name.
# the encoder is there to capture important features and patterns in the input.
# on the other hand, the decoder takes these highlevel representations and generates something 
# meaningful from it, such as text, image, etc. 
# This process is known as decoding and hence the name decoder.
#

# cross-attention:
# The attention mechanism as we briefly pointed out, is quite versatile and can be 
# utilized in various ways. One such example/way is the creation of something called cross-attention.
# cross-attention is basically related to the scenarios in which we have encoder/decoder blocks in our
# model,in which the queries originate from decoder's input while the key and value are derived from
# an external source, sometimes from the encoder block itself. 
# Cross-attention comes into play when we have a separate source of information we would like to 
# extract from and utilize. This is commonly seen in sequence-to-sequence models, such as machine translation.
# In such models, we have two sources of information: the source text and the generated text (translation output). 
# The goal is to maximize the translation accuracy by attending to the source material and enhancing
# its relationship with the output as much as possible(getting them as close as possible). 
# The key and value are applied to the encoder/source material, while the query is applied to the 
# decoder's input. this process helps in creating a more accurate and contextually relevant translation.
# infact the original paper's uscase was machine translation! and it incorporates an encoder and a decoder
# just like we described. 
# 
# The Attention being a communication mechanism between tokens, is not position aware for the most part,
# that is, by default there is nothing in this mechanism that provides or enforces a notion of 
# space/position for tokens involved.
# by default these tokens/nodes/points whatever we call them, dont have any idea about where they are
# or how they are positioned (with resepect to eachother). 
# to have a better mental picture, imagine each token as a node in a directed graph, each node has a
# connection to some other nodes, (for example each node has a connection to itself, and another 
# connection to other nodes,etc) as you can see there is no notion of space or better said order 
# between nodes. which one would we call is number 1 or 2, in a graph, in which the structure simply
# doesnt define any order by itself, unless we define one using a mechanism of some sort) 
# The attention simply acts on a set of vectors in this graph, so if the order or position of these 
# nodes is of any significance to us, then we need to encode them as well, 
#
# as it happens in our case, the positional information is important to us, becasue we want to generate
# text, and order matters here(text are inherntly sequential, and also we need to prevent leftward 
# information flow in the decoder to preserve the auto-regressive property). 
# models such as RNNs or CNNs, are inherently position aware (RNNs process the input sequentially,
# and CNNs have spatial information, as they operate by sliding a fixed window over the input)
# but this is not the case for attention. as we just stated, there is no notion of position, its 
# just a set of individual vectors being operated on) 
#
# As the original authors put it: 
# "Since our model contains no recurrence and no convolution, in order for the model to make use of
# the order of the sequence, we must inject some information about the relative or absolute position
# of the tokens in the sequence."
#
# note that the(statistics from) connection between nodes, does give us a sense of structure, but it
# may not be enough to infer the underlying semantic, imagine a case, where a word for example has 
# several meaning, and may very well have high relevancy to a few tokens at the same time, but without
# additional positional information, an ambiguous semantic can be infered between the tokens involved,
# however when positional information is also present, such ambiguity can be avoided) 
# (e.g. live to work! vs work to live! completely different meaninig based on the order of words)
#
# (See section 3.2.3 Applications of Attention in our Model in the paper)
# 
# we can do better with attention.
# 

#%%
# Positional encoding:
# As we pointed out earlier, the attention mechanism does not have any notion of position, unlike the
# cnn or rnns that are inheritly sequential and are posision aware. 
# without positional information, a sequence such as ABC would be the same as CBA, ACB, BCA, or any 
# permutations of the tokens involved. this is specifically bad for an autoregressive task like generating text. 
# in an autoregressive model, given a token, the model needs to comeup with a good prediction! 
# obviously this requires the knwoledge about the order of the tokens to be taken into account to 
# create meaningful sequences/texts. 
# imagine you expecing the model to greet you like: Hello, nice to meet you! but instead you get a 
# random sequence of tokens like: ',to nice! Hello you meet' which is not what we are after! 
# in reality/practice the network might comeup with a weak notion of order, simply based on the statistics 
# it saw in the dataset, however, it cant utilize that to comeup with complex structures certainly
# not the ones that have subtle meaning changes, when the order of some parts are reversed! like :
# live to work!
# work to live!
# as you can see the order plays a crucial role not only in generating text but also infering the
# undderlying meaning/sematic.
# 
# So how can we add this positional information. There are several ways we can think of to add this
# information. 
# we can learn these so called positional encoding the same way we learn the token embeddings. 
# this is called abosulte positional encoding! it has some pros and cons we dont get to here (I explained them at the end). 
# We can simply pre-compute these positional encodings. this is what the original paper of "attention is all you need", did.
# their choice is refered to as sinusoidal positional encoding which provides relative positioning 
# as well. 
# Basically the combination of sin/cos acts as a kind of unique identifier, some even thought of it
# as a counter (imagine how binary system works, something to that effect), and this way the positional
# info for each token is calculated and provided to the network. (in practice, sin/cos paris is used 
# to encode different unique values for each position, form 0 up to infinity, they have some nice 
# features/attributes that are crucial for our job and they are explained at the end of this tutorial)
# 
# the authors in their previous work(Order Matters: Sequence to sequence for sets 2015), pointed out
# that counting was was a hard problem, and 2 years later they came up with the sinusoidal positioning encoding. 
# see 
# https://www.reddit.com/r/learnmachinelearning/comments/9e4j4q/positional_encoding_in_transformer_model/
# http://fastml.com/introduction-to-pointer-networks/
# https://www.reddit.com/r/MachineLearning/comments/cttefo/d_positional_encoding_in_transformer/
# https://arxiv.org/abs/1905.04226 )
# 
# hence why the chose sinusoidal positional encoding, to act like a counter, giving each token a identifiable indetifier!
# later works however, showed that such embeddings can be learned and work just as well, infact BERT 
# did exactly that and opted to use learned positional emebddings instead of the sinusoidal positional
# encoding used by the original paper.
# (the original authors also tested with learned embeddings and reported the near identical results,
# but oppted out to use sinusoidal positional encoding for its ability to model also relative distance)
# later on, other versions such as rope or rotary positional encoding were introduced to provide relative
# positional encoding (better). after that another work AliBi, completely removed the positional embedding
# and instead used constraints on relationships with respect to their distance! (i.e. decaying the strength
# of relationship based on distance) so this is an active field of research and today as I write this, 
# rope seems to be being used along with learned positional encoding. So its still a field of active 
# research. we can use any of these, rope is more populare followed by learned positions though.
# 
# side note: --------------------------------------------------------------------------------------
# the authors also tested with learned embedding and got nearly identical results but ultimately 
# chose sinusoidal embedding becasue they thought: 
# "...because we hypothesized it would allow the model to easily learn to attend by relative positions, since for any
# fixed offset k, P_Epos+k can be represented as a linear function of P_Epos.
# and
# "...it may allow the model to extrapolate to sequence lengths longer than the ones encountered during training."
# 
# theres alot to talk about positional encoding, we are just scratching the surface! as we go on
# we introduce more details!
#--------------------------------------------------------------------------------------------------

# %%
#Ok positional encoding is good and we need to implement it. 
# we can also add some flexibility to our attention module to allow us to test different configurations
# such as with positional encoding, without positional encoding, stuff like that. 
# But this is not the time now, as they may not show much difference using a single head but when
# later on we add more heads, that would be a better time as it can show us how they really affect
# the performance. 
#
# For now lets keep it simple. 
# lets build our language model with the new attention head. 
# since we are using the the attention head, we need more arguments
class BigramModelWithAttention(nn.Module):
    def __init__(self, vocab_size, context_size, embd_size, head_size, device, use_bias_att=False) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embd_size = embd_size
        self.context_size = context_size
        # for gpu/cpu acceleration during training
        self.device = device
        # unlike the previous model, lets decouple the final logits from 
        # the number of embeddings, because we are using attentions, and
        # we want to have multiple operations inbetween obviously.
        self.embeddings = torch.nn.Embedding(vocab_size, embd_size)
        # lets now add the positional embedding as well 
        # using this, we try to retain the position embedding for our tokens
        # up to the current token in context_size
        self.position_embd = torch.nn.Embedding(context_size, embd_size)
        # add a self-attention head
        self.head = AttentionHead(context_size, embd_size, head_size, use_bias=use_bias_att)
        
        # in order to get the final logits, we need a linea layer at end
        # since we now have attention before this layer, the output dim of attention which is
        # head_size will be used here
        self.fc = torch.nn.Linear(head_size, vocab_size)
        
    def __call__(self, inputs:torch.Tensor, labels=None) -> torch.Tensor:
        # lets grab the shapes, since we will be using them 
        B,T = inputs.shape
        token_embeddings = self.embeddings(inputs) # has the shape (B,T,E) e is embd_size
        # since we have a position_embedding lets use that as well
        # and notice that we didnt use the inputs, but rather torch.arange(T)
        # this means, for each input, as we process it, we also get embeddings up to
        # the current token count as well, if the inputs has 3 tokens currently, we
        # will creeate position embeddings for 0,1 and 2, and for the next input
        # this continues likewise. this results in (T,E)
        #
        position_embeddings = self.position_embd(torch.arange(T,device=self.device))
        # print(f'pos_embd:{position_embeddings.shape}')
        # and lets add the two embeddings together
        # this effectively gives us, not only the token embeddings(identity)
        # but also its position in the sequence. note that this doesnt really
        # help in a bigram model, but when it comes to attention it really does!
        embeddings = token_embeddings + position_embeddings
        # now lets feed this to attention head
        out_attention = self.head(embeddings)
        # lets feed this embedding to our fc at the end instead
        logits = self.fc(out_attention)         # has the shape (B,T,C) c is vocabsize
        loss = None
        if labels is not None:
            # recall that crossentropy likes its input to be B,C,T and we are B,T,C
            # so lets permute and make it happy!
            loss = F.cross_entropy(logits.permute(0,2,1), labels)
        return logits, loss
    
    def generate(self, idxs, max_token_count)-> list[torch.Tensor]:
        # lets generate an output as long as num_max_token
        for i in range(max_token_count):
            # print(f'{i}/{max_token_count}) idxs: {tuple(idxs.shape)}')
            # make sure idx is 2d
            assert idxs.ndim >1, f"idx.shape '({tuple(idxs.shape)})' is invalid({idxs.ndim}). it must have the form (B,T)"
            # now lets feed it to the model and sample from the probablities it produces
            # but since, we now have postional embeddings as well, we can no longer have 
            # more than block_size/contex_size in, becasue if our idx is more than context_size
            # our positional embedding will go out of scope and error out
            # so here we are basically getting as many as context_size
            # note that we dont destroy the idxs! each time we get the last context_size tokens
            # from it and feed it to the model to generate the next token, and keep going
            # if we replace the idxs by sth like idx=idx[:,-self.context_size:], we would
            # only create a sequence of only self.context_size, no matter how many iterations
            # we do, we just repeat the same sequence again and again!
            idxs_cropped = idxs[:, -self.context_size:]
            logits,_ = self(idxs_cropped)
            # since we are bigram still, lets only get the last token as the next token predicted!
            logits = logits[:,-1,:]
            # convert to probs 
            probs = logits.softmax(dim=-1)
            # now lets sample from it 
            new_idx = torch.multinomial(probs, num_samples=1, replacement=True)
            # now concatenate the new token to the previous one and feed it back to the model
            # for the next round of prediction
            # also remember that we are creating a sequence, so we concat them at dim=1 to get 
            # a longer sequence (we are gradually increasing the sequence length from 1 up to
            # max_token_count)
            idxs = torch.cat((idxs,new_idx), dim=-1)
           
        return idxs
    
# now lets test this and see if it works 
x,y = get_batch('train',4)
model = BigramModelWithAttention(vocab_size, 
                                 context_size, 
                                 embd_size=16,
                                 head_size=16,
                                 device='cpu',
                                 use_bias_att=False)
# model.cuda()
# x,y = (t.cuda() for t in zip(x,y))
logits,loss = model(x,y)
print(f'{logits.shape=} {loss=:.4f}')
# and now we can train this : 
device='cpu'
batch_size = 32
head_size = 16 # 100 of course using bigger sizes result in better numbers, but lets start small and then use larger values 
embd_size = 16 #100
context_size = 8 #100
vocab_size = len(vocab_list)
max_iter = 5000
# only to test the effect of bias in k,q,v calculations
use_bias_attn=False
# attention requires much lower lr compared to plain bigram model
lr = 1e-3
model = BigramModelWithAttention(vocab_size=vocab_size,
                                 context_size=context_size,
                                 embd_size=embd_size,
                                 head_size=head_size,
                                 device=device,
                                 use_bias_att=use_bias_attn)

param_count = sum([p.nelement() for p in model.parameters()])
print(f'param count:  {param_count:,}')
print(f'head size:    {head_size}')
print(f'embd size:    {embd_size}')
print(f'context size: {context_size}')

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
model = model.to(device)
# set model to train mode explicitly
model.train()
for pos in range(max_iter):
    # get the batch
    x,y = get_batch('train', batch_size=batch_size)
    # feed the model and get the logits
    logits, loss = model(x,y)
    # evaluate the model
    if pos%1000==0:
        losses=evaluate_loss(100, device)
        print(f'train: {losses["train"]:.4f}  val: {losses["val"]:.4f}')
    # zeroout_grads
    model.zero_grad(True)
    # we ccould also do 
    # optimizers.zero_grad() 
    # but since our optimizer 'only' uses our models parameters, they are basically the same
    # note we talk about this in more details later which one to use and where later on inshaalah.
    
    loss.backward() 
    optimizer.step()
print(f'done!')

# now lets try its output
input = torch.zeros(size=(1,1)).int()
output = model.generate(input, 500).squeeze(0).tolist()
print(f"{''.join(decode(output))}")
# edit, I moved the plots code to the end of the tutorial when I explained positional encoding in depth
# I plan on usingthe plots on the final version when everything makes more sense and is in one place. 
# lets plot them, they maynot look very intresting now, but we try them at the end once more time
# # and see what we can get out of these visualizations and plots
# plot_heatmap(model.position_embd.weight.detach(),'position_embeddings')
# plot_positional_encoding_distances(model.position_embd.weight.detach().numpy())
# plot_positional_encoding_heatmap(model.position_embd.weight.detach().numpy())
# plot_positional_encoding_dot_product_heatmap(model.position_embd.weight.detach().numpy())
# plot_positional_encoding_distance_total_3d(model.position_embd.weight.detach().numpy())
#%%
# prints
# here are a few tries:
# param count:  3,041
# head size:    16
# embd size:    16
# context size: 8
# train: 4.2551  val: 4.2533
# train: 2.7175  val: 2.7354
# train: 2.5917  val: 2.5681
# train: 2.5299  val: 2.5265
# train: 2.4684  val: 2.4759
# done!

# Wonedimy
# Y:
# ARUS:
# LAnouver; pof th, sthe the mae,
# Wor ut barrioreche savarduncou?

# hiler bathakm,
# I O:
# AK:
# Acat ED:
# et alliro dow wowilou ttherss; whest owur, garsacwe Ie hithaceyidous sou f'ze iwot ry tohien fm.
# h.

# Ar'm-ore tr Fofhou
# Mr ojous omerrastheasncy yous Wowuce whor tu I I:
# Hbeotth spat fel woro bel, aneicudidy ktiferrd thout ngord. th Iid aral, be'nd.

# Pal tels Ms eimu Me myo on I teasthe del,
# AChinout,
# Wowure. pu tcher muss-renon; wingh ocet im
# r Ddr,
# Do ne thu,
# LI ir't
# Operst ry wu

# param count:  5,745
# head size:    16
# embd size:    32
# context size: 32
# train: 4.1374  val: 4.1414
# train: 2.6264  val: 2.6210
# train: 2.5100  val: 2.5092
# train: 2.4434  val: 2.4578
# train: 2.4140  val: 2.4252
# done!

# Thee to an tal my,
# Thake ced arce chart bre le be bizoresf thak des garg
# Lur Wentescaln
# Hooco beins pord sth ds hom span gr; ther. My Why ort thallils ofrof not tos gas hyine ind hand
# Thid One the shond nd,
# AFenosee omn st hy no fe.
# GEwank, iand!
# The sbe qurss yod hem snel.
# CLerle,
# Whow Vou bem!
# Theewourtoave, shougen t:
# Ber hangn toutince mor ed,
# AD Scheapy thatho ABEN:
# AD:
# SUEDithy nde ndel mmy Tongusate hfolit, gintoher, hat gwe tiet what whingimst hpoourde wowopal,
# The by br des qO: I no os,

# try2: decreasing context size
# param count:  4,977
# head size:    16
# embd size:    32
# context size: 8
# train: 4.2156  val: 4.2269
# train: 2.5848  val: 2.6063
# train: 2.4926  val: 2.4850
# train: 2.4587  val: 2.4601
# train: 2.4199  val: 2.4242
# done!

#try 3: decreasing embd_size, but keeping context_size 32
# param count:  2,265
# head size:    16
# embd size:    8
# context size: 32
# train: 4.2202  val: 4.2158
# train: 2.9840  val: 3.0093
# train: 2.7667  val: 2.7621
# train: 2.6355  val: 2.6411
# train: 2.6070  val: 2.6077
# done!

# o
# TI:
# BOGot be
# DOus ssh.
# Bea hashren v:
# ANULINII feso sid izeis by tofuso te I be fnowe?
# Wledu ha m, thope w,
# ARUTI land med cu
# SI bve fme
# ARIThilaserothan omyhee
# Oina c;
# UCAngose.
# Terve t thirre my tthunulpthemaf wof.

#  corlur yoNE:
# NHy w te hen cod, tonos vengo veschete tors t ithaimeqlutreqRoosoye chin y h thuasafaous.
# ANCathelo-
# The?
# Shen br uyh sbe ans I:
# Gouthyykey m Authirou ae tovas ber ut lnt:,

# TAMENI sf rousinthsin h'howenthor sg.
# hidandhe waresd!

# I f
# Se loner mrdin gsvette n thest?

# try 4: decreasing head_size, keeping embd_size and context_size the same
# param count:  4,457
# head size:    8
# embd size:    32
# context size: 32
# train: 4.3416  val: 4.3409
# train: 2.7354  val: 2.7333
# train: 2.6400  val: 2.6309
# train: 2.5716  val: 2.5692
# train: 2.5217  val: 2.5182
# done!

# IF
# S:
# To ofldickeens we fofane for o warus
# Herdliion bis, te,
# The ta, d ounor;
# Ad h ank; sor Cito-'d turses tierlllle,
# Whe pen, tint yt Ging sino leles fes hesr sst.
# Than hinean.


# Nofl an udn f hak,
# Wh stharovernes;
# And meno scerr tth Gbl y hy I t stit pand sy, pefif.
# US dik thith be tagag, harlovenppsh!
#  cegshil,
# Wheanst thiges eprine sy wabat tha.

# Wheas
# Wit:
# I
# Thouw'arsangort,
# Giseal?

# Wh; mo tands t's.

# NRNULECADSAxave!

# ANNDIEI Fh Fod meveeas pee be,
# Bes arner,

# Woo t'le.


# VIPORLEOG
# CAOO:

# try 5:
# param count:  4,977
# head size:    16
# embd size:    32
# context size: 8
# train: 4.2156  val: 4.2269
# train: 2.5848  val: 2.6063
# train: 2.4926  val: 2.4850
# train: 2.4587  val: 2.4601
# train: 2.4199  val: 2.4242
# done!

# DANGUTTA:
# Bert hatire on the wo A? sewas ker haywins wet.

# mousill akerd fous sther Le fke sth,, bepeere gn ssst I oun amand serewos, cobear song.


# AMEfediengg yit othesshint.

# Anc:
# A must shan ghety ak is, shet.

# MEO:
# Tom yot stths!
# NCIBus hand akt imy berdr,
# Thulll id mere thassut,
# Moureleray, fous mante. QBO decencepous ariveed hart;
# I; bifo wthen chant tht tho hy youn,
# Grolomy, ucel.

# BRIINE ERIYht I arum arw t; odeat ngher bly wehe foru thas sot our:
# Bn; sshiutithemowith her fourpler thant

#
# which looks depressing not gonna lie! but its better than the simple bigram model we 
# built earlier, anyway, can we improve it? certainly we can! 
# for example we could use dropout on our attention! we could use normalization layers
# and we could use multiple heads instead of just one! beside crancking up the numbers obviously!
# This takes us to the next subject which is multi-head attention! it is really nothing except 
# several normal! attention blocks run in parallell! 
# so lets implement this first and see how much we can improve upon this
#%%
class MultiHeadAttention(nn.Module):
    def __init__(self, num_head, head_size, embd_size, context_size, bias_attn=False) -> None:
        super().__init__()
        self.num_head = num_head
        self.head_size = head_size
        # we assume the head_size is already split between the num_heads and thus we dont split
        # it again
        # assert head_size//num_head == 0, f'head_size({head_size}) must be divisable by head_num({num_head})'
        # 
        # we need to create n heads so lets do it 
        # self.heads = [AttentionHead(context_size, 
        #                             embd_size, 
        #                             head_size, 
        #                             bias_attn) for _ in range(num_head)]
        # but we can also use pytorch's nn.ModuleList which is a better equivalent than python list
        # becasue all the modules it contains are properly registered, and will be visible by all 
        # Module methods which is not the case for python lists (i.e. .parameters() .children(), 
        # .zero_grad, etc, e.g.) and therefore its best to use modulelist instead of pure python lists.
        # 
        # note that, nn.Sequential cant be used, becasue it runs the modules in succesion
        # i.e. serially, one after the other (feeds the output of the previous module to 
        # the next module, etc) which is not what we want. we want to calculate each head
        # independetly and aggregate their outputs so, either a python list or torch moudlelist
        # can be used
        # sidenote: this module that we are building, is also known as, a masked multi-attention-head
        # becasue we are using the single-head self-attention which uses a constrain we imposed
        # by masking if you recall that!
        self.heads = torch.nn.ModuleList(AttentionHead(context_size, 
                                         embd_size, 
                                         head_size,
                                         bias_attn) for _ in range(num_head))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # note that head_size is usually the embd_size, so it covers the whole 
        # embeddings obviously, and note that each head will work on a portion
        # of the given head_size, if e.g. we have head_size or embd_size = 16
        # if we had 1 head, the head_size would be 16. however if we had 2 heads
        # we had to split the head_size in half for each head, and later on
        # we had to concat them to get the full-size head_size or embd_size.
        # therefore here, we need to concat their results along the cols
        # so multi-head-attention is akin to group convolution, and thus the head_size
        # and num_head must align properly.
        outputs = torch.cat([head(inputs) for head in self.heads], dim=-1)
        return outputs
# lets test 
# note that we set the split value for head_size based on our num_head
m = MultiHeadAttention(num_head=4, head_size=16//4, embd_size=16, context_size=8, bias_attn=False)
logits = m(torch.randn(size=(1,8,16)))
print(f'{logits.shape=}')
# 
# now lets use this in our model
class BigramModelWithAttention(nn.Module):
    def __init__(self, vocab_size, context_size, embd_size, num_head, head_size, device='cpu', bias_attn=False) -> None:
        super().__init__()
        self.vocab_size=  vocab_size
        self.context_size = context_size
        self.embd_size = embd_size
        self.num_head = num_head
        self.head_size = head_size
        self.device = device
        # token/character embeddings
        self.token_embeddings = nn.Embedding(vocab_size, embd_size)
        # position embeddings
        self.position_embeddings = nn.Embedding(context_size, embd_size)
        # multihead attention instead of a single head attention block
        # note that the head_size is going to be divided between the 
        # heads, so ultimately we have the same number for head_size
        # globally (basically each head gets its share of head_size
        # which is head_size//num_head, but at the end since we have
        # num_head heads, their output makes us the whole head_size) 
        # 
        self.multi_head_attention = MultiHeadAttention(num_head, head_size//num_head, embd_size, context_size,bias_attn)
        # finally the output fc layer 
        self.fc = nn.Linear(head_size, vocab_size)
        
    def forward(self, inputs:torch.Tensor, labels:torch.Tensor=None)->torch.Tensor:
        B,T = inputs.shape
        # print(f'{inputs.shape=} {self.context_size=} {self.embd_size=}')
        token_embds = self.token_embeddings(inputs)
        position_embds = self.position_embeddings(torch.arange(T,device=self.device))
        embds_combilned = token_embds + position_embds
        # now lets feed them to our multihead attention block 
        out = self.multi_head_attention(embds_combilned)
        # and finally the logits 
        logits = self.fc(out)
        loss = None
        if labels is not None:
            # remember the cross entropy wanted its input in B,C,T while ours is in (B,T,C)
            loss = F.cross_entropy(logits.permute(0,2,1), labels)
        return logits, loss 

    def generate(self, idxs, max_token_count):
        assert idxs.ndim>1 , f'idxs.ndim({idxs.ndim}) must be 2 (in the form of (B,T))'
        for i in range (max_token_count):
            idxs_cropped = idxs[:, -self.context_size:]
            logits,_ = self(idxs_cropped)
            logits = logits[:,-1,:] 
            # calulate the probs
            probs = logits.softmax(dim=-1)
            # sample the next character/token 
            idx_token_next = torch.multinomial(probs, num_samples=1, replacement=True)
            # add this to our existing tokens in idxs 
            idxs = torch.cat((idxs, idx_token_next), dim=-1)
            
        return idxs 
# now lets train this model and see how it performs this time
torch.manual_seed(255)
random.seed(255)

head_num = 4
head_size = 32
embd_size = 32
context_size = 8 
vocab_size = len(vocab_list)
device = 'cpu'
use_bias_attn = False

lr = 0.001
batch_size = 32
max_iter = 5000
eval_period = 1000
model = BigramModelWithAttention(vocab_size=vocab_size, 
                                 context_size=context_size, 
                                 embd_size=embd_size,
                                 num_head=head_num, 
                                 head_size=head_size, 
                                 device=device,
                                 bias_attn=use_bias_attn)
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr)
param_count = sum(p.nelement() for p in model.parameters())

print(f'param_count  =  {param_count:,}')
print(f'head_num     =  {head_num}')
print(f'head_size    =  {head_size}')
print(f'embd_size    =  {embd_size}')
print(f'context_size =  {context_size}')
print(f'device       =  {device}')
print(f'use_bias_attn=  {use_bias_attn}')

for pos in range(max_iter):
    # read a batch 
    x, y = get_batch('train', batch_size=batch_size)
    logits, loss = model(x,y)
    
    # calculate the smoother loss on multiple batches on train/val splits
    if pos%eval_period == 0:
        losses = evaluate_loss(200, device)
        print(f"train: {losses['train']:.4f}  val: {losses['val']:.4}")
    # zero-out gradients 
    model.zero_grad()
    # we ccould also do 
    # optimizers.zero_grad() 
    # but since our optimizer 'only' uses our models parameters, they are basically the same
    # note we talk about this in more details later which one to use and where, later on inshaalah.
    
    # do a backward pass 
    loss.backward()
    # do a single optimization step 
    optimizer.step()

print(f'done!')
# lets see how this model fairs now and what it generates 
initial_token = torch.zeros(size=(1,1)).int()
output = model.generate(initial_token, max_token_count=500).squeeze().tolist()
print(f"{''.join(decode(output))}")
# outputs : 
# param_count  =  3,041
# head_num     =  4
# head_size    =  16
# embd_size    =  16
# context_size =  8
# device       =  cpu
# use_bias_attn=  False
# train: 4.2041  val: 4.204
# train: 2.6895  val: 2.685
# train: 2.5449  val: 2.528
# train: 2.4570  val: 2.477
# train: 2.4238  val: 2.433
# done!

# I'd In mame
# Ast owu hapland.

# Mayl thew youres?
# IRisire dis lhend gitim whorder, sphorgunct Beye skek boutrig!

# I, sho sciltr not bpreathl harr'd doess elowt cof or iny poovigre.
# Thalrit?

# HThary?

# Lerim he bamat:
# Thiet hey, krepably bose bour cave grosr benemece wleir hon'g.

# MADIUMI but th Mond im's veoo tit. IDf met Cerve's wue rrer hiftren'tr'd:
# Anoks:
# HOt
# We I phess chi bouine mares yon ther; tlotirtlinl therot:
# Io
# Wh per,
# Fous thayigso ceany pate.

# Annd wosh fong uat
# Now ICUSo CCgodertiove
#
# trying with larger embedding size seems to improve the results: 
# param_count  =  7,553
# head_num     =  4
# head_size    =  32
# embd_size    =  32
# context_size =  8
# device       =  cpu
# use_bias_attn=  False
# train: 4.1828  val: 4.186
# train: 2.4703  val: 2.472
# train: 2.3649  val: 2.367
# train: 2.3009  val: 2.32
# train: 2.2713  val: 2.288
# done!

# Cly Toste?
# IOK:
# Whan his me' tis it I ond ber'thse?

# PO:
# Theray delling, dothoinerk an to the, or
# E VINGON: rocO:
# Thes win now thatin knir:
# Wigh fy, bund werar is wet my;
# I ene:
# JUu'd sow prut hat amver'd ENDUE VUF E VO?

# Se gings nom and.
# CArwak mandesplay beesire brit mand.

# MED:
# Thou ased ith is whis wentemprt hits this in ther.
# Whis are mald thanle wive itis the menjuth weat dert, nen you vimt siomB ighichtr
# LUT: am le the wids my thousecarto my haty wind WLIERME:
# Wher a and
# Whesh to wyt wir

#--------------
# compared to single head attention, with the same hyper parameters, train: 2.4684  val: 2.4759
# and now we got train: 2.4238  val: 2.433 which is better(we also got train: 2.2713  val: 2.288 
# with just increasing the embedding_size, so playing with parameters even blindly making model bigger
# seems to give us a boost), so we had improvements, but we still need a long way ahead of us!
# 
# to improve upon our results, there are couple of more things we need to add to our attention 
# block. if you look at the paper, you'll see a few concepts that we havent talked about or implemented
# yet, including the feed-forward(position-wise feedforward network) and layernorm, so lets talk about 
# them.
# feed-forward network or as its called in the paper,'position-wise feedforward network', is simply
# a "fully connected network which is applied to each position separately and identically", this 
# consists of two linear layer with a relu activation function inbetween. 
# so its a linear layer with a relu activation function followed by another linear layer basically. 
# you may also hear this network be refered to as 'computation after communication' so to speak!
# signifying the fact that it runs a computation after the attention module.
# the idea behind this extra operation is simple, to provide a higher representation out of attention work
# this network, causes every single token to have a nonlinear transformation and achieve a higher abstraction
# possibly yielding new information benficial to the task. 
# to put it in casual way, it can be seen as though the attention gatheres some stats/information,
# and this step is akin to looking into it and thinking about it comming up with some new findings,
# that is , if attention part is refered to as communication part, this is the computation part/ or
# thinking part for the lack of a better word.
# 
# lets implement this network in our bigram model and see if this seemingly simple change, affects us at all
class BigramModelWithAttention(nn.Module):
    def __init__(self, vocab_size, context_size, embd_size, num_head, head_size, device='cpu', bias_attn=False) -> None:
        super().__init__()
        self.vocab_size=  vocab_size
        self.context_size = context_size
        self.embd_size = embd_size
        self.num_head = num_head
        self.head_size = head_size
        self.device = device
        # token/character embeddings
        self.token_embeddings = nn.Embedding(vocab_size, embd_size)
        # position embeddings
        self.position_embeddings = nn.Embedding(context_size, embd_size)
        # multihead attention instead of a single head attention block
        # note that the head_size is going to be divided between the 
        # heads, so ultimately we have the same number for head_size
        # globally (basically each head gets its share of head_size
        # which is head_size//num_head, but at the end since we have
        # num_head heads, their output makes us the whole head_size) 
        self.multi_head_attention = MultiHeadAttention(num_head, head_size//num_head, embd_size, context_size,bias_attn)
        # now let us create the feedforward network, which basically is a linear layer with relu
        # followed by another linear layer! for this so called network, the in_features and out_features
        # are simply the same, and is embd_size, because its a sandwich layer between our attention output
        # and the last layer. 
        # this layer usually is (embedsize,embed_size) if you recall head_size is usally
        # equal to embed_size. also note that in the paper the feedforward network is two linear layers
        # with a relu (one linear layer with a relu plus another linear layer).
        # if you look closely you'll notice we already have a last layer after the attention, 
        # so we dont need to put a nother linear layer afterward, becasue that one linear layer
        # suffices (multiple linear layers, dont provide any higher abstraction anyway, 
        # so its prefectly fine)
        # but on the other hand the paper's implementation has another change, it states 
        # the inner layers dim are increased by 4x, so to be faithful to the paper we also 
        # add the second linear layer, with the suggested change, this wont change the output 
        # shape, so we are fine. we just added an extra linear projection layer. 
        # this additional projection operation actually improves the result,however if we simply 
        # use the same dim linear layer (i.e. have sth like linear(head_size, head_size)) adds nothing
        # to the representational power of the network and you wont see anything substantial vs 
        # if you completely remove this layer and only use linear/relu only.
        # why this works is becasue, we increase the nonlinear output neurons of the first layer by 4,
        # increasing its representational capacity, and then use the second linear layer to get the 
        # output size compatible for the next layer (doing a linear projection), and hence our 
        # improvements lie in the nonlinearity this addition provides. 
        self.feedforwardnet = nn.Sequential(nn.Linear(embd_size, head_size*4), nn.ReLU(),
                                            nn.Linear(head_size*4, head_size))
        # finally the output fc layer 
        self.fc = nn.Linear(head_size, vocab_size)
        
    def forward(self, inputs:torch.Tensor, labels:torch.Tensor=None)->torch.Tensor:
        B,T = inputs.shape
        # print(f'{inputs.shape=} {self.context_size=} {self.embd_size=}')
        token_embds = self.token_embeddings(inputs)
        position_embds = self.position_embeddings(torch.arange(T,device=self.device))
        embds_combilned = token_embds + position_embds
        # now lets feed them to our multihead attention block 
        out = self.multi_head_attention(embds_combilned)
        # add our new addition, feedforwardnet
        out = self.feedforwardnet(out)
        # and finally the logits 
        logits = self.fc(out)
        loss = None
        if labels is not None:
            # remember the cross entropy wanted its input in B,C,T while ours is in (B,T,C)
            loss = F.cross_entropy(logits.permute(0,2,1), labels)
        return logits, loss 

    def generate(self, idxs, max_token_count):
        assert idxs.ndim>1 , f'idxs.ndim({idxs.ndim}) must be 2 (in the form of (B,T))'
        for i in range (max_token_count):
            # we keep feeding the input to the model and get the next character
            # but since we use positional embeddings, we are limited to context_size
            # of tokens at anygiven time to feed the network or we face an error 
            # so we always tke the last T tokens from our input. we use negative
            # slicing, so if we have less than context_size, we only grab that many
            # otherwise, we get an error, becasue obviously at the begining we may 
            # start from a single token, denoting context_size of 1, while our model
            # expects like full context_size (e.g. 8 or more)
            idxs_cropped = idxs[:, -self.context_size:]
            logits,_ = self(idxs_cropped)
            # since we are after the next character only and we have to choose among 
            # context_size number of tokens, we get the last one and treat it as the next
            # character to calculate its probablity to sample from 
            logits = logits[:,-1,:] 
            # calulate the probs
            probs = logits.softmax(dim=-1)
            # sample the next character/token 
            idx_token_next = torch.multinomial(probs, num_samples=1, replacement=True)
            # add this to our existing tokens in idxs 
            idxs = torch.cat((idxs, idx_token_next), dim=-1)
            
        return idxs 

# and now lets train with the new change and see how it performs:
print(f'using feedforwardnet added')
torch.manual_seed(255)
random.seed(255)

head_num = 4
head_size = 32
embd_size = 32
context_size = 8 
vocab_size = len(vocab_list)
device = 'cpu'
use_bias_attn = False

lr = 0.001
batch_size = 32
max_iter = 5000
eval_period = 1000
model = BigramModelWithAttention(vocab_size=vocab_size, 
                                 context_size=context_size, 
                                 embd_size=embd_size,
                                 num_head=head_num, 
                                 head_size=head_size, 
                                 device=device,
                                 bias_attn=use_bias_attn)
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr)
param_count = sum(p.nelement() for p in model.parameters())

print(f'param_count  =  {param_count:,}')
print(f'head_num     =  {head_num}')
print(f'head_size    =  {head_size}')
print(f'embd_size    =  {embd_size}')
print(f'context_size =  {context_size}')
print(f'device       =  {device}')
print(f'use_bias_attn=  {use_bias_attn}')

for pos in range(max_iter):
    # read a batch 
    x, y = get_batch('train', batch_size=batch_size)
    logits, loss = model(x,y)
    
    # calculate the smoother loss on multiple batches on train/val splits
    if pos%eval_period == 0:
        losses = evaluate_loss(200, device)
        print(f"train: {losses['train']:.4f}  val: {losses['val']:.4}")
    # zero-out gradients 
    model.zero_grad()
    # we ccould also do 
    # optimizers.zero_grad() 
    # but since our optimizer 'only' uses our models parameters, they are basically the same
    # note we talk about this in more details later which one to use and where later on inshaalah
    
    # do a backward pass 
    loss.backward()
    # do a single optimization step 
    optimizer.step()

print(f'done!')
# lets see how this model fairs now and what it generates 
initial_token = torch.zeros(size=(1,1)).int()
output = model.generate(initial_token, max_token_count=500).squeeze().tolist()
print(f"{''.join(decode(output))}")
# which prints : 
# using feedforwardnet added
# param_count  =  5,169
# head_num     =  4
# head_size    =  16
# embd_size    =  16
# context_size =  8
# device       =  cpu
# use_bias_attn=  False
# train: 4.1975  val: 4.198
# train: 2.6222  val: 2.634
# train: 2.4919  val: 2.481
# train: 2.4320  val: 2.435
# train: 2.3979  val: 2.392
# done!

# Selwils iur.

# AnNdt
# By thak, yue, nein ende vat Rout'w haler,
# The hot hes? is, whas bles. Yig,
# ROm
# QUS:
# Ay Whigice rea:
# Therer a youst um mith dins I dorseancer by gou:
# I sot the, aserer tom sne arobfs-Rid,
# This sas whe ou,
# Acet Yark no.

# OT:
# Fely ip you pel hivet seatly. hial pand,
# Thand repwat,
# I tarve ther se this ten thio drord dot tho,
# Rulee res le, il fet!
# Nord Wotiilet homom for my woue cuve wid.

# Angilene.

# Fhivy, ih:
# Boumlke uthereaerf
# Whe ithe hares.

# Whos sthik thenth
# Tait:
# She tove w

# the loss didnt get better! but maybe if we increase the context_size, and embedding_size, it 
# perorm better?
# using a larger embedding_size and thus larger model did improve the result, 
#
# using feedforwardnet added
# param_count  =  15,905
# head_num     =  4
# head_size    =  32
# embd_size    =  32
# context_size =  8
# device       =  cpu
# use_bias_attn=  False
# train: 4.1569  val: 4.159
# train: 2.3885  val: 2.396
# train: 2.2789  val: 2.312
# train: 2.2064  val: 2.249
# train: 2.1599  val: 2.198
# done!

# To get fer arast deve lad, I' fale are you up this riany;
# Weareien,
# Tho Guser, me ter is! Do lothe, to ling wo whe Pist to deave mat not, in and- air besface, y to fron; use ler weet herefalf thop an der pretrier
# nesly toide
# Thats ous.

# RENT:
# I for the haw to, to noss teave ler shat earelss ater, want Heanct herry deeter erry, heave masere dacke; nochat onsemy ning Meib,--?

# LUCLUER:
# If Feropeake mo shat onsivoole washy the well beWarquent;
# I on well he ome it you lare to lo to kit youser,
# Weves

# the larger network, seems to be doing much better than its counter part from before
# the one without the ffnet, we got train-loss: 2.1599  val-loss: 2.198 here whereas 
# previously we got train-loss: 2.2713  val-loss: 2.288. so we have improvements despite the 
# text still not being that good(but still better than before). so we need more enhancements)
#%%
# the next improvement, would be to use more of these, as its shown in the paper, if one block
# works, then adding more should work better right? we had single head, it improved our condition, 
# so we used more heads, and now lets more multi-attention heads! to make things easier, lets
# make a module out of it the same way we created previous modules. 

# first lets create our ffnet as a seprate block, so we use it after the attention
class FeedForward(nn.Module):
    def __init__(self, n_features, bias=True) -> None:
        super().__init__()
        # since this is only used with attention, the in/out features are the same
        # and usually the embedding size in our case but the paper states that the
        # inner layer dims are increased 4 times, so lets also reflect this change here.
        # as we saw earlier, this improves our results.
        self.n_features = n_features
        self.block = nn.Sequential(nn.Linear(in_features=n_features, out_features=n_features*4, bias=bias),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(in_features=n_features*4, out_features=n_features, bias=bias))
    def forward(self, inputs):
        return self.block(inputs)
    
class AttentionwithFFNetBlock(nn.Module):
    def __init__(self, context_size, embd_size, num_head, head_size, bias_attn=False ) -> None:
        super().__init__()
        self.head_size = head_size
        self.context_size = context_size
        self.num_head = num_head
        self.embd_size = embd_size
        self.bias_attn = bias_attn
        
        self.attn = MultiHeadAttention(num_head=num_head,
                                       head_size=head_size//num_head,
                                       embd_size=embd_size, 
                                       context_size=context_size,
                                       bias_attn=bias_attn)
        # note as a reminder, this is being applied
        # after an attention module, and attention module's input is embd_size, while its 
        # output-dim is head_size (we know they are usually the same, but to be flexible
        # we always use head_size to be on the safe side))
        self.ffnet = FeedForward(embd_size, head_size)

    def forward(self, inputs:torch.Tensor) -> torch.Tensor:
        return self.ffnet(self.attn(inputs))

# now lets add this to our base model and see how it performs this time: 
class BigramModelWithAttention(nn.Module):
    def __init__(self, vocab_size, context_size, embd_size, num_head, head_size, num_blocks, device='cpu', bias_attn=False) -> None:
        super().__init__()
        self.vocab_size=  vocab_size
        self.context_size = context_size
        self.embd_size = embd_size
        self.num_head = num_head
        self.head_size = head_size
        # now lets add the number of blocks 
        self.num_blocks = num_blocks
        self.device = device
        # token/character embeddings
        self.token_embeddings = nn.Embedding(vocab_size, embd_size)
        # position embeddings
        self.position_embeddings = nn.Embedding(context_size, embd_size)
        # now lets use attention blocks instead of a multi-head-attention and ffnet
        # note that for this to work properly serialy, the output of this needs to be
        # the same as its input (which is embd_size) which by default for our case should
        # be ok.
        self.blocks = nn.Sequential(*[AttentionwithFFNetBlock(context_size=context_size, 
                                              embd_size=embd_size,
                                              num_head=num_head,
                                              head_size=head_size,
                                              bias_attn=bias_attn)
                                     for _ in range(num_blocks)])
        # finally the output fc layer 
        self.fc = nn.Linear(head_size, vocab_size)
        
    def forward(self, inputs:torch.Tensor, labels:torch.Tensor=None)->torch.Tensor:
        B,T = inputs.shape
        # print(f'{inputs.shape=} {self.context_size=} {self.embd_size=}')
        token_embds = self.token_embeddings(inputs)
        # dont forget, our positional embd only involves the token position information
        position_embds = self.position_embeddings(torch.arange(T,device=self.device))
        embds_combilned = token_embds + position_embds
        # now lets have several multi-head-attentions instead of 1, one after the other
        out = self.blocks (embds_combilned)
        # and finally the logits 
        logits = self.fc(out)
        loss = None
        if labels is not None:
            # remember the cross entropy wanted its input in B,C,T while ours is in (B,T,C)
            loss = F.cross_entropy(logits.permute(0,2,1), labels)
        return logits, loss 

    def generate(self, idxs, max_token_count):
        assert idxs.ndim>1 , f'idxs.ndim({idxs.ndim}) must be 2 (in the form of (B,T))'
        for i in range (max_token_count):
            # we keep feeding the input to the model and get the next character
            # but since we use positional embeddings, we are limited to context_size
            # of tokens at anygiven time to feed the network or we face an error 
            # so we always tke the last T tokens from our input. we use negative
            # slicing, so if we have less than context_size, we only grab that many
            # otherwise, we get an error, becasue obviously at the begining we may 
            # start from a single token, denoting context_size of 1, while our model
            # expects like full context_size (e.g. 8 or more)
            idxs_cropped = idxs[:, -self.context_size:]
            logits,_ = self(idxs_cropped)
            # since we are after the next character only and we have to choose among 
            # context_size number of tokens, we get the last one and treat it as the next
            # character to calculate its probablity to sample from 
            logits = logits[:,-1,:] 
            # calulate the probs
            probs = logits.softmax(dim=-1)
            # sample the next character/token 
            idx_token_next = torch.multinomial(probs, num_samples=1, replacement=True)
            # add this to our existing tokens in idxs 
            idxs = torch.cat((idxs, idx_token_next), dim=-1)
            
        return idxs 

# and now lets train with the new change and see how it performs:
print(f'using more blocks!')
torch.manual_seed(255)
random.seed(255)

head_num = 4
block_num = 3
head_size = 32
embd_size = 32
context_size = 8 
vocab_size = len(vocab_list)
device = 'cpu'
use_bias_attn = False

lr = 0.001
batch_size = 32
max_iter = 5000
eval_period = 1000
model = BigramModelWithAttention(vocab_size=vocab_size, 
                                 context_size=context_size, 
                                 embd_size=embd_size,
                                 num_head=head_num, 
                                 head_size=head_size,
                                 num_blocks=block_num,
                                 device=device,
                                 bias_attn=use_bias_attn)
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr)
param_count = sum(p.nelement() for p in model.parameters())

print(f'param_count  =  {param_count:,}')
print(f'head_num     =  {head_num}')
print(f'block_num    =  {block_num}')
print(f'head_size    =  {head_size}')
print(f'embd_size    =  {embd_size}')
print(f'context_size =  {context_size}')
print(f'device       =  {device}')
print(f'use_bias_attn=  {use_bias_attn}')

for pos in range(max_iter):
    # read a batch 
    x, y = get_batch('train', batch_size=batch_size)
    logits, loss = model(x,y)
    
    # calculate the smoother loss on multiple batches on train/val splits
    if pos%eval_period == 0:
        losses = evaluate_loss(200, device)
        print(f"train: {losses['train']:.4f}  val: {losses['val']:.4}")
    # zero-out gradients 
    model.zero_grad()
    # we ccould also do 
    # optimizers.zero_grad() 
    # but since our optimizer 'only' uses our models parameters, they are basically the same
    # note we talk about this in more details later which one to use and where later on inshaalah.
    
    # do a backward pass 
    loss.backward()
    # do a single optimization step 
    optimizer.step()

print(f'done!')
# lets see how this model fairs now and what it generates 
initial_token = torch.zeros(size=(1,1)).int()
output = model.generate(initial_token, max_token_count=500).squeeze().tolist()
print(f"{''.join(decode(output))}")
# prints 
# using more blocks!
# param_count  =  38,753
# head_num     =  4
# block_num    =  3
# head_size    =  32
# embd_size    =  32
# context_size =  8
# device       =  cpu
# use_bias_attn=  False
# train: 4.2087  val: 4.211
# train: 2.6084  val: 2.606
# train: 2.3737  val: 2.39
# train: 2.2865  val: 2.307
# train: 2.2087  val: 2.237
# done!

# FAUKIN YRE:
# Dato':
# wis.

# BERDEWEE:

# Piwit ancon; awt
# Emuke this his fred not pe,
# Har,
# And him my bird-wray sluf. Ile:
# With tulel wik, antegatorg.

# EREN VELV:
# Whath a meane, ones
# sheirss,
# Bose til
# Ilf bodtund ba,
# Aripce nerod is the blilshil hit hit mef?

# GESINCE:
# Sard, with the deoorrnsicominle thid: as.

# ANRE:
# Hy prall anges what micciths delre
# To tike comf hon ebat'd mur. Kin'sa, Fharth:
# Af'?

# ROASA:
# Ect mily fledd.
# SHricest.
# At dey bompe novor where lisg.

# KETILNINCE:
# There gay his my nefonk
#
#
# as you can see, we got wrose results than before! despite making the network larger, our loss
# really didnt improve as we expected. 
# is our initial hypothesis that having more blocks and higher nonlinearity/representation benificial
# wrong? or is there something else thats causing the issue? 
# as you might have guessed, its the latter. we are basically creating more layers, and with
# more layers, we face training issues that we discussed earlier. 
# so how should we tackle this, we cant use BN, becasue BatchNOrm, accumulates the statistics
# from different samples, this is a no no for us. we dont want other samples to interfer with 
# our sample (or basically each other!) in anyway, so what should we do? 
# the paper utilizes two mechanisms or operations to tackle this issue. one being skip-connections
# (also known as residual connction) and the other, layer-normalization. 
# you should be familiar with skip-connections as they are the founding factor for resenets
# and have been extremely influential. to cut a long story short, skip connections are simply
# connections from input that skip the operations involved in the block they reside in and are
# directly added to the output which is then returned as the result.(basically F(x) + x)
# as we know, the gradients are distributed equally when they reach addition, so input gets the
# gradients without being weakened due to large depth of the network. 
# so before we implement the layer normalization part, lets see how much of a change adding 
# a skip-connection causes and whether it proves our initial hypothesis about depth and gradient
# signal weakening, we add this to our AttentionBlock (but we could add this to any submodule)
#%%
class FeedForward(nn.Module):
    def __init__(self, n_features, bias=True) -> None:
        super().__init__()
        # since this is only used with attention, the in/out features are the same
        # and usually the embedding size in our case but the paper states that the
        # inner layer dims are increased 4 times, so lets also reflect this change here.
        # as we saw earlier, this improves our results.
        self.n_features = n_features
        self.block = nn.Sequential(nn.Linear(in_features=n_features, out_features=n_features*4, bias=bias),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(in_features=n_features*4, out_features=n_features, bias=bias))
    def forward(self, inputs):
        return self.block(inputs)
    
# lets add a skip-connection to this block
class AttentionwithFFNetBlock(nn.Module):
    def __init__(self, context_size, embd_size, num_head, head_size, bias_attn=False ) -> None:
        super().__init__()
        self.head_size = head_size
        self.context_size = context_size
        self.num_head = num_head
        self.embd_size = embd_size
        self.bias_attn = bias_attn
        
        self.attn = MultiHeadAttention(num_head=num_head,
                                       head_size=head_size//num_head,
                                       embd_size=embd_size, 
                                       context_size=context_size,
                                       bias_attn=bias_attn)
        # note as a reminder, this is being applied
        # after an attention module, and attention module's input is embd_size, while its 
        # output-dim is head_size (we know they are usually the same, but to be flexible
        # we always use head_size to be on the safe side))
        self.ffnet = FeedForward(embd_size, head_size)

    def forward(self, inputs:torch.Tensor) -> torch.Tensor:
        # since we have two blocks, we add skip-connection to both of them here
        # we could aggerate them as one and a add the skip connection to their output
        # but this is less benificial than creating seprate skip-connections for each block
        # to see this in action, uncomment this and comment the latter part and run the test
        # out = self.attn(inputs)
        # out =  self.ffnet(out)
        # return out + inputs 
        # (TODO add comments/imporvements from my imagexlet project)
        # sidenote:
        # Our block works fine but its old and based on my knowledge of 2020-era GPT-2/GPT-3
        # conventions. Modern LLMs do things differently now, but for now we keep going and
        # at the end, I'll address these changes and talk about them.
        out = self.attn(inputs) + inputs
        # out =  self.ffnet(out)  + out looks/sounds more natual,
        # but in our case, using inputs on both ops, does slightly better!
        out =  self.ffnet(out)  + inputs
        return out

# now lets add this to our base model and see how it performs this time: 
class BigramModelWithAttention(nn.Module):
    def __init__(self, vocab_size, context_size, embd_size, num_head, head_size, num_blocks, device='cpu', bias_attn=False) -> None:
        super().__init__()
        self.vocab_size=  vocab_size
        self.context_size = context_size
        self.embd_size = embd_size
        self.num_head = num_head
        self.head_size = head_size
        # now lets add the number of blocks 
        self.num_blocks = num_blocks
        self.device = device
        # token/character embeddings
        self.token_embeddings = nn.Embedding(vocab_size, embd_size)
        # position embeddings
        self.position_embeddings = nn.Embedding(context_size, embd_size)
        # now lets use attention blocks instead of a multi-head-attention and ffnet
        # note that for this to work properly serialy, the output of this needs to be
        # the same as its input (which is embd_size) which by default for our case should
        # be ok.
        self.blocks = nn.Sequential(*[AttentionwithFFNetBlock(context_size=context_size, 
                                              embd_size=embd_size,
                                              num_head=num_head,
                                              head_size=head_size,
                                              bias_attn=bias_attn)
                                     for _ in range(num_blocks)])
        # finally the output fc layer 
        self.fc = nn.Linear(head_size, vocab_size)
        
    def forward(self, inputs:torch.Tensor, labels:torch.Tensor=None)->torch.Tensor:
        B,T = inputs.shape
        # print(f'{inputs.shape=} {self.context_size=} {self.embd_size=}')
        token_embds = self.token_embeddings(inputs)
        # dont forget, our positional embd only involves the token position information
        position_embds = self.position_embeddings(torch.arange(T,device=self.device))
        embds_combilned = token_embds + position_embds
        # now lets have several multi-head-attentions instead of 1, one after the other
        out = self.blocks (embds_combilned)
        # and finally the logits 
        logits = self.fc(out)
        loss = None
        if labels is not None:
            # remember the cross entropy wanted its input in B,C,T while ours is in (B,T,C)
            loss = F.cross_entropy(logits.permute(0,2,1), labels)
        return logits, loss 

    def generate(self, idxs, max_token_count):
        assert idxs.ndim>1 , f'idxs.ndim({idxs.ndim}) must be 2 (in the form of (B,T))'
        for i in range (max_token_count):
            # we keep feeding the input to the model and get the next character
            # but since we use positional embeddings, we are limited to context_size
            # of tokens at anygiven time to feed the network or we face an error 
            # so we always tke the last T tokens from our input. we use negative
            # slicing, so if we have less than context_size, we only grab that many
            # otherwise, we get an error, becasue obviously at the begining we may 
            # start from a single token, denoting context_size of 1, while our model
            # expects like full context_size (e.g. 8 or more)
            idxs_cropped = idxs[:, -self.context_size:]
            logits,_ = self(idxs_cropped)
            # since we are after the next character only and we have to choose among 
            # context_size number of tokens, we get the last one and treat it as the next
            # character to calculate its probablity to sample from 
            logits = logits[:,-1,:] 
            # calulate the probs
            probs = logits.softmax(dim=-1)
            # sample the next character/token 
            idx_token_next = torch.multinomial(probs, num_samples=1, replacement=True)
            # add this to our existing tokens in idxs 
            idxs = torch.cat((idxs, idx_token_next), dim=-1)
            
        return idxs 

# and now lets train with the new change and see how it performs:
print(f'using more blocks with skip-connection')
torch.manual_seed(255)
random.seed(255)

head_num = 4
block_num = 3
head_size = 32
embd_size = 32
context_size = 8 
vocab_size = len(vocab_list)
device = 'cpu'
use_bias_attn = False

lr = 0.001
batch_size = 32
max_iter = 5000
eval_period = 1000
model = BigramModelWithAttention(vocab_size=vocab_size, 
                                 context_size=context_size, 
                                 embd_size=embd_size,
                                 num_head=head_num, 
                                 head_size=head_size,
                                 num_blocks=block_num,
                                 device=device,
                                 bias_attn=use_bias_attn)
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr)
param_count = sum(p.nelement() for p in model.parameters())

print(f'param_count  =  {param_count:,}')
print(f'head_num     =  {head_num}')
print(f'block_num    =  {block_num}')
print(f'head_size    =  {head_size}')
print(f'embd_size    =  {embd_size}')
print(f'context_size =  {context_size}')
print(f'device       =  {device}')
print(f'use_bias_attn=  {use_bias_attn}')

for pos in range(max_iter):
    # read a batch 
    x, y = get_batch('train', batch_size=batch_size)
    logits, loss = model(x,y)
    
    # calculate the smoother loss on multiple batches on train/val splits
    if pos%eval_period == 0:
        losses = evaluate_loss(200, device)
        print(f"train: {losses['train']:.4f}  val: {losses['val']:.4}")
    # zero-out gradients 
    model.zero_grad()
    # we ccould also do 
    # optimizers.zero_grad() 
    # but since our optimizer 'only' uses our models parameters, they are basically the same
    # note we talk about this in more details later which one to use and where later on inshaalah.
    
    # do a backward pass 
    loss.backward()
    # do a single optimization step 
    optimizer.step()

print(f'done!')
# lets see how this model fairs now and what it generates 
initial_token = torch.zeros(size=(1,1)).int()
output = model.generate(initial_token, max_token_count=500).squeeze().tolist()
print(f"{''.join(decode(output))}")
# prints 
# using more blocks with skip-connection
# param_count  =  38,753
# head_num     =  4
# block_num    =  3
# head_size    =  32
# embd_size    =  32
# context_size =  8
# device       =  cpu
# use_bias_attn=  False
# train: 4.5636  val: 4.553
# train: 2.2666  val: 2.281
# train: 2.1330  val: 2.183
# train: 2.0708  val: 2.14
# train: 2.0204  val: 2.109
# done!

# Zunt will jucked's whreef onst's so
# wippplalct, sawter.

# Firfor have deal pere,
# Shal,
# thich many.
# Frig.

# Thall have.

# WANWHAM:
# Sad king theave,
# Cnomlen nare, near was?


# CKINGlLOROMBOENGBETH:
# Dich loverd und by, may;
# ''Kh
# Godie the bline plock's minstell the denelsbad, with think
# What
# sicond lead this undrut, too, the but me my se ming'ld;
# Shing
# To tiot comford, engelan this it's heare: hear them the vinclamil.
# Is done, the stronced,
# Lion?
# Hartow where lisg.

# DOLILINA:
# Lord ISe and
# Lud'd nef hel

# and test with one skip-connection on the 'output only' to prove or assumption on skip-connections
# using more blocks with skip-connection
# param_count  =  38,753
# head_num     =  4
# block_num    =  3
# head_size    =  32
# embd_size    =  32
# context_size =  8
# device       =  cpu
# use_bias_attn=  False
# train: 4.5616  val: 4.552
# train: 2.3196  val: 2.322
# train: 2.1862  val: 2.221
# train: 2.1175  val: 2.163
# train: 2.0640  val: 2.137
# done!

# FARY VI:
# A-bame 'tries.

# BERGEWES:

# PABLLO:
# Yon; aws
# Then ath-pmate for, not pe,
# PARILA:
# So man.

# FORY.

# TAsUS:
# I cenpine nou, lad king theak,
# Anny, In nartine-'

# An yeant, of ladie ust, good ting lover that by, may;
# And hordie the
# llils plove ond sofful and heave
# Take with the deo,
# And mort leath and usir the dom the but me my crombasill;
# Shim are, con comlmork engeland your, trabe,
# Beth:
# Yor?

# CORIAR EFt mily fle--spoles, strence'd ono do now, will?
# Mursice,
# Ntormorest
# The juge ernus'd neforki

# as you can see, it greatly improved our results, and the text also got much better!
# as we already pointed out, having skip-connections per modules, help much more than a single 
# per module output only, nevertheless, we notice, using skip-connection really improved 
# our results.
# now lets add the second operation, layernorm. 
# layernorm(https://arxiv.org/abs/1607.06450) is a normalization layer, just like batchnormalization
# it came a year later than batchnormalization paper, but the difference between them is that,
# unlike batchnormalization it works on a per sample basis and does not involve using othersamples
# to normalize a specific sample (basically samples dont affect eachother)
# As pytorch docs puts it:
# "Unlike Batch Normalization and Instance Normalization, 
#  which applies scalar scale and bias for each entire channel/plane with the affine option,
#  Layer Normalization applies per-element scale and bias with elementwise_affine"
#
# Layernorm makes sure that the mean and standard deviation of each feature across each example
# is (0,1). It preserves the relative relationships between the features within each example.
# on the other hand BatchNorm makes sure the mean and standard deviation of each feature across
# the *entire batch* is (0,1)! because of this, it introduces dependencies between examples in 
# the batch during training so for inference its disabled and a moving average it calculated during
# training is used instead to allow independent predictions. unlike batchnormalization, layernorm
# behaves the same during training and inference as it normalizes each example independently.

# When Layernorm was published it was mostly used in recurrent neural networks (RNNs), such as
# LSTMs and GRUs, where the normalization is applied along the time steps (sequence length) 
# dimension, where BatchNorm wouldnt work, but on cnn it was not as effective as BatchNormalization.
# later, it was also used on transformer architectures. now we need to use it here. 
#  
# Layernorm implementation is similar to the batchnormalization, and it does not require calculating
# running_mean/var we can use the pytorch module just fine, but since its really similar to BN,
# lets implement it here 
# 
#%%
class LayerNorm(nn.Module):
    def __init__(self, in_features, eps=1e-5) -> None:
        super().__init__()
        self.in_features = in_features
        self.eps = eps
        # alpha_ln_gain
        # note that since we are inheriting from nn.Module, and are in pytorch teritory
        # we need to use nn.Parameter to mark our parameters trainable! otherwise 
        # pytorch will ignore them, even if we set the requires_grad as true!
        self.alpha_ln_gain = torch.nn.Parameter(torch.ones(size=(1,in_features)))
        # beta_ln_bias
        self.beta_ln_bias = torch.nn.Parameter(torch.zeros(size=(1,in_features)))
    
    def forward(self, inputs):
        # calculate the mean and var
        # we calculate the mean for the last dims, that is feature dims (we are trying to normalize the features)
        # so we specify what we consider feature dims.
        #
        # note: initially I wrote 
        # dims = tuple(i for i in range(inputs.ndim-1,0,-1)) to grab dims for calculating
        # min/var for those dims dynamically for different input shapes!(2d or more)
        # which didnt cause any errors, and infact resulted in a very low loss, however the 
        # text generation seemed kind of random, and differed from the pytorch's Layernorm 
        # (0.60 vs 1.90 of pytorch's which is a huge diference). 
        # Further investigation revealed the issue was caused by the dims involved. 
        # we needed to only use the last column which would be (-1) in our specific case. 
        # basically what we want is to calculate mean/var for the dims that belong to features,
        # e.g. in a 2d case like (B,C), we want to mean/var on dim=1. 
        # Having worked with BN, we might also assume the same here that is if we have sth like 
        # (B,T,C) we would want to treat, B,T as batch and aggregate the mean/var along dim=(1,2).
        # however, as we learned earlier, LayerNorm does not involve other samples at all, 
        # and infact if we do this, we'll see despite the loss decreasing rapidly, our text generation
        # quality becoming abysmal! so when it comes to LayerNorm we always normalize the feature dimensions,
        # and in our case its just the last diminsion. 
        # what was causing the issue here was this exact issue, since I was calculating the dims
        # dynamically based on the inputs shape, for 2d inputs(ignoring batch) this would work as expected,
        # but for 3d+, it would aggregate other samples stats and therefore create the discrepency in 
        # the output between ours and pytorch's.
        # (we were calculating the mean/var for dims=(1,2) while Pytorch was only calculating it on 
        # the last dim, and hence the difference. (the output shows that involving other samples adversly
        # affect our output and hence why BN is not used and instead LN is used.) 
        # this happened becasue we dynamically tried to infer the dims by looking at the input shape
        # the behavior for 2d shapes will be the same, however, for the 3d shapes, our results differ.
        # we can define the aggregation along feature dims in Pytorch, so if we wanted to get the same 
        # behavior in pytorch we had to write sth like this: 
        # self.ln1 = nn.LayerNorm((context_size,head_size)) 
        # that is instead of using a single number denoting the dimension's size, we specify the dim's 
        # size explicticly. 
        # obviously since we are coding this for our usecase, we dont bother getting the input shape here
        # and use -1 to get the job done, otherwise, we would be getting the dim's size as input just 
        # like pytorch and based on the dim's size, decide how to do mean/var. 
        # side note: 
        # note that if we use sth self.ln1 = nn.LayerNorm((context_size,head_size))  in our code, 
        # during text generation, we no longer can start with context_size of 1, we must always start
        # with sth like initial_token = torch.zeros(size=(1,8)).int() to get it working otherwise it'd
        # complain about shape mismatch which is expected because unlike our method, its static 
        # (while ours dynamically would calculate the mean/var based on the inputsize) anyway, 
        # this shouldnt be an issue, becasue we dont need to do this as not only it doesnt benifit us 
        # but also creates more hassle!
        # side note2: 
        # aggregation in LN, may be benificial in other domains, as BN was, but for us, now it isnt
        # so we only make the one that works with the last dim! 
        # here's the wrong snippet that would create the wrong result for 3d inputs.
        #dims = tuple(i for i in range(inputs.ndim-1,0,-1)) 
        # print(f'{dims=}')
        dims = -1
        mean = inputs.mean(dim=dims, keepdim=True)
        # to get the same outputas pytorch's, we use the biased version as well
        var = inputs.var(dim=dims, keepdim=True, unbiased=False)
        # ​ (x−E[x])​
        # ----------    ∗ γ + β
        # sqrt(Var[x]+ϵ)
        # 
        xhat = (inputs - mean)/torch.sqrt(var+self.eps)  
        out = xhat * self.alpha_ln_gain + self.beta_ln_bias
        return out

# lets test it 
x = torch.randn(size = (3,8,300))
ln = LayerNorm(300)
ln_torch = nn.LayerNorm(300)
out = ln(x)
out_torch = ln_torch(x)
# now we expect that each sample/row now to have mean=0 and var=1
# (peviously for batchnorm this was the opposite, the mean/var 
# were computed for the whole samples,for each columns so that
# out[:,0] would be mean=0 var=1, likewise out[:,1] and etc )
# for layernorm however, we need out[0,:] to be mean=0 var=1 now
print(f"our's: {out[0,:].mean().item(), out[0,:].var().item()}")
print(f"torch: {out_torch[0,:].mean().item(), out_torch[0,:].var().item()}")
# ok now lets add this to our AttentionwithFFNetBlock and test it 
#
#%%
class AttentionwithFFNetBlock(nn.Module):
    def __init__(self, context_size, embd_size, num_head, head_size, bias_attn=False ) -> None:
        super().__init__()
        self.head_size = head_size
        self.context_size = context_size
        self.num_head = num_head
        self.embd_size = embd_size
        self.bias_attn = bias_attn
        
        self.attn = MultiHeadAttention(num_head=num_head,
                                       head_size=head_size//num_head,
                                       embd_size=embd_size, 
                                       context_size=context_size,
                                       bias_attn=bias_attn)
        # note as a reminder, this is being applied
        # after an attention module, and attention module's input is embd_size, while its 
        # output-dim is head_size (we know they are usually the same, but to be flexible
        # we always use head_size to be on the safe side))
        self.ffnet = FeedForward(embd_size, head_size)
        # lets add the layernorm and to be sure our implementation works lets also test 
        # with pytorch's layernorm
        # self.ln1 = nn.LayerNorm(head_size)
        # self.ln2 = nn.LayerNorm(head_size)
        self.ln1 = LayerNorm(head_size)
        self.ln2 = LayerNorm(head_size)

    def forward(self, inputs:torch.Tensor) -> torch.Tensor:
        # note that when it comes to applying normalization, there are two ways of going about it
        # 1.normalize the outputs of the attention block
        # 2.normalize the inputs before going to the attention block
        # the first one is the one the paper initially used, but later on it was shown that actually
        # normalzing the input yields better result, so we test both cases here
        # so check and see how it affects it! (personally,however, I found the first approach
        # yielding midly better loss, but I only tested it on small scale, so we will test this more)
        # method 1:
        # out = self.ln1(self.attn(inputs)) + inputs
        # out = self.ln2(self.ffnet(out))  + inputs
        # method 2:
        out = self.attn(self.ln1(inputs)) + inputs
        out =  self.ffnet(self.ln2(out))  + inputs
        return out

# we also need to add layernorm at the end of the block before feeding it to the nextlayer
class BigramModelWithAttention(nn.Module):
    def __init__(self, vocab_size, context_size, embd_size, num_head, head_size, num_blocks, device='cpu', bias_attn=False) -> None:
        super().__init__()
        self.vocab_size=  vocab_size
        self.context_size = context_size
        self.embd_size = embd_size
        self.num_head = num_head
        self.head_size = head_size
        # now lets add the number of blocks 
        self.num_blocks = num_blocks
        self.device = device
        # token/character embeddings
        self.token_embeddings = nn.Embedding(vocab_size, embd_size)
        # position embeddings
        self.position_embeddings = nn.Embedding(context_size, embd_size)
        # now lets use attention blocks instead of a multi-head-attention and ffnet
        # note that for this to work properly serialy, the output of this needs to be
        # the same as its input (which is embd_size) which by default for our case should
        # be ok.
        self.blocks = nn.Sequential(*[AttentionwithFFNetBlock(context_size=context_size, 
                                              embd_size=embd_size,
                                              num_head=num_head,
                                              head_size=head_size,
                                              bias_attn=bias_attn)
                                     for _ in range(num_blocks)])
        # lets test both layers
        # self.ln1 = nn.LayerNorm(head_size)
        self.ln1 = LayerNorm(head_size)
        
        # finally the output fc layer 
        self.fc = nn.Linear(head_size, vocab_size)
        
    def forward(self, inputs:torch.Tensor, labels:torch.Tensor=None)->torch.Tensor:
        B,T = inputs.shape
        # print(f'{inputs.shape=} {self.context_size=} {self.embd_size=}')
        token_embds = self.token_embeddings(inputs)
        # dont forget, our positional embd only involves the token position information
        # if we didnt have self.device we coulds use model's parameters device to dynamically
        # set the proper device here which would be:
        # device = next(self.parameters()).device
        # and this way, setting model.cuda() or model.cpu() would do the trick here, but
        # currently, we have to explictily set model.device in order to update this snippets
        # device or otherwise it will fail becasue of mistamtching devices (if we train on 
        # gpu and intend on running on cpu imediately! of course we can always save states
        # and load a cpu only model, but we need to be consistent in our code)
        position_embds = self.position_embeddings(torch.arange(T,device=device))
        embds_combilned = token_embds + position_embds
        # now lets have several multi-head-attentions instead of 1, one after the other
        out = self.blocks (embds_combilned)
        # normalize by layernorm 
        out = self.ln1(out)
        # and finally the logits 
        logits = self.fc(out)
        loss = None
        if labels is not None:
            # remember the cross entropy wanted its input in B,C,T while ours is in (B,T,C)
            loss = F.cross_entropy(logits.permute(0,2,1), labels)
        return logits, loss 

    def generate(self, idxs, max_token_count):
        assert idxs.ndim>1 , f'idxs.ndim({idxs.ndim}) must be 2 (in the form of (B,T))'
        for i in range (max_token_count):
            # we keep feeding the input to the model and get the next character
            # but since we use positional embeddings, we are limited to context_size
            # of tokens at anygiven time to feed the network or we face an error 
            # so we always tke the last T tokens from our input. we use negative
            # slicing, so if we have less than context_size, we only grab that many
            # otherwise, we get an error, becasue obviously at the begining we may 
            # start from a single token, denoting context_size of 1, while our model
            # expects like full context_size (e.g. 8 or more)
            idxs_cropped = idxs[:, -self.context_size:]
            logits,_ = self(idxs_cropped)
            # since we are after the next character only and we have to choose among 
            # context_size number of tokens, we get the last one and treat it as the next
            # character to calculate its probablity to sample from 
            logits = logits[:,-1,:] 
            # calulate the probs
            probs = logits.softmax(dim=-1)
            # sample the next character/token 
            idx_token_next = torch.multinomial(probs, num_samples=1, replacement=True)
            # add this to our existing tokens in idxs 
            idxs = torch.cat((idxs, idx_token_next), dim=-1)
            
        return idxs 
#%%
# and now lets train with the new change and see how it performs:
print(f'using more blocks with skip-connection-layernorm')
torch.manual_seed(255)
random.seed(255)

head_num = 4
block_num = 3
head_size = 32
embd_size = 32
context_size = 8 
vocab_size = len(vocab_list)
device = 'cpu'
use_bias_attn = False

lr = 0.001
batch_size = 32
max_iter = 5000
eval_period = 1000
model = BigramModelWithAttention(vocab_size=vocab_size, 
                                 context_size=context_size, 
                                 embd_size=embd_size,
                                 num_head=head_num, 
                                 head_size=head_size,
                                 num_blocks=block_num,
                                 device=device,
                                 bias_attn=use_bias_attn)
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr)
param_count = sum(p.nelement() for p in model.parameters())

print(f'param_count  =  {param_count:,}')
print(f'head_num     =  {head_num}')
print(f'block_num    =  {block_num}')
print(f'head_size    =  {head_size}')
print(f'embd_size    =  {embd_size}')
print(f'context_size =  {context_size}')
print(f'device       =  {device}')
print(f'use_bias_attn=  {use_bias_attn}')

for pos in range(max_iter):
    # read a batch 
    x, y = get_batch('train', batch_size=batch_size)
    logits, loss = model(x,y)
    
    # calculate the smoother loss on multiple batches on train/val splits
    if pos%eval_period == 0:
        losses = evaluate_loss(200, device)
        print(f"train: {losses['train']:.4f}  val: {losses['val']:.4}")
    # zero-out gradients 
    model.zero_grad()
    # we ccould also do 
    # optimizers.zero_grad() 
    # but since our optimizer 'only' uses our models parameters, they are basically the same
    # note we talk about this in more details later which one to use and where later on inshaalah.
    
    # do a backward pass 
    loss.backward()
    # do a single optimization step 
    optimizer.step()

print(f'done!')
# lets see how this model fairs now and what it generates 
initial_token = torch.zeros(size=(1,1)).int()
output = model.generate(initial_token, max_token_count=500).squeeze().tolist()
print(f"{''.join(decode(output))}")
# prints
# ----------------
#using more blocks with skip-connection-layernorm(using ours!-the wrong layernorm implementation)
# see the note at the end.
# param_count  =  38,753
# head_num     =  4
# block_num    =  3
# head_size    =  32
# embd_size    =  32
# context_size =  8
# device       =  cpu
# use_bias_attn=  False
# train: 4.3881  val: 4.381
# train: 1.5804  val: 1.594
# train: 1.0614  val: 1.102
# train: 0.8326  val: 0.8784
# train: 0.7059  val: 0.7436
# done!
#
#
#
# VNIUKIVUTADES:'Kow seof on tith, owiw that thasawtde us athly hakef Bot pele,
# I al,
# I windis,
# And i-wradtsl! u slenerugh talelad kied togange,
# norl wink
# Tint ar in yoroqusendsld elust, gh hote woll brath hiss,
# A yoce nerdd verelLe I'lampl hat thisseflled
# Shell wharderw by in belourrns mominle thid;
# Aulir,
# Antoor plall'aege myane momo'thead bust ofting cow mol, esth'ang s. mit'sa, arak h:
# I thouge;
# Shat tam
# BONose,
# Low she storncexey an pot owoe whe?
# My sted-s't foot song
# qoulixe?
# Lur'deae:
# fol
#
#--------------
#
# using more blocks with skip-connection-layernorm(using pytorch's)
# param_count  =  39,201
# head_num     =  4
# block_num    =  3
# head_size    =  32
# embd_size    =  32
# context_size =  8
# device       =  cpu
# use_bias_attn=  False
# train: 4.3906  val: 4.385
# train: 2.2358  val: 2.242
# train: 2.1059  val: 2.154
# train: 2.0323  val: 2.098
# train: 1.9834  val: 2.068
# done!
#
# Fontive that mad's wise:
# But tith, luidien:
# Yord awto-must this have deat perefed al,
# thing is this is.
#
# PARUABE:
# O.
#
# WANWHAR:
# Yow king, what wirny.
# No nare, near in yeane, of lady. Whe, ghue tinst of this him,
# Murspose him dis the
# thine plock, this the pandse:
# Which will contend
# Whiresim thele this:
# Wutire
# A too, the but me wicHe micciodsh him are ting comf honamble's mur. Kin'sabhanke hear rend the thy the
# But seades, the strenced,
# Liven. How earse?
# Mursin.
#
# Nt from sing, I'll be
# Lur'd neforri
#
# --------------
# our layernorm fixed: 
# using more blocks with skip-connection-layernorm
# param_count  =  39,201
# head_num     =  4
# block_num    =  3
# head_size    =  32
# embd_size    =  32
# context_size =  8
# device       =  cpu
# use_bias_attn=  False
# train: 4.3906  val: 4.385
# train: 2.2373  val: 2.244
# train: 2.1070  val: 2.155
# train: 2.0339  val: 2.102
# train: 1.9812  val: 2.069
# done!

# For the that mad'
# With of on tith, luidie anct, saws
# Eule at botht if thou
# Wlefted' you windian.

# FLO-wike sluke slen:
# I have, ladik, anteraves,
# norl wink
# the bard a yeane, of lady. Whe, goshot I
# thy brath his what:
# Nown hondin the
# blinsh, shat this the pandself livadet brives nefors.
# ISIUS:
# Aren that usir the of me
# he hange wicHe mirs,
# Behat!


# TOLAT:
# But ford, engelan this it'sabhanke hear rend these this in usse,--
# Lord, strence'er thy?
# In woes. IDandry,
# Soset frongs neved
# With haul'd nefonk 
#
#
#
# for some reason, our layernorm achieves a much lower loss, much quicker, 
# however the text seems to be worse than pytorch's for some reason and
# I have no idea what is causing this! 
# ok there were two issues, 1.our gamma and beta werent trained! 
# becasue we forgot to wrap them in nn.Parameter and 
# the second reason was, we were aggregating the last two dims like BN,
# whereas we should have only used the last dim, 
# as we should not involve other samples in normalization. 
# (I explained this thoroughly in the LayerNorm class)
# 
# now how can we improve more? we implemented the paper, 
# basically we implemented transormer from scratch
# and what remains is to test with different hyperparameters 
# to see how well it can generate texts similar to our dataset. 
# so lets increase the model size now and see how much improvement we can get 
# 
# %%
print(f'using more blocks with skip-connection-layernorm-beefed up!')
import time
import torch
torch.manual_seed(255)
random.seed(255)

torch.cuda.memory.reset_max_memory_allocated(0)

#%%
torch.cuda.memory.empty_cache()
# head_num < block_num , increase block_num over head_num
# context_size doesnt afffect loss much than embd/block num
# between embd/bluck_num, increase block_num to achieve better result with less param
# with the same embd_size.(6 blocks of 144 embd_size > 4 blocks of 288 embd_size) or not!
# more embd_size with same block_num, performs much better at the cost of twice param count
# but if lower param count is required, then block_num with same embd_size is better(some times
# higher block_num achives comparable result with fewerr param counts)
# head_num doesnt impose much overhead, and no param increase, when all params are sorted out
# try increasing head_hum, you'll notice when it stops benifiting or is not just worth it,
# so start with 1, adjust other params, and then play with head_num.
start = time.time()
head_num = 2        #2 #4  #6
block_num = 6 # aka layers!# 1 # 2# 4# 6s
head_size = 288     #18 #36 #72 #144 #288 #396 #384
embd_size = 288     #18 #36 #72 #144 #288 #396 #384
context_size = 64  #8  #16 #32 #64*  #128 #256
vocab_size = len(vocab_list)
device = 'cuda'
use_bias_attn = False

lr = 0.0001
batch_size = 128
max_iter = 10000
eval_period = 1000
model = BigramModelWithAttention(vocab_size=vocab_size, 
                                 context_size=context_size, 
                                 embd_size=embd_size,
                                 num_head=head_num, 
                                 head_size=head_size,
                                 num_blocks=block_num,
                                 device=device,
                                 bias_attn=use_bias_attn)
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr)
param_count = sum(p.nelement() for p in model.parameters())

print(f'param_count  =  {param_count:,}')
print(f'head_num     =  {head_num}')
print(f'block_num    =  {block_num}')
print(f'head_size    =  {head_size}')
print(f'embd_size    =  {embd_size}')
print(f'context_size =  {context_size}')
print(f'batch_size   =  {batch_size}')
print(f'device       =  {device}')
print(f'use_bias_attn=  {use_bias_attn}')

for pos in range(max_iter):
    # read a batch 
    x, y = get_batch('train', batch_size=batch_size)
    x,y= tuple(t.to(device) for t in (x,y))
    logits, loss = model(x,y)
    
    # calculate the smoother loss on multiple batches on train/val splits
    if pos%eval_period == 0:
        losses = evaluate_loss(200, device)
        print(f"train: {losses['train']:.4f}  val: {losses['val']:.4}")
    # zero-out gradients 
    model.zero_grad()
    # we ccould also do 
    # optimizers.zero_grad() 
    # but since our optimizer 'only' uses our models parameters, they are basically the same
    # note we talk about this in more details later which one to use and where later on inshaalah.
        
    # do a backward pass 
    loss.backward()
    # do a single optimization step 
    optimizer.step()

print(f'done!')
print(f'elapsed: {time.time() - start} ')

# lets see how this model fairs now and what it generates 
model.cuda()
#model.device ='cuda'
initial_token = torch.zeros(size=(1,1)).int().cuda()
output = model.generate(initial_token, max_token_count=1000).squeeze().tolist()
print(f"{''.join(decode(output))}")
# which prints 
# using more blocks with skip-connection-layernorm-beefed up!
# param_count  =  9,901,889
# head_num     =  6
# block_num    =  6
# head_size    =  384
# embd_size    =  384
# context_size =  256
# device       =  cuda
# use_bias_attn=  False
# train: 4.3845  val: 4.388
# train: 1.9630  val: 2.045
# train: 1.5727  val: 1.755
# train: 1.4054  val: 1.632
# train: 1.3029  val: 1.57
# done!
#
#
# What, my Lady of Angelo?
#
# BUCKINGHAM:
# At at the untimely of doth bed, much,
# I'll be a kind his worth sea of the
# live mone conclaim my sorrow and amblitude
# Dut jot the watching of goos, Caius York,
# Whom hath power tire mild, I will leave you tell the
# is it subjects.

# GLOUCESTER:
# So rass, swing play his manner great his love
# Death is cloid of his fine.

# POLIXENES:
# No.

# Ghost offence, sea, when we he is name
# Where it were become me one years.

# TLBONA:
# Who, is he can you for the horse,
# That he clave upon him. Grim to't her.

# BRUTUS:
# It be more fair climits night rest thinke I
# not, and what all worse this.

# SAETER:
# Sir you are you must you wed?
# Doth mendles when you have once descries them,
# The harm of cullent you cursue for this:
# Resole he must be abused you.

# Clown:
# Kill, by my your highne, and that all bitter thee.

# JULIET:
# Gaunt, I'll make thee.
# Untired my brother birisment buriest!
# I pray, minister in my eight; it is it he
# not fastic that that fant for let thou
# of thine, or sleepter.
#
# second try
# param_count  =  9,901,889
# head_num     =  6
# block_num    =  6
# head_size    =  384
# embd_size    =  384
# context_size =  256
# device       =  cuda
# use_bias_attn=  False
# train: 4.3845  val: 4.388
# train: 1.9630  val: 2.045
# train: 1.5727  val: 1.755
# train: 1.4054  val: 1.632
# train: 1.3029  val: 1.57
# done!
#
# DUKE VINCENTIO:
# Scent the Faulia.

# KING EDWARD:
# Sweet, good conceive these are to be.

# CAPULET:
# Not gentle Menenius, here well they pend lives and I
# Rengeath one much buried.

# HARCID:
# O, God I'll: Bidish, Brobet, not--
# Hath said, quotest me, is a done.

# CORIOLANUS:
# Then, no proud heret! what sat withile: woen so,
# ha, wary thou comest, why thought that art thou nelsest;
# yet fights on his dearth-hoad yoking hered.

# KING EGBRY:
# Very Grey, my lord?

# CORIOLANUS:
# The people.
# Why let, sir, who have been sweet, Let to thee.

# BARBIUNCENA:
# In to the rest, my blood; Warwick's house.
# Take you in his worth train'd then he mone.

# ROMEO:
# Those wasted damnable, Eauton,
# Like, and brisothe comes do I nay,
# and that you know I were, and you lead us
# I tept brail, if such enemi: knack'd love from
# Unsuing person, and never. O, you should woman joys;
# But in this seven strem one of mine honestes
# Their woman to die: nay, sir, and were back
# if the imploteful of late, our chante faults
# He that wretche with this.
#----------------------------------
#
# which is remarkably better than all of our previous outputs. 
# so as we increased our model capacity we witnessed much better results. 
# side note:note that our positinal emebdding's need to be placed on the cpu or gpu explicitly 
# after model instantiation, or otherwise, as its set separately as the model, simply doing model.cpu()
# or model.cuda() wouldnt do it. so here I simply used cuda. (or we have to set the device in forward
# dynamically sth like device = next(model.parameters()).device)
#
#%%
# training this with large batchsize was really hard as it consumed a lot of vram, can we somehow
# do sth about it maybe? yes, pytorch supports half-precision training/and quantization for inference
# we use half-precision training(or as some refer to by mixed precision becasue for specific ops fp32 is 
# used to not hinder the optimization process) to train in fp16 rather than full precision or fp32 and it should 
# boost our training speed and decrease our vram consumption. we look at quantization in its respective 
# section in the future. 
# https://pytorch.org/blog/what-every-user-should-know-about-mixed-precision-training-in-pytorch/
# https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html
# cuzwe are in an ipython environment lets free the vram cache manually
# Best Practices
# We strongly recommend using mixed precision with torch.amp or the TF32 mode (on Ampere and later CUDA devices) whenever possible when training a network. If one of those approaches doesn’t work, however, we recommend the following:
#     High Performance Computing (HPC) applications, regression tasks, and generative networks may simply require full float32 IEEE precision to converge as expected.
#     Try selectively applying torch.amp. In particular we recommend first disabling it on regions performing operations from the torch.linalg module or when doing pre- or post-processing. These operations are often especially sensitive. Note that TF32 mode is a global switch and can’t be used selectively on regions of a network. Enable TF32 first to check if a network’s operators are sensitive to the mode, otherwise disable it.
#     If you encounter type mismatches while using torch.amp we don’t suggest inserting manual casts to start. This error is indicative of something being off with the network, and it’s usually worth investigating first.
#     Figure out by experimentation if your network is sensitive to range and/or precision of a format. For example fine-tuning bfloat16-pretrained models in float16 can easily run into range issues in float16 because of the potentially large range from training in bfloat16, so users should stick with bfloat16 fine-tuning if the model was trained in bfloat16.
#     The performance gain of mixed precision training can depend on multiple factors (e.g. compute-bound vs memory-bound problems) and users should use the tuning guide to remove other bottlenecks in their training scripts. Although having similar theoretical performance benefits, BF16 and FP16 can have different speeds in practice. It’s recommended to try the mentioned formats and use the one with best speed while maintaining the desired numeric behavior.
# https://nvlabs.github.io/eccv2020-mixed-precision-tutorial/
# https://blog.paperspace.com/automatic-mixed-precision-using-pytorch/
torch.cuda.memory.empty_cache()
#%%
start = time.time()
head_num = 2        #2 #4  #6
block_num = 6 # aka layers!# 1 # 2# 4# 6s
head_size = 288     #18 #36 #72 #144 #288 #396 #384
embd_size = 288     #18 #36 #72 #144 #288 #396 #384
context_size = 64  #8  #16 #32 #64*  #128 #256
vocab_size = len(vocab_list)
device = 'cuda'
use_bias_attn = False

lr = 0.0001
batch_size = 256
max_iter = 10000
eval_period = 1000
# enable mixed-precision 
# enabling it can boost our training speed and depending on the model lower the memory
# usgae drastically. it may also even improve our results to some extend!
use_mix_precision = True

model = BigramModelWithAttention(vocab_size=vocab_size, 
                                 context_size=context_size, 
                                 embd_size=embd_size,
                                 num_head=head_num, 
                                 head_size=head_size,
                                 num_blocks=block_num,
                                 device=device,
                                 bias_attn=use_bias_attn)
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr)
param_count = sum(p.nelement() for p in model.parameters())


print(f'param_count  =  {param_count:,}')
print(f'head_num     =  {head_num}')
print(f'block_num    =  {block_num}')
print(f'head_size    =  {head_size}')
print(f'embd_size    =  {embd_size}')
print(f'context_size =  {context_size}')
print(f'batch_size   =  {batch_size}')
print(f'device       =  {device}')
print(f'use_bias_attn=  {use_bias_attn}')
print(f"mixed-precision {'Enabled' if use_mix_precision else 'Disabled'}")

# before we start the training loop, we create a scaler 
# using the enabled argument we can easily enable/disable autocast
scaler = torch.cuda.amp.grad_scaler.GradScaler(enabled=use_mix_precision)


for pos in range(max_iter):
    # this context manager will take care of the dtype conversions for us
    # when we set enabled=False, the scaler and autocast basically become no op!
    # and we can seemlessly switch between them without any code changes at all!
    with torch.cuda.amp.autocast(enabled=use_mix_precision):
     # read a batch 
        x, y = get_batch('train', batch_size=batch_size)
        x,y= tuple(t.to(device) for t in (x,y))
        logits, loss = model(x,y)
        
        # calculate the smoother loss on multiple batches on train/val splits
        if pos%eval_period == 0:
            losses = evaluate_loss(200, device)
            print(f"train: {losses['train']:.4f}  val: {losses['val']:.4}")
        # zero-out gradients 
        model.zero_grad()
        # we ccould also do 
        # optimizers.zero_grad() 
        # but since our optimizer 'only' uses our models parameters, they are basically the same
        # note we talk about this in more details later which one to use and where later on inshaalah.
        
        # scale the loss and then do a backward pass 
        scaler.scale(loss).backward()
        # do a single optimization step - but making sure we take into account the mixed dtypes now!
        # so we instead use scaler.step and pass the optimzier so it takes care of everything for us.
        #  also note that if for whateevr reason we wanted to check the gradients, or clip them, etc
        # we have to first unscale them. 
        # basically all gradients produced by scaler.scale(loss).backward() are scaled. 
        # so if we wish to modify or inspect the parameters .grad attributes between backward()
        # and scaler.step(optimizer), we need to unscale them first using scaler.unscale_(optimizer)
        # and then carry on whith whatever it is that we intend on doing!(we will come back to this later on inshaallah)
        scaler.step(optimizer)
        # and finally update the scaler for the next iteration 
        scaler.update()
        # save model params 
        if pos%5000==0:
            # When saving, save the scaler state dict alongside the usual model and optimizer state dicts. 
            # we either  do this at the beginning of an iteration before any forward passes, 
            # or at the end of an iteration after scaler.update()
            # read more here https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html
            checkpoint = {"model": model.state_dict(),
                         "optimizer": optimizer.state_dict(),
                         "scaler": scaler.state_dict()}

torch.save(checkpoint, 'checkpoint_model_mx.pth')
print(f'done!')
print(f'elapsed: {time.time() - start} ')

# lets see how this model fairs now and what it generates 
model.cuda()
#model.device ='cuda'
initial_token = torch.zeros(size=(1,1)).int().cuda()
output = model.generate(initial_token, max_token_count=1000).squeeze().tolist()
print(f"{''.join(decode(output))}")
#prints
# param_count  =  5,546,369
# head_num     =  2
# block_num    =  6
# head_size    =  288
# embd_size    =  288
# context_size =  64
# device       =  cuda
# use_bias_attn=  False
# train: 4.3098  val: 4.309
# train: 1.7753  val: 1.891
# train: 1.5260  val: 1.706
# train: 1.4175  val: 1.622
# train: 1.3548  val: 1.582
# done!
# elapsed: 174.44584369659424 

# And never be why tears do counsel the beared?
# By all whish no hours of thy enemest, and what
# are would cries lack is thine are fiving:
# Oved did your commit i'
# since, you have of the town to tear thee I
# he was-larded their sight? and to what have me deserve;
# Had no kind hath handsbury'd thou noble,
# That I so? 'Tis found too against stand
# Hererated endured with her souls.

# queen:
# No honoured thee to bear it.

# Third Messenger:
# Our hap an happy death?
# O, they no follow losom
# To cure take, noble womb
# I nobly well every divince was it enough.

# NORTHUMBERLAND:
# I let him not with your grace.
# Curses not murderer, of a vow the acces from a bird,
# Which all seem her charbised herest ignorance
# From that kishes this sound, as your honour.

# ProfanedOH:
# I do haste no deputy?

# ROMEO:
# Reled by your now could save your false to you:
# Learer hope wast not Comtanding and will my son
# To murde his lands are your counterlands
# May sworehe for a doubted pawnicless. Alason, though I again:
# But ha! Farwarrand, gen
#
# 
# this was a transformer!
# Good job we finally finished implementing every bits of a transformer model. Now as we said earlier
# lets repeat all of this in a way that allows us to test each section in a more fine-grained manner.
#%%
# implement self-attention, mltihead-attention, with skipconnection, ffnet, layernorm, positionalencoding, plotting functions
# and test them in action
#
#
#%%
#  lets talk about the models, glue activiation ufnction, efficiancy , chatgpt vs us, 
# document completer vs chatgptetc 
#
#
# a few introductary videos:
# https://www.youtube.com/watch?v=2ih6BHD4v3I
# https://www.youtube.com/watch?v=VoRQiKQcdcI
# https://www.youtube.com/watch?v=3B6q4xnuFUE
# https://www.youtube.com/watch?v=HobIo2oT0xY
# https://www.youtube.com/watch?v=tFYxJZBAbE8


#%% plotting refresher - needs tidying up
# manifold related plots 
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import sklearn
from sklearn import datasets
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD

# plot a manifold in 3d example
# create bunch of random data points
num_points = 500
# we want 3 points for 3 axes.
xs = np.random.uniform(-1, 1, size=num_points)
ys = np.random.uniform(-1, 1, size=num_points)
# we can use sth like sin to map our 2D input to 1D output
zs = np.sin(np.sqrt(xs**2 + ys**2))

# and now the actual plotting 
fig = plt.figure()
# heres the kicker, by setting projection='3d' we get 3d plot!
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xs, ys, zs)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

# example using a dataset with 6 classes
digits = datasets.load_digits(n_class=6)
X = digits.data
y = digits.target
n_samples, n_features = X.shape

# use tsne for dimensionality reduction, instead of pca
tsne = TSNE(n_components=2, init='pca', random_state=0)
X_tsne = tsne.fit_transform(X)

plt.figure(figsize=(6, 5))
colors = "r", "g", "b", "c", "m", "y"
for pos, c, label in zip(range(6), colors, digits.target_names):
    plt.scatter(X_tsne[y == pos, 0], X_tsne[y == pos, 1], c=c, label=label)
plt.legend()
plt.show()

#%% now lets experiment with the sinusoidal positional encoding 
# sine only
def positional_encoding_sine_only(position, d_model):
    angle_rates = 1 / np.power(10_000, (2 * (np.arange(d_model) // 2)) / np.float32(d_model))
    angle_rads = position * angle_rates
    sines = np.sin(angle_rads)
    return sines

# standard sine/cosine pair
def positional_encoding(position, d_model):
    dimensions = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1 / np.power(10_000,(2 * (dimensions // 2)) / np.float32(d_model))
    angle_rads = position * angle_rates
    pos_encoding = np.sin(angle_rads)
    pos_encoding[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return pos_encoding

# Generate the positional encodings
positions = np.arange(1000)[:, np.newaxis]
embd_dim = 512
# pos_encodings = positional_encoding_sine_only(positions, embd_dim)
pos_encodings = positional_encoding(positions, embd_dim)

tsne_2d = TSNE(n_components=2, init='pca', random_state=0)
X_tsne = tsne_2d.fit_transform(pos_encodings)

plt.figure(figsize=(6, 5))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
plt.suptitle('TSNE - 2d projection of Sinusodal Positional Encoding')
plt.show()

# 3d
tsne_3d = TSNE(n_components=3, init='pca', random_state=0)
X_tsne = tsne_3d.fit_transform(pos_encodings)

fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_tsne[:, 0], X_tsne[:, 1], X_tsne[:, 2])
plt.suptitle('TSNE - 3d projection of Sinusodal Positional Encoding')
plt.show()

# PCA
pca = PCA(n_components=3)
X_pca = pca.fit_transform(pos_encodings)

fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2])
plt.suptitle('PCA - 3d projection of Sinusodal Positional Encoding')
plt.show()

# SVD
svd = TruncatedSVD(n_components=3)
X_svd = svd.fit_transform(pos_encodings)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_svd[:, 0], X_svd[:, 1], X_svd[:, 2], c=positions, cmap='viridis', alpha=0.6)

ax.set_title('SVD - 3D Visualization of Sinusoidal Positional Encodings', fontsize=16)
ax.set_xlabel('Component 1', fontsize=12)
ax.set_ylabel('Component 2', fontsize=12)
ax.set_zlabel('Component 3', fontsize=12)
fig.colorbar(scatter, ax=ax, label='Position')
plt.show()
#%%
# Generate the positional encodings
positions = np.arange(1000)[:, np.newaxis]
embd_dim = 512
pos_encodings = positional_encoding(positions, embd_dim)

# Select three dimensions to plot
dim1, dim2, dim3 = 0, 1, 2 # Change these to select different dimensions
# dim1, dim2, dim3 = 100, 101, 102  
# dim1, dim2, dim3 = 10, 200, 499  

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(pos_encodings[:, dim1], pos_encodings[:, dim2], pos_encodings[:, dim3], c=positions, cmap='viridis', alpha=0.6)

# Make the plot more visually appealing
ax.set_title('3D Visualization of Sinusoidal Positional Encodings using 3 raw dims', fontsize=16)
ax.set_xlabel('Dimension {}'.format(dim1), fontsize=12)
ax.set_ylabel('Dimension {}'.format(dim2), fontsize=12)
ax.set_zlabel('Dimension {}'.format(dim3), fontsize=12)
fig.colorbar(scatter, ax=ax, label='Position')
plt.show()

####################### START OF POSITIONAL ENCODING IN DEPTH EXPLANATION #####################################
###############################################################################################################
###############################################################################################################
###############################################################################################################
###############################################################################################################
###############################################################################################################
###############################################################################################################
#%%
# positional encoding information and explanations:
# There some attributes concerning positional encodings that are either desirable or critical to have
# different people developed different intuitions some are more relavent some are less. below I tried to 
# include some of the main attributes I found to be intestring and  or critical from different sources and 
# tried to briefly expand on them and explain them a bit:
# I added my corrections as well, so this is not just purely transcription (although some sections are, 
# usually its clear as the tone of the text changes (I also state the reference if its from somewhere)
#
# Explanation P1: 
# (ref1: https://www.youtube.com/watch?v=1biZfFLPRSY )
# 1.Every position should have the same identifier regardless of the sequence length or what the input is
#  so when the input changes, the position embedding remains the same)
# 
# 2.Since these positinal embeddings push the token/embedding towards a 'specific positional cluster' they should not 
#  be too large otherwise they will push vectors into very distinct subspaces where the positional similarity
#   or dismilarity overshadows/overtunes the semantic similarity.
# 
# lets expand on this a little bit more and make it a bit more intuitive.
# When we plot the token/word embeddings, you'll notice that tokens/words with similar
# context are clustered together, you may find queen and king closer to eachother than they are to car, tire etc.
# and likewise, you'll find car, tire, steering wheel, etc to cluster together.
# This shows, words used together or in a similar context tend to (and usually do) form a cluster around eachother.
# 
# When we add the positional embedding which really is another vector of numbers, what happens is
# that during trainig, the network does the same thing with these new information as well, the queen/king token
# in the feature space, will be ever-so slightly moved to a distinct subregion where their index refers to.
# imagine, king is the first token, so it gets closer to the 'first token's cluster, the network creates other clusters
# for other positions likewise, and as you can see, if the positional embedding is too large, it can disrupt the semantic
# clustering of the tokens.
# This will lead to new clusters based on the position information only (clusters of firsts, seconds, thirds , etc) at
# the expense of destroying the semantic clusteres previously developed due to similar contexts/etc, thus degrading the
# performance altogether.
# as you can imagine, having a too little impact from positional info or lack of any, we will only be left with semantic
# clusters and lose any kind of incentive for the model to use, to make a disntinction between two sequence of the same 
# tokens (becasue there is no solid notion of order) so we need to comeup with something that satisfies these conditions.) 
#
#
# The paper uses something called sinusoidal positional encoding, to solve this issue by providing positional information. 
# but why sinusoidal encoding? what if we tried to use numbers to index the positions ?
# we know we can treat text as a sequence of words coming one after another, so why not count the tokens as
# we go like 1,2,3,4,etc? 
# We simply dont becasue, it violates the second requirement of the positional encoding we mentioned just now.
# The change/shift/translation needs to be small and bounded. why? 
# Think about it for a moment, if the later tokens have larger numbers as position index, then the model assigns more
# importance to them as apposed to the earlier tokens with smaller numbers. The contribution of later tokens will be 
# emphasized while the earlier tokens can be just ignored. all positions must be uniform and evenly paced (fixed delta).
# If the sequence lengths are large it hinders the model performance, on the other hand, as the sequence
# gets longer and longer, larger and larger numbers will be used, which will cause exploding gradients,etc (the majority
# of weight values are close to zero, we want them to be close to zero to have a stable optimization, compare that to decimal
# numbers and see how gigantic they are to them!) and will create a lot of issues in the optimization process. 
# This is why we need them to be bounded and not go to infinity as the sequence gets longer.
# Ok, What about normalizing the numbers based on sequence length ?
# Good point, this solves our issue by making all numbers fall into a range between 0-1, but it creates 
# a new one, the same position, can now take different values based on the sequence length. take for example a sequence
# of 4, if we normalize it, the 3rd position in our sequence of 4, will be (3/4=0.75), but the same position will have a
# different value  if the sequence length was for exmaple 10 (3/10=0.3)! so this will confuse the model! we dont want this! 
#
# The paper's authors opted out to use sin/cos becasue among other reasons, they are both bounded to -1,1 and they
# can offer inifite range of numbers(periodically result in [-1,1] and also using phase-shifting we can control the rate of change)! 
# This means, it can support any sequence length even up to inifity! (as we can have values between -1 and 1 for infinity)
# You might think to yourself, something like sogmoid can be used as well, it is bounded to [0,1] after all and unlike
# normalizing crudly, doesnt depend on the sequence length right?
# Yes but no, the sigmoid function is a saturating function and although it gives us an infinit bounded range, the actual 
# range of usable numbers for us is pretty limited especially for larger numbers(test it with 1-10 e.g. or larger numbers 
# to see why!) unlike the sin/cos that has a lot of variability for large numbers.
# Intrestingly, the sin/cos functions have a problem of their own, that being, they are periodic! that is, the same number
# repeats for different positions! this obvioulsy violates the first requirement of identifiers being unique for each 
# position! we cant encode two different positions with the same number and viceversa. (imagine your nth token gets reset to 1
# it will completely mess up the order we so tireslly trying to embed!)
# The problem stems from their 'frequency'! 
# if we lower the frequency it means it takes longer to repeat a number twice.(the wavelength gets larger)
# The higher the frequency the shorter the range of numbers. In other words, The low-frequency sine wave (sin(x))
# has fewer oscillations within a given range of x, while the high-frequency sine wave (sin(10x) e.g.) has more 
# rapid oscillations within the same range.
# The following snippet shows this:
#
#
# print(plt.style.available)
# ['Solarize_Light2', '_classic_test_patch', '_mpl-gallery',
# '_mpl-gallery-nogrid', 'bmh', 'classic', 'dark_background',
# 'fast', 'fivethirtyeight', 'ggplot', 'grayscale', 
# 'seaborn-v0_8', 'seaborn-v0_8-bright', 'seaborn-v0_8-colorblind',
# 'seaborn-v0_8-dark', 'seaborn-v0_8-dark-palette', 'seaborn-v0_8-darkgrid',
# 'seaborn-v0_8-deep', 'seaborn-v0_8-muted', 'seaborn-v0_8-notebook', 
# 'seaborn-v0_8-paper', 'seaborn-v0_8-pastel', 'seaborn-v0_8-poster', 
# 'seaborn-v0_8-talk', 'seaborn-v0_8-ticks', 'seaborn-v0_8-white', 
# 'seaborn-v0_8-whitegrid', 'tableau-colorblind10']
# 

def plot_sin_cos_frequency(scale=5):
    # lets create 1000 points in the span of (0,4pi), 
    # for refresher see https://www.mathsisfun.com/algebra/trig-sin-cos-tan-graphs.html and 
    # https://www.math.net/sinusoidal
    # we use pi, instead of degrees (0,360) becasue np.sin() uses radian instead of degrees
    x = np.linspace(0, 4*np.pi, 1000)
    # lets do this for sin and cos
    # uncomment this and see how playing with frequency can change the range of numbers available to you
    # freqs = [f(x) for x in [(1/scale)*x, scale*x] for f in [np.sin, np.cos]]
    freqs = [f(x) for x in [x, scale*x] for f in [np.sin, np.cos]]
    plt.style.use('seaborn-v0_8')
    plt.figure(figsize=(8, 4))
    fig, axs  = plt.subplots(2)
    fig.suptitle('Low Frequency vs High Frequency')

    for i,freq in enumerate(freqs):
        label = 'sin-' if i%2==0 else 'cos-'
        label += 'Low Frequency' if i<2 else 'High Frequency'    
        
        axs[i%2].plot(x, freq, label=label)
        axs[i%2].set(xlabel="x-frequency", ylabel="Amplitude")
        axs[i%2].legend(loc='upper right')
    plt.show()
plot_sin_cos_frequency(scale=10)
#%%
# !the concept of delta-distance (a fixed distance between all locations, bywhich the network can move relatively between tokens)
# now if we manage to lower the frequency low enough to the point where it gives us a huge, 
# preferably inifinit numbers before the next period starts, that would be like our first suggetsion,
# linear range of numbers, but bounded!
# but it we dont simply do that? this can provide us with an absolute positional information, which
# should work, but there are two issues, how low should we set the frequency? becasue that affects 
# the generated numbers, if we choose a very low frequency, we might endup with tiny numbers, which
# would make it hard for the network to properly distinuish between tokens/positions, especially the
# adjacent ones. 
# (imagine if we trained our model on frequency x, and generated k positions, and later at test time,
# we wanted to use longer sequences, so we had to use a lower frequency, this would generate different
# set of numbers compared to previous frequency which is not desirable and can mess up our model output
# (becasue they are added to the word/token embeddings thus a change in positional value can result in
# the change in output, therefore from this angle this is also important to be consistent))
# we might even, at the extreme end, face underflow issues, the numbers get so tiny, the float cant 
# represent them properly. 
# these reasons in addition to another intersting attribute that adding a cosine to the mix provides 
# us with, made the main authors to use the sin/cos pair. the sin/cos pair is used extensively in some
# engineering fields such as electrical engineering/signal processing, becasue of their desired attributes.
# for us one of such attributes is they allow the model to have relative positional information in addition
# to the absolute positional information. 
# as we see in a bit later on, sine and cosine waves do different things which comes handy. but lets not
# get ahead of oursevlves. one of the main reasons that sin/cos are mixed, is to make each position as 
# unique as it is possible, and avoide duplicates. we start off with a somewhat higher frequency for the 
# first embeddingd imension,and as we go on, we lower the frequency. this way, we dont repeat a number,
# and the model can distinuish different positions from different wavelengths(high frequency low wavelength,
# low frequency high wavelength). 
#  
# Questions: 
# Now we have a good idea why this works, and its sufficient imho. 
# from this point onward, I try to provide more intuition about different concepts involved.
# these intuions come from different prespectives some may be intuitive and some may be 
# streaching the idea a bit too far so I present them all here and hopefully its for the best.:
# 
# sidenote: ****************************************************************************
# before we continue, I need to add this, we can think of sinusoidal positions as unique identifiers
# that each make each position unique it doesnt have to be a counter to make it intuitive!
# now lets continue.
# 
# lets dive one level deeper and expand a bit more: 
# we are going to gradually explain different points again with new prespectives, so we dont lose
# track of the ideas by creating one giant block of text!
#
# Previously we just talked about adding positional informations as a vector of numbers, like 
# an embedding and add it to the word embedding for each position. 
# but we could also simply add the tokens absolute position to the token embedding as an extra
# dimension. Why dont we do this instead? 
#
# First off if we use a normal decimal number, it might mess up the whole training procedure, 
# if we opt out to learn it, then a single value, may not be descriptive enough and the model
# wouldnt have enough capacity to encode the required positional information into that single number,
# in which case if we increase the embedding size, this would work but two issues pop up here. 
# First how large should we take the positional embedding vector so that we can ensure the positional
# information is not dominated by the token/word information or vice versa? if we use a shorter one,
# the word information may overwhelm the positional information, if we use larger one, it may overwhelem
# the word information. so we use the same size for both of them. but we do that, would over burdened
# us with more overhead, becasue in essence we would be contatenating the positional embedding to 
# the word/token embedding, which just increases the overhead (the overhead increases quadradicly
# in multihead attention just think about it for a moment) 
# and all of this is if we learn it, if we want it precomputed to enjoy the attributes we just stated 
# for sinuidoidal positional embedding, we need to comeup with something to encode
# the positional information in a single embedding cell or several one, which we just went over now with.
# so in practice we use a separate vector the same size as a word/token embedding, and add them together
# to get the benifits we pointed out, like by clustering each position into its own feature subsapce 
# the same way we have this for word embedding which allows us to do king - man = queen! or queen - woman = king! 
# do you get the idea? the idea is to somehow nodge each word ever so slightly to a specific position,
# so that its semantic is preserved, and also the position is also separate from others. for this all 
# dimensions of the embedding need to be taken into account, so thats why the emebdding portion makes sense! 
# now if we take this route, we see that the scale needs to be the same, and all positions need to use the 
# same identifier regardless of sequence length so network can actually differentiate between these positions
# properly.
# from there, we either learn it, or precompute it. the sin/cos is the pre computed one!
# 
#
#=============================================================================================
#=============================================================================================
#=============================================================================================
#=============================================================================================
# Before we continue, lets review some concepts that come handy when we are explaining some of
# the concepts in depth from now on: 
#  
# reminders about concepts we deal with here, if you know, you can skip it but I suggest to 
# skim over it, you may find something you didnt know or has forgot!
#
# Whats an embedding?
# Embeddings can be thought of as a way to represent complex, high-dimensional data in a more simplified, 
# lower-dimensional space while maintaining meaningful relationships between the data points. 
# It's like capturing the essence of something intricate in a simpler form that retains its essence
# or crucial characteristics.
# We can imagine the embedding in different ways. Like for example a map, just as a map condenses
# geographic information into a flat surface without losing the relative positions of countries/places/etc, 
# embeddings condense data without losing crucial relationships. we can also think of it as distilling
# the essence of a painting into a smaller sketch that still captures the main elements and style of
# the original painting.(or summarizing a long story and keeping its key plots and points etc) 
# (we could also view it as somewhat like a library, in which embedding makes sure that books on 
# similar topics are placed nearby, making it easier to find related information, we can go on 
# with more analogies, but you get the idea)
# 
# so an embedding, intuitively, is a condensed representation that retains essential information or 
# relationships while simplifying the complexity of high-dimensional data. It's like summarizing a 
# story without losing its essence or key plot points.
# as for practical examples, we have already seen word embeddings, as words represented as vectors in
# which closer words in the embedding space often have similar meanings or usage contexts.(we say often
# becasue this is not 100% all of the times, I guess it was around 75+% as reported by mikolov in their
# word2vec paper! imnot sure though I need to recheck (!check this))
# or the image embeddings, inwhich the images are represented in a lower-dimensional space where 
# similar images are closer together.
# 
#
# Whats a manifold? (todo: make it shorter and more on point 1 example should suffice imho)
# simply put, manifold is a topological space that looks locally like a Euclidean space, meaning that
# in a small enough region, it resembles a familiar space like a plane. 
# This has some implications for us which we get to rightaway but before that, what is a manifold really?
# its really kind of vague!
# we can get better intuition about manifolds by visualizing them, and there are a lot of ways of doin this.
# we can picture a rubber sheet that can bend and curve. The manifold is like the surface of this sheet,
# able to take on various shapes within the higher-dimensional space.
# or we can imagine this rubber sheet to exist within a 3D space, but the manifold itself is a 2D surface. 
# (the embedding process places data points on this flexible surface.)
# Apart from these, there are other analogies that make this even more intuitive, for example, just 
# as a paper map represents a curved Earth's surface on a flat sheet, a manifold represents complex data
# in a lower-dimensional space or if you think of a landscape with hills and valleys, the trails can
# be seen as the manifold, navigating the terrain while being constrained by the landscape's overall structure.
#
# so to put it simply, a manifold can be seen a flexible, curved surface in a higher-dimensional space,
# on which, each point represents a data point. the goal is to position these points on the surface
# so that the relationships between them are preserved from the original, higher-dimensional data.
# moreover, the curvature of the manifold reflects the relationships between the data points. 
# Smooth curves indicate similar relationships, while abrupt turns may represent significant changes 
# in the data.
# In the context of embedding manifolds, it serves as the reduced-dimensional space where data points
# are positioned after undergoing the embedding process, capturing essential relationships in a more
# manageable form.
# 
# (a manifold retains certain intrinsic/inherent/underlying properties, such as local linearity 
# or smoothness, even if embedded in a higher-dimensional space.)
# 
# Now lets get back to our points, insted of explaning everything in one big block,
# I decided to divide each point and present them in a question/answer form. Here they are:
#
# Q: sin and cos are periodic functions, i.e. they repeat their values, how is this not a
# problem? How is this addressed in here?
# we explained this earlier, the periodicity could pose a problem if we were to use these 
# functions directly as positional encodings without any changes(i.e. alone by themselevs!)
# we encode the positions within the sequence using the combination of sine and cosine functions
# with different frequencies and phases. 
# each pair is treated as a single positional point and each pair uses the same frequency, 
# so position 1 uses 1 frenquency, position 2 uses another, etc.
# this way some waves change quickly, while others change slowly therefore when we combine 
# many waves each with different frequencies, this gives each position a very distinctive/unique pattern.
# the fact that one wave repeats does not mean that all of the other waves repeat at the same time.
# 
# also note that we dont use independently chosen phases for each dimension. for each frequency,
# sine and cosine are used together, they are naturally 90 degrees out of phase (i.e. cos(x) = sin(x + pi/2)) 
# so the core idea here is not different frequencies and different phases alone, but rather, 
# the usage of many frequencies with sine/cosine pair for each frequency, yielding unique identifiers 
# when we combine all of them and thus can use it to represent positions.
# 
# sidenote:
# The frequency determines how quickly the function oscillates, while the phase determines the 
# starting point of the oscillation.
#
# 
# Q: why are sin and cos interleaved/alternated like this whats the intuition or reason behind it? 
# There are several explanations for this online (at the time of writing this in 2022/2023)!
# but not all of them are actually correct.
# I just explained the correct reason, that being, to create uniqueness for each position. 
# 
# Theres another explanation you may also encounter (I did) and it goes like this, 
# the sine function captures the position-dependent changes with a periodic pattern, while 
# the cosine function captures the position-independent changes with a 'constant pattern'.
# this is completely bogus and incorrect. the reasoning/intuition behind this was that 
# interleaving pattern lied in the properties of sine and cosine functions. the sine function
# is an odd function, that is, it is symmetric about the origin, while the cosine function is
# an even function, meaning it is symmetric about the y-axis.
# (remember that a function is considered odd if for any number x, f(-x) = -f(x) and a function
# is considered even if for any number x, f(-x) = f(x). and sin(-x)=-sin(x) while cos(-x)=cos(x))
# thus one captures position-dependent while the other captures position independent changes.
#
# but actually sine and cosine are not doing two fundamentally different jobs such as sine is
# position-dependent and cosine is position-independent. both of them are position-dependent!
# as we said earlier, sine and cosine form a pair that represents the phase of "one" periodic
# signal. for example, for one frequency w, we can look at: [sin(w*pos), cos(w*pos)] 
# 
# sidenote:
# the `w` symbol that we are using here is actually omega and its angular frequency, 
# (w=2Pif which is in radians), since we are using angular frequency, itd be better to use w
# instead of just f, cuz for positional encoding we are actually doing angle_rads=position x angle_rate)
#  
# we can view this pair as a point rotating around the unit circle as the position changes,
# at position 0: [sin(0), cos(0)] = [0, 1] as the position increases, the point moves around 
# the circle. so the pair gives us a compact way of representing where we are within that 
# periodic cycle.
#
# this is one of the main reasons for using both sine and cosine, together they preserve the
# phase of the periodic signal. using only sine would lose some of this information because
# sine takes the same value at multiple points in its cycle.(we can see this in our plot)
#
# moreover, there is also a particularly useful mathematical property here. a shift
# in position corresponds to a rotation of the sine/cosine pair, that is we have:
# sin(w(pos+k)) = sin(w*pos)cos(w*k) + cos(w*pos)sin(w*k)
# cos(w(pos+k)) = cos(w*pos)cos(w*k) - sin(w*pos)sin(w*k)
#
# meaning moving by k positions produces a predictable transformation of the encoding.
# This is one reason sinusoidal positional encodings have useful structure for representing
# relative positions.
#
# so why are sine and cosine interleaved like this?
# we usually write the encoding as [sin(w0*pos), cos(w0*pos), sin(w1*pos), cos(w1*pos), sin(w2*pos), cos(w2*pos), ...]
# where each adjacent pair belongs to the same frequency, dimensions 0,1 -> frequency w0,
# dimensions 2,3 -> frequency w1, dimensions 4,5 -> frequency w2.
# interleaving simply keeps the two coordinates belonging to each frequency together.
# it is a convenient organizational choice it is not what creates uniqueness by itself.
# we could just as well arrange the dimensions as [sin(w0*pos), sin(w1*pos), sin(w2*pos), ...,
# cos(w0*pos), cos(w1*pos), cos(w2*pos), ...] and the fundamental positional information would
# still be there. 
# in other words, the important idea here is when sine and cosine are used as a pair
# it gives us one frequency/one periodic (clock), so with many frequencies well have many clocks
# running at different speeds, and the combined state of all the clocks gives us the distinctive
# positional representation we are after.
#
# we can imagine many clocks where fast clock changes quickly with position a medium clock
# changes at a medium rate and finally a slow clock changes slowly!
# each of these clocks are represented by a sine/cosine pair. the positional encoding records 
# the current state of all of these clocks at once.
#
# any one clock will eventually repeat, because it is periodic but the complete combination of
# many clocks running at different frequencies is much more distinctive for different positions.
#
# so the core intuition is not only we do not avoid periodicity, we deliberately use periodic
# functions because they provide smooth, structured signals. we use lots of frequencies so that
# their combined pattern carries detailed information about the position.
#
# the sine/cosine pair is best understood as two coordinates describing the phase of one
# periodic signal, rather than as "position-dependent" versus "position-independent" functions.
#
# sidenote:
# note that if we swap sine and cosine it would not change the positional encoding fundamentally!
# also interleaving them or grouping all the sine dimensions and cosine dimensions separately 
# would not fundamentally change the representation either. those are mainly choices about
# how the dimensions are organized.(more on this later in plotting section) 
#
#(this is the first question, theres another one I'll explain later which approaches this from 
# another angle)

# why is taking into account the phase or timing of the position within the sequence important?
# first of all, phase is not a separate feature that is explicitly added
# to the encoding, rather the position changes the phase/location of the sinusoidal signal. 
# in our sinusoidal positional encoding we typical have sth like:
# PE(pos, 2i)   = sin(w_i * pos)
# PE(pos, 2i+1) = cos(w_i * pos)
# here sine and cosine are evaluated at different points along their oscillations
# and the position(i.e. pos) determines where we are in the sinusoidal cycle.
# this position also determines the "phase" of the sinusoidal signal.
# now this is useful for us for mainly 3 practical reasons, 
# the first of which is it gives us distinguishable/unique positions, 
# i.e. different positions produce different patterns (of values) across the sinusoidal dimensions.
# second it helps represent relative positional relationships because sine and cosine
# have a useful angle-addition property, where the representation at position p+k can be
# mathematically related to the representation at position p. 
# for example, sin(w * (p+k)) = sin(w*p) * cos(w*k) + cos(w*p) * sin(w*k)
# all of this means that relative offsets between positions can be represented through
# the structure of the sinusoidal encoding.(again meaning that moving by k positions 
# produces a predictable transformation of the encoding)
# third sine and cosine provide complementary coordinates. cosine is a phase-shifted
# version of sine(i.e. cos(x) = sin(x + pi/2)). 
# to reiterate a previously mentioned point, sine and cosine act as the two coordinates of a 
# point rotating around a circle(cos(theta), sin(theta)). as the position changes,
# the point moves around the circle. Therefore, position can be thought of as movement
# through the phase of the oscillation.
# 
# what does it mean when we say that sine and cosine are orthogonal, 
# and why do we care about orthogonality?
#
# mathematically speaking, two vectors are orthogonal when their dot product is zero( a · b = 0)
# this means the two vectors are perpendicular in relation to each other.
# when we say sine and cosine are orthogonal functions, what we actually mean is that
# their inner product is zero (under the appropriate interval, because they are not always orthogonal!)
# for example, over a complete period integral[0 -> 2pi] sin(x) * cos(x) dx = 0) and 
# this has an important implication which is sine and cosine give us distinct directions/components
# in the representation.
# note though, orthogonality does not mean that sine captures one type of information while
# cosine "captures another type of information.
# im saying this because one of the intuitions I encountered online when researching this
# was that,"sine captures periodic patterns while cosine captures constant features" because they 
# are independent. which is incorrect. Both sine and cosine are periodic functions. 
# cosine doest encode "constant features". 
# we already have a much better intuition ie. they are two coordinates describing 
# the "same" oscillation (cos(theta), sin(theta)). cosine gives the horizontal
# coordinate and sine gives the vertical coordinate and together they tell us 
# where we are in the cycle, or equivalently, what the phase is. 
# 
# this is useful for us in encoding positions because changing the position changes the phase. 
# and moving from p to p + delta corresponds to rotating the point by an amount related to delta.
# in fact:
# [ cos(theta + delta) ]   [ cos(delta)  -sin(delta) ] [ cos(theta) ]
# [ sin(theta + delta) ] = [ sin(delta)   cos(delta) ] [ sin(theta) ]
#
# meaning that moving forward in position can be viewed as a rotation in the
# sine/cosine representation.
#
# so to cut a long story short, sine and cosine together provide complementary 
# coordinates that represent the phase of a periodic signal changing position changes that phase.
# also orthogonality is useful because orthogonal components correspond to distinct directions
# (not necessarily distinct (types of) information as we dicussed earlier) in the representation
# space and avoid simply duplicating the same direction of information.
#
# so to recap, positional encoding turns position into phase, as we move through the sequence, 
# we move/rotate through sinusoidal cycles, and using multiple frequencies gives
# the model a rich signature of where we are in the sequence.
# sine and cosine are not two completely different types of information they are complementary
# coordinates of the same oscillation.

# sidenote/reminder 
# what does orthogonality in the context of neural networks mean?
# the definition is not really any different than its traditional mathematical definition, 
# and the fundamental idea remains the same which simply put, is independence and lack 
# of correlation between components.
# when we talk about orthogonality in neural networks, we usually mean weight orthogonality, 
# which refers to the orthogonal relationships between weight vectors in the weight space. 
# that is ensuring that weight vectors are as orthogonal(independent) as possible to 
# each other. when weight vectors are orthogonal, it means they are less likely to duplicate
# or redundantly represent the same information, which can also help us in reducing overfitting!
# furthermore it can help us achieve a more stable and efficient training becasue  when 
# weight vectors are orthogonal, updating one weight vector doesnt strongly influence
# the others, which results in a more independent learning and a more diverse and expressive
# representation of the input data. (cuz each weight vector can capture unique features or 
# aspects of the data without interference from others)
# This will in turn affect the model's ability to better generalize and adapt to
# different patterns in the input data. it also allows the model to learn a wide range 
# of features without being overly constrained by correlations.
# additionally they can help with vanishing/exploding gradients issues during backpropagation
# aswll. for example, consider a typical convolutional network, if the weight vectors 
# corresponding to different convolutional filters are orthogonal, it means that each 
# filter is specialized in capturing a unique aspect of an image(whether it's edges,
# textures or higher level features), which leads to a more robust and generalizable
# representation of the input data. Orthogonal matrices have the property of preserving eigenvalues,
# contributing to numerical stability during training and optimization processes.
# 
# q: how to change frequency for a sin/cos? 
# changing the frequency involves modifying the rate at which these functions
# oscillate or complete cycles within a given interval. 
# The frequency of a sine or cosine function determines how rapidly it repeats 
# its pattern over time.
# Changing Frequency in Sinusoidal Functions:
# The formula for a sin/cos function is f(x)=Asin(Bx+C) and f(x)=Acos(Bx+C) in which:
# (A) represents the amplitude (the peak value of the function).
# (B) corresponds to the frequency, determining how quickly the function oscillates.
# (C) represents the phase shift (a horizontal shift of the function).
# To change the frequency adjust the (B) parameter:
# Increasing (B) will accelerate the oscillation, compressing the function horizontally. 
# This effectively increases the frequency.
# Decreasing (B) will decelerate the oscillation, stretching the function horizontally. 
# This effectively decreases the frequency.
# The frequency and the period are inversely related. 
# Frequency (f) and period (T) are related by the equation (f=1/T ), where (T) represents the
# period (the length of one complete cycle).
# 
# (cosine and sine have the same frequency but with a phase shift of (pi/2) radians or (90').)
# 
# imagine sin(1), sin(1/2), sin(1/100), ..., sin(1/100^2), sin(1/100^3),... 
# The frequency of sin(1/100^n) as n increases is inversely proportional to 
# the period of the function. 
# The period of sin(1/100^n) is 2π/(1/100^n) = 2π100^n. Therefore, the frequency 
# of sin(1/100^n) is 1/(2π100^n).
# As n increases, the frequency of sin(1/100^n) decreases exponentially. 
# This means that the function oscillates more slowly as n increases, 
# and the time between each oscillation increases.
# 
# side note: 
# The 2π in the formula there because sine repeats itself every 2π radians.
# the period of sin(1/100^n) is 2π/(1/100^n) = 2π*100^n 
# ****************************************************************************
#
# =============================================================================================
# =============================================================================================
# =============================================================================================
# =============================================================================================
# =============================================================================================
#
# Section 2 of explanation and my notes 
# From shaw etal 2018: 
# Recurrent neural networks (RNNs) typically compute a hidden state ht, as a function of their
# input at time t and a previous hidden state ht−1, capturing relative and absolute positions along the
# time dimension directly through their sequential structure. 
# Non-recurrent models do not necessarily consider input elements sequentially and may
# hence require explicitly encoding position information to be able to use sequence order.
# One common approach is to use position encodings which are combined with input elements to
# expose position information to the model. These position encodings can be a deterministic func-
# tion of position (Sukhbaatar et al., 2015; Vaswaniet al., 2017) or learned representations. 
# 
# Convolutional neural networks inherently capture relative positions within the kernel size of each 
# convolution. They have been shown to still benefit from position encodings (Gehring et al., 2017), however.
#
# For the Transformer, which employs neither convolution nor recurrence, incorporating explicit
# representations of position information is an especially important consideration since the model is
# otherwise entirely invariant to sequence ordering. Attention-based models have therefore used posi-
# tion encodings or biased attention weights based on distance(Parikh et al., 2016).
# 
# Side note: (also from shaw etal 2018) 
# The Transformer (Vaswani et al., 2017) employs an encoder-decoder structure, consisting of
# stacked encoder and decoder layers. Encoder layers consist of two sublayers: self-attention
# followed by a position-wise feed-forward layer.
# Decoder layers consist of three sublayers: selfattention followed by encoder-decoder attention,
# followed by a position-wise feed-forward layer. It uses residual connections around each of the
# sublayers, followed by layer normalization (Baet al., 2016). The decoder uses masking in its self-
# attention to prevent a given output position from incorporating information about future output po-
# sitions during training.
# Position encodings based on sinusoids of varying frequency are added to encoder and decoder
# input elements prior to the first layer. In contrast to learned, absolute position representations, the
# authors hypothesized that sinusoidal position encodings would help the model to generalize to se-
# quence lengths unseen during training by allowing it to learn to attend also by relative position. This
# property is shared by our relative position representations which, in contrast to absolute position
# representations, are invariant to the total sequence length. Residual connections help propagate position information to higher layers.

# so in short, we need to encode the position information in our attention if we want better result!. 
# 
# good refs for positional embeddings : 
# https://medium.com/@hunter-j-phillips/positional-encoding-7a93db4109e6
# this blogpost does a very good job at explaining the implementation of the sinusoidal positional encoding
# and pretty much explains all the questions concerning the formula and why its implemented a certain way. 
# https://towardsdatascience.com/master-positional-encoding-part-i-63c05d90a0c3
# this blog post,does a good job at explaining the intuitions behind the sinusoidal positional encoding.
# Ive watched and read alot of videos and explanations on this, some videos(also linked below) are good some
# not as much, as they say things that are not backed, or the explanation is superficial. 
# I tried to ask and answer them using different sources I found
# but these two links that I wrote here, do a good job nonetheless. (however, read the following information aswell.)
# finally sinusoidal positional embedding is not used anymore to my knowledge(at least widely as far as im aware), instead the learned 
# positions are used (this is what we implemented in our example, and BERT uses it, but sinusoidal posintioning had
# a lot of intresting intuitions and ideas behind it that can give me/you a new prespective and possibly allow you 
# to learn and comeup with similar improvements knowing the concepts/reasons behind it)
#
# # update 2026:
# unlike back in 2022/2023, modern architectures use RoPE pretty heavily. 
# it's basically a precomputed encoding scheme that gives the model positional
# information by rotating the query/key vectors based on where the tokens are,
# so attention can naturally keep track of relative positions. 
# I also removed many incorrect intuitions and explanations and moved this down, it
# used to be at the top of the explanations concerning positional encoding. 
# below is my cluttered/uncleaned code snippets/experiments with positional encodings
# while I was learning about it. so not everything is correct initially as I was just
# learning things from different sources, some of which didnt have a clue themeselves.
# lucklily, today, there are much better resources, including the LLMs such as ChatGPT, 
# Gemini,Grok,Claude,etc which I myself also used to rectify many of such issues I previously had.
# 
# 
# 
# %%
############################SINUSOIDAL POSITIONAL EMBEDDING IMPLEMENTATIONS####################################
###############################################################################################################
###############################################################################################################
###############################################################################################################
###############################################################################################################
###############################################################################################################
#
# 
#! https://www.youtube.com/watch?v=ZMxVe-HK174&t=289s intresting alternative 
# implementation(might remove it as my own explanation and implemenetations 
# seem to be more intuitive!)
#
# lets implement sinusoidal positional embedding 
# the sinusoidal equation is given in the paper and is as follows: 
#P E(pos,2i) = sin(pos/10000^(2i/dmodel))
#P E(pos,2i+1) = cos(pos/10000^(2i/dmodel))
# basically for each position we interleave sin and cosine functions for all embd entries.
# so it would be sth like this 
import numpy as np 
import matplotlib.pyplot as plt 
def sin_pos_enc_simple(pos, embd_d):
    # return a sinusiodal positional vector for the given position 
    pos_vector = np.zeros(shape=(embd_d))
    for i in range(0,embd_d,2):
        pos_vector[i] = np.sin(pos/10_000 ** (2*i/embd_d))
        # if embd is odd check so we dont go over the last index
        if i+1<embd_d:
            pos_vector[i+1] = np.cos(pos/10_000 ** (2*i/embd_d))
    return pos_vector
# now we can have a positional vector for each position, form 0 to infinity!
# lets plot this for a few positions and see the result
def plot_vec(func, pos_cnt, embd_d,figsize=(6,4)):
    plt.figure(figsize=figsize)
    plt.plot([func(pos, embd_d) for pos in range(pos_cnt)])
    plt.xlabel("Position")
    plt.ylabel("Encoding Value")
    plt.title("Sinusoidal Positional Encoding")
    plt.show()
    
plot_vec(sin_pos_enc_simple,pos_cnt=50, embd_d=512)

#%%
# in practice however, we dont use for loops, so you may see vectorized implementation like this: 
def sin_pos_enc_vectorized(pos, embd_d):
    pos_vec = np.zeros(embd_d)
    # instead of a for loop, we utilize the numpy's array slicing capabilities
    # we first initialize all even entries in pos_vec with sin, and then we do
    # the same for all the odd entries in pos_vec with cosine. 
    # to do this we need a vectorized operation on the right side and it is achieved
    # using np.arange() function.
    # basically, the np.arange(0, embd_d) here, generates an array of numbers from 0 to embd_d-1 
    # and then this array is used in the division and multiplication operations (element-wise).
    # this way it is much faster than using a for loop.
    # note that we have to use step=2 to half the dims so it fits into each half
    # pos_vec[0::2] = np.sin(pos/10_000 ** (2*np.arange(0,embd_d,2)/embd_d))
    # pos_vec[1::2] = np.cos(pos/10_000 ** (2*np.arange(0,embd_d,2)/embd_d))
    # but this means we are using the same dimensions(evens) for all dimensions(evens and odds),
    # we can separat this and use the even dims with sin and the odd ones with the cosine.
    # lets separate that operation into two parts, remove the pos part and create a standalone div_term 
    div_term = 10_000 ** (2*np.arange(0,embd_d)/embd_d)
    # lets make it clear that we only want the even dims for sin
    pos_vec[0::2] = np.sin(pos/div_term[0::2])
    # and the odd ones for cosine
    pos_vec[1::2] = np.cos(pos/div_term[1::2])
    return pos_vec

# in practice, we don't need a for loop here. we can use NumPy's vectorized
# operations and array slicing to compute all dimensions at once.

# and we get the same result
plot_vec(sin_pos_enc_vectorized,pos_cnt=50, embd_d=512)
# in fact theres a slight difference, but its not that significant so in practice 
# we dont really care about the odd/even separation and usually use the even dims 
# for everything!
# so to recap: the exact offset of 1 in the exponent doesnt make a significant difference 
# in the positional encodings, and using the same term for both sine and cosine simplifies
# the implementation so thats why in some implementations people started doing that.

#%%
# However in practice we instead use a more efficient implementation which is as folllows:
def sin_pos_enc_eff(pos, embd_d):
    pos_vec = np.zeros(embd_d)
    # instead of doing power when dealing with floats, which can get problematic 
    # we instead use their equivalent using log() and exp() operations. 
    # we know we can write division as a multiplication operation, so we 
    # can write pos/10000^x as pos * 1/10000^x  (x being 2i/embd_d)
    # we can then write 1/10000^x as 10000^-x becasue we know negative exponentiation
    # is eual to fraction.  
    # then we can write it as : e^log(10000^-x)
    # because exp and log are the inverse of eachothers and e^log(num) is num.
    # its usually done for several reasons including numerical stability which is 
    # what we want here. but why?
    # we know that if a^b = e^(b * log(a))
    # so if we use this we get
    # to write e^(-x * log(10000))
    # which is then simply e^(-2i/embd_d * log(10000) which in turn is :
    # e^(-2i*log(10000)/embd_d)
    # note the minus sign (if you omit it, you have to use division instead of 
    # multiplication with pos!)
    div_term = np.exp(-2*np.arange(0, embd_d) * np.log(10_000)/embd_d)
    # all that remains is to multiply this by pos
    pos_vec[0::2] = np.sin(pos * div_term[0::2])
    pos_vec[1::2] = np.cos(pos * div_term[1::2])
    return pos_vec

plot_vec(sin_pos_enc_eff, 50, 512)
# this plot is the same as the following one!
#%%
# and finally here is an alternative implementation, which uses the 
# same embedding values for all embeddings (even or odd)
# this
import numpy as np
def sin_pos_enc_v2(pos, embd_d):
    pos_vec = np.zeros(embd_d)
    # note that we are using the step=2, and removed the 2! from 2i term as well
    # this is another form of simplification that doesnt drastically change the 
    # positional embedding (except for the fact that without it the output changes
    # more slowly and fewer dimensions towards the end get constant looking values)
    div_term = np.exp(-np.arange(0, embd_d, 2) * (np.log(10_000) / embd_d))
    pos_vec[:, 0::2] = np.sin(pos * div_term)
    pos_vec[:, 1::2] = np.cos(pos * div_term)
    return pos_vec
plot_vec(sin_pos_enc_eff, 50, 512)
#%%
# we can further change this so it can calculate the embeddings for all positions
def sin_pos_enc_all(position_count, embd_dim):
    #pos_vec is a 2d tensor now 
    pos_vec = np.zeros(shape=(position_count, embd_dim))
    # note the 2 behind np.arange() is removed (2i). this is a common simplification 
    # which overall doesnt make much difference (except for the fact that without it
    # the output changes more slowly and fewer dimensions towards the end get constant
    # looking values, we'll see how this looks visually in a moment)
    # we also calculate the exponent only for the even dimensions (hence step=2) and
    # use that for both sine/cosine. this is another simplification thats common.
    div_term = np.exp(-np.arange(0, embd_dim,2) * np.log(10_000)/embd_dim)
    # now for all positions we need to create an array like we did for embd dims
    # since we need to do an elementwise multiplication with div_term which is 
    # an array of (embd//2), our final output should be (pos_max, embd//2)
    # they are not compatible, so we add a new dim to positions
    # so when they multiply it becomes (max_pos,1) * (1,embd//2) then get broadcasted
    # into (maxpos,embd//2) and then the calculation is carried out.
    # ((embd//2) is the same as (1,embd//2) so it doesnt need any changes and all
    # should work now!)
    positions = np.arange(0, position_count)[:,None]
    # calculate the positions for all positions all atonce
    pos_vec[:,0::2] = np.sin(positions * div_term)
    pos_vec[:,1::2] = np.cos(positions * div_term)
    return pos_vec

plt.figure(figsize=(15, 5))
y = sin_pos_enc_all(position_count=100, embd_dim=20)
# lets plot 4 embd values for 100 positions, 
# (we used 4:8 becasue they demonstrate pretty graphs! 
# use other numbers and see the outcome) 
dims = (4,8)
plt.plot(range(100), y[0:100, slice(*dims)])
# plt.plot(np.arange(100), y[:100, 8:12])
plt.legend(["dim %d"%p for p in range(*dims)])
# %%
# side note: 
# given the equations:
# P E(pos,2i) = sin(pos/10000^(2i/dmodel))
# P E(pos,2i+1) = cos(pos/10000^(2i/dmodel))
# 
# we see that as pos increases, the argument of the sine function increases as well. 
# This results in the output of the sine function cycling through its range from -1 to 1.
# However, because of the denominator 10000^2i/dmodel​, the rate at which the output cycles,
# decreases as i increases. This means that for larger i, the output of the function changes
# more slowly as pos increases.
# to be more specific:
# the denominator term(10000^2i/dmodel)​ effectively determines the “wavelength” of the sine 
# function. As i increases, the denominator 10000^2i/dmodel​ increases, which means the argument
# of the sine function increases more slowly. 
# This corresponds to an increase in the wavelength of the sine function.
# So, for larger i, the "wavelength" (or the distance between successive peaks or troughs) increases.
# This means the function changes more slowly as pos increases, allowing the model to capture 
# longer-term dependencies between words in a sentence. 
# Conversely, for smaller i, the "wavelength" is shorter, and the function changes more quickly 
# with increasing pos, allowing the model to capture shorter-term dependencies.
# This combination of different wavelengths at different dimensions helps the model capture 
# complex patterns in the positional relationships between words.
# now lets build better intuitions by visually seeing what we just described here:
from pprint import pprint
import random
import numpy as np
import matplotlib.pyplot as plt

# lets draw a heatmap/pseudocolor plot of our positional encodings and see how they look and behave
# visually: 
def get_sinusoidal_positional_encoding(position_count, embd_dim):
    assert embd_dim%2==0, "this needs to be an even number, otherwise odd/even count wont match! and we'll face an error"
    pos_vec = np.zeros((position_count, embd_dim))
    # use exp instead of the paper's implementation so its numerically more stable 
    # note that we are using the simplified version of the equation (even dims without the '2' scaler!)
    div_term = np.exp(-np.arange(0, embd_dim,2) * (np.log(10000) / embd_dim))
    positions = np.arange(0, position_count)[:, np.newaxis]
    pos_vec[:, 0::2] = np.sin(positions * div_term)
    pos_vec[:, 1::2] = np.cos(positions * div_term)
    return pos_vec

def plot_positional_encoding(positional_encoding):
    plt.figure(figsize=(128, 64))
    # concerning colormaps read this first : https://matplotlib.org/stable/users/explain/colors/colormaps.html#colormaps 
    # https://matplotlib.org/stable/gallery/color/colormap_reference.html
    # initially I used viridis, but later chose to use RdBu instead (redblue) becasue it was more coherent imho
    # but for colorimpered,virdis is the way to go so I leave my previous explanation here:
    # side note for why we chose viridis : https://sjmgarnier.github.io/viridis/articles/intro-to-viridis.html 
    # what other colormaps we have? simply check plt.colormaps() to see your other options
    # uncomment the following line instead of the next line and see the effect of different colormaps.
    # of course not all colormaps suit all usecases, read the first link if you havent. 
    # basically viridis belongs to a so called 'Perceptually Uniform Sequential' colormap group. 'magma', 'inferno', 
    # 'plasma', 'cividis' and turbo are other examples of what we call a preceptually uniform sequential colormap.
    # Perceptually uniform, means values close to each other have similar-appearing colors and values
    # far away from each other have more different-appearing colors, consistently across the range of values.
    # and sequential simply refers to the fact that the lightness value increases monotonically through the colormap.
    # we have other types such as Diverging, Cyclic and Qualitative, which each have their own specific usecases
    # for example Qualitive colormaps which are usually miscellaneous colors, are used to represent information
    # that does not have ordering or relationships. 
    # The Cyclic colormaps on the otherhand as the name suggest, refer to change in lightness of two different colors that 
    # meet in the middle and beginning/end at an unsaturated color; 
    # and are used for values that wrap around at the endpoints, such as phase angle, wind direction, or time of day.
    # The Diverging ones, refer to change in lightness and possibly saturation of two different colors that meet in the 
    # middle at an unsaturated color. 
    # They are used when the information being plotted has a critical middle value, such as topography or when the data 
    # deviates around zero. 
    # cmap = random.choice(plt.colormaps())
    # As for the ‘viridis’ colormap, it is a perceptually uniform colormap that is designed to be bright, 
    # attractive, and colorblind-friendly. It provides a smooth, monotonically increasing color range that 
    # significantly improves the readability of data visualizations. The viridis scales provide color maps 
    # that are perceptually uniform in both color and black-and-white. 
    # They are also designed to be perceived by viewers with common forms of color blindness 
    # or  maybe the rdbu is better!
    cmap = 'RdBu'
    # play with the values and see how as we near the end of embd, the value seem to become constant!
    # and shows the relationship of pos with our denominator which as i increases the output changes more slowly
    # and as pos increases with increasing i(dim) output changes evern more slowly to the point they all
    # look like constant. 
    # also note that  as we increase the i, the periods of the function also increases so when i reaches 
    # the value of d, a large number of pos vectors are needed to cover the entire period of the functions.
    # (explained in the latter plots in a moment)
    # use :10, :100, :200, then 100:200, 150:200, etc for embddiing dimension
    # plt.pcolormesh(positional_encoding[:,:200], cmap=cmap)
    # for cmap in plt.colormaps():
    plt.pcolormesh(positional_encoding[:,:], cmap=cmap)
    plt.xlabel('Embedding Dimensions')
    plt.ylabel('Position')
    # lets add a colorbar show the mapping of colors-to-values in the heatmap.
    plt.colorbar(label=f'Value({cmap})')
    plt.title('Sinusoidal Positional Encoding')
    plt.show()

# now lets plot this first with a few positions/embeddings and then much larger numbers
# in both cases we should see the effect of pos/embd as they increase.
def draw_postion_vector_heatmap(position_count, embd_dim, show_position_vec=False):
    pos_vec = get_sinusoidal_positional_encoding(position_count, embd_dim)
    if show_position_vec:
        print(f'{pos_vec=}')
    plot_positional_encoding(pos_vec)

# test with a small number of positions and embeddings 
# this shows us how distinct each position is
draw_postion_vector_heatmap(position_count=5, embd_dim=6)
# a bit larger
draw_postion_vector_heatmap(position_count=20, embd_dim=30)
#  and now lets see larger pos/embd_size
draw_postion_vector_heatmap(position_count=1000, embd_dim=512)
# lets start with 50x more position to fill as much encoding space as we can
draw_postion_vector_heatmap(position_count=50_000, embd_dim=512)
# As we have just seen, the position vector has shorter wavelengths for lower dimensions, 
# and longer for higher dimensions. as we increase the i, the periods of the function also
# increases so when when i reaches the value of d, a large number of pos vectors are needed
# to cover the entire period of the function, this can be seen in the two plots we have here.
# 
# !The values of the early positions at higher indexes are almost constant. take the first position
# in the first plot, and the first 5-10 positions in the second plot for example.  
# !This can be observed in the first two plot especially in the second plot better, where the colors
# of columns 15-30 hardly change(its barely visible).as the number of positions increases, this effect diminesh
# 


# Recap about what we can understand from these plots: 
# so lets expand on this a bit more: 

# wavelength pattern: 
# We can see clear wave patterns in the plot, which reflects the sinusoidal nature of the encoding. 
# These waves indicate how different positions along the sequence are represented in the embedding space.
#
# Frequency Variation: 
# we saw that the frequency of the waves varies across different dimensions of the embedding. 
# The lower dimensions may capture shorter-range dependencies, while
# higher dimensions may focus on longer-range dependencies.( more explaination ahead)
#
# !Alternating Colors: 
# the alternating color bands we see in the heatmap is caused by use of sine and cosine functions.
# This alternation ensures that the model can distinguish between adjacent positions and each position is
# uniquly indentifiable.
# 
# Positional Diversity: 
# as we just pointed out, the heatmap illustrates this fact by showing how each position in 
# the sequence has a unique representation in the embedding space. This is crucial for the 
# model to distinguish between tokens based on their absolute or relative positions.

# !looking athe plot we see alot of blue/white strips towards the right end of the plot and much
# less other colors, they seem constant values being repeated.
# these blue/white stripes represent the values of the positional vectors and the reason 
# we see fewer changes (less red/white/bule stripes (when we use RdBu cmap)) towards the end of the plot is due to
# the nature of the positional encoding scheme. 
# As we move towards higher dimensions, the frequency of these functions decreases more,
# leading to fewer changes in the values and hence fewer stripes in the plot.
# 
# by the way note that the stripes at the far end of the embedding dimensions do not represent
# a single value they are many tiny numbers that are simply too small to make a significant difference
# (and they are very close to each other, after all they change a tiny bit each time), 
# and hence they are shown as blue/white for all positions(they are very similar in value so their
# color ends up indistinguishable for us/looks the same to us).
# 
# when increasing the position count, we can see for the same number of embeddings, the plot changes
# in a way that the number of stripes/ seemingly constant values to the far end of the embeddings decreases
# !The difference between the 1k plot and the 50,000 positions plot could be due to the difference in
# the total number of positions encoded in each plot. A plot with more positions (like the 50,000 positions
# plot) would naturally have more stripes as it represents more positional information.
# The key point to understand from visualizing these positional vectors is how positional information 
# is encoded in transformer models. It helps us see that the positional encoding scheme can capture the 
# order of data points in a sequence, which is crucial for tasks like natural language processing where 
# the order of words in a sentence carries important semantic information. 
# The plot also shows how this positional information varies across different dimensions, providing 
# insights into the workings of high-dimensional data in machine learning models.
#
# Note that the frequency is actually decreasing in our equation as i increases. 
# This is because the div_term is an exponential decay term, where the base of the exponent is 
# less than 1. (-np.exp(np.arange(0, embd_size, 2)) * (np.log(10_000.0) / embd_size)). 
# This means that as you move along the embedding size, the frequency of the sine and cosine terms
# in the positional encoding decreases. 
# this allows the model to capture both short-term and long-term dependencies in the input sequence. 
# The sine and cosine functions provide a way to encode the position with a unique representation 
# that can capture relative positions and is invariant to the sequence length. 
# The decreasing frequency ensures that the model can distinguish positions across a wide range of 
# sequence lengths.
# 
# reminder about relationship with wavelength:
# In the context of waves, frequency and wavelength are inversely related. 
# As the frequency of a wave increases, the wavelength decreases, and vice versa. 
# This relationship is governed by the equation:
# v=fλ
# where:
# (v) is the speed of the wave,
# (f) is the frequency, and
# (λ(lambda)) is the wavelength.
# In the positional encoding scheme used in the Transformer model, 
# the decreasing frequency can be thought of as an increasing "wavelength" along the dimensions of 
# the positional encoding vector. This means that the positional information encoded by higher 
# dimensions changes more slowly (longer "wavelength"), allowing the model to capture longer-term
# dependencies in the data. 
# Conversely, the positional information encoded by lower dimensions changes more quickly 
# (shorter "wavelength"), enabling the model to capture shorter-term dependencies. 
# This balance allows the model to understand both the local and global structure of the sequence.
# 
# More explanation: 
# Here, the concept of "longer waveform" is analogous to the slower changing positional encoding values
# in higher dimensions. 
# The positional encoding in Transformer models uses a mix of sine and cosine functions with different
# frequencies. The frequency of these functions decreases (or the "wavelength" increases) as you move 
# to higher dimensions in the positional encoding vector. 
# This means that for lower dimensions, the positional encoding values change rapidly (short "wavelength"), 
# allowing the model to capture changes and patterns that occur over short distances in the sequence (short-term 
# dependencies). 
# On the other hand, in higher dimensions, the positional encoding values change more slowly (long "wavelength").
# This allows the model to capture patterns and dependencies that occur over longer distances in the sequence 
# (long-term dependencies). 
# For example, in a sentence, a word might be influenced not just by the word next to it, but also by a word 
# much further away. The slower changing positional encodings in the higher dimensions allow the model to 
# capture these longer-term dependencies.
# So, the "longer waveform" (or slower changing positional encoding values) helps the model to understand the 
# broader context in the sequence, while the "shorter waveform" (or rapidly changing positional encoding values)
# helps the model to understand the local structure of the sequence. This balance is crucial for the model's 
# performance on tasks like language translation, where understanding both the local syntax and the broader 
# semantic context is important.
#

#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Now let us get some intuitions by looking at these positional embeddings from another angle.
# lets try to compare each embedding vector against others and see what we get and whether we can interpret 
# them or not  (use this between explanations)
# 
def get_positional_encoding(position_count, embd_size):
    positional_encoding = np.zeros((position_count, embd_size))
    div_term = np.exp(-np.arange(0, embd_size, 2) * (np.log(10000.0) / embd_size))
    pos = np.arange(position_count)[:, np.newaxis]
    positional_encoding[:, 0::2] = np.sin(pos * div_term)
    positional_encoding[:, 1::2] = np.cos(pos * div_term)
    return positional_encoding

#! this may not be what I want!
#! This function calculates the Euclidean distance between the positional encoding vectors of 
# neighboring time-steps and plots these distances.
# The plot will show that the distances between neighboring time-steps decrease as we move along
# the time axis, illustrating the decay of positional information over time in the sinusoidal positional
# encoding scheme.
def plot_positional_encoding_distances(positional_encoding):
    distances = np.square(positional_encoding[0:-1] - positional_encoding[1:])
    plt.plot(distances[:])
    plt.ylabel('Distance')
    plt.xlabel('Time-step/embd dim')
    plt.title('Distance between neighboring time-steps in positional encoding')
    plt.show()

def plot_positional_encoding_total_distances(positional_encoding):
    distances = np.sum(np.square(positional_encoding[0:-1] - positional_encoding[1:]),axis=-1)
    plt.plot(distances[:])
    plt.ylabel('Distance')
    plt.xlabel('Time-step/embd dim')
    plt.title('Distance between neighboring time-steps in positional encoding')
    plt.show()

# now lets see how each position fairs as we get closer to the end of the emebdding dims
def plot_positional_encoding_distance_total_2d(positional_encoding):
    # we use square to accentuate the differences
    distances = np.sum(np.square(positional_encoding[:, np.newaxis] - positional_encoding[np.newaxis, :]), axis=1)
    plt.plot(distances)
    plt.ylabel('Position')
    plt.xlabel('Position')
    plt.title('Heatmap of distances between positional encoding vectors')
    plt.show()

from mpl_toolkits.mplot3d import Axes3D
# this should give us a better view, when viewed in 3d, as we can see, the earlier dimensions are much active
# but as we get closer to the end dimensions, the distance between dims gets close to zero!
def plot_positional_encoding_distance_total_3d(positional_encoding, elev=10, azim=40, func='square'):
    if func == 'square':
        func = np.square
    elif func== 'abs':
        func = np.abs
    elif func == None:
        func = lambda x: x 
        
    distances = np.sum(func(positional_encoding[:, np.newaxis] - positional_encoding[np.newaxis, :]), axis=1)
    fig = plt.figure(figsize=(24,18))
    ax = fig.add_subplot(111, projection='3d')
    x = np.arange(distances.shape[0])
    y = np.arange(distances.shape[1])
    X, Y = np.meshgrid(x, y)
    Z = distances[X, Y]
    ax.plot_surface(X, Y, Z)
    ax.set_xlabel('Position')
    ax.set_ylabel('Embeddings')
    ax.set_zlabel('Distance')
    ax.set_title('3D plot of distances between positional encoding vectors')
    # Change the viewing angle
    # elev sets the elevation angle in the z plane. 
    # azim sets the azimuth angle in the x,y plane.
    ax.view_init(elev=elev, azim=azim)  
    plt.show()
#! intresting plot! but doesnt give us much information! its just pretty!    
plot_positional_encoding_distances(get_positional_encoding(1000, 512))
# intresting as well, but not much useful, maybe used with the next plot gives it a merit! it
#! shows a pretty plot with large dims smaller dims show entangled sins which doesnt give us anything really
plot_positional_encoding_distance_total_2d(get_positional_encoding(100, 500))
plot_positional_encoding_distance_total_3d(get_positional_encoding(100, 50))
plot_positional_encoding_distance_total_3d(get_positional_encoding(100, 50),azim=10)

# we can use no functions on the differences and simply visualize the raw differences  
plot_positional_encoding_distance_total_3d(get_positional_encoding(100, 50), func=None)
plot_positional_encoding_distance_total_3d(get_positional_encoding(100, 50),azim=10, func=None)
#%%
# now if we try to display this as a heatmap, we will get a much more intersting result: 
def plot_positional_encoding_heatmap(positional_encoding):
    # lets calculate the eucleadian distance
    distances = np.sum(np.square(positional_encoding[:, np.newaxis] - positional_encoding[np.newaxis, :]), axis=2)
    # distances = np.sum(positional_encoding[:, np.newaxis] - positional_encoding[np.newaxis, :], axis=2)
    sns.heatmap(distances,cmap='Blues')
    plt.ylabel('Position')
    plt.xlabel('Position')
    plt.title('Heatmap of distances between positional encoding vectors')
    plt.show()

def plot_positional_encoding_dot_product_heatmap(positional_encoding):
    """this function calculates the dot product between all pairs of 
    positional encoding vectors and plots these dot products as a heatmap.
    The heatmap will show the dot product between positional encoding 
    vectors at different positions in the sequence. 
    
    The diagonal line in the heatmap represents the dot product of a 
    position with itself, which is the maximum possible value. 
    The symmetry of the heatmap reflects the fact that the dot product 
    from position i to position j is the same as the dot product from 
    position j to position i. 
    
    Args:
        positional_encoding (_type_): _description_
    """
    dot_product = np.dot(positional_encoding, positional_encoding.T)
    sns.heatmap(dot_product, cmap='Blues')
    plt.ylabel('Position')
    plt.xlabel('Position')
    plt.title('Heatmap of dot product between all pairs of time-steps in positional encoding')
    plt.show()

position_count = 512
# smaller dims shows the shades much better than a larger dim such as 512
embd_dim = 100
positional_encoding = get_positional_encoding(position_count, embd_dim)
# the information is given below (explanation part)
plot_positional_encoding_heatmap(positional_encoding)
# showing that the distance between neighboring time-steps are symmetrical and decays nicely with time.
# that is, the diagnol axis has the highest score, which really says, each token/position has the highest
# relationship with itself, as we get farther away, we see the blue turns to white slowly, showing the relation
# ship between nearer position is stronger than those far away, and the shades show that this gradually and symetrically
# decreases. try sin/cos only and see why we use both of them together!
plot_positional_encoding_dot_product_heatmap(positional_encoding)

# Generate x values
x = np.linspace(0, 4 * np.pi, 100)
# Generate intermediary variable for frequency transition
freq = np.linspace(5, 5.5, len(x))
# Generate sine waves with varying frequency
sine_waves = np.sin(freq * x[:, None])
# Plotting
plt.figure(figsize=(16, 8))
for i in range(len(x)):
    plt.plot(x, sine_waves[:, i], color='blue', alpha=0.6)
plt.xlabel('x')
plt.ylabel('Amplitude')
plt.title('Transition from Low Frequency to High Frequency - Sine Waves')
plt.show()
#%%
#
#=============================================================================================
#%%
# Side note/reminder -Frequency 
# 
# The frequency of a sine or cosine wave is determined by the rate at which it oscillates or completes cycles over time.
# In mathematical terms, frequency (f) is the number of cycles per unit of time, usually measured in Hertz (Hz). 
# The relationship between frequency, angular frequency, and the period of a wave is given by the equation:
# [ f = 1/T ]
# Where:
#     ( f ) is the frequency,
#     ( T ) is the period of the wave (the time it takes to complete one cycle),
#     ( ω(omega) ) (angular frequency) is related to ( f ) by ( ω = 2pi*f ).
# For a sine or cosine wave, the general form is:
# [ y(t) = A * sin(2pi*f*t + phi) ]
# Where:
#     ( A ) is the amplitude,
#     ( f ) is the frequency,
#     ( t ) is time,
#     ( phi ) is the phase.
# Now, to increase the frequency of a sine or cosine wave, you can do one of the following:
#     Increase the angular frequency ( ω ):
#         ( ω = 2pi f )
#         By increasing ( ω ), you effectively increase the rate at which the sine or cosine function oscillates.
#     Decrease the period ( T ):
#         The period ( T ) is the reciprocal of frequency, so by decreasing the period, you increase the frequency.
#         ( T = 1/f )
# Here's a more detailed explanation:
#     Angular Frequency ( omega ):
#         The angular frequency ( ω ) represents how quickly the wave oscillates in radians per unit of time.
#         If you increase ( omega ), the wave will complete more cycles in the same amount of time.
#         This is often more intuitive when thinking about the periodicity of the wave in terms of its angular 
#         measure (radians) rather than cycles.
#     Period ( T ):
#         The period ( T ) is the time it takes for one complete cycle of the wave.
#         If you decrease the period, the wave completes cycles more quickly, effectively increasing the frequency.
# In summary, to increase the frequency of a sine or cosine wave, you can either increase the angular frequency ( ω )
# or decrease the period ( T ). These changes will result in a wave that oscillates more rapidly over time.

import numpy as np
import matplotlib.pyplot as plt
#! this plot shows the frequency changes better than my previous one that shows sin/cos together with high/low freq
# use this instead of that
def plot_sine_wave(amplitude, frequency):
    # Generate time values from 0 to 2*pi with small intervals
    time = np.linspace(0, 2 * np.pi, 1000)
    
    # Calculate the sine values for each time point
    sine_wave = amplitude * np.sin(2 * np.pi * frequency * time)
    
    # Plot the sine wave
    plt.figure(figsize=(8, 4))
    plt.plot(time, sine_wave, label=f'Sine Wave\nAmplitude: {amplitude}, Frequency: {frequency} Hz')
    
    # Add labels and title
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Sine Wave - Amplitude and Frequency')
    
    # Add a legend
    plt.legend()
    
    # Show the plot
    plt.grid(True)
    plt.show()

# Set the amplitude and frequency values
amplitude = 1.0
frequency = 1.0  # in Hertz

# Call the function to plot the sine wave
plot_sine_wave(amplitude, frequency)
#%% 
################################### my initial notes/refs/unedited######################################
# the previous notes have been edited a little bit, but this section not. the reason you see one concept
# being repeatedly explained/noted is I tried several times using different prespectives/lectures/blogs/etc
# to explain this to myself, but havent really had the time to consolidate them properly. Itried to provide
# a somewhat more coherent path of reading what I have found/learned, if you read from top down, you'll get
# more information, and should be able to read all of these as well and spot where I made a mistake and not.
# 
#
# ----------------------------------
# !all entries interact with all other entries at the same time, and the order of words/tokens is lost
# 'how are you' would be the same as 'are how you', 'you how are', 'you are how', 'how you are', etc
# since we are using this in a autoregressive model, given a token, we want the next one, its obvious
# that we want the order of words/tokens to be preserved, so we get the output that makes sense!, otherwise,
# this would create the same output for all of these input, which is not what we want!.(theres nothing in
# the input to specify the order, that is we treat it as if theres none!). 
# in other usecases, such as translation from one language to another, we dont have this issue, as we want to grab
# one paragraph in one language completely and then produce a translation later, the first part doesnt
# need this as we arent after producing the next word/token, but the second part does, we talk about this in more
# details later, but for now, we know we want to preserve the order as well. so what do we do?  
# (youtube channel link below)
# there are several ways we can add this positional information, each with its own set of merits and drawbacks
# for one, we can add a number for each token, like 0 for how, 1 for are, and 2 for you, and so on.
# but this has several issues, and doesnt work as expected, becasue, suppose we have large number of tokens
# doing so can create a bias for later tokens, it puts more value/attention on tokens with larger numbers
# also if the length of a sentences is increased, we get different value, for the positions, 
# moreover, if an input has two sentences, this way of positional embedding makes the model think, these 
# sentences always come together, because thier positions indicate so (after all they are numbered consecutively!)
# we might think, normalizing these values to be 1 would solve the issue, but it creates another issue which is
# if the length of the sentence is different, we would get a different value for each same token. (imagine 
# one sentence was 'how are you' and the other 'how are you bob' they are nearly the same except the last word
# but since the length is now different, each word, would get a different value becasue we normalzied them on
# the sequence length! p3 = 3/3 = 1 vs p3 =3/4=0.75) so we dont want any dependency on the length of the sequence.
# so what should we do? the values shouldnt be too large so they create a bias towards later tokens!
# what we know that each value must be unique and it must not depend on the length of the sequence!
# and finally this value must be within a range! and vary nonlinearily. one option is to use sinocidal embedding
# and this is infact what the original paper came up with, (a hand crafted positional embedding), with an 
# embedding vector of length d, they used
# they used sin, but simply using sin function isnt enought as it violates our first point of uniqueness,
# sin(0) is always zero, we dont want our 0th token to always be 0! so sin(pos) is not enough. 
# the paper uses cos as well, and alternates between them, sin(pos) followd by cos(pos+1) this way
# the even positions are handled by sin and the odd positions in our embedding vector are handled by cos
# becaue the frequency of sin and cos changes, therefore each vector gets a different value, thats how we
# satisfy our first point of uniqueness for each position regardless of sequence length. 
#  
# absolute positional embedding:
# relative positional embedding:
# frequency based positional embedding - sinosodal positioning 
# rotary positional embedding -rope
# concatenated vs added positional embedding 
# learned positional embedding
#
# ------------------------------------
# The following notes were taken from The Stanford XCS224U: NLU I Contextual Word Representations, Part 3: 
# Positional Encoding I Spring 2023
#
# --The role of positional encoding: 
# transformers/attention mechanism has a very limited capacity to keep track of word order
# the attention connections are not directional, they are just bunch of dot products and 
# there are no interactions between the columns, 
# and becasue of this we need to ensure theres a difference between sequence A,B,C  and C,B,A (i.e. 
# its is different thatn C,A,B, or C,B,A ,etc cuz we are not keeping track of token order in attention)
# positional encoding therefore, makes sure that these sequences are different, regardless of what we do with 
# the representattions that come out of the model. theres another role they fill in, they have been used to
# to keep track of the hierarchial notions of position (premise/hypothesis in natural language inference which is
# one of the important features of BERT model we later on will discuss)
# 
# --Evaluating positional encoding schemes: 
# There are a lot of prespective we can take on postional encoding, but two major questions that can be asked
# are : 
# 1. does the set of positions need to be decided ahead of time? 
# 2. does the positional encoding scheme hinder generalization to new positions?
# as we know, models tend to impose a max length on the sequences they can process, for reasons relating to
# their learned weights(training, optimization, etc). we will ask whether different positional embedding 
# schemes are imposing anything about length generalization separate from this. 
# so we are asking if we set this fact aside for a moment, does the positional encoding scheme itself, 
# is imposing anytihng about sequence length generalization?
# 
# so lets start with absolute positional encoding 
# in this scheme, (we can have different ways of implementing it but this is one of them) we have a separate position
# embedding(which we learn!) alongsize our token embedding which we add together. this scheme obviouly suffers
# from the fixed sequence length issue, as we need to, ahead of time, decide on the length of the embedding vector
# and if for example we decided on sth like embd=512, we cant use larger sequences (we dont have position inofrmation
# after 512, we wouldnot have positional representation for those positions) also note that the emebddings here 
# will be different, for example consider the phrase, 'the rock' at the begining of a sentence which would be sth like this: 
# input: 'the rock was big' we would have sth like : embd(the)+em_pos(the), embd(rock)+em_pos(rock),...
# will be diferent than if ' the rock' came at a diferent position, the representation for the same tokens will be different!
# 
# so to recap: 
# the limitations we face in this scheme are: 
# 1.set of position needs to be decided ahead of time
# 2. may hinder generalization to new positions, even for similar phanamena ([the]+[1] [rock]+[2] 
# has a different representation than [the]+[15] [rock]+[16]) 
# there will be some similarity between them as we have the same wordvectors involved(for 'the' and 'rock')
# but since we add position embedding, the result will be very heavy handed, when it comes to learning representation
# that are heavily position dependent, and this could make it harder for the model to see 'Rock' forexample
# is the same phrase whether its used in the begining of the sequence or middle or end of it.
# 
# --Frequency based positional encoding scheme: 
# another scheme we can use is the frequency based positional encoding scheme, which there are a lot of ways
# for setting this up! but the essential idea is that we define a mathamatical function that given a position
# will give us back a vector that encodes information about that position, semantically and in structure. 
# this is infact what the attention is all you need paper, opted to use and presents in the paper. 
# basically they use the fe frequency oscillation of sin and cos functions for this. 
# basically higher frequency oscilate more frequently and they use that information in the position vector
# that we create. the following codesnippet shows how their encoding function looks like
#%%
import matplotlib.pyplot as plt 
import numpy as np 
def pos_enc(pos, embd_size):
    div_term = np.exp(-np.arange(0, embd_size, 2 )) * (np.log(10_000.0)/embd_size)
    rep = np.zeros(embd_size)
    rep[0::2] = np.sin(pos * div_term)
    rep[1::2] = np.cos(pos * div_term)
    return rep
plt.figure(figsize=(8,6))
embd_size = 100 
pos = 50
pos_vec = pos_enc(pos,embd_size)
plt.plot(pos_vec)

# the good thing about this function is, if you give it pos =1 it will give us back a vector, if we 
# give pos=1000, it will give us back a vector, if we give pos=1000_000 it will give us back the vector
# you get the idea, we are no more bound to the embedding length. and all of those vectors, manifestly do
# is to encode information about relative position of that input. so we have definitely overcome the first
# limitation (i.e. the 'set of positions need to be decided ahead of time' is overcome now!)
# so we can fire up a vector for any position given to us.
# the second question/limitation however still exists and remains pressing! like before this scheme can 
# hinder generalization to new positions even for familiar phenamona, in virtue of the fact that we are
# taking those word representations and adding them with positional vectors. as we explained before, this
# makes it harder for te model to see that the same phrase can occure at different positions,
# 
# -- The relative positional encoding (shaw etal 2018- self attention with relative position represenation)
# https://arxiv.org/pdf/1803.02155.pdf
# https://www.youtube.com/watch?v=DwaBQbqh5aE
# this scheme may be the most promissing one, as the position information is added to the input inside attention
# module as apposed to the embeddings themselevs. more importantly, the concept of a window is used here
# that actually encodes this information. basically using a sliding window of size d, we use a fixed number
# of weights on all input to convey relative positions for each token. for example, if we set window-size 
# to d=2, and have an input like the following :
#  1    2    3    4   5   6
# 'the rock fell from the sky' 
# where the numbers signify each token's position and the line after, shows our sequence.
# now in an absolute positional encoding scheme, each token 'emebedding' would have its own 'position embedding
# which are added together forming the final embedding with the positional information. 
# however, for a relative positional encoding case, the relative distane between (adjacent) tokens are taken into
# account. the extend to which this relative distance is calculated is determinted using a window size k.
# this in effect, creates 'k' weights that are used to encode the relative positions of the tokens in a sequence.
# suppose we have a window-size of k=2, in the given example, the weights would be used like this
# suppose we want to calculate relative distance/position for the 'rock' token.
#  1    2     3   4     5   6   7
# 'the red  rock fell from the sky' 
#  w-2  w-1  w0   w1   w2   w2  w2
#  w-1  w0   w1   w2   -    -   - (clipped!)
# w0 refers to the relative distance to self, w1 refers to the next token in the sequence, and w-1 refers to
# the previous token in the sequence, since we have windows size of 2, we have 5 weights in total.
# note that the very same weights are used for each token entery. when its 'rock', the w0 is used to reflect
# the self(rock), but for the 'red' token, w0 refers to that token as self, and likewise, w-1 refers to its
# previous token which is 'the' and so on and so forth. 
# in other words, w0 means we are 0 hops away from the source token/node(in a graph), w1 means 1 hop away from
# the next token/(outgoing connection to next node) while w-1 refers to 1 hops away to the left of the token/node
# also note that after certain legnth (windows-size), tokens get the same value, and this doesnt provide much useful
# information, so in the paper they are clipped!(second row shows this where I didnt write anything for the remaining tokens)
# also note that with small ds (or ks as used in the paper), we lose long term relations/dependencies, and 
# long windowsize may grab noise and doesnt result in better performance (this is shown after some size k the 
# performance stops improving so the window size is an important parameter here. (https://www.youtube.com/watch?v=DwaBQbqh5aE)
# As you saw, we use the same set of positional weights
# for all tokens and this happens to all tokens at once. The input is added with this information and fed to 
# !rest of the attention pipeline(this information is added to both key and value embeddings). explain more! 
# so this way we learn a small set of position vectors and slide around and encode relative position
# information for each set of tokens and this gives us a lot of ability to generalize to new positions based
# on 'combinations' that we've seen before, possibly in other parts of these inputs.  
# 
# the concept of absolute positional encodings doesnt make sense for graphs for example, however, relative
# positions make sense for them, so the paper of shaw etal 2018 from google brain, proposes that since the
# attention mechanism unlike rnns and cnns, doesnot explicitly model the relative or absolute position information
# in its structure, they propose one that does! and they introduce attention with relative positional information.
# they argue that the absolute order of tokens is not that important(ideal), rather the relation of them, or their relative distance/position 
# between sequence elements in a sequence matters more and infact results in improved performance. 
# Furthermore, they also add that, combining relative and absolute position representations yields no further improvement in
# translation quality. they then go and describe an efficient implementation of their method and cast it as an 
# instance of relation-aware self-attention mechanisms that can generalize to arbitrary graphlabeled inputs
# they basically said: 
# "Our approach can be cast as a special case of extending the self-attention mechanism of the Trans-
# former to considering arbitrary relations between any two elements of the input, a direction we plan
# to explore in future work on modeling labeled, directed graphs"
# side note: 
# note that in their paper, they model the input as a labeled, directed, fully-connected graph.
# For linear sequences, edges can capture information about the relative position differences between 
# input elements. The maximum relative position we consider is clipped to a maximum absolute value of k. 
# We hypothesized that precise relative position information is not useful beyond a certain distance. 
# Clipping the maximum distance also enables the model to generalize to sequence lengths not seen during
# training. Therefore, we consider 2k + 1 unique edge labels.
# this in practice hasnt been used much! (especially in text sequences)
# transfomer relative position : https://www.youtube.com/watch?v=Ws2RAh_VDyU
#
# 00:00 Permutation Equivariance
# 01:12 Absolute Position Embedding
# 02:42 Limitation of absolute positions
# 03:56 Relative Position Bias intuition
# 07:57 Relative Position Bias in theory
# 12:53 PyTorch Implementation
# 
#
# A very good and intuitive way as to why positional encodings are added vs concatenated (ref https://www.reddit.com/r/MachineLearning/comments/cttefo/d_positional_encoding_in_transformer/):
# (to understand this fully, you need to read the previous notes I wrote/collected like orthogonality (which simply means independency!)) 
# In attention, we basically take two word embeddings (x and y), pass one through a Query transformation matrix (Q) and the second through a Key transformation matrix (K), and compare how similar the resulting query and key vectors are by their dot product. So, basically, we want the dot product between Qx and Ky, which we write as:
# (Qx)'(Ky) = x' (Q'Ky). So equivalently we just need to learn one joint Query-Key transformation (Q'K) that transform the secondary inputs y into a new space in which we can compare x.
# By adding positional encodings e and f to x and y, respectively, we essentially change the dot product to
# (Q(x+e))' (K(y+f)) = (Qx+Qe)' (Ky+Kf) = (Qx)' Ky + (Qx)' Kf + (Qe)' Ky + (Qe)' Kf = x' (Q'Ky) + x' (Q'Kf) + e' (Q'Ky) + e' (Q'K f), where in addition to the original x' (Q'Ky) term, which asks the question "how much attention should we pay to word x given word y", we also have x' (Q'Kf) + e' (Q'Ky) + e' (Q'K f), which ask the additional questions, "how much attention should we pay to word x given the position f of word y", "how much attention should we pay to y given the position e of word x", and "how much attention should we pay to the position e of word x given the position f of word y".
# Essentially, the learned transformation matrix Q'K with positional encodings has to do all four of these tasks simultaneously. This is the part that may appear inefficient, since intuitively, there should be a trade-off in the ability of Q'K to do four tasks simultaneously and well.
# HOWEVER, MY GUESS is that there isn't actually a trade-off when we force Q'K to do all four of these tasks, because of some approximate orthogonality condition that is satisfied of in high dimensions. The intuition for this is that randomly chosen vectors in high dimensions are almost always approximately orthogonal. There's no reason to think that the word vectors and position encoding vectors are related in any way. If the word embeddings form a smaller dimensional subspace and the positional encodings form another smaller dimensional subspace, then perhaps the two subspaces themselves are approximately orthogonal, so presumably these subspaces can be transformed approx. independently through the same learned Q'K transformation (since they basically exist on different axes in high dimensional space). I don't know if this is true, but it seems intuitively possible.
# If true, this would explain why adding positional encodings, instead of concatenation, is essentially fine. Concatenation would ensure that the positional dimensions are orthogonal to the word dimensions, but my guess is that, because these embedding spaces are so high dimensional, you can get approximate orthogonality for free even when adding, without the costs of concatenation (many more parameters to learn). Adding layers would only help with this, by allowing for nonlinearities.
# We also ultimately want e and f to behave in some nice ways, so that there's some kind of "closeness" in the vector representation with respect to small changes in positions. The sin and cos representation is nice since nearby positions have high similarity in their positional encodings, which may make it easier to learn transformations that "preserve" this desired closeness.
# (Maybe I'm wrong, and the approximate orthogonality arises from stacking multiple layers or non-linearities in the fully-connected parts of the transformer).
# tl;dr: It is intuitively possible that, in high dimensions, the word vectors form a smaller dimensional subspace within the full embedding space, and the positional vectors form a different smaller dimensional subspace approximately orthogonal to the one spanned by word vectors. Thus despite vector addition, the two subspaces can be manipulated essentially independently of each other by some single learned transformation. Thus, concatenation doesn't add much, but greatly increases cost in terms of parameters to learn.
#
# also this image: https://imgur.com/kaADdQB
# shows the relative postional learning capability of the sinosoidal positional encoding 
# ref: https://www.reddit.com/r/MachineLearning/comments/cttefo/comment/exoy4ww/"
# The second image just shows Euclidean distances between the added embedding for a given position.
# The thing is that their choice of positional encoding function reflects not only absolute, but also the relative distance among tokens in sequences.
# In the paper they write:
#     We chose this function because we hypothesized it would allow the model to easily learn to attend by relative positions, since for any fixed off set k,PE_pos+k can be represented as a linear function of PE_pos.
# Unfortunately, I don't really wave a better way to elaborate it further in my mind at the moment. I'm sorry for that.
# Answering your question, I personally think that they seem to simply pick up the first elegant solution to solve this problem.
# However, I believe that there are a lot of more interesting ways to take advantage of positional encoding trick. I'm currently working on it for my own dataset.
#"
# Theres a wealth of information and good points in that thread! read it all!
# such as : https://www.reddit.com/r/MachineLearning/comments/cttefo/comment/exnw14a/
# "Because the distances between the neighboring values of the sine/cosine positional embeddings are "nice" 
# (symmetrical, and decay sensibly with distance), You could come up with another function that does something
# similar."
# what does that mean? 
# "Symmetrical" - see image 2. https://imgur.com/a/p7NcUm1
# "Decay sensibly with distance" - In this gif you can see entries in the distance matrix which have similar value [https://imgur.com/a/p7NcUm1]
# White dots are points (corresponding to different pairwise positions) which lay in some epsilon neighborhood from some threshold tau which changes over time from 0 to maximum distance.
#
# this issue has some good points as well : https://github.com/tensorflow/tensor2tensor/issues/1591 
# and this one : https://github.com/tensorflow/tensor2tensor/pull/177
# intuitions such as addition creates subspace clusters, etc and concat vs add is discussed here
# ref: https://colab.research.google.com/drive/14RGALTsPIYGAuIByXGutK-aYN-PikWzF
# Why are positional embeddings (PE) added to word embeddings (WE) instead of concatenated?
# Assume: the PE and WE are vectors of length 512.
# Hypothesis: Many elements of the PE vector do not actually move very much as a function of the position - these are ~constant. Since those elements are constant, they do not add noise to the WE after the sum operation.
# In effect, the PE only contains information in the positions 1 through say 256. And the WE only contains information the positions say 257 through 512.
# So really... sum(WE, PE) generates the same amount of information as concat(PE[:256], WE[256:].
# this repo and book have good ipython notebooks that show each operation with easy to follow explanations 
# on googlecolab : https://github.com/Denis2054/Transformers-for-NLP-2nd-Edition





# refs: 
# Andre karpathy's video lecture on gpt : youtube link:
# https://timodenk.com/blog/linear-relationships-in-the-transformers-positional-encoding/ (this is really good and many used his intuition to explain this)
# https://www.youtube.com/watch?v=3mTsYm9qQFA
# https://www.youtube.com/watch?v=o29P0Kpobz0
# https://www.youtube.com/watch?v=JERXX2Byr90
# https://www.youtube.com/watch?v=M2ToEXF6Olw
#!https://www.youtube.com/watch?v=4AzsiCMw_-s
# https://www.youtube.com/watch?v=IWmpRaJ9Dz0
# https://www.youtube.com/watch?v=S27pHKBEp30 # epsecially the end of the lecture/question/answering has good stuff!
# https://notesonai.com/Positional+Encoding
# http://nlp.seas.harvard.edu/2018/04/03/attention.html#positional-encoding (might be good!)
# stanford's 2023 course on nlu seems like a good resource though I myself havent watched it fully : https://www.youtube.com/watch?v=K_Dh0Sxujuc&list=PLoROMvodv4rOwvldxftJTmoR3kRcWkJBp
# theres another course in-context I guess which are good as well: part1 is here: https://www.youtube.com/watch?v=eyNLkiQ89KI
# https://datascience.stackexchange.com/questions/55901/in-a-transformer-model-why-does-one-sum-positional-encoding-to-the-embedding-ra/117128#117128
# https://datascience.stackexchange.com/questions/110180/why-cant-positions-in-transformers-be-simply-appended-to-the-input-to-preserve
# https://assets.researchsquare.com/files/rs-2525471/v1/8624e58c681929fd04425437.pdf?c=1675250963
# file:///home/hossein/Downloads/00_2021_TransformerReview.pdf (https://www.researchgate.net/publication/360066821_Exploring_Recent_Advancements_of_Transformer_Based_Architectures_in_Computer_Vision?enrichId=rgreq-495e0a1aba907f54e457b0542f3306d0-XXX&enrichSource=Y292ZXJQYWdlOzM2MDA2NjgyMTtBUzoxMTQ2OTA4ODA2NTEyNjQxQDE2NTA0NTU3NzYzNDc%3D&el=1_x_3&_esc=publicationCoverPdf)
# Transformers in Vision: A Survey https://arxiv.org/pdf/2101.01169.pdf
# Exploring recent advancements of Transformer based architectures in computer vision
#
# https://www.tensorflow.org/text/tutorials/transformer
# https://github.com/jalammar/jalammar.github.io/blob/master/notebookes/transformer/transformer_positional_encoding_graph.ipynb
# https://www.scaler.com/topics/nlp/positional-encoding/
# https://towardsdatascience.com/master-positional-encoding-part-i-63c05d90a0c3


#%%
# misc sidenotes: 
# sidenote:----------------------------------------------------------------------------
# deprecated!
# TODO: use scaled_dot_product_attention to use Flash Attention v2 which speeds up
# the calculation a lot since pytorch 2.1 we can use this, but in pytorch 2.2 
# flash attention v2 was implemented which gives 2x more speed compared to 
# previous version which was fast by itself alone!
# ref : https://github.com/pytorch/pytorch/releases/tag/v2.2.0 
# Updated flash attention kernel in scaled_dot_product_attention to use Flash Attention v2 (#105602)
# Previously, the v1 Flash Attention kernel had a Windows implementation. 
# So if a user on Windows had explicitly forced the flash attention kernel
# to be run by using sdp_kernel context manager with only flash attention enabled,
# it would work. 
# In 2.2, if the sdp_kernel context manager must be used, use the memory efficient
# or math kernel if on Windows. 
# with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
#   torch.nn.functional.scaled_dot_product_attention(q,k,v)
# # Don't force flash attention to be used if using sdp_kernel on Windows
# with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=True):
#   torch.nn.functional.scaled_dot_product_attention(q,k,v)

# update 2026
# TODO: use scaled_dot_product_attention (SDPA) to take advantage of
# note in practice we dont write the attention mechanism ourselves
# because to actually get the most out of our hardware, we need lots
# of optimizations both for much better efficient memory usage and
# computtaion speed. for that we have a few options. lets talk about them. 
# In Pytorch we can use its optimized attention kernels such as
# FlashAttention when available for our software/hardware stack.
# PyTorch automatically selects an appropriate SDPA backend depending on
# the input tensors, hardware, dtype, sequence length, etc.
# This can include FlashAttention, memory-efficient attention, cuDNN attention,
# or fall back to the standard math implementation when an optimized kernel
# is not available/supported.
# In older versions, we had to pay much more attention to which
# backend was being used. PyTorch 2.2 introduced FlashAttention-2 support
# in scaled_dot_product_attention, which provided a significant performance
# improvement over the previous FlashAttention implementation.
# The API for manually controlling the SDPA backend has also changed.
# We dont use the older torch.backends.cuda.sdp_kernel API for new code
# anymoe, instead we use torch.nn.attention.sdpa_kernel from torch.nn.attention module:
#
# from torch.nn.attention import SDPBackend, sdpa_kernel
#
# with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
#     output = torch.nn.functional.scaled_dot_product_attention(q, k, v)
#
# Also, we generally should *not* force a specific backend unless we have
# benchmarked it and know that it is beneficial for our particular workload.
# SDPA can automatically select the appropriate backend, so in most cases
# simply calling scaled_dot_product_attention is the right way.
#
# Also note that Pytorch's built-in FlashAttention backend and the standalone
# FlashAttention package are related but are not exactly the same thing.
# The official FlashAttention project provides its own highly optimized CUDA
# implementations and Python bindings, which can be installed separately.
#
# For example, installing the standalone package flash-attn (e.g. pip install flash-attn)
# allows us to directly use the FlashAttention kernels.
# (like e.g. 
# from flash_attn import flash_attn_func
# output = flash_attn_func(q, k, v, causal=True)
# )
#
# This can sometimes be significantly faster than the default Pytorch
# implementation for particular workloads, but it is NOT guaranteed to be
# faster. The result depends heavily on the GPU architecture, tensor shapes,
# dtype, sequence length, head dimension, CUDA/Pytorch versions and the
# particular kernel that ends up being selected.
#
# The FlashAttention project has also continued to evolve considerably.
# FlashAttention-2 was followed by FlashAttention-3, which targets Hopper
# GPUs such as the H100, and the project now at the time or writing this(sep 2026) 
# also has FlashAttention-4, which uses CuTeDSL and targets newer Hopper
# and Blackwell GPUs These newer implementations are specifically optimized around the
# capabilities of newer NVIDIA hardware.
#
# Therefore, if performance is important, it can be worth benchmarking:
#
# 1. regular PyTorch attention
# 2. PyTorch SDPA with automatic backend selection
# 3. SDPA with a specific backend forced
# 4. the standalone FlashAttention package
#
# The fastest option can change depending on the GPU and input dimensions,
# so we should benchmark the actual workload instead of assuming that
# FlashAttention is always faster.
#
# In particular, dimensions such as head_dim, sequence length, batch size
# and number of heads can have a large impact on which kernel is selected
# and how efficiently the GPU can execute it.
#
# Another interesting PyTorch regarding attention implementation is its FlexAttention module.
# we will learn more about it later on. (see the end for more)
# -------------------------------------------------------------------------
# Flex attention explanation: 
# TODO: we need to talk about AliBi,RoPE, so the explanation below fully clicks. 
# I need to bring the code/explanations from my ImageLet project over here
# and then this part 
# FlexAttention is a bit different from regular scaled_dot_product_attention (SDPA).
# With SDPA, we basically describe standard attention and let PyTorch choose
# the best available optimized attention backend.

# however FlexAttention is built for cases where we want to customize the attention
# computation itself, for example with custom score modifications, masks,
# sliding-window attention, ALiBi, document masking, soft-capping, etc.
#
# Instead of writing our own CUDA/Triton kernel for every custom attention
# variant, FlexAttention allows us to describe the modification in Python
# through things such as score_mod and mask_mod, and torch.compile can then
# generate a fused attention kernel for us.
#
# This is another good example of why fusing operations is not as simple
# as just combining several Python operations into one.
# The compiler can generate a completely different kernel underneath,
# with different tiling strategies, memory accesses, register usage, etc.


# todo : add updates and notes about transformer architectural updates, 
# and positional encodings used from my imagelet project here. 
# maybe even have a clean version without repeative comments 
# and include only the novel version, and at the very end show
# what we actually use in practice? 