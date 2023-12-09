#%%
# in the name of God the most compassionate the most merciful
# in this section we will have a look at how a GPT (which is 
# short for Generative Pretrained Transformer) model works,
# in this section we will implement the attention mechanism
# which is the foundation of transformers, and see 
# how it works and lean more about it (self-attention, cross
# attention, multi-head attention, etc)
# 
# before we delve into the implementation details, we have to take a detour
# and discuss some underlying concepts and techniques involved.
# 
# so how are we going about this. we need to create a language model
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
with open('./tiny_shakespear.txt','r') as file:
    # this time we read the whole text as one big str
    dataset = file.read()
    
print(f'{dataset[:100]=}') 
# prints:
# dataset[:100]='First Citizen:\nBefore we proceed any further, hear me speak.\n\nAll:\nSpeak, speak.\n\nFirst Citizen:\nYou'
# now lets create our atoi and itoa dictionaries for mapping
# lets create our unique character list, which is infact our vocabulary or vocab for short
vocab_list = sorted(set(''.join(dataset)))
vocab_size = len(vocab_list)
# lets create our mapping dictionaries, we are basically going to use them
# for tokenization, converting our input into tokens which here are characters 
# and ultimately their integer representations
# so tokenization simply put, refers to converting an input into a list of numbers based on some creteria
# here, we just use a simple index number to map our vocabulary characters, which represent all character
# our model can use to generate text, to integer representation. 
# for realworld applications, we usually use libraries such as sentecepeace by google which works on a subword
# level which means, it doesnt encode the whole word, neither does it encode based on individual characters, it
# use sth in between. 
# (from its github repo: https://github.com/google/sentencepiece )
# SentencePiece is a re-implementation of sub-word units, an effective way to alleviate the open vocabulary 
# problems in neural machine translation. SentencePiece supports two segmentation algorithms, 
# byte-pair-encoding (BPE) [Sennrich et al.] and unigram language model [Kudo.]. 
#
# (For the reference:  
# Byte Pair Encoding (BPE) is a subword tokenization technique used in natural language processing (NLP) 
# and text processing tasks. It is a data compression algorithm that splits words into subword units. 
# BPE is commonly used in tasks such as machine translation, text generation, and language modeling.
# The basic idea behind BPE is to iteratively merge the most frequent pairs of characters or subword units
# in a corpus to create a new subword vocabulary. This merging process is based on the statistical properties
# of the corpus, specifically the frequency of character or subword pairs.
#
# Here's a high-level overview of the BPE algorithm:
# 1. Initialize the vocabulary with all the characters or subwords in the corpus.
# 2. Calculate the frequency of each character or subword in the corpus.
# 3. While the desired vocabulary size or a maximum number of iterations is not reached:
#    - Find the most frequent pair of characters or subwords in the corpus.
#    - Merge the pair into a new subword unit by concatenating them.
#    - Update the corpus by replacing occurrences of the merged pair with the new subword unit.
#    - Update the vocabulary and frequency counts based on the new corpus.
# 4. The final vocabulary is the set of subword units obtained after the desired number of iterations 
#    or vocabulary size is reached.
#
# BPE allows for the representation of both known and unknown words in a corpus. It is effective in 
# handling out-of-vocabulary (OOV) words and reducing the vocabulary size, which can improve the 
# efficiency and performance of NLP models. By breaking down words into subword units, BPE can capture
# morphological and semantic information more effectively, especially for languages with complex word 
# formations and agglutinative structures.
# 
# tiktokenize repo explains BPE in a rather friendlier way: 
# Models don't see text like you and I, instead they see a sequence of numbers (known as tokens). 
# Byte pair encoding (BPE) is a way of converting text into tokens. It has a couple desirable properties:
# It's reversible and lossless, so you can convert tokens back into the original text
# It works on arbitrary text, even text that is not in the tokeniser's training data
# It compresses the text: the token sequence is shorter than the bytes corresponding to the original text. 
# On average, in practice, each token corresponds to about 4 bytes.(i.e. 4 characters!)
# It attempts to let the model see common subwords. For instance, "ing" is a common subword in English, 
# so BPE encodings will often split "encoding" into tokens like "encod" and "ing" 
# (instead of e.g. "enc" and "oding"). Because the model will then see the "ing" token again and again 
# in different contexts, it helps models generalise and better understand grammar.
#
# 
# A unigram language model however, is a type of statistical language model that predicts the probability 
# of each word in a sequence independently, based solely on the frequency of occurrence of individual 
# words in the training data. It does not consider the context or the order of the words in the sequence.
# In a unigram language model, the probability of a particular word is estimated by counting the frequency 
# of that word in the training corpus and normalizing it by the total number of words in the corpus. 
# The probability of a sequence of words is then calculated by multiplying the probabilities of each 
# individual word in the sequence.
# For example, consider the sentence "I love to eat pizza." In a unigram language model, the probability
# of this sentence would be calculated as the product of the probabilities of each word which would be: 
# P(I) * P(love) * P(to) * P(eat) * P(pizza).
# Unigram models are the simplest form of language models and do not capture any contextual information
# or dependencies between words. They are often used as a baseline or reference model in natural language 
# processing tasks. While unigram models are not very accurate in capturing the complexities of natural 
# language, they can be computationally efficient and useful in tasks where the context is less important, 
# such as certain text classification tasks or language generation tasks where only the frequency of 
# individual words matters.
# 
#
# Openai/chatgpt for example uses its own tokenizer, called tiktoken (https://github.com/openai/tiktoken)
# there are other tokenizers as well.
# huggingface has a good article on tokenizers (recommend it): https://huggingface.co/docs/transformers/main/tokenizer_summary 
# its a given that, each model only works with the tokenizer by which it was used to train. 
# so BERT, DistilBERT, and Electra only work with wordpiece, while chatgpt uses titoken and we! use
# simple characters-indexes! also note that the choice of tokenizer obviously affects our vocab size and
# model overhead/performance as well. 
# for example lets first implement our tokenizer and then compare it with sth like tiktoken module
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
# so thats why in practice, these subwords tokenizers are used for real world applications. but for our case
# we stick to our primitive tokenizer to keep things as simple as possible. 
# 
# Ok, so far so good. now we need to create a dataset, like before we need to have an input/label pair
# our input is a series of characters, (a sequence of some length), and our label is the next character
# sicne we are using a simple bigram model, given a single character, we want the probablity of what comes
# next. but, we also want to incorporate attention, and we want a context for our prediction, we want to
# be able to look at the past, the past characters, and based on that do sth. this is the essence
# of attention (although this is not accurate, but for now let assume this is the case, we will elaborate on this and expand
# this metaphor and reasoning inshallah)
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
# note that we do not shuffle the data here like before, because we are not dealing with a list of samples!
# like names, here we are dealing with the whole text, and if we shuffle it like that, we just destroy it
# making it into a batch of random characters! which would not be useable for us anymore. we are trying to
# learn the underlying semantic and relationships hidden in our data, and randomizing them like that just 
# destroys those information! 
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
# note that since we are planning on creating a simple transformer model, we usually dont feed the whole dataset
# becaue its prohibitevly computation intensive, instead, what happens in practice is that we, grab chunks 
# of data from the dataset and feed it to the model. these chunks, of course have a length, what length?
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
# one intresting observation we can make here is that, as simple as this seemingly ordindary list of numbers
# looks, this sequence of numbers, actually contains several examples. 
# if you think about it, each number, is a token, representing a character (or word, subword, etc),
# and it shows what comes after what. basically not only it shows several pairs so to speak, it also
# shows, which characters are more likely to come, before a specific character comes later. 
# what we are actually going to do is that, we are going to train all of these characters simultaneously
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
# thats why we started from 1, becasue the input started from 0, and since the input ended at block_size
# its label(next character) would obviously be block_size+1, pretty obvious right?
#
# lets print this for better understanding 
for i in range(block_size):
    # note that since we are using slice, x[:i+1], gives us 0 up to ith element (inclusive)
    # this should be obvious! but I said it in case you forgot!!
    print(f'input is {x[:i+1].tolist()} label is {y[i]}')
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
# note that we do not do this simply for the efficancy aspect of it, but also, when we feed these to
# our transformer model, we are making sure that the transformer model gets used to see all combinations
# of our input as well (from the context size of 1 up tp the context size of 8). 
# this way not only it sees the whole context_size as we initially expected
# but also all sequences before it, and basically what consituted to make the sample. 
# this allows us to later on, at test time be able to create sequences as small as context_size of only 1
# up to the max_length which is our context_size of 8 in our case(and more).
# so by doing this, the model can learn how to predict/create/genrate text up to context_size, and after
# it reached that, we have to truncate it, becausse the model never recieves more than the
# context_size as input when its predicting the next character.(we can continue generating infintely, but
# really, what the model does, is to always generate the next character based on the 'last' context_size 
# number of tokens. if context size is 8, only the last 8 characters are taken into account for creating
# the next one. this matters if you think about it, the model can only memorize 8 tokens! all that came 
# before is just gone! the model cant use any of them to infer new information! it can only use the last
# context_size tokens! but nevertheless as we later see, we can continue to generate infinit characters
# thogh they may not make sense if the context size is small as you can guess!)
# 
# so far what we covered here was the time dimension of our input. we have a sequence, and each entery
# basically denotes a time t dimension, at which, a character is introduced. 
# another imporatnt aspect we need to take care of is the batch dimension, cuz we are going to feed 
# multiple examples at once, we use this to harness the gpu parallilization capalibity in pytorch
# as without it, training this simple model would take a lot of time on cpu! 
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
    # by adding the next 8(context_size) characters to them, in order not to go past the 
    # last index, we subtract the length of data(last valid index) from context size 
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
    # as you can see we have a list of tensors. since we want a batch, we just stack them or concat them
    # on top of each other
    # using stack!
    # x = torch.stack(x)
    # y = torch.stack(y)
    # stack() is intrestingly implemented by concat(), so if we want, we can do the same using concat!
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
    # concatenating them with torch.cat/concat/concatenate.(side tip: concat and concatenate are aliases for cat) 
    # This effectively replicates the behavior of torch.stack.
    # see torch.cat concatenates tensors along an existing dimension, and we only have 1 dimensional
    # tensors, so it will concatenate them along that, effectively making one large 1 dimensional vector
    # however, when we use unsqueeze on each tensor, using torch.unsqueeze(0), we are adding a new dimension
    # (dim 0) to that tensor, making it 1,x instead of the original shape of (x,).  
    # So, by unsqueezing each tensor along the desired dimension, which for us is the 0ths dimension (or row dim) 
    # and then concatenating them with torch.cat, we achieve the same result as torch.stack.
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
# now this is our batch of data, 4 samples with 8 characters, bascially 32 examples, lets see that as well
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
# as we said, we are going to use a bigram model and later add attention mechanism to it. 
# to make things easier and more self contained, lets add all the required logic to this model
# like when we do a forward, we be able to calculate loss as well if we are given the targets
# so lets go
class BigramModel(nn.Module):
    def __init__(self, vocab_size) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        # our bigram model was nothing more than a 2d array of vocab_size, we can achieve that using 
        # a single weight matrix or torch.Embedding. we use torch.Embedding to not reinvent the wheel!
        # and since all new language models use word-embeddings! so its a good choice for our base model anyway!
        self.token_embedding = torch.nn.Embedding(vocab_size, vocab_size)
    
    # since we want to be able to calculate loss, if there are labels, we get Y as well
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
            # so we need to account for that.
            # we can go on permuting the dimensions, like this and rectify the issue:
            # logits = logits.permute((0,2,1))
            # however, if for some reason the output of logits in the form of (4,8,65) is not ideal for us, 
            # and instead we want the usual form of (B,C) e.g, we can easily do it that way instead, 
            # think about it, what really it is, is 4*8=32 examples arranged in a (4,8) shape and 
            # if we rather a normal 2d tensor of shape (32,65), we can simply merge the first two dims!  
            # lets do just that, flatten the two dimensions into one and carry on!
            # we are basically concatenating the batch and time dimensions
            # into one dimension (effectively, stacking samples on top of each other), instead of having 
            # four compartments, each having 8 segments, we are going to have 1 long compartment with 32 segments/rows
            # for the lack of better words!!
            # so lets first get the shapes 
            # B,T,C = logits.shape
            # logits = logits.view(B*T,C)
            # and we also need to do the same for the labels 
            # labels = labels.view(B*T)
            # we could also do,
            # labels = labels.view(-1)
            # but thats not really needed, so we just simply permute logits temporarily so the logits shape
            # stays the same regardless of calculating the loss or not :-)
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
        # -create our final text output
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
            # sidenote: you might ask we have 7 choices, why are we choosing the last character/token?
            # can we choose any of those 7 time dimensions? like 1,2,3...,7 as well?
            # in this case it really doesnt matter and the loss stays the same, as its just a bigram model!
            # if we chose the last token its not like, the model cares about all the context_size, no, its just 
            # meaningless to it, it does not have the capability to utilize the context_size at all! 
            # so for this specific case, we can select any dimension other than 0 obviously and the loss wont change!
            # but to be consistent with our future changes, and not changing the codebase as much as possible, 
            # and the fact that when generating text with the starting token (zeros(1,1) as we will see in a moment)
            # we have a single time dimension (i.e. 0) so any other value (except -1) would result in an error, we 
            # dont hardcode a specific dim, and instead use -1 to refer to the last dim whatever it happens to
            # be at the time of execution. 
            # also note that while the choice of dim here doesnt affect the loss, it 'does' affect the text generation. 
            # try different dimensions when trying to generate text after you trained the model and see for yourself 
            # you can also use different dimensions as we talked about during training and see the loss wont change!
            # (provided you set seeds so the output becomes determinstic)
            # 
            logits = logits[:,-1,:] # this now becomes (B,C) instead of the initial (B,T,C)
            probs = torch.softmax(logits, dim=-1) 
            # now lets sample from it! what! why? you may ask!
            # see genrating output really has nothing to do with the model, in a sense that, comming up
            # with what we consider satisfactory can be more than just selecting the entry with highest probablity
            # we can do all sorts of things to direct the generation process toward what we find satisfactory
            # we can use temperature, top-p, etc to do just that. Using sampling using here is for this very reason
            # To this end, we use torch.multinomial for generating the next token, this is a technique known as sampling! 
            # instead of simply choosing the token with the highest probability (which is known as greedy decoding by the way),
            # sampling selects the next token randomly according to the probability distribution produced by the model1.
            # the reason we use sampling instead of the greedy decoding method (i.e. the normal way!) is to introduce more diversity
            # and randomness into the generated output. 
            # If we always choose the token with the highest probability, the generated text can become repetitive 
            # and deterministic, especially over long sequences. By introducing some randomness like this, we can 
            # generate more diverse and interesting output.
            # note that this can sometimes generate less probable (and potentially less coherent) sequences
            # as well (obviously!). The balance between diversity and coherence is a common challenge in text generation, and 
            # different decoding strategies (like greedy decoding, sampling, beam search, using different temperatures, 
            # or top-p, etc) offer different trade-offs.
            # also replacement if true, means, if something is selected, it can be selected again! (basically 
            # its place will be filled/replaced and ready to be used again, like if e.g. you chose/pickedup an
            # apple! another apple will be repalce the old one, so you always have apple!)
            # basically if a token is picked one, replacement=True, means it can be selected again
            idx_next_char = torch.multinomial(probs, num_samples=1, replacement=True) # shape is (B,1)
            # now lets add this to the next input to be fed to the model 
            # since idx has the shape(batch, T), we should add this tothe second dimension
            # to the time dimension/ or sequence dimension. this as the loop goes on, 
            # creates the shape (B,T+1) 
            #this doesnt make sense for this particular model, becasue we are always checking
            # the next character given the previous one, so all the concatenation we are doing
            # is just useless now. the reason we are implementing this like this, as I pointed out earlier
            # is to create a base foundation, so that we can improve upon it when we add attention later on 
            # which will use the history of previous characters.
            idx = torch.cat((idx, idx_next_char), dim=1) 
        
        # and finally when all is done return the idx which by now should have the whole output
        return idx
    
model = BigramModel(vocab_size)
out,loss = model(x,y)
print(f'{out.shape} {loss}')
# prints:
# torch.Size([32, 65]) 4.574285984039307
# the loss is good, we learned previously that we can evaluate a base loss provided our number of classes
# since we have 65 classes (our vocab_size or number of characters involved) the uniform probability for each class
# would be 1/65 =0.015384615, which if we take its negative log would turn out to be -ln(1/65) = 4.17438727, 
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
for i in range(max_iter):
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
    
    if i%1000==0:
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
# relying on single batch loss is not a good idea to measure the performance of our model. moreover
# relying on training loss, is not good either, so it would be much better if we considered more batches
# for loss and even better we could also investivate the models performance on our validation set. 
# so lets do just this and define a function that calculates loss for training and validation sets alike
# but considers more batches for loss calculation

# this decorator signals torch not to calculate gradients for its operations!
@torch.no_grad()
def evaluate_loss (iterations, device=None):
    results={}
    # before calculating the loss, lets switch to eval mode,although for our specific case this doesnt matter
    # but its good practice, as later on, we will add layers that their behavior do change depending on traing
    # val mode.  
    model.eval()
    if device is None:
        # use the device assigned to what model params are assigned
        device = next(model.parameters()).device
    # since we already used no_grad decorator, we dont need to use no_grad context manager here
    # with torch.no_grad():
    for split in ['train','val']:
        losses = torch.zeros(size=(iterations,), device=device)
        for i in range(iterations):
            x,y = get_batch(split,batch_size)
            x,y = tuple(t.to(device) for t in (x,y))
            logits, loss = model(x,y)
            losses[i] += loss
        results[split] = losses.mean(0)
    # since we want to use this inside training loop, make sure we set the model back to train mode
    # incase we use layers such as batchnorm,etc that the require being trained!
    model.train()
    return results

loss= evaluate_loss(200)
print(f'{loss["train"]=} {loss["val"]=}')
# so now that our function seems to be working lets use it in the training loop :
model = BigramModel(vocab_size)
optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-3)
model = model.to(device=device)
model.train()
for i in range(max_iter):
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
    
    if i%1000==0:
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
# for our case here, this is specifically about the past history (i.e. past tokens and how they are related 
# or show/affect the next token probablity)
# but in general attention can utilize both past and future connections/sequence tokens. 
# what we just described here is not the whole story, but its enough to gives us a foundation to build our
# intuition as we continue on. 
# we will elaborate more of course and hopefully gradually improve our definition
# and understanding of the attention mechanism.
# we said attention is a kind of communication mechanism, but what does that mean? how do tokens communicate?
# to put it simply, by communication, we mean to somehow involve the value of one or more tokens in the operation
# so that they can play a role in the final output. so one simple way could be to just sum all the values/weights 
# assosiated with each token (previous tokens), and use that to determine the value/weight for the current token!
# another way could be to use average instead of sum, of the said tokens! 
# now lets expand on this and get an intuitive undrestanding what all of this means.
# before we venture any further into the crux of the matter!(big words:-)), let us learn about a technique
# thats used to efficiently implement attention mechanism, after this you should get a good idea about all of this.
# For this purpose,lets imagine we have a simple input like the following: 
# lets create an input of the following shape
B,T,C = (4,8,2)
# to make it more intuitive lets make a tensor with known numbers and then reshape it
x = torch.arange(0,64,dtype=torch.float).view(B,T,C)
print(f'x={x}')
# imagine we have an input like what we encountered previously in our examples. in this sample 
# input, we have a batch of 4 samples, each having 8 sequences with each sequence having a vector of 2 values
# what we are planning to do is to provide a way by which each token can communicate with other 
# tokens. we have 8 tokens in our sequence. so we want our tokens to be able to communicate with
# all the previous tokens that came before them. 
# The reason we are only looking in the past token is simply becasue we are trying to perdict the future,
# so it only makes sense to look at the past and current timestamp and infer on what to do for the future.  
# as we said earlier, one of the easiest ways we could come up to implement such communication mechanism 
# between tokens could be to sum or average the values of all previous tokens as a way of taking into account
# their contribution to the [final] answer/output.
# that is, lets say if we are currently at token 5, we take the average of 
# the current token and all previous tokens before it, effectively making a feature vector that
# reflects our current status of the sequence so far, having taken all previous tokens/steps up to now.
# note that as you may also have guessed, summing/averaging arent the best way to model such interations.
# in fact they are an extremely weak form of interaction between tokens,
# this kind of communicating is extremely lossy so to speak, that is we lose a great deal of information 
# concerning the underlying relationships between tokens, their arrangements,their implicit interactions,
# semantics, etc. but for now this is ok. we will later on see how we can fix this issue.
# so now what we want to do, is to calculate the sum or average of all tokens up to the current token in 
# all batches at the same time.
# a naive way would be to do sth like this using a for loop:
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
# You may notice that, some people/researchers refer to what we did here(averaging) as BoW, or bag of words. 
# We are effectively averaging embeddings here and averaging embeddings can be considered a form of 
# Bag of Words (BoW) representation. In BoW, the focus is on the occurrence and frequency of words, 
# rather than their order or structure. By averaging embeddings, we are essentially treating each 'word'
# as an independent feature and capturing its representation in the form of a numerical vector.
#
# Although what we have is basically embeddings and averaging embeddings does not capture the exact 
# frequency of each word, it does however, capture the overall distribution and semantic information
# present in the text. 
# Like BoW, our approach here disregards word order and focuses on the presence and representation of words. 
# also note that averaging embeddings may preserve some semantic relationships between words, which BoW 
# representations might not capture as effectively.
# 
# Side note: 
# Bag of Words (BoW) is a commonly used technique in natural language processing (NLP) for representing text
# as a numerical feature vector. It disregards the order and structure of words in a document and focuses only
# on their occurrence and frequency.
# In the BoW model, a document or a piece of text is represented as a "bag" (unordered set) of words, where 
# each word is treated as an independent feature. The presence or absence of words in the document is encoded
# as a binary value (0 or 1), and the frequency of each word is often used as the value in the feature vector.
# 
# for the reference, this is how a BoW process works:
# First, we have the tokenization phase, in which the text is split into individual words or tokens. 
# punctuation marks, whitespace, and other special characters are usually removed or treated as separate tokens.
# Then, a vocabulary is created by taking all unique words from the entire corpus (collection of documents). 
# each unique word is assigned a unique index or position in the vocabulary. this is the vocabulary creation phase!
# next each document is represented as a feature vector, typically a as one-hot encoded  vector or a count vector. 
# (as the name suggests, in the one-hot encoding scheme, each word in the vocabulary corresponds to a binary feature, and the vector
#    contains 1s in the positions where the word occurs and 0s elsewhere. 
# while in a count vector, the value at each position represents the frequency of the corresponding word in the document)
# and finally, the resulting feature vectors can be used as input to out models for tasks such as text classification,
# sentiment analysis, document clustering, information retrieval, etc. 
# Needless to say, BoW has some limitations as well. It does not capture the semantic meaning or context of the words,
# as it treats each word independently. It also ignores the grammar and word order. 
# Having all these said, BoW is simple, efficient, and can be a useful baseline representation for various NLP tasks.
# 
# so to recap one more time, we are basiaclly treating each timestep/sequence dimension, as a 'word', so we have
# 8 tokens/words, and we are averaging them (we are infact averaging their embeddings, but thats obvious!)
# so we got ourselves a bow representation!
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
# if you look closely, you'll notice that each row, contains the mean of all the rows before it
# consider the first 3 rows in x[0], 
# tensor([[ 0.,  1.],
#         [ 2.,  3.],
#         [ 4.,  5.],
# now refer to the 3rd row in results[0] which is :
#         [2., 3.],
# likewise, consider the last row in results which contains the average for all the rows in x:
# 0+2+4+6+8+10+12+14 = 56 which when divided by their count, 8, results in 7, 
# and this is the same for the second column, thus we get:
# results[0,7] = [7., 8.]])
# this all good but the problem is using for loops to calculate this is very inefficient, it
# happens that this operation can be efficiently calculated using matrix multiplication.
# lets learn this trick using an example: 
# suppose we have the following as the input: 
a = torch.ones(size=(3,3))
b = torch.randint(0,10,size=(3,2)).float()
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
# nothing fancy here, we have matrix multiplication, the first row of 'a' is dot-producted by first col
# of 'b', then sumed, it makes up the first col of first row in c. likewise the first row of 'a' dot 
# the second col of 'b', then summed the results, makes up the second col of first row in c. and this 
# goes on for the rest of the matrixes. this is what we learned back in higheschool, but what is it 
# exactly that we are learning here exactly?!
# if you look closely, you'll notice that, the c cols are actually the sum of all the rows in b!
#        [9]
# b[:,0]=[5] 
#        [6]
# 9+5+6 is 20! likewise, 
#        [3]
# b[:,1]=[5] 
#        [5]
# 3+5+5 is 13!
# hence c = [20., 13.] which is repeated obviously because the second and third rows of a are all 1s as well.
#           [20., 13.]  
#           [20., 13.] 
# I guess you are now starting to get where we are going with this, if we can some how alter the 'a' matrix,
# we may very well be able to achieve our goal! 
# how you may ask? the answer is using torch.tril!
# torch.tril() is a function that returns a matrix from a given tensor, so that half of it set to zero,
# basially it creates a triangular tensor, where the right half is just zeros! lets see how it works, 
# lets apply it on 'a'
a_tril = torch.tril(a)
print(f'a_tril:\n{a_tril}')
# it prints
# a_tril=tensor(
#        [[1., 0., 0.],
#         [1., 1., 0.],
#         [1., 1., 1.]])
# as you can see the the right half is set to zero and we are left with a triangle shape of 1s on the left side
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
# now if you look closely, you'll notice that, this time, each row in c, is effectively the sum of the previous
# rows in b, like the first row of 'a' is only 1 in the 0ths column, so the first row of b is copied in c intact
# (workout the math and see why). 
# the second row in 'a', now has two 1s in col 0 and 1 respectively, which effectively translates to summing the first
# two rows in b. (9+5 =14, 3+5=8). likewise, the third row in 'a' is all 1s, signfigying all rows in b will
# be summed which gives us (9+5+5=20, 3+5+5=13). 
# so basically we are doing sums here, becasue our tensor a is all ones. so if we want to somehow calculate the
# average, instead of sum, we can easily change 'a' by normalizing it so that the each row sums to 1 (i.e. all cols
# sum to 1), this way the end result will be the average (becasue the 'b' is multiplied by a fraction/scale and then summed)
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
# likewise, in the third row, as there are 3 1s, the probablity is divided between all of them, making each
# to have the value 0.33
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
# side note:
# one more thing before we continue on, note that the weight matrix is TxT while 
# the input is (BxTxC). (T,T) and (B,T,C) are not compatible, so what happens is
# that (T,T) is reshaped and a batch dimension is added to (T,T),making it (1,T,T)
# and then this is broadcasted along the batch dimension (replicated) to become 
# (B,T,T), then this will be multiplied by the (B,T,C)( note that at this stage
# a@b will be a batch multiplication operation, if you set the batch dimension aside, youll
# see that the rest of the dimensions match up, we have (T,T) and (T,C) which will
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
for i in range(b.shape[0]):
    c2[i,...] = a[i]@b[i] # (T,T) x (T,C) -> (T,C)
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
# wouldnt provide us with. think about it for a moment, what would we do if we didnt want uniform probbablity
# for all the tokens in a row? currently as you can see, we have uniform probablities in each row, we used that 
# to calculate the average, becasue that was what we were after initially! but we said it earlier that 
# average is a very weak form of communication! we lose a lot of information if we only use simple averaging 
# like this. if you think a bit more, you'll notice that its really the uniform probablity that each token gets, 
# uniform probablity means they are basically the same, in terms of impact on the output. 
# thats like they are as important as the other tokens in every single situation! this assumption is
# simply and clearly not true at all. again to understand why this is the case remember the weights matrices
# in our models? if our model predicts all the classes the same, like at the very begining, when we havent 
# started the training, the model is practically useless! until we train the model, and the weights get 
# different values. remember trained weights vs raw weights! so this is the same here really.)
# now if you see the previous implementation vs the newer one, you'll see that we basically 
# the 'weight' matrix can essentially be anything and not just zeros! 
# in other words, we have decoupled the 'weight' part from the constraint part (i.e. tril), which's job is 
# to set the right half of the tensor to zero, so we actually get the expected behavior later on. 
# (which is setting a constraint really (more later on))
# if you havent yet figured it out, the 'wieght' matrix, can be truly a weight matrix
# which can show different strengths for each token, basically it can learn the interations
# between tokens and manifest them. currently it is us who set it to all zeros, so we get
# uniform probablities, which inturn manifest itself as an average. but what if instead of 
# all zeros, we actually learn the values from the data itself? that would make sense, and 
# make it so that each token, can have a different connection(strength) to any other tokens
# now, and thus build semantic/meaningful relation. 
# This is infact what we are after. we want for tokens to learn associations and relations with other tokens
# based on the data present in the dataset, having uniform weight like what we initially did, is really a 
# far cry from what we intend, and therefore, we opt in to use this new approach that allows us to actually
# exploit this new capability. 
# This is infact the problem that attention solves, that is gathering
# information from the past but in a data driven manner.(side note/reminder, we are not limited to 'past' 
# information only per say, in this example, this is the case however, we will explain this 
# in more detail)), we will see how attention does this exactly in a moment. 
#
# but before we jump into attention implementation also note that the tril part, infact is a hard-constrain here
# that prevents tokens from the past from interacting with the tokens from the future. 
# so to recap here, basically the idea is, this triangular form, allows us to have 
# weighted aggregations of past elements. each element in the lower triangular part, 
# specifies, the degree by which it plays a rule in the said outcome. 
# that is how much of each element gets to get incorporated into the result of this specific position(i.e. current token's)
# so now lets incorporate attention into our model.
#
# Attention does its job by using two vectors called, key and query.
# !basically every single token, emits two vectors called key and query, the query verctor
# as the name suggests, implies, what we are looking for, and the key vetcor, again as the
# name suggets, implies, the contents, what it contains. 
# the way we get our 'weights' for these tokens is we simply dotproduct them together, 
# the ones that yield a high output/value, signify they are related positively.
# so our query is dotproducted with all the token's(keys) and the outputs reveal their 
# closeness/relevancy/similarity so to speak(if they align so to speak, they result in larger number)
# so effectively, the ones with higher number, are more similar to the query, the highest, 
# obviously having the most relavancy/similarity to the query.s
# so lets implemenet this
# #%%
# we want to implement a single attention head, we can later use this to create 
# multi-attention-head which is basically several single attention heads working in 
# parallel. so how do we implement one! 
# first lets create some random input 
torch.manual_seed(255)
# lets specify the batch, context_size and vocab_size
B,T,C = 4,8,32
# lets create an input 
x = torch.randn(size=(B,T,C))
# we said that attention works with two vectors, key and query, lets implement them
# we can use nn.Linear to implement them but beore that, what are the dims of such vectors,
# we are dealing with text and thus our inputs are tokens/vocabs, so the first dim would be 
# our vocab_size to account for all tokens, the next dim, is sth called a head_sizes, which 
# specifies the size of the key/query output.  
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
# it to reveal the relation/similarity/relevance of every token with every other tokens.
# and as we explained earlier, this is infact our weight matrix (which was initially all zeros)
# but is now learned from the data!
# print(f'weight_raw\n{weight_raw}')
# now we can apply the constraint on it so that tokens can only communicate with the past so 
# we use the tril trick now!
tril_constrain = torch.tril(torch.ones(size=(T,T)))
# apply the tril on our weight_matrix, so we thanos snaped:d the right half values so each token can only
# talk to its past! (convinietly our weightmatrix is exactly the same dims as our tril!)
raw_wieghts_masked = weight_raw.masked_fill(tril_constrain==0, float('-inf'))
# apply sotmax to get probablity for each token,
# remember weight is (B,T,T) and we use the last dim, we could use -1 as well, 
# but I wanted to make it explicitly clear here!
weight = raw_wieghts_masked.softmax(dim=2) 
# and finally we can apply our weight on the input (we called it raw for a reason, read on)
# (by the way this is also called self-attention!)
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
# as you can see, our weight is initialized for each batch based on the input data, and each token has its own
# weight, that is they are not uniform!, to get a better understanding lets consider the first batch of 
# raw-weights[0], raw_masked_wieghts[0] and weight[0]:
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
# take the last token, which is 8.6020e-02, notice this token, not only knows its content and own position in 
# the sequence(its the 8th token after all!) but also knows which tokens comes before it and how much its 
# related to any of them.(basically when it knows which token comes before it, it creates its own query so
# to speak, and talks to every single previous token to find about which ones are more or less relavent to
# it and to what extend.) 
# For example,lets say, we have this sentence: "I saw the Eiffel Tower when I visited Paris" and suppose
# each token is a word, so I, saw, the, ... are all tokens, therefore we have a sequence of 9 tokens here. 
# In this sentence, if we focus on the word "visited", the attention mechanism allows it to understand that
# "visited" token is most closely related to "Paris". 
# This is because in the context of visiting, a location(in this case, "Paris") is usually involved.
# The attention mechanism assigns higher weights to "Paris" when processing the word "visited", indicating that
# "Paris" is important for understanding the context of "visited".
# Similarly, when the model processes "Eiffel Tower", it understands that "Eiffel Tower" is closely related to Paris",
# and thus assigns a higher weight to "Paris".
# This way, the attention mechanism allows the model to capture the dependencies between words in a sentence, 
# even if they are far apart. It’s like the model’s way of paying "attention" to important words/tokens while 
# processing a given word/token.
# now back at our own example, lets visualize the weights for better understanding
def plot_heatmap(weight, label, cmap='plasma'):
    import matplotlib.pyplot as plt 
    import seaborn as sns 
    import numpy as np 
    
    # lets draw the heatmap for each tensor 
    plt.figure(figsize=(12,6))
    sns.heatmap(weight.numpy(),cmap=cmap)
    plt.title(label)
    plt.show()
    
plot_heatmap(tril_constrain.detach(), label='tril_constrain', cmap='Blues')
# since we have a batch here, basically 4 samples, we flatten the first two dims 
# so we can plot all as a 2d heatmap 
plot_heatmap(raw_wieghts_masked.view(-1,8).detach(), label='raw_wieghts_masked-all batches',cmap='RdBu')
# now lets look at the first batch in more detail
plot_heatmap(raw_wieghts_masked[0].detach(), label='raw_wieghts_masked-first batch',cmap='RdBu')
# since the colors are not that defined, maybe gray cmap shows this better! but either are ok
plot_heatmap(weight[0].detach(),'weight-first batch',cmap='Blues')
plot_heatmap(bow_raw[0].detach(),'bow_raw-first batch')
# 
# in our example, For the last token, it seems the 2nd and 4th tokens (index 2 and 4 respectively) are particularly intresting/relavent/important,
# followed by the 6th(index 6) token.
# likewise, this happens for every token, i.e. token number 4, says the same thing and searches in its past toekns
# so on and so forth! 
# so what happens next is that when we get a high relevancy scale /response in the weights, like we just described
# when we do a softmax it will assign a large probablity to them, and this instructs the network that, we need more
# information from them, effectively allowing for aggregating a lot of their information into our position(lets say e.g. 8th token. 
# and we happen to learn more about them this way.
# now in practice, we are not intrested in aggregating the x raw values per say, rather we want their information, 
# so instead of just using the raw values of x, we instead use a representation of them, so to speak. 
# this is achieved using a third vector known as, 'value' and is the last vector we use. 
# !explain when we say vector, note that, we are talking from the prespective of a token, (every token has some vector 
# !of information and it gets to aggregate information via a weighted sum from all the tokens that point to it, and it
# )# !in practice we use a matrix, and hence the linear module to implement this to run the operation for all tokens in parallell. 
# just like the key and query, we set its bias to False, so we only get a simple vector,
# and a dotproduct output (again this is the intuition, but how much adding a bias would
# affect this we will see later, for now we stick to the default no bias version)
value = torch.nn.Linear(C,head_size, bias=False) # produces (B,T,head_size) just like the other two key,query vectors
x_processed = value(x)
# this is not yet final final, we still need to do one more thing (read on!)
bow_final = weight@x_processed
# !note that, as we previously pointed out, multiple times!, attention is a communication mechanism between tokens, and 
# by default there is nothing in this mechanism that provides a notion of space/position for tokens involved, i.e.
# by default these tokens/nodes/points, dont have any idea about where they are or how they are positioned (with resepect
# to others) etc, so we need to encode this information as well. (to better visualizing it, consider each token as 
# a node in a directed graph, each node has a connection to another node, (for example each node has a connection to
# itself, and another connection to other nodes,etc) and as you can see there is no notion of space here, theres no order here
# in the sense, which one do you call 1 or 2, etc when every node is connected to some other nodes and they are all connected,
# )the attention simply acts on a set of vectors in this graph, and thats why we need to encode them positionally 
# as well, so they have information about their position with regards to other tokens. 
# !check (compare this to the convolution case and images
# or even text where the spatial aspect of data is preserved, but in attention, as we just stated, there is no notion
# of position, its just a set of individual vectors being operated on)-note that the connection between nodes, do give
# us a sense of structure, but it may not be enough to infer the underlying semantic, imagine a case, where a word
# for example has several meaning, and may very well have high relevancy to some tokens at the same time, but without
# additional positional information, an ambiguous semantic can be infered between the tokens involved, however when
# positional information is also present, such ambiguity can be avoided)
# also note that, in attention, samples do not interact with each other at all. when we have a batch of 4, each sample
# is processed in isolation, but in parallell to other samples, based on our graph example earlier, we would have 4 
# graphs of 8 nodes for example for each input sample. 
#!
# we said earlier that what we implemented here is known as self-attention, the reason it is called self attention
# is that the key and query and values are applied on the same input(the use the same source!), and hence the name,
# self attention.
# also note that, in our specific case, tokens/nodes are can not communicate with the future nodes, but in general
# this constraint can be removed (and infact is removed/not implemented for some applications) where its benificial
# to be able to communicate with all the tokens. one example is sentiment analysis, where you want all the tokens to
# able to communicate with eachother so you can get an accurate analysis. and for this case, we would use an encoder
# block, which is basically what we have here, minus the constraint section(tril/mask part), what we have implemented
# here is called a decoder block, where we are decoding bunch of tokens, and it makes sense that the previous tokens
# do not comunicate with the future ones(becasue they would give the answer! and it defeats the whole purpose here!),
# becasue its a given that only the previous tokens must be used to predict the future/next token, hence the filtering/constraint part to prevent tokens from 
# comunicating with the future nodes/tokens.
# !so far we explained about the self-attention, which we saw, is called that way solely for the fact that key, query
# and value use the same source. the attention mechanism as we briefly pointed out, is much more general and can be
# used in different ways. one of such ways, is what is used to create sth called cross-attention. 
# cross-attention basically refers to the case where we have an encoder/decoder blocks, in which the queries come from x
# but the key and value come from an external source and sometimes from the encoder block. so cross-attention is used
# when theres a separate source of information we would like to pool from and use it as well.
#
# so far we implemented the attention based on the original paper(there are some differences we get to later on)
# except the part where we need to divide by the sqrt of the head_size. that is called scaled-attention
# so lets talk about this, and see why its needed. 
# the reason we add this so called 'scale' to our computation, is that, without it, the probablities will be saturated
# and when we add this term to the mix, it will make the 'weight' matrix to be 'unit variance', when Q and K are unit variance
# and this allows softamx to stay diffuse and not saturate too much.
# in other words, if we simply multiply key and query like that, the variance of the resulting weight matrix will be
# around the head_size instead of 1 which is bad and makes optimization really hard.
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
# why is it important? if you recall, the weight matrix is fed into softmax, so its really important, especially during
# initialization that weight matrix be fairly diffuse, if we look at weight matrix here, we'll notice that they are now
# fairly diffuse 
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
print(w_scaled.masked_fill(torch.tril(torch.ones(T,T))==0,float('-inf')).softmax(dim=-1)[0])
# tensor([[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.8026, 0.1974, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.1901, 0.4814, 0.3285, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.0499, 0.8038, 0.0987, 0.0476, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.1989, 0.2226, 0.1070, 0.2869, 0.1845, 0.0000, 0.0000, 0.0000],
#         [0.2148, 0.1049, 0.3329, 0.0440, 0.0661, 0.2374, 0.0000, 0.0000],
#         [0.3027, 0.0263, 0.0233, 0.1562, 0.0618, 0.2176, 0.2121, 0.0000],
#         [0.0901, 0.0035, 0.0312, 0.0962, 0.0029, 0.3478, 0.3901, 0.0382]])
#
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
print(w_unscaled.masked_fill(torch.tril(torch.ones(T,T))==0,float('-inf')).softmax(dim=-1)[0])
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
# and this is a recipe for disaster! the reason it is a problem is that, with softamx, when numbers take very positive
# and nagative values, softmax actually converges towards one-hot-vector. basically the values shrink towards the max.
# this can be seen the above, but the example below should demonstrate it more clearly.
# lets apply softmax on a list of numbers that are close to each other, we see the probablities are difuse and normal
print(f'{torch.softmax(torch.tensor([0.1,0.5,-0.3,-0.2]),dim=-1)}')
# we get a diffused probablity out of softmax
# tensor([0.2562, 0.3822, 0.1717, 0.1898])
# however if we increase the magnitude of the numbers(sharpen them) (and thus the difference between them) by like multiplying by 
# a number like 8 (just to simulate the effect here and so we can compare it with the previous case)
print(f'{torch.softmax(torch.tensor([0.1,0.5,-0.3,-0.2])*8,dim=-1)}')
# we see that the softmax, starts to sharpen towards the max, shrinking others except the 
# largest number, and effectively
# converging toward a one-hot-encoded vector.
# tensor([0.0390, 0.9559, 0.0016, 0.0035])
# so we dont want these values to be extreme, especially during initialization or otherwise, softmax will be way too picky!
# and we are basically aggregating the information from a single node instead of multiple ones (becasue one has the largest
# nvalue, its as if our sequence length is 1! and we lose access to the wealth of information the past history offers)
# so we want the probablities to be diffuse and not peaked like the second example here.
# so this scaling is used to retain the variance at a good value especially at initialization.
# so now that we are finally finished the self attention head, lets implement it as a module and incorporate everything
# we just discussed here.
class AttentionHead(nn.Module):
    def __init__(self, context_size, embd_size, head_size=16, use_bias=False) -> None:
        super().__init__()
        # we need context_size or block_size for creating the tril constrain
        self.context_size = context_size
        # we want this as the input dim for our key,query and value, this is 
        # infact the input_dim, since we plan on using the embeddings, we named
        # it embd_size
        self.embd_size = embd_size
        # head_size is the output dimension of our attnetion
        # !note that usually the head_size is equal to embd_size
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
        #! check the shapes!! 
        B,T,C = inputs.shape # 4,8,16
        # print(f'{inputs.shape=}')
        # create the weight by using k,q,v
        k = self.key(inputs)    #! (B,T,C) or (B,E,16)? (4,8,16)
        q = self.query(inputs)  #! (B,T,C) or (B,E,16)? (4,8,16)
        v = self.value(inputs)  #! (B,T,C) or (B,E,16)? (4,8,16)
        # create the weight matrix and scale it by 1/sqrt(head_size) to keep weight unit variance 
        weight = q@k.transpose(-2,-1)* self.head_size**-0.5 # !(B,E,E)
        # weight_C = q@k.transpose(-2,-1)* C**-0.5 # !(B,E,E)
        # print(f'weight.var: {weight.var().item():.4f}')
        # print(f'weight_C.var: {weight_C.var().item():.4f}')
        # print(f'x:{tuple(inputs.shape)} k:{tuple(k.shape)} q:{tuple(q.shape)} v:{tuple(v.shape)} w:{tuple(weight.shape)} hs:{self.head_size}')
        # apply the tril constrain - (this makes this a decoder block!)
        # important note: notice we used tril[:T,:T] and not simply tril
        # this is because, when the input has a small context_size < self.context_size
        # like when we wantto generate inputs with sth like zeros((1,1)) which says
        # there is a single token (context_size 1) of 0 as the begining of the sequence
        # when this is input, the tril by default makes a context_size,context_size matrix
        # which will be different than the input context_size which is 1 e.g. or 2 e.g.
        # and it will fail becasue our weight would be 1,1,1 or 1,2,2, but trail is 1,8,8
        # and clearly this will cause an error. for this reason, we always create the 
        # tril check dynamcally by explicitly specifying the context_size based on the 
        # current input context_size so in case the context_size is smaller, tril is resized
        # dynamically accordingly. 
        # print(f'tril[:T,:T]==0: {(self.tril[:T,:T]==0).shape}')
        # print(f'tril==0: {(self.tril==0).shape}')
        weight = weight.masked_fill(self.tril[:T,:T]==0,float('-inf'))
        # weight2 = weight.masked_fill(self.tril==0,float('-inf'))
        # print(f'{weight.shape=}')
        # print(f'{weight2.shape=}')
        # note that we are using batch, so instead of hardcodin 2,
        # we use -1 to refer to the last dim
        weight = weight.softmax(dim=-1)
        # finally apply the weight on the v
        bow = weight@v  # !(B,E,16)
        return bow        

# now lets add this to our model 
        
# so to recap, we first calculated the relavancy between all tokens against eachother, then constrained them so that 
# each token can only use the information from/interact with its past tokens. then since we needed probablity distribution so
# we then normalized it and then used that to pickout which tokens information(in the past) to aggregate/use with the inputs to 
# achieve our goal.(which is to predict the next character based on everything seen so far!)
# we dont want to aggregate inputs value (with respect to the weight matrix), we instead would like to use their representation
# so we use a new layer to do this, its called value, and we instead use its output instead of inputs raw value.
#
# 
#%%
# TODO:  add the efficient/fused version 
# lets check  to see if  fusing kqv can improve our speed!
# this works, but one might think its not really that efficient provided that usually when we have multiple operations, its much better
# to create 1 larger operation than several smaller one, and when it comes to multiplication, we can do better
# for example, we can calculate k,q,v in one go! we need to merge their weights, do the calcs, and then split
# the result! well, lets first see how its done and then do a simple benchmark to see if it actually is any better!
# to give you an intuitive undrestanding try the following example ,
# suppose we have our k,q,v layers
# key = nn.Linear(5,5,bias=False)
# query = nn.Linear(5,5,bias=False)
# value = nn.Linear(5,5,bias=False)
# x = torch.randn(size = (3,2,5))
# and we want to calculate their outputs, before we do the multiplication, lets make sure their weights
# are the something fixed so we can easily see it for ourselves whats going on. 
# since the key,query and values's weights are 5,5, we have 25 values for each so lets initilize them with 0-25!
# key.weight.data   =  torch.arange(0,25).view(5,5).float()
# query.weight.data =  torch.arange(0,25).view(5,5).float()
# value.weight.data =  torch.arange(0,25).view(5,5).float()
# now lets calculte their outputs :
# k,q,v = [m(x) for m in (key, query, value)]
# now we said that we can do a single multiplication instead of 3 by merging the weights of key,query and
# value how do we do that? we simply create a linear layer with 3 times the output size of the initial size
# used for k,q,v layers. 
# kqv = nn.Linear(5,15, bias=False)
# now if we calculate the output of kqv, we should see the outputs match
# kqv_output = kqv(x)
# but the shape is (3,2,15)! shapes dont match! how are we going to compare them then? 
# no problem we split the last dimension into 3 seprate tensors!
# k2,q2,v2 = torch.split(kqv_output, 5, dim=-1)
# and now if we do 
# (k==k2).all(), (q==q2).all(), (v==v2).all()
# we get True,True,True , signifying they are actually doing the very same thing!
# now that we know how to implement this, lets implement our attention head using this new trick!
# and see if its any faster! 

# side note about Autoregressive models : https://www.youtube.com/watch?v=vwG3KWzuACo
# the class of models we are tyring to implement here is called autoregressive models and they 
# are shown to outperform recurrent networks (rnns/lstm/gru/etc)
# an autoregressive model is a feedforward model that predicts the next variable xt  in a time series
# based on k previous variables (xt-1, xt-2,...)
# in RNNs, the parameters are shared across time (same function (f(.) at different t))
# while Autoregressive models, make a strong conditional independence assumption. 
# watch the video to have a clue!

class AttentionHead2(nn.Module):
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
at2 = AttentionHead2(cs,es,hs).cuda()
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

# as you can see we get contradictory results. and its expected. 
# In general, fusing operations into a single operation can potentially improve performance by 
# reducing memory access and overhead associated with separate operations. However, it's important
# to consider the trade-off between fusion and memory usage. 
# Fusing all the operations into a single linear layer (kqv) and then splitting the output may reduce
# memory access and overhead, but it may also lead to increased memory consumption due to the larger 
# output tensor.
# The efficiency of fusion and splitting depends on factors like the size of the input, the dimensions of
# the layers, and the available hardware resources. 
# Thereore It's always recommended to benchmark and compare the performance of different approaches on our
# specific hardware and input size to determine the most efficient solution.
# In some cases, hardware optimizations like tensor cores on certain GPUs or specialized operations can 
# further impact the performance. Therefore, it's beneficial to consider hardware-specific optimizations
# and consult the hardware documentation or performance guides for guidance on optimizing specific 
# operations.
# In summary, whether fusion provides better performance and if it's more efficient to use a single linear
# layer and split the output can depend on various factors, including the specific operations, input size,
# layer dimensions, and hardware being used. It's recommended to benchmark and compare different approaches 
# to determine the most efficient solution for your specific scenario.

# We then compare the number of parameters in the original layers and the fused layer.
# In general, fusing layers can be beneficial when the input size is large, and the number 
# of parameters in the fused layer is smaller than the sum of the parameters in the original layers. 
# This is because the fused layer requires fewer memory accesses and computations than the original 
# layers.


#%%
#
# before we jump in and add the attention module, lets review our base model and see how we can 
# improve/prepare it before we incorporate the attention. 
# class BigramModelWithAttention(nn.Module):
#     def __init__(self, vocab_size, embd_size) -> None:
#         super().__init__()
#         self.vocab_size = vocab_size
#         self.embd_size = embd_size
#         # unlike the previous model, lets decouple the final logits from 
#         # the number of embeddings, because we are using attentions, and
#         # we want to have multiple operations inbetween obviously.
#         self.embeddings = torch.nn.Embedding(vocab_size, embd_size)
#         # in order to get the final logits, we need a linea layer at end
#         self.fc = torch.nn.Linear(embd_size, vocab_size)
#        
#     def __call__(self, inputs:torch.Tensor, labels=None) -> torch.Tensor:
#         out = self.embeddings(inputs) # has the shape (B,T,E) e is embd_size
#         logits = self.fc(out)         # has the shape (B,T,C) c is vocabsize
#         loss = None
#         if labels is not None:
#             # recall that crossentropy likes its input to be B,C,T and we are B,T,C
#             # so lets permute and make it happy!
#             loss = F.cross_entropy(logits.permute(0,2,1), labels)
#         return logits, loss
#    
#     def generate(self, idxs, max_token_count)-> list[torch.Tensor]:
#         # lets generate an output as long as num_max_token
#         for i in range(max_token_count):
#             # make sure idx is 2d
#             assert len(idxs) >1, f"idx.shape '({tuple(idxs.shape)})' is invalid. it must have the form (B,T)"
#             # now lets feed it to the model and sample from the probablities it produces
#             preds,_ = self(idxs)
#             # convert to probs 
#             probs = preds.softmax(dim=1)
#             # since we are bigram still, lets only get the last token as the next token predicted!
#             probs = probs[:,-1,:]
#             # now lets sample from it 
#             new_idx = torch.multinomial(probs, num_samples=1, replacement=True)
#             # now concatenate the new token to the previous one and feed it back to the model
#             # for the next round of prediction
#             # also remember that we are creating a sequence, so we concat them at dim=1 to get 
#             # a longer sequence (we are gradually increasing the sequence length from 1 up to
#             # max_token_count)
#             idxs = torch.cat((idxs,new_idx), dim=1)
#            
#         return idxs

# this works fine, however, we can do better. here we just encoded the tokens, but
# as we already explained, attention mechansim has no notion of position or spatial structure like convs,
# or rnns e.g. 
# when dealing with rnns the sequential nature of the input is preserved, as each word/token is
# processed one after another, so the order would be preserved. 
# from shaw etal 2018: 
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

# so this is not the case here as we are using attention mechanism as well,
# and hence we need to encode the position information as well. 
# good refs for positional embeddings : 
# https://medium.com/@hunter-j-phillips/positional-encoding-7a93db4109e6
# this blogpost does a very good job at explaining the implementation of the sinusoidal positional encoding
# and pretty much explains all the questions concerning the formula and why its implemented a certain way. 
# https://towardsdatascience.com/master-positional-encoding-part-i-63c05d90a0c3
# this blog post,does a very good job at explaining the intuitions behind the sinusoidal positional encoding.
# Ive watched and read alot of videos and explanations on this, some videos(also linked below) are good some
# arent as they say things that are not backed, I tried to ask and answer them using different sources I find
# but these two links that I wrote here, do a good job. however, read the following information aswell. 
# finally sinusoidal positional embedding isnot used anymore, instead the learned positions are used (this is
# what we implemented in our example, and BERT uses it, but sinusoidal posintioning had a lot of intresting
# intuitions and ideas that can give me/you a new prespective and possibly allow you to learn and comeup with
# similar improvements knowing the concepts/reasons behind it)
#
#  
# Side notes: 
# Q:Whats an embedding intuitively?
# Understanding Embeddings Intuitively
# Embeddings can be thought of as a way to represent complex, high-dimensional data in a more simplified, lower-dimensional space while maintaining meaningful relationships between the data points. It's like capturing the essence of something intricate in a simpler form that retains its essence or crucial characteristics.
# Simplification with Meaning:
#     Analogous to Maps:
#         Just as a map condenses geographic information into a flat surface without losing the relative positions of cities, embeddings condense data without losing crucial relationships.
#     Capturing Essence:
#         Think of it as distilling the essence of a painting into a smaller sketch that still captures the main elements and style.
#     Relationships Preservation:
#         Imagine organizing a library: embedding ensures that books on similar topics are placed nearby, making it easier to find related information.
# Practical Examples:
#     Word Embeddings:
#         Words represented as vectors: closer words in the embedding space often have similar meanings or usage contexts.
#     Image Embeddings:
#         Images represented in a lower-dimensional space where similar images are closer together, aiding tasks like image similarity search.
# Key Takeaway:
# An embedding, intuitively, is a condensed representation that retains essential information or relationships while simplifying the complexity of high-dimensional data. It's like summarizing a story without losing its essence or key plot points.
#
# Q:Whats a manifold intuitively:
# Understanding Manifolds Intuitively
# A manifold, intuitively, can be visualized as a flexible and curved surface embedded in a higher-dimensional space. In the context of embedding manifolds, it refers to the reduced-dimensional space where data points are situated after the embedding process.
# Visualizing Manifolds:
#     Flexible Surface:
#         Picture a rubber sheet that can bend and curve. The manifold is like the surface of this sheet, able to take on various shapes within the higher-dimensional space.
#     Embedded in Higher-Dimensional Space:
#         Imagine the rubber sheet exists within a 3D space, but the manifold itself is a 2D surface. The embedding process places data points on this flexible surface.
# Analogies for Understanding:
#     Paper Map Analogy:
#         Just as a paper map represents a curved Earth's surface on a flat sheet, a manifold represents complex data in a lower-dimensional space.
#     Hiking Trails Analogy:
#         If you think of a landscape with hills and valleys, the trails can be seen as the manifold, navigating the terrain while being constrained by the landscape's overall structure.
# Connection to Data Representation:
#     Data Points on the Surface:
#         In the manifold, each point represents a data point. The goal is to position these points on the surface so that relationships between them are preserved from the original, higher-dimensional data.
#     Curvature and Relationships:
#         The curvature of the manifold reflects the relationships between data points. Smooth curves indicate similar relationships, while abrupt turns may represent significant changes in the data.
# Key Takeaway:
# A manifold, intuitively, is a flexible, curved surface in a higher-dimensional space. In the context of embedding manifolds, it serves as the reduced-dimensional space where data points are positioned after undergoing the embedding process, capturing essential relationships in a more manageable form.
#
# explain more: 
# Sure, diving deeper into the concept of embedding manifolds involves understanding how data points are transformed from a high-dimensional space to a lower-dimensional one while preserving their intrinsic relationships.
# Mathematics of Manifolds:
#     Topological Spaces: A manifold is a topological space that looks locally like Euclidean space, meaning that in a small enough region, it resembles a familiar space like a plane.
#     Intrinsic Properties: It retains certain intrinsic properties, such as local linearity or smoothness, even if embedded in a higher-dimensional space.
# Embedding Process:
#     Dimensionality Reduction: The primary goal is to reduce the dimensions while retaining relevant information. This is achieved by finding a way to project the data onto a lower-dimensional surface while preserving the structure and relationships within the data.
#     Preserving Relationships: Techniques used, like PCA or t-SNE, aim to maintain the proximity or similarity between data points. Similar points in the original space should remain close in the lower-dimensional manifold.
# Understanding the Manifold's Shape:
#     Local Structure Preservation: The manifold captures local structures or relationships among data points. Think of it as preserving how nearby points relate to each other.
#     Global Shape: Understanding the overall shape of the manifold is crucial. It might have complex twists, turns, and curvatures, reflecting intricate relationships among data points.
# Challenges and Considerations:
#     Curse of Dimensionality: As the dimensionality decreases, some information may be lost. Finding an optimal balance is essential to retain meaningful information.
#     Choosing the Right Technique: Different techniques suit different types of data. Linear techniques like PCA might not capture nonlinear relationships, which methods like t-SNE can handle.
# Practical Applications:
#     Machine Learning: Embedding manifolds play a vital role in various ML tasks, like clustering, classification, and visualization, where high-dimensional data needs to be comprehensively analyzed.
#     Semantic Understanding: In natural language processing, embeddings help understand word relationships, sentiments, and semantic meaning.
# Conclusion:
# An embedding manifold is a reduced-dimensional space where complex high-dimensional data is transformed while maintaining essential relationships. Achieving this involves a delicate balance between reducing dimensions and preserving meaningful information, crucial for various applications across multiple domains.
#
#
#
#
# some attributes concerning positional encodings : https://www.youtube.com/watch?v=1biZfFLPRSY
# 1.Every position should have the same identifier regardless of the sequence length or what the input is
#  so when the input changes, the position embedding remains the same)
# 2. since these positinal embeddings push the token/embedding towards a 'specific positional cluster' they should not 
#  be too large otherwise they will push vectors into very distinct subspaces where the positional similarity
#   or dismilarity overshadows/overtunes the semantic similarity. 
#   lets expand on this a little bit and make it more
#   intuitive, when we plot the token embeddings (after they are learned), you'll notice that tokens/words with similar
#   context are clustered together, you find queen and king to be next to eachother, car, tire, steering wheel to cluster
#   together, etc, showing words used together or in similar context bunchup as a cluster. 
#   when we add the positional embedding which really is another vectors with number, what happens is
#   during trainig, the network does the same thing with these new information as well, that is, the queen/king token
#   in the feature space, will be ever-so slightly moved to a distinct subregion where their index refers to.
#   imagine, king is the first token, so it gets closer to the 'first token's cluster, the network creates other clusters
#   for other positions likewise, and as you can see, if the positional embedding is too large, it can disrupt the semantic
#   clustering of tokens.
#   it will create new clusters based on the position information only (clusters of first, seconds, third tokens, etc) at the
#   expense of destroying the semantic clusteres created due to similar contexts/etc, thus degrading the performance altogether
#   as you can imagine, having a too little impact from positional info or lack of any, we are only left with semantic clusters
#   and lose any kind of incentive for the model to use to make a disntinction between two sequence of the same tokens (becasue
#   theres no solid notion of order) so we need to comeup with something that satisfies these conditions.) 
#
# the paper uses something called sinusodal positional encoding, but why? what if we use simple numbers to 
# index the positions since we can treat text as words coming sequentially? so why not count the tokens as
# we go like 1,2,3,4,etc? 
# we dont use that becasue it violates the second requirement of the positional encoding, where we said
# change/shift/translations need be small! or in other words, bounded. 
# Think about it for a second, if the later tokens has larger
# numbers assigned to them, then the model assigns more importance to them as apposed to the earlier tokens with 
# smaller numbers, the contribution of later tokens will be emphasized while the earlier tokens can be just ignored
# if the sequence lengths are large enough and hence hinder the model performance, 
# thats why the paper's author opted out to use sin/cos
# for their functions of choice for positional encoding, as they are both bounded to -1,1 and they are 
# inifite(periodically result in [-1,1])! that is can support any sequence length up to inifity (as we can
# have values between -1 and 1 for infinity)! you might think about something like sogmoid which also is
# bounded to [0,1], but sigmoid is a saturating function and the actual range of numbers is pretty limited 
# especially for larger numbers(test it with 1-10 e.g. and larger numbers!) unlike the sin/cos that has a lot of variability for large numbers,
# the sin/cos function has its own problem, the problem is they are periodic! that is, the same number
# repeats for different positions! and this violates the first requirement of identifiers being unique for each position!
# the problem stems from its 'frequency'! if we lower the frequency it means it takes longer to repeat a number twice
# the higher the frequency the shorter the range of numbers. In other words, The low-frequency sine wave (sin(x))
# has fewer oscillations within a given range of x, while the high-frequency sine wave (sin(10x) e.g.) has more 
# rapid oscillations within the same range.
# look at the following example to see this better.
# %%
#
import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors
import numpy as np 

# lets create 1000 points in the span of (0,4pi), 
# for refresher see https://www.mathsisfun.com/algebra/trig-sin-cos-tan-graphs.html and 
# https://www.math.net/sinusoidal
# we use pi, instead of degrees (0,360) here becasue np.sin() uses radian instead of degrees
x = np.linspace(0, 4*np.pi, 1000)
# low_frequency_s = np.sin(x)
# high_frequency_s = np.sin(10*x)
# instead lets draw both for sin and cos
freqs = [f(x) for x in [x, 5*x] for f in [np.sin, np.cos]]
plt.style.use('seaborn')
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
#%%
# 
# now if we manage to lower the frequency low enough to the point where it gives us a huge preferably inifinit
# numbers, that would be like our previous suggetsion, linear range of numbers, but bounded this time!
# but the second issue still exists, that is, these numbers still, when added to the embeddings, will push them
# !into specific subregions in the feature space(really this means, they cluster up together somewhere!)
# and we dont want to push this too much, as otherwise the distance between the two adjacent tokens will be overwhelemed
# by this position and not the semantic between the two adjacent tokens. we also dont want the difference between two tokens
# be too little, as this would cause the semantic distance to overwhelm/dominate the positional distance/information, 
# basically render it useless! so what should we do? 
# if we go and use sin only, the adjacent numbers would be close to each other, we need to somehow create a bit 
# of contrast that shows theres some kind of distance between the two tokens (and by extension others in the sequence)
# so what do we do? if you look at the plots, you'll notice that, intrestingly we can utilize cos() for this
# and alternate between them. if we use a low frequency sin/cos for the first token, we can use a higher frequency 
# sin/cos for the next token, and for the third token, we can use sin again, followed by a cos for the forth and so on 
# and so forth! the more we increase the frequency, the more different the values we get between two tokens. 
# note that
# we create a single positional vector by alternating sin/cos for every feature/dimension of the vector, so if we have d=100,
# we fill the even indexes using sin() and the odd ones with cos(), this way, we get a unique value for each position
# !and by increasing the frequency as we go, we make sure .... continue explanation
# 
# !fact check these answers
# question : why are sin and cos interleaved/alternated like this whats the intuition or reason behind it? 
#
# The reason for interleaving sine and cosine functions in sinusoidal positional encoding is to capture 
# different frequencies and phases in a systematic manner. This interleaving pattern ensures that each 
# dimension in the encoding captures unique positional information.
# By alternating between sine and cosine functions, the encoding matrix can represent different frequencies 
# and phases. The sine function captures the position-dependent changes with a periodic pattern, while the 
# cosine function captures the position-independent changes with a constant pattern.
# The intuition behind this interleaving pattern lies in the properties of sine and cosine functions. 
# The sine function is an odd function, meaning it is symmetric about the origin, while the cosine function 
# is an even function, meaning it is symmetric about the y-axis.
# When we use these functions in positional encoding, the interleaving ensures that each dimension captures 
# different positional information. The even-indexed dimensions (0, 2, 4, etc.) capture position-independent
# changes with cosine functions, which remain consistent regardless of the position. 
# The odd-indexed dimensions (1, 3, 5, etc.) capture position-dependent changes with sine functions, which 
# introduce periodic variations as the position changes.
# By combining both sine and cosine functions in this interleaved manner, the positional encoding can effectively
# represent different frequencies and phases across the dimensions. This allows the model to differentiate 
# between positions and capture the relative ordering of elements in the sequence.
# Overall, the interleaving of sine and cosine functions in sinusoidal positional encoding ensures that each 
# dimension captures unique positional information, utilizing the properties of these trigonometric functions
# to represent frequencies and phases.
#
# question :why do we need to have postion-independent changes? isnt sin enough to capture position dependent inormation for our task?
# You raise a valid point. In some tasks, using only sine functions might be sufficient to capture position-
# dependent information. The inclusion of cosine functions in positional encoding is not strictly necessary 
# for all tasks. It depends on the specific requirements and characteristics of the problem at hand.
# The reason for including cosine functions alongside sine functions in sinusoidal positional encoding is to
# provide a richer representation of positional information. While sine functions capture position-dependent
# changes with a periodic pattern, cosine functions capture position-independent changes with a constant pattern.
# In some cases, position-independent information can be valuable for the model to learn and reason about the
# sequence. It can provide a reference point or a baseline for the model to understand how each position deviates
# from a standard or average position. By incorporating both sine and cosine functions, the model can potentially
# gain a more comprehensive understanding of the sequence and its positional relationships.
# However, it's important to note that the necessity of including cosine functions may vary depending on the 
# task. In certain scenarios, using only sine functions may be sufficient, especially when the position-dependent
# changes are the primary focus.
# Ultimately, the decision to include cosine functions in the positional encoding can be considered as an 
# architectural choice, and it may require experimentation and exploration to determine the optimal approach 
# for a specific task.
#
# can you explain the first paragraph more? like which tasks for example?
# Certainly! The necessity of including cosine functions alongside sine functions in positional encoding can 
# vary depending on the task at hand. Here are a few examples to illustrate when including position-independent
# changes might be beneficial:
# 1. Machine Translation: In machine translation tasks, the relative positions of words in the source and target
#    sentences are crucial for accurate translation. By including cosine functions in positional encoding, the 
#    model can capture position-independent information, such as the difference between the first and last words
#    in a sentence or the average position of words. These position-independent changes can provide valuable 
#    contextual information for the translation process.
#
# 2. Speech Recognition: In speech recognition tasks, the timing and ordering of phonemes or speech segments 
#    play a vital role. By incorporating cosine functions in positional encoding, the model can capture 
#    position-independent information, such as the overall duration or rhythm of a speech sequence. This can
#    help the model understand the temporal relationships between different speech units.
#
# 3. Natural Language Understanding: In tasks involving natural language understanding, such as sentiment analysis
#    or question answering, the relative positions of words or phrases can be important for context comprehension.
#    By including cosine functions, the model can capture position-independent information, such as average position
#    or general positional trends, which can aid in understanding the overall context or sentiment of a sentence.
#
# 4. Music Generation: In tasks related to music generation, such as composing melodies or generating harmonies,
#    positional information is crucial for creating coherent musical sequences. Including cosine functions alongside
#    sine functions enables the model to capture both position-dependent variations (captured by sine functions)
#    and position-independent patterns (captured by cosine functions), helping to generate musically meaningful 
#    sequences.
# These are just a few examples where including position-independent changes through cosine functions in 
# positional encoding can be beneficial. However, it's important to note that the specific requirements and
# characteristics of each task may vary, and the decision to include cosine functions should be based on 
# empirical evaluation and experimentation, considering the specific nature of the task and dataset.
#
# what are more examples?
# Certainly! Here are a few more examples where including cosine functions in positional encoding can be beneficial:
# 1. Image Captioning: In image captioning tasks, the relative positions of objects or regions within an image can provide important contextual information. By including cosine functions in positional encoding, the model can capture position-independent information, such as the global layout or arrangement of objects in the image. This can aid in generating more accurate and contextually relevant captions for the image.
# 2. Document Classification: In document classification tasks, the position of words or paragraphs within a document can be indicative of the document's topic or structure. By incorporating cosine functions in positional encoding, the model can capture position-independent information, such as the overall organization or hierarchy of the document. This can assist in better understanding the document's content and making more informed classification decisions.
# 3. Time Series Forecasting: In time series forecasting tasks, the temporal ordering and duration of events play a critical role in predicting future values. By including cosine functions in positional encoding, the model can capture position-independent information, such as the overall trend or seasonality in the time series. This can help the model understand long-term patterns and make more accurate predictions.
# 4. Video Processing: In video processing tasks, the temporal ordering and relationships between frames are crucial for tasks like action recognition or video captioning. By incorporating cosine functions in positional encoding, the model can capture position-independent information, such as the overall motion or rhythm in the video sequence. This can aid in recognizing actions or generating coherent captions for the video.
# These are just a few additional examples where including cosine functions in positional encoding can be beneficial. The specific applicability of cosine functions may depend on the characteristics and requirements of the task, as well as the nature of the data being processed. It's important to experiment and evaluate different encoding strategies to find the most suitable approach for each specific task.
#
# in which task this doesnt help?
# Including cosine functions in positional encoding may not provide significant benefits or may not be necessary in certain tasks where position-independent changes are not as relevant. Here are a few examples where the inclusion of cosine functions may not be as helpful:
# 1. Sequence Tagging: In tasks like part-of-speech tagging or named entity recognition, the primary focus is on capturing position-dependent information. The model needs to understand the relative positions of words to assign appropriate tags. In such cases, using only sine functions for positional encoding may be sufficient, as they can effectively capture position-dependent changes without the need for cosine functions.
# 2. Sequence Generation: In tasks like text generation or image synthesis, the primary objective is to generate new sequences or samples based on the given input. The position-independent changes may not play a significant role in the generation process, as the focus is on capturing the distribution of data and generating coherent sequences. In such cases, using only sine functions for positional encoding may be adequate.
# 3. Sequence Classification: In tasks like sentiment analysis or document classification, the position-independent changes may have minimal impact on the classification task. The model primarily needs to understand the local context and relationships between words or phrases. In these cases, using only sine functions for positional encoding may be sufficient to capture the position-dependent information.
# It's important to note that the applicability of cosine functions in positional encoding depends on the specific task and dataset characteristics. While including cosine functions can provide a richer representation of positional information in many cases, there may be scenarios where the position-independent changes captured by cosine functions are not as relevant or impactful. It's always recommended to experiment and evaluate different encoding strategies to determine the most suitable approach for a given task.
#
# can you explain this part in depth?
# "While sine functions capture position-dependent changes with a periodic pattern, cosine functions capture position-independent changes with a constant pattern."
# Certainly! Let's dive deeper into the explanation of how sine and cosine functions capture position-dependent and position-independent changes, respectively:
# Sine Functions: Sine functions are trigonometric functions that oscillate between -1 and 1 in a periodic manner. The sine function is an odd function, meaning it is symmetric about the origin. It captures position-dependent changes because it introduces variations based on the position within the sequence. As the position changes, the sine function produces different values, resulting in a periodic pattern. This periodicity is useful for capturing the relative ordering and position-dependent information in a sequence.
# Cosine Functions: Cosine functions are also trigonometric functions that oscillate between -1 and 1, but they have a different pattern compared to sine functions. The cosine function is an even function, meaning it is symmetric about the y-axis. Unlike the sine function, which captures position-dependent changes, the cosine function captures position-independent changes. It produces a constant pattern that remains the same regardless of the position within the sequence. This pattern provides a reference point or a baseline for the model to understand how each position deviates from a standard or average position. It helps capture position-independent information that may be relevant for certain tasks.
# To summarize, sine functions capture position-dependent changes by introducing variations and a periodic pattern based on the position within the sequence. On the other hand, cosine functions capture position-independent changes by providing a constant pattern that remains the same throughout the sequence. This combination of sine and cosine functions in positional encoding allows the model to capture both position-dependent and position-independent information, providing a more comprehensive representation of positional relationships within a sequence.
#
# Question: sin and cos are periodic functions, i.e. they repeat their values, so how is this not a problem in this case? how do they address this issue?
#
# You are correct that sine and cosine functions are periodic, meaning they repeat their values over a specific interval. This periodicity could potentially pose a problem if we were to use these functions directly as positional encodings. However, in the context of sinusoidal positional encoding, the periodic nature of sine and cosine functions is actually leveraged in a way that addresses this issue effectively.
# In sinusoidal positional encoding, the position within the sequence is encoded using a combination of sine and cosine functions with different frequencies and phases. The frequency determines how quickly the function oscillates, while the phase determines the starting point of the oscillation.
# By using different frequencies and phases for each dimension in the encoding, the sinusoidal positional encoding achieves a unique representation for each position in the sequence. This means that even though sine and cosine functions are periodic, the combination of different frequencies and phases ensures that the positional encoding values do not repeat exactly for each position.
# For example, consider encoding a sequence of length L. Each position in the sequence corresponds to a unique set of frequencies and phases for the sine and cosine functions. As a result, the encoding values for each position will have a distinct combination of sine and cosine values, ensuring that the positional information is uniquely represented.
# Therefore, the periodicity of sine and cosine functions is not a problem in sinusoidal positional encoding because the combination of different frequencies and phases allows for the creation of unique positional encodings that capture the necessary positional information without exact repetition.
# Overall, the use of different frequencies and phases in sinusoidal positional encoding effectively addresses the periodic nature of sine and cosine functions, ensuring that the encoding values for each position are distinct and provide unique representation within the sequence.
#
# if we used a low frequency with sin, it would provide us a large range of numbers, wouldnt that alone be enough?
# Using a low frequency with sine functions can indeed provide a larger range of numbers, but it may not be sufficient on its own to capture all the necessary positional information. While it can help differentiate between positions, it doesn't take into account the phase or timing of the position within the sequence.
# Including cosine functions alongside sine functions in sinusoidal positional encoding adds an additional dimension that captures position-independent changes. The cosine functions provide a constant pattern that remains the same throughout the sequence, allowing the model to understand the relative position of each element with respect to a reference point or baseline.
# By combining sine and cosine functions, sinusoidal positional encoding captures both position-dependent and position-independent changes. This richer representation allows the model to not only differentiate between positions but also understand the overall context and relationships between different positions within the sequence.
# Additionally, using a low frequency with sine functions alone could lead to potential overlaps or interference between positional encodings, especially in longer sequences. Incorporating multiple frequencies and phases for both sine and cosine functions helps ensure that the positional encodings are distinct and can be easily distinguished by the model.
# So while a low frequency with sine functions can provide a larger range of numbers, including cosine functions alongside sine functions enhances the positional encoding by capturing position-independent changes and ensuring a more comprehensive representation of positional information.
#
# Q: can you elaborate more on how using a low frequency with sine functions alone could lead to potential overlaps or interference between positional encodings?
# Certainly! Let's dive deeper into why using a low frequency with sine functions alone could lead to potential overlaps or interference between positional encodings.
# When using sine functions for positional encoding, the frequency determines how quickly the function oscillates. A low frequency means that the sine function will complete fewer oscillations over a given range of positions. As a result, the values of the sine function will change more slowly as the position increases, leading to a larger range of numbers.
# While this can help differentiate between positions to some extent, it may not be sufficient to capture all the necessary positional information, especially in longer sequences. Here's why:
# 1. Overlapping Encodings: In longer sequences, using a low frequency with sine functions alone can result in overlapping encodings. Since the sine function changes slowly, adjacent positions in the sequence may have similar or overlapping encoding values. This can lead to a loss of distinctiveness between positions, making it harder for the model to differentiate between them accurately.
# 2. Lack of Precision: Using a low frequency alone may not provide enough precision to capture subtle positional differences. The encoding values may not be fine-grained enough to accurately represent the relative positions within the sequence. This lack of precision can limit the model's ability to understand the precise relationships between elements in the sequence.
# 3. Limited Contextual Information: Sine functions alone do not capture position-independent changes or provide contextual information. They only capture position-dependent changes. By incorporating cosine functions in addition to sine functions, sinusoidal positional encoding captures both position-dependent patterns (sine functions) and position-independent patterns (cosine functions). This allows the model to understand the overall context and relationships between different positions in the sequence.
# By including multiple frequencies and phases for both sine and cosine functions, sinusoidal positional encoding ensures that each position has a distinct encoding value. This helps prevent overlaps or interference between positional encodings and provides a more comprehensive representation of positional information.
# In summary, using a low frequency with sine functions alone may result in overlapping encodings, lack of precision, and limited contextual information. Incorporating multiple frequencies and phases, including the use of cosine functions, in sinusoidal positional encoding addresses these issues and provides a more effective representation of positional information.
#
#
# Q: why does taking into account the phase or timing of the position within the sequence important?
# Taking into account the phase or timing of the position within the sequence is important because it provides additional information about the relative ordering and relationships between elements in the sequence. Here's why phase or timing is crucial in positional encoding:
# 1. Capture Sequential Dependencies: The phase component of the positional encoding helps capture sequential dependencies between elements in the sequence. It specifies the starting point or reference for the oscillation of the sine and cosine functions. By incorporating phase information, the model can understand the sequential order of the elements and how they relate to each other in the context of the task. This is particularly essential in tasks where the order of the elements carries significant meaning, such as natural language processing tasks or time series analysis.
# 2. Encode Relative Positional Information: Phase information enables the model to encode the relative positional information of elements within the sequence. It indicates how far along the sequence an element is compared to others. By considering the phase, the model can differentiate between positions and understand the relative distances or intervals between elements. This is crucial for tasks that require understanding positional relationships, such as machine translation or sentiment analysis.
# 3. Differentiate Positions with Same Frequency: When using sine functions alone, different positions can have the same frequency but different phases. By incorporating the phase component, each position obtains a unique encoding value, even if they share the same frequency. This ensures that the model can distinguish between positions that have similar frequency-based changes but occur at different points within the sequence.
# 4. Capture Temporal or Spatial Patterns: Phase information can capture temporal or spatial patterns in the data. For example, in time series analysis, the phase component can help capture the seasonality or periodic patterns in the data. In spatial data analysis, it can capture the spatial arrangement or layout of objects within an image or a graph. By considering the phase, the model can learn to recognize and utilize these patterns effectively.
# Incorporating the phase or timing of the position within the sequence in positional encoding provides the model with crucial information about sequential dependencies, positional relationships, and patterns in the data. It enhances the model's ability to understand and exploit the temporal or spatial characteristics of the sequence, leading to improved performance in various tasks.
# 
#
# Part2:why do we need to have postion-independent changes? isnt sin enough to capture position dependent inormation for our task?
# While the sine function is effective at capturing position-dependent information due to its periodic nature, incorporating the cosine function to capture position-independent changes provides additional flexibility and ensures a more comprehensive representation. Here are a few reasons why both components, sine and cosine, are beneficial:
#     Versatility and Generalization:
#         Sine for Periodic Patterns: Sine is well-suited for encoding positions with periodic patterns, such as sequences where certain positions exhibit recurring behaviors or variations.
#         Cosine for Constant Features: Cosine, with its constant oscillation, can effectively represent features that are consistent across different positions. This helps in capturing position-independent characteristics that may not follow a periodic trend.
#     Robustness to Shifts:
#         Shift Invariance: The combination of sine and cosine allows the positional encoding to exhibit a form of shift invariance. When a sequence is shifted, the phase relationships between sine and cosine components change accordingly, preserving the relative positional information.
#     Handling Different Time Scales:
#         Sine for Short-Term Changes: Sine can capture short-term variations or changes that occur with a certain periodicity.
#         Cosine for Long-Term Stability: Cosine, being constant over time, is suitable for encoding long-term stability or features that remain consistent irrespective of position changes.
#     Reducing Redundancy:
#         Orthogonality: The orthogonal nature of sine and cosine functions ensures that the information captured by each component is independent and non-redundant. This enhances the model's ability to distinguish between different positional characteristics.
#     Adaptability to Varied Sequences:
#         Handling Diverse Patterns: Many sequences exhibit a mix of periodic and non-periodic changes. The combination of sine and cosine allows the model to adapt to diverse patterns of positional information.
# In summary, incorporating both sine and cosine components in positional encoding provides a more versatile and adaptable representation of positional information. This approach enables the model to capture a wide range of patterns, both periodic and non-periodic, enhancing its ability to understand and generalize across different sequences and tasks.
# Let's illustrate the points mentioned using a unified example in the context of time series data representing temperature variations throughout the year:
# Consider a dataset that records daily temperature readings over a year. Each day's temperature can be seen as a position in the sequence. Here's how sine and cosine functions are beneficial in this scenario:
# Versatility and Generalization:
#     Sine for Periodic Patterns:
#         Example: Sine captures the seasonal periodicity in temperature, such as warmer temperatures during summer and colder temperatures during winter.
#     Cosine for Constant Features:
#         Example: Cosine represents features that remain constant, like the average temperature across different seasons, providing a position-independent encoding.
# Robustness to Shifts:
#     Shift Invariance:
#         Example: Shifting the entire temperature sequence (e.g., moving the start of the year) changes the phase relationship between sine and cosine, preserving the relative information despite the shift.
# Handling Different Time Scales:
#     Sine for Short-Term Changes:
#         Example: Sine captures short-term variations like daily temperature fluctuations, exhibiting a periodic pattern.
#     Cosine for Long-Term Stability:
#         Example: Cosine represents long-term stability, such as the overall trend of temperature changes over the entire year.
# Reducing Redundancy:
#     Orthogonality:
#         Example: The orthogonal nature of sine and cosine ensures that the information about daily fluctuations and overall trends is independent, reducing redundancy in the positional encoding.
# Adaptability to Varied Sequences:
#     Handling Diverse Patterns:
#         Example: Temperature data often involves a mix of periodic patterns (seasonal changes) and non-periodic variations (unpredictable weather events). The combination of sine and cosine allows the model to adapt to these diverse patterns.
# In summary, by incorporating both sine and cosine components in the positional encoding of temperature data, the model becomes more versatile. It can effectively capture both periodic and non-periodic patterns, enabling better generalization and understanding of various temperature sequences and tasks.
# Example 2: 
# Let's adapt the example to use text/word data, making it more intuitive:
# Versatility and Generalization:
#     Sine for Periodic Patterns:
#         Example: Consider a dataset of daily news headlines. Sine captures the periodicity in topics that recur, such as weekly trends in news coverage.
#     Cosine for Constant Features:
#         Example: Cosine represents features that are constant across different positions, like the consistent presence of certain keywords, providing a position-independent encoding.
# Robustness to Shifts:
#     Shift Invariance:
#         Example: Shifting the entire sequence of news headlines (e.g., moving the start of the dataset) changes the phase relationship between sine and cosine, preserving the relative information despite the shift.
# Handling Different Time Scales:
#     Sine for Short-Term Changes:
#         Example: Sine captures short-term variations like daily fluctuations in the frequency of specific words or topics in the news.
#     Cosine for Long-Term Stability:
#         Example: Cosine represents long-term stability, such as the overall trend of changes in the prevalence of certain themes over the entire dataset.
# Reducing Redundancy:
#     Orthogonality:
#         Example: The orthogonal nature of sine and cosine ensures that the information about daily fluctuations and overall trends in news coverage is independent, reducing redundancy in the positional encoding.
# Adaptability to Varied Sequences:
#     Handling Diverse Patterns:
#         Example: News headlines often exhibit a mix of periodic patterns (coverage of recurring events) and non-periodic variations (unpredictable news events). The combination of sine and cosine allows the model to adapt to these diverse patterns.
# In summary, by incorporating both sine and cosine components in the positional encoding of daily news headlines, the model becomes more versatile. It can effectively capture both periodic and non-periodic patterns, enabling better generalization and understanding of various text sequences and tasks.
# 
# Q: what does Orthogonality refers to and how is it relavent or intuitive here? 
# Orthogonality in the Context of Positional Encoding:
# In mathematics, orthogonality refers to the relationship between two vectors being perpendicular to each other. In the context of the positional encoding using sine and cosine functions, orthogonality is a crucial concept that enhances the effectiveness of the encoding.
# Let's break down how orthogonality is relevant and intuitive in this scenario:
#     Independence of Components:
#         The sine and cosine functions are orthogonal to each other. This means that the information encoded by the sine component is independent of the information encoded by the cosine component, and vice versa.
#     Reducing Redundancy:
#         In positional encoding, the goal is to represent various aspects of the sequence in a way that minimizes redundancy. If the sine and cosine components were not orthogonal, there might be overlapping information between them, diminishing the effectiveness of the encoding.
#     Distinct Encoding of Features:
#         The orthogonal nature ensures that each component is responsible for encoding different aspects of the sequence. Sine may capture periodic patterns, while cosine encodes constant features. Their orthogonality guarantees that the information captured by one does not overlap or interfere with the information captured by the other.
#     Enhanced Discrimination:
#         Orthogonality enhances the model's ability to discriminate between different positional characteristics. When the model processes the encoded sequence, it can rely on the fact that changes in one component do not inherently imply changes in the other. This separation of information contributes to a more nuanced understanding of the sequence.
#     Mathematical Simplicity:
#         The orthogonal relationship simplifies mathematical operations involving these components. When combining sine and cosine components, their orthogonality ensures that their interactions are well-defined and do not introduce complex dependencies.
# Example:
# Consider a scenario where a text sequence involves both daily fluctuations (modeled by sine) and long-term stability (modeled by cosine). The orthogonality ensures that the model can distinguish between the daily topics (captured by sine) and persistent themes (captured by cosine) without confusion.
# In summary, orthogonality in the context of sine and cosine functions used in positional encoding ensures independence between components, reduces redundancy, allows for distinct encoding of features, enhances discrimination capabilities, and simplifies mathematical operations. This property is crucial for creating a versatile and effective positional encoding scheme in various sequence-related tasks.
# 
# Q: is orthogonality in neural networks different or does it refer to the same thing? explain in depth
#
# Orthogonality in Neural Networks:
# In the context of neural networks, orthogonality takes on a slightly different meaning compared to its traditional mathematical definition, but the fundamental idea remains rooted in independence and lack of correlation.
# Weight Orthogonality:
# In neural networks, weight orthogonality refers to the orthogonal relationships between weight vectors in the weight space. Specifically, it involves ensuring that weight vectors are as orthogonal as possible to each other during training. This concept is particularly relevant in deep learning architectures.
# Relevance and Intuition:
#     Reducing Redundancy and Overfitting:
#         When weight vectors are orthogonal, they are less likely to duplicate or redundantly represent the same information. This can help in reducing overfitting, where a model may learn noise or specific training examples rather than general patterns.
#     Facilitating Training:
#         Orthogonality can aid in a more stable and efficient training process. When weight vectors are orthogonal, updates to one weight vector do not strongly influence others, promoting more independent learning.
#     Enhancing Representational Capacity:
#         Orthogonal weight vectors can contribute to a more diverse and expressive representation of the input data. Each weight vector can capture unique features or aspects of the data without interference from others.
#     Generalization and Adaptability:
#         Similar to the positional encoding example, orthogonality in neural networks enhances the model's ability to generalize and adapt to different patterns in the data. Independent weight vectors allow the model to learn a wide range of features without being overly constrained by correlations.
#     Mitigating Vanishing or Exploding Gradients:
#         Orthogonal weight matrices can also help address issues like vanishing or exploding gradients during backpropagation. This is particularly relevant in deep networks, where maintaining a stable gradient flow is crucial for effective training.
# Example:
# Consider a deep neural network processing images. If the weight vectors corresponding to different convolutional filters are orthogonal, it means that each filter is specialized in capturing a unique aspect of the image, whether it's edges, textures, or higher-level features. This diversity contributes to a more robust and generalizable representation of the input data.
# Summary:
# While the term "orthogonality" may have a different application in neural networks compared to its traditional mathematical context, the core idea remains centered around independence, lack of correlation, and promoting diverse and efficient learning. In neural networks, weight orthogonality specifically addresses the relationships between weight vectors, contributing to improved generalization, reduced redundancy, and more effective training.
# 
# More Explanation : 
#Orthogonality in Neural Networks: A Deeper Dive
# In neural networks, orthogonality extends beyond its traditional geometric interpretation and takes on a specialized meaning within the context of weight matrices. Let's delve deeper into the nuances of weight orthogonality and its implications in the realm of deep learning.
# 1. Geometric Perspective:
#     Traditional Orthogonality: In mathematics, orthogonality between vectors implies a right-angle relationship. In the context of neural networks, this concept is adapted to the weight space. Weight vectors are considered orthogonal if their dot product is close to zero, signifying independence.
# 2. Weight Orthogonality:
#     Defining Weight Orthogonality: In neural networks, weight orthogonality refers to the idea that weight matrices (collections of weight vectors) are as orthogonal as possible. This concept is often applied to weight initialization or regularization techniques.
# 3. Reducing Redundancy and Overfitting:
#     Overfitting Mitigation: When weight vectors are orthogonal, they are less likely to redundantly encode similar patterns. This property can mitigate overfitting by encouraging the model to learn distinctive features, reducing reliance on specific training examples.
# 4. Facilitating Training Stability:
#     Independent Learning: Orthogonal weight vectors contribute to stable training. Updates to one weight vector have less impact on others, promoting more independent learning. This is particularly important in deep networks where instability in training can be a challenge.
# 5. Enhancing Representational Capacity:
#     Diverse Representations: Orthogonal weight matrices enhance the network's representational capacity. Each weight vector can specialize in capturing unique features or patterns, allowing the model to learn a rich and diverse set of representations.
# 6. Generalization and Adaptability:
#     Improved Generalization: Orthogonality fosters better generalization by ensuring that the model can adapt to a wide range of patterns. The independence between weight vectors allows the network to handle diverse input data effectively.
# 7. Mitigating Gradient Issues:
#     Addressing Gradient Challenges: Orthogonal weight matrices can help mitigate issues like vanishing or exploding gradients during backpropagation. This is critical for maintaining a stable gradient flow, especially in deep networks.
# 8. Example: Image Processing in Convolutional Networks:
#     Role in Convolutional Filters: Consider a convolutional neural network (CNN) processing images. If the weight vectors corresponding to different convolutional filters are orthogonal, each filter specializes in capturing distinct visual features (edges, textures). This diversity enhances the model's ability to recognize a broad range of image patterns.
# 9. Mathematical Rigor:
#     Eigenvalue Preservation: Orthogonal matrices have the property of preserving eigenvalues, contributing to numerical stability during training and optimization processes.
# 10. Practical Implementation:
#     Orthogonal Initialization: Researchers and practitioners often use orthogonal weight initialization techniques to encourage the orthogonality of weight matrices at the beginning of training.
# 11. Advanced Considerations:
#     Adaptive Orthogonality: Some advanced techniques focus on maintaining orthogonality dynamically during training, adapting to the evolving nature of the learned features.
# 12. Open Questions and Research:
#     Ongoing Exploration: The exploration of orthogonality in neural networks is an ongoing area of research, with scientists seeking to uncover more insights into its impact on optimization, generalization, and the learning dynamics of deep networks.
# In summary, weight orthogonality in neural networks is a multifaceted concept that goes beyond its geometric roots. It plays a crucial role in shaping the learning dynamics, stability, and generalization capabilities of deep learning models, contributing to the ongoing refinement of training techniques in the field.
# 
# 
# Q: how do you change frequency for a sin/cos? 
# In the context of sine and cosine functions used for positional encoding or signal processing, changing the frequency involves modifying the rate at which these functions oscillate or complete cycles within a given interval. The frequency of a sine or cosine function determines how rapidly it repeats its pattern over time.
# Changing Frequency in Sinusoidal Functions:
#     Frequency Parameter: The formula for a sinusoidal function ( f(x) = A \cdot \sin(Bx + C) ) consists of several components:
#         ( A ) represents the amplitude (the peak value of the function).
#         ( B ) corresponds to the frequency, determining how quickly the function oscillates.
#         ( C ) represents the phase shift (a horizontal shift of the function).
#     Modifying Frequency: To change the frequency of a sinusoidal function, adjust the ( B ) parameter:
#         Increasing ( B ) will accelerate the oscillation, compressing the function horizontally. This effectively increases the frequency.
#         Decreasing ( B ) will decelerate the oscillation, stretching the function horizontally. This effectively decreases the frequency.
#     Relationship with Period: The frequency and the period of a sinusoidal function are inversely related. Frequency ( f ) and period ( T ) are related by the equation ( f = \frac{1}{T} ), where ( T ) represents the period (the length of one complete cycle).
# Changing Frequency in Cosine Functions:
# Similar to sinusoidal functions, cosine functions follow a similar formula ( g(x) = A \cdot \cos(Bx + C) ), with ( A ) as the amplitude, ( B ) as the frequency, and ( C ) as the phase shift.
#     Frequency Modification: Adjusting the ( B ) parameter in a cosine function will also change its frequency:
#         Increasing ( B ) will speed up the oscillation, effectively increasing the frequency.
#         Decreasing ( B ) will slow down the oscillation, effectively decreasing the frequency.
#     Correlation with Sine Function: Cosine functions are related to sine functions, typically having the same frequency but with a phase shift of ( \frac{\pi}{2} ) radians or ( 90^\circ ).
# Application in Positional Encoding:
# In positional encoding, altering the frequency of sine and cosine functions helps represent different positional information within a sequence. By adjusting the frequency parameters for sine and cosine functions, distinct patterns at various scales or positions can be encoded, allowing models to differentiate between different positions in a sequence.
# Overall, changing the frequency parameter in sine and cosine functions involves adjusting the rate of oscillation, thereby affecting the speed at which these functions repeat their patterns. This adjustment is fundamental in encoding different positional information in sequence-related tasks within neural networks or signal processing applications.
# 
# imagine sin(1), sin(1/2), sin(1/100), ..., sin(1/100^2), sin(1/100^3),... 
# The frequency of sin(1/100^n) as n increases is inversely proportional to the period of the function. 
# The period of sin(1/100^n) is 2π/(1/100^n) = 2π100^n. Therefore, the frequency of sin(1/100^n) is 1/(2π100^n) 1.
# As n increases, the frequency of sin(1/100^n) decreases exponentially. This means that the function oscillates 
# more slowly as n increases, and the time between each oscillation increases.
# side note: 
# The 2π in the formula for the period of the sine function comes from the fact that the sine function
# is periodic with a period of 2π radians. 
# This means that the sine function repeats itself every 2π radians. 
# The period of sin(1/100^n) is 2π/(1/100^n) = 2π*100^n 
# siden ote2:
# the reason we use increasing value for frequency fraction(lowering it for sin/cos as we go) for sin/cos
# is to comeup with unique values for each dimension. note that its is true that cos has a phase shift of 90 degress
# so you may think using the same freq with both of them would give different values (like sin(0)=0/cos(0)=1) but
# they will have the same value at certain points like (sin(45)=cos(45)) so we use a different frequency for each
# as you can imagine, up to a point it works, and atfer that we basically endup with constant values 
# this is visible in our plot below.
#
# sidenote 2: 
# Q: are sin and cos orthogonal ?
# Yes, the sine and cosine functions are orthogonal to each other. Two functions are said to be orthogonal if their inner product is zero. The inner product of two functions is defined as the integral of the product of the two functions over a given interval. In the case of the sine and cosine functions, their inner product over one period is zero, which means they are orthogonal 1
# Q: but sin(45)==cos(45), how can they be orthogonal with respect to each other when they both produce the same value ?
# You are correct that the sine and cosine functions have the same value at 45 degrees or pi/4 radians. However, orthogonality of two functions is defined as their inner product being zero. The inner product of two functions is defined as the integral of the product of the two functions over a given interval. In the case of the sine and cosine functions, their inner product over one period is zero, which means they are orthogonal 
# 
# 
# 
# 
# 
# # %%
# #  https://www.youtube.com/watch?v=ZMxVe-HK174&t=289s intresting alternative implementation
import numpy as np
import matplotlib.pyplot as plt
plt.figure(figsize=(32, 8))

sequence_length = 50
embedding_dim = 4

positions = np.arange(sequence_length)
embeddings = np.zeros((sequence_length, embedding_dim))

# Calculate sine and cosine values for each position and dimension
for pos in positions:
    for dim in range(embedding_dim):
        freq = 1 / (10000 ** (2 * dim / embedding_dim))
        embeddings[pos, dim] = np.sin(pos / freq)
        embeddings[pos, dim] = np.cos(pos / freq)

# Plot the positional encodings
for dim in range(embedding_dim):
    plt.plot(positions, embeddings[:, dim], label=f"Dimension {dim+1}")

plt.xlabel("Position")
plt.ylabel("Encoding Value")
plt.title("Sinusoidal Positional Encoding")
plt.legend()
plt.show()
#%%
#
#
#
#
#
#%%
#
#
#%%
import numpy as np
import matplotlib.pyplot as plt

def pos_enc(pos, embd_size):
    div_term = np.exp(np.arange(0, embd_size, 2)) * -(np.log(10_000.0) / embd_size)
    rep = np.zeros(embd_size)
    rep[0::2] = np.sin(pos * div_term)
    rep[1::2] = np.cos(pos * div_term)
    return rep

sequence_length = 100
embedding_dim = 16

positions = np.arange(sequence_length)
embeddings = np.zeros((sequence_length, embedding_dim))

# Calculate positional encodings using pos_enc function
for pos in positions:
    embeddings[pos] = pos_enc(pos, embedding_dim)

# Plot the positional encodings
for dim in range(embedding_dim):
    plt.plot(positions, embeddings[:, dim], label=f"Dimension {dim+1}")

plt.xlabel("Position")
plt.ylabel("Encoding Value")
plt.title("Sinusoidal Positional Encoding (pos_enc)")
plt.legend()
plt.show()
# this plot is the same as the following one!
#%%
import math
import torch
import numpy as np
def pos_enc(embd_size, max_seq_length=10000):
    # Compute the positional encodings once in log space.
    position_embd = np.zeros(shape=(max_seq_length, embd_size))
    position = np.arange(0, max_seq_length)[:, np.newaxis]
    div_term = np.exp(np.arange(0, embd_size, 2) * -(math.log(10000.0) / embd_size))
    position_embd[:, 0::2] = np.sin(position * div_term)
    position_embd[:, 1::2] = np.cos(position * div_term)
    return position_embd
    
plt.figure(figsize=(15, 5))
y = pos_enc(embd_size=20)
# lets plot 4 embd values for 100 positions, (we used 4:8 becasue they demonstrate pretty graphs! 
# use other numbers and see the outcome) 
dims = (4,8)
plt.plot(range(100), y[0:100, slice(*dims)])
# plt.plot(np.arange(100), y[:100, 8:12])
plt.legend(["dim %d"%p for p in range(*dims)])
# %%
# side notes: 
# sin = pos/n^(2i/d_model)
# The highest number that i can be set to is d_model divided by 2 since the equations alternate for each element 
# in the embedding. n in the original paper recommends 10,000.
#  
from pprint import pprint
import random
import numpy as np
import matplotlib.pyplot as plt
# write a function to plot this, what pecularities are evident when plotting this?
# https://www.tensorflow.org/text/tutorials/transformer
# https://github.com/jalammar/jalammar.github.io/blob/master/notebookes/transformer/transformer_positional_encoding_graph.ipynb
# https://www.scaler.com/topics/nlp/positional-encoding/
# https://towardsdatascience.com/master-positional-encoding-part-i-63c05d90a0c3

def sinusoidal_positional_encoding(max_position, d_model):
    
    assert d_model%2==0, "this needs to be an even number, otherwise odd/even count wont match! and we'll face an error"
    position = np.arange(0, max_position)[:, np.newaxis]
    # we use exp instead of the paper's implementation so its numerically more stable (As we involve tiny numbers
    # exp allows us to not face such issues!)
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000) / d_model))
    
    # Calculate sinusoidal embeddings
    pos_enc = np.zeros((max_position, d_model))
    pos_enc[:, 0::2] = np.sin(position * div_term)
    pos_enc[:, 1::2] = np.cos(position * div_term)
    
    return pos_enc
pprint(plt.colormaps())
def plot_positional_encoding(positional_encoding):
    plt.figure(figsize=(128, 64))
    # concerning colormaps read this first : https://matplotlib.org/stable/users/explain/colors/colormaps.html#colormaps 
    # https://matplotlib.org/stable/gallery/color/colormap_reference.html
    # side note for why we chose viridis : https://sjmgarnier.github.io/viridis/articles/intro-to-viridis.html 
    # what other colormaps we have? simply check plt.colormaps() to see your other options
    # uncomment the following line instead of the next line and see the effect of different colormaps.
    # of course not all colormaps suit all usecases, read the first link if you havent. 
    # basically viridis belongs to a so called 'Perceptually Uniform Sequential' colormap group. 'magma', 'inferno', 
    # 'plasma', 'cividis' and turbo are other examples of what we call a preceptually uniform sequential colormap.
    # Perceptually uniform, means values close to each other have similar-appearing colors and values
    # far away from each other have more different-appearing colors, consistently across the range of values.
    # and sequential simply refers to the fact that the lightness value increases monotonically through the colormap.
    # we have other types such as Diverging, Cyclic and Qualitative, which each has its own specific usecase
    # for example Qualitive colormaps which are usually miscellaneous colors, are used to represent information
    # that does not have ordering or relationships. 
    # The Cyclic colormaps on the otherhand as the name suggest, refer to change in lightness of two different colors that 
    # meet in the middle and beginning/end at an unsaturated color; 
    # are used for values that wrap around at the endpoints, such as phase angle, wind direction, or time of day.
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
    #  or  maybe the rdbu is better!
    cmap = 'RdBu'
    # play with the values and see how as we near the end of embd, the value seem to become constant!
    # use :10, :100, :200, then 100:200, 150:200, etc for embddiing dimension
    # plt.pcolormesh(positional_encoding[:,:200], cmap=cmap)
    # for cmap in plt.colormaps():
    plt.pcolormesh(positional_encoding[:,:], cmap=cmap)
    plt.xlabel('Embedding Dimensions')
    plt.ylabel('Position')
    plt.colorbar(label=f'Value({cmap})')
    plt.title('Sinusoidal Positional Encoding')
    plt.show()

# Example usage
max_position = 5
d_model = 6
pos_enc = sinusoidal_positional_encoding(max_position, d_model)
print(f'{pos_enc=}')
plot_positional_encoding(pos_enc)
#  and now lets see larger pos/embd_size
max_position = 1000
d_model = 512
pos_enc = sinusoidal_positional_encoding(max_position, d_model)
plot_positional_encoding(pos_enc)

# heres how this function is doing its job: 
# 1. **Figure Initialization**:
#    ```python
#    plt.figure(figsize=(12, 6))
#    ```
#    This line initializes a new figure for the plot with a specific size. The `figsize=(12, 6)` parameter 
#    specifies the width and height of the figure in inches.
# 2. **Heatmap Generation**:
#    ```python
#    plt.pcolormesh(positional_encoding, cmap='viridis')
#    ```
#    - `plt.pcolormesh()` creates a pseudocolor plot (heatmap) from a 2D array, which in this case is the 
#    `positional_encoding` matrix.
#    - `positional_encoding` is the matrix containing the sinusoidal positional encoding values. It represents
#    the embeddings for different positions along the sequence and their respective dimensions.
#    - `cmap='viridis'` specifies the colormap to use for the heatmap. In this case, the 'viridis' colormap 
#    ranges from yellow to blue, providing good perceptual uniformity for different values.
# 3. **Axis Labels**:
#    ```python
#    plt.xlabel('Embedding Dimensions')
#    plt.ylabel('Position')
#    ```
#    - `plt.xlabel()` and `plt.ylabel()` set labels for the x-axis and y-axis, respectively. 
#    'Embedding Dimensions' represents the different dimensions of the embeddings, and 'Position' 
#    denotes the positions along the sequence.
# 4. **Colorbar**:
#    ```python
#    plt.colorbar(label='Value')
#    ```
#    - `plt.colorbar()` adds a colorbar to the plot, indicating the mapping of colors to values in the heatmap.
#    The 'Value' label describes the quantity represented by the colors in the heatmap.
# 5. **Title**:
#    ```python
#    plt.title('Sinusoidal Positional Encoding')
#    ```
#    - `plt.title()` sets the title of the plot as 'Sinusoidal Positional Encoding'. This title provides 
#    context for what the heatmap represents.
# 6. **Displaying the Plot**:
#    ```python
#    plt.show()
#    ```
#    - `plt.show()` displays the generated plot. 
# It's necessary to view the heatmap within the Jupyter Notebook, Python script, or any interactive environment.
# In summary, the `plot_positional_encoding()` function utilizes `matplotlib` to create a heatmap representation 
# of the sinusoidal positional encoding matrix. 
# This visualization helps in understanding how positions along a sequence are encoded in a neural network model,
# especially in scenarios like attention mechanisms or positional embeddings within transformers.
#
# Now, regarding peculiarities when plotting sinusoidal positional encoding:
# Wave Patterns: You'll observe clear wave patterns in the plot, reflecting the sinusoidal nature of the encoding. These waves indicate how different positions along the sequence are represented in the embedding space.
# Frequency Variation: The frequency of the waves varies across different dimensions of the embedding. Lower dimensions may capture longer-range dependencies, while higher dimensions may focus on shorter-range dependencies.
# Alternating Colors: Due to the use of sine and cosine functions, you'll notice alternating dark and light bands in the heatmap. This alternation ensures that the model can distinguish between adjacent positions.
# Positional Diversity: The heatmap will illustrate how each position in the sequence has a unique representation in the embedding space. This is crucial for the model to distinguish between tokens based on their absolute or relative positions.

# In the previous plot, the blue/white stripes represent the values of the positional vectors. 
# The reason you see fewer changes (less blue/white stripes) towards the end of the plot is due to
# the nature of the positional encoding scheme used in transformer models. 
# This scheme uses sine and cosine functions of different frequencies to generate the positional 
# encoding vectors. 
# As you move towards higher dimensions, the frequency of these functions decreases,
# leading to fewer changes in the values and hence fewer stripes in the plot.
# The stripes at the far end of the embedding dimensions do not represent a single value. 
# They are indeed many tiny numbers that are simply too small to make a significant difference, 
# thus they are shown as blue/white for all positions.
# The difference in the number of stripes between the 1k plot and the 50,000 positions plot could 
# be due to the difference in the total number of positions encoded in each plot. A plot with more 
# positions (like the 50,000 positions plot) would naturally have more stripes as it represents more 
# positional information.
# The key point to understand from visualizing these positional vectors is how positional information 
# is encoded in transformer models. It helps us see that the positional encoding scheme can capture the 
# order of data points in a sequence, which is crucial for tasks like natural language processing where 
# the order of words in a sentence carries important semantic information1. The plot also shows how this 
# positional information varies across different dimensions, providing insights into the workings of 
# high-dimensional data in machine learning models.
#
# Note that the frequency is actually decreasing in this snippet. 
# This is because the div_term is an exponential decay term, where the base of the exponent is less than 1 
# (np.exp(np.arange(0, embd_size, 2)) * -(np.log(10_000.0) / embd_size)). 
# This means that as you move along the embedding size, the frequency of the sine and cosine terms in the
# positional encoding decreases. 
# This is a key aspect of the Transformer’s positional encoding, allowing it to capture both short-term and
# long-term dependencies in the input sequence. The sine and cosine functions provide a way to encode the 
# position with a unique representation that can capture relative positions and is invariant to the sequence
# length. The decreasing frequency ensures that the model can distinguish positions across a wide range of 
# sequence lengths.
# 
# relationship with wavelength:
# In the context of waves, frequency and wavelength are inversely related. 
# As the frequency of a wave increases, the wavelength decreases, and vice versa. 
# This relationship is governed by the equation:
# v=fλ
# where:
# (v) is the speed of the wave,
# (f) is the frequency, and
# (λ(lambda)) is the wavelength.
# In the positional encoding scheme used in the Transformer model, 
# the decreasing frequency can be thought of as an increasing “wavelength” along the dimensions of the positional
# encoding vector. This means that the positional information encoded by higher dimensions changes more slowly 
# (longer “wavelength”), allowing the model to capture longer-term dependencies in the data. 
# Conversely, the positional information encoded by lower dimensions changes more quickly (shorter “wavelength”),
# enabling the model to capture shorter-term dependencies. 
# This balance allows the model to understand both the local and global structure of the sequence.
#
#In the context of Transformer models, 
# the concept of "longer waveform" is analogous to the slower changing positional encoding values in higher 
# dimensions. 
# The positional encoding in Transformer models uses a mix of sine and cosine functions with different frequencies.
# The frequency of these functions decreases (or the "wavelength" increases) as you move to higher dimensions in 
# the positional encoding vector. 
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
# from 
# Function Periods
# The position vector has shorter wavelengths for lower dimensions. 
# As i increases, the periods of the function also increase. 
# When i reaches the value of d, a large number of pos vectors are needed to cover the entire period 
# of the function. The values of the first 20 positions at higher indexes are almost constant. 
# This can be observed in the figure below, where the colors of columns 30-50 hardly change.

#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#
# this function calculates the Euclidean distance between the positional encoding vectors of 
# neighboring time-steps and plots these distances.
# The plot will show that the distances between neighboring time-steps decrease as you move along
# the time axis, illustrating the decay of positional information over time in the sinusoidal positional
# encoding scheme.
def get_positional_encoding(max_len, d_model):
    pos = np.arange(max_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    positional_encoding = np.zeros((max_len, d_model))
    positional_encoding[:, 0::2] = np.sin(pos * div_term)
    positional_encoding[:, 1::2] = np.cos(pos * div_term)
    return positional_encoding

def plot_positional_encoding_distances(positional_encoding):
    distances = np.sum(np.square(positional_encoding[:-1] - positional_encoding[1:]), axis=1)
    plt.plot(distances[:])
    plt.ylabel('Distance')
    plt.xlabel('Time-step/embd dim')
    plt.title('Distance between neighboring time-steps in positional encoding')
    plt.show()

max_len = 50000
d_model = 512

positional_encoding = get_positional_encoding(max_len, d_model)
plot_positional_encoding_distances(positional_encoding)

# now if we try to display this as a heatmap, we will get this: 
def plot_positional_encoding_heatmap(positional_encoding):
    distances = np.sum(np.square(positional_encoding[:, np.newaxis] - positional_encoding[np.newaxis, :]), axis=2)
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

max_len = 512
# smaller dims shows the shades much better than a larger dim such as 512
d_model = 100

positional_encoding = get_positional_encoding(max_len, d_model)
# the information is given below (explanation part)
plot_positional_encoding_heatmap(positional_encoding)
# showing that the distance between neighboring time-steps are symmetrical and decays nicely with time.
# that is, the diagnol axis has the highest score, which really says, each token/position has the highest
# relationship with itself, as we get farther away, we see the blue turns to white slowly, showing the relation
# ship between nearer position is stronger than those far away, and the shades show that this gradually and symetrically
# decreases. try sin/cos only and see why we use both of them together!
plot_positional_encoding_dot_product_heatmap(positional_encoding)
# explanation 
# Sure, I'd be happy to explain the `plot_positional_encoding_heatmap()` function in detail. Here's what each line of code does:
# ```python
# def plot_positional_encoding_heatmap(positional_encoding):
# ```
# This line defines the function `plot_positional_encoding_heatmap()`, which takes one argument: `positional_encoding`. The `positional_encoding` argument is expected to be a 2D numpy array containing the positional encoding vectors.
# ```python
#     distances = np.sum(np.square(positional_encoding[:, np.newaxis] - positional_encoding[np.newaxis, :]), axis=2)
# ```
# This line calculates the Euclidean distance between all pairs of positional encoding vectors. 
# Here's how it works:
# - `positional_encoding[:, np.newaxis] - positional_encoding[np.newaxis, :]` computes the difference between 
#    all pairs of positional encoding vectors. The `np.newaxis` is used to add an extra dimension to the arrays,
#    allowing numpy to broadcast the subtraction operation across all pairs of vectors.
# - `np.square(...)` squares each element of the resulting array. This is the first step in the calculation of
#    the Euclidean distance, which is the square root of the sum of the squares of the differences.
# - `np.sum(..., axis=2)` sums the squared differences along the last dimension (dimension 2) of the array. 
#    This completes the calculation of the Euclidean distance.
# ```python
#     sns.heatmap(distances)
# ```
# This line uses the seaborn library's `heatmap()` function to create a heatmap of the distances. 
# Each cell in the heatmap corresponds to the distance between a pair of positional encoding vectors.
# ```python
#     plt.ylabel('Position')
#     plt.xlabel('Position')
# ```
# These lines set the labels for the y-axis and x-axis of the heatmap to 'Position'.
# ```python
#     plt.title('Heatmap of distances between positional encoding vectors')
# ```
# This line sets the title of the heatmap to 'Heatmap of distances between positional encoding vectors'.
# ```python
#     plt.show()
# ```
# This line displays the heatmap. If this line were not included, the heatmap would be created but not displayed.
# In summary, the `plot_positional_encoding_heatmap()` function calculates the Euclidean distance between all 
# pairs of positional encoding vectors and displays these distances as a heatmap. The heatmap provides a visual
# representation of how the positional encoding changes across different positions in the sequence. 
# The diagonal line in the heatmap represents the distance of a position with itself, which is zero. 
# The symmetry of the heatmap reflects the fact that the distance from position i to position j is the same as
# the distance from position j to position i. This symmetry and the structure of the heatmap can provide insights
# into the nature of the positional encoding scheme used in Transformer models.



# # Generate x values
# x = np.linspace(0, 4 * np.pi, 100)
# # Generate intermediary variable for frequency transition
# freq = np.linspace(5, 5.5, len(x))
# # Generate sine waves with varying frequency
# sine_waves = np.sin(freq * x[:, np.newaxis])
# # Plotting
# plt.figure(figsize=(16, 8))
# for i in range(len(x)):
#     plt.plot(x, sine_waves[:, i], color='blue', alpha=0.6)
# plt.xlabel('x')
# plt.ylabel('Amplitude')
# plt.title('Transition from Low Frequency to High Frequency - Sine Waves')
# plt.show()
#%%
# 
#
# The following notes were taken from The Stanford XCS224U: NLU I Contextual Word Representations, Part 3: Positional Encoding I Spring 2023
# --The role of positional encoding: 
# transformers/attention mechanism has a very limited capacity to keep track of word order
# the attention connections are not directional, they are just bunch of dot products and there are no interactions between the columns, 
# and becasue of this we need to ensure theres a difference between sequence A,B,C  and C,B,A (i.e. 
# its is different thatn C,A,B, or C,B,A ,etc cuz we are not keeping track of token order in attention)
# positional encoding therefore, makes sure that these sequences are different, regardless of what we do with 
# the representattions that come out of the model. theres another role they fill in, they have been used to
# to keep track of the hierarchial notions of position (premise/hypothesis in natural language inference which is
# one of the important features of BERT model we later on will discuss)
# --Evaluating positional encoding schemes: 
# there are a lot of prespective we can take on postional encoding, but two major questions that can be asked
# are : 
# 1. does the set of positions need to be decided ahead of time? 
# 2. does the positional encoding scheme hinder generalization to new positions?
# as we know, models tend to impose a max length on the sequences they can process, for reasons relating to
# their learned weights(training, optimization, etc). we will ask whether different positional embedding schemes are imposing anything
# about length generalization separate from this. so we are asking if we set this fact aside for a moment, 
# does the positional encoding scheme itself, is imposing anytihng about sequence length generalization?
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
# so to recap: 
# the limitations we face in this scheme are: 
# 1.set of position needs to be decided ahead of time
# 2. may hinder generalization to new positions, even for similar phanamena ([the]+[1] [rock]+[2] 
# has a different representation than [the]+[15] [rock]+[16]) 
# there will be some similarity between them as we have the same wordvectors involved(for the and rock)
# but since we add position embedding, the result will be very heavy handed, when it comes to learning representation
# tht are heavily position dependent, and this could make it harder for the model to see Rock forexample
# is the same phrase whether its used in the begining of the sequence or middle or end of it.
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
    div_term = np.exp(np.arange(0, embd_size, 2 )) * -(np.log(10_000.0)/embd_size)
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
# positions make sense for them, so the paper pf shaw etal 2018 from google brain, proposes that since the
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
#%%
# !all entries interact with all other entries at the same time, and the order of words/tokens is lost
# 'how are you' would be the same as 'are how you', 'you how are', 'you are how', 'how you are', etc
# since we are using this in a autoregressive model, given a character, we want the next one, its obvious
# that we want the order of words/tokens to be preserved, so we get the output that makes sense!, otherwise,
# this would create the same output for all of these input, which is not what we want!.(theres nothing in
# the input to specify the order, that is we treat it as if theres none!). in other
# usecases, such as translation from one language to another, we dont have this issue, as we want to grab
# one paragraph in one language completely and then produce a translation later, the first part doesnt
# need this as we arent after producing the next word/token, but the second part does, we talk about this in more
# details later, but for now, we know we want to preserve the order as well. so what do we do?  
# so we need to also incorporate the position information for each token as well.
# there are several ways we can add this inormation, each with its own set of merits and drawbacks
# for one, we can add a number for to each token, like 0 for how, 1 for are, and 2 for you, and so on and so forth
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
#
# 
#
#
# Positional encoding:
# As we pointed out earlier, the attention mechanism does not have any notion of position, unlike the cnn or rnns
# that are inheritly sequential and are posision aware. 
# without positional information, a sequence such as ABC would be the same as CBA, ACB, BCA, or any permutations of the 
# tokens involved. this is specifically bad for an autoregressive task like generating text. 
# An autoregressive model, given a token, the model needs to comeup with a good prediction! obviously this requires the order
# of tokens to be taken into account to create meaningful sequences/texts. 
# imagine you expecing the model to greet you like: Hello, nice to meet you! but instead you get a random sequence of tokens
# like: ',to nice! Hello you meet' which is not what we are after! in reality the network might comeup with a weak notion of 
# order, simply based on the statistics it saw in the dataset, however, it cant utilize that to comeup with complex structures
# certainly not the ones that have subtle meaning changes, when the order of some parts are reversed! like :
# live to work!
# work to live!
# as you can see the order plays a crucial role not only in generating text but also infering the undderlying meaning.
# 
# so how can we add this positional information. there are several ways we can think of to add this information. 
# we can learn these so called positional encoding the same way we learn the token embeddings. this is called abosulte positional encoding!
# it has some pros and cons we will get to in a moment. 
# we can simply pre-compute these positional encodings. this is what the original paper of attention is all you need, did.
# their choice is refered to as sinusoidal positional encoding which provides relative positioning as well.
# basically the combination of sin/cos acts as a binary counter, and this way the positional info for each token
# is calculated and provided to the network.
# to be more specific, the frequency for sin/cos increases for each dimension, so the idea is, the network can, by looking,
# at the frequencies, figureout, which token has come after which one, and using this, it can understand the 
# relative posiosions of the tokens involved. ()
# (the author in their previous work(Order Matters: Sequence to sequence for sets 2015), pointed out that counting was
# was a hard problem, and 2 years later they came up with the sinusoidal positioning encoding. 
# see 
# https://www.reddit.com/r/learnmachinelearning/comments/9e4j4q/positional_encoding_in_transformer_model/
# http://fastml.com/introduction-to-pointer-networks/
# https://www.reddit.com/r/MachineLearning/comments/cttefo/d_positional_encoding_in_transformer/
# https://arxiv.org/abs/1905.04226 )
# later works however, showed that such embeddings can be learned and infact BERT did exactly that and opted to use
# learned positional emebddings instead of the sinusoidal positional encoding used by the original paper.
# later on, other versions such as rope or rotary positional encoding were introduced to provide relative positional 
# encoding (better). after than Another work AliBi, completely removed the positional embedding and instead used
# constraints on relationships with respect to their distance! (i.e. decaying the strength of relationship based on
# distance) so this is an activate field of research and today as I write this, rope is being used along with 
# learned positional encoding. This needs more 
# 
# method 1: 
# for one we can simply add the tokens absolute position to the token embedding as an extra dimension. 
# this method has several problems. 
# 1. as the sequence length incresaes, larger and larger numbers are asinged, this creates
# a nonintentional bias towards the tokens at the end, apart from that, it can hinder the optimization process and cause all
# sorts of problems such as gradient explosion, etc. 
# also the contribution of the tokens will be different, early tokens would have a minscule impact compared to the later tokens
# that have a larger index. normalizing the values so all numbers fall into a range can mitigate this issue, but it creates 
# another issue, the same position, can now take different values based on the sequence length. the 3rd position in a sequence of 
# 4, has a different value (3/4=0.75) compared to a sequence of 10 (3/10=0.3). this will confuse the model!
# so this is not a good idea! apart from this, this might clutter up the semantic meaning of the word emebddings
# if we just append it, other than, if instead of a single dimension, we somehow could encode that number to a vector
# the same length as the word/token embedding, we would get a better result. imagine this could give us a positional embedding
# as apposed to position number/info/hint/condition! with the positional embedding, like word embedding, the network
# could comeup with relative positional information by itself. like by clustering each position into its own feature subsapce
# the same way we have this for word embedding which allows us to do kin - man = queen! or queen - woman = king! 
# do you get the idea? the idea is to somehow nodge each word ever so slightly to a specific position, so that 
# its semantic is preserved, and also the position is also separate from others. for this all dimensions of the embedding
# need to be taken into account, so thats why the emebdding portion makes sense! 
# now if we take this route, we see that the scale needs to be the same, and all positions need to use the same identifier
# regardless of sequence length so network can actually differentiate between these positions properly.
# from there, we either learn it, or precompute it. the sin/cos is the pre computed one!
# what else can we do?
# we can   
# are not different
#
# this is called absolute position embedding itsl ike sinusodal position embedding that was
# introduced in the original paper! explain more 
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
# %%[markdown]
# side note 
# 
# The frequency of a sine or cosine wave is determined by the rate at which it oscillates or completes cycles over time. In mathematical terms, frequency (f) is the number of cycles per unit of time, usually measured in Hertz (Hz). The relationship between frequency, angular frequency, and the period of a wave is given by the equation:
# [ f = \frac{1}{T} ]
# Where:
#     ( f ) is the frequency,
#     ( T ) is the period of the wave (the time it takes to complete one cycle),
#     ( \omega ) (angular frequency) is related to ( f ) by ( \omega = 2\pi f ).
# For a sine or cosine wave, the general form is:
# [ y(t) = A \cdot \sin(2\pi f t + \phi) ]
# Where:
#     ( A ) is the amplitude,
#     ( f ) is the frequency,
#     ( t ) is time,
#     ( \phi ) is the phase.
# Now, to increase the frequency of a sine or cosine wave, you can do one of the following:
#     Increase the angular frequency ( \omega ):
#         ( \omega = 2\pi f )
#         By increasing ( \omega ), you effectively increase the rate at which the sine or cosine function oscillates.
#     Decrease the period ( T ):
#         The period ( T ) is the reciprocal of frequency, so by decreasing the period, you increase the frequency.
#         ( T = \frac{1}{f} )
# Here's a more detailed explanation:
#     Angular Frequency ( \omega ):
#         The angular frequency ( \omega ) represents how quickly the wave oscillates in radians per unit of time.
#         If you increase ( \omega ), the wave will complete more cycles in the same amount of time.
#         This is often more intuitive when thinking about the periodicity of the wave in terms of its angular measure (radians) rather than cycles.
#     Period ( T ):
#         The period ( T ) is the time it takes for one complete cycle of the wave.
#         If you decrease the period, the wave completes cycles more quickly, effectively increasing the frequency.
# In summary, to increase the frequency of a sine or cosine wave, you can either increase the angular frequency ( \omega ) or decrease the period ( T ). These changes will result in a wave that oscillates more rapidly over time.
import numpy as np
import matplotlib.pyplot as plt

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
# %%
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
        #! explain positional embeddings 
        # this is called absolute position embedding itsl ike sinusodal position embedding that was
        # introduced in the original paper! explain more 
        # https://www.youtube.com/watch?v=3mTsYm9qQFA
        # https://www.youtube.com/watch?v=o29P0Kpobz0
        # https://www.youtube.com/watch?v=JERXX2Byr90
        # https://www.youtube.com/watch?v=M2ToEXF6Olw
        #!https://www.youtube.com/watch?v=4AzsiCMw_-s
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
head_size = 16
embd_size = 16 
context_size = 8
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
for i in range(max_iter):
    # get the batch
    x,y = get_batch('train', batch_size=batch_size)
    # feed the model and get the logits
    logits, loss = model(x,y)
    # evaluate the model
    if i%1000==0:
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
# prints
#here are a few tries:
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
# built earlier, anyway, can we improve it? certainly! 
# for example we could use dropout on our attention! we could use normalization layers
# and we could use multiple heads instead of just one! which takes us to the next
# subject which is multi-head attention! which is  really nothing except several normal!
# attention blocks run in parallell! 
# so lets implement these and see how much we can improve upon this
#%%
class MultiHeadAttention(nn.Module):
    def __init__(self, num_head, head_size, embd_size, context_size,bias_attn=False) -> None:
        super().__init__()
        self.num_head = num_head
        self.head_size = head_size
        # we assume the head_size is already split between the num_heads and thus we dont split
        # it again
        # assert head_size//num_head == 0, f'head_size({head_size}) must be divisable by head_num({num_head})'
        # we need to create n heads so lets do it 
        # self.heads = [AttentionHead(context_size, 
        #                             embd_size, 
        #                             head_size, 
        #                             bias_attn) for _ in range(num_head)]
        # but we can also use pytorch's nn.ModuleList which is a better equivalent than python list
        # becasue all the modules it contains are properly registered, and will be visible by all 
        # Module methods. which is not the case for python lists (i.e. .parameters() .children(), 
        # .zero_grad, etc, e.g.) and therefore its best to use modulelist instead of pure python lists.
        # note that, nn.Sequential cant be used, becasue it runs the modules in succesion
        # i.e. serially, one after the other (feeds the output of the previous module to 
        # the next module, etc) which is not what we want. we want to calculate each head
        # independetly and aggregate their outputs so, either a python list or torch moudle list
        # can be used
        # sidenote: this module that we are building, is also known as, masked multi-attention-head
        # becasue we are using the single-head self-attention which uses a constrain we imposed
        # by masking if you recall that!
        self.heads = torch.nn.ModuleList(AttentionHead(context_size, 
                                         embd_size, 
                                         head_size,
                                         bias_attn) for _ in range(num_head))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # note that head_size is usually the embd_size, so it covers the whole 
        # embeddings obviously, and note that each head will work on a portion
        # of the given head_size, if e.g. we have head_size/embd_size = 16
        # if we had 1 head, the head_size would be 16. however if we had 2 heads
        # we had to split the head_size in half for each head, and later on
        # we had to concat them to get the full-size head_size/embd_size.
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
        # ! check -> usually head_size == embd_size so each head works on the 
        # ! whole embedding 
        self.multi_head_attention = MultiHeadAttention(num_head, head_size//num_head, embd_size, context_size,bias_attn)
        # finally the output fc layer 
        self.fc = nn.Linear(head_size, vocab_size)
        
    def forward(self, inputs:torch.Tensor, labels:torch.Tensor=None)->torch.Tensor:
        B,T = inputs.shape
        # print(f'{inputs.shape=} {self.context_size=} {self.embd_size=}')
        token_embds = self.token_embeddings(inputs)
        # dont forget, our positional embd only involves the token position information
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

for i in range(max_iter):
    # read a batch 
    x, y = get_batch('train', batch_size=batch_size)
    logits, loss = model(x,y)
    
    # calculate the smoother loss on multiple batches on train/val splits
    if i%eval_period == 0:
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

# compared to single head attention, with the same hyper parameters, train: 2.4684  val: 2.4759
# and now we got train: 2.4238  val: 2.433 which is better(we also got train: 2.2713  val: 2.288 
# with just increasing the embedding_size, so playing with parameters even blindly making model bigger
# seems to give us a boost), so we had improvements, but we still need a long way ahead of us!
# 
# Ok, to improve upon our results, there are couple of more things we need to add to our attention 
# block. if you look at the paper, we'll see a few concepts that we havent talked about or implemented
# yet, including the feed-forward(position-wise feedforward network) and layernorm, so lets talk about 
# them.
# feed-forward network or as its called in the paper,'position-wise feedforward network', is simply
# a "fully connected network which is applied to each position separately and identically", this 
# consists of two linear layer with a relu activation function inbetween. 
# so its a linear layer with a relu activation function followed by another linear layer basically. 
# you may also hear this network be refered to as 'computation after communication' so to speak!
# signifying the fact that it runs a computation after the attention module.
# the idea behind this extra addition is simple, to provide a higher representation out of attention work
# this network, causes every single token to have a nonlinear transformation and achieve a higher abstraction
# possibly yielding new information benficial to the task. to put it in casual way, it can be seen as though
# the attention gatheres some stats/information, and this step is akin to looking into it and thinking about it
# comming up with some new findings, that is , if attention part is refered to as communication part, this is
# the computation part/ or thinking part for the lack of a better word.
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
        # ! check -> usually head_size == embd_size so each head works on the 
        # ! whole embedding 
        self.multi_head_attention = MultiHeadAttention(num_head, head_size//num_head, embd_size, context_size,bias_attn)
        # now let us create the feedforward network, which basically is a linear layer with relu
        # followed by another linear layer! for this so called network, the in_features and out_features
        # are simply the same, and is embd_size, because its a sandwich layer between our attention output
        # and the last layer. this layer usually is (embedsize,embed_size) if you recall head_size is usally
        # equal to embed_size. also note that in the paper the feedforward network is two linear layers with
        # a relu (one linear layer with a relu plus another linear layer).if you look closely you'll notice 
        # we already have a last layer after the attention, so we dont need to put a nother linear layer
        # afterward, becasue that one linear layer suffices (multiple linear layers, dont provide
        # any higher abstraction anyway, so its prefectly fine)
        # but on the other hand the paper's implementation has another change, it states 
        # the inner layers dim are increased by 4x, so to be faithful to the paper we also 
        # add the second linear layer, with the suggested change, this wont change the output 
        # shape, so we are fine. we just added an extra linear projection layer. 
        # this additional projection operation actually improves the result,however if we simply 
        # use the same dim linear layer (i.e. have sth like linear(head_size, head_size)) adds nothing
        # to the representational power of the network and you wont see anything substantial vs if you
        # completely remove this layer and only use linear/relu only.
        # why this works is becasue, we increase the nonlinear output neurons of the first layer by 4,
        # increasing its representational capacity, and then use the second linear layer to get the output
        # size compatible for the next layer (doing a linear projection), and hence our improvements lie
        # in the nonlinearity this addition provides. 
        self.feedforwardnet = nn.Sequential(nn.Linear(embd_size, head_size*4), nn.ReLU(),
                                            nn.Linear(head_size*4, head_size))
        # finally the output fc layer 
        self.fc = nn.Linear(head_size, vocab_size)
        
    def forward(self, inputs:torch.Tensor, labels:torch.Tensor=None)->torch.Tensor:
        B,T = inputs.shape
        # print(f'{inputs.shape=} {self.context_size=} {self.embd_size=}')
        token_embds = self.token_embeddings(inputs)
        # dont forget, our positional embd only involves the token position information
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

for i in range(max_iter):
    # read a batch 
    x, y = get_batch('train', batch_size=batch_size)
    logits, loss = model(x,y)
    
    # calculate the smoother loss on multiple batches on train/val splits
    if i%eval_period == 0:
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

for i in range(max_iter):
    # read a batch 
    x, y = get_batch('train', batch_size=batch_size)
    logits, loss = model(x,y)
    
    # calculate the smoother loss on multiple batches on train/val splits
    if i%eval_period == 0:
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
# as you can see, we got wrose results than before! despite making the network larger, our loss
# really didnt improve as we expected. 
# is our initial hypothesis that having more blocks and higher nonlinearity/representation is benificial
# wrong? or is there something else thats causing the issue? 
# as you might have guessed, its the latter. we are basically creating more layers, and with
# more layers, we face training issues that we discussed earlier. 
# so how should we tackle this, we cant use BN, becasue BatchNOrm, accumulates the statistics
# from different samples, this is a no no for us. we dont want other samples to interfer with 
# our sample (or basically each other!) in anyway, so what should we do? 
# the paper utilizes two mechanisms or operations to tackle this issue. one being skip-connections
# (also known as residual connction) and the other, layer-normalization. 
# you should be familiar with skip-connections as they are the founding factor or resenets
# and have been extremely influential. so to cut a long story short, skip connections are simply
# connections from input skipping the operations involved in the block they reside and directly 
# being added to the output and then returned the result.(basically F(x) + x)
# as we know, the gradients are distributed equally when they reach addition, so input gets the gradients
# without being weakened due to large depth of the network. 
# so before we implement the layer normalization part, lets see howmuch of a change adding skip-connection
# causes and whether it proves our initial hypothesis about depth and gradient signal weakening or not
# we add this to our AttentionBlock (but we could add this to any submodule)
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
        out = self.attn(inputs) + inputs
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

for i in range(max_iter):
    # read a batch 
    x, y = get_batch('train', batch_size=batch_size)
    logits, loss = model(x,y)
    
    # calculate the smoother loss on multiple batches on train/val splits
    if i%eval_period == 0:
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
# per module output only, nevertheless, we notice, using skip-connection really improved our results.
# now lets add the second operation, layernorm. layernorm(https://arxiv.org/abs/1607.06450) is a normalization layer, just like batchnormalization
# came a year later than batchnormalization paper, but the difference between them is that, unline batchnormalization
# it works on a per sample basis and does not involve using othersamples to normalize a specific sample (basically
# samples dont affect eachother)
# As pytorch docs puts it:
# "Unlike Batch Normalization and Instance Normalization, 
#  which applies scalar scale and bias for each entire channel/plane with the affine option,
#  Layer Normalization applies per-element scale and bias with elementwise_affine"
#! check and expand
# Layer normalization (LayerNorm) and batch normalization (BatchNorm) are both normalization techniques commonly used in deep learning models. While they aim to normalize the input data, they differ in their operation, goal, and outcome. Here's a detailed comparison:
# 1. Operation:
#    - LayerNorm: LayerNorm normalizes the input tensor across the feature dimension (last dimension) independently for each example in the batch. It calculates the mean and variance along the feature dimension and applies a normalization operation.
#    - BatchNorm: BatchNorm normalizes the input tensor across the batch dimension (first dimension) as well as the feature dimension. It calculates the mean and variance along the batch dimension and applies a normalization operation.
# 2. Goal:
#    - LayerNorm: The goal of LayerNorm is to normalize the activations of each individual example in the batch. It aims to reduce the internal covariate shift, helping the network converge faster and generalize better.
#    - BatchNorm: The goal of BatchNorm is to normalize the activations across the batch dimension. It aims to reduce the effects of internal covariate shift, stabilize the network during training, and improve generalization by reducing overfitting.
# 3. Outcome:
#    - LayerNorm: LayerNorm ensures that the mean of each feature across each example is zero and the standard deviation is one. It preserves the relative relationships between the features within each example.
#    - BatchNorm: BatchNorm ensures that the mean of each feature across the entire batch is zero and the standard deviation is one. It introduces dependencies between examples in the batch during training but can be disabled during inference to allow independent predictions.
# 4. Applicability:
#    - LayerNorm: LayerNorm is commonly used in recurrent neural networks (RNNs), such as LSTMs and GRUs, where the normalization is applied along the time steps (sequence length) dimension.
#    - BatchNorm: BatchNorm is typically used in convolutional neural networks (CNNs) and fully connected layers, where the normalization is applied across the batch dimension.
# 5. Training vs. Inference:
#    - LayerNorm: LayerNorm behaves the same during training and inference. It normalizes each example independently, making it suitable for both training and inference.
#    - BatchNorm: BatchNorm behaves differently during training and inference. During training, it normalizes the activations across the batch dimension. During inference, the statistics (mean and variance) are usually calculated using a running average from the training phase, and normalization is applied based on these fixed statistics.
# In summary, LayerNorm and BatchNorm are both normalization techniques, but they differ in terms of the dimension over which normalization is applied, the goal of normalization, and the outcome. LayerNorm normalizes each example independently along the feature dimension, while BatchNorm normalizes across the batch dimension as well. They serve different purposes and are commonly used in different types of neural networks.

# the implementation is similar to the batchnormalization, and it does not require calculating running_mean/var
# we can use the pytorch module just fine, but since its really similar to BN, lets implement it here 
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
        # dims = tuple(i for i in range(inputs.ndim-1,0,-1)) which didnt cause any errors, and infact
        # resulted in a very low loss, however the text generation seemed kind of random, and differed
        # from the pytorch's Layernorm version(0.60 vs 1.90 of pytorch's which is a huge diference). 
        # Further investigation revealed the issue which was caused by the dims involved. basically 
        # We needed to only use the last column which would be (-1) in our specific case. basicallys
        # what we want is to calculate mean/var for the dims that belong to features, e.g. in a 2d case like
        # (B,C), we want to mean/var on dim=1. Having worked with BN, we might also assume the same here 
        # that is if we have sth like (B,T,C) we would want to treat, B,T as batch and aggregate the mean/var
        # along dim=(1,2). however, as we learned earlier, LayerNorm does not involve other samples at all, 
        # and infact if we do this, we'll see despite the loss decreasing rapidly, our text generation are not
        # good at all! so when it comes to LayerNorm we always normalize the feature dimensions, and in our case
        # its just the last diminsion. 
        # what was causing the issue here was this exact issue, since we were calculating the dims dynamically
        # based on the inputs shape, for 2d inputs(ignoring batch) this would work as expected, but for 3d+, 
        # it would aggregate other samples stats and therefore creating the discrepency in the output between 
        # ours and pytorch's.s 
        # (we were calculating the mean/var for dims=(1,2) while Pytorch was only calculating it on the last dim, 
        # and hence the difference. (the output  shows that involving other samples adversly affect our output
        # and hence  why BN is not used and instead LN is used.) 
        # this happened becasue we dynamically tried to infer the dims by looking at the input shape
        # the behavior for 2d shapes will be the same, however, for the 3d shapes, our results differ.
        # we can define the aggregation along feature dims in Pytorch, so if we wanted to get the same 
        # behavior in pytorch we had to write sth like this: 
        # self.ln1 = nn.LayerNorm((context_size,head_size)) 
        # that is instead of useing a single number denoting the dimension's size, we specify the dim's size
        # explicticly. 
        # obviously since we are coding this for our usecase, we dont bother getting the input shape here
        # and use -1 to get the job done, otherwise, we would be getting the dim's size as input just like
        # pytorch and based on the dim's size, decide how to do mean/var. 
        # side note: note that if we use sth self.ln1 = nn.LayerNorm((context_size,head_size))  in our code, 
        # during text generation, we no longer can start with context_size of 1, we must always start with
        # sth like initial_token = torch.zeros(size=(1,8)).int() to get it working otherwise it'd complain
        # about shape mismatch which is expected because unlike our method, its static (while ours dynamically
        # would calculate the mean/var based on the inputsize) anyway, this shouldnt be an issue, becasue we
        # dont need to do this as not only it doesnt benifit us but also creates more hassle!.
        # side note2: aggregation in LN, may be benificial in other domains, as BN was, but for us, now it isnt
        # so we only make the one that works with the last dim! 
        # here's the wrong snipped that would create the wrong result for 3d inputs.
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
        # since we have two blocks, we add skip-connection to both of them here
        # we could aggerate them as one and a add the skip connection to their output
        # but this is less benificial than creating seprate skip-connections for each block
        # to see this in action, uncomment this and comment the latter part and run the test
        # out = self.attn(inputs)
        # out =  self.ffnet(out)
        # return out + inputs 
        # note that when it comes to applying normalization, there are two ways of going about it
        # 1.normalize the outputs of the attention block
        # 2.normalize the inputs before going to the attention block
        # the first one is the one the paper initially used, but later on it was shown that actually
        # normalzing the input yields better result, so we test both cases here
        # so check and see how it affects it! (personally,however, I found the first approach yielding midly
        # better loss, but I only tested it on small scale, so we will test this more)
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

for i in range(max_iter):
    # read a batch 
    x, y = get_batch('train', batch_size=batch_size)
    logits, loss = model(x,y)
    
    # calculate the smoother loss on multiple batches on train/val splits
    if i%eval_period == 0:
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
# for some reason, our layernorm achieves a much lower loss, much quicker, however the text seems to be worse
# than pytorch's for some reason and I have no idea what is causing this! 
# ok there were two issues, 1.our gamma and beta werent trained! becasue we forgot to wrap them in nn.Parameter
# and the second reason was, we were aggregating the last two dims like BN, whereas we should have only used the
# last dim, as we should not involve other samples in normalization. (I explained this thoroughly in the LayerNorm class)
# 
# now how can we improve more? we implemented the paper, basically we implemented transormer from scratch
# and what remains is to test with different hyperparameters to see how well it can generate texts similar 
# to our dataset. (we implemented the decore version, but the encoder as we explained earlier is the same
# without the constrains! well talk about this in a moment )
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

for i in range(max_iter):
    # read a batch 
    x, y = get_batch('train', batch_size=batch_size)
    x,y= tuple(t.to(device) for t in (x,y))
    logits, loss = model(x,y)
    
    # calculate the smoother loss on multiple batches on train/val splits
    if i%eval_period == 0:
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
# which is remarkably better than all of our previous outputs. so as we increased our model capacity
# we witnessed much better results. 
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


for i in range(max_iter):
    # this context manager will take care of the dtype conversions for us
    # when we set enabled=False, the scaler and autocast basically become no op!
    # and we can seemlessly switch between them without any code changes at all!
    with torch.cuda.amp.autocast(enabled=use_mix_precision):
     # read a batch 
        x, y = get_batch('train', batch_size=batch_size)
        x,y= tuple(t.to(device) for t in (x,y))
        logits, loss = model(x,y)
        
        # calculate the smoother loss on multiple batches on train/val splits
        if i%eval_period == 0:
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
        if i%5000==0:
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



#%%
# this was gpt! lets talk about the models, glue activiation ufnction, efficiancy , chatgpt vs us, 
# document completer vs chatgptetc 

#ask chatgpt to write an introduction for each section (and explain someparts as if im 5!)

# explain what an autoregressive mean and whats an autoregressive model and how its related to transformers, rnns or sequences, in depth
# An autoregressive model is a type of statistical model where the current value in a sequence depends on the previous values in the same sequence. It assumes that the output is a function of the previous outputs, making it suitable for modeling sequential data.
# In an autoregressive model, the prediction at each time step is conditioned on the previous values. This can be represented mathematically as:
# y_t = f(y_{t-1}, y_{t-2}, ..., y_{t-n})
# where y_t is the predicted value at time step t, f is the function that maps the previous values to the current prediction, and n is the order of the autoregressive model (i.e., the number of previous values considered).
# Autoregressive models have been extensively used in various domains, including time series analysis, speech recognition, natural language processing, and image generation. They are particularly effective for modeling sequential data with temporal dependencies.
# Now, let's discuss the relationship between autoregressive models and transformers, recurrent neural networks (RNNs), and sequences:
# 1. Transformers:
#    Transformers are a type of neural network architecture that have revolutionized various natural language processing tasks. They are based on the self-attention mechanism, which allows them to model dependencies between different positions in the input sequence. Transformers do not have an inherent autoregressive structure, as they can process all positions of the input sequence in parallel. However, they can be trained in an autoregressive manner using a technique called "masked language modeling" (e.g., in the case of the Transformer-based language model GPT).
# 2. Recurrent Neural Networks (RNNs):
#    RNNs are a class of neural networks that are widely used for modeling sequential data. They have a natural autoregressive structure, where the hidden state at each time step depends on the previous hidden states. RNNs can capture temporal dependencies in sequences by propagating information through time. However, RNNs suffer from vanishing or exploding gradients, limiting their ability to capture long-term dependencies effectively.
# 3. Sequences:
#    Autoregressive models are commonly used to model sequences, where the prediction at each time step depends on the previous values. This can be applied to various types of sequences, such as time series data, text, audio, or images. By considering the autoregressive nature of the data, these models can capture the underlying dependencies and generate coherent and contextually relevant predictions or generate new samples.
# In summary, autoregressive models capture dependencies between previous and current values in a sequence. Transformers, RNNs, and other models can be designed with autoregressive components to model sequences effectively. Transformers, while not inherently autoregressive, can be trained in an autoregressive manner. RNNs have a natural autoregressive structure, allowing them to model sequential data. Autoregressive models are commonly used for modeling various types of sequences, enabling tasks such as prediction, generation, or sequence completion.

# more exlpanation  for each point : 
# Certainly! Let's delve deeper into each point:
# 1. Transformers:
#    Transformers are a type of neural network architecture that has gained significant attention in natural language processing tasks. Unlike traditional sequential models such as RNNs, transformers are based on the self-attention mechanism. This mechanism allows transformers to capture dependencies between different positions in the input sequence without relying on sequential processing. Thus, transformers do not have an inherent autoregressive structure as they can process all positions of the input sequence in parallel.
#    However, transformers can be trained in an autoregressive manner using a technique called "masked language modeling." In this approach, during training, the model is fed with the input sequence where some positions are masked out. The model is then trained to predict the original values at the masked positions based on the unmasked context. This autoregressive training allows the model to learn the conditional dependencies between the previous and current positions in the sequence.
# 2. Recurrent Neural Networks (RNNs):
#    RNNs are a class of neural networks designed to model sequential data. They have a natural autoregressive structure due to their recurrent nature. At each time step, an RNN takes the current input and the previous hidden state as input and produces an output and a new hidden state. The hidden state at each time step encapsulates the information from previous time steps, allowing the model to capture the temporal dependencies within the sequence.
#    RNNs are effective at modeling sequences with long-term dependencies, as they can propagate information through time. However, RNNs suffer from vanishing or exploding gradients, which can make it challenging for them to capture long-term dependencies effectively. This limitation led to the development of more advanced architectures like LSTMs and GRUs, which are variants of RNNs that address the gradient-related issues.
# 3. Sequences:
#    Autoregressive models are commonly used to model various types of sequences, including time series data, text, audio, or images. By considering the autoregressive nature of the data, these models can capture the underlying dependencies and generate coherent and contextually relevant predictions or generate new samples.
#    For example, in time series analysis, autoregressive models such as ARIMA (AutoRegressive Integrated Moving Average) are widely used to predict future values based on past observations. In text generation, autoregressive language models like GPT (Generative Pre-trained Transformer) use the previously generated tokens to predict the next token, allowing for the generation of coherent and contextually relevant text.
#    Autoregressive models are also employed in image generation tasks, such as PixelCNN, where each pixel is generated based on the previously generated pixels in a row-major or column-major order. This autoregressive approach ensures that the generated image is coherent and structurally meaningful.
# In summary, while transformers do not have an inherent autoregressive structure, they can be trained in an autoregressive manner using techniques like masked language modeling. RNNs have a natural autoregressive structure, making them suitable for modeling sequential data. Autoregressive models can be applied to various types of sequences and are effective in capturing dependencies and generating coherent predictions or samples.


# A transformer is a deep learning model architecture introduced in the paper "Attention Is All You Need" by Vaswani et al. in 2017. It was designed as an alternative to recurrent neural networks (RNNs) for tasks like machine translation but has since found applications in various natural language processing (NLP) and computer vision tasks.
# The key idea behind the transformer architecture is the use of self-attention mechanisms. Instead of relying on recurrent connections, transformers utilize self-attention to capture dependencies between different positions in the input sequence. This allows the model to focus on relevant parts of the input sequence, improving parallelization and reducing the dependency on the sequence length.
# Over the years, transformers have undergone various improvements and variations. Some notable developments include:
# 1. Transformer-XL: Introduced in 2019, Transformer-XL addressed the limitation of fixed-length context windows in the original transformer. It introduced a recurrence mechanism to allow the model to consider longer-term dependencies.
# 2. BERT: In 2018, Bidirectional Encoder Representations from Transformers (BERT) revolutionized NLP by pretraining transformers on large amounts of unlabeled text data. BERT models achieved state-of-the-art performance across multiple NLP tasks by fine-tuning on specific downstream tasks.
# 3. GPT-3: Released in 2020, GPT-3 (Generative Pre-trained Transformer 3) is one of the largest transformer models ever built, with 175 billion parameters. It has demonstrated impressive performance on various language-related tasks, including language translation, text generation, and question-answering.
# Transformers have found applications in a wide range of tasks, including machine translation, sentiment analysis, text generation, question-answering, image recognition, and more. Their ability to capture long-range dependencies and model context has made them highly effective in these domains.
# In PyTorch, you can use the `torch.nn.Transformer` module to implement a transformer model. Training a transformer involves several important considerations:
# 1. Dataset Preparation: Prepare your dataset by tokenizing text inputs, creating input sequences, and generating target sequences.
# 2. Model Architecture: Design the transformer architecture by specifying the number of encoder and decoder layers, the size of the embedding and hidden layers, and the attention mechanism used.
# 3. Hyperparameter Tuning: Important hyperparameters to tune include the learning rate, batch size, number of layers, hidden size, and dropout rate. You can experiment with different values and use techniques like grid search or random search to find optimal hyperparameters.
# 4. Training Loop: Train the model using backpropagation and optimize the model parameters. Monitor the loss and perform gradient updates using an optimizer like Adam.
# 5. Evaluation: Evaluate the trained model on a separate validation or test set using appropriate metrics for the specific task. Adjust the model and hyperparameters as necessary based on the evaluation results.
# 6. Regularization Techniques: Apply regularization techniques such as dropout, weight decay, or layer normalization to prevent overfitting and improve generalization.
# 7. Pretrained Models: Utilize pretrained transformer models like BERT or GPT-3, which have been trained on large amounts of data. Fine-tune these models on your specific task to benefit from their transfer learning capabilities.
# Remember that training a transformer can be computationally expensive due to the large number of parameters. It is recommended to use GPUs and distributed training techniques to speed up the training process.
# Overall, transformers have significantly impacted the field of deep learning, especially in NLP tasks. They provide a powerful alternative to traditional recurrent architectures and have achieved state-of-the-art performance in various domains.

# Certainly! Let's dive deeper into the first three paragraphs to provide a more detailed explanation of the transformer architecture, its key idea, and the subsequent developments:
# 1. Transformer Architecture:
#    The transformer architecture is a type of neural network model introduced in the paper "Attention Is All You Need". It was primarily designed as an alternative to recurrent neural networks (RNNs) for sequence-to-sequence tasks, such as machine translation. Unlike RNNs, which process sequential data sequentially, transformers operate on the entire input sequence simultaneously. This parallelization allows for more efficient computation and faster training.
#    The transformer architecture consists of two main components: the encoder and the decoder. The encoder takes the input sequence and maps it into a set of high-dimensional representations, capturing the contextual information of each input element. The decoder then generates the output sequence based on the encoder's representations and the previously generated output elements. The key innovation of the transformer lies in its use of self-attention mechanisms, which enable the model to capture relationships between different positions in the input sequence.
# 2. Key Idea - Self-Attention Mechanism:
#    The key idea behind the transformer architecture is the self-attention mechanism. Self-attention allows the model to weigh the importance of different input elements when generating a specific output element. It achieves this by computing attention scores between pairs of input elements, determining how much each element contributes to the representation of another element. These attention scores are then used to compute a weighted sum of the input elements, producing the final output representation.
#    Self-attention mechanisms enable the model to focus on the most relevant parts of the input sequence, irrespective of their positions. This attention-based approach provides a more flexible and powerful way of modeling dependencies between elements compared to the fixed sequential processing of RNNs. By attending to relevant information in the input sequence, transformers can capture long-range dependencies and produce more accurate representations.
# 3. Developments and Variations:
#    Since the introduction of the transformer architecture, several developments and variations have emerged to address specific challenges or improve performance in different tasks. Two notable examples are Transformer-XL and BERT:
#    - Transformer-XL: Introduced in 2019, Transformer-XL aimed to address the limitation of fixed-length context windows in the original transformer. It introduced a recurrence mechanism, allowing the model to consider longer-term dependencies within the input sequence. This recurrence mechanism, called "relative positional encoding," enables the model to capture context beyond the window size.
#    - BERT: In 2018, BERT (Bidirectional Encoder Representations from Transformers) made significant advancements in NLP tasks. BERT models are pretrained on large amounts of unlabeled text data using a masked language modeling objective. This pretraining enables the model to learn contextual representations of words and sentences. BERT models can then be fine-tuned on specific downstream tasks, achieving state-of-the-art performance across various NLP benchmarks.
# These developments highlight the adaptability and versatility of the transformer architecture. Researchers have explored different modifications, improvements, and pretraining techniques to make transformers more effective in specific domains and tasks. The transformer architecture has proven to be a transformative advancement in deep learning, particularly in the field of natural language processing, revolutionizing the way we approach sequence modeling and generation tasks.
#
# explain  more
#Certainly! Let's delve deeper into the third point, focusing on the developments and variations in transformer architecture, particularly Transformer-XL and BERT:
#
# 1. Transformer-XL:
#    Transformer-XL is an extension of the original transformer architecture introduced in 2019. It addresses the limitation of fixed-length context windows in the original transformer, which can restrict the model's ability to capture long-range dependencies.
#    The main innovation of Transformer-XL is the introduction of a recurrence mechanism called "relative positional encoding." In the original transformer, positional encoding provides information about the absolute position of each element in the input sequence. However, it does not explicitly model the relative distances between elements.
#    Transformer-XL overcomes this limitation by introducing relative positional encoding, which allows the model to capture relative distances between input elements. By considering relative positions, the model can capture longer-term dependencies within the input sequence, exceeding the fixed context window size of the original transformer.
#    This recurrence mechanism enables Transformer-XL to handle longer sequences and better capture context beyond the fixed window size. It has been particularly effective in tasks that require modeling long-range dependencies, such as language modeling and document understanding.
# 2. BERT (Bidirectional Encoder Representations from Transformers):
#    BERT, introduced in 2018, made significant strides in natural language processing (NLP) tasks by leveraging transformer-based architectures. BERT pioneered the use of unsupervised pretraining followed by supervised fine-tuning, demonstrating remarkable performance across various NLP benchmarks.
#    BERT is pretrained on large amounts of unlabeled text data using a masked language modeling (MLM) objective. During pretraining, a certain percentage of input tokens are randomly masked, and the model is trained to predict the original masked tokens based on the surrounding context. This process allows BERT to learn contextual representations of words and sentences, capturing rich semantic and syntactic information.
#    After pretraining, BERT models can be fine-tuned on specific downstream tasks with labeled data. This fine-tuning stage involves adding task-specific layers and training the model on the task-specific objective. By leveraging the knowledge learned during pretraining, BERT models achieve state-of-the-art performance on a wide range of NLP tasks, including text classification, named entity recognition, question-answering, and more.
#    BERT's bidirectional approach, where the model can access both left and right context during pretraining, allows it to capture a deeper understanding of the language. This bidirectional context modeling is achieved through the use of a "masked self-attention" mechanism, where attention is computed not only between different input elements but also between masked and unmasked elements.
#    BERT has had a profound impact on NLP research and applications, setting new benchmarks and pushing the boundaries of language understanding tasks. It has also paved the way for subsequent models, such as GPT (Generative Pre-trained Transformer) and RoBERTa, which further improved upon the BERT architecture.
# In summary, Transformer-XL and BERT are notable developments in the transformer architecture. Transformer-XL introduces a recurrence mechanism to capture longer-range dependencies, while BERT leverages unsupervised pretraining followed by supervised fine-tuning to achieve state-of-the-art performance on a range of NLP tasks. These advancements have significantly influenced the field of deep learning and have propelled the transformer architecture to the forefront of NLP research and applications.

# more explanations 
# Certainly! Let's delve even deeper into Transformer-XL and BERT, exploring their architecture and the specific innovations they bring to the transformer model:
# 1. Transformer-XL:
#    Transformer-XL is an extension of the original transformer architecture that addresses the limitation of fixed-length context windows. It introduces a recurrence mechanism called "relative positional encoding" to capture longer-range dependencies within the input sequence.
#    In the original transformer, positional encoding provides information about the absolute positions of elements in the input sequence. However, it does not explicitly model the relative distances between elements. This limitation can lead to difficulties in capturing long-range dependencies, especially when the distance between relevant elements is larger than the fixed context window.
#    Transformer-XL overcomes this limitation by incorporating memory mechanisms into the transformer architecture. It introduces two types of memories: segment-level recurrence and positional recurrence.
#    - Segment-Level Recurrence: Transformer-XL maintains a memory of previously seen segments and incorporates this information into the current segment's processing. By doing so, the model can capture dependencies that span across segments, enabling it to consider longer-term context.
#    - Positional Recurrence: To capture dependencies within a segment, Transformer-XL introduces a recurrence mechanism at the positional level. Instead of processing the segment in a strictly left-to-right manner, it allows information to flow from any position to any other position within the segment. This allows the model to capture longer-range dependencies within the fixed context window.
#    By incorporating these recurrence mechanisms, Transformer-XL can capture longer-term dependencies both within segments and across segments. This makes it particularly effective for tasks that require modeling long-range context, such as language modeling, where understanding the context beyond a fixed window is crucial.
# 2. BERT (Bidirectional Encoder Representations from Transformers):
#    BERT, introduced in 2018, made significant advancements in NLP tasks by leveraging transformer-based architectures. It introduced the concept of unsupervised pretraining followed by supervised fine-tuning, allowing the model to learn rich contextual representations of words and sentences.
#    BERT follows a two-step process: pretraining and fine-tuning.
#    - Pretraining: BERT is pretrained on large amounts of unlabeled text data using a masked language modeling (MLM) objective. During pretraining, a certain percentage of input tokens are randomly masked, and the model is trained to predict the original masked tokens based on the surrounding context. By training on this MLM objective, BERT learns to capture deep contextual information and understand the relationships between words.
#    - Fine-Tuning: After pretraining, BERT models are fine-tuned on specific downstream tasks using labeled data. During fine-tuning, task-specific layers are added on top of the pretrained BERT model, and the entire model is trained on the task-specific objective. Fine-tuning allows BERT to adapt its learned representations to the specific task at hand.
#    The bidirectional nature of BERT is a key innovation. Unlike the original transformer, which processes input sequentially in one direction, BERT employs a "masked self-attention" mechanism. This mechanism allows the model to access both left and right context during pretraining, providing a deeper understanding of the language.
#    By leveraging the contextual information learned during pretraining, BERT models achieve remarkable performance across various NLP tasks. The ability to capture rich semantic and syntactic information makes BERT highly effective for tasks such as text classification, named entity recognition, question-answering, and more.
#    The success of BERT has spurred further developments, with subsequent models like GPT and RoBERTa adopting similar approaches to achieve even better performance on NLP benchmarks.
# In summary, Transformer-XL introduces recurrence mechanisms to capture longer-range dependencies within the input sequence, while BERT leverages unsupervised pretraining followed by supervised fine-tuning to capture rich contextual representations of words and sentences. These advancements have significantly impacted the field of deep learning, particularly in the realm of NLP, setting new benchmarks and pushing the boundaries of language understanding tasks.

# transformer in more depth
# Certainly! Here's an in-depth explanation of transformers, including their underlying idea, how they work, the different versions, and the importance of libraries like Hugging Face:
# 1. Underlying Idea:
#    Transformers are a type of neural network architecture introduced in the paper "Attention Is All You Need" to tackle sequence-to-sequence tasks, such as machine translation. The underlying idea of transformers is to enable parallel processing of the entire input sequence, as opposed to the sequential processing of recurrent neural networks (RNNs).
#    The key innovation of transformers lies in their use of self-attention mechanisms. Self-attention allows the model to weigh the importance of different input elements when generating a specific output element. By computing attention scores between pairs of input elements, transformers determine how much each element contributes to the representation of another element. This attention-based approach provides a flexible and powerful way of modeling dependencies between elements, capturing long-range relationships effectively.
# 2. How Transformers Work:
#    Transformers consist of two main components: the encoder and the decoder. Let's explain their functioning:
#    - Encoder: The encoder takes the input sequence and maps it into a set of high-dimensional representations. It applies self-attention mechanisms to capture relationships between different positions in the input sequence. The encoder stack typically consists of multiple layers, each containing a self-attention mechanism followed by feed-forward neural networks. The self-attention mechanism allows the encoder to attend to relevant information from the entire input sequence, producing rich representations for each element.
#    - Decoder: The decoder generates the output sequence based on the encoder's representations and the previously generated output elements. Like the encoder, the decoder also consists of multiple layers, but it additionally includes an attention mechanism that allows it to attend to the encoder's representations. This attention mechanism enables the decoder to focus on the most relevant parts of the input sequence while generating the output sequence step by step.
#    By leveraging self-attention mechanisms in both the encoder and decoder, transformers can capture the relationships between input and output elements efficiently. This parallel processing of the entire sequence allows transformers to model long-range dependencies and generate accurate representations.
# 3. Versions of Transformers:
#    Since the introduction of the transformer architecture, several versions and variations have emerged to address specific challenges and improve performance in different tasks. Some notable versions include:
#    - Transformer-XL: Introduced in 2019, Transformer-XL incorporates recurrence mechanisms to capture longer-range dependencies within the input sequence. It introduces segment-level recurrence and positional recurrence to handle longer sequences and context beyond the fixed window size.
#    - BERT (Bidirectional Encoder Representations from Transformers): BERT revolutionized the field of NLP in 2018. It employs unsupervised pretraining followed by supervised fine-tuning to learn contextual representations of words and sentences. BERT models are pretrained on large amounts of unlabeled text data using a masked language modeling objective. After pretraining, they can be fine-tuned on specific supervised tasks, achieving state-of-the-art performance across various NLP benchmarks.
#    - GPT (Generative Pre-trained Transformer): GPT models, introduced in 2018, focus on language generation tasks. They adopt a similar pretraining and fine-tuning approach as BERT but use a generative objective during pretraining. GPT models have been successful in tasks like text generation, language translation, and summarization.
#    - RoBERTa: RoBERTa, introduced in 2019, builds upon BERT's architecture and training methodology. It modifies the training procedure and hyperparameters to achieve better performance. RoBERTa has shown improved results on a wide range of NLP tasks.
#    These versions highlight the continuous advancements and refinements in the transformer architecture, addressing specific challenges and pushing the boundaries of performance in various domains.
# 4. Importance of Libraries like Hugging Face:
#    Hugging Face is a popular open-source library that provides tools and pre-trained models for natural language processing, with a strong focus on transformers. It offers a wide range of pre-trained transformer models, including BERT, GPT, RoBERTa, and many others.
#    Hugging Face's library simplifies the process of working with transformers, providing easy-to-use APIs and tools for tasks like text classification, named entity recognition, question-answering, and more. It also enables users to fine-tune pre-trained models on specific downstream tasks with minimal effort.
#    The library plays a crucial role in democratizing access to state-of-the-art transformer models, making them more accessible to researchers, developers, and practitioners. It has become a central hub for sharing, exploring, and utilizing transformer models, fostering rapid progress in the field of NLP.
# In summary, transformers are neural network architectures that leverage self-attention mechanisms to process input sequences in parallel and capture long-range dependencies effectively. Versions like BERT, GPT, and RoBERTa have further improved upon the original transformer architecture, achieving remarkable results in various NLP tasks. Libraries like Hugging Face have played a vital role in making transformer models more accessible and facilitating their usage in a wide range of applications.The importance of Hugging Face library in the context of transformers cannot be overstated. Hugging Face is an open-source library that provides a comprehensive set of tools and pre-trained models for natural language processing, with a particular focus on transformers.
# Here are some key aspects and the relation of Hugging Face to transformers:
# 1. Pre-trained Models: Hugging Face offers a vast collection of pre-trained transformer models, including BERT, GPT, RoBERTa, and more. These models have been trained on large amounts of text data and have learned rich contextual representations that can be fine-tuned for specific NLP tasks.
# 2. Model Hub: Hugging Face provides a centralized Model Hub, which serves as a repository for sharing and accessing pre-trained transformer models. Researchers and practitioners can upload their trained models to the Model Hub, making them readily available for the community. This fosters collaboration, knowledge sharing, and enables easy experimentation with the latest transformer models.
# 3. Tokenizers: Hugging Face provides efficient tokenization libraries that can handle various text processing tasks. Tokenization is a crucial step in transforming raw text data into a format suitable for input to transformer models. Hugging Face's tokenizers support multiple languages and offer customizable options for different tokenization strategies.
# 4. Model Pipelines: Hugging Face offers high-level model pipelines that simplify the process of using transformer models for specific NLP tasks. These pipelines provide easy-to-use APIs for tasks like text classification, named entity recognition, question-answering, and more. With just a few lines of code, users can leverage pre-trained models and achieve state-of-the-art performance on various NLP benchmarks.
# 5. Fine-tuning Framework: Hugging Face provides a fine-tuning framework that allows users to adapt pre-trained transformer models to specific downstream tasks. Fine-tuning involves training the pre-trained models on task-specific data to improve their performance on specific NLP tasks. Hugging Face's framework simplifies the fine-tuning process by providing utilities and APIs for efficient model adaptation.
# 6. Community and Documentation: Hugging Face has a vibrant community of developers, researchers, and NLP enthusiasts. They actively contribute to the library, share their experiences, and provide support to fellow users. Hugging Face's documentation is comprehensive and user-friendly, making it easy for newcomers to get started with transformers and the library.
# Overall, Hugging Face plays a pivotal role in making transformers more accessible and user-friendly. It provides a platform for sharing, exploring, and utilizing pre-trained models, tokenization tools, model pipelines, and fine-tuning frameworks. The library has become a go-to resource for researchers, developers, and practitioners working with transformers, enabling rapid progress and advancements in the field of NLP.


# lash attention latest improvements https://arxiv.org/pdf/2205.14135.pdf
# linformer, sparseattention(openai), Performer, Reformer, etc
# https://www.youtube.com/watch?v=1RaIS98jj1Q  
# Transformers are slow and memory-hungry on long sequences, since the time and memory complexity
# of self-attention are quadratic in sequence length. Approximate attention methods have attempted
# to address this problem by trading off model quality to reduce the compute complexity, but often do
# not achieve wall-clock speedup. We argue that a missing principle is making attention algorithms IO-
# aware—accounting for reads and writes between levels of GPU memory. We propose FlashAttention,
# an IO-aware exact attention algorithm that uses tiling to reduce the number of memory reads/writes
# between GPU high bandwidth memory (HBM) and GPU on-chip SRAM. We analyze the IO complexity
# of FlashAttention, showing that it requires fewer HBM accesses than standard attention, and is
# optimal for a range of SRAM sizes. We also extend FlashAttention to block-sparse attention, yielding
# an approximate attention algorithm that is faster than any existing approximate attention method.
# FlashAttention trains Transformers faster than existing baselines: 15% end-to-end wall-clock speedup
# on BERT-large (seq. length 512) compared to the MLPerf 1.1 training speed record, 3 speedup on
# GPT-2 (seq. length 1K), and 2.4 speedup on long-range arena (seq. length 1K-4K). FlashAttention
# and block-sparse FlashAttention enable longer context in Transformers, yielding higher quality models
# (0.7 better perplexity on GPT-2 and 6.4 points of lift on long-document classification) and entirely new
# capabilities: the first Transformers to achieve better-than-chance performance on the Path-X challenge
# (seq. length 16K, 61.4% accuracy) and Path-256 (seq. length 64K, 63.1% accuracy).

#
# https://python.plainenglish.io/swin-transformer-from-scratch-in-pytorch-31275152bf03
# about vision transormers 
# Certainly! Here's an in-depth explanation of vision transformers, including their underlying idea, how they work, the different versions, and the importance of libraries like Hugging Face:
# 1. Underlying Idea:
#    Vision transformers are a variant of transformers that have been adapted for computer vision tasks. While convolutional neural networks (CNNs) have traditionally been the dominant architecture for image processing, vision transformers aim to explore the effectiveness of transformers in visual tasks. The underlying idea is to apply self-attention mechanisms to capture global dependencies in images and enable parallel processing of the entire image.
#    The key intuition behind vision transformers is that images can be reshaped into sequences of patches, similar to sentences in natural language processing. These patches are then fed into the transformer model, allowing it to capture relationships between patches and learn meaningful representations.
# 2. How Vision Transformers Work:
#    Vision transformers consist of an encoder, similar to the original transformer architecture. Let's explain their functioning:
#    - Patch Extraction: The input image is divided into a set of smaller patches. Each patch represents a local region of the image and is typically represented as a vector.
#    - Positional Embedding: Similar to transformers in NLP, vision transformers require positional information to capture spatial relationships between patches. Positional embeddings are added to each patch vector to encode its relative position within the image.
#    - Encoder: The encoder processes the sequence of patch embeddings and performs self-attention operations to model the dependencies between patches. The self-attention mechanism allows each patch to attend to other patches, capturing global relationships in the image. The encoder stack typically consists of multiple layers, each containing self-attention mechanisms and feed-forward neural networks.
#    - Classification Head: At the end of the encoder, a classification head is added to produce the final output. This head can be a simple linear layer that maps the transformer's output to the desired number of classes for classification tasks.
#    By leveraging self-attention mechanisms, vision transformers can capture long-range dependencies and global context in images, allowing them to achieve competitive performance on various computer vision tasks.
# 3. Versions of Vision Transformers:
#    Vision transformers are a relatively new development, and several versions and variations have emerged to explore their effectiveness in different settings. Here are a few notable versions:
#    - ViT (Vision Transformer): The Vision Transformer introduced in the paper "An Image Is Worth 16x16 Words" is the foundational work in this field. It demonstrated that transformers can achieve competitive performance on image classification tasks when applied to image patches.
#    - DeiT (Data-efficient Image Transformers): The Data-efficient Image Transformers introduced in the paper "Training ViT Models on Noisy Datasets Improves ImageNet Classification" focused on improving the data efficiency of vision transformers. It introduced techniques such as distillation and noisy student training to achieve state-of-the-art performance even with limited labeled data.
#    - TNT (Transformer in Transformer): The Transformer in Transformer introduced in the paper "Transformer in Transformer" explores an architecture with multiple layers of transformers. It applies a second-level self-attention mechanism within each transformer layer, allowing for more fine-grained modeling of local and global dependencies.
#    These versions highlight the ongoing research and exploration in the field of vision transformers, aiming to improve performance, efficiency, and applicability in various computer vision tasks.
# 4. Importance of Libraries like Hugging Face:
#    Hugging Face, as mentioned in the previous response, is an essential library in the context of vision transformers. It provides tools and resources that facilitate working with vision transformers and accelerate research and development in computer vision. Here's how it relates to vision transformers:
#    - Pre-trained Models: Hugging Face's Model Hub includes pre-trained vision transformer models like ViT and DeiT. These pre-trained models can be readily used for various computer vision tasks, such as image classification and object detection.
#    - Transformers Library: Hugging Face's Transformers library, originally focused on NLP, has expanded to include vision transformers as well. It provides a consistent and user-friendly API for working with transformers, including vision transformers. This allows developers and researchers to leverage existing methodologies and tools to work with vision transformer models seamlessly.
#    - Data Processing and Augmentation: Hugging Face's library also includes utilities for data processing and augmentation specific to computer vision tasks. These tools help prepare and preprocess image data, enabling efficient training and evaluation of vision transformer models.
#    - Community and Documentation: Hugging Face has a thriving community of researchers, developers, and enthusiasts working in the field of computer vision and transformers. The community actively contributes to the library, shares their experiences, and provides support to fellow users. Hugging Face's documentation is comprehensive and user-friendly, allowing users to quickly understand and utilize vision transformer models.
#    The availability of pre-trained models, consistent APIs, data processing tools, and a supportive community makes Hugging Face an essential resource for researchers and practitioners working with vision transformers. It simplifies the development and deployment of vision transformer models, fostering rapid progress and advancements in the field of computer vision.
# In summary, vision transformers adapt the transformer architecture for computervision tasks by dividing images into patches, applying self-attention mechanisms, and capturing global relationships. They have several versions such as ViT, DeiT, and TNT, each exploring different aspects of vision transformer architectures. Libraries like Hugging Face provide pre-trained models, tools for data processing, and a supportive community to facilitate working with vision transformers and accelerate research in computer vision.
#
#
# https://www.youtube.com/watch?v=2ih6BHD4v3I
# https://www.youtube.com/watch?v=VoRQiKQcdcI
# https://www.youtube.com/watch?v=3B6q4xnuFUE
# https://www.youtube.com/watch?v=HobIo2oT0xY
# https://www.youtube.com/watch?v=tFYxJZBAbE8