#%%
# in the name of God the most compassionate the most merciful
# in this section we will have a look at how a GPT model works
# in this section we will implement attention module, and see 
# how it works and lean more about its (sel-attention, cross
# attention, multi-head attention, etc)
# 
# so how are we going about this. we need to create a language model
# and then progressively imporve it with attention mechanism.
# we will basically be creating a kind of chatgpt (minus its chat capability and
# its obviously great perormance :)) 
# to keep this as simple as possible and not lose track of the important concepts involved,
# we can use our initial bigram model as the base model and work on improving that with attention.
# so lets start
#
# first lets import the basic stuff 

# for type hints
from collections.abc import Iterable

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
# now lets create our atoi and itoa dictionaries for mapping
# lets create our unique character list, which is infact our vocabulary or vocab for short
vocab_list = sorted(set(''.join(dataset)))
vocab_size = len(vocab_list)
# ! explain token
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
# tiktokenize repo explains BPE in rather friendlier way: 
# Models don't see text like you and I, instead they see a sequence of numbers (known as tokens). 
# Byte pair encoding (BPE) is a way of converting text into tokens. It has a couple desirable properties:
# It's reversible and lossless, so you can convert tokens back into the original text
# It works on arbitrary text, even text that is not in the tokeniser's training data
# It compresses the text: the token sequence is shorter than the bytes corresponding to the original text. 
# On average, in practice, each token corresponds to about 4 bytes.
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
# lets also create two helper functions to convert a list of these to the other part
def encode(characters:Iterable[str] ):
    return [atoi[c] for c in characters]

def decode(token_lst:Iterable[int]):
    return [itoa[n] for n in token_lst]

# lets test these 
print(f'{encode(dataset[:10])}')
print(f'{decode(encode(dataset[:10]))}')

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
# be able to look at the past, and look at the past characters, and based on that do sth. this is the essence
# of attention (although this is not accurate, but for now this is the case, we will elaborate on this and expand
# this metaphor and reasoning inshallah)
# 
# lets first tokenize the whle dataset or corpus as its usually called in nlp nomenclature!
data = encode(dataset)
# since we are using pytorch lets convert that to a tensor
data_tensor = torch.tensor(data)
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
# of data from the dataset and feed it to the transformer. and these chunks, of course has a length, what length?
# we usually specify a maximum_length for the input on which our transformer model works.
# this maximum_length is usually refered to as block_size or context_size.
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
# lets elaborate on this more. 
# 1-in the context of 18, the next character is 47
# 2-in the context of 18,47, the next character is 56
# 3-in the context of 18,47,56, the next character is 57
# 4-in the context of 18,47,56,57 the next character is 58
# 5-in the context of 18,47,56,57,58 the next character is 1
# 6-in the context of 18,47,56,57,58,1 the next character is 15
# 7-and finally, in the context of 18,47,56,57,58,1,15 the next character is 47
# so this is infact 8 7 individual example embedded in a single context, 
# since we want the xontext_size to be 8, then we should grab one more character to have 
# 8 contexts
# lets visualize this in example in code
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
# up to the max_length which is our context_size of 8 in our case.
# ! recheck and elaborate to clear any confusion
# !so by doing this, the transformer can learn how to predict/create/genrate text up to context_size, and after
# !it reached thta, we have to truncate it, becausse the transformer model never recieves more than the
# !context_size as input when its predicting the next character.
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
    # we want a batch of 4 of 8 characters (context-size). 
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
        # our bigram model was nothing more than a 2d array of vocab_size, we can achive that using 
        # a single weight matrix or torch.Embedding. we use torch.Embedding to not reinvent the wheel!
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
            # we can go on permute the dimensions, like this and make crossentropy happy 
            # logits = logits.permute((0,2,1))
            # however, if for some reason the output of logits in the form of (4,8,65) is not ideal, 
            # maybe for example because really what it is 32 examples arranged in a (4,8) shape and 
            # we rather a normal 2d tensor of shape (32,65), 
            # we can instead, do just that, flatten the two dimensions into one and carry on!
            # we basically are concatenating the batch and time dimensions
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
            # stays the same regardless of calculating the loss or not 
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
            # each time obviously, so we only take the last timestep to see what the model predicted
            logits = logits[:,-1,:] # this now becomes (B,C) instead of the initial (B,T,C)
            probs = torch.softmax(logits, dim=-1) 
            # now lets sample from it
            idx_next_char = torch.multinomial(probs, num_samples=1, replacement=True) # shape is (B,1)
            # now lets add this to the next input to be fed to the model 
            # since idx has the shape(batch, T), we should add this tothe second dimension
            # to the time dimension/ or sequence dimension. this as the loop goes on, 
            # creates the shape (B,T+1) 
            #this doesnt make sense for this particular model, becasue we are always checking
            # the next character given the previous one, so all the concatenation we are doing
            # is just useless. the reason we are implementing this like this, is to create a 
            # base, so that we can improve upon it when we add attention later on which will use
            # the history of previous characters.
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
# which is pretty close to the loss we got here, signifying its a pretty decent value to begin with.
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

batch_size = 64
context_size = 8
vocab_size = len(vocab_list)
max_iter = 20000

model = BigramModel(vocab_size)
optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-3)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
# relying on single batch loss is not a good idea to measure the performance of a model. moreover
# relying on training loss, is not good either, so it would be much better if we considered more batches
# for loss and even better we could also investivate the models performance on our validation set. 
# so lets do just this and define a function that calculates loss for training and validation sets alike
# but considers more batches for loss calculation

@torch.no_grad()
def evaluate_loss (iterations, device=None):
    results={}
    # before calculating the loss, lets switch to eval mode,although for our specific case this doesnt matter
    # but its goo practice, as later on, we will add layers that their behavior do change depending on traing
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
# train_loss: 4.7491,  val_loss: 4.7430
# train_loss: 3.6673,  val_loss: 3.6670
# train_loss: 3.0460,  val_loss: 3.0511
# train_loss: 2.7335,  val_loss: 2.7492
# train_loss: 2.5948,  val_loss: 2.6156
# train_loss: 2.5292,  val_loss: 2.5470
# train_loss: 2.4982,  val_loss: 2.5194
# train_loss: 2.4807,  val_loss: 2.4991
# train_loss: 2.4737,  val_loss: 2.5036
# train_loss: 2.4650,  val_loss: 2.4949
# train_loss: 2.4648,  val_loss: 2.4910
# train_loss: 2.4626,  val_loss: 2.4836
# train_loss: 2.4570,  val_loss: 2.4893
# train_loss: 2.4517,  val_loss: 2.4823
# train_loss: 2.4508,  val_loss: 2.4819
# train_loss: 2.4564,  val_loss: 2.4839
# train_loss: 2.4555,  val_loss: 2.4802
# train_loss: 2.4503,  val_loss: 2.4927
# train_loss: 2.4522,  val_loss: 2.4905
# train_loss: 2.4572,  val_loss: 2.4888
#
# now lets check its output again after some training 
inputs = torch.zeros(size=(1,1), device=device, dtype=torch.int32)
output = model.generate(inputs, max_token_count=500)
print(''.join(decode(output.squeeze(0).tolist())))
# prints :
# Ane pod BUL:
#
# Oringe
# nfote s Rinsou oe the,
# An ETwe
#
# PUSENI bmilft!'d ate t Ifur
# Athomeg?
# Whagre,
# TCo belangead ne bl e, bednth ftor veso.
# US tla lin:
# Yourthiee he w whetitrd;
# Angoknghothartro ll heshethor hie douluk t, s se, denopit s h wom uspouct blyowie s'nos t prou gmesat
# Shd, tieithy.
# Asiour hofo bay avear hanousthaie
# Calonth wolulo.
#
# The hancondors pthighar:
# WAME outhaser d!
# S:
# h im t w s, inowo ORILOUCHoou isecetoubecompis
# Ay gnd, teve luthorea, therpeclsthevecthede fichim?
# PUSa VI aken 
# 
# which looks much better than the initial random output we got earlier.
#%% 
torch.manual_seed(255)
# now lets add attention mechanism to our base model. 
# attention mechanism at its core tries to take advantage of the rich information embedded
# in the sequence. for our work, this is specifically about the past history but in general 
# attention can utilize both past and future connections/sequence tokens. what we described 
# here just now is not exactly accurate, but gives us a foundation to build our intuition as
# we continue on. we will elaborate more of course and hopefully get it all.
# before we venture any further into the crux of the matter, let us learn about a technique
# thats used to efficiently implement attention mechanism.
# for this purpose,lets imagine we have a simple input like the following: 
# lets create and input of the following shape
B,T,C = (4,8,2)
# to make it more intuitive lets make a tensor with known numbers andthen reshape it
x = torch.arange(0,64,dtype=torch.float).view(B,T,C)
# imagine we have an input like what we encountered previously in our examples. in this sample 
# input, we have a batch of 4 samples, each having 8 sequences with each sequence having a vector of 2 values
# what we are planning to do is to provide a way by which each token can communicate with other 
# tokens. we have 8 tokens in our sequence. so we want our tokens to be able to communicate with
# all previous tokens that came before it. the reason we are only looking in the past token is 
# simply becasue the we are trying to perdict the future, so it only makes sense to look at the
# past and current timestamp and infer on what to do for the future.  
# so the easiest way to implement a kind of communication between tokens could be to sum or average the
# values of all previous tokens plus the current one as a way of taking into account their contribution
# to the final answer.
# that is, lets say if we are currently at token 5, we take the average of 
# the current token and all previous tokens before it, effectively making a feature vector that
# reflects our current status of the sequence so far, having taken all previous tokens/steps up to now.
# note that as you may also have thought, summing or averaging arent the best way to model such interations.
# in fact they are an extremely weak form of interaction between tokens,
# this kind of communicating is extremely lossy so to speak, that is we lose a great deal of information 
# concerning the underlying relationships between tokens, their arrangements,their implicit interactions,
# semantics, etc. but for now this is ok. we will later on see how to bring back such information.
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
# You may find out that, some researchers refer to this operation here as BoW, or bag of words. 
# We are effectively averaging embeddings here and averaging embeddings can be considered a form of 
# Bag of Words (BoW) representation. In BoW, the focus is on the occurrence and frequency of words, 
# rather than their order or structure. By averaging embeddings, we are essentially treating each word 
# as an independent feature and capturing its representation in the form of a numerical vector.
#
# While averaging embeddings does not capture the exact frequency of each word, it does capture the 
# overall distribution and semantic information present in the text. Similar to BoW, this approach 
# disregards word order and focuses on the presence and representation of words. However, it should be
# noted that averaging embeddings may preserve some semantic relationships between words, which BoW 
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
# Here's a step-by-step overview of the BoW process:
# 1. Tokenization: The text is split into individual words or tokens. Punctuation marks, whitespace, and other
#    special characters are usually removed or treated as separate tokens.
# 2. Vocabulary Creation: A vocabulary is created by taking all unique words from the entire corpus 
#    (collection of documents). Each unique word is assigned a unique index or position in the vocabulary.
# 3. Vectorization: Each document is represented as a feature vector, typically a one-hot encoding or a count
#    vector. In a one-hot encoding, each word in the vocabulary corresponds to a binary feature, and the vector
#    contains 1s in the positions where the word occurs and 0s elsewhere. In a count vector, the value at each 
#    position represents the frequency of the corresponding word in the document.
# 4. Classification or Analysis: The resulting feature vectors can be used as input to machine learning models
#    for tasks such as text classification, sentiment analysis, document clustering, or information retrieval.
# Needless to say, BoW has some limitations. It does not capture the semantic meaning or context of words, as
# it treats each word independently. It also ignores the grammar and word order. However, BoW is simple, 
# efficient, and can be a useful baseline representation for various NLP tasks.
# 
# so to recap one more time, we are basiaclly treating each timestep/sequence dimension, as a word, so we have
# 8 tokens/words, and we are averaging them (we are infact averaging their embeddings, but thats obvious!)
# so we got ourselves bow representation!
# now back to our discussion, if we  try to visualize the results we get 
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
# the second col of 'b', then summed the results, makes up the second col of first row in c. and this goes on for the rest of the matrixes. this is
# what we learned back in higheschool, so what is it exactly that we are learning exactly?
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
# we may very well be able to achieve our goal! how you may ask? the answer is using torch.tril!
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
# as you can see the the right half is set to zero and we are left with a triangle shape of 1s! on the left side
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
a = a/a.sum(dim=1, keepdim=True)
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
# prevbiously we had : 
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
# now back to our discussion. as we just saw, the traingualr shape in our weighted
# sum matrix, allows that each token at t dimension, can only interact with the tokens
# before it.
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
# so effectively what happens here is that, we setting each entery in trail_tensor
# with zero to -inf, and leave the rest as zeros. when this tensor goes through softmax
# the 0s will be 1s and -infs will be 0s before they are normalized, when they are normalized
# the probablity of 1 will be split between all the enteries with the value of 1.
# so the first row has a single 1, so it will be 1.0, the second row has two 1s, so
# each one will take 0.5, the third will have 3 1s, so they each will become 0.333
# and so on.
weight=weight.masked_fill(tril_tensor==0, -torch.inf)
# now calculate the probs for each row, treating all cols as probs so their sum is 1
weight = weight.softmax(dim=-1)
bow_results2 = weight@x
print(f'{torch.all(bow_results==bow_results2)}')
# so you might ask, why would we want to make things more complicated like this to only
# achieve what we already achieved pretyy efficiently before?
# the answer is, this approach provides us with a flexibility that the previous ones
# wouldnt provide us withs. if you think about it for a moment, you'll notice that
# the 'weight' matrix can essentially be anything and not just zeros! 
# we have decoupled it from tril, which's job is to set the right half of the tensor
# to zero, so we actually get the expected behavior later on.(which is setting a constrain really (more layer on))
# if you havent yet figured it out, the 'wieght' matrix, can be truly a weight matrix
# which can show different strengths for each token, basically it can learn the interations
# between tokens and manifest them. currently it is us who sets it to all zeros, so we get
# uniform probablities, which inturn manifest itself as an average. but what if instead of 
# all zeros, we actually learn the values from the data itself? that would make sense, and 
# make it so that each token, can have a different connection(strength) to any other tokens
# now, and thus build semantic/meaningful relation. this is inafct what we are after. we want
# for tokens to learn associations and relations with other tokens based on the data present
# in the dataset, having uniform weight like what we initially did really is a far cry from
# what we intend, and therefore, we opt in to use this new approach that allows us to actually
# exploit this new capability. 
# This is infact the problem that attention solves, that is gathering
# information from the past but in a data driven manner.(side note, we are not limited to 'past' 
# information only per say, in this example, this is the case however, we will explain this 
# in more detail)), we will see how attention does this exactly in a moment. 
#
# but before we jump into attention implementation also note that the tril part, infact is a hard-constrain here that prevents tokens from
# the past from interacting with the tokens from the future. 
# so to recap here, basically the idea is, this triangular form, allows us to have 
# weighted aggregations of past elements. each element in the lower triangular part, 
# specifies, the degree by which it plays a rule in the said outcome. 
# that is how much of each element gets to fuse into this specific position(i.e. current token's)
# so now lets incorporate attention into our model
# Attention does its job by using two vectors called, key and query.
# basically every single token, emits two vectors called key and query, the query verctor
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
# specifies the size of the key/query output usually 16 is used a lot for head size so we 
# use that as well
head_size = 16
# lets not forget to set bias=False, so what it does is exactly dotproduct 
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
 #! check transpose result is it okto use 2,1 or -2,-1 or 1,2
weight_raw = q@k.transpose(2,1)
# now lets for a moment think about what is happening here, the key and query are applied
# on the input and each return an output of (B,T,head_size), they are in fact, processing
# all the tokens in the input, individually, simultaneously, all the same time. so each 
# token is both a query, and a key, and when we do a dotproduct, we are basically telling 
# it to reveal the relation/similarity/relevance of every token with every other tokens.
# and as we explained earlier, this is infact our weight matrix (which was initially zeros)
# but is now learned from the data!
# print(f'weight_raw\n{weight_raw}')
# now we can apply constrain on it so that tokens can only communicate with the past so 
# we use the tril trick now!
tril_constrain = torch.tril(torch.ones(size=(T,T)))
raw_wieghts_masked = weight_raw.masked_fill(tril_constrain==0, float('-inf'))
# apply sotmax to get probablity for each token
weight = raw_wieghts_masked.softmax(dim=2) # remember weight is (B,T,T)
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
# take the last token, which 8.6020e-02, notice this token, not only knows its content and own position in 
# the sequence(its the 8th token after all) but also knows which tokens comes before it and how much its 
# related to any of them.(basically when it knows which token comes before it, it creates its own query so
# to speak, and talks to every single previous token to find about which ones are more or less relavent to
# it and to what extend.) 
# For example,lets say, (since we are dealing with character level text generation!), the last token is a 
# vowel and says hey im a vowel and im looking for everyone else thats a vowel! and uses query to talk to 
# other tokens(keys). and intrestingly another token (lets say number 4) says im a vowel as well, and the 
# response is thus generate a larger number.(this is not a good example, words in sentence would make more sense
# !give a better example!)
# in our example, For the last token, it seems the 5th and 3rd token are particularly intresting/relavent/important,
# followed by the 7th token.
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
# just like the key and query, we set its bias to False, so we only get a simple vector, and a dotproduct output
value = torch.nn.Linear(C,head_size, bias=False) # produces (B,T,head_size) just like the other two key,query vectors
x_processed = value(x)
# this is not yet final final, we still need to do one more thing (read on!)
bow_final = weight@x_processed
#
#!check also note that, as we previously once pointed out, attention is a communication mechanism between tokens, and 
# by default there is nothing in this mechanism that provides a notion of space/position for tokens involved, i.e.
# by default these tokens/nodes/points, dont have any idea about where they are or how they are positioned (with resepect
# to others) etc, so we need to encode this information as well.(to better visualizing it, consider each token as 
# a node in a directed graph, each node has a connection to another node, (for example each node has a connection to
# itself, and another connection to other nodes,etc) and as you can see there is no notion of space here,the attention
# simply acts on a set of vectors in this graph, and thats why we need to encode them positionally as well, so they 
# have information about their position with regards to other tokens. 
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
#
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
class Head(nn.Module):
    def __init__(self, vocab_size, embed_size) -> None:
        super().__init__()


# so to recap, we first calculated the relavancy between all tokens against eachother, then constrained them so that 
# each token can only use the information from/interact with its past tokens. then since we needed probablity distribution so
# we then normalized it and then used that to pickout which tokens information(in the past) to aggregate/use with the inputs to 
# achieve our goal.(which is to predict the next character based on everything seen so far!)
# we dont want to aggregate inputs value (with respect to the weight matrix), we instead would like to use their representation
# so we use a new layer to do this, its called value, and we instead use its output instead of inputs raw value.
#
# 

#%%

class BigramModelWithAttention(nn.Module):
    def __init__(self, vocab_size, embd_size) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embd_size = embd_size
        # unlike the previous model, lets decouple the final logits from 
        # the number of embeddings, because we are using attentions, and
        # we want to have multiple operations inbetween obviously.
        self.embeddings = torch.nn.Embedding(vocab_size, embd_size)
        # in order to get the final logits, we need a linea layer at end
        self.fc = torch.nn.Linear(embd_size, vocab_size)
        
    def __call__(self, inputs:torch.Tensor, labels=None) -> torch.Tensor:
        out = self.embeddings(inputs) # has the shape (B,T,E) e is embd_size
        logits = self.fc(out)         # has the shape (B,T,C) c is vocabsize
        loss = None
        if labels is not None:
            # recall that crossentropy likes its input to be B,C,T and we are B,T,C
            # so lets permute and make it happy!
            loss = F.cross_entropy(logits.permute(0,2,1), labels)
        return logits, loss
    
    def generate(self, idxs, max_token_count)-> list[torch.Tensor]:
        # lets generate an output as long as num_max_token
        for i in range(max_token_count):
            # make sure idx is 2d
            assert len(idxs) >1, f"idx.shape '({tuple(idxs.shape)})' is invalid. it must have the form (B,T)"
            # now lets feed it to the model and sample from the probablities it produces
            preds,_ = self(idxs)
            # convert to probs 
            probs = preds.softmax(dim=1)
            # since we are bigram still, lets only get the last token as the next token predicted!
            probs = probs[:,-1,:]
            # now lets sample from it 
            new_idx = torch.multinomial(probs, num_samples=1, replacement=True)
            # now concatenate the new token to the previous one and feed it back to the model
            # for the next round of prediction
            # also remember that we are creating a sequence, so we concat them at dim=1 to get 
            # a longer sequence (we are gradually increasing the sequence length from 1 up to
            # max_token_count)
            idxs = torch.cat((idxs,new_idx), dim=1)
            
        return idxs

# this works fine, however, we can do better. here we just encoded the tokens, but
# we can also encode their position as well. lets do this as well
class BigramModelWithAttention(nn.Module):
    def __init__(self, vocab_size, embd_size,device) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embd_size = embd_size
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
        
        # in order to get the final logits, we need a linea layer at end
        self.fc = torch.nn.Linear(embd_size, vocab_size)
        
    def __call__(self, inputs:torch.Tensor, labels=None) -> torch.Tensor:
        # lets grab the shapes, since we will be using them 
        B,T,_ = inputs.shape
        token_embeddings = self.embeddings(inputs) # has the shape (B,T,E) e is embd_size
        # since we have a position_embedding lets use that as well
        # and notice that we didnt use the inputs, but rather torch.arange(T)
        # this means, for each input, as we process it, we also get embeddings up to
        # the current token count as well, if the inputs has 3 tokens currently, we
        # will creeate position embeddings for 0,1 and 2, and for the next input
        # this continues likewise. this results in (T,E)
        position_embeddings = self.position_embd(torch.arange(T,device=self.device))
        
        # and lets add the two embeddings together
        # this effectively gives us, not only the token embeddings(identity)
        # but also its position in the sequence. note that this doesnt really
        # help in a bigram model, but when it comes to attention it really does!
        embeddings = token_embeddings + position_embeddings
        
        # lets feed this embedding to our fc at the end instead
        logits = self.fc(embeddings)         # has the shape (B,T,C) c is vocabsize
        loss = None
        if labels is not None:
            # recall that crossentropy likes its input to be B,C,T and we are B,T,C
            # so lets permute and make it happy!
            loss = F.cross_entropy(logits.permute(0,2,1), labels)
        return logits, loss
    
    def generate(self, idxs, max_token_count)-> list[torch.Tensor]:
        # lets generate an output as long as num_max_token
        for i in range(max_token_count):
            # make sure idx is 2d
            assert len(idxs) >1, f"idx.shape '({tuple(idxs.shape)})' is invalid. it must have the form (B,T)"
            # now lets feed it to the model and sample from the probablities it produces
            preds,_ = self(idxs)
            # convert to probs 
            probs = preds.softmax(dim=1)
            # since we are bigram still, lets only get the last token as the next token predicted!
            probs = probs[:,-1,:]
            # now lets sample from it 
            new_idx = torch.multinomial(probs, num_samples=1, replacement=True)
            # now concatenate the new token to the previous one and feed it back to the model
            # for the next round of prediction
            # also remember that we are creating a sequence, so we concat them at dim=1 to get 
            # a longer sequence (we are gradually increasing the sequence length from 1 up to
            # max_token_count)
            idxs = torch.cat((idxs,new_idx), dim=1)
            
        return idxs
    