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
# now lets add attention mechanism to our base model. 
# attention mechanism at its core tries to take advantage of the rich information embedded
# in the sequence. for our work, this is specifically about the past history but in general 
# attention can utilize both past and future connections/sequence tokens. what we described 
# here just now is not exactly accurate, but gives us a foundation to build our intuition as
# we continue on. we will elaborate more of course and hopefully get it all.
# before we venture any further into the crux of the matter, let us learn about a technique
# thats used to efficiently implement attention mechanism.
# for this purpose,lets imagine we have a simple input like the following: 
 