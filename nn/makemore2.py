#%%
# in the name of God the most compassionate the most merciful
# !talk about pitfalls of log and exp 
# log give -inf when 0 is met
# exp fails when numbers in a sequence are not in the same range and theres big difference between them
# exp(-5) is ok, exp(-3,2,0,-1) is ok but exp(-3,-1.2, 2, 100) creates inf! 
# now what is usually done to avoid this, is people do exp(iterable-max(iterable))
#! Explain THIS IN DETAIl 
# so normalizing input when the range of all numbers is kept the same is one of the reasons we do normalization
# (this is not all , just wanted to say it as a reminder for their relationship, and have intuitve undrestanding with
# previous knowledge )
# in this section we are going to take another approach that is more sensible and practical. 
# we are going to implement a 2003 paper by bengio etal named A neural probablistic language model 
# for several reasons that Im about to explain in a moment. 
# basically the idea of this paper is that, knowing using contexts size larger 
# than 4 or 5 in an ngram model can impose a huge overhead (and we are talking about words by the way)
# to give you an idea, our character level bigram model had a context of 1 character, 


# lets read the words, create an embedding this time. 
# the input recieves 3 input character, and perdicts the 4th
# we will create a network that we feed 3 inputs, these inputs
# will be converted into an embedding, where each character is
# represented by a vector of length x (2,3 or whatever), then 
# that embedding output is fed to another layer (tanh e.g.)
# and finally the output is fed to the last layer which outputs
# a vector of 27 length, providing the probablities for each 
# character as being the next one!
import torch 
import matplotlib.pyplot as plt 
# %matplotlib inline # acually this line doesnt work inside vscode ipython 
# it only works on jupyter notebook and infact its not even needed there 
# anymore to have figures/plots inlined! but im just writing it here for
#muscle memory of mine!!!

names = open('./names.txt').read().splitlines()
# lets extract the alphabet and create atoi and itoa mappings
atoi={'.':0}
atoi.update({ch:i for i,ch in enumerate(sorted(set(''.join(names))),start=1)})

itoa = {v:k for k,v in atoi.items()}
# lets test 
print(f'{atoi=}')
print(f'{itoa=}')
# now we need to create dataset of inputs with 3 characters as input
# and the next one be the label
context = ''
X=[]
Y=[]
block_size=3
for name in names[:5]:
    context = [0]*block_size
    for ch in name+'.':
        idx = atoi[ch]
        X.append(context)
        Y.append(idx)
        context = context[1:]+[idx]
        print(context)
        
X=torch.tensor(X)
Y=torch.tensor(Y)

print(f'{X.shape}')
print(f'{Y.shape}')
#  now that we have our data, let us create our embedding layer.
# embedding layer is simply an array of some vectors, basically
# each word/character is given a vector of length x, e.g. 2,10,30,etc
# depending on the dataset. so if we have 5 characters, we will have 5 vectors
# the idea here is to represent more information in these 'emebedings'
# each character can now be represented better and more freely
# lets create our embedding matrix with initial random values
# lets choose embeding size of 2 , each character is represented by 2 floats!
C=torch.randn(size=(X.shape[0],2))
print(f'{C.shape=}')
# now how can we extract a specific emebding related to a specific input you say?
idx0 = X[20][0]
idx1 = X[20][1]
idx2 = X[20][2]
print(f'\nX[20]: {X[20]}')
print(f'C[{idx0}]{C[idx0]}')
print(f'C[{idx1}]{C[idx1]}')
print(f'C[{idx2}]{C[idx2]}')
# so each input contains 3 characters, so it contains 3 indexes 
# instead of separately fetching each inputs embd we can do it in one go!
print(f'\nC[X[20]]{C[X[20]]}')
#python fetches each value from X[20] and send it to C and get a result, 
# then do it for the next and the next until all the results are gathered
# then returns the result. basically pytorch internally does a forloop for
# us !
# likewise, we can do this with all values of X and not just a single index!
print(f'C[X].shape: {C[X].shape}\nC[X][1].shape: {C[X][:1].shape}\nC[X][1]:{C[X][:1]}')
# gives us the embeddings for all inputs (basically each sample has 3 embeddings for
# each character triplet) in total our result shape would be 32(number of samples) 
# x3(number of inputs which is 3 characters(make up a single sample) x 2(embedingsize)
# )
# now 