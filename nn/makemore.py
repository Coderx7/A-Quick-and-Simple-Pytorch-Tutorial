#%% in the name of God the most compassionate the most merciful
from itertools import pairwise,tee

alphabet = {chr(ord('a')+i):i for i in range(26)}
name = 'alice'
for ch1,ch2 in pairwise(name):
    print(ch1, ch2)

print(f'-----')

ch1,ch2 = tee(name)
next(ch2)
for ch1, ch2 in zip(ch1,ch2):
    print(ch1,ch2)
print(f'-----')

# is the equivalent of doing 
for ch1, ch2 in zip(name,name[1:]):
    print(ch1,ch2)

# negative loglikelihood ro tozih bede chie va chera esteafde mishe 
# product o probablities - > we  use log and sum them -> why? becasue stability (small float numbers
# multiplication!!!) 
# since the worse numbers give negative numbers, and the best results in 0 if we add them all
# it gives us a big large negative number if all or most of the entries are wrong/negative
# so we use - on it (multiply it by -1) so it becomes a positive number, now if there are more
# wrong /bad entries, a larger number is produced, and the better the entries, the lower the number
# so we can now use this number as loss! 
# also instead of simply adding them all, we can simply average them, so the loss becomes 
# more managebale and smaller! thats what we call negative log likelihood or nll for short!)
# 
# we create a bigram model first and then try to create an mlp doing the same thing and learn 
# many concepts and stuff along the way and get a good intutin concerning them inshaallah

words=open('./names.txt', 'r').read().splitlines()
words[:5]

#%% in the name of God the most compassionate the most merciul 
# Whats the deal here? 
# we are going to learn about LLMs and how they work, in the meantime, we are going 
# to also learn new concepts and refresh our memories on previous concepts we may have alreadyknow
# we start off with an ngram model, a bigram to be more speciic,  to create human name sounding 
# outputs, i.e. it prints new names for us. 
# we then proceed to make it better and then create a neural network version of it. 
# we then go on improving that and create rnns, grus, and ultimately finish off with a transformer
# inshaallah. 
# %%
# bigram model 
# A bigram model, also known as a 2-gram model, is a statistical language model that predicts 
# the probability of a word based on the previous word or context. It is a type of n-gram model 
# where "n" represents the number of words taken into consideration for prediction.
# In a bigram model, the prediction of the current word relies solely on the immediately preceding word.
# It assumes that the probability of a word depends only on the word that directly precedes it in a 
# sequence of words.
# For example, given the sentence "I love to eat ice cream," a bigram model would consider pairs 
# of consecutive words:
# 1. "I love"
# 2. "love to"
# 3. "to eat"
# 4. "eat ice"
# 5. "ice cream"
# The bigram model estimates the conditional probability of the current word given the previous word. 
# For instance, to predict the word following "I love," the bigram model calculates the probability of 
# different words based on the occurrence of those words after "I love" in a training corpus.
# Bigram models are relatively simple compared to higher-order n-gram models, but they capture 
# some level of local context and can be used for tasks like language generation, text completion, 
# and speech recognition.

# in our case, we are going to build a character level bigram model, not a word level one!
# we are going to use a name dataset that holds populare names in the united states of america
# lets see a few of these names
names = open('./names.txt').read().splitlines()
print(f'{len(names)=:,}')
print(f'shortest name has a length of: {min(len(name) for name in names)}')
print(f'longest  name has a length of: {max(len(name) for name in names)}')
print('afew names from the dataset:')
print(*names[:10], sep='\n')

# at the first glance we might think theres not much info in this dataset, but if we look closely
# we notice that infact theres quite a bit of information
# like for example, looking at the characters starting the names, and the ones ending it
# can give us a sense of what characters are more likely to come at the start and what at the end
# we can also go on with the logic and see, given a character a for example, whats the most likely
# character to come after ward? and or which character is most likely to aprear given the two previous
# characters, so on. 
# using a simply 2-gram or bigram model, we are able to capture some of these hidden relationships/structures
# and actually are able to generate somewhat intresting name suggestion. 
# lets create a bigram model then
# lets create a dictionary holding the bigrams and their number of occurances 
# (since we said the bigram model calculates the probability of 
# different words based on the occurrence of those words, its the same here, we are just using characters
# # instead of words )
# for creating pairs we can use several methods in python, 
# the simplest way is to use itertools.pairwise, we could also use zip which I demosntrate in a momemnt
#%%
import itertools
btable = {}
# for start lets see how it does with 1 name
for name in names[:1]:
    for ch1, ch2 in itertools.pairwise(name):
        # # since the pair may not exist in the begining we use get() in a dictionary
        btable[(ch1,ch2)] = btable.get((ch1,ch2),0) +1
        print(ch1,ch2)

print()
# we could do this with zip as well 
for name in names[:1]:
    for ch1, ch2 in zip(name, name[1:]):
        print(ch1,ch2)
print()    
# now before we go on, theres a technique in language modeling where people add a special character
# denoting the start of a new sequence and another one denoting the end of the sequence. 
# if you look closely at our names, an implicit information is that, obviously, each name ends after a few characers
# so how should we model this ? this is one way of modeling this information. so lets add this to our
# names. 
# lets add <s> as the starting symbol and <e> as the ending symbol
# but theres a problem our symbol itself is 3 characters! it will mess up our name and statistics right!
# we can either use a one-character symbol thats not used in our dataset (is not a part of any name)
# or we can treat all of these as standalone list entries! that is instead of a str we are going to 
# have a list of values, this way, <s> and <e> can be treated like a single elemented!
for name in names[:1]:
    name = ['<s>']+list(name)+['<e>']
    for ch1, ch2 in itertools.pairwise(name):
        # # since the pair may not exist in the begining we use get() in a dictionary
        btable[(ch1,ch2)] = btable.get((ch1,ch2),0) +1
        print(ch1,ch2)

# ok now lets do this for the whole dataset:
btable = {}
for name in names:
    name = ['<s>']+list(name)+['<e>']
    for ch1, ch2 in itertools.pairwise(name):
        # # since the pair may not exist in the begining we use get() in a dictionary
        btable[(ch1,ch2)] = btable.get((ch1,ch2),0) +1
        # print(ch1,ch2)
        
print(btable)
# lets see the pairs with highest frequency/probablity
sorted(btable.items(), key=lambda kv: kv[1] ,reverse=True)
# side tip: we can use operator.itemgetter to do this as well
from operator import itemgetter
sorted(btable.items(), key=itemgetter(1), reverse=True)

# but this takes a lot of memory! we can make it better by using numbers instead of characters
#  lets use a 2d array to do this instead. since we wantto use array, we cant use characters anymore
# so we need to somehow use numbers for each characters
# so we create a dictionary that maps our characters in our datasets to numbers 
# one way would be to just include the alphabet characters 
atoi={chr(ord('a')+i):i for i in range(26)}
# but the problem is, there might be some other characters in names that we didnt acount for (like unicode, ., etc)
# a better way would be to grab all the data we have and get unique entries from it , we can do that
# using set()s in python which dont allow duplicate values (we use sorted so they are sorted for visualization later on!)
characters = sorted(set(ch for name in names for ch in name))
atoi={ch:i for i,ch in enumerate(characters)}
# or better we could also do sth like this which creates a long string out of all list entries
# which is then fed to set(), and set removes all the duplicate characters, keeping only one occurance of each character
atoi={ch:i for i,ch in enumerate(sorted(set(''.join(names))))}
print(f'{atoi=}')
# so now we can easily get a number corrosponiding to any character
print(atoi['a'])
# we also need to add our special characters as well, cuz recall that they are not part of our data
atoi['<s>']=26
atoi['<e>']=27
# now lets create reverse lookup table as well, so that given a number we fetch its corrosponding character!
itoa={v:k for k,v in atoi.items()}
print(atoi['<e>'], itoa[atoi['<e>']])
# now we are good to go, lets create our 2darray now
import torch
barray = torch.zeros(size=(28,28), dtype=torch.int32)
# lets fill this up
for name in names:
    name = ['<s>']+list(name)+['<e>']
    for ch1,ch2 in zip(name, name[1:]):
        idx1,idx2 = atoi[ch1], atoi[ch2]
        barray[idx1,idx2] +=1
# now lets visualize this 
import matplotlib.pyplot as plt
%matplotlib inline

plt.imshow(barray)
# this is not really that helpful lets make it a bit more readable
fig = plt.figure(figsize=(16,16))
# 
# here we set the cmap for better visualization, 
# Basically cmap='Blues' specifies the colormap to be used for mapping the values in the barray 
# to colors in the plot created by imshow() function.
# A colormap (or cmap for short) is a set of colors used to represent different values in a plot. 
# It maps the numerical values of the data to colors, allowing for visual differentiation and interpretation.
# The Blues colormap consists of shades of blue, ranging from lighter shades for smaller values to
# darker shades for larger values. This colormap is commonly used for visualizing data that represents
# gradients or magnitudes.
# By using cmap='Blues', the imshow() function will assign colors from the Blues colormap to the values
# in barray when generating the plot. This allows for easy visualization of the data, with lighter shades 
# of blue representing lower values and darker shades representing higher values.
# There are other colormaps to use as wel, such as 'Reds, Greys, Greens, and much more.
# if we dont specify any values for cmap, the default is 'viridis' which starts from a 
# deep blue color for low values to a vibrant yellow color for high values. 
# Blues make it much easier to see so we go with Blues (test viridis and see for yourself)
plt.imshow(barray, cmap='Blues')
for i in range(28):
    for j in range(28):
        chstr = itoa[i]+itoa[j]
        # here we swap the i,j so each row contains all the characters staring from each character
        # that is the first row starts with a, and has all combs of a+ its next character
        # the second row starts with b, and shows the frequency of all characters after b
        # the third row starts with c, and shows the frequency of all characters after c
        # and so on
        plt.text(j,i, chstr, ha='center', va='bottom', color='gray')
        plt.text(j,i, barray[i,j].item(), ha='center', va='top', color='gray')
plt.axis('off')

# now looking at this new image, we can see that each row shows the frequency of each character after
# it, for example in the first row we see pairs of a- from aa up to az, the second row shows the same 
# information for character b and so on. 
# so we can see that an, a<e> and n<e> are among the most frequent pairs.
# another obseravation that we can have here, is that the row for <e> symbol and <s> 
# are all zeros, wasting precious memory and yet giving nothing useful in particular
# like theres no way our ending symbol is going to appear at the start of a name!
# likewise its impossible for <s> to showup at the end of name! because by definition
# we only place them at the start and end of a name to denote its begining and its end respectively
# so its a given that such impossible occurance to be out of question. 
# so how can we fix this and make it better? 
# why not use a single symbol to denote the start and end of a string/name? 
# lets just do that, instead of <s> and <e> lets use use . for example!
# lets do this
# lets create the barray anew! cuz now we have 1 character less!
# the dtype matters cuz we want to count the occurances of each pair, so int is used!
barray = torch.zeros(size=(27,27), dtype=torch.int32)
# before we go on, we also need to update our itoa and atoi respectively to reflect the changes
# lets put our special character at the begining i.e. assigne 0 to our special character
# so for this, lets create our dictionary like normal, but make enumerate start from 1 instead of 0
atoi = {ch:i for i,ch in enumerate(sorted(set(''.join(names))),1)}
# now we add our special character at 0
atoi['.']=0
# now lets create our itoa
itoa = {v:k for k,v in atoi.items()}

for name in names:
    # now we can simply concat them cuz they are all chars! we can still use the previous method 
    # of ['.']+list(name)+['.'] but its not needed obviously!
    name = '.'+name+'.'
    for ch1, ch2 in zip(name, name[1:]):
        idx1, idx2 = atoi[ch1], atoi[ch2]
        barray[idx1,idx2] +=1 
        
# now lets visualize this again 
plt.figure(figsize=(16,16))
# lets display our array
plt.imshow(barray, cmap='Blues')
# now lest edit how it looks 
for i in range(27):
    for j in range(27):
        chstr = itoa[i]+itoa[j]
        plt.text(j,i,chstr, ha='center', va='bottom', color='gray')
        plt.text(j,i, barray[i,j].item(), ha='center', va='top', color='gray')
plt.axis('off')
# now looking at the new image, we can see that the empty row/cols are gone and '..' count is 0 
# meaning we dont have empty names in our dataset!
# now we have a table that shows the frequency of all pairs we should be able to use this information
# to make new names! 
# we have all the characters that start a name (if you look at the first row, thats all the characters
# that start a name)
# now if we were to make a new name, based on these information, how would we go about it? 
# well, for the start, we would want to see what character is the most likely one to come at first
# and we would look at its count, that would tell us which one is more likely compared to others
# then what character would be the most likely one to come after this? and this goes on like this
# the first row shows us all the starting characters and their frequencies, and the rest of the rows
# belong to each character and shows, whats the likelihood of a specific character to come after it
# now lets make a new name. we go to the first row , but which one to take? we cant possibly always
# choose the pair with the highest number, as it would always result the same. if we do it randomly
# then we disregarded every connection or semantic we discovered so far, so what do we do!
# the solution is to sample from a distrbution that contains this information. 
# we said that the first row, contains "ALL" the characters that start a name, so their counts
# summed together should give us the overal count and each characters count divided by that total
# should give us the probablity of that character being chosen for the first character of a name.
# so basically in order to get the probablities , we just need to sum all the counts in the first row
# and divide each character count by the sum to get their probablity. if we have the probablity 
# we can sample from it !
# if you look closely you can see that, this actually applies to all other rows as well, each row
# contains a single character like a, followed by all other characters with the number of occurances
# basically giving us the likelihood of them occuring. 
# so lets calculate the probablites for each row 
# note we have to convert it to float() cuz we want probablities! 
# and so we are basically dealing with floats!
first_row = barray[0].float()
p=first_row/first_row.sum()
print(f'{p=}')
# which gives us 
# p=tensor([0.0000, 0.1377, 0.0408, 0.0481, 0.0528, 0.0478, 0.0130, 0.0209, 0.0273,
#           0.0184, 0.0756, 0.0925, 0.0491, 0.0792, 0.0358, 0.0123, 0.0161, 0.0029,
#           0.0512, 0.0642, 0.0408, 0.0024, 0.0117, 0.0096, 0.0042, 0.0167, 0.0290])
# 
# which is inline with our first row counts
# now we can do this in one go and calculate the probablity distribution for the whole barray! 
# calculate all rows all atonce!)
probs = barray.float()/barray.sum(dim=1, keepdim=True)
print(f'{probs[:1]=}')

# now lets explain whats happening here, 
# since we want to sum all the rows columnwise (that is we want all columns from left to righ to
# be summed into a single cell, so we tell sum to sum over the dim=1, dim=0 is the rows)
# keepdim=True is essential here as its required for a valid broadcast 
bsum = barray.sum(dim=1, keepdim=True)
print(f'{bsum.shape=}')
# prints bsum.shape=torch.Size([27, 1])
# now 27,27 / 27,1 means to replicate single cell 27 times along the columns 
# basically creating a 27x27 matrix each row will have 27 columns with the same value
# which when divided gives us the correct value
# however if we remove that trailing 1 (which keepdim=False) does, we would endup with a shape
# of [27]
# when we go for a division of a 27,27 matrix by 27 vector! the broadcasting does this 
# it converts it to 27,27 divided by 1,27, basically creates 27 rows of 1 with one column of 
# the actual values 
probs_wrong = barray.float()/barray.sum(dim=1,keepdim=False)
print(f'{probs_wrong[:1]=}')
# which prints 
# probs_wrong[:1]=
# tensor([[0.0000, 0.1301, 0.4938, 0.4366, 0.3075, 0.0750, 0.4608, 0.3472, 0.1148,
#          0.0334, 0.8352, 0.5879, 0.1126, 0.3821, 0.0625, 0.0497, 0.5019, 0.3382,
#          0.1291, 0.2535, 0.2348, 0.0249, 0.1461, 0.3305, 0.1923, 0.0547, 0.3874]])
#
#
#for reference this is the values of the first row : 
# first_row=tensor([   0., 4410., 1306., 1542., 1690., 1531.,  417.,  669.,  874.,  591.,
#                   2422., 2963., 1572., 2538., 1146.,  394.,  515.,   92., 1639., 2055.,
#                   1308.,   78.,  376.,  307.,  134.,  535.,  929.])
# tensor(32033.) is the sum for the first row and all values in the first row must be dividied by this number
# and if we look at the correct value for the second entry, its 0.1377 which is 4410/32033 =0.1377
# however, if we look at the result of barray.sum(dim=1,keepdim=False), we basically get this:
# tensor(32033)
# tensor(33885)
# tensor(2645)
# tensor(3532)
# tensor(5496)
# tensor(20423)
# tensor(905)
# tensor(1927)
# tensor(7616)
# tensor(17701)
# tensor(2900)
# tensor(5040)
# tensor(13958)
# tensor(6642)
# tensor(18327)
# tensor(7934)
# tensor(1026)
# tensor(272)
# tensor(12700)
# tensor(8106)
# tensor(5570)
# tensor(3135)
# tensor(2573)
# tensor(929)
# tensor(697)
# tensor(9776)
# tensor(2398)
# this is [27] numbers, basically when we divide 27,27 from 27 
# the broadcaster needs to broadcast our vector into a compatible shape of 27,27 so it can carry on
# the elementwise division. in doing so it replicates the elements how? it takes the first entry
# and copies it along the first dimension in all rows so you'll have sth like this 
# tensor(32033)
# tensor(32033)
# tensor(32033)
# tensor(32033)
# ....
# next it goes to the second value in the row and creates a new column out of that 
# tensor(32033) tensor(33885)
# tensor(32033) tensor(33885)
# tensor(32033) tensor(33885)
# tensor(32033) tensor(33885)
# ....          ....
# and so on. as you can see, this completely messes up everything 
# and now the probablity is 1301, which comes from 4410/33885=0.1301 which is wrong!
# we should instead be having sth like this:  
# tensor(32033) tensor(32033) tensor(32033) tensor(32033) ...
# tensor(33885) tensor(33885) tensor(33885) tensor(33885) ...
# tensor(2645)  tensor(2645)  tensor(2645)  tensor(2645)  ...
# ...
# where for the first row, all columns are the same values becasue they all belong to that row only
# so to recap, when doing division here we want to retain the shape of resuling sum, so 
# the broadcast can replicate each value along that dimension, for 27,1 
# torch comes and takes the first item and replicates it to the right to fill all the columns of that row
# for the second (row)value, it reads 33885 and replicates it along the column axis so this row only has 33885
# and repeats this for all the rows which is what we actually want. 
# if we dont do keepdim=True, it will simply create a column vector [27] which when torch tries to
# broadcast, it has to comeup with the first dimension,(which is missing now (its needs to be sthx27 and sth is 27 )
# however, it cant let it have random values, so what it does is it tries to fill the missing values
# what does is, for every column, it creates a row of the first column, whatever value is in the first column
# it replicates it from top to the bottom of that column, creating a row of that number, then it goes
# to the next column and repeats this and so on. as you can see this is not what we want, 
# now that we sorted all of this information and got ourselves our probs 
# we can go and sample from it. for sampling we use torch.multinomial function 
s = torch.multinomial(probs[0], num_samples=1)
print(f'samples :{s}')
# now lets do better and explain a bit of stuff. 
# before going any further lets have a fixed seed so the results all comeout the same
# each time we run this, for this we create a a random generator in torch and set it
g = torch.Generator('cpu').manual_seed(255)
# and now we feed this to our ops 
sample = torch.multinomial(probs[0], num_samples=1, replacement=True, generator=g).item()
print(f'{sample=},{itoa[sample]}')
# now each time we run this we get the same result, becasue we made it determinstic!
# multinomial accepts an input, a probablity distribution, which samples from us
# the num_samples, specifies the number of samples we want it to generate based off of the probablity
# distribution, if we give it a vector, it gives us the number of samples we asked for, if we give it
# a matrix, it will give us the samples for each row. thats why I wrote probs[0] for the test so it
# only samples from the first row only. (remember we were trying to sample from the first row and 
# then follow that up from other rows until we get our name)
# the replacement, argument simply says, if you take a sample, should we replinsh/replace/restock! it again or not
# (basically imagine a vending machine, if you took a coke, if replacement=True, it will immediately
# replace the empty space with a new coke!, so next time you can get it again!)
# and finally generator is used to create determinstic behavior. 
# so now we have everything in place how are we going to create a new name? 
# 1.start from the first row and sample sth, 
# 2.get that, get what character its, and go for its row, sample from it and continue until . is met
# start from first row
idx = 0
chstr=''
while True:
    idx = torch.multinomial(probs[idx], num_samples=1, replacement=True, generator=g).item()
    #convert it to character
    chstr+=itoa[idx]
    if idx==0:
        break
print(chstr)
# now if we want to create more names we can iterate this more 
for i in range (20):
    chstr=''
    # always start from first row
    idx=0
    while True:
        prob = probs[idx]
        print(f'{prob=}')
        idx = torch.multinomial(probs[idx], num_samples=1, replacement=True,generator=g).item()
        chstr += itoa[idx]
        if idx==0:
            print(chstr)
            break
# now they dont particularly look very good! but our model did learn sth and in order to see
# if they are really random or not, we can change the probablities learned by our model with 
# a uniform probablity that makes the probablity for all characters the same! see how that does
new_prob = torch.ones(27)/27.0
print(f'{new_prob=}')
for i in range (10):
    chstr=''
    # always start from first row
    idx=0
    while True:
        new_prob = torch.ones(27)/27.0
        idx = torch.multinomial(new_prob, num_samples=1, replacement=True,generator=g).item()
        chstr += itoa[idx]
        if idx==0:
            print(chstr)
            break
# so we actually did learn something and thats why each character has a different probablity of showing up
#  and it shows in our results 
# but how can we quantify how good of a model we have here? 
# to that end, we usually use a loss function! so how do we get a loss function? or how do we create one?
# Lets talk about 
# probablities, likelihood, log-likeliehood, negative-log-likelihood!
# we so far we managed to train a model that learned some probablities concerning the dataset
# how likely certain characters are to appear in a name given different creteria. 
# we actually saw first hand that, in fact these probablities (probablity distribution) is 
# what makes us different from a completely random selection. we saw that in a uniform probablity
# the model just spits nonsense! so accurate probablity distribution is important to us and comes
# handy! 
# for this very reason, we have something called, likelihood, which is simply the product of all probablities
# yes likelihood is the product of all probabalities in our model. 
# in other words, likelihood refers to the probability of the observed data given a specific model or parameter values. 
# basically it tells us how the model assigned probablities to parameters with respect to obsertved data in the dataset
# therefore the higher the product of these probablities, the higher the likelihood of the model representing
# the actual behavior of the underlying data. 
# to put it differently, The likelihood of a model represents how well it assigns probabilities to its parameters 
# given the observed data in the dataset. The higher the product of these probabilities, the higher the likelihood 
# of the model accurately representing the underlying behavior of the data.
# In other words, the likelihood measures the goodness-of-fit of the model to the observed data. 
# A higher likelihood indicates that the model's parameters align well with the data, 
# suggesting that the model is a better representation of the true data-generating process.
# the problem is, multiplying probablities result in a tiny number especially when most probablities
# are small float numbers! 
# to fix this issue we instead use the log of the probablities and instead sum them all
# because log(a*b*c) = log(a)+log(b)+log(c). 
# (this is called logits!) this has some nice properties as well, not only it substitudes multiplication with addition
# which is faster(hardware implementation wise), it makes taking its gradients easier, and is much more numerically stable (we see why) 
# since our probability is between 0-1 ultimately, log(x) from 0 to 1 is 1 to negative infinity
# the closer the value get to zero, the smaller the result of log gets (log(0) is infinity!
# so unless our probablities are not exactly 0, we are fine (this why sometimes we add an eps to some 
# operations involving logs, for probablities, we may also do this, add the probs to sth like 1e-10 qne then take its log)
# so far so good lets run a simple example and observe what we just talked about 

chstr = ''
likelihood=1# likelihood calculated by the product of probs
log_likelihood = 0 # specifies how good the model is at assigning the right probablity to parameters according to the underlying dataset
i=0
for word in names[:3]:
    word = ['.']+list(word)+['.']
    for ch1,ch2 in zip(word,word[1:]):
        idx1 = atoi[ch1]
        idx2 = atoi[ch2]
        # lets see what probablity  our model has assigned to this pair
        prob = probs[idx1, idx2]
        log_prob = prob.log()
        likelihood *= prob.item()
        log_likelihood += log_prob.item()
        i+=1# a simple counter!
        print(f'{ch1}, {ch2}, {prob=:.4f} {log_prob=:.4f}')
print(f'{likelihood=}, {log_likelihood=:.4f}, log_likelihood(avg): {log_likelihood/i:.4f}')
# it results in 
# ., e, prob=0.0478 log_prob=-3.0408
# e, m, prob=0.0377 log_prob=-3.2793
# m, m, prob=0.0253 log_prob=-3.6772
# m, a, prob=0.3899 log_prob=-0.9418
# a, ., prob=0.1960 log_prob=-1.6299
# ., o, prob=0.0123 log_prob=-4.3982
# o, l, prob=0.0780 log_prob=-2.5508
# l, i, prob=0.1777 log_prob=-1.7278
# i, v, prob=0.0152 log_prob=-4.1867
# v, i, prob=0.3541 log_prob=-1.0383
# i, a, prob=0.1381 log_prob=-1.9796
# a, ., prob=0.1960 log_prob=-1.6299
# ., a, prob=0.1377 log_prob=-1.9829
# a, v, prob=0.0246 log_prob=-3.7045
# v, a, prob=0.2495 log_prob=-1.3882
# a, ., prob=0.1960 log_prob=-1.6299
# likelihood=1.4309086525157158e-17, log_likelihood=-38.7856, log_likelihood(avg): -2.4241
#
# we see that the one with higher probablity such as 0.3899 (around 40% prob) has close to zero log_prob(-0.9418)
# whereas the ones that have low probs such as (0.0478) their log_probs are much worse (-3.0408)
# and summing them all gives us -38.7856! (meanwhile look at our likelihhod! its 1.4e-17!!! its extremely tiny! we cant simply properly utilize that!)
# while this is great compared to likelihood (the product of probs) by leaps and bounds, you may see people
# actually normalize this number, basically instead of simply summing it all, they average it, thats whats usually done
# and hence we get to the number -2.4241. 
# our log-likelihood which is supposed to be showing how good of a repreesentation is, 
# has become negative! it means, if the model does absolutely great, and get everything 100%, it gets likelihood of 0!
# (sum of logs of (1) is 0) and if it gets it wrong, we would have a negative number. this is not desirable
# we would like to change it so that, if it does bad it increases in magnitude, and if it gets it right it gets a low number
# preferably 0 (when prefect). what do we do? we simply take its negative! 
# we simply multiply it by -1, and this makes it a negative log-likelihood or nll for short! 
# which is our loss. 
# so this models loss would be 2.4241!
# if we were to run this on the whole dataset we would get the same value 
# see 
chstr = ''
likelihood=1# likelihood calculated by the product of probs
log_likelihood = 0 # specifies how good the model is at assigning the right probablity to parameters according to the underlying dataset
i=0
for word in names:
    word = ['.']+list(word)+['.']
    for ch1,ch2 in zip(word,word[1:]):
        idx1 = atoi[ch1]
        idx2 = atoi[ch2]
        # lets see what probablity  our model has assigned to this pair
        prob = probs[idx1, idx2]
        log_prob = prob.log()
        likelihood *= prob.item()
        log_likelihood += log_prob.item()
        i+=1# a simple counter!
        print(f'{ch1}, {ch2}, {prob=:.4f} {log_prob=:.4f}')
print(f'{likelihood=}, {log_likelihood=:.4f}, log_likelihood(avg): {log_likelihood/i:.4f}')
# likelihood=0.0, log_likelihood=-559873.5909, log_likelihood(avg): -2.4540
# so the lower our loss (negative log liklihood loss), it means the better the model is (at giving more accurate probs to the data observed in the dataset)
# with this we can also for example, calculate the likelihood of certain words: 
# see this 

likelihood=1# likelihood calculated by the product of probs
log_likelihood = 0 # specifies how good the model is at assigning the right probablity to parameters according to the underlying dataset
i=0
for word in ["annon"]:
    print(f'{word=}')
    word = ['.']+list(word)+['.']
    for ch1,ch2 in zip(word,word[1:]):
        idx1 = atoi[ch1]
        idx2 = atoi[ch2]
        # lets see what probablity  our model has assigned to this pair
        prob = probs[idx1, idx2]
        log_prob = prob.log()
        likelihood *= prob.item()
        log_likelihood += log_prob.item()
        i+=1# a simple counter!
        print(f'{ch1}, {ch2}, {prob=:.4f} {log_prob=:.4f}')
print(f'{likelihood=}, {log_likelihood=:.4f}, log_likelihood(avg): {log_likelihood/i:.4f}')
# word=emma
# ., e, prob=0.0478 log_prob=-3.0408
# e, m, prob=0.0377 log_prob=-3.2793
# m, m, prob=0.0253 log_prob=-3.6772
# m, a, prob=0.3899 log_prob=-0.9418
# a, ., prob=0.1960 log_prob=-1.6299
# likelihood=3.478213834606263e-06, log_likelihood=-12.5690, log_likelihood(avg): -2.5138
# says this is not really likely, and looking at the data we see that .e em and mm are particularly
# unlikely to popup but annon is much more likely :
# word='annon'
# ., a, prob=0.1377 log_prob=-1.9829
# a, n, prob=0.1605 log_prob=-1.8296
# n, n, prob=0.1040 log_prob=-2.2634
# n, o, prob=0.0271 log_prob=-3.6096
# o, n, prob=0.3039 log_prob=-1.1911
# n, ., prob=0.3690 log_prob=-0.9969
# likelihood=6.9734472756582705e-06, log_likelihood=-11.8734, log_likelihood(avg): -1.9789
# but before we end this lets consider this example: 
likelihood=1# likelihood calculated by the product of probs
log_likelihood = 0 # specifies how good the model is at assigning the right probablity to parameters according to the underlying dataset
i=0
for word in ["anonvs"]:
    print(f'{word=}')
    word = ['.']+list(word)+['.']
    for ch1,ch2 in zip(word,word[1:]):
        idx1 = atoi[ch1]
        idx2 = atoi[ch2]
        # lets see what probablity  our model has assigned to this pair
        prob = probs[idx1, idx2]
        log_prob = prob.log()
        likelihood *= prob.item()
        log_likelihood += log_prob.item()
        i+=1# a simple counter!
        print(f'{ch1}, {ch2}, {prob=:.4f} {log_prob=:.4f}')
print(f'{likelihood=}, {log_likelihood=:.4f}, log_likelihood(avg): {log_likelihood/i:.4f}')
# prints
# word='anontivs'
# ., a, prob=0.1377 log_prob=-1.9829
# a, n, prob=0.1605 log_prob=-1.8296
# n, o, prob=0.0271 log_prob=-3.6096
# o, n, prob=0.3039 log_prob=-1.1911
# n, t, prob=0.0242 log_prob=-3.7226
# t, i, prob=0.0955 log_prob=-2.3485
# i, v, prob=0.0152 log_prob=-4.1867
# v, s, prob=0.0000 log_prob=-inf
# s, ., prob=0.1442 log_prob=-1.9365
# likelihood=0.0, log_likelihood=-inf, log_likelihood(avg): -inf
# as you can see, log_liklihood is -inf! this happens when log(0) is encountered
# as we previously pointed out briefly. this happened becasue the probabbily of vs
# was 0. cuz its number of occurances was 0 (theres not any sample in the dataset where in a word s comes after v (basically vs doesnt exist at all)
# thus resulting in -inf. 
# we also mentioned a solution to deal with these which was adding an eps to the probablities
# before taking the log. 
# lets do this infact!  
probs_smoothed = probs.add(1e-10)
# now lets try again 
likelihood=1# likelihood calculated by the product of probs
log_likelihood = 0 # specifies how good the model is at assigning the right probablity to parameters according to the underlying dataset
i=0
for word in ["anonvs"]:
    print(f'{word=}')
    word = ['.']+list(word)+['.']
    for ch1,ch2 in zip(word,word[1:]):
        idx1 = atoi[ch1]
        idx2 = atoi[ch2]
        # lets see what probablity  our model has assigned to this pair
        prob = probs_smoothed[idx1, idx2]
        log_prob = prob.log()
        likelihood *= prob.item()
        log_likelihood += log_prob.item()
        i+=1# a simple counter!
        print(f'{ch1}, {ch2}, {prob=:.10f} {log_prob=:.4f}')
print(f'{likelihood=}, {log_likelihood=:.4f}, log_likelihood(avg): {log_likelihood/i:.4f}')
# this now results in : 
# word='anonvs'
# ., a, prob=0.1376705319 log_prob=-1.9829
# a, n, prob=0.1604839861 log_prob=-1.8296
# n, o, prob=0.0270638950 log_prob=-3.6096
# o, n, prob=0.3038820326 log_prob=-1.1911
# n, v, prob=0.0030010368 log_prob=-5.8088
# v, s, prob=0.0000000001 log_prob=-23.0259 <---- see the prob is 1e-10! (I had to change the print to :.10f for it to show properly here! otherwise you wouldnt know! and thats why we use a small eps to have the smallest effect on the model outpt as much as possible while preventing -inf)
# s, ., prob=0.1442141682 log_prob=-1.9365
# likelihood=7.864068270946832e-18, log_likelihood=-39.3842, log_likelihood(avg): -5.6263
# this is still a very unlikely word as shown by the -5.6263 log-likelihood but we dont get an error!
# this is what we call smoothing, by adding a value, we smoothen our probablity distribution and remove or 
# decrease the sharp peaks in our distribution. if we increase smoothing, way higher than we should, 
# it overwhelms the whole distribution, make it completely uniform.
# well what we said here is not entirely true! lets elaborate more 
# lets for example consider a case here, suppose the maximum value in our occurances is 100, if we add 900 to it
# 900 just completely destroys the actual effect of the original number, if one entry had occurance count of 1
# it basically has the same likelihood that the one with occurnce count of 100 had. so when doing smoothing
# we must not overdo it, smoothing the peaks gives us better generalization as well
# so what about smoothing probablities then? 
# you might see people do sth that results in a smoother probablity distribution. 
# like they alter the underlying occurances by which the probablity is derived, 
# for example they increase the occurance count like this
barray_smoothed = barray+1 # we can add any number, but 1 is considered a good choice, much higher number can distored the distribution making it uniform and we lose information
probs_smoothed = barray_smoothed/barray_smoothed.sum(dim=1, keepdim=True)
# this has the same effect lets see 
likelihood=1# likelihood calculated by the product of probs
log_likelihood = 0 # specifies how good the model is at assigning the right probablity to parameters according to the underlying dataset
i=0
for word in ["anonvs"]:
    print(f'{word=}')
    word = ['.']+list(word)+['.']
    for ch1,ch2 in zip(word,word[1:]):
        idx1 = atoi[ch1]
        idx2 = atoi[ch2]
        # lets see what probablity  our model has assigned to this pair
        prob = probs_smoothed[idx1, idx2]
        log_prob = prob.log()
        likelihood *= prob.item()
        log_likelihood += log_prob.item()
        i+=1# a simple counter!
        print(f'{ch1}, {ch2}, {prob=:.10f} {log_prob=:.4f}')
print(f'{likelihood=}, {log_likelihood=:.4f}, log_likelihood(avg): {log_likelihood/i:.4f}')
# word='anonvs'
# ., a, prob=0.1375857741 log_prob=-1.9835
# a, n, prob=0.1603856981 log_prob=-1.8302
# n, o, prob=0.0270785652 log_prob=-3.6090
# o, n, prob=0.3029770255 log_prob=-1.1941
# n, v, prob=0.0030511061 log_prob=-5.7923
# v, s, prob=0.0003846154 log_prob=-7.8633
# s, ., prob=0.1438583583 log_prob=-1.9389
# likelihood=3.056283114414935e-11, log_likelihood=-24.2112, log_likelihood(avg): -3.4587
# by increasing the occurance count of all entries in our barray, not only we removed all instances of 0
# which would result in -inf, but also made the probablities smoother.(the sudden peaks are smoothed a bit, the larger the number the more accentuated this effect )
# infact as you can see, it gave us a much smaller loss compared to the previous one 
# this second method actually change the probablities more than our previous method
# infact the previous method is usually used to provide numerical stability in our model operations
# so it doesnt crash. for that reason, we use a very small eps value that doesnt change the probablity distribution much
# but here, we are indeed making probabilities smoother, previously the vs had 0 probablity, but now
# its still very unlikely but much much more probable than previously (0.0003 vs 0.0000000001)


# so to recap: 
# we want to maximize the likelihood of the data with respect to the model parameters (statical modeling), basically make model find the most accurate underlying probablity distribution of data points in our dataset
# which is equivalent of maximzing the "log" likelihood (becasue log is monotonic)(its just scaling the number!)
# which is equivalent of minimizing the negative log likelihood 
# which is equivalent of minimizing the average negative log likelihood
# so finally we quantified our model performance by a negative log likelihood loss which shows how
# it performs, the lower the number the better the model at representing the underlying data probablity distribution
# 

#%%
# next we are going to implement this as a neural network! 
# 