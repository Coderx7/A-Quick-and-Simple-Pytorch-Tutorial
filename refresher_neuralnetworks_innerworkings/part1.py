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

words=open('./data/names.txt', 'r').read().splitlines()
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
names = open('./data/names.txt').read().splitlines()
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
# since we want to sum all the rows columnwise(from left to right) (that is we want all columns from left to righ to
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
# the elementwise division. in doing so it replicates the elements but how? it takes the first entry
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
# so to recap, when doing division here we want to retain the shape of the resuling sum, so 
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
# to recap, consider this, if you have a shape of 27x1, that 1 dimension will be filled when broadcasted
# by what? by whatever value that exist. (imagine you have a single row of 27 numbers like below:
# 27,1: 
# 1 
# 2
# 3
# 4
# ...
# , and now you need to replicate this along the columns (basically make it look like this )
# c0    c1  c2  c..     c26
# 1     1    1  ...     1
# 2     2    2  ...     2
# 3     3    3  ...     3
# 4     4    4  ...     4
# ...           ...
# as you can see this single row of 27 numbers, are each coped to create 27 columns of the same values
# so you can imagine it this way, wherever theres a dim=1, that dim needs to be created and it will 
# be simply the copies of row[i] (row[0] will be copied for all columns, row[1] will be copied for all columns
# basically create full row out of a single value )
# but when we have a single dim, like [27], when it comes to broadcasting in a e.g. (27x27) x (27)
# its infact like 
# 27x27
#    27
# so the first dim here is missing, making this a row vector and treat it as 1x27!
# new explanation: since this is a single row, for our operation we need 26 more so it becomes 27x72
# what is actually done is this whole row is replicated down so we would have : 
# sth like this
# 1x27:
# row0  1     2   3   4   ....    27
# and it will become like this :
# r0    1     2   3   4   ...     27
# r1    1     2   3   4   ...     27
# r2    1     2   3   4   ...     27
# ...
# r26   1     2   3   4   ...     27
# so to recap this, when 
# we have a shape (27,1), its actually a single column vector(this is called a column vector!), 
# and we need to fill the columns
# so what happens is basically this column is replicated 26 times more so it becomes 27x27 as you can 
# imagine, each row would consist of the same values, the previous row consits of.
# what happens is for each row, that single column is replicated 27 times to make up for the missing values
# row 1 will simply be a single value replicated accros all columns.
# but if we have a shape (27), its infact (1x27) or a row vector, with a single row. so what happens
# is the rows need to be created and each row will simply be the replicated row0 27 times. 
# so each row has 27 numbers (they are different column wise)
# now that we sorted all of this information and got ourselves our probs 
# we can go and sample from it. for sampling we use torch.multinomial function 
s = torch.multinomial(probs[0], num_samples=1)
print(f'samples :{s}')
# now lets do better and explain a bit of stuff. 
# before going any further lets have a fixed seed so the results all comeout the same
# each time we run this, for this we create a a random generator in torch and set it
# note that setting different seeds will result in different outputs, sometimes yielding
# better, and sometimes yielding worse results!
g = torch.Generator().manual_seed(255)
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
print(f'Printing a few more samples: ')
for i in range (10):
    chstr=''
    # always start from first row
    idx=0
    while True:
        prob = probs[idx]
        # print(f'{prob=}')
        idx = torch.multinomial(probs[idx], num_samples=1, replacement=True,generator=g).item()
        chstr += itoa[idx]
        if idx==0:
            print(chstr)
            break
#prints
# hel.
# lieyn.
# denarilie.
# desa.
# jaizax.
# adegheviy.
# kann.
# beleavelaylirendon.
# jele.
# jeril.

# now they dont particularly look very good! but our model did learn sth and in order to see
# if they are really random or not, we can change the probablities learned by our model with 
# a uniform probablity that makes the probablity for all characters the same! see how that does
new_prob = torch.ones(27)/27.0
print(f'uniform probabality: {new_prob}')
# prints 
# new_prob=tensor([0.0370, 0.0370, 0.0370, 0.0370, 0.0370, 0.0370, 0.0370, 0.0370, 0.0370,
#                  0.0370, 0.0370, 0.0370, 0.0370, 0.0370, 0.0370, 0.0370, 0.0370, 0.0370,
#                  0.0370, 0.0370, 0.0370, 0.0370, 0.0370, 0.0370, 0.0370, 0.0370, 0.0370])

print(f'Testing sampling with uniform probablity distibution:')
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
#prints 
# bkfklwog.
# ccfxqnjndljso.
# zyekueihst.
# pyapshpo.
# ibmsfzh.
# rufyxkqlxmexaiozwdqgbrdjpaacgxahnwmvtxswuq.
# bmnpqzltgojwpsbn.
# thyyiefgeponyacqhpuxziywniykokdlucesfzvtwogoodipu.
# jkznrtrnmygbohmemqbywdsqfzrdjex.
# gofvbankznqypzsxvk.
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
print(f'{likelihood=}, {log_likelihood=:.4f}, log_likelihood(avg): {-log_likelihood/i:.4f}')
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

# total loss after smoothing:
print(f'loss after smoothing')
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
        prob = probs_smoothed[idx1, idx2]
        log_prob = prob.log()
        likelihood *= prob.item()
        log_likelihood += log_prob.item()
        i+=1# a simple counter!
        print(f'{ch1}, {ch2}, {prob=:.4f} {log_prob=:.4f}')
print(f'{likelihood=}, {log_likelihood=:.4f}, log_likelihood(avg): {log_likelihood/i:.4f}')
#print
# likelihood=0.0, log_likelihood=-560001.8843, log_likelihood(avg): -2.4546

print(f'sampling output of smoothed probs')
# to test the smoothed prob :
for i in range (3):
    chstr=''
    # always start from first row
    idx=0
    while True:
        prob = probs[idx]
        # print(f'{prob=}')
        idx = torch.multinomial(probs_smoothed[idx], num_samples=1, replacement=True,generator=g).item()
        chstr += itoa[idx]
        if idx==0:
            print(chstr)
            break
#output: 
# manakel.
# e.
# su.

# so to recap: 
# we want to maximize the likelihood of the data with respect to the model parameters (statical modeling), basically make model find the most accurate underlying probablity distribution of data points in our dataset
# which is equivalent of maximzing the "log" likelihood (becasue log is monotonic)(its just scaling the number!)
# which is equivalent of minimizing the negative log likelihood 
# which is equivalent of minimizing the average negative log likelihood
# so finally we quantified our model performance by a negative log likelihood loss which shows how
# it performs, the lower the number the better the model at representing the underlying data probablity distribution
# 
# so in otherwords, (in statistical modeling), 
# the likelihood is often defined as the product of probabilities for each individual sample in the dataset. 
# This means that the likelihood represents the probability of observing the entire dataset given the model parameters.
# To calculate the likelihood for the whole dataset, we would multiply the probabilities produced by the model for each sample.
# Alternatively, if we want to calculate the log likelihood (for the reasons we mentioned earlier such as for numerical stability, etc),
# you would sum the log probabilities for each sample.
# For a single example, the likelihood is simply its probability as produced by the model. 
# Taking the logarithm of this probability gives us the log likelihood for that particular example.
# It is a way to transform the likelihood into a more convenient scale for calculations, 
# as well as to simplify computations when dealing with a large number of samples.

#%%
# next we are going to implement this as a neural network! 
# beofor we go on, lets recount what we did previously. 
# we grabed our dataset and tried to infer some character level relationships
# by which we could create new name like output. 
# we say character level becasue it involves only two adjacent characters relationship only
# at a time, and name like, becasue as you saw, they are not prefect suggesting the two character
# relationship is not enough by itself to result in a great name generator!
# we extracted such relationships by simply counting each observed pair's appearance in the dataset
# and creating a table out of it so for a given character we could calculate the probablity of it
# being used with another character. 
# now here we are going to automate this using a nn, a very simple one, a one layer neural network
# to be more precies, one that doesnt even have an activation function! so lets go
# since we are dealing with neural networks, we deal with matrix multiplications, we cant use
# characters, we cant use simple decimal integers either. 
# we know we can map our characters to decimal integers or indexes, so thats solved. 
# but for matrix multiplication involved, we need to code these indexes as well. so we 
# one-hot encode them and then use them. 
# another issue is, in a nn, we need a dataset, a data and a label. how do we create those?
# also our neural network's weight matrix is float not integer, so how can we count!? 
# yes our weights are floats, but we would be able to comeup with the same semantic as the bigram
# model just fine, though using a different way. in nn, we would have log-count to achieve this
# to create a dataset the data would be each character, and its corrosponding label would be
# the character after it, basically we are creating the pairs this way and allow the network to
# learn their relationship this way. 
# lets create our dataset 
xs = []
ys = []
for name in names:
    name = ['.'] + list(name)+['.']
    for ch1, ch2 in pairwise(name):
        idx1 = atoi[ch1]
        idx2 = atoi[ch2]
        xs.append(idx1)
        ys.append(idx2)
# now xs has the first part of our pair, while ys, has the second part of each pair!
# to convert these into onehot vectors, we can use torch.nn.functional one_hot function
import torch.nn.functional as F
# for that we need them to tensors!
xs = torch.tensor(xs)
ys = torch.tensor(ys)
print(xs[:3]) # tensor([ 0,  5, 13])
# make them as float cuz we feed them to our nn
xs_enc = F.one_hot(xs, num_classes=27).float()
ys_enc = F.one_hot(ys, num_classes=27).float()
print(xs_enc.shape, f'{xs_enc[:3]=}')
print(ys_enc.shape, f'{ys_enc[:3]=}') 
# prints 
# (228146, 27) 
# xs[:3]=tensor([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#          0., 0., 0., 0., 0., 0., 0., 0., 0.],
#         [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#          0., 0., 0., 0., 0., 0., 0., 0., 0.],
#         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
#          0., 0., 0., 0., 0., 0., 0., 0., 0.]])
# (228146, 27) 
# ys[:3]=tensor([[0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#          0., 0., 0., 0., 0., 0., 0., 0., 0.],
#         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
#          0., 0., 0., 0., 0., 0., 0., 0., 0.],
#         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
#          0., 0., 0., 0., 0., 0., 0., 0., 0.]])
# and if we visualize it, we see there are 3 rows, and each entry thats 1 is shown in bright yellow
plt.imshow(xs_enc[:3])
# now we need to create our neurons and for that we need a weight matrix
# but what should be the dimensions of our weight matrix? the input is (228146, 27), so our matrix
# must at the very leaast be 27xsth. what should we use for sth here? 
# lets see, if we go for a xs_enc@w where xs_enc is (200kx27) and w is 27x1, we would get 200kx1 which means
# we get a single output per each input. but remember that our input is a vector, our label is also a vector
# a one hot vector of length 27, so when we want to predict the next character, we should also produce the same
# vector so we can compare them. thats why we need an output size as many as input size which is 27 so 
# our weight matrix would be 27x27
# so infact what this means is that we are creating 27 neurons and all of these 27 neurons are looking
# at the all the inputs and create an output. (a neuron had 1-to-many input and only 1 output)
# here we are creating our weight matrix (in fact a column vector) and initialzing it with normal distribution
# where the majority of values are near zero. 
w = torch.randn(size=(27,27))
# since python3.5 @ is a matrix multiplication operator
preds = xs_enc@w 
print(preds[:3])
# tensor([[ 1.1458, -2.0494,  0.1168,  0.7858, -0.0109, -0.9591,  0.3196, -1.2274,
#           0.8380, -1.2459, -0.4174,  0.8508,  0.0758, -0.2004,  0.8828, -0.1573,
#           1.4613, -0.3898, -0.2099,  0.9161,  0.8053,  1.5845, -0.8689,  0.4387,
#           1.0069,  0.1445, -0.1201],
#         [ 1.2338, -0.7942, -0.6653,  1.6304, -0.0830,  0.5843,  0.7250, -1.5760,
#           1.1257, -0.8882,  0.0499, -0.3004,  0.8052, -0.3012,  0.0736,  0.7548,
#           1.2550, -0.2785, -0.6582, -0.5313,  0.8607, -0.2436,  0.2155,  0.2051,
#           1.1669, -0.8757,  0.4197],
#         [-0.4706,  1.4474, -0.3733, -0.2354, -0.5600,  0.6512, -0.0716, -0.7241,
#           0.9495,  0.2969, -0.9082,  1.3144,  0.9458, -0.8236, -0.7642,  0.0828,
#          -1.3355,  1.1673, -1.4464, -1.2507, -0.4051, -1.1882, -1.1765,  1.4431,
#          -0.0098, -1.2935,  1.1376]])

# what we want here intuitively is for our nn to produce a probablity distribution for the next character in the sequence given the input example. (we want probablities),
# but these are negative and postive numbers that we get as the output!(because our weight is initialize by a normal distribution so it contains numbers like that) for a probbability 
# it needs to be some positive numbers that sum up to 1. but these are not like that as you can see.
# we want somehow these numbers to represent probablities for the next character, 
# these numbers are not counts either, because counts are positive integer numbers tha a neural network cant produce
# so what are these numbers? and how should we interpret them? 
# these numbers are infact log-counts (also known as logits), so in order to get the counts, we need to exponentiate them 
# what exp() does, is that it takes negative or positive numbers, and returns a number e^x
# if you feed it a negative number(<0), it always returns a number <1 and if you feed it a number>0
# it will return a number>1. 
print(preds[:3].exp())
print(preds[:3][0,:2])
print(preds[:3].exp()[0,:2])
# print 
# tensor([[3.1450, 0.1288, 1.1239, 2.1941, 0.9892, 0.3833, 1.3766, 0.2931, 2.3117,
#          0.2877, 0.6588, 2.3414, 1.0787, 0.8184, 2.4178, 0.8544, 4.3117, 0.6772,
#          0.8106, 2.4994, 2.2373, 4.8768, 0.4194, 1.5507, 2.7371, 1.1555, 0.8869],
#         [3.4342, 0.4519, 0.5141, 5.1058, 0.9203, 1.7938, 2.0648, 0.2068, 3.0825,
#          0.4114, 1.0512, 0.7405, 2.2370, 0.7399, 1.0763, 2.1271, 3.5077, 0.7569,
#          0.5178, 0.5878, 2.3649, 0.7838, 1.2405, 1.2276, 3.2121, 0.4166, 1.5216],
#         [0.6246, 4.2522, 0.6885, 0.7903, 0.5712, 1.9179, 0.9309, 0.4848, 2.5845,
#          1.3457, 0.4033, 3.7226, 2.5748, 0.4388, 0.4657, 1.0863, 0.2630, 3.2132,
#          0.2354, 0.2863, 0.6669, 0.3048, 0.3083, 4.2339, 0.9903, 0.2743, 3.1194]])
# tensor([ 1.1458, -2.0494])
# tensor([3.1450, 0.1288])

# now as we can see, all the negative numbers are gone, they are turned into positive numbers, and 
# previously positive numbers have become even more positive (larger that is, like for example preds[:3][0,:2])
# now we have access to sth that our nn can use to represent counts, the min value is 0 and it can go up
# so it now can replicate/simulate the barray which contained the occurance counts(basically counts for the next character) of pairs here as well 
# so now we can also calculate the probablities and do what we did previously that is :
# exponentiate to get rid of negatives and represent each value as counts 
log_counts = preds.exp()
# now to calculate probability just divide all counts by their total sum!
# this is infact how we normalize each row to get our probablities (just like before)
probs = log_counts / log_counts.sum(dim=1, keepdim=True)
# so if we sum a row, we know that it sums to 1 signifying its a probablity now!
print(f'{probs[0].sum()=}')
# which now if we print it , 
print(f'{probs[:3]=}')
# prints
# probs[:3]=
# tensor([[0.0602, 0.0074, 0.0276, 0.0302, 0.0132, 0.0338, 0.0224, 0.0045, 0.0526,
#          0.0828, 0.0487, 0.0020, 0.0465, 0.1311, 0.0150, 0.0455, 0.0537, 0.0132,
#          0.0120, 0.0906, 0.0093, 0.0546, 0.0211, 0.0061, 0.0360, 0.0121, 0.0677],
#         [0.0139, 0.0069, 0.0177, 0.1227, 0.0365, 0.0060, 0.0051, 0.0573, 0.0445,
#          0.0148, 0.0230, 0.0249, 0.0283, 0.0059, 0.0233, 0.0156, 0.0280, 0.0438,
#          0.0296, 0.1299, 0.0311, 0.0457, 0.0355, 0.0765, 0.0920, 0.0183, 0.0232],
#         [0.0432, 0.0027, 0.0267, 0.0288, 0.0545, 0.0068, 0.0255, 0.0270, 0.0605,
#          0.0344, 0.0105, 0.0184, 0.0259, 0.0308, 0.0092, 0.0285, 0.1127, 0.0267,
#          0.0261, 0.0230, 0.0516, 0.0239, 0.1957, 0.0576, 0.0167, 0.0155, 0.0171]])

# we see there are 3 output for our 3 input examples, each of these rows now give us probablities
# for the next character given that input.  
# so to recap: 
# when we input@w we got logits, 
# we interpret it as logcount so used exp() to get sth that look likes count (gives us the same behavior)
# when we got our log-count, we normalize it and got probability distribution
# you might remember these last two line resemble sth called softmax, and thats exactly it!
#
# ok now lets have some observations here, to be precise, lets pick a word
# and see what loss our current model (which is a randomly initialized weight w) gives us

nll_all = 0
i=0
log_likelihood_all = 0
# run this for this name only
for name in ["emma"]:
    name = ['.']+list(name)+['.']
    for ch1,ch2 in pairwise(name):
        idx1 = atoi[ch1]
        idx2 = atoi[ch2]
        # we have already calculated the nn prob so lets use it
        prob = probs[idx1,idx2]
        # lets calculate the loglikelihood and 
        # negative loglikelihood loss for this sample as well
        # note that we said likelihood is the product of all probablaties
        # but here we are just taking log of one. if you remember likelihood is per sample
        # if we were to calculate the likelihood of model on the whole dataset, we needed to
        # multiply all the probablities the model produced for each sample, or for the log likelihood
        # sum all of the loglikelihood, but since here we are dealing with one sample, we simply take its log
        # negative loglikelihood is self explanetory then. 
        log_likelihood = prob.log()
        nll = -log_likelihood.item()
        nll_all += nll
        log_likelihood_all +=log_likelihood.item()
        i+=1
        print(f'{ch1},{ch2}, {prob=:.4f} {log_likelihood=:.4f} {nll=:.4f}')
print(f'total nll: {nll_all/i:.4f} total loglikelihood: {log_likelihood_all/i:.4f}')
# prints
# .,e, prob=0.0374 log_likelihood=-3.2860 nll=3.2860
# e,m, prob=0.1176 log_likelihood=-2.1407 nll=2.1407
# m,m, prob=0.0058 log_likelihood=-5.1573 nll=5.1573
# m,a, prob=0.0838 log_likelihood=-2.4798 nll=2.4798
# a,., prob=0.0072 log_likelihood=-4.9288 nll=4.9288
# total nll: 3.5985 total loglikelihood: -3.5985
# 
# so by default observe that the w matrix govern all of this, and since its randomly initialized
# theres not much it can do so we need to optimize w so that it actually gets close to the bigram
# model so lets do that
# what do we need to do this? 
# feed the input and multiply it by the weightmatrix
# calculate the loss
# backprop
# use the gradients and update the weights by a small step towards the opposite direction of gradients
# a loop that does the optimization a few times  
# lets do this once 
# feed the input and multiply it by weight matrix 
# we need to convert our input to onehot encoded form
data = F.one_hot(xs, num_classes=27).float()
# lets do this with our labels 
labels = F.one_hot(ys, num_classes=27)
# lets create a weight matrix with grads enabled!
# lets also add a generator to keep things determinestic for the sake of testing
g = torch.Generator('cpu').manual_seed(255)
W = torch.randn(size=(27,27), requires_grad=True, generator=g)
# now lets do X@W since our memory is limited lets grab a few examples only
logits = data[:5]@W
# convert to log-count 
counts = logits.exp()
# normalize it and get probabality distribution
probs = counts/counts.sum(dim=1, keepdim=True)
# we now probs lets calculate loglikeloohd , we take the log and take its mean
# but note that we want to see how the network predicted according to our label
# so here we say get all rows representing all input samples, and return the corrosponding column
# from the respective label row which is 1 (basically saying extract the same probablity from 
# the probs where label for it says 1)
log_likelihood = probs[:5,labels[:5]].log().mean()
# now lets get our loss which is the negative loglikelihood
nll_loss = -log_likelihood
print(f'loss:{nll_loss:.4f}')
# do a backward but before that make sure to zeroout the weights gradients
W.grad = None
nll_loss.backward()
print(f"W's gradients: {W.grad=}")
# now lets update the weights , note that we need to use w.data to update the actual data
# otherwise we get an error! (note that we do this so this doesnt become part of the computation graph
# that is, so this operation is not considered as a normal operation that needs to be saved in the graph 
# to be used for gradient calculations!)
W.data += -0.01*W.grad

# so lets do this a few times to see the effect 
for i in range(5):
    # forward 5 samples
    logits = data[:5]@W
    # convert to log-count 
    counts = logits.exp()
    # normalize it and get probabality distribution
    probs = counts/counts.sum(dim=1, keepdim=True)
    # we now probs lets calculate loglikeloohd , we take the log and take its mean
    # but note that we want to see how the network predicted according to our label
    # so here we say get all rows representing all input samples, and return the corrosponding column
    # from the respective label row which is 1 (basically saying extract the same probablity from 
    # the probs where label for it says 1)
    log_likelihood = probs[:5,labels[:5]].log().mean()
    # now lets get our loss which is the negative loglikelihood
    nll_loss = -log_likelihood
    print(f'loss:{nll_loss:.4f}')
    # do a backward but before that make sure to zeroout the weights gradients
    W.grad = None
    nll_loss.backward()
    print(f"W's gradients: {W.grad[:5]=}")
    # now lets update the weights , note that we need to use w.data to update the actual data
    # otherwise we get an error! (note that we do this so this doesnt become part of the computation graph
    # that is, so this operation is not considered as a normal operation that needs to be saved in the graph 
    # to be used for gradient calculations!)
    W.data += -0.01*W.grad

# and we see that the loss increased from loss:4.1761 down to loss:4.1626 so if we keep going 
# it should get down even more.
# and to do it for all the data
num_samples = xs.nelement()

for i in range(150):
    # forward 5 samples
    logits = data@W
    # convert to log-count 
    counts = logits.exp()
    # normalize it and get probabality distribution
    probs = counts/counts.sum(dim=1, keepdim=True)
    # we now probs lets calculate loglikeloohd , we take the log and take its mean
    # but note that we want to see how the network predicted according to our label
    # so here we say that lets pick a row (arange returns an iterable of numbers from 0 up to num-1),
    # and, get the second index using label (which is an iterable itself), thats our probability
    # and then go for the next row, use ys to get the second index and get that probability as well
    # continue this until we have a resulting tensor, then run log on the resulting tensor followed 
    # by a mean() to get the whole tensors mean which would be the loglikelihood of the whole samples!
    # the difference with previous snippet is that previously we used slice indexsing so everything
    # happened all at once (copied all into memory atonce and ran the operations) while here, it happens in steps
    # in a loop if you will. we cant use the former way becasue it consumes alot of memory but this approach
    # works fine!
    log_likelihood = probs[torch.arange(num_samples), ys].log().mean()
    # now lets get our loss which is the negative loglikelihood
    nll_loss = -log_likelihood
    print(f'loss:{nll_loss:.4f}')
    # do a backward but before that make sure to zeroout the weights gradients
    W.grad = None
    nll_loss.backward()
    # print(f"W's gradients: {W.grad=}")
    # now lets update the weights , note that we need to use w.data to update the actual data
    # otherwise we get an error! (note that we do this so this doesnt become part of the computation graph
    # that is, so this operation is not considered as a normal operation that needs to be saved in the graph 
    # to be used for gradient calculations!)
    # we tested this and noticed with lr = 50!! it goes fine so lets use that instead of 0.1! to speed up the training!
    W.data += -50*W.grad

# and what do you know? we got loss of loss: 2.4658 and intrestingly its around the same loss of the 
# bigram model which was 2.4540, so we basically got the same result with a gradient based learning
# and the reason is the information is the same here, as we said given the character calculate the 
# probablity of the next character!
# so basically this means, our W must be the same(functionally see below!) as the barray previously, lets print that 
print(f'W={W.data}')
# but this doesnt look like counts right!? yes becasue its a log-count! and we need to exp() it
# to see the real representation 
print(f'W={W.data.exp()}')
print(f'W.min={W.exp().data.min():.4f}')
print(f'W.max={W.exp().data.max():.4f}')
# now we see that all the values are positive and carry on the same role as the numbers in our original barray 
# in the bigram model played. they are not the same exact values, but they have the same function
# and this shows when we plot the weight (note the exp() as its a logcount!) note that it doesnt replicate
# the bigram barray image, and we dont want that eiher, but here we are showing the similarity in function
plt.figure(figsize=(16,16))
plt.imshow(W.data.exp(), cmap='Blues')

# also before we end this : 
# recall that we had smoothing applied on our previous model, in a neural network the smoothing can
# be achieved using 
# in neual network, if we for example initialize all W with zeros, what happens here is that 
# when we do data@W, the logits become zero, and when we exp() that to get count_log
# it all become 1s. basically makes all the probablities uniform. its like adding a large number to 
# barray in our previous bigram model. 
# so the intuision is, if we somehow get Ws value towards zero, closer to zero, it causes a uniformity
# in probablities, it plays a smoothing role just like our previous model and thus gives us a smoother 
# distribution. too much of this effect and we see the same issue, model cant learn, cuz everything
# basically has the same effect! everything is the same, thus it becomes random prediction 
# so if we somehow incentify the W values to be zero, or close to zero, we get the smoothing effect
# so how can we do that? we are using loss to guide the backpropagation , so if we can augment it
# to incentify this, then we get that. 
# what is usually done is we can square all the values of our weight matrix and sum or mean it
# (i.e (W**2).sum() or (W**2).mean, mean may be better as sum can result in a large number)
# by doing this all the negative numbers are gone(by squaring) and we endup with a single number 
# after sum/mean. so this loss is only zero when W is zero, and if not the loss goes up.
# we can then use this in the loss and assign a factor to it(to specify its effect), like this
# loss = negative_log_likelihood_loss + 0.01*(W**2).mean()
# now our loss is two part, and it needs to fullfill both terms to be minimized. 
# the effect is that the first lost tries to learn the probability distribution of the underlying data
# and the second loss tries to find a uniform distribution, therefore the result will be a smoother
# probablity distribution that the model learns. 
# this is what is known as regularization loss or weight decay (l2)
# too much importance to the weight being near zero, and it will overwhelm the loss
# and prevents the learing process altogether thats why high to too much regularization slows and
# flat out stops training process
# the lr in the regularization part is the same as adding numbers to our previous model's barray
# to make it smoother, too much lr, can overwhelm the first loss and create a completely uniform
# distribution that destroys the undelying data probability distribution
#
# And finally lets see how we can test our new model and create some new names:
# for this like before, we start off with an index=0 feed it to our network 
# and then get the probablity and use that to get the second index and go on. 
# recall that we start from index 0 cuz it was ., . was at index ., and all names
# started with . and ended with ., having said that lets do this
for i in range(3):
    idx = 0
    chstr=''
    while True:
        # note the list, this makes it have the shape 1,27 otherwise it wont work in matrix multiplication
        x_enc = F.one_hot(torch.tensor([idx]), num_classes=27).float()
        logits = x_enc@W
        counts = logits.exp()
        probs = counts/counts.sum(dim=1, keepdim=True)
        # now lets sample from it!
        idx = torch.multinomial(probs, num_samples=1, replacement=True, generator=g).item()
        chstr += itoa[idx]
        if idx ==0:
            print(f'{chstr}')
            break
#
#aden.
#atchann.
#lapey.

# 
# (side note, The exponential function (exp()) and the natural logarithm function (log()) are inverse functions of each other. This means that applying log() to the result of exp() will return the original input, and vice versa.) 
# (side note 2)
# Whats a logit exactly?
# In the context of neural networks, the term "logit" refers to the raw, unnormalized output 
# of a neuron or a layer before it is transformed into a probability using a softmax function.
# It represents the log-odds or log-likelihood of a certain class being the correct prediction.
# The reason it is called a logit is because it is the logarithm of the odds ratio, similar to
# its usage in logistic regression. By applying the softmax function to the logits, the output
# is transformed into a valid probability distribution over the classes.
# The logits are often used in the training phase of a neural network, where they are compared
# with the ground truth labels to calculate a loss function. The network then adjusts its parameters
# through backpropagation to minimize this loss and improve the accuracy of its predictions.
# So, in summary, in the context of neural networks, logits represent the unnormalized outputs 
# before softmax transformation, and they are used to compute the probability distribution over
# classes.
# 
# outside of neural networks, the term "logit" is derived from the words "log" and "odds ratio."
# In logistic regression, the logit function is defined as the natural logarithm of the odds ratio.
# The odds ratio is the ratio of the probability of an event occurring to the probability of it 
# not occurring. By taking the logarithm of this ratio, the logit function transforms the 
# probability into a linear scale, allowing for regression analysis.
# The logit function is also used to model the relationship between predictor variables and the
# probability of an event, providing a useful way to estimate the probability of a binary 
# outcome based on the values of the predictors.
