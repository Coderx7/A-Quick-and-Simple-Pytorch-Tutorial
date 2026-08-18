# # in the name of God  the most compassionate the most merciful 
# #%%
# import numpy as np 
# import torch 
# from collections import Counter # used for getting word frequencies
# from string import punctuation  # used to get list of punctuation letters
# import random # used to generate random number in subsample function

# # we are going to implement embedding. one example of embedding is word2vec
# # When you're dealing with words in text, you end up with tens of thousands 
# # of word classes to analyze; one for each word in a vocabulary. Trying to 
# # one-hot encode these words is massively inefficient because most values 
# # in a one-hot vector will be set to zero. So, the matrix multiplication 
# # that happens in between a one-hot input vector and a first, hidden 
# # layer will result in mostly zero-valued hidden outputs.
# # 
# # To solve this problem and greatly increase the efficiency of our networks,
# # we use what are called **embeddings**. Embeddings are just a fully connected
# # layer like you've seen before. We call this layer the embedding layer and the
# # weights are embedding weights. We skip the multiplication into the embedding 
# # layer by instead directly grabbing the hidden layer values from the weight matrix.
# # We can do this because the multiplication of a one-hot encoded vector with a 
# # matrix returns the row of the matrix corresponding the index of the "on" input unit.
# # 
# # Instead of doing the matrix multiplication, we use the weight matrix as a lookup table.
# # We encode the words as integers, for example "heart" is encoded as 958, "mind" as 18094.
# # Then to get hidden layer values for "heart", you just take the 958th row of the 
# # embedding matrix. This process is called an **embedding lookup** and the number 
# # of hidden units is the **embedding dimension**. There is nothing magical going on here.
# # The embedding lookup table is just a weight matrix. The embedding layer is just a
# # hidden layer. The lookup is just a shortcut for the matrix multiplication. 
# # The lookup table is trained just like any weight matrix.
# # Embeddings aren't only used for words of course. You can use them for any model 
# # where you have a massive number of classes. A particular type of model called 
# # **Word2Vec** uses the embedding layer to find vector representations of words
# # that contain semantic meaning.

# with open('data/text8','r') as file: 
#     text = file.read()

# # lets visualize it 
# print(repr(text[:100]))

# #%%
# # Preprocess the text so that 
# # 1.all punctuations are converted into tokens. i.e '.' is <PERIOD>
# # 2.remove all words that have occurences less than 5 ,This will greatly reduce
# #   issues due to noise in the data and improve the quality of the vector representations.
# # 3.return a list of words!
# def preprocess(input_text, is_sorted= False): 
#     from string import punctuation 
#     # print(punctuation) # :-> !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
#     input_text = input_text.replace('!', ' <EXCLAMATION_MARK> ')
#     input_text = input_text.replace('"', ' <QUOTATION_MARK> ')
#     # input_text = input_text.replace('#', ' <Hash_MARK> ')
#     # input_text = input_text.replace('$', ' <DOLLAR_SIGN> ')
#     # input_text = input_text.replace('%', ' <PERCENT_MARK> ')
#     # input_text = input_text.replace('&', ' <APPERCENT_MARK> ')
#     input_text = input_text.replace('(', ' <PAR_LEFT> ')
#     input_text = input_text.replace(')', ' <PAR_RIGHT> ')
#     input_text = input_text.replace('?', ' <QUESTION_MARK> ')
#     input_text = input_text.replace('--', ' <HYPHEN_MARK> ')
#     input_text = input_text.replace(':', ' <COLON> ')
#     input_text = input_text.replace(';', ' <SEMICOLON> ')

# # get count statistics in our text 
#     from collections import Counter
#     if is_sorted:
#         result_dict = Counter(w for w in input_text.split())
#         # a dictionary with words with occurecne more than 5 
#         results_truncated_dict_ = {word:count for word, count in result_dict.items() if count>5}
#         # the sorted() funnction, applies the function (key) 
#         # on each element in the iterable (results_truncated_dict)
#         # this means, the get method is applied on the key (here word name i.e. 'the')
#         # and thus returns its count, then the sort is carried out based on this count!
#         # reversed is true so the most frequenet word sits on top!
#         final_list = sorted(results_truncated_dict_, key=results_truncated_dict_.get, reverse=True)
#     else:
#     # method 2 
#     # we could also do 
#         result_dict = Counter(input_text.split())
#         final_list = [word for word in result_dict if result_dict[word]>5]
     
#     return final_list

# words_list = preprocess(text, is_sorted=True)
# print (words_list[:30])
# #%%
# # some stats 
# print(f'Total number of words: {len(text)}')
# print(f'Total number of Unique words: {len(words_list)}')


# #%%
# # lets create a lookup table (that is word2int and int2word!)
# def lookup_table(words_list): 
#     int2word = dict(enumerate(words_list))
#     word2int = {word:int for int, word in int2word.items()}
#     return word2int, int2word

# word2int, int2word = lookup_table(words_list)
# words_int_list = [word2int[w] for w in words_list]
# print(words_list[:30])
# print(words_int_list[:30])


# #%% [Markdown]
# # ##SubSmpling 
# # the words, such as the, of, that show up often, dont provide
# # much context for nearby words, so if we discard "some" of them 
# # (careful here! we said "some" of them not "all" of them!)
# # we can remove some of the noise from our data and not only have faster
# # training, but also achieve better representation! 
# # This is called subsampling! by Mikolov ( the author of word2vec)
# # and we discard such words, with a probability, this probability
# # is calculated for each word, based on its frquency, and a threashold!
# # the formula is as follows : $$ p(w_i) = 1 - \sqrt{\frac{t}{f(w_i)}} $$
# # here, t is our threshold, f(w_i) is the frequency, and p(w_i) is the probablity
# # we are after! lets implement this 
# def subsample (words_int_list):
#     # note : frequence = word_count / total_number_of_words
#     import random
#     result_dict = Counter(words_int_list)
#     total_word_count = len(words_int_list)
#     threshold = 1e-5 
#     results_with_freq = {word: count/total_word_count for word, count in result_dict.items()}
#     final_list = [word for word,freq in results_with_freq.items() if random.random() < (1 - np.sqrt(threshold/freq)) ]

#     return final_list

# train_words_int = subsample(words_int_list)
# print(train_words_int[:30])

# #%%
# # Now its time for Batching! there is something to note here
# # before we say what we want to do, we know that the surrounding words
# # of a word, correlate more than those that are further away. that is
# # those words that are nearby (around a word w), are more corrolated 
# # than the ones that are further away! Mikolov said, therefore, lets
# # create a window of size C, and randomly choose a number 'R' which 
# # is in the range [1:C], and choose R words from history(i.e before)
# # our word w, and 'R' words from future (words that came after our word w)
# # these words will be our targets. our task will be to get a word and 
# # produce words that are related to that word! so here lets create a 
# # function that creates this kind of sequence form a list of input words!
# # our function will accept a list of words, and index, and a window size
# def get_targets(words_int_list, index, window_size):

#     R = random.randint(1,window_size+1)
#     idx = index
#     start_idx =  idx-R if (idx-R) > 0 else 0
#     end_idx = idx+R 
#     new_list = words_int_list[start_idx:idx] + words_int_list[idx+1:end_idx+1]
#     return new_list

# test_list = [5233, 58, 741, 10571, 27349, 0, 15067, 58112, 3580, 58, 10712]
# print(get_targets(test_list,2,2))

# # test your code!

# # run this cell multiple times to check for random window selection
# int_text = [i for i in range(10)]
# print('Input: ', int_text)
# idx=5 # word index of interest

# target = get_targets(int_text, index=idx, window_size=5)
# print('Target: ', target)  # you should get some indices around the idx
# #%%


# #%%
# # now lets create our get_batches generator
# # our generator function will gives us baatches (one word and its targets)
# def get_batches (words_int_list, batch_size, window_size):

#     number_of_batches = len(words_int_list)//batch_size
#     words_list = words_int_list[:number_of_batches * batch_size]


#     for i in range(0, len(words_list), batch_size):
#         batch = words_list[i: i + batch_size]
#         X = []
#         Y = []
#         for ii in range(len(batch)):
#             batch_x = batch[ii]
#             batch_y = get_targets(batch, index=ii, window_size=window_size)
#             # in order to be able to know how many targets were generated
#             # for each x, (cause its random number in [1:C])
#             # we repeat our number, as many as the target counts
#             # meaning if e.g. we chose, 1, and the targets were 0,2,3
#             # we repeat 1 3 times as well. this way, later on we can easily
#             # distinuish which target belongs to which number!
#             X.extend([batch_x] * len(batch_y))
#             Y.extend(batch_y)
#         yield X,Y 

# # test 
# int_text = [i for i in range(20)]
# x,y = next(get_batches(int_text, batch_size=4, window_size=5))

# print('x\n', x)
# print('y\n', y)

# #%%
# # validation
# # here lets create a function that we can use in our training 
# # to show how well our model is learning. That is, lets choose
# # some common and uncommn words, and then try to show the words
# # similar to them. this would act as our validation and show us 
# # how well our model is doing. in order to 'find' the semantically
# # similar words from 'our embeddings', the one we trained,  
# # we will be using cosine similarity metric. Cosine similarity 
# # is like euclidean metric, which can be used to measure how two vectors
# # are similar(or different). its a much better measure than simple euclidean
# # metric. this is how you calculate cosine similarity : 
# # cosine_sim = cos(theta) = (vec_a . vec_b) / (|vec_a| . |vec_b|)
# def cosine_similarity (embedding_layer, validation_size=16, validation_window=100, device=torch.device('cuda')): 

#     # note that embedding: is our Pytorch module based embedding.
#     # we will work with weights, as you remember, embedding layer 
#     # is simply a 2d matrix, the rows are word indexes and the columns
#     # represent the embedding vector, or hidden vector, or 'word vector' 
#     # so each word, has one word vector .
#     # what we need to do is to fetch the validation words, 
#     # fetch their respective embeddings from our emebedding layer
#     # and compare it with all other embeddings, and show those that 
#     # are similare. 

#     emebddings = embedding_layer.weight
#     print(f'embeddings.shape: {emebddings.shape}')
#     # we know how to calculate similarity for that we need
#     # the magnitute of all vectors , remeber the formula was: 
#     # cos(t) = a.b/|a||b|
#     magnitutes = emebddings.pow(2).sum(dim=1).sqrt().unsqueeze(0)
    
#     # now lets create our validation examples (we could get them form input)
#     # but we can also generate some ourselevs, since all we need is to generate
#     # some numeric numbers (such as 1, 100,23, 1000, etc)
#     # lets create some common words we use sampler, for this which takes
#     # a population (the source) and the size of the new list. then it will
#     # generate random numbers as many as the size we provided from the source
#     # aka population we specified! 
#     validation_samples = np.array([random.sample(range(validation_window),
#                                                  validation_size//2)])
#     # now let us create some uncommen words and add it to our previously 
#     # created validation samples. since our vocabulary (words list) were
#     # initially sorted, sampling from lower bound, i.e 0, would give us 
#     # most frequent (aka common) words, while, the upper bound, would 
#     # give us less frequent and thus more uncommen words. so a word at 
#     # index 1000, is much less frequent than a word located at index 10!
#     validation_samples = np.append(validation_samples,
#                                   random.sample(range(1000,1000+validation_window),
#                                                  validation_size//2))
#     # now that we have our samples ready lets fetch thier corrosponding embeddings(word vectors)
#     # from our embedding. we simply feed our embedding layer the new samples and voila!
#     validation_samples = torch.LongTensor(validation_samples).to(device) 
#     validation_embedding_vectors = embedding_layer(validation_samples)
#     # and now finally! let us calculate the cosine similarity metric!
#     similarities = torch.mm(validation_embedding_vectors, emebddings.t())/magnitutes

#     return validation_samples, similarities




# #%%
# # Now let us create Skipgram model .
# # for skipgram model we need a embedding layer and a softmaxlayer at the end. 
# # and then normally we would use NLLLoss(Negative log likelihood loss) .
# # the below is the normal skipgram model one could write!
# import torch.nn as nn 
# import torch.nn.functional as F
# class Skipgram(nn.Module): 
#     def __init__(self, vocab_size, embedding_dim):
#         super().__init__()
#         self.embedding = nn.Embedding(vocab_size, embedding_dim)
#         self.fc = nn.Linear(embedding_dim, vocab_size)
    
#     def forward(self, input): 
#         output = self.embedding(input)
#         output = self.fc(output)
#         return F.log_softmax(output, dim=1)

# # however since we are going to implement the Negatie sampling our skipgram requires some modification!
# # it will need to have two emebedding layer, one from input to hidden states, and the other from output 
# # to hiddenstates
# class Skipgram_NegativeSampling(nn.Module):
#     def __init__(self, vocab_size, embedding_dim, noise_distribution=None):
#         super().__init__()
#         self.vocab_size = vocab_size
#         self.embedding_dim = emebedding_dim 
#         self.noise_distribution = noise_distribution

#         self.input_embedding = nn.Embedding(vocab_size, emebedding_dim)
#         self.output_embedding = nn.Embedding(vocab_size, emebedding_dim)

#     def forward_input(self, input):
#         return self.input_embedding(input)

#     def forward_output(self, input):
#         return self.output_embedding(input)

#     # this is the actual bit of code that !
#     # basically here we create some random noise words!
#     # and return their embeddings, we later use these embeddings 
#     # in our negative sampling loss that we will be implementing
#     def forward_input_noise(self, batch_size, num_samples):
#         # here the output of this function would be a (batchsize, n_sample, embed_dim)
#         if self.noise_distribution is None : 
#             noise_dist = torch.ones(self.vocab_size)
#         else:
#             noise_words = self.noise_distribution
#         # torch.multinomial 
#         # accepts 3 arguments : 
#         #   input (Tensor) – the input tensor containing probabilities
#         #   num_samples (int) – number of samples to draw
#         #   replacement (bool, optional) – whether to draw with replacement or not    
#         # And returns a tensor where each row contains num_samples indices 
#         # sampled from the multinomial probability distribution located
#         # in the corresponding row of tensor input.
        
#         noiseword = torch.multinomial(self.noise_distribution,
#                                      batch_size * num_samples, 
#                                      replacement=True )

#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         noise_words = noise_words.to(device)

#         # now get the embeddings !
#         noise_embeddings = self.output_embedding(noise_words).view(batch_size, num_samples, self.embedding_dim)

#         return noise_embeddings


# class NegativeSampling(nn.Module):
#     def __init__(self):
#         super().__init__()
        
#     def forward(self, input_embeddings, output_embeddings, noise_embeddings ):

#         batch_size,embedding_size = input_embeddings.shape
#         # input embedding should be a batch of column vectors
#         input_embeddings = input_embeddings.view(batch_size, embedding_size, 1)
#         # while output embedding should be a batch of row vectors (cause we are going to multiply them together!)
#         output_embeddings = output_embeddings.view(batch_size, 1, embedding_size)
#         # torch.bmm is a batch compatible mm! it accepts tensors of shape(batch, m, n)x(batch, n, p)
#         loss = torch.bmm(output_embeddings, input_embeddings).sigmoid().log().squeeze()
#         all_noise_losses = torch.bmm(noise_embeddings.neg(), input_embeddings).sigmoid().log().squeeze()
#         noise_loss = all_noise_losses.sum(1)

#         return -(loss + noise_loss).mean()
# #%%
# # now training loop 
# # first using our simple skipgram model 
# vocab_size = len(word2int)
# embedding_dim = 400
# window_size = 5
# batch_size = 128 
# epochs = 4
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model = Skipgram(vocab_size, embedding_dim=embedding_dim).to(device)
# print(model)
# # since in our first method of skipgram, we use log_softmax,
# # therefore we use negative log likelihood loss (NLLLoss)
# criterion = nn.NLLLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
# step=0
# interval = 500
# for e in range(epochs): 
#     for words, targets in get_batches(words_int_list, batch_size=batch_size, window_size=window_size):

#         words = torch.LongTensor(words).to(device)
#         targets = torch.LongTensor(targets).to(device)

#         outputs = model(words)
#         loss = criterion(outputs, targets)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         if step%interval == 0:
#             with torch.no_grad():
#                 validation_examples, validation_similarities = cosine_similarity(model.embedding, device=device)
#                 _, indexes = validation_examples.topk(6) # top 6 similar vectors!

#                 validation_examples = validation_examples.to('cpu')
#                 indexes = indexes.to('cpu')

#                 for ii, valid_idx in enumerate(validation_examples):
#                     closest_word = [int2word[idx.item()] for idx in indexes[ii]][1:]
#                     print(int2word[valid_idx.item()] + ', '.join(closest_word))
#                     print('...')

#                 # # getting examples and similarities      
#                 # valid_examples, valid_similarities = cosine_similarity(model.embed, device=device)
#                 # _, closest_idxs = valid_examples.topk(6) # topk highest similarities
                
#                 # valid_examples, closest_idxs = valid_examples.to('cpu'), closest_idxs.to('cpu')

#                 # for ii, valid_idx in enumerate(valid_examples):
#                 #     closest_words = [int2word[idx.item()] for idx in closest_idxs[ii]][1:]
#                 #     print(int2word[valid_idx.item()] + " | " + ', '.join(closest_words))
#                 # print("...")    

#%%
# lets start over! here we are going to implement a skipgram and skipgram with negative sampling 
# here is a list of what we should be doing 
# 1.read the text 
# 2.tokenize all the punctionations (since we want to learn word embedding not punctuation embedding!!)
# 3.since our embedding requires intergers, we need to preprocess our text to integers (word2int, int2word)
# 4.we need sampling for our skipgram model. mikolov proposed a method. choose a random number from [1-C], 
#   C being the windows size, return that random amount from the two side of a given word.
# 5.create a batching generator that gives us training data and their targets words(labels)
# 6.create a method that generates some target words for a given word, this way we can know
#   how we are doing in the training process! it will act as our validation! mechanism! of some sort!! 
#   

# import needed modules 
import time
import numpy as np 
import torch 
import torch.nn as nn
import torch.nn.functional as F 
import random
from string import punctuation
from collections import Counter 

# read the text 
with open('data/text8','r') as file: 
    text_raw = file.read()
# inspect the contents     
print(repr(text_raw[:100]))

# create a function that replaces punctualtions with tokens
def replace_punctuations(text_raw):
    print(f'\nlist of all punctuations :\n{punctuation}')
    text_raw = text_raw.replace('.','< PERIOD >')
    text_raw = text_raw.replace(')','< PAR_RIGHT >')
    text_raw = text_raw.replace('(','< PAR_LEFT >')
    text_raw = text_raw.replace('}','< BRACE_RIGHT >')
    text_raw = text_raw.replace('{','< BRACE_LEFT >')
    text_raw = text_raw.replace(']','< BRACKET_RIGHT >')
    text_raw = text_raw.replace('[','< BRACKET_LEFT >')
    text_raw = text_raw.replace(',','< COMMA >')
    text_raw = text_raw.replace(';','< SEMICOLON >')
    text_raw = text_raw.replace('"','< D_QOUTE >')
    text_raw = text_raw.replace("'",'< S_QUOTE >')
    text_raw = text_raw.replace(':','< COLON >')
    text_raw = text_raw.replace('!','< EXCL_MARK >')
    text_raw = text_raw.replace('?','< QUESTION_MARK >')
    text_raw = text_raw.replace('=','< EQUAL_MARK >')
    text_raw = text_raw.replace('+','< PLUS_MARK >')
    text_raw = text_raw.replace('~','< TILDE_MARK >')

    return text_raw

text_raw_puncless = replace_punctuations(text_raw)
#%%
# now lets convert into interger 
# def get_lookup_table_inefficient(text_raw_puncless, is_sorted=False): 
#     # first we need to get only 'words!' and not characters! so we do 
#     raw_word_list = text_raw_puncless.split()
#     if is_sorted:
#         # get the occurence of each word!
#         result_dict = Counter(raw_word_list)
#         # sort them and get a new list!
#         int2word = dict(enumerate(sorted(result_dict, key=result_dict.get,reverse=True)))
#     else:
#         int2word = dict(enumerate(raw_word_list))
#     word2int = {word:int for int,word in int2word.items()}
#     return word2int, int2word

# word2int, int2word = get_lookup_table_inefficient(text_raw_puncless, is_sorted=True)
# print(word2int)
# print('...')
# print(int2word)
#%%
# there is a problem with former implementation, by discarding 
# words that are rarely used in our text, we can prevent noise
# in our data and thus have a better ebmedding later!
def preprocess(text_raw_puncless):
    raw_word_list = text_raw_puncless.split()
    words_dict = Counter(raw_word_list)
    words_list_cleaned = [word for word in raw_word_list if words_dict[word]>5]
    return words_list_cleaned

raw_words_list_processed = preprocess(text_raw_puncless)
print('raw word count: ', len(text_raw_puncless.split()))
print('cleaned word count: ', len(raw_words_list_processed))
print('unique word count: ', len(set(raw_words_list_processed)))

print(raw_words_list_processed[:10])

#%%
def get_lookup_table(raw_words_list_processed, is_sorted):
    
    words_cnt_dict = Counter(raw_words_list_processed)
    if is_sorted:
        x = sorted(words_cnt_dict, key=words_cnt_dict.get, reverse=True)
        int2word = dict(enumerate(x))
        word2int = {word:int for int,word in int2word.items()}
    else:         
        int2word = dict(enumerate(raw_words_list_processed))
        word2int = {word:int for int,word in int2word.items()}

    return word2int, int2word

word2int, int2word = get_lookup_table(raw_words_list_processed, is_sorted=True)
print(len(word2int))
print('the:',word2int['the'])
print(int2word[0])
print('...')

# word2int2, int2word2 = get_lookup_table(raw_words_list_processed, is_sorted=False)
# print('the:',word2int2['the'])
# print(int2word2[0])
# print('...')
#%% 
#create integer based vocab!
word_list_train = [word2int[word] for word in raw_words_list_processed]
print(len(word_list_train)) 
print(word_list_train[:30])

# we need to discard some common and uncommon words in our data such as of, the,  
# etc, this will help prevent noises in our data! 
# but how do we discard a word? what word should we discard? mikolov answered this
# for us. he said discarding a word with probablity p=1- sqrt(threshold/word_freq_in_dataset)
# will give us good result so we use his formula as well. 
def subsample(words_list):

    # first lets get the list of all words frequency
    # for that we first need each words count in the datset 
    #all_word_lists = text_raw_puncless.split()
    total_word_count = len(words_list)
    threshold = 1e-5
    word_count_dict = Counter(words_list)
    freqs = {word: count/total_word_count for word, count in word_count_dict.items()}
    p_drop = {word: 1 - np.sqrt(threshold/freqs[word]) for word in word_count_dict}
    # discard some frequent words, according to the subsampling equation
    # create a new list of words for training
    new_word_list = [word for word in words_list if random.random() < (1 - p_drop[word])]
    return new_word_list

word_list_train = subsample(word_list_train)
print(len(word_list_train))
# word_list_train = [word2int[word] for word in word_list_raw_subsampled]  
# visualize the result
print(word_list_train[:100])

#%%
# now  that we have preprocess our data by removing punctuations, discarding words fewer than 5, 
# and finally discarded some words by subsampling them, we have our training data ready!
# we no need a batching mechanism and then training! 
# for batching, for each input word, we need some targets. where do we get these targets from you ask?
# mikolov said we will randomly choose 'r' number of words from [1:C] from history and future of a given word
#  in the window C!
# that is, we get ;r; words come before our given word, and 'r' words after our given word! 
# suppose our text was : In the name of God the most compassionate the most merciful.
# suppose, our [word]window size is 3. this means, we will fetch 3 words before and after ou rgiven word!
# let our input word be 'name' and window_size 3. our targets will be: [In, the, of, God, the]
# so lets implement a function that accepts an input word (word index in fact!) and returns its targets 
def get_targets(word_list_train, input_word_idx, window_size=5):
    
    # our random length 
    #np.random.seed(1)
    r = random.randint(1, window_size)
    start = (input_word_idx - r) if (input_word_idx-r) > 0 else 0 
    end = (input_word_idx + r) if (input_word_idx + r) < len(word_list_train) else len(word_list_train)

    targets = word_list_train[start : input_word_idx]  +  word_list_train[input_word_idx+1 : end+1]
    return targets

# #%%
# # run this cell multiple times to check for random window selection
# int_text = [i for i in range(10)]
# print('Input: ', int_text)
# idx=5 # word index of interest

# target = get_target(int_text, idx=idx, window_size=5)
# print('Target: ', target)  # you should get some indices around the idx
#%%
# test the function 
string = [i for i in range(10)]
idx = random.randint(1,5)
window = 5
print(f'input : {string}')
print (f'idx: {idx} window: {window}')
targets = get_targets(string, input_word_idx = idx, window_size = window)
print(targets)
#%%
# implement the batching! create a function that accepts batchsize, window_size and training_data
# and returns a batch of data containing batch_number of words + their target words 
def get_batch(word_list_train, batch_size, window_size=5):

    total_words_count = len(word_list_train)
    n_batches = total_words_count//batch_size

    words_list = word_list_train[: batch_size * n_batches]

    for i in range(0 , len(words_list), batch_size):
        batch = words_list[i: i+batch_size]

        X, Y = [], []
        for ii in range(len(batch)):
            x = batch[ii]
            y = get_targets(batch, ii, window_size=window_size)
            X.extend([x] * len(y))
            Y.extend(y)
        yield X, Y

(words, targets) = next(iter(get_batch(word_list_train, 3, 5)))
print(f'words: {words} ')
print(f'targets: {targets}')

#%% [markdown]
# now lets create a validation method that will print some random words
# and couple of its closest counter parts(word vectors, embeddings, etc)
# for checking the similarity we will be using the cosine similarity check
# so here we go 
# we first get a set of random words. 
# then we should check all the embeddings and retrieve those that are similar
# how do we do that? we have our inputs, we need to get each ones embeddings
# then we check if any embeddings resembles or is close to our embedings
# so for this we would need an embeding layer. we simply feed our input and get
# our embedding since if you remember, our embedding layer was simply a lookup table.(
# think of it as a simple dictionary of some sort!) we also use this embeding layers weight
# matrix to calculate each emebddings magnitute (since we need it for cosine similarity metric).
# the cosine similarity formula is as follows : vec_a . vec_b / |vec_a||vec_b|
# basically this means, we should multiply our input embeddings, by each embedding 
# and divide them by the magnitute of them! 
def cosine_similarity_validation(embedding_layer, validation_size=16, validation_window=100, device='cpu'):
    embeddings = embedding_layer.weight
    # now calculate the magnitute for each embedding(wordvector)
    magnitutes = embeddings.pow(2).sum(dim=1).sqrt().unsqueeze(0)

    # create some random words ! how do we do that? 
    # we can randomly do this, but also, in order to have better assessment
    # of how we are doing, lets create some common words, and some less common ones
    # how do we do that? if you remember, our training set (word_list_train) is 
    # ordered ascendingly meaning, the most common word is ranked first and 
    # the least one ranked last! so the most common ones are in lower indexes and 
    # less frequent ones in larges index ones. 
    validation_words_common = np.array(random.sample(range(0, validation_window),
                                                     validation_size//2))
    validation_words_less_common = np.array(random.sample(range(1000, 1000+validation_window),
                                                          validation_size//2))

    validation_words = validation_words_common + validation_words_less_common

    # before feeding our words into embedding layer and get its respective embeddings
    # we need to make this into torch tensor 
    # since we need them to be long. simply converting them to torch.long will do it!
    validation_words_tensors = torch.LongTensor(validation_words).to(device)
    # getting the embeddings 
    validation_embeddings = embedding_layer(validation_words_tensors)

    # now lets get the similar embeddings. by multiplying our embeddings with 
    # the weight matrix (i.e all the embeddings) and dvicid by the magnitutes, we 
    # can get which one is more similare (the similre vectors, should have vector
    # values in the same range, and thus when multiplied, get bigger, and when different
    # will get smaller, and thus dividing by their magnitute, can give us an idea how similar
    # their values are and therefore how similar they are to each other)
    similarities = torch.mm(validation_embeddings, embeddings.t()) / magnitutes

    return validation_words_tensors, similarities

#%% 
# now that we create all previous ingredients, let us create our skipgram model 
class SkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.embeddingLayer = nn.Embedding(vocab_size, embedding_dim)
        self.fc = nn.Linear(embedding_dim, vocab_size)

    def forward(self, input):
        output = self.embeddingLayer(input)
        scores = self.fc(output)
        logs_probs =  F.log_softmax(scores, dim=1)
        return logs_probs

print(len(word_list_train))
#%% 
print(len(word_list_train))
import time
# now let us create our training loop ! 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vocab_size = len(int2word)
embedding_dim = 600
model = SkipGram(vocab_size, embedding_dim).to(device)
print(model)
# since we are using a log_softmax in the last layer
# we must use negative log likelihood loss to form a crossentropy loss! 
criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

epochs = 4
intervals = 500 
batch_size = 512
window_size = 5
i=0

print(f'training started : {time.ctime()}')
print(f'total epochs :{epochs}')
print(f'total iterations per epoch :{len(word_list_train)/batch_size}')

for e in range(epochs): 

    for words, targets in get_batch(word_list_train, batch_size):
        i+=1
        words = torch.LongTensor(words).to(device)
        targets = torch.LongTensor(targets).to(device)

        # words = words
        # targets= targets

        output = model(words)
        loss = criterion(output, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % intervals == 0: 
            validation_words, similarities = cosine_similarity_validation(model.embeddingLayer,
                                                                            validation_size=16,
                                                                            validation_window=100
                                                                            ,device=device)

            # get the top 6 similar vectors 
            data, indexes = similarities.topk(6)
            validation_words = validation_words.to('cpu')
            indexes = indexes.to('cpu')
            print(f'({time.ctime()}) iter/epoch: {i}/{e}')    
            for ii, example_idx in enumerate(validation_words): 
                closes_words = [int2word[idx.item()] for idx in indexes[ii]]
                print(f'{int2word[example_idx.item()]}'+ ''.join(f'{closes_words}' ))
                print('...')

print('done!')
#%%
# now lets visualize these embeddings in tsne 
import matplotlib.pyplot as plt 
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

from sklearn.manifold import TSNE
# getting the embeddings
embeddings_numpy = model.embeddingLayer.weight.to('cpu').detach().numpy()

vis_words = 600 
tsne = TSNE()
embed_tsne = tsne.fit_transform(embeddings_numpy[:vis_words, :])

fig, ax = plt.subplots(figsize=(16,16))
ax.set_facecolor("white")
for idx in range(vis_words):
    plt.scatter(*embed_tsne[idx,:], color='blue')
    plt.annotate(int2word[idx], (embed_tsne[idx, 0], embed_tsne[idx, 1]), color='black', alpha=0.9)
    

#%%
# now let us create our training loop ! 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vocab_size = len(int2word)
# it seems as the embedding _dim increases, the granularity decreases and 
# it becomes coarser, meaning, when embedding_dim is lower, more words that 
# are closely related are grouped together, whereas when embedding_dim is larger
# more words are being found as related (meaing the relationship is not that close!)
# however, it seems as the dimensionality increases, more relateable words are also 
# discovered! so I dont know what to say!!! my 50d model see to work much better than
# my 10d model!
embedding_dim = 50
model = SkipGram(vocab_size, embedding_dim).to(device)
print(model)
# since we are using a log_softmax in the last layer
# we must use negative log likelihood loss to form a crossentropy loss! 
criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

epochs = 4
intervals = 500 
batch_size = 512
window_size = 5
i=0

print(f'training started : {time.ctime()}')
print(f'total epochs :{epochs}')
print(f'total iterations per epoch :{len(word_list_train)/batch_size}')

for e in range(epochs): 

    for words, targets in get_batch(word_list_train, batch_size):
        i+=1
        words = torch.LongTensor(words).to(device)
        targets = torch.LongTensor(targets).to(device)

        # words = words
        # targets= targets

        output = model(words)
        loss = criterion(output, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % intervals == 0: 
            validation_words, similarities = cosine_similarity_validation(model.embeddingLayer,
                                                                            validation_size=16,
                                                                            validation_window=100
                                                                            ,device=device)

            # get the top 6 similar vectors 
            data, indexes = similarities.topk(6)
            validation_words = validation_words.to('cpu')
            indexes = indexes.to('cpu')
            print(f'({time.ctime()}) iter/epoch: {i}/{e}')    
            for ii, example_idx in enumerate(validation_words): 
                closes_words = [int2word[idx.item()] for idx in indexes[ii]]
                print(f'{int2word[example_idx.item()]}'+ ''.join(f'{closes_words}' ))
                print('...')

print('done!')

#%%
# now lets visualize these embeddings in tsne 
import matplotlib.pyplot as plt 
#%matplotlib inline
%config InlineBackend.figure_format = 'retina'

from sklearn.manifold import TSNE
# getting the embeddings
embeddings_numpy = model.embeddingLayer.weight.to('cpu').detach().numpy()

vis_words = 600 
tsne = TSNE()
embed_tsne = tsne.fit_transform(embeddings_numpy[:vis_words, :])

fig, ax = plt.subplots(figsize=(16,16))
ax.set_facecolor("white")
for idx in range(vis_words):
    plt.scatter(*embed_tsne[idx,:], color='blue')
    plt.annotate(int2word[idx], (embed_tsne[idx, 0], embed_tsne[idx, 1]), color='black', alpha=0.9)
#%%
model_name = 'model_50.t'
states = {'embedding_dim':50,
         'batch_size':batch_size,
         'epochs': epochs,
         'state_dict':model.state_dict(),
         'int2words':int2word,
         'window_size':5}
with open(model_name, 'wb') as f:          
    torch.save(states, f)       

#%%
# now let us create our training loop ! 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vocab_size = len(int2word)
# it seems as the embedding _dim increases, the granularity decreases and 
# it becomes coarser, meaning, when embedding_dim is lower, more words that 
# are closely related are grouped together, whereas when embedding_dim is larger
# more words are being found as related (meaing the relationship is not that close!)
#####
#### OK Here is the final verdict!
# 1. The more embedding dimension seems to provide much better relatiohships 
# 2. it seems more epochs /excessive epochs, doesnt contribute to better embeddings
#    it may very badly affect the result ! 
# 3.  
####
embedding_dim = 10
model_10 = SkipGram(vocab_size, embedding_dim).to(device)
print(model_10)
# since we are using a log_softmax in the last layer
# we must use negative log likelihood loss to form a crossentropy loss! 
criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model_10.parameters(), lr=0.003)

epochs = 20
intervals = 500 
batch_size = 512
window_size = 5
i=0

print(f'training started : {time.ctime()}')
print(f'total epochs :{epochs}')
print(f'total iterations per epoch :{len(word_list_train)/batch_size}')

for e in range(epochs): 

    for words, targets in get_batch(word_list_train, batch_size):
        i+=1
        words = torch.LongTensor(words).to(device)
        targets = torch.LongTensor(targets).to(device)

        # words = words
        # targets= targets

        output = model_10(words)
        loss = criterion(output, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % intervals == 0: 
            validation_words, similarities = cosine_similarity_validation(model_10.embeddingLayer,
                                                                            validation_size=16,
                                                                            validation_window=100
                                                                            ,device=device)

            # get the top 6 similar vectors 
            data, indexes = similarities.topk(6)
            validation_words = validation_words.to('cpu')
            indexes = indexes.to('cpu')
            print(f'({time.ctime()}) iter/epoch: {i}/{e}')    
            for ii, example_idx in enumerate(validation_words): 
                closes_words = [int2word[idx.item()] for idx in indexes[ii]]
                print(f'{int2word[example_idx.item()]}'+ ''.join(f'{closes_words}' ))
                print('...')

print('done!')

#%%
model_name = 'model_10_e20.t'
states = {'embedding_dim':embedding_dim,
         'batch_size':batch_size,
         'epochs': epochs,
         'state_dict':model_10.state_dict(),
         'int2words':int2word,
         'window_size':5}
with open(model_name, 'wb') as f:          
    torch.save(states, f)         
#%%
# now lets visualize these embeddings in tsne 
import matplotlib.pyplot as plt 
#%matplotlib inline
%config InlineBackend.figure_format = 'retina'

from sklearn.manifold import TSNE
# getting the embeddings
embeddings_numpy = model_10.embeddingLayer.weight.to('cpu').detach().numpy()

vis_words = 600 
tsne = TSNE()
embed_tsne = tsne.fit_transform(embeddings_numpy[:vis_words, :])

fig, ax = plt.subplots(figsize=(16,16))
ax.set_facecolor("white")
for idx in range(vis_words):
    plt.scatter(*embed_tsne[idx,:], color='blue')
    plt.annotate(int2word[idx], (embed_tsne[idx, 0], embed_tsne[idx, 1]), color='black', alpha=0.9)
    
#%% 
# lets test our models 
import torch
def get_similar_words(model, word_list, int2word_dict, topk=6, device='cpu'):
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model
    # first lets create the word2int 
    word2int = {word:int for int, word in int2word_dict.items()}
    
    word_idx = [word2int[word] for word in word_list]
    print(word_idx)
    # check for similarity 
    # 1. first feed our word and get its embedding
    #   1.1. Convert it into torch.LongTensor()
    # 2. calculate all emebedding's magnitue
    # 3. multiply our word embedding with all embeddings based on our cosine formula
    # 4. get the topk similarities  

    # 1. get our word embedding
    # becasue we used two models, (skipgram and skipgram with negative samplng)
    # and I made two different embedding layer names in each!!! 
    # for this function  to be used by both of them I need to do a shenanigan!!
    # basically I check if an attribute of our model exists, 
    # if not use the other name! read this :
    #  https://www.pythoncentral.io/how-to-check-if-an-object-has-an-attribute-in-python/
    # method 1 : 
    # if hasattr(model, 'embeddingLayer'):
    #     embedding_layer = model.embeddingLayer
    # else:
    #     # for when we are using negative sampling model
    #     embedding_layer = model.input_embedding
    # or method 2 : 
    try : 
        embedding_layer = model.embeddingLayer
    except AttributeError:
        print('member doesnt exits, using input_embedding attribute instead!') 
        embedding_layer = model.input_embedding

    word_idx_tensor = torch.LongTensor(word_idx)
    print(word_idx_tensor)
    # get our input word embeddings
    word_embedding = embedding_layer(word_idx_tensor)

    # 2.calculate all embeddings magnitute 
    magnitutes = embedding_layer.weight.pow(2).sum(dim=1).sqrt()
    # all of our embedding
    embeddings = embedding_layer.weight

    print(word_embedding.shape)
    print(embeddings.shape)
    # 3. according to cosine similarity, lets multiply them! 
    similarity = torch.mm(word_embedding, embedding_layer.weight.t())/magnitutes

    values, indexes = similarity.topk(topk)
    print(f'similar words : ')
    #print(indexes)
    for i, indexe in enumerate(indexes): 
        
        print(word_list[i]+': ' ,', '.join([ int2word[idx.item()] for idx in indexe ] )) 


#%% 
# load model 
model_name = 'model_10_e20.t'
states = torch.load(model_name)
embedding_dim = states['embedding_dim']
batch_size = states['batch_size']
epochs = states['epochs']
int2word= states['int2words']
window_size = states['window_size']
print(len(int2word))
model_10 = SkipGram(len(int2word), embedding_dim=embedding_dim)
model_10.load_state_dict(states['state_dict'])

words_list = ['penis', 'woman', 'girl', 'gold', 'iran', 'persia', 'horse']
get_similar_words(model_10.cpu(), words_list, int2word)
#%%
import matplotlib.pyplot as plt 
#%matplotlib inline
%config InlineBackend.figure_format = 'retina'

from sklearn.manifold import TSNE
# getting the embeddings
embeddings_numpy = model_10.embeddingLayer.weight.to('cpu').detach().numpy()

vis_words = 600 
tsne = TSNE()
embed_tsne = tsne.fit_transform(embeddings_numpy[:vis_words, :])

fig, ax = plt.subplots(figsize=(16,16))
ax.set_facecolor("white")
for idx in range(vis_words):
    plt.scatter(*embed_tsne[idx,:], color='blue')
    plt.annotate(int2word[idx], (embed_tsne[idx, 0], embed_tsne[idx, 1]), color='black', alpha=0.9)


#%%
# Now that we have done this using simple SkipGram method. lets also implement 
# Negative Sampling Loss. and use it in our SkipGram model for better and faster results 
#
#  basically in negative sampling, we are going to do this 
# 1. get our input word embedding
# 2. get our output word emebedding 
# 3. it means we feed our word, get a target, and then feed that target
#    and we should get back our input word(embedding) 
# 4. meanwhile for the incorrect samples, we randomly choose some words
# in our loss, we are going to maximise the similarity (and thus loss between) ouput vector
# and input vector embeddings and increase the loss for 

class Skipgram_with_NegativeSampling(nn.Module):
    def __init__(self, vocab_size, embedding_dim, noise_dist = None):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.noise_dist = noise_dist

        self.input_embedding = nn.Embedding(vocab_size, embedding_dim=embedding_dim)
        self.output_embedding = nn.Embedding(vocab_size, embedding_dim=embedding_dim)
    
    def forward_input(self, input_words):
        return self.input_embedding(input_words)
    
    def forward_output(self, output_words):
        return self.output_embedding(output_words)
    
    def forward_noise(self, batch_size, n_samples, device):
        # first lets specify a set of noise words
        # we create our noise words, how do we do that?
        # we simply sample from a distribution our choice!
        if self.noise_dist == None: 
            # sample word uniformally!
            noise_distribution = torch.ones(self.vocab_size)
        else :
            noise_distribution = self.noise_dist
        
        # now lets sample some noise words from our distribution 
        # we use torch.multinomial() for this 
        noise_words = torch.multinomial(noise_distribution, 
                                        batch_size * n_samples,
                                        replacement=True)
        noise_words = noise_words.to(device)

        # now that we have our noise words, lets get their embeddings
        # from our output embedding layer! 
        noise_embedding = self.output_embedding(noise_words).view(batch_size,
                                                                  n_samples,
                                                                  self.embedding_dim)
        return noise_embedding

#%%
# now we need to implement our loss (Negative Sampling loss ) and  take into account
# our input embedding, output embedding and noise emebeddings 
class NegativeSamplingLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_embeddings, output_embeddings, noise_embeddings, debug=False):

        #multiply outout embeddings and input_embeddings (based on our formula)
        # read the note before the actual multiplication as well!!!
        # before that lets reshape them for the multiplication
        # they have shape of (batchsize, embedsize) so lets do :
        if debug:
            print(f'input_embeddings.shape = {input_embeddings.shape}')
            print(f'output_embeddings.shape = {output_embeddings.shape}')

        batch_size, embedding_dim = input_embeddings.shape 

        input_embeddings = input_embeddings.view(batch_size, embedding_dim, 1)
        output_embeddings = output_embeddings.view(batch_size, 1, embedding_dim)

        # basically if we ignore the batchsize, they are now (1, embedding) and (embedding,1)
        # and we can easily multiply them and get a (1x1) result 
        # remember we are multiplying output_emb by input_emb so the shape becomes 1x1!
        # otherwise we would get a (embedding x embedding) tensor which would not work! for our loss
        # as we need a single value to add to the second loss below(noise loss) as well. 
        # note: log(1) is 0, log(0) is 1. so if both embeddings, are the same, their multiplication
        # will ultimately be 1. and log(1) is 0, which means the loss is 0 . similarly if 
        # the result is anything but 1 (i.e. less than 1 like 0.5 or 0.8 etc) the log of that
        # will be a negative number, correctly signifying our loss obviously that needs to be 
        # subtracted
        loss1 = torch.bmm(output_embeddings, input_embeddings).sigmoid().log()
        loss1 = loss1.squeeze()

        if debug:
            print(f'loss1.shape = {loss1.shape}')

        # now the second term of our loss, we multiply our noise words (unrelated words)
        # with our input embeddings. we use the negative of their values 
        # so as the multiplication result increases in magnitute this means more related here
        # (and thus the actual values become more unrelated!) 
        # and this is what we want.(remember we are negating the noise words embeddings values,
        # and then we use log and it becomes positive so we basically try to minimize this loss
        # the multiplication becomes, negative, the log makes it possitive, the decrease in loss
        # means, less negativeness (i.e more positiveness, which means more relateness to input word)
        if debug:
            print(f'input_embeddings.shape = {input_embeddings.shape}')
            print(f'noise_embeddings.shape = {noise_embeddings.shape}')
        noise_loss = torch.bmm(noise_embeddings.neg(), input_embeddings).sigmoid().log()
        noise_loss = noise_loss.squeeze().sum(1)
        if debug:
            print(f'noise_loss.shape = {noise_loss.shape}')

        return -(loss1 + noise_loss).mean()
#%%
# lets test our models 
import torch
def get_similar_words(model, word_list, int2word_dict, topk=6, device='cpu'):
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model
    # first lets create the word2int 
    word2int = {word:int for int, word in int2word_dict.items()}
    
    word_idx = [word2int[word] for word in word_list]
    print(word_idx)
    # check for similarity 
    # 1. first feed our word and get its embedding
    #   1.1. Convert it into torch.LongTensor()
    # 2. calculate all emebedding's magnitue
    # 3. multiply our word embedding with all embeddings based on our cosine formula
    # 4. get the topk similarities  

    # 1. get our word embedding
    # becasue we used two models, (skipgram and skipgram with negative samplng)
    # and I made two different embedding layer names in each!!! 
    # for this function  to be used by both of them I need to do a shenanigan!!
    # basically I check if an attribute of our model exists, 
    # if not use the other name! read this :
    #  https://www.pythoncentral.io/how-to-check-if-an-object-has-an-attribute-in-python/
    # method 1 : 
    if hasattr(model, 'embeddingLayer'):
        embedding_layer = model.embeddingLayer
    else:
        # for when we are using negative sampling model
        embedding_layer = model.input_embedding
    # or method 2 : 
    # try : 
    #     embedding_layer = model.embeddingLayer
    # except AttributeError:
    #     print('member doesnt exits, using input_embedding attribute instead!') 
    #     embedding_layer = model.input_embedding

    word_idx_tensor = torch.LongTensor(word_idx)
    print(word_idx_tensor)
    # get our input word embeddings
    word_embedding = embedding_layer(word_idx_tensor)

    # 2.calculate all embeddings magnitute 
    magnitutes = embedding_layer.weight.pow(2).sum(dim=1).sqrt()
    # all of our embedding
    embeddings = embedding_layer.weight

    print(word_embedding.shape)
    print(embeddings.shape)
    # 3. according to cosine similarity, lets multiply them! 
    similarity = torch.mm(word_embedding, embedding_layer.weight.t())/magnitutes

    values, indexes = similarity.topk(topk)
    print(f'similar words : ')
    #print(indexes)
    for i, indexe in enumerate(indexes): 
        
        print(word_list[i]+': ' ,', '.join([ int2word[idx.item()] for idx in indexe ] )) 


#%% 
# training loop 
batch_size = 512
window_size = 5
epochs = 10
embedding_dim = 300
intervals = 1500
vocab_size = len(int2word)
noise_dist = None
flag=True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_neg_50 = Skipgram_with_NegativeSampling(vocab_size, embedding_dim,noise_dist=noise_dist).to(device)
criterion = NegativeSamplingLoss()
optimizer = torch.optim.Adam(model_neg_50.parameters(), lr=0.003)
for e in range(epochs):

    for i, (words, targets) in enumerate(get_batch(word_list_train,
                                                   batch_size=batch_size,
                                                   window_size=window_size)):
        words = torch.LongTensor(words).to(device)
        targets = torch.LongTensor(targets).to(device)

        input_embeddings = model_neg_50.forward_input(words)
        output_embeddings = model_neg_50.forward_output(targets)
        noise_embeddings = model_neg_50.forward_noise(words.shape[0], 5,device=device)

        if i==0:
            flag=True
        else:
            flag=False

        loss = criterion(input_embeddings, output_embeddings, noise_embeddings,debug=flag)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i% intervals ==0  :          
            validation_words, similarities = cosine_similarity_validation(model_neg_50.input_embedding,
                                                                            validation_size=16,
                                                                            validation_window=100
                                                                            ,device=device)
            # get the top 6 similar vectors 
            data, indexes = similarities.topk(6)
            validation_words = validation_words.to('cpu')
            indexes = indexes.to('cpu')
            print(f'({time.ctime()}) iter/epoch: {i}/{e} loss : {loss.item()}')    
            print(f'indexes: {indexes}')
            for ii, example_idx in enumerate(validation_words): 
                closes_words = [int2word[idx.item()] for idx in indexes[ii]]
                print(f'{int2word[example_idx.item()]}'+ ''.join(f'{closes_words}' ))
                print('...')


model_name = f'model_skipneg_{embedding_dim}_e{epochs}.t'
print(f'saving model: {model_name} to the disk')
states = {'embedding_dim':embedding_dim,
         'batch_size':batch_size,
         'epochs': epochs,
         'state_dict':model_neg_50.state_dict(),
         'int2words':int2word,
         'window_size':window_size}

with open(model_name, 'wb') as f:          
    torch.save(states, f)    
print('done!')


#%%
def load_model(model_name):
   
    states = torch.load(model_name)
    embedding_dim = states['embedding_dim']
    batch_size = states['batch_size']
    epochs = states['epochs']
    int2word= states['int2words']
    window_size = states['window_size']
    model = Skipgram_with_NegativeSampling(len(int2word), embedding_dim=embedding_dim)
    model.load_state_dict(states['state_dict'])
    return model, int2word

#%% 
# load model 
words_list = ['penis', 'woman', 'girl', 'gold', 'iran', 'persia', 'horse','like', 'son','daughter','wife']

model_name = 'model_skipneg_50_e5.t'
model, int2word = load_model(model_name)
get_similar_words(model.cpu(), words_list, int2word)
#%%

model_name = 'model_skipneg_300_e5.t'
model, int2word = load_model(model_name)
get_similar_words(model.cpu(), words_list, int2word)
#%%

model_name = 'model_skipneg_300_e10.t'
model, int2word = load_model(model_name)
print(model)
get_similar_words(model.cpu(), words_list, int2word)

#%%
import matplotlib.pyplot as plt 
#%matplotlib inline
%config InlineBackend.figure_format = 'retina'

from sklearn.manifold import TSNE
# getting the embeddings
embeddings_numpy = model.input_embedding.weight.to('cpu').detach().numpy()

vis_words = 600 
tsne = TSNE()
embed_tsne = tsne.fit_transform(embeddings_numpy[:vis_words, :])

fig, ax = plt.subplots(figsize=(16,16))
ax.set_facecolor("white")
for idx in range(vis_words):
    plt.scatter(*embed_tsne[idx,:], color='blue')
    plt.annotate(int2word[idx], (embed_tsne[idx, 0], embed_tsne[idx, 1]), color='black', alpha=0.9)

# it seems, the more diminesions, need more training!!!

#%%



#%%
