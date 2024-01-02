#%% in this section lets do sentiment analysis, that is to look at a given text
# and determine whether its positive or negative.
# since it deals with text, we will be using an rnn network, we can use autoregressive
# models such as transformers(which are the standards now) but for now lets stick to 
# our good old lstm!
# lets import our needed modules
import math
import numpy as np 
import torch 
import torch.nn as nn 
# for plotting
import matplotlib.pyplot as plt 
# for punctuation related chores, like removing punctuations, etc
import string
from tqdm import tqdm

# we will be using a dataset for reviews, which is bunch of comments with a label in a separate file
# which specifies each comment as positive or negative.
# since we are dealing with sentiment analysis and not text generation, we do a few steps differently
# like, we build our model arounds words this time and not characters, lets see it in action 
comments_raw = open('/media/hossein/SSD1/code_dl/reviews.txt').read().splitlines()
labels_raw =  open('/media/hossein/SSD1/code_dl/labels.txt').read().lower().splitlines()
print(f'{len(comments_raw)=:,}') #25,000
print(f'{len(labels_raw)=:,}')   #25000,
# so we've got 25k comments 
# lets read afew
for comment,label in zip(comments_raw[:3], labels_raw[:3]):
    print(f'{comment:.35}... is {label}')
# now lets do some normalization on our datasets, so that its simpler and the model find it easier
# to optimize, work around. we can start by removing the punctuations
# lets use translate. it requires a dictionary of what to lookfor and replace with what! we make
# that dictionary using str.maketrans. giving two empty strings, followed by a list of characters
# gives us a dictionary that replaces all punctuations with None!
# the two first strings are empty, if they werent, it meant, replace each character from one string
# with the ones in the other strings. the third argument, is for all the characters we want to remove
# so since we dont need the first two arguments (cuz we dont want to do mapping), and only want to
# remove some characters, we simply only use the last argument and give the first two arguments, empty
# strings.
# so basically str.maketrans() takes up to three arguments: two strings of equal length, and one string
# containing characters to be removed from the original string.
# the first two arguments are used to create a mapping of characters: each character in the first string
# is replaced by the character at the same position in the second string.
# The third argument is a string containing characters to be removed from the original string.
# also the resulting dictionary contains character numerical codes. 
print(str.maketrans('','',string.punctuation))
# prints 
# {33: None, 34: None, 35: None, 36: None, 37: None, 38: None, 39: None, 40: None, 41: None, 42: None, 43: None, 44: None, 45: None, 46: None, 47: None, 58: None, 59: None, 60: None, 61: None, 62: None, 63: None, 64: None, 91: None, 92: None, 93: None, 94: None, 95: None, 96: None, 123: None, 124: None, 125: None, 126: None}
# ok, since only comments have punctuations, we only normalize comments.
comments_new = [cmnt.translate(str.maketrans('','',string.punctuation)) for cmnt in comments_raw]
print(f'{comments_raw[:3]}')
print(f'{comments_new[:3]}')
# since we are not after text generation, we can split our vocabulary based on words.(tokenize it)
# after all we want to train a language model as part of our model, and we want to convert our input from text
# to numbers and we either do it on character level or word level etc. 
# each have their pros and cons like : 
# in word level encoding we have lower computational cost as the model deals with fewer tokens 
# (words) compared to character-level encoding(sequence length is much smaller when using words, compared to characters).
# while in character level encoding the model can handle out-of-vocabulary words, becasue it can 
# generate any words as it operates at character level.
# but since we dont want to generate text, word level should be fine and also it imposes less overhead.
# note that wordlevel encoding is not generally used much as of late, instead subword encodings are 
# used in practice to cover word-level encodings shortcommings.
# (Byte-Pair Encoding(BPE) and WordPiece are two common subword encoding methods, and sentencepiece 
# and tiktoken are two common tokeniziers based on BPE for example that are used extensivly in state-of-the-art
# models for example (chatgpt uses tiktoken for example))
# but for the sake of simplicity we use word-level encoding for the tokenization process here.
# we can go on ahead and split all the words, lets do just that
# note that we are using split() instead of split(' '). split() with no argument, splits based on
# all whitespace characters, so ' ', '\n',etc are all included and will be used to split the sentences,etc
all_words = ''.join(comments_new).split()
print(f'{len(all_words)=:,}')
# there are 6,020,196 words! but using set and saving the unique ones
words = set(all_words)
# and we get 74,072 unique ones!
print(f'{len(words)=}')
print(f'{words=}')
# before we go ahead and create our dictionaries, 
# lets have some insight about our dataset and see how many of each word we have, 
# we use collections.Counter to get that information 
from collections import Counter
word_counts = Counter(all_words)
print(f'{word_counts=}')
# lets see the count for 'the', 
print(f'{word_counts["bromwell"]=}')
# lets sort them based on ascending order 
# note that sorted(word_counts) also works, but it sorts using the keys, not values,
# remember that if we simply iterate over a dictionary, we get the keys only, but since 
# we want the values, we instruct the sorted() to use the .get() method which returns the value
# for a given key, it iteates the dictionary sending each key to this function, and the function
# gives us back the values, which is then used to do the sorting. 
# if we dont use .get(), it sorts the dictionary based on alphabetical order.
print(f'{sorted(word_counts, key=word_counts.get)}')
# ok, now lets see whats the most frequently used word and the least frequently used word
most_used = max(word_counts,key=word_counts.get)
least_used = min(word_counts,key=word_counts.get)
print(f"most frequently used word: '{most_used}' occures {word_counts[most_used]:,} times")
print(f"least frequently used word: '{least_used}' occures {word_counts[least_used]:,} time(s)")
# so lets go on ahead and create our dictionaries word2int and int2word
# we start from 1, because we intend on using the 0 for padding our inputs later on
itow = dict(enumerate(words,1))
wtoi = {w:i for i,w in itow.items()}
# now lets create a simple function to convert betwene them
conver_to_int = lambda word_sequence : [wtoi[w] for w in word_sequence.split()]
conver_to_word = lambda int_iterable : [itow[w] for w in int_iterable]

print(conver_to_int(comments_new[0]))
print(' '.join(conver_to_word((conver_to_int(comments_new[0])))))
#%%
# OK, now we have our itow and wtoi dictionaries, so lets 
# the next step is to create a batching mechanism, here we
# will write a function that grabs comments, and creates batches
# of a fixed length, since not all comments have the smae length,
# some are bound to be shorter than others, so we pad the shorter 
# ones with 0s so each sequence sample has the same length
# to choose the right length, lets first have a clue about the min/max length
# for this we check the whole comments, take convert them to int sequences and
# then check the min, max 
comments_digitized = [conver_to_int(sequence) for sequence in comments_new]
# now lets convert our labels as well 
labels_digitized = [1 if l =='positive' else 0 for l in labels_raw]
# lets inspect it 
print(*comments_digitized[:2],sep='\n')
print(*[" ".join(conver_to_word(cmnt)) for cmnt in comments_digitized[:2]], sep='\n')
print(labels_digitized[:2])
print(*['positive' if l is 1 else 'negative' for l in labels_digitized[:2]])
# good now lets get the max, min seq length in our dataset 
max_len = max([len(cmnt) for cmnt in comments_digitized])
min_len = min([len(cmnt) for cmnt in comments_digitized])
# now lets grab all the sequences iwth their lengths 
seq_len_dict = Counter([len(cmnt) for cmnt in comments_digitized])
print(f'{max_len=}')
print(f'{min_len=}')
print(f'{seq_len_dict[10]}')
# lets see the most common lengths: 
# so we have 185 sequences with the length of 132
print(f'{seq_len_dict.most_common(3)=}')
# lets see whats the index of the comment with the min length: (note the , after idx
# I did that to unpack the list into idx as I know for a fact that theres only 1 item in the list) 
# idxs, = [idx for idx,cmnt in enumerate(comments_digitized) if len(cmnt)== min_len]
# but lets not do that and write it so if we wanted to change min_len to sth else that
# has more sequences, we dont face errors
idxs = [idx for idx,cmnt in enumerate(comments_digitized) if len(cmnt)== min_len]
print(f'{idxs=}')
print(f'cmnt with min len: {idxs=} {" ".join(conver_to_word(comments_digitized[idxs[0]]))}')
# anyway, all seems fine, we have a single 10 word review, 
# and bunch of others 1 (to seem them call most_common() with no argument)
# lets leave it as is 
# we dont really need to bother much here! just check the min/max and see 
# if later on it messed things up, remove these 
# for example if we happen to want to remove these, we simply remove the comments adn t heir labels likethis
# comments_digitized = [cmnt for idx, cmnt in enumerate(comments_digitized) if idx not in idxs]
# do the same for labels
# labels_raw = [lbl for idx, lbl in enumerate(labels_raw) if idx not in idxs]
# print(f'{len(comments_digitized)}') #24999 instead of 25k
# print(f'{len(labels_raw)}')         #24999 instead of 25k
#%%
# ok we have our comments and labels, lets create our padding function 
# we pad our whole dataset
def pad_input(input_dataset, max_len=200, pad_right=True):
    # create a zero array with the desired size, 
    # start filing it with what we have, leave t
    # the rest with zero! 
    arr = np.zeros(shape=(len(input_dataset), max_len), dtype=np.int32)
    for i in range(len(input_dataset)):
        seq_len = len(input_dataset[i])
        # we have two options, to pad the begining or the end of the sequence
        # the padding on the begining is called pre-padding and the padding on
        # the end of the sequence is post-padding. which one to use depends on 
        # our usecase.
        if pad_right:
            arr[i,:seq_len] = input_dataset[i][:max_len]
        else:
            arr[i,-seq_len:] = input_dataset[i][:max_len]
    return arr 
# note that choosing a high max length where the majority of sequences are not that long
# will result in very bad result, as the sequence will be filled with more zeros than 
# the actual data, or it will contain just more zeros that make the model struggle 
comments_digitized = pad_input(comments_digitized,max_len=100,pad_right=1)
print(f'{comments_digitized[0]}')

# now its time to create our dataset. since our data is in numpy format
# we use pytorch tensor dataset 
# before that lets split our data into training/val/test sets 
# lets convert labels to numpy 
labels_digitized = np.array(labels_digitized)
traing_ratio = 0.8
training_length = int(0.8*len(comments_digitized))
trainig_data = comments_digitized[:training_length]
trainig_label = labels_digitized[:training_length]
#val,test
remaining_data = comments_digitized[training_length:]
remaining_label = labels_digitized[training_length:]
split_ratio = 0.5
split_idx = int(split_ratio*len(remaining_data))
val_data = remaining_data[:split_idx]
test_data = remaining_data[split_idx:]
val_label = remaining_label[:split_idx]
test_label = remaining_label[split_idx:]

# lets review 
print(f'{trainig_data.shape=}')
print(f'{trainig_label.shape=}')
print(f'{val_data.shape=}')
print(f'{val_label.shape=}')
print(f'{test_data.shape=}')
print(f'{test_label.shape=}')
#%%
# now lets create our dataset and then dataloaders , since we are using numpy arrays, we need to use
# tensordataset, and for that we need to convert our numpy arrays to torch tensors.
from torch.utils import data
training_dataset = data.TensorDataset(torch.from_numpy(trainig_data), torch.from_numpy(trainig_label))
val_dataset = data.TensorDataset(torch.from_numpy(val_data), torch.from_numpy(val_label))
test_dataset = data.TensorDataset(torch.from_numpy(test_data), torch.from_numpy(test_label))

batch_size = 128
num_worker = 8
training_dataloader = data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True, pin_memory=True,num_workers=num_worker)
val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, pin_memory=True,num_workers=num_worker)
test_dataloader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True,num_workers=num_worker)


features, labels = next(iter(training_dataloader))
print(features)
print(labels)
#%% 
# now lets implement our model 
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_size=200, embd_size=130,num_layers=1, dropout_ratio=0.0, bidirectional=False) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embd_size = embd_size
        self.hidden_size = hidden_size
        self.output_size = 1 # becasue we want to know if its positive or not
        self.num_layers = num_layers
        self.bidirection = bidirectional
        self.direction = 2 if self.bidirection else 1 
        self.drpout_ratio = dropout_ratio
        
        self.embd = nn.Embedding(vocab_size, embedding_dim=embd_size)
        self.rnn = nn.LSTM(embd_size,  
                           hidden_size=hidden_size,
                           num_layers= num_layers, 
                           dropout=dropout_ratio,
                           batch_first=True,
                           bidirectional = bidirectional
                           )
        self.fc = nn.Linear(self.hidden_size*self.direction, 1)
        
    def forward(self, input, hidden_size):
        embd = self.embd(input)
        outputs, final_hiddenstate = self.rnn(embd, hidden_size)
        # lets use the outputs, reshape it so the fc doesnt complain 
        output = self.fc(outputs.reshape(-1, self.hidden_size*self.direction))
        # becasue we want a 0 or 1 at the end so we use sigmoid, reshape to its original form
        output =output.sigmoid().view(input.size(0),-1)
        # since we want the last sequence (as its what we are after!)
        return output[:,-1], final_hiddenstate

# lets test it 
print(f'{len(wtoi)}')
model = SentimentLSTM(len(wtoi),hidden_size=100, embd_size=20, num_layers=1, bidirectional=True)
output = model(features.long(),None)
print(f'{output[0].shape}')
#%%
print(f'{len(training_dataloader)=} {len(val_dataloader)=}') 
#%%
# ok we make our model , now lets train it 
vocab_size = len(wtoi)+1 # becasue of 0 
embd_size = 300
hidden_size = 300
num_layers = 2
drp = 0.1 # only applies if we have more than 2 lstm layers
bidirectional = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'
epoches = 30
interval=312
model = SentimentLSTM(vocab_size, 
                      hidden_size, 
                      embd_size, 
                      num_layers=num_layers,
                      dropout_ratio=drp, 
                      bidirectional=bidirectional)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=10, gamma=0.1)
criterion = nn.BCELoss()


print(f'{device=}')
print(f'{epoches=}')
print(f'{num_layers=}')
print(f'{embd_size=}')
print(f'{hidden_size=}')

model.to(device)
for epoch in range(epoches):
    hidden_state = None 
    losses = []
    model.train()
    accs = []
    for i, (data, labels) in tqdm(enumerate(training_dataloader)):
        
        data, labels = (t.to(device) for t in (data,labels))
        if hidden_state and hidden_state[0].size(0)!=data.size(0):
                hidden_state =None
        output, hidden_state = model(data, hidden_state)
        hidden_state = tuple(h.detach() for h in hidden_state)
        # now lets calculate loss
        labels = labels.view(*output.shape).float()
        loss = criterion(output, labels)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        # calculate loss and accuracy
        losses.append(loss.item())
        accs.append(((output>0.5).float()== labels).float().mean().item())
        
    # calculate train_accuracy and loss 
    train_acc = np.mean(accs)
    train_loss = np.mean(losses)
    # after each epoch update lr
    scheduler.step()
    # run validation 
    with torch.no_grad():
        model.eval()
        hidden_state = None
        accs = []
        losses=[]
        for data, labels in tqdm(val_dataloader):
            data, labels = (t.to(device) for t in (data,labels))
            if hidden_state and hidden_state[0].size(0)!=data.size(0):
                hidden_state =None
            output, hidden_state = model(data, hidden_state)
            hidden_state = tuple(h.detach() for h in hidden_state)
            # now lets calculate loss and accuracy 
            labels = labels.view(*output.shape).float()
            losses.append(criterion(output, labels).item())
            accs.append(torch.eq((output>0.5).float(), labels).float().mean().item())
            
        val_acc = np.mean(accs)
        val_loss = np.mean(losses)
    print(f'{epoch+1}/{epoches}) train-loss: {train_loss:.4f} train-accuacy: {train_acc:.2f} val loss: {val_loss:.4f} val-acc: {val_acc:.2f}')

# now lets test this on the testset 
with torch.no_grad():
    model.eval()
    hidden_state = None
    accs = []
    losses=[]
    for data, labels in tqdm(val_dataloader):
        data, labels = (t.to(device) for t in (data,labels))
        if hidden_state and hidden_state[0].size(0)!=data.size(0):
            hidden_state =None
        output, hidden_state = model(data, hidden_state)
        hidden_state = tuple(h.detach() for h in hidden_state)
        # now lets calculate loss and accuracy 
        labels = labels.view(*output.shape).float()
        losses.append(criterion(output, labels).item())
        accs.append(((output>0.5).float()== labels).float().mean().item())
        
    test_acc = np.mean(accs)
    test_loss = np.mean(losses)
print(f'train-loss: {train_loss:.4f} train-accuacy: {train_acc:.2f} test-loss: {test_loss:.4f} test-acc: {test_acc:.2f}')
#%% ok now lets use some text and see if it works properly 
def classify_text(input="damn it, it was aweful!"):
    print(f'text: {input}')
    # first lets tokenize our input text and convert them into digits 
    # before that lets normalize our input, lets remove all the punctuations
    input = input.translate(str.maketrans('','',string.punctuation)).lower()
    sequence = conver_to_int(input)
    # now lets padd it and then feed it to our model
    sequence_padded = pad_input(np.array([sequence]),130)
    # convert to tensor 
    sequence_tensor = torch.from_numpy(sequence_padded).to(next(model.parameters()).device)
    # feed into the model 
    output,_ = model(sequence_tensor,None)
    print("output: ","positive" if output[0]>0.5 else 'negative')

classify_text('it was awefully wonderfully badly good')
