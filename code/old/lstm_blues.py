'''

The LSTM Blues
   a lyrical generator

as ammended 
by James King

W266-Fall 2016
Berkeley/iSchool/MIDS


* Original code stolen from the keras example library:
https://github.com/fchollet/keras/tree/master/examples

with additional ideas adapted from 
https://github.com/fchollet/keras/blob/master/examples/babi_rnn.py
'''

from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop, SGD
from keras.utils.data_utils import get_file
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding
import numpy as np
import random
import sys


NUM_ITERATIONS = 25

#path = '/data/W266/data/songsTextAll.txt'
path = '/data/W266/data/songsAndHHG.txt'
text = open(path).read().lower().split(' ') # splitting on space keeps other whitespace, which is good

print('corpus length:', len(text), 'words.')

#***
vocab = sorted(list(set(text)))
print('Total vocabulary size: ', len(vocab), ' words.')
# Reserve 0 for masking via pad_sequences
vocab_size = len(vocab)
word_idx = dict((c, i) for i, c in enumerate(vocab))
idx_word = dict((i, c) for i, c in enumerate(vocab))
#***

# cut the text in semi-redundant sequences of maxlen words
maxlen = 8 # text is chopped up into chunks this big
step = 5    # amount of overlap in chunks

sentences = []  # a chunk of maxlen -- "context"
next_words = [] # the single word following the context

for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])   
    next_words.append(text[i + maxlen])

print('nb sequences:', len(sentences))

print('Vectorization...') # converting data to make it suitable for matrix math

# Set up a matrix of all possibilities.  Label a 1 for each
# possibility observed in the corpus
X = np.zeros((len(sentences), maxlen, len(vocab)), dtype=np.bool)
y = np.zeros((len(sentences), len(vocab)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, word in enumerate(sentence):
        X[i, t, word_idx[word]] = 1
    y[i, word_idx[next_words[i]]] = 1


# build the model
print('Building model...')
model = Sequential()
model.add(Embedding(vocab_size+1, 50))
# First argument is the output dimension
model.add(LSTM(50, input_shape=(maxlen, len(vocab))))
model.add(Dropout(0.1))
model.add(Dense(len(vocab)))
model.add(Activation('softmax'))



#optimizer = RMSprop(lr=0.02)
optimizer = SGD(lr=0.2)

model.compile(loss='categorical_crossentropy', optimizer=optimizer)

######################################################################

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# train the model, output generated text after each iteration
#saveModelFile = '/data/W266/data/trainedModel.keras'
saveModelFile = 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'

for iteration in range(1, NUM_ITERATIONS):  
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(X, y, batch_size=128, nb_epoch=1)

    start_index = random.randint(0, len(text) - maxlen - 1)

    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print()
        print('----- diversity:', diversity)

        generated = []
        sentence = text[start_index: start_index + maxlen]
        generated += sentence
        print('----- Generating with seed: "' + " ".join(sentence) + '"')
        sys.stdout.write(" ".join(generated))

        for i in range(40):
            x = np.zeros((1, maxlen, len(vocab)))
            for t, word in enumerate(sentence):
                x[0, t, word_idx[word]] = 1.

            preds = model.predict(x, verbose=0)[0] #"preds" = "predictions"
            next_index = sample(preds, diversity)
            next_word = idx_word[next_index]

            generated += [next_word]
            sentence = sentence[1:] + [next_word]

            sys.stdout.write(" " + next_word + " ")
            sys.stdout.flush()
        print()

    # Save the model after each iteration
    print("\nWriting model to file: "+saveModelFile)

    ModelCheckpoint(saveModelFile, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto')




