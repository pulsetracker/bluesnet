'''

The LSTM Blues
   a lyrical generator

as ammended 
by James King

W266-Fall 2016
Berkeley/iSchool/MIDS


* Original code stolen from the keras example library:
https://github.com/fchollet/keras/tree/master/examples

'''

from __future__ import print_function
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop, SGD
from keras.utils.data_utils import get_file
from keras.callbacks import ModelCheckpoint
import numpy as np
import random
import sys
import os
import os.path

pathRoot = '/data/W266/data/'
path = pathRoot + 'songsTextAll.txt'
#path = '/data/W266/data/songsAndHHG.txt'
#path = '/data/W266/data/songsAndUHHG.txt'


text = open(path).read().lower()
print('corpus length:', len(text))

chars = sorted(list(set(text))) # All berthe characters from the corpus
print('Total number of Characters in Corpus:', len(chars))

# Set up dictionaries mapping each character to its
# index (and vice versa) in the chars set
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 100 # text is chopped up into chunks this big
step = 3    # amount of overlap in chunks
numIters = 5
learning_rate = 0.2  # The learning rate for the model optimizer

sentences = []  # a chunk of maxlen -- "context"
next_chars = [] # the single character following the context

for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])   
    next_chars.append(text[i + maxlen])

print('nb sequences:', len(sentences))

print('Vectorization...') # converting data to make it suitable for matrix math

# Set up a matrix of all possibilities.  Label a 1 for each
# possibility observed in the corpus
X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

dataFiles = os.listdir(pathRoot)
trainedModels = [fn for fn in dataFiles if fn.startswith('trainedModel-')]

if len(trainedModels) > 0:
    # If there's already a trained model, pick up where it left off
    print('Previous models found, continuing from last location.')
    trainedModels.sort()
    newestFile = trainedModels[-1]
    model = load_model(os.path.join(pathRoot,newestFile))
    firstIterlNum = int(newestFile.split('.')[0].split('-')[1]) + 1

else:
    # Start from scratch and
    # build the model
    print('Building model...')
    model = Sequential()

    # 128 is the output dimension
    model.add(LSTM(64, input_shape=(maxlen, len(chars))))
    model.add(Dense(len(chars)))
    model.add(Activation('softmax'))
    model.add(Dense(len(chars)))
    model.add(Activation('softmax'))
    # model.add(Dense(len(chars)))
    # model.add(Activation('softmax'))

    optimizer = RMSprop(lr=learning_rate)
    #optimizer = SGD(lr=learning_rate)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    firstIterlNum = 1

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
#saveModelFile = 'bluesweights.{epoch:02d}-{val_loss:.2f}.hdf5'

for iteration in range(firstIterlNum, firstIterlNum+numIters):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(X, y, batch_size=128, nb_epoch=1)

    start_index = random.randint(0, len(text) - maxlen - 1)

    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print()
        print('----- diversity:', diversity)

        generated = ''
        sentence = text[start_index: start_index + maxlen]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(400):
            x = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x[0, t, char_indices[char]] = 1.

            preds = model.predict(x, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()

    # Save the model after each iteration
    outFileName = '/data/W266/data/trainedModel-'+str(iteration)+'.hdf5'
    print("Writing model to file: "+outFileName)
    model.save(outFileName)
#)
    #ModelCheckpoint(saveModelFile, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto')




