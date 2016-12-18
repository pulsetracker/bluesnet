#LSTM Blues Generator

import w266Final as ww
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

modelLoc = '/data/W266/results/OneLayer'
filename = 'trainedModel-7.hdf5'
model = load_model(os.path.join(modelLoc,filename);