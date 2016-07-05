import data
import numpy as np
from theano.tensor import *
from keras.models import Sequential
from keras.layers import Activation, Dense, LSTM, AveragePooling1D

# RNN parameters
MAX_BACKWARD_STEPS = 100

# 3D input: dimensions are (number of samples, length of word, alphabet)
def setup(alphaSize):
    print('  Compiling model...')

    model = Sequential()
    model.add(LSTM(alphaSize, activation = 'sigmoid',
        input_shape = (None, alphaSize), return_sequences = True))
    model.add(LSTM(alphaSize, activation = 'sigmoid'))
    """
    model.add(AveragePooling1D(pool_length = 2, border_mode='valid'))
    """
    model.add(Dense(2))
    model.compile(loss = 'mse', optimizer = 'rmsprop')

    print('  ...done')
    return model


def train(model, nnInput, refOutput):
    print('  Training model...')
    model.fit(nnInput, refOutput, nb_epoch = 500, batch_size = 16)
    print('    Model weights:')
    # print(model.summary())
    # print(model.get_weights())
    print('  ...done')


def predict(model, nnInput, refOutput):
    print('  Predicting...')
    pre = model.predict(nnInput, batch_size = 16)
    preAvg1 = 0.0
    refAvg1 = 0.0
    misAvg1 = 0.0
    preAvg2 = 0.0
    refAvg2 = 0.0
    misAvg2 = 0.0
    for i in range(len(pre)):
        preAvg1 += pre[i][0]
        refAvg1 += refOutput[i][0]
        misAvg1 += abs(pre[i][0] - refOutput[i][0])
        preAvg2 += pre[i][1]
        refAvg2 += refOutput[i][1]
        misAvg2 += abs(pre[i][1] - refOutput[i][1])
        print("    prediction1: {}, reference1: {}".format(pre[i][0], 
            refOutput[i][0]))
        print("    prediction2: {}, reference2: {}".format(pre[i][1], 
            refOutput[i][1]))
    preAvg1 /= len(pre)
    refAvg1 /= len(pre)
    misAvg1 /= len(pre)
    preAvg2 /= len(pre)
    refAvg2 /= len(pre)
    misAvg2 /= len(pre)
    print("    prediction1 average: {} | prediction2 average: {}"
            .format(preAvg1, preAvg2))
    print("    reference1 average:  {} | reference2 average: {}"
            .format(refAvg1, refAvg2))
    print("    mistake1 average:    {} | mistake2 average:{}"
            .format(misAvg1, refAvg2))
    print('  ...done')


def test(model, nnInput, refOutput):
    # should work, untested
    print("  Testing...")
    score = model.evaluate(nnInput, refOutput, batch_size = 16)
    print("    Score on traning data: {}".format(score))
    print("  ...done")


def run():
    nnInput, refOutput, alphaSize = data.prepareData()
    model = setup(alphaSize)
    train(model, nnInput, refOutput)
    predict(model, nnInput, refOutput)
    # test(model, nnInput, refOutput)
