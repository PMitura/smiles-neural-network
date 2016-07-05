import data
import numpy as np
from theano.tensor import *
from keras.models import Sequential
from keras.layers import Activation, Dense, LSTM, AveragePooling1D

# RNN parameters
LAYER_MULTIPLIER = 1
EPOCHS = 500

# 3D input: dimensions are (number of samples, length of word, alphabet)
def setup(alphaSize):
    print('  Initializing and compiling...')

    model = Sequential()
    model.add(LSTM(LAYER_MULTIPLIER * alphaSize, activation = 'sigmoid',
        input_shape = (None, alphaSize), return_sequences = True))
    model.add(LSTM(LAYER_MULTIPLIER * alphaSize, activation = 'sigmoid'))
    """
    model.add(AveragePooling1D(pool_length = 2, border_mode='valid'))
    """
    model.add(Dense(1))
    model.compile(loss = 'mse', optimizer = 'rmsprop')

    print('  ...done')
    return model

def setupInitialized(alphaSize, weights):
    print('  Initializing and compiling...')

    model = Sequential()
    model.add(LSTM(LAYER_MULTIPLIER * alphaSize, activation = 'sigmoid',
        input_shape = (None, alphaSize), return_sequences = True))
    model.add(LSTM(LAYER_MULTIPLIER * alphaSize, activation = 'sigmoid'))
    """
    model.add(AveragePooling1D(pool_length = 2, border_mode='valid'))
    """
    model.add(Dense(1))
    model.compile(loss = 'mse', optimizer = 'rmsprop')
    model.set_weights(weights)

    print('  ...done')
    return model


def train(model, nnInput, refOutput):
    print('  Training model...')
    model.fit(nnInput, refOutput, nb_epoch = EPOCHS, batch_size = 16)
    print('    Model weights:')
    # print(model.summary())
    # print(model.get_weights())
    print('  ...done')
    return model.get_weights()


def predict(model, nnInput, refOutput):
    print('  Predicting...')
    pre = model.predict(nnInput, batch_size = 16)
    preAvg = 0.0
    refAvg = 0.0
    misAvg = 0.0
    for i in range(len(pre)):
        preAvg += pre[i][0]
        refAvg += refOutput[i]
        misAvg += abs(pre[i][0] - refOutput[i])
        print("    prediction: {}, reference: {}".format(pre[i][0], 
            refOutput[i]))
    preAvg /= len(pre)
    refAvg /= len(pre)
    misAvg /= len(pre)
    print("    prediction average: {}".format(preAvg1))
    print("    reference average:  {}".format(refAvg1))
    print("    mistake average:    {}".format(misAvg1))
    print('  ...done')


def test(model, nnInput, refOutput):
    # should work, untested
    print("  Testing...")
    score = model.evaluate(nnInput, refOutput, batch_size = 16)
    print("    Score on traning data: {}".format(score))
    print("  ...done")


def run():
    nnInput, ref1, ref2, alphaSize = data.prepareData()
    model1 = setup(alphaSize)
    weights = train(model1, nnInput, ref1)
    model2 = setupInitialized(alphaSize, weights)
    train(model2, nnInput, ref2)
    predict(model1, nnInput, ref1)
    predict(model2, nnInput, ref2)
    # test(model, nnInput, refOutput)
