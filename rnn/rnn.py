import data
import numpy as np
from theano.tensor import *
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, LSTM, AveragePooling1D
from keras.layers import TimeDistributed, SimpleRNN, GRU

# RNN parameters
LAYER_MULTIPLIER = 1
EPOCHS = 1500
BATCH = 16
RANDOM_SAMPLES = 20


# 3D input: dimensions are (number of samples, length of word, alphabet)
def setup(alphaSize):
    print('  Initializing and compiling...')

    model = Sequential()
    """
    model.add(LSTM(LAYER_MULTIPLIER * alphaSize, activation = 'sigmoid',
        input_shape = (None, alphaSize), return_sequences = True))
    """
    model.add(TimeDistributed(Dense(LAYER_MULTIPLIER * alphaSize),
        input_shape = (None, alphaSize)))
    model.add(TimeDistributed(Dense(LAYER_MULTIPLIER * alphaSize)))
    model.add(GRU(LAYER_MULTIPLIER * alphaSize, activation = 'sigmoid'))
    # model.add(SimpleRNN(2 * LAYER_MULTIPLIER * alphaSize, activation = 'sigmoid'))
    # model.add(AveragePooling1D(pool_length = 2, border_mode='valid'))
    # model.add(Dropout(0.5))
    model.add(Dense(1))
    # model.add(Activation('tanh'))
    model.compile(loss = 'mse', optimizer = 'rmsprop')

    print('  ...done')
    return model


def setupInitialized(alphaSize, weights):
    print('  Initializing and compiling...')

    model = Sequential()
    model.add(LSTM(LAYER_MULTIPLIER * alphaSize, activation = 'sigmoid',
        input_shape = (None, alphaSize), return_sequences = True))
    model.add(LSTM(LAYER_MULTIPLIER * alphaSize, activation = 'sigmoid'))
    model.add(Dense(1))
    model.compile(loss = 'mse', optimizer = 'rmsprop')
    model.set_weights(weights)

    print('  ...done')
    return model


def train(model, nnInput, refOutput):
    print('  Training model...')
    model.fit(nnInput, refOutput, nb_epoch = EPOCHS, batch_size = BATCH)
    print('    Model weights:')
    # print(model.summary())
    # print(model.get_weights())
    print('  ...done')
    return model.get_weights()


def predict(model, nnInput, refOutput):
    print('  Predicting...')
    pre = model.predict(nnInput, batch_size = BATCH)
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
    print("    prediction average: {}".format(preAvg))
    print("    reference average:  {}".format(refAvg))
    print("    mistake average:    {}".format(misAvg))
    print('  ...done')


def test(model, nnInput, refOutput):
    # should work, untested
    print("  Testing...")
    score = model.evaluate(nnInput, refOutput, batch_size = BATCH)
    print("    Score on traning data: {}".format(score))
    print("  ...done")


def run():
    # Initialize using same seed (to get stable results on comparisons)
    np.random.seed(12345)
    nnInput, ref1, ref2, alphaSize = data.prepareData()

    """ Chained models setup
    model1 = setup(alphaSize)
    weights = train(model1, nnInput, ref1)
    model2 = setupInitialized(alphaSize, weights)
    train(model2, nnInput, ref2)
    predict(model1, nnInput, ref1)
    predict(model2, nnInput, ref2)
    """

    """ Single model setup """
    nnInput, ref2 = data.randomSelection(RANDOM_SAMPLES, nnInput, ref2)
    model = setup(alphaSize)
    train(model, nnInput, ref2)
    predict(model, nnInput, ref2)
    # test(model, nnInput, refOutput)

