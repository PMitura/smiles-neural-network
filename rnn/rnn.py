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
    model.add(LSTM(32, activation = 'sigmoid', input_shape =
        (None, alphaSize)))
    """
    model.add(AveragePooling1D(pool_length = 2, border_mode='valid'))
    """
    model.add(Dense(1))
    model.compile(loss = 'mse', optimizer = 'rmsprop')

    print('  ...done')
    return model


def train(model, nnInput, refOutput):
    print('  Training model...')
    model.fit(nnInput, refOutput, nb_epoch = 10, batch_size = 16)
    print('  ...done')


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
    print("    prediction average: {}".format(preAvg))
    print("    reference average:  {}".format(refAvg))
    print("    mistake average:    {}".format(misAvg))
    print('  ...done')


def test(model, nnInput, refOutput):
    score = model.evaluate(nnInput, refOutput, batch_size = 16)
    print("Score on traning data: {}".format(score))


def main():
    nnInput, refOutput, alphaSize = data.prepareData()
    model = setup(alphaSize)
    train(model, nnInput, refOutput)
    predict(model, nnInput, refOutput)
    # test(model, nnInput, refOutput)


if __name__ == '__main__':
    main()
