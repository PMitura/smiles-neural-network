import data
import numpy as np
from theano.tensor import *
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, LSTM, AveragePooling1D
from keras.layers import TimeDistributed, SimpleRNN, GRU
from keras.optimizers import RMSprop

# RNN parameters
LAYER_MULTIPLIER = 1
EPOCHS = 150
BATCH = 16
RANDOM_SAMPLES = 20
PREDICT_PRINT_SAMPLES = 10
HOLDOUT_RATIO = 0.8
LEARNING_RATE = 0.01


def configureModel(alphaSize):
    print('  Initializing and compiling...')

    model = Sequential()
    """
    model.add(LSTM(LAYER_MULTIPLIER * alphaSize, activation = 'sigmoid',
        input_shape = (None, alphaSize), return_sequences = True))
    """
    model.add(TimeDistributed(Dense(LAYER_MULTIPLIER * alphaSize),
        input_shape = (None, alphaSize)))
    # model.add(TimeDistributed(Dense(LAYER_MULTIPLIER * alphaSize)))
    model.add(GRU(LAYER_MULTIPLIER * alphaSize, activation = 'sigmoid'))
    # model.add(SimpleRNN(2 * LAYER_MULTIPLIER * alphaSize, activation = 'sigmoid'))
    # model.add(AveragePooling1D(pool_length = 2, border_mode='valid'))
    # model.add(Dropout(0.5))
    model.add(Dense(1))
    # model.add(Activation('tanh'))

    # default learning rate 0.001
    model.compile(loss = 'mse', optimizer = RMSprop(lr = LEARNING_RATE))

    print('  ...done')
    return model


# 3D input: dimensions are (number of samples, length of word, alphabet)
def setup(alphaSize):
    return configureModel(alphaSize)


def setupInitialized(alphaSize, weights):
    print('  Initializing and compiling...')

    model = configureModel(alphaSize)
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
    pre = model.predict(nnInput, batch_size = BATCH)
    preAvg = 0.0
    refAvg = 0.0
    misAvg = 0.0
    for i in range(len(pre)):
        preAvg += pre[i][0]
        refAvg += refOutput[i]
        misAvg += abs(pre[i][0] - refOutput[i])

    for i in range(PREDICT_PRINT_SAMPLES):
        print("    prediction: {}, reference: {}".format(pre[i][0], 
            refOutput[i]))

    preAvg /= len(pre)
    refAvg /= len(pre)
    misAvg /= len(pre)
    print("    prediction average: {}".format(preAvg))
    print("    reference average:  {}".format(refAvg))
    print("    mistake average:    {}".format(misAvg))


def test(model, nnInput, refOutput):
    # should work, untested
    print("  Scoring using evaluate...")
    score = model.evaluate(nnInput, refOutput, batch_size = BATCH,
            verbose = False)
    print("    Score on traning data: {}".format(score))
    print("  ...done")


def run():
    # Initialize using same seed (to get stable results on comparisons)
    np.random.seed(12345)
    fullIn, labelWeight, labelALogP, alphaSize = data.prepareData()

    """ In case a subset is wanted
    nnInput, ref2 = data.randomSelection(RANDOM_SAMPLES, nnInput, ref2)
    """

    """ Single model setup
    trainIn, trainLabel, testIn, testLabel = data.holdout(HOLDOUT_RATIO,
            nnInput, labelALogP)
    model = setup(alphaSize)
    train(model, trainIn, trainLabel)
    print("  \nPrediction of training data:")
    predict(model, trainIn, trainLabel)
    print("  \nPrediction of testing data:")
    predict(model, testIn, testLabel)
    test(model, testIn, testLabel)
    """

    """ Chained models setup """
    modelWeights = setup(alphaSize)
    chainSetup = train(modelWeights, fullIn, labelWeight)
    predict(modelWeights, fullIn, labelWeight)
    modelAlogP = setupInitialized(alphaSize, chainSetup)

    trainIn, trainLabel, testIn, testLabel = data.holdout(HOLDOUT_RATIO,
            fullIn, labelALogP)
    train(modelAlogP, trainIn, trainLabel)
    print("  \nPrediction of training data:")
    predict(modelAlogP, trainIn, trainLabel)
    print("  \nPrediction of testing data:")
    predict(modelAlogP, testIn, testLabel)
    test(modelAlogP, testIn, testLabel)
