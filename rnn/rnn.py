import data, utility
import numpy as np

from math import sqrt, exp

from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, LSTM, AveragePooling1D
from keras.layers import TimeDistributed, SimpleRNN, GRU, Flatten
from keras.optimizers import Adam, RMSprop
from keras.regularizers import l1l2, l2

# RNN parameters
LABEL_IDX = 0
LAYER_MULTIPLIER = 1
EPOCHS = 150
BATCH = 16
RANDOM_SAMPLES = 20
PREDICT_PRINT_SAMPLES = 10
HOLDOUT_RATIO = 0.8
LEARNING_RATE = 0.01
ZSCORE_NORM = False
LOGARITHM = False


def configureModel(alphaSize):
    print('  Initializing and compiling...')

    model = Sequential()
    """
    model.add(LSTM(LAYER_MULTIPLIER * alphaSize, activation = 'sigmoid',
        input_shape = (None, alphaSize), return_sequences = True))
    """
    model.add(TimeDistributed(Dense(int(LAYER_MULTIPLIER * alphaSize)),
        input_shape = (None, alphaSize)))
    # model.add(TimeDistributed(Dense(LAYER_MULTIPLIER * alphaSize)))
    model.add(GRU(int(LAYER_MULTIPLIER * alphaSize), activation = 'relu'))
    # model.add(SimpleRNN(2 * LAYER_MULTIPLIER * alphaSize, activation = 'sigmoid'))
    # model.add(AveragePooling1D(pool_length = alphaSize, border_mode='valid'))
    # model.add(Dropout(0.5))
    model.add(Dense(1))
    # model.add(Activation('tanh'))

    # default learning rate 0.001
    model.compile(loss = 'mse', optimizer = Adam(lr = LEARNING_RATE))

    print('  ...done')
    return model


# 3D input: dimensions are (number of samples, length of word, alphabet)
def setup(alphaSize):
    return configureModel(alphaSize)


def setupInitialized(alphaSize, weights):
    model = configureModel(alphaSize)
    model.set_weights(weights)
    return model


def train(model, nnInput, refOutput):
    print('  Training model...')
    model.fit(nnInput, refOutput, nb_epoch = EPOCHS, batch_size = BATCH)
    print('    Model weights:')
    print(model.summary())
    # print(model.get_weights())
    print('  ...done')
    return model.get_weights()


# Serves as extended version of test, gives averages
def predict(model, nnInput, refOutput):
    preRaw = model.predict(nnInput, batch_size = BATCH)
    pre = []
    for i in range(len(preRaw)):
        pre.append(preRaw[i][0])

    # temporarily undo z-score normalization, if applied
    if ZSCORE_NORM:
        pre = data.zScoreDenormalize(pre, zMean, zDev)
        refOutput = data.zScoreDenormalize(refOutput, zMean, zDev)

    # print samples of predictions
    for i in range(PREDICT_PRINT_SAMPLES):
        if LOGARITHM:
            print("    prediction: {}, reference: {}".format(exp(pre[i]),
                exp(refOutput[i])))
        else:
            print("    prediction: {}, reference: {}".format(pre[i],
                refOutput[i]))

    # array of errors
    error = []
    errorSqr = []
    for i in range(len(pre)):
        if LOGARITHM:
            e = abs(exp(pre[i]) - exp(refOutput[i]))
        else:
            e = abs(pre[i] - refOutput[i])
        error.append(e)
        errorSqr.append(e * e)

    # averages of everything
    preAvg = 0.0
    preAvg = utility.mean(pre, len(pre))
    refAvg = utility.mean(refOutput, len(pre))
    errAvg = utility.mean(error, len(pre))
    errSqr = 0.0
    for i in range(len(pre)):
        errSqr += errorSqr[i]
    errSqr = sqrt(errSqr / len(pre))

    # std. deviation of error
    errDev = 0.0
    for i in range(len(pre)):
        errDev += (error[i] - errAvg) * (error[i] - errAvg)
    errDev = sqrt(errDev / len(pre))

    print("    prediction mean:        {}".format(preAvg))
    print("    reference mean:         {}".format(refAvg))
    print("    mean absolute error:    {}".format(errAvg))
    print("    error std. deviation:   {}".format(errDev))
    print("    root mean square error: {}".format(errSqr))


def test(model, nnInput, refOutput):
    print("\n  Scoring using evaluate...")
    score = model.evaluate(nnInput, refOutput, batch_size = BATCH,
            verbose = False)
    print("    Score on traning data: {}".format(score))
    print("  ...done")


def run(source):
    # Initialize using same seed (to get stable results on comparisons)
    np.random.seed(12345)

    fullIn, labels, alphaSize = data.prepareData(source)

    if LOGARITHM:
        labels[LABEL_IDX] = data.logarithm(labels[LABEL_IDX])

    global zMean, zDev
    if ZSCORE_NORM:
        labels[LABEL_IDX], zMean, zDev = data.zScoreNormalize(labels[LABEL_IDX])

    """ In case a subset is wanted
    nnInput, ref2 = data.randomSelection(RANDOM_SAMPLES, nnInput, ref2)
    """

    """ Single model setup """
    trainIn, trainLabel, testIn, testLabel = data.holdout(HOLDOUT_RATIO,
            fullIn, labels[LABEL_IDX])
    model = setup(alphaSize)
    train(model, trainIn, trainLabel)
    print("\n  Prediction of training data:")
    predict(model, trainIn, trainLabel)
    print("\n  Prediction of testing data:")
    predict(model, testIn, testLabel)
    test(model, testIn, testLabel)

    """ Chained models setup
    modelCol1s = setup(alphaSize)
    chainSetup = train(modelCol1s, fullIn, labels[0])
    predict(modelCol1s, fullIn, labels[0])
    modelCol2 = setupInitialized(alphaSize, chainSetup)

    trainIn, trainLabel, testIn, testLabel = data.holdout(HOLDOUT_RATIO,
            fullIn, labels[1])
    train(modelCol2, trainIn, trainLabel)
    print("\n  Prediction of training data:")
    predict(modelCol2, trainIn, trainLabel)
    print("\n  Prediction of testing data:")
    predict(modelCol2, testIn, testLabel)
    test(modelCol2, testIn, testLabel)
    """

    # predict(5, labels[0], labels[1])
