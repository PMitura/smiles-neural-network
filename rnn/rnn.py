import data, utility
import numpy as np
import keras.callbacks

# from scipy.stats.stats import pearsonr
# from sklearn.metrics import roc_auc_score
from math import sqrt, exp

# TODO: Remove unused imports after experiments are done
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, LSTM, AveragePooling1D
from keras.layers import TimeDistributed, SimpleRNN, GRU, Flatten
from keras.optimizers import Adam, RMSprop
# from keras.regularizers import l1

from keras import backend as K

# RNN parameters
GRU_LAYER_MULTIPLIER = 1
TD_LAYER_MULTIPLIER = 0.25
EPOCHS = 150
BATCH = 160 # metacentrum recommended: 128 - 160
LEARNING_RATE = 0.005
EARLY_STOP = 10

# Holdout settings
FLAG_BASED_HOLD = True
HOLDOUT_RATIO = 0.8
CLASSIFY = False

# Input settings
LABEL_IDX = 0
PREDICT_PRINT_SAMPLES = 15

# Preprocessing switches
ZSCORE_NORM = False
LOGARITHM = True


def configureModel(alphaSize, nomiSize = 0):
    print('  Initializing and compiling...')

    model = Sequential()
    """
    model.add(LSTM(LAYER_MULTIPLIER * alphaSize, activation = 'sigmoid',
        input_shape = (None, alphaSize), return_sequences = True))
    """
    model.add(TimeDistributed(Dense(int(TD_LAYER_MULTIPLIER * (alphaSize +
        nomiSize))), input_shape = (None, alphaSize + nomiSize)))
    model.add(TimeDistributed(Dense(int(TD_LAYER_MULTIPLIER * (alphaSize +
        nomiSize)))))
    # model.add(TimeDistributed(Dense(1), input_shape = (None, alphaSize + nomiSize)))
    # model.add(TimeDistributed(Dense(LAYER_MULTIPLIER * alphaSize)))
    model.add(GRU(int(GRU_LAYER_MULTIPLIER * alphaSize), activation = 'sigmoid'))
    # model.add(SimpleRNN(2 * LAYER_MULTIPLIER * alphaSize, activation = 'sigmoid'))
    # model.add(AveragePooling1D(pool_length = alphaSize, border_mode='valid'))
    # model.add(Dropout(0.5))
    # model.add(Dense(1, W_regularizer = l1(0.01)))
    model.add(Dense(1))
    # model.add(Activation('tanh'))

    # default learning rate 0.001
    model.compile(loss = 'mse', optimizer = Adam(lr = LEARNING_RATE))

    print('  ...done')
    return model


# 3D input: dimensions are (number of samples, length of word, alphabet)
def setup(alphaSize, nomiSize = 0):
    return configureModel(alphaSize, nomiSize)


def setupInitialized(alphaSize, weights):
    model = configureModel(alphaSize)
    model.set_weights(weights)
    return model


def train(model, nnInput, refOutput):
    print('  Training model...')
    early = keras.callbacks.EarlyStopping(monitor = 'loss', patience = EARLY_STOP)
    model.fit(nnInput, refOutput, nb_epoch = EPOCHS, batch_size = BATCH,
            callbacks = [early])
    print('    Model weights:')
    print(model.summary())
    # print(model.get_weights())
    print('  ...done')
    return model.get_weights()


# Serves as extended version of test, gives averages
def predict(model, nnInput, rawLabel):
    preRaw = model.predict(nnInput, batch_size = BATCH)
    pre = []
    label = []
    for i in range(len(preRaw)):
        pre.append(preRaw[i][0])
        if LOGARITHM:
            label.append(exp(-rawLabel[i]))
        else:
            label.append(rawLabel[i])

    # temporarily undo z-score normalization, if applied
    if ZSCORE_NORM:
        pre = data.zScoreDenormalize(pre, zMean, zDev)
        refOutput = data.zScoreDenormalize(label, zMean, zDev)

    if LOGARITHM:
        for i in range(len(pre)):
            pre[i] = exp(-pre[i])

    # print samples of predictions
    for i in range(min(PREDICT_PRINT_SAMPLES, len(pre))):
        print("    prediction: {}, reference: {}".format(pre[i],
            label[i]))

    # array of errors
    error = []
    errorSqr = []
    for i in range(len(pre)):
        e = abs(pre[i] - label[i])
        error.append(e)
        errorSqr.append(e * e)

    # averages of everything
    preAvg = utility.mean(pre, len(pre))
    refAvg = utility.mean(label, len(pre))
    errAvg = utility.mean(error, len(pre))
    errSqr = 0.0
    for i in range(len(pre)):
        errSqr += errorSqr[i]
    errSqr = sqrt(errSqr / len(pre))

    merged = np.zeros((2, len(pre)))
    for i in range(len(pre)):
        merged[0][i] = pre[i]
        merged[1][i] = label[i]
    pearCr = np.corrcoef(merged)

    # std. deviation of error
    errDev = 0.0
    for i in range(len(pre)):
        errDev += (error[i] - errAvg) * (error[i] - errAvg)
    errDev = sqrt(errDev / len(pre))

    print("    prediction mean:         {}".format(preAvg))
    print("    label mean:              {}".format(refAvg))
    print("    mean absolute error:     {}".format(errAvg))
    print("    error std. deviation:    {}".format(errDev))
    print("    root mean square error:  {}".format(errSqr))
    print("    correlation coefficient: {}".format(pearCr[0][1]))
    print("    R2:                      {}".format(pearCr[0][1] * pearCr[0][1]))


# Two classes, bins to closest of {-1, 1}
# TODO: return imports, removed because of cluster runs
def classify(model, nnInput, label):
    preRaw = model.predict(nnInput, batch_size = BATCH)
    pre = []
    for i in range(len(preRaw)):
        pre.append(preRaw[i][0])

    falseNegative = 0.0
    falsePositive = 0.0
    truePositive  = 0.0
    trueNegative  = 0.0

    for i in range(len(pre)):
        if i < PREDICT_PRINT_SAMPLES:
            print "    Predicted: {} Label: {}".format(pre[i], label[i])
        if pre[i] < 0 and label[i] == 1:
            falseNegative += 1
        elif pre[i] > 0 and label[i] == -1:
            falsePositive += 1
        elif pre[i] > 0 and label[i] == 1:
            truePositive += 1
        elif pre[i] < 0 and label[i] == -1:
            trueNegative += 1

    errors = falseNegative + falsePositive

    # array of errors
    error = []
    errorSqr = []
    for i in range(len(pre)):
        e = abs(pre[i] - label[i])
        error.append(e)
        errorSqr.append(e * e)

    # averages of everything
    preAvg = utility.mean(pre, len(pre))
    refAvg = utility.mean(label, len(pre))
    errAvg = utility.mean(error, len(pre))
    errSqr = 0.0
    for i in range(len(pre)):
        errSqr += errorSqr[i]
    errSqr = sqrt(errSqr / len(pre))
    pearCr = pearsonr(pre, label)

    # std. deviation of error
    errDev = 0.0
    for i in range(len(pre)):
        errDev += (error[i] - errAvg) * (error[i] - errAvg)
    errDev = sqrt(errDev / len(pre))

    sensitivity = truePositive / (truePositive + falseNegative)
    specificity = trueNegative / (trueNegative + falsePositive)
    precision   = truePositive / (truePositive + trueNegative)
    accuracy    = (1 - (errors / len(pre)))
    if precision + sensitivity != 0:
        fmeasure = (2 * precision * sensitivity) / (precision + sensitivity)
    else:
        fmeasure = float('nan')

    print("    prediction mean:         {}".format(preAvg))
    print("    label mean:              {}".format(refAvg))
    print("    mean absolute error:     {}".format(errAvg))
    print("    error std. deviation:    {}".format(errDev))
    print("    root mean square error:  {}".format(errSqr))
    print("    correlation coefficient: {}".format(pearCr[0]))
    print("    R2:                      {}".format(pearCr[0] * pearCr[0]))

    print("    Classification accuracy: {}%"
            .format(accuracy * 100))
    print("    ROC AUC score:           {}"
            .format(roc_auc_score(label, pre)))
    print("    Sensitivity:             {}".format(sensitivity))
    print("    Specificity:             {}".format(specificity))
    print("    Precision:               {}".format(precision))
    print("    F-measure:               {}".format(fmeasure))


def test(model, nnInput, refOutput):
    print("\n  Scoring using evaluate...")
    score = model.evaluate(nnInput, refOutput, batch_size = BATCH,
            verbose = False)
    print("    Score on traning data: {}".format(score))
    print("  ...done")


def run(source):
    # Initialize using same seed (to get stable results on comparisons)
    np.random.seed(12345)

    tables = ['target_protein_p00734_ec50',
              'target_protein_p00734_ic50',
              'target_protein_p03372_ec50',
              'target_protein_p03372_ic50']

    """ Single model setup """
    fullIn, labels, alphaSize, nomiSize, testFlags = data.prepareData(source)

    if LOGARITHM:
        labels[LABEL_IDX] = data.logarithm(labels[LABEL_IDX])

    global zMean, zDev
    if ZSCORE_NORM:
        labels[LABEL_IDX], zMean, zDev = data.zScoreNormalize(labels[LABEL_IDX])

    if FLAG_BASED_HOLD:
        trainIn, trainLabel, testIn, testLabel = data.holdoutBased(testFlags,
                fullIn, labels[LABEL_IDX])
    else:
        trainIn, trainLabel, testIn, testLabel = data.holdout(HOLDOUT_RATIO,
                fullIn, labels[LABEL_IDX])
    model = setup(alphaSize, nomiSize)
    train(model, trainIn, trainLabel)

    print model.get_weights()

    """
    print("\n  Visualisation test:")
    vlayer = K.function([model.layers[0].input], [model.layers[1].output])
    result = vlayer([testIn])[0]
    print "\n  Len of layer: {} Len of data: {}".format(len(result),
            len(testIn))
    for line in result:
        for val in line:
            print '{0:.20f}'.format(val)
    """

    print("\n  Prediction of training data:")
    if CLASSIFY:
        classify(model, trainIn, trainLabel)
    else:
        predict(model, trainIn, trainLabel)
    print("\n  Prediction of testing data:")
    if CLASSIFY:
        classify(model, testIn, testLabel)
    else:
        predict(model, testIn, testLabel)

    # test(model, testIn, testLabel)
    """ """

    """ Chained models setup
    testInputs = []
    trainInputs = []
    trainLabels = []
    testLabels = []

    chainSetup = 0
    model = 0
    # Chained training
    for i in range(4):
        initIn, initLabel, alphaSize = data.prepareData(source, tables[i])
        if LOGARITHM:
            initLabel[LABEL_IDX] = data.logarithm(initLabel[LABEL_IDX])
        trainIn, trainLabel, testIn, testLabel = data.holdout(HOLDOUT_RATIO,
                initIn, initLabel[LABEL_IDX])

        trainInputs.append(trainIn)
        testInputs.append(testIn)
        trainLabels.append(trainLabel)
        testLabels.append(testLabel)

        # Do not reuse weights on first model
        if i == 0:
            model = setup(alphaSize)
        else:
            model = setupInitialized(alphaSize, chainSetup)
        chainSetup = train(model, trainInputs[i], trainLabels[i])

    # Test on all
    for i in range(4):
        print("\n  Prediction of training data in table {}:".format(tables[i]))
        predict(model, trainInputs[i], trainLabels[i])
        print("\n  Prediction of testing data in table {}:".format(tables[i]))
        predict(model, testInputs[i], testLabels[i])
    """


    """ Simple chained models setup
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

