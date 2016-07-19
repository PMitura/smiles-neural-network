import data, utility
import numpy as np
import keras.callbacks

# from scipy.stats.stats import pearsonr
# from sklearn.metrics import roc_auc_score
# not used to relieve metacentrum of some dependecies
from math import sqrt, exp

# TODO: Remove unused imports after experiments are done
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, LSTM, AveragePooling1D
from keras.layers import TimeDistributed, SimpleRNN, GRU, Flatten, Merge
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.optimizers import Adam, RMSprop
# from keras.regularizers import l1

from keras import backend as K

# RNN parameters
TD_LAYER_MULTIPLIER = 0.5   # Time-distributed layer modifier of neuron count
GRU_LAYER_MULTIPLIER = 1    # -||- for GRU
EPOCHS = 150
BATCH = 160                 # metacentrum.cz recommended: 128 - 160
LEARNING_RATE = 0.01
EARLY_STOP = 10             # Number of tolerated epochs without improvement

# Preprocessing switches
LABEL_IDX = 0               # Index of column to use as label
ZSCORE_NORM = False         # Undone after testing
LOGARITHM = False           # Dtto, sets all values (x) to -log(x) 

# Holdout settings
FLAG_BASED_HOLD = True      # Bases holdout on col called 'is_testing'
HOLDOUT_RATIO = 0.8         # Used if flag based holdout is disabled

# Testing settings
PREDICT_PRINT_SAMPLES = 15  # Samples printed to stdout
CLASSIFY = False            # Regression if False
DISTRIB_BINS = 15           # Bins form visualising output distribution


def configureModel(alphaSize, nomiSize = 0):
    print('  Initializing and compiling...')

    model = Sequential()
    
    # TD - GRU - out setup
    # model.add(TimeDistributed(Dense(int(TD_LAYER_MULTIPLIER * (alphaSize +
    #     nomiSize)), activation = 'tanh'),
    #     input_shape = (None, alphaSize + nomiSize)))
    model.add(TimeDistributed(Dense(int(TD_LAYER_MULTIPLIER * (alphaSize +
        nomiSize)), activation = 'tanh'),
        input_shape = (None, alphaSize + nomiSize)))
    model.add(GRU(int(GRU_LAYER_MULTIPLIER * alphaSize), activation = 'tanh'))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation(PReLU()))

    """ Bidirectional version, submitted as issue at
        https://github.com/fchollet/keras/issues/2646

    model.add(TimeDistributed(Dense(int(TD_LAYER_MULTIPLIER * (alphaSize +
        nomiSize)), activation = 'tanh'),
        input_shape = (None, alphaSize + nomiSize)))

    fwModel = Sequential()
    fwModel.add(GRU(int(GRU_LAYER_MULTIPLIER * alphaSize),
            activation = 'sigmoid',
            input_shape = (None, alphaSize + nomiSize)))

    bwModel = Sequential()
    bwModel.add(GRU(int(GRU_LAYER_MULTIPLIER * alphaSize), 
            activation = 'sigmoid', go_backwards = True,
            input_shape = (None, alphaSize + nomiSize)))

    model.add(Merge([fwModel, bwModel], mode = 'sum'))
    model.add(Dense(1))
    """

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
    # Care! Doesn't work on metacentrum.cz (cause: dependencies)
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


# Time distributed layers require additional nesting
def outputDistribution(model, layerID, testIn, withtime = False):
    print("\n  Generating output distribution for layer {}".format(layerID))
    vlayer = K.function([model.layers[0].input], [model.layers[layerID].output])
    result = vlayer([testIn])

    # divide into bins
    array = []
    for instance in result:
        for line in instance:
            for val in line:
                if withtime:
                    for deepval in val:
                        array.append(deepval)
                else:
                    array.append(val)
    bins = np.histogram(array)

    # print bins
    print bins[1]
    for i in range(len(bins[0])):
        print "({}, {})".format(bins[1][i], bins[0][i])
    print("  ...done")


def run(source):
    # Initialize using the same seed (to get stable results on comparisons)
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

    # print model.get_weights()
    # outputDistribution(model, 0, testIn, withtime = True)
    # outputDistribution(model, 1, testIn)
    # outputDistribution(model, 2, testIn)

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

