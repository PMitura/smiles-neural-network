import data, utility, baseline
import time
import numpy as np
import keras.callbacks
import chembl as ch
import psutil

# from scipy.stats.stats import pearsonr
# from sklearn.metrics import roc_auc_score
# not used to relieve metacentrum of some dependecies
from math import sqrt, exp, log, ceil

# TODO: Remove unused imports after experiments are done
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, LSTM, AveragePooling1D
from keras.layers import TimeDistributed, SimpleRNN, GRU, Flatten, Merge
from keras.layers import BatchNormalization, Embedding
from keras.layers.advanced_activations import PReLU
from keras.optimizers import Adam, RMSprop
# from keras.regularizers import l1


# RNN parameters
SEED = 12346
TD_LAYER_MULTIPLIER = 0.5   # Time-distributed layer modifier of neuron count
GRU_LAYER_MULTIPLIER = 1    # -||- for GRU
EPOCHS = 3
BATCH = 96                  # metacentrum.cz: 128 - 160, optimum by grid: 96
LEARNING_RATE = 0.01
EARLY_STOP = 10             # Number of tolerated epochs without improvement
OPTIMIZER = Adam(lr = LEARNING_RATE)
USE_EMBEDDING = data.USE_EMBEDDING
EMBEDDING_OUTPUTS = 30      # Number of inputs generated by Embedding if used
CHAINED_MODELS = True
CHAINED_LABELS = [[0], [1]]
CHAINED_PREDICT = [2]

# Preprocessing switches
LABEL_IDXS = [0, 1]            # Indexes of columns to use as label
ZSCORE_NORM = True          # Undone after testing
LOGARITHM = True            # Dtto, sets all values (x) to -log(x) 

# Holdout settings
FLAG_BASED_HOLD = True      # Bases holdout on col called 'is_testing'
HOLDOUT_RATIO = 0.8         # Used if flag based holdout is disabled

# Testing settings
PREDICT_PRINT_SAMPLES = 15  # Samples printed to stdout
CLASSIFY = False            # Regression if False
DISTRIB_BINS = 15           # Bins form visualising output distribution
USE_PARTITIONS = True       # Partition test set and compute averages
NUM_PARTITIONS = 5

# Statistics settings
SEND_STATISTICS = False
COMMENT = 'Chained prediction: molweight(train), alogp(train), std_val(test)'
SCATTER_VISUALIZE = True


def configureModel(alphaSize, nomiSize = (0, 0), outputLen = len(LABEL_IDXS)):
    print('  Initializing and compiling...')

    model = Sequential()
    
    if USE_EMBEDDING:
        # second value in nomiSize tuple is shift while using embedding
        model.add(Embedding(1 << nomiSize[1], EMBEDDING_OUTPUTS)) 
        model.add(TimeDistributed(Dense(int(TD_LAYER_MULTIPLIER * (alphaSize +
            nomiSize[0])), activation = 'tanh')))
    else:
        model.add(TimeDistributed(Dense(int(TD_LAYER_MULTIPLIER * (alphaSize +
            nomiSize)), activation = 'tanh'),
            input_shape = (None, alphaSize + nomiSize)))
    model.add(GRU(int(GRU_LAYER_MULTIPLIER * alphaSize)))
    model.add(Activation('relu'))
    model.add(Dense(outputLen))

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
    model.compile(loss = 'mse', optimizer = OPTIMIZER)

    print('  ...done')
    return model

"""
def setup(alphaSize, nomiSize = 0, outputLen = ):
    return configureModel(alphaSize, nomiSize)


def setupInitialized(alphaSize, nomiSize = (0, 0) weights):
    model = configureModel(alphaSize, nomiSize)
    model.set_weights(weights)
    return model
"""

def train(model, nnInput, labels, validation, makePlot = True,
        labelIndexes = LABEL_IDXS):
    print('  Training model...')

    # needed format is orthogonal to ours
    formattedLabels = np.zeros((len(labels[0]), len(labelIndexes)))
    formattedValid = np.zeros((len(validation[1][labelIndexes[0]]), 
        len(labelIndexes)))
    for i in range(len(labelIndexes)):
        for j in range(len(labels[0])):
            formattedLabels[j][i] = labels[labelIndexes[i]][j]
        for j in range(len(validation[1][labelIndexes[i]])):
            formattedValid[j][i] = validation[1][labelIndexes[i]][j]

    early = keras.callbacks.EarlyStopping(monitor = 'loss',
            patience = EARLY_STOP)
    history = model.fit(nnInput, formattedLabels, nb_epoch = EPOCHS,
            batch_size = BATCH, callbacks = [early],
            validation_data = (validation[0], formattedValid))

    if makePlot:
        values = np.zeros((len(history.history['loss']), 2))
        for i in range(len(history.history['loss'])):
            values[i][0] = history.history['loss'][i]
            values[i][1] = history.history['val_loss'][i]
        utility.plotLoss(values)

    print('    Model weights:')
    print(model.summary())
    # print(model.get_weights())
    print('  ...done')
    return len(history.history['loss'])


# Serves as extended version of test, gives statistics
def predict(model, nnInput, rawLabel, labelIndexes = LABEL_IDXS):
    preRaw = model.predict(nnInput, batch_size = BATCH)

    labCtr = 0
    for labidx in labelIndexes:
        print '  Predictions for label {}'.format(labidx)

        pre = []
        label = []
        for i in range(len(preRaw)):
            pre.append(preRaw[i][labCtr])
            label.append(rawLabel[labidx][i])
        labCtr += 1

        # temporarily undo z-score normalization, if applied
        if ZSCORE_NORM:
            pre = data.zScoreDenormalize(pre, zMean, zDev)
            label = data.zScoreDenormalize(label, zMean, zDev)

        if LOGARITHM:
            for i in range(len(pre)):
                pre[i] = exp(-pre[i])
                label[i] = exp(-label[i])

        # print samples of predictions
        for i in range(min(PREDICT_PRINT_SAMPLES, len(pre))):
            print("    prediction: {}, label: {}".format(pre[i],
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

        print("      prediction mean:         {}".format(preAvg))
        print("      label mean:              {}".format(refAvg))
        print("      mean absolute error:     {}".format(errAvg))
        print("      error std. deviation:    {}".format(errDev))
        print("      root mean square error:  {}".format(errSqr))
        print("      correlation coefficient: {}".format(pearCr[0][1]))
        print("      R2:                      {}".format(pearCr[0][1] * pearCr[0][1]))

    return pearCr[0][1] * pearCr[0][1]


# Modification of predict, divides data and computes avg and dev of R2
def predictSplit(model, nnInput, rawLabel, labelIndexes = LABEL_IDXS):
    partSize = len(nnInput) / NUM_PARTITIONS
    rsqrs = np.zeros((len(labelIndexes), NUM_PARTITIONS))
    for i in range(NUM_PARTITIONS):
        print '\n    Partition {}'.format(i)
        if USE_EMBEDDING:
            partInput = np.zeros([partSize, len(nnInput[0])])
        else:
            partInput = np.zeros([partSize, len(nnInput[0]), len(nnInput[0][0])])
        base = i * partSize
        for j in range(base, base + partSize):
            partInput[j - base] = nnInput[j]

        preRaw = model.predict(partInput, batch_size = BATCH)

        ctr = 0
        labCtr = 0
        for labidx in labelIndexes:
            print '      Prediction for label {}'.format(labidx)

            pre = []
            label = []
            for j in range(len(preRaw)):
                pre.append(preRaw[j][labCtr])
                label.append(rawLabel[labidx][base + j])
            labCtr += 1

            if ZSCORE_NORM:
                pre = data.zScoreDenormalize(pre, zMean, zDev)
                label = data.zScoreDenormalize(label, zMean, zDev)

            if LOGARITHM:
                for j in range(len(pre)):
                    pre[j] = exp(-pre[j])
                    label[j] = exp(-label[j])

            # print samples of predictions
            for j in range(min(PREDICT_PRINT_SAMPLES / NUM_PARTITIONS, len(pre))):
                print("        prediction: {}, label: {}".format(pre[j],
                    label[j]))

            merged = np.zeros((2, len(pre)))
            for j in range(len(pre)):
                merged[0][j] = pre[j]
                merged[1][j] = label[j]
            pearCr = np.corrcoef(merged)
            rsqr = pearCr[0][1] * pearCr[0][1]
            rsqrs[ctr][i] = rsqr
            ctr += 1
            print '      R2 Value: {}'.format(rsqr)

    print '\n    R2 Statistics:'
    rsqrAvg = []
    rsqrDev = []
    for lab in range(len(rsqrs)):
        rsqrSum = 0.0
        for sqr in rsqrs[lab]:
            rsqrSum += sqr
        rsqrAvg.append(rsqrSum / NUM_PARTITIONS)

        rsqrDev.append(0.0)
        for i in range(NUM_PARTITIONS):
            rsqrDev[lab] += (rsqrs[lab][i] - rsqrAvg[lab]) * (rsqrs[lab][i] -
                    rsqrAvg[lab])
        rsqrDev[lab] = sqrt(rsqrDev[lab] / NUM_PARTITIONS)

        print '      label {} R2 Average:   {}'.format(lab, rsqrAvg[lab])
        print '      label {} R2 Deviation: {}'.format(lab, rsqrDev[lab])

    rsqrAvgOverall = utility.mean(rsqrAvg, len(rsqrAvg))
    rsqrDevOverall = utility.mean(rsqrDev, len(rsqrDev))
    print '\n  R2 mean of avgs: {}'.format(rsqrAvgOverall)
    print '  R2 mean of devs: {}'.format(rsqrDevOverall)
    return rsqrAvgOverall, rsqrDevOverall


# Two classes, bins to closest of {-1, 1}
# TODO: math imports, removed because of cluster runs
# TODO: add log_loss
# TODO: doesn't work with multicol labels - needs rework, don't use!
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


# TODO: encapsulate training rnn on a label to a function, not working yet
def modelOnLabels(trainIn, trainLabel, testIn, testLabel, alphaSize, nomiSize, 
        indexes, weights = None):
    if weights == None:
        model = configureModel(alphaSize, nomiSize, outputLen = len(indexes))
    else:
        model = configureModel(alphaSize, nomiSize, outputLen = len(indexes))
        model.set_weights(weights)

    epochsDone = train(model, trainIn, trainLabel, (testIn, testLabel),
            labelIndexes = indexes)
    return model, epochsDone


def run(source, grid = None):
    startTime = time.time()

    # Initializ) using the same seed (to get stable results on comparisons)
    np.random.seed(SEED)

    """ Single model setup """
    fullIn, labels, alphaSize, nomiSize, testFlags = data.prepareData(source)

    # Do preprocessing and train/test data division
    if LOGARITHM:
        for idx in LABEL_IDXS:
            labels[idx] = data.logarithm(labels[idx])

    global zMean, zDev
    if ZSCORE_NORM:
        for idx in LABEL_IDXS:
            labels[idx], zMean, zDev = data.zScoreNormalize(labels[idx])

    if FLAG_BASED_HOLD:
        trainIn, trainLabel, testIn, testLabel = data.holdoutBased(testFlags,
                fullIn, labels)
    else:
        trainIn, trainLabel, testIn, testLabel = data.holdout(HOLDOUT_RATIO,
                fullIn, labels)

    if not CHAINED_MODELS:
        model = configureModel(alphaSize, nomiSize)

        epochsDone = train(model, trainIn, trainLabel, (testIn, testLabel))

        if SCATTER_VISUALIZE:
            utility.visualize2D(model, 1, testIn, testLabel[LABEL_IDXS[0]])

        print("\n  Prediction of training data:")
        if CLASSIFY:
            raise NotImplementedError('Classifications are TODO')
            classify(model, trainIn, trainLabel)
        else:
            relevanceTrain = predict(model, trainIn, trainLabel)

        print("\n  Prediction of testing data:")
        if CLASSIFY:
            classify(model, testIn, testLabel)
        else:
            if USE_PARTITIONS:
                relevanceTest, stdTest = predictSplit(model, testIn, testLabel)
            else:
                relevanceTest = predict(model, testIn, testLabel)
    else:
        # Chained setup
        model, epochsDone = modelOnLabels(trainIn, trainLabel, testIn, testLabel,
                alphaSize, nomiSize, [0])
        for idx in range(1, len(CHAINED_LABELS)):
            model, epochsDone = modelOnLabels(trainIn, trainLabel, testIn, 
                testLabel, alphaSize, nomiSize, CHAINED_LABELS[idx],
                model.get_weights())

        if SCATTER_VISUALIZE:
            utility.visualize2D(model, 1, testIn, testLabel[CHAINED_PREDICT[0]])

        print("\n  Prediction of training data:")
        if CLASSIFY:
            raise NotImplementedError('Classifications are TODO')
            classify(model, trainIn, trainLabel)
        else:
            relevanceTrain = predict(model, trainIn, trainLabel, 
                    labelIndexes = CHAINED_PREDICT)

        print("\n  Prediction of testing data:")
        if CLASSIFY:
            classify(model, testIn, testLabel)
        else:
            if USE_PARTITIONS:
                relevanceTest, stdTest = predictSplit(model, testIn, testLabel,
                    labelIndexes = CHAINED_PREDICT)
            else:
                relevanceTest = predict(model, testIn, testLabel, labelIndexes
                    = CHAINED_PREDICT)

    endTime = time.time()
    deltaTime = endTime - startTime

    if CLASSIFY:
        taskType = 'classification'
    else:
        taskType = 'regression'
    modelSummary = utility.modelToString(model)

    # get memory usage
    thisProc = psutil.Process()
    memRss = thisProc.memory_info().rss / 1000000.0
    memVms = thisProc.memory_info().vms / 1000000.0
    print '  Memory usage:'
    print '    Physical: {} MB'.format(memRss)
    print '    Virtual:  {} MB'.format(memVms)

    # TODO: add memory_pm_mb, memory_vm_bm
    if SEND_STATISTICS:
        ch.sendStatistics(
            training_row_count = len(trainLabel),
            testing_row_count = len(testLabel),
            task = taskType,
            relevance_training = relevanceTrain,
            relevance_testing = relevanceTest,
            relevance_testing_std = stdTest,
            epoch_max = EPOCHS,
            epoch_count = epochsDone,
            runtime_second = deltaTime,
            parameter_count = model.count_params(),
            learning_rate = LEARNING_RATE,
            optimization_method = OPTIMIZER.__class__.__name__,
            batch_size = BATCH,
            comment = COMMENT,
            label_name = ch.LABELNAME,
            model = modelSummary,
            seed = SEED,
            memory_pm_mb = memRss,
            memory_vm_mb = memVms,
            learning_curve = open('plots/{}'.format(utility.PLOT_NAME),
                'rb').read())

