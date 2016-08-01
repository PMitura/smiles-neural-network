import data, utility
import time
import numpy as np
import keras.callbacks

import db.db as db
from config import config as cc

import socket

from scipy.stats.stats import pearsonr
from sklearn.metrics import roc_auc_score
# not used to relieve MetaCentrum of some dependencies
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
SEED = cc.exp['params']['rnn']['seed']
TD_LAYER_MULTIPLIER = cc.exp['params']['rnn']['td_layer_multiplier']   # Time-distributed layer modifier of neuron count
GRU_LAYER_MULTIPLIER = cc.exp['params']['rnn']['gru_layer_multiplier']    # -||- for GRU
EPOCHS = cc.exp['params']['rnn']['epochs']
BATCH = cc.exp['params']['rnn']['batch']                 # metacentrum.cz: 128 - 160, optimum by grid: 96
LEARNING_RATE = cc.exp['params']['rnn']['learning_rate']
EARLY_STOP = cc.exp['params']['rnn']['early_stop']
OPTIMIZER = Adam(lr = LEARNING_RATE)
USE_EMBEDDING = cc.exp['params']['data']['use_embedding']
EMBEDDING_OUTPUTS = cc.exp['params']['rnn']['embedding_outputs']
CHAINED_MODELS = cc.exp['params']['rnn']['chained_models']

# need to eval!
CHAINED_LABELS = eval(cc.exp['params']['rnn']['chained_labels'])
CHAINED_PREDICT = eval(cc.exp['params']['rnn']['chained_predict'])
FREEZE_IDXS = eval(cc.exp['params']['rnn']['freeze_idxs'])

TRAINABLE_INNER = cc.exp['params']['rnn']['trainable_inner']

# Learning rate decay settings
LEARNING_RATE_DECAY = cc.exp['params']['rnn']['learning_rate_decay']
LEARNING_RATE_DECAY_TYPE = cc.exp['params']['rnn']['learning_rate_decay_type']
LEARNING_RATE_DECAY_STEP_CONFIG_STEPS = cc.exp['params']['rnn']['learning_rate_decay_step_config_steps']
LEARNING_RATE_DECAY_STEP_CONFIG_RATIO = cc.exp['params']['rnn']['learning_rate_decay_step_config_ratio']

# Classification settings
CLASSIFY_THRESHOLD = cc.exp['params']['rnn']['classify_threshold']
CLASSIFY_LABEL_POS = cc.exp['params']['rnn']['classify_label_pos']
CLASSIFY_LABEL_NEG = cc.exp['params']['rnn']['classify_label_neg']
CLASSIFY_ACTIVATION = cc.exp['params']['rnn']['classify_activation']

# Preprocessing switches
# need to eval!
LABEL_IDXS = eval(cc.exp['params']['rnn']['label_idxs'])
ZSCORE_NORM = cc.exp['params']['rnn']['zscore_norm']
LOGARITHM = cc.exp['params']['rnn']['logarithm']

# Holdout settings
FLAG_BASED_HOLD = cc.exp['params']['rnn']['flag_based_hold']
HOLDOUT_RATIO = cc.exp['params']['rnn']['holdout_ratio']

# Testing settings
PREDICT_PRINT_SAMPLES = cc.exp['params']['rnn']['predict_print_samples']
CLASSIFY = cc.exp['params']['rnn']['classify']
DISTRIB_BINS = cc.exp['params']['rnn']['distrib_bins']
USE_PARTITIONS = cc.exp['params']['rnn']['use_partitions']
NUM_PARTITIONS = cc.exp['params']['rnn']['num_partitions']

# Statistics settings
COMMENT = cc.exp['params']['rnn']['comment']
SCATTER_VISUALIZE = cc.exp['params']['rnn']['scatter_visualize']


def configureModel(alphaSize, nomiSize = (0, 0), outputLen = len(LABEL_IDXS)):
    print('  Initializing and compiling...')

    model = Sequential()

    if USE_EMBEDDING:
        # second value in nomiSize tuple is shift while using embedding
        model.add(Embedding(1 << nomiSize[1], EMBEDDING_OUTPUTS))
        model.add(TimeDistributed(Dense(int(TD_LAYER_MULTIPLIER * (alphaSize +
            nomiSize[0])), activation = 'tanh', trainable = TRAINABLE_INNER)))
    else:
        model.add(TimeDistributed(Dense(int(TD_LAYER_MULTIPLIER * (alphaSize +
            nomiSize)), activation = 'tanh', trainable = TRAINABLE_INNER),
            input_shape = (None, alphaSize + nomiSize)))
    model.add(GRU(int(GRU_LAYER_MULTIPLIER * alphaSize), trainable =
            TRAINABLE_INNER))
    model.add(Activation('relu', trainable = TRAINABLE_INNER))
    model.add(Dense(outputLen))

    if CLASSIFY:
        model.add(Activation(CLASSIFY_ACTIVATION, trainable = TRAINABLE_INNER))

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

def learningRateDecayer(epoch):
    if not LEARNING_RATE_DECAY:
        return LEARNING_RATE

    if LEARNING_RATE_DECAY_TYPE == 'step':
        drop = np.floor((epoch)/LEARNING_RATE_DECAY_STEP_CONFIG_STEPS)
        new_lr = float(LEARNING_RATE * np.power(LEARNING_RATE_DECAY_STEP_CONFIG_RATIO,drop))
        print('lr',epoch,new_lr)
        return new_lr
    elif LEARNING_RATE_DECAY_TYPE == 'time':
        raise NotImplementedError('learning rate decay: time')
    elif LEARNING_RATE_DECAY_TYPE == 'peter':
        raise NotImplementedError('learning rate decay: peter')
    else:
        raise RuntimeError('learning rate decat: unknown type {}'.format(LEARNING_RATE_DECAY_TYPE))


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

    learningRateScheduler = keras.callbacks.LearningRateScheduler(learningRateDecayer)

    history = model.fit(nnInput, formattedLabels, nb_epoch = EPOCHS,
            batch_size = BATCH, callbacks = [early,learningRateScheduler],
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

    for labCtr,labidx in enumerate(labelIndexes):
        print '  Predictions for label {}'.format(labidx)

        pre = []
        label = []
        for i in range(len(preRaw)):
            pre.append(preRaw[i][labCtr])
            label.append(rawLabel[labidx][i])

        # temporarily undo z-score normalization, if applied
        if ZSCORE_NORM:
            pre = data.zScoreDenormalize(pre, zMean[labidx], zDev[labidx])
            label = data.zScoreDenormalize(label, zMean[labidx], zDev[labidx])

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
                pre = data.zScoreDenormalize(pre, zMean[labidx], zDev[labidx])
                label = data.zScoreDenormalize(label, zMean[labidx], zDev[labidx])

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

        print '      label {} R2 Average:   {}'.format(labelIndexes[lab],
                rsqrAvg[lab])
        print '      label {} R2 Deviation: {}'.format(labelIndexes[lab],
                rsqrDev[lab])

    rsqrAvgOverall = utility.mean(rsqrAvg, len(rsqrAvg))
    rsqrDevOverall = utility.mean(rsqrDev, len(rsqrDev))
    print '\n  R2 mean of avgs: {}'.format(rsqrAvgOverall)
    print '  R2 mean of devs: {}'.format(rsqrDevOverall)
    return rsqrAvgOverall, rsqrDevOverall


# Classification task
# pos class defined by CLASSIFY_LABEL_POS
# neg class defined by CLASSIFY_LABEL_NEG
# decision threshold defined by CLASSIFY_THRESHOLD
# activation function defined by CLASSIFY_ACTIVATION
def classify(model, nnInput, rawLabel, labelIndexes = LABEL_IDXS):
    preRaw = model.predict(nnInput, batch_size = BATCH)

    for labCtr, labidx in enumerate(labelIndexes):
        print '  Predictions for label {}'.format(labidx)

        pre = []
        label = []
        for i in range(len(preRaw)):
            pre.append(preRaw[i][0])
            label.append(rawLabel[labidx][i])

        falseNegative = 0.0
        falsePositive = 0.0
        truePositive  = 0.0
        trueNegative  = 0.0

        for i in range(len(pre)):
            if i < PREDICT_PRINT_SAMPLES:
                print "    Predicted: {} Label: {}".format(pre[i], label[i])
            if pre[i] < CLASSIFY_THRESHOLD and label[i] == CLASSIFY_LABEL_POS:
                falseNegative += 1
            elif pre[i] > CLASSIFY_THRESHOLD and label[i] == CLASSIFY_LABEL_NEG:
                falsePositive += 1
            elif pre[i] > CLASSIFY_THRESHOLD and label[i] == CLASSIFY_LABEL_POS:
                truePositive += 1
            elif pre[i] < CLASSIFY_THRESHOLD and label[i] == CLASSIFY_LABEL_NEG:
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

        logloss = utility.logloss(pre,label)

        print("    prediction mean:         {}".format(preAvg))
        print("    label mean:              {}".format(refAvg))
        print("    mean absolute error:     {}".format(errAvg))
        print("    error std. deviation:    {}".format(errDev))
        print("    root mean square error:  {}".format(errSqr))
        print("    correlation coefficient: {}".format(pearCr[0]))
        print("    R2:                      {}".format(pearCr[0] * pearCr[0]))
        print("    logloss:                      {}".format(logloss))

        print("    Classification accuracy: {}%"
                .format(accuracy * 100))
        # Care! Doesn't work on metacentrum.cz (cause: dependencies)
        print("    ROC AUC score:           {}"
                .format(roc_auc_score(label, pre)))
        print("    Sensitivity:             {}".format(sensitivity))
        print("    Specificity:             {}".format(specificity))
        print("    Precision:               {}".format(precision))
        print("    F-measure:               {}".format(fmeasure))\

        return accuracy

def classifySplit(model, nnInput, rawLabel, labelIndexes = LABEL_IDXS):
    partSize = len(nnInput) / NUM_PARTITIONS
    loglosses = np.zeros((len(labelIndexes), NUM_PARTITIONS))

    accuracies = np.zeros((len(labelIndexes), NUM_PARTITIONS))

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

        metricidx = 0
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
                pre = data.zScoreDenormalize(pre, zMean[labidx], zDev[labidx])
                label = data.zScoreDenormalize(label, zMean[labidx], zDev[labidx])

            if LOGARITHM:
                for j in range(len(pre)):
                    pre[j] = exp(-pre[j])
                    label[j] = exp(-label[j])

            # print samples of predictions
            for j in range(min(PREDICT_PRINT_SAMPLES / NUM_PARTITIONS, len(pre))):
                print("        prediction: {}, label: {}".format(pre[j],label[j]))


            falseNegative = 0.0
            falsePositive = 0.0
            truePositive  = 0.0
            trueNegative  = 0.0

            for j in range(len(pre)):
                if j < PREDICT_PRINT_SAMPLES:
                    print "    Predicted: {} Label: {}".format(pre[j], label[j])
                if pre[j] < CLASSIFY_THRESHOLD and label[j] == CLASSIFY_LABEL_POS:
                    falseNegative += 1
                elif pre[j] > CLASSIFY_THRESHOLD and label[j] == CLASSIFY_LABEL_NEG:
                    falsePositive += 1
                elif pre[j] > CLASSIFY_THRESHOLD and label[j] == CLASSIFY_LABEL_POS:
                    truePositive += 1
                elif pre[j] < CLASSIFY_THRESHOLD and label[j] == CLASSIFY_LABEL_NEG:
                    trueNegative += 1

            loglosses[metricidx][i] = utility.logloss(pre,label)
            accuracies[metricidx][i] = 1 - (falseNegative + falsePositive) / len(label)

            print '      Logloss Value: {}'.format(loglosses[metricidx][i])
            print '      Accuracy: {}'.format(accuracies[metricidx][i])

            metricidx += 1

    print '\n    Classification Statistics:'
    loglossAvg = []
    loglossDev = []

    accuracyAvg = []
    accuracyDev = []

    for lab in range(len(loglosses)):
        loglossAvg.append(np.sum(loglosses[lab]) / NUM_PARTITIONS)
        accuracyAvg.append(np.sum(accuracies[lab]) / NUM_PARTITIONS)

        loglossDev.append(0.0)
        accuracyDev.append(0.0)
        for i in range(NUM_PARTITIONS):
            loglossDev[lab] += (loglosses[lab][i] - loglossAvg[lab])**2
            accuracyDev[lab] += (accuracies[lab][i] - accuracyAvg[lab])**2
        loglossDev[lab] = sqrt(loglossDev[lab] / NUM_PARTITIONS)
        accuracyDev[lab] = sqrt(accuracyDev[lab] / NUM_PARTITIONS)

        print '      label {} Logloss Average:   {}'.format(lab, loglossAvg[lab])
        print '      label {} Logloss Deviation: {}'.format(lab, loglossDev[lab])
        print '      label {} Accuracy Average:   {}'.format(lab, accuracyAvg[lab])
        print '      label {} Accuracy Deviation: {}'.format(lab, accuracyDev[lab])


    loglossAvgOverall = utility.mean(loglossAvg, len(loglossAvg))
    loglossDevOverall = utility.mean(loglossDev, len(loglossDev))
    accuracyAvgOverall = utility.mean(accuracyAvg, len(accuracyAvg))
    accuracyDevOverall = utility.mean(accuracyDev, len(accuracyDev))

    print '\n  Logloss mean of avgs: {}'.format(loglossAvgOverall)
    print '  Logloss mean of devs: {}'.format(loglossDevOverall)
    print '\n  Accuracy mean of avgs: {}'.format(accuracyAvgOverall)
    print '  Accuracy mean of devs: {}'.format(accuracyDevOverall)
    return accuracyAvgOverall, accuracyDevOverall


# TODO: encapsulate training rnn on a label to a function, not working yet
def modelOnLabels(trainIn, trainLabel, testIn, testLabel, alphaSize, nomiSize,
        indexes, weights = None, uniOutput = False):
    model = configureModel(alphaSize, nomiSize, outputLen = len(indexes))

    if uniOutput:
        if weights == None:
            weights = model.get_weights()
        # uniform output layer weights
        for i in range(len(weights[11])):
            for j in range(len(weights[11][i])):
                weights[11][i][j] = 0.1
        model.set_weights(weights)
    elif weights != None:
        # preserve randomized weights
        oriW = model.get_weights()
        weights[11] = oriW[11]
        model.set_weights(weights)

    epochsDone = train(model, trainIn, trainLabel, (testIn, testLabel),
            labelIndexes = indexes)
    return model, epochsDone


# Do preprocessing and train/test data division
def preprocess(fullIn, labels, testFlags):

    if LOGARITHM:
        for idx in LABEL_IDXS:
            labels[idx] = data.logarithm(labels[idx])

    global zMean, zDev
    zMean = {}
    zDev = {}
    if ZSCORE_NORM:
        for idx in LABEL_IDXS:
            labels[idx], m, d = data.zScoreNormalize(labels[idx])
            zMean[idx] = m
            zDev[idx] = d



    # check for NaN or inf values, which break our RNN
    for idx in LABEL_IDXS:
        for label in labels[idx]:
            if np.isnan(label) or np.isinf(label):
                raise ValueError('Preprocess error: bad value in data: {}'.format(label))


    if FLAG_BASED_HOLD:
        trainIn, trainLabel, testIn, testLabel = data.holdoutBased(testFlags,
                fullIn, labels)
    else:
        trainIn, trainLabel, testIn, testLabel = data.holdout(HOLDOUT_RATIO,
                fullIn, labels)


    return trainIn, trainLabel, testIn, testLabel


def run(grid = None):
    startTime = time.time()

    # Initialize using the same seed (to get stable results on comparisons)
    np.random.seed(SEED)

    fullIn, labels, alphaSize, nomiSize, testFlags = data.prepareData()

    trainIn, trainLabel, testIn, testLabel = preprocess(fullIn, labels,
            testFlags)

    if not CHAINED_MODELS:
        model = configureModel(alphaSize, nomiSize)

        epochsDone = train(model, trainIn, trainLabel, (testIn, testLabel))

        if SCATTER_VISUALIZE:
            utility.visualize2D(model, 1, testIn, testLabel[LABEL_IDXS[0]])

        print("\n  Prediction of training data:")
        if CLASSIFY:
            relevanceTrain = classify(model, trainIn, trainLabel)
        else:
            relevanceTrain = predict(model, trainIn, trainLabel)

        print("\n  Prediction of testing data:")
        if CLASSIFY:
            if USE_PARTITIONS:
                relevanceTest, stdTest = classifySplit(model, testIn, testLabel)
            else:
                relevanceTest = classify(model, testIn, testLabel)
        else:
            if USE_PARTITIONS:
                relevanceTest, stdTest = predictSplit(model, testIn, testLabel)
            else:
                relevanceTest = predict(model, testIn, testLabel)
    else:
        for idx in range(len(CHAINED_LABELS)):
            if idx in FREEZE_IDXS:
                print '    Freezing inner layers.'
                TRAINABLE_INNER = False

            if idx == 0:
                model, epochsDone = modelOnLabels(trainIn, trainLabel, testIn, testLabel,
                    alphaSize, nomiSize, CHAINED_LABELS[idx], uniOutput =
                    True)
            elif idx == len(CHAINED_LABELS) - 1:
                model, epochsDone = modelOnLabels(trainIn, trainLabel, testIn,
                    testLabel, alphaSize, nomiSize, CHAINED_LABELS[idx],
                    model.get_weights(), uniOutput = True)
            else:
                model, epochsDone = modelOnLabels(trainIn, trainLabel, testIn,
                    testLabel, alphaSize, nomiSize, CHAINED_LABELS[idx],
                    model.get_weights())

            print("\n  Prediction of training data:")
            if CLASSIFY:
                relevanceTrain = classify(model, trainIn, trainLabel,
                        labelIndexes = CHAINED_LABELS[idx])
            else:
                relevanceTrain = predict(model, trainIn, trainLabel,
                        labelIndexes = CHAINED_LABELS[idx])

            print("\n  Prediction of testing data:")
            if CLASSIFY:
                if USE_PARTITIONS:
                    relevanceTest, stdTest = classifySplit(model, testIn, testLabel,
                        labelIndexes = CHAINED_LABELS[idx])
                else:
                    relevanceTest = classify(model, testIn, testLabel, labelIndexes
                        = CHAINED_LABELS[idx])
            else:
                if USE_PARTITIONS:
                    relevanceTest, stdTest = predictSplit(model, testIn, testLabel,
                        labelIndexes = CHAINED_LABELS[idx])
                else:
                    relevanceTest = predict(model, testIn, testLabel, labelIndexes
                        = CHAINED_LABELS[idx])

        if SCATTER_VISUALIZE:
            utility.visualize2D(model, 1, testIn,
                    testLabel[CHAINED_PREDICT[0]])

    endTime = time.time()
    deltaTime = endTime - startTime

    if CLASSIFY:
        taskType = 'classification'
    else:
        taskType = 'regression'
    modelSummary = utility.modelToString(model)

    memRss, memVms = utility.getMemoryUsage()

    if np.isnan(relevanceTest):
        relevanceTest = -1
    if np.isnan(relevanceTrain):
        relevanceTrain = -1
    if np.isnan(stdTest):
        stdTest = -1

    # TODO: add memory_pm_mb, memory_vm_bm

    db.sendStatistics(
        dataset_name = cc.exp['fetch']['table'],
        split_name = cc.exp['params']['data']['testing'],
        training_row_count = len(trainLabel[0]),
        testing_row_count = len(testLabel[0]),
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
        label_name = ','.join(cc.exp['params']['data']['labels']),
        model = modelSummary,
        seed = SEED,
        memory_pm_mb = memRss,
        memory_vm_mb = memVms,
        learning_curve = open('local/plots/{}'.format(utility.PLOT_NAME),
            'rb').read(),
        hostname = socket.gethostname())
