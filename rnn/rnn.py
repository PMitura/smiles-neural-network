import data, utility
import time

import numpy as np
from math import sqrt, exp, log, ceil
from scipy.stats.stats import pearsonr
from sklearn.metrics import roc_auc_score

import db.db as db
import visualization

from config import config as cc
import yaml

import socket

# not used to relieve MetaCentrum of some dependencies

# TODO: Remove unused imports after experiments are done
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, LSTM, AveragePooling1D
from keras.layers import TimeDistributed, SimpleRNN, GRU
from keras.layers import BatchNormalization, Embedding, merge
from keras.optimizers import Adam, RMSprop, Adadelta, Adagrad
import keras.callbacks
# from keras.regularizers import l1

# handy aliases for config
RP = cc.exp['params']['rnn']
RD = cc.exp['params']['data']

# manual eval where needed
RP['chained_labels'] = eval(str(cc.exp['params']['rnn']['chained_labels']))
RP['chained_predict'] = eval(str(cc.exp['params']['rnn']['chained_predict']))
RP['chained_test_labels'] = eval(str(cc.exp['params']['rnn']['chained_test_labels']))
RP['freeze_idxs'] = eval(str(cc.exp['params']['rnn']['freeze_idxs']))
RP['label_idxs'] = eval(str(cc.exp['params']['rnn']['label_idxs']))

OPTIMIZER = Adam(lr = RP['learning_rate'])
# OPTIMIZER = Adadelta()
# OPTIMIZER = Adagrad()

def configureModel(alphaSize, nomiSize = (0, 0), outputLen = len(RP['label_idxs'])):
    print('  Initializing and compiling...')

    model = Sequential()


    if RD['use_embedding']:
        # second value in nomiSize tuple is shift while using embedding
        model.add(Embedding(1 << nomiSize[1], RP['embedding_outputs']))
        model.add(TimeDistributed(Dense(int(RP['td_layer_multiplier'] * (alphaSize +
            nomiSize[0])), activation = 'tanh', trainable = RP['trainable_inner'])))
    else:
        model.add(TimeDistributed(Dense(int(RP['td_layer_multiplier'] * (alphaSize+nomiSize)), activation = 'tanh', trainable = RP['trainable_inner']),
            input_shape = (None, alphaSize + nomiSize)))


    # model.add(GRU(int(RP['gru_layer_multiplier'] * 300), trainable = RP['trainable_inner'], return_sequences = True ))
    model.add(GRU(int(RP['gru_layer_multiplier'] * alphaSize), trainable = RP['trainable_inner']))
    model.add(Activation('relu', trainable = RP['trainable_inner']))
    model.add(Dense(outputLen) )

    if RP['classify']:
        model.add(Activation(RP['classify_activation'], trainable = RP['trainable_inner']))

    # default learning rate 0.001
    model.compile(loss = RP['objective'], optimizer = OPTIMIZER)

    print('  ...done')
    return model


def learningRateDecayer(epoch):
    if not RP['learning_rate_decay']:
        return RP['learning_rate']

    if RP['learning_rate_decay_type'] == 'step':
        drop = np.floor((epoch)/RP['learning_rate_decay_step_config_steps'])
        new_lr = float(RP['learning_rate'] * np.power(RP['learning_rate_decay_step_config_ratio'],drop))
        print('lr',epoch,new_lr)
        return new_lr
    elif RP['learning_rate_decay_type'] == 'time':
        raise NotImplementedError('learning rate decay: time')
    elif RP['learning_rate_decay_type'] == 'peter':
        raise NotImplementedError('learning rate decay: peter')
    else:
        raise RuntimeError('learning rate decat: unknown type {}'.format(RP['learning_rate_decay_type']))


def train(model, nnInput, labels, validation, makePlot = True,
        labelIndexes = RP['label_idxs']):
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

    early = keras.callbacks.EarlyStopping(monitor = 'val_loss',
            patience = RP['early_stop'])

    learningRateScheduler = keras.callbacks.LearningRateScheduler(learningRateDecayer)

    modelLogger = visualization.ModelLogger()

    history = model.fit(nnInput, formattedLabels, nb_epoch = RP['epochs'],
            batch_size = RP['batch'], callbacks = [learningRateScheduler,modelLogger],
            validation_data = (validation[0], formattedValid))

    if makePlot:
        values = np.zeros((len(history.history['loss']), 2))
        for i in range(len(history.history['loss'])):
            values[i][0] = history.history['loss'][i]
            values[i][1] = history.history['val_loss'][i]
        utility.plotLoss(values)

    visualization.histograms(modelLogger)

    print('    Model weights:')
    print(model.summary())
    # print(model.get_weights())
    print('  ...done')
    return len(history.history['loss'])


# Serves as extended version of test, gives statistics
def predict(model, nnInput, rawLabel, labelIndexes = RP['label_idxs']):
    preRaw = model.predict(nnInput, batch_size = RP['batch'])

    for labCtr,labidx in enumerate(labelIndexes):
        print '  Predictions for label {}'.format(labidx)

        pre = []
        label = []
        for i in range(len(preRaw)):
            pre.append(preRaw[i][labCtr])
            label.append(rawLabel[labidx][i])

        # temporarily undo z-score normalization, if applied
        if RP['zscore_norm']:
            pre = data.zScoreDenormalize(pre, zMean[labidx], zDev[labidx])
            label = data.zScoreDenormalize(label, zMean[labidx], zDev[labidx])

        if RP['logarithm']:
            for i in range(len(pre)):
                pre[i] = exp(-pre[i])
                label[i] = exp(-label[i])

        # print samples of predictions
        for i in range(min(RP['predict_print_samples'], len(pre))):
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
def predictSplit(model, nnInput, rawLabel, labelIndexes = RP['label_idxs']):
    partSize = len(nnInput) / RP['num_partitions']
    rsqrs = np.zeros((len(labelIndexes), RP['num_partitions']))
    for i in range(RP['num_partitions']):
        print '\n    Partition {}'.format(i)
        if RD['use_embedding']:
            partInput = np.zeros([partSize, len(nnInput[0])])
        else:
            partInput = np.zeros([partSize, len(nnInput[0]), len(nnInput[0][0])])
        base = i * partSize
        for j in range(base, base + partSize):
            partInput[j - base] = nnInput[j]

        preRaw = model.predict(partInput, batch_size = RP['batch'])

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

            if RP['zscore_norm']:
                pre = data.zScoreDenormalize(pre, zMean[labidx], zDev[labidx])
                label = data.zScoreDenormalize(label, zMean[labidx], zDev[labidx])

            if RP['logarithm']:
                for j in range(len(pre)):
                    pre[j] = exp(-pre[j])
                    label[j] = exp(-label[j])

            # print samples of predictions
            for j in range(min(RP['predict_print_samples'] / RP['num_partitions'], len(pre))):
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
        rsqrAvg.append(rsqrSum / RP['num_partitions'])

        rsqrDev.append(0.0)
        for i in range(RP['num_partitions']):
            rsqrDev[lab] += (rsqrs[lab][i] - rsqrAvg[lab]) * (rsqrs[lab][i] -
                    rsqrAvg[lab])
        rsqrDev[lab] = sqrt(rsqrDev[lab] / RP['num_partitions'])

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
# pos class defined by RP['classify_label_pos']
# neg class defined by RP['classify_label_neg']
# decision threshold defined by RP['classify_threshold']
# activation function defined by RP['classify_activation']
def classify(model, nnInput, rawLabel, labelIndexes = RP['label_idxs']):

    preRaw = model.predict(nnInput, batch_size = RP['batch'])

    for labCtr, labidx in enumerate(labelIndexes):
        print '  Predictions for label {}'.format(labidx)

        pre = []
        label = []
        for i in range(len(preRaw)):
            pre.append(preRaw[i][0])
            label.append(rawLabel[labidx][i])
        if len(pre) <= 0:
            raise ValueError('Cannot predict on zero or negative size set')

        falseNegative = 0.0
        falsePositive = 0.0
        truePositive  = 0.0
        trueNegative  = 0.0

        for i in range(len(pre)):
            if i < RP['predict_print_samples']:
                print "    Predicted: {} Label: {}".format(pre[i], label[i])
            if pre[i] < RP['classify_threshold'] and utility.equals(label[i],
                    RP['classify_label_pos']):
                falseNegative += 1
            elif pre[i] > RP['classify_threshold'] and utility.equals(label[i],
                    RP['classify_label_neg']):
                falsePositive += 1
            elif pre[i] > RP['classify_threshold'] and utility.equals(label[i],
                    RP['classify_label_pos']):
                truePositive += 1
            elif pre[i] < RP['classify_threshold'] and utility.equals(label[i],
                    RP['classify_label_neg']):
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

        if truePositive + falseNegative != 0:
            sensitivity = truePositive / (truePositive + falseNegative)
        else:
            sensitivity = np.nan
        if trueNegative + falsePositive != 0:
            specificity = trueNegative / (trueNegative + falsePositive)
        else:
            specificity = np.nan
        if truePositive + trueNegative != 0:
            precision = truePositive / (truePositive + trueNegative)
        else:
            precision = np.nan
        accuracy = (1 - (errors / len(pre))) # sanitized earlier
        if precision + sensitivity != 0:
            fmeasure = (2 * precision * sensitivity) / (precision + sensitivity)
        else:
            fmeasure = np.nan

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
        # Temporarily disabled, not working because of wrong label format
        # print("    ROC AUC score:           {}".format(roc_auc_score(label, pre)))
        print("    Sensitivity:             {}".format(sensitivity))
        print("    Specificity:             {}".format(specificity))
        print("    Precision:               {}".format(precision))
        print("    F-measure:               {}".format(fmeasure))\

        return accuracy

def classifySplit(model, nnInput, rawLabel, labelIndexes = RP['label_idxs']):
    partSize = len(nnInput) / RP['num_partitions']
    loglosses = np.zeros((len(labelIndexes), RP['num_partitions']))
    accuracies = np.zeros((len(labelIndexes), RP['num_partitions']))
    aucs = np.zeros((len(labelIndexes), RP['num_partitions']))

    # print(rawLabel)

    for i in range(RP['num_partitions']):
        print '\n    Partition {}'.format(i)
        if RD['use_embedding']:
            partInput = np.zeros([partSize, len(nnInput[0])])
        else:
            partInput = np.zeros([partSize, len(nnInput[0]), len(nnInput[0][0])])
        base = i * partSize
        for j in range(base, base + partSize):
            partInput[j - base] = nnInput[j]

        preRaw = model.predict(partInput, batch_size = RP['batch'])

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

            # print samples of predictions
            for j in range(min(RP['predict_print_samples'] / RP['num_partitions'], len(pre))):
                print("        prediction: {}, label: {}".format(pre[j],label[j]))

            falseNegative = 0.0
            falsePositive = 0.0
            truePositive  = 0.0
            trueNegative  = 0.0

            for j in range(len(pre)):
                if j < RP['predict_print_samples']:
                    print "    Predicted: {} Label: {}".format(pre[j], label[j])
                if pre[j] < RP['classify_threshold'] and utility.equals(label[j],
                        RP['classify_label_pos']):
                    falseNegative += 1
                elif pre[j] > RP['classify_threshold'] and utility.equals(label[j],
                        RP['classify_label_neg']):
                    falsePositive += 1
                elif pre[j] > RP['classify_threshold'] and utility.equals(label[j],
                        RP['classify_label_pos']):
                    truePositive += 1
                elif pre[j] < RP['classify_threshold'] and utility.equals(label[j],
                        RP['classify_label_neg']):
                    trueNegative += 1

            loglosses[metricidx][i] = utility.logloss(pre,label)
            accuracies[metricidx][i] = 1 - (falseNegative + falsePositive) / len(label)


            # we need to normalize the confidences to [0,1]?
            try:
                aucs[metricidx][i] = roc_auc_score(label, pre)
            except:
                aucs[metricidx][i] = np.nan

            print '      Logloss Value: {}'.format(loglosses[metricidx][i])
            print '      Accuracy: {}'.format(accuracies[metricidx][i])
            print '      AUC: {}'.format(aucs[metricidx][i])

            metricidx += 1

    print '\n    Classification Statistics:'
    loglossAvg = []
    loglossDev = []

    accuracyAvg = []
    accuracyDev = []

    aucAvg = []
    aucDev = []

    for lab in range(len(loglosses)):
        loglossAvg.append(np.sum(loglosses[lab]) / RP['num_partitions'])
        accuracyAvg.append(np.sum(accuracies[lab]) / RP['num_partitions'])
        aucAvg.append(np.sum(aucs[lab]) / RP['num_partitions'])

        loglossDev.append(0.0)
        accuracyDev.append(0.0)
        aucDev.append(0.0)
        for i in range(RP['num_partitions']):
            loglossDev[lab] += (loglosses[lab][i] - loglossAvg[lab])**2
            accuracyDev[lab] += (accuracies[lab][i] - accuracyAvg[lab])**2
            aucDev[lab] += (aucs[lab][i] - aucAvg[lab])**2
        loglossDev[lab] = sqrt(loglossDev[lab] / RP['num_partitions'])
        accuracyDev[lab] = sqrt(accuracyDev[lab] / RP['num_partitions'])
        aucDev[lab] = sqrt(aucDev[lab] / RP['num_partitions'])

        print '      label {} Logloss Average:   {}'.format(lab, loglossAvg[lab])
        print '      label {} Logloss Deviation: {}'.format(lab, loglossDev[lab])
        print '      label {} Accuracy Average:   {}'.format(lab, accuracyAvg[lab])
        print '      label {} Accuracy Deviation: {}'.format(lab, accuracyDev[lab])
        print '      label {} AUC Average:   {}'.format(lab, aucAvg[lab])
        print '      label {} AUC Deviation: {}'.format(lab, aucDev[lab])


    loglossAvgOverall = utility.mean(loglossAvg, len(loglossAvg))
    loglossDevOverall = utility.mean(loglossDev, len(loglossDev))
    accuracyAvgOverall = utility.mean(accuracyAvg, len(accuracyAvg))
    accuracyDevOverall = utility.mean(accuracyDev, len(accuracyDev))
    aucAvgOverall = utility.mean(aucAvg, len(aucAvg))
    aucDevOverall = utility.mean(aucDev, len(aucDev))

    print '\n  Logloss mean of avgs: {}'.format(loglossAvgOverall)
    print '  Logloss mean of devs: {}'.format(loglossDevOverall)
    print '\n  Accuracy mean of avgs: {}'.format(accuracyAvgOverall)
    print '  Accuracy mean of devs: {}'.format(accuracyDevOverall)
    print '\n  AUC mean of avgs: {}'.format(aucAvgOverall)
    print '  AUC mean of devs: {}'.format(aucDevOverall)
    return accuracyAvgOverall, accuracyDevOverall, loglossAvgOverall, loglossDevOverall, aucAvgOverall, aucDevOverall


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
        oriW = model.get_weights()
        weights[11] = oriW[11]
        model.set_weights(weights)

    epochsDone = train(model, trainIn, trainLabel, (testIn, testLabel),
            labelIndexes = indexes)
    return model, epochsDone


# Do preprocessing and train/test data division
def preprocess(fullIn, labels, testFlags):

    if RP['logarithm']:
        for idx in RP['label_idxs']:
            labels[idx] = data.logarithm(labels[idx])

    global zMean, zDev
    zMean = {}
    zDev = {}
    if RP['zscore_norm']:
        for idx in RP['label_idxs']:
            labels[idx], m, d = data.zScoreNormalize(labels[idx])
            zMean[idx] = m
            zDev[idx] = d

    # check for NaN or inf values, which break our RNN
    for idx in RP['label_idxs']:
        for label in labels[idx]:
            if np.isnan(label) or np.isinf(label):
                raise ValueError('Preprocess error: bad value in data: {}'
                        .format(label))

    if RP['label_binning'] and RP['classify']:
        for idx in RP['label_idxs']:
            labels[idx] = utility.bin(labels[idx], RP['label_binning_ratio'],
                    classA = RP['classify_label_neg'], classB = RP['classify_label_pos'])

    if RP['flag_based_hold']:
        trainIn, trainLabel, testIn, testLabel = data.holdoutBased(testFlags,
                fullIn, labels)
    else:
        trainIn, trainLabel, testIn, testLabel = data.holdout(RP['holdout_ratio'],
                fullIn, labels)

    return trainIn, trainLabel, testIn, testLabel


def run(grid = None):
    startTime = time.time()

    # Initialize using the same seed (to get stable results on comparisons)
    np.random.seed(RP['seed'])

    fullIn, labels, alphaSize, nomiSize, testFlags = data.prepareData()

    trainIn, trainLabel, testIn, testLabel = preprocess(fullIn, labels, testFlags)

    if not RP['chained_models']:
        model = configureModel(alphaSize, nomiSize)
        epochsDone = train(model, trainIn, trainLabel, (testIn, testLabel))

        if RP['classify']:
            if RP['label_binning_after_train'] and not RP['label_binning']:
                for idx in RP['label_idxs']:
                    trainLabel[idx] = utility.bin(trainLabel[idx], RP['label_binning_ratio'],
                            classA = RP['classify_label_neg'],
                            classB = RP['classify_label_pos'])
                    testLabel[idx] = utility.bin(testLabel[idx], RP['label_binning_ratio'],
                            classA = RP['classify_label_neg'],
                            classB = RP['classify_label_pos'])
                model, epochsDone = modelOnLabels(trainIn, trainLabel, testIn, testLabel,
                    alphaSize, nomiSize, [0], model.get_weights())
            elif not RP['label_binning_after_train'] and not RP['label_binning']:
                for idx in RP['label_idxs']:
                    testLabel[idx] = utility.bin(testLabel[idx], RP['label_binning_ratio'],
                            classA = RP['classify_label_neg'],
                            classB = RP['classify_label_pos'])

        if RP['scatter_visualize']:
            utility.visualize2D(model, 1, testIn, testLabel[RP['label_idxs'][0]])

        print("\n  Prediction of training data:")
        if RP['classify']:
            relevanceTrain = classify(model, trainIn, trainLabel)

        else:
            relevanceTrain = predict(model, trainIn, trainLabel)

        print("\n  Prediction of testing data:")
        if RP['classify']:
            if RP['use_partitions']:
                relevanceTest, stdTest, loglossTest, loglossStdTest, aucTest, aucStdTest = classifySplit(model, testIn, testLabel)
            else:
                relevanceTest = classify(model, testIn, testLabel)
        else:
            if RP['use_partitions']:
                relevanceTest, stdTest = predictSplit(model, testIn, testLabel)
            else:
                relevanceTest = predict(model, testIn, testLabel)
    else:
        model = None
        for idx in range(len(RP['chained_labels'])):
            if idx in RP['freeze_idxs']:
                print '    Freezing inner layers.'
                RP['trainable_inner'] = False
            else:
                RP['trainable_inner'] = True

            if idx == 0:
                model, epochsDone = modelOnLabels(trainIn, trainLabel, testIn, testLabel,
                    alphaSize, nomiSize, RP['chained_labels'][idx], uniOutput = True)
            elif idx == len(RP['chained_labels']) - 1:
                model, epochsDone = modelOnLabels(trainIn, trainLabel, testIn,
                    testLabel, alphaSize, nomiSize, RP['chained_labels'][idx],
                    model.get_weights(), uniOutput = True)
            else:
                model, epochsDone = modelOnLabels(trainIn, trainLabel, testIn,
                    testLabel, alphaSize, nomiSize, RP['chained_labels'][idx],
                    model.get_weights())

            print("\n  Prediction of training data:")
            if RP['classify']:
                relevanceTrain = classify(model, trainIn, trainLabel,
                        labelIndexes = RP['chained_labels'][idx])
            else:
                relevanceTrain = predict(model, trainIn, trainLabel,
                        labelIndexes = RP['chained_labels'][idx])

            print("\n  Prediction of testing data:")
            if RP['classify']:
                if RP['use_partitions']:
                    relevanceTest, stdTest, loglossTest, loglossStdTest, \
                        aucTest, aucStdTest = classifySplit(model, \
                        testIn, testLabel, labelIndexes = RP['chained_labels'][idx])
                else:
                    relevanceTest = classify(model, testIn, testLabel, labelIndexes
                        = RP['chained_labels'][idx])
            else:
                if RP['use_partitions']:
                    relevanceTest, stdTest = predictSplit(model, testIn, testLabel,
                        labelIndexes = RP['chained_labels'][idx])
                else:
                    relevanceTest = predict(model, testIn, testLabel, labelIndexes
                        = RP['chained_labels'][idx])

        # Permafreeze for last training. Not sure if good idea.
        RP['trainable_inner'] = False

        # Train and test on split testing data
        nTrainIn, nTrainLabel, nTestIn, nTestLabel = data.holdout(RP['holdout_ratio'],
                testIn, testLabel)
        for idxes in RP['chained_test_labels']:
            model, epochsDone = modelOnLabels(nTrainIn, nTrainLabel, nTestIn,
                nTestLabel, alphaSize, nomiSize, idxes,
                model.get_weights(), uniOutput = True)

            print("\n  Prediction of training data:")
            if RP['classify']:
                relevanceTrain = classify(model, nTrainIn, nTrainLabel,
                        labelIndexes = idxes)
            else:
                relevanceTrain = predict(model, nTrainIn, nTrainLabel,
                        labelIndexes = idxes)

            print("\n  Prediction of testing data:")
            if RP['classify']:
                if RP['use_partitions']:
                    relevanceTest, stdTest, loglossTest, loglossStdTest, \
                        aucTest, aucStdTest = classifySplit(model, nTestIn, \
                        nTestLabel, labelIndexes = idxes)
                else:
                    relevanceTest = classify(model, nTestIn, nTestLabel,
                        labelIndexes = idxes)
            else:
                if RP['use_partitions']:
                    relevanceTest, stdTest = predictSplit(model, nTestIn,
                        nTestLabel, labelIndexes = idxes)
                else:
                    relevanceTest = predict(model, nTestIn, nTestLabel,
                        labelIndexes = idxes)

        if RP['scatter_visualize']:
            utility.visualize2D(model, 1, testIn,
                    testLabel[RP['chained_predict'][0]])

    endTime = time.time()
    deltaTime = endTime - startTime

    modelSummary = utility.modelToString(model)

    memRss, memVms = utility.getMemoryUsage()

    if np.isnan(relevanceTest):
        relevanceTest = -1
    if np.isnan(relevanceTrain):
        relevanceTrain = -1
    if np.isnan(stdTest):
        stdTest = -1

    import subprocess
    try:
        gitCommit = subprocess.check_output('git rev-parse HEAD', shell=True).strip()
    except:
        gitCommit = None


    # TODO: add memory_pm_mb, memory_vm_bm

    if not RP['classify']:
        loglossTest = None
        loglossStdTest = None
        aucTest = None
        aucStdTest = None

    db.sendStatistics(
        dataset_name = cc.exp['fetch']['table'],
        split_name = cc.exp['params']['data']['testing'],
        training_row_count = len(trainLabel[0]),
        testing_row_count = len(testLabel[0]),
        task = 'classification' if RP['classify'] else 'regression',
        relevance_training = relevanceTrain,
        relevance_testing = relevanceTest,
        relevance_testing_std = stdTest,
        log_loss = loglossTest,
        log_loss_std = loglossStdTest,
        auc = aucTest,
        auc_std = aucStdTest,
        epoch_max = RP['epochs'],
        epoch_count = epochsDone,
        runtime_second = deltaTime,
        parameter_count = model.count_params(),
        learning_rate = RP['learning_rate'],
        optimization_method = OPTIMIZER.__class__.__name__,
        batch_size = RP['batch'],
        comment = RP['comment'],
        label_name = ','.join(cc.exp['params']['data']['labels']),
        model = modelSummary,
        seed = RP['seed'],
        memory_pm_mb = memRss,
        memory_vm_mb = memVms,
        learning_curve = {'val':open('{}/{}'.format(cc.cfg['plots']['dir'], utility.PLOT_NAME),'rb').read(),'type':'bin'},
        hostname = socket.gethostname(),
        experiment_config = yaml.dump(cc.exp,default_flow_style=False),
        git_commit = gitCommit,
        objective = RP['objective'])
