import random

import pubchem as pc

import numpy as np
import pandas as pd
import sklearn as sk

import utility

import db.db as db
from config import config as cc
import sys

from sets import Set
import data

RD = cc.exp['params']['data']
RP = cc.exp['params']['rnn']


# not entirely correct, in one partition can appear two same elements, since we are concatenating two permutations
class PermutationPartitioner:
    def __init__(self, samplesCount, partitionSize):
        self.samplesCount = samplesCount
        self.partitionSize = partitionSize
        self.permutation = np.random.permutation(samplesCount)
        self.idx = 0

    def get(self):
        part = np.copy(self.permutation[self.idx:self.idx+self.partitionSize])
        if len(part) < self.partitionSize:
            np.random.shuffle(self.permutation)
            self.idx = self.partitionSize - len(part)
            part = np.concatenate((part,self.permutation[:self.idx]))
        else:
            self.idx += self.partitionSize

        return part

def computeR2(pred, truth):
    return np.corrcoef([pred,truth])[0][1]**2

def computeMSE(pred, truth):
    return ((pred - truth)**2).mean()

def computeMAE(pred, truth):
    return (np.absolute(pred - truth)).mean()

def predict(model, input, labels, meta):
    if RP['edge_prediction']:
        partitioner = PermutationPartitioner(len(input[0]), len(input[0]) / RP['num_partitions'])
    else:
        partitioner = PermutationPartitioner(len(input), len(input) / RP['num_partitions'])
    iterations = RP['num_partitions']**2

    metrics = {
        'r2': np.zeros((labels.shape[1], iterations)),
        'mse': np.zeros((labels.shape[1], iterations)),
        'mae': np.zeros((labels.shape[1], iterations)),
    }

    # first denormalize labels, so we do it only once
    labels = data.denormalize(labels, meta)

    for iteration in range(iterations):
        print('\titer:\t{}/{}'.format(iteration+1, iterations))

        part = partitioner.get()

        if RP['edge_prediction']:
            partIn = [input[0][part],input[1][part]]
        else:
            partIn = input[part]
        partLabelsT = labels[part].T
        partPredT = model.predict(partIn, batch_size = RP['batch']).T

        for i in range(labels.shape[1]):
            metrics['r2'][i][iteration] = computeR2(partPredT[i], partLabelsT[i])
            metrics['mse'][i][iteration] = computeMSE(partPredT[i], partLabelsT[i])
            metrics['mae'][i][iteration] = computeMAE(partPredT[i], partLabelsT[i])

        del partIn
        del partLabelsT
        del partPredT

    metricsPerLabel = {
        'r2_avg': np.nanmean(metrics['r2'], axis = 1),
        'r2_std': np.nanstd(metrics['r2'], axis = 1),
        'mse_avg': np.nanmean(metrics['mse'], axis = 1),
        'mse_std': np.nanstd(metrics['mse'], axis = 1),
        'mae_avg': np.nanmean(metrics['mae'], axis = 1),
        'mae_std': np.nanstd(metrics['mae'], axis = 1),
    }

    metricsOverall = {
        'r2_avg': np.nanmean(metrics['r2']),
        'r2_std': np.nanstd(metrics['r2']),
        'mse_avg': np.nanmean(metrics['mse']),
        'mse_std': np.nanstd(metrics['mse']),
        'mae_avg': np.nanmean(metrics['mae']),
        'mae_std': np.nanstd(metrics['mae']),
    }

    for i,labelName in enumerate(RD['labels']):
        print('{}/{} - {}:'.format(i+1, len(RD['labels']),labelName))
        print('\tR2:\t{0:.3f}\t+/-\t{1:.3f}'.format(metricsPerLabel['r2_avg'][i],metricsPerLabel['r2_std'][i]))
        print('\tMSE:\t{0:.3f}\t+/-\t{1:.3f}'.format(metricsPerLabel['mse_avg'][i],metricsPerLabel['mse_std'][i]))
        print('\tMAE:\t{0:.3f}\t+/-\t{1:.3f}'.format(metricsPerLabel['mae_avg'][i],metricsPerLabel['mae_std'][i]))

    print('Overall metrics:')
    print('\tR2:\t{0:.3f}\t+/-\t{1:.3f}'.format(metricsOverall['r2_avg'],metricsOverall['r2_std']))
    print('\tMSE:\t{0:.3f}\t+/-\t{1:.3f}'.format(metricsOverall['mse_avg'],metricsOverall['mse_std']))
    print('\tMAE:\t{0:.3f}\t+/-\t{1:.3f}'.format(metricsOverall['mae_avg'],metricsOverall['mae_std']))

    return metricsOverall


def computeConfusion(pred, truth):
    #          pred_pos  pred_neg
    # true_pos   TP         FN
    # true_neg   FP         TN
    confusion = np.zeros((2,2))

    thr,pos,neg = RP['classify_threshold'],RP['classify_label_pos'],RP['classify_label_neg']

    for i in range(len(pred)):
        if pred[i] < thr:
            if utility.equals(truth[i], pos):
                confusion[0][1]+=1
            elif utility.equals(truth[i], neg):
                confusion[1][1]+=1
        elif pred[i] >= thr:
            if utility.equals(truth[i], pos):
                confusion[0][0]+=1
            elif utility.equals(truth[i], neg):
                confusion[1][0]+=1
    return confusion

def computeAUC(pred, truth):
    try:
        return sk.metrics.roc_auc_score(truth, pred)
    except:
        return np.nan


def classify(model, input, labels, meta):
    if RP['edge_prediction']:
        partitioner = PermutationPartitioner(len(input[0]), len(input[0]) / RP['num_partitions'])
    else:
        partitioner = PermutationPartitioner(len(input), len(input) / RP['num_partitions'])
    iterations = RP['num_partitions']**2

    metrics = {
        'acc': np.zeros((labels.shape[1], iterations)),
        'log_loss': np.zeros((labels.shape[1], iterations)),
        'auc': np.zeros((labels.shape[1], iterations)),
        'confusion': np.zeros((labels.shape[1], iterations, 2, 2)),
    }

    # first denormalize labels, so we do it only once
    labels = data.denormalize(labels, meta)

    for iteration in range(iterations):
        print('\titer:\t{}/{}'.format(iteration, iterations))

        part = partitioner.get()

        if RP['edge_prediction']:
            partIn = [input[0][part],input[1][part]]
        else:
            partIn = input[part]
        partLabelsT = labels[part].T
        partPredT = model.predict(partIn, batch_size = RP['batch']).T

        for i in range(labels.shape[1]):
            confusion = computeConfusion(partPredT[i], partLabelsT[i])

            metrics['confusion'][i][iteration] = confusion
            metrics['acc'][i][iteration] = (confusion[0][0]+confusion[1][1]) / confusion.sum()
            metrics['log_loss'][i][iteration] = utility.logloss(partPredT[i],partLabelsT[i],RP['classify_label_neg'],RP['classify_label_pos'])
            metrics['auc'][i][iteration] = computeAUC(partPredT[i], partLabelsT[i])

        del partIn
        del partLabelsT
        del partPredT

    metricsPerLabel = {
        'acc_avg': np.nanmean(metrics['acc'], axis = 1),
        'acc_std': np.nanstd(metrics['acc'], axis = 1),
        'log_loss_avg': np.nanmean(metrics['log_loss'], axis = 1),
        'log_loss_std': np.nanstd(metrics['log_loss'], axis = 1),
        'auc_avg': np.nanmean(metrics['auc'], axis = 1),
        'auc_std': np.nanstd(metrics['auc'], axis = 1)
    }

    metricsOverall = {
        'acc_avg': np.nanmean(metrics['acc']),
        'acc_std': np.nanstd(metrics['acc']),
        'log_loss_avg': np.nanmean(metrics['log_loss']),
        'log_loss_std': np.nanstd(metrics['log_loss']),
        'auc_avg': np.nanmean(metrics['auc']),
        'auc_std': np.nanstd(metrics['auc'])
    }

    for i,labelName in enumerate(RD['labels']):
        print('{}/{} - {}:'.format(i+1, len(RD['labels']),labelName))
        print('\tACC:\t{0:.3f}\t+/-\t{1:.3f}'.format(metricsPerLabel['acc_avg'][i],metricsPerLabel['acc_std'][i]))
        print('\tLogLos:\t{0:.3f}\t+/-\t{1:.3f}'.format(metricsPerLabel['log_loss_avg'][i],metricsPerLabel['log_loss_std'][i]))
        print('\tAUC:\t{0:.3f}\t+/-\t{1:.3f}'.format(metricsPerLabel['auc_avg'][i],metricsPerLabel['auc_std'][i]))

    print('Overall metrics:')
    print('\tACC:\t{0:.3f}\t+/-\t{1:.3f}'.format(metricsOverall['acc_avg'],metricsOverall['acc_std']))
    print('\tLogLos:\t{0:.3f}\t+/-\t{1:.3f}'.format(metricsOverall['log_loss_avg'],metricsOverall['log_loss_std']))
    print('\tAUC:\t{0:.3f}\t+/-\t{1:.3f}'.format(metricsOverall['auc_avg'],metricsOverall['auc_std']))

    return metricsOverall


def discreteClassify(model, input, labels, meta):
    partitioner = PermutationPartitioner(len(input), len(input) / RP['num_partitions'])
    iterations = RP['num_partitions']**2

    metrics = {
        'acc': np.zeros((iterations)),
        'log_loss': np.zeros((iterations)),
        # 'auc': np.zeros((iterations)),
        # 'auc_micro': np.zeros((iterations))
    }

    # first denormalize labels, so we do it only once
    labels = data.denormalize(labels, meta)

    for iteration in range(iterations):
        # print('\titer:\t{}/{}'.format(iteration, iterations))

        part = partitioner.get()

        partIn = input[part]
        partLabels = labels[part]
        partPred = model.predict(partIn, batch_size = RP['batch'])
        binarizedPred = np.zeros((len(partPred), len(partPred[0])))

        """
        for row in range(len(partLabels)):
            for idx, val in enumerate(partLabels[row]):
                if val == 1:
                    sys.stdout.write('{}, '.format(idx))
            for val in partPred[row]:
                sys.stdout.write('{}, '.format(val))
            sys.stdout.write('\n')
            sys.stdout.flush()
        """

        for i in range(len(partPred)):
            maxValue = 0
            maxIndex = 0
            for index in range(len(partPred[i])):
                value = partPred[i][index]
                if value > maxValue:
                    maxValue = value
                    maxIndex = index
            binarizedPred[i][maxIndex] = 1

        metrics['acc'][iteration] = sk.metrics.accuracy_score(partLabels,
                binarizedPred)
        metrics['log_loss'][iteration] = sk.metrics.log_loss(partLabels,
                binarizedPred)

        '''
        keepVec = []
        for col in range(len(partLabels[0])):
            wasOne = 0
            for row in range(len(partLabels)):
                if partLabels[row][col] == 1:
                    wasOne = 1
                    break
            if wasOne:
                keepVec.append(col)

        cutLabels = np.zeros((len(partLabels), len(keepVec)))
        cutPreds  = np.zeros((len(partLabels), len(keepVec)))
        for idx, keep in enumerate(keepVec):
            for row in range(len(partLabels)):
                cutLabels[row][idx] = partLabels[row][keep]
                cutPreds[row][idx]  = binarizedPred[row][keep]

        metrics['auc'][iteration] = sk.metrics.roc_auc_score(cutLabels,
                cutPreds, average = 'macro')
        metrics['auc_micro'][iteration] = sk.metrics.roc_auc_score(cutLabels,
                cutPreds, average = 'micro')
        '''

    metricsOverall = {
        'acc_avg': np.nanmean(metrics['acc']),
        'acc_std': np.nanstd(metrics['acc']),
        'log_loss_avg': np.nanmean(metrics['log_loss']),
        'log_loss_std': np.nanstd(metrics['log_loss']),
        # 'auc_avg': np.nanmean(metrics['auc']),
        # 'auc_std': np.nanstd(metrics['auc']),
        # 'auc_micro_avg': np.nanmean(metrics['auc_micro']),
        # 'auc_micro_std': np.nanstd(metrics['auc_micro'])
        'auc_avg': None,
        'auc_std': None,
        'auc_micro_avg': None,
        'auc_micro_std': None,
    }

    print('Overall metrics:')
    print('\tACC:\t{0:.3f}\t+/-\t{1:.3f}'.format(metricsOverall['acc_avg'],metricsOverall['acc_std']))
    print('\tLogLos:\t{0:.3f}\t+/-\t{1:.3f}'.format(metricsOverall['log_loss_avg'],metricsOverall['log_loss_std']))
    # print('\tAUC:\t{0:.3f}\t+/-\t{1:.3f}'.format(metricsOverall['auc_avg'],metricsOverall['auc_std']))
    # print('\tAUC Micro:\t{0:.3f}\t+/-\t{1:.3f}'.format(metricsOverall['auc_micro_avg'],metricsOverall['auc_micro_std']))

    return metricsOverall
