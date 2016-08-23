import random

import pubchem as pc

import numpy as np
import pandas as pd
import sklearn as sk

import utility

import db.db as db
from config import config as cc

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

def predict(model, input, labels, meta):
    partitioner = PermutationPartitioner(len(input), len(input) / RP['num_partitions'])
    iterations = RP['num_partitions']**2

    metrics = {
        'r2': np.zeros((labels.shape[1], iterations)),
        'mse': np.zeros((labels.shape[1], iterations))
    }

    # first denormalize labels, so we do it only once
    labels = data.denormalize(labels, meta)

    for iteration in range(iterations):
        print('   iter: {}/{}'.format(iteration, iterations))

        part = partitioner.get()

        partIn = input[part]
        partLabels = labels[part]
        partPred = model.predict(partIn, batch_size = RP['batch'])

        for i in range(labels.shape[1]):
            metrics['r2'][i][iteration] = computeR2(partPred.T[i], partLabels.T[i])
            metrics['mse'][i][iteration] = computeMSE(partPred.T[i], partLabels.T[i])

    metricsPerLabel = {
        'r2_avg': np.nanmean(metrics['r2'], axis = 1),
        'r2_std': np.nanstd(metrics['r2'], axis = 1),
        'mse_avg': np.nanmean(metrics['mse'], axis = 1),
        'mse_std': np.nanstd(metrics['mse'], axis = 1)
    }

    metricsOverall = {
        'r2_avg': np.nanmean(metrics['r2']),
        'r2_std': np.nanstd(metrics['r2']),
        'mse_avg': np.nanmean(metrics['mse']),
        'mse_std': np.nanstd(metrics['mse']),
    }

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
        print('   iter: {}/{}'.format(iteration, iterations))

        part = partitioner.get()

        partIn = input[part]
        partLabels = labels[part]
        partPred = model.predict(partIn, batch_size = RP['batch'])

        for i in range(labels.shape[1]):
            confusion = computeConfusion(partPred.T[i], partLabels.T[i])

            metrics['confusion'][i][iteration] = confusion
            metrics['acc'][i][iteration] = (confusion[0][0]+confusion[1][1]) / confusion.sum()
            metrics['log_loss'][i][iteration] = utility.logloss(partPred.T[i],partLabels.T[i],RP['classify_label_neg'],RP['classify_label_pos'])
            metrics['auc'][i][iteration] = computeAUC(partPred.T[i], partLabels.T[i])

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

    return metricsOverall