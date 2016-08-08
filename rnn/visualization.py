import keras.callbacks

import numpy as np
import matplotlib.pyplot as plt
import collections

from config import config as cc
import os

def flatten(x):
    if isinstance(x, collections.Iterable):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]

class ModelLogger(keras.callbacks.Callback):

    def __init__(self):
        self.epochLogs = []
        self.batchLog = []
        self.loggedLayers = []

    def _set_params(self, params):
        self.params = params

    def _set_model(self, model):
        self.model = model

    def on_epoch_begin(self, epoch, logs={}):
        epochLog = {
            'epoch': epoch,
            'weights': []
        }

        self.epochLogs.append(epochLog)

    def on_epoch_end(self, epoch, logs={}):
        self.epochLogs[-1]['weights'] = self.getWeights()

    def on_batch_begin(self, batch, logs={}):
        batchDict = {
            'epoch': self.epochLogs[-1]['epoch'],
            'weights': []
        }

        self.batchLog.append(batchDict)

    def on_batch_end(self, batch, logs={}):
        self.batchLog[-1]['weights'] = self.getWeights()

    # log initial weights and stuff
    def on_train_begin(self, logs={}):
        self.on_epoch_begin(0)
        self.on_epoch_end(0)

        for i,layer in enumerate(self.model.layers):
            if len(layer.get_weights()) != 0:
                self.loggedLayers.append(i)


    # compute everything from logs
    def on_train_end(self, logs={}):
        # transform epochlog weights to flattened numpy arrays
        for i,epochLog in enumerate(self.epochLogs):
            newWeights = []
            for layerWeights in epochLog['weights']:
                newWeights.append(np.array(flatten(layerWeights)))

            self.epochLogs[i]['weights'] = np.array(newWeights)

        # compute updates
        self.epochLogs[0]['updates'] = []
        for i in range(1,len(self.epochLogs)):
            newUpdates = []
            for x in range(len(self.epochLogs[i]['weights'])):
                newUpdates.append(self.epochLogs[i-1]['weights'][x] - self.epochLogs[i]['weights'][x])

            self.epochLogs[i]['updates'] = np.array(newUpdates)

    def getWeights(self):
        weights = []
        for layer in self.model.layers:
            if len(layer.get_weights()) == 0:
                continue
            weights.append(layer.get_weights())
        return weights


def weightsHistogram(modelLogger):

    if not os.path.exists(cc.cfg['plots']['weight_histograms']):
        os.makedirs(cc.cfg['plots']['weight_histograms'])

    cntEpochs = len(modelLogger.epochLogs)
    cntLayers = len(modelLogger.epochLogs[-1]['weights'])

    for x in range(cntEpochs):
        plt.figure()
        plt.suptitle('Weights histogram per layer, epoch {}/{}'.format(x,cntEpochs-1), fontsize=14)
        for i,layerWeights in enumerate(modelLogger.epochLogs[x]['weights']):
            histmin = layerWeights.min()
            histmax = layerWeights.max()

            plt.subplot(2, cntLayers/2, i+1)
            plt.title(modelLogger.model.layers[modelLogger.loggedLayers[i]].name)
            plt.hist(layerWeights, range = (histmin, histmax), bins = 30, color = 'blue')
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.savefig('{}/weight_hist_{e:03d}.png'.format(cc.cfg['plots']['weight_histograms'], e = x))
        plt.close()


def updatesHistogram(modelLogger):

    if not os.path.exists(cc.cfg['plots']['update_histograms']):
        os.makedirs(cc.cfg['plots']['update_histograms'])

    cntEpochs = len(modelLogger.epochLogs)
    cntLayers = len(modelLogger.epochLogs[-1]['updates'])

    for x in range(1,cntEpochs):
        plt.figure()
        plt.suptitle('Updates histogram per layer, epoch {}/{}'.format(x,cntEpochs-1), fontsize=14)
        for i,layerUpdates in enumerate(modelLogger.epochLogs[x]['updates']):
            histmin = layerUpdates.min()
            histmax = layerUpdates.max()

            plt.subplot(2, cntLayers/2, i+1)
            plt.title(modelLogger.model.layers[modelLogger.loggedLayers[i]].name)
            plt.hist(layerUpdates, range = (histmin, histmax), bins = 30, color = 'red')
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.savefig('{}/weight_hist_{e:03d}.png'.format(cc.cfg['plots']['update_histograms'], e = x))
        plt.close()

