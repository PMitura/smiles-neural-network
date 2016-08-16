import keras.callbacks
from keras import backend as K

import numpy as np
import matplotlib.pyplot as plt
import collections
import pandas as pd

from sklearn.manifold import TSNE


from config import config as cc
import os
from math import ceil

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
        self.epochLogs[0]['ratios'] = []
        for i in range(1,len(self.epochLogs)):
            newUpdates = []
            newRatios = []
            for x in range(len(self.epochLogs[i]['weights'])):
                newUpdates.append(self.epochLogs[i-1]['weights'][x] - self.epochLogs[i]['weights'][x])
                newRatios.append(newUpdates[-1] / self.epochLogs[i]['weights'][x])

            self.epochLogs[i]['updates'] = np.array(newUpdates)
            self.epochLogs[i]['ratios'] = np.array(newRatios)

    def getWeights(self):
        weights = []
        for layer in self.model.layers:
            if len(layer.get_weights()) == 0:
                continue
            weights.append(layer.get_weights())
        return weights


def histograms(modelLogger):
    if not cc.cfg['plots']['histograms']:
        return

    if not os.path.exists(cc.cfg['plots']['histograms_dir']):
        os.makedirs(cc.cfg['plots']['histograms_dir'])

    cntEpochs = len(modelLogger.epochLogs)
    cntLayers = len(modelLogger.epochLogs[-1]['weights'])

    logVals = [
        {'name':'weights','color':'blue'},
        {'name':'updates','color':'red'},
        {'name':'ratios','color':'green'}
    ]

    subplotRows = len(logVals)
    subplotCols = cntLayers

    for x in range(cntEpochs):
        subplotIdx = 1

        plt.figure(figsize=(5*subplotCols,5*subplotRows))
        plt.suptitle('Histograms per layer, epoch {}/{}'.format(x,cntEpochs-1), fontsize=14)

        for logVal in logVals:
            for i,layer in enumerate(modelLogger.epochLogs[x][logVal['name']]):
                histmin = layer.min()
                histmax = layer.max()

                plt.subplot(subplotRows, subplotCols, subplotIdx)
                plt.title('{}, {}'.format(modelLogger.model.layers[modelLogger.loggedLayers[i]].name,logVal['name']))
                plt.hist(layer, range = (histmin, histmax), bins = 30, color = logVal['color'])

                subplotIdx+=1

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.savefig('{}/hist_{e:03d}.png'.format(cc.cfg['plots']['histograms_dir'], e = x))
        plt.close()


# data = [items][symbols][onehot]
def layerActivations(model, data, labels):
    if not cc.cfg['plots']['layer_activations']:
        return

    print('Visualizing activations with tSNE...')

    if not os.path.exists(cc.cfg['plots']['layer_activations_dir']):
        os.makedirs(cc.cfg['plots']['layer_activations_dir'])


    numLabels = cc.cfg['plots']['layer_activations_label_cap']

    data = data[:cc.cfg['plots']['layer_activations_points_cap']]
    labels = labels[:numLabels,:cc.cfg['plots']['layer_activations_points_cap']]

    subplotCols = numLabels
    subplotRows = len(model.layers)-1
    subplotIdx = 1

    plt.figure(figsize=(5*subplotCols,5*subplotRows))


    for i in range(1,len(model.layers)):
        print('Running tSNE for layer {}/{}'.format(i,len(model.layers)))

        func = K.function([model.layers[0].input], [model.layers[i].output])
        out = func([data])[0]

        tsneModel = TSNE(n_components = 2, random_state = 0)
        tsneOut = tsneModel.fit_transform(out).T

        # labeledTsneOut = np.hstack((tsneOut, labels[0].reshape(-1,1)))

        for j in range(numLabels):
            plt.subplot(subplotRows, subplotCols, subplotIdx)
            plt.title('layer {} / label {}'.format(i,cc.exp['params']['data']['labels'][j]))
            plt.scatter(tsneOut[0],tsneOut[1],c=labels[j],cmap = 'plasma')

            subplotIdx += 1

        # tsneDF = pd.DataFrame(labeledTsneOut, columns = ('a', 'b', 'c'))
        # plot = tsneDF.plot.scatter(x = 'a', y = 'b', c = 'c', cmap = 'plasma')



    plt.savefig('{}/activations.png'.format(cc.cfg['plots']['layer_activations_dir']))
    plt.close()

    print('...done')