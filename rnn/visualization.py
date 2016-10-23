import keras.callbacks
from keras import backend as K
from keras.models import Model,Sequential


import numpy as np

import prettyplotlib as ppl
from prettyplotlib import plt

#import seaborn as sns
import scipy.cluster.hierarchy as sch


import collections
import pandas as pd

from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import roc_curve, auc


import data
from config import config as cc
import os
from math import ceil
from sets import Set

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
        print('Running tSNE for layer {}/{}'.format(i+1,len(model.layers)))

        func = K.function([model.layers[0].input], [model.layers[i].output])
        out = func([data])[0]

        tsneModel = TSNE(n_components = 2, random_state = 0)
        tsneOut = tsneModel.fit_transform(out).T

        # labeledTsneOut = np.hstack((tsneOut, labels[0].reshape(-1,1)))

        for j in range(numLabels):
            plt.subplot(subplotRows, subplotCols, subplotIdx)
            plt.title('{} / {}'.format(model.layers[i].name,cc.exp['params']['data']['labels'][j]))
            plt.scatter(tsneOut[0],tsneOut[1],c=labels[j],cmap = 'plasma')

            subplotIdx += 1

        # tsneDF = pd.DataFrame(labeledTsneOut, columns = ('a', 'b', 'c'))
        # plot = tsneDF.plot.scatter(x = 'a', y = 'b', c = 'c', cmap = 'plasma')


    plt.tight_layout()
    plt.savefig('{}/activations.png'.format(cc.cfg['plots']['layer_activations_dir']))
    plt.close()

    print('...done')

def visualizeSequentialOutput(model, layerIdx, df):

    if not os.path.exists(cc.cfg['plots']['seq_output_dir']):
        os.makedirs(cc.cfg['plots']['seq_output_dir'])


    if cc.cfg['plots']['seq_output_seq_input_name'] == 'smiles':
        input = data.formatSequentialInput(df)
    elif cc.cfg['plots']['seq_output_seq_input_name'] == 'fasta':
        input = data.formatFastaInput(df)
    else:
        raise 'visual err'


    # model.layers[layerIdx].return_sequences = True
    # model.compile(loss="mean_squared_error", optimizer="rmsprop")


    cfg = model.get_config()[:4]

    cfg = model.get_config()[:layerIdx+1]
    del cfg[2]
    layerIdx -= 1

    # print cfg
    cfg[layerIdx]['config']['return_sequences'] = True

    seqModel = Sequential.from_config(cfg)
    seqModel.set_weights(model.get_weights())
    seqModel.layers[layerIdx].return_sequences = True


    outputFunction = K.function([seqModel.layers[0].input],
              [seqModel.layers[layerIdx].output])

    output = outputFunction([input])[0]

    '''
    sns.set()
    for i,smilesOutput in enumerate(output):
        g = sns.clustermap(smilesOutput.T, col_cluster=False,  method='single',metric='cosine')
        g.savefig('{}/seq_output.png'.format(cc.cfg['plots']['seq_output_dir']))
    '''

    dropSet = Set(cc.cfg['plots']['seq_output_ignore_neurons'])
    if cc.cfg['plots']['seq_output_select_neurons']:
        arrMask = cc.cfg['plots']['seq_output_select_neurons']
    else:
        arrMask = list(range(output.shape[2]))
    arrMask = np.array([x for x in arrMask if not x in dropSet])

    fig = plt.figure(figsize=(input.shape[1] * 0.3,len(arrMask) * len(df) * 1.5))


    for i,seqOutput in enumerate(output):

        # print seqOutput.shape
        # print seqOutput

        selected = seqOutput.T[arrMask]

        Z = sch.linkage(selected, method='single', metric='cosine')
        leaves = sch.leaves_list(Z)
        # leaves = range(len(selected))
        reordered = selected[leaves]

        ax = fig.add_subplot(len(df),1,i+1)

        print 'foo'

        ppl.pcolormesh(fig, ax, reordered,
               xticklabels=list(df.values[i][0]),
               yticklabels=arrMask[leaves],
               vmin=-1,
               vmax=1)

        print 'foo'

    print 'bar'

    fig.savefig('{}/{}'.format(cc.cfg['plots']['seq_output_dir'],cc.cfg['plots']['seq_output_name']))

    print 'bar'

def printPrediction(model, smilesData):
    # FIXME hardcoded

    smilesDf = pd.DataFrame(smilesData, columns=[cc.exp['params']['data']['smiles']])

    input = data.formatSequentialInput(smilesDf)

    output = model.predict(input)

    for i, smiles in enumerate(smilesData):
        print 'Prediction for {}'.format(smiles)
        print output[i]

    distanceMatrixCosine = pairwise_distances(output, metric='cosine')
    distanceMatrixCorrel = pairwise_distances(output, metric='correlation')
    distanceMatrixEuclid = pairwise_distances(output, metric='euclidean')

    print 'Distance matrix cosine'
    print distanceMatrixCosine
    print 'Distance matrix correlation'
    print distanceMatrixCorrel
    print 'Distance matrix euclid'
    print distanceMatrixEuclid

    '''

    layerIdx = 1
    cfg = model.get_config()[:layerIdx+1]
    cfg[0]['config']['dropout_U'] = 0
    cfg[0]['config']['dropout_W'] = 0

    print cfg[0]
    print cfg[1]
    # del cfg[1]
    # layerIdx -= 1
    # print cfg
    cfg[layerIdx]['config']['return_sequences'] = True
    '''


    layerIdx = 2
    cfg = model.get_config()[:layerIdx+1]
    del cfg[1]
    layerIdx -= 1
    # print cfg
    cfg[layerIdx]['config']['return_sequences'] = True

    seqModel = Sequential.from_config(cfg)
    seqModel.set_weights(model.get_weights())
    seqModel.layers[layerIdx].return_sequences = True


    outputFunction = K.function([seqModel.layers[0].input],
              [seqModel.layers[layerIdx].output])

    outputSymbols = outputFunction([input])[0]

    outputLastSymbol = outputSymbols[:,outputSymbols.shape[1]-1,:]

    distanceMatrixLastSymbolCorrel = np.corrcoef(outputLastSymbol)

    print 'Distance matrix last symbol correlation'
    print distanceMatrixLastSymbolCorrel

def printTrainTestPred(model, cnt, trainIn, trainLabel, testIn, testLabel, meta):
    print('Printing train/test predictions:')

    predTrain = data.denormalize(model.predict(trainIn[:cnt]), meta)
    labelTrain = data.denormalize(trainLabel[:cnt], meta)

    trainData = []
    trainCols = []

    for i in xrange(trainLabel.shape[1]):
        trainData.append(list(predTrain[:,i]))
        trainCols.append('train pred label{}'.format(i))
        trainData.append(list(labelTrain[:,i]))
        trainCols.append('train true label{}'.format(i))

    traindf = pd.DataFrame(np.array(trainData).T, columns=trainCols)
    print traindf

    predTest = data.denormalize(model.predict(testIn[:cnt]), meta)
    labelTest = data.denormalize(testLabel[:cnt], meta)

    testData = []
    testCols = []

    for i in xrange(testLabel.shape[1]):
        testData.append(list(predTest[:,i]))
        testCols.append('test pred label{}'.format(i))
        testData.append(list(labelTest[:,i]))
        testCols.append('test true label{}'.format(i))

    testdf = pd.DataFrame(np.array(testData).T, columns=testCols)
    print testdf

def plotROCCurve(model, testIn, testLabel):

    if not os.path.exists(cc.cfg['plots']['roc_curve_dir']):
        os.makedirs(cc.cfg['plots']['roc_curve_dir'])

    dataPred = model.predict(testIn, batch_size = 80)

    fpr, tpr, _ = roc_curve(testLabel.T[0],dataPred.T[0])

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    # plt.legend(loc="lower right")

    plt.savefig('{}/{}'.format(cc.cfg['plots']['roc_curve_dir'],cc.cfg['plots']['roc_curve_name']))
    plt.close()