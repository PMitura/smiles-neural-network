import keras.callbacks

import numpy as np
import matplotlib.pyplot as plt


class ModelLogger(keras.callbacks.Callback):

    def __init__(self):
        self.epochs = []

    def _set_params(self, params):
        self.params = params

    def _set_model(self, model):
        self.model = model

    def on_epoch_begin(self, epoch, logs={}):
        epochDict = {
            'idx': epoch,
            'batches': [],
            'layer_weights': []
        }
        self.epochs.append(epochDict)

    def on_epoch_end(self, epoch, logs={}):
        epochDict = self.epochs[-1]

        for layer in self.model.layers:
            epochDict['layer_weights'].append(np.array(layer.get_weights()).flatten())

    def on_batch_begin(self, batch, logs={}):
        print('bat begin', batch, logs)

    def on_batch_end(self, batch, logs={}):
        print('bat end', batch, logs)

    def on_train_begin(self, logs={}):
        print('train begin', logs)

    def on_train_end(self, logs={}):
        print('train end', logs)


def weightHistogram(modelLogger):
    lastEpoch = modelLogger.epochs[-1]
    layerWeights = lastEpoch['layer_weights'][0]

    plt.figure()
    plt.hist(layerWeights)
    plt.savefig('local/plots/hist.png')
