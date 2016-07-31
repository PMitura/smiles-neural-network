import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as pltib
import psutil

from math import sqrt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

PLOT_NAME = 'loss_plot.pdf'
SCATTER_NAME = 'scatter.pdf'

from keras import backend as K

# Mean of given values
def mean(array, size):
    arraySum = 0.0
    for i in range(size):
        arraySum += array[i]
    return arraySum / size


# Variance of given values
def variance(array, size):
    avg = mean(array, size)
    dev = 0.0
    for i in range(size):
        dev += (array[i] - avg) * (array[i] - avg)
    return dev / size


# Standard deviation of given values
def stddev(array, size):
    return sqrt(variance(array, size))


# Root mean square of given values
def rms(array, size):
    sqrSum = 0.0
    for i in range(size):
        sqrSum += array[i] * array[i]
    return sqrt(sqrSum / size)

# Logloss
def logloss(predictions, labels):
    if len(predictions) != len(labels):
        raise RuntimeError('logloss: lengths of arrays differ: predictions: %d, labels: %d'.format(len(predictions), len(labels)))

    N = len(predictions)

    p = np.array(predictions)
    y = np.array(labels)

    # avoid NaN and Inf
    eps = 1e-15
    p = np.maximum(eps, p)
    p = np.minimum(1-eps, p)

    return -np.sum(y*np.log(p)+(1-y)*np.log(1-p))/N

def modelToString(model):
    string = ''
    for layer in model.layers:
        modelDict = {}
        config = layer.get_config()
        # Restrict to 'interesting' attributes
        if 'name' in config:
            modelDict['name'] = config['name']
        if 'input_dim' in config:
            modelDict['input_dim'] = config['input_dim']
        if 'output_dim' in config:
            modelDict['output_dim'] = config['output_dim']
        if 'activation' in config:
            modelDict['activation'] = config['activation']
        modelDict['parameters_num'] = layer.count_params()
        string += str(modelDict)
    return string


def plotLoss(values):
    print '    Plotting losses'

    matplotlib.style.use('ggplot')
    dFrame = pd.DataFrame(values, index = pd.Series(list(range(len(values)))),
        columns = ['training', 'validation'])
    plot = dFrame.plot()
    plot.set_xlabel('epoch')
    plot.set_ylabel('loss')
    fig = plot.get_figure()
    fig.savefig('plots/{}'.format(PLOT_NAME))


# PCA-like visualisation of layer output
def visualize2D(model, layerID, inputData, labels, withTime = False):
    print("\n  Generating output distribution for layer {}".format(layerID))
    vLayer = K.function([model.layers[0].input], [model.layers[layerID].output])
    result = vLayer([inputData])

    values = []
    for instance in result:
        for line in instance:
            array = []
            for val in line:
                if withTime:
                    for deepVal in val:
                        array.append(deepVal)
                else:
                    array.append(val)
            values.append(array)
    npvalues = np.array(values)

    model = TSNE(n_components = 2, random_state = 0)
    # model = PCA(n_components = 2)
    scatterValues = model.fit_transform(npvalues)
    labels2D = np.zeros((len(labels), 1))
    for i in range(len(labels)):
        labels2D[i][0] = labels[i]
    scatterValues = np.hstack((scatterValues, labels2D))

    dFrame = pd.DataFrame(scatterValues, columns = ('a', 'b', 'c'))
    plot = dFrame.plot.scatter(x = 'a', y = 'b', c = 'c', cmap = 'plasma')
    fig = plot.get_figure()
    fig.savefig('plots/{}'.format(SCATTER_NAME))

    print("  ...done")

def getMemoryUsage():
    thisProcess = psutil.Process()
    memRss = thisProcess.memory_info().rss / 1000000.0
    memVms = thisProcess.memory_info().vms / 1000000.0
    print '  Memory usage:'
    print '    Physical: {} MB'.format(memRss)
    print '    Virtual:  {} MB'.format(memVms)
    return memRss, memVms


