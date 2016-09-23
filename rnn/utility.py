import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as pltib
import psutil,subprocess,os

from math import sqrt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from keras.models import model_from_yaml
from keras.layers import TimeDistributed, SimpleRNN, GRU, LSTM


import utility

from config import config as cc

PLOT_NAME = 'loss_plot.pdf'
SCATTER_NAME = 'scatter.pdf'

EPS = 0.00001

from keras import backend as K

def equals(a, b):
    if abs(a - b) < EPS:
        return True
    return False

# Mean of given values
def mean(array, size):
    if size == 0:
        return np.nan

    arraySum = 0.0
    for i in range(size):
        arraySum += array[i]
    return arraySum / size


# Variance of given values
def variance(array, size):
    if size == 0:
        return np.nan

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
# test1:
#   labels = [-1, -1, -1, 1, 1, 1]
#   predictions = [-1, -0.6, -0.4, -0.52, 0.8, 1]
#   logloss = 0.352049227758
# test2:
#   labels = [-1, -1, -1, 1, 1, 1]
#   predictions = [-1, -0.6, -0.4, -0.52, 0.6, 1]
#   logloss = 0.371679733701

def logloss(predictions, labels, lo = -1, hi = 1):
    if len(predictions) != len(labels):
        raise RuntimeError('logloss: lengths of arrays differ: predictions: %d, labels: %d'.format(len(predictions), len(labels)))

    N = len(predictions)

    p = np.array(predictions)
    y = np.array(labels)


    # Squash values to [0,1]
    p = (p-lo)/(hi-lo)
    y = (y-lo)/(hi-lo)

    # avoid NaN and Inf
    eps = 1e-15
    p = np.maximum(eps, p)
    p = np.minimum(1-eps, p)

    return -np.sum(y*np.log(p)+(1-y)*np.log(1-p))/N


# Bin data into two numerical classes by given ratio of set sizes
# Return binned copy of input array
def bin(data, ratio, classA = -1, classB = 1):
    sortedData = np.sort(data)
    pivot = sortedData[int(len(data) * ratio)]
    # reuse sortedData for result to prevent another allocation
    for idx in range(len(data)):
        if data[idx] < pivot:
            sortedData[idx] = classA
        else:
            sortedData[idx] = classB
    return sortedData


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

def getGitCommitHash():
    try:
        gitCommit = subprocess.check_output('git rev-parse HEAD', shell=True).strip()
    except:
        gitCommit = 'default'

    return gitCommit

def setModelConsumeLess(model, mode='cpu'):
    for i in xrange(len(model.layers)):
        if type(model.layers[i]) is GRU or type(model.layers[i]) is LSTM:
            model.layers[i].consume_less = mode



def saveModel(model, name):
    if not os.path.exists(cc.cfg['persistence']['model_dir']):
        os.makedirs(cc.cfg['persistence']['model_dir'])

    yamlModel = model.to_yaml()

    open(os.path.join(cc.cfg['persistence']['model_dir'], name +'.yml'), 'w').write(yamlModel)
    model.save_weights(os.path.join(cc.cfg['persistence']['model_dir'], name +'.h5'), overwrite=True)


def loadModel(modelName, layerPrefix=None):
    model = model_from_yaml(open(os.path.join(cc.cfg['persistence']['model_dir'], modelName+'.yml')).read())
    model.load_weights(os.path.join(os.path.join(cc.cfg['persistence']['model_dir'], modelName+'.h5')))

    if layerPrefix:
        for i in xrange(len(model.layers)):
            model.layers[i].name = layerPrefix + model.layers[i].name

    return model

def plotLoss(values):
    print '    Plotting losses'

    matplotlib.style.use('ggplot')
    dFrame = pd.DataFrame(values, index = pd.Series(list(range(len(values)))),
        columns = ['training', 'validation'])
    plot = dFrame.plot()
    plot.set_xlabel('epoch')
    plot.set_ylabel('loss')
    fig = plot.get_figure()
    fig.savefig('{}/{}'.format(cc.cfg['plots']['dir'],PLOT_NAME))


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
    fig.savefig('{}/{}'.format(cc.cfg['plots']['dir'],SCATTER_NAME))

    print("  ...done")

def getMemoryUsage():
    thisProcess = psutil.Process()
    memRss = thisProcess.memory_info().rss / 1000000.0
    memVms = thisProcess.memory_info().vms / 1000000.0
    print '  Memory usage:'
    print '    Physical: {} MB'.format(memRss)
    print '    Virtual:  {} MB'.format(memVms)
    return memRss, memVms


