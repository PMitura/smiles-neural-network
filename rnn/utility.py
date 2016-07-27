import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as pltib

from math import sqrt

PLOT_NAME = 'loss_plot.pdf'


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
    print '    Plotting results'

    matplotlib.style.use('ggplot')
    dframe = pd.DataFrame(values, index = pd.Series(list(range(len(values)))),
        columns = ['training', 'validation'])
    plot = dframe.plot()
    plot.set_xlabel('epoch')
    plot.set_ylabel('loss')
    fig = plot.get_figure()
    fig.savefig('plots/{}'.format(PLOT_NAME))


