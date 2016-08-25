import data, utility, metrics
import time

import numpy as np
from math import sqrt, exp, log, ceil
from scipy.stats.stats import pearsonr
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing

from keras.regularizers import l2, activity_l2
import matplotlib
import matplotlib.pyplot as plt

import pandas as pd
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


def rsqrComp(pred, truth):
    merged = pd.concat([pred, truth],axis=1).T

    pearCr = np.corrcoef(merged)
    return pearCr[0][1] * pearCr[0][1]

def run():
    stats = {}
    stats['runtime_second'] = time.time()

    startTime = time.time()

    # Initialize using the same seed (to get stable results on comparisons)
    np.random.seed(RP['seed'])

    rawData = db.getData()

    # filter infs and nans from data cols
    cols = rawData.columns.tolist()[1:-1]

    print(cols)

    for col in cols:
        rawData = rawData.drop(rawData[np.isinf(rawData[col])].index)
        rawData = rawData.drop(rawData[np.isnan(rawData[col])].index)

    rawData.reset_index(drop=True,inplace=True)
    rawData.reindex(np.random.permutation(rawData.index))

    # print(rawData)

    X_raw = rawData.iloc[:, 2:-1]
    y_raw = rawData.iloc[:, 1:2]

    scalerX = preprocessing.StandardScaler(copy=False)
    scalerX.fit(X_raw)
    scalery = preprocessing.StandardScaler(copy=False)
    scalery.fit(y_raw)

    if RP['zscore_norm']:
        X = pd.DataFrame(scalerX.transform(X_raw), columns=X_raw.columns.values)
        y = pd.DataFrame(scalery.transform(y_raw), columns=y_raw.columns.values)
    else:
        X = X_raw
        y = y_raw

    # print(X.head(), y.head())

    model = Sequential()

    # hidden
    model.add(Dense(100, W_regularizer=l2(0.3),activity_regularizer=activity_l2(0.01), input_shape=(X.shape[1], )))
    model.add(Activation('relu'))
    model.add(Dense(1))

    model.compile(loss = 'mse', optimizer = OPTIMIZER)

    if RD['use_test_flags']:
        maskTrain = np.zeros(len(X),dtype=bool)
        maskTest = np.zeros(len(X),dtype=bool)
        for i in range(len(X)):
            maskTrain[i] = rawData[RD['testing']][i] == 0
            maskTest[i] = rawData[RD['testing']][i] == 1

        trainX = X.loc[maskTrain]
        testX = X.loc[maskTest]
        trainy = y.loc[maskTrain]
        testy = y.loc[maskTest]

    else:
        ratio = 0.8
        split = int(X.shape[0] * ratio)


        trainX, testX = X.iloc[:split], X.iloc[split:]
        trainy, testy = y.iloc[:split], y.iloc[split:]

    trainX.reset_index(drop=True,inplace=True)
    testX.reset_index(drop=True,inplace=True)
    trainy.reset_index(drop=True,inplace=True)
    testy.reset_index(drop=True,inplace=True)



    stats['training_row_count'] = len(trainX)
    stats['testing_row_count'] = len(testX)


    print(trainX.shape, testX.shape, trainy.shape, testy.shape)


    history = model.fit(trainX.values, trainy.values, nb_epoch = RP['epochs'],
            batch_size = RP['batch'],
            validation_data = (testX.values, testy.values))


    preprocessMeta = {
        'scaler': scalery
    }

    # compute metrics for the model based on the task for both testing and training data
    print('\nGetting metrics for training data:')
    if RP['classify']:
        trainMetrics = metrics.classify(model, trainX.values, trainy.values, preprocessMeta)
    else:
        trainMetrics = metrics.predict(model, trainX.values, trainy.values, preprocessMeta)

    print('\nGetting metrics for test data:')
    if RP['classify']:
        testMetrics = metrics.classify(model, testX.values, testy.values, preprocessMeta)
    else:
        testMetrics = metrics.predict(model, testX.values, testy.values, preprocessMeta)


    print('Plot:')
    values = np.zeros((len(history.history['loss']), 2))
    for i in range(len(history.history['loss'])):
        values[i][0] = history.history['loss'][i]
        values[i][1] = history.history['val_loss'][i]
    utility.plotLoss(values)

    print('Dump csv pred')
    pred = model.predict(testX.values, batch_size = RP['batch'])


    if RP['zscore_norm']:
        predScaled = pd.DataFrame(scalery.inverse_transform(pred), columns=['pred'])
        testScaled = pd.DataFrame(scalery.inverse_transform(testy), columns=['true'])
    else:
        predScaled = pd.DataFrame(pred,columns=['pred'])
        testScaled = pd.DataFrame(testy,columns=['true'])

    predByTruth = pd.concat([predScaled, testScaled],axis=1)

    # predByTruth.plot(x='pred',y='true', kind='scatter')
    # plt.show()
    # predByTruth.to_csv('local/pred.csv')


    # statistics to send to journal
    stats['runtime_second'] = time.time() - stats['runtime_second']
    stats['memory_pm_mb'], stats['memory_vm_mb'] = utility.getMemoryUsage()
    stats['git_commit'] = utility.getGitCommitHash()
    stats['comment'] = RP['comment']
    stats['hostname'] = socket.gethostname()
    stats['experiment_config'] = yaml.dump(cc.exp,default_flow_style=False)

    stats['model'] = utility.modelToString(model)
    stats['loaded_model'] = RP['load_model']
    stats['parameter_count'] = model.count_params()
    stats['task'] = 'classification' if RP['classify'] else 'regression'

    stats['dataset_name'] = cc.exp['fetch']['table']
    stats['split_name'] = RD['testing']
    stats['label_name'] = ','.join(RD['labels'])

    stats['epoch_max'] = RP['epochs']
    stats['learning_rate'] = RP['learning_rate']
    stats['optimization_method'] = OPTIMIZER.__class__.__name__
    stats['batch_size'] = RP['batch']
    stats['seed'] = RP['seed']
    stats['objective'] = RP['objective']
    stats['learning_curve'] = {'val':open('{}/{}'.format(cc.cfg['plots']['dir'], utility.PLOT_NAME),'rb').read(),'type':'bin'}

    # metric statistics to send
    metricStats = {}

    if RP['classify']:
        metricStats['relevance_training'] = trainMetrics['acc_avg']
        metricStats['relevance_training_std'] = trainMetrics['acc_std']
        metricStats['relevance_testing'] = testMetrics['acc_avg']
        metricStats['relevance_testing_std'] = testMetrics['acc_std']
        metricStats['log_loss'] = testMetrics['log_loss_avg']
        metricStats['log_loss_std'] = testMetrics['log_loss_std']
        metricStats['auc'] = testMetrics['auc_avg']
        metricStats['auc_std'] = testMetrics['auc_std']
    else:
        metricStats['relevance_training'] = trainMetrics['r2_avg']
        metricStats['relevance_training_std'] = trainMetrics['r2_std']
        metricStats['relevance_testing'] = testMetrics['r2_avg']
        metricStats['relevance_testing_std'] = testMetrics['r2_std']
        metricStats['mse'] = testMetrics['mse_avg']
        metricStats['mse_std'] = testMetrics['mse_std']

    stats.update(metricStats)
    db.sendStatistics(**stats)