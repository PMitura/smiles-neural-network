import data, utility, metrics
import time

import numpy as np
from math import sqrt, exp, log, ceil
from scipy.stats.stats import pearsonr
from sklearn.metrics import roc_auc_score

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
from keras.regularizers import l2, activity_l2
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

def configureModel(input):
    print('  Initializing and compiling...')

    alphaSize = input.shape[2]
    outputLen = len(RD['labels'])

    model = Sequential()

    '''
    if RD['use_embedding']:
        # second value in nomiSize tuple is shift while using embedding
        model.add(Embedding(1 << nomiSize[1], RP['embedding_outputs']))
        model.add(TimeDistributed(Dense(int(RP['td_layer_multiplier'] * (alphaSize +
            nomiSize[0])), activation = 'tanh', trainable = RP['trainable_inner'])))
    else:
    '''


    # {'parameters_num': 23100, 'name': 'timedistributed_1'}
    # {'parameters_num': 0, 'name': 'dropout_1'}
    # {'output_dim': 300, 'parameters_num': 540900, 'activation': 'tanh', 'name': 'gru_1', 'input_dim': 300}
    # {'activation': 'relu', 'parameters_num': 0, 'name': 'activation_1'}
    # {'parameters_num': 0, 'name': 'dropout_2'}
    # {'output_dim': 53, 'parameters_num': 15953, 'activation': 'linear', 'name': 'dense_2', 'input_dim': None}

    model.add(TimeDistributed(Dense(300, activation = 'tanh'), trainable = True, input_shape = (None, alphaSize )))
    model.add(Dropout(0.5))
    model.add(GRU(300, trainable = True, ))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(outputLen) )

    if RP['classify']:
        model.add(Activation(RP['classify_activation'], trainable = RP['trainable_inner']))

    model.compile(loss = RP['objective'], optimizer = OPTIMIZER)

    pretrainedModel = utility.loadModel('6f7c468746e19ab2ed4c6adb4c15ab7ff50f9088')

    # for i in range(2):
    model.layers[0].set_weights(pretrainedModel.layers[0].get_weights())
    model.layers[0].trainable = True
    model.layers[2].set_weights(pretrainedModel.layers[2].get_weights())
    model.layers[2].trainable = True


    print('  ...done')
    return model

def learningRateDecayer(epoch):
    if not RP['learning_rate_decay']:
        return RP['learning_rate']

    if RP['learning_rate_decay_type'] == 'step':
        drop = np.floor((epoch)/RP['learning_rate_decay_step_config_steps'])
        new_lr = float(RP['learning_rate'] * np.power(RP['learning_rate_decay_step_config_ratio'],drop))
        print('lr',epoch,new_lr)
        return new_lr
    elif RP['learning_rate_decay_type'] == 'time':
        raise NotImplementedError('learning rate decay: time')
    elif RP['learning_rate_decay_type'] == 'peter':
        raise NotImplementedError('learning rate decay: peter')
    else:
        raise RuntimeError('learning rate decat: unknown type {}'.format(RP['learning_rate_decay_type']))


def train(model, nnInput, labels, validation, makePlot = True,
        labelIndexes = RP['label_idxs']):
    print('  Training model...')


    # needed format is orthogonal to ours
    '''
    formattedLabels = np.zeros((len(labels[0]), len(labelIndexes)))
    formattedValid = np.zeros((len(validation[1][labelIndexes[0]]),
        len(labelIndexes)))
    for i in range(len(labelIndexes)):
        for j in range(len(labels[0])):
            formattedLabels[j][i] = labels[labelIndexes[i]][j]
        for j in range(len(validation[1][labelIndexes[i]])):
            formattedValid[j][i] = validation[1][labelIndexes[i]][j]
    '''
    early = keras.callbacks.EarlyStopping(monitor = 'val_loss',
            patience = RP['early_stop'])

    learningRateScheduler = keras.callbacks.LearningRateScheduler(learningRateDecayer)

    modelLogger = visualization.ModelLogger()

    history = model.fit(nnInput, labels, nb_epoch = RP['epochs'],
            batch_size = RP['batch'], callbacks = [early],
            validation_data = (validation[0], validation[1]))

    if makePlot:
        values = np.zeros((len(history.history['loss']), 2))
        for i in range(len(history.history['loss'])):
            values[i][0] = history.history['loss'][i]
            values[i][1] = history.history['val_loss'][i]
        utility.plotLoss(values)

    visualization.histograms(modelLogger)

    print('    Model weights:')
    print(model.summary())
    # print(model.get_weights())
    print('  ...done')
    return len(history.history['loss'])

def run(grid = None):
    stats = {}
    stats['runtime_second'] = time.time()

    # initialize using the same seed (to get stable results on comparisons)
    np.random.seed(RP['seed'])

    # get the training and testing datasets along with some meta info
    trainIn, trainLabel, testIn, testLabel, preprocessMeta = data.preprocessData(db.getData())

    stats['training_row_count'] = len(trainIn)
    stats['testing_row_count'] = len(testIn)

    # load model from file or create and train one from scratch
    if RP['load_model']:
        model = utility.loadModel(RP['load_model'])
    else:
        model = configureModel(trainIn)
        stats['epoch_count'] = train(model, trainIn, trainLabel, (testIn, testLabel))

    # persistence first
    if cc.cfg['persistence']['model']:
        utility.saveModel(model)

    # compute metrics for the model based on the task for both testing and training data
    print('\nGetting metrics for training data:')
    if RP['classify']:
        trainMetrics = metrics.classify(model, trainIn, trainLabel, preprocessMeta)
    else:
        trainMetrics = metrics.predict(model, trainIn, trainLabel, preprocessMeta)

    print('\nGetting metrics for test data:')
    if RP['classify']:
        testMetrics = metrics.classify(model, testIn, testLabel, preprocessMeta)
    else:
        testMetrics = metrics.predict(model, testIn, testLabel, preprocessMeta)

    # utilities and visualizations
    if cc.cfg['plots']['layer_activations']:
        visualization.layerActivations(model, testIn, testLabel)

    if cc.cfg['plots']['seq_output']:
        visualization.visualizeSequentialOutput(model, cc.cfg['plots']['seq_output_layer_idx'], cc.cfg['plots']['seq_output_smiles'])

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
