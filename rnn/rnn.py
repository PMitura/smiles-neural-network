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

import pandas as pd

import socket

# not used to relieve MetaCentrum of some dependencies

# TODO: Remove unused imports after experiments are done
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, LSTM, AveragePooling1D
from keras.layers import TimeDistributed, SimpleRNN, GRU
from keras.layers import BatchNormalization, Embedding, Merge
from keras.optimizers import Adam, RMSprop, Adadelta, Adagrad
from keras.regularizers import l2, activity_l2
import keras.callbacks
# from keras.regularizers import l1

# handy aliases for config
RP = cc.exp['params']['rnn']
RD = cc.exp['params']['data']
RG = cc.exp['grid']


# manual eval where needed
RP['chained_labels'] = eval(str(cc.exp['params']['rnn']['chained_labels']))
RP['chained_predict'] = eval(str(cc.exp['params']['rnn']['chained_predict']))
RP['chained_test_labels'] = eval(str(cc.exp['params']['rnn']['chained_test_labels']))
RP['freeze_idxs'] = eval(str(cc.exp['params']['rnn']['freeze_idxs']))
RP['label_idxs'] = eval(str(cc.exp['params']['rnn']['label_idxs']))

OPTIMIZER = Adam(lr = RP['learning_rate'], clipnorm = 1.)

def configureModel(input, outputLen = len(RD['labels'])):
    print('  Initializing and compiling...')

    alphaSize = input.shape[2]

    model = Sequential()

    '''
    if RD['use_embedding']:
        # second value in nomiSize tuple is shift while using embedding
        model.add(Embedding(1 << nomiSize[1], RP['embedding_outputs']))
        model.add(TimeDistributed(Dense(int(RP['td_layer_multiplier'] * (alphaSize +
            nomiSize[0])), activation = 'tanh', trainable = RP['trainable_inner'])))
    else:
    '''


    model.add(TimeDistributed(Dense(300*RG['ratios'][0], activation = 'tanh', trainable = RP['trainable_inner']), input_shape = (None, alphaSize )))
    model.add(Dropout(0.30))
    model.add(GRU(300*RG['ratios'][1], trainable = RP['trainable_inner'], return_sequences = True))
    model.add(Activation('tanh', trainable = RP['trainable_inner']))
    model.add(Dropout(0.30))
    model.add(GRU(300*RG['ratios'][2], trainable = RP['trainable_inner']))
    model.add(Activation('tanh', trainable = RP['trainable_inner']))
    model.add(Dropout(0.30))
    model.add(Dense(outputLen))

    # molweight
    # model = utility.loadModel('b3d9609da78bfbf0ad1a62ee6740df3b52f104b4', 'mol_')
    # all compounds
    # model = utility.loadModel('eab15a05a70b35d119c02fcc36b1cfaf27a0f36a', 'mol_')
    # maccs
    # model = utility.loadModel('67b51a1543b5d32b05671e4a08d193eed702ca54', 'mol_')

    # model.pop()
    # model.pop()

    # for i in xrange(len(model.layers)):
        # model.layers[0].trainable = False

    '''
    model.add(Dropout(0.50))
    model.add(Dense(500))
    model.add(Activation('relu'))
    model.add(Dropout(0.50))
    model.add(Dense(500))
    model.add(Activation('relu'))
    model.add(Dropout(0.30))
    '''
    # model.add(Dense(outputLen))

    if RP['classify']:
        model.add(Activation(RP['classify_activation'], trainable = RP['trainable_inner']))

    metrics = []
    if RP['classify']:
        metrics.append('accuracy')

    model.compile(loss = RP['objective'], optimizer = OPTIMIZER, metrics = metrics)

    print('  ...done')
    return model

def configureEdgeModel(inputSmiles, inputFasta):
    print('  Initializing edge model and compiling...')

    smilesGRUInputShape = (None, inputSmiles.shape[2])
    # smilesGRUSize = int(RP['gru_layer_multiplier'] * smilesGRUInputShape[1])

    fastaGRUInputShape = (None, inputFasta.shape[2])
    # fastaGRUSize = int(RP['fasta_gru_layer_multiplier'] * fastaGRUInputShape[1])

    mergedOutputLen = len(RD['labels'])

    smilesModel = Sequential()
    smilesModel.add(TimeDistributed(Dense(300, activation = 'tanh', trainable = RP['trainable_inner']), input_shape = smilesGRUInputShape))
    smilesModel.add(Dropout(0.30))
    smilesModel.add(GRU(300, trainable = RP['trainable_inner'], return_sequences = True))
    smilesModel.add(Activation('tanh', trainable = RP['trainable_inner']))
    smilesModel.add(Dropout(0.30))
    smilesModel.add(GRU(300, trainable = RP['trainable_inner']))
    smilesModel.add(Activation('tanh', trainable = RP['trainable_inner']))

    # utility.setModelConsumeLess(smilesModel, 'mem')

    '''
    smilesModel = utility.loadModel('24e62794bb6d5b5c562e41a3a2cccc3525fa625f', 'smiles_')
    smilesModel.pop() # output
    smilesModel.pop() # dropout
    '''
    # utility.setModelConsumeLess(smilesModel, 'gpu')
    fastaModel = Sequential()
    fastaModel.add(TimeDistributed(Dense(300, activation = 'tanh', trainable = RP['trainable_inner']), input_shape = fastaGRUInputShape))
    fastaModel.add(Dropout(0.30))
    fastaModel.add(GRU(300, trainable = RP['trainable_inner'], return_sequences = True))
    fastaModel.add(Activation('tanh', trainable = RP['trainable_inner']))
    fastaModel.add(Dropout(0.30))
    fastaModel.add(GRU(300, trainable = RP['trainable_inner']))
    fastaModel.add(Activation('tanh', trainable = RP['trainable_inner']))

    # utility.setModelConsumeLess(fastaModel, 'mem')

    '''
    fastaModel = utility.loadModel('e6beb8b7e146b9ab46a71db8f3001bf62d96ff08', 'fasta_')
    fastaModel.pop() # activation
    fastaModel.pop() # output
    fastaModel.pop() # dropout
    '''

    # utility.setModelConsumeLess(fastaModel, 'gpu')

    merged = Merge([smilesModel, fastaModel], mode='concat')

    mergedModel = Sequential()
    mergedModel.add(merged)

    mergedModel.add(Dense(300))
    mergedModel.add(Activation('relu'))
    mergedModel.add(Dropout(0.3))

    mergedModel.add(Dense(300))
    mergedModel.add(Activation('relu'))
    mergedModel.add(Dropout(0.3))

    mergedModel.add(Dense(mergedOutputLen))

    if RP['classify']:
        mergedModel.add(Activation(RP['classify_activation'], trainable = RP['trainable_inner']))

    metrics = []
    if RP['classify']:
        metrics.append('accuracy')

    mergedModel.compile(loss = RP['objective'], optimizer = OPTIMIZER, metrics = metrics)

    print('  ...done')
    return mergedModel


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

    # grab the commit at start
    stats['git_commit'] = utility.getGitCommitHash()

    # get the training and testing datasets along with some meta info

    if RP['edge_prediction']:
        trainIn, trainLabel, testIn, testLabel, preprocessMeta = data.preprocessEdgeData(db.getData())
    else:
        trainIn, trainLabel, testIn, testLabel, preprocessMeta = data.preprocessData(db.getData())
        # trainIn, trainLabel, testIn, testLabel, preprocessMeta = data.preprocessFastaOneHotData(db.getData())

    stats['training_row_count'] = len(testLabel)
    stats['testing_row_count'] = len(testLabel)

    # load model from file or create and train one from scratch
    if RP['load_model']:
        model = utility.loadModel(RP['load_model'])
    else:
        if RP['edge_prediction']:
            model = configureEdgeModel(trainIn[0],trainIn[1])
        elif RP['discrete_label']:
            model = configureModel(trainIn, len(trainLabel[0]))
        else:
            model = configureModel(trainIn)
        stats['epoch_count'] = train(model, trainIn, trainLabel, (testIn, testLabel))

    # persistence first
    if cc.cfg['persistence']['model']:
        name = '{}_rg_{}'.format(stats['git_commit'],':'.join(RG['ratios']))
        # name = stats['git_commit']
        stats['persistent_model_name'] = name
        utility.saveModel(model, name)

    # compute metrics for the model based on the task for both testing and training data
    print('\nGetting metrics for training data:')
    if RP['classify']:
        if RP['discrete_label']:
            trainMetrics = metrics.discreteClassify(model, trainIn, trainLabel, preprocessMeta)
        else:
            trainMetrics = metrics.classify(model, trainIn, trainLabel, preprocessMeta)
    else:
        trainMetrics = metrics.predict(model, trainIn, trainLabel, preprocessMeta)

    print('\nGetting metrics for test data:')
    if RP['classify']:
        if RP['discrete_label']:
            testMetrics = metrics.discreteClassify(model, testIn, testLabel, preprocessMeta)
        else:
            testMetrics = metrics.classify(model, testIn, testLabel, preprocessMeta)
    else:
        testMetrics = metrics.predict(model, testIn, testLabel, preprocessMeta)


    # utilities and visualizations
    if cc.cfg['plots']['layer_activations']:
        visualization.layerActivations(model, testIn, testLabel)

    if cc.cfg['plots']['seq_output']:
        df = pd.DataFrame(cc.cfg['plots']['seq_output_seq_input'], columns=[RD['fasta'] if cc.cfg['plots']['seq_output_seq_input_name'] == 'fasta' else RD['smiles']])
        visualization.visualizeSequentialOutput(model, cc.cfg['plots']['seq_output_layer_idx'], df)

    if cc.cfg['plots']['print_pred']:
        visualization.printPrediction(model, cc.cfg['plots']['print_pred_smiles'])

    if cc.cfg['plots']['print_train_test_pred']:
        visualization.printTrainTestPred(model, cc.cfg['plots']['print_train_test_pred_cnt'], trainIn, trainLabel, testIn, testLabel, preprocessMeta)

    # statistics to send to journal
    stats['runtime_second'] = time.time() - stats['runtime_second']
    stats['memory_pm_mb'], stats['memory_vm_mb'] = utility.getMemoryUsage()
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
        metricStats['auc_micro'] = testMetrics['auc_avg']
        metricStats['auc_micro_std'] = testMetrics['auc_std']
    else:
        metricStats['relevance_training'] = trainMetrics['r2_avg']
        metricStats['relevance_training_std'] = trainMetrics['r2_std']
        metricStats['relevance_testing'] = testMetrics['r2_avg']
        metricStats['relevance_testing_std'] = testMetrics['r2_std']
        metricStats['mse'] = testMetrics['mse_avg']
        metricStats['mse_std'] = testMetrics['mse_std']
        metricStats['mae'] = testMetrics['mae_avg']
        metricStats['mae_std'] = testMetrics['mae_std']

    stats.update(metricStats)
    db.sendStatistics(**stats)

    utility.freeModel(model)
