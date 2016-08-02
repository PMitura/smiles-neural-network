import numpy as np
import random
import pubchem as pc

import pandas as pd
import utility

import db.db as db
from config import config as cc

from math import floor, log, isnan, sqrt, ceil

RD = cc.exp['params']['data']

# Transforms data into 1 of k encoding
# Output format is 3D array of integers, representing positions of binary 1
def formatSMILES(rawData, col):
    print('  Formatting SMILES data column...')

    # Get a set of all used characters
    alphabet = set()
    for item in rawData:
        for char in item[col]:
            alphabet.add(char)

    # Map columns to letters
    colMapping = {}
    size = 1
    for char in alphabet:
        colMapping[char] = size
        size += 1

    maxLen = 0
    for item in rawData:
        if RD['use_test_flags']:
            tStat = item[RD['label_count'] + RD['input_count']]
            if not (tStat == 0 or tStat == 1):
                continue
        maxLen = max(maxLen, len(item[col]))

    # DEBUG, data properties
    print("    Number of samples: {}".format(len(rawData)))
    print("    Maximum length of sample: {}".format(maxLen))
    print("    Size of alphabet: {}".format(size))

    if RD['alpha_fixed']:
        size = RD['alpha_fixed_size']
    output = np.zeros((len(rawData), maxLen, size))

    itemCtr = 0
    for item in rawData:
        charCtr = 0
        for char in item[col]:
            if charCtr >= maxLen:
                break
            charIdx = colMapping[char]
            output[itemCtr][charCtr][charIdx] = 1
            charCtr += 1
        for i in range(charCtr, maxLen):
            output[itemCtr][i][0] = 1
        itemCtr += 1

    print('  ...done')
    return size, maxLen, output


def formatNominal(rawData, timesteps, col):
    print('  Formatting nominal data column...')

    # Get a set of all possible values
    nominals = set()
    for item in rawData:
        nominals.add(item[col])

    # Map columns to nominals
    colMapping = {}
    size = 0
    for value in nominals:
        colMapping[value] = size
        size += 1

    print("    Number of samples: {}".format(len(rawData)))
    print("    Number of unique values: {}".format(size))

    itemCtr = 0
    output = np.zeros((len(rawData), timesteps, size))
    for item in rawData:
        valIdx = colMapping[item[col]]
        for step in range(timesteps):
            output[itemCtr][step][valIdx] = 1
        itemCtr += 1

    print('  ...done')
    return size, output


# Use 2D format, containing indexes of one hot bit, suitable for usage in
# Embedding layers
def formatSMILESEmbedded(rawData, col):
    print('  Formatting SMILES data column...')

    # Get a set of all used characters
    alphabet = set()
    for item in rawData:
        for char in item[col]:
            alphabet.add(char)

    # Map columns to letters
    colMapping = {}
    size = 1
    for char in alphabet:
        colMapping[char] = size
        size += 1

    maxLen = 0
    for item in rawData:
        maxLen = max(maxLen, len(item[col]))

    # DEBUG, data properties
    print("    Number of samples: {}".format(len(rawData)))
    print("    Maximum length of sample: {}".format(maxLen))
    print("    Size of alphabet: {}".format(size))

    output = np.zeros((len(rawData), maxLen))

    itemCtr = 0
    for item in rawData:
        charCtr = 0
        for char in item[col]:
            charIdx = colMapping[char]
            output[itemCtr][charCtr] = charIdx
            charCtr += 1
        for i in range(charCtr, maxLen):
            output[itemCtr][i] = 0
        itemCtr += 1

    print('  ...done')
    return size, maxLen, output


# Same as formatSmilesEmbedded, but uses words instead of characters
# Shift parameter us used to encode multiple columns in one value for Embedding
def formatNominalEmbedded(rawData, timesteps, output, col, shift = 0):
    print('  Formatting nominal data column...')

    # Get a set of all possible values
    nominals = set()
    for item in rawData:
        nominals.add(item[col])

    # Map columns to nominals
    colMapping = {}
    size = 1
    for value in nominals:
        colMapping[value] = size
        size += 1

    print("    Number of samples: {}".format(len(rawData)))
    print("    Number of unique values: {}".format(size))

    itemCtr = 0
    for item in rawData:
        valIdx = colMapping[item[col]]
        valIdx <<= shift
        for step in range(timesteps):
            output[itemCtr][step] += valIdx
        itemCtr += 1

    print('  ...done')
    return size, output


# Randomly choose n samples, with repetition
def randomSelection(n, words, ref):
    rWords = np.zeros((n, len(words[0]), len(words[0][0])))
    rRef = np.zeros(n)
    for i in range(n):
        rng = random.randint(0, len(words) - 1)
        rWords[i] = words[rng]
        rRef[i] = ref[rng]
    return rWords, rRef


# Splits the data into training and testing set
def holdout(ratio, words, label):
    if ratio >= 1 or ratio <= 0:
        raise ValueError('Ratio must be in (0, 1) interval')


    # Prepare training set
    trainSize = int(floor(len(words) * ratio))
    trainWords = np.zeros((trainSize, len(words[0]), len(words[0][0])))
    trainLabel = np.zeros((len(label), trainSize))
    for i in range(trainSize):
        trainWords[i] = words[i]
        for j in range(len(label)):
            trainLabel[j][i] = label[j][i]


    # Prepare testing set
    testSize = int(floor(len(words) - (len(words) * ratio)))
    testWords = np.zeros((testSize, len(words[0]), len(words[0][0])))
    testLabel = np.zeros((len(label), testSize))
    for i in range(testSize):
        testWords[i] = words[i + trainSize]
        for j in range(len(label)):
            testLabel[j][i] = label[j][i + trainSize]

    return trainWords, trainLabel, testWords, testLabel


# Bases holdout on column of dataset
# 1 - use in testing
# 0 - use in training
# null (None) - don't use
def holdoutBased(testFlags, words, label):
    testSize = 0
    trainSize = 0
    for i in range(len(testFlags)):
        if testFlags[i] == 1:
            testSize += 1
        elif testFlags[i] == 0:
            trainSize += 1
        elif not pd.isnull(testFlags[i]): # pandas treats Nones as NaNs
            raise ValueError("Unknown value in test flags: {}".format(testFlags[i]))

    if RD['use_embedding']:
        trainWords = np.zeros((trainSize, len(words[0])))
        trainLabel = np.zeros((len(label), trainSize))
        trainIdx = 0
        testWords = np.zeros((testSize, len(words[0])))
        testLabel = np.zeros((len(label), testSize))
        testIdx = 0
    else:
        trainWords = np.zeros((trainSize, len(words[0]), len(words[0][0])))
        trainLabel = np.zeros((len(label), trainSize))
        trainIdx = 0
        testWords = np.zeros((testSize, len(words[0]), len(words[0][0])))
        testLabel = np.zeros((len(label), testSize))
        testIdx = 0

    for i in range(len(words)):
        if testFlags[i] == 1:
            testWords[testIdx] = words[i]
            for j in range(len(label)):
                testLabel[j][testIdx] = label[j][i]
            testIdx += 1
        elif testFlags[i] == 0:
            trainWords[trainIdx] = words[i]
            for j in range(len(label)):
                trainLabel[j][trainIdx] = label[j][i]
            trainIdx += 1
    return trainWords, trainLabel, testWords, testLabel


# Preprocess data using logarithm
# NOTE: -log(x) apparently works better than log(x + 1)
def logarithm(array):
    loged = np.zeros(len(array))
    for i in range(len(array)):
        if array[i] > RD['eps']:
            loged[i] = -log(array[i])
    return loged


# Performs z-score normalization on given data
def zScoreNormalize(array):
    normalized = np.zeros(len(array))
    avg = utility.mean(array, len(array))
    dev = utility.stddev(array, len(array))
    for i in range(len(array)):
        normalized[i] = (array[i] - avg) / dev
        # print "Original ", array[i]
        # print "Normalized ", normalized[i]
    return normalized, avg, dev


# Undo normalization
def zScoreDenormalize(array, mean, dev):
    denormalized = np.zeros(len(array))
    for i in range(len(array)):
        denormalized[i] = (array[i] * dev) + mean
    return denormalized


# Replaces missing labels with zero
def resolveMissingLabels(labels):
    print('  Resolving missing labels...')
    for col in range(len(labels)):
        for i in range(len(labels[col])):
            if isnan(labels[col][i]):
                labels[col][i] = 0
    print('  ...done')


def getRawData(source = 'chembl', table = ''):
    if source == 'pubchem':
        data = pc.getData()
    elif source == 'chembl':
        if table == '':
            data = ch.getData()
        else:
            data = ch.getData(dbTable = table)
    else:
        raise ValueError('Unknown data source.')
    return data


# Call all routines to prepare data for neural network
def prepareData(source = 'chembl', table = ''):
    # changed from 'nan' to safer np.nan
    np.set_printoptions(threshold = np.nan, suppress = True)
    data = db.getData().values

    if not RD['use_embedding']:
        # SMILES column
        alphaSize, timesteps, formattedWords = formatSMILES(data, 0)
        # Nominal data columns
        nomiSize = 0
        if RD['extra_nominals'] > 0:
            n, formattedNominals = formatNominal(data, timesteps, 1)
            formattedWords = np.concatenate((formattedWords, formattedNominals),
                    axis = 2)
            nomiSize += n
            n, formattedNominals = formatNominal(data, timesteps, 2)
            formattedWords = np.concatenate((formattedWords, formattedNominals),
                    axis = 2)
            nomiSize += n
    else:
        nomiSize = 0
        alphaSize, timesteps, formattedWords = formatSMILESEmbedded(data, 0)

        # Shift defines offset inside of integer, used for coding multiple
        # small numeric values as one variable (needed in embedding)
        shift = int(log(alphaSize, 2) + 1)
        if RD['extra_nominals'] > 0:
            n, formattedWords = formatNominalEmbedded(data, timesteps, formattedWords,
                    1, shift)
            shift += int(log(n, 2) + 1)
            nomiSize += n
            n, formattedWords = formatNominalEmbedded(data, timesteps, formattedWords,
                    2, shift)
            shift += int(log(n, 2) + 1)
            nomiSize += n

    # put labels into array
    labels = []
    for i in range(RD['label_count']):
        labels.append(np.zeros((len(data))))
    i = 0
    for item in data:
        for labelID in range(RD['label_count']):
            labels[labelID][i] = item[labelID + RD['input_count']]
        i += 1
    resolveMissingLabels(labels)

    # put test flags into array - test flags are expected in last column
    testFlags = []
    if RD['use_test_flags']:
        for item in data:
            testFlags.append(item[RD['label_count'] + RD['input_count']])

    # include shift value to nomiSize if embedding is used
    if RD['use_embedding']:
        return formattedWords, labels, alphaSize, (nomiSize, shift), testFlags
    else:
        return formattedWords, labels, alphaSize, nomiSize, testFlags

