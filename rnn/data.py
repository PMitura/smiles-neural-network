import numpy as np
import random
import pubchem as pc

import pandas as pd
import utility

import db.db as db
from config import config as cc

from math import floor, log, isnan, sqrt, ceil

from sets import Set

RD = cc.exp['params']['data']

SMILES_ALPHABET_UNKNOWN = '?'
SMILES_ALPHABET = [SMILES_ALPHABET_UNKNOWN,'-','=','#','*','.','(',')','[',']','{','}','-','+',
    '0','1','2','3','4','5','6','7','8','9',
    'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',
    'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
SMILES_ALPHABET_LOOKUP_TABLE = { v:k for k,v in enumerate(SMILES_ALPHABET) }
SMILES_ALPHABET_LEN = len(SMILES_ALPHABET)
SMILES_ALPHABET_BITS = int(ceil(log(SMILES_ALPHABET_LEN,2)))

# Transforms data into 1 of k encoding
# Output format is 3D array of integers, representing positions of binary 1
def formatSMILES(rawData, col):
    print('  Formatting SMILES data column...')


    maxLen = 0
    for item in rawData:
        if RD['use_test_flags']:
            tStat = item[RD['label_count'] + RD['input_count'] + RD['extra_nominals']]
            if not (tStat == 0 or tStat == 1):
                continue
        maxLen = max(maxLen, len(item[col]))

    # DEBUG, data properties
    print("    Number of samples: {}".format(len(rawData)))
    print("    Maximum length of sample: {}".format(maxLen))
    print("    Size of alphabet: {}".format(SMILES_ALPHABET_LEN))

    output = np.zeros((len(rawData), maxLen, SMILES_ALPHABET_LEN))

    for itemIdx,item in enumerate(rawData):
        for charIdx,char in enumerate(item[col]):
            if charIdx >= maxLen:
                break
            char = char if char in SMILES_ALPHABET_LOOKUP_TABLE else SMILES_ALPHABET_UNKNOWN
            output[itemIdx][charIdx][SMILES_ALPHABET_LOOKUP_TABLE[char]] = 1

        for i in range(len(item[col]), maxLen):
            output[itemIdx][i][SMILES_ALPHABET_LOOKUP_TABLE[SMILES_ALPHABET_UNKNOWN]] = 1

    print('  ...done')
    return SMILES_ALPHABET_LEN, maxLen, output


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

    maxLen = 0
    for item in rawData:
        maxLen = max(maxLen, len(item[col]))

    # DEBUG, data properties
    print("    Number of samples: {}".format(len(rawData)))
    print("    Maximum length of sample: {}".format(maxLen))
    print("    Size of alphabet: {}".format(SMILES_ALPHABET_LEN))

    output = np.zeros((len(rawData), maxLen))

    for itemIdx,item in enumerate(rawData):
        for charIdx,char in enumerate(item[col]):
            char = char if char in SMILES_ALPHABET_LOOKUP_TABLE else SMILES_ALPHABET_UNKNOWN
            output[itemIdx][charIdx] = SMILES_ALPHABET_LOOKUP_TABLE[char]
        for i in range(len(item[col]), maxLen):
            output[itemIdx][i] = SMILES_ALPHABET_LOOKUP_TABLE[SMILES_ALPHABET_UNKNOWN]

    print('  ...done')
    return SMILES_ALPHABET_LEN, maxLen, output


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
    size = 0
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

    splitPoint = int(floor(len(words) * ratio))

    trainWords = words[:splitPoint]
    trainLabel = [lab[:splitPoint] for lab in label]

    testWords = words[splitPoint:]
    testLabel = [lab[splitPoint:] for lab in label]

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
        elif testFlags[i] == 0 or pd.isnull(testFlags[i]):
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
        elif testFlags[i] == 0 or pd.isnull(testFlags[i]):
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

    # Maybe hacky, but is in theory right
    if dev == 0:
        return normalized, avg, dev

    for i in range(len(array)):
        if pd.isnull(array[i]):
            raise ValueError('Cannot normalize \"None\" value')
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
            for i in range(RD['extra_nominals']):
                n, formattedNominals = formatNominal(data, timesteps, 1+i)
                formattedWords = np.concatenate((formattedWords, formattedNominals),
                        axis = 2)
                nomiSize += n
            #n, formattedNominals = formatNominal(data, timesteps, 2)
            #formattedWords = np.concatenate((formattedWords, formattedNominals),
            #        axis = 2)
            #nomiSize += n
    else:
        nomiSize = 0
        alphaSize, timesteps, formattedWords = formatSMILESEmbedded(data, 0)


        # Shift defines offset inside of integer, used for coding multiple
        # small numeric values as one variable (needed in embedding)
        shift = SMILES_ALPHABET_BITS
        if RD['extra_nominals'] > 0:
            for i in range(RD['extra_nominals']):
                n, formattedWords = formatNominalEmbedded(data, timesteps, formattedWords,
                        1+i, shift)

                shift += int(log(n, 2) + 1)
                nomiSize += n
            #n, formattedWords = formatNominalEmbedded(data, timesteps, formattedWords,
            #        2, shift)
            #shift += int(log(n, 2) + 1)
            #nomiSize += n



    # put labels into array
    labels = []
    for i in range(RD['label_count']):
        labels.append(np.zeros((len(data))))
    i = 0
    for item in data:
        for labelID in range(RD['label_count']):
            labels[labelID][i] = item[labelID + RD['input_count'] + RD['extra_nominals']]
        i += 1
    resolveMissingLabels(labels)

    # put test flags into array - test flags are expected in last column
    testFlags = []
    if RD['use_test_flags']:
        for item in data:
            testFlags.append(item[RD['label_count'] + RD['input_count'] + RD['extra_nominals']])

    # include shift value to nomiSize if embedding is used
    if RD['use_embedding']:
        return formattedWords, labels, alphaSize, (nomiSize, shift), testFlags
    else:
        return formattedWords, labels, alphaSize, nomiSize, testFlags

