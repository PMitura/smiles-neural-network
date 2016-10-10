import random

import pubchem as pc

import numpy as np
import pandas as pd
import sklearn as sk

import utility

import db.db as db
from config import config as cc

from sets import Set

from math import ceil, log

RD = cc.exp['params']['data']
RP = cc.exp['params']['rnn']

SMILES_ALPHABET_UNKNOWN = '?'
SMILES_ALPHABET = [SMILES_ALPHABET_UNKNOWN,'-',
    '=','#','*','/','\\',
    '.','(',')','[',']',
    '{','}','@','+','0',
    '1','2','3','4','5',
    '6','7','8','9','a',
    'b','c','d','e','f',
    'g','h','i','j','k',
    'l','m','n','o','p',
    'q','r','s','t','u',
    'v','w','x','y','z',
    'A','B','C','D','E',
    'F','G','H','I','J',
    'K','L','M','N','O',
    'P','Q','R','S','T',
    'U','V','W','X','Y',
    'Z']


# old aphabet to use with some models
'''
SMILES_ALPHABET = [SMILES_ALPHABET_UNKNOWN,'-','=','#','*','.','(',')','[',']','{','}','-','+',
    '0','1','2','3','4','5','6','7','8','9',
    'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',
    'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
'''

SMILES_ALPHABET_LOOKUP_TABLE = { v:k for k,v in enumerate(SMILES_ALPHABET) }
SMILES_ALPHABET_LEN = len(SMILES_ALPHABET)
SMILES_ALPHABET_BITS = int(ceil(log(SMILES_ALPHABET_LEN,2)))

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

########################################################################################################################

def formatSequentialInput(df):
    numSamples = len(df)

    smilesDf = df[RD['smiles']]
    smilesMaxLen = max([len(x) for x in smilesDf])

    seqInput = np.zeros((numSamples, smilesMaxLen, SMILES_ALPHABET_LEN), dtype=bool)

    # translate to one hot for smiles
    for i,smiles in enumerate(smilesDf):
        for j in range(smilesMaxLen):

            transChar = SMILES_ALPHABET_LOOKUP_TABLE[SMILES_ALPHABET_UNKNOWN]
            if j < len(smiles) and smiles[j] in SMILES_ALPHABET_LOOKUP_TABLE:
                transChar = SMILES_ALPHABET_LOOKUP_TABLE[smiles[j]]

            seqInput[i][j][transChar] = 1

    # translate to one hot for nominals
    if RD['nominals'] and len(RD['nominals']) > 0:
        inputs = [seqInput]

        for nominal in RD['nominals']:
            nominalLookup = {x:i for (i,x) in enumerate(df[nominal].unique())}
            nominalSize = len(nominalLookup)

            nominalInput = np.zeros((numSamples, smilesMaxLen, nominalSize), dtype=bool)
            for i,nom in enumerate(df[nominal]):
                for j in range(smilesMaxLen):
                    nominalInput[i][j][nominalLookup[nom]] = 1

            inputs.append(nominalInput)

        # concatenate one hot representations
        seqInput = np.concatenate(tuple(inputs), axis = 2)

    return seqInput

########################################################################################################################

FASTA_ALPHABET_UNKNOWN = '?'
FASTA_ALPHABET = [FASTA_ALPHABET_UNKNOWN,
'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

FASTA_ALPHABET_LOOKUP_TABLE = { v:k for k,v in enumerate(FASTA_ALPHABET) }
FASTA_ALPHABET_LEN = len(FASTA_ALPHABET)
FASTA_ALPHABET_BITS = int(ceil(log(FASTA_ALPHABET_LEN,2)))

def formatFastaInput(df):
    numSamples = len(df)

    # FIXME: hardcoded fasta
    fastaDf = df[RD['fasta']]
    fastaMaxLen = max([len(x) for x in fastaDf])

    seqInput = np.zeros((numSamples, fastaMaxLen, FASTA_ALPHABET_LEN), dtype=bool)

    # translate to one hot for fasta
    for i,fasta in enumerate(fastaDf):
        for j in range(fastaMaxLen):

            transChar = FASTA_ALPHABET_LOOKUP_TABLE[FASTA_ALPHABET_UNKNOWN]
            if j < len(fasta) and fasta[j] in FASTA_ALPHABET_LOOKUP_TABLE:
                transChar = FASTA_ALPHABET_LOOKUP_TABLE[fasta[j]]

            seqInput[i][j][transChar] = 1

    return seqInput

########################################################################################################################

def normalize(arr):
    meta = {}

    if RP['logarithm']:
        mask = arr < RD['eps']
        arr = -np.log(arr)
        arr[mask] = 0

    if RP['zscore_norm']:
        meta['scaler'] = sk.preprocessing.StandardScaler()
        arr = meta['scaler'].fit_transform(arr)

    return arr, meta

def denormalize(arr, meta):
    if RP['zscore_norm']:
        arr = meta['scaler'].inverse_transform(arr)

    if RP['logarithm']:
        arr = np.exp(-arr)

    return arr

def labelBinning(labels):
    for i, label in enumerate(labels):
        labels[i] = utility.bin(labels[i], RP['label_binning_ratio'],
                classA = RP['classify_label_neg'], classB = RP['classify_label_pos'])

    return labels

def preprocessData(df):
    # filter out inf and NaN (nulls) values
    df = df.replace([np.inf, -np.inf],np.nan).dropna()
    df.reset_index(drop=True, inplace=True)

    # filter out rows with malformed test_flags, if we use them
    if RD['use_test_flags']:
        df = df[(df[RD['testing']] == 0) | (df[RD['testing']] == 1)]
        df.reset_index(drop=True, inplace=True)

    # nicely split data
    input = formatSequentialInput(df)
    labels = df[RD['labels']].values
    testing = df[RD['testing']].values.astype(bool)

    # preprocessing
    labels, meta = normalize(labels)

    if RP['label_binning'] and RP['classify']:
        labels = labelBinning(labels)

    # create training and testing sets
    if RP['flag_based_hold']:
        trainIn, trainLabel = input[~testing], labels[~testing]
        testIn, testLabel = input[testing], labels[testing]
    else:
        split = int(len(input) * RP['holdout_ratio'])

        trainIn, trainLabel = input[:split], labels[:split]
        testIn, testLabel = input[split:], labels[split:]

    return trainIn,trainLabel,testIn,testLabel,meta

def preprocessFastaOneHotData(df):
    # filter out inf and NaN (nulls) values
    df = df.replace([np.inf, -np.inf],np.nan).dropna()
    df.reset_index(drop=True, inplace=True)

    # filter out rows with malformed test_flags, if we use them
    if RD['use_test_flags']:
        df = df[(df[RD['testing']] == 0) | (df[RD['testing']] == 1)]
        df.reset_index(drop=True, inplace=True)

    # inputSmiles = formatSequentialInput(df)
    inputFasta = formatFastaInput(df)

    labels = df[RD['labels']].values

    # preprocessing
    labels, meta = normalize(labels)

    # FIXME or abandon branch: hardcoded pasta stuff
    '''
    pfSet = set()
    for label in labels:
        pfSet.add(label[0])
    pfMapping = {}
    size = 0
    for value in pfSet:
        pfMapping[value] = size
        size += 1
    print size
    newLabels = np.zeros((len(labels), size), dtype=bool)
    for i in range(len(labels)):
        newLabels[i][pfMapping[labels[i][0]]] = 1
    labels = newLabels
    '''

    # create training and testing sets
    if RP['flag_based_hold']:
        testing = df[RD['testing']].values.astype(bool)
        trainFastaIn, trainLabel = inputFasta[~testing], labels[~testing]
        testFastaIn, testLabel   = inputFasta[testing],  labels[testing]
    else:
        split = int(len(inputFasta) * RP['holdout_ratio'])

        trainFastaIn, trainLabel = inputFasta[:split], labels[:split]
        testFastaIn, testLabel   = inputFasta[split:], labels[split:]

    return trainFastaIn, trainLabel, testFastaIn, testLabel, meta


def preprocessEdgeData(df):
    # filter out inf and NaN (nulls) values
    df = df.replace([np.inf, -np.inf],np.nan).dropna()
    df.reset_index(drop=True, inplace=True)

    # filter out rows with malformed test_flags, if we use them
    if RD['use_test_flags']:
        df = df[(df[RD['testing']] == 0) | (df[RD['testing']] == 1)]
        df.reset_index(drop=True, inplace=True)

    # shuffle the data
    df.reindex(np.random.permutation(df.index))

    inputSmiles = formatSequentialInput(df)
    inputFasta = formatFastaInput(df)

    labels = df[RD['labels']].values

    # preprocessing
    labels, meta = normalize(labels)

    # create training and testing sets
    if RP['flag_based_hold']:
        testing = df[RD['testing']].values.astype(bool)
        trainSmilesIn, trainFastaIn, trainLabel = inputSmiles[~testing], inputFasta[~testing], labels[~testing]
        testSmilesIn,  testFastaIn,  testLabel  = inputSmiles[testing], inputFasta[testing],  labels[testing]
    else:
        split = int(len(inputSmiles) * RP['holdout_ratio'])

        trainSmilesIn, trainFastaIn, trainLabel = inputSmiles[:split], inputFasta[:split], labels[:split]
        testSmilesIn,  testFastaIn,  testLabel  = inputSmiles[split:], inputFasta[split:], labels[split:]

    return [trainSmilesIn, trainFastaIn], trainLabel, [testSmilesIn, testFastaIn], testLabel, meta