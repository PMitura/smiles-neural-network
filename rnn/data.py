import numpy as np
import random
import pubchem as pc
import chembl as ch
import utility

from math import floor, log, isnan, sqrt

# Number of label columns to prepare
INPUT_COUNT = 3
LABEL_COUNT = 1

# Epsilon for catching numbers close to zero
EPS = 0.0001

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
        maxLen = max(maxLen, len(item[col]))

    # DEBUG, data properties
    print("    Number of samples: {}".format(len(rawData)))
    print("    Maximum length of sample: {}".format(maxLen))
    print("    Size of alphabet: {}".format(size))
    output = np.zeros((len(rawData), maxLen, size))

    itemCtr = 0
    for item in rawData:
        charCtr = 0
        for char in item[col]:
            charIdx = colMapping[char]
            output[itemCtr][charCtr][charIdx] = 1
            charCtr += 1
        for i in range(charCtr, maxLen):
            output[itemCtr][i][0] = 1
        itemCtr += 1

    """ DEBUG: print whole data
    np.set_printoptions(threshold='nan')
    print output[0]
    print output
    """

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


# Randomly choose n samples, with repetition
def randomSelection(n, words, ref):
    rWords = np.zeros((n, len(words[0]), len(words[0][0])))
    rRef = np.zeros((n))
    for i in range(n):
        rng = random.randint(0, len(words) - 1)
        rWords[i] = words[rng]
        rRef[i] = ref[rng]
    return rWords, rRef


# Splits the data into training and testing set
def holdout(ratio, words, ref):
    if ratio >= 1 or ratio <= 0:
        raise ValueError('Ratio must be in (0, 1) interval')

    # Prepare training set
    trainSize = int(floor(len(words) * ratio))
    trainWords = np.zeros((trainSize, len(words[0]), len(words[0][0])))
    trainRef = np.zeros((trainSize))
    for i in range(trainSize):
        trainWords[i] = words[i]
        trainRef[i] = ref[i]

    # Prepare testing set
    testSize = int(floor(len(words) - (len(words) * ratio)))
    testWords = np.zeros((testSize, len(words[0]), len(words[0][0])))
    testRef = np.zeros((testSize))
    for i in range(testSize):
        testWords[i] = words[i + testSize]
        testRef[i] = ref[i + testSize]

    return trainWords, trainRef, testWords, testRef


# Preprocess data using logarithm
def logarithm(array):
    loged = np.zeros(len(array))
    for i in range(len(array)):
        if array[i] > EPS:
            loged[i] = log(array[i])
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


# Call all routines to prepare data for neural network
def prepareData(source = 'chembl'):
    np.set_printoptions(threshold='nan')

    if source == 'pubchem':
        data = pc.getData()
    elif source == 'chembl':
        data = ch.getData()
    else:
        raise ValueError('Unknown data source.')

    # SMILES column
    alphaSize, timesteps, formattedWords = formatSMILES(data, 0)
    # Nominal data columns
    nomiSize = 0.0
    n, formattedNominals = formatNominal(data, timesteps, 1)
    formattedWords = np.concatenate((formattedWords, formattedNominals),
            axis = 2)
    nomiSize += n
    n, formattedNominals = formatNominal(data, timesteps, 2)
    formattedWords = np.concatenate((formattedWords, formattedNominals),
            axis = 2)
    nomiSize += n

    """ DEBUG: print sample row
    print formattedWords[30]
    """

    labels = []
    for i in range(LABEL_COUNT):
        labels.append(np.zeros((len(data))))
    i = 0
    for item in data:
        for labelID in range(LABEL_COUNT):
            labels[labelID][i] = item[labelID + INPUT_COUNT]
        i += 1
    resolveMissingLabels(labels)

    return formattedWords, labels, alphaSize, nomiSize

