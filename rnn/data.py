import numpy as np
import random
import pubchem as pc
import chembl as ch

from math import floor

# Number of label columns to prepare
LABEL_COUNT = 2

# Transforms data into 1 of k encoding
# Output format is 3D array of integers, representing positions of binary 1
def formatData(rawData):
    print('  Formatting data...')

    # Get a set of all used characters
    alphabet = set()
    for item in rawData:
        for char in item[0]:
            alphabet.add(char)

    # Map columns to letters
    colMapping = {}
    size = 1
    for char in alphabet:
        colMapping[char] = size
        size += 1

    maxLen = 0
    for item in rawData:
        maxLen = max(maxLen, len(item[0]))
    padArray = np.zeros((1, size))
    padArray[0] = 1

    # DEBUG, data properties
    print("    Number of samples: {}".format(len(rawData)))
    print("    Maximum length of sample: {}".format(maxLen))
    print("    Size of alphabet: {}".format(size))
    output = np.zeros((len(rawData), maxLen, size))

    itemCtr = 0
    for item in rawData:
        charCtr = 0
        for char in item[0]:
            charIdx = colMapping[char]
            output[itemCtr][charCtr][charIdx] = 1
            charCtr += 1
        for i in range(charCtr, maxLen):
            output[itemCtr][charCtr][0] = 1
        itemCtr += 1

    """ DEBUG: print whole data
    np.set_printoptions(threshold='nan')
    print output
    """

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


# Call all routines to prepare data for neural network
def prepareData(source = 'chembl'):
    if source == 'pubchem':
        data = pc.getData()
    elif source == 'chembl':
        data = ch.getData()
    else:
        raise ValueError('Unknown data source.')

    alphaSize, formattedWords = formatData(data)

    labels = []
    for i in range(LABEL_COUNT):
        labels.append(np.zeros((len(data))))
    i = 0
    for item in data:
        for labelID in range(LABEL_COUNT):
            labels[labelID][i] = item[labelID + 1]
        i += 1

    """ DEBUG
    for item in formattedWords:
        print item
    """

    return formattedWords, labels, alphaSize

