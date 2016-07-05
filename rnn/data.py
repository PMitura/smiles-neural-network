import mysql.connector
import numpy as np

# login credentials
DB_USER = 'petermitura'
DB_PASS = 'qsar'
DB_HOST = 'relational.fit.cvut.cz'
DB_NAME = 'ctu_qsar'

# data range
# DB_TABLE = 'target_molweight_1000'
# DB_COLS = 'canonical_smiles, molweight'
DB_TABLE = 'target_properties_1000'
DB_COLS = 'canonical_smiles, mw_freebase, alogp'

# Connects to remote DB, reads input data into array.
def getDataFromDb():
    print('  Downloading data...')

    cnx = mysql.connector.connect(user = DB_USER, password = DB_PASS,
            host = DB_HOST, database= DB_NAME)
    query = ('SELECT {} FROM {}'.format(DB_COLS, DB_TABLE))
    cursor = cnx.cursor()
    cursor.execute(query)

    array = []
    for a, b, c in cursor:
        array.append((a, b, c))
    cursor.close()
    cnx.close()

    print('  ...done')
    return array


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


def prepareData():
    data = getDataFromDb()
    alphaSize, formattedWords = formatData(data)
    reference = np.zeros((len(data), 2))
    i = 0
    for item in data:
        reference[i][0] = item[1]
        reference[i][1] = item[2]
        i += 1

    """ DEBUG
    for item in formattedWords:
        print item
    """

    return formattedWords, reference, alphaSize


