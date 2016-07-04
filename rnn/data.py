import mysql.connector

# login credentials
DB_USER = 'petermitura'
DB_PASS = 'qsar'
DB_HOST = 'relational.fit.cvut.cz'
DB_NAME = 'ctu_qsar'

# data range
DB_TABLE = 'target_molweight_1000'
DB_COLS = 'canonical_smiles, molweight'


# Connects to remote DB, reads input data into array.
def getDataFromDb():
    print('  Downloading data...')

    cnx = mysql.connector.connect(user=DB_USER, password=DB_PASS,
            host=DB_HOST, database=DB_NAME)
    query = ('SELECT {} FROM {}'.format(DB_COLS, DB_TABLE))
    cursor = cnx.cursor()
    cursor.execute(query)

    array = []
    for canonical_smiles, molweight in cursor:
        array.append((canonical_smiles, molweight))
    cursor.close()
    cnx.close()

    print('  ...done')
    return array


# Transforms data into 1 of k encoding
# Output format is 2D array of integers, representing positions of binary 1
def formatData(rawData):
    print('  Formatting data...')

    # Get a set of all used characters
    alphabet = set()
    for item in rawData:
        for char in item[0]:
            alphabet.add(char)

    # Map columns to letters
    colMapping = {}
    ctr = 0
    for char in alphabet:
        colMapping[char] = ctr
        ctr += 1

    output = []
    for item in rawData:
        itemArray = []
        for char in item[0]:
            itemArray.append(colMapping[char])
        output.append(itemArray)

    print('  ...done')
    return output


def prepareData():
    data = getDataFromDb()
    formattedWords = formatData(data)
    reference = []
    for item in data:
        reference.append(item[1])
    """ DEBUG
    for item in formattedData:
        print item
    """
    return formattedWords, reference


