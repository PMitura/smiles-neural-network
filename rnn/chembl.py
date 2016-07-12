import mysql.connector

# login credentials
DB_USER = 'petermitura'
DB_PASS = 'qsar'
DB_HOST = 'relational.fit.cvut.cz'
DB_NAME = 'ctu_qsar'

# data range
# DB_TABLE = 'target_molweight_1000'
# DB_TABLE = 'target_properties_norm_1000'
DB_TABLE = 'target_206_1683'

# DB_COLS = 'canonical_smiles, molweight'
# DB_COLS = 'canonical_smiles, mw_freebase, alogp'
# DB_COLS = 'canonical_smiles, hba, hbd'
DB_COLS = 'canonical_smiles, log_value, standard_value'

# Connects to remote DB, reads input data into array.
def getData():
    print('  Downloading data...')

    cnx = mysql.connector.connect(user = DB_USER, password = DB_PASS,
            host = DB_HOST, database= DB_NAME)
    query = ('SELECT {} FROM {}'.format(DB_COLS, DB_TABLE))
    cursor = cnx.cursor()
    cursor.execute(query)

    array = []
    # care, this is still fixed at LABEL_SIZE = 2
    # TODO: generalize
    for a, b, c in cursor:
        array.append((a, b, c))
    cursor.close()
    cnx.close()

    print('  ...done')
    return array
