import mysql.connector

# login credentials
DB_USER = 'guest'
DB_PASS = 'relational'
DB_HOST = 'relational.fit.cvut.cz'
DB_NAME = 'ctu_qsar'

# DB_TABLE = 'target_molweight_1000'
# DB_TABLE = 'target_protein_1000'
# DB_TABLE = 'target_206_1683'
# DB_TABLE = 'target_206_1977'
DB_TABLE = 'target_protein_big_cleaned_log'
# DB_TABLE = 'target_protein_big_cleaned_deduplicated'
# DB_TABLE = 'target_protein_p03372_ic50_binary'

# DB_COLS = 'canonical_smiles, molweight'
# DB_COLS = 'canonical_smiles, mw_freebase, alogp'
# DB_COLS = 'canonical_smiles, hba, hbd'
# DB_COLS = 'canonical_smiles, log_value, standard_value'
DB_COLS = 'canonical_smiles, standard_type, protein_accession,\
    standard_value_log_avg, is_testing'
#     standard_value_log_median_centered, is_testing'
# DB_COLS = 'canonical_smiles, standard_value_50'
# DB_COLS = 'canonical_smiles, standard_value, is_testing'

# maximum number of downloaded rows
CAP_SIZE = 1000

# sending options
SEND_TABLE = 'journal'

# Connects to remote DB, reads input data into array.
def getData(dbCols = DB_COLS, dbTable = DB_TABLE):
    print('  Downloading data...')

    cnx = mysql.connector.connect(user = DB_USER, password = DB_PASS,
            host = DB_HOST, database= DB_NAME)
    query = ('SELECT {} FROM {} LIMIT {}'.format(dbCols, dbTable, CAP_SIZE))
    cursor = cnx.cursor()
    cursor.execute(query)

    array = []
    # care, this is still fixed at certain number of cols
    # TODO: generalize

    for a, b, c, d, e in cursor:
        array.append((a, b, c, d, e))
    cursor.close()
    cnx.close()

    print('  ...done')
    return array


def sendStatistics(dataset_name = DB_TABLE,
        training_row_count = None,
        task = None,                    # classification / regression
        relevance_training = None,      # R2 or AUC
        relevance_testing = None,       # -||-
        comment = None,
        epoch_count = None,
        runtime_second = None,
        parameter_count = None,
        learning_rate = None,
        optimization_method = None,
        batch_size = None):
    print('  Sending statistics...')

    cnx = mysql.connector.connect(user = DB_USER, password = DB_PASS,
            host = DB_HOST, database = DB_NAME)

    query = ('INSERT INTO {}\
            (training_row_count, task, relevance_testing,\
            relevance_training, comment, epoch_count, runtime_second,\
            parameter_count, learning_rate, optimization_method, batch_size)\
            VALUES\
            ({}, \'{}\', {}, {}, \'{}\', {}, {}, {}, {}, \'{}\', {})'.format(SEND_TABLE,
            training_row_count, task, relevance_testing,
            relevance_training, comment, epoch_count, runtime_second,
            parameter_count, learning_rate, optimization_method, batch_size))
    cursor = cnx.cursor()
    try:
        cursor.execute(query)
        cnx.commit()
    except:
        cnx.rollback()
        print '    EXCEPTED'

    cursor.close()
    cnx.close()
    print('  ...done')