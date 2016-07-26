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
# DB_TABLE = 'target_protein_big_cleaned_log'
DB_TABLE = 'target_protein_big_cleaned_deduplicated'
# DB_TABLE = 'target_protein_p03372_ic50_binary'

# DB_COLS = 'canonical_smiles, molweight'
# DB_COLS = 'canonical_smiles, mw_freebase, alogp'
# DB_COLS = 'canonical_smiles, hba, hbd'
# DB_COLS = 'canonical_smiles, log_value, standard_value'
LABELNAME = 'standard_value_log_median_centered'
TESTNAME = 'is_testing_99_short'
DB_COLS = 'canonical_smiles, standard_type, protein_accession,\
        {}, {}'.format(LABELNAME, TESTNAME)
# DB_COLS = 'canonical_smiles, standard_value_50'
# DB_COLS = 'canonical_smiles, standard_value, is_testing'

# maximum number of downloaded rows
CAP_SIZE = 5000

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
        testing_row_count = None,
        task = None,                    # classification / regression
        relevance_training = None,      # R2 or AUC
        relevance_testing = None,       # -||-
        relevance_testing_std = None,       # -||-
        comment = None,
        epoch_max = None,
        epoch_count = None,
        runtime_second = None,
        parameter_count = None,
        learning_rate = None,
        optimization_method = None,
        batch_size = None,
        label_name = None,
        model = None,
        seed = None,
        split_name = TESTNAME):
    print('  Sending statistics...')

    cnx = mysql.connector.connect(user = DB_USER, password = DB_PASS,
            host = DB_HOST, database = DB_NAME)

    query = 'INSERT INTO {}\
            (dataset_name, training_row_count, testing_row_count, task,\
            relevance_testing, relevance_training, relevance_testing_std,\
            comment, epoch_max,\
            epoch_count ,runtime_second, parameter_count, learning_rate,\
            optimization_method, batch_size, label_name, model, seed,\
            split_name)\
            VALUES\
            (\'{}\', {}, {}, \'{}\', {}, {}, {}, \'{}\', {}, {}, {}, {}, {},\
            \'{}\', {}, \'{}\', \"{}\", {}, \"{}\")'.format(SEND_TABLE,
            dataset_name, training_row_count, testing_row_count, task,
            relevance_testing, relevance_training, relevance_testing_std,
            comment, epoch_max, epoch_count,
            runtime_second, parameter_count, learning_rate,
            optimization_method, batch_size, label_name, model, seed,
            split_name)
    cursor = cnx.cursor()
    try:
        cursor.execute(query)
        cnx.commit()
    except Exception as e:
        cnx.rollback()
        print '    EXCEPTED:'
        print e

    cursor.close()
    cnx.close()
    print('  ...done')
