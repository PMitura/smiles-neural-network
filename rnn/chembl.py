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
DB_TABLE = 'target_206_1977_features_wide'
# DB_TABLE = 'target_protein_big_cleaned_log'
# DB_TABLE = 'target_protein_big_cleaned_deduplicated'
# DB_TABLE = 'target_molweight'
# DB_TABLE = 'target_protein_p03372_ic50_binary'
# DB_TABLE = 'target_protein_bcd_best5'

# DB_COLS = 'canonical_smiles, molweight'
# DB_COLS = 'canonical_smiles, mw_freebase, alogp'
# DB_COLS = 'canonical_smiles, hba, hbd'
# DB_COLS = 'canonical_smiles, log_value, standard_value'
# LABELNAME = 'MOLWEIGHT, ALOGP, STANDARD_VALUE'
LABELNAME = 'standard_value_log,MinAbsPartialCharge,HeavyAtomMolWt,MaxAbsPartialCharge,MinAbsEStateIndex,Chi3n,HallKierAlpha,PEOE_VSA1,PEOE_VSA10,PEOE_VSA11,PEOE_VSA12,PEOE_VSA13,PEOE_VSA14,PEOE_VSA2,PEOE_VSA3,PEOE_VSA6,PEOE_VSA8,PEOE_VSA9,SMR_VSA1,SMR_VSA10,SMR_VSA3,SMR_VSA6,SMR_VSA9,SlogP_VSA10,SlogP_VSA3,SlogP_VSA4,SlogP_VSA6,TPSA,EState_VSA3,EState_VSA5,EState_VSA7,EState_VSA8,VSA_EState9,NHOHCount,NumAliphaticHeterocycles,NumAromaticHeterocycles,MolLogP,fr_Ar_COO,fr_C_O,fr_Imine,fr_NH1,fr_Ndealkylation2,fr_amide,fr_aryl_methyl,fr_ester,fr_ether,fr_furan,fr_imidazole,fr_methoxy,fr_piperzine,fr_pyridine,fr_sulfide,fr_thiazole,fr_urea'
TESTNAME = 'is_testing'
# DB_COLS = 'canonical_smiles, standard_type, protein_accession,\
DB_COLS = 'CANONICAL_SMILES,\
        {}, {}'.format(LABELNAME, TESTNAME)
# DB_COLS = 'canonical_smiles, standard_value_50'
# DB_COLS = 'canonical_smiles, standard_value, is_testing'

# maximum number of downloaded rows
CAP_SIZE = 10000

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

    # care, this is still fixed at certain number of cols
    # TODO: generalize

    # array = []
    # for a, b, c, d, e in cursor:
    #     array.append((a, b, c, d, e))
    array = cursor.fetchall()
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
        relevance_testing_std = None,   # -||-
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
        split_name = TESTNAME,
        memory_pm_mb = None,
        memory_vm_mb = None,
        learning_curve = None,
        hostname = None):
    print('  Sending statistics...')

    cnx = mysql.connector.connect(user = DB_USER, password = DB_PASS,
            host = DB_HOST, database = DB_NAME)

    query = 'INSERT INTO {}\
            (dataset_name, training_row_count, testing_row_count, task,\
            relevance_testing, relevance_training, relevance_testing_std,\
            comment, epoch_max,\
            epoch_count ,runtime_second, parameter_count, learning_rate,\
            optimization_method, batch_size, label_name, model, seed,\
            split_name, memory_pm_mb, memory_vm_mb, learning_curve, hostname)\
            VALUES\
            (\'{}\', {}, {}, \'{}\', {}, {}, {}, \'{}\', {}, {}, {}, {}, {},\
            \'{}\', {}, \'{}\', \"{}\", {}, \"{}\", {}, {}, (%s), \"{}\")'.format(SEND_TABLE,
            dataset_name, training_row_count, testing_row_count, task,
            relevance_testing, relevance_training, relevance_testing_std,
            comment, epoch_max, epoch_count,
            runtime_second, parameter_count, learning_rate,
            optimization_method, batch_size, label_name, model, seed,
            split_name, memory_pm_mb, memory_vm_mb, hostname)
    cursor = cnx.cursor()
    try:
        cursor.execute(query, (learning_curve,))
        cnx.commit()
    except Exception as e:
        cnx.rollback()
        print '    EXCEPTED:'
        print e

    cursor.close()
    cnx.close()
    print('  ...done')
