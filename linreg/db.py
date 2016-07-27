
import mysql.connector

# login credentials
DB_USER = 'guest'
DB_PASS = 'relational'
DB_HOST = 'relational.fit.cvut.cz'
DB_NAME = 'ctu_qsar'

def fetchData(query_str):
    print('Downloading data...')
    cnx = mysql.connector.connect(user = DB_USER, password = DB_PASS, host = DB_HOST, database= DB_NAME)
    query = (query_str)
    cursor = cnx.cursor()
    cursor.execute(query)

    array = cursor.fetchall()
    cnx.close()
    print('...done')
    return array

def getTargetProteinBcdBest5():
    DB_TABLE = 'target_protein_bcd_best5'
    DB_COLS = 'canonical_smiles,Chi3v,Chi4v,SlogP_VSA10,NumAromaticCarbocycles,fr_benzene,standard_value_log_median_centered,is_testing_99_short_NP_705927'
    CAP_SIZE = 1887
    return fetchData('SELECT {} FROM {} LIMIT {}'.format(DB_COLS, DB_TABLE, CAP_SIZE))


def getTargetProteinBcdRdkitFeatureSelection():
    DB_TABLE = 'target_protein_big_cleaned_deduplicated'
    DB_COLS = 'canonical_smiles,standard_value_log_median_centered,is_testing_99_short_NP_705927'
    CAP_SIZE = 1000
    return fetchData('SELECT {} FROM {} WHERE is_testing_99_short_NP_705927 IS NOT NULL LIMIT {}'.format(DB_COLS, DB_TABLE, CAP_SIZE))
    