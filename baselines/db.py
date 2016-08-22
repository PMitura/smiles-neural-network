import sys
sys.path.append('..')

from config import config as cc
import psycopg2
import mysql.connector
import pandas as pd
import numpy


def getCon():
    dbcfg = {
        'user': cc.cfg['db']['user'],
        'password': cc.cfg['db']['pass'],
        'host': cc.cfg['db']['host'],
        'database': cc.cfg['db']['name']
    }

    if cc.cfg['db']['driver']=='postgresql':
        con = psycopg2.connect(**dbcfg)
    elif cc.cfg['db']['driver']=='mysql':
        con = mysql.connector.connect(**dbcfg)
    else:
        raise Exception('getCon: unknown db driver {}'.format(cc.cfg['db']['driver']))

    return con

def fetchData(query_str):
    print('Downloading data...')
    cnx = getCon()
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
    CAP_SIZE = 1887
    return fetchData('SELECT {} FROM {} WHERE is_testing_99_short_NP_705927 IS NOT NULL LIMIT {}'.format(DB_COLS, DB_TABLE, CAP_SIZE))


def getTarget_206_1977():
    DB_TABLE = 'target_206_1977'
    DB_COLS = 'canonical_smiles,standard_value'
    CAP_SIZE = 1976
    # CAP_SIZE = 10
    return fetchData('SELECT {} FROM {} LIMIT {}'.format(DB_COLS, DB_TABLE, CAP_SIZE))

def getTarget_geminin():
    DB_TABLE = 'output.target_geminin_deduplicated'
    DB_COLS = 'canonical_smiles,standard_value_median'
    # CAP_SIZE = 1976
    CAP_SIZE = 100000
    return fetchData('SELECT {} FROM {} LIMIT {}'.format(DB_COLS, DB_TABLE, CAP_SIZE))