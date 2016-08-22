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

def getTarget_206_1977_features_wide():
    DB_TABLE = 'output.target_206_1977_features'

    cols = ['smiles','sval','MinAbsPartialCharge','HeavyAtomMolWt','MaxAbsPartialCharge','MinAbsEStateIndex','Chi3n','HallKierAlpha','PEOE_VSA1','PEOE_VSA10','PEOE_VSA11','PEOE_VSA12','PEOE_VSA13','PEOE_VSA14','PEOE_VSA2','PEOE_VSA3','PEOE_VSA6','PEOE_VSA8','PEOE_VSA9','SMR_VSA1','SMR_VSA10','SMR_VSA3','SMR_VSA6','SMR_VSA9','SlogP_VSA10','SlogP_VSA3','SlogP_VSA4','SlogP_VSA6','TPSA','EState_VSA3','EState_VSA5','EState_VSA7','EState_VSA8','VSA_EState9','NHOHCount','NumAliphaticHeterocycles','NumAromaticHeterocycles','MolLogP','fr_Ar_COO','fr_C_O','fr_Imine','fr_NH1','fr_Ndealkylation2','fr_amide','fr_aryl_methyl','fr_ester','fr_ether','fr_furan','fr_imidazole','fr_methoxy','fr_piperzine','fr_pyridine','fr_sulfide','fr_thiazole','fr_urea']

    DB_COLS = ','.join(['"{}"'.format(x) for x in cols])
    # CAP_SIZE = 1976
    CAP_SIZE = 2000
    return fetchData('SELECT {} FROM {} LIMIT {}'.format(DB_COLS, DB_TABLE, CAP_SIZE))

def getTarget_geminin():
    DB_TABLE = 'output.target_geminin_deduplicated_features_wide'

    cols = ['canonical_smiles','standard_value_median','MinAbsPartialCharge','HeavyAtomMolWt','MaxAbsPartialCharge','MinAbsEStateIndex','Chi3n','HallKierAlpha','PEOE_VSA1','PEOE_VSA10','PEOE_VSA11','PEOE_VSA12','PEOE_VSA13','PEOE_VSA14','PEOE_VSA2','PEOE_VSA3','PEOE_VSA6','PEOE_VSA8','PEOE_VSA9','SMR_VSA1','SMR_VSA10','SMR_VSA3','SMR_VSA6','SMR_VSA9','SlogP_VSA10','SlogP_VSA3','SlogP_VSA4','SlogP_VSA6','TPSA','EState_VSA3','EState_VSA5','EState_VSA7','EState_VSA8','VSA_EState9','NHOHCount','NumAliphaticHeterocycles','NumAromaticHeterocycles','MolLogP','fr_Ar_COO','fr_C_O','fr_Imine','fr_NH1','fr_Ndealkylation2','fr_amide','fr_aryl_methyl','fr_ester','fr_ether','fr_furan','fr_imidazole','fr_methoxy','fr_piperzine','fr_pyridine','fr_sulfide','fr_thiazole','fr_urea']

    DB_COLS = ','.join(['"{}"'.format(x) for x in cols])
    # CAP_SIZE = 1976
    CAP_SIZE = 8000
    return fetchData('SELECT {} FROM {} LIMIT {}'.format(DB_COLS, DB_TABLE, CAP_SIZE))


def getTarget_a549():
    DB_TABLE = 'output.target_a549_features'

    cols = ['canonical_smiles','standard_value_log','MinAbsPartialCharge','HeavyAtomMolWt','MaxAbsPartialCharge','MinAbsEStateIndex','Chi3n','HallKierAlpha','PEOE_VSA1','PEOE_VSA10','PEOE_VSA11','PEOE_VSA12','PEOE_VSA13','PEOE_VSA14','PEOE_VSA2','PEOE_VSA3','PEOE_VSA6','PEOE_VSA8','PEOE_VSA9','SMR_VSA1','SMR_VSA10','SMR_VSA3','SMR_VSA6','SMR_VSA9','SlogP_VSA10','SlogP_VSA3','SlogP_VSA4','SlogP_VSA6','TPSA','EState_VSA3','EState_VSA5','EState_VSA7','EState_VSA8','VSA_EState9','NHOHCount','NumAliphaticHeterocycles','NumAromaticHeterocycles','MolLogP','fr_Ar_COO','fr_C_O','fr_Imine','fr_NH1','fr_Ndealkylation2','fr_amide','fr_aryl_methyl','fr_ester','fr_ether','fr_furan','fr_imidazole','fr_methoxy','fr_piperzine','fr_pyridine','fr_sulfide','fr_thiazole','fr_urea']

    DB_COLS = ','.join(['"{}"'.format(x) for x in cols])
    # CAP_SIZE = 1976
    CAP_SIZE = 16000
    return fetchData('SELECT {} FROM {} LIMIT {}'.format(DB_COLS, DB_TABLE, CAP_SIZE))