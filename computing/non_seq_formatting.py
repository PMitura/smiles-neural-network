#! /usr/bin/env python

import sys
sys.path.insert(0, '../')
sys.path.insert(0, '../rnn')
from config import config as cc

cc.loadConfig('../local/config.yml')
cc.exp['params'] = {}
cc.exp['params']['data']={}
cc.exp['params']['rnn']={}




import db.db as db
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import MACCSkeys

import data

from sets import Set

LELIMIT = 10000

SICHO_RIPTORS = Set(['MinAbsPartialCharge','HeavyAtomMolWt','MaxAbsPartialCharge','MinAbsEStateIndex','Chi3n','HallKierAlpha','PEOE_VSA1','PEOE_VSA10','PEOE_VSA11','PEOE_VSA12','PEOE_VSA13','PEOE_VSA14','PEOE_VSA2','PEOE_VSA3','PEOE_VSA6','PEOE_VSA8','PEOE_VSA9','SMR_VSA1','SMR_VSA10','SMR_VSA3','SMR_VSA6','SMR_VSA9','SlogP_VSA10','SlogP_VSA3','SlogP_VSA4','SlogP_VSA6','TPSA','EState_VSA3','EState_VSA5','EState_VSA7','EState_VSA8','VSA_EState9','NHOHCount','NumAliphaticHeterocycles','NumAromaticHeterocycles','MolLogP','fr_Ar_COO','fr_C_O','fr_Imine','fr_NH1','fr_Ndealkylation2','fr_amide','fr_aryl_methyl','fr_ester','fr_ether','fr_furan','fr_imidazole','fr_methoxy','fr_piperzine','fr_pyridine','fr_sulfide','fr_thiazole','fr_urea'])

DOWNLOAD_TABLE = 'output.target_206_1977_features_computed'
DOWNLOAD_COLS = ['canonical_smiles','standard_value_log']
WHERE = 'length(canonical_smiles) <= 80'
LIMIT = None
# DOWNLOAD_TABLE = 'output.target_geminin_deduplicated'
# DOWNLOAD_COLS = ['molregno','canonical_smiles','is_testing','standard_value_min','standard_value_max','standard_value_count','standard_value_std','standard_value_relative_std','standard_value_median']
# SEND_TABLE = 'output.target_geminin_deduplicated_rdkit_maccs'

def getData(con):

    query = 'SELECT {} FROM {}'.format(
        ','.join(['"{}"'.format(x) for x in DOWNLOAD_COLS]),
        DOWNLOAD_TABLE)

    if WHERE:
        query += ' WHERE {}'.format(WHERE)

    if LIMIT:
        query += ' LIMIT {}'.format(LIMIT)

    print(query)

    df = pd.read_sql(
        sql = query,
        con = con)

    return df

def formatNonSequential(smilesDf):
    smilesMaxLen = 80
    nonSeq = np.zeros((len(smilesDf), smilesMaxLen, data.SMILES_ALPHABET_LEN))

    # translate to one hot for smiles
    for i,smiles in enumerate(smilesDf):
        for j in range(smilesMaxLen):

            transChar = data.SMILES_ALPHABET_LOOKUP_TABLE[data.SMILES_ALPHABET_UNKNOWN]
            if j < len(smiles) and smiles[j] in data.SMILES_ALPHABET_LOOKUP_TABLE:
                transChar = data.SMILES_ALPHABET_LOOKUP_TABLE[smiles[j]]
            nonSeq[i][j][transChar] = 1

    # print(nonSeq.tolist()[0][0])
    nonSeq = nonSeq.reshape(len(smilesDf),smilesMaxLen*data.SMILES_ALPHABET_LEN)

    return nonSeq

def formatBackOff(smilesDf):
    smilesMaxLen = 80
    nonSeq = np.zeros((len(smilesDf),data.SMILES_ALPHABET_LEN))

    for i,smiles in enumerate(smilesDf):
        for j in range(smilesMaxLen):
            transChar = data.SMILES_ALPHABET_LOOKUP_TABLE[data.SMILES_ALPHABET_UNKNOWN]
            if j < len(smiles) and smiles[j] in data.SMILES_ALPHABET_LOOKUP_TABLE:
                transChar = data.SMILES_ALPHABET_LOOKUP_TABLE[smiles[j]]
            nonSeq[i][transChar]+=1

    return nonSeq

con = db.getCon()
df = getData(con)
con.close()


# nonSeq = formatNonSequential(df.canonical_smiles)
nonSeq = formatBackOff(df.canonical_smiles)

dfNonSeq = pd.concat((pd.DataFrame(nonSeq), df.standard_value_log), axis = 1)

# print(df.canonical_smiles)
# print(dfNonSeq.values)

dfNonSeq.to_csv('target_206_1977_nonseq_smiles_backoff.csv')
