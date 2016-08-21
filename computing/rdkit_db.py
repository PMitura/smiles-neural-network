#! /usr/bin/env python

import sys
sys.path.insert(0, '../')
from config import config as cc

cc.loadConfig('../local/config.yml')

import db.db as db
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import MACCSkeys

from sets import Set

LELIMIT = 10000

SICHO_RIPTORS = Set(['MinAbsPartialCharge','HeavyAtomMolWt','MaxAbsPartialCharge','MinAbsEStateIndex','Chi3n','HallKierAlpha','PEOE_VSA1','PEOE_VSA10','PEOE_VSA11','PEOE_VSA12','PEOE_VSA13','PEOE_VSA14','PEOE_VSA2','PEOE_VSA3','PEOE_VSA6','PEOE_VSA8','PEOE_VSA9','SMR_VSA1','SMR_VSA10','SMR_VSA3','SMR_VSA6','SMR_VSA9','SlogP_VSA10','SlogP_VSA3','SlogP_VSA4','SlogP_VSA6','TPSA','EState_VSA3','EState_VSA5','EState_VSA7','EState_VSA8','VSA_EState9','NHOHCount','NumAliphaticHeterocycles','NumAromaticHeterocycles','MolLogP','fr_Ar_COO','fr_C_O','fr_Imine','fr_NH1','fr_Ndealkylation2','fr_amide','fr_aryl_methyl','fr_ester','fr_ether','fr_furan','fr_imidazole','fr_methoxy','fr_piperzine','fr_pyridine','fr_sulfide','fr_thiazole','fr_urea'])

DOWNLOAD_TABLE = 'output.target_molweight'
DOWNLOAD_COLS = ['molregno','canonical_smiles', 'molweight','alogp','hba','hbd','psa','rtb','is_testing','is_testing_without_206_1977_testing']

SEND_TABLE = 'output.target_molweight_rdkit_maccs'

def computeDescriptors(smilesDf):
    dfData = {}

    for name, function in Descriptors.descList:
        if name in SICHO_RIPTORS:
            dfData[name] = []

    for smiles in smilesDf.canonical_smiles:
        mol = Chem.MolFromSmiles(smiles)
        for name, function in Descriptors.descList:
            if name in SICHO_RIPTORS:
                try:
                    dfData[name].append(function(mol))
                except:
                    dfData[name].append(None)

    return pd.DataFrame(dfData)

def computeMACCS(smilesDf):
    dfData = {}

    maccsKeyNames = []

    for i in range(0,167):
        maccsKeyNames.append('maccs{i:03d}'.format(i=i))

    for i in range(1,167):
        dfData[maccsKeyNames[i]] = []

    for smiles in smilesDf.canonical_smiles:
        mol = Chem.MolFromSmiles(smiles)

        try:
            maccsBitVector = MACCSkeys.GenMACCSKeys(mol)
            for i in range(1,167):
                dfData[maccsKeyNames[i]].append(maccsBitVector[i])
        except:
            for i in range(1,167):
                dfData[maccsKeyNames[i]].append(None)


    return pd.DataFrame(dfData)


def getData(con, lo):
    query = 'SELECT {} FROM {} LIMIT {} OFFSET {}'.format(
        ','.join(['"{}"'.format(x) for x in DOWNLOAD_COLS]),
        DOWNLOAD_TABLE,
        LELIMIT,
        lo)

    print(query)

    df = pd.read_sql(
        sql = query,
        con = con)

    # fingerDf = computeDescriptors(df)
    fingerDf = computeMACCS(df)

    mergedDf = pd.concat([df, fingerDf], axis=1)

    return mergedDf

def sendData(con, df):

    cursor = con.cursor()
    cols = df.columns.tolist()

    values = df.values

    for vals in values:
        for i,val in enumerate(vals):
            if pd.isnull(val):
                vals[i]=None

        query = 'INSERT INTO {} ({}) VALUES ({})'.format(
            SEND_TABLE,
            ','.join(['"{}"'.format(x) for x in cols]),
            ','.join(['%s']*len(cols)))

        cursor.execute(query, tuple(vals))

    con.commit()
    cursor.close()


con = db.getCon()

lo = 0
while True:
    print('Getting: {}'.format(lo))
    df = getData(con,lo)

    print('Sending...')
    sendData(con,df)
    print('Done')

    break
    print(len(df),LELIMIT)
    if len(df) < LELIMIT:
        break
    lo+=LELIMIT

con.close()