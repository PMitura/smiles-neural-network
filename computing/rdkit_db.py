#! /usr/bin/env python

import sys
sys.path.insert(0, '../')
from config import config as cc

cc.loadConfig('../local/config.yml')

import db.db as db
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from sets import Set

sicho_descriptors = Set([
'MinAbsPartialCharge',
'HeavyAtomMolWt',
'MaxAbsPartialCharge',
'MinAbsEStateIndex',
'Chi3n',
'HallKierAlpha',
'PEOE_VSA1',
'PEOE_VSA10',
'PEOE_VSA11',
'PEOE_VSA12',
'PEOE_VSA13',
'PEOE_VSA14',
'PEOE_VSA2',
'PEOE_VSA3',
'PEOE_VSA6',
'PEOE_VSA8',
'PEOE_VSA9',
'SMR_VSA1',
'SMR_VSA10',
'SMR_VSA3',
'SMR_VSA6',
'SMR_VSA9',
'SlogP_VSA10',
'SlogP_VSA3',
'SlogP_VSA4',
'SlogP_VSA6',
'TPSA',
'EState_VSA3',
'EState_VSA5',
'EState_VSA7',
'EState_VSA8',
'VSA_EState9',
'NHOHCount',
'NumAliphaticHeterocycles',
'NumAromaticHeterocycles',
'MolLogP',
'fr_Ar_COO',
'fr_C_O',
'fr_Imine',
'fr_NH1',
'fr_Ndealkylation2',
'fr_amide',
'fr_aryl_methyl',
'fr_ester',
'fr_ether',
'fr_furan',
'fr_imidazole',
'fr_methoxy',
'fr_piperzine',
'fr_pyridine',
'fr_sulfide',
'fr_thiazole',
'fr_urea'
])

LELIMIT = 10000

def getData(con, lo):
    query = 'SELECT * FROM target_molweight LIMIT {}, {}'.format(lo,LELIMIT)

    df = pd.read_sql(
        sql = query,
        con = con)

    df_data = {}

    for name, function in Descriptors.descList:
        if name in sicho_descriptors:
            df_data[name] = []

    for smiles in df.canonical_smiles:
        mol = Chem.MolFromSmiles(smiles)
        for name, function in Descriptors.descList:
            if name in sicho_descriptors:
                df_data[name].append(function(mol))

    return pd.concat([df, pd.DataFrame(df_data)], axis=1)

def sendData(con, df):

    cursor = con.cursor()
    cols = df.columns.tolist()


    values = df.values

    for vals in values:
        for i,val in enumerate(vals):
            if pd.isnull(val):
                vals[i]=None

        query = 'INSERT INTO target_molweight_features_wide ({}) VALUES ({})'.format(
            ','.join(cols),
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
    print(len(df),LELIMIT)
    if len(df) < LELIMIT:
        break
    lo+=LELIMIT

con.close()