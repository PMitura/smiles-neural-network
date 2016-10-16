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
from rdkit.Chem import SDMolSupplier
from rdkit.Chem import AllChem

from sets import Set

SEND_TABLE = 'output.target_molweight_padel_3d_desc'

LIMIT = 100

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


        dat = []

        for d in vals:
            if type(d) is float:
                dat.append(round(d,30))
            else:
                dat.append(d)

        cursor.execute(query, tuple(dat))

    con.commit()
    cursor.close()


con = db.getCon()
suppl = SDMolSupplier('../local/data/output.target_molweight_features_wide.sdf')

idx = 0

total = 0

while not suppl.atEnd():

    smiles = []

    cnt = 0
    for mol in suppl:
        smiles.append(Chem.MolToSmiles(mol))
        cnt += 1
        if cnt >= LIMIT:
            break

    df = pd.read_csv('../local/data/output.target_molweight_features_wide.3d_desc.csv',skiprows=idx,nrows=LIMIT,low_memory=False).drop('Name', 1)

    df.apply(lambda x: pd.to_numeric(x, errors='coerce'))

    df['canonical_smiles'] = pd.Series(smiles,index=df.index)

    df = df.dropna()
    sendData(con,df)
    idx += cnt

    total += len(df)
    print len(df)


con.close()

print total