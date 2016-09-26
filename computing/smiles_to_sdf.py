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
from rdkit.Chem import SDWriter
from rdkit.Chem import AllChem

from sets import Set

cc.exp['fetch'] = {
    'table': 'output.target_molweight_features_wide',
    'cols': ['canonical_smiles','is_testing_all'],
    'order': 'canonical_smiles',
    'index_col': None,
    'limit': None,
    'input_mode': 'smiles',
    'where':  'is_testing_all IS NOT NULL and length(canonical_smiles) <= 120',
}

dbDf = db.getData()

writer = SDWriter('{}.sdf'.format(cc.exp['fetch']['table']))

outputRows = []

for i,row in dbDf.iterrows():
    print('{}/{}'.format(i+1,len(dbDf)))

    print row.values

    try:
        mol = Chem.AddHs(Chem.MolFromSmiles(row['canonical_smiles']))

        AllChem.EmbedMolecule(mol)
        AllChem.MMFFOptimizeMolecule(mol)

        writer.write(mol)
        outputRows.append(row.values)
    except:
        pass

outDf = pd.DataFrame(outputRows, columns=dbDf.columns)

outDf.to_csv('{}.csv'.format(cc.exp['fetch']['table']))

writer.close()
