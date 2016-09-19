#! /usr/bin/env python

import db
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

from sets import Set
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import SDWriter
from sklearn.feature_selection import VarianceThreshold

data = db.getTarget_206_1977()

SDF_FILE_MAX_LINES = 1000

# read db and write .sdf files split to 1000 mols per file
'''
sdf_name_suffix = '.sdf'
sdf_name_prefix = 'sdf_206_1977'

for step in range(int((len(data)+SDF_FILE_MAX_LINES-1)/SDF_FILE_MAX_LINES)):
    lo = step*SDF_FILE_MAX_LINES
    hi = min((step+1)*SDF_FILE_MAX_LINES,len(data))

    sdf_name = sdf_name_prefix + '_' + str(step) + sdf_name_suffix

    writer = SDWriter(sdf_name)

    for i in range(lo, hi):
        smiles = data[i][0]
        mol = Chem.MolFromSmiles(smiles)
        writer.write(mol)

    writer.close()
'''

# maccs sparse to dense representation
'''
df_data = {}

df_data['canonical_smiles'] = []

for i in range(166):
    df_data['maccs'+str(i)] = []

rdkit_maccs_csv_files = ['rdkit_maccs_206_1977_0.csv', 'rdkit_maccs_206_1977_1.csv']

db_idx = 0

for fname in rdkit_maccs_csv_files:
    file = open(fname)
    lines = file.readlines()

    for line in lines:
        vals = Set([int(x) for x in line.split(',')])


        df_data['canonical_smiles'].append(data[db_idx][0])
        for i in range(166):            
            df_data['maccs'+str(i)].append(int(i in vals))

        db_idx = db_idx+1

df = pd.DataFrame(df_data)
df.set_index('canonical_smiles', inplace=True)

print(df.describe())

df = df.loc[:,df.apply(pd.Series.nunique) != 1]

print(df.describe())

# print(df.to_csv('rdkit_maccs_206_1977_dense.csv'))
'''


# cdk klekota roth sparse to dense representation
'''
df_data = {}

df_data['canonical_smiles'] = []

for i in range(4680):
    df_data['kr'+str(i)] = []

kr_csv_files = ['cdk_kr_206_1977_0.csv', 'cdk_kr_206_1977_1.csv']

db_idx = 0

for fname in kr_csv_files:
    file = open(fname)
    lines = file.readlines()
    for line in lines:

        # print(db_idx,len(data))
        df_data['canonical_smiles'].append(data[db_idx][0])

        line = line.rstrip()
        if len(line) !=0: 
            vals = Set([int(x) for x in line.split(',')])


            for i in range(4680):            
                df_data['kr'+str(i)].append(int(i in vals))
        else:
            for i in range(4680):            
                df_data['kr'+str(i)].append(0)

        db_idx = db_idx+1

df = pd.DataFrame(df_data)
df.set_index('canonical_smiles', inplace=True)

print(df.describe())

df = df.loc[:,df.apply(pd.Series.nunique) != 1]

print(df.describe())

print(df.to_csv('cdk_kr_206_1977_dense_pruned.csv'))
'''

