#! /usr/bin/env python

import db
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sets import Set
from rdkit import Chem
from rdkit.Chem import Descriptors

from sklearn.feature_selection import VarianceThreshold

import pylab

data = db.getTarget_206_1977()

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

df_data = {}

df_data['smiles'] = []
df_data['sval'] = []
df_reorder = ['smiles','sval']
for name, function in Descriptors.descList: 
    if name in sicho_descriptors:    
        df_data[name] = []
        df_reorder.append(name)

for i in range(len(data)):
    smiles = data[i][0]
    sval = data[i][1]

    mol = Chem.MolFromSmiles(smiles)
    for name, function in Descriptors.descList:  
        if name in sicho_descriptors:    
            df_data[name].append(function(mol))

    df_data['smiles'].append(smiles)
    df_data['sval'].append(sval)

# create dataframe, reorder values so that smiles is first, sval is second
df = pd.DataFrame(df_data)
df = df[df_reorder]
df.set_index('smiles', inplace=True)

print(df.to_csv('sicho_db.csv'))
