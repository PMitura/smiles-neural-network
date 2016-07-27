#! /usr/bin/env python

import sys
sys.path.append('/usr/lib/python2.7/dist-packages')

import db
import numpy as np
from sklearn import linear_model
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.metrics import r2_score
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
import os
from sets import Set

# Downloads rows from the table target_protein_big_cleaned_deduplicated
#   uses db.getTargetProteinBcdRdkitFeatureSelection()
# Computes all available chem descriptors from rdkit for each row from SMILEs
# Performs univariate feature selection (SelectKBest, f_regression), selects 5 best features
# Performs linear regression with selected features

compounds = db.getTargetProteinBcdRdkitFeatureSelection()

compounds_X_test = []
compounds_y_test = []

compounds_X_train = []
compounds_y_train = []

print('Computing descriptors...')

def descriptorsFromSmiles(smiles):
    m = Chem.MolFromSmiles(smiles)
    descriptors = []
    # Compute all available descriptors in Rdkit
    for desc_name, function in Descriptors.descList:
        descriptors.append(function(m))
    return descriptors

def filterFeatures(feats, support):
    new_feats = []
    for i in range(len(support)):
        if support[i]:
            new_feats.append(feats[i])
    return new_feats

def createSupportForDescriptors(desc_names):
    support = []
    desc_set = Set(desc_names)
    for i in range(len(Descriptors.descList)):
        support.append(Descriptors.descList[i][0] in desc_set)

    return support

for c in compounds:
    try:

        # get descriptors for compound
        feats = descriptorsFromSmiles(c[0])

        if c[2]==1:
            compounds_X_test.append(tuple(feats))
            compounds_y_test.append(float(c[1]))
        else:
            compounds_X_train.append(tuple(feats))
            compounds_y_train.append(float(c[1]))

    except:
        pass

print('...done')

print('Selecting KBest...')

selector = SelectKBest(f_regression,k=5)
selector.fit(compounds_X_train, compounds_y_train)

# change this to select different features
# used_support = selector.get_support()

# best 5 from limit 1887
used_support = createSupportForDescriptors(['SlogP_VSA10', 'SlogP_VSA4', 'fr_aniline', 'fr_phenol', 'fr_pyridine'])

# best 5 from limit 1000
# used_support = createSupportForDescriptors(['Chi3v', 'Chi4v', 'SlogP_VSA10', 'NumAromaticCarbocycles', 'fr_benzene'])

k_compounds_X_train = []
k_compounds_X_test = []

for cc in compounds_X_train:
    k_compounds_X_train.append(tuple(filterFeatures(cc, used_support)))

for cc in compounds_X_test:
    k_compounds_X_test.append(tuple(filterFeatures(cc, used_support)))

compounds_X_train = k_compounds_X_train
compounds_X_test = k_compounds_X_test


selected_features = []
for i in range(len(used_support)):
    if used_support[i]:
        selected_features.append(Descriptors.descList[i][0])

print('Selected features', selected_features)

print('...done')

print('Running regression...')

regr = linear_model.LinearRegression()
regr.fit(compounds_X_train, compounds_y_train)

prediction = regr.predict(compounds_X_test)

res_matrix = np.zeros((2, len(prediction)))
for i in range(len(prediction)):
    res_matrix[0][i] = prediction[i]
    res_matrix[1][i] = compounds_y_test[i]

corr = np.corrcoef(res_matrix)
R2 = corr[0][1] * corr[0][1]

# print('Coefficients: \n', regr.coef_)
# print('Intercept: \n', regr.intercept_)
print('R2: \n', R2)
print('...done')

print('Reg test...')

test_compound = 'COC(=O)CCCC=C(/c1cc(C)c2OC(=O)N(C)c2c1)c3cc(C)c4ON(C)C(=O)c4c3'
test_truth = [-4.691965102767360000000000000000]

test_features = descriptorsFromSmiles(test_compound)
test_features = filterFeatures(test_features, used_support)

test_feats = []
test_feats.append(tuple(test_features))
test_pred = regr.predict(test_feats)

print('Truth:', test_truth[0])
print('Pred:', test_pred[0])

print('...done')

print('Test SlogP_VSA10...')

test_mol = Chem.MolFromSmiles(test_compound)
print('SlogP_VSA10: ', Descriptors.SlogP_VSA10(test_mol))

print('...done')
