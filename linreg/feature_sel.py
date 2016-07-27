#! /usr/bin/env python

import sys
sys.path.append('/usr/lib/python2.7/dist-packages')

import db
import numpy as np
from sklearn import linear_model
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
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

for c in compounds:
    try:

        m = Chem.MolFromSmiles(c[0])

        feats = []

        # Compute all available descriptors in Rdkit
        for desc_name, function in Descriptors.descList:
            feats.append(function(m))

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

selected_features = []

for i in range(len(selector.get_support())):
    if selector.get_support()[i]:
    	selected_features.append(Descriptors.descList[i][0])

print('...done')

print('Selected features', selected_features)

print('Running regression...')

k_compounds_X_train = []
k_compounds_X_test = []

for cc in compounds_X_train:
    new_cc = []
    for i in range(len(selector.get_support())):
        if selector.get_support()[i]:
            new_cc.append(cc[i])

    k_compounds_X_train.append(tuple(new_cc))


for cc in compounds_X_test:
    new_cc = []
    for i in range(len(selector.get_support())):
        if selector.get_support()[i]:
            new_cc.append(cc[i])

    k_compounds_X_test.append(tuple(new_cc))

compounds_X_train = k_compounds_X_train
compounds_X_test = k_compounds_X_test


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

# '''
