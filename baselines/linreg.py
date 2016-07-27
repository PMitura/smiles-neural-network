#! /usr/bin/env python

import db
import numpy as np
from sklearn import linear_model

# Downloads rows from the table target_protein_bcd_best5 
#	uses db.getTargetProteinBcdBest5()
# Performs linear regression on this dataset
# ex.row: canonical_smiles,Chi3v,Chi4v,SlogP_VSA10,NumAromaticCarbocycles,fr_benzene,standard_value_log_median_centered,is_testing

data = db.getTargetProteinBcdBest5()

data_X_train = []
data_y_train = []
data_X_test = []
data_y_test = []

for datum in data:
    if datum[7]:
        data_X_train.append(datum[1:6])
        data_y_train.append(datum[6])
    else:
        data_X_test.append(datum[1:6])
        data_y_test.append(datum[6])

regr = linear_model.LinearRegression()
regr.fit(data_X_train, data_y_train)

pred = regr.predict(data_X_test)

res_mat = np.zeros((2, len(pred)))
for i in range(len(pred)):
    res_mat[0][i] = pred[i]
    res_mat[1][i] = data_y_test[i]

corr = np.corrcoef(res_mat)
R2 = corr[0][1] * corr[0][1]

print('Coefficients: \n', regr.coef_)
print('Intercept: \n', regr.intercept_)
print('R2: \n', R2)
