#! /usr/bin/env python

import db
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Descriptors

from sklearn.feature_selection import VarianceThreshold

import pylab

data = db.getTarget_206_1977()

duplicates = {}

for datum in data:
    if datum[0] in duplicates:
        duplicates[datum[0]].append(datum[1])
    else:
        duplicates[datum[0]] = [datum[1]]

new_data = []
for smile, sval_arr in duplicates.iteritems():
    
    lemin = np.amin(sval_arr)
    lemax = np.amax(sval_arr)

    if len(sval_arr) == 1:
        new_data.append([smile,sval_arr[0]])
    elif lemin != 0 and lemax != 0:
        if not (len(sval_arr) < 20 and int(math.log(lemin, 10)) != int(math.log(lemax, 10))):
            new_data.append([smile,np.median(sval_arr)])
        
data = new_data

df_data = {}

df_data['smiles'] = []
df_data['sval'] = []
df_reorder = ['smiles','sval']
for name, function in Descriptors.descList:     
    df_data[name] = []
    df_reorder.append(name)

for i in range(len(data)):
    smiles = data[i][0]
    sval = data[i][1]

    mol = Chem.MolFromSmiles(smiles)
    for name, function in Descriptors.descList:     
        df_data[name].append(function(mol))

    df_data['smiles'].append(smiles)
    df_data['sval'].append(sval)

# create dataframe, reorder values so that smiles is first, sval is second
df = pd.DataFrame(df_data)
df = df[df_reorder]
df.set_index('smiles', inplace=True)

# we convert the IC50 values to pIC50 
df.sval = df.sval.apply(lambda x : -1.0 * np.log10(x / 1.0e9))

# drop infinite values
df = df.drop(df[df.sval == np.inf].index)

def get_removed_feats(df, model):
    return df.columns.values[1:][~model.get_support()]

def update_df(df, removed_descriptors, inplace=True):
    if inplace:
        df.drop(removed_descriptors, 1, inplace=True)
        # print(df.shape)
        return df
    else:
        new_df = df.drop(removed_descriptors, 1, inplace=False)
        # print(new_df.shape)
        return new_df

# find the names of the columns with zero variance
var_sel = VarianceThreshold()
var_sel.fit(df.iloc[:,1:])
removed_descriptors = get_removed_feats(df, var_sel)

# update the data frame
update_df(df, removed_descriptors)

# correlation filter
def find_correlated(data):
    correlation_matrix = data.iloc[:,1:].corr(method='spearman')
    removed_descs = set()
    all_descs = correlation_matrix.columns.values
    for label in all_descs:
        if label not in removed_descs:
            correlations_abs = correlation_matrix[label].abs()
            mask = (correlations_abs > 0.7).values
            to_remove = set(all_descs[mask])
            to_remove.remove(label)
            removed_descs.update(to_remove)
        
    return removed_descs

update_df(df, find_correlated(df))

# regression tests
from sklearn.feature_selection import SelectPercentile,SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import StandardScaler


# keep only the descriptors that show significant 
# correlation with the target variable (pIC50)
regre_sele = SelectPercentile(f_regression, percentile=50)
regre_sele.fit(df.iloc[:,1:], df.sval)
removed_descriptors = get_removed_feats(df, regre_sele)

# update the data frame
update_df(df, removed_descriptors)


# print selected features
print(df.columns.tolist())

from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import train_test_split

from sklearn.svm import LinearSVR
from sklearn.feature_selection import RFECV
from sklearn.cross_validation import StratifiedKFold


scaler = StandardScaler(copy=False)
scaler.fit(df.iloc[:,1:])
scaled_features = pd.DataFrame(scaler.transform(df.iloc[:,1:]), columns=df.iloc[:,1:].columns)


from sklearn.cross_validation import train_test_split


# sicho optimized
estimator = LinearSVR()

rfecv = RFECV(estimator, step=1)
rfecv= rfecv.fit(scaled_features, df.sval)

print(rfecv.ranking_)

print("Optimal number of features : %d" % rfecv.n_features_)
# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()




'''
from sklearn.manifold import TSNE

model = TSNE(n_components=2)
TSNEdata = model.fit_transform(df.iloc[:,1:])

TSNEdf = pd.DataFrame(TSNEdata, columns=('x','y'))

TSNEdf['c'] = pd.Series(df.sval.values,index=TSNEdf.index)

plot = TSNEdf.plot.scatter(x = 'x', y = 'y', c = 'c', cmap = 'plasma')

plt.show()
'''
