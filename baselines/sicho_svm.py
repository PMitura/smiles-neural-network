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

from config import config as cc


cc.loadConfig('../local/config.yml')

# data = db.getTarget_206_1977()
data = db.getTarget_geminin()


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

print('Computing riptors:')

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

    try:
    	mol = Chem.MolFromSmiles(smiles)
    	for name, function in Descriptors.descList:
    		df_data[name].append(function(mol))

    	df_data['smiles'].append(smiles)
    	df_data['sval'].append(sval)
    except:
        pass

# create dataframe, reorder values so that smiles is first, sval is second
df = pd.DataFrame(df_data)
df = df[df_reorder]
df.set_index('smiles', inplace=True)

# we convert the IC50 values to pIC50
df.sval = df.sval.apply(lambda x : -1.0 * np.log10(x / 1.0e9))

# drop infinite values
df = df.drop(df[df.sval == np.inf].index)

print('Feature selection:')

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
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_regression

# keep only the descriptors that show significant
# correlation with the target variable (pIC50)
regre_sele = SelectPercentile(f_regression, percentile=50)
regre_sele.fit(df.iloc[:,1:], df.sval)
removed_descriptors = get_removed_feats(df, regre_sele)

# update the data frame
update_df(df, removed_descriptors)

# print selected features
# print(df.columns.tolist())


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler(copy=False)
scaler.fit(df.iloc[:,1:])

features_mean = df.iloc[:,1:].mean(axis=0)
features_std = df.iloc[:,1:].std(axis=0)

# print(features_mean.values)

# scaled_features = pd.DataFrame(((df.iloc[:,1:]-features_mean)/features_std).values, columns=df.iloc[:,1:].columns)
# print(scaled_features)

scaled_features = pd.DataFrame(scaler.transform(df.iloc[:,1:]), columns=df.iloc[:,1:].columns)

# Next, we create the datasets for cross-validation and testing:

from sklearn.cross_validation import train_test_split

features_train, features_test, sval_train, sval_test = train_test_split(
    scaled_features
    , df.sval
    , test_size=0.4
    , random_state=42
)

# and build the model:
print('Building model:')


from sklearn.svm import SVR
from sklearn.grid_search import GridSearchCV

param_grid = [
#  {'C': [1, 10, 100, 1000], 'epsilon': [0.0, 0.1, 0.2, 0.3, 0.4], 'kernel': ['linear']},
#  {'C': [1, 10, 100, 1000], 'epsilon': [0.0, 0.1, 0.2, 0.3, 0.4], 'kernel': ['poly'], 'degree' : [2, 3, 4, 5]},
  {'C': [1, 10, 100, 1000], 'epsilon': [0.0, 0.1, 0.2, 0.3, 0.4], 'gamma': [0.01, 0.001, 0.0001], 'kernel': ['rbf']},
 ]

model = GridSearchCV(SVR(), param_grid, n_jobs=2, cv=5)
model.fit(features_train, sval_train)
model = model.best_estimator_

print('Model params:')
print(model.get_params())
print()

# cross validation results

from sklearn.cross_validation import cross_val_score

scores = cross_val_score(model, features_train, sval_train, cv=5)
scores_mse = cross_val_score(model, features_train, sval_train, cv=5, scoring='mean_squared_error')


print('Cross validation:')

print("Mean R^2: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print("Mean R: %0.2f (+/- %0.2f)" % (np.sqrt(scores).mean(), np.sqrt(scores).std() * 2))
print("Mean MSE: %0.2f (+/- %0.2f)" % (abs(scores_mse.mean()), scores_mse.std() * 2))
print()



# test validation results

from sklearn.metrics import mean_squared_error

print('Test validation:')

print("R^2: %0.2f" % model.score(features_test, sval_test))
print("R: %0.2f" % np.sqrt(model.score(features_test, sval_test)))
print("MSE: %0.2f" %  mean_squared_error(model.predict(features_test), sval_test))
print()

# error plot


# (sval_test - model.predict(features_test)).abs().hist(bins=30).plot()
# plt.show()

# PCA plot

print("Computing decomposition:")

from sklearn import decomposition

'''
n_components = 5

pca = decomposition.PCA(n_components=n_components)
pca.fit(df.iloc[:,1:])
pca_result = pca.transform(df.iloc[:,1:])

from mpl_toolkits.mplot3d import Axes3D
from itertools import combinations

plt.rcParams["figure.figsize"] = [15, 15]
fig = plt.figure()

ax = fig.add_subplot(1,1,1,projection='3d')

PCAcombo = [3,1,0]

ax.scatter(
    pca_result[:,PCAcombo[0]]
    , pca_result[:,PCAcombo[1]]
    , pca_result[:,PCAcombo[2]]
    , c=df.sval
    , cmap='YlOrRd'
)
ax.view_init(elev=30, azim=45)
ax.set_xlabel('PC%s' % (PCAcombo[0] + 1))
ax.set_ylabel('PC%s' % (PCAcombo[1] + 1))
ax.set_zlabel('PC%s' % (PCAcombo[2] + 1))

plt.show()
'''
'''
combos = list(combinations(range(n_components), 3))

plt.rcParams["figure.figsize"] = [15, 30]
fig = plt.figure(len(combos) / 2)

for idx, combo in enumerate(combos):
    ax = fig.add_subplot(len(combos) / 2, 2, idx + 1, projection='3d')
    ax.scatter(
        pca_result[:,combo[0]]
        , pca_result[:,combo[1]]
        , pca_result[:,combo[2]]
        , c=df.sval
        , s=20
        , cmap='YlOrRd' # red are the compounds with higher values of pIC50
    )
    ax.view_init(elev=30, azim=45)
    ax.set_xlabel('PC%s' % (combo[0] + 1))
    ax.set_ylabel('PC%s' % (combo[1] + 1))
    ax.set_zlabel('PC%s' % (combo[2] + 1))

plt.show()

'''

from sklearn.manifold import TSNE

model = TSNE(n_components=2)
TSNEdata = model.fit_transform(df.iloc[:2000,1:])

TSNEdf = pd.DataFrame(TSNEdata, columns=('x','y'))

TSNEdf['c'] = pd.Series(df.sval.values,index=TSNEdf.index)

plot = TSNEdf.plot.scatter(x = 'x', y = 'y', c = 'c', cmap = 'plasma')

plt.show()
