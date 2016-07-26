import numpy as np
from sklearn import datasets, linear_model
from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
import os


def regression(trainIn, trainLabel, testIn, testLabel):
    print ("  Running baseline regression model...")


    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(trainIn, trainLabel)

    # Score test data
    scored = regr.predict(testIn)

    print scored
    print ("  ...done")
