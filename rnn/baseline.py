import numpy as np
from sklearn import datasets, linear_model
from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
import os


# Get format suitable for linear model
def processRawData(rawData):
    print '    Processing data using RDKit...'

    
    # Create factory for generating features 
    fdefName = os.path.join(RDConfig.RDDataDir,'BaseFeatures.fdef')
    factory = ChemicalFeatures.BuildFeatureFactory(fdefName)

    # Get features of each row
    features = []
    for row in rawData:
        mol = Chem.MolFromSmiles(row[0])

        # Some features cannot be computed because of atom valence exceptions,
        # whatever that means.
        #
        # Let's count these, then ignore them.
        feat = factory.GetFeaturesForMol(mol) # This causes mutex crashes :(
        features.append(feat)

    print '      Exception count: {} of {}'.format(
            len(rawData) - len(features), len(rawData))
    print '    ...done'
    return features


def regression(rawData):
    print ("  Running baseline regression model...")

    processRawData(rawData)

    """
    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(trainIn, trainLabel)

    # Score test data
    scored = regr.predict(testIn)

    print scored
    """
    print ("  ...done")
