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

from sets import Set

smiles = 'Oc1ccc(cc1)C(=C(CC(F)(F)F)c2ccccc2)c3ccc(O)cc3'

writer = SDWriter('test.sdf')

mol = Chem.MolFromSmiles(smiles)
writer.write(mol)

writer.close()


