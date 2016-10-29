# Requires installation of protpy/propy (name inconsistent) package from 
# https://code.google.com/archive/p/protpy/
#
# Launch directly

from propy import PyPro

import db.db as db

def dbGet():
    return db.getData()

def pfamToPropy():
    proteins = dbGet();
    descArray = []
    for protein in proteins['pfam_sequence']:
        descObject = PyPro.GetProDes(protein)
        descriptors = descObject.GetCTD()
        descriptors['pfam_sequence'] = protein
        descArray.append(descriptors)
    print descArray


