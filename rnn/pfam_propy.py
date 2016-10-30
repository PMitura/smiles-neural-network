# Requires installation of protpy/propy (name inconsistent) package from 
# https://code.google.com/archive/p/protpy/
#
# Launch directly

from propy import PyPro

import db.db as db
import csv

def dbGet():
    return db.getData()

def pfamToPropy():
    proteins = dbGet();

    print '  Computing propy decriptors...'

    descArray = []
    for protein in proteins['pfam_sequence']:
        descObject = PyPro.GetProDes(protein)
        descriptors = descObject.GetCTD()
        descriptors['pfam_sequence'] = protein
        descArray.append(descriptors)

    print '  ...done'
    print '  Writing CSV...'

    with open('propy.csv', 'wb') as csvfile:
        keys = []
        writer = csv.writer(csvfile, delimiter=',')
        for key in descArray[0].items():
            keys.append(key[0])
        writer.writerow(keys)
        for row in descArray:
            vals = []
            for key, value in row.items():
                vals.append(value)
            writer.writerow(vals)

    print '  ...done'

