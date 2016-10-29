#! /usr/bin/env python

# Requires installation of protpy/propy (name inconsistent) package from 
# https://code.google.com/archive/p/protpy/
#
# Launch directly

from propy import PyPro

def pfamToPropy():
    print 'propy descriptor getter'
    protein = 'ADFTIFQDFYAYRSGIYVHATGKQLGGHAIKILGWGTEDNVDYWVGQTVIMDL'
    descObject = PyPro.GetProDes(protein)
    descriptors = descObject.GetCTD()
    descriptors['pfam_sequence'] = protein
    print descriptors

if __name__ == '__main__':
    pfamToPropy()
