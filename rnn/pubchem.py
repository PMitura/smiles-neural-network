import pubchempy as pcp

def getData():
    # dummy, will not work
    return []

# Dummy function for testing capabilities of pubchempy
def play():
    aspirin = pcp.get_compounds('Aspirin', 'name')
    for compound in aspirin:
        print "{} has weight {}".format(compound.isomeric_smiles,
                compound.molecular_weight)
    print "Downloading CSV..."
    pcp.download('CSV', 'test.csv', range(1, 10000), 
            operation = 'property/CanonicalSMILES,MolecularWeight',
            overwrite = True)
    print "Done"


if __name__ == '__main__':
    play()
