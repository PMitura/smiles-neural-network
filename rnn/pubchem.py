import pubchempy as pcp
import csv, json
from pprint import pprint

DOWNLOAD = True
CSVFILE = 'data/pubchem'
LABELS='CanonicalSMILES,MolecularWeight,\
XLogP,ExactMass,MonoisotopicMass,TPSA,\
Complexity,Charge,HBondDonorCount,HBondAcceptorCount,RotatableBondCount,\
HeavyAtomCount,IsotopeAtomCount,AtomStereoCount,DefinedAtomStereoCount,\
UndefinedAtomStereoCount,BondStereoCount,DefinedBondStereoCount,\
UndefinedBondStereoCount,CovalentUnitCount,Volume3D,XStericQuadrupole3D,\
YStericQuadrupole3D,ZStericQuadrupole3D,FeatureCount3D,\
FeatureAcceptorCount3D,FeatureDonorCount3D,FeatureAnionCount3D,\
FeatureCationCount3D,FeatureRingCount3D,FeatureHydrophobeCount3D,\
ConformerModelRMSD3D,EffectiveRotorCount3D,ConformerCount3D'
CID_LOW = 50000000
CID_HIGH = 51000000
STEP = 100


def downloadData():
    i = CID_LOW
    ctr = 1
    while i <= CID_HIGH:
        pcp.download('CSV', '{}-{}.csv'.format(CSVFILE, ctr), range(i, i + 500), 
                operation = 'property/{}'.format(LABELS))
        print 'step {}/{}'.format(ctr, int((CID_HIGH - CID_LOW) / STEP))
        i += 500
        ctr += 1


def readCSV():
    with open(CSVFILE, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter = ',')
        for row in reader:
            print ', '.join(row)


def getData():
    if DOWNLOAD:
        downloadData()

    return readCSV()


# Dummy function for testing capabilities of pubchempy
def play():
    """
    aspirin = pcp.get_compounds('Aspirin', 'name')
    for compound in aspirin:
        print "{} has aids {}".format(compound.isomeric_smiles,
                compound.aids)
    """

    assayCID = pcp.request(1, domain = 'assay', namespace = 'aid', assay_type =
            'all', operation = 'cids')
    jsonCID = json.loads(assayCID.read())
    for line in jsonCID['InformationList']['Information']:
        for cid in line['CID']:
            data = pcp.request(1, domain = 'assay', namespace = 'aid', assay_type =
                    'all', operation = 'record', cid = str(cid))
            jsonData = json.loads(data.read())
            for contLine in jsonData['PC_AssayContainer']:
                for dataLine in contLine['data']:
                    for labelLine in dataLine['data']:
                        if labelLine['tid'] == 1:
                            print labelLine['value']['fval']

    """
    ax = pcp.request(1, domain = 'assay', namespace = 'aid', assay_type =
            'all', operation = 'record', sid = '495487')
    pprint(vars(ax))
    print ax.read()
    """
    """
    print "Downloading CSV..."
    pcp.download('CSV', 'test.csv', range(1, 10000), 
            operation = 'property/CanonicalSMILES',
            overwrite = True)
    print "Done"
    """


if __name__ == '__main__':
    play()
