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
from rdkit.Chem import SDMolSupplier
from rdkit.Chem import AllChem

from sets import Set

SEND_TABLE = 'output.target_molweight_padel_3d_desc'

LIMIT = 1000

def sendData(con, df):

    cursor = con.cursor()
    cols = df.columns.tolist()

    values = df.values

    for vals in values:
        for i,val in enumerate(vals):
            if pd.isnull(val):
                vals[i]=None

        query = 'INSERT INTO {} ({}) VALUES ({})'.format(
            SEND_TABLE,
            ','.join(['"{}"'.format(x) for x in cols]),
            ','.join(['%s']*len(cols)))


        dat = []

        for d in vals:
            if type(d) is float:
                dat.append(round(d,17))
            else:
                dat.append(d)

        cursor.execute(query, tuple(dat))

    con.commit()
    cursor.close()


con = db.getCon()
suppl = SDMolSupplier('../local/data/output.target_molweight_features_wide.sdf')

leskip=172406
# leskip=5
cnt = 0

for mol in suppl:
    cnt+=1
    if cnt>=leskip:
        break

idx = 1+leskip

print 'x'

total = 0

while not suppl.atEnd():

    smiles = []

    cnt = 0
    for mol in suppl:
        smiles.append(Chem.MolToSmiles(mol))
        cnt += 1
        if cnt >= LIMIT:
            break

    df = pd.read_csv('../local/data/output.target_molweight_features_wide.3d_desc.csv',skiprows=idx,nrows=LIMIT,
        low_memory=False,header=None,
        names = ['Name','TDB1u','TDB2u','TDB3u','TDB4u','TDB5u','TDB6u','TDB7u','TDB8u','TDB9u','TDB10u','TDB1m','TDB2m','TDB3m','TDB4m','TDB5m','TDB6m','TDB7m','TDB8m','TDB9m','TDB10m','TDB1v','TDB2v','TDB3v','TDB4v','TDB5v','TDB6v','TDB7v','TDB8v','TDB9v','TDB10v','TDB1e','TDB2e','TDB3e','TDB4e','TDB5e','TDB6e','TDB7e','TDB8e','TDB9e','TDB10e','TDB1p','TDB2p','TDB3p','TDB4p','TDB5p','TDB6p','TDB7p','TDB8p','TDB9p','TDB10p','TDB1i','TDB2i','TDB3i','TDB4i','TDB5i','TDB6i','TDB7i','TDB8i','TDB9i','TDB10i','TDB1s','TDB2s','TDB3s','TDB4s','TDB5s','TDB6s','TDB7s','TDB8s','TDB9s','TDB10s','TDB1r','TDB2r','TDB3r','TDB4r','TDB5r','TDB6r','TDB7r','TDB8r','TDB9r','TDB10r','PPSA-1','PPSA-2','PPSA-3','PNSA-1','PNSA-2','PNSA-3','DPSA-1','DPSA-2','DPSA-3','FPSA-1','FPSA-2','FPSA-3','FNSA-1','FNSA-2','FNSA-3','WPSA-1','WPSA-2','WPSA-3','WNSA-1','WNSA-2','WNSA-3','RPCG','RNCG','RPCS','RNCS','THSA','TPSA','RHSA','RPSA','GRAV-1','GRAV-2','GRAV-3','GRAVH-1','GRAVH-2','GRAVH-3','GRAV-4','GRAV-5','GRAV-6','LOBMAX','LOBMIN','MOMI-X','MOMI-Y','MOMI-Z','MOMI-XY','MOMI-XZ','MOMI-YZ','MOMI-R','geomRadius','geomDiameter','geomShape','RDF10u','RDF15u','RDF20u','RDF25u','RDF30u','RDF35u','RDF40u','RDF45u','RDF50u','RDF55u','RDF60u','RDF65u','RDF70u','RDF75u','RDF80u','RDF85u','RDF90u','RDF95u','RDF100u','RDF105u','RDF110u','RDF115u','RDF120u','RDF125u','RDF130u','RDF135u','RDF140u','RDF145u','RDF150u','RDF155u','RDF10m','RDF15m','RDF20m','RDF25m','RDF30m','RDF35m','RDF40m','RDF45m','RDF50m','RDF55m','RDF60m','RDF65m','RDF70m','RDF75m','RDF80m','RDF85m','RDF90m','RDF95m','RDF100m','RDF105m','RDF110m','RDF115m','RDF120m','RDF125m','RDF130m','RDF135m','RDF140m','RDF145m','RDF150m','RDF155m','RDF10v','RDF15v','RDF20v','RDF25v','RDF30v','RDF35v','RDF40v','RDF45v','RDF50v','RDF55v','RDF60v','RDF65v','RDF70v','RDF75v','RDF80v','RDF85v','RDF90v','RDF95v','RDF100v','RDF105v','RDF110v','RDF115v','RDF120v','RDF125v','RDF130v','RDF135v','RDF140v','RDF145v','RDF150v','RDF155v','RDF10e','RDF15e','RDF20e','RDF25e','RDF30e','RDF35e','RDF40e','RDF45e','RDF50e','RDF55e','RDF60e','RDF65e','RDF70e','RDF75e','RDF80e','RDF85e','RDF90e','RDF95e','RDF100e','RDF105e','RDF110e','RDF115e','RDF120e','RDF125e','RDF130e','RDF135e','RDF140e','RDF145e','RDF150e','RDF155e','RDF10p','RDF15p','RDF20p','RDF25p','RDF30p','RDF35p','RDF40p','RDF45p','RDF50p','RDF55p','RDF60p','RDF65p','RDF70p','RDF75p','RDF80p','RDF85p','RDF90p','RDF95p','RDF100p','RDF105p','RDF110p','RDF115p','RDF120p','RDF125p','RDF130p','RDF135p','RDF140p','RDF145p','RDF150p','RDF155p','RDF10i','RDF15i','RDF20i','RDF25i','RDF30i','RDF35i','RDF40i','RDF45i','RDF50i','RDF55i','RDF60i','RDF65i','RDF70i','RDF75i','RDF80i','RDF85i','RDF90i','RDF95i','RDF100i','RDF105i','RDF110i','RDF115i','RDF120i','RDF125i','RDF130i','RDF135i','RDF140i','RDF145i','RDF150i','RDF155i','RDF10s','RDF15s','RDF20s','RDF25s','RDF30s','RDF35s','RDF40s','RDF45s','RDF50s','RDF55s','RDF60s','RDF65s','RDF70s','RDF75s','RDF80s','RDF85s','RDF90s','RDF95s','RDF100s','RDF105s','RDF110s','RDF115s','RDF120s','RDF125s','RDF130s','RDF135s','RDF140s','RDF145s','RDF150s','RDF155s','L1u','L2u','L3u','P1u','P2u','E1u','E2u','E3u','Tu','Au','Vu','Ku','Du','L1m','L2m','L3m','P1m','P2m','E1m','E2m','E3m','Tm','Am','Vm','Km','Dm','L1v','L2v','L3v','P1v','P2v','E1v','E2v','E3v','Tv','Av','Vv','Kv','Dv','L1e','L2e','L3e','P1e','P2e','E1e','E2e','E3e','Te','Ae','Ve','Ke','De','L1p','L2p','L3p','P1p','P2p','E1p','E2p','E3p','Tp','Ap','Vp','Kp','Dp','L1i','L2i','L3i','P1i','P2i','E1i','E2i','E3i','Ti','Ai','Vi','Ki','Di','L1s','L2s','L3s','P1s','P2s','E1s','E2s','E3s','Ts','As','Vs','Ks','Ds'])

    # print df.columns.tolist()

    df = df.drop('Name', 1)

    df.apply(lambda x: pd.to_numeric(x, errors='coerce'))

    df['canonical_smiles'] = pd.Series(smiles,index=df.index)

    df = df.dropna()
    sendData(con,df)
    idx += cnt

    total += len(df)
    print len(df)


    sys.stdout.flush()


con.close()

print total