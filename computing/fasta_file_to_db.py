#! /usr/bin/env python

import sys
sys.path.insert(0, '../')
sys.path.insert(0, '../rnn')
from config import config as cc

cc.loadConfig('../local/config.yml')
cc.exp['params'] = {}
cc.exp['params']['data']={}
cc.exp['params']['rnn']={}


import db.db as db
import pandas as pd
import numpy as np

import data

from sets import Set

FASTA_FILE = '../local/data/9606.protein.sequences.v10.fa'

with open(FASTA_FILE) as f:
    fasta_content = f.readlines()

fasta_idx = []

for i in xrange(len(fasta_content)):
    if fasta_content[i][0] == '>':
        fasta_idx.append(i)

fasta_data = []

for i in xrange(len(fasta_idx)):
    ilo = fasta_idx[i]

    ihi = len(fasta_content) + 1
    if i + 1 < len(fasta_idx):
        ihi = fasta_idx[i+1]

    fasta_str = (''.join([x.rstrip() for x in fasta_content[ilo+1:ihi]])).rstrip()
    fasta_id = (fasta_content[ilo][1:]).rstrip()


    fasta_data.append([fasta_id, fasta_str])

dffasta = pd.DataFrame(fasta_data, columns=['string_id', 'fasta'])

dffasta.to_csv('string_protein.csv', index=False)

