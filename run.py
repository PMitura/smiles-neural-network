#! /usr/bin/env python

import sys, getopt
from rnn.rnn import run
from config import config as cc


def main(argv):

    path = 'local/config.yml'

    if len(argv) >= 1:
        path = argv[0]

    try:
        cc.loadConfig(path)
    except IOError as ioe:
        print("{0}".format(ioe))

    for experiment in cc.config['experiments']:
        # load configuration into global var and run it
        cc.loadExperiment(experiment)
        run(cc.config['run']['source'])

if __name__ == '__main__':
    main(sys.argv[1:])
