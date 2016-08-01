#! /usr/bin/env python

import sys, getopt
from rnn.rnn import run
from config import config as cc


def main(argv):

    configPath = 'local/config.yml'

    if len(argv) >= 1:
        configPath = argv[0]

    try:
        cc.load(configPath)
    except IOError as ioe:
        print("{0}".format(ioe))

    run(cc.config['run']['source'])

if __name__ == '__main__':
    main(sys.argv[1:])
