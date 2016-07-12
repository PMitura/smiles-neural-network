#! /usr/bin/env python

import sys, getopt
from rnn.rnn import run

def main(argv):
    # parse arguments
    source = 'chembl'
    try:
        opts, args = getopt.getopt(argv, 's:')
    except getopt.GetoptError:
        print 'run.py -s [source]'
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-s':
            source = arg

    run(source)

if __name__ == '__main__':
    main(sys.argv[1:])
