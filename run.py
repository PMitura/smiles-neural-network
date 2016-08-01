#! /usr/bin/env python

import sys, getopt
import rnn.rnn

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

    lrValues = [0.1, 0.05, 0.03, 0.01, 0.0075, 0.005, 0.003, 0.001]
    for lr in lrValues:
        rnn.rnn.LEARNING_RATE = lr
        rnn.rnn.run(source)

if __name__ == '__main__':
    main(sys.argv[1:])
