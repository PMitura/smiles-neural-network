#! /usr/bin/env python

import sys, getopt
from config import config as cc

def main(argv):
    path = 'local/config.yml'

    if len(argv) >= 1:
        path = argv[0]

    try:
        cc.loadConfig(path)
    except IOError as ioe:
        print("{0}".format(ioe))

    for experiment in cc.cfg['experiments']:
        # load configuration into global var and run it
        cc.loadExperiment(experiment)

        # reloads the rnn.rnn module which populates new config values to their respective vars
        import rnn.rnn
        reload(rnn.rnn)

        print(cc.cfg,cc.exp)

        # Grid search hacking HERE!
        #for sd in range(12346, 12355):
        #    rnn.rnn.SEED = sd
        rnn.rnn.run()


if __name__ == '__main__':
    main(sys.argv[1:])
