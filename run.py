#! /usr/bin/env python

import sys, getopt
from config import config as cc
from keras.optimizers import Adam, Adadelta, Adagrad, Nadam, Adamax

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


        # limitGrid = [10,50,100,500,1000,5000,10000,50000,100000]

        # for lg in limitGrid:
            # cc.exp['fetch']['limit'] = lg
            # cc.exp['params']['rnn']['comment'] = '[LEARNING_CURVE][LIMIT_{}] limiting data to {} rows'.format(lg,lg)

            # reloads the rnn.rnn module which populates new config values to their respective vars
        import rnn.rnn
        reload(rnn.rnn)

        print(cc.cfg,cc.exp)
        rnn.rnn.run()

if __name__ == '__main__':
    main(sys.argv[1:])
