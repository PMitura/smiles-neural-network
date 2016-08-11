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

        # reloads the dnn.dnn module which populates new config values to their respective vars
        import dnn.dnn
        reload(dnn.dnn)

        print(cc.cfg,cc.exp)
        dnn.dnn.run()

if __name__ == '__main__':
    main(sys.argv[1:])
