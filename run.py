#! /usr/bin/env python

import sys, getopt
from config import config as cc
from keras.optimizers import Adam, RMSprop, Adadelta, Adagrad

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

        # optimizer grid search
        optimizers = [Adam(lr = RP['learning_rate']), Adadelta(), Adagrad()]
        for o in optimizers:
            rnn.rnn.OPTIMIZER = o
            rnn.rnn.run()
        # rnn.rnn.run()


if __name__ == '__main__':
    main(sys.argv[1:])
