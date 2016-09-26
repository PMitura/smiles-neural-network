#! /usr/bin/env python

import matplotlib
matplotlib.use('PDF')

import sys, getopt
from config import config as cc
from keras.optimizers import Adam, Adadelta, Adagrad, Nadam, Adamax
import theano
from multiprocessing import Process


class Unbuffered(object):
   def __init__(self, stream):
       self.stream = stream
   def write(self, data):
       self.stream.write(data)
       self.stream.flush()
   def __getattr__(self, attr):
       return getattr(self.stream, attr)

def subprocessrun(target):
    p = Process(target=target)
    p.start()
    p.join()

def main(argv):
    # buffer hack
    sys.stdout = Unbuffered(sys.stdout)

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


        if cc.cfg['model'] == 'rnn':
            # limitGrid = [10,50,100,500,1000,5000,10000,50000,100000]

            # for lg in limitGrid:
                # cc.exp['fetch']['limit'] = lg
                # cc.exp['params']['rnn']['comment'] = '[LEARNING_CURVE][LIMIT_{}] limiting data to {} rows'.format(lg,lg)

                # reloads the rnn.rnn module which populates new config values to their respective vars

            # rnn.rnn.run()


            # REMOVEME: hardcoded grid for neurons
            grid = [550,600]
            cc.exp['grid'] = {}

            import rnn.rnn
            reload(rnn.rnn)

            comment = cc.exp['params']['rnn']['comment']
            for neurons in grid:
                cc.exp['params']['rnn']['comment'] = comment.format(neurons)
                cc.exp['grid']['neurons'] = neurons

                print(cc.cfg,cc.exp)
                subprocessrun(rnn.rnn.run)


        elif cc.cfg['model'] == 'dnn':
            import rnn.dnn
            reload(rnn.dnn)

            print(cc.cfg,cc.exp)
            subprocessrun(rnn.dnn.run)
        else:
            raise Exception('Run: unknown model')

if __name__ == '__main__':
    main(sys.argv[1:])
