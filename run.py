#! /usr/bin/env python

import matplotlib
matplotlib.use('PDF')

import sys, getopt
from config import config as cc
from keras.optimizers import Adam, Adadelta, Adagrad, Nadam, Adamax
import theano

class Unbuffered(object):
   def __init__(self, stream):
       self.stream = stream
   def write(self, data):
       self.stream.write(data)
       self.stream.flush()
   def __getattr__(self, attr):
       return getattr(self.stream, attr)

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

            if cc.cfg['grid']:

                # REMOVEME: hardcoded grid for ratio
                grid = [
                    [1,1,1],
                    [1,1,2],
                    [1,2,1],
                    [2,1,1],
                    [1,2,2],
                    [2,1,2],
                    [2,2,1]
                ]
                cc.exp['grid'] = {}

                import rnn.rnn
                reload(rnn.rnn)

                comment = '[GRID][RATIO={}][A549][TDGRUGRU] performing grid search for ratio'
                for ratios in grid:
                    cc.exp['params']['rnn']['comment'] = comment.format(':'.join([str(x) for x in ratios]))
                    cc.exp['grid']['ratios'] = ratios

                    print(cc.cfg,cc.exp)
                    rnn.rnn.run()
            else:
                import rnn.rnn
                reload(rnn.rnn)

                print(cc.cfg,cc.exp)
                rnn.rnn.run()

        elif cc.cfg['model'] == 'dnn':
            import rnn.dnn
            reload(rnn.dnn)

            print(cc.cfg,cc.exp)
            rnn.dnn.run()
        else:
            raise Exception('Run: unknown model')

if __name__ == '__main__':
    main(sys.argv[1:])
