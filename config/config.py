import yaml
import os
import collections
import sys

cfg = { }

exp = { }


def update(d, u):
    for k, v in u.iteritems():
        if isinstance(v, collections.Mapping):
            r = update(d.get(k, {}), v)
            d[k] = r
        else:
            d[k] = u[k]
    return d

def loadYAML(path):
    with open(path,'r') as f:
        return yaml.load(f) or {}

def loadConfig(path):
    global cfg

    cfg = loadYAML(os.path.dirname(__file__)+'/config.yml')

    try:
        update(cfg,loadYAML(path))
    except:
        raise IOError('Config: error loading config from path: {}'.format(path))

def loadExperiment(experimentPath):
    global exp

    exp = { }

    expObj = loadYAML(experimentPath)

    if 'template' in expObj and expObj['template']:
        exp = loadYAML(os.path.dirname(__file__)+'/templates/'+expObj['template'])

    update(exp, expObj)
