import yaml
import os

config = { }

exp = { }

def loadYAML(path):
    with open(path,'r') as f:
        return yaml.load(f) or {}

def loadConfig(path):
    global config

    config = loadYAML(os.path.dirname(__file__)+'/config.yml')

    try:
        config.update(loadYAML(path))
    except:
        raise IOError('Config: error loading config from path: {}'.format(path))

def loadExperiment(experimentPath):
    global exp

    exp = { }
    try:
        expObj = loadYAML(experimentPath)
        if 'template' in expObj and expObj['template']:
            exp = loadYAML(os.path.dirname(__file__)+'/templates/'+expObj['template'])
        exp.update(expObj)
    except:
        raise IOError('Config: error loading config from path: {}'.format(experimentPath))
