import yaml
import os

config = { }

def load(path):
    global config
    with open(os.path.dirname(__file__)+'/default.yml','r') as f:
        config = yaml.load(f)
    try:
        with open(path,'r') as f:
            obj = yaml.load(f) or {}
            config.update(obj)
    except:
        raise IOError('Config: error loading config from path: {}'.format(path))
