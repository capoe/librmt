import numpy as np
import copy
import pickle

# ============
# STATE OBJECT
# ============

class State(object):
    def __init__(self, **kwargs):
        self.state = kwargs
        self.storage = {}
        self.history = []
        return
    def __getitem__(self, key):
        return self.state[key]
    def __setitem__(self, key, value):
        self.state[key] = value
        return
    def register(self, stamp, options):
        self.history.append((stamp, options))
        return
    def store(self, key, value):
        self.storage[key] = value
        return
    def clone(self):
        return copy.deepcopy(self)
    def printInfo(self, log):
        for item in self.history:
            log << log.mb << item[0] << log.endl
            for key in item[1]:
                log << log.item << key << item[1][key] << log.endl
        return
    def pickle(self, pfile='state.jar'):
        pstr = pickle.dumps(self)
        with open(pfile, 'w') as f:
            f.write(pstr)
        return
