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
        self["has_T"] = False # Targets
        self["has_L"] = False # Labels
        self["has_K"] = False # Kernel
        return
    def __getitem__(self, key):
        return self.state[key]
    def __setitem__(self, key, value):
        self.state[key] = value
        return
    def __contains__(self, key):
        return key in self.state
    def __len__(self):
        return len(self["configs"])
    def register(self, stamp, options):
        self.history.append((stamp, options))
        return
    def keys(self):
        return sorted(self.state.keys())
    def printKeys(self, log):
        for k in self.keys():
            v = self[k]
            t = type(v)
            if t == list:
                log << "%-15s : %s %s" % (k, str(t), str(type(v[0]))) << log.endl
                if type(v[0]) == dict:
                    for k in sorted(v[0].keys()):
                        log << "%-15s . - %-15s %s" % ('', str(k), str(type(v[0][k]))) << log.endl
            else:
                log << "%-15s : %s" % (k, str(t)) << log.endl
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
    def unpickle(self, pfile, log=None):
        if log: log << "Loading state from '%s'" % pfile << log.endl
        self = pickle.load(open(pfile, 'rb'))
        return self
