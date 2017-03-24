#! /usr/bin/env python
import json
import numpy as np
import os

# ==============
# READ/FILTER IO
# ==============

class ConfigASE(object):
    def __init__(self):
        self.info = {}
        self.cell = None
        self.pbc = np.array([False, False, False])
        self.atoms = []
        self.positions = []
        self.symbols = []
    def get_positions(self):
        return self.positions
    def get_chemical_symbols(self):
        return self.symbols
    def create(self, n_atoms, fs):
        header = fs.readline().split()
        # Read key-value pairs
        for key_value in header:
            kv = key_value.split('=')
            key = kv[0]
            value = '='.join(kv[1:])
            value = value.replace('"','').replace('\'','')
            # Float?
            if '.' in value:
                try:
                    value = float(value)
                except: pass
            else:
                # Int?
                try:
                    value = int(value)
                except: pass
            self.info[kv[0]] = value
        # Read atoms
        self.positions = []
        self.symbols = []
        for i in range(n_atoms):
            new_atom = self.create_atom(fs.readline())
            self.positions.append(new_atom.pos)
            self.symbols.append(new_atom.name)
        return
    def create_atom(self, ln):
        ln = ln.split()
        name = ln[0]
        pos = map(float, ln[1:4])
        pos = np.array(pos)
        new_atom = AtomASE(name, pos)
        self.atoms.append(new_atom)
        return new_atom

class AtomASE(object):
    def __init__(self, name, pos):
        self.name = name
        self.pos = pos

def read(
        config_file,
        index=':'):
    configs = []
    ifs = open(config_file, 'r')
    while True:
        header = ifs.readline().split()
        if header != []:
            assert len(header) == 1
            n_atoms = int(header[0])
            config = ConfigASE()
            config.create(n_atoms, ifs)
            configs.append(config)
        else: break
    return configs

def read_filter_configs(
        config_file, 
        index=':', 
        filter_types=None, 
        types=[],
        do_remove_duplicates=False, 
        key=lambda c: c.info['label'],
        log=None):
    if log: log << "Reading" << config_file << log.endl
    configs = read_xyz(config_file, index=index)
    if log: log << log.item << "Have %d initial configurations" % len(configs) << log.endl
    if do_remove_duplicates:
        configs, duplics = remove_duplicates(configs, key=key)
        if log: log << log.item << "Removed %d duplicates" % len(duplics) << log.endl
    if filter_types:
        configs_filtered = []
        for config in configs:
            types_config = config.get_chemical_symbols()
            keep = True
            for t in types_config:
                if not t in types:
                    keep = False
                    break
            if keep: configs_filtered.append(config)
        configs = configs_filtered
        if log: log << log.item << "Have %d configurations after filtering" % len(configs) << log.endl
    return configs

def remove_duplicates(array, key=lambda a: a):
    len_in = len(array)
    label = {}
    array_curated = []
    array_duplicates = []
    for a in array:
        key_a = key(a)
        if key_a in label:
            array_duplicates.append(a)
        else:
            array_curated.append(a)
            label[key_a] = True
    len_out = len(array_curated)
    return array_curated, array_duplicates

