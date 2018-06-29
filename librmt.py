#! /usr/bin/env python
import json
import numpy as np
import pickle
import h5py
import time
import datetime
import os
import numpy.linalg as la
import scipy.integrate
import scipy.stats
import copy
import csv
# Internal libraries
import ising as ising
import nonlinear as nonlinear
import labs as labs
import utils as utils
import launch as launch
import hamiltonian as hamiltonian
import npfga as npfga
import sklearn.metrics
from state import State
from pipe import Pipe
# External libraries
from sklearn import linear_model
try:
    from rdkit.Chem import AllChem as chem
except:
    pass
try:
    #import soap
    raise ImportError
except ImportError:
    from external import soap
    log = soap.soapy.momo.osio
try:
    #import ase.io as ase_io
    raise ImportError
except ImportError:
    import external.ase.io as ase_io

# ==========
# CONFIG I/O
# ==========

def load_configs(state, options, log):
    log.prefix += '[load] '
    if "load_configs" in options:
        options = options["load_configs"]
    method = options['load_method']
    log.prefix = log.prefix[0:-7]
    return load_configs_factory[method](state, options[method], log)

def load_pairs(state, options, log):
    log.prefix += '[load] '
    log << log.mg << "Loading pairs from initial configurations ..." << log.endl
    if "load_pairs" in options:
        options = options["load_pairs"]
    n_dim_in = state["n_dim"]
    n_dim_out = 2*n_dim_in
    n_samples_in = state["n_samples"]
    combination_rule = options["combination_rule"]
    log << "Combination rule for descriptors: '%s'" % combination_rule << log.endl
    # CREATE LABEL-TO-IDX MAP
    idx_map = { }
    for i in range(state["n_samples"]):
        label = state["labels"][i]["label"]
        idx_map[label] = i
    # READ PAIR DATA
    csv_file = options["pair_table"]
    ifs = open(csv_file, 'r')
    reader = csv.DictReader(ifs, delimiter=',', quotechar='"')
    pairs = []
    IX_out = []
    labels_out = []
    configs_out = []
    for row in reader:
        l1, l2 = row['pair'].split(':')
        i1, i2 = idx_map[l1], idx_map[l2]
        info = { key:row[key] for key in row }
        info["pair-label-a"] = state["labels"][i1]
        info["pair-label-b"] = state["labels"][i2]
        info["idx"] = len(configs_out)
        # SYMMETRISE & COMBINE DESCRIPTORS
        x1, x2 = state["IX"][i1], state["IX"][i2]
        if combination_rule == "symmetrise":
            x_sym_p = 0.5*(x1+x2)
            x_sym_m = 0.5*np.abs(x1-x2)
            x = np.zeros((n_dim_out,), dtype=state["IX"].dtype)
            x[0:n_dim_in] = x_sym_p
            x[n_dim_in:n_dim_out] = x_sym_m
        elif combination_rule == "concatenate":
            x = np.zeros((n_dim_out,), dtype=state["IX"].dtype)
            x[0:n_dim_in] = x1
            x[n_dim_in:n_dim_out] = x2
        else:
            raise NotImplementedError("Combination rule: '%s'" % combination_rule)
        IX_out.append(x)
        # GENERATE PAIR LABELS AND CONFIG
        labels_out.append(info)
        pair_config = soap.soapy.momo.ExtendableNamespace()
        pair_config.a = state["configs"][i1]
        pair_config.b = state["configs"][i2]
        pair_config.info = info
        configs_out.append(pair_config)
        #print x1, x2
        #print x
        #print x1.sum()
        #print x2.sum()
        #print x.sum()
        #print labels_out[-1]
        #raw_input('...')
    IX_out = np.array(IX_out)
    n_samples_out = IX_out.shape[0]
    ifs.close()
    # SAVE STATE
    log << "# pairs:" << n_samples_out << log.endl
    log << "Pair descriptor dimension:" << n_dim_out << log.endl
    state.register("load_pairs", options)
    state["configs"] = configs_out
    state["labels"] = labels_out
    state["IX"] = IX_out
    state["n_samples"] = n_samples_out
    state["n_dim"] = n_dim_out
    log.prefix = log.prefix[0:-7]
    return state


def load_configs_extended_xyz(state, options, log):
    """
    # INPUT
      - Extended xyz
    # OUTPUT
      - state['configs']
      - state['labels']
    """
    log.prefix += '[load] '
    log << log.mg << "Load configurations <load_configs_extended_xyz> ..." << log.endl
    xyz_file = options["xyz_file"]
    n_select = options["n_select"]
    subsample_method = options["subsample_method"]
    filter_types = options["filter_types"]
    types = options["types"]
    if "rm_duplicates" in options:
        rm_dupls = options["rm_duplicates"]
        rm_dupls_fct = options["rm_duplicates_decision_fct"]
    else:
        rm_dupls = True
        rm_dupls_fct = lambda c: c.info['label']
    # READ & FILTER
    configs_A = read_filter_configs(
        xyz_file,
        index=':',
        filter_types=filter_types,
        types=types,
        do_remove_duplicates=rm_dupls,
        key=rm_dupls_fct,
        log=log)
    configs_A, configs_A2 = soap.soapy.learn.subsample_array(
        configs_A, n_select=n_select, method=subsample_method, stride_shift=0)
    configs = configs_A
    log << "Selected" << len(configs) << "configurations" << log.endl
    # Index; find all chemical elements in dataset
    for idx, c in enumerate(configs):
        c.info['idx'] = idx
    # COMPILE TYPES
    types_global = []
    for idx, c in enumerate(configs):
        types_global = types_global + c.get_chemical_symbols()
        types_global = list(set(types_global))
    types_global = sorted(types_global)
    log << "Compiled global types list:" << types_global << log.endl
    # COMPILE LABELS
    labels = [ c.info for c in configs ]
    # SAVE STATE
    state.register("load_configs_extended_xyz", options)
    state["configs"] = configs
    state["labels"] = labels
    state.store("types_global", types_global)
    log.prefix = log.prefix[0:-7]
    return state

def read_filter_configs(
        config_file,
        index=':',
        filter_types=None,
        types=[],
        do_remove_duplicates=True,
        key=lambda c: c.info['label'],
        log=None):
    if log: log << "Reading" << config_file << log.endl
    configs = ase_io.read(config_file, index=index)
    if log: log << log.item << "Have %d initial configurations" % len(configs) << log.endl
    if do_remove_duplicates:
        configs, duplics = remove_duplicates(configs, key=key)
        if log:
            log << log.item << "Removed %d duplicates" % len(duplics) << log.endl
            #for d in duplics:
            #    log << d.info['label'] << log.endl
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

def import_hdf5_basic(state, options, log):
    f = h5py.File(options["import_hdf5_basic"]["hdf5_file"], 'r')
    state['IX'] = f['IX'].value
    state['has_IX'] = True
    state['T'] = f['T'].value
    state['has_T'] = True
    configs = []
    for i in range(state['IX'].shape[0]):
        configs.append(soap.soapy.momo.ExtendableNamespace())
        configs[-1].info = { 'idx': i, 'tag': 's%06d' % i }
    state['configs'] = configs
    return state

def import_hdf5(state, options, log):
    log.prefix += '[load] '
    if "import_hdf5" in options:
        options = options["import_hdf5"]
    hdf5_file = options["hdf5_file"]
    filter_fct = options["filter"]
    if "import_graphs" in options:
        do_import_graphs = options["import_graphs"]
    else:
        do_import_graphs = False
    # Open hdf5
    f = h5py.File(hdf5_file, 'r')
    # Data arrays: labels, targets, kernel
    log << log.mg << "Importing data from hdf5 '%s'" % hdf5_file << log.endl
    labels = f['labels']['label_mat'].value
    T = labels['target-value']
    K = f['kernel']['kernel_mat'].value
    # Filter if requested
    idcs = []
    if filter_fct:
        log << "Filtering ..." << log.endl
        for i in range(labels.shape[0]):
            if filter_fct(labels[i]):
                idcs.append(i)
            else: pass
    labels = labels[idcs]
    T = T[idcs]
    K = K[idcs][:,idcs]
    # Create configs from labels
    configs = []
    n_samples = labels.shape[0]
    log << "Generating %d configurations from labels" % n_samples << log.endl
    fields = labels.dtype.names
    for i in range(n_samples):
        config = soap.soapy.momo.ExtendableNamespace()
        config.info = {}
        config.idx = i
        for field in fields:
            config.info[field] = labels[field][i]
        configs.append(config)
    # Import graphs (i.e., descriptors)
    if do_import_graphs:
        IJX = [] # i->config, j->atom, X->descriptor
        IX = []
        for c in configs:
            g = f['graphs']['%06d' % c.idx]
            ijx = g['feature_mat'].value
            IJX.append(ijx)
            ix = np.sum(ijx, axis=0)
            IX.append(ix)
            g_info = g.attrs.keys()
            c.info['vertex_info'] = g.attrs['vertex_info']
            c.info['graph_info'] = g.attrs['graph_info']
        IX = np.array(IX)
    else:
        IX = np.copy(K)
        IJX = []
    # Push data
    state["configs"] = configs
    state["labels"] = labels
    state["has_K"] = True
    state["K"] = K
    state["IX"] = IX # TODO Fix this: has_IX flag?
    state["IJX"] = IJX
    log << "Descriptor matrix: %dx%d %s" % (IX.shape[0], IX.shape[1], IX.dtype) << log.endl
    # Close hdf5 and return
    f.close()
    log.prefix = log.prefix[:-7]
    return state

# =======
# TARGETS
# =======

def load_labels(state, options, log):
    log.prefix += '[labl] '
    if 'load_labels' in options:
        options = options['load_labels']
    log << log.mg << "Load labels" << log.endl
    key = options['label_key']
    fct = options['label_conv_fct']
    L = [ fct(config.info[key]) for config in state["configs"] ]
    if "label_map" in options:
        label_map = options["label_map"]
        idx_to_label_map = { label_map[l]: l for l in label_map }
        log << "Using user-defined label map:" << label_map << log.endl
        log << "Inverse map:" << idx_to_label_map << log.endl
    else:
        label_map = { l: idx for idx, l in enumerate( sorted(list(set(L))) ) }
        idx_to_label_map = { idx: l for idx, l in enumerate( sorted(list(set(L))) ) }
    L = np.array([ label_map[l] for l in L ])
    for l in label_map:
        log << "Label '%s' -> '%d' : %d samples" % (
            l, label_map[l], np.where(L == label_map[l])[0].shape[0]) << log.endl
    state.register("load_labels", options)
    state["L"] = L
    state["has_L"] = True
    state["label_to_idx_map"] = label_map
    state["idx_to_label_map"] = idx_to_label_map
    log.prefix = log.prefix[0:-7]
    return state

def load_targets(state, options, log):
    log.prefix += '[targ] '
    if 'load_targets' in options:
        options = options['load_targets']
    log << log.mg << "Load targets" << log.endl
    if 'extraction_fct' in options and options['extraction_fct'] != None:
        log << "Using extraction function (ignoring 'target_key' and 'target_conv_fct')" << log.endl
        fct = options['extraction_fct']
        T = np.array([ fct(config) for config in state["configs"] ])
    else:
        key = options['target_key']
        fct = options['target_conv_fct']
        T = np.array([ fct(config.info[key]) for config in state["configs"] ])
    log << "<t>" << np.average(T) << "(min, max =" << np.min(T) << np.max(T) << ")" << log.endl
    log << "sqrt<dt^2>" << np.std(T) << log.endl
    state.register("load_targets", options)
    state["T"] = T
    state["has_T"] = True
    log.prefix = log.prefix[0:-7]
    return state

def normalise_targets(state, options, log):
    log.prefix += '[targ] '
    if 'normalise_targets' in options:
        options = options['normalise_targets']
    log << log.mg << "Normalise targets" << log.endl
    centre = options['centre']
    fct = options['target_conv_fct']
    T_train = state["T_train"]
    T_test = state["T_test"]
    log << "Before normalisation" << log.endl
    log << "<t_train> min, avg, max, std =" << np.min(T_train) << np.average(T_train) << np.max(T_train) << np.std(T_train) << log.endl
    log << "<t_test>  min, avg, max, std =" << np.min(T_test) << np.average(T_test) << np.max(T_test) << np.std(T_test) << log.endl
    if fct:
        T_train = fct(T_train)
        T_test = fct(T_test)
    if centre:
        T_train_avg = np.average(T_train)
        T_train = T_train - T_train_avg
        T_test = T_test - T_train_avg
    else:
        T_train_avg = 0.0
    state.register("normalise_targets", options)
    state["T_train"] = T_train
    state["T_test"] = T_test
    state["t_average"] = T_train_avg
    log << "After normalisation" << log.endl
    #log << "<t_train>" << np.average(T_train) << np.std(T_train) << log.endl
    #log << "<t_test>" << np.average(T_test) << np.std(T_test) << log.endl
    log << "<t_train> min, avg, max, std =" << np.min(T_train) << np.average(T_train) << np.max(T_train) << np.std(T_train) << log.endl
    log << "<t_test>  min, avg, max, std =" << np.min(T_test) << np.average(T_test) << np.max(T_test) << np.std(T_test) << log.endl
    log << "t_average" << T_train_avg << log.endl
    log.prefix = log.prefix[0:-7]
    return state

# ===========
# DESCRIPTORS
# ===========

def compute_descriptor(state, options, log):
    log.prefix += '[dtor] '
    if "compute_descriptor" in options:
        options = options["compute_descriptor"]
    method = options["descriptor_method"]
    state = compute_descriptor_factory[method](state, options[method], log)
    log.prefix = log.prefix[0:-7]
    return state

def compute_descriptor_morgan(state, options, log):
    log << log.mg << "Compute descriptor <compute_descriptor_morgan> ..." << log.endl
    configs = state["configs"]
    morgan_radius = options['morgan_radius']
    length = options['morgan_length']
    log << "Computing Morgan descriptor with:" << log.endl
    # Options
    log << log.item << "Morgan radius:" << morgan_radius << log.endl
    log << log.item << "Bit length:" << length << log.endl
    # SMILES to molecule
    log << "Generating molecules from smiles ..." << log.endl
    #for c in configs:
    #    print c.info
    smiles = [ config.info['smiles'] for config in configs ]
    mols = [chem.MolFromSmiles(c) for c in smiles]
    #for config, c in zip(configs, smiles):
    #    mol = chem.MolFromSmiles(c)
    #    if str(type(mol)) != "<class 'rdkit.Chem.rdchem.Mol'>":
    #        print config.info
    #        raise ValueError("Could not create molecule from SMILES")
    # Compute
    log << "Computing fingerprints ..." << log.endl
    fingerprints = [chem.GetMorganFingerprintAsBitVect(mol, radius=morgan_radius,
                    nBits=length).ToBitString() for mol in mols]
    # Format
    fingerprints = [list(map(float, list(fingerprint))) for fingerprint in fingerprints]
    fingerprints = np.array([np.array(fingerprint) for fingerprint in fingerprints])
    # SAVE STATE
    state.register("compute_descriptor_morgan", options)
    state["IX"] = fingerprints
    state["n_samples"] = fingerprints.shape[0]
    state["n_dim"] = fingerprints.shape[1]
    return state

def compute_soap(state, options, log):
    raise NotImplementedError(options)
    return None

def upconvert_descriptor(state, options, log):
    log.prefix += '[dtor] '
    if "upconvert_descriptor" in options:
        options = options["upconvert_descriptor"]
    log << options << log.endl
    if "upconvert" in options:
        if options["upconvert"] == False:
            log << "No up-conversion requested, return." << log.endl
            return state
    log << log.mg << "Upconverting descriptor" << log.endl
    # OPTIONS
    upscale = options["upscale"]
    if "concatenate" in options:
        concatenate = options["concatenate"]
    else: concatenate = False
    # READ STATE
    n_train = state["n_train"]
    n_test = state["n_test"]
    IX_train = np.copy(state["IX_train"])
    IX_test = np.copy(state["IX_test"])
    n_dim = IX_train.shape[1]
    # INCOMING
    n_comps = int(upscale*n_dim)
    IX_train = IX_train[:,-n_comps:]
    IX_test = IX_test[:,-n_comps:]
    # OUTGOING
    n_dim_up = n_comps*(n_comps+1)/2
    log << "Dimension (in)" << n_dim << " (up)" << n_dim_up << log.endl
    IX_train_up = np.zeros((n_train, n_dim_up), dtype='float64')
    IX_test_up = np.zeros((n_test, n_dim_up), dtype='float64')
    idcs_upper = np.triu_indices(n_comps)
    for i in range(n_train):
        IX_train_up[i] = np.outer(IX_train[i], IX_train[i])[idcs_upper]
    for i in range(n_test):
        IX_test_up[i] = np.outer(IX_test[i], IX_test[i])[idcs_upper]
    if concatenate:
        log << "Concatenating descriptors" << log.endl
        IX_train_full = np.zeros((n_train, n_dim_up+n_dim), dtype='float64')
        IX_test_full = np.zeros((n_test, n_dim_up+n_dim), dtype='float64')
        IX_train_full[:,0:n_dim] = np.copy(state["IX_train"])
        IX_test_full[:,0:n_dim] = np.copy(state["IX_test"])
        IX_train_full[:,n_dim:n_dim+n_dim_up] = IX_train_up
        IX_test_full[:,n_dim:n_dim+n_dim_up] = IX_test_up
        n_dim_full = n_dim + n_dim_up
    else:
        IX_train_full = IX_train_up
        IX_test_full = IX_test_up
        n_dim_full = n_dim_up
    # SAVE STATE
    state.register("upconvert_descriptor", options)
    state["IX_train"] = IX_train_full
    state["IX_test"] = IX_test_full
    state["n_dim"] = n_dim_full
    state["mp_gamma"] = float(n_dim_full)/n_train
    log << "New descriptor dimension:" << n_dim_full << log.endl
    log.prefix = log.prefix[0:-7]
    return state

# =============
# RM Operations
# =============

def feature_select(state, options, log):
    log.prefix += '[dtor] '
    if "feature_select" in options:
        options = options["feature_select"]
    method = options["method"]
    if method == 'random':
        n_select = options["n_select"]
        n_dim = state["n_dim"]
        log << "Select features at random:" << n_select << log.endl
        idcs_select = np.random.randint(0, n_dim, n_select)
        state["IX_train"] = state["IX_train"][:,idcs_select]
        state["IX_test"] = state["IX_test"][:,idcs_select]
        descriptor_dim = state["IX_train"].shape[1]
        mp_gamma = float(descriptor_dim)/state["n_train"]
        log << "Reduced" << n_dim << "to" << idcs_select.shape[0] << "components" << log.endl
        log << "MP gamma:" << mp_gamma << log.endl
    else:
        raise NotImplementedError("[feature_select] " + method)
    # SAVE STATE
    state.register("feature_select", options)
    state["n_dim"] = descriptor_dim
    state["mp_gamma"] = mp_gamma
    log.prefix = log.prefix[0:-7]
    return state

def clean_descriptor_matrix(state, options, log):
    log.prefix += '[dtor] '
    if 'clean_descriptor_matrix' in options:
        options = options['clean_descriptor_matrix']
    std_threshold = options["std_threshold"]
    # CLEAN DESCRIPTOR MATRIX
    log << log.mg << "Clean descriptor matrix" << log.endl
    IX_train_std = np.std(state["IX_train"], axis=0)
    idcs_std_non_zero = np.where(IX_train_std > std_threshold)[0]
    state["IX_train"] = state["IX_train"][:,idcs_std_non_zero]
    state["IX_test"] = state["IX_test"][:,idcs_std_non_zero]
    descriptor_dim = state["IX_train"].shape[1]
    mp_gamma = float(descriptor_dim)/state["n_train"]
    log << "Reduced to" << idcs_std_non_zero.shape[0] << "components" << log.endl
    log << "MP gamma:" << mp_gamma << log.endl
    # SAVE STATE
    state.register("clean_descriptor_matrix", options)
    state["n_dim"] = descriptor_dim
    state["mp_gamma"] = mp_gamma
    log.prefix = log.prefix[0:-7]
    return state

def clean_descriptor_pca_by_class(state, options, log):
    log.prefix += '[pca+] '
    log << log.mg << "Running multimodal PCA" << log.endl
    # OPTIONS
    if 'clean_descriptor_pca_by_class' in options:
        options = options['clean_descriptor_pca_by_class']
    dec_fcts = options["decision_fcts"]
    n_classes = len(dec_fcts)
    verbose = False
    # ... pca-related
    select_pc_method = options["select_pc_method"]
    n_select = options["n_select"]
    norm_std = options["norm_std"]
    norm_avg = options["norm_avg"]
    # PARTITION ONTO CLASSES
    T_train = state["T_train"]
    idcs_classes = []
    for i in range(n_classes):
        dec_fct = dec_fcts[i]
        idcs = []
        for j in range(T_train.shape[0]):
            if dec_fct(T_train[j]):
                idcs.append(j)
            else: pass
        idcs_classes.append(idcs)
    # PERFORM PCA FOR EACH CLASS
    n_dim_cum = 0
    pca_props = []
    for class_idx, idcs in enumerate(idcs_classes):
        IX_train = state["IX_train"][idcs]
        T_train = state["T_train"][idcs]
        if verbose: log << T_train << log.endl
        n_dim = IX_train.shape[1]
        n_train_class = len(idcs)
        mp_gamma_class = float(n_dim)/n_train_class
        # Z-SCORE PCA
        IX_train_norm_pca, IZ, X_mean, X_std, S, L, V = pca_compute(
            IX=IX_train,
            log=None,
            norm_div_std=norm_std,
            norm_sub_mean=norm_avg,
            eps=0.,
            ddof=1)
        # DECOMPOSE INTO NOISE + SIGNAL
        n_dim_mp_signal = np.where( L.diagonal() > dist_mp_bounds(mp_gamma_class)[1] )[0].shape[0]
        if select_pc_method == "mp":
            if verbose: log << "Select MP signal ..." << log.endl
            idcs_eigen_signal = np.where( L.diagonal() > dist_mp_bounds(mp_gamma_class)[1] )[0]
            L_signal = L.diagonal()[idcs_eigen_signal]
            V_signal = V[:,idcs_eigen_signal]
            if verbose: log << "Components above MP threshold:" << len(idcs_eigen_signal) << log.endl
        elif select_pc_method == "n_largest":
            if n_select < 0:
                threshold = L.diagonal()[0]-1.
            else:
                threshold = L.diagonal()[-n_select-1]
            if verbose: log << "Select n =" << n_select << "largest ..." << log.endl
            if verbose: log << "Threshold:" << threshold << log.endl
            idcs_eigen_signal = np.where( L.diagonal() > threshold )[0]
            L_signal = L.diagonal()[idcs_eigen_signal]
            V_signal = V[:,idcs_eigen_signal]
        else: raise NotImplementedError(select_pc_method)
        log << "Class idx %d: %4d training samples, %2d signal components" % (
            class_idx, n_train_class, V_signal.shape[1]) << log.endl
        pca_props.append([X_mean, X_std, V_signal, L_signal, n_dim_cum, n_dim_cum+V_signal.shape[1]])
        n_dim_cum += V_signal.shape[1]
    n_dim_mp_signal = n_dim_cum
    V_signal_all = np.zeros((state["IX_train"].shape[1], n_dim_cum), dtype='float64')
    L_signal_all = np.zeros((n_dim_cum,), dtype='float64')
    IZ_pc_signal_train = np.zeros((state["IX_train"].shape[0], n_dim_cum), dtype='float64')
    IZ_pc_signal_test = np.zeros((state["IX_test"].shape[0], n_dim_cum), dtype='float64')
    for i in range(n_classes):
        props = pca_props[i]
        X_mean = props[0]
        X_std = props[1]
        V_signal = props[2]
        L_signal = props[3]
        i0 = props[4]
        i1 = props[5]
        V_signal_all[:,i0:i1] = V_signal
        L_signal_all[i0:i1] = L_signal
        # Normalise & project
        IZ_train = utils.div0(state["IX_train"] - X_mean, X_std)
        IZ_test = utils.div0(state["IX_test"] - X_mean, X_std)
        IZ_pc_signal_train_class = IZ_train.dot(V_signal)
        IZ_pc_signal_test_class = IZ_test.dot(V_signal)
        # Store in "super"-descriptor
        log << "Projection for class %d, components %d:%d" % (i, i0, i1) << log.endl
        IZ_pc_signal_train[:,i0:i1] = IZ_pc_signal_train_class
        IZ_pc_signal_test[:,i0:i1] = IZ_pc_signal_test_class
    np.set_printoptions(precision=1)
    # Inspect PC vector overlap between classes
    #print L_signal_all
    #k = V_signal_all.T.dot(V_signal_all)
    #np.savetxt('tmp.txt', k)
    # Register
    state.register("clean_descriptor_pca", options)
    state["pca_params"] = pca_props
    state["pca_V_signal"] = V_signal_all
    state["pca_L_signal"] = L_signal_all
    state["IX_train"] = IZ_pc_signal_train # transformed coordinates
    state["IX_test"] = IZ_pc_signal_test
    state["n_dim"] = IZ_pc_signal_train.shape[1]
    state["mp_gamma"] = float(state["IX_train"].shape[1])/state["IX_train"].shape[0]
    log.prefix = log.prefix[0:-7]
    return state 

def clean_descriptor_pca(state, options, log):
    log.prefix += '[dtor] '
    output_eigenspectrum = False
    if 'clean_descriptor_pca' in options:
        options = options['clean_descriptor_pca']
    log << log.mg << "Transform descriptor <clean_descriptor_pca>" << log.endl
    # OPTIONS
    select_pc_method = options["select_pc_method"]
    n_select = options["n_select"]
    norm_std = options["norm_std"]
    norm_avg = options["norm_avg"]
    # RETRIEVE STATE VARIABLES
    IX_train = state["IX_train"]
    IX_test = state["IX_test"]
    mp_gamma = state["mp_gamma"]
    log << "gamma_mp = D/N =" <<  mp_gamma << log.endl
    assert mp_gamma == float(IX_train.shape[1])/IX_train.shape[0]
    # Z-SCORE PCA
    IX_train_norm_pca, IZ, X_mean, X_std, S, L, V = pca_compute(
        IX=IX_train,
        log=log,
        norm_div_std=norm_std,
        norm_sub_mean=norm_avg,
        eps=0.,
        ddof=1)
    # DECOMPOSE INTO NOISE + SIGNAL
    n_dim_mp_signal = np.where( L.diagonal() > dist_mp_bounds(mp_gamma)[1] )[0].shape[0]
    if select_pc_method == "mp":
        log << "Select MP signal ..." << log.endl
        idcs_eigen_signal = np.where( L.diagonal() > dist_mp_bounds(mp_gamma)[1] )[0]
        L_signal = L.diagonal()[idcs_eigen_signal]
        V_signal = V[:,idcs_eigen_signal]
        log << "Components above MP threshold:" << len(idcs_eigen_signal) << log.endl
    elif select_pc_method == "n_largest":
        if n_select < 0:
            threshold = L.diagonal()[0]-1.
        else:
            threshold = L.diagonal()[-n_select-1]
        log << "Select n =" << n_select << "largest ..." << log.endl
        log << "Threshold:" << threshold << log.endl
        idcs_eigen_signal = np.where( L.diagonal() > threshold )[0]
        L_signal = L.diagonal()[idcs_eigen_signal]
        V_signal = V[:,idcs_eigen_signal]
    else: raise NotImplementedError(select_pc_method)
    # ANALYSE EIGENSPECTRUM
    #bins = np.arange(0, int(np.max(L.diagonal())), 0.1)
    hist, bin_edges = np.histogram(a=L.diagonal(), bins=100, density=True)
    hist_signal, bin_edges = np.histogram(a=L_signal, bins=bin_edges, density=True)
    bin_centres = np.zeros((bin_edges.shape[0]-1,), dtype='float64')
    for i in range(bin_edges.shape[0]-1):
        bin_centres[i] = 0.5*(bin_edges[i]+bin_edges[i+1])
    mp_sample = dist_mp_sample(bin_centres, mp_gamma)
    if output_eigenspectrum:
        np.savetxt('out.pca_hist.txt', np.array([ bin_centres, hist, mp_sample, hist_signal ]).T)
        np.savetxt('out.pca_eigv.txt', L.diagonal())
    # TRANSFORM INTO PCA SPACE
    log << log.mg << "Project onto signal PCs" << log.endl

    # TODO TODO TODO vvvv
    IZ_train = utils.div0(IX_train - X_mean, X_std)
    IZ_test = utils.div0(IX_test - X_mean, X_std)
    #IZ_train = utils.div0(IX_train, X_std)
    #IZ_test = utils.div0(IX_test, X_std)
    #IZ_train = np.copy(IX_train)
    #IZ_test = np.copy(IX_test)
    #s_h = state["model"].s_h
    #s_h = s_h/np.dot(s_h, s_h)**0.5
    #V_signal = np.array([ s_h ]).T
    # ^^^^^


    IZ_pc_signal_train = IZ_train.dot(V_signal)
    IZ_pc_signal_test = IZ_test.dot(V_signal)
    # SAVE STATE
    state.register("clean_descriptor_pca", options)
    state["pca_IX_train"] = IX_train # original data matrix
    state["pca_IX_test"] = IX_test
    state["pca_X_mean"] = X_mean
    state["pca_X_std"] = X_std
    state["pca_S"] = S
    state["pca_L"] = L # eigenspace
    state["pca_V"] = V
    state["pca_L_signal"] = L_signal
    state["pca_V_signal"] = V_signal
    state["pca_idcs_signal"] = idcs_eigen_signal
    state["pca_IZ_train"] = IZ_train # z-scored data matrix
    state["pca_IZ_test"] = IZ_test
    state["IX_train"] = IZ_pc_signal_train # transformed coordinates
    state["IX_test"] = IZ_pc_signal_test
    state["n_dim_mp_signal"] = n_dim_mp_signal
    state["n_dim"] = IZ_pc_signal_train.shape[1]
    log.prefix = log.prefix[0:-7]
    return state

def dist_mp(x, gamma):
    # gamma = #dim / #samples
    l, u = dist_mp_bounds(gamma)
    if x <= l or x >= u:
        return 0.
    else:
        return ( (u - x)*(x - l) )**0.5 / (2*np.pi*gamma*x)

def dist_mp_bounds(gamma):
    #return (1.-gamma**0.5)**2, (1.+gamma**0.5)**2
    if gamma > 1.:
        return 0.0, (1.+gamma**0.5)**2
    else:
        return (1.-gamma**0.5)**2, (1.+gamma**0.5)**2

def dist_mp_sample(xs, gamma):
    ys = np.array([ dist_mp(x, gamma) for x in xs ])
    return ys

def dist_mp_test():
    xs = np.arange(0.,100.,0.01)
    ys = np.array([ [ dist_mp(x, 0.25), dist_mp(x, 0.5), dist_mp(x, 1.0), dist_mp(x, 2.0) ] for x in xs ])
    xys = np.zeros((xs.shape[0],5))
    xys[:,0] = xs
    xys[:,1:5] = ys
    np.savetxt('tmp.txt', xys)
    int_, err = scipy.integrate.quad( lambda x: dist_mp(x, 0.25), 0, 10.)
    print int_, "+/-", err
    return

def normalize_descriptor_zscore_deprecated(IX, ddof_=1):
    # Rows of x are individual observations
    mu = np.mean(IX, axis=0)
    muTile = np.tile(mu, (IX.shape[0],1))
    std = np.std(IX, axis=0, ddof=ddof_)
    stdTile = np.tile(std, (IX.shape[0], 1))
    IZ = utils.div0(IX - muTile, stdTile)
    return IZ, mu, std

def normalise_descriptor_centre(state, options, log):
    log.prefix += '[desc] '
    IX_train = state["IX_train"]
    IX_test = state["IX_test"]
    if IX_train.shape[1] > 0:
        log << log.mg << "Centering descriptor" << log.endl
        # Normalise
        X_mean = np.mean(IX_train, axis=0)
        log << "Mean min max:" << np.min(X_mean) << np.max(X_mean) << log.endl
        IX_train_norm = IX_train - X_mean
        IX_test_norm  = IX_test - X_mean
        # Store
        state.register("normalise_descriptor_centre", options)
        state["IX_train"] = IX_train_norm
        state["IX_test"] = IX_test_norm
        state["X_mean"] = X_mean
    else:
        print "WARNING Cannot z-score 0-dimensional descriptor"
    log.prefix = log.prefix[:-7]
    return state

def normalise_descriptor_zscore(state, options, log):
    log.prefix += '[desc] '
    if "normalise_descriptor_zscore" in options:
        options = options["normalise_descriptor_zscore"]
    if options["zscore"]:
        IX_train = state["IX_train"]
        IX_test = state["IX_test"]
        if IX_train.shape[1] > 0:
            log << log.mg << "Z-scoring descriptor" << log.endl
            # Normalise
            X_mean = np.mean(IX_train, axis=0)
            X_std = np.std(IX_train, axis=0, ddof=1)
            log << "Mean min max:" << np.min(X_mean) << np.max(X_mean) << log.endl
            log << "Std  min max:" << np.min(X_std) << np.max(X_std) << log.endl
            if "z" in options:
                log << "Have z target:" << options["z"] << log.endl
                X_std = utils.div0(X_std, options["z"])
            IX_train_norm = utils.div0(IX_train - X_mean, X_std)
            IX_test_norm = utils.div0(IX_test - X_mean, X_std)
            # Store
            state.register("clean_descriptor_zscore", options)
            state["IX_train"] = IX_train_norm
            state["IX_test"] = IX_test_norm
            state["X_mean"] = X_mean
            state["X_std"] = X_std
            if "data_logger" in state:
                log << "Logging z-score mean and std" << log.endl
                state["data_logger"].append({
                    "tag": "zscore", 
                    "data": { "mean": X_mean, "std": X_std }
                })
        else:
            print "WARNING Cannot z-score 0-dimensional descriptor"
    log.prefix = log.prefix[:-7]
    return state

def pca_compute(IX, log=None, norm_div_std=True, norm_sub_mean=True, ddof=1, eps=0.0):
    """
    To check result consider:
    IX_norm_pca.T.dot(IX_norm_pca)/IX_norm_pca.shape[0]
    """
    # Normalize: mean, std
    if log: log << "PCA: Normalize ..." << log.endl
    X_mean = np.mean(IX, axis=0)
    X_std = np.std(IX, axis=0, ddof=ddof)
    IX_norm = IX
    if norm_sub_mean:
        IX_norm = IX - X_mean
    else:
        X_mean = 0.0
    if norm_div_std:
        #IX_norm = IX_norm/(X_std+eps)
        IX_norm = utils.div0(IX_norm, X_std+eps)
    else:
        X_std = 1.0
    # Correlation matrix
    if log: log << "PCA: Correlate ..." << log.endl
    S = IX_norm.T.dot(IX_norm)/(IX_norm.shape[0]-ddof)
    #S = np.cov(IX_norm.T, ddof=1)
    # Diagonalize
    if log: log << "PCA: Eigenspace ..." << log.endl
    lambda_, U = np.linalg.eigh(S)
    idcs = lambda_.argsort()[::+1]
    lambda_ = lambda_[idcs]
    U = U[:,idcs]
    L = np.identity(lambda_.shape[0])*lambda_
    #print L-U.T.dot(S).dot(U) # <- should be zero
    #print S-U.dot(L).dot(U.T) # <- should be zero
    # Transform
    if log: log << "PCA: Transform ..." << log.endl
    IX_norm_pca = U.T.dot(IX_norm.T).T
    return IX_norm_pca, IX_norm, X_mean, X_std, S, L, U

# ========
# LEARNING
# ========

def split_test_train(state, options, log):
    if log: log.prefix += '[pred] '
    if 'split_test_train' in options:
        options = options['split_test_train']
    if "stride_shift" in options:
        stride_shift = options["stride_shift"]
    else:
        stride_shift = 0
    if "seed" in options:
        seed = options["seed"]
    else:
        seed = None
    if log: log << log.mg << "Split onto training and test set" << log.endl
    # Read options
    subsample_method = options["method"]
    # Calc n_train, n_test
    IX = state["IX"]
    n_samples = IX.shape[0]
    # Use decision fct?
    if "decision_fct" in options and options["decision_fct"] != None:
        if log: log << "Using decision function" << log.endl
        dfct = options["decision_fct"]
        idcs_test = []
        idcs_train = []
        for idx, config in enumerate(state["configs"]):
            if dfct(config):
                idcs_test.append(idx)
            else: idcs_train.append(idx)
    elif subsample_method in ["stride", "random"]:
        f_train = options["f_train"]
        n_train = int(f_train*n_samples+0.5)
        n_test = n_samples - n_train
        if log: 
            log << "Method:" << subsample_method
            if subsample_method == "stride":
                log << "(stride = %d)" % stride_shift
            log << log.endl
        # Subsampling
        idcs_train, idcs_test = soap.soapy.learn.subsample_array(
            np.arange(0, n_samples).tolist(), 
            n_select=n_train, 
            method=subsample_method, 
            stride_shift=stride_shift,
            seed=seed)
    elif subsample_method == "mask":
        if log: log << "Method:" << subsample_method << log.endl
        idcs_train = np.where(options["mask"] > 0)[0]
        idcs_test = np.where(options["mask"] == 0)[0]
    else:
        raise NotImplementedError("subsample_method '%s'" % subsample_method)
    if log:
        log << "Train idcs teaser:" << idcs_train[0:5] << log.endl
        log << "Test  idcs teaser:" << idcs_test[0:5] << log.endl
    n_train = len(idcs_train)
    n_test = len(idcs_test)
    # Train
    IX_train = IX[idcs_train]
    mp_gamma = float(IX_train.shape[1])/n_train
    if state["has_T"]:
        T = state["T"]
        T_train = T[idcs_train]
        T_test = T[idcs_test]
    if state["has_L"]:
        L = state["L"]
        L_train = L[idcs_train]
        L_test = L[idcs_test]
        #if len(list(set(list(L_train)))) != len(list(set(list(L_test)))):
        #    print L_train
        #    print L_test
        #    assert False # Not all classes present in train and test
    if state["has_K"]:
        K_train = np.zeros((n_train, n_train), dtype=state["K"].dtype)
        K_train = state["K"][idcs_train][:,idcs_train]
        K_test = np.zeros((n_test, n_train), dtype=state["K"].dtype)
        K_test = state["K"][idcs_test][:,idcs_train]
    if "IY" in state:
        IY = state["IY"]
        IY_train = IY[idcs_train]
        IY_test = IY[idcs_test]
    # Test
    IX_test = IX[idcs_test]
    if log:
        log << "n_samples:" << n_samples << log.endl
        log << "n_train:" << n_train << log.endl
        log << "n_test:" << n_test << log.endl
    if "labels" in state:
        labels = state["labels"]
        labels_train = [ labels[i] for i in idcs_train ]
        labels_test = [ labels[i] for i in idcs_test ]
    # SAVE STATE
    state.register("split_test_train", options)
    state["n_train"] = n_train
    state["IX_train"] = IX_train
    if "labels" in state:
        state["labels_train"] = labels_train
        state["labels_test"] = labels_test
    state["idcs_train"] = idcs_train
    state["n_test"] = n_test
    state["IX_test"] = IX_test
    state["idcs_test"] = idcs_test
    state["mp_gamma"] = mp_gamma # Note that this gamma applies to the uncleaned dataset
    if state["has_T"]:
        state["T_train"] = T_train
        state["T_test"] = T_test
    if state["has_L"]:
        state["L_train"] = L_train
        state["L_test"] = L_test
    if state["has_K"]:
        state["K_train"] = K_train
        state["K_test"] = K_test
    if "IY" in state:
        state["IY_train"] = IY_train
        state["IY_test"] = IY_test
    if log: log.prefix = log.prefix[0:-7]
    return state

def random_forest_regression(state, options, log):
    log.prefix += '[pred] '
    verbose = False
    if "random_forest_regression" in options:
        options = options["random_forest_regression"]
    if verbose: log << log.mg << "Learn via RNDF" << log.endl
    from sklearn.ensemble import RandomForestRegressor
    if verbose: log << "Imported RNDF" << log.endl
    T_train = state["T_train"]
    T_test = state["T_test"]
    n_train = state["n_train"]
    n_test = state["n_test"]
    # FIT
    regr = RandomForestRegressor(**options)
    regr.fit(state["IX_train"], T_train)
    # PREDICT
    T_train_pred = regr.predict(state["IX_train"])
    T_test_pred = regr.predict(state["IX_test"])
    # EVALUATE ERRORS
    rmse_train = (np.sum((T_train_pred-T_train)**2)/n_train)**0.5
    rmse_test = (np.sum((T_test_pred-T_test)**2)/n_test)**0.5
    mae_train = np.sum(np.abs(T_train_pred-T_train))/n_train
    mae_test = np.sum(np.abs(T_test_pred-T_test))/n_test
    if T_train.shape[0] > 1:
        spearmanr_train = scipy.stats.spearmanr(T_train, T_train_pred).correlation
    else: spearmanr_train = np.nan
    if T_test.shape[0] > 1:
        spearmanr_test = scipy.stats.spearmanr(T_test, T_test_pred).correlation
    else: spearmanr_test = np.nan
    r2_fit = utils.r2_value_from_fit(T_test_pred, T_test)
    r2 = sklearn.metrics.r2_score(T_test, T_test_pred)

    #short = np.where(regr.feature_importances_ > 0.0)[0]
    #lmin = np.min(np.log10(regr.feature_importances_[short]))
    #lmax = np.max(np.log10(regr.feature_importances_[short]))
    #print lmin, lmax
    #for i in range(state["IX_tags"].shape[1]):
    #    barlength = int(20*(np.log10(regr.feature_importances_[i]+1e-20)-lmin)/(lmax-lmin))
    #    if barlength < 0: barlength = 0
    #    bar = barlength*"+"
    #    print "%15s %+1.4e   %s" % (state["IX_tags"][0][i], regr.feature_importances_[i], bar)
    #raw_input('...')
    # RETURN RESULTS OBJECT
    res = {
        't_train_avg': state["t_average"] if "t_average" in state else 0.0,
        'T_train_pred': T_train_pred,
        'T_test_pred': T_test_pred,
        'T_train': np.copy(state["T_train"]),
        'T_test': np.copy(state["T_test"]),
        'rmse_train': rmse_train,
        'rmse_test': rmse_test,
        'mae_train': mae_train,
        'mae_test': mae_test,
        'spearmanr_train': spearmanr_train,
        'spearmanr_test': spearmanr_test,
        'model': regr,
        'n_train': n_train,
        'n_test': n_test,
        'std_data_train': np.std(state["T_train"]),
        'std_data_test': np.std(state["T_test"]),
        'r2_fit': r2_fit,
        'r2': r2
    }
    log.prefix = log.prefix[0:-7]
    return state, res



    return state


def learn(state, options, log, verbose=False):
    log.prefix += '[pred] '
    if 'learn' in options:
        options = options['learn']
    method = options['method']
    method_options = options[method]
    if verbose: log << log.mg << "Learn via" << method << log.endl
    T_train = state["T_train"]
    T_test = state["T_test"]
    n_train = state["n_train"]
    n_test = state["n_test"]
    # FIT
    regr = learn_method_factory[method](**method_options)
    regr.fit(state["IX_train"], T_train)
    # PREDICT
    if verbose: print "Coefficients:", regr.intercept_, regr.coef_
    T_train_pred = regr.predict(state["IX_train"])
    T_test_pred = regr.predict(state["IX_test"])
    # EVALUATE ERRORS
    rmse_train = (np.sum((T_train_pred-T_train)**2)/n_train)**0.5
    rmse_test = (np.sum((T_test_pred-T_test)**2)/n_test)**0.5
    mae_train = np.sum(np.abs(T_train_pred-T_train))/n_train
    mae_test = np.sum(np.abs(T_test_pred-T_test))/n_test
    if T_train.shape[0] > 1:
        spearmanr_train = scipy.stats.spearmanr(T_train, T_train_pred).correlation
    else: spearmanr_train = np.nan
    if T_test.shape[0] > 1:
        spearmanr_test = scipy.stats.spearmanr(T_test, T_test_pred).correlation
    else: spearmanr_test = np.nan
    r2_fit = utils.r2_value_from_fit(T_test_pred, T_test)
    r2 = sklearn.metrics.r2_score(T_test, T_test_pred)
    #np.savetxt('out.learn_train.txt', np.array([T_train, T_train_pred]).T)
    #np.savetxt('out.learn_test.txt', np.array([T_test, T_test_pred]).T)
    # RETURN RESULTS OBJECT
    res = {
        't_train_avg': state["t_average"] if "t_average" in state else 0.0,
        'T_train_pred': T_train_pred,
        'T_test_pred': T_test_pred,
        'T_train': np.copy(state["T_train"]),
        'T_test': np.copy(state["T_test"]),
        'rmse_train': rmse_train,
        'rmse_test': rmse_test,
        'mae_train': mae_train,
        'mae_test': mae_test,
        'spearmanr_train': spearmanr_train,
        'spearmanr_test': spearmanr_test,
        'model': regr,
        'n_train': n_train,
        'n_test': n_test,
        'std_data_train': np.std(state["T_train"]),
        'std_data_test': np.std(state["T_test"]),
        'r2_fit': r2_fit,
        'r2': r2
    }
    log.prefix = log.prefix[0:-7]
    return state, res

def apply_parameter(options, path, value):
    target = options
    keys = path.split('/')
    for i in range(len(keys)-1):
        k = keys[i]
        target = target[k]
    target[keys[-1]] = value
    return options

def learn_optimal(state, options, log, verbose=False):
    log.prefix += '[scan] '
    log << log.mr << "WARNING <learn_optimal> is deprecated, use <scan_optimal> instead" << log.endl
    path = options['learn_optimal']['path']
    values = options['learn_optimal']['values']
    log << "Grid search: path=" << path << log.endl
    v_res = []
    out = []
    for v in values:
        options = apply_parameter(options, path, v)
        state, res = learn(state, options, log, verbose)
        log << path << ("%+1.4e" % v if type(v) in [ float, int ] else v) << "rmse_train" << "%+1.4e" % res["rmse_train"] << "rmse_test" << "%+1.4e" % res["rmse_test"] << log.endl
        out.append([v, res])
        v_res.append([v, res])
    out = sorted(out, key=lambda o: o[1]["rmse_test"])
    out[0][1]["value_res"] = v_res
    log.prefix = log.prefix[0:-7]
    return state, out[0][1]

def scan_optimal(state, options, log):
    log.prefix += '[scan] '
    path = options['scan_optimal']['path']
    values = options['scan_optimal']['values']
    fct = options['scan_optimal']['fct']
    obj = options['scan_optimal']['obj']
    log << "Grid search: path=" << path << log.endl
    v_res = []
    out = []
    for v in values:
        options = apply_parameter(options, path, v)
        state, res = fct(state, options, log)
        log << path << ("%+1.4e" % v if type(v) in [ float, int ] else v ) << obj << "%+1.4e" % res[obj] << log.endl
        out.append([v, res])
        v_res.append([v, res])
    out = sorted(out, key=lambda o: o[1][obj])
    out[0][1]["value_res"] = v_res
    log << "Minimum at" << ("%+1.4e" % out[0][0] if type(out[0][0]) in [ float, int ] else out[0][0] ) << obj << "%+1.4e" % out[0][1][obj] << log.endl
    log << "Maximum at" << ("%+1.4e" % out[-1][0] if type(out[-1][0]) in [ float, int ] else out[-1][0] ) << obj << "%+1.4e" % out[-1][1][obj] << log.endl
    log.prefix = log.prefix[0:-7]
    return state, out[0][1]

def learn_repeat_aggregate(state, options, log, verbose=False):
    log.prefix += '[aggr] '
    # Options
    n_reps = options["learn_repeat_aggregate"]["repetitions"]
    model = options["learn_repeat_aggregate"]["pipe"]
    assert n_reps > 0
    # Execute model n_rep times
    out = []
    for i in range(n_reps):
        state_clone = state.clone()
        model_out = model.execute(state_clone, options, log)
        out.append(model_out)
    # Aggregate
    res_agg = {}
    n_channels = len(out[0])
    for c in range(n_channels):
        tag = out[0][c].tag
        T_train = []
        T_test = []
        T_train_pred = []       
        T_test_pred = []
        for i in range(n_reps):
            r = out[i][c].res # repetition->channel->result
            T_train = T_train + list(r["T_train"])
            T_test = T_test + list(r["T_test"])
            T_train_pred = T_train_pred + list(r["T_train_pred"])
            T_test_pred = T_test_pred + list(r["T_test_pred"])
            n_train = r["T_train"].shape[0]
            n_test = r["T_test"].shape[0]
        T_train = np.array(T_train)
        T_test = np.array(T_test)
        T_train_pred = np.array(T_train_pred)
        T_test_pred = np.array(T_test_pred)
        rmse_train = np.average((T_train-T_train_pred)**2)**0.5
        rmse_test = np.average((T_test-T_test_pred)**2)**0.5
        mae_train = np.sum(np.abs(T_train_pred-T_train))/n_train
        mae_test = np.sum(np.abs(T_test_pred-T_test))/n_test
        res = soap.soapy.momo.ExtendableNamespace()
        res.tag = tag
        res.res = { 
            "T_train": T_train, 
            "T_test": T_test, 
            "T_train_pred": T_train_pred, 
            "T_test_pred": T_test_pred, 
            "rmse_train": rmse_train, 
            "rmse_test": rmse_test,
            "mae_train": mae_train, 
            "mae_test": mae_test,
            'std_data_train': np.std(T_train),
            'std_data_test': np.std(T_test),
            'n_train': n_train,
            'n_test': n_test,
            'n_train_pred_total': T_train.shape[0],
            'n_test_pred_total': T_test.shape[0]
        }
        assert tag not in res_agg
        res_agg[tag] = res.res
    log.prefix = log.prefix[0:-7]
    return state, res_agg

# =========
# FACTORIES
# =========

load_configs_factory = {
    'load_extended_xyz': load_configs_extended_xyz
}

compute_descriptor_factory = {
    'descriptor_morgan': compute_descriptor_morgan
}

learn_method_factory = {
    'linear': linear_model.LinearRegression,
    'ridge': linear_model.Ridge,
    'lasso': linear_model.Lasso
}

# ==========
# EVALUATION
# ==========

def compute_auc_threshold(
        class_list, 
        sigma_list,
        is_positive = lambda class_: class_ > 0,
        appears_positive = lambda sigma, threshold: sigma > threshold,
        invert=False,
        ds=None,
        outfile=''):
    min_s = min(sigma_list)
    max_s = max(sigma_list)
    norm_s = max([abs(min_s), abs(max_s)])
    if ds == None:
        ds = 0.01*(max_s - min_s)
    fp_tp_fn_tn_threshold = []
    threshold = max_s + ds
    while threshold >= min_s - ds:
        n_p = 0 # true
        n_n = 0 # false
        n_tp = 0 # true +
        n_fp = 0 # false +
        n_tn = 0 # true -
        n_fn = 0 # false -
        for c,s in zip(class_list, sigma_list):
            is_p = is_positive(c)
            ap_p = appears_positive(s, threshold)
            if invert: ap_p = not ap_p
            if is_p: 
                n_p += 1
            else:
                n_n += 1
            if is_p and ap_p:
                n_tp += 1
            elif is_p and not ap_p:
                n_fn += 1
            elif not is_p and ap_p:
                n_fp += 1
            elif not is_p and not ap_p:
                n_tn += 1
            else: assert False
        if n_p > 0:
            f_tp = float(n_tp)/n_p # p(+|+)
            f_fn = float(n_fn)/n_p
        else:
            f_tp = 1.0
            f_fn = 0.0
        if n_n > 0:
            f_fp = float(n_fp)/n_n # p(+|-)
            f_tn = float(n_tn)/n_n # p(-|-)
        else:
            f_fp = 0.0
            f_tn = 1.0
        threshold_norm = threshold/norm_s
        threshold_sym = threshold/max_s if threshold > 0 else -threshold/min_s
        mcc = matthews_corr_coeff(n_n*f_fp, n_p*f_tp, n_p*f_fn, n_n*f_tn)
        fp_tp_fn_tn_threshold.append([
            f_fp, f_tp, f_fn, f_tn, n_p, n_n, threshold, threshold_norm, threshold_sym, mcc])
        threshold -= ds
    # Output receiver operating characteristic
    mccs = []
    accs = []
    precs = []
    recs = []
    for fp, tp, fn, tn, n_p, n_n, t, norm_t, norm_t_sym, mcc in fp_tp_fn_tn_threshold:
        #if fp < 1e-10 or tp < 1e-10 or fn < 1e-10 or tn < 1e-10:
        #    continue
        # TODO Choose: weight both classes equally for mcc, acc?
        mcc = matthews_corr_coeff(n_n*fp, n_p*tp, n_p*fn, n_n*tn)
        if np.isnan(mcc): 
            continue
        acc = float(n_p*tp + n_n*tn)/(n_p+n_n)
        prec = float(n_p*tp)/(n_p*tp+n_n*fp)
        rec = float(n_p*tp)/(n_p*tp+n_p*fn)

        #mcc = matthews_corr_coeff(fp, tp, fn, tn)
        #acc = 0.5*float(tp+tn)
        #prec = float(tp)/(tp+fp)
        #rec = float(tp)/(tp+fn)

        mccs.append([mcc, t])
        accs.append([acc, t])
        precs.append([prec, t])
        recs.append([rec, t])

    if len(mccs):
        mcc_max = sorted(mccs, key = lambda m: -m[0])[0]
        acc_max = sorted(accs, key = lambda m: -m[0])[0]
        prec_max = sorted(precs, key = lambda m: -m[0])[0]
        rec_max = sorted(recs, key = lambda m: -m[0])[0]
    else:
        soap.soapy.momo.osio << soap.soapy.momo.osio.mr << "WARNING Could not compute MCC" << soap.soapy.momo.osio.endl
        mcc_max = [None, None]
        acc_max = [None, None]
        prec_max = [None, None]
        rec_max = [None, None]

    if outfile:
        ofs = open(outfile, 'w')
        for fp, tp, fn, tn, n_p, n_n, t, norm_t, norm_t_sym, mcc in fp_tp_fn_tn_threshold:
            ofs.write('%1.7f %1.7f %+1.7f %+1.7f %+1.7f %+1.7f %+1.7f, %+1.2f\n' % (
                fp, tp, fn, tn, t, norm_t, norm_t_sym, mcc))
        ofs.close()
    # Compute AUC
    auc = 0.
    for i in range(len(fp_tp_fn_tn_threshold)-1):
        x0 = fp_tp_fn_tn_threshold[i][0]
        x1 = fp_tp_fn_tn_threshold[i+1][0]
        y = fp_tp_fn_tn_threshold[i][1]
        y1 = fp_tp_fn_tn_threshold[i+1][1]
        dA = y*(x1-x0) + 0.5*(y1-y)*(x1-x0)
        if invert: dA *= -1
        auc += dA
    # Done.
    return auc, mcc_max, acc_max, prec_max, rec_max, fp_tp_fn_tn_threshold

def matthews_corr_coeff(fp, tp, fn, tn):
    return (tp*tn - fp*fn)/np.sqrt( (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn) )

def rms_error(y, y_ref):
    n = y_ref.shape[0]
    dy = y-y_ref
    dy2 = dy**2
    rms = np.sum(dy2/n)**0.5
    return rms

def mae_error(y, y_ref):
    n = y_ref.shape[0]
    dy = y-y_ref
    dyabs = np.abs(y-y_ref)
    mae = np.sum(dyabs)/n
    return mae

# =======
# KERNELS
# =======

def kernel_svm_binary(state, options, log):
    assert state["has_K"]
    K_train = state["K_train"]
    K_test = state["K_test"]
    L_train = state["L_train"]
    L_test = state["L_test"]
    if "xi" in options["kernel_svm"]:
        xi = options["kernel_svm"]["xi"]
    else:
        xi = 1.
    if "class_weight" in options:
        class_weight = options["class-weight"]
    else:
        class_weight = None
    if "C" in options["kernel_svm"]:
        C = options["kernel_svm"]["C"]
    else:
        C = 1.
    #mode = options["kernel_svm"]["mode"]
    # Create target reference for multi-class threshold scoring
    # E.g., class idx = 2 => y = [ -1, -1, +1 ]
    n_classes = len(set(list(L_train)))
    y_true = L_test
    # Fit
    from sklearn import svm # TODO Set SVC options
    if class_weight == None:
        class_weight = 'balanced'
        #log << "Using class weights" << class_weight << log.endl
    clf = svm.SVC(
        kernel='precomputed',
        class_weight=class_weight,#'balanced', # TODO This can make a large difference!
        C=C)
    clf.fit(K_train**xi, L_train)
    if K_test.shape[0] > 0:
        # Predict
        y_score = clf.decision_function(K_test**xi)
        y_pred = clf.predict(K_test**xi)
        # Compute AUC
        auc_out, mcc_out, acc_out, prec_out, rec_out, res = compute_auc_threshold(
            class_list=y_true, 
            sigma_list=y_score,
            is_positive = lambda class_: class_ > 0,
            appears_positive = lambda sigma, threshold: sigma > threshold,
            invert=False,
            ds=0.001,
            outfile=None) #'roc.tab') #'roc_class-%d_xi-%1.0f.tab' % (i, xi))
        #print auc_out
        #print mcc_out
        #auc = sklearn.metrics.roc_auc_score(y_true, y_score)
        #mcc = sklearn.metrics.matthews_corrcoef(y_true, y_pred)
        #auc_out = auc
        #mcc_out = [ mcc, 0.0 ]
    else:
        log << log.my << "WARNING No samples in test set, skip predictions" << log.endl
        y_score = np.array([])
        auc_out = -1.0
        mcc_out = -1.0
    #print "AUC", auc_out, mcc_out
    res = {
        'clf': clf,
        'auc': auc_out,
        'mcc': mcc_out,
        'n_train': L_train.shape[0],
        'n_test': L_test.shape[0],
        'n_classes': n_classes,
        'L_test_score': y_score,
        'L_test': y_true,
        'idcs_test': state["idcs_test"],
        'user_var_0': state["user_var_0"] if "user_var_0" in state else None  # TODO Feed from options
    }
    return state, res

def kernel_svm(state, options, log):
    assert state["has_K"]
    K_train = state["K_train"]
    K_test = state["K_test"]
    L_train = state["L_train"]
    L_test = state["L_test"]
    if "xi" in options["kernel_svm"]:
        xi = options["kernel_svm"]["xi"]
    else:
        xi = 1.
    if "class_weight" in options:
        class_weight = options["class-weight"]
    else:
        class_weight = None
    mode = options["kernel_svm"]["mode"]
    # Create target reference for multi-class threshold scoring
    # E.g., class idx = 2 => y = [ -1, -1, +1 ]
    n_classes = len(set(list(L_train)))
    y_true = np.zeros((L_test.shape[0], n_classes))
    y_true = y_true - 1
    for i in range(n_classes):
        idcs_i = np.where(L_test == i)
        y_true[idcs_i,i] = +1
    # Fit
    from sklearn import svm # TODO Set SVC options
    if mode == 'std':
        if class_weight == None:
            class_weight = {}
            for i in range(n_classes):
                class_weight[i] = 1.
            log << "Using class weights" << class_weight << log.endl
        clf = svm.SVC(
            kernel='precomputed',
            decision_function_shape='ovr', 
            class_weight=class_weight,#'balanced', # TODO This can make a large difference!
            C=1)
    elif mode == 'ovr':
        log << "Using OneVsRestClassifier object" << log.endl
        from sklearn.multiclass import OneVsRestClassifier
        if class_weight == None:
            class_weight = {0: 1., 1: 1.}
        clf = OneVsRestClassifier(
            svm.SVC(
                kernel='precomputed', 
                class_weight=class_weight, 
                probability=True, 
                random_state=987131, 
                decision_function_shape='ovr'))
    clf.fit(K_train**xi, L_train)
    # Predict
    y_score = clf.decision_function(K_test**xi)
    # Compute AUC
    map_class_auc = {}
    map_class_mcc = {}
    for i in range(n_classes):
        auc_out, mcc_out, acc_out, prec_out, rec_out, res = compute_auc_threshold(
            class_list=y_true[:, i], 
            sigma_list=y_score[:, i],
            is_positive = lambda class_: class_ > 0,
            appears_positive = lambda sigma, threshold: sigma > threshold,
            invert=False,
            ds=0.001,
            outfile='roc_class-%d.tab' % i) #'roc_class-%d_xi-%1.0f.tab' % (i, xi))
        print "AUC", i, auc_out, mcc_out
        map_class_auc[i] = auc_out
        map_class_mcc[i] = mcc_out[0]
    res = {
        'auc': map_class_auc,
        'mcc': map_class_mcc,
        'n_classes': n_classes,
        'L_test_score': y_score,
        'L_test': y_true,
        'idcs_test': state["idcs_test"],
        'user_var_0': state["user_var_0"] if "user_var_0" in state else None  # TODO Feed from options
    }
    return state, res

def kernel_rr(state, options, log):
    K_train = state["K_train"]
    K_test = state["K_test"]
    T_train = state["T_train"]
    T_test = state["T_test"]
    lreg = options["kernel_rr"]["lreg"]
    if "xi" in options["kernel_rr"]:
        xi = options["kernel_rr"]["xi"]
    else:
        xi = 1.0
    from sklearn.kernel_ridge import KernelRidge
    krrbox = KernelRidge(
        alpha=lreg,
        kernel='precomputed')
    krrbox.fit(K_train**xi, T_train)
    # Predict
    y_train = krrbox.predict(K_train**xi)
    y_test = krrbox.predict(K_test**xi)
    # Evaluate
    rmse_train = rms_error(y_train, T_train)
    rmse_test = rms_error(y_test, T_test)
    mae_train = np.sum(np.abs(y_train-T_train))/y_train.shape[0]
    mae_test = np.sum(np.abs(y_test-T_test))/y_test.shape[0]
    if y_train.shape[0] > 1:
        spearmanr_train = scipy.stats.spearmanr(y_train, T_train).correlation
    else: spearmanr_train = np.nan
    if y_test.shape[0] > 1:
        spearmanr_test = scipy.stats.spearmanr(y_test, T_test).correlation
    else: spearmanr_test = np.nan
    # Store
    out_train = np.array([y_train, T_train]).T
    out_test = np.array([y_test, T_test]).T
    r2_fit = utils.r2_value_from_fit(y_test, T_test)
    r2 = sklearn.metrics.r2_score(T_test, y_test)
    res = {
        'krr': krrbox,
        'model': krrbox,
        't_train_avg': state["t_average"] if "t_average" in state else 0.0,
        'T_train_pred': y_train,
        'T_test_pred': y_test,
        'T_train': np.copy(state["T_train"]),
        'T_test': np.copy(state["T_test"]),
        'rmse_train': rmse_train,
        'rmse_test': rmse_test,
        'mae_train': mae_train,
        'mae_test': mae_test,
        'spearmanr_train': spearmanr_train,
        'spearmanr_test': spearmanr_test,
        'std_data_train': np.std(state["T_train"]),
        'std_data_test': np.std(state["T_test"]),
        'n_train': y_train.shape[0],
        'n_test': y_test.shape[0],
        'r2_fit': r2_fit,
        'r2': r2
    }
    return state, res

def kernel_dot(IX_train, IX_test, options):
    xi = options["xi"]
    normalise = options["normalise"]
    n_dim = IX_train.shape[1]
    if normalise:
        norm_train = np.sum(IX_train*IX_train, axis=1)**0.5
        norm_train = np.tile(norm_train, (n_dim,1)).T
        norm_test = np.sum(IX_test*IX_test, axis=1)**0.5
        norm_test = np.tile(norm_test, (n_dim,1)).T
        #IX_train_norm = IX_train / norm_train
        #IX_test_norm = IX_test / norm_test
        IX_train_norm = utils.div0(IX_train, norm_train)
        IX_test_norm  = utils.div0(IX_test, norm_test)
    else:
        IX_train_norm = IX_train
        IX_test_norm = IX_test
    return IX_train_norm.dot(IX_train_norm.T)**xi, IX_test_norm.dot(IX_train_norm.T)**xi

def kernel_dot_tf(IX_train, IX_test, options):
    #print IX_train
    #print np.min(IX_train), np.average(IX_train), np.max(IX_train)
    K_train, K_cross = kernel_dot(IX_train, IX_test, { "xi": 1, "normalise": options["normalise"] })
    K_train = options["tf"](K_train)
    K_cross = options["tf"](K_cross)
    #print K_train
    #print np.min(K_train), np.average(K_train), np.max(K_train)
    #raw_input('...')
    return K_train, K_cross

def shift_kernel(state, options, log):
    state["K_train"] = 0.5*(1. + state["K_train"])
    state["K_test"] = 0.5*(1. + state["K_test"])
    return state

def compute_kernel_train_test(state, options, log):
    log.prefix += '[kern] '
    if "compute_kernel_train_test" in options:
        options = options["compute_kernel_train_test"]
    log << log.mg << "Computing kernel" << log.endl
    kernel_method = options["type"]
    suboptions = options[kernel_method]
    kernel_fct = kernel_method_factory[kernel_method]
    log << "Kernel type: '%s'" % kernel_method << log.endl
    K_train, K_test = kernel_fct(state["IX_train"], state["IX_test"], suboptions)
    log << "Kernel computation complete" << log.endl
    state.register("compute_kernel_train_test", options)
    if state["has_K"]:
        log << log.my << "WARNING Overwriting existing kernel" << log.endl
    state["has_K"] = True
    state["K_train"] = K_train
    state["K_test"] = K_test
    log.prefix = log.prefix[0:-7]
    return state

kernel_method_factory = {
    'dot': kernel_dot,
    'dot-tf': kernel_dot_tf
}

if __name__ == "__main__":

    # Command-line options
    log = soap.soapy.momo.osio
    log.Connect()
    log.AddArg('n_procs', typ=int, default=1, help="Number of processors")
    log.AddArg('mp_kernel_block_size', typ=int, default=30, help="Linear block size for kernel computation")
    log.AddArg('folder', typ=str, help="Data folder as execution target")
    log.AddArg('options', typ=str, help="Options file (json)")
    log.AddArg('select', typ=int, help="Number of items to select from dataset")
    log.AddArg('learn', typ=float, help="Training set fraction")
    log.AddArg('tag', typ=str, help="Tag to label hdf5 file")
    cmdline_options = log.Parse()
    json_options = soap.soapy.util.json_load_utf8(open(cmdline_options.options))

    # Run
    log.cd(cmdline_options.folder)
    soap.silence()
    run(log=log, cmdline_options=cmdline_options, json_options=json_options)
    log.root()
