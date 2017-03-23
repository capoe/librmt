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
import copy
# Internal libraries
import ising as ising
import nonlinear as nonlinear
from state import State
# External libraries
from sklearn import linear_model
from rdkit.Chem import AllChem as chem
try:
    #import soap
    raise ImportError
except ImportError:
    from external import soap
try:
    #import ase.io as ase_io
    raise ImportError
except ImportError:
    import external.ase.io as ase_io

# ==========
# CONFIG I/O
# ==========

def load_configs(state, options, log):
    if "load_configs" in options:
        options = options["load_configs"]
    method = options['load_method']
    return load_configs_factory[method](state, options[method], log)

def load_configs_extended_xyz(state, options, log):
    log << log.mg << "Load configurations <load_configs_extended_xyz> ..." << log.endl
    xyz_file = options["xyz_file"]
    n_select = options["n_select"]
    subsample_method = options["subsample_method"]
    filter_types = options["filter_types"]
    types = options["types"]
    # READ & FILTER
    configs_A = read_filter_configs(
        xyz_file,
        index=':',
        filter_types=filter_types,
        types=types,
        do_remove_duplicates=True,
        key=lambda c: c.info['label'],
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

# =======
# TARGETS
# =======

def load_targets(state, options, log):
    if 'load_targets' in options:
        options = options['load_targets']
    log << log.mg << "Load targets" << log.endl
    key = options['target_key']
    fct = options['target_conv_fct']
    T = np.array([ fct(config.info[key]) for config in state["configs"] ])
    log << "<t>" << np.average(T) << log.endl
    log << "sqrt<dt^2>" << np.std(T) << log.endl
    state.register("load_targets", options)
    state["T"] = T
    return state

def normalise_targets(state, options, log):
    if 'normalise_targets' in options:
        options = options['normalise_targets']
    log << log.mg << "Normalise targets" << log.endl
    centre = options['centre']
    fct = options['target_conv_fct']
    T_train = state["T_train"]
    T_test = state["T_test"]
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
    log << "<t_train>" << np.average(T_train) << np.std(T_train) << log.endl
    log << "<t_test>" << np.average(T_test) << np.std(T_test) << log.endl
    log << "t_average" << T_train_avg << log.endl
    return state

# ===========
# DESCRIPTORS
# ===========

def compute_descriptor(state, options, log):
    if "compute_descriptor" in options:
        options = options["compute_descriptor"]
    method = options["descriptor_method"]
    return compute_descriptor_factory[method](state, options[method], log)

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
    smiles = [ config.info['smiles'] for config in configs ]
    mols = [chem.MolFromSmiles(c) for c in smiles]
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
    n_dim = state["n_dim"]
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
    log << "New descriptor dimension:" << n_dim_full << log.endl
    return state

# =============
# RM Operations
# =============

def feature_select(state, options, log):
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
    return state

def clean_descriptor_matrix(state, options, log):
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
    return state

def clean_descriptor_pca(state, options, log):
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
    np.savetxt('out.pca_hist.txt', np.array([ bin_centres, hist, mp_sample, hist_signal ]).T)
    np.savetxt('out.pca_eigv.txt', L.diagonal())
    # TRANSFORM INTO PCA SPACE
    log << log.mg << "Project onto signal PCs" << log.endl
    IZ_train = div0(IX_train - X_mean, X_std)
    IZ_test = div0(IX_test - X_mean, X_std)
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
    return state

def dist_mp(x, gamma):
    # gamma = #dim / #samples
    l, u = dist_mp_bounds(gamma)
    if x <= l or x >= u:
        return 0.
    else:
        return ( (u - x)*(x - l) )**0.5 / (2*np.pi*gamma*x)

def dist_mp_bounds(gamma):
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

def div0(a, b):
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide(a, b)
        c[~np.isfinite(c)] = 0
        return c

def normalize_descriptor_zscore(IX, ddof_=1):
    # Rows of x are individual observations
    mu = np.mean(IX, axis=0)
    muTile = np.tile(mu, (IX.shape[0],1))
    std = np.std(IX, axis=0, ddof=ddof_)
    stdTile = np.tile(std, (IX.shape[0], 1))
    IZ = div0(IX - muTile, stdTile)
    return IZ, mu, std

def pca_compute(IX, log=None, norm_div_std=True, norm_sub_mean=True, ddof=1, eps=0.0):
    """
    To check result consider:
    IX_norm_pca.T.dot(IX_norm_pca)/IX_norm_pca.shape[0]
    """
    # Normalize: mean, std
    if log: log << "PCA: Normalize ..." << log.endl
    X_mean = np.mean(IX, axis=0)
    X_std = np.std(IX, axis=0, ddof=1)
    IX_norm = IX
    if norm_sub_mean:
        IX_norm = IX - X_mean
    else:
        X_mean = 0.0
    if norm_div_std:
        #IX_norm = IX_norm/(X_std+eps)
        IX_norm = div0(IX_norm, X_std+eps)
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
    if 'split_test_train' in options:
        options = options['split_test_train']
    log << log.mg << "Split onto training and test set" << log.endl
    # Read options
    f_train = options["f_train"]
    subsample_method = options["method"]
    # Calc n_train, n_test
    IX = state["IX"]
    T = state["T"]
    labels = state["labels"]
    n_samples = state["n_samples"]
    n_train = int(f_train*n_samples+0.5)
    n_test = n_samples - n_train
    idcs_train, idcs_test = soap.soapy.learn.subsample_array(
        np.arange(0, n_samples).tolist(), n_select=n_train, method=subsample_method, stride_shift=0)
    # Train
    n_train = len(idcs_train)
    IX_train = IX[idcs_train]
    T_train = T[idcs_train]
    labels_train = [ labels[i] for i in idcs_train ]
    mp_gamma = float(IX_train.shape[1])/n_train
    # Test
    n_test = len(idcs_test)
    IX_test = IX[idcs_test]
    T_test = T[idcs_test]
    labels_test = [ labels[i] for i in idcs_test ]
    log << "n_samples:" << n_samples << log.endl
    log << "n_train:" << n_train << log.endl
    log << "n_test:" << n_test << log.endl
    # SAVE STATE
    state.register("split_test_train", options)
    state["n_train"] = n_train
    state["IX_train"] = IX_train
    state["T_train"] = T_train
    state["labels_train"] = labels_train
    state["n_test"] = n_test
    state["IX_test"] = IX_test
    state["T_test"] = T_test
    state["labels_test"] = labels_test
    state["mp_gamma"] = mp_gamma # Note that this gamma applies to the uncleaned dataset
    return state

def learn(state, options, log, verbose=False):
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
    np.savetxt('out.learn_train.txt', np.array([T_train, T_train_pred]).T)
    np.savetxt('out.learn_test.txt', np.array([T_test, T_test_pred]).T)
    # RETURN RESULTS OBJECT
    res = {
        'T_train_pred': T_train_pred,
        'T_test_pred': T_test_pred,
        'rmse_train': rmse_train,
        'rmse_test': rmse_test,
        'model': regr
    }
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
    path = options['learn_optimal']['path']
    values = options['learn_optimal']['values']
    log << "Grid learning: " << path << values << log.endl
    v_res = []
    out = []
    for v in values:
        options = apply_parameter(options, path, v)
        state, res = learn(state, options, log, verbose)
        log << path << v << "rmse_train" << res["rmse_train"] << "rmse_test" << res["rmse_test"] << log.endl
        out.append([v, res])
        v_res.append([v, res])
    out = sorted(out, key=lambda o: o[1]["rmse_test"])
    out[0][1]["value_res"] = v_res
    return state, out[0][1]

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
