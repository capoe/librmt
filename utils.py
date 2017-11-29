import numpy as np
import copy

def generate_orthonormal_vectors(N, d, seed=None): # number of vecs N, dimension d
    assert N <= d
    if seed != None:
        np.random.seed(seed)
    vecs = []
    for i in range(N):
        v = np.random.uniform(-1., 1., size=d)
        vecs.append(v/np.dot(v,v)**0.5)
    vecs_ortho = []
    for v in vecs:
        for u in vecs_ortho:
            v = v - np.dot(u,v)*u
        vecs_ortho.append(v/np.dot(v,v)**0.5)
    vecs_ortho = np.array(vecs_ortho)
    return vecs_ortho

def div0(a, b):
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide(a, b)
        c[~np.isfinite(c)] = 0
        return c

def binomial_coeff(n, k):
    b = 1.
    for i in range(1, k+1):
        b *= (n+1-i)/i
    return b

def zscore(IX):
    X_mean = np.mean(IX, axis=0)
    X_std = np.std(IX, axis=0, ddof=1)
    return div0(IX-X_mean, X_std), X_mean, X_std

def apply_zscore(IX, X_mean, X_std):
    return div0(IX-X_mean, X_std)

def dict_lookup_path(dictionary, path):
    path = path.split('/')
    v = dictionary
    for p in path:
        v = v[p]
    return v

def dict_set_path(dictionary, path, value):
    path = path.split('/')
    v = dictionary
    for p in path[:-1]:
        v = v[p]
    v[path[-1]] = value
    return

def dict_compile(options, fields, mode):
    if mode == "combinatorial":
        return dict_compile_combinatorial(options, fields)
    elif mode == "linear":
        return dict_compile_linear(options, fields)
    else: raise NotImplementedError(mode)

def dict_compile_combinatorial(options, fields):
    options_array = [ options ]
    for scan in fields:
        #log << log.mg << scan << log.endl
        path = scan[0]
        values = scan[1]
        options_array_out = []
        for options in options_array:
            for v in values:
                options_mod = copy.deepcopy(options)
                target = dict_set_path(options_mod, path, v)
                options_array_out.append(options_mod)
        options_array = options_array_out
    return options_array

def dict_compile_linear(options, fields):
    # fields = [
    #    ["path/to/parameter", [ p_opt_0, p_opt_1, p_opt_2 ]],
    #    [ ...],
    #    ...
    # ]
    options_array = []
    n_combos = len(fields[0][1])
    for i in range(n_combos):
        options_mod = copy.deepcopy(options)
        for field in fields:
            dict_set_path(options_mod, field[0], field[1][i])
        options_array.append(options_mod)
    return options_array

