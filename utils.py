import numpy as np
import copy
import json
from external import soap
log = soap.soapy.momo.osio

def generate_orthonormal_vectors(N, d, seed=None): # number of vecs N, dimension d
    # Returns matrix, rows correspond to orthonormal vectors
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

def r2_value_from_fit(x, y):
    import scipy.stats
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
    return r_value

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

def color_hex_to_int(c):
    return int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)

def color_int_to_hex(r, g, b):
    return "#%02x%02x%02x" % (r, g, b)

def color_interpolate(c1, c2, f):
    r1, g1, b1 = color_hex_to_int(c1)
    r2, g2, b2 = color_hex_to_int(c2)
    r = int(r1 + (r2-r1)*f + 0.5)
    g = int(g1 + (g2-g1)*f + 0.5)
    b = int(b1 + (b2-b1)*f + 0.5)
    return color_int_to_hex(r, g, b)

def color_palette(n_samples, colors, fractions):
    colors_out = []
    sample_fractions = [ float(i)/(n_samples-1) for i in range(n_samples) ]
    cidx = 1
    for f_sample in sample_fractions:
        if fractions[cidx] < f_sample: cidx += 1
        c1 = colors[cidx-1]
        c2 = colors[cidx]
        f12 = (fractions[cidx] - f_sample)/(fractions[cidx]-fractions[cidx-1])
        colors_out.append(color_interpolate(c1, c2, f12))
    return colors_out

def plot_gp_semilogx(mat, x, y, 
        lt='-', lw=1.0, lcol='#333333', 
        mt='o', ms=3.0, mcol='#999999', medgecol='#333333', mlw=2.0, 
        n=100, sigma=3, reg=0.001, 
        xerr=None, yerr=None):
    x_gp, y_gp = rmt.nonlinear.gp_interpolate_1d(np.log(x), y, n, sigma, reg)
    mat.semilogx(np.exp(x_gp), y_gp, lt, markersize=0, linewidth=lw, color=lcol)
    mat.semilogx(x, y, mt, markersize=ms, linewidth=lw, color=mcol, markeredgecolor=medgecol, markeredgewidth=mlw)
    if type(xerr) != type(None) or type(yerr) != type(None):
        mat.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="none", ecolor=lcol, elinewidth=0.5*lw)
    return mat

def plot_loglog(mat, x, y, 
        lt=None, lw=1.0, lcol=None, 
        mt=None, ms=3.0, mcol='#999999', medgecol='#333333', mlw=2.0, 
        n=None, sigma=None, reg=None):
    mat.loglog(x, y, mt, markersize=ms, linewidth=lw, color=mcol, markeredgecolor=medgecol, markeredgewidth=mlw)
    return mat

def plot_scatter(mat, x, y, 
        lt='-', lw=1.0, lcol='#333333', 
        mt='o', ms=3.0, mcol='#999999', medgecol='#333333', mlw=2.0, 
        n=100, sigma=3, reg=0.001, 
        xerr=None, yerr=None):
    mat.scatter(x, y, s=ms, marker=mt, linewidths=mlw, c=mcol, edgecolors=medgecol)
    return mat

def enable_latex(mpl, font_family='serif'):
    mpl.rcParams['text.usetex'] = True
    if font_family == 'sans-serif':
        mpl.rcParams['text.latex.preamble'] = [r'\usepackage[cm]{sfmath}']
    mpl.rcParams['font.family'] = font_family
    mpl.rcParams['font.sans-serif'] = 'cm'
    return

def mpl_eq(raw):
    return r"$\displaystyle "+raw+r" $"

def jsonize(records, delete_keys=[]):
    for r in records:
        for key in r:
            if key in delete_keys: r[key] = None
            elif type(r[key]) in [ list, int, float, str, unicode ]: continue
            elif type(r[key]) == np.ndarray:
                r[key] = r[key].tolist()
            elif type(r[key]) == np.float64:
                r[key] = float(r[key])
            else:
                r[key] = str(type(r[key]))
    return records

def mpl_layout():
    # Location examples
    top_left     = [ 0.1, 0.5, 0.3, 0.3 ]
    cb1          = [ 0.42, 0.5, 0.01, 0.3 ]
    top_right    = [ 0.5, 0.5, 0.3, 0.3 ]
    cb2          = [ 0.82, 0.5, 0.01, 0.3 ]
    bottom_left  = [ 0.1, 0.1, 0.3, 0.3 ]
    cb3          = [ 0.42, 0.1, 0.01, 0.3 ]
    bottom_right = [ 0.5, 0.1, 0.3, 0.3 ]
    cb4          = [ 0.82, 0.1, 0.01, 0.3 ]
    # Colormap example
    from matplotlib.colors import LinearSegmentedColormap
    colors = ["#ff0066", "#ffffff", "#0066ff"]
    colors.reverse()
    cm = LinearSegmentedColormap.from_list(
        "f06", colors, N=100)
    return
    
def get_size(obj, seen=None):
    # from stackexchange.com
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size





