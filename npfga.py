#! /usr/bin/env python
import numpy as np
import json
import utils
import sys
import scipy.optimize
import scipy.stats
import pickle
import math
log = utils.log

# ================
# NOTES AND TODO's
# ================
# - Add 'maybe_zero' flag to features to bypass division by such quantities

# UNITS
# - IR INTENSITY   1 Debye2-angstrom-2-amu-1 (IR intensity unit) = 42.2561 km-mol-1
#                 => km*mol^-1 = 1./42.2561*(0.20819434)**2 e^2*amu^-1
# - IR FREQUENCY   cm^-1 = 0.000124 eV
# - THERMAL ENERGY beta = 0.0256 eV at room temperature as we are all aware

# =========
# OPERATORS
# =========

class BOperator(object):
    def __init__(self):
        pass

class UOperator(object):
    def __init__(self):
        pass

class OIdentity(UOperator):
    def __init__(self):
        self.tag = "I"
    def check(self, f1):
        return True
    def generate(self, f1):
        assert False
    def apply(self, v1):
        return True
    def latex(self, args, priorities):
        return args[0]

class OPlus(BOperator):
    def __init__(self):
        self.tag = "+"
    def check(self, f1, f2):
        matches = (np.sum(np.abs(f1.uvec - f2.uvec)) < 1e-10)
        return matches
    def generate(self, f1, f2):
        tag = r"%s+%s" % (f1.tag, f2.tag)
        maybe_neg = (f1.maybe_neg or f2.maybe_neg)
        return FNode(idx=-1, tag=tag, uvec=f1.uvec, coeff=1.0, parent_idcs=[f1.idx, f2.idx], op=self, maybe_neg=maybe_neg)
    def apply(self, v1, v2):
        return v1+v2
    def latex(self, args, priorities):
        return r"%s+%s" % tuple(args)

class OMinus(BOperator):
    def __init__(self):
        self.tag = "-"
    def check(self, f1, f2):
        matches = (np.sum(np.abs(f1.uvec - f2.uvec)) < 1e-10)
        return matches
    def generate(self, f1, f2):
        tag = r"%s - %s" % (f1.tag, f2.tag)
        return FNode(idx=-1, tag=tag, uvec=f1.uvec, coeff=1.0, parent_idcs=[f1.idx, f2.idx], op=self, maybe_neg=True)
    def apply(self, v1, v2):
        return v1-v2
    def latex(self, args, priorities):
        return r"%s-%s" % tuple(args)

class OMult(BOperator):
    def __init__(self):
        self.tag = "x"
    def check(self, f1, f2):
        return True
    def generate(self, f1, f2):
        tag = r"%s %s" % (f1.tag, f2.tag)
        if (not f1.maybe_neg) and (not f2.maybe_neg):
            maybe_neg = False
        else:
            maybe_neg = True
        return FNode(idx=-1, tag=tag, uvec=f1.uvec+f2.uvec, coeff=1.0, parent_idcs=[f1.idx, f2.idx], op=self, maybe_neg=maybe_neg)
    def apply(self, v1, v2):
        return v1*v2
    def latex(self, args, priorities):
        if op_priority[self.tag] < priorities[0]:
            args[0] = "(%s)" % args[0]
        if op_priority[self.tag] < priorities[1]:
            args[1] = "(%s)" % args[1]
        return r"%s\\,%s" % tuple(args)

class ODiv(BOperator):
    def __init__(self):
        self.tag = ":"
    def check(self, f1, f2):
        return True
    def generate(self, f1, f2):
        tag = r"\\frac{%s}{%s}" % (f1.tag, f2.tag)
        maybe_neg = (f1.maybe_neg or f2.maybe_neg)
        return FNode(idx=-1, tag=tag, uvec=f1.uvec-f2.uvec, coeff=1.0, parent_idcs=[f1.idx, f2.idx], op=self, maybe_neg=maybe_neg)
    def apply(self, v1, v2):
        return v1/v2
    def latex(self, args, priorities):
        return r"\\frac{%s}{%s}" % tuple(args)

class OExp(UOperator):
    def __init__(self):
        self.tag = "exp"
    def check(self, f1):
        ok = True
        dimless = (np.sum(np.abs(f1.uvec)) < 1e-10)
        if not dimless: ok = False
        else:
            isexp = (f1.op.tag == "exp")
            if isexp: ok = False
            else:
                islog = (f1.op.tag == "log")
                if islog: ok = False
        return ok
    def generate(self, f1):
        tag = r"\\exp(%s)" % f1.tag
        return FNode(idx=-1, tag=tag, uvec=f1.uvec, coeff=1.0, parent_idcs=[f1.idx], op=self, maybe_neg=False)
    def apply(self, v1):
        return np.exp(v1)
    def latex(self, args, priorities):
        return r"\\exp(%s)" % tuple(args)

class OLog(UOperator):
    def __init__(self):
        self.tag = "log"
    def check(self, f1):
        ok = True
        dimless = (np.sum(np.abs(f1.uvec)) < 1e-10)
        if not dimless: ok = False
        else:
            isexp = (f1.op.tag == "exp")
            if isexp: ok = False
            else:
                islog = (f1.op.tag == "log")
                if islog: ok = False
        return ok
    def generate(self, f1):
        tag = r"\\ln(%s)" % f1.tag
        return FNode(idx=-1, tag=tag, uvec=f1.uvec, coeff=1.0, parent_idcs=[f1.idx], op=self, maybe_neg=True)
    def apply(self, v1):
        return np.log(v1)
    def latex(self, args, priorities):
        return r"\\ln(%s)" % tuple(args)

class OMod(UOperator):
    def __init__(self):
        self.tag = "|"
    def check(self, f1):
        return f1.maybe_neg
    def generate(self, f1):
        tag = r"|%s|" % f1.tag
        return FNode(idx=-1, tag=tag, uvec=f1.uvec, coeff=1.0, parent_idcs=[f1.idx], op=self, maybe_neg=False)
    def apply(self, v1):
        return np.abs(v1)
    def latex(self, args, priorities):
        return r"|%s|" % tuple(args)

class OSqrt(UOperator):
    def __init__(self):
        self.tag = "^1/2"
    def check(self, f1):
        ok = True
        if f1.maybe_neg: ok = False
        elif f1.op.tag == "^2": ok = False
        else: pass
        return ok
    def generate(self, f1):
        tag = r"\\sqrt{%s}" % f1.tag
        return FNode(idx=-1, tag=tag, uvec=0.5*f1.uvec, coeff=1.0, parent_idcs=[f1.idx], op=self, maybe_neg=False)
    def apply(self, v1):
        return np.sqrt(v1)
    def latex(self, args, priorities):
        return r"\\sqrt{%s}" % tuple(args)

class OInv(UOperator):
    def __init__(self):
        self.tag = "^-1"
    def check(self, f1):
        ok = True
        if f1.op.tag == "^-1": ok = False
        return ok
    def generate(self, f1):
        tag = r"(%s)^{-1}" % f1.tag
        return FNode(idx=-1, tag=tag, uvec=-f1.uvec, coeff=1.0, parent_idcs=[f1.idx], op=self, maybe_neg=f1.maybe_neg)
    def apply(self, v1):
        return 1./v1
    def latex(self, args, priorities):
        if priorities[0] > op_priority[self.tag]:
            args[0] = "(%s)" % args[0]
        return r"%s^{-1}" % tuple(args)

class O2(UOperator):
    def __init__(self):
        self.tag = "^2"
    def check(self, f1):
        ok = True
        if f1.op.tag == "^1/2": ok = False
        return ok
    def generate(self, f1):
        tag = r"(%s)^2" % f1.tag
        return FNode(idx=-1, tag=tag, uvec=2*f1.uvec, coeff=1.0, parent_idcs=[f1.idx], op=self, maybe_neg=False)
    def apply(self, v1):
        return v1**2
    def latex(self, args, priorities):
        if priorities[0] > op_priority[self.tag]:
            args[0] = "(%s)" % args[0]
        return r"%s^2" % tuple(args)

bop_map = {
    "+": OPlus(),
    "-": OMinus(),
    ":": ODiv(),
    "*": OMult()
}

uop_map = {
    "e": OExp(),
    "l": OLog(),
    "|": OMod(),
    "s": OSqrt(),
    "r": OInv(),
    "2": O2(),
}

op_priority = {
    "I":0,
    "x":1,
    ":":1,
    "+":2,
    "-":2, 
    "exp":0,
    "log":0, 
    "|":0, 
    "^1/2":1, 
    "^-1":1,
    "^2":1,
}

# ============
# CORE OBJECTS
# ============

class FNode(object):
    def __init__(self, 
            idx, 
            tag="", 
            unit=[], 
            coeff=None, 
            convert={}, 
            uvec=None, 
            parent_idcs=[], 
            op=None, 
            maybe_neg=True, 
            root=False):
        self.idx = idx
        self.root = root
        self.tag = tag if root else ""
        self.uvec = uvec
        self.parent_idcs = parent_idcs
        self.maybe_neg = maybe_neg if type(maybe_neg) == bool else { "+": False, "+-": True }[maybe_neg]
        self.op = op if op != None else OIdentity()
        self.coeff = coeff if coeff != None else 1.0
        self.has_unary_children = False
        self.selected = False
        self.confidence = -1.
        if unit != []:
            self.unit = {}
            if unit == "":
                pass
            else:
                for u in unit.split("*"):
                    self.addUnitFactor(u, convert)
        else:
            self.unit = unit
    def getDependencies(self, fnodes):
        deps = []
        if self.root: return deps
        for p in self.parent_idcs:
            deps.append(p)
            deps = deps + fnodes[p].getDependencies(fnodes)
        return deps
    def seed(self, x):
        return self.coeff*x
    def calculateTag(self, fnodes):
        if self.tag != "":
            return self.tag
        elif self.root:
            return self.tag
        else:
            self.tag = "OP_"+self.op.tag+"(%s)" % (",".join([fnodes[_].calculateTag(fnodes) for _ in self.parent_idcs]))
            return self.tag
    def calculateLatexExpr(self, fnodes, decode, format_string=r"$%s$"):
        if self.root:
            return format_string % (decode[self.tag] if self.tag in decode else self.tag)
        else:
            args = [ fnodes[i].calculateLatexExpr(
                fnodes, decode, format_string=r"%s") for i in self.parent_idcs ]
            priorities = [ op_priority[fnodes[i].op.tag] for i in self.parent_idcs ]
            return format_string % self.op.latex(args, priorities)
    def applyMatrixRecursive(self, IX, x_tag_to_idx, fnodes):
        if self.root:
            return self.coeff*IX[:, x_tag_to_idx[self.tag]]
        elif len(self.parent_idcs) == 1:
            return self.coeff*self.op.apply(
                fnodes[self.parent_idcs[0]].applyMatrixRecursive(IX, x_tag_to_idx, fnodes)
            )
        elif len(self.parent_idcs) == 2:
            return self.coeff*self.op.apply(
                fnodes[self.parent_idcs[0]].applyMatrixRecursive(IX, x_tag_to_idx, fnodes),
                fnodes[self.parent_idcs[1]].applyMatrixRecursive(IX, x_tag_to_idx, fnodes)
            )
        else: assert False
    def apply(self, features, checklist):
        if self.root:
            return self.coeff*self.op.apply(features[self.idx])
        elif len(self.parent_idcs) == 1:
            assert checklist[self.parent_idcs[0]]
            return self.coeff*self.op.apply(features[self.parent_idcs[0]])
        elif len(self.parent_idcs) == 2:
            assert checklist[self.parent_idcs[0]] and checklist[self.parent_idcs[1]]
            return self.coeff*self.op.apply(features[self.parent_idcs[0]], features[self.parent_idcs[1]])
        else: assert False
    def applyMatrix(self, fmat, checklist):
        if self.root:
            return self.coeff*self.op.apply(fmat[:, self.idx])
        elif len(self.parent_idcs) == 1:
            assert checklist[self.parent_idcs[0]]
            return self.coeff*self.op.apply(fmat[:, self.parent_idcs[0]])
        elif len(self.parent_idcs) == 2:
            assert checklist[self.parent_idcs[0]] and checklist[self.parent_idcs[1]]
            return self.coeff*self.op.apply(fmat[:,self.parent_idcs[0]], fmat[:,self.parent_idcs[1]])
        else: assert False
    def addUnitFactor(self, u, convert={}):
        if "^" in u:
            name, exponent = tuple(u.split("^"))
            exponent = float(exponent)
        else:
            name = u
            exponent = 1.0
        if name in convert:
            factor = convert[name][0]
            converted = convert[name][1]
            self.coeff = self.coeff * factor**exponent
            for u in converted.split("*"):
                self.addUnitFactor(u)
        else:
            self.unit[name] = exponent
    def vectorizeUnit(self, umap):
        unit = [ 0 for _ in range(len(umap)) ]
        for u in self.unit:
            unit[umap[u]] = self.unit[u]
        self.uvec = np.array(unit, dtype='float16')
        print "%-30s %-40s => %-40s" % (self.tag, self.unit, self.uvec)
        return
    def calculateCovSequence(self, fnodes):
        seq = [ { "idx": self.idx, "tag": self.calculateTag(fnodes), "cov": self.cov } ]
        for p in self.parent_idcs:
            seq = seq + fnodes[p].calculateCovSequence(fnodes)
        return seq
    def __str__(self):
        return "%-55s units=%-25s coeff=%+1.2e par=%10s %3s" % (
            self.tag, self.uvec, self.coeff, self.parent_idcs, "+" if not self.maybe_neg else "+/-")

class FGraph(object):
    def __init__(self, features, constants, conversions):
        self.constants = constants
        self.fnodes_in = self.createNodes(features, constants, conversions)    
        self.fnodes = []
        self.tag_to_fidx = { f.tag: idx for idx, f in enumerate(self.fnodes_in) }
        umap_keys = []
        for f in self.fnodes_in:
            umap_keys = list(set(umap_keys + f.unit.keys()))
        umap_keys = sorted(umap_keys)
        print "Have units:", umap_keys
        self.umap = { u: i for i, u in enumerate(umap_keys) }
        for f in self.fnodes_in:
            f.vectorizeUnit(self.umap)
    def createNodes(self, features, constants, conversions):
        fnodes = []
        for tag, f in features.iteritems():
            node = FNode(idx=len(fnodes), tag=tag, unit=f[0], coeff=f[2], convert=conversions, root=True, maybe_neg=True if f[1] == "+-" else "+")
            fnodes.append(node)
        for tag, f in constants.iteritems():
            node = FNode(idx=len(fnodes), tag=tag, unit=f[0], coeff=f[2], convert=conversions, root=True, maybe_neg=True if f[1] == "+-" else "+")
            fnodes.append(node)
        return fnodes
    def embedChildIdcs(self):
        for fnode in self.fnodes:
            fnode.child_idcs = []
        for fnode in self.fnodes:
            for pidx in fnode.parent_idcs:
                self.fnodes[pidx].child_idcs.append(fnode.idx)
        return
    def mapGenerations(self):
        root_nodes = filter(lambda f: f.root, self.fnodes)
        for f in self.fnodes:
            f.generation = -1
        def assign_generation_recursive(fnode, fgraph, generation=0):
            if fnode.generation <= generation:
                fnode.generation = generation
            for c in fnode.child_idcs:
                assign_generation_recursive(fgraph.fnodes[c], fgraph, generation=generation+1)
            return
        for r in root_nodes:
            assign_generation_recursive(r, self)
        map_generations = {}
        for f in self.fnodes:
            if not f.generation in map_generations: map_generations[f.generation] = []
            map_generations[f.generation].append(f)
        return map_generations
    def truncate_strict(self, fnode_idcs, keep_input_nodes=True):
        len_in = len(self.fnodes)
        keep_idcs = { i: True for i in fnode_idcs }
        for i in fnode_idcs:
            deps = self.fnodes[i].getDependencies(self.fnodes)
            for d in deps:
                if not d in keep_idcs:
                    keep_idcs[i] = False
                    break
        keep_idcs = { i: True for i in fnode_idcs if keep_idcs[i] }
        if keep_input_nodes:
            for idx, f in enumerate(self.fnodes_in):
                keep_idcs[idx] = True
        keep_idcs = sorted(keep_idcs)
        idx_old_to_idx_new = { k: kk for kk, k in enumerate(keep_idcs) }
        self.fnodes = [ self.fnodes[i] for i in keep_idcs ]
        for f in self.fnodes:
            f.parent_idcs = [ idx_old_to_idx_new[p] for p in f.parent_idcs ]
            f.idx = idx_old_to_idx_new[f.idx]
        len_out = len(self.fnodes)
        print "Reduced from %d to %d feature nodes" % (len_in, len_out)
        return
    def truncate(self, fnode_idcs, keep_input_nodes=True):
        len_in = len(self.fnodes)
        keep_idcs = { i: True for i in fnode_idcs }
        for i in fnode_idcs:
            deps = self.fnodes[i].getDependencies(self.fnodes)
            for d in deps: keep_idcs[d] = True
        if keep_input_nodes:
            for idx, f in enumerate(self.fnodes_in):
                keep_idcs[idx] = True
        keep_idcs = sorted(keep_idcs)
        idx_old_to_idx_new = { k: kk for kk, k in enumerate(keep_idcs) }
        self.fnodes = [ self.fnodes[i] for i in keep_idcs ]
        for f in self.fnodes:
            f.parent_idcs = [ idx_old_to_idx_new[p] for p in f.parent_idcs ]
            f.idx = idx_old_to_idx_new[f.idx]
        len_out = len(self.fnodes)
        print "Reduced from %d to %d feature nodes" % (len_in, len_out)
        return
    def apply(self, x_dict, verbose=False):
        x_out = np.zeros((len(self.fnodes),), dtype='float32')
        checklist = np.zeros((len(self.fnodes),), dtype='bool')
        for idx, f in enumerate(self.fnodes_in):
            xi = f.seed(x_dict[f.tag])
            x_out[idx] = xi
            checklist[idx] = True
            if verbose: print f.tag, x_dict[f.tag], xi
        for idx, f in enumerate(self.fnodes):
            if checklist[idx]: continue
            xi = f.apply(x_out, checklist)
            x_out[idx] = xi
            checklist[idx] = True
            if verbose: print f.calculateTag(self.fnodes), xi
        #assert checklist.all()
        return x_out
    def formatSeedDescriptors(self, IX, x_tags):
        # Add constants to input descriptor matrix
        c_tags = sorted([ t for t in self.constants ])
        IC_const = np.ones((IX.shape[0], len(c_tags)), dtype='float32')
        if len(c_tags):
            print "Added %d constants to descriptor matrix" % len(c_tags)
        x_tags = list(x_tags) + c_tags
        IX = np.concatenate((IX, IC_const), axis=1)
        # Index by descriptor tag
        x_tag_to_idx = { t: i for i, t, in enumerate(x_tags) }
        return IX, x_tags, x_tag_to_idx
    def applyMatrix(self, IX, x_tags, verbose=False):
        IX, x_tags, x_tag_to_idx = self.formatSeedDescriptors(IX, x_tags)
        # Allocate output array and feature checklist
        IX_out = np.zeros((IX.shape[0], len(self.fnodes)), dtype='float32')
        x_tag_to_idx = { t: i for i, t, in enumerate(x_tags) }
        checklist = np.zeros((len(self.fnodes),), dtype='bool')
        # Seed root features
        for idx, f in enumerate(self.fnodes_in):
            IX_col_i = f.seed(IX[:, x_tag_to_idx[f.tag]])
            IX_out[:,idx] = IX_col_i
            checklist[idx] = True
            if verbose: print f.tag, IX[0,x_tag_to_idx[f.tag]], IX_out[0,idx]
        # Percolate through tree
        for idx, f in enumerate(self.fnodes):
            if checklist[idx]: continue
            IX_col_i = f.applyMatrix(IX_out, checklist)
            IX_out[:,idx] = IX_col_i
            checklist[idx] = True
            if verbose: print f.calculateTag(self.fnodes), IX_out[0,idx]
        return IX_out
    def generate(self, uops, bops, depth, max_exp=3, min_exp=0.4, verbose=False, halt=False):
        if len(self.fnodes) == 0:
            print "Starting from root nodes"
            self.fnodes = [ f for f in self.fnodes_in ]
        else:
            print "Starting from root+child nodes"
        def check_in_unary(f, op):
            if f.op.tag == op.tag: return False # Do not apply same unary operator twice
            return True
        def check_in_binary(f1, f2, op):
            return True
        def check_out(f):
            ok = True
            if f.uvec.shape[0] > 0:
                if np.max(np.abs(f.uvec)) > max_exp: ok = False
                nonzero = np.where(np.abs(f.uvec) > 1e-2)[0]
                if len(nonzero) > 0:
                    p = np.min(np.abs(f.uvec[nonzero])) 
                    if p < min_exp: ok = False
            return ok
        idx_tracker = len(self.fnodes)
        for step in range(depth): 
            # Unary operators
            print "Unary, level=%d" % (step)
            f_new_list = []
            for fidx in range(len(self.fnodes)):
                if self.fnodes[fidx].has_unary_children: continue
                self.fnodes[fidx].has_unary_children = True
                for op in uops:
                    f = self.fnodes[fidx]
                    add = False
                    # Check generic, pre-op
                    if check_in_unary(f, op):
                        # Check op-specific
                        if op.check(f):
                            f_new = op.generate(f)
                            # Check generic, post-op
                            if check_out(f_new):
                                add = True
                            else: pass
                        else: pass
                    else: pass
                    if add:
                        f_new.idx = idx_tracker
                        f_new_list.append(f_new)
                        idx_tracker += 1
                        if idx_tracker % 1000 == 0: 
                            print "Op=%8d   %-40s %-8s %s" % (
                                idx_tracker, 
                                [ "%+1.1f" % _ for _ in f_new.uvec ], 
                                f_new.tag, 
                                "+-" if f_new.maybe_neg else "+")
                        if verbose: print "Generated %-30s %30s %8d %s" % (
                            f_new.tag, 
                            f_new.uvec, 
                            f_new.idx, 
                            "+-" if f_new.maybe_neg else "+")
                    else:
                        if verbose: print "Prevented %s > %s" % (op.tag, f.tag)
                        pass
            if halt: raw_input('...')
            print "... Generated %d features" % len(f_new_list)
            self.fnodes.extend(f_new_list)
            # Binary operators
            print "Binary, level=%d, len=%d" % (step, len(self.fnodes))
            f_new_list = []
            for op in bops:
                for f1idx in range(len(self.fnodes)):
                    f1 = self.fnodes[f1idx]
                    for f2idx in range(f1idx+1, len(self.fnodes)):
                        f2 = self.fnodes[f2idx]
                        add = False
                        # Check generic, pre-application
                        if check_in_binary(f1, f2, op):
                            # Check op-specific
                            if op.check(f1, f2):
                                f_new = op.generate(f1, f2)
                                # Check generic, post-application
                                if check_out(f_new):
                                    add = True
                                else: pass
                            else: pass
                        else: pass
                        if add:
                            f_new.idx = idx_tracker
                            f_new_list.append(f_new)
                            idx_tracker += 1
                            if idx_tracker % 1000 == 0: 
                                print "Op=%s %8d %8d %8d   %-40s %-8s %s" % (op.tag, f1idx, f2idx, idx_tracker, [ "%+1.1f" % _ for _ in f_new.uvec ], f_new.tag, "+-" if f_new.maybe_neg else "+")
                            if verbose: print "Generated %-50s %30s %8d %s" % (f_new.tag, f_new.uvec, f_new.idx, "+-" if f_new.maybe_neg else "+")
                        else:
                            if verbose: print "Prevented %s > %s %s" % (op.tag, f1.tag, f2.tag)
                            pass
            print "... Generated %d features" % len(f_new_list)
            self.fnodes.extend(f_new_list)
            if halt: raw_input('...')
        print "Generated a total of %d features" % (len(self.fnodes))
        return 

def initialize_fgraph(
        features_in, 
        feature_groups, 
        feature_props, 
        constants,
        conversions,
        log=utils.log):
    # COMPILE LIST OF SELECTED FEATURES FROM FEATURE GROUPS
    log << log.mg << "Select features" << log.endl
    feature_select = []
    for group in feature_groups:
        feature_select = feature_select + feature_groups[group]
    for f in feature_select:
        log << " -" << f << log.endl
    feature_select = { f: True for f in feature_select }
    # FILTER FEATURES TO COMPRISE ONLY THOSE FROM SELECTED GROUPS
    log << "Filtering input descriptor based on selection" << log.endl
    feature_tags = features_in
    feature_tags_dict = { f: True for f in feature_tags }
    for f in feature_select:
        if not f in feature_tags_dict:
            log << log.my << "WARNING '%s' not found among input features" % f << log.endl
    feature_tags = filter(lambda t: t in feature_select, feature_tags)
    # LOAD FEATURE PROPERTIES
    features = { f: feature_props[f] for f in feature_tags }
    # FEATURE GRAPH SETUP
    log << log.mg << "Feature graph setup" << log.endl
    fgraph = FGraph(features, constants, conversions)
    # GENERATE FEATURE GRAPH
    log << log.mg << "Feature graph generation" << log.endl
    return fgraph

def generate_fgraph(fgraph, uop_strings, bop_strings):
    assert len(uop_strings) == len(bop_strings)
    for level in range(len(uop_strings)):
        log << log.mg << "Generate level" << level << log.endl
        bops = [ bop_map[_] for _ in bop_strings[level] ]
        uops = [ uop_map[_] for _ in uop_strings[level] ]
        fgraph.generate(uops, bops, min_exp=0.4, max_exp=3, depth=1)
    # Handle graph depth = 0 case (only input features)
    if len(uop_strings) < 1:
        fgraph.fnodes = fgraph.fnodes_in
    return fgraph

def represent_graph_2d(fgraph):
    fgraph = pickle.load(open(sys.argv[1], 'rb'))
    fgraph.embedChildIdcs()
    map_generations = fgraph.mapGenerations()
    root_nodes = filter(lambda f: f.root, fgraph.fnodes)
    # POSITION NODES
    dphi_root = 2*np.pi/len(map_generations[0])
    radius_root = 1.0
    radius_scale = 2.5
    for gen in range(len(map_generations)):
        nodes = map_generations[gen]
        print "GENERATION", gen
        for idx, node in enumerate(nodes):
            if gen == 0:
                node.phi = idx*dphi_root
                node.radius = radius_root
                print node.phi, node.tag
            elif len(node.parent_idcs) == 1:
                # Unary case
                pidx = node.parent_idcs[0]
                # Positioning
                node.phi = fgraph.fnodes[pidx].phi + (np.abs(node.cov*node.confidence))*dphi_root
                node.radius = (1.+gen)**2*radius_root + radius_scale*(np.abs(node.cov*node.confidence))*radius_root
            elif len(node.parent_idcs) == 2:
                pidx1 = node.parent_idcs[0]
                pidx2 = node.parent_idcs[1]
                pidx_key = '%d_%d' % tuple(sorted(node.parent_idcs))
                # Positioning
                phi_parents = sorted([ fgraph.fnodes[pidx].phi for pidx in node.parent_idcs ])
                dphi = phi_parents[1]-phi_parents[0]
                if dphi <= np.pi:
                    node.phi = phi_parents[0] + 0.5*dphi
                else:
                    node.phi = (phi_parents[1] + 0.5*(2*np.pi - dphi)) % (2*np.pi)
                node.phi = node.phi + (np.abs(node.cov*node.confidence))*dphi_root
                node.radius = (1.+gen)**2*radius_root + radius_scale*(np.abs(node.cov*node.confidence))*radius_root
    # LINKS BETWEEN NODES
    def connect_straight(f1, f2):
        x1 = f1.radius*np.cos(f1.phi)
        y1 = f1.radius*np.sin(f1.phi)
        x2 = f2.radius*np.cos(f2.phi)
        y2 = f2.radius*np.sin(f2.phi)
        w = np.abs(f2.cov*f2.confidence)
        return [ [x1,y1,w], [x2,y2,w] ]
    def connect_arc(f0, f1, f2, samples=15):
        x0 = f0.radius*np.cos(f0.phi)
        y0 = f0.radius*np.sin(f0.phi)
        x1 = f1.radius*np.cos(f1.phi)
        y1 = f1.radius*np.sin(f1.phi)
        x2 = f2.radius*np.cos(f2.phi)
        y2 = f2.radius*np.sin(f2.phi)
        w = np.abs(f2.cov*f2.confidence)
        r1 = ((x1-x0)**2+(y1-y0)**2)**0.5
        r2 = ((x2-x0)**2+(y2-y0)**2)**0.5
        phi1 = np.arctan2(y1-y0, x1-x0)
        phi2 = np.arctan2(y2-y0, x2-x0)
        if phi1 < 0.: phi1 = 2*np.pi + phi1
        if phi2 < 0.: phi2 = 2*np.pi + phi2
        phi_start = phi1
        dphi = phi2-phi1
        if dphi >= np.pi:
            dphi = 2*np.pi - dphi
            phi_end = phi_start-dphi
        elif dphi <= -np.pi:
            dphi = 2*np.pi + dphi
            phi_end = phi_start+dphi
        else:
            phi_end = phi_start + dphi
        coords = []
        for i in range(samples):
            phi_i = phi_start + float(i)/(samples-1)*(phi_end-phi_start)
            rad_i = r1 + float(i)/(samples-1)*(r2-r1)

            x_i = x0 + rad_i*np.cos(phi_i)
            y_i = y0 + rad_i*np.sin(phi_i)
            coords.append([x_i, y_i, w])
        return coords
    curves = []
    for fnode in fgraph.fnodes:
        if len(fnode.parent_idcs) == 1:
            pidx = fnode.parent_idcs[0]
            curves.append(connect_straight(fgraph.fnodes[pidx], fnode))
        elif len(fnode.parent_idcs) == 2:
            pidx1 = fnode.parent_idcs[0]
            pidx2 = fnode.parent_idcs[1]
            curves.append(connect_arc(fgraph.fnodes[pidx1], fgraph.fnodes[pidx2], fnode))
            curves.append(connect_arc(fgraph.fnodes[pidx2], fgraph.fnodes[pidx1], fnode))
        else: pass
    # Sort curves so important ones are in the foreground
    curves = sorted(curves, key=lambda c: c[0][-1])
    return fgraph, curves

def filter_fgraph(fgraph, X_probe, x_tags):
    # Filter out feature channels with vanishing spread or nan's/inf's
    keep = []
    IX_up = fgraph.applyMatrix(X_probe, x_tags, verbose=False)
    for cidx in range(IX_up.shape[1]):
        ok = True
        is_finite = np.isfinite(IX_up[:,cidx]).all()
        if not is_finite:
            print "Value error(s) [inf] for channel '%s'" % fgraph.fnodes[cidx].calculateTag(fgraph.fnodes)
            ok = False
        else:
            has_spread = (np.std(IX_up[:,cidx]) > 1e-10)
            if not has_spread:
                print "Value error(s) [std] for channel '%s'" % fgraph.fnodes[cidx].calculateTag(fgraph.fnodes)
                ok = False
        if ok: keep.append(cidx)
    if len(keep) < len(fgraph.fnodes):
        log << log.my << "Truncating from %d to %d channels due to value range errors" % (len(fgraph.fnodes), len(keep)) << log.endl
    fgraph.truncate_strict(keep)
    return fgraph

def trace_fgraph_weights(fgraph, assign_mode="weighted"):
    """
    Embeds fnode.weight member in FNodes
    """
    # Initialize
    fgraph.embedChildIdcs()
    for fnode in fgraph.fnodes:
        fnode.weight = 0.0
    root_nodes = filter(lambda f: f.root, fgraph.fnodes)
    # Trace
    def calculate_weight_recursive(fnode, fgraph, parent=None):
        w_this = np.abs(fnode.cov*fnode.confidence)
        n_parents = len(fnode.parent_idcs)
        n_visited = 1
        if fnode.root:
            print "ROOT @", fnode
        if n_parents == 0 or parent == None:
            w_assign = w_this
        elif n_parents == 1:
            w_assign = w_this
        elif n_parents == 2:
            if assign_mode == "weighted":
                p1 = fgraph.fnodes[fnode.parent_idcs[0]]
                p2 = fgraph.fnodes[fnode.parent_idcs[1]]
                w1 = np.abs(p1.cov*p1.confidence)
                w2 = np.abs(p2.cov*p2.confidence)
                if parent.idx == p1.idx:
                    w_assign = w_this*w1/(w1+w2+1e-10)
                elif parent.idx == p2.idx:
                    w_assign = w_this*w2/(w1+w2+1e-10)
                else: raise RuntimeError("Recursion violated relationships")
            elif assign_mode == "equal":
                w_assign = w_this*0.5
            else: raise ValueError(assign_mode)
        else: raise ValueError("Too many parents")
        for c in fnode.child_idcs:
            w_assign_add, n_visited_add = calculate_weight_recursive(fgraph.fnodes[c], fgraph, parent=fnode)
            w_assign += w_assign_add
            n_visited += n_visited_add
        return w_assign, n_visited
    weight_sum_direct = 0.0
    for f in fgraph.fnodes:
        weight_sum_direct += np.abs(f.cov*f.confidence)
    weight_sum_recursive = 0.0
    for r in root_nodes:
        weight, n_visited = calculate_weight_recursive(r, fgraph)
        weight_sum_recursive += weight
        r.weight = weight/n_visited
    print "Weight sum through summation:", weight_sum_direct
    print "Weight sum through recursion:", weight_sum_recursive
    root_nodes = sorted(root_nodes, key=lambda r: -r.weight)
    return root_nodes

def rank_fgraph(fgraph, p_threshold=0.85):
    """
    Embeds fnode.rank member in FNodes
    """
    for fnode in fgraph.fnodes:
        if fnode.root:
            deps = [ fnode ]
        else:
            deps = fnode.getDependencies(fgraph.fnodes)
            deps = [ fgraph.fnodes[d] for d in deps ]
            deps = filter(lambda f: f.root, deps)
        w = 1.0
        for d in deps: w *= d.weight
        w = w**(1./len(deps))
        #w *= math.factorial(len(deps))
        if fnode.confidence < p_threshold:
            fnode.rank = 0.0
        else:
            fnode.rank = np.abs(fnode.cov*fnode.confidence)*w
    ranked = filter(lambda f: f.rank > 1e-10, fgraph.fnodes)
    return sorted(ranked, key=lambda f: -f.rank)

def apply_fgraph(state, options, log):
    # Load feature graph
    log << "Loading feature graph" << log.endl
    fgraph = pickle.load(open(options["apply_fgraph"]["jarfile"], "rb"))
    # Graph feature transform
    log << "Apply graph transform" << log.endl
    IX = state["IX"]
    x_tags = state["IX_tags"][0]
    y = np.copy(state["T"])
    IX_up = fgraph.applyMatrix(IX, x_tags)
    IX_up_tags = [ f.calculateTag(fgraph.fnodes) for f in fgraph.fnodes ]
    if options["apply_fgraph"]["prefix_fct"] != None:
        IX_up_tags = [ options["apply_fgraph"]["prefix_fct"](_) + _ for _ in IX_up_tags ]
    IX_up_tags = np.tile(IX_up_tags, (len(state),1))
    if options["apply_fgraph"]["selected_only"]:
        selected = [ f.idx for f in fgraph.fnodes if f.selected ]
        selected = sorted(selected, key=lambda s: -fgraph.fnodes[s].cov**2)
        IX_up = IX_up[:, selected]
        IX_up_tags = IX_up_tags[:, selected]
    # Covariance analysis (optional)
    covs, order, top, cleaned = sis(IX_up=IX_up, y=np.copy(state["T"]), delta=1.0)
    for o in order:
        log << "rho=%+1.4f %s" % (covs[o], IX_up_tags[0,o]) << log.endl
        print "rho=%+1.4f %s" % (covs[o], IX_up_tags[0,o])
    state["IX"] = IX_up
    state["IX_tags"] = IX_up_tags
    return state

def sis_alt(IX_up, y, 
    zscore=True, 
    file_out=True, 
    fgraph=None, 
    IX_up_test=None, 
    method='?'):
    # Z-score feature matrix
    if zscore:
        std = np.std(IX_up, axis=0)
        idcs_zero = np.where(std < 1e-10)[0]
        #if len(idcs_zero) > 0: assert False
        IX_up[:,idcs_zero] = 0.0
        IX_up, mean, std = utils.zscore(IX_up)

    # Targets
    y_raw = np.copy(y)
    y_norm = (y - np.average(y))/np.std(y)

    # Covariance
    if method == "moment":
        covs = IX_up.T.dot(y_norm)/y_norm.shape[0]
    elif method == "spearman":
        # Find ranks
        ix_order = np.argsort(IX_up, axis=0)
        y_order = np.argsort(y_norm)
        ranks = np.arange(y_order.shape[0])
        # Convert variables to rank 
        ix_tf = np.zeros(IX_up.shape, dtype='float32')
        y_tf = np.zeros(y_norm.shape, dtype='float32')
        for ii in range(IX_up.shape[1]):
            ix_tf[ix_order[:,ii],ii] = ranks
        y_tf[y_order] = ranks
        # Z-score
        rank_std = np.std(ranks)
        rank_mean = np.average(ranks)
        # Correlate
        covs = (ix_tf-rank_mean).T.dot((y_tf-rank_mean))/(y_tf.shape[0]*rank_std**2)
        covs[idcs_zero] = 0.0
    elif method == "mixed":
        # TODO Rank transformation duplicated from branch above, unify
        covs_p = IX_up.T.dot(y_norm)/y_norm.shape[0]
        # Find ranks
        ix_order = np.argsort(IX_up, axis=0)
        y_order = np.argsort(y_norm)
        ranks = np.arange(y_order.shape[0])
        # Convert variables to rank
        ix_tf = np.zeros(IX_up.shape, dtype='float32')
        y_tf = np.zeros(y_norm.shape, dtype='float32')
        for ii in range(IX_up.shape[1]):
            ix_tf[ix_order[:,ii],ii] = ranks
        y_tf[y_order] = ranks
        # Z-score
        rank_std = np.std(ranks)
        rank_mean = np.average(ranks)
        # Correlated
        covs_s = (ix_tf-rank_mean).T.dot((y_tf-rank_mean))/(y_tf.shape[0]*rank_std**2)
        covs_s[idcs_zero] = 0.0
        # Mix
        covs = 0.8*covs_p+0.2*covs_s
    else:
        raise NotImplementedError(method)

    covs_avg = covs
    covs_obj = np.abs(covs)
    covs_std = np.zeros((len(covs),))

    order = np.argsort(covs_obj)[::-1]
    top = [ order[0] ]
    cleaned = [ order[0] ]

    if fgraph is not None:
        for idx, f in enumerate(fgraph.fnodes):
            fgraph.fnodes[idx].cov = covs[idx]
        for c in cleaned:
            fgraph.fnodes[c].selected = True
            print "Largest correlation:", fgraph.fnodes[c].calculateTag(fgraph.fnodes), covs_avg[c], covs_std[c]
            print "... Min, max, std =", np.min(IX_up[:,c]), ",", np.max(IX_up[:,c]), ",", np.std(IX_up[:,c])
    return covs, order, top, cleaned

def sis(IX_up, y, delta, file_out=True, fgraph=None, sim_threshold=0.8, cov_margin=None, min_cov=None):
    print "Z-scoring feature matrix"
    # Z-score feature matrix
    std = np.std(IX_up, axis=0)
    IX_up[:,np.where(std < 1e-5)[0]] = 0.0
    IX_up, mean, std = utils.zscore(IX_up)
    # Targets
    y_raw = np.copy(y)
    y_norm = (y - np.average(y))/np.std(y)
    # Covariance
    print "Calculating covariance with target"
    covs = IX_up.T.dot(y_norm)/y_norm.shape[0]
    order = np.argsort(np.abs(covs))[::-1]
    # Output
    if file_out:
        hist, bin_edges = np.histogram(np.abs(covs), bins=100, range=(0.,1.))
        bin_edges = bin_edges[:-1]+0.5*(bin_edges[1]-bin_edges[0])
        np.savetxt('out_sis_abs.hist', np.array([bin_edges, hist]).T)
    # Randomized control covariance
    if cov_margin == None:
        print "Running randomized covariance control"
        covs_rnd_agg = []
        for _ in range(100):
            log << log.back << "Repetition %d" % _ << log.flush
            y_rnd = np.copy(y_norm)
            np.random.shuffle(y_rnd)
            covs_rnd = IX_up.T.dot(y_rnd)/y_rnd.shape[0]
            covs_rnd_agg.append(covs_rnd)
        log << log.endl
        covs_rnd_agg = np.array(covs_rnd_agg).flatten()
        cov_margin = 2*np.std(covs_rnd_agg)
        if file_out:
            hist, bin_edges = np.histogram(np.abs(covs_rnd_agg), bins=200, range=(0,1.))
            bin_edges = bin_edges[:-1]+0.5*(bin_edges[1]-bin_edges[0])
            np.savetxt('out_sis_null_abs.hist', np.array([bin_edges, hist]).T)
        print "Max |cov| = %1.2f (rnd: = %1.2f)" % (np.max(np.abs(covs)), np.max(np.abs(covs_rnd_agg)))
    else: pass
    print "Covariance margin is %+1.4f" % (cov_margin)
    # Top delta*# features
    print "Short-listing highest-ranked features"
    top = order[0:int(delta*IX_up.shape[1]+0.5)]
    if fgraph != None:
        assert len(covs) == len(fgraph.fnodes)
        for fidx, f in enumerate(fgraph.fnodes):
            fgraph.fnodes[fidx].cov = covs[fidx]
        print "Screening genealogy"
        screened = {}
        node_sel_list = []
        for t in top:
            seq = fgraph.fnodes[t].calculateCovSequence(fgraph.fnodes)
            cov_max = np.abs(seq[0]["cov"])
            cov_sel = seq[0]
            for item in seq:
                #print "%+1.3f %s" % (item["cov"], item["tag"])
                if np.abs(item["cov"]) > cov_max - cov_margin:
                    cov_sel = item
            #print "SELECT", cov_sel
            if cov_sel["idx"] not in screened:
                #print "ADD"
                screened[cov_sel["idx"]] = True
                node_sel_list.append(cov_sel)
            else:
                pass
        print "Reduced top features from %d to %d" % (len(top), len(node_sel_list))
        if min_cov != None:
            print "Imposing minimum covariance"
            #node_idcs = [ n for idx, n in enumerate(node_idcs) if np.abs(node_sel_list[idx]["cov"]) >= min_cov ]
            node_sel_list = [ n for n in node_sel_list if n["cov"] >= min_cov ]
        node_idcs = [ n["idx"] for n in node_sel_list ]
        #for n in node_sel_list:
        #    print "%+1.4f %s" % (n["cov"], n["tag"])
        print "Sparsifying features"
        IX_up_slice = IX_up[:,node_idcs]
        K = IX_up_slice.T.dot(IX_up_slice)/IX_up_slice.shape[0]
        K = np.abs(K)
        screened = {}
        node_subsel_list = []
        nn_subsel_list = []
        for nn, item in enumerate(node_sel_list):
            row = K[nn] 
            collisions = np.where(row > sim_threshold)[0]
            add = True
            for c in collisions:
                if c in screened: add = False
            screened[nn] = True
            if add:
                node_subsel_list.append(item)
                nn_subsel_list.append(nn)
        print "Reduced top features from %d to %d" % (len(node_sel_list), len(node_subsel_list))
        for n in node_subsel_list:
            fgraph.fnodes[n["idx"]].selected = True
            print "%+1.4f %s" % (n["cov"], n["tag"])
        cleaned = [ _["idx"] for _ in node_subsel_list ]
        print "Covariance between selected feature nodes"
        #print K[nn_subsel_list][:,nn_subsel_list]

    else:
        cleaned = []

    return covs, order, top, cleaned
    
def cedf_analysis(xt, Ct, f_tail, ff_tail, fileout=False, cedf_type='gpd', verbose=False):
    """
    xt: numpy array of random samples in ascending order
    Ct: numpy array of sample complementary cumulative distribution function
    The CEDF is fitted to the largest f_tail samples
    Tail correction performed for largest ff_tail samples
    Returns: 
     - u: threshold value]
     - Cu: P(x >= u)
     - cedf_fct: function(params, support) for CEDF
     - params: fitted parameters for cedf_fct
     - tail_corr: tail correction ratio
    """
    assert np.abs(Ct[-1] - 0.5/xt.shape[0]) < 1e-5 # In sample C-CDF, smallest value should be 1./(2*N)

    # Establish threshold u
    N = xt.shape[0]
    n_tail = int(f_tail*N)
    nn_tail = int(ff_tail*N)
    i_tail = N-n_tail
    u = xt[i_tail] # threshold
    if verbose: print "Threshold:", u

    # Calculate sample conditional excess distribution function
    Cu = Ct[i_tail] # C-CDF at threshold
    Fu = 1.-Cu # CDF at threshold
    deltas = [] # excesses
    Fu_target = [] # conditional excess distribution function
    for i in range(i_tail, xt.shape[0]):
        x = xt[i] # u + delta
        Cx = Ct[i] # C-CDF @ u+delta
        Fx = 1.-Cx # CDF @ u+delta
        delta = x-u # excess
        Fu_delta =  (Fx-Fu)/Cu # CEDF
        deltas.append(delta)
        Fu_target.append(Fu_delta)
    def betainc(ab, support, args):
        return scipy.special.betainc(ab[0], ab[1], args["scale"]*support)
    def gpd_k0(s, support, args):
        return 1. - np.exp(-support/s[0])
    def gpd(ks, support, args):
        return 1. - (1. + ks[0]*support/ks[1])**(-1./ks[0])
    # Fit objective functions
    def cedf_opt(ks, support, cedf_target, fct_args):
        cedf = gpd(ks, support, fct_args)
        rmse = (np.sum((cedf-cedf_target)**2)/cedf.shape[0])**0.5
        if verbose: print "k = %1.4f   s = %1.4f   rmse = %1.4f" % (ks[0], ks[1], rmse)
        return rmse
    def cedf_opt_k0(s, support, cedf_target, fct_args):
        cedf = gpd_k0(s, support, fct_args)
        rmse = (np.sum((cedf-cedf_target)**2)/cedf.shape[0])**0.5
        if verbose: print "s = %1.4f   rmse = %1.4f" % (s[0], rmse)
        return rmse
    def beta_opt(ab, support, cedf_target, fct_args):
        cedf = betainc(ab, support, fct_args)
        rmse = (np.sum((cedf-cedf_target)**2)/cedf.shape[0])**0.5
        if verbose: print "a = %1.4f   b = %1.4f   rmse = %1.4f" % (ab[0], ab[1], rmse)
        return rmse

    if cedf_type == 'gpd':
        params_in = [1.,1.]
        obj_fct = cedf_opt
        cedf_fct = gpd
        cedf_fct_args = {}
    elif cedf_type == 'beta':
        params_in = [1.2, 2.7]
        obj_fct = beta_opt
        cedf_fct = betainc
        cedf_fct_args = { "scale": 1./(1.-u) }
    else: raise NotImplementedError("CEDF type '%s'" % cedf_type)

    # Add support limits for fit?
    add_bounds = False
    if add_bounds:
        Fu_target.append(1.)
        deltas.append(1.-u)

    # Fit tail to Generalized Pareto distribution
    Fu_target = np.array(Fu_target)
    support = np.array(deltas)
    params = scipy.optimize.fmin(obj_fct, params_in, args=(support, Fu_target, cedf_fct_args), disp=verbose)
    Fu_fit = cedf_fct(params, support, cedf_fct_args)
    if fileout: np.savetxt('cedf.txt', np.array([u+support, Fu_target, Fu_fit]).T)

    # Remove limits 
    if add_bounds:
        Fu_target = Fu_target[:-1]
        support = support[:-1]

    # Calculate tail correction
    delta_nn = support[-nn_tail:]
    Fu_nn_target = Fu_target[-nn_tail:]
    Fu_nn_fit = cedf_fct(params, delta_nn, cedf_fct_args)
    tail_corr = np.average((1.-Fu_nn_target)/(1.-Fu_nn_fit))
    print "dn %+1.4f cn_target %+1.4e cn_fit %+1.4e tail corr. %+1.4e" % (delta_nn[-1], Fu_nn_fit[-1], Fu_nn_target[-1], tail_corr)

    return u, Cu, cedf_fct, cedf_fct_args, params, tail_corr

def calculate_null_distribution(fgraph, rand_IX_list, rand_IX_tags, rand_Y, options):
    # Null distribution for covariance (as measured across random-feature ensemble)
    rand_covs = np.zeros((len(rand_IX_list), len(fgraph.fnodes)), dtype='float32')
    for i, rand_IX in enumerate(rand_IX_list):
        log << log.back << "Randomized control, instance" << i << log.flush
        rand_IX_up = fgraph.applyMatrix(rand_IX, rand_IX_tags[0], verbose=False)
        r_covs, r_order, r_top, r_cleaned = sis_alt(
            IX_up=rand_IX_up,
            #IX_up=rand_IX_up[idcs_train],
            zscore=True,
            y=rand_Y, 
            #y=np.copy(state["T"][idcs_train]), 
            fgraph=None,
            method=options.correlation_measure)
        rand_covs[i,:] = r_covs
    log << log.endl

    file_out = True
    p_threshold = 1. - options.tail_fraction
    n_channels = len(fgraph.fnodes)
    n_samples = rand_covs.shape[0]
    i_threshold = int(p_threshold*n_samples+0.5)
    log << "Tail contains %d samples" % (n_samples-i_threshold) << log.endl

    # Random-sampling convariance matrix
    # Rows -> sampling instances
    # Cols -> feature channels
    rand_cov_mat = np.copy(rand_covs)
    rand_cov_mat = np.abs(rand_cov_mat)
    # Sort covariance observations for each channel
    rand_covs = np.abs(rand_covs)
    rand_covs = np.sort(rand_covs, axis=0)
    # Cumulative distribution for each channel
    rand_cum = np.ones((n_samples,1), dtype='float32')
    rand_cum = np.cumsum(rand_cum, axis=0)
    rand_cum = (rand_cum-0.5) / rand_cum[-1,0]
    rand_cum = rand_cum[::-1,:]
    if file_out: np.savetxt('out_sis_channel_cov.hist', np.concatenate((rand_cum, rand_covs), axis=1))
    # Establish threshold for each channel
    thresholds = rand_covs[-int((1.-p_threshold)*n_samples),:]
    thresholds[np.where(thresholds < 1e-2)] = 1e-2
    t_min = np.min(thresholds)
    t_max = np.max(thresholds)
    t_std = np.std(thresholds)
    t_avg = np.average(thresholds)
    log << "Channel-dependent thresholds: min avg max +/- std = %1.2f %1.2f %1.2f +/- %1.4f" % (t_min, t_avg, t_max, t_std) << log.endl
    # Peaks over threshold: calculate excesses for random samples
    log << "Calculating excess for random samples" << log.endl
    pots = rand_covs[i_threshold:n_samples,:]
    rand_exs_mat = np.zeros((n_samples,n_channels), dtype='float32')
    for s in range(n_samples):
        log << log.back << "- Sample %d/%d" % (s+1, n_samples) << log.flush
        rand_cov_sample = rand_cov_mat[s]
        exs = -np.average((pots+1e-10-rand_cov_sample)/(pots+1e-10), axis=0)
        rand_exs_mat[s,:] = exs
    # Random excess distributions
    rand_exs = np.sort(rand_exs_mat, axis=1) # n_samples x n_channels
    rand_exs_cum = np.ones((n_channels,1), dtype='float32') # n_channels x 1
    rand_exs_cum = np.cumsum(rand_exs_cum, axis=0)
    rand_exs_cum = (rand_exs_cum-0.5) / rand_exs_cum[-1,0]
    rand_exs_cum = rand_exs_cum[::-1,:]
    rand_exs_avg = np.average(rand_exs, axis=0)
    rand_exs_std = np.std(rand_exs, axis=0)
    # Rank distributions
    rand_exs_rank = np.sort(rand_exs, axis=0) # n_samples x n_channels
    rand_exs_rank = rand_exs_rank[:,::-1]
    rand_exs_rank_cum = np.ones((n_samples,1), dtype='float32') # n_channels x 1
    rand_exs_rank_cum = np.cumsum(rand_exs_rank_cum, axis=0)
    rand_exs_rank_cum = (rand_exs_rank_cum-0.5) / rand_exs_rank_cum[-1,0]
    rand_exs_rank_cum = rand_exs_rank_cum[::-1,:]
    if file_out: np.savetxt('out_exs_rank_rand.txt', np.concatenate([ rand_exs_rank_cum, rand_exs_rank ], axis=1))
    # ... Histogram
    if file_out: np.savetxt('out_exs_rand.txt', np.array([rand_exs_cum[:,0], rand_exs_avg, rand_exs_std]).T)
    log << log.endl
    return pots, rand_exs_cum, rand_exs_rank, rand_exs_rank_cum

def rank_ptest(
        fgraph, # used only for info and sorting
        exs, 
        exs_cum, 
        rand_exs_rank, 
        rand_exs_rank_cum, 
        threshold, # lower confidence threshold
        verbose):
    n_channels = exs.shape[0]
    # ... Calculate observation probabilities given rank as well as absolutely
    idcs_sorted = np.argsort(exs)[::-1]
    p_first_list = np.zeros((n_channels,))
    p_rank_list = np.zeros((n_channels,))
    for rank, c in enumerate(idcs_sorted):
        # Calculate probability to observe feature given its rank
        ii = np.searchsorted(rand_exs_rank[:,rank], exs[c])
        if ii >= rand_exs_rank_cum.shape[0]:
            p0 = rand_exs_rank_cum[ii-1,0]
            p1 = 0.0
        elif ii <= 0:
            p0 = 1.0
            p1 = rand_exs_rank_cum[ii,0]
        else:
            p0 = rand_exs_rank_cum[ii-1,0]
            p1 = rand_exs_rank_cum[ii,0]
        p_rank = 0.5*(p0+p1)
        # Calculate probability to observe feature as highest-ranked
        ii = np.searchsorted(rand_exs_rank[:,0], exs[c])
        if ii >= rand_exs_rank_cum.shape[0]:
            p0 = rand_exs_rank_cum[ii-1,0]
            p1 = 0.0
        elif ii <= 0:
            p0 = 1.0
            p1 = rand_exs_rank_cum[ii,0]
        else:
            p0 = rand_exs_rank_cum[ii-1,0]
            p1 = rand_exs_rank_cum[ii,0]
        p_first = 0.5*(p0+p1)

        if verbose:
            if rank <= 100:
                print "Rank=%4d  Excess=%+1.4e  p_1=%+1.4e  p_r=%+1.4e  Feature=%s" % (rank, exs[c], p_first, p_rank, fgraph.fnodes[c].calculateTag(fgraph.fnodes))
            elif rank == 100:
                print "..."
            else: pass
        p_first_list[c] = p_first
        p_rank_list[c] = p_rank
    if verbose: np.savetxt('out_exs_phys.txt', np.array([exs_cum[::-1,0], exs[idcs_sorted], p_rank_list[idcs_sorted], p_first_list[idcs_sorted]]).T)
    cleaned = np.where(p_first_list <= 1.-threshold)[0]
    cleaned = sorted(cleaned, key=lambda c: (1.-p_first_list[c])*-np.abs(fgraph.fnodes[c].cov))
    confidence_level = [ 1.-p_first_list[c] for c in range(n_channels) ]
    return cleaned, confidence_level, p_first_list[idcs_sorted[0]], exs[idcs_sorted[0]]

def sparsify_vectors_greedy(IX, threshold):
    IX, mean, std = utils.zscore(IX)
    K = IX.T.dot(IX)/(IX.shape[0]-1.)
    K = np.abs(K)
    print K
    check_short_listed = {}
    short_listed = []
    for row_idx in range(IX.shape[1]):
        row = K[row_idx]
        collisions = np.where(row >= threshold)[0]
        add = True
        for col_idx in collisions:
            if col_idx == row_idx: continue
            if col_idx in check_short_listed:
                add = False
                break
        if add:
            check_short_listed[row_idx] = True
            short_listed.append(row_idx)
    return short_listed







