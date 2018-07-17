import numpy as np

# =========
# OPERATORS
# =========

class BOperator(object):
    def __init__(self):
        self.optype = "b"
        pass

class UOperator(object):
    def __init__(self):
        self.optype = "u"
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
        maybe_neg = (f1.maybe_neg or f2.maybe_neg)
        return f1.uvec, maybe_neg
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
        maybe_neg = True
        return f1.uvec, maybe_neg
    def apply(self, v1, v2):
        return v1-v2
    def latex(self, args, priorities):
        return r"%s-%s" % tuple(args)

class OMult(BOperator):
    def __init__(self):
        self.tag = "*"
    def check(self, f1, f2):
        return True
    def generate(self, f1, f2):
        if (not f1.maybe_neg) and (not f2.maybe_neg):
            maybe_neg = False
        else:
            maybe_neg = True
        return f1.uvec + f2.uvec, maybe_neg
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
        maybe_neg = (f1.maybe_neg or f2.maybe_neg)
        return f1.uvec-f2.uvec, maybe_neg
    def apply(self, v1, v2):
        return v1/v2
    def latex(self, args, priorities):
        return r"\\frac{%s}{%s}" % tuple(args)

class OExp(UOperator):
    def __init__(self):
        self.tag = "e"
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
        maybe_neg = False
        return f1.uvec, maybe_neg
    def apply(self, v1):
        return np.exp(v1)
    def latex(self, args, priorities):
        return r"\\exp(%s)" % tuple(args)

class OLog(UOperator):
    def __init__(self):
        self.tag = "l"
    def check(self, f1):
        ok = True
        if f1.maybe_neg: ok = False
        else:
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
        maybe_neg = True
        return f1.uvec, maybe_neg
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
        maybe_neg = False
        return f1.uvec, maybe_neg
    def apply(self, v1):
        return np.abs(v1)
    def latex(self, args, priorities):
        return r"|%s|" % tuple(args)

class OSqrt(UOperator):
    def __init__(self):
        self.tag = "s"
    def check(self, f1):
        ok = True
        if f1.maybe_neg: ok = False
        elif f1.op.tag == "^2": ok = False
        else: pass
        return ok
    def generate(self, f1):
        maybe_neg = False
        return 0.5*f1.uvec, maybe_neg
    def apply(self, v1):
        return np.sqrt(v1)
    def latex(self, args, priorities):
        return r"\\sqrt{%s}" % tuple(args)

class OInv(UOperator):
    def __init__(self):
        self.tag = "r"
    def check(self, f1):
        ok = True
        if f1.op.tag == "^-1": ok = False
        return ok
    def generate(self, f1):
        return -f1.uvec, f1.maybe_neg
    def apply(self, v1):
        return 1./v1
    def latex(self, args, priorities):
        if priorities[0] > op_priority[self.tag]:
            args[0] = "(%s)" % args[0]
        return r"%s^{-1}" % tuple(args)

class O2(UOperator):
    def __init__(self):
        self.tag = "2"
    def check(self, f1):
        ok = True
        if f1.op.tag == "^1/2": ok = False
        return ok
    def generate(self, f1):
        maybe_neg = False
        return 2*f1.uvec, maybe_neg
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
    None: OIdentity(),
    "I": OIdentity(),
    "e": OExp(),
    "l": OLog(),
    "|": OMod(),
    "s": OSqrt(),
    "r": OInv(),
    "2": O2(),
}

op_map = {}
op_map.update(bop_map)
op_map.update(uop_map)

op_priority = {
    None: 0,
    "I":0,
    "*":1,
    ":":1,
    "+":2,
    "-":2, 
    "e":0,
    "l":0, 
    "|":0, 
    "s":1, 
    "r":1,
    "2":1,
}
