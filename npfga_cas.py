import copy
from npfga_ops import *

def get_instruction_basename(instruction):
    op = instruction[0]
    args = instruction[1:]
    if op is None:
        assert len(args) == 1
        instruction_str = args[0][0]
    else:
        args = map(stringify_instruction, args)
        instruction_str = "%s(%s)" % (op, ','.join(args))
    return instruction_str

def stringify_instruction(instruction, pretty_print=True):
    op = instruction[0]
    args = instruction[1:]
    if op is None:
        assert len(args) == 1
        if args[0][1] == 1: power = ""
        elif type(args[0][1]) == int: power = "^{%d}" % args[0][1]
        elif args[0][1] % 0.5 < 1e-10: power = "^{%1.1f}" % args[0][1]
        else: power = "^{%1.2f}" % args[0][1]
        instruction_str = args[0][0] + power
    else:
        op = op_map[op]
        priorities = [ op_priority[a[0]] for a in args ]
        args = [ stringify_instruction(a, pretty_print) for a in args ]
        if pretty_print:
            instruction_str = op.latex(args, priorities)
        else:
            instruction_str = "%s(%s)" % (op.tag, ','.join(args))
    return instruction_str

def take_instruction_to_power(instruction, n):
    op = instruction[0]
    if op is None:
        assert len(instruction) == 2
        instruction[1][1] *= n
    elif op in "+-":
        instruction = [ None, [ stringify_instruction(instruction), n ] ] 
    elif op in "*":
        for i in range(1, len(instruction)):
            instruction[i] = take_instruction_to_power(instruction[i], n)
    else: raise ValueError(op)
    return instruction

def get_instruction_factors(instruction):
    factors = []
    if instruction[0] != "*": factors = [ instruction ]
    else: 
        for arg in instruction[1:]:
            factors = factors + get_instruction_factors(arg)
    return factors

def get_instruction_elements(instruction):
    elements = []
    if instruction[0] == None:
        assert len(instruction) == 2
        elements.append(instruction[1][0])
    else:
        for arg in instruction[1:]:
            elements = elements + get_instruction_elements(arg)
    return elements

def simplify_instruction(instruction):
    if instruction[0] == "*":
        factors = get_instruction_factors(instruction)
        factors_map = { get_instruction_basename(f): [] for f in factors }
        for f in factors:
            factors_map[get_instruction_basename(f)].append(f)
        factors_out = []
        for fstr, fs in factors_map.iteritems():
            # Calculate new exponents
            for i in range(1, len(fs)):
                fs[0][1][1] += fs[i][1][1]
            if fs[0][1][1] != 0:
                factors_out.append(fs[0])
        if len(factors_out) == 0:
            instruction = [ None, [ "1", 1 ] ]
        elif len(factors_out) == 1:
            instruction = factors_out[0]
        else:
            factors_out = sorted(factors_out, key = lambda f: stringify_instruction(f))
            instruction = [ "*" ] + factors_out
    else: pass
    return instruction

def copy_instruction(instruction):
    return copy.deepcopy(instruction)

