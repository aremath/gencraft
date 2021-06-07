
class Assignment(object):

    def __init__(self, ident, priority, require_str, preexpr):
        self.ident = ident
        self.preexpr = preexpr
        self.priority = priority
        self.require_str = require_str

    def __repr__(self):
        return "{} := {} @ {} with {}".format(self.ident, self.preexpr, self.priority, self.require_str)

# Expressions
class Expr(object):
    pass

    def compile(self, env):
        pass

class BlockFunctionDef(Expr):

    def __init__(self, blocks_out, blocks_in):
        self.blocks_out = blocks_out
        self.blocks_in = blocks_in

    def compile(self, env):
        pass

    def __repr__(self):
        return "BlockFunction({} -> {})".format(self.blocks_in.__repr__(), self.blocks_out.__repr__())

class ExprGraph(Expr):

    def __init__(self, block_tree):
        self.block_tree = block_tree

    def compile(self, env):
        pass

class Namespace(Expr):

    def __init__(self, assignments):
        self.assignments = assignments

    def compile(self, env):
        # Compile assignments in order
        self.assignments.sort(key=lambda x: x.priority)
        self_dict = {}
        for assignment in self.assignments:
            key = assignment.ident
            self_env = env | self_dict
            use_env = eval(assignment.require_str, self_env)
            # Assignments are compiled with env | self dict
            val = assignment.preexpr.compile(self_env | use_env)
            # Add them to the self dict
            self_dict[key] = val
        # Return either self_dict, or self_dict["return"]
        if "return" in self_dict:
            return self_dict["return"]
        else:
            return self_dict

    def __repr__(self):
        return self.assignments.__repr__()

class Lambda(Expr):

    def __init__(self, var_list, expr):
        self.var_list = var_list
        self.expr = expr

    #TODO: should we do variable capture like this?
    # Where do we get the env from if not?
    def compile(self, env):
        return Closure(self.var_list, self.expr, env)

class String(Expr):

    def __init__(self, s):
        self.s = s

    def compile(self, env):
        return eval(self.s, env)

    def __repr__(self):
        return "String({})".format(self.s)

class BlockExpr(Expr):

    def __init__(self, b):
        self.b = b

    def compile(self, env):
        try:
            return env[self.b]
        except KeyError:
            return BlockFunction(set([(self.b, frozenset([self.b, air]))]))

    def __repr__(self):
        return "BlockExpr({})".format(self.b)

class UnionDef(Expr):

    def __init__(self, exprs):
        self.exprs = exprs

    def compile(self, env):
        out = map(lambda x: x.compile(env), self.exprs)
        return Union(out)

class FunCall(Expr):

    def __init__(self, e_caller, e_args):
        self.e_caller = e_caller
        self.e_args = e_args

    def compile(self, env):
        v_args = map(lambda x: x.compile(env), self.e_args)
        v_caller = e_caller.compile(env)
        return v_caller(*v_args)

class Graph(object):

    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges

# Compiled Values
class Value(object):
    pass

    def build(self, level, pos, start_direction):
        pass


# No class Namespace, instead namespace is just a dict
# -> easier use with |

class BlockFunction(Value):

    def __init__(self, s):
        # (block * block set) set
        self.set = s

    def __call__(self, block):
        return set([b for (b, s) in self.set if block in s])

    def __or__(self, other):
        return BlockFunction(self.set | other.set)

class Unknown(BlockFunction):

    def __init__():
        pass

    def __call__(self, block):
        return block

    #TODO: this is problematic
    def __or__(self, other):
        pass

class ValueGraph(Value):
    pass

class Closure(Value):

    def __init__(self, var_list, pre_expr, env):
        self.var_list = var_list
        self.expr = pre_expr
        self.env = env

    def __call__(self, *args):
        arg_env = {}
        for ident, arg in zip(self.var_list, args):
            arg_env[ident] = arg
        return self.expr.compile(self.env | arg_env)

class Union(Value):

    def __init__(self, val_list):
        self.val_list = val_list

class ParseInfo(object):
    """ Holds all the info for parsing """

    def __init__(self, level, signs, assignment_pos, assignment_signs, assignment_set, namespaces, namespace_as, a_namespaces):
        self.level = level
        self.signs = signs
        # Key: assignment pos, Value: {pos} on assigment component
        self.assignment_pos = assignment_pos
        # Key: assignment pos, Value: assignment sign
        self.assignment_signs = assignment_signs
        # Set of all positions that belong to an assignment
        self.assignment_set = assignment_set
        # Key: namespace def position, Value: (min, max) bounding box for namespace
        self.namespaces = namespaces
        # Key: Namespace pos, Value: [assignment pos] in that namespace
        self.namespaces_as = namespace_as
        # Key: Assignment pos, Value: [namespaces pos] the assignment is in
        self.a_namespaces = a_namespaces

