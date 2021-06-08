from functools import reduce

class Assignment(object):

    def __init__(self, ident, priority, require_str, preexpr, pos):
        self.ident = ident
        self.preexpr = preexpr
        self.priority = priority
        self.require_str = require_str
        self.pos = pos

    def __repr__(self):
        if self.require_str != "":
            return "{}: {} := {} @ {} with {}".format(self.pos, self.ident, self.preexpr, self.priority, self.require_str)
        else:
            return "{}: {} := {} @ {}".format(self.pos, self.ident, self.preexpr, self.priority)

# Expressions
class Expr(object):
    pass

    def compile(self, env):
        pass

class BlockFunctionDef(Expr):

    def __init__(self, blocks_out, blocks_in, pos):
        self.blocks_out = blocks_out
        self.blocks_in = blocks_in
        self.pos = pos

    #TODO
    def compile(self, env):
        pass

    def __repr__(self):
        return "{}: BlockFunction({} -> {})".format(self.pos, self.blocks_in, self.blocks_out)

class ExprGraph(Expr):

    def __init__(self, block_tree, pos):
        self.block_tree = block_tree
        self.pos = pos

    def compile(self, env):
        pass

class Namespace(Expr):

    def __init__(self, assignments, pos):
        self.assignments = assignments
        self.pos = pos

    def compile(self, env):
        print("Compiling Namespace")
        # Compile assignments in order
        self.assignments.sort(key=lambda x: x.priority)
        self_dict = {}
        for assignment in self.assignments:
            print(assignment)
            key = assignment.ident
            self_env = env | self_dict
            if assignment.require_str != "":
                use_envs = eval(assignment.require_str, self_env)
                use_env = reduce(lambda x, y: x | y, use_envs, {})
            else:
                use_env = {}
            # Assignments are compiled with env | self dict
            val = assignment.preexpr.compile(self_env | use_env)
            # Add them to the self dict
            self_dict[key] = val
            print(self_dict)
        # Return either self_dict, or self_dict["return"]
        if "return" in self_dict:
            return self_dict["return"]
        else:
            return self_dict

    def __repr__(self):
        #return "{}: Namespace({})".format(self.pos, self.assignments)
        return "{}: Namespace(...)".format(self.pos)

class Lambda(Expr):

    def __init__(self, var_list, expr, pos):
        self.var_list = var_list
        self.expr = expr
        self.pos = pos

    #TODO: should we do variable capture like this?
    # Where do we get the env from if not?
    def compile(self, env):
        return Closure(self.var_list, self.expr, env)

    def __repr__(self):
        #return "{}: Lambda({})".format(self.pos, self.expr)
        return "{}: Lambda(...)".format(self.pos)

class String(Expr):

    def __init__(self, s, pos):
        self.s = s
        self.pos = pos

    def compile(self, env):
        print("Compiling String: {}".format(self))
        return eval(self.s, env)

    def __repr__(self):
        return "{}: String({})".format(self.pos, self.s)

class BlockExpr(Expr):

    def __init__(self, b, pos):
        self.b = b
        self.pos = pos

    def compile(self, env):
        try:
            return env[self.b]
        except KeyError:
            return BlockFunction(set([(self.b, frozenset([self.b, air]))]))

    def __repr__(self):
        return "{}: BlockExpr({})".format(self.pos, self.b)

class UnionDef(Expr):

    def __init__(self, exprs, pos):
        self.exprs = exprs
        self.pos = pos

    def compile(self, env):
        out = map(lambda x: x.compile(env), self.exprs)
        return Union(out)

    def __repr__(self):
        #return "{}: Union({})".format(self.pos, self.exprs)
        return "{}: Union(...)".format(self.pos)

class FunCall(Expr):

    def __init__(self, e_caller, e_args, pos):
        self.e_caller = e_caller
        self.e_args = e_args
        self.pos = pos

    def compile(self, env):
        v_args = map(lambda x: x.compile(env), self.e_args)
        v_caller = e_caller.compile(env)
        return v_caller(*v_args)

    def __repr__(self):
        return "{}: FunCall({}, ({}))".format(self.pos, self.e_caller, self.e_args)
        return "{}: FunCall(...)".format(self.pos)

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

