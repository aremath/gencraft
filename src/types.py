
class PreAssignment(object):

    def __init__(self, ident, priority, require_str, preexpr):
        self.ident = ident
        self.preexpr = preexpr
        self.priority = priority
        self.require_str = require_str

class PreExpr(object):
    pass

    def compile(self, env):
        pass

class PreBlockFunction(PreExpr):

    def __init__(self, blocks_out, blocks_in):
        self.blocks_out = blocks_out
        self.blocks_in = blocks_in

    def compile(self, env):
        pass

class PrePattern(PreExpr):

    def __init__(self, block_tree):
        self.block_tree = block_tree

    def compile(self, env):
        pass

class PreNamespace(PreExpr):

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

class PreLambda(PreExpr):

    def __init__(self, var_list, expr):
        self.var_list = var_list
        self.expr = expr

    #TODO: should we do variable capture like this?
    # Where do we get the env from if not?
    def compile(self, env):
        return Lambda(self.var_list, self.expr, env)

class Expr(object):
    pass

    def build(self, level, pos, start_direction):
        pass

# No class Namespace, instead namespace is just a dict
# -> easier use with |

class BlockFunction(Expr):

    def __init__(self, s):
        # (block * block set) set
        self.set = s

    def __call__(self, block):
        return set([b for (b, s) in self.set if block in s])

    def __or__(self, other):
        return BlockFunction(self.set || other.set)

class Unknown(BlockFunction):

    def __init__():
        pass

    def __call__(self, block):
        return block

    #TODO: this is problematic
    def __or__(self, other):
        pass

class Pattern(Expr):
    pass

class Lambda(Expr):

    def __init__(self, var_list, pre_expr, env):
        self.var_list = var_list
        self.pre_expr = pre_expr
        self.env = env

    def __call__(self, *args):
        arg_env = {}
        for ident, arg in zip(self.var_list, args):
            arg_env[ident] = arg
        return self.pre_expr.compile(self.env | arg_env)

class ParseInfo(object):
    """ Holds all the info for parsing """

    def __init__(self, level, signs, assignment_pos, assingment_signs, assignment_set, namespaces, namespace_as, a_namespaces):
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

