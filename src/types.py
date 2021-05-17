class BlockFunction(object):

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

class Pattern(object):
    
    def __init__(self):
        pass

class JustPattern(Pattern):

    def __init__(self, func_array, interfaces):
        self.func_array = func_array
        self.interfaces = interfaces
