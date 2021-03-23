import numpy as np
from bidict import bidict
from collections import deque

from amulet.api.block import Block, blockstate_to_block
from amulet.api.selection import SelectionBox, SelectionGroup
#from amulet.operations.fill import fill
from amulet import load_level
import amulet_nbt

define_block = blockstate_to_block("universal_minecraft:bedrock[infiniburn=\"false\"]")
context_block = blockstate_to_block("universal_minecraft:wool[color=\"blue\"]")
pattern_block = blockstate_to_block("universal_minecraft:wool[color=\"red\"]")
function_block = blockstate_to_block("universal_minecraft:wool[color=\"lime\"]")
torch_up = blockstate_to_block("universal_minecraft:torch[facing=\"up\"]")

# What blocks count as valid signs
sign_blocks = set([
    blockstate_to_block("universal_minecraft:oak_wall_sign"),
    blockstate_to_block("universal_minecraft:oak_sign")
    ])

# Rotation matrix around z-axis (right hand rule)
# (1,0,0) -> (0,1,0)
rot90z = np.array([
    [0, -1, 0],
    [1, 0, 0],
    [0, 0, 1]])

# Rotation matrix around y-axis
# (0,0,1) -> (1,0,0)
rot90y = np.array([
    [0, 0, 1],
    [0, 1, 0],
    [-1, 0, 0]])

# Rotation matrix around z-axis
# (0,1,0) -> (0,0,1)
rot90x = np.array([
    [1, 0, 0],
    [0, 0, -1],
    [0, 1, 0]])

# Mirror the x-axis
mirrorx = np.array([
    [-1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]])

# Mirror the y-axis
mirrory = np.array([
    [1, 0, 0],
    [0, -1, 0],
    [0, 0, 1]])

# Mirror the z-axis
mirrorz = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, -1]])

adjacents = set([
    (1,0,0),
    (-1,0,0),
    (0,1,0),
    (0,-1,0),
    (0,0,1),
    (0,0,-1)
    ])

# Map the facing of a sign or stairs to a direction
facing_to_dir = bidict({
    "up": (0,1,0),
    "down": (0,-1,0),
    "east": (-1,0,0),
    "west": (1,0,0),
    "north": (0,0,-1),
    "south": (0,0,1)
        })

z_dirs = ["up", "down"]

axis_to_dir = bidict({
    "x" : (1,0,0),
    "y" : (0,1,0),
    "z" : (0,0,1)
    })

half_to_dir = bidict({
    "top": (0,1,0),
    "bottom": (0,-1,0)
    })

dir_vals = {
    "n": 1,
    "s": 2,
    "e": 3,
    "w": 4
    }

def transform_block_array(block_array, transform_matrix):
    shape_transform = transform_matrix @ np.array(block_array.shape)
    new_shape = tuple([abs(s) for s in shape_transform])
    offset = tuple([abs(min(s+1, 0)) for s in shape_transform])
    new_array = np.ndarray(new_shape)
    for index, b in np.ndenumerate(block_array):
        b_t = transform_block(b, transform_matrix)
        pos = (transform_matrix @ index) + offset
        new_array[tuple(pos)] = b_t
    return new_array

def set_block_property(block, key, val):
    p = block.properties.copy()
    nbt_val = None
    if isinstance(val, str):
        nbt_val = amulet_nbt.TAG_String(val)
    #TODO: can blocks actually have non-string properties?
    # Why are things like glass panes facings stored as strings?
    else:
        assert False, "Bad Property"
    p[key] = nbt_val
    block2 = Block(block.namespace, block.base_name, p)
    return block2

# Wheeee spaghetti code
def transform_block(block, transform_matrix):
    """ 
    Apply a linear transformation to a block by setting properties
    """
    # Facing (Stairs, Ladders, Signs, etc.)
    if "facing" in block.properties:
        # Get the transformed facing direction
        facing_vec = transform_matrix @ facing_to_dir[block.properties["facing"][:]]
        # Convert back to a string
        facing_str = facing_to_dir.inverse[tuple(facing_vec)]
        # Stairs need both Half and Facing handled together
        if "half" in block.properties:
            half_vec = transform_matrix @ half_to_dir[block.properties["half"][:]]
            u = facing_to_dir["up"]
            hu = u @ half_vec
            fu = u @ facing_vec
            # Need at least one vector pointing in the y-direction
            # Which vector controls facing versus half might change
            if hu != 0 or fu != 0:
                if hu != 0:
                    y_vec = half_vec
                    h_vec = facing_vec
                elif fu != 0:
                    y_vec = facing_vec
                    h_vec = half_vec
                # Vertical controls half
                half_str = half_to_dir.inverse[tuple(y_vec)]
                # Horizontal controls facing
                facing_str = facing_to_dir.inverse[tuple(h_vec)]
                block = set_block_property(block, "facing", facing_str)
                block = set_block_property(block, "half", half_str)
            # If both vectors are in the xz plane, then can't do the transformation
            else:
                pass
        else:
            # Pistons are allowed to face up!
            if block.base_name == "piston" or block.base_name == "sticky_piston":
                block = set_block_property(block, "facing", facing_str)
            # The requested transformation may not be possible if not a piston
            elif not facing_str in z_dirs:
                block = set_block_property(block, "facing", facing_str)
    # Axis (Logs)
    if "axis" in block.properties:
        axis_vec = transform_matrix @ axis_to_dir[block.properties["axis"][:]]
        # Axis is direction agnostic
        axis_vec = tuple([abs(s) for s in axis_vec])
        axis_str = axis_to_dir.inverse[axis_vec]
        block = set_block_property(block, "axis", axis_str)
    # East, South, West, North (Glass, Fences, etc.)
    if "east" in block.properties:
        # Get the "true"s
        t_strs = [x for x in facing_to_dir.keys() if x not in z_dirs and block.properties[x][:] == "true"]
        t_vecs = [facing_to_dir[x] for x in t_strs]
        # Transform the "true"s
        tt_vecs = [tuple(transform_matrix @ x) for x in t_vecs]
        tt_strs = [facing_to_dir.inverse[x] for x in tt_vecs]
        # Don't transform it if any transformed true is a z-direction
        if not any([t in z_dirs for t in tt_strs]):
            # If they're all h-directions, set all relevant properties
            for f_dir in facing_to_dir.keys():
                if f_dir not in z_dirs:
                    if f_dir in tt_strs:
                        block = set_block_property(block, f_dir, "true")
                    else:
                        block = set_block_property(block, f_dir, "false")
    # Shape (For rails)
    if "shape" in block.properties:
        a = block.properties["shape"][:].split("_")
        if len(a) == 2:
            x,y = a
            # Rails are either "ascending"
            if x == "ascending":
                dir_vec = transform_matrix @ facing_to_dir[y]
                dir_str = facing_to_dir.inverse[axis_vec]
                if not dir_str in z_dirs:
                    dir_str = "{}_{}".format(x, dir_str)
                    block = set_block_property(block, "shape", dir_str)
            # Or d1_d2 indicating corners
            elif x in facing_to_dir and y in facing_to_dir:
                d1 = transform_matrix @ facing_to_dir[x]
                s1 = facing_to_dir.inverse[d1]
                d2 = transform_matrix @ facing_to_dir[y]
                s2 = facing_to_dir.inverse[d2]
                if not s1 in z_dirs and not s2 in z_dirs:
                    s1, s2 = sorted([s1, s2], key=lambda x: dir_vals[x[0]])
                    dir_str = "{}_{}".format(s1, s2)
                    block = set_block_property(block, "shape", dir_str)
            # Other blocks' shapes are not direction-relevant #TODO is this true?
            else:
                pass
    #TODO Type (Top, Bottom for slabs)
    if "type" in block.properties:
        pass
    #TODO Rotation (For ground signs)
    if "rotation" in block.properties:
        # Convert rotation to a vector (4 "ticks" = pi/2 radians)
        pass
    # Return the transformed block (defaults to no transformation)
    return block

def parse(level_folder, coord_box):
    level = load_level(level_folder)
    coord_start, coord_end = coord_box
    print("Reference: {}".format(coord_start))
    box = SelectionBox(coord_start, coord_end)
    blocks = get_level(level, box)
    coord_offset = np.array(box.min)
    #signs = get_signs(level, box)
    signs = read_signs("level_test_signs", coord_start)
    #print(signs)
    #parse_contexts(blocks, signs)
    i = np.array([11,5,169]) - coord_offset
    block = blocks[tuple(i)]
    #block.properties["facing"] = "east"
    print(block)
    print(transform_block(block, rot90x @ rot90x))
    print(transform_block(transform_block(block, rot90x), rot90x))
    return blocks, coord_offset

def get_signs(level, box):
    for x,y in box.chunk_locations():
        c = level.get_chunk(x,y,"overworld")
        for e in c.block_entities:
            # Get the facing using level.get_block().properties["facing"]
            print(e)
            print(e.nbt)
            print(e.nbt.keys())
            t1 = e.nbt["utags"]["Text1"]
            t2 = e.nbt["utags"]["Text2"]
            t3 = e.nbt["utags"]["Text3"]
            t4 = e.nbt["utags"]["Text4"]
    pass

class Sign(object):

    def __init__(self, pos, on, text):
        self.pos = pos
        self.on = on
        self.text = text

    def __repr__(self):
        return str((self.pos, self.on, self.text))

def sign_on(pos, facing):
    if facing is None:
        return pos
    else:
        return tuple(np.array(pos) + np.array(facing_to_dir[facing]))

def read_signs(f, reference):
    signs = []
    texts = []
    pos = None
    facing = None
    with open(f) as sign_file:
        for line in sign_file.readlines():
            if line.startswith("sign:"):
                if pos is not None:
                    on = sign_on(pos, facing)
                    sign = Sign(pos, on, texts)
                    signs.append(sign)
                    # Reset
                    texts = []
                    facing = None
                    pos = None
                a = line.split("sign:")
                # Careful!
                t = a[1].strip()
                ldict = {}
                exec("pos =" + t,globals(),ldict)
                pos = ldict["pos"]
                pos = tuple(np.array(pos) - reference)
            elif line.startswith("facing:"):
                a = line.split("facing:")
                t = a[1].strip()
                ldict = {}
                facing = t
            else:
                texts.append(line)
    # Add the last sign
    if pos is not None:
        on = sign_on(pos, facing)
        sign = Sign(pos, on, texts)
        signs.append(sign)
        # Reset
        texts = []
        facing = None
        pos = None
    return signs

def get_level(level, box):
    shape = np.array(box.max) - np.array(box.min)
    level_array = np.ndarray(shape, dtype="object")
    print(level_array.shape)
    for (x,y,z) in box:
        block = level.get_block(x,y,z,"overworld")
        np_index = tuple(np.array([x,y,z]) - box.min)
        level_array[np_index] = block
    return level_array

def inbounds(pos, level):
    min_coord = np.array([0,0,0])
    if all(pos >= min_coord) and all(pos < np.array(level.shape)):
        return True
    else:
        return False

#TODO: Dijkstras for goal searching
def bfs(seed, level, lambda_ok, goal=None):
    assert inbounds(seed, level)
    q = deque([np.array(seed)])
    offers = {}
    finished = set([])
    while len(q) > 0:
        pos = q.popleft()
        pos_t = tuple(pos)
        for a in adjacents:
            a_pos = pos + a
            a_pos_t = tuple(a_pos)
            block_ok = lambda_ok(level[a_pos_t])
            # Conditions for failing a_pos_t
            if a_pos_t in finished:
                continue
            if not block_ok:
                continue
            if not inbounds(a_pos, level):
                continue
            # Otherwise, the block is ok
            finished.add(a_pos_t)
            offers[a_pos_t] = pos_t
            if goal is not None and a_pos_t == goal:
                return offers, finished
            q.append(a_pos)
    return offers, finished

def component(seed, level):
    block = level[seed]
    lambda_ok = lambda x: x == block
    _,f = bfs(seed, level, lambda_ok)
    return f

def get_level_block_pos(level, block, subset=None):
    pos = set([])
    for index, b in np.ndenumerate(level):
        if b == block:
            if subset is None:
                pos.add(index)
            elif pos in subset:
                pos.add(index)
    return pos

def get_def_locations(level, block):
    locs = get_level_block_pos(level, block)
    print(locs)
    new_locs = set([])
    for loc in locs:
        top = np.array(loc) + np.array(facing_to_dir["up"])
        bot = np.array(loc) + np.array(facing_to_dir["down"])
        # Torch on top
        top_ok = (inbounds(top, level) and level[tuple(top)] == torch_up)
        # Bedrock on bottom
        bot_ok = (inbounds(bot, level) and level[tuple(bot)] == define_block)
        if top_ok and bot_ok:
            new_locs.add(loc)
    print(new_locs)
    return new_locs

def get_signs_on(locs, signs):
    return [s for s in signs if s.on in locs]

class Context(object):

    def __init__(self, d):
        self.d = d

def parse_contexts(level, signs):
    contexts = {}
    named_contexts = {}
    context_locs = get_def_locations(level, context_block)
    for loc in context_locs:
        print(loc)
        context, name = parse_context(level, loc, signs)
        contexts[loc] = context
        if name is not None:
            named_contexts[name] = context
    return contexts, named_contexts

def prepare_to_parse(level, loc, signs):
    # Get the bedrock that defines the component
    bot = tuple(np.array(loc) + np.array(facing_to_dir["down"]))
    define_component = component(bot, level)
    signs = get_signs_on(define_component, signs)
    return define_component, signs

def parse_context(level, loc, signs):
    define_component, signs = prepare_to_parse(loc)

def parse_patterns(level, signs):
    patterns = {}
    pattern_locs = get_def_locations(level, pattern_block)
    for loc in pattern_locs:
        pattern, name = parse_pattern(level, loc, signs)
        patterns[name] = pattern
    return patterns

def parse_pattern(level, loc, signs):
    define_component, signs = prepare_to_parse(loc)

def parse_fun_calls(level, signs):
    fun_calls = {}
    fun_call_locs = get_def_locations(level, function_block)
    for loc in fun_call_locs:
        fun_call, name = parse_fun_call(level, loc, signs)
        fun_calls[name] = fun_call
    return fun_calls

def parse_fun_call(level, loc, signs):
    define_component, signs = prepare_to_parse(loc)

# 1.
#   a) What is a replacement semantically?
#       - Find matches for pattern A, somehow map pattern B onto pattern A?
#       - Find matches for pattern B under relaxed matching assumptions?
#       - Somehow both?
#   b) How to write down / parse a replacement?
# 2.
#   a) What are interfaces? What happens when we append two patterns with conflicting interfaces?
#       Point and a direction -> normal vector to the mating surface
#   b) How to define interfaces? Is it possible to have an interface which is an entire pattern?
#       Blocks with no collision to define a mating volume?
#       Mateability / interfaces as a property of blocks
# 3. What happens when we find_placements() for a union type? Should be the same as find_placements() for a
#   Kleene star, right?
#       - How to preserve the lazy execution of the union?
#       Danger zone and re-constrain if something is placed in the danger zone
#       Schrodinger's Flagpole
#       Constrain Early or constrain late?
# 4. Define search() patterns more formally!
