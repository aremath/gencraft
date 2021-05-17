import numpy as np
from bidict import bidict
from collections import deque

from amulet.api.block import Block
from amulet.api.selection import SelectionBox, SelectionGroup
#from amulet.operations.fill import fill
from amulet import load_level
import amulet_nbt

# Deprecated now that Block.from_string_blockstate exists
def blockstate_to_block(s):
    namespace, base_name, properties = Block.parse_blockstate_string(s)
    print(properties)
    return Block(namespace, base_name, properties)

define_block = Block.from_string_blockstate("universal_minecraft:bedrock[infiniburn=false]")
context_block = Block.from_string_blockstate("universal_minecraft:wool[color=blue]")
pattern_block = Block.from_string_blockstate("universal_minecraft:wool[color=red]")
function_block = Block.from_string_blockstate("universal_minecraft:wool[color=lime]")
attach_block = Block.from_string_blockstate("universal_minecraft:wool[color=purple]")
torch_up = Block.from_string_blockstate("universal_minecraft:torch[facing=up]")
air = Block.from_string_blockstate("universal_minecraft:air")

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

h_adjacents = set([
    (1,0,0),
    (-1,0,0),
    (0,0,1),
    (0,0,-1)
    ])

v_adjacents = set([
    (0,1,0),
    (0,-1,0)
    ])

# Map the facing of a sign or stairs to a direction
facing_to_dir = bidict({
    "up": (0,1,0),
    "down": (0,-1,0),
    "east": (1,0,0),
    "west": (-1,0,0),
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

def inbounds(pos, level):
    min_coord = np.array([0,0,0])
    if all(pos >= min_coord) and all(pos < np.array(level.shape)):
        return True
    else:
        return False

def neighbors(component, directions=adjacents):
    n = [tuple(np.array(c) + np.array(d)) for c in component for d in directions]
    # Only elements not in component
    n = set(n) - component
    return n

def extent(component):
    xs = [c[0] for c in component]
    ys = [c[1] for c in component]
    zs = [c[2] for c in component]
    mins = (min(xs), min(ys), min(zs))
    maxs = (max(xs), max(ys), max(zs))
    return mins, tuple(np.array(maxs) - np.array(mins))

#TODO: Dijkstras for goal searching
def bfs(seed, level, lambda_ok, directions=adjacents, goal=None):
    assert inbounds(seed, level)
    q = deque([np.array(seed)])
    offers = {}
    finished = set([seed])
    while len(q) > 0:
        pos = q.popleft()
        pos_t = tuple(pos)
        for a in directions:
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

def find_roots(dependencies):
    all_objects = set(dependencies.keys())
    on = set([])
    for o in all_objects:
        on |= dependencies[o]
    roots = [o for o in all_objects if o not in on]
    return roots

def dependency_dfs(root, parse_order, dependencies, visited_all, cycle_visited):
    if root in visited_all:
        return
    assert root not in cycle_visited, "Circular dependency: {}".format(root)

    cycle_visited.add(root)
    for d in dependencies[root]:
        if d in dependencies:
            dependency_dfs(d, parse_order, dependencies, visited_all, cycle_visited)

    visited_all.add(root)
    parse_order.insert(0, root)

# DFS-based parse order
def get_parse_order(dependencies):
    roots = find_roots(dependencies)
    parse_order = []
    visited = set()
    all_nodes = set(dependencies.keys())
    while visited != all_nodes:
        root = roots.pop()
        dependency_dfs(root, parse_order, dependencies, visited, set())
    parse_order.reverse()
    return parse_order

        
        

