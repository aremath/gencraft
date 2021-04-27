import numpy as np
from bidict import bidict
from collections import deque
import itertools
import re

from amulet.api.block import Block
from amulet.api.selection import SelectionBox, SelectionGroup
#from amulet.operations.fill import fill
from amulet import load_level
import amulet_nbt

from src.utils import *

def parse(level_folder, coord_box):
    # World is a amulet data structure
    world = load_level(level_folder)
    coord_start, coord_end = coord_box
    print("Reference: {}".format(coord_start))
    box = SelectionBox(coord_start, coord_end)
    # Level is a np array of Blocks
    level = get_level(world, box)
    coord_offset = np.array(box.min)
    print("Offset: {}".format(coord_offset))
    signs = get_signs(world, box, coord_offset)
    #b = world.get_block(17, 11, 170, "overworld")
    #print(b)
    #print(pattern_block)
    #print(b == pattern_block)
    #signs = read_signs_file("level_test_signs", coord_start)
    #print(signs)
    #parse_contexts(blocks, signs)
    #i = np.array([11,5,169]) - coord_offset
    #block = blocks[tuple(i)]
    #block.properties["facing"] = "east"
    #print(block)
    #print(transform_block(block, rot90x @ rot90x))
    #print(transform_block(transform_block(block, rot90x), rot90x))
    all_locs = get_def_locations(level)
    print("Locs: {}".format(all_locs))
    _ = all_preparse(all_locs, level, signs)
    return level, coord_offset

def all_preparse(all_locs, level, signs):
    # Key: name, value: {name} is a dependency DAG of patterns
    #TODO: how to verify acyclic?
    all_dependencies = {}
    # Key: name, value: preparse components
    # Everything that is required to actually build the required structure
    context_preparse = {}
    pattern_preparse = {}
    funcall_preparse = {}
    for loc_type, loc in all_locs:
        if loc_type == "Context":
            name, define_text, assignments, dependencies = preparse_context(level, loc, signs)
            print("Name: {}".format(name))
            print("Location: {}".format(loc))
            print("Dependencies: {}".format(dependencies))
            all_dependencies[name] = dependencies
            context_preparse[name] = (name, define_text, assignments)
        elif loc_type == "Pattern":
            preparse_pattern(level, loc, signs)
        elif loc_type == "Funcall":
            pass
    return all_dependencies, context_preparse, pattern_preparse, funcall_preparse

nbt_texts = ["Text1", "Text2", "Text3", "Text4"]

def extract_text(text):
    return text[9:-2]

def get_sign_text(e):
    text_list = [e.nbt["utags"][t].value for t in nbt_texts]
    sign_texts = list(map(extract_text, text_list))
    # Now "preparse" to get correct newlines
    new_sign_texts = []
    current_index = 0
    for text in sign_texts:
        if len(text) == 0:
            continue
        if current_index >= len(new_sign_texts):
            new_sign_texts.append(text)
        else:
            new_sign_texts[current_index] += text
        if text[-1] != "_":
            current_index += 1
    return new_sign_texts

def get_signs(level, box, offset):
    signs = []
    for x,y in box.chunk_locations():
        c = level.get_chunk(x,y,"overworld")
        for e in c.block_entities:
            # Get the facing using level.get_block().properties["facing"]
            pos = (e.x, e.y, e.z)
            block = level.get_block(*pos, "overworld")
            if "Text1" in e.nbt["utags"]:
                texts = get_sign_text(e)
                # On is where the sign is semantically
                if "facing" in block.properties:
                    facing = block.properties["facing"].value
                    on = sign_on(pos, facing)
                else:
                    on = pos
                # Sign position should be stored relative to the real array
                level_pos = tuple(np.array(pos) - offset)
                level_on = tuple(np.array(on) - offset)
                print(pos, level_pos)
                s = Sign(level_pos, level_on, texts)
                signs.append(s)
    return signs

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

def read_signs_file(f, reference):
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
    #print(level_array.shape)
    for (x,y,z) in box:
        block = level.get_block(x,y,z,"overworld")
        np_index = tuple(np.array([x,y,z]) - box.min)
        level_array[np_index] = block
    return level_array

def component(seed, level, directions=adjacents):
    block = level[seed]
    lambda_ok = lambda x: x == block
    _,f = bfs(seed, level, lambda_ok, directions)
    return f

def component_size(component, direction):
    """
    Get the overall maximal distance within the given component in the given direction
    """
    # Use dot product to get the correct directional size
    assert len(component) > 0, "Empty component has no size"
    cs = [np.array(c) @ np.array(direction) for c in component]
    return max(cs) - min(cs)

def get_level_block_pos(level, block, subset=None):
    """
    Gets all positions of a given block within the level
    """
    pos = set([])
    for index, b in np.ndenumerate(level):
        if b == block:
            if subset is None:
                pos.add(index)
            elif pos in subset:
                pos.add(index)
    return pos

def get_def_locations(level):
    all_defs = [("Context", context_block), ("Pattern", pattern_block), ("Funcall", function_block)]
    all_locs = set([])
    for def_type, block in all_defs:
        locs = get_level_block_pos(level, block)
        #print("GetDef {}: {}".format(def_type, locs))
        new_locs = set([])
        for loc in locs:
            top = np.array(loc) + np.array(facing_to_dir["up"])
            bot = np.array(loc) + np.array(facing_to_dir["down"])
            # Torch on top
            top_ok = (inbounds(top, level) and level[tuple(top)] == torch_up)
            # Bedrock on bottom
            bot_ok = (inbounds(bot, level) and level[tuple(bot)] == define_block)
            if top_ok and bot_ok:
                new_locs.add((def_type, loc))
            all_locs |= new_locs
    return all_locs

def get_signs_on(locs, signs):
    return [s for s in signs if s.on in locs]

class Context(object):

    def __init__(self, d):
        self.d = d

# For patterns
# 1. Search to get the definition body
# 2. Get all signs on the def body + condense to text
#   -> Identifiers in this text are dependencies (ordered)
# 3. Look for connectors attached to the definition body
# 4. Get the seed for all connectors -> those anonymous contexts are dependencies too

# For Contexts
# 1. Search down from seed to find "definition solid"
# 2. Extract signs from def solid + condense to text
#   -> Identifiers in this text are dependencies
# 3. Determine assignment direction + seed for "assignment surface"
# 4. Search to find assignment surface

# 5. If assignment surface has a neighbor in the assignment direction then
# 6. Search in assignment to find "connector"
# 7. If connector has a neighbor in the assignment surface directions
# 8. Search to find "Replacement surface"
# 9. Verify that the shapes of the replacement surface match the shapes of the assignment surface
#   using the length of the connector

identifier_only = re.compile(r"^[^\d\W]\w*\Z", re.UNICODE)
identifier = re.compile(r"[^\d\W]\w*", re.UNICODE)

# List of identifiers which all 
pre_defines = ["use"]

def get_dependencies(texts, self_name):
    dependencies = set([])
    for t in texts:
        idents = re.findall(identifier, t)
        new_idents = [i for i in idents if i not in pre_defines and i != self_name]
        new_idents = set(new_idents)
        dependencies |= new_idents
    return dependencies

def look_for_one_connector(level, component, block, directions):
    assignment_d = None
    for d in h_adjacents:
        news = set([tuple(np.array(a) + np.array(d)) for a in component]) - component
        news = set([n for n in news if level[n] == block])
        if len(news) != 0:
            if assignment_d is not None:
                return None, "Context: Multiple assignment surfaces: {} @ {}"
            if len(news) != 1:
                return None, "Context: Bad connector: {} @ {}"
            assignment_seed = list(news)[0]
            assignment_d = d
    if assignment_d is None:
        return None, "Context: No assignment surface: {} @ {}"
    return (assignment_d, assignment_seed), ""

def get_blocks_from(level, signs, pos, direction):
    deps = set([])
    blocks = set([])
    current_pos = pos
    block = level[current_pos]
    while block != air:
        if block in sign_blocks:
            s = [s for s in signs if s.on == current_pos or s.pos == current_pos]
            assert len(s) == 1, "Bad sign at {}".format(current_pos)
            sign = s[0]
            deps |= get_dependencies(sign.texts)
        blocks.add(block)
        current_pos = tuple(np.array(current_pos) + np.array(direction))
        block = level[current_pos]
    return blocks, deps

def preparse_context(level, loc, signs):
    """
    Preparse a context to get the salient features. Parsing cannot be done in one pass since the
    context may be using references in other contexts to refer to blocks
    """
    print("Preparse - Loc: {}".format(loc))
    bot = tuple(np.array(loc) + np.array(facing_to_dir["down"]))
    # Search down from seed to find "define component"
    define_component = component(bot, level, directions=set([(0,-1,0)]))
    print("Preparse - Component: {}".format(define_component))
    assert len(define_component) > 0, "Context: Missing definition: ? @ {}".format(loc)
    define_signs = get_signs_on(define_component, signs)
    print("Preparse - Signs: {}".format(define_signs))
    # Combine the sign text in y-coordinate order
    define_signs.sort(key = lambda s: s.on[1])
    define_text = list(itertools.chain.from_iterable([s.text for s in define_signs]))
    if len(define_text) > 0:
        name = define_text[0]
        assert re.match(identifier_only, name) is not None, "Context: Bad name: {} @ {}".format(name, loc)
        assert name not in pre_defines, "Context: Name is already defined: {} @ {}".format(name, loc)
    else:
        name = "anonymous_context_{}".format(loc)
    print("Preparse - Name: {}".format(name))
    # Dependencies from the define component
    dependencies = get_dependencies(define_text, name)
    # Determine assignment direction + seed for "assignment surface"
    block = level[bot]
    r, error = look_for_one_connector(level, define_component, block, h_adjacents)
    assert r is not None, error.format(name, loc)
    assignment_d, assignment_seed = r
    # Look for the assignment surface
    # Assignment plane is perpendicular to the assignment direction
    assignment_plane_d = tuple(-rot90y @ np.array(assignment_d))
    print("Preparse - Assignment D: {}".format(assignment_d))
    print("Preparse - Assignment Plane D: {}".format(assignment_plane_d))
    assignment_ps = set([assignment_plane_d, tuple(-np.array(assignment_plane_d))])
    assignment_plane_ds = assignment_ps | v_adjacents
    assignment_plane = component(assignment_seed, level, directions=assignment_plane_ds)
    # Look for the replacement surface
    r, error = look_for_one_connector(level, assignment_plane | define_component, block, assignment_d)
    if error != "" and error != "Context: No assignment surface: {} @ {}":
        assert False, error.format(name, loc)
    # If there is a replacement surface
    if r is not None:
        print("Preparse - Replacement: {}".format(name))
        r_d, r_seed = r
        replacement_bar = component(r_seed, level, directions=set([assignment_d]))
        replacement_distance = component_size(replacement_bar, r_d)
        # Replacement plane is parallel to assignment plane
        all_but_replacement = replacement_bar | assignment_plane | define_component
        r, error = look_for_one_connector(level, all_but_replacement, block, directions=assignment_ps)
        assert r is not None, error.format(name, loc)
        replacement_d, replacement_seed = r
        replacement_plane = component(replacement_seed, level, directions=assignment_plane_ds)
    else:
        replacement_plane = None
        replacement_distance = None
    # Preparse the replacement / assignment and get additional dependencies
    assignments = list(assignment_plane)
    # Sort the assignments by assignment direction and y-coordinate
    assignments.sort(key = lambda x: (np.array(x) @ np.array(assignment_plane_d), np.array(x) @ np.array((0,-1,0))))
    assignments = {}
    for a in assignments:
        # The block being assigned to
        assign_to = level[tuple(np.array(a) - np.array(assignment_d))]
        if assign_to == air:
            continue
        a_start = tuple(np.array(a) + np.array(assignment_d))
        assign_from, a_deps = get_blocks_from(level, signs, a_start, assignment_d)
        replacing_a = tuple(np.array(a) + replacement_distance * np.array(assignment_d))
        dependencies |= a_deps
        assert level[a] == level[replacing_a]
        replacing_start = tuple(np.array(replacing_a) + np.array(assignment_d))
        replacing, r_deps = get_blocks_from(level, signs, replacing_start, assignment_d)
        assert len(r_deps) == 0, "Context: Can only replace pure blocks: {} @ {}".format(name, loc)
        assignments[a] = (assign_from, replacing)
    return name, define_text, assignments, dependencies

def preparse_pattern(level, loc, signs):
    pass

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
# Use the context order to determine how to concretize patterns / what order

# Sky islands
# 3D Simplex Noise w/ thresholding / Perlin Noise
# Then squash it
# Then peak the y-axis and trail off..
# Distortion with other simplex noise?
#   -> distort the axes
# Foci -> Voronoi? w/ concentrate mass near points
