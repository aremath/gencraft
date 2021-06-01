import numpy as np
from bidict import bidict
from collections import deque, defaultdict
import itertools
import re
from functools import reduce
import pickle

from amulet.api.block import Block
from amulet.api.selection import SelectionBox, SelectionGroup
#from amulet.operations.fill import fill
from amulet import load_level
import amulet_nbt

from src.utils import *

def load_level_from_world(level_folder, coord_box):
    # World is a amulet data structure
    world = load_level(level_folder)
    coord_start, coord_end = coord_box
    box = SelectionBox(coord_start, coord_end)
    # Level is a np array of Blocks
    coord_offset = np.array(box.min)
    print("Offset: {}".format(coord_offset))
    level = level_from_world(world, box)
    signs = get_signs(world, box, coord_offset)
    return level, signs

def load_level_from_npy(level_path, signs_path):
    with open(level_path, "rb") as level_file:
        level = np.load(level_file, allow_pickle=True)
    with open(signs_path, "rb") as signs_file:
        signs = pickle.load(signs_file)
    return level, signs

def save_level(level, signs, level_path, signs_path):
    with open(level_path, "wb") as level_file:
        np.save(level_file, level, allow_pickle=True)
    with open(signs_path, "wb") as signs_file:
        pickle.dump(signs, signs_file)

def parse(level, signs):
    print("Level shape: {}".format(level.shape))
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
    #all_locs = get_def_locations(level)
    #print("Locs: {}".format(all_locs))
    info = prepare_parse(level, signs)
    return level

def all_preparse(all_locs, level, signs):
    # Key: name, value: {name} is a dependency DAG of patterns
    #TODO: how to verify acyclic?
    all_dependencies = {}
    # Key: name, value: preparse components
    # Everything that is required to actually build the required structure
    context_preparse = {}
    pattern_preparse = {}
    for loc_type, loc in all_locs:
        if loc_type == "Context":
            name, define_text, assignments, dependencies = preparse_context(level, loc, signs)
            print("Context Name: {}".format(name))
            print("Location: {}".format(loc))
            print("Dependencies: {}".format(dependencies))
            all_dependencies[name] = dependencies
            context_preparse[name] = (name, define_text, assignments)
        elif loc_type == "Pattern":
            name, define_text, predefinition, interior_signs, dependencies = preparse_pattern(level, loc, signs)
            print("Pattern Name: {}".format(name))
            print("Location: {}".format(loc))
            print("Dependencies: {}".format(dependencies))
            all_dependencies[name] = dependencies
            pattern_preparse[name] = (name, define_text, predefinition, interior_signs)
        elif loc_type == "Funcall":
            pass
    return all_dependencies, context_preparse, pattern_preparse

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

    def __eq__(self,other):
        # Two signs can't occupy the same location
        return self.pos == other.pos

def sign_on(pos, facing):
    if facing is None:
        return pos
    else:
        return tuple(np.array(pos) - np.array(facing_to_dir[facing]))

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

def level_from_world(world, box):
    shape = np.array(box.max) - np.array(box.min)
    level_array = np.ndarray(shape, dtype="object")
    #print(level_array.shape)
    for (x,y,z) in box:
        block = world.get_block(x,y,z,"overworld")
        np_index = tuple(np.array([x,y,z]) - box.min)
        level_array[np_index] = block
    return level_array

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

def mk_assignments(level, signs):
    # Key: assignment pos, Value: {pos} on assigment component
    assignment_pos = {}
    # Key: assignment pos, Value: assignment sign
    assignment_signs = {}
    # Set of all positions that belong to an assignment
    assignment_set = set([])
    assign_locs = [s for s in signs if s.text[0] == ":=" or s.text[0] == "=:"]
    for sign in assign_locs:
        loc = sign.on
        # It's only a valid assignment if there's a torch
        if level[tuple(np.array(loc) + np.array([0,1,0]))] == torch_up:
            assignment_signs = {}
            loc_component = component(loc, level, define_block)
            assert len(loc_component) > 0
            assignment_pos[loc] = loc_component
            assignment_set |= loc_component
    return assignment_pos, assignment_signs, assignment_set

def mk_namespaces(level, signs, assignment_set):
    # Key: namespace def position, Value: (min, max) bounding box for namespace
    namespaces = {}
    for pos in get_level_block_pos(level, namespace_block):
        print(pos)
        l = 0
        namespace_component = None
        for a in adjacents:
            p = tuple(np.array(pos) + np.array(a))
            bad_signs = [s for s in signs if s.on == p]
            if len(bad_signs) > 0:
                continue
            a_component = component(p, level, define_block)
            a_component = a_component - assignment_set
            if len(a_component) > 0:
                l += 1
                namespace_component = a_component
        assert l == 1, "Namespace at {} has multiple components".format(pos)
        bbox = component_bbox(namespace_component)
        namespaces[pos] = bbox
    namespaces["global"] = ((0,0,0), level.shape)
    return namespaces

def mk_namespace_graph(namespaces):
    g = {}
    for n, b in namespaces.items():
        g[n] = set([])
        for n2, b2 in namespaces.items():
            if bbox_in(b, b2):
                if n != n2:
                    g[n].add(n2)
    return g

def namespace_assignments(assignments, namespaces):
    """ Produce mappings from namespace to [assignments] and 
        assignment pos to [namespaces] """
    # Key: Namespace pos, Value: [assignment pos] in that namespace
    namespace_as = defaultdict(list)
    # Key: Assignment pos, Value: [namespaces pos] the assignment is in
    a_namespaces = {}
    g = mk_namespace_graph(namespaces)
    # Get an ordering of namespaces based on containment
    o = get_parse_order(g)
    for a in assignments.keys():
        # All namespaces the assignment is in
        ins = [n for n, b in namespaces.items() if pos_in_bbox(a, b)]
        ordered_ins = []
        for o in order:
            if o in ins:
                ordered_ins.append(o)
        assert len(ordered_ins) > 0
        namespace_as[ordered_ins[-1]].append(a)
        a_namespaces[a] = ordered_ins
    return namespace_as, a_namespaces

def prepare_parse(level, signs):
    assignment_pos, assignment_signs, assignment_set = mk_assignments(level, signs)
    namespaces = mk_namespaces(level, signs, assignment_set)
    namespace_as, a_namespaces = namespace_assignments(assignment_pos, namespaces)
    ParseInfo(level, signs, assignment_pos, assingment_signs, assignment_set, namespaces, namespace_as, a_namespaces)

def get_signs_on(locs, signs):
    return [s for s in signs if s.on in locs]

def get_sign_at(loc, signs):
    ss = [s for s in signs if s.pos == loc]
    return ss

def match_ids_with_vals(id_posns, val_posns):
    matched = []
    for id_pos in id_posns:
        # Sort by distance, closest one is the desired one
        val_posns.sort(key=lambda x: np.linalg.norm(np.array(id_pos) - np.array(x)))
        matched.append((id_pos, val_posns[0]))
    return matched

def get_ident(pos, info):
    ss = get_sign_at(pos, info.signs)
    if len(ss) == 0:
        assert info.level[pos] != air
        return info.level[pos]
    else:
        assert len(ss) == 1
        sign = ss[0]
        assert re.match(identifier_only, sign.text[0]) is not None, "Bad identifier at {}".format(pos)
        return sign.text[0]

# Get the preparsed level AST
def parse_assignment(info, pos, direction):
    # Direction is unknown, so should be 
    assert direction is None, "Bad assignment: {}".format(pos)
    s = info.assignment_signs[pos]
    assign_component = info.assignment_pos[pos]
    assign_sign_facing = np.array(s.on) - np.array(s.pos)
    # Assign direction points from the ident towards the assignment body
    if s.text[0] == ":=":
        assign_direction = assign_sign_facing @ rot90y
    elif s.text[0] == "=:":
        assign_direction = assign_sign_facing @ (-rot90y)
    else
        assert False, "Bad assignment: {}".format(pos)
    ident_posns = neighbors(assign_component, directions=[tuple(-assign_direction)])
    # All the assignments created by this component
    ident_posns = [i for i in ident_posns if level[i] != air]
    val_posns = neighbors(assign_component, directions=[tuple(assign_direction)])
    matched_posns = match_ids_with_vals(ident_posns, val_posns)
    # Identifier -> pos
    pre_mapping = [(get_ident(k, info), v) for k,v in matched_posns]
    # Identifier -> parsed thing
    mapping = {}
    for ident, v_pos in pre_mapping:
        val = parse_any(info, v_pos, assignment_direction)
        mapping[ident] = val
    # Do the extra work to get the priority and the require_str
    assign_signs = get_signs_on(assignment_component, info.signs)
    # Remove all the signs which are keys
    for k, v in matched_posns:
        ss = get_sign_at(k)
        if len(ss) == 1:
            assign_signs.remove(ss[0])
    # Sort them by decreasing y-coordinate
    assign_signs.sort(key=lambda x: -x.on[1])
    assert assign_signs[0] == s, "Bad assignment: {}".format(pos)
    priority = int(s.text[1])
    # List of strings
    rest_text = s.text[2:] + reduce(lambda x,y: x + y, [s.text for s in assign_signs[1:]])
    # String
    require_str = reduce(lambda x,y: x + y, rest_text)
    assignments = [PreAssignment(ident, priority, require_str, val) for ident, val in mapping.items()]
    return assignments

def parse_namespace(info, pos, direction):
    assignments = []
    for a in info.namespaces_as[pos]:
        assignments.extend(parse_assignment(info, a, None))
    return PreNamespace(assignments)

def parse_blockfunction(info, pos, direction):
    output_start = tuple(np.array(pos) + np.array(direction))
    ok = lambda x: (x!=air and x!=define_block)
    _, component1 = bfs(output_start, info.level, ok, directions=[direction])
    boundary = neighbors(component1, directions=[direction])
    assert len(boundary) == 1, "Bad blockfunction: {}".format(pos)
    boundary = boundary[0]
    component2 = set()
    if level[boundary] == define_block:
        input_start = tuple(np.array(boundary) + np.array(direction))
        _, component2 = bfs(input_start, info.level, ok, directions=[direction])
    blocks_out = set(map(lambda x: level[x], component1))
    blocks_in = set(map(lambda x: level[x], component2))
    return PreBlockFunction(blocks_out, blocks_in)

def parse_pattern(info, pos, direction):
    pass

def parse_lambda(info, pos, direction):
    inputs_start = tuple(np.array(pos) + np.array(direction))
    inputs_component = component(inputs_start, info.level, directions=[direction])
    ident_posns = neighbors(assign_component)
    ident_posns = [i for i in ident_posns if level[i] != air]
    # Get them in order
    ident_posns.sort(key=lambda x: np.array(x) @ direction)
    idents = list(map(lambda x: get_ident(x, info), ident_posns))
    expr_pos = neighbors(inputs_component, directions=[direction])
    assert len(expr_pos) == 1, "Bad lambda: {}".format(pos)
    expr_pos = expr_pos[0]
    val = parse_any(info, expr_pos, direction)
    return PreLambda(idents, expr)

parsers = {
    blockfunction_block: parse_blockfunction,
    pattern_block: parse_pattern,
    namespace_block: parse_namespace,
    lambda_block: parse_lambda
}

def parse_any(info, pos, direction):
    block_type = level[pos]
    assert block_type in parsers, "No parser for the block type at: {}".format(pos)
    parser = parsers[block_type]
    val = parser(info, pos, direction)

class Context(object):

    def __init__(self, d):
        self.d = d

# PREPARSING
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

def sign_to_str(s):
    t = s.text
    return reduce(lambda x,y: x+y, t)

def get_dependencies(texts, self_name=""):
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
        #TODO: inefficient - instead create a map of positions to signs
        signs = [s for s in signs if s.on == current_pos or s.pos == current_pos]
        if len(signs) > 0:
            assert len(signs) == 1, "Bad sign at {}".format(current_pos)
            sign = signs[0]
            sign_deps = get_dependencies(sign.text)
            #print("Context: Sign in assign!")
            #print(sign.text)
            #print(sign_deps)
            deps |= sign_deps
            blocks.add(sign_to_str(sign))
        else:
            blocks.add(block)
        current_pos = tuple(np.array(current_pos) + np.array(direction))
        block = level[current_pos]
    return blocks, deps

def prepare_preparse(level, loc, signs, directions=adjacents):
    #print("Preparse - Loc: {}".format(loc))
    bot = tuple(np.array(loc) + np.array(facing_to_dir["down"]))
    # Search down from seed to find "define component"
    define_component = component(bot, level, directions=directions)
    #print("Preparse - Component: {}".format(define_component))
    assert len(define_component) > 0, "Context: Missing definition: ? @ {}".format(loc)
    define_signs = get_signs_on(define_component, signs)
    #print("Preparse - Signs: {}".format(define_signs))
    # Combine the sign text in y-coordinate order
    define_signs.sort(key = lambda s: -s.on[1])
    define_text = list(itertools.chain.from_iterable([s.text for s in define_signs]))
    return define_component, define_text

def preparse_context(level, loc, signs):
    """
    Preparse a context to get the salient features. Parsing cannot be done in one pass since the
    context may be using references in other contexts to refer to blocks
    """
    define_component, define_text = prepare_preparse(level, loc, signs, directions=set([(0,-1,0)]))
    if len(define_text) > 0:
        name = define_text[0]
        assert re.match(identifier_only, name) is not None, "Context: Bad name: {} @ {}".format(name, loc)
        assert name not in pre_defines, "Context: Name is already defined: {} @ {}".format(name, loc)
    else:
        name = "anonymous_context_{}".format(loc)
    #print("Preparse - Name: {}".format(name))
    # Dependencies from the define component
    dependencies = get_dependencies(define_text, name)
    # Determine assignment direction + seed for "assignment surface"
    block = define_block
    r, error = look_for_one_connector(level, define_component, block, h_adjacents)
    assert r is not None, error.format(name, loc)
    assignment_d, assignment_seed = r
    # Look for the assignment surface
    # Assignment plane is perpendicular to the assignment direction
    assignment_plane_d = tuple(-rot90y @ np.array(assignment_d))
    #print("Preparse - Assignment D: {}".format(assignment_d))
    #print("Preparse - Assignment Plane D: {}".format(assignment_plane_d))
    assignment_ps = set([assignment_plane_d, tuple(-np.array(assignment_plane_d))])
    assignment_plane_ds = assignment_ps | v_adjacents
    assignment_plane = component(assignment_seed, level, directions=assignment_plane_ds)
    # Look for the replacement surface
    r, error = look_for_one_connector(level, assignment_plane | define_component, block, assignment_d)
    if error != "" and error != "Context: No assignment surface: {} @ {}":
        assert False, error.format(name, loc)
    # If there is a replacement surface
    if r is not None:
        #print("Preparse - Replacement: {}".format(name))
        r_d, r_seed = r
        replacement_bar = component(r_seed, level, directions=set([assignment_d]))
        replacement_distance = component_size(replacement_bar, r_d) + 1
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
    assignment_locs = list(assignment_plane)
    # Sort the assignments by assignment direction and y-coordinate
    assignment_locs.sort(key = lambda x: (np.array(x) @ np.array(assignment_plane_d), np.array(x) @ np.array((0,-1,0))))
    assignments = {}
    # Get the assignments
    for a in assignment_locs:
        # The block being assigned to
        assign_to = level[tuple(np.array(a) - np.array(assignment_d))]
        if assign_to == air or assign_to == define_block:
            continue
        #print("Preparse - Assign {}".format(a))
        a_start = tuple(np.array(a) + np.array(assignment_d))
        assign_from, a_deps = get_blocks_from(level, signs, a_start, assignment_d)
        dependencies |= a_deps
        if replacement_distance is not None:
            replacing_a = tuple(np.array(a) + replacement_distance * np.array(assignment_d))
            assert level[a] == level[replacing_a]
            replacing_start = tuple(np.array(replacing_a) + np.array(assignment_d))
            replacing, r_deps = get_blocks_from(level, signs, replacing_start, assignment_d)
            # Freeze for hashing
            replacing = frozenset(replacing)
            assert len(r_deps) == 0, "Context: Can only replace pure blocks: {} @ {}".format(name, loc)
        else:
            replacing = set([])
        assignments[assign_to] = (assign_from, replacing)
    #print("Preparse - Done with {}".format(name))
    return name, define_text, assignments, dependencies

def preparse_pattern(level, loc, signs):
    define_component, define_text = prepare_preparse(level, loc, signs)
    assert len(define_text) > 0, "Pattern: Bad name: @ {}".format(loc)
    # Unlike contexts, Patterns can be named things like f(x)
    # Find the first match
    name_text = define_text[0]
    try:
        name = re.findall(identifier, name_text)[0]
    except IndexError:
        assert False, "Pattern: Bad name: {} @ {}".format(name, loc)
    assert name not in pre_defines, "Pattern: Name is already defined: {} @ {}".format(name, loc)
    # Dependencies from the define component
    dependencies = get_dependencies(define_text, name)
    # Also get dependencies from attached anonymous contexts
    #TODO: Want to only get attachments OUTSIDE the definition so that
    # purple wool can be used inside patterns
    attach_points = set([n for n in neighbors(define_component) if level[n] == attach_block])
    for a in attach_points:
        context_seed = [n for n in neighbors(set([a])) if level[n] == define_block and n not in define_component]
        assert len(context_seed) == 1, "Pattern: Multiple contexts from attachment: {} @ {}".format(name, loc)
        context_component = component(context_seed[0], level)
        context_start = [n for n in neighbors(context_component, [(0,1,0)]) if level[n] == context_block]
        assert len(context_start) == 1, "Pattern: Invalid anonymous context: {} @ {}".format(name, loc)
        assert level[tuple(np.array(context_start[0]) + np.array((0,1,0)))] == torch_up, \
            "Pattern: Invalid anonymous context: {} @ {}".format(name, loc)
        dependencies.add("anonymous_context_{}".format(context_start[0]))
    # Get the blocks within the prepattern as well as preinterfaces
    # The array of blocks within bounds
    mins, shape = extent(define_component)
    mins = np.array(mins)
    predefinition = np.empty(shape, dtype="object")
    indices = set()
    for i, _ in np.ndenumerate(predefinition):
        level_index = tuple(mins + np.array(i))
        if level_index not in define_component:
            put_block = level[level_index]
        else:
            put_block = air
        predefinition[i] = put_block
        indices.add(i)
    interior_signs = set([s for s in signs if s.on in indices])
    # Interfaces: Key - direction, Value - block array
    return name, define_text, predefinition, interior_signs, dependencies

def prepare_to_parse(level, loc, signs):
    # Get the bedrock that defines the component
    bot = tuple(np.array(loc) + np.array(facing_to_dir["down"]))
    define_component = component(bot, level)
    signs = get_signs_on(define_component, signs)
    return define_component, signs

def default_bf(block):
    return BlockFunction(set([(block, frozenset([block, air]))]))

def default_pattern(block):
    bf = default_bf(block)
    a = np.empty((1,1,1), dtype=object)
    a[(0,0,0)] = bf
    return JustPattern(a, {})

def lookup_block(block, contexts):
    for c in contexts:
        if block in c:
            return c[block]
    return default_pattern(block)

def parse_context(preparse, contexts, patterns, fvars=None):
    name, define_text, assignments = preparse
    print(name)
    print(assignments)
    # Block -> Pattern
    d = {}
    # Assignment order is preserved
    for k, v in assignments.items():
        d[k] = eval(v, locals=(patterns | locals()))
        if isinstance(k, str):
            pass
        #elif ...
        #    pass

def parse_pattern(preparse, contexts, patterns):
    name, define_text, predefinition, interior_signs = preparse

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
