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
from src.types import *

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
    #print(info.assignment_signs)
    global_expr, _ = parse_namespace(info, "global", None)
    global_val = global_expr.compile({})
    return level

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

    def __eq__(self, other):
        # Two signs can't occupy the same location
        return self.pos == other.pos
    
    @property
    def full_text(self):
        """
        All the sign text as one string
        """
        return reduce(lambda x,y: x + y, [t for t in self.text])

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
    assign_signs = [s for s in signs if s.text[0] == ":=" or s.text[0] == "=:"]
    for sign in assign_signs:
        loc = sign.on
        # It's only a valid assignment if there's a torch
        if level[tuple(np.array(loc) + np.array([0,1,0]))] == torch_up:
            assignment_signs[loc] = sign
            loc_component = component(loc, level, define_block)
            assert len(loc_component) > 0
            assignment_pos[loc] = loc_component
            assignment_set |= loc_component
    return assignment_pos, assignment_signs, assignment_set

def mk_namespaces(level, signs, assignment_set):
    # Key: namespace def position, Value: (min, max) bounding box for namespace
    namespaces = {}
    for pos in get_level_block_pos(level, namespace_block):
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
    order = get_parse_order(g)
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
    return ParseInfo(level, signs, assignment_pos, assignment_signs, assignment_set, namespaces, namespace_as, a_namespaces)

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
        #print(val_posns)
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
    #print("Parsing Assignment at {}".format(pos))
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
    else:
        assert False, "Bad assignment: {}".format(pos)
    ident_posns = neighbors(assign_component, directions=[tuple(-assign_direction)])
    # All the assignments created by this component
    ident_posns = list(filter(lambda x: info.level[x] != air, ident_posns))
    val_posns = neighbors(assign_component, directions=[tuple(assign_direction)])
    val_posns = list(filter(lambda x: info.level[x] != air, val_posns))
    #print("Ident:", ident_posns)
    #print("Val:", val_posns)
    matched_posns = match_ids_with_vals(ident_posns, val_posns)
    assert len(matched_posns) > 0, "Bad assignment: {}".format(pos)
    #print(matched_posns)
    # Identifier -> pos
    pre_mapping = [(get_ident(k, info), v) for k,v in matched_posns]
    # Identifier -> parsed thing
    mapping = {}
    for ident, v_pos in pre_mapping:
        val, finished = parse_any(info, v_pos, assign_direction)
        mapping[ident] = val
    # Do the extra work to get the priority and the require_str
    assign_signs = get_signs_on(assign_component, info.signs)
    #print(assign_signs)
    # Remove all the signs which are keys
    for k, v in matched_posns:
        ss = get_sign_at(k, info.signs)
        if len(ss) == 1:
            #print(k)
            assign_signs.remove(ss[0])
    #print(assign_signs)
    # Sort them by decreasing y-coordinate
    assign_signs.sort(key=lambda x: -x.on[1])
    assert assign_signs[0] == s, "Bad assignment: {}".format(pos)
    priority = int(s.text[1])
    # List of strings
    rest_text = s.text[2:] + reduce(lambda x,y: x + y, [s.text for s in assign_signs[1:]], [])
    # String
    require_str = reduce(lambda x,y: x + y, rest_text, "")
    assignments = [Assignment(ident, priority, require_str, val, pos) for ident, val in mapping.items()]
    return assignments

def parse_expr_graph(info, pos, directions=adjacents, finished=None):
    offers = {}
    # pos -> [(direction, pos)]
    edges = defaultdict(list)
    # pos -> expr
    nodes = {}
    q = deque([pos])
    if finished is None:
        finished = set()
    finished.add(pos)
    while len(q) > 0:
        pos = q.popleft()
        parser = get_parser(info.level[pos])
        p_d = get_parse_direction(pos, info.level, finished)
        p_expr, p_finished = parser(info, pos, p_d)
        # Add to finished the nodes that were parsed by the subparser
        finished |= p_finished
        nodes[pos] = p_expr
        for d in directions:
            d_pos = tuple(np.array(pos) + np.array(d))
            # Failure conditions
            if d_pos in finished:
                # If it's already been parsed, add edges
                if d_pos in nodes:
                    edges[pos].append((d, d_pos))
                    edges[d_pos].append((tuple(-np.array(d)), pos))
                continue
            if not parse_ok(info.level[d_pos]):
                continue
            if not inbounds(d_pos, info.level):
                continue
            # Otherwise, add it
            finished.add(d_pos)
            offers[d_pos] = pos
            q.append(d_pos)
    return offers, finished, Graph(nodes, edges)

def expr_graph_to_list(graph, root):
    out_pos = []
    finished = {root}
    assert root in graph.nodes
    q = deque([root])
    while len(q) > 0:
        pos = q.popleft()
        out_pos.append(pos)
        for n in graph.edges[pos]:
            if n not in finished:
                q.append(n)
                finished.add(n)
    return list(map(lambda x: graph.nodes[x], out_pos))

def parse_namespace(info, pos, direction):
    #print("Parsing Namespace at {}".format(pos))
    assignments = []
    for a in info.namespaces_as[pos]:
        assignments.extend(parse_assignment(info, a, None))
    return Namespace(assignments, pos), set([pos])

def parse_blockfunction(info, pos, direction):
    #print("Parsing BlockFunction at {}".format(pos))
    output_start = tuple(np.array(pos) + np.array(direction))
    finished = set([output_start])
    _, component1, g1 = parse_expr_graph(info, output_start, [direction], finished)
    boundary = neighbors(component1, directions=[direction])
    assert len(boundary) == 1, "Bad blockfunction: {}".format(pos)
    boundary = next(iter(boundary))
    exprs_in = []
    if info.level[boundary] == define_block:
        input_start = tuple(np.array(boundary) + np.array(direction))
        finished.add(input_start)
        _, component2, g2 = parse_expr_graph(info, input_start, [direction], finished)
        exprs_in = expr_graph_to_list(g2, input_start)
    else:
        component2 = set()
    exprs_out = expr_graph_to_list(g1, output_start)
    finished = component1 | component2 | set([pos, boundary])
    return BlockFunctionDef(exprs_out, exprs_in, pos), finished

def parse_pattern(info, pos, direction):
    #print("Parsing Pattern at {}".format(pos))
    start = tuple(np.array(pos) + np.array(direction))
    o, f, g = parse_expr_graph(info, start, finished=set([pos]))
    return ExprGraph(g, pos), f

def parse_lambda(info, pos, direction):
    #print("Parsing Lambda at {}".format(pos))
    inputs_start = tuple(np.array(pos) + np.array(direction))
    inputs_component = component(inputs_start, info.level, directions=[direction])
    ident_posns = neighbors(inputs_component)
    ident_posns = [i for i in ident_posns if info.level[i] != air]
    # Get them in order
    ident_posns.sort(key=lambda x: np.array(x) @ direction)
    idents = list(map(lambda x: get_ident(x, info), ident_posns))
    expr_pos = neighbors(inputs_component, directions=[direction])
    assert len(expr_pos) == 1, "Bad lambda: {}".format(pos)
    expr_pos = next(iter(expr_pos))
    expr, finished = parse_any(info, expr_pos, direction)
    return Lambda(idents, expr, pos), inputs_component | finished

def parse_string(info, pos, direction):
    #print("Parsing String at {}".format(pos))
    signs_on = [s for s in info.signs if s.on == pos]
    assert len(signs_on) == 1
    sign = signs_on[0]
    t = sign.full_text
    return String(t, pos), set([pos, sign.pos])

def parse_funcall(info, pos, direction):
    #print("Parsing FunCall at {}".format(pos))
    fn_start = tuple(np.array(pos) + np.array(direction))
    finished = set([fn_start])
    _, component1, g1 = parse_expr_graph(info, fn_start, [direction], finished)
    fn_expr = expr_graph_to_list(g1, fn_start)
    assert len(fn_expr) == 1
    boundary = neighbors(component1, directions=[direction])
    assert len(boundary) == 1, "Bad blockfunction: {}".format(pos)
    boundary = boundary[0]
    args = []
    if info.level[boundary] == define_block:
        args_start = tuple(np.array(boundary) + np.array(direction))
        finished.add(args_start)
        _, component2, g2 = parse_expr_graph(info, args_start, [direction], finished)
        args = expr_graph_to_list(g2, args_start)
    else:
        component2 = set()
    finished = component1 | component2 | set([pos, boundary])
    return FunCall(fn, args, pos), finished

def parse_union(info, pos, direction):
    #print("Parsing Union at {}".format(pos))
    start = tuple(np.array(pos) + np.array(direction))
    finished = set([start])
    _, component, g = parse_expr_graph(info, start, [direction], finished)
    exprs = expr_graph_to_list(g, start)
    return UnionDef(exprs, pos), component | set([pos])

def parse_block(info, pos, direction):
    #print("Parsing Block at {}".format(pos))
    return BlockExpr(info.level[pos], pos), set([pos])

parsers = {
    blockfunction_block: parse_blockfunction,
    pattern_block: parse_pattern,
    namespace_block: parse_namespace,
    lambda_block: parse_lambda,
    string_block: parse_string,
    funcall_block: parse_funcall,
    union_block: parse_union
}

# Binding order. Higher binds more loosely
priorities = {
    blockfunction_block: 2,
    pattern_block: 2,
    namespace_block: 2,
    lambda_block: 2,
    string_block: 1,
    funcall_block: 2,
    union_block: 2
}

def get_parser(block):
    if block in parsers:
        return parsers[block]
    else:
        return parse_block

#TODO: precedence
def get_parse_direction(pos, level, finished):
    ds = []
    priorities = []
    for n in neighbors(set([pos])):
        # Skip neighbors that have already been parsed
        if n in finished:
            continue
        n_block = level[n]
        if n_block in priorities:
            block_priority = priorities[n_block]
        # Parser is a Block, => highest binding order
        else:
            block_priority = 0
        ds.append(tuple(np.array(n) - np.array(pos)))
        priorities.append(block_priority)
    # Sort them by priority
    dps = list(zip(ds, priorities))
    dps.sort(key=lambda x: x[1])
    # If there are no valid neighbors, return None (which may be ok for some parsers)
    if len(dps) == 0:
        return None
    else:
        initial_d, initial_p = dps[0]
        if len(dps) == 1:
            return initial_d
        # Also should return None if there are multiple neighbors at the same priority
        # as the highest priority neighbor. Parsing is ambiguous
        elif dps[1][1] == initial_p:
            return None
        else:
            return initial_d

def parse_any(info, pos, direction):
    block_type = info.level[pos]
    assert block_type in parsers, "No parser for the block type at: {}".format(pos)
    parser = parsers[block_type]
    expr, finished = parser(info, pos, direction)
    return expr, finished

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

def default_bf(block):
    return BlockFunction(set([(block, frozenset([block, air]))]))

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
