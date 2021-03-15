import numpy as np

#from amulet.api.block import blockstate_to_block
from amulet.api.selection import SelectionBox, SelectionGroup
#from amulet.operations.fill import fill
from amulet import load_level

def parse(level_folder, coord_box):
    l, c = get_level(level_folder, coord_box)
    pass

def get_level(level_folder, coord_box):
    level = load_level(level_folder)
    coord_start, coord_end = coord_box
    print(coord_end - coord_start)
    box = SelectionBox(coord_start, coord_end)
    level_array = np.ndarray((coord_end - coord_start), dtype="object")
    print(level_array.shape)
    for (x,y,z) in box:
        block = level.get_block(x,y,z,"overworld")
        np_index = tuple(np.array([x,y,z]) - coord_start)
        level_array[np_index] = block
    return level_array, coord_start
