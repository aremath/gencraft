import random

from inspiration import *
from gencraft import base

def closeto(loc):
    return lambda x: euclidean(x, loc) + random.random(0, 9)

def farm(location):
    l_nearby = base.nearby(location) # Or something like this
    house = base.make_placement(farmhouse, l_nearby)
    fields = []
    for i in random.randrange(0, 5):
        f = base.make_placement(farmhouse, l_nearby, closeto(house.pos))
        fields.append(f)
    extra = random.choice([windmill, barn])
    e = base.make_placement(extra, l_nearby, closeto(house.pos))
    for f in fields:
        base.make_connect(house, f, roads)
    base.make_connect(house, e, roads)

