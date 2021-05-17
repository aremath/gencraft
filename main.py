import sys
import numpy as np
from argparse import ArgumentParser

from src import parser

def get_args(arg_list):
    argparser = ArgumentParser(description="Welcome to gencraft!")
    subparsers = argparser.add_subparsers(dest="subparser")

    # Reading the level from the world
    parser_world = subparsers.add_parser("world", help="Read the level data from a world")
    # Positional
    parser_world.add_argument("world", type=str, help="A Minecraft level folder")
    parser_world.add_argument("--start", type=int, nargs=3)
    parser_world.add_argument("--end", type=int, nargs=3)

    # Reading the level from a saved .npy
    parser_npy = subparsers.add_parser("npy", help="Read the level data from a saved .npy")
    # Positional
    parser_npy.add_argument("level_file", type=str, help="A saved .npy file with level data")
    parser_npy.add_argument("sign_file", type=str, help="A pickled signs data structure")

    # Reading a level from a world and writing back to dumped objects
    parser_convert = subparsers.add_parser("convert", help="Convert minecraft to .npy")
    parser_convert.add_argument("world", type=str, help="A Minecraft level folder")
    parser_convert.add_argument("level_file", type=str, help="Where to save the level data")
    parser_convert.add_argument("sign_file", type=str, help="Where to save the sign data")
    parser_convert.add_argument("--start", type=int, nargs=3)
    parser_convert.add_argument("--end", type=int, nargs=3)

    args = argparser.parse_args()
    return args

def main(arg_list):
    args = get_args(arg_list)
    if args.subparser == "world":
        level, signs = parser.load_level_from_world(args.world, \
                (np.array(args.start), np.array(args.end)))
        l = parser.parse(level, signs)
    elif args.subparser == "npy":
        level, signs = parser.load_level_from_npy(args.level_file, args.sign_file)
        l = parser.parse(level,signs)
    elif args.subparser == "convert":
        level, signs = parser.load_level_from_world(args.world, \
                (np.array(args.start), np.array(args.end)))
        parser.save_level(level, signs, args.level_file, args.sign_file)
    else:
        assert False, "Bad subparser: {}".format(args.subparser)

if __name__ == "__main__":
    main(sys.argv[1:])
