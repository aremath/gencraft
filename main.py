import sys
import numpy as np
from argparse import ArgumentParser

from src import parser

def get_args(arg_list):
    argparser = ArgumentParser(description="Welcome to gencraft!")
    argparser.add_argument("dir", type=str, help="A Minecraft level folder")
    argparser.add_argument("--start", type=int, nargs=3)
    argparser.add_argument("--end", type=int, nargs=3)
    args = argparser.parse_args()
    args.start = np.array(args.start)
    args.end = np.array(args.end)
    return args

def main(arg_list):
    args = get_args(arg_list)
    l,c = parser.parse(args.dir, (args.start, args.end))

if __name__ == "__main__":
    main(sys.argv[1:])
