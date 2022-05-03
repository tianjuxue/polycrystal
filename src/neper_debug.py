'''
这个文件在不久以后应该会被废弃
The file will be deprecated later
'''
import os
import numpy as onp
from src.arguments import args


def run():
    os.system(f'neper -T -n 20 -reg 0 -o data/neper/debug/domain -format tess,ori,obj')
    os.system(f'neper -T -loadtess data/neper/debug/domain.tess -statcell facelist,npolylist -statface nfaces')
    os.system(f'neper -M data/neper/debug/domain.tess -rcl 1 -elttype hex')
 

if __name__ == "__main__":
    run()
