import os
import numpy as onp
from src.arguments import args
 

def generate_tess():
    # os.system(f'neper -T -n {args.num_grains} -id 1 -domain "cube(1,1,0.3)" -ori "file(data/neper/input.ori)" -o data/neper/domain -format tess,ori')
    os.system(f'neper -T -n {args.num_grains} -id 1 -regularization 0 -domain "cube({args.domain_length},{args.domain_width},{args.domain_height})" \
                -o data/neper/domain -format tess')


def visualize(): 
    # os.system(f'neper -V data/neper/domain.tess -datacellcol ori:"file(data/neper/input.ori)" -datacellcolscheme "ipf(z)" -datacelltrs 0 -print data/neper/img')
    os.system(f'neper -V data/neper/domain.tess -datacellcol id -datacelltrs 0.5 -print data/neper/img')


def post_processing():
    os.system(f'neper -T -loadtess data/neper/domain.tess -statcell x,y,z,vol,facelist -statface x,y,z,area')


def exp():
    generate_tess()
    post_processing()
    # visualize()


def simple():
    os.system(f'neper -T -n 20 -reg 0 -o data/neper/debug/simple -format tess')
    os.system(f'neper -T -loadtess data/neper/debug/simple.tess -statcell facelist,npolylist -statface nfaces')

    # os.system(f'neper -M -rcl 2 -elttype hex data/neper/debug/simple.tess')

    os.system(f'neper -M -rcl 1 -elttype hex data/neper/domain.tess')


def show_part():

    cut = args.domain_width/2.

    # temperature
    # os.system(f'neper -V data/neper/domain.tess -showcell "y>{cut}" -datacellcol real:"file(data/neper/temp)" \
    #     -datacellcolscheme plasma -datacellscale 280:1400 -datacelltrs 0.5 -showcsys 1 -print data/png/temp')

    # os.system(f'neper -V data/neper/domain.tess -showcell "y>{cut}" -datacellcol real:"file(data/neper/temp)" \
    #     -datacellcolscheme plasma -datacelltrs 0.5 -print data/png/temp')

    # phase
    # os.system(f'neper -V data/neper/domain.tess -showedge 0 -showcell "y>{cut}" -datacellcol real:"file(data/neper/phase)" \
    #     -datacellcolscheme plasma -datacellscale 0:1 -datacelltrs 0 -print data/png/phase')

    # ori
    # os.system(f'neper -V data/neper/domain.tess -showedge 0 -datacellcol ori:"file(data/neper/oris)" \
    #     -datacellcolscheme "ipf(z)" -datacelltrs 0 -print data/png/oris')

    # os.system(f'neper -V data/neper/domain.tess -showedge 0 -datacellcol ori:"file(data/neper/oris)" \
    #     -datacellcolscheme "ipf(z)" -datacelltrs 0 -cameracoo {args.domain_length/2.}:{args.domain_width/2.}:2.5 \
    #     -cameralookat {args.domain_length/2.}:{args.domain_width/2.}:{args.domain_height} -print data/png/oris_above')

    # os.system(f'neper -V data/neper/domain.tess -showedge 0 -showcell "y>{cut}" -datacellcol ori:"file(data/neper/oris)" \
    #     -datacellcolscheme "ipf(z)" -datacelltrs 0 -cameracoo {args.domain_length/2.}:-3:{args.domain_height} \
    #     -cameralookat {args.domain_length/2.}:{args.domain_width/2.}:{args.domain_height} -print data/png/oris_side')


if __name__ == "__main__":
    # exp()
    # show_part()
    simple()
