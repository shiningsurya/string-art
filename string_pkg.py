"""

BGA from here
__author__ = "Bigzhao Tan"
__email__ = "tandazhao@email.szu.edu.cn"
__version__ = "0.0.1"

Bresenham
# Bresenham line algorithm
# https://gist.github.com/flags/1132363

binary genetic algorithm is failing 
need to use greedy approach

there would be N**2 loss evaluations
"""
import os
import sys
import random
import numpy as np
import pandas as pd
import scipy
import scipy.sparse
import scipy.sparse.linalg
# from imageio import imread, imsave

import math
from collections import defaultdict

import multiprocessing

from tqdm import tqdm

from functools import partial

###################################################
NPOL       = 2048 # number of DNAs
MAX_ROUND  = 100  # 
VERBOSE    = 50
NCPU       = os.cpu_count()   # ncpu
RADIUS     = 84
POWER      = 0.15
N          = 148
#N          = 32
# N          = 8
output_prefix = 'n148_c48'
###################################################

import matplotlib.pyplot as plt
fig    = plt.figure ('strings', figsize=(8,8))

# ((axii, axrr), (axuu, axvv)) = fig.subplots ( 2, 2, sharex=True, sharey=True )
axii, axrr = fig.subplots ( 2, 1, sharex=True, sharey=True , subplot_kw={'box_aspect':1.0})

class bresenham:
    def __init__(self, start, end):
        self.start = list(start)
        self.end = list(end)
        self.path = []

        self.steep = abs(self.end[1]-self.start[1]) > abs(self.end[0]-self.start[0])

        if self.steep:
            self.start = self.swap(self.start[0],self.start[1])
            self.end = self.swap(self.end[0],self.end[1])

        if self.start[0] > self.end[0]:
            _x0 = int(self.start[0])
            _x1 = int(self.end[0])
            self.start[0] = _x1
            self.end[0] = _x0

            _y0 = int(self.start[1])
            _y1 = int(self.end[1])
            self.start[1] = _y1
            self.end[1] = _y0

        dx = self.end[0] - self.start[0]
        dy = abs(self.end[1] - self.start[1])
        error = 0
        derr = dy/float(dx)

        ystep = 0
        y = self.start[1]

        if self.start[1] < self.end[1]: ystep = 1
        else: ystep = -1

        for x in range(self.start[0],self.end[0]+1):
            if self.steep:
                self.path.append((y,x))
            else:
                self.path.append((x,y))

            error += derr

            if error >= 0.5:
                y += ystep
                error -= 1.0

    def swap(self,n1,n2):
        return [n2,n1]

# Bresenham circle algorithm
# https://www.daniweb.com/programming/software-development/threads/321181/python-bresenham-circle-arc-algorithm
def circle(radius):
    # init vars
    switch = 3 - (2 * radius)
    points = set()
    x = 0
    y = radius
    # first quarter/octant starts clockwise at 12 o'clock
    while x <= y:
        # first quarter first octant
        points.add((x,-y))
        # first quarter 2nd octant
        points.add((y,-x))
        # second quarter 3rd octant
        points.add((y,x))
        # second quarter 4.octant
        points.add((x,y))
        # third quarter 5.octant
        points.add((-x,y))
        # third quarter 6.octant
        points.add((-y,x))
        # fourth quarter 7.octant
        points.add((-y,-x))
        # fourth quarter 8.octant
        points.add((-x,-y))
        if switch < 0:
            switch = switch + (4 * x) + 6
        else:
            switch = switch + (4 * (x - y)) + 10
            y = y - 1
        x = x + 1
    return points

def image(filename, size):
    # img = imresize(rgb2gray(imread(filename)), (size, size))
    # img = imresize ( imread ( filename, as_gray=True ), ( size, size ) )
    img = imresize ( imread ( filename )[...,0], ( size, size ) )
    """
    binarize the image

    for gs
    i already grayscaled it
    """
    # bimg = np.zeros ( (size,size), dtype=bool )
    # bimg[:] = img[...,0]
    # bimg = np.array ( img, dtype=bool)
    # print ( bimg.shape )
    # bimg[img>0.3] = True
    # return bimg
    return img


def build_arc_adjecency_matrix(n, radius):
    print("building sparse adjecency matrix")
    hooks = np.array([[math.cos(np.pi*2*i/n), math.sin(np.pi*2*i/n)] for i in range(n)])
    hooks = (radius * hooks).astype(int)
    edge_codes = []
    row_ind = []
    col_ind = []
    bd      = dict ()
    for i, ni in enumerate(hooks):
        for j, nj in enumerate(hooks[i+1:], start=i+1):
            edge_codes.append((i, j))
            pixels = bresenham(ni, nj).path
            edge = []
            for pixel in pixels:
                pixel_code = (pixel[1]+radius)*(radius*2+1) + (pixel[0]+radius)
                edge.append(pixel_code)
            row_ind += edge
            col_ind += [len(edge_codes)-1] * len(edge)
            ## dictionary mappings
            bd[i] = j
    # creating the edge-pixel adjecency matrix:
    # rows are indexed with pixel codes, columns are indexed with edge codes.
    sparse = scipy.sparse.csr_matrix(([1.0]*len(row_ind), (row_ind, col_ind)), shape=((2*radius+1)*(2*radius+1), len(edge_codes)))
    return sparse, hooks, edge_codes, bd


def build_circle_adjecency_matrix(radius, small_radius):
    print("building sparse adjecency matrix")
    edge_codes = []
    row_ind = []
    col_ind = []
    pixels = circle(small_radius)
    for i, cx in enumerate(range(-radius+small_radius+1, radius-small_radius-1, 1)):
        for j, cy in enumerate(range(-radius+small_radius+1, radius-small_radius-1, 1)):
            edge_codes.append((i, j))
            edge = []
            for pixel in pixels:
                px, py = cx+pixel[0], cy+pixel[1]
                pixel_code = (py+radius)*(radius*2+1) + (px+radius)
                edge.append(pixel_code)
            row_ind += edge
            col_ind += [len(edge_codes)-1] * len(edge)
    # creating the edge-pixel adjecency matrix:
    # rows are indexed with pixel codes, columns are indexed with edge codes.
    sparse = scipy.sparse.csr_matrix(([1.0]*len(row_ind), (row_ind, col_ind)), shape=((2*radius+1)*(2*radius+1), len(edge_codes)))
    hooks = []
    return sparse, hooks, edge_codes


def build_image_vector(img, radius):
    # representing the input image as a sparse column vector of pixels:
    assert img.shape[0] == img.shape[1]
    img_size = img.shape[0]
    row_ind = []
    col_ind = []
    data = []
    for y, line in enumerate(img):
        for x, pixel_value in enumerate(line):
            global_x = x - img_size//2
            global_y = y - img_size//2
            pixel_code = (global_y+radius)*(radius*2+1) + (global_x+radius)
            data.append(float(pixel_value))
            row_ind.append(pixel_code)
            col_ind.append(0)
    sparse_b = scipy.sparse.csr_matrix((data, (row_ind, col_ind)), shape=((2*radius+1)*(2*radius+1), 1))
    return sparse_b


def reconstruct(x, sparse, radius, brightness_correction=1.0):
    b_approx = sparse.dot(x * brightness_correction)
    b_image = b_approx.reshape((2*radius+1, 2*radius+1))
    b_image = np.clip(b_image, 0, 255)
    return b_image

def imshower ( ax, f, cmap='gray' ):
    fi  = f.copy()
    fi[fi == 0.] = np.nan
    ax.imshow ( fi, aspect='auto', cmap=cmap, origin='lower', interpolation='none' )

def GS_SOLVE_METHOD ( state, i ):
    """
        bare bones method
    """
    ## if the edge is already active
    ## quick return
    if state[i] > 0:
        # print (f" early-return at state[{i:d}]={self.state[i]:d}")
        return (i, np.inf)

    ## if not then compute y
    y     = state.copy ()
    y[i,0]  = 1

    
    ## model computation
    model = SPARSE_A.dot ( y )
    model /= model.max()
    err   = model - SPARSE_B[:,0]
    loss  = np.mean ( np.power ( err, 2 ) )

    ## update
    # print (f" At hook={i:d} loss={loss:.2f}")
    return (i, loss)

if __name__ == "__main__":

    #####
    SPARSE_A, hooks, edge_codes, bd = build_arc_adjecency_matrix(N, RADIUS)

    #####
    img      = np.load ('gs_image.npz')['img']
    img      = np.power ( img, POWER )
    SPARSE_B = build_image_vector(img, RADIUS)

    #####
    axii.imshow ( img, aspect='auto', cmap='gray', origin='lower' )
    axii.set_title ('What is read')

    #####
    # finding the solution, a weighting of edges:
    print("solving linear system")

    N       = SPARSE_A.shape[1]
    STATE   = np.zeros ( (N,1), dtype=np.uint8 )
    MINLOSSITER = np.zeros ( N, dtype=np.float32 )

    for ite in tqdm ( range (N), desc='Hook', unit='h' ):
        ## evaluation
        with multiprocessing.Pool ( NCPU ) as p:
            ret = p.map ( partial ( GS_SOLVE_METHOD, STATE ), range ( N ) ) 
        # ret = [ GS_SOLVE_METHOD ( STATE, i ) for i in range (N) ]

        ## post-eval
        ## ## pick best
        # imin    = sorted ( ret, key=lambda k : k[1] )[0]
        imin    = min ( ret, key=lambda k : k[1] )
        iarg    = imin[0]
        mloss   = imin[1]
        # print (f" At {ite:d} mloss={mloss:.3f} for hook={iarg:d}")
        ## if we do not find any good edges
        if mloss >= np.inf:
            continue
        MINLOSSITER[ite] = mloss
        ## ## update state
        STATE [ iarg ] = 1.0
    # x = result[0]
    x   = STATE
    # np.savez ("result.npz", x=x)
    print("done")
    # x, istop, itn, r1norm, r2norm, anorm, acond, arnorm = result

    imshower ( axrr, reconstruct ( x, SPARSE_A, RADIUS), cmap='coolwarm' )
    axrr.set_title ('Solved')

    ## what is min, max, mean, std
    # print (f" {x.min():.2f} {x.max():.2f} {x.mean():.2f} {x.std():.2f}")
    print (f" Solution state statistics")
    print (f" \t Zeros: {np.sum(x==0):.2f}")
    print (f" \t Ones:  {np.sum(x==1):.2f}")
    print (f" \t Total: {x.size:d}")

    # plt.show () # afd
    fig.savefig (f"{output_prefix}_sol.png", dpi=300, bbox_inches='tight')
    np.savez (f"{output_prefix}_sol.npz", x=x )
