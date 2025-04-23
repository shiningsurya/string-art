"""
shows MPI result
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

import matplotlib.pyplot as plt
plt.style.use('dark_background')

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
    ax.imshow ( fi, aspect='equal', cmap=cmap, origin='upper', interpolation='none' )

def get_args ():
    import argparse
    agp  = argparse.ArgumentParser ('string_show')
    add  = agp.add_argument
    add ('sol_npz', help='Output string_mpi')
    add ('-s','--save', help='Save png to', dest='opng', default=None)
    return agp.parse_args ()

if __name__ == "__main__":
    args   = get_args ()

    pkg    = np.load ( args.sol_npz )

    ##
    N      = int ( pkg['nspoke'] )
    RADIUS = int ( pkg['radius'] )
    img    = pkg['img']
    x      = pkg['state']

    fig    = plt.figure ('strings', figsize=(8,8))
    axii, axrr = fig.subplots ( 1, 2, sharex=True, sharey=True , subplot_kw={'box_aspect':1.0})

    #####
    SPARSE_A, hooks, edge_codes, bd = build_arc_adjecency_matrix(N, RADIUS)
    SPARSE_B = build_image_vector(img, RADIUS)

    #####
    axii.imshow ( img, aspect='equal', cmap='gray', origin='upper' )
    axii.set_title ('What is read')


    imshower ( axrr, reconstruct ( x, SPARSE_A, RADIUS), cmap='coolwarm' )
    axrr.set_title ('Solved')

    ## what is min, max, mean, std
    # print (f" {x.min():.2f} {x.max():.2f} {x.mean():.2f} {x.std():.2f}")
    print (f" Solution state statistics")
    print (f" \t Zeros: {np.sum(x==0):.2f}")
    print (f" \t Ones:  {np.sum(x==1):.2f}")
    print (f" \t Total: {x.size:d}")

    if args.opng:
        fig.savefig (args.opng, dpi=300, bbox_inches='tight')
    else:
        plt.show ()
