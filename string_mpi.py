"""
MPI capable

Bresenham
# Bresenham line algorithm
# https://gist.github.com/flags/1132363
"""
import os
import sys

import numpy as np

import scipy
import scipy.sparse
import scipy.sparse.linalg

from mpi4py import MPI

###################################################
RADIUS     = 96
# N          = 192
# N          = 960
N          = 640
## rn Nedges should be perfectly divisible by numproc
Nedges     = int ( 0.5 * N * (N-1) )
# IMG_PKG    = "gs_image.npz"
# IMG_PKG    = "gs_qimage.npz"
IMG_PKG    = "psr_image.npz"
_bimg      = os.path.basename ( IMG_PKG )[:-len(".npz")]
SAVE_FILE  = f"strings_{N:d}_{RADIUS:d}_{_bimg}.npz"
THREAD_START = 0
PER_PROCESS  = 48 # updated in the text
## someone has to do the math
###################################################

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
    hooks = np.array (
        [
            [np.cos(np.pi*2*i/n), np.sin(np.pi*2*i/n)] for i in range(n)
        ]
    )
    hooks = (radius * hooks).astype(int)
    edge_codes = []
    row_ind = []
    col_ind = []
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
    # creating the edge-pixel adjecency matrix:
    # rows are indexed with pixel codes, columns are indexed with edge codes.
    sparse = scipy.sparse.csr_matrix(
        ([1.0]*len(row_ind), (row_ind, col_ind)), 
        shape=(
            (2*radius+1)*(2*radius+1), len(edge_codes)
        )
    )
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

if __name__ == "__main__":
    ## setup MPI
    comm   = MPI.COMM_WORLD
    rank   = comm.Get_rank ()
    size   = comm.Get_size ()

    PER_PROCESS = int ( Nedges / size )

    ## make sparse matrix
    SPARSE_A, hooks, edge_codes = build_arc_adjecency_matrix(N, RADIUS)

    if rank == 0:
        print (" ... built sparse ...")

    if SPARSE_A.shape[1] != Nedges:
        raise RuntimeError("Nedges arithmetic failed")

    ## load the image
    img      = np.load (IMG_PKG) ['img']

    ## build the  image vector
    SPARSE_B = build_image_vector(img, RADIUS)

    if rank == 0:
        print (" ... built image ...")
        print (" solving linear system ... ")

    if rank == 0:
        print (f" Number of spokes   = {N:d}")
        print (f" Radius             = {RADIUS:d}")
        print (f" A.shape            = {SPARSE_A.shape}")
        print (f" Number of edges    = {len(edge_codes):d}")
        print (f" B.shape            = {SPARSE_B.shape}")

        print (f" PER_PROCESS        = {PER_PROCESS:d}")
        print (f" MPI size           = {size:d}")

    sys.stdout.flush ()


    ####################################################
    current_thread   = THREAD_START
    ##
    loss             = 0.0
    work_loss        = np.zeros ((PER_PROCESS,), dtype=np.float32)
    state_thread     = np.zeros ((N,), dtype=np.uint32)
    state_art        = np.zeros ((Nedges), dtype=np.float32)
    state_loss       = np.zeros ((Nedges), dtype=np.float32)
    ####
    ## work division
    this_start = rank * PER_PROCESS
    this_stop  = (rank + 1 ) * PER_PROCESS
    this_edges = edge_codes [ this_start:this_stop ]
    ##
    for i in range ( N ):
        ## broadcast state in root
        ## this also synchronizes
        comm.Bcast ( state_art, root=0 )
        ## broadcast current_thread
        current_thread   = comm.bcast ( current_thread, root=0 )

        ## work starts
        ## ## compute individual losses
        for iedge, edge in enumerate ( this_edges ):
            art_edge         = this_start + iedge
            ### default case
            work_loss[iedge] = np.nan
            ### 
            ### if testedge is connected to current_thread
            ### and if there is no string already
            if current_thread in edge and state_art[art_edge] == 0.:
                ## flip art
                state_art[art_edge]    = 1.0
                ## compute loss
                ##### this is the costly computation
                model  = SPARSE_A.dot ( state_art ).reshape ((-1, 1))
                model  /= model.max ()
                err    = model - SPARSE_B
                loss   = np.mean ( np.power ( err, 2 ) )
                ##### this is the costly computation
                ## record loss
                work_loss[iedge]    = loss
                ## unflip art
                state_art[art_edge]    = 0.0

        ## gather individual losses
        ## in root
        comm.Gather ( work_loss, state_loss, root=0 )

        ## do selection in root
        if rank == 0:
            min_loss = np.nanargmin ( state_loss )
            new_edge = edge_codes [ min_loss ]

            ## check if thread already was placed
            ## but this is error condition
            if state_art[min_loss] == 1.0:
                raise RuntimeError ("Selected thread is already in the art.\n"f" selected edge={min_loss:d} edge={new_edge} at current_thread={current_thread:d}")

            ## safety check
            if current_thread not in new_edge:
                raise RuntimeError ("New edge does not current thread")

            ## update state
            print ( f" At iteration {i:d} with current_thread={current_thread:d}, edge {new_edge} added at min_loss={min_loss:d}, initial start_art={state_art[min_loss]:.0f}")
            state_art[min_loss] = 1.0    

            ## update current_thread
            if current_thread == new_edge[0]:
                current_thread = new_edge[1]
            else:
                current_thread = new_edge[0]

            # keep track of thread
            state_thread[i] = current_thread

            ## flush stdout
            sys.stdout.flush ()

    if rank == 0:
        print(" ... done ... ")
        sys.stdout.flush ()
        np.savez (
            SAVE_FILE, 
            threads=state_thread,
            state=state_art,
            loss=state_loss,
            img=img,
            radius=RADIUS,
            nspoke=N,
            Nedges=Nedges
        )
