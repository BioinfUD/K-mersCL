import numpy as np
import pyopencl as cl
from sys import argv
from time import time

from utils.read_conversion import file_to_matrix
from utils.superkmer_utils import cut_minimizer_matrix

def extract_superkmers(minimizer_matrix, m=4):
    n_superkmers = 0
    for row in minimizer_matrix:
        print "superkmers: {}".format(str(row))
        for v in row:
            minimizer = (v & 0b11111111111111000000000000000000) >> 18
            pos = (v & 0b00000000000000111111111100000000) >> 8
            size = v & 0b00000000000000000000000011111111
            end =  pos + size
            minimizer_str = str(minimizer)
            print "Valor {} Min {},  ini {}, size {}, end{}".format(v, minimizer, pos, size, end)
            n_superkmers+=1
    return n_superkmers



""" Remove this comment to test with this data """
# Matrix with reads
SR = [
     [3,3,2,2,0,2,0,2,3,3,3,2,0,3,1,1,3,2,2,1,3,1,0,2,2,0,3,2,0,0,1,2,1,3,2,2,1,2,2,1,2,3,2,1,1,3,0,0,3,0,1,0,3,2,1,0,0,2,3,1,2,0,2,1,2,0,0,3,2,2,0,3,3,0,0,2,0,2,1,3,3,2,1,3,1,3,3,0,3,2,0,0,2,3,3,0,2,1,2,2,1,2,2,0,1,2,2,2,3,2,0,2,3,0,0,1,0,1,2,3,2,2,2,3,0,0,1,1,3,2,1,1,1,0,3,0,0,2,0,1,3,2,2,2,0,3,0,0,1,3,1,1,2,2,2,0,0,0,1,1,2,2,2,2,1,3,0,0,3,0,1,1,2,2,0,3,0,0,1,0],
    # [1,0,1,2,2,2,1,2,3,3,1,0,0,1,2,3,1,2,3,0,0,1,2,2,1,1,1,2,3,2,1,2,2,3,1,1,1,2,3,3,2,2,1,0,0,0,1,2,3,1,1,1,0,1,1,2,3,2,2,2,2,1,1,1,1,1,1,1,2,3,2,1,0,0,1,1,1,3,3,2,0,0,0,1,2,1,1,2,3,0,1,0,1,2,2,2,1,2,3,3,1,0,0,1,2,3,1,2,3,0,0,1,2,2,1,1,1,2,3,2,1,2,2,3,1,1,1,2,3,3,2,2,1,0,0,0,1,2,3,1,1,1,0,1,1,2,3,2,2,2,2,1,1,1,1,1,1,1,2,3,2,1,0,0,1,1,1,3,3,2,0,0,0,1,2,1,1,2,3,0],
    # [1,0,1,2,2,2,1,2,3,3,1,0,0,1,2,3,1,2,3,0,0,1,2,2,1,1,1,2,3,2,1,2,2,3,1,1,1,2,3,3,2,2,1,0,0,0,1,2,3,1,1,1,0,1,1,2,3,2,2,2,2,1,1,1,1,1,1,1,2,3,2,1,0,0,1,1,1,3,3,2,0,0,0,1,2,1,1,2,3,0,1,0,1,2,2,2,1,2,3,3,1,0,0,1,2,3,1,2,3,0,0,1,2,2,1,1,1,2,3,2,1,2,2,3,1,1,1,2,3,3,2,2,1,0,0,0,1,2,3,1,1,1,0,1,1,2,3,2,2,2,2,1,1,1,1,1,1,1,2,3,2,1,0,0,1,1,1,3,3,2,0,0,0,1,2,1,1,2,3,0],
    # [1,0,1,2,2,2,1,2,3,3,1,0,0,1,2,3,1,2,3,0,0,1,2,2,1,1,1,2,3,2,1,2,2,3,1,1,1,2,3,3,2,2,1,0,0,0,1,2,3,1,1,1,0,1,1,2,3,2,2,2,2,1,1,1,1,1,1,1,2,3,2,1,0,0,1,1,1,3,3,2,0,0,0,1,2,1,1,2,3,0,1,0,1,2,2,2,1,2,3,3,1,0,0,1,2,3,1,2,3,0,0,1,2,2,1,1,1,2,3,2,1,2,2,3,1,1,1,2,3,3,2,2,1,0,0,0,1,2,3,1,1,1,0,1,1,2,3,2,2,2,2,1,1,1,1,1,1,1,2,3,2,1,0,0,1,1,1,3,3,2,0,0,0,1,2,1,1,2,3,0],
    # [1,0,1,2,2,2,1,2,3,3,1,0,0,1,2,3,1,2,3,0,0,1,2,2,1,1,1,2,3,2,1,2,2,3,1,1,1,2,3,3,2,2,1,0,0,0,1,2,3,1,1,1,0,1,1,2,3,2,2,2,2,1,1,1,1,1,1,1,2,3,2,1,0,0,1,1,1,3,3,2,0,0,0,1,2,1,1,2,3,0,1,0,1,2,2,2,1,2,3,3,1,0,0,1,2,3,1,2,3,0,0,1,2,2,1,1,1,2,3,2,1,2,2,3,1,1,1,2,3,3,2,2,1,0,0,0,1,2,3,1,1,1,0,1,1,2,3,2,2,2,2,1,1,1,1,1,1,1,2,3,2,1,0,0,1,1,1,3,3,2,0,0,0,1,2,1,1,2,3,0]
    ]
h_R2M_G =np.ndarray((len(SR), len(SR[0]))).astype(np.uint32)
h_R2M_G[:] = SR

def getSuperK_M(input_file, output_path, r):
    # h_R2M_G = file_to_matrix(input_file, r)
    global h_R2M_G
    h_R2M_G = h_R2M_G
    # OpenCL things
    contexto = cl.create_some_context()
    cola = cl.CommandQueue(contexto)
    codigo_kernel = open("kernels/getSuperK2_M.cl").read()
    programa = cl.Program(contexto, codigo_kernel).build()
    getSuperK_M = programa.getSuperK_M
    getSuperK_M.set_scalar_arg_dtypes([None, None, np.uint32, np.uint32, np.uint32,np.uint32])
    # Copy data from host to device
    d_R2M_G = cl.Buffer(contexto, cl.mem_flags.COPY_HOST_PTR, hostbuf=h_R2M_G)
    h_R2M_G_test  = np.empty(h_R2M_G.shape).astype(np.uint32)

    # Kernel parameters
    nr = h_R2M_G.shape[0]
    r = h_R2M_G.shape[1]
    m = 7
    k = 31
    nmk = k - m + 1
    # Execution parameters
    # X = (((k-1)//32) + 1)*32
    X = nmk
    print "Ejecutando X hilos, X: {}".format(X)
    rango_global = (X, nr)
    rango_local = (X, 1)

    # Output matrix
    nm = r - m + 1;
    h_counters = np.empty((nr,1)).astype(np.uint32)
    d_counters = cl.Buffer(contexto, cl.mem_flags.WRITE_ONLY, h_counters.nbytes)
    # Execution
    getSuperK_M(cola, rango_global, rango_local, d_R2M_G, d_counters, nr, r, k, m)
    print "Execution finished, copying data from device to host memory"
    cola.finish()
    cl.enqueue_copy(cola, h_counters, d_counters)
    cl.enqueue_copy(cola, h_R2M_G, d_R2M_G)
    # Cut the output matrix based on counters
    print "Cutting the matrix based on available superkmers"
    minimizer_matrix = cut_minimizer_matrix(h_R2M_G, h_counters)
   # Debug data
    print "Retrieved data"
    print h_counters
    print h_R2M_G
    del(h_R2M_G)
    print "Writing superkmers to disk"
    extract_superkmers(minimizer_matrix, m=4)


if __name__ == "__main__":
    """
    Execution:
    python getSuperK_M filename.fasta outputpath N
    filename <- Fasta file with reads
    outputpath <- Folder where the buckets will be placed (must exists)
    N <- Read size (optional)
    """
    input_file = ""
    output_path = ""
    r = 180
    getSuperK_M(input_file, output_path, r)
