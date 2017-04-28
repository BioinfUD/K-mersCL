import numpy as np
import pyopencl as cl
from sys import argv
from time import time

from utils.read_conversion import file_to_matrix
from utils.superkmer_utils import cut_minimizer_matrix, extract_superkmers


"""
Remove this comment to test with this data
# Matrix with reads
SR = [
   [1,0,1,2,2,2,1,2,3,3,1,0,0,1,2,3,1,2,3,0,0,1,2,2,1,1,1,2,3,2,1,2,2,3,1,1,1,2,3,3,2,2,1,0,0,0,1,2,3,1,1,1,0,1,1,2,3,2,2,2,2,1,1,1,1,1,1,1,2,3,2,1,0,0,1,1,1,3,3,2,0,0,0,1,2,1,1,2,3,0,1,0,1,2,2,2,1,2,3,3,1,0,0,1,2,3,1,2,3,0,0,1,2,2,1,1,1,2,3,2,1,2,2,3,1,1,1,2,3,3,2,2,1,0,0,0,1,2,3,1,1,1,0,1,1,2,3,2,2,2,2,1,1,1,1,1,1,1,2,3,2,1,0,0,1,1,1,3,3,2,0,0,0,1,2,1,1,2,3,0]
    # [1,0,1,2,2,2,1,2,3,3,1,0,0,1,2,3,1,2,3,0,0,1,2,2,1,1,1,2,3,2,1,2,2,3,1,1,1,2,3,3,2,2,1,0,0,0,1,2,3,1,1,1,0,1,1,2,3,2,2,2,2,1,1,1,1,1,1,1,2,3,2,1,0,0,1,1,1,3,3,2,0,0,0,1,2,1,1,2,3,0,1,0,1,2,2,2,1,2,3,3,1,0,0,1,2,3,1,2,3,0,0,1,2,2,1,1,1,2,3,2,1,2,2,3,1,1,1,2,3,3,2,2,1,0,0,0,1,2,3,1,1,1,0,1,1,2,3,2,2,2,2,1,1,1,1,1,1,1,2,3,2,1,0,0,1,1,1,3,3,2,0,0,0,1,2,1,1,2,3,0],
    # [1,0,1,2,2,2,1,2,3,3,1,0,0,1,2,3,1,2,3,0,0,1,2,2,1,1,1,2,3,2,1,2,2,3,1,1,1,2,3,3,2,2,1,0,0,0,1,2,3,1,1,1,0,1,1,2,3,2,2,2,2,1,1,1,1,1,1,1,2,3,2,1,0,0,1,1,1,3,3,2,0,0,0,1,2,1,1,2,3,0,1,0,1,2,2,2,1,2,3,3,1,0,0,1,2,3,1,2,3,0,0,1,2,2,1,1,1,2,3,2,1,2,2,3,1,1,1,2,3,3,2,2,1,0,0,0,1,2,3,1,1,1,0,1,1,2,3,2,2,2,2,1,1,1,1,1,1,1,2,3,2,1,0,0,1,1,1,3,3,2,0,0,0,1,2,1,1,2,3,0],
    # [1,0,1,2,2,2,1,2,3,3,1,0,0,1,2,3,1,2,3,0,0,1,2,2,1,1,1,2,3,2,1,2,2,3,1,1,1,2,3,3,2,2,1,0,0,0,1,2,3,1,1,1,0,1,1,2,3,2,2,2,2,1,1,1,1,1,1,1,2,3,2,1,0,0,1,1,1,3,3,2,0,0,0,1,2,1,1,2,3,0,1,0,1,2,2,2,1,2,3,3,1,0,0,1,2,3,1,2,3,0,0,1,2,2,1,1,1,2,3,2,1,2,2,3,1,1,1,2,3,3,2,2,1,0,0,0,1,2,3,1,1,1,0,1,1,2,3,2,2,2,2,1,1,1,1,1,1,1,2,3,2,1,0,0,1,1,1,3,3,2,0,0,0,1,2,1,1,2,3,0],
    # [1,0,1,2,2,2,1,2,3,3,1,0,0,1,2,3,1,2,3,0,0,1,2,2,1,1,1,2,3,2,1,2,2,3,1,1,1,2,3,3,2,2,1,0,0,0,1,2,3,1,1,1,0,1,1,2,3,2,2,2,2,1,1,1,1,1,1,1,2,3,2,1,0,0,1,1,1,3,3,2,0,0,0,1,2,1,1,2,3,0,1,0,1,2,2,2,1,2,3,3,1,0,0,1,2,3,1,2,3,0,0,1,2,2,1,1,1,2,3,2,1,2,2,3,1,1,1,2,3,3,2,2,1,0,0,0,1,2,3,1,1,1,0,1,1,2,3,2,2,2,2,1,1,1,1,1,1,1,2,3,2,1,0,0,1,1,1,3,3,2,0,0,0,1,2,1,1,2,3,0],
    # [1,0,1,2,2,2,1,2,3,3,1,0,0,1,2,3,1,2,3,0,0,1,2,2,1,1,1,2,3,2,1,2,2,3,1,1,1,2,3,3,2,2,1,0,0,0,1,2,3,1,1,1,0,1,1,2,3,2,2,2,2,1,1,1,1,1,1,1,2,3,2,1,0,0,1,1,1,3,3,2,0,0,0,1,2,1,1,2,3,0,1,0,1,2,2,2,1,2,3,3,1,0,0,1,2,3,1,2,3,0,0,1,2,2,1,1,1,2,3,2,1,2,2,3,1,1,1,2,3,3,2,2,1,0,0,0,1,2,3,1,1,1,0,1,1,2,3,2,2,2,2,1,1,1,1,1,1,1,2,3,2,1,0,0,1,1,1,3,3,2,0,0,0,1,2,1,1,2,3,0]
    ]
h_R2M_G =np.ndarray((len(SR), len(SR[0]))).astype(np.uint32)
h_R2M_G[:] = SR
"""
def getSuperK_M(input_file, output_path, r):
    h_R2M_G = file_to_matrix(input_file, r)
    # OpenCL things
    contexto = cl.create_some_context()
    cola = cl.CommandQueue(contexto)
    codigo_kernel = open("kernels/getSuperK_M.cl").read()
    programa = cl.Program(contexto, codigo_kernel).build()
    getSuperK_M = programa.getSuperK_M
    getSuperK_M.set_scalar_arg_dtypes([None, None, None, np.uint32, np.uint32, np.uint32,np.uint32])
    # Copy data from host to device
    d_R2M_G = cl.Buffer(contexto, cl.mem_flags.COPY_HOST_PTR, hostbuf=h_R2M_G)
    h_R2M_G_test  = np.empty(h_R2M_G.shape).astype(np.uint32)

    # Kernel parameters
    nr = h_R2M_G.shape[0]
    r = h_R2M_G.shape[1]
    m = 4
    k = 31
    # Execution parameters
    X = (((256/k)*k - 1)/32 + 1)*32
    print "Ejecutando X hilos, X: {}".format(X)
    rango_global = (X, nr)
    rango_local = (X, 1)

    # Output matrix
    nm = r - m + 1;
    h_TMP = np.empty((nr, nm)).astype(np.uint32)
    d_TMP = cl.Buffer(contexto, cl.mem_flags.WRITE_ONLY, h_TMP.nbytes)
    h_counters = np.empty((nr,1)).astype(np.uint32)
    d_counters = cl.Buffer(contexto, cl.mem_flags.WRITE_ONLY, h_counters.nbytes)
    # Execution
    getSuperK_M(cola, rango_global, rango_local, d_R2M_G, d_counters, d_TMP, nr, r, k, m)
    print "Execution finished, copying data from device to host memory"
    cola.finish()
    cl.enqueue_copy(cola, h_TMP, d_TMP)
    cl.enqueue_copy(cola, h_counters, d_counters)
    cl.enqueue_copy(cola, h_R2M_M, d_R2M_G)
    # Cut the output matrix based on counters
    print "Cutting the matrix based on available superkmers"
    minimizer_matrix = cut_minimizer_matrix(h_R2M_G, h_counters)
    print "Writing superkmers to disk"
    extract_superkmers = extract_superkmers(minimizer_matrix, input_file, output_path, m=4)

    print "Retrieved data"
    print h_counters
    print h_TMP[0]
    print h_R2M_G[0]
    del(h_R2M_G)




if __name__ == "__main__":
    """
    Execution:
    python getSuperK_M filename.fasta outputpath N
    filename <- Fasta file with reads
    outputpath <- Folder where the buckets will be placed (must exists)
    N <- Read size (optional)
    """
    input_file = argv[1]
    output_path = argv[2]
    r = int(argv[3]) if argv[3] else None
    getSuperK_M(input_file, output_path, r)
