import pyopencl as cl
import numpy as np
from time import time
import scipy
from python_functions.utils import memory_usage_psutil

def sequential_function(SK_2_64,cSK_4, S):
    print "Memory Usage %s" % memory_usage_psutil()
    h_SK_4 = np.ndarray((S, cSK_4)).astype(np.uint8)
    SK_2_64  = SK_2_64.reshape((S*(cSK_4/32)))
    for i in range(0, cSK_4):
        for j in range(0, S):
            h_SK_4[j][i] = (SK_2_64[(j*(cSK_4/32))+(i/32)] >> np.uint8(62-((i%32)*2))) & np.uint8(3)
    print "Memory Usage %s" % memory_usage_psutil()
    return h_SK_4

# BASE 10 (64bits) to BASE 4
# Random Data
pairs = [(16, 500000), (48, 500000), (16, 1000000), (48, 1000000)]
for K, S in pairs:
    total_bases = (S) * (K/32 + 1)
    h_SK_2_64 = np.random.randint(4**(31), size=total_bases).astype(np.uint64).reshape((S, (K/32 + 1)))
    # Parameters
    k = K
    cSK_4 = h_SK_2_64.shape[1]*32 # Ncols output matrix
    S = h_SK_2_64.shape[0] # Total kmers

    # Output matrix
    h_SK_4 = np.ndarray((S, cSK_4)).astype(np.uint8)

    print "#############  KMERS BASE 2 (64 bits - 32 bases) A BASE 4 (8bits por base) K {} S {}#################".format(K,S)
    # OpenCL Things
    contexto = cl.create_some_context()
    cola = cl.CommandQueue(contexto)
    codigo_kernel = open("kernels/B642N.cl").read()
    programa = cl.Program(contexto, codigo_kernel).build()
    B642N = programa.B642N
    B642N.set_scalar_arg_dtypes([None, None, np.uint32, np.uint32])

    # Copy input data from host to device
    d_SK_2_64 = cl.Buffer(contexto, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_SK_2_64)

    # Output buffer
    d_SK_4 = cl.Buffer(contexto, cl.mem_flags.WRITE_ONLY, h_SK_4.nbytes)

    # Execution Range
    rango_global = (cSK_4, S)
    # Kernel Execution
    print "Executing kernel"
    t1 = time()
    B642N(cola, rango_global, None, d_SK_4, d_SK_2_64, cSK_4, S)
    cola.finish()
    print "Kernel took {} seconds in the execution".format(time()-t1)
    # Retrieve output
    cl.enqueue_copy(cola, h_SK_4, d_SK_4)
    print "Executing sequential function"
    t1 = time()
    h_SK_4_s = sequential_function(h_SK_2_64, cSK_4, S)
    print "Function took {} seconds in the execution".format(time()-t1)
    # Borro columnas que sobran
    #h_SK_4 = scipy.delete(h_SK_4,range(cSK_4-k),1)
    if (h_SK_4==h_SK_4_s).all():
        print "Matrix are equal"
    else:
        print "Matrix not equal"
