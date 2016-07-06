import pyopencl as cl
import numpy as np
from time import time
from python_functions.utils import memory_usage_psutil
import psutil, os

# BASE 4 A CARACTERES
#Memoria para lectura en host
# Matriz con KMERS
def massive_or (elements):
    el = elements[0]
    for i in elements[1:]:
        el |= i
    return el

def sequential_conversion(SK_4, cSK_4, S, cTMP):
    SK_4 =  SK_4.reshape((SK_4.shape[0]*SK_4.shape[1]))
    SK_2_64=np.ndarray((rTMP * cTMP/32)).astype(np.uint64)
    TMP=np.ndarray((rTMP * cTMP)).astype(np.uint64)
    # Fill TMP matrix
    print "Filling TMP matrix"
    for i in range(0, cTMP):
        for j in range(0, S):
            # Fill TMP
            if (i <= cSK_4-1) and (j<=S-1):
                TMP[j*cTMP + (i + cTMP - cSK_4)] = SK_4[(j*cSK_4)+i];

            if (i < (cTMP-cSK_4)) and (j<= S-1):
                #print "Filling with 0, i {} j {}".format(i,j)
                TMP[j*cTMP + i] = 0;
            # Left Shift
            if (i <= cTMP-1) and i>=(cTMP-cSK_4) and (j<=S-1):
                TMP[(j*cTMP)+i] = TMP[(j*cTMP)+i] << np.uint64(64 - 2*((i%32)+1))
    print "Fussion"
    for j in range(0, S):
        for i in range(cTMP, 0, -32):
            #print "Pos {}, i {}, j{}, TMP[{}:{}]".format((cTMP*j)+(i-32),i,j, (j*cTMP)+i-32,(j*cTMP)+i)
            TMP[(cTMP*j)+(i-32)] = massive_or(TMP[(j*cTMP)+i-32:(j*cTMP)+i])
    print "Results"
    for j in range(0, S):
        for i in range(0,cTMP/32):
            SK_2_64[(j*cTMP/32)+i] = TMP[(j*cTMP)+i*32];

    return SK_2_64.reshape((rTMP , cTMP/32))

# Random data
pairs = [(16, 500000), (48, 500000), (16, 1000000), (48, 1000000)]
for K, S in pairs:
    total_bases = (K * S)
    h_SK_4 = np.random.choice(range(0,4), total_bases).astype("uint8").reshape((S,K))
    rSK_4 = h_SK_4.shape[0] # Number of kmers
    cSK_4 = h_SK_4.shape[1] # Kmer size

    rTMP = h_SK_4.shape[0]
    if (h_SK_4.shape[1]%32)==0:
        cTMP =  cSK_4
    else:
        cTMP = (32 - cSK_4%32) + cSK_4

    # Output matrix
    h_SK_2_64=np.ndarray((rTMP, cTMP/32)).astype(np.uint64)

    print "#############  KMERS BASE 4 A BASE 2 (64bits) K = {}, S = {}  #################".format(K, S)
    # OpenCL Things
    contexto = cl.create_some_context()
    cola = cl.CommandQueue(contexto)
    codigo_kernel = open("kernels/N2B64.cl").read()
    programa = cl.Program(contexto, codigo_kernel).build()
    N2B64 = programa.N2B64
    N2B64.set_scalar_arg_dtypes([None, None, None, np.uint32, np.uint32, np.uint32])

    #Buffer for intermedial matrix
    # Num bits / 8
    size_TMP = (64 * rTMP * cTMP) / 8
    d_TMP = cl.Buffer(contexto, cl.mem_flags.WRITE_ONLY, size_TMP)

    # Copy input data from host to device
    d_SK_4 = cl.Buffer(contexto, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_SK_4)
    # Output buffer
    d_SK_2_64 = cl.Buffer(contexto, cl.mem_flags.WRITE_ONLY, h_SK_2_64.nbytes)
    # Execution Range
    rango_global = (cTMP, rTMP)
    # Kernel Execution
    print "Executing kernel"
    t1 = time()
    N2B64(cola, rango_global, None, d_SK_4, d_TMP, d_SK_2_64, cSK_4, cTMP, rTMP)
    cola.finish()
    print "Kernel took {} seconds in the execution".format(time()-t1)
    # Retrieve output
    cl.enqueue_copy(cola, h_SK_2_64, d_SK_2_64)
    cola.finish()
    print "Executing sequential function"
    t1 = time()
    print "Memory Usage %s" % memory_usage_psutil()
    h_SK_2_64_s = sequential_conversion(h_SK_4, cSK_4, rTMP, cTMP)
    print "Function took {} seconds in the execution".format(time()-t1)
    print "Memory Usage %s" % memory_usage_psutil()

    if (h_SK_2_64_s==h_SK_2_64).all():
        print "Matrix are equal"
    else:
        print "Matrix not equal"
