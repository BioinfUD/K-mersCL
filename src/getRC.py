# Get kmers complement using heterogeneous computing
import pyopencl as cl
import numpy as np
from time import time
import psutil, os
from python_functions.utils import memory_usage_psutil

def sequential_function (SK_4, cSK_4, S):
    print "Memory Usage %s" % memory_usage_psutil()
    RCSK_4 =  np.ndarray(SK_4.shape).astype(np.uint8)
    RSK_4 =  np.ndarray(SK_4.shape).astype(np.uint8)
    SK_4 = SK_4.reshape((K*S))
    # Reverse
    for j in range(S):
        for i in range(cSK_4):
            RSK_4[j][i]= SK_4[(j*cSK_4)+(cSK_4-i-1)]
    # Complement
    for j in range(S):
        for i in range(cSK_4):
            if RSK_4[j][i] == 0:
                RCSK_4[j][i] = 3
            elif RSK_4[j][i] == 1:
                RCSK_4[j][i] = 2
            elif RSK_4[j][i] == 2:
                RCSK_4[j][i] = 1
            elif RSK_4[j][i] == 3:
                RCSK_4[j][i] = 0
    print "Memory Usage %s" % memory_usage_psutil()
    return RCSK_4

pairs = [(16, 500000), (48, 500000), (16, 1000000), (48, 1000000)]
for K, S in pairs:
    total_bases = (K * S)
    h_SK_4 = np.random.choice(range(0,4), total_bases).astype("uint8").reshape((S,K))

    # Output matrix de complemento
    h_CSK_4 = np.ndarray(h_SK_4.shape).astype(np.uint8)
    # Matriz de complemento reverso
    h_RCSK_4 = np.ndarray(h_SK_4.shape).astype(np.uint8)

    print "Get kmer complement reverse"
    # OpenCL Things
    contexto = cl.create_some_context()
    cola = cl.CommandQueue(contexto)

    # Program for reverse complement
    codigo_kernel_c = open("kernels/getRC.cl").read()
    programa_c = cl.Program(contexto, codigo_kernel_c).build()
    getRC = programa_c.getRC
    getRC.set_scalar_arg_dtypes([None, None, None, np.uint32, np.uint32])
    # NDRange
    rango_global = (K, S)

    # Copy input data from host to device
    d_SK_4 = cl.Buffer(contexto, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_SK_4)
    #Complement buffer
    d_RSK_4 = cl.Buffer(contexto, cl.mem_flags.WRITE_ONLY, h_CSK_4.nbytes)
    # Reverse complement buffer
    d_RCSK_4 = cl.Buffer(contexto, cl.mem_flags.WRITE_ONLY, h_RCSK_4.nbytes)

    # Kernel execution for reverse complement
    print "Executing kernel"
    t1 = time()
    getRC(cola, rango_global, None, d_SK_4, d_RSK_4, d_RCSK_4, K, S)
    cola.finish()
    cl.enqueue_copy(cola, h_RCSK_4, d_RCSK_4)
    print "Kernel took {} seconds in the execution".format(time()-t1)
    print "Executing sequential function"
    t1 = time()
    h_RCSK_4_s = sequential_function (h_SK_4, K, S)
    print "Function took {} seconds in the execution".format(time()-t1)
    import ipdb; ipdb.set_trace()
    if (h_RCSK_4_s==h_RCSK_4).all():
        print h_RCSK_4_s[0:10]
        print h_RCSK_4[0:10]
        print "Matrix are equal"
    else:
        print h_RCSK_4_s[0:10]
        print h_RCSK_4[0:10]
        print "Matrix not equal"
