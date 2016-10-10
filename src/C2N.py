import pyopencl as cl
import numpy as np
from time import time
from python_functions.utils import memory_usage_psutil

def objects_memory_size(ol):
    # return the memory usage in MB
    total = 0
    for o in ol:
        total += sys.getsizeof(o)
    return total/ float(2 ** 20)

def sequential_conversion(SK_h):
    SK4 =  np.ndarray((SK_h.shape)).astype(np.uint8)
    for i in range(0, SK_h.shape[0]):
        for j in range(0, SK_h.shape[1]):
            if SK_h[i][j] == 'A':
                SK4[i][j] = 0
            elif SK_h[i][j] == 'C':
                SK4[i][j] = 1
            elif SK_h[i][j] == 'T':
                SK4[i][j] = 3
            elif SK_h[i][j] == 'G':
                SK4[i][j] = 2
    return SK4

# Random data
pairs = [(16, 500000), (48, 500000), (16, 1000000), (48, 1000000)]

for k, s in pairs:
    total_bases = (k * s)
    SK_h = np.random.choice(["A", "C", "T", "G"], total_bases).astype("S").reshape((s, k))
    print "#############  KMERS  A BASE 4 K = {}, S = {}  #################".format(k, s)
    # OpenCL Things
    contexto = cl.create_some_context()
    cola = cl.CommandQueue(contexto)
    codigo_kernel = open("kernels/c2n.cl").read()
    programa = cl.Program(contexto, codigo_kernel).build()
    C2N = programa.C2N
    C2N.set_scalar_arg_dtypes([None, None, np.uint32, np.uint32])
    # Vector de salida
    SK4_h = np.empty(SK_h.shape).astype(np.uint8)
    SK4_d = cl.Buffer(contexto, cl.mem_flags.WRITE_ONLY, SK4_h.nbytes)
    # Copy input data from host to device
    SK_d = cl.Buffer(contexto, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=SK_h)
    cola.finish()
    # Execution Range
    rango_global = (k, s)
    rango_local = (k, 1)
    # Kernel Execution
    print "Executing kernel"
    t1 = time()
    C2N(cola, rango_global, rango_local, SK_d, SK4_d, k, s)
    cola.finish()
    print "Kernel took {} seconds in the execution".format(time()-t1)
    # Retrieve output
    cl.enqueue_copy(cola, SK4_h, SK4_d)
    print "Executing sequential function"
    print "Memory Usage %s" % memory_usage_psutil()
    t1 = time()
    SK4_h_s = sequential_conversion(SK_h)
    print "Memory Usage %s" % memory_usage_psutil()
    print "Function took {} seconds in the execution".format(time()-t1)
    if (SK4_h_s==SK4_h).all():
        print "Matrix are equal"
    else:
        print "Matrix not equal"
