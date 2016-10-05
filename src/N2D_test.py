# Este script define una matriz de kmers en base 4 y la transforma a una matriz de kmers en base 10 de 8 bits.
import pyopencl as cl
import numpy as np
from time import time
# BASE 4 A CARACTERES
#Memoria para lectura en host
# Matriz con KMERS

K = 64
S = 3000000
total_bases = (K * S)
SK_4 = np.random.choice(range(0,4), total_bases).astype("uint8").reshape((S,K))

"""
SK_4 = [
    [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,2,1,0,0,0,1,2,3,1,1,0,0,0,1,2,3,2,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,2,1,0,0,0,1,2,3,1,1,0,0,0,1,2,3,2],
    [1,1,0,0,0,2,0,0,0,0,0,0,0,0,0,2,1,1,0,2,1,2,3,1,1,0,0,0,1,2,3,2,1,1,0,0,0,2,0,0,0,0,0,0,0,0,0,2,1,1,0,2,1,2,3,1,1,0,0,0,1,2,3,2],
    [1,0,0,0,1,2,3,1,1,0,1,2,1,2,3,2,1,0,0,0,1,2,3,1,1,0,0,0,1,2,3,2,1,0,0,0,1,2,3,1,1,0,1,2,1,2,3,2,1,0,0,0,1,2,3,1,1,0,0,0,1,2,3,2]
    ]
"""
#SK_4 = [[3,3,3,3],[2,3,2,1],[1,2,3,1],[1,2,3,1],[0,1,2,3]] # Matriz con KMERS
h_SK_4 =np.ndarray((len(SK_4), len(SK_4[0]))).astype(np.uint8)
h_SK_4[:] = SK_4

rSK_4 = h_SK_4.shape[0] # Number of kmers
cSK_4 = h_SK_4.shape[1] # Kmer size

if cSK_4 > 64:
    print "Max supported kmer size 64"


# Output matrix
cSK_10 = 2
#cSK_10 = cSK_4/16
h_SK_10=np.ndarray((rSK_4, cSK_10)).astype(np.uint64)

print "#############  KMERS BASE 4 A BASE 2 (64bits) #################"
# OpenCL Things
contexto = cl.create_some_context()
cola = cl.CommandQueue(contexto)
codigo_kernel = open("kernels/N2D_test.cl").read()
programa = cl.Program(contexto, codigo_kernel).build()
N2D = programa.N2D
N2D.set_scalar_arg_dtypes([None, None, np.uint32, np.uint32, np.uint32])


# Copy input data from host to device
d_SK_4 = cl.Buffer(contexto, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_SK_4)
# Output buffer
d_SK_10 = cl.Buffer(contexto, cl.mem_flags.WRITE_ONLY, h_SK_10.nbytes)
# Execution Range
rango_global = (cSK_4, rSK_4)
# Kernel Execution

print "Executing kernel"
t1 = time()


N2D(cola, rango_global, None, d_SK_4, d_SK_10, cSK_4, cSK_10, rSK_4)
cola.finish()
print "Kernel took {} seconds in the execution".format(time()-t1)

# Retrieve output
cl.enqueue_copy(cola, h_SK_10, d_SK_10)
cola.flush()

print "Matriz entrada"
print h_SK_4
print "Matriz salida"
print h_SK_10
