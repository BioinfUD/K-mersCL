import pyopencl as cl
import numpy as np
from time import time
import scipy
# BASE 10 A BASE 4

# Input Matrix
SK_10 = [[6148914691236517205],[287390336450102269],[123006441751904693]]
h_SK_10 =np.ndarray((len(SK_10), len(SK_10[0]))).astype(np.uint64)
h_SK_10[:] = SK_10

# Parameters
k = 9

cSK_4 = h_SK_10.shape[1]*32 # Ncols output matrix
S = h_SK_10.shape[0] # Total kmers

# Output matrix
h_SK_4 = np.ndarray((S, cSK_4)).astype(np.uint8)

print "#############  KMERS BASE 2 (64 bits - 32 bases) A BASE 4 (8bits por base) #################"
# OpenCL Things
contexto = cl.create_some_context()
cola = cl.CommandQueue(contexto)
codigo_kernel = open("kernels/D2N.cl").read()
programa = cl.Program(contexto, codigo_kernel).build()
D2N = programa.D2N
D2N.set_scalar_arg_dtypes([None, None, np.uint32, np.uint32])

# Copy input data from host to device
d_SK_10 = cl.Buffer(contexto, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_SK_10)

# Output buffer
d_SK_4 = cl.Buffer(contexto, cl.mem_flags.WRITE_ONLY, h_SK_4.nbytes)

# Execution Range
rango_global = (cSK_4, S)
# Kernel Execution
D2N(cola, rango_global, None, d_SK_4, d_SK_10, cSK_4, S)
cola.finish()
# Retrieve output
cl.enqueue_copy(cola, h_SK_4, d_SK_4)
# Borro columnas que sobran
#h_SK_4 = scipy.delete(h_SK_4,range(cSK_4-k),1)
print "Input Matrix"
print h_SK_10
print "Output matrix"
print h_SK_4
