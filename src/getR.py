import pyopencl as cl
import numpy as np
from time import time

SK_4 = [[3,3,3,3],[2,3,2,1],[1,2,3,1],[1,2,3,1],[0,1,2,3]] # Matriz con KMERS
h_SK_4 =np.ndarray((len(SK_4), len(SK_4[0]))).astype(np.uint8)
h_SK_4[:] = SK_4

S = h_SK_4.shape[0] # Numeero de kmes
K = h_SK_4.shape[1] # Tamano del kmer

# Output matrix
h_RSK_4 = np.ndarray(h_SK_4.shape).astype(np.uint8)

print "#############  KMERS BASE 4 A BASE 2 (8bits) #################"
# OpenCL Things
contexto = cl.create_some_context()
cola = cl.CommandQueue(contexto)
codigo_kernel = open("kernels/getR.cl").read()
programa = cl.Program(contexto, codigo_kernel).build()
getR = programa.getR
getR.set_scalar_arg_dtypes([None, None, np.uint32, np.uint32])
# Copy input data from host to device
d_SK_4 = cl.Buffer(contexto, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_SK_4)
# Output buffer
d_RSK_4 = cl.Buffer(contexto, cl.mem_flags.WRITE_ONLY, h_RSK_4.nbytes)
# Execution Range
rango_global = (K, S)
# Kernel Execution
getR(cola, rango_global, None, d_SK_4, d_RSK_4, K, S)
cola.finish()
# Retrieve output
cl.enqueue_copy(cola, h_RSK_4, d_RSK_4)
print h_SK_4
print h_RSK_4
