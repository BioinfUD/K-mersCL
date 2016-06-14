# Get kmers complement using heterogeneous computing
import pyopencl as cl
import numpy as np
from time import time

SK_4 = [[3,3,3,3],[2,3,2,1],[1,2,3,1],[1,2,3,1],[0,1,2,3],[2,2,3,3]] # Matriz con KMERS
h_SK_4 =np.ndarray((len(SK_4), len(SK_4[0]))).astype(np.uint8)
h_SK_4[:] = SK_4

S = h_SK_4.shape[0] # Numeero de kmes
K = h_SK_4.shape[1] # Tamano del kmer

# Matriz de salida de complemento
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

#Copio datos de entrada a dispositivo
d_SK_4 = cl.Buffer(contexto, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_SK_4)
#Complement buffer
d_RSK_4 = cl.Buffer(contexto, cl.mem_flags.WRITE_ONLY, h_CSK_4.nbytes)
# Reverse complement buffer
d_RCSK_4 = cl.Buffer(contexto, cl.mem_flags.WRITE_ONLY, h_RCSK_4.nbytes)
# Kernel execution for reverse complement
getRC(cola, rango_global, None, d_SK_4, d_RSK_4, d_RCSK_4, K, S)
cola.finish()

# Traigo datos
cl.enqueue_copy(cola, h_RCSK_4, d_RCSK_4)
print h_SK_4
print h_RCSK_4
