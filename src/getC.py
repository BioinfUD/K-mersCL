# Get kmers complement using heterogeneous computing
import pyopencl as cl
import numpy as np
from time import time

SK_4 = [[3,3,3,3],[2,3,2,1],[1,2,3,1],[1,2,3,1],[0,1,2,3]] # Matriz con KMERS
h_SK_4 =np.ndarray((len(SK_4), len(SK_4[0]))).astype(np.uint8)
h_SK_4[:] = SK_4

S = h_SK_4.shape[0] # Numeero de kmes
K = h_SK_4.shape[1] # Tamano del kmer

# Matriz de salida
h_CSK_4 = np.ndarray(h_SK_4.shape).astype(np.uint8)

print "#############  KMERS BASE 4 A BASE 2 (8bits) #################"
# OpenCL Things
contexto = cl.create_some_context()
cola = cl.CommandQueue(contexto)
codigo_kernel = open("kernels/getC.cl").read()
programa = cl.Program(contexto, codigo_kernel).build()
getC = programa.getC
getC.set_scalar_arg_dtypes([None, None, np.uint32, np.uint32])
#Copio datos de entrada a dispositivo
d_SK_4 = cl.Buffer(contexto, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_SK_4)
#Bufer para la salida
d_CSK_4 = cl.Buffer(contexto, cl.mem_flags.WRITE_ONLY, h_CSK_4.nbytes)
# Dimensiones de ejecucion
rango_global = (K, S)
# Ejecucion del kernel
getC(cola, rango_global, None, d_SK_4, d_CSK_4, K, S)
cola.finish()
# Traigo datos
cl.enqueue_copy(cola, h_CSK_4, d_CSK_4)
print h_SK_4
print h_CSK_4
