# Este script define una matriz de kmers en base 4 y la transforma a una matriz de kmers en base 10 de 8 bits.
import pyopencl as cl
import numpy as np
from time import time
import scipy
# BASE 10 A BASE 4

# Matriz de entrada
K_2_8 = [[255,255,1,1,1,1,1,1,1,2]]

#SK_2_8 = [[254,255],[254,255]]
#SK_2_8 = [[255,255],[2,2],[3,3],[4,4], [4,4]]

h_SK_2_8 =np.ndarray((len(SK_2_8), len(SK_2_8[0]))).astype(np.uint8)
h_SK_2_8[:] = SK_2_8


c_sk_2_8 = h_SK_2_8.shape[1]
s = h_SK_2_8.shape[0] # Numero de kmes


# Matriz de salida


h_SK_2_16= np.ndarray((h_SK_2_8.shape[0], h_SK_2_8.shape[1]/2)).astype(np.uint16)




print "#############  KMERS BASE 2 (8BITS) A BASE 2 (16 BITS)  #################"
# OpenCL Things
contexto = cl.create_some_context()
cola = cl.CommandQueue(contexto)
codigo_kernel = open("kernels/B82B16.cl").read()
programa = cl.Program(contexto, codigo_kernel).build()
B82B16 = programa.B82B16
B82B16.set_scalar_arg_dtypes([None, None, np.uint32, np.uint32, np.uint32])

#Copio datos de entrada a dispositivo
d_SK_2_8 = cl.Buffer(contexto, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_SK_2_8)

#Bufer para la salida
d_SK_2_16 = cl.Buffer(contexto, cl.mem_flags.WRITE_ONLY, h_SK_2_16.nbytes)

# Dimensiones de ejecucion
rango_global = (h_SK_2_16.shape[1], s)

c_sk_2_16 = h_SK_2_16.shape[1]
# Ejecucion del kernel
B82B16(cola, rango_global, None, d_SK_2_8, d_SK_2_16, c_sk_2_8, c_sk_2_16, s)
#B82B16(cola, (10,10), None, d_SK_2_8, d_SK_2_16, c_sk_2_8, c_sk_2_16, s)

cola.finish()
# Traigo datos
cl.enqueue_copy(cola, h_SK_2_16, d_SK_2_16)

print h_SK_2_8
print h_SK_2_16
