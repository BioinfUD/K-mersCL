# Este script define una matriz de kmers en base 4 y la transforma a una matriz de kmers en base 10 de 8 bits.
import pyopencl as cl
import numpy as np
from time import time
import scipy
# BASE 10 A BASE 4

# Matriz de entrada
SK_2_8 = [[3,64,255],[1,1,64],[1,0,220]]
h_SK_2_8 =np.ndarray((len(SK_2_8), len(SK_2_8[0]))).astype(np.uint8)
h_SK_2_8[:] = SK_2_8

#Parametros
k = 9

c_sk_4 = h_SK_2_8.shape[1]*4
s = h_SK_2_8.shape[0] # Numero de kmes

# Matriz de salida
h_SK_4 = np.ndarray((s, c_sk_4)).astype(np.uint8)


print "#############  KMERS BASE 2 (8 bits - 4 bases) A BASE 4 (8bits por bases) #################"
# OpenCL Things
contexto = cl.create_some_context()
cola = cl.CommandQueue(contexto)
codigo_kernel = open("kernels/B82N.cl").read()
programa = cl.Program(contexto, codigo_kernel).build()
B82N = programa.B82N
B82N.set_scalar_arg_dtypes([None, None, np.uint32, np.uint32])

#Copio datos de entrada a dispositivo
d_SK_2_8 = cl.Buffer(contexto, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_SK_2_8)

#Bufer para la salida
d_SK_4 = cl.Buffer(contexto, cl.mem_flags.WRITE_ONLY, h_SK_4.nbytes)

# Dimensiones de ejecucion
rango_global = (c_sk_4, s)
# Ejecucion del kernel
B82N(cola, rango_global, None, d_SK_4, d_SK_2_8,c_sk_4, s)
cola.finish()
# Traigo datos
cl.enqueue_copy(cola, h_SK_4, d_SK_4)
# Borro columnas que sobran
h_SK_4 = scipy.delete(h_SK_4,range(c_sk_4-k),1)
print "Matriz de entrada"
print h_SK_2_8
print "Matriz de salida"
print h_SK_4
