# Este script define una matriz de kmers en base 4 y la transforma a una matriz de kmers en base 10 de 8 bits.
import pyopencl as cl
import numpy as np
from time import time
import scipy


# Matriz de entrada
SK_2_32 = [[1,2,6500,255],[6500,255,1,2],[300,100,7,3],[16,7,300,100], [1,4,300,100]]
h_SK_2_32 =np.ndarray((len(SK_2_32), len(SK_2_32[0]))).astype(np.uint32)
h_SK_2_32[:] = SK_2_32


cSK_2_32 = h_SK_2_32.shape[1]
rSK_2_32 = h_SK_2_32.shape[0]

rTMP = rSK_2_32

if (cSK_2_32%2)==0:
    cTMP = cSK_2_32
else:
    cTMP = (2  - cSK_2_32%2)  + cSK_2_32


# Matriz de salida
h_SK_2_64= np.ndarray((rTMP, cTMP/2)).astype(np.uint64)


print "#############  KMERS BASE 2 (8BITS) A BASE 2 (64 BITS)  #################"
# OpenCL Things
contexto = cl.create_some_context()
cola = cl.CommandQueue(contexto)
codigo_kernel = open("kernels/B322B64.cl").read()
programa = cl.Program(contexto, codigo_kernel).build()
B322B64 = programa.B322B64
B322B64.set_scalar_arg_dtypes([None, None, None, np.uint32, np.uint32, np.uint32])

#Copio datos de entrada a dispositivo
d_SK_2_32 = cl.Buffer(contexto, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_SK_2_32)

#Matriz temporal , tamano de h_SK_2_32 x 2 dado que va a ser de 16 bits por elemento
size_TMP = (64 * rTMP * cTMP) / 8
d_TMP = cl.Buffer(contexto, cl.mem_flags.WRITE_ONLY, size_TMP)

#Bufer para la salida
d_SK_2_64 = cl.Buffer(contexto, cl.mem_flags.WRITE_ONLY, h_SK_2_64.nbytes)

# Dimensiones de ejecucion
rango_global = (cTMP, rTMP)

cSK_2_64 = h_SK_2_64.shape[1]
# Ejecucion del kernel
B322B64(cola, rango_global, None, d_SK_2_32, d_TMP, d_SK_2_64, cSK_2_32, cTMP, rTMP)
#B322B64(cola, (10,10), None, d_SK_2_32, d_SK_2_64, cSK_2_32, c_sk_2_64, s)

cola.finish()
# Traigo datos
cl.enqueue_copy(cola, h_SK_2_64, d_SK_2_64)
h_TMP = np.ndarray((rTMP, cTMP)).astype(np.uint64)
cl.enqueue_copy(cola, h_TMP, d_TMP)

print "h_TMP"
print "cTMP : " + str(cTMP) + " cSK_2_32: " + str(cSK_2_32)
print h_TMP
print h_SK_2_32
print h_SK_2_64
