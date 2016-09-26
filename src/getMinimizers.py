import pyopencl as cl
import numpy as np
from time import time
import scipy

#Kmer size
K = 33
#Number of kmers
s = 15
#minimizers size
m = 4

# K-mer generation
h_SK_4 = np.random.choice(range(0,4), K*s).astype("uint8").reshape((s,K))
rSK_4 = h_SK_4.shape[0] # Number of kmers
cSK_4 = h_SK_4.shape[1] # Kmer size


# Output memory allocation
h_ML_4 = np.ndarray((rSK_4, m)).astype(np.uint8)
h_ML_P = np.ndarray((rSK_4, 1)).astype(np.uint8)


contexto = cl.create_some_context()
cola = cl.CommandQueue(contexto)
codigo_kernel = open("kernels/getMinimizers.cl").read()
programa = cl.Program(contexto, codigo_kernel).build()
getMinimizers = programa.getMinimizers
getMinimizers.set_scalar_arg_dtypes([None, None, None, None, np.uint32, np.uint32, np.uint32, np.uint32, np.uint32])

###############################################
# Memory allocation and copy in device
###############################################
d_SK_4 = cl.Buffer(contexto, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_SK_4)

rTMP_P = rSK_4
n_m = K-m+1

# Number of columns of TMP_P matrix
cTMP_P = n_m


size_TMP_P = (8 * rTMP_P * cTMP_P) / 8 # Bytes

d_TMP_P = cl.Buffer(contexto, cl.mem_flags.WRITE_ONLY, size_TMP_P)

# Output matrix
d_ML_4 = cl.Buffer(contexto, cl.mem_flags.WRITE_ONLY, h_ML_4.nbytes)
d_ML_P = cl.Buffer(contexto, cl.mem_flags.WRITE_ONLY, h_ML_P.nbytes)


# Kernel Execution
rango_global = (cTMP_P, s)
t1 = time()
getMinimizers(cola, rango_global, None, d_SK_4, d_ML_4, d_ML_P, d_TMP_P, cSK_4, cTMP_P, m, n_m, s)
cola.finish()
print "Kernel took {} seconds in the execution".format(time()-t1)
# Copy data from device to host
dims = (rTMP_P, cTMP_P)
#h_TMP = np.ndarray(dims).astype(np.uint8)
h_ML_P = np.ndarray((rTMP_P,1)).astype(np.uint8)
#cl.enqueue_copy(cola, h_TMP, d_TMP_P)
cl.enqueue_copy(cola, h_ML_P, d_ML_P)
cl.enqueue_copy(cola, h_ML_4, d_ML_4)
cola.finish()

print "Input Matrix"
print h_SK_4
print "Output Matrix"
print h_ML_4
print "Output vector"
print h_TMP_P
"""
print "Temporal matrix"
print h_TMP
"""

#import ipdb; ipdb.set_trace()
