# Get kmers complement using heterogeneous computing
import pyopencl as cl
import numpy as np
from time import time
from python_functions.utils import memory_usage_psutil

def massive_or (elements):
    el = elements[0]
    for i in elements[1:]:
        el |= i
    return el

def sequential_function(SK_4, cSK_4, cTMP, S):
    print "Memory Usage %s" % memory_usage_psutil()
    RCSK_4 =  np.ndarray(SK_4.shape).astype(np.uint8)
    RSK_4 =  np.ndarray(SK_4.shape).astype(np.uint8)
    SK_4 = SK_4.reshape((K*S))
    TMP = np.ndarray((cTMP*S)).astype(np.uint64)
    TMPRC = np.ndarray((cTMP*S)).astype(np.uint64)
    CASK_4 = np.ndarray(K*S).astype(np.uint8)
    RCSK_10 =np.ndarray((S * cTMP/32)).astype(np.uint64)
    SK_10 =np.ndarray((S * cTMP/32)).astype(np.uint64)

    # Reverse
    for j in range(S):
        for i in range(cSK_4):
            RSK_4[j][i]= SK_4[(j*cSK_4)+(cSK_4-i-1)]

    # Complement
    for j in range(S):
        for i in range(cSK_4):
            if RSK_4[j][i] == 0:
                RCSK_4[j][i] = 3
            elif RSK_4[j][i] == 1:
                RCSK_4[j][i] = 2
            elif RSK_4[j][i] == 2:
                RCSK_4[j][i] = 1
            elif RSK_4[j][i] == 3:
                RCSK_4[j][i] = 0
    RCSK_4 = RCSK_4.reshape((K*S))

    print "Filling TMP matrix"
    for i in range(0, cTMP):
        for j in range(0, S):
            # Fill TMP
            if (i <= cSK_4-1) and (j<=S-1):
                TMP[j*cTMP + (i + cTMP - cSK_4)] = SK_4[(j*cSK_4)+i];
                TMPRC[j*cTMP + (i + cTMP - cSK_4)] = RCSK_4[(j*cSK_4)+i];

            if (i < (cTMP-cSK_4)) and (j<= S-1):
                #print "Filling with 0, i {} j {}".format(i,j)
                TMP[j*cTMP + i] = 0;
                TMPRC[j*cTMP + i] = 0;
            # Left Shift
            if (i <= cTMP-1) and i>=(cTMP-cSK_4) and (j<=S-1):
                TMP[(j*cTMP)+i] = TMP[(j*cTMP)+i] << np.uint64(64 - 2*((i%32)+1))
                TMPRC[(j*cTMP)+i] = TMPRC[(j*cTMP)+i] << np.uint64(64 - 2*((i%32)+1))

    print "Fussion"
    for j in range(0, S):
        for i in range(cTMP, 0, -32):
            #print "Pos {}, i {}, j{}, TMP[{}:{}]".format((cTMP*j)+(i-32),i,j, (j*cTMP)+i-32,(j*cTMP)+i)
            TMP[(cTMP*j)+(i-32)] = massive_or(TMP[(j*cTMP)+i-32:(j*cTMP)+i])
            TMPRC[(cTMP*j)+(i-32)] = massive_or(TMPRC[(j*cTMP)+i-32:(j*cTMP)+i])

    print "Results"
    for j in range(0, S):
        for i in range(0,cTMP/32):
            SK_10[(j*cTMP/32)+i] = TMP[(j*cTMP)+i*32];
            RCSK_10[(j*cTMP/32)+i] = TMPRC[(j*cTMP)+i*32];

    print "COMPARISSION"
    for j in range(S):
        for i in range(cSK_4):
            if (cSK_4<=32):
                if((i <= cSK_4-1)and (j<=S-1)):
                    if(SK_10[(j*cTMP/32)]<RCSK_10[(j*cTMP/32)]): # Unique position
                        CASK_4[j*cSK_4 + i] = SK_4[j*cSK_4 + i]
                else:
                    CASK_4[j*cSK_4 + i] = RCSK_4[j*cSK_4 + i] ;



    for j in range(S):
        for i in range(cSK_4):
             if (cSK_4>32 and cSK_4<=64):
                 if((i <= cSK_4-1) and (j<=S-1)):
                     if(SK_10[(j*cTMP/32)]<RCSK_10[(j*cTMP/32)]): # Unique position
                         CASK_4[j*cSK_4 + i] = SK_4[j*cSK_4 + i]
                     elif (SK_10[(j*cTMP/32)]>RCSK_10[(j*cTMP/32)]):
                         CASK_4[j*cSK_4 + i] = RCSK_4[j*cSK_4 + i] ;
                 else:
                     if(SK_10[(j*cTMP/32)+1]<RCSK_10[(j*cTMP/32)+1]): # Unique position
                         CASK_4[j*cSK_4 + i] = SK_4[j*cSK_4 + i] ;
                     else:
                        CASK_4[j*cSK_4 + i] = RCSK_4[j*cSK_4 + i] ;

    print "Memory Usage %s" % memory_usage_psutil()
    print "Memory Usage %s" % memory_usage_psutil()
    return CASK_4.reshape((S,K))


# Random data
pairs = [(16, 500000), (48, 500000), (16, 1000000), (48, 1000000)]
for K, S in pairs:
    total_bases = (K * S)
    h_SK_4 = np.random.choice(range(0,4), total_bases).astype("uint8").reshape((S,K))
    cSK_4 = h_SK_4.shape[1]

    # Output matrix de complemento
    h_CSK_4 = np.ndarray(h_SK_4.shape).astype(np.uint8)
    # Matriz de complemento reverso
    h_RCSK_4 = np.ndarray(h_SK_4.shape).astype(np.uint8)



    rTMP = h_SK_4.shape[0]
    if (h_SK_4.shape[1]%32)==0:
        cTMP =  cSK_4
    else:
        cTMP = (32 - cSK_4%32) + cSK_4
    print "Doing for K {} S {}".format(K,S)
    print "Get canonical"
    # OpenCL Things
    contexto = cl.create_some_context()
    cola = cl.CommandQueue(contexto)

    # Program for reverse complement
    codigo_kernel_c = open("kernels/getCanonical.cl").read()
    programa_c = cl.Program(contexto, codigo_kernel_c).build()
    getCanonical = programa_c.getCanonical
    getCanonical.set_scalar_arg_dtypes([None, None, None, None, None, None, None, None, np.uint32, np.uint32, np.uint32])
    # NDRange
    rango_global = (cTMP, S)

    # Copy input data from host to device
    d_SK_4 = cl.Buffer(contexto, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_SK_4)
    #Complement buffer
    d_RSK_4 = cl.Buffer(contexto, cl.mem_flags.WRITE_ONLY, h_CSK_4.nbytes)
    # Reverse complement buffer
    d_RCSK_4 = cl.Buffer(contexto, cl.mem_flags.WRITE_ONLY, h_RCSK_4.nbytes)
    # TMPs Buffers
    size_TMP = (64 * rTMP * cTMP) / 8
    d_TMP = cl.Buffer(contexto, cl.mem_flags.WRITE_ONLY, size_TMP)
    d_TMPRC = cl.Buffer(contexto, cl.mem_flags.WRITE_ONLY, size_TMP)
    # Buffers for decimal representation
    h_RCSK_10 =np.ndarray((rTMP, cTMP/32)).astype(np.uint64)
    h_SK_10 =np.ndarray((rTMP,cTMP/32)).astype(np.uint64)
    d_SK_10 = cl.Buffer(contexto, cl.mem_flags.WRITE_ONLY, h_RCSK_10.nbytes)
    d_RCSK_10 = cl.Buffer(contexto, cl.mem_flags.WRITE_ONLY, h_RCSK_10.nbytes)
    # Buffer for output matrix
    d_CASK_4 = cl.Buffer(contexto, cl.mem_flags.WRITE_ONLY, h_SK_4.nbytes)
    h_CASK_4 =np.ndarray((S, K)).astype(np.uint8)
    # Kernel execution for canonical kmer
    print "Executing kernel"
    t1 = time()
    getCanonical(cola, rango_global, None, d_SK_4, d_RSK_4, d_RCSK_4, d_TMP, d_TMPRC, d_SK_10, d_RCSK_10, d_CASK_4, cSK_4, cTMP, S)
    cola.finish()
    print "Kernel took {} seconds in the execution".format(time()-t1)
    cl.enqueue_copy(cola, h_CASK_4, d_CASK_4)

    print "Executing sequential function"
    t1 = time()
    CASK_4_s = sequential_function(h_SK_4, cSK_4, cTMP, S)
    print "Function took {} seconds in the execution".format(time()-t1)

    # Retrieve output

    if (CASK_4_s==h_CASK_4).all():
        print "Matrix are equal"
    else:
        print "Matrix not equal"
