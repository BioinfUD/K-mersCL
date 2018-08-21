import numpy as np
import pyopencl as cl
import sys
import argparse
import os
import time

from utils.read_conversion import file_to_matrix
from utils.superkmer_utils import cut_minimizer_matrix, extract_superkmers

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Obtains superkmers (Based on signatures) from a input file given a mmer, kmer sizes. Does the computing on GPU")
    parser.add_argument('--kmer', dest="kmer", default="31",
                        help="Kmer size to perform performance assesment. Default value: 31")
    parser.add_argument('--mmer', dest="mmer", default="4",
                        help="Mmer size to perform performance assesment")
    parser.add_argument('--input_file', dest="input_file", help="List of paths to evaluate files")
    parser.add_argument('--read_size', dest="read_size",
                        help="Read size of each file specified on --input_files option")
    parser.add_argument('--output_path', dest="output_path", default="output_superkmers",
                        help="Folder where the stats and output will be stored")
    parser.add_argument('--n_reads', dest="n_reads", default=None, help="Number of reads in each file. If not specified this will be estimated")
    args = parser.parse_args()
    kmer = args.kmer
    mmer = args.mmer
    input_file = args.input_file
    read_size = args.read_size
    output_path = args.output_path
    n_reads = args.n_reads
    if any(x is None for x in [kmer, mmer, input_file, read_size, output_path]):
        parser.print_help()
        sys.exit(0)
    return int(kmer), int(mmer), input_file, int(read_size), n_reads, output_path

def customize_kernel_template(X, k, m, r, kernel_template):
    if (k <= 36):
        nt3 = 6
    elif (k <= 64):
        nt3 = 8
    elif (k <= 100):
        nt3 = 10
    else:
        nt3 = 12
    nm = r - m + 1;
    lsd = X // m;
    ts = ((nm-1)//lsd) + 1
    nt2 = ((nm-1)//ts) + 1
    a_mask = (2**((2*m)-2)) - 1
    shift1 = 32 - (2*m)
    mask1 = ((2**(m*2)) - 1) << shift1;
    mask2 = ((2**(25-(2*m))) - 1) << 7;
    kernel_template = kernel_template.replace("READ_SIZE", str(r))\
        .replace("SECOND_MT", str(nt2))\
        .replace("NT_MAX", str(max(nt2, nt3)))\
        .replace("A_MASK", str(a_mask))\
        .replace("SHIFT1", str(shift1))\
        .replace("MASK1", str(mask1))\
        .replace("MASK2", str(mask2))
    return kernel_template

def getSuperK_M(kmer, mmer, input_file, read_size, n_reads, output_path):

    h_R2M_G = file_to_matrix(input_file, int(read_size), n_reads)
    nr = h_R2M_G.shape[0]
    nmk = kmer - mmer + 1
    X = nmk
    # OpenCL things
    contexto = cl.create_some_context()
    cola = cl.CommandQueue(contexto)
    kernel = "kernels/getSuperK2_M_signature.cl.tpl"
    codigo_kernel = open(kernel).read()
    codigo_kernel = customize_kernel_template(X, kmer, mmer, read_size, codigo_kernel)
    programa = cl.Program(contexto, codigo_kernel).build()
    getSuperK_M = programa.getSuperK_M
    getSuperK_M.set_scalar_arg_dtypes([None, None, None, np.uint32, np.uint32, np.uint32,np.uint32])
    # Copy data from host to device
    t1 = time.time()
    sys.stdout.write("Copying data from host to device memory \n")
    d_R2M_G = cl.Buffer(contexto, cl.mem_flags.COPY_HOST_PTR | cl.mem_flags.READ_ONLY, hostbuf=h_R2M_G)
    sys.stdout.write("Copying data took {} seconds \n".format(time.time() - t1))

    # Kernel parameters
    sys.stdout.write("Ejecutando X hilos, X: {}\n".format(X))
    rango_global = (X, nr)
    rango_local = (X, 1)
    # Output matrix
    nm = read_size - mmer + 1;
    h_counters = np.empty((nr, 1)).astype(np.uint32)
    d_counters = cl.Buffer(contexto, cl.mem_flags.WRITE_ONLY, h_counters.nbytes)
    # Execution
    t1 = time.time()
    h_cisk = np.ndarray(shape=(nr, (read_size - kmer + 1)/2), dtype=np.uint32)
    d_cisk = cl.Buffer(contexto, cl.mem_flags.WRITE_ONLY | cl.mem_flags.HOST_READ_ONLY , h_cisk.nbytes)
    getSuperK_M(cola, rango_global, rango_local, d_R2M_G, d_counters, d_cisk, nr, read_size, kmer, mmer)
    cola.finish()
    sys.stdout.write("Kernel execution took {}, copying data from device to host memory\n".format(time.time() - t1))
    t1 = time.time()
    cl.enqueue_copy(cola, h_counters, d_counters)
    cl.enqueue_copy(cola, h_cisk, d_cisk)
    sys.stdout.write("Copying data took {} seconds \n".format(time.time() - t1))
    # Cut the output matrix based on counters
    sys.stdout.write("Copy Done, cutting the matrix based on available superkmers\n")
    minimizer_matrix = cut_minimizer_matrix(h_cisk, h_counters)
    del(h_cisk)
    sys.stdout.write("Writing superkmers to disk\n")
    extract_superkmers(minimizer_matrix, input_file, output_path, kmer, m=mmer)
    sys.stdout.write("Done execution\n")

if __name__ == "__main__":
    kmer, mmer, input_file, read_size, n_reads, output_path = parse_arguments()
    if not os.path.exists(output_path):
        os.system('mkdir -p {}'.format(output_path))
    getSuperK_M(kmer, mmer, input_file, read_size, n_reads, output_path)
