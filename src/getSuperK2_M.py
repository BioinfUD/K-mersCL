import numpy as np
import pyopencl as cl
import sys
import argparse
import os

from utils.read_conversion import file_to_matrix
from utils.superkmer_utils import cut_minimizer_matrix, extract_superkmers

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Obtains superkmers (Based on substrings) from a input file given a mmer, kmer sizes. Does the computing on GPU")
    parser.add_argument('--kmer', dest="kmer", default="31",
                        help="Kmer size to perform performance assesment (Comma separated). Default value: 31")
    parser.add_argument('--mmer', dest="mmer", default="4",
                        help="Mmer size to perform performance assesment (Comma separated)")
    parser.add_argument('--input_file', dest="input_file", help="List of paths to evaluate files (Comma separated)")
    parser.add_argument('--read_size', dest="read_size",
                        help="Read size of each file specified on --input_files option")
    parser.add_argument('--output_path', dest="output_path", default="output_superkmers",
                        help="Folder where the stats and output will be stored")
    args = parser.parse_args()
    kmer = args.kmer
    mmer = args.mmer
    input_file = args.input_file
    read_size = args.read_size
    output_path = args.output_path
    return int(kmer), int(mmer), input_file, int(read_size), output_path

def extract_superkmers(minimizer_matrix, input_file, output_path, m=4):
    n_superkmers = 0
    for row in minimizer_matrix:
        print "superkmers: {}".format(str(row))
        for v in row:
            minimizer = (v & 0b11111111111111000000000000000000) >> 18
            pos = (v & 0b00000000000000111111111100000000) >> 8
            size = v & 0b00000000000000000000000011111111
            end = pos + size
            minimizer_str = str(minimizer)
            print "Valor {} Min {},  ini {}, size {}, end{}".format(v, minimizer, pos, size, end)
            n_superkmers+=1
    return n_superkmers

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
    kernel_template = kernel_template.replace("READ_SIZE", str(r))\
                    .replace("SECOND_MT", str(nt2)).replace("NT_MAX", str(max(nt2, nt3)))\
                    .replace("A_MASK", str(a_mask))
    return kernel_template

def getSuperK_M(kmer, mmer, input_file, read_size, output_path):
    # Kernel parameters
    sys.stdout.write("Loading sequences from file {}\n".format(input_file))
    h_R2M_G = file_to_matrix(input_file, int(read_size))
    nr = h_R2M_G.shape[0]
    nmk = kmer - mmer + 1
    X = nmk
    # OpenCL things
    contexto = cl.create_some_context()
    cola = cl.CommandQueue(contexto)
    codigo_kernel = open("kernels/getSuperK2_M.cl.tpl").read()
    codigo_kernel = customize_kernel_template(X, kmer, mmer, read_size, codigo_kernel)
    programa = cl.Program(contexto, codigo_kernel).build()
    getSuperK_M = programa.getSuperK_M
    getSuperK_M.set_scalar_arg_dtypes([None, None, np.uint32, np.uint32, np.uint32,np.uint32])
    # Copy data from host to device
    sys.stdout.write("Copying data from host to device memory \n")
    d_R2M_G = cl.Buffer(contexto, cl.mem_flags.COPY_HOST_PTR, hostbuf=h_R2M_G)
    # Kernel parameters
    sys.stdout.write("Ejecutando X hilos, X: {}\n".format(X))
    rango_global = (X, nr)
    rango_local = (X, 1)
    # Output matrix
    nm = read_size - mmer + 1;
    h_counters = np.empty((nr, 1)).astype(np.uint32)
    d_counters = cl.Buffer(contexto, cl.mem_flags.WRITE_ONLY, h_counters.nbytes)
    # Execution
    getSuperK_M(cola, rango_global, rango_local, d_R2M_G, d_counters, nr, read_size, kmer, mmer)
    cola.finish()
    sys.stdout.write("Execution finished, copying data from device to host memory\n")
    cl.enqueue_copy(cola, h_counters, d_counters)
    cl.enqueue_copy(cola, h_R2M_G, d_R2M_G)
    # Cut the output matrix based on counters
    sys.stdout.write("Cutting the matrix based on available superkmers\n")
    minimizer_matrix = cut_minimizer_matrix(h_R2M_G, h_counters)
    del(h_R2M_G)
    sys.stdout.write("Writing superkmers to disk\n")
    extract_superkmers(minimizer_matrix, input_file, output_path, m=mmer)

if __name__ == "__main__":
    kmer, mmer, input_file, read_size, output_path = parse_arguments()
    if not os.path.exists(output_path):
        os.system('mkdir -p {}'.format(output_path))
    getSuperK_M(kmer, mmer, input_file, read_size, output_path)
