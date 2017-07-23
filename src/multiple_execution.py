import argparse
import os
import subprocess
from utils.analyze_taken_metrics import merge_metrics

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Executes multiple times the get super kmer script to do a performance assesment")
    parser.add_argument('--kmers', dest="kmers", default="31",
                        help="Kmer size to perform performance assesment (Comma separated). Default value: 31")
    parser.add_argument('--mmers', dest="mmers", default="4",
                        help="Mmer size to perform performance assesment (Comma separated)")
    parser.add_argument('--input_files', dest="input_files", help="List of paths to evaluate files (Comma separated)")
    parser.add_argument('--read_sizes', dest="read_sizes",
                        help="Read size of each file specified on --input_files option")
    parser.add_argument('--output_path', dest="output_path", default="output_superkmers",
                        help="Folder where the stats and output will be stored")
    args = parser.parse_args()
    kmers = args.kmers.split(",")
    mmers = args.mmers.split(",")
    input_files = args.input_files.split(",")
    read_sizes = args.read_sizes.split(",")
    output_path = args.output_path
    assert (len(input_files) == len(read_sizes), "Read sizes options are not of the same lenght of input_files options")
    return kmers, mmers, input_files, read_sizes, output_path

def execute_metrics_collection(full_output_path):
    # This should be async
    path = os.path.join(full_output_path, "metrics")
    if not os.path.exists(path):
        os.system('mkdir -p {}'.format(path))
    cpu_command = "sar -P ALL 1 99999 > {}/sar_cpu_file.log".format(path)
    memio_command = "sar -b -r 1 99999 > {}/sar_mem_io_file.log".format(path)
    nvidia_command = "nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv -l 1 | ts %s >> {}/nvidia_gpu.log ".format(path)
    process_cpu = subprocess.Popen("LC_TIME='C' exec " + cpu_command, shell=True)
    process_memio = subprocess.Popen("LC_TIME='C' exec " +memio_command, shell=True)
    process_nvidia = subprocess.Popen("LC_TIME='C' exec " +nvidia_command, shell=True)
    return process_cpu, process_memio, process_nvidia

def execute_kmers_obtaining(params):
    # Sync
    pid = 0
    params['output_path'] = "{output_path}/output_files".format(**params)
    command = "python2 getSuperK2_M.py --kmer {kmer} --mmer {mmer} --input_file {input_file} --read_size {read_size} --output_path {output_path}".format(**params)
    print command
    command = "sleep 10"
    subprocess.call("exec "+command, shell=True)
    return pid

def execute_metrics_summary(full_output_path):
    path = path = os.path.join(full_output_path, "metrics")
    print "Merging metrics in {}".format(path)
    merge_metrics(path, 4, "2017-07-23")
    print "Doing summarization"
    print "Generating plots"

def execute_assesment(kmer, mmer, input_file, read_size, output_path):
    params = {'mmer': mmer, 'input_file_name': input_file.split("/")[-1], 'kmer': kmer, 'output_path': output_path,
              'read_size': read_size, 'input_file': input_file}
    full_output_path = "{output_path}/k{kmer}-m{mmer}-r{read_size}-{input_file_name}/".format(**params)
    # Rewrite for specific output
    params['output_path'] = full_output_path
    process_cpu, process_memio, process_nvidia = execute_metrics_collection(full_output_path)
    execute_kmers_obtaining(params)
    process_cpu.kill()
    process_memio.kill()
    process_nvidia.kill()
    # Kill pid_metrics_collection
    execute_metrics_summary(full_output_path)
    print full_output_path


kmers, mmers, input_files, read_sizes, output_path = parse_arguments()
for kmer in kmers:
    for mmer in mmers:
        for idx, input_file in enumerate(input_files):
            execute_assesment(kmer, mmer, input_file, read_sizes[idx], output_path)

