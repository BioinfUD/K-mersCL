import argparse
parser = argparse.ArgumentParser(description="Executes multiple times the get super kmer script to do a performance assesment")
parser.add_argument('--kmers', dest="kmers", default="31", help="Kmer size to perform performance assesment (Comma separated). Default value: 31")
parser.add_argument('--mmers', dest="mmers", default="4", help="Mmer size to perform performance assesment (Comma separated)")
parser.add_argument('--input_files', dest="input_files", help="List of paths to evaluate files (Comma separated)")
parser.add_argument('--read_sizes', dest="read_sizes", help="Read size of each file specified on --input_files option")
parser.add_argument('--output_path', dest="output_path", default="output_superkmers", help="Folder where the stats and output will be stored")
args = parser.parse_args()

def execute_metrics_collection(output_path):
    # This should be async
    pid = 0
    return pid

def execute_kmers_obtaining(output_path, params):
    # Sync
    pid = 0
    return pid

def execute_metrics_summary(output_path):
    pid = 0
    path = "{}/metrics".format(output_path)
    print "Merging metrics in {}".format(path)
    print "Doing summarization"
    print "Generating plots"
    return pid

kmers = args.kmers.split(",")
mmers = args.mmers.split(",")
input_files = args.input_files.split(",")
read_sizes = args.read_sizes.split(",")
output_path = args.output_path
assert(len(input_files) == len(read_sizes), "Read sizes options are not of the same lenght of input_files options")
for kmer in kmers:
    for mmer in mmers:
        for idx, input_file in enumerate(input_files):
            params = {'mmer': mmer, 'input_file_name': input_file.split("/")[-1], 'kmer': kmer, 'output_path': output_path, 'read_size': read_sizes[idx]}
            full_output_path = "{output_path}/k{kmer}-m{mmer}-r{read_size}-{input_file_name}/".format(**params)
            pid_metrics_collection = execute_metrics_collection(full_output_path)
            execute_kmers_obtaining(output_path, params)
            # Kill pid_metrics_collection
            execute_metrics_summary(output_path)
            print full_output_path

