from Bio import SeqIO
from read_conversion import int_to_base, integer_to_bases

def cut_minimizer_matrix(minimizer_matrix, counter_vector):
    n_reads = counter_vector.shape[0]
    cutted_matrix = []
    for i in range(counter_vector.shape[0]):
        n_superkmers = counter_vector[i][0]
        cutted_matrix.append(minimizer_matrix[i][0:n_superkmers])
        # Get the values
        pass
    return cutted_matrix


def extract_superkmers(minimizer_matrix, input_file_path, output_path, m=4):
    parser = SeqIO.parse(input_file_path, "fasta")
    output_files = {}
    n_superkmers = 0
    for row in minimizer_matrix:
        record = parser.next()
        print "Extracting superkmers for seqID: {}, superkmers: {}".format(record.id, str(row))
        for v in row:
            minimizer = (v & 0b11111111111100000000000000000000) >> 20
            pos = (v & 0b00000000000011111111111100000000) >> 8
            size = v & 0b00000000000000000000000011111111
            end =  pos + size
            if minimizer not in output_files:
                minimizer = integer_to_bases(minimizer)
                output_files[minimizer] = open(output_path+"/"+minimizer, "w")
            output_files[minimizer].write(str(record.seq)[pos:end]+"\n")
            n_superkmers+=1
    return n_superkmers

"""
Test case for extract_superkmers_locations
Min = 228 # TGCA 000011100100
Pos = 7  # 0000000111
Size = 15 # 0000001111
v = 0b00001110010000000001110000001111
minimizer = (v & 0b11111111111100000000000000000000) >> 20
pos = (v & 0b00000000000011111111110000000000) >> 10
size = v & 0b00000000000000000000001111111111
"""
