from read_conversion import int_to_base

def cut_minimizer_matrix(minimizer_matrix, counter_vector):
    n_reads = counter_vector.shape[0]
    cutted_matrix = []
    for i in range(counter_vector.shape[0]):
        n_superkmers = counter_vector[i][0]
        cutted_matrix.append(minimizer_matrix[i][0:n_superkmers])
        # Get the values
        pass
    return cutted_matrix


def extract_superkmers(reads_matrix, minimizer_matrix):
    superkmers = [] # Composed of tuples (superkmer, minimizer)
    for read_number, row in minimizer_matrix:
        for v in row:
            start =
            end =
            minimizer = v >> 20
            superkmers.append((reads_matrix[read_number][start:end], minimizer)
    return superkmers
