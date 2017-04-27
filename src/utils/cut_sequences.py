# Reduce sequence to MAX
MAX = 190
from Bio.Seq import Seq
from Bio import SeqIO
in_file = open("/tmp/UnAligSeq24606.txt", "rU")
out_file = open("/tmp/outfile.txt", "w")
records = [r for r in SeqIO.parse(in_file, "fasta")]
for i in range(len(records)):
  records[i].seq = Seq(records[i].seq.tostring()[0:MAX])

SeqIO.write(records, "/tmp/salida.fasta", "fasta")
