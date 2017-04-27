# Reduce sequence to MAX
MAX = 180
from Bio.Seq import Seq
from Bio import SeqIO
in_file = open("/tmp/UnAligSeq24606.txt", "rU")
records = [r for r in SeqIO.parse(in_file, "fasta")]
for i in range(len(records)):
  records[i].seq = Seq(records[i].seq.tostring()[0:MAX])

SeqIO.write(records, "/tmp/outfile.fasta", "fasta")
