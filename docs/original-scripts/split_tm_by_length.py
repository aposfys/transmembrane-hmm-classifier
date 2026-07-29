from Bio import SeqIO

# Parameters
input_file = "extracted_tm_regions.fasta"  # the file you already have
short_output = "short_tm_regions.fasta"
long_output = "long_tm_regions.fasta"
threshold = 40  # you can change this if needed

# Lists to hold sequences
short_tm = []
long_tm = []

# Read sequences and split them
for record in SeqIO.parse(input_file, "fasta"):
    seq_length = len(record.seq)
    if seq_length <= threshold:
        short_tm.append(record)
    else:
        long_tm.append(record)

# Save to new FASTA files
SeqIO.write(short_tm, short_output, "fasta")
SeqIO.write(long_tm, long_output, "fasta")

print(f"✅ Split completed: {len(short_tm)} short TM regions, {len(long_tm)} long TM regions.")
print(f"Short regions saved to '{short_output}', Long regions saved to '{long_output}'.")
