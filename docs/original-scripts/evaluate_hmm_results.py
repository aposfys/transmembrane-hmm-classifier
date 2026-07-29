from Bio import SeqIO

# Files
positive_fasta = "positive_test_set.fasta"
gpcr_fasta = "gpcr_test_set.fasta"
typeII_fasta = "typeII_test_set.fasta"
globular_fasta = "globular_test_set.fasta"

positive_tbl = "results_positive_max.tbl"
gpcr_tbl = "results_gpcr.tbl"
typeII_tbl = "results_typeII.tbl"
globular_tbl = "results_globular.tbl"

# Count sequences
def count_fasta_sequences(fasta_file):
    return sum(1 for _ in SeqIO.parse(fasta_file, "fasta"))

n_positive = count_fasta_sequences(positive_fasta)
n_gpcr = count_fasta_sequences(gpcr_fasta)
n_typeII = count_fasta_sequences(typeII_fasta)
n_globular = count_fasta_sequences(globular_fasta)

n_negatives = n_gpcr + n_typeII + n_globular

# Count hits
def count_hits(tbl_file):
    with open(tbl_file) as f:
        lines = [line for line in f if not line.startswith("#")]
    return len(lines)

tp = count_hits(positive_tbl)
fp = count_hits(gpcr_tbl) + count_hits(typeII_tbl) + count_hits(globular_tbl)

tn = n_negatives - fp

# Calculate
sensitivity = tp / n_positive
specificity = tn / n_negatives

# Results
print(f"✅ Sensitivity: {sensitivity:.2f} ({sensitivity*100:.1f}%)")
print(f"✅ Specificity: {specificity:.2f} ({specificity*100:.1f}%)")

print("\nDetails:")
print(f"- Positive test set: {tp}/{n_positive} detected")
print(f"- False positives: {fp}/{n_negatives}")
print(f"- True negatives: {tn}/{n_negatives}")
