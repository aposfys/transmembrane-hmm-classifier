from Bio import SeqIO
import random

# Parameters
input_fasta = "clustered_training.fasta"
clstr_file = "clustered_training.fasta.clstr"
train_output = "final_training.fasta"
test_output = "positive_test_set.fasta"
test_percentage = 0.1  # 10% for testing

# Step 1: Parse the .clstr file to find representative sequences
representatives = []
with open(clstr_file, "r") as f:
    cluster = []
    for line in f:
        if line.startswith(">Cluster"):
            continue
        if "*" in line:
            # Line with '*' is the representative
            parts = line.split(">")
            seq_id = parts[1].split("...")[0]
            representatives.append(seq_id)

print(f"Found {len(representatives)} representatives.")

# Step 2: Split representatives into train/test
random.shuffle(representatives)
split_idx = int(len(representatives) * (1 - test_percentage))
train_ids = set(representatives[:split_idx])
test_ids = set(representatives[split_idx:])

print(f"Training set size: {len(train_ids)}")
print(f"Test set size: {len(test_ids)}")

# Step 3: Load the sequences
records = SeqIO.to_dict(SeqIO.parse(input_fasta, "fasta"))

# Step 4: Save training and test FASTA files
SeqIO.write([records[id] for id in train_ids if id in records], train_output, "fasta")
SeqIO.write([records[id] for id in test_ids if id in records], test_output, "fasta")

print("✅ Files written: final_training.fasta and positive_test_set.fasta")
