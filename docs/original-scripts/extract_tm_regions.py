from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq

# Files
input_topcons = "topcons_result.txt"
output_fasta = "extracted_tm_regions.fasta"
padding = 10  # Add 10 amino acids before and after each TM region

# Storage
extracted_records = []

# State variables
current_id = None
current_seq = None
current_topology = None

with open(input_topcons, "r") as f:
    lines = f.readlines()

# Parse topcons_result.txt
for i, line in enumerate(lines):
    line = line.strip()

    if line.startswith("Sequence name:"):
        current_id = line.split()[2].split("|")[1]  # Extract UniProt ID between |

    elif line.startswith("Sequence:"):
        sequence_lines = []
        j = i + 1
        while j < len(lines) and not lines[j].startswith("TOPCONS predicted topology:"):
            sequence_lines.append(lines[j].strip())
            j += 1
        current_seq = ''.join(sequence_lines)

    elif line.startswith("TOPCONS predicted topology:"):
        topology_lines = []
        j = i + 1
        while j < len(lines) and not lines[j].startswith("OCTOPUS predicted topology:"):
            topology_lines.append(lines[j].strip())
            j += 1
        current_topology = ''.join(topology_lines)

        # At this point, we have id, sequence, and topology
        if current_id and current_seq and current_topology:
            # Find TM regions (M blocks)
            inside_tm = False
            start = 0
            for idx, char in enumerate(current_topology):
                if char == "M" and not inside_tm:
                    start = idx
                    inside_tm = True
                elif char != "M" and inside_tm:
                    end = idx
                    inside_tm = False
                    # Extract with padding
                    padded_start = max(0, start - padding)
                    padded_end = min(len(current_seq), end + padding)
                    new_seq = current_seq[padded_start:padded_end]
                    new_id = f"{current_id}_TM{start}-{end}"
                    extracted_records.append(SeqRecord(Seq(new_seq), id=new_id, description=""))
            # Handle if ends with M
            if inside_tm:
                padded_start = max(0, start - padding)
                padded_end = len(current_seq)
                new_seq = current_seq[padded_start:padded_end]
                new_id = f"{current_id}_TM{start}-{padded_end}"
                extracted_records.append(SeqRecord(Seq(new_seq), id=new_id, description=""))

            # Reset for next protein
            current_id = None
            current_seq = None
            current_topology = None

# Save all extracted TM regions
SeqIO.write(extracted_records, output_fasta, "fasta")

print(f"✅ Extracted {len(extracted_records)} TM regions and saved to {output_fasta}")
