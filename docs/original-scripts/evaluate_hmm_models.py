# evaluate_hmm_models.py

def parse_tblout(filename):
    evalues = []
    with open(filename, 'r') as file:
        for line in file:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            if len(parts) > 4:  
                try:
                    evalue = float(parts[4])  # E-value is in the 5th column
                    evalues.append(evalue)
                except ValueError:
                    continue
    return evalues

def evaluate_model(positive_file, gpcr_file, typeII_file, globular_file, evalue_threshold=0.05):
    TP_evalues = parse_tblout(positive_file)
    GPCR_evalues = parse_tblout(gpcr_file)
    TypeII_evalues = parse_tblout(typeII_file)
    Globular_evalues = parse_tblout(globular_file)

    TP = len([e for e in TP_evalues if e <= evalue_threshold])
    FN = len(TP_evalues) - TP

    FP = len([e for e in GPCR_evalues + TypeII_evalues + Globular_evalues if e <= evalue_threshold])
    TN = (len(GPCR_evalues) + len(TypeII_evalues) + len(Globular_evalues)) - FP

    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

    print("\n=== Positive Set E-values ===")
    print(TP_evalues)
    print("\nResults:")
    print(f"True Positives (TP): {TP}")
    print(f"False Negatives (FN): {FN}")
    print(f"False Positives (FP): {FP}")
    print(f"True Negatives (TN): {TN}")
    print(f"Sensitivity: {sensitivity:.3f}")
    print(f"Specificity: {specificity:.3f}")

# Edit your file paths if needed
evaluate_model(
    positive_file='results_long_positive.tbl',
    gpcr_file='results_long_gpcr.tbl',
    typeII_file='results_long_typeII.tbl',
    globular_file='results_long_globular.tbl',
    evalue_threshold=0.05  # Relaxed threshold
)
