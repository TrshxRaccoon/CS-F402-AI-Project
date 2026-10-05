"""
generate_mock_data.py
Creates a realistic mock genomic dataset conforming to Omkar's data contract.
Features:
- Sequence ID
- Raw DNA sequence (with length variation)
- Class label: 'disease' vs 'healthy'
- Injected disease motifs (simulating biological signal)
- Space-delimited k-mers for k=3, k=4, and k=6
"""

import os
import random
import argparse
import pandas as pd

def get_kmers(sequence, k):
    """Generates space-separated k-mers from a raw sequence with a sliding window."""
    return " ".join([sequence[i:i+k] for i in range(len(sequence) - k + 1)])

def generate_mock_dataset(num_samples=100, min_len=200, max_len=500, output_path="data/mock_genomic_data.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    random.seed(42)
    bases = ['A', 'C', 'G', 'T']
    
    # Disease-specific motifs to simulate realistic biological differences
    disease_motifs = ["CGCGCG", "TATAAA", "GGATCC"]
    
    data = []
    for i in range(1, num_samples + 1):
        seq_id = f"SEQ_{i:04d}"
        label = "disease" if random.random() < 0.5 else "healthy"
        length = random.randint(min_len, max_len)
        
        # Base sequence generation with realistic nucleotide probabilities
        if label == "disease":
            # Slightly higher GC bias and injected motifs
            seq_list = random.choices(bases, weights=[0.22, 0.28, 0.28, 0.22], k=length)
            motif = random.choice(disease_motifs)
            insert_pos = random.randint(10, length - len(motif) - 10)
            seq_list[insert_pos:insert_pos + len(motif)] = list(motif)
        else:
            # Healthy sequences
            seq_list = random.choices(bases, weights=[0.26, 0.24, 0.24, 0.26], k=length)
            
        sequence = "".join(seq_list)
        
        kmers_3 = get_kmers(sequence, 3)
        kmers_4 = get_kmers(sequence, 4)
        kmers_6 = get_kmers(sequence, 6)
        
        data.append({
            "id": seq_id,
            "sequence": sequence,
            "label": label,
            "kmers_3": kmers_3,
            "kmers_4": kmers_4,
            "kmers_6": kmers_6
        })
        
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Mock dataset successfully generated with {len(df)} samples -> {output_path}")
    print(f"Class distribution:\n{df['label'].value_counts()}")
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate mock genomic dataset.")
    parser.add_argument("--samples", type=int, default=100, help="Number of sequences to generate")
    parser.add_argument("--output", type=str, default="data/mock_genomic_data.csv", help="Output file path")
    args = parser.parse_args()
    generate_mock_dataset(num_samples=args.samples, output_path=args.output)
