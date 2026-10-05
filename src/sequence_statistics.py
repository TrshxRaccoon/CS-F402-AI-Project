"""
sequence_statistics.py
Author: Arya (Tasks 1 & 2 - Sequence Statistics & Exploration)

Performs comprehensive quantitative analysis on genomic sequences:
1. Baseline corpus & sequence statistics (lengths, GC content, nucleotide distributions)
2. k-mer frequency analysis (Top common 'stop words', rare words, vocabulary richness)
3. Disease vs Healthy differential k-mer enrichment (Log2 Fold Change)
4. Combinatorial explosion & matrix sparsity analysis across k=3, 4, 6
5. Generates publication/presentation-ready visualization charts
"""

import os
import argparse
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual style for presentation-ready figures
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({'font.sans-serif': 'Arial', 'font.family': 'sans-serif'})

def load_data(file_path):
    """Loads dataset and ensures k-mer columns exist, generating them on the fly if needed."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Cannot find dataset at: {file_path}. Please check the path.")
    
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} sequences from {file_path}.")
    
    # Ensure kmers columns exist for downstream analysis
    for k in [3, 4, 6]:
        col = f"kmers_{k}"
        if col not in df.columns:
            print(f"Generating missing {col} column dynamically from raw sequences...")
            df[col] = df["sequence"].apply(lambda seq: " ".join([seq[i:i+k] for i in range(len(seq) - k + 1)]))
            
    return df

def analyze_baseline_statistics(df, output_dir):
    """Computes basic length, GC content, and nucleotide distribution."""
    print("\n" + "="*50)
    print("1. BASELINE SEQUENCE STATISTICS")
    print("="*50)
    
    df["length"] = df["sequence"].apply(len)
    df["gc_content"] = df["sequence"].apply(
        lambda s: (s.count("G") + s.count("C")) / len(s) * 100 if len(s) > 0 else 0
    )
    
    # Nucleotide overall frequencies
    all_bases = "".join(df["sequence"])
    base_counts = Counter(all_bases)
    total_bases = sum(base_counts.values())
    base_freqs = {b: (base_counts[b] / total_bases) * 100 for b in ["A", "C", "G", "T"]}
    
    summary = {
        "Total Sequences": len(df),
        "Disease Count": (df["label"] == "disease").sum(),
        "Healthy Count": (df["label"] == "healthy").sum(),
        "Mean Length (bp)": df["length"].mean(),
        "Median Length (bp)": df["length"].median(),
        "Min Length (bp)": df["length"].min(),
        "Max Length (bp)": df["length"].max(),
        "Mean GC Content (%)": df["gc_content"].mean(),
        "Disease Mean GC (%)": df[df["label"] == "disease"]["gc_content"].mean(),
        "Healthy Mean GC (%)": df[df["label"] == "healthy"]["gc_content"].mean()
    }
    
    summary_df = pd.DataFrame(list(summary.items()), columns=["Metric", "Value"])
    print(summary_df.to_string(index=False))
    
    print("\nBase Composition (%):")
    for b, pct in base_freqs.items():
        print(f"  {b}: {pct:.2f}%")
        
    # Plot Sequence Length and GC Content Distributions
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    sns.histplot(data=df, x="length", hue="label", kde=True, ax=axes[0], bins=20)
    axes[0].set_title("Sequence Length Distribution by Class", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Sequence Length (base pairs)")
    axes[0].set_ylabel("Count")
    
    sns.boxplot(data=df, x="label", y="gc_content", ax=axes[1])
    axes[1].set_title("GC Content (%) by Class", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Class Label")
    axes[1].set_ylabel("GC Content (%)")
    
    plt.tight_layout()
    fig_path = os.path.join(output_dir, "1_sequence_length_and_gc_distribution.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved figure: {fig_path}")
    
    return summary_df

def compute_kmer_frequencies(df, k=3):
    """Computes overall and class-specific k-mer counts."""
    col = f"kmers_{k}"
    
    overall_counter = Counter()
    disease_counter = Counter()
    healthy_counter = Counter()
    
    for _, row in df.iterrows():
        tokens = str(row[col]).split()
        overall_counter.update(tokens)
        if row["label"] == "disease":
            disease_counter.update(tokens)
        else:
            healthy_counter.update(tokens)
            
    return overall_counter, disease_counter, healthy_counter

def analyze_kmer_patterns(df, k=3, output_dir="figures"):
    """Analyzes frequent k-mers ('stop words'), rare k-mers, and disease-enriched motifs."""
    print("\n" + "="*50)
    print(f"2. {k}-MER FREQUENCY & PATTERN ANALYSIS")
    print("="*50)
    
    overall, disease, healthy = compute_kmer_frequencies(df, k=k)
    total_tokens = sum(overall.values())
    total_disease = sum(disease.values())
    total_healthy = sum(healthy.values())
    
    top_10 = overall.most_common(10)
    least_10 = overall.most_common()[:-11:-1]
    
    print(f"\nTop 10 Most Common {k}-mers ('Genomic Stop Words'):")
    top_df = pd.DataFrame([{"k-mer": km, "Count": cnt, "Frequency (%)": (cnt/total_tokens)*100} for km, cnt in top_10])
    print(top_df.to_string(index=False))
    
    print(f"\nTop 10 Rarest {k}-mers:")
    rare_df = pd.DataFrame([{"k-mer": km, "Count": cnt, "Frequency (%)": (cnt/total_tokens)*100} for km, cnt in least_10])
    print(rare_df.to_string(index=False))
    
    # Enrichment Analysis: Log2 Fold Change between Disease and Healthy
    enrichment = []
    all_kmers = set(overall.keys())
    for km in all_kmers:
        freq_d = (disease[km] + 1) / (total_disease + len(all_kmers))  # Laplace smoothing
        freq_h = (healthy[km] + 1) / (total_healthy + len(all_kmers))
        log2_fc = np.log2(freq_d / freq_h)
        enrichment.append({
            "k-mer": km,
            "disease_count": disease[km],
            "healthy_count": healthy[km],
            "log2_fold_change": log2_fc
        })
        
    enrich_df = pd.DataFrame(enrichment).sort_values(by="log2_fold_change", ascending=False)
    
    # Plot Top 15 most frequent k-mers
    plt.figure(figsize=(12, 5))
    top_15 = overall.most_common(15)
    km_names, km_counts = zip(*top_15)
    sns.barplot(x=list(km_names), y=list(km_counts), color="#1f77b4")
    plt.title(f"Top 15 Most Frequent {k}-mers ('Genomic Stop Words')", fontsize=14, fontweight="bold")
    plt.xlabel(f"{k}-mer Token")
    plt.ylabel("Raw Frequency Count")
    plt.tight_layout()
    fig_path = os.path.join(output_dir, f"2_top_{k}mer_frequencies.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved figure: {fig_path}")
    
    # Plot Disease vs Healthy Differential Enrichment
    plt.figure(figsize=(12, 6))
    diff_subset = pd.concat([enrich_df.head(10), enrich_df.tail(10)])
    colors = ["#d62728" if fc > 0 else "#2ca02c" for fc in diff_subset["log2_fold_change"]]
    
    sns.barplot(data=diff_subset, x="k-mer", y="log2_fold_change", hue="k-mer", palette=colors, legend=False)
    plt.title(f"Differential {k}-mer Enrichment (Disease vs Healthy Log2 Fold Change)", fontsize=14, fontweight="bold")
    plt.xlabel(f"{k}-mer Token")
    plt.ylabel("Log2 Fold Change (>0: Disease Enriched, <0: Healthy Enriched)")
    plt.axhline(0, color="black", linestyle="--", linewidth=0.8)
    plt.tight_layout()
    diff_path = os.path.join(output_dir, f"3_{k}mer_differential_enrichment.png")
    plt.savefig(diff_path, dpi=300)
    plt.close()
    print(f"Saved figure: {diff_path}")
    
    return enrich_df

def analyze_combinatorial_explosion_and_sparsity(df, k_values=[3, 4, 6], output_dir="figures"):
    """
    Evaluates vocabulary expansion and matrix sparsity from k=3 to k=6.
    4^k theoretical vs observed vocabulary.
    """
    print("\n" + "="*50)
    print("3. COMBINATORIAL EXPLOSION & SPARSITY EVALUATION")
    print("="*50)
    
    results = []
    
    for k in k_values:
        theoretical_vocab = 4 ** k
        col = f"kmers_{k}"
        
        vocab = set()
        zero_count_in_docs = 0
        total_matrix_elements = len(df) * theoretical_vocab
        
        doc_counters = [Counter(str(row[col]).split()) for _, row in df.iterrows()]
        
        for c in doc_counters:
            vocab.update(c.keys())
            unseen = theoretical_vocab - len(c)
            zero_count_in_docs += unseen
            
        observed_vocab = len(vocab)
        vocab_richness_pct = (observed_vocab / theoretical_vocab) * 100
        sparsity_pct = (zero_count_in_docs / total_matrix_elements) * 100
        
        results.append({
            "k": k,
            "Theoretical Vocab (4^k)": theoretical_vocab,
            "Observed Vocab": observed_vocab,
            "Vocabulary Coverage (%)": round(vocab_richness_pct, 2),
            "Document-Term Sparsity (%)": round(sparsity_pct, 2)
        })
        
    comb_df = pd.DataFrame(results)
    print(comb_df.to_string(index=False))
    
    # Plot Sparsity vs Vocabulary
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    color1 = '#1f77b4'
    ax1.set_xlabel('k-mer size (k)', fontsize=12, fontweight="bold")
    ax1.set_ylabel('Theoretical Vocabulary Size (Log Scale)', color=color1, fontsize=12, fontweight="bold")
    ax1.plot(comb_df['k'], comb_df['Theoretical Vocab (4^k)'], color=color1, marker='o', linewidth=2, label="Vocab Size")
    ax1.set_yscale('log')
    ax1.tick_params(axis='y', labelcolor=color1)
    
    ax2 = ax1.twinx()
    color2 = '#d62728'
    ax2.set_ylabel('Document-Term Matrix Sparsity (%)', color=color2, fontsize=12, fontweight="bold")
    ax2.plot(comb_df['k'], comb_df['Document-Term Sparsity (%)'], color=color2, marker='s', linestyle='--', linewidth=2, label="Sparsity (%)")
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim([0, 105])
    
    plt.title("Combinatorial Explosion: Vocabulary Expansion vs Matrix Sparsity", fontsize=14, fontweight="bold")
    plt.tight_layout()
    sparsity_path = os.path.join(output_dir, "4_combinatorial_explosion_and_sparsity.png")
    plt.savefig(sparsity_path, dpi=300)
    plt.close()
    print(f"Saved figure: {sparsity_path}")
    
    return comb_df

def run_analysis(data_path="data/mock_genomic_data.csv", output_dir="figures", results_dir="results"):
    """Main execution function for Arya's tasks."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"Running Sequence Statistics & Exploration Pipeline on: {data_path}")
    df = load_data(data_path)
    
    # 1. Baseline statistics
    summary_df = analyze_baseline_statistics(df, output_dir)
    
    # 2. k-mer pattern analysis for k=3 and k=6
    enrich_3 = analyze_kmer_patterns(df, k=3, output_dir=output_dir)
    enrich_6 = analyze_kmer_patterns(df, k=6, output_dir=output_dir)
    
    # 3. Combinatorial explosion & sparsity
    comb_df = analyze_combinatorial_explosion_and_sparsity(df, k_values=[3, 4, 6], output_dir=output_dir)
    
    # Save CSV tables to results directory
    summary_df.to_csv(os.path.join(results_dir, "summary_baseline_stats.csv"), index=False)
    comb_df.to_csv(os.path.join(results_dir, "summary_sparsity_evaluation.csv"), index=False)
    enrich_3.head(20).to_csv(os.path.join(results_dir, "top_disease_enriched_3mers.csv"), index=False)
    enrich_6.head(20).to_csv(os.path.join(results_dir, "top_disease_enriched_6mers.csv"), index=False)
    
    print("\n" + "="*50)
    print("SUCCESS: ANALYSIS COMPLETE!")
    print(f"Reports saved in: {os.path.abspath(results_dir)}")
    print(f"Figures saved in: {os.path.abspath(output_dir)}")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genomic Sequence Statistics & Language Exploration")
    parser.add_argument("--data", type=str, default="data/mock_genomic_data.csv", help="Path to input sequence CSV")
    parser.add_argument("--figures", type=str, default="figures", help="Directory to save generated charts")
    parser.add_argument("--results", type=str, default="results", help="Directory to save summary tables")
    args = parser.parse_args()
    
    run_analysis(data_path=args.data, output_dir=args.figures, results_dir=args.results)
