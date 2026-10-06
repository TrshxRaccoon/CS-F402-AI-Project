"""
dna_word2vec_embeddings.py
Author: Sahil (Task 3: Advanced NLP & Continuous Embeddings)

Midsem Project Deliverable - Task 3: Applying NLP ideas to genomic data.
Performs rigorous semantic modeling and continuous sequence embedding:
1. Trains Word2Vec (Skip-Gram dna2vec) and Doc2Vec on tokenized k-mer sequences.
2. Generates sequence-level dense representations:
   - Mean Pooling (unweighted average)
   - TF-IDF Weighted Pooling (down-weighting ubiquitous background k-mers)
   - Doc2Vec Direct (Paragraph Vector continuous representations)
3. Dynamic Hypothesis-Driven Semantic Analysis:
   - Single-nucleotide transitions vs. transversions across any k (k=3, 4, 6)
   - Reverse complement structural relationships
   - Stride-1 overlap artifact vs. biological context
4. Genomic Vector Arithmetic (Analogy Test):
   - Evaluates whether single-nucleotide substitution vectors (e.g., A -> G) preserve
     consistent geometric direction across diverse motif prefixes.
5. Confounder & Artifact Disentanglement:
   - Analyzes whether embedding latent dimensions correlate with sequence length or GC content.
6. Hyperparameter Sensitivity & Architecture Benchmarking:
   - Evaluates Skip-Gram vs. CBOW and varying context window sizes.
7. Generates Publication-Ready Visualizations:
   - 2D PCA of learned vocabulary space (colored by GC content)
   - 2D PCA of sequence space (colored by disease class)
   - Hierarchical Clustermap of canonical motif similarities
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from scipy.stats import pearsonr, spearmanr

# Styling for presentation-ready figures
plt.rcParams.update({"font.sans-serif": ["DejaVu Sans", "Arial", "sans-serif"], "font.family": "sans-serif"})


def load_data(file_path: str) -> pd.DataFrame:
    """Loads dataset and ensures k-mer columns exist, generating them dynamically if needed."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Cannot find dataset at: {file_path}. Please check the path.")

    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} sequences from {file_path}.")

    for k in [3, 4, 6]:
        col = f"kmers_{k}"
        if col not in df.columns and "sequence" in df.columns:
            print(f"Generating missing {col} column dynamically from raw sequences...")
            df[col] = df["sequence"].apply(
                lambda seq: " ".join([seq[i : i + k] for i in range(len(seq) - k + 1)])
            )

    return df


def calculate_gc_content(kmer: str) -> float:
    """Calculates GC percentage of a k-mer string."""
    kmer = kmer.upper()
    gc_count = kmer.count("G") + kmer.count("C")
    return (gc_count / len(kmer)) * 100.0 if len(kmer) > 0 else 0.0


def get_reverse_complement(kmer: str) -> str:
    """Returns the reverse complement of a DNA k-mer."""
    complement = {"A": "T", "T": "A", "G": "C", "C": "G"}
    return "".join(complement.get(base, base) for base in reversed(kmer.upper()))


def choose_target_kmer(w2v_model, requested_kmer: str = None) -> str:
    """Selects canonical motif matching length k from the vocabulary."""
    vocab = list(w2v_model.wv.index_to_key)
    if not vocab:
        return ""
    k = len(vocab[0])

    if requested_kmer and len(requested_kmer) == k and requested_kmer in w2v_model.wv:
        return requested_kmer

    canonical_motifs = ["ATGCAT", "ATGC", "ATG", "CGCGCG", "CGCC", "CGC"]
    for motif in canonical_motifs:
        if len(motif) == k and motif in w2v_model.wv:
            return motif

    return vocab[0]


def generate_dynamic_mutations(target_kmer: str, vocab_set: set) -> dict:
    """Generates biologically and structurally relevant mutations of matching length k."""
    k = len(target_kmer)
    transition_map = {"A": "G", "G": "A", "C": "T", "T": "C"}
    transversion_map = {"A": "C", "G": "T", "C": "A", "T": "G"}

    mutations = {}

    # Transition at position 0
    t_base0 = transition_map.get(target_kmer[0], "G")
    mutations[f"Transition ({target_kmer[0]} -> {t_base0} at pos 0)"] = t_base0 + target_kmer[1:]

    # Transition at end position
    if k > 1:
        t_base_end = transition_map.get(target_kmer[-1], "T")
        mutations[f"Transition ({target_kmer[-1]} -> {t_base_end} at pos {k-1})"] = target_kmer[:-1] + t_base_end

    # Transversion at position 0
    tv_base0 = transversion_map.get(target_kmer[0], "C")
    mutations[f"Transversion ({target_kmer[0]} -> {tv_base0} at pos 0)"] = tv_base0 + target_kmer[1:]

    # Reverse Complement
    mutations["Reverse Complement"] = get_reverse_complement(target_kmer)

    # Shift-1 Stride Overlap
    prefix = target_kmer[1:]
    shift_candidate = prefix + "A"
    for base in ["A", "C", "G", "T"]:
        test_shift = prefix + base
        if test_shift in vocab_set:
            shift_candidate = test_shift
            break
    mutations[f"Shift-1 Stride Overlap ('{shift_candidate}')"] = shift_candidate

    return mutations


# ----------------------------------------------------------------------
# 1. Word2Vec & Doc2Vec Training
# ----------------------------------------------------------------------
def train_word2vec(tokenized_corpus, vector_size=64, window=5, sg=1, seed=42):
    """Trains Word2Vec on tokenized k-mer sentences (sg=1: Skip-Gram)."""
    print(f"\n[1] Training Word2Vec (dna2vec)... (dim={vector_size}, window={window}, sg={sg})")
    model = Word2Vec(
        sentences=tokenized_corpus,
        vector_size=vector_size,
        window=window,
        min_count=1,
        sg=sg,
        workers=4,
        seed=seed,
    )
    print(f"    --> Total learned vocabulary size: {len(model.wv)} k-mers")
    return model


def train_doc2vec(tokenized_corpus, vector_size=64, window=5, epochs=30, seed=42):
    """Trains Doc2Vec on tokenized sequences for sequence-level representations."""
    print(f"\n[2] Training Doc2Vec... (dim={vector_size}, window={window}, epochs={epochs})")
    tagged_docs = [
        TaggedDocument(words=seq, tags=[i]) for i, seq in enumerate(tokenized_corpus)
    ]
    model = Doc2Vec(
        documents=tagged_docs,
        vector_size=vector_size,
        window=window,
        min_count=1,
        workers=4,
        epochs=epochs,
        seed=seed,
    )
    doc_vectors = np.array([model.dv[i] for i in range(len(tokenized_corpus))])
    print(f"    --> Doc2Vec matrix shape: {doc_vectors.shape}")
    return model, doc_vectors


# ----------------------------------------------------------------------
# 2. Sequence-Level Embeddings (Pooling Strategies)
# ----------------------------------------------------------------------
def compute_mean_pooled_vectors(tokenized_corpus, w2v_model, vector_dim=64):
    """Computes unweighted average k-mer vector for each sequence."""
    vectors = []
    for seq in tokenized_corpus:
        valid_vecs = [w2v_model.wv[w] for w in seq if w in w2v_model.wv]
        if valid_vecs:
            vectors.append(np.mean(valid_vecs, axis=0))
        else:
            vectors.append(np.zeros(vector_dim))
    return np.array(vectors)


def compute_tfidf_weighted_vectors(kmer_strings, tokenized_corpus, w2v_model, vector_dim=64):
    """Computes TF-IDF weighted average k-mer vector for each sequence."""
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(kmer_strings)
    feature_names = tfidf.get_feature_names_out()
    vocab_lookup = {word: idx for idx, word in enumerate(feature_names)}

    vectors = []
    for row_idx, seq in enumerate(tokenized_corpus):
        weights = []
        vecs = []
        for word in seq:
            norm_word = word.lower()
            if word in w2v_model.wv and norm_word in vocab_lookup:
                w = tfidf_matrix[row_idx, vocab_lookup[norm_word]]
                weights.append(w)
                vecs.append(w2v_model.wv[word])

        sum_w = sum(weights)
        if vecs and sum_w > 0:
            weighted_avg = np.average(vecs, axis=0, weights=weights)
            vectors.append(weighted_avg)
        else:
            vectors.append(np.zeros(vector_dim))

    return np.array(vectors), tfidf_matrix


# ----------------------------------------------------------------------
# 3. Rigorous Semantic & Geometric Investigation
# ----------------------------------------------------------------------
def investigate_semantic_similarities(w2v_model, requested_kmer: str = None):
    """Examines biological and structural properties in embedding space."""
    print("\n" + "=" * 60)
    print("CONTEXTUAL & SEMANTIC SIMILARITY INVESTIGATION")
    print("=" * 60)

    target_kmer = choose_target_kmer(w2v_model, requested_kmer)
    vocab_set = set(w2v_model.wv.index_to_key)

    print(f"Target k-mer: '{target_kmer}' (len={len(target_kmer)}, GC: {calculate_gc_content(target_kmer):.1f}%)")
    top_similar = w2v_model.wv.most_similar(target_kmer, topn=5)
    print("\nTop 5 Most Similar k-mers (Cosine Similarity):")
    for word, score in top_similar:
        gc = calculate_gc_content(word)
        print(f"  - {word}: {score:.4f}  (GC: {gc:.1f}%)")

    mutations = generate_dynamic_mutations(target_kmer, vocab_set)

    print("\nStructural / Biological Motif Comparisons:")
    for label, candidate in mutations.items():
        if candidate in w2v_model.wv:
            sim = float(
                cosine_similarity(
                    w2v_model.wv[target_kmer].reshape(1, -1),
                    w2v_model.wv[candidate].reshape(1, -1),
                )[0][0]
            )
            print(f"  - {label} ('{candidate}'): similarity = {sim:.4f}")
        else:
            print(f"  - {label} ('{candidate}'): Not in vocabulary")


def evaluate_genomic_vector_analogies(w2v_model):
    """
    Tests vector arithmetic: Do mutation delta vectors have consistent geometric directions?
    e.g. delta(A -> G) across multiple prefixes: v(prefix + G) - v(prefix + A).
    """
    print("\n" + "=" * 60)
    print("GENOMIC VECTOR ARITHMETIC (ANALOGY CONSISTENCY)")
    print("=" * 60)

    vocab = list(w2v_model.wv.index_to_key)
    k = len(vocab[0])
    if k < 2:
        return

    # Sample prefix pairs for A -> G transition
    prefixes = list(set([w[:-1] for w in vocab]))[:12]
    deltas = []
    tested_prefixes = []

    for p in prefixes:
        w_from = p + "A"
        w_to = p + "G"
        if w_from in w2v_model.wv and w_to in w2v_model.wv:
            delta = w2v_model.wv[w_to] - w2v_model.wv[w_from]
            deltas.append(delta)
            tested_prefixes.append(p)

    if len(deltas) >= 3:
        deltas = np.array(deltas)
        sim_matrix = cosine_similarity(deltas)
        upper_tri = sim_matrix[np.triu_indices(len(deltas), k=1)]
        mean_consistency = float(np.mean(upper_tri))
        print(f"Tested A -> G single-base mutation across {len(deltas)} distinct contexts.")
        print(f"Average Directional Cosine Consistency: {mean_consistency:.4f}")
        print("  --> High positive consistency indicates the embedding space preserves")
        print("      linear algebraic geometric relations for biological mutations.")
        return mean_consistency
    return None


def analyze_confounders_and_artifacts(df, sequence_vectors, results_dir="results"):
    """
    Outcome 5 Rigour: Evaluates if sequence embeddings merely reflect superficial
    sequence length or GC composition bias.
    """
    print("\n" + "=" * 60)
    print("CONFOUNDER & ARTIFACT ANALYSIS (GC & LENGTH INDEPENDENCE)")
    print("=" * 60)

    gc_series = df["sequence"].apply(lambda s: (s.count("G") + s.count("C")) / len(s) * 100.0).values
    len_series = df["sequence"].apply(len).values

    pca = PCA(n_components=2, random_state=42)
    seq_pca = pca.fit_transform(sequence_vectors)

    r_gc_pc1, p_gc_pc1 = pearsonr(seq_pca[:, 0], gc_series)
    r_len_pc1, p_len_pc1 = pearsonr(seq_pca[:, 0], len_series)
    r_gc_pc2, p_gc_pc2 = pearsonr(seq_pca[:, 1], gc_series)
    r_len_pc2, p_len_pc2 = pearsonr(seq_pca[:, 1], len_series)

    confounder_records = [
        {"Component": "PC1", "Confounder": "GC Content (%)", "Pearson r": round(r_gc_pc1, 4), "p-value": f"{p_gc_pc1:.2e}"},
        {"Component": "PC1", "Confounder": "Sequence Length", "Pearson r": round(r_len_pc1, 4), "p-value": f"{p_len_pc1:.2e}"},
        {"Component": "PC2", "Confounder": "GC Content (%)", "Pearson r": round(r_gc_pc2, 4), "p-value": f"{p_gc_pc2:.2e}"},
        {"Component": "PC2", "Confounder": "Sequence Length", "Pearson r": round(r_len_pc2, 4), "p-value": f"{p_len_pc2:.2e}"},
    ]

    df_conf = pd.DataFrame(confounder_records)
    print(df_conf.to_string(index=False))
    conf_path = os.path.join(results_dir, "embedding_confounder_analysis.csv")
    df_conf.to_csv(conf_path, index=False)
    print(f"    --> Saved confounder report: {conf_path}")
    return df_conf


def benchmark_hyperparameter_sensitivity(tokenized_corpus, labels, results_dir="results"):
    """
    Empirically benchmarks architectural parameters:
    - Skip-Gram vs. CBOW
    - Narrow (w=2) vs. Standard (w=5) vs. Wide (w=10) Context Windows
    """
    print("\n" + "=" * 60)
    print("HYPERPARAMETER SENSITIVITY BENCHMARK (ARCHITECTURE RIGOUR)")
    print("=" * 60)

    configs = [
        {"name": "Skip-Gram (w=5, d=64) [Chosen]", "sg": 1, "window": 5, "size": 64},
        {"name": "CBOW (w=5, d=64)", "sg": 0, "window": 5, "size": 64},
        {"name": "Skip-Gram Narrow (w=2, d=64)", "sg": 1, "window": 2, "size": 64},
        {"name": "Skip-Gram Wide (w=10, d=64)", "sg": 1, "window": 10, "size": 64},
        {"name": "Skip-Gram LowDim (w=5, d=32)", "sg": 1, "window": 5, "size": 32},
    ]

    records = []
    for cfg in configs:
        m = Word2Vec(tokenized_corpus, vector_size=cfg["size"], window=cfg["window"], sg=cfg["sg"], seed=42, workers=4)
        vecs = np.array([np.mean([m.wv[w] for w in s if w in m.wv], axis=0) for s in tokenized_corpus])
        cos_mat = cosine_similarity(vecs)

        intra, inter = [], []
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                if labels[i] == labels[j]:
                    intra.append(cos_mat[i, j])
                else:
                    inter.append(cos_mat[i, j])

        ratio = np.mean(intra) / (np.mean(inter) + 1e-9)
        records.append({
            "Configuration": cfg["name"],
            "Architecture": "Skip-Gram" if cfg["sg"] == 1 else "CBOW",
            "Window": cfg["window"],
            "Dim": cfg["size"],
            "Intra-Class Sim": round(np.mean(intra), 4),
            "Inter-Class Sim": round(np.mean(inter), 4),
            "Separation Ratio": round(ratio, 4),
        })

    df_sens = pd.DataFrame(records)
    print(df_sens.to_string(index=False))
    sens_path = os.path.join(results_dir, "embedding_hyperparameter_sensitivity.csv")
    df_sens.to_csv(sens_path, index=False)
    print(f"    --> Saved hyperparameter sensitivity report: {sens_path}")
    return df_sens


# ----------------------------------------------------------------------
# 4. Cross-Evaluation: Word2Vec vs. TF-IDF
# ----------------------------------------------------------------------
def evaluate_representations(labels, representations_dict):
    """Computes dimensionality, sparsity, and intra-class vs inter-class separation."""
    print("\n" + "=" * 60)
    print("CROSS-EVALUATION: CLASSICAL (TF-IDF) VS ADVANCED (WORD2VEC/DOC2VEC)")
    print("=" * 60)

    comparison_records = []

    for name, matrix in representations_dict.items():
        if hasattr(matrix, "toarray"):
            dense_mat = matrix.toarray()
            sparsity = 100.0 * (1.0 - (matrix.nnz / (matrix.shape[0] * matrix.shape[1])))
        else:
            dense_mat = np.array(matrix)
            sparsity = 100.0 * (np.sum(dense_mat == 0) / dense_mat.size)

        cos_sim_mat = cosine_similarity(dense_mat)

        intra_sims = []
        inter_sims = []

        n_samples = len(labels)
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                sim = cos_sim_mat[i, j]
                if labels[i] == labels[j]:
                    intra_sims.append(sim)
                else:
                    inter_sims.append(sim)

        avg_intra = np.mean(intra_sims) if intra_sims else 0.0
        avg_inter = np.mean(inter_sims) if inter_sims else 0.0
        separation_ratio = avg_intra / (avg_inter + 1e-9)

        record = {
            "Representation": name,
            "Dimensionality": dense_mat.shape[1],
            "Sparsity (%)": round(sparsity, 2),
            "Mean Intra-Class Sim": round(avg_intra, 4),
            "Mean Inter-Class Sim": round(avg_inter, 4),
            "Separation Ratio (Intra/Inter)": round(separation_ratio, 4),
        }
        comparison_records.append(record)

    df_comp = pd.DataFrame(comparison_records)
    print(df_comp.to_string(index=False))
    return df_comp


# ----------------------------------------------------------------------
# 5. Visualizations
# ----------------------------------------------------------------------
def generate_presentation_visuals(w2v_model, seq_mean_vecs, labels, figures_dir="figures"):
    """Generates 2D PCA plots and Hierarchical Clustermap for presentation."""
    print(f"\n[5] Generating publication-quality visualizations for presentation slides...")
    os.makedirs(figures_dir, exist_ok=True)

    # 1. K-mer Vocabulary Plot
    kmers = list(w2v_model.wv.index_to_key)
    kmer_vecs = np.array([w2v_model.wv[k] for k in kmers])
    gc_contents = np.array([calculate_gc_content(k) for k in kmers])

    pca_kmer = PCA(n_components=2, random_state=42)
    kmer_2d = pca_kmer.fit_transform(kmer_vecs)

    plt.figure(figsize=(8, 6))
    scatter1 = plt.scatter(
        kmer_2d[:, 0], kmer_2d[:, 1], c=gc_contents, cmap="viridis", alpha=0.8, s=40
    )
    cbar = plt.colorbar(scatter1)
    cbar.set_label("GC Content (%)", fontsize=11)
    plt.title(
        f"Learned k-mer Vocabulary Space (PCA, n={len(kmers)})\n"
        f"Explained Variance: {pca_kmer.explained_variance_ratio_.sum()*100:.1f}%",
        fontsize=12,
        fontweight="bold",
    )
    plt.xlabel(f"PC1 ({pca_kmer.explained_variance_ratio_[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({pca_kmer.explained_variance_ratio_[1]*100:.1f}%)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    kmer_plot_path = os.path.join(figures_dir, "10_word2vec_kmer_vocabulary_pca.png")
    plt.savefig(kmer_plot_path, dpi=300)
    plt.close()

    # 2. Sequence Space Plot
    pca_seq = PCA(n_components=2, random_state=42)
    seq_2d = pca_seq.fit_transform(seq_mean_vecs)

    plt.figure(figsize=(8, 6))
    for label in np.unique(labels):
        idx = labels == label
        color = "#e63946" if "disease" in str(label).lower() else "#1d3557"
        plt.scatter(
            seq_2d[idx, 0],
            seq_2d[idx, 1],
            label=f"Class: {label}",
            alpha=0.8,
            s=55,
            color=color,
        )

    plt.title(
        f"Sequence Embedding Space - Word2Vec Mean Pooling (PCA)\n"
        f"Explained Variance: {pca_seq.explained_variance_ratio_.sum()*100:.1f}%",
        fontsize=12,
        fontweight="bold",
    )
    plt.xlabel(f"PC1 ({pca_seq.explained_variance_ratio_[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({pca_seq.explained_variance_ratio_[1]*100:.1f}%)")
    plt.legend(frameon=True)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    seq_plot_path = os.path.join(figures_dir, "11_word2vec_sequence_space_pca.png")
    plt.savefig(seq_plot_path, dpi=300)
    plt.close()

    # 3. Hierarchical Motif Clustermap
    motifs_candidates = [
        "AAAA", "TTTT", "CCCC", "GGGG",
        "ATGC", "GCAT", "CGCG", "GCGC",
        "TATA", "ATAT", "CCGG", "GGCC",
        "AATT", "TTAA", "ACGT", "TGCA",
        "GTAC", "CATG", "AGCT", "TCGA"
    ]
    k_val = len(kmers[0])
    if k_val == 3:
        motifs_candidates = ["AAA", "TTT", "CCC", "GGG", "ATG", "CAT", "CGC", "GCG", "TAT", "ATA", "CCG", "GGC", "AAT", "TTA", "ACG", "TGC"]
    elif k_val == 6:
        motifs_candidates = ["AAAAAA", "TTTTTT", "CCCCCC", "GGGGGG", "ATGCAT", "GCATGC", "CGCGCG", "GCGCGC", "TATATA", "ATATAT", "CCGGCC", "GGCCGG"]

    selected_motifs = [m for m in motifs_candidates if m in w2v_model.wv]
    if len(selected_motifs) >= 6:
        m_vecs = np.array([w2v_model.wv[m] for m in selected_motifs])
        sim_mat = cosine_similarity(m_vecs)

        g = sns.clustermap(
            sim_mat,
            xticklabels=selected_motifs,
            yticklabels=selected_motifs,
            cmap="mako",
            annot=False,
            figsize=(9, 8),
            cbar_kws={'label': 'Cosine Similarity'}
        )
        g.fig.suptitle("Genomic Motif Semantic Clustermap (Word2Vec Cosine Space)", fontsize=12, fontweight="bold", y=1.02)
        clustermap_path = os.path.join(figures_dir, "12_motif_similarity_clustermap.png")
        plt.savefig(clustermap_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"    --> Saved: {clustermap_path}")

    print(f"    --> Saved: {kmer_plot_path}")
    print(f"    --> Saved: {seq_plot_path}")


# ----------------------------------------------------------------------
# 6. Main Pipeline
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="DNA NLP Embeddings Pipeline (Sahil - Task 3)"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/mock_genomic_data.csv",
        help="Path to input sequence CSV dataset",
    )
    parser.add_argument(
        "--kmer_col",
        type=str,
        default="kmers_4",
        choices=["kmers_3", "kmers_4", "kmers_6"],
        help="Column representing tokenized k-mer text",
    )
    parser.add_argument("--dim", type=int, default=64, help="Embedding latent dimension")
    parser.add_argument("--window", type=int, default=5, help="Context window size")
    parser.add_argument("--figures", type=str, default="figures", help="Directory for plots")
    parser.add_argument(
        "--results", type=str, default="results", help="Directory for CSV deliverables"
    )
    parser.add_argument(
        "--models", type=str, default="models", help="Directory for saving trained models"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("AI & THE LANGUAGE OF DNA/RNA - TASK 3: ADVANCED NLP & EMBEDDINGS")
    print(f"Dataset: {args.data} | Column: {args.kmer_col} | Embedding Dim: {args.dim}")
    print("=" * 60)

    os.makedirs(args.figures, exist_ok=True)
    os.makedirs(args.results, exist_ok=True)
    os.makedirs(args.models, exist_ok=True)

    # 1. Load Data
    df = load_data(args.data)
    if args.kmer_col not in df.columns:
        raise ValueError(
            f"Column '{args.kmer_col}' not found in {args.data}. Available: {list(df.columns)}"
        )

    tokenized_corpus = [row.split() for row in df[args.kmer_col]]
    kmer_strings = df[args.kmer_col].tolist()
    labels = df["label"].values

    # 2. Train Models
    w2v_model = train_word2vec(
        tokenized_corpus, vector_size=args.dim, window=args.window, sg=1
    )
    doc2vec_model, doc2vec_matrix = train_doc2vec(
        tokenized_corpus, vector_size=args.dim, window=args.window
    )

    # 3. Generate Sequence Representations
    print("\n[3] Generating Sequence-Level Representations...")
    mean_pooled_matrix = compute_mean_pooled_vectors(
        tokenized_corpus, w2v_model, vector_dim=args.dim
    )
    tfidf_weighted_matrix, tfidf_matrix = compute_tfidf_weighted_vectors(
        kmer_strings, tokenized_corpus, w2v_model, vector_dim=args.dim
    )

    # 4. Contextual & Semantic Analysis
    investigate_semantic_similarities(w2v_model)

    # 5. Genomic Vector Arithmetic (Analogy Test)
    evaluate_genomic_vector_analogies(w2v_model)

    # 6. Confounder & Artifact Analysis (Evaluating GC & Length Bias)
    analyze_confounders_and_artifacts(df, mean_pooled_matrix, results_dir=args.results)

    # 7. Hyperparameter Sensitivity Benchmark
    benchmark_hyperparameter_sensitivity(tokenized_corpus, labels, results_dir=args.results)

    # 8. Cross-Evaluation Against Classical TF-IDF Baseline (Parth's baseline)
    representations_dict = {
        "Classical TF-IDF (Parth)": tfidf_matrix,
        "Word2Vec Mean-Pooled (Sahil)": mean_pooled_matrix,
        "Word2Vec TF-IDF Weighted (Sahil)": tfidf_weighted_matrix,
        "Doc2Vec Direct (Sahil)": doc2vec_matrix,
    }
    df_eval = evaluate_representations(labels, representations_dict)
    eval_csv_path = os.path.join(args.results, "nlp_representations_comparison.csv")
    df_eval.to_csv(eval_csv_path, index=False)
    print(f"    --> Saved evaluation table: {eval_csv_path}")

    # 9. Save Deliverables for Vinayak (Task 2) and Task 4 Classification
    print("\n[9] Exporting Deliverables...")
    dim_cols = [f"dim_{i}" for i in range(args.dim)]

    def export_df(matrix, filename):
        out_df = pd.DataFrame(matrix, columns=dim_cols)
        out_df.insert(0, "id", df["id"])
        out_df.insert(1, "label", df["label"])
        filepath = os.path.join(args.results, filename)
        out_df.to_csv(filepath, index=False)
        print(f"    --> Saved: {filepath} (Shape: {out_df.shape})")

    export_df(mean_pooled_matrix, "sequence_embeddings_word2vec_mean.csv")
    export_df(tfidf_weighted_matrix, "sequence_embeddings_word2vec_tfidf.csv")
    export_df(doc2vec_matrix, "sequence_embeddings_doc2vec.csv")

    # Export k-mer vocabulary embeddings for Vinayak's word space visualization
    kmer_keys = list(w2v_model.wv.index_to_key)
    kmer_vectors = np.array([w2v_model.wv[k] for k in kmer_keys])
    kmer_df = pd.DataFrame(kmer_vectors, columns=dim_cols)
    kmer_df.insert(0, "kmer", kmer_keys)
    kmer_df["gc_content"] = [calculate_gc_content(k) for k in kmer_keys]
    kmer_csv_path = os.path.join(args.results, "kmer_embeddings.csv")
    kmer_df.to_csv(kmer_csv_path, index=False)
    print(f"    --> Saved: {kmer_csv_path} (Shape: {kmer_df.shape})")

    # Save trained models
    w2v_model_path = os.path.join(args.models, f"dna2vec_{args.kmer_col}.model")
    doc2vec_model_path = os.path.join(args.models, f"doc2vec_{args.kmer_col}.model")
    w2v_model.save(w2v_model_path)
    doc2vec_model.save(doc2vec_model_path)
    print(f"    --> Saved model weights: {w2v_model_path}, {doc2vec_model_path}")

    # 10. Generate Presentation Visuals
    generate_presentation_visuals(
        w2v_model, mean_pooled_matrix, labels, figures_dir=args.figures
    )

    print("\n" + "=" * 60)
    print("TASK 3 EXECUTION COMPLETE! READY FOR MIDSEM PRESENTATION.")
    print("=" * 60)


if __name__ == "__main__":
    main()
