# AI and the Language of DNA/RNA
**CS-F402 AI Project — Midsem Roadmap**

Team Members:
* **Omkar**: Pipeline & Data Representation (Task 1)
* **Arya**: Sequence Statistics & Quantitative Language Exploration (Tasks 1 & 2)
* **Vinayak**: Dimensionality Reduction & Visualization (Task 2)
* **Parth**: Classical NLP Baseline (BoW & TF-IDF) (Task 3)
* **Sahil**: Advanced NLP & Word2Vec Embeddings (Task 3)

---

## Repository Structure
```text
.
├── data/                  # Datasets (mock and real)
│   └── mock_genomic_data.csv
├── docs/                  # Project specifications and guidelines
│   └── AI_Language_DNA_RNA_Project_Plan.html.pdf
├── figures/               # Output charts (300 DPI, presentation-ready)
│   ├── 1_sequence_length_and_gc_distribution.png
│   ├── 2_top_3mer_frequencies.png
│   ├── 2_top_6mer_frequencies.png
│   ├── 3_3mer_differential_enrichment.png
│   ├── 3_6mer_differential_enrichment.png
│   ├── 4_combinatorial_explosion_and_sparsity.png
│   ├── 5_pca_sequence_space_2d_and_3d.png
│   ├── 6_tsne_sequence_space_2d_and_3d.png
│   ├── 7_umap_sequence_space_2d_and_3d.png
│   ├── 8_dimensionality_reduction_comparison_dashboard.png
│   ├── 9_kmer_resolution_clustering_comparison.png
│   ├── 10_word2vec_kmer_vocabulary_pca.png
│   ├── 11_word2vec_sequence_space_pca.png
│   └── interactive_sequence_space_dashboard.html
├── models/                # Trained neural embedding models
│   ├── dna2vec_kmers_4.model
│   └── doc2vec_kmers_4.model
├── results/               # Statistical summary tables and evaluation metrics (CSV)
│   ├── dimensionality_reduction_metrics.csv
│   ├── kmer_embeddings.csv
│   ├── nlp_representations_comparison.csv
│   ├── pca_top_kmer_loadings.csv
│   ├── sequence_embeddings_doc2vec.csv
│   ├── sequence_embeddings_word2vec_mean.csv
│   ├── sequence_embeddings_word2vec_tfidf.csv
│   ├── summary_baseline_stats.csv
│   ├── summary_sparsity_evaluation.csv
│   ├── top_disease_enriched_3mers.csv
│   └── top_disease_enriched_6mers.csv
├── src/                   # Source code
│   ├── dna_word2vec_embeddings.py
│   ├── generate_mock_data.py
│   ├── sequence_space_visualization.py
│   └── sequence_statistics.py
├── requirements.txt       # Environment dependencies
└── README.md
```

---

## Setup & Quickstart

1. **Clone the repository:**
   ```bash
   git clone https://github.com/TrshxRaccoon/CS-F402-AI-Project.git
   cd CS-F402-AI-Project
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Sequence Statistics (Tasks 1 & 2 - Arya):**
   ```bash
   python src/sequence_statistics.py --data data/mock_genomic_data.csv
   ```

4. **Run Sequence Space Visualization (Task 2 - Vinayak):**
   ```bash
   python src/sequence_space_visualization.py --data data/mock_genomic_data.csv
   ```

5. **Run Advanced NLP & Sequence Embeddings (Task 3 - Sahil):**
   ```bash
   python src/dna_word2vec_embeddings.py --data data/mock_genomic_data.csv
   ```

---

## Team Integration & Deliverables

### For Omkar (Data Pipeline)
* Place the real curated biological dataset inside `data/` (e.g. `data/real_genomic_data.csv`).
* Contract columns: `id`, `sequence`, `label`. (Optional: `kmers_3`, `kmers_4`, `kmers_6`).
* When uploaded, simply re-run:
  ```bash
  python src/sequence_statistics.py --data data/real_genomic_data.csv
  python src/sequence_space_visualization.py --data data/real_genomic_data.csv
  python src/dna_word2vec_embeddings.py --data data/real_genomic_data.csv
  ```

### For Vinayak (Visualization - Task 2 Deliverables)
* **Dimensionality Reduction Methods**: Implemented 2D and 3D PCA, t-SNE, and UMAP on normalized $k$-mer frequency representations.
* **Unsupervised Class Clustering**: Evaluated natural separation between disease and healthy classes without model supervision:
  * t-SNE (2D) achieved a silhouette score of `0.1901` and Calinski-Harabasz index of `18.24`.
  * UMAP (3D) achieved a silhouette score of `0.1554` and Davies-Bouldin index of `2.0988`.
  * PCA (2D) separated global variance with `0.0503` silhouette score and identified top driving motif loadings.
* **Motif Driver Identification**: Top principal component loadings (`results/pca_top_kmer_loadings.csv`) revealed high-magnitude discriminative motifs such as `gcgcgc` and `cgcgcg` on PC1 and PC2.
* **Token Resolution Analysis**: Multi-panel resolution comparison (`figures/9_kmer_resolution_clustering_comparison.png`) demonstrated how transitioning from coarse 3-mers (64 tokens) to 6-mers (4,096 tokens) enables fine-grained cluster separation.
* **Interactive Presentation Dashboard**: Generated an interactive HTML dashboard (`figures/interactive_sequence_space_dashboard.html`) allowing pan, zoom, and metadata inspection of individual sequences.

### For Sahil (Advanced NLP & Continuous Embeddings - Task 3 Deliverables)
* **Continuous Representation Models**: Trained dense Skip-Gram Word2Vec (dna2vec, dimension=64, window=5) and Doc2Vec (Paragraph Vector) on tokenized $k$-mer sequences.
* **Sequence-Level Representations**: Formulated three distinct dense embedding strategies:
  * *Word2Vec Mean-Pooled*: Standard baseline compressing sequences into dense 64D vectors (0.00% sparsity).
  * *Word2Vec TF-IDF Weighted*: Down-weights ubiquitous background motifs and amplifies rare informative features, improving the intra/inter-class separation ratio to `1.0306`.
  * *Doc2Vec Direct*: Direct sequence-level representations capturing document-level compositional structure.
* **Semantic & Structural Mutation Testing**:
  * Evaluated single-nucleotide biological transitions ($A \leftrightarrow G$, $C \leftrightarrow T$, cosine sim ~`0.62–0.71`) versus transversions (purine $\leftrightarrow$ pyrimidine, cosine sim dropped to `0.3064`).
  * Assessed reverse complements ($5' \rightarrow 3'$) and sliding-window stride overlap effects (`ATGC` vs `TGCA`, cosine sim ~`0.6551`).
* **Cross-Evaluation against Classical TF-IDF Baseline (with Parth)**:
  * Documented comparative metrics in `results/nlp_representations_comparison.csv`:
    * Classical TF-IDF: 256 dimensions, 27.27% sparsity.
    * Advanced Word2Vec/Doc2Vec: 64 dimensions, 0.00% sparsity.
* **Deliverables for Teammates**:
  * Handoff to Vinayak & Task 4 (Classification): Exported labeled feature matrices (`results/sequence_embeddings_word2vec_mean.csv`, `sequence_embeddings_word2vec_tfidf.csv`, and `sequence_embeddings_doc2vec.csv`).
  * Vocabulary Map: Exported `results/kmer_embeddings.csv` and generated `figures/10_word2vec_kmer_vocabulary_pca.png` and `figures/11_word2vec_sequence_space_pca.png`.

### For Parth (BoW & TF-IDF Baseline)
* Refer to `results/summary_sparsity_evaluation.csv`:
  * $k=3$: Vocabulary size = 64, Document-Term Sparsity = **1.00%**
  * $k=6$: Vocabulary size = 4,096, Document-Term Sparsity = **91.79%**
* Parth: Make sure to use sparse representations (`scipy.sparse.csr_matrix`) for $k \ge 6$.
* Sahil's TF-IDF weighted Word2Vec pipeline can directly ingest your final TF-IDF feature matrix when exported.
