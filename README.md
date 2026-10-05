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
│   └── interactive_sequence_space_dashboard.html
├── results/               # Statistical summary tables and evaluation metrics (CSV)
│   ├── dimensionality_reduction_metrics.csv
│   ├── pca_top_kmer_loadings.csv
│   ├── summary_baseline_stats.csv
│   ├── summary_sparsity_evaluation.csv
│   ├── top_disease_enriched_3mers.csv
│   └── top_disease_enriched_6mers.csv
├── src/                   # Source code
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

---

## Team Integration & Deliverables

### For Omkar (Data Pipeline)
* Place the real curated biological dataset inside `data/` (e.g. `data/real_genomic_data.csv`).
* Contract columns: `id`, `sequence`, `label`. (Optional: `kmers_3`, `kmers_4`, `kmers_6`).
* When uploaded, simply re-run:
  ```bash
  python src/sequence_statistics.py --data data/real_genomic_data.csv
  python src/sequence_space_visualization.py --data data/real_genomic_data.csv
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

### For Parth (BoW & TF-IDF) & Sahil (Word2Vec)
* Refer to `results/summary_sparsity_evaluation.csv`:
  * $k=3$: Vocabulary size = 64, Document-Term Sparsity = **1.00%**
  * $k=6$: Vocabulary size = 4,096, Document-Term Sparsity = **91.79%**
* Parth: Make sure to use sparse representations (`scipy.sparse.csr_matrix`) for $k \ge 6$.
* Sahil: 6-mers provide a rich vocabulary (~4,000 tokens) suitable for Skip-gram/CBOW context windows.
