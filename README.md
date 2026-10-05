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
│   └── 4_combinatorial_explosion_and_sparsity.png
├── results/               # Statistical summary tables (CSV)
│   ├── summary_baseline_stats.csv
│   ├── summary_sparsity_evaluation.csv
│   ├── top_disease_enriched_3mers.csv
│   └── top_disease_enriched_6mers.csv
├── src/                   # Source code
│   ├── generate_mock_data.py
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

3. **Run Sequence Statistics (Tasks 1 & 2):**
   ```bash
   python src/sequence_statistics.py --data data/mock_genomic_data.csv
   ```

---

## Team Integration Guide

### For Omkar (Data Pipeline)
* Place the real curated biological dataset inside `data/` (e.g. `data/real_genomic_data.csv`).
* Contract columns: `id`, `sequence`, `label`. (Optional: `kmers_3`, `kmers_4`, `kmers_6`).
* When uploaded, simply re-run:
  ```bash
  python src/sequence_statistics.py --data data/real_genomic_data.csv
  ```

### For Vinayak (Visualization)
* The figures directory (`figures/`) contains high-resolution plots for:
  * Length & GC distributions
  * Most frequent $k$-mers ("stop words")
  * Disease vs Healthy differential enrichment ($\log_2 \text{FC}$)
  * Sparsity vs Vocabulary explosion curves

### For Parth (BoW & TF-IDF) & Sahil (Word2Vec)
* Refer to `results/summary_sparsity_evaluation.csv`:
  * $k=3$: Vocabulary size = 64, Document-Term Sparsity = **1.00%**
  * $k=6$: Vocabulary size = 4,096, Document-Term Sparsity = **91.79%**
* Parth: Make sure to use sparse representations (`scipy.sparse.csr_matrix`) for $k \ge 6$.
* Sahil: 6-mers provide a rich vocabulary (~4,000 tokens) suitable for Skip-gram/CBOW context windows.
