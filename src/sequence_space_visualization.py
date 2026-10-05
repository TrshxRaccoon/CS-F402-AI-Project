"""
sequence_space_visualization.py
Author: Vinayak (Task 2 - Visualization of Sequence Space & Dimensionality Reduction)

Explores whether treating DNA/RNA as a language naturally reveals hidden biological
structures and class separation (disease vs healthy) without explicit model supervision:
1. Vectorizes genomic sequences into k-mer frequency representations (k=3, k=4, k=6).
2. Performs Principal Component Analysis (PCA) in 2D and 3D sequence space.
3. Computes explained variance ratios and identifies top feature loadings driving separation.
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


def load_dataset(file_path):
    """Loads genomic dataset and validates expected columns."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Cannot find dataset at: {file_path}")
    
    df = pd.read_csv(file_path)
    required_cols = {"id", "sequence", "label"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Dataset must contain {required_cols}, found {set(df.columns)}")
    
    return df


def vectorize_kmers(df, k=6, normalize=True):
    """
    Transforms space-separated k-mers into a numerical document-term matrix.
    When normalize=True, converts raw counts to relative term frequencies (L1 norm per sequence)
    to eliminate confounding effects of variable sequence lengths.
    """
    col_name = f"kmers_{k}"
    if col_name not in df.columns:
        # Generate k-mers dynamically if not present
        tokenized_corpus = df["sequence"].apply(
            lambda seq: " ".join([seq[i:i+k] for i in range(len(seq) - k + 1)])
        )
    else:
        tokenized_corpus = df[col_name].astype(str)

    vectorizer = CountVectorizer()
    count_matrix = vectorizer.fit_transform(tokenized_corpus).toarray()
    feature_names = np.array(vectorizer.get_feature_names_out())

    if normalize:
        row_sums = count_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        X = count_matrix / row_sums
    else:
        X = count_matrix.astype(float)

    return X, feature_names


def compute_pca(X, feature_names=None, n_components=3, random_state=42):
    """
    Computes 2D and 3D PCA projections and extracts explained variance ratios
    along with the top contributing k-mers (loadings) for the principal components.
    """
    pca_model = PCA(n_components=n_components, random_state=random_state)
    coords = pca_model.fit_transform(X)
    explained_var = pca_model.explained_variance_ratio_

    loadings_df = None
    if feature_names is not None:
        loadings = []
        for i in range(min(n_components, 2)):
            pc_loading = pca_model.components_[i]
            sorted_idx = np.argsort(np.abs(pc_loading))[::-1][:15]
            for idx in sorted_idx:
                loadings.append({
                    "Principal Component": f"PC{i+1}",
                    "k-mer": feature_names[idx],
                    "Loading Value": pc_loading[idx],
                    "Absolute Loading": abs(pc_loading[idx])
                })
        loadings_df = pd.DataFrame(loadings)

    return coords, explained_var, loadings_df


def compute_tsne(X, n_components=2, perplexity=25, metric="cosine", random_state=42):
    """
    Computes 2D or 3D t-SNE projections preserving local neighborhood topology.
    Uses cosine distance by default to align with high-dimensional k-mer word distributions.
    """
    n_samples = X.shape[0]
    safe_perplexity = min(perplexity, max(5, (n_samples - 1) // 3))
    
    tsne_model = TSNE(
        n_components=n_components,
        perplexity=safe_perplexity,
        metric=metric,
        init="random" if metric != "euclidean" else "pca",
        learning_rate="auto",
        random_state=random_state
    )
    coords = tsne_model.fit_transform(X)
    return coords


def compute_umap(X, n_components=2, n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42):
    """
    Computes 2D or 3D UMAP projections capturing both local and global manifold structure.
    """
    n_samples = X.shape[0]
    safe_neighbors = min(n_neighbors, n_samples - 1)

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=safe_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state
    )
    coords = reducer.fit_transform(X)
    return coords


def evaluate_clustering_quality(embeddings_dict, labels):
    """
    Evaluates cluster compactness and separation across dimensionality reduction methods:
    - Silhouette Score (higher is better, [-1, 1])
    - Davies-Bouldin Index (lower is better, [0, inf))
    - Calinski-Harabasz Index (higher is better)
    """
    metrics = []
    for name, coords in embeddings_dict.items():
        sil = silhouette_score(coords, labels)
        db = davies_bouldin_score(coords, labels)
        ch = calinski_harabasz_score(coords, labels)
        metrics.append({
            "Method": name,
            "Dimensions": coords.shape[1],
            "Silhouette Score": round(float(sil), 4),
            "Davies-Bouldin Index": round(float(db), 4),
            "Calinski-Harabasz Index": round(float(ch), 4)
        })
    return pd.DataFrame(metrics)
