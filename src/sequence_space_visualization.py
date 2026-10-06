"""
sequence_space_visualization.py
Author: Vinayak (Task 2 - Visualization of Sequence Space & Dimensionality Reduction)

Explores whether treating DNA/RNA as a language naturally reveals hidden biological
structures and class separation (disease vs healthy) without explicit model supervision:
1. Vectorizes genomic sequences into k-mer frequency representations (k=3, k=4, k=6).
2. Performs Principal Component Analysis (PCA) in 2D and 3D sequence space.
3. Applies non-linear manifold learning via t-SNE and UMAP in 2D and 3D.
4. Evaluates quantitative cluster separation (Silhouette, Davies-Bouldin, Calinski-Harabasz).
5. Generates high-resolution 2D/3D presentation charts and an interactive visualization dashboard.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set consistent presentation styling
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({'font.sans-serif': 'Arial', 'font.family': 'sans-serif'})

COLOR_PALETTE = {
    "disease": "#D9381E",   # Crimson Red
    "healthy": "#2A9D8F"    # Forest Teal
}


def load_dataset(file_path):
    """Loads genomic dataset and validates expected columns."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Cannot find dataset at: {file_path}")
    
    df = pd.read_csv(file_path)
    required_cols = {"id", "sequence", "label"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Dataset must contain {required_cols}, found {set(df.columns)}")
    
    # Calculate GC content if not present for visual storytelling
    if "gc_content" not in df.columns:
        df["gc_content"] = df["sequence"].apply(
            lambda s: (s.count("G") + s.count("C")) / len(s) * 100 if len(s) > 0 else 0
        )
    if "length" not in df.columns:
        df["length"] = df["sequence"].apply(len)
        
    return df


def vectorize_kmers(df, k=6, normalize=True):
    """
    Transforms space-separated k-mers into a numerical document-term matrix.
    When normalize=True, converts raw counts to relative term frequencies (L1 norm per sequence)
    to eliminate confounding effects of variable sequence lengths.
    """
    col_name = f"kmers_{k}"
    if col_name not in df.columns:
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
                    "Loading Value": round(float(pc_loading[idx]), 5),
                    "Absolute Loading": round(float(abs(pc_loading[idx])), 5)
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


def _draw_confidence_ellipse(x, y, ax, n_std=1.8, facecolor="none", edgecolor="black", **kwargs):
    """Draws a 2D covariance confidence ellipse for class clustering visualization."""
    if len(x) < 4:
        return
    cov = np.cov(x, y)
    denom = np.sqrt(cov[0, 0] * cov[1, 1])
    if denom == 0:
        return
    pearson = np.clip(cov[0, 1] / denom, -1.0, 1.0)
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, edgecolor=edgecolor, **kwargs)
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)
    transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean_x, mean_y)
    ellipse.set_transform(transf + ax.transData)
    ax.add_patch(ellipse)


def plot_projection_2d_and_3d(coords_2d, coords_3d, labels, method_name, output_path,
                              axis_labels_2d, axis_labels_3d, silhouette_val=None):
    """
    Creates a dual-panel figure (2D projection with confidence ellipses and 3D scatter plot)
    formatted for presentation slides.
    """
    fig = plt.figure(figsize=(16, 6))

    # 1. 2D Scatter Plot
    ax1 = fig.add_subplot(1, 2, 1)
    for label_val, color in COLOR_PALETTE.items():
        mask = (labels == label_val)
        x_pts = coords_2d[mask, 0]
        y_pts = coords_2d[mask, 1]
        ax1.scatter(x_pts, y_pts, c=color, label=label_val.capitalize(),
                    alpha=0.85, s=60, edgecolors="white", linewidth=0.6)
        _draw_confidence_ellipse(x_pts, y_pts, ax1, edgecolor=color, linestyle="--", linewidth=1.5, alpha=0.7)
        # Class Centroid Marker
        ax1.scatter(np.mean(x_pts), np.mean(y_pts), c=color, marker="X", s=180, edgecolors="black", linewidth=1.2)

    sil_str = f" | Silhouette: {silhouette_val:.3f}" if silhouette_val is not None else ""
    ax1.set_title(f"{method_name} (2D Sequence Space{sil_str})", fontsize=14, fontweight="bold")
    ax1.set_xlabel(axis_labels_2d[0], fontsize=12)
    ax1.set_ylabel(axis_labels_2d[1], fontsize=12)
    ax1.legend(loc="best", frameon=True)

    # 2. 3D Scatter Plot
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    for label_val, color in COLOR_PALETTE.items():
        mask = (labels == label_val)
        ax2.scatter(coords_3d[mask, 0], coords_3d[mask, 1], coords_3d[mask, 2],
                    c=color, label=label_val.capitalize(), alpha=0.85, s=55,
                    edgecolors="white", linewidth=0.5)

    ax2.set_title(f"{method_name} (3D Sequence Space)", fontsize=14, fontweight="bold")
    ax2.set_xlabel(axis_labels_3d[0], fontsize=10, labelpad=8)
    ax2.set_ylabel(axis_labels_3d[1], fontsize=10, labelpad=8)
    ax2.set_zlabel(axis_labels_3d[2], fontsize=10, labelpad=8)
    ax2.view_init(elev=24, azim=45)
    ax2.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved figure: {output_path}")


def plot_comparison_dashboard(df, embeddings_2d, metrics_df, output_path):
    """
    Creates an executive presentation dashboard comparing PCA, t-SNE, and UMAP
    side-by-side in 2D sequence space with performance metrics.
    """
    labels = df["label"].values
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    plot_configs = [
        ("PCA (Linear Decomposition)", embeddings_2d["PCA-2D"], "PC1", "PC2"),
        ("t-SNE (Local Manifold)", embeddings_2d["t-SNE-2D"], "t-SNE 1", "t-SNE 2"),
        ("UMAP (Topological Manifold)", embeddings_2d["UMAP-2D"], "UMAP 1", "UMAP 2")
    ]

    for ax, (title, coords, xlabel, ylabel) in zip(axes, plot_configs):
        method_key = title.split(" ")[0] + "-2D"
        metric_row = metrics_df[metrics_df["Method"] == method_key]
        sil_val = metric_row["Silhouette Score"].values[0] if not metric_row.empty else None

        for label_val, color in COLOR_PALETTE.items():
            mask = (labels == label_val)
            x_pts = coords[mask, 0]
            y_pts = coords[mask, 1]
            ax.scatter(x_pts, y_pts, c=color, label=label_val.capitalize(),
                       alpha=0.85, s=65, edgecolors="white", linewidth=0.6)
            _draw_confidence_ellipse(x_pts, y_pts, ax, edgecolor=color, linestyle="--", linewidth=1.5, alpha=0.7)
            ax.scatter(np.mean(x_pts), np.mean(y_pts), c=color, marker="X", s=150, edgecolors="black", linewidth=1.2)

        sil_text = f"\nSilhouette Score: {sil_val:.3f}" if sil_val is not None else ""
        ax.set_title(f"{title}{sil_text}", fontsize=13, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.legend(loc="best", frameon=True)

    plt.suptitle("Sequence Space Dimensionality Reduction Dashboard (k=6 Representation)",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved dashboard figure: {output_path}")


def plot_kmer_resolution_comparison(df, output_path):
    """
    Compares sequence clustering across token lengths (k=3 vs k=6) in 2D PCA space,
    illustrating how language token granularity affects biological class separation.
    """
    labels = df["label"].values
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    for idx, k in enumerate([3, 6]):
        X_k, _ = vectorize_kmers(df, k=k, normalize=True)
        pca_k = PCA(n_components=2, random_state=42)
        coords = pca_k.fit_transform(X_k)
        var_k = pca_k.explained_variance_ratio_
        sil = silhouette_score(coords, labels)

        ax = axes[idx]
        for label_val, color in COLOR_PALETTE.items():
            mask = (labels == label_val)
            x_pts = coords[mask, 0]
            y_pts = coords[mask, 1]
            ax.scatter(x_pts, y_pts, c=color, label=label_val.capitalize(),
                       alpha=0.85, s=65, edgecolors="white", linewidth=0.6)
            _draw_confidence_ellipse(x_pts, y_pts, ax, edgecolor=color, linestyle="--", linewidth=1.5, alpha=0.7)
            ax.scatter(np.mean(x_pts), np.mean(y_pts), c=color, marker="X", s=160, edgecolors="black", linewidth=1.2)

        vocab_size = 4 ** k
        ax.set_title(f"k={k} Tokens ({vocab_size} possible words)\n"
                     f"PC1: {var_k[0]*100:.1f}%, PC2: {var_k[1]*100:.1f}% | Silhouette: {sil:.3f}",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel(f"PC1 ({var_k[0]*100:.1f}% var)", fontsize=11)
        ax.set_ylabel(f"PC2 ({var_k[1]*100:.1f}% var)", fontsize=11)
        ax.legend(loc="best", frameon=True)

    plt.suptitle("Impact of k-mer Token Resolution on Sequence Space Clustering",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved k-mer comparison figure: {output_path}")


def export_interactive_dashboard(df, embeddings_2d, output_path, w2v_embeddings_2d=None):
    """
    Exports an interactive Plotly HTML dashboard enabling pan, zoom, and sequence metadata inspection.
    Includes both discrete k-mer and continuous Word2Vec embedding sequence spaces when available.
    """
    labels = df["label"].values
    hover_texts = [
        f"<b>ID:</b> {row['id']}<br>"
        f"<b>Class:</b> {row['label']}<br>"
        f"<b>Length:</b> {row['length']} bp<br>"
        f"<b>GC Content:</b> {row['gc_content']:.1f}%"
        for _, row in df.iterrows()
    ]

    has_w2v = (w2v_embeddings_2d is not None and len(w2v_embeddings_2d) > 0)
    n_rows = 2 if has_w2v else 1
    
    subplot_titles = [
        "k-mer Space (k=6): PCA", "k-mer Space (k=6): t-SNE", "k-mer Space (k=6): UMAP"
    ]
    if has_w2v:
        subplot_titles.extend([
            "Word2Vec Space: PCA", "Word2Vec Space: t-SNE", "Word2Vec Space: UMAP"
        ])

    fig = make_subplots(
        rows=n_rows, cols=3,
        subplot_titles=tuple(subplot_titles),
        horizontal_spacing=0.08,
        vertical_spacing=0.14 if has_w2v else 0.08
    )

    methods_kmer = [
        ("PCA-2D", "PC1", "PC2", 1),
        ("t-SNE-2D", "t-SNE 1", "t-SNE 2", 2),
        ("UMAP-2D", "UMAP 1", "UMAP 2", 3)
    ]

    for method_key, xlabel, ylabel, col_idx in methods_kmer:
        coords = embeddings_2d[method_key]
        for label_val, color in COLOR_PALETTE.items():
            mask = (labels == label_val)
            fig.add_trace(
                go.Scatter(
                    x=coords[mask, 0],
                    y=coords[mask, 1],
                    mode="markers",
                    name=label_val.capitalize(),
                    legendgroup=label_val,
                    showlegend=(col_idx == 1),
                    text=[hover_texts[i] for i in np.where(mask)[0]],
                    hoverinfo="text",
                    marker=dict(size=8, color=color, line=dict(width=1, color="white"))
                ),
                row=1, col=col_idx
            )
        fig.update_xaxes(title_text=xlabel, row=1, col=col_idx)
        fig.update_yaxes(title_text=ylabel, row=1, col=col_idx)

    if has_w2v:
        methods_w2v = [
            ("Word2Vec-PCA-2D", "PC1", "PC2", 1),
            ("Word2Vec-t-SNE-2D", "t-SNE 1", "t-SNE 2", 2),
            ("Word2Vec-UMAP-2D", "UMAP 1", "UMAP 2", 3)
        ]
        for method_key, xlabel, ylabel, col_idx in methods_w2v:
            coords = w2v_embeddings_2d[method_key]
            for label_val, color in COLOR_PALETTE.items():
                mask = (labels == label_val)
                fig.add_trace(
                    go.Scatter(
                        x=coords[mask, 0],
                        y=coords[mask, 1],
                        mode="markers",
                        name=label_val.capitalize(),
                        legendgroup=label_val,
                        showlegend=False,
                        text=[hover_texts[i] for i in np.where(mask)[0]],
                        hoverinfo="text",
                        marker=dict(size=8, color=color, line=dict(width=1, color="white"))
                    ),
                    row=2, col=col_idx
                )
            fig.update_xaxes(title_text=xlabel, row=2, col=col_idx)
            fig.update_yaxes(title_text=ylabel, row=2, col=col_idx)

    fig.update_layout(
        title_text="DNA/RNA Sequence Space Visualization Dashboard: k-mer vs Word2Vec",
        template="plotly_white",
        height=900 if has_w2v else 550,
        width=1350,
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5)
    )

    fig.write_html(output_path)
    print(f"Saved interactive HTML dashboard: {output_path}")

def run_sequence_space_pipeline(data_path="data/mock_genomic_data.csv",
                                output_dir="figures",
                                results_dir="results",
                                k=6):
    """Executes the complete Task 2 visualization & dimensionality reduction pipeline."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    print("\n" + "="*60)
    print(f"RUNNING SEQUENCE SPACE VISUALIZATION PIPELINE (k={k})")
    print(f"Dataset: {data_path}")
    print("="*60)

    # 1. Ingest Data & Vectorize
    df = load_dataset(data_path)
    labels = df["label"].values
    X, feature_names = vectorize_kmers(df, k=k, normalize=True)
    print(f"Document-term matrix constructed: {X.shape[0]} sequences x {X.shape[1]} features.")

    # 2. PCA (2D & 3D) + Loadings
    pca_3d, pca_explained, loadings_df = compute_pca(X, feature_names=feature_names, n_components=3)
    pca_2d = pca_3d[:, :2]
    print(f"PCA Explained Variance: PC1={pca_explained[0]*100:.2f}%, "
          f"PC2={pca_explained[1]*100:.2f}%, PC3={pca_explained[2]*100:.2f}%")

    # 3. t-SNE (2D & 3D)
    tsne_2d = compute_tsne(X, n_components=2, perplexity=25, metric="cosine")
    tsne_3d = compute_tsne(X, n_components=3, perplexity=25, metric="cosine")

    # 4. UMAP (2D & 3D)
    umap_2d = compute_umap(X, n_components=2, n_neighbors=15, min_dist=0.1, metric="cosine")
    umap_3d = compute_umap(X, n_components=3, n_neighbors=15, min_dist=0.1, metric="cosine")

    embeddings_2d = {
        "PCA-2D": pca_2d,
        "t-SNE-2D": tsne_2d,
        "UMAP-2D": umap_2d
    }
    all_embeddings = {
        "PCA-2D": pca_2d,
        "PCA-3D": pca_3d,
        "t-SNE-2D": tsne_2d,
        "t-SNE-3D": tsne_3d,
        "UMAP-2D": umap_2d,
        "UMAP-3D": umap_3d
    }

    # Ingest team continuous sequence embeddings if available
    w2v_path = os.path.join(results_dir, "sequence_embeddings_word2vec_tfidf.csv")
    w2v_embeddings_2d = None
    if os.path.exists(w2v_path):
        print(f"Loading team Word2Vec embeddings from: {w2v_path}")
        w2v_df = pd.read_csv(w2v_path)
        dim_cols = [c for c in w2v_df.columns if c.startswith("dim_")]
        if len(dim_cols) > 0:
            X_w2v = w2v_df[dim_cols].values
            w2v_pca_3d, _, _ = compute_pca(X_w2v, n_components=3)
            w2v_pca_2d = w2v_pca_3d[:, :2]
            w2v_tsne_2d = compute_tsne(X_w2v, n_components=2, perplexity=25, metric="cosine")
            w2v_umap_2d = compute_umap(X_w2v, n_components=2, n_neighbors=15, min_dist=0.1, metric="cosine")
            w2v_embeddings_2d = {
                "Word2Vec-PCA-2D": w2v_pca_2d,
                "Word2Vec-t-SNE-2D": w2v_tsne_2d,
                "Word2Vec-UMAP-2D": w2v_umap_2d
            }
            all_embeddings.update({
                "Word2Vec-PCA-2D": w2v_pca_2d,
                "Word2Vec-PCA-3D": w2v_pca_3d,
                "Word2Vec-t-SNE-2D": w2v_tsne_2d,
                "Word2Vec-UMAP-2D": w2v_umap_2d
            })

    # 5. Quantitative Clustering Evaluation
    metrics_df = evaluate_clustering_quality(all_embeddings, labels)
    print("\nDimensionality Reduction Clustering Evaluation:")
    print(metrics_df.to_string(index=False))

    # Save CSV reports
    metrics_path = os.path.join(results_dir, "dimensionality_reduction_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved clustering metrics table: {metrics_path}")

    if loadings_df is not None:
        loadings_path = os.path.join(results_dir, "pca_top_kmer_loadings.csv")
        loadings_df.to_csv(loadings_path, index=False)
        print(f"Saved PCA top loadings table: {loadings_path}")

    # 6. Generate Publication/Presentation Plots
    sil_pca = metrics_df.loc[metrics_df["Method"] == "PCA-2D", "Silhouette Score"].values[0]
    sil_tsne = metrics_df.loc[metrics_df["Method"] == "t-SNE-2D", "Silhouette Score"].values[0]
    sil_umap = metrics_df.loc[metrics_df["Method"] == "UMAP-2D", "Silhouette Score"].values[0]

    plot_projection_2d_and_3d(
        pca_2d, pca_3d, labels,
        method_name="Principal Component Analysis (PCA)",
        output_path=os.path.join(output_dir, "5_pca_sequence_space_2d_and_3d.png"),
        axis_labels_2d=[f"PC1 ({pca_explained[0]*100:.1f}% var)", f"PC2 ({pca_explained[1]*100:.1f}% var)"],
        axis_labels_3d=[f"PC1 ({pca_explained[0]*100:.1f}%)", f"PC2 ({pca_explained[1]*100:.1f}%)", f"PC3 ({pca_explained[2]*100:.1f}%)"],
        silhouette_val=sil_pca
    )

    plot_projection_2d_and_3d(
        tsne_2d, tsne_3d, labels,
        method_name="t-SNE Manifold",
        output_path=os.path.join(output_dir, "6_tsne_sequence_space_2d_and_3d.png"),
        axis_labels_2d=["t-SNE Dimension 1", "t-SNE Dimension 2"],
        axis_labels_3d=["t-SNE 1", "t-SNE 2", "t-SNE 3"],
        silhouette_val=sil_tsne
    )

    plot_projection_2d_and_3d(
        umap_2d, umap_3d, labels,
        method_name="UMAP Manifold",
        output_path=os.path.join(output_dir, "7_umap_sequence_space_2d_and_3d.png"),
        axis_labels_2d=["UMAP Dimension 1", "UMAP Dimension 2"],
        axis_labels_3d=["UMAP 1", "UMAP 2", "UMAP 3"],
        silhouette_val=sil_umap
    )

    plot_comparison_dashboard(
        df, embeddings_2d, metrics_df,
        output_path=os.path.join(output_dir, "8_dimensionality_reduction_comparison_dashboard.png")
    )

    plot_kmer_resolution_comparison(
        df,
        output_path=os.path.join(output_dir, "9_kmer_resolution_clustering_comparison.png")
    )

    export_interactive_dashboard(
        df, embeddings_2d,
        output_path=os.path.join(output_dir, "interactive_sequence_space_dashboard.html"),
        w2v_embeddings_2d=w2v_embeddings_2d
    )

    print("\n" + "="*60)
    print("TASK 2 VISUALIZATION PIPELINE COMPLETE")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genomic Sequence Space Visualization & Dimensionality Reduction (Task 2)")
    parser.add_argument("--data", type=str, default="data/mock_genomic_data.csv", help="Path to input sequence CSV")
    parser.add_argument("--k", type=int, default=6, help="k-mer token size for representation (default: 6)")
    parser.add_argument("--figures", type=str, default="figures", help="Directory to save generated charts")
    parser.add_argument("--results", type=str, default="results", help="Directory to save summary tables")
    args = parser.parse_args()

    run_sequence_space_pipeline(
        data_path=args.data,
        output_dir=args.figures,
        results_dir=args.results,
        k=args.k
    )
