"""
cluster.py — Stage 7: Semantic clustering of question records.

Uses sentence-transformers (paraphrase-multilingual-MiniLM-L12-v2) for embeddings,
then HDBSCAN (or AgglomerativeClustering as fallback) to group similar questions.

Writes cluster_id, cluster_size, and is_exemplar (longest member per cluster) to records.

Usage:
    python src/cluster.py
    python src/cluster.py --algorithm agglomerative  # explicit fallback
"""

import argparse
import sys
import pathlib
import logging
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from src.storage import get_db, load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def embed_texts(texts: list[str], model_name: str) -> np.ndarray:
    """Compute L2-normalized sentence embeddings."""
    from sentence_transformers import SentenceTransformer
    log.info("Loading embedding model '%s'...", model_name)
    model = SentenceTransformer(model_name)
    log.info("Encoding %d texts...", len(texts))
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,   # L2-normalize so cosine ≈ dot product
    )
    return embeddings


def cluster_hdbscan(embeddings: np.ndarray, min_cluster_size: int, min_samples: int) -> np.ndarray:
    """Run HDBSCAN. Returns label array (-1 = noise)."""
    import hdbscan
    log.info("Running HDBSCAN (min_cluster_size=%d, min_samples=%d)...", min_cluster_size, min_samples)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",     # on L2-normalized vectors ≈ cosine
        cluster_selection_method="eom",
        core_dist_n_jobs=-1,
    )
    labels = clusterer.fit_predict(embeddings)
    return labels


def cluster_agglomerative(embeddings: np.ndarray, distance_threshold: float, linkage: str) -> np.ndarray:
    """Run AgglomerativeClustering as fallback. Returns label array (no noise label)."""
    from sklearn.cluster import AgglomerativeClustering
    log.info("Running AgglomerativeClustering (distance_threshold=%.2f)...", distance_threshold)
    clusterer = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        linkage=linkage,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(embeddings)
    return labels


def find_exemplars(ids: list[str], bodies: list[str], labels: np.ndarray) -> dict[int, str]:
    """Return {cluster_id: record_id_of_longest_body}."""
    cluster_members: dict[int, list[tuple[int, str, str]]] = {}
    for i, (rid, body, label) in enumerate(zip(ids, bodies, labels)):
        label = int(label)
        if label < 0:  # noise
            continue
        cluster_members.setdefault(label, []).append((i, rid, body or ""))
    exemplars = {}
    for label, members in cluster_members.items():
        best = max(members, key=lambda x: len(x[2]))
        exemplars[label] = best[1]  # record id
    return exemplars


def main():
    parser = argparse.ArgumentParser(description="Cluster question records with sentence embeddings")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--algorithm", choices=["hdbscan", "agglomerative"],
                        help="Override clustering algorithm from config")
    parser.add_argument("--reset", action="store_true", help="Re-cluster from scratch")
    args = parser.parse_args()

    cfg = load_config(args.config)
    conn = get_db(cfg["paths"]["db_path"])
    cluster_cfg = cfg["clustering"]
    algorithm = args.algorithm or cluster_cfg.get("algorithm", "hdbscan")
    model_name = cluster_cfg["model"]

    if args.reset:
        conn.execute("UPDATE records SET cluster_id = NULL, cluster_size = NULL, is_exemplar = NULL")
        conn.commit()

    rows = conn.execute("""
        SELECT id, COALESCE(body_clean, body) as text
        FROM records
        WHERE is_question = TRUE AND body IS NOT NULL AND cluster_id IS NULL
        ORDER BY id
    """).fetchall()

    if not rows:
        log.info("No unclustered records found. Use --reset to re-cluster.")
        conn.close()
        return

    ids = [r[0] for r in rows]
    texts = [r[1] or "" for r in rows]
    log.info("Clustering %d records...", len(ids))

    # Embed
    embeddings = embed_texts(texts, model_name)

    # Cluster
    if algorithm == "hdbscan":
        try:
            labels = cluster_hdbscan(
                embeddings,
                min_cluster_size=cluster_cfg["hdbscan"]["min_cluster_size"],
                min_samples=cluster_cfg["hdbscan"]["min_samples"],
            )
        except ImportError:
            log.warning("hdbscan not installed — falling back to AgglomerativeClustering")
            algorithm = "agglomerative"

    if algorithm == "agglomerative":
        labels = cluster_agglomerative(
            embeddings,
            distance_threshold=cluster_cfg["agglomerative"]["distance_threshold"],
            linkage=cluster_cfg["agglomerative"]["linkage"],
        )

    # Compute cluster sizes
    label_arr = labels.astype(int)
    unique, counts = np.unique(label_arr[label_arr >= 0], return_counts=True)
    cluster_size_map: dict[int, int] = dict(zip(unique.tolist(), counts.tolist()))

    # Find exemplars (longest body per cluster)
    bodies = texts
    exemplar_ids = find_exemplars(ids, bodies, label_arr)

    # Write back to DB
    log.info("Writing cluster assignments...")
    batch = []
    for rid, label in zip(ids, label_arr.tolist()):
        label = int(label)
        size = cluster_size_map.get(label, 1) if label >= 0 else 1
        exemplar = (exemplar_ids.get(label) == rid) if label >= 0 else False
        cluster_id = label  # -1 means noise/singleton in HDBSCAN
        batch.append((cluster_id, size, exemplar, rid))
        if len(batch) >= 1000:
            conn.executemany(
                "UPDATE records SET cluster_id = ?, cluster_size = ?, is_exemplar = ? WHERE id = ?",
                batch
            )
            batch = []
    if batch:
        conn.executemany(
            "UPDATE records SET cluster_id = ?, cluster_size = ?, is_exemplar = ? WHERE id = ?",
            batch
        )
    conn.commit()

    # Summary
    n_clustered = sum(1 for l in label_arr if l >= 0)
    n_noise = sum(1 for l in label_arr if l < 0)
    n_clusters = len(unique)
    log.info("Done. %d clusters, %d records clustered, %d noise/singletons",
             n_clusters, n_clustered, n_noise)

    top_clusters = sorted(cluster_size_map.items(), key=lambda x: -x[1])[:10]
    print("\n=== Top 10 clusters by size ===")
    for cid, sz in top_clusters:
        ex_id = exemplar_ids.get(cid, "?")
        ex_text = conn.execute("SELECT COALESCE(body_clean, body) FROM records WHERE id = ?", [ex_id]).fetchone()
        ex_preview = (ex_text[0] or "")[:120].replace("\n", " ") if ex_text else ""
        print(f"  Cluster {cid:>4} ({sz:>3} members): {ex_preview}")

    conn.close()


if __name__ == "__main__":
    main()
