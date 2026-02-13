import json
from pathlib import Path
import pandas as pd
from sentence_transformers import SentenceTransformer
import hdbscan
import umap


EVIDENCE_ROOT = Path(r"C:\Users\Yinpeng Li\CLIMsystems Dropbox\Yinpeng Li\climsystems_ai\evidence_library")
NORM = EVIDENCE_ROOT / "02_normalised"
META = EVIDENCE_ROOT / "05_metadata" / "documents.csv"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # fast + decent

def load_doc_text(doc_id: str) -> str:
    p = NORM / doc_id / "text.json"
    if not p.exists():
        return ""
    data = json.loads(p.read_text(encoding="utf-8"))
    blocks = data.get("blocks", [])[:6] + data.get("blocks", [])[int(len(data.get("blocks", []))/2) -1 :int(len(data.get("blocks", []))/2)+1] + data.get("blocks", [])[-3:] 
    return "\n".join(b.get("text","") for b in blocks)

def main():
    df = pd.read_csv(META, dtype=str, keep_default_na=False).astype("object")
    for c in ["cluster_id","cluster_label","cluster_size","cluster_confidence"]:
        if c not in df.columns:
            df[c] = ""

    texts, doc_ids = [], []
    for i, doc_id in enumerate(df["doc_id"].tolist()):
        doc_id = (doc_id or "").strip()
        if not doc_id:
            continue
        if df.loc[i, "language_action"] != "include":
            #print("poop")
            continue
        #print("pee")
        t = load_doc_text(doc_id)
        if t.strip():
            doc_ids.append(doc_id)
            texts.append(t)
    
    if not texts:
        print("No text found (missing text.json).")
        return


    model = SentenceTransformer(MODEL_NAME)
    emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

    # UMAP: reduce to a cluster-friendly space
    reducer = umap.UMAP(
        n_neighbors=15,        # try 10–50
        n_components=10,       # try 5–15
        min_dist=0.0,
        metric="cosine",
        random_state=42
    )
    emb_umap = reducer.fit_transform(emb)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=10,           # try 10–30
        min_samples=1,                # try 5–20
        metric="euclidean",            # in UMAP space
        cluster_selection_method="leaf"
    )
    labels = clusterer.fit_predict(emb_umap)
    probs = clusterer.probabilities_


    id_to = {doc_ids[i]: (int(labels[i]), float(probs[i])) for i in range(len(doc_ids))}

    # write results
    for i in df.index:
        did = df.at[i, "doc_id"]
        if did in id_to:
            cid, conf = id_to[did]
            df.at[i, "cluster_id"] = str(cid)
            df.at[i, "cluster_confidence"] = f"{conf:.2f}"

    # After writing df["cluster_id"] for clustered docs...
    df["cluster_id"] = df["cluster_id"].astype(str).str.strip()

    # Convert to numeric; invalid/blank becomes NaN
    df["cluster_id_int"] = pd.to_numeric(df["cluster_id"], errors="coerce")

    # Compute cluster sizes ONLY for real clusters (exclude -1 and NaN)
    sizes = (
        df.loc[df["cluster_id_int"].notna() & (df["cluster_id_int"] != -1), "cluster_id_int"]
        .value_counts()
        .to_dict()
    )

    # Assign sizes back
    df["cluster_size"] = ""
    mask = df["cluster_id_int"].notna() & (df["cluster_id_int"] != -1)
    df.loc[mask, "cluster_size"] = df.loc[mask, "cluster_id_int"].map(sizes).astype(int).astype(str)
    print(list(df['cluster_id_int']).count(-1))

    vc = df["cluster_id"].value_counts()

    print("Total docs:", len(df))
    print("Clusters (excl -1):", (vc.index != "-1").sum())
    print("Outlier %:", 100 * vc.get("-1", 0) / len(df))
    print("Largest cluster:", vc.drop("-1", errors="ignore").max())
    print("Median cluster size:", vc.drop("-1", errors="ignore").median())


    df.to_csv(META, index=False, encoding="utf-8")

    print("Done. cluster_id=-1 indicates outliers.")

if __name__ == "__main__":
    main()
