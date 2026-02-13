import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
import json 
import datetime 
import faiss

EVIDENCE_ROOT = Path(r"C:\Users\Yinpeng Li\CLIMsystems Dropbox\Yinpeng Li\climsystems_ai\evidence_library")
META_CSV = EVIDENCE_ROOT/ "05_metadata" / "documents.csv"
CHUNKS_DIR = EVIDENCE_ROOT/ "03_chunks"
INDEX_DIR  = EVIDENCE_ROOT/ "04_index"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2" 
INDEX_BACKEND = "faiss" #OR "chroma"

def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main():
    missing = 0
    total_chunks = 0
    df = pd.read_csv(META_CSV, dtype = str)
    #print(df)
    en_df = df[df.get("language_action", "") == "include"]

    # Choose what you want to index: reviewed or approved
    approved_doc_ids = en_df[en_df["status"].str.lower().isin(["reviewed", "approved"])]
    #en_df = df.loc[df['language_action'] == "include"]
    #print(en_df)
    #approved_doc_ids = en_df.loc[en_df["status"] == "reviewed", "doc_id"]
    #print(approved_doc_ids)
    if len(approved_doc_ids.value_counts()) == 0:
        print("No approved documents, nothing to index")
   
    meta_by_id = {row["doc_id"]: row for _, row in approved_doc_ids.iterrows()}
    records = []
    texts = []
    for doc_id in meta_by_id.keys():
        chunk_file = CHUNKS_DIR / (doc_id + ".chunks.jsonl")
        print(chunk_file)
        if not chunk_file.is_file():
            print(f"{doc_id} chunks missing")
            missing +=1
            continue
        doc_meta = meta_by_id[doc_id]
        #chunks = pd.read_json(path_or_buf=chunk_file, lines=True)
        #print(chunks)
        #doc_meta = df.loc[i]
        for chunk in iter_jsonl(chunk_file):
            #print(row)
            #chunk = row
            #print(chunk)
            text = (chunk.get("text") or "").strip()
            if not text:
                continue
            record = {
                "id": chunk.get("citation", f"DOC:{doc_id}.c{chunk.get('chunk_id','')}"),
                "text": text,
                "doc_id": doc_id,
                "chunk_id": chunk.get("chunk_id", ""),
                "citation": chunk.get("citation", ""),
                "anchors": chunk.get("anchors", []),
                "meta": {
                    "doc_type": doc_meta.get("doc_type") or doc_meta.get("auto_doc_type", ""),
                    "perils": doc_meta.get("perils") or doc_meta.get("auto_perils", ""),
                    "jurisdiction": doc_meta.get("jurisdiction") or doc_meta.get("auto_jurisdiction", ""),
                    "auto_doc_type": doc_meta.get("auto_doc_type", ""),
                    "auto_perils":  doc_meta.get("auto_perils", ""),
                    "auto_jurisdiction": doc_meta.get("auto_jurisdiction", ""),
                    "cluster_label": doc_meta.get("cluster_label", ""),
                    "title": doc_meta.get("title") or doc_meta.get("detected_title", ""),
                    "date": doc_meta.get("date", ""),
                    "version": doc_meta.get("version", ""),
                    "source": doc_meta.get("source", ""),
                }
            }
            records.append(record)
            texts.append(text)
            total_chunks+=1
        if not records:
            print("No chunks found for approved docs")
            return
    model = SentenceTransformer(EMBED_MODEL_NAME)
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_DIR / "index.faiss"))
    with (INDEX_DIR / "records.jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    manifest = {
        "created_at_utc": datetime.datetime.now().isoformat(),
        "embedding_model": EMBED_MODEL_NAME,
        "num_chunks": len(records),
        "num_documents": len(approved_doc_ids),
        "vector_dim": dim,
    }

    (INDEX_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8"
    )

    print("Index build complete.")
    print(f"Index directory: {INDEX_DIR}")

if __name__ == "__main__":
    main()

# continue from C:\Users\Yinpeng Li\CLIMsystems Dropbox\Yinpeng Li\climsystems_ai\evidence_library\03_chunks\doc_ra_ver2_v2.chunks.jsonl