import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import json 
import datetime 
import faiss #pip install faiss
EVIDENCE_ROOT = Path(r"C:\Users\Yinpeng Li\CLIMsystems Dropbox\Yinpeng Li\climsystems_ai\evidence_library")
INDEX_DIR  = EVIDENCE_ROOT/ "04_index"
INDEX_FILE = INDEX_DIR / "index.faiss"
RECORDS_FILE = INDEX_DIR / "records.jsonl"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2" 

QUERY_TEST_CASES = [

    # --- Smoke tests ---
    ["climate risk disclosure requirements", 10, "", "", "", "smoke test"],
    ["flood risk methodology", 10, "", "flood", "", "smoke test"],
    ["physical climate risk assessment", 10, "", "", "", "smoke test"],
    ["climate scenario analysis", 10, "", "", "", "smoke test"],

    # --- Standards & disclosure ---
    ["ISSB S2 physical climate risk disclosure", 10, "", "", "standard", "issb"],
    ["TCFD recommended disclosures physical risk", 10, "", "", "standard", "tcfd"],
    ["TNFD dependency and impact assessment", 10, "", "", "standard", "tnfd"],
    ["climate-related financial disclosures governance strategy risk metrics", 10, "", "", "standard", ""],

    # --- Methodology & technical ---
    ["flood hazard modelling methodology", 10, "", "flood", "methodology", ""],
    ["how is flood risk quantified", 10, "", "flood", "methodology", ""],
    ["physical risk score calculation", 10, "", "", "methodology", ""],
    ["exposure vulnerability hazard relationship", 10, "", "", "methodology", ""],

    # --- Jurisdiction-specific ---
    ["New Zealand flood risk assessment", 10, "nz", "flood", "", ""],
    ["Australia climate risk disclosure requirements", 10, "au", "", "standard", ""],
    ["NZ climate hazard data methodology", 10, "nz", "", "methodology", ""],

    # --- Peril-specific ---
    ["flood hazard exposure assessment", 10, "", "flood", "", ""],
    ["heat stress impact on assets", 10, "", "heat", "", ""],
    ["coastal inundation sea level rise risk", 10, "", "slr", "", ""],
    ["drought water stress dependency", 10, "", "drought", "", ""],

    # --- Explain-style queries ---
    ["explain how physical climate risk is assessed", 10, "", "", "", "explain"],
    ["what data is used for flood risk scoring", 10, "", "flood", "", "explain"],
    ["how are dependencies and impacts evaluated", 10, "", "", "", "explain"],

    # --- Cluster sanity checks ---
    ["supply chain climate risk", 10, "", "", "", "cluster test"],
    ["portfolio climate risk assessment", 10, "", "", "", "cluster test"],
    ["financial impact of physical climate risk", 10, "", "", "", "cluster test"],

    # --- Negative / edge cases ---
    ["carbon offset pricing methodology", 10, "", "", "", "negative"],
    ["scope 3 emissions calculation", 10, "", "", "", "negative"],
    ["biodiversity net gain policy", 10, "", "", "", "negative"],

    # --- Short vs long ---
    ["flood risk", 10, "", "flood", "", "short"],
    [
        "how should organisations disclose physical climate risks related to flooding "
        "and extreme rainfall under ISSB standards",
        10, "", "flood", "standard", "long"
    ],
]


def main():
    f= open(r"C:\Users\Yinpeng Li\CLIMsystems Dropbox\Yinpeng Li\climsystems_ai\agent_service\output.txt", "a", encoding = "utf-8") 
    if not INDEX_FILE.is_file():
        print(f"{INDEX_FILE} does not exist")
    if not RECORDS_FILE.is_file():
        print(f"{RECORDS_FILE} does not exist")
    index = faiss.read_index(str(INDEX_FILE))
    records = []
    record_df = pd.read_json(path_or_buf = RECORDS_FILE, lines = True)
    for i, row in record_df.iterrows():
        records.append(row)
    if len(records) != index.ntotal:
        print("Records length does not match index length")
    model = SentenceTransformer(EMBED_MODEL_NAME)
    for q, k, juris, peril, doc_type, note in QUERY_TEST_CASES:
        query = q #input("Enter query: ")
        top_k = k#int(input("top_k (default 10): "))

        
        filter_doc_type = juris#input("doc type: ")
        filter_jurisdiction = juris#input("jursidiction: ")
        filter_perils_contains = peril#input("perils: ")
        filter_cluster_label_contains = doc_type#input("cluster label keywords: ")
        print(f"===== Query: {query} =====\n===== Top K: {top_k} =====\n===== Doc type: {filter_doc_type} =====\n===== Jurisdiction: {filter_jurisdiction} =====\n===== Perils: {filter_perils_contains} =====\n===== Cluster Label Keywords: {filter_cluster_label_contains} =====", file = f)
        

        q_vec = model.encode([query], normalize_embeddings = True, show_progress_bar = True)
        q_vec = np.asarray(q_vec, dtype="float32")
        retrieve_n = max(top_k * 5, 50)
        distances, indices = index.search(q_vec, retrieve_n)

        results = []
        for rank in range(retrieve_n):
            rec_idx = indices[0][rank]
            if rec_idx <0:
                continue
            rec = records[rec_idx]
            score = distances[0][rank]
            meta = rec.meta
            """
            filtering
            """
            if filter_doc_type and meta["doc_type"] != filter_doc_type:
                continue
            if filter_jurisdiction and meta["jurisdiction"] != filter_jurisdiction:
                continue
            if filter_perils_contains and filter_perils_contains not in meta["perils"]:
                continue
            if filter_cluster_label_contains and meta["cluster_label"] and filter_cluster_label_contains not in meta["cluster_label"]:
                continue
            results.append([score, rec])

            if len(results) == top_k:
                break
        print("Top results: ", file = f)
        if not results:
            print("No matches after filtering. Try removing filters or increasing retrieve_n", file = f)
        for item in results:
            rec = item[1]
            #print(rec.meta)
            score = item[0]
            
            print("---------------------------------------------------", file = f)
            print(f"Score: {round(score, 3)}", file = f)
            print(f"Citation: {rec.citation}", file = f)
            print(f"Doc: {rec.doc_id} Chunk: {rec.chunk_id}", file = f)
            print(f"Type: {rec.meta["doc_type"]} Perils: {rec.meta["perils"]} Jurisdiction: {rec.meta["jurisdiction"]}", file = f)
            print(f"Cluster: {rec.meta["cluster_label"]}", file = f)
            print(f"Anchors: {rec.anchors}", file = f)
            print(f"Snippet: {rec.text[0:300]}", file = f)

if __name__ == "__main__":
    main()