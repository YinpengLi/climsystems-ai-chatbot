from collections import Counter
import pandas as pd
from pathlib import Path

EVIDENCE_ROOT = Path(__file__).resolve.parent().parent() / "evidence_library"
NORM = EVIDENCE_ROOT / "02_normalised"
META = EVIDENCE_ROOT / "05_metadata" / "documents.csv"
EXCLUSION_LIST = ['...', 'climate', 'change', 'vol']

def top_keywords(df, cluster_id):
    mask = df['cluster_id'] == str(cluster_id)
    keywords = str(df.loc[mask, 'top_keywords'])
    keywords = keywords.replace(';',' ')
    keywords = keywords.split()
    #print(keywords)

    most_common=[w for w,c in Counter(keywords).most_common(6) if w not in EXCLUSION_LIST]
    return most_common

def mode(series):
    vals = [v.strip() for v in series.tolist() if str(v).strip()]
    return Counter(vals).most_common(1)[0][0] if vals else ""

def main():
    df = pd.read_csv(META, dtype=str, keep_default_na=False).astype("object")
    #cluster_id = list(df["cluster_id"])
    #df_mask = list(cluster_id.mask(df['cluster_id']!=''))
    #cluster_id.sort()
    #print(cluster_id)
    #cluster_num = cluster_id[-1]
    #print(cluster_num)
    for cluster_id in range(176):
        rows = df[df["cluster_id"] == str(cluster_id)]
        kws = top_keywords(df, cluster_id)
        dt = mode(rows["auto_doc_type"]) if "auto_doc_type" in df.columns else ""
        peril = mode(rows["auto_perils"]) if "auto_perils" in df.columns else ""
        juris = mode(rows["auto_jurisdiction"]) if "auto_jurisdiction" in df.columns else ""
        parts = []
        #print(f"peril: {peril}")
        if dt: parts.append(dt)
        parts.append(' '.join(kws))
        tail = []
        if peril: tail.append(peril.replace(";", "/"))
        if juris: tail.append(juris)
        if tail: parts.append(f"({', '.join(tail)})")
        label = ": ".join([parts[0], " ".join(parts[1:])]) if len(parts) > 1 else parts[0]
        #print(label)
        df.loc[df["cluster_id"] == str(cluster_id), 'cluster_label_suggested'] = label
        #print(df.loc[df["cluster_id"] == cluster_id, 'cluster_label_suggested'])
    #print(df['cluster_label_suggested'])
    df.to_csv(META, index=False, encoding="utf-8")
    print("Done: wrote cluster_label_suggested")

if __name__ == "__main__":
      main()