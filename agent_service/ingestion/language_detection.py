from pathlib import Path
import pandas as pd
from sentence_transformers import SentenceTransformer
from langdetect import detect, DetectorFactory
import json

DetectorFactory.seed = 0

EVIDENCE_ROOT = Path(__file__).resolve.parent().parent() / "evidence_library"
NORM = EVIDENCE_ROOT / "02_normalised"
META = EVIDENCE_ROOT / "05_metadata" / "documents.csv"


def detect_language(text):
    if not text.strip():
        return "unknown"
    try:
        lang, prob = detect(text)
        return lang, prob
    except Exception:
        return "unknown"

def main():
    df = pd.read_csv(META, dtype=str, keep_default_na=False).astype("object")
    for i, doc_id in enumerate(df["doc_id"]):
        file_path =  EVIDENCE_ROOT/"03_chunks"/f"{doc_id}.chunks.jsonl"
        jsonObj = pd.read_json(path_or_buf=file_path, lines=True)
        print(doc_id)
        try:
            text = ' '.join(jsonObj["text"])
        except:
            continue
        try:
            lang = detect(text)
        except:
            continue
        df.loc[i, 'language'] = lang
        if lang == 'en':
            language_action = 'include'
        else:
            language_action = "exclude_non_english"
        df.loc[i, 'language_action'] = language_action
        
    print(df)
    df.to_csv(META, index = False, encoding = "utf-8")

if __name__ == "__main__":
    main()