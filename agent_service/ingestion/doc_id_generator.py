"""
doc_id_generator.py

Purpose
- Auto-generate stable, human-readable doc_id values for Evidence Library ingestion.
- Detect duplicates/collisions and ensure uniqueness.
- Optionally write proposals into documents.csv (as draft), so humans can confirm/edit before approval.

Assumptions
- Windows file system
- Evidence library layout as discussed:
  D:\climsystems_ai\evidence_library\
    00_inbox\auto_drop\
    01_sources\
    05_metadata\documents.csv

How to use
1) Edit EVIDENCE_ROOT to your path.
2) Put files into 00_inbox\auto_drop\ (or local/dropbox).
3) Run:  python doc_id_generator.py
4) Check 05_metadata\documents.csv:
   - proposed_doc_id (auto)
   - doc_id (blank until you confirm)
   - status=draft
5) After human review, set doc_id=proposed_doc_id (or edited value) and promote the file to 01_sources/<doc_id>.<ext>.

Notes
- This script does NOT move files by default (safer). It only proposes IDs and updates CSV.
- You can enable "promote_to_sources=True" after you're comfortable.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

# ===========
# CONFIG
# ===========
EVIDENCE_ROOT = Path(__file__).resolve.parent().parent() / "evidence_library"

INBOX_DIRS = [
    EVIDENCE_ROOT / "00_inbox"
]

SOURCES_DIR = EVIDENCE_ROOT / "01_sources"
METADATA_DIR = EVIDENCE_ROOT / "05_metadata"
DOCS_CSV = METADATA_DIR / "documents.csv"

SUPPORTED_EXTS = {".pdf", ".docx", ".pptx"}

# If True, after proposing doc_id, copy originals into 01_sources/<doc_id>.<ext>
# Recommended: keep False until you have reviewed a first batch.
PROMOTE_TO_SOURCES = False

# ===========
# CSV SCHEMA
# ===========
BASE_COLS = [
    "doc_id",  # final ID (locked at approval)
    "proposed_doc_id",  # auto-proposed ID (can be edited before approval)
    "original_filename",
    "source_path",
    "title",
    "version",
    "date",
    "owner",
    "doc_type",
    "scope",
    "perils",
    "jurisdiction",
    "status",
    "source",
    "file_hash_sha256",
    "file_ext",
    "file_size_bytes",
    "file_modified_utc",
    "ingested_at_utc",
]

DEFAULT_ROW = {
    "doc_id": "",
    "title": "",
    "version": "",
    "date": "",
    "owner": "",
    "doc_type": "",
    "scope": "",
    "perils": "",
    "jurisdiction": "",
    "status": "draft",
    "source": "",
}

# ===========
# DOC_ID LOGIC
# ===========

STOPWORDS = {
    "the", "and", "for", "with", "this", "that", "from", "into", "over", "under", "within",
    "final", "draft", "version", "ver", "v", "copy", "new", "latest", "updated", "update",
    "climsystems",  # optional: you can keep/remove; often redundant in doc_id
}

DOC_TYPE_RULES: List[Tuple[str, List[str]]] = [
    ("mthd", ["methodology", "method", "model methodology", "technical note", "tech note", "approach", "calibration"]),
    ("dict", ["dictionary", "data dictionary", "indicator dictionary", "glossary", "definitions"]),
    ("std",  ["ifrs", "issb", "aasb", "asrs", "tcfd", "standard", "guidance", "disclosure"]),
    ("rpt",  ["report", "assessment", "portfolio", "executive summary"]),
    ("ppt",  ["presentation", "slide deck", "slides"]),
    ("qa",   ["qa", "quality assurance", "validation", "verification", "testing"]),
]

PERIL_HINTS = {
    "flood": ["flood", "inundation", "riverine", "pluvial", "coastal flood", "storm surge"],
    "heat": ["heat", "temperature", "hot days", "extreme heat"],
    "fire": ["fire", "wildfire", "bushfire"],
    "wind": ["wind", "cyclone", "storm", "gust"],
    "slr": ["sea level", "sea-level", "slr", "coastal"],
    "drought": ["drought", "dry", "water stress"],
}

JURIS_HINTS = {
    "nz": ["new zealand", "nz", "auckland", "wellington", "canterbury", "waikato", "otago"],
    "au": ["australia", "au", "nsw", "victoria", "queensland", "tasmania", "sa", "wa"],
    "global": ["global", "worldwide", "international"],
}

VERSION_PATTERNS = [
    re.compile(r"\bv(?:ersion)?\s*([0-9]+(?:\.[0-9]+)*)\b", re.IGNORECASE),
    re.compile(r"\bver\s*([0-9]+(?:\.[0-9]+)*)\b", re.IGNORECASE),
]

YEAR_PATTERN = re.compile(r"\b(19[8-9][0-9]|20[0-3][0-9])\b")  # 1980-2039


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def utc_iso_from_ts(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def slugify(text: str, max_len: int = 60) -> str:
    """
    Turn arbitrary text into lowercase underscore slug suitable for IDs.
    """
    text = text.lower().strip()
    # replace separators with space
    text = re.sub(r"[\-/\\|:;,()\[\]{}]+", " ", text)
    # remove non-alphanumeric (keep spaces)
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    tokens = [t for t in text.split() if t and t not in STOPWORDS]
    slug = "_".join(tokens)
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug[:max_len].strip("_")


def infer_doc_type(name: str) -> str:
    low = name.lower()
    # extension-based hint
    if low.endswith(".pptx"):
        return "ppt"
    # keyword rules
    for code, keys in DOC_TYPE_RULES:
        for k in keys:
            if k in low:
                return code
    return "doc"


def infer_peril(name: str) -> str:
    low = name.lower()
    hits = []
    for peril, keys in PERIL_HINTS.items():
        if any(k in low for k in keys):
            hits.append(peril)
    if not hits:
        return "multi"
    # choose first for doc_id simplicity; keep full tags in metadata later
    return hits[0]


def infer_jurisdiction(name: str) -> str:
    low = name.lower()
    # priority: nz/au/global
    if any(k in low for k in JURIS_HINTS["nz"]):
        return "nz"
    if any(k in low for k in JURIS_HINTS["au"]):
        return "au"
    if any(k in low for k in JURIS_HINTS["global"]):
        return "global"
    return ""


def extract_version(text: str) -> str:
    for pat in VERSION_PATTERNS:
        m = pat.search(text)
        if m:
            return m.group(1)
    return ""


def extract_year(text: str) -> str:
    years = YEAR_PATTERN.findall(text)
    if not years:
        return ""
    # choose the latest year present (often best guess)
    return str(max(int(y) for y in years))


def build_doc_id(
    filename: str,
    existing_ids: set[str],
) -> Tuple[str, Dict[str, str]]:
    """
    Create a proposed doc_id from filename alone (fast & safe).
    Returns (unique_doc_id, extra_fields)
    """
    ext = Path(filename).suffix.lower()
    stem = Path(filename).stem

    doc_type = infer_doc_type(filename)
    peril = infer_peril(filename)
    juris = infer_jurisdiction(filename)

    version = extract_version(stem)
    year = extract_year(stem)

    # topic slug from stem (remove very common patterns first)
    cleaned = re.sub(r"\b(final|draft|rev(?:ision)?|copy|latest|updated)\b", " ", stem, flags=re.IGNORECASE)
    topic_slug = slugify(cleaned, max_len=40)

    # If the slug is empty (e.g., filename was symbols), fallback
    if not topic_slug:
        topic_slug = "untitled"

    # Build base ID
    parts = [doc_type, topic_slug]
    if peril and peril != "multi":
        # Often useful; you can comment this out if too noisy
        parts.append(peril)
    if version:
        parts.append(f"v{version.replace('.', '_')}")
    if year:
        parts.append(year)
    if juris:
        parts.append(juris)

    base = "_".join([p for p in parts if p]).lower()
    base = re.sub(r"_+", "_", base).strip("_")

    unique = ensure_unique_id(base, existing_ids)
    extra = {"doc_type": doc_type, "perils": peril, "jurisdiction": juris, "version": version, "date": year}
    return unique, extra


def ensure_unique_id(base_id: str, existing_ids: set[str]) -> str:
    """
    Ensure doc_id is unique in CSV + current run.
    If collision, append _01, _02, ...
    """
    if base_id not in existing_ids:
        existing_ids.add(base_id)
        return base_id

    n = 1
    while True:
        candidate = f"{base_id}_{n:02d}"
        if candidate not in existing_ids:
            existing_ids.add(candidate)
            return candidate
        n += 1


# ===========
# INGEST / CSV
# ===========

def load_or_init_documents_csv() -> pd.DataFrame:
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    if DOCS_CSV.exists() and DOCS_CSV.stat().st_size > 0:
        df = pd.read_csv(DOCS_CSV)
    else:
        df = pd.DataFrame(columns=BASE_COLS)

    # Ensure all required columns exist
    for c in BASE_COLS:
        if c not in df.columns:
            df[c] = ""
    return df


def list_inbox_files() -> List[Path]:
    files: List[Path] = []

    for d in INBOX_DIRS:
        if not d.exists():
            print(f"[WARN] Inbox dir missing: {d}")
            continue

        # Use os.walk for robustness against transient Dropbox paths
        try:
            for root, dirnames, filenames in os.walk(d, topdown=True):
                # Optional: prune hidden/system folders to reduce Dropbox weirdness
                dirnames[:] = [
                    dn for dn in dirnames
                    if not dn.startswith("~") and dn not in {".git", "__pycache__"}
                ]

                for fn in filenames:
                    ext = Path(fn).suffix.lower()
                    if ext not in SUPPORTED_EXTS:
                        continue

                    p = Path(root) / fn
                    # Confirm it is a file and accessible (may still fail for placeholders)
                    try:
                        if p.is_file():
                            files.append(p)
                    except (FileNotFoundError, OSError) as e:
                        print(f"[SKIP] Unreadable file during scan: {p} | {e}")
                        continue

        except (FileNotFoundError, OSError) as e:
            # If the root folder becomes unavailable mid-walk
            print(f"[SKIP] Walk failed for inbox dir: {d} | {e}")
            continue
    print(f"Scan complete: {len(files)} files found")

    return files




def detect_source_bucket(path: Path) -> str:
    # crude but practical
    p = str(path).lower()
    if "dropbox" in p:
        return "dropbox"
    if "auto_drop" in p:
        return "auto_drop"
    return "local"


def upsert_row(df: pd.DataFrame, row: Dict[str, str]) -> pd.DataFrame:
    """
    Upsert by file hash if present, else by source_path.
    """
    file_hash = row.get("file_hash_sha256", "")
    if file_hash:
        hit = df.index[df["file_hash_sha256"] == file_hash]
        if len(hit) > 0:
            i = hit[0]
            # Only update empty fields (do not overwrite human-edited values)
            for k, v in row.items():
                if k not in df.columns:
                    continue
                if str(df.loc[i, k]).strip() == "" and str(v).strip() != "":
                    df.loc[i, k] = v
            return df

    # fallback: source_path match
    sp = row.get("source_path", "")
    hit = df.index[df["source_path"] == sp] if sp else []
    if len(hit) > 0:
        i = hit[0]
        for k, v in row.items():
            if k not in df.columns:
                continue
            if str(df.loc[i, k]).strip() == "" and str(v).strip() != "":
                df.loc[i, k] = v
        return df

    # insert new row
    base = {c: "" for c in df.columns}
    base.update(DEFAULT_ROW)
    base.update(row)
    return pd.concat([df, pd.DataFrame([base])], ignore_index=True)


def existing_doc_ids(df: pd.DataFrame) -> set[str]:
    ids = set()
    if "doc_id" in df.columns:
        ids.update({str(x).strip() for x in df["doc_id"].tolist() if str(x).strip()})
    if "proposed_doc_id" in df.columns:
        ids.update({str(x).strip() for x in df["proposed_doc_id"].tolist() if str(x).strip()})
    return ids


def promote_file_to_sources(src: Path, doc_id: str) -> Path:
    """
    Copy src to 01_sources/<doc_id>.<ext>. Never overwrite.
    """
    SOURCES_DIR.mkdir(parents=True, exist_ok=True)
    dest = SOURCES_DIR / f"{doc_id}{src.suffix.lower()}"
    if dest.exists():
        # avoid overwrite: pick a unique suffix (should be rare if doc_id unique)
        n = 1
        while True:
            alt = SOURCES_DIR / f"{doc_id}_{n:02d}{src.suffix.lower()}"
            if not alt.exists():
                dest = alt
                break
            n += 1
    dest.write_bytes(src.read_bytes())
    return dest


def main():
    df = load_or_init_documents_csv()
    seen_ids = existing_doc_ids(df)

    inbox_files = list_inbox_files()
    if not inbox_files:
        print("No supported files found in inbox folders.")
        return

    now_utc = datetime.now(timezone.utc).isoformat()

    for f in inbox_files:
        try:
            # Re-check existence & readability right before use (Dropbox can change)
            stat = f.stat()
        except (FileNotFoundError, OSError) as e:
            print(f"[SKIP] Cannot stat file (missing/transient): {f} | {e}")
            continue

        try:
            file_hash = sha256_file(f)
        except (FileNotFoundError, PermissionError, OSError) as e:
            print(f"[SKIP] Cannot read file for hash: {f} | {e}")
            continue

        src_bucket = detect_source_bucket(f)
    
        # If already registered by hash, skip proposing new doc_id (avoid duplicates)
        try:
            if (df["file_hash_sha256"] == file_hash).any():
                row = {
                    "doc_id": "",
                    "proposed_doc_id": proposed_id,
                    "original_filename": f.name,
                    "source_path": str(f),
                    "file_hash_sha256": file_hash,
                    "file_ext": f.suffix.lower().lstrip("."),
                    "file_size_bytes": str(stat.st_size),
                    "file_modified_utc": utc_iso_from_ts(stat.st_mtime),
                    "ingested_at_utc": now_utc,
                    "status": "draft",
                    "source": src_bucket,
                    "doc_type": inferred.get("doc_type", ""),
                    "perils": inferred.get("perils", ""),
                    "jurisdiction": inferred.get("jurisdiction", ""),
                    "version": inferred.get("version", ""),
                    "date": inferred.get("date", ""),
                }
                continue
        except Exception as e:
            print(f"[WARN] Hash compare failed for {f}: {e}")

        proposed_id, inferred = build_doc_id(f.name, seen_ids)

        row = {
            "doc_id": "",
            "proposed_doc_id": proposed_id,
            "original_filename": f.name,
            "source_path": str(f),
            "file_hash_sha256": file_hash,
            "file_ext": f.suffix.lower().lstrip("."),
            "file_size_bytes": str(stat.st_size),
            "file_modified_utc": utc_iso_from_ts(stat.st_mtime),
            "ingested_at_utc": now_utc,
            "status": "draft",
            "source": src_bucket,
            "doc_type": inferred.get("doc_type", ""),
            "perils": inferred.get("perils", ""),
            "jurisdiction": inferred.get("jurisdiction", ""),
            "version": inferred.get("version", ""),
            "date": inferred.get("date", ""),
        }

        df = upsert_row(df, row)
        """
        if PROMOTE_TO_SOURCES:
            try:
                dest = promote_file_to_sources(f, proposed_id)
            except (FileNotFoundError, PermissionError, OSError) as e:
                print(f"[SKIP] Promote failed for {f} -> {proposed_id}: {e}")
                continue"""


    df.to_csv(DOCS_CSV, index=False, encoding="utf-8")
    print(f"Done. Updated: {DOCS_CSV}")
    print("Next: open documents.csv and confirm doc_id values before approval.")


if __name__ == "__main__":
    main()
