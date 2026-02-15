"""
promote_to_sources.py

Purpose
-------
Safely promote documents from inbox folders into the immutable
01_sources directory, using the confirmed doc_id from documents.csv.

This script:
- Only promotes rows with:
    - doc_id NOT empty
    - status in {"reviewed", "approved"}
- Never overwrites existing files
- Verifies file existence via source_path or hash
- Writes a promotion log for auditability

Run this AFTER:
- doc_id_generator.py
- human review of proposed_doc_id
"""

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import pandas as pd
import shutil

# =====================
# CONFIG
# =====================
EVIDENCE_ROOT = Path(__file__).resolve.parent().parent() / "evidence_library"

INBOX_DIRS = [
    EVIDENCE_ROOT / "00_inbox"
]

SOURCES_DIR = EVIDENCE_ROOT / "01_sources"
META_DIR = EVIDENCE_ROOT / "05_metadata"
DOCS_CSV = META_DIR / "documents.csv"
PROMOTION_LOG = META_DIR / "promotions_log.csv"

ALLOWED_STATUSES = {"reviewed", "approved"}
SUPPORTED_EXTS = {".pdf", ".docx", ".pptx"}

# =====================
# HELPERS
# =====================
def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def resolve_path_to_file(path: Path) -> Path | None:
    """
    Resolve a path to a single supported document file.

    Rules:
    - If path is a file → return it
    - If path is a directory:
        - find supported files inside (recursive)
        - if exactly one → return it
        - else → return None
    """
    if not path.exists():
        return None

    if path.is_file():
        if path.suffix.lower() in SUPPORTED_EXTS:
            return path
        return None

    if path.is_dir():
        candidates = []
        for p in path.rglob("*"):
            try:
                if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
                    candidates.append(p)
            except OSError:
                continue

        if len(candidates) == 1:
            return candidates[0]

        # ambiguous or empty folder → force human decision
        return None

    return None

def find_source_file(row) -> Path | None:
    sp = str(row.get("source_path", "")).strip()
    if sp:
        p = resolve_path_to_file(Path(sp))
        if p:
            return p

    target_hash = str(row.get("file_hash_sha256", "")).strip()
    if not target_hash:
        return None

    for d in INBOX_DIRS:
        if not d.exists():
            continue
        for f in d.rglob("*"):
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS:
                try:
                    if sha256_file(f) == target_hash:
                        return f
                except Exception:
                    continue
    return None



def ensure_unique_dest(doc_id: str, ext: str) -> Path:
    """
    Ensure we never overwrite a file in 01_sources.
    """
    base = SOURCES_DIR / f"{doc_id}{ext}"
    if not base.exists():
        return base

    i = 1
    while True:
        alt = SOURCES_DIR / f"{doc_id}_{i:02d}{ext}"
        if not alt.exists():
            return alt
        i += 1


# =====================
# MAIN
# =====================
def main():
    if not DOCS_CSV.exists():
        raise FileNotFoundError(f"Missing documents.csv at {DOCS_CSV}")

    SOURCES_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DOCS_CSV)

    # Prepare promotion log
    if PROMOTION_LOG.exists():
        log_df = pd.read_csv(PROMOTION_LOG)
    else:
        log_df = pd.DataFrame(columns=[
            "doc_id",
            "original_filename",
            "promoted_path",
            "status",
            "promoted_at_utc",
            "note"
        ])

    promoted_count = 0

    for _, row in df.iterrows():
        doc_id = str(row.get("doc_id", "")).strip()
        print(doc_id)
        status = str(row.get("status", "")).strip().lower()
        print(status)

        if not doc_id or status not in ALLOWED_STATUSES:
            continue

        # Skip if already promoted
        existing = list(SOURCES_DIR.glob(f"{doc_id}.*"))
        if existing:
            continue

        src = find_source_file(row)
        if not src:
            log_df.loc[len(log_df)] = {
                "doc_id": doc_id,
                "original_filename": row.get("original_filename", ""),
                "promoted_path": "",
                "status": "failed",
                "promoted_at_utc": utc_now(),
                "note": "source file not found"
            }
            continue

        dest = ensure_unique_dest(doc_id, src.suffix.lower())

        try:
            shutil.copy2(src, dest)
            promoted_count += 1
            print(f"[PROMOTED] {src.name} → {dest}")

            log_df.loc[len(log_df)] = {
                "doc_id": doc_id,
                "original_filename": src.name,
                "promoted_path": str(dest),
                "status": "success",
                "promoted_at_utc": utc_now(),
                "note": ""
            }

        except Exception as e:
            log_df.loc[len(log_df)] = {
                "doc_id": doc_id,
                "original_filename": src.name,
                "promoted_path": "",
                "status": "failed",
                "promoted_at_utc": utc_now(),
                "note": str(e)
            }

    log_df.to_csv(PROMOTION_LOG, index=False, encoding="utf-8")

    print(f"Promotion complete. Files promoted: {promoted_count}")
    print(f"Log written to: {PROMOTION_LOG}")


if __name__ == "__main__":
    main()
