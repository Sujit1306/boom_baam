# pepgenx_chunk_json_no_args.py
# -*- coding: utf-8 -*-
"""
Chunk a JSON file using PepGenX Chunking API WITHOUT Azure Blob or CLI args.

How it works:
- You populate the CONFIG section below (strings left blank) with your values.
- The script calls POST /v1/cs/ch/load-and-chunk
- It expects the PepGenX service to be able to read your file from:
    <SERVER_FILE_PATH>/<FILE_NAME>
- Outputs:
    <input_stem>_chunks.jsonl  (always)
    <input_stem>_chunks.csv     (if ENABLE_CSV_OUTPUT = True)

Dependencies:
    pip install requests
"""

import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests

# =========================
# ======== CONFIG =========
# =========================
# --- API endpoints ---
API_BASE = "https://apim-na.qa.mypepsico.com/cgf/pepgenx"  # e.g., "https://apim-na.qa.mypepsico.com/cgf/pepgenx" or "https://apim-na.mypepsico.com/cgf/pepgenx"

# --- Required headers ---
X_PEPGENX_APIKEY = "89a3a808-f382-45a5-b23a-9f296c824bfc"   # e.g., "YOUR_API_KEY"
TEAM_ID = "f7f79747-49ea-4956-a25d-1586c2c0d42c"            # e.g., "my-team"
PROJECT_ID = "3dc415b0-bcb4-4137-9a5e-223395be7f2f"         # e.g., "my-project"

# --- File location (server-visible path) ---
# The PepGenX service must be able to read: SERVER_FILE_PATH + "/" + FILE_NAME
SERVER_FILE_PATH = r"C:\Users\sujitnanasaheb.khare\OneDrive - HCL TECHNOLOGIES LIMITED\Desktop\DB_Usecase\JSON_FOR_VDB"   # e.g., "/mnt/shared/json_docs"
FILE_NAME = "SQL_1.json"          # e.g., "sample.json"    (do NOT include leading slashes)

# --- Output options ---
# Where to write outputs on YOUR local machine (this script runs here)
# If OUT_JSONL is empty, the script writes "<stem>_chunks.jsonl" in the current directory.
OUT_JSONL = r"C:\Users\sujitnanasaheb.khare\OneDrive - HCL TECHNOLOGIES LIMITED\Desktop\DB_Usecase\JSONL_FOR_VDB"          # e.g., "/home/me/output/sample_chunks.jsonl" or leave blank
ENABLE_CSV_OUTPUT = True
# If CSV path is empty, the script writes "<stem>_chunks.csv" next to the JSONL.
OUT_CSV = r"C:\Users\sujitnanasaheb.khare\OneDrive - HCL TECHNOLOGIES LIMITED\Desktop\DB_Usecase\CSV_FOR_VDB"            # e.g., "/home/me/output/sample_chunks.csv" or leave blank

# --- Chunking params (JSON-optimized defaults from your API spec) ---
CHUNKING_STRATEGY = "recursivejsonsplitter"  # recommended for JSON
CHUNK_SIZE = 800
CHUNK_OVERLAP = 20
MIN_CHUNK_LENGTH = 500
MAX_CHUNK_LENGTH = 800
NUM_ROWS_PER_CHUNK = 2

# Optional: JQ schema (either inline JQ string or a path to a .jq file).
# Leave empty if you don't want to pre-transform the JSON.
JQ_SCHEMA = ""  # e.g., '.items[] | {id, title, body}' OR '/path/to/schema.jq'

# --- HTTP ---
REQUEST_TIMEOUT_SECS = 300


# =========================
# ====== IMPLEMENT =========
# =========================
def human_ts() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def build_headers() -> Dict[str, str]:
    if not API_BASE or not X_PEPGENX_APIKEY or not TEAM_ID or not PROJECT_ID:
        raise RuntimeError("Please fill API_BASE, X_PEPGENX_APIKEY, TEAM_ID, and PROJECT_ID in the CONFIG section.")

    headers = {
        "Content-Type": "application/json",
        "x-pepgenx-apikey": X_PEPGENX_APIKEY,
        "team_id": TEAM_ID,
        "project_id": PROJECT_ID,
    }
    return headers


def load_jq_schema_value(jq_schema: str) -> Optional[str]:
    if not jq_schema:
        return None
    # If provided value is a readable file, load content; otherwise return as-is
    try:
        if os.path.isfile(jq_schema):
            with open(jq_schema, "r", encoding="utf-8") as f:
                return f.read()
    except Exception:
        pass
    return jq_schema


def call_load_and_chunk(
    api_base: str,
    headers: Dict[str, str],
    files_list: List[str],
    file_path: str,
    chunking_strategy: str,
    jq_schema: Optional[str],
    chunk_size: int,
    chunk_overlap: int,
    min_chunk_length: int,
    max_chunk_length: int,
    num_rows_per_chunk: int,
    timeout: int,
) -> Dict[str, Any]:
    url = api_base.rstrip("/") + "/v1/cs/ch/load-and-chunk"

    payload: Dict[str, Any] = {
        "files": files_list,
        "file_path": file_path or "",
        "chunking_strategy": chunking_strategy,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "min_chunk_length": min_chunk_length,
        "max_chunk_length": max_chunk_length,
        "num_rows_per_chunk": num_rows_per_chunk,
    }

    if jq_schema:
        payload["jq_schema"] = jq_schema

    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    if resp.status_code != 200:
        try:
            err = resp.json()
        except Exception:
            err = {"error": resp.text}
        raise RuntimeError(f"PepGenX API error {resp.status_code}: {json.dumps(err, ensure_ascii=False)}")

    try:
        return resp.json()
    except Exception as e:
        raise RuntimeError(f"Failed to parse JSON response: {e}\nRaw: {resp.text[:1000]}")


def write_jsonl(docs: List[Dict[str, Any]], out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")


def write_csv(docs: List[Dict[str, Any]], out_path: str) -> None:
    import csv
    fieldnames = ["chunk_id", "doc_id", "title", "page_num", "content"]
    with open(out_path, "w", encoding="utf-8", newline="}") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for d in docs:
            writer.writerow({
                "chunk_id": d.get("chunk_id", ""),
                "doc_id": d.get("doc_id", ""),
                "title": d.get("title", ""),
                "page_num": d.get("page_num", ""),
                "content": d.get("content", ""),
            })


def main():
    # Basic validations for configuration
    if not FILE_NAME or not SERVER_FILE_PATH:
        print("[ERROR] Please set FILE_NAME and SERVER_FILE_PATH in the CONFIG section.", file=sys.stderr)
        sys.exit(2)

    # For your convenience, optionally warn if the file doesn't exist locally.
    # (The API reads from SERVER_FILE_PATH on the service side.)
    local_hint_path = os.path.join(os.getcwd(), FILE_NAME)
    if not os.path.exists(local_hint_path):
        sys.stderr.write(
            f"[WARN] Local hint: '{local_hint_path}' not found.\n"
            f"       The service will read: '{os.path.join(SERVER_FILE_PATH, FILE_NAME)}'\n"
            "       Make sure the PepGenX service has access to that path and file.\n"
        )

    headers = build_headers()

    files_list = [FILE_NAME]
    file_path_for_api = SERVER_FILE_PATH
    jq_schema_value = load_jq_schema_value(JQ_SCHEMA)

    print(f"[{human_ts()}] Calling PepGenX /v1/cs/ch/load-and-chunk ...")
    try:
        response = call_load_and_chunk(
            api_base=API_BASE,
            headers=headers,
            files_list=files_list,
            file_path=file_path_for_api,
            chunking_strategy=CHUNKING_STRATEGY,
            jq_schema=jq_schema_value,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            min_chunk_length=MIN_CHUNK_LENGTH,
            max_chunk_length=MAX_CHUNK_LENGTH,
            num_rows_per_chunk=NUM_ROWS_PER_CHUNK,
            timeout=REQUEST_TIMEOUT_SECS,
        )
    except Exception as e:
        print(f"[ERROR] API call failed: {e}", file=sys.stderr)
        sys.exit(1)

    docs = response.get("documents") or []
    if not isinstance(docs, list):
        print(f"[ERROR] Unexpected API response shape: {json.dumps(response)[:1000]}", file=sys.stderr)
        sys.exit(1)

    total_chunks = len(docs)
    print(f"[{human_ts()}] Received {total_chunks} chunks.")

    # Decide jsonl output path
    if OUT_JSONL:
        out_jsonl = OUT_JSONL
    else:
        stem = os.path.splitext(os.path.basename(FILE_NAME))[0]
        out_jsonl = os.path.abspath(f"{stem}_chunks.jsonl")

    write_jsonl(docs, out_jsonl)
    print(f"[{human_ts()}] Wrote JSONL: {out_jsonl}")

    # Optional CSV
    if ENABLE_CSV_OUTPUT:
        out_csv = OUT_CSV or (os.path.splitext(out_jsonl)[0] + ".csv")
        write_csv(docs, out_csv)
        print(f"[{human_ts()}] Wrote CSV:   {out_csv}")

    print("Done.")


if __name__ == "__main__":
    main()