from __future__ import annotations

import base64
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, List
from datetime import datetime, timezone
import json

import requests

try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=False)
except Exception:
    pass


# -----------------------------
# Utilities
# -----------------------------

def _require_env(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise RuntimeError(f"Missing required env var: {name}")
    return val


def human_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# -----------------------------
# Okta token provider (Client Credentials)
# -----------------------------
@dataclass
class OktaConfig:
    token_url: str
    client_id: str
    client_secret: str
    scope: str = ""
    use_basic_auth: bool = True
    refresh_skew_s: int = 30
    timeout_s: float = 30.0


class OktaTokenProvider:
    """Fetch and cache an OAuth2 access token from Okta (client_credentials)."""

    def __init__(self, cfg: OktaConfig):
        self.cfg = cfg
        self._token: Optional[str] = None
        self._expires_at: float = 0.0
        self._session = requests.Session()

    def get_token(self) -> str:
        now = time.time()
        if self._token and now < (self._expires_at - self.cfg.refresh_skew_s):
            return self._token
        return self.force_refresh()

    def force_refresh(self) -> str:
        data = {"grant_type": "client_credentials"}
        if self.cfg.scope:
            data["scope"] = self.cfg.scope

        headers: Dict[str, str] = {"Content-Type": "application/x-www-form-urlencoded"}

        if self.cfg.use_basic_auth:
            basic = base64.b64encode(
                f"{self.cfg.client_id}:{self.cfg.client_secret}".encode("utf-8")
            ).decode("utf-8")
            headers["Authorization"] = f"Basic {basic}"
        else:
            data["client_id"] = self.cfg.client_id
            data["client_secret"] = self.cfg.client_secret

        resp = self._session.post(
            self.cfg.token_url,
            data=data,
            headers=headers,
            timeout=self.cfg.timeout_s,
        )
        resp.raise_for_status()
        payload = resp.json()

        token = payload.get("access_token")
        expires_in = payload.get("expires_in", 3600)
        if not token:
            raise RuntimeError(f"Okta did not return access_token. Response: {payload}")

        self._token = token
        self._expires_at = time.time() + float(expires_in)
        return token


# -----------------------------
# PepGenX client
# -----------------------------
@dataclass
class PepGenXConfig:
    # Only chunking-related settings are used in this script
    verify_tls: bool = True
    timeout_s: float = 60.0
    max_retries: int = 2
    retry_backoff_s: float = 0.8
    team_id: str = "f7f79747-49ea-4956-a25d-1586c2c0d42c"
    project_id: str = "3dc415b0-bcb4-4137-9a5e-223395be7f2f"
    x_pepgenx_apikey: str = "89a3a808-f382-45a5-b23a-9f296c824bfc"


class PepGenXClient:
    """PepGenX HTTP client with retries (for chunking)."""

    def __init__(self, cfg: PepGenXConfig, token_provider: OktaTokenProvider):
        self.cfg = cfg
        self.token_provider = token_provider
        self._session = requests.Session()

    def chunk_json_file(
        self,
        *,
        chunk_url: str,
        server_file_path: str,
        file_name: str,
        jq_schema: Optional[str] = None,
        chunking_strategy: str = "recursivejsonsplitter",
        chunk_size: int = 800,
        chunk_overlap: int = 20,
        min_chunk_length: int = 500,
        max_chunk_length: int = 800,
        num_rows_per_chunk: int = 2,
    ) -> Dict[str, Any]:
        """
        Chunk a JSON document using PepGenX /v1/cs/ch/load-and-chunk (NO Azure Blob).

        The API will read the file at: <server_file_path>/<file_name>
        - server_file_path: directory path visible to the PepGenX backend (no trailing slash required)
        - file_name: just the file name (no directories)

        Returns the full response dict (chunks in response["documents"]).
        """
        if not chunk_url:
            raise ValueError("chunk_url is required")
        if not server_file_path or not file_name:
            raise ValueError("server_file_path and file_name are required")

        payload: Dict[str, Any] = {
            "files": [file_name],
            "file_path": server_file_path,
            "chunking_strategy": chunking_strategy,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "min_chunk_length": min_chunk_length,
            "max_chunk_length": max_chunk_length,
            "num_rows_per_chunk": num_rows_per_chunk,
        }
        if jq_schema:
            payload["jq_schema"] = jq_schema

        last_exc: Optional[Exception] = None
        for attempt in range(self.cfg.max_retries + 1):
            try:
                token = self.token_provider.get_token()
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {token}",
                }
                if self.cfg.team_id:
                    headers["team_id"] = self.cfg.team_id
                if self.cfg.project_id:
                    headers["project_id"] = self.cfg.project_id
                if self.cfg.x_pepgenx_apikey:
                    headers["x-pepgenx-apikey"] = self.cfg.x_pepgenx_apikey

                resp = self._session.post(
                    chunk_url,
                    json=payload,
                    headers=headers,
                    timeout=self.cfg.timeout_s,
                    verify=self.cfg.verify_tls,
                )

                if resp.status_code in (401, 403):
                    # Refresh token once
                    self.token_provider.force_refresh()
                    token2 = self.token_provider.get_token()
                    headers["Authorization"] = f"Bearer {token2}"
                    resp = self._session.post(
                        chunk_url,
                        json=payload,
                        headers=headers,
                        timeout=self.cfg.timeout_s,
                        verify=self.cfg.verify_tls,
                    )

                resp.raise_for_status()
                return resp.json()

            except Exception as e:
                last_exc = e
                if attempt < self.cfg.max_retries:
                    time.sleep(self.cfg.retry_backoff_s * (2 ** attempt))
                    continue
                raise

        raise last_exc or RuntimeError("PepGenX chunking request failed")


# -----------------------------
# Helpers
# -----------------------------

def extract_chunks(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    docs = data.get("documents") or []
    if not isinstance(docs, list):
        raise RuntimeError(f"Unexpected response shape: {str(data)[:500]}")
    return docs


def write_jsonl(docs: List[Dict[str, Any]], out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")


def write_csv(docs: List[Dict[str, Any]], out_path: str) -> None:
    import csv
    fieldnames = ["chunk_id", "doc_id", "title", "page_num", "content"]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
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


# -----------------------------
# Factory & Runner
# -----------------------------

def get_pepgenx_client() -> PepGenXClient:
    okta_cfg = OktaConfig(
        token_url=_require_env("OKTA_TOKEN_URL"),
        client_id=_require_env("OKTA_CLIENT_ID"),
        client_secret=_require_env("OKTA_CLIENT_SECRET"),
        scope=os.getenv("OKTA_SCOPE", ""),
        use_basic_auth=True,
    )
    provider = OktaTokenProvider(okta_cfg)

    cfg = PepGenXConfig(
        verify_tls=os.getenv("PEPGENX_VERIFY_TLS", "true").lower() == "true",
        timeout_s=float(os.getenv("PEPGENX_TIMEOUT_S", "60")),
        max_retries=int(os.getenv("PEPGENX_MAX_RETRIES", "2")),
        retry_backoff_s=float(os.getenv("PEPGENX_RETRY_BACKOFF_S", "0.8")),
        team_id=os.getenv("TEAM_ID", ""),
        project_id=os.getenv("PROJECT_ID", ""),
        x_pepgenx_apikey=os.getenv("X_PEPGENX_APIKEY", ""),
    )
    return PepGenXClient(cfg, provider)


def _load_jq_schema(value: str) -> Optional[str]:
    value = (value or "").strip()
    if not value:
        return None
    # If points to a file path, load contents; otherwise treat as inline jq
    try:
        if os.path.isfile(value):
            with open(value, "r", encoding="utf-8") as f:
                return f.read()
    except Exception:
        pass
    return value


def main() -> None:
    chunk_url = os.getenv(
        "PEPGENX_CHUNK_URL",
        "https://apim-na.qa.mypepsico.com/cgf/pepgenx/v1/cs/ch/load-and-chunk",
    )
    server_file_path = _require_env("SERVER_FILE_PATH")  # directory only
    file_name = _require_env("FILE_NAME")                # file name only

    # Chunking params
    chunking_strategy = os.getenv("CHUNKING_STRATEGY", "recursivejsonsplitter")
    chunk_size = int(os.getenv("CHUNK_SIZE", "800"))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "20"))
    min_chunk_length = int(os.getenv("MIN_CHUNK_LENGTH", "500"))
    max_chunk_length = int(os.getenv("MAX_CHUNK_LENGTH", "800"))
    num_rows_per_chunk = int(os.getenv("NUM_ROWS_PER_CHUNK", "2"))
    jq_schema = _load_jq_schema(os.getenv("JQ_SCHEMA", ""))

    client = get_pepgenx_client()

    print(f"[{human_ts()}] Calling PepGenX /v1/cs/ch/load-and-chunk ...")
    data = client.chunk_json_file(
        chunk_url=chunk_url,
        server_file_path=server_file_path,
        file_name=file_name,
        jq_schema=jq_schema,
        chunking_strategy=chunking_strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        min_chunk_length=min_chunk_length,
        max_chunk_length=max_chunk_length,
        num_rows_per_chunk=num_rows_per_chunk,
    )

    docs = extract_chunks(data)
    print(f"[{human_ts()}] Received {len(docs)} chunks.")

    # Outputs
    out_jsonl = os.getenv("OUT_JSONL", "").strip()
    if not out_jsonl:
        stem = os.path.splitext(os.path.basename(file_name))[0]
        out_jsonl = os.path.abspath(f"{stem}_chunks.jsonl")

    write_jsonl(docs, out_jsonl)
    print(f"[{human_ts()}] Wrote JSONL: {out_jsonl}")

    write_csv_flag = os.getenv("WRITE_CSV", "true").lower() == "true"
    if write_csv_flag:
        out_csv = os.getenv("OUT_CSV", "").strip() or (os.path.splitext(out_jsonl)[0] + ".csv")
        write_csv(docs, out_csv)
        print(f"[{human_ts()}] Wrote CSV:   {out_csv}")

    print("Done.")


if __name__ == "__main__":
    main()
