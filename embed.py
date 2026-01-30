# pepgenx_embed_runner.py
# -*- coding: utf-8 -*-
"""
Self-contained runner for PepGenX Embedding APIs (per provided Swagger).

Supports:
  - /v1/cs/em/embed-documents   (V1: documents is List[List[str]])
  - /v2/cs/em/embed-documents   (V2: documents is List[List[str]] + embedding_mode)
  - /v1/cs/em/embed-query       (single query string)

Auth:
  - Okta client-credentials (access token in Authorization: Bearer <token>)

Required headers sent on every call:
  - team_id
  - project_id
  - x-pepgenx-apikey
  - token_encoding
  - embedding_model_name

Configuration via environment variables (optionally .env with python-dotenv):
  # Okta
  OKTA_TOKEN_URL=https://<yourOktaDomain>/oauth2/<authz_server>/v1/token
  OKTA_CLIENT_ID=<client_id>
  OKTA_CLIENT_SECRET=<client_secret>
  OKTA_SCOPE=                     # optional

  # Required PepGenX headers
  TEAM_ID=<team_id>
  PROJECT_ID=<project_id>
  X_PEPGENX_APIKEY=<api_key>

  # Embedding headers (per swagger defaults; override as needed)
  TOKEN_ENCODING=cl100k_base
  EMBEDDING_MODEL_NAME=text-embedding-3-small   # or text-embedding-3-large, text-embedding-ada-002

  # Endpoints (QA defaults below; override for Prod if needed)
  PEPGENX_EMBED_V1_URL=https://apim-na.qa.mypepsico.com/cgf/pepgenx/v1/cs/em/embed-documents
  PEPGENX_EMBED_V2_URL=https://apim-na.qa.mypepsico.com/cgf/pepgenx/v2/cs/em/embed-documents
  PEPGENX_EMBED_Q_URL=https://apim-na.qa.mypepsico.com/cgf/pepgenx/v1/cs/em/embed-query

  # Runner mode: V1 | V2 | QUERY
  MODE=V2

  # Input (choose ONE source)
  # 1) Provide chunks as a semicolon/comma separated list:
  #    EMBED_STRINGS=First chunk;Second chunk;Third chunk
  # 2) OR provide a TXT file with one chunk per non-empty line:
  #    DOCS_TXT=C:\path\to\chunks.txt
  # 3) OR provide a JSONL file, each line is JSON; extract with JSONL_KEY (optional):
  #    DOCS_JSONL=C:\path\to\chunks.jsonl
  #    JSONL_KEY=content

  # V2 only
  EMBEDDING_MODE=multiple_lists   # or single_list
  TRUNCATE=NONE                   # NONE | END | START

  # Query mode
  # MODE=QUERY
  # QUERY=What is the SOP for SQL backup failure?
  # TRUNCATE=NONE

  # Output
  OUT_JSON=embeddings.json
  OUT_CSV=embeddings.csv
  WRITE_CSV=true

  # HTTP/TLS
  PEPGENX_VERIFY_TLS=true
  PEPGENX_TIMEOUT_S=60
  PEPGENX_MAX_RETRIES=2
  PEPGENX_RETRY_BACKOFF_S=0.8
"""

from __future__ import annotations

import base64
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

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


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name, "true" if default else "false").strip().lower()
    return raw in ("1", "true", "yes", "y", "on")


def human_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _split_semicolon(raw: str) -> List[str]:
    raw = (raw or "").strip()
    if not raw:
        return []
    parts = [p.strip() for p in raw.replace(",", ";").split(";") if p.strip()]
    return parts


# -----------------------------
# Okta client-credentials
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

        headers = {"Content-Type": "application/x-www-form-urlencoded"}
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
        try:
            resp.raise_for_status()
        except requests.HTTPError:
            print(f"[TOKEN ERROR] HTTP {resp.status_code} body: {resp.text[:1000]}", flush=True)
            raise

        try:
            payload = resp.json()
        except Exception:
            print(f"[TOKEN ERROR] Non-JSON response from Okta. Status={resp.status_code}, "
                  f"Body starts with: {resp.text[:1000]!r}", flush=True)
            raise

        token = payload.get("access_token")
        if not token:
            raise RuntimeError(f"Okta did not return access_token. Response: {payload}")
        expires_in = float(payload.get("expires_in", 3600))
        self._token = token
        self._expires_at = time.time() + expires_in
        return token


# -----------------------------
# PepGenX Embedding client
# -----------------------------
@dataclass
class PepGenXConfig:
    verify_tls: bool = True
    timeout_s: float = 60.0
    max_retries: int = 2
    retry_backoff_s: float = 0.8
    team_id: str = ""
    project_id: str = ""
    x_pepgenx_apikey: str = ""
    token_encoding: str = "cl100k_base"
    embedding_model_name: str = "text-embedding-3-small"


class PepGenXEmbedClient:
    def __init__(self, cfg: PepGenXConfig, token_provider: OktaTokenProvider):
        self.cfg = cfg
        self.token_provider = token_provider
        self._session = requests.Session()

    # /v1/cs/em/embed-documents
    def embed_documents_v1(
        self,
        *,
        url: str,
        documents: List[List[str]],  # list of list of strings
        truncate: str = "NONE",
    ) -> Dict[str, Any]:
        payload = {"documents": documents, "truncate": truncate}
        return self._post_json(url, payload)

    # /v2/cs/em/embed-documents
    def embed_documents_v2(
        self,
        *,
        url: str,
        documents: List[List[str]],
        embedding_mode: str = "multiple_lists",  # 'single_list' | 'multiple_lists'
        truncate: str = "NONE",
    ) -> Dict[str, Any]:
        payload = {"documents": documents, "embedding_mode": embedding_mode, "truncate": truncate}
        return self._post_json(url, payload)

    # /v1/cs/em/embed-query
    def embed_query(
        self,
        *,
        url: str,
        query: str,
        truncate: str = "NONE",
    ) -> Dict[str, Any]:
        payload = {"query": query, "truncate": truncate}
        return self._post_json(url, payload)

    def _post_json(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not url:
            raise ValueError("URL is required")

        last_exc: Optional[Exception] = None
        for attempt in range(self.cfg.max_retries + 1):
            try:
                token = self.token_provider.get_token()
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {token}",
                    "team_id": self.cfg.team_id,
                    "project_id": self.cfg.project_id,
                    "x-pepgenx-apikey": self.cfg.x_pepgenx_apikey,
                    # Embedding headers per swagger
                    "token_encoding": self.cfg.token_encoding,
                    "embedding_model_name": self.cfg.embedding_model_name,
                }

                resp = self._session.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=self.cfg.timeout_s,
                    verify=self.cfg.verify_tls,
                )

                if resp.status_code in (401, 403):
                    # refresh once
                    self.token_provider.force_refresh()
                    headers["Authorization"] = f"Bearer {self.token_provider.get_token()}"
                    resp = self._session.post(
                        url,
                        json=payload,
                        headers=headers,
                        timeout=self.cfg.timeout_s,
                        verify=self.cfg.verify_tls,
                    )

                if resp.status_code != 200:
                    print(f"[EMBED ERROR] HTTP {resp.status_code}. Body starts with:\n{resp.text[:1200]}", flush=True)
                resp.raise_for_status()
                return resp.json()

            except Exception as e:
                last_exc = e
                if attempt < self.cfg.max_retries:
                    time.sleep(self.cfg.retry_backoff_s * (2 ** attempt))
                    continue
                raise
        raise last_exc or RuntimeError("PepGenX embedding request failed")


# -----------------------------
# Helpers (IO)
# -----------------------------
def get_client_from_env() -> PepGenXEmbedClient:
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
        team_id=_require_env("TEAM_ID"),
        project_id=_require_env("PROJECT_ID"),
        x_pepgenx_apikey=_require_env("X_PEPGENX_APIKEY"),
        token_encoding=os.getenv("TOKEN_ENCODING", "cl100k_base"),
        embedding_model_name=os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-small"),
    )
    return PepGenXEmbedClient(cfg, provider)


def _read_lines(path: str) -> List[str]:
    out: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.rstrip("\n")
            if ln:
                out.append(ln)
    return out


def _read_jsonl(path: str, key: Optional[str]) -> List[str]:
    out: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                obj = json.loads(ln)
                if key:
                    val = obj.get(key, "")
                    if isinstance(val, str) and val:
                        out.append(val)
                else:
                    if isinstance(obj, str):
                        out.append(obj)
                    else:
                        out.append(json.dumps(obj, ensure_ascii=False))
            except Exception:
                out.append(ln)
    return out


def write_embeddings_csv(embeddings: List[List[float]], path: str) -> None:
    import csv
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        for vec in embeddings:
            w.writerow(vec)


def write_embeddings_json(embeddings: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(embeddings, f, ensure_ascii=False, indent=2)


# -----------------------------
# Runner (MODE: V1 | V2 | QUERY)
# -----------------------------
def main() -> None:
    mode = os.getenv("MODE", "V2").upper()  # V1 | V2 | QUERY

    client = get_client_from_env()

    # Endpoints
    url_v1 = os.getenv(
        "PEPGENX_EMBED_V1_URL",
        "https://apim-na.qa.mypepsico.com/cgf/pepgenx/v1/cs/em/embed-documents",
    )
    url_v2 = os.getenv(
        "PEPGENX_EMBED_V2_URL",
        "https://apim-na.qa.mypepsico.com/cgf/pepgenx/v2/cs/em/embed-documents",
    )
    url_q = os.getenv(
        "PEPGENX_EMBED_Q_URL",
        "https://apim-na.qa.mypepsico.com/cgf/pepgenx/v1/cs/em/embed-query",
    )

    # Output
    out_json = os.getenv("OUT_JSON", "embeddings.json")
    out_csv = os.getenv("OUT_CSV", "embeddings.csv")
    write_csv = _bool_env("WRITE_CSV", True)

    print(f"[{human_ts()}] MODE={mode}")

    if mode == "QUERY":
        query = _require_env("QUERY")
        truncate = os.getenv("TRUNCATE", "NONE")
        print(f"[{human_ts()}] Calling /v1/cs/em/embed-query ...")
        data = client.embed_query(url=url_q, query=query, truncate=truncate)
        embs = data.get("embeddings") or []
        print(f"[{human_ts()}] Received vector length={len(embs)}")
        write_embeddings_json({"query": query, "embeddings": embs}, out_json)
        print(f"[{human_ts()}] Wrote JSON: {out_json}")
        if write_csv and isinstance(embs, list) and embs and isinstance(embs[0], (float, int)):
            write_embeddings_csv([embs], out_csv)
            print(f"[{human_ts()}] Wrote CSV:  {out_csv}")
        print("Done.")
        return

    # Documents mode
    embed_strings = _split_semicolon(os.getenv("EMBED_STRINGS", ""))
    docs_txt = os.getenv("DOCS_TXT", "").strip()
    docs_jsonl = os.getenv("DOCS_JSONL", "").strip()
    jsonl_key = os.getenv("JSONL_KEY", "").strip() or None

    chunks: List[str] = []
    if embed_strings:
        chunks = embed_strings
    elif docs_txt:
        chunks = _read_lines(docs_txt)
    elif docs_jsonl:
        chunks = _read_jsonl(docs_jsonl, jsonl_key)
    else:
        raise RuntimeError("No inputs provided. Set EMBED_STRINGS or DOCS_TXT or DOCS_JSONL (with optional JSONL_KEY).")

    if not chunks:
        raise RuntimeError("No non-empty chunks found in the selected input source.")

    # By default, treat all chunks as a single 'document' (one sublist).
    documents: List[List[str]] = [chunks]
    truncate = os.getenv("TRUNCATE", "NONE")

    if mode == "V1":
        print(f"[{human_ts()}] Calling /v1/cs/em/embed-documents ...")
        data = client.embed_documents_v1(url=url_v1, documents=documents, truncate=truncate)
        # Shape: {"embeddings": List[List[List[float]]]}
        embeddings = data.get("embeddings") or []
        flat: List[List[float]] = []
        if embeddings and isinstance(embeddings[0], list):
            flat = embeddings[0] if len(embeddings) == 1 else [vec for doc in embeddings for vec in doc]

        total = sum(len(doc) for doc in embeddings) if embeddings else 0
        print(f"[{human_ts()}] Embedded chunks: {total}")
        write_embeddings_json(data, out_json)
        print(f"[{human_ts()}] Wrote JSON: {out_json}")
        if write_csv and flat:
            write_embeddings_csv(flat, out_csv)
            print(f"[{human_ts()}] Wrote CSV:  {out_csv}")

    elif mode == "V2":
        embedding_mode = os.getenv("EMBEDDING_MODE", "multiple_lists")
        print(f"[{human_ts()}] Calling /v2/cs/em/embed-documents (embedding_mode={embedding_mode}) ...")
        data = client.embed_documents_v2(
            url=url_v2,
            documents=documents,
            embedding_mode=embedding_mode,
            truncate=truncate,
        )
        # Shape: {"embeddings": List[List[List[float]]]}
        embeddings = data.get("embeddings") or []
        flat: List[List[float]] = []
        if embeddings and isinstance(embeddings[0], list):
            flat = embeddings[0] if len(embeddings) == 1 else [vec for doc in embeddings for vec in doc]

        total = sum(len(doc) for doc in embeddings) if embeddings else 0
        print(f"[{human_ts()}] Embedded chunks: {total}")
        write_embeddings_json(data, out_json)
        print(f"[{human_ts()}] Wrote JSON: {out_json}")
        if write_csv and flat:
            write_embeddings_csv(flat, out_csv)
            print(f"[{human_ts()}] Wrote CSV:  {out_csv}")

    else:
        raise RuntimeError("Unsupported MODE. Set MODE=V1 | V2 | QUERY")

    print("Done.")


if __name__ == "__main__":
    main()
