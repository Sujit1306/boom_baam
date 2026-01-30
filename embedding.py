# pepgenx_embed_jsonl_runner.py
# -*- coding: utf-8 -*-
"""
PepGenX Embedding runner (JSONL-only input), per the provided Swagger.

Supports:
  - /v1/cs/em/embed-documents   (V1: documents = List[List[str]])
  - /v2/cs/em/embed-documents   (V2: documents = List[List[str]] + embedding_mode)

Input (REQUIRED):
  - JSONL file path via DOCS_JSONL
  - Optional JSONL_KEY to pull a specific field from each JSON line.
    If JSONL_KEY is omitted:
      * if line is a JSON string -> used as-is
      * if line is a JSON object -> entire object is stringified

Auth:
  - Okta client-credentials

Required PepGenX headers on every request:
  - team_id
  - project_id
  - x-pepgenx-apikey
  - token_encoding
  - embedding_model_name

Environment variables (typical):
  OKTA_TOKEN_URL=https://<okta>/oauth2/<authz_server>/v1/token
  OKTA_CLIENT_ID=<client_id>
  OKTA_CLIENT_SECRET=<client_secret>
  OKTA_SCOPE=                         # optional

  TEAM_ID=<team_id>
  PROJECT_ID=<project_id>
  X_PEPGENX_APIKEY=<api_key>

  TOKEN_ENCODING=cl100k_base
  EMBEDDING_MODEL_NAME=text-embedding-3-small   # or text-embedding-3-large, text-embedding-ada-002

  # Mode: V1 | V2 (default V2)
  MODE=V2
  EMBEDDING_MODE=multiple_lists                  # V2 only: single_list | multiple_lists
  TRUNCATE=NONE                                  # NONE | START | END

  # JSONL input (REQUIRED)
  DOCS_JSONL=C:\path\to\chunks.jsonl
  JSONL_KEY=content                               # optional

  # Endpoints (QA defaults below; override for Prod if needed)
  PEPGENX_EMBED_V1_URL=https://apim-na.qa.mypepsico.com/cgf/pepgenx/v1/cs/em/embed-documents
  PEPGENX_EMBED_V2_URL=https://apim-na.qa.mypepsico.com/cgf/pepgenx/v2/cs/em/embed-documents

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
        documents: List[List[str]],
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
        embedding_mode: str = "multiple_lists",
        truncate: str = "NONE",
    ) -> Dict[str, Any]:
        payload = {"documents": documents, "embedding_mode": embedding_mode, "truncate": truncate}
        return self._post_json(url, payload)

    # Shared POST with retries
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
                    # Embedding headers per Swagger
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
# IO helpers (JSONL-only)
# -----------------------------
def _read_jsonl(path: str, key: Optional[str]) -> List[str]:
    """
    Read a JSONL file and produce a list of strings:
      - if key is provided: extract obj[key] when present and is a string
      - otherwise:
           * if line is JSON string, use it
           * if line is JSON object/array, stringify the object
           * if parsing fails, use the raw line
    Empty strings are discarded.
    """
    out: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            ln = raw.strip()
            if not ln:
                continue
            try:
                obj = json.loads(ln)
                if key:
                    val = obj.get(key, "") if isinstance(obj, dict) else ""
                    if isinstance(val, str) and val.strip():
                        out.append(val)
                else:
                    if isinstance(obj, str):
                        if obj.strip():
                            out.append(obj)
                    else:
                        # stringify non-string JSON
                        s = json.dumps(obj, ensure_ascii=False)
                        if s.strip():
                            out.append(s)
            except Exception:
                # treat line as raw text if not valid JSON
                if ln:
                    out.append(ln)
    # deduplicate empties
    out = [s for s in out if isinstance(s, str) and s.strip()]
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
# Client factory & Runner
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


def main() -> None:
    mode = os.getenv("MODE", "V2").upper()  # V1 | V2

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

    # JSONL-only inputs
    jsonl_path = _require_env("DOCS_JSONL")
    jsonl_key = os.getenv("JSONL_KEY", "").strip() or None

    # Output
    out_json = os.getenv("OUT_JSON", "embeddings.json")
    out_csv = os.getenv("OUT_CSV", "embeddings.csv")
    write_csv = _bool_env("WRITE_CSV", True)

    # V2-only parameter + truncate
    embedding_mode = os.getenv("EMBEDDING_MODE", "multiple_lists")
    truncate = os.getenv("TRUNCATE", "NONE")

    print(f"[{human_ts()}] MODE={mode}")
    print(f"[{human_ts()}] Reading JSONL: {jsonl_path} (key={jsonl_key or '<none>'})")

    chunks = _read_jsonl(jsonl_path, jsonl_key)
    if not chunks:
        raise RuntimeError("No non-empty strings produced from DOCS_JSONL. Check JSONL format and JSONL_KEY.")

    # Put all chunks into a single 'document' (one sublist).
    documents: List[List[str]] = [chunks]
    print(f"[{human_ts()}] Prepared {len(chunks)} chunk(s) from JSONL.")

    if mode == "V1":
        print(f"[{human_ts()}] Calling /v1/cs/em/embed-documents ...")
        data = client.embed_documents_v1(url=url_v1, documents=documents, truncate=truncate)
    elif mode == "V2":
        print(f"[{human_ts()}] Calling /v2/cs/em/embed-documents (embedding_mode={embedding_mode}) ...")
        data = client.embed_documents_v2(
            url=url_v2, documents=documents, embedding_mode=embedding_mode, truncate=truncate
        )
    else:
        raise RuntimeError("Unsupported MODE. Set MODE=V1 or MODE=V2.")

    # Expect: {"embeddings": List[List[List[float]]]}
    embeddings = data.get("embeddings") or []
    total_vecs = sum(len(doc) for doc in embeddings) if embeddings else 0
    print(f"[{human_ts()}] Embedded chunks: {total_vecs}")

    # Write the entire API response to JSON (preserve structure)
    write_embeddings_json(data, out_json)
    print(f"[{human_ts()}] Wrote JSON: {out_json}")

    # Optionally flatten to CSV (one row per vector)
    if write_csv and embeddings and isinstance(embeddings[0], list):
        flat: List[List[float]] = embeddings[0] if len(embeddings) == 1 else [vec for doc in embeddings for vec in doc]
        if flat:
            write_embeddings_csv(flat, out_csv)
            print(f"[{human_ts()}] Wrote CSV:  {out_csv}")

    print("Done.")


if __name__ == "__main__":
    main()
