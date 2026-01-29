from __future__ import annotations

import base64
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

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
            self.cfg.token_url, data=data, headers=headers, timeout=self.cfg.timeout_s
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
# PepGenX client (load-only)
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


class PepGenXClient:
    def __init__(self, cfg: PepGenXConfig, token_provider: OktaTokenProvider):
        self.cfg = cfg
        self.token_provider = token_provider
        self._session = requests.Session()

    def load_files(
        self,
        *,
        load_url: str,
        server_file_path: str,
        files: List[str],
        jq_schema: Optional[str] = None,
        table_output_format: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not load_url:
            raise ValueError("load_url is required")
        if not server_file_path or not files:
            raise ValueError("server_file_path and files are required")

        payload: Dict[str, Any] = {
            "files": files,
            "file_path": server_file_path,
        }
        if jq_schema:
            payload["jq_schema"] = jq_schema
        if table_output_format:
            payload["table_output_format"] = table_output_format

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
                }

                resp = self._session.post(
                    load_url,
                    json=payload,
                    headers=headers,
                    timeout=self.cfg.timeout_s,
                    verify=self.cfg.verify_tls,
                )

                if resp.status_code in (401, 403):
                    # refresh token once
                    self.token_provider.force_refresh()
                    headers["Authorization"] = f"Bearer {self.token_provider.get_token()}"
                    resp = self._session.post(
                        load_url,
                        json=payload,
                        headers=headers,
                        timeout=self.cfg.timeout_s,
                        verify=self.cfg.verify_tls,
                    )

                if resp.status_code != 200:
                    print(f"[LOAD ERROR] HTTP {resp.status_code}. Body starts with:\n{resp.text[:1200]}", flush=True)
                resp.raise_for_status()
                return resp.json()

            except Exception as e:
                last_exc = e
                if attempt < self.cfg.max_retries:
                    time.sleep(self.cfg.retry_backoff_s * (2 ** attempt))
                    continue
                raise
        raise last_exc or RuntimeError("PepGenX load-files request failed")


# -----------------------------
# Helpers
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

    team_id = _require_env("TEAM_ID")
    project_id = _require_env("PROJECT_ID")
    x_pepgenx_apikey = _require_env("X_PEPGENX_APIKEY")

    cfg = PepGenXConfig(
        verify_tls=os.getenv("PEPGENX_VERIFY_TLS", "true").lower() == "true",
        timeout_s=float(os.getenv("PEPGENX_TIMEOUT_S", "60")),
        max_retries=int(os.getenv("PEPGENX_MAX_RETRIES", "2")),
        retry_backoff_s=float(os.getenv("PEPGENX_RETRY_BACKOFF_S", "0.8")),
        team_id=team_id,
        project_id=project_id,
        x_pepgenx_apikey=x_pepgenx_apikey,
    )
    return PepGenXClient(cfg, provider)


def _split_filenames(raw: str) -> List[str]:
    raw = (raw or "").strip()
    if not raw:
        return []
    # allow semicolon, comma, or whitespace separated lists
    parts = [p.strip() for p in raw.replace(",", ";").split(";") if p.strip()]
    return parts


def write_jsonl_from_loader(processed: List[Dict[str, Any]], out_jsonl: str) -> int:
    """Write each page_content as one JSON line."""
    count = 0
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for pf in processed:
            file_data = pf.get("file_data") or []
            for item in file_data:
                pc = item.get("page_content", "")
                if isinstance(pc, str):
                    f.write(json.dumps({"page_content": pc}, ensure_ascii=False) + "\n")
                    count += 1
    return count


def write_txt_from_loader(processed: List[Dict[str, Any]], out_txt: str) -> int:
    """Write all page_content concatenated as plain text."""
    count = 0
    with open(out_txt, "w", encoding="utf-8") as f:
        for pf in processed:
            file_data = pf.get("file_data") or []
            for item in file_data:
                pc = item.get("page_content", "")
                if isinstance(pc, str):
                    f.write(pc)
                    f.write("\n\n---\n\n")
                    count += 1
    return count


# -----------------------------
# Runner
# -----------------------------
def main() -> None:
    load_url = os.getenv(
        "PEPGENX_LOAD_URL",
        "https://apim-na.qa.mypepsico.com/cgf/pepgenx/v1/cs/ch/load-files",
    )

    # Path and files the SERVICE can read:
    server_file_path = _require_env("SERVER_FILE_PATH")  # directory only
    files = _split_filenames(_require_env("FILES"))      # filenames only

    # JQ schema is optional here; for JSON you typically need one:
    #   - "tostring" -> one big text block (string)
    #   - "."       -> pass JSON structure (useful for json splitters later)
    #   - ".[]" / ".data" / ".items[]" -> if your JSON is array/wrapped
    jq_schema = os.getenv("JQ_SCHEMA", "").strip() or None

    client = get_pepgenx_client()

    print(f"[{human_ts()}] Calling PepGenX /v1/cs/ch/load-files ...")
    data = client.load_files(
        load_url=load_url,
        server_file_path=server_file_path,
        files=files,
        jq_schema=jq_schema,
    )

    processed = data.get("processed_files") or []
    print(f"[{human_ts()}] processed_files: {len(processed)}")

    total_items = 0
    total_chars = 0
    first_preview = ""

    for pf in processed:
        file_name = pf.get("file_name", "<unknown>")
        file_data = pf.get("file_data") or []
        print(f"  - {file_name}: {len(file_data)} item(s)")
        total_items += len(file_data)

        for idx, item in enumerate(file_data):
            pc = item.get("page_content", "")
            if isinstance(pc, str):
                total_chars += len(pc)
                if not first_preview and pc:
                    first_preview = pc[:400].replace("\r", " ").replace("\n", " ")
            else:
                # ensure we only count string content
                pass

    print(f"[{human_ts()}] total loader items: {total_items}, total chars: {total_chars}")
    if first_preview:
        print(f"[{human_ts()}] preview: {first_preview[:400]}...")
    else:
        print("[INFO] No page_content returned by loader. This usually means:")
        print(" - The service cannot access your file path (use a UNC share if backend is remote), or")
        print(" - Your jq_schema did not select any text / structure, or")
        print(" - The file list/path has a mismatch (files must be filenames only; file_path is the directory).")

    # Optional: write JSONL/TXT
    if os.getenv("WRITE_JSONL", "true").lower() == "true":
        out_jsonl = os.getenv("OUT_JSONL", "loader_preview.jsonl")
        n = write_jsonl_from_loader(processed, out_jsonl)
        print(f"[{human_ts()}] Wrote JSONL ({n} lines): {out_jsonl}")

    if os.getenv("WRITE_TXT", "false").lower() == "true":
        out_txt = os.getenv("OUT_TXT", "loader_preview.txt")
        n = write_txt_from_loader(processed, out_txt)
        print(f"[{human_ts()}] Wrote TXT ({n} blocks): {out_txt}")

    print("Done.")


if __name__ == "__main__":
    main()
