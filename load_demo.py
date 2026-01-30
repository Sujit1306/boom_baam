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
    ) -> Dict[str, Any]:

        payload: Dict[str, Any] = {
            "files": files,
            "file_path": server_file_path,
            "jq_schema": jq_schema,
        }

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

    team_id = "f7f79747-49ea-4956-a25d-1586c2c0d42c"
    project_id = "3dc415b0-bcb4-4137-9a5e-223395be7f2f"
    x_pepgenx_apikey = "89a3a808-f382-45a5-b23a-9f296c824bfc"

    cfg = PepGenXConfig(
        verify_tls=True,
        timeout_s=60.0,
        max_retries=2,
        retry_backoff_s=0.8,
        team_id=team_id,
        project_id=project_id,
        x_pepgenx_apikey=x_pepgenx_apikey,
    )
    return PepGenXClient(cfg, provider)


# -----------------------------
# Runner
# -----------------------------
def main() -> None:
    load_url = "https://apim-na.qa.mypepsico.com/cgf/pepgenx/v1/cs/ch/load-files"

    # Path and files:
    server_file_path = ""
    files = ["1_SQL.json",]  
    jq_schema ="."

    client = get_pepgenx_client()

    print(f"[{human_ts()}] Calling PepGenX /v1/cs/ch/load-files ...")
    data = client.load_files(
        load_url=load_url,
        server_file_path=server_file_path,
        files=files,
        jq_schema=jq_schema,
    )
    print(data)
    print("Done.")


if __name__ == "__main__":
    main()