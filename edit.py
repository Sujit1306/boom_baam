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
    try:
        resp.raise_for_status()
    except requests.HTTPError as he:
        # Print body for clarity on 4xx/5xx
        print(f"[TOKEN ERROR] HTTP {resp.status_code} body: {resp.text[:1000]}", flush=True)
        raise

    try:
        payload = resp.json()
    except Exception:
        print(f"[TOKEN ERROR] Non-JSON response from Okta. Status={resp.status_code}, "
              f"Body starts with: {resp.text[:1000]!r}", flush=True)
        raise

    token = payload.get("access_token")
    expires_in = payload.get("expires_in", 3600)
    if not token:
        raise RuntimeError(f"Okta did not return access_token. Response: {payload}")

    self._token = token
    self._expires_at = time.time() + float(expires_in)
    return token
