"""Safe LLM transport-channel classification (never exposes API keys)."""
from __future__ import annotations

import os
from urllib.parse import urlparse


def classify_channel(base_url: str) -> dict:
    raw = (base_url or "").strip()
    parsed = urlparse(raw if "://" in raw else f"https://{raw}")
    host = (parsed.hostname or "").lower()
    port = parsed.port
    official = host in {"api.openai.com"}
    configured_cc = {
        x.strip().lower().rstrip("/")
        for x in os.environ.get("CC_SWITCH_BASE_URLS", "").split(",") if x.strip()
    }
    normalized = raw.lower().rstrip("/")
    local_cc = host in {"127.0.0.1", "localhost", "::1"} and port in {15721}
    named_cc = "ccswitch" in host or "cc-switch" in host or normalized in configured_cc
    if official:
        channel, label = "openai_official", "OpenAI 官方直连"
    elif local_cc or named_cc:
        channel, label = "cc_switch", "CC Switch 中转"
    else:
        channel, label = "other_relay", "其他 OpenAI 兼容中转"
    endpoint = host + (f":{port}" if port else "")
    return {
        "channel": channel,
        "label": label,
        "endpoint": endpoint,
        "official_direct": official,
        "basis": "configured_base_url",
        "warning": "只能确认本项目的直接请求目的地，无法证明中转站的最终上游。" if not official else "",
    }


def configured_channel(*, use_aux: bool = False) -> dict:
    from app.config import get_settings

    settings = get_settings().llm
    base_url = settings.aux_base_url if use_aux and settings.aux_base_url else settings.base_url
    model = settings.aux_model if use_aux and settings.aux_model else settings.model
    info = classify_channel(base_url)
    return {**info, "model": model, "tier": "aux" if use_aux and settings.aux_model else "main"}
