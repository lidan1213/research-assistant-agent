from app.llm.channel import classify_channel


def test_openai_official_channel():
    info = classify_channel("https://api.openai.com/v1")
    assert info["channel"] == "openai_official"
    assert info["official_direct"] is True


def test_cc_switch_local_channel():
    info = classify_channel("http://127.0.0.1:15721/v1")
    assert info["channel"] == "cc_switch"
    assert info["official_direct"] is False


def test_unknown_compatible_endpoint_is_other_relay():
    info = classify_channel("https://ruoli.dev/v1")
    assert info["channel"] == "other_relay"
    assert info["endpoint"] == "ruoli.dev"
    assert "最终上游" in info["warning"]


def test_api_key_never_appears_in_channel_info():
    info = classify_channel("https://api.openai.com/v1")
    assert "api_key" not in info
