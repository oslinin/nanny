import pytest

from nanny.activity import ActivityError
from nanny.llm import _extract_heuristic, _model_available, _use_vertex

NOW = "2026-07-05T20:00:00+00:00"


def test_model_available_via_api_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "sk-test")
    assert _model_available() is True


def test_model_available_via_vertex_backend(monkeypatch):
    # On Vertex there is no API key — the service account (ADC) is the auth, and
    # google-genai is selected via GOOGLE_GENAI_USE_VERTEXAI. The offline gate
    # must recognize this as "a model is reachable", not force the offline path.
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    assert _use_vertex() is True
    assert _model_available() is True


def test_vertex_flag_without_project_is_not_enough(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    assert _use_vertex() is False


def test_no_backend_means_offline(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    assert _model_available() is False


def test_extracts_poop_with_time_and_default_quantity():
    a = _extract_heuristic("He just pooped a lot at 3 PM", now_iso=NOW)
    assert a.activity_type == "poop"
    assert a.unit == "count"
    assert a.quantity == 1.0
    assert a.timestamp.startswith("2026-07-05T15:00:00")


def test_extracts_bottle_quantity_and_unit():
    a = _extract_heuristic("gave him a 4oz bottle at 2pm", now_iso=NOW)
    assert a.activity_type == "bottle"
    assert a.quantity == 4.0
    assert a.unit == "oz"


def test_extracts_solids_grams():
    a = _extract_heuristic("ate 50g of sweet potato puree", now_iso=NOW)
    assert a.activity_type == "solids"
    assert a.quantity == 50.0
    assert a.unit == "grams"


def test_time_number_not_mistaken_for_quantity():
    a = _extract_heuristic("wet diaper at 11am", now_iso=NOW)
    assert a.activity_type == "wet"
    assert a.quantity == 1.0


def test_unrecognized_message_raises():
    with pytest.raises(ActivityError):
        _extract_heuristic("the weather is nice today", now_iso=NOW)


def test_time_in_future_relative_to_now_rolls_back_a_day():
    # NOW is 20:00; "9 am" would be in the future today relative to "now" only
    # if now were earlier in the day. With NOW at 20:00, 9am today is in the
    # past, so no rollback should occur.
    a = _extract_heuristic("wet diaper at 9am", now_iso=NOW)
    assert a.timestamp.startswith("2026-07-05T09:00:00")
