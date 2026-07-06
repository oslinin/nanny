"""When a configured model call fails at runtime (invalid key, quota
exhausted, timeout), each LlmAgent must degrade to the same offline path its
no-key fallback uses — not abort the turn with a 500. These test the
``on_model_error_callback`` hooks directly, since reproducing a live model
failure end-to-end would require network + a broken credential.
"""

from types import SimpleNamespace

from nanny.agents import (
    _classifier_model_error_callback,
    _responder_model_error_callback,
    build_classifier_agent,
    build_responder_agent,
)
from nanny.research import _insights_model_error_callback, build_insights_agent


def _ctx(state: dict) -> SimpleNamespace:
    """A stand-in CallbackContext: the callbacks only touch ``.state``."""
    return SimpleNamespace(state=state)


def _text(resp) -> str:
    return resp.content.parts[0].text


def test_responder_model_error_falls_back_to_template():
    state = {
        "save_result": {
            "saved": {"activity_type": "bottle", "quantity": 4.0, "unit": "oz"},
            "today_count": 1,
            "today_total": 4.0,
            "today_unit": "oz",
        }
    }
    resp = _responder_model_error_callback(
        callback_context=_ctx(state), llm_request=None, error=Exception("boom")
    )
    assert resp is not None
    assert "4" in _text(resp)
    assert state["used_llm_response"] is False


def test_classifier_model_error_falls_back_to_heuristic():
    state = {"chat_text": "he pooped a lot", "now_iso": "2026-07-05T20:00:00+00:00"}
    resp = _classifier_model_error_callback(
        callback_context=_ctx(state), llm_request=None, error=Exception("boom")
    )
    assert resp is not None
    assert "poop" in _text(resp)
    assert state["used_llm_extraction"] is False


def test_classifier_model_error_on_unrecognized_text_sets_heuristic_error():
    # Heuristic can't extract → returns the schema-valid sentinel and flags the
    # error so classifier_postprocess_node routes to the friendly error branch.
    state = {"chat_text": "the weather is nice", "now_iso": "2026-07-05T20:00:00+00:00"}
    resp = _classifier_model_error_callback(
        callback_context=_ctx(state), llm_request=None, error=Exception("boom")
    )
    assert resp is not None
    assert state["used_llm_extraction"] is False
    assert "heuristic_error" in state


def test_insights_model_error_falls_back_to_summary():
    state = {
        "insights_context": {"total_records": 2, "days": 1},
        "question": "",
    }
    resp = _insights_model_error_callback(
        callback_context=_ctx(state), llm_request=None, error=Exception("boom")
    )
    assert resp is not None
    assert _text(resp).strip() != ""
    assert state["used_llm_response"] is False


def test_agents_have_error_callback_wired():
    # Guards against the callback existing but never being attached to the agent.
    assert build_classifier_agent().on_model_error_callback is not None
    assert build_responder_agent().on_model_error_callback is not None
    assert build_insights_agent().on_model_error_callback is not None
