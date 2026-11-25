"""End-to-end tests for GlanosAnonymizationTools against the real Glanos API.

The tool methods are called directly, as an agent would call them. Skipped
automatically if GLANOS_TOKEN or GLANOS_BASE_URL is missing.
"""

import base64
import os
import time
from pathlib import Path

import pytest
from markitdown import MarkItDown

from glanos_anonymization import AnonymizationSettings, GlanosAnonymizationTools

GLANOS_API_KEY = os.environ.get("GLANOS_TOKEN", "")
GLANOS_BASE_URL = os.environ.get("GLANOS_BASE_URL", "")

EXAMPLE_DOCX = Path(__file__).parent / "data" / "example.docx"

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        not GLANOS_API_KEY, reason="GLANOS_TOKEN not set - skipping e2e tests"
    ),
    pytest.mark.skipif(
        not GLANOS_BASE_URL, reason="GLANOS_BASE_URL not set - skipping e2e tests"
    ),
]


@pytest.fixture
def tools():
    t = GlanosAnonymizationTools(
        token=GLANOS_API_KEY, base_url=GLANOS_BASE_URL, session_max_age_ms=60_000
    )
    sessions: list[str] = []
    yield t, sessions
    for session in set(sessions):
        t.remove_session(session)
    t.close()


def test_text_round_trip_with_the_session_it_started(tools):
    tools, sessions = tools

    anonymized = tools.anonymize(text="Mein Name ist Michael Huber.")
    sessions.append(anonymized["session"])

    assert set(anonymized) == {"text", "session"}
    assert "Huber" not in anonymized["text"]
    restored = tools.deanonymize(text=anonymized["text"], session=anonymized["session"])
    assert "Michael Huber" in restored["text"]


def test_a_reused_session_keeps_the_pseudonym(tools):
    tools, sessions = tools

    placeholders = AnonymizationSettings(mode="TYPE")

    first = tools.anonymize(text="Mein Name ist Michael Huber.", settings=placeholders)
    sessions.append(first["session"])
    # A new person first: in a new session, Anna Schmidt would be PERSON1.
    second = tools.anonymize(
        text="Anna Schmidt wohnt bei Michael Huber.",
        settings=placeholders,
        session=first["session"],
    )

    assert second["session"] == first["session"]
    assert first["text"].strip() == "Mein Name ist PERSON1."
    assert second["text"].strip() == "PERSON2 wohnt bei PERSON1."


def test_document_round_trip_with_the_session(tools):
    tools, sessions = tools
    original = EXAMPLE_DOCX.read_bytes()

    anonymized = tools.anonymize(
        file_base64=base64.b64encode(original).decode(), file_name=EXAMPLE_DOCX.name
    )
    sessions.append(anonymized["session"])

    assert set(anonymized) == {"text", "file_base64", "file_name", "session"}
    restored = tools.deanonymize(
        file_base64=anonymized["file_base64"],
        file_name=anonymized["file_name"],
        session=anonymized["session"],
    )
    original_text = MarkItDown().convert(str(EXAMPLE_DOCX)).text_content
    assert anonymized["text"].split() != original_text.split()
    assert restored["text"].split() == original_text.split()


def test_a_document_is_collected_with_fetch_and_restored_with_its_session(tools):
    tools, sessions = tools
    pending = tools.client.anonymize(
        path=EXAMPLE_DOCX, processing="no_wait", session_max_age_ms=60_000
    )

    fetched = tools.fetch(pending.fetch_key)
    for _ in range(30):
        if "file_base64" in fetched:
            break
        time.sleep(1)
        fetched = tools.fetch(pending.fetch_key)
    assert set(fetched) == {"text", "file_base64", "file_name", "session"}
    sessions.append(fetched["session"])

    restored = tools.deanonymize(
        file_base64=fetched["file_base64"],
        file_name=fetched["file_name"],
        session=fetched["session"],
    )
    original_text = MarkItDown().convert(str(EXAMPLE_DOCX)).text_content
    assert fetched["text"].split() != original_text.split()
    assert restored["text"].split() == original_text.split()
