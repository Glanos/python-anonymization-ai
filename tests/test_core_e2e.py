"""End-to-end tests for glanos_anonymization.core against the real Glanos API.

Reads GLANOS_TOKEN and GLANOS_BASE_URL from the environment (loaded from the
repo's .env file by pytest-dotenv). These tests are skipped automatically if
either is missing.
"""

import base64
import json as json_module
import os
from pathlib import Path

import pytest

from glanos_anonymization.core import (
    AnonymizationConfig,
    DepseudoPaths,
    GlanosClient,
    InnerResponseData,
    InputData,
)

GLANOS_API_KEY = os.environ.get("GLANOS_TOKEN", "")
GLANOS_BASE_URL = os.environ.get("GLANOS_BASE_URL", "")

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        not GLANOS_API_KEY, reason="GLANOS_API_KEY not set - skipping e2e tests"
    ),
    pytest.mark.skipif(
        not GLANOS_BASE_URL, reason="GLANOS_BASE_URL not set - skipping e2e tests"
    ),
]


@pytest.fixture
def example_docx_file() -> Path:
    return Path(__file__).parent / "data" / "example.docx"


@pytest.fixture
def client():
    c = GlanosClient(token=GLANOS_API_KEY, base_url=GLANOS_BASE_URL)
    yield c
    c.close()


class TestAnonymizeRealApi:
    def test_anonymize_text(self, client):
        result = client.anonymize(text="Mein Name ist Michael Huber")
        assert result.text

    def test_anonymize_docx_file(self, client, example_docx_file):
        result = client.anonymize(path=example_docx_file)
        assert result.text


class TestPseudoFetchDepseudoRoundTrip:
    def test_sync_pseudo_returns_completed_result_with_pseudo_key(self, tmp_path):
        config = AnonymizationConfig(
            a_sync=False, return_pseudo_key=True, return_encrypt_key=True
        )
        client = GlanosClient(
            token=GLANOS_API_KEY, base_url=GLANOS_BASE_URL, config=config
        )
        try:
            result = client.pseudo_sync(text="Mein Name ist Michael Huber")
            assert isinstance(result.data, InnerResponseData)
            assert result.data.pseudoKey
            # Note: the real API does not always return encryptKey even when
            # return_encrypt_key=True is requested - not asserted here.
        finally:
            client.close()

    def test_async_pseudo_then_fetch(self):
        config = AnonymizationConfig(a_sync=True)
        client = GlanosClient(
            token=GLANOS_API_KEY, base_url=GLANOS_BASE_URL, config=config
        )
        try:
            submitted = client.pseudo_sync(text="Mein Name ist Michael Huber")
            fetched = client.fetch(submitted)
            assert isinstance(fetched.data, InnerResponseData)
        finally:
            client.close()

    def test_depseudo_with_meta_key_via_fetch_restores_original_text(self, tmp_path):
        # pseudoKeyRetentionMS is only honored for the async submit -> fetch
        # workflow, and must be set at submission time (not at fetch time).
        # A *sync* pseudo_sync call discards its pseudoKey server-side
        # immediately regardless of pseudoKeyRetentionMS - see
        # TestPseudoKeyRetentionBehavior.test_immediate_meta_key_only_depseudo_after_sync_pseudo.
        # metaRetentionMS must ALSO be set explicitly - without it, depseudo
        # fails with "No such id ..." even with pseudoKeyRetentionMS set.
        config = AnonymizationConfig(
            a_sync=True, pseudo_key_retention_ms=60_000, meta_retention_ms=60_000
        )
        client = GlanosClient(
            token=GLANOS_API_KEY, base_url=GLANOS_BASE_URL, config=config
        )
        try:
            original_text = "Mein Name ist Michael Huber und ich wohne in Berlin"
            submitted = client.pseudo_sync(text=original_text)
            pseudo_result = client.fetch(submitted)
            assert isinstance(pseudo_result.data, InnerResponseData)

            anonymized_file = tmp_path / "anonymized.txt"
            anonymized_file.write_bytes(base64.b64decode(pseudo_result.data.data))

            depseudo_paths = DepseudoPaths(meta_key=pseudo_result.data.metaKey)
            depseudo_result = client.depseudo(
                anonymized_file, depseudo_paths, filename=pseudo_result.data.fileName
            )

            assert isinstance(depseudo_result.data, InnerResponseData)
        finally:
            client.close()


class TestPseudoKeyRetentionBehavior:
    """pseudoKeyRetentionMS is only honored for the async submit -> fetch
    workflow, and must be set at submission time. A synchronous pseudo_sync
    call discards its pseudoKey server-side immediately regardless of this
    setting. Run with -s to see the printed request/response details."""

    def test_pseudo_key_retention_ms_is_actually_transmitted(self, tmp_path):
        """Verify pseudoKeyRetentionMS is present in the exact JSON body sent
        to the API."""
        config = AnonymizationConfig(a_sync=False, pseudo_key_retention_ms=60_000)
        client = GlanosClient(
            token=GLANOS_API_KEY, base_url=GLANOS_BASE_URL, config=config
        )
        try:
            input_data = InputData(
                content="Mein Name ist Michael Huber",
                source="string",
                filename="plain_text.txt",
            )
            body = client.build_pseudo_request_body(input_data)
            print("\n--- pseudo request body ---")
            print(json_module.dumps(body, indent=2))

            assert body["pseudoKeyRetentionMS"] == 60_000

            result = client.pseudo_sync(text="Mein Name ist Michael Huber")
            assert isinstance(result.data, InnerResponseData)
            print("\n--- pseudo response data ---")
            print(result.data.model_dump_json(indent=2))
        finally:
            client.close()

    def test_async_pseudo_with_retention_then_fetch_then_meta_key_depseudo(
        self, tmp_path
    ):
        """pseudoKeyRetentionMS set at submit time, combined with an async
        submission + explicit fetch, is the combination that keeps the
        pseudoKey available for a later metaKey-only depseudo (contrast with
        test_immediate_meta_key_only_depseudo_after_sync_pseudo, where a
        sync call discards it regardless of retention settings)."""
        config = AnonymizationConfig(
            a_sync=True, pseudo_key_retention_ms=60_000, meta_retention_ms=60_000
        )
        client = GlanosClient(
            token=GLANOS_API_KEY, base_url=GLANOS_BASE_URL, config=config
        )
        try:
            submitted = client.pseudo_sync(text="Mein Name ist Michael Huber")
            pseudo_result = client.fetch(submitted)
            assert isinstance(pseudo_result.data, InnerResponseData)
            print("\n--- fetch response ---")
            print(pseudo_result.data.model_dump_json(indent=2))

            anonymized_file = tmp_path / "anonymized.txt"
            anonymized_file.write_bytes(base64.b64decode(pseudo_result.data.data))

            depseudo_paths = DepseudoPaths(meta_key=pseudo_result.data.metaKey)
            depseudo_result = client.depseudo(
                anonymized_file, depseudo_paths, filename=pseudo_result.data.fileName
            )
            print("\n--- depseudo response ---")
            print(depseudo_result.data.model_dump_json(indent=2))

            assert isinstance(depseudo_result.data, InnerResponseData)
        finally:
            client.close()

    @pytest.mark.xfail(
        reason="A sync pseudo_sync call discards its pseudoKey server-side "
        "immediately, even with a generous pseudoKeyRetentionMS - metaKey-only "
        "depseudo of a sync result is not possible (use return_pseudo_key=True "
        "+ pseudo_key_path instead).",
        strict=True,
    )
    def test_immediate_meta_key_only_depseudo_after_sync_pseudo(self, tmp_path):
        """pseudoKeyRetentionMS has no effect for sync requests - depseudo by
        metaKey alone fails even immediately after the pseudo call, with a
        generous retention window set."""
        config = AnonymizationConfig(
            a_sync=False, pseudo_key_retention_ms=60_000, meta_retention_ms=60_000
        )
        client = GlanosClient(
            token=GLANOS_API_KEY, base_url=GLANOS_BASE_URL, config=config
        )
        try:
            pseudo_result = client.pseudo_sync(text="Mein Name ist Michael Huber")
            assert isinstance(pseudo_result.data, InnerResponseData)
            print("\n--- pseudo response ---")
            print(pseudo_result.data.model_dump_json(indent=2))

            anonymized_file = tmp_path / "anonymized.txt"
            anonymized_file.write_bytes(base64.b64decode(pseudo_result.data.data))

            depseudo_paths = DepseudoPaths(meta_key=pseudo_result.data.metaKey)
            depseudo_body = client.build_depseudo_request_body(
                InputData(
                    content=base64.b64decode(pseudo_result.data.data),
                    source="file",
                    filename=pseudo_result.data.fileName,
                ),
                depseudo_paths,
            )
            print("\n--- depseudo request body ---")
            print(json_module.dumps(depseudo_body, indent=2))

            depseudo_result = client.depseudo(
                anonymized_file, depseudo_paths, filename=pseudo_result.data.fileName
            )
            print("\n--- depseudo response ---")
            print(depseudo_result.data.model_dump_json(indent=2))

            assert isinstance(depseudo_result.data, InnerResponseData)
        finally:
            client.close()


class TestSessionRealApi:
    def test_pseudo_with_session_then_remove_session(self):
        # A session is created by passing sessionMaxAge with 'session' left
        # unset - the server generates the id and returns it. Passing a
        # self-chosen id (session=...) is treated as referencing an existing
        # (and here, nonexistent) session, which the API rejects as expired.
        config = AnonymizationConfig(a_sync=False, session_max_age=60_000)
        client = GlanosClient(
            token=GLANOS_API_KEY, base_url=GLANOS_BASE_URL, config=config
        )
        try:
            result = client.pseudo_sync(text="Mein Name ist Michael Huber")
            assert isinstance(result.data, InnerResponseData)
            assert result.data.session

            removed = client.remove_session(result.data.session)
            assert removed.success is True
        finally:
            client.close()

    def test_depseudo_text_restores_pseudonym_via_session(self):
        """The exact flow the middleware relies on: anonymize text with a
        session, then depseudonymize (different) text with that same
        session and confirm the pseudonym it introduced gets inverted back
        to the original value - without ever touching a file."""
        client = GlanosClient(
            token=GLANOS_API_KEY,
            base_url=GLANOS_BASE_URL,
            config=AnonymizationConfig(a_sync=False),
        )
        try:
            original_text = "Mein Name ist Michael Huber."

            # 1. Create a session (sessionMaxAge set, session left unset).
            created = client.pseudo_sync(text=".", session_max_age=60_000)
            assert isinstance(created.data, InnerResponseData)
            session_id = created.data.session
            assert session_id
            print(f"\nsession_id: {session_id}")

            # 2. Anonymize real content using that session.
            anonymized = client.pseudo_sync(text=original_text, session=session_id)
            assert isinstance(anonymized.data, InnerResponseData)
            anonymized_text = client.get_ano_result(anonymized).text
            print(f"anonymized_text: {anonymized_text!r}")
            assert "Michael" not in anonymized_text

            # 3. Simulate an LLM echoing the pseudonym back, and restore it
            # via depseudo_text using the SAME session - no file involved.
            restored_text = client.depseudo_text(anonymized_text, session=session_id)
            print(f"restored_text: {restored_text!r}")

            assert "Michael" in restored_text

            client.remove_session(session_id)
        finally:
            client.close()
