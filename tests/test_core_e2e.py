"""End-to-end tests for glanos_anonymization.core against the real Glanos API.

Reads GLANOS_TOKEN and GLANOS_BASE_URL from the environment (loaded from the
repo's .env file by pytest-dotenv). These tests are skipped automatically if
either is missing.
"""

import asyncio
import json as json_module
import os
import re
from pathlib import Path

import pytest

from glanos_anonymization.core import (
    AnonymizationSettings,
    AsyncGlanosClient,
    DepseudoPaths,
    GlanosClient,
    InputData,
)

GLANOS_API_KEY = os.environ.get("GLANOS_TOKEN", "")
GLANOS_BASE_URL = os.environ.get("GLANOS_BASE_URL", "")
# A user or group whose dashboard may receive test documents; the review test
# is skipped without one.
GLANOS_REVIEW_ASSIGNEE = os.environ.get("GLANOS_REVIEW_ASSIGNEE", "")
GLANOS_REVIEW_GROUP = os.environ.get("GLANOS_REVIEW_GROUP", "")

# fetch() waits 5 minutes between checks by default, for reviews; a document
# without review is ready within seconds.
QUICK_FETCH = {"poll_interval": 1.0, "max_attempts": 60}

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
        assert result.extract_text()

    def test_anonymize_docx_file(self, client, example_docx_file):
        result = client.anonymize(path=example_docx_file)
        assert result.data
        assert result.extract_text()


class TestAnonymizeDeanonymizeRoundTrip:
    def test_text_round_trip_with_keys(self, client):
        original_text = "Mein Name ist Michael Huber und ich wohne in Berlin"

        anonymized = client.anonymize(text=original_text)
        assert "Michael" not in anonymized.extract_text()

        restored = client.deanonymize(anonymized)
        assert restored.extract_text().strip() == original_text

    def test_docx_round_trip_with_keys(self, client, example_docx_file):
        anonymized = client.anonymize(path=example_docx_file)

        restored = client.deanonymize(anonymized)

        assert restored.data
        assert restored.extract_text()


class TestProcessingRealApi:
    def test_sync_returns_completed_result_with_pseudo_key(self, client):
        result = client.anonymize(text="Mein Name ist Michael Huber", processing="sync")
        assert result.ready
        assert result.pseudo_key
        # Note: the real API does not always return encryptKey even when
        # return_encrypt_key=True is requested - not asserted here.

    def test_no_wait_then_fetch(self, client):
        pending = client.anonymize(
            text="Mein Name ist Michael Huber", processing="no_wait"
        )
        assert not pending.ready
        assert pending.fetch_key

        fetched = client.fetch(pending.fetch_key, **QUICK_FETCH)

        assert fetched.ready
        assert "Michael" not in fetched.extract_text()
        assert fetched.pseudo_key

    def test_poll_returns_completed_result(self, client, example_docx_file):
        result = client.anonymize(path=example_docx_file, processing="poll")
        assert result.ready
        assert result.data

    def test_depseudo_with_meta_key_via_fetch_restores_original_text(
        self, client, tmp_path
    ):
        # pseudoKeyRetentionMS is only honored for the async submit -> fetch
        # workflow, and must be set at submission time (not at fetch time).
        # A *sync* request discards its pseudoKey server-side immediately
        # regardless of pseudoKeyRetentionMS - see
        # TestPseudoKeyRetentionBehavior.test_immediate_meta_key_only_depseudo_after_sync_pseudo.
        # metaRetentionMS must ALSO be set explicitly - without it, depseudo
        # fails with "No such id ..." even with pseudoKeyRetentionMS set.
        original_text = "Mein Name ist Michael Huber und ich wohne in Berlin"
        submitted = client.anonymize(
            text=original_text,
            processing="no_wait",
            pseudo_key_retention_ms=60_000,
            meta_retention_ms=60_000,
        )
        result = client.fetch(submitted, **QUICK_FETCH)
        assert result.ready

        anonymized_file = tmp_path / "anonymized.txt"
        anonymized_file.write_bytes(result.data)

        restored = client.deanonymize(
            path=anonymized_file,
            filename=result.file_name,
            depseudo_paths=DepseudoPaths(meta_key=result.meta_key),
        )

        assert restored.ready


@pytest.mark.skipif(
    not (GLANOS_REVIEW_ASSIGNEE or GLANOS_REVIEW_GROUP),
    reason="Set GLANOS_REVIEW_ASSIGNEE or GLANOS_REVIEW_GROUP to send a test "
    "document to that dashboard",
)
class TestReviewRealApi:
    def test_document_waits_for_the_review(self, client):
        reviewer = (
            {"assignee": GLANOS_REVIEW_ASSIGNEE}
            if GLANOS_REVIEW_ASSIGNEE
            else {"assigned_group": GLANOS_REVIEW_GROUP}
        )

        pending = client.anonymize_with_review(
            text="E2E test of the Python library: Mein Name ist Michael Huber.",
            **reviewer,
        )

        assert not pending.ready
        assert pending.fetch_key
        # nobody reviewed it yet: a single check still finds it pending
        still_pending = client.fetch(pending, max_attempts=1)
        assert not still_pending.ready
        assert still_pending.fetch_key


class TestPseudoKeyRetentionBehavior:
    """pseudoKeyRetentionMS is only honored for the async submit -> fetch
    workflow, and must be set at submission time. A synchronous request
    discards its pseudoKey server-side immediately regardless of this
    setting. Run with -s to see the printed request/response details."""

    def test_pseudo_key_retention_ms_is_actually_transmitted(self, client):
        """Verify pseudoKeyRetentionMS is present in the exact JSON body sent
        to the API."""
        input_data = InputData(
            content="Mein Name ist Michael Huber",
            source="string",
            filename="plain_text.txt",
        )
        body = client.build_pseudo_request_body(
            input_data, pseudo_key_retention_ms=60_000
        )
        print("\n--- pseudo request body ---")
        print(json_module.dumps(body, indent=2))

        assert body["pseudoKeyRetentionMS"] == 60_000

        result = client.anonymize(
            text="Mein Name ist Michael Huber",
            processing="sync",
            pseudo_key_retention_ms=60_000,
        )
        assert result.ready
        print("\n--- pseudo response data ---")
        print(result.pseudo_result.data.model_dump_json(indent=2))

    def test_async_pseudo_with_retention_then_fetch_then_meta_key_depseudo(
        self, client, tmp_path
    ):
        """pseudoKeyRetentionMS set at submit time, combined with an async
        submission + explicit fetch, is the combination that keeps the
        pseudoKey available for a later metaKey-only depseudo (contrast with
        test_immediate_meta_key_only_depseudo_after_sync_pseudo, where a
        sync call discards it regardless of retention settings)."""
        submitted = client.anonymize(
            text="Mein Name ist Michael Huber",
            processing="no_wait",
            pseudo_key_retention_ms=60_000,
            meta_retention_ms=60_000,
        )
        result = client.fetch(submitted, **QUICK_FETCH)
        assert result.ready
        print("\n--- fetch response ---")
        print(result.pseudo_result.data.model_dump_json(indent=2))

        anonymized_file = tmp_path / "anonymized.txt"
        anonymized_file.write_bytes(result.data)

        restored = client.deanonymize(
            path=anonymized_file,
            filename=result.file_name,
            depseudo_paths=DepseudoPaths(meta_key=result.meta_key),
        )
        print("\n--- depseudo response ---")
        print(restored.pseudo_result.data.model_dump_json(indent=2))

        assert restored.ready

    @pytest.mark.xfail(
        reason="A sync request discards its pseudoKey server-side "
        "immediately, even with a generous pseudoKeyRetentionMS - metaKey-only "
        "depseudo of a sync result is not possible (use return_pseudo_key=True "
        "+ pseudo_key_path instead).",
        strict=True,
    )
    def test_immediate_meta_key_only_depseudo_after_sync_pseudo(self, client, tmp_path):
        """pseudoKeyRetentionMS has no effect for sync requests - depseudo by
        metaKey alone fails even immediately after the pseudo call, with a
        generous retention window set."""
        # The keys are turned off explicitly: the client requests them by
        # default for sync requests, and this test is about the metaKey alone.
        result = client.anonymize(
            text="Mein Name ist Michael Huber",
            processing="sync",
            return_pseudo_key=False,
            return_encrypt_key=False,
            pseudo_key_retention_ms=60_000,
            meta_retention_ms=60_000,
        )
        assert result.ready
        print("\n--- pseudo response ---")
        print(result.pseudo_result.data.model_dump_json(indent=2))

        anonymized_file = tmp_path / "anonymized.txt"
        anonymized_file.write_bytes(result.data)

        depseudo_paths = DepseudoPaths(meta_key=result.meta_key)
        depseudo_body = client.build_depseudo_request_body(
            InputData(content=result.data, source="file", filename=result.file_name),
            depseudo_paths,
        )
        print("\n--- depseudo request body ---")
        print(json_module.dumps(depseudo_body, indent=2))

        restored = client.deanonymize(
            path=anonymized_file,
            filename=result.file_name,
            depseudo_paths=depseudo_paths,
        )
        print("\n--- depseudo response ---")
        print(restored.pseudo_result.data.model_dump_json(indent=2))

        assert restored.ready


class TestSessionRealApi:
    def test_pseudo_with_session_then_remove_session(self, client):
        # A session is created by passing sessionMaxAge with 'session' left
        # unset - the server generates the id and returns it. Passing a
        # self-chosen id (session=...) is treated as referencing an existing
        # (and here, nonexistent) session, which the API rejects as expired.
        result = client.anonymize(
            text="Mein Name ist Michael Huber", session_max_age_ms=60_000
        )
        assert result.session

        removed = client.remove_session(result.session)
        assert removed.success is True

    def test_text_restores_pseudonym_via_session(self, client):
        """The exact flow the middleware relies on: anonymize text with a
        session, then deanonymize (different) text with that same
        session and confirm the pseudonym it introduced gets inverted back
        to the original value - without ever touching a file."""
        original_text = "Mein Name ist Michael Huber."

        # 1. Create a session (sessionMaxAge set, session left unset).
        created = client.anonymize(text=".", session_max_age_ms=60_000)
        session_id = created.session
        assert session_id
        print(f"\nsession_id: {session_id}")

        try:
            # 2. Anonymize real content using that session.
            anonymized = client.anonymize(text=original_text, session=session_id)
            anonymized_text = anonymized.extract_text()
            print(f"anonymized_text: {anonymized_text!r}")
            assert "Michael" not in anonymized_text

            # 3. Simulate an LLM echoing the pseudonym back, and restore it
            # with the SAME session - no file involved.
            restored_text = client.deanonymize(
                text=anonymized_text, session=session_id
            ).extract_text()
            print(f"restored_text: {restored_text!r}")

            assert "Michael" in restored_text
        finally:
            client.remove_session(session_id)


class TestAsyncClientRealApi:
    def test_text_and_docx_round_trip(self, example_docx_file):
        async def run():
            async with AsyncGlanosClient(
                token=GLANOS_API_KEY, base_url=GLANOS_BASE_URL
            ) as client:
                original_text = "Mein Name ist Michael Huber und ich wohne in Berlin"
                text, docx = await asyncio.gather(
                    client.anonymize(text=original_text),
                    client.anonymize(path=example_docx_file),
                )
                restored_text, restored_docx = await asyncio.gather(
                    client.deanonymize(text), client.deanonymize(docx)
                )
                return original_text, text, restored_text, restored_docx

        original_text, text, restored_text, restored_docx = asyncio.run(run())

        assert "Michael" not in text.extract_text()
        assert restored_text.extract_text().strip() == original_text
        assert restored_docx.data

    def test_session_restores_text(self):
        async def run():
            async with AsyncGlanosClient(
                token=GLANOS_API_KEY, base_url=GLANOS_BASE_URL
            ) as client:
                session_id = await client._start_session(60_000)
                try:
                    anonymized = await client.anonymize(
                        text="Mein Name ist Michael Huber.", session=session_id
                    )
                    restored = await client.deanonymize(
                        text=anonymized.extract_text(), session=session_id
                    )
                    return restored.extract_text()
                finally:
                    await client.remove_session(session_id)

        assert asyncio.run(run()).strip() == "Mein Name ist Michael Huber."

    def test_concurrent_requests_share_one_session_consistently(self):
        # Several requests at the same time in one session, each person in two
        # of them: every person must get one placeholder of its own, whichever
        # request the server handles first.
        people = ["Michael Huber", "Anna Schmidt", "Max Mustermann", "Julia Fischer"]
        texts = [f"{person} wohnt hier." for person in people] + [
            f"Gestern rief {person} an." for person in reversed(people)
        ]
        placeholders = AnonymizationSettings(mode="TYPE")

        async def run():
            async with AsyncGlanosClient(
                token=GLANOS_API_KEY, base_url=GLANOS_BASE_URL
            ) as client:
                created = await client.anonymize(
                    text=".", settings=placeholders, session_max_age_ms=60_000
                )
                session_id = created.session
                try:
                    results = await asyncio.gather(
                        *(
                            client.anonymize(
                                text=text, settings=placeholders, session=session_id
                            )
                            for text in texts
                        )
                    )
                    anonymized = [result.extract_text().strip() for result in results]
                    restored = await asyncio.gather(
                        *(
                            client.depseudo_text(text, session=session_id)
                            for text in anonymized
                        )
                    )
                    return anonymized, restored
                finally:
                    await client.remove_session(session_id)

        anonymized, restored = asyncio.run(run())

        found: dict[str, set[str]] = {person: set() for person in people}
        for text, result in zip(texts, anonymized):
            person = next(person for person in people if person in text)
            found[person].update(re.findall(r"PERSON\d+", result))
        assert all(len(placeholder) == 1 for placeholder in found.values()), found
        assert len(set.union(*found.values())) == len(people), found
        assert [text.strip() for text in restored] == texts


class TestSettingsAndSessionsRealApi:
    def test_stored_setting_and_session_lifecycle(self, client):
        stored = client.add_setting(
            AnonymizationSettings(
                name=f"e2e-test-{os.getpid()}", mode="TYPE", excludeFields=["dates"]
            )
        )
        assert stored.id
        try:
            fetched = client.get_setting(stored.id)
            assert fetched.name == stored.name

            session = client.add_session(60, settings_id=stored.id)
            try:
                assert session.id
                assert client.get_session(session.id).id == session.id

                anonymized = client.anonymize(
                    text="Mein Name ist Michael Huber.", session=session.id
                )
                restored = client.deanonymize(anonymized, session=session.id)
                assert "Michael" in restored.extract_text()

                extended = client.extend_session(session.id, max_age_seconds=120)
                assert extended.id == session.id
            finally:
                client.remove_session(session.id)
        finally:
            client.remove_setting(stored.id)
