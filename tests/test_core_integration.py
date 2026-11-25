"""Integration tests for glanos_anonymization.core.

These exercise GlanosClient's higher-level methods end-to-end (request building
+ HTTP call + response parsing) against a fake HTTP layer built with
httpx.MockTransport - no real network access, but no internals are mocked out
either.
"""

import base64
import json

import httpx
import pytest

from glanos_anonymization.core import (
    AnonymizationResult,
    AnonymizationSettings,
    DataExpiredError,
    DeanonymizationResult,
    DepseudoPaths,
    DownloadOption,
    GlanosClient,
    GlanosClientError,
    ProcessingFailedError,
    PseudoResult,
    QuotaExceededError,
    RateLimitedError,
    ServerError,
    SessionInfo,
    SettingNotFoundError,
    SettingsConflictError,
    SettingsSort,
    SortOrder,
    SuccessMessage,
)

pytestmark = pytest.mark.integration


def b64(content: str | bytes) -> str:
    if isinstance(content, str):
        content = content.encode("utf-8")
    return base64.b64encode(content).decode("ascii")


class FakeTextExtractor:
    def extract_text(self, pseudo_result) -> str:
        return f"text of {pseudo_result.data.fileName}"


class FailingTextExtractor:
    def extract_text(self, pseudo_result) -> str:
        raise AssertionError("text extraction was not requested")


def make_client(handler, **kwargs) -> GlanosClient:
    transport = httpx.MockTransport(handler)
    return GlanosClient(
        token="tok-123",
        base_url="https://glanos.example.test",
        transport=transport,
        **kwargs,
    )


class RequestRecorder:
    """A MockTransport handler that records every request and returns canned responses."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]

    def paths(self) -> list[str]:
        return [
            request.url.path.removeprefix("/be/api-v2") for request in self.requests
        ]


@pytest.fixture
def sleeps(monkeypatch) -> list[float]:
    """Records the pauses between two polls instead of waiting."""
    recorded: list[float] = []
    monkeypatch.setattr("glanos_anonymization.core.time.sleep", recorded.append)
    return recorded


class TestAnonymizeRequest:
    def test_completed_result_and_request_shape(self, completed_pseudo_response_json):
        recorder = RequestRecorder(
            [httpx.Response(200, json=completed_pseudo_response_json)]
        )
        client = make_client(recorder)

        result = client.anonymize(text="Mein Name ist Michael Huber")

        assert result.ready
        request = recorder.requests[0]
        assert request.method == "POST"
        assert str(request.url).startswith(
            "https://glanos.example.test/be/api-v2/document/pseudo?"
        )
        assert request.url.params["token"] == "tok-123"

        body = json.loads(request.content)
        assert body["plainText"] == "Mein Name ist Michael Huber"
        assert body["fileName"] == "plain_text.txt"
        client.close()

    def test_text_defaults_to_one_sync_request_with_both_keys(
        self, completed_pseudo_response_json
    ):
        """The keys are requested so that every result can be restored."""
        recorder = RequestRecorder(
            [httpx.Response(200, json=completed_pseudo_response_json)]
        )
        client = make_client(recorder)

        client.anonymize(text="hi")

        assert recorder.paths() == ["/document/pseudo"]
        params = recorder.requests[0].url.params
        assert params["async"] == "false"
        assert params["returnPseudoKey"] == "true"
        assert params["returnEncryptKey"] == "true"
        client.close()

    def test_keys_can_be_turned_off(self, completed_pseudo_response_json):
        recorder = RequestRecorder(
            [httpx.Response(200, json=completed_pseudo_response_json)]
        )
        client = make_client(recorder)

        client.anonymize(text="hi", return_pseudo_key=False, return_encrypt_key=False)

        params = recorder.requests[0].url.params
        assert params["returnPseudoKey"] == "false"
        assert params["returnEncryptKey"] == "false"
        client.close()

    def test_request_options_are_sent(self, completed_pseudo_response_json):
        recorder = RequestRecorder(
            [httpx.Response(200, json=completed_pseudo_response_json)]
        )
        client = make_client(recorder)

        client.anonymize(
            text="hi",
            settings=AnonymizationSettings(mode="XXX"),
            session="sess-1",
            session_max_age_ms=60_000,
            language="de",
            options=["opt1"],
        )
        client.anonymize(text="hi", settings_id="set-1")

        body = json.loads(recorder.requests[0].content)
        assert body["settings"]["mode"] == "XXX"
        assert body["session"] == "sess-1"
        assert body["sessionMaxAge"] == 60_000
        assert body["language"] == "de"
        assert body["options"] == ["opt1"]
        assert "assignee" not in body
        assert json.loads(recorder.requests[1].content)["settingsId"] == "set-1"
        client.close()

    def test_settings_mode_conflict_raises_typed_exception(self):
        client = make_client(
            lambda request: httpx.Response(
                400,
                json={
                    "success": False,
                    "message": "Session settings mode and provided settings "
                    "mode must be equal",
                },
            )
        )
        with pytest.raises(SettingsConflictError):
            client.anonymize(
                text="hi", session="sess-1", settings=AnonymizationSettings(mode="XXX")
            )
        client.close()

    def test_failed_processing_raises_typed_exception(
        self, completed_pseudo_response_json
    ):
        completed_pseudo_response_json["data"].update(
            success=False, failReason="Invalid document"
        )
        client = make_client(
            lambda request: httpx.Response(200, json=completed_pseudo_response_json)
        )
        with pytest.raises(ProcessingFailedError):
            client.anonymize(text="hi")
        client.close()

    def test_quota_exceeded_raises_typed_exception(self):
        client = make_client(lambda request: httpx.Response(402, text="quota exceeded"))
        with pytest.raises(QuotaExceededError):
            client.anonymize(text="hi")
        client.close()

    def test_rate_limited_raises_typed_exception(self):
        client = make_client(lambda request: httpx.Response(429, text="slow down"))
        with pytest.raises(RateLimitedError):
            client.anonymize(text="hi")
        client.close()

    def test_file_upload_sends_base64_data(
        self, completed_pseudo_response_json, tmp_path
    ):
        recorder = RequestRecorder(
            [httpx.Response(200, json=completed_pseudo_response_json)]
        )
        client = make_client(recorder)

        src = tmp_path / "input.txt"
        src.write_bytes(b"file contents")

        client.anonymize(path=src)

        body = json.loads(recorder.requests[0].content)
        assert body["fileName"] == "input.txt"
        assert "data" in body
        assert "plainText" not in body
        client.close()


class TestAnonymizeProcessing:
    def test_sync_returns_a_pending_result_without_polling(
        self, accepted_pseudo_response_json
    ):
        """The server answers 202 when it did not finish in time."""
        recorder = RequestRecorder(
            [httpx.Response(202, json=accepted_pseudo_response_json)]
        )
        client = make_client(recorder)

        result = client.anonymize(text="hi", processing="sync")

        assert not result.ready
        assert result.fetch_key == "fetch-abc-123"
        assert recorder.paths() == ["/document/pseudo"]
        client.close()

    def test_no_wait_submits_and_returns_at_once(self, accepted_pseudo_response_json):
        recorder = RequestRecorder(
            [httpx.Response(202, json=accepted_pseudo_response_json)]
        )
        client = make_client(recorder)

        result = client.anonymize(data=b"docx", filename="a.docx", processing="no_wait")

        assert not result.ready
        assert result.fetch_key == "fetch-abc-123"
        assert recorder.paths() == ["/document/pseudo"]
        params = recorder.requests[0].url.params
        assert params["async"] == "true"
        assert "returnPseudoKey" not in params
        client.close()

    def test_documents_default_to_poll(
        self, accepted_pseudo_response_json, completed_pseudo_response_json, sleeps
    ):
        recorder = RequestRecorder(
            [
                httpx.Response(202, json=accepted_pseudo_response_json),
                httpx.Response(202, json=accepted_pseudo_response_json),
                httpx.Response(200, json=completed_pseudo_response_json),
            ]
        )
        client = make_client(recorder)

        result = client.anonymize(data=b"docx", filename="a.docx")

        assert result.ready
        assert recorder.paths() == [
            "/document/pseudo",
            "/document/pseudo/fetch",
            "/document/pseudo/fetch",
        ]
        # submitted asynchronously; the keys are requested by the fetch
        submit_params = recorder.requests[0].url.params
        assert submit_params["async"] == "true"
        assert "returnPseudoKey" not in submit_params
        fetch_params = recorder.requests[1].url.params
        assert fetch_params["fetchKey"] == "fetch-abc-123"
        assert fetch_params["returnPseudoKey"] == "true"
        assert fetch_params["returnEncryptKey"] == "true"
        assert sleeps == [10.0]
        client.close()

    def test_poll_returns_the_fetch_key_after_the_last_attempt(
        self, accepted_pseudo_response_json, sleeps
    ):
        recorder = RequestRecorder(
            [httpx.Response(202, json=accepted_pseudo_response_json)]
        )
        client = make_client(recorder)

        result = client.anonymize(text="hi", processing="poll")

        assert not result.ready
        assert result.fetch_key == "fetch-abc-123"
        # one submission, 30 checks 10 seconds apart, no pause after the last
        assert recorder.paths().count("/document/pseudo/fetch") == 30
        assert sleeps == [10.0] * 29
        client.close()

    def test_poll_interval_and_max_attempts_can_be_set(
        self, accepted_pseudo_response_json, sleeps
    ):
        recorder = RequestRecorder(
            [httpx.Response(202, json=accepted_pseudo_response_json)]
        )
        client = make_client(recorder)

        result = client.anonymize(
            text="hi", processing="poll", poll_interval=0.5, max_attempts=3
        )

        assert not result.ready
        assert recorder.paths().count("/document/pseudo/fetch") == 3
        assert sleeps == [0.5, 0.5]
        client.close()

    def test_poll_fetches_the_keys_the_call_asked_for(
        self, accepted_pseudo_response_json, completed_pseudo_response_json
    ):
        recorder = RequestRecorder(
            [
                httpx.Response(202, json=accepted_pseudo_response_json),
                httpx.Response(200, json=completed_pseudo_response_json),
            ]
        )
        client = make_client(recorder)

        client.anonymize(text="hi", processing="poll", return_pseudo_key=False)

        fetch_params = recorder.requests[1].url.params
        assert fetch_params["returnPseudoKey"] == "false"
        assert fetch_params["returnEncryptKey"] == "true"
        client.close()

    def test_an_error_while_polling_carries_the_fetch_key(
        self, accepted_pseudo_response_json, sleeps
    ):
        recorder = RequestRecorder(
            [
                httpx.Response(202, json=accepted_pseudo_response_json),
                httpx.Response(202, json=accepted_pseudo_response_json),
                httpx.Response(503, text="restarting"),
            ]
        )
        client = make_client(recorder)

        with pytest.raises(ServerError) as excinfo:
            client.anonymize(text="hi", processing="poll")

        assert any("fetch-abc-123" in note for note in excinfo.value.__notes__)
        assert len(recorder.requests) == 3
        client.close()


class TestAnonymizeWithReview:
    @pytest.mark.parametrize(
        "reviewer, field",
        [
            ({"assignee": "anna@example.com"}, "assignee"),
            ({"assigned_group": "legal"}, "assignedGroup"),
        ],
    )
    def test_submits_and_returns_the_fetch_key_at_once(
        self, accepted_pseudo_response_json, reviewer, field
    ):
        recorder = RequestRecorder(
            [httpx.Response(202, json=accepted_pseudo_response_json)]
        )
        client = make_client(recorder)

        result = client.anonymize_with_review(
            data=b"docx", filename="a.docx", **reviewer
        )

        assert not result.ready
        assert result.fetch_key == "fetch-abc-123"
        # no polling, even though the document is not ready
        assert recorder.paths() == ["/document/pseudo"]
        request = recorder.requests[0]
        assert request.url.params["async"] == "true"
        assert "returnPseudoKey" not in request.url.params
        body = json.loads(request.content)
        (value,) = reviewer.values()
        assert body[field] == value
        (other_field,) = {"assignee", "assignedGroup"} - {field}
        assert other_field not in body
        client.close()

    def test_request_options_are_sent(self, accepted_pseudo_response_json):
        recorder = RequestRecorder(
            [httpx.Response(202, json=accepted_pseudo_response_json)]
        )
        client = make_client(recorder)

        client.anonymize_with_review(
            text="hi",
            assignee="anna@example.com",
            settings_id="set-1",
            session="sess-1",
            data_retention_ms=7 * 24 * 3600 * 1000,
            language="de",
        )

        body = json.loads(recorder.requests[0].content)
        assert body["plainText"] == "hi"
        assert body["settingsId"] == "set-1"
        assert body["session"] == "sess-1"
        assert body["dataRetentionMS"] == 7 * 24 * 3600 * 1000
        assert body["language"] == "de"
        client.close()

    def test_reviewed_document_is_collected_with_fetch(
        self, accepted_pseudo_response_json, completed_pseudo_response_json, sleeps
    ):
        recorder = RequestRecorder(
            [httpx.Response(202, json=accepted_pseudo_response_json)] * 3
            + [httpx.Response(200, json=completed_pseudo_response_json)]
        )
        client = make_client(recorder)

        pending = client.anonymize_with_review(text="hi", assignee="anna@example.com")
        result = client.fetch(pending)

        assert result.ready
        assert result.extract_text() == "Anonymisiertes Dokument"
        assert sleeps == [300.0, 300.0]
        client.close()


class TestFetch:
    def test_polls_until_completed(
        self, accepted_pseudo_response_json, completed_pseudo_response_json, sleeps
    ):
        recorder = RequestRecorder(
            [
                httpx.Response(202, json=accepted_pseudo_response_json),
                httpx.Response(202, json=accepted_pseudo_response_json),
                httpx.Response(200, json=completed_pseudo_response_json),
            ]
        )
        client = make_client(recorder)

        result = client.fetch("abc", poll_interval=1.5)

        assert isinstance(result, AnonymizationResult)
        assert result.ready
        assert len(recorder.requests) == 3
        assert recorder.requests[0].url.params["fetchKey"] == "abc"
        assert sleeps == [1.5, 1.5]
        client.close()

    def test_accepts_a_pending_result(
        self, accepted_pseudo_response_json, completed_pseudo_response_json
    ):
        recorder = RequestRecorder(
            [httpx.Response(200, json=completed_pseudo_response_json)]
        )
        client = make_client(recorder)
        pending = AnonymizationResult(
            PseudoResult.model_validate(accepted_pseudo_response_json)
        )

        result = client.fetch(pending)

        assert result.ready
        assert recorder.requests[0].url.params["fetchKey"] == "fetch-abc-123"
        client.close()

    def test_already_completed_result_short_circuits_without_a_request(
        self, completed_pseudo_response_json
    ):
        recorder = RequestRecorder(
            [httpx.Response(200, json=completed_pseudo_response_json)]
        )
        client = make_client(recorder)
        already_done = AnonymizationResult(
            PseudoResult.model_validate(completed_pseudo_response_json)
        )

        result = client.fetch(already_done)

        assert result is already_done
        assert recorder.requests == []
        client.close()

    def test_waits_every_five_minutes_without_a_limit_by_default(
        self, accepted_pseudo_response_json, completed_pseudo_response_json, sleeps
    ):
        # far more checks than anonymize() would make
        recorder = RequestRecorder(
            [httpx.Response(202, json=accepted_pseudo_response_json)] * 100
            + [httpx.Response(200, json=completed_pseudo_response_json)]
        )
        client = make_client(recorder)

        result = client.fetch("abc")

        assert result.ready
        assert len(recorder.requests) == 101
        assert sleeps == [300.0] * 100
        client.close()

    def test_returns_the_pending_result_after_max_attempts(
        self, accepted_pseudo_response_json, sleeps
    ):
        recorder = RequestRecorder(
            [httpx.Response(202, json=accepted_pseudo_response_json)]
        )
        client = make_client(recorder)

        result = client.fetch("abc", max_attempts=2)

        assert not result.ready
        assert result.fetch_key == "fetch-abc-123"
        assert len(recorder.requests) == 2
        assert sleeps == [300.0]
        client.close()

    def test_an_error_ends_the_polling_and_carries_the_fetch_key(
        self, accepted_pseudo_response_json, sleeps
    ):
        recorder = RequestRecorder(
            [
                httpx.Response(202, json=accepted_pseudo_response_json),
                httpx.Response(410, text="expired"),
                httpx.Response(200, json={}),
            ]
        )
        client = make_client(recorder)

        with pytest.raises(DataExpiredError) as excinfo:
            client.fetch("abc")

        assert any("abc" in note for note in excinfo.value.__notes__)
        assert len(recorder.requests) == 2
        client.close()

    def test_a_network_error_ends_the_polling_too(self):
        def unreachable(request):
            raise httpx.ConnectError("connection refused", request=request)

        client = make_client(unreachable)

        with pytest.raises(httpx.ConnectError) as excinfo:
            client.fetch("abc")

        assert any("abc" in note for note in excinfo.value.__notes__)
        client.close()

    @pytest.mark.parametrize(
        "option", [{"max_attempts": 0}, {"poll_interval": -1.0}], ids=str
    )
    def test_invalid_polling_options_raise_before_sending(self, option):
        recorder = RequestRecorder([httpx.Response(200, json={})])
        client = make_client(recorder)

        with pytest.raises(ValueError):
            client.fetch("abc", **option)

        assert recorder.requests == []
        client.close()

    def test_download_options_and_start_retention_period_are_sent(
        self, completed_pseudo_response_json
    ):
        recorder = RequestRecorder(
            [httpx.Response(200, json=completed_pseudo_response_json)]
        )
        client = make_client(recorder)

        client.fetch(
            "abc",
            start_retention_period=False,
            download_options=DownloadOption.PDF_AS_PDF,
        )

        params = recorder.requests[0].url.params
        assert params["startRetentionPeriod"] == "false"
        assert params["downloadOptions"] == "PDF_AS_PDF"
        client.close()

    def test_default_return_keys_are_true(self, completed_pseudo_response_json):
        """Regression: sending returnPseudoKey=false on fetch was observed
        against the real API to make later depseudo-by-metaKey fail with
        "No such id ..." - fetch() must request the keys by default."""
        recorder = RequestRecorder(
            [httpx.Response(200, json=completed_pseudo_response_json)]
        )
        client = make_client(recorder)

        client.fetch("abc")

        assert recorder.requests[0].url.params["returnPseudoKey"] == "true"
        assert recorder.requests[0].url.params["returnEncryptKey"] == "true"
        client.close()

    def test_return_pseudo_key_can_be_overridden_to_false(
        self, completed_pseudo_response_json
    ):
        recorder = RequestRecorder(
            [httpx.Response(200, json=completed_pseudo_response_json)]
        )
        client = make_client(recorder)

        client.fetch("abc", return_pseudo_key=False)

        assert recorder.requests[0].url.params["returnPseudoKey"] == "false"
        client.close()


class TestDeanonymizeInput:
    """deanonymize() with anonymized text, a file or bytes instead of a result."""

    def test_sends_key_file_contents_and_returns_completed_result(
        self,
        completed_pseudo_response_json,
        pseudo_key_file,
        encrypt_key_file,
        tmp_path,
    ):
        recorder = RequestRecorder(
            [httpx.Response(200, json=completed_pseudo_response_json)]
        )
        client = make_client(recorder)

        src = tmp_path / "anonymized.docx"
        src.write_bytes(b"anonymized bytes")

        depseudo_paths = DepseudoPaths(
            pseudo_key_path=str(pseudo_key_file), encrypt_key_path=str(encrypt_key_file)
        )
        result = client.deanonymize(path=src, depseudo_paths=depseudo_paths)

        assert isinstance(result, DeanonymizationResult)
        assert result.ready
        body = json.loads(recorder.requests[0].content)
        assert body["pseudoKey"] == "pseudo-key-value"
        assert body["encryptKey"] == "encrypt-key-value"
        assert recorder.requests[0].url.params["outputMode"] == "json"
        client.close()

    def test_filename_override_is_sent_instead_of_local_basename(
        self, completed_pseudo_response_json, tmp_path
    ):
        """Regression: the API tracks a document by the fileName it originally
        assigned, not whatever the local file happens to be named."""
        recorder = RequestRecorder(
            [httpx.Response(200, json=completed_pseudo_response_json)]
        )
        client = make_client(recorder)

        src = tmp_path / "locally-renamed.txt"
        src.write_bytes(b"anonymized bytes")

        client.deanonymize(
            path=src, session="s-1", filename="plain_text_anonym#12345.txt"
        )

        body = json.loads(recorder.requests[0].content)
        assert body["fileName"] == "plain_text_anonym#12345.txt"
        client.close()

    def test_session_based_depseudo(self, completed_pseudo_response_json, tmp_path):
        recorder = RequestRecorder(
            [httpx.Response(200, json=completed_pseudo_response_json)]
        )
        client = make_client(recorder)

        src = tmp_path / "anonymized.docx"
        src.write_bytes(b"anonymized bytes")

        client.deanonymize(path=src, session="session-xyz")

        body = json.loads(recorder.requests[0].content)
        assert body["session"] == "session-xyz"
        assert "pseudoKey" not in body
        client.close()

    def test_text_is_restored_with_a_session(self):
        # The API ends plain-text output with a newline; it is dropped.
        recorder = RequestRecorder([restored_response("Michael kommt morgen\n")])
        client = make_client(recorder)

        restored = client.deanonymize(text="PERSON1 kommt morgen", session="sess-1")

        assert restored.extract_text().strip() == "Michael kommt morgen"
        body = json.loads(recorder.requests[0].content)
        assert body["plainText"] == "PERSON1 kommt morgen"
        assert body["fileName"] == "plain_text.txt"
        assert body["session"] == "sess-1"
        client.close()

    def test_depseudo_text_returns_the_restored_string(self):
        recorder = RequestRecorder([restored_response("Michael kommt morgen\n")])
        client = make_client(recorder)

        restored = client.depseudo_text("PERSON1 kommt morgen", session="sess-1")

        assert restored.strip() == "Michael kommt morgen"
        body = json.loads(recorder.requests[0].content)
        assert body["plainText"] == "PERSON1 kommt morgen"
        assert body["session"] == "sess-1"
        client.close()

    def test_bytes_are_restored_with_keys(self):
        recorder = RequestRecorder(
            [restored_response("restored", file_name="report.docx")]
        )
        client = make_client(recorder)

        restored = client.deanonymize(
            data=b"anonymized docx",
            filename="report_anonym.docx",
            depseudo_paths=DepseudoPaths(pseudo_key="pk", encrypt_key="ek"),
        )

        assert restored.file_name == "report.docx"
        body = json.loads(recorder.requests[0].content)
        assert base64.b64decode(body["data"]) == b"anonymized docx"
        assert body["fileName"] == "report_anonym.docx"
        assert body["pseudoKey"] == "pk"
        assert body["encryptKey"] == "ek"
        client.close()

    def test_data_expired_raises_typed_exception(self, tmp_path):
        client = make_client(lambda request: httpx.Response(410, text="expired"))
        src = tmp_path / "anonymized.docx"
        src.write_bytes(b"bytes")

        with pytest.raises(DataExpiredError):
            client.deanonymize(path=src, session="session-xyz")
        client.close()


class TestAnonymizeEndToEnd:
    def test_submit_then_fetch_then_extract(
        self, accepted_pseudo_response_json, completed_pseudo_response_json
    ):
        recorder = RequestRecorder(
            [
                httpx.Response(202, json=accepted_pseudo_response_json),
                httpx.Response(200, json=completed_pseudo_response_json),
            ]
        )
        client = make_client(recorder)

        result = client.anonymize(text="Mein Name ist Michael Huber", processing="poll")

        expected_text = base64.b64decode(
            completed_pseudo_response_json["data"]["data"]
        ).decode("utf-8")
        assert result.extract_text() == expected_text
        # first request submits the document, second fetches the result
        assert recorder.paths() == ["/document/pseudo", "/document/pseudo/fetch"]
        client.close()

    def test_document_text_comes_from_the_text_extractor(
        self, completed_pseudo_response_json
    ):
        completed_pseudo_response_json["data"].update(
            data=b64(b"anonymized docx"),
            fileName="report_anonym.docx",
            uploadName="report.docx",
        )
        recorder = RequestRecorder(
            [httpx.Response(200, json=completed_pseudo_response_json)]
        )
        client = make_client(recorder, text_extractor=FakeTextExtractor())

        result = client.anonymize(data=b"original docx", filename="report.docx")

        assert result.extract_text() == "text of report_anonym.docx"
        assert result.data == b"anonymized docx"
        assert result.file_name == "report_anonym.docx"
        body = json.loads(recorder.requests[0].content)
        assert base64.b64decode(body["data"]) == b"original docx"
        assert body["fileName"] == "report.docx"
        client.close()

    def test_document_bytes_and_keys_without_text_extraction(
        self, completed_pseudo_response_json
    ):
        completed_pseudo_response_json["data"].update(
            data=b64(b"anonymized docx"),
            fileName="report_anonym.docx",
            uploadName="report.docx",
            pseudoKey="pk",
            encryptKey="ek",
        )
        recorder = RequestRecorder(
            [httpx.Response(200, json=completed_pseudo_response_json)]
        )
        client = make_client(recorder, text_extractor=FailingTextExtractor())

        result = client.anonymize(data=b"original docx", filename="report.docx")

        assert result.data == b"anonymized docx"
        assert result.pseudo_key == "pk"
        assert result.encrypt_key == "ek"
        client.close()

    def test_text_input_never_touches_the_extractor(
        self, completed_pseudo_response_json
    ):
        recorder = RequestRecorder(
            [httpx.Response(200, json=completed_pseudo_response_json)]
        )
        client = make_client(recorder, text_extractor=FailingTextExtractor())

        result = client.anonymize(text="Mein Name ist Michael Huber")

        assert result.extract_text() == "Anonymisiertes Dokument"
        client.close()


def make_anonymized_result(response_json: dict, **data_fields) -> AnonymizationResult:
    response_json["data"].update(data_fields)
    return AnonymizationResult(PseudoResult.model_validate(response_json))


def restored_response(text: str, file_name: str = "plain_text.txt") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "success": True,
            "data": {"data": b64(text), "fileName": file_name},
        },
    )


class TestDeanonymize:
    def test_restores_text_with_the_keys_of_the_result(
        self, completed_pseudo_response_json
    ):
        anonymized = make_anonymized_result(
            completed_pseudo_response_json, pseudoKey="pk", encryptKey="ek"
        )
        recorder = RequestRecorder([restored_response("Mein Name ist Michael Huber")])
        client = make_client(recorder)

        restored = client.deanonymize(anonymized)

        assert isinstance(restored, DeanonymizationResult)
        assert restored.extract_text() == "Mein Name ist Michael Huber"
        request = recorder.requests[0]
        assert request.url.path == "/be/api-v2/document/depseudo"
        assert request.url.params["outputMode"] == "json"
        body = json.loads(request.content)
        assert body["plainText"] == "Anonymisiertes Dokument"
        assert body["fileName"] == "plain_text.txt"
        assert body["pseudoKey"] == "pk"
        assert body["encryptKey"] == "ek"
        assert "metaKey" not in body
        client.close()

    def test_falls_back_to_the_session_of_the_result(
        self, completed_pseudo_response_json
    ):
        anonymized = make_anonymized_result(
            completed_pseudo_response_json, metaKey=None, session="sess-1"
        )
        recorder = RequestRecorder([restored_response("restored")])
        client = make_client(recorder)

        client.deanonymize(anonymized)

        body = json.loads(recorder.requests[0].content)
        assert body["session"] == "sess-1"
        assert "pseudoKey" not in body
        client.close()

    def test_explicit_session_is_used_instead_of_the_keys(
        self, completed_pseudo_response_json
    ):
        anonymized = make_anonymized_result(
            completed_pseudo_response_json, pseudoKey="pk", encryptKey="ek"
        )
        recorder = RequestRecorder([restored_response("restored")])
        client = make_client(recorder)

        client.deanonymize(anonymized, session="sess-2")

        body = json.loads(recorder.requests[0].content)
        assert body["session"] == "sess-2"
        assert "pseudoKey" not in body
        assert "encryptKey" not in body
        client.close()

    def test_result_without_keys_or_session_raises_before_sending(
        self, completed_pseudo_response_json
    ):
        anonymized = make_anonymized_result(
            completed_pseudo_response_json, metaKey=None
        )
        recorder = RequestRecorder([restored_response("restored")])
        client = make_client(recorder)

        with pytest.raises(ValueError, match="no keys"):
            client.deanonymize(anonymized)

        assert recorder.requests == []
        client.close()

    def test_document_is_sent_as_data_and_text_comes_from_extractor(
        self, completed_pseudo_response_json
    ):
        anonymized = make_anonymized_result(
            completed_pseudo_response_json,
            data=b64(b"anonymized docx"),
            fileName="report_anonym.docx",
            uploadName="report.docx",
            pseudoKey="pk",
            encryptKey="ek",
        )
        recorder = RequestRecorder(
            [restored_response("restored docx", file_name="report.docx")]
        )
        client = make_client(recorder, text_extractor=FakeTextExtractor())

        restored = client.deanonymize(anonymized)

        body = json.loads(recorder.requests[0].content)
        assert base64.b64decode(body["data"]) == b"anonymized docx"
        assert "plainText" not in body
        assert body["fileName"] == "report_anonym.docx"
        assert restored.data == b"restored docx"
        assert restored.extract_text() == "text of report.docx"
        client.close()


def session_response(**fields) -> httpx.Response:
    return httpx.Response(
        200, json={"success": True, "data": {"id": "sess-1", **fields}}
    )


class TestSessionEndpoint:
    def test_add_session(self):
        recorder = RequestRecorder([session_response(expiresAt=1_700_000_000_000)])
        client = make_client(recorder)

        session = client.add_session(3600, settings_id="set-1", name="chat")

        assert isinstance(session, SessionInfo)
        assert session.id == "sess-1"
        assert session.expiresAt == 1_700_000_000_000
        request = recorder.requests[0]
        assert request.method == "POST"
        assert request.url.path == "/be/api-v2/session/add"
        assert json.loads(request.content) == {
            "name": "chat",
            "settingsId": "set-1",
            "sessionMaxAge": 3600,
        }
        client.close()

    def test_get_session(self):
        recorder = RequestRecorder([session_response()])
        client = make_client(recorder)

        session = client.get_session("sess-1")

        assert session.id == "sess-1"
        request = recorder.requests[0]
        assert request.method == "GET"
        assert request.url.path == "/be/api-v2/session/get"
        assert request.url.params["session"] == "sess-1"
        client.close()

    def test_extend_session_uses_put_and_seconds(self):
        recorder = RequestRecorder([session_response()])
        client = make_client(recorder)

        client.extend_session("sess-1", max_age_seconds=7200)

        request = recorder.requests[0]
        assert request.method == "PUT"
        assert request.url.path == "/be/api-v2/session/extend"
        assert request.url.params["session"] == "sess-1"
        assert request.url.params["sessionMaxAge"] == "7200"
        client.close()

    def test_unsuccessful_session_response_raises(self):
        client = make_client(
            lambda request: httpx.Response(
                200, json={"success": False, "message": "Session expired"}
            )
        )
        with pytest.raises(GlanosClientError, match="Session expired"):
            client.get_session("sess-1")
        client.close()

    def test_start_session_from_stored_setting_rounds_up_to_seconds(self):
        recorder = RequestRecorder([session_response()])
        client = make_client(recorder)

        session_id = client._start_session(1500, settings_id="set-1")

        assert session_id == "sess-1"
        body = json.loads(recorder.requests[0].content)
        assert recorder.requests[0].url.path == "/be/api-v2/session/add"
        assert body["sessionMaxAge"] == 2
        assert body["settingsId"] == "set-1"
        client.close()

    def test_start_session_without_stored_setting_uses_a_pseudo_request(
        self, completed_pseudo_response_json
    ):
        completed_pseudo_response_json["data"]["session"] = "sess-new"
        recorder = RequestRecorder(
            [httpx.Response(200, json=completed_pseudo_response_json)]
        )
        client = make_client(recorder)

        session_id = client._start_session(
            60_000, settings=AnonymizationSettings(mode="TYPE")
        )

        assert session_id == "sess-new"
        body = json.loads(recorder.requests[0].content)
        assert recorder.requests[0].url.path == "/be/api-v2/document/pseudo"
        assert body["sessionMaxAge"] == 60_000
        assert body["settings"]["mode"] == "TYPE"
        # only the session is needed, not the keys of the dummy text
        assert recorder.requests[0].url.params["returnPseudoKey"] == "false"
        assert recorder.requests[0].url.params["returnEncryptKey"] == "false"
        client.close()

    def test_start_session_polls_when_not_ready_in_time(
        self, accepted_pseudo_response_json, completed_pseudo_response_json, sleeps
    ):
        completed_pseudo_response_json["data"]["session"] = "sess-new"
        recorder = RequestRecorder(
            [
                httpx.Response(202, json=accepted_pseudo_response_json),
                httpx.Response(202, json=accepted_pseudo_response_json),
                httpx.Response(200, json=completed_pseudo_response_json),
            ]
        )
        client = make_client(recorder)

        session_id = client._start_session(60_000)

        assert session_id == "sess-new"
        assert recorder.paths() == [
            "/document/pseudo",
            "/document/pseudo/fetch",
            "/document/pseudo/fetch",
        ]
        assert recorder.requests[1].url.params["returnPseudoKey"] == "false"
        assert sleeps == [10.0]
        client.close()

    def test_start_session_without_returned_session_id_raises(
        self, completed_pseudo_response_json
    ):
        client = make_client(
            lambda request: httpx.Response(200, json=completed_pseudo_response_json)
        )
        with pytest.raises(GlanosClientError, match="session id"):
            client._start_session(60_000)
        client.close()

    def test_remove_session(self):
        recorder = RequestRecorder([httpx.Response(200, json={"success": True})])
        client = make_client(recorder)

        result = client.remove_session("my-session")

        assert isinstance(result, SuccessMessage)
        assert result.success is True
        assert recorder.requests[0].method == "PUT"
        assert recorder.requests[0].url.params["session"] == "my-session"
        client.close()

    def test_remove_session_with_wrapped_envelope(self):
        """Regression: the deployed API wraps some /session responses in
        {"success": ..., "data": {...}} even where the spec shows a bare
        schema."""
        payload = {"success": True, "data": {"success": True}}
        recorder = RequestRecorder([httpx.Response(200, json=payload)])
        client = make_client(recorder)

        result = client.remove_session("my-session")

        assert result.success is True
        client.close()


def setting_response(**fields) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "success": True,
            "data": {"id": "set-1", "name": "Legal DE", "mode": "TYPE", **fields},
        },
    )


class TestSettingsEndpoints:
    def test_add_setting(self):
        recorder = RequestRecorder([setting_response(mode="TYPE_PRESERVING")])
        client = make_client(recorder)

        stored = client.add_setting(
            AnonymizationSettings(name="Legal DE", mode="TYPE_PRESERVING")
        )

        assert stored.id == "set-1"
        request = recorder.requests[0]
        assert request.method == "POST"
        assert request.url.path == "/be/api-v2/setting/add"
        assert request.url.params["applyDefaults"] == "true"
        body = json.loads(request.content)
        assert body["name"] == "Legal DE"
        assert body["mode"] == "TYPE_PRESERVING"
        assert "id" not in body
        client.close()

    def test_apply_defaults_can_be_turned_off(self):
        recorder = RequestRecorder([setting_response()])
        client = make_client(recorder)

        client.add_setting(AnonymizationSettings(name="Legal DE"), apply_defaults=False)

        assert recorder.requests[0].url.params["applyDefaults"] == "false"
        client.close()

    @pytest.mark.parametrize(
        "settings,message",
        [
            (AnonymizationSettings(id="set-1", name="Legal DE"), "update_setting"),
            (AnonymizationSettings(), "name"),
        ],
    )
    def test_add_setting_rejects_invalid_settings_before_sending(
        self, settings, message
    ):
        recorder = RequestRecorder([setting_response()])
        client = make_client(recorder)

        with pytest.raises(ValueError, match=message):
            client.add_setting(settings)

        assert recorder.requests == []
        client.close()

    def test_update_setting_sends_the_complete_setting(self):
        recorder = RequestRecorder([setting_response(excludeFields=["dates"])])
        client = make_client(recorder)

        client.update_setting(
            AnonymizationSettings(id="set-1", name="Legal DE", excludeFields=["dates"])
        )

        request = recorder.requests[0]
        assert request.url.path == "/be/api-v2/setting/update"
        body = json.loads(request.content)
        assert body["id"] == "set-1"
        assert body["name"] == "Legal DE"
        assert body["excludeFields"] == ["dates"]
        client.close()

    def test_update_setting_requires_an_id(self):
        recorder = RequestRecorder([setting_response()])
        client = make_client(recorder)

        with pytest.raises(ValueError, match="id"):
            client.update_setting(AnonymizationSettings(name="Legal DE"))

        assert recorder.requests == []
        client.close()

    def test_update_of_unknown_setting_raises_setting_not_found(self):
        client = make_client(
            lambda request: httpx.Response(
                400, json={"success": False, "message": "No such settings: set-x"}
            )
        )
        with pytest.raises(SettingNotFoundError):
            client.update_setting(AnonymizationSettings(id="set-x", name="X"))
        client.close()

    def test_get_setting(self):
        recorder = RequestRecorder([setting_response()])
        client = make_client(recorder)

        setting = client.get_setting("set-1")

        assert setting.id == "set-1"
        assert setting.name == "Legal DE"
        request = recorder.requests[0]
        assert request.method == "GET"
        assert request.url.path == "/be/api-v2/setting/get"
        assert request.url.params["setting"] == "set-1"
        client.close()

    def test_list_settings(self):
        payload = {
            "success": True,
            "data": {
                "limit": 20,
                "offset": 40,
                "total": 41,
                "list": [{"id": "set-1", "name": "Legal DE", "mode": "TYPE"}],
            },
        }
        recorder = RequestRecorder([httpx.Response(200, json=payload)])
        client = make_client(recorder)

        page = client.list_settings(
            query="legal",
            creator=["alice", "bob"],
            sort=SettingsSort.NAME,
            sort_order=SortOrder.ASC,
            limit=20,
            offset=40,
        )

        assert page.total == 41
        assert [item.id for item in page.list] == ["set-1"]
        params = recorder.requests[0].url.params
        assert recorder.requests[0].url.path == "/be/api-v2/setting/list"
        assert params["query"] == "legal"
        assert params["creator"] == "alice,bob"
        assert params["sort"] == "name"
        assert params["sortOrder"] == "asc"
        assert params["limit"] == "20"
        assert params["offset"] == "40"
        client.close()

    def test_remove_setting(self):
        recorder = RequestRecorder([httpx.Response(200, json={"success": True})])
        client = make_client(recorder)

        result = client.remove_setting("set-1")

        assert result.success is True
        request = recorder.requests[0]
        assert request.method == "PUT"
        assert request.url.path == "/be/api-v2/setting/remove"
        assert request.url.params["setting"] == "set-1"
        client.close()
