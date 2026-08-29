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
    AnonymizationConfig,
    DataExpiredError,
    DepseudoPaths,
    DownloadOption,
    FetchKeyData,
    FetchTimeoutError,
    GlanosClient,
    InnerResponseData,
    PseudoResult,
    QuotaExceededError,
    RateLimitedError,
    SuccessMessage,
)

pytestmark = pytest.mark.integration


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


class TestPseudoSync:
    def test_completed_result_and_request_shape(self, completed_pseudo_response_json):
        recorder = RequestRecorder(
            [httpx.Response(200, json=completed_pseudo_response_json)]
        )
        client = make_client(recorder)

        result = client.pseudo_sync(text="Mein Name ist Michael Huber")

        assert isinstance(result.data, InnerResponseData)
        request = recorder.requests[0]
        assert request.method == "POST"
        assert str(request.url).startswith(
            "https://glanos.example.test/be/api-v2/document/pseudo?"
        )
        assert request.url.params["token"] == "tok-123"
        assert (
            request.url.params["async"] == "true"
        )  # default AnonymizationConfig.a_sync=True

        body = json.loads(request.content)
        assert body["plainText"] == "Mein Name ist Michael Huber"
        assert body["fileName"] == "plain_text.txt"
        client.close()

    def test_accepted_result_for_async_request(self, accepted_pseudo_response_json):
        recorder = RequestRecorder(
            [httpx.Response(202, json=accepted_pseudo_response_json)]
        )
        client = make_client(recorder)

        result = client.pseudo_sync(text="hi")

        assert isinstance(result.data, FetchKeyData)
        client.close()

    def test_quota_exceeded_raises_typed_exception(self):
        client = make_client(lambda request: httpx.Response(402, text="quota exceeded"))
        with pytest.raises(QuotaExceededError):
            client.pseudo_sync(text="hi")
        client.close()

    def test_rate_limited_raises_typed_exception(self):
        client = make_client(lambda request: httpx.Response(429, text="slow down"))
        with pytest.raises(RateLimitedError):
            client.pseudo_sync(text="hi")
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

        client.pseudo_sync(path=src)

        body = json.loads(recorder.requests[0].content)
        assert body["fileName"] == "input.txt"
        assert "data" in body
        assert "plainText" not in body
        client.close()


class TestFetch:
    def test_polls_until_completed(
        self, accepted_pseudo_response_json, completed_pseudo_response_json
    ):
        recorder = RequestRecorder(
            [
                httpx.Response(202, json=accepted_pseudo_response_json),
                httpx.Response(202, json=accepted_pseudo_response_json),
                httpx.Response(200, json=completed_pseudo_response_json),
            ]
        )
        client = make_client(recorder)
        pending = PseudoResult(success=True, data=FetchKeyData(fetchKey="abc"))

        result = client.fetch(pending, max_retries=5, poll_interval=(0.0, 0.0))

        assert isinstance(result.data, InnerResponseData)
        assert len(recorder.requests) == 3
        assert recorder.requests[0].url.params["fetchKey"] == "abc"
        client.close()

    def test_already_completed_result_short_circuits_without_a_request(
        self, completed_pseudo_response_json
    ):
        recorder = RequestRecorder(
            [httpx.Response(200, json=completed_pseudo_response_json)]
        )
        client = make_client(recorder)
        already_done = PseudoResult.model_validate(completed_pseudo_response_json)

        result = client.fetch(already_done)

        assert result is already_done
        assert recorder.requests == []
        client.close()

    def test_raises_fetch_timeout_after_max_retries(
        self, accepted_pseudo_response_json
    ):
        client = make_client(
            lambda request: httpx.Response(202, json=accepted_pseudo_response_json)
        )
        pending = PseudoResult(success=True, data=FetchKeyData(fetchKey="abc"))

        with pytest.raises(FetchTimeoutError):
            client.fetch(pending, max_retries=2, poll_interval=(0.0, 0.0))
        client.close()

    def test_download_options_and_start_retention_period_are_sent(
        self, completed_pseudo_response_json
    ):
        recorder = RequestRecorder(
            [httpx.Response(200, json=completed_pseudo_response_json)]
        )
        client = make_client(recorder)
        pending = PseudoResult(success=True, data=FetchKeyData(fetchKey="abc"))

        client.fetch(
            pending,
            start_retention_period=False,
            download_options=DownloadOption.PDF_AS_PDF,
        )

        params = recorder.requests[0].url.params
        assert params["startRetentionPeriod"] == "false"
        assert params["downloadOptions"] == "PDF_AS_PDF"
        client.close()

    def test_default_return_pseudo_key_is_true_regardless_of_config(
        self, completed_pseudo_response_json
    ):
        """Regression: fetch()'s returnPseudoKey used to be sourced from
        AnonymizationConfig.return_pseudo_key (default False, matching POST
        /pseudo's default), but GET /fetch's own documented default is True.
        Sending returnPseudoKey=false on fetch was observed against the real
        API to make later depseudo-by-metaKey fail with "No such id ..." -
        fetch() must default to true on its own, independent of config."""
        recorder = RequestRecorder(
            [httpx.Response(200, json=completed_pseudo_response_json)]
        )
        # Config explicitly has return_pseudo_key at its own (False) default.
        client = make_client(recorder, config=AnonymizationConfig(a_sync=True))
        pending = PseudoResult(success=True, data=FetchKeyData(fetchKey="abc"))

        client.fetch(pending)

        assert recorder.requests[0].url.params["returnPseudoKey"] == "true"
        client.close()

    def test_return_pseudo_key_can_be_overridden_to_false(
        self, completed_pseudo_response_json
    ):
        recorder = RequestRecorder(
            [httpx.Response(200, json=completed_pseudo_response_json)]
        )
        client = make_client(recorder)
        pending = PseudoResult(success=True, data=FetchKeyData(fetchKey="abc"))

        client.fetch(pending, return_pseudo_key=False)

        assert recorder.requests[0].url.params["returnPseudoKey"] == "false"
        client.close()


class TestDepseudo:
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
        result = client.depseudo(src, depseudo_paths)

        assert isinstance(result.data, InnerResponseData)
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

        client.depseudo(src, session="s-1", filename="plain_text_anonym#12345.txt")

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

        client.depseudo(src, session="session-xyz")

        body = json.loads(recorder.requests[0].content)
        assert body["session"] == "session-xyz"
        assert "pseudoKey" not in body
        client.close()

    def test_data_expired_raises_typed_exception(self, tmp_path):
        client = make_client(lambda request: httpx.Response(410, text="expired"))
        src = tmp_path / "anonymized.docx"
        src.write_bytes(b"bytes")

        with pytest.raises(DataExpiredError):
            client.depseudo(src, session="session-xyz")
        client.close()


class TestAnonymizeEndToEnd:
    def test_pseudo_then_fetch_then_extract(
        self, accepted_pseudo_response_json, completed_pseudo_response_json
    ):
        recorder = RequestRecorder(
            [
                httpx.Response(202, json=accepted_pseudo_response_json),
                httpx.Response(200, json=completed_pseudo_response_json),
            ]
        )
        client = make_client(recorder)

        result = client.anonymize(text="Mein Name ist Michael Huber")

        expected_text = base64.b64decode(
            completed_pseudo_response_json["data"]["data"]
        ).decode("utf-8")
        assert result.text == expected_text
        # first request submits the document, second fetches the result
        assert recorder.requests[0].url.path == "/be/api-v2/document/pseudo"
        assert recorder.requests[1].url.path == "/be/api-v2/document/pseudo/fetch"
        client.close()


class TestSessionEndpoint:
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
