"""Integration tests for glanos_anonymization.cli.

Runs cli.main()/exec_command() end-to-end (argv -> GlanosClient -> printed
output / files), with the HTTP layer intercepted globally via monkeypatching
httpx.Client.request - GlanosClient itself is constructed exactly as the real
CLI would build it (cli does not expose a way to inject a transport).
"""

import json
import httpx
import pytest
from glanos_anonymization import cli

pytestmark = pytest.mark.integration


class RequestRecorder:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]


@pytest.fixture
def patch_http(monkeypatch):
    """Replace httpx.Client.request class-wide with a fake that never touches
    the network. httpx normally attaches the originating request to a response
    itself (needed by response.raise_for_status()); since we bypass that
    machinery entirely, we attach it here so responses behave like real ones.
    """

    def _patch(handler):
        def fake_request(self, method, url, **kwargs):
            request = httpx.Request(
                method,
                url,
                params=kwargs.get("params"),
                content=kwargs.get("content"),
                headers=kwargs.get("headers"),
            )
            response = handler(request)
            response.request = request
            return response

        monkeypatch.setattr(httpx.Client, "request", fake_request)

    return _patch


class TestPseudoCommand:
    def test_writes_output_file_and_prints_meta_key(
        self, patch_http, capsys, tmp_path, completed_pseudo_response_json
    ):
        recorder = RequestRecorder(
            [httpx.Response(200, json=completed_pseudo_response_json)]
        )
        patch_http(recorder)
        out = tmp_path / "out.txt"

        cli.main(
            [
                "pseudo",
                "--token",
                "tok",
                "--baseUrl",
                "https://x.test",
                "--text",
                "Mein Name ist Michael Huber",
                "--outputFile",
                str(out),
            ]
        )

        assert out.exists()
        assert "metaKey: meta-123" in capsys.readouterr().out

        body = json.loads(recorder.requests[0].content)
        assert body["plainText"] == "Mein Name ist Michael Huber"

    def test_file_input(self, patch_http, tmp_path, completed_pseudo_response_json):
        recorder = RequestRecorder(
            [httpx.Response(200, json=completed_pseudo_response_json)]
        )
        patch_http(recorder)

        src = tmp_path / "input.txt"
        src.write_text("hello")
        out = tmp_path / "out.txt"

        cli.main(
            [
                "pseudo",
                "--token",
                "tok",
                "--baseUrl",
                "https://x.test",
                "--file",
                str(src),
                "--outputFile",
                str(out),
            ]
        )

        body = json.loads(recorder.requests[0].content)
        assert body["fileName"] == "input.txt"

    def test_async_prints_raw_json_without_writing_file(
        self, patch_http, capsys, tmp_path, accepted_pseudo_response_json
    ):
        recorder = RequestRecorder(
            [httpx.Response(202, json=accepted_pseudo_response_json)]
        )
        patch_http(recorder)
        out = tmp_path / "out.txt"

        cli.main(
            [
                "pseudo",
                "--token",
                "tok",
                "--baseUrl",
                "https://x.test",
                "--text",
                "hi",
                "--async",
                "--outputFile",
                str(out),
            ]
        )

        assert not out.exists()
        assert "fetchKey" in capsys.readouterr().out

    def test_quota_exceeded_exits_with_error_on_stderr(
        self, patch_http, capsys, tmp_path
    ):
        patch_http(lambda request: httpx.Response(402, text="quota exceeded"))

        with pytest.raises(SystemExit) as exc_info:
            cli.main(
                [
                    "pseudo",
                    "--token",
                    "tok",
                    "--baseUrl",
                    "https://x.test",
                    "--text",
                    "hi",
                    "--outputFile",
                    str(tmp_path / "out.txt"),
                ]
            )

        assert exc_info.value.code == 1
        stderr = capsys.readouterr().err
        assert "quota exceeded" in stderr.lower() or "402" in stderr


class TestFetchCommand:
    def test_fetch_writes_output_file(
        self, patch_http, tmp_path, completed_pseudo_response_json
    ):
        recorder = RequestRecorder(
            [httpx.Response(200, json=completed_pseudo_response_json)]
        )
        patch_http(recorder)
        out = tmp_path / "out.txt"

        cli.main(
            [
                "fetch",
                "--token",
                "tok",
                "--baseUrl",
                "https://x.test",
                "--fetchKey",
                "fk-1",
                "--outputFile",
                str(out),
            ]
        )

        assert out.exists()
        assert recorder.requests[0].url.params["fetchKey"] == "fk-1"


class TestDepseudoCommand:
    def test_meta_key_flow(self, patch_http, tmp_path, completed_pseudo_response_json):
        recorder = RequestRecorder(
            [httpx.Response(200, json=completed_pseudo_response_json)]
        )
        patch_http(recorder)

        src = tmp_path / "anonymized.txt"
        src.write_bytes(b"anonymized bytes")
        out = tmp_path / "out.txt"

        cli.main(
            [
                "depseudo",
                "--token",
                "tok",
                "--baseUrl",
                "https://x.test",
                "--file",
                str(src),
                "--metaKey",
                "mk-1",
                "--outputFile",
                str(out),
            ]
        )

        assert out.exists()
        body = json.loads(recorder.requests[0].content)
        assert body["metaKey"] == "mk-1"

    def test_key_files_flow(
        self,
        patch_http,
        tmp_path,
        pseudo_key_file,
        encrypt_key_file,
        completed_pseudo_response_json,
    ):
        recorder = RequestRecorder(
            [httpx.Response(200, json=completed_pseudo_response_json)]
        )
        patch_http(recorder)

        src = tmp_path / "anonymized.txt"
        src.write_bytes(b"anonymized bytes")
        out = tmp_path / "out.txt"

        cli.main(
            [
                "depseudo",
                "--token",
                "tok",
                "--baseUrl",
                "https://x.test",
                "--file",
                str(src),
                "--pseudoKeyFile",
                str(pseudo_key_file),
                "--encryptKeyFile",
                str(encrypt_key_file),
                "--outputFile",
                str(out),
            ]
        )

        body = json.loads(recorder.requests[0].content)
        assert body["pseudoKey"] == "pseudo-key-value"
        assert body["encryptKey"] == "encrypt-key-value"

    def test_missing_key_file_exits_without_a_network_call(
        self, patch_http, tmp_path, capsys
    ):
        recorder = RequestRecorder([httpx.Response(200, json={})])
        patch_http(recorder)

        src = tmp_path / "anonymized.txt"
        src.write_bytes(b"bytes")

        with pytest.raises(SystemExit) as exc_info:
            cli.main(
                [
                    "depseudo",
                    "--token",
                    "tok",
                    "--baseUrl",
                    "https://x.test",
                    "--file",
                    str(src),
                    "--pseudoKeyFile",
                    str(tmp_path / "does-not-exist.pkey"),
                    "--encryptKeyFile",
                    str(tmp_path / "also-missing.ekey"),
                    "--outputFile",
                    str(tmp_path / "out.txt"),
                ]
            )

        assert exc_info.value.code == 1
        assert recorder.requests == []
        assert "does not exist" in capsys.readouterr().err


class TestSessionRemoveCommand:
    def test_session_remove_prints_success(self, patch_http, capsys):
        patch_http(lambda request: httpx.Response(200, json={"success": True}))

        cli.main(
            [
                "session-remove",
                "--token",
                "tok",
                "--baseUrl",
                "https://x.test",
                "--session",
                "sess-1",
            ]
        )

        out = json.loads(capsys.readouterr().out)
        assert out["success"] is True
