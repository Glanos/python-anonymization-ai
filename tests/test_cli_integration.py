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

    def test_no_wait_prints_raw_json_without_writing_file(
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
                "--processing",
                "no_wait",
                "--outputFile",
                str(out),
            ]
        )

        assert not out.exists()
        assert "fetchKey" in capsys.readouterr().out
        assert recorder.requests[0].url.params["async"] == "true"

    def test_poll_prints_the_fetch_key_when_out_of_attempts(
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
                "--file",
                __file__,
                "--pollInterval",
                "0",
                "--maxAttempts",
                "2",
                "--outputFile",
                str(out),
            ]
        )

        assert not out.exists()
        captured = capsys.readouterr()
        assert "fetch-abc-123" in captured.out
        assert "fetch --fetchKey fetch-abc-123" in captured.err
        # one submission and two checks
        assert len(recorder.requests) == 3

    def test_assignee_submits_for_review_and_prints_the_fetch_key(
        self, patch_http, capsys, accepted_pseudo_response_json
    ):
        recorder = RequestRecorder(
            [httpx.Response(202, json=accepted_pseudo_response_json)]
        )
        patch_http(recorder)

        cli.main(
            [
                "pseudo",
                "--token",
                "tok",
                "--baseUrl",
                "https://x.test",
                "--file",
                __file__,
                "--assignee",
                "anna@example.com",
            ]
        )

        assert "fetch-abc-123" in capsys.readouterr().out
        assert len(recorder.requests) == 1
        assert recorder.requests[0].url.params["async"] == "true"
        assert json.loads(recorder.requests[0].content)["assignee"] == (
            "anna@example.com"
        )

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

    def test_settings_flags_are_sent_as_inline_settings(
        self, patch_http, tmp_path, completed_pseudo_response_json
    ):
        recorder = RequestRecorder(
            [httpx.Response(200, json=completed_pseudo_response_json)]
        )
        patch_http(recorder)

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
                "--mode",
                "TYPE_PRESERVING",
                "--excludeFields",
                "dates",
            ]
        )

        body = json.loads(recorder.requests[0].content)
        assert body["settings"]["mode"] == "TYPE_PRESERVING"
        assert body["settings"]["excludeFields"] == ["dates"]
        assert "settingsId" not in body

    def test_settings_flags_together_with_settings_id_fail_without_a_request(
        self, patch_http, capsys, tmp_path
    ):
        recorder = RequestRecorder([httpx.Response(200, json={})])
        patch_http(recorder)

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
                    "--mode",
                    "XXX",
                    "--settingsId",
                    "set-1",
                ]
            )

        assert exc_info.value.code == 1
        assert "settings_id" in capsys.readouterr().err
        assert recorder.requests == []

    def test_without_settings_flags_no_settings_are_sent(
        self, patch_http, tmp_path, completed_pseudo_response_json
    ):
        recorder = RequestRecorder(
            [httpx.Response(200, json=completed_pseudo_response_json)]
        )
        patch_http(recorder)

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

        assert "settings" not in json.loads(recorder.requests[0].content)

    def test_sync_pseudo_requests_and_writes_both_keys_by_default(
        self, patch_http, tmp_path, completed_pseudo_response_json
    ):
        completed_pseudo_response_json["data"].update(
            pseudoKey="pk-value", encryptKey="ek-value"
        )
        recorder = RequestRecorder(
            [httpx.Response(200, json=completed_pseudo_response_json)]
        )
        patch_http(recorder)
        out = tmp_path / "anonymized.txt"

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
                str(out),
            ]
        )

        params = recorder.requests[0].url.params
        assert params["returnPseudoKey"] == "true"
        assert params["returnEncryptKey"] == "true"
        assert (tmp_path / "anonymized.pkey").read_text() == "pk-value"
        assert (tmp_path / "anonymized.ekey").read_text() == "ek-value"


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

    def test_fetch_polls_with_the_given_interval_and_attempts(
        self, patch_http, monkeypatch, capsys, tmp_path, accepted_pseudo_response_json
    ):
        sleeps: list[float] = []
        monkeypatch.setattr("glanos_anonymization.core.time.sleep", sleeps.append)
        recorder = RequestRecorder(
            [httpx.Response(202, json=accepted_pseudo_response_json)]
        )
        patch_http(recorder)

        cli.main(
            [
                "fetch",
                "--token",
                "tok",
                "--baseUrl",
                "https://x.test",
                "--fetchKey",
                "fetch-abc-123",
                "--pollInterval",
                "60",
                "--maxAttempts",
                "3",
                "--outputFile",
                str(tmp_path / "out.txt"),
            ]
        )

        assert len(recorder.requests) == 3
        assert sleeps == [60.0, 60.0]
        assert "fetch-abc-123" in capsys.readouterr().out


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


class TestSettingUpdateCommand:
    def test_reads_the_setting_and_sends_it_back_complete(self, patch_http, capsys):
        stored = {
            "id": "set-1",
            "name": "Legal DE",
            "mode": "TYPE_PRESERVING",
            "excludeFields": ["dates"],
        }
        recorder = RequestRecorder(
            [
                httpx.Response(200, json={"success": True, "data": stored}),
                httpx.Response(
                    200,
                    json={
                        "success": True,
                        "data": {**stored, "excludeFields": ["locations"]},
                    },
                ),
            ]
        )
        patch_http(recorder)

        cli.main(
            [
                "setting-update",
                "--token",
                "tok",
                "--baseUrl",
                "https://x.test",
                "--setting",
                "set-1",
                "--excludeFields",
                "locations",
            ]
        )

        get_request, update_request = recorder.requests
        assert get_request.url.path == "/be/api-v2/setting/get"
        assert update_request.url.path == "/be/api-v2/setting/update"
        body = json.loads(update_request.content)
        assert body["id"] == "set-1"
        assert body["name"] == "Legal DE"
        assert body["mode"] == "TYPE_PRESERVING"
        assert body["excludeFields"] == ["locations"]
        assert json.loads(capsys.readouterr().out)["excludeFields"] == ["locations"]


class TestSessionAddCommand:
    def test_prints_the_created_session(self, patch_http, capsys):
        recorder = RequestRecorder(
            [httpx.Response(200, json={"success": True, "data": {"id": "sess-1"}})]
        )
        patch_http(recorder)

        cli.main(
            [
                "session-add",
                "--token",
                "tok",
                "--baseUrl",
                "https://x.test",
                "--settingsId",
                "set-1",
                "--maxAgeSeconds",
                "3600",
            ]
        )

        body = json.loads(recorder.requests[0].content)
        assert body == {"settingsId": "set-1", "sessionMaxAge": 3600}
        assert json.loads(capsys.readouterr().out)["id"] == "sess-1"
