"""Tests for AsyncGlanosClient.

Both clients share request building and response parsing, so the core test
runs every operation on GlanosClient and on AsyncGlanosClient against the same
fake API (httpx.MockTransport) and asserts identical requests and results.
"""

import asyncio
import base64
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from glanos_anonymization import core
from glanos_anonymization.core import (
    AnonymizationResult,
    AnonymizationSettings,
    AsyncGlanosClient,
    DepseudoPaths,
    GlanosClient,
    PseudoResult,
    SettingNotFoundError,
)

pytestmark = pytest.mark.integration

TOKEN = "tok-123"
BASE_URL = "https://glanos.example.test"


def b64(content: bytes) -> str:
    return base64.b64encode(content).decode("ascii")


def completed(**fields) -> httpx.Response:
    data = {
        "data": b64(b"PERSON1 was here\n"),
        "fileName": "plain_text_anonym.txt",
        "uploadName": "plain_text.txt",
        "pseudoKey": "pk",
        "encryptKey": "ek",
        **fields,
    }
    return httpx.Response(200, json={"success": True, "data": data})


def accepted() -> httpx.Response:
    return httpx.Response(202, json={"success": True, "data": {"fetchKey": "fk-1"}})


def session(**fields) -> httpx.Response:
    return httpx.Response(200, json={"success": True, "data": {"id": "s1", **fields}})


def setting(**fields) -> httpx.Response:
    data = {"id": "set-1", "name": "Legal DE", "mode": "TYPE", **fields}
    return httpx.Response(200, json={"success": True, "data": data})


def success() -> httpx.Response:
    return httpx.Response(200, json={"success": True, "data": {"success": True}})


ANONYMIZED = AnonymizationResult(
    PseudoResult.model_validate(
        {
            "success": True,
            "data": {
                "data": b64(b"PERSON1 was here\n"),
                "fileName": "plain_text.txt",
                "uploadName": "plain_text.txt",
                "pseudoKey": "pk",
                "encryptKey": "ek",
            },
        }
    )
)


class Recorder:
    def __init__(self, responses: list[httpx.Response]) -> None:
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.responses.pop(0)


# (name, call, responses): call works on both clients - on the async client it
# returns a coroutine.
OPERATIONS: list[tuple[str, Callable[[Any], Any], Callable[[], list]]] = [
    (
        "anonymize text with options",
        lambda c: c.anonymize(
            text="Michael was here",
            settings=AnonymizationSettings(mode="XXX", excludeFields=["dates"]),
            language="de",
            meta_retention_ms=60_000,
        ),
        lambda: [completed()],
    ),
    (
        "anonymize sync, not ready in time",
        lambda c: c.anonymize(text="Michael", return_pseudo_key=False),
        lambda: [accepted()],
    ),
    (
        "anonymize a document with polling",
        lambda c: c.anonymize(data=b"docx", filename="a.docx"),
        lambda: [accepted(), accepted(), completed(fileName="a_anonym.docx")],
    ),
    (
        "anonymize with polling, out of attempts",
        lambda c: c.anonymize(text="Michael", processing="poll", max_attempts=2),
        lambda: [accepted()] * 3,
    ),
    (
        "anonymize no_wait",
        lambda c: c.anonymize(data=b"docx", filename="a.docx", processing="no_wait"),
        lambda: [accepted()],
    ),
    (
        "anonymize_with_review",
        lambda c: c.anonymize_with_review(
            data=b"docx", filename="a.docx", assigned_group="legal"
        ),
        lambda: [accepted()],
    ),
    (
        "fetch by fetch key",
        lambda c: c.fetch("fk-1", poll_interval=1, max_attempts=3),
        lambda: [accepted(), completed()],
    ),
    (
        "deanonymize text with keys",
        lambda c: c.deanonymize(
            text="PERSON1",
            depseudo_paths=DepseudoPaths(pseudo_key="pk", encrypt_key="ek"),
        ),
        lambda: [completed()],
    ),
    (
        "deanonymize a result",
        lambda c: c.deanonymize(ANONYMIZED),
        lambda: [completed(data=b64(b"Michael was here\n"))],
    ),
    (
        "deanonymize text with a session",
        lambda c: c.deanonymize(text="PERSON1", session="s1"),
        lambda: [completed(data=b64(b"Michael\n"))],
    ),
    (
        "depseudo_text",
        lambda c: c.depseudo_text("PERSON1", session="s1"),
        lambda: [completed(data=b64(b"Michael\n"))],
    ),
    (
        "add_session",
        lambda c: c.add_session(3600, settings_id="set-1", share_users=["u"]),
        lambda: [session(expiresAt=1)],
    ),
    ("get_session", lambda c: c.get_session("s1"), lambda: [session()]),
    ("extend_session", lambda c: c.extend_session("s1", 7200), lambda: [session()]),
    ("remove_session", lambda c: c.remove_session("s1"), lambda: [success()]),
    (
        "_start_session from a setting",
        lambda c: c._start_session(90_500, settings_id="set-1"),
        lambda: [session()],
    ),
    (
        "_start_session from inline settings",
        lambda c: c._start_session(60_000, settings=AnonymizationSettings()),
        lambda: [completed(session="s-new")],
    ),
    (
        "_start_session not ready in time",
        lambda c: c._start_session(60_000),
        lambda: [accepted(), completed(session="s-new")],
    ),
    ("get_setting", lambda c: c.get_setting("set-1"), lambda: [setting()]),
    (
        "list_settings",
        lambda c: c.list_settings(
            query="legal", creator=["a", "b"], sort="name", limit=5
        ),
        lambda: [
            httpx.Response(
                200, json={"success": True, "data": {"list": [], "total": 0}}
            )
        ],
    ),
    (
        "add_setting",
        lambda c: c.add_setting(AnonymizationSettings(name="Legal DE")),
        lambda: [setting()],
    ),
    (
        "update_setting",
        lambda c: c.update_setting(AnonymizationSettings(id="set-1", name="X")),
        lambda: [setting(name="X")],
    ),
    ("remove_setting", lambda c: c.remove_setting("set-1"), lambda: [success()]),
]


def run_sync(call, responses) -> tuple[Recorder, Any]:
    recorder = Recorder(responses)
    with GlanosClient(
        token=TOKEN, base_url=BASE_URL, transport=httpx.MockTransport(recorder)
    ) as client:
        return recorder, call(client)


def run_async(call, responses) -> tuple[Recorder, Any]:
    recorder = Recorder(responses)

    async def run():
        async with AsyncGlanosClient(
            token=TOKEN, base_url=BASE_URL, transport=httpx.MockTransport(recorder)
        ) as client:
            return await call(client)

    return recorder, asyncio.run(run())


def comparable(result: Any) -> Any:
    if isinstance(result, AnonymizationResult):
        return type(result), result.pseudo_result
    return result


@pytest.fixture
def no_sleep(monkeypatch) -> None:
    async def no_async_sleep(seconds):
        pass

    monkeypatch.setattr(core.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(core.asyncio, "sleep", no_async_sleep)


@pytest.mark.parametrize(
    "call, responses", [op[1:] for op in OPERATIONS], ids=[op[0] for op in OPERATIONS]
)
def test_async_client_sends_and_returns_the_same_as_the_sync_client(
    call, responses, no_sleep
):
    sync_recorder, sync_result = run_sync(call, responses())
    async_recorder, async_result = run_async(call, responses())

    assert sync_recorder.requests
    assert [
        (r.method, str(r.url), r.headers.get("content-type"), r.content)
        for r in async_recorder.requests
    ] == [
        (r.method, str(r.url), r.headers.get("content-type"), r.content)
        for r in sync_recorder.requests
    ]
    assert comparable(async_result) == comparable(sync_result)


class TestAsyncSpecifics:
    def test_fetch_polls_with_asyncio_sleep(self, monkeypatch):
        def blocking_sleep(seconds):
            raise AssertionError("time.sleep would block the event loop")

        slept: list[float] = []

        async def fake_sleep(seconds):
            slept.append(seconds)

        monkeypatch.setattr(core.time, "sleep", blocking_sleep)
        monkeypatch.setattr(core.asyncio, "sleep", fake_sleep)
        recorder, result = run_async(
            lambda c: c.anonymize(text="hi", processing="poll"),
            [accepted(), accepted(), completed()],
        )

        assert len(recorder.requests) == 3
        assert slept == [10.0]
        assert result.extract_text().strip() == "PERSON1 was here"

    def test_fetch_returns_the_pending_result_after_max_attempts(self, no_sleep):
        recorder, result = run_async(
            lambda c: c.fetch("fk-1", max_attempts=2), [accepted()] * 2
        )

        assert not result.ready
        assert result.fetch_key == "fk-1"
        assert len(recorder.requests) == 2

    def test_an_error_while_polling_carries_the_fetch_key(self, no_sleep):
        with pytest.raises(core.ServerError) as excinfo:
            run_async(
                lambda c: c.fetch("fk-1"),
                [accepted(), httpx.Response(503, text="restarting")],
            )

        assert any("fk-1" in note for note in excinfo.value.__notes__)

    def test_update_of_a_missing_setting_raises_setting_not_found(self):
        response = httpx.Response(
            400, json={"success": False, "message": "No such settings: set-9"}
        )

        with pytest.raises(SettingNotFoundError):
            run_async(
                lambda c: c.update_setting(AnonymizationSettings(id="set-9")),
                [response],
            )

    @pytest.mark.parametrize(
        "call",
        [
            lambda c: c.anonymize(
                text="hi", processing="no_wait", return_pseudo_key=True
            ),
            lambda c: c.anonymize(text="hi", processing="sync", max_attempts=3),
            lambda c: c.anonymize_with_review(text="hi"),
            lambda c: c.deanonymize(text="PERSON1"),
            lambda c: c.fetch("fk-1", max_attempts=0),
        ],
        ids=["no_wait keys", "sync polling", "no reviewer", "no keys", "0 attempts"],
    )
    def test_argument_errors_are_raised_before_sending(self, call):
        with pytest.raises(ValueError):
            run_async(call, [])

    def test_context_manager_closes_the_http_client(self):
        async def run():
            async with AsyncGlanosClient(token=TOKEN, base_url=BASE_URL) as client:
                pass
            return client

        client = asyncio.run(run())

        assert client.client is None

    def test_aclose_is_idempotent(self):
        async def run():
            client = AsyncGlanosClient(token=TOKEN, base_url=BASE_URL)
            await client.aclose()
            await client.aclose()

        asyncio.run(run())

    def test_is_exported_from_the_package(self):
        import glanos_anonymization

        assert glanos_anonymization.AsyncGlanosClient is AsyncGlanosClient
