"""Tests for GlanosAnonymizationTools against a fake API (httpx.MockTransport).

The tool methods are called directly, as an agent framework would call them;
`definitions()` and the methods do not need LangChain.
"""

import base64
import json

import httpx
import pytest

from glanos_anonymization import GlanosAnonymizationTools, GlanosClient

pytestmark = pytest.mark.integration


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


class Recorder:
    def __init__(self, responses: list[httpx.Response]) -> None:
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]


@pytest.fixture
def make_tools(monkeypatch):
    monkeypatch.setattr("glanos_anonymization.core.time.sleep", lambda seconds: None)
    created: list[GlanosAnonymizationTools] = []

    def make(
        responses: list[httpx.Response],
    ) -> tuple[GlanosAnonymizationTools, Recorder]:
        recorder = Recorder(responses)
        tools = GlanosAnonymizationTools(token="t", base_url="https://x.test")
        tools.client.close()
        tools.client = GlanosClient(
            token="t",
            base_url="https://x.test",
            transport=httpx.MockTransport(recorder),
        )
        created.append(tools)
        return tools, recorder

    yield make
    for tools in created:
        tools.close()


def test_definitions_offer_review_and_fetch():
    tools = GlanosAnonymizationTools(token="t", base_url="https://x.test")

    names = [name for name, _, _ in tools.definitions()]

    assert names[:4] == ["anonymize", "anonymize_with_review", "fetch", "deanonymize"]
    tools.close()


class TestAnonymize:
    def test_text_starts_a_session_and_returns_no_keys(self, make_tools):
        tools, recorder = make_tools([completed(session="s-new")])

        output = tools.anonymize(text="Michael was here")

        assert output.pop("text").strip() == "PERSON1 was here"
        assert output == {"session": "s-new"}
        request = recorder.requests[0]
        assert json.loads(request.content)["sessionMaxAge"] == 24 * 60 * 60 * 1000
        assert request.url.params["returnPseudoKey"] == "false"
        assert request.url.params["returnEncryptKey"] == "false"

    def test_reuses_the_session_it_is_given(self, make_tools):
        tools, recorder = make_tools([completed(session="s1")])

        output = tools.anonymize(text="Michael was here", session="s1")

        assert output["session"] == "s1"
        body = json.loads(recorder.requests[0].content)
        assert body["session"] == "s1"
        assert "sessionMaxAge" not in body

    def test_new_sessions_get_the_configured_lifetime(self, make_tools):
        tools, recorder = make_tools([completed(session="s-new")])
        tools.session_max_age_ms = 60_000

        tools.anonymize(text="Michael was here")

        assert json.loads(recorder.requests[0].content)["sessionMaxAge"] == 60_000

    def test_document(self, make_tools):
        tools, recorder = make_tools(
            [
                accepted(),
                completed(data=b64(b"docx"), fileName="a_anonym.txt", session="s-new"),
            ]
        )

        output = tools.anonymize(file_base64=b64(b"original"), file_name="a.txt")

        assert output == {
            "text": "docx",
            "file_base64": b64(b"docx"),
            "file_name": "a_anonym.txt",
            "session": "s-new",
        }
        # a document is collected with fetch, which must not ask for the keys either
        fetch = recorder.requests[1]
        assert fetch.url.params["returnPseudoKey"] == "false"
        assert fetch.url.params["returnEncryptKey"] == "false"

    def test_a_document_not_ready_in_time_returns_the_fetch_key(self, make_tools):
        tools, _ = make_tools([accepted()])

        output = tools.anonymize(file_base64=b64(b"original"), file_name="a.docx")

        assert set(output) == {"message", "fetch_key"}
        assert "not ready" in output["message"]
        assert output["fetch_key"] == "fk-1"


class TestAnonymizeWithReview:
    def test_returns_the_fetch_key_at_once(self, make_tools):
        tools, recorder = make_tools([accepted()])

        output = tools.anonymize_with_review(
            text="Michael was here", assignee="anna@example.com"
        )

        assert output == {"fetch_key": "fk-1"}
        assert len(recorder.requests) == 1
        assert json.loads(recorder.requests[0].content)["assignee"] == (
            "anna@example.com"
        )

    def test_needs_exactly_one_reviewer(self, make_tools):
        tools, recorder = make_tools([accepted()])

        with pytest.raises(ValueError, match="exactly one"):
            tools.anonymize_with_review(text="hi")

        assert recorder.requests == []


class TestFetch:
    def test_returns_the_document_and_no_keys(self, make_tools):
        tools, recorder = make_tools(
            [completed(data=b64(b"docx"), fileName="a_anonym.txt", session="s1")]
        )

        output = tools.fetch("fk-1")

        assert output == {
            "text": "docx",
            "file_base64": b64(b"docx"),
            "file_name": "a_anonym.txt",
            "session": "s1",
        }
        request = recorder.requests[0]
        assert request.url.params["fetchKey"] == "fk-1"
        assert request.url.params["returnPseudoKey"] == "false"
        assert request.url.params["returnEncryptKey"] == "false"

    def test_checks_once_and_returns_the_fetch_key_if_not_ready(self, make_tools):
        tools, recorder = make_tools([accepted()])

        output = tools.fetch("fk-1")

        assert len(recorder.requests) == 1
        assert set(output) == {"message", "fetch_key"}
        assert "not ready" in output["message"]
        assert output["fetch_key"] == "fk-1"


@pytest.mark.parametrize(
    "error", [ImportError("needs text-extraction"), RuntimeError("cannot convert")]
)
class TestDocumentWhoseTextCannotBeExtracted:
    @pytest.fixture(autouse=True)
    def failing_extraction(self, monkeypatch, error):
        def fail():
            raise error

        monkeypatch.setattr("glanos_anonymization.core._default_text_extractor", fail)

    def test_anonymize_returns_the_document_without_text(self, make_tools, error):
        document = completed(data=b64(b"docx"), fileName="a_anonym.docx", session="s1")
        tools, _ = make_tools([accepted(), document])

        output = tools.anonymize(file_base64=b64(b"original"), file_name="a.docx")

        assert str(error) in output.pop("message")
        assert output == {
            "file_base64": b64(b"docx"),
            "file_name": "a_anonym.docx",
            "session": "s1",
        }

    def test_fetch_returns_the_document_without_text(self, make_tools, error):
        document = completed(data=b64(b"docx"), fileName="a_anonym.docx", session="s1")
        tools, _ = make_tools([document])

        output = tools.fetch("fk-1")

        assert str(error) in output.pop("message")
        assert output == {
            "file_base64": b64(b"docx"),
            "file_name": "a_anonym.docx",
            "session": "s1",
        }

    def test_deanonymize_returns_the_document_without_text(self, make_tools, error):
        tools, _ = make_tools([completed(data=b64(b"docx"), fileName="a.docx")])

        output = tools.deanonymize(
            file_base64=b64(b"anonymized"), file_name="a_anonym.docx", session="s1"
        )

        assert str(error) in output.pop("message")
        assert output == {"file_base64": b64(b"docx"), "file_name": "a.docx"}


class TestDeanonymize:
    def test_text_with_a_session(self, make_tools):
        tools, recorder = make_tools([completed(data=b64(b"Michael was here\n"))])

        output = tools.deanonymize(text="PERSON1 was here", session="s1")

        assert output["text"].strip() == "Michael was here"
        body = json.loads(recorder.requests[0].content)
        assert body["plainText"] == "PERSON1 was here"
        assert body["session"] == "s1"

    def test_document_with_a_session(self, make_tools):
        tools, recorder = make_tools(
            [completed(data=b64(b"Michael\n"), fileName="a.txt")]
        )

        output = tools.deanonymize(
            file_base64=b64(b"PERSON1"), file_name="a_anonym.txt", session="s1"
        )

        assert output.pop("text").strip() == "Michael"
        assert output == {"file_base64": b64(b"Michael\n"), "file_name": "a.txt"}
        body = json.loads(recorder.requests[0].content)
        assert body["session"] == "s1"
        assert "pseudoKey" not in body
        assert "encryptKey" not in body
