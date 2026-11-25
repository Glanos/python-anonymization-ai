"""Tests for GlanosAnonymizationMiddleware.

A real LangChain agent runs against a chat model that records every request
it receives, while a fake Glanos client replaces fixed values by pseudonyms.
The core assertion: no original value ever appears in what the model gets.
"""

import asyncio
from typing import Any

import pytest
from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import Field

from glanos_anonymization.core import (
    AnonymizationSettings,
    GlanosClient,
    GlanosClientError,
)
from glanos_anonymization.middleware import GlanosAnonymizationMiddleware

pytestmark = pytest.mark.integration

PSEUDONYMS = {
    "Michael Huber": "Adam Boe",
    "Berlin": "Paris",
    "michael@example.com": "adam@example.org",
    "1711234567": "5550000000",
}
ORIGINALS = list(PSEUDONYMS)


def replace_all(text: str, mapping: dict[str, str]) -> str:
    for old, new in mapping.items():
        text = text.replace(old, new)
    return text


class FakeResult:
    def __init__(self, text: str, ready: bool = True) -> None:
        self.text = text
        self.ready = ready

    def extract_text(self) -> str:
        assert self.ready
        return self.text


class FakeGlanosClient:
    def __init__(self) -> None:
        self.anonymize_calls: list[str] = []
        self.depseudo_calls: list[str] = []
        self.anonymize_kwargs: list[dict[str, Any]] = []
        self.fetch_kwargs: list[dict[str, Any]] = []
        self._COLLECT_IN_TIME = GlanosClient._COLLECT_IN_TIME
        # appended to every result, as the real API appends a newline
        self.newline = ""

    def _start_session(self, max_age_ms, settings=None, settings_id=None) -> str:
        return "sess-1"

    def anonymize(self, *, text: str, session: str, **kwargs) -> FakeResult:
        assert session == "sess-1"
        self.anonymize_calls.append(text)
        self.anonymize_kwargs.append(kwargs)
        return FakeResult(replace_all(text, PSEUDONYMS) + self.newline)

    def fetch(self, result: FakeResult, **kwargs) -> FakeResult:
        if result.ready:
            return result
        self.fetch_kwargs.append(kwargs)
        return FakeResult(result.text)

    def depseudo_text(self, text: str, session: str) -> str:
        assert session == "sess-1"
        self.depseudo_calls.append(text)
        restored = replace_all(text, {new: old for old, new in PSEUDONYMS.items()})
        return restored + self.newline

    def close(self) -> None:
        pass


class RecordingChatModel(BaseChatModel):
    """Answers with the given replies in order and records every request."""

    replies: list[AIMessage]
    requests: list[list[BaseMessage]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "recording"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.requests.append(list(messages))
        reply = self.replies[len(self.requests) - 1]
        return ChatResult(generations=[ChatGeneration(message=reply)])

    def bind_tools(self, tools, **kwargs):
        return self


def assert_no_original_reached_the_model(model: RecordingChatModel) -> None:
    assert model.requests, "the model was never called"
    for request in model.requests:
        for message in request:
            dumped = message.model_dump_json()
            for original in ORIGINALS:
                assert original not in dumped, (
                    f"{original!r} reached the model in {type(message).__name__}: "
                    f"{dumped}"
                )


@pytest.fixture
def fake_client() -> FakeGlanosClient:
    return FakeGlanosClient()


@pytest.fixture
def middleware(fake_client) -> GlanosAnonymizationMiddleware:
    mw = GlanosAnonymizationMiddleware(token="t", base_url="https://x.test")
    mw.client.close()
    mw.client = fake_client
    return mw


class TestNoOriginalValueReachesTheModel:
    def test_multiple_turns_with_a_checkpointer(self, middleware):
        model = RecordingChatModel(
            replies=[
                AIMessage("Hello Adam Boe."),
                AIMessage("Adam Boe, you live in Paris."),
            ]
        )
        agent = create_agent(
            model, middleware=[middleware], checkpointer=InMemorySaver()
        )
        thread = {"configurable": {"thread_id": "t1"}}

        first = agent.invoke(
            {"messages": [HumanMessage("I am Michael Huber.")]}, thread
        )
        second = agent.invoke(
            {"messages": [HumanMessage("Where do I live? Berlin, I think.")]}, thread
        )

        assert_no_original_reached_the_model(model)
        assert first["messages"][-1].content == "Hello Michael Huber."
        assert second["messages"][-1].content == "Michael Huber, you live in Berlin."
        # the second request carries the whole conversation, anonymized
        assert [m.content for m in model.requests[1]] == [
            "I am Adam Boe.",
            "Hello Adam Boe.",
            "Where do I live? Paris, I think.",
        ]

    def test_tool_calls_and_tool_results_in_one_invoke(self, middleware):
        received_by_tool: list[str] = []

        @tool
        def lookup_customer(name: str) -> str:
            """Look up a customer by name."""
            received_by_tool.append(name)
            return "Michael Huber lives in Berlin, mail michael@example.com"

        model = RecordingChatModel(
            replies=[
                AIMessage(
                    "",
                    tool_calls=[
                        {
                            "name": "lookup_customer",
                            "args": {"name": "Adam Boe"},
                            "id": "call-1",
                        }
                    ],
                ),
                AIMessage("Adam Boe lives in Paris."),
            ]
        )
        agent = create_agent(model, tools=[lookup_customer], middleware=[middleware])

        result = agent.invoke(
            {"messages": [HumanMessage("Where does Michael Huber live?")]}
        )

        assert_no_original_reached_the_model(model)
        # the tool works with the original value
        assert received_by_tool == ["Michael Huber"]
        assert result["messages"][-1].content == "Michael Huber lives in Berlin."
        # the model gets back its own tool call and the anonymized tool result
        second_request = model.requests[1]
        assert second_request[1].tool_calls[0]["args"] == {"name": "Adam Boe"}
        assert second_request[2].content == (
            "Adam Boe lives in Paris, mail adam@example.org"
        )

    def test_system_prompt(self, middleware):
        model = RecordingChatModel(replies=[AIMessage("Hi.")])
        agent = create_agent(
            model,
            system_prompt="The user is Michael Huber from Berlin.",
            middleware=[middleware],
        )

        agent.invoke({"messages": [HumanMessage("Hello")]})

        assert_no_original_reached_the_model(model)
        assert model.requests[0][0].content == "The user is Adam Boe from Paris."

    def test_list_content_and_non_text_blocks(self, middleware):
        model = RecordingChatModel(replies=[AIMessage("Hi.")])
        agent = create_agent(model, middleware=[middleware])
        image = {"type": "image", "url": "https://x.test/photo.png"}

        agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        [
                            {"type": "text", "text": "I am Michael Huber."},
                            "Call me at 1711234567.",
                            image,
                        ]
                    )
                ]
            }
        )

        assert_no_original_reached_the_model(model)
        # non-text content is not anonymized: it reaches the model unchanged
        assert model.requests[0][0].content == [
            {"type": "text", "text": "I am Adam Boe."},
            "Call me at 5550000000.",
            image,
        ]

    def test_ai_messages_in_the_input(self, middleware):
        # e.g. a conversation history the application passes in itself
        model = RecordingChatModel(replies=[AIMessage("Sure.")])
        agent = create_agent(model, middleware=[middleware])

        agent.invoke(
            {
                "messages": [
                    HumanMessage("Hi"),
                    AIMessage(
                        "Hello Michael Huber",
                        tool_calls=[
                            {
                                "name": "call",
                                "args": {"phone": 1711234567, "who": ["Berlin"]},
                                "id": "c1",
                            }
                        ],
                        additional_kwargs={
                            "tool_calls": [
                                {
                                    "id": "c1",
                                    "function": {
                                        "name": "call",
                                        "arguments": '{"who": "Michael Huber"}',
                                    },
                                }
                            ]
                        },
                    ),
                    HumanMessage("Thanks"),
                ]
            }
        )

        assert_no_original_reached_the_model(model)
        ai_message = model.requests[0][1]
        assert ai_message.tool_calls[0]["args"] == {
            "phone": "5550000000",
            "who": ["Paris"],
        }

    def test_async_agent(self, middleware):
        model = RecordingChatModel(replies=[AIMessage("Hello Adam Boe.")])
        agent = create_agent(model, middleware=[middleware])

        result = asyncio.run(
            agent.ainvoke({"messages": [HumanMessage("I am Michael Huber.")]})
        )

        assert_no_original_reached_the_model(model)
        assert result["messages"][-1].content == "Hello Michael Huber."


def event_loop_running() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


class TestAsyncAgent:
    def test_glanos_calls_do_not_block_the_event_loop(
        self, middleware, fake_client, monkeypatch
    ):
        on_event_loop: dict[str, bool] = {}
        for name in ("_start_session", "anonymize", "depseudo_text"):
            original = getattr(fake_client, name)

            def recording(*args, _name=name, _original=original, **kwargs):
                on_event_loop[_name] = event_loop_running()
                return _original(*args, **kwargs)

            monkeypatch.setattr(fake_client, name, recording)
        model = RecordingChatModel(replies=[AIMessage("Hello Adam Boe.")])
        agent = create_agent(model, middleware=[middleware])

        asyncio.run(agent.ainvoke({"messages": [HumanMessage("I am Michael Huber.")]}))

        assert on_event_loop == {
            "_start_session": False,
            "anonymize": False,
            "depseudo_text": False,
        }


class TestMiddlewareBehavior:
    def test_each_text_is_anonymized_once(self, middleware, fake_client):
        model = RecordingChatModel(
            replies=[AIMessage("Hello Adam Boe."), AIMessage("Still Adam Boe.")]
        )
        agent = create_agent(
            model, middleware=[middleware], checkpointer=InMemorySaver()
        )
        thread = {"configurable": {"thread_id": "t1"}}

        agent.invoke({"messages": [HumanMessage("I am Michael Huber.")]}, thread)
        agent.invoke({"messages": [HumanMessage("Again.")]}, thread)

        # the restored reply is not sent to the API again: the model gets back
        # what it wrote
        assert fake_client.anonymize_calls == ["I am Michael Huber.", "Again."]

    # The real API appends a newline to every text; nothing may change either
    # once it stops doing so.
    @pytest.mark.parametrize("newline", ["\n", ""])
    def test_the_newline_the_api_appends_is_dropped(
        self, middleware, fake_client, newline
    ):
        fake_client.newline = newline
        received: list[str] = []

        @tool
        def greet(name: str) -> str:
            """Greet a person by name."""
            received.append(name)
            return f"Hallo {name}"

        model = RecordingChatModel(
            replies=[
                AIMessage(
                    "",
                    tool_calls=[
                        {"name": "greet", "args": {"name": "Adam Boe"}, "id": "c1"}
                    ],
                ),
                AIMessage("Done, Adam Boe."),
            ]
        )
        agent = create_agent(model, tools=[greet], middleware=[middleware])

        result = agent.invoke({"messages": [HumanMessage("Greet Michael Huber.")]})

        assert model.requests[0][0].content == "Greet Adam Boe."
        assert received == ["Michael Huber"]
        assert result["messages"][-1].content == "Done, Michael Huber."

    def test_keys_are_not_requested(self, middleware, fake_client):
        model = RecordingChatModel(replies=[AIMessage("Hi.")])
        agent = create_agent(model, middleware=[middleware])

        agent.invoke({"messages": [HumanMessage("I am Michael Huber.")]})

        assert fake_client.anonymize_kwargs[0]["return_pseudo_key"] is False
        assert fake_client.anonymize_kwargs[0]["return_encrypt_key"] is False

    def test_without_settings_it_uses_placeholders(self, middleware, fake_client):
        model = RecordingChatModel(replies=[AIMessage("Hi.")])
        agent = create_agent(model, middleware=[middleware])

        agent.invoke({"messages": [HumanMessage("I am Michael Huber.")]})

        assert fake_client.anonymize_kwargs[0]["settings"].mode == "TYPE"
        assert fake_client.anonymize_kwargs[0]["settings_id"] is None

    def test_a_stored_setting_replaces_the_default(self, fake_client):
        mw = GlanosAnonymizationMiddleware(
            token="t", base_url="https://x.test", settings_id="s"
        )
        mw.client.close()
        mw.client = fake_client
        model = RecordingChatModel(replies=[AIMessage("Hi.")])
        agent = create_agent(model, middleware=[mw])

        agent.invoke({"messages": [HumanMessage("I am Michael Huber.")]})

        assert fake_client.anonymize_kwargs[0]["settings"] is None
        assert fake_client.anonymize_kwargs[0]["settings_id"] == "s"

    def test_settings_take_the_place_of_the_settings_id(self, fake_client):
        # the client rejects both together; the session already carries the
        # stored setting
        settings = AnonymizationSettings(mode="TYPE")
        mw = GlanosAnonymizationMiddleware(
            token="t", base_url="https://x.test", settings=settings, settings_id="s"
        )
        mw.client.close()
        mw.client = fake_client
        model = RecordingChatModel(replies=[AIMessage("Hi.")])
        agent = create_agent(model, middleware=[mw])

        agent.invoke({"messages": [HumanMessage("I am Michael Huber.")]})

        assert fake_client.anonymize_kwargs[0]["settings"] is settings
        assert fake_client.anonymize_kwargs[0]["settings_id"] is None

    def test_a_text_not_ready_in_time_is_collected_by_polling(
        self, middleware, fake_client
    ):
        anonymize = fake_client.anonymize

        def not_ready(**kwargs):
            return FakeResult(anonymize(**kwargs).text, ready=False)

        fake_client.anonymize = not_ready
        model = RecordingChatModel(replies=[AIMessage("Hi.")])
        agent = create_agent(model, middleware=[middleware])

        agent.invoke({"messages": [HumanMessage("I am Michael Huber.")]})

        assert_no_original_reached_the_model(model)
        assert fake_client.fetch_kwargs == [
            {
                "poll_interval": 10.0,
                "max_attempts": 30,
                "return_pseudo_key": False,
                "return_encrypt_key": False,
            }
        ]

    def test_verbose_prints_no_original_value(self, fake_client, capsys):
        mw = GlanosAnonymizationMiddleware(
            token="t", base_url="https://x.test", verbose=True
        )
        mw.client.close()
        mw.client = fake_client
        model = RecordingChatModel(replies=[AIMessage("Hello Adam Boe.")])
        agent = create_agent(model, middleware=[mw])

        agent.invoke({"messages": [HumanMessage("I am Michael Huber.")]})

        out = capsys.readouterr().out
        assert "Adam Boe" in out
        assert "Michael Huber" not in out

    def test_without_a_session_nothing_is_sent(self, middleware):
        model = RecordingChatModel(replies=[AIMessage("Hi.")])
        request = type("Request", (), {"state": {"messages": []}, "messages": []})()

        with pytest.raises(GlanosClientError, match="unanonymized"):
            middleware.wrap_model_call(request, lambda r: model.invoke([]))

        assert model.requests == []
