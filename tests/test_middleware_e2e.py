"""End-to-end test for GlanosAnonymizationMiddleware against the real Glanos API.

The model is a fake that echoes what it receives, so the test needs no LLM.
Skipped automatically if GLANOS_TOKEN or GLANOS_BASE_URL is missing.
"""

import os

import pytest
from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import Field

from glanos_anonymization.middleware import GlanosAnonymizationMiddleware

GLANOS_API_KEY = os.environ.get("GLANOS_TOKEN", "")
GLANOS_BASE_URL = os.environ.get("GLANOS_BASE_URL", "")

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        not GLANOS_API_KEY, reason="GLANOS_TOKEN not set - skipping e2e tests"
    ),
    pytest.mark.skipif(
        not GLANOS_BASE_URL, reason="GLANOS_BASE_URL not set - skipping e2e tests"
    ),
]


class EchoChatModel(BaseChatModel):
    """Repeats the last message it receives and records every request."""

    requests: list[list[BaseMessage]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "echo"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.requests.append(list(messages))
        reply = AIMessage(f"You said: {messages[-1].content}")
        return ChatResult(generations=[ChatGeneration(message=reply)])

    def bind_tools(self, tools, **kwargs):
        return self


@pytest.fixture
def middleware():
    mw = GlanosAnonymizationMiddleware(
        token=GLANOS_API_KEY, base_url=GLANOS_BASE_URL, session_max_age_ms=60_000
    )
    yield mw
    mw.close()


def test_model_never_sees_the_original_and_the_reply_is_restored(middleware):
    model = EchoChatModel()
    agent = create_agent(
        model,
        system_prompt="Der Nutzer heißt Michael Huber.",
        middleware=[middleware],
        checkpointer=InMemorySaver(),
    )
    thread = {"configurable": {"thread_id": "e2e"}}

    try:
        first = agent.invoke(
            {"messages": [HumanMessage("Mein Name ist Michael Huber.")]}, thread
        )
        agent.invoke({"messages": [HumanMessage("Wie heiße ich?")]}, thread)

        for request in model.requests:
            for message in request:
                assert "Michael" not in message.model_dump_json()
                assert "Huber" not in message.model_dump_json()
        assert "Michael Huber" in first["messages"][-1].content
    finally:
        session_id = agent.get_state(thread).values.get("glanos_session_id")
        if session_id:
            middleware.client.remove_session(session_id)


def test_tool_receives_the_original_value(middleware):
    received: list[str] = []

    @tool
    def greet(name: str) -> str:
        """Greet a person by name."""
        received.append(name)
        return f"Hallo {name}"

    class ToolCallingModel(EchoChatModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            self.requests.append(list(messages))
            if len(self.requests) == 1:
                # call the tool with the pseudonym the model was given
                name = messages[-1].content.strip().removeprefix("Begrüße ").rstrip(".")
                reply = AIMessage(
                    "",
                    tool_calls=[{"name": "greet", "args": {"name": name}, "id": "c1"}],
                )
            else:
                reply = AIMessage(f"Erledigt: {messages[-1].content}")
            return ChatResult(generations=[ChatGeneration(message=reply)])

    model = ToolCallingModel()
    agent = create_agent(
        model,
        tools=[greet],
        middleware=[middleware],
        checkpointer=InMemorySaver(),
    )
    thread = {"configurable": {"thread_id": "e2e-tool"}}

    try:
        result = agent.invoke(
            {"messages": [HumanMessage("Begrüße Michael Huber.")]}, thread
        )

        for request in model.requests:
            for message in request:
                assert "Huber" not in message.model_dump_json()
        assert received == ["Michael Huber"]
        assert "Michael Huber" in result["messages"][-1].content
    finally:
        session_id = agent.get_state(thread).values.get("glanos_session_id")
        if session_id:
            middleware.client.remove_session(session_id)
