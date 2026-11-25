"""A conversation with a real LLM through GlanosAnonymizationMiddleware.

The agent talks to OpenAI in two turns and has one tool that looks up customer
data. Every request that goes to OpenAI is printed, so you can see with your
own eyes that it holds pseudonyms only, while the answers you get back hold
the real values. Needs OPENAI_API_KEY (e.g. in .env) besides the Glanos
variables. Run it with -s:

    uv run pytest tests/manual/test_langchain_middleware.py -s

The answers come from a real model, so they differ from run to run; the test
only asserts what must always hold.
"""

import os

import pytest
from langchain.agents import create_agent
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

from glanos_anonymization import GlanosAnonymizationMiddleware

MODEL = "gpt-6-luna"
NAME = "Michael Huber"
CITY = "Regensburg"

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        not (os.environ.get("GLANOS_TOKEN") and os.environ.get("GLANOS_BASE_URL")),
        reason="GLANOS_TOKEN or GLANOS_BASE_URL not set",
    ),
    pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set"
    ),
]


class SentToModel(BaseCallbackHandler):
    """Records and prints every request exactly as it goes to OpenAI."""

    def __init__(self) -> None:
        self.requests: list[list[BaseMessage]] = []

    def on_chat_model_start(self, serialized, messages, **kwargs) -> None:
        for request in messages:
            self.requests.append(request)
            print(f"\n   --- request {len(self.requests)} to {MODEL} ---")
            for message in request:
                content = message.content or ""
                calls = getattr(message, "tool_calls", None)
                print(f"   [{message.type}] {content}" + (f" {calls}" if calls else ""))


def test_the_model_sees_pseudonyms_and_you_see_the_real_values():
    looked_up: list[str] = []

    @tool
    def look_up_customer(name: str) -> str:
        """Look up a customer's data by their full name."""
        looked_up.append(name)
        return f"{name}, Kundennummer 4711, wohnt in {CITY}, Kunde seit 2019."

    sent = SentToModel()
    middleware = GlanosAnonymizationMiddleware(
        token=os.environ["GLANOS_TOKEN"],
        base_url=os.environ["GLANOS_BASE_URL"],
        session_max_age_ms=10 * 60 * 1000,
    )
    agent = create_agent(
        # gpt-6-luna rejects tools unless reasoning_effort is "none"
        ChatOpenAI(model=MODEL, reasoning_effort="none", callbacks=[sent]),
        tools=[look_up_customer],
        system_prompt="Du bist ein Kundenservice-Assistent. Antworte kurz.",
        middleware=[middleware],
        checkpointer=InMemorySaver(),
    )
    thread = {"configurable": {"thread_id": "manual-middleware"}}

    try:
        # 1. A turn with a tool call: the tool needs the real name, and its
        #    result (with the city) goes back to the model anonymized.
        print(f"\n1. You: Ich bin {NAME}. Schlag bitte meine Kundendaten nach.")
        first = agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        f"Ich bin {NAME}. Schlag bitte meine Kundendaten nach."
                    )
                ]
            },
            thread,
        )
        print(f"\n   Agent: {first['messages'][-1].content}")
        print(f"   The tool was called with: {looked_up}")
        first_session = agent.get_state(thread).values["glanos_session_id"]

        # 2. A later turn: it uses the same session, and the answer is restored.
        print("\n2. You: Wie heiße ich und wo wohne ich?")
        second = agent.invoke(
            {"messages": [HumanMessage("Wie heiße ich und wo wohne ich?")]}, thread
        )
        answer = second["messages"][-1].content
        print(f"\n   Agent: {answer}")

        print(
            f"\n3. Check above: every request shows {NAME} and {CITY} only as "
            "pseudonyms."
        )
        for request in sent.requests:
            for message in request:
                assert "Huber" not in message.model_dump_json()
        assert looked_up and all("Huber" in name for name in looked_up), (
            "the tool did not get the real name"
        )
        assert "Huber" in answer, "the answer was not restored"
        # The numbering cannot show this: every request repeats the whole
        # conversation, so a new session would give the same PERSON1.
        second_session = agent.get_state(thread).values["glanos_session_id"]
        assert second_session == first_session, "the second turn used a new session"
    finally:
        session_id = agent.get_state(thread).values.get("glanos_session_id")
        if session_id:
            middleware.client.remove_session(session_id)
        middleware.close()
