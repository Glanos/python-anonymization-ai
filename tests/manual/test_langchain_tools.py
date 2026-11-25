"""An agent with a real LLM using the GlanosAnonymizationTools.

You talk to the agent in plain words, and it decides which Glanos tool to call.
Every tool call and its result are printed (long ones cut off), so you can see
whether the agent picks the right tool and passes the right arguments. Here the
model does see the real values: the tools anonymize on request, they do not
hide anything from the model (that is the middleware's job).

Needs OPENAI_API_KEY (e.g. in .env) besides the Glanos variables. With
GLANOS_REVIEW_ASSIGNEE or GLANOS_REVIEW_GROUP set, it also sends a text for
review; the document then waits in that reviewer's dashboard. Run it with -s:

    uv run pytest tests/manual/test_langchain_tools.py -s

The answers come from a real model, so they differ from run to run; the test
only asserts which tools were called.
"""

import os

import pytest
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

from glanos_anonymization import GlanosAnonymizationTools

MODEL = "gpt-6-luna"
ASSIGNEE = os.environ.get("GLANOS_REVIEW_ASSIGNEE", "")
GROUP = os.environ.get("GLANOS_REVIEW_GROUP", "")
TEXT = "Michael Huber wohnt in der Lindenstraße 5 in Regensburg."

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


def _shorten(text: str, limit: int = 300) -> str:
    return text if len(text) <= limit else text[:limit] + "..."


class Conversation:
    """Sends your messages to the agent and prints what it did in between."""

    def __init__(self, agent) -> None:
        self.agent = agent
        self.thread = {"configurable": {"thread_id": "manual-tools"}}
        self.seen = 0

    def say(self, text: str) -> list[str]:
        """Returns the names of the tools the agent called for this message."""
        print(f"\nYou: {text}")
        state = self.agent.invoke({"messages": [HumanMessage(text)]}, self.thread)
        new = state["messages"][self.seen + 1 :]
        self.seen = len(state["messages"])

        called = []
        for message in new:
            if isinstance(message, AIMessage):
                for call in message.tool_calls:
                    called.append(call["name"])
                    print(f"   -> {call['name']}({_shorten(str(call['args']))})")
            elif isinstance(message, ToolMessage):
                print(f"   <- {message.status}: {_shorten(str(message.content))}")
        print(f"Agent: {new[-1].content}")
        return called


def test_the_agent_uses_the_glanos_tools():
    glanos_tools = GlanosAnonymizationTools(
        token=os.environ["GLANOS_TOKEN"],
        base_url=os.environ["GLANOS_BASE_URL"],
    )
    agent = create_agent(
        # gpt-6-luna rejects tools unless reasoning_effort is "none"
        ChatOpenAI(model=MODEL, reasoning_effort="none"),
        tools=glanos_tools.tools,
        system_prompt=glanos_tools.usage_guide,
        checkpointer=InMemorySaver(),
    )
    talk = Conversation(agent)

    try:
        # 1. Anonymize a text: the agent gets the text and a new session back.
        called = talk.say(f"Anonymisiere diesen Text: {TEXT}")
        assert "anonymize" in called

        # 2. Restore it: the agent has to pass the session from step 1 on.
        called = talk.say("Stelle jetzt den Originaltext wieder her.")
        assert "deanonymize" in called

        # 3. Read the stored settings.
        called = talk.say("Welche gespeicherten Einstellungen gibt es?")
        assert "list_settings" in called

        # 4. Send a text for review, if a reviewer is given.
        if ASSIGNEE or GROUP:
            reviewer = f"den Benutzer {ASSIGNEE}" if ASSIGNEE else f"die Gruppe {GROUP}"
            called = talk.say(f"Schicke diesen Text zur Prüfung an {reviewer}: {TEXT}")
            assert "anonymize_with_review" in called
        else:
            print(
                "\n(Review skipped: set GLANOS_REVIEW_ASSIGNEE or GLANOS_REVIEW_GROUP)"
            )

        print("\nCheck above: the right arguments, and answers that make sense.")
    finally:
        glanos_tools.close()
