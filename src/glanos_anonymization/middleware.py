"""LangChain agent middleware that anonymizes user input before it reaches
the model and de-anonymizes the model's replies, using a single Glanos
*session* per conversation so pseudonym mappings stay consistent across every
turn - e.g. once "Michael" is mapped to "PERSON1", it stays "PERSON1" in every
later message of the same conversation, and gets correctly restored to
"Michael" in every later model reply.

IMPORTANT - persistence across separate agent.invoke()/ainvoke() calls (not
just within a single call) requires the agent to be compiled with a
LangGraph checkpointer and invoked with a stable thread_id - the Glanos
session id is stored in that thread's checkpointed state (see
GlanosSessionState). Without a checkpointer, a fresh session is created on
every invoke() call and consistency only holds within that one call.

    from langgraph.checkpoint.memory import MemorySaver
    from langchain.agents import create_agent
    from glanos_anonymization.middleware import GlanosAnonymizationMiddleware

    glanos_middleware = GlanosAnonymizationMiddleware(
        token=GLANOS_TOKEN, base_url=GLANOS_BASE_URL,
    )
    agent = create_agent(
        model=model,
        middleware=[glanos_middleware],
        checkpointer=MemorySaver(),  # or a persistent checkpointer in production
    )
    config = {"configurable": {"thread_id": "conversation-123"}}
    agent.invoke({"messages": [...]}, config)  # same thread_id -> same session

Note: thread_id and the Glanos session id are deliberately NOT the same
value, and should not be conflated:
  - thread_id must be chosen by the caller *before* the first invoke() call,
    but the Glanos session id does not exist yet at that point - it is only
    returned by the server the first time before_agent runs.
  - If the Glanos session expires and a new one is minted, only the
    glanos_session_id *field* in the state changes - the thread_id (and thus
    the conversation's message history) is unaffected. Using the session id
    as the thread_id itself would tie the conversation's identity to the
    anonymization session's lifecycle, which is unrelated.
"""

from typing import Any, NotRequired

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.runtime import Runtime

from glanos_anonymization.core import (
    AnonymizationConfig,
    GlanosClient,
    GlanosClientError,
)

_ANONYMIZED_FLAG = "glanos_anonymized"
_DEANONYMIZED_FLAG = "glanos_deanonymized"


class GlanosSessionState(AgentState):
    """Default agent state, plus the Glanos session id for this conversation."""

    glanos_session_id: NotRequired[str | None]


class GlanosAnonymizationMiddleware(AgentMiddleware):
    """
    Anonymizes `HumanMessage` content before it reaches the model
    (`before_model`) and de-anonymizes `AIMessage` content before it reaches
    the caller (`after_model`, with `after_agent` as a safety net), using one
    Glanos session per conversation (created once in `before_agent`) so the
    same pseudonym mapping is reused consistently across every turn.
    """

    state_schema = GlanosSessionState

    def __init__(
        self,
        token: str,
        base_url: str,
        config: AnonymizationConfig | None = None,
        session_max_age_ms: int = 24 * 60 * 60 * 1000,
        verbose: bool = False,
    ) -> None:
        """
        Args:
            token: Glanos API token.
            base_url: Glanos API base url.
            config: Optional AnonymizationConfig for the underlying client
                (e.g. to set anonymization_mode, exclude_fields, ...). A copy
                is used with a_sync forced to False, since this middleware
                needs completed results immediately - the original object
                passed in is not mutated.
            session_max_age_ms: Expiration age (in ms) requested for newly
                created Glanos sessions. Default: 24h.
            verbose: If True, print what gets sent to the model (anonymized)
                and what comes back at each stage (still-anonymized model
                reply, then the de-anonymized final text) - useful to see
                the anonymization actually happening. Off by default.
        """
        base_config = config if config is not None else AnonymizationConfig()
        effective_config = base_config.model_copy(update={"a_sync": False})
        self.client = GlanosClient(
            token=token, base_url=base_url, config=effective_config
        )
        self.session_max_age_ms = session_max_age_ms
        self.verbose = verbose

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self.client.close()

    def before_agent(
        self, state: GlanosSessionState, runtime: Runtime[Any]
    ) -> dict[str, Any] | None:
        """Create a Glanos session for this conversation if it doesn't have
        one yet. A session, once created, is reused for every later call
        (persistence across separate invoke() calls requires a checkpointer
        - see the module docstring)."""

        if state.get("glanos_session_id"):
            return None

        # A minimal placeholder is submitted purely to have the server mint
        # a session id - no real conversation content is sent here.
        result = self.client.pseudo_sync(
            text=".", session_max_age=self.session_max_age_ms
        )
        session_id = result.data.session

        if not session_id:
            raise GlanosClientError(
                "Glanos did not return a session id despite session_max_age being set"
            )

        return {"glanos_session_id": session_id}

    def before_model(
        self, state: GlanosSessionState, runtime: Runtime[Any]
    ) -> dict[str, Any] | None:
        """Anonymize any not-yet-anonymized `HumanMessage` content, using
        the conversation's Glanos session so pseudonyms stay consistent."""

        session_id = state.get("glanos_session_id")
        if not session_id:
            # before_agent should have set this; without a session we cannot
            # anonymize consistently, so leave messages untouched rather
            # than silently sending unanonymized PII to the model would be
            # worse than raising - but before_agent already raises if
            # session creation fails, so this branch should not normally
            # be reached.
            return None

        messages = []
        changed = False

        for message in state["messages"]:
            if (
                isinstance(message, HumanMessage)
                and isinstance(message.content, str)
                and message.content.strip()
                and not message.additional_kwargs.get(_ANONYMIZED_FLAG)
            ):
                pseudo_result = self.client.pseudo_sync(
                    text=message.content, session=session_id
                )
                anonymized_text = self.client.get_ano_result(pseudo_result).text
                if self.verbose:
                    print(f"[Glanos to model]:      {anonymized_text!r}")
                messages.append(
                    message.model_copy(
                        update={
                            "content": anonymized_text,
                            "additional_kwargs": {
                                **message.additional_kwargs,
                                _ANONYMIZED_FLAG: True,
                            },
                        }
                    )
                )
                changed = True
            else:
                messages.append(message)

        return {"messages": messages} if changed else None

    def after_model(
        self, state: GlanosSessionState, runtime: Runtime[Any]
    ) -> dict[str, Any] | None:
        """De-anonymize the model's latest reply, restoring pseudonyms back
        to their original values via the conversation's session."""

        return self._deanonymize_last_ai_message(state)

    def after_agent(
        self, state: GlanosSessionState, runtime: Runtime[Any]
    ) -> dict[str, Any] | None:
        """Safety net: ensure the final message is de-anonymized even if it
        somehow bypassed after_model (e.g. an early graph exit)."""

        return self._deanonymize_last_ai_message(state)

    def _deanonymize_last_ai_message(
        self, state: GlanosSessionState
    ) -> dict[str, Any] | None:
        session_id = state.get("glanos_session_id")
        if not session_id:
            return None

        messages = state["messages"]
        if not messages:
            return None

        last_message = messages[-1]
        if (
            not isinstance(last_message, AIMessage)
            or not isinstance(last_message.content, str)
            or not last_message.content.strip()
            or last_message.additional_kwargs.get(_DEANONYMIZED_FLAG)
        ):
            return None

        if self.verbose:
            print(f"[Glanos raw model reply]: {last_message.content!r}")

        restored_text = self.client.depseudo_text(
            last_message.content, session=session_id
        )

        if self.verbose:
            print(f"[Glanos de-anonymized]:  {restored_text!r}")

        return {
            "messages": [
                last_message.model_copy(
                    update={
                        "content": restored_text,
                        "additional_kwargs": {
                            **last_message.additional_kwargs,
                            _DEANONYMIZED_FLAG: True,
                        },
                    }
                )
            ]
        }
