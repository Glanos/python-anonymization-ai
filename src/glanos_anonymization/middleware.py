from typing import Any, NotRequired

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.runtime import Runtime

from glanos_anonymization.core import (
    AnonymizationSettings,
    GlanosClient,
)

_ANONYMIZED_FLAG = "glanos_anonymized"
_DEANONYMIZED_FLAG = "glanos_deanonymized"


class GlanosSessionState(AgentState):
    glanos_session_id: NotRequired[str | None]


class GlanosAnonymizationMiddleware(AgentMiddleware):
    state_schema = GlanosSessionState

    def __init__(
        self,
        token: str,
        base_url: str,
        session_max_age_ms: int = 24 * 60 * 60 * 1000,
        verbose: bool = False,
        settings: AnonymizationSettings | None = None,
        settings_id: str | None = None,
        language: str | None = None,
        no_ssl_verify: bool = False,
    ) -> None:
        self.client = GlanosClient(
            token=token, base_url=base_url, no_ssl_verify=no_ssl_verify
        )
        self.session_max_age_ms = session_max_age_ms
        self.verbose = verbose
        self.settings = settings
        self.settings_id = settings_id
        self.language = language

    def close(self) -> None:
        self.client.close()

    def before_agent(
        self, state: GlanosSessionState, runtime: Runtime[Any]
    ) -> dict[str, Any] | None:

        if state.get("glanos_session_id"):
            return None

        session_id = self.client._start_session(
            self.session_max_age_ms,
            settings=self.settings,
            settings_id=self.settings_id,
        )
        return {"glanos_session_id": session_id}

    def before_model(
        self, state: GlanosSessionState, runtime: Runtime[Any]
    ) -> dict[str, Any] | None:

        session_id = state.get("glanos_session_id")
        if not session_id:
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
                anonymized_text = self.client.anonymize(
                    text=message.content,
                    settings=self.settings,
                    settings_id=self.settings_id,
                    session=session_id,
                    language=self.language,
                ).text
                if self.verbose:
                    print(
                        f"[Glanos anonymization.ai to model]:      {anonymized_text!r}"
                    )
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

        return self._deanonymize_last_ai_message(state)

    def after_agent(
        self, state: GlanosSessionState, runtime: Runtime[Any]
    ) -> dict[str, Any] | None:

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
            print(
                f"[Glanos anonymization.ai raw model reply]: {last_message.content!r}"
            )

        restored_text = self.client.depseudo_text(
            last_message.content, session=session_id
        )

        if self.verbose:
            print(f"[Glanos anonymization.ai de-anonymized]:  {restored_text!r}")

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
