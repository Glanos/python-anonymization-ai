import asyncio
import dataclasses
import threading
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from typing import Any, NotRequired

from glanos_anonymization.core import (
    AnonymizationSettings,
    GlanosClient,
    GlanosClientError,
    _missing_extra,
)

try:
    from langchain.agents.middleware import (
        AgentMiddleware,
        AgentState,
        ModelRequest,
        ModelResponse,
    )
    from langchain_core.messages import AIMessage, BaseMessage
    from langgraph.runtime import Runtime
except ImportError as exc:
    raise _missing_extra("GlanosAnonymizationMiddleware", "langchain") from exc

_CACHE_SIZE = 4096


class GlanosSessionState(AgentState):
    glanos_session_id: NotRequired[str | None]


class GlanosAnonymizationMiddleware(AgentMiddleware):
    """Anonymizes every request to the model and restores the model's reply.

    The agent state keeps the original values. Right before each model call,
    the whole request - system prompt, every message and the arguments of
    earlier tool calls - is anonymized with the conversation's session; the
    reply, including the arguments of new tool calls, is restored before it
    enters the state. Tools therefore work with the original values, and the
    model only ever sees pseudonyms.

    Put this middleware last in the agent's middleware list, so that it sees
    the request exactly as it is sent to the model.
    """

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
        if settings is None and settings_id is None:
            settings = AnonymizationSettings(mode="TYPE")
        self.settings = settings
        self.settings_id = settings_id
        self.language = language
        # (session, original text) -> the anonymized text the model sees
        self._anonymized: OrderedDict[tuple[str, str], str] = OrderedDict()
        self._lock = threading.Lock()

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

    # The async hooks run the Glanos calls in a worker thread, so that they do
    # not block the event loop; without them, LangGraph would run the sync
    # hooks on the event loop itself.
    async def abefore_agent(
        self, state: GlanosSessionState, runtime: Runtime[Any]
    ) -> dict[str, Any] | None:
        return await asyncio.to_thread(self.before_agent, state, runtime)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        session_id = _session_id(request)
        response = handler(self._anonymize_request(request, session_id))
        return self._restore_response(response, session_id)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        session_id = _session_id(request)
        anonymized = await asyncio.to_thread(
            self._anonymize_request, request, session_id
        )
        response = await handler(anonymized)
        return await asyncio.to_thread(self._restore_response, response, session_id)

    def _anonymize_request(
        self, request: ModelRequest, session_id: str
    ) -> ModelRequest:
        system_message = request.system_message
        if system_message is not None:
            system_message = self._anonymize_message(system_message, session_id)
        return request.override(
            messages=[
                self._anonymize_message(message, session_id)
                for message in request.messages
            ],
            system_message=system_message,
        )

    def _anonymize_message(self, message: BaseMessage, session_id: str) -> Any:
        def anonymize(text: str) -> str:
            return self._anonymize_text(text, session_id)

        if not isinstance(message, AIMessage):
            return message.model_copy(
                update={"content": _map_content(message.content, anonymize)}
            )

        additional_kwargs = {
            key: value
            for key, value in message.additional_kwargs.items()
            if key not in ("tool_calls", "function_call")
        }
        return message.model_copy(
            update={
                "content": _map_content(message.content, anonymize),
                "tool_calls": [
                    {**call, "args": _map_values(call["args"], anonymize, numbers=True)}
                    for call in message.tool_calls
                ],
                "invalid_tool_calls": [
                    {**call, "args": anonymize(call["args"])}
                    if isinstance(call.get("args"), str)
                    else call
                    for call in message.invalid_tool_calls
                ],
                "additional_kwargs": additional_kwargs,
            }
        )

    def _restore_response(
        self, response: ModelResponse, session_id: str
    ) -> ModelResponse:
        return dataclasses.replace(
            response,
            result=[
                self._restore_message(message, session_id)
                if isinstance(message, AIMessage)
                else message
                for message in response.result
            ],
        )

    def _restore_message(self, message: AIMessage, session_id: str) -> AIMessage:
        def restore(text: str) -> str:
            return self._restore_text(text, session_id)

        return message.model_copy(
            update={
                "content": _map_content(message.content, restore),
                "tool_calls": [
                    {**call, "args": _map_values(call["args"], restore)}
                    for call in message.tool_calls
                ],
            }
        )

    def _anonymize_text(self, text: str, session_id: str) -> str:
        if not text.strip():
            return text

        key = (session_id, text)
        with self._lock:
            if key in self._anonymized:
                self._anonymized.move_to_end(key)
                return self._anonymized[key]

        result = self.client.anonymize(
            text=text,
            settings=self.settings,
            # inline settings take precedence over the stored setting anyway
            settings_id=self.settings_id if self.settings is None else None,
            session=session_id,
            language=self.language,
            return_pseudo_key=False,
            return_encrypt_key=False,
        )
        anonymized = _without_added_newline(
            text,
            self.client.fetch(result, **self.client._COLLECT_IN_TIME).extract_text(),
        )
        if self.verbose:
            print(f"[Glanos anonymization.ai to model]:    {anonymized!r}")
        self._remember(session_id, text, anonymized)
        return anonymized

    def _restore_text(self, text: str, session_id: str) -> str:
        if not text.strip():
            return text

        if self.verbose:
            print(f"[Glanos anonymization.ai from model]:  {text!r}")
        restored = _without_added_newline(
            text, self.client.depseudo_text(text, session=session_id)
        )
        # When the restored text is sent again, the model gets back exactly
        # what it wrote, without another API call.
        self._remember(session_id, restored, text)
        return restored

    def _remember(self, session_id: str, original: str, anonymized: str) -> None:
        with self._lock:
            self._anonymized[(session_id, original)] = anonymized
            self._anonymized.move_to_end((session_id, original))
            while len(self._anonymized) > _CACHE_SIZE:
                self._anonymized.popitem(last=False)


def _session_id(request: ModelRequest) -> str:
    session_id = request.state.get("glanos_session_id")
    if not session_id:
        raise GlanosClientError(
            "No Glanos anonymization.ai session in the agent state; refusing to "
            "send the request to the model unanonymized"
        )
    return session_id


def _without_added_newline(original: str, result: str) -> str:
    # The API ends every plain-text result with a newline, whether or not the
    # input had one. Drop it when the input had none, so that tool arguments and
    # replies keep their exact value; a result without it is left as it is.
    if result.endswith("\n") and not original.endswith("\n"):
        return result[:-1]
    return result


def _map_content(content: str | list, convert: Callable[[str], str]) -> str | list:
    if isinstance(content, str):
        return convert(content)

    blocks: list = []
    for block in content:
        if isinstance(block, str):
            blocks.append(convert(block))
        elif block.get("type") == "text":
            blocks.append({**block, "text": convert(block["text"])})
        else:
            # images, files, audio, ... are passed on as they are, not anonymized
            blocks.append(block)
    return blocks


def _map_values(
    value: Any, convert: Callable[[str], str], numbers: bool = False
) -> Any:
    if isinstance(value, str):
        return convert(value)
    if isinstance(value, dict):
        return {key: _map_values(item, convert, numbers) for key, item in value.items()}
    if isinstance(value, list):
        return [_map_values(item, convert, numbers) for item in value]
    if numbers and isinstance(value, int | float) and not isinstance(value, bool):
        # e.g. a phone number passed as an integer
        text = str(value)
        converted = convert(text)
        return value if converted == text else converted
    return value
