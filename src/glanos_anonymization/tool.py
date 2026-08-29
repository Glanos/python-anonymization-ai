"""LangChain tools wrapping GlanosClient's anonymize/depseudonymize
operations, for agents that should decide themselves when to anonymize or
restore text (as opposed to glanos_anonymization.middleware, which does this
automatically around every model call).

Both tools share one Glanos session, created lazily on first use, so a value
anonymized via `anonymize_text` (e.g. "Michael" -> "PERSON1") is restored
correctly by a later `deanonymize_text` call within the same
GlanosAnonymizationTools instance.

    from glanos_anonymization.tool import GlanosAnonymizationTools

    glanos_tools = GlanosAnonymizationTools(token=GLANOS_TOKEN, base_url=GLANOS_BASE_URL)
    agent = create_agent(model=model, tools=glanos_tools.tools)
    ...
    glanos_tools.close()
"""

from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from glanos_anonymization.core import (
    AnonymizationConfig,
    GlanosClient,
    GlanosClientError,
)


class AnonymizeTextInput(BaseModel):
    text: str = Field(
        description="Text potentially containing personal or sensitive information to anonymize."
    )


class DeanonymizeTextInput(BaseModel):
    text: str = Field(
        description="Previously anonymized text (containing placeholders such as PERSON1 or "
        "LOCATION1) to restore to its original values."
    )


class GlanosAnonymizationTools:
    """Bundles `anonymize_text` and `deanonymize_text` as LangChain tools
    backed by a single Glanos session, so agents can anonymize sensitive
    input on demand and later restore it consistently."""

    def __init__(
        self,
        token: str,
        base_url: str,
        config: AnonymizationConfig | None = None,
        session_max_age_ms: int = 24 * 60 * 60 * 1000,
    ) -> None:
        """
        Args:
            token: Glanos API token.
            base_url: Glanos API base url.
            config: Optional AnonymizationConfig for the underlying client
                (e.g. to set anonymization_mode, exclude_fields, ...). A copy
                is used with a_sync forced to False, since these tools need
                completed results immediately - the original object passed
                in is not mutated.
            session_max_age_ms: Expiration age (in ms) requested for the
                Glanos session shared by both tools. Default: 24h.
        """
        base_config = config if config is not None else AnonymizationConfig()
        effective_config = base_config.model_copy(update={"a_sync": False})
        self.client = GlanosClient(
            token=token, base_url=base_url, config=effective_config
        )
        self.session_max_age_ms = session_max_age_ms
        self._session_id: str | None = None

        self.anonymize_tool = StructuredTool.from_function(
            func=self.anonymize_text,
            name="anonymize_text",
            description=(
                "Anonymize personal or sensitive information (names, addresses, phone "
                "numbers, ...) in a piece of text, replacing it with placeholders like "
                "PERSON1 or LOCATION1. Use this before passing user-provided text on to "
                "a system or model that should not see the real values. The same value "
                "always maps to the same placeholder for the lifetime of this tool."
            ),
            args_schema=AnonymizeTextInput,
        )

        self.deanonymize_tool = StructuredTool.from_function(
            func=self.deanonymize_text,
            name="deanonymize_text",
            description=(
                "Restore the original values for placeholders (e.g. PERSON1, LOCATION1) "
                "that anonymize_text previously introduced."
            ),
            args_schema=DeanonymizeTextInput,
        )

    @property
    def tools(self) -> list[StructuredTool]:
        """Both tools, ready to pass as `tools=glanos_tools.tools` to an agent."""
        return [self.anonymize_tool, self.deanonymize_tool]

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self.client.close()

    def _ensure_session(self) -> str:
        if self._session_id is None:
            # A minimal placeholder is submitted purely to have the server
            # mint a session id - no real content is sent here.
            result = self.client.pseudo_sync(
                text=".", session_max_age=self.session_max_age_ms
            )
            session_id = result.data.session
            if not session_id:
                raise GlanosClientError(
                    "Glanos did not return a session id despite session_max_age being set"
                )
            self._session_id = session_id
        return self._session_id

    def anonymize_text(self, text: str) -> str:
        """Anonymize `text`, returning the anonymized text."""
        pseudo_result = self.client.pseudo_sync(
            text=text, session=self._ensure_session()
        )
        return self.client.get_ano_result(pseudo_result).text

    def deanonymize_text(self, text: str) -> str:
        """Restore placeholders in `text` to their original values."""
        return self.client.depseudo_text(text, session=self._ensure_session())
