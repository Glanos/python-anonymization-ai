import base64
import functools
from collections.abc import Callable
from typing import TYPE_CHECKING, Annotated, Any

from pydantic import Field

from glanos_anonymization.core import (
    AnonymizationResult,
    AnonymizationSettings,
    GlanosAPIError,
    GlanosClient,
    GlanosClientError,
    SettingsSort,
    SortOrder,
    _missing_extra,
)

if TYPE_CHECKING:
    from langchain_core.tools import StructuredTool

USAGE_GUIDE = (
    "Glanos anonymization.ai tools - replace personal data in text and documents by "
    "placeholders (PERSON1, LOCATION1, ...) or realistic pseudonyms, and restore it later.\n"
    "1. Anonymize: call `anonymize` with `text`, or with a file as `file_base64` plus "
    "`file_name` (with its extension, e.g. contract.docx). The result holds the anonymized "
    "`text` (for files `file_base64` and `file_name`, and `text` if it can be extracted) "
    "and the `session` it was anonymized with.\n"
    "2. Restore: call `deanonymize` with the anonymized text or file and that `session`. "
    "It restores any text containing the session's placeholders, e.g. a reply written by "
    "a model.\n"
    "3. Same placeholders across several texts or documents (e.g. a conversation): pass "
    "the `session` from the first `anonymize` call, or one the user gives you, to every "
    "further `anonymize` call. Without `session`, `anonymize` starts a new session.\n"
    "4. Settings decide what is anonymized and how. Without `settings` or `settings_id`, "
    "the installation's default setting applies. Pass `settings` for a single call, or "
    "store a setting with `add_setting` and reference it by `settings_id`. "
    "`update_setting` replaces a stored setting completely: read it with `get_setting`, "
    "change the fields and pass it back in full.\n"
    "5. Sessions expire, and an anonymized text or document can only be restored while "
    "its session exists. `get_session` shows when it expires, `extend_session` prolongs "
    "it, `remove_session` deletes it together with its placeholder mapping.\n"
    "6. If a document is not ready (processing takes longer, or it is in review), you get "
    "a `fetch_key` instead of it. Call `fetch` with it later to collect the document."
)

ANONYMIZE = (
    "Anonymize personal or sensitive data (names, addresses, phone numbers, e-mails, "
    "companies, dates, ...) in a text or a document. Pass either `text`, or a document as "
    "`file_base64` plus `file_name` (docx, pdf, pptx, xlsx, txt, html, xml, zip, ...). "
    "Returns the anonymized `text` (for documents the anonymized document as "
    "`file_base64` with its `file_name`, and its `text` if it can be extracted) and the "
    "`session` it was anonymized with, which "
    "`deanonymize` needs to restore it - keep it. Pass `session` (from an earlier call, "
    "or from the user) to reuse its placeholders; without it, a new session is started. "
    "Without `settings` or `settings_id` the installation's default setting is used. "
    "If the document is not ready in time, returns a `fetch_key` instead, which `fetch` "
    "collects it with."
)

ANONYMIZE_WITH_REVIEW = (
    "Anonymize a text or document and send it to a person for review in the Glanos "
    "dashboard. Pass exactly one of `assignee` (a user id, usually the lower-cased "
    "e-mail address) or `assigned_group` (a group id). Returns at once with a "
    "`fetch_key`; after the review, `fetch` collects the reviewed document with it."
)

FETCH = (
    "Collect a document that was not ready yet, with the `fetch_key` from `anonymize` or "
    "`anonymize_with_review`. Checks once. Returns the anonymized document as "
    "`file_base64` with its `file_name`, its `text` if it can be extracted, and the "
    "`session` it was anonymized with, if any. If the document is still not ready (e.g. "
    "in review), returns the `fetch_key` again: try again later."
)

DEANONYMIZE = (
    "Restore the original data in a text or document anonymized by `anonymize`. Pass "
    "either `text`, or the anonymized document as `file_base64` plus `file_name`, "
    "together with the `session` it was anonymized with. Any text containing the "
    "session's placeholders can be restored, e.g. a reply written by a model. Returns "
    "the restored `text` (for documents `file_base64` and `file_name`, and `text` if it "
    "can be extracted)."
)

LIST_SETTINGS = (
    "List the stored anonymization settings (id, name, mode, excluded entity types, ...). "
    "Use it to find the id of a setting for `settings_id` in `anonymize` or for "
    "`add_session`. The setting named 'Default' is the installation's default setting."
)

GET_SETTING = (
    "Get one stored anonymization setting by its id. Use it before `update_setting`, which "
    "needs the complete setting."
)

ADD_SETTING = (
    "Store a new anonymization setting, e.g. to use it for sessions or by `settings_id`. "
    "`name` is required and must not be taken yet; do not pass an `id`, the server "
    "assigns it and returns it with the stored setting."
)

UPDATE_SETTING = (
    "Change a stored anonymization setting. The setting is replaced as a whole: fields "
    "left out are reset, the name included. So first read it with `get_setting`, change "
    "the fields you need and pass the complete setting, including its `id`."
)

REMOVE_SETTING = "Delete a stored anonymization setting by its id."

ADD_SESSION = (
    "Create a session from a stored setting, e.g. to share it with other users or "
    "groups. Not needed otherwise: `anonymize` starts a session by itself. Needs "
    "`settings_id`, the id of a stored setting (see `list_settings` or `add_setting`). "
    "Returns the session with its `id` - pass it as `session` to `anonymize` and "
    "`deanonymize`."
)

GET_SESSION = (
    "Get a session by its id, e.g. to see when it expires "
    "(`expiresAt`, epoch milliseconds)."
)

EXTEND_SESSION = (
    "Extend a session: it then expires `max_age_seconds` from now. Use it to keep a "
    "longer conversation restorable."
)

REMOVE_SESSION = (
    "Delete a session together with its placeholder mapping. Texts anonymized with it "
    "can no longer be restored with the session."
)

APPLY_DEFAULTS = "Whether the server applies its system defaults (keep true unless you know otherwise)."

# Parameters shared by the anonymize and anonymize_with_review tools
Text = Annotated[
    str | None, Field(description="Text to anonymize (instead of a file).")
]
FileBase64 = Annotated[
    str | None,
    Field(description="Document to anonymize, base64-encoded (instead of text)."),
]
FileName = Annotated[
    str | None,
    Field(
        description="File name of the document, with its extension "
        "(e.g. contract.docx). Required with file_base64."
    ),
]
Settings = Annotated[
    AnonymizationSettings | None,
    Field(
        description="Settings for this call only: mode, entity types to leave "
        "untouched (excludeFields), ... Do not combine with settings_id."
    ),
]
SettingsId = Annotated[str | None, Field(description="Id of a stored setting to use.")]
Session = Annotated[
    str | None,
    Field(
        description="Id of a session (from an earlier anonymize call, add_session or "
        "the user), for the same placeholders as before."
    ),
]
Language = Annotated[
    str | None,
    Field(
        description="ISO 639-1 language code (e.g. de, en); detected "
        "automatically if left out."
    ),
]
DataRetentionMs = Annotated[
    int | None,
    Field(
        description="How long the anonymized document stays on the server in "
        "milliseconds, -1 for ever (default 0: removed after download)."
    ),
]
PseudoKeyRetentionMs = Annotated[
    int | None,
    Field(
        description="How long the pseudo key stays on the server in "
        "milliseconds, -1 for ever (default 0)."
    ),
]
MetaRetentionMs = Annotated[
    int | None,
    Field(
        description="How long the meta data stays on the server in "
        "milliseconds, -1 for ever (default 0). Documents only."
    ),
]
Options = Annotated[
    list[str] | None, Field(description="Installation-specific upload options.")
]


class GlanosAnonymizationTools:
    usage_guide = USAGE_GUIDE

    def __init__(
        self,
        token: str,
        base_url: str,
        no_ssl_verify: bool = False,
        session_max_age_ms: int = 24 * 60 * 60 * 1000,
    ) -> None:
        self.client = GlanosClient(
            token=token, base_url=base_url, no_ssl_verify=no_ssl_verify
        )
        # lifetime of the session anonymize starts when the agent passes none
        self.session_max_age_ms = session_max_age_ms

    def close(self) -> None:
        self.client.close()

    def anonymize(
        self,
        text: Text = None,
        file_base64: FileBase64 = None,
        file_name: FileName = None,
        settings: Settings = None,
        settings_id: SettingsId = None,
        session: Session = None,
        language: Language = None,
        data_retention_ms: DataRetentionMs = None,
        pseudo_key_retention_ms: PseudoKeyRetentionMs = None,
        meta_retention_ms: MetaRetentionMs = None,
        options: Options = None,
    ) -> dict[str, Any]:
        # The agent restores with the session only, so the keys are never requested
        # and never enter the model's context.
        result = self.client.anonymize(
            text=text,
            data=base64.b64decode(file_base64) if file_base64 is not None else None,
            filename=file_name,
            settings=settings,
            settings_id=settings_id,
            session=session,
            # without a session, the server starts a new one with this request
            session_max_age_ms=None if session else self.session_max_age_ms,
            return_pseudo_key=False,
            return_encrypt_key=False,
            language=language,
            data_retention_ms=data_retention_ms,
            pseudo_key_retention_ms=pseudo_key_retention_ms,
            meta_retention_ms=meta_retention_ms,
            options=options,
        )
        if not result.ready:
            return _not_ready(result)
        if file_base64 is not None:
            output = _document(result)
        else:
            output = {"text": result.extract_text()}
        output["session"] = result.session
        return output

    def anonymize_with_review(
        self,
        assignee: Annotated[
            str | None,
            Field(
                description="User who reviews the document (usually the "
                "lower-cased e-mail address)."
            ),
        ] = None,
        assigned_group: Annotated[
            str | None,
            Field(description="Group whose members review the document."),
        ] = None,
        text: Text = None,
        file_base64: FileBase64 = None,
        file_name: FileName = None,
        settings: Settings = None,
        settings_id: SettingsId = None,
        session: Session = None,
        language: Language = None,
        data_retention_ms: DataRetentionMs = None,
        pseudo_key_retention_ms: PseudoKeyRetentionMs = None,
        meta_retention_ms: MetaRetentionMs = None,
        options: Options = None,
    ) -> dict[str, Any]:
        result = self.client.anonymize_with_review(
            text=text,
            data=base64.b64decode(file_base64) if file_base64 is not None else None,
            filename=file_name,
            assignee=assignee,
            assigned_group=assigned_group,
            settings=settings,
            settings_id=settings_id,
            session=session,
            language=language,
            data_retention_ms=data_retention_ms,
            pseudo_key_retention_ms=pseudo_key_retention_ms,
            meta_retention_ms=meta_retention_ms,
            options=options,
        )
        return {"fetch_key": result.fetch_key}

    def fetch(
        self,
        fetch_key: Annotated[
            str,
            Field(
                description="Key of the document (fetch_key from anonymize or "
                "anonymize_with_review)."
            ),
        ],
    ) -> dict[str, Any]:
        result = self.client.fetch(
            fetch_key,
            max_attempts=1,
            return_pseudo_key=False,
            return_encrypt_key=False,
        )
        if not result.ready:
            return _not_ready(result)
        return {**_document(result), "session": result.session}

    def deanonymize(
        self,
        session: Annotated[
            str,
            Field(
                description="Session the text or document was anonymized with "
                "(session from the result of anonymize)."
            ),
        ],
        text: Annotated[
            str | None,
            Field(description="Anonymized text to restore (instead of a file)."),
        ] = None,
        file_base64: Annotated[
            str | None,
            Field(
                description="Anonymized document to restore, base64-encoded "
                "(file_base64 from anonymize)."
            ),
        ] = None,
        file_name: Annotated[
            str | None,
            Field(
                description="File name of the anonymized document (file_name from "
                "anonymize). Required with file_base64."
            ),
        ] = None,
    ) -> dict[str, Any]:
        if file_base64 is None:
            if text is None:
                raise ValueError("Provide 'text', or 'file_base64' with 'file_name'")
            return {"text": self.client.depseudo_text(text, session=session)}
        if text is not None:
            raise ValueError("Provide either 'text' or 'file_base64', not both")
        restored = self.client.deanonymize(
            data=base64.b64decode(file_base64), filename=file_name, session=session
        )
        return _document(restored)

    def list_settings(
        self,
        query: Annotated[
            str | None, Field(description="Fuzzy match on the setting name.")
        ] = None,
        creator: Annotated[
            list[str] | None, Field(description="Only settings of these creators.")
        ] = None,
        sort: Annotated[SettingsSort | None, Field(description="Sort field.")] = None,
        sort_order: Annotated[
            SortOrder | None, Field(description="Sort order.")
        ] = None,
        limit: Annotated[
            int | None,
            Field(description="Maximal number of settings (API default 10)."),
        ] = None,
        offset: Annotated[
            int | None, Field(description="0-based offset for paging.")
        ] = None,
    ) -> dict[str, Any]:
        return self.client.list_settings(
            query=query,
            creator=creator,
            sort=sort,
            sort_order=sort_order,
            limit=limit,
            offset=offset,
        ).model_dump(mode="json", exclude_none=True)

    def get_setting(
        self,
        setting_id: Annotated[str, Field(description="Id of the setting.")],
    ) -> dict[str, Any]:
        return self.client.get_setting(setting_id).model_dump(
            mode="json", exclude_none=True
        )

    def add_setting(
        self,
        settings: Annotated[
            AnonymizationSettings,
            Field(description="The new setting; name is required, no id."),
        ],
        apply_defaults: Annotated[bool, Field(description=APPLY_DEFAULTS)] = True,
    ) -> dict[str, Any]:
        return self.client.add_setting(
            settings, apply_defaults=apply_defaults
        ).model_dump(mode="json", exclude_none=True)

    def update_setting(
        self,
        settings: Annotated[
            AnonymizationSettings,
            Field(
                description="The complete new setting including its id - fields "
                "left out are reset."
            ),
        ],
        apply_defaults: Annotated[bool, Field(description=APPLY_DEFAULTS)] = True,
    ) -> dict[str, Any]:
        return self.client.update_setting(
            settings, apply_defaults=apply_defaults
        ).model_dump(mode="json", exclude_none=True)

    def remove_setting(
        self,
        setting_id: Annotated[str, Field(description="Id of the setting to delete.")],
    ) -> dict[str, Any]:
        return self.client.remove_setting(setting_id).model_dump(mode="json")

    def add_session(
        self,
        settings_id: Annotated[
            str, Field(description="Id of the stored setting the session uses.")
        ],
        max_age_seconds: Annotated[
            int, Field(description="Lifetime of the session in seconds.")
        ] = 24 * 60 * 60,
        name: Annotated[str | None, Field(description="Name of the session.")] = None,
        share_users: Annotated[
            list[str] | None, Field(description="Users allowed to use the session.")
        ] = None,
        share_groups: Annotated[
            list[str] | None, Field(description="Groups allowed to use the session.")
        ] = None,
    ) -> dict[str, Any]:
        return self.client.add_session(
            max_age_seconds,
            settings_id=settings_id,
            name=name,
            share_users=share_users,
            share_groups=share_groups,
        ).model_dump(mode="json", exclude_none=True)

    def get_session(
        self,
        session_id: Annotated[str, Field(description="Id of the session.")],
    ) -> dict[str, Any]:
        return self.client.get_session(session_id).model_dump(
            mode="json", exclude_none=True
        )

    def extend_session(
        self,
        session_id: Annotated[str, Field(description="Id of the session.")],
        max_age_seconds: Annotated[
            int, Field(description="The session expires this many seconds from now.")
        ],
    ) -> dict[str, Any]:
        return self.client.extend_session(session_id, max_age_seconds).model_dump(
            mode="json", exclude_none=True
        )

    def remove_session(
        self,
        session_id: Annotated[str, Field(description="Id of the session to delete.")],
    ) -> dict[str, Any]:
        return self.client.remove_session(session_id).model_dump(mode="json")

    def definitions(self) -> list[tuple[str, str, Callable[..., dict[str, Any]]]]:
        return [
            ("anonymize", ANONYMIZE, self.anonymize),
            (
                "anonymize_with_review",
                ANONYMIZE_WITH_REVIEW,
                self.anonymize_with_review,
            ),
            ("fetch", FETCH, self.fetch),
            ("deanonymize", DEANONYMIZE, self.deanonymize),
            ("list_settings", LIST_SETTINGS, self.list_settings),
            ("get_setting", GET_SETTING, self.get_setting),
            ("add_setting", ADD_SETTING, self.add_setting),
            ("update_setting", UPDATE_SETTING, self.update_setting),
            ("remove_setting", REMOVE_SETTING, self.remove_setting),
            ("add_session", ADD_SESSION, self.add_session),
            ("get_session", GET_SESSION, self.get_session),
            ("extend_session", EXTEND_SESSION, self.extend_session),
            ("remove_session", REMOVE_SESSION, self.remove_session),
        ]

    @property
    def tools(self) -> list["StructuredTool"]:
        try:
            from langchain_core.tools import StructuredTool, ToolException
        except ImportError as exc:
            raise _missing_extra("GlanosAnonymizationTools.tools", "langchain") from exc

        return [
            StructuredTool.from_function(
                func=_reporting_errors(method, ToolException),
                name=name,
                description=description,
                handle_tool_error=True,
            )
            for name, description, method in self.definitions()
        ]


def _document(result: AnonymizationResult) -> dict[str, Any]:
    output: dict[str, Any] = {
        "file_name": result.file_name,
        "file_base64": result.pseudo_result.data.data,
    }
    try:
        output["text"] = result.extract_text()
    except Exception as exc:
        # The document is already downloaded and may have expired on the server: it
        # must not get lost because its text cannot be extracted (e.g. without the
        # text-extraction extra).
        output["message"] = f"The text of the document could not be extracted: {exc}"
    return output


def _not_ready(result: AnonymizationResult) -> dict[str, Any]:
    return {
        "message": "The document is not ready yet. Collect it later with `fetch` "
        "and this fetch_key.",
        "fetch_key": result.fetch_key,
    }


def _reporting_errors(
    method: Callable[..., dict[str, Any]],
    error_type: type[Exception],
) -> Callable[..., dict[str, Any]]:
    @functools.wraps(method)
    def run(*args: Any, **kwargs: Any) -> dict[str, Any]:
        try:
            return method(*args, **kwargs)
        except GlanosAPIError as exc:
            raise error_type(
                f"HTTP {exc.status_code}: {exc.api_message or 'request failed'}"
            ) from exc
        except (GlanosClientError, ValueError, ImportError) as exc:
            raise error_type(str(exc)) from exc

    return run
