import base64
import functools
from collections.abc import Callable
from typing import Annotated, Any

from langchain_core.tools import StructuredTool, ToolException
from pydantic import Field

from glanos_anonymization.core import (
    AnonymizationSettings,
    DepseudoPaths,
    GlanosAPIError,
    GlanosClient,
    GlanosClientError,
    InnerResponseData,
    SettingsSort,
    SortOrder,
)

USAGE_GUIDE = (
    "Glanos anonymization.ai tools - replace personal data in text and documents by "
    "placeholders (PERSON1, LOCATION1, ...) or realistic pseudonyms, and restore it later.\n"
    "1. Anonymize: call `anonymize` with `text`, or with a file as `file_base64` plus "
    "`file_name` (with its extension, e.g. contract.docx). The result holds the anonymized "
    "`text` (for files also `file_base64` and `file_name`) and the keys `pseudo_key` and "
    "`encrypt_key`.\n"
    "2. Restore: call `deanonymize` with the anonymized text or file and the `pseudo_key` "
    "and `encrypt_key` from step 1 - or with the `session` it was anonymized with.\n"
    "3. Same placeholders across several texts or documents (e.g. a conversation): create "
    "a session with `add_session` (it needs the id of a stored setting, see "
    "`list_settings` or `add_setting`) and pass its id as `session` to every `anonymize` "
    "call. `deanonymize` with that session restores any text containing those "
    "placeholders, e.g. a reply written by a model.\n"
    "4. Settings decide what is anonymized and how. Without `settings` or `settings_id`, "
    "the installation's default setting applies. Pass `settings` for a single call, or "
    "store a setting with `add_setting` and reference it by `settings_id`. "
    "`update_setting` replaces a stored setting completely: read it with `get_setting`, "
    "change the fields and pass it back in full.\n"
    "5. Sessions expire after `max_age_seconds`; `extend_session` prolongs a session, "
    "`remove_session` deletes it together with its placeholder mapping."
)

ANONYMIZE = (
    "Anonymize personal or sensitive data (names, addresses, phone numbers, e-mails, "
    "companies, dates, ...) in a text or a document. Pass either `text`, or a document as "
    "`file_base64` plus `file_name` (docx, pdf, pptx, xlsx, txt, html, xml, zip, ...). "
    "Returns the anonymized `text` (for documents also the anonymized document as "
    "`file_base64` with its `file_name`) and the keys `pseudo_key` and `encrypt_key`, "
    "which `deanonymize` needs to restore it - keep them. Pass `session` (from "
    "`add_session`) to get the same placeholders across several calls and to restore "
    "with the session instead of keys. Without `settings` or `settings_id` the "
    "installation's default setting is used."
)

DEANONYMIZE = (
    "Restore the original data in a text or document anonymized by `anonymize`. Pass "
    "either `text`, or the anonymized document as `file_base64` plus `file_name`. To "
    "restore, pass `pseudo_key` and `encrypt_key` from the result of `anonymize`, or the "
    "`session` it was anonymized with. With a session, any text containing the session's "
    "placeholders can be restored, e.g. a reply written by a model. Returns the restored "
    "`text` (for documents also `file_base64` and `file_name`)."
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
    "Create a session, so that several `anonymize` calls use the same placeholders for "
    "the same values (e.g. within a conversation) and `deanonymize` can restore any text "
    "with the session alone. Needs `settings_id`, the id of a stored setting (see "
    "`list_settings` or `add_setting`). Returns the session with its `id` - pass it as "
    "`session` to `anonymize` and `deanonymize`."
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


class GlanosAnonymizationTools:
    usage_guide = USAGE_GUIDE

    def __init__(self, token: str, base_url: str, no_ssl_verify: bool = False) -> None:
        self.client = GlanosClient(
            token=token, base_url=base_url, no_ssl_verify=no_ssl_verify
        )

    def close(self) -> None:
        self.client.close()

    def anonymize(
        self,
        text: Annotated[
            str | None, Field(description="Text to anonymize (instead of a file).")
        ] = None,
        file_base64: Annotated[
            str | None,
            Field(
                description="Document to anonymize, base64-encoded (instead of text)."
            ),
        ] = None,
        file_name: Annotated[
            str | None,
            Field(
                description="File name of the document, with its extension "
                "(e.g. contract.docx). Required with file_base64."
            ),
        ] = None,
        settings: Annotated[
            AnonymizationSettings | None,
            Field(
                description="Settings for this call only: mode, entity types to leave "
                "untouched (excludeFields), ... Takes precedence over settings_id."
            ),
        ] = None,
        settings_id: Annotated[
            str | None, Field(description="Id of a stored setting to use.")
        ] = None,
        session: Annotated[
            str | None,
            Field(
                description="Id of a session from add_session, for the same "
                "placeholders across several calls."
            ),
        ] = None,
        language: Annotated[
            str | None,
            Field(
                description="ISO 639-1 language code (e.g. de, en); detected "
                "automatically if left out."
            ),
        ] = None,
        data_retention_ms: Annotated[
            int | None,
            Field(
                description="How long the anonymized document stays on the server in "
                "milliseconds, -1 for ever (default 0: removed after download)."
            ),
        ] = None,
        pseudo_key_retention_ms: Annotated[
            int | None,
            Field(
                description="How long the pseudo key stays on the server in "
                "milliseconds, -1 for ever (default 0)."
            ),
        ] = None,
        meta_retention_ms: Annotated[
            int | None,
            Field(
                description="How long the meta data stays on the server in "
                "milliseconds, -1 for ever (default 0). Documents only: while it is "
                "kept, meta_key alone restores the document."
            ),
        ] = None,
        assignee: Annotated[
            str | None,
            Field(description="User to assign the document to for review."),
        ] = None,
        assigned_group: Annotated[
            str | None,
            Field(description="Group to assign the document to for review."),
        ] = None,
        options: Annotated[
            list[str] | None,
            Field(description="Installation-specific upload options."),
        ] = None,
    ) -> dict[str, Any]:
        result = self.client.anonymize(
            text=text,
            data=base64.b64decode(file_base64) if file_base64 is not None else None,
            filename=file_name,
            settings=settings,
            settings_id=settings_id,
            session=session,
            language=language,
            data_retention_ms=data_retention_ms,
            pseudo_key_retention_ms=pseudo_key_retention_ms,
            meta_retention_ms=meta_retention_ms,
            assignee=assignee,
            assigned_group=assigned_group,
            options=options,
        )
        data = result.pseudo_result.data
        assert isinstance(data, InnerResponseData)
        output: dict[str, Any] = {"text": result.text}
        if file_base64 is not None:
            output["file_base64"] = data.data
            output["file_name"] = data.fileName
        output.update(
            pseudo_key=data.pseudoKey,
            encrypt_key=data.encryptKey,
            meta_key=data.metaKey,
            session=data.session,
        )
        return output

    def deanonymize(
        self,
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
        pseudo_key: Annotated[
            str | None, Field(description="pseudo_key from the result of anonymize.")
        ] = None,
        encrypt_key: Annotated[
            str | None, Field(description="encrypt_key from the result of anonymize.")
        ] = None,
        meta_key: Annotated[
            str | None,
            Field(
                description="meta_key from the result of anonymize (documents only); "
                "not needed when pseudo_key and encrypt_key are given."
            ),
        ] = None,
        session: Annotated[
            str | None,
            Field(
                description="Session the text or document was anonymized with "
                "(instead of keys)."
            ),
        ] = None,
    ) -> dict[str, Any]:
        if pseudo_key and encrypt_key:
            meta_key = None
        keys = (
            DepseudoPaths(
                pseudo_key=pseudo_key, encrypt_key=encrypt_key, meta_key=meta_key
            )
            if pseudo_key or encrypt_key or meta_key
            else None
        )
        if file_base64 is None:
            if text is None:
                raise ValueError("Provide 'text', or 'file_base64' with 'file_name'")
            return {
                "text": self.client.depseudo_text(
                    text, session=session, depseudo_paths=keys
                )
            }
        if text is not None:
            raise ValueError("Provide either 'text' or 'file_base64', not both")
        restored = self.client.depseudo(
            data=base64.b64decode(file_base64),
            filename=file_name,
            depseudo_paths=keys,
            session=session,
        )
        data = restored.data
        assert isinstance(data, InnerResponseData)
        return {
            "text": self.client.get_ano_result(restored).text,
            "file_base64": data.data,
            "file_name": data.fileName,
        }

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
    def tools(self) -> list[StructuredTool]:
        return [
            StructuredTool.from_function(
                func=_reporting_errors(method),
                name=name,
                description=description,
                handle_tool_error=True,
            )
            for name, description, method in self.definitions()
        ]


def _reporting_errors(
    method: Callable[..., dict[str, Any]],
) -> Callable[..., dict[str, Any]]:
    @functools.wraps(method)
    def run(*args: Any, **kwargs: Any) -> dict[str, Any]:
        try:
            return method(*args, **kwargs)
        except GlanosAPIError as exc:
            raise ToolException(
                f"HTTP {exc.status_code}: {exc.api_message or 'request failed'}"
            ) from exc
        except (GlanosClientError, ValueError) as exc:
            raise ToolException(str(exc)) from exc

    return run
