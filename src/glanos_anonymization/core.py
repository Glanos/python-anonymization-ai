import os
import asyncio
import base64
import builtins
import dataclasses
import functools
import math
import httpx
from enum import Enum
from typing import Annotated, Literal, Union, Protocol
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pathlib import Path
import json
import tempfile
import time

DEFAULT_TIMEOUT = 30.0
# anonymize(processing="poll"): enough for the server to process a document
DEFAULT_POLL_INTERVAL = 10.0
DEFAULT_POLL_MAX_ATTEMPTS = 30
# fetch(): long enough for a person to review a document
DEFAULT_FETCH_POLL_INTERVAL = 300.0
DEFAULT_FETCH_MAX_ATTEMPTS = None

API_BASE_PATH = "/be/api-v2"


def _missing_extra(feature: str, extra: str) -> ImportError:
    return ImportError(
        f"{feature} needs the optional '{extra}' dependencies: "
        f"pip install 'glanos-anonymization[{extra}]'"
    )


class GlanosClientError(Exception):
    pass


class DepseudoKeyFileError(GlanosClientError):
    pass


class ResultNotReadyError(GlanosClientError):
    def __init__(self, fetch_key: str) -> None:
        super().__init__(
            "The document is not processed yet; collect it with "
            f"fetch() and its fetch_key {fetch_key!r}"
        )
        self.fetch_key = fetch_key


class SettingNotFoundError(GlanosClientError):
    pass


class ProcessingFailedError(GlanosClientError):
    def __init__(self, message: str, fail_reason: str | None = None) -> None:
        super().__init__(message)
        self.fail_reason = fail_reason


class GlanosAPIError(GlanosClientError, httpx.HTTPStatusError):
    def __init__(self, message: str, *, response: httpx.Response) -> None:
        httpx.HTTPStatusError.__init__(
            self, message, request=response.request, response=response
        )
        self.status_code = response.status_code
        self.api_message = _api_message(response)


class InvalidInputError(GlanosAPIError):
    pass


class AuthenticationError(InvalidInputError):
    pass


class SettingsConflictError(InvalidInputError):
    pass


class RateLimitedError(GlanosAPIError):
    pass


class QuotaExceededError(GlanosAPIError):
    pass


class DataExpiredError(GlanosAPIError):
    pass


class ServerError(GlanosAPIError):
    pass


def _is_settings_conflict(response: httpx.Response) -> bool:
    message = (_api_message(response) or "").lower()
    return "settings mode" in message and "must be equal" in message


def _api_message(response: httpx.Response) -> str | None:
    try:
        body = response.json()
    except ValueError:
        return None
    if isinstance(body, dict) and isinstance(body.get("message"), str):
        return body["message"]
    return None


class AnonymizationMode(str, Enum):
    TYPE = "TYPE"
    TYPE_PRESERVING = "TYPE_PRESERVING"
    XXX = "XXX"
    MT = "MT"


class ExcludeField(str, Enum):
    PERSON_NAMES = "personNames"
    STREETS = "streets"
    LOCATIONS = "locations"
    COUNTRIES = "countries"
    ADDRESSES = "addresses"
    MAILS = "mails"
    COMPANIES = "companies"
    ORGS = "orgs"
    PHONES = "phones"
    HOMEPAGES = "homepages"
    FILEPATHS = "filepaths"
    LEGALS = "legals"
    DATES = "dates"
    SIMPLE_NUMBERS = "simpleNumbers"
    BANKS = "banks"
    MONEY_AMOUNTS = "moneyAmounts"
    ID_NUMBERS = "idNumbers"
    PERCENTS = "percents"
    MEASURES = "measures"
    PRODUCTS = "products"
    IMAGES = "images"
    DOC_META = "docMeta"


class BlankOutMode(str, Enum):
    XXX = "XXX"
    BLACK = "BLACK"
    XLEN = "XLEN"


class FoldMode(str, Enum):
    STRICT = "STRICT"
    MINOR_VAR = "MINOR_VAR"
    MINOR_VAR_CASING = "MINOR_VAR_CASING"
    MINOR_VAR_AMBIG = "MINOR_VAR_AMBIG"
    SAME_ENTITY = "SAME_ENTITY"


class ImageReplaceMode(str, Enum):
    DUMMY = "DUMMY"
    BLANK = "BLANK"
    TRANSPARENT = "TRANSPARENT"
    AUTO = "AUTO"


class DownloadOption(str, Enum):
    PDF_AS_PDF = "PDF_AS_PDF"


class Processing(str, Enum):
    """How anonymize() waits for the anonymized document."""

    # One request that waits for the document; pending if the server
    # did not finish in time.
    SYNC = "sync"
    # Submit, then poll until the document is ready or the attempts run out.
    POLL = "poll"
    # Submit and return at once with the fetch key.
    NO_WAIT = "no_wait"


class SettingsSort(str, Enum):
    NAME = "name"
    CREATED_AT = "createdAt"


class SortOrder(str, Enum):
    ASC = "asc"
    DESC = "desc"


def _unwrap_data_envelope(value):
    if isinstance(value, dict) and "data" in value and "success" in value:
        return value["data"]
    return value


OpenExcludeField = Annotated[ExcludeField | str, Field(union_mode="left_to_right")]
OpenAnonymizationMode = Annotated[
    AnonymizationMode | str, Field(union_mode="left_to_right")
]
OpenBlankOutMode = Annotated[BlankOutMode | str, Field(union_mode="left_to_right")]
OpenFoldMode = Annotated[FoldMode | str, Field(union_mode="left_to_right")]
OpenImageReplaceMode = Annotated[
    ImageReplaceMode | str, Field(union_mode="left_to_right")
]


class AnonymizationSettings(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str | None = None
    name: str | None = None
    excludeFields: list[OpenExcludeField] = Field(default_factory=list)
    mode: OpenAnonymizationMode = AnonymizationMode.TYPE
    workflow: str | None = None
    allowAmbigReplacement: bool | None = Field(
        default=None, description="Deprecated by the API in favor of foldMode."
    )
    addNoTranslateMarkup: bool | None = None
    preserveLength: bool | None = Field(
        default=None, description="Deprecated by the API in favor of blankOutMode."
    )
    blankOutMode: OpenBlankOutMode | None = None
    foldMode: OpenFoldMode | None = FoldMode.MINOR_VAR
    imageReplace: OpenImageReplaceMode | None = ImageReplaceMode.AUTO
    custom: list[str] = Field(default_factory=list)


class ApiUploadDocument(BaseModel):
    data: str | None = None
    plainText: str | None = None
    language: str | None = None
    fileName: str
    settings: AnonymizationSettings | None = None
    settingsId: str | None = None
    dataRetentionMS: int | None = None
    pseudoKeyRetentionMS: int | None = None
    metaRetentionMS: int | None = None
    assignee: str | None = None
    assignedGroup: str | None = None
    session: str | None = None
    sessionMaxAge: int | None = None
    options: list[str] | None = None

    @model_validator(mode="after")
    def _validate_content(self) -> "ApiUploadDocument":
        if self.data is None and self.plainText is None:
            raise ValueError("Either 'data' or 'plainText' is required")
        return self


class ApiUploadDepseudoDocument(BaseModel):
    data: str | None = None
    plainText: str | None = None
    pseudoKey: str | None = None
    encryptKey: str | None = None
    metaKey: str | None = None
    fileName: str
    session: str | None = None

    @model_validator(mode="after")
    def _validate_content(self) -> "ApiUploadDepseudoDocument":
        if self.data is None and self.plainText is None:
            raise ValueError("Either 'data' or 'plainText' is required")
        return self


class InnerResponseData(BaseModel):
    data: str
    fetchKey: str | None = None
    metaKey: str | None = None
    encryptKey: str | None = None
    pseudoKey: str | None = None
    fileName: str
    uploadName: str | None = None
    language: str | None = None
    success: bool | None = None
    failReason: str | None = None
    session: str | None = None


class FetchKeyData(BaseModel):
    fetchKey: str


class PseudoResult(BaseModel):
    success: bool
    data: Union[InnerResponseData, FetchKeyData]
    message: str | None = None

    @property
    def is_pending(self) -> bool:
        return isinstance(self.data, FetchKeyData)


class SuccessMessage(BaseModel):
    @model_validator(mode="before")
    @classmethod
    def _unwrap(cls, value):
        if isinstance(value, dict) and "data" in value:
            return value["data"]
        return value

    success: bool


class AnonymizationSettingsList(BaseModel):
    limit: int | None = None
    offset: int | None = None
    total: int | None = None
    list: builtins.list[AnonymizationSettings] = Field(default_factory=builtins.list)


class SessionInfo(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    name: str | None = None
    user: str | None = None
    shareUsers: list[str] = Field(default_factory=list)
    shareGroups: list[str] = Field(default_factory=list)
    createdAt: int | None = None
    lastUpdatedAt: int | None = None
    expiresAt: int | None = None


class AnonymizationSessionAddData(BaseModel):
    name: str | None = None
    settingsId: str | None = None
    shareUsers: list[str] | None = None
    shareGroups: list[str] | None = None
    sessionMaxAge: int | None = None


SourceType = Literal["string", "file"]


class InputData(BaseModel):
    content: str | bytes
    source: SourceType
    filename: str | None = None


class DepseudoPaths(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pseudo_key_path: str | None = None
    pseudo_key: str | None = None
    meta_key: str | None = None
    encrypt_key_path: str | None = None
    encrypt_key: str | None = None

    @model_validator(mode="after")
    def _one_source_per_key(self) -> "DepseudoPaths":
        if self.pseudo_key_path and self.pseudo_key:
            raise ValueError("Pass pseudo_key_path or pseudo_key, not both")
        if self.encrypt_key_path and self.encrypt_key:
            raise ValueError("Pass encrypt_key_path or encrypt_key, not both")
        return self

    @classmethod
    def from_result(cls, pseudo_result: "PseudoResult") -> "DepseudoPaths":
        data = pseudo_result.data
        if not isinstance(data, InnerResponseData):
            raise ValueError("pseudo_result does not contain a completed result")
        if data.pseudoKey and data.encryptKey:
            return cls(pseudo_key=data.pseudoKey, encrypt_key=data.encryptKey)
        if data.metaKey:
            return cls(meta_key=data.metaKey, pseudo_key=data.pseudoKey)
        raise ValueError(
            "The result carries no keys to restore it (request them with "
            "return_pseudo_key/return_encrypt_key, or use a session)"
        )


class AnonymizationResult:
    """The document returned by the API, with the keys to restore it.

    The document is kept as returned: `data` holds its bytes. Its text is only
    extracted when `extract_text()` is called.

    A result can be pending (`ready` is False): the server has not finished
    the document yet. It then only carries the `fetch_key` to collect it with
    `fetch()`; reading anything else raises ResultNotReadyError.
    """

    def __init__(
        self,
        pseudo_result: PseudoResult,
        text_extractor: "TextExtractor | None" = None,
    ):
        self.pseudo_result = pseudo_result
        self._text_extractor = text_extractor

    @property
    def ready(self) -> bool:
        return isinstance(self.pseudo_result.data, InnerResponseData)

    @property
    def fetch_key(self) -> str | None:
        return self.pseudo_result.data.fetchKey

    @property
    def data(self) -> bytes:
        return base64.b64decode(self._completed_data().data)

    @property
    def file_name(self) -> str:
        return self._completed_data().fileName

    @property
    def pseudo_key(self) -> str | None:
        return self._completed_data().pseudoKey

    @property
    def encrypt_key(self) -> str | None:
        return self._completed_data().encryptKey

    @property
    def meta_key(self) -> str | None:
        return self._completed_data().metaKey

    @property
    def session(self) -> str | None:
        return self._completed_data().session

    def extract_text(self, extractor: "TextExtractor | None" = None) -> str:
        """Return the text of the document.

        Text results are decoded directly; other documents go through
        `extractor`, else the client's text extractor (MarkItDown by default).
        """
        data = self._completed_data()
        if _is_text_file(data.fileName):
            return _decode_text(data.data)
        extractor = extractor or self._text_extractor or _default_text_extractor()
        return extractor.extract_text(self.pseudo_result)

    def _completed_data(self) -> "InnerResponseData":
        data = self.pseudo_result.data
        if not isinstance(data, InnerResponseData):
            raise ResultNotReadyError(data.fetchKey)
        return data


class DeanonymizationResult(AnonymizationResult):
    pass


class TextExtractor(Protocol):
    def extract_text(self, pseudo_result: PseudoResult) -> str: ...


class MarkItDownTextExtractor(TextExtractor):
    def __init__(self):
        try:
            from markitdown import MarkItDown
        except ImportError as exc:
            raise _missing_extra(
                "Extracting text from documents", "text-extraction"
            ) from exc

        self.mid = MarkItDown()

    def extract_text(self, pseudo_result: PseudoResult) -> str:
        text_data = base64.b64decode(pseudo_result.data.data)
        suffix = Path(pseudo_result.data.fileName).suffix

        with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
            tmp.write(text_data)
            tmp.flush()

            text_content = self.mid.convert(tmp.name).text_content

        return text_content


@functools.cache
def _default_text_extractor() -> TextExtractor:
    return MarkItDownTextExtractor()


def _is_text_file(file_name: str) -> bool:
    return Path(file_name).suffix.lower() == ".txt"


def _decode_text(data: str) -> str:
    # Returned as the API sends it: every plain-text result ends with a newline,
    # whether or not the input had one.
    return base64.b64decode(data).decode("utf-8")


def _input_data(
    text: str | None,
    path: str | Path | None,
    data: bytes | None,
    filename: str | None,
) -> "InputData":
    if sum(value is not None for value in (text, path, data)) != 1:
        raise ValueError("Provide exactly one of 'text', 'path' or 'data'")
    if data is not None:
        if not filename:
            raise ValueError("'data' needs a 'filename' with the correct extension")
        return InputData(content=data, source="file", filename=filename)
    if path is not None:
        path = Path(path)
        return InputData(
            content=path.read_bytes(),
            source="file",
            filename=filename or os.path.basename(path),
        )
    return InputData(
        content=text, source="string", filename=filename or "plain_text.txt"
    )


def _is_plain_text(data: "InnerResponseData") -> bool:
    return (
        data.uploadName is not None
        and data.uploadName.startswith("plain_text")
        and data.fileName.startswith("plain_text")
    )


def _check_polling(poll_interval: float, max_attempts: int | None) -> None:
    if poll_interval < 0:
        raise ValueError("poll_interval must not be negative")
    if max_attempts is not None and max_attempts < 1:
        raise ValueError("max_attempts must be at least 1, or None for no limit")


def _default_keys(
    return_pseudo_key: bool | None, return_encrypt_key: bool | None
) -> tuple[bool, bool]:
    return (
        True if return_pseudo_key is None else return_pseudo_key,
        True if return_encrypt_key is None else return_encrypt_key,
    )


@dataclasses.dataclass(frozen=True)
class _ApiRequest:
    method: str
    url: str
    params: dict
    json_body: dict | None = None

    def httpx_kwargs(self) -> dict:
        kwargs: dict = {"params": self.params}
        if self.json_body is not None:
            kwargs["content"] = json.dumps(self.json_body)
            kwargs["headers"] = {"Content-Type": "application/json"}
        return kwargs


class _GlanosClientBase:
    """Builds the API requests and parses the responses.

    GlanosClient and AsyncGlanosClient share all of it and only add how a
    request is sent; each of their methods is: build the request, send it,
    parse the response.
    """

    # For a text the server did not finish within a sync request: fetch() it
    # like anonymize(processing="poll") would, without the keys.
    _COLLECT_IN_TIME = {
        "poll_interval": DEFAULT_POLL_INTERVAL,
        "max_attempts": DEFAULT_POLL_MAX_ATTEMPTS,
        "return_pseudo_key": False,
        "return_encrypt_key": False,
    }

    _STATUS_EXCEPTIONS: dict[int, type[GlanosAPIError]] = {
        401: AuthenticationError,
        402: QuotaExceededError,
        403: AuthenticationError,
        410: DataExpiredError,
        429: RateLimitedError,
    }

    def __init__(
        self,
        token: str,
        base_url: str,
        text_extractor: TextExtractor | None = None,
    ) -> None:
        self.token = token
        self.base_url = base_url.rstrip("/")
        self.text_extractor = text_extractor

    @property
    def text_extractor(self) -> TextExtractor:
        return self._text_extractor or _default_text_extractor()

    @text_extractor.setter
    def text_extractor(self, extractor: TextExtractor | None):
        self._text_extractor = extractor

    def _raise_for_status(self, response: httpx.Response) -> None:
        if not response.is_error:
            return

        exc_type = self._STATUS_EXCEPTIONS.get(response.status_code)
        if exc_type is None:
            exc_type = ServerError if response.is_server_error else InvalidInputError
        if exc_type is InvalidInputError and _is_settings_conflict(response):
            exc_type = SettingsConflictError

        url = response.url.copy_with(query=None)
        raise exc_type(
            f"HTTP {response.status_code} for {url}: {response.text}",
            response=response,
        )

    def _url(self, path: str) -> str:
        return f"{self.base_url}{API_BASE_PATH}{path}"

    def _api_request(
        self,
        method: str,
        url: str,
        params: dict | None = None,
        json_body: dict | None = None,
    ) -> _ApiRequest:
        return _ApiRequest(
            method, url, {"token": self.token, **(params or {})}, json_body
        )

    @staticmethod
    def _to_json_dict(model: BaseModel) -> dict:
        return model.model_dump(mode="json", exclude_none=True)

    @staticmethod
    def _parse_pseudo_response(response: httpx.Response) -> PseudoResult:

        body = response.json()
        data = body.get("data") or {}

        pending = response.status_code == 202 or (
            "fetchKey" in data and "data" not in data
        )
        if pending:
            parsed_data: InnerResponseData | FetchKeyData = FetchKeyData.model_validate(
                data
            )
        else:
            if data.get("success") is False:
                fail_reason = data.get("failReason")
                raise ProcessingFailedError(
                    "Glanos anonymization.ai could not process the document"
                    + (f": {fail_reason}" if fail_reason else ""),
                    fail_reason=fail_reason,
                )
            parsed_data = InnerResponseData.model_validate(data)

        return PseudoResult(
            success=body.get("success", True),
            data=parsed_data,
            message=body.get("message"),
        )

    def build_pseudo_request_url(self) -> str:
        return f"{self.base_url}{API_BASE_PATH}/document/pseudo"

    @staticmethod
    def _inline_settings_body(settings: AnonymizationSettings) -> dict:
        body = settings.model_dump(mode="json", exclude_none=True)
        return {
            key: value
            for key, value in body.items()
            if key in AnonymizationSettings.model_fields and key not in ("id", "name")
        }

    def build_pseudo_request_body(
        self,
        input_data: InputData,
        settings: AnonymizationSettings | None = None,
        settings_id: str | None = None,
        session: str | None = None,
        session_max_age_ms: int | None = None,
        language: str | None = None,
        data_retention_ms: int | None = None,
        pseudo_key_retention_ms: int | None = None,
        meta_retention_ms: int | None = None,
        assignee: str | None = None,
        assigned_group: str | None = None,
        options: list[str] | None = None,
    ) -> dict:
        if settings is not None and settings_id is not None:
            raise ValueError(
                "Pass either 'settings' or 'settings_id', not both (the API would "
                "ignore 'settings_id')"
            )

        upload = ApiUploadDocument(
            data=base64.b64encode(input_data.content).decode("utf-8")
            if input_data.source == "file"
            else None,
            plainText=input_data.content if input_data.source == "string" else None,
            language=language,
            fileName=input_data.filename,
            settingsId=settings_id,
            dataRetentionMS=data_retention_ms,
            pseudoKeyRetentionMS=pseudo_key_retention_ms,
            metaRetentionMS=meta_retention_ms,
            assignee=assignee,
            assignedGroup=assigned_group,
            session=session,
            sessionMaxAge=session_max_age_ms,
            options=options,
        )

        body = self._to_json_dict(upload)
        if settings is not None:
            body["settings"] = self._inline_settings_body(settings)
        return body

    def _pseudo_request(
        self,
        input_data: InputData,
        *,
        server_async: bool,
        return_pseudo_key: bool | None = None,
        return_encrypt_key: bool | None = None,
        **body_options,
    ) -> _ApiRequest:
        params = {"async": str(server_async).lower()}
        # An asynchronous request gets its keys from the fetch request.
        if not server_async:
            want_pseudo_key, want_encrypt_key = _default_keys(
                return_pseudo_key, return_encrypt_key
            )
            params["returnPseudoKey"] = str(want_pseudo_key).lower()
            params["returnEncryptKey"] = str(want_encrypt_key).lower()

        return self._api_request(
            "POST",
            self.build_pseudo_request_url(),
            params,
            self.build_pseudo_request_body(input_data, **body_options),
        )

    @staticmethod
    def _anonymize_plan(
        processing: Processing | str | None,
        input_data: InputData,
        poll_interval: float | None,
        max_attempts: int | None,
        return_pseudo_key: bool | None,
        return_encrypt_key: bool | None,
    ) -> tuple[Processing, dict | None]:
        """Check the options of anonymize() before anything is sent.

        Returns the processing and, for "poll", the arguments for fetch().
        """
        if processing is None:
            # A text is usually done within one request, a document may take
            # longer.
            processing = (
                Processing.SYNC if input_data.source == "string" else Processing.POLL
            )
        processing = Processing(processing)

        if processing is not Processing.POLL:
            if poll_interval is not None or max_attempts is not None:
                raise ValueError(
                    "poll_interval and max_attempts only apply to processing='poll'"
                )
            if processing is Processing.NO_WAIT and (
                return_pseudo_key is not None or return_encrypt_key is not None
            ):
                raise ValueError(
                    "With processing='no_wait' the keys are requested when the "
                    "result is collected: pass return_pseudo_key/return_encrypt_key "
                    "to fetch()"
                )
            return processing, None

        want_pseudo_key, want_encrypt_key = _default_keys(
            return_pseudo_key, return_encrypt_key
        )
        polling = {
            "poll_interval": DEFAULT_POLL_INTERVAL
            if poll_interval is None
            else poll_interval,
            "max_attempts": DEFAULT_POLL_MAX_ATTEMPTS
            if max_attempts is None
            else max_attempts,
            "return_pseudo_key": want_pseudo_key,
            "return_encrypt_key": want_encrypt_key,
        }
        _check_polling(polling["poll_interval"], polling["max_attempts"])
        return processing, polling

    def _review_request(
        self,
        input_data: InputData,
        assignee: str | None,
        assigned_group: str | None,
        **body_options,
    ) -> _ApiRequest:
        if bool(assignee) == bool(assigned_group):
            raise ValueError("Pass either 'assignee' or 'assigned_group' (exactly one)")
        return self._pseudo_request(
            input_data,
            server_async=True,
            assignee=assignee,
            assigned_group=assigned_group,
            **body_options,
        )

    def build_fetch_request_url(self) -> str:
        return f"{self.base_url}{API_BASE_PATH}/document/pseudo/fetch"

    def _fetch_request(
        self,
        fetch_key: str,
        start_retention_period: bool,
        download_options: DownloadOption | None,
        return_pseudo_key: bool,
        return_encrypt_key: bool,
    ) -> _ApiRequest:
        params = {
            "fetchKey": fetch_key,
            "returnPseudoKey": str(return_pseudo_key).lower(),
            "returnEncryptKey": str(return_encrypt_key).lower(),
            "startRetentionPeriod": str(start_retention_period).lower(),
        }
        if download_options is not None:
            params["downloadOptions"] = download_options.value
        return self._api_request("GET", self.build_fetch_request_url(), params)

    @staticmethod
    def _pending_fetch_key(pending: AnonymizationResult | str) -> str | None:
        if isinstance(pending, str):
            return pending
        return None if pending.ready else pending.fetch_key

    @staticmethod
    def _note_fetch_key(exc: Exception, fetch_key: str) -> None:
        # Polling stops at the first error; the fetch key lets the caller
        # pick it up again.
        exc.add_note(f"fetch_key: {fetch_key} (pass it to fetch() to try again)")

    def _result(self, pseudo_result: PseudoResult) -> AnonymizationResult:
        return AnonymizationResult(pseudo_result, self._text_extractor)

    def build_depseudo_request_url(self) -> str:
        return f"{self.base_url}{API_BASE_PATH}/document/depseudo"

    def _read_key_file(self, key_path: str, key_label: str) -> str:
        if not os.path.isfile(key_path):
            raise FileNotFoundError(
                f"{key_label} does not exist or is not a file: {key_path}"
            )

        try:
            with open(key_path) as f:
                return f.read()
        except OSError as e:
            raise DepseudoKeyFileError(
                f"There was an error while trying to open the {key_label} at {key_path}: {e}"
            ) from e

    def build_depseudo_request_body(
        self,
        input_data: InputData,
        depseudo_paths: DepseudoPaths | None = None,
        session: str | None = None,
    ) -> dict:

        depseudo_paths = depseudo_paths or DepseudoPaths()

        upload = ApiUploadDepseudoDocument(
            data=base64.b64encode(input_data.content).decode("utf-8")
            if input_data.source == "file"
            else None,
            plainText=input_data.content if input_data.source == "string" else None,
            fileName=input_data.filename,
            pseudoKey=self._read_key_file(
                depseudo_paths.pseudo_key_path, "pseudo key file"
            )
            if depseudo_paths.pseudo_key_path
            else depseudo_paths.pseudo_key,
            encryptKey=self._read_key_file(
                depseudo_paths.encrypt_key_path, "encrypt key file"
            )
            if depseudo_paths.encrypt_key_path
            else depseudo_paths.encrypt_key,
            metaKey=depseudo_paths.meta_key,
            session=session,
        )

        return self._to_json_dict(upload)

    def _depseudo_request(
        self,
        input_data: InputData,
        depseudo_paths: DepseudoPaths | None,
        session: str | None,
    ) -> _ApiRequest:
        return self._api_request(
            "POST",
            self.build_depseudo_request_url(),
            {"outputMode": "json"},
            self.build_depseudo_request_body(
                input_data=input_data, depseudo_paths=depseudo_paths, session=session
            ),
        )

    def _deanonymize_request(
        self,
        result: AnonymizationResult | None,
        text: str | None,
        path: str | Path | None,
        data: bytes | None,
        filename: str | None,
        session: str | None,
        depseudo_paths: DepseudoPaths | None,
    ) -> _ApiRequest:
        if result is None:
            if session is None and depseudo_paths is None:
                raise ValueError(
                    "Restoring 'text', 'path' or 'data' needs a 'session' or the "
                    "keys ('depseudo_paths')"
                )
            return self._depseudo_request(
                _input_data(text, path, data, filename), depseudo_paths, session
            )

        if any(value is not None for value in (text, path, data, filename)):
            raise ValueError(
                "Pass either a result or 'text'/'path'/'data' to restore, not both"
            )
        completed = result._completed_data()
        if session is None and depseudo_paths is None:
            try:
                depseudo_paths = DepseudoPaths.from_result(result.pseudo_result)
            except ValueError:
                if not completed.session:
                    raise
                session = completed.session

        content = base64.b64decode(completed.data)
        if _is_plain_text(completed):
            input_data = InputData(
                content=content.decode("utf-8"),
                source="string",
                filename=completed.fileName,
            )
        else:
            input_data = InputData(
                content=content, source="file", filename=completed.fileName
            )
        return self._depseudo_request(input_data, depseudo_paths, session)

    def _deanonymization_result(self, restored: PseudoResult) -> DeanonymizationResult:
        if not isinstance(restored.data, InnerResponseData):
            raise ValueError("depseudo did not return a completed result")

        return DeanonymizationResult(restored, self._text_extractor)

    @staticmethod
    def _parse_session_response(response: httpx.Response) -> SessionInfo:
        body = response.json()
        if isinstance(body, dict) and body.get("success") is False:
            raise GlanosClientError(
                f"Session request failed: {body.get('message') or 'no reason given'}"
            )
        return SessionInfo.model_validate(_unwrap_data_envelope(body))

    @staticmethod
    def _parse_success(response: httpx.Response) -> SuccessMessage:
        return SuccessMessage.model_validate(response.json())

    def _add_session_request(
        self,
        max_age_seconds: int,
        settings_id: str,
        name: str | None,
        share_users: list[str] | None,
        share_groups: list[str] | None,
    ) -> _ApiRequest:
        body = AnonymizationSessionAddData(
            name=name,
            settingsId=settings_id,
            shareUsers=share_users,
            shareGroups=share_groups,
            sessionMaxAge=max_age_seconds,
        )
        return self._api_request(
            "POST", self._url("/session/add"), json_body=self._to_json_dict(body)
        )

    @staticmethod
    def _session_max_age_seconds(max_age_ms: int) -> int:
        return max(1, math.ceil(max_age_ms / 1000))

    @staticmethod
    def _session_of(result: AnonymizationResult) -> str:
        if not result.session:
            raise GlanosClientError(
                "Glanos anonymization.ai did not return a session id despite session_max_age_ms being set"
            )
        return result.session

    def _get_session_request(self, session_id: str) -> _ApiRequest:
        return self._api_request(
            "GET", self._url("/session/get"), {"session": session_id}
        )

    def _extend_session_request(
        self, session_id: str, max_age_seconds: int
    ) -> _ApiRequest:
        return self._api_request(
            "PUT",
            self._url("/session/extend"),
            {"session": session_id, "sessionMaxAge": max_age_seconds},
        )

    def _remove_session_request(self, session_id: str) -> _ApiRequest:
        return self._api_request(
            "PUT", self._url("/session/remove"), {"session": session_id}
        )

    @staticmethod
    def _parse_setting(response: httpx.Response) -> AnonymizationSettings:
        return AnonymizationSettings.model_validate(
            _unwrap_data_envelope(response.json())
        )

    def _get_setting_request(self, setting_id: str) -> _ApiRequest:
        return self._api_request(
            "GET", self._url("/setting/get"), {"setting": setting_id}
        )

    def _list_settings_request(
        self,
        query: str | None,
        creator: str | list[str] | None,
        sort: SettingsSort | None,
        sort_order: SortOrder | None,
        limit: int | None,
        offset: int | None,
    ) -> _ApiRequest:
        params: dict = {}
        if query is not None:
            params["query"] = query
        if creator is not None:
            params["creator"] = (
                creator if isinstance(creator, str) else ",".join(creator)
            )
        if sort is not None:
            params["sort"] = SettingsSort(sort).value
        if sort_order is not None:
            params["sortOrder"] = SortOrder(sort_order).value
        if limit is not None:
            params["limit"] = limit
        if offset is not None:
            params["offset"] = offset

        return self._api_request("GET", self._url("/setting/list"), params)

    @staticmethod
    def _parse_settings_list(response: httpx.Response) -> AnonymizationSettingsList:
        return AnonymizationSettingsList.model_validate(
            _unwrap_data_envelope(response.json())
        )

    @staticmethod
    def _stored_setting_body(settings: AnonymizationSettings) -> dict:
        body = settings.model_dump(mode="json", exclude_none=True)
        return {
            key: value
            for key, value in body.items()
            if key in AnonymizationSettings.model_fields
        }

    def _write_setting_request(
        self, endpoint: str, settings: AnonymizationSettings, apply_defaults: bool
    ) -> _ApiRequest:
        return self._api_request(
            "POST",
            self._url(f"/setting/{endpoint}"),
            {"applyDefaults": str(apply_defaults).lower()},
            self._stored_setting_body(settings),
        )

    def _add_setting_request(
        self, settings: AnonymizationSettings, apply_defaults: bool
    ) -> _ApiRequest:
        if settings.id is not None:
            raise ValueError(
                "add_setting() creates a new setting and the server assigns its id; "
                "use update_setting() to change the setting with id "
                f"{settings.id!r}"
            )
        if not settings.name:
            raise ValueError("A new setting needs a name (settings.name)")
        return self._write_setting_request("add", settings, apply_defaults)

    def _update_setting_request(
        self, settings: AnonymizationSettings, apply_defaults: bool
    ) -> _ApiRequest:
        if not settings.id:
            raise ValueError(
                "update_setting() changes an existing setting and needs its id "
                "(settings.id); use add_setting() to create a new one"
            )
        return self._write_setting_request("update", settings, apply_defaults)

    @staticmethod
    def _setting_not_found(
        exc: GlanosAPIError, settings: AnonymizationSettings
    ) -> SettingNotFoundError | None:
        if (exc.api_message or "").startswith("No such settings"):
            return SettingNotFoundError(f"No stored setting with id {settings.id!r}")
        return None

    def _remove_setting_request(self, setting_id: str) -> _ApiRequest:
        return self._api_request(
            "PUT", self._url("/setting/remove"), {"setting": setting_id}
        )


class GlanosClient(_GlanosClientBase):
    """Client for the Glanos anonymization.ai API; every call blocks until done.

    See AsyncGlanosClient for the same API with `async`/`await`.
    """

    def __init__(
        self,
        token: str,
        base_url: str,
        text_extractor: TextExtractor | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        transport: httpx.BaseTransport | None = None,
        no_ssl_verify: bool = False,
    ) -> None:
        super().__init__(token, base_url, text_extractor)
        self._client: httpx.Client | None = httpx.Client(
            verify=not no_ssl_verify,
            timeout=timeout,
            transport=transport,
        )

    def __enter__(self) -> "GlanosClient":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    @property
    def client(self) -> httpx.Client:
        return self._client

    def _send(self, request: _ApiRequest) -> httpx.Response:
        response = self.client.request(
            request.method, request.url, **request.httpx_kwargs()
        )
        self._raise_for_status(response)
        return response

    def anonymize(
        self,
        text: str | None = None,
        path: str | Path | None = None,
        filename: str | None = None,
        settings: AnonymizationSettings | None = None,
        settings_id: str | None = None,
        session: str | None = None,
        session_max_age_ms: int | None = None,
        *,
        data: bytes | None = None,
        processing: Processing | str | None = None,
        poll_interval: float | None = None,
        max_attempts: int | None = None,
        return_pseudo_key: bool | None = None,
        return_encrypt_key: bool | None = None,
        language: str | None = None,
        data_retention_ms: int | None = None,
        pseudo_key_retention_ms: int | None = None,
        meta_retention_ms: int | None = None,
        options: list[str] | None = None,
    ) -> AnonymizationResult:
        """Anonymize a text or a document.

        `processing` decides how long to wait for the result (default: "sync"
        for text, "poll" for documents):

        - "sync": one request that waits for the document.
        - "poll": submit, then poll every `poll_interval` seconds (default 10),
          at most `max_attempts` times (default 30). Only these two options
          are about polling, and they only apply to "poll".
        - "no_wait": submit and return at once.

        If the document is not ready in time, the result is pending
        (`result.ready` is False): collect it later with `fetch()` and
        `result.fetch_key`.
        """
        input_data = _input_data(text, path, data, filename)
        processing, polling = self._anonymize_plan(
            processing,
            input_data,
            poll_interval,
            max_attempts,
            return_pseudo_key,
            return_encrypt_key,
        )
        request = self._pseudo_request(
            input_data,
            server_async=processing is not Processing.SYNC,
            return_pseudo_key=return_pseudo_key,
            return_encrypt_key=return_encrypt_key,
            settings=settings,
            settings_id=settings_id,
            session=session,
            session_max_age_ms=session_max_age_ms,
            language=language,
            data_retention_ms=data_retention_ms,
            pseudo_key_retention_ms=pseudo_key_retention_ms,
            meta_retention_ms=meta_retention_ms,
            options=options,
        )
        result = self._result(self._parse_pseudo_response(self._send(request)))
        if polling is None:
            return result
        return self.fetch(result, **polling)

    def anonymize_with_review(
        self,
        text: str | None = None,
        path: str | Path | None = None,
        filename: str | None = None,
        settings: AnonymizationSettings | None = None,
        settings_id: str | None = None,
        session: str | None = None,
        session_max_age_ms: int | None = None,
        *,
        assignee: str | None = None,
        assigned_group: str | None = None,
        data: bytes | None = None,
        language: str | None = None,
        data_retention_ms: int | None = None,
        pseudo_key_retention_ms: int | None = None,
        meta_retention_ms: int | None = None,
        options: list[str] | None = None,
    ) -> AnonymizationResult:
        """Anonymize a document that a person reviews in the Glanos dashboard.

        Pass exactly one of `assignee` (a user) or `assigned_group` (a group).
        It returns at once, without polling: the result is pending until the
        review is done. Collect it with `fetch()` and `result.fetch_key`.
        """
        request = self._review_request(
            _input_data(text, path, data, filename),
            assignee,
            assigned_group,
            settings=settings,
            settings_id=settings_id,
            session=session,
            session_max_age_ms=session_max_age_ms,
            language=language,
            data_retention_ms=data_retention_ms,
            pseudo_key_retention_ms=pseudo_key_retention_ms,
            meta_retention_ms=meta_retention_ms,
            options=options,
        )
        return self._result(self._parse_pseudo_response(self._send(request)))

    def fetch(
        self,
        pending: AnonymizationResult | str,
        *,
        poll_interval: float = DEFAULT_FETCH_POLL_INTERVAL,
        max_attempts: int | None = DEFAULT_FETCH_MAX_ATTEMPTS,
        start_retention_period: bool = True,
        download_options: DownloadOption | None = None,
        return_pseudo_key: bool = True,
        return_encrypt_key: bool = True,
    ) -> AnonymizationResult:
        """Collect a pending result, given as the result or its fetch key.

        Asks every `poll_interval` seconds (default 300, for documents in
        review) until the document is ready. After `max_attempts` requests
        (default None: no limit) it returns the result still pending. An error
        ends the polling; its note carries the fetch key.
        """
        fetch_key = self._pending_fetch_key(pending)
        if fetch_key is None:
            return pending
        _check_polling(poll_interval, max_attempts)

        request = self._fetch_request(
            fetch_key,
            start_retention_period,
            download_options,
            return_pseudo_key,
            return_encrypt_key,
        )
        attempts = 0
        while True:
            attempts += 1
            try:
                result = self._result(self._parse_pseudo_response(self._send(request)))
            except Exception as exc:
                self._note_fetch_key(exc, fetch_key)
                raise
            if result.ready or attempts == max_attempts:
                return result
            time.sleep(poll_interval)

    def deanonymize(
        self,
        result: AnonymizationResult | None = None,
        *,
        text: str | None = None,
        path: str | Path | None = None,
        data: bytes | None = None,
        filename: str | None = None,
        session: str | None = None,
        depseudo_paths: DepseudoPaths | None = None,
    ) -> DeanonymizationResult:
        """Restore the original values.

        Pass the `result` of anonymize(): its keys (or session) are used unless
        `session` or `depseudo_paths` is given. Or pass anonymized `text`, a
        file (`path`) or bytes (`data` with `filename`) together with
        `session` or `depseudo_paths`.
        """
        request = self._deanonymize_request(
            result, text, path, data, filename, session, depseudo_paths
        )
        restored = self._parse_pseudo_response(self._send(request))
        return self._deanonymization_result(restored)

    def depseudo_text(
        self,
        text: str,
        session: str | None = None,
        depseudo_paths: DepseudoPaths | None = None,
    ) -> str:
        """Restore a text, e.g. a model's reply, and return it as a string."""
        return self.deanonymize(
            text=text, session=session, depseudo_paths=depseudo_paths
        ).extract_text()

    def add_session(
        self,
        max_age_seconds: int,
        *,
        settings_id: str,
        name: str | None = None,
        share_users: list[str] | None = None,
        share_groups: list[str] | None = None,
    ) -> SessionInfo:
        request = self._add_session_request(
            max_age_seconds, settings_id, name, share_users, share_groups
        )
        return self._parse_session_response(self._send(request))

    def _start_session(
        self,
        max_age_ms: int,
        settings: AnonymizationSettings | None = None,
        settings_id: str | None = None,
    ) -> str:
        if settings_id is not None:
            return self.add_session(
                self._session_max_age_seconds(max_age_ms), settings_id=settings_id
            ).id

        result = self.anonymize(
            text=".",
            settings=settings,
            session_max_age_ms=max_age_ms,
            return_pseudo_key=False,
            return_encrypt_key=False,
        )
        result = self.fetch(result, **self._COLLECT_IN_TIME)
        return self._session_of(result)

    def get_session(self, session_id: str) -> SessionInfo:
        request = self._get_session_request(session_id)
        return self._parse_session_response(self._send(request))

    def extend_session(self, session_id: str, max_age_seconds: int) -> SessionInfo:
        request = self._extend_session_request(session_id, max_age_seconds)
        return self._parse_session_response(self._send(request))

    def remove_session(self, session_id: str) -> SuccessMessage:
        request = self._remove_session_request(session_id)
        return self._parse_success(self._send(request))

    def get_setting(self, setting_id: str) -> AnonymizationSettings:
        request = self._get_setting_request(setting_id)
        return self._parse_setting(self._send(request))

    def list_settings(
        self,
        query: str | None = None,
        creator: str | list[str] | None = None,
        sort: SettingsSort | None = None,
        sort_order: SortOrder | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> AnonymizationSettingsList:
        request = self._list_settings_request(
            query, creator, sort, sort_order, limit, offset
        )
        return self._parse_settings_list(self._send(request))

    def add_setting(
        self, settings: AnonymizationSettings, apply_defaults: bool = True
    ) -> AnonymizationSettings:
        request = self._add_setting_request(settings, apply_defaults)
        return self._parse_setting(self._send(request))

    def update_setting(
        self, settings: AnonymizationSettings, apply_defaults: bool = True
    ) -> AnonymizationSettings:
        request = self._update_setting_request(settings, apply_defaults)
        try:
            response = self._send(request)
        except GlanosAPIError as exc:
            not_found = self._setting_not_found(exc, settings)
            if not_found is None:
                raise
            raise not_found from exc
        return self._parse_setting(response)

    def remove_setting(self, setting_id: str) -> SuccessMessage:
        request = self._remove_setting_request(setting_id)
        return self._parse_success(self._send(request))


class AsyncGlanosClient(_GlanosClientBase):
    """Client for the Glanos anonymization.ai API with `async`/`await`.

    The same methods as GlanosClient, as coroutines, built on
    httpx.AsyncClient: `async with AsyncGlanosClient(...) as client:
    result = await client.anonymize(text=...)`. Reading input files and
    `extract_text()` on a result still run synchronously.
    """

    def __init__(
        self,
        token: str,
        base_url: str,
        text_extractor: TextExtractor | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        transport: httpx.AsyncBaseTransport | None = None,
        no_ssl_verify: bool = False,
    ) -> None:
        super().__init__(token, base_url, text_extractor)
        self._client: httpx.AsyncClient | None = httpx.AsyncClient(
            verify=not no_ssl_verify,
            timeout=timeout,
            transport=transport,
        )

    async def __aenter__(self) -> "AsyncGlanosClient":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        return self._client

    async def _send(self, request: _ApiRequest) -> httpx.Response:
        response = await self.client.request(
            request.method, request.url, **request.httpx_kwargs()
        )
        self._raise_for_status(response)
        return response

    async def anonymize(
        self,
        text: str | None = None,
        path: str | Path | None = None,
        filename: str | None = None,
        settings: AnonymizationSettings | None = None,
        settings_id: str | None = None,
        session: str | None = None,
        session_max_age_ms: int | None = None,
        *,
        data: bytes | None = None,
        processing: Processing | str | None = None,
        poll_interval: float | None = None,
        max_attempts: int | None = None,
        return_pseudo_key: bool | None = None,
        return_encrypt_key: bool | None = None,
        language: str | None = None,
        data_retention_ms: int | None = None,
        pseudo_key_retention_ms: int | None = None,
        meta_retention_ms: int | None = None,
        options: list[str] | None = None,
    ) -> AnonymizationResult:
        """Anonymize a text or a document; see GlanosClient.anonymize()."""
        input_data = _input_data(text, path, data, filename)
        processing, polling = self._anonymize_plan(
            processing,
            input_data,
            poll_interval,
            max_attempts,
            return_pseudo_key,
            return_encrypt_key,
        )
        request = self._pseudo_request(
            input_data,
            server_async=processing is not Processing.SYNC,
            return_pseudo_key=return_pseudo_key,
            return_encrypt_key=return_encrypt_key,
            settings=settings,
            settings_id=settings_id,
            session=session,
            session_max_age_ms=session_max_age_ms,
            language=language,
            data_retention_ms=data_retention_ms,
            pseudo_key_retention_ms=pseudo_key_retention_ms,
            meta_retention_ms=meta_retention_ms,
            options=options,
        )
        result = self._result(self._parse_pseudo_response(await self._send(request)))
        if polling is None:
            return result
        return await self.fetch(result, **polling)

    async def anonymize_with_review(
        self,
        text: str | None = None,
        path: str | Path | None = None,
        filename: str | None = None,
        settings: AnonymizationSettings | None = None,
        settings_id: str | None = None,
        session: str | None = None,
        session_max_age_ms: int | None = None,
        *,
        assignee: str | None = None,
        assigned_group: str | None = None,
        data: bytes | None = None,
        language: str | None = None,
        data_retention_ms: int | None = None,
        pseudo_key_retention_ms: int | None = None,
        meta_retention_ms: int | None = None,
        options: list[str] | None = None,
    ) -> AnonymizationResult:
        """Anonymize a document for review; see GlanosClient.anonymize_with_review()."""
        request = self._review_request(
            _input_data(text, path, data, filename),
            assignee,
            assigned_group,
            settings=settings,
            settings_id=settings_id,
            session=session,
            session_max_age_ms=session_max_age_ms,
            language=language,
            data_retention_ms=data_retention_ms,
            pseudo_key_retention_ms=pseudo_key_retention_ms,
            meta_retention_ms=meta_retention_ms,
            options=options,
        )
        return self._result(self._parse_pseudo_response(await self._send(request)))

    async def fetch(
        self,
        pending: AnonymizationResult | str,
        *,
        poll_interval: float = DEFAULT_FETCH_POLL_INTERVAL,
        max_attempts: int | None = DEFAULT_FETCH_MAX_ATTEMPTS,
        start_retention_period: bool = True,
        download_options: DownloadOption | None = None,
        return_pseudo_key: bool = True,
        return_encrypt_key: bool = True,
    ) -> AnonymizationResult:
        """Collect a pending result; see GlanosClient.fetch()."""
        fetch_key = self._pending_fetch_key(pending)
        if fetch_key is None:
            return pending
        _check_polling(poll_interval, max_attempts)

        request = self._fetch_request(
            fetch_key,
            start_retention_period,
            download_options,
            return_pseudo_key,
            return_encrypt_key,
        )
        attempts = 0
        while True:
            attempts += 1
            try:
                result = self._result(
                    self._parse_pseudo_response(await self._send(request))
                )
            except Exception as exc:
                self._note_fetch_key(exc, fetch_key)
                raise
            if result.ready or attempts == max_attempts:
                return result
            await asyncio.sleep(poll_interval)

    async def deanonymize(
        self,
        result: AnonymizationResult | None = None,
        *,
        text: str | None = None,
        path: str | Path | None = None,
        data: bytes | None = None,
        filename: str | None = None,
        session: str | None = None,
        depseudo_paths: DepseudoPaths | None = None,
    ) -> DeanonymizationResult:
        """Restore the original values; see GlanosClient.deanonymize()."""
        request = self._deanonymize_request(
            result, text, path, data, filename, session, depseudo_paths
        )
        restored = self._parse_pseudo_response(await self._send(request))
        return self._deanonymization_result(restored)

    async def depseudo_text(
        self,
        text: str,
        session: str | None = None,
        depseudo_paths: DepseudoPaths | None = None,
    ) -> str:
        """Restore a text; see GlanosClient.depseudo_text()."""
        restored = await self.deanonymize(
            text=text, session=session, depseudo_paths=depseudo_paths
        )
        return restored.extract_text()

    async def add_session(
        self,
        max_age_seconds: int,
        *,
        settings_id: str,
        name: str | None = None,
        share_users: list[str] | None = None,
        share_groups: list[str] | None = None,
    ) -> SessionInfo:
        request = self._add_session_request(
            max_age_seconds, settings_id, name, share_users, share_groups
        )
        return self._parse_session_response(await self._send(request))

    async def _start_session(
        self,
        max_age_ms: int,
        settings: AnonymizationSettings | None = None,
        settings_id: str | None = None,
    ) -> str:
        if settings_id is not None:
            session = await self.add_session(
                self._session_max_age_seconds(max_age_ms), settings_id=settings_id
            )
            return session.id

        result = await self.anonymize(
            text=".",
            settings=settings,
            session_max_age_ms=max_age_ms,
            return_pseudo_key=False,
            return_encrypt_key=False,
        )
        result = await self.fetch(result, **self._COLLECT_IN_TIME)
        return self._session_of(result)

    async def get_session(self, session_id: str) -> SessionInfo:
        request = self._get_session_request(session_id)
        return self._parse_session_response(await self._send(request))

    async def extend_session(
        self, session_id: str, max_age_seconds: int
    ) -> SessionInfo:
        request = self._extend_session_request(session_id, max_age_seconds)
        return self._parse_session_response(await self._send(request))

    async def remove_session(self, session_id: str) -> SuccessMessage:
        request = self._remove_session_request(session_id)
        return self._parse_success(await self._send(request))

    async def get_setting(self, setting_id: str) -> AnonymizationSettings:
        request = self._get_setting_request(setting_id)
        return self._parse_setting(await self._send(request))

    async def list_settings(
        self,
        query: str | None = None,
        creator: str | list[str] | None = None,
        sort: SettingsSort | None = None,
        sort_order: SortOrder | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> AnonymizationSettingsList:
        request = self._list_settings_request(
            query, creator, sort, sort_order, limit, offset
        )
        return self._parse_settings_list(await self._send(request))

    async def add_setting(
        self, settings: AnonymizationSettings, apply_defaults: bool = True
    ) -> AnonymizationSettings:
        request = self._add_setting_request(settings, apply_defaults)
        return self._parse_setting(await self._send(request))

    async def update_setting(
        self, settings: AnonymizationSettings, apply_defaults: bool = True
    ) -> AnonymizationSettings:
        request = self._update_setting_request(settings, apply_defaults)
        try:
            response = await self._send(request)
        except GlanosAPIError as exc:
            not_found = self._setting_not_found(exc, settings)
            if not_found is None:
                raise
            raise not_found from exc
        return self._parse_setting(response)

    async def remove_setting(self, setting_id: str) -> SuccessMessage:
        request = self._remove_setting_request(setting_id)
        return self._parse_success(await self._send(request))
