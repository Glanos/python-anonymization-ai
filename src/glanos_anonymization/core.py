import os
import base64
import builtins
import math
import random
import sys
import httpx
from enum import Enum
from typing import Annotated, Literal, Union, Protocol
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pathlib import Path
import json
import tempfile
from markitdown import MarkItDown
import time

DEFAULT_TIMEOUT = 30.0
DEFAULT_FETCH_MAX_RETRIES = 30
DEFAULT_FETCH_POLL_INTERVAL = (1.0, 2.0)

API_BASE_PATH = "/be/api-v2"


class GlanosClientError(Exception):
    pass


class DepseudoKeyFileError(GlanosClientError):
    pass


class FetchTimeoutError(GlanosClientError):
    pass


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


class OutputMode(str, Enum):
    RAW = "raw"
    JSON = "json"


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
    def __init__(self, text: str, pseudo_result: PseudoResult):
        self.text = text
        self.pseudo_result = pseudo_result

    @property
    def data(self) -> bytes:
        return base64.b64decode(self._completed_data().data)

    @property
    def file_name(self) -> str:
        return self._completed_data().fileName

    def _completed_data(self) -> "InnerResponseData":
        data = self.pseudo_result.data
        if not isinstance(data, InnerResponseData):
            raise ValueError("pseudo_result does not contain a completed result")
        return data


class DeanonymizationResult(AnonymizationResult):
    pass


class TextExtractor(Protocol):
    def extract_text(self, pseudo_result: PseudoResult) -> str: ...


class MarkItDownTextExtractor(TextExtractor):
    def __init__(self):
        self.mid = MarkItDown()

    def extract_text(self, pseudo_result: PseudoResult) -> str:
        text_data = base64.b64decode(pseudo_result.data.data)
        suffix = Path(pseudo_result.data.fileName).suffix

        with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
            tmp.write(text_data)
            tmp.flush()

            text_content = self.mid.convert(tmp.name).text_content

        return text_content


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


def _default_keys(
    return_pseudo_key: bool | None, return_encrypt_key: bool | None
) -> tuple[bool, bool]:
    return (
        True if return_pseudo_key is None else return_pseudo_key,
        True if return_encrypt_key is None else return_encrypt_key,
    )


class GlanosClient:
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
        timeout: float = DEFAULT_TIMEOUT,
        transport: httpx.BaseTransport | None = None,
        no_ssl_verify: bool = False,
    ) -> None:
        self.token = token
        self.base_url = base_url.rstrip("/")
        self._client: httpx.Client | None = httpx.Client(
            verify=not no_ssl_verify,
            timeout=timeout,
            transport=transport,
        )
        self.text_extractor = text_extractor

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

    @property
    def text_extractor(self):
        return self._text_extractor

    @text_extractor.setter
    def text_extractor(self, extractor: TextExtractor | None):
        self._text_extractor = (
            extractor if extractor is not None else MarkItDownTextExtractor()
        )

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

    def _request(
        self,
        method: str,
        url: str,
        params: dict | None = None,
        json_body: dict | None = None,
    ) -> httpx.Response:
        kwargs: dict = {}
        if params is not None:
            kwargs["params"] = params
        if json_body is not None:
            kwargs["content"] = json.dumps(json_body)
            kwargs["headers"] = {"Content-Type": "application/json"}

        response = self.client.request(method, url, **kwargs)
        self._raise_for_status(response)

        return response

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
        session_max_age: int | None = None,
        language: str | None = None,
        data_retention_ms: int | None = None,
        pseudo_key_retention_ms: int | None = None,
        meta_retention_ms: int | None = None,
        assignee: str | None = None,
        assigned_group: str | None = None,
        options: list[str] | None = None,
    ) -> dict:

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
            sessionMaxAge=session_max_age,
            options=options,
        )

        body = self._to_json_dict(upload)
        if settings is not None:
            body["settings"] = self._inline_settings_body(settings)
        return body

    def pseudo_sync(
        self,
        text: str | None = None,
        path: str | Path | None = None,
        filename: str | None = None,
        settings: AnonymizationSettings | None = None,
        settings_id: str | None = None,
        session: str | None = None,
        session_max_age: int | None = None,
        *,
        data: bytes | None = None,
        a_sync: bool | None = None,
        return_pseudo_key: bool | None = None,
        return_encrypt_key: bool | None = None,
        language: str | None = None,
        data_retention_ms: int | None = None,
        pseudo_key_retention_ms: int | None = None,
        meta_retention_ms: int | None = None,
        assignee: str | None = None,
        assigned_group: str | None = None,
        options: list[str] | None = None,
    ) -> PseudoResult:

        if a_sync and (return_pseudo_key or return_encrypt_key):
            raise ValueError(
                "Invalid parameter combination: return_pseudo_key/return_encrypt_key "
                "require a synchronous request, since keys are only available on the "
                "completed result."
            )
        if not a_sync:
            return_pseudo_key, return_encrypt_key = _default_keys(
                return_pseudo_key, return_encrypt_key
            )

        input_data = _input_data(text, path, data, filename)

        params = {"token": self.token}
        for name, value in (
            ("async", a_sync),
            ("returnPseudoKey", return_pseudo_key),
            ("returnEncryptKey", return_encrypt_key),
        ):
            if value is not None:
                params[name] = str(value).lower()

        request_body = self.build_pseudo_request_body(
            input_data=input_data,
            settings=settings,
            settings_id=settings_id,
            session=session,
            session_max_age=session_max_age,
            language=language,
            data_retention_ms=data_retention_ms,
            pseudo_key_retention_ms=pseudo_key_retention_ms,
            meta_retention_ms=meta_retention_ms,
            assignee=assignee,
            assigned_group=assigned_group,
            options=options,
        )

        response = self._request(
            "POST",
            self.build_pseudo_request_url(),
            params=params,
            json_body=request_body,
        )

        return self._parse_pseudo_response(response)

    def build_fetch_request_url(self) -> str:
        return f"{self.base_url}{API_BASE_PATH}/document/pseudo/fetch"

    def fetch(
        self,
        pseudo_result: PseudoResult,
        max_retries: int = DEFAULT_FETCH_MAX_RETRIES,
        poll_interval: tuple[float, float] = DEFAULT_FETCH_POLL_INTERVAL,
        start_retention_period: bool = True,
        download_options: DownloadOption | None = None,
        return_pseudo_key: bool = True,
        return_encrypt_key: bool = True,
    ) -> PseudoResult:

        if isinstance(pseudo_result.data, InnerResponseData):
            return pseudo_result

        fetch_key = pseudo_result.data.fetchKey
        url = self.build_fetch_request_url()
        params = {
            "token": self.token,
            "fetchKey": fetch_key,
            "returnPseudoKey": str(return_pseudo_key).lower(),
            "returnEncryptKey": str(return_encrypt_key).lower(),
            "startRetentionPeriod": str(start_retention_period).lower(),
        }
        if download_options is not None:
            params["downloadOptions"] = download_options.value

        for _ in range(max_retries):
            response = self._request("GET", url, params=params)
            pseudo_result = self._parse_pseudo_response(response)

            if isinstance(pseudo_result.data, InnerResponseData):
                return pseudo_result

            time.sleep(random.uniform(*poll_interval))

        raise FetchTimeoutError(
            f"Anonymization result not ready after {max_retries} attempts "
            f"(fetchKey={fetch_key})"
        )

    def get_ano_result(self, pseudo_result: PseudoResult) -> AnonymizationResult:

        data = pseudo_result.data
        if not isinstance(data, InnerResponseData):
            raise ValueError(
                "pseudo_result does not contain a completed result; call fetch() first"
            )

        if _is_plain_text(data):
            text_content = base64.b64decode(data.data).decode("utf-8")
        else:
            text_content = self.text_extractor.extract_text(pseudo_result)

        return AnonymizationResult(text_content, pseudo_result)

    def write_to_output(
        self,
        data: str | bytes,
        output_file: str | Path,
        extension: str | None = None,
        mode: str = "w",
    ) -> None:

        path = Path(output_file)
        if extension:
            path = path.with_suffix(f".{extension}")

        with open(path, mode=mode) as f:
            f.write(data)

    def manage_data(
        self, pseudo_result: PseudoResult, output_file: str | Path | None = None
    ) -> None:

        if not isinstance(pseudo_result.data, InnerResponseData):
            print(
                "Document accepted but not yet processed. Use the fetch key together "
                "with the fetch script to request the anonymized document again",
                file=sys.stderr,
            )
            print(pseudo_result.data.model_dump_json())
            return

        content = base64.b64decode(pseudo_result.data.data)
        content_length = len(content)

        self.write_to_output(content, output_file=output_file, mode="wb")

        mk = pseudo_result.data.metaKey
        print(f"metaKey: {mk}")
        print(f"fileName: {pseudo_result.data.fileName}")
        if pseudo_result.data.session is not None:
            print(f"session: {pseudo_result.data.session}")

        pk = pseudo_result.data.pseudoKey
        if pk is not None:
            print(f"pseudoKey: {pk}")
            self.write_to_output(pk, output_file=output_file, extension="pkey")

        ek = pseudo_result.data.encryptKey
        if ek is not None:
            print(f"encryptKey: {ek}")
            self.write_to_output(ek, output_file=output_file, extension="ekey")

        print(f"{output_file} (SZ: {content_length})")

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

    def depseudo(
        self,
        path: str | Path | None = None,
        depseudo_paths: DepseudoPaths | None = None,
        session: str | None = None,
        output_mode: OutputMode = OutputMode.JSON,
        filename: str | None = None,
        text: str | None = None,
        *,
        data: bytes | None = None,
    ) -> PseudoResult:

        input_data = _input_data(text, path, data, filename)

        return self._send_depseudo(input_data, depseudo_paths, session, output_mode)

    def _send_depseudo(
        self,
        input_data: InputData,
        depseudo_paths: DepseudoPaths | None,
        session: str | None,
        output_mode: OutputMode = OutputMode.JSON,
    ) -> PseudoResult:
        request_body = self.build_depseudo_request_body(
            input_data=input_data, depseudo_paths=depseudo_paths, session=session
        )
        params = {"token": self.token, "outputMode": output_mode.value}

        response = self._request(
            "POST",
            self.build_depseudo_request_url(),
            params=params,
            json_body=request_body,
        )

        return self._parse_pseudo_response(response)

    def deanonymize(
        self,
        result: AnonymizationResult | PseudoResult,
        *,
        session: str | None = None,
        depseudo_paths: DepseudoPaths | None = None,
    ) -> DeanonymizationResult:
        pseudo_result = (
            result.pseudo_result if isinstance(result, AnonymizationResult) else result
        )
        data = pseudo_result.data
        if not isinstance(data, InnerResponseData):
            raise ValueError(
                "result does not contain a completed result; call fetch() first"
            )

        if session is None and depseudo_paths is None:
            try:
                depseudo_paths = DepseudoPaths.from_result(pseudo_result)
            except ValueError:
                if not data.session:
                    raise
                session = data.session

        content = base64.b64decode(data.data)
        if _is_plain_text(data):
            input_data = InputData(
                content=content.decode("utf-8"), source="string", filename=data.fileName
            )
        else:
            input_data = InputData(
                content=content, source="file", filename=data.fileName
            )
        restored = self._send_depseudo(input_data, depseudo_paths, session)
        restored_data = restored.data
        if not isinstance(restored_data, InnerResponseData):
            raise ValueError("depseudo did not return a completed result")

        if Path(restored_data.fileName).suffix.lower() == ".txt":
            text = base64.b64decode(restored_data.data).decode("utf-8")
        else:
            text = self.text_extractor.extract_text(restored)
        return DeanonymizationResult(text, restored)

    def depseudo_text(
        self,
        text: str,
        session: str | None = None,
        depseudo_paths: DepseudoPaths | None = None,
        filename: str | None = None,
    ) -> str:

        result = self.depseudo(
            text=text, session=session, depseudo_paths=depseudo_paths, filename=filename
        )
        if not isinstance(result.data, InnerResponseData):
            raise ValueError("depseudo did not return a completed result")

        return base64.b64decode(result.data.data).decode("utf-8")

    def anonymize(
        self,
        text: str | None = None,
        path: str | Path | None = None,
        filename: str | None = None,
        settings: AnonymizationSettings | None = None,
        settings_id: str | None = None,
        session: str | None = None,
        session_max_age: int | None = None,
        *,
        data: bytes | None = None,
        a_sync: bool | None = None,
        return_pseudo_key: bool | None = None,
        return_encrypt_key: bool | None = None,
        language: str | None = None,
        data_retention_ms: int | None = None,
        pseudo_key_retention_ms: int | None = None,
        meta_retention_ms: int | None = None,
        assignee: str | None = None,
        assigned_group: str | None = None,
        options: list[str] | None = None,
    ) -> AnonymizationResult:

        pseudo_result = self.pseudo_sync(
            text=text,
            path=path,
            filename=filename,
            data=data,
            settings=settings,
            settings_id=settings_id,
            session=session,
            session_max_age=session_max_age,
            a_sync=a_sync,
            return_pseudo_key=return_pseudo_key,
            return_encrypt_key=return_encrypt_key,
            language=language,
            data_retention_ms=data_retention_ms,
            pseudo_key_retention_ms=pseudo_key_retention_ms,
            meta_retention_ms=meta_retention_ms,
            assignee=assignee,
            assigned_group=assigned_group,
            options=options,
        )

        if a_sync:
            fetch_pseudo_result = self.fetch(pseudo_result)
        else:
            want_pseudo_key, want_encrypt_key = _default_keys(
                return_pseudo_key, return_encrypt_key
            )
            fetch_pseudo_result = self.fetch(
                pseudo_result,
                return_pseudo_key=want_pseudo_key,
                return_encrypt_key=want_encrypt_key,
            )
        anonymization_result = self.get_ano_result(fetch_pseudo_result)

        return anonymization_result

    @staticmethod
    def _parse_session_response(response: httpx.Response) -> SessionInfo:
        body = response.json()
        if isinstance(body, dict) and body.get("success") is False:
            raise GlanosClientError(
                f"Session request failed: {body.get('message') or 'no reason given'}"
            )
        return SessionInfo.model_validate(_unwrap_data_envelope(body))

    def add_session(
        self,
        max_age_seconds: int,
        *,
        settings_id: str,
        name: str | None = None,
        share_users: list[str] | None = None,
        share_groups: list[str] | None = None,
    ) -> SessionInfo:
        body = AnonymizationSessionAddData(
            name=name,
            settingsId=settings_id,
            shareUsers=share_users,
            shareGroups=share_groups,
            sessionMaxAge=max_age_seconds,
        )
        response = self._request(
            "POST",
            f"{self.base_url}{API_BASE_PATH}/session/add",
            params={"token": self.token},
            json_body=self._to_json_dict(body),
        )
        return self._parse_session_response(response)

    def _start_session(
        self,
        max_age_ms: int,
        settings: AnonymizationSettings | None = None,
        settings_id: str | None = None,
    ) -> str:
        if settings_id is not None:
            return self.add_session(
                max_age_seconds=max(1, math.ceil(max_age_ms / 1000)),
                settings_id=settings_id,
            ).id

        result = self.fetch(
            self.pseudo_sync(text=".", settings=settings, session_max_age=max_age_ms)
        )
        session_id = (
            result.data.session if isinstance(result.data, InnerResponseData) else None
        )
        if not session_id:
            raise GlanosClientError(
                "Glanos anonymization.ai did not return a session id despite session_max_age being set"
            )
        return session_id

    def get_session(self, session_id: str) -> SessionInfo:
        response = self._request(
            "GET",
            f"{self.base_url}{API_BASE_PATH}/session/get",
            params={"token": self.token, "session": session_id},
        )
        return self._parse_session_response(response)

    def extend_session(self, session_id: str, max_age_seconds: int) -> SessionInfo:
        response = self._request(
            "GET",
            f"{self.base_url}{API_BASE_PATH}/session/extend",
            params={
                "token": self.token,
                "session": session_id,
                "sessionMaxAge": max_age_seconds,
            },
        )
        return self._parse_session_response(response)

    def remove_session(self, session_id: str) -> SuccessMessage:
        response = self._request(
            "PUT",
            f"{self.base_url}{API_BASE_PATH}/session/remove",
            params={"token": self.token, "session": session_id},
        )
        return SuccessMessage.model_validate(response.json())

    def get_setting(self, setting_id: str) -> AnonymizationSettings:
        response = self._request(
            "GET",
            f"{self.base_url}{API_BASE_PATH}/setting/get",
            params={"token": self.token, "setting": setting_id},
        )
        return AnonymizationSettings.model_validate(
            _unwrap_data_envelope(response.json())
        )

    def list_settings(
        self,
        query: str | None = None,
        creator: str | list[str] | None = None,
        sort: SettingsSort | None = None,
        sort_order: SortOrder | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> AnonymizationSettingsList:
        params: dict = {"token": self.token}
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

        response = self._request(
            "GET", f"{self.base_url}{API_BASE_PATH}/setting/list", params=params
        )
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

    def _write_setting(
        self, endpoint: str, body: dict, apply_defaults: bool
    ) -> AnonymizationSettings:
        response = self._request(
            "POST",
            f"{self.base_url}{API_BASE_PATH}/setting/{endpoint}",
            params={"token": self.token, "applyDefaults": str(apply_defaults).lower()},
            json_body=body,
        )
        return AnonymizationSettings.model_validate(
            _unwrap_data_envelope(response.json())
        )

    def add_setting(
        self, settings: AnonymizationSettings, apply_defaults: bool = True
    ) -> AnonymizationSettings:
        if settings.id is not None:
            raise ValueError(
                "add_setting() creates a new setting and the server assigns its id; "
                "use update_setting() to change the setting with id "
                f"{settings.id!r}"
            )
        if not settings.name:
            raise ValueError("A new setting needs a name (settings.name)")
        return self._write_setting(
            "add", self._stored_setting_body(settings), apply_defaults
        )

    def update_setting(
        self, settings: AnonymizationSettings, apply_defaults: bool = True
    ) -> AnonymizationSettings:
        if not settings.id:
            raise ValueError(
                "update_setting() changes an existing setting and needs its id "
                "(settings.id); use add_setting() to create a new one"
            )
        body = self._stored_setting_body(settings)
        try:
            return self._write_setting("update", body, apply_defaults)
        except GlanosAPIError as exc:
            if (exc.api_message or "").startswith("No such settings"):
                raise SettingNotFoundError(
                    f"No stored setting with id {settings.id!r}"
                ) from exc
            raise

    def remove_setting(self, setting_id: str) -> SuccessMessage:
        response = self._request(
            "PUT",
            f"{self.base_url}{API_BASE_PATH}/setting/remove",
            params={"token": self.token, "setting": setting_id},
        )
        return SuccessMessage.model_validate(response.json())
