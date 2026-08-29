import os
import base64
import random
import httpx
from enum import Enum
from typing import Literal, Union, Protocol
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
    """Base exception for errors raised by GlanosClient."""


class DepseudoKeyFileError(GlanosClientError):
    """Raised when a depseudonymization key file cannot be read."""


class FetchTimeoutError(GlanosClientError):
    """Raised when an asynchronous anonymization result is not ready in time."""


class RateLimitedError(GlanosClientError):
    """Raised on HTTP 429 (too many requests)."""


class QuotaExceededError(GlanosClientError):
    """Raised on HTTP 402 (license expired or quota exceeded)."""


class DataExpiredError(GlanosClientError):
    """Raised on HTTP 410 (data or session expired)."""


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


def _unwrap_data_envelope(value):
    """The deployed API wraps some /session responses in a
    {"success": ..., "data": {...}} envelope even where the spec documents a
    bare schema. Transparently unwrap it if present; leave the value alone
    otherwise"""
    if isinstance(value, dict) and "data" in value and "success" in value:
        return value["data"]
    return value


class AnonymizationSettings(BaseModel):
    """Mirrors the `AnonymizationSettings` schema"""

    excludeFields: list[ExcludeField] = Field(default_factory=list)
    mode: AnonymizationMode = AnonymizationMode.TYPE
    workflow: str | None = None
    allowAmbigReplacement: bool | None = Field(
        default=None, description="Deprecated by the API in favor of foldMode."
    )
    addNoTranslateMarkup: bool | None = None
    preserveLength: bool | None = Field(
        default=None, description="Deprecated by the API in favor of blankOutMode."
    )
    blankOutMode: BlankOutMode | None = None
    foldMode: FoldMode | None = None
    imageReplace: ImageReplaceMode | None = None
    custom: list[str] = Field(default_factory=list)


class ApiUploadDocument(BaseModel):
    """Mirrors the `ApiUploadDocument` request schema"""

    data: str | None = None
    plainText: str | None = None
    language: str | None = None
    fileName: str
    settings: AnonymizationSettings | None = None
    settingsId: str | None = None
    dataRetentionMS: int = 0
    pseudoKeyRetentionMS: int = 0
    metaRetentionMS: int = 0
    assignee: str | None = None
    assignedGroup: str | None = None
    session: str | None = None
    sessionMaxAge: int | None = None
    options: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_content(self) -> "ApiUploadDocument":
        if self.data is None and self.plainText is None:
            raise ValueError("Either 'data' or 'plainText' is required")
        return self


class ApiUploadDepseudoDocument(BaseModel):
    """Mirrors the `ApiUploadDepseudoDocument` request schema"""

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
    """Mirrors `ApiPseudoData` for a completed (HTTP 200) response."""

    data: str
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
    """Mirrors `ApiAcceptedPseudoData` for an accepted-but-pending (HTTP 202) response."""

    fetchKey: str


class PseudoResult(BaseModel):
    """Mirrors `ApiPseudoResponseMessage` / `ApiPseudoAcceptedResponseMessage`.

    Which variant `data` parses as is used to distinguish a completed result
    from a still-pending one.
    """

    success: bool
    data: Union[InnerResponseData, FetchKeyData]
    message: str | None = None


class SuccessMessage(BaseModel):
    """Mirrors the generic `SuccessMessage` response schema."""

    @model_validator(mode="before")
    @classmethod
    def _unwrap(cls, value):
        if isinstance(value, dict) and "data" in value:
            return value["data"]
        return value

    success: bool


class AnonymizationConfig(BaseModel):
    """
    Configuration for anonymization requests and the underlying HTTP client.

    Groups together: client-only settings, the pseudo/fetch query parameters,
    the top-level ApiUploadDocument fields, and the nested AnonymizationSettings
    fields (converted via `to_settings()` when building a request body).
    """

    model_config = ConfigDict(validate_assignment=True)

    # client-only
    no_ssl_verify: bool = False

    # query parameters for POST /document/pseudo. NOTE: GET /document/pseudo/fetch
    # has its own returnPseudoKey/returnEncryptKey query parameters with
    # different defaults (true/false) - fetch() takes those as its own
    # explicit arguments rather than reading these fields, since reusing
    # them here previously caused fetch() to silently suppress the pseudo
    # key by default.
    a_sync: bool = True
    return_pseudo_key: bool = False
    return_encrypt_key: bool = False

    language: str | None = None
    data_retention_ms: int = 0
    pseudo_key_retention_ms: int = 0
    meta_retention_ms: int = 0
    settings_id: str | None = None
    assignee: str | None = None
    assigned_group: str | None = None
    session: str | None = None
    session_max_age: int | None = None
    options: list[str] = Field(default_factory=list)

    anonymization_mode: AnonymizationMode = AnonymizationMode.TYPE
    exclude_fields: list[ExcludeField] = Field(default_factory=list)
    workflow: str | None = None
    fold_mode: FoldMode | None = None
    blank_out_mode: BlankOutMode | None = None
    image_replace: ImageReplaceMode | None = None
    add_no_translate_markup: bool | None = None
    allow_ambig_replacement: bool | None = None
    preserve_length: bool | None = None
    custom: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_combination(self) -> "AnonymizationConfig":
        if self.a_sync and (self.return_pseudo_key or self.return_encrypt_key):
            raise ValueError(
                "Invalid parameter combination: return_pseudo_key/return_encrypt_key "
                "require a_sync=False, since keys are only available on the completed result."
            )
        return self

    def to_settings(self) -> AnonymizationSettings:
        """Build the nested `settings` object sent with pseudo requests."""
        return AnonymizationSettings(
            excludeFields=self.exclude_fields,
            mode=self.anonymization_mode,
            workflow=self.workflow,
            allowAmbigReplacement=self.allow_ambig_replacement,
            addNoTranslateMarkup=self.add_no_translate_markup,
            preserveLength=self.preserve_length,
            blankOutMode=self.blank_out_mode,
            foldMode=self.fold_mode,
            imageReplace=self.image_replace,
            custom=self.custom,
        )


SourceType = Literal["string", "file"]


class InputData(BaseModel):
    """Represents input data submitted for anonymization."""

    content: str | bytes
    source: SourceType
    filename: str | None = None


class DepseudoPaths(BaseModel):
    """
    Contains paths to (or values of) keys required for reversing anonymization.

    Attributes:
        pseudo_key_path: Path to the pseudo key file.
        meta_key: Metadata key value as returned by pseudonymization (not a file path).
        encrypt_key_path: Path to the encryption key file.
    """

    model_config = ConfigDict(extra="forbid")

    pseudo_key_path: str | None = None
    meta_key: str | None = None
    encrypt_key_path: str | None = None


class AnonymizationResult:
    """
    Represents the final anonymization result.

    Attributes:
        text: Extracted anonymized text content.
        pseudo_result: Original service response containing metadata and keys.
    """

    def __init__(self, text: str, pseudo_result: PseudoResult):
        self.text = text
        self.pseudo_result = pseudo_result


class TextExtractor(Protocol):
    """Protocol defining an interface for extracting text from anonymized files."""

    def extract_text(self, pseudo_result: PseudoResult) -> str: ...


class MarkItDownTextExtractor(TextExtractor):
    """Extracts text content from anonymized files using MarkItDown."""

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


class GlanosClient:
    """Client for communicating with the Glanos anonymization API."""

    _STATUS_EXCEPTIONS: dict[int, type[GlanosClientError]] = {
        402: QuotaExceededError,
        410: DataExpiredError,
        429: RateLimitedError,
    }

    def __init__(
        self,
        token: str,
        base_url: str,
        config: AnonymizationConfig | None = None,
        text_extractor: TextExtractor | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        """
        Args:
            transport: Optional httpx transport override, e.g. httpx.MockTransport
                for tests. Left as None, httpx uses its normal network transport.
        """
        self.token = token
        self.base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._transport = transport
        self._client: httpx.Client | None = None
        self.config = config
        self.text_extractor = text_extractor

    def __enter__(self) -> "GlanosClient":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._client is not None:
            self._client.close()
            self._client = None

    @property
    def client(self) -> httpx.Client:
        """Get the underlying HTTP client."""
        return self._client

    def _rebuild_http_client(self) -> None:
        """(Re)build the HTTP client from the current config."""
        if self._client is not None:
            self._client.close()

        self._client = httpx.Client(
            verify=not self._config.no_ssl_verify,
            timeout=self._timeout,
            transport=self._transport,
        )

    @property
    def config(self):
        """Get the current anonymization configuration."""
        return self._config

    @config.setter
    def config(self, config: AnonymizationConfig | None):
        """Set the anonymization configuration."""
        self._config = config if config is not None else AnonymizationConfig()
        self._rebuild_http_client()

    @property
    def text_extractor(self):
        """Get the configured text extractor."""
        return self._text_extractor

    @text_extractor.setter
    def text_extractor(self, extractor: TextExtractor | None):
        """Set the text extractor implementation."""
        self._text_extractor = (
            extractor if extractor is not None else MarkItDownTextExtractor()
        )

    def _raise_for_status(self, response: httpx.Response) -> None:
        exc_type = self._STATUS_EXCEPTIONS.get(response.status_code)
        if exc_type is not None:
            raise exc_type(f"HTTP {response.status_code}: {response.text}")

        if response.is_error:
            # Include the response body: it's frequently the only place the
            # API explains *why* a request was rejected (e.g. 400s).
            message = f"HTTP {response.status_code} for {response.url}: {response.text}"
            raise httpx.HTTPStatusError(
                message, request=response.request, response=response
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
        """Dump a model to a plain JSON-compatible dict (enums -> their .value, etc.)."""
        return model.model_dump(mode="json", exclude_none=True)

    # --- /document/pseudo ---------------------------------------------

    def build_pseudo_request_url(self) -> str:
        """Build the API URL for POST /document/pseudo (path only, no query)."""
        return f"{self.base_url}{API_BASE_PATH}/document/pseudo"

    def build_pseudo_request_body(
        self,
        input_data: InputData,
        session: str | None = None,
        session_max_age: int | None = None,
    ) -> dict:
        """Build the `ApiUploadDocument` request payload for anonymization.

        session/session_max_age, when given, override self.config.session /
        self.config.session_max_age for this call only - this lets callers
        (e.g. a middleware serving concurrent conversations) target a
        specific session per-request without mutating the shared client's
        config, which would not be safe under concurrent use.
        """

        upload = ApiUploadDocument(
            data=base64.b64encode(input_data.content).decode("utf-8")
            if input_data.source == "file"
            else None,
            plainText=input_data.content if input_data.source == "string" else None,
            language=self.config.language,
            fileName=input_data.filename,
            settings=self.config.to_settings(),
            settingsId=self.config.settings_id,
            dataRetentionMS=self.config.data_retention_ms,
            pseudoKeyRetentionMS=self.config.pseudo_key_retention_ms,
            metaRetentionMS=self.config.meta_retention_ms,
            assignee=self.config.assignee,
            assignedGroup=self.config.assigned_group,
            session=session if session is not None else self.config.session,
            sessionMaxAge=session_max_age
            if session_max_age is not None
            else self.config.session_max_age,
            options=self.config.options,
        )

        return self._to_json_dict(upload)

    def pseudo_sync(
        self,
        text: str | None = None,
        path: str | Path | None = None,
        filename: str | None = None,
        session: str | None = None,
        session_max_age: int | None = None,
    ) -> PseudoResult:
        """
        Submit text or a file for anonymization.

        Args:
            text: Text content to anonymize.
            path: Path to a file to anonymize.
            filename: Override for the fileName sent to the API. Required by
                the API even for plainText uploads (must carry the correct
                extension); defaults to "plain_text.txt" for text input.
            session: Overrides self.config.session for this call only (see
                build_pseudo_request_body).
            session_max_age: Overrides self.config.session_max_age for this
                call only.

        Returns:
            The anonymization service response.
        """

        if text is not None and path is not None:
            raise ValueError("Provide either 'content' or 'path', not both")

        if text is None and path is None:
            raise ValueError("Provide either 'text' or 'path'")

        if path is not None:
            if isinstance(path, str):
                path = Path(path)
            content = path.read_bytes()
            input_data = InputData(
                content=content,
                source="file",
                filename=filename or os.path.basename(path),
            )
        else:
            input_data = InputData(
                content=text, source="string", filename=filename or "plain_text.txt"
            )

        params = {
            "token": self.token,
            "async": str(self.config.a_sync).lower(),
            "returnPseudoKey": str(self.config.return_pseudo_key).lower(),
            "returnEncryptKey": str(self.config.return_encrypt_key).lower(),
        }
        request_body = self.build_pseudo_request_body(
            input_data=input_data, session=session, session_max_age=session_max_age
        )

        response = self._request(
            "POST",
            self.build_pseudo_request_url(),
            params=params,
            json_body=request_body,
        )

        return PseudoResult.model_validate(response.json())

    def build_fetch_request_url(self) -> str:
        """Build the API URL for GET /document/pseudo/fetch (path only, no query)."""
        return f"{self.base_url}{API_BASE_PATH}/document/pseudo/fetch"

    def fetch(
        self,
        pseudo_result: PseudoResult,
        max_retries: int = DEFAULT_FETCH_MAX_RETRIES,
        poll_interval: tuple[float, float] = DEFAULT_FETCH_POLL_INTERVAL,
        start_retention_period: bool = True,
        download_options: DownloadOption | None = None,
        return_pseudo_key: bool = True,
        return_encrypt_key: bool = False,
    ) -> PseudoResult:
        """
        Fetch the result of an asynchronous anonymization request.

        Args:
            pseudo_result: Initial response containing a fetch key.
            max_retries: Maximum number of polling attempts before giving up.
            poll_interval: (min, max) seconds to randomly wait between polls.
            start_retention_period: Whether a successful fetch starts the
                retention period countdown (API default: true).
            download_options: Optional output conversion, e.g. PDF_AS_PDF.
            return_pseudo_key: Whether to ask for the pseudo key (API default:
                true - NOTE this differs from the POST /pseudo endpoint,
                which defaults to false; AnonymizationConfig.return_pseudo_key
                is intentionally NOT reused here for that reason).
            return_encrypt_key: Whether to ask for the encrypt key (API
                default: false).

        Returns:
            Completed anonymization response.

        Raises:
            FetchTimeoutError: If the result is not ready after max_retries attempts.
        """

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
            pseudo_result = PseudoResult.model_validate(response.json())

            if isinstance(pseudo_result.data, InnerResponseData):
                return pseudo_result

            time.sleep(random.uniform(*poll_interval))

        raise FetchTimeoutError(
            f"Anonymization result not ready after {max_retries} attempts "
            f"(fetchKey={fetch_key})"
        )

    def get_ano_result(self, pseudo_result: PseudoResult) -> AnonymizationResult:
        """Convert a completed API response into a usable anonymization result."""

        data = pseudo_result.data
        if not isinstance(data, InnerResponseData):
            raise ValueError(
                "pseudo_result does not contain a completed result; call fetch() first"
            )

        if (
            data.uploadName is not None
            and data.uploadName.startswith("plain_text")
            and data.fileName.startswith("plain_text")
        ):
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
        """Write data to an output file."""

        path = Path(output_file)
        if extension:
            path = path.with_suffix(f".{extension}")

        with open(path, mode=mode) as f:
            f.write(data)

    def manage_data(
        self, pseudo_result: PseudoResult, output_file: str | Path | None = None
    ) -> None:
        """Handle output and key management depending on configuration."""

        if self.config.a_sync:
            print(pseudo_result.data.model_dump_json())
            return

        if not isinstance(pseudo_result.data, InnerResponseData):
            print(
                "Document accepted but not yet processed. Use the fetch key together "
                "with the fetch script to request the anonymized document again"
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

        if self.config.return_pseudo_key:
            pk = pseudo_result.data.pseudoKey
            print(f"pseudoKey: {pk}")
            if pk is not None:
                self.write_to_output(pk, output_file=output_file, extension="pkey")

        if self.config.return_encrypt_key:
            ek = pseudo_result.data.encryptKey
            print(f"encryptKey: {ek}")
            if ek is not None:
                self.write_to_output(ek, output_file=output_file, extension="ekey")

        print(f"{output_file} (SZ: {content_length})")

    def build_depseudo_request_url(self) -> str:
        """Build the API URL for POST /document/depseudo (path only, no query)."""
        return f"{self.base_url}{API_BASE_PATH}/document/depseudo"

    def _read_key_file(self, key_path: str, key_label: str) -> str:
        """Read a depseudonymization key file, raising a clear error on failure."""
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
        """Build the `ApiUploadDepseudoDocument` request payload."""

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
            else None,
            encryptKey=self._read_key_file(
                depseudo_paths.encrypt_key_path, "encrypt key file"
            )
            if depseudo_paths.encrypt_key_path
            else None,
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
    ) -> PseudoResult:
        """
        Reverse anonymization for a previously anonymized document or text.

        Args:
            path: Path to the anonymized document. Mutually exclusive with 'text'.
            text: Anonymized plain text to restore (e.g. an LLM response that
                may contain pseudonyms). Mutually exclusive with 'path'.
                Works with session-based depseudo.
            depseudo_paths: Keys (or key files) for restoring original data.
                Not needed if 'session' is provided. pseudo_key_path alone is
                sufficient; combining it with meta_key fails server-side
                ("No such id ..."), so pass one or the other, not both. For
                metaKey-only depseudo to work, the original pseudo_sync/
                anonymize call must have been made with a_sync=True and both
                pseudo_key_retention_ms and meta_retention_ms set to a
                positive value - either one alone is not enough, and a
                *sync* call discards its pseudoKey server-side immediately
                regardless of these settings. For sync results, use
                pseudo_key_path instead (obtained via return_pseudo_key=True).
            session: Session ID to use instead of pseudoKey/encryptKey/metaKey.
            output_mode: "json" (default, parsed as PseudoResult by this client)
                or "raw" (application/octet-stream - not handled by this client).
            filename: fileName to send with the request. For file input, the
                API tracks documents by the fileName it originally assigned
                (see InnerResponseData.fileName from the pseudo/fetch
                response) - that value, not the local file's name, must be
                passed here. For text input there is no prior document to
                match, so any filename with a supported extension is fine;
                defaults to "plain_text.txt".
        """

        if text is not None and path is not None:
            raise ValueError("Provide either 'text' or 'path', not both")

        if text is None and path is None:
            raise ValueError("Provide either 'text' or 'path'")

        if path is not None:
            if isinstance(path, str):
                path = Path(path)
            content = path.read_bytes()
            input_data = InputData(
                content=content,
                source="file",
                filename=filename or os.path.basename(path),
            )
        else:
            input_data = InputData(
                content=text, source="string", filename=filename or "plain_text.txt"
            )

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

        return PseudoResult.model_validate(response.json())

    def depseudo_text(
        self,
        text: str,
        session: str | None = None,
        depseudo_paths: DepseudoPaths | None = None,
        filename: str | None = None,
    ) -> str:
        """Convenience wrapper: depseudonymize plain text and return the
        restored text directly (decoded from the response), without needing
        to write/read a file."""

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
        session: str | None = None,
        session_max_age: int | None = None,
    ) -> AnonymizationResult:
        """Anonymize text or file content and extract the resulting text."""

        if text is not None and path is not None:
            raise ValueError("Provide either 'content' or 'path', not both")

        if text is None and path is None:
            raise ValueError("Provide either 'text' or 'path'")

        if text is not None:
            pseudo_result = self.pseudo_sync(
                text=text,
                filename=filename,
                session=session,
                session_max_age=session_max_age,
            )
        else:
            pseudo_result = self.pseudo_sync(
                path=path,
                filename=filename,
                session=session,
                session_max_age=session_max_age,
            )

        fetch_pseudo_result = self.fetch(pseudo_result)
        anonymization_result = self.get_ano_result(fetch_pseudo_result)

        return anonymization_result

    def remove_session(self, session_id: str) -> SuccessMessage:
        """PUT /session/remove - remove a session (and its pseudonym mapping) by id."""
        response = self._request(
            "PUT",
            f"{self.base_url}{API_BASE_PATH}/session/remove",
            params={"token": self.token, "session": session_id},
        )
        return SuccessMessage.model_validate(response.json())
