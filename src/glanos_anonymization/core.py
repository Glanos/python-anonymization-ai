import os
import base64
import random
import httpx
from typing import Literal, Union, Protocol
from pydantic import BaseModel
from pathlib import Path
import json
import tempfile
from markitdown import MarkItDown
import time


class AnonymizationConfig(BaseModel):
    """
    Configuration settings for the anonymization process.

    Attributes:
        anonymization_mode: Defines the anonymization strategy used by the service.
        a_sync: Determines whether the anonymization request is processed asynchronously.
        language: Optional language hint for the anonymization engine.
        no_ssl_verify: Disables SSL certificate verification for HTTP requests.
        return_pseudo_key: Requests returning a pseudo key from the anonymization service.
        return_encrypt_key: Requests returning an encryption key from the anonymization service.
        data_retention_ms: Defines how long anonymized data is retained.
        pseudo_key_retention_ms: Defines how long pseudo keys are retained.
        meta_retention_ms: Defines how long metadata is retained.
        exclude_fields: List of fields that should be excluded from anonymization.
    """

    anonymization_mode: str = "TYPE"
    a_sync: bool = True
    language: str = None
    no_ssl_verify: bool = False
    return_pseudo_key: bool = False
    return_encrypt_key: bool = False
    data_retention_ms: int = 0
    pseudo_key_retention_ms: int = 0
    meta_retention_ms: int = 0
    exclude_fields: list[str] = []

    def __post_init__(self):
        """Validate configuration parameter combinations."""
        if self.a_sync and (self.return_pseudo_key or self.return_encrypt_key):
            raise ValueError("Invalid parameter combination. If a_sync = True then ...")


SourceType = Literal["string", "file"]


class InputData(BaseModel):
    """
    Represents input data submitted for anonymization.

    Attributes:
        content: Raw text or binary file content.
        source: Defines whether the input originates from a string or a file.
        filename: Optional filename for file-based inputs.
    """

    content: str | bytes
    source: SourceType
    filename: str | None = None


class DepseudoPaths(BaseModel):
    """
    Contains paths to keys required for reversing anonymization.

    Attributes:
        pseudo_key_path: Path to the pseudo key file.
        meta_key: Metadata key used for depseudonymization.
        encrypt_key_path: Path to the encryption key file.
    """

    pseudo_key_path: str | None = None
    meta_key: str | None = None
    encrypt_key_path: str | None = None


class InnerResponseData(BaseModel):
    """Contains the completed anonymization response data."""

    data: str
    metaKey: str | None = None
    encryptKey: str | None = None
    pseudoKey: str | None = None
    fileName: str
    uploadName: str | None = None
    language: str | None = None


class FetchKeyData(BaseModel):
    """Contains the fetch key returned for asynchronous anonymization jobs."""

    fetchKey: str


class PseudoResult(BaseModel):
    """
    Represents the response returned by the anonymization service.

    The data field contains either the completed anonymized document
    or a fetch key for retrieving the result later.
    """

    success: bool
    data: Union[InnerResponseData, FetchKeyData]


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
    """
    Protocol defining an interface for extracting text from anonymized files.
    """

    def extract_text(self, pseudo_result: PseudoResult) -> str: ...


class MarkItDownTextExtractor(TextExtractor):
    """Extracts text content from anonymized files using MarkItDown."""

    def __init__(self):
        """Initialize the MarkItDown text converter."""
        self.mid = MarkItDown()

    def extract_text(self, pseudo_result: PseudoResult) -> str:
        """
        Extract text from an anonymized file response.

        Args:
            pseudo_result: Response containing base64 encoded file data.

        """

        text_data = base64.b64decode(pseudo_result.data.data)
        suffix = Path(pseudo_result.data.fileName).suffix

        with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
            tmp.write(text_data)
            tmp.flush()

            text_content = self.mid.convert(tmp.name).text_content

        return text_content


class GlanosClient:
    """
    Client for communicating with the Glanos anonymization API.
    """

    def __init__(
        self,
        token: str,
        base_url: str,
        config: AnonymizationConfig | None = None,
        text_extractor: TextExtractor | None = None,
    ) -> None:
        """
        Initialize the Glanos API client.

        Args:
            token: Authentication token for the API.
            base_url: Base URL of the Glanos service.
            config: Optional anonymization configuration.
            text_extractor: Optional custom text extractor.
        """
        self.token = token
        self.base_url = base_url
        self.config = config
        self.text_extractor = text_extractor
        self.client = httpx.Client(verify=self.config.no_ssl_verify)

    @property
    def config(self):
        """Get the current anonymization configuration."""
        return self._config

    @config.setter
    def config(self, config: AnonymizationConfig | None):
        """Set the anonymization configuration."""
        if config is None:
            self._config = AnonymizationConfig()
        else:
            self._config = config

    @property
    def text_extractor(self):
        """Get the configured text extractor."""
        return self._text_extractor

    @text_extractor.setter
    def text_extractor(self, extractor: TextExtractor | None):
        """Set the text extractor implementation."""
        if extractor is None:
            self._text_extractor = MarkItDownTextExtractor()
        else:
            self._text_extractor = extractor

    def pseudo_sync(
        self, text: str | None = None, path: str | Path | None = None
    ) -> PseudoResult:
        """
        Submit text or a file for anonymization.

        Args:
            text: Text content to anonymize.
            path: Path to a file to anonymize.

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
            filename = os.path.basename(path)
            input_data = InputData(content=content, source="file", filename=filename)
        else:
            input_data = InputData(content=text, source="string")

        request_url = self.build_pseudo_request_url()
        request_body = self.build_pseudo_request_body(input_data=input_data)

        response = self.client.post(
            request_url,
            content=json.dumps(request_body),
            headers={"Content-type": "application/json"},
        )
        
        response.raise_for_status()

        return PseudoResult.model_validate(response.json())

    def fetch(self, pseudo_result: PseudoResult) -> PseudoResult:
        """
        Fetch the result of an asynchronous anonymization request.

        Args:
            pseudo_result: Initial response containing a fetch key.

        Returns:
            Completed anonymization response.
        """

        if isinstance(pseudo_result.data, InnerResponseData):
            return pseudo_result
        else:
            fetch_request_url = self.build_fetch_request_url(
                pseudo_result.data.fetchKey
            )
            wait_factor = 2

            while True:
                response = self.client.get(
                    fetch_request_url, headers={"Content-type": "application/json"}
                )
                response.raise_for_status()
                pseudo_result = PseudoResult.model_validate(response.json())

                if isinstance(pseudo_result.data, InnerResponseData):
                    break

                time.sleep(random.uniform(1, 2))
                wait_factor += 1

            return pseudo_result

    def get_ano_result(self, pseudo_result: PseudoResult) -> AnonymizationResult:
        """
        Convert an API response into a usable anonymization result.

        Args:
            pseudo_result: Completed anonymization response.

        Returns:
            Extracted anonymized text and metadata.
        """

        if pseudo_result.data.uploadName.startswith(
            "plain_text"
        ) and pseudo_result.data.fileName.startswith("plain_text"):
            text_content = base64.b64decode(pseudo_result.data.data).decode("utf-8")
        else:
            text_content = self.text_extractor.extract_text(pseudo_result)

        return AnonymizationResult(text_content, pseudo_result)

    def write_to_output(self, data, output_file=None, extension=None, mode="w") -> None:
        """
        Write data to an output file.

        Args:
            data: Data to write.
            output_file: Destination file path.
            extension: Optional file extension override.
            mode: File open mode.
        """

        filename = (
            f"{output_file.split('.')[0]}.{extension}" if extension else output_file
        )

        with open(filename, mode=mode) as f:
            f.write(data)

    def manage_data(self, pseudo_result: PseudoResult, output_file=None) -> None:
        """
        Handle output and key management depending on configuration.

        Args:
            pseudo_result: Anonymization response.
            output_file: Optional output file path.
        """

        if self.config.a_sync:
            print(pseudo_result.data.model_dump_json())
            exit(0)
        else:
            if isinstance(pseudo_result.data, InnerResponseData):
                content = base64.b64decode(pseudo_result.data.data)
                content_length = len(content)

                self.write_to_output(content, output_file=output_file, mode="wb")

                mk = pseudo_result.data.metaKey
                print(f"metaKey: {mk}")

                if self.config.return_pseudo_key:
                    pk = pseudo_result.data.pseudoKey
                    print(f"pseudoKey: {pk}")
                    self.write_to_output(pk, output_file=output_file, extension="pkey")

                if self.config.return_encrypt_key:
                    ek = pseudo_result.data.encryptKey
                    print(f"encryptKey: {ek}")
                    self.write_to_output(ek, output_file=output_file, extension="ekey")

                print(f"{output_file} (SZ: {content_length})")

                exit(0)
            else:
                print(
                    "Document accepted but not yet processed. Use the fetch key together with the fetch script to request the anonymized document again"
                )
                print(pseudo_result.data.model_dump_json())
                exit(0)

    def build_depseudo_request_url(self) -> str:
        """
        Build the API URL for depseudonymization requests.

        Returns:
            Fully qualified depseudonymization endpoint URL.
        """
        ...

        request_url = f"{self.base_url}/be/api-v2/document/depseudo?token={self.token}&outputMode=json"

        return request_url

    def build_depseudo_request_body(
        self, input_data: InputData, depseudo_paths: DepseudoPaths
    ) -> dict:
        """
        Build the request payload for depseudonymization.

        Args:
            input_data: Input document data.
            depseudo_paths: Paths to required depseudonymization keys.

        Returns:
            Request payload dictionary.
        """

        body = {}

        body["data"] = base64.b64encode(input_data.content).decode("utf-8")
        body["fileName"] = os.path.basename(input_data.filename)

        if depseudo_paths.pseudo_key_path is not None:
            if not os.path.isfile(depseudo_paths.pseudo_key_path):
                print(
                    f"{depseudo_paths.pseudo_key_path} does not exists or is not a file!"
                )
                exit(1)
            else:
                try:
                    with open(depseudo_paths.pseudo_key_path) as f:
                        body["pseudoKey"] = f.read()
                except Exception as e:
                    print(
                        f"There was an error while trying to open the file at {depseudo_paths.pseudo_key_path}: {e}"
                    )
                    exit(1)

        if depseudo_paths.encrypt_key_path is not None:
            if not os.path.isfile(depseudo_paths.encrypt_key_path):
                print(
                    f"{depseudo_paths.encrypt_key_path} does not exists or is not a file!"
                )
                exit(1)
            else:
                try:
                    with open(depseudo_paths.encrypt_key_path) as f:
                        body["encryptKey"] = f.read()
                except Exception as e:
                    print(
                        f"There was an error while trying to open the file at {depseudo_paths.encrypt_key_path}: {e}"
                    )
                    exit(1)

        if depseudo_paths.meta_key_path is not None:
            if not os.path.isfile(depseudo_paths.meta_key_path):
                print(
                    f"{depseudo_paths.meta_key_path} does not exists or is not a file!"
                )
                exit(1)
            else:
                try:
                    with open(depseudo_paths.meta_key_path) as f:
                        body["metaKey"] = f.read()
                except Exception as e:
                    print(
                        f"There was an error while trying to open the file at {depseudo_paths.meta_key_path}: {e}"
                    )
                    exit(1)

        return body

    def depseudo(self, path: str | Path, depseudo_paths: DepseudoPaths) -> PseudoResult:
        """
        Reverse anonymization for a previously anonymized document.

        Args:
            path: Path to the anonymized document.
            depseudo_paths: Required keys for restoring original data.

        Returns:
            Depseudonymization service response.
        """

        if isinstance(path, str):
            path = Path(path)

        content = path.read_bytes()
        filename = os.path.basename(path)
        input_data = InputData(content=content, source="file", filename=filename)

        request_url = self.build_depseudo_request_url()
        request_body = self.build_depseudo_request_body(
            input_data=input_data, depseudo_paths=depseudo_paths
        )

        response = self.client.post(
            request_url,
            content=json.dumps(request_body),
            headers={"Content-type": "application/json"},
        )

        response.raise_for_status()

        return PseudoResult.model_validate(response.json())

    def anonymize(
        self, text: str | None = None, path: str | Path | None = None
    ) -> AnonymizationResult:
        """
        Anonymize text or file content.

        Args:
            text: Text content to anonymize.
            path: File path to anonymize.

        Returns:
            Anonymization result containing text and metadata.
        """

        if text is not None and path is not None:
            raise ValueError("Provide either 'content' or 'path', not both")

        if text is None and path is None:
            raise ValueError("Provide either 'text' or 'path'")

        if text is not None:
            pseudo_result = self.pseudo_sync(text=text)
        else:
            pseudo_result = self.pseudo_sync(path=path)

        fetch_pseudo_result = self.fetch(pseudo_result)
        anonymization_result = self.get_ano_result(fetch_pseudo_result)

        return anonymization_result

    def build_pseudo_request_url(self) -> str:
        """
        Build the API URL for anonymization requests.

        Returns:
            Fully qualified anonymization endpoint URL.
        """

        request_url = (
            f"{self.base_url}/be/api-v2/document/pseudo?"
            f"token={self.token}"
            f"&async={str(self.config.a_sync).lower()}"
            f"&returnPseudoKey={str(self.config.return_pseudo_key).lower()}"
            f"&returnEncryptKey={str(self.config.return_encrypt_key).lower()}"
        )

        return request_url

    def build_fetch_request_url(self, fetch_key) -> str:
        """
        Build the API URL used to fetch asynchronous results.

        Args:
            fetch_key: Fetch identifier returned by the service.

        Returns:
            Fully qualified fetch endpoint URL.
        """

        request_url = (
            f"{self.base_url}/be/api-v2/document/pseudo/fetch?"
            f"token={self.token}"
            f"&fetchKey={fetch_key}"
            f"&returnPseudoKey={str(self.config.return_pseudo_key).lower()}"
            f"&returnEncryptKey={str(self.config.return_encrypt_key).lower()}"
        )

        return request_url

    def build_pseudo_request_body(self, input_data: InputData) -> dict:
        """
        Build the request payload for anonymization.

        Args:
            input_data: Input data to anonymize.

        Returns:
            Request payload dictionary.
        """

        body = {}

        if input_data.source == "file":
            body["data"] = base64.b64encode(input_data.content).decode("utf-8")
            body["fileName"] = input_data.filename
        else:
            body["plainText"] = input_data.content

        if self.config.language:
            body["language"] = self.config.language

        settings = {
            "excludeFields": list(set(self.config.exclude_fields)),
            "mode": self.config.anonymization_mode,
        }

        body["settings"] = settings
        body["dataRetentionMS"] = self.config.data_retention_ms
        body["pseudoKeyRetentionMS"] = self.config.pseudo_key_retention_ms
        body["metaRetentionMS"] = self.config.meta_retention_ms

        return body
