"""Unit tests for glanos_anonymization.core.

These tests exercise individual pieces (models, pure helper methods, string/URL
building, file I/O helpers) in isolation. No network calls are made - even where
a GlanosClient is constructed, only methods that never call `_request` are used.
"""

import base64

import httpx
import pytest
from pydantic import ValidationError

from glanos_anonymization.core import (
    AnonymizationMode,
    AnonymizationResult,
    AnonymizationSettings,
    ApiUploadDepseudoDocument,
    ApiUploadDocument,
    AuthenticationError,
    DataExpiredError,
    DepseudoKeyFileError,
    DepseudoPaths,
    ExcludeField,
    FetchKeyData,
    FoldMode,
    GlanosAPIError,
    GlanosClient,
    ImageReplaceMode,
    InnerResponseData,
    InputData,
    InvalidInputError,
    MarkItDownTextExtractor,
    ProcessingFailedError,
    PseudoResult,
    QuotaExceededError,
    RateLimitedError,
    ResultNotReadyError,
    ServerError,
    SettingsConflictError,
    SuccessMessage,
)

pytestmark = pytest.mark.unit


# --- AnonymizationSettings ----------------------------------------------------


class TestAnonymizationSettings:
    def test_defaults(self):
        settings = AnonymizationSettings()
        assert settings.id is None
        assert settings.mode == AnonymizationMode.TYPE
        assert settings.excludeFields == []
        assert settings.foldMode == FoldMode.MINOR_VAR
        assert settings.imageReplace == ImageReplaceMode.AUTO

    def test_known_exclude_fields_are_coerced_to_enum(self):
        settings = AnonymizationSettings(excludeFields=["mails", "phones"])
        assert settings.excludeFields == [ExcludeField.MAILS, ExcludeField.PHONES]
        assert all(isinstance(f, ExcludeField) for f in settings.excludeFields)

    def test_unknown_enum_values_are_kept_as_strings(self):
        """The enums are open: values a newer API release adds must not be
        rejected by the client."""
        settings = AnonymizationSettings(excludeFields=["zips"], mode="NEW_MODE")
        assert settings.excludeFields == ["zips"]
        assert not isinstance(settings.excludeFields[0], ExcludeField)
        assert settings.mode == "NEW_MODE"

    def test_unknown_fields_are_kept(self):
        settings = AnonymizationSettings(futureField=True)
        assert settings.model_dump()["futureField"] is True


# --- DepseudoPaths ------------------------------------------------------------


class TestDepseudoPaths:
    def test_rejects_unknown_field(self):
        """Regression: cli.py once passed meta_key_path (wrong name) and pydantic
        silently swallowed it. extra='forbid' must turn that into a loud error."""
        with pytest.raises(ValidationError):
            DepseudoPaths(meta_key_path="value")

    def test_accepts_known_fields(self):
        paths = DepseudoPaths(pseudo_key_path="a", meta_key="b", encrypt_key_path="c")
        assert paths.pseudo_key_path == "a"
        assert paths.meta_key == "b"
        assert paths.encrypt_key_path == "c"

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"pseudo_key_path": "a.pkey", "pseudo_key": "pk"},
            {"encrypt_key_path": "a.ekey", "encrypt_key": "ek"},
        ],
    )
    def test_rejects_file_and_value_for_the_same_key(self, kwargs):
        with pytest.raises(ValidationError):
            DepseudoPaths(**kwargs)


class TestDepseudoPathsFromResult:
    def test_prefers_pseudo_and_encrypt_key(self, completed_pseudo_response_json):
        completed_pseudo_response_json["data"].update(
            pseudoKey="pk", encryptKey="ek", metaKey="mk"
        )
        result = PseudoResult.model_validate(completed_pseudo_response_json)

        paths = DepseudoPaths.from_result(result)

        assert paths.pseudo_key == "pk"
        assert paths.encrypt_key == "ek"
        assert paths.meta_key is None

    def test_falls_back_to_meta_key(self, completed_pseudo_response_json):
        result = PseudoResult.model_validate(completed_pseudo_response_json)

        paths = DepseudoPaths.from_result(result)

        assert paths.meta_key == "meta-123"

    def test_result_without_keys_raises(self, completed_pseudo_response_json):
        completed_pseudo_response_json["data"]["metaKey"] = None
        result = PseudoResult.model_validate(completed_pseudo_response_json)

        with pytest.raises(ValueError, match="no keys"):
            DepseudoPaths.from_result(result)

    def test_pending_result_raises(self, accepted_pseudo_response_json):
        result = PseudoResult.model_validate(accepted_pseudo_response_json)

        with pytest.raises(ValueError, match="completed result"):
            DepseudoPaths.from_result(result)


# --- Enums mirror the spec exactly ------------------------------------------


class TestEnumsMatchSpec:
    def test_exclude_field_matches_spec_exactly(self):
        expected = {
            "personNames",
            "streets",
            "locations",
            "countries",
            "addresses",
            "mails",
            "companies",
            "orgs",
            "phones",
            "homepages",
            "filepaths",
            "legals",
            "dates",
            "simpleNumbers",
            "banks",
            "moneyAmounts",
            "idNumbers",
            "percents",
            "measures",
            "products",
            "images",
            "docMeta",
        }
        assert {member.value for member in ExcludeField} == expected

    def test_anonymization_mode_matches_spec(self):
        assert {member.value for member in AnonymizationMode} == {
            "TYPE",
            "TYPE_PRESERVING",
            "XXX",
            "MT",
        }

    def test_fold_mode_matches_spec(self):
        assert {member.value for member in FoldMode} == {
            "STRICT",
            "MINOR_VAR",
            "MINOR_VAR_CASING",
            "MINOR_VAR_AMBIG",
            "SAME_ENTITY",
        }


# --- ApiUploadDocument / ApiUploadDepseudoDocument --------------------------


class TestApiUploadDocument:
    def test_requires_data_or_plain_text(self):
        with pytest.raises(ValidationError):
            ApiUploadDocument(fileName="a.txt")

    def test_accepts_plain_text_only(self):
        doc = ApiUploadDocument(fileName="a.txt", plainText="hello")
        assert doc.data is None
        assert doc.plainText == "hello"

    def test_file_name_is_required(self):
        with pytest.raises(ValidationError):
            ApiUploadDocument(plainText="hello")


class TestApiUploadDepseudoDocument:
    def test_requires_data_or_plain_text(self):
        with pytest.raises(ValidationError):
            ApiUploadDepseudoDocument(fileName="a.txt")


# --- PseudoResult discrimination --------------------------------------------


class TestPseudoResultDiscrimination:
    def test_parses_completed_response_as_inner_response_data(
        self, completed_pseudo_response_json
    ):
        result = PseudoResult.model_validate(completed_pseudo_response_json)
        assert isinstance(result.data, InnerResponseData)

    def test_parses_accepted_response_as_fetch_key_data(
        self, accepted_pseudo_response_json
    ):
        result = PseudoResult.model_validate(accepted_pseudo_response_json)
        assert isinstance(result.data, FetchKeyData)
        assert result.data.fetchKey == "fetch-abc-123"


class TestSuccessEnvelopeUnwrapping:
    """Regression: the deployed API wraps some /session responses in
    {"success": ..., "data": {...}} even where the spec shows a bare schema.
    SuccessMessage must tolerate both shapes."""

    def test_success_message_unwraps_envelope(self):
        wrapped = {"success": True, "data": {"success": True}}
        parsed = SuccessMessage.model_validate(wrapped)
        assert parsed.success is True

    def test_success_message_accepts_flat_shape_too(self):
        parsed = SuccessMessage.model_validate({"success": True})
        assert parsed.success is True


# --- GlanosClient: pure request-building (no network) -----------------------


@pytest.fixture
def client():
    c = GlanosClient(token="tok-123", base_url="https://glanos.example.test/")
    yield c
    c.close()


class TestBuildUrls:
    def test_pseudo_url_uses_be_api_v2_base_path(self, client):
        assert (
            client.build_pseudo_request_url()
            == "https://glanos.example.test/be/api-v2/document/pseudo"
        )

    def test_fetch_url(self, client):
        assert (
            client.build_fetch_request_url()
            == "https://glanos.example.test/be/api-v2/document/pseudo/fetch"
        )

    def test_depseudo_url(self, client):
        assert (
            client.build_depseudo_request_url()
            == "https://glanos.example.test/be/api-v2/document/depseudo"
        )

    def test_trailing_slash_on_base_url_is_stripped(self):
        c = GlanosClient(token="t", base_url="https://x.test///")
        try:
            assert (
                c.build_pseudo_request_url()
                == "https://x.test/be/api-v2/document/pseudo"
            )
        finally:
            c.close()


class TestBuildPseudoRequestBody:
    def test_file_source_encodes_data_as_base64(self, client):
        input_data = InputData(content=b"raw bytes", source="file", filename="doc.txt")
        body = client.build_pseudo_request_body(input_data)

        assert body["fileName"] == "doc.txt"
        assert base64.b64decode(body["data"]) == b"raw bytes"
        assert "plainText" not in body

    def test_string_source_sends_plain_text(self, client):
        input_data = InputData(
            content="hello world", source="string", filename="plain_text.txt"
        )
        body = client.build_pseudo_request_body(input_data)

        assert body["plainText"] == "hello world"
        assert "data" not in body

    def test_settings_are_nested_and_enum_values_are_plain_strings(self, client):
        settings = AnonymizationSettings(
            excludeFields=[ExcludeField.MAILS], mode=AnonymizationMode.XXX
        )
        input_data = InputData(content="hi", source="string", filename="plain_text.txt")

        body = client.build_pseudo_request_body(input_data, settings=settings)

        assert body["settings"]["mode"] == "XXX"
        assert body["settings"]["excludeFields"] == ["mails"]

    def test_inline_settings_leave_out_id_name_and_unknown_fields(self, client):
        """Inline settings apply to one request only - the identity of a stored
        setting (id, name) and fields the spec doesn't know must not be sent."""
        settings = AnonymizationSettings(id="set-1", name="Stored", futureField=True)
        input_data = InputData(content="hi", source="string", filename="plain_text.txt")

        body = client.build_pseudo_request_body(input_data, settings=settings)

        assert "id" not in body["settings"]
        assert "name" not in body["settings"]
        assert "futureField" not in body["settings"]

    def test_without_settings_no_settings_are_sent(self, client):
        """Without settings the installation's default setting applies - the
        client must not send the schema defaults of AnonymizationSettings."""
        input_data = InputData(content="hi", source="string", filename="plain_text.txt")

        body = client.build_pseudo_request_body(input_data)

        assert "settings" not in body
        assert "settingsId" not in body
        assert "session" not in body

    def test_settings_id_and_session_are_top_level(self, client):
        input_data = InputData(content="hi", source="string", filename="plain_text.txt")

        body = client.build_pseudo_request_body(
            input_data,
            settings_id="set-1",
            session="session-xyz",
            session_max_age_ms=60_000,
        )

        assert body["settingsId"] == "set-1"
        assert body["session"] == "session-xyz"
        assert body["sessionMaxAge"] == 60_000

    def test_rejects_settings_together_with_settings_id(self, client):
        # the API would silently ignore settings_id
        input_data = InputData(content="hi", source="string", filename="plain_text.txt")

        with pytest.raises(ValueError, match="settings_id"):
            client.build_pseudo_request_body(
                input_data, settings=AnonymizationSettings(), settings_id="set-1"
            )

    def test_retention_fields_are_top_level_not_nested(self, client):
        input_data = InputData(content="hi", source="string", filename="plain_text.txt")
        body = client.build_pseudo_request_body(
            input_data,
            settings=AnonymizationSettings(),
            data_retention_ms=1000,
            pseudo_key_retention_ms=2000,
            meta_retention_ms=3000,
        )

        assert body["dataRetentionMS"] == 1000
        assert body["pseudoKeyRetentionMS"] == 2000
        assert body["metaRetentionMS"] == 3000
        assert "dataRetentionMS" not in body["settings"]


class TestAnonymizeArgumentChecks:
    """anonymize(), anonymize_with_review() and deanonymize() reject invalid
    arguments before anything is sent - the client below points to an
    unreachable URL, so a request would fail the test with another error."""

    @pytest.mark.parametrize("key", ["return_pseudo_key", "return_encrypt_key"])
    def test_rejects_no_wait_with_return_key(self, client, key):
        with pytest.raises(ValueError, match="fetch"):
            client.anonymize(text="hi", processing="no_wait", **{key: True})

    @pytest.mark.parametrize("processing", ["sync", "no_wait"])
    @pytest.mark.parametrize(
        "option", [{"poll_interval": 1.0}, {"max_attempts": 3}], ids=str
    )
    def test_polling_options_only_apply_to_poll(self, client, processing, option):
        with pytest.raises(ValueError, match="processing='poll'"):
            client.anonymize(text="hi", processing=processing, **option)

    def test_polling_options_do_not_apply_to_the_text_default(self, client):
        # text defaults to "sync"
        with pytest.raises(ValueError, match="processing='poll'"):
            client.anonymize(text="hi", max_attempts=3)

    @pytest.mark.parametrize(
        "option, message",
        [
            ({"max_attempts": 0}, "max_attempts"),
            ({"poll_interval": -1}, "poll_interval"),
        ],
    )
    def test_rejects_invalid_polling_options(self, client, option, message):
        with pytest.raises(ValueError, match=message):
            client.anonymize(text="hi", processing="poll", **option)

    def test_rejects_unknown_processing(self, client):
        with pytest.raises(ValueError, match="later"):
            client.anonymize(text="hi", processing="later")

    def test_rejects_settings_together_with_settings_id(self, client):
        with pytest.raises(ValueError, match="settings_id"):
            client.anonymize(
                text="hi", settings=AnonymizationSettings(), settings_id="set-1"
            )

    def test_rejects_more_than_one_input(self, client):
        with pytest.raises(ValueError, match="exactly one"):
            client.anonymize(text="hi", path="a.txt")

    def test_rejects_no_input(self, client):
        with pytest.raises(ValueError, match="exactly one"):
            client.anonymize()

    def test_bytes_need_a_file_name(self, client):
        with pytest.raises(ValueError, match="filename"):
            client.anonymize(data=b"bytes")

    @pytest.mark.parametrize(
        "reviewers",
        [{}, {"assignee": "anna@example.com", "assigned_group": "legal"}],
        ids=["none", "both"],
    )
    def test_review_needs_exactly_one_reviewer(self, client, reviewers):
        with pytest.raises(ValueError, match="exactly one"):
            client.anonymize_with_review(text="hi", **reviewers)

    def test_deanonymize_rejects_a_result_together_with_text(self, client):
        result = AnonymizationResult(docx_pseudo_result())

        with pytest.raises(ValueError, match="not both"):
            client.deanonymize(result, text="PERSON1")

    def test_deanonymize_text_needs_a_session_or_keys(self, client):
        with pytest.raises(ValueError, match="session"):
            client.deanonymize(text="PERSON1")

    def test_deanonymize_rejects_a_pending_result(
        self, client, accepted_pseudo_response_json
    ):
        pending = AnonymizationResult(
            PseudoResult.model_validate(accepted_pseudo_response_json)
        )

        with pytest.raises(ResultNotReadyError, match="fetch-abc-123"):
            client.deanonymize(pending)


class TestBuildDepseudoRequestBody:
    def test_reads_key_files(self, client, pseudo_key_file, encrypt_key_file):
        input_data = InputData(content=b"cipher", source="file", filename="doc.txt")
        depseudo_paths = DepseudoPaths(
            pseudo_key_path=str(pseudo_key_file), encrypt_key_path=str(encrypt_key_file)
        )

        body = client.build_depseudo_request_body(input_data, depseudo_paths)

        assert body["pseudoKey"] == "pseudo-key-value"
        assert body["encryptKey"] == "encrypt-key-value"

    def test_meta_key_is_used_as_a_raw_value_not_a_file_path(self, client):
        """Regression: meta_key is documented as 'metaKey as returned by
        pseudonymization' - a value, not something to read from disk."""
        input_data = InputData(content=b"cipher", source="file", filename="doc.txt")
        depseudo_paths = DepseudoPaths(meta_key="the-actual-meta-key-value")

        body = client.build_depseudo_request_body(input_data, depseudo_paths)

        assert body["metaKey"] == "the-actual-meta-key-value"

    def test_missing_key_file_raises_file_not_found(self, client, tmp_path):
        input_data = InputData(content=b"cipher", source="file", filename="doc.txt")
        depseudo_paths = DepseudoPaths(
            pseudo_key_path=str(tmp_path / "does-not-exist.pkey")
        )

        with pytest.raises(FileNotFoundError):
            client.build_depseudo_request_body(input_data, depseudo_paths)

    def test_unreadable_key_file_raises_depseudo_key_file_error(
        self, client, pseudo_key_file, monkeypatch
    ):
        def raise_oserror(*args, **kwargs):
            raise OSError("permission denied")

        monkeypatch.setattr("builtins.open", raise_oserror)

        input_data = InputData(content=b"cipher", source="file", filename="doc.txt")
        depseudo_paths = DepseudoPaths(pseudo_key_path=str(pseudo_key_file))

        with pytest.raises(DepseudoKeyFileError):
            client.build_depseudo_request_body(input_data, depseudo_paths)

    def test_session_is_included_when_given(self, client):
        input_data = InputData(content=b"cipher", source="file", filename="doc.txt")
        body = client.build_depseudo_request_body(input_data, session="session-xyz")
        assert body["session"] == "session-xyz"


# --- _result ------------------------------------------------------------------


class FakeTextExtractor:
    def __init__(self):
        self.called_with = None

    def extract_text(self, pseudo_result) -> str:
        self.called_with = pseudo_result
        return "extracted via fake extractor"


def docx_pseudo_result() -> PseudoResult:
    return PseudoResult.model_validate(
        {
            "success": True,
            "data": {
                "data": base64.b64encode(b"docx bytes").decode("ascii"),
                "fileName": "report.docx",
                "uploadName": "report.docx",
            },
        }
    )


class TestResult:
    def test_plain_text_upload_is_decoded_directly(
        self, client, completed_pseudo_response_json
    ):
        result = PseudoResult.model_validate(completed_pseudo_response_json)
        ano_result = client._result(result)
        expected_text = base64.b64decode(
            completed_pseudo_response_json["data"]["data"]
        ).decode("utf-8")
        assert ano_result.extract_text() == expected_text

    def test_does_not_extract_text(self, client):
        extractor = FakeTextExtractor()
        client.text_extractor = extractor

        client._result(docx_pseudo_result())

        assert extractor.called_with is None

    def test_non_plain_text_upload_uses_text_extractor(self, client):
        extractor = FakeTextExtractor()
        client.text_extractor = extractor
        result = docx_pseudo_result()

        ano_result = client._result(result)

        assert ano_result.extract_text() == "extracted via fake extractor"
        assert extractor.called_with is result

    def test_pending_result_is_returned_pending(
        self, client, accepted_pseudo_response_json
    ):
        result = PseudoResult.model_validate(accepted_pseudo_response_json)

        ano_result = client._result(result)

        assert not ano_result.ready
        assert ano_result.fetch_key == "fetch-abc-123"


# --- AnonymizationResult ------------------------------------------------------


class TestAnonymizationResult:
    def test_exposes_document_bytes_and_file_name(self, completed_pseudo_response_json):
        pseudo_result = PseudoResult.model_validate(completed_pseudo_response_json)

        result = AnonymizationResult(pseudo_result)

        assert result.data == base64.b64decode(
            completed_pseudo_response_json["data"]["data"]
        )
        assert result.file_name == "plain_text.txt"

    def test_exposes_keys_and_session(self):
        pseudo_result = PseudoResult.model_validate(
            {
                "success": True,
                "data": {
                    "data": "",
                    "fileName": "plain_text.txt",
                    "pseudoKey": "pk",
                    "encryptKey": "ek",
                    "metaKey": "mk",
                    "session": "s1",
                },
            }
        )

        result = AnonymizationResult(pseudo_result)

        assert result.pseudo_key == "pk"
        assert result.encrypt_key == "ek"
        assert result.meta_key == "mk"
        assert result.session == "s1"

    def test_completed_result_is_ready(self, completed_pseudo_response_json):
        result = AnonymizationResult(
            PseudoResult.model_validate(completed_pseudo_response_json)
        )

        assert result.ready
        assert result.fetch_key is None

    def test_pending_result_carries_only_the_fetch_key(
        self, accepted_pseudo_response_json
    ):
        result = AnonymizationResult(
            PseudoResult.model_validate(accepted_pseudo_response_json)
        )

        assert not result.ready
        assert result.fetch_key == "fetch-abc-123"

    @pytest.mark.parametrize(
        "read",
        [
            lambda r: r.data,
            lambda r: r.file_name,
            lambda r: r.pseudo_key,
            lambda r: r.session,
            lambda r: r.extract_text(),
        ],
        ids=["data", "file_name", "pseudo_key", "session", "extract_text"],
    )
    def test_pending_result_raises_result_not_ready(
        self, accepted_pseudo_response_json, read
    ):
        result = AnonymizationResult(
            PseudoResult.model_validate(accepted_pseudo_response_json)
        )

        with pytest.raises(ResultNotReadyError) as excinfo:
            read(result)
        assert excinfo.value.fetch_key == "fetch-abc-123"

    def test_text_keeps_the_newline_the_api_appends(self):
        pseudo_result = PseudoResult.model_validate(
            {
                "success": True,
                "data": {
                    "data": base64.b64encode(b"PERSON1 was here\n").decode("ascii"),
                    "fileName": "plain_text_anonym.txt",
                },
            }
        )

        result = AnonymizationResult(pseudo_result)

        assert result.extract_text() == "PERSON1 was here\n"

    def test_text_result_never_uses_the_extractor(self):
        pseudo_result = PseudoResult.model_validate(
            {
                "success": True,
                "data": {
                    "data": base64.b64encode(b"hello").decode("ascii"),
                    "fileName": "notes_anonym.TXT",
                },
            }
        )
        extractor = FakeTextExtractor()

        result = AnonymizationResult(pseudo_result, extractor)

        assert result.extract_text() == "hello"
        assert extractor.called_with is None

    def test_extractor_argument_overrides_the_client_extractor(self):
        client_extractor = FakeTextExtractor()
        call_extractor = FakeTextExtractor()
        result = AnonymizationResult(docx_pseudo_result(), client_extractor)

        result.extract_text(call_extractor)

        assert call_extractor.called_with is result.pseudo_result
        assert client_extractor.called_with is None

    def test_has_no_text_attribute(self):
        # result.text was replaced by extract_text(); old code must fail loudly.
        result = AnonymizationResult(docx_pseudo_result())

        with pytest.raises(AttributeError):
            result.text


# --- _parse_pseudo_response -----------------------------------------------------


class TestParsePseudoResponse:
    def test_http_202_is_pending(self, accepted_pseudo_response_json):
        response = httpx.Response(202, json=accepted_pseudo_response_json)

        result = GlanosClient._parse_pseudo_response(response)

        assert result.is_pending
        assert result.data.fetchKey == "fetch-abc-123"

    def test_http_200_with_document_is_completed(self, completed_pseudo_response_json):
        response = httpx.Response(200, json=completed_pseudo_response_json)

        result = GlanosClient._parse_pseudo_response(response)

        assert not result.is_pending
        assert isinstance(result.data, InnerResponseData)

    def test_failed_processing_raises_with_fail_reason(
        self, completed_pseudo_response_json
    ):
        completed_pseudo_response_json["data"].update(
            success=False, failReason="Invalid document"
        )
        response = httpx.Response(200, json=completed_pseudo_response_json)

        with pytest.raises(ProcessingFailedError, match="Invalid document") as exc_info:
            GlanosClient._parse_pseudo_response(response)
        assert exc_info.value.fail_reason == "Invalid document"


# --- _raise_for_status --------------------------------------------------------


class TestRaiseForStatus:
    @pytest.mark.parametrize(
        "status_code,expected_exc",
        [
            (401, AuthenticationError),
            (402, QuotaExceededError),
            (403, AuthenticationError),
            (410, DataExpiredError),
            (429, RateLimitedError),
            (500, ServerError),
            (503, ServerError),
            (404, InvalidInputError),
        ],
    )
    def test_maps_status_codes(self, client, status_code, expected_exc):
        response = httpx.Response(
            status_code, request=httpx.Request("GET", "https://x.test"), text="boom"
        )
        with pytest.raises(expected_exc) as exc_info:
            client._raise_for_status(response)
        assert exc_info.value.status_code == status_code

    def test_authentication_error_is_an_invalid_input_error(self):
        assert issubclass(AuthenticationError, InvalidInputError)

    def test_other_error_codes_raise_http_status_error_with_body_text(self, client):
        """GlanosAPIError stays an httpx.HTTPStatusError, so code catching that
        keeps working."""
        response = httpx.Response(
            400,
            request=httpx.Request("GET", "https://x.test"),
            text="fileName does not match",
        )
        with pytest.raises(httpx.HTTPStatusError, match="fileName does not match"):
            client._raise_for_status(response)

    def test_api_message_is_extracted_from_json_body(self, client):
        response = httpx.Response(
            400,
            request=httpx.Request("GET", "https://x.test"),
            json={"success": False, "message": "Invalid document"},
        )
        with pytest.raises(GlanosAPIError) as exc_info:
            client._raise_for_status(response)
        assert exc_info.value.api_message == "Invalid document"

    def test_settings_mode_conflict_raises_settings_conflict_error(self, client):
        response = httpx.Response(
            400,
            request=httpx.Request("POST", "https://x.test"),
            json={
                "success": False,
                "message": "Session settings mode and provided settings mode "
                "must be equal",
            },
        )
        with pytest.raises(SettingsConflictError):
            client._raise_for_status(response)

    def test_error_message_does_not_contain_the_token(self, client):
        request = httpx.Request(
            "GET", "https://x.test/be/api-v2/setting/get?token=secret-tok&setting=s"
        )
        response = httpx.Response(400, request=request, text="bad request")

        with pytest.raises(InvalidInputError) as exc_info:
            client._raise_for_status(response)
        assert "secret-tok" not in str(exc_info.value)

    def test_success_codes_do_not_raise(self, client):
        response = httpx.Response(200, request=httpx.Request("GET", "https://x.test"))
        client._raise_for_status(response)  # must not raise


# --- client lifecycle ----------------------------------------------------------


class TestClientLifecycle:
    def test_default_text_extractor_is_markitdown(self, client):
        assert isinstance(client.text_extractor, MarkItDownTextExtractor)

    def test_default_text_extractor_is_not_created_with_the_client(self, client):
        assert client._text_extractor is None

    def test_custom_text_extractor_is_used(self):
        extractor = FakeTextExtractor()
        with GlanosClient(
            token="t", base_url="https://x.test", text_extractor=extractor
        ) as c:
            assert c.text_extractor is extractor

    def test_context_manager_closes_client(self):
        with GlanosClient(token="t", base_url="https://x.test") as c:
            assert c.client is not None
        assert c.client is None

    def test_close_is_idempotent(self, client):
        client.close()
        client.close()  # must not raise
        assert client.client is None
