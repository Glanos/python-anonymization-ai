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
    AnonymizationConfig,
    AnonymizationSettings,
    AnonymizationMode,
    ApiUploadDepseudoDocument,
    ApiUploadDocument,
    BlankOutMode,
    DataExpiredError,
    DepseudoKeyFileError,
    DepseudoPaths,
    ExcludeField,
    FetchKeyData,
    FoldMode,
    GlanosClient,
    ImageReplaceMode,
    InnerResponseData,
    InputData,
    PseudoResult,
    QuotaExceededError,
    RateLimitedError,
    SuccessMessage,
)

pytestmark = pytest.mark.unit


# --- AnonymizationConfig ----------------------------------------------------


class TestAnonymizationConfig:
    def test_defaults(self):
        config = AnonymizationConfig()
        assert config.a_sync is True
        assert config.anonymization_mode == AnonymizationMode.TYPE
        assert config.exclude_fields == []
        assert config.no_ssl_verify is False

    def test_rejects_async_with_return_pseudo_key(self):
        with pytest.raises(ValidationError):
            AnonymizationConfig(a_sync=True, return_pseudo_key=True)

    def test_rejects_async_with_return_encrypt_key(self):
        with pytest.raises(ValidationError):
            AnonymizationConfig(a_sync=True, return_encrypt_key=True)

    def test_allows_sync_with_return_keys(self):
        config = AnonymizationConfig(
            a_sync=False, return_pseudo_key=True, return_encrypt_key=True
        )
        assert config.return_pseudo_key is True
        assert config.return_encrypt_key is True

    def test_validate_assignment_catches_invalid_combo_after_construction(self):
        """Regression: cli mutates a_sync after construction; the
        combination check must still fire, not just at __init__ time."""
        config = AnonymizationConfig(a_sync=False, return_pseudo_key=True)
        with pytest.raises(ValidationError):
            config.a_sync = True

    def test_exclude_fields_coerced_to_enum(self):
        config = AnonymizationConfig(exclude_fields=["mails", "phones"])
        assert config.exclude_fields == [ExcludeField.MAILS, ExcludeField.PHONES]

    def test_rejects_unknown_exclude_field(self):
        with pytest.raises(ValidationError):
            AnonymizationConfig(exclude_fields=["zips"])  # not part of the spec's enum

    def test_to_settings_maps_all_fields(self):
        config = AnonymizationConfig(
            anonymization_mode=AnonymizationMode.MT,
            exclude_fields=[ExcludeField.MAILS],
            workflow="my-workflow",
            fold_mode=FoldMode.STRICT,
            blank_out_mode=BlankOutMode.BLACK,
            image_replace=ImageReplaceMode.BLANK,
            add_no_translate_markup=True,
            custom=["opt1"],
        )

        settings = config.to_settings()

        assert isinstance(settings, AnonymizationSettings)
        assert settings.mode == AnonymizationMode.MT
        assert settings.excludeFields == [ExcludeField.MAILS]
        assert settings.workflow == "my-workflow"
        assert settings.foldMode == FoldMode.STRICT
        assert settings.blankOutMode == BlankOutMode.BLACK
        assert settings.imageReplace == ImageReplaceMode.BLANK
        assert settings.addNoTranslateMarkup is True
        assert settings.custom == ["opt1"]


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

    def test_settings_are_nested_and_enum_values_are_plain_strings(self):
        config = AnonymizationConfig(
            exclude_fields=[ExcludeField.MAILS],
            anonymization_mode=AnonymizationMode.XXX,
        )
        client = GlanosClient(token="t", base_url="https://x.test", config=config)
        try:
            input_data = InputData(
                content="hi", source="string", filename="plain_text.txt"
            )
            body = client.build_pseudo_request_body(input_data)

            assert body["settings"]["mode"] == "XXX"
            assert body["settings"]["excludeFields"] == ["mails"]
        finally:
            client.close()

    def test_retention_fields_are_top_level_not_nested(self, client):
        client.config = AnonymizationConfig(
            data_retention_ms=1000, pseudo_key_retention_ms=2000, meta_retention_ms=3000
        )
        input_data = InputData(content="hi", source="string", filename="plain_text.txt")
        body = client.build_pseudo_request_body(input_data)

        assert body["dataRetentionMS"] == 1000
        assert body["pseudoKeyRetentionMS"] == 2000
        assert body["metaRetentionMS"] == 3000
        assert "dataRetentionMS" not in body.get("settings", {})


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


# --- write_to_output / manage_data (pure I/O, no network) -------------------


class TestWriteToOutput:
    def test_writes_bytes(self, client, tmp_path):
        out = tmp_path / "result.bin"
        client.write_to_output(b"hello", output_file=out, mode="wb")
        assert out.read_bytes() == b"hello"

    def test_extension_override_replaces_suffix(self, client, tmp_path):
        out = tmp_path / "result.bin"
        client.write_to_output("secret", output_file=out, extension="pkey")
        assert (tmp_path / "result.pkey").read_text() == "secret"
        assert not out.exists()


class TestManageData:
    def test_async_prints_raw_json_and_writes_nothing(
        self, client, tmp_path, capsys, accepted_pseudo_response_json
    ):
        client.config = AnonymizationConfig(a_sync=True)
        result = PseudoResult.model_validate(accepted_pseudo_response_json)

        client.manage_data(result, output_file=tmp_path / "out.txt")

        assert not (tmp_path / "out.txt").exists()
        captured = capsys.readouterr()
        assert "fetchKey" in captured.out

    def test_pending_result_prints_notice(
        self, client, capsys, accepted_pseudo_response_json
    ):
        client.config = AnonymizationConfig(a_sync=False)
        result = PseudoResult.model_validate(accepted_pseudo_response_json)

        client.manage_data(result, output_file="ignored.txt")

        captured = capsys.readouterr()
        assert "not yet processed" in captured.out

    def test_completed_result_writes_file_and_prints_meta_key(
        self, client, tmp_path, capsys, completed_pseudo_response_json
    ):
        client.config = AnonymizationConfig(a_sync=False)
        result = PseudoResult.model_validate(completed_pseudo_response_json)
        out = tmp_path / "out.txt"

        client.manage_data(result, output_file=out)

        expected_bytes = base64.b64decode(
            completed_pseudo_response_json["data"]["data"]
        )
        assert out.read_bytes() == expected_bytes
        captured = capsys.readouterr()
        assert "metaKey: meta-123" in captured.out
        assert not (tmp_path / "out.pkey").exists()
        # No session was used for this upload, so nothing should be printed.
        assert "session:" not in captured.out

    def test_prints_session_when_present(self, client, tmp_path, capsys):
        client.config = AnonymizationConfig(a_sync=False)
        payload = {
            "success": True,
            "data": {
                "data": base64.b64encode(b"content").decode("ascii"),
                "fileName": "doc.docx",
                "uploadName": "doc.docx",
                "session": "session-abc",
            },
        }
        result = PseudoResult.model_validate(payload)

        client.manage_data(result, output_file=tmp_path / "out.docx")

        assert "session: session-abc" in capsys.readouterr().out

    def test_requested_key_not_returned_by_api_does_not_crash(
        self, client, tmp_path, capsys
    ):
        """Regression: the real API can omit pseudoKey/encryptKey even when
        return_pseudo_key/return_encrypt_key were requested. manage_data must
        not crash trying to write a None value to disk."""
        client.config = AnonymizationConfig(
            a_sync=False, return_pseudo_key=True, return_encrypt_key=True
        )
        payload = {
            "success": True,
            "data": {
                "data": base64.b64encode(b"content").decode("ascii"),
                "fileName": "doc.docx",
                "uploadName": "doc.docx",
                "pseudoKey": "pk-value",
                "encryptKey": None,
            },
        }
        result = PseudoResult.model_validate(payload)
        out = tmp_path / "out.docx"

        client.manage_data(result, output_file=out)  # must not raise

        assert (tmp_path / "out.pkey").exists()
        assert not (tmp_path / "out.ekey").exists()
        assert "encryptKey: None" in capsys.readouterr().out

    def test_return_pseudo_and_encrypt_key_writes_extra_files(self, client, tmp_path):
        client.config = AnonymizationConfig(
            a_sync=False, return_pseudo_key=True, return_encrypt_key=True
        )
        payload = {
            "success": True,
            "data": {
                "data": base64.b64encode(b"content").decode("ascii"),
                "fileName": "doc.docx",
                "uploadName": "doc.docx",
                "pseudoKey": "pk-value",
                "encryptKey": "ek-value",
            },
        }
        result = PseudoResult.model_validate(payload)
        out = tmp_path / "out.docx"

        client.manage_data(result, output_file=out)

        assert (tmp_path / "out.pkey").read_text() == "pk-value"
        assert (tmp_path / "out.ekey").read_text() == "ek-value"


# --- get_ano_result -----------------------------------------------------------


class FakeTextExtractor:
    def __init__(self):
        self.called_with = None

    def extract_text(self, pseudo_result) -> str:
        self.called_with = pseudo_result
        return "extracted via fake extractor"


class TestGetAnoResult:
    def test_plain_text_upload_is_decoded_directly(
        self, client, completed_pseudo_response_json
    ):
        result = PseudoResult.model_validate(completed_pseudo_response_json)
        ano_result = client.get_ano_result(result)
        expected_text = base64.b64decode(
            completed_pseudo_response_json["data"]["data"]
        ).decode("utf-8")
        assert ano_result.text == expected_text

    def test_non_plain_text_upload_uses_text_extractor(self, client):
        extractor = FakeTextExtractor()
        client.text_extractor = extractor
        payload = {
            "success": True,
            "data": {
                "data": base64.b64encode(b"docx bytes").decode("ascii"),
                "fileName": "report.docx",
                "uploadName": "report.docx",
            },
        }
        result = PseudoResult.model_validate(payload)

        ano_result = client.get_ano_result(result)

        assert ano_result.text == "extracted via fake extractor"
        assert extractor.called_with is result

    def test_pending_result_raises_value_error(
        self, client, accepted_pseudo_response_json
    ):
        result = PseudoResult.model_validate(accepted_pseudo_response_json)
        with pytest.raises(ValueError):
            client.get_ano_result(result)


# --- _raise_for_status --------------------------------------------------------


class TestRaiseForStatus:
    @pytest.mark.parametrize(
        "status_code,expected_exc",
        [(402, QuotaExceededError), (410, DataExpiredError), (429, RateLimitedError)],
    )
    def test_maps_documented_status_codes(self, client, status_code, expected_exc):
        response = httpx.Response(
            status_code, request=httpx.Request("GET", "https://x.test"), text="boom"
        )
        with pytest.raises(expected_exc):
            client._raise_for_status(response)

    def test_other_error_codes_raise_http_status_error_with_body_text(self, client):
        response = httpx.Response(
            400,
            request=httpx.Request("GET", "https://x.test"),
            text="fileName does not match",
        )
        with pytest.raises(httpx.HTTPStatusError, match="fileName does not match"):
            client._raise_for_status(response)

    def test_success_codes_do_not_raise(self, client):
        response = httpx.Response(200, request=httpx.Request("GET", "https://x.test"))
        client._raise_for_status(response)  # must not raise


# --- client lifecycle ----------------------------------------------------------


class TestClientLifecycle:
    def test_config_setter_rebuilds_http_client(self, client):
        old_http_client = client.client
        client.config = AnonymizationConfig(no_ssl_verify=True)
        assert client.client is not old_http_client
        assert old_http_client.is_closed

    def test_context_manager_closes_client(self):
        with GlanosClient(token="t", base_url="https://x.test") as c:
            assert c.client is not None
        assert c.client is None

    def test_close_is_idempotent(self, client):
        client.close()
        client.close()  # must not raise
        assert client.client is None
