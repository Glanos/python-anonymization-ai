"""Unit tests for glanos_anonymization.cli.

Pure argument-parsing, settings-building and output-writing logic - no
network, no GlanosClient construction with real HTTP.
"""

import base64
import stat

import pytest

from glanos_anonymization import cli
from glanos_anonymization.core import (
    AnonymizationSettings,
    ExcludeField,
    PseudoResult,
)

pytestmark = pytest.mark.unit

BASE_PSEUDO_ARGS = ["pseudo", "--token", "tok", "--baseUrl", "https://x.test"]
BASE_FETCH_ARGS = [
    "fetch",
    "--token",
    "tok",
    "--baseUrl",
    "https://x.test",
    "--fetchKey",
    "fk-1",
    "--outputFile",
    "out.txt",
]
BASE_DEPSEUDO_ARGS = [
    "depseudo",
    "--token",
    "tok",
    "--baseUrl",
    "https://x.test",
    "--file",
    "anon.txt",
    "--outputFile",
    "out.txt",
]


class TestParseArgsPseudo:
    def test_requires_file_or_text(self):
        with pytest.raises(SystemExit):
            cli.parse_args(
                [
                    "pseudo",
                    "--token",
                    "t",
                    "--baseUrl",
                    "https://x.test",
                    "--outputFile",
                    "o",
                ]
            )

    def test_file_and_text_are_mutually_exclusive(self):
        with pytest.raises(SystemExit):
            cli.parse_args(
                BASE_PSEUDO_ARGS
                + ["--file", "a.txt", "--text", "hi", "--outputFile", "o"]
            )

    def test_missing_output_file_is_rejected_when_waiting(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            cli.parse_args(BASE_PSEUDO_ARGS + ["--file", "a.txt"])
        assert exc_info.value.code == 2
        assert "output filename" in capsys.readouterr().err

    @pytest.mark.parametrize(
        "only_submit",
        [
            ["--processing", "no_wait"],
            ["--assignee", "anna@example.com"],
            ["--assignedGroup", "legal"],
        ],
        ids=["no_wait", "assignee", "assignedGroup"],
    )
    def test_output_file_is_optional_when_only_submitting(self, only_submit):
        args = cli.parse_args(BASE_PSEUDO_ARGS + ["--file", "a.txt"] + only_submit)
        assert args.output_file is None

    @pytest.mark.parametrize(
        "only_submit",
        [["--processing", "no_wait"], ["--assignee", "anna@example.com"]],
        ids=["no_wait", "assignee"],
    )
    def test_return_keys_when_only_submitting_are_rejected(self, capsys, only_submit):
        with pytest.raises(SystemExit) as exc_info:
            cli.parse_args(
                BASE_PSEUDO_ARGS
                + ["--file", "a.txt", "--returnPseudoKey"]
                + only_submit
            )
        assert exc_info.value.code == 2
        assert "'fetch'" in capsys.readouterr().err

    def test_valid_minimal_args_parse(self):
        args = cli.parse_args(
            BASE_PSEUDO_ARGS + ["--file", "a.txt", "--outputFile", "o.txt"]
        )
        assert args.command == "pseudo"
        assert args.file == "a.txt"
        assert args.text is None
        assert args.processing is None
        assert args.poll_interval is None
        assert args.max_attempts is None

    def test_processing_rejects_unknown_values(self, capsys):
        with pytest.raises(SystemExit):
            cli.parse_args(
                BASE_PSEUDO_ARGS
                + ["--file", "a.txt", "--outputFile", "o", "--processing", "later"]
            )
        assert "invalid choice: 'later'" in capsys.readouterr().err

    def test_polling_flags_with_poll(self):
        args = cli.parse_args(
            BASE_PSEUDO_ARGS
            + ["--file", "a.txt", "--outputFile", "o", "--pollInterval", "0.5"]
            + ["--maxAttempts", "10"]
        )
        assert args.poll_interval == 0.5
        assert args.max_attempts == 10

    @pytest.mark.parametrize(
        "argv",
        [
            # --text defaults to sync
            ["--text", "hi", "--outputFile", "o", "--maxAttempts", "3"],
            ["--file", "a.txt", "--processing", "no_wait", "--pollInterval", "1"],
        ],
        ids=["text default", "no_wait"],
    )
    def test_polling_flags_without_poll_are_rejected(self, capsys, argv):
        with pytest.raises(SystemExit) as exc_info:
            cli.parse_args(BASE_PSEUDO_ARGS + argv)
        assert exc_info.value.code == 2
        assert "--processing poll" in capsys.readouterr().err

    def test_assignee_and_assigned_group_are_exclusive(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            cli.parse_args(
                BASE_PSEUDO_ARGS
                + ["--file", "a.txt", "--assignee", "a", "--assignedGroup", "g"]
            )
        assert exc_info.value.code == 2
        assert "not both" in capsys.readouterr().err

    @pytest.mark.parametrize(
        "flags",
        [["--processing", "poll"], ["--maxAttempts", "3"]],
        ids=["processing", "maxAttempts"],
    )
    def test_review_does_not_take_processing_flags(self, capsys, flags):
        with pytest.raises(SystemExit) as exc_info:
            cli.parse_args(
                BASE_PSEUDO_ARGS + ["--file", "a.txt", "--assignee", "a"] + flags
            )
        assert exc_info.value.code == 2
        assert "only submitted" in capsys.readouterr().err

    def test_text_input_is_accepted(self):
        args = cli.parse_args(
            BASE_PSEUDO_ARGS + ["--text", "hello", "--outputFile", "o.txt"]
        )
        assert args.text == "hello"
        assert args.file is None

    def test_exclude_fields_rejects_values_outside_the_spec(self, capsys):
        with pytest.raises(SystemExit):
            cli.parse_args(
                BASE_PSEUDO_ARGS
                + [
                    "--file",
                    "a.txt",
                    "--outputFile",
                    "o.txt",
                    "--excludeFields",
                    "zips",
                ]
            )
        assert "invalid choice: 'zips'" in capsys.readouterr().err

    def test_exclude_fields_accepts_spec_values(self):
        args = cli.parse_args(
            BASE_PSEUDO_ARGS
            + [
                "--file",
                "a.txt",
                "--outputFile",
                "o.txt",
                "--excludeFields",
                "mails",
                "phones",
            ]
        )
        assert args.exclude_fields == ["mails", "phones"]

    def test_return_key_switches_default_to_none(self):
        """None means "not given": the client then applies its own default
        (keys requested for sync requests)."""
        args = cli.parse_args(
            BASE_PSEUDO_ARGS + ["--file", "a.txt", "--outputFile", "o.txt"]
        )
        assert args.return_pseudo_key is None
        assert args.return_encrypt_key is None

    def test_return_key_switches_can_be_turned_off(self):
        args = cli.parse_args(
            BASE_PSEUDO_ARGS
            + [
                "--file",
                "a.txt",
                "--outputFile",
                "o.txt",
                "--noReturnPseudoKey",
                "--noReturnEncryptKey",
            ]
        )
        assert args.return_pseudo_key is False
        assert args.return_encrypt_key is False

    def test_flags_cannot_be_abbreviated(self):
        """allow_abbrev=False: a typo'd or shortened flag must fail loudly
        instead of silently matching a different option."""
        with pytest.raises(SystemExit):
            cli.parse_args(
                BASE_PSEUDO_ARGS
                + ["--file", "a.txt", "--outputFile", "o.txt", "--exclude", "mails"]
            )


class TestParseArgsDepseudo:
    def test_requires_at_least_one_credential(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            cli.parse_args(BASE_DEPSEUDO_ARGS)
        assert exc_info.value.code == 2
        assert "session" in capsys.readouterr().err.lower()

    def test_session_alone_is_sufficient(self):
        args = cli.parse_args(BASE_DEPSEUDO_ARGS + ["--session", "s-1"])
        assert args.depseudo_session == "s-1"

    def test_meta_key_alone_is_sufficient(self):
        args = cli.parse_args(BASE_DEPSEUDO_ARGS + ["--metaKey", "mk-1"])
        assert args.meta_key == "mk-1"

    def test_both_pseudo_and_encrypt_key_files_are_sufficient(self):
        args = cli.parse_args(
            BASE_DEPSEUDO_ARGS
            + ["--pseudoKeyFile", "p.pkey", "--encryptKeyFile", "e.ekey"]
        )
        assert args.pseudo_key_file == "p.pkey"

    def test_pseudo_key_file_alone_is_sufficient(self):
        """The (data, pseudoKey) combo isn't spelled out in the spec's prose,
        but works against the real API - the CLI shouldn't block it."""
        args = cli.parse_args(BASE_DEPSEUDO_ARGS + ["--pseudoKeyFile", "p.pkey"])
        assert args.pseudo_key_file == "p.pkey"

    def test_encrypt_key_file_alone_is_sufficient(self):
        args = cli.parse_args(BASE_DEPSEUDO_ARGS + ["--encryptKeyFile", "e.ekey"])
        assert args.encrypt_key_file == "e.ekey"

    def test_depseudo_session_dest_does_not_collide_with_pseudo_session(self):
        """Regression: depseudo's --session must not be read as pseudo's own
        --session flag (dest 'session')."""
        args = cli.parse_args(BASE_DEPSEUDO_ARGS + ["--session", "s-1"])
        assert not hasattr(args, "session")


class TestParseArgsFetch:
    def test_minimal_valid_args(self):
        args = cli.parse_args(BASE_FETCH_ARGS)
        assert args.command == "fetch"
        assert args.fetch_key == "fk-1"
        assert args.start_retention_period is None
        assert args.download_options is None
        # None: the client's defaults apply (every 300 s, no limit)
        assert args.poll_interval is None
        assert args.max_attempts is None

    def test_polling_flags(self):
        args = cli.parse_args(
            BASE_FETCH_ARGS + ["--pollInterval", "60", "--maxAttempts", "1"]
        )
        assert args.poll_interval == 60.0
        assert args.max_attempts == 1

    def test_no_start_retention_period_flag(self):
        args = cli.parse_args(BASE_FETCH_ARGS + ["--noStartRetentionPeriod"])
        assert args.start_retention_period is False

    def test_download_options_choice(self):
        args = cli.parse_args(BASE_FETCH_ARGS + ["--downloadOptions", "PDF_AS_PDF"])
        assert args.download_options == "PDF_AS_PDF"


class TestParseArgsSessionRemove:
    def test_session_remove_requires_session(self):
        with pytest.raises(SystemExit):
            cli.parse_args(
                ["session-remove", "--token", "t", "--baseUrl", "https://x.test"]
            )

    def test_session_remove_minimal_valid_args(self):
        args = cli.parse_args(
            [
                "session-remove",
                "--token",
                "t",
                "--baseUrl",
                "https://x.test",
                "--session",
                "s1",
            ]
        )
        assert args.command == "session-remove"
        assert args.session_id == "s1"


class TestBuildInlineSettings:
    def test_settings_flags_become_inline_settings(self):
        args = cli.parse_args(
            BASE_PSEUDO_ARGS
            + [
                "--file",
                "a.txt",
                "--outputFile",
                "o.txt",
                "--mode",
                "XXX",
                "--excludeFields",
                "mails",
            ]
        )
        settings = cli.build_inline_settings(args)
        assert isinstance(settings, AnonymizationSettings)
        assert settings.mode == "XXX"
        assert settings.excludeFields == [ExcludeField.MAILS]

    def test_without_settings_flags_no_settings_are_sent(self):
        """Without settings flags the installation's default setting must
        apply, so the CLI must not send any inline settings."""
        args = cli.parse_args(
            BASE_PSEUDO_ARGS + ["--file", "a.txt", "--processing", "no_wait"]
        )
        assert cli.build_inline_settings(args) is None

    def test_only_given_flags_are_included(self):
        args = cli.parse_args(
            BASE_PSEUDO_ARGS
            + ["--file", "a.txt", "--processing", "no_wait", "--foldMode", "STRICT"]
            + ["--token", "t2"]
        )
        assert cli.given_settings_fields(args) == {"foldMode": "STRICT"}

    @pytest.mark.parametrize(
        "argv",
        [
            BASE_FETCH_ARGS,
            BASE_DEPSEUDO_ARGS + ["--metaKey", "mk"],
            [
                "session-remove",
                "--token",
                "t",
                "--baseUrl",
                "https://x.test",
                "--session",
                "s1",
            ],
        ],
    )
    def test_commands_without_settings_flags_do_not_crash(self, argv):
        """Regression: commands whose parser has no settings flags must not
        crash with AttributeError when settings are collected."""
        assert cli.given_settings_fields(cli.parse_args(argv)) == {}


class FakeSettingsClient:
    def __init__(self, stored: AnonymizationSettings):
        self.stored = stored
        self.updated: AnonymizationSettings | None = None

    def get_setting(self, setting_id: str) -> AnonymizationSettings:
        assert setting_id == self.stored.id
        return self.stored

    def update_setting(self, settings: AnonymizationSettings) -> AnonymizationSettings:
        self.updated = settings
        return settings


SETTING_UPDATE_ARGS = [
    "setting-update",
    "--token",
    "t",
    "--baseUrl",
    "https://x.test",
    "--setting",
    "set-1",
]


class TestUpdateSetting:
    """The API replaces a setting as a whole, so setting-update has to merge
    the given flags into the stored setting and send it back complete."""

    @pytest.fixture
    def fake_client(self):
        return FakeSettingsClient(
            AnonymizationSettings(
                id="set-1",
                name="Legal DE",
                mode="TYPE_PRESERVING",
                excludeFields=["dates"],
                foldMode="STRICT",
            )
        )

    def test_keeps_fields_that_were_not_given(self, fake_client):
        args = cli.parse_args(SETTING_UPDATE_ARGS + ["--excludeFields", "locations"])

        cli.update_setting(fake_client, args)

        updated = fake_client.updated
        assert updated.id == "set-1"
        assert updated.name == "Legal DE"
        assert updated.mode == "TYPE_PRESERVING"
        assert updated.foldMode == "STRICT"
        assert updated.excludeFields == ["locations"]

    def test_name_flag_renames_the_setting(self, fake_client):
        args = cli.parse_args(SETTING_UPDATE_ARGS + ["--name", "Legal EN"])

        cli.update_setting(fake_client, args)

        assert fake_client.updated.name == "Legal EN"
        assert fake_client.updated.excludeFields == ["dates"]


class TestChoicesHelper:
    def test_returns_plain_string_values(self):
        from glanos_anonymization.core import BlankOutMode

        assert cli._choices(BlankOutMode) == ["XXX", "BLACK", "XLEN"]


class TestConnectionFromEnvironment:
    @pytest.fixture(autouse=True)
    def clean_environment(self, monkeypatch):
        # pytest-dotenv may have loaded real credentials from .env
        monkeypatch.delenv("GLANOS_TOKEN", raising=False)
        monkeypatch.delenv("GLANOS_BASE_URL", raising=False)

    @pytest.mark.parametrize(
        "command",
        [
            ["pseudo", "--text", "hi", "--outputFile", "o.txt"],
            ["fetch", "--fetchKey", "fk", "--outputFile", "o.txt"],
            ["depseudo", "--file", "a.txt", "--outputFile", "o.txt", "--metaKey", "m"],
            ["session-get", "--session", "s1"],
            ["setting-list"],
        ],
    )
    def test_token_and_base_url_come_from_the_environment(self, monkeypatch, command):
        monkeypatch.setenv("GLANOS_TOKEN", "env-token")
        monkeypatch.setenv("GLANOS_BASE_URL", "https://env.test")

        args = cli.parse_args(command)

        assert args.token == "env-token"
        assert args.base_url == "https://env.test"

    def test_flags_take_precedence_over_the_environment(self, monkeypatch):
        monkeypatch.setenv("GLANOS_TOKEN", "env-token")
        monkeypatch.setenv("GLANOS_BASE_URL", "https://env.test")

        args = cli.parse_args(
            ["setting-list", "--token", "flag-token", "--baseUrl", "https://flag.test"]
        )

        assert args.token == "flag-token"
        assert args.base_url == "https://flag.test"

    def test_token_is_required_without_the_environment(self, capsys):
        with pytest.raises(SystemExit):
            cli.parse_args(["setting-list", "--baseUrl", "https://x.test"])
        assert "--token" in capsys.readouterr().err

    def test_empty_variable_counts_as_missing(self, monkeypatch):
        monkeypatch.setenv("GLANOS_TOKEN", "")

        with pytest.raises(SystemExit):
            cli.parse_args(["setting-list", "--baseUrl", "https://x.test"])

    def test_help_does_not_show_the_token(self, monkeypatch, capsys):
        monkeypatch.setenv("GLANOS_TOKEN", "secret-env-token")

        with pytest.raises(SystemExit):
            cli.parse_args(["setting-list", "--help"])

        out = capsys.readouterr().out
        assert "$GLANOS_TOKEN" in out
        assert "secret-env-token" not in out


# --- write_to_output / manage_data (pure I/O, no network) -------------------


def completed_result(**data_fields) -> PseudoResult:
    return PseudoResult.model_validate(
        {
            "success": True,
            "data": {
                "data": base64.b64encode(b"content").decode("ascii"),
                "fileName": "doc.docx",
                "uploadName": "doc.docx",
                **data_fields,
            },
        }
    )


class TestWriteToOutput:
    def test_writes_bytes(self, tmp_path):
        out = tmp_path / "result.bin"
        cli.write_to_output(b"hello", output_file=out, mode="wb")
        assert out.read_bytes() == b"hello"

    def test_extension_override_replaces_suffix(self, tmp_path):
        out = tmp_path / "result.bin"
        cli.write_to_output("text", output_file=out, extension="txt")
        assert (tmp_path / "result.txt").read_text() == "text"
        assert not out.exists()


class TestWriteKeyFile:
    def test_is_readable_by_its_owner_only(self, tmp_path):
        path = cli.write_key_file(tmp_path / "out.docx", "pkey", "secret")

        assert path == tmp_path / "out.pkey"
        assert path.read_text() == "secret"
        assert stat.S_IMODE(path.stat().st_mode) == 0o600

    def test_restricts_an_existing_file(self, tmp_path):
        existing = tmp_path / "out.pkey"
        existing.write_text("old key, longer than the new one")
        existing.chmod(0o644)

        cli.write_key_file(tmp_path / "out.docx", "pkey", "new")

        assert existing.read_text() == "new"
        assert stat.S_IMODE(existing.stat().st_mode) == 0o600


class TestManageData:
    def test_pending_result_prints_fetch_key_and_writes_nothing(
        self, tmp_path, capsys, accepted_pseudo_response_json
    ):
        """The fetchKey JSON goes to stdout (so it can be piped into a script),
        the human-readable notice to stderr."""
        result = PseudoResult.model_validate(accepted_pseudo_response_json)

        cli.manage_data(result, output_file=tmp_path / "out.txt")

        assert not (tmp_path / "out.txt").exists()
        captured = capsys.readouterr()
        assert "fetchKey" in captured.out
        assert "not processed yet" in captured.err
        assert "fetch --fetchKey fetch-abc-123" in captured.err

    def test_completed_result_writes_file_and_prints_meta_key(
        self, tmp_path, capsys, completed_pseudo_response_json
    ):
        result = PseudoResult.model_validate(completed_pseudo_response_json)
        out = tmp_path / "out.txt"

        cli.manage_data(result, output_file=out)

        expected_bytes = base64.b64decode(
            completed_pseudo_response_json["data"]["data"]
        )
        assert out.read_bytes() == expected_bytes
        captured = capsys.readouterr()
        assert "metaKey: meta-123" in captured.out
        assert not (tmp_path / "out.pkey").exists()
        # No session was used for this upload, so nothing should be printed.
        assert "session:" not in captured.out

    def test_prints_the_file_names_of_the_server_and_the_output(self, tmp_path, capsys):
        result = completed_result(fileName="s_deanonym.txt")

        cli.manage_data(result, output_file=tmp_path / "s_restored.txt")

        printed = capsys.readouterr().out
        assert "fileName (in server): s_deanonym.txt\n" in printed
        assert "fileName (in output): s_restored.txt\n" in printed

    def test_prints_session_when_present(self, tmp_path, capsys):
        result = completed_result(session="session-abc")

        cli.manage_data(result, output_file=tmp_path / "out.docx")

        assert "session: session-abc" in capsys.readouterr().out

    def test_requested_key_not_returned_by_api_does_not_crash(self, tmp_path, capsys):
        """Regression: the real API can omit pseudoKey/encryptKey even when
        return_pseudo_key/return_encrypt_key were requested. manage_data must
        not crash trying to write a None value to disk."""
        result = completed_result(pseudoKey="pk-value", encryptKey=None)
        out = tmp_path / "out.docx"

        cli.manage_data(result, output_file=out)  # must not raise

        assert (tmp_path / "out.pkey").exists()
        assert not (tmp_path / "out.ekey").exists()
        assert "encryptKey" not in capsys.readouterr().out

    def test_keys_are_written_to_files_and_never_printed(self, tmp_path, capsys):
        result = completed_result(pseudoKey="pk-value", encryptKey="ek-value")
        out = tmp_path / "out.docx"

        cli.manage_data(result, output_file=out)

        assert (tmp_path / "out.pkey").read_text() == "pk-value"
        assert (tmp_path / "out.ekey").read_text() == "ek-value"
        printed = capsys.readouterr().out
        assert "pk-value" not in printed
        assert "ek-value" not in printed
        assert f"pseudoKey written to {tmp_path / 'out.pkey'}" in printed
        assert f"encryptKey written to {tmp_path / 'out.ekey'}" in printed
