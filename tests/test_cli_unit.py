"""Unit tests for glanos_anonymization.cli.

Pure argument-parsing and config-building logic - no network, no GlanosClient
construction with real HTTP.
"""

import argparse

import pytest

from glanos_anonymization import cli
from glanos_anonymization.core import AnonymizationConfig

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

    def test_missing_output_file_without_async_is_rejected(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            cli.parse_args(BASE_PSEUDO_ARGS + ["--file", "a.txt"])
        assert exc_info.value.code == 2
        assert "output filename" in capsys.readouterr().err

    def test_return_keys_with_async_is_rejected(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            cli.parse_args(
                BASE_PSEUDO_ARGS + ["--file", "a.txt", "--async", "--returnPseudoKey"]
            )
        assert exc_info.value.code == 2
        assert "Async mode" in capsys.readouterr().err

    def test_valid_minimal_args_parse(self):
        args = cli.parse_args(
            BASE_PSEUDO_ARGS + ["--file", "a.txt", "--outputFile", "o.txt"]
        )
        assert args.command == "pseudo"
        assert args.file == "a.txt"
        assert args.text is None
        assert args.a_sync is None

    def test_text_input_is_accepted(self):
        args = cli.parse_args(
            BASE_PSEUDO_ARGS + ["--text", "hello", "--outputFile", "o.txt"]
        )
        assert args.text == "hello"
        assert args.file is None

    def test_exclude_fields_rejects_values_outside_the_spec(self):
        with pytest.raises(SystemExit):
            cli.parse_args(
                BASE_PSEUDO_ARGS
                + [
                    "--file",
                    "a.txt",
                    "--outputFile",
                    "o.txt",
                    "--exclude_fields",
                    "zips",
                ]
            )

    def test_exclude_fields_accepts_spec_values(self):
        args = cli.parse_args(
            BASE_PSEUDO_ARGS
            + [
                "--file",
                "a.txt",
                "--outputFile",
                "o.txt",
                "--exclude_fields",
                "mails",
                "phones",
            ]
        )
        assert args.exclude_fields == ["mails", "phones"]


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
        """Regression: depseudo's --session must not feed into
        AnonymizationConfig.session (that's pseudo's own --session flag)."""
        args = cli.parse_args(BASE_DEPSEUDO_ARGS + ["--session", "s-1"])
        assert not hasattr(args, "session")


class TestParseArgsFetch:
    def test_minimal_valid_args(self):
        args = cli.parse_args(BASE_FETCH_ARGS)
        assert args.command == "fetch"
        assert args.fetch_key == "fk-1"
        assert args.start_retention_period is None
        assert args.download_options is None

    def test_no_start_retention_period_flag(self):
        args = cli.parse_args(BASE_FETCH_ARGS + ["--no-start-retention-period"])
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


class TestCreateAnoConfig:
    def test_pseudo_command_builds_config_with_settings_fields(self):
        args = cli.parse_args(
            BASE_PSEUDO_ARGS
            + [
                "--file",
                "a.txt",
                "--outputFile",
                "o.txt",
                "--mode",
                "XXX",
                "--exclude_fields",
                "mails",
            ]
        )
        config = cli.create_ano_config(args)
        assert isinstance(config, AnonymizationConfig)
        assert config.anonymization_mode.value == "XXX"
        assert config.exclude_fields == ["mails"]
        assert config.a_sync is False  # --async not passed

    def test_pseudo_command_async_flag_sets_a_sync_true(self):
        args = cli.parse_args(BASE_PSEUDO_ARGS + ["--file", "a.txt", "--async"])
        config = cli.create_ano_config(args)
        assert config.a_sync is True

    def test_fetch_command_does_not_crash_without_async_attribute(self):
        """Regression: the original cli.py unconditionally read args.a_sync,
        which doesn't exist on the fetch subparser's namespace and crashed
        with AttributeError."""
        args = cli.parse_args(BASE_FETCH_ARGS)
        config = cli.create_ano_config(args)
        assert config.a_sync is False

    def test_depseudo_command_does_not_crash_without_async_attribute(self):
        args = cli.parse_args(BASE_DEPSEUDO_ARGS + ["--metaKey", "mk"])
        config = cli.create_ano_config(args)
        assert config.a_sync is False

    def test_session_remove_command_builds_config_without_crashing(self):
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
        config = cli.create_ano_config(args)
        assert config.a_sync is False

    def test_unrelated_cli_args_are_not_forwarded_to_config(self):
        args = cli.parse_args(BASE_PSEUDO_ARGS + ["--file", "a.txt", "--async"])
        config = cli.create_ano_config(args)
        assert not hasattr(config, "file")
        assert not hasattr(config, "token")


class TestChoicesHelper:
    def test_returns_plain_string_values(self):
        from glanos_anonymization.core import BlankOutMode

        assert cli._choices(BlankOutMode) == ["XXX", "BLACK", "XLEN"]
