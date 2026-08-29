"""End-to-end tests for glanos_anonymization.cli against the real Glanos API.

Reads GLANOS_TOKEN and GLANOS_BASE_URL from the environment (loaded from the
repo's .env file by pytest-dotenv). The tests that hit the real API are
skipped automatically if either is missing. The --help smoke test always
runs, since it needs no credentials.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from glanos_anonymization import cli

GLANOS_API_KEY = os.environ.get("GLANOS_TOKEN", "")
GLANOS_BASE_URL = os.environ.get("GLANOS_BASE_URL", "")

requires_credentials = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        not GLANOS_API_KEY, reason="GLANOS_API_KEY not set - skipping e2e tests"
    ),
    pytest.mark.skipif(
        not GLANOS_BASE_URL, reason="GLANOS_BASE_URL not set - skipping e2e tests"
    ),
]


@pytest.mark.e2e
def test_help_runs_as_a_real_subprocess():
    """Smoke test: the CLI is actually invocable as a script, independent of
    any real credentials."""
    src_dir = Path(__file__).resolve().parent.parent / "src"
    result = subprocess.run(
        [sys.executable, "-m", "glanos_anonymization.cli", "-h"],
        capture_output=True,
        text=True,
        cwd=src_dir,
    )
    assert result.returncode == 0
    assert "pseudo" in result.stdout


class TestPseudoDepseudoRoundTripViaCli:
    pytestmark = requires_credentials

    def test_sync_pseudo_writes_file_and_prints_keys(self, tmp_path, capsys):
        out = tmp_path / "anonymized.txt"

        cli.main(
            [
                "pseudo",
                "--token",
                GLANOS_API_KEY,
                "--baseUrl",
                GLANOS_BASE_URL,
                "--text",
                "Mein Name ist Michael Huber",
                "--outputFile",
                str(out),
                "--returnPseudoKey",
                "--returnEncryptKey",
            ]
        )

        assert out.exists()
        captured = capsys.readouterr()
        assert "metaKey:" in captured.out
        assert "pseudoKey:" in captured.out
        assert "encryptKey:" in captured.out
        assert (tmp_path / "anonymized.pkey").exists()
        # Note: the real API does not always return encryptKey even when
        # --returnEncryptKey is passed, so anonymized.ekey is not guaranteed
        # to exist - not asserted here.

    def test_async_pseudo_then_fetch(self, tmp_path, capsys):
        cli.main(
            [
                "pseudo",
                "--token",
                GLANOS_API_KEY,
                "--baseUrl",
                GLANOS_BASE_URL,
                "--text",
                "Mein Name ist Michael Huber",
                "--async",
            ]
        )
        submitted = json.loads(capsys.readouterr().out)
        fetch_key = submitted["fetchKey"]

        fetch_out = tmp_path / "fetched.txt"
        cli.main(
            [
                "fetch",
                "--token",
                GLANOS_API_KEY,
                "--baseUrl",
                GLANOS_BASE_URL,
                "--fetchKey",
                fetch_key,
                "--outputFile",
                str(fetch_out),
            ]
        )

        assert fetch_out.exists()

    def test_depseudo_with_meta_key_via_fetch(self, tmp_path, capsys):
        # pseudoKeyRetentionMS is only honored for the async submit -> fetch
        # workflow, and must be set at submission time (not at fetch time).
        # A *sync* pseudo call discards its pseudoKey server-side immediately
        # regardless of pseudoKeyRetentionMS - see the diagnostics in
        # test_core_e2e.py. metaRetentionMS must ALSO be set explicitly -
        # without it, depseudo fails with "No such id ..." even with
        # pseudoKeyRetentionMS set.
        cli.main(
            [
                "pseudo",
                "--token",
                GLANOS_API_KEY,
                "--baseUrl",
                GLANOS_BASE_URL,
                "--text",
                "Mein Name ist Michael Huber",
                "--async",
                "--pseudoKeyRetentionMS",
                "60000",
                "--metaRetentionMS",
                "60000",
            ]
        )
        fetch_key = json.loads(capsys.readouterr().out)["fetchKey"]

        anonymized = tmp_path / "anonymized.txt"
        cli.main(
            [
                "fetch",
                "--token",
                GLANOS_API_KEY,
                "--baseUrl",
                GLANOS_BASE_URL,
                "--fetchKey",
                fetch_key,
                "--outputFile",
                str(anonymized),
            ]
        )
        printed_lines = capsys.readouterr().out.splitlines()
        meta_key = (
            next(l for l in printed_lines if l.startswith("metaKey:"))
            .split(":", 1)[1]
            .strip()
        )
        # The API tracks the document by the fileName it originally assigned,
        # not the local --outputFile name, so it must be echoed back exactly.
        server_file_name = (
            next(l for l in printed_lines if l.startswith("fileName:"))
            .split(":", 1)[1]
            .strip()
        )

        depseudo_out = tmp_path / "restored.txt"
        cli.main(
            [
                "depseudo",
                "--token",
                GLANOS_API_KEY,
                "--baseUrl",
                GLANOS_BASE_URL,
                "--file",
                str(anonymized),
                "--metaKey",
                meta_key,
                "--filename",
                server_file_name,
                "--outputFile",
                str(depseudo_out),
            ]
        )

        assert depseudo_out.exists()


class TestSessionRemoveViaCli:
    pytestmark = requires_credentials

    def test_pseudo_with_session_then_remove_session(self, tmp_path, capsys):
        # A session is created via --sessionMaxAge with --session left unset
        # - the server generates the id, printed as "session: <value>".
        cli.main(
            [
                "pseudo",
                "--token",
                GLANOS_API_KEY,
                "--baseUrl",
                GLANOS_BASE_URL,
                "--text",
                "Mein Name ist Michael Huber",
                "--sessionMaxAge",
                "60000",
                "--outputFile",
                str(tmp_path / "anonymized.txt"),
            ]
        )
        printed_lines = capsys.readouterr().out.splitlines()
        session_id = (
            next(l for l in printed_lines if l.startswith("session:"))
            .split(":", 1)[1]
            .strip()
        )

        cli.main(
            [
                "session-remove",
                "--token",
                GLANOS_API_KEY,
                "--baseUrl",
                GLANOS_BASE_URL,
                "--session",
                session_id,
            ]
        )
        removed = json.loads(capsys.readouterr().out)
        assert removed["success"] is True
