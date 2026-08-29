import base64

import pytest

COMPLETED_TEXT = "Anonymisiertes Dokument"


def b64(text: str) -> str:
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


def make_completed_response_json(
    *,
    text: str = COMPLETED_TEXT,
    file_name: str = "plain_text.txt",
    upload_name: str | None = "plain_text.txt",
    meta_key: str | None = "meta-123",
    pseudo_key: str | None = None,
    encrypt_key: str | None = None,
) -> dict:
    """Build a JSON dict matching ApiPseudoResponseMessage for a completed (HTTP 200) result."""
    return {
        "success": True,
        "message": None,
        "data": {
            "data": b64(text),
            "fileName": file_name,
            "uploadName": upload_name,
            "metaKey": meta_key,
            "pseudoKey": pseudo_key,
            "encryptKey": encrypt_key,
            "language": "de",
            "success": True,
            "failReason": None,
            "session": None,
        },
    }


def make_accepted_response_json(fetch_key: str = "fetch-abc-123") -> dict:
    """Build a JSON dict matching ApiPseudoAcceptedResponseMessage (HTTP 202)."""
    return {"success": True, "message": None, "data": {"fetchKey": fetch_key}}


@pytest.fixture
def completed_pseudo_response_json() -> dict:
    return make_completed_response_json()


@pytest.fixture
def accepted_pseudo_response_json() -> dict:
    return make_accepted_response_json()


@pytest.fixture
def pseudo_key_file(tmp_path):
    path = tmp_path / "secret.pkey"
    path.write_text("pseudo-key-value")
    return path


@pytest.fixture
def encrypt_key_file(tmp_path):
    path = tmp_path / "secret.ekey"
    path.write_text("encrypt-key-value")
    return path
