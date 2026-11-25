"""Which error does an expired document give, against the real Glanos API.

Anonymizes a document with a retention of one minute, without review, and
downloads it: the retention starts with the first download, not with the
processing (an earlier run of this test showed a document that was never
downloaded still there after 90 seconds). Then it fetches it again after 90
seconds. The API spec answers expired data with HTTP 410. Run it with -s to
see the steps:

    uv run pytest tests/manual/test_retention_expiry.py -s
"""

import os
import time
from pathlib import Path

import pytest

from glanos_anonymization import DataExpiredError, GlanosAPIError, GlanosClient

RETENTION_MS = 60 * 1000
# longer than the retention, so that it has surely run out
PAST_RETENTION_SECONDS = 90

CONTRACT = Path(__file__).parent.parent / "data" / "example.docx"

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        not (os.environ.get("GLANOS_TOKEN") and os.environ.get("GLANOS_BASE_URL")),
        reason="GLANOS_TOKEN or GLANOS_BASE_URL not set",
    ),
]


def test_an_expired_document_gives_http_410():
    with GlanosClient(
        token=os.environ["GLANOS_TOKEN"],
        base_url=os.environ["GLANOS_BASE_URL"],
    ) as client:
        # 1. Submit only.
        pending = client.anonymize(
            path=CONTRACT,
            processing="no_wait",
            data_retention_ms=RETENTION_MS,
            meta_retention_ms=RETENTION_MS,
        )
        print(f"\n1. Sent {CONTRACT.name}, retention 1 min.")
        print(f"   fetch_key: {pending.fetch_key}")

        # 2. Download it; this starts the retention.
        downloaded = client.fetch(pending.fetch_key, poll_interval=5, max_attempts=24)
        assert downloaded.ready, "not processed within 2 minutes"
        print(
            f"2. Downloaded {downloaded.file_name}. Waiting {PAST_RETENTION_SECONDS} s..."
        )

        # 3. Let the retention run out, then fetch it again.
        time.sleep(PAST_RETENTION_SECONDS)
        try:
            result = client.fetch(pending.fetch_key, max_attempts=1)
        except DataExpiredError as exc:
            print(f"3. Expired: HTTP {exc.status_code}, {exc.api_message!r}")
            return
        except GlanosAPIError as exc:
            pytest.fail(
                f"The expired document answered HTTP {exc.status_code} "
                f"({exc.api_message!r}) instead of 410"
            )

        if result.ready:
            pytest.fail("Still downloadable 90 s after the download")
        pytest.fail("Still reported as not ready (HTTP 202) after 90 s")
