"""When does a document in review expire, against the real Glanos API.

The document is sent with a retention of one minute. The test shows that
1. before the review, the document does not expire, even after that minute;
2. after the review, it expires one minute later, and what fetch() then gets.

Run it with a reviewer named on the command line, and -s to see the steps:

    GLANOS_REVIEW_ASSIGNEE=you@example.com uv run pytest tests/manual/test_review_expiry.py -s

GLANOS_REVIEW_GROUP=<group> sends it to a group instead. Approve the document
only when the test asks for it. It takes about 3 minutes plus the review.
"""

import os
import time
from pathlib import Path

import pytest

from glanos_anonymization import DataExpiredError, GlanosAPIError, GlanosClient

ASSIGNEE = os.environ.get("GLANOS_REVIEW_ASSIGNEE", "")
GROUP = os.environ.get("GLANOS_REVIEW_GROUP", "")
MAX_WAIT_MINUTES = int(os.environ.get("GLANOS_REVIEW_MAX_WAIT_MINUTES", "10"))
CHECK_EVERY_SECONDS = 10
RETENTION_MS = 60 * 1000
# longer than the retention, so that it has surely run out
PAST_RETENTION_SECONDS = 90

CONTRACT = Path(__file__).parent.parent / "data" / "example.docx"

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        not (ASSIGNEE or GROUP),
        reason="Set GLANOS_REVIEW_ASSIGNEE or GLANOS_REVIEW_GROUP to run the review "
        "expiry test",
    ),
]


def test_a_document_in_review_expires_only_after_the_review():
    reviewer = {"assignee": ASSIGNEE} if ASSIGNEE else {"assigned_group": GROUP}

    with GlanosClient(
        token=os.environ["GLANOS_TOKEN"],
        base_url=os.environ["GLANOS_BASE_URL"],
    ) as client:
        # 1. Send the document with a retention of one minute.
        pending = client.anonymize_with_review(
            path=CONTRACT,
            data_retention_ms=RETENTION_MS,
            meta_retention_ms=RETENTION_MS,
            **reviewer,
        )
        assert not pending.ready
        print(f"\n1. Sent {CONTRACT.name} for review to {reviewer}, retention 1 min.")
        print(f"   fetch_key: {pending.fetch_key}")
        print(
            f"   Do NOT approve it yet. Waiting {PAST_RETENTION_SECONDS} s, longer "
            f"than the retention..."
        )

        # 2. Before the review, the retention does not run: still pending.
        time.sleep(PAST_RETENTION_SECONDS)
        before_review = client.fetch(pending.fetch_key, max_attempts=1)
        assert not before_review.ready
        print("2. Still waiting for the review, not expired.")

        # 3. The review starts the retention.
        print(
            f"3. Approve the document in the Glanos dashboard now. Waiting up to "
            f"{MAX_WAIT_MINUTES} min, checking every {CHECK_EVERY_SECONDS} s..."
        )
        reviewed = client.fetch(
            pending.fetch_key,
            poll_interval=CHECK_EVERY_SECONDS,
            max_attempts=MAX_WAIT_MINUTES * 60 // CHECK_EVERY_SECONDS,
        )
        if not reviewed.ready:
            pytest.fail(f"Not reviewed within {MAX_WAIT_MINUTES} min.")
        print(f"   Reviewed document received: {reviewed.file_name}")

        # 4. One minute after the review, the document has expired.
        print(f"4. Waiting {PAST_RETENTION_SECONDS} s for the retention to run out...")
        time.sleep(PAST_RETENTION_SECONDS)
        try:
            after_expiry = client.fetch(pending.fetch_key, max_attempts=1)
        except DataExpiredError as exc:
            print(f"   Expired: HTTP {exc.status_code}, {exc.api_message!r}")
            return
        except GlanosAPIError as exc:
            pytest.fail(
                f"The expired document answered HTTP {exc.status_code} "
                f"({exc.api_message!r}) instead of 410"
            )

        if after_expiry.ready:
            pytest.fail("The document can still be fetched after its retention")
        pytest.fail(
            "The expired document is still reported as not ready (HTTP 202): "
            "fetch() without max_attempts would wait for it forever"
        )
