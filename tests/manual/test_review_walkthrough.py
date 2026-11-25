"""A user's walk through human review, against the real Glanos API.

It sends tests/data/example.docx to a reviewer and waits until you approve it
in the Glanos anonymization.ai dashboard. Name the reviewer on the command line
(not in .env, so that normal e2e runs skip it), and use -s to see the steps:

    GLANOS_REVIEW_ASSIGNEE=you@example.com uv run pytest tests/manual/test_review_walkthrough.py -s

GLANOS_REVIEW_GROUP=<group> sends it to a group instead. The test waits up to
GLANOS_REVIEW_MAX_WAIT_MINUTES (default 10) for the review.
"""

import os
from pathlib import Path

import pytest

from glanos_anonymization import GlanosClient

ASSIGNEE = os.environ.get("GLANOS_REVIEW_ASSIGNEE", "")
GROUP = os.environ.get("GLANOS_REVIEW_GROUP", "")
MAX_WAIT_MINUTES = int(os.environ.get("GLANOS_REVIEW_MAX_WAIT_MINUTES", "10"))
CHECK_EVERY_SECONDS = 10
ONE_DAY_MS = 24 * 60 * 60 * 1000

CONTRACT = Path(__file__).parent.parent / "data" / "example.docx"

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        not (ASSIGNEE or GROUP),
        reason="Set GLANOS_REVIEW_ASSIGNEE or GLANOS_REVIEW_GROUP to run the review "
        "walkthrough",
    ),
]


def test_user_sends_a_document_for_review_and_collects_it(tmp_path):
    reviewer = {"assignee": ASSIGNEE} if ASSIGNEE else {"assigned_group": GROUP}

    with GlanosClient(
        token=os.environ["GLANOS_TOKEN"],
        base_url=os.environ["GLANOS_BASE_URL"],
    ) as client:
        # 1. Send the document for review. The data must outlive the review,
        #    otherwise it disappears from the reviewer's dashboard.
        pending = client.anonymize_with_review(
            path=CONTRACT,
            data_retention_ms=ONE_DAY_MS,
            meta_retention_ms=ONE_DAY_MS,
            **reviewer,
        )
        assert not pending.ready
        assert pending.fetch_key
        print(f"\n1. Sent {CONTRACT.name} for review to {reviewer}.")
        print(f"   fetch_key: {pending.fetch_key}")

        # 2. Right away, the review cannot be done yet.
        first_check = client.fetch(pending.fetch_key, max_attempts=1)
        assert not first_check.ready, "the document was released without a review"
        print("2. Not ready yet, as expected.")

        # 3. Wait for the reviewer.
        print(
            f"3. Approve the document in the Glanos dashboard now. Waiting up to "
            f"{MAX_WAIT_MINUTES} min, checking every {CHECK_EVERY_SECONDS} s..."
        )
        result = client.fetch(
            pending.fetch_key,
            poll_interval=CHECK_EVERY_SECONDS,
            max_attempts=MAX_WAIT_MINUTES * 60 // CHECK_EVERY_SECONDS,
        )
        if not result.ready:
            pytest.fail(
                f"Not reviewed within {MAX_WAIT_MINUTES} min. Collect it later with "
                f"client.fetch({pending.fetch_key!r})."
            )

        # 4. Use the reviewed document.
        reviewed = tmp_path / result.file_name
        reviewed.write_bytes(result.data)
        print(f"4. Reviewed document received: {result.file_name}")
        print(f"   {result.extract_text()[:200]!r}")
        assert result.data
        assert result.pseudo_key

        # 5. Restore the original from the reviewed document.
        restored = client.deanonymize(result)
        print(f"5. Restored: {restored.file_name}")
        assert restored.data
