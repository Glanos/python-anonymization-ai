from glanos_anonymization import GlanosClient
import os
from pathlib import Path
import pytest

@pytest.fixture
def example_docx_file():
    return Path(__file__).parent / "data" / "example.docx"


def test_client_functions_docx(example_docx_file):
    client = GlanosClient(token=os.environ["GLANOS_TOKEN"], base_url=os.environ["GLANOS_BASE_URL"])
    text = "Mein Name ist Michael Huber"
    text_ano_result = client.anonymize(text=text)
    print(text_ano_result.text)

    file_ano_result = client.anonymize(path=example_docx_file)
    print(file_ano_result.text)