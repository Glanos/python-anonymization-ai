"""The package works without its optional dependencies.

Each test runs Python in a subprocess in which the optional packages cannot be
imported (their sys.modules entry is None), as if they were not installed.
"""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

SRC_DIR = Path(__file__).resolve().parent.parent / "src"

LANGCHAIN = ("langchain", "langchain_core", "langgraph")
MARKITDOWN = ("markitdown",)

# A client whose fake API answers every pseudo request with an "anonymized"
# document of the same type, carrying keys.
FAKE_API_CLIENT = """
import base64, json
import httpx
from glanos_anonymization import GlanosClient

def handler(request):
    file_name = json.loads(request.content)["fileName"]
    return httpx.Response(200, json={"success": True, "data": {
        "data": base64.b64encode(b"anonymized\\n").decode(),
        "fileName": file_name.replace(".", "_anonym.", 1),
        "uploadName": file_name,
        "pseudoKey": "pk",
        "encryptKey": "ek",
    }})

def fake_api_client():
    return GlanosClient(
        token="t", base_url="https://x.test", transport=httpx.MockTransport(handler)
    )
"""


def run_python(code: str, blocked: tuple[str, ...] = ()) -> subprocess.CompletedProcess:
    block = "".join(f"sys.modules[{name!r}] = None\n" for name in blocked)
    script = "import sys\n" + block + textwrap.dedent(code)
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=SRC_DIR,
    )


def assert_ok(result: subprocess.CompletedProcess) -> None:
    assert result.returncode == 0, result.stderr


class TestWithoutLangChain:
    def test_importing_the_package_does_not_import_langchain(self):
        result = run_python(
            """
            import glanos_anonymization
            assert not any(
                name.split(".")[0] in ("langchain", "langchain_core", "langgraph")
                for name in sys.modules
            ), sorted(sys.modules)
            """
        )
        assert_ok(result)

    def test_client_and_tool_definitions_work(self):
        result = run_python(
            """
            from glanos_anonymization import GlanosAnonymizationTools, GlanosClient

            GlanosClient(token="t", base_url="https://x.test").close()
            tools = GlanosAnonymizationTools(token="t", base_url="https://x.test")
            assert [name for name, _, _ in tools.definitions()][0] == "anonymize"
            tools.close()
            """,
            blocked=LANGCHAIN,
        )
        assert_ok(result)

    def test_middleware_raises_a_helpful_import_error(self):
        result = run_python(
            """
            try:
                from glanos_anonymization import GlanosAnonymizationMiddleware
            except ImportError as exc:
                print(exc)
            else:
                raise SystemExit("no ImportError")
            """,
            blocked=LANGCHAIN,
        )
        assert_ok(result)
        assert "pip install 'glanos-anonymization[langchain]'" in result.stdout

    def test_langchain_tools_raise_a_helpful_import_error(self):
        result = run_python(
            """
            from glanos_anonymization import GlanosAnonymizationTools

            tools = GlanosAnonymizationTools(token="t", base_url="https://x.test")
            try:
                tools.tools
            except ImportError as exc:
                print(exc)
            else:
                raise SystemExit("no ImportError")
            """,
            blocked=LANGCHAIN,
        )
        assert_ok(result)
        assert "pip install 'glanos-anonymization[langchain]'" in result.stdout

    def test_unknown_attribute_still_raises_attribute_error(self):
        result = run_python(
            """
            import glanos_anonymization
            try:
                glanos_anonymization.DoesNotExist
            except AttributeError:
                pass
            else:
                raise SystemExit("no AttributeError")
            """
        )
        assert_ok(result)


class TestWithoutMarkItDown:
    def test_importing_the_package_does_not_import_markitdown(self):
        result = run_python(
            """
            import glanos_anonymization
            from glanos_anonymization import GlanosClient

            GlanosClient(token="t", base_url="https://x.test").close()
            assert "markitdown" not in sys.modules
            """
        )
        assert_ok(result)

    def test_text_and_documents_are_anonymized(self):
        result = run_python(
            FAKE_API_CLIENT
            + """
with fake_api_client() as client:
    text_result = client.anonymize(text="Michael")
    assert text_result.extract_text().strip() == "anonymized"

    document = client.anonymize(data=b"docx bytes", filename="report.docx")
    assert document.data == b"anonymized\\n"
    assert document.file_name == "report_anonym.docx"
    assert document.pseudo_key == "pk"
""",
            blocked=MARKITDOWN,
        )
        assert_ok(result)

    def test_text_of_a_document_raises_a_helpful_import_error(self):
        result = run_python(
            FAKE_API_CLIENT
            + """
with fake_api_client() as client:
    document = client.anonymize(data=b"docx bytes", filename="report.docx")
    try:
        document.extract_text()
    except ImportError as exc:
        print(exc)
    else:
        raise SystemExit("no ImportError")
""",
            blocked=MARKITDOWN,
        )
        assert_ok(result)
        assert "pip install 'glanos-anonymization[text-extraction]'" in result.stdout

    def test_langchain_tool_reports_the_missing_extra_to_the_agent(self):
        result = run_python(
            FAKE_API_CLIENT
            + """
import base64
from glanos_anonymization import GlanosAnonymizationTools

tools = GlanosAnonymizationTools(token="t", base_url="https://x.test")
tools.client.close()
tools.client = fake_api_client()
anonymize = tools.tools[0]
print(anonymize.invoke({
    "file_base64": base64.b64encode(b"docx bytes").decode(),
    "file_name": "report.docx",
}))
""",
            blocked=MARKITDOWN,
        )
        assert_ok(result)
        assert "pip install 'glanos-anonymization[text-extraction]'" in result.stdout
        assert "report_anonym.docx" in result.stdout


class TestWithLangChain:
    def test_middleware_is_exported(self):
        from glanos_anonymization import GlanosAnonymizationMiddleware
        from glanos_anonymization.middleware import (
            GlanosAnonymizationMiddleware as Direct,
        )

        assert GlanosAnonymizationMiddleware is Direct
