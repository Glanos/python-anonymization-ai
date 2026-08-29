# Glanos Anonymization Client

A Python client library for the Glanos anonymization API — anonymize (pseudonymize) text and documents, and reliably restore them later. Includes a command-line tool and a LangChain agent middleware for anonymizing user input before it reaches an LLM and restoring the original values in the model's replies.

## Installation

Using `uv`:

```bash
uv add glanos-anonymization
```

Using `pip`:

```bash
pip install glanos-anonymization
```

## Quick Start

```python
from glanos_anonymization import GlanosClient, AnonymizationConfig

client = GlanosClient(
    token="YOUR_GLANOS_TOKEN",
    base_url="https://your-glanos-instance.com",
    config=AnonymizationConfig(a_sync=False),
)

result = client.anonymize(
    text="""
    My name is Max Mustermann.
    I live in Munich.
    My phone number is 01711234567.
    """
)

print(result.text)
```

Example output:

```text
My name is PERSON1.
I live in LOCATION1.
My phone number is PHONE1.
```

The full API response (including the `metaKey` needed to restore the text later) is available via `result.pseudo_result`.

## Anonymizing Files

```python
result = client.anonymize(path="./documents/example.docx")
print(result.text)
```

The client automatically waits for asynchronous processing to finish and extracts text from supported document types (`.docx`, `.pdf`, `.pptx`, `.xlsx`, and more).

## Restoring Anonymized Text (Depseudonymization)

Anonymization is reversible. The simplest way to restore text is with a **session**: create one, anonymize with it, and later depseudonymize with the same session id — no file or key management required.

```python
with GlanosClient(token=token, base_url=base_url, config=AnonymizationConfig(a_sync=False)) as client:
    # Mint a session - sessionMaxAge is set, session is left unset so the
    # server generates and returns a new id.
    session = client.pseudo_sync(text=".", session_max_age=24 * 60 * 60 * 1000).data.session

    pseudo_result = client.pseudo_sync(text="Michael was in Munich.", session=session)
    anonymized_text = client.get_ano_result(pseudo_result).text
    print(anonymized_text)  # "PERSON1 was in LOCATION1."

    restored_text = client.depseudo_text(anonymized_text, session=session)
    print(restored_text)  # "Michael was in Munich."

    client.remove_session(session)
```

Documents can also be restored via `client.depseudo(path=...)`, using either the `metaKey` from the original response or a `pseudoKey` file (see `GlanosClient.depseudo`'s docstring for the retention requirements of each approach).

## Configuration

`AnonymizationConfig` controls how text is anonymized and how long the API retains anonymized data and keys:

```python
config = AnonymizationConfig(
    anonymization_mode="TYPE_PRESERVING",  # TYPE, TYPE_PRESERVING, XXX, MT
    exclude_fields=["locations"],          # entity types to leave untouched
    a_sync=False,                          # wait for the result instead of polling
    return_pseudo_key=True,                # get the pseudoKey back (a_sync must be False)
    data_retention_ms=0,
    pseudo_key_retention_ms=0,
    meta_retention_ms=0,
)
```

See `AnonymizationConfig` for the full list of fields (workflow, fold mode, image handling, custom options, ...).

## Error Handling

`GlanosClient` raises typed exceptions for the API's documented error responses, all deriving from `GlanosClientError`:

```python
from glanos_anonymization import QuotaExceededError, RateLimitedError, DataExpiredError, GlanosClientError

try:
    client.pseudo_sync(text="...")
except QuotaExceededError:
    ...  # license expired or quota exceeded (HTTP 402)
except RateLimitedError:
    ...  # too many requests (HTTP 429)
except DataExpiredError:
    ...  # data or session expired (HTTP 410)
except GlanosClientError:
    ...  # any other client/API error
```

## Command-Line Interface

Installing the package also installs a `glanos-anonymization` command:

```bash
glanos-anonymization pseudo --token TOKEN --baseUrl URL --file document.docx --outputFile anonymized.docx
```

```bash
glanos-anonymization pseudo --token TOKEN --baseUrl URL --text "Michael was in Munich." --outputFile out.txt
```

```bash
glanos-anonymization fetch --token TOKEN --baseUrl URL --fetchKey FETCH_KEY --outputFile out.txt
```

```bash
glanos-anonymization depseudo --token TOKEN --baseUrl URL --file anonymized.txt --metaKey META_KEY --outputFile restored.txt
```

```bash
glanos-anonymization session-remove --token TOKEN --baseUrl URL --session SESSION_ID
```

Run `glanos-anonymization <command> --help` for the full set of options per command (anonymization mode, excluded entity types, retention windows, sessions, and more).

## LangChain Middleware

`GlanosAnonymizationMiddleware` anonymizes user messages before they reach the model and restores the model's replies afterward, using one Glanos session per conversation so the same entity always maps to the same pseudonym across every turn - e.g. once "Michael" becomes "PERSON1", it stays "PERSON1" for the rest of the conversation, and any reply mentioning "PERSON1" is turned back into "Michael" before you see it.

Keeping that mapping consistent across separate `invoke()` calls requires a LangGraph checkpointer and a stable `thread_id`:

```python
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

from glanos_anonymization import GlanosAnonymizationMiddleware

middleware = GlanosAnonymizationMiddleware(
    token="YOUR_GLANOS_TOKEN",
    base_url="https://your-glanos-instance.com",
)

agent = create_agent(
    model=ChatOpenAI(model="gpt-4.1"),
    middleware=[middleware],
    checkpointer=MemorySaver(),  # use a persistent checkpointer in production
)

thread = {"configurable": {"thread_id": "conversation-123"}}

response = agent.invoke(
    {"messages": [{"role": "user", "content": "My name is Max Mustermann. Summarize this."}]},
    thread,
)
print(response["messages"][-1].content)

# Later, in the same conversation - "Max Mustermann" maps to the same
# pseudonym as above, and the reply is restored automatically.
response = agent.invoke(
    {"messages": [{"role": "user", "content": "What was my name again?"}]},
    thread,
)
print(response["messages"][-1].content)

middleware.close()
```

Without a checkpointer, a fresh session is created on every `invoke()` call, and consistency only holds within that single call.

## LangChain Tools

For agents that should decide themselves when to anonymize or restore text (rather than having every message anonymized automatically, as with the middleware above), `GlanosAnonymizationTools` exposes `anonymize_text` and `deanonymize_text` as LangChain tools. Both share one Glanos session, so a value anonymized by one call is restored correctly by a later one.

```python
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from glanos_anonymization import GlanosAnonymizationTools

glanos_tools = GlanosAnonymizationTools(token="YOUR_GLANOS_TOKEN", base_url="https://your-glanos-instance.com")

agent = create_agent(model=ChatOpenAI(model="gpt-4.1"), tools=glanos_tools.tools)

response = agent.invoke(
    {"messages": [{"role": "user", "content": "Anonymize this: Michael was in Munich."}]}
)
print(response["messages"][-1].content)

glanos_tools.close()
```
