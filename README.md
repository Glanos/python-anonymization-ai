# Glanos anonymization.ai Client

> [!IMPORTANT]
> **License / Access Required**
>
> To use the Glanos anonymization.ai Library, you need a valid **API token** and **Base URL**.
>
> To obtain your credentials, please contact **license@glanos.de**.
>
> Once you have received your token and Base URL, you can proceed with the installation and usage instructions below.
> Further information can be found at https://www.glanos.de/anonymization-ai/


A Python client library for the Glanos anonymization.ai API — anonymize (pseudonymize) text and documents, and reliably restore them later. Includes a command-line tool and a LangChain agent middleware for anonymizing user input before it reaches an LLM and restoring the original values in the model's replies.

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
from glanos_anonymization import AnonymizationSettings, GlanosClient

client = GlanosClient(
    token="YOUR_GLANOS_TOKEN",
    base_url="https://your-glanos-instance.com",
)

result = client.anonymize(
    text="""
    My name is Max Mustermann.
    I live in Munich.
    My phone number is 01711234567.
    """)

print(result.text)
```

Example output:

```text
My name is PERSON1.
I live in LOCATION1.
My phone number is PHONE1.
```

`result.text` is the document's text, extracted so that it can be read directly in Python (via [MarkItDown](https://github.com/microsoft/markitdown) for files). The anonymized document itself is `result.data` (file bytes, named `result.file_name`); the full API response, including the keys to restore it, is `result.pseudo_result`.

## Anonymizing Files

```python
from pathlib import Path

result = client.anonymize(path="./documents/example.docx")
print(result.text)                                  # extracted text, for reading
Path(result.file_name).write_bytes(result.data)     # the anonymized .docx
```

A document already in memory is passed as bytes: `client.anonymize(data=content, filename="example.docx")` (the file name with its extension is required then).

The client automatically waits for asynchronous processing to finish and extracts text from supported document types (`.docx`, `.pdf`, `.pptx`, `.xlsx`, and more).

### Pending results (HTTP 202)

The API answers with HTTP 202 and a `fetchKey` instead of the anonymized document while a document is still being processed — always in async mode (`a_sync=True`), and in sync mode (the API default) when the request hits the server-side timeout. `pseudo_sync()` returns such a result with `result.is_pending == True`; `fetch()` polls until it is done:

```python
result = client.pseudo_sync(path="./documents/large.pdf")
if result.is_pending:
    result = client.fetch(result)
```

`anonymize()` does this for you. After a sync timeout it fetches exactly the keys the sync request asked for (`return_pseudo_key`, `return_encrypt_key`), so the result looks the same as a sync call that finished in time.

## Restoring Anonymized Documents (Deanonymization)

Anonymization is reversible. `deanonymize()` restores the result of `anonymize()` — text and files alike. It restores the anonymized document itself (`result.data`), not the extracted `result.text`:

```python
from pathlib import Path
from glanos_anonymization import AnonymizationSettings

result = client.anonymize(path="./documents/example.docx")
restored = client.deanonymize(result)

Path(restored.file_name).write_bytes(restored.data)   # the restored .docx
print(restored.text)                                  # its text, for reading
```

By default it uses the keys of the result: the client requests the `pseudoKey` and the `encryptKey` with every synchronous request, and that pair restores a document regardless of the retention. A result anonymized with a **session** can also be restored with it:

```python
from glanos_anonymization import AnonymizationSettings, GlanosClient

with GlanosClient(token=token, base_url=base_url) as client:
    stored = client.add_setting(AnonymizationSettings(name="My settings"))
    session = client.add_session(max_age_seconds=24 * 60 * 60, settings_id=stored.id)

    result = client.anonymize(text="Michael was in Munich.", session=session.id)
    print(result.text)  # "PERSON1 was in LOCATION1."

    print(client.deanonymize(result, session=session.id).text)  # "Michael was in Munich."

    client.remove_session(session.id)
```

## Anonymization Settings

How text is anonymized (mode, excluded entity types, fold mode, ...) is described by an `AnonymizationSettings` object. Settings are passed **per call**, and the client only sends what you pass:

```python
from glanos_anonymization import AnonymizationSettings

settings = AnonymizationSettings(mode="TYPE_PRESERVING", excludeFields=["dates"])

client.anonymize(text="...", settings=settings)          # self-defined (inline) settings
client.anonymize(text="...", settings_id=stored.id)      # a setting stored on the server
client.anonymize(text="...", session=session.id)         # the setting the session was created with
```

`anonymize()` and `pseudo_sync()` accept any combination of `settings`, `settings_id` and `session`. The API decides which settings apply, in this order:

1. `settings` (self-defined settings)
2. `settings_id` (a stored setting)
3. the setting of the `session`

When a session is used together with `settings` or `settings_id`, their `mode` must be the same as the session's. Otherwise the API rejects the request, and the client raises `SettingsConflictError` (a subclass of `InvalidInputError`):

```python
from glanos_anonymization import SettingsConflictError

try:
    client.anonymize(text="...", session=session.id, settings=AnonymizationSettings(mode="XXX"))
except SettingsConflictError as e:
    print(e.api_message)  # "Session settings mode and provided settings mode must be equal"
```

**Without settings, the installation's default setting applies.** If a request carries neither `settings`, `settings_id` nor `session`, the API uses the default setting of the installation (every base URL has one; it cannot be changed or removed). These are *not* the defaults of the API schema, nor those of `AnonymizationSettings()`: a default setting may, for example, use mode MT and leave dates untouched. `list_settings()` shows it as the setting named `Default`. Pass `settings` or `settings_id` whenever a request needs specific settings.

The API also stores inline settings sent with a pseudo request as a stored setting named `Upload <timestamp>` (one per distinct combination).

## Sessions

A session stores the pseudonym mapping, so the same value gets the same pseudonym across multiple requests and can be restored later. Sessions are not needed for the anonymization of single documents. Create a session when several anonymization calls should use consistent placeholders.

A session is based on a stored setting (see [Stored Settings](#stored-settings)). Requests that pass only the session are anonymized with that setting.

```python
session = client.add_session(max_age_seconds=3600, settings_id=stored.id)  # POST /session/add
info = client.get_session(session.id)                              # GET  /session/get
info = client.extend_session(session.id, max_age_seconds=7200)     # GET  /session/extend
client.remove_session(session.id)                                  # PUT  /session/remove
```

`share_users=` and `share_groups=` control who else may use the session.

Mind the units: `add_session` and `extend_session` take **seconds**, as `/session/add` and `/session/extend` do.

`SessionInfo` timestamps (`createdAt`, `lastUpdatedAt`, `expiresAt`) are epoch milliseconds.

## Stored Settings

Settings can be stored on the server and referenced by id, via `settings_id=` or `add_session(settings_id=...)`:

```python
from glanos_anonymization import AnonymizationSettings, SettingsSort, SortOrder

stored = client.add_setting(                                       # POST /setting/add
    AnonymizationSettings(name="Legal DE", mode="TYPE_PRESERVING", excludeFields=["dates"])
)
print(stored.id)  # assigned by the server

setting = client.get_setting(stored.id)                            # GET  /setting/get
setting.excludeFields = []
client.update_setting(setting)                                     # POST /setting/update

page = client.list_settings(query="legal", sort=SettingsSort.NAME, sort_order=SortOrder.ASC, limit=20)
for item in page.list:                                             # GET  /setting/list
    print(item.id, item.name, item.mode)

client.remove_setting(stored.id)                                   # PUT  /setting/remove
```

- `add_setting(settings)` creates a new setting. The server assigns the id, so `settings.id` must not be set.
- `update_setting(settings)` replaces an existing setting, identified by `settings.id` (the only required field). **Pass the complete setting**, not only the fields that change: fields left out are reset, the name included. To change a field, read the setting with `get_setting()`, change it and pass it back, as above. If no setting has that id, it raises `SettingNotFoundError`.

Both take `apply_defaults` (default `True`, sent as `applyDefaults`): whether the API applies its system defaults to the setting — the spec calls `False` an expert setting.

## Request Options

The client has no configuration of its own. Like the settings, the request options of `/document/pseudo` are passed per call to `anonymize()` and `pseudo_sync()`. Anything left out is up to the API's defaults — except the keys: synchronous requests ask for the `pseudoKey` and the `encryptKey` unless turned off, so that every result can be restored (see [Restoring](#restoring-anonymized-documents-deanonymization)).

```python
result = client.pseudo_sync(
    path="./documents/example.docx",
    a_sync=False,                  # API default: false - wait for the result
    return_pseudo_key=True,        # client default for sync requests: true (API: false)
    return_encrypt_key=True,       # client default for sync requests: true (API: false)
    language="de",                 # API default: detected automatically
    data_retention_ms=0,           # API default: 0 = expires right after download, -1 = keep forever
    pseudo_key_retention_ms=0,     # API default: 0 = expires right after download
    meta_retention_ms=600_000,     # API default: 0; > 0 keeps the metaKey usable for restoring
    assignee="reviewer@example.com",
    options=["..."],               # installation-specific upload options
)
```

`return_pseudo_key` and `return_encrypt_key` only apply to synchronous requests; setting them to `True` together with `a_sync=True` raises a `ValueError` before anything is sent. An asynchronous result gets its keys from `fetch()`, which also requests both by default.

Text passed with `text=` is sent as `plainText`. It gets no `metaKey`, so it is restored with the `pseudoKey` + `encryptKey` pair (or a session); the retention and the `metaKey` only apply to files.

To turn off SSL certificate verification, construct the client with `GlanosClient(..., no_ssl_verify=True)`.

## Error Handling

`GlanosClient` raises a typed exception for every HTTP error status, all deriving from `GlanosClientError`:

| Exception | When |
| --- | --- |
| `QuotaExceededError` | HTTP 402 — license expired or quota exceeded |
| `AuthenticationError` | HTTP 401/403 — e.g. an invalid token |
| `DataExpiredError` | HTTP 410 — data or session expired |
| `RateLimitedError` | HTTP 429 — too many requests |
| `SettingsConflictError` | HTTP 400 — the mode of the given settings differs from the session's |
| `InvalidInputError` | any other HTTP 4XX — invalid input (base class of `AuthenticationError` and `SettingsConflictError`) |
| `ServerError` | HTTP 5XX — internal error |
| `ProcessingFailedError` | a 2xx response whose `data.success` is `false`; carries `fail_reason` |
| `FetchTimeoutError` | `fetch()` gave up polling a pending result |
| `SettingNotFoundError` | `update_setting()` was given the id of a setting that does not exist |

All HTTP errors derive from `GlanosAPIError`, which exposes `status_code` and `api_message` (the API's own explanation, e.g. `"Invalid document"`). `GlanosAPIError` is also an `httpx.HTTPStatusError`, so existing code catching that keeps working. The API token is never part of an error message.

```python
from glanos_anonymization import (
    AuthenticationError, DataExpiredError, GlanosAPIError, GlanosClientError,
    QuotaExceededError, RateLimitedError,
)

try:
    client.pseudo_sync(text="...")
except QuotaExceededError:
    ...  # license expired or quota exceeded (HTTP 402)
except AuthenticationError:
    ...  # invalid token (HTTP 401/403)
except RateLimitedError:
    ...  # too many requests (HTTP 429)
except DataExpiredError:
    ...  # data or session expired (HTTP 410)
except GlanosAPIError as e:
    print(e.status_code, e.api_message)  # any other HTTP error
except GlanosClientError:
    ...  # any other client error
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
glanos-anonymization depseudo --token TOKEN --baseUrl URL --file anonymized.txt --pseudoKeyFile anonymized.pkey --encryptKeyFile anonymized.ekey --outputFile restored.txt
```

Without `--async`, `pseudo` writes the keys next to the output file (`anonymized.pkey`, `anonymized.ekey`); `depseudo` restores the document with them as shown above. `--noReturnPseudoKey` / `--noReturnEncryptKey` turn that off.

```bash
glanos-anonymization setting-add --token TOKEN --baseUrl URL --name "Legal DE" --mode TYPE_PRESERVING --excludeFields dates
glanos-anonymization setting-update --token TOKEN --baseUrl URL --setting SETTING_ID --excludeFields locations
glanos-anonymization setting-list --token TOKEN --baseUrl URL --query legal --sort name --limit 20
glanos-anonymization setting-get --token TOKEN --baseUrl URL --setting SETTING_ID
glanos-anonymization setting-remove --token TOKEN --baseUrl URL --setting SETTING_ID
```

`setting-add` stores a new setting; the server assigns its id. `setting-update` changes only the fields given: it reads the setting and sends it back complete with the changes applied.

```bash
glanos-anonymization session-add --token TOKEN --baseUrl URL --settingsId SETTING_ID --maxAgeSeconds 3600
glanos-anonymization session-get --token TOKEN --baseUrl URL --session SESSION_ID
glanos-anonymization session-extend --token TOKEN --baseUrl URL --session SESSION_ID --maxAgeSeconds 7200
glanos-anonymization session-remove --token TOKEN --baseUrl URL --session SESSION_ID
```

The session and setting commands print the API's result as JSON.

`pseudo` sends settings only when asked to. Settings flags given to `pseudo` (`--mode`, `--excludeFields`, `--foldMode`, ...) are sent as inline settings for that request only, without storing anything. `--settingsId` and `--session` reference a stored setting or a session, with the precedence described in [Anonymization Settings](#anonymization-settings):

```bash
glanos-anonymization pseudo --token TOKEN --baseUrl URL --file document.docx --mode TYPE --excludeFields dates
glanos-anonymization pseudo --token TOKEN --baseUrl URL --file document.docx --settingsId SETTING_ID
glanos-anonymization pseudo --token TOKEN --baseUrl URL --file document.docx --session SESSION_ID
```

Run `glanos-anonymization <command> --help` for the full set of options per command (anonymization mode, excluded entity types, retention windows, sessions, and more).

## LangChain Middleware

`GlanosAnonymizationMiddleware` anonymizes user messages before they reach the model and restores the model's replies afterward, using one Glanos anonymization.ai session per conversation so the same entity always maps to the same pseudonym across every turn - e.g. once "Michael" becomes "PERSON1", it stays "PERSON1" for the rest of the conversation, and any reply mentioning "PERSON1" is turned back into "Michael" before you see it.

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

The middleware accepts `settings=` (an `AnonymizationSettings` object), `settings_id=` (a stored setting), `language=` and `no_ssl_verify=`. With `settings_id`, the conversation's session is created from that setting via `/session/add`. Given settings are also sent with every anonymize call, with the precedence described in [Anonymization Settings](#anonymization-settings). With neither, no settings are sent and the installation's default setting applies:

```python
middleware = GlanosAnonymizationMiddleware(token=token, base_url=base_url, settings_id=stored.id)
```

## Tools for Agents

`GlanosAnonymizationTools` offers everything the library can do as tools, so that an agent decides itself what to anonymize, restore or configure:

| Tool | Does |
| --- | --- |
| `anonymize` | anonymize `text`, or a document given as `file_base64` + `file_name`; returns the anonymized text (and document) plus `pseudo_key` / `encrypt_key` |
| `deanonymize` | restore a text or document with `pseudo_key` + `encrypt_key`, or with the `session` it was anonymized with |
| `list_settings`, `get_setting`, `add_setting`, `update_setting`, `remove_setting` | manage stored settings |
| `add_session`, `get_session`, `extend_session`, `remove_session` | manage sessions for consistent placeholders across calls |

Every tool describes when to use it and how it fits together with the others, and `GlanosAnonymizationTools.usage_guide` sums up the typical flows for the agent's system prompt. The tools are stateless: documents go in and out base64-encoded, and keys and session ids are returned to the agent and passed back by it. Errors (e.g. `HTTP 400: Settings name already exists`) are reported to the agent as the tool's result.

```python
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from glanos_anonymization import GlanosAnonymizationTools

glanos_tools = GlanosAnonymizationTools(token="YOUR_GLANOS_TOKEN", base_url="https://your-glanos-instance.com")

agent = create_agent(
    model=ChatOpenAI(model="gpt-4.1"),
    tools=glanos_tools.tools,
    system_prompt=glanos_tools.usage_guide,
)

response = agent.invoke(
    {"messages": [{"role": "user", "content": "Anonymize this: Michael was in Munich."}]}
)
print(response["messages"][-1].content)

glanos_tools.close()
```

`glanos_tools.definitions()` lists every tool as `(name, description, method)`. The methods have typed, described parameters and return JSON-compatible dicts, so they can be registered unchanged with other frameworks, e.g. an MCP server: `mcp.tool(name=name, description=description)(method)`. One `GlanosAnonymizationTools` instance works with one Glanos anonymization.ai token.
