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


A Python client library for the Glanos anonymization.ai API — anonymize (pseudonymize) text and documents, and reliably restore them later. Includes an async client (`AsyncGlanosClient`), a command-line tool and a LangChain agent middleware for anonymizing user input before it reaches an LLM and restoring the original values in the model's replies.

## Installation

Python **3.11, 3.12, 3.13 or 3.14**. Install only what you need:

| Installation | What you get | pip | uv |
| --- | --- | --- | --- |
| **Core** | Sync and async client, CLI, anonymizing and restoring text and documents, settings, sessions | `pip install glanos-anonymization` | `uv add glanos-anonymization` |
| **Text extraction** | Core + `extract_text()` for documents (`.docx`, `.pdf`, `.pptx`, `.xlsx`, ...) via MarkItDown | `pip install 'glanos-anonymization[text-extraction]'` | `uv add 'glanos-anonymization[text-extraction]'` |
| **LangChain** | Core + LangChain middleware and LangChain tools | `pip install 'glanos-anonymization[langchain]'` | `uv add 'glanos-anonymization[langchain]'` |
| **Both** | All of the above | `pip install 'glanos-anonymization[text-extraction,langchain]'` | `uv add 'glanos-anonymization[text-extraction,langchain]'` |

The Core installation needs only `httpx` and `pydantic`. The extras are imported only when you use them; using a feature whose extra is missing raises an `ImportError` that names the command to install it.

The `langchain` extra does not install a model provider. For the OpenAI examples below, also install `langchain-openai`.

## Credentials

Keep the token out of your source code. The examples read it from the environment variables `GLANOS_TOKEN` and `GLANOS_BASE_URL`, which the CLI also reads by itself:

```bash
export GLANOS_TOKEN=...
export GLANOS_BASE_URL=https://your-glanos-instance.com
```

`GLANOS_BASE_URL` is the root URL of your installation, without `/be/api-v2`.

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

print(result.extract_text())
```

Example output:

```text
My name is PERSON1.
I live in LOCATION1.
My phone number is PHONE1.
```

The exact replacements depend on your installation and its anonymization settings.

### What a result contains

`anonymize()` and `deanonymize()` return what the API returned, nothing is converted up front:

| Attribute | Content |
| --- | --- |
| `result.ready` | `False` while the document is not processed yet (see [Waiting for the result](#waiting-for-the-result)) |
| `result.fetch_key` | for a result that is not ready: the key to collect it with `fetch()` |
| `result.data` | the anonymized (or restored) document, as bytes |
| `result.file_name` | its file name |
| `result.pseudo_key`, `result.encrypt_key`, `result.meta_key`, `result.session` | what restores it (see [Restoring](#restoring-anonymized-documents-deanonymization)) |
| `result.pseudo_result` | the full API response |
| `result.extract_text()` | the document's text, extracted when you call it |

On a result that is not ready, everything but `ready` and `fetch_key` raises `ResultNotReadyError`.

`extract_text()` decodes text results directly. Other documents are converted with [MarkItDown](https://github.com/microsoft/markitdown), which needs the `text-extraction` extra. To use another converter, pass `text_extractor=` to the client (any object with an `extract_text(pseudo_result) -> str` method), or to a single call: `result.extract_text(my_extractor)`.

## Anonymizing Files

```python
from pathlib import Path

result = client.anonymize(path="./documents/example.docx")
print(result.extract_text())                        # extracted text, for reading
Path(result.file_name).write_bytes(result.data)     # the anonymized .docx
```

A document already in memory is passed as bytes: `client.anonymize(data=content, filename="example.docx")` (the file name with its extension is required then).

Anonymizing and restoring files works with the Core installation; only `extract_text()` on a document needs the `text-extraction` extra.

## Waiting for the Result

Large documents can take a while. Instead of the document, the API then answers with a `fetchKey`, which collects the document later. `processing=` decides how long `anonymize()` waits:

| `processing` | What happens | Default for |
| --- | --- | --- |
| `"sync"` | One request that waits for the document. If the server does not finish in time, the result is not ready. | `text=` |
| `"poll"` | Submits the document, then checks every `poll_interval` seconds (default 10) whether it is ready, at most `max_attempts` times (default 30). | `path=`, `data=` |
| `"no_wait"` | Submits the document and returns at once; the result is never ready. | |

`poll_interval` and `max_attempts` only apply to `"poll"`; passing them with another mode raises a `ValueError`. Whatever the mode, **running out of time is not an error**: `anonymize()` returns a result with `ready == False` and the `fetch_key`. Collect the document with `fetch()`, now or later, e.g. from another process:

```python
result = client.anonymize(path="./documents/large.pdf")
if not result.ready:
    save(result.fetch_key)  # e.g. in your database

# later
result = client.fetch(saved_fetch_key, poll_interval=10, max_attempts=60)
```

`fetch()` takes the result or its fetch key. It checks every `poll_interval` seconds, by default every **5 minutes without a limit**, which suits [documents in review](#human-review). With `max_attempts`, it gives up after that many checks and returns the result still not ready, with its `fetch_key`. Any error (HTTP error, network problem, ...) ends the polling; the error carries the fetch key in a note, so you can call `fetch()` again later.

> **`processing` is not Python `async`.** It decides how the client waits for the server. For non-blocking Python code, use `AsyncGlanosClient` (below).

## Human Review

`anonymize_with_review()` sends a document to a person who checks the anonymization in the Glanos anonymization.ai dashboard. Pass exactly one of `assignee` (a user, usually the lower-cased e-mail address) or `assigned_group` (all members of a group):

```python
pending = client.anonymize_with_review(path="./documents/contract.docx", assignee="reviewer@example.com")
save(pending.fetch_key)

# later, e.g. in a scheduled job: one check
result = client.fetch(saved_fetch_key, max_attempts=1)
if result.ready:
    Path(result.file_name).write_bytes(result.data)

# or wait as long as the review takes, checking every 5 minutes
result = client.fetch(saved_fetch_key)
```

It always returns at once, without polling: the document is ready only once the reviewer is done, which may take days. Keep the `fetch_key`, so that waiting can resume after your program restarts. It takes the same arguments as `anonymize()` except the processing and polling options.

The API does not check that the user or group exists. When the data expires (`data_retention_ms`, `meta_retention_ms`), the document also disappears from the reviewer's dashboard, so choose a retention that covers the time the review may take.

## Async Client

`AsyncGlanosClient` offers the same methods as `GlanosClient`, as coroutines, built on `httpx.AsyncClient`:

```python
import asyncio
import os
from glanos_anonymization import AsyncGlanosClient


async def main() -> None:
    async with AsyncGlanosClient(
        token=os.environ["GLANOS_TOKEN"],
        base_url=os.environ["GLANOS_BASE_URL"],
    ) as client:
        texts = ["Max Mustermann was in Munich.", "Anna Schmidt lives in Berlin."]
        results = await asyncio.gather(*(client.anonymize(text=t) for t in texts))
        for result in results:
            print(result.extract_text())


asyncio.run(main())
```

Both clients build the same requests and return the same results. Two things stay synchronous in the async client: reading a file passed as `path=`, and `extract_text()`. For large documents, run the extraction in a thread: `await asyncio.to_thread(result.extract_text)`. Close a client that is not used with `async with` by `await client.aclose()`.

## Restoring Anonymized Documents (Deanonymization)

Anonymization is reversible. `deanonymize()` restores the result of `anonymize()` — text and files alike.

```python
from pathlib import Path

result = client.anonymize(path="./documents/example.docx")
restored = client.deanonymize(result)

Path(restored.file_name).write_bytes(restored.data)   # the restored .docx
print(restored.extract_text())                        # its text, for reading
```

By default it uses the keys of the result: the client requests the `pseudoKey` and the `encryptKey` for every document it waits for, and that pair restores a document regardless of the retention. A result anonymized with a **session** can also be restored with it:

```python
from glanos_anonymization import AnonymizationSettings

stored = client.add_setting(AnonymizationSettings(name="My settings"))
session = client.add_session(max_age_seconds=24 * 60 * 60, settings_id=stored.id)

result = client.anonymize(text="Michael Huber was in Munich.", session=session.id)
print(result.extract_text())  # e.g. "PERSON1 was in LOCATION1."

print(client.deanonymize(result, session=session.id).extract_text())  # "Michael Huber was in Munich."
```

`deanonymize()` also restores anonymized content without the original result: pass `text=`, a file (`path=`) or bytes (`data=` with `filename=`), together with the `session` or the keys (`depseudo_paths=DepseudoPaths(pseudo_key=..., encrypt_key=...)`). With a session, any text containing its pseudonyms can be restored, e.g. a reply written by a model:

```python
restored = client.deanonymize(text="PERSON1 was in LOCATION1.", session=session.id)
print(restored.extract_text())  # "Michael Huber was in Munich."

# the same in one step, returning the string
print(client.depseudo_text("PERSON1 was in LOCATION1.", session=session.id))
```

**Handle the keys as secrets.** Together with the anonymized document, `pseudoKey` and `encryptKey` give back the original. Keep them out of logs, model prompts, traces and source control. Pseudonymization is reversible by design; it is not irreversible anonymization (for that, use mode `XXX`).

## Anonymization Settings

How text is anonymized (mode, excluded entity types, fold mode, ...) is described by an `AnonymizationSettings` object. Settings are passed **per call**, and the client only sends what you pass:

```python
from glanos_anonymization import AnonymizationSettings

settings = AnonymizationSettings(mode="TYPE_PRESERVING", excludeFields=["dates"])

client.anonymize(text="...", settings=settings)          # self-defined (inline) settings
client.anonymize(text="...", settings_id=stored.id)      # a setting stored on the server
client.anonymize(text="...", session=session.id)         # the setting the session was created with
```

Pass either `settings` or `settings_id`, not both: the API would ignore `settings_id`, so the client raises a `ValueError`. Either can be combined with a `session`; the given settings then take precedence over the setting of the session, and their `mode` must be the same as the session's. Otherwise the API rejects the request, and the client raises `SettingsConflictError` (a subclass of `InvalidInputError`):

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
info = client.extend_session(session.id, max_age_seconds=7200)     # PUT  /session/extend
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

The client has no configuration of its own. Like the settings, the request options of `/document/pseudo` are passed per call to `anonymize()` (and, except the processing options, to `anonymize_with_review()`). Anything left out is up to the API's defaults — except the keys: the client asks for the `pseudoKey` and the `encryptKey` unless turned off, so that every result can be restored (see [Restoring](#restoring-anonymized-documents-deanonymization)).

```python
result = client.anonymize(
    path="./documents/example.docx",
    processing="poll",             # see "Waiting for the result"
    return_pseudo_key=True,        # client default: true (API: false)
    return_encrypt_key=True,       # client default: true (API: false)
    language="de",                 # API default: detected automatically
    session_max_age_ms=3_600_000,  # without session=: creates a session with this lifetime
    data_retention_ms=0,           # API default: 0 = expires right after download, -1 = keep forever
    pseudo_key_retention_ms=0,     # API default: 0 = expires right after download
    meta_retention_ms=600_000,     # API default: 0; > 0 keeps the metaKey usable for restoring
    options=["..."],               # installation-specific upload options
)
```

With `processing="no_wait"` the keys are requested when the document is collected, so `return_pseudo_key` and `return_encrypt_key` raise a `ValueError` there; pass them to `fetch()` instead, which requests both by default.

Text passed with `text=` is sent as `plainText`. It gets no `metaKey`, so it is restored with the `pseudoKey` + `encryptKey` pair (or a session); the retention and the `metaKey` only apply to files.

To turn off SSL certificate verification, construct the client with `GlanosClient(..., no_ssl_verify=True)`. `timeout=` sets the HTTP timeout in seconds (default 30).

## Error Handling

The clients raise a typed exception for every HTTP error status, all deriving from `GlanosClientError`:

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
| `ResultNotReadyError` | the result of a document that is not ready was read; carries `fetch_key` |
| `SettingNotFoundError` | `update_setting()` was given the id of a setting that does not exist |

All HTTP errors derive from `GlanosAPIError`, which exposes `status_code` and `api_message` (the API's own explanation, e.g. `"Invalid document"`). `GlanosAPIError` is also an `httpx.HTTPStatusError`, so existing code catching that keeps working. The API token is never part of an error message.

```python
from glanos_anonymization import (
    AuthenticationError, DataExpiredError, GlanosAPIError, GlanosClientError,
    QuotaExceededError, RateLimitedError,
)

try:
    client.anonymize(text="...")
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

Installing the package also installs a `glanos-anonymization` command. With `GLANOS_TOKEN` and `GLANOS_BASE_URL` set (see [Credentials](#credentials)), `--token` and `--baseUrl` can be left out, which keeps the token out of the shell history and the process list:

```bash
glanos-anonymization pseudo --file document.docx --outputFile anonymized.docx
```

```bash
glanos-anonymization pseudo --text "Michael Huber was in Munich." --outputFile out.txt
```

```bash
glanos-anonymization fetch --fetchKey FETCH_KEY --outputFile out.txt
```

```bash
glanos-anonymization depseudo --file anonymized.txt --pseudoKeyFile anonymized.pkey --encryptKeyFile anonymized.ekey --outputFile restored.txt
```

The flags still work and take precedence: `glanos-anonymization pseudo --token TOKEN --baseUrl URL ...`.

`pseudo` waits for the result as described in [Waiting for the result](#waiting-for-the-result): `--processing sync|poll|no_wait`, and for `poll` also `--pollInterval` (seconds) and `--maxAttempts`. If the document is not ready in time, or with `--processing no_wait`, it prints the fetchKey as JSON and the `fetch` command to collect it. `fetch` checks every 5 minutes without a limit by default; `--pollInterval` and `--maxAttempts` change that (`--maxAttempts 1` checks once).

```bash
glanos-anonymization pseudo --file large.pdf --processing no_wait
glanos-anonymization fetch --fetchKey FETCH_KEY --pollInterval 10 --maxAttempts 60 --outputFile anonymized.pdf
```

With `--assignee` or `--assignedGroup`, `pseudo` sends the document for [human review](#human-review): it only submits it and prints the fetchKey; collect the reviewed document with `fetch`.

When `pseudo` or `fetch` gets the document, it writes the keys next to the output file (`anonymized.pkey`, `anonymized.ekey`); `depseudo` restores the document with them as shown above. `--noReturnPseudoKey` / `--noReturnEncryptKey` turn that off. The keys are never printed, and the key files are readable by their owner only (`0600`): together they restore the original document, so treat them like the document itself.

```bash
glanos-anonymization setting-add --name "Legal DE" --mode TYPE_PRESERVING --excludeFields dates
glanos-anonymization setting-update --setting SETTING_ID --excludeFields locations
glanos-anonymization setting-list --query legal --sort name --limit 20
glanos-anonymization setting-get --setting SETTING_ID
glanos-anonymization setting-remove --setting SETTING_ID
```

`setting-add` stores a new setting; the server assigns its id. `setting-update` changes only the fields given: it reads the setting and sends it back complete with the changes applied.

```bash
glanos-anonymization session-add --settingsId SETTING_ID --maxAgeSeconds 3600
glanos-anonymization session-get --session SESSION_ID
glanos-anonymization session-extend --session SESSION_ID --maxAgeSeconds 7200
glanos-anonymization session-remove --session SESSION_ID
```

The session and setting commands print the API's result as JSON.

`pseudo` sends settings only when asked to. Settings flags given to `pseudo` (`--mode`, `--excludeFields`, `--foldMode`, ...) are sent as inline settings for that request only, without storing anything. `--settingsId` and `--session` reference a stored setting or a session, as described in [Anonymization Settings](#anonymization-settings) (settings flags and `--settingsId` exclude each other):

```bash
glanos-anonymization pseudo --file document.docx --mode TYPE --excludeFields dates --outputFile anonymized.docx
glanos-anonymization pseudo --file document.docx --settingsId SETTING_ID --outputFile anonymized.docx
glanos-anonymization pseudo --file document.docx --session SESSION_ID --outputFile anonymized.docx
```

Run `glanos-anonymization <command> --help` for the full set of options per command (anonymization mode, excluded entity types, retention windows, sessions, and more).

## LangChain Middleware

Needs the `langchain` extra (and a model provider, here `langchain-openai`).

`GlanosAnonymizationMiddleware` makes sure the model never sees the original values. Right before every model call, it anonymizes the whole request: the system prompt, user messages, the model's earlier replies and tool calls, and tool results. The model's reply, including the arguments of the tools it calls, is restored before it enters the agent state. It uses one Glanos anonymization.ai session per conversation, so the same entity always maps to the same pseudonym across every turn - e.g. once "Michael Huber" becomes "PERSON1", it stays "PERSON1" for the rest of the conversation, and any reply mentioning "PERSON1" is turned back into "Michael Huber" before you see it.

Keeping that mapping consistent across separate `invoke()` calls requires a LangGraph checkpointer and a stable `thread_id`:

```python
import os

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

from glanos_anonymization import GlanosAnonymizationMiddleware

middleware = GlanosAnonymizationMiddleware(
    token=os.environ["GLANOS_TOKEN"],
    base_url=os.environ["GLANOS_BASE_URL"],
)
try:
    agent = create_agent(
        model=ChatOpenAI(model="gpt-6-luna", reasoning_effort="none"),
        middleware=[middleware],      # keep it last in the list
        checkpointer=InMemorySaver(),  # use a persistent checkpointer in production
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
finally:
    middleware.close()
```

Without a checkpointer, a fresh session is created on every `invoke()` call, and consistency only holds within that single call.

What this means in practice:

- **The agent state holds the original values.** Your application and your tools work with real data (a tool called with "PERSON1" by the model receives "Michael Huber"); a checkpointer therefore stores real data too.
- **Put the middleware last** in the `middleware` list, so that it sees the request exactly as it goes to the model, after other middleware changed it.
- **Use the middleware from the beginning of a conversation.** In a conversation started without it, the model may have stored its tool calls as content blocks with real values; these are not anonymized afterwards.
- **Content that is not text** (images, files, audio, ...) is **not anonymized**: it reaches the model as it is. Personal data in a photo, a scanned document or a voice message is therefore sent to the model in clear; send such content at your own risk.
- **Each text costs one API call, once.** Results are cached per session, and a restored reply is sent back to the model exactly as the model wrote it.
- **Async agents** (`agent.ainvoke()`) are supported; the Glanos calls run in a worker thread and do not block the event loop.
- **Not covered:** middleware that calls a model on its own (e.g. `SummarizationMiddleware`) and the `structured_response` of `response_format`, which is not restored.
- If the conversation has no session, the middleware raises `GlanosClientError` instead of sending the request.
- `verbose=True` prints what is sent to and received from the model - pseudonymized text only, never the restored values.

The middleware accepts `settings=` (an `AnonymizationSettings` object), `settings_id=` (a stored setting), `language=`, `session_max_age_ms=` and `no_ssl_verify=`. With `settings_id`, the conversation's session is created from that setting via `/session/add`. Given settings are also sent with every anonymize call (with both, only `settings`, which take precedence anyway). With neither, the middleware uses `AnonymizationSettings(mode="TYPE")`. To use a stored setting instead:

```python
middleware = GlanosAnonymizationMiddleware(token=token, base_url=base_url, settings_id=stored.id)
```

## Tools for Agents

`GlanosAnonymizationTools` offers everything the library can do as tools, so that an agent decides itself what to anonymize, restore or configure:

| Tool | Does |
| --- | --- |
| `anonymize` | anonymize `text`, or a document given as `file_base64` + `file_name`, with the `session` the agent passes, or in a new session; returns the anonymized text (for documents the document, and its text if it can be extracted) plus the `session` |
| `anonymize_with_review` | send a text or document for [human review](#human-review); returns at once with the `fetch_key` |
| `fetch` | collect a document that was not ready yet, e.g. in review, with its `fetch_key`; checks once, and returns the `fetch_key` again if the document is still not ready |
| `deanonymize` | restore a text or document with the `session` it was anonymized with |
| `list_settings`, `get_setting`, `add_setting`, `update_setting`, `remove_setting` | manage stored settings |
| `add_session`, `get_session`, `extend_session`, `remove_session` | manage sessions, e.g. to share one or to keep it longer |

Every tool describes when to use it and how it fits together with the others, and `GlanosAnonymizationTools.usage_guide` sums up the typical flows for the agent's system prompt. The tools are stateless: documents go in and out base64-encoded, and session ids are returned to the agent and passed back by it. Errors (e.g. `HTTP 400: Settings name already exists`) are reported to the agent as the tool's result.

The tools restore only with sessions: `anonymize` never requests the `pseudoKey` and `encryptKey`, so no keys enter the model's context or your traces. The agent decides whether to continue a session, by passing its id (from an earlier call, or one your application gives it), or to start a new one. A new session lives 24 hours; `GlanosAnonymizationTools(..., session_max_age_ms=...)` changes that. Once a session has expired, what was anonymized with it can no longer be restored, unless the agent prolonged it with `extend_session`.

```python
import os

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from glanos_anonymization import GlanosAnonymizationTools

glanos_tools = GlanosAnonymizationTools(
    token=os.environ["GLANOS_TOKEN"],
    base_url=os.environ["GLANOS_BASE_URL"],
)
try:
    agent = create_agent(
        model=ChatOpenAI(model="gpt-6-luna", reasoning_effort="none"),
        tools=glanos_tools.tools,
        system_prompt=glanos_tools.usage_guide,
    )
    response = agent.invoke(
        {"messages": [{"role": "user", "content": "Anonymize this: Michael Huber was in Munich."}]}
    )
    print(response["messages"][-1].content)
finally:
    glanos_tools.close()
```

`glanos_tools.definitions()` lists every tool as `(name, description, method)`. The methods have typed, described parameters and return JSON-compatible dicts, so they can be registered unchanged with other frameworks, e.g. an MCP server: `mcp.tool(name=name, description=description)(method)`. `definitions()` does not need LangChain; only `glanos_tools.tools` does. One `GlanosAnonymizationTools` instance works with one Glanos anonymization.ai token.

## Development

```bash
uv sync                       # creates .venv with all extras and the dev tools
uv run pytest -m "not e2e"    # unit and integration tests, no network
uv run ruff format --check src tests && uv run ruff check src tests
```

The end-to-end tests run against a real installation. Put the credentials into a `.env` file in the repository root (it is git-ignored; pytest loads it automatically):

```dotenv
GLANOS_TOKEN=your-api-token
GLANOS_BASE_URL=https://your-glanos-instance.com
```

```bash
uv run pytest -m e2e          # uses API quota; creates and removes temporary settings and sessions
```

To test another supported Python version in a throwaway environment:

```bash
uv run --isolated --python 3.11 --all-extras pytest -m "not e2e"
```

`tests/test_optional_dependencies.py` checks that the package works without LangChain and without MarkItDown; for a real check, install the package without extras into a fresh virtual environment (`pip install .`) and import it.

## License

See [LICENSE](LICENSE).
