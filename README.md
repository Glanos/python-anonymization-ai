# Glanos Anonymization Client

A Python client library for interacting with the Glanos anonymization API.

The client provides a simple interface for anonymizing text or documents. It automatically handles asynchronous jobs and extracts text from supported document formats.

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
from glanos_anonymization import (
    GlanosClient,
    AnonymizationConfig
)

client = GlanosClient(
    token="YOUR_GLANOS_TOKEN",
    base_url="https://your-glanos-instance.com"
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

The original API response is also available:

```python
print(result.pseudo_result)
```

## Anonymizing Files

```python
result = client.anonymize(
    path="./documents/example.docx"
)

print(result.text)
```

The client automatically waits for asynchronous processing to finish (if enabled) and extracts text from supported document types.

## Running Tests

```bash
uv run pytest
```

# LangChain Middleware

The client can be integrated into a LangChain agent using a custom middleware that anonymizes all user messages before they are sent to the language model.

Example usage:

```python
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

middleware = GlanosAnonymizationMiddleware(
    token="YOUR_GLANOS_TOKEN",
    base_url="https://your-glanos-instance.com",
)

agent = create_agent(
    model=ChatOpenAI(model="gpt-4.1"),
    middleware=[middleware],
)

response = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": (
                    "My name is Max Mustermann. "
                    "I live in Munich. "
                    "Summarize this text."
                ),
            }
        ]
    }
)

print(response["messages"][-1].content)
```