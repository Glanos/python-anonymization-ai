import os
import pytest
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from glanos_anonymization.core import AnonymizationConfig
from glanos_anonymization.middleware import GlanosAnonymizationMiddleware


@pytest.mark.integration
def test_glanos_middleware():
    model = ChatOpenAI(
        model="gpt-4.1",
        api_key=os.environ["OPENAI_API_KEY"],
        temperature=0,
    )

    glanos_middleware = GlanosAnonymizationMiddleware(
        token=os.environ["GLANOS_TOKEN"],
        base_url=os.environ["GLANOS_BASE_URL"],
        config=AnonymizationConfig(),
    )

    agent = create_agent(
        model=model,
        middleware=[glanos_middleware],
    )

    response = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Mein Name ist Max Mustermann. "
                        "Ich wohne in München. "
                        "Meine Telefonnummer lautet 01711234567. "
                        "Fasse den Text in einem Satz zusammen."
                    ),
                }
            ]
        }
    )

    answer = response["messages"][-1].content
    print(answer)

    assert isinstance(answer, str)
    assert len(answer) > 0