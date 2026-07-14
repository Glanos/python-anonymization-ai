from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import HumanMessage
from glanos_anonymization.core import GlanosClient, AnonymizationConfig


class GlanosAnonymizationMiddleware(AgentMiddleware):
    """
    Anonymizes all HumanMessages before they are sent to the model
    """

    def __init__(
        self,
        token: str,
        base_url: str,
        config: AnonymizationConfig | None = None,
    ):
        self.client = GlanosClient(
            token=token,
            base_url=base_url,
            config=config,
        )

    def before_model(self, state, runtime):

        messages = []

        for message in state["messages"]:
            if isinstance(message, HumanMessage):
                result = self.client.anonymize(text=message.content)

                messages.append(
                    HumanMessage(
                        content=result.text,
                        id=message.id,
                        name=message.name,
                        additional_kwargs={
                            **message.additional_kwargs,
                            "glanos": result.pseudo_result.model_dump(),
                        },
                    )
                )
            else:
                messages.append(message)

        return {"messages": messages}
