from datetime import datetime

from pydantic_ai import ModelSettings, RequestUsage
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models import Model, ModelRequestParameters


class EchoModel(Model):
    """
    Echo model.

    Echoes the last message back to the user. Useful for testing.
    """

    def __init__(self):
        super().__init__(settings=None, profile=None)

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        last_message = messages[-1].parts[-1].content
        return ModelResponse(
            parts=[TextPart(content=last_message)],
            model_name=self.model_name,
            timestamp=datetime.now(),
            usage=RequestUsage(),
            provider_name=self.system,
        )

    @property
    def model_name(self) -> str:
        """The model name."""
        return "echo"

    @property
    def system(self) -> str:
        """The model provider, ex: openai."""
        return "promptdev"
