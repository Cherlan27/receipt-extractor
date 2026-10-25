import logging

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import PydanticOutputParser

from receipt_extractor.llm.client import LlmClient
from receipt_extractor.llm.prompts.classify_prompt import CLASSIFY_PROMPT
from receipt_extractor.models.receipt import ReceiptClassification

CLASSIFY_MAX_NEW_TOKENS = 4096

_ROLES = {"system": "system", "human": "user", "ai": "assistant"}

logger = logging.getLogger(__name__)


class ClassificationError(Exception):
    """The language model's output could not be parsed into a classification."""

    def __init__(self, message: str, raw_output: str):
        super().__init__(message)
        self.raw_output = raw_output


class ReceiptClassifier:
    def __init__(self, llm_client: LlmClient):
        self.llm_client = llm_client
        self.parser = PydanticOutputParser(pydantic_object=ReceiptClassification)
        self.prompt = CLASSIFY_PROMPT.partial(
            format_instructions=self.parser.get_format_instructions()
        )

    def classify_text(self, text: str) -> ReceiptClassification:
        """
        Classify the products of a receipt into categories.

        Args:
            text (str): The receipt text, usually produced by OCR

        Returns:
            ReceiptClassification: The ordered, validated list of items

        Raises:
            ClassificationError: If the model output is not a valid classification
        """
        messages = [
            {"role": _ROLES[m.type], "content": m.content}
            for m in self.prompt.format_messages(receipt_text=text)
        ]
        raw_output = self.llm_client.chat(
            messages, max_new_tokens=CLASSIFY_MAX_NEW_TOKENS
        )

        try:
            return self.parser.parse(raw_output)
        except OutputParserException as e:
            logger.error(f"Invalid classification output: {raw_output!r}")
            raise ClassificationError(str(e), raw_output) from e
