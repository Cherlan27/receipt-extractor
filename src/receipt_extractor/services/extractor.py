import io
import logging

import torch
from PIL import Image, UnidentifiedImageError
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_PATH = "zai-org/GLM-OCR"

logger = logging.getLogger(__name__)


def strip_special_tokens(text: str, tokenizer) -> str:
    """
    Remove every special token the tokenizer knows from the text.

    `skip_special_tokens=True` is not enough for GLM-OCR: tokens such as
    `<|begin_of_box|>` are listed as special but are still decoded.
    """
    for token in sorted(tokenizer.all_special_tokens, key=len, reverse=True):
        text = text.replace(token, "")
    return text


class ReceiptExtractor:
    def __init__(self, model_path="zai-org/GLM-OCR"):
        self.model_path = model_path
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.model = AutoModelForImageTextToText.from_pretrained(
            pretrained_model_name_or_path=self.model_path,
            torch_dtype="auto",
            device_map="auto",
        )

    def get_gpu_state():
        """
        Get state of CUDA and GPU.
        """

        print(f"CUDA verfügbar: {torch.cuda.is_available()}")
        print(f"GPU-Anzahl: {torch.cuda.device_count()}")
        if torch.cuda.is_available():
            print(f"GPU-Name: {torch.cuda.get_device_name(0)}")
            print(
                f"GPU-Speicher: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
            )

    def get_text(self, image: bytes) -> str:
        """
        Process the image and extract based on OCR model the text content:

        Args:
            image (bytes): The extracting image

        Returns:
            str: Extracting text
        """
        try:
            img = Image.open(io.BytesIO(image))
            img.load()  # Image.open is lazy: fail here on truncated/corrupt bodies
        except UnidentifiedImageError as e:
            logger.error(f"Invalid image format: {e}")
            return ""
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return ""

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": "Text Recognition:"},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        inputs.pop("token_type_ids", None)
        generated_ids = self.model.generate(**inputs, max_new_tokens=8192)

        output_text = self.processor.decode(
            generated_ids[0][inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )

        return strip_special_tokens(output_text, self.processor.tokenizer)
