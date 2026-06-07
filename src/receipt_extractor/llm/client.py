import os
from typing import Optional

import torch
from load_dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

from receipt_extractor.models.llm_message import ChatMessage, PromptData

load_dotenv()


class LlmClient:
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or os.getenv("LLM_MODEL_PATH", "./openchat_model")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, torch_dtype=torch.float16
        ).to(self.device)
        self.model.eval()

    @property
    def system_prompt(self):
        return ChatMessage(
            role="system",
            content="Answer in French",
        )

    def generate(self, req: PromptData) -> dict:
        prompt = self.tokenizer.apply_chat_template(
            [self.system_prompt.model_dump()] + [m.model_dump() for m in req.messages],
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=req.max_new_tokens)

        input_length = inputs["input_ids"].shape[-1]
        output_text = self.tokenizer.decode(
            outputs[0][input_length:], skip_special_tokens=True
        )

        return {"response": output_text}
