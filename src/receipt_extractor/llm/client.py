import os
from typing import Optional

import torch
from load_dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

from receipt_extractor.models.llm_message import PromptData

load_dotenv()

DEFAULT_MODEL_PATH = "Qwen/Qwen3.5-9B"


class LlmClient:
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or os.getenv("LLM_MODEL_PATH", DEFAULT_MODEL_PATH)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.model.eval()

    def chat(
        self,
        messages: list[dict],
        max_new_tokens: int,
        do_sample: bool = False,
    ) -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=do_sample
            )

        input_length = inputs["input_ids"].shape[-1]
        return self.tokenizer.decode(
            outputs[0][input_length:], skip_special_tokens=True
        )

    def generate(self, req: PromptData) -> dict:
        response = self.chat([m.model_dump() for m in req.messages], req.max_new_tokens)
        return {"response": response}
