from typing import Optional, Union

from pydantic import BaseModel


class Word(BaseModel):
    french: str
    tag: Union[list, Optional[str]]


class ChatMessage(BaseModel):
    role: str
    content: str


class PromptData(BaseModel):
    messages: list[ChatMessage]
    max_new_tokens: int = 1000


class TopicRequest(BaseModel):
    topic: str
