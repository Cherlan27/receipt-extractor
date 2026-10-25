# llm-generation Specification

## Purpose
Provides general-purpose chat-style text generation with the locally hosted language model, independent of any specific task such as receipt classification, and defines how that model is selected.

## Requirements

### Requirement: Generic chat generation
The system SHALL expose `POST /llm/generate` accepting `{"messages": [{"role": "...", "content": "..."}], "max_new_tokens": <int, optional>}` and returning `{"response": "<generated text>"}`. The system SHALL pass only the client-supplied messages to the model and SHALL NOT inject any task-specific or language-specific system prompt.

#### Scenario: Only client messages reach the model
- **WHEN** a client posts `{"messages": [{"role": "user", "content": "Hello"}]}` to `/llm/generate`
- **THEN** the model's chat input consists of exactly that one user message, with no additional system message, and the response has the form `{"response": "..."}`

#### Scenario: Client-supplied system message is honored
- **WHEN** a client includes `{"role": "system", "content": "Answer in French"}` as the first message
- **THEN** that system message is passed to the model unchanged as the first message

#### Scenario: Token limit is respected
- **WHEN** a client sets `max_new_tokens` to 42
- **THEN** generation produces at most 42 new tokens

#### Scenario: Default token limit
- **WHEN** a client omits `max_new_tokens`
- **THEN** generation uses a default limit of 8192 new tokens

### Requirement: Configurable model selection
The local language model SHALL be selected by the `LLM_MODEL_PATH` environment variable, which may be a local filesystem path or a HuggingFace model id. When `LLM_MODEL_PATH` is not set, the system SHALL use `Qwen/Qwen3.5-9B`. The same model SHALL serve both `/llm/generate` and receipt classification.

#### Scenario: Environment variable selects the model
- **WHEN** the server starts with `LLM_MODEL_PATH=D:\llm_models\some_model`
- **THEN** the tokenizer and model are loaded from `D:\llm_models\some_model`

#### Scenario: Default model when unset
- **WHEN** the server starts without `LLM_MODEL_PATH` set
- **THEN** the tokenizer and model are loaded from `Qwen/Qwen3.5-9B`
