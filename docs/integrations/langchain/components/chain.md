# LangChain Chain

## Overview

The LangChain chain provides a bridge between the AI Agent Pattern core framework and the [LangChain](https://python.langchain.com/) framework. It implements `ChatCausalMultiTurnsChain`, which extends `BaseCausalMultiTurnsChain` to handle LangChain's message-based conversation model. The chain supports both simple chat model invocations and multi-turn tool calling, with full support for binding and swapping tools at runtime.

## Architecture

The LangChain chain sits within the integration layer, translating between the framework-agnostic `AgentMessage` type and LangChain's `BaseMessage` hierarchy (`SystemMessage`, `HumanMessage`, `AIMessage`, `ToolMessage`). It manages the LLM chain internally, rebuilding it when tools or prompts change.

The chain integrates with LangChain's core abstractions:

- **`BaseChatModel`**: The underlying chat model (e.g., `ChatOpenAI`, `ChatAnthropic`). The chain stores a reference and can be reconfigured at runtime.
- **`BaseMessage`**: LangChain's message type hierarchy. The chain builds a conversation list of these messages for each invocation.
- **`BaseTool` / `Callable`**: Both LangChain tools and plain Python functions are supported as tool implementations.

![diagram](img/chain-architecture.jpg)

## Key Concepts

### Conversation Preparation

The `_prepare_conversation` method constructs a LangChain conversation list from an `AgentMessage`. It always starts with a `SystemMessage` (from `system_prompt`) followed by a `HumanMessage` (formatted from `user_prompt_template` and the message's query). Previous turns from `message.responses` are appended in order, with each response mapped to the appropriate LangChain message type:

| Response Type | LangChain Message |
|---------------|-------------------|
| `"user"` | `HumanMessage` |
| `"assistant"` (default) | `AIMessage` |
| `"tool"` | `ToolMessage` |
| `"system"` | `SystemMessage` |

The number of previous turns included is controlled by `include_history`. A value of `-1` includes all turns; a non-negative value limits to that many recent turns.

### Multimodal Input

Media carried in `AgentMessage.query_media` reaches the model as native LangChain content parts on the current `HumanMessage`, never as a string interpolated into the prompt. The chain asks the core funnel for each modality it maps — `self._image_parts(message)`, `self._audio_parts(message)` and `self._video_parts(message)` — and converts every resolved `MediaReference` with `media_ref_to_content_block`. Image, audio and video share one mapping because LangChain's block types for them carry the same `type`/`base64`/`mime_type`/`url` fields. Document is the exception and stays unmapped: LangChain's file block requires a filename that a resolved reference does not carry.

```python
from aap_core.types import AgentMessage
from aap_langchain.chain import ChatCausalMultiTurnsChain
from langchain_google_genai import ChatGoogleGenerativeAI

chain = ChatCausalMultiTurnsChain(model=ChatGoogleGenerativeAI(model="gemini-2.5-flash"),
                                  system_prompt="You are helpful.")

message = AgentMessage(
    query="What is being said in this clip?",
    query_media=[("audio", "/data/clips/meeting.wav")],
)
result = chain.invoke(message)
```

Local paths and `data:` URIs are read and delivered as base64; `http(s)` URLs pass through as-is. Parts are grouped by modality (images, then audio, then video) rather than kept in `query_media` order, because a resolved reference carries no index back to its declaration.

```python
message = AgentMessage(
    query="What happens in this clip?",
    query_media=[("video", "/data/clips/demo.mp4")],
)
```

Video reaches the model as a single container part on Gemini adapters, which consume LangChain's `VideoContentBlock` directly — no extra configuration, and the format set is core's `SUPPORTED_MIME_TYPES["video"]` (Gemini's published list). Frame sampling is the provider's decision (Gemini samples at 1 FPS by default).

Audio means audio: the bytes go to the model as an audio part. A pipeline that wants a transcript first is a separate speech-to-text stage, not this path. Video means one video container: the bytes go to the model as a single video part, and sampling frames out of it is the provider's decision (Gemini samples at 1 FPS by default). This package never decomposes a video into image parts — a silent substitution would change what the model was told it received.

**What decides whether parts are attached.** Per modality, in priority order:

1. `media_support` entries you set explicitly (`"enabled"` / `"disabled"`), applied by the core gate before detection runs.
2. The model profile (`model.profile`) — the capability table the provider ships for the exact revision, read through `profile_to_capabilities` (`image_inputs`, `audio_inputs`, `video_inputs`, `pdf_inputs`).
3. Hugging Face model-card task tags, for models identified by repo id (needs the `hf` extra). `audio-text-to-text` affirms audio input; `text-to-speech` does not, since it generates audio rather than accepting it.
4. A transport ceiling from the adapter's request format itself (`adapter_capabilities`), applied last because the wire format is a hard limit. The Ollama chat message type has no audio or video field, so both are denied for every Ollama model regardless of what a model card claims; the ceiling is read from the installed schema, not hardcoded. For the OpenAI family the probe is langchain-core's own translator: a block type it refuses is undeliverable through that transport. Adapters that serialize `message.content` verbatim — `langchain_huggingface`'s `ChatHuggingFace` and `langchain_community`'s `ChatTongyi` — get a deny-all ceiling for image, audio and video: their modules name no block translator or media part shape anywhere, so a content-block list reaches the wire unchanged and dies at the API whatever the model behind it understands. The check reads the adapter's installed source, not a model name, and an unreadable module stays undecidable rather than denied.

Model-family names are never evidence: variants of one family differ per modality. A modality no source reports on stays undecidable and is assumed capable, so the provider's own error surfaces instead of a silent drop.

```python
# Deliver audio even though nothing can detect it, or never deliver it.
chain = ChatCausalMultiTurnsChain(model=model, system_prompt="s",
                                  media_support={"image": "enabled",
                                                 "audio": "disabled"})
```

Three limits live below this package, in LangChain's provider translators, and cannot be fixed at the block level:

- **Audio URLs are not portable.** `langchain_core`'s OpenAI translator accepts only base64 for audio and raises otherwise, while image URLs pass through. Pass a local path or a `data:` URI when targeting OpenAI-family endpoints.
- **The format token comes from the MIME subtype.** That translator derives OpenAI's `input_audio.format` from the subtype, so `audio/mpeg` becomes `"mpeg"` and is rejected by an enum accepting only `"wav"` and `"mp3"`. Rewriting the MIME string here would break the adapters that read it as a MIME type, so it is passed through unchanged.
- **No OpenAI transport delivers video.** The same translator raises on every video block, on either API, so `adapter_capabilities` denies video for the whole `langchain_openai` family before a part is ever built. A custom `base_url` does not lift this: the adapter translates every data block before the request reaches any endpoint, so pointing `ChatOpenAI` at an OpenAI-compatible video service (DashScope compatible-mode for Qwen-VL, say) fails in the adapter, not the API. Reaching those endpoints' video parts needs an adapter that emits the endpoint's own part shape. Gemini-family adapters, which have their own video handling, are unaffected.
- **Native Hugging Face adapters carry no media at all.** `ChatHuggingFace` and the community `ChatTongyi` pass message content through raw, so all three media modalities degrade to text on them (or error, under an explicit `media_support` override). For locally hosted multimodal models use an OpenAI-compatible server (`vLLM`, `TGI`) behind `ChatOpenAI(base_url=...)` — image and audio then deliver, video stays denied by the translator — or the `aap_transformers` integration, which is the transport itself.

### Tool Binding and Management

Tools are bound to the chain via the `bind_tools()` method, which accepts a sequence of `BaseTool` objects or plain `Callable` functions. The chain maintains an internal `_tool_dict` that maps tool names to their implementations:

- For `BaseTool` instances, the tool's `.name` attribute is used as the key.
- For `Callable` functions, the function's `__name__` is used as the key.

After binding, the chain calls `self._model.bind_tools()` to create a tool-aware model chain stored in `self._chain`. This allows the LLM to discover and invoke tools by name.

```python
from aap_langchain.chain import ChatCausalMultiTurnsChain
from langchain_core.tools import tool

@tool
def get_weather(location: str) -> str:
    """Get the weather at a location."""
    return f"The weather in {location} is sunny."

chain = ChatCausalMultiTurnsChain(
    model=ChatOpenAI(model="gpt-4"),
    system_prompt="You are a helpful assistant.",
    tools=[get_weather],
)
```

### Tool Execution

When the LLM returns an `AIMessage` with `tool_calls`, the `_process_tools` method executes them:

1. For each tool call, it looks up the tool name in `_tool_dict`.
2. If the tool exists and is a `BaseTool`, it invokes `tool.invoke(tool_call["args"])`.
3. If the tool exists and is a `Callable`, it invokes `tool_func(**tool_call["args"])`.
4. The result is wrapped in a `ToolMessage` and appended to the conversation.
5. If the tool doesn't exist or an error occurs, an error `ToolMessage` is appended instead.

This loop continues until the LLM returns a response with no tool calls.

### Runtime Reconfiguration

The chain supports full runtime reconfiguration:

- **`update_prompt(system_prompt, user_prompt_template)`**: Changes both prompts and rebuilds the tool-bound chain.
- **`model` property (setter)**: Replaces the underlying model and rebuilds the chain with the currently bound tools.
- **`tools` property**: Returns the list of currently bound tools.

This makes the chain suitable for scenarios where prompts or models need to change between invocations without recreating the chain object.

## Usage

### Basic Example: Simple Chat

```python
from aap_langchain.chain import ChatCausalMultiTurnsChain
from langchain_openai import ChatOpenAI
from aap_core.types import AgentMessage


chain = ChatCausalMultiTurnsChain(
    model=ChatOpenAI(model="gpt-4"),
    system_prompt="You are a helpful assistant.",
    user_prompt_template="Answer the following: {query}",
    name="langchain-agent",
)

message = AgentMessage(query="What is machine learning?")
result = chain.invoke(message)
print(result.responses)
```

### Advanced Example: With Tools

```python
from aap_langchain.chain import ChatCausalMultiTurnsChain
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from aap_core.types import AgentMessage


@tool
def search_knowledge_base(query: str) -> str:
    """Search the knowledge base for relevant information."""
    return f"Results for: {query}"


chain = ChatCausalMultiTurnsChain(
    model=ChatOpenAI(model="gpt-4"),
    system_prompt="You are a research assistant. Use tools to find information.",
    tools=[search_knowledge_base],
    name="research-assistant",
    max_turns=10,
)

message = AgentMessage(query="Tell me about transformer architectures")
result = chain.invoke(message)
```

### Runtime Reconfiguration

```python
# Update prompts at runtime
chain.update_prompt(
    system_prompt="You are a coding assistant.",
    user_prompt_template="Solve this problem: {query}",
)

# Swap the model
from langchain_anthropic import ChatAnthropic
chain.model = ChatAnthropic(model="claude-3-sonnet-20240229")
```

## Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `BaseChatModel` | *(required)* | The LangChain chat model to use (e.g., `ChatOpenAI`, `ChatAnthropic`). |
| `system_prompt` | `str` | *(required)* | The system prompt that sets the assistant's behavior. |
| `user_prompt_template` | `str` | `"{query}"` | Template for formatting the user's query. Must contain `{query}`. |
| `tools` | `Sequence[Callable \| BaseTool]` | `[]` | Tools available to the model. Can be `BaseTool` objects or plain functions. |
| `tool_choice` | `str \| None` | `None` | LangChain's `tool_choice` parameter for `bind_tools`. Controls how the model selects tools. |
| `name` | `str` | — | The name of the chain, used as the source name in responses. |
| `max_turns` | `int` | `50` | Maximum number of tool-call turns before the loop terminates. |
| `include_history` | `int` | — | Number of previous turns to include. `-1` means all. |
| `store_immediate_steps` | `bool` | `False` | Whether to store all intermediate steps or only the final response. |
| `media_support` | `Dict[ContentType, "auto" \| "enabled" \| "disabled"]` | `{}` | Per-modality policy for attaching `query_media` as native content parts. Unlisted modalities default to `"auto"` and follow capability detection. |

## API Reference

::: aap_langchain.ChatCausalMultiTurnsChain
    options:
        show_root_heading: true
        show_signature: true

## See Also

- [Core Chain](../../../core/components/chain.md) — The base chain concept and `BaseCausalMultiTurnsChain` interface
- [LangChain Overview](../overview.md) — LangChain integration overview
- [LangChain Documentation](https://python.langchain.com) — Official LangChain documentation
