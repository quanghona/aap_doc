# Transformers Chain

## Overview

The Transformers chain provides a bridge between the AI Agent Pattern core framework and the [Hugging Face Transformers](https://huggingface.co/transformers/) library. It implements `ChatCausalMultiTurnsChain` to run local causal language models with built-in tool calling support. The chain uses the model's tokenizer chat template for conversation formatting and extracts tool calls from model output using regex pattern matching on the `<tool_call>` format.

## Architecture

The Transformers chain sits within the integration layer, translating between the framework-agnostic `AgentMessage` type and a simple dictionary-based message format (`TransformersChainMessage`). It manages both the model and tokenizer internally, loading them from a model identifier or a pre-loaded `(tokenizer, model)` tuple.

The chain integrates with Transformers' core abstractions:

- **`AutoModelForCausalLM`**: The Hugging Face causal language model. The chain loads and manages the model on the specified device (CPU or CUDA).
- **`AutoTokenizer`**: The tokenizer that converts conversations into model inputs. It uses `apply_chat_template()` with tool support for proper message formatting.
- **`TransformersChainMessage`**: A `TypedDict` with `role` and `content` keys that represents each message in the conversation.

![diagram](img/chain-architecture.jpg)

## Key Concepts

### Conversation Preparation

The `_prepare_conversation` method builds a list of `TransformersChainMessage` dictionaries from an `AgentMessage`. The conversation starts with a `system` message followed by a `user` message. Previous turns are appended with role mapping:

| Response Type | Transformers Role |
|---------------|-------------------|
| `"user"` | `"user"` |
| `"assistant"` (default) | `"assistant"` |
| `"tool"` | `"tool"` |
| `"system"` | `"system"` |

The `include_history` parameter controls how many previous turns are included.

### Model Input Generation

The chain uses the tokenizer's `apply_chat_template()` method to convert the conversation into model inputs:

```python
inputs = self._tokenizer.apply_chat_template(
    conversation,
    tools=self.tools,          # Tools passed to the template for proper formatting
    add_generation_prompt=True, # Adds the assistant prompt token
    return_dict=True,
    return_tensors="pt",
)
```

The `tools` parameter ensures the chat template includes proper tool definitions in the formatted prompt, enabling the model to understand available tools.

### Token Usage Tracking

The chain tracks token usage precisely:

- **Input tokens**: `inputs.input_ids.shape[-1]` — the length of the tokenized conversation.
- **Output tokens**: The length of the generated tokens beyond the input — `output[:, input_tokens:].shape[-1]`.
- **Total tokens**: Sum of input and output tokens.

This information is stored in a `TokenUsage` object and attached to the resulting `AgentMessage`.

### Tool Call Extraction

The Transformers chain uses regex-based extraction to find tool calls in the model's output. It looks for the `<tool_call>(?s:.*?)<\/tool_call>` pattern in the generated text. Each match is stripped of its delimiters and parsed as JSON to extract the tool name and arguments.

```python
# From the chain source
tool_calls = ChatCausalMultiTurnsChain.extract_json(response)
# Returns: [{"name": "get_weather", "arguments": {"location": "Tokyo"}}]
```

If the model output contains no `<tool_call>` markers, the chain treats the entire output as a plain text response and terminates the loop.

### Tool Execution

When tool calls are extracted, `_process_tools` executes each one:

1. Parses the tool call JSON to get the tool name and arguments.
2. Looks up the tool name in `_tool_dict`.
3. If found, invokes `tool_func(**arguments)` and appends a `tool` message with the result.
4. If not found or an error occurs, appends a `tool` message with an error description.

### Device Management

The chain requires a `device` parameter (`"cpu"` or `"cuda"`) that determines where the model runs. The model is loaded once during initialization and stays on the specified device throughout the chain's lifetime. You can also replace the model at runtime via the `model` property setter, which reloads both the tokenizer and model from the new path or tuple.

### Multimodal Input

Media carried in `AgentMessage.query_media` reaches the model as native Hugging Face chat-template content parts (`{"type": "image", ...}`, `{"type": "audio", ...}`, `{"type": "video", ...}`) on the current user message's part list, never as a string interpolated into the prompt. The chain asks the core funnel for each modality it maps — `self._image_parts(message)`, `self._audio_parts(message)` and `self._video_parts(message)` — and converts every resolved `MediaReference` with `media_ref_to_image_part`, `media_ref_to_audio_part` or `media_ref_to_video_part`. Document is not mapped yet.

```python
from aap_core.types import AgentMessage
from aap_transformers.chain import ChatCausalMultiTurnsChain

chain = ChatCausalMultiTurnsChain(
    model="Qwen/Qwen2-Audio-7B-Instruct",
    system_prompt="You are helpful.",
    user_prompt_template="{query}",
    device="cuda",
)

message = AgentMessage(
    query="What is being said in this clip?",
    query_media=[("audio", "/data/clips/meeting.wav")],
)
result = chain.invoke(message)
```

```python
from aap_core.types import AgentMessage
from aap_transformers.chain import ChatCausalMultiTurnsChain

chain = ChatCausalMultiTurnsChain(
    model="Qwen/Qwen2.5-VL-7B-Instruct",
    system_prompt="You are helpful.",
    user_prompt_template="{query}",
    device="cuda",
)

message = AgentMessage(
    query="What happens in this video?",
    query_media=[("video", "https://example.com/clip.mp4")],
)
result = chain.invoke(message)
```

Local paths and `data:` URIs are resolved to base64 by the core and decoded at our layer, at prepare time: images become PIL objects, audio becomes a **mono float32 numpy array resampled to the model's feature-extractor sampling rate** (16 kHz when no extractor is loaded), and video becomes a **`uint8` RGB array of shape `(frames, H, W, 3)` sampled uniformly to the model's frame budget** (8 frames when no video processor is loaded). This matters because transformers' own media loaders (`torchcodec`/`librosa`) are optional extras — an ndarray bypasses them entirely inside `apply_chat_template`, so decoding and resampling must happen before the model call. Image and audio `http(s)` URLs pass through untouched and are fetched by the processor at tokenization time, which does require the optional backend.

Audio means audio: the bytes go to the model as an audio part. A pipeline that wants a transcript first is a separate speech-to-text stage, not this path.

**Video breaks the URL passthrough, deliberately.** transformers decodes a string video with `torchcodec` (its default) or falls back to `torchvision`, which needs PyAV — neither ships guaranteed wheels on every platform, so a passthrough part would raise deep inside the processor. `media_ref_to_video_part` therefore decodes every payload itself with OpenCV (whose wheels bundle FFmpeg), including `http(s)` URLs, which are fetched with `httpx` first. Frame sampling happens here too, because an array input skips the processor's own sampling: the budget is read from the loaded video processor's `num_frames` so an array matches what a string would have been sampled to. Loading every frame of a real clip would exhaust memory before the model saw one, so the read is capped at `num_frames × stride`.

Video means one video, never N images: a clip is delivered as a single video part. Turning it into image frames would change what the model was told it received.

**Codec coverage without extras.** The built-in audio decoder (soundfile) handles `wav`, `mp3`, `flac`, `ogg` (Vorbis) and `aiff` base64 refs. `mp4`/`m4a`, `aac`, `opus`, `webm`, `amr` and `x-ms-wma` audio refs — and every audio URL ref — need `pip install "transformers[audio]"` (librosa) or torchcodec. Video decoding is OpenCV's FFmpeg build, so the supported containers are whatever that build includes; the formats core accepts for video are `mp4`, `mpeg`, `quicktime`, `avi`, `webm`, `flv`, `wmv` and `3gpp`. Parts are grouped by modality (images, then audio, then video) rather than kept in `query_media` order, because a resolved reference carries no index back to its declaration.

**What decides whether parts are attached.** Per modality, in priority order:

1. `media_support` entries you set explicitly (`"enabled"` / `"disabled"`), applied by the core gate before detection runs.
2. The processor actually loaded for this exact revision: an `image_processor` affirms image, a `feature_extractor` affirms audio, and a `video_processor` affirms video — decisively in both directions, since these are the pipelines that will run. Sub-processor attributes are populated only when the model's processor declares them, so an image-only processor reports video `False` rather than missing. Audio-only chat models (Qwen2-Audio, Voxtral, Phi4-multimodal) load through `AutoModelForMultimodalLM`; hybrid vision+audio models (Gemma-3n, Qwen2.5-Omni) keep the `AutoModelForImageTextToText` path, whose mapping already covers them.
3. Hugging Face model-card task tags, for models identified by repo id when no processor was loadable. `audio-text-to-text` affirms audio input and `video-text-to-text` affirms video; `text-to-speech` affirms neither, since it generates media rather than accepting it.

There is no transport ceiling here, unlike the API-based integrations: transformers *is* the transport, and the chat template carries whatever parts the loaded processor's pipeline can expand. For video that pipeline is fed already-decoded frames, so the one environmental limit is OpenCV's codec set, not a missing wire field. Model-family names are never evidence — variants of one family differ per modality. A modality no source reports on stays undecidable and is assumed capable, so the model's own error surfaces instead of a silent drop.

```python
# Deliver audio even though nothing can detect it, or never deliver it.
chain = ChatCausalMultiTurnsChain(
    model=model_tuple, system_prompt="s", device="cuda",
    media_support={"image": "enabled", "audio": "disabled"},
)
```

## Usage

### Basic Example: Simple Chat

```python
from aap_transformers.chain import ChatCausalMultiTurnsChain
from aap_core.types import AgentMessage


chain = ChatCausalMultiTurnsChain(
    model="meta-llama/Llama-3.1-8B-Instruct",
    system_prompt="You are a helpful assistant.",
    user_prompt_template="Answer the following: {query}",
    device="cuda",
    name="transformers-agent",
)

message = AgentMessage(query="What is the capital of France?")
result = chain.invoke(message)
print(result.responses)
```

### Advanced Example: With Tools

```python
from aap_transformers.chain import ChatCausalMultiTurnsChain
from aap_core.types import AgentMessage


def get_weather(location: str) -> str:
    """Get the weather at a location."""
    return f"The weather in {location} is sunny, 22°C."


chain = ChatCausalMultiTurnsChain(
    model="meta-llama/Llama-3.1-8B-Instruct",
    system_prompt="You are a weather assistant. Use tools to find information.",
    tools=[get_weather],
    device="cuda",
    name="weather-assistant",
    max_turns=10,
)

message = AgentMessage(query="What's the weather in Tokyo?")
result = chain.invoke(message)
```

### Using a Pre-loaded Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from aap_transformers.chain import ChatCausalMultiTurnsChain


tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    device_map="cuda",
)

chain = ChatCausalMultiTurnsChain(
    model=(tokenizer, model),  # Pass pre-loaded tuple
    system_prompt="You are a helpful assistant.",
    device="cuda",
    name="transformers-agent",
)
```

## Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str \| os.PathLike \| Tuple[Any, Any]` | *(required)* | Model identifier string (e.g., `"meta-llama/Llama-3.1-8B-Instruct"`), path to a local model, or a `(tokenizer, model)` tuple. |
| `device` | `Literal["cpu", "cuda"]` | *(required)* | Device to run the model on. `"cuda"` for GPU, `"cpu"` for CPU. |
| `system_prompt` | `str` | *(required)* | The system prompt that sets the assistant's behavior. |
| `user_prompt_template` | `str` | *(required)* | Template for formatting the user's query. Must contain `{query}`. |
| `tools` | `Sequence[Callable]` | `[]` | Tools available to the model. Plain functions are stored by their `__name__`. |
| `name` | `str` | — | The name of the chain, used as the source name in responses. |
| `max_turns` | `int` | `50` | Maximum number of tool-call turns before the loop terminates. |
| `include_history` | `int` | — | Number of previous turns to include. `-1` means all. |
| `store_immediate_steps` | `bool` | `False` | Whether to store all intermediate steps or only the final response. |
| `media_support` | `Dict[ContentType, Literal["auto", "enabled", "disabled"]]` | `{}` | Per-modality policy for delivering `query_media` as native content parts. Unlisted modalities default to `"auto"` (detection). |

## API Reference

::: aap_transformers.ChatCausalMultiTurnsChain
    options:
        show_root_heading: true
        show_signature: true

## See Also

- [Core Chain](../../../core/components/chain.md) — The base chain concept and `BaseCausalMultiTurnsChain` interface
- [Transformers Overview](../overview.md) — Transformers integration overview
- [Hugging Face Transformers Documentation](https://huggingface.co/transformers) — Official Transformers documentation