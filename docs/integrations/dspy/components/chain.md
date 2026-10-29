# DSPy Chain

## Overview

The DSPy chain provides a bridge between the AI Agent Pattern core framework and the [DSPy](https://dspy.ai/) programming framework. It implements `ChatCausalMultiTurnsChain`, which extends `BaseCausalMultiTurnsChain` to handle DSPy's signature-based programming model. The chain supports both fully-managed tool calling (e.g., `dspy.ReAct`) and manual tool handling patterns, making it flexible for a wide range of DSPy workflows.

## Architecture

The DSPy chain sits within the integration layer, translating between the framework-agnostic `AgentMessage` type and DSPy's `dspy.Signature` and `dspy.Prediction` types. It uses an adapter pattern (`BaseSignatureAdapter`) to manage the bidirectional conversion between these representations.

The chain integrates with DSPy's core abstractions:

- **`dspy.Signature`**: Defines the input and output fields for a DSPy program. The chain uses signatures to structure the conversation history and model inputs.
- **`dspy.Prediction`**: The output type from DSPy predictors. The chain extracts responses, tool calls, and token usage from predictions.
- **`dspy.Module`**: The predictor that the chain invokes. This can be any DSPy module, from simple predictors to complex multi-step programs.

## Key Concepts

### Signature Adapter Pattern

The `BaseSignatureAdapter` class is the core abstraction that enables communication between the AI Agent Pattern framework and DSPy. It handles two conversions:

1. **`msg2sig`**: Converts `AgentMessage` objects into a list of `dspy.Signature` objects that represent the conversation history. This includes both static prefill values (set via `with_prefill`) and dynamic values from the message's context.

2. **`sig2msg`**: Converts DSPy `Prediction` outputs back into `AgentResponse` objects. The adapter determines whether a response is an assistant message or a tool message based on the presence of `OutputField` and `ToolCalls` in the signature.

The adapter supports a **prefill dictionary** — a set of static key-value pairs that are injected into every signature. This is useful for fields that don't exist in `AgentMessage` but are required by the DSPy signature, such as system instructions or retrieved context.

```python
adapter = MySignatureAdapter.with_prefill({
    "system_instruction": "You are a helpful assistant.",
    "retrieved_context": "Some static context...",
})
```

### Tool Calling Approaches

DSPy supports two distinct patterns for tool calling, and the chain handles both:

#### Approach 1: Fully Managed (e.g., `dspy.ReAct`)

When using DSPy modules that manage tool calling internally (like `dspy.ReAct`), the module's signature does **not** include a `dspy.ToolCalls` field. All tool execution happens inside the DSPy module, and the chain only receives the final output. In this case, `_process_tools` is never called.

```python
import dspy

# dspy.ReAct manages tools internally
predictor = dspy.ReAct("query -> answer", tools=[my_tool])
chain = ChatCausalMultiTurnsChain(
    predictor=predictor,
    adapter=MyAdapter(),
)
```

#### Approach 2: Manual Tool Handling

When the DSPy signature includes a `dspy.ToolCalls` output field, the chain takes over tool execution. After `_generate_response` produces a prediction with tool calls, the chain's `_process_tools` method executes each tool and appends the results back to the conversation, continuing the loop until no more tool calls are needed.

```python
# Signature with explicit ToolCalls field
class MySignature(dspy.Signature):
    query: dspy.InputField
    tool_calls: dspy.ToolCalls
    answer: dspy.OutputField

chain = ChatCausalMultiTurnsChain(
    signature=MySignature,
    predictor=my_predictor,
    adapter=MyAdapter(),
)
```

The chain automatically detects the presence of a `dspy.ToolCalls` field in the signature's output fields during initialization and sets the `_tool_calls_field` attribute accordingly.

### Language Model Context

The chain supports setting a custom `dspy.LM` via the `with_lm()` method. This allows you to swap the language model at runtime without recreating the chain:

```python
chain = ChatCausalMultiTurnsChain(
    predictor=predictor,
    adapter=MyAdapter(),
)
chain = chain.with_lm(dspy.LM("anthropic/claude-3-opus-20240229"))
```

When an LM is set, the chain uses `dspy.context` to apply it during prediction, with token usage tracking enabled.

### History Field Management

The chain automatically detects the `dspy.History` field in the signature during initialization. When present, it converts the history from a dictionary (in the signature) to a `dspy.History` object (required by the DSPy library) before passing it to the predictor.

### Multimodal Input

dspy delivers media through **signature fields typed with the native `dspy.Image` / `dspy.Audio` types** (and the custom `aap_dspy.utils.Video` type for video), not through message content parts. An adapter that supports media input declares the field on its signature (as `Optional[...]` with `default=None`, so text-only turns carry no media) and maps `AgentMessage.query_media` through the chain's gated funnel — `self.chain._image_parts(message)` / `self.chain._audio_parts(message)` / `self.chain._video_parts(message)` — converting each resolved `MediaReference` with `media_ref_to_image` / `media_ref_to_audio` / `media_ref_to_video`. Document is not mapped.

```python
import dspy
from typing import Optional, List
from aap_core.types import AgentMessage
from aap_dspy.chain import BaseSignatureAdapter, ChatCausalMultiTurnsChain
from aap_dspy.utils import media_ref_to_audio

class AudioSig(dspy.Signature):
    """Answer the question about the audio."""
    audio: Optional[dspy.Audio] = dspy.InputField(default=None)
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

class AudioAdapter(BaseSignatureAdapter):
    def __init__(self):
        self.chain = None  # set right after chain construction

    def msg2sig(self, message: AgentMessage) -> List[AudioSig]:
        audios = self.chain._audio_parts(message)  # gate applied here
        return [AudioSig(
            question=message.query,
            audio=(media_ref_to_audio(audios[0]) if audios else None),
            answer="",
        )]

    def sig2msg(self, signatures, name):
        return [(name, sig.answer) for sig in signatures]

chain = ChatCausalMultiTurnsChain(signature=AudioSig,
                                  predictor=dspy.Predict(AudioSig),
                                  adapter=AudioAdapter())
chain.adapter.chain = chain

message = AgentMessage(
    query="What is being said in this clip?",
    query_media=[("audio", "/data/clips/meeting.wav")],
)
result = chain(message)
```

Video follows the same pattern through `chain._video_parts` and `media_ref_to_video`:

```python
from aap_dspy.utils import Video, media_ref_to_video

class VideoSig(dspy.Signature):
    """Answer the question about the video."""
    video: Optional[Video] = dspy.InputField(default=None)
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

class VideoAdapter(BaseSignatureAdapter):
    def __init__(self):
        self.chain = None

    def msg2sig(self, message: AgentMessage) -> List[VideoSig]:
        videos = self.chain._video_parts(message)  # gate applied here
        return [VideoSig(
            question=message.query,
            video=(media_ref_to_video(videos[0]) if videos else None),
            answer="",
        )]

    def sig2msg(self, signatures, name):
        return [(name, sig.answer) for sig in signatures]

chain = ChatCausalMultiTurnsChain(signature=VideoSig,
                                  predictor=dspy.Predict(VideoSig),
                                  adapter=VideoAdapter())
chain.adapter.chain = chain

message = AgentMessage(
    query="What happens in this video?",
    query_media=[("video", "https://example.com/clip.mp4")],
)
result = chain(message)
```

Local paths and `data:` URIs are read and delivered as base64 by the core resolver. `http(s)` URLs pass through as-is for images and video; for audio they are **downloaded at mapping time** (`dspy.Audio` carries no URL field — its wire part is inline base64 plus a format token), so prefer local paths or `data:` URIs when fetch latency matters.

Audio means audio, video means video: the bytes go to the model as an audio or video part. A pipeline that wants a transcript first is a separate speech-to-text stage, not this path. Video is delivered as a container, never sampled into frames — frame extraction is the provider's decision (Gemini samples at 1 FPS, DashScope exposes `fps`/`max_frames`).

`Video` is a custom `dspy.Type` subclass (dspy has no native video type through 3.4.1 — the upstream `dspy.Video` PRs were closed unmerged). It emits a `file` content part, which litellm's provider handlers already dispatch for non-image binary media: Gemini converts it to `inline_data` (base64) or `file_data` (URL) with the correct MIME type; providers without video input reject it at the API level.

**What decides whether media is attached.** Per modality, in priority order:

1. `media_support` entries you set explicitly (`"enabled"` / `"disabled"`), applied by the core gate before detection runs.
2. The litellm model-info table (`litellm.get_model_info`), read through `model_info_to_capabilities` — `supports_vision`, `supports_audio_input`, `supports_pdf_input`. dspy has no capability metadata of its own; it delegates model handling to litellm, so this is the authoritative source. litellm has no video-input flag, so video is never reported from this source and stays undecidable in auto mode.
3. Hugging Face model-card task tags, for models identified by repo id (needs the `hf` extra).

Model-family names are never evidence: variants of one family differ per modality. A modality no source reports on stays undecidable and is assumed capable, so the provider's own error surfaces instead of a silent drop. Detection results are cached per LM object.

```python
# Deliver audio even though nothing can detect it, or never deliver it.
chain = ChatCausalMultiTurnsChain(
    signature=AudioSig, predictor=dspy.Predict(AudioSig), adapter=AudioAdapter(),
    media_support={"image": "enabled", "audio": "disabled"},
)
```

Two dspy-specific mechanics worth knowing:

- **Round-trip rehydration.** `Signature.model_dump()` serializes `Image`/`Audio`/`Video` fields to custom-type marker strings, and feeding those back into a signature or predictor degrades the media to text (or fails validation outright for `Audio`). The chain rehydrates media-typed fields around the round-trip in `_generate_response`; adapters need no extra handling.
- **The format token comes from the MIME subtype** (with any `x-` prefix stripped, matching dspy's own convention). Whether a provider accepts it is below this layer: OpenAI chat endpoints take only `wav` and `mp3`, so other formats fail at the provider rather than here. Rewriting MIME strings would break the adapters that read them, so they pass through unchanged.

## Usage

### Basic Example: Simple Predictor

```python
import dspy
from aap_dspy.chain import ChatCausalMultiTurnsChain
from aap_dspy.utils import MySignatureAdapter
from aap_core.types import AgentMessage


class MyAdapter(BaseSignatureAdapter):
    def msg2sig(self, message: AgentMessage) -> list[dspy.Signature]:
        # Convert AgentMessage to DSPy signatures
        ...

    def sig2msg(self, signatures: list[dspy.Signature], name: str) -> list[AgentResponse]:
        # Convert DSPy predictions back to AgentResponse
        ...


# Define a simple signature
class QASignature(dspy.Signature):
    """Answer the user's question."""
    query: dspy.InputField
    answer: dspy.OutputField

# Create a predictor
predictor = dspy.Predict(QASignature)

# Create the chain
chain = ChatCausalMultiTurnsChain(
    signature=QASignature,
    predictor=predictor,
    adapter=MyAdapter(),
    name="dspy-agent",
)

# Invoke with a message
message = AgentMessage(query="What is the capital of France?")
result = chain.invoke(message)
```

### Advanced Example: Multi-Turn with Tools

```python
import dspy
from aap_dspy.chain import ChatCausalMultiTurnsChain


# Define a signature with explicit tool calls
class ToolSignature(dspy.Signature):
    """Use tools to answer the user's question."""
    query: dspy.InputField
    tool_calls: dspy.ToolCalls
    answer: dspy.OutputField


# Create a predictor that uses the signature
predictor = MyCustomModule()  # Any dspy.Module

# Create the chain with manual tool handling
chain = ChatCausalMultiTurnsChain(
    signature=ToolSignature,
    predictor=predictor,
    adapter=MyAdapter(),
    name="dspy-tool-agent",
    max_turns=10,
)

# Set a specific language model
chain = chain.with_lm(dspy.LM("google/gemini-2.0-flash"))

# Invoke — the chain will loop through tool calls automatically
message = AgentMessage(query="What's the weather in Tokyo?")
result = chain.invoke(message)
```

### Using Prefill Values

```python
# Create an adapter with static prefill values
adapter = MyAdapter.with_prefill({
    "system_instruction": "You are a helpful chemistry tutor.",
    "domain": "chemistry",
})

chain = ChatCausalMultiTurnsChain(
    signature=MySignature,
    predictor=predictor,
    adapter=adapter,
)

# Add or remove prefill values at runtime
adapter.add_prefill("retrieved_context", "Some retrieved document...")
adapter.remove_prefill("domain")
```

## Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `predictor` | `dspy.Module` | *(required)* | The DSPy predictor module to invoke. Can be a `dspy.Predict`, `dspy.ReAct`, or any custom `dspy.Module`. |
| `adapter` | `BaseSignatureAdapter` | *(required)* | The signature adapter that converts between `AgentMessage` and `dspy.Signature`. |
| `signature` | `str \| type[dspy.Signature]` | *(required)* | The DSPy signature defining input/output fields. Can be a string or a signature class. |
| `name` | `str` | — | The name of the chain, used as the source name in responses. |
| `max_turns` | `int` | `50` | Maximum number of tool-call turns before the loop terminates. |
| `include_history` | `int` | — | Number of previous turns to include in the conversation. `-1` means all. |
| `store_immediate_steps` | `bool` | `False` | Whether to store all intermediate steps or only the final response. |
| `media_support` | `Dict[ContentType, "auto" \| "enabled" \| "disabled"]` | `{}` | Per-modality policy for delivering media from `query_media`; unlisted modalities use capability detection. |
| `with_lm()` | `dspy.LM \| None` | `None` | Sets a custom language model via the chain's fluent API. |

## API Reference

::: aap_dspy.ChatCausalMultiTurnsChain
    options:
        show_root_heading: true
        show_signature: true

::: aap_dspy.BaseSignatureAdapter
    options:
        show_root_heading: true
        show_signature: true

## See Also

- [Core Chain](../../../core/components/chain.md) — The base chain concept and `BaseCausalMultiTurnsChain` interface
- [DSPy Overview](../overview.md) — DSPy integration overview
- [DSPy Documentation](https://dspy.ai) — Official DSPy framework documentation
