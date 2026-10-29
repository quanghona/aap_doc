# AI Agent Pattern (AAP)

**A modular framework for orchestrating multi-agent systems across popular LLM libraries.**

AI Agent Pattern provides a unified, framework-agnostic layer for building and coordinating multi-agent workflows. While LLM interaction is delegated to dedicated libraries like LangChain, LlamaIndex, DSPy, and Hugging Face Transformers, AAP focuses on the orchestration logic — how agents communicate, chain reasoning steps, and collaborate to solve complex tasks.

## What It Solves

Building multi-agent systems involves recurring patterns: chaining agents for multi-turn reasoning, coordinating parallel agent teams, implementing self-reflection loops, and managing retrieval-augmented workflows. AAP extracts these patterns into reusable abstractions so you can focus on agent design rather than boilerplate orchestration code.

### Key Features

- **Framework-agnostic core** — Shared abstractions (`Agent`, `Chain`, `Orchestration`) that work across all supported LLM libraries
- **4 integration packages** — Native adapters for LangChain, LlamaIndex, DSPy, and Hugging Face Transformers
- **Pre-built orchestration patterns** — Coordinator, debate, voting, parallel, sequential, self-reflection, cross-reflection, and iterative refinement
- **Prompt augmentation & retrieval** — Built-in utilities for prompt enhancement and external data retrieval
- **Jupyter-ready examples** — Notebook-based examples for every integration and pattern

## Architecture

```
┌─────────────────────────────────────────────┐
│           Your Agent Application             │
└─────────────────────────────────────────────┘
                  │
    ┌─────────────┼─────────────┬─────────────┬─────────────┐
    ▼             ▼             ▼             ▼             ▼
┌────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│  Core  │  │ LangChain│  │LlamaIndex│  │  DSPy    │  │Transformers│
└────┬───┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘
     │           │              │             │              │
     └───────────┼──────────────┼─────────────┼──────────────┘
                 ▼              ▼             ▼              ▼
        ┌────────────────────────────────────────────────────────┐
        │  LLM Providers                                         │
        │  (OpenAI, Ollama, HF, Anthropic, etc.)                 │
        └────────────────────────────────────────────────────────┘
```

The core package defines shared types (`AgentMessage`, `TokenUsage`, `BaseLLMChain`) and base interfaces. Every integration package implements these interfaces using its respective LLM library.

## Getting Started

### 1. Install a Package

Choose the integration that matches your preferred LLM library:

#### LangChain integration
```bash
pip install aap_langchain
```

#### LlamaIndex integration
```bash
pip install aap_llamaindex
```

#### Hugging Face Transformers integration
```bash
pip install aap_transformers
```

#### DSPy integration
```bash
pip install aap_dspy
```

### 2. Define Your Agent

Each integration provides an `Agent` class backed by its respective library. Define the agent's role, LLM chain, and any retrievers or prompt augmenters.

### 3. Chain and Orchestrate

Connect agents into chains for multi-turn reasoning, or use orchestration patterns (coordinator, debate, voting, etc.) for multi-agent collaboration.

### 4. Run

Execute the flow and get structured results with token usage tracking.

## Example Notebooks

Ready-to-run Jupyter notebooks demonstrating every pattern across all integrations:

| Pattern | LangChain | LlamaIndex | DSPy | Transformers |
|---------|-----------|------------|------|--------------|
| Sequential | [→](https://github.com/quanghona/aap_example/blob/main/notebook/langchain/sequential.ipynb) | [→](https://github.com/quanghona/aap_example/blob/main/notebook/llamaindex/sequential.ipynb) | [→](https://github.com/quanghona/aap_example/blob/main/notebook/dspy/sequential.ipynb) | [→](https://github.com/quanghona/aap_example/blob/main/notebook/transformers/sequential.ipynb) |
| Coordinator | [→](https://github.com/quanghona/aap_example/blob/main/notebook/langchain/coordinator.ipynb) | [→](https://github.com/quanghona/aap_example/blob/main/notebook/llamaindex/coordinator.ipynb) | [→](https://github.com/quanghona/aap_example/blob/main/notebook/dspy/coordinator.ipynb) | [→](https://github.com/quanghona/aap_example/blob/main/notebook/transformers/coordinator.ipynb) |
| Self-Reflection | [→](https://github.com/quanghona/aap_example/blob/main/notebook/langchain/self_reflection.ipynb) | [→](https://github.com/quanghona/aap_example/blob/main/notebook/llamaindex/self_reflection.ipynb) | [→](https://github.com/quanghona/aap_example/blob/main/notebook/dspy/self_reflection.ipynb) | [→](https://github.com/quanghona/aap_example/blob/main/notebook/transformers/self_reflection.ipynb) |
| Parallel | [→](https://github.com/quanghona/aap_example/blob/main/notebook/langchain/parallel.ipynb) | [→](https://github.com/quanghona/aap_example/blob/main/notebook/llamaindex/parallel.ipynb) | [→](https://github.com/quanghona/aap_example/blob/main/notebook/dspy/parallel.ipynb) | [→](https://github.com/quanghona/aap_example/blob/main/notebook/transformers/parallel.ipynb) |
| Debate | [→](https://github.com/quanghona/aap_example/blob/main/notebook/langchain/debate.ipynb) | [→](https://github.com/quanghona/aap_example/blob/main/notebook/llamaindex/debate.ipynb) | [→](https://github.com/quanghona/aap_example/blob/main/notebook/dspy/debate.ipynb) | [→](https://github.com/quanghona/aap_example/blob/main/notebook/transformers/debate.ipynb) |
| Voting | [→](https://github.com/quanghona/aap_example/blob/main/notebook/langchain/voting.ipynb) | [→](https://github.com/quanghona/aap_example/blob/main/notebook/llamaindex/voting.ipynb) | [→](https://github.com/quanghona/aap_example/blob/main/notebook/dspy/voting.ipynb) | [→](https://github.com/quanghona/aap_example/blob/main/notebook/transformers/voting.ipynb) |
| Retriever | [→](https://github.com/quanghona/aap_example/blob/main/notebook/langchain/retriever.ipynb) | [→](https://github.com/quanghona/aap_example/blob/main/notebook/llamaindex/retriever.ipynb) | [→](https://github.com/quanghona/aap_example/blob/main/notebook/dspy/retriever.ipynb) | — |
| Iterative Refinement | [→](https://github.com/quanghona/aap_example/blob/main/notebook/langchain/iterative_refinement.ipynb) | [→](https://github.com/quanghona/aap_example/blob/main/notebook/llamaindex/iterative_refinement.ipynb) | [→](https://github.com/quanghona/aap_example/blob/main/notebook/dspy/iterative_refinement.ipynb) | — |
| Cross-Reflection | [→](https://github.com/quanghona/aap_example/blob/main/notebook/langchain/cross_reflection.ipynb) | [→](https://github.com/quanghona/aap_example/blob/main/notebook/llamaindex/cross_reflection.ipynb) | [→](https://github.com/quanghona/aap_example/blob/main/notebook/dspy/cross_reflection.ipynb) | [→](https://github.com/quanghona/aap_example/blob/main/notebook/transformers/cross_reflection.ipynb) |
| Simple Loop | [→](https://github.com/quanghona/aap_example/blob/main/notebook/langchain/simple_loop.ipynb) | [→](https://github.com/quanghona/aap_example/blob/main/notebook/llamaindex/simple_loop.ipynb) | [→](https://github.com/quanghona/aap_example/blob/main/notebook/dspy/simple_loop.ipynb) | [→](https://github.com/quanghona/aap_example/blob/main/notebook/transformers/simple_loop.ipynb) |

## Documentation

- [Core Package](core/overview.md) — Foundational abstractions and types
- [Integrations](integrations/langchain/overview.md) — Framework-specific packages
- [API Reference](api-reference.md) — Auto-generated API documentation

## Status

This project is in active development. Components are subject to change between versions.
