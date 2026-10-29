# Core Package

## Overview

The `aap_core` package is the foundational framework for the AI Agent Pattern project. It defines the abstractions and building blocks for constructing multi-agent systems: `Agent` as the fundamental execution unit, `Chain` for multi-turn reasoning loops, and `Orchestration` patterns for coordinating multiple agents. The package also provides supporting infrastructure including prompt augmentation, data retrieval, deduplication, guardrails, and policy-based reinforcement learning for prompt optimization.

## Architecture

The core package sits at the base of the architecture, providing framework-agnostic abstractions that all integration packages (`aap_langchain`, `aap_dspy`, `aap_llamaindex`, `aap_transformers`) build upon. It defines the shared type system (`AgentMessage`, `TokenUsage`, `BaseLLMChain`) and the `BaseAgent` / `BaseCausalMultiTurnsChain` interfaces that every integration must implement.

## Components

The core package provides the foundation for all other integrations:

- [Agent](components/agent.md) — The fundamental agent unit
- [Chain](components/chain.md) — Chaining agents together
- [Orchestration](components/orchestration.md) — Agent coordination patterns
- [Prompt Augmenter](components/prompt_augmenter.md) — Prompt enhancement
- [Retriever](components/retriever.md) — External data retrieval
- [Types](components/types.md) — Shared type definitions

The following modules have no dedicated page yet; see their source and the [API Reference](../api-reference.md):

- [Dedup](https://github.com/quanghona/aap_core/blob/main/src/aap_core/dedup.py) — Deduplication utilities (Bloom, MinHash, SimHash, Suffix Array)
- [Guardrail](https://github.com/quanghona/aap_core/blob/main/src/aap_core/guardrail.py) — Input/output safety constraints
- [Policy](https://github.com/quanghona/aap_core/blob/main/src/aap_core/policy.py) — RL policy models
- [Policy Trainer](https://github.com/quanghona/aap_core/blob/main/src/aap_core/policy_trainer.py) — Replay buffers, exploration modules, and trainers (PPO, DPO, GRPO, Reinforce++)
- [Utils](https://github.com/quanghona/aap_core/blob/main/src/aap_core/utils.py) — Utility functions

## API Reference

::: aap_core
    options:
        show_root_heading: true
        show_signature: true

## See Also

- [Integrations](../integrations/langchain/overview.md) — Framework-specific packages built on core
- [Example Notebooks](https://github.com/quanghona/aap_example/tree/main/notebook/langchain/) — Jupyter notebooks with usage examples
