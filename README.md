# cortextual

yet another context and memory system for AI agents, but we're
- hand crafted, no vibe coded slop, all code, all bytes, every char is typed out by a human and written by hand
- clankers are only used for guidance, brainstorming, and design triage, but all results are acted upon materially by a human once again
- built in Rust
- built for personal use

> **Status:** pre-alpha. Built step by step as a learning project. Nothing here is stable as of yet, but maybe will be someday

## What is this?

An embeddable library that gives an agent three things:

- **Memory:** an append-only log of what happened (episodes) plus a derived store of typed memories (semantic, episodic, procedural, reflection), each with provenance, a trust level, and bi-temporal validity.
- **Recall:** hybrid retrieval (dense vectors, BM25, entities) fused with reciprocal rank fusion, rescored by recency and importance, and diversified with MMR.
- **Context:** token-budgeted prompt assembly from pinned blocks, retrieved memories, and the recent window, with deterministic eviction.

## Design principles

- Episodes are the source of truth. Memories are a rebuildable view.
- Facts are never edited or deleted by an LLM. They are superseded.
- LLMs sit at the edges behind traits. The core is deterministic and testable offline.
- Scope (tenant, user, agent, session) is a required argument everywhere.

Full design: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Build

```shell
$ cargo test
```

## License

MIT
