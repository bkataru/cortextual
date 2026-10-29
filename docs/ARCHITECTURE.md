# cortextual: architecture

Status: draft. This document is the plan, the code is the truth. When they disagree, fix whichever one is wrong.

## 1. Goals and non-goals

Goals:

- Embeddable library first. A CLI and an MCP server come later and sit on top of the library.
- Local-first. One SQLite file holds all durable state.
- Deterministic core. Anything that touches an LLM, an embedding model, a decision model, or the network, sits behind a trait and has a fake/mock for tests.
- Temporal correctness. We can answer both "what was true at time T" and "what did we believe at time T".
- Bounded context. Prompt assembly respects a token budget and degrades in a predictable order.
- Safe by construction. Scope isolation, provenance, and trust live in the types.

Non-goals, for now:

- Multi-noe or hosted operations.
- Running LLM inference ourselves.
- Training or fine-tuning.

## 2. Prior art

| System | What we take |
|---|---|
| MemGPT / Letta | Pinned memory blocks, tiered storage (in-context, recall, archival), background consolidation |
| Zep / Graphiti | Episodes as immutable ground truth, bi-temporal facts, invalidate instead of delete |
| Mem0 | Single-pass add-only extraction, multi-signal retrieval (semantic, BM25, entity) |
| Generative Agents | Retrieval score from relevance, recency, and importance; reflection triggered by accumulated importance |
| Context-engineering work | Clear stale tool output first, deterministic eviction, evicted content stays recoverable |

References:

- Zep, a temporal knowledge graph architecture for agent memory: [https://arxiv.org/abs/2501.13956](https://arxiv.org/abs/2501.13956)
- Generative Agents: Interactive Simulacra of Human Behavior: [https://arxiv.org/abs/2304.03442](https://arxiv.org/abs/2304.03442)
- Letta, MemGPT architecture: [https://docs.letta.com/guides/agents/architectures/memgpt](https://docs.letta.com/guides/agents/architectures/memgpt)
- Mem0, token-efficient memory algorithm: [https://mem0.ai/blog/mem0-the-token-efficient-memory-algorithm](https://mem0.ai/blog/mem0-the-token-efficient-memory-algorithm)

## 3. Design decisions

1. Episodes are the source of truth. They are immutable and append-only. Memories are a derived view that can be rebuilt from them.
2. Writes are add-only. A changed fact does not overwrite the old one. The old one is superseded by closing its validity interval.
3. LLMs sit at the edges. Extraction and importance scoring are behind traits. Fusion, rescoring, packing, and eviction are pure functions.
4. Ports and adapters. `Embedder`, `Extractor`, `Tokenizer`, `Clock`, and `Store` are traits, and each has a deterministic mock/fake.
5. `Scope` is a required argument on every read and write, so tenant isolation and context boundaries cannot be forgotten or bypassed.
6. Retrieval is a pipeline: candidates, fusion, rescoring, diversification, budget packing.
7. Every memory carries provenance (which episodes produced it) and a trust level.
8. The core is synchronous. Async lives only at the network edges where it makes natural sense.
9. Storage is SQLite for durability and FTS5, plus an in-process vector index that we write ourselves.

## 4. High-level design

```text
observe(event) -> [Episode log] -> [Extractor] -> [Dedupe + Supersede] -> [Memory store]
                   append-only       LLM port      cosine, entity, time      SQLite + FTS5
                                                                            + vector index

recall(q, budget) -> [Candidates] -> [RRF fuse] -> [Rescore] -> [MMR] -> [Budget pack]
                      dense, BM25,                  recency,
                      entity                        importance

build_context()   -> [Pinned blocks] + [Retrieved] + [Recent window] -> [Evictor] -> prompt

background        -> decay, merge, promote episodic to semantic, reflect, re-embed
```

## 5. Data model


Two clocks on every fact:

- Valid time (world time): when the fact was true in the world. Fields `valid_from`, `valid_to`.
- Record time (system time): when we learned it and when we stopped believing it. Fields `recorded_at`, `superseded_at`.

Below are approximate sketches, the real signatures live in the code.

```rust
pub struct Scope {
    tenant: String,
    user: String,
    agent: Option<String>,
    session: Option<String>,
}

pub enum EpisodeKind {
    UserMessage,
    AssistantMessage,
    ToolCall,
    ToolResult,
    Observation,
    Document,
}

pub struct Episode {
    id: EpisodeId,              // UUIDv7, time-sortable
    scope: Scope,
    kind: EpisodeKind,
    content: String,
    occurred_at: Timestamp,    // world time
    ingested_at: Timestamp,    // system time
    content_hash: [u8, 32],    // blake3, for exact dedupe
}

pub enum MemoryKind { Semantic, Episodic, Procedural, Reflection }
pub enum Trust { System, User, Tool, External }
pub enum Status { Active, Superseded, Archived, Quarantined }

pub struct Memory {
    id: MemoryId,
    scope: Scope,
    kind: MemoryKind,
    text: String,
    importance: f32,                     // 0.0 to 10.0
    confidence: f32,                     // 0.0 to 1.0
    trust: Trust,
    status: Status,
    valid_from: Option<Timestamp>,
    valid_to: Option<Timestamp>,
    recorded_at: Timestamp,
    superseded_at: Option<Timestamp>,
    superseded_by: Option<MemoryId>,
    sources: Vec<EpisodeId>,             // provenance
    last_accessed: Timestamp,
    access_count: u32,
    embed_model: String,                 // vectors are tagged so we can re-embed
}

pub struct Block {
    label: String,
    value: String,
    token_limit: usize,
    version: u32,
    trust: Trust,
}
```

## 6. Algorithms

Candidate generation. Dense top-N by cosine similarity, BM25 top-N from FTS5, and later an entity match. Vectors are L2-normalized at insert time, so cosine is a dot product.

Reciprocal rank fusion:

```text
score(d) = sum over lists i of w_i / (k + rank_i(d))       k = 60
```

Rescoring:

```text
final = w_rel * rel_norm
        + w_rec * exp(-ln(2) * age / half_life[kind])
        + w_imp * importance / 10
```

Weights are configurable per query profile. Half-lives vary by memory kind. Episodic decays fast, procedural does not decay.

Diversification with maximal marginal relevance:

```text
pick d maximizing lambda * sim(q, d) - (1 - lambda) * max over selected s of sim(d, s)
```

Budget packing. Greedy by utility per token. Pinned blocks and the latest user turns are hard constraints that are always included.

Dedupe and supersede on the write path:

- If cosine similarity to an existing memory is at least tau_dup (about 0.92), reinforce the existing memory instead of inserting.
- If a new memory has the same entity and predicate as an existing one but a different object, close the old memory's `valid_to`, set `superseded_at` and `superseded_by`, and mark it `Superseded`.

Eviction, in order:

1. Clear old tool results.
2. Drop low-utility retrieved items.
3. Summarize the oldest turns.

Pinned blocks and the latest user turns are never evicted. Everything evicted is already in the episode log.

Forgetting:

```text
strength = importance * (1 + ln(1 + access_count)) * exp(-age / tau[kind])
```

Below a threshold, a memory is archived. It is never hard-deleted by default.

Reflection. Accumulate the importance of new memories since the last reflection. When the sum crosses a threshold, generate higher-level memories that cite their source memories.

## 7. Storage

- SQLite through rusqlite, with the bundled SQLite build.
- Pragmas: `journal_mode = WAL`, `synchronous = NORMAL`, `foreign_keys = ON`.
- Schema versioning through `PRAGMA user_version`, applied by an ordered list of migrations.
- Ids are UUIDv7, stored as 16-byte BLOBs.
- Timestamps are `i64` microseconds since the Unix epoch.
- Embeddings are stored as little-endian `f32` BLOBs, L2-normalized, tagged with the model id.
- Full-text search uses an FTS5 external-content table over `memory.text`.
- The vector index is loaded into RAM when the store opens: a contiguous row-major `Vec<f32>`, a parallel id vector, and a tombstone bitset. At 384 dimensions, 100k vectors is approx 150 MB.
- If scan latency becomes a problem later on, we add int8 quantization first, then HNSW.

## 8. Security and governance

- Scope is a required argument everywhere. There is no unscoped query path.
- Every memory records its source episodes and a trust level.
- Retrieved memories are data, not instructions. They are rendered into prompts inside clearly delimited, labelled sections.
- Memories derived from `Tool` or `External` content are never auto-promoted into pinned blocks.
- A write hook can redact or veto a memory before it is stored.
- A `Quarantined` status keeps suspicious memories out of recall until reviewed.

## 9. Roadmap

| Phase | Deliverable |
|---|---|
| 0 | Scaffold, toolchain, lint policy, docs |
| 1 | Errors, ids, timestamps, scope |
| 2 | Ports (`Clock`, `Embedder`, `Tokenizer`) and their mocks/fakes |
| 3 | Episode and memory types, SQLite store, migrations |
| 4 | Flat vector index, top-k heap, benchmark |
| 5 | RRF, rescoring, MMR, the `recall()` pipeline |
| 6 | Blocks, budget packer, evictor, `build_context()` |
| 7 | Extractor port, dedupe, supersession, write pipeline |
| 8 | Real adapters: local embeddings, HTTP LLM extractor |
| 9 | Decay, reflection, background worker |
| 10 | Entities, edges, entity-boosted retrieval |
| 11 | CLI and MCP server |
| 12 | Evaluation harness, int8 quantization, HNSW |

## 10. Open questions

- Which local embedding model is the default, and what dimension?
- Where is the crossover where a flat scan loses to HNSW on our hardware?
- How do we extract entities and predicates reliably without an LLM in the loop?
- What are good default half-lives, weights, and thresholds? We should tune them with an evaluation harness, not by feel.
