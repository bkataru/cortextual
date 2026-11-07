//! Deterministic test doubles for the ports.
//!
//! These are ordinary public types, not `#[cfg(test)]` items, so integration
//! tests, benchmarks, and downstream crates can use them too. They are meant
//! for tests and examples, not for production:
//!
//! - [`ManualClock`] is a clock that only moves when you move it.
//! - [`HashEmbedder`] turns text into vectors without a model, with real
//!   similarity behavior: texts that share words end up closer together.
//! - [`WordTokenizer`] counts whitespace-separated words, exactly.
//!
//! Layering: `testing` may use `ports` and `types`.

pub mod hash_embedder;
pub mod manual_clock;
pub mod word_tokenizer;

pub use hash_embedder::HashEmbedder;
pub use manual_clock::ManualClock;
pub use word_tokenizer::WordTokenizer;
