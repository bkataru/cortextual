//! Ports: the traits that isolate the code from the outside world.
//!
//! Anything non-deterministic or expensive that cortextual depends on (the
//! wall clock, an embedding model, a tokenizer, an LLM) is reached through a
//! trait defined here. The core is written against the trait. Real
//! implementations (adapters) and deterministic fakes plug in from outside.
//! This is the "ports and adapters" software architecture pattern, also called
//! hexagonal architecture.
//!
//! Layering: `ports` may use `types`, never the reverse.

pub mod clock;

pub use clock::{Clock, SystemClock};
