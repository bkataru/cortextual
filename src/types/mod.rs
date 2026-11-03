//! Core domain types.
//!
//! Identifiers, time, and scope live here. Everything else in the crate is
//! build on these, so they depend on nothing but `std`, `crate::error`, and a
//! few small crates.

pub mod ids;
pub mod time;

pub use ids::{EntityId, EpisodeId, Id, IdTag, MemoryId};
pub use time::{Interval, Timestamp};
