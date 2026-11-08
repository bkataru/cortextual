//! Core domain types.
//!
//! Identifiers, time, and scope live here. Everything else in the crate is
//! built on these, so they depend on nothing but `std`, `crate::error`, and a
//! few small crates.

pub mod content_hash;
pub mod episode;
pub mod ids;
pub mod scope;
pub mod time;

pub use content_hash::ContentHash;
pub use episode::{Episode, EpisodeKind, NewEpisode};
pub use ids::{EntityId, EpisodeId, Id, IdTag, MemoryId};
pub use scope::{AgentId, Scope, SessionId, TenantId, UserId};
pub use time::{Interval, Timestamp};
