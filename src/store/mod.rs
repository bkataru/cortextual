//! Durable storage.
//!
//! All of cortextual's state lives in one SQLite file, reached through
//! [`SqliteStore`]. A few choices are worth knowing about:
//!
//! - **A concrete type, not a trait.** A trait designed against a single
//!   implementation just mirrors that implementation. Tests use an in-memory
//!   SQLite database, which is fast and needs no fake. If a second backend
//!   ever appears, the trait can be extracted from two real examples.
//! - **One file in WAL mode.** Readers do not block the writer and a crash
//!   cannot corrupt the file.
//! - **The schema defends itself.** Constraints reject impossible rows, and
//!   triggers refuse to rewrite history: episodes and the core of a memory
//!   cannot be updated, and a superseded memory is final. A bug in this
//!   crate cannot quietly edit the past.
//! - **Migrations are numbered SQL files**, each applied in its own
//!   transaction, with the version kept in the database header. The file is
//!   stamped with an application id, so cortextual refuses to open SQLite
//!   files that are not its own.

mod migrate;
mod sqlite;

pub use sqlite::SqliteStore;
