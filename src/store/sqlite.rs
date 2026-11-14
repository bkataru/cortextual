//! The SQLite store.

use std::fmt;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use rusqlite::Connection;

use crate::error::{Error, Result};
use crate::ports::Clock;
use crate::store::migrate;

/// How long a connection waits for a lock held by another connection before
/// giving up.
const BUSY_TIMEOUT: Duration = Duration::from_secs(5);

/// cortextual's durable store, backed by a single SQLite file.
///
/// The store owns one database connection and the [`Clock`] it uses to stamp
/// what it records. It is `Send` but not `Sync`: one thread uses it at a time.
/// To share it between threads, wrap it in a `Mutex`.
pub struct SqliteStore {
    conn: Connection,
    clock: Arc<dyn Clock>,
}

impl SqliteStore {
    /// Opens the database at `path`, creating and migrating it if needed.
    ///
    /// # Errors
    ///
    /// Returns [`Error::IncompatibleDatabase`] if the file is not a
    /// cortextual database or was written by a newer version, and
    /// [`Error::Storage`] if SQLite cannot open or set up the file.
    pub fn open(path: impl AsRef<Path>, clock: Arc<dyn Clock>) -> Result<Self> {
        let conn = Connection::open(path).map_err(Error::storage)?;
        Self::from_connection(conn, clock)
    }

    /// Opens a private in-memory database. It disappears when the store is
    /// dropped, which makes it the right choice for tests.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Storage`] if SQLite cannot set the database up.
    pub fn open_in_memory(clock: Arc<dyn Clock>) -> Result<Self> {
        let conn = Connection::open_in_memory().map_err(Error::storage)?;
        Self::from_connection(conn, clock)
    }

    fn from_connection(mut conn: Connection, clock: Arc<dyn Clock>) -> Result<Self> {
        // Check ownership before configuring anything: switching a foreign
        // file to WAL mode would already be modifying it.
        migrate::ensure_ours(&conn)?;
        configure(&conn)?;
        migrate::migrate(&mut conn)?;
        Ok(Self { conn, clock })
    }

    /// The schema version of the open database.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Storage`] if the version cannot be read.
    pub fn schema_version(&self) -> Result<u32> {
        migrate::read_user_version(&self.conn)
    }

    /// The clock the store stamps records with.
    pub fn clock(&self) -> &Arc<dyn Clock> {
        &self.clock
    }
}

impl fmt::Debug for SqliteStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SqliteStore").finish_non_exhaustive()
    }
}

/// Per-connection settings. SQLite forgets these when a connection closes, so
/// they are applied on every open.
fn configure(conn: &Connection) -> Result<()> {
    // Without this SQLite ignores FOREIGN KEY constraints entirely.
    conn.pragma_update(None, "foreign_keys", true)
        .map_err(Error::storage)?;
    // With WAL, NORMAL cannot corrupt the file. A power cut can lose the most
    // recent commits, but never the database.
    conn.pragma_update(None, "synchronous", "NORMAL")
        .map_err(Error::storage)?;
    conn.busy_timeout(BUSY_TIMEOUT).map_err(Error::storage)?;
    // Changing the journal mode returns the mode now in effect as a row, so
    // it needs the "and check" form. In-memory databases report "memory" and
    // ignore the request, and some filesystems refuse WAL: both are fine.
    conn.pragma_update_and_check(None, "journal_mode", "WAL", |row| row.get::<_, String>(0))
        .map_err(Error::storage)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::migrate::{APPLICATION_ID, latest_version};
    use crate::testing::ManualClock;
    use crate::types::Timestamp;

    fn clock() -> Arc<dyn Clock> {
        Arc::new(ManualClock::new(Timestamp::from_secs(1_000).unwrap()))
    }

    fn pragma_string(conn: &Connection, name: &str) -> String {
        conn.pragma_query_value(None, name, |row| row.get::<_, String>(0))
            .unwrap()
    }

    #[test]
    fn the_store_can_move_between_threads() {
        fn check<T: Send>(_: &T) {}
        check(&SqliteStore::open_in_memory(clock()).unwrap());
    }

    #[test]
    fn an_in_memory_store_starts_at_the_latest_schema() {
        let store = SqliteStore::open_in_memory(clock()).unwrap();
        assert_eq!(store.schema_version().unwrap(), latest_version());
    }

    #[test]
    fn foreign_keys_are_enforced() {
        let store = SqliteStore::open_in_memory(clock()).unwrap();
        let enabled: i64 = store
            .conn
            .pragma_query_value(None, "foreign_keys", |row| row.get(0))
            .unwrap();
        assert_eq!(enabled, 1);
    }

    #[test]
    fn the_store_uses_the_injected_clock() {
        let manual = Arc::new(ManualClock::new(Timestamp::from_secs(42).unwrap()));
        let shared: Arc<dyn Clock> = manual.clone();
        let store = SqliteStore::open_in_memory(shared).unwrap();

        assert_eq!(store.clock().now(), Timestamp::from_secs(42).unwrap());
        manual.advance(Duration::from_secs(1));
        assert_eq!(store.clock().now(), Timestamp::from_secs(43).unwrap());
    }

    #[test]
    fn file_databases_use_wal_and_survive_reopening() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cortextual.db");

        {
            let store = SqliteStore::open(&path, clock()).unwrap();
            assert_eq!(pragma_string(&store.conn, "journal_mode"), "wal");
            assert_eq!(store.schema_version().unwrap(), latest_version());
        }

        let reopened = SqliteStore::open(&path, clock()).unwrap();
        assert_eq!(reopened.schema_version().unwrap(), latest_version());
    }

    #[test]
    fn foreign_sqlite_files_are_refused_and_left_untouched() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("someone_elses.db");
        {
            let other = Connection::open(&path).unwrap();
            other
                .execute_batch("CREATE TABLE notes (body TEXT)")
                .unwrap();
        }

        let err = SqliteStore::open(&path, clock()).unwrap_err();
        assert!(matches!(err, Error::IncompatibleDatabase(_)), "{err}");

        let after = Connection::open(&path).unwrap();
        assert_eq!(pragma_string(&after, "journal_mode"), "delete");
        let application_id: i32 = after
            .pragma_query_value(None, "application_id", |row| row.get(0))
            .unwrap();
        assert_eq!(application_id, 0);
        let tables: i64 = after
            .query_row("SELECT count(*) FROM sqlite_master", [], |row| row.get(0))
            .unwrap();
        assert_eq!(tables, 1);
    }

    #[test]
    fn files_from_a_newer_version_are_refused() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("future.db");
        drop(SqliteStore::open(&path, clock()).unwrap());
        {
            let raw = Connection::open(&path).unwrap();
            raw.pragma_update(None, "user_version", latest_version() + 1)
                .unwrap();
        }

        let err = SqliteStore::open(&path, clock()).unwrap_err();
        assert!(
            matches!(&err, Error::IncompatibleDatabase(msg) if msg.contains("newer")),
            "{err}"
        );
    }

    #[test]
    fn a_stamped_file_carries_the_application_id() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("stamped.db");
        drop(SqliteStore::open(&path, clock()).unwrap());

        let raw = Connection::open(&path).unwrap();
        let id: i32 = raw
            .pragma_query_value(None, "application_id", |row| row.get(0))
            .unwrap();
        assert_eq!(id, APPLICATION_ID);
    }
}
