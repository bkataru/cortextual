//! Schema migrations
//!
//! The schema is a list of numbered SQL files, applied in order. The database
//! remembers how far it has got in SQLite's `user_version` header field, so
//! opening a database applies exactly the migrations it is missing.
//!
//! A released migration is never edited. A schema change is a new file with
//! the next number.
//!
//! Every database file is also stamped with an application id, and a file
//! that is neither empty nor stamped is refused. Pointing cortextual at some
//! other program's SQLite file must fail, not "migrate" it.

use rusqlite::Connection;

use crate::error::{Error, Result};

/// Marks a SQLite file as a cortextual database: "CXTX" in ASCII.
pub(crate) const APPLICATION_ID: i32 = 0x4358_5458;

struct Migration {
    version: u32,
    sql: &'static str,
}

const MIGRATIONS: &[Migration] = &[Migration {
    version: 1,
    sql: include_str!("migrations/0001_initial.sql"),
}];

/// The schema version a fully migrated database has.
pub(crate) fn latest_version() -> u32 {
    MIGRATIONS.last().map_or(0, |migration| migration.version)
}

/// The schema version recorded in the database, 0 for a fresh one.
pub(crate) fn read_user_version(conn: &Connection) -> Result<u32> {
    let raw: i64 = conn
        .pragma_query_value(None, "user_version", |row| row.get(0))
        .map_err(Error::storage)?;
    u32::try_from(raw)
        .map_err(|_| Error::IncompatibleDatabase(format!("invalid schema version {raw}")))
}

/// Checks that the database is either empty or already ours, without
/// changing anything.
pub(crate) fn ensure_ours(conn: &Connection) -> Result<()> {
    let application_id: i32 = conn
        .pragma_query_value(None, "application_id", |row| row.get(0))
        .map_err(Error::storage)?;
    if application_id == APPLICATION_ID {
        return Ok(());
    }

    let objects: i64 = conn
        .query_row("SELECT count(*) FROM sqlite_master", [], |row| row.get(0))
        .map_err(Error::storage)?;
    let fresh = application_id == 0 && objects == 0 && read_user_version(conn)? == 0;
    if fresh {
        Ok(())
    } else {
        Err(Error::IncompatibleDatabase(format!(
            "not a cortextual database (application id {application_id}, \
            {objects} existing objects)"
        )))
    }
}

/// Applies every migration the database is missing.
pub(crate) fn migrate(conn: &mut Connection) -> Result<()> {
    let latest = latest_version();
    let current = read_user_version(conn)?;
    if current > latest {
        return Err(Error::IncompatibleDatabase(format!(
            "database schema version {current} is newer than the newest this \
            version of cortextual supports ({latest})"
        )));
    }

    for migration in MIGRATIONS.iter().filter(|m| m.version > current) {
        apply(conn, migration)?;
    }

    Ok(())
}

/// Applies one migration atomically: the schema, the application id, and the
/// new version all land together, or none of them do.
fn apply(conn: &mut Connection, migration: &Migration) -> Result<()> {
    // TODO: i dont like the repetition of .map_err(Error::storage)? here,
    // is there a way to design this better? or do error propagation/handling differently so as to not need
    // to map errors at every step of the way for this function?
    // or is this the idiomatic/recommended way to go about it here?

    let tx = conn.transaction().map_err(Error::storage)?;
    tx.pragma_update(None, "application_id", APPLICATION_ID)
        .map_err(Error::storage)?;
    tx.execute_batch(migration.sql).map_err(Error::storage)?;
    tx.pragma_update(None, "user_version", migration.version)
        .map_err(Error::storage)?;
    tx.commit().map_err(Error::storage)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn connection() -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        conn.pragma_update(None, "foreign_keys", true).unwrap();
        conn
    }

    fn object_count(conn: &Connection) -> i64 {
        conn.query_row("SELECT count(*) FROM sqlite_master", [], |row| row.get(0))
            .unwrap()
    }

    fn application_id(conn: &Connection) -> i32 {
        conn.pragma_query_value(None, "application_id", |row| row.get(0))
            .unwrap()
    }

    fn is_incompatible(result: &Result<()>, needle: &str) -> bool {
        matches!(result, Err(Error::IncompatibleDatabase(msg)) if msg.contains(needle))
    }

    #[test]
    fn migration_versions_are_consecutive_from_one() {
        for (index, migration) in MIGRATIONS.iter().enumerate() {
            assert_eq!(migration.version, u32::try_from(index + 1).unwrap());
        }
        assert_eq!(latest_version(), u32::try_from(MIGRATIONS.len()).unwrap());
    }

    #[test]
    fn a_fresh_database_is_migrated_and_stamped() {
        let mut conn = connection();
        ensure_ours(&conn).unwrap();
        migrate(&mut conn).unwrap();

        assert_eq!(read_user_version(&conn).unwrap(), latest_version());
        assert_eq!(application_id(&conn), APPLICATION_ID);
        assert!(object_count(&conn) > 0);
        ensure_ours(&conn).unwrap();
    }

    #[test]
    fn migrating_again_changes_nothing() {
        let mut conn = connection();
        migrate(&mut conn).unwrap();
        let objects = object_count(&conn);

        migrate(&mut conn).unwrap();
        assert_eq!(object_count(&conn), objects);
        assert_eq!(read_user_version(&conn).unwrap(), latest_version());
    }

    #[test]
    fn foreign_databases_are_refused() {
        let with_a_table = connection();
        with_a_table.execute_batch("CREATE TABLE t (x)").unwrap();
        assert!(is_incompatible(
            &ensure_ours(&with_a_table),
            "not a cortextual database"
        ));

        let other_application = connection();
        other_application
            .pragma_update(None, "application_id", 1234)
            .unwrap();
        assert!(is_incompatible(
            &ensure_ours(&other_application),
            "application id 1234"
        ));

        let versioned_elsewhere = connection();
        versioned_elsewhere
            .pragma_update(None, "user_version", 7)
            .unwrap();
        assert!(is_incompatible(
            &ensure_ours(&versioned_elsewhere),
            "not a cortextual database"
        ));
    }

    #[test]
    fn databases_from_a_newer_version_are_refused() {
        let mut conn = connection();
        migrate(&mut conn).unwrap();
        conn.pragma_update(None, "user_version", latest_version() + 1)
            .unwrap();

        assert!(is_incompatible(&migrate(&mut conn), "newer"));
    }

    #[test]
    fn a_failed_migration_leaves_no_trace() {
        let mut conn = connection();
        let broken = Migration {
            version: 1,
            sql: "CREATE TABLE fine (x); CREATE TABLE broken (",
        };

        let err = apply(&mut conn, &broken).unwrap_err();
        assert!(matches!(err, Error::Storage(_)));
        assert_eq!(object_count(&conn), 0);
        assert_eq!(read_user_version(&conn).unwrap(), 0);
        assert_eq!(application_id(&conn), 0);
    }
}
