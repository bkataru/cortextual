//! Tests that the schema keeps its promises.
//!
//! These talk to SQLite directly, with hand-written SQL, instead of going
//! through the store. That is the point: the constraints and triggers must
//! hold for *any* writer, including a buggy one.

use rusqlite::{Connection, params};

use crate::store::migrate;
use crate::types::{
    AgentId, ContentHash, EpisodeId, EpisodeKind, MemoryId, MemoryKind, MemoryStatus, Scope,
    SessionId, Trust,
};

/// A fully-migrated in-memory database with foreign keys enforced.
fn db() -> Connection {
    let mut conn = Connection::open_in_memory().unwrap();
    conn.pragma_update(None, "foreign_keys", true).unwrap();
    migrate::migrate(&mut conn).unwrap();
    conn
}

/// The parts of a scope as database values: tenant, user, agent, session.
fn scope_columns(key: &str) -> (String, String, Option<String>, Option<String>) {
    let scope: Scope = key.parse().unwrap();
    (
        scope.tenant().as_str().to_owned(),
        scope.user().as_str().to_owned(),
        scope.agent().map(AgentId::as_str).map(str::to_owned),
        scope.session().map(SessionId::as_str).map(str::to_owned),
    )
}

/// An episode row. Override fields with struct update syntax.
struct Ep<'a> {
    id: Vec<u8>,
    scope: &'a str,
    kind: &'a str,
    content: &'a str,
    content_hash: Vec<u8>,
}

impl Default for Ep<'_> {
    fn default() -> Self {
        Self {
            id: EpisodeId::generate().as_bytes().to_vec(),
            scope: "acme/alice//",
            kind: "user_message",
            content: "hello",
            content_hash: ContentHash::of("hello").as_bytes().to_vec(),
        }
    }
}

fn insert_episode(conn: &Connection, e: &Ep<'_>) -> rusqlite::Result<Vec<u8>> {
    let (tenant, user, agent, session) = scope_columns(e.scope);
    conn.execute(
        "INSERT INTO episode
             (id, tenant_id, user_id, agent_id, session_id, kind, content,
              content_hash, occurred_at, ingested_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, 100, 100)",
        params![
            e.id,
            tenant,
            user,
            agent,
            session,
            e.kind,
            e.content,
            e.content_hash
        ],
    )?;
    Ok(e.id.clone())
}

/// A memory row. Override fields with struct update syntax.
struct Mem<'a> {
    id: Vec<u8>,
    scope: &'a str,
    kind: &'a str,
    text: &'a str,
    importance: f64,
    confidence: f64,
    trust: &'a str,
    status: &'a str,
    valid_from: Option<i64>,
    valid_to: Option<i64>,
    recorded_at: i64,
    superseded_at: Option<i64>,
    superseded_by: Option<Vec<u8>>,
}

impl Default for Mem<'_> {
    fn default() -> Self {
        Self {
            id: MemoryId::generate().as_bytes().to_vec(),
            scope: "acme/alice//",
            kind: "semantic",
            text: "a fact",
            importance: 5.0,
            confidence: 1.0,
            trust: "user",
            status: "active",
            valid_from: None,
            valid_to: None,
            recorded_at: 100,
            superseded_at: None,
            superseded_by: None,
        }
    }
}

fn insert_memory(conn: &Connection, m: &Mem<'_>) -> rusqlite::Result<Vec<u8>> {
    let (tenant, user, agent, session) = scope_columns(m.scope);
    conn.execute(
        "INSERT INTO memory
             (id, tenant_id, user_id, agent_id, session_id, kind, text,
              importance, confidence, trust, status, valid_from, valid_to,
              recorded_at, superseded_at, superseded_by, last_accessed)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13,
                 ?14, ?15, ?16, ?14)",
        params![
            m.id,
            tenant,
            user,
            agent,
            session,
            m.kind,
            m.text,
            m.importance,
            m.confidence,
            m.trust,
            m.status,
            m.valid_from,
            m.valid_to,
            m.recorded_at,
            m.superseded_at,
            m.superseded_by
        ],
    )?;
    Ok(m.id.clone())
}

fn link(
    conn: &Connection,
    memory: &[u8],
    episode: &[u8],
    position: i64,
) -> rusqlite::Result<usize> {
    conn.execute(
        "INSERT INTO memory_source (memory_id, episode_id, position) VALUES (?1, ?2, ?3)",
        params![memory, episode, position],
    )
}

fn message_of(error: &rusqlite::Error) -> String {
    error.to_string()
}

#[test]
fn every_rust_enum_name_is_accepted_by_its_check_constraint() {
    let conn = db();

    for kind in EpisodeKind::ALL {
        let row = Ep {
            kind: kind.as_str(),
            ..Ep::default()
        };
        insert_episode(&conn, &row).unwrap();
    }
    for &kind in MemoryKind::ALL {
        let row = Mem {
            kind: kind.as_str(),
            ..Mem::default()
        };
        insert_memory(&conn, &row).unwrap();
    }
    for &trust in Trust::ALL {
        let row = Mem {
            trust: trust.as_str(),
            ..Mem::default()
        };
        insert_memory(&conn, &row).unwrap();
    }

    let successor = insert_memory(&conn, &Mem::default()).unwrap();
    for &status in MemoryStatus::ALL {
        let row = if status == MemoryStatus::Superseded {
            Mem {
                status: status.as_str(),
                superseded_at: Some(200),
                superseded_by: Some(successor.clone()),
                ..Mem::default()
            }
        } else {
            Mem {
                status: status.as_str(),
                ..Mem::default()
            }
        };
        insert_memory(&conn, &row).unwrap();
    }
}

#[test]
fn unknown_enum_names_are_rejected() {
    let conn = db();
    let bad_episode = Ep {
        kind: "tweet",
        ..Ep::default()
    };
    assert!(insert_episode(&conn, &bad_episode).is_err());

    for (column, row) in [
        (
            "kind",
            Mem {
                kind: "fact",
                ..Mem::default()
            },
        ),
        (
            "trust",
            Mem {
                trust: "root",
                ..Mem::default()
            },
        ),
        (
            "status",
            Mem {
                status: "gone",
                ..Mem::default()
            },
        ),
    ] {
        assert!(insert_memory(&conn, &row).is_err(), "bad {column}");
    }
}

#[test]
fn malformed_ids_hashes_and_content_are_rejected() {
    let conn = db();
    let short_id = Ep {
        id: vec![1, 2, 3],
        ..Ep::default()
    };
    let short_hash = Ep {
        content_hash: vec![0; 31],
        ..Ep::default()
    };
    let no_content = Ep {
        content: "",
        ..Ep::default()
    };
    for row in [short_id, short_hash, no_content] {
        assert!(insert_episode(&conn, &row).is_err());
    }

    let short_memory_id = Mem {
        id: vec![1, 2, 3],
        ..Mem::default()
    };
    assert!(insert_memory(&conn, &short_memory_id).is_err());
}

#[test]
fn ids_are_unique() {
    let conn = db();
    let episode = insert_episode(&conn, &Ep::default()).unwrap();
    let again = Ep {
        id: episode,
        ..Ep::default()
    };
    assert!(insert_episode(&conn, &again).is_err());

    let memory = insert_memory(&conn, &Mem::default()).unwrap();
    let again = Mem {
        id: memory,
        ..Mem::default()
    };
    assert!(insert_memory(&conn, &again).is_err());
}

#[test]
fn check_constraints_reject_impossible_memories() {
    let conn = db();
    let other = insert_memory(&conn, &Mem::default()).unwrap();
    let own_id = MemoryId::generate().as_bytes().to_vec();

    let rejected = [
        (
            "importance above 10",
            Mem {
                importance: 10.5,
                ..Mem::default()
            },
        ),
        (
            "importance below 0",
            Mem {
                importance: -0.5,
                ..Mem::default()
            },
        ),
        (
            "confidence above 1",
            Mem {
                confidence: 1.5,
                ..Mem::default()
            },
        ),
        (
            "confidence below 0",
            Mem {
                confidence: -0.1,
                ..Mem::default()
            },
        ),
        (
            "empty text",
            Mem {
                text: "",
                ..Mem::default()
            },
        ),
        (
            "inverted validity",
            Mem {
                valid_from: Some(20),
                valid_to: Some(10),
                ..Mem::default()
            },
        ),
        (
            "superseded without a successor",
            Mem {
                status: "superseded",
                superseded_at: Some(200),
                ..Mem::default()
            },
        ),
        (
            "superseded without a time",
            Mem {
                status: "superseded",
                superseded_by: Some(other.clone()),
                ..Mem::default()
            },
        ),
        (
            "successor recorded on an active memory",
            Mem {
                superseded_at: Some(200),
                superseded_by: Some(other.clone()),
                ..Mem::default()
            },
        ),
        (
            "superseded before it was recorded",
            Mem {
                status: "superseded",
                superseded_at: Some(50),
                superseded_by: Some(other.clone()),
                ..Mem::default()
            },
        ),
        (
            "superseded by itself",
            Mem {
                id: own_id.clone(),
                status: "superseded",
                superseded_at: Some(200),
                superseded_by: Some(own_id.clone()),
                ..Mem::default()
            },
        ),
    ];
    for (why, row) in rejected {
        assert!(insert_memory(&conn, &row).is_err(), "{why}");
    }
}

#[test]
fn boundary_values_are_accepted() {
    let conn = db();
    let other = insert_memory(&conn, &Mem::default()).unwrap();

    let accepted = [
        Mem {
            importance: 0.0,
            confidence: 0.0,
            ..Mem::default()
        },
        Mem {
            importance: 10.0,
            confidence: 1.0,
            ..Mem::default()
        },
        Mem {
            valid_from: Some(10),
            valid_to: Some(10),
            ..Mem::default()
        },
        Mem {
            status: "superseded",
            superseded_at: Some(100),
            superseded_by: Some(other),
            ..Mem::default()
        },
    ];
    for row in accepted {
        insert_memory(&conn, &row).unwrap();
    }
}

#[test]
fn episodes_cannot_be_updated() {
    let conn = db();
    let id = insert_episode(&conn, &Ep::default()).unwrap();

    let err = conn
        .execute(
            "UPDATE episode SET content = 'edited' WHERE id = ?1",
            params![id],
        )
        .unwrap_err();
    assert!(message_of(&err).contains("episodes are immutable"), "{err}");

    // Even an update that changes nothing is refused.
    assert!(conn.execute("UPDATE episode SET kind = kind", []).is_err());
}

#[test]
fn the_core_of_a_memory_cannot_be_updated() {
    let conn = db();
    let id = insert_memory(&conn, &Mem::default()).unwrap();

    for assignment in [
        "text = 'edited'",
        "kind = 'episodic'",
        "importance = 1",
        "confidence = 0.1",
        "trust = 'system'",
        "tenant_id = 'globex'",
        "user_id = 'bob'",
        "agent_id = 'bot'",
        "session_id = 's9'",
        "valid_from = 5",
        "recorded_at = 1",
        "id = x'00000000000000000000000000000000'",
        "seq = 999",
    ] {
        let sql = format!("UPDATE memory SET {assignment} WHERE id = ?1");
        let err = conn.execute(&sql, params![id]).unwrap_err();
        assert!(
            message_of(&err).contains("immutable"),
            "{assignment}: {err}"
        );
    }
}

#[test]
fn bookkeeping_columns_can_change() {
    let conn = db();
    let id = insert_memory(&conn, &Mem::default()).unwrap();
    let successor = insert_memory(&conn, &Mem::default()).unwrap();

    conn.execute(
        "UPDATE memory SET access_count = access_count + 1, last_accessed = 500
         WHERE id = ?1",
        params![id],
    )
    .unwrap();
    conn.execute(
        "UPDATE memory SET status = 'archived' WHERE id = ?1",
        params![id],
    )
    .unwrap();
    conn.execute(
        "UPDATE memory SET status = 'active' WHERE id = ?1",
        params![id],
    )
    .unwrap();
    conn.execute(
        "UPDATE memory
         SET status = 'superseded', valid_to = 150, superseded_at = 200,
             superseded_by = ?2
         WHERE id = ?1",
        params![id, successor],
    )
    .unwrap();
}

#[test]
fn a_superseded_memory_is_final() {
    let conn = db();
    let id = insert_memory(&conn, &Mem::default()).unwrap();
    let successor = insert_memory(&conn, &Mem::default()).unwrap();
    conn.execute(
        "UPDATE memory
         SET status = 'superseded', valid_to = 150, superseded_at = 200,
             superseded_by = ?2
         WHERE id = ?1",
        params![id, successor],
    )
    .unwrap();

    for assignment in [
        "status = 'active'",
        "status = 'archived'",
        "valid_to = NULL",
        "superseded_at = 300",
        "superseded_by = NULL",
    ] {
        let sql = format!("UPDATE memory SET {assignment} WHERE id = ?1");
        let err = conn.execute(&sql, params![id]).unwrap_err();
        assert!(message_of(&err).contains("final"), "{assignment}: {err}");
    }

    // History can still be read, and counted as accessed.
    conn.execute(
        "UPDATE memory SET access_count = 5 WHERE id = ?1",
        params![id],
    )
    .unwrap();
}

#[test]
fn provenance_links_are_checked_and_immutable() {
    let conn = db();
    let episode = insert_episode(&conn, &Ep::default()).unwrap();
    let memory = insert_memory(&conn, &Mem::default()).unwrap();
    let missing_episode = EpisodeId::generate().as_bytes().to_vec();
    let missing_memory = MemoryId::generate().as_bytes().to_vec();

    assert!(link(&conn, &memory, &missing_episode, 0).is_err());
    assert!(link(&conn, &missing_memory, &episode, 0).is_err());

    link(&conn, &memory, &episode, 0).unwrap();
    let err = conn
        .execute("UPDATE memory_source SET position = 1", [])
        .unwrap_err();
    assert!(
        message_of(&err).contains("provenance is immutable"),
        "{err}"
    );
}

#[test]
fn provenance_has_no_duplicate_links_or_positions() {
    let conn = db();
    let first = insert_episode(&conn, &Ep::default()).unwrap();
    let second = insert_episode(&conn, &Ep::default()).unwrap();
    let memory = insert_memory(&conn, &Mem::default()).unwrap();

    link(&conn, &memory, &first, 0).unwrap();
    assert!(link(&conn, &memory, &first, 1).is_err(), "same pair twice");
    assert!(
        link(&conn, &memory, &second, 0).is_err(),
        "same position twice"
    );
    link(&conn, &memory, &second, 1).unwrap();
}

#[test]
fn a_successor_must_exist() {
    let conn = db();
    let nobody = MemoryId::generate().as_bytes().to_vec();
    let row = Mem {
        status: "superseded",
        superseded_at: Some(200),
        superseded_by: Some(nobody),
        ..Mem::default()
    };
    assert!(insert_memory(&conn, &row).is_err());
}

#[test]
fn an_episode_with_derived_memories_cannot_be_deleted() {
    let conn = db();
    let used = insert_episode(&conn, &Ep::default()).unwrap();
    let unused = insert_episode(&conn, &Ep::default()).unwrap();
    let memory = insert_memory(&conn, &Mem::default()).unwrap();
    link(&conn, &memory, &used, 0).unwrap();

    assert!(
        conn.execute("DELETE FROM episode WHERE id = ?1", params![used])
            .is_err()
    );
    assert_eq!(
        conn.execute("DELETE FROM episode WHERE id = ?1", params![unused])
            .unwrap(),
        1
    );
}

#[test]
fn seq_records_insertion_order_even_when_ids_sort_the_other_way() {
    let conn = db();
    let high = Ep {
        id: vec![0xff; 16],
        ..Ep::default()
    };
    let low = Ep {
        id: vec![0x00; 16],
        ..Ep::default()
    };
    insert_episode(&conn, &high).unwrap();
    insert_episode(&conn, &low).unwrap();

    let mut statement = conn.prepare("SELECT id FROM episode ORDER BY seq").unwrap();
    let ids: Vec<Vec<u8>> = statement
        .query_map([], |row| row.get(0))
        .unwrap()
        .map(Result::unwrap)
        .collect();
    assert_eq!(ids, [vec![0xff; 16], vec![0x00; 16]]);
}

#[test]
fn full_text_search_finds_memories_and_forgets_deleted_ones() {
    let conn = db();
    let paris = insert_memory(
        &conn,
        &Mem {
            text: "Alice lives in Paris",
            ..Mem::default()
        },
    )
    .unwrap();
    insert_memory(
        &conn,
        &Mem {
            text: "Bob likes caf\u{e9} au lait",
            ..Mem::default()
        },
    )
    .unwrap();

    let search = |query: &str| -> Vec<String> {
        let mut statement = conn
            .prepare(
                "SELECT memory.text
                 FROM memory_fts JOIN memory ON memory.seq = memory_fts.rowid
                 WHERE memory_fts MATCH ?1
                 ORDER BY memory.seq",
            )
            .unwrap();
        statement
            .query_map(params![query], |row| row.get(0))
            .unwrap()
            .map(Result::unwrap)
            .collect()
    };

    assert_eq!(search("paris"), ["Alice lives in Paris"]);
    assert_eq!(search("CAFE"), ["Bob likes caf\u{e9} au lait"]);
    assert_eq!(search("lives OR likes").len(), 2);
    assert_eq!(search("berlin"), Vec::<String>::new());

    conn.execute("DELETE FROM memory WHERE id = ?1", params![paris])
        .unwrap();
    assert_eq!(search("paris"), Vec::<String>::new());
}

const SCOPES: [&str; 8] = [
    "acme/alice//",
    "acme/alice/bot/",
    "acme/alice//s1",
    "acme/alice/bot/s1",
    "acme/alice/bot/s2",
    "acme/alice/other/",
    "acme/bob//",
    "globex/alice//",
];

#[test]
fn the_sql_scope_predicate_agrees_with_scope_covers() {
    let conn = db();
    for key in SCOPES {
        let row = Mem {
            scope: key,
            text: key,
            ..Mem::default()
        };
        insert_memory(&conn, &row).unwrap();
    }

    let mut statement = conn
        .prepare(
            "SELECT text FROM memory
             WHERE tenant_id = ?1 AND user_id = ?2
               AND (agent_id IS NULL OR agent_id = ?3)
               AND (session_id IS NULL OR session_id = ?4)",
        )
        .unwrap();

    for reader_key in SCOPES {
        let reader: Scope = reader_key.parse().unwrap();
        let (tenant, user, agent, session) = scope_columns(reader_key);

        let mut visible_in_sql: Vec<String> = statement
            .query_map(params![tenant, user, agent, session], |row| row.get(0))
            .unwrap()
            .map(Result::unwrap)
            .collect();
        visible_in_sql.sort();

        let mut visible_in_rust: Vec<String> = SCOPES
            .iter()
            .filter(|writer| writer.parse::<Scope>().unwrap().covers(&reader))
            .map(|writer| (*writer).to_owned())
            .collect();
        visible_in_rust.sort();

        assert_eq!(visible_in_sql, visible_in_rust, "reader {reader_key}");
        assert!(visible_in_sql.contains(&reader_key.to_owned()));
    }
}
