//! Episode storage: appending to the log and reading episodes back.

use rusqlite::{OptionalExtension, Transaction, params};

use crate::error::{Error, Result};
use crate::store::{SqliteStore, codec};
use crate::types::{Episode, EpisodeId, EpisodeKind, NewEpisode, Scope, Timestamp};

impl SqliteStore {
    /// Stamps `draft` with a fresh id and the current reading of the store's
    /// clock, and appends it to the episode log inside a transaction.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidTimestamp`] if the store's clock is outside the
    /// range an [`EpisodeId`] can represent, or [`Error::Storage`] if writing
    /// to SQLite fails.
    pub fn append_episode(&mut self, draft: NewEpisode) -> Result<Episode> {
        let now = self.clock.now();
        let id = EpisodeId::generate_at(now)?;
        let episode = draft.into_episode(id, now);

        let tx = self.conn.transaction().map_err(Error::storage)?;
        insert_episode(&tx, &episode)?;
        tx.commit().map_err(Error::storage)?;

        Ok(episode)
    }

    /// Reads one episode by `id` if it is visible to `scope`, or `None` if it
    /// does not exist or belongs to a narrower or unrelated scope.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Storage`] if querying SQLite fails, or
    /// [`Error::CorruptData`] if the stored row fails invariant checks.
    pub fn get_episode(&self, scope: &Scope, id: EpisodeId) -> Result<Option<Episode>> {
        let (tenant, user, agent, session) = codec::scope_parts(scope);
        let mut stmt = self
            .conn
            .prepare_cached(
                "SELECT id, tenant_id, user_id, agent_id, session_id, kind, content,
                        content_hash, occurred_at, ingested_at
                 FROM episode
                 WHERE id = ?1
                   AND tenant_id = ?2
                   AND user_id = ?3
                   AND (agent_id IS NULL OR agent_id = ?4)
                   AND (session_id IS NULL OR session_id = ?5)",
            )
            .map_err(Error::storage)?;

        let raw = stmt
            .query_row(
                params![id.as_bytes().as_slice(), tenant, user, agent, session],
                RawEpisodeRow::from_row,
            )
            .optional()
            .map_err(Error::storage)?;

        raw.map(RawEpisodeRow::into_episode).transpose()
    }

    /// Lists up to `limit` most recent episodes visible to `scope`.
    ///
    /// Ordered newest-first by `occurred_at`, with ties broken by insertion
    /// order (newest first). Returns an empty vector when `limit` is zero.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Storage`] if querying SQLite fails, or
    /// [`Error::CorruptData`] if any returned row fails invariant checks.
    pub fn recent_episodes(&self, scope: &Scope, limit: usize) -> Result<Vec<Episode>> {
        if limit == 0 {
            return Ok(Vec::new());
        }

        let (tenant, user, agent, session) = codec::scope_parts(scope);
        let limit = i64::try_from(limit).unwrap_or(i64::MAX);
        let mut stmt = self
            .conn
            .prepare_cached(
                "SELECT id, tenant_id, user_id, agent_id, session_id, kind, content,
                        content_hash, occurred_at, ingested_at
                 FROM episode
                 WHERE tenant_id = ?1
                   AND user_id = ?2
                   AND (agent_id IS NULL OR agent_id = ?3)
                   AND (session_id IS NULL OR session_id = ?4)
                 ORDER BY occurred_at DESC, seq DESC
                 LIMIT ?5",
            )
            .map_err(Error::storage)?;

        let rows = stmt
            .query_map(
                params![tenant, user, agent, session, limit],
                RawEpisodeRow::from_row,
            )
            .map_err(Error::storage)?;

        let mut episodes = Vec::new();
        for row in rows {
            let raw = row.map_err(Error::storage)?;
            episodes.push(raw.into_episode()?);
        }
        Ok(episodes)
    }
}

fn insert_episode(tx: &Transaction<'_>, episode: &Episode) -> Result<()> {
    let (tenant, user, agent, session) = codec::scope_parts(episode.scope());
    let mut stmt = tx
        .prepare_cached(
            "INSERT into episode
                (id, tenant_id, user_id, agent_id, session_id, kind, content,
                 content_hash, occurred_at, ingested_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
        )
        .map_err(Error::storage)?;

    stmt.execute(params![
        episode.id().as_bytes().as_slice(),
        tenant,
        user,
        agent,
        session,
        episode.kind().as_str(),
        episode.content(),
        episode.content_hash().as_bytes().as_slice(),
        episode.occurred_at().to_micros(),
        episode.ingested_at().to_micros(),
    ])
    .map_err(Error::storage)?;

    Ok(())
}

struct RawEpisodeRow {
    id: Vec<u8>,
    tenant_id: String,
    user_id: String,
    agent_id: Option<String>,
    session_id: Option<String>,
    kind: String,
    content: String,
    content_hash: Vec<u8>,
    occurred_at: i64,
    ingested_at: i64,
}

impl RawEpisodeRow {
    fn from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<Self> {
        Ok(Self {
            id: row.get(0)?,
            tenant_id: row.get(1)?,
            user_id: row.get(2)?,
            agent_id: row.get(3)?,
            session_id: row.get(4)?,
            kind: row.get(5)?,
            content: row.get(6)?,
            content_hash: row.get(7)?,
            occurred_at: row.get(8)?,
            ingested_at: row.get(9)?,
        })
    }

    fn into_episode(self) -> Result<Episode> {
        let id: EpisodeId = codec::decode_id(&self.id)?;
        let scope =
            codec::decode_scope(self.tenant_id, self.user_id, self.agent_id, self.session_id)?;
        let kind: EpisodeKind = codec::decode_enum(&self.kind)?;
        let content_hash = codec::decode_content_hash(&self.content_hash)?;
        let occurred_at = Timestamp::from_micros(self.occurred_at);
        let ingested_at = Timestamp::from_micros(self.ingested_at);

        Episode::restore(
            id,
            scope,
            kind,
            self.content,
            content_hash,
            occurred_at,
            ingested_at,
        )
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use super::*;
    use crate::ports::Clock;
    use crate::testing::ManualClock;
    use crate::types::ContentHash;

    fn secs(s: i64) -> Timestamp {
        Timestamp::from_secs(s).unwrap()
    }

    fn scope(key: &str) -> Scope {
        key.parse().unwrap()
    }

    fn store_at(seconds: i64) -> (SqliteStore, Arc<ManualClock>) {
        let manual = Arc::new(ManualClock::new(secs(seconds)));
        let shared: Arc<dyn Clock> = manual.clone();
        let store = SqliteStore::open_in_memory(shared).unwrap();
        (store, manual)
    }

    #[test]
    fn append_and_get_round_trips_every_field() {
        let (mut store, clock) = store_at(1_700_000_000);
        let bare_scope = scope("acme/alice//");
        let full_scope = scope("acme/alice/bot/s1");

        let bare = store
            .append_episode(
                NewEpisode::new(bare_scope.clone(), EpisodeKind::UserMessage, "hello").unwrap(),
            )
            .unwrap();
        assert_eq!(bare.scope(), &bare_scope);
        assert_eq!(bare.kind(), EpisodeKind::UserMessage);
        assert_eq!(bare.content(), "hello");
        assert_eq!(bare.content_hash(), ContentHash::of("hello"));
        assert_eq!(bare.ingested_at(), secs(1_700_000_000));
        assert_eq!(bare.occurred_at(), secs(1_700_000_000));
        assert_eq!(bare.id().timestamp(), Some(secs(1_700_000_000)));
        assert_eq!(
            store.get_episode(&bare_scope, bare.id()).unwrap(),
            Some(bare)
        );

        clock.advance(Duration::from_secs(60));
        for kind in EpisodeKind::ALL {
            let saved = store
                .append_episode(
                    NewEpisode::new(full_scope.clone(), kind, format!("event {kind}"))
                        .unwrap()
                        .with_occurred_at(secs(1_699_999_500)),
                )
                .unwrap();
            assert_eq!(saved.scope(), &full_scope);
            assert_eq!(saved.kind(), kind);
            assert_eq!(saved.occurred_at(), secs(1_699_999_500));
            assert_eq!(saved.ingested_at(), secs(1_700_000_060));
            assert_eq!(
                store.get_episode(&full_scope, saved.id()).unwrap(),
                Some(saved)
            );
        }
    }

    #[test]
    fn get_episode_returns_none_when_missing() {
        let (store, _) = store_at(1_000);
        assert_eq!(
            store
                .get_episode(&scope("acme/alice//"), EpisodeId::generate())
                .unwrap(),
            None
        );
    }

    #[test]
    fn append_fails_before_writing_if_clock_is_outside_id_range() {
        let manual = Arc::new(ManualClock::new(Timestamp::from_micros(-1)));
        let clock: Arc<dyn Clock> = manual;
        let mut store = SqliteStore::open_in_memory(clock).unwrap();

        let err = store
            .append_episode(
                NewEpisode::new(scope("acme/alice//"), EpisodeKind::UserMessage, "hello").unwrap(),
            )
            .unwrap_err();
        assert!(matches!(err, Error::InvalidTimestamp(_)), "{err}");

        let count: i64 = store
            .conn
            .query_row("SELECT count(*) FROM episode", [], |row| row.get(0))
            .unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn recent_episodes_orders_newest_first_and_breaks_ties_by_insertion_order() {
        let (mut store, clock) = store_at(1_000);
        let s = scope("acme/alice//");

        let append = |store: &mut SqliteStore, text: &str, occurred_at: Option<Timestamp>| {
            let mut draft = NewEpisode::new(s.clone(), EpisodeKind::UserMessage, text).unwrap();
            if let Some(at) = occurred_at {
                draft = draft.with_occurred_at(at);
            }
            store.append_episode(draft).unwrap()
        };

        let first = append(&mut store, "first", None);
        let second = append(&mut store, "second", None);
        let third = append(&mut store, "third", None);

        clock.advance(Duration::from_secs(10));
        let backfilled = append(&mut store, "backfilled", Some(secs(500)));
        let newest = append(&mut store, "newest", None);

        assert_eq!(
            store.recent_episodes(&s, 10).unwrap(),
            vec![newest, third, second, first, backfilled]
        );
    }

    #[test]
    fn recent_episodes_respects_limit_including_zero() {
        let (mut store, clock) = store_at(1_000);
        let s = scope("acme/alice//");

        let first = store
            .append_episode(NewEpisode::new(s.clone(), EpisodeKind::UserMessage, "one").unwrap())
            .unwrap();
        clock.advance(Duration::from_secs(1));
        let second = store
            .append_episode(NewEpisode::new(s.clone(), EpisodeKind::UserMessage, "two").unwrap())
            .unwrap();
        clock.advance(Duration::from_secs(1));
        let third = store
            .append_episode(NewEpisode::new(s.clone(), EpisodeKind::UserMessage, "three").unwrap())
            .unwrap();

        assert_eq!(store.recent_episodes(&s, 0).unwrap(), []);
        assert_eq!(
            store.recent_episodes(&s, 2).unwrap(),
            vec![third.clone(), second.clone()]
        );
        assert_eq!(
            store.recent_episodes(&s, 10).unwrap(),
            vec![third, second, first]
        );
    }

    #[test]
    fn queries_enforce_scope_coverage() {
        let (mut store, _) = store_at(1_000);
        let save = |store: &mut SqliteStore, key: &str| {
            store
                .append_episode(NewEpisode::new(scope(key), EpisodeKind::Observation, key).unwrap())
                .unwrap()
        };

        let user_wide = save(&mut store, "acme/alice//");
        let agent_wide = save(&mut store, "acme/alice/bot/");
        let session_wide = save(&mut store, "acme/alice//s1");
        let narrow = save(&mut store, "acme/alice/bot/s1");
        let other_session = save(&mut store, "acme/alice/bot/s2");
        let other_agent = save(&mut store, "acme/alice/other/");
        let other_user = save(&mut store, "acme/bob//");
        let other_tenant = save(&mut store, "globex/alice//");

        let reader = scope("acme/alice/bot/s1");
        let visible = store.recent_episodes(&reader, 20).unwrap();
        assert_eq!(
            visible,
            vec![
                narrow.clone(),
                session_wide.clone(),
                agent_wide.clone(),
                user_wide.clone(),
            ]
        );

        for allowed in [&user_wide, &agent_wide, &session_wide, &narrow] {
            assert_eq!(
                store.get_episode(&reader, allowed.id()).unwrap().as_ref(),
                Some(allowed)
            );
        }
        for forbidden in [&other_session, &other_agent, &other_user, &other_tenant] {
            assert_eq!(store.get_episode(&reader, forbidden.id()).unwrap(), None);
        }

        // A broader reader cannot see episodes scoped to a narrower agent or session.
        let broad_reader = scope("acme/alice//");
        assert_eq!(
            store.recent_episodes(&broad_reader, 20).unwrap(),
            vec![user_wide]
        );
        assert_eq!(
            store.get_episode(&broad_reader, agent_wide.id()).unwrap(),
            None
        );
        assert_eq!(
            store.get_episode(&broad_reader, session_wide.id()).unwrap(),
            None
        );
    }

    #[test]
    fn episodes_survive_reopening_a_file_database() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("episodes.db");
        let s = scope("acme/alice/bot/");

        let saved = {
            let clock: Arc<dyn Clock> = Arc::new(ManualClock::new(secs(5_000)));
            let mut store = SqliteStore::open(&path, clock).unwrap();
            store
                .append_episode(
                    NewEpisode::new(s.clone(), EpisodeKind::Document, "persisted fact").unwrap(),
                )
                .unwrap()
        };

        let clock: Arc<dyn Clock> = Arc::new(ManualClock::new(secs(6_000)));
        let reopened = SqliteStore::open(&path, clock).unwrap();
        assert_eq!(
            reopened.get_episode(&s, saved.id()).unwrap(),
            Some(saved.clone())
        );
        assert_eq!(reopened.recent_episodes(&s, 10).unwrap(), vec![saved]);
    }

    #[test]
    fn tampered_content_is_rejected_with_corrupt_data() {
        let (mut store, _) = store_at(1_000);
        let s = scope("acme/alice//");
        let saved = store
            .append_episode(
                NewEpisode::new(s.clone(), EpisodeKind::UserMessage, "original").unwrap(),
            )
            .unwrap();

        store
            .conn
            .execute_batch("DROP TRIGGER episode_is_immutable")
            .unwrap();

        store
            .conn
            .execute(
                "UPDATE episode SET content = 'tampered' WHERE id = ?1",
                params![saved.id().as_bytes().as_slice()],
            )
            .unwrap();

        let by_id = store.get_episode(&s, saved.id()).unwrap_err();
        assert!(matches!(by_id, Error::CorruptData(_)), "{by_id}");
        assert!(by_id.to_string().contains("hash mismatch"), "{by_id}");

        let listed = store.recent_episodes(&s, 10).unwrap_err();
        assert!(matches!(listed, Error::CorruptData(_)), "{listed}");
    }

    #[test]
    fn corrupt_rows_are_rejected_with_corrupt_data() {
        struct BadRow<'a> {
            label: &'a str,
            id: Vec<u8>,
            kind: &'a str,
            content: &'a str,
            hash: Vec<u8>,
        }

        let reader = scope("acme/alice//");
        let valid_id = EpisodeId::generate().as_bytes().to_vec();
        let valid_hash = ContentHash::of("hello").as_bytes().to_vec();

        let bad_rows = [
            BadRow {
                label: "short id",
                id: vec![1, 2, 3],
                kind: "user_message",
                content: "hello",
                hash: valid_hash.clone(),
            },
            BadRow {
                label: "short hash",
                id: valid_id.clone(),
                kind: "user_message",
                content: "hello",
                hash: vec![0; 31],
            },
            BadRow {
                label: "unknown kind",
                id: valid_id.clone(),
                kind: "tweet",
                content: "hello",
                hash: valid_hash,
            },
            BadRow {
                label: "whitespace-only content even with matching hash",
                id: valid_id,
                kind: "user_message",
                content: "   ",
                hash: ContentHash::of("   ").as_bytes().to_vec(),
            },
        ];

        for row in bad_rows {
            let (store, _) = store_at(1_000);
            store
                .conn
                .pragma_update(None, "ignore_check_constraints", true)
                .unwrap();
            store
                .conn
                .execute(
                    "INSERT into episode
                         (id, tenant_id, user_id, agent_id, session_id, kind, content,
                          content_hash, occurred_at, ingested_at)
                     VALUES (?1, 'acme', 'alice', NULL, NULL, ?2, ?3, ?4, 1000, 1000)",
                    params![row.id, row.kind, row.content, row.hash],
                )
                .unwrap();

            let err = store.recent_episodes(&reader, 10).unwrap_err();
            assert!(matches!(err, Error::CorruptData(_)), "{}: {err}", row.label);
        }
    }
}
