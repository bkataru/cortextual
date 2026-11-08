//! Episodes: the ummutable record of what happened.
//!
//! An episode is one event an agent observed: a user message, a tool result, a
//! document. Episodes are the source of truth. They are never edited or
//! deleted, and every memory is derived from them (see
//! `docs/ARCHITECTURE.md`).
//!
//! There are two types, because two parties fill in an episode:
//!
//! - [`NewEpisode`] is what the caller supplies: scope, kind, content, and
//!   optionally when it happened. It is validated on creation.
//! - [`Episode`] is the stored record. Its id, ingestion time, and content hash
//!   are assigned when a draft is stamped by the store, never read from caller
//!   input.
//!
//! ```
//!  use cortextual::types::{EpisodeId, EpisodeKind, NewEpisode, Scope, Timestamp};
//!
//!  let scope: Scope = "acme/alice//".parse().unwrap();
//!  let draft = NewEpisode::new(scope, EpisodeKind::UserMessage, "remember the milk").
//!
//!  let now = Timestamp::from_secs(1_700_000_000).unwrap();
//!  let episode = draft.into_episode(EpisodeId::generate_at(now).unwrap(), now);
//!  assert_eq!(episode.content(), "remember the milk");
//!  assert_eq!(episode.ingested_at(), now);
//! ```

use std::fmt;
use std::str::FromStr;

use crate::error::{Error, Result};
use crate::types::{ContentHash, EpisodeId, Scope, Timestamp};

/// What kind of event an episode records.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EpisodeKind {
    /// A message from the user.
    UserMessage,
    /// A message from the assistant.
    AssistantMessage,
    /// A tool invocation requested by the agent.
    ToolCall,
    /// The output a tool returned.
    ToolResult,
    /// Something the agent noticed that is not a message, such as a sensor
    /// reading or an event from another system.
    Observation,
    /// A document or a chunk of one, ingested as reference material.
    Document,
}

impl EpisodeKind {
    /// Every kind, in declaration order.
    pub const ALL: [Self; 6] = [
        Self::UserMessage,
        Self::AssistantMessage,
        Self::ToolCall,
        Self::ToolResult,
        Self::Observation,
        Self::Document,
    ];

    /// The stable name used in storage and text forms.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::UserMessage => "user_message",
            Self::AssistantMessage => "assistant_message",
            Self::ToolCall => "tool_call",
            Self::ToolResult => "tool_result",
            Self::Observation => "observation",
            Self::Document => "document",
        }
    }
}

impl fmt::Display for EpisodeKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for EpisodeKind {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::ALL
            .into_iter()
            .find(|kind| kind.as_str() == s)
            .ok_or_else(|| Error::InvalidEpisode(format!("unknown episode kind {s:?}")))
    }
}

/// An episode as supplied by a caller, before the system has stamped it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NewEpisode {
    scope: Scope,
    kind: EpisodeKind,
    content: String,
    occurred_at: Option<Timestamp>,
}

impl NewEpisode {
    /// Creates a draft episode.
    ///
    /// The content is stored exactly as given, with no trimming.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidEpisode`] if the content is empty or only
    /// whitespace, or longer than [`Episode::MAX_CONTENT_BYTES`].
    pub fn new(scope: Scope, kind: EpisodeKind, content: impl Into<String>) -> Result<Self> {
        let content = content.into();
        validate_content(&content)?;
        Ok(Self {
            scope,
            kind,
            content,
            occurred_at: None,
        })
    }

    /// Says when the event happened, if that differs from when it is ingested.
    ///
    /// Without this, an episode is taken to have happened at the moment it is
    /// ingested.
    #[must_use]
    pub fn with_occurred_at(mut self, at: Timestamp) -> Self {
        self.occurred_at = Some(at);
        self
    }

    /// The scope the episode will belong to.
    pub fn scope(&self) -> &Scope {
        &self.scope
    }

    /// The kind of event.
    pub fn kind(&self) -> EpisodeKind {
        self.kind
    }

    /// The content.
    pub fn content(&self) -> &str {
        &self.content
    }

    /// Stamps the draft into a stored [`Episode`].
    ///
    /// The caller supplies the id and the ingestion time, normally from an
    /// injected clock: `EpisodeId::generate_at(now)` and `now`. The content
    /// hash is computed here. If the draft did not say when the event
    /// happened, `ingested_at` is used.
    ///
    /// An `occurred_at` later than `ingested_at` is accepted, because the
    /// caller's clock and ours can disagree.
    pub fn into_episode(self, id: EpisodeId, ingested_at: Timestamp) -> Episode {
        let content_hash = ContentHash::of(&self.content);
        Episode {
            id,
            scope: self.scope,
            kind: self.kind,
            content: self.content,
            occurred_at: self.occurred_at.unwrap_or(ingested_at),
            ingested_at,
            content_hash,
        }
    }
}

fn validate_content(content: &str) -> Result<()> {
    if content.trim().is_empty() {
        return Err(Error::InvalidEpisode(
            "content must not be empty".to_owned(),
        ));
    }
    let len = content.len();
    if len > Episode::MAX_CONTENT_BYTES {
        return Err(Error::InvalidEpisode(format!(
            "content is {len} bytes long; the limit is {}",
            Episode::MAX_CONTENT_BYTES
        )));
    }
    Ok(())
}

/// A stored, immutable record of one event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Episode {
    id: EpisodeId,
    scope: Scope,
    kind: EpisodeKind,
    content: String,
    occurred_at: Timestamp,
    ingested_at: Timestamp,
    content_hash: ContentHash,
}

impl Episode {
    /// The largest content an episode can hold, in bytes (1 MiB).
    ///
    /// Longer material should be split into chunks before ingestion.
    pub const MAX_CONTENT_BYTES: usize = 1 << 20;

    /// The episode's id.
    pub fn id(&self) -> EpisodeId {
        self.id
    }

    /// The scope the episode belongs to
    pub fn scope(&self) -> &Scope {
        &self.scope
    }

    /// The kind of event.
    pub fn kind(&self) -> EpisodeKind {
        self.kind
    }

    /// The content, exactly as it was supplied.
    pub fn content(&self) -> &str {
        &self.content
    }

    /// When the event happened (world time).
    pub fn occurred_at(&self) -> Timestamp {
        self.occurred_at
    }

    /// When the system ingested the event (system time).
    pub fn ingested_at(&self) -> Timestamp {
        self.ingested_at
    }

    /// The hash of the content, for exact-duplicate detection.
    pub fn content_hash(&self) -> ContentHash {
        self.content_hash
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    fn scope(key: &str) -> Scope {
        key.parse().unwrap()
    }

    fn draft(content: &str) -> NewEpisode {
        NewEpisode::new(scope("acme/alice//"), EpisodeKind::UserMessage, content).unwrap()
    }

    fn secs(s: i64) -> Timestamp {
        Timestamp::from_secs(s).unwrap()
    }

    #[test]
    fn episodes_are_send_sync_static() {
        fn check<T: Send + Sync + 'static>(_: &T) {}
        let now = secs(1_700_000_000);
        check(&draft("x"));
        check(&draft("x").into_episode(EpisodeId::generate_at(now).unwrap(), now));
    }

    #[test]
    fn kinds_round_trip_through_their_names() {
        for kind in EpisodeKind::ALL {
            assert_eq!(kind.as_str().parse::<EpisodeKind>().unwrap(), kind);
            assert_eq!(kind.to_string(), kind.as_str());
        }
    }

    #[test]
    fn kind_names_are_unique() {
        let names: HashSet<_> = EpisodeKind::ALL.iter().map(|k| k.as_str()).collect();
        assert_eq!(names.len(), EpisodeKind::ALL.len());
    }

    #[test]
    fn unknown_kinds_are_rejected() {
        let err = "tool".parse::<EpisodeKind>().unwrap_err();
        assert_eq!(
            err.to_string(),
            r#"invalid episode: unknown episode kind "tool""#
        );
    }

    #[test]
    fn empty_and_whitespace_only_content_is_rejected() {
        for bad in ["", "   ", "\n\t "] {
            let err =
                NewEpisode::new(scope("acme/alice//"), EpisodeKind::Document, bad).unwrap_err();
            assert_eq!(
                err.to_string(),
                "invalid episode: content must not be empty"
            );
        }
    }

    #[test]
    fn oversized_content_is_rejected_at_the_limit() {
        let at_limit = "x".repeat(Episode::MAX_CONTENT_BYTES);
        assert!(NewEpisode::new(scope("acme/alice//"), EpisodeKind::Document, at_limit).is_ok());

        let over = "x".repeat(Episode::MAX_CONTENT_BYTES + 1);
        let err = NewEpisode::new(scope("acme/alice//"), EpisodeKind::Document, over).unwrap_err();
        assert_eq!(
            err.to_string(),
            "invalid episode: content is 1048577 bytes long; the limit is 1048576"
        );
    }

    #[test]
    fn content_is_kept_exactly_as_given() {
        let episode =
            draft("  padded\n").into_episode(EpisodeId::generate_at(secs(1)).unwrap(), secs(1));
        assert_eq!(episode.content(), "  padded\n");
    }

    #[test]
    fn into_episode_fills_in_the_system_fields() {
        let now = secs(1_700_000_000);
        let id = EpisodeId::generate_at(now).unwrap();
        let episode = draft("remember the milk").into_episode(id, now);

        assert_eq!(episode.id(), id);
        assert_eq!(episode.ingested_at(), now);
        assert_eq!(episode.content_hash(), ContentHash::of("remember the milk"));
        assert_eq!(episode.scope(), &scope("acme/alice//"));
        assert_eq!(episode.kind(), EpisodeKind::UserMessage);
        assert_eq!(episode.id().timestamp(), Some(now));
    }

    #[test]
    fn occurrence_defaults_to_ingestion_time() {
        let now = secs(100);
        let episode = draft("x").into_episode(EpisodeId::generate_at(now).unwrap(), now);
        assert_eq!(episode.occurred_at(), now);
    }

    #[test]
    fn occurrence_can_be_set_earlier_or_later_than_ingestion() {
        let now = secs(100);
        for at in [secs(40), secs(160)] {
            let episode = draft("x")
                .with_occurred_at(at)
                .into_episode(EpisodeId::generate_at(now).unwrap(), now);
            assert_eq!(episode.occurred_at(), at);
            assert_eq!(episode.ingested_at(), now);
        }
    }

    #[test]
    fn the_hash_covers_content_only_not_scope() {
        let now = secs(1);
        let make = |key: &str| {
            NewEpisode::new(scope(key), EpisodeKind::Document, "same text")
                .unwrap()
                .into_episode(EpisodeId::generate_at(now).unwrap(), now)
        };
        assert_eq!(
            make("acme/alice//").content_hash(),
            make("globex/bob//").content_hash()
        );
    }

    #[test]
    fn drafts_expose_what_they_were_built_from() {
        let d = draft("hello");
        assert_eq!(d.scope(), &scope("acme/alice//"));
        assert_eq!(d.kind(), EpisodeKind::UserMessage);
        assert_eq!(d.content(), "hello");
    }
}
