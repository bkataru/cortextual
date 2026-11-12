//! Memories: derived knowledge with provenance, trust, and two clocks.
//!
//! Episodes record what happened. A memory is something worth remembering
//! that was derived from episodes: a fact, a summary, a rule, a conclusion.
//! Where an episode is immutable, a memory has a small life cycle: it can be
//! recalled, superseded by a newer memory, archived, or quarantined.
//!
//! Like episodes, memories come in two types. [`NewMemory`] is the validated
//! draft produced by extraction. [`Memory`] is the stored record, stamped with
//! an id and a recording time when the store accepts the draft.
//!
//! # Two clocks
//!
//! Every memory has a *valid time* ([`Memory::valid`]): when the fact is true
//! in the world. It also has a *record time*: when we learned it
//! ([`Memory::recorded_at`]) and, if it was replaced, when we stopped treating
//! it as current ([`Memory::superseded_at`]).
//!
//! That supports three kinds of question, each answered by a different method:
//!
//! - What is true now? The memories whose status is [`MemoryStatus::Active`].
//! - What was true in the world at time `v`, as we know it today?
//!   [`Memory::valid_at`].
//! - What did we hold as current at time `r`? [`Memory::believed_at`].
//!
//! ```
//! use cortextual::types::{
//!     EpisodeId, EpisodeKind, MemoryId, MemoryKind, NewEpisode, NewMemory, Scope, Timestamp,
//! };
//!
//! let scope: Scope = "acme/alice//".parse().unwrap();
//! let now = Timestamp::from_secs(1_700_000_000).unwrap();
//!
//! let episode = NewEpisode::new(scope.clone(), EpisodeKind::UserMessage, "I live in Paris")
//!     .unwrap()
//!     .into_episode(EpisodeId::generate_at(now).unwrap(), now);
//!
//! let draft =
//!     NewMemory::from_episodes(scope, MemoryKind::Semantic, "Alice lives in Paris", &[episode])
//!         .unwrap();
//! let memory = draft.into_memory(MemoryId::generate_at(now).unwrap(), now);
//!
//! assert!(memory.believed_at(now));
//! assert!(memory.valid_at(now));
//! ```

use crate::error::{Error, Result};
use crate::types::string_enum::string_enum;
use crate::types::{Episode, EpisodeId, Interval, MemoryId, Scope, Timestamp, Trust};

string_enum! {
    /// What sort of knowledge a memory holds.
    MemoryKind, "memory kind", InvalidMemory {
        /// A fact about the world or the user, true until superseded.
        Semantic = "semantic",
        /// A summary of something that happened.
        Episodic = "episodic",
        /// How to do something: a rule, a procedure, a standing preference
        /// about behavior.
        Procedural = "procedural",
        /// A higher-level conclusion drawn from other memories.
        Reflection = "reflection",
    }
}

string_enum! {
    /// Where a memory is in its life cycle.
    MemoryStatus, "memory status", InvalidMemory {
        /// Current, and eligible for recall.
        Active = "active",
        /// Replaced by a newer memory. Kept as history, and never final: see
        /// [`Memory::supersede`].
        Superseded = "superseded",
        /// Retired from recall but kept, for example after decaying below the
        /// forgetting threshold.
        Archived = "archived",
        /// Held back for review because its source or content looks unsafe.
        Quarantined = "quarantined",
    }
}

impl MemoryStatus {
    /// Whether [`Memory::set_status`] may move a memory from `self` to `to`.
    ///
    /// Active, archived, and quarantined memories can move among themselves.
    /// A superseded memory never moves again, and nothing is moved *into*
    /// superseded this way, because that transition also needs a successor and
    /// a time: use [`Memory::supersede`].
    pub const fn can_move_to(self, to: Self) -> bool {
        matches!(
            (self, to),
            (Self::Active, Self::Archived | Self::Quarantined)
                | (Self::Archived, Self::Active | Self::Quarantined)
                | (Self::Quarantined, Self::Active | Self::Archived)
        )
    }
}

fn validate_text(text: &str) -> Result<()> {
    if text.trim().is_empty() {
        return Err(Error::InvalidMemory("text must not be empty".to_owned()));
    }
    let len = text.len();
    if len > Memory::MAX_TEXT_BYTES {
        return Err(Error::InvalidMemory(format!(
            "text is {len} bytes long; the limit is {}",
            Memory::MAX_TEXT_BYTES
        )));
    }
    Ok(())
}

fn check_score(name: &str, value: f32, max: f32) -> Result<f32> {
    if (0.0..=max).contains(&value) {
        Ok(value)
    } else {
        Err(Error::InvalidMemory(format!(
            "{name} must be between 0 and {max}, got {value}"
        )))
    }
}

/// A memory as produced by extraction, before the store has stamped it.
///
/// A draft always has provenance. Either it was derived from episodes
/// ([`from_episodes`](Self::from_episodes)), in which case its trust is the
/// weakest of its sources, or it was written by the operator
/// ([`authored`](Self::authored)), in which case it has no sources and is
/// trusted at the system level. There is no third way in.
#[derive(Debug, Clone, PartialEq)]
pub struct NewMemory {
    scope: Scope,
    kind: MemoryKind,
    text: String,
    importance: f32,
    confidence: f32,
    trust: Trust,
    sources: Vec<EpisodeId>,
    valid: Interval,
}

impl NewMemory {
    /// Importance given to a draft until it is set explicitly.
    pub const DEFAULT_IMPORTANCE: f32 = 5.0;

    /// Confidence given to a draft until it is set explicitly: no doubt
    /// recorded.
    pub const DEFAULT_CONFIDENCE: f32 = 1.0;

    /// Creates a draft derived from `sources`.
    ///
    /// The memory's trust is the weakest trust among its sources. Its valid
    /// time starts at the earliest time any source event occurred, and is open
    /// ended; change that with [`with_valid`](Self::with_valid). Duplicate
    /// sources are listed once, in first-seen order.
    ///
    /// The memory's scope may be broader than a source's scope (a fact learned
    /// in one session can belong to the whole user), but never narrower or
    /// different: every source's scope must be covered by `scope`. That keeps
    /// a memory from mixing content across tenants, users, or agents.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidMemory`] if the text is empty or too long, if
    /// there are no sources, or if `scope` does not cover a source's scope.
    pub fn from_episodes(
        scope: Scope,
        kind: MemoryKind,
        text: impl Into<String>,
        sources: &[Episode],
    ) -> Result<Self> {
        let text = text.into();
        validate_text(&text)?;

        if sources.is_empty() {
            return Err(Error::InvalidMemory(
                "a memory derived from episodes needs at least one source episode".to_owned(),
            ));
        }
        for episode in sources {
            if !scope.covers(episode.scope()) {
                return Err(Error::InvalidMemory(format!(
                    "memory scope {scope} does not cover scope {} of source episode {}",
                    episode.scope(),
                    episode.id()
                )));
            }
        }

        // Sources are known to be non-empty here, so these fallbacks never
        // apply; they are the conservative answers if that ever changed.
        let trust = Trust::weakest(sources.iter().map(|e| Trust::of_episode(e.kind())))
            .unwrap_or(Trust::External);
        let valid = sources
            .iter()
            .map(Episode::occurred_at)
            .min()
            .map_or(Interval::ALWAYS, Interval::since);

        let mut ids: Vec<EpisodeId> = Vec::with_capacity(sources.len());
        for episode in sources {
            if !ids.contains(&episode.id()) {
                ids.push(episode.id());
            }
        }

        Ok(Self {
            scope,
            kind,
            text,
            importance: Self::DEFAULT_IMPORTANCE,
            confidence: Self::DEFAULT_CONFIDENCE,
            trust,
            sources: ids,
            valid,
        })
    }

    /// Creates a draft written by the operator rather than derived from
    /// episodes.
    ///
    /// It has no sources, is trusted at [`Trust::System`], and is valid for
    /// all time unless changed with [`with_valid`](Self::with_valid).
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidMemory`] if the text is empty or too long.
    pub fn authored(scope: Scope, kind: MemoryKind, text: impl Into<String>) -> Result<Self> {
        let text = text.into();
        validate_text(&text)?;
        Ok(Self {
            scope,
            kind,

            text,
            importance: Self::DEFAULT_IMPORTANCE,
            confidence: Self::DEFAULT_CONFIDENCE,
            trust: Trust::System,
            sources: Vec::new(),
            valid: Interval::ALWAYS,
        })
    }

    /// Sets how important the memory is, from 0 to
    /// [`Memory::MAX_IMPORTANCE`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidMemory`] if the value is outside that range or
    /// not a number.
    pub fn with_importance(mut self, importance: f32) -> Result<Self> {
        self.importance = check_score("importance", importance, Memory::MAX_IMPORTANCE)?;
        Ok(self)
    }

    /// Sets how confident the extractor was, from 0 to 1.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidMemory`] if the value is outside that range or
    /// not a number.
    pub fn with_confidence(mut self, confidence: f32) -> Result<Self> {
        self.confidence = check_score("confidence", confidence, 1.0)?;
        Ok(self)
    }

    /// Sets when the memory is true in the world.
    #[must_use]
    pub fn with_valid(mut self, valid: Interval) -> Self {
        self.valid = valid;
        self
    }

    /// The scope the memory will belong to.
    pub fn scope(&self) -> &Scope {
        &self.scope
    }

    /// The kind of knowledge.
    pub fn kind(&self) -> MemoryKind {
        self.kind
    }

    /// The text.
    pub fn text(&self) -> &str {
        &self.text
    }

    /// How important the memory is, from 0 to 10.
    pub fn importance(&self) -> f32 {
        self.importance
    }

    /// How confident the extractor was, from 0 to 1.
    pub fn confidence(&self) -> f32 {
        self.confidence
    }

    /// The trust level, computed from the sources.
    pub fn trust(&self) -> Trust {
        self.trust
    }

    /// The episodes the memory was derived from. Empty only for authored
    /// memories.
    pub fn sources(&self) -> &[EpisodeId] {
        &self.sources
    }

    /// When the memory is true in the world.
    pub fn valid(&self) -> Interval {
        self.valid
    }

    /// Stamps the draft into a stored [`Memory`].
    ///
    /// The new memory is [`Active`](MemoryStatus::Active), was last accessed
    /// when it was recorded, and has never been accessed since.
    pub fn into_memory(self, id: MemoryId, recorded_at: Timestamp) -> Memory {
        Memory {
            id,
            scope: self.scope,
            kind: self.kind,
            text: self.text,
            importance: self.importance,
            confidence: self.confidence,
            trust: self.trust,
            status: MemoryStatus::Active,
            valid: self.valid,
            recorded_at,
            superseded_at: None,
            superseded_by: None,
            sources: self.sources,
            last_accessed: recorded_at,
            access_count: 0,
        }
    }
}

/// A stored memory.
///
/// The text, scope, kind, scores, trust, sources, and recording time never
/// change. What does change is bookkeeping, through methods that check the
/// move is legal before making it: [`touch`](Self::touch),
/// [`supersede`](Self::supersede), and [`set_status`](Self::set_status). A
/// refused change leaves the memory exactly as it was.
///
/// The embedding of the text is derived data and lives next to the memory in
/// the store, not in this type.
#[derive(Debug, Clone, PartialEq)]
pub struct Memory {
    id: MemoryId,
    scope: Scope,
    kind: MemoryKind,
    text: String,
    importance: f32,
    confidence: f32,
    trust: Trust,
    status: MemoryStatus,
    valid: Interval,
    recorded_at: Timestamp,
    superseded_at: Option<Timestamp>,
    superseded_by: Option<MemoryId>,
    sources: Vec<EpisodeId>,
    last_accessed: Timestamp,
    access_count: u32,
}

impl Memory {
    /// The longest text a memory can hold, in bytes (16 KiB).
    pub const MAX_TEXT_BYTES: usize = 16 * 1024;

    /// The largest importance score.
    pub const MAX_IMPORTANCE: f32 = 10.0;

    /// The memory's id.
    pub fn id(&self) -> MemoryId {
        self.id
    }

    /// The scope the memory belongs to
    pub fn scope(&self) -> &Scope {
        &self.scope
    }

    /// The kind of knowledge.
    pub fn kind(&self) -> MemoryKind {
        self.kind
    }

    /// The text, exactly as it was supplied.
    pub fn text(&self) -> &str {
        &self.text
    }

    /// How important the memory is, from 0 to 10.
    pub fn importance(&self) -> f32 {
        self.importance
    }

    /// How confident the extractor was, from 0 to 1.
    pub fn confidence(&self) -> f32 {
        self.confidence
    }

    /// The trust level, computed from the sources.
    pub fn trust(&self) -> Trust {
        self.trust
    }

    /// Where the memory is in its life cycle.
    pub fn status(&self) -> MemoryStatus {
        self.status
    }

    /// When the memory is true in the world (valid time).
    pub fn valid(&self) -> Interval {
        self.valid
    }

    /// When the system recorded the memory (record time).
    pub fn recorded_at(&self) -> Timestamp {
        self.recorded_at
    }

    /// When the memory stopped being treated as current, if it was superseded.
    pub fn superseded_at(&self) -> Option<Timestamp> {
        self.superseded_at
    }

    /// The memory that replaced this one, if it was superseded.
    pub fn superseded_by(&self) -> Option<MemoryId> {
        self.superseded_by
    }

    /// The episodes the memory was derived from.
    pub fn sources(&self) -> &[EpisodeId] {
        &self.sources
    }

    /// When the memory was last recalled, or recorded if it never was.
    pub fn last_accessed(&self) -> Timestamp {
        self.last_accessed
    }

    /// How many times the memory has been recalled.
    pub fn access_count(&self) -> u32 {
        self.access_count
    }

    /// Whether the fact was true in the world at `at`, according to what we
    /// know today.
    ///
    /// This uses the valid interval as it stands now. If the memory was
    /// superseded, that interval has been closed at the moment the successor's
    /// fact began.
    pub fn valid_at(&self, at: Timestamp) -> bool {
        self.valid.contains(at)
    }

    /// Whether we held this memory as current at record time `at`.
    ///
    /// True from the moment it was recorded until it was superseded. Archiving
    /// and quarantine do not change what we believed, so they are ignored.
    ///
    /// This says which memories were held at that time. It does not rewind the
    /// valid interval: that is the interval as we know it today.
    pub fn believed_at(&self, at: Timestamp) -> bool {
        self.recorded_at <= at && self.superseded_at.is_none_or(|ended| at < ended)
    }

    /// Notes that the memory was recalled at `at`.
    ///
    /// Counts the access (saturating) and moves the last-accessed time
    /// forward. If `at` is earlier than the last access, which a skewed clock
    /// can cause, the last-accessed time is left alone.
    pub fn touch(&mut self, at: Timestamp) {
        self.access_count = self.access_count.saturating_add(1);
        self.last_accessed = self.last_accessed.max(at);
    }

    /// Marks the memory as replaced by `successor`.
    ///
    /// `valid_end` is when the old fact stopped being true in the world,
    /// normally the instnat the successor's validity begins. It closes the
    /// valid interval, but never extends one that already ended earlier. `at`
    /// is the record time of the change.
    ///
    /// The change is all or nothing. If anything is refused, the memory is
    /// left untouched.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidMemory`] if the memory is not active, if
    /// `successor` is the memory itself, or if `at` is earlier than the time
    /// the memory was recorded. Returns [`Error::InvalidInterval`] if
    /// `valid_end` is before the start of the valid interval.
    pub fn supersede(
        &mut self,
        successor: MemoryId,
        valid_end: Timestamp,
        at: Timestamp,
    ) -> Result<()> {
        if self.status != MemoryStatus::Active {
            return Err(Error::InvalidMemory(format!(
                "only an active memory can be superseded, but this one is {}",
                self.status
            )));
        }
        if successor == self.id {
            return Err(Error::InvalidMemory(
                "a memory cannot supersede itself".to_owned(),
            ));
        }
        if at < self.recorded_at {
            return Err(Error::InvalidMemory(
                "a memory cannot be superseded before it was recorded".to_owned(),
            ));
        }

        let end = self
            .valid
            .end()
            .map_or(valid_end, |existing| existing.min(valid_end));
        let valid = self.valid.with_end(end)?;

        self.valid = valid;
        self.status = MemoryStatus::Superseded;
        self.superseded_at = Some(at);
        self.superseded_by = Some(successor);
        Ok(())
    }

    /// Moves the memory between the active, archived, and quarantined states.
    ///
    /// See [`MemoryStatus::can_move_to`] for the allowed moves.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidMemory`] if the move is not allowed. The memory
    /// is left unchanged.
    pub fn set_status(&mut self, to: MemoryStatus) -> Result<()> {
        if !self.status.can_move_to(to) {
            return Err(Error::InvalidMemory(format!(
                "cannot move memory from {} to {to}",
                self.status
            )));
        }
        self.status = to;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{EpisodeKind, NewEpisode};

    fn secs(s: i64) -> Timestamp {
        Timestamp::from_secs(s).unwrap()
    }

    fn scope(key: &str) -> Scope {
        key.parse().unwrap()
    }

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-6
    }

    fn episode(scope_key: &str, kind: EpisodeKind, occurred: i64) -> Episode {
        let now = secs(1_000);
        NewEpisode::new(scope(scope_key), kind, "some content")
            .unwrap()
            .with_occurred_at(secs(occurred))
            .into_episode(EpisodeId::generate_at(now).unwrap(), now)
    }

    fn draft() -> NewMemory {
        let source = episode("acme/alice//", EpisodeKind::UserMessage, 10);
        NewMemory::from_episodes(
            scope("acme/alice//"),
            MemoryKind::Semantic,
            "Alice lives in Paris",
            &[source],
        )
        .unwrap()
    }

    fn memory() -> Memory {
        draft().into_memory(MemoryId::generate_at(secs(100)).unwrap(), secs(100))
    }

    fn authored(text: &str) -> Result<NewMemory> {
        NewMemory::authored(scope("acme/alice//"), MemoryKind::Semantic, text)
    }

    #[test]
    fn memories_are_send_sync_static() {
        fn check<T: Send + Sync + 'static>(_: &T) {}
        check(&draft());
        check(&memory());
    }

    #[test]
    fn kinds_and_statuses_round_trip_their_names() {
        assert_eq!(MemoryKind::ALL.len(), 4);
        assert_eq!(MemoryStatus::ALL.len(), 4);
        for &kind in MemoryKind::ALL {
            assert_eq!(kind.as_str().parse::<MemoryKind>().unwrap(), kind);
        }
        for &status in MemoryStatus::ALL {
            assert_eq!(status.to_string().parse::<MemoryStatus>().unwrap(), status);
        }
    }

    #[test]
    fn unknown_names_are_rejected_with_clear_messages() {
        assert_eq!(
            "fact".parse::<MemoryKind>().unwrap_err().to_string(),
            r#"invalid memory: unknown memory kind "fact""#
        );
        assert_eq!(
            "gone".parse::<MemoryStatus>().unwrap_err().to_string(),
            r#"invalid memory: unknown memory status "gone""#
        );
    }

    #[test]
    fn status_moves_follow_the_table() {
        use MemoryStatus::{Active, Archived, Quarantined};

        let allowed = [
            (Active, Archived),
            (Active, Quarantined),
            (Archived, Active),
            (Archived, Quarantined),
            (Quarantined, Active),
            (Quarantined, Archived),
        ];
        for &from in MemoryStatus::ALL {
            for &to in MemoryStatus::ALL {
                assert_eq!(
                    from.can_move_to(to),
                    allowed.contains(&(from, to)),
                    "{from} -> {to}"
                );
            }
        }
    }

    #[test]
    fn trust_is_the_weakest_source() {
        let user = episode("acme/alice//", EpisodeKind::UserMessage, 10);
        let document = episode("acme/alice//", EpisodeKind::Document, 20);
        let trust_of = |sources: &[Episode]| {
            NewMemory::from_episodes(scope("acme/alice//"), MemoryKind::Semantic, "fact", sources)
                .unwrap()
                .trust()
        };

        assert_eq!(trust_of(std::slice::from_ref(&user)), Trust::User);
        assert_eq!(trust_of(std::slice::from_ref(&document)), Trust::External);
        assert_eq!(trust_of([user, document].as_ref()), Trust::External);
    }

    #[test]
    fn derived_memories_need_at_least_one_source() {
        let err =
            NewMemory::from_episodes(scope("acme/alice//"), MemoryKind::Semantic, "fact", &[])
                .unwrap_err();
        assert_eq!(
            err.to_string(),
            "invalid memory: a memory derived from episodes needs at least one source episode"
        );
    }

    #[test]
    fn memory_scope_must_cover_every_source_scope() {
        let from_session = episode("acme/alice/bot/s1", EpisodeKind::UserMessage, 10);
        let derive_at = |key: &str, source: &Episode| {
            NewMemory::from_episodes(
                scope(key),
                MemoryKind::Semantic,
                "fact",
                std::slice::from_ref(source),
            )
        };

        for broader_or_equal in [
            "acme/alice/bot/s1",
            "acme/alice/bot/",
            "acme/alice//s1",
            "acme/alice//",
        ] {
            assert!(derive_at(broader_or_equal, &from_session).is_ok());
        }
        for unrelated in [
            "acme/alice/bot/s2",
            "acme/alice/other/",
            "acme/bob//",
            "globex/alice//",
        ] {
            assert!(derive_at(unrelated, &from_session).is_err());
        }

        let from_whole_user = episode("acme/alice//", EpisodeKind::UserMessage, 10);
        assert!(derive_at("acme/alice/bot/", &from_whole_user).is_err());
    }

    #[test]
    fn scope_errors_name_both_scopes() {
        let source = episode("acme/alice/bot/s1", EpisodeKind::UserMessage, 10);
        let err =
            NewMemory::from_episodes(scope("acme/bob//"), MemoryKind::Semantic, "fact", &[source])
                .unwrap_err();
        assert!(
            err.to_string().starts_with(
                "invalid memory: memory scope acme/bob// does not cover scope \
                acme/alice/bot/s1 of source episode "
            ),
            "{err}"
        );
    }

    #[test]
    fn valid_time_starts_at_the_earliest_source_event() {
        let late = episode("acme/alice//", EpisodeKind::UserMessage, 80);
        let early = episode("acme/alice//", EpisodeKind::UserMessage, 50);
        let derived = NewMemory::from_episodes(
            scope("acme/alice//"),
            MemoryKind::Semantic,
            "fact",
            &[late, early],
        )
        .unwrap();
        assert_eq!(derived.valid(), Interval::since(secs(50)));
    }

    #[test]
    fn duplicate_sources_are_listed_once_in_first_seen_order() {
        let first = episode("acme/alice//", EpisodeKind::UserMessage, 10);
        let second = episode("acme/alice//", EpisodeKind::UserMessage, 20);
        let derived = NewMemory::from_episodes(
            scope("acme/alice//"),
            MemoryKind::Semantic,
            "fact",
            &[first.clone(), second.clone(), first.clone()],
        )
        .unwrap();
        assert_eq!(derived.sources(), &[first.id(), second.id()]);
    }

    #[test]
    fn authored_memories_are_system_trusted_and_sourceless() {
        let written = NewMemory::authored(
            scope("acme/alice//"),
            MemoryKind::Procedural,
            "Always answer in French",
        )
        .unwrap();
        assert_eq!(written.trust(), Trust::System);
        assert_eq!(written.sources(), []);
        assert_eq!(written.valid(), Interval::ALWAYS);
    }

    #[test]
    fn text_must_be_present_and_bounded() {
        for bad in ["", "   ", "\n"] {
            assert_eq!(
                authored(bad).unwrap_err().to_string(),
                "invalid memory: text must not be empty"
            );
        }

        assert!(authored(&"x".repeat(Memory::MAX_TEXT_BYTES)).is_ok());
        assert_eq!(
            authored(&"x".repeat(Memory::MAX_TEXT_BYTES + 1))
                .unwrap_err()
                .to_string(),
            "invalid memory: text is 16385 bytes long; the limit is 16384"
        );
    }

    #[test]
    fn scores_are_defaulted_and_validated() {
        let fresh = draft();
        assert!(approx(fresh.importance(), NewMemory::DEFAULT_IMPORTANCE));
        assert!(approx(fresh.confidence(), NewMemory::DEFAULT_CONFIDENCE));

        assert!(approx(
            draft().with_importance(0.0).unwrap().importance(),
            0.0
        ));
        assert!(approx(
            draft().with_importance(10.0).unwrap().importance(),
            10.0
        ));
        for bad in [-0.1, 10.1, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(draft().with_importance(bad).is_err(), "importance {bad}");
        }

        assert!(approx(
            draft().with_confidence(0.25).unwrap().confidence(),
            0.25
        ));
        for bad in [-0.01, 1.01, f32::NAN] {
            assert!(draft().with_confidence(bad).is_err(), "confidence {bad}");
        }
    }

    #[test]
    fn score_errors_name_the_score_and_its_limit() {
        assert_eq!(
            draft().with_importance(11.0).unwrap_err().to_string(),
            "invalid memory: importance must be between 0 and 10, got 11"
        );
        assert_eq!(
            draft().with_confidence(1.5).unwrap_err().to_string(),
            "invalid memory: confidence must be between 0 and 1, got 1.5"
        );
    }

    #[test]
    fn valid_time_can_be_overridden() {
        let window = Interval::between(secs(5), secs(15)).unwrap();
        assert_eq!(draft().with_valid(window).valid(), window);
    }

    #[test]
    fn into_memory_starts_active_and_untouched() {
        let new = draft()
            .with_importance(7.0)
            .unwrap()
            .with_confidence(0.5)
            .unwrap();
        let sources = new.sources().to_vec();
        let id = MemoryId::generate_at(secs(100)).unwrap();
        let stored = new.into_memory(id, secs(100));

        assert_eq!(stored.id(), id);
        assert_eq!(stored.scope(), &scope("acme/alice//"));
        assert_eq!(stored.kind(), MemoryKind::Semantic);
        assert_eq!(stored.text(), "Alice lives in Paris");
        assert_eq!(stored.trust(), Trust::User);
        assert_eq!(stored.sources(), sources.as_slice());
        assert!(approx(stored.importance(), 7.0));
        assert!(approx(stored.confidence(), 0.5));
        assert_eq!(stored.valid(), Interval::since(secs(10)));

        assert_eq!(stored.status(), MemoryStatus::Active);
        assert_eq!(stored.recorded_at(), secs(100));
        assert_eq!(stored.last_accessed(), secs(100));
        assert_eq!(stored.access_count(), 0);
        assert_eq!(stored.superseded_at(), None);
        assert_eq!(stored.superseded_by(), None);
    }

    #[test]
    fn touch_counts_accesses_and_never_moves_time_backwards() {
        let mut m = memory();
        m.touch(secs(150));
        assert_eq!(m.access_count(), 1);
        assert_eq!(m.last_accessed(), secs(150));

        m.touch(secs(120));
        assert_eq!(m.access_count(), 2);
        assert_eq!(m.last_accessed(), secs(150));
    }

    #[test]
    fn supersede_closes_valid_time_and_records_the_successor() {
        let mut m = memory();
        let successor = MemoryId::generate_at(secs(200)).unwrap();
        m.supersede(successor, secs(150), secs(200)).unwrap();

        assert_eq!(m.status(), MemoryStatus::Superseded);
        assert_eq!(m.valid(), Interval::between(secs(10), secs(150)).unwrap());
        assert_eq!(m.superseded_at(), Some(secs(200)));
        assert_eq!(m.superseded_by(), Some(successor));
    }

    #[test]
    fn supersede_never_extends_a_validity_that_already_ended() {
        let ended = Interval::between(secs(10), secs(120)).unwrap();
        let mut m = draft()
            .with_valid(ended)
            .into_memory(MemoryId::generate_at(secs(100)).unwrap(), secs(100));

        let successor = MemoryId::generate_at(secs(200)).unwrap();
        m.supersede(successor, secs(150), secs(200)).unwrap();
        assert_eq!(m.valid().end(), Some(secs(120)));
    }

    #[test]
    fn refused_supersession_changes_nothing() {
        let successor = MemoryId::generate_at(secs(200)).unwrap();
        let fresh = memory();

        let mut archived = fresh.clone();
        archived.set_status(MemoryStatus::Archived).unwrap();
        let before = archived.clone();
        let err = archived
            .supersede(successor, secs(150), secs(200))
            .unwrap_err();
        assert_eq!(
            err.to_string(),
            "invalid memory: only an active memory can be superseded, but this one is archived"
        );
        assert_eq!(archived, before);

        let mut m = fresh.clone();
        let own_id = m.id();
        assert!(m.supersede(own_id, secs(150), secs(200)).is_err());
        assert!(m.supersede(successor, secs(150), secs(99)).is_err());
        assert!(m.supersede(successor, secs(5), secs(200)).is_err());
        assert_eq!(m, fresh);
    }

    #[test]
    fn superseded_memories_cannot_change_status() {
        let mut m = memory();
        let successor = MemoryId::generate_at(secs(200)).unwrap();
        m.supersede(successor, secs(150), secs(200)).unwrap();

        for &to in MemoryStatus::ALL {
            assert!(m.set_status(to).is_err(), "superseded -> {to}");
        }
        assert_eq!(m.status(), MemoryStatus::Superseded);
    }

    #[test]
    fn set_status_moves_among_the_reversible_states() {
        let mut m = memory();
        m.set_status(MemoryStatus::Archived).unwrap();
        m.set_status(MemoryStatus::Quarantined).unwrap();
        m.set_status(MemoryStatus::Active).unwrap();
        assert_eq!(m.status(), MemoryStatus::Active);

        let err = m.set_status(MemoryStatus::Superseded).unwrap_err();
        assert_eq!(
            err.to_string(),
            "invalid memory: cannot move memory from active to superseded"
        );
        assert_eq!(m.status(), MemoryStatus::Active);
    }

    #[test]
    fn believed_at_spans_recording_to_supersession() {
        let mut m = memory();
        assert!(!m.believed_at(secs(99)));
        assert!(m.believed_at(secs(100)));
        assert!(m.believed_at(secs(10_000)));

        let successor = MemoryId::generate_at(secs(200)).unwrap();
        m.supersede(successor, secs(150), secs(200)).unwrap();
        assert!(m.believed_at(secs(199)));
        assert!(!m.believed_at(secs(200)));
    }

    #[test]
    fn archiving_does_not_change_what_we_believed() {
        let mut m = memory();
        m.set_status(MemoryStatus::Archived).unwrap();
        assert!(m.believed_at(secs(150)));
    }

    #[test]
    fn valid_at_follows_the_half_open_valid_interval() {
        let window = Interval::between(secs(10), secs(20)).unwrap();
        let m = draft()
            .with_valid(window)
            .into_memory(MemoryId::generate_at(secs(100)).unwrap(), secs(100));

        assert!(!m.valid_at(secs(9)));
        assert!(m.valid_at(secs(10)));
        assert!(m.valid_at(secs(19)));
        assert!(!m.valid_at(secs(20)));
    }

    #[test]
    fn the_two_clocks_answer_different_questions() {
        // Paris is true in the world from t=100 and recorded at t=110. Berlin
        // is true from t=200, recorded at t=210, and replaces Paris.
        let make = |text: &str, valid_from: i64, recorded: i64| {
            authored(text)
                .unwrap()
                .with_valid(Interval::since(secs(valid_from)))
                .into_memory(
                    MemoryId::generate_at(secs(recorded)).unwrap(),
                    secs(recorded),
                )
        };
        let mut paris = make("Alice lives in Paris", 100, 110);
        let berlin = make("Alice lives in Berlin", 200, 210);
        let berlin_begins = berlin.valid().start().unwrap();
        paris
            .supersede(berlin.id(), berlin_begins, berlin.recorded_at())
            .unwrap();

        // What was true in the world, as we know it today?
        assert!(paris.valid_at(secs(150)) && !berlin.valid_at(secs(150)));
        assert!(!paris.valid_at(secs(250)) && berlin.valid_at(secs(250)));

        // What did we hold as current at the time?
        assert!(paris.believed_at(secs(150)) && !berlin.believed_at(secs(150)));
        assert!(!paris.believed_at(secs(250)) && berlin.believed_at(secs(250)));

        // What is current today?
        assert_eq!(paris.status(), MemoryStatus::Superseded);
        assert_eq!(berlin.status(), MemoryStatus::Active);
    }
}
