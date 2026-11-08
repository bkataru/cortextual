//! Strongly typed identifiers
//!
//! Every persistent thing in cortextual has an id. Ids are UUID version 7
//! values: the high bits are a millisecond timestamp, so sorting ids sorts
//! them roughly by creation time, which keeps database indexes append-friendly.
//!
//! [`Id::generate`] stamps the current time. [`Id::generate_at`] stamps a time
//! you choose, so code driven by an injected clock can mint ids that agree with
//! it. Ids created within the same millisecond have no guaranteed order.
//!
//! An [`Id`] carries a marker type, so a [`MemoryId`] and an [`EpisodeId`] are
//! different types even though both wrap the same 128 bits. Mixing them up is a
//! compile error enforced by the type system:
//!
//! ```compile_fail,E0308
//! use cortextual::types::ids::{EpisodeId, MemoryId};
//!
//! fn wants_memory(_: MemoryId) {}
//!
//! let episode = EpisodeId::generate();
//! wants_memory(episode);
//! ```
//!
//! Ids print as hyphenated UUIDs and parse back from them:
//!
//! ```
//! use cortextual::types::ids::MemoryId;
//!
//! let id = MemoryId::generate();
//! let parsed: MemoryId = id.to_string().parse().unwrap();
//! assert_eq!(id, parsed);
//! ```

use std::cmp::Ordering;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::str::FromStr;

use uuid::Uuid;

use crate::error::{Error, Result};
use crate::types::time::Timestamp;

/// A v7 UUID carries its timestamp in 48 bits of milliseconds.
const MAX_V7_MILLIS: u64 = 1 << 48;

/// Type-level tag saying which kind of thing an [`Id`] identifies.
pub trait IdTag {
    /// Human-readable name of the kind, used in `Debug` output and errors.
    const NAME: &'static str;
}

/// Tag for [`MemoryId`]
pub enum MemoryTag {}

/// Tag for [`EpisodeId`]
pub enum EpisodeTag {}

/// Tag for [`EntityId`]
pub enum EntityTag {}

impl IdTag for MemoryTag {
    const NAME: &'static str = "memory";
}

impl IdTag for EpisodeTag {
    const NAME: &'static str = "episode";
}

impl IdTag for EntityTag {
    const NAME: &'static str = "entity";
}

/// Identifier of a memory.
pub type MemoryId = Id<MemoryTag>;

/// Identifier of an episode.
pub type EpisodeId = Id<EpisodeTag>;

/// Identifier of an entity.
pub type EntityId = Id<EntityTag>;

/// A UUID version 7 identifier, tagged with the kind of thing it names.
pub struct Id<K> {
    uuid: Uuid,
    _tag: PhantomData<fn() -> K>,
}

impl<K> Id<K> {
    /// Wraps an existing UUID.
    pub const fn from_uuid(uuid: Uuid) -> Self {
        Self {
            uuid,
            _tag: PhantomData,
        }
    }

    /// Creates a new id from the current time and fresh randomness.
    pub fn generate() -> Self {
        Self::from_uuid(Uuid::now_v7())
    }

    /// Creates a new id stamped with `at` instead of the current time, plus
    /// fresh randomness.
    ///
    /// Use this with an injected clock so ids agree with the rest of the
    /// system's notion of "now". The stamp has millisecond precision.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidTimestamp`] if `at` is before the Unix epoch or
    /// beyond the year 10889, the range is a 48-bit millisecond stamp can hold.
    pub fn generate_at(at: Timestamp) -> Result<Self> {
        v7_timestamp(at)
            .map(|stamp| Self::from_uuid(Uuid::new_v7(stamp)))
            .ok_or_else(|| {
                Error::InvalidTimestamp(format!(
                    "{} microseconds since the epoch is outside the range of an id",
                    at.to_micros()
                ))
            })
    }

    /// Wraps 16 raw bytes in big-endian UUID layout.
    pub const fn from_bytes(bytes: [u8; 16]) -> Self {
        Self::from_uuid(Uuid::from_bytes(bytes))
    }

    /// The underlying UUID.
    pub const fn as_uuid(&self) -> &Uuid {
        &self.uuid
    }

    /// The raw 16 bytes, suitable for storing as a database BLOB.
    pub const fn as_bytes(&self) -> &[u8; 16] {
        self.uuid.as_bytes()
    }

    /// The instant stamped into the id, at millisecond precision.
    ///
    /// `None` for UUIDs that carry no timestamp, such as the nil UUID.
    pub fn timestamp(&self) -> Option<Timestamp> {
        let (secs, nanos) = self.uuid.get_timestamp()?.to_unix();
        let micros = secs
            .checked_mul(1_000_000)?
            .checked_add(u64::from(nanos / 1_000))?;
        i64::try_from(micros).ok().map(Timestamp::from_micros)
    }
}

/// Converts our timestamp into the `uuid` crate's, or `None` if a v7 UUID
/// cannot represent it.
fn v7_timestamp(at: Timestamp) -> Option<uuid::Timestamp> {
    let millis = u64::try_from(at.to_millis()).ok()?;
    if millis >= MAX_V7_MILLIS {
        return None;
    }
    let nanos = u32::try_from((millis % 1_000) * 1_000_000).ok()?;
    Some(uuid::Timestamp::from_unix(
        uuid::NoContext,
        millis / 1_000,
        nanos,
    ))
}

impl<K: IdTag> Id<K> {
    /// Reads an id from a byte slice, such as a database BLOB.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidId`] if the lisce is not exactly 16 bytes long.
    pub fn from_slice(bytes: &[u8]) -> Result<Self> {
        Uuid::from_slice(bytes)
            .map(Self::from_uuid)
            .map_err(|e| Error::InvalidId {
                kind: K::NAME,
                input: format!("<{} bytes>", bytes.len()),
                reason: e.to_string(),
            })
    }
}

impl<K> Copy for Id<K> {}

impl<K> Clone for Id<K> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<K> Eq for Id<K> {}

impl<K> PartialEq for Id<K> {
    fn eq(&self, other: &Self) -> bool {
        self.uuid == other.uuid
    }
}

impl<K> Hash for Id<K> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.uuid.hash(state);
    }
}

impl<K> PartialOrd for Id<K> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<K> Ord for Id<K> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.uuid.cmp(&other.uuid)
    }
}

impl<K: IdTag> fmt::Debug for Id<K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}({})", K::NAME, self.uuid)
    }
}

impl<K> fmt::Display for Id<K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.uuid, f)
    }
}

impl<K: IdTag> FromStr for Id<K> {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Uuid::parse_str(s)
            .map(Self::from_uuid)
            .map_err(|e| Error::InvalidId {
                kind: K::NAME,
                input: s.to_owned(),
                reason: e.to_string(),
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn id_is_send_sync_static() {
        fn check<T: Send + Sync + 'static>(_: &T) {}
        check(&MemoryId::generate());
    }

    #[test]
    fn generated_ids_are_distinct() {
        assert_ne!(MemoryId::generate(), MemoryId::generate());
    }

    #[test]
    fn display_and_parse_round_trip() {
        let id = MemoryId::generate();
        let text = id.to_string();
        assert_eq!(text.len(), 36);
        assert_eq!(text.parse::<MemoryId>().unwrap(), id);
    }

    #[test]
    fn parse_failure_names_kind_and_input() {
        let err = "nope".parse::<MemoryId>().unwrap_err();
        assert!(err.to_string().starts_with(r#"invalid memory id "nope": "#));
    }

    #[test]
    fn bytes_round_trip() {
        let id = MemoryId::generate();
        assert_eq!(MemoryId::from_bytes(*id.as_bytes()), id);
        assert_eq!(MemoryId::from_slice(id.as_bytes()).unwrap(), id);
    }

    #[test]
    fn from_slice_rejects_wrong_length() {
        let err = EpisodeId::from_slice(&[1, 2, 3]).unwrap_err();
        assert!(err.to_string().starts_with("invalid episode id"));
    }

    #[test]
    fn ids_from_earlier_timestamps_sort_first() {
        let at = |secs: u64| {
            let ts = uuid::Timestamp::from_unix(uuid::NoContext, secs, 0);
            MemoryId::from_uuid(Uuid::new_v7(ts))
        };
        let earlier = at(1_700_000_000);
        let later = at(1_700_000_001);
        assert!(earlier < later);
        assert!(earlier.as_bytes() < later.as_bytes());
    }

    #[test]
    fn debug_output_names_the_kind() {
        let id = EntityId::from_uuid(Uuid::nil());
        assert_eq!(
            format!("{id:?}"),
            "entity(00000000-0000-0000-0000-000000000000)"
        );
    }

    #[test]
    fn ids_are_copy_and_hashable() {
        let id = MemoryId::generate();
        let mut set = std::collections::HashSet::new();
        set.insert(id);
        assert!(set.contains(&id));
    }

    #[test]
    fn generate_at_stamps_the_requested_millisecond() {
        let at = Timestamp::from_micros(1_700_000_000_123_456);
        let id = MemoryId::generate_at(at).unwrap();
        assert_eq!(
            id.timestamp(),
            Some(Timestamp::from_micros(1_700_000_000_123_000))
        );
    }

    #[test]
    fn generate_at_orders_ids_by_time() {
        let at = |ms: i64| MemoryId::generate_at(Timestamp::from_millis(ms).unwrap()).unwrap();
        assert!(at(1_000) < at(1_001));
        assert!(at(1_000) < at(2_000));
    }

    #[test]
    fn generate_at_covers_the_48_bit_range_and_rejects_the_rest() {
        let last = Timestamp::from_millis((1 << 48) - 1).unwrap();
        let beyond = Timestamp::from_millis(1 << 48).unwrap();

        assert!(MemoryId::generate_at(Timestamp::EPOCH).is_ok());
        assert!(MemoryId::generate_at(last).is_ok());
        assert!(MemoryId::generate_at(beyond).is_err());
        assert!(MemoryId::generate_at(Timestamp::from_micros(-1)).is_err());
        assert!(MemoryId::generate_at(Timestamp::MIN).is_err());
        assert!(MemoryId::generate_at(Timestamp::MAX).is_err());
    }

    #[test]
    fn timestamp_is_none_without_an_embedded_time() {
        assert_eq!(MemoryId::from_uuid(Uuid::nil()).timestamp(), None);
    }

    #[test]
    fn generate_stamps_the_current_time() {
        let after_2020 = Timestamp::from_secs(1_577_836_800).unwrap();
        assert!(MemoryId::generate().timestamp().unwrap() > after_2020);
    }
}
