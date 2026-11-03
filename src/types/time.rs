//! Time primitives: instants and validity intervals.
//!
//! [`Timestamp`] is a point in time, stored as microseconds since the Unix
//! epoch. [`Interval`] is a half-open range of timestamps, used for the
//! valid-time axis of bi-temporal facts (see `docs/ARCHITECTURE.md`)
//!
//! Nothing in this module reads the system clock by itself. Code that needs
//! "now" will get it from a `Clock`, so that tests can control time.

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::error::{Error, Result};

/// A point in time: microseconds since the Unix epoch, in UTC.
///
/// Stored as an `i64`, which covers roughly 292,000 years on either side of
/// 1970. Ordering is chronological.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Timestamp(i64);

impl Timestamp {
    /// The Unix epoch, 1970-01-01T00:00:00Z.
    pub const EPOCH: Self = Self(0);

    /// The earliest representable timestamp.
    pub const MIN: Self = Self(i64::MIN);

    /// The latest representable timestamp.
    pub const MAX: Self = Self(i64::MAX);

    /// Creates a timestamp from microseconds since the Unix epoch.
    pub const fn from_micros(micros: i64) -> Self {
        Self(micros)
    }

    /// Creates a timestamp from milliseconds since the Unix epoch.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidTimestamp`] if the value is out of range.
    pub fn from_millis(millis: i64) -> Result<Self> {
        millis
            .checked_mul(1_000)
            .map(Self)
            .ok_or_else(|| out_of_range("milliseconds", millis))
    }

    /// Creates a timestamp from seconds since the Unix epoch.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidTimestamp`] if the value is out of range.
    pub fn from_secs(secs: i64) -> Result<Self> {
        secs.checked_mul(1_000_000)
            .map(Self)
            .ok_or_else(|| out_of_range("seconds", secs))
    }

    /// Converts a [`SystemTime`], truncating anything finer than a microsecond.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidTimestamp`] if the time is outside the
    /// representable range.
    pub fn from_system_time(time: SystemTime) -> Result<Self> {
        let micros = match time.duration_since(UNIX_EPOCH) {
            Ok(after) => i64::try_from(after.as_micros()).ok(),
            Err(before) => i64::try_from(before.duration().as_micros())
                .ok()
                .and_then(i64::checked_neg),
        };
        micros.map(Self).ok_or_else(|| {
            Error::InvalidTimestamp("system time is outside the representable range".to_owned())
        })
    }

    /// Microseconds since the Unix epoch.
    pub const fn to_micros(self) -> i64 {
        self.0
    }

    /// Milliseconds since the Unix epoch, rounded toward negative infinity.
    pub const fn to_millis(self) -> i64 {
        self.0.div_euclid(1_000)
    }

    /// Seconds since the Unix epoch, rounded toward negative infinity.
    pub const fn to_secs(self) -> i64 {
        self.0.div_euclid(1_000_000)
    }

    /// Time elapsed from `earlier` to `self`, or `None` if `earlier` is later.
    pub fn checked_duration_since(self, earlier: Self) -> Option<Duration> {
        let diff = i128::from(self.0) - i128::from(earlier.0);
        u64::try_from(diff).ok().map(Duration::from_micros)
    }

    /// Like [`Timestamp::checked_duration_since`], but returns zero instead of
    /// `None`. Handy for ages, which should never be negative when clocks skew.
    pub fn saturating_duration_since(self, earlier: Self) -> Duration {
        self.checked_duration_since(earlier).unwrap_or_default()
    }

    /// Adds a duration, or returns `None` on overflow.
    ///
    /// Anything finer than a microsecond in `delta` is truncated.
    pub fn checked_add(self, delta: Duration) -> Option<Self> {
        let micros = i64::try_from(delta.as_micros()).ok()?;
        self.0.checked_add(micros).map(Self)
    }

    /// Subtracts a duration, or returns `None` on overflow.
    ///
    /// Anything finer than a microsecond in `delta` is truncated.
    pub fn checked_sub(self, delta: Duration) -> Option<Self> {
        let micros = i64::try_from(delta.as_micros()).ok()?;
        self.0.checked_sub(micros).map(Self)
    }
}

fn out_of_range(unit: &str, value: i64) -> Error {
    Error::InvalidTimestamp(format!("{value} {unit} since the epoch is out of range"))
}

/// A half-open range of timestamps, `[start, end)`.
///
/// A missing `start` means "since the beginning of time". A missing `end`
/// means "still ongoing". The default is [`Interval::ALWAYS`].
///
/// An interval with `start == end` is empty: it contains no instant and
/// overlaps nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Interval {
    start: Option<Timestamp>,
    end: Option<Timestamp>,
}

impl Interval {
    /// The interval with no bounds: every instant belongs to it.
    pub const ALWAYS: Self = Self {
        start: None,
        end: None,
    };

    /// The interval `[start, inf)`.
    pub const fn since(start: Timestamp) -> Self {
        Self {
            start: Some(start),
            end: None,
        }
    }

    /// The interval `(-inf, end)`.
    pub const fn until(end: Timestamp) -> Self {
        Self {
            start: None,
            end: Some(end),
        }
    }

    /// Creates an interval from optional bounds.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidInterval`] if both bounds are present and
    /// `start` is after `end`.
    pub fn new(start: Option<Timestamp>, end: Option<Timestamp>) -> Result<Self> {
        match (start, end) {
            (Some(s), Some(e)) if s > e => Err(Error::InvalidInterval(format!(
                "start {} is after end {} (microseconds since the epoch)",
                s.to_micros(),
                e.to_micros()
            ))),
            _ => Ok(Self { start, end }),
        }
    }

    /// The interval `[start, end)`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidInterval`] if `start` is after `end`.
    pub fn between(start: Timestamp, end: Timestamp) -> Result<Self> {
        Self::new(Some(start), Some(end))
    }

    /// The inclusive lower bound, or `None` if unbounded.
    pub fn start(&self) -> Option<Timestamp> {
        self.start
    }

    /// The exclusive upper bound, or `None` if ongoing.
    pub fn end(&self) -> Option<Timestamp> {
        self.end
    }

    /// Whether the interval contains no instants.
    pub fn is_empty(&self) -> bool {
        matches!((self.start, self.end), (Some(s), Some(e)) if s == e)
    }

    /// Whether `t` lies in `[start, end)`.
    pub fn contains(&self, t: Timestamp) -> bool {
        self.start.is_none_or(|s| s <= t) && self.end.is_none_or(|e| t < e)
    }

    /// Whether the two intervals share at least one instant.
    ///
    /// Adjacent intervals such as `[1, 2)` and `[2, 3)` do not overlap, and
    /// empty intervals overlap nothing.
    pub fn overlaps(&self, other: &Self) -> bool {
        !self.is_empty()
            && !other.is_empty()
            && starts_before_end(self.start, other.end)
            && starts_before_end(other.start, self.end)
    }

    /// The same interval, ended at `end`.
    ///
    /// This is how a fact gets superseded: its validity is closed at the
    /// instant its replacement begins.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidInterval`] if `end` is before the start.
    pub fn with_end(self, end: Timestamp) -> Result<Self> {
        Self::new(self.start, Some(end))
    }
}

/// `start < end`, treating a missing bound as infinitely far away.
fn starts_before_end(start: Option<Timestamp>, end: Option<Timestamp>) -> bool {
    match (start, end) {
        (Some(s), Some(e)) => s < e,
        _ => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ts(micros: i64) -> Timestamp {
        Timestamp::from_micros(micros)
    }

    #[test]
    fn micros_round_trip_and_order() {
        assert_eq!(ts(42).to_micros(), 42);
        assert!(ts(-1) < Timestamp::EPOCH);
        assert!(Timestamp::EPOCH < ts(1));
        assert!(Timestamp::MIN < Timestamp::MAX);
    }

    #[test]
    fn millis_and_secs_scale_correctly() {
        assert_eq!(Timestamp::from_millis(3).unwrap().to_micros(), 3_000);
        assert_eq!(Timestamp::from_secs(2).unwrap().to_micros(), 2_000_000);
    }

    #[test]
    fn out_of_range_conversions_are_errors() {
        assert!(Timestamp::from_millis(i64::MAX).is_err());
        assert!(Timestamp::from_secs(i64::MIN).is_err());
    }

    #[test]
    fn coarser_units_round_toward_negative_infinity() {
        assert_eq!(ts(1_999).to_millis(), 1);
        assert_eq!(ts(-1).to_millis(), -1);
        assert_eq!(ts(-1).to_secs(), -1);
        assert_eq!(ts(1_000_000).to_secs(), 1);
    }

    #[test]
    fn duration_arithmetic() {
        let t = ts(1_000);
        assert_eq!(t.checked_add(Duration::from_micros(500)), Some(ts(1_500)));
        assert_eq!(t.checked_sub(Duration::from_micros(500)), Some(ts(500)));
        assert_eq!(Timestamp::MAX.checked_add(Duration::from_micros(1)), None);
        assert_eq!(Timestamp::MIN.checked_sub(Duration::from_micros(1)), None);
        assert_eq!(t.checked_add(Duration::MAX), None);
    }

    #[test]
    fn duration_since_handles_direction_and_extremes() {
        let (early, late) = (ts(1_000), ts(4_000));
        assert_eq!(
            late.checked_duration_since(early),
            Some(Duration::from_micros(3_000))
        );
        assert_eq!(early.checked_duration_since(late), None);
        assert_eq!(early.saturating_duration_since(late), Duration::ZERO);
        assert_eq!(
            Timestamp::MAX.checked_duration_since(Timestamp::MIN),
            Some(Duration::from_micros(u64::MAX))
        );
    }

    #[test]
    fn converts_system_time_on_both_sides_of_the_epoch() {
        assert_eq!(
            Timestamp::from_system_time(UNIX_EPOCH).unwrap(),
            Timestamp::EPOCH
        );

        let after = UNIX_EPOCH + Duration::from_millis(1_500);
        assert_eq!(Timestamp::from_system_time(after).unwrap(), ts(1_500_000));

        let before = UNIX_EPOCH.checked_sub(Duration::from_secs(2)).unwrap();
        assert_eq!(Timestamp::from_system_time(before).unwrap(), ts(-2_000_000));
    }

    #[test]
    fn contains_is_half_open() {
        let i = Interval::between(ts(10), ts(20)).unwrap();
        assert!(!i.contains(ts(9)));
        assert!(i.contains(ts(10)));
        assert!(i.contains(ts(19)));
        assert!(!i.contains(ts(20)));
    }

    #[test]
    fn unbounded_sides_extend_forever() {
        assert!(Interval::ALWAYS.contains(Timestamp::MIN));
        assert!(Interval::ALWAYS.contains(Timestamp::MAX));

        let since = Interval::since(ts(5));
        assert!(!since.contains(ts(4)));
        assert!(since.contains(ts(5)));
        assert!(since.contains(Timestamp::MAX));

        let until = Interval::until(ts(5));
        assert!(until.contains(Timestamp::MIN));
        assert!(until.contains(ts(4)));
        assert!(!until.contains(ts(5)));
    }

    #[test]
    fn new_rejects_inverted_but_allows_empty() {
        assert!(Interval::between(ts(5), ts(4)).is_err());

        let empty = Interval::between(ts(5), ts(5)).unwrap();
        assert!(empty.is_empty());
        assert!(!empty.contains(ts(5)));
    }

    #[test]
    fn overlaps_respects_half_open_ends() {
        let a = Interval::between(ts(1), ts(2)).unwrap();
        let b = Interval::between(ts(2), ts(3)).unwrap();
        let c = Interval::between(ts(1), ts(3)).unwrap();
        assert!(!a.overlaps(&b), "adjacent intervals do not overlap");
        assert!(a.overlaps(&c));
        assert!(b.overlaps(&c));
        assert!(Interval::ALWAYS.overlaps(&a));
        assert!(Interval::since(ts(5)).overlaps(&Interval::until(ts(6))));
        assert!(!Interval::since(ts(5)).overlaps(&Interval::until(ts(5))));
    }

    #[test]
    fn empty_intervals_overlap_nothing() {
        let empty = Interval::between(ts(5), ts(5)).unwrap();
        assert!(!empty.overlaps(&Interval::ALWAYS));
        assert!(!Interval::ALWAYS.overlaps(&empty));
    }

    #[test]
    fn with_end_closes_an_open_interval() {
        let open = Interval::since(ts(10));
        let closed = open.with_end(ts(20)).unwrap();
        assert_eq!(closed.start(), Some(ts(10)));
        assert_eq!(closed.end(), Some(ts(20)));
        assert!(open.with_end(ts(9)).is_err());
    }
}
