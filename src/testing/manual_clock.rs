//! A clock that only moves when told to.

use std::sync::atomic::{AtomicI64, Ordering};
use std::time::Duration;

use crate::ports::Clock;
use crate::types::Timestamp;

/// A [`Clock`] whose time changes only when you call [`set`](Self::set),
/// [`advance`](Self::advance), or [`rewind`](Self::rewind).
///
/// Every method takes `&self`, so a test can keep one handle for moving time
/// while the code under test holds another, typically as `Arc<dyn Clock>`;
///
/// ```
/// use std::sync::Arc;
/// use std::time::Duration;
///
/// use cortextual::ports::Clock;
/// use cortextual::testing::ManualClock;
/// use cortextual::types::Timestamp;
///
/// let handle = Arc::new(ManualClock::new(Timestamp::EPOCH));
/// let for_code_under_test: Arc<dyn Clock> = handle.clone();
///
/// handle.advance(Duration::from_secs(3));
/// assert_eq!(for_code_under_test.now().to_secs(), 3);
/// ```
#[derive(Debug, Default)]
pub struct ManualClock {
    micros: AtomicI64,
}

impl ManualClock {
    /// Creates a clock showing `start`. `ManualClock::default()` starts at the
    /// Unix epoch.
    pub fn new(start: Timestamp) -> Self {
        Self {
            micros: AtomicI64::new(start.to_micros()),
        }
    }

    /// Jumps to `to`, forward or backward.
    pub fn set(&self, to: Timestamp) {
        self.micros.store(to.to_micros(), Ordering::SeqCst);
    }

    /// Moves time forward by `by`, saturating at [`Timestamp::MAX`].
    pub fn advance(&self, by: Duration) {
        let delta = micros_of(by);
        self.update(|now| now.saturating_add(delta));
    }

    /// Moves time backward by `by`, saturating at [`Timestamp::MIN`].
    ///
    /// Wall clocks really do go backwards (NTP corrections, manual changes),
    /// and this is how to test that code copes.
    pub fn rewind(&self, by: Duration) {
        let delta = micros_of(by);
        self.update(|now| now.saturating_sub(delta));
    }

    fn update(&self, change: impl Fn(i64) -> i64) {
        // The closure always returns `Some`, so this update cannot fail.
        let _ = self
            .micros
            .try_update(Ordering::SeqCst, Ordering::SeqCst, |now| Some(change(now)));
    }
}

fn micros_of(span: Duration) -> i64 {
    i64::try_from(span.as_micros()).unwrap_or(i64::MAX)
}

impl Clock for ManualClock {
    fn now(&self) -> Timestamp {
        Timestamp::from_micros(self.micros.load(Ordering::SeqCst))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    fn secs(s: i64) -> Timestamp {
        Timestamp::from_secs(s).unwrap()
    }

    #[test]
    fn manual_clock_is_send_sync_static() {
        fn check<T: Send + Sync + 'static>(_: &T) {}
        check(&ManualClock::default());
    }

    #[test]
    fn starts_where_told_and_defaults_to_the_epoch() {
        assert_eq!(ManualClock::new(secs(42)).now(), secs(42));
        assert_eq!(ManualClock::default().now(), Timestamp::EPOCH);
    }

    #[test]
    fn advance_and_rewind_move_time() {
        let clock = ManualClock::new(secs(100));
        clock.advance(Duration::from_secs(5));
        assert_eq!(clock.now(), secs(105));
        clock.rewind(Duration::from_secs(10));
        assert_eq!(clock.now(), secs(95));
    }

    #[test]
    fn set_jumps_anywhere() {
        let clock = ManualClock::default();
        clock.set(secs(-7));
        assert_eq!(clock.now(), secs(-7));
    }

    #[test]
    fn movement_saturates_at_the_ends_of_time() {
        let clock = ManualClock::default();
        clock.advance(Duration::MAX);
        assert_eq!(clock.now(), Timestamp::MAX);

        clock.set(Timestamp::MAX);
        clock.advance(Duration::from_micros(1));
        assert_eq!(clock.now(), Timestamp::MAX);

        clock.set(Timestamp::MIN);
        clock.rewind(Duration::from_micros(1));
        assert_eq!(clock.now(), Timestamp::MIN);
    }

    #[test]
    fn shared_handles_see_each_others_changes() {
        let handle = Arc::new(ManualClock::default());
        let seen_by_code: Arc<dyn Clock> = handle.clone();

        handle.advance(Duration::from_secs(2));
        assert_eq!(seen_by_code.now(), secs(2));
    }

    #[test]
    fn a_rewound_clock_yields_zero_age_not_a_panic() {
        let clock = ManualClock::new(secs(100));
        let seen = clock.now();
        clock.rewind(Duration::from_secs(10));
        assert_eq!(clock.now().saturating_duration_since(seen), Duration::ZERO);
    }

    #[test]
    fn concurrent_advances_are_not_lost() {
        let clock = ManualClock::default();
        std::thread::scope(|scope| {
            for _ in 0..4 {
                scope.spawn(|| {
                    for _ in 0..1_000 {
                        clock.advance(Duration::from_micros(1));
                    }
                });
            }
        });
        assert_eq!(clock.now().to_micros(), 4_000);
    }
}
