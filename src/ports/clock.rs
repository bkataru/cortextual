//! The clock port: where "now" comes from.
//!
//! Code that needs the current time asks a [`Clock`] instead of reading the
//! system clock directly, so tests can control time. [`SystemClock`] is the
//! real implementation.
//!
//! Clocks report wall-clock time, which can jump backwards when the system
//! clock is adjusted. Measure ages with
//! [`Timestamp::saturating_duration_since`], which clamps negative spans to
//! zero.
//!
//! ```
//! use cortextual::ports::clock::{Clock, SystemClock};
//!
//! let now = SystemClock.now();
//! assert!(now.to_secs() > 1_577_836_800); // after 2020-01-01
//! ```

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::types::Timestamp;

/// A source of the current time.
pub trait Clock: Send + Sync {
    /// The current time.
    fn now(&self) -> Timestamp;
}

/// The real wall clock, backed by [`SystemTime`].
#[derive(Debug, Clone, Copy, Default)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn now(&self) -> Timestamp {
        let now = SystemTime::now();
        // Unreachable on any real machine (the range is about +/-292,000 years)
        // but saturating is a safer behavior than panicking.
        let saturated = if now >= UNIX_EPOCH {
            Timestamp::MAX
        } else {
            Timestamp::MIN
        };
        Timestamp::from_system_time(now).unwrap_or(saturated)
    }
}

impl<C: Clock + ?Sized> Clock for &C {
    fn now(&self) -> Timestamp {
        C::now(self)
    }
}

impl<C: Clock + ?Sized> Clock for Box<C> {
    fn now(&self) -> Timestamp {
        C::now(self)
    }
}

impl<C: Clock + ?Sized> Clock for Arc<C> {
    fn now(&self) -> Timestamp {
        C::now(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read<C: Clock>(clock: &C) -> Timestamp {
        clock.now()
    }

    #[test]
    fn clocks_are_send_sync_static() {
        fn check<T: Send + Sync + 'static>(_: &T) {}
        let shared: Arc<dyn Clock> = Arc::new(SystemClock);
        check(&SystemClock);
        check(&shared);
    }

    #[test]
    fn system_clock_reports_a_plausible_time() {
        let after_2020 = Timestamp::from_secs(1_577_836_800).unwrap();
        assert!(SystemClock.now() > after_2020);
    }

    #[test]
    fn references_boxes_and_arcs_are_clocks() {
        let boxed: Box<dyn Clock> = Box::new(SystemClock);
        let shared: Arc<dyn Clock> = Arc::new(SystemClock);
        let readings = [
            read(&SystemClock),
            read(&&SystemClock),
            read(&boxed),
            read(&shared),
        ];
        for t in readings {
            assert!(t > Timestamp::EPOCH);
        }
    }
}
