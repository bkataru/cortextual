//! Trust levels: how far a piece of information can be believed.
//!
//! Memories are derived from content that came from somewhere, and where it
//! came from matters. Text a user typed is not the same as the contents of a
//! web page a tool fetched, which may contain instructions aimed at the model
//! (prompt injection). Every memory carries a [`Trust`] level, computed from
//! its sources, and later stages use it to decide what may be pinned into the
//! prompt or acted on without review.
//!
//! The rule is that a memory is only as trustworthy as its least trusted
//! source. [`Trust::weakest`] computes that.

use crate::types::EpisodeKind;
use crate::types::string_enum::string_enum;

string_enum! {
    /// How far a piece of information can be trusted.
    ///
    /// Variants are ordered from least to most trusted, so the derived
    /// ordering means "more trusted is greater", and `min` of several levels
    /// is the weakest link.
    #[derive(PartialOrd, Ord)]
    Trust, "trust level", InvalidMemory {
        /// Content from outside: documents, tool output, events from other
        /// systems. This is where prompt injection can arrive.
        External = "external",
        /// Content produced by tools or by the model itself, which can repeat
        /// injected text or simply be wrong.
        Tool = "tool",
        /// Content stated directly by the user.
        User = "user",
        /// Content authored by the operator of the system, such as
        /// configuration. No episode produces this level.
        System = "system",
    }
}

impl Trust {
    /// The trust level of content from an episode of the given kind.
    pub const fn of_episode(kind: EpisodeKind) -> Self {
        match kind {
            EpisodeKind::UserMessage => Self::User,
            EpisodeKind::AssistantMessage | EpisodeKind::ToolCall => Self::Tool,
            EpisodeKind::ToolResult | EpisodeKind::Observation | EpisodeKind::Document => {
                Self::External
            }
        }
    }

    /// The least trusted of the given levels, or `None` if there are none.
    pub fn weakest(levels: impl IntoIterator<Item = Self>) -> Option<Self> {
        levels.into_iter().min()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trust_is_send_sync_static() {
        fn check<T: Send + Sync + 'static>(_: &T) {}
        check(&Trust::User);
    }

    #[test]
    fn more_trusted_is_greater() {
        assert!(Trust::External < Trust::Tool);
        assert!(Trust::Tool < Trust::User);
        assert!(Trust::User < Trust::System);
    }

    #[test]
    fn all_lists_variants_from_least_to_most_trusted() {
        assert_eq!(
            Trust::ALL,
            &[Trust::External, Trust::Tool, Trust::User, Trust::System]
        );
        assert!(Trust::ALL.windows(2).all(|pair| pair[0] < pair[1]));
    }

    #[test]
    fn names_round_trip_and_are_unique() {
        for &trust in Trust::ALL {
            assert_eq!(trust.as_str().parse::<Trust>().unwrap(), trust);
            assert_eq!(trust.to_string(), trust.as_str());
        }
        let names: std::collections::HashSet<_> = Trust::ALL.iter().map(|t| t.as_str()).collect();
        assert_eq!(names.len(), Trust::ALL.len());
    }

    #[test]
    fn unknown_names_are_rejected_with_a_clear_message() {
        let err = "root".parse::<Trust>().unwrap_err();
        assert_eq!(
            err.to_string(),
            r#"invalid memory: unknown trust level "root""#
        );
    }

    #[test]
    fn episode_kinds_map_to_the_documented_levels() {
        let expected = [
            (EpisodeKind::UserMessage, Trust::User),
            (EpisodeKind::AssistantMessage, Trust::Tool),
            (EpisodeKind::ToolCall, Trust::Tool),
            (EpisodeKind::ToolResult, Trust::External),
            (EpisodeKind::Observation, Trust::External),
            (EpisodeKind::Document, Trust::External),
        ];
        assert_eq!(expected.len(), EpisodeKind::ALL.len());
        for (kind, trust) in expected {
            assert_eq!(Trust::of_episode(kind), trust, "{kind}");
        }
    }

    #[test]
    fn no_episode_is_trusted_as_much_as_the_system() {
        for kind in EpisodeKind::ALL {
            assert!(Trust::of_episode(kind) < Trust::System);
        }
    }

    #[test]
    fn weakest_picks_the_least_trusted_level() {
        assert_eq!(Trust::weakest([]), None);
        assert_eq!(Trust::weakest([Trust::User]), Some(Trust::User));
        assert_eq!(
            Trust::weakest([Trust::System, Trust::External, Trust::User]),
            Some(Trust::External)
        );
    }
}
