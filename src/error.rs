//! Error type for the whole crate.

use thiserror::Error;

/// Everything that can go wrong in contextrual.
///
/// This enum is `#[non_exhaustive]`: new variants will be added as
/// needed, and consuming `match` expressions must include a wildcard arm.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Error {
    /// A string could not be parsed as an identifier of the given kind.
    #[error("invalid {kind} id {input:?}: {reason}")]
    InvalidId {
        /// Which kind of id was being parsed, such as `"memory"` or `"episode"`.
        kind: &'static str,
        /// The offending input.
        input: String,
        /// Why parsing failed.
        reason: String,
    },

    /// A scope field failed validation.
    #[error("invalid scope: {0}")]
    InvalidScope(String),

    /// A timestamp is out of range, or arithmetic on it overflowed.
    #[error("invalid timestamp: {0}")]
    InvalidTimestamp(String),

    /// An interval's start is after its end.
    #[error("invalid interval: {0}")]
    InvalidInterval(String),

    /// An embedding failed validation: wrong shape, non-finite values, or a
    /// vector that cannot be normalized.
    #[error("invalid embedding: {0}")]
    InvalidEmbedding(String),

    /// An episode failed validation: empty or oversized content, or an
    /// unknown kind.
    #[error("invalid episode: {0}")]
    InvalidEpisode(String),

    /// An embedder adapter failed, for example on a network error or a
    /// missing model file. The underlying error is available through
    /// [`std::error::Error::source`].
    #[error("embedder failed")]
    Embedder(#[source] Box<dyn std::error::Error + Send + Sync>),
}

impl Error {
    /// Wraps a failure from an embedder adapter.
    ///
    /// Accepts anything convertible into a boxed error: a `&str`, a `String`,
    /// or any error type that is `Send + Sync + 'static`.
    pub fn embedder(source: impl Into<Box<dyn std::error::Error + Send + Sync>>) -> Self {
        Self::Embedder(source.into())
    }
}

/// Crate-wide result type.
///
/// The second parameter defaults to [`Error`]. so `Result<T>` means
/// `Result<T, cortextual::Error>`; and `Result<T, OtherError>` still works.
pub type Result<T, E = Error> = std::result::Result<T, E>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_is_send_sync_static() {
        fn check<T: Send + Sync + 'static>(_: &T) {}
        let err = Error::InvalidScope(String::new());
        check(&err);
    }

    #[test]
    fn invalid_id_message_names_kind_and_input() {
        let err = Error::InvalidId {
            kind: "memory",
            input: "not-a-uuid".to_owned(),
            reason: "expected 36 characters".to_owned(),
        };
        assert_eq!(
            err.to_string(),
            r#"invalid memory id "not-a-uuid": expected 36 characters"#
        );
    }

    #[test]
    fn tuple_variants_render_their_payload() {
        assert_eq!(
            Error::InvalidScope("tenant must not be empty".to_owned()).to_string(),
            "invalid scope: tenant must not be empty"
        );
        assert_eq!(
            Error::InvalidTimestamp("overflow".to_owned()).to_string(),
            "invalid timestamp: overflow"
        );
        assert_eq!(
            Error::InvalidInterval("start after end".to_owned()).to_string(),
            "invalid interval: start after end"
        );
        assert_eq!(
            Error::InvalidEmbedding("zero vector".to_owned()).to_string(),
            "invalid embedding: zero vector"
        );
        assert_eq!(
            Error::InvalidEpisode("content must not be empty".to_owned()).to_string(),
            "invalid episode: content must not be empty"
        );
    }

    #[test]
    fn embedder_error_keeps_its_source() {
        use std::error::Error as _;

        let err = Error::embedder("model file missing");
        assert_eq!(err.to_string(), "embedder failed");
        assert_eq!(err.source().unwrap().to_string(), "model file missing");
    }

    #[test]
    fn result_alias_defaults_to_crate_error() {
        fn assert_defaults_to_crate_error<T>(_: &Result<T, Error>) {}

        let ok: Result<u8> = Ok(1);
        let custom: Result<u8, &str> = Err("other error type");

        // basic checks
        assert!(ok.is_ok());
        assert!(custom.is_err());

        assert_defaults_to_crate_error(&ok);
    }
}
