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
