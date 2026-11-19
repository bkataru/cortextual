//! Converting between domain types and SQLite columns.
//!
//! Values read back from the database are treated as untrusted: if a stored id,
//! scope, hash, or enum name violates our invariants, decoding fails with
//! [`Error::CorruptData`] instead of panicking or returning bad values.

use std::str::FromStr;

use crate::error::{Error, Result};
use crate::types::{AgentId, ContentHash, Id, IdTag, Scope, SessionId, TenantId, UserId};

fn corrupt(err: Error) -> Error {
    match err {
        Error::CorruptData(_) => err,
        other => Error::CorruptData(other.to_string()),
    }
}

/// Splits a [`Scope`] into borrowed column values `(tenant, user, agent, session)`.
pub(super) fn scope_parts(scope: &Scope) -> (&str, &str, Option<&str>, Option<&str>) {
    (
        scope.tenant().as_str(),
        scope.user().as_str(),
        scope.agent().map(AgentId::as_str),
        scope.session().map(SessionId::as_str),
    )
}

/// Rebuilds and validates a [`Scope`] from stored column values.
pub(super) fn decode_scope(
    tenant: String,
    user: String,
    agent: Option<String>,
    session: Option<String>,
) -> Result<Scope> {
    let tenant = TenantId::new(tenant).map_err(corrupt)?;
    let user = UserId::new(user).map_err(corrupt)?;
    let mut scope = Scope::new(tenant, user);
    if let Some(agent) = agent {
        scope = scope.with_agent(AgentId::new(agent).map_err(corrupt)?);
    }
    if let Some(session) = session {
        scope = scope.with_session(SessionId::new(session).map_err(corrupt)?);
    }
    Ok(scope)
}

/// Decodes a 16-byte BLOB into a typed [`Id`].
pub(super) fn decode_id<K: IdTag>(bytes: &[u8]) -> Result<Id<K>> {
    Id::from_slice(bytes).map_err(corrupt)
}

/// Decodes a 32-byte BLOB into a [`ContentHash`].
pub(super) fn decode_content_hash(bytes: &[u8]) -> Result<ContentHash> {
    let array: [u8; 32] = bytes.try_into().map_err(|_| {
        Error::CorruptData(format!(
            "expected 32-byte content hash, found {} byte(s)",
            bytes.len()
        ))
    })?;
    Ok(ContentHash::from_bytes(array))
}

/// Parses a stored enum name using its [`FromStr`] implementation.
pub(super) fn decode_enum<T>(raw: &str) -> Result<T>
where
    T: FromStr<Err = Error>,
{
    raw.parse().map_err(corrupt)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{EpisodeId, EpisodeKind};

    #[test]
    fn scope_round_trips_through_parts() {
        for key in [
            "acme/alice//",
            "acme/alice/bot/",
            "acme/alice//s1",
            "acme/alice/bot/s1",
        ] {
            let scope: Scope = key.parse().unwrap();
            let (t, u, a, s) = scope_parts(&scope);
            let decoded = decode_scope(
                t.to_owned(),
                u.to_owned(),
                a.map(str::to_owned),
                s.map(str::to_owned),
            )
            .unwrap();
            assert_eq!(decoded, scope);
        }
    }

    #[test]
    fn invalid_stored_scope_maps_to_corrupt_data() {
        let bad_tenant = decode_scope(String::new(), "alice".to_owned(), None, None).unwrap_err();
        assert!(matches!(bad_tenant, Error::CorruptData(_)), "{bad_tenant}");

        let bad_agent = decode_scope(
            "acme".to_owned(),
            "alice".to_owned(),
            Some("a/b".to_owned()),
            None,
        )
        .unwrap_err();
        assert!(matches!(bad_agent, Error::CorruptData(_)), "{bad_agent}");
    }

    #[test]
    fn wrong_blob_sizes_map_to_corrupt_data() {
        let bad_id: Result<EpisodeId> = decode_id(&[1, 2, 3]);
        assert!(matches!(bad_id, Err(Error::CorruptData(_))), "{bad_id:?}");

        let bad_hash = decode_content_hash(&[0_u8; 31]).unwrap_err();
        assert!(matches!(bad_hash, Error::CorruptData(_)), "{bad_hash}");

        let valid_id = EpisodeId::generate();
        assert_eq!(decode_id(valid_id.as_bytes()).unwrap(), valid_id);

        let valid_hash = ContentHash::of("hello");
        assert_eq!(
            decode_content_hash(valid_hash.as_bytes()).unwrap(),
            valid_hash
        );
    }

    #[test]
    fn unknown_enum_name_maps_to_corrupt_data() {
        let err = decode_enum::<EpisodeKind>("tweet").unwrap_err();
        assert!(matches!(err, Error::CorruptData(_)), "{err}");
        assert_eq!(
            decode_enum::<EpisodeKind>("user_message").unwrap(),
            EpisodeKind::UserMessage
        );
    }
}
