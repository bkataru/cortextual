//! Scopes: the isolation boundary for memories.
//!
//! A [`Scope`] names where a memory lives and where a reader stands. It has two
//! required parts, a tenant and a user, and two optional narrowing parts, an
//! agent and a session:
//!
//! ```text
//! tenant / user / agent? / session?
//! ```
//!
//! A memory written without an agent is shared by all of that user's agents. A
//! memory written without a session outlives any single session. A reader may
//! see a memory when the memory's scope *covers* the reader's scope; see
//! [`Scope::covers`].
//!
//! Each part has its own type, so swapping two arguments is prevented at compile time:
//!
//! ```compile_fail,E0308
//! use cortextual::types::scope::{Scope, TenantId, UserId};
//!
//! let tenant: TenantId = "acme".parse().unwrap();
//! let user: UserId = "alice".parse().unwrap();
//! let _ = Scope::new(user, tenant);
//! ```
//!
//! A scope prints as four slash-separated slots, leaving unused ones empty.
//! Components are never empty and never contain `/`, so the encoding is
//! unambiguous and parsing is its exact inverse:
//!
//! ```
//! use cortextual::types::scope::Scope;
//!
//! let scope: Scope = "acme/alice/bot/".parse().unwrap();
//! assert_eq!(scope.to_string(), "acme/alice/bot/");
//! ```
//!
//! Components are ASCII-only and case-sensitive on purpose: it rules out
//! lookalike names that would be different tenants to the database but the
//! same tenant to a human.

use std::fmt;
use std::str::FromStr;

use crate::error::{Error, Result};

/// Longest allowed scope component, in bytes.
pub const MAX_COMPONENT_LEN: usize = 128;

const ALLOWED: &str = "A-Z a-z 0-9 - _ . : @";

fn is_allowed(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | ':' | '@')
}

fn validate(label: &str, value: &str) -> Result<()> {
    if value.is_empty() {
        return Err(Error::InvalidScope(format!("{label} must not be empty")));
    }

    let len = value.len();
    if len > MAX_COMPONENT_LEN {
        return Err(Error::InvalidScope(format!(
            "{label} is {len} bytes long; the limit is {MAX_COMPONENT_LEN}"
        )));
    }

    if let Some((index, c)) = value.char_indices().find(|&(_, c)| !is_allowed(c)) {
        return Err(Error::InvalidScope(format!(
            "{label} contains {c:?} at byte {index}; allowed characters are {ALLOWED}"
        )));
    }

    Ok(())
}

macro_rules! scope_component {
    ($(#[$meta:meta])* $name:ident, $label:literal) => {
        $(#[$meta])*
        #[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
        pub struct $name(String);

        impl $name {
            /// Validates and wraps `value`.
            ///
            /// # Errors
            ///
            /// Returns [`Error::InvalidScope`] if the value is empty, longer
            /// than [`MAX_COMPONENT_LEN`] bytes, or contains a character
            /// outside `A-Z a-z 0-9 - _ . : @`.
            pub fn new(value: impl Into<String>) -> Result<Self> {
                let value = value.into();
                validate($label, &value)?;
                Ok(Self(value))
            }

            /// The validated string.
            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt::Display::fmt(&self.0, f)
            }
        }

        impl AsRef<str> for $name {
            fn as_ref(&self) -> &str {
                &self.0
            }
        }

        impl FromStr for $name {
            type Err = Error;

            fn from_str(s: &str) -> Result<Self, Self::Err> {
                Self::new(s)
            }
        }
    };
}

scope_component! {
    /// Identifier of a tenant: the outermost isolation boundary.
    TenantId, "tenant"
}

scope_component! {
    /// Identifier of a user within a tenant.
    UserId, "user"
}

scope_component! {
    /// Identifier of an agent acting on behalf of a user.
    AgentId, "agent"
}

scope_component! {
    /// Identifier of one session, such as a single conversation or run.
    SessionId, "session"
}

/// Where a memory lives, and where a reader stands.
///
/// Tenant and user are required. Agent and session are optional and narrow the
/// scope when present.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Scope {
    tenant: TenantId,
    user: UserId,
    agent: Option<AgentId>,
    session: Option<SessionId>,
}

impl Scope {
    /// A scope for one user of one tenant, across all of that user's agents
    /// and sessions.
    pub fn new(tenant: TenantId, user: UserId) -> Self {
        Self {
            tenant,
            user,
            agent: None,
            session: None,
        }
    }

    /// Narrows the scope to one agent.
    #[must_use]
    pub fn with_agent(mut self, agent: AgentId) -> Self {
        self.agent = Some(agent);
        self
    }

    /// Narrows the scope to one session.
    #[must_use]
    pub fn with_session(mut self, session: SessionId) -> Self {
        self.session = Some(session);
        self
    }

    /// The tenant.
    pub fn tenant(&self) -> &TenantId {
        &self.tenant
    }

    /// The user.
    pub fn user(&self) -> &UserId {
        &self.user
    }

    /// The agent, if the scope is narrowed to one.
    pub fn agent(&self) -> Option<&AgentId> {
        self.agent.as_ref()
    }

    /// The session, if the scope is narrowed to one.
    pub fn session(&self) -> Option<&SessionId> {
        self.session.as_ref()
    }

    /// Whether a memory written at `self` is visible to a reader at `other`.
    ///
    /// True when both scopes share a tenant and user and, for the agent and
    /// for the session, `self` either leaves that part open or names the same
    /// value as `other`. In short: `self` is equal to or broader than `other`.
    pub fn covers(&self, other: &Self) -> bool {
        self.tenant == other.tenant
            && self.user == other.user
            && widens(self.agent.as_ref(), other.agent.as_ref())
            && widens(self.session.as_ref(), other.session.as_ref())
    }
}

/// `broad` is open, or names exactly what `narrow` names.
fn widens<T: PartialEq>(broad: Option<&T>, narrow: Option<&T>) -> bool {
    match broad {
        None => true,
        Some(b) => narrow == Some(b),
    }
}

impl fmt::Display for Scope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let agent = self.agent.as_ref().map_or("", AgentId::as_str);
        let session = self.session.as_ref().map_or("", SessionId::as_str);
        write!(f, "{}/{}/{agent}/{session}", self.tenant, self.user)
    }
}

impl FromStr for Scope {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let parts: Vec<&str> = s.split('/').collect();
        let [tenant, user, agent, session] = parts.as_slice() else {
            return Err(Error::InvalidScope(format!(
                "expected 4 slash-separated parts (tenant/user/agent/session), found {}",
                parts.len()
            )));
        };

        let mut scope = Self::new(tenant.parse()?, user.parse()?);
        if !agent.is_empty() {
            scope = scope.with_agent(agent.parse()?);
        }
        if !session.is_empty() {
            scope = scope.with_session(session.parse()?);
        }
        Ok(scope)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scope(key: &str) -> Scope {
        key.parse().unwrap()
    }

    #[test]
    fn scope_is_send_sync_static() {
        fn check<T: Send + Sync + 'static>(_: &T) {}
        check(&scope("acme/alice//"));
    }

    #[test]
    fn components_accept_typical_identifiers() {
        for ok in ["a", "alice", "alice@example.com", "org:team-1_x.y", "0123"] {
            assert!(ok.parse::<UserId>().is_ok(), "{ok:?} should be accepted");
        }
        let longest = "x".repeat(MAX_COMPONENT_LEN);
        assert!(longest.parse::<UserId>().is_ok());
    }

    #[test]
    fn components_reject_bad_characters() {
        let bad_inputs = [
            "",
            "a/b",
            "a b",
            "tab\t",
            "line\nbreak",
            "caf\u{e9}",
            "smile\u{1F642}",
            "back\\slash",
            "star*",
        ];
        for bad in bad_inputs {
            assert!(
                bad.parse::<TenantId>().is_err(),
                "{bad:?} should be rejected"
            );
        }
    }

    #[test]
    fn components_reject_overlong_values() {
        let too_long = "x".repeat(MAX_COMPONENT_LEN + 1);
        let err = too_long.parse::<SessionId>().unwrap_err();
        assert_eq!(
            err.to_string(),
            "invalid scope: session is 129 bytes long; the limit is 128"
        );
    }

    #[test]
    fn component_errors_name_the_component_and_the_problem() {
        let empty = "".parse::<UserId>().unwrap_err();
        assert_eq!(empty.to_string(), "invalid scope: user must not be empty");

        let slash = "a/b".parse::<AgentId>().unwrap_err();
        assert!(
            slash
                .to_string()
                .starts_with("invalid scope: agent contains '/' at byte 1")
        );
    }

    #[test]
    fn display_always_has_four_slots() {
        let base = Scope::new("acme".parse().unwrap(), "alice".parse().unwrap());
        assert_eq!(base.to_string(), "acme/alice//");

        let with_agent = base.clone().with_agent("bot".parse().unwrap());
        assert_eq!(with_agent.to_string(), "acme/alice/bot/");

        let with_session = base.clone().with_session("s1".parse().unwrap());
        assert_eq!(with_session.to_string(), "acme/alice//s1");

        let both = base
            .with_agent("bot".parse().unwrap())
            .with_session("s1".parse().unwrap());
        assert_eq!(both.to_string(), "acme/alice/bot/s1");
    }

    #[test]
    fn parse_round_trips_canonical_keys() {
        for key in [
            "acme/alice//",
            "acme/alice/bot/",
            "acme/alice//s1",
            "acme/alice/bot/s1",
        ] {
            assert_eq!(scope(key).to_string(), key);
        }
    }

    #[test]
    fn parse_rejects_malformed_keys() {
        let bad_keys = [
            "",
            "acme",
            "acme/alice",
            "acme/alice/",
            "acme/alice/bot",
            "a/b/c/d/e",
            "/alice//",
            "acme///",
            "acme/alice/b d/",
        ];
        for bad in bad_keys {
            assert!(bad.parse::<Scope>().is_err(), "{bad:?} should be rejected");
        }
    }

    #[test]
    fn covers_orders_scopes_from_broad_to_narrow() {
        let user_level = scope("acme/alice//");
        let agent_level = scope("acme/alice/bot/");
        let session_level = scope("acme/alice//s1");
        let narrowest = scope("acme/alice/bot/s1");

        for s in [&user_level, &agent_level, &session_level, &narrowest] {
            assert!(s.covers(s), "{s} should cover itself");
        }

        assert!(user_level.covers(&agent_level));
        assert!(user_level.covers(&session_level));
        assert!(user_level.covers(&narrowest));
        assert!(agent_level.covers(&narrowest));
        assert!(session_level.covers(&narrowest));

        assert!(!agent_level.covers(&user_level));
        assert!(!narrowest.covers(&user_level));
        assert!(!narrowest.covers(&agent_level));

        assert!(!agent_level.covers(&session_level));
        assert!(!session_level.covers(&agent_level));
    }

    #[test]
    fn covers_never_crosses_users_tenants_or_siblings() {
        assert!(!scope("acme/alice/bot/").covers(&scope("acme/alice/other/")));
        assert!(!scope("acme/alice//").covers(&scope("acme/bob//")));
        assert!(!scope("acme/alice//").covers(&scope("globex/alice//")));
        assert!(!scope("acme/alice//s1").covers(&scope("acme/alice//s2")));
    }

    #[test]
    fn accessors_expose_what_was_set() {
        let s = scope("acme/alice/bot/s1");
        assert_eq!(s.tenant().as_str(), "acme");
        assert_eq!(s.user().as_str(), "alice");
        assert_eq!(s.agent().map(AgentId::as_str), Some("bot"));
        assert_eq!(s.session().map(SessionId::as_str), Some("s1"));
        assert!(scope("acme/alice//").agent().is_none());
    }

    #[test]
    fn scopes_work_as_hash_keys() {
        let mut set = std::collections::HashSet::new();
        set.insert(scope("acme/alice//"));
        assert!(set.contains(&scope("acme/alice//")));
        assert!(!set.contains(&scope("acme/alice/bot/")));
    }
}
