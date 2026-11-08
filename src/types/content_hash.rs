//! Content hashes for exact-duplicate detection.

use std::fmt;

/// The BLAKE3 digest of a piece of content: 32 bytes.
///
/// Two pieces of content have the same hash only if their bytes are identical
/// (barring astronomically unlikely collisions), so a hash is a cheap key for
/// exact-duplicate detection. It is computed over the raw bytes, with no case
/// folding and no whitespace trimming. Catching near-duplicates is a job for
/// embeddings, not hashes.
///
/// The hash covers the content only, not who owns it. Any uniqueness rule that
/// uses it must be scoped to a [`Scope`](crate::types::Scope), so that one
/// tenant can never learn that another tenant stored the same text.
///
/// ```
/// use cortextual::types::ContentHash;
///
/// let a = ContentHash::of("remember the milk");
/// let b = ContentHash::of("remember the milk");
/// let c = ContentHash::of("Remember the milk");
/// assert_eq!(a, b);
/// assert_ne!(a, c);
/// ```
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ContentHash([u8; 32]);

impl ContentHash {
    /// Hashes `content`.
    pub fn of(content: &str) -> Self {
        Self(*blake3::hash(content.as_bytes()).as_bytes())
    }

    /// Wraps 32 raw digest bytes, such as ones read from a database BLOB.
    pub const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// The raw digest bytes, suitable for storing as a database BLOB.
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

impl fmt::Display for ContentHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in self.0 {
            write!(f, "{byte:02x}")?;
        }
        Ok(())
    }
}

impl fmt::Debug for ContentHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ContentHash({self})")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn content_hash_is_send_sync_static() {
        fn check<T: Send + Sync + 'static>(_: &T) {}
        check(&ContentHash::of("x"));
    }

    #[test]
    fn matches_the_published_blake3_vector_for_empty_input() {
        assert_eq!(
            ContentHash::of("").to_string(),
            "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262"
        );
    }

    #[test]
    fn equal_content_hashes_equal() {
        assert_eq!(ContentHash::of("milk"), ContentHash::of("milk"));
        assert_ne!(ContentHash::of("milk"), ContentHash::of("eggs"));
    }

    #[test]
    fn hashing_is_exact_not_normalized() {
        assert_ne!(ContentHash::of("Milk"), ContentHash::of("milk"));
        assert_ne!(ContentHash::of("milk "), ContentHash::of("milk"));
    }

    #[test]
    fn bytes_round_trip() {
        let hash = ContentHash::of("milk");
        assert_eq!(ContentHash::from_bytes(*hash.as_bytes()), hash);
    }

    #[test]
    fn display_is_64_lowercase_hex_digits() {
        let text = ContentHash::of("milk").to_string();
        assert_eq!(text.len(), 64);
        assert!(
            text.chars()
                .all(|c| c.is_ascii_digit() || ('a'..='f').contains(&c))
        );
    }

    #[test]
    fn debug_wraps_the_hex() {
        let hash = ContentHash::of("");
        assert_eq!(format!("{hash:?}"), format!("ContentHash({hash})"));
    }

    #[test]
    fn hashes_work_as_set_keys() {
        let mut seen = std::collections::HashSet::new();
        assert!(seen.insert(ContentHash::of("milk")));
        assert!(!seen.insert(ContentHash::of("milk")));
        assert!(seen.insert(ContentHash::of("eggs")));
    }
}
