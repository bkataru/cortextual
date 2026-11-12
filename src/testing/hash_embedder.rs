//! A model-free embedder with real similarity behavior.

use std::sync::{Mutex, MutexGuard, PoisonError};

use crate::error::{Error, Result};
use crate::ports::{Embedder, EmbeddingBatch, InputKind};

const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// 64-bit FNV-1a: xor in a byte, multiply by a prime, repeat.
///
/// Chosen because it is tiny, has no dependencies, and its output is fixed
/// forever, unlike `std`'s hasher, which may change between Rust releases.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash = FNV_OFFSET_BASIS;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// An [`Embedder`] that needs no model.
///
/// It uses the hashing trick. Text is split into lowercase words, and each
/// word is hashed to one of `dim` buckets with a pseudo-random sign. The
/// bucket values are summed, and the row is normalized by
/// [`EmbeddingBatch`].
///
/// The result behaves like a (crude) embedding model in the ways retrieval
/// tests care about:
///
/// - Identical texts have cosine similarity 1
/// - Texts that share words are more similar than texts that do not, and texts
///   with no words in common are (almost always) orthogonal.
/// - Case, punctuation, and word order do not matter.
///
/// It has no notion of meaning: no synonyms, no stopword handling, and two
/// different words can collide into one bucket. Text without any words
/// embeds to a fixed vector.
///
/// Queries and documents embed identically, but every call is recorded so
/// tests can check which [`InputKind`] the code under test asked for. See
/// [`calls`](Self::calls).
///
/// ```
/// use cortextual::ports::{Embedder, InputKind};
/// use cortextual::testing::HashEmbedder;
///
/// let embedder = HashEmbedder::default();
/// let batch = embedder
///     .embed(InputKind::Document, &["the cat set", "the cat ran"])
///     .unwrap();
///
/// let dot: f32 = batch
///     .row(0)
///     .unwrap()
///     .iter()
///     .zip(batch.row(1).unwrap())
///     .map(|(a, b)| a * b)
///     .sum();
/// assert!(dot > 0.5);
/// ```
#[derive(Debug)]
pub struct HashEmbedder {
    dim: usize,
    model_id: String,
    calls: Mutex<Vec<(InputKind, usize)>>,
}

impl HashEmbedder {
    /// The dimension used by [`HashEmbedder::default`].
    pub const DEFAULT_DIM: usize = 256;

    /// Crates an embedder producing vectors `dim` wide.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidEmbedding`] if `dim` is zero.
    pub fn new(dim: usize) -> Result<Self> {
        if dim == 0 {
            return Err(Error::InvalidEmbedding(
                "dimension must be at least 1".to_owned(),
            ));
        }
        Ok(Self::with_dim(dim))
    }

    fn with_dim(dim: usize) -> Self {
        Self {
            dim,
            model_id: format!("hash-d{dim}"),
            calls: Mutex::new(Vec::new()),
        }
    }

    /// Every `embed` call so far, in order, as `(kind, number of texts)`.
    pub fn calls(&self) -> Vec<(InputKind, usize)> {
        self.lock_calls().clone()
    }

    fn lock_calls(&self) -> MutexGuard<'_, Vec<(InputKind, usize)>> {
        // A panic in another test thread must not hide this one's results, so
        // a poisoned lock is treated as unusable.
        self.calls.lock().unwrap_or_else(PoisonError::into_inner)
    }

    #[allow(
        clippy::cast_possible_truncation,
        reason = "the remainder is below `dim`, so it fits in usize"
    )]
    fn bucket(&self, hash: u64) -> usize {
        (hash % self.dim as u64) as usize
    }

    fn embed_text(&self, row: &mut [f32], text: &str) {
        let words = text
            .split(|c: char| !c.is_alphanumeric())
            .filter(|word| !word.is_empty());
        for word in words {
            let hash = fnv1a(word.to_lowercase().as_bytes());
            let sign = if hash >> 63 == 0 { 1.0 } else { -1.0 };
            row[self.bucket(hash)] += sign;
        }

        // No words, or words that cancel out exactly, would give a zero
        // vector, which cannot be normalized. Give it a fixed direction.
        if row.iter().all(|&x| x == 0.0) {
            row[0] = 1.0;
        }
    }
}

impl Default for HashEmbedder {
    fn default() -> Self {
        Self::with_dim(Self::DEFAULT_DIM)
    }
}

impl Embedder for HashEmbedder {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn embed(&self, kind: InputKind, texts: &[&str]) -> Result<EmbeddingBatch> {
        self.lock_calls().push((kind, texts.len()));

        let mut data = vec![0.0_f32; self.dim * texts.len()];
        for (row, text) in data.chunks_exact_mut(self.dim).zip(texts) {
            self.embed_text(row, text);
        }
        EmbeddingBatch::from_raw(self.dim, data)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::ports::embedder::embed_checked;

    fn embed(texts: &[&str]) -> EmbeddingBatch {
        HashEmbedder::default()
            .embed(InputKind::Document, texts)
            .unwrap()
    }

    fn cosine(batch: &EmbeddingBatch, i: usize, j: usize) -> f32 {
        let a = batch.row(i).unwrap();
        let b = batch.row(j).unwrap();
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    #[test]
    fn fnv1a_matches_published_test_vectors() {
        assert_eq!(fnv1a(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv1a(b"a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(fnv1a(b"foobar"), 0x8594_4171_f739_67e8);
    }

    #[test]
    fn embedding_is_send_sync_static() {
        fn check<T: Send + Sync + 'static>(_: &T) {}
        let shared: Arc<dyn Embedder> = Arc::new(HashEmbedder::default());
        check(&shared);
    }

    #[test]
    fn reports_its_identity() {
        let embedder = HashEmbedder::new(64).unwrap();
        assert_eq!(embedder.dim(), 64);
        assert_eq!(embedder.model_id(), "hash-d64");
        assert_eq!(HashEmbedder::default().dim(), HashEmbedder::DEFAULT_DIM);
        assert!(HashEmbedder::new(0).is_err());
    }

    #[test]
    fn same_text_gives_the_same_vector() {
        let batch = embed(&["the cat sat", "the cat sat"]);
        assert_eq!(batch.row(0), batch.row(1));
        assert_eq!(embed(&["the cat sat"]).row(0), batch.row(0));
    }

    #[test]
    fn ignores_case_punctuation_and_word_order() {
        let batch = embed(&["Remember the milk!", "milk, the REMEMBER"]);
        assert_eq!(batch.row(0), batch.row(1));
    }

    #[test]
    fn rows_are_unit_length() {
        let batch = embed(&["the cat sat on the warm mat", "a cat sat on the sofa", ""]);
        for i in 0..batch.len() {
            assert!((cosine(&batch, i, i) - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn matches_a_hand_computed_vector() {
        let batch = embed(&["the cat sat on the warm mat"]);
        let row = batch.row(0).unwrap();

        let nonzero = row.iter().filter(|&&x| x != 0.0).count();
        assert_eq!(nonzero, 6);
        assert!((row[124] - 2.0 / 3.0).abs() < 1e-6, "'the' appears twice");
        assert!(
            (row[39] + 1.0 / 3.0).abs() < 1e-6,
            "'cat' has a negative sign"
        );
    }

    #[test]
    fn shared_words_raise_similarity() {
        let batch = embed(&[
            "the cat sat on the warm mat",
            "a cat sat on the sofa",
            "quantum chromodynamics lattice simulation",
        ]);
        let related = cosine(&batch, 0, 1);
        let unrelated = cosine(&batch, 0, 2);
        assert!((related - 0.6804).abs() < 1e-3, "related = {related}");
        assert!(unrelated.abs() < 1e-6, "unrelated = {unrelated}");
    }

    #[test]
    fn text_without_words_gets_a_fixed_valid_vector() {
        let batch = embed(&["", "  ", "!!!"]);
        assert_eq!(batch.row(0), batch.row(1));
        assert_eq!(batch.row(1), batch.row(2));
        assert!((batch.row(0).unwrap()[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn records_calls_and_embeds_queries_like_documents() {
        let embedder = HashEmbedder::default();
        let queries = embed_checked(&embedder, InputKind::Query, &["milk", "eggs"]).unwrap();
        let documents = embed_checked(&embedder, InputKind::Document, &["milk"]).unwrap();

        assert_eq!(
            embedder.calls(),
            vec![(InputKind::Query, 2), (InputKind::Document, 1)]
        );
        assert_eq!(queries.row(0), documents.row(0));
    }

    #[test]
    fn empty_input_never_reaches_the_embedder() {
        let embedder = HashEmbedder::default();
        let batch = embed_checked(&embedder, InputKind::Document, &[]).unwrap();
        assert!(batch.is_empty());
        assert_eq!(embedder.calls(), []);
    }

    #[test]
    fn works_behind_a_shared_trait_object() {
        let shared: Arc<dyn Embedder> = Arc::new(HashEmbedder::default());
        let batch = embed_checked(&shared, InputKind::Query, &["a", "b", "c"]).unwrap();
        assert_eq!(batch.len(), 3);
        assert_eq!(batch.dim(), HashEmbedder::DEFAULT_DIM);
    }
}
