//! The embedder port: turning text into vectors.
//!
//! An [`Embedder`] maps text to fixed-width vectors. The core never talks to a
//! model directly. It calls [`embed_checked`], which verifies that the
//! embedder kept its promises, and gets back an [`EmbeddingBatch`].
//!
//! Three rules keep vectors comparable and cheap to search:
//!
//! - Every vector in a batch is L2-normalized when the batch is built, so
//!   cosine similarity is a plain dot product.
//! - An embedder reports a [`model_id`](Embedder::model_id). Vectors are only
//!   ever compared with vectors that share it.
//! - Queries and documents may be embedded differently, so callers say which
//!   one they are passing with [`InputKind`].

use std::sync::Arc;

use crate::error::{Error, Result};

/// What a piece of text will be used for.
///
/// Some models embed a search query differently from a stored passage. Models
/// that do not care can ignore this.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InputKind {
    /// Text to be stored and searched over.
    Document,
    /// A search query.
    Query,
}

/// A batch of unit-length embedding vectors, stored contiguously.
///
/// Rows are laid out back to back in one allocation. Every row has exactly
/// [`dim`](Self::dim) values, is finite, and has been normalized to length 1.
///
/// ```
/// use cortextual::ports::embedder::EmbeddingBatch;
///
/// let batch = EmbeddingBatch::from_raw(2, vec![3.0, 4.0]).unwrap();
/// let row = batch.row(0).unwrap();
/// assert!((row[0] - 0.6).abs() < 1e-6);
/// assert!((row[1] - 0.8).abs() < 1e-6);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct EmbeddingBatch {
    dim: usize,
    data: Vec<f32>,
}

impl EmbeddingBatch {
    /// Builds a batch from raw row-major values, normalizing each row.
    ///
    /// `data.len()` must be a multiple of `dim`. An empty `data` gives an
    /// empty batch.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidEmbedding`] if `dim` is zero, if `data` does
    /// not divide evenly into rows, if any value is not finite, or if any row
    /// is a zero vector.
    pub fn from_raw(dim: usize, mut data: Vec<f32>) -> Result<Self> {
        if dim == 0 {
            return Err(Error::InvalidEmbedding(
                "dimension must be at least 1".to_owned(),
            ));
        }
        if !data.len().is_multiple_of(dim) {
            return Err(Error::InvalidEmbedding(format!(
                "{} values do not divide evenly into rows of {dim}",
                data.len()
            )));
        }

        for (row_index, row) in data.chunks_exact_mut(dim).enumerate() {
            normalize_row(row_index, row)?;
        }

        Ok(Self { dim, data })
    }

    /// The width of every row.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The number of rows.
    pub fn len(&self) -> usize {
        self.data.len() / self.dim
    }

    /// Whether the batch has no rows.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// One row, or `None` if `index` is out of range.
    pub fn row(&self, index: usize) -> Option<&[f32]> {
        let start = index.checked_mul(self.dim)?;
        let end = start.checked_add(self.dim)?;
        self.data.get(start..end)
    }

    /// Iterates over the rows in order.
    pub fn rows(&self) -> std::slice::ChunksExact<'_, f32> {
        self.data.chunks_exact(self.dim)
    }

    /// All values, row after row.
    pub fn as_flat(&self) -> &[f32] {
        &self.data
    }

    /// Consumes the batch and returns all values, row after row.
    pub fn into_flat(self) -> Vec<f32> {
        self.data
    }
}

#[allow(
    clippy::cast_possible_truncation,
    reason = "narrowing the normalized f64 back to f32 is the intent"
)]
fn normalize_row(row_index: usize, row: &mut [f32]) -> Result<()> {
    // Accumulate in f64 so long rows do not lose precision in the norm.
    let mut sum_sq = 0.0_f64;
    for x in row.iter().copied() {
        if !x.is_finite() {
            return Err(Error::InvalidEmbedding(format!(
                "row {row_index} contains a non-finite value"
            )));
        }
        sum_sq += f64::from(x) * f64::from(x);
    }
    if sum_sq <= 0.0 {
        return Err(Error::InvalidEmbedding(format!(
            "row {row_index} is a zero vector and cannot be normalized"
        )));
    }

    let scale = 1.0 / sum_sq.sqrt();
    for x in row {
        *x = (f64::from(*x) * scale) as f32;
    }
    Ok(())
}

/// A source of text embeddings.
///
/// Implement [`embed`](Self::embed). Callers should go through
/// [`embed_checked`], which verifies the result.
pub trait Embedder: Send + Sync {
    /// Identifies the exact model configuration behind the vectors.
    ///
    /// This must change whenever the outputs would: different weights, a
    /// different dimension, a different prompt prefix, a different truncation
    /// rule. Vectors with different ids are never compared.
    fn model_id(&self) -> &str;

    /// The width of the vectors this embedder produces.
    fn dim(&self) -> usize;

    /// Embeds `texts`, returning one row per text, in order, each
    /// [`dim`](Self::dim) wide.
    ///
    /// Implementations should split oversized batches internally to respect
    /// the model's limits.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Embedder`] if the underlying model or service fails.
    fn embed(&self, kind: InputKind, texts: &[&str]) -> Result<EmbeddingBatch>;
}

/// Calls [`Embedder::embed`] and verifies the result.
///
/// Checks that the embedder returned one row per text and the dimension it
/// advertises. An empty `texts` returns an empty batch without calling the
/// embedder at all.
///
/// # Errors
///
/// Returns the embedder's own error, or [`Error::InvalidEmbedding`] if the
/// result has the wrong number of rows or the wrong dimension.
pub fn embed_checked<E: Embedder + ?Sized>(
    embedder: &E,
    kind: InputKind,
    texts: &[&str],
) -> Result<EmbeddingBatch> {
    if texts.is_empty() {
        return EmbeddingBatch::from_raw(embedder.dim(), Vec::new());
    }

    let batch = embedder.embed(kind, texts)?;
    if batch.len() != texts.len() {
        return Err(Error::InvalidEmbedding(format!(
            "embedder {:?} returned {} row(s) for {} text(s)",
            embedder.model_id(),
            batch.len(),
            texts.len()
        )));
    }
    if batch.dim() != embedder.dim() {
        return Err(Error::InvalidEmbedding(format!(
            "embedder {:?} returned dimension {}, expected {}",
            embedder.model_id(),
            batch.dim(),
            embedder.dim()
        )));
    }
    Ok(batch)
}

macro_rules! forward_to_inner {
    ($($wrapper:ty),+ $(,)?) => {
        $(
            impl<E: Embedder + ?Sized> Embedder for $wrapper {
                fn model_id(&self) -> &str {
                    E::model_id(self)
                }

                fn dim(&self) -> usize {
                    E::dim(self)
                }

                fn embed(&self, kind: InputKind, texts: &[&str]) -> Result<EmbeddingBatch> {
                    E::embed(self, kind, texts)
                }
            }
        )+
    };
}

forward_to_inner!(&E, Box<E>, Arc<E>);

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-6
    }

    fn norm(row: &[f32]) -> f32 {
        row.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    /// An embedder that returns whatever shape it is told to, for testing
    /// `embed_checked`.
    struct Stub {
        claimed_dim: usize,
        out_dim: usize,
        out_rows: Option<usize>,
    }

    impl Stub {
        fn honest(dim: usize) -> Self {
            Self {
                claimed_dim: dim,
                out_dim: dim,
                out_rows: None,
            }
        }
    }

    impl Embedder for Stub {
        fn model_id(&self) -> &'static str {
            "stub"
        }

        fn dim(&self) -> usize {
            self.claimed_dim
        }

        fn embed(&self, _kind: InputKind, texts: &[&str]) -> Result<EmbeddingBatch> {
            let rows = self.out_rows.unwrap_or(texts.len());
            EmbeddingBatch::from_raw(self.out_dim, vec![1.0; self.out_dim * rows])
        }
    }

    struct MustNotBeCalled;

    impl Embedder for MustNotBeCalled {
        fn model_id(&self) -> &'static str {
            "must-not-be-called"
        }

        fn dim(&self) -> usize {
            8
        }

        fn embed(&self, _kind: InputKind, _texts: &[&str]) -> Result<EmbeddingBatch> {
            panic!("embed must not be called for empty input");
        }
    }

    #[test]
    fn embedder_is_send_sync_static() {
        fn check<T: Send + Sync + 'static>(_: &T) {}
        let shared: Arc<dyn Embedder> = Arc::new(Stub::honest(4));
        check(&shared);
    }

    #[test]
    fn from_raw_normalizes_each_row() {
        let batch = EmbeddingBatch::from_raw(2, vec![3.0, 4.0, 0.0, -2.0]).unwrap();

        let first = batch.row(0).unwrap();
        assert!(approx(first[0], 0.6) && approx(first[1], 0.8));

        let second = batch.row(1).unwrap();
        assert!(approx(second[0], 0.0) && approx(second[1], -1.0));

        for row in batch.rows() {
            assert!(approx(norm(row), 1.0));
        }
    }

    #[test]
    fn from_raw_rejects_zero_dimension_and_ragged_data() {
        assert!(EmbeddingBatch::from_raw(0, vec![]).is_err());
        assert!(EmbeddingBatch::from_raw(0, vec![1.0]).is_err());
        assert!(EmbeddingBatch::from_raw(3, vec![1.0, 2.0, 3.0, 4.0]).is_err());
    }

    #[test]
    fn from_raw_rejects_non_finite_values() {
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let err = EmbeddingBatch::from_raw(2, vec![1.0, bad]).unwrap_err();
            assert!(err.to_string().contains("non-finite"), "{err}");
        }
    }

    #[test]
    fn from_raw_rejects_zero_vectors_and_names_the_row() {
        let err = EmbeddingBatch::from_raw(2, vec![1.0, 0.0, 0.0, 0.0]).unwrap_err();
        assert_eq!(
            err.to_string(),
            "invalid embedding: row 1 is a zero vector and cannot be normalized"
        );
    }

    #[test]
    fn empty_batches_are_allowed() {
        let batch = EmbeddingBatch::from_raw(4, vec![]).unwrap();
        assert!(batch.is_empty());
        assert_eq!(batch.len(), 0);
        assert_eq!(batch.dim(), 4);
        assert_eq!(batch.rows().count(), 0);
    }

    #[test]
    fn accessors_describe_the_layout() {
        let batch = EmbeddingBatch::from_raw(3, vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0]).unwrap();
        assert_eq!(batch.len(), 2);
        assert_eq!(batch.dim(), 3);
        assert!(!batch.is_empty());
        assert_eq!(batch.row(0).unwrap().len(), 3);
        assert!(batch.row(2).is_none());
        assert!(batch.row(usize::MAX).is_none());
        assert_eq!(batch.rows().count(), 2);
        assert_eq!(batch.as_flat().len(), 6);
        assert_eq!(batch.into_flat().len(), 6);
    }

    #[test]
    fn embed_checked_passes_well_behaved_embedders_through() {
        let batch = embed_checked(&Stub::honest(4), InputKind::Document, &["a", "b"]).unwrap();
        assert_eq!(batch.len(), 2);
        assert_eq!(batch.dim(), 4);
    }

    #[test]
    fn embed_checked_rejects_wrong_row_count() {
        let stub = Stub {
            out_rows: Some(1),
            ..Stub::honest(4)
        };
        let err = embed_checked(&stub, InputKind::Query, &["a", "b"]).unwrap_err();
        assert_eq!(
            err.to_string(),
            r#"invalid embedding: embedder "stub" returned 1 row(s) for 2 text(s)"#
        );
    }

    #[test]
    fn embed_checked_rejects_wrong_dimension() {
        let stub = Stub {
            out_dim: 3,
            ..Stub::honest(4)
        };
        let err = embed_checked(&stub, InputKind::Query, &["a"]).unwrap_err();
        assert_eq!(
            err.to_string(),
            r#"invalid embedding: embedder "stub" returned dimension 3, expected 4"#
        );
    }

    #[test]
    fn embed_checked_skips_the_embedder_for_empty_input() {
        let batch = embed_checked(&MustNotBeCalled, InputKind::Document, &[]).unwrap();
        assert!(batch.is_empty());
        assert_eq!(batch.dim(), 8);
    }

    #[test]
    fn references_boxes_and_arcs_are_embedders() {
        let texts = ["a", "b"];
        let boxed: Box<dyn Embedder> = Box::new(Stub::honest(2));
        let shared: Arc<dyn Embedder> = Arc::new(Stub::honest(2));
        let stub = Stub::honest(2);

        for len in [
            embed_checked(&&stub, InputKind::Document, &texts)
                .unwrap()
                .len(),
            embed_checked(&boxed, InputKind::Document, &texts)
                .unwrap()
                .len(),
            embed_checked(&shared, InputKind::Document, &texts)
                .unwrap()
                .len(),
        ] {
            assert_eq!(len, 2);
        }
    }
}
