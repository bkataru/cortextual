//! A tokenizer with exact, predictable counts.

use crate::ports::{Accuracy, Tokenizer};

/// Counts whitespace-separated words, one token each.
///
/// Real tokenizers split text in model-specific ways that are hard to predict
/// by hand. This one makes budget arithmetic in tests obvious, and it reports
/// [`Accuracy::Exact`] because the count is exactly what it promises.
///
/// ```
/// use cortextual::ports::Tokenizer;
/// use cortextual::testing::WordTokenizer;
///
/// assert_eq!(WordTokenizer.count("remember the milk"), 3);
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct WordTokenizer;

impl Tokenizer for WordTokenizer {
    fn count(&self, text: &str) -> usize {
        text.split_whitespace().count()
    }

    fn accuracy(&self) -> Accuracy {
        Accuracy::Exact
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    #[test]
    fn tokenizer_is_send_sync_static() {
        fn check<T: Send + Sync + 'static>(_: &T) {}
        let shared: Arc<dyn Tokenizer> = Arc::new(WordTokenizer);
        check(&shared);
    }

    #[test]
    fn count_words() {
        assert_eq!(WordTokenizer.count(""), 0);
        assert_eq!(WordTokenizer.count("one"), 1);
        assert_eq!(WordTokenizer.count("one two three"), 3);
    }

    #[test]
    fn any_whitespace_separates_words_and_runs_collapse() {
        assert_eq!(WordTokenizer.count("  a \t b\n\nc  "), 3);
        assert_eq!(WordTokenizer.count("a\u{3000}b\u{a0}c"), 3);
        assert_eq!(WordTokenizer.count(" \t\n "), 0);
    }

    #[test]
    fn punctuation_stays_attached_to_its_word() {
        assert_eq!(WordTokenizer.count("hello, world!"), 2);
    }

    #[test]
    fn reports_exact_accuracy() {
        assert_eq!(WordTokenizer.accuracy(), Accuracy::Exact);
    }

    #[test]
    fn count_never_decreases_as_text_grows() {
        let text = "memory \u{e9}pisodique  42, with\tpunctuation!";
        let mut previous = 0;
        for (end, _) in text.char_indices().chain([(text.len(), ' ')]) {
            let n = WordTokenizer.count(&text[..end]);
            assert!(n >= previous, "count dropped at byte {end}");
            previous = n;
        }
    }
}
