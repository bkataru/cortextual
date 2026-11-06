//! The tokenizer port: counting tokens.
//!
//! Context budgets are measured in tokens, so the context builder needs to
//! know how many tokens a piece of text costs. A [`Tokenizer`] answers that
//! question.
//!
//! Counting is infallible, synchronous, and pure: it runs on every candidate
//! while packing, so it must be fast and cannot do I/O. A tokenizer that needs
//! the network to count exactly should be wrapped by an adapter that caches
//! results and falls back to a local estimate on failure.
//!
//! Tokenizers say how far to trust them through [`Accuracy`]. Budgeting code
//! leaves headroom when the answer is [`Accuracy::Estimate`].
//!
//! ```
//! use cortextual::ports::tokenizer::{HeuristicTokenizer, Tokenizer};
//!
//! let tokens = HeuristicTokenizer.count("remember the milk");
//! assert_eq!(tokens, 6); // 17 ASCII bytes / 3, rounded up
//! ```

use std::sync::Arc;

/// How far to trust a tokenizer's counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Accuracy {
    /// Counts match what the target model counts.
    Exact,
    /// Counts are approximate. Budgetins code should leave headroom.
    Estimate,
}

/// Counts tokens in text.
///
/// Implementations must be deterministic and side-effect free, and
/// `count("")` must be zero.
pub trait Tokenizer: Send + Sync {
    /// The number of tokens `text` costs.
    fn count(&self, text: &str) -> usize;

    /// How far to trust [`count`](Self::count).
    fn accuracy(&self) -> Accuracy;
}

/// A model-free estimate based on text shape.
///
/// ASCII text costs one token per 3 characters, rounded up. That slightly
/// overestimates English prose, which averages closer to 4 characters per
/// token, and is about right for code. Every non-ASCII character costs 2
/// tokens, which is conservative for most scripts. Dense identifiers such as
/// hashes tokenize worse than either rule assumes, which is why this reports
/// [`Accuracy::Estimate`].
#[derive(Debug, Clone, Copy, Default)]
pub struct HeuristicTokenizer;

impl HeuristicTokenizer {
    /// ASCII characters assumed per token.
    pub const ASCII_CHARS_PER_TOKEN: usize = 3;

    /// Tokens assumed per non-ASCII character.
    pub const TOKENS_PER_NON_ASCII_CHAR: usize = 2;
}

impl Tokenizer for HeuristicTokenizer {
    fn count(&self, text: &str) -> usize {
        // Fast path: `is_ascii` is vectorized in std, and for ASCII the byte
        // length is the character count.
        if text.is_ascii() {
            return text.len().div_ceil(Self::ASCII_CHARS_PER_TOKEN);
        }

        let mut ascii = 0_usize;
        let mut other = 0_usize;
        for c in text.chars() {
            if c.is_ascii() {
                ascii += 1;
            } else {
                other += 1;
            }
        }
        ascii.div_ceil(Self::ASCII_CHARS_PER_TOKEN) + other * Self::TOKENS_PER_NON_ASCII_CHAR
    }

    fn accuracy(&self) -> Accuracy {
        Accuracy::Estimate
    }
}

macro_rules! forward_to_inner {
    ($($wrapper:ty),+ $(,)?) => {
        $(
            impl<T: Tokenizer + ?Sized> Tokenizer for $wrapper {
                fn count(&self, text: &str) -> usize {
                    T::count(self, text)
                }

                fn accuracy(&self) -> Accuracy {
                    T::accuracy(self)
                }
            }
        )+
    };
}

forward_to_inner!(&T, Box<T>, Arc<T>);

#[cfg(test)]
mod tests {
    use super::*;

    fn count_with<T: Tokenizer>(tokenizer: &T, text: &str) -> usize {
        tokenizer.count(text)
    }

    #[test]
    fn tokenizer_is_send_sync_static() {
        fn check<T: Send + Sync + 'static>(_: &T) {}
        let shared: Arc<dyn Tokenizer> = Arc::new(HeuristicTokenizer);
        check(&shared);
    }

    #[test]
    fn ascii_text_costs_one_token_per_three_characters_rounded_up() {
        let t = HeuristicTokenizer;
        assert_eq!(t.count(""), 0);
        for (text, expected) in [
            ("a", 1),
            ("abc", 1),
            ("abcd", 2),
            ("abcdef", 2),
            ("abcdefg", 3),
        ] {
            assert_eq!(t.count(text), expected, "{text:?}");
        }
    }

    #[test]
    fn non_ascii_characters_cost_two_tokens_each() {
        let t = HeuristicTokenizer;
        assert_eq!(t.count("\u{e9}"), 2);
        assert_eq!(t.count("\u{65e5}\u{672c}\u{8a9e}"), 6);
        assert_eq!(t.count("\u{1F642}"), 2);
    }

    #[test]
    fn mixed_text_adds_both_parts() {
        // "h", "l", "l", "o" are ASCII: ceil(4 / 3) = 2. The accented e adds 2.
        assert_eq!(HeuristicTokenizer.count("h\u{e9}llo"), 4);
    }

    #[test]
    fn estimate_never_decreases_as_text_grows() {
        let t = HeuristicTokenizer;
        let text = "memory \u{e9}episodique \u{65e5}\u{672c} 42, with punctuation!";
        let boundaries = text
            .char_indices()
            .map(|(i, _)| i)
            .chain(std::iter::once(text.len()));

        let mut previous = 0;
        for end in boundaries {
            let n = t.count(&text[..end]);
            assert!(n >= previous, "count dropped at byte {end}");
            previous = n;
        }
    }

    #[test]
    fn heuristic_reports_estimate_accuracy() {
        assert_eq!(HeuristicTokenizer.accuracy(), Accuracy::Estimate);
    }

    #[test]
    fn references_boxes_and_arcs_are_tokenizers() {
        let boxed: Box<dyn Tokenizer> = Box::new(HeuristicTokenizer);
        let shared: Arc<dyn Tokenizer> = Arc::new(HeuristicTokenizer);
        let counts = [
            count_with(&HeuristicTokenizer, "abcd"),
            count_with(&&HeuristicTokenizer, "abcd"),
            count_with(&boxed, "abcd"),
            count_with(&shared, "abcd"),
        ];
        for n in counts {
            assert_eq!(n, 2);
        }
        assert_eq!(shared.accuracy(), Accuracy::Estimate);
    }
}
