//! A small macro for enums whose variants have stable text names.

/// Defines a fieldless enum whose variants each have a stable text name.
///
/// The generated enum derives `Debug`, `Clone`, `Copy`, `PartialEq`, `Eq`, and
/// `Hash`, and gets:
///
/// - `ALL`, a slice of every variant in declaration order
/// - `as_str`, the variant's name
/// - `Display`, which prints that name
/// - `FromStr`, the exact inverse of `as_str`, failing with the named
///   [`Error`](crate::Error) variant
///
/// Extra derives, such as `PartialOrd` and `Ord`, can be added as attributes.
///
/// The names are a storage contract: they end up in database rows, so a name
/// is never changed once released.
macro_rules! string_enum {
    (
        $(#[$meta:meta])*
        $name:ident, $what:literal, $error:ident {
            $(
                $(#[$variant_meta:meta])*
                $variant:ident = $text:literal,
            )+
        }
    ) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub enum $name {
            $(
                $(#[$variant_meta])*
                $variant,
            )+
        }

        impl $name {
            /// Every variant, in declaration order.
            pub const ALL: &[Self] = &[$(Self::$variant),+];

            /// The stable name used in storage and text forms.
            pub const fn as_str(self) -> &'static str {
                match self {
                    $(Self::$variant => $text,)+
                }
            }
        }

        impl ::std::fmt::Display for $name {
            fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
                f.write_str(self.as_str())
            }
        }

        impl ::std::str::FromStr for $name {
            type Err = $crate::error::Error;

            fn from_str(s: &str) -> ::std::result::Result<Self, Self::Err> {
                Self::ALL
                    .iter()
                    .copied()
                    .find(|variant| variant.as_str() == s)
                    .ok_or_else(|| {
                        $crate::error::Error::$error(format!(
                            concat!("unknown ", $what, " {:?}"),
                            s
                        ))
                    })
            }
        }
    };
}

pub(crate) use string_enum;
