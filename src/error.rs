use std::fmt;

/// Errors produced when loading or building a NAM model.
#[derive(Debug)]
pub enum Error {
    /// The `.nam` JSON could not be parsed.
    Json(serde_json::Error),
    /// The file was read but its contents are not a valid/supported model.
    Io(std::io::Error),
    /// The model's `architecture` field is not one this crate can run.
    UnsupportedArchitecture(String),
    /// A layer's `activation` field names a function this crate does not implement.
    UnsupportedActivation(String),
    /// The flat `weights` array did not contain the number of values the
    /// `config` implies (corrupt file, or a config/weights mismatch).
    WeightCountMismatch { expected: usize, found: usize },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Json(e) => write!(f, "failed to parse .nam JSON: {e}"),
            Self::Io(e) => write!(f, "failed to read .nam file: {e}"),
            Self::UnsupportedArchitecture(a) => {
                write!(f, "unsupported model architecture: {a:?}")
            }
            Self::UnsupportedActivation(a) => {
                write!(f, "unsupported activation function: {a:?}")
            }
            Self::WeightCountMismatch { expected, found } => write!(
                f,
                "weight count mismatch: config implies {expected} weights, file has {found}"
            ),
        }
    }
}

impl std::error::Error for Error {}

impl From<serde_json::Error> for Error {
    fn from(e: serde_json::Error) -> Self {
        Self::Json(e)
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}
