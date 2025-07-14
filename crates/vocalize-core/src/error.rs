//! Error types for the Vocalize TTS engine.


/// Result type alias for Vocalize operations
pub type VocalizeResult<T> = Result<T, VocalizeError>;

/// Main error type for Vocalize TTS operations
#[derive(thiserror::Error, Debug, Clone, PartialEq, Eq)]
pub enum VocalizeError {
    /// TTS synthesis failed
    #[error("TTS synthesis failed: {message}")]
    SynthesisError {
        /// Error message describing the failure
        message: String,
    },

    /// Audio device error
    #[error("Audio device error: {message}")]
    AudioDeviceError {
        /// Error message describing the device issue
        message: String,
    },

    /// Audio format or processing error
    #[error("Audio processing error: {message}")]
    AudioProcessingError {
        /// Error message describing the processing issue
        message: String,
    },

    /// File I/O error
    #[error("File I/O error: {message}")]
    FileError {
        /// Error message describing the file operation failure
        message: String,
    },

    /// Voice not found error
    #[error("Voice '{voice_id}' not found")]
    VoiceNotFound {
        /// The voice ID that was not found
        voice_id: String,
    },

    /// Invalid input error
    #[error("Invalid input: {message}")]
    InvalidInput {
        /// Error message describing the invalid input
        message: String,
    },

    /// Configuration error
    #[error("Configuration error: {message}")]
    ConfigurationError {
        /// Error message describing the configuration issue
        message: String,
    },

    /// Model loading error
    #[error("Model loading error: {message}")]
    ModelError {
        /// Error message describing the model loading failure
        message: String,
    },

    /// Network or download error
    #[error("Network error: {message}")]
    NetworkError {
        /// Error message describing the network issue
        message: String,
    },

    /// Memory allocation error
    #[error("Memory allocation error: {message}")]
    MemoryError {
        /// Error message describing the memory issue
        message: String,
    },

    /// Timeout error
    #[error("Operation timed out: {message}")]
    TimeoutError {
        /// Error message describing the timeout
        message: String,
    },

    /// Thread or concurrency error
    #[error("Concurrency error: {message}")]
    ConcurrencyError {
        /// Error message describing the concurrency issue
        message: String,
    },
}

impl VocalizeError {
    /// Create a new synthesis error
    #[must_use]
    pub fn synthesis<S: Into<String>>(message: S) -> Self {
        Self::SynthesisError {
            message: message.into(),
        }
    }

    /// Create a new audio device error
    #[must_use]
    pub fn audio_device<S: Into<String>>(message: S) -> Self {
        Self::AudioDeviceError {
            message: message.into(),
        }
    }

    /// Create a new audio processing error
    #[must_use]
    pub fn audio_processing<S: Into<String>>(message: S) -> Self {
        Self::AudioProcessingError {
            message: message.into(),
        }
    }

    /// Create a new file error
    #[must_use]
    pub fn file<S: Into<String>>(message: S) -> Self {
        Self::FileError {
            message: message.into(),
        }
    }

    /// Create a new voice not found error
    #[must_use]
    pub fn voice_not_found<S: Into<String>>(voice_id: S) -> Self {
        Self::VoiceNotFound {
            voice_id: voice_id.into(),
        }
    }

    /// Create a new invalid input error
    #[must_use]
    pub fn invalid_input<S: Into<String>>(message: S) -> Self {
        Self::InvalidInput {
            message: message.into(),
        }
    }

    /// Create a new configuration error
    #[must_use]
    pub fn configuration<S: Into<String>>(message: S) -> Self {
        Self::ConfigurationError {
            message: message.into(),
        }
    }

    /// Create a new model error
    #[must_use]
    pub fn model<S: Into<String>>(message: S) -> Self {
        Self::ModelError {
            message: message.into(),
        }
    }

    /// Create a new model not found error
    #[must_use]
    pub fn model_not_found<S: Into<String>>(model_id: S) -> Self {
        Self::ModelError {
            message: format!("Model '{}' not found", model_id.into()),
        }
    }

    /// Create a new network error
    #[must_use]
    pub fn network<S: Into<String>>(message: S) -> Self {
        Self::NetworkError {
            message: message.into(),
        }
    }

    /// Create a new memory error
    #[must_use]
    pub fn memory<S: Into<String>>(message: S) -> Self {
        Self::MemoryError {
            message: message.into(),
        }
    }

    /// Create a new timeout error
    #[must_use]
    pub fn timeout<S: Into<String>>(message: S) -> Self {
        Self::TimeoutError {
            message: message.into(),
        }
    }

    /// Create a new concurrency error
    #[must_use]
    pub fn concurrency<S: Into<String>>(message: S) -> Self {
        Self::ConcurrencyError {
            message: message.into(),
        }
    }

    /// Check if this error is retriable
    #[must_use]
    pub const fn is_retriable(&self) -> bool {
        matches!(
            self,
            Self::NetworkError { .. } | Self::TimeoutError { .. } | Self::MemoryError { .. }
        )
    }

    /// Check if this error is due to invalid user input
    #[must_use]
    pub const fn is_user_error(&self) -> bool {
        matches!(
            self,
            Self::InvalidInput { .. }
                | Self::VoiceNotFound { .. }
                | Self::ConfigurationError { .. }
        )
    }

    /// Get the error category for logging/metrics
    #[must_use]
    pub const fn category(&self) -> &'static str {
        match self {
            Self::SynthesisError { .. } => "synthesis",
            Self::AudioDeviceError { .. } => "audio_device",
            Self::AudioProcessingError { .. } => "audio_processing",
            Self::FileError { .. } => "file",
            Self::VoiceNotFound { .. } => "voice",
            Self::InvalidInput { .. } => "input",
            Self::ConfigurationError { .. } => "configuration",
            Self::ModelError { .. } => "model",
            Self::NetworkError { .. } => "network",
            Self::MemoryError { .. } => "memory",
            Self::TimeoutError { .. } => "timeout",
            Self::ConcurrencyError { .. } => "concurrency",
        }
    }
}

// Convert from common error types
impl From<std::io::Error> for VocalizeError {
    fn from(err: std::io::Error) -> Self {
        Self::file(err.to_string())
    }
}



impl From<tokio::time::error::Elapsed> for VocalizeError {
    fn from(err: tokio::time::error::Elapsed) -> Self {
        Self::timeout(format!("Operation timed out: {err}"))
    }
}

impl From<serde_json::Error> for VocalizeError {
    fn from(err: serde_json::Error) -> Self {
        Self::file(format!("JSON serialization error: {err}"))
    }
}

impl From<anyhow::Error> for VocalizeError {
    fn from(err: anyhow::Error) -> Self {
        Self::synthesis(err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let err = VocalizeError::synthesis("Test synthesis error");
        assert_eq!(err.category(), "synthesis");
        assert!(!err.is_retriable());
        assert!(!err.is_user_error());
    }

    #[test]
    fn test_error_display() {
        let err = VocalizeError::voice_not_found("test_voice");
        assert_eq!(err.to_string(), "Voice 'test_voice' not found");
    }

    #[test]
    fn test_error_categories() {
        assert_eq!(VocalizeError::synthesis("test").category(), "synthesis");
        assert_eq!(VocalizeError::audio_device("test").category(), "audio_device");
        assert_eq!(VocalizeError::file("test").category(), "file");
        assert_eq!(VocalizeError::voice_not_found("test").category(), "voice");
        assert_eq!(VocalizeError::invalid_input("test").category(), "input");
        assert_eq!(VocalizeError::configuration("test").category(), "configuration");
        assert_eq!(VocalizeError::model("test").category(), "model");
        assert_eq!(VocalizeError::network("test").category(), "network");
        assert_eq!(VocalizeError::memory("test").category(), "memory");
        assert_eq!(VocalizeError::timeout("test").category(), "timeout");
        assert_eq!(VocalizeError::concurrency("test").category(), "concurrency");
    }

    #[test]
    fn test_retriable_errors() {
        assert!(VocalizeError::network("test").is_retriable());
        assert!(VocalizeError::timeout("test").is_retriable());
        assert!(VocalizeError::memory("test").is_retriable());
        assert!(!VocalizeError::synthesis("test").is_retriable());
        assert!(!VocalizeError::invalid_input("test").is_retriable());
    }

    #[test]
    fn test_user_errors() {
        assert!(VocalizeError::invalid_input("test").is_user_error());
        assert!(VocalizeError::voice_not_found("test").is_user_error());
        assert!(VocalizeError::configuration("test").is_user_error());
        assert!(!VocalizeError::synthesis("test").is_user_error());
        assert!(!VocalizeError::network("test").is_user_error());
    }

    #[test]
    fn test_from_io_error() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "File not found");
        let vocalize_err = VocalizeError::from(io_err);
        assert!(matches!(vocalize_err, VocalizeError::FileError { .. }));
    }

    #[test]
    fn test_error_equality() {
        let err1 = VocalizeError::synthesis("test message");
        let err2 = VocalizeError::synthesis("test message");
        let err3 = VocalizeError::synthesis("different message");
        
        assert_eq!(err1, err2);
        assert_ne!(err1, err3);
    }

    #[test]
    fn test_error_clone() {
        let err1 = VocalizeError::voice_not_found("test_voice");
        let err2 = err1.clone();
        assert_eq!(err1, err2);
    }

    #[test]
    fn test_error_debug() {
        let err = VocalizeError::audio_device("Test audio error");
        let debug_str = format!("{err:?}");
        assert!(debug_str.contains("AudioDeviceError"));
        assert!(debug_str.contains("Test audio error"));
    }
}