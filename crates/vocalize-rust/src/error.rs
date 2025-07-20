//! Error handling for Python bindings

use pyo3::{create_exception, exceptions::PyException, prelude::*};
use vocalize_core::VocalizeError;

// Create custom Python exception type
create_exception!(vocalize, VocalizeException, PyException, "Base exception for Vocalize TTS errors");

/// Python wrapper for VocalizeError
#[pyclass(name = "VocalizeError")]
#[derive(Debug, Clone)]
pub struct PyVocalizeError {
    pub error: VocalizeError,
}

impl PyVocalizeError {
    pub fn new(error: VocalizeError) -> Self {
        Self { error }
    }
}

#[pymethods]
impl PyVocalizeError {
    #[getter]
    fn message(&self) -> String {
        self.error.to_string()
    }

    #[getter]
    fn error_type(&self) -> String {
        match &self.error {
            VocalizeError::InvalidInput { .. } => "InvalidInput".to_string(),
            VocalizeError::SynthesisError { .. } => "SynthesisError".to_string(),
            VocalizeError::AudioDeviceError { .. } => "AudioDeviceError".to_string(),
            VocalizeError::AudioProcessingError { .. } => "AudioProcessingError".to_string(),
            VocalizeError::FileError { .. } => "FileError".to_string(),
            VocalizeError::VoiceNotFound { .. } => "VoiceNotFound".to_string(),
            VocalizeError::ConfigurationError { .. } => "ConfigurationError".to_string(),
            VocalizeError::ModelError { .. } => "ModelError".to_string(),
            VocalizeError::NetworkError { .. } => "NetworkError".to_string(),
            VocalizeError::MemoryError { .. } => "MemoryError".to_string(),
            VocalizeError::TimeoutError { .. } => "TimeoutError".to_string(),
            VocalizeError::ConcurrencyError { .. } => "ConcurrencyError".to_string(),
        }
    }

    #[getter]
    fn is_retriable(&self) -> bool {
        self.error.is_retriable()
    }

    #[getter]
    fn is_user_error(&self) -> bool {
        self.error.is_user_error()
    }

    fn __str__(&self) -> String {
        self.error.to_string()
    }

    fn __repr__(&self) -> String {
        format!("VocalizeError({})", self.error)
    }
}

/// Convert core VocalizeError to PyErr
pub fn vocalize_error_to_pyerr(err: vocalize_core::VocalizeError) -> PyErr {
    VocalizeException::new_err(err.to_string())
}

impl From<PyVocalizeError> for PyErr {
    fn from(err: PyVocalizeError) -> Self {
        VocalizeException::new_err(err.error.to_string())
    }
}

// Note: Cannot implement From<VocalizeError> for PyErr due to orphan rules
// Use vocalize_error_to_pyerr function instead

// Helper trait for converting Results
pub trait IntoPyResult<T> {
    fn into_py_result(self) -> PyResult<T>;
}

impl<T> IntoPyResult<T> for vocalize_core::VocalizeResult<T> {
    fn into_py_result(self) -> PyResult<T> {
        self.map_err(vocalize_error_to_pyerr)
    }
}

impl PyVocalizeError {
    pub fn new_err(message: String) -> PyErr {
        VocalizeException::new_err(message)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_py_vocalize_error_creation() {
        let rust_error = VocalizeError::invalid_input("Test error".to_string());
        let py_error = PyVocalizeError::new(rust_error);
        
        assert_eq!(py_error.error_type(), "InvalidInput");
        assert!(py_error.is_user_error());
        assert!(!py_error.is_retriable());
    }

    #[test]
    fn test_py_vocalize_error_display() {
        let rust_error = VocalizeError::synthesis_error("Synthesis failed".to_string());
        let py_error = PyVocalizeError::new(rust_error);
        
        assert!(py_error.message().contains("Synthesis failed"));
        assert!(py_error.__str__().contains("Synthesis failed"));
        assert!(py_error.__repr__().contains("VocalizeError"));
    }

    #[test]
    fn test_error_conversion() {
        let rust_error = VocalizeError::timeout("Operation timed out".to_string());
        let py_error = PyVocalizeError::new(rust_error.clone());
        
        assert_eq!(py_error.error_type(), "Timeout");
        assert!(py_error.is_retriable());
        assert!(!py_error.is_user_error());
    }

    #[test]
    fn test_all_error_types() {
        let test_cases = vec![
            (VocalizeError::invalid_input("test".to_string()), "InvalidInput"),
            (VocalizeError::synthesis_error("test".to_string()), "SynthesisError"),
            (VocalizeError::audio_device("test".to_string()), "AudioDevice"),
            (VocalizeError::file("test".to_string()), "FileError"),
            (VocalizeError::network("test".to_string()), "NetworkError"),
            (VocalizeError::model("test".to_string()), "ModelError"),
            (VocalizeError::configuration("test".to_string()), "ConfigurationError"),
            (VocalizeError::timeout("test".to_string()), "Timeout"),
            (VocalizeError::internal("test".to_string()), "InternalError"),
        ];

        for (error, expected_type) in test_cases {
            let py_error = PyVocalizeError::new(error);
            assert_eq!(py_error.error_type(), expected_type);
        }
    }
}