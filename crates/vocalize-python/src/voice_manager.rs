//! Python bindings for voice management

use pyo3::prelude::*;
use std::collections::HashMap;
use vocalize_core::{Gender, Voice, VoiceManager, VoiceStyle};

use crate::error::IntoPyResult;

/// Python wrapper for Gender enum
#[pyclass(name = "Gender")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PyGender {
    Male,
    Female,
    Neutral,
}

impl From<Gender> for PyGender {
    fn from(gender: Gender) -> Self {
        match gender {
            Gender::Male => PyGender::Male,
            Gender::Female => PyGender::Female,
            Gender::Neutral => PyGender::Neutral,
        }
    }
}

impl From<PyGender> for Gender {
    fn from(py_gender: PyGender) -> Self {
        match py_gender {
            PyGender::Male => Gender::Male,
            PyGender::Female => Gender::Female,
            PyGender::Neutral => Gender::Neutral,
        }
    }
}

#[pymethods]
impl PyGender {
    pub fn __str__(&self) -> String {
        match self {
            PyGender::Male => "Male".to_string(),
            PyGender::Female => "Female".to_string(),
            PyGender::Neutral => "Neutral".to_string(),
        }
    }

    fn __repr__(&self) -> String {
        format!("Gender.{}", self.__str__())
    }

    #[classattr]
    const MALE: PyGender = PyGender::Male;

    #[classattr]
    const FEMALE: PyGender = PyGender::Female;

    #[classattr]
    const NEUTRAL: PyGender = PyGender::Neutral;
}

/// Python wrapper for VoiceStyle enum
#[pyclass(name = "VoiceStyle")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PyVoiceStyle {
    Natural,
    Professional,
    Expressive,
    Calm,
    Energetic,
}

impl From<VoiceStyle> for PyVoiceStyle {
    fn from(style: VoiceStyle) -> Self {
        match style {
            VoiceStyle::Natural => PyVoiceStyle::Natural,
            VoiceStyle::Professional => PyVoiceStyle::Professional,
            VoiceStyle::Expressive => PyVoiceStyle::Expressive,
            VoiceStyle::Calm => PyVoiceStyle::Calm,
            VoiceStyle::Energetic => PyVoiceStyle::Energetic,
        }
    }
}

impl From<PyVoiceStyle> for VoiceStyle {
    fn from(py_style: PyVoiceStyle) -> Self {
        match py_style {
            PyVoiceStyle::Natural => VoiceStyle::Natural,
            PyVoiceStyle::Professional => VoiceStyle::Professional,
            PyVoiceStyle::Expressive => VoiceStyle::Expressive,
            PyVoiceStyle::Calm => VoiceStyle::Calm,
            PyVoiceStyle::Energetic => VoiceStyle::Energetic,
        }
    }
}

#[pymethods]
impl PyVoiceStyle {
    fn __str__(&self) -> String {
        match self {
            PyVoiceStyle::Natural => "Natural".to_string(),
            PyVoiceStyle::Professional => "Professional".to_string(),
            PyVoiceStyle::Expressive => "Expressive".to_string(),
            PyVoiceStyle::Calm => "Calm".to_string(),
            PyVoiceStyle::Energetic => "Energetic".to_string(),
        }
    }

    fn __repr__(&self) -> String {
        format!("VoiceStyle.{}", self.__str__())
    }

    #[classattr]
    const NATURAL: PyVoiceStyle = PyVoiceStyle::Natural;

    #[classattr]
    const PROFESSIONAL: PyVoiceStyle = PyVoiceStyle::Professional;

    #[classattr]
    const EXPRESSIVE: PyVoiceStyle = PyVoiceStyle::Expressive;

    #[classattr]
    const CALM: PyVoiceStyle = PyVoiceStyle::Calm;

    #[classattr]
    const ENERGETIC: PyVoiceStyle = PyVoiceStyle::Energetic;
}

/// Python wrapper for Voice
#[pyclass(name = "Voice")]
#[derive(Debug, Clone)]
pub struct PyVoice {
    inner: Voice,
}

impl PyVoice {
    pub fn new(voice: Voice) -> Self {
        Self { inner: voice }
    }

    pub fn inner(&self) -> &Voice {
        &self.inner
    }

    pub fn into_inner(self) -> Voice {
        self.inner
    }
}

#[pymethods]
impl PyVoice {
    #[new]
    fn py_new(
        id: String,
        name: String,
        language: String,
        gender: PyGender,
        style: PyVoiceStyle,
    ) -> Self {
        let voice = Voice::new(id, name, language, gender.into(), style.into());
        Self::new(voice)
    }

    #[staticmethod]
    fn default() -> Self {
        Self::new(Voice::default())
    }

    #[getter]
    pub fn id(&self) -> String {
        self.inner.id.clone()
    }

    #[getter]
    pub fn name(&self) -> String {
        self.inner.name.clone()
    }

    #[getter]
    pub fn language(&self) -> String {
        self.inner.language.clone()
    }

    #[getter]
    pub fn gender(&self) -> PyGender {
        self.inner.gender.into()
    }

    #[getter]
    fn style(&self) -> PyVoiceStyle {
        self.inner.style.into()
    }

    #[getter]
    fn description(&self) -> String {
        self.inner.description.clone()
    }

    #[getter]
    fn sample_rate(&self) -> u32 {
        self.inner.sample_rate
    }

    #[getter]
    fn speed(&self) -> f32 {
        self.inner.speed
    }

    #[getter]
    fn pitch(&self) -> f32 {
        self.inner.pitch
    }

    fn with_description(&self, description: String) -> PyVoice {
        let mut voice = self.inner.clone();
        voice.description = description;
        Self::new(voice)
    }

    fn with_sample_rate(&self, sample_rate: u32) -> PyVoice {
        let mut voice = self.inner.clone();
        voice.sample_rate = sample_rate;
        Self::new(voice)
    }

    fn with_speed(&self, speed: f32) -> PyResult<PyVoice> {
        let voice = self.inner.clone().with_speed(speed).into_py_result()?;
        Ok(Self::new(voice))
    }

    fn with_pitch(&self, pitch: f32) -> PyResult<PyVoice> {
        let voice = self.inner.clone().with_pitch(pitch).into_py_result()?;
        Ok(Self::new(voice))
    }

    fn supports_language(&self, language: &str) -> bool {
        self.inner.supports_language(language)
    }

    fn __str__(&self) -> String {
        format!("{} ({})", self.inner.name, self.inner.id)
    }

    fn __repr__(&self) -> String {
        format!(
            "Voice(id='{}', name='{}', language='{}', gender={}, style={})",
            self.inner.id,
            self.inner.name,
            self.inner.language,
            PyGender::from(self.inner.gender).__repr__(),
            PyVoiceStyle::from(self.inner.style).__repr__()
        )
    }

    fn to_dict(&self) -> HashMap<String, String> {
        let mut dict = HashMap::new();
        dict.insert("id".to_string(), self.inner.id.clone());
        dict.insert("name".to_string(), self.inner.name.clone());
        dict.insert("language".to_string(), self.inner.language.clone());
        dict.insert("gender".to_string(), PyGender::from(self.inner.gender).__str__());
        dict.insert("style".to_string(), PyVoiceStyle::from(self.inner.style).__str__());
        dict.insert("description".to_string(), self.inner.description.clone());
        dict.insert("sample_rate".to_string(), self.inner.sample_rate.to_string());
        dict.insert("speed".to_string(), self.inner.speed.to_string());
        dict.insert("pitch".to_string(), self.inner.pitch.to_string());
        dict
    }
}

/// Python wrapper for VoiceManager
#[pyclass(name = "VoiceManager")]
#[derive(Debug)]
pub struct PyVoiceManager {
    inner: VoiceManager,
}

impl PyVoiceManager {
    pub fn new(manager: VoiceManager) -> Self {
        Self { inner: manager }
    }
}

#[pymethods]
impl PyVoiceManager {
    #[new]
    pub fn py_new() -> Self {
        Self::new(VoiceManager::new())
    }

    pub fn get_available_voices(&self) -> Vec<PyVoice> {
        self.inner
            .get_available_voices()
            .into_iter()
            .map(|v| PyVoice::new(v.clone()))
            .collect()
    }

    fn get_voice(&self, voice_id: &str) -> PyResult<PyVoice> {
        let voice = self.inner.get_voice(voice_id).into_py_result()?;
        Ok(PyVoice::new(voice))
    }

    fn get_default_voice(&self) -> PyVoice {
        PyVoice::new(self.inner.get_default_voice().clone())
    }

    fn get_voices_by_gender(&self, gender: PyGender) -> Vec<PyVoice> {
        self.inner
            .get_voices_by_gender(gender.into())
            .into_iter()
            .map(|v| PyVoice::new(v.clone()))
            .collect()
    }

    fn get_voices_by_style(&self, style: PyVoiceStyle) -> Vec<PyVoice> {
        self.inner
            .get_voices_by_style(style.into())
            .into_iter()
            .map(|v| PyVoice::new(v.clone()))
            .collect()
    }

    fn get_voices_by_language(&self, language: &str) -> Vec<PyVoice> {
        self.inner
            .get_voices_by_language(language)
            .into_iter()
            .map(|v| PyVoice::new(v.clone()))
            .collect()
    }

    fn get_supported_languages(&self) -> Vec<String> {
        self.inner.get_supported_languages()
    }

    fn is_voice_available(&self, voice_id: &str) -> bool {
        self.inner.is_voice_available(voice_id)
    }

    fn with_voices(&self, voices: Vec<PyVoice>) -> PyVoiceManager {
        let rust_voices: Vec<Voice> = voices.into_iter().map(|v| v.into_inner()).collect();
        Self::new(VoiceManager::with_voices(rust_voices))
    }

    fn __len__(&self) -> usize {
        self.inner.get_available_voices().len()
    }

    fn __repr__(&self) -> String {
        format!("VoiceManager({} voices)", self.__len__())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_py_gender_conversion() {
        assert_eq!(PyGender::from(Gender::Male), PyGender::Male);
        assert_eq!(PyGender::from(Gender::Female), PyGender::Female);
        assert_eq!(Gender::from(PyGender::Male), Gender::Male);
        assert_eq!(Gender::from(PyGender::Female), Gender::Female);
    }

    #[test]
    fn test_py_voice_style_conversion() {
        assert_eq!(PyVoiceStyle::from(VoiceStyle::Natural), PyVoiceStyle::Natural);
        assert_eq!(PyVoiceStyle::from(VoiceStyle::Professional), PyVoiceStyle::Professional);
        assert_eq!(VoiceStyle::from(PyVoiceStyle::Calm), VoiceStyle::Calm);
        assert_eq!(VoiceStyle::from(PyVoiceStyle::Expressive), VoiceStyle::Expressive);
    }

    #[test]
    fn test_py_voice_creation() {
        let voice = PyVoice::py_new(
            "test_voice".to_string(),
            "Test Voice".to_string(),
            "en-US".to_string(),
            PyGender::Female,
            PyVoiceStyle::Professional,
        );

        assert_eq!(voice.id(), "test_voice");
        assert_eq!(voice.name(), "Test Voice");
        assert_eq!(voice.language(), "en-US");
        assert_eq!(voice.gender(), PyGender::Female);
        assert_eq!(voice.style(), PyVoiceStyle::Professional);
    }

    #[test]
    fn test_py_voice_default() {
        let voice = PyVoice::default();
        assert_eq!(voice.id(), "af_bella");
        assert_eq!(voice.name(), "Bella");
    }

    #[test]
    fn test_py_voice_modifications() {
        let voice = PyVoice::default();
        
        let with_desc = voice.with_description("Test description".to_string());
        assert_eq!(with_desc.description(), Some("Test description".to_string()));

        let with_sr = voice.with_sample_rate(48000);
        assert_eq!(with_sr.sample_rate(), 48000);
    }

    #[test]
    fn test_py_voice_speed_validation() {
        let voice = PyVoice::default();
        
        // Valid speed
        assert!(voice.with_speed(1.5).is_ok());
        
        // Invalid speed (too low)
        assert!(voice.with_speed(0.05).is_err());
        
        // Invalid speed (too high)
        assert!(voice.with_speed(5.0).is_err());
    }

    #[test]
    fn test_py_voice_pitch_validation() {
        let voice = PyVoice::default();
        
        // Valid pitch
        assert!(voice.with_pitch(0.5).is_ok());
        
        // Invalid pitch (too low)
        assert!(voice.with_pitch(-1.5).is_err());
        
        // Invalid pitch (too high)
        assert!(voice.with_pitch(2.0).is_err());
    }

    #[test]
    fn test_py_voice_supports_language() {
        let voice = PyVoice::default(); // en-US voice
        assert!(voice.supports_language("en-US"));
        assert!(voice.supports_language("en"));
        assert!(!voice.supports_language("es-ES"));
    }

    #[test]
    fn test_py_voice_to_dict() {
        let voice = PyVoice::default();
        let dict = voice.to_dict();
        
        assert_eq!(dict.get("id"), Some(&"af_bella".to_string()));
        assert_eq!(dict.get("name"), Some(&"Bella".to_string()));
        assert_eq!(dict.get("language"), Some(&"en-US".to_string()));
        assert_eq!(dict.get("gender"), Some(&"Female".to_string()));
        assert_eq!(dict.get("style"), Some(&"Natural".to_string()));
    }

    #[test]
    fn test_py_voice_manager_creation() {
        let manager = PyVoiceManager::py_new();
        assert!(manager.__len__() > 0);
    }

    #[test]
    fn test_py_voice_manager_get_voices() {
        let manager = PyVoiceManager::py_new();
        
        let voices = manager.get_available_voices();
        assert!(!voices.is_empty());
        
        let default_voice = manager.get_default_voice();
        assert_eq!(default_voice.id(), "af_bella");
        
        let voice = manager.get_voice("af_bella");
        assert!(voice.is_some());
        
        let nonexistent = manager.get_voice("nonexistent");
        assert!(nonexistent.is_none());
    }

    #[test]
    fn test_py_voice_manager_filtering() {
        let manager = PyVoiceManager::py_new();
        
        let female_voices = manager.get_voices_by_gender(PyGender::Female);
        assert!(!female_voices.is_empty());
        
        let male_voices = manager.get_voices_by_gender(PyGender::Male);
        assert!(!male_voices.is_empty());
        
        let professional_voices = manager.get_voices_by_style(PyVoiceStyle::Professional);
        assert!(!professional_voices.is_empty());
        
        let en_voices = manager.get_voices_by_language("en-US");
        assert!(!en_voices.is_empty());
    }

    #[test]
    fn test_py_voice_manager_supported_languages() {
        let manager = PyVoiceManager::py_new();
        let languages = manager.get_supported_languages();
        assert!(languages.contains(&"en-US".to_string()));
        assert!(languages.contains(&"en-GB".to_string()));
    }

    #[test]
    fn test_py_voice_manager_voice_availability() {
        let manager = PyVoiceManager::py_new();
        assert!(manager.is_voice_available("af_bella"));
        assert!(!manager.is_voice_available("nonexistent"));
    }

    #[test]
    fn test_py_voice_manager_with_voices() {
        let manager = PyVoiceManager::py_new();
        let custom_voice = PyVoice::py_new(
            "custom".to_string(),
            "Custom".to_string(),
            "en-US".to_string(),
            PyGender::Male,
            PyVoiceStyle::Natural,
        );
        
        let new_manager = manager.with_voices(vec![custom_voice]);
        assert_eq!(new_manager.__len__(), 1);
        assert!(new_manager.is_voice_available("custom"));
    }

    #[test]
    fn test_enum_string_representations() {
        assert_eq!(PyGender::Male.__str__(), "Male");
        assert_eq!(PyGender::Female.__str__(), "Female");
        assert_eq!(PyGender::Male.__repr__(), "Gender.Male");
        
        assert_eq!(PyVoiceStyle::Natural.__str__(), "Natural");
        assert_eq!(PyVoiceStyle::Professional.__str__(), "Professional");
        assert_eq!(PyVoiceStyle::Natural.__repr__(), "VoiceStyle.Natural");
    }
}