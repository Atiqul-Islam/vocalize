//! Voice management and selection for TTS synthesis.

use crate::error::{VocalizeError, VocalizeResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// Gender classification for voices
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Gender {
    /// Male voice
    Male,
    /// Female voice
    Female,
    /// Non-binary or neutral voice
    Neutral,
}

impl std::fmt::Display for Gender {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Male => write!(f, "Male"),
            Self::Female => write!(f, "Female"),
            Self::Neutral => write!(f, "Neutral"),
        }
    }
}

/// Voice style characteristics
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VoiceStyle {
    /// Natural, conversational style
    Natural,
    /// Professional, formal style
    Professional,
    /// Expressive, emotional style
    Expressive,
    /// Calm, soothing style
    Calm,
    /// Energetic, enthusiastic style
    Energetic,
}

impl std::fmt::Display for VoiceStyle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Natural => write!(f, "Natural"),
            Self::Professional => write!(f, "Professional"),
            Self::Expressive => write!(f, "Expressive"),
            Self::Calm => write!(f, "Calm"),
            Self::Energetic => write!(f, "Energetic"),
        }
    }
}

/// Voice configuration for TTS synthesis
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Voice {
    /// Unique identifier for the voice
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// Language code (e.g., "en-US", "es-ES")
    pub language: String,
    /// Voice gender
    pub gender: Gender,
    /// Voice style
    pub style: VoiceStyle,
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Description of the voice
    pub description: String,
    /// Whether this voice is available for use
    pub available: bool,
    /// Speed multiplier (1.0 = normal speed)
    pub speed: f32,
    /// Pitch adjustment (-1.0 to 1.0, 0.0 = no change)
    pub pitch: f32,
}

impl Voice {
    /// Create a new voice configuration
    #[must_use]
    pub fn new(
        id: String,
        name: String,
        language: String,
        gender: Gender,
        style: VoiceStyle,
    ) -> Self {
        Self {
            id,
            name,
            language,
            gender,
            style,
            sample_rate: crate::DEFAULT_SAMPLE_RATE,
            description: String::new(),
            available: true,
            speed: 1.0,
            pitch: 0.0,
        }
    }

    /// Set the voice description
    #[must_use]
    pub fn with_description(mut self, description: String) -> Self {
        self.description = description;
        self
    }

    /// Set the sample rate
    #[must_use]
    pub fn with_sample_rate(mut self, sample_rate: u32) -> Self {
        self.sample_rate = sample_rate;
        self
    }

    /// Set the speed multiplier
    ///
    /// # Errors
    ///
    /// Returns an error if speed is not in the valid range (0.1 to 3.0)
    pub fn with_speed(mut self, speed: f32) -> VocalizeResult<Self> {
        if !(0.1..=3.0).contains(&speed) {
            return Err(VocalizeError::invalid_input(format!(
                "Speed must be between 0.1 and 3.0, got {speed}"
            )));
        }
        self.speed = speed;
        Ok(self)
    }

    /// Set the pitch adjustment
    ///
    /// # Errors
    ///
    /// Returns an error if pitch is not in the valid range (-1.0 to 1.0)
    pub fn with_pitch(mut self, pitch: f32) -> VocalizeResult<Self> {
        if !(-1.0..=1.0).contains(&pitch) {
            return Err(VocalizeError::invalid_input(format!(
                "Pitch must be between -1.0 and 1.0, got {pitch}"
            )));
        }
        self.pitch = pitch;
        Ok(self)
    }

    /// Set availability status
    #[must_use]
    pub fn with_availability(mut self, available: bool) -> Self {
        self.available = available;
        self
    }

    /// Check if voice supports the given language
    #[must_use]
    pub fn supports_language(&self, language: &str) -> bool {
        self.language.eq_ignore_ascii_case(language)
            || self.language.split('-').next().unwrap_or("").eq_ignore_ascii_case(language)
    }

    /// Validate voice configuration
    pub fn validate(&self) -> VocalizeResult<()> {
        if self.id.is_empty() {
            return Err(VocalizeError::invalid_input("Voice ID cannot be empty"));
        }

        if self.name.is_empty() {
            return Err(VocalizeError::invalid_input("Voice name cannot be empty"));
        }

        if self.language.is_empty() {
            return Err(VocalizeError::invalid_input("Voice language cannot be empty"));
        }

        if !(0.1..=3.0).contains(&self.speed) {
            return Err(VocalizeError::invalid_input(format!(
                "Speed must be between 0.1 and 3.0, got {}",
                self.speed
            )));
        }

        if !(-1.0..=1.0).contains(&self.pitch) {
            return Err(VocalizeError::invalid_input(format!(
                "Pitch must be between -1.0 and 1.0, got {}",
                self.pitch
            )));
        }

        if self.sample_rate < 8000 || self.sample_rate > 48000 {
            return Err(VocalizeError::invalid_input(format!(
                "Sample rate must be between 8000 and 48000 Hz, got {}",
                self.sample_rate
            )));
        }

        Ok(())
    }
}

impl Default for Voice {
    fn default() -> Self {
        Self::new(
            "af_bella".to_string(),
            "Bella".to_string(),
            "en-US".to_string(),
            Gender::Female,
            VoiceStyle::Natural,
        )
        .with_description("Young, friendly female voice".to_string())
    }
}

/// Voice manager for handling voice selection and configuration
#[derive(Debug, Clone)]
pub struct VoiceManager {
    voices: Arc<HashMap<String, Voice>>,
}

impl VoiceManager {
    /// Create a new voice manager with default voices
    #[must_use]
    pub fn new() -> Self {
        let mut voices = HashMap::new();

        // Add default Kokoro voices
        let default_voices = [
            Voice::new(
                "af_bella".to_string(),
                "Bella".to_string(),
                "en-US".to_string(),
                Gender::Female,
                VoiceStyle::Natural,
            )
            .with_description("Young, friendly female voice".to_string()),
            Voice::new(
                "am_david".to_string(),
                "David".to_string(),
                "en-US".to_string(),
                Gender::Male,
                VoiceStyle::Professional,
            )
            .with_description("Professional male voice".to_string()),
            Voice::new(
                "af_sarah".to_string(),
                "Sarah".to_string(),
                "en-US".to_string(),
                Gender::Female,
                VoiceStyle::Calm,
            )
            .with_description("Mature, warm female voice".to_string()),
            Voice::new(
                "bf_emma".to_string(),
                "Emma".to_string(),
                "en-GB".to_string(),
                Gender::Female,
                VoiceStyle::Professional,
            )
            .with_description("British female voice".to_string()),
            Voice::new(
                "bm_james".to_string(),
                "James".to_string(),
                "en-GB".to_string(),
                Gender::Male,
                VoiceStyle::Natural,
            )
            .with_description("British male voice".to_string()),
        ];

        for voice in default_voices {
            voices.insert(voice.id.clone(), voice);
        }

        Self {
            voices: Arc::new(voices),
        }
    }

    /// Create a voice manager with custom voices
    #[must_use]
    pub fn with_voices(voices: Vec<Voice>) -> Self {
        let voice_map = voices
            .into_iter()
            .map(|voice| (voice.id.clone(), voice))
            .collect();

        Self {
            voices: Arc::new(voice_map),
        }
    }

    /// Get all available voices
    #[must_use]
    pub fn get_available_voices(&self) -> Vec<Voice> {
        self.voices
            .values()
            .filter(|voice| voice.available)
            .cloned()
            .collect()
    }

    /// Get all voices (including unavailable ones)
    #[must_use]
    pub fn get_all_voices(&self) -> Vec<Voice> {
        self.voices.values().cloned().collect()
    }

    /// Get a specific voice by ID
    pub fn get_voice(&self, voice_id: &str) -> VocalizeResult<Voice> {
        self.voices
            .get(voice_id)
            .cloned()
            .ok_or_else(|| VocalizeError::voice_not_found(voice_id))
    }

    /// Check if a voice exists and is available
    #[must_use]
    pub fn is_voice_available(&self, voice_id: &str) -> bool {
        self.voices
            .get(voice_id)
            .map_or(false, |voice| voice.available)
    }

    /// Get voices filtered by language
    #[must_use]
    pub fn get_voices_by_language(&self, language: &str) -> Vec<Voice> {
        self.voices
            .values()
            .filter(|voice| voice.available && voice.supports_language(language))
            .cloned()
            .collect()
    }

    /// Get voices filtered by gender
    #[must_use]
    pub fn get_voices_by_gender(&self, gender: Gender) -> Vec<Voice> {
        self.voices
            .values()
            .filter(|voice| voice.available && voice.gender == gender)
            .cloned()
            .collect()
    }

    /// Get voices filtered by style
    #[must_use]
    pub fn get_voices_by_style(&self, style: VoiceStyle) -> Vec<Voice> {
        self.voices
            .values()
            .filter(|voice| voice.available && voice.style == style)
            .cloned()
            .collect()
    }

    /// Get the default voice
    #[must_use]
    pub fn get_default_voice(&self) -> Voice {
        self.get_voice("af_bella")
            .unwrap_or_else(|_| Voice::default())
    }

    /// Get voice count
    #[must_use]
    pub fn voice_count(&self) -> usize {
        self.voices.len()
    }

    /// Get available voice count
    #[must_use]
    pub fn available_voice_count(&self) -> usize {
        self.voices.values().filter(|voice| voice.available).count()
    }

    /// Get supported languages
    #[must_use]
    pub fn get_supported_languages(&self) -> Vec<String> {
        let mut languages: Vec<String> = self
            .voices
            .values()
            .filter(|voice| voice.available)
            .map(|voice| voice.language.clone())
            .collect();
        languages.sort();
        languages.dedup();
        languages
    }
}

impl Default for VoiceManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gender_display() {
        assert_eq!(Gender::Male.to_string(), "Male");
        assert_eq!(Gender::Female.to_string(), "Female");
        assert_eq!(Gender::Neutral.to_string(), "Neutral");
    }

    #[test]
    fn test_voice_style_display() {
        assert_eq!(VoiceStyle::Natural.to_string(), "Natural");
        assert_eq!(VoiceStyle::Professional.to_string(), "Professional");
        assert_eq!(VoiceStyle::Expressive.to_string(), "Expressive");
        assert_eq!(VoiceStyle::Calm.to_string(), "Calm");
        assert_eq!(VoiceStyle::Energetic.to_string(), "Energetic");
    }

    #[test]
    fn test_voice_creation() {
        let voice = Voice::new(
            "test_voice".to_string(),
            "Test Voice".to_string(),
            "en-US".to_string(),
            Gender::Female,
            VoiceStyle::Natural,
        );

        assert_eq!(voice.id, "test_voice");
        assert_eq!(voice.name, "Test Voice");
        assert_eq!(voice.language, "en-US");
        assert_eq!(voice.gender, Gender::Female);
        assert_eq!(voice.style, VoiceStyle::Natural);
        assert_eq!(voice.sample_rate, crate::DEFAULT_SAMPLE_RATE);
        assert!(voice.available);
        assert_eq!(voice.speed, 1.0);
        assert_eq!(voice.pitch, 0.0);
    }

    #[test]
    fn test_voice_with_description() {
        let voice = Voice::new(
            "test".to_string(),
            "Test".to_string(),
            "en".to_string(),
            Gender::Male,
            VoiceStyle::Professional,
        )
        .with_description("Test description".to_string());

        assert_eq!(voice.description, "Test description");
    }

    #[test]
    fn test_voice_with_sample_rate() {
        let voice = Voice::new(
            "test".to_string(),
            "Test".to_string(),
            "en".to_string(),
            Gender::Male,
            VoiceStyle::Professional,
        )
        .with_sample_rate(48000);

        assert_eq!(voice.sample_rate, 48000);
    }

    #[test]
    fn test_voice_with_speed_valid() {
        let voice = Voice::new(
            "test".to_string(),
            "Test".to_string(),
            "en".to_string(),
            Gender::Male,
            VoiceStyle::Professional,
        )
        .with_speed(1.5)
        .expect("Valid speed should work");

        assert_eq!(voice.speed, 1.5);
    }

    #[test]
    fn test_voice_with_speed_invalid() {
        let voice = Voice::new(
            "test".to_string(),
            "Test".to_string(),
            "en".to_string(),
            Gender::Male,
            VoiceStyle::Professional,
        );

        assert!(voice.clone().with_speed(0.05).is_err());
        assert!(voice.with_speed(5.0).is_err());
    }

    #[test]
    fn test_voice_with_pitch_valid() {
        let voice = Voice::new(
            "test".to_string(),
            "Test".to_string(),
            "en".to_string(),
            Gender::Male,
            VoiceStyle::Professional,
        )
        .with_pitch(0.5)
        .expect("Valid pitch should work");

        assert_eq!(voice.pitch, 0.5);
    }

    #[test]
    fn test_voice_with_pitch_invalid() {
        let voice = Voice::new(
            "test".to_string(),
            "Test".to_string(),
            "en".to_string(),
            Gender::Male,
            VoiceStyle::Professional,
        );

        assert!(voice.clone().with_pitch(-1.5).is_err());
        assert!(voice.with_pitch(2.0).is_err());
    }

    #[test]
    fn test_voice_supports_language() {
        let voice = Voice::new(
            "test".to_string(),
            "Test".to_string(),
            "en-US".to_string(),
            Gender::Male,
            VoiceStyle::Professional,
        );

        assert!(voice.supports_language("en-US"));
        assert!(voice.supports_language("en"));
        assert!(voice.supports_language("EN"));
        assert!(!voice.supports_language("es"));
        assert!(!voice.supports_language("fr-FR"));
    }

    #[test]
    fn test_voice_validation() {
        // Valid voice
        let voice = Voice::default();
        assert!(voice.validate().is_ok());

        // Empty ID
        let mut voice = Voice::default();
        voice.id.clear();
        assert!(voice.validate().is_err());

        // Empty name
        let mut voice = Voice::default();
        voice.name.clear();
        assert!(voice.validate().is_err());

        // Empty language
        let mut voice = Voice::default();
        voice.language.clear();
        assert!(voice.validate().is_err());

        // Invalid speed
        let mut voice = Voice::default();
        voice.speed = 0.05;
        assert!(voice.validate().is_err());

        // Invalid pitch
        let mut voice = Voice::default();
        voice.pitch = 2.0;
        assert!(voice.validate().is_err());

        // Invalid sample rate
        let mut voice = Voice::default();
        voice.sample_rate = 1000;
        assert!(voice.validate().is_err());
    }

    #[test]
    fn test_voice_default() {
        let voice = Voice::default();
        assert_eq!(voice.id, "af_bella");
        assert_eq!(voice.name, "Bella");
        assert_eq!(voice.language, "en-US");
        assert_eq!(voice.gender, Gender::Female);
        assert_eq!(voice.style, VoiceStyle::Natural);
    }

    #[test]
    fn test_voice_manager_new() {
        let manager = VoiceManager::new();
        assert!(manager.voice_count() > 0);
        assert!(manager.available_voice_count() > 0);
    }

    #[test]
    fn test_voice_manager_get_voice() {
        let manager = VoiceManager::new();
        
        let voice = manager.get_voice("af_bella").expect("Should find bella");
        assert_eq!(voice.id, "af_bella");

        let result = manager.get_voice("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_voice_manager_is_voice_available() {
        let manager = VoiceManager::new();
        
        assert!(manager.is_voice_available("af_bella"));
        assert!(!manager.is_voice_available("nonexistent"));
    }

    #[test]
    fn test_voice_manager_get_available_voices() {
        let manager = VoiceManager::new();
        let voices = manager.get_available_voices();
        
        assert!(!voices.is_empty());
        assert!(voices.iter().all(|v| v.available));
    }

    #[test]
    fn test_voice_manager_get_voices_by_language() {
        let manager = VoiceManager::new();
        let en_voices = manager.get_voices_by_language("en-US");
        
        assert!(!en_voices.is_empty());
        assert!(en_voices.iter().all(|v| v.supports_language("en-US")));
    }

    #[test]
    fn test_voice_manager_get_voices_by_gender() {
        let manager = VoiceManager::new();
        let female_voices = manager.get_voices_by_gender(Gender::Female);
        
        assert!(!female_voices.is_empty());
        assert!(female_voices.iter().all(|v| v.gender == Gender::Female));
    }

    #[test]
    fn test_voice_manager_get_voices_by_style() {
        let manager = VoiceManager::new();
        let natural_voices = manager.get_voices_by_style(VoiceStyle::Natural);
        
        assert!(!natural_voices.is_empty());
        assert!(natural_voices.iter().all(|v| v.style == VoiceStyle::Natural));
    }

    #[test]
    fn test_voice_manager_get_default_voice() {
        let manager = VoiceManager::new();
        let default_voice = manager.get_default_voice();
        
        assert_eq!(default_voice.id, "af_bella");
    }

    #[test]
    fn test_voice_manager_get_supported_languages() {
        let manager = VoiceManager::new();
        let languages = manager.get_supported_languages();
        
        assert!(!languages.is_empty());
        assert!(languages.contains(&"en-US".to_string()));
        assert!(languages.contains(&"en-GB".to_string()));
    }

    #[test]
    fn test_voice_manager_with_voices() {
        let custom_voice = Voice::new(
            "custom".to_string(),
            "Custom".to_string(),
            "es-ES".to_string(),
            Gender::Male,
            VoiceStyle::Energetic,
        );

        let manager = VoiceManager::with_voices(vec![custom_voice.clone()]);
        
        assert_eq!(manager.voice_count(), 1);
        let retrieved = manager.get_voice("custom").expect("Should find custom voice");
        assert_eq!(retrieved, custom_voice);
    }

    #[test]
    fn test_voice_serialization() {
        let voice = Voice::default();
        let json = serde_json::to_string(&voice).expect("Should serialize");
        let deserialized: Voice = serde_json::from_str(&json).expect("Should deserialize");
        assert_eq!(voice, deserialized);
    }
}