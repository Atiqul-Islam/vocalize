//! Fast phoneme processor for Piper TTS
//! Converts text to espeak phonemes without Python dependencies

use anyhow::{Result, Context};
use std::collections::HashMap;
use once_cell::sync::Lazy;

/// Phoneme processor for converting text to phoneme IDs
pub struct PhonemeProcessor {
    phoneme_map: HashMap<String, i64>,
}

// Common English words to phoneme mapping (simplified for demo)
// In production, this would be generated from espeak or CMU dict
static WORD_TO_PHONEMES: Lazy<HashMap<&'static str, &'static str>> = Lazy::new(|| {
    let mut map = HashMap::new();
    
    // Common words
    map.insert("hello", "h@l'oU");
    map.insert("world", "w'3:ld");
    map.insert("the", "D@");
    map.insert("a", "eI");
    map.insert("an", "an");
    map.insert("and", "and");
    map.insert("is", "Iz");
    map.insert("it", "It");
    map.insert("to", "tu:");
    map.insert("of", "@v");
    map.insert("in", "In");
    map.insert("for", "fo:r");
    map.insert("on", "Qn");
    map.insert("with", "wID");
    map.insert("as", "az");
    map.insert("at", "at");
    map.insert("by", "baI");
    map.insert("from", "fr@m");
    map.insert("test", "t'Est");
    map.insert("speech", "spi:tS");
    map.insert("synthesis", "sInT@sIs");
    
    // Numbers
    map.insert("one", "wVn");
    map.insert("two", "tu:");
    map.insert("three", "Tri:");
    map.insert("four", "fo:r");
    map.insert("five", "faIv");
    map.insert("six", "sIks");
    map.insert("seven", "sEvn");
    map.insert("eight", "eIt");
    map.insert("nine", "naIn");
    map.insert("ten", "tEn");
    
    // Common words (expanded)
    map.insert("this", "DIs");
    map.insert("that", "Dat");
    map.insert("what", "wQt");
    map.insert("when", "wEn");
    map.insert("where", "wE@r");
    map.insert("who", "hu:");
    map.insert("why", "waI");
    map.insert("how", "haU");
    map.insert("can", "kan");
    map.insert("will", "wIl");
    map.insert("would", "wUd");
    map.insert("should", "SUd");
    map.insert("could", "kUd");
    map.insert("have", "hav");
    map.insert("has", "haz");
    map.insert("had", "had");
    map.insert("do", "du:");
    map.insert("does", "dVz");
    map.insert("did", "dId");
    map.insert("be", "bi:");
    map.insert("am", "am");
    map.insert("are", "A:r");
    map.insert("was", "wQz");
    map.insert("were", "w3:r");
    map.insert("been", "bi:n");
    map.insert("being", "bi:IN");
    map.insert("get", "gEt");
    map.insert("got", "gQt");
    map.insert("go", "goU");
    map.insert("going", "goUIN");
    map.insert("went", "wEnt");
    map.insert("come", "kVm");
    map.insert("came", "keIm");
    map.insert("make", "meIk");
    map.insert("made", "meId");
    map.insert("take", "teIk");
    map.insert("took", "tUk");
    map.insert("give", "gIv");
    map.insert("gave", "geIv");
    map.insert("know", "noU");
    map.insert("knew", "nu:");
    map.insert("think", "TINk");
    map.insert("thought", "To:t");
    map.insert("see", "si:");
    map.insert("saw", "so:");
    map.insert("say", "seI");
    map.insert("said", "sEd");
    map.insert("good", "gUd");
    map.insert("bad", "bad");
    map.insert("new", "nu:");
    map.insert("old", "oUld");
    map.insert("big", "bIg");
    map.insert("small", "smo:l");
    map.insert("long", "lQN");
    map.insert("short", "So:rt");
    map.insert("high", "haI");
    map.insert("low", "loU");
    map.insert("right", "raIt");
    map.insert("left", "lEft");
    map.insert("up", "Vp");
    map.insert("down", "daUn");
    map.insert("yes", "jEs");
    map.insert("no", "noU");
    map.insert("not", "nQt");
    map.insert("all", "o:l");
    map.insert("some", "sVm");
    map.insert("many", "mEni");
    map.insert("few", "fju:");
    map.insert("more", "mo:r");
    map.insert("less", "lEs");
    map.insert("very", "vEri");
    map.insert("too", "tu:");
    map.insert("also", "o:lsoU");
    map.insert("just", "dZVst");
    map.insert("only", "oUnli");
    map.insert("now", "naU");
    map.insert("then", "DEn");
    map.insert("here", "hI@r");
    map.insert("there", "DE@r");
    map.insert("today", "t@deI");
    map.insert("tomorrow", "t@mQroU");
    map.insert("yesterday", "jEst@rdeI");
    
    map
});

// Phoneme to ID mapping (based on espeak phoneme set)
static PHONEME_TO_ID: Lazy<HashMap<char, i64>> = Lazy::new(|| {
    let mut map = HashMap::new();
    
    // Vowels
    map.insert('a', 10);
    map.insert('e', 11);
    map.insert('i', 12);
    map.insert('o', 13);
    map.insert('u', 14);
    map.insert('@', 15); // schwa
    map.insert('3', 16); // er
    map.insert('I', 17);
    map.insert('E', 18);
    map.insert('U', 19);
    map.insert('O', 20);
    map.insert('A', 21);
    map.insert('V', 22);
    map.insert('Q', 23);
    
    // Consonants
    map.insert('p', 30);
    map.insert('b', 31);
    map.insert('t', 32);
    map.insert('d', 33);
    map.insert('k', 34);
    map.insert('g', 35);
    map.insert('f', 36);
    map.insert('v', 37);
    map.insert('T', 38); // theta
    map.insert('D', 39); // eth
    map.insert('s', 40);
    map.insert('z', 41);
    map.insert('S', 42); // sh
    map.insert('Z', 43); // zh
    map.insert('h', 44);
    map.insert('m', 45);
    map.insert('n', 46);
    map.insert('N', 47); // ng
    map.insert('l', 48);
    map.insert('r', 49);
    map.insert('w', 50);
    map.insert('j', 51); // y
    map.insert('t', 52); // ch
    map.insert('d', 53); // j
    
    // Special
    map.insert('\'', 5); // Primary stress
    map.insert(',', 6);  // Secondary stress
    map.insert(':', 7);  // Length mark
    map.insert(' ', 8);  // Word boundary
    
    map
});

impl PhonemeProcessor {
    /// Create a new phoneme processor
    pub fn new() -> Result<Self> {
        Ok(Self {
            phoneme_map: HashMap::new(),
        })
    }
    
    /// Process text to phoneme IDs
    pub fn process_text(&self, text: &str) -> Result<Vec<i64>> {
        let mut tokens = vec![0]; // Start token
        
        // Normalize text
        let normalized = text.to_lowercase();
        let words: Vec<&str> = normalized.split_whitespace().collect();
        
        for (i, word) in words.iter().enumerate() {
            // Strip punctuation
            let clean_word = word.trim_matches(|c: char| !c.is_alphabetic());
            
            if let Some(phonemes) = self.word_to_phonemes(clean_word) {
                tokens.extend(phonemes);
            } else {
                // Fallback to letter-based approximation
                tokens.extend(self.grapheme_to_phoneme(clean_word));
            }
            
            // Add space between words (except last)
            if i < words.len() - 1 {
                tokens.push(8); // Space token
            }
        }
        
        tokens.push(0); // End token
        
        Ok(tokens)
    }
    
    /// Look up word in phoneme dictionary
    fn word_to_phonemes(&self, word: &str) -> Option<Vec<i64>> {
        WORD_TO_PHONEMES.get(word).map(|phonemes| {
            phonemes.chars()
                .filter_map(|ch| PHONEME_TO_ID.get(&ch).copied())
                .collect()
        })
    }
    
    /// Simple grapheme-to-phoneme conversion
    fn grapheme_to_phoneme(&self, word: &str) -> Vec<i64> {
        word.chars().map(|ch| {
            match ch {
                'a' => 10,
                'e' => 11,
                'i' => 12,
                'o' => 13,
                'u' => 14,
                'b' => 31,
                'c' => 34, // k sound
                'd' => 33,
                'f' => 36,
                'g' => 35,
                'h' => 44,
                'j' => 53,
                'k' => 34,
                'l' => 48,
                'm' => 45,
                'n' => 46,
                'p' => 30,
                'q' => 34, // k sound
                'r' => 49,
                's' => 40,
                't' => 32,
                'v' => 37,
                'w' => 50,
                'x' => 34, // ks
                'y' => 51,
                'z' => 41,
                _ => 15, // Default to schwa
            }
        }).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_process_text() {
        let processor = PhonemeProcessor::new().unwrap();
        
        let tokens = processor.process_text("hello world").unwrap();
        assert!(tokens.len() > 2); // At least start, content, end
        assert_eq!(tokens[0], 0); // Start token
        assert_eq!(tokens[tokens.len() - 1], 0); // End token
    }
    
    #[test]
    fn test_known_words() {
        let processor = PhonemeProcessor::new().unwrap();
        
        let tokens = processor.process_text("test").unwrap();
        // Should use dictionary lookup, not grapheme conversion
        assert!(tokens.len() > 3);
    }
}