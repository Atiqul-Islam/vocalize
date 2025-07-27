"""
Cache builder for pre-computing common word tokens.
Builds a comprehensive cache of phoneme tokens for faster TTS.
"""

import sqlite3
import pickle
import time
import os
from pathlib import Path
from typing import List, Set
import requests

from .config import get_config


def build_default_cache(cache_file: Path, force: bool = False):
    """
    Build default token cache with common English words.
    
    Args:
        cache_file: Path to cache database file
        force: Force rebuild even if exists
    """
    # Check if cache is enabled
    if not get_config().get('optimizations.token_cache_enabled', False):
        print("Token cache is disabled. Enable it first with: vocalize optimize cache enable")
        return
    
    if cache_file.exists() and not force:
        print("Cache already exists. Use --force to rebuild.")
        return
    
    # Create database
    conn = sqlite3.connect(str(cache_file))
    cursor = conn.cursor()
    
    # Create schema
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS token_cache (
            text TEXT PRIMARY KEY,
            tokens BLOB NOT NULL
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_text ON token_cache(text)
    """)
    
    # Set version
    cursor.execute(
        "INSERT OR REPLACE INTO metadata (key, value) VALUES ('version', ?)",
        ("1.0",)
    )
    
    # Try to import tokenizer
    try:
        print("Loading tokenizer...")
        from ttstokenizer import IPATokenizer
        tokenizer = IPATokenizer()
        has_tokenizer = True
    except ImportError:
        print("Warning: ttstokenizer not available. Building cache with placeholder data.")
        print("Install with: pip install ttstokenizer")
        tokenizer = None
        has_tokenizer = False
    
    # Get word lists
    words = set()
    
    # 1. Top English words
    print("Fetching common English words...")
    common_words = fetch_common_words()
    if common_words:
        words.update(common_words)
        print(f"  Added {len(common_words)} common words")
    
    # 2. Common phrases
    print("Adding common phrases...")
    phrases = get_common_phrases()
    words.update(phrases)
    print(f"  Added {len(phrases)} common phrases")
    
    # 3. Numbers and special cases
    print("Adding numbers and special cases...")
    numbers = generate_numbers()
    words.update(numbers)
    print(f"  Added {len(numbers)} number words")
    
    # 4. Additional vocabulary
    print("Adding additional vocabulary...")
    vocab = get_additional_vocabulary()
    words.update(vocab)
    print(f"  Added {len(vocab)} vocabulary words")
    
    # Remove empty strings and normalize
    words = {w.strip() for w in words if w.strip()}
    
    print(f"\nTotal unique words/phrases to cache: {len(words)}")
    
    # Tokenize and cache
    if has_tokenizer:
        print("\nTokenizing and caching...")
        
        # Use tqdm if available for progress bar
        try:
            from tqdm import tqdm
            word_iter = tqdm(sorted(words), desc="Building cache")
        except ImportError:
            word_iter = sorted(words)
            print("Processing (this may take a minute)...")
        
        batch_size = 100
        batch = []
        processed = 0
        errors = 0
        
        for word in word_iter:
            try:
                tokens = tokenizer(word)
                if hasattr(tokens, 'tolist'):
                    token_list = tokens.tolist()
                else:
                    token_list = list(tokens)
                
                batch.append((word.lower(), pickle.dumps(token_list)))
                
                if len(batch) >= batch_size:
                    cursor.executemany(
                        "INSERT OR IGNORE INTO token_cache (text, tokens) VALUES (?, ?)",
                        batch
                    )
                    batch = []
                    processed += batch_size
                    
            except Exception as e:
                errors += 1
                if errors < 5:  # Only show first few errors
                    print(f"\nWarning: Failed to tokenize '{word}': {e}")
        
        # Insert remaining batch
        if batch:
            cursor.executemany(
                "INSERT OR IGNORE INTO token_cache (text, tokens) VALUES (?, ?)",
                batch
            )
            processed += len(batch)
        
        print(f"\n✅ Successfully cached {processed} words")
        if errors > 0:
            print(f"⚠️  Failed to tokenize {errors} words")
    
    else:
        # Create cache with placeholder tokens for testing
        print("\nCreating placeholder cache for testing...")
        for i, word in enumerate(sorted(words)[:1000]):  # Limit to 1000 for testing
            # Create mock tokens
            mock_tokens = [ord(c) % 256 for c in word.lower()][:50]
            cursor.execute(
                "INSERT OR IGNORE INTO token_cache (text, tokens) VALUES (?, ?)",
                (word.lower(), pickle.dumps(mock_tokens))
            )
    
    # Add build metadata
    cursor.execute(
        "INSERT OR REPLACE INTO metadata (key, value) VALUES ('build_time', ?)",
        (str(int(time.time())),)
    )
    
    cursor.execute(
        "INSERT OR REPLACE INTO metadata (key, value) VALUES ('word_count', ?)",
        (str(len(words)),)
    )
    
    conn.commit()
    
    # Show final statistics
    cursor.execute("SELECT COUNT(*) FROM token_cache")
    final_count = cursor.fetchone()[0]
    
    conn.close()
    
    print(f"\n✅ Token cache built successfully at {cache_file}")
    print(f"   Final entries: {final_count}")


def fetch_common_words() -> Set[str]:
    """Fetch common English words from online sources."""
    words = set()
    
    # Try multiple sources for robustness
    sources = [
        {
            "name": "Google 10000",
            "url": "https://raw.githubusercontent.com/first20hours/google-10000-english/master/google-10000-english.txt",
            "process": lambda text: text.strip().split('\n')
        },
        {
            "name": "Common English",
            "url": "https://raw.githubusercontent.com/dolph/dictionary/master/popular.txt",
            "process": lambda text: text.strip().split('\n')
        }
    ]
    
    for source in sources:
        try:
            response = requests.get(source["url"], timeout=10)
            response.raise_for_status()
            source_words = source["process"](response.text)
            words.update(w.strip() for w in source_words if w.strip())
            print(f"  ✓ Fetched {len(source_words)} words from {source['name']}")
            break  # Use first successful source
        except Exception as e:
            print(f"  ⚠ Could not fetch {source['name']}: {e}")
    
    return words


def get_common_phrases() -> Set[str]:
    """Get common English phrases and expressions."""
    return {
        # Greetings
        "hello", "hi", "hey", "hello world", "good morning", "good afternoon", 
        "good evening", "good night", "goodbye", "bye", "see you later",
        "see you soon", "take care", "have a nice day", "have a good day",
        
        # Polite expressions
        "thank you", "thanks", "thank you very much", "thanks a lot",
        "you're welcome", "you are welcome", "excuse me", "sorry", 
        "i'm sorry", "i am sorry", "pardon me", "please", "no problem",
        
        # Common questions
        "how are you", "how do you do", "what's your name", "what is your name",
        "where are you from", "what time is it", "what's the time",
        "how much is it", "how much does it cost", "can you help me",
        "do you speak english", "where is the bathroom", "where is",
        
        # Common responses
        "yes", "no", "maybe", "sure", "okay", "ok", "alright", "fine",
        "i don't know", "i do not know", "i understand", "i see",
        "that's right", "that is right", "exactly", "of course",
        
        # Time expressions
        "today", "tomorrow", "yesterday", "now", "later", "soon",
        "this morning", "this afternoon", "this evening", "tonight",
        
        # Common verbs and actions
        "let's go", "let us go", "come here", "go away", "wait",
        "stop", "start", "continue", "help", "look", "listen",
        
        # Technology/AI related
        "hey assistant", "hello assistant", "test", "testing",
        "one two three", "can you hear me", "is this working",
        
        # Emotions and states
        "i'm happy", "i am happy", "i'm sad", "i am sad",
        "i'm tired", "i am tired", "i'm hungry", "i am hungry",
        
        # Common sentences
        "the quick brown fox jumps over the lazy dog",
        "how much wood would a woodchuck chuck",
        "she sells seashells by the seashore",
    }


def generate_numbers() -> Set[str]:
    """Generate number words and related vocabulary."""
    words = set()
    
    # Basic numbers
    units = ["zero", "one", "two", "three", "four", "five", "six", "seven", 
             "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
             "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
    words.update(units)
    
    # Tens
    tens = ["twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
    words.update(tens)
    
    # Compound numbers 21-99
    for ten in tens:
        for i in range(1, 10):
            words.add(f"{ten} {units[i]}")
            words.add(f"{ten}-{units[i]}")  # Both formats
    
    # Hundreds, thousands, etc
    scales = ["hundred", "thousand", "million", "billion", "trillion"]
    words.update(scales)
    
    # Common number combinations
    for i in range(1, 10):
        words.add(f"{units[i]} hundred")
        words.add(f"{units[i]} thousand")
    
    # Ordinals
    ordinals = ["first", "second", "third", "fourth", "fifth", "sixth", 
                "seventh", "eighth", "ninth", "tenth", "eleventh", "twelfth",
                "thirteenth", "fourteenth", "fifteenth", "sixteenth",
                "seventeenth", "eighteenth", "nineteenth", "twentieth"]
    words.update(ordinals)
    
    # Ordinal patterns
    ordinal_suffixes = ["st", "nd", "rd", "th"]
    for i in range(1, 32):  # Days of month
        words.add(str(i))
        words.add(f"the {i}")
        
    # Years
    for year in range(2020, 2026):
        words.add(str(year))
        # Different ways to say years
        words.add(f"twenty {str(year)[2:]}")
    
    # Common numeric expressions
    words.update([
        "a dozen", "half a dozen", "a couple", "a few", "several",
        "once", "twice", "three times", "percent", "percentage",
        "dollar", "dollars", "cent", "cents", "euro", "euros",
        "pound", "pounds", "yen",
    ])
    
    return words


def get_additional_vocabulary() -> Set[str]:
    """Get additional common vocabulary."""
    words = set()
    
    # Days of week
    words.update([
        "monday", "tuesday", "wednesday", "thursday", "friday", 
        "saturday", "sunday", "weekend", "weekday"
    ])
    
    # Months
    words.update([
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december"
    ])
    
    # Seasons
    words.update(["spring", "summer", "fall", "autumn", "winter"])
    
    # Common names (top 20 each)
    male_names = [
        "james", "john", "robert", "michael", "william", "david",
        "richard", "joseph", "thomas", "christopher", "daniel", "matthew",
        "donald", "anthony", "paul", "mark", "george", "steven", "kenneth", "andrew"
    ]
    female_names = [
        "mary", "patricia", "jennifer", "linda", "elizabeth", "barbara",
        "susan", "jessica", "sarah", "karen", "nancy", "betty", "helen",
        "sandra", "donna", "carol", "ruth", "sharon", "michelle", "laura"
    ]
    words.update(male_names)
    words.update(female_names)
    
    # Common words from various categories
    categories = {
        "colors": ["red", "blue", "green", "yellow", "orange", "purple", "black", "white", "gray", "pink"],
        "animals": ["dog", "cat", "bird", "fish", "horse", "cow", "pig", "chicken", "sheep", "mouse"],
        "food": ["bread", "milk", "cheese", "apple", "banana", "coffee", "tea", "water", "juice", "pizza"],
        "verbs": ["be", "have", "do", "say", "go", "get", "make", "know", "think", "take"],
        "adjectives": ["good", "bad", "big", "small", "hot", "cold", "new", "old", "happy", "sad"],
        "pronouns": ["i", "you", "he", "she", "it", "we", "they", "me", "him", "her"],
    }
    
    for category, items in categories.items():
        words.update(items)
    
    return words


if __name__ == "__main__":
    # Test cache building
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--force":
        force = True
    else:
        force = False
    
    cache_file = Path("test_token_cache.db")
    build_default_cache(cache_file, force=force)