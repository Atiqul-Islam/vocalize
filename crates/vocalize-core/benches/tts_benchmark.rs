use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use tokio::runtime::Runtime;
use vocalize_core::{TtsEngine, Voice, SynthesisParams, VoiceStyle, Gender};

fn bench_tts_synthesis(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let engine = rt.block_on(TtsEngine::new()).unwrap();
    
    let mut group = c.benchmark_group("tts_synthesis");
    
    // Test different text lengths
    let test_texts = vec![
        ("short", "Hello world"),
        ("medium", "This is a medium length text for testing TTS synthesis performance with multiple words and sentences."),
        ("long", &"This is a much longer text that should test the performance of the TTS engine with extended content. ".repeat(10)),
    ];
    
    let voice = Voice::default();
    let params = SynthesisParams::new(voice);
    
    for (name, text) in test_texts {
        group.bench_with_input(BenchmarkId::new("synthesize", name), text, |b, text| {
            b.to_async(&rt).iter(|| async {
                let result = engine.synthesize(black_box(text), black_box(&params)).await;
                black_box(result.unwrap())
            });
        });
    }
    
    group.finish();
}

fn bench_tts_streaming(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let engine = rt.block_on(TtsEngine::new()).unwrap();
    
    let mut group = c.benchmark_group("tts_streaming");
    
    let text = "This is a test text for streaming synthesis performance measurement with multiple sentences and words.";
    let voice = Voice::default();
    let params = SynthesisParams::new(voice).with_streaming(512);
    
    group.bench_function("streaming_synthesis", |b| {
        b.to_async(&rt).iter(|| async {
            let result = engine.synthesize_streaming(black_box(text), black_box(&params)).await;
            black_box(result.unwrap())
        });
    });
    
    group.finish();
}

fn bench_voice_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("voice_operations");
    
    // Benchmark voice creation
    group.bench_function("voice_creation", |b| {
        b.iter(|| {
            let voice = Voice::new(
                black_box("test_voice".to_string()),
                black_box("Test Voice".to_string()),
                black_box("en-US".to_string()),
                black_box(Gender::Female),
                black_box(VoiceStyle::Natural),
            );
            black_box(voice)
        });
    });
    
    // Benchmark voice validation
    let voice = Voice::default();
    group.bench_function("voice_validation", |b| {
        b.iter(|| {
            let result = black_box(&voice).validate();
            black_box(result)
        });
    });
    
    group.finish();
}

fn bench_different_voices(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let engine = rt.block_on(TtsEngine::new()).unwrap();
    
    let mut group = c.benchmark_group("different_voices");
    
    let text = "Performance test with different voice types and characteristics.";
    
    let voices = vec![
        ("female_natural", Voice::new("f_nat".to_string(), "Female Natural".to_string(), "en-US".to_string(), Gender::Female, VoiceStyle::Natural)),
        ("male_professional", Voice::new("m_prof".to_string(), "Male Professional".to_string(), "en-US".to_string(), Gender::Male, VoiceStyle::Professional)),
        ("neutral_calm", Voice::new("n_calm".to_string(), "Neutral Calm".to_string(), "en-US".to_string(), Gender::Neutral, VoiceStyle::Calm)),
    ];
    
    for (name, voice) in voices {
        let params = SynthesisParams::new(voice);
        group.bench_with_input(BenchmarkId::new("voice_synthesis", name), &params, |b, params| {
            b.to_async(&rt).iter(|| async {
                let result = engine.synthesize(black_box(text), black_box(params)).await;
                black_box(result.unwrap())
            });
        });
    }
    
    group.finish();
}

fn bench_synthesis_params(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let engine = rt.block_on(TtsEngine::new()).unwrap();
    
    let mut group = c.benchmark_group("synthesis_params");
    
    let text = "Testing synthesis with different speed and pitch parameters.";
    let voice = Voice::default();
    
    // Test different speeds
    let speeds = vec![0.5, 1.0, 1.5, 2.0];
    for speed in speeds {
        let params = SynthesisParams::new(voice.clone()).with_speed(speed).unwrap();
        group.bench_with_input(BenchmarkId::new("speed", speed), &params, |b, params| {
            b.to_async(&rt).iter(|| async {
                let result = engine.synthesize(black_box(text), black_box(params)).await;
                black_box(result.unwrap())
            });
        });
    }
    
    // Test different pitches
    let pitches = vec![-0.5, 0.0, 0.5];
    for pitch in pitches {
        let params = SynthesisParams::new(voice.clone()).with_pitch(pitch).unwrap();
        group.bench_with_input(BenchmarkId::new("pitch", pitch), &params, |b, params| {
            b.to_async(&rt).iter(|| async {
                let result = engine.synthesize(black_box(text), black_box(params)).await;
                black_box(result.unwrap())
            });
        });
    }
    
    group.finish();
}

criterion_group!(
    benches,
    bench_tts_synthesis,
    bench_tts_streaming,
    bench_voice_operations,
    bench_different_voices,
    bench_synthesis_params
);
criterion_main!(benches);