use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use tokio::runtime::Runtime;
use vocalize_core::{AudioWriter, AudioFormat, EncodingSettings};
use tempfile::NamedTempFile;

fn bench_audio_writing(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let writer = AudioWriter::new();
    
    let mut group = c.benchmark_group("audio_writing");
    
    // Test different audio lengths
    let audio_lengths = vec![
        ("1sec", 24_000),      // 1 second at 24kHz
        ("5sec", 120_000),     // 5 seconds
        ("10sec", 240_000),    // 10 seconds
    ];
    
    let settings = EncodingSettings::default();
    
    for (name, length) in audio_lengths {
        let audio_data: Vec<f32> = (0..length)
            .map(|i| (i as f32 * 0.001).sin() * 0.5)
            .collect();
        
        // Benchmark WAV writing (the only implemented format)
        group.bench_with_input(BenchmarkId::new("wav_write", name), &audio_data, |b, audio| {
            b.to_async(&rt).iter(|| async {
                let temp_file = NamedTempFile::with_suffix(".wav").unwrap();
                let result = writer.write_file(
                    black_box(audio),
                    black_box(temp_file.path()),
                    black_box(AudioFormat::Wav),
                    black_box(Some(settings.clone()))
                ).await;
                black_box(result.unwrap())
            });
        });
    }
    
    group.finish();
}

fn bench_audio_format_detection(c: &mut Criterion) {
    let mut group = c.benchmark_group("format_detection");
    
    let extensions = vec!["wav", "mp3", "flac", "ogg"];
    let paths = vec![
        "audio.wav",
        "/path/to/music.mp3",
        "recording.flac",
        "podcast.ogg",
        "test/directory/voice.WAV",
    ];
    
    // Benchmark extension detection
    group.bench_function("from_extension", |b| {
        b.iter(|| {
            for ext in &extensions {
                let result = AudioFormat::from_extension(black_box(ext));
                black_box(result.unwrap());
            }
        });
    });
    
    // Benchmark path detection
    group.bench_function("from_path", |b| {
        b.iter(|| {
            for path in &paths {
                let result = AudioFormat::from_path(black_box(path));
                black_box(result.unwrap());
            }
        });
    });
    
    group.finish();
}

fn bench_encoding_settings(c: &mut Criterion) {
    let mut group = c.benchmark_group("encoding_settings");
    
    // Benchmark settings creation
    group.bench_function("settings_creation", |b| {
        b.iter(|| {
            let settings = EncodingSettings::new(black_box(48000), black_box(2))
                .with_bit_depth(black_box(24))
                .with_quality(black_box(0.8))
                .with_variable_bitrate();
            black_box(settings)
        });
    });
    
    // Benchmark settings validation
    let settings = EncodingSettings::default();
    group.bench_function("settings_validation", |b| {
        b.iter(|| {
            let result = black_box(&settings).validate();
            black_box(result)
        });
    });
    
    group.finish();
}

fn bench_file_size_estimation(c: &mut Criterion) {
    let writer = AudioWriter::new();
    let mut group = c.benchmark_group("file_size_estimation");
    
    // Test different audio lengths
    let audio_lengths = vec![
        ("short", 1000),
        ("medium", 50_000),
        ("long", 240_000),
    ];
    
    let settings = EncodingSettings::default();
    let formats = AudioFormat::all();
    
    for (length_name, length) in audio_lengths {
        let audio_data: Vec<f32> = (0..length)
            .map(|i| (i as f32 * 0.001).sin() * 0.5)
            .collect();
        
        for &format in formats {
            group.bench_with_input(
                BenchmarkId::new(format!("{}_{}", format.extension(), length_name), ""),
                &audio_data,
                |b, audio| {
                    b.iter(|| {
                        let size = writer.estimate_file_size(
                            black_box(audio),
                            black_box(format),
                            black_box(&settings)
                        );
                        black_box(size)
                    });
                }
            );
        }
    }
    
    group.finish();
}

fn bench_audio_validation(c: &mut Criterion) {
    let writer = AudioWriter::new();
    let mut group = c.benchmark_group("audio_validation");
    
    // Test different audio data sizes
    let sizes = vec![
        ("small", 1000),
        ("medium", 24_000),
        ("large", 240_000),
    ];
    
    let settings = EncodingSettings::default();
    
    for (name, size) in sizes {
        let audio_data: Vec<f32> = (0..size)
            .map(|i| (i as f32 * 0.001).sin() * 0.5)
            .collect();
        
        group.bench_with_input(BenchmarkId::new("validate_inputs", name), &audio_data, |b, audio| {
            b.iter(|| {
                let result = writer.validate_inputs(black_box(audio), black_box(&settings));
                black_box(result)
            });
        });
    }
    
    group.finish();
}

fn bench_different_bit_depths(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let writer = AudioWriter::new();
    
    let mut group = c.benchmark_group("bit_depths");
    
    let audio_data: Vec<f32> = (0..24_000)
        .map(|i| (i as f32 * 0.001).sin() * 0.5)
        .collect();
    
    let bit_depths = vec![8, 16, 24, 32];
    
    for bit_depth in bit_depths {
        let settings = EncodingSettings::new(24000, 1).with_bit_depth(bit_depth);
        
        group.bench_with_input(BenchmarkId::new("wav_write", bit_depth), &audio_data, |b, audio| {
            b.to_async(&rt).iter(|| async {
                let temp_file = NamedTempFile::with_suffix(".wav").unwrap();
                let result = writer.write_file(
                    black_box(audio),
                    black_box(temp_file.path()),
                    black_box(AudioFormat::Wav),
                    black_box(Some(settings.clone()))
                ).await;
                black_box(result.unwrap())
            });
        });
    }
    
    group.finish();
}

criterion_group!(
    benches,
    bench_audio_writing,
    bench_audio_format_detection,
    bench_encoding_settings,
    bench_file_size_estimation,
    bench_audio_validation,
    bench_different_bit_depths
);
criterion_main!(benches);