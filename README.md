# asr-model-benchmark
Benchmarking Whisper, Faster-Whisper and Wav2Vec2 for customer support ASR systems
# ASR Model Benchmarking for Customer Support Systems

## Project Overview

This project benchmarks three Automatic Speech Recognition (ASR) models for a voice-based AI assistant designed for customer support calls operating in noisy environments and supporting multiple accents.

The goal is to compare model performance based on:

- Word Error Rate (WER)
- Inference Time
- Memory Usage
- Setup Complexity
- Deployment Feasibility

---

## Models Evaluated

### 1. Whisper Small
- Developed by OpenAI
- Transformer Encoder-Decoder architecture
- Strong multilingual support
- Good robustness to accents and noise

### 2. Faster-Whisper Small
- Optimized implementation of Whisper
- Uses CTranslate2 inference engine
- Lower memory consumption
- CPU-friendly deployment

### 3. Wav2Vec2 Base-960h
- Developed by Meta AI
- Self-supervised learning approach
- High transcription accuracy
- Fast inference speed

---

## Dataset

A subset of the LibriSpeech dataset was used.

Dataset Characteristics:

- English speech recordings
- Ground-truth transcripts available
- 10 audio samples used for benchmarking

Directory Structure:

```
data/
├── 61-70968-0000.flac
├── 61-70968-0001.flac
├── ...
└── 61-70968-0009.flac
```

Ground Truth:

```
ground_truth.csv
```

Format:

```csv
filename,transcript
61-70968-0000.flac,HE BEGAN A CONFUSED COMPLAINT AGAINST THE WIZARD...
```

---

## Project Structure

```
asr-benchmark/
│
├── benchmark_whisper.py
├── benchmark_faster_whisper.py
├── benchmark_wav2vec2.py
├── compare_results.py
│
├── ground_truth.csv
│
├── data/
│   └── audio files
│
├── results/
│   ├── whisper_results.csv
│   ├── faster_whisper_results.csv
│   ├── wav2vec2_results.csv
│   └── final_comparison.csv
│
├── requirements.txt
├── README.md
└── report.pdf
```

---

## Installation

### Create Virtual Environment

Windows:

```bash
python -m venv .venv
```

Activate:

```bash
.venv\Scripts\activate
```

---

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

### Install FFmpeg

Whisper requires FFmpeg.

Verify installation:

```bash
ffmpeg -version
```

---

## Running Benchmarks

### Whisper

```bash
python benchmark_whisper.py
```

Output:

```
results/whisper_results.csv
```

---

### Faster-Whisper

```bash
python benchmark_faster_whisper.py
```

Output:

```
results/faster_whisper_results.csv
```

---

### Wav2Vec2

```bash
python benchmark_wav2vec2.py
```

Output:

```
results/wav2vec2_results.csv
```

---

### Compare All Models

```bash
python compare_results.py
```

Output:

```
results/final_comparison.csv
```

---

## Evaluation Metrics

### Word Error Rate (WER)

WER measures transcription quality.

Formula:

```
WER = (Substitutions + Insertions + Deletions) / Total Words
```

Lower WER indicates better performance.

---

### Inference Time

Average time required to transcribe an audio file.

Lower inference time indicates faster processing.

---

### Memory Usage

Approximate RAM consumed during model loading and inference.

---

### Setup Complexity

Measures installation effort and deployment requirements.

---

## Benchmark Results

| Model | Average WER | Average Time (s) |
|---------|------------|------------------|
| Whisper Small | 0.1688 | 3.435 |
| Faster-Whisper Small | 0.1555 | 4.084 |
| Wav2Vec2 Base-960h | 0.0425 | 0.545 |

### WER Percentage

| Model | WER (%) |
|---------|---------|
| Whisper Small | 16.88% |
| Faster-Whisper Small | 15.55% |
| Wav2Vec2 Base-960h | 4.25% |

---

## Analysis

### Accuracy

Wav2Vec2 achieved the lowest Word Error Rate:

- Wav2Vec2: 4.25%
- Faster-Whisper: 15.55%
- Whisper: 16.88%

### Speed

Wav2Vec2 achieved the fastest inference speed:

- Wav2Vec2: 0.545 s
- Whisper: 3.435 s
- Faster-Whisper: 4.084 s

### Deployment

- Whisper offers strong multilingual support.
- Faster-Whisper provides optimized deployment.
- Wav2Vec2 offers the best benchmark performance for English speech recognition.

---

## Recommendation

### Best Overall Benchmark Performance

**Wav2Vec2 Base-960h**

Reasons:

- Lowest WER
- Fastest inference
- Easy deployment

### Best for Multilingual Customer Support

**Faster-Whisper Small**

Reasons:

- Better robustness to accents
- Lower resource consumption
- Production-friendly deployment

---

## Technologies Used

- Python 3.11
- PyTorch
- OpenAI Whisper
- Faster-Whisper
- Hugging Face Transformers
- Librosa
- Pandas
- JiWER
- FFmpeg

---

## Author

ASR Benchmarking Assignment

Prepared for evaluation of speech recognition models in customer support environments.
