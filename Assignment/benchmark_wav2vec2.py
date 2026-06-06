import librosa
import torch
import pandas as pd
import os
import time

from jiwer import wer

from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2ForCTC
)

print("Loading Wav2Vec2 model...")

processor = Wav2Vec2Processor.from_pretrained(
    "facebook/wav2vec2-base-960h"
)

model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-base-960h"
)

print("Model loaded.")

gt = pd.read_csv(
    "ground_truth.csv"
)

print("Total rows:", len(gt))

results = []

for _, row in gt.iterrows():

    print("\nProcessing:", row["filename"])

    filepath = os.path.join(
        "data",
        str(row["filename"]).strip()
    )

    audio, sr = librosa.load(
        filepath,
        sr=16000
    )

    start = time.time()

    inputs = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt"
    )

    with torch.no_grad():

        logits = model(
            inputs.input_values
        ).logits

    predicted_ids = torch.argmax(
        logits,
        dim=-1
    )

    prediction = processor.batch_decode(
        predicted_ids
    )[0].strip().lower()

    end = time.time()

    truth = str(
        row["transcript"]
    ).strip().lower()

    error = wer(
        truth,
        prediction
    )

    print("====================")
    print("FILE :", row["filename"])
    print("TRUTH:", truth)
    print("PRED :", prediction)
    print("WER  :", error)

    results.append({
        "file": row["filename"],
        "wer": error,
        "time": end - start
    })

print("\nResults count =", len(results))

df = pd.DataFrame(results)

df.to_csv(
    "results/wav2vec2_results.csv",
    index=False
)

print("\n====================")
print(df)

print("\nAverage WER :", df["wer"].mean())
print("Average Time:", df["time"].mean())