import os
import shutil
import time
import whisper
import psutil
import pandas as pd
from jiwer import wer

if shutil.which("ffmpeg") is None:
    raise SystemExit(
        "ffmpeg was not found on PATH. Install ffmpeg and ensure the ffmpeg executable is available in PATH before running this script."
    )

print("Loading Whisper Small model...")

process = psutil.Process()

memory_before = process.memory_info().rss / (1024 * 1024)

model = whisper.load_model("small")

memory_after = process.memory_info().rss / (1024 * 1024)

memory_used = memory_after - memory_before

print("\n====================")
print(f"Memory Used By Model : {memory_used:.2f} MB")
print("Setup Complexity     : Medium")
print("====================\n")

gt = pd.read_csv("ground_truth.csv")

results = []

for _, row in gt.iterrows():

    filepath = os.path.join(
        "data",
        str(row["filename"]).strip()
    )

    start = time.time()

    output = model.transcribe(
        filepath,
        language="en"
    )

    end = time.time()

    prediction = output["text"].strip().lower()

    truth = str(
        row["transcript"]
    ).strip().lower()

    error = wer(
        truth,
        prediction
    )

    print("\n====================")
    print("FILE :", row["filename"])
    print("TRUTH:", truth)
    print("PRED :", prediction)
    print("WER  :", error)

    results.append({
        "file": row["filename"],
        "wer": error,
        "time": end - start
    })

df = pd.DataFrame(results)

df.to_csv(
    "results/whisper_results.csv",
    index=False
)

print("\n====================")
print(df)

print("\nAverage WER :", df["wer"].mean())
print("Average Time:", df["time"].mean())
print("Memory Used :", round(memory_used, 2), "MB")
print("Setup Complexity : Medium")