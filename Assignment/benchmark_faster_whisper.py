from faster_whisper import WhisperModel
import pandas as pd
import os
import time
from jiwer import wer

model = WhisperModel(
    "small",
    device="cpu",
    compute_type="int8"
)

gt = pd.read_csv("ground_truth.csv")

results = []

for _, row in gt.iterrows():

    print("Processing:", row["filename"])

    filepath = os.path.join(
        "data",
        str(row["filename"]).strip()
    )

    start = time.time()

    segments, _ = model.transcribe(
        filepath,
        language="en"
    )

    prediction = " ".join(
        seg.text for seg in segments
    ).strip().lower()

    end = time.time()

    truth = str(row["transcript"]).strip().lower()

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

print("Results count =", len(results))

df = pd.DataFrame(results)

df.to_csv(
    "results/faster_whisper_results.csv",
    index=False
)

print(df)

print("\nAverage WER :", df["wer"].mean())
print("Average Time:", df["time"].mean())