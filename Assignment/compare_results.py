import pandas as pd

w1 = pd.read_csv(
    "results/whisper_results.csv"
)

w2 = pd.read_csv(
    "results/faster_whisper_results.csv"
)

w3 = pd.read_csv(
    "results/wav2vec2_results.csv"
)

summary = pd.DataFrame({

    "Model":[
        "Whisper",
        "Faster Whisper",
        "Wav2Vec2"
    ],

    "Average WER":[
        w1["wer"].mean(),
        w2["wer"].mean(),
        w3["wer"].mean()
    ],

    "Average Time":[
        w1["time"].mean(),
        w2["time"].mean(),
        w3["time"].mean()
    ]
})

print("\nFinal Benchmark Comparison")
print(summary)

summary.to_csv(
    "results/final_comparison.csv",
    index=False
)

print("\nSaved: results/final_comparison.csv")