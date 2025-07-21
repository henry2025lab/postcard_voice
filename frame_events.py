import os
import argparse
import json
import csv
import numpy as np
import torch
from scipy.ndimage import median_filter
from funasr import AutoModel


def load_temperature(script_dir):
    path = os.path.join(script_dir, "temperature.npy")
    if os.path.isfile(path):
        return float(np.load(path)[0])
    return 1.0


def load_vocab(model_path):
    tok_path = os.path.join(model_path, "tokens.txt")
    if os.path.exists(tok_path):
        with open(tok_path, "r", encoding="utf-8") as f:
            return [l.split("/")[-1].strip() for l in f]
    return [str(i) for i in range(9)]


def infer_frames(wav, model_dir, hop_length=640):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel(model=model_dir, device=device, disable_update=True)
    frames = model.generate(wav, granularity="frame", hop_length=hop_length)
    return frames, model.model_path


def smooth_and_merge(frames, T, labels, min_dur=0.6):
    p = np.stack([f["scores"] for f in frames])
    logits = np.log(p + 1e-8)
    p = torch.softmax(torch.tensor(logits / T), dim=-1).numpy()

    mode = median_filter(np.argmax(p, axis=1), size=7)

    events = []
    cur = int(mode[0])
    t0 = frames[0]["start"]
    scores = [p[0]]
    for i, lab in enumerate(mode[1:], 1):
        lab = int(lab)
        if lab != cur:
            t1 = frames[i]["start"]
            if t1 - t0 >= min_dur:
                mean_score = np.mean([s[cur] for s in scores])
                events.append({
                    "start": round(t0, 2),
                    "end": round(t1, 2),
                    "duration": round(t1 - t0, 2),
                    "label": labels[cur],
                    "mean_score": round(float(mean_score), 4)
                })
            cur = lab
            t0 = t1
            scores = []
        scores.append(p[i])
    # tail
    t1 = frames[-1]["end"]
    if t1 - t0 >= min_dur:
        mean_score = np.mean([s[cur] for s in scores])
        events.append({
            "start": round(t0, 2),
            "end": round(t1, 2),
            "duration": round(t1 - t0, 2),
            "label": labels[cur],
            "mean_score": round(float(mean_score), 4)
        })
    return events


def save_csv(events, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["start", "end", "duration", "label", "mean_score"])
        writer.writeheader()
        for e in events:
            writer.writerow(e)


def main():
    parser = argparse.ArgumentParser(description="Frame-level emotion events")
    parser.add_argument("--wav", required=True, help="16kHz wav file")
    parser.add_argument("--model", default="iic/emotion2vec_plus_large")
    parser.add_argument("--out", default="events.csv")
    parser.add_argument("--hop", type=int, default=640, help="hop length samples")
    parser.add_argument("--min_dur", type=float, default=0.6, help="min event duration")
    args = parser.parse_args()

    frames, model_path = infer_frames(args.wav, args.model, hop_length=args.hop)
    script_dir = os.path.abspath(os.path.dirname(__file__))
    T = load_temperature(script_dir)
    labels = load_vocab(model_path)

    events = smooth_and_merge(frames, T, labels, min_dur=args.min_dur)
    save_csv(events, args.out)
    print(json.dumps(events, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()