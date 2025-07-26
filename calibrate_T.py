import numpy as np
import torch
import glob
from funasr import AutoModel
from sklearn.metrics import accuracy_score, log_loss

MODEL = "iic/emotion2vec_plus_large"
VALDIR = "valid"
labels = np.load("valid_labels.npy")

wav_list = sorted(glob.glob(f"{VALDIR}/*.wav"))
assert len(wav_list) == len(labels)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModel(model=MODEL, device=device, disable_update=True)

probs = []
for r in model.generate(wav_list, granularity="utterance"):
    probs.append(r["scores"])
probs = np.asarray(probs)

logits = np.log(np.clip(probs, 1e-8, 1.0))

def ece(p):
    bins = np.linspace(0, 1, 11)
    confid = p.max(1)
    preds = p.argmax(1)
    acc = preds == labels
    ece = 0
    for i in range(10):
        msk = (confid >= bins[i]) & (confid < bins[i+1])
        if msk.any():
            ece += abs(acc[msk].mean() - confid[msk].mean()) * msk.mean()
    return ece

best_T, best_ece = 1.0, 1.0
for T in np.linspace(0.3, 2.0, 34):
    scaled = torch.softmax(torch.tensor(logits / T), dim=-1).numpy()
    e = ece(scaled)
    if e < best_ece:
        best_T, best_ece = T, e

print(f"Best T={best_T:.2f}  ECE={best_ece:.4f}")
np.save("temperature.npy", np.array([best_T]))