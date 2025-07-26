import os
import tempfile
import numpy as np
import torch
import soundfile as sf
from funasr import AutoModel


class EmotionClassifier:
    def __init__(self, model_name="iic/emotion2vec_plus_large", threshold=0.1):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModel(model=model_name, device=device, disable_update=True)
        self.threshold = threshold

        tok = os.path.join(self.model.model_path, "tokens.txt")
        if os.path.exists(tok):
            with open(tok, "r", encoding="utf-8") as f:
                self.labels = [l.strip().split("/")[-1] for l in f]
        else:
            self.labels = []

        temp_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), "temperature.npy")
        if os.path.isfile(temp_file):
            try:
                self.temperature = float(np.load(temp_file)[0])
            except Exception:
                self.temperature = 1.0
        else:
            self.temperature = 1.0

    def _run(self, wav_path):
        return self.model.generate(wav_path, granularity="utterance", extract_embedding=False)

    def predict(self, samples, sr):
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        sf.write(path, samples, sr)
        try:
            res = self._run(path)
        finally:
            os.unlink(path)

        label = None
        score = None
        if isinstance(res, list) and res:
            item = res[0]
            label = item.get("text") or item.get("label")
            score = item.get("score")
            if score is None:
                scores = item.get("scores") or item.get("prob")
                if scores:
                    s = scores[0] if isinstance(scores[0], (list, tuple)) else scores
                    s = [float(x) for x in s]
                    logits = torch.log(torch.tensor(s) + 1e-8)
                    prob = torch.softmax(logits / self.temperature, dim=-1).numpy()
                    idx = int(np.argmax(prob))
                    score = float(prob[idx])
                    if label is None and self.labels:
                        if 0 <= idx < len(self.labels):
                            label = self.labels[idx]
        return label, score