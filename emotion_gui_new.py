import os
import threading
import queue
import tempfile
import argparse
import tkinter as tk
from tkinter.scrolledtext import ScrolledText

# External dependencies. They may need to be installed in the runtime
import sounddevice as sd
import soundfile as sf
import numpy as np
import torch
from funasr import AutoModel


def noise_reduce(samples, prev_level, strength=1.5):
    """Simple noise gate based on previous noise level estimate."""
    cur_level = np.percentile(np.abs(samples), 20)
    level = 0.7 * prev_level + 0.3 * cur_level if prev_level > 0 else cur_level
    threshold = level * strength
    cleaned = np.where(np.abs(samples) < threshold, 0.0, samples)
    return cleaned, level


def normalize_volume(samples, target=0.1):
    rms = np.sqrt(np.mean(samples ** 2))
    if rms > 0:
        samples = samples * (target / rms)
    return samples


def vad_filter(samples, sr, noise_level, frame_ms=30):
    frame = int(sr * frame_ms / 1000)
    voiced = []
    threshold = max(noise_level * 1.5, 1e-5)
    for i in range(0, len(samples), frame):
        chunk = samples[i:i + frame]
        if len(chunk) < frame:
            break
        if np.sqrt(np.mean(chunk ** 2)) >= threshold:
            voiced.append(chunk)
    if voiced:
        return np.concatenate(voiced)
    return np.array([], dtype=samples.dtype)


class EmotionRecognizer:
    def __init__(self, model_dir, threshold=0.1):
        """Load the emotion recognition model.

        Parameters
        ----------
        model_dir : str
            Directory containing model files.
        threshold : float, optional
            Probability threshold for displaying a label.
        """

        self.threshold = threshold
        temp_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "temperature.npy")
        if os.path.isfile(temp_path):
            self.temperature = float(np.load(temp_path)[0])
        else:
            self.temperature = 1.0
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModel(
            model=model_dir,
            device=device,
            disable_update=True,
        )

        # Load label vocabulary for fallback when the model output does not
        # include a label string but only probabilities.
        token_path = os.path.join(self.model.model_path, "tokens.txt")
        if os.path.exists(token_path):
            with open(token_path, "r", encoding="utf-8") as f:
                self.label_vocab = [line.split("/")[-1].strip() for line in f]
        else:
            self.label_vocab = []

    def predict(self, wav_path):
        res = self.model.generate(
            wav_path,
            granularity="utterance",
            extract_embedding=False,
        )

        label = None
        score = None

        if isinstance(res, list) and res:
            first = res[0]
            label = first.get("text") or first.get("label")

            if isinstance(first.get("score"), (int, float)):
                score = float(first["score"])
            else:
                scores = first.get("scores") or first.get("prob")
                if isinstance(scores, (list, tuple)) and scores:
                    if scores and isinstance(scores[0], (list, tuple)):
                        scores = scores[0]
                    scores = [float(s) for s in scores]
                    logits = torch.log(torch.tensor(scores) + 1e-8)
                    scaled = torch.softmax(logits / self.temperature, dim=-1).numpy()
                    scores = scaled.tolist()
                    score = max(scores)

                    # Fallback: derive label from the score index if label text
                    # is not provided by the model.
                    if label is None and self.label_vocab:
                        idx = scores.index(score)
                        if 0 <= idx < len(self.label_vocab):
                            label = self.label_vocab[idx]

        return label, score


class Recorder(threading.Thread):
    def __init__(self, callback, samplerate=16000, chunk_seconds=3):
        super().__init__(daemon=True)
        self.callback = callback
        self.samplerate = samplerate
        self.chunk_seconds = chunk_seconds
        self._recording = threading.Event()
        self._queue = queue.Queue()

    def run(self):
        with sd.InputStream(channels=1, samplerate=self.samplerate, callback=self._audio_callback):
            self._recording.set()
            while self._recording.is_set():
                sd.sleep(int(self.chunk_seconds * 1000))
                self._emit_chunk()
            self._emit_chunk()

    def _audio_callback(self, indata, frames, time, status):
        self._queue.put(indata.copy())

    def _emit_chunk(self):
        frames = []
        while not self._queue.empty():
            frames.append(self._queue.get())
        if frames:
            data = np.concatenate(frames, axis=0)
            self.callback(data, self.samplerate)

    def stop(self):
        self._recording.clear()


class EmotionApp:
    def __init__(self, root, model_dir, threshold=0.1):
        self.recognizer = EmotionRecognizer(model_dir, threshold=threshold)
        self.recorder = None
        self.noise_level = 0.0

        root.title("Emotion Recorder")
        self.root = root

        self.start_btn = tk.Button(root, text="Start Recording", command=self.start)
        self.start_btn.pack(pady=5)
        self.stop_btn = tk.Button(root, text="Stop Recording", state=tk.DISABLED, command=self.stop)
        self.stop_btn.pack(pady=5)
        self.output = ScrolledText(root, height=10, width=50)
        self.output.pack(padx=5, pady=5)
        self.output.insert(tk.END, "Ready\n")

    def start(self):
        if self.recorder:
            return
        self.noise_level = 0.0
        self.recorder = Recorder(self.process_chunk)
        self.recorder.start()
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)

    def stop(self):
        if self.recorder:
            self.recorder.stop()
            self.recorder.join()
            self.recorder = None
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)

    def process_chunk(self, data, samplerate):
        # Basic noise suppression and silence trimming
        data = data.astype(np.float32)
        data, self.noise_level = noise_reduce(data, self.noise_level)
        data = normalize_volume(data)
        data = vad_filter(data, samplerate, self.noise_level)

        rms = np.sqrt(np.mean(data ** 2)) if len(data) else 0.0
        print(f"Audio RMS (volume): {rms:.6f}")
        if rms < 1e-5:
            print("Chunk is silent after VAD, skipping...")
            return

        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        label = None
        score = None
        try:
            sf.write(path, data, samplerate)
            label, score = self.recognizer.predict(path)
        except Exception as e:
            print("Prediction error:", e)
        finally:
            os.unlink(path)
        print("chunk result:", label, score)
        if label and score is not None and score >= self.recognizer.threshold:
            self.root.after(0, self._append_label, label, score)

    def _append_label(self, label, score):
        self.output.insert(tk.END, f"{label} ({score:.2f})\n")
        self.output.see(tk.END)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="score threshold for displaying a label",
    )
    parser.add_argument(
        "--model",
        default="iic/emotion2vec_plus_large",
        help="model directory or identifier"
    )
    args = parser.parse_args()

    model_dir = args.model
    root = tk.Tk()
    app = EmotionApp(root, model_dir, threshold=args.threshold)
    root.mainloop()


if __name__ == '__main__':
    main()