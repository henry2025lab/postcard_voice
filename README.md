# Sound Postcard – Real‑Time Speech‑Emotion Recorder

**A lightweight Python toolkit for recording short speech segments, classifying emotions with the Emotion2Vec+ model, and exporting time‑stamped trajectories for downstream matching tasks.**

---

## ✨ Key Features

| Module                     | Purpose                                                                             | File             |
| -------------------------- | ----------------------------------------------------------------------------------- | ---------------- |
| **AudioRecorder**          | Real‑time microphone capture, noise‑gate, RMS normalisation, 0.5 s slicing          | `recorder1.py`   |
| **EmotionClassifier**      | Utterance‑level inference with Emotion2Vec+ Large, optional temperature calibration | `algorithm1.py`  |
| **Temperature Calibrator** | Offline grid‑search of *T* to minimise ECE; saves `temperature.npy`                 | `calibrate_T.py` |
| **GUI (Tk)**               | Start/stop recording, live log, CSV export, one‑click playback of matched audio     | `gui1.py`        |
| **Main Entrypoint**        | Thread orchestration of recorder → classifier → GUI                                 | `main.py`        |

---

## 🔧 Prerequisites

| Package                                                       | Tested Version |
| ------------------------------------------------------------- | -------------- |
| Python                                                        |  3.9 – 3.11    |
| PyTorch                                                       |  ≥ 2.1         |
| **funasr** (with `iic/emotion2vec_plus_large`)                |  1.1.x         |
| NumPy, SoundFile, SoundDevice                                 | latest         |
| Tkinter *(bundled with CPython)*                              | –              |
| Windows ▶ `winsound` *(built‑in)* · macOS/Linux ▶ `playsound` | –              |

> **Tip 🚀** GPU is auto‑detected; the code falls back to CPU when `torch.cuda.is_available()` is **False**.

```bash
# 1 – create & activate venv (optional but recommended)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2 – install dependencies
pip install torch sounddevice soundfile numpy playsound funasr
```

---

## ▶️ Quick Start

```bash
python main.py
```

1. Click **Start Recording** and speak near the mic.
2. Every 0.5 s, the recognised emotion and confidence are streamed to the log.
3. Click **Stop Recording** to finish.
4. Use **Export to CSV** to save `[timestamp, emotion, score]` for later trajectory matching.
5. Optional: place `matched_audio.wav` beside `gui1.py` – the **Play Matched Audio** button will play it asynchronously.

---

## 📏 Calibrating Confidence (Temperature Scaling)

Run the following once you have a validation set of labelled `.wav` files and an aligned `valid_labels.npy`:

```bash
python calibrate_T.py
```

This script grid‑searches *T ∈ \[0.3, 2.0]* to minimise Expected Calibration Error (ECE), then stores the best scalar in `temperature.npy`. `EmotionClassifier` automatically loads it at start‑up.

---

## 🗂️ Project Tree

```text
├── algorithm1.py      # EmotionClassifier (inference + softmax(T))
├── calibrate_T.py     # offline ECE minimisation
├── gui1.py            # Tk GUI & playback helper
├── main.py            # app bootstrap & threading
├── recorder1.py       # microphone stream & preprocessing
└── temperature.npy    # (auto‑generated) optimal T
```

---

## 📄 License

Distributed under the **MIT License** – see `LICENSE` for full text.

> The pre‑trained Emotion2Vec+ model is released by the original authors under their own licence; please respect their terms.

---

## 🌐 Citing This Code

If you use this toolkit in academic work, please cite the corresponding Zenodo release:

```bibtex
@software{li_2025_sound_postcard,
  author       = {Henry Li},
  title        = {Sound Postcard Codebase},
  version      = {1.0.0},
  doi          = {10.5281/zenodo.1234567},
  url          = {https://github.com/henry-li/sound-postcard},
  year         = {2025}
}
```

---

## 🤝 Contributing

Pull requests are welcome! Please:

1. Fork → feature‑branch → PR.
2. Ensure `flake8` passes and add unit tests where feasible.
3. Describe your change succinctly in `CHANGELOG.md`.

---

## 💬 Acknowledgements

This project builds upon the **Emotion2Vec+** model • **funasr** toolkit • **PyTorch** ecosystem. Huge thanks to their respective authors and maintainers.
“This repository relies on the Emotion2Vec+ model released by Z. Ma et al. under Apache‑2.0. All rights reserved by the original authors.”
