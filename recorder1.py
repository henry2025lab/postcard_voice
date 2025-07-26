import queue
import threading
import time
from typing import Callable, Optional

import numpy as np
import sounddevice as sd


def noise_gate(data: np.ndarray, level: float, strength: float = 1.5):
    current = np.percentile(np.abs(data), 20)
    level = 0.7 * level + 0.3 * current if level > 0 else current
    threshold = level * strength
    cleaned = np.where(np.abs(data) < threshold, 0.0, data)
    return cleaned, level


def normalize(data: np.ndarray, target: float = 0.1):
    if len(data) == 0:
        return data
    rms = np.sqrt(np.mean(data ** 2))
    if rms > 0:
        data = data * (target / rms)
    return data


class AudioRecorder(threading.Thread):
    def __init__(
        self,
        callback: Callable[[np.ndarray, int, float], None],
        chunk_seconds: float = 3.0,
        sub_seconds: float = 0.5,
        samplerate: int = 16000,
    ) -> None:
        super().__init__(daemon=True)
        self.callback = callback
        self.chunk_seconds = chunk_seconds
        self.sub_seconds = sub_seconds
        self.samplerate = samplerate
        self._recording = threading.Event()
        self._queue: "queue.Queue[np.ndarray]" = queue.Queue()
        self.noise_level = 0.0

    def start_recording(self):
        if not self.is_alive():
            self._recording.set()
            super().start()

    def stop_recording(self):
        self._recording.clear()
        if self.is_alive():
            self.join()

    def run(self):
        base_ts = time.time()
        with sd.InputStream(channels=1, samplerate=self.samplerate, callback=self._audio_cb):
            while self._recording.is_set():
                sd.sleep(int(self.chunk_seconds * 1000))
                frames = self._collect_frames()
                if frames is not None:
                    self._process_chunk(frames, base_ts)
                base_ts = time.time()
            frames = self._collect_frames()
            if frames is not None:
                self._process_chunk(frames, base_ts)

    def _audio_cb(self, indata, frames, time_info, status):
        self._queue.put(indata.copy().reshape(-1))

    def _collect_frames(self) -> Optional[np.ndarray]:
        chunks = []
        while not self._queue.empty():
            chunks.append(self._queue.get())
        if chunks:
            return np.concatenate(chunks)
        return None

    def _process_chunk(self, data: np.ndarray, ts_base: float):
        data = data.astype(np.float32)
        data, self.noise_level = noise_gate(data, self.noise_level)
        data = normalize(data)

        step = int(self.sub_seconds * self.samplerate)
        for i in range(0, len(data), step):
            seg = data[i:i + step]
            if len(seg) == 0:
                continue
            if np.sqrt(np.mean(seg ** 2)) < 1e-5:
                continue
            ts = ts_base + i / self.samplerate
            if self.callback:
                self.callback(seg, self.samplerate, ts)