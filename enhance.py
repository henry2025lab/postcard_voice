import subprocess
import os
import sys
import wave
import webrtcvad
import numpy as np
import soundfile as sf

RAW_DIR = "raw48k"
CLEAN_DIR = "clean16k"
os.makedirs(CLEAN_DIR, exist_ok=True)

def denoise_and_loudnorm(src, tmp):
    """Apply RNNoise denoising and loudness normalization via FFmpeg."""
    cmd1 = [
        "ffmpeg", "-y", "-i", src,
        "-af", "arnndn,meter=snr|pts", "-ar", "16000", "-ac", "1",
        tmp
    ]
    cmd2 = [
        "ffmpeg", "-y", "-i", tmp,
        "-af",
        "loudnorm=I=-23:LRA=7:TP=-2:print_format=summary",
        tmp
    ]
    subprocess.run(cmd1, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    subprocess.run(cmd2, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def vad_trim(tmp, dst, aggressiveness=2):
    """Remove long silences with WebRTC VAD."""
    audio, sr = sf.read(tmp)
    pcm16 = (audio * 32768).astype(np.int16).tobytes()
    vad = webrtcvad.Vad(aggressiveness)
    frame = 20
    n = int(sr * frame / 1000)
    voiced = bytearray()
    for i in range(0, len(pcm16), n * 2):
        chunk = pcm16[i:i + n * 2]
        if len(chunk) < n * 2:
            break
        if vad.is_speech(chunk, sr):
            voiced.extend(chunk)
    samples = np.frombuffer(voiced, dtype=np.int16).astype(np.float32) / 32768
    sf.write(dst, samples, sr)


tmp_wav = "_tmp.wav"

for fn in os.listdir(RAW_DIR):
    if not fn.lower().endswith(".wav"):
        continue
    src = os.path.join(RAW_DIR, fn)
    mid = tmp_wav
    denoise_and_loudnorm(src, mid)
    dst = os.path.join(CLEAN_DIR, fn.replace(".wav", "_clean.wav"))
    vad_trim(mid, dst)
    print("\u2713", fn, "\u2192", dst)

os.remove(tmp_wav)